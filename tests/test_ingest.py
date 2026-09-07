"""Behaviour of the one module that talks to the network.

Two things are pinned here.  The **retry policy**: backing off costs
wall-clock, so the distinction that matters is whether a second identical
request could plausibly succeed -- a sprint that was never on the calendar
cannot, a dropped connection can.  And the **degraded-results guard**: a
session can load without raising and still carry nothing worth keeping.

Nothing here touches the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

import sys
import types

import numpy as np

from src.data import ingest
from src.data.ingest import (
    DegradedResultsError,
    IngestReport,
    extract_results,
    is_permanent_failure,
    load_session,
    optional_session_attr,
)


# --------------------------------------------------------------------------- #
# The predicate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message",
    [
        # The message FastF1 raises for a sprint before 2021.
        "Session type 'S' does not exist for this event",
        "Session number 6 does not exist for this event",
        "Invalid session type 'X'",
        "Invalid round: 99",
        "Test event number 3 does not exist",
        "Cannot get testing event by round number!",
    ],
)
def test_calendar_value_errors_are_permanent(message: str) -> None:
    assert is_permanent_failure(ValueError(message))


@pytest.mark.parametrize(
    "exc",
    [
        # Raised when the schedule request itself fails: a network problem,
        # not a statement about the calendar.
        ValueError("Failed to load any schedule data."),
        ConnectionError("connection reset by peer"),
        TimeoutError("read timed out"),
        OSError("temporary failure in name resolution"),
        RuntimeError("something unexpected"),
    ],
)
def test_transient_failures_stay_retryable(exc: BaseException) -> None:
    assert not is_permanent_failure(exc)


@pytest.mark.parametrize(
    "reason",
    ["Too Many Requests", "Service Unavailable", "Bad Gateway",
     "Gateway Timeout", "Internal Server Error"],
)
def test_http_errors_from_ergast_stay_retryable(reason: str) -> None:
    """FastF1 raises ErgastInvalidRequestError for *every* non-200.

    A 429 and a malformed request arrive as the same class, so the message is
    the only thing separating "back off and try again" from "never going to
    work".  Getting this wrong means one rate-limited burst silently drops
    races from the dataset.
    """
    pytest.importorskip("fastf1")
    from fastf1.exceptions import ErgastInvalidRequestError

    exc = ErgastInvalidRequestError(
        "Invalid request to Ergast (https://api.jolpi.ca/ergast/f1/2021/17/results.json)\n"
        f"Server response: '{reason}'"
    )
    assert not is_permanent_failure(exc)


def test_genuinely_invalid_ergast_request_is_permanent() -> None:
    pytest.importorskip("fastf1")
    from fastf1.exceptions import ErgastInvalidRequestError

    exc = ErgastInvalidRequestError(
        "Invalid request to Ergast (https://api.jolpi.ca/ergast/f1/1949/1/results.json)\n"
        "Server response: 'Not Found'"
    )
    assert is_permanent_failure(exc)


def test_fastf1_exception_classes_are_permanent() -> None:
    pytest.importorskip("fastf1")
    from fastf1._api import SessionNotAvailableError
    from fastf1.exceptions import (
        ErgastInvalidRequestError,
        InvalidSessionError,
        NoLapDataError,
        RateLimitExceededError,
    )

    for exc_type in (
        InvalidSessionError,
        NoLapDataError,
        ErgastInvalidRequestError,
        RateLimitExceededError,
        SessionNotAvailableError,
    ):
        assert is_permanent_failure(exc_type()), exc_type.__name__
    # ...but the message can still overrule the type.
    assert not is_permanent_failure(
        ErgastInvalidRequestError("Server response: 'Too Many Requests'")
    )


def test_fastf1_own_limiter_is_permanent() -> None:
    """FastF1's client-side limiter refuses the call before it is sent.

    Its window is an hour, so a 2-4 second backoff cannot clear it; failing
    fast beats sleeping three times and failing anyway.  The message must not
    trip the transient markers.
    """
    pytest.importorskip("fastf1")
    from fastf1.exceptions import RateLimitExceededError

    for info in ("ergast.com: 200 calls/h", "any API: 500 calls/h"):
        assert is_permanent_failure(RateLimitExceededError(info)), info


def test_data_not_loaded_error_is_not_permanent() -> None:
    """A load-order bug in our own code, not a verdict on the session."""
    pytest.importorskip("fastf1")
    from fastf1.exceptions import DataNotLoadedError

    assert not is_permanent_failure(DataNotLoadedError())


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


@pytest.fixture
def spy(monkeypatch):
    """Replace ``fastf1.get_session`` and ``time.sleep`` with recorders."""
    pytest.importorskip("fastf1")
    calls: dict[str, list] = {"attempts": [], "sleeps": []}

    def install(exc: BaseException) -> dict[str, list]:
        def fake_get_session(year, event, session_type):
            calls["attempts"].append((year, event, session_type))
            raise exc

        monkeypatch.setattr("fastf1.get_session", fake_get_session)
        monkeypatch.setattr("src.data.ingest.time.sleep", calls["sleeps"].append)
        return calls

    return install


def test_permanent_failure_is_not_slept_over(spy) -> None:
    calls = spy(ValueError("Session type 'S' does not exist for this event"))

    with pytest.raises(RuntimeError, match="could not load 2018 1 S"):
        load_session(2018, 1, "S", retries=3, backoff_s=2.0)

    assert len(calls["attempts"]) == 1, "a permanent failure must not be retried"
    assert calls["sleeps"] == [], "a permanent failure must not back off"


def test_transient_failure_still_retries_with_backoff(spy) -> None:
    calls = spy(ConnectionError("connection reset by peer"))

    with pytest.raises(RuntimeError, match="could not load 2024 1 R"):
        load_session(2024, 1, "R", retries=3, backoff_s=2.0)

    assert len(calls["attempts"]) == 3
    assert calls["sleeps"] == [2.0, 4.0]


def test_original_exception_is_chained(spy) -> None:
    """The report shows why, not just that."""
    spy(ValueError("Session type 'S' does not exist for this event"))

    with pytest.raises(RuntimeError) as excinfo:
        load_session(2018, 1, "S")

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "does not exist" in str(excinfo.value.__cause__)


# --------------------------------------------------------------------------- #
# Optional session properties
# --------------------------------------------------------------------------- #


class _Session:
    """Stands in for a FastF1 ``Session`` with a partially loaded state."""

    def __init__(self, exc: BaseException | None = None, value=None):
        self._exc = exc
        self._value = value

    @property
    def total_laps(self):
        if self._exc is not None:
            raise self._exc
        return self._value


def test_unloaded_property_falls_back_to_default() -> None:
    """The results pass loads ``laps=False``, so ``total_laps`` is never filled.

    FastF1 raises DataNotLoadedError rather than AttributeError, which plain
    ``getattr(obj, name, default)`` does not catch -- so reading it blew up
    ``extract_results`` for every race in the build.
    """
    pytest.importorskip("fastf1")
    from fastf1.exceptions import DataNotLoadedError

    session = _Session(exc=DataNotLoadedError("not loaded"))
    assert pd.isna(optional_session_attr(session, "total_laps"))


def test_none_is_treated_as_absent() -> None:
    """FastF1 sets ``_total_laps = None`` for sessions that have no lap count."""
    assert pd.isna(optional_session_attr(_Session(value=None), "total_laps"))


def test_present_value_passes_through() -> None:
    assert optional_session_attr(_Session(value=58), "total_laps") == 58


def test_missing_attribute_falls_back() -> None:
    assert pd.isna(optional_session_attr(object(), "total_laps"))


def test_explicit_default_is_respected() -> None:
    assert optional_session_attr(_Session(value=None), "total_laps", default=0) == 0


# --------------------------------------------------------------------------- #
# The degraded-results guard
# --------------------------------------------------------------------------- #
#
# `Status` comes from the Ergast backend, not F1 timing, and `Session.load`
# catches an Ergast failure internally: it logs "No result data for this
# session available on Ergast!" at WARNING and carries on from timing data
# alone.  So `load_session` returns normally and `is_permanent_failure` never
# sees anything -- the session simply arrives with a full grid of drivers and
# a Status column of empty strings, which the labeller reads as an
# all-retirement race.


def make_session(
    statuses,
    *,
    n_drivers: int | None = None,
    drop_status: bool = False,
    round_number: int = 8,
    event_name: str = "Monaco Grand Prix",
    year: int = 2024,
):
    """A stand-in for a loaded FastF1 session, carrying the columns we read.

    ``statuses`` is the ``Status`` column verbatim, so a test can hand in the
    empty strings FastF1 leaves behind when Ergast declines to answer.
    """
    n = n_drivers if n_drivers is not None else len(statuses)
    results = pd.DataFrame(
        {
            "DriverNumber": [str(i + 1) for i in range(n)],
            "Abbreviation": [f"D{i:02d}" for i in range(n)],
            "DriverId": [f"driver_{i:02d}" for i in range(n)],
            "FullName": [f"Driver {i:02d}" for i in range(n)],
            "TeamName": [f"team_{i // 2:02d}" for i in range(n)],
            "TeamId": [f"team_{i // 2:02d}" for i in range(n)],
            "Position": [float(i + 1) for i in range(n)],
            "ClassifiedPosition": [str(i + 1) for i in range(n)],
            "GridPosition": [float(i + 1) for i in range(n)],
            "Status": list(statuses),
            "Points": [0.0] * n,
            "Laps": [57.0] * n,
            "Time": [pd.NaT] * n,
        }
    )
    if drop_status:
        results = results.drop(columns=["Status"])

    session = types.SimpleNamespace()
    session.results = results
    session.event = pd.Series(
        {
            "EventDate": pd.Timestamp(f"{year}-05-26"),
            "RoundNumber": round_number,
            "EventName": event_name,
            "Location": "Monte-Carlo",
            "Country": "Monaco",
        }
    )
    session.total_laps = 78
    session.session_info = {"Meeting": {"Circuit": {"Key": 101, "ShortName": "Monaco"}}}
    return session


#: What FastF1 actually hands back when the Ergast call is rate-limited: a full
#: grid from timing data, and a Status column of empty strings.  Not nulls --
#: ``isna()`` reports nothing wrong with it.
RATE_LIMITED_STATUS = [""] * 20

GOOD_STATUS = ["Finished"] * 15 + ["Engine", "Collision", "+ 1 Lap", "Accident", "Gearbox"]


def test_extract_results_keeps_a_healthy_frame() -> None:
    frame = extract_results(make_session(GOOD_STATUS))
    assert len(frame) == 20
    assert frame["Status"].eq("Finished").sum() == 15
    assert frame["RoundNumber"].eq(8).all()


def test_rate_limited_results_are_rejected_not_returned() -> None:
    """The signature failure: 20 drivers, no Status, and load() raised nothing."""
    with pytest.raises(DegradedResultsError, match="no finishing status"):
        extract_results(make_session(RATE_LIMITED_STATUS))


@pytest.mark.parametrize(
    "statuses",
    [
        pytest.param([""] * 20, id="empty-strings"),
        pytest.param([np.nan] * 20, id="nulls"),
        pytest.param([None] * 20, id="none"),
        pytest.param(["   ", "\t", ""] * 6 + ["  ", ""], id="whitespace"),
    ],
)
def test_every_flavour_of_blank_status_is_rejected(statuses) -> None:
    with pytest.raises(DegradedResultsError):
        extract_results(make_session(statuses))


def test_missing_status_column_is_rejected_too() -> None:
    """An absent column has the same cause and consequence as an empty one."""
    with pytest.raises(DegradedResultsError):
        extract_results(make_session(GOOD_STATUS, drop_status=True))


def test_one_surviving_status_is_enough_to_keep_the_round() -> None:
    """Partial degradation is not the rate-limit signature; keep the rows.

    A single genuinely-blank driver row happens; the guard is aimed at the
    all-or-nothing failure, and dropping a whole round over one blank would
    cost more data than it saves.
    """
    partial = [""] * 19 + ["Finished"]
    frame = extract_results(make_session(partial))
    assert len(frame) == 20


def test_guard_can_be_switched_off() -> None:
    frame = extract_results(make_session(RATE_LIMITED_STATUS), require_status=False)
    assert len(frame) == 20


def test_guard_ignores_sessions_that_carry_no_status_by_design() -> None:
    """Ergast has no results for sprint qualifying, so a blank Status is normal."""
    frame = extract_results(make_session(RATE_LIMITED_STATUS), session_label="SQ")
    assert len(frame) == 20
    assert frame["session_type"].eq("SQ").all()


def test_an_empty_results_frame_is_left_to_the_existing_paths() -> None:
    """No rows means no mislabelling; that case is already handled downstream."""
    frame = extract_results(make_session([], n_drivers=0))
    assert frame.empty


# --------------------------------------------------------------------------- #
# Degraded rounds reach the report
# --------------------------------------------------------------------------- #

#: The stubbed season: round 1 Bahrain, round 2 Monaco.
EVENT_NAMES = {1: "Bahrain Grand Prix", 2: "Monaco Grand Prix"}


@pytest.fixture
def stub_fastf1(monkeypatch):
    """Install a fake ``fastf1`` module with a two-round schedule."""
    schedule = pd.DataFrame(
        {
            "RoundNumber": list(EVENT_NAMES),
            "EventName": list(EVENT_NAMES.values()),
        }
    )
    module = types.ModuleType("fastf1")
    module.get_event_schedule = lambda year, include_testing=False: schedule
    monkeypatch.setitem(sys.modules, "fastf1", module)
    return module


def test_a_rate_limited_round_is_reported_as_a_failure(monkeypatch, stub_fastf1) -> None:
    """Round 2 is degraded: it must not contribute rows, and must be reported."""

    def fake_load(year, event, session_type, **kwargs):
        if session_type != "R":
            raise RuntimeError("no sprint this weekend")
        return make_session(
            RATE_LIMITED_STATUS if event == 2 else GOOD_STATUS,
            round_number=event,
            event_name=EVENT_NAMES[event],
        )

    monkeypatch.setattr(ingest, "load_session", fake_load)
    monkeypatch.setattr(ingest, "extract_weather", lambda session: {})

    report = IngestReport()
    frame = ingest.collect_season_results(2024, report=report)

    assert set(frame["RoundNumber"]) == {1}, "degraded round leaked into the frame"
    assert len(report.loaded) == 1
    assert len(report.failed) == 1
    label, message = report.failed[0]
    assert "Monaco" in label
    assert "no finishing status" in message
    assert "Monaco" in report.summary()


def test_a_degraded_sprint_is_reported_even_though_a_missing_one_is_not(
    monkeypatch, stub_fastf1
) -> None:
    """An absent sprint is routine; one that answered with no status is not."""

    def fake_load(year, event, session_type, **kwargs):
        statuses = GOOD_STATUS
        if session_type == "S":
            if event == 1:
                raise RuntimeError("no sprint this weekend")
            statuses = RATE_LIMITED_STATUS
        return make_session(
            statuses, round_number=event, event_name=EVENT_NAMES[event]
        )

    monkeypatch.setattr(ingest, "load_session", fake_load)
    monkeypatch.setattr(ingest, "extract_weather", lambda session: {})

    report = IngestReport()
    ingest.collect_season_results(2024, report=report)

    failed = dict(report.failed)
    assert "2024 Monaco Grand Prix [S]" in failed
    assert "2024 Bahrain Grand Prix [S]" not in failed


def test_a_clean_season_reports_no_failures(monkeypatch, stub_fastf1) -> None:
    monkeypatch.setattr(
        ingest, "load_session",
        lambda year, event, session_type, **kw: make_session(
            GOOD_STATUS, round_number=event, event_name=EVENT_NAMES[event]
        ),
    )
    monkeypatch.setattr(ingest, "extract_weather", lambda session: {})

    report = IngestReport()
    frame = ingest.collect_season_results(2024, report=report)

    assert report.failed == []
    assert len(frame) == 80  # 2 rounds x (race + sprint) x 20 drivers
    assert bool(report) is True


# --------------------------------------------------------------------------- #
# Retry-loop paths the `spy` fixture above cannot reach
# --------------------------------------------------------------------------- #
#
# `spy` installs a single always-raising exception, which covers giving up.
# Recovering part-way through, and the flags the loop forwards to `load`, need
# a stub that varies per attempt.


def install_get_session(monkeypatch, side_effects):
    """Stub ``fastf1.get_session``; ``side_effects`` is consumed one per call."""
    calls: list[tuple] = []
    remaining = list(side_effects)

    def get_session(year, event, session_type):
        calls.append((year, event, session_type))
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    module = types.ModuleType("fastf1")
    module.get_session = get_session
    monkeypatch.setitem(sys.modules, "fastf1", module)
    return calls


class _Loadable:
    """A session whose ``load`` succeeds and records the flags it was given."""

    def __init__(self) -> None:
        self.load_kwargs: dict | None = None

    def load(self, **kwargs) -> None:
        self.load_kwargs = kwargs


@pytest.fixture
def no_sleep(monkeypatch):
    """Record backoff delays instead of serving them."""
    slept: list[float] = []
    monkeypatch.setattr(ingest.time, "sleep", slept.append)
    return slept


def test_load_session_returns_on_the_first_success(monkeypatch, no_sleep) -> None:
    session = _Loadable()
    calls = install_get_session(monkeypatch, [session])

    assert load_session(2024, 8, "R") is session
    assert len(calls) == 1
    assert no_sleep == [], "no backoff should be served on a clean load"


def test_load_session_recovers_after_a_transient_failure(monkeypatch, no_sleep) -> None:
    session = _Loadable()
    calls = install_get_session(
        monkeypatch,
        [ConnectionError("connection reset by peer"),
         ConnectionError("connection reset by peer"),
         session],
    )

    assert load_session(2024, 8, "R", backoff_s=2.0) is session
    assert len(calls) == 3
    assert no_sleep == [2.0, 4.0], "backoff should double between attempts"


def test_load_session_honours_the_retry_count(monkeypatch, no_sleep) -> None:
    calls = install_get_session(
        monkeypatch, [ConnectionError("connection reset by peer")] * 5
    )

    with pytest.raises(RuntimeError):
        load_session(2024, 8, "R", retries=5)

    assert len(calls) == 5
    assert no_sleep == [2.0, 4.0, 8.0, 16.0]


def test_load_session_passes_the_load_flags_through(monkeypatch, no_sleep) -> None:
    session = _Loadable()
    install_get_session(monkeypatch, [session])

    load_session(2024, 8, "R", laps=False, telemetry=False, weather=True)

    assert session.load_kwargs == {
        "laps": False, "telemetry": False, "weather": True, "messages": False,
    }


# --------------------------------------------------------------------------- #
# A calendar that will not load
# --------------------------------------------------------------------------- #
#
# FastF1 tries three backends for a schedule and raises only when all three
# fail, which means the network is down or Ergast is rate-limiting. Neither is
# a season's own fault, and neither should end the run with a traceback.


def test_a_failed_schedule_is_reported_not_raised(monkeypatch) -> None:
    module = types.ModuleType("fastf1")

    def explode(year, include_testing=False):
        raise ValueError("Failed to load any schedule data.")

    module.get_event_schedule = explode
    monkeypatch.setitem(sys.modules, "fastf1", module)

    report = IngestReport()
    frame = ingest.collect_season_results(2024, report=report)

    assert frame.empty
    assert len(report.failed) == 1
    label, message = report.failed[0]
    assert label == "2024 schedule"
    assert "Failed to load any schedule data" in message


def test_one_bad_season_does_not_stop_the_others(monkeypatch) -> None:
    """A rate limit that clears mid-run must not cost the remaining seasons."""
    schedule = pd.DataFrame(
        {"RoundNumber": list(EVENT_NAMES), "EventName": list(EVENT_NAMES.values())}
    )
    module = types.ModuleType("fastf1")

    def sometimes(year, include_testing=False):
        if year == 2024:
            raise ValueError("Failed to load any schedule data.")
        return schedule

    module.get_event_schedule = sometimes
    monkeypatch.setitem(sys.modules, "fastf1", module)
    monkeypatch.setattr(
        ingest, "load_session",
        lambda year, event, session_type, **kw: make_session(
            GOOD_STATUS, round_number=event, event_name=EVENT_NAMES[event],
            year=year,
        ),
    )
    monkeypatch.setattr(ingest, "extract_weather", lambda session: {})

    frame, report = ingest.collect_results([2024, 2025])

    assert not frame.empty, "2025 should still have been collected"
    assert set(frame["Year"]) == {2025}
    assert any(label == "2024 schedule" for label, _ in report.failed)
