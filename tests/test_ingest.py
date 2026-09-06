"""Ingest-layer behaviour: the retry policy, and the degraded-results guard.

Nothing here touches the network.  ``load_session`` and the FastF1 entry points
are stubbed, which is the whole point of keeping network access confined to
:mod:`src.data.ingest`.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.data import ingest
from src.data.ingest import (
    DegradedResultsError,
    IngestReport,
    extract_results,
    load_session,
)

# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


def make_session(
    statuses,
    *,
    n_drivers: int | None = None,
    drop_status: bool = False,
    round_number: int = 8,
    event_name: str = "Monaco Grand Prix",
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
            "EventDate": pd.Timestamp("2024-05-26"),
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


# --------------------------------------------------------------------------- #
# extract_results: the guard
# --------------------------------------------------------------------------- #


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
# collect_season_results: degraded rounds land in the report
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
# load_session: the retry policy
# --------------------------------------------------------------------------- #


@pytest.fixture
def no_sleep(monkeypatch):
    """Record backoff delays instead of serving them."""
    slept: list[float] = []
    monkeypatch.setattr(ingest.time, "sleep", slept.append)
    return slept


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


def test_load_session_returns_on_the_first_success(monkeypatch, no_sleep) -> None:
    session = _Loadable()
    calls = install_get_session(monkeypatch, [session])

    assert load_session(2024, 8, "R") is session
    assert len(calls) == 1
    assert no_sleep == [], "no backoff should be served on a clean load"


def test_load_session_retries_then_succeeds(monkeypatch, no_sleep) -> None:
    session = _Loadable()
    calls = install_get_session(
        monkeypatch, [ConnectionError("boom"), ConnectionError("boom"), session]
    )

    assert load_session(2024, 8, "R", backoff_s=2.0) is session
    assert len(calls) == 3
    assert no_sleep == [2.0, 4.0], "backoff should double between attempts"


def test_load_session_gives_up_after_the_last_attempt(monkeypatch, no_sleep) -> None:
    calls = install_get_session(monkeypatch, [ConnectionError("boom")] * 3)

    with pytest.raises(RuntimeError, match="could not load 2024 8 R") as excinfo:
        load_session(2024, 8, "R")

    assert len(calls) == 3
    assert len(no_sleep) == 2, "no backoff after the final attempt"
    assert isinstance(excinfo.value.__cause__, ConnectionError)


def test_load_session_honours_the_retry_count(monkeypatch, no_sleep) -> None:
    calls = install_get_session(monkeypatch, [ConnectionError("boom")] * 5)

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
