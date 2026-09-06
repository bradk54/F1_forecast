"""Retry policy for the one module that talks to the network.

Backing off costs wall-clock, so the distinction that matters is whether a
second identical request could plausibly succeed.  A sprint that was never on
the calendar cannot; a dropped connection can.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingest import (
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
