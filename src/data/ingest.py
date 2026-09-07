"""Pull race results, weather and reference-lap telemetry out of FastF1.

This is the only module that talks to the network.  Everything downstream
operates on DataFrames, which is what lets the rest of the pipeline be tested
without reaching the F1 timing API at all.

Three practical notes about running it:

* **Cache first.**  FastF1's cache turns a twenty-minute season download into
  a few seconds on the second run.  :func:`configure` enables it against
  :data:`src.config.FASTF1_CACHE_DIR` and should be called once at start-up.
  Telemetry is tens of megabytes per session, so budget a few GB for a full
  2018-2025 pull.
* **Partial failures are normal.**  Some sessions have no telemetry, some
  events are cancelled, and the 2020 season is missing races that were
  scheduled and never run.  Every loader collects failures into a report
  rather than aborting, so one bad session does not cost a whole season.
* **A load can succeed and still be useless.**  ``Status`` comes from Ergast,
  not from F1 timing, and ``Session.load`` swallows an Ergast failure -- a
  rate-limited round yields a full grid of drivers with a blank ``Status``,
  which the labeller would read as an all-retirement race.
  :func:`extract_results` rejects such a frame with
  :class:`DegradedResultsError` so it lands in the report instead of the
  dataset.
* **Egress may be blocked.**  ``livetiming.formula1.com`` and ``api.jolpi.ca``
  are refused outright by some corporate and sandboxed networks.
  :func:`check_connectivity` reports that clearly instead of leaving you to
  interpret a stack trace.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src import config
from src.features.labels import has_usable_status
from src.features.track_profile import build_lap_profile

log = logging.getLogger(__name__)

#: Session identifiers, in the order the reference-lap picker prefers them.
#: Qualifying first: it is the cleanest representation of the layout at the
#: limit, on low fuel and without traffic.  The race is the fallback.
REFERENCE_SESSION_ORDER = ("Q", "R", "SQ", "S")

#: Columns kept from ``Session.results``.  Everything the labelling and feature
#: code needs, nothing else, so the raw frame stays small enough to keep in
#: memory for eight seasons at once.
RESULT_COLUMNS = (
    "DriverNumber", "Abbreviation", "DriverId", "FullName",
    "TeamName", "TeamId",
    "Position", "ClassifiedPosition", "GridPosition",
    "Status", "Points", "Laps", "Time",
)


#: Sessions whose results are expected to carry an Ergast ``Status``.  The
#: quali-like sprint sessions are not among them: Ergast has no results for
#: those, and FastF1 says so explicitly when it falls back to timing data.
STATUS_BEARING_SESSIONS = frozenset({"R", "S"})


class DegradedResultsError(RuntimeError):
    """A session loaded, but its results carry no finishing status at all.

    Raised rather than returning the rows, because the rows are worse than
    useless: :func:`src.features.labels.add_race_outcome_labels` reads a blank
    ``Status`` as a retirement, so accepting them writes a round in which every
    driver retired.  Failing here puts the round in the :class:`IngestReport`
    instead, where it can be re-pulled.

    The usual cause is HTTP 429 from ``api.jolpi.ca``.  FastF1's
    ``ergast/interface.py::_get`` raises ``ErgastInvalidRequestError`` on any
    non-200, ``Session.load`` catches it, logs "No result data for this session
    available on Ergast!" at WARNING and carries on with timing data alone --
    so the load *succeeds*, with a full grid of drivers and an empty ``Status``
    column.  Slow the pull down or retry the affected rounds.
    """


@dataclass
class IngestReport:
    """What succeeded and what did not, for a batch load."""

    loaded: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def add_success(self, label: str) -> None:
        self.loaded.append(label)

    def add_failure(self, label: str, error: BaseException | str) -> None:
        self.failed.append((label, str(error)))

    def summary(self) -> str:
        lines = [f"loaded {len(self.loaded)}, failed {len(self.failed)}"]
        lines.extend(f"  FAILED {label}: {err}" for label, err in self.failed)
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return bool(self.loaded)


def configure(cache_dir=None, *, offline: bool = False) -> None:
    """Enable the FastF1 cache.  Call once before any loading.

    Order matters: ``enable_cache`` must precede ``offline_mode`` or the
    offline flag is silently ignored.
    """
    import fastf1

    cache_dir = cache_dir or config.FASTF1_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    if offline:
        fastf1.Cache.offline_mode(True)
    log.info("FastF1 cache at %s (offline=%s)", cache_dir, offline)


def check_connectivity(timeout: float = 15.0) -> tuple[bool, str]:
    """Probe the hosts FastF1 depends on.

    Returns ``(reachable, message)``.  Run this before a long pull so a
    blocked egress policy surfaces in one line instead of as a wall of
    retries.
    """
    import requests

    hosts = {
        "livetiming.formula1.com": "https://livetiming.formula1.com/static/2024/Index.json",
        "api.jolpi.ca": "https://api.jolpi.ca/ergast/f1/2024/1/results.json?limit=1",
    }
    problems = []
    for name, url in hosts.items():
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code >= 400:
                problems.append(f"{name} returned HTTP {response.status_code}")
        except Exception as exc:
            problems.append(f"{name} unreachable ({type(exc).__name__}: {exc})")
    if problems:
        return False, (
            "F1 data hosts are not reachable from here:\n  "
            + "\n  ".join(problems)
            + "\nRun the pull from a network that permits these hosts, or "
              "populate the FastF1 cache elsewhere and copy it in."
        )
    return True, "F1 data hosts reachable."


#: Substrings that mark a failure as transient whatever exception type carries
#: it.  FastF1 raises ``ErgastInvalidRequestError`` for *every* non-200 from the
#: Ergast backend, so an HTTP 429 and a genuinely malformed request arrive as
#: the same class and can only be told apart by the message.  These are checked
#: first and win over every permanent rule below.
TRANSIENT_MESSAGE_MARKERS = (
    "too many requests",
    "rate limit",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
)

#: ``ValueError`` messages from FastF1's schedule lookup that mean the session
#: is simply not in the calendar.  Kept as substrings because FastF1 formats
#: the identifier into the message.
PERMANENT_VALUE_ERROR_MARKERS = (
    "does not exist",
    "invalid session type",
    "invalid round",
    "cannot get testing event",
)


@lru_cache(maxsize=1)
def _permanent_error_types() -> tuple[type[BaseException], ...]:
    """FastF1 exception classes that mean "this session will never load".

    Resolved lazily so this module stays importable without fastf1, and
    defensively because ``SessionNotAvailableError`` lives in a private module:
    the ``fastf1.api`` shim that re-exports it warns on import.
    """
    try:
        from fastf1.exceptions import (
            ErgastInvalidRequestError,
            InvalidSessionError,
            NoLapDataError,
            RateLimitExceededError,
        )
    except ImportError:  # fastf1 absent; the message check below still applies
        return ()

    types: list[type[BaseException]] = [
        InvalidSessionError,        # no session matches this event/type/year
        NoLapDataError,             # request succeeded, no usable data returned
        ErgastInvalidRequestError,  # server rejected the request as invalid
                                    # (except HTTP 429 -- see the markers above)
        RateLimitExceededError,     # fastf1's own limiter; the window is hours,
                                    # so a seconds-long backoff cannot clear it
    ]
    try:
        from fastf1._api import SessionNotAvailableError
    except ImportError:  # pragma: no cover - depends on the fastf1 version
        pass
    else:
        types.append(SessionNotAvailableError)  # cancelled or absent session
    return tuple(types)


def is_permanent_failure(exc: BaseException) -> bool:
    """True when re-requesting the same session cannot change the outcome.

    A sprint before 2021, a cancelled event or a round outside the calendar
    fails identically every time.  Backing off between identical attempts only
    burns wall-clock: a 2018-2020 pull spends about six seconds per event
    retrying a sprint that was never scheduled.
    """
    message = str(exc).lower()
    # Checked first: a rate-limited or briefly-broken server says nothing about
    # whether the session exists.
    if any(marker in message for marker in TRANSIENT_MESSAGE_MARKERS):
        return False
    if isinstance(exc, _permanent_error_types()):
        return True
    # ``ValueError`` is overloaded in fastf1.events.  Most instances mean "not
    # in the calendar", but "Failed to load any schedule data." is a network
    # failure and has to stay retryable.
    if isinstance(exc, ValueError):
        return any(marker in message for marker in PERMANENT_VALUE_ERROR_MARKERS)
    return False


def load_session(
    year: int,
    event: str | int,
    session_type: str,
    *,
    laps: bool = True,
    telemetry: bool = False,
    weather: bool = True,
    messages: bool = False,
    retries: int = 3,
    backoff_s: float = 2.0,
):
    """Load one session, retrying transient failures with exponential backoff.

    Failures that cannot resolve themselves -- a sprint that was never on the
    calendar, a cancelled session -- are raised on the first attempt instead of
    being slept over; see :func:`is_permanent_failure`.

    ``telemetry=True`` is expensive; leave it off unless the caller needs a
    reference lap.
    """
    import fastf1

    last: BaseException | None = None
    for attempt in range(retries):
        try:
            session = fastf1.get_session(year, event, session_type)
            session.load(
                laps=laps, telemetry=telemetry, weather=weather, messages=messages
            )
            return session
        except Exception as exc:  # noqa: BLE001 - surfaced through the report
            last = exc
            if is_permanent_failure(exc):
                log.debug(
                    "load %s %s %s failed permanently (%s); not retrying",
                    year, event, session_type, exc,
                )
                break
            if attempt < retries - 1:
                delay = backoff_s * (2**attempt)
                log.warning(
                    "load %s %s %s failed (%s); retrying in %.0fs",
                    year, event, session_type, exc, delay,
                )
                time.sleep(delay)
    raise RuntimeError(f"could not load {year} {event} {session_type}") from last


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def optional_session_attr(session, name: str, default=np.nan):
    """Read a ``Session`` property that the requested load may not have filled.

    ``getattr(session, name, default)`` looks like it covers this, but does
    not: FastF1 properties raise
    :class:`~fastf1.exceptions.DataNotLoadedError` when the relevant part of
    the session was never loaded, and ``getattr`` only swallows
    ``AttributeError``.  ``total_laps`` is the case that bites -- it is
    populated solely by the lap-loading path, which the results pass skips on
    purpose because laps are expensive and nothing downstream models them.
    """
    try:
        value = getattr(session, name)
    except Exception:  # noqa: BLE001 - any load-state failure means "absent"
        return default
    return default if value is None else value


def extract_results(
    session, *, session_label: str = "R", require_status: bool = True
) -> pd.DataFrame:
    """One row per driver from a loaded session, with event identity attached.

    Args:
        session: A loaded FastF1 session.
        session_label: Value written to the ``session_type`` column.
        require_status: Reject a results frame in which no row carries a
            finishing status.  See :class:`DegradedResultsError` for why this
            is a hard failure rather than a warning.

    Raises:
        DegradedResultsError: ``require_status`` is set, the session is one of
            :data:`STATUS_BEARING_SESSIONS`, and its ``Status`` column is
            missing or blank on every row.
    """
    results = pd.DataFrame(session.results).copy()
    keep = [c for c in RESULT_COLUMNS if c in results.columns]
    frame = results[keep].reset_index(drop=True)

    if (
        require_status
        and session_label in STATUS_BEARING_SESSIONS
        and len(frame)
    ):
        # An absent column and an all-blank one have the same cause and the
        # same consequence, so they get the same treatment.
        usable = (
            has_usable_status(frame["Status"])
            if "Status" in frame.columns
            else pd.Series(False, index=frame.index)
        )
        if not usable.any():
            raise DegradedResultsError(
                f"{len(frame)} driver row(s) but no finishing status on any of "
                f"them -- the Ergast backend almost certainly rate-limited this "
                f"session, so the rows are dropped rather than labelled as an "
                f"all-retirement race"
            )

    event = session.event
    frame["Year"] = int(event["EventDate"].year)
    frame["RoundNumber"] = int(event["RoundNumber"])
    frame["EventName"] = event["EventName"]
    frame["Location"] = event.get("Location")
    frame["Country"] = event.get("Country")
    frame["RaceDate"] = pd.to_datetime(event["EventDate"])
    frame["session_type"] = session_label
    frame["total_laps"] = optional_session_attr(session, "total_laps")

    try:
        meeting = session.session_info["Meeting"]["Circuit"]
        frame["circuit_key"] = meeting["Key"]
        frame["circuit_short_name"] = meeting.get("ShortName")
    except Exception:  # pragma: no cover - depends on API payload shape
        frame["circuit_key"] = np.nan
        frame["circuit_short_name"] = None
    return frame


def extract_weather(session) -> dict[str, Any]:
    """Race-condition summary.

    These describe conditions *during* the race, so they are tagged
    ``race_day`` in the feature registry and are not available at forecast
    time.  They are collected anyway: a retrospective model that includes them
    tells you how much of retirement risk weather explains, which is worth
    knowing even if you cannot use it on Thursday.
    """
    try:
        weather = pd.DataFrame(session.weather_data)
    except Exception:  # pragma: no cover - session may carry no weather
        return {}
    if weather.empty:
        return {}
    out: dict[str, Any] = {}
    for column, name in (
        ("AirTemp", "air_temp_c"),
        ("TrackTemp", "track_temp_c"),
        ("Humidity", "humidity_pct"),
        ("Pressure", "pressure_mbar"),
        ("WindSpeed", "wind_speed_ms"),
    ):
        if column in weather.columns:
            values = pd.to_numeric(weather[column], errors="coerce")
            out[f"{name}_mean"] = float(values.mean())
            out[f"{name}_max"] = float(values.max())
    if "Rainfall" in weather.columns:
        rain = weather["Rainfall"].astype(bool)
        out["rain_share"] = float(rain.mean())
        out["any_rain"] = int(rain.any())
    return out


def collect_season_results(
    year: int,
    *,
    include_sprints: bool = True,
    report: IngestReport | None = None,
) -> pd.DataFrame:
    """Every race (and optionally sprint) result for one season.

    Sprints are collected because they are extra observations of car
    reliability, which the rolling features can use even though sprint rows are
    excluded from the modelling set.
    """
    import fastf1

    report = report if report is not None else IngestReport()
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:  # noqa: BLE001 - surfaced through the report
        # FastF1 tries three backends for a calendar and raises only when all
        # three fail, which means the network is down or the Ergast backend is
        # rate-limiting.  Neither is this season's fault, and neither should
        # take down the seasons after it, so it is reported like any other
        # failure -- the same way collect_circuit_profiles has always handled it.
        log.warning("could not load the %s schedule: %s", year, exc)
        report.add_failure(f"{year} schedule", exc)
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []

    for _, event in schedule.iterrows():
        label = f"{year} {event['EventName']}"
        wanted = ["R"] + (["S"] if include_sprints else [])
        for session_type in wanted:
            try:
                session = load_session(
                    year, event["RoundNumber"], session_type,
                    laps=False, telemetry=False, weather=True,
                )
                frame = extract_results(session, session_label=session_type)
                for key, value in extract_weather(session).items():
                    frame[key] = value
                frames.append(frame)
                report.add_success(f"{label} [{session_type}]")
            except DegradedResultsError as exc:
                # Always reported, sprint included: unlike an absent sprint,
                # this means the backend answered and the answer was unusable,
                # which is worth seeing for every session it touched.
                log.warning("degraded results for %s [%s]: %s",
                            label, session_type, exc)
                report.add_failure(f"{label} [{session_type}]", exc)
            except Exception as exc:  # noqa: BLE001
                # A missing sprint is expected on most weekends; a missing race
                # is worth seeing in the report.
                if session_type == "R":
                    report.add_failure(f"{label} [{session_type}]", exc)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_results(
    seasons: Sequence[int] = config.DEFAULT_SEASONS,
    *,
    include_sprints: bool = True,
) -> tuple[pd.DataFrame, IngestReport]:
    """Race results across several seasons, plus a report of what failed."""
    report = IngestReport()
    frames = []
    for year in seasons:
        log.info("collecting results for %s", year)
        frame = collect_season_results(
            year, include_sprints=include_sprints, report=report
        )
        if not frame.empty:
            frames.append(frame)
    combined = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    return combined, report


# --------------------------------------------------------------------------- #
# Reference laps and circuit profiles
# --------------------------------------------------------------------------- #


def pick_reference_lap(year: int, event: str | int) -> tuple[Any, Any, str]:
    """Find the cleanest representative lap for a circuit in a given season.

    Tries qualifying first, then the race, then the sprint sessions.  Returns
    ``(lap, session, session_type)``.

    Raises:
        RuntimeError: if no session yields a usable fastest lap.
    """
    problems = []
    for session_type in REFERENCE_SESSION_ORDER:
        try:
            session = load_session(
                year, event, session_type,
                laps=True, telemetry=True, weather=False, retries=2,
            )
            lap = session.laps.pick_fastest()
            if lap is None or (hasattr(lap, "empty") and lap.empty):
                problems.append(f"{session_type}: no fastest lap")
                continue
            return lap, session, session_type
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{session_type}: {exc}")
    raise RuntimeError(
        f"no usable reference lap for {year} {event}: " + "; ".join(problems)
    )


def profile_event(
    year: int, event: str | int, *, step_m: float = config.TRACK_RESAMPLE_STEP_M
) -> dict[str, Any]:
    """Build one circuit profile row for a single event in a single season.

    Profiles are keyed on ``(circuit_key, year)`` rather than circuit alone
    because layouts change.  Zandvoort gained banking in 2021, Melbourne was
    reprofiled in 2022 and Yas Marina reworked in 2021; each produces a
    genuinely different profile, and averaging across the change would blur
    both.
    """
    lap, session, session_type = pick_reference_lap(year, event)
    telemetry = lap.get_telemetry()

    corners = None
    try:
        circuit_info = session.get_circuit_info()
        if circuit_info is not None:
            corners = circuit_info.corners
    except Exception as exc:  # pragma: no cover - MultiViewer data may be absent
        log.debug("no circuit info for %s %s: %s", year, event, exc)

    meta: dict[str, Any] = {
        "year": int(year),
        "reference_session": session_type,
        "reference_driver": lap.get("Driver") if hasattr(lap, "get") else None,
        "reference_lap_time_s": (
            lap["LapTime"].total_seconds()
            if pd.notna(lap.get("LapTime"))
            else np.nan
        ),
    }
    try:
        circuit = session.session_info["Meeting"]["Circuit"]
        meta["circuit_key"] = circuit["Key"]
        meta["circuit_name"] = circuit.get("ShortName")
    except Exception:  # pragma: no cover
        meta["circuit_key"] = np.nan
        meta["circuit_name"] = None
    meta["event_name"] = session.event["EventName"]

    return build_lap_profile(telemetry, step_m=step_m, corners=corners, metadata=meta)


def collect_circuit_profiles(
    seasons: Sequence[int] = config.DEFAULT_SEASONS,
    *,
    step_m: float = config.TRACK_RESAMPLE_STEP_M,
) -> tuple[pd.DataFrame, IngestReport]:
    """Profile every circuit in every requested season.

    This is the expensive part of the pipeline: it downloads telemetry for one
    session per event.  Expect roughly a minute per event on a cold cache and
    seconds on a warm one.
    """
    import fastf1

    report = IngestReport()
    rows = []
    for year in seasons:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:  # noqa: BLE001
            report.add_failure(f"{year} schedule", exc)
            continue
        for _, event in schedule.iterrows():
            label = f"{year} {event['EventName']}"
            try:
                rows.append(profile_event(year, event["RoundNumber"], step_m=step_m))
                report.add_success(label)
                log.info("profiled %s", label)
            except Exception as exc:  # noqa: BLE001
                report.add_failure(label, exc)
    return (pd.DataFrame(rows) if rows else pd.DataFrame()), report
