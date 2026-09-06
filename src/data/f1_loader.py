"""fastf1 season loader for the F1 forecast project.

Replaces the ad-hoc ``F1DataFetcher`` cell in
``Notebooks/2_0_Model_Development.ipynb``.

Give it a start year and an end year and it returns one row per
driver-race — race result, qualifying and sprint columns side by side —
with the remaining (not-yet-run) races of the final season carried
forward so the same frame can be fed straight into feature building and
prediction.

Design points
-------------
* **Wide output.** ``F1Data.wide`` is one row per ``(Year, Round,
  DriverId)``. The tidy per-session frame it was built from is kept on
  ``F1Data.long`` because it costs nothing and the existing notebook
  feature code speaks that dialect (``Event`` column).
* **Offline-first with a live re-check of the in-progress season.**
  Sessions are served from the fastf1 cache; misses go online for that
  one load and then drop back to offline. Events of the current season
  that finished inside ``recheck_days`` are re-read so a just-completed
  race shows up without a manual prefetch.
* **Parquet snapshots.** Each season's tidy frame is written to
  ``Data/processed/fastf1/sessions_<year>.parquet``. Later calls read
  that instead of walking the cache. When the snapshot is absent *or
  incomplete*, the loader falls back to the fastf1 cache / network for
  exactly the sessions it is missing and rewrites the snapshot.
* **Completeness auditing.** Every load is checked against the official
  schedule: missing events, missing sessions, short driver counts and
  duplicate keys are reported on ``F1Data.audit``. Partial seasons are
  repaired rather than silently returned.

Sprint qualifying is deliberately not loaded. fastf1 3.6.1 returns a
sprint-shootout / sprint-qualifying session with no classification at all
(``Position``, ``DriverId`` and ``TeamId`` are empty, with or without laps
loaded), because Ergast never covered it. ``SprintGridPosition`` — taken
from the sprint race results — is the faithful stand-in, so that is what
the wide frame carries. ``SprintQualiDate`` is still tracked on the event
table for scheduling.

Usage
-----
From a notebook in ``Notebooks/``::

    import sys; sys.path.append('..')
    from src.data.f1_loader import load_seasons

    data = load_seasons(2022, 2026)   # inclusive; 2026's remaining races included
    data.summary()                    # per-season audit + sanity table
    df = data.wide                    # one row per driver-race

From the shell::

    python src/data/f1_loader.py 2022 2026 --out Data/processed/wide.parquet
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

import fastf1
import pandas as pd

__all__ = [
    "F1SeasonLoader",
    "F1Data",
    "SeasonAudit",
    "SessionGap",
    "load_seasons",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths — resolved relative to this file so the module works from anywhere
# (repo root, Notebooks/, a script) without hardcoded absolute paths.
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "Data" / "raw"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "Data" / "processed" / "fastf1"

# Canonical session buckets. fastf1 renamed the sprint sessions twice
# ('Sprint Qualifying' in 2021-22 meant the sprint race itself, 'Sprint
# Shootout' in 2023, 'Sprint Qualifying' again from 2024), so full
# session names from the schedule are normalised through this map instead
# of guessing from EventFormat.
SESSION_KEYS: dict[str, str] = {
    "Race": "Race",
    "Qualifying": "Qualifying",
    "Sprint": "Sprint",
    "Sprint Qualifying": "SprintQualifying",
    "Sprint Shootout": "SprintQualifying",
}

# Sessions that actually yield results. See the module docstring for why
# SprintQualifying is not one of them.
LOADABLE_SESSIONS = ("Race", "Qualifying", "Sprint")

# Results-only load. The notebook version passed laps/telemetry/weather/
# messages=True, which pulls tens of MB per session; nothing downstream of
# session.results needs it, and skipping it turns a season load into
# roughly a second.
RESULTS_ONLY_KWARGS = dict(laps=False, telemetry=False, weather=False, messages=False)

# Columns taken straight off session.results.
_RESULT_COLS = [
    "DriverId",
    "DriverNumber",
    "Abbreviation",
    "FullName",
    "TeamId",
    "TeamName",
    "Position",
    "ClassifiedPosition",
    "GridPosition",
    "Status",
    "Points",
    "Laps",
    "Time",
    "Q1",
    "Q2",
    "Q3",
]

_EVENT_COLS = [
    "Year",
    "Round",
    "Race",
    "EventFormat",
    "Country",
    "Location",
    "EventDate",
    "RaceDate",
    "QualiDate",
    "SprintDate",
    "SprintQualiDate",
]

FutureRows = Literal["drivers", "events", "none"]


# --------------------------------------------------------------------------
# Audit types
# --------------------------------------------------------------------------
@dataclass
class SessionGap:
    """One thing that is missing or suspicious about a loaded season."""

    year: int
    round: Optional[int]
    event: str
    session: str
    reason: str  # missing | thin | error | duplicate | missing_event
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "Year": self.year,
            "Round": self.round,
            "Event": self.event,
            "Session": self.session,
            "Reason": self.reason,
            "Detail": self.detail,
        }


@dataclass
class SeasonAudit:
    """Completeness report for a single season."""

    year: int
    expected_sessions: int = 0
    loaded_sessions: int = 0
    expected_events: int = 0
    loaded_events: int = 0
    future_events: int = 0
    source: str = "unknown"  # parquet | cache | mixed | empty
    gaps: list[SessionGap] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.gaps

    def as_dict(self) -> dict:
        return {
            "Year": self.year,
            "Source": self.source,
            "EventsLoaded": self.loaded_events,
            "EventsExpected": self.expected_events,
            "SessionsLoaded": self.loaded_sessions,
            "SessionsExpected": self.expected_sessions,
            "FutureEvents": self.future_events,
            "Gaps": len(self.gaps),
            "Complete": self.complete,
        }


@dataclass
class F1Data:
    """Result bundle returned by :meth:`F1SeasonLoader.load`."""

    wide: pd.DataFrame
    long: pd.DataFrame
    schedule: pd.DataFrame
    events: pd.DataFrame
    future_events: pd.DataFrame
    audit: list[SeasonAudit]

    @property
    def complete(self) -> bool:
        return all(a.complete for a in self.audit)

    def audit_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.as_dict() for a in self.audit])

    def gap_frame(self) -> pd.DataFrame:
        rows = [g.as_dict() for a in self.audit for g in a.gaps]
        return pd.DataFrame(rows, columns=["Year", "Round", "Event", "Session", "Reason", "Detail"])

    def summary(self) -> pd.DataFrame:
        """Print a per-season sanity table and return the audit frame."""
        frame = self.audit_frame()
        print(frame.to_string(index=False))
        gaps = self.gap_frame()
        if len(gaps):
            print(f"\n{len(gaps)} gap(s):")
            print(gaps.to_string(index=False))
        else:
            print("\nNo gaps — every expected session is present.")

        past = self.wide[~self.wide["IsFuture"]]
        if len(past):
            per_year = (
                past.groupby("Year")
                .agg(
                    Races=("Round", "nunique"),
                    DriverRows=("DriverId", "size"),
                    RacePoints=("RacePoints", "sum"),
                    MissingRaceResult=("RacePosition", lambda s: int(s.isna().sum())),
                    MissingQuali=("QualiPosition", lambda s: int(s.isna().sum())),
                )
                .reset_index()
            )
            print("\nCompleted races by season:")
            print(per_year.to_string(index=False))
        n_future = int(self.wide["IsFuture"].sum())
        if n_future:
            print(f"\n{n_future} forward-populated row(s) across "
                  f"{self.future_events['Round'].nunique()} remaining event(s).")
        return frame


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------
class F1SeasonLoader:
    """Cache-aware, self-auditing multi-season fastf1 loader.

    Parameters
    ----------
    cache_dir:
        fastf1 cache directory. Defaults to ``Data/raw`` (which is also
        where the pitwall pickles live — fastf1 keeps to its own
        subdirectories inside it).
    snapshot_dir:
        Where per-season parquet snapshots are written. Defaults to
        ``Data/processed/fastf1``.
    min_drivers:
        A completed session with fewer classified drivers than this is
        flagged as ``thin`` and reloaded. 15 clears every real field from
        2000 on (modern grids are 20, mid-2000s were 22-24).
    recheck_days:
        For the in-progress season, events whose race finished within
        this many days are re-read from the cache/network instead of
        being trusted from the snapshot.
    quiet:
        Suppress fastf1's own logging (it is very chatty).
    """

    def __init__(
        self,
        cache_dir: os.PathLike | str = DEFAULT_CACHE_DIR,
        snapshot_dir: os.PathLike | str = DEFAULT_SNAPSHOT_DIR,
        min_drivers: int = 15,
        recheck_days: int = 21,
        quiet: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.min_drivers = min_drivers
        self.recheck_days = recheck_days

        if quiet:
            fastf1.set_log_level(logging.ERROR)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Order matters: enable_cache before offline_mode.
        fastf1.Cache.enable_cache(str(self.cache_dir))
        fastf1.Cache.offline_mode(True)

    # ----------------------------------------------------------------- net
    @contextmanager
    def _online(self, renew: bool = False):
        """Lift offline mode for one load, then restore it.

        ``renew=True`` additionally bypasses the on-disk ``.ff1pkl``
        pickle for this load, which is what a re-check of a
        just-completed session needs. It sets ``Cache._FORCE_RENEW``
        directly rather than calling ``enable_cache(force_renew=True)``,
        because that public path also calls ``cache.clear()`` on the
        whole HTTP request cache — many GB of already-downloaded data.
        """
        fastf1.Cache.offline_mode(False)
        previous = fastf1.Cache._FORCE_RENEW
        if renew:
            fastf1.Cache._FORCE_RENEW = True
        try:
            yield
        finally:
            fastf1.Cache._FORCE_RENEW = previous
            fastf1.Cache.offline_mode(True)

    @staticmethod
    def _now() -> pd.Timestamp:
        return pd.Timestamp.utcnow().tz_localize(None)

    # ------------------------------------------------------------ schedule
    def _schedule_path(self, year: int) -> Path:
        return self.snapshot_dir / f"schedule_{year}.parquet"

    def get_schedule(self, year: int, refresh: bool = False) -> pd.DataFrame:
        """Event schedule for a season, without testing events.

        Cached to parquet. The in-progress (and any future) season is
        always refreshed, since rounds get added, moved and cancelled.
        """
        path = self._schedule_path(year)
        stale = refresh or year >= self._now().year
        if path.exists() and not stale:
            return pd.read_parquet(path)

        try:
            with self._online():
                schedule = fastf1.get_event_schedule(year, include_testing=False)
            schedule = pd.DataFrame(schedule).reset_index(drop=True)
            schedule.to_parquet(path, index=False)
            return schedule
        except Exception as exc:  # offline, or a season with no schedule yet
            if path.exists():
                logger.warning("%s schedule fetch failed (%s) — using cached copy", year, exc)
                return pd.read_parquet(path)
            logger.warning("%s schedule unavailable: %s", year, exc)
            return pd.DataFrame()

    def _event_table(self, year: int) -> pd.DataFrame:
        """One row per event with normalised session dates and flags.

        Columns: the ``_EVENT_COLS`` identity block plus ``Has*``
        (session scheduled), ``*Done`` (session finished) and
        ``IsFuture`` (the race itself has not finished).
        """
        schedule = self.get_schedule(year)
        if schedule.empty:
            return pd.DataFrame(columns=_EVENT_COLS)

        now = self._now()
        rows = []
        for _, event in schedule.iterrows():
            dates: dict[str, pd.Timestamp] = {}
            for n in range(1, 6):
                name = event.get(f"Session{n}")
                key = SESSION_KEYS.get(str(name).strip()) if pd.notna(name) else None
                if key:
                    dates[key] = pd.to_datetime(event.get(f"Session{n}DateUtc"))

            race_date = dates.get("Race")
            rows.append(
                {
                    "Year": int(year),
                    "Round": int(event["RoundNumber"]),
                    "Race": event["EventName"],
                    "EventFormat": event.get("EventFormat"),
                    "Country": event.get("Country"),
                    "Location": event.get("Location"),
                    "EventDate": pd.to_datetime(event.get("EventDate")),
                    "RaceDate": race_date,
                    "QualiDate": dates.get("Qualifying"),
                    "SprintDate": dates.get("Sprint"),
                    "SprintQualiDate": dates.get("SprintQualifying"),
                    "HasRace": "Race" in dates,
                    "HasQuali": "Qualifying" in dates,
                    "HasSprint": "Sprint" in dates,
                    "HasSprintQuali": "SprintQualifying" in dates,
                    # A session is treated as finished 4h after its start.
                    "RaceDone": self._done(race_date, now),
                    "QualiDone": self._done(dates.get("Qualifying"), now),
                    "SprintDone": self._done(dates.get("Sprint"), now),
                    "SprintQualiDone": self._done(dates.get("SprintQualifying"), now),
                }
            )

        table = pd.DataFrame(rows).sort_values("Round").reset_index(drop=True)
        # A season with no sprint has an all-None SprintQualiDate, which would
        # land as object dtype and upset the concat across seasons.
        for col in ("EventDate", "RaceDate", "QualiDate", "SprintDate", "SprintQualiDate"):
            table[col] = pd.to_datetime(table[col], errors="coerce")
        table["IsFuture"] = ~table["RaceDone"]
        return table

    @staticmethod
    def _done(when: Optional[pd.Timestamp], now: pd.Timestamp, buffer_hours: int = 4) -> bool:
        if when is None or pd.isna(when):
            return False
        return bool(when + pd.Timedelta(hours=buffer_hours) < now)

    # ------------------------------------------------------------- session
    def _expected_sessions(self, events: pd.DataFrame, include_sprint: bool = True) -> pd.DataFrame:
        """The (Round, SessionKey) pairs that should have results by now."""
        wanted = [s for s in LOADABLE_SESSIONS if include_sprint or s != "Sprint"]
        rows = []
        for _, event in events.iterrows():
            for key in wanted:
                if event[f"Has{_short(key)}"] and event[f"{_short(key)}Done"]:
                    rows.append(
                        {
                            "Year": event["Year"],
                            "Round": event["Round"],
                            "Race": event["Race"],
                            "SessionKey": key,
                            "SessionDate": event[f"{_short(key)}Date"],
                        }
                    )
        return pd.DataFrame(rows, columns=["Year", "Round", "Race", "SessionKey", "SessionDate"])

    def _load_session(
        self,
        year: int,
        round_number: int,
        session_key: str,
        renew: bool = False,
    ) -> pd.DataFrame:
        """Results for one session as a tidy frame. Empty frame on failure."""
        identifier = _SESSION_IDENTIFIER[session_key]

        def _fetch() -> pd.DataFrame:
            session = fastf1.get_session(year, round_number, identifier)
            session.load(**RESULTS_ONLY_KWARGS)
            results = pd.DataFrame(session.results).reset_index(drop=True)
            if results.empty:
                return pd.DataFrame()
            for col in _RESULT_COLS:
                if col not in results.columns:
                    results[col] = pd.NA
            out = _normalize_ids(results[_RESULT_COLS].copy())
            out["Year"] = int(year)
            out["Round"] = int(round_number)
            out["Race"] = session.event["EventName"]
            out["SessionKey"] = session_key
            out["SessionName"] = session.name
            out["SessionDate"] = pd.to_datetime(session.date)
            # Kept for the notebook's existing Event-based feature code.
            out["Event"] = _LEGACY_EVENT[session_key]
            return out

        if not renew:
            try:
                cached = _fetch()
                # A partially cached session loads without raising but comes
                # back empty or without a classification — treat that as a
                # miss so the online attempt below still happens.
                if self._usable(cached):
                    return cached
            except Exception:
                pass
        try:
            with self._online(renew=renew):
                fresh = _fetch()
            if not self._usable(fresh):
                logger.info("%s R%s %s returned no classification",
                            year, round_number, session_key)
                return pd.DataFrame()
            return fresh
        except Exception as exc:
            logger.info("%s R%s %s failed: %s", year, round_number, session_key, exc)
            return pd.DataFrame()

    @staticmethod
    def _usable(results: pd.DataFrame) -> bool:
        """Did a session actually come back with a classification?"""
        if results.empty:
            return False
        return bool(results["DriverId"].notna().any() and results["Position"].notna().any())

    # -------------------------------------------------------------- season
    def _snapshot_path(self, year: int) -> Path:
        return self.snapshot_dir / f"sessions_{year}.parquet"

    def load_season(
        self,
        year: int,
        rebuild: bool = False,
        include_sprint: bool = True,
        use_snapshot: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, SeasonAudit]:
        """Load one season's tidy session results.

        Returns ``(long, events, audit)``. Reads the parquet snapshot when
        one exists, then fills any session the snapshot is missing (or is
        thin on, or that finished recently in the in-progress season) from
        the fastf1 cache / network, and rewrites the snapshot if anything
        changed.
        """
        events = self._event_table(year)
        audit = SeasonAudit(year=year)
        if events.empty:
            audit.source = "empty"
            audit.gaps.append(SessionGap(year, None, "-", "-", "missing_event",
                                         "no schedule available for this season"))
            return pd.DataFrame(), events, audit

        expected = self._expected_sessions(events, include_sprint=include_sprint)
        audit.expected_sessions = len(expected)
        audit.expected_events = int(events["RaceDone"].sum())
        audit.future_events = int(events["IsFuture"].sum())

        path = self._snapshot_path(year)
        long = pd.DataFrame()
        source = "cache"
        if use_snapshot and not rebuild and path.exists():
            try:
                long = self._normalize_snapshot(pd.read_parquet(path))
                source = "parquet"
            except Exception as exc:
                logger.warning("%s snapshot unreadable (%s) — rebuilding", year, exc)

        # What still needs fetching: anything absent, anything with an
        # implausibly small field, and — for the in-progress season —
        # anything that finished inside the recheck window.
        todo = self._sessions_to_fetch(expected, long, year)
        if todo:
            fetched = []
            for round_number, session_key, renew in todo:
                frame = self._load_session(year, round_number, session_key, renew=renew)
                if not frame.empty:
                    fetched.append(frame)
            if fetched:
                new = pd.concat(fetched, ignore_index=True)
                long = new if long.empty else pd.concat([long, new], ignore_index=True)
                # Resolve ids before deduping — NaN keys compare equal to each
                # other and would take real rows with them.
                long = self._backfill_ids(long)
                # Newly fetched rows win over snapshot rows.
                long = long.drop_duplicates(
                    subset=["Year", "Round", "SessionKey", "DriverId"], keep="last"
                )
                source = "parquet+fetch" if source == "parquet" else "cache"
                try:
                    long.reset_index(drop=True).to_parquet(path, index=False)
                except Exception as exc:
                    logger.warning("%s snapshot write failed: %s", year, exc)

        long = self._backfill_ids(long) if not long.empty else long
        audit.source = source
        if long.empty:
            audit.loaded_sessions = 0
            audit.loaded_events = 0
            for _, row in expected.iterrows():
                audit.gaps.append(
                    SessionGap(year, int(row["Round"]), row["Race"], row["SessionKey"], "missing")
                )
            return long, events, audit

        long = long.sort_values(["Round", "SessionKey", "Position"]).reset_index(drop=True)
        self._audit_season(audit, expected, long, events)
        return long, events, audit

    @staticmethod
    def _normalize_snapshot(long: pd.DataFrame) -> pd.DataFrame:
        """Make a snapshot written by an older run safe to reuse."""
        if long.empty:
            return long
        return _normalize_ids(long[long["SessionKey"].isin(LOADABLE_SESSIONS)].copy())

    def _sessions_to_fetch(
        self,
        expected: pd.DataFrame,
        long: pd.DataFrame,
        year: int,
    ) -> list[tuple[int, str, bool]]:
        """(round, session_key, renew) triples that need loading."""
        if expected.empty:
            return []

        counts = (
            long.groupby(["Round", "SessionKey"])["DriverId"].nunique()
            if not long.empty
            else pd.Series(dtype=int)
        )
        now = self._now()
        in_progress = year >= now.year
        recheck_cutoff = now - pd.Timedelta(days=self.recheck_days)

        todo: list[tuple[int, str, bool]] = []
        for _, row in expected.iterrows():
            key = (int(row["Round"]), row["SessionKey"])
            have = int(counts.get(key, 0))
            recent = (
                in_progress
                and pd.notna(row["SessionDate"])
                and row["SessionDate"] >= recheck_cutoff
            )
            if have == 0:
                todo.append((key[0], key[1], False))
            elif have < self.min_drivers:
                # Present but suspicious — bypass the pickle and re-read.
                todo.append((key[0], key[1], True))
            elif recent:
                todo.append((key[0], key[1], True))
        return todo

    def _audit_season(
        self,
        audit: SeasonAudit,
        expected: pd.DataFrame,
        long: pd.DataFrame,
        events: pd.DataFrame,
    ) -> None:
        counts = long.groupby(["Round", "SessionKey"])["DriverId"].nunique()
        audit.loaded_sessions = int(counts.gt(0).sum())
        audit.loaded_events = int(long.loc[long["SessionKey"] == "Race", "Round"].nunique())

        for _, row in expected.iterrows():
            have = int(counts.get((int(row["Round"]), row["SessionKey"]), 0))
            if have == 0:
                audit.gaps.append(
                    SessionGap(audit.year, int(row["Round"]), row["Race"],
                               row["SessionKey"], "missing")
                )
            elif have < self.min_drivers:
                audit.gaps.append(
                    SessionGap(audit.year, int(row["Round"]), row["Race"], row["SessionKey"],
                               "thin", f"only {have} drivers")
                )

        # Every completed event should have produced race rows.
        loaded_rounds = set(long.loc[long["SessionKey"] == "Race", "Round"].unique())
        for _, event in events[events["RaceDone"]].iterrows():
            if int(event["Round"]) not in loaded_rounds:
                audit.gaps.append(
                    SessionGap(audit.year, int(event["Round"]), event["Race"],
                               "Race", "missing_event", "completed event has no race rows")
                )

        dupes = long.duplicated(subset=["Year", "Round", "SessionKey", "DriverId"]).sum()
        if dupes:
            audit.gaps.append(
                SessionGap(audit.year, None, "-", "-", "duplicate",
                           f"{dupes} duplicate driver-session rows")
            )

    # --------------------------------------------------------------- public
    def load(
        self,
        start_year: int,
        end_year: Optional[int] = None,
        future_rows: FutureRows = "drivers",
        rebuild: bool = False,
        include_sprint: bool = True,
        strict: bool = False,
    ) -> F1Data:
        """Load every season in ``[start_year, end_year]`` inclusive.

        Parameters
        ----------
        start_year, end_year:
            Inclusive season range. ``end_year=None`` loads the single
            ``start_year`` season.
        future_rows:
            How the not-yet-run races of the final season appear in
            ``wide``. ``'drivers'`` emits one row per driver per remaining
            event, using the most recent completed race's lineup, with all
            result columns NaN. ``'events'`` emits one row per remaining
            event with no driver attached. ``'none'`` omits them.
            ``F1Data.future_events`` is populated either way.
        rebuild:
            Ignore the parquet snapshots and rebuild from the fastf1
            cache / network.
        include_sprint:
            Load sprint and sprint-qualifying sessions.
        strict:
            Raise if the completeness audit finds any gap.
        """
        end_year = start_year if end_year is None else end_year
        if end_year < start_year:
            raise ValueError(f"end_year ({end_year}) is before start_year ({start_year})")

        longs, event_tables, audits = [], [], []
        for year in range(start_year, end_year + 1):
            long, events, audit = self.load_season(
                year, rebuild=rebuild, include_sprint=include_sprint
            )
            audits.append(audit)
            if not long.empty:
                longs.append(long)
            if not events.empty:
                event_tables.append(events)

        long_all = (
            pd.concat(longs, ignore_index=True) if longs else pd.DataFrame(columns=_RESULT_COLS)
        )
        events_all = (
            pd.concat(event_tables, ignore_index=True)
            if event_tables
            else pd.DataFrame(columns=_EVENT_COLS)
        )

        wide = self._to_wide(long_all, events_all)
        future_events = events_all[events_all["IsFuture"]].copy() if len(events_all) else events_all

        if future_rows != "none" and len(future_events):
            wide = pd.concat([wide, self._future_frame(wide, future_events, future_rows)],
                             ignore_index=True)

        if len(wide):
            wide = wide.sort_values(["Year", "Round", "RacePosition", "QualiPosition"],
                                    na_position="last").reset_index(drop=True)

        schedule = pd.concat(
            [self.get_schedule(y).assign(Year=y) for y in range(start_year, end_year + 1)],
            ignore_index=True,
        )

        data = F1Data(
            wide=wide,
            long=long_all,
            schedule=schedule,
            events=events_all,
            future_events=future_events,
            audit=audits,
        )
        if strict and not data.complete:
            raise RuntimeError(
                "incomplete load:\n" + data.gap_frame().to_string(index=False)
            )
        return data

    # ----------------------------------------------------------- reshaping
    def _to_wide(self, long: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """One row per (Year, Round, DriverId) from the tidy frame."""
        if long.empty:
            return pd.DataFrame(columns=_EVENT_COLS + ["DriverId", "IsFuture"])

        long = self._backfill_ids(long)
        keys = ["Year", "Round", "DriverId"]

        # Driver/team identity: prefer the race entry, fall back to
        # qualifying then sprint (a driver can qualify and then not start).
        priority = {"Race": 0, "Qualifying": 1, "Sprint": 2, "SprintQualifying": 3}
        ident_cols = [c for c in _IDENTITY_COLS if c != "DriverId"]
        identity = (
            long.assign(_p=long["SessionKey"].map(priority))
            .sort_values(keys + ["_p"])
            .groupby(keys, as_index=False)[ident_cols]
            .first()
        )

        frames = {
            "Race": {
                "Position": "RacePosition",
                "ClassifiedPosition": "RaceClassifiedPosition",
                "GridPosition": "GridPosition",
                "Status": "RaceStatus",
                "Points": "RacePoints",
                "Laps": "RaceLaps",
                "Time": "RaceTime",
            },
            "Qualifying": {
                "Position": "QualiPosition",
                "Q1": "Q1",
                "Q2": "Q2",
                "Q3": "Q3",
            },
            "Sprint": {
                "Position": "SprintPosition",
                "ClassifiedPosition": "SprintClassifiedPosition",
                "GridPosition": "SprintGridPosition",
                "Status": "SprintStatus",
                "Points": "SprintPoints",
            },
        }

        wide = identity
        for session_key, mapping in frames.items():
            part = long[long["SessionKey"] == session_key]
            if part.empty:
                for out_col in mapping.values():
                    if out_col not in wide.columns:
                        wide[out_col] = pd.NA
                continue
            part = (
                part.drop_duplicates(subset=keys, keep="last")[keys + list(mapping)]
                .rename(columns=mapping)
            )
            wide = wide.merge(part, on=keys, how="outer")

        # Event identity and dates come from the schedule, so future
        # events and events with a failed session still get them.
        if len(events):
            wide = wide.merge(
                events[_EVENT_COLS + ["IsFuture"]], on=["Year", "Round"], how="left"
            )
        else:
            for col in _EVENT_COLS:
                if col not in wide.columns:
                    wide[col] = pd.NA
            wide["IsFuture"] = False

        wide["IsFuture"] = wide["IsFuture"].fillna(False).astype(bool)
        return self._derive(wide)

    @staticmethod
    def _backfill_ids(long: pd.DataFrame) -> pd.DataFrame:
        """Fill missing DriverId/TeamId from other sessions of the same event.

        The Ergast-backed fields go blank on some sessions (and Ergast has
        been frozen since 2024), which would otherwise collapse a whole
        session onto a single NaN key. Backfills by car number within the
        event, then by Abbreviation across the season.
        """
        long = long.copy()
        for id_col, keys in (("DriverId", ["DriverNumber"]), ("TeamId", ["DriverNumber"])):
            if long[id_col].isna().any():
                lookup = (
                    long.dropna(subset=[id_col])
                    .drop_duplicates(["Year", "Round"] + keys)
                    .set_index(["Year", "Round"] + keys)[id_col]
                )
                idx = pd.MultiIndex.from_frame(long[["Year", "Round"] + keys])
                long[id_col] = long[id_col].fillna(pd.Series(lookup.reindex(idx).values,
                                                            index=long.index))
        if long["DriverId"].isna().any():
            season_lookup = (
                long.dropna(subset=["DriverId", "Abbreviation"])
                .drop_duplicates(["Year", "Abbreviation"])
                .set_index(["Year", "Abbreviation"])["DriverId"]
            )
            idx = pd.MultiIndex.from_frame(long[["Year", "Abbreviation"]])
            long["DriverId"] = long["DriverId"].fillna(
                pd.Series(season_lookup.reindex(idx).values, index=long.index)
            )
        unresolved = long["DriverId"].isna().sum()
        if unresolved:
            logger.warning("%s session row(s) have no resolvable DriverId and were dropped",
                           unresolved)
            long = long.dropna(subset=["DriverId"])
        return long

    @staticmethod
    def _derive(wide: pd.DataFrame) -> pd.DataFrame:
        """Add the small conveniences every downstream notebook wants."""
        wide["DriverTeamId"] = wide["DriverId"].astype("string") + "-" + wide["TeamId"].astype("string")

        status = wide.get("RaceStatus")
        if status is not None:
            wide["Finished"] = (
                status.astype("string").str.strip().str.lower().eq("finished").astype("Int64")
            )
            wide.loc[status.isna(), "Finished"] = pd.NA

        for col, out in [("Q1", "Q1Sec"), ("Q2", "Q2Sec"), ("Q3", "Q3Sec")]:
            if col in wide.columns:
                wide[out] = pd.to_timedelta(wide[col], errors="coerce").dt.total_seconds()
        sec_cols = [c for c in ("Q1Sec", "Q2Sec", "Q3Sec") if c in wide.columns]
        if sec_cols:
            wide["BestQualiSec"] = wide[sec_cols].min(axis=1)

        wide["TotalPoints"] = wide[["RacePoints", "SprintPoints"]].sum(axis=1, min_count=1)
        wide["HasRaceResult"] = wide["RacePosition"].notna()
        # Only future rows carry a real value; keep the column either way so
        # the schema does not change with future_rows=.
        if "LineupSource" not in wide.columns:
            wide["LineupSource"] = pd.Series(pd.NA, index=wide.index, dtype="string")
        return wide

    def _future_frame(
        self,
        wide: pd.DataFrame,
        future_events: pd.DataFrame,
        mode: FutureRows,
    ) -> pd.DataFrame:
        """Rows for the remaining events of the season(s) loaded."""
        events = future_events[_EVENT_COLS + ["IsFuture"]].copy()
        events["IsFuture"] = True

        def blanked(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
            # Match the dtypes already in `wide` so the concat in load()
            # doesn't re-infer types off all-null columns.
            for col in cols:
                dtype = wide[col].dtype if col in wide.columns else "object"
                rows[col] = _na_series(rows.index, dtype)
            return rows

        if mode == "events":
            rows = blanked(events.reset_index(drop=True), _IDENTITY_COLS + _RESULT_OUT_COLS)
            return self._derive(rows).assign(LineupSource="none")

        lineup = self._latest_lineup(wide)
        if lineup.empty:
            logger.warning("no completed race found to source a lineup from — "
                           "falling back to event-level future rows")
            return self._future_frame(wide, future_events, "events")

        rows = events.merge(lineup, how="cross").reset_index(drop=True)
        return self._derive(blanked(rows, _RESULT_OUT_COLS))

    @staticmethod
    def _latest_lineup(wide: pd.DataFrame) -> pd.DataFrame:
        """Driver/team lineup from the most recent completed race.

        Falls back through earlier races (and earlier seasons) if the most
        recent one is unusable, so a pre-season load still gets a grid.
        """
        raced = wide[wide["HasRaceResult"] & ~wide["IsFuture"]]
        if raced.empty:
            return pd.DataFrame()
        cols = _IDENTITY_COLS
        for _, group in sorted(
            raced.groupby(["Year", "Round"]), key=lambda kv: kv[0], reverse=True
        ):
            lineup = group[cols].dropna(subset=["DriverId"]).drop_duplicates("DriverId")
            if len(lineup) >= 10:
                return lineup.assign(
                    LineupSource=f"{int(group['Year'].iloc[0])} R{int(group['Round'].iloc[0])}"
                ).reset_index(drop=True)
        return pd.DataFrame()

    # ---------------------------------------------------------- prefetch
    def prefetch(
        self,
        years: Iterable[int],
        include_sprint: bool = True,
        results_only: bool = True,
    ) -> pd.DataFrame:
        """Download and cache every completed session for ``years``.

        ``results_only=False`` also pulls laps/telemetry/weather/messages
        (large and slow — only needed for lap-level work).
        """
        kwargs = RESULTS_ONLY_KWARGS if results_only else dict(
            laps=True, telemetry=True, weather=True, messages=True
        )
        report = []
        for year in years:
            events = self._event_table(year)
            if events.empty:
                report.append({"Year": year, "Ok": 0, "Failed": 0, "Note": "no schedule"})
                continue
            expected = self._expected_sessions(events, include_sprint=include_sprint)
            ok = failed = 0
            for _, row in expected.iterrows():
                try:
                    with self._online():
                        session = fastf1.get_session(
                            year, int(row["Round"]), _SESSION_IDENTIFIER[row["SessionKey"]]
                        )
                        session.load(**kwargs)
                    ok += 1
                except Exception as exc:
                    failed += 1
                    logger.info("prefetch %s R%s %s: %s",
                                year, row["Round"], row["SessionKey"], exc)
            report.append({"Year": year, "Ok": ok, "Failed": failed, "Note": ""})
            print(f"{year}: cached {ok}/{ok + failed} sessions")
        return pd.DataFrame(report)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_SESSION_IDENTIFIER = {
    "Race": "R",
    "Qualifying": "Q",
    # 'S' resolves correctly in every era: fastf1 maps it to the 2021-22
    # 'Sprint Qualifying' name and the later 'Sprint' name alike.
    "Sprint": "S",
}

_IDENTITY_COLS = [
    "DriverId",
    "DriverNumber",
    "Abbreviation",
    "FullName",
    "TeamId",
    "TeamName",
]

# Result columns on the wide frame, blanked out for future races.
_RESULT_OUT_COLS = [
    "RacePosition",
    "RaceClassifiedPosition",
    "GridPosition",
    "RaceStatus",
    "RacePoints",
    "RaceLaps",
    "RaceTime",
    "QualiPosition",
    "Q1",
    "Q2",
    "Q3",
    "SprintPosition",
    "SprintClassifiedPosition",
    "SprintGridPosition",
    "SprintStatus",
    "SprintPoints",
]

# Values the notebook's existing feature functions filter on.
_LEGACY_EVENT = {
    "Race": "Race",
    "Qualifying": "Qualifying",
    "Sprint": "Sprint",
}

_SHORT = {
    "Race": "Race",
    "Qualifying": "Quali",
    "Sprint": "Sprint",
    "SprintQualifying": "SprintQuali",
}


def _short(session_key: str) -> str:
    """Column-name stem used in the event table ('Quali' not 'Qualifying')."""
    return _SHORT[session_key]


# fastf1 signals "unknown" with '' on some fields and a float NaN on others;
# a mixed object column then round-trips through parquet as the literal
# string 'nan'. Everything here has to collapse to a real null, or a whole
# session can end up keyed on one bogus driver.
_NULLISH = ["", "nan", "NaN", "None", "<NA>", "NaT"]
_TEXT_COLS = ("DriverId", "TeamId", "TeamName", "Abbreviation", "FullName",
              "DriverNumber", "Status", "ClassifiedPosition")


def _normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Null out placeholder strings and give text columns a nullable dtype."""
    for col in _TEXT_COLS:
        if col in frame.columns:
            frame[col] = (
                frame[col].astype("object").replace(_NULLISH, None).astype("string")
            )
    return frame


def _na_series(index: pd.Index, dtype) -> pd.Series:
    """An all-null Series of ``dtype`` — keeps concat from re-inferring types."""
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return pd.Series(pd.NaT, index=index, dtype=dtype)
    try:
        return pd.Series(pd.NA, index=index, dtype=dtype)
    except (TypeError, ValueError):
        return pd.Series(float("nan"), index=index, dtype="float64")


def load_seasons(
    start_year: int,
    end_year: Optional[int] = None,
    future_rows: FutureRows = "drivers",
    **kwargs,
) -> F1Data:
    """One-liner wrapper around :class:`F1SeasonLoader`."""
    loader_kwargs = {
        k: kwargs.pop(k)
        for k in ("cache_dir", "snapshot_dir", "min_drivers", "recheck_days", "quiet")
        if k in kwargs
    }
    return F1SeasonLoader(**loader_kwargs).load(
        start_year, end_year, future_rows=future_rows, **kwargs
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load F1 season data from the fastf1 cache.")
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int, nargs="?")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--future-rows", default="drivers",
                        choices=["drivers", "events", "none"])
    parser.add_argument("--out", help="optional parquet path for the wide frame")
    args = parser.parse_args()

    result = load_seasons(
        args.start_year, args.end_year,
        future_rows=args.future_rows, rebuild=args.rebuild,
    )
    result.summary()
    print(f"\nwide: {result.wide.shape}   long: {result.long.shape}")
    if args.out:
        result.wide.to_parquet(args.out, index=False)
        print(f"wrote {args.out}")
