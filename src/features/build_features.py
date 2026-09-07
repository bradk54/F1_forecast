"""Leakage-safe history features for retirement prediction.

Every feature here answers one question: *what did we know before the lights
went out?*  A rolling DNF rate that includes the race being predicted is not a
feature, it is the answer, and a model trained on one will look excellent in
backtest and fail on Sunday.

Two rules enforce that, and every builder in this module obeys both:

1. **Shift before you aggregate.**  Order rows by race date within the group,
   drop the current row, and only then take the mean, sum or count.  Never
   ``cumsum()`` without a shift — that is the single most common leak, and one
   the existing notebook's ``Points_cumsum_season_team`` column still has.
2. **Aggregate at the level you shift at.**  A circuit's historical attrition
   is computed from race-level totals, shifted by race, then joined back to
   drivers.  Shifting per driver-row would let one driver's retirement inform
   their team mate's feature in the same race.

:func:`detect_target_leakage` turns rule 1 into a test rather than a promise:
it rebuilds the feature table with one race's outcomes flipped and asserts that
nothing at or before that race moved.

The convention throughout is ``<entity>_<quantity>_<window>``, where the window
is ``5``/``10`` for a rolling count of races, ``career`` for expanding over all
prior races, and ``season`` for expanding within the current season.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from src import config
from src.features.labels import COLLISION, DRIVER_ERROR, MECHANICAL

log = logging.getLogger(__name__)

#: Columns identifying a race.  ``session_type`` is part of the key whenever it
#: is present: a sprint and a grand prix share a round number but are separate
#: events, and treating them as one makes the driver-race key non-unique.
RACE_KEYS = ("Year", "RoundNumber")
SESSION_COL = "session_type"
#: Column used to order history.
ORDER_COL = "RaceDate"

#: Sort rank within a weekend, so a Saturday sprint precedes a Sunday race even
#: when both carry the same event date.
SESSION_ORDER = {"S": 0, "SQ": 0, "R": 1}


def race_keys(frame: pd.DataFrame) -> list[str]:
    """Columns that uniquely identify one race in ``frame``."""
    keys = [c for c in RACE_KEYS if c in frame.columns]
    if SESSION_COL in frame.columns:
        keys.append(SESSION_COL)
    return keys


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    """Chronological order, with a stable tiebreak so results reproduce.

    A sprint and a grand prix can carry the same event date, so session rank
    breaks that tie in running order rather than leaving it to sort stability.
    """
    working = frame
    keys = [c for c in (ORDER_COL, *RACE_KEYS) if c in frame.columns]
    if SESSION_COL in frame.columns:
        working = frame.assign(
            _session_rank=frame[SESSION_COL].map(SESSION_ORDER).fillna(9)
        )
        keys = [c for c in (ORDER_COL, "Year", "RoundNumber", "_session_rank")
                if c in working.columns]
        return working.sort_values(keys, kind="mergesort").drop(columns="_session_rank")
    return working.sort_values(keys, kind="mergesort")


def prior_rolling(
    frame: pd.DataFrame,
    group: str | Sequence[str],
    value: str,
    window: int,
    *,
    stat: str = "mean",
    min_periods: int = 1,
) -> pd.Series:
    """Rolling statistic over the ``window`` races *before* each row.

    The ``shift(1)`` is what makes it a feature rather than an answer.  Note
    that ``min_periods=1`` means an entity's second race gets a one-race
    average; combine with the matching ``*_races_to_date`` count so the model
    can tell a confident estimate from a noisy one.
    """
    ordered = _sorted(frame)
    grouped = ordered.groupby(list(np.atleast_1d(group)), observed=True)[value]
    rolled = grouped.transform(
        lambda s: getattr(s.shift(1).rolling(window, min_periods=min_periods), stat)()
    )
    return rolled.reindex(frame.index)


def prior_expanding(
    frame: pd.DataFrame,
    group: str | Sequence[str],
    value: str,
    *,
    stat: str = "mean",
) -> pd.Series:
    """Expanding statistic over every race *before* each row."""
    ordered = _sorted(frame)
    grouped = ordered.groupby(list(np.atleast_1d(group)), observed=True)[value]
    expanded = grouped.transform(
        lambda s: getattr(s.shift(1).expanding(min_periods=1), stat)()
    )
    return expanded.reindex(frame.index)


def prior_count(frame: pd.DataFrame, group: str | Sequence[str]) -> pd.Series:
    """Number of prior appearances for each row's group."""
    ordered = _sorted(frame)
    counted = ordered.groupby(list(np.atleast_1d(group)), observed=True).cumcount()
    return counted.reindex(frame.index).astype("float64")


def races_since(frame: pd.DataFrame, group: str, flag: str) -> pd.Series:
    """Races since ``flag`` was last 1, counting only prior races.

    Returns NaN until the flag has been seen once, so "never retired" is
    distinguishable from "retired last time out".
    """
    ordered = _sorted(frame)
    out = pd.Series(np.nan, index=ordered.index, dtype="float64")
    for _, idx in ordered.groupby(group, observed=True).groups.items():
        flags = ordered.loc[idx, flag].to_numpy()
        gap, running = np.full(len(flags), np.nan), np.nan
        for i in range(len(flags)):
            gap[i] = running  # value *before* seeing this race
            if flags[i] == 1:
                running = 0.0
            elif not np.isnan(running):
                running += 1.0
        out.loc[idx] = gap
    return out.reindex(frame.index)


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #


def add_driver_history(
    frame: pd.DataFrame, *, driver_col: str = "DriverId"
) -> pd.DataFrame:
    """Per-driver reliability, experience and form, from prior races only."""
    out = frame.copy()
    out["driver_races_to_date"] = prior_count(out, driver_col)

    for window in (5, 10):
        out[f"driver_dnf_rate_{window}"] = prior_rolling(out, driver_col, "dnf", window)
        out[f"driver_mech_dnf_rate_{window}"] = prior_rolling(
            out, driver_col, "dnf_mechanical", window
        )
        out[f"driver_incident_dnf_rate_{window}"] = prior_rolling(
            out, driver_col, "dnf_incident", window
        )

    out["driver_dnf_rate_career"] = prior_expanding(out, driver_col, "dnf")
    out["driver_races_since_dnf"] = races_since(out, driver_col, "dnf")

    # Form.  Points rate is mostly a proxy for car quality, which matters here
    # because the back of the grid both breaks more and gets collected more.
    if "Points" in out.columns:
        out["driver_points_rate_5"] = prior_rolling(out, driver_col, "Points", 5)
    if "GridPosition" in out.columns:
        out["driver_avg_grid_5"] = prior_rolling(out, driver_col, "GridPosition", 5)
    return out


def add_team_history(
    frame: pd.DataFrame,
    *,
    team_col: str = "TeamId",
    driver_col: str = "DriverId",
) -> pd.DataFrame:
    """Per-team reliability, including the team mate's separate record.

    A team's cars share a power unit, a gearbox and a design office, so the
    other car's recent failures carry information about this one that the
    driver's own record does not.  The team mate feature is built from races
    strictly before the current one, so it never peeks at what happened to the
    other car today.
    """
    out = frame.copy()
    keys = race_keys(out)

    # Team-level history is computed at the team-race level, then joined back.
    # Averaging the driver rows directly would let a driver's own retirement
    # leak into their team mate's feature through the shared team mean.
    team_race = (
        _sorted(out)
        .groupby([team_col, *keys, ORDER_COL], observed=True)
        .agg(
            team_dnf_count=("dnf", "sum"),
            team_cars=("dnf", "size"),
            team_mech_count=("dnf_mechanical", "sum"),
        )
        .reset_index()
    )
    team_race["team_dnf_share"] = team_race["team_dnf_count"] / team_race["team_cars"]
    team_race["team_mech_share"] = team_race["team_mech_count"] / team_race["team_cars"]

    for window in (5, 10):
        team_race[f"team_dnf_rate_{window}"] = prior_rolling(
            team_race, team_col, "team_dnf_share", window
        )
        team_race[f"team_mech_dnf_rate_{window}"] = prior_rolling(
            team_race, team_col, "team_mech_share", window
        )
    team_race["team_dnf_rate_career"] = prior_expanding(
        team_race, team_col, "team_dnf_share"
    )
    team_race["team_races_to_date"] = prior_count(team_race, team_col)
    # Races the team has run this season: a proxy for how far a new car is
    # through its shakedown, when reliability is at its worst.
    team_race["team_races_this_season"] = prior_count(team_race, [team_col, "Year"])

    # Only the shifted columns may travel back to the driver rows.
    # ``team_dnf_share`` and ``team_mech_share`` describe *this* race and exist
    # solely as inputs to the rolling windows above; joining them back would
    # hand the model the answer, which is precisely what
    # :func:`detect_target_leakage` is built to catch.
    CURRENT_RACE_INTERMEDIATES = {
        "team_dnf_count",
        "team_mech_count",
        "team_dnf_share",
        "team_mech_share",
    }
    join_cols = [
        c
        for c in team_race.columns
        if c.startswith("team_") and c not in CURRENT_RACE_INTERMEDIATES
    ]
    out = out.merge(
        team_race[[team_col, *keys, *join_cols]],
        on=[team_col, *keys],
        how="left",
        validate="many_to_one",
    )

    # Team mate: the same team's other driver, their own prior-race record.
    mate = out[[*keys, team_col, driver_col, "driver_dnf_rate_10"]].copy()
    merged = out[[*keys, team_col, driver_col]].merge(
        mate.rename(
            columns={
                driver_col: "_mate_id",
                "driver_dnf_rate_10": "teammate_dnf_rate_10",
            }
        ),
        on=[*keys, team_col],
        how="left",
    )
    merged = merged.loc[merged[driver_col] != merged["_mate_id"]]
    mate_feature = merged.groupby([*keys, team_col, driver_col], observed=True)[
        "teammate_dnf_rate_10"
    ].mean()
    out = out.merge(
        mate_feature.reset_index(),
        on=[*keys, team_col, driver_col],
        how="left",
        validate="one_to_one",
    )
    return out


def add_team_performance(
    frame: pd.DataFrame,
    *,
    team_col: str = "TeamId",
    driver_col: str = "DriverId",
) -> pd.DataFrame:
    """How well the team is *performing*, as distinct from how often it breaks.

    :func:`add_team_history` measures a team's reliability.  This measures its
    competitiveness, which is a different axis and moves retirement risk in its
    own right: a slow car spends the race being lapped and passed, which is
    where collisions happen, while a fast car runs in clean air.  Grid and
    points also track development within a season -- an upgrade that works
    shows up here weeks before it shows up in reliability.

    Built at the team-race level and joined back, for the same reason as
    :func:`add_team_history`: averaging driver rows directly would let a
    driver's own result reach their team mate's feature through the shared mean.
    """
    out = frame.copy()
    keys = race_keys(out)

    agg: dict[str, tuple[str, str]] = {}
    if "Points" in out.columns:
        agg["team_points_race"] = ("Points", "sum")
    if "Position" in out.columns:
        # Mean over finishers only: ``Position`` is NaN for a retirement, so a
        # team that lost both cars contributes NaN rather than a fabricated
        # value.  Conditioning on finishing is deliberate -- the question is
        # "when this car gets to the end, where does it end up".
        agg["team_finish_race"] = ("Position", "mean")
        agg["team_best_finish_race"] = ("Position", "min")
    if "GridPosition" in out.columns:
        agg["team_grid_race"] = ("GridPosition", "mean")
    if not agg:
        return out

    team_race = (
        _sorted(out)
        .groupby([team_col, *keys, ORDER_COL], observed=True)
        .agg(**agg)
        .reset_index()
    )

    # Every column below is a rolling window over races strictly before the
    # current one; the per-race aggregates above never travel back.
    if "team_points_race" in team_race.columns:
        for window in (5, 10):
            team_race[f"team_points_rate_{window}"] = prior_rolling(
                team_race, team_col, "team_points_race", window
            )
        team_race["team_points_rate_career"] = prior_expanding(
            team_race, team_col, "team_points_race"
        )
        # Positive means the team is scoring above its own historical rate:
        # a development curve, which a level rate cannot express.
        team_race["team_form_delta"] = (
            team_race["team_points_rate_5"] - team_race["team_points_rate_career"]
        )
    if "team_finish_race" in team_race.columns:
        team_race["team_avg_finish_5"] = prior_rolling(
            team_race, team_col, "team_finish_race", 5
        )
        team_race["team_best_finish_5"] = prior_rolling(
            team_race, team_col, "team_best_finish_race", 5
        )
    if "team_grid_race" in team_race.columns:
        for window in (5, 10):
            team_race[f"team_avg_grid_{window}"] = prior_rolling(
                team_race, team_col, "team_grid_race", window
            )

    current_race_intermediates = set(agg)
    join_cols = [
        c
        for c in team_race.columns
        if c.startswith("team_") and c not in current_race_intermediates
    ]
    return out.merge(
        team_race[[team_col, *keys, *join_cols]],
        on=[team_col, *keys],
        how="left",
        validate="many_to_one",
    )


def add_teammate_comparison(
    frame: pd.DataFrame,
    *,
    team_col: str = "TeamId",
    driver_col: str = "DriverId",
) -> pd.DataFrame:
    """Where this driver qualifies relative to the other car.

    The team mate is the only genuine control for car quality in the sport: two
    drivers, the same machinery, the same weekend.  A rolling qualifying delta
    is therefore much closer to a measure of the driver than raw grid position,
    which mostly measures the car.

    Both columns are known once the grid is set, so they are ``post_quali``
    features.  ``teammate_grid_delta`` describes the current race and is legal
    precisely because grid position is settled before the race runs.
    """
    out = frame.copy()
    keys = race_keys(out)
    if "GridPosition" not in out.columns:
        return out

    grid = pd.to_numeric(out["GridPosition"], errors="coerce")
    working = out[[*keys, team_col, driver_col]].assign(_grid=grid.to_numpy())

    # The other car's grid slot, matched within team and race.
    pairs = working.merge(
        working.rename(columns={driver_col: "_mate", "_grid": "_mate_grid"}),
        on=[*keys, team_col],
        how="left",
    )
    pairs = pairs.loc[pairs[driver_col] != pairs["_mate"]]
    mate_grid = (
        pairs.groupby([*keys, team_col, driver_col], observed=True)["_mate_grid"]
        .mean()
        .reset_index()
    )
    out = out.merge(
        mate_grid, on=[*keys, team_col, driver_col], how="left", validate="one_to_one"
    )

    # Negative is better: this driver started ahead of the other car.
    out["teammate_grid_delta"] = grid - out["_mate_grid"]
    out = out.drop(columns="_mate_grid")
    out["driver_grid_vs_teammate_5"] = prior_rolling(
        out, driver_col, "teammate_grid_delta", 5
    )
    return out


def add_field_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Properties of the whole grid, assembled from features already built.

    Attrition is partly a property of the field rather than of any one car: a
    grid of fragile cars retires more, and a grid where everyone is on similar
    pace produces more close racing and so more contact.  Both are computable
    before the race because every input is a prior-race rolling feature.

    Depends on :func:`add_driver_history` and :func:`add_team_performance`
    having run first.
    """
    out = frame.copy()
    keys = race_keys(out)

    if "driver_dnf_rate_10" in out.columns:
        # How fragile this particular field is, on recent form.
        out["field_dnf_rate_mean"] = out.groupby(keys, observed=True)[
            "driver_dnf_rate_10"
        ].transform("mean")
        # Where this driver sits against the field they are actually racing:
        # a 20% rate means something different on a fragile grid.
        out["driver_dnf_rate_vs_field"] = (
            out["driver_dnf_rate_10"] - out["field_dnf_rate_mean"]
        )

    if "team_points_rate_5" in out.columns:
        # Spread of car quality across the grid.  A compressed field races
        # closer together, which is where contact comes from.
        out["field_pace_spread"] = out.groupby(keys, observed=True)[
            "team_points_rate_5"
        ].transform("std")
        rank = out.groupby(keys, observed=True)["team_points_rate_5"].rank(
            pct=True, ascending=False
        )
        out["team_rank_in_field"] = rank

    if "RoundNumber" in out.columns and "Year" in out.columns:
        rounds = out.groupby("Year", observed=True)["RoundNumber"].transform("max")
        # Late-season cars are developed and well understood; early-season ones
        # are neither.  Expressed as a fraction so seasons of different length
        # are comparable.
        out["season_progress"] = out["RoundNumber"] / rounds
    return out


def add_pairing_history(
    frame: pd.DataFrame,
    *,
    driver_col: str = "DriverId",
    team_col: str = "TeamId",
    settling_races: int = 5,
) -> pd.DataFrame:
    """How long this driver and this car have been working together."""
    out = frame.copy()
    out["pair_id"] = out[driver_col].astype(str) + "__" + out[team_col].astype(str)
    out["pair_races_to_date"] = prior_count(out, "pair_id")
    out["is_new_pairing"] = (
        out["pair_races_to_date"] < settling_races
    ).astype("int8")
    out["pair_dnf_rate_10"] = prior_rolling(out, "pair_id", "dnf", 10)
    return out


def add_circuit_history(
    frame: pd.DataFrame,
    *,
    circuit_col: str = "circuit_key",
    driver_col: str = "DriverId",
) -> pd.DataFrame:
    """Attrition history of the venue, and each driver's record at it.

    The circuit rate is built from race-level totals so that no driver's
    retirement today can inform anybody's feature today.
    """
    out = frame.copy()
    keys = race_keys(out)

    circuit_race = (
        _sorted(out)
        .groupby([circuit_col, *keys, ORDER_COL], observed=True)
        .agg(race_dnf_share=("dnf", "mean"), race_starters=("dnf", "size"))
        .reset_index()
    )
    circuit_race["circuit_dnf_rate_prior"] = prior_expanding(
        circuit_race, circuit_col, "race_dnf_share"
    )
    circuit_race["circuit_dnf_rate_3"] = prior_rolling(
        circuit_race, circuit_col, "race_dnf_share", 3
    )
    circuit_race["circuit_races_prior"] = prior_count(circuit_race, circuit_col)

    out = out.merge(
        circuit_race[
            [
                circuit_col,
                *keys,
                "circuit_dnf_rate_prior",
                "circuit_dnf_rate_3",
                "circuit_races_prior",
            ]
        ],
        on=[circuit_col, *keys],
        how="left",
        validate="many_to_one",
    )

    out["driver_dnf_rate_at_circuit"] = prior_expanding(
        out, [driver_col, circuit_col], "dnf"
    )
    out["driver_starts_at_circuit"] = prior_count(out, [driver_col, circuit_col])
    return out


def add_race_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Facts about the race itself that are known before it starts."""
    out = frame.copy()

    out["season_round"] = out["RoundNumber"].astype("float64")
    out["is_season_opener"] = (out["RoundNumber"] == 1).astype("int8")
    out["regulation_era"] = out["Year"].map(config.regulation_era).astype("object")

    field = out.groupby(race_keys(out), observed=True)["dnf"].transform("size")
    out["field_size"] = field.astype("float64")

    if "GridPosition" in out.columns:
        grid = pd.to_numeric(out["GridPosition"], errors="coerce")
        # A grid slot of 0 is FastF1's code for a pit-lane start.  That is a
        # real and distinct starting condition, not a missing value, so it gets
        # its own flag and is moved to the back for the numeric feature.
        out["starts_from_pit_lane"] = (grid == 0).astype("int8")
        grid = grid.where(grid > 0, out["field_size"])
        out["grid_position"] = grid
        out["grid_position_pct"] = grid / out["field_size"]
        out["is_back_half_of_grid"] = (out["grid_position_pct"] > 0.5).astype("int8")
        if "QualifyingPosition" in out.columns:
            # Grid penalties: qualified well, started badly.  Only built when
            # qualifying classification is present, so the column is never a
            # column of NaNs pretending to be a feature.
            quali = pd.to_numeric(out["QualifyingPosition"], errors="coerce")
            out["grid_penalty_places"] = (grid - quali).clip(lower=0)

    # Gap since the entity last raced: a long break means a rebuilt car and a
    # rusty driver, both of which move retirement risk.
    ordered = _sorted(out)
    gap = ordered.groupby("DriverId", observed=True)[ORDER_COL].transform(
        lambda s: s.diff().dt.days
    )
    out["days_since_last_race"] = gap.reindex(out.index)
    return out


def add_cause_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Split the target into the two causes that behave differently.

    Mechanical failures track the car and the circuit's power-unit and brake
    load; incidents track the driver, the grid slot and how much room the
    circuit leaves.  Keeping both lets the rolling builders above produce
    cause-specific rates.
    """
    out = frame.copy()
    if "dnf_cause" not in out.columns:
        raise KeyError("run labels.add_race_outcome_labels first")
    cause = out["dnf_cause"].astype("object")
    out["dnf_mechanical"] = (
        (cause == MECHANICAL) & (out["dnf"] == 1)
    ).astype("int8")
    out["dnf_incident"] = (
        cause.isin([COLLISION, DRIVER_ERROR]) & (out["dnf"] == 1)
    ).astype("int8")
    # The remainder, and it is not small: roughly a third of retirements in a
    # 2018-2025 pull carry the bare status ``Retired``, which states that the
    # car stopped and nothing else.  Naming the bucket keeps it visible --
    # mechanical and incident together do not add up to ``dnf``, and a
    # cause-specific model that ignores this would be silently modelling two
    # thirds of the target.
    out["dnf_other"] = (
        (out["dnf"] == 1)
        & (out["dnf_mechanical"] == 0)
        & (out["dnf_incident"] == 0)
    ).astype("int8")
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build_history_features(
    results: pd.DataFrame,
    *,
    driver_col: str = "DriverId",
    team_col: str = "TeamId",
    circuit_col: str = "circuit_key",
) -> pd.DataFrame:
    """Run every history builder in dependency order.

    Args:
        results: Labelled race results, one row per driver per race, carrying
            ``dnf``, ``dnf_cause``, ``Year``, ``RoundNumber`` and ``RaceDate``.
            Rows for drivers who never started should already be removed —
            predicting a non-start is a different problem with different
            features.

    Returns:
        The input frame with every history feature appended, in the original
        row order.
    """
    required = {"dnf", "dnf_cause", "RaceDate", *RACE_KEYS, driver_col, team_col}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    frame = results.reset_index(drop=True).copy()
    frame[ORDER_COL] = pd.to_datetime(frame[ORDER_COL])

    frame = add_cause_indicators(frame)
    frame = add_driver_history(frame, driver_col=driver_col)
    frame = add_team_history(frame, team_col=team_col, driver_col=driver_col)
    frame = add_team_performance(frame, team_col=team_col, driver_col=driver_col)
    frame = add_pairing_history(frame, driver_col=driver_col, team_col=team_col)
    if circuit_col in frame.columns:
        frame = add_circuit_history(
            frame, circuit_col=circuit_col, driver_col=driver_col
        )
    frame = add_race_context(frame)
    frame = add_teammate_comparison(frame, team_col=team_col, driver_col=driver_col)
    # Last: reads the rolling columns the builders above produced.
    frame = add_field_context(frame)
    return frame


# --------------------------------------------------------------------------- #
# Leakage detection
# --------------------------------------------------------------------------- #

#: Columns that legitimately describe the current race's outcome and are
#: therefore expected to change when outcomes change.
OUTCOME_COLUMNS = frozenset(
    {
        "dnf", "dnf_cause", "dnf_strict", "dnf_classified", "finished_on_track",
        "classified", "classification_code", "started", "Status", "Position",
        "ClassifiedPosition", "Points", "Laps", "dnf_mechanical", "dnf_incident",
        "dnf_other",
    }
)


def event_ordinal(frame: pd.DataFrame) -> pd.Series:
    """Rank every row by which event it belongs to, earliest first.

    Ordering by ``RaceDate`` alone is not enough on a sprint weekend, where
    Saturday's sprint and Sunday's grand prix are two events that may share an
    event date.  The sprint genuinely precedes the race, and a feature on the
    race row is entitled to use it, so the two need distinct ranks.
    """
    keys = [c for c in ("RaceDate", "Year", "RoundNumber") if c in frame.columns]
    working = frame[keys].copy()
    if SESSION_COL in frame.columns:
        working["_session_rank"] = frame[SESSION_COL].map(SESSION_ORDER).fillna(9)
    unique = (
        working.drop_duplicates()
        .sort_values(list(working.columns), kind="mergesort")
        .reset_index(drop=True)
    )
    unique["_ordinal"] = np.arange(len(unique), dtype="int64")
    merged = working.merge(unique, on=list(working.columns), how="left")
    merged.index = frame.index
    return merged["_ordinal"]


def detect_target_leakage(
    results: pd.DataFrame,
    *,
    builder: Callable[[pd.DataFrame], pd.DataFrame] = build_history_features,
    flip_round: int | None = None,
    flip_year: int | None = None,
    flip_session: str = "R",
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Find features that respond to the outcome of the race they describe.

    The test is direct: flip every driver's outcome in one event, rebuild the
    features, and compare.  A feature computed only from prior events cannot
    move for that event or any earlier one.  Anything that does move is reading
    the answer.

    Exactly one *event* is flipped, not one round.  On a sprint weekend a round
    holds two events, and the Sunday race is entitled to learn from Saturday's
    sprint; flipping both and comparing on date alone would report that
    legitimate dependency as a leak.

    Args:
        results: Labelled results, as passed to :func:`build_history_features`.
        builder: The feature builder under test.
        flip_round, flip_year: Which event to corrupt.  Defaults to the middle
            round of the last season present, so there is history on both sides.
        flip_session: Which session of that round to corrupt, when the frame
            distinguishes them.
        tolerance: Absolute difference treated as equal, for float noise.

    Returns:
        One row per leaking feature with the number of affected rows and the
        largest absolute change.  An empty frame means no leakage was found.
    """
    frame = results.reset_index(drop=True).copy()
    frame["RaceDate"] = pd.to_datetime(frame["RaceDate"])

    if flip_year is None:
        flip_year = int(frame["Year"].max())
    season = frame.loc[frame["Year"] == flip_year, "RoundNumber"]
    if season.empty:
        raise ValueError(f"no rows for year={flip_year}")
    if flip_round is None:
        flip_round = int(season.median())

    mask = (frame["Year"] == flip_year) & (frame["RoundNumber"] == flip_round)
    if SESSION_COL in frame.columns:
        session_mask = mask & (frame[SESSION_COL] == flip_session)
        # Fall back to the whole round if that session is not present, which is
        # the common case: most rounds have no sprint.
        if session_mask.any():
            mask = session_mask
    if not mask.any():
        raise ValueError(f"no rows for year={flip_year} round={flip_round}")

    corrupted = frame.copy()
    corrupted.loc[mask, "dnf"] = 1 - corrupted.loc[mask, "dnf"]
    corrupted.loc[mask, "dnf_cause"] = np.where(
        corrupted.loc[mask, "dnf"] == 1, MECHANICAL, "finished"
    )

    baseline = builder(frame)
    altered = builder(corrupted)

    # Compare everything strictly before the flipped event, plus the flipped
    # event itself.  Later events may legitimately move.
    ordinals = event_ordinal(baseline)
    flip_ordinal = int(event_ordinal(frame)[mask].max())
    at_or_before = ordinals <= flip_ordinal

    rows = []
    for column in baseline.columns:
        if column in OUTCOME_COLUMNS or column not in altered.columns:
            continue
        left = baseline.loc[at_or_before, column]
        right = altered.loc[at_or_before, column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            diff = (left.fillna(-9e18) - right.fillna(-9e18)).abs()
            changed = diff > tolerance
            worst = float(diff[changed].max()) if changed.any() else 0.0
        else:
            changed = left.astype("object").ne(right.astype("object")) & ~(
                left.isna() & right.isna()
            )
            worst = float("nan")
        if changed.any():
            rows.append(
                {
                    "feature": column,
                    "rows_changed": int(changed.sum()),
                    "max_abs_change": worst,
                }
            )

    report = pd.DataFrame(rows, columns=["feature", "rows_changed", "max_abs_change"])
    return report.sort_values("rows_changed", ascending=False, ignore_index=True)
