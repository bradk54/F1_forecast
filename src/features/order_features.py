"""Leakage-safe features for the finishing-order model.

The retirement features in :mod:`src.features.build_features` answer "will this
car stop".  These answer a different question -- *if it does not stop, where
does it finish* -- and the evidence about what drives that is different enough
that the features are built differently.  Each choice below rests on a
measurement from this dataset (2018-2026), not on intuition:

**Car pace is the dominant term, and it moves slowly.**  Team-within-race means
explain 82% of the variance in finishing percentile, and team-season means
explain 58%.  A team's race-to-race grid percentile has an autocorrelation of
0.69 at lag 1 and still 0.67 at lag 10 within a season: most race-to-race
movement is noise around a stable level, so pace wants a *long* average.  That
is the opposite of attrition, which the DNF work found to shift fast, and it is
why every window here is an exponentially weighted mean with a tunable
half-life rather than the fixed 5-race windows the DNF features use.

**Pace survives the winter.**  Season-mean team grid percentile correlates with
the previous season's at 0.79-0.95, including across the 2022 and 2026
regulation breaks.  So nothing here resets at the season boundary (with one
deliberate exception kept as a contrast), and team history follows the
organisation rather than the entry name -- see :mod:`src.data.teams`.

**Driver skill is only visible against the team-mate.**  Raw position is mostly
the car.  The within-team qualifying delta persists at r = 0.61 between halves
of a season, so it measures something stable about the driver.

**Some cars race better than they qualify.**  Places gained from the grid,
residualised on grid slot, persist at r = 0.46 between season halves.  Without
the residualisation it is 0.15, because a pole-sitter can only lose places and a
back-marker can only gain them.

**Grid matters more at some circuits than others.**  Mean Spearman correlation
of grid and finish runs from 0.63 (Interlagos) to 0.88 (Suzuka), but it is only
moderately persistent (r = 0.36 between alternate visits), so the circuit
estimate is shrunk heavily toward the field mean.

One property of the model these feed shapes the feature set: in a Plackett-Luce
model **anything constant within a race cancels out**.  The probability of an
order depends only on differences in strength between the cars in that race, so
a circuit feature, a weather feature or a calendar feature has no effect on its
own.  They can only matter as interactions with something that varies between
drivers -- which is why the circuit and rain features here are built as
``grid x circuit`` and ``grid x rain`` terms, and why ``circuit_grid_retention``
is registered but never enters a model alone.

The same two rules as the DNF builders apply, and :func:`detect_order_leakage`
enforces them:

1. Shift before you aggregate -- every feature reads prior races only.
2. Aggregate at the level you shift at -- team history is built from team-race
   means and joined back, so one car's result cannot reach its team-mate's
   feature for the same race.

Only grands prix feed this history.  The DNF features also read sprints, which
on a sprint weekend puts Saturday's sprint into a ``pre_weekend`` feature that
was not knowable on the Monday.  For a retirement model that is a small
blemish; for a pace model it would quietly hand the pre-weekend forecast a
same-weekend race result, so sprint rows are excluded here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.teams import add_team_lineage
from src.features.build_features import (
    ORDER_COL,
    RACE_KEYS,
    event_ordinal,
    prior_ewma,
    prior_expanding,
    prior_rolling,
)

log = logging.getLogger(__name__)

#: How far the finishing order regresses toward the middle of the field from
#: the grid.  The mean per-race Spearman correlation of grid and finish among
#: classified finishers is 0.74 over 2018-2026 (0.68-0.80 by season).  Used
#: only to define what "gaining places" means net of where a car started; it
#: is a fixed structural constant, not something re-estimated per race, so it
#: carries no race's outcome into another race's feature.
GRID_REGRESSION_SLOPE = 0.75

#: Prior for circuit grid retention before any race has been seen.  Only the
#: first race of the dataset ever uses it.
DEFAULT_RETENTION_PRIOR = 0.74


@dataclass(frozen=True)
class OrderFeatureConfig:
    """The tunable knobs of the order-feature build.

    These are hyperparameters, not constants: :mod:`src.models.tuning` searches
    them in the same nested walk-forward that selects the features, and the
    defaults below are what that search chose.  They live here, rather than
    frozen into a stored parquet, precisely so they can be searched.
    """

    #: Half-life, in races, of the team pace averages.
    team_halflife: float = 8.0
    #: Half-life, in races, of the driver-versus-team-mate averages.  Driver
    #: skill moves more slowly than a car under development, so this is
    #: searched separately.
    driver_halflife: float = 12.0
    #: Pseudo-visits of the field mean mixed into a circuit's own grid
    #: retention.  Split-half persistence of 0.36 over ~3.5 visits per half
    #: implies a single-visit reliability of about 0.14, and the matching
    #: empirical-Bayes weight is (1 - 0.14) / 0.14, roughly 6 visits.
    retention_shrinkage: float = 6.0
    #: Largest per-race team-mate delta, in field percentiles, allowed into the
    #: driver averages; ``None`` leaves them unclipped.  ``GridPosition`` carries
    #: penalties and qualifying accidents -- the dataset has no clean qualifying
    #: position to separate them -- and one back-of-grid start can dominate a
    #: driver's average: Antonelli started 19th at 2026 R13 and won the race.
    #: Clipping turns that into "lost that head-to-head" rather than "was most
    #: of a field slower".  Searched like every other knob.
    mate_delta_clip: float | None = None


DEFAULT_CONFIG = OrderFeatureConfig()


# --------------------------------------------------------------------------- #
# Per-race quantities
# --------------------------------------------------------------------------- #
# Everything in this section describes the *current* race.  These columns are
# inputs to the shifted aggregates below and never travel into a feature on
# their own; they carry a leading underscore and are dropped before returning.


def _within_race_pct(values: pd.Series, race: pd.Series) -> pd.Series:
    """Rank within race scaled to [0, 1], 0 for the best; NaN rows excluded."""
    rank = values.groupby(race, observed=True).rank(method="average")
    count = values.notna().groupby(race, observed=True).transform("sum")
    return ((rank - 1) / (count - 1)).where(count > 1)


def add_race_quantities(
    frame: pd.DataFrame, config: OrderFeatureConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Per-race grid, finish and places-gained percentiles, and team-mate deltas.

    Percentiles rather than raw positions, so a 20-car and a 22-car grid are
    comparable and so the number of finishers -- which varies from 12 to 20 --
    does not masquerade as pace.
    """
    out = frame.copy()
    race = out[list(RACE_KEYS)].astype(str).agg("|".join, axis=1)

    grid = pd.to_numeric(out["grid_position"], errors="coerce")
    out["_grid_pct"] = _within_race_pct(grid, race)

    classified = pd.to_numeric(out["ClassifiedPosition"], errors="coerce")
    out["_finish_pct"] = _within_race_pct(classified, race)

    # Places gained, net of where the car started.  Grid percentile is taken
    # among finishers only, so both sides of the comparison share a field.
    grid_among_finishers = _within_race_pct(grid.where(classified.notna()), race)
    expected = 0.5 + GRID_REGRESSION_SLOPE * (grid_among_finishers - 0.5)
    out["_gain"] = expected - out["_finish_pct"]

    # Team-mate deltas: negative is better, matching every other percentile.
    team_race = [*RACE_KEYS, "TeamId"]
    for value, name in (("_grid_pct", "_mate_grid_delta"),
                        ("_finish_pct", "_mate_finish_delta")):
        grouped = out.groupby(team_race, observed=True)[value]
        total, count = grouped.transform("sum"), grouped.transform("count")
        mate = (total - out[value]) / (count - 1)
        # Exactly one team-mate with a value; a lone car has no comparison.
        delta = (out[value] - mate).where((count == 2) & out[value].notna())
        if config.mate_delta_clip is not None:
            delta = delta.clip(-config.mate_delta_clip, config.mate_delta_clip)
        out[name] = delta

    out["_race_retention"] = _race_retention(out, race)
    return out


def _race_retention(frame: pd.DataFrame, race: pd.Series) -> pd.Series:
    """Spearman correlation of grid and finish among each race's finishers."""
    classified = pd.to_numeric(frame["ClassifiedPosition"], errors="coerce")
    grid = pd.to_numeric(frame["grid_position"], errors="coerce")
    working = pd.DataFrame({"race": race, "grid": grid, "fin": classified}).dropna()

    def rho(block: pd.DataFrame) -> float:
        if len(block) < 5:
            return np.nan
        return float(spearmanr(block["grid"], block["fin"])[0])

    per_race = working.groupby("race", observed=True)[["grid", "fin"]].apply(rho)
    return race.map(per_race)


# --------------------------------------------------------------------------- #
# Prior-race aggregates
# --------------------------------------------------------------------------- #


def _team_race_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per team per race: the level team history is shifted at."""
    return (
        frame.groupby(["team_lineage", *RACE_KEYS, ORDER_COL, "circuit_key"],
                      observed=True, dropna=False)
        .agg(
            _grid_pct=("_grid_pct", "mean"),
            _finish_pct=("_finish_pct", "mean"),
            _gain=("_gain", "mean"),
        )
        .reset_index()
    )


def add_team_pace(frame: pd.DataFrame, config: OrderFeatureConfig) -> pd.DataFrame:
    """Team pace from prior races, keyed on lineage so rebrands keep history."""
    out = frame.copy()
    team_race = _team_race_table(out)
    h = config.team_halflife

    team_race["team_grid_pct_ewma"] = prior_ewma(
        team_race, "team_lineage", "_grid_pct", halflife=h)
    team_race["team_finish_pct_ewma"] = prior_ewma(
        team_race, "team_lineage", "_finish_pct", halflife=h)
    team_race["team_gain_ewma"] = prior_ewma(
        team_race, "team_lineage", "_gain", halflife=h)
    # The contrast for the winter-carryover hypothesis: the same quantity,
    # reset at every season boundary.  NaN at each season's first race.
    team_race["team_grid_pct_season"] = prior_expanding(
        team_race, ["team_lineage", "Year"], "_grid_pct")

    # Circuit suitability: how much better or worse than its own current form
    # the team qualified on its last three visits here.  The residual is taken
    # against the *prior* form, so this race never enters its own baseline.
    team_race["_circuit_residual"] = (
        team_race["_grid_pct"] - team_race["team_grid_pct_ewma"])
    team_race["team_circuit_grid_residual_3"] = prior_rolling(
        team_race, ["team_lineage", "circuit_key"], "_circuit_residual", 3)

    features = ["team_grid_pct_ewma", "team_finish_pct_ewma", "team_gain_ewma",
                "team_grid_pct_season", "team_circuit_grid_residual_3"]
    return out.merge(
        team_race[["team_lineage", *RACE_KEYS, *features]],
        on=["team_lineage", *RACE_KEYS], how="left", validate="many_to_one",
    )


def add_driver_form(frame: pd.DataFrame, config: OrderFeatureConfig) -> pd.DataFrame:
    """Driver skill from prior races: against the team-mate, and in absolute terms."""
    out = frame.copy()
    h = config.driver_halflife
    out["driver_mate_grid_delta_ewma"] = prior_ewma(
        out, "DriverId", "_mate_grid_delta", halflife=h)
    out["driver_mate_finish_delta_ewma"] = prior_ewma(
        out, "DriverId", "_mate_finish_delta", halflife=h)
    out["driver_gain_ewma"] = prior_ewma(out, "DriverId", "_gain", halflife=h)
    out["driver_grid_pct_ewma"] = prior_ewma(out, "DriverId", "_grid_pct", halflife=h)
    # The drivers' counterpart to team_grid_pct_season: team-mate deltas so far
    # this season only.  Drivers change over a winter as cars do -- a rookie's
    # second season is the obvious case -- and a long average cannot see it.
    out["driver_mate_grid_delta_season"] = prior_expanding(
        out, ["DriverId", "Year"], "_mate_grid_delta")
    out["driver_mate_finish_delta_season"] = prior_expanding(
        out, ["DriverId", "Year"], "_mate_finish_delta")
    return out


def add_circuit_retention(
    frame: pd.DataFrame, config: OrderFeatureConfig
) -> pd.DataFrame:
    """How strongly grid order holds at this circuit, from prior visits, shrunk.

    Built at the race level: one retention value per race, an expanding mean
    over the circuit's previous visits, blended with the field-wide mean over
    all previous races.  A first visit -- Madrid in 2026, Las Vegas in 2023 --
    gets the field mean, which is the honest answer to "we have never seen
    this place".
    """
    out = frame.copy()
    race_level = (
        out.groupby([*RACE_KEYS, ORDER_COL, "circuit_key"], observed=True, dropna=False)
        .agg(_race_retention=("_race_retention", "first"))
        .reset_index()
    )
    race_level["_all"] = 0
    field = prior_expanding(race_level, "_all", "_race_retention")
    field = field.fillna(DEFAULT_RETENTION_PRIOR)
    circuit_sum = prior_expanding(race_level, "circuit_key", "_race_retention", stat="sum")
    circuit_n = prior_expanding(race_level, "circuit_key", "_race_retention", stat="count")
    k = config.retention_shrinkage
    race_level["circuit_grid_retention"] = (
        (circuit_sum.fillna(0.0) + k * field) / (circuit_n.fillna(0.0) + k)
    )
    race_level["_field_retention"] = field
    return out.merge(
        race_level[[*RACE_KEYS, "circuit_grid_retention", "_field_retention"]],
        on=list(RACE_KEYS), how="left", validate="many_to_one",
    )


def add_grid_terms(frame: pd.DataFrame) -> pd.DataFrame:
    """Functional forms and interactions of grid slot (post-qualifying).

    ``log_grid_position`` tests whether the gap between P1 and P2 is worth more
    than the gap between P15 and P16, which a linear term cannot express.  The
    interactions are centred on the field so their coefficient reads as "how
    much more grid matters here than usual".
    """
    out = frame.copy()
    grid = pd.to_numeric(out["grid_position"], errors="coerce")
    pct = pd.to_numeric(out["grid_position_pct"], errors="coerce")
    out["log_grid_position"] = np.log(grid.clip(lower=1))
    out["grid_x_circuit_retention"] = pct * (
        out["circuit_grid_retention"] - out["_field_retention"])
    if "rain_share" in out.columns:
        out["grid_x_rain_share"] = pct * pd.to_numeric(out["rain_share"], errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

#: Every column :func:`add_order_features` produces, each registered in
#: :mod:`src.features.registry` with the stage at which it becomes knowable.
ORDER_FEATURES = (
    "team_grid_pct_ewma",
    "team_finish_pct_ewma",
    "team_gain_ewma",
    "team_grid_pct_season",
    "team_circuit_grid_residual_3",
    "driver_mate_grid_delta_ewma",
    "driver_mate_finish_delta_ewma",
    "driver_gain_ewma",
    "driver_grid_pct_ewma",
    "driver_mate_grid_delta_season",
    "driver_mate_finish_delta_season",
    "circuit_grid_retention",
    "log_grid_position",
    "grid_x_circuit_retention",
    "grid_x_rain_share",
)


def add_order_features(
    dataset: pd.DataFrame, config: OrderFeatureConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Append every finishing-order feature to the modelling table.

    Args:
        dataset: Grand-prix modelling rows, one per starter per race, as built
            by :mod:`src.data.generate_dataset`.  Needs ``ClassifiedPosition``,
            ``grid_position``, ``grid_position_pct``, ``TeamId``, ``DriverId``
            and ``circuit_key`` alongside the race keys.  Sprint rows, if
            present, are dropped: see the module docstring.
        config: Half-lives and circuit shrinkage.

    Returns:
        The input rows, in their original order, with :data:`ORDER_FEATURES`
        appended.
    """
    frame = dataset.copy()
    if "session_type" in frame.columns:
        frame = frame.loc[frame["session_type"].fillna("R") == "R"]
    frame = frame.reset_index(drop=True)
    frame[ORDER_COL] = pd.to_datetime(frame[ORDER_COL])
    frame = add_team_lineage(frame)

    frame = add_race_quantities(frame, config)
    frame = add_team_pace(frame, config)
    frame = add_driver_form(frame, config)
    frame = add_circuit_retention(frame, config)
    frame = add_grid_terms(frame)
    return frame.drop(columns=[c for c in frame.columns if c.startswith("_")])


# --------------------------------------------------------------------------- #
# Leakage detection
# --------------------------------------------------------------------------- #


def detect_order_leakage(
    dataset: pd.DataFrame,
    *,
    builder: Callable[[pd.DataFrame], pd.DataFrame] = add_order_features,
    flip_year: int | None = None,
    flip_round: int | None = None,
    stages: dict[str, str] | None = None,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Find order features that respond to what they are not allowed to see.

    :func:`src.features.build_features.detect_target_leakage` flips ``dnf``,
    which cannot catch a feature built from finishing *position*.  This runs
    two corruptions of one race and checks each against the stage rules:

    * **Reverse the finishing order.**  No feature, at any stage, may move at
      or before that race -- the result is never knowable in advance.
    * **Reverse the grid.**  ``pre_weekend`` features may not move at or before
      that race.  ``post_quali`` and later features may move *in* that race,
      because the grid is known by then, but not before it.

    Returns:
        One row per violation: feature, corruption, rows changed.  Empty means
        clean.
    """
    from src.features import registry

    stages = stages or {name: registry.BY_NAME[name].stage for name in ORDER_FEATURES
                        if name in registry.BY_NAME}
    frame = dataset.reset_index(drop=True).copy()
    if flip_year is None:
        flip_year = int(frame["Year"].max())
    if flip_round is None:
        flip_round = int(frame.loc[frame["Year"] == flip_year, "RoundNumber"].median())
    in_race = ((frame["Year"] == flip_year) & (frame["RoundNumber"] == flip_round)).to_numpy()
    if not in_race.any():
        raise ValueError(f"no rows for {flip_year} R{flip_round}")

    def reverse(column: str) -> pd.DataFrame:
        corrupted = frame.copy()
        values = pd.to_numeric(corrupted.loc[in_race, column], errors="coerce")
        valid = values.notna().to_numpy()
        idx = np.flatnonzero(in_race)[valid]
        flipped = values[valid].to_numpy()[::-1]
        if column == "ClassifiedPosition":
            corrupted[column] = corrupted[column].astype("object")
            corrupted.loc[idx, column] = [str(int(v)) for v in flipped]
        else:
            corrupted.loc[idx, column] = flipped
            if column == "grid_position":
                corrupted.loc[idx, "grid_position_pct"] = (
                    flipped / corrupted.loc[idx, "field_size"].to_numpy())
        return corrupted

    baseline = builder(frame)
    ordinal = event_ordinal(baseline)
    flip_ordinal = int(ordinal[in_race].max())
    at_or_before = (ordinal <= flip_ordinal).to_numpy()
    strictly_before = (ordinal < flip_ordinal).to_numpy()

    rows = []
    for corruption, column in (("finish_order", "ClassifiedPosition"),
                               ("grid_order", "grid_position")):
        altered = builder(reverse(column))
        for name, stage in stages.items():
            if name not in baseline.columns:
                continue
            if corruption == "finish_order" or stage == "pre_weekend":
                scope = at_or_before
            else:
                scope = strictly_before
            left = baseline.loc[scope, name].fillna(-9e18).to_numpy()
            right = altered.loc[scope, name].fillna(-9e18).to_numpy()
            changed = int((np.abs(left - right) > tolerance).sum())
            if changed:
                rows.append({"feature": name, "corruption": corruption,
                             "rows_changed": changed})
    return pd.DataFrame(rows, columns=["feature", "corruption", "rows_changed"])
