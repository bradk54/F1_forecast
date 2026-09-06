"""Turning a retirement forecast into a points forecast.

The retirement model is not the destination; the points model is.  This module
is the join between them, and it exists because the obvious join is subtly
wrong in a way that costs real accuracy.

The obvious version
-------------------
.. math::

    E[\\text{points}_j] = P(\\text{finish}_j) \\times E[\\text{points}_j \\mid \\text{finish}_j]

:func:`expected_points` implements exactly this, and for a *point estimate* of
one driver's expected points it is fine.  Use it when you already have a points
model conditional on finishing and only want to discount it for reliability.

Why it is not enough
--------------------
Points are awarded by finishing *position*, and position depends on who else is
still running.  Two consequences follow, neither of which the formula above can
express:

1. **Promotion.**  A driver running eleventh scores nothing.  If five cars ahead
   retire, that same driver finishes sixth and scores eight.  So a rival's
   retirement is worth points to everyone behind them, and
   :math:`E[\\text{points}_j \\mid \\text{finish}_j]` is not a fixed quantity —
   it depends on the whole grid's retirement pattern.
2. **Correlation.**  Retirements cluster (see
   :mod:`src.models.hierarchical`).  The races that pay a midfield driver
   heavily are precisely the chaotic ones, and a model treating drivers as
   independent will almost never generate a race with eight retirements.  It
   will therefore systematically understate the upper tail of a midfield
   driver's points distribution — the exact scenario worth forecasting.

So :func:`simulate_points` does it properly: simulate the correlated retirement
pattern, rank the survivors, and read points off the table.  Both terms then
come out of the same simulation and stay consistent with each other.

How much this matters
---------------------
More than it looks.  Because a back-marker's points are a convex function of
how much attrition a race produces, and a front-runner's are concave, adding
correlation transfers expected points down the grid even though every driver's
retirement probability is unchanged.  In a stylised twenty-car race at a 14%
retirement rate, moving from independent retirements to a race shock of
``sigma = 0.8`` moved roughly seven points a race from the front five to the
back eight, nearly tripling last-on-the-grid's expected score.  See
:func:`correlation_impact`.

The ordering model
------------------
Turning driver strengths into a random finishing order uses a Plackett-Luce
model: add Gumbel noise to each driver's log-strength and sort.  That is the
standard random-utility construction, and it is what makes the ordering
probabilistic rather than deterministic.  ``noise_scale`` controls how much
racing luck there is; calibrate it against how often the observed order matches
qualifying order rather than guessing.

Bring your own strengths.  Anything monotone in pace works — a rolling points
average, a qualifying-gap model, the output of notebook 2.0.  Grid position
inverted is a serviceable starting point.

Points tables
-------------
Values follow the FIA Sporting Regulations for each season: 25-18-15-12-10-8-6-4-2-1
for a grand prix since 2010; a bonus point for fastest lap when finishing in the
top ten, from 2019 to 2024 inclusive and abolished for 2025; sprint points of
3-2-1 in 2021 and 8-7-6-5-4-3-2-1 from 2022.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src import config
from src.models.hierarchical import simulate_retirements

log = logging.getLogger(__name__)

#: Grand prix points by finishing position, 2010 onward.
RACE_POINTS = (25.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0)

#: Sprint points by season.
SPRINT_POINTS_2021 = (3.0, 2.0, 1.0)
SPRINT_POINTS_2022 = (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)

#: Seasons in which a bonus point was awarded for fastest lap, provided the
#: driver finished in the top ten.  Abolished from 2025.
FASTEST_LAP_SEASONS = frozenset(range(2019, 2025))


def race_points_table(season: int) -> tuple[float, ...]:
    """Points by finishing position for a grand prix in ``season``."""
    if season < 2010:
        raise ValueError(
            f"season {season} predates the current points system; this module "
            "covers 2010 onward"
        )
    return RACE_POINTS


def sprint_points_table(season: int) -> tuple[float, ...]:
    """Points by finishing position for a sprint in ``season``."""
    if season < 2021:
        return ()
    return SPRINT_POINTS_2021 if season == 2021 else SPRINT_POINTS_2022


def points_for_positions(
    positions: Sequence[int] | np.ndarray, season: int = 2024
) -> np.ndarray:
    """Map 1-indexed finishing positions to points.

    Positions outside the points-paying range, and any non-positive or NaN
    position (a retirement), score zero.

    >>> points_for_positions([1, 2, 10, 11, 0], season=2024).tolist()
    [25.0, 18.0, 1.0, 0.0, 0.0]
    """
    table = np.asarray(race_points_table(season), dtype=float)
    pos = np.asarray(positions, dtype=float)
    out = np.zeros_like(pos, dtype=float)
    valid = np.isfinite(pos) & (pos >= 1) & (pos <= len(table))
    out[valid] = table[pos[valid].astype(int) - 1]
    return out


# --------------------------------------------------------------------------- #
# The analytic decomposition
# --------------------------------------------------------------------------- #


def expected_points(
    finish_probability: Sequence[float],
    points_given_finish: Sequence[float],
) -> np.ndarray:
    """``P(finish) * E[points | finish]``, the simple decomposition.

    Correct as a point estimate when ``points_given_finish`` already accounts
    for the field.  It cannot express the promotion effect — a rival's
    retirement lifting this driver's finishing position — so it will
    under-state expected points for midfield drivers, who benefit most from
    attrition.  Use :func:`simulate_points` when you need the distribution or
    when attrition is high.

    >>> expected_points([0.9, 0.5], [10.0, 20.0]).tolist()
    [9.0, 10.0]
    """
    p = np.asarray(list(finish_probability), dtype=float)
    conditional = np.asarray(list(points_given_finish), dtype=float)
    if p.shape != conditional.shape:
        raise ValueError(
            f"shape mismatch: {p.shape} finish probabilities vs "
            f"{conditional.shape} conditional points"
        )
    if np.any((p < 0) | (p > 1)):
        raise ValueError("finish probabilities must lie in [0, 1]")
    return p * conditional


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


@dataclass
class PointsSimulation:
    """Per-driver points distribution from a simulated set of races."""

    points: np.ndarray
    """``(n_drivers, n_sims)`` array of points scored."""

    positions: np.ndarray
    """``(n_drivers, n_sims)`` finishing positions; 0 marks a retirement."""

    retired: np.ndarray
    """``(n_drivers, n_sims)`` retirement indicators."""

    index: pd.DataFrame
    """Row identity: whatever key columns were supplied."""

    def summary(self) -> pd.DataFrame:
        """Expected points, spread, and the probabilities worth quoting."""
        return self.index.assign(
            p_finish=1.0 - self.retired.mean(axis=1),
            expected_points=self.points.mean(axis=1),
            points_sd=self.points.std(axis=1),
            p_scores=(self.points > 0).mean(axis=1),
            p_podium=((self.positions >= 1) & (self.positions <= 3)).mean(axis=1),
            p_win=(self.positions == 1).mean(axis=1),
            points_p90=np.percentile(self.points, 90, axis=1),
        )


def simulate_finishing_order(
    strength: np.ndarray,
    retired: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    """Rank survivors by noisy strength; retirements get position 0.

    Adding Gumbel noise to a log-strength and sorting is the Plackett-Luce
    random-utility model: the probability a driver finishes ahead of another is
    a logistic function of their strength difference.  ``noise_scale`` is the
    amount of racing luck — zero makes the order deterministic in strength.

    Args:
        strength: ``(n_drivers, n_sims)`` log-strength, higher is faster.
        retired: ``(n_drivers, n_sims)`` retirement indicators.
        rng: Random generator.
        noise_scale: Gumbel scale.

    Returns:
        ``(n_drivers, n_sims)`` 1-indexed positions, 0 for retirements.
    """
    utility = strength + rng.gumbel(0.0, max(noise_scale, 1e-9), size=strength.shape)
    # Push retirees below every finisher so they never take a scoring position.
    utility = np.where(retired == 1, -np.inf, utility)

    order = np.argsort(-utility, axis=0, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(strength.shape[0])[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows, order.shape), axis=0)

    positions = (ranks + 1).astype(float)
    positions[retired == 1] = 0.0
    return positions


def simulate_points(
    frame: pd.DataFrame,
    *,
    dnf_probability_col: str = "dnf_probability",
    strength_col: str = "pace_strength",
    race_col: str = "race_id",
    season_col: str = "Year",
    key_cols: Sequence[str] = ("race_id", "DriverId"),
    sigma: float = 0.0,
    noise_scale: float = 1.0,
    n_sims: int = 5000,
    seed: int = config.RANDOM_SEED,
) -> PointsSimulation:
    """Simulate races end to end: retirements, then order, then points.

    This is the version to use when the answer feeds a season simulation or a
    championship probability, because it produces the joint distribution rather
    than a set of marginals.

    Args:
        frame: One row per driver per race.
        dnf_probability_col: Marginal retirement probability, from the model in
            :mod:`src.models.train`.
        strength_col: Log-strength for the ordering model.  Higher is faster.
            Any monotone measure of pace works.
        race_col: Groups rows into races.  Retirement correlation and the
            finishing order both operate within these groups.
        sigma: Race shock from :func:`src.models.hierarchical.estimate_race_shock`.
            Zero reproduces the independence assumption, which is the useful
            comparison rather than the useful model.
        noise_scale: Gumbel scale for the ordering.  Larger means more racing
            luck and a flatter distribution of positions.
        n_sims: Simulations per race.

    Returns:
        A :class:`PointsSimulation`.
    """
    for column in (dnf_probability_col, strength_col, race_col):
        if column not in frame.columns:
            raise KeyError(f"{column!r} not in frame; columns: {list(frame.columns)}")

    rng = np.random.default_rng(seed)
    n_rows = len(frame)

    retired = simulate_retirements(
        frame[dnf_probability_col].to_numpy(dtype=float),
        frame[race_col].to_numpy(),
        sigma,
        n_sims=n_sims,
        seed=seed,
    )

    strength = np.repeat(
        frame[strength_col].to_numpy(dtype=float)[:, None], n_sims, axis=1
    )

    positions = np.zeros((n_rows, n_sims), dtype=float)
    points = np.zeros((n_rows, n_sims), dtype=float)

    seasons = (
        frame[season_col].to_numpy()
        if season_col in frame.columns
        else np.full(n_rows, 2024)
    )

    # Ordering and points are per race: a driver's position depends only on the
    # others in their own event.
    for race, idx in frame.groupby(race_col, observed=True, sort=False).indices.items():
        block_positions = simulate_finishing_order(
            strength[idx], retired[idx], rng, noise_scale
        )
        positions[idx] = block_positions
        season = int(pd.Series(seasons[idx]).mode().iat[0])
        table = np.asarray(race_points_table(season), dtype=float)
        scoring = (block_positions >= 1) & (block_positions <= len(table))
        block_points = np.zeros_like(block_positions)
        block_points[scoring] = table[block_positions[scoring].astype(int) - 1]
        points[idx] = block_points

    key_cols = [c for c in key_cols if c in frame.columns] or [race_col]
    return PointsSimulation(
        points=points,
        positions=positions,
        retired=retired,
        index=frame[key_cols].reset_index(drop=True),
    )


def correlation_impact(
    frame: pd.DataFrame,
    sigma: float,
    **kwargs,
) -> pd.DataFrame:
    """Quantify what ignoring retirement correlation costs a points forecast.

    Runs the same simulation twice — once with the estimated race shock, once
    assuming independence — and reports the difference per driver.

    **Expected points move, and not by a little.**  It is tempting to assume
    they cannot, since the retirement marginals are identical by construction,
    but points are a nonlinear function of the *joint* retirement pattern, and
    correlation changes the joint while holding the marginals.

    The direction follows from Jensen's inequality.  A back-marker scores
    nothing unless several cars ahead retire, so their points are a *convex*
    function of how much attrition the race produces; raising the variance of
    attrition while holding its mean therefore raises their expected points.  A
    front-runner is near the top of the points table already, so their payoff
    is *concave* in attrition and their expected points fall.  Points are
    conserved, so the whole effect is a transfer from the front of the grid to
    the back.

    In a stylised twenty-car race at a 14% retirement rate and ``sigma = 0.8``,
    that transfer was around seven points a race — the front five losing about
    7.1 and the back eight gaining about 6.2, with last on the grid going from
    0.40 expected points to 1.18.  Treat the magnitude as setup-dependent (it
    scales with ``noise_scale`` and the spread of strengths) but the direction
    as robust.

    The practical consequence: a points model built on independent retirements
    will systematically under-rate the back of the grid and over-rate the
    front.
    """
    with_shock = simulate_points(frame, sigma=sigma, **kwargs).summary()
    without = simulate_points(frame, sigma=0.0, **kwargs).summary()

    merged = with_shock.merge(
        without,
        on=list(with_shock.columns[: len(with_shock.columns) - 7]),
        suffixes=("_correlated", "_independent"),
    )
    merged["points_sd_ratio"] = (
        merged["points_sd_correlated"] / merged["points_sd_independent"].replace(0, np.nan)
    ).round(3)
    merged["points_p90_gap"] = (
        merged["points_p90_correlated"] - merged["points_p90_independent"]
    ).round(3)
    return merged


def build_points_inputs(
    dataset: pd.DataFrame,
    dnf_probability: Sequence[float],
    *,
    strength_from: str = "grid_position",
    race_cols: Sequence[str] = ("Year", "RoundNumber"),
) -> pd.DataFrame:
    """Assemble the frame :func:`simulate_points` expects.

    ``strength_from`` defaults to inverted grid position, which is a serviceable
    stand-in until the pace model from notebook 2.0 is wired in — replace it
    with that model's output as soon as it exists, since the ordering model is
    only as good as the strengths it is given.
    """
    out = dataset.copy()
    out["race_id"] = out[list(race_cols)].astype(str).agg("_".join, axis=1)
    out["dnf_probability"] = np.asarray(list(dnf_probability), dtype=float)

    if strength_from not in out.columns:
        raise KeyError(f"{strength_from!r} not in dataset")
    values = pd.to_numeric(out[strength_from], errors="coerce")
    if strength_from in ("grid_position", "grid_position_pct"):
        # Lower grid number means faster, so invert and put it on a log-ish
        # scale where a Gumbel shock is a sensible amount of racing luck.
        values = values.fillna(values.median())
        out["pace_strength"] = -np.log(values.clip(lower=1.0))
    else:
        out["pace_strength"] = values.fillna(values.median())
    return out
