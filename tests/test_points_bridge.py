"""The join between the retirement model and the points model.

The claims worth testing are conservation (points cannot be created), that
retirement marginals survive the simulation, and the direction of the
correlation effect — expected points transferring from the front of the grid to
the back when attrition becomes more variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.points_bridge import (
    RACE_POINTS,
    build_points_inputs,
    correlation_impact,
    expected_points,
    points_for_positions,
    race_points_table,
    simulate_points,
    sprint_points_table,
)


@pytest.fixture
def grid_frame() -> pd.DataFrame:
    """Twenty cars, grid order equals pace order, a 14% retirement rate."""
    n = 20
    frame = pd.DataFrame(
        {
            "Year": 2024,
            "RoundNumber": 1,
            "DriverId": [f"d{i:02d}" for i in range(n)],
            "grid_position": np.arange(1, n + 1),
        }
    )
    return build_points_inputs(frame, np.full(n, 0.14))


# --------------------------------------------------------------------------- #
# Points tables
# --------------------------------------------------------------------------- #


def test_race_points_table() -> None:
    assert race_points_table(2024) == RACE_POINTS
    assert sum(RACE_POINTS) == 101.0  # 10 scoring positions


def test_sprint_points_by_season() -> None:
    assert sprint_points_table(2020) == ()
    assert sprint_points_table(2021) == (3.0, 2.0, 1.0)
    assert sprint_points_table(2023) == (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)


def test_points_for_positions() -> None:
    assert points_for_positions([1, 3, 10, 11], 2024).tolist() == [25.0, 15.0, 1.0, 0.0]


def test_retirement_scores_nothing() -> None:
    """Position 0 marks a retirement and must never score."""
    assert points_for_positions([0, np.nan, -1], 2024).tolist() == [0.0, 0.0, 0.0]


def test_pre_2010_season_is_rejected() -> None:
    with pytest.raises(ValueError, match="predates"):
        race_points_table(2008)


# --------------------------------------------------------------------------- #
# The analytic decomposition
# --------------------------------------------------------------------------- #


def test_expected_points_decomposition() -> None:
    assert expected_points([0.8, 1.0, 0.0], [10.0, 5.0, 25.0]).tolist() == [8.0, 5.0, 0.0]


def test_expected_points_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        expected_points([0.5, 0.5], [1.0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_points([1.4], [1.0])


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def test_simulation_preserves_retirement_marginals(grid_frame) -> None:
    for sigma in (0.0, 0.8):
        sim = simulate_points(grid_frame, sigma=sigma, n_sims=20000, seed=1)
        assert sim.retired.mean(axis=1) == pytest.approx(0.14, abs=0.015)


def test_points_are_conserved(grid_frame) -> None:
    """The table is fixed, so a race cannot award more than 101 points.

    It can award fewer: with eleven or more retirements there are not ten
    finishers to pay.
    """
    sim = simulate_points(grid_frame, sigma=0.8, n_sims=5000, seed=2)
    per_race_totals = sim.points.sum(axis=0)
    assert per_race_totals.max() <= sum(RACE_POINTS) + 1e-9
    assert per_race_totals.mean() > sum(RACE_POINTS) * 0.95


def test_retirees_never_score(grid_frame) -> None:
    sim = simulate_points(grid_frame, sigma=0.5, n_sims=3000, seed=3)
    assert sim.points[sim.retired == 1].max() == 0.0
    assert sim.positions[sim.retired == 1].max() == 0.0


def test_positions_are_a_valid_ranking(grid_frame) -> None:
    """Survivors must occupy 1..n_finishers with no duplicates or gaps."""
    sim = simulate_points(grid_frame, sigma=0.6, n_sims=200, seed=4)
    for column in range(sim.positions.shape[1]):
        finishers = sim.positions[:, column]
        ranked = np.sort(finishers[finishers > 0])
        assert ranked.tolist() == list(range(1, len(ranked) + 1))


def test_faster_cars_score_more(grid_frame) -> None:
    summary = simulate_points(grid_frame, sigma=0.0, n_sims=8000, seed=5).summary()
    points = summary["expected_points"].to_numpy()
    # Monotone in grid slot, allowing for simulation noise at the tail.
    assert points[0] > points[5] > points[12] > points[19]


def test_zero_noise_makes_the_order_deterministic(grid_frame) -> None:
    sim = simulate_points(
        grid_frame, sigma=0.0, noise_scale=1e-9, n_sims=300, seed=6
    )
    clean = sim.positions[:, sim.retired.sum(axis=0) == 0]
    if clean.shape[1]:
        assert np.all(clean[:, 0] == np.arange(1, len(grid_frame) + 1))


# --------------------------------------------------------------------------- #
# The correlation effect
# --------------------------------------------------------------------------- #


def test_correlation_transfers_points_down_the_grid(grid_frame) -> None:
    """The headline claim, and the reason the random effect earns its place.

    A back-marker's points are a convex function of how much attrition a race
    produces, so adding variance to attrition while holding its mean raises
    their expectation.  A front-runner's payoff is concave, so theirs falls.
    """
    independent = simulate_points(
        grid_frame, sigma=0.0, noise_scale=0.6, n_sims=30000, seed=7
    ).summary()["expected_points"].to_numpy()
    correlated = simulate_points(
        grid_frame, sigma=0.8, noise_scale=0.6, n_sims=30000, seed=7
    ).summary()["expected_points"].to_numpy()

    front, back = slice(0, 5), slice(12, 20)
    assert correlated[front].sum() < independent[front].sum()
    assert correlated[back].sum() > independent[back].sum()
    # Conservation: the transfer nets out.
    assert correlated.sum() == pytest.approx(independent.sum(), rel=0.02)


def test_correlation_widens_the_points_distribution(grid_frame) -> None:
    independent = simulate_points(
        grid_frame, sigma=0.0, noise_scale=0.6, n_sims=20000, seed=8
    ).summary()
    correlated = simulate_points(
        grid_frame, sigma=0.8, noise_scale=0.6, n_sims=20000, seed=8
    ).summary()
    midfield = slice(10, 16)
    assert (
        correlated["points_sd"].to_numpy()[midfield].mean()
        > independent["points_sd"].to_numpy()[midfield].mean()
    )


def test_correlation_impact_reports_both_scenarios(grid_frame) -> None:
    impact = correlation_impact(grid_frame, 0.8, n_sims=4000, noise_scale=0.6)
    for column in (
        "expected_points_correlated",
        "expected_points_independent",
        "points_sd_ratio",
        "points_p90_gap",
    ):
        assert column in impact.columns
    assert len(impact) == len(grid_frame)


def test_build_points_inputs_requires_the_strength_column() -> None:
    frame = pd.DataFrame({"Year": [2024], "RoundNumber": [1]})
    with pytest.raises(KeyError, match="not_a_column"):
        build_points_inputs(frame, [0.1], strength_from="not_a_column")


def test_simulate_points_requires_its_columns(grid_frame) -> None:
    with pytest.raises(KeyError, match="pace_strength"):
        simulate_points(grid_frame.drop(columns=["pace_strength"]), n_sims=10)
