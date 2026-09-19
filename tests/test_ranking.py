"""Plackett-Luce is only useful if its likelihood, gradient and sampler agree.

Every check here is against something known exactly: a finite difference, a
closed-form probability, or the coefficients that generated the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ranking import (
    PlackettLuce,
    _objective,
    _stagewise_objective,
    pack_races,
    race_log_likelihood,
    sample_finishing_positions,
    sample_positions,
    stage_scales,
    uniform_log_likelihood,
)


def _synthetic_races(n_races=300, n_cars=18, beta=(1.2, -0.6, 0.0), alpha=None, seed=0):
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta)
    rows = []
    for r in range(n_races):
        X = rng.normal(size=(n_cars, len(beta)))
        pos = sample_finishing_positions(X @ beta, None, 1, rng, alpha=alpha)[0]
        for i in range(n_cars):
            rows.append({"Year": 2000 + r // 25, "RoundNumber": r % 25 + 1, "pos": pos[i],
                         **{f"f{k}": X[i, k] for k in range(len(beta))}})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def races() -> pd.DataFrame:
    return _synthetic_races()


def _numeric_grad(f, x, eps=1e-6):
    return np.array([(f(x + e) - f(x - e)) / (2 * eps) for e in np.eye(len(x)) * eps])


@pytest.mark.parametrize("truncate", [None, 10])
def test_gradient_matches_finite_differences(races, truncate) -> None:
    batch = pack_races(races, races[["f0", "f1", "f2"]].to_numpy(), races["pos"])
    w = np.ones(len(batch.n_ranked))
    beta = np.array([0.3, -0.2, 0.1])
    _, grad = _objective(beta, batch, w, 0.5, truncate)
    numeric = _numeric_grad(lambda b: _objective(b, batch, w, 0.5, truncate)[0], beta)
    np.testing.assert_allclose(grad, numeric, atol=1e-4)


def test_stagewise_gradient_matches_finite_differences(races) -> None:
    batch = pack_races(races, races[["f0", "f1", "f2"]].to_numpy(), races["pos"])
    w = np.ones(len(batch.n_ranked))
    theta = np.array([0.4, -0.3, 0.1, -0.2, 0.1, -0.5])
    _, grad = _stagewise_objective(theta, batch, w, 0.3, 10, 3, 0.5)
    numeric = _numeric_grad(
        lambda t: _stagewise_objective(t, batch, w, 0.3, 10, 3, 0.5)[0], theta)
    np.testing.assert_allclose(grad, numeric, atol=1e-4)


def test_unit_scales_reduce_to_standard_plackett_luce(races) -> None:
    batch = pack_races(races, races[["f0", "f1", "f2"]].to_numpy(), races["pos"])
    w = np.ones(len(batch.n_ranked))
    beta = np.array([0.3, -0.2, 0.1])
    standard = _objective(beta, batch, w, 0.3, None)[0]
    stagewise = _stagewise_objective(np.r_[beta, 0.0, 0.0], batch, w, 0.3, None, 2, 0.5)[0]
    assert standard == pytest.approx(stagewise)


def test_recovers_the_coefficients_that_generated_the_data(races) -> None:
    model = PlackettLuce(["f0", "f1", "f2"], l2=0.01).fit(races, races["pos"])
    np.testing.assert_allclose(model.coef_ / model.scale_, [1.2, -0.6, 0.0], atol=0.12)


def test_recovers_stage_scales() -> None:
    true_alpha = stage_scales(np.log([0.7, 0.5]))
    data = _synthetic_races(n_races=400, beta=(1.5, -0.7), alpha=true_alpha, seed=1)
    model = PlackettLuce(["f0", "f1"], l2=0.01, n_scales=2, l2_scale=0.01).fit(data, data["pos"])
    np.testing.assert_allclose(model.alpha_[:3], [1.0, 0.7, 0.5], atol=0.12)


def test_truncating_at_the_field_size_is_the_full_likelihood(races) -> None:
    batch = pack_races(races, races[["f0", "f1", "f2"]].to_numpy(), races["pos"])
    s = batch.X @ np.array([0.5, -0.5, 0.2])
    full = race_log_likelihood(s, batch.mask, batch.n_ranked, None)
    np.testing.assert_allclose(
        race_log_likelihood(s, batch.mask, batch.n_ranked, 17), full)
    np.testing.assert_allclose(
        race_log_likelihood(s, batch.mask, batch.n_ranked, None, np.ones(30)), full)


def test_uniform_log_likelihood_counts_the_orders() -> None:
    # Top ten of twenty, every order equally likely: 20 * 19 * ... * 11 of them.
    expected = -np.log(np.arange(11, 21)).sum()
    assert uniform_log_likelihood(np.array([20]), 10)[0] == pytest.approx(expected)


def test_win_probability_is_the_softmax() -> None:
    rng = np.random.default_rng(3)
    s = np.array([2.0, 1.0, 0.0, -1.0])
    pos = sample_finishing_positions(s, None, 200_000, rng)
    np.testing.assert_allclose((pos == 1).mean(axis=0), np.exp(s) / np.exp(s).sum(), atol=0.005)


def test_stagewise_second_place_uses_its_own_scale() -> None:
    rng = np.random.default_rng(4)
    s = np.array([1.0, 0.5, 0.0, -0.5])
    pos = sample_finishing_positions(s, None, 300_000, rng, alpha=stage_scales(np.log([0.4])))
    won = pos[:, 0] == 1
    empirical = [(pos[won, i] == 2).mean() for i in (1, 2, 3)]
    exact = np.exp(0.4 * s[1:]) / np.exp(0.4 * s[1:]).sum()
    np.testing.assert_allclose(empirical, exact, atol=0.006)


@pytest.mark.parametrize("alpha", [None, stage_scales(np.log([0.6, 0.4]))])
def test_retired_cars_are_never_placed(alpha) -> None:
    rng = np.random.default_rng(5)
    pos = sample_finishing_positions(np.linspace(1, -1, 8), np.full(8, 0.3), 2000, rng,
                                     alpha=alpha)
    for row in pos:
        placed = np.sort(row[row > 0])
        np.testing.assert_array_equal(placed, np.arange(1, len(placed) + 1))


def test_each_sample_can_carry_its_own_strengths() -> None:
    rng = np.random.default_rng(6)
    s = np.tile([2.0, 1.0, 0.0, -1.0], (40_000, 1))
    s[20_000:, 3] = 6.0
    pos = sample_positions(s, np.zeros_like(s, dtype=bool), None, rng)
    assert (pos[:20_000, 3] == 1).mean() < 0.05
    assert (pos[20_000:, 3] == 1).mean() > 0.9
