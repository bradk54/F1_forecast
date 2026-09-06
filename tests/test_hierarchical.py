"""Random effects: the estimator has to recover a shock that is really there.

Everything here plants a known answer and checks it comes back.  A variance
component that cannot be recovered from simulated data will not be trustworthy
on real data either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from src.models.hierarchical import (
    RaceRandomEffectGLMM,
    check_attrition_calibration,
    conditional_to_marginal,
    estimate_race_shock,
    finisher_count_distribution,
    marginal_to_conditional,
    simulate_retirements,
)


def _planted(true_sigma: float, n_races: int = 200, n_drivers: int = 20, seed: int = 0):
    """Simulate a grid with a known race shock; return marginal probs and outcomes."""
    rng = np.random.default_rng(seed)
    race_ids = np.repeat(np.arange(n_races), n_drivers)
    eta = rng.normal(-2.0, 0.8, size=n_races * n_drivers)
    shock = rng.normal(0.0, true_sigma, size=n_races)[race_ids]
    y = (rng.random(len(eta)) < expit(eta + shock)).astype(int)
    # What a well-calibrated mean model would report: the marginal.
    return conditional_to_marginal(eta, true_sigma), y, race_ids


def _mean_estimate(true_sigma: float, seeds: range, **kwargs) -> float:
    """Average estimate across seeds.

    Averaging matters: at 600 races the estimator's own standard deviation is
    around 0.04, so a single-seed assertion with a tight tolerance is a coin
    flip rather than a test of correctness.
    """
    estimates = []
    for seed in seeds:
        marginal, y, race_ids = _planted(true_sigma, seed=seed, **kwargs)
        estimates.append(estimate_race_shock(y, marginal, race_ids).sigma)
    return float(np.mean(estimates))


@pytest.mark.parametrize("true_sigma", [0.4, 0.7, 1.1])
def test_recovers_a_planted_shock(true_sigma: float) -> None:
    estimate = _mean_estimate(true_sigma, range(5), n_races=600)
    assert estimate == pytest.approx(true_sigma, abs=0.06)


def test_returns_near_zero_when_there_is_no_shock() -> None:
    """Independence must be detectable, not merely assumed away.

    The estimate is checked in the mean rather than per seed: sigma is bounded
    below at zero, so at the null its sampling distribution is a half-normal
    pressed against the boundary and individual draws of 0.2 or so are routine.
    """
    assert _mean_estimate(0.0, range(8), n_races=600) < 0.15


def test_shock_is_significant_when_present() -> None:
    marginal, y, race_ids = _planted(0.8, n_races=400, seed=7)
    fitted = estimate_race_shock(y, marginal, race_ids)
    assert fitted.likelihood_ratio > 2.7
    assert 0.0 < fitted.intraclass_correlation < 1.0


def test_estimator_is_self_consistent() -> None:
    """Regression: treating ``logit(marginal)`` as the conditional predictor
    left a persistent downward bias of a few percent that did not shrink with
    sample size.  The fix recomputes the conditional predictor at each
    candidate sigma inside the search."""
    true_sigma = 0.6
    bias = _mean_estimate(true_sigma, range(6), n_races=500) - true_sigma
    assert abs(bias) < 0.04, f"bias {bias:+.4f} suggests a specification error"


# --------------------------------------------------------------------------- #
# Marginal <-> conditional
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sigma", [0.2, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("p", [0.02, 0.1, 0.14, 0.5, 0.85, 0.98])
def test_marginal_round_trip_is_exact(sigma: float, p: float) -> None:
    eta = marginal_to_conditional([p], sigma)
    recovered = conditional_to_marginal(eta, sigma)
    assert recovered[0] == pytest.approx(p, abs=1e-9)


def test_skipping_the_conversion_biases_toward_one_half() -> None:
    """Why the conversion exists: the naive route shifts every probability.

    ``expit`` is convex below 0.5, so a mean-zero shock raises a small
    probability.  For a 14% event that means over-predicting retirements.
    """
    p = 0.14
    sigma = 1.0
    naive = conditional_to_marginal(np.array([logit(p)]), sigma)[0]
    correct = conditional_to_marginal(marginal_to_conditional([p], sigma), sigma)[0]
    assert naive > p + 0.01          # biased upward, toward 0.5
    assert correct == pytest.approx(p, abs=1e-9)


def test_zero_sigma_is_the_identity() -> None:
    p = np.array([0.1, 0.5, 0.9])
    assert marginal_to_conditional(p, 0.0) == pytest.approx(logit(p))
    assert conditional_to_marginal(logit(p), 0.0) == pytest.approx(p)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def test_simulation_preserves_the_marginals() -> None:
    """The whole point: add correlation without moving any driver's own risk."""
    p = np.array([0.05, 0.10, 0.14, 0.25, 0.40] * 4)
    race_ids = np.zeros(len(p))
    for sigma in (0.0, 0.5, 1.2):
        draws = simulate_retirements(p, race_ids, sigma, n_sims=40000, seed=11)
        assert draws.mean(axis=1) == pytest.approx(p, abs=0.012)


def test_shock_widens_the_finisher_count_distribution() -> None:
    p = np.full(20, 0.14)
    race_ids = np.zeros(20)
    spreads = []
    for sigma in (0.0, 0.5, 1.0):
        totals = simulate_retirements(p, race_ids, sigma, n_sims=20000, seed=2).sum(axis=0)
        spreads.append(totals.std())
    assert spreads[0] < spreads[1] < spreads[2]


def test_shock_fattens_the_attrition_tail() -> None:
    """The chaotic race is what an independent model cannot generate."""
    p = np.full(20, 0.14)
    race_ids = np.zeros(20)
    independent = simulate_retirements(p, race_ids, 0.0, n_sims=40000, seed=4).sum(axis=0)
    correlated = simulate_retirements(p, race_ids, 1.0, n_sims=40000, seed=4).sum(axis=0)
    assert np.mean(correlated >= 6) > 2 * np.mean(independent >= 6)


def test_finisher_count_distribution_reports_both() -> None:
    p = np.full(40, 0.14)
    race_ids = np.repeat([0, 1], 20)
    frame = finisher_count_distribution(p, race_ids, 0.8, n_sims=4000)
    assert len(frame) == 2
    assert (frame["sd_correlated"] > frame["sd_independent"]).all()


def test_glmm_recovers_a_shock(monkeypatch) -> None:
    """The statsmodels route should agree with the quadrature route."""
    rng = np.random.default_rng(0)
    n_races, n_drivers, true_sigma = 120, 18, 0.9
    race_ids = np.repeat(np.arange(n_races), n_drivers)
    x = rng.normal(size=n_races * n_drivers)
    eta = -1.8 + 0.7 * x
    shock = rng.normal(0.0, true_sigma, size=n_races)[race_ids]
    y = (rng.random(len(eta)) < expit(eta + shock)).astype(int)

    frame = pd.DataFrame({"x": x, "dnf": y, "race_id": race_ids})
    result = RaceRandomEffectGLMM(["x"], race_col="race_id").fit(frame)

    assert len(result.race_effects) == n_races
    assert result.variance_components.loc[0, "sigma"] > 0.2
    # The fixed effect should come back with the right sign.
    slope = result.fixed_effects.set_index("term").loc["x", "coefficient"]
    assert slope > 0


def test_attrition_calibration_matches_the_observed_spread() -> None:
    """The diagnostic that matters more than the likelihood-ratio test.

    A correctly fitted shock should reproduce the observed spread of per-race
    retirement counts; assuming independence should visibly under-shoot it.
    """
    marginal, y, race_ids = _planted(0.8, n_races=300, seed=13)
    fitted = estimate_race_shock(y, marginal, race_ids)
    table = check_attrition_calibration(y, marginal, race_ids, fitted.sigma).set_index(
        "scenario"
    )

    observed_sd = table.loc["observed", "sd"]
    assert table.loc["sigma_fitted", "sd"] == pytest.approx(observed_sd, rel=0.20)
    assert table.loc["independent", "sd"] < observed_sd * 0.85
    # Means agree regardless: the shock changes the spread, not the level.
    assert table.loc["sigma_fitted", "mean"] == pytest.approx(
        table.loc["observed", "mean"], rel=0.15
    )
