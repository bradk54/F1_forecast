"""Race-level random effects: the shared shock that makes retirements cluster.

Every model in :mod:`src.models.train` treats driver-races as independent.  For
a single driver's marginal probability that is fine.  For anything that adds
drivers up — expected points, a season simulation, a championship probability —
it is wrong, and wrong in a direction that matters.

Retirements cluster.  Rain, a first-lap pile-up at turn one, a safety car that
catches half the field out, a red flag: one event takes several cars out at
once.  Baku 2021, Spa 2021, Monza 2020.  A model assuming independence will get
each driver's *marginal* probability right and still produce a distribution of
finisher counts that is far too narrow.

Why that matters for points specifically
----------------------------------------
Points are awarded by finishing position, so a retirement ahead of you is a
promotion.  A driver running eleventh scores nothing; if five cars ahead retire,
that same driver finishes sixth and scores eight.  So retirement correlation
does not just add noise to a points forecast, it drives the upside — and a model
that cannot produce "eight cars retired" will never produce the races where a
midfield driver scores heavily.  Getting the joint distribution right is the
difference between a points model that is directionally sensible and one you
could bet on.

The model
---------
A logistic random-intercept model, one intercept per race:

.. math::

    \\mathrm{logit}(p_{ij}) = \\eta_{ij} + u_i, \\qquad u_i \\sim N(0, \\sigma^2)

where :math:`i` indexes races and :math:`j` drivers.  :math:`\\eta_{ij}` is the
linear predictor from *any* model — the XGBoost fit, the logistic baseline,
whatever — so this composes with the existing pipeline rather than replacing it.
Conditional on :math:`u_i` the drivers in a race are independent, which makes
simulation trivial: draw the shock, then draw the drivers.

Two subtleties handled here
---------------------------
1. **Marginal versus conditional.**  Adding a mean-zero shock inside a logit
   does *not* leave the marginal probability unchanged, because the logistic is
   convex below 0.5 and concave above.  Feeding a marginally-calibrated
   probability straight in as :math:`\\eta` and then adding :math:`u` shifts
   every prediction.  :func:`marginal_to_conditional` solves for the
   :math:`\\eta` that reproduces the marginal exactly, by Newton iteration on
   the quadrature integral rather than the usual
   :math:`\\sqrt{1 + 0.346\\sigma^2}` approximation.
2. **Estimating** :math:`\\sigma` **without refitting.**  :func:`estimate_race_shock`
   maximises the exact marginal likelihood by Gauss-Hermite quadrature, holding
   :math:`\\eta` fixed.  So you keep whichever model won the comparison and add
   the correlation structure on top.

:class:`RaceRandomEffectGLMM` fits the whole thing jointly through statsmodels
instead, which is slower and less flexible about the mean model but gives
interpretable variance components and per-race posterior shocks — useful for
asking *which* races were the chaotic ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit, logit

from src import config

log = logging.getLogger(__name__)

#: Quadrature nodes for integrating over the race shock.  Twenty is ample for a
#: one-dimensional Gaussian integral and keeps the likelihood cheap.
N_QUADRATURE_NODES = 20


def _gauss_hermite(n: int = N_QUADRATURE_NODES) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for integrating against a standard normal density.

    ``numpy``'s Hermite rule integrates against ``exp(-x^2)``; rescaling by
    ``sqrt(2)`` and dividing the weights by ``sqrt(pi)`` converts it to an
    expectation under ``N(0, 1)``, so ``sum(w) == 1``.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return nodes * np.sqrt(2.0), weights / np.sqrt(np.pi)


# --------------------------------------------------------------------------- #
# Estimating the shock
# --------------------------------------------------------------------------- #


def _race_log_likelihood(
    sigma: float,
    y: np.ndarray,
    eta: np.ndarray,
    race_index: np.ndarray,
    n_races: int,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Exact marginal log-likelihood of the random-intercept model.

    For each race the Bernoulli likelihood is integrated over the shock:

        L_i = sum_k w_k * prod_j Bernoulli(y_ij | expit(eta_ij + sigma * z_k))

    Products are accumulated in log space and combined with ``logsumexp`` so a
    race with twenty drivers does not underflow.
    """
    from scipy.special import logsumexp

    shift = sigma * nodes                      # (K,)
    linear = eta[:, None] + shift[None, :]     # (N, K)
    # log Bernoulli pmf, numerically stable via -log1p(exp(-|x|)) style terms.
    log_p = -np.logaddexp(0.0, -linear)        # log(expit(x))
    log_q = -np.logaddexp(0.0, linear)         # log(1 - expit(x))
    per_row = np.where(y[:, None] == 1, log_p, log_q)

    per_race = np.zeros((n_races, len(nodes)))
    np.add.at(per_race, race_index, per_row)

    return float(np.sum(logsumexp(per_race + np.log(weights)[None, :], axis=1)))


@dataclass
class RaceShock:
    """The fitted race-level shock and what it implies."""

    sigma: float
    """Standard deviation of the per-race intercept, on the logit scale."""

    log_likelihood: float
    log_likelihood_null: float
    """Log-likelihood at ``sigma = 0``, i.e. assuming independence."""

    n_races: int
    n_rows: int

    @property
    def likelihood_ratio(self) -> float:
        """Twice the log-likelihood gain from allowing a shock.

        The usual reference is a 50:50 mixture of chi-squared with 0 and 1
        degrees of freedom, since the variance is bounded below at zero, which
        puts the 5% critical value at about 2.71.

        Treat that as a rough guide rather than an exact test.  Simulating from
        the null at F1-like sample sizes rejected nearer 15% of the time than
        5%, so a marginal value is weak evidence.  Two better checks, in order:
        the effect size — an
        :attr:`intraclass_correlation` below roughly 0.01 is not worth
        modelling however significant it looks — and
        :func:`check_attrition_calibration`, which compares the simulated
        spread of per-race retirement counts against the observed one and is a
        direct test of the thing you actually care about.
        """
        return 2.0 * (self.log_likelihood - self.log_likelihood_null)

    @property
    def intraclass_correlation(self) -> float:
        """Share of latent variance attributable to the race.

        On the logit scale the residual variance is fixed at
        :math:`\\pi^2/3 \\approx 3.29`, which gives the standard
        ``sigma^2 / (sigma^2 + pi^2/3)``.  A value of 0.05 means 5% of the
        latent variation in retirement risk is common to everyone in the race.
        """
        return self.sigma**2 / (self.sigma**2 + np.pi**2 / 3.0)

    def summary(self) -> str:
        return (
            f"race shock sigma = {self.sigma:.4f} (logit scale)\n"
            f"  intraclass correlation : {self.intraclass_correlation:.4f}\n"
            f"  likelihood ratio vs independence: {self.likelihood_ratio:.2f} "
            f"(a rough guide; ~2.7 is the nominal 5% bar, but the test runs "
            f"anti-conservative at this sample size)\n"
            f"  fitted on {self.n_rows} rows across {self.n_races} races"
        )


def estimate_race_shock(
    y: Sequence[int],
    probabilities: Sequence[float],
    race_ids: Sequence[Any],
    *,
    max_sigma: float = 3.0,
    n_nodes: int = N_QUADRATURE_NODES,
) -> RaceShock:
    """Estimate the per-race shared shock from any model's predictions.

    The mean model is taken as given: its predicted probabilities become the
    linear predictor, and only ``sigma`` is fitted.  That is what lets this sit
    on top of whichever model won the comparison instead of replacing it.

    Args:
        y: Observed outcomes, 1 for a retirement.
        probabilities: Predicted marginal probabilities from the mean model.
            **Use out-of-sample predictions.**  In-sample predictions are too
            close to the outcomes and will estimate the shock toward zero.
        race_ids: Race identifier per row; rows sharing one are one event.
        max_sigma: Upper bound for the search.
        n_nodes: Gauss-Hermite nodes.

    Returns:
        A :class:`RaceShock`.

    Note:
        The estimate absorbs any residual overdispersion, not only genuine
        shared shocks — an underfitted mean model will inflate it.  Fit the mean
        model properly first, then read this as "what remains that is common
        within a race".
    """
    y_arr = np.asarray(list(y), dtype=float)
    p = np.clip(np.asarray(list(probabilities), dtype=float), 1e-6, 1 - 1e-6)

    codes, _ = pd.factorize(pd.Series(list(race_ids)))
    race_index = np.asarray(codes, dtype=int)
    n_races = int(race_index.max()) + 1 if len(race_index) else 0

    nodes, weights = _gauss_hermite(n_nodes)

    def negative_ll(sigma: float) -> float:
        # The supplied probabilities are *marginal*, but the likelihood needs
        # the *conditional* linear predictor, and the map between them depends
        # on sigma.  Recomputing it inside the search keeps the estimator
        # self-consistent; holding eta fixed at logit(p) instead leaves a
        # persistent downward bias of a few percent that does not shrink with
        # sample size, because the mean model is then systematically flatter
        # than the one the likelihood assumes.
        if sigma <= 1e-9:
            eta = logit(p)
            return -_race_log_likelihood(
                0.0, y_arr, eta, race_index, n_races, nodes, weights
            )
        eta = marginal_to_conditional(p, sigma, n_nodes=n_nodes)
        return -_race_log_likelihood(
            sigma, y_arr, eta, race_index, n_races, nodes, weights
        )

    result = optimize.minimize_scalar(
        negative_ll, bounds=(1e-6, max_sigma), method="bounded"
    )
    sigma = float(result.x) if result.success else 0.0
    # A boundary solution at ~0 means independence fits at least as well.
    if sigma < 1e-4:
        sigma = 0.0

    return RaceShock(
        sigma=sigma,
        log_likelihood=-negative_ll(sigma),
        log_likelihood_null=-negative_ll(0.0),
        n_races=n_races,
        n_rows=len(y_arr),
    )


# --------------------------------------------------------------------------- #
# Marginal <-> conditional
# --------------------------------------------------------------------------- #


def conditional_to_marginal(
    eta: np.ndarray, sigma: float, *, n_nodes: int = N_QUADRATURE_NODES
) -> np.ndarray:
    """Integrate a conditional linear predictor over the shock.

    ``E_u[expit(eta + u)]`` with ``u ~ N(0, sigma^2)``.
    """
    if sigma <= 0:
        return expit(eta)
    nodes, weights = _gauss_hermite(n_nodes)
    return expit(eta[:, None] + sigma * nodes[None, :]) @ weights


def marginal_to_conditional(
    marginal: Sequence[float],
    sigma: float,
    *,
    n_nodes: int = N_QUADRATURE_NODES,
    tol: float = 1e-10,
    max_iter: int = 60,
) -> np.ndarray:
    """Find the conditional linear predictor that reproduces a marginal.

    Adding a mean-zero shock inside a logit shifts the marginal probability,
    because ``expit`` is convex below 0.5 and concave above.  A model calibrated
    on observed data predicts the *marginal*; simulating with a shock therefore
    needs the ``eta`` satisfying

        E_u[expit(eta + u)] = p_marginal

    which is solved here by Newton iteration on the quadrature integral.  This
    is exact to tolerance, unlike the common
    ``eta * sqrt(1 + 0.346 * sigma^2)`` approximation, which is a probit
    matching argument and drifts at large sigma or extreme probabilities.

    Getting this wrong is not cosmetic: it biases every simulated probability
    toward 0.5, which for a 14% event means systematically over-predicting
    retirements.
    """
    p = np.clip(np.asarray(list(marginal), dtype=float), 1e-9, 1 - 1e-9)
    if sigma <= 0:
        return logit(p)

    nodes, weights = _gauss_hermite(n_nodes)
    eta = logit(p) * np.sqrt(1 + 0.346 * sigma**2)  # good starting point

    for _ in range(max_iter):
        z = eta[:, None] + sigma * nodes[None, :]
        probs = expit(z)
        value = probs @ weights - p
        derivative = (probs * (1 - probs)) @ weights
        step = value / np.maximum(derivative, 1e-12)
        eta = eta - step
        if np.max(np.abs(value)) < tol:
            break
    return eta


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def simulate_retirements(
    marginal: Sequence[float],
    race_ids: Sequence[Any],
    sigma: float,
    *,
    n_sims: int = 2000,
    seed: int = config.RANDOM_SEED,
    preserve_marginal: bool = True,
) -> np.ndarray:
    """Draw correlated retirement outcomes, one column per simulation.

    Conditional on the race shock the drivers are independent, so each
    simulation draws one shock per race and then one Bernoulli per driver.  The
    result reproduces each driver's marginal probability while giving a
    realistic spread in how many cars retire.

    Args:
        marginal: Per-driver marginal retirement probabilities.
        race_ids: Race identifier per row.
        sigma: Race shock, from :func:`estimate_race_shock`.
        n_sims: Number of simulations.
        preserve_marginal: Solve for the conditional predictor first, so the
            simulated marginals match the input.  Leave this on unless you are
            deliberately supplying an already-conditional predictor.

    Returns:
        Array of shape ``(len(marginal), n_sims)`` of 0/1 outcomes.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(list(marginal), dtype=float)
    codes, _ = pd.factorize(pd.Series(list(race_ids)))
    race_index = np.asarray(codes, dtype=int)
    n_races = int(race_index.max()) + 1 if len(race_index) else 0

    eta = marginal_to_conditional(p, sigma) if preserve_marginal else logit(
        np.clip(p, 1e-9, 1 - 1e-9)
    )

    if sigma > 0:
        shocks = rng.normal(0.0, sigma, size=(n_races, n_sims))[race_index]
    else:
        shocks = np.zeros((len(p), n_sims))

    probabilities = expit(eta[:, None] + shocks)
    return (rng.random(probabilities.shape) < probabilities).astype(np.int8)


def finisher_count_distribution(
    marginal: Sequence[float],
    race_ids: Sequence[Any],
    sigma: float,
    **kwargs,
) -> pd.DataFrame:
    """Per-race distribution of how many cars retire, with and without the shock.

    The comparison is the point.  Both columns have the same mean by
    construction; the independent one has a much narrower spread, and that
    missing spread is exactly the "chaotic race" scenario a points model needs
    in order to produce a midfield driver's big score.
    """
    draws_correlated = simulate_retirements(marginal, race_ids, sigma, **kwargs)
    draws_independent = simulate_retirements(marginal, race_ids, 0.0, **kwargs)

    codes, labels = pd.factorize(pd.Series(list(race_ids)))
    race_index = np.asarray(codes, dtype=int)
    n_races = int(race_index.max()) + 1

    def per_race(draws: np.ndarray) -> np.ndarray:
        totals = np.zeros((n_races, draws.shape[1]))
        np.add.at(totals, race_index, draws)
        return totals

    correlated = per_race(draws_correlated)
    independent = per_race(draws_independent)

    return pd.DataFrame(
        {
            "race": labels,
            "expected_retirements": correlated.mean(axis=1).round(3),
            "sd_correlated": correlated.std(axis=1).round(3),
            "sd_independent": independent.std(axis=1).round(3),
            "p95_correlated": np.percentile(correlated, 95, axis=1),
            "p95_independent": np.percentile(independent, 95, axis=1),
        }
    )


def check_attrition_calibration(
    y: Sequence[int],
    marginal: Sequence[float],
    race_ids: Sequence[Any],
    sigma: float,
    *,
    n_sims: int = 4000,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Does the fitted shock reproduce the observed spread of retirements?

    This is the diagnostic to trust.  A likelihood-ratio test on a variance
    bounded at zero is awkward to calibrate; this instead compares the
    simulated distribution of per-race retirement counts against the one
    actually observed, which is the quantity a points model depends on.

    Read the ``observed`` row against ``sigma_fitted``.  If the fitted standard
    deviation is close to the observed one and the independent column is
    visibly too narrow, the shock is doing its job.  If ``sigma_fitted``
    overshoots, the mean model is probably underfitted and the shock is
    absorbing structure that belongs in the features.

    Returns:
        One row per scenario (observed, independent, fitted) with the mean,
        standard deviation and upper quantiles of the per-race retirement count.
    """
    y_arr = np.asarray(list(y), dtype=float)
    codes, _ = pd.factorize(pd.Series(list(race_ids)))
    race_index = np.asarray(codes, dtype=int)
    n_races = int(race_index.max()) + 1 if len(race_index) else 0

    observed_counts = np.zeros(n_races)
    np.add.at(observed_counts, race_index, y_arr)

    rows = [
        {
            "scenario": "observed",
            "mean": observed_counts.mean(),
            "sd": observed_counts.std(),
            "p90": np.percentile(observed_counts, 90),
            "max": observed_counts.max(),
        }
    ]
    for label, s in (("independent", 0.0), ("sigma_fitted", sigma)):
        draws = simulate_retirements(
            marginal, race_ids, s, n_sims=n_sims, seed=seed
        )
        totals = np.zeros((n_races, draws.shape[1]))
        np.add.at(totals, race_index, draws)
        # Spread across races, averaged over simulations: comparable with the
        # single observed realisation.
        rows.append(
            {
                "scenario": label,
                "mean": totals.mean(),
                "sd": float(totals.std(axis=0).mean()),
                "p90": float(np.percentile(totals, 90)),
                "max": float(totals.max(axis=0).mean()),
            }
        )
    return pd.DataFrame(rows).round(3)


# --------------------------------------------------------------------------- #
# Joint GLMM
# --------------------------------------------------------------------------- #


@dataclass
class GLMMResult:
    """Fitted variance components and per-race posterior shocks."""

    variance_components: pd.DataFrame
    race_effects: pd.DataFrame
    fixed_effects: pd.DataFrame
    model: Any

    def most_chaotic(self, n: int = 10) -> pd.DataFrame:
        """Races whose shock was most positive: more retirements than the
        features alone explain.  A useful sanity check — these should be the
        wet ones and the ones with a first-lap pile-up."""
        return self.race_effects.nlargest(n, "effect")


class RaceRandomEffectGLMM:
    """Logistic mixed model with a random intercept per race, via statsmodels.

    Where :func:`estimate_race_shock` bolts a shock onto an existing model, this
    fits the mean model and the variance components together.  It is slower and
    restricted to a linear mean, but it gives per-race posterior shocks, so you
    can ask *which* races were chaotic rather than only how chaotic races are on
    average.

    Uses ``BinomialBayesMixedGLM`` with variational Bayes.  That is fast and
    stable, but variational methods understate posterior variance, so treat the
    reported uncertainty as a lower bound.  For a properly calibrated posterior,
    fit the same model in PyMC.
    """

    def __init__(
        self,
        features: Sequence[str],
        *,
        race_col: str = "race_id",
        target: str = "dnf",
        standardise: bool = True,
    ) -> None:
        self.features = list(features)
        self.race_col = race_col
        self.target = target
        self.standardise = standardise
        self._means: pd.Series | None = None
        self._stds: pd.Series | None = None

    def _design(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        block = frame[self.features].apply(pd.to_numeric, errors="coerce")
        if fit:
            self._means = block.median()
        block = block.fillna(self._means)
        if self.standardise:
            if fit:
                self._stds = block.std(ddof=0).replace(0.0, 1.0)
            block = (block - self._means) / self._stds
        return block

    def fit(self, frame: pd.DataFrame, **kwargs) -> GLMMResult:
        """Fit the model.  ``frame`` needs the features, target and race column."""
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        design = self._design(frame, fit=True)
        design.insert(0, "Intercept", 1.0)
        y = frame[self.target].astype(float).to_numpy()

        races = pd.Categorical(frame[self.race_col])
        exog_vc = pd.get_dummies(races, dtype=float)
        ident = np.zeros(exog_vc.shape[1], dtype=int)

        model = BinomialBayesMixedGLM(
            endog=y,
            exog=design.to_numpy(dtype=float),
            exog_vc=exog_vc.to_numpy(dtype=float),
            ident=ident,
            vcp_p=1.0,
            fe_p=2.0,
        )
        fitted = model.fit_vb(verbose=False, **kwargs)

        n_fixed = design.shape[1]
        fixed = pd.DataFrame(
            {
                "term": design.columns,
                "coefficient": fitted.fe_mean,
                "sd": fitted.fe_sd,
            }
        ).sort_values("coefficient", key=np.abs, ascending=False, ignore_index=True)

        race_effects = pd.DataFrame(
            {
                "race": races.categories,
                "effect": fitted.vc_mean,
                "sd": fitted.vc_sd,
            }
        )

        # vcp_mean is the log standard deviation of the variance component.
        sigma = float(np.exp(fitted.vcp_mean[0])) if len(fitted.vcp_mean) else 0.0
        components = pd.DataFrame(
            [
                {
                    "component": "race",
                    "sigma": sigma,
                    "variance": sigma**2,
                    "intraclass_correlation": sigma**2 / (sigma**2 + np.pi**2 / 3),
                }
            ]
        )
        log.info("GLMM fitted: race sigma = %.4f", sigma)
        return GLMMResult(
            variance_components=components,
            race_effects=race_effects,
            fixed_effects=fixed,
            model=fitted,
        )
