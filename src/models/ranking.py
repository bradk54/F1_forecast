"""Plackett-Luce: a probability distribution over the finishing order.

**Why not predict each driver's position?**  Because a race result is a
permutation.  Twenty independent per-driver predictions assign P1 to two cars
and P7 to none, and their marginals cannot be rolled up into a championship,
which depends on the joint distribution.  See
``References/finishing_order_models.md`` for the full argument; this module is
Method 1 of that brief.

The model gives each car a log-strength ``s_i = beta . x_i``.  The winner is
drawn with probability ``exp(s_i) / sum_j exp(s_j)``, removed, and the draw
repeats for second place, and so on:

    P(i_1 > i_2 > ... > i_m)  =  prod_k  exp(s_{i_k}) / sum_{t >= k} exp(s_{i_t})

Properties that make it the right first choice here:

* It is a proper distribution over orderings, so coherence holds by
  construction -- exactly one winner, positions that sum correctly.
* The log-likelihood is concave in ``beta``, so there is one optimum, no
  restarts and no seeds.
* Sampling is exact and costs one line: add Gumbel noise to the log-strengths
  and sort (the Gumbel-max trick).  That is what makes the season simulation in
  :mod:`src.models.season` cheap enough to run ten thousand times.

**Retirements are left out of the ranking, not ranked last.**  The ordering is
fitted on classified finishers only -- treatment (b) of the brief -- because a
car that retires on lap 3 from second place has told you nothing about its
pace.  Scoring it as last would drag fast-but-fragile cars toward the back and
double-count reliability, once here and once in the DNF model that sits in
front of this one.  The price, which must be stated wherever a strength is
reported: **a strength here is pace conditional on finishing, and nothing
else.**  A "driver ranking" read off these coefficients without the DNF model
attached will flatter a fast crasher.

**Truncation.**  ``truncate=k`` keeps only the first *k* factors of the
likelihood, so the fit spends its information on the part of the order that
decides points and not on who beat whom for fourteenth.  Evaluation always uses
k = 10, the points cliff, whatever the training truncation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from src.features.build_features import RACE_KEYS

#: Positions that score in a grand prix.  Every evaluation truncates here.
POINTS_POSITIONS = 10


# --------------------------------------------------------------------------- #
# Packing races into arrays
# --------------------------------------------------------------------------- #


@dataclass
class RaceBatch:
    """Races padded to a common width, each sorted into finishing order.

    Row ``r`` holds race ``r``'s ranked cars first, best first, then padding.
    Padding is masked out of every sum, so races of different sizes share one
    array and the likelihood is evaluated for all of them in a single pass.
    """

    X: np.ndarray            # (races, width, features)
    mask: np.ndarray         # (races, width), True for a real ranked car
    n_ranked: np.ndarray     # (races,)
    keys: pd.DataFrame       # (races, RACE_KEYS)


def pack_races(
    frame: pd.DataFrame, X: np.ndarray, order: pd.Series | np.ndarray
) -> RaceBatch:
    """Group rows into races and sort each race by ``order`` (1 = winner).

    Rows whose ``order`` is NaN are not part of the ranking and are dropped:
    under treatment (b) that is every retirement.
    """
    order = np.asarray(order, dtype=float)
    ranked = np.isfinite(order)
    keys = frame.loc[ranked, list(RACE_KEYS)].reset_index(drop=True)
    Xr, orr = X[ranked], order[ranked]

    race_codes, uniques = pd.factorize(pd.MultiIndex.from_frame(keys), sort=False)
    n_races = len(uniques)
    sort_idx = np.lexsort((orr, race_codes))
    race_sorted = race_codes[sort_idx]
    counts = np.bincount(race_sorted, minlength=n_races)
    width = int(counts.max()) if n_races else 0
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(len(sort_idx)) - starts[race_sorted]

    batch_X = np.zeros((n_races, width, X.shape[1]))
    mask = np.zeros((n_races, width), dtype=bool)
    batch_X[race_sorted, slot] = Xr[sort_idx]
    mask[race_sorted, slot] = True
    race_keys = pd.DataFrame(list(uniques), columns=list(RACE_KEYS))
    return RaceBatch(X=batch_X, mask=mask, n_ranked=counts, keys=race_keys)


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #


def _terms(n_ranked: np.ndarray, width: int, truncate: int | None) -> np.ndarray:
    """Which (race, position) factors enter the likelihood.

    The last finisher's factor is always ``exp(s)/exp(s) = 1`` and is skipped,
    so a race with *m* ranked cars contributes at most ``m - 1`` factors.
    """
    limit = n_ranked - 1
    if truncate is not None:
        limit = np.minimum(limit, truncate)
    return np.arange(width)[None, :] < limit[:, None]


def race_log_likelihood(
    strengths: np.ndarray, mask: np.ndarray, n_ranked: np.ndarray,
    truncate: int | None = POINTS_POSITIONS, alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Log-probability of each race's observed order, one value per race.

    Args:
        strengths: ``(races, width)`` log-strengths in finishing order.
        alpha: Per-position scales for the stagewise model; ``None`` is
            standard Plackett-Luce (every scale 1).
    """
    terms = _terms(n_ranked, strengths.shape[1], truncate)
    if alpha is None:
        s = np.where(mask, strengths, -np.inf)
        # log of sum_{t >= k} exp(s_t), accumulated from the back of the order.
        log_tail = np.logaddexp.accumulate(s[:, ::-1], axis=1)[:, ::-1]
        return np.where(terms, s - np.where(terms, log_tail, 0.0), 0.0).sum(axis=1)
    alpha = np.concatenate([alpha, np.full(max(0, strengths.shape[1] - len(alpha)),
                                           alpha[-1])])
    s = np.where(mask, strengths, 0.0)
    a, lse, _ = _stage_tensor(s, mask, alpha)
    return np.where(terms, a[None, :] * s - lse, 0.0).sum(axis=1)


def uniform_log_likelihood(
    n_ranked: np.ndarray, truncate: int | None = POINTS_POSITIONS
) -> np.ndarray:
    """The same quantity under a model that knows nothing: every order equally likely."""
    width = int(n_ranked.max()) if len(n_ranked) else 0
    terms = _terms(n_ranked, width, truncate)
    remaining = n_ranked[:, None] - np.arange(width)[None, :]
    with np.errstate(divide="ignore"):
        return -np.where(terms, np.log(np.maximum(remaining, 1)), 0.0).sum(axis=1)


def _objective(
    beta: np.ndarray, batch: RaceBatch, weight: np.ndarray, l2: float,
    truncate: int | None,
) -> tuple[float, np.ndarray]:
    """Penalised negative log-likelihood and its exact gradient."""
    X, mask = batch.X, batch.mask
    s = X @ beta
    s_masked = np.where(mask, s, -np.inf)
    top = np.max(s_masked, axis=1, keepdims=True)
    e = np.where(mask, np.exp(s_masked - top), 0.0)
    tail = np.cumsum(e[:, ::-1], axis=1)[:, ::-1]
    tail_x = np.cumsum((e[..., None] * X)[:, ::-1], axis=1)[:, ::-1]
    terms = _terms(batch.n_ranked, s.shape[1], truncate)
    safe_tail = np.where(terms, tail, 1.0)

    per_term = np.where(terms, s - (np.log(safe_tail) + top), 0.0)
    ll = (per_term.sum(axis=1) * weight).sum()

    expected_x = tail_x / safe_tail[..., None]
    grad_terms = np.where(terms[..., None], X - expected_x, 0.0)
    grad = (grad_terms.sum(axis=1) * weight[:, None]).sum(axis=0)

    return -ll + 0.5 * l2 * beta @ beta, -grad + l2 * beta


def stage_scales(eta: np.ndarray, width: int = 40) -> np.ndarray:
    """Expand free log-scales into one scale per finishing position.

    Position 1 is fixed at scale 1, which is what identifies the model: scaling
    every strength by *c* and every scale by *1/c* would otherwise describe the
    same distribution.  Positions beyond the last free one share its scale.
    """
    alpha = np.ones(width)
    if len(eta):
        free = np.exp(eta)
        alpha[1:len(free) + 1] = free
        alpha[len(free) + 1:] = free[-1]
    return alpha


def _stage_tensor(
    s: np.ndarray, mask: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every stage's softmax at once, as a ``(race, stage, car)`` tensor.

    Stage *j* chooses among the cars from position *j* onward at scale
    ``alpha_j``.  Laying all stages out together replaces a Python loop over
    twenty positions with a handful of array operations, which is what makes a
    few hundred walk-forwards affordable in the tuning.
    """
    width = s.shape[1]
    a = alpha[:width]
    later = np.arange(width)[None, :] >= np.arange(width)[:, None]      # (stage, car)
    valid = mask[:, None, :] & later[None, :, :]
    z = np.where(valid, a[None, :, None] * s[:, None, :], -np.inf)
    lse = logsumexp(z, axis=2)
    lse = np.where(np.isfinite(lse), lse, 0.0)
    p = np.where(valid, np.exp(z - lse[..., None]), 0.0)
    return a, lse, p


def _stagewise_objective(
    theta: np.ndarray, batch: RaceBatch, weight: np.ndarray, l2: float,
    truncate: int | None, n_scales: int, l2_scale: float,
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood of the stagewise model, and its exact gradient.

    Stage *j* -- the choice of who finishes in position j+1 among the cars not
    yet placed -- is a softmax over ``alpha_j * s``.  ``theta`` holds the
    coefficients and then the free log-scales ``eta``; ``alpha_j = exp(eta)``.
    The penalty on ``eta`` pulls the scales toward 1, i.e. toward standard
    Plackett-Luce, so the extra freedom has to be paid for by the data.
    """
    X, mask = batch.X, batch.mask
    n_features = X.shape[2]
    beta, eta = theta[:n_features], theta[n_features:]
    width = X.shape[1]
    alpha = stage_scales(eta, width)
    s = X @ beta
    a, lse, p = _stage_tensor(s, mask, alpha)
    live = _terms(batch.n_ranked, width, truncate) * weight[:, None]   # (race, stage)

    ll = (live * (a[None, :] * s - lse)).sum()
    expected_x = np.einsum("rjt,rtf->rjf", p, X)
    grad_beta = np.einsum("rj,rjf->f", live * a[None, :], X - expected_x)
    per_stage = (live * a[None, :] * (s - np.einsum("rjt,rt->rj", p, s))).sum(axis=0)
    # Stage j >= 1 belongs to free scale min(j, n) - 1; stage 0 is fixed at 1.
    owner = np.minimum(np.arange(1, width), len(eta)) - 1
    grad_eta = np.bincount(owner, weights=per_stage[1:], minlength=len(eta))

    nll = -ll + 0.5 * l2 * beta @ beta + 0.5 * l2_scale * eta @ eta
    grad = np.concatenate([-grad_beta + l2 * beta, -grad_eta + l2_scale * eta])
    return nll, grad


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass
class PlackettLuce:
    """Plackett-Luce with covariates, an L2 prior and optional truncation.

    Features are median-imputed and standardised on the training rows, so the
    penalty treats every coefficient alike.  ``l2`` is the precision of a
    Gaussian prior on the standardised coefficients -- an absolute amount of
    prior information, so a wider training window is regularised relatively
    less, which is what a prior should do.

    **Stagewise scales.**  With ``n_scales > 0`` each finishing position gets
    its own scale on the strengths: the winner is drawn from a softmax over
    ``s``, second place from a softmax over ``alpha_2 * s``, and so on, with
    ``alpha_1 = 1``.  Standard Plackett-Luce forces one scale on every
    position, and on this data that is measurably wrong: the front of a grand
    prix is far more predictable than the midfield, so a single scale is too
    timid about the winner and too sure about eighth.  Fitted freely on
    2019-2023 the scale falls to 0.64 for second place and 0.19 for tenth.
    The idea is Benter's (1994) correction for place and show pricing in
    horse racing, and his discounts for second and third (~0.81, ~0.65)
    happen to match an exponential fit to ours.

    Args:
        features: Columns forming ``x_i``.  No intercept: anything shared by
            every car in a race cancels out of the likelihood.
        l2: Prior precision on standardised coefficients.
        truncate: Keep only the first *k* likelihood factors per race;
            ``None`` fits the full order.
        n_scales: Free position scales beyond the winner's; positions past the
            last share its value.  0 is standard Plackett-Luce.
        l2_scale: Prior precision on the log-scales, pulling them toward 1.
    """

    features: Sequence[str]
    l2: float = 1.0
    truncate: int | None = None
    n_scales: int = 0
    l2_scale: float = 1.0
    coef_: np.ndarray = field(default=None, init=False, repr=False)
    alpha_: np.ndarray | None = field(default=None, init=False, repr=False)
    center_: np.ndarray = field(default=None, init=False, repr=False)
    scale_: np.ndarray = field(default=None, init=False, repr=False)
    fill_: np.ndarray = field(default=None, init=False, repr=False)

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[list(self.features)].to_numpy(dtype=float)
        raw = np.where(np.isnan(raw), self.fill_, raw)
        return (raw - self.center_) / self.scale_

    def fit(
        self,
        frame: pd.DataFrame,
        order: pd.Series,
        race_weight: pd.Series | None = None,
    ) -> "PlackettLuce":
        """Fit on ``frame``, where ``order`` gives each ranked car's position.

        Args:
            order: Finishing position among the ranked cars, NaN for a car not
                in the ranking (a retirement).
            race_weight: Optional weight per row, constant within a race --
                e.g. a time decay that lets recent races count for more.
        """
        raw = frame[list(self.features)].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            fill = np.nanmedian(raw, axis=0)
        self.fill_ = np.where(np.isnan(fill), 0.0, fill)
        filled = np.where(np.isnan(raw), self.fill_, raw)
        self.center_ = filled.mean(axis=0)
        scale = filled.std(axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)

        X = (filled - self.center_) / self.scale_
        batch = pack_races(frame, X, order)
        if race_weight is None:
            weight = np.ones(len(batch.n_ranked))
        else:
            per_race = (
                frame.assign(_w=np.asarray(race_weight, dtype=float))
                .groupby(list(RACE_KEYS), observed=True)["_w"].first()
            )
            weight = (
                batch.keys.merge(per_race.reset_index(), on=list(RACE_KEYS), how="left")
                ["_w"].fillna(1.0).to_numpy()
            )

        if self.n_scales:
            result = minimize(
                _stagewise_objective, np.zeros(X.shape[1] + self.n_scales),
                args=(batch, weight, self.l2, self.truncate, self.n_scales,
                      self.l2_scale),
                jac=True, method="L-BFGS-B",
            )
            self.coef_ = result.x[:X.shape[1]]
            self.alpha_ = stage_scales(result.x[X.shape[1]:])
        else:
            result = minimize(
                _objective, np.zeros(X.shape[1]),
                args=(batch, weight, self.l2, self.truncate),
                jac=True, method="L-BFGS-B",
            )
            self.coef_ = result.x
            self.alpha_ = None
        return self

    def strength(self, frame: pd.DataFrame) -> np.ndarray:
        """Log-strength ``beta . x`` for every row.  Higher is faster."""
        if self.coef_ is None:
            raise RuntimeError("fit the model first")
        return self._design(frame) @ self.coef_

    def coefficients(self) -> pd.Series:
        """Coefficients on the standardised features, largest effect first."""
        return (pd.Series(self.coef_, index=list(self.features))
                .sort_values(key=np.abs, ascending=False))


def score_orders(
    frame: pd.DataFrame, strength: np.ndarray, order: pd.Series,
    truncate: int | None = POINTS_POSITIONS,
    alphas: dict[tuple, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per-race log-likelihood of the observed order, and the uniform null.

    Args:
        alphas: Stage scales per race key, for races scored by a stagewise
            model; races not in it are scored as standard Plackett-Luce.
    """
    batch = pack_races(frame, np.asarray(strength, dtype=float)[:, None], order)
    s = batch.X[..., 0]
    out = batch.keys.copy()
    out["n_ranked"] = batch.n_ranked
    if not alphas:
        out["log_lik"] = race_log_likelihood(s, batch.mask, batch.n_ranked, truncate)
    else:
        ll = np.empty(len(out))
        for r, key in enumerate(out[list(RACE_KEYS)].itertuples(index=False, name=None)):
            ll[r] = race_log_likelihood(s[r:r + 1], batch.mask[r:r + 1],
                                        batch.n_ranked[r:r + 1], truncate,
                                        alphas.get(key))[0]
        out["log_lik"] = ll
    out["log_lik_uniform"] = uniform_log_likelihood(batch.n_ranked, truncate)
    return out


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def sample_finishing_positions(
    strength: np.ndarray,
    p_dnf: np.ndarray | None,
    n_samples: int,
    rng: np.random.Generator,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Draw whole race results: attrition first, then an order among survivors.

    This is the generative story the whole design rests on:

        P(result) = P(who survives) x P(order | survivors)

    Each sample draws every car's retirement independently from ``p_dnf``,
    then orders the survivors -- see :func:`sample_positions`.

    Returns:
        ``(n_samples, n_cars)`` integer positions, 1 for the winner, and 0 for
        a car that retired.  Positions count survivors only, as a
        classification does.
    """
    n_cars = len(strength)
    if p_dnf is not None:
        retired = rng.random((n_samples, n_cars)) < np.asarray(p_dnf)[None, :]
    else:
        retired = np.zeros((n_samples, n_cars), dtype=bool)
    return sample_positions(np.asarray(strength, dtype=float), retired, alpha, rng)


def sample_positions(
    strength: np.ndarray,
    retired: np.ndarray,
    alpha: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Order the surviving cars in every sample; retired cars get position 0.

    Standard Plackett-Luce is one sort: add Gumbel noise to the log-strengths
    and rank (the Gumbel-max trick, an exact draw).  Under stagewise scales the
    positions are drawn one at a time instead, each from a fresh Gumbel draw at
    its own scale -- still exact, twenty steps instead of one sort.

    Args:
        strength: ``(n_cars,)``, or ``(n_samples, n_cars)`` when every sample
            carries its own strengths -- as a season trial does, with its own
            pace offsets.
        retired: ``(n_samples, n_cars)``, True for a car that does not finish.
        alpha: Stage scales, ``(width,)`` or per sample ``(n_samples, width)``;
            ``None`` for standard Plackett-Luce.
    """
    n_samples, n_cars = retired.shape
    s = np.broadcast_to(np.asarray(strength, dtype=float), (n_samples, n_cars))
    rows = np.arange(n_samples)
    if alpha is None:
        noisy = np.where(retired, -np.inf, s + rng.gumbel(size=(n_samples, n_cars)))
        order = np.argsort(-noisy, axis=1)
        positions = np.empty_like(order)
        positions[rows[:, None], order] = np.arange(1, n_cars + 1)[None, :]
        return np.where(retired, 0, positions)

    scales = np.atleast_2d(np.asarray(alpha, dtype=float))
    placed = retired.copy()
    positions = np.zeros((n_samples, n_cars), dtype=int)
    survivors = (~retired).sum(axis=1)
    for j in range(n_cars):
        active = j < survivors
        if not active.any():
            break
        a = scales[:, min(j, scales.shape[1] - 1)][:, None]
        z = a * s + rng.gumbel(size=(n_samples, n_cars))
        z[placed] = -np.inf
        pick = np.argmax(z, axis=1)
        positions[rows[active], pick[active]] = j + 1
        placed[rows[active], pick[active]] = True
    return positions
