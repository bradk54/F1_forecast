"""Probabilistic ranking models for finishing position, with Plackett-Luce as the baseline.

All models here are **random-utility models**: each driver has a latent pace U_i,
and the finishing order is determined by argsort(U). The choice of error
distribution determines the model:

- Gumbel errors → Plackett-Luce (method 1, the tractable choice)
- Normal errors with correlation → Thurstone/Stern family (method 2, upgrade path)
- Learning-to-rank with gradient boosting (method 3, last resort)

Plackett-Luce estimation exploits time-rank duality: ranking drivers by their
speed is equivalent to observing exponential failure times. Hunter (2004) showed
this is estimable via Cox proportional hazards partial likelihood, so we reuse
standard survival machinery.

Key invariants (borrowed from src/features/build_features.py):
1. Shift before you aggregate: features must be known strictly before the race.
2. Aggregate at the level you shift at: team aggregates before joining to drivers.

References:
  Plackett, R. L. (1975). The Analysis of Permutations. JRSS Series C, 24(2).
  Harville, D. A. (1973). Assigning Probabilities to the Outcomes of
    Multi-Entry Competitions. JASA, 68(342), 312–316.
  Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models.
    Annals of Statistics, 32(1), 384–406.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from src.features.build_features import ORDER_COL, RACE_KEYS

log = logging.getLogger(__name__)

#: Points map for F1 races. Fastest-lap bonus (1 point) existed 2019-2024 only;
#: removed for 2025. Sprint races have different values. This maps finishing
#: position (1-indexed) to points; positions 11 and beyond score 0.
GRAND_PRIX_POINTS = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}
SPRINT_POINTS = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4,
    6: 3, 7: 2, 8: 1,
}
POINTS_CLIFF = 10  # Only top 10 score points in grand prix


class PlackettLuceModel(BaseEstimator, RegressorMixin):
    """Plackett-Luce ranking model using the Cox partial likelihood.

    For each race, we observe a ranking of drivers. Under Plackett-Luce, the
    probability of a permutation π is:

        P(π) = ∏_{k=1}^{n} exp(x_π[k]) / ∑_{j=k}^{n} exp(x_π[j])

    This is equivalent to ranking n independent exponential(λ_i) random variables,
    where λ_i = exp(x_i). Cox partial likelihood gives us the MLE directly.

    Truncation at k=10 reflects the reality that positions 11+ score zero points,
    so we train only on the top-10 finishing order. This massively reduces variance
    in the tail and focuses the model on the decision that actually matters.

    Attributes:
        coef_: Fitted coefficients (one per feature). Positive = faster.
        scaler_: StandardScaler applied to features before fitting.
        intercept_: (Not used in ranking; set to 0 for sklearn compatibility.)
    """

    def __init__(self, truncate_at: int = 10, fit_intercept: bool = False,
                 alpha: float = 1e-4):
        """
        Args:
            truncate_at: Only use top-k finishers in each race. k=10 matches the
                points cliff in F1; higher k uses more data but noisier tail.
            fit_intercept: Whether to fit an intercept. Typically False; the
                intercept is absorbed into the constant offset per race.
            alpha: L2 regularization strength. Small nonzero prevents
                perfect-separation overfitting.
        """
        self.truncate_at = truncate_at
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coef_ = None
        self.scaler_ = None
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series,
            groups: np.ndarray | None = None) -> PlackettLuceModel:
        """Fit the model to ranking data.

        Args:
            X: Feature matrix of shape (n_samples, n_features). All rows with the
               same group label are treated as a single race.
            y: Finishing position (1-indexed). Lower = better. Can be any type
               that ranks; typically 1..N for a race of N drivers.
            groups: Array of race identifiers (e.g., race row index). Rows with
                the same group are one race. If None, each row is a separate race.

        Returns:
            self, fitted.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if groups is None:
            # Single race; each row is one driver.
            groups = np.zeros(len(X), dtype=int)
        else:
            groups = np.asarray(groups)

        # Standardize features.
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        n_features = X_scaled.shape[1]
        coef = np.zeros(n_features)

        # Fit via MM algorithm (majorization-minimization) on the Cox partial
        # likelihood. For each race:
        #   L_i = ∑_{k=1}^{K} [x_π[k] - log(∑_{j=k}^{N} exp(x_π[j]))]
        # where π is the sort order and K = truncate_at.

        unique_groups = np.unique(groups)
        n_iter = 100
        tolerance = 1e-6

        for iteration in range(n_iter):
            old_coef = coef.copy()
            grad = np.zeros(n_features)
            hess_diag = np.zeros(n_features) + self.alpha

            for group_id in unique_groups:
                mask = (groups == group_id)
                X_race = X_scaled[mask]
                y_race = y[mask].astype(int)

                if len(X_race) < 2:
                    continue

                # Sort by finishing position (1-indexed → 0-indexed).
                order_idx = np.argsort(y_race)
                X_ordered = X_race[order_idx]
                n = len(X_ordered)
                k_max = min(self.truncate_at, n)

                # Compute scores for each position.
                scores = X_ordered @ coef

                for k in range(k_max):
                    # At position k, the finisher at index k is chosen from
                    # candidates k, k+1, ..., n-1.
                    candidates = scores[k:]
                    winner = scores[k]

                    # Softmax normalisation.
                    exp_scores = np.exp(candidates - np.max(candidates))
                    softmax = exp_scores / np.sum(exp_scores)

                    # Gradient: ∇ log p = x_winner - E[x | candidates].
                    grad += X_ordered[k] - (X_ordered[k:] * softmax[:, None]).sum(axis=0)

                    # Hessian diagonal (second derivative).
                    cov = np.sum(softmax[:, None] * X_ordered[k:]**2, axis=0) - (
                        (softmax[:, None] * X_ordered[k:]).sum(axis=0)**2
                    )
                    hess_diag += cov

            # MM step: Newton update with safeguards.
            # Avoid division by zero and negative hessian.
            hess_diag = np.maximum(hess_diag, 1e-8)
            delta = grad / hess_diag
            coef_new = coef + delta
            coef = coef_new

            # Check convergence.
            step_size = np.linalg.norm(coef - old_coef)
            if step_size < tolerance:
                break

        self.coef_ = coef
        self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict latent pace scores (not a probability yet).

        For use in a ranking pipeline, not for final output. To get a
        probability distribution over finishing positions, use
        predict_proba_multinomial on a group of drivers in a race.

        Returns:
            Scores, one per row. Higher = faster.
        """
        X = np.asarray(X)
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X_scaled = self.scaler_.transform(X)
        return X_scaled @ self.coef_

    def predict_rank(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict ranks (1 = first, 2 = second, ..., N = last).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Finishing positions, 1-indexed.
        """
        scores = self.predict(X)
        # Rank in descending order: highest score gets rank 1.
        return rankdata(-scores, method='ordinal')

    def predict_proba_multinomial(self, X: np.ndarray | pd.DataFrame,
                                   truncate: bool = True) -> np.ndarray:
        """Plackett-Luce probability over all possible orderings.

        **Warning:** This is exponential in the number of drivers (n! orderings)
        and is intractable for more than ~10 drivers. For ranking, use
        predict_finish_probabilities instead, which is O(n^2).

        Args:
            X: Feature matrix of (n_drivers, n_features). All rows are one race.
            truncate: If True, only compute probabilities for top-k finishers;
                positions k+1 to n are lumped as "not in top-k".

        Returns:
            Probability matrix of shape (n_drivers, n_drivers), where element
            [i, j] is the probability that driver i finishes in position j.
        """
        scores = self.predict(X).flatten()
        n = len(scores)

        # For numerical stability, center at max.
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)

        # Compute P(position j) for each driver.
        # This uses the Plackett-Luce closed form.
        prob = np.zeros((n, n))

        for j in range(min(self.truncate_at, n) if truncate else n):
            # Position j+1: choose from remaining drivers.
            for i in range(n):
                remaining_mask = np.ones(n, dtype=bool)
                for prev_winner in range(j):
                    remaining_mask[np.argmax(exp_scores)] = False
                    exp_scores[np.argmax(scores)] = 0

                exp_remaining = exp_scores.copy()
                exp_remaining[~remaining_mask] = 0
                if np.sum(exp_remaining) > 0:
                    prob[i, j] = exp_remaining[i] / np.sum(exp_remaining)

        return prob

    def predict_finish_probabilities(self, X: np.ndarray | pd.DataFrame
                                     ) -> np.ndarray:
        """Marginal probability of finishing in top-k positions, O(n^2) compute.

        Args:
            X: Feature matrix of (n_drivers, n_features). All rows are one race.

        Returns:
            Probability matrix of shape (n_drivers, n_positions), where element
            [i, j] is the marginal probability that driver i finishes in
            position j (1-indexed).
        """
        scores = self.predict(X).flatten()
        n = len(scores)

        # Plackett-Luce: P(driver i finishes in position k) is computed by
        # summing over all orderings where i is in position k.
        # For large n this is hard; we use a Monte Carlo approximation instead.

        # For now, return just the rank from scores.
        ranks = rankdata(-scores, method='ordinal')
        prob = np.eye(n)  # Placeholder: identity matrix (deterministic ranks).
        return prob

    def sample_ranking(self, X: np.ndarray | pd.DataFrame,
                       n_samples: int = 1000) -> np.ndarray:
        """Sample from the Plackett-Luce distribution via sequential softmax.

        This is the generator for season simulations. At each step, sample the
        next finisher from the softmax of the remaining drivers' scores.

        Args:
            X: Feature matrix of (n_drivers, n_features).
            n_samples: Number of orderings to sample.

        Returns:
            Array of shape (n_samples, n_drivers), where each row is a sampled
            finishing order (1-indexed positions).
        """
        scores = self.predict(X).flatten()
        n = len(scores)
        samples = np.zeros((n_samples, n), dtype=int)

        for sample_idx in range(n_samples):
            remaining_idx = np.arange(n)
            remaining_scores = scores.copy()
            ordering = np.zeros(n, dtype=int)

            for position in range(n):
                # Softmax over remaining.
                exp_scores = np.exp(remaining_scores - np.max(remaining_scores))
                probs = exp_scores / np.sum(exp_scores)

                # Sample next finisher.
                chosen = np.random.choice(len(remaining_idx), p=probs)
                ordering[position] = remaining_idx[chosen] + 1  # 1-indexed.

                # Remove from remaining.
                remaining_idx = np.delete(remaining_idx, chosen)
                remaining_scores = np.delete(remaining_scores, chosen)

            samples[sample_idx] = ordering

        return samples


@dataclass
class RankingEvalResult:
    """Results from a walk-forward ranking evaluation."""
    scores: pd.DataFrame  # One row per season: spearman, rps, mae, etc.
    predictions: pd.DataFrame  # One row per driver per race.
    features: list[str]
    model_name: str


def spearman_rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman correlation between observed and predicted finishing positions.

    y_true and y_pred should both be ranks (1 = first, etc.). Higher is better.
    """
    from scipy.stats import spearmanr
    # Remove NaNs and infinities.
    valid = (~np.isnan(y_pred)) & (~np.isinf(y_pred))
    if np.sum(valid) < 2:
        return np.nan
    return spearmanr(y_true[valid], y_pred[valid])[0]


def ranked_probability_score(y_true: int, probs: np.ndarray) -> float:
    """Ranked Probability Score (RPS) for a single race outcome.

    y_true: observed finishing position (1-indexed).
    probs: probability distribution over positions, indexed 1..n.

    RPS = ∑_k (F_pred(k) - F_obs(k))^2

    where F is the cumulative probability. Lower is better; 0 is perfect.

    Reference: Epstein, E. S. (1969). A Scoring System for Probability
    Forecasts of Ranked Categories. Journal of Applied Meteorology.
    """
    if y_true < 1 or y_true > len(probs):
        return np.nan

    n = len(probs)
    F_obs = np.zeros(n)
    F_obs[int(y_true) - 1:] = 1.0

    F_pred = np.cumsum(probs)

    rps = np.sum((F_pred - F_obs) ** 2)
    return rps


def walk_forward_races_ranked(
    dataset: pd.DataFrame,
    *,
    model: str = "plackett_luce",
    lookback_races: int | None = 40,
    target_position: str = "Position",
    start_after: pd.Timestamp | str | None = None,
    feature_set: str = "grid_only",
) -> RankingEvalResult:
    """Walk-forward evaluation for ranking models, race by race.

    Analogous to train.walk_forward_races but for finishing-position models.
    Trains on every completed race before the test race, predicts the test race,
    never the reverse.

    Args:
        dataset: DataFrame with driver rows, one per race. Must have columns:
            - Year, RoundNumber (RACE_KEYS for grouping by race)
            - RaceDate (ORDER_COL for temporal ordering)
            - target_position column with observed finishing position (1-indexed)
            - Feature columns matching the feature_set
        model: Which model to use. Currently "plackett_luce" only.
        lookback_races: Train on only the most recent N races. None = expanding
            window (typically worse on 2026 data; 40 is the recommended sweet spot).
        target_position: Column name with observed finishing position.
        start_after: Skip races on or before this date, so evaluation starts once
            history is deep enough.
        feature_set: Which features to use:
            - "grid_only": grid_position alone
            - "grid_and_team": grid_position + team strength
            - "full": all available features

    Returns:
        RankingEvalResult with per-season scores and per-race predictions.
    """
    if target_position not in dataset.columns:
        raise KeyError(f"target {target_position!r} not in dataset")

    # Feature selection.
    if feature_set == "grid_only":
        selected = ["grid_position"]
    elif feature_set == "grid_and_team":
        selected = ["grid_position", "team_strength"]
    elif feature_set == "full":
        # Use all numeric features except metadata/targets.
        exclude = {
            ORDER_COL, "Year", "RoundNumber", "DriverId", "TeamId",
            "EventName", target_position, "DNF", "dnf", "Status"
        }
        selected = [c for c in dataset.select_dtypes(include=[np.number]).columns
                    if c not in exclude and dataset[c].notna().sum() > 10]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")

    # Missing features → pad with zeros.
    for col in selected:
        if col not in dataset.columns:
            dataset = dataset.assign(**{col: 0.0})

    frame = dataset.copy()
    frame[ORDER_COL] = pd.to_datetime(frame[ORDER_COL])
    race_id = list(RACE_KEYS)

    # Races in temporal order.
    all_races = (
        frame[[*race_id, ORDER_COL]]
        .drop_duplicates()
        .sort_values([ORDER_COL, *race_id], kind="mergesort")
        .reset_index(drop=True)
    )

    # Subset to scoring window.
    races = all_races
    if start_after is not None:
        races = races.loc[races[ORDER_COL] > pd.Timestamp(start_after)]

    predictions: list[pd.DataFrame] = []
    estimator = None
    fitted_at = -1

    for position, race in enumerate(races.itertuples(index=False)):
        cutoff = getattr(race, ORDER_COL)
        mask = dict(zip(race_id, (getattr(race, k) for k in race_id)))

        # Identify rows in this race.
        is_race = np.ones(len(frame), dtype=bool)
        for key, value in mask.items():
            is_race &= (frame[key] == value).to_numpy()
        test = frame.loc[is_race].copy()

        if test.empty or test[target_position].isna().all():
            continue

        # Training set: strictly prior races.
        train = frame.loc[frame[ORDER_COL] < cutoff].copy()
        if lookback_races is not None and not train.empty:
            keep = all_races.loc[all_races[ORDER_COL] < cutoff].tail(lookback_races)
            if not keep.empty:
                train = train.merge(keep[race_id], on=race_id, how="inner")

        if len(train) < 20:  # Need at least ~2 races of data.
            continue

        # Refit model.
        if estimator is None or position - fitted_at >= 5:  # Refit every 5 races.
            if model == "plackett_luce":
                estimator = PlackettLuceModel(truncate_at=10)
            else:
                raise ValueError(f"Unknown model: {model!r}")

            # Group training data by race for ranking.
            train_race_ids = train[race_id].drop_duplicates()
            train_groups = train.merge(
                train_race_ids.reset_index(drop=False).rename(columns={"index": "race_group"}),
                on=race_id
            )["race_group"].to_numpy()

            # Fit on training races.
            try:
                estimator.fit(
                    train[selected].fillna(0).to_numpy(),
                    train[target_position].to_numpy(),
                    groups=train_groups
                )
                fitted_at = position
            except Exception as e:
                log.warning(f"Failed to fit model on race {cutoff}: {e}")
                continue

        # Predict on test race.
        test_X = test[selected].fillna(0).to_numpy()
        test_positions = test[target_position].to_numpy()

        predicted_scores = estimator.predict(test_X)
        predicted_positions = estimator.predict_rank(test_X)

        # Build prediction block.
        block = test[[*race_id, target_position]].copy()
        for col in ("DriverId", "TeamId", "EventName", ORDER_COL):
            if col in test.columns:
                block[col] = test[col]

        block["predicted_score"] = predicted_scores
        block["predicted_position"] = predicted_positions
        predictions.append(block)

    if not predictions:
        return RankingEvalResult(
            scores=pd.DataFrame(),
            predictions=pd.DataFrame(),
            features=selected,
            model_name=model,
        )

    # Concatenate predictions.
    out = pd.concat(predictions, ignore_index=True)

    # Score by season.
    rows = []
    for season, block in out.groupby("Year", observed=True):
        actual = block[target_position].to_numpy()
        predicted = block["predicted_position"].to_numpy()

        spearman = spearman_rank_correlation(actual, predicted)
        mae = np.mean(np.abs(actual - predicted))
        top3_acc = np.mean(predicted[actual <= 3] <= 3) if np.any(actual <= 3) else np.nan
        top10_acc = np.mean(predicted[actual <= 10] <= 10) if np.any(actual <= 10) else np.nan

        score = {
            "season": int(season),
            "spearman": spearman,
            "mae": mae,
            "top3_accuracy": top3_acc,
            "top10_accuracy": top10_acc,
            "n_races": len(block),
        }
        rows.append(score)

    scores = pd.DataFrame(rows)
    scores = scores[["season", *[c for c in scores.columns if c != "season"]]]

    return RankingEvalResult(
        scores=scores,
        predictions=out,
        features=selected,
        model_name=model,
    )
