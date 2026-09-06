"""Baselines and honest evaluation for retirement prediction.

**Accuracy is the wrong metric here and will mislead you.**  Roughly one car in
seven retires from a modern grand prix, so a model that predicts "everyone
finishes" scores about 86% accuracy while carrying no information whatsoever.
Nothing in this module reports accuracy.  What it reports instead:

``brier``
    Mean squared error of the predicted probability.  Lower is better.
``brier_skill``
    The headline number.  How much the Brier score improves on always
    predicting the training-set base rate: ``1 - brier/brier_base``.  Zero
    means the model has learnt nothing beyond the base rate; negative means it
    is actively worse.  This is the only metric that answers "is this model
    worth anything".
``log_loss``
    Punishes confident mistakes harder than Brier does.
``roc_auc``
    Ranking quality: can the model order a retirement above a finish?  Useful
    for "who is most at risk this weekend" even when the probabilities are
    poorly calibrated.
``pr_auc``
    Average precision.  With a 14% positive rate this is far more informative
    than ROC-AUC, whose baseline is 0.5 regardless of class balance.  Compare
    it against the base rate, not against 0.5.
``calibration_slope``
    Slope of observed frequency against predicted probability, from a logistic
    recalibration.  1.0 is perfect; below 1 means the model is overconfident.

Evaluation is **walk-forward by season**: train on every season before the test
season, predict the test season, never the reverse.  A random split would let
2024 races inform a 2019 prediction, and the resulting scores would be fiction.

One caveat worth carrying: rows within a race are not independent.  A first-lap
pile-up retires four cars from one event, so the effective sample size is closer
to the number of races than the number of driver-races.  :func:`race_bootstrap`
resamples whole races rather than rows, which is the honest way to put an
interval around any of these numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.features import registry

log = logging.getLogger(__name__)

TARGET = "dnf"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def make_logistic(numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    """Regularised logistic regression: the interpretable reference point.

    Median imputation is deliberate.  A driver's first race genuinely has no
    rolling history, and the median is the least opinionated stand-in; pair it
    with the matching ``*_races_to_date`` count so the model can learn to
    distrust an imputed value.
    """
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    steps: list[tuple[str, Any, list[str]]] = [("num", numeric_pipe, list(numeric))]
    if categorical:
        steps.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                list(categorical),
            )
        )
    return Pipeline(
        [
            ("prep", ColumnTransformer(steps, remainder="drop")),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, C=1.0, class_weight=None,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def make_gradient_boosting(
    numeric: Sequence[str], categorical: Sequence[str]
) -> Pipeline:
    """Gradient boosting that takes NaN natively.

    ``HistGradientBoostingClassifier`` routes missing values down their own
    branch, so a cold-start row keeps its "we do not know" status instead of
    being silently imputed to the median.  That matters here: rookies and new
    teams are exactly the rows where risk is unusual.

    ``class_weight`` is left alone on purpose.  Re-weighting improves ranking
    metrics and destroys calibration, and a retirement model is only useful if
    its probabilities mean what they say.
    """
    steps: list[tuple[str, Any, list[str]]] = [("num", "passthrough", list(numeric))]
    if categorical:
        steps.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                list(categorical),
            )
        )
    return Pipeline(
        [
            ("prep", ColumnTransformer(steps, remainder="drop")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=25,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def _tree_preprocessor(
    numeric: Sequence[str], categorical: Sequence[str]
) -> ColumnTransformer:
    """Passthrough for numerics, one-hot for categoricals.

    Tree ensembles that handle NaN natively need no imputation, so the numeric
    block is passed straight through and missingness stays informative.
    """
    steps: list[tuple[str, Any, list[str]]] = [("num", "passthrough", list(numeric))]
    if categorical:
        steps.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), list(categorical))
        )
    return ColumnTransformer(steps, remainder="drop")


def make_xgboost(numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    """XGBoost, tuned conservatively for a small, noisy, rare-event problem.

    Roughly 4,000 driver-races with a 14% positive rate is not much signal, so
    the settings lean hard on regularisation: shallow trees, heavy subsampling,
    a large ``min_child_weight`` and explicit L1/L2 penalties.  Left at library
    defaults, XGBoost will happily memorise which driver retired at which
    circuit in 2019.

    ``scale_pos_weight`` is deliberately **not** set.  It improves ranking
    metrics and wrecks calibration by inflating every predicted probability,
    and the downstream points model needs probabilities that mean what they
    say — a P(finish) of 0.85 has to show up as a finish 85% of the time or
    expected points come out wrong.
    """
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("prep", _tree_preprocessor(numeric, categorical)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.03,
                    max_depth=3,
                    min_child_weight=20,
                    subsample=0.8,
                    colsample_bytree=0.6,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    gamma=0.1,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    missing=np.nan,
                    random_state=config.RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_lightgbm(numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    """LightGBM, the leaf-wise counterpart to XGBoost's depth-wise growth.

    Included because it disagrees with XGBoost in useful ways on small data:
    leaf-wise growth finds interactions sooner and overfits sooner, so
    ``num_leaves`` is kept small and ``min_child_samples`` large.  If the two
    land in the same place, that is evidence the signal is real rather than an
    artefact of one library's inductive bias.
    """
    from lightgbm import LGBMClassifier

    return Pipeline(
        [
            ("prep", _tree_preprocessor(numeric, categorical)),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.03,
                    num_leaves=8,
                    max_depth=4,
                    min_child_samples=30,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.6,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    random_state=config.RANDOM_SEED,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def make_random_forest(numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    """Random forest: a high-variance, low-bias contrast to the boosters.

    Forests average many deep trees rather than correcting a weak one, so they
    tend to be better calibrated out of the box and worse at ranking.  Useful
    as a third opinion.  Needs imputation — scikit-learn's forest has no NaN
    branch.
    """
    from sklearn.ensemble import RandomForestClassifier

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    steps: list[tuple[str, Any, list[str]]] = [("num", numeric_pipe, list(numeric))]
    if categorical:
        steps.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), list(categorical))
        )
    return Pipeline(
        [
            ("prep", ColumnTransformer(steps, remainder="drop")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    random_state=config.RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_calibrated(
    base: str, method: str = "isotonic", cv: int = 5
) -> "Callable[[Sequence[str], Sequence[str]], Pipeline]":
    """Wrap a model factory in cross-validated probability calibration.

    Boosted trees are usually overconfident at the extremes, which matters more
    here than anywhere: expected points is ``P(finish) * E[points | finish]``,
    so a probability that is wrong by 5 points is a points forecast that is
    wrong by 5%.

    ``isotonic`` is non-parametric and can fix any monotone distortion, but it
    needs data and will overfit a small calibration fold; ``sigmoid`` (Platt)
    fits two parameters and is the safer choice under a few thousand rows.  Try
    both — :func:`compare_models` does.

    The calibration folds come from the training seasons only, so this stays
    inside the walk-forward discipline.
    """

    def factory(numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
        from sklearn.calibration import CalibratedClassifierCV

        inner = MODEL_FACTORIES[base](numeric, categorical)
        return Pipeline([("calibrated", CalibratedClassifierCV(inner, method=method, cv=cv))])

    return factory


MODEL_FACTORIES: dict[str, Any] = {
    "logistic": make_logistic,
    "gradient_boosting": make_gradient_boosting,
    "xgboost": make_xgboost,
    "lightgbm": make_lightgbm,
    "random_forest": make_random_forest,
}

# Calibrated variants, registered after the base models they wrap.
MODEL_FACTORIES["xgboost_isotonic"] = make_calibrated("xgboost", "isotonic")
MODEL_FACTORIES["xgboost_sigmoid"] = make_calibrated("xgboost", "sigmoid")
MODEL_FACTORIES["gradient_boosting_sigmoid"] = make_calibrated(
    "gradient_boosting", "sigmoid"
)
MODEL_FACTORIES["lightgbm_sigmoid"] = make_calibrated("lightgbm", "sigmoid")

#: A sensible sweep for :func:`compare_models`, ordered simplest first.
DEFAULT_MODEL_SUITE = (
    "logistic",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "xgboost_sigmoid",
    "xgboost_isotonic",
    "gradient_boosting_sigmoid",
)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Slope of a logistic refit of outcomes on predicted log-odds.

    1.0 means the probabilities can be taken at face value.  Below 1 means the
    model is overconfident and its extremes should be shrunk toward the base
    rate; above 1 means it is underconfident.
    """
    prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit = np.log(prob / (1 - prob)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fitted = LogisticRegression(max_iter=1000).fit(logit, y_true)
    return float(fitted.coef_[0][0])


def score_predictions(
    y_true: Iterable[int], y_prob: Iterable[float], base_rate: float
) -> dict[str, float]:
    """Every metric that matters, plus the base-rate comparison.

    Args:
        base_rate: Positive rate of the *training* data.  Using the test set's
            own rate would be a subtle leak — you do not know it in advance.
    """
    y_true = np.asarray(list(y_true), dtype=float)
    y_prob = np.asarray(list(y_prob), dtype=float)
    if len(y_true) == 0:
        return {}

    brier = brier_score_loss(y_true, y_prob)
    brier_base = brier_score_loss(y_true, np.full_like(y_prob, base_rate))
    both_classes = len(np.unique(y_true)) > 1

    return {
        "n": float(len(y_true)),
        "observed_rate": float(y_true.mean()),
        "base_rate": float(base_rate),
        "brier": float(brier),
        "brier_base_rate": float(brier_base),
        "brier_skill": float(1 - brier / brier_base) if brier_base > 0 else np.nan,
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-9, 1 - 1e-9), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob))
        if both_classes
        else np.nan,
        "calibration_slope": calibration_slope(y_true, y_prob)
        if both_classes
        else np.nan,
        "mean_predicted": float(y_prob.mean()),
    }


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #


@dataclass
class WalkForwardResult:
    """Per-season scores plus the predictions that produced them."""

    scores: pd.DataFrame
    predictions: pd.DataFrame
    features: list[str] = field(default_factory=list)
    model_name: str = ""

    def summary(self) -> pd.Series:
        """Row-weighted mean across seasons, so a short season does not dominate."""
        numeric = self.scores.drop(columns=["season"], errors="ignore")
        weights = numeric["n"]
        weighted = numeric.drop(columns=["n"]).multiply(weights, axis=0).sum() / weights.sum()
        weighted["n"] = weights.sum()
        return weighted.round(4)


def walk_forward_evaluate(
    dataset: pd.DataFrame,
    *,
    stage: registry.Stage = "post_quali",
    model: str = "gradient_boosting",
    target: str = TARGET,
    season_col: str = "Year",
    test_seasons: Sequence[int] | None = None,
    min_train_rows: int = 200,
    extra_features: Sequence[str] = (),
    drop_features: Sequence[str] = (),
) -> WalkForwardResult:
    """Train on every prior season, score the next one, and repeat.

    Args:
        dataset: The built modelling table.
        stage: Latest information stage the model may use.  ``pre_weekend``
            excludes grid position; ``race_day`` includes observed weather and
            is retrospective only.
        model: Key into :data:`MODEL_FACTORIES`.
        test_seasons: Seasons to score.  Defaults to every season that has at
            least one prior season available.
        min_train_rows: Skip a fold whose training set is smaller than this.
        extra_features: Columns to add beyond the registry's selection.
        drop_features: Columns to remove, for ablation.

    Returns:
        A :class:`WalkForwardResult` carrying per-season scores and the
        out-of-sample predictions.
    """
    if target not in dataset.columns:
        raise KeyError(f"target {target!r} not in dataset")

    available = set(dataset.columns)
    selected = registry.feature_columns(stage, available=available)
    selected = [c for c in [*selected, *extra_features] if c in available]
    selected = [c for c in selected if c not in set(drop_features)]
    if not selected:
        raise ValueError(f"no registered features for stage {stage!r} in this dataset")

    numeric = [
        c for c in selected if registry.BY_NAME[c].kind in ("numeric", "binary")
    ] if all(c in registry.BY_NAME for c in selected) else [
        c for c in selected if pd.api.types.is_numeric_dtype(dataset[c])
    ]
    categorical = [c for c in selected if c not in numeric]

    seasons = sorted(dataset[season_col].dropna().unique())
    if test_seasons is None:
        test_seasons = seasons[1:]

    rows, predictions = [], []
    for season in test_seasons:
        train = dataset.loc[dataset[season_col] < season]
        test = dataset.loc[dataset[season_col] == season]
        if len(train) < min_train_rows or test.empty:
            log.info("skipping %s: train=%d test=%d", season, len(train), len(test))
            continue
        if train[target].nunique() < 2:
            log.info("skipping %s: training target has one class", season)
            continue

        estimator = MODEL_FACTORIES[model](numeric, categorical)
        estimator.fit(train[selected], train[target])
        probability = estimator.predict_proba(test[selected])[:, 1]

        base_rate = float(train[target].mean())
        score = score_predictions(test[target], probability, base_rate)
        score["season"] = int(season)
        rows.append(score)

        block = test[[season_col, "RoundNumber", target]].copy()
        for column in ("DriverId", "TeamId", "EventName"):
            if column in test.columns:
                block[column] = test[column]
        block["predicted"] = probability
        predictions.append(block)

    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores = scores[["season", *[c for c in scores.columns if c != "season"]]]
    return WalkForwardResult(
        scores=scores,
        predictions=pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(),
        features=selected,
        model_name=model,
    )


# --------------------------------------------------------------------------- #
# Ablation and importance
# --------------------------------------------------------------------------- #


def ablate_features(
    dataset: pd.DataFrame,
    groups: dict[str, Sequence[str]],
    *,
    metric: str = "brier_skill",
    **kwargs,
) -> pd.DataFrame:
    """Drop each group of features in turn and measure what it cost.

    Grouping matters more than it looks.  Dropping ``track_speed_index`` alone
    proves little when ``speed_mean_kph`` says nearly the same thing; drop the
    whole track block and the question becomes answerable.

    Returns:
        One row per group with the metric and its change from the full model.
        A negative ``delta`` means removing the group hurt, so the group was
        carrying weight.
    """
    baseline = walk_forward_evaluate(dataset, **kwargs)
    if baseline.scores.empty:
        raise ValueError("baseline evaluation produced no folds")
    base_value = float(baseline.summary()[metric])

    rows = [{"group": "(full model)", metric: round(base_value, 4), "delta": 0.0,
             "n_features": len(baseline.features)}]
    for name, columns in groups.items():
        result = walk_forward_evaluate(dataset, drop_features=list(columns), **kwargs)
        if result.scores.empty:
            continue
        value = float(result.summary()[metric])
        rows.append(
            {
                "group": name,
                metric: round(value, 4),
                "delta": round(value - base_value, 4),
                "n_features": len(result.features),
            }
        )
    return pd.DataFrame(rows).sort_values("delta", ignore_index=True)


def track_feature_group(dataset: pd.DataFrame) -> list[str]:
    """Every circuit-derived feature present, for a one-line ablation.

    Answers the question this dataset was built to ask: does measuring the
    circuit beat ignoring it?
    """
    track_names = {
        f.name
        for f in registry.FEATURES
        if any(
            token in f.name
            for token in (
                "track_", "speed_", "corner", "braking", "lat_g", "decel_g",
                "curvature", "elevation", "lap_length", "throttle", "drs",
                "gear", "rpm", "gradient", "street", "night", "altitude",
                "runoff", "mechanical_stress", "incident_exposure",
            )
        )
    }
    return sorted(track_names & set(dataset.columns))


def permutation_importance_report(
    dataset: pd.DataFrame,
    *,
    stage: registry.Stage = "post_quali",
    model: str = "gradient_boosting",
    test_season: int | None = None,
    n_repeats: int = 10,
    target: str = TARGET,
    season_col: str = "Year",
) -> pd.DataFrame:
    """Permutation importance on a held-out season, scored by Brier skill.

    Model-agnostic and measured out of sample, so it reflects what a feature
    contributes to a forecast rather than how often a tree happened to split
    on it.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer

    seasons = sorted(dataset[season_col].dropna().unique())
    test_season = test_season if test_season is not None else seasons[-1]
    train = dataset.loc[dataset[season_col] < test_season]
    test = dataset.loc[dataset[season_col] == test_season]
    if train.empty or test.empty:
        raise ValueError(f"no train/test split available at season {test_season}")

    available = set(dataset.columns)
    selected = registry.feature_columns(stage, available=available)
    numeric = [c for c in selected if registry.BY_NAME[c].kind in ("numeric", "binary")]
    categorical = [c for c in selected if c not in numeric]

    estimator = MODEL_FACTORIES[model](numeric, categorical)
    estimator.fit(train[selected], train[target])

    base_rate = float(train[target].mean())

    def neg_brier(y_true, y_prob):
        return -brier_score_loss(y_true, y_prob)

    scorer = make_scorer(neg_brier, response_method="predict_proba", greater_is_better=True)
    importance = permutation_importance(
        estimator,
        test[selected],
        test[target],
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=config.RANDOM_SEED,
    )
    brier_base = brier_score_loss(
        test[target], np.full(len(test), base_rate)
    )
    return (
        pd.DataFrame(
            {
                "feature": selected,
                "brier_increase": importance.importances_mean,
                "std": importance.importances_std,
            }
        )
        .assign(skill_cost=lambda d: d["brier_increase"] / brier_base)
        .sort_values("brier_increase", ascending=False, ignore_index=True)
    )


def race_bootstrap(
    predictions: pd.DataFrame,
    *,
    metric: str = "brier_skill",
    base_rate: float | None = None,
    n_boot: int = 500,
    target: str = TARGET,
    seed: int = config.RANDOM_SEED,
) -> dict[str, float]:
    """Confidence interval that resamples whole races, not rows.

    A first-lap pile-up retires several cars at once, so driver-race rows are
    correlated within an event.  Resampling rows would understate the interval;
    resampling races respects the correlation.
    """
    rng = np.random.default_rng(seed)
    keys = [c for c in ("Year", "RoundNumber") if c in predictions.columns]
    if not keys:
        raise KeyError("predictions must carry Year and RoundNumber to resample races")

    races = predictions.groupby(keys, observed=True).indices
    race_keys = list(races)
    base = base_rate if base_rate is not None else float(predictions[target].mean())

    samples = []
    for _ in range(n_boot):
        picked = rng.choice(len(race_keys), size=len(race_keys), replace=True)
        idx = np.concatenate([races[race_keys[i]] for i in picked])
        block = predictions.iloc[idx]
        if block[target].nunique() < 2:
            continue
        scored = score_predictions(block[target], block["predicted"], base)
        samples.append(scored[metric])

    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "metric": metric,
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "n_boot": int(values.size),
    }


def compare_models(
    dataset: pd.DataFrame,
    *,
    models: Sequence[str] = DEFAULT_MODEL_SUITE,
    stages: Sequence[registry.Stage] = ("pre_weekend", "post_quali"),
    sort_by: str = "brier_skill",
    **kwargs,
) -> pd.DataFrame:
    """Walk-forward every model at every stage and rank the results.

    Read the table in this order:

    1. ``brier_skill`` — is the model worth anything at all?  Zero means it has
       learnt nothing beyond the base rate.
    2. ``calibration_slope`` — can the probabilities be taken at face value?
       This is the column that matters most for a downstream points model, and
       it is routinely the one where the best-ranking model loses.
    3. ``pr_auc`` — ranking quality at a 14% positive rate.  Compare it against
       the base rate, never against 0.5.

    A model that wins on skill and sits at a slope of 0.7 is overconfident, and
    feeding its probabilities into an expected-points calculation will bias
    every driver's forecast.  Prefer the calibrated variant even at a small
    cost in skill.
    """
    rows = []
    for stage in stages:
        for model in models:
            try:
                result = walk_forward_evaluate(
                    dataset, stage=stage, model=model, **kwargs
                )
            except Exception as exc:  # noqa: BLE001 - one bad model must not stop the sweep
                log.warning("%s at %s failed: %s", model, stage, exc)
                continue
            if result.scores.empty:
                continue
            summary = result.summary()
            rows.append(
                {
                    "stage": stage,
                    "model": model,
                    "n_features": len(result.features),
                    **{
                        key: round(float(summary[key]), 4)
                        for key in (
                            "brier", "brier_skill", "log_loss",
                            "roc_auc", "pr_auc", "calibration_slope",
                            "mean_predicted", "observed_rate",
                        )
                        if key in summary
                    },
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["stage", sort_by], ascending=[True, False], ignore_index=True
    )


def fit_final_model(
    dataset: pd.DataFrame,
    *,
    stage: registry.Stage = "post_quali",
    model: str = "xgboost_sigmoid",
    target: str = TARGET,
    train_through_season: int | None = None,
) -> tuple[Any, list[str]]:
    """Fit one model on everything up to and including a season, ready to serve.

    Args:
        train_through_season: Last season to train on, inclusive.  Defaults to
            every season in the dataset, which is what you want when predicting
            the next race rather than backtesting.

    Returns:
        ``(fitted_estimator, feature_names)``.  Pass the feature names to
        ``estimator.predict_proba`` in the same order.
    """
    frame = dataset
    if train_through_season is not None:
        frame = frame.loc[frame["Year"] <= train_through_season]

    available = set(frame.columns)
    selected = registry.feature_columns(stage, available=available)
    numeric = [c for c in selected if registry.BY_NAME[c].kind in ("numeric", "binary")]
    categorical = [c for c in selected if c not in numeric]

    estimator = MODEL_FACTORIES[model](numeric, categorical)
    estimator.fit(frame[selected], frame[target])
    log.info(
        "fitted %s on %d rows through %s using %d features",
        model, len(frame), frame["Year"].max(), len(selected),
    )
    return estimator, selected
