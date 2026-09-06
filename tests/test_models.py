"""Metrics must be right before scores mean anything.

The headline risk in a rare-event problem is reporting a metric that flatters
a model carrying no information.  These tests pin the metrics to cases whose
answers are known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    calibration_slope,
    race_bootstrap,
    score_predictions,
    walk_forward_evaluate,
)


def test_base_rate_prediction_scores_zero_skill() -> None:
    """The whole point of Brier skill: predicting the base rate earns nothing."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.15, size=2000)
    scores = score_predictions(y, np.full(2000, 0.15), base_rate=0.15)
    assert scores["brier_skill"] == pytest.approx(0.0, abs=1e-12)


def test_a_perfect_forecast_scores_full_skill() -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    scores = score_predictions(y, y.astype(float), base_rate=0.5)
    assert scores["brier"] == pytest.approx(0.0)
    assert scores["brier_skill"] == pytest.approx(1.0)
    assert scores["roc_auc"] == pytest.approx(1.0)


def test_an_inverted_forecast_scores_negative_skill() -> None:
    """Worse than the base rate must read as worse, not merely as low."""
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.2, size=1000)
    inverted = np.where(y == 1, 0.05, 0.6)
    scores = score_predictions(y, inverted, base_rate=0.2)
    assert scores["brier_skill"] < 0
    assert scores["roc_auc"] < 0.5


def test_accuracy_is_not_reported() -> None:
    """A model that predicts 'nobody retires' is 86% accurate and worthless.

    Accuracy is excluded on purpose; this test stops it creeping back in.
    """
    scores = score_predictions([0] * 86 + [1] * 14, [0.0] * 100, base_rate=0.14)
    assert "accuracy" not in scores


def test_calibration_slope_detects_overconfidence() -> None:
    rng = np.random.default_rng(2)
    truth = rng.uniform(0.05, 0.6, size=4000)
    y = rng.binomial(1, truth)

    honest = calibration_slope(y, truth)
    # Push probabilities away from the middle: the classic overconfident model.
    logit = np.log(truth / (1 - truth))
    overconfident = 1 / (1 + np.exp(-2.0 * logit))

    assert honest == pytest.approx(1.0, abs=0.2)
    assert calibration_slope(y, overconfident) < honest


def test_score_predictions_handles_a_single_class() -> None:
    scores = score_predictions([0, 0, 0], [0.1, 0.2, 0.1], base_rate=0.1)
    assert np.isnan(scores["roc_auc"])
    assert not np.isnan(scores["brier"])


@pytest.fixture(scope="module")
def modelling_dataset(raw_results, circuit_profiles):
    from src.data.generate_dataset import build_dataset

    dataset, _ = build_dataset(raw_results, circuit_profiles, run_checks=False)
    return dataset


def test_walk_forward_never_trains_on_the_future(modelling_dataset) -> None:
    result = walk_forward_evaluate(modelling_dataset, stage="post_quali")
    assert not result.scores.empty
    # The first season can never be a test fold: it has no prior data.
    assert result.scores["season"].min() > modelling_dataset["Year"].min()


def test_pre_weekend_stage_excludes_grid_position(modelling_dataset) -> None:
    result = walk_forward_evaluate(modelling_dataset, stage="pre_weekend")
    assert "grid_position" not in result.features
    assert "driver_dnf_rate_10" in result.features


def test_post_quali_stage_includes_grid_position(modelling_dataset) -> None:
    result = walk_forward_evaluate(modelling_dataset, stage="post_quali")
    assert "grid_position" in result.features


def test_both_model_families_run(modelling_dataset) -> None:
    for model in ("logistic", "gradient_boosting"):
        result = walk_forward_evaluate(modelling_dataset, model=model)
        assert not result.scores.empty, model
        assert result.predictions["predicted"].between(0, 1).all(), model


def test_drop_features_removes_them(modelling_dataset) -> None:
    result = walk_forward_evaluate(
        modelling_dataset, drop_features=["grid_position", "driver_dnf_rate_10"]
    )
    assert "grid_position" not in result.features
    assert "driver_dnf_rate_10" not in result.features


def test_race_bootstrap_resamples_whole_races(modelling_dataset) -> None:
    """Rows inside one race are correlated, so races are the sampling unit."""
    result = walk_forward_evaluate(modelling_dataset)
    interval = race_bootstrap(
        result.predictions, base_rate=float(modelling_dataset["dnf"].mean()), n_boot=100
    )
    assert interval["ci_low"] <= interval["mean"] <= interval["ci_high"]
    assert interval["n_boot"] > 0


def test_missing_target_raises(modelling_dataset) -> None:
    with pytest.raises(KeyError, match="target"):
        walk_forward_evaluate(modelling_dataset, target="not_a_column")


# --------------------------------------------------------------------------- #
# The extended model zoo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model",
    ["logistic", "random_forest", "gradient_boosting", "xgboost", "lightgbm",
     "xgboost_sigmoid", "xgboost_isotonic", "gradient_boosting_sigmoid"],
)
def test_every_registered_model_runs(modelling_dataset, model: str) -> None:
    from src.models.train import walk_forward_evaluate

    result = walk_forward_evaluate(modelling_dataset, model=model)
    assert not result.scores.empty, model
    assert result.predictions["predicted"].between(0, 1).all(), model


def test_boosters_do_not_reweight_the_positive_class() -> None:
    """``scale_pos_weight`` buys ranking and sells calibration.

    Expected points scales linearly in P(finish), so an inflated probability is
    a biased points forecast even when the ordering is perfect.
    """
    from src.models.train import make_xgboost

    pipeline = make_xgboost(["a", "b"], [])
    classifier = pipeline.named_steps["clf"]
    assert getattr(classifier, "scale_pos_weight", None) in (None, 1, 1.0)


def test_nan_native_models_receive_unimputed_values() -> None:
    """Cold-start rows are exactly where risk is unusual, so a rookie's missing
    history must stay missing rather than become the median."""
    from src.models.train import make_xgboost

    prep = make_xgboost(["a", "b"], []).named_steps["prep"]
    assert prep.transformers[0][1] == "passthrough"


def test_xgboost_handles_missing_features(modelling_dataset) -> None:
    from src.models.train import walk_forward_evaluate

    frame = modelling_dataset.copy()
    frame.loc[frame.index[:50], "driver_dnf_rate_10"] = np.nan
    result = walk_forward_evaluate(frame, model="xgboost")
    assert result.predictions["predicted"].notna().all()


def test_compare_models_ranks_by_skill(modelling_dataset) -> None:
    from src.models.train import compare_models

    comparison = compare_models(
        modelling_dataset,
        models=("logistic", "xgboost", "random_forest"),
        stages=("post_quali",),
    )
    assert len(comparison) == 3
    assert comparison["brier_skill"].is_monotonic_decreasing


def test_compare_models_survives_a_bad_model(modelling_dataset) -> None:
    """One failing model must not abort the sweep."""
    from src.models.train import compare_models

    comparison = compare_models(
        modelling_dataset,
        models=("logistic", "does_not_exist"),
        stages=("post_quali",),
    )
    assert set(comparison["model"]) == {"logistic"}


def test_fit_final_model_trains_through_a_season(modelling_dataset) -> None:
    from src.models.train import fit_final_model

    estimator, features = fit_final_model(modelling_dataset, model="xgboost")
    probabilities = estimator.predict_proba(modelling_dataset[features])[:, 1]
    assert len(features) > 10
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
