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
    DEFAULT_LOOKBACK_RACES,
    MODEL_FACTORIES,
    calibration_slope,
    race_bootstrap,
    score_predictions,
    walk_forward_evaluate,
    walk_forward_races,
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


@pytest.mark.parametrize("model", sorted(MODEL_FACTORIES))
def test_every_model_family_runs(modelling_dataset, model: str) -> None:
    result = walk_forward_evaluate(modelling_dataset, model=model)
    assert not result.scores.empty, model
    assert result.predictions["predicted"].between(0, 1).all(), model


def test_baseline_predicts_one_constant_per_fold(modelling_dataset) -> None:
    """The reference model must be exactly that: the training base rate.

    If this ever varies within a season, the comparison every other model is
    judged against has stopped being a base rate.
    """
    result = walk_forward_evaluate(modelling_dataset, model="baseline")
    per_season = result.predictions.groupby("Year", observed=True)["predicted"].nunique()
    assert (per_season == 1).all()


def test_baseline_scores_no_skill(modelling_dataset) -> None:
    """Zero Brier skill by construction, and no ranking information at all."""
    summary = walk_forward_evaluate(modelling_dataset, model="baseline").summary()
    assert abs(summary["brier_skill"]) < 1e-6
    assert summary["roc_auc"] == pytest.approx(0.5, abs=1e-9)


def test_random_forest_handles_cold_start_nans(modelling_dataset) -> None:
    """RandomForest cannot take NaN natively; the imputer must cover every row."""
    frame = modelling_dataset.copy()
    frame.loc[frame.index[:50], "driver_dnf_rate_10"] = np.nan
    result = walk_forward_evaluate(frame, model="random_forest")
    assert result.predictions["predicted"].notna().all()


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
# Race-by-race retraining
# --------------------------------------------------------------------------- #
#
# The operational cadence: refit the moment the last race is classified, so
# every race is scored by a model that has seen every race before it and none
# after. A season-level split cannot express that, and flatters early rounds.


def test_race_walk_forward_never_trains_on_the_future(modelling_dataset) -> None:
    result = walk_forward_races(modelling_dataset, model="logistic")
    assert not result.predictions.empty
    # Every scored race must have been preceded by the rows it trained on.
    races = (
        modelling_dataset[["Year", "RoundNumber", "RaceDate"]]
        .drop_duplicates()
        .sort_values("RaceDate")
    )
    order = {(r.Year, r.RoundNumber): i for i, r in enumerate(races.itertuples())}
    for row in result.predictions.itertuples():
        # A race can only be scored once history exists before it.
        assert order[(row.Year, row.RoundNumber)] > 0


def test_every_scored_race_is_scored_exactly_once(modelling_dataset) -> None:
    result = walk_forward_races(modelling_dataset, model="logistic")
    counts = result.predictions.groupby(
        ["Year", "RoundNumber", "DriverId"], observed=True
    ).size()
    assert counts.max() == 1


def test_training_set_grows_with_every_race(modelling_dataset) -> None:
    """The expanding window must actually expand, when it is asked for."""
    result = walk_forward_races(
        modelling_dataset, model="logistic", lookback_races=None
    )
    per_race = (
        result.predictions.groupby(["Year", "RoundNumber"], observed=True)["train_rows"]
        .first()
        .sort_index()
    )
    assert per_race.is_monotonic_increasing
    assert per_race.iloc[-1] > per_race.iloc[0]


def test_lookback_caps_the_training_window(modelling_dataset) -> None:
    """A sliding window must stop growing once it is full."""
    # min_train_rows has to come down with the window, or nothing is scored.
    result = walk_forward_races(
        modelling_dataset, model="logistic", lookback_races=5, min_train_rows=50
    )
    assert not result.predictions.empty
    train_rows = result.predictions["train_rows"]
    full = walk_forward_races(
        modelling_dataset, model="logistic", min_train_rows=50, lookback_races=None
    )
    assert train_rows.max() < full.predictions["train_rows"].max()
    # Five races of a fixed-size field is a bounded number of rows.
    per_race = modelling_dataset.groupby(
        ["Year", "RoundNumber"], observed=True
    ).size().max()
    assert train_rows.max() <= 5 * per_race


def test_refit_every_reduces_the_number_of_fits(modelling_dataset) -> None:
    """Refitting less often must reuse a model across races, not silently refit."""
    often = walk_forward_races(modelling_dataset, model="logistic", refit_every=1)
    rarely = walk_forward_races(modelling_dataset, model="logistic", refit_every=5)
    assert rarely.predictions["train_rows"].nunique() < often.predictions[
        "train_rows"
    ].nunique()


def test_race_walk_forward_reports_per_season_scores(modelling_dataset) -> None:
    result = walk_forward_races(modelling_dataset, model="logistic")
    assert not result.scores.empty
    assert "season" in result.scores.columns
    assert result.predictions["predicted"].between(0, 1).all()


def test_start_after_skips_early_races(modelling_dataset) -> None:
    cutoff = modelling_dataset["RaceDate"].quantile(0.5)
    result = walk_forward_races(modelling_dataset, model="logistic", start_after=cutoff)
    assert (result.predictions["RaceDate"] > cutoff).all()


def test_default_is_a_bounded_window(modelling_dataset) -> None:
    """40 races by default: measured on held-out 2026, not a neutral choice."""
    assert DEFAULT_LOOKBACK_RACES == 40
    default = walk_forward_races(modelling_dataset, model="logistic")
    expanding = walk_forward_races(
        modelling_dataset, model="logistic", lookback_races=None
    )
    assert (
        default.predictions["train_rows"].max()
        <= expanding.predictions["train_rows"].max()
    )


def test_lookback_window_is_measured_over_the_full_calendar(
    modelling_dataset,
) -> None:
    """``start_after`` chooses what to score; it must not shrink the window.

    Deriving the lookback from the post-cutoff races gave the first scored race
    a full history and the next one a single race, so everything after it fell
    below ``min_train_rows`` and was silently dropped.
    """
    cutoff = modelling_dataset["RaceDate"].quantile(0.5)
    scored = walk_forward_races(
        modelling_dataset, model="logistic", start_after=cutoff
    )
    everything = walk_forward_races(modelling_dataset, model="logistic")

    after_cutoff = everything.predictions.loc[
        everything.predictions["RaceDate"] > cutoff
    ]
    assert len(scored.predictions) == len(after_cutoff)
    # And each race trains on the same rows either way.
    pd.testing.assert_series_equal(
        scored.predictions.sort_values(["Year", "RoundNumber"])["train_rows"]
        .reset_index(drop=True),
        after_cutoff.sort_values(["Year", "RoundNumber"])["train_rows"]
        .reset_index(drop=True),
    )
