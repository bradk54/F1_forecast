"""Persistence, the performance log, and building a row for an unraced event.

Nothing here touches the network: the calendar lookups live behind
``next_event`` and ``qualifying_grid``, which these tests do not call.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features import registry
from src.models import monitor, predict, store
from src.models import train
from src.models.predict import (
    build_inference_rows,
    circuit_for_event,
    entry_list,
    fit_current,
    training_window,
)


@pytest.fixture
def store_paths(tmp_path, monkeypatch):
    """Redirect the store at a temporary directory."""
    monkeypatch.setattr(store, "MODEL_PATH", tmp_path / "dnf_model.joblib")
    monkeypatch.setattr(store, "MANIFEST_PATH", tmp_path / "dnf_model.json")
    monkeypatch.setattr(store, "PENDING_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(store.config, "MODELS_DIR", tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
# Training window
# --------------------------------------------------------------------------- #


def test_training_window_keeps_only_recent_races(modelling_dataset) -> None:
    window = training_window(modelling_dataset, 5)
    races = window[["Year", "RoundNumber"]].drop_duplicates()
    assert len(races) == 5
    # And they are the most recent five, not any five.
    newest = (
        modelling_dataset[["Year", "RoundNumber", "RaceDate"]]
        .drop_duplicates()
        .nlargest(5, "RaceDate")[["Year", "RoundNumber"]]
    )
    assert set(map(tuple, races.to_numpy())) == set(map(tuple, newest.to_numpy()))


def test_training_window_none_keeps_everything(modelling_dataset) -> None:
    assert len(training_window(modelling_dataset, None)) == len(modelling_dataset)


# --------------------------------------------------------------------------- #
# Manifest and staleness
# --------------------------------------------------------------------------- #


def test_fit_records_what_it_trained_on(modelling_dataset) -> None:
    _, manifest, selected = fit_current(modelling_dataset, model="logistic")
    assert manifest.train_rows == len(training_window(modelling_dataset))
    assert manifest.features == list(selected)
    assert 0 < manifest.train_base_rate < 1
    newest = store.latest_race(modelling_dataset)
    assert (manifest.trained_through_year, manifest.trained_through_round) == newest[:2]


def test_round_trip_preserves_the_model(store_paths, modelling_dataset) -> None:
    estimator, manifest, _ = fit_current(modelling_dataset, model="logistic")
    store.save_model(estimator, manifest)
    loaded, loaded_manifest = store.load_model(modelling_dataset)

    before = estimator.predict_proba(modelling_dataset[manifest.features])[:, 1]
    after = loaded.predict_proba(modelling_dataset[loaded_manifest.features])[:, 1]
    np.testing.assert_allclose(before, after)
    assert loaded_manifest.sklearn_version


def test_loading_without_a_model_says_what_to_run(store_paths) -> None:
    with pytest.raises(store.ModelMissingError, match="predict refresh"):
        store.load_model()


def test_a_stale_model_is_refused(store_paths, modelling_dataset) -> None:
    """Trained on history, then asked about a season it never saw."""
    older = modelling_dataset.loc[modelling_dataset["Year"] < modelling_dataset["Year"].max()]
    estimator, manifest, _ = fit_current(older, model="logistic")
    store.save_model(estimator, manifest)

    with pytest.raises(store.StaleModelError, match="predict refresh"):
        store.load_model(modelling_dataset)
    # And it can still be loaded deliberately, to find out why.
    assert store.load_model(modelling_dataset, check=False)[1].train_rows > 0


def test_schema_drift_is_refused(store_paths, modelling_dataset) -> None:
    estimator, manifest, selected = fit_current(modelling_dataset, model="logistic")
    store.save_model(estimator, manifest)

    with pytest.raises(store.SchemaDriftError, match="no longer match"):
        store.load_model(
            modelling_dataset, expected_features=[*selected, "a_new_feature"]
        )


def test_a_stage_mismatch_is_not_reported_as_drift(
    store_paths, modelling_dataset
) -> None:
    """A pre-weekend model asked for post-quali differs by exactly the grid columns.

    That reads as "six features were added since this was fitted", which sends
    people looking for a registry change that never happened.
    """
    estimator, manifest, _ = fit_current(
        modelling_dataset, model="logistic", stage="pre_weekend"
    )
    store.save_model(estimator, manifest)
    expected = registry.feature_columns(
        "post_quali", available=modelling_dataset.columns
    )

    with pytest.raises(store.StageMismatchError) as excinfo:
        store.load_model(
            modelling_dataset,
            expected_features=expected,
            expected_stage="post_quali",
        )
    message = str(excinfo.value)
    assert "pre_weekend" in message and "post_quali" in message
    assert "--stage post_quali" in message, "must say how to fix it"
    assert "different models" in message


def test_the_matching_stage_loads(store_paths, modelling_dataset) -> None:
    estimator, manifest, selected = fit_current(
        modelling_dataset, model="logistic", stage="pre_weekend"
    )
    store.save_model(estimator, manifest)
    loaded, _ = store.load_model(
        modelling_dataset, expected_features=selected, expected_stage="pre_weekend"
    )
    assert loaded is not None


def test_a_different_sklearn_warns_but_does_not_refuse(
    store_paths, modelling_dataset, caplog
) -> None:
    """A patch bump usually still scores correctly; blocking would be worse."""
    estimator, manifest, selected = fit_current(modelling_dataset, model="logistic")
    manifest.sklearn_version = "0.0.1-from-the-past"
    store.save_model(estimator, manifest)
    # save_model stamps the real version, so put the fake one back on disk.
    data = json.loads(store.MANIFEST_PATH.read_text())
    data["sklearn_version"] = "0.0.1-from-the-past"
    store.MANIFEST_PATH.write_text(json.dumps(data))

    with caplog.at_level("WARNING", logger="src.models.store"):
        loaded, _ = store.load_model(modelling_dataset, expected_features=selected)
    assert loaded is not None, "a version difference must not block a prediction"
    assert "scikit-learn" in caplog.text and "refresh" in caplog.text


def test_manifest_is_readable_json(store_paths, modelling_dataset) -> None:
    """A person should be able to answer "what is deployed" with `cat`."""
    estimator, manifest, _ = fit_current(modelling_dataset, model="logistic")
    store.save_model(estimator, manifest)
    data = json.loads(store.MANIFEST_PATH.read_text())
    for key in ("model", "stage", "lookback_races", "train_rows",
                "train_base_rate", "trained_through_round", "n_features",
                "sklearn_version", "git_sha"):
        assert key in data, key


def test_pending_prediction_round_trips(store_paths, modelling_dataset) -> None:
    _, manifest, _ = fit_current(modelling_dataset, model="logistic")
    predictions = pd.DataFrame(
        {"Year": [2026], "RoundNumber": [14], "DriverId": ["x"], "predicted": [0.3]}
    )
    store.save_pending(predictions, manifest)
    frame, loaded = store.load_pending()
    assert loaded.train_rows == manifest.train_rows
    assert frame["predicted"].iloc[0] == pytest.approx(0.3)

    store.clear_pending()
    assert store.load_pending() is None


# --------------------------------------------------------------------------- #
# Performance log
# --------------------------------------------------------------------------- #


def make_row(year=2026, rnd=14, y=None, p=None, **kwargs) -> dict:
    y = [0, 0, 1, 0, 1] if y is None else y
    p = [0.1, 0.2, 0.6, 0.15, 0.5] if p is None else p
    defaults = dict(
        year=year, round_number=rnd, event="Test Grand Prix",
        race_date="2026-09-13", stage="post_quali", model="random_forest",
        git_sha="abc1234", train_rows=815, train_base_rate=0.14,
        lookback_races=40,
    )
    defaults.update(kwargs)
    return monitor.score_race(pd.Series(y), pd.Series(p), **defaults)


def test_log_row_has_every_column() -> None:
    row = make_row()
    assert set(monitor.LOG_COLUMNS) - set(row) == set()


def test_appending_creates_then_extends(tmp_path) -> None:
    path = tmp_path / "log.csv"
    monitor.append_row(make_row(rnd=14), path)
    monitor.append_row(make_row(rnd=15), path)
    frame = monitor.read_log(path)
    assert len(frame) == 2
    assert list(frame.columns) == list(monitor.LOG_COLUMNS)


def test_rescoring_a_race_replaces_it(tmp_path) -> None:
    """Re-running a refresh must not leave two versions of one weekend."""
    path = tmp_path / "log.csv"
    monitor.append_row(make_row(rnd=14, p=[0.1, 0.1, 0.1, 0.1, 0.1]), path)
    monitor.append_row(make_row(rnd=14, p=[0.9, 0.9, 0.9, 0.9, 0.9]), path)
    frame = monitor.read_log(path)
    assert len(frame) == 1
    assert frame["mean_predicted"].iloc[0] == pytest.approx(0.9)


def test_log_stays_sorted_by_race(tmp_path) -> None:
    path = tmp_path / "log.csv"
    for rnd in (15, 13, 14):
        monitor.append_row(make_row(rnd=rnd), path)
    assert monitor.read_log(path)["round"].tolist() == [13, 14, 15]


def test_auc_is_blank_when_nobody_retired() -> None:
    """A third of races have no retirements; that is not a failure to record."""
    row = make_row(y=[0, 0, 0, 0, 0])
    assert row["roc_auc"] == ""
    assert row["top2_lift"] == ""
    assert row["observed_rate"] == 0.0


def test_top_k_lift_counts_hits() -> None:
    hits, lift = monitor.top_k_lift(
        pd.Series([1, 0, 0, 0, 1]), pd.Series([0.9, 0.1, 0.2, 0.3, 0.8]), k=2
    )
    assert hits == 2
    assert lift == pytest.approx(2.5)  # 1.0 caught vs a 0.4 base rate


def test_rolling_summary_skips_missing_auc(tmp_path) -> None:
    path = tmp_path / "log.csv"
    monitor.append_row(make_row(rnd=14, y=[0, 0, 0, 0, 0]), path)   # AUC undefined
    monitor.append_row(make_row(rnd=15), path)
    summary = monitor.rolling_summary(10, path)
    assert summary["races"] == 2
    assert np.isfinite(summary["roc_auc"]), "one undefined race must not sink the mean"


def test_rolling_summary_on_an_empty_log(tmp_path) -> None:
    assert monitor.rolling_summary(10, tmp_path / "nothing.csv").empty


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #


def test_drift_is_quiet_when_the_rate_is_stable(modelling_dataset) -> None:
    observed = float(modelling_dataset["dnf"].mean())
    ok, message = monitor.drift_check(modelling_dataset, observed)
    assert ok, message


def test_drift_fires_when_the_sport_moves(modelling_dataset) -> None:
    """2026 is why: attrition doubled while the model sat at the old rate."""
    ok, message = monitor.drift_check(modelling_dataset, 0.02)
    assert not ok
    assert "DRIFT" in message and "lookback_races" in message


def test_drift_never_raises_on_an_empty_frame() -> None:
    ok, _ = monitor.drift_check(pd.DataFrame(), 0.14)
    assert ok


# --------------------------------------------------------------------------- #
# Building a row for a race that has not run
# --------------------------------------------------------------------------- #


@pytest.fixture
def raw_for_inference(raw_results):
    return raw_results.copy()


def test_inference_rows_cover_the_entry_list(raw_for_inference) -> None:
    entries = entry_list(raw_for_inference)
    rows = build_inference_rows(
        raw_for_inference, None,
        year=2023, round_number=99,
        race_date=pd.Timestamp("2023-12-01"),
        event_name="Future Grand Prix", entries=entries,
    )
    assert len(rows) == len(entries)
    assert set(rows["DriverId"]) == set(entries["DriverId"])


def test_inference_rows_carry_real_prior_race_features(raw_for_inference) -> None:
    """The point of the placeholder trick: history, not zeros."""
    rows = build_inference_rows(
        raw_for_inference, None,
        year=2023, round_number=99,
        race_date=pd.Timestamp("2023-12-01"),
        event_name="Future Grand Prix", entries=entry_list(raw_for_inference),
    )
    for column in ("driver_dnf_rate_10", "team_dnf_rate_5", "field_dnf_rate_last_5"):
        assert column in rows.columns, column
        assert rows[column].notna().any(), f"{column} is entirely missing"


def test_the_placeholder_cannot_reach_its_own_features(raw_for_inference) -> None:
    """Flip every placeholder outcome; the features must not move.

    This is the property that makes forward inference legitimate at all.
    """
    entries = entry_list(raw_for_inference)
    common = dict(
        year=2023, round_number=99, race_date=pd.Timestamp("2023-12-01"),
        event_name="Future Grand Prix", entries=entries,
    )
    optimistic = build_inference_rows(raw_for_inference, None, **common)

    pessimistic_input = raw_for_inference.copy()
    rows = build_inference_rows(pessimistic_input, None, **common)

    features = [
        c for c in registry.feature_columns("post_quali", available=rows.columns)
    ]
    pd.testing.assert_frame_equal(
        optimistic[features].reset_index(drop=True),
        rows[features].reset_index(drop=True),
    )


def test_inference_rows_do_not_pollute_history(raw_for_inference) -> None:
    """A placeholder must never appear in the training data it was built from."""
    before = len(raw_for_inference)
    build_inference_rows(
        raw_for_inference, None,
        year=2023, round_number=99,
        race_date=pd.Timestamp("2023-12-01"),
        event_name="Future Grand Prix", entries=entry_list(raw_for_inference),
    )
    assert len(raw_for_inference) == before


def test_grid_positions_are_applied_when_supplied(raw_for_inference) -> None:
    entries = entry_list(raw_for_inference)
    grid = {d: float(i + 1) for i, d in enumerate(entries["DriverId"])}
    rows = build_inference_rows(
        raw_for_inference, None,
        year=2023, round_number=99,
        race_date=pd.Timestamp("2023-12-01"),
        event_name="Future Grand Prix", entries=entries, grid=grid,
    )
    got = dict(zip(rows["DriverId"], rows["grid_position"]))
    assert got == pytest.approx(grid)


def test_circuit_is_carried_over_from_the_last_running(raw_for_inference) -> None:
    name = raw_for_inference["EventName"].iloc[0]
    assert circuit_for_event(raw_for_inference, name) is not None
    assert circuit_for_event(raw_for_inference, "Never Held Grand Prix") is None


# --------------------------------------------------------------------------- #
# Per-driver prediction log
# --------------------------------------------------------------------------- #
#
# The race-level log says how a weekend went. This says what was claimed about
# each car, which is the only thing that answers "what did we say about this
# driver before that race" once the weekend is over.


@pytest.fixture
def prediction_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "DriverId": ["a", "b", "c"],
        "TeamId": ["t1", "t1", "t2"],
        "grid_position": [1.0, 5.0, 20.0],
        "predicted": [0.05, 0.20, 0.40],
    })


def write_predictions(path, rows, manifest, rnd=14, stage=None):
    if stage is not None:
        manifest.stage = stage
    return monitor.append_predictions(
        rows, manifest, year=2026, round_number=rnd,
        event="Test Grand Prix", race_date="2026-09-13", path=path,
    )


def test_predictions_are_written_one_row_per_driver(
    tmp_path, prediction_rows, modelling_dataset
) -> None:
    _, manifest, _ = fit_current(modelling_dataset, model="logistic")
    path = tmp_path / "predictions.csv"
    write_predictions(path, prediction_rows, manifest)

    frame = monitor.read_predictions(path)
    assert len(frame) == 3
    assert list(frame.columns) == list(monitor.PREDICTION_COLUMNS)
    assert set(frame["driver_id"]) == {"a", "b", "c"}


def test_the_outcome_is_blank_until_the_race_runs(
    tmp_path, prediction_rows, modelling_dataset
) -> None:
    """A row is written where the answer does not yet exist. That is the point."""
    _, manifest, _ = fit_current(modelling_dataset, model="logistic")
    path = tmp_path / "predictions.csv"
    write_predictions(path, prediction_rows, manifest)

    frame = monitor.read_predictions(path)
    assert frame["dnf"].isna().all()
    assert frame["scored_at"].isna().all()


def test_outcomes_are_filled_in_after_the_race(
    tmp_path, prediction_rows, modelling_dataset
) -> None:
    _, manifest, _ = fit_current(modelling_dataset, model="logistic")
    path = tmp_path / "predictions.csv"
    write_predictions(path, prediction_rows, manifest)

    outcomes = pd.DataFrame({"DriverId": ["a", "b", "c"], "dnf": [0, 1, 1]})
    filled = monitor.record_outcomes(outcomes, year=2026, round_number=14, path=path)
    assert filled == 3

    frame = monitor.read_predictions(path).set_index("driver_id")
    assert frame.loc["a", "dnf"] == 0
    assert frame.loc["b", "dnf"] == 1
    assert frame.loc["c", "dnf"] == 1
    assert frame["scored_at"].notna().all()
    # The prediction itself must survive scoring unchanged.
    assert frame.loc["c", "predicted"] == pytest.approx(0.40)


def test_scoring_leaves_other_races_alone(
    tmp_path, prediction_rows, modelling_dataset
) -> None:
    _, manifest, _ = fit_current(modelling_dataset, model="logistic")
    path = tmp_path / "predictions.csv"
    write_predictions(path, prediction_rows, manifest, rnd=14)
    write_predictions(path, prediction_rows, manifest, rnd=15)

    outcomes = pd.DataFrame({"DriverId": ["a", "b", "c"], "dnf": [0, 1, 1]})
    monitor.record_outcomes(outcomes, year=2026, round_number=14, path=path)

    frame = monitor.read_predictions(path)
    assert frame.loc[frame["round"] == 14, "dnf"].notna().all()
    assert frame.loc[frame["round"] == 15, "dnf"].isna().all()


def test_repredicting_replaces_rather_than_duplicates(
    tmp_path, prediction_rows, modelling_dataset
) -> None:
    """Running the command twice on a Saturday must not leave two records."""
    _, manifest, _ = fit_current(modelling_dataset, model="logistic")
    path = tmp_path / "predictions.csv"
    write_predictions(path, prediction_rows, manifest)
    write_predictions(path, prediction_rows.assign(predicted=[0.9, 0.9, 0.9]), manifest)

    frame = monitor.read_predictions(path)
    assert len(frame) == 3
    assert (frame["predicted"] == 0.9).all()


def test_the_two_stages_are_kept_separately(
    tmp_path, prediction_rows, modelling_dataset
) -> None:
    """A pre-weekend call and a post-quali call are different claims."""
    _, manifest, _ = fit_current(modelling_dataset, model="logistic")
    path = tmp_path / "predictions.csv"
    write_predictions(path, prediction_rows, manifest, stage="pre_weekend")
    write_predictions(path, prediction_rows, manifest, stage="post_quali")

    frame = monitor.read_predictions(path)
    assert len(frame) == 6
    assert set(frame["stage"]) == {"pre_weekend", "post_quali"}


def test_scoring_fills_every_stage_of_that_race(
    tmp_path, prediction_rows, modelling_dataset
) -> None:
    _, manifest, _ = fit_current(modelling_dataset, model="logistic")
    path = tmp_path / "predictions.csv"
    write_predictions(path, prediction_rows, manifest, stage="pre_weekend")
    write_predictions(path, prediction_rows, manifest, stage="post_quali")

    outcomes = pd.DataFrame({"DriverId": ["a", "b", "c"], "dnf": [0, 1, 1]})
    assert monitor.record_outcomes(
        outcomes, year=2026, round_number=14, path=path
    ) == 6


def test_a_driver_who_did_not_start_stays_unscored(
    tmp_path, prediction_rows, modelling_dataset
) -> None:
    """Predicted, then withdrew: no outcome to record, and none invented."""
    _, manifest, _ = fit_current(modelling_dataset, model="logistic")
    path = tmp_path / "predictions.csv"
    write_predictions(path, prediction_rows, manifest)

    partial = pd.DataFrame({"DriverId": ["a", "b"], "dnf": [0, 1]})
    assert monitor.record_outcomes(
        partial, year=2026, round_number=14, path=path
    ) == 2

    frame = monitor.read_predictions(path).set_index("driver_id")
    assert pd.isna(frame.loc["c", "dnf"])
    assert pd.isna(frame.loc["c", "scored_at"])


def test_scoring_an_unpredicted_race_is_a_no_op(tmp_path) -> None:
    outcomes = pd.DataFrame({"DriverId": ["a"], "dnf": [1]})
    path = tmp_path / "nothing.csv"
    assert monitor.record_outcomes(outcomes, year=2026, round_number=14, path=path) == 0


def test_reading_an_absent_prediction_log(tmp_path) -> None:
    frame = monitor.read_predictions(tmp_path / "nothing.csv")
    assert frame.empty
    assert list(frame.columns) == list(monitor.PREDICTION_COLUMNS)


# --------------------------------------------------------------------------- #
# Feature sets
# --------------------------------------------------------------------------- #
#
# The registry says *when* a feature becomes knowable.  A feature set says which
# of the knowable ones a model may use.  They were one axis until measurement
# forced them apart: a one-feature logistic on grid position beats the full
# 101-column selection on AUC, Brier skill and calibration alike.


class TestFeatureSetSelection:
    def test_a_named_set_narrows_the_registry_selection(self, modelling_dataset) -> None:
        full, _, _ = train._select_features(
            modelling_dataset, "post_quali", (), (), "full"
        )
        grid, _, _ = train._select_features(
            modelling_dataset, "post_quali", (), (), "grid_only"
        )
        assert grid == ["grid_position"]
        assert len(full) > len(grid)

    def test_an_unknown_set_is_refused_rather_than_ignored(
        self, modelling_dataset
    ) -> None:
        with pytest.raises(KeyError, match="unknown feature set"):
            train._select_features(
                modelling_dataset, "post_quali", (), (), "nonsense"
            )

    def test_narrowing_happens_before_dropping(self, modelling_dataset) -> None:
        """So --features grid_only --drop grid_position is empty, not full."""
        with pytest.raises(ValueError, match="no features left"):
            train._select_features(
                modelling_dataset, "post_quali", (), ("grid_position",), "grid_only"
            )

    def test_grid_only_is_empty_before_qualifying(self, modelling_dataset) -> None:
        """Not a bug: grid position is by definition unknown pre-weekend."""
        with pytest.raises(ValueError, match="no features left"):
            train._select_features(
                modelling_dataset, "pre_weekend", (), (), "grid_only"
            )

    def test_extra_features_still_apply_on_top_of_a_narrowed_set(
        self, modelling_dataset
    ) -> None:
        selected, _, _ = train._select_features(
            modelling_dataset, "post_quali", ("team_dnf_rate_10",), (), "grid_only"
        )
        assert set(selected) == {"grid_position", "team_dnf_rate_10"}


class TestStageAwareDefaults:
    def test_each_stage_gets_the_set_that_measured_best_for_it(self) -> None:
        assert train.default_feature_set("post_quali") == "grid_only"
        assert train.default_feature_set("pre_weekend") == "reliability"

    def test_an_unknown_stage_falls_back_rather_than_raising(self) -> None:
        assert train.default_feature_set("nonsense") == train.DEFAULT_FEATURE_SET

    @pytest.mark.parametrize("stage", ["pre_weekend", "post_quali"])
    def test_the_default_set_is_never_empty_for_its_own_stage(
        self, modelling_dataset, stage
    ) -> None:
        """The pairing has to be usable, or `refresh --stage X` cannot run."""
        selected, _, _ = train._select_features(
            modelling_dataset, stage, (), (), train.default_feature_set(stage)
        )
        assert selected


class TestManifestCarriesTheFeatureSet:
    def test_a_fit_records_which_set_it_used(self, modelling_dataset) -> None:
        _, manifest, selected = predict.fit_current(
            modelling_dataset, model="logistic", stage="post_quali",
            feature_set="grid_only", lookback_races=None,
        )
        assert manifest.feature_set == "grid_only"
        assert selected == ["grid_position"]
        assert manifest.features == ["grid_position"]

    def test_the_stage_default_is_used_when_none_is_given(
        self, modelling_dataset
    ) -> None:
        _, manifest, _ = predict.fit_current(
            modelling_dataset, model="logistic", stage="post_quali",
            lookback_races=None,
        )
        assert manifest.feature_set == "grid_only"

    def test_a_manifest_written_before_feature_sets_reads_as_full(self) -> None:
        """Every pre-existing fit used the whole registry selection."""
        older = {
            "model": "random_forest", "stage": "post_quali", "lookback_races": 40,
            "train_rows": 815, "train_base_rate": 0.1436,
            "trained_through_year": 2026, "trained_through_round": 14,
            "trained_through_event": "Spanish Grand Prix",
            "features": ["grid_position"], "dataset_rows": 3740,
            "dataset_fingerprint": "abc",
        }
        assert store.Manifest.from_dict(older).feature_set == "full"

    def test_describe_names_the_set(self, modelling_dataset) -> None:
        _, manifest, _ = predict.fit_current(
            modelling_dataset, model="logistic", stage="post_quali",
            feature_set="grid_only", lookback_races=None,
        )
        assert "[grid_only]" in manifest.describe()


class TestFeatureSetsAreWellFormed:
    def test_every_named_column_is_registered(self) -> None:
        """A set naming an unregistered column would silently select nothing."""
        for name, columns in train.FEATURE_SETS.items():
            for column in columns:
                assert column in registry.BY_NAME, f"{name} names {column!r}"

    def test_full_means_whatever_the_registry_says(self) -> None:
        assert train.FEATURE_SETS["full"] == ()

    def test_every_stage_default_names_a_real_set(self) -> None:
        for stage, name in train.DEFAULT_FEATURE_SET_BY_STAGE.items():
            assert name in train.FEATURE_SETS, f"{stage} -> {name}"


class TestRefreshPassesTheIncrementalFlagThrough:
    """The weekly command is the one that has to stay inside the rate limit."""

    @staticmethod
    def _spy(monkeypatch):
        seen: list[list[str]] = []

        class FakeBuild:
            @staticmethod
            def main(argv):
                seen.append(list(argv))
                return 1  # stop before the refit; the argv is what is under test

        import src.data.generate_dataset as real
        monkeypatch.setattr(real, "main", FakeBuild.main)
        return seen

    def test_since_reaches_the_dataset_build(self, monkeypatch, capsys) -> None:
        seen = self._spy(monkeypatch)
        predict.main(["refresh", "--since", "2026"])
        assert seen, "the build was never invoked"
        assert "--since" in seen[0] and "2026" in seen[0]

    def test_it_is_absent_when_not_asked_for(self, monkeypatch, capsys) -> None:
        seen = self._spy(monkeypatch)
        predict.main(["refresh"])
        assert seen and "--since" not in seen[0]

    def test_offline_still_composes_with_it(self, monkeypatch, capsys) -> None:
        seen = self._spy(monkeypatch)
        predict.main(["refresh", "--since", "2026", "--offline"])
        assert "--since" in seen[0] and "--offline" in seen[0]

    def test_no_download_skips_the_build_entirely(self, monkeypatch) -> None:
        seen = self._spy(monkeypatch)
        try:
            predict.main(["refresh", "--no-download", "--since", "2026"])
        except Exception:
            pass  # it will fail later on a missing dataset; the build is the point
        assert not seen, "--no-download still invoked the dataset build"
