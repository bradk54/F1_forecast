"""Order features must read only the past, follow the team, and track the truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import registry
from src.features.order_features import (
    ORDER_FEATURES,
    add_order_features,
    detect_order_leakage,
)
from tests.synthetic import make_order_results


def _modelling_table(results: pd.DataFrame) -> pd.DataFrame:
    from src.data.generate_dataset import build_dataset

    dataset, _ = build_dataset(results, None, circuit_reference=False, run_checks=False)
    return dataset


@pytest.fixture(scope="module")
def order_dataset() -> pd.DataFrame:
    return _modelling_table(make_order_results(seed=3))


def test_every_order_feature_is_registered_as_built_at_fit_time() -> None:
    for name in ORDER_FEATURES:
        assert name in registry.BY_NAME, name
        assert registry.BY_NAME[name].table == "order_frame", name


def test_grid_terms_are_not_available_before_qualifying() -> None:
    pre = set(registry.feature_columns("pre_weekend"))
    for name in ("log_grid_position", "grid_x_circuit_retention", "grid_x_rain_share"):
        assert name not in pre


def test_order_features_do_not_leak(order_dataset) -> None:
    for year, rnd in ((2022, 6), (2023, 7), (2023, 12)):
        report = detect_order_leakage(order_dataset, flip_year=year, flip_round=rnd)
        assert report.empty, report.to_string()


def test_the_leakage_detector_catches_a_leak(order_dataset) -> None:
    """A detector that never fires proves nothing; plant a leak and find it."""

    def leaky(frame: pd.DataFrame) -> pd.DataFrame:
        out = add_order_features(frame)
        out["team_grid_pct_ewma"] = pd.to_numeric(out["ClassifiedPosition"], errors="coerce")
        return out

    report = detect_order_leakage(order_dataset, builder=leaky, flip_year=2023, flip_round=6)
    assert "team_grid_pct_ewma" in set(report["feature"])


def test_pre_weekend_features_ignore_the_current_grid(order_dataset) -> None:
    """Reversing a race's grid may move its post-quali terms, never a Monday feature."""
    report = detect_order_leakage(order_dataset, flip_year=2023, flip_round=8)
    assert report.empty
    built = add_order_features(order_dataset)
    race = (built["Year"] == 2023) & (built["RoundNumber"] == 8)
    assert built.loc[race, "log_grid_position"].notna().all()


def _first_race_under_new_name(results: pd.DataFrame, new_id: str) -> pd.Series:
    built = add_order_features(_modelling_table(results))
    return built.loc[built["TeamId"] == new_id].sort_values("RaceDate").iloc[0]


def test_team_history_survives_a_known_rebrand() -> None:
    """Renault became Alpine; the history should carry straight across."""
    results = make_order_results(rebrand=("team_02", "alpine", 2023), seed=4)
    results.loc[results["TeamId"] == "team_02", "TeamId"] = "renault"
    first = _first_race_under_new_name(results, "alpine")
    assert first["Year"] == 2023 and first["RoundNumber"] == 1
    assert np.isfinite(first["team_grid_pct_ewma"])
    assert np.isfinite(first["team_finish_pct_ewma"])


def test_an_unknown_rename_starts_from_nothing() -> None:
    """The contrast: keyed on the entry name, a rename throws the history away."""
    results = make_order_results(rebrand=("team_02", "team_new", 2023), seed=4)
    first = _first_race_under_new_name(results, "team_new")
    assert np.isnan(first["team_grid_pct_ewma"])


def test_team_pace_tracks_the_true_pace(order_dataset) -> None:
    built = add_order_features(order_dataset)
    truth = make_order_results(seed=3)[["Year", "RoundNumber", "DriverId", "_true_team_pace"]]
    merged = built.merge(truth, on=["Year", "RoundNumber", "DriverId"])
    late = merged.loc[merged["Year"] >= 2022]
    # Lower percentile is faster, so the correlation with true pace is negative.
    assert late["team_grid_pct_ewma"].corr(late["_true_team_pace"]) < -0.9


def test_percentiles_are_percentiles(order_dataset) -> None:
    built = add_order_features(order_dataset)
    for name in ("team_grid_pct_ewma", "team_finish_pct_ewma", "driver_grid_pct_ewma"):
        values = built[name].dropna()
        assert values.between(0, 1).all(), name
