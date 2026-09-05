"""Stage tags are what stop a Saturday feature reaching a Monday model."""

from __future__ import annotations

import pytest

from src.features import registry


def test_stage_gating_is_cumulative() -> None:
    pre = set(registry.feature_columns("pre_weekend"))
    post = set(registry.feature_columns("post_quali"))
    race = set(registry.feature_columns("race_day"))
    assert pre < post < race


def test_grid_position_is_not_available_pre_weekend() -> None:
    """The single most important gate: grid slot is unknown until Saturday."""
    assert "grid_position" not in registry.feature_columns("pre_weekend")
    assert "grid_position" in registry.feature_columns("post_quali")


def test_weather_is_not_available_before_race_day() -> None:
    """Observed weather is retrospective, never a forecast input."""
    for stage in ("pre_weekend", "post_quali"):
        columns = registry.feature_columns(stage)
        assert "rain_share" not in columns
        assert "track_temp_c_mean" not in columns
    assert "rain_share" in registry.feature_columns("race_day")


def test_track_features_are_available_pre_weekend() -> None:
    """Circuit character is known months ahead; that is the point of it."""
    columns = registry.feature_columns("pre_weekend")
    for name in (
        "track_speed_index",
        "mechanical_stress_index",
        "incident_exposure_index",
        "corners_per_km",
        "pct_full_throttle",
    ):
        assert name in columns, name


def test_available_filter_respects_the_frame(labelled_results) -> None:
    columns = registry.feature_columns("post_quali", available={"grid_position"})
    assert columns == ["grid_position"]


def test_kind_filter() -> None:
    numeric_only = registry.feature_columns("race_day", kinds=("numeric",))
    assert "regulation_era" not in numeric_only
    assert "track_speed_index" in numeric_only


def test_unknown_stage_raises() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        registry.feature_columns("friday_practice")  # type: ignore[arg-type]


def test_every_feature_has_a_description() -> None:
    for feature in registry.FEATURES:
        assert feature.description.strip(), feature.name
        assert feature.description.strip().endswith("."), feature.name


def test_no_duplicate_feature_names() -> None:
    names = [f.name for f in registry.FEATURES]
    assert len(names) == len(set(names))


def test_registry_frame_round_trips() -> None:
    frame = registry.registry_frame()
    assert len(frame) == len(registry.FEATURES)
    assert set(frame["stage"]) <= set(registry.STAGE_ORDER)
