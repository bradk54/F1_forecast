"""Geometry has closed-form answers, so the profiler can be checked exactly.

A circle of radius R has curvature 1/R at every point and turns through 2*pi.
If the profiler reproduces that, the differentiation, smoothing and resampling
are all sound, and the derived quantities built on them (lateral load, corner
radius, straight share) inherit that confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.track_profile import (
    add_composite_indices,
    aggregate_circuit_profiles,
    build_lap_profile,
    compute_curvature,
    resample_to_distance_grid,
)
from tests.synthetic import make_circuit_telemetry, oval


@pytest.mark.parametrize("radius", [80.0, 200.0, 500.0, 1200.0])
def test_curvature_matches_closed_form_on_a_circle(radius: float) -> None:
    telemetry = make_circuit_telemetry(oval(radius), step_m=1.0, sample_by_time=False)
    x_m = telemetry["X"].to_numpy() / 10.0
    y_m = telemetry["Y"].to_numpy() / 10.0

    kappa, computed_radius = compute_curvature(x_m, y_m, 1.0)

    assert kappa.mean() == pytest.approx(1.0 / radius, rel=1e-4)
    assert computed_radius.mean() == pytest.approx(radius, rel=1e-4)
    # Curvature is constant on a circle, so dispersion must be negligible.
    assert kappa.std() < 1e-8


def test_total_turning_of_a_closed_lap_is_two_pi() -> None:
    """The turning-tangent theorem, used to detect track direction."""
    telemetry = make_circuit_telemetry(oval(400.0), step_m=1.0, sample_by_time=False)
    kappa, _ = compute_curvature(
        telemetry["X"].to_numpy() / 10.0, telemetry["Y"].to_numpy() / 10.0, 1.0
    )
    assert kappa.sum() * 1.0 == pytest.approx(2 * np.pi, rel=1e-3)


def test_turn_direction_sign_flips_with_the_layout() -> None:
    from tests.synthetic import Segment, build_trace

    left = make_circuit_telemetry(oval(300.0), step_m=1.0, sample_by_time=False)
    right = make_circuit_telemetry(
        [Segment(2 * np.pi * 300.0, -300.0)], step_m=1.0, sample_by_time=False
    )
    left_profile = build_lap_profile(left)
    right_profile = build_lap_profile(right)
    assert left_profile["turn_direction_sign"] == -right_profile["turn_direction_sign"]


def test_distance_resampling_removes_the_slow_corner_bias() -> None:
    """Telemetry is sampled in time, so slow corners are over-represented.

    Averaging raw samples therefore reports a lap as slower than it is.
    Resampling by distance is what makes circuits comparable.
    """
    telemetry = make_circuit_telemetry(
        oval(120.0), step_m=1.0, sample_by_time=True
    )
    # A track with one constant radius has one constant speed, so this circuit
    # cannot show the bias; use a mixed layout instead.
    from tests.synthetic import monaco_like

    mixed = make_circuit_telemetry(monaco_like(), step_m=1.0, sample_by_time=True)
    raw_mean = mixed["Speed"].mean()
    grid_mean = resample_to_distance_grid(mixed, step_m=10.0)["Speed"].mean()
    assert grid_mean > raw_mean  # the time-sampled mean is dragged down


def test_fast_and_slow_circuits_separate(monza_telemetry, monaco_telemetry) -> None:
    fast = build_lap_profile(monza_telemetry)
    slow = build_lap_profile(monaco_telemetry)

    assert fast["speed_mean_kph"] > slow["speed_mean_kph"]
    assert fast["pct_dist_above_250kph"] > slow["pct_dist_above_250kph"]
    assert fast["corners_per_km"] < slow["corners_per_km"]
    assert fast["median_corner_radius_m"] > slow["median_corner_radius_m"]
    assert fast["pct_dist_straight"] > slow["pct_dist_straight"]
    assert fast["braking_zones_per_km"] < slow["braking_zones_per_km"]


def test_speed_index_orders_circuits(monza_telemetry, monaco_telemetry) -> None:
    profiles = pd.DataFrame(
        [
            build_lap_profile(monza_telemetry, metadata={"circuit_key": 1}),
            build_lap_profile(monaco_telemetry, metadata={"circuit_key": 2}),
        ]
    )
    scored = add_composite_indices(profiles)
    assert scored.loc[0, "track_speed_index"] > scored.loc[1, "track_speed_index"]
    assert scored.loc[1, "incident_exposure_index"] > scored.loc[0, "incident_exposure_index"]


def test_elevation_is_recovered() -> None:
    telemetry = make_circuit_telemetry(
        oval(500.0), step_m=1.0, elevation_amplitude_m=60.0, sample_by_time=False
    )
    profile = build_lap_profile(telemetry)
    assert profile["elevation_range_m"] == pytest.approx(60.0, rel=0.02)


def test_closed_trace_is_detected(closed_telemetry) -> None:
    """A Fourier layout closes, so the periodic smoothing path should run."""
    profile = build_lap_profile(closed_telemetry)
    assert profile["trace_closure_gap_m"] < 50.0
    assert abs(profile["signed_turning_rad"]) == pytest.approx(2 * np.pi, rel=0.15)


def test_missing_optional_channels_degrade_to_nan() -> None:
    telemetry = make_circuit_telemetry(oval(400.0), step_m=1.0, sample_by_time=False)
    stripped = telemetry.drop(columns=["Throttle", "Brake", "nGear", "DRS", "RPM", "Z"])
    profile = build_lap_profile(stripped)
    # Required geometry still computes.
    assert profile["speed_mean_kph"] > 0
    # Optional channels report NaN rather than a fabricated zero.
    for column in ("pct_full_throttle", "pct_braking", "n_gear_changes", "pct_drs_open"):
        assert np.isnan(profile[column]), column


def test_missing_required_channel_raises() -> None:
    telemetry = make_circuit_telemetry(oval(400.0), step_m=1.0, sample_by_time=False)
    with pytest.raises(KeyError, match="required channels"):
        build_lap_profile(telemetry.drop(columns=["X"]))


def test_too_few_samples_raises() -> None:
    tiny = pd.DataFrame(
        {"Distance": [0.0, 1.0], "X": [0.0, 1.0], "Y": [0.0, 0.0], "Speed": [10.0, 11.0]}
    )
    with pytest.raises(ValueError, match="too few"):
        build_lap_profile(tiny)


def test_corner_count_prefers_the_official_table(monaco_telemetry) -> None:
    corners = pd.DataFrame({"Number": range(1, 20), "X": 0.0, "Y": 0.0})
    profile = build_lap_profile(monaco_telemetry, corners=corners)
    assert profile["n_corners"] == 19.0


def test_aggregate_handles_a_numeric_group_key(circuit_profiles) -> None:
    """Regression: ``circuit_key`` is numeric and was colliding on reset_index."""
    aggregated = aggregate_circuit_profiles(circuit_profiles)
    assert "circuit_key" in aggregated.columns
    assert aggregated["circuit_key"].is_unique
    assert (aggregated["profile_years"] > 0).all()


def test_resample_step_changes_resolution_not_conclusions(monaco_telemetry) -> None:
    """A finer grid should not move the headline numbers much."""
    coarse = build_lap_profile(monaco_telemetry, step_m=20.0)
    fine = build_lap_profile(monaco_telemetry, step_m=5.0)
    assert fine["speed_mean_kph"] == pytest.approx(coarse["speed_mean_kph"], rel=0.05)
    assert fine["lap_length_m"] == pytest.approx(coarse["lap_length_m"], rel=0.02)
