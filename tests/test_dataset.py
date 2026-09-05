"""End-to-end assembly: joins, fallbacks and the checks that gate a write."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generate_dataset import (
    build_dataset,
    join_circuit_profiles,
    parse_seasons,
    prepare_circuit_profiles,
)
from src.features import registry


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2018-2020", (2018, 2019, 2020)),
        ("2022,2024", (2022, 2024)),
        ("2023", (2023,)),
        (" 2019 - 2021 ", (2019, 2020, 2021)),
    ],
)
def test_parse_seasons(text: str, expected: tuple[int, ...]) -> None:
    assert parse_seasons(text) == expected


def test_parse_seasons_rejects_a_backwards_range() -> None:
    with pytest.raises(ValueError, match="ends before it starts"):
        parse_seasons("2025-2018")


@pytest.fixture(scope="module")
def built(raw_results, circuit_profiles):
    return build_dataset(raw_results, circuit_profiles, run_checks=True)


def test_dataset_is_one_row_per_driver_race(built) -> None:
    dataset, _ = built
    keys = dataset.groupby(["Year", "RoundNumber", "DriverId"], observed=True).size()
    assert keys.max() == 1


def test_dataset_passes_its_own_leakage_check(built) -> None:
    _, diagnostics = built
    report = diagnostics["leakage"]
    assert report.empty, report.to_string(index=False)


def test_non_starters_are_dropped(built) -> None:
    dataset, _ = built
    assert (dataset["started"] == 1).all()


def test_target_is_binary_and_plausible(built) -> None:
    dataset, _ = built
    assert set(dataset["dnf"].unique()) <= {0, 1}
    # A grand prix retires some cars but not most of them.
    assert 0.02 < dataset["dnf"].mean() < 0.5


def test_track_features_are_joined(built) -> None:
    dataset, _ = built
    assert "track_speed_index" in dataset.columns
    assert dataset["has_track_profile"].mean() == 1.0


def test_circuit_median_fills_a_missing_season(raw_results, circuit_profiles) -> None:
    """Circuit 103 has no 2023 profile; the median across its other years fills it."""
    prepared = prepare_circuit_profiles(circuit_profiles)
    assert (prepared["profile_source"] == "circuit_median").any()

    joined = join_circuit_profiles(
        raw_results.assign(Year=raw_results["Year"]), prepared
    )
    gap = joined.loc[(joined["circuit_key"] == 103) & (joined["Year"] == 2023)]
    assert not gap.empty
    assert gap["corners_per_km"].notna().all()


def test_profile_diagnostics_do_not_become_features(built) -> None:
    """Measurement metadata describes the reference lap, not the circuit."""
    dataset, _ = built
    for column in ("grid_samples", "resample_step_m", "reference_lap_time_s"):
        assert column not in dataset.columns, column


def test_registry_audit_finds_no_unregistered_features(built) -> None:
    dataset, diagnostics = built
    audit = diagnostics["registry_audit"]
    unregistered = audit.loc[audit["issue"] == "in dataset but not registered"]
    assert unregistered.empty, unregistered.to_string(index=False)


def test_sprints_inform_history_but_are_not_modelling_rows(
    raw_results, circuit_profiles
) -> None:
    with_sprints = raw_results.copy()
    sprints = with_sprints.loc[with_sprints["RoundNumber"] <= 3].copy()
    sprints["session_type"] = "S"
    combined = pd.concat([with_sprints, sprints], ignore_index=True)

    dataset, _ = build_dataset(combined, circuit_profiles, run_checks=False)
    assert (dataset["session_type"] == "R").all()
    # The sprint rows still moved the rolling history.
    baseline, _ = build_dataset(raw_results, circuit_profiles, run_checks=False)
    assert not dataset["driver_dnf_rate_10"].equals(baseline["driver_dnf_rate_10"])


def test_build_without_profiles_still_works(raw_results) -> None:
    dataset, _ = build_dataset(raw_results, None, run_checks=False)
    assert len(dataset) > 0
    assert "track_speed_index" not in dataset.columns
