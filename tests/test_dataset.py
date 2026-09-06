"""End-to-end assembly: joins, fallbacks and the checks that gate a write."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.generate_dataset import (
    build_dataset,
    main,
    join_circuit_profiles,
    parse_seasons,
    prepare_circuit_profiles,
    status_coverage,
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


# --------------------------------------------------------------------------- #
# Status coverage
# --------------------------------------------------------------------------- #
#
# A rate-limited Ergast call leaves FastF1 loading the session from timing data
# alone: a full grid of drivers and a Status column of empty strings.  Nothing
# raises, and add_race_outcome_labels reads every blank status as a retirement,
# so the round is written as a race in which all 20 cars retired.  These tests
# pin the backstop that stops such a round reaching the parquet.


def blank_a_race(results: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    """Reproduce a rate-limited round: keep the rows, empty the Status."""
    out = results.copy()
    target = (out["Year"] == year) & (out["RoundNumber"] == rnd)
    assert target.any(), "fixture has no such race"
    out.loc[target, "Status"] = ""
    return out


def test_status_coverage_is_empty_on_a_healthy_dataset(built) -> None:
    dataset, diagnostics = built
    assert diagnostics["status_coverage"].empty
    assert status_coverage(dataset).empty


def test_status_coverage_names_the_race_that_lost_its_status(raw_results) -> None:
    degraded = blank_a_race(raw_results, 2022, 5)
    report = status_coverage(degraded)

    assert len(report) == 1
    row = report.iloc[0]
    assert row["Year"] == 2022
    assert row["RoundNumber"] == 5
    assert row["with_status"] == 0
    assert row["status_share"] == 0.0
    assert row["rows"] == int(
        ((degraded["Year"] == 2022) & (degraded["RoundNumber"] == 5)).sum()
    )


def test_status_coverage_reports_partial_degradation(raw_results) -> None:
    """One blank driver row is not a rate limit, but it is still visible."""
    degraded = raw_results.copy()
    target = degraded.index[
        (degraded["Year"] == 2022) & (degraded["RoundNumber"] == 5)
    ][0]
    degraded.loc[target, "Status"] = ""

    report = status_coverage(degraded)
    assert len(report) == 1
    assert report.iloc[0]["with_status"] > 0
    assert 0 < report.iloc[0]["status_share"] < 1


def test_status_coverage_sorts_the_worst_race_first(raw_results) -> None:
    degraded = blank_a_race(raw_results, 2022, 5)
    partial = degraded.index[
        (degraded["Year"] == 2023) & (degraded["RoundNumber"] == 2)
    ][0]
    degraded.loc[partial, "Status"] = ""

    report = status_coverage(degraded)
    assert len(report) == 2
    assert report.iloc[0]["status_share"] == 0.0
    assert report.iloc[0]["RoundNumber"] == 5


@pytest.mark.parametrize(
    "blank", [pytest.param("", id="empty-string"), pytest.param(np.nan, id="null")]
)
def test_build_dataset_flags_a_degraded_race(
    raw_results, circuit_profiles, blank
) -> None:
    degraded = raw_results.copy()
    target = (degraded["Year"] == 2022) & (degraded["RoundNumber"] == 5)
    degraded.loc[target, "Status"] = blank

    dataset, diagnostics = build_dataset(degraded, circuit_profiles, run_checks=True)
    report = diagnostics["status_coverage"]

    assert not report.empty, "a fully blanked race must be reported"
    assert (report["with_status"] == 0).any()

    # And this is what the check exists to prevent: the round is in the dataset
    # with every car marked as a retirement.
    written = dataset.loc[(dataset["Year"] == 2022) & (dataset["RoundNumber"] == 5)]
    assert written["dnf"].mean() == 1.0


def test_status_coverage_survives_a_frame_without_a_status_column() -> None:
    assert status_coverage(pd.DataFrame({"Year": [2022]})).empty
    assert status_coverage(pd.DataFrame()).empty


def test_status_coverage_still_reports_when_it_cannot_name_the_race() -> None:
    """Missing race identity must not turn a real problem into a pass."""
    report = status_coverage(pd.DataFrame({"Status": ["", "", "Finished"]}))
    assert len(report) == 1
    assert report.iloc[0]["with_status"] == 1
    assert status_coverage(pd.DataFrame({"Status": ["Finished"]})).empty


# --------------------------------------------------------------------------- #
# The CLI gate
# --------------------------------------------------------------------------- #


@pytest.fixture
def cli(tmp_path, monkeypatch, circuit_profiles):
    """Run ``main`` against cached parquet, with ``--skip-download``.

    Returns a callable taking the results frame and giving back
    ``(exit_code, stdout, out_path)``.
    """
    results_path = tmp_path / "results.parquet"
    profiles_path = tmp_path / "profiles.parquet"
    out_path = tmp_path / "dnf.parquet"

    monkeypatch.setattr(config, "RACE_RESULTS_PATH", results_path)
    monkeypatch.setattr(config, "CIRCUIT_PROFILE_PATH", profiles_path)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    circuit_profiles.to_parquet(profiles_path, index=False)

    def run(results: pd.DataFrame, capsys):
        results.to_parquet(results_path, index=False)
        code = main(
            ["--skip-download", "--seasons", "2021-2023", "--out", str(out_path)]
        )
        return code, capsys.readouterr().out, out_path

    return run


def test_main_writes_a_clean_dataset(cli, raw_results, capsys) -> None:
    code, out, out_path = cli(raw_results, capsys)

    assert code == 0
    assert "status check: PASS" in out
    assert out_path.exists()


def test_main_refuses_to_write_a_race_with_no_status(
    cli, raw_results, capsys
) -> None:
    """The whole point: a rate-limited round fails the run instead of shipping."""
    code, out, out_path = cli(blank_a_race(raw_results, 2022, 5), capsys)

    assert code == 4, out
    assert "status check: FAILED" in out
    assert "1 with none at all" in out
    assert not out_path.exists(), "the poisoned dataset must not be written"


def test_main_reports_partial_degradation_without_failing(
    cli, raw_results, capsys
) -> None:
    """A single blank row is visible in the diagnostics but does not stop the run."""
    degraded = raw_results.copy()
    target = degraded.index[
        (degraded["Year"] == 2022) & (degraded["RoundNumber"] == 5)
    ][0]
    degraded.loc[target, "Status"] = ""

    code, out, out_path = cli(degraded, capsys)

    assert code == 0
    assert "status check: DEGRADED" in out
    assert "0 with none at all" in out
    assert out_path.exists()


def test_main_counts_the_rows_that_carry_a_status(cli, raw_results, capsys) -> None:
    _, out, _ = cli(raw_results, capsys)
    (line,) = [ln for ln in out.splitlines() if "status check" in ln]
    assert re.search(r"\((\d+)/\1 rows carry a finishing status\)", line), line
