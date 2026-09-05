"""Shared fixtures.  Adds the repo root to ``sys.path`` so ``src`` imports work."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.synthetic import (  # noqa: E402
    FAST_FOURIER_COEFFS,
    TWISTY_FOURIER_COEFFS,
    make_circuit_telemetry,
    make_fourier_telemetry,
    make_race_results,
    monaco_like,
    monza_like,
)


@pytest.fixture(scope="session")
def monza_telemetry() -> pd.DataFrame:
    return make_circuit_telemetry(monza_like(), step_m=1.0, elevation_amplitude_m=10.0)


@pytest.fixture(scope="session")
def monaco_telemetry() -> pd.DataFrame:
    return make_circuit_telemetry(monaco_like(), step_m=1.0, elevation_amplitude_m=42.0)


@pytest.fixture(scope="session")
def closed_telemetry() -> pd.DataFrame:
    """An exactly closed trace, to exercise the periodic code paths."""
    return make_fourier_telemetry(FAST_FOURIER_COEFFS, base_radius_m=900.0)


@pytest.fixture(scope="session")
def twisty_telemetry() -> pd.DataFrame:
    return make_fourier_telemetry(TWISTY_FOURIER_COEFFS, base_radius_m=600.0)


@pytest.fixture(scope="session")
def raw_results() -> pd.DataFrame:
    results = make_race_results(
        seasons=(2021, 2022, 2023), races_per_season=12, n_drivers=14, seed=17
    )
    results["session_type"] = "R"
    return results


@pytest.fixture(scope="session")
def labelled_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    from src.features.labels import add_race_outcome_labels

    labelled = add_race_outcome_labels(raw_results, warn_on_unmapped=False)
    return labelled.loc[labelled["started"] == 1].reset_index(drop=True)


@pytest.fixture(scope="session")
def circuit_profiles() -> pd.DataFrame:
    """A profile per circuit per season, with one deliberately withheld."""
    from src.features.track_profile import build_lap_profile

    builders = {
        100: lambda: make_circuit_telemetry(monza_like(), step_m=1.0),
        101: lambda: make_circuit_telemetry(monaco_like(), step_m=1.0),
        102: lambda: make_fourier_telemetry(FAST_FOURIER_COEFFS, base_radius_m=900.0),
        103: lambda: make_fourier_telemetry(TWISTY_FOURIER_COEFFS, base_radius_m=600.0),
        104: lambda: make_fourier_telemetry(FAST_FOURIER_COEFFS, base_radius_m=700.0),
    }
    rows = []
    for key, builder in builders.items():
        telemetry = builder()
        for year in (2021, 2022, 2023):
            if key == 103 and year == 2023:
                continue  # withheld: forces the circuit-median fallback
            rows.append(
                build_lap_profile(
                    telemetry,
                    metadata={"circuit_key": key, "circuit_name": f"c{key}", "year": year},
                )
            )
    return pd.DataFrame(rows)
