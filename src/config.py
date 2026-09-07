"""Project-wide configuration.

Every path is resolved relative to the repository root, so the code runs
unchanged on a laptop, in CI, or in a container.  The notebooks previously
hard-coded ``/Users/bradkittrell/Projects/...``; import from here instead.

Override any directory with an environment variable of the same name, e.g.::

    F1_DATA_DIR=/mnt/big-disk/f1 python -m scripts.build_dnf_dataset
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dir(env_var: str, default: Path) -> Path:
    """Resolve a directory from the environment, falling back to ``default``."""
    return Path(os.environ.get(env_var, default)).expanduser().resolve()


DATA_DIR = _dir("F1_DATA_DIR", REPO_ROOT / "Data")
RAW_DIR = _dir("F1_RAW_DIR", DATA_DIR / "raw")
PROCESSED_DIR = _dir("F1_PROCESSED_DIR", DATA_DIR / "processed")
EXTERNAL_DIR = _dir("F1_EXTERNAL_DIR", DATA_DIR / "external")
INTERIM_DIR = _dir("F1_INTERIM_DIR", DATA_DIR / "interim")

# FastF1 keeps its HTTP cache here.  It is large (telemetry is tens of MB per
# session) and is deliberately kept outside version control.
FASTF1_CACHE_DIR = _dir("F1_FASTF1_CACHE", RAW_DIR / "fastf1_cache")

MODELS_DIR = _dir("F1_MODELS_DIR", REPO_ROOT / "Models")
REPORTS_DIR = _dir("F1_REPORTS_DIR", REPO_ROOT / "Reports")
FIGURES_DIR = _dir("F1_FIGURES_DIR", REPORTS_DIR / "figures")

# Outputs of the DNF pipeline.
CIRCUIT_PROFILE_PATH = PROCESSED_DIR / "circuit_profiles.parquet"
RACE_RESULTS_PATH = PROCESSED_DIR / "race_results.parquet"
DNF_DATASET_PATH = PROCESSED_DIR / "dnf_dataset.parquet"

# --------------------------------------------------------------------------- #
# Season coverage
# --------------------------------------------------------------------------- #

# FastF1 exposes car and position telemetry from 2018 onward.  Results and
# finishing-status codes reach back to 1950 through Ergast/Jolpica, but the
# track-speed features this project depends on need telemetry, so 2018 is the
# floor for the modelling window.
FIRST_TELEMETRY_SEASON = 2018
#: Upper bound is exclusive and needs bumping each January.  A season still in
#: progress is worth including: its completed rounds are ordinary training rows,
#: and the rounds that have not run yet simply produce no results to collect.
LATEST_SEASON = 2027
DEFAULT_SEASONS = tuple(range(FIRST_TELEMETRY_SEASON, LATEST_SEASON))

# Formula 1's technical regulations changed materially in 2022 (ground-effect
# aerodynamics, 18-inch wheels).  Reliability and crash dynamics differ enough
# either side of that line to be worth a feature.
REGULATION_ERAS = {
    "hybrid_v2": range(2017, 2022),      # 2017-2021 wide-body cars
    "ground_effect": range(2022, 2026),  # 2022-2025
    # 2026 is a bigger break than 2022: new power units with a far larger
    # electrical share, active aerodynamics and lighter cars.  A first-year
    # formula is exactly when reliability is worst, so the era label matters
    # more here than in a settled season.
    "hybrid_v3": range(2026, 2031),
}

# --------------------------------------------------------------------------- #
# Pipeline knobs
# --------------------------------------------------------------------------- #

# Resolution of the distance grid used to resample a reference lap before
# computing curvature.  10 m keeps Monaco's tightest hairpin (~10 m radius)
# resolvable while staying cheap for a 7 km lap.
TRACK_RESAMPLE_STEP_M = 10.0

# Savitzky-Golay window (in samples of the resampled grid) for smoothing the
# X/Y trace before differentiating.  Position data is noisy at ~10 Hz; without
# smoothing the second derivative is dominated by sampling jitter.
CURVATURE_SMOOTH_WINDOW = 11
CURVATURE_SMOOTH_POLYORDER = 3

# A section of track is called a "straight" when its radius of curvature
# exceeds this.  500 m is roughly the point at which a modern F1 car is no
# longer lateral-grip limited.
STRAIGHT_RADIUS_THRESHOLD_M = 500.0

# Speed bands used to describe a circuit, in km/h.
HIGH_SPEED_THRESHOLD_KPH = 250.0
LOW_SPEED_THRESHOLD_KPH = 120.0

# Throttle pedal percentage counted as "full throttle".  FastF1 sometimes
# reports 104 as an error code; anything >= 95 is treated as full.
FULL_THROTTLE_THRESHOLD = 95.0

RANDOM_SEED = 42


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        EXTERNAL_DIR,
        INTERIM_DIR,
        FASTF1_CACHE_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def regulation_era(year: int) -> str:
    """Return the technical-regulation era label for a season."""
    for era, years in REGULATION_ERAS.items():
        if year in years:
            return era
    return "pre_2017" if year < 2017 else "unknown"
