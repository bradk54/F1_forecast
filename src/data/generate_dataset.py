"""Assemble the retirement-prediction dataset.

The pipeline runs in four stages, each of which can be re-run on its own:

1. **Results.**  One row per driver per race from FastF1, 2018 onward.
2. **Circuit profiles.**  One row per circuit per season, measured from a
   reference lap's telemetry.
3. **Labels and history.**  Outcome columns, then leakage-safe rolling
   reliability features.
4. **Join and check.**  Merge the profiles onto the results, then run the
   leakage detector and the registry audit before writing anything out.

The expensive stage is the second: it downloads telemetry for one session per
event.  Its output is cached to parquet, so a rebuild that only changes feature
logic never touches the network.

Run it with::

    python -m src.data.generate_dataset --seasons 2018-2025

or, for a rebuild that reuses cached intermediates::

    python -m src.data.generate_dataset --skip-download
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src import config
from src.data import circuits as circuits_mod
from src.features import registry
from src.features.build_features import build_history_features, detect_target_leakage
from src.features.labels import add_race_outcome_labels, label_summary
from src.features.track_profile import add_composite_indices, aggregate_circuit_profiles

log = logging.getLogger(__name__)

#: Profile columns that describe the measurement rather than the circuit, and
#: so should not travel into the modelling table as features.
PROFILE_DIAGNOSTIC_COLUMNS = (
    "grid_samples", "resample_step_m", "trace_closure_gap_m",
    "reference_session", "reference_driver", "reference_lap_time_s",
    "signed_turning_rad", "turn_direction_sign",
)


def parse_seasons(text: str) -> tuple[int, ...]:
    """Parse ``2018-2025`` or ``2022,2023,2024`` into a tuple of years.

    >>> parse_seasons("2018-2020")
    (2018, 2019, 2020)
    >>> parse_seasons("2022,2024")
    (2022, 2024)
    """
    text = text.strip()
    if "-" in text and "," not in text:
        start, end = (int(part) for part in text.split("-", 1))
        if end < start:
            raise ValueError(f"season range {text!r} ends before it starts")
        return tuple(range(start, end + 1))
    return tuple(int(part) for part in text.split(",") if part.strip())


# --------------------------------------------------------------------------- #
# Stage 1-2: raw collection
# --------------------------------------------------------------------------- #


def collect_raw(
    seasons: Sequence[int], *, include_sprints: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download results and circuit profiles.  Requires network access."""
    from src.data import ingest

    ingest.configure()
    reachable, message = ingest.check_connectivity()
    if not reachable:
        raise RuntimeError(message)
    log.info(message)

    results, results_report = ingest.collect_results(
        seasons, include_sprints=include_sprints
    )
    log.info("results: %s", results_report.summary())

    profiles, profile_report = ingest.collect_circuit_profiles(seasons)
    log.info("circuit profiles: %s", profile_report.summary())
    return results, profiles


def _write(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    log.info("wrote %s: %d rows x %d cols -> %s", label, len(frame), frame.shape[1], path)


# --------------------------------------------------------------------------- #
# Stage 3-4: assembly
# --------------------------------------------------------------------------- #


def prepare_circuit_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Add composite indices and a circuit-level fallback row.

    Profiles are keyed on ``(circuit_key, year)`` because layouts change.  A
    season whose telemetry failed to load falls back to the circuit's median
    profile across the seasons that did, which is better than dropping the
    race and much better than a zero.
    """
    if profiles.empty:
        return profiles
    scored = add_composite_indices(profiles)
    scored["profile_source"] = "measured"

    fallback = aggregate_circuit_profiles(scored)
    fallback = add_composite_indices(fallback)
    fallback["profile_source"] = "circuit_median"
    fallback["year"] = np.nan
    return pd.concat([scored, fallback], ignore_index=True)


def join_circuit_profiles(
    results: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Attach the season's circuit profile, falling back to the circuit median."""
    if profiles.empty:
        log.warning("no circuit profiles supplied; track features will be absent")
        return results

    feature_cols = [
        c
        for c in profiles.select_dtypes("number").columns
        if c not in ("circuit_key", "year", "profile_years")
        and c not in PROFILE_DIAGNOSTIC_COLUMNS
    ]

    exact = profiles.loc[profiles["profile_source"] == "measured"]
    median = profiles.loc[profiles["profile_source"] == "circuit_median"]

    out = results.merge(
        exact[["circuit_key", "year", *feature_cols]].rename(columns={"year": "Year"}),
        on=["circuit_key", "Year"],
        how="left",
        validate="many_to_one",
    )

    # Fill rows with no measured profile for that season from the circuit median.
    missing = out[feature_cols].isna().all(axis=1)
    if missing.any() and not median.empty:
        filled = out.loc[missing, ["circuit_key"]].merge(
            median[["circuit_key", *feature_cols]], on="circuit_key", how="left"
        )
        filled.index = out.index[missing]
        out.loc[missing, feature_cols] = filled[feature_cols]
        log.info(
            "filled %d row(s) from the circuit-median profile", int(missing.sum())
        )

    still_missing = out[feature_cols].isna().all(axis=1).sum()
    if still_missing:
        log.warning(
            "%d row(s) have no track profile at all; they will carry NaN track "
            "features", int(still_missing)
        )
    out["has_track_profile"] = (~out[feature_cols].isna().all(axis=1)).astype("int8")
    return out


def build_dataset(
    results: pd.DataFrame,
    profiles: pd.DataFrame | None = None,
    *,
    circuit_reference: bool = True,
    run_checks: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Turn raw results and circuit profiles into the modelling table.

    Args:
        results: Raw driver-race rows, as returned by
            :func:`src.data.ingest.collect_results`.
        profiles: Circuit profiles from
            :func:`src.data.ingest.collect_circuit_profiles`, or None.
        circuit_reference: Join the static street/night/altitude reference.
        run_checks: Run the leakage detector and registry audit.

    Returns:
        ``(dataset, diagnostics)``.  ``diagnostics`` carries the label summary,
        the leakage report and the registry audit, all as DataFrames.
    """
    diagnostics: dict[str, pd.DataFrame] = {}

    labelled = add_race_outcome_labels(results)
    diagnostics["label_summary"] = label_summary(labelled).reset_index()

    # Sprints are extra observations of reliability, so they stay in the
    # history that the rolling features see.  They are not modelling rows: a
    # 100 km sprint and a 305 km grand prix do not share an attrition process.
    if "session_type" in labelled.columns:
        history = labelled.loc[labelled["session_type"].isin(["R", "S"])].copy()
        n_sprints = int((history["session_type"] == "S").sum())
        if n_sprints:
            log.info("keeping %d sprint rows in the rolling history", n_sprints)
    else:
        history = labelled.copy()
        history["session_type"] = "R"

    # Non-starters are dropped: predicting a withdrawal is a different problem,
    # with different features, and leaving them in biases the base rate.
    started = history.loc[history["started"] == 1].reset_index(drop=True)
    dropped = len(history) - len(started)
    if dropped:
        log.info("dropped %d non-starter row(s)", dropped)

    featured = build_history_features(started)

    if profiles is not None and not profiles.empty:
        featured = join_circuit_profiles(featured, prepare_circuit_profiles(profiles))

    if circuit_reference and "circuit_id" in featured.columns:
        featured = circuits_mod.attach_reference(featured)
    elif circuit_reference:
        log.warning(
            "no circuit_id column, so the street/night/altitude reference was "
            "not joined; map circuit_key to Ergast circuitId to enable it"
        )

    # Modelling rows are grands prix only.
    dataset = featured.loc[featured["session_type"] == "R"].reset_index(drop=True)

    if run_checks:
        diagnostics["leakage"] = detect_target_leakage(started)
        diagnostics["registry_audit"] = registry.audit_coverage(dataset)
        if not diagnostics["leakage"].empty:
            log.error(
                "LEAKAGE DETECTED in %d feature(s):\n%s",
                len(diagnostics["leakage"]),
                diagnostics["leakage"].to_string(index=False),
            )

    return dataset, diagnostics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the F1 retirement-prediction dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        default=f"{config.DEFAULT_SEASONS[0]}-{config.DEFAULT_SEASONS[-1]}",
        help="Season range ('2018-2025') or list ('2022,2023').",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse cached results and profiles instead of hitting the network.",
    )
    parser.add_argument(
        "--no-sprints",
        action="store_true",
        help="Exclude sprint results from the rolling history.",
    )
    parser.add_argument(
        "--out", type=Path, default=config.DNF_DATASET_PATH, help="Output parquet path."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    config.ensure_dirs()
    seasons = parse_seasons(args.seasons)
    log.info("seasons: %s", ", ".join(map(str, seasons)))

    if args.skip_download:
        for path, label in (
            (config.RACE_RESULTS_PATH, "results"),
            (config.CIRCUIT_PROFILE_PATH, "circuit profiles"),
        ):
            if not path.exists():
                log.error("--skip-download needs %s but %s is missing", label, path)
                return 2
        results = pd.read_parquet(config.RACE_RESULTS_PATH)
        profiles = pd.read_parquet(config.CIRCUIT_PROFILE_PATH)
        log.info("reusing %d result rows and %d profiles", len(results), len(profiles))
    else:
        try:
            results, profiles = collect_raw(
                seasons, include_sprints=not args.no_sprints
            )
        except RuntimeError as exc:
            log.error("%s", exc)
            return 1
        if results.empty:
            log.error("no results collected; nothing to build")
            return 1
        _write(results, config.RACE_RESULTS_PATH, "results")
        if not profiles.empty:
            _write(profiles, config.CIRCUIT_PROFILE_PATH, "circuit profiles")

    results = results.loc[results["Year"].isin(seasons)]
    dataset, diagnostics = build_dataset(results, profiles)

    print("\n=== label summary ===")
    print(diagnostics["label_summary"].to_string(index=False))

    leakage = diagnostics.get("leakage")
    if leakage is not None:
        if leakage.empty:
            print("\n=== leakage check: PASS (no feature reads the current race) ===")
        else:
            print("\n=== leakage check: FAILED ===")
            print(leakage.to_string(index=False))
            return 3

    audit = diagnostics.get("registry_audit")
    if audit is not None and not audit.empty:
        print("\n=== registry audit ===")
        print(audit.to_string(index=False))

    _write(dataset, args.out, "dnf dataset")
    print(f"\nrows={len(dataset)}  columns={dataset.shape[1]}  "
          f"dnf rate={dataset['dnf'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
