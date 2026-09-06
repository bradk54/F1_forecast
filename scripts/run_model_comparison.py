"""Compare every model, estimate the race shock, and report what it means.

Run this after the dataset exists::

    python -m src.data.generate_dataset --seasons 2018-2025
    python -m scripts.run_model_comparison

It writes a set of CSVs to ``Reports/`` and prints a summary.  Read the output
in this order:

1. **Model comparison** — ``brier_skill`` first (is it worth anything?), then
   ``calibration_slope`` (can the probabilities be trusted?).  For a downstream
   points model the second column matters more than the first: a model that
   ranks well but sits at a slope of 0.7 will bias every expected-points
   number.
2. **Track ablation** — does measuring the circuit beat ignoring it?
3. **Race shock** — how much of retirement risk is common to everyone in a
   race, and therefore invisible to a model that treats drivers as independent.
4. **Points impact** — what that correlation is worth in points per driver.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.models import train  # noqa: E402
from src.models.hierarchical import estimate_race_shock  # noqa: E402
from src.models.points_bridge import build_points_inputs, correlation_impact  # noqa: E402

log = logging.getLogger("model_comparison")


def _banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare retirement models and quantify the race shock.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=Path, default=config.DNF_DATASET_PATH,
        help="Parquet written by src.data.generate_dataset.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=config.REPORTS_DIR,
        help="Where to write the result CSVs.",
    )
    parser.add_argument(
        "--models", default=",".join(train.DEFAULT_MODEL_SUITE),
        help="Comma-separated model keys to compare.",
    )
    parser.add_argument(
        "--stage", default="post_quali",
        choices=("pre_weekend", "post_quali", "race_day"),
        help="Stage used for the ablation, shock and points sections.",
    )
    parser.add_argument("--n-sims", type=int, default=20000)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    if not args.verbose:
        # The one-hot encoder is configured with handle_unknown="ignore" on
        # purpose: a season can introduce a regulation era the training folds
        # have never seen, and encoding it as all-zeros is the intended
        # behaviour.  The warning fires once per fold and drowns the report.
        import warnings

        warnings.filterwarnings(
            "ignore", message="Found unknown categories", category=UserWarning
        )

    if not args.dataset.exists():
        print(
            f"Dataset not found at {args.dataset}.\n"
            "Build it first:\n"
            "    python -m src.data.generate_dataset --seasons 2018-2025",
            file=sys.stderr,
        )
        return 2

    dataset = pd.read_parquet(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _banner("DATASET")
    print(f"  rows          : {len(dataset):,}")
    print(f"  columns       : {dataset.shape[1]}")
    print(f"  seasons       : {int(dataset['Year'].min())}-{int(dataset['Year'].max())}")
    print(f"  retirement rate: {dataset['dnf'].mean():.4f}")
    if "dnf_cause" in dataset.columns:
        causes = dataset.loc[dataset["dnf"] == 1, "dnf_cause"].value_counts()
        print("  causes        :", ", ".join(f"{k}={v}" for k, v in causes.items()))

    # ---------------------------------------------------------------- models
    _banner("MODEL COMPARISON (walk-forward by season)")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    comparison = train.compare_models(dataset, models=models)
    if comparison.empty:
        print("  no model produced a usable fold", file=sys.stderr)
        return 1
    print(comparison.to_string(index=False))
    comparison.to_csv(args.out_dir / "model_comparison.csv", index=False)

    stage_rows = comparison.loc[comparison["stage"] == args.stage]
    best_skill = stage_rows.iloc[0]["model"]

    # Pick the model to carry forward on skill *subject to* calibration being
    # good enough, rather than on calibration alone.  Chasing the flattest
    # calibration slope will happily trade away real skill for a cosmetic
    # improvement, and a well-calibrated model that knows nothing is no use to
    # a points forecast either.
    CALIBRATION_BAND = (0.85, 1.15)
    candidates = stage_rows.loc[
        stage_rows["calibration_slope"].between(*CALIBRATION_BAND)
        & (stage_rows["brier_skill"] > 0)
    ]
    if not candidates.empty:
        best_calibrated = candidates.sort_values(
            "brier_skill", ascending=False
        ).iloc[0]["model"]
        note = f"best skill among models calibrated within {CALIBRATION_BAND}"
    else:
        fallback = stage_rows.loc[stage_rows["brier_skill"] > 0].copy()
        if fallback.empty:
            fallback = stage_rows.copy()
        fallback["slope_error"] = (fallback["calibration_slope"] - 1.0).abs()
        best_calibrated = fallback.sort_values("slope_error").iloc[0]["model"]
        note = "no model was well calibrated; fell back to the closest slope"

    print()
    print(f"  best by skill        : {best_skill}")
    print(f"  carried forward      : {best_calibrated}")
    print(f"    ({note})")
    print("    Calibration is the binding constraint for a downstream points model:")
    print("    expected points scales linearly in P(finish), so a slope of 0.7 biases")
    print("    every driver's forecast even when the ranking is right.")

    # -------------------------------------------------------------- ablation
    if not args.skip_ablation:
        _banner("FEATURE ABLATION (delta < 0 means the group was carrying weight)")
        groups = {
            "track profile": train.track_feature_group(dataset),
            "driver history": [c for c in dataset.columns if c.startswith("driver_")],
            "team history": [
                c for c in dataset.columns if c.startswith(("team_", "teammate_"))
            ],
            "circuit attrition history": [
                c for c in dataset.columns if c.startswith("circuit_")
            ],
            "grid position": [
                "grid_position", "grid_position_pct", "is_back_half_of_grid"
            ],
        }
        ablation = train.ablate_features(
            dataset, groups, stage=args.stage, model=best_calibrated
        )
        print(ablation.to_string(index=False))
        ablation.to_csv(args.out_dir / "feature_ablation.csv", index=False)

    # ------------------------------------------------------------ race shock
    _banner("RACE-LEVEL SHOCK (shared risk a per-driver model cannot see)")
    result = train.walk_forward_evaluate(
        dataset, stage=args.stage, model=best_calibrated
    )
    predictions = result.predictions.copy()
    predictions["race_id"] = (
        predictions["Year"].astype(str) + "_" + predictions["RoundNumber"].astype(str)
    )
    # Out-of-sample predictions only: in-sample ones sit too close to the
    # outcomes and shrink the estimated shock toward zero.
    shock = estimate_race_shock(
        predictions["dnf"], predictions["predicted"], predictions["race_id"]
    )
    print("  " + shock.summary().replace("\n", "\n  "))
    pd.DataFrame([{
        "sigma": shock.sigma,
        "intraclass_correlation": shock.intraclass_correlation,
        "likelihood_ratio": shock.likelihood_ratio,
        "n_races": shock.n_races,
        "n_rows": shock.n_rows,
    }]).to_csv(args.out_dir / "race_shock.csv", index=False)

    if shock.sigma == 0:
        print("\n  No shared shock detected: retirements in this data look independent")
        print("  once the features are accounted for. That is a real finding, not a")
        print("  failure — it means the marginal model is enough for points.")
        return 0

    # --------------------------------------------------------- points impact
    _banner("WHAT THE SHOCK IS WORTH IN POINTS")
    latest = predictions.loc[predictions["Year"] == predictions["Year"].max()].copy()
    inputs = build_points_inputs(
        latest.merge(
            dataset[["Year", "RoundNumber", "DriverId", "grid_position"]],
            on=["Year", "RoundNumber", "DriverId"],
            how="left",
        ),
        latest["predicted"],
    )
    impact = correlation_impact(
        inputs, shock.sigma, n_sims=args.n_sims, noise_scale=0.6
    )
    by_grid = (
        impact.assign(grid=pd.to_numeric(inputs["grid_position"], errors="coerce"))
        .assign(grid_band=lambda d: pd.cut(
            d["grid"], [0, 5, 10, 15, 30],
            labels=["1-5", "6-10", "11-15", "16+"],
        ))
        .groupby("grid_band", observed=True)[
            ["expected_points_correlated", "expected_points_independent",
             "points_sd_correlated", "points_sd_independent"]
        ]
        .mean()
        .round(3)
    )
    by_grid["expected_points_delta"] = (
        by_grid["expected_points_correlated"] - by_grid["expected_points_independent"]
    ).round(3)
    print(by_grid.to_string())
    impact.to_csv(args.out_dir / "points_correlation_impact.csv", index=False)

    print()
    print("  A negative delta at the front and a positive one at the back is the")
    print("  expected pattern: a back-marker's points are a convex function of how")
    print("  much attrition a race produces, so adding variance to attrition while")
    print("  holding its mean transfers expected points down the grid. A points")
    print("  model built on independent retirements under-rates the back of the")
    print("  grid by roughly this much.")

    _banner("FILES WRITTEN")
    for name in (
        "model_comparison.csv", "feature_ablation.csv",
        "race_shock.csv", "points_correlation_impact.csv",
    ):
        path = args.out_dir / name
        if path.exists():
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
