"""CLI for ranking/finishing-position models.

Evaluate Plackett-Luce and other ranking models from the terminal::

    python -m src.models.rank_predict evaluate --feature-set grid_only
    python -m src.models.rank_predict evaluate --feature-set grid_and_team
    python -m src.models.rank_predict evaluate --feature-set full

The key differences from the DNF model (src.models.predict):

1. **By-race training**: Ranking models train and score at the race level
   (one ranking per race), not the driver-race level.

2. **No stage parameter**: Grid position is always available post-qualifying.
   For now, we do not build pre-weekend rankings.

3. **Different metrics**: Spearman correlation, Mean Absolute Error in positions,
   top-3 and top-10 accuracy (how often we predict the points-scorers correctly).

4. **Baselines are simpler**: Grid order as a baseline; no logistic on grid alone.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src import config
from src.features.build_features import ORDER_COL
from src.models.ranking import walk_forward_races_ranked

log = logging.getLogger(__name__)


def load_dataset() -> pd.DataFrame:
    """Load the DNF dataset with finishing position information."""
    if not config.DNF_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"no dataset at {config.DNF_DATASET_PATH}; build it with "
            f"'python -m src.data.generate_dataset --seasons 2018-2025'"
        )
    frame = pd.read_parquet(config.DNF_DATASET_PATH)
    frame[ORDER_COL] = pd.to_datetime(frame[ORDER_COL])
    return frame


def main(argv: list[str] | None = None) -> int:
    """Entry point for ranking model CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate finishing-position ranking models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run walk-forward evaluation on a feature set.",
    )
    eval_parser.add_argument(
        "--feature-set",
        choices=["grid_only", "grid_and_team", "full"],
        default="grid_only",
        help="Which features to use.",
    )
    eval_parser.add_argument(
        "--model",
        choices=["plackett_luce"],
        default="plackett_luce",
        help="Which model to use.",
    )
    eval_parser.add_argument(
        "--lookback",
        type=int,
        default=40,
        help="Train on only the last N races. 0 = expanding window.",
    )
    eval_parser.add_argument(
        "--start-after",
        type=str,
        default=None,
        help="Only score races after this date (YYYY-MM-DD).",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save predictions to this CSV file.",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Load dataset.
    try:
        dataset = load_dataset()
    except FileNotFoundError as e:
        log.error(str(e))
        return 1

    if args.command == "evaluate":
        lookback = args.lookback if args.lookback > 0 else None
        result = walk_forward_races_ranked(
            dataset,
            model=args.model,
            lookback_races=lookback,
            start_after=args.start_after,
            feature_set=args.feature_set,
        )

        # Print results.
        print("\n=== Scores by Season ===\n")
        print(result.scores.to_string(index=False))

        print("\n=== Summary ===\n")
        summary = {
            "Model": result.model_name,
            "Features": ", ".join(result.features),
            "Mean Spearman": f"{result.scores['spearman'].mean():.3f}",
            "Mean MAE": f"{result.scores['mae'].mean():.2f}",
            "Mean Top-3 Acc": f"{result.scores['top3_accuracy'].mean():.1%}",
            "Mean Top-10 Acc": f"{result.scores['top10_accuracy'].mean():.1%}",
        }
        for key, value in summary.items():
            print(f"{key:20} {value}")

        # Save predictions if requested.
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.predictions.to_csv(output_path, index=False)
            print(f"\nPredictions saved to {output_path}")

        return 0

    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
