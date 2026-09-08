"""Reconstruct the per-race performance log from a walk-forward replay.

``Reports/model_log.csv`` fills one race at a time as ``predict refresh`` runs
after each weekend, which means a freshly cloned repository has nothing to plot
and no way to see whether the model has been drifting.  This backfills it.

**What makes the replayed numbers honest.**  :func:`walk_forward_races` refits
before every race on strictly prior data, which is the same cadence service
runs at, so each row is a genuine out-of-sample score.  Nothing here scores a
race with a model that saw it.

**What makes them weaker than a live row, and why ``source`` exists.**  A
replay reads the dataset as it stands *today*.  A live row was produced against
whatever the dataset held that week -- an entry list taken from the previous
round, a grid pulled from qualifying that evening, occasionally a result that
was later corrected.  The gap is usually nil and occasionally is not, and it is
exactly the gap you would want to see if the weekly loop started producing
numbers the replay does not reproduce.  So replayed rows are written with
``source=replay`` and live rows are never overwritten by one.

Usage::

    python -m scripts.backfill_model_log                    # defaults, dry run off
    python -m scripts.backfill_model_log --start-after 2021-01-01
    python -m scripts.backfill_model_log --lookback 0       # expanding window
    python -m scripts.backfill_model_log --dry-run          # print, write nothing

Exit codes: 1 no dataset, 2 the replay produced no scoreable races.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

import pandas as pd

from src import config
from src.features.build_features import ORDER_COL
from src.models import monitor, store
from src.models.predict import DEFAULT_MODEL, DEFAULT_STAGE
from src.models.train import DEFAULT_LOOKBACK_RACES, TARGET, walk_forward_races

log = logging.getLogger(__name__)


def replay_rows(
    dataset: pd.DataFrame,
    *,
    stage: str = DEFAULT_STAGE,
    model: str = DEFAULT_MODEL,
    lookback_races: int | None = DEFAULT_LOOKBACK_RACES,
    start_after: str | None = None,
    git_sha: str | None = None,
) -> list[dict]:
    """One log row per race the replay was able to score, oldest first.

    The per-race grouping is the same unit the live log uses, so a replayed row
    and a live row are directly comparable field for field.
    """
    result = walk_forward_races(
        dataset,
        stage=stage,
        model=model,
        lookback_races=lookback_races,
        start_after=start_after,
    )
    if result.predictions.empty:
        return []

    sha = git_sha if git_sha is not None else store.git_sha()
    # RaceDate rides along on the predictions frame, so the log's race_date is
    # the real one rather than something reconstructed from a merge.
    dates = (
        dataset[["Year", "RoundNumber", ORDER_COL, "EventName"]]
        .drop_duplicates(subset=["Year", "RoundNumber"])
        .set_index(["Year", "RoundNumber"])
    )

    rows = []
    grouped = result.predictions.groupby(["Year", "RoundNumber"], observed=True)
    for (year, round_number), block in grouped:
        meta = dates.loc[(year, round_number)]
        rows.append(
            monitor.score_race(
                block[TARGET],
                block["predicted"],
                year=int(year),
                round_number=int(round_number),
                event=str(meta["EventName"]),
                race_date=meta[ORDER_COL],
                stage=stage,
                model=model,
                git_sha=sha,
                # Both vary race to race under a sliding window, and the
                # replay recorded what each fit actually used.
                train_rows=int(block["train_rows"].iloc[0]),
                train_base_rate=float(block["train_base_rate"].iloc[0]),
                lookback_races=lookback_races,
                source="replay",
            )
        )
    rows.sort(key=lambda r: (r["year"], r["round"]))
    return rows


def merge_into_log(rows: Sequence[dict], path=None) -> tuple[int, int]:
    """Write replayed rows without disturbing anything scored live.

    A live row is the stronger evidence and cost a week of waiting to earn, so
    a replay defers to it.  Returns ``(written, skipped)``.
    """
    path = path if path is not None else monitor.LOG_PATH
    existing = monitor.read_log(path)

    # A log written before the ``source`` column existed is entirely live: those
    # rows were produced by ``predict refresh`` scoring a real pending
    # prediction.  Reading a missing column as "unknown, therefore replaceable"
    # would let a backfill quietly delete the only rows that prove the weekly
    # loop ran, which is the exact failure this function exists to prevent.
    protected: set[tuple[int, int, str]] = set()
    if not existing.empty:
        source = (
            existing["source"] if "source" in existing.columns
            else pd.Series("live", index=existing.index)
        )
        live = existing.loc[source.fillna("live") == "live"]
        protected = {
            (int(r.year), int(r.round), str(r.stage))
            for r in live.itertuples(index=False)
        }

    fresh, skipped = [], 0
    for row in rows:
        if (row["year"], row["round"], row["stage"]) in protected:
            skipped += 1
            continue
        fresh.append({c: row.get(c, "") for c in monitor.LOG_COLUMNS})

    if not fresh:
        return 0, skipped

    frame = pd.DataFrame(fresh)
    if not existing.empty:
        # Drop any replayed row for a race we are about to rewrite, so a rerun
        # updates in place rather than duplicating the weekend.
        keys = {(r["year"], r["round"], r["stage"]) for r in fresh}
        keep = ~existing.apply(
            lambda r: (int(r["year"]), int(r["round"]), str(r["stage"])) in keys,
            axis=1,
        )
        existing = existing.loc[keep]
        frame = pd.concat([existing, frame], ignore_index=True)

    frame = frame.reindex(columns=list(monitor.LOG_COLUMNS))
    # Same reasoning on the way out: rows carried over from a pre-``source`` log
    # are live, and must not be written back with an empty provenance.
    frame["source"] = frame["source"].fillna("live").replace("", "live")
    frame = frame.sort_values(["year", "round"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return len(fresh), skipped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", default=DEFAULT_STAGE,
                        choices=["pre_weekend", "post_quali"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_RACES,
                        help="Races of history per fit. 0 means an expanding window.")
    parser.add_argument("--start-after", default=None,
                        help="Only score races after this date (YYYY-MM-DD). "
                             "Earlier races still count toward the lookback.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written and stop.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not config.DNF_DATASET_PATH.exists():
        print(
            f"no dataset at {config.DNF_DATASET_PATH}; build it with "
            f"'python -m src.data.generate_dataset'"
        )
        return 1

    dataset = pd.read_parquet(config.DNF_DATASET_PATH)
    dataset[ORDER_COL] = pd.to_datetime(dataset[ORDER_COL])

    lookback = None if args.lookback == 0 else args.lookback
    print(
        f"replaying {args.model}/{args.stage}, lookback "
        f"{'expanding' if lookback is None else lookback} over "
        f"{len(dataset)} rows ..."
    )
    rows = replay_rows(
        dataset,
        stage=args.stage,
        model=args.model,
        lookback_races=lookback,
        start_after=args.start_after,
    )
    if not rows:
        print(
            "the replay scored no races. Every fold was below min_train_rows, "
            "or --start-after excluded the whole calendar."
        )
        return 2

    first, last = rows[0], rows[-1]
    print(
        f"scored {len(rows)} races, {first['year']} R{first['round']} "
        f"through {last['year']} R{last['round']}"
    )
    if args.dry_run:
        frame = pd.DataFrame(rows)[
            ["year", "round", "event", "n", "observed_rate",
             "mean_predicted", "brier_skill", "top2_hits"]
        ]
        print(frame.to_string(index=False))
        print("\n--dry-run: nothing written")
        return 0

    written, skipped = merge_into_log(rows)
    print(f"wrote {written} rows to {monitor.LOG_PATH}")
    if skipped:
        print(f"kept {skipped} existing live row(s) rather than overwriting them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
