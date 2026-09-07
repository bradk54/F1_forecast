"""Track how the model does, one race at a time.

The unit is a race, not a season and not a row.  A race is the cadence the
model runs at, it keeps the file to roughly twenty-four rows a season, and it
matches how a problem actually announces itself -- three bad weekends in a row,
not a bad year in hindsight.

Two properties of this log are worth stating because they change how it reads:

* **``roc_auc`` is often NaN, and that is not a bug.**  It is undefined when a
  race has no retirements, which at current rates is about a third of them.
  Read it as a rolling mean over five or ten races; a single race's AUC is
  noise built from twenty rows.
* **Skill is measured against the training base rate, not the race's own.**  A
  model cannot know in advance that this particular Sunday would be a 25%
  attrition race, so scoring it against 25% would flatter it for something it
  never predicted.  :func:`src.models.train.score_predictions` takes the same
  view.

The file is committed.  It is a few kilobytes, and the entire point is to be
able to read drift out of the diff.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.models.train import score_predictions

log = logging.getLogger(__name__)

LOG_PATH = config.REPORTS_DIR / "model_log.csv"

#: Column order, fixed so the CSV diffs cleanly.
LOG_COLUMNS = (
    "scored_at", "year", "round", "event", "race_date",
    "stage", "model", "git_sha",
    "train_rows", "train_base_rate", "lookback_races",
    "n", "observed_rate", "mean_predicted",
    "brier", "brier_base", "brier_skill", "log_loss",
    "roc_auc", "pr_auc", "top2_hits", "top2_lift",
)


def top_k_lift(
    outcomes: pd.Series, predicted: pd.Series, k: int = 2
) -> tuple[int, float]:
    """Retirements caught in the k riskiest picks, and the lift over chance.

    This is the number a person actually acts on: if you flagged two cars this
    weekend, did either of them stop.  Lift is NaN when nobody retired, because
    dividing by a zero base rate says nothing.
    """
    frame = pd.DataFrame({"y": np.asarray(outcomes), "p": np.asarray(predicted)})
    top = frame.nlargest(k, "p")
    hits = int(top["y"].sum())
    base = float(frame["y"].mean())
    lift = float(top["y"].mean() / base) if base > 0 else np.nan
    return hits, lift


def score_race(
    outcomes: pd.Series,
    predicted: pd.Series,
    *,
    year: int,
    round_number: int,
    event: str,
    race_date,
    stage: str,
    model: str,
    git_sha: str,
    train_rows: int,
    train_base_rate: float,
    lookback_races: int | None,
) -> dict:
    """Build one log row from a race's outcomes and the predictions made for it."""
    scores = score_predictions(outcomes, predicted, train_base_rate)
    hits, lift = top_k_lift(outcomes, predicted)
    return {
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "year": int(year),
        "round": int(round_number),
        "event": event,
        "race_date": pd.Timestamp(race_date).date().isoformat(),
        "stage": stage,
        "model": model,
        "git_sha": git_sha,
        "train_rows": int(train_rows),
        "train_base_rate": round(float(train_base_rate), 5),
        "lookback_races": "" if lookback_races is None else int(lookback_races),
        "n": int(scores.get("n", 0)),
        "observed_rate": round(float(scores.get("observed_rate", np.nan)), 5),
        "mean_predicted": round(float(scores.get("mean_predicted", np.nan)), 5),
        "brier": round(float(scores.get("brier", np.nan)), 6),
        "brier_base": round(float(scores.get("brier_base_rate", np.nan)), 6),
        "brier_skill": round(float(scores.get("brier_skill", np.nan)), 5),
        "log_loss": round(float(scores.get("log_loss", np.nan)), 5),
        "roc_auc": round(float(scores.get("roc_auc", np.nan)), 5)
        if np.isfinite(scores.get("roc_auc", np.nan)) else "",
        "pr_auc": round(float(scores.get("pr_auc", np.nan)), 5)
        if np.isfinite(scores.get("pr_auc", np.nan)) else "",
        "top2_hits": hits,
        "top2_lift": round(lift, 3) if np.isfinite(lift) else "",
    }


def append_row(row: dict, path: Path = LOG_PATH) -> Path:
    """Append one race to the log, creating it with a header if absent.

    A race already present is replaced rather than duplicated, so re-running a
    refresh after fixing something does not leave two versions of the same
    weekend in the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{c: row.get(c, "") for c in LOG_COLUMNS}])
    if path.exists():
        existing = pd.read_csv(path)
        duplicate = (existing["year"] == row["year"]) & (
            existing["round"] == row["round"]
        )
        if "stage" in existing.columns:
            duplicate &= existing["stage"] == row["stage"]
        if duplicate.any():
            log.info("replacing existing log row for %s R%s", row["year"], row["round"])
            existing = existing.loc[~duplicate]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.sort_values(["year", "round"]).reset_index(drop=True)
    frame.to_csv(path, index=False)
    return path


def read_log(path: Path = LOG_PATH) -> pd.DataFrame:
    """The log, or an empty frame with the right columns if there is none."""
    if not path.exists():
        return pd.DataFrame(columns=list(LOG_COLUMNS))
    return pd.read_csv(path)


def rolling_summary(window: int = 10, path: Path = LOG_PATH) -> pd.Series:
    """Headline numbers over the most recent ``window`` races.

    Means skip NaN, which is what makes ``roc_auc`` readable here and unreadable
    race by race.
    """
    frame = read_log(path)
    if frame.empty:
        return pd.Series(dtype="float64")
    recent = frame.tail(window)
    out = {
        "races": len(recent),
        "rows": recent["n"].sum(),
        "observed_rate": recent["observed_rate"].mean(),
        "mean_predicted": recent["mean_predicted"].mean(),
        "brier_skill": recent["brier_skill"].mean(),
        "roc_auc": pd.to_numeric(recent["roc_auc"], errors="coerce").mean(),
        "pr_auc": pd.to_numeric(recent["pr_auc"], errors="coerce").mean(),
        "top2_lift": pd.to_numeric(recent["top2_lift"], errors="coerce").mean(),
    }
    return pd.Series(out).round(4)


def drift_check(
    dataset: pd.DataFrame,
    train_base_rate: float,
    *,
    window: int = 5,
    tolerance: float = 0.04,
) -> tuple[bool, str]:
    """Compare recent observed attrition against what the model was trained on.

    2026 is the reason this exists.  New power-unit regulations took the
    retirement rate from 10.3% to 19.7% while the model was still trained at
    14.2%, and nothing in the pipeline said so -- the predictions simply came
    out too low.  A gap this check would have caught in two races.

    Returns ``(ok, message)``.  It never fails a run: a regime change is
    information, not an error, and the response to it is to re-tune the lookback
    window rather than to stop predicting.
    """
    if dataset.empty or "dnf" not in dataset.columns:
        return True, "no data to check drift against"
    races = (
        dataset[["Year", "RoundNumber", "RaceDate"]]
        .drop_duplicates()
        .sort_values("RaceDate")
        .tail(window)
    )
    recent = dataset.merge(races[["Year", "RoundNumber"]], on=["Year", "RoundNumber"])
    if recent.empty:
        return True, "no recent races to check drift against"
    observed = float(recent["dnf"].mean())
    gap = observed - train_base_rate
    message = (
        f"last {len(races)} races observed {observed:.1%} vs "
        f"{train_base_rate:.1%} trained ({gap:+.1%})"
    )
    if abs(gap) > tolerance:
        return False, (
            f"DRIFT: {message}. The model is predicting against a base rate the "
            f"sport has moved away from; consider re-tuning lookback_races."
        )
    return True, message
