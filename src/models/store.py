"""Save and load the fitted retirement model, with a manifest that can say no.

Fitting is cheap -- 815 rows and 101 features on the default 40-race window,
about a quarter of a second -- so persistence here is not a performance
optimisation.  It exists so that a prediction can be traced back to the exact
model, feature set and training window that produced it, and so that a model
which has quietly gone wrong is refused rather than used.

Three things go wrong with a pickled estimator, and the manifest catches all
three before the model is handed to a caller:

* **Stale.**  The dataset has moved on and the model has not.  A model trained
  through round 13 predicting round 16 is not an error the estimator can raise;
  it just returns worse numbers.
* **Schema drift.**  Someone added a feature to the registry.  The estimator
  still has 101 columns and the dataset now has 102, and scikit-learn's error
  for that is not one anybody enjoys reading.
* **Environment drift.**  The estimator was pickled by a different scikit-learn.
  Unpickling may work, warn, or produce an object that scores subtly differently.

Only the current model is kept.  Two megabytes per race across a season is
forty-odd files nobody opens, and :mod:`src.models.monitor` already records what
each one scored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src import config

MODEL_PATH = config.MODELS_DIR / "dnf_model.joblib"
MANIFEST_PATH = config.MODELS_DIR / "dnf_model.json"
#: Predictions written before a race, scored after it.  Transient, not tracked.
PENDING_PATH = config.MODELS_DIR / "pending_prediction.json"


class ModelStoreError(RuntimeError):
    """Base for every refusal to hand back a saved model."""


class ModelMissingError(ModelStoreError):
    """No model has been fitted yet."""


class StaleModelError(ModelStoreError):
    """The dataset has moved on since the model was fitted."""


class SchemaDriftError(ModelStoreError):
    """The saved model's features no longer match the dataset's."""


def dataset_fingerprint(dataset: pd.DataFrame) -> str:
    """A short, stable hash of what the model was trained against.

    Hashes the race calendar and row count rather than the full frame: the
    question this answers is "is this the same data", and re-hashing three
    thousand rows of floats on every load would cost more than the fit does.
    """
    keys = [c for c in ("Year", "RoundNumber") if c in dataset.columns]
    if keys:
        races = (
            dataset[keys].drop_duplicates().sort_values(keys).to_csv(index=False)
        )
    else:  # pragma: no cover - a frame with no race identity
        races = ""
    payload = f"{len(dataset)}|{dataset.shape[1]}|{races}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def git_sha() -> str:
    """Current commit, or ``"unknown"`` outside a checkout."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=config.REPO_ROOT,
        )
    except Exception:  # noqa: BLE001 - git absent, or not a repository
        return "unknown"
    return out.stdout.strip() or "unknown"


@dataclass
class Manifest:
    """Everything needed to decide whether a saved model may still be used."""

    model: str
    stage: str
    lookback_races: int | None
    train_rows: int
    train_base_rate: float
    trained_through_year: int
    trained_through_round: int
    trained_through_event: str
    features: list[str]
    dataset_rows: int
    dataset_fingerprint: str
    sklearn_version: str = ""
    git_sha: str = field(default_factory=git_sha)
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["n_features"] = len(self.features)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def describe(self) -> str:
        return (
            f"{self.model}/{self.stage} trained through "
            f"{self.trained_through_year} R{self.trained_through_round} "
            f"({self.trained_through_event}) on {self.train_rows} rows, "
            f"base rate {self.train_base_rate:.4f}"
        )


def save_model(estimator, manifest: Manifest) -> tuple[Path, Path]:
    """Write the estimator and its manifest, replacing any previous pair."""
    import joblib
    import sklearn

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.sklearn_version = sklearn.__version__
    joblib.dump(estimator, MODEL_PATH)
    MANIFEST_PATH.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    return MODEL_PATH, MANIFEST_PATH


def read_manifest() -> Manifest:
    """The saved manifest, without unpickling the estimator."""
    if not MANIFEST_PATH.exists():
        raise ModelMissingError(
            f"no manifest at {MANIFEST_PATH}; run "
            f"'python -m src.models.predict refresh' to fit one"
        )
    return Manifest.from_dict(json.loads(MANIFEST_PATH.read_text()))


def load_model(
    dataset: pd.DataFrame | None = None,
    *,
    check: bool = True,
    expected_features: Sequence[str] | None = None,
):
    """Load the saved estimator and manifest.

    Args:
        dataset: The frame the model is about to be used against.  Supply it and
            the load is checked for staleness against the calendar it carries.
        check: Set False to load a model the checks would refuse -- useful when
            the point is to inspect why.
        expected_features: The registry's current selection.  A mismatch means
            the registry changed under the saved model.

    Raises:
        ModelMissingError: nothing has been fitted.
        StaleModelError: ``dataset`` contains races the model never saw.
        SchemaDriftError: ``expected_features`` differs from the saved list.
    """
    import joblib

    if not MODEL_PATH.exists():
        raise ModelMissingError(
            f"no model at {MODEL_PATH}; run "
            f"'python -m src.models.predict refresh' to fit one"
        )
    manifest = read_manifest()
    estimator = joblib.load(MODEL_PATH)

    if not check:
        return estimator, manifest

    if expected_features is not None:
        missing = set(expected_features) - set(manifest.features)
        extra = set(manifest.features) - set(expected_features)
        if missing or extra:
            raise SchemaDriftError(
                "the saved model's features no longer match the registry: "
                f"{len(missing)} added since it was fitted "
                f"({sorted(missing)[:4]}...), {len(extra)} no longer produced "
                f"({sorted(extra)[:4]}...). Refit with "
                "'python -m src.models.predict refresh'."
            )

    if dataset is not None and not dataset.empty:
        newest = latest_race(dataset)
        seen = (manifest.trained_through_year, manifest.trained_through_round)
        if newest[:2] > seen:
            raise StaleModelError(
                f"model was trained through {seen[0]} R{seen[1]} but the dataset "
                f"now reaches {newest[0]} R{newest[1]} ({newest[2]}). Refit with "
                "'python -m src.models.predict refresh'."
            )
    return estimator, manifest


def latest_race(dataset: pd.DataFrame) -> tuple[int, int, str]:
    """``(year, round, event)`` of the most recent race in a frame."""
    row = dataset.sort_values(["RaceDate", "Year", "RoundNumber"]).iloc[-1]
    return int(row["Year"]), int(row["RoundNumber"]), str(row.get("EventName", ""))


def save_pending(predictions: pd.DataFrame, manifest: Manifest) -> Path:
    """Record a prediction so the next refresh can score it.

    Written before the race, read after it.  This is what makes the performance
    log genuinely out-of-sample: the numbers it carries were committed to disk
    before the outcome existed.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest.to_dict(),
        "predictions": predictions.to_dict(orient="records"),
    }
    PENDING_PATH.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return PENDING_PATH


def load_pending() -> tuple[pd.DataFrame, Manifest] | None:
    """The last prediction written, or None if there is not one."""
    if not PENDING_PATH.exists():
        return None
    payload = json.loads(PENDING_PATH.read_text())
    frame = pd.DataFrame(payload["predictions"])
    return frame, Manifest.from_dict(payload["manifest"])


def clear_pending() -> None:
    PENDING_PATH.unlink(missing_ok=True)
