"""Everything the dashboard needs from ``src/``, cached and made safe to call.

The rule this module exists to enforce: **the app never reimplements the
pipeline.**  Every number on screen comes from the same function the CLI calls,
so a chart cannot drift away from what ``python -m src.models.predict`` would
print.  What lives here is caching, error handling, and the small amount of
reshaping a chart needs.

**Cache invalidation is keyed on file mtimes, not on time.**  The dataset gets
rebuilt from a terminal while the app is open -- that is the normal workflow, not
an edge case -- and a dashboard showing yesterday's parquet with today's title
is worse than no dashboard.  :func:`data_version` stats the artefacts and is
threaded through every cached function as an argument, so a rebuild invalidates
exactly the entries that depended on the rebuilt file.

**The version parameter must never be named with a leading underscore.**
Streamlit reads that prefix as "do not hash this argument", so an underscored
version tuple is excluded from the cache key and the entry never invalidates --
the app goes on serving the frame it loaded at startup, through any number of
rebuilds, with no error and no visible symptom until a number looks wrong.  The
underscore is reserved here for the opposite case: a large frame passed only to
be read, which is genuinely too expensive to hash and is already accounted for
by the version alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import streamlit as st

from src import config
from src.features import registry
from src.features.build_features import ORDER_COL
from src.models import monitor, store
from src.models.train import (
    DEFAULT_LOOKBACK_RACES,
    MODEL_FACTORIES,
    TARGET,
    WalkForwardResult,
    ablate_features,
    permutation_importance_report,
    race_bootstrap,
    track_feature_group,
    walk_forward_evaluate,
    walk_forward_races,
)

STAGES = ("pre_weekend", "post_quali")
MODELS = tuple(sorted(MODEL_FACTORIES))

#: Metrics that read better when higher, for colouring deltas.
HIGHER_IS_BETTER = frozenset(
    {"brier_skill", "roc_auc", "pr_auc", "top2_lift", "top2_hits"}
)


# --------------------------------------------------------------------------- #
# Cache keys
# --------------------------------------------------------------------------- #


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def data_version() -> tuple[float, ...]:
    """A fingerprint of every artefact on disk the app reads.

    Passed as an argument to cached functions so that rebuilding the dataset in
    another terminal drops the stale entries.  Cheap enough to call on every
    rerun; four ``stat`` calls.
    """
    return (
        _mtime(config.DNF_DATASET_PATH),
        _mtime(config.RACE_RESULTS_PATH),
        _mtime(config.CIRCUIT_PROFILE_PATH),
        _mtime(store.MANIFEST_PATH),
        _mtime(monitor.LOG_PATH),
    )


# --------------------------------------------------------------------------- #
# Loading what is on disk
# --------------------------------------------------------------------------- #


@dataclass
class Artefacts:
    """What exists on disk, so a page can degrade instead of raising."""

    dataset: bool
    results: bool
    profiles: bool
    model: bool
    log: bool

    @property
    def ready(self) -> bool:
        return self.dataset


def artefacts() -> Artefacts:
    return Artefacts(
        dataset=config.DNF_DATASET_PATH.exists(),
        results=config.RACE_RESULTS_PATH.exists(),
        profiles=config.CIRCUIT_PROFILE_PATH.exists(),
        model=store.MODEL_PATH.exists(),
        log=monitor.LOG_PATH.exists(),
    )


@st.cache_data(show_spinner="Loading dataset ...")
def load_dataset(version: tuple[float, ...]) -> pd.DataFrame:
    """The modelling table, with dates typed the way the pipeline expects."""
    frame = pd.read_parquet(config.DNF_DATASET_PATH)
    frame[ORDER_COL] = pd.to_datetime(frame[ORDER_COL])
    return frame


@st.cache_data(show_spinner="Loading raw results ...")
def load_raw(version: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Raw results and circuit profiles: what an upcoming race is built from."""
    results = pd.read_parquet(config.RACE_RESULTS_PATH)
    results[ORDER_COL] = pd.to_datetime(results[ORDER_COL])
    profiles = (
        pd.read_parquet(config.CIRCUIT_PROFILE_PATH)
        if config.CIRCUIT_PROFILE_PATH.exists()
        else None
    )
    return results, profiles


@st.cache_data
def load_log(version: tuple[float, ...]) -> pd.DataFrame:
    """The per-race performance log, with numerics coerced.

    ``roc_auc`` and ``pr_auc`` are written as empty strings when a race had no
    retirements and the metric is undefined, so every numeric column goes
    through ``to_numeric`` rather than being trusted from ``read_csv``.
    """
    frame = monitor.read_log()
    if frame.empty:
        return frame
    numeric = [
        "year", "round", "n", "observed_rate", "mean_predicted", "brier",
        "brier_base", "brier_skill", "log_loss", "roc_auc", "pr_auc",
        "top2_hits", "top2_lift", "train_rows", "train_base_rate",
    ]
    for column in numeric:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "race_date" in frame.columns:
        frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    # Older logs predate the provenance column; treat them as live, which is
    # what they were.
    if "source" not in frame.columns:
        frame["source"] = "live"
    frame["source"] = frame["source"].fillna("live")
    return frame.sort_values(["year", "round"]).reset_index(drop=True)


def manifest_or_none() -> store.Manifest | None:
    """The saved manifest, or None when nothing has been fitted yet."""
    try:
        return store.read_manifest()
    except store.ModelStoreError:
        return None


@st.cache_resource(show_spinner="Loading model ...")
def load_model_unchecked(version: tuple[float, ...]):
    """The saved estimator without the staleness guard.

    ``check=False`` is deliberate and is the one place the app diverges from the
    CLI's posture.  ``predict`` refuses a stale model because it is about to
    publish a number someone will act on; the dashboard's job is to *show* that
    the model is stale, which it cannot do if loading raises.  Every page that
    calls this pairs it with :func:`model_staleness` and says so on screen.
    """
    return store.load_model(check=False)


def model_staleness(
    dataset: pd.DataFrame, manifest: store.Manifest | None
) -> tuple[bool, str]:
    """Whether the saved model has seen every race in the dataset.

    Returns ``(ok, message)`` rather than raising, because "the model is two
    races behind" is a thing to display, not an error to crash on.
    """
    if manifest is None:
        return False, "no model has been fitted yet"
    if dataset.empty:
        return True, "no dataset to compare against"
    year, rnd, event = store.latest_race(dataset)
    seen = (manifest.trained_through_year, manifest.trained_through_round)
    if (year, rnd) > seen:
        return False, (
            f"trained through {seen[0]} R{seen[1]} but the dataset reaches "
            f"{year} R{rnd} ({event}) — refit to use the latest races"
        )
    return True, f"current: trained through {seen[0]} R{seen[1]}"


def schema_drift(
    dataset: pd.DataFrame, manifest: store.Manifest | None, stage: str
) -> tuple[bool, str]:
    """Whether the registry has moved since the saved model was fitted."""
    if manifest is None:
        return False, "no model has been fitted yet"
    expected = set(registry.feature_columns(stage, available=dataset.columns))
    saved = set(manifest.features)
    if manifest.stage != stage:
        return False, (
            f"the saved model is a {manifest.stage} model; {stage} needs its "
            f"own fit"
        )
    added, gone = expected - saved, saved - expected
    if added or gone:
        return False, (
            f"registry drift: {len(added)} feature(s) added since the fit, "
            f"{len(gone)} no longer produced — refit before trusting predictions"
        )
    return True, f"{len(saved)} features, matching the registry"


# --------------------------------------------------------------------------- #
# Race calendar helpers
# --------------------------------------------------------------------------- #


@st.cache_data
def race_index(
    version: tuple[float, ...], _dataset: pd.DataFrame
) -> pd.DataFrame:
    """One row per race in the dataset, newest first, for pickers.

    ``_dataset`` is underscore-prefixed so Streamlit does not hash a 3,700-row
    frame on every rerun; ``version`` is what actually keys the cache, and it
    must NOT be underscored or the entry would never invalidate.
    """
    frame = (
        _dataset[["Year", "RoundNumber", "EventName", ORDER_COL]]
        .drop_duplicates(subset=["Year", "RoundNumber"])
        .sort_values(ORDER_COL, ascending=False)
        .reset_index(drop=True)
    )
    frame["label"] = (
        frame["Year"].astype(str)
        + " R"
        + frame["RoundNumber"].astype(int).astype(str).str.zfill(2)
        + "  "
        + frame["EventName"]
    )
    return frame


def season_attrition(dataset: pd.DataFrame) -> pd.DataFrame:
    """Retirement rate by season, with the count behind each rate.

    The headline story of this dataset is that attrition is not stationary, so
    this is the first thing worth plotting anywhere.
    """
    frame = (
        dataset.groupby("Year", observed=True)
        .agg(
            rows=(TARGET, "size"),
            races=("RoundNumber", "nunique"),
            retirements=(TARGET, "sum"),
            dnf_rate=(TARGET, "mean"),
        )
        .reset_index()
    )
    frame["era"] = frame["Year"].map(config.regulation_era)
    return frame


def race_attrition(dataset: pd.DataFrame) -> pd.DataFrame:
    """Retirement rate race by race, for the long trend line."""
    frame = (
        dataset.groupby(["Year", "RoundNumber"], observed=True)
        .agg(
            race_date=(ORDER_COL, "first"),
            event=("EventName", "first"),
            n=(TARGET, "size"),
            retirements=(TARGET, "sum"),
            dnf_rate=(TARGET, "mean"),
        )
        .reset_index()
        .sort_values("race_date")
    )
    return frame


# --------------------------------------------------------------------------- #
# Evaluation, cached
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Replaying races ...", ttl=3600)
def cached_walk_forward_races(
    version: tuple[float, ...],
    *,
    stage: str,
    model: str,
    lookback_races: int | None,
    start_after: str | None,
    refit_every: int,
    drop_features: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """A per-race walk-forward, returned as plain frames so Streamlit can hash it.

    A ``WalkForwardResult`` is a dataclass holding frames; returning the frames
    directly keeps the cache key honest and the return value picklable.
    """
    dataset = load_dataset(version)
    result: WalkForwardResult = walk_forward_races(
        dataset,
        stage=stage,
        model=model,
        lookback_races=lookback_races,
        start_after=start_after,
        refit_every=refit_every,
        drop_features=list(drop_features),
    )
    return result.scores, result.predictions, result.features


@st.cache_data(show_spinner="Evaluating seasons ...", ttl=3600)
def cached_walk_forward_seasons(
    version: tuple[float, ...],
    *,
    stage: str,
    model: str,
    drop_features: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Season-level walk-forward: train on every prior season, score the next."""
    dataset = load_dataset(version)
    result = walk_forward_evaluate(
        dataset, stage=stage, model=model, drop_features=list(drop_features)
    )
    return result.scores, result.predictions, result.features


@st.cache_data(show_spinner="Running ablation ...", ttl=3600)
def cached_ablation(
    version: tuple[float, ...],
    *,
    stage: str,
    model: str,
    metric: str,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
) -> pd.DataFrame:
    """Drop each feature group in turn and report what it cost."""
    dataset = load_dataset(version)
    return ablate_features(
        dataset,
        {name: list(columns) for name, columns in groups},
        metric=metric,
        stage=stage,
        model=model,
    )


@st.cache_data(show_spinner="Permuting features ...", ttl=3600)
def cached_permutation_importance(
    version: tuple[float, ...],
    *,
    stage: str,
    model: str,
    test_season: int,
    n_repeats: int,
) -> pd.DataFrame:
    """Out-of-sample permutation importance on one held-out season."""
    dataset = load_dataset(version)
    return permutation_importance_report(
        dataset,
        stage=stage,
        model=model,
        test_season=test_season,
        n_repeats=n_repeats,
    )


def bootstrap_interval(
    predictions: pd.DataFrame, *, metric: str = "brier_skill", n_boot: int = 500
) -> dict[str, float]:
    """Race-resampled confidence interval for a walk-forward metric.

    Resampling whole races rather than rows is the point: a first-lap pile-up
    correlates several rows within one event.
    """
    base = (
        float(predictions["train_base_rate"].mean())
        if "train_base_rate" in predictions.columns
        else None
    )
    return race_bootstrap(predictions, metric=metric, base_rate=base, n_boot=n_boot)


# --------------------------------------------------------------------------- #
# Feature groups for ablation
# --------------------------------------------------------------------------- #


def feature_groups(dataset: pd.DataFrame, stage: str) -> dict[str, list[str]]:
    """Ablation groups, built from the registry rather than hand-listed.

    Grouping is what makes an ablation answerable.  Dropping
    ``track_speed_index`` alone proves nothing when ``speed_mean_kph`` says
    almost the same thing; dropping the whole circuit block asks a question with
    an answer.
    """
    available = set(dataset.columns)
    selected = set(registry.feature_columns(stage, available=available))

    track = set(track_feature_group(dataset)) & selected
    groups = {
        "driver history": sorted(
            c for c in selected if c.startswith("driver_") and c not in track
        ),
        "team history": sorted(
            c for c in selected
            if (c.startswith("team") or c.startswith("teammate")) and c not in track
        ),
        "pairing": sorted(c for c in selected if c.startswith("pair") or c == "is_new_pairing"),
        "circuit history": sorted(
            c for c in selected if c.startswith("circuit_") and c not in track
        ),
        "circuit geometry": sorted(track),
        "race context": sorted(
            c for c in selected
            if c in {
                "season_round", "is_season_opener", "field_size",
                "days_since_last_race", "regulation_era",
            }
        ),
        "grid": sorted(
            c for c in selected
            if "grid" in c or c == "starts_from_pit_lane"
        ),
    }
    return {name: columns for name, columns in groups.items() if columns}


@st.cache_data
def registry_table(
    version: tuple[float, ...], _dataset: pd.DataFrame
) -> pd.DataFrame:
    """The registry joined to what the dataset actually delivers.

    Coverage is the useful column: a registered feature that is 100% null is a
    build problem the registry alone cannot show.

    ``_dataset`` is unhashed for cost; ``version`` is the real cache key.
    """
    frame = registry.registry_frame()
    present, coverage, nunique = [], [], []
    for name in frame["feature"]:
        if name in _dataset.columns:
            column = _dataset[name]
            present.append(True)
            coverage.append(float(column.notna().mean()))
            nunique.append(int(column.nunique(dropna=True)))
        else:
            present.append(False)
            coverage.append(0.0)
            nunique.append(0)
    frame["in_dataset"] = present
    frame["coverage"] = coverage
    frame["distinct"] = nunique
    return frame


# --------------------------------------------------------------------------- #
# Fitting and saving
# --------------------------------------------------------------------------- #


def refit_and_save(
    dataset: pd.DataFrame,
    *,
    stage: str,
    model: str,
    lookback_races: int | None,
) -> store.Manifest:
    """Fit on the current window and write the serving artefacts.

    Deliberately *not* cached and deliberately not wrapped in a spinner
    decorator: this is the one action in the app that changes state on disk, and
    it should happen exactly when a button is pressed and not one time more.
    """
    from src.models.predict import fit_current

    estimator, manifest, _ = fit_current(
        dataset, model=model, stage=stage, lookback_races=lookback_races
    )
    store.save_model(estimator, manifest)
    return manifest


def drift(dataset: pd.DataFrame, manifest: store.Manifest | None, window: int = 5):
    """Recent observed attrition against what the model was trained on."""
    if manifest is None:
        return True, "no model to check drift against"
    return monitor.drift_check(dataset, manifest.train_base_rate, window=window)


def rolling_summary(window: int) -> pd.Series:
    return monitor.rolling_summary(window)


def stage_features(dataset: pd.DataFrame, stage: str) -> list[str]:
    return registry.feature_columns(stage, available=dataset.columns)


def describe_feature(name: str) -> str:
    feature = registry.BY_NAME.get(name)
    return feature.description if feature else ""


__all__ = [
    "Artefacts", "HIGHER_IS_BETTER", "MODELS", "STAGES", "TARGET",
    "ORDER_COL", "DEFAULT_LOOKBACK_RACES",
    "artefacts", "bootstrap_interval", "cached_ablation",
    "cached_permutation_importance", "cached_walk_forward_races",
    "cached_walk_forward_seasons", "data_version", "describe_feature", "drift",
    "feature_groups", "load_dataset", "load_log", "load_model_unchecked",
    "load_raw", "manifest_or_none", "model_staleness", "race_attrition",
    "race_index", "refit_and_save", "registry_table", "rolling_summary",
    "schema_drift", "season_attrition", "stage_features",
]
