"""Page furniture: setup, the status sidebar, and the few shared computations.

Kept apart from :mod:`app.services` because none of it touches the pipeline —
this is presentation, and mixing the two is how a dashboard ends up with its own
quietly divergent copy of a metric.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import services  # noqa: E402
from src import config  # noqa: E402


def page_setup(title: str, *, icon: str = "🏁") -> None:
    st.set_page_config(
        page_title=title, page_icon=icon, layout="wide",
        initial_sidebar_state="expanded",
    )
    # Streamlit's default metric label is small and grey to the point of being
    # decorative; these are the numbers the page exists to show.
    st.markdown(
        """
        <style>
          [data-testid="stMetricLabel"] p { font-size: 0.82rem; }
          [data-testid="stMetricValue"] { font-size: 1.55rem; }
          [data-testid="stMetricDelta"] { font-size: 0.78rem; }
          div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px; }
          section[data-testid="stSidebar"] { min-width: 260px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def require_dataset() -> bool:
    """Show the build instructions instead of a stack trace when there is no data."""
    if config.DNF_DATASET_PATH.exists():
        return True
    st.title("No dataset yet")
    st.error(f"Nothing at `{config.DNF_DATASET_PATH}`.")
    st.code(
        "./.venv/bin/python -m src.data.generate_dataset --seasons 2018-2026",
        language="bash",
    )
    st.caption(
        "`--offline` rebuilds from the FastF1 cache without touching the "
        "network, which is faster and cannot provoke a rate limit."
    )
    return False


def metric_row(items: Sequence[tuple[str, str, str]]) -> None:
    """A row of metrics as ``(label, value, caption)``."""
    columns = st.columns(len(items))
    for column, (label, value, caption) in zip(columns, items):
        with column:
            st.metric(label, value)
            if caption:
                st.caption(caption)


def sidebar_status(dataset: pd.DataFrame, manifest) -> None:
    """The saved model, and the one control that writes to disk.

    Refitting lives in the sidebar on every page because it is the answer to
    most of the warnings the pages raise, and because burying it would make the
    warnings advice rather than an action.
    """
    with st.sidebar:
        st.subheader("Saved model")
        if manifest is None:
            st.warning("Nothing fitted yet.")
        else:
            st.caption(
                f"**{manifest.model}** · `{manifest.stage}`\n\n"
                f"through {manifest.trained_through_year} "
                f"R{manifest.trained_through_round} "
                f"({manifest.trained_through_event})\n\n"
                f"{manifest.train_rows} rows · base rate "
                f"{manifest.train_base_rate:.1%} · "
                f"{len(manifest.features)} features\n\n"
                f"lookback `{manifest.lookback_races}` · git `{manifest.git_sha}`\n\n"
                f"fitted {manifest.trained_at}"
            )

        st.divider()
        with st.expander("Refit and save", expanded=manifest is None):
            st.caption(
                "Writes `Models/dnf_model.joblib` and its manifest. It does "
                "**not** fetch new races — rebuild the dataset from a terminal "
                "for that."
            )
            stage = st.selectbox(
                "Stage", services.STAGES,
                index=services.STAGES.index(manifest.stage)
                if manifest and manifest.stage in services.STAGES else 1,
                key="refit_stage",
            )
            model = st.selectbox(
                "Model", services.MODELS,
                index=services.MODELS.index(manifest.model)
                if manifest and manifest.model in services.MODELS
                else services.MODELS.index("random_forest"),
                key="refit_model",
            )
            lookback = st.number_input(
                "Lookback races", min_value=0, max_value=400,
                value=int(manifest.lookback_races)
                if manifest and manifest.lookback_races else
                services.DEFAULT_LOOKBACK_RACES,
                step=5, key="refit_lookback",
                help="0 trains on every prior race. 40 is the default and is a "
                     "bet on recency during a regime change, not a free win.",
            )
            if st.button("Refit on the current window", type="primary",
                         width="stretch"):
                try:
                    fitted = services.refit_and_save(
                        dataset, stage=stage, model=model,
                        lookback_races=None if lookback == 0 else int(lookback),
                    )
                except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                    st.error(f"Refit failed: {exc}")
                else:
                    st.success(f"Saved. {fitted.describe()}")
                    st.cache_resource.clear()
                    st.rerun()

        st.divider()
        st.caption(
            f"`{config.DNF_DATASET_PATH.parent}`\n\n"
            "Rebuild the dataset:\n\n"
            "`python -m src.data.generate_dataset --offline`"
        )


def pooled_scores(log_block: pd.DataFrame) -> dict[str, float]:
    """Headline numbers over a set of logged races, pooled rather than averaged.

    Averaging per-race Brier skill is the wrong summary and not by a little: a
    race with no retirements produces a large negative skill from a model that
    did nothing unusual, and there are enough of those to drag the mean below
    zero while the pooled figure is positive.  Pooling reconstructs the total
    squared error and the total base-rate error, then takes the ratio once.
    """
    if log_block.empty:
        return {
            "brier_skill": np.nan, "roc_auc": np.nan, "top2_rate": np.nan,
            "auc_races": 0, "observed_rate": np.nan, "mean_predicted": np.nan,
        }
    weights = log_block["n"].fillna(0)
    total = weights.sum()
    brier = float((log_block["brier"] * weights).sum())
    brier_base = float((log_block["brier_base"] * weights).sum())
    auc = pd.to_numeric(log_block["roc_auc"], errors="coerce").dropna()
    return {
        "brier_skill": 1 - brier / brier_base if brier_base > 0 else np.nan,
        "roc_auc": float(auc.mean()) if not auc.empty else np.nan,
        "auc_races": int(len(auc)),
        "top2_rate": float(log_block["top2_hits"].sum() / (2 * len(log_block))),
        "observed_rate": float((log_block["observed_rate"] * weights).sum() / total)
        if total else np.nan,
        "mean_predicted": float((log_block["mean_predicted"] * weights).sum() / total)
        if total else np.nan,
    }


def calibration_bins(
    outcomes: pd.Series, predicted: pd.Series, bins: int = 10
) -> pd.DataFrame:
    """Reliability table: mean predicted against observed, by predicted decile.

    Quantile bins rather than equal-width ones, because predictions cluster near
    the base rate and equal-width bins would put almost everything in one bucket
    and draw a confident line through three points.
    """
    frame = pd.DataFrame(
        {"y": pd.to_numeric(outcomes, errors="coerce"),
         "p": pd.to_numeric(predicted, errors="coerce")}
    ).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["bin", "predicted", "observed", "n"])
    try:
        frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    except ValueError:
        frame["bin"] = pd.cut(frame["p"], bins=bins)
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def source_note(log_block: pd.DataFrame) -> None:
    """Say plainly how much of what is on screen was replayed rather than lived."""
    if log_block.empty or "source" not in log_block.columns:
        return
    counts = log_block["source"].value_counts()
    live, replay = int(counts.get("live", 0)), int(counts.get("replay", 0))
    if replay and not live:
        st.caption(
            f"All {replay} races here are **replayed** — reconstructed by "
            "walk-forward over the current dataset. Honest out-of-sample "
            "scores, but not evidence that the weekly loop ran."
        )
    elif replay and live:
        st.caption(
            f"{live} race(s) scored **live** (predicted before, scored after) "
            f"and {replay} **replayed**. Only the live ones prove the weekly "
            "loop works."
        )


def fmt_pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def fmt_num(value: float, places: int = 3) -> str:
    return "—" if pd.isna(value) else f"{value:.{places}f}"
