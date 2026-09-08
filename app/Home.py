"""Overview: is the model current, is it calibrated, and has the sport moved?

The three questions that decide whether anything else on this dashboard is worth
reading, answered above the fold.  Everything else is a page.

Run it from the repository root::

    ./.venv/bin/python -m streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit executes this file as a script rather than importing it as part of
# the package, so the repository root is not on the path by default.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import charts, services  # noqa: E402
from app.common import (  # noqa: E402
    metric_row, page_setup, pooled_scores, require_dataset, sidebar_status,
)

page_setup("F1 retirement model", icon="🏁")

version = services.data_version()
if not require_dataset():
    st.stop()

dataset = services.load_dataset(version)
manifest = services.manifest_or_none()
log = services.load_log(version)

sidebar_status(dataset, manifest)

st.title("Retirement model")
st.caption(
    "Will a driver retire from this race? One row per driver per race, 2018 "
    "onward. Every number here comes from the same functions the CLI calls."
)

# --------------------------------------------------------------------------- #
# Is the model current?
# --------------------------------------------------------------------------- #

year, rnd, event = (
    dataset.sort_values(services.ORDER_COL).iloc[-1][
        ["Year", "RoundNumber", "EventName"]
    ]
)
fresh, freshness = services.model_staleness(dataset, manifest)
drift_ok, drift_message = services.drift(dataset, manifest)

metric_row(
    [
        ("Races in dataset", f"{dataset[['Year', 'RoundNumber']].drop_duplicates().shape[0]:,}",
         f"{len(dataset):,} driver-races"),
        ("Latest race", f"{int(year)} R{int(rnd)}", str(event)),
        ("Model", manifest.model if manifest else "—",
         f"{manifest.stage}, lookback {manifest.lookback_races}" if manifest else "not fitted"),
        ("Trained base rate", f"{manifest.train_base_rate:.1%}" if manifest else "—",
         f"{manifest.train_rows} rows" if manifest else ""),
    ]
)

col_a, col_b = st.columns(2)
with col_a:
    (st.success if fresh else st.warning)(f"**Freshness** — {freshness}")
with col_b:
    (st.success if drift_ok else st.error)(f"**Drift** — {drift_message}")

if not fresh and manifest is not None:
    st.caption(
        "A stale model is not wrong, only behind. Refit from the sidebar, or run "
        "`./.venv/bin/python -m src.models.predict refresh`."
    )

st.divider()

# --------------------------------------------------------------------------- #
# Has the sport moved?
# --------------------------------------------------------------------------- #

st.subheader("Attrition is not stationary")
st.caption(
    "The premise the whole training-window argument rests on. A model fitted on "
    "a settled formula predicts against a base rate the sport can walk away "
    "from inside one winter."
)

seasons = services.season_attrition(dataset)
left, right = st.columns([3, 2])

with left:
    fig = charts.bars(
        seasons, x="Year", y="dnf_rate",
        ytitle="retirement rate", xtitle="season", height=300,
        hover=("races", "rows", "retirements", "era"),
        reference=float(manifest.train_base_rate) if manifest else None,
        reference_label="training base rate",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(type="category")
    charts.show(fig)

with right:
    display = seasons.copy()
    display["dnf_rate"] = display["dnf_rate"].map("{:.1%}".format)
    st.dataframe(
        display[["Year", "races", "rows", "retirements", "dnf_rate", "era"]],
        hide_index=True, width="stretch", height=300,
    )

# --------------------------------------------------------------------------- #
# Is it calibrated?
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("How it has been scoring")

if log.empty:
    st.info(
        "No scored races yet. Backfill the log with "
        "`./.venv/bin/python -m scripts.backfill_model_log`, or let it fill one "
        "race at a time as `predict refresh` runs after each weekend."
    )
else:
    stage_options = sorted(log["stage"].dropna().unique())
    default = manifest.stage if manifest and manifest.stage in stage_options else stage_options[-1]
    stage = st.radio(
        "Stage", stage_options, index=stage_options.index(default),
        horizontal=True, key="home_stage",
        help="pre_weekend excludes grid position; post_quali includes it. They "
             "are different models, not the same model with more confidence.",
    )
    block = log.loc[log["stage"] == stage].sort_values(["year", "round"])
    pooled = pooled_scores(block)

    metric_row(
        [
            ("Races scored", f"{len(block):,}",
             f"{int(block['n'].sum()):,} driver-races"),
            ("Brier skill (pooled)", f"{pooled['brier_skill']:+.4f}",
             "vs the training base rate"),
            ("Mean ROC AUC", f"{pooled['roc_auc']:.3f}",
             f"{int(pooled['auc_races'])} races where it is defined"),
            ("Top-2 hit rate", f"{pooled['top2_rate']:.1%}",
             f"{int(block['top2_hits'].sum())} caught in {len(block) * 2} picks"),
        ]
    )
    st.caption(
        "Pooled, not averaged per race: a race with no retirements has an "
        "undefined AUC and a wildly negative Brier skill, so a mean over races "
        "is dominated by the quietest Sundays."
    )

    trend = block.copy()
    trend["rolling"] = trend["brier_skill"].rolling(10, min_periods=3).mean()
    fig = charts.series_lines(
        trend, x="race_date",
        series={"brier_skill": "per race", "rolling": "10-race mean"},
        xtitle="", ytitle="Brier skill", height=300,
        reference=0.0, reference_label="no better than the base rate",
    )
    fig.data[0].update(
        mode="markers", marker=dict(size=6, line=dict(width=0)), opacity=0.45
    )
    charts.show(fig)
    st.caption(
        "A single race's Brier skill is twenty rows of noise. The ten-race mean "
        "is the line to read; the dots are there to show how wide the noise is."
    )

st.divider()
st.caption(
    "Pages: **Race weekend** ranks a grid · **Model health** reads the log · "
    "**Lab** runs walk-forward, ablation and importance · **Data explorer** "
    "browses the dataset and the feature registry."
)
