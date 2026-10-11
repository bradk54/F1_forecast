"""How the forecast has scored: the finishing order first, then the attrition stage.

The finishing-order section reads the two reports the CLI writes -- held-out
scores from ``forecast evaluate --write`` and season calibration from
``forecast backtest --write`` -- rather than recomputing them, because both
are minutes of simulation and belong in a terminal.  Below them is the
attrition model's per-race log.

On the attrition log:

The unit is a race, because that is the cadence the model runs at and it is how
a problem actually announces itself — three bad weekends in a row, not a bad
year in hindsight.

Two things this page insists on:

* **Pool, don't average.**  A race with no retirements has an undefined AUC and
  a large negative Brier skill from a model that did nothing unusual.  Averaging
  per-race skill lets the quietest Sundays set the headline; pooling reconstructs
  the total squared error and takes the ratio once.
* **Skill is measured against the training base rate, not the race's own.**  A
  model cannot know on Saturday that this particular Sunday would be a 25%
  attrition race, so scoring it against 25% would credit it for something it
  never predicted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import charts, services  # noqa: E402
from app.common import (  # noqa: E402
    fmt_pct, metric_row, page_setup, pooled_scores, require_dataset,
    sidebar_models, source_note,
)
from src.models import forecast  # noqa: E402

page_setup("Track record", icon="📈")

version = services.data_version()
if not require_dataset():
    st.stop()

dataset = services.load_dataset(version)
manifest = services.manifest_or_none()
log = services.load_log(version)
sidebar_models(dataset, manifest)

st.title("Track record")
st.caption(
    "Every score here is out-of-sample: each race was forecast by a model "
    "fitted only on the races before it. Choices were made on 2020-2023 and "
    "scored on 2024 onward."
)

# --------------------------------------------------------------------------- #
# The finishing order
# --------------------------------------------------------------------------- #

st.subheader("Finishing order")

#: The rows of the evaluation ladder worth showing, cheapest first: the naive
#: baseline the model has to beat, then the shipped model.
_BASELINE = {"pre_weekend": "team order", "post_quali": "grid order"}
_SHIPPED = "selected + tuned"

evaluation = services.order_evaluation()
if evaluation is None:
    st.info(
        "No held-out scores on disk yet. Write them once with "
        "`./.venv/bin/python -m src.models.forecast evaluate --write` "
        "(several minutes) and this section fills in."
    )
else:
    stage_labels = {"pre_weekend": "Before qualifying", "post_quali": "After qualifying"}
    for column, stage in zip(st.columns(2), ("pre_weekend", "post_quali")):
        block = evaluation.loc[evaluation["stage"] == stage].set_index("model")
        with column:
            st.markdown(f"**{stage_labels[stage]}**")
            if _SHIPPED not in block.index:
                st.caption("Not in the report on disk.")
                continue
            ours = block.loc[_SHIPPED]
            base = block.loc[_BASELINE[stage]] if _BASELINE[stage] in block.index else None
            metric_row([
                ("RPS@10", f"{ours['rps10']:.4f}",
                 f"{_BASELINE[stage]}: {base['rps10']:.4f}" if base is not None
                 else "lower is better"),
                ("Races", f"{int(ours['races'])}", "held out"),
            ])
            metric_row([
                ("Calibration: win", f"{ours['calib_win']:.2f}", "1.00 is perfect"),
                ("podium", f"{ours['calib_podium']:.2f}", ""),
                ("points", f"{ours['calib_points']:.2f}", ""),
            ])
    with st.expander("The full evaluation ladder"):
        columns = ["stage", "model", "races", "ll10_gain", "rps10", "points_rmse",
                   "spearman", "brier_win", "calib_win", "calib_podium", "calib_points"]
        st.dataframe(evaluation[[c for c in columns if c in evaluation.columns]],
                     hide_index=True, width="stretch")
        st.caption(
            "Each rung has to beat the one above it to justify itself. RPS@10 "
            "is the ranked probability score over the points positions; a "
            "calibration slope above 1 means the model was underconfident, "
            "below 1 overconfident."
        )

st.markdown("**Season simulation**")
backtest = services.season_calibration()
shipped = forecast.DEFAULT_SEASON_NOISE.team_sd
row = None
if backtest is not None:
    calibration, seasons = backtest
    row = calibration.loc[
        (calibration["team_sd"].sub(shipped).abs() < 1e-9)
        & (calibration["driver_sd"] == 0)
    ].set_index("kind")
if row is None or row.empty:
    st.info(
        "No season backtest at the shipped setting on disk. Build it with "
        "`./.venv/bin/python -m src.models.forecast backtest --write`."
    )
else:
    metric_row([
        ("Driver 80% band held", fmt_pct(row.loc["driver", "coverage80"]),
         f"should be 80% · {int(row.loc['driver', 'entities'])} driver-seasons"),
        ("Constructor 80% band held", fmt_pct(row.loc["team", "coverage80"]),
         f"should be 80% · {int(row.loc['team', 'entities'])} team-seasons"),
        ("Seasons", ", ".join(map(str, seasons)), f"team_sd {shipped}"),
    ])
    st.caption(
        "Both short of 80%: the championship bands are too narrow and the "
        "extreme odds too confident. The open fix is per-driver pace noise, "
        "so team-mates can move apart."
    )

# --------------------------------------------------------------------------- #
# The attrition stage
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Attrition stage")
st.caption(
    "The retirement model's per-race log, one row per scored race. It feeds "
    "the forecast's P(out), and it is the one stage with a live weekly loop."
)

if log.empty:
    st.info(
        "The log is empty. It fills one race at a time as `predict refresh` runs "
        "after each weekend — or backfill it now from a walk-forward replay:"
    )
    st.code("./.venv/bin/python -m scripts.backfill_model_log", language="bash")
    st.caption(
        "A replay refits before every race on strictly prior data, so the rows "
        "it writes are honest out-of-sample scores. They are marked "
        "`source=replay` and never overwrite a live row."
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #

# Two rows, so the selects keep their values legible in a narrow pane.
stages = sorted(log["stage"].dropna().unique())
top = st.columns(3)
with top[0]:
    stage = st.selectbox(
        "Stage", stages,
        index=stages.index(manifest.stage) if manifest and manifest.stage in stages
        else len(stages) - 1,
    )
with top[1]:
    sources = ["all", *sorted(log["source"].dropna().unique())]
    source = st.selectbox("Source", sources, index=0)
with top[2]:
    window = st.number_input("Rolling window", 3, 40, 10, step=1)
years = sorted(log["year"].dropna().unique().astype(int))
span = st.select_slider(
    "Seasons", options=years,
    value=(years[0], years[-1]) if len(years) > 1 else (years[0], years[0]),
)

block = log.loc[log["stage"] == stage]
if source != "all":
    block = block.loc[block["source"] == source]
block = block.loc[block["year"].between(span[0], span[1])].sort_values(
    ["year", "round"]
).reset_index(drop=True)

if block.empty:
    st.warning("Nothing logged for that combination.")
    st.stop()

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #

pooled = pooled_scores(block)
metric_row(
    [
        ("Races", f"{len(block):,}", f"{int(block['n'].sum()):,} driver-races"),
        ("Brier skill (pooled)", f"{pooled['brier_skill']:+.4f}",
         "positive beats the base rate"),
        ("Mean ROC AUC", f"{pooled['roc_auc']:.3f}",
         f"defined in {pooled['auc_races']} of {len(block)} races"),
        ("Top-2 hit rate", f"{pooled['top2_rate']:.1%}",
         f"{int(block['top2_hits'].sum())} caught in {2 * len(block)} picks"),
    ]
)
source_note(block)

mean_pred, observed = pooled["mean_predicted"], pooled["observed_rate"]
gap = mean_pred - observed
if abs(gap) > 0.02:
    st.warning(
        f"**Systematically {'over' if gap > 0 else 'under'}-predicting** — "
        f"mean prediction {mean_pred:.1%} against {observed:.1%} observed "
        f"({gap:+.1%}). That is a base-rate problem, not a ranking problem: "
        f"check the training window before touching features."
    )

st.divider()

# --------------------------------------------------------------------------- #
# Skill over time
# --------------------------------------------------------------------------- #

st.subheader("Skill over time")
trend = block.copy()
trend["rolling"] = trend["brier_skill"].rolling(window, min_periods=3).mean()
fig = charts.series_lines(
    trend, x="race_date",
    series={"brier_skill": "per race", "rolling": f"{window}-race mean"},
    ytitle="Brier skill", height=320,
    reference=0.0, reference_label="no better than the base rate",
)
fig.data[0].update(mode="markers", marker=dict(size=6, line=dict(width=0)), opacity=0.4)
charts.show(fig)
st.caption(
    "A single race is twenty rows and mostly noise. Read the line; the dots are "
    "there to show how wide the noise around it is."
)

# --------------------------------------------------------------------------- #
# Predicted against observed
# --------------------------------------------------------------------------- #

st.subheader("Predicted against observed attrition")
st.caption(
    "One axis, two rates. When the lines separate persistently the model is "
    "anchored to a base rate the sport has moved away from — which is exactly "
    "what the drift check watches for."
)
rates = block.copy()
rates["predicted_roll"] = rates["mean_predicted"].rolling(window, min_periods=3).mean()
rates["observed_roll"] = rates["observed_rate"].rolling(window, min_periods=3).mean()
fig = charts.series_lines(
    rates, x="race_date",
    series={"predicted_roll": f"predicted ({window}-race mean)",
            "observed_roll": f"observed ({window}-race mean)"},
    ytitle="retirement rate", height=320,
)
fig.update_yaxes(tickformat=".0%")
charts.show(fig)

drift_ok, drift_message = services.drift(dataset, manifest)
(st.success if drift_ok else st.error)(drift_message)

st.divider()

# --------------------------------------------------------------------------- #
# Reliability
# --------------------------------------------------------------------------- #

left, right = st.columns(2)

with left:
    st.subheader("Race-level reliability")
    st.caption(
        "Each dot is a race: what the model predicted on average against what "
        "happened. The dashed line is perfect. Row-level calibration lives on "
        "the **Lab** page — the log only carries per-race aggregates."
    )
    points = block.copy()
    points["label"] = (
        points["year"].astype(int).astype(str) + " R"
        + points["round"].astype(int).astype(str) + " " + points["event"]
    )
    fig = charts.scatter(
        points, x="mean_predicted", y="observed_rate", text="label",
        hover=("n",), xtitle="mean predicted", ytitle="observed rate",
        height=360, diagonal=True,
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    charts.show(fig)

with right:
    st.subheader("Top-2 picks")
    st.caption(
        "The number a person actually acts on: of the two cars flagged each "
        "weekend, how many stopped. Cumulative, against what chance would give."
    )
    picks = block.copy()
    picks["caught"] = picks["top2_hits"].cumsum()
    # Chance is 2 x the race's own retirement rate, accumulated -- what you
    # would expect from picking two cars at random that weekend.
    picks["by_chance"] = (2 * picks["observed_rate"]).cumsum()
    fig = charts.series_lines(
        picks, x="race_date",
        series={"caught": "caught by the model", "by_chance": "expected by chance"},
        ytitle="cumulative retirements caught", height=360,
    )
    charts.show(fig)

st.divider()

# --------------------------------------------------------------------------- #
# What the grid buys
# --------------------------------------------------------------------------- #

if len(stages) > 1:
    st.subheader("What knowing the grid buys")
    st.caption(
        "`pre_weekend` against `post_quali` on the same races. Grid position is "
        "the model's strongest feature, so the gap here is the value of waiting "
        "until Saturday evening."
    )
    both = log.loc[log["year"].between(span[0], span[1])]
    if source != "all":
        both = both.loc[both["source"] == source]

    comparison = []
    for name, group in both.groupby("stage"):
        scores = pooled_scores(group)
        comparison.append({
            "stage": name, "races": len(group),
            "brier_skill": scores["brier_skill"],
            "roc_auc": scores["roc_auc"],
            "top2_rate": scores["top2_rate"],
        })
    table = pd.DataFrame(comparison)

    grid_col, chart_col = st.columns([2, 3])
    with grid_col:
        display = table.copy()
        display["brier_skill"] = display["brier_skill"].map("{:+.4f}".format)
        display["roc_auc"] = display["roc_auc"].map("{:.3f}".format)
        display["top2_rate"] = display["top2_rate"].map("{:.1%}".format)
        st.dataframe(display, hide_index=True, width="stretch")
        if len(table) == 2:
            delta = (
                table.set_index("stage")["roc_auc"].get("post_quali", np.nan)
                - table.set_index("stage")["roc_auc"].get("pre_weekend", np.nan)
            )
            if pd.notna(delta):
                st.metric("AUC gained from the grid", f"{delta:+.3f}")

    with chart_col:
        wide = (
            both.pivot_table(
                index="race_date", columns="stage", values="brier_skill",
                aggfunc="mean",
            )
            .sort_index()
            .rolling(window, min_periods=3)
            .mean()
            .reset_index()
        )
        available = [c for c in stages if c in wide.columns]
        fig = charts.series_lines(
            wide, x="race_date",
            series={c: c for c in available},
            ytitle=f"Brier skill ({window}-race mean)", height=320,
            reference=0.0,
        )
        charts.show(fig)

st.divider()

# --------------------------------------------------------------------------- #
# The rows
# --------------------------------------------------------------------------- #

with st.expander(f"The log — {len(block)} rows", expanded=False):
    columns = [
        "source", "year", "round", "event", "n", "observed_rate",
        "mean_predicted", "brier_skill", "roc_auc", "top2_hits", "top2_lift",
        "train_rows", "train_base_rate", "git_sha",
    ]
    st.dataframe(
        block[[c for c in columns if c in block.columns]].iloc[::-1],
        hide_index=True, width="stretch", height=420,
    )
    st.download_button(
        "Download this view as CSV",
        block.to_csv(index=False).encode(),
        file_name=f"model_log_{stage}.csv", mime="text/csv",
    )

worst = block.nsmallest(5, "brier_skill")[
    ["year", "round", "event", "n", "observed_rate", "mean_predicted", "brier_skill"]
]
with st.expander("Worst five races", expanded=False):
    st.caption(
        "Usually races with no retirements at all, where any positive "
        "prediction is punished. Look for the ones where attrition was high and "
        "the model still predicted low — those are the real misses."
    )
    st.dataframe(worst, hide_index=True, width="stretch")
