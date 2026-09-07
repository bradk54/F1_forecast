"""Walk-forward, ablation and importance — the experiments, run on demand.

Nothing on this page reads a saved artefact.  Every number is computed from the
dataset by the same functions the CLI and the tests use, so a configuration you
try here is a configuration you can ship.

**Two walk-forwards, and they answer different questions.**  Refitting *before
every race* is how the model is actually run: retrained the moment the last race
is classified.  Refitting *once a season* answers "how good would this have been
in 2025" and gives the first race of a season the same stale model as the last,
which flatters early rounds and penalises late ones.  The per-race view is the
honest one for a weekly cadence; the season view is the coarser, more familiar
one.

**Runs are kept so they can be compared.**  A single walk-forward number means
little on its own — the interesting quantity is almost always a difference
between two configurations, and a difference is only readable if both were
computed the same way on the same data.
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
    calibration_bins, metric_row, page_setup, require_dataset, sidebar_status,
)

page_setup("Lab", icon="🔬")

version = services.data_version()
if not require_dataset():
    st.stop()

dataset = services.load_dataset(version)
manifest = services.manifest_or_none()
sidebar_status(dataset, manifest)

st.title("Lab")
st.caption(
    "Walk-forward evaluation, feature ablation and permutation importance, "
    "computed live from the dataset. Results are cached for an hour per "
    "configuration."
)

if "lab_runs" not in st.session_state:
    st.session_state["lab_runs"] = {}

tab_wf, tab_ablate, tab_importance, tab_compare = st.tabs(
    ["Walk-forward", "Ablation", "Importance", "Saved runs"]
)

# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #

with tab_wf:
    # Two rows rather than five columns: five selects in one row truncate their
    # values to a single character in a narrow pane.
    top = st.columns(3)
    with top[0]:
        cadence = st.selectbox(
            "Cadence", ["Refit before every race", "Refit once a season"],
            help="Per-race is how the model is actually run. Per-season is the "
                 "coarser, more familiar view.",
        )
    with top[1]:
        stage = st.selectbox("Stage", services.STAGES, index=1, key="wf_stage")
    with top[2]:
        model = st.selectbox(
            "Model", services.MODELS,
            index=services.MODELS.index("random_forest"), key="wf_model",
        )
    per_race = cadence.startswith("Refit before")
    bottom = st.columns([1, 2])
    with bottom[0]:
        lookback = st.number_input(
            "Lookback races", 0, 400, services.DEFAULT_LOOKBACK_RACES, step=5,
            key="wf_lookback", disabled=not per_race,
            help="0 is an expanding window over every prior race.",
        )
    with bottom[1]:
        seasons = sorted(dataset["Year"].dropna().unique().astype(int))
        # Defaults to the last three seasons: a per-race walk-forward refits
        # once per race, so widening this costs roughly a quarter-second a race
        # and the whole calendar is a forty-second first click.
        start_year = st.select_slider(
            "Score races from", options=seasons, value=seasons[max(0, len(seasons) - 3)],
            disabled=not per_race,
            help="Earlier races still count toward the lookback window; this "
                 "only chooses what gets scored. Roughly a quarter-second per "
                 "race scored.",
        )

    if st.button("Run walk-forward", type="primary", key="wf_run"):
        st.session_state["wf_go"] = True

    if st.session_state.get("wf_go"):
        if per_race:
            scores, predictions, features = services.cached_walk_forward_races(
                version, stage=stage, model=model,
                lookback_races=None if lookback == 0 else int(lookback),
                start_after=f"{start_year - 1}-12-31", refit_every=1,
            )
            label = (
                f"{model}/{stage} · per-race · lookback "
                f"{'expanding' if lookback == 0 else lookback} · from {start_year}"
            )
        else:
            scores, predictions, features = services.cached_walk_forward_seasons(
                version, stage=stage, model=model,
            )
            label = f"{model}/{stage} · per-season"

        if predictions.empty:
            st.error(
                "No races were scored. Every fold fell below `min_train_rows`, "
                "or the start year excluded the whole calendar."
            )
            st.stop()

        # Pool once over every scored row, rather than averaging season scores.
        from src.models.train import score_predictions

        pooled = score_predictions(
            predictions[services.TARGET], predictions["predicted"],
            float(predictions["train_base_rate"].mean()),
        )
        races = predictions[["Year", "RoundNumber"]].drop_duplicates().shape[0]

        st.caption(f"**{label}** — {len(features)} features, {races} races, "
                   f"{len(predictions):,} driver-races")

        metric_row(
            [
                ("Brier skill", f"{pooled['brier_skill']:+.4f}",
                 "vs always predicting the training base rate"),
                ("ROC AUC", f"{pooled['roc_auc']:.3f}", "ranking quality"),
                ("PR AUC", f"{pooled['pr_auc']:.3f}",
                 f"base rate {pooled['observed_rate']:.1%}"),
                ("Calibration slope", f"{pooled['calibration_slope']:.3f}",
                 "1.0 means take the probabilities at face value"),
            ]
        )

        if pooled["calibration_slope"] < 0.8:
            st.warning(
                f"Calibration slope {pooled['calibration_slope']:.2f} — the model "
                "is overconfident and its extremes should be shrunk toward the "
                "base rate before they are read as probabilities."
            )

        with st.spinner("Bootstrapping over races ..."):
            interval = services.bootstrap_interval(predictions, n_boot=400)
        low = interval.get("ci_low", np.nan)
        high = interval.get("ci_high", np.nan)
        st.caption(
            "The bootstrap resamples **whole races**, not rows — a first-lap "
            "pile-up retires several cars at once, so rows within an event are "
            "correlated and row-resampling would understate the interval."
        )
        if np.isnan(low):
            st.info("Not enough two-class resamples to form an interval.")
        else:
            st.metric(
                "Brier skill, 95% interval over races",
                f"{interval['mean']:+.4f}",
                f"[{low:+.4f}, {high:+.4f}] from {interval['n_boot']} resamples",
                delta_color="off",
            )
            if low > 0:
                st.success("The improvement over the base rate clears zero.")
            elif high < 0:
                st.error("The interval sits entirely below zero: worse than the base rate.")
            else:
                st.info(
                    "The interval crosses zero — on this data the model is not "
                    "distinguishable from always predicting the base rate."
                )

        st.session_state["lab_runs"][label] = {
            "brier_skill": pooled["brier_skill"], "roc_auc": pooled["roc_auc"],
            "pr_auc": pooled["pr_auc"],
            "calibration_slope": pooled["calibration_slope"],
            "n": pooled["n"], "races": races, "features": len(features),
        }

        st.divider()
        left, right = st.columns(2)

        with left:
            st.subheader("Calibration")
            st.caption(
                "Predicted decile against what happened in it. Quantile bins, "
                "not equal-width: predictions cluster near the base rate, and "
                "equal-width bins would put almost everything in one bucket."
            )
            bins = calibration_bins(
                predictions[services.TARGET], predictions["predicted"], bins=10
            )
            fig = charts.scatter(
                bins, x="predicted", y="observed", hover=("n",),
                xtitle="mean predicted", ytitle="observed rate",
                height=340, diagonal=True, size="n",
            )
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            charts.show(fig)

        with right:
            st.subheader("By season")
            display = scores.copy()
            for column in ("brier_skill", "roc_auc", "pr_auc", "calibration_slope"):
                if column in display.columns:
                    display[column] = display[column].round(4)
            st.dataframe(
                display, hide_index=True, width="stretch", height=340,
            )

        if "season" in scores.columns and len(scores) > 1:
            fig = charts.series_lines(
                scores, x="season",
                series={"brier_skill": "Brier skill", "roc_auc": "ROC AUC"},
                xtitle="season", height=300, markers=True,
            )
            charts.show(fig)
            st.caption(
                "Two unitless quantities on one axis, which is the only reason "
                "they may share a chart. Never add a second y-scale."
            )

        st.divider()
        st.subheader("Riskiest calls the model made")
        st.caption("Its highest-probability predictions, and whether they landed.")
        top = predictions.nlargest(25, "predicted").copy()
        top["result"] = np.where(top[services.TARGET] == 1, "retired", "finished")
        columns = [c for c in ["Year", "RoundNumber", "EventName", "DriverId",
                               "TeamId", "predicted", "result"] if c in top.columns]
        st.dataframe(top[columns], hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #

with tab_ablate:
    st.caption(
        "Drop a whole group of features and measure what it cost. Grouping is "
        "what makes the question answerable: dropping `track_speed_index` alone "
        "proves nothing when `speed_mean_kph` says nearly the same thing."
    )
    controls = st.columns([2, 2, 2])
    with controls[0]:
        ab_stage = st.selectbox("Stage", services.STAGES, index=1, key="ab_stage")
    with controls[1]:
        ab_model = st.selectbox(
            "Model", services.MODELS,
            index=services.MODELS.index("gradient_boosting")
            if "gradient_boosting" in services.MODELS else 0,
            key="ab_model",
        )
    with controls[2]:
        metric = st.selectbox(
            "Metric", ["brier_skill", "roc_auc", "pr_auc"], key="ab_metric"
        )

    groups = services.feature_groups(dataset, ab_stage)
    chosen = st.multiselect(
        "Groups to test", list(groups), default=list(groups),
        format_func=lambda name: f"{name} ({len(groups[name])})",
    )
    st.caption(
        "A season-level walk-forward per group, plus one for the full model. "
        "Roughly half a second per group on this dataset, and cached after the "
        "first run."
    )

    if st.button("Run ablation", type="primary", key="ab_run") and chosen:
        payload = tuple((name, tuple(groups[name])) for name in chosen)
        table = services.cached_ablation(
            version, stage=ab_stage, model=ab_model, metric=metric, groups=payload
        )
        left, right = st.columns([3, 2])
        with left:
            deltas = table.loc[table["group"] != "(full model)"].copy()
            fig = charts.diverging_bars(
                deltas, label="group", value="delta",
                xtitle=f"change in {metric} when the group is removed",
                height=max(280, 46 * len(deltas)),
            )
            charts.show(fig)
            st.caption(
                "**Negative means the group was carrying weight** — removing it "
                "hurt. Positive means the model was better without it."
            )
        with right:
            st.dataframe(
                table.round(4), hide_index=True, width="stretch",
                height=max(280, 46 * len(table)),
            )

# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #

with tab_importance:
    st.caption(
        "Shuffle one feature in a held-out season and measure what the forecast "
        "loses. Model-agnostic and measured out of sample, so it reflects what a "
        "feature contributes to a forecast rather than how often a tree happened "
        "to split on it."
    )
    controls = st.columns([2, 2, 2, 2])
    with controls[0]:
        pi_stage = st.selectbox("Stage", services.STAGES, index=1, key="pi_stage")
    with controls[1]:
        pi_model = st.selectbox(
            "Model", services.MODELS,
            index=services.MODELS.index("gradient_boosting")
            if "gradient_boosting" in services.MODELS else 0,
            key="pi_model",
        )
    with controls[2]:
        seasons = sorted(dataset["Year"].dropna().unique().astype(int))
        test_season = st.selectbox(
            "Held-out season", seasons[::-1], index=0, key="pi_season"
        )
    with controls[3]:
        repeats = st.number_input("Repeats", 3, 30, 10, step=1, key="pi_repeats")

    top_n = st.slider("Show top N", 5, 40, 20, key="pi_top")

    if st.button("Run importance", type="primary", key="pi_run"):
        table = services.cached_permutation_importance(
            version, stage=pi_stage, model=pi_model,
            test_season=int(test_season), n_repeats=int(repeats),
        )
        # ``skill_cost`` is the Brier-skill the forecast loses when the feature
        # is shuffled, which is the quantity worth ranking on; ``brier_increase``
        # is the same thing before it is put on the skill scale.
        value_col = next(
            (c for c in ("skill_cost", "brier_increase") if c in table.columns),
            table.select_dtypes("number").columns[0],
        )
        name_col = next(
            (c for c in ("feature", "name") if c in table.columns), table.columns[0]
        )
        top = table.nlargest(top_n, value_col).copy()
        top["description"] = top[name_col].map(services.describe_feature)

        left, right = st.columns([3, 2])
        with left:
            fig = charts.ranked_bars(
                top, label=name_col, value=value_col,
                hover=("description",), highlight=3,
                xtitle=(
                    "Brier skill lost when shuffled"
                    if value_col == "skill_cost"
                    else "Brier score added when shuffled"
                ),
                height=max(320, 24 * len(top)),
            )
            charts.show(fig)
            st.caption(
                "The three darkened bars are the features the forecast leans on "
                "hardest. A feature near zero is not useless — it may be "
                "saying the same thing as one above it."
            )
        with right:
            st.dataframe(
                table.round(5), hide_index=True, width="stretch",
                height=max(320, 24 * len(top)),
            )

# --------------------------------------------------------------------------- #
# Saved runs
# --------------------------------------------------------------------------- #

with tab_compare:
    runs = st.session_state["lab_runs"]
    if not runs:
        st.info(
            "Run a walk-forward and it is kept here, so two configurations can "
            "be read side by side. Cleared when the app restarts."
        )
    else:
        table = pd.DataFrame(runs).T.reset_index(names="run")
        st.dataframe(
            table.round(4), hide_index=True, width="stretch"
        )
        metric = st.selectbox(
            "Compare on", ["brier_skill", "roc_auc", "pr_auc", "calibration_slope"]
        )
        fig = charts.ranked_bars(
            table, label="run", value=metric, xtitle=metric,
            height=max(240, 52 * len(table)),
        )
        charts.show(fig)
        if st.button("Clear saved runs"):
            st.session_state["lab_runs"] = {}
            st.rerun()
