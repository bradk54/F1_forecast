"""Rank a grid by retirement risk — the "who do I flag on Sunday" page.

Two modes, and the difference between them is the whole point:

* **A race in the dataset** is a backtest.  If the saved model was trained
  through that race it has already seen the answer, and the page says so rather
  than quietly showing flattering numbers.
* **An upcoming race** has no feature row, so one is built by appending
  placeholder rows to the raw results and running the ordinary feature builders
  over the whole thing.  That is safe only because every feature in this
  pipeline is strictly prior-race, which is the same property that makes the
  training numbers trustworthy.

**Stage is not a confidence dial.**  ``grid_position`` is the model's strongest
feature, and a missing grid is imputed to the back of the field — which inflates
every prediction into something that looks like an opinion rather than an
absence of information.  So a ``post_quali`` run without a grid is refused here
exactly as it is in the CLI, and the escape hatch is labelled for what it does.
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
    fmt_pct, metric_row, page_setup, require_dataset, sidebar_status,
)
from src.models import monitor, store  # noqa: E402
from src.models.predict import (  # noqa: E402
    build_inference_rows, entry_list, next_event, qualifying_grid,
)

page_setup("Race weekend", icon="🏎️")

version = services.data_version()
if not require_dataset():
    st.stop()

dataset = services.load_dataset(version)
manifest = services.manifest_or_none()
sidebar_status(dataset, manifest)

st.title("Race weekend")

if manifest is None:
    st.error("No model has been fitted. Use **Refit and save** in the sidebar.")
    st.stop()
if not services.artefacts().results:
    st.error("No raw results on disk; an upcoming race cannot be assembled.")
    st.stop()

stage = manifest.stage
st.caption(
    f"Using the saved **{manifest.model}** model at stage **`{stage}`**, trained "
    f"through {manifest.trained_through_year} R{manifest.trained_through_round}. "
    "Change the stage by refitting in the sidebar — a stage is a different "
    "model, not a setting."
)

# --------------------------------------------------------------------------- #
# Pick a race
# --------------------------------------------------------------------------- #

races = services.race_index(version, dataset)
choice = st.radio(
    "Race", ["A completed race (backtest)", "The next scheduled race"],
    horizontal=True, key="weekend_mode",
)

event: dict | None = None
grid: dict[str, float] | None = None
grid_source = "not supplied"

if choice.startswith("A completed"):
    label = st.selectbox("Which race", races["label"], index=0)
    row = races.loc[races["label"] == label].iloc[0]
    event = {
        "year": int(row["Year"]), "round_number": int(row["RoundNumber"]),
        "event_name": str(row["EventName"]),
        "race_date": pd.Timestamp(row[services.ORDER_COL]),
    }
    if stage == "post_quali":
        # The real grid is already in the dataset for a race that has run, so
        # there is no reason to go to the network for it.
        actual = dataset.loc[
            (dataset["Year"] == event["year"])
            & (dataset["RoundNumber"] == event["round_number"])
        ]
        grid = {
            str(r["DriverId"]): float(r["grid_position"])
            for _, r in actual.iterrows()
            if pd.notna(r.get("grid_position"))
        } or None
        grid_source = "the dataset's own recorded grid"
else:
    st.caption(
        "Reading the calendar needs FastF1, which will use the network if the "
        "cache is cold. Nothing is written."
    )
    if st.button("Look up the next race", type="primary"):
        try:
            st.session_state["weekend_next"] = next_event(dataset)
        except Exception as exc:  # noqa: BLE001 - network and cache both fail here
            st.session_state["weekend_next"] = None
            st.error(f"Could not read the calendar: {exc}")
    event = st.session_state.get("weekend_next")
    if event is None:
        st.info("No upcoming race loaded yet.")
    else:
        st.success(
            f"{event['year']} R{event['round_number']} {event['event_name']} — "
            f"{pd.Timestamp(event['race_date']).date()}"
        )
        if stage == "post_quali":
            fetch = st.checkbox(
                "Fetch the grid from qualifying (uses the network)", value=False
            )
            if fetch:
                with st.spinner("Loading qualifying ..."):
                    grid = qualifying_grid(event["year"], event["round_number"])
                grid_source = "qualifying" if grid else "qualifying (not available)"

if event is None:
    st.stop()

# --------------------------------------------------------------------------- #
# The post_quali guard
# --------------------------------------------------------------------------- #

allow_missing = False
if stage == "post_quali" and grid is None:
    st.error(
        "**No grid, and this is a `post_quali` model.** A missing grid slot is "
        "imputed to the back of the field, so every driver would be scored as a "
        "back-marker and every probability would come out inflated — roughly "
        "0.20–0.46 against a 14% base rate. That looks like a model with an "
        "opinion rather than one with no information."
    )
    st.caption(
        "Refit at `pre_weekend` in the sidebar for a model fitted without grid "
        "position at all, which is the correct thing to run before qualifying."
    )
    allow_missing = st.checkbox(
        "Predict anyway, treating every driver as starting last", value=False
    )
    if not allow_missing:
        st.stop()

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #

results, profiles = services.load_raw(version)
estimator, loaded = services.load_model_unchecked(version)

try:
    rows = build_inference_rows(
        results, profiles,
        year=event["year"], round_number=event["round_number"],
        race_date=event["race_date"], event_name=event["event_name"],
        entries=entry_list(results), grid=grid,
    )
except Exception as exc:  # noqa: BLE001 - a build failure is the answer here
    st.error(f"Could not assemble feature rows for this race: {exc}")
    st.stop()

missing = [c for c in loaded.features if c not in rows.columns]
if missing:
    st.error(
        f"{len(missing)} feature(s) the model was fitted on were not produced "
        f"for this race ({missing[:5]}). Refit from the sidebar."
    )
    st.stop()

rows["predicted"] = estimator.predict_proba(rows[loaded.features])[:, 1]

already_run = (
    (results["Year"] == event["year"])
    & (results["RoundNumber"] == event["round_number"])
).any()
in_sample = already_run and (event["year"], event["round_number"]) <= (
    manifest.trained_through_year, manifest.trained_through_round
)
if in_sample:
    st.warning(
        "**In-sample.** This race has run *and* the saved model was trained "
        "through it, so these numbers flatter the model. Use the **Lab** page "
        "for an honest out-of-sample score."
    )
elif already_run:
    st.info("This race has already run; the model was trained before it.")

# --------------------------------------------------------------------------- #
# The ranking
# --------------------------------------------------------------------------- #

ranked = rows.sort_values("predicted", ascending=False).reset_index(drop=True)
ranked.insert(0, "rank", ranked.index + 1)
show_grid = grid is not None

metric_row(
    [
        ("Entries", f"{len(ranked)}", f"grid from {grid_source}"),
        ("Mean P(dnf)", fmt_pct(ranked["predicted"].mean()),
         f"trained base rate {manifest.train_base_rate:.1%}"),
        ("Riskiest", str(ranked.iloc[0]["DriverId"]),
         fmt_pct(ranked.iloc[0]["predicted"])),
        ("Spread", f"{ranked['predicted'].max() - ranked['predicted'].min():.3f}",
         "max − min across the field"),
    ]
)

drift_ok, drift_message = services.drift(dataset, manifest)
(st.success if drift_ok else st.error)(drift_message)

left, right = st.columns([3, 2])

with left:
    frame = ranked.copy()
    frame["driver"] = frame["DriverId"].astype(str)
    frame["team"] = frame["TeamId"].astype(str)
    frame["grid"] = (
        frame["grid_position"].map(lambda v: f"{int(v)}" if pd.notna(v) else "n/a")
        if show_grid else "n/a"
    )
    fig = charts.ranked_bars(
        frame, label="driver", value="predicted",
        hover=("team", "grid"), highlight=2,
        reference=float(manifest.train_base_rate),
        reference_label="training base rate",
        xtitle="P(retirement)", height=max(360, 26 * len(frame)),
    )
    fig.update_xaxes(tickformat=".0%")
    charts.show(fig)
    st.caption(
        "The two darkened bars are the picks the top-2 metric scores. Colour "
        "carries nothing else — the bar length is the probability."
    )

with right:
    display = ranked[["rank", "DriverId", "TeamId", "predicted"]].copy()
    display.columns = ["#", "driver", "team", "P(dnf)"]
    if show_grid:
        display.insert(3, "grid", ranked["grid_position"].astype("Int64"))
    st.dataframe(
        display, hide_index=True, width="stretch",
        height=max(360, 26 * len(display)),
        column_config={
            "P(dnf)": st.column_config.ProgressColumn(
                "P(dnf)", format="%.3f", min_value=0.0,
                max_value=float(max(0.3, ranked["predicted"].max())),
            )
        },
    )

# --------------------------------------------------------------------------- #
# Grid effect, and what actually happened
# --------------------------------------------------------------------------- #

if show_grid:
    st.divider()
    st.subheader("Grid position against predicted risk")
    st.caption(
        "The strongest single feature in the model. The back of the grid both "
        "breaks more and gets collected at turn one more."
    )
    points = ranked.dropna(subset=["grid_position"]).copy()
    points["driver"] = points["DriverId"].astype(str)
    points["team"] = points["TeamId"].astype(str)
    fig = charts.scatter(
        points, x="grid_position", y="predicted", text="driver",
        hover=("team",), xtitle="grid position", ytitle="P(retirement)",
        height=340,
    )
    fig.update_yaxes(tickformat=".0%")
    charts.show(fig)

if already_run:
    st.divider()
    st.subheader("What actually happened")

    # The inference rows carry their own placeholder ``dnf``/``Status`` columns,
    # so the outcome columns are renamed before the join rather than left to
    # collide into ``_x``/``_y`` suffixes.
    actual = (
        dataset.loc[
            (dataset["Year"] == event["year"])
            & (dataset["RoundNumber"] == event["round_number"])
        ][["DriverId", "dnf", "Status", "ClassifiedPosition"]]
        .rename(columns={
            "dnf": "actual_dnf", "Status": "actual_status",
            "ClassifiedPosition": "actual_position",
        })
    )
    merged = ranked.drop(columns=["dnf", "Status"], errors="ignore").merge(
        actual, on="DriverId", how="left"
    )
    scored = merged.dropna(subset=["actual_dnf"])

    if scored.empty:
        st.info("No outcomes joined for this race.")
    else:
        from src.models.monitor import top_k_lift
        from src.models.train import score_predictions

        scores = score_predictions(
            scored["actual_dnf"], scored["predicted"], manifest.train_base_rate
        )
        hits, lift = top_k_lift(scored["actual_dnf"], scored["predicted"])
        metric_row(
            [
                ("Retirements",
                 f"{int(scored['actual_dnf'].sum())} of {len(scored)}",
                 fmt_pct(scores["observed_rate"])),
                ("Brier skill", f"{scores['brier_skill']:+.4f}",
                 "vs the training base rate"),
                ("ROC AUC",
                 "undefined" if np.isnan(scores["roc_auc"]) else f"{scores['roc_auc']:.3f}",
                 "undefined when nobody retired"),
                ("Top-2 caught", f"{hits} of 2",
                 "—" if np.isnan(lift) else f"{lift:.1f}× chance"),
            ]
        )
        if in_sample:
            st.caption("These are in-sample numbers. Read them as a sanity check, not a score.")

        outcome = scored[
            ["rank", "DriverId", "TeamId", "predicted", "actual_dnf", "actual_status"]
        ].copy()
        outcome["result"] = np.where(
            outcome["actual_dnf"] == 1, "retired", "finished"
        )
        outcome = outcome.drop(columns=["actual_dnf"])
        outcome.columns = ["#", "driver", "team", "P(dnf)", "status", "result"]
        st.dataframe(outcome, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Saving a pending prediction
# --------------------------------------------------------------------------- #

if not already_run:
    st.divider()
    st.caption(
        "Saving a pending prediction is what makes the next `refresh` able to "
        "score it. The log entry it produces is a **live** row — the only kind "
        "that proves the weekly loop works."
    )
    if st.button("Save this prediction for scoring at the next refresh"):
        # Both writes, in the same order as ``_predict_event``: the pending file
        # is what the next refresh scores, and the per-driver log is what
        # answers "what did we say about this car beforehand". Doing only the
        # first would leave the app and the CLI writing different histories.
        keep = ["Year", "RoundNumber", "DriverId", "TeamId", "predicted"]
        if "grid_position" in ranked.columns:
            keep.append("grid_position")
        pending_path = store.save_pending(ranked[keep], manifest)
        predictions_path = monitor.append_predictions(
            ranked, manifest,
            year=event["year"], round_number=event["round_number"],
            event=event["event_name"], race_date=event["race_date"],
        )
        st.success(
            f"Queued for scoring in `{pending_path.name}`, and {len(ranked)} "
            f"per-driver rows appended to `{predictions_path.name}`."
        )
