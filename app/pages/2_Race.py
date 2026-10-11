"""One race: the finishing order, and the retirement risk underneath it.

The page a race weekend is read from.  Pick the next race or any completed
one; the finishing-order forecast comes first because that is the product, and
the attrition stage -- the saved retirement model, ranked -- sits beneath it.

**Stage follows what the race has, not a setting.**  Before qualifying the
forecast is the pre-weekend model, fitted without grid position at all.  With a
grid -- fetched after qualifying, or a completed race's own -- it is the
post-qualifying model.  They are different models, not one model at two
confidence levels.

**A completed race is always out-of-sample for the finishing order**: it is
refitted on the races before it.  The saved retirement model is not refitted,
so if it was trained through the race, the attrition section says so.

**The retirement guard survives the merge.**  ``grid_position`` is the
retirement model's strongest feature and a missing grid is imputed to the back
of the field, so a ``post_quali`` retirement model with no grid is refused here
exactly as in the CLI.  That refusal stops only the attrition section; the
finishing order above it has already rendered.
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
    fmt_pct, format_odds, metric_row, outlook_table, page_setup,
    require_dataset, sidebar_models, sidebar_trials,
)
from src.models import monitor, store  # noqa: E402
from src.models.predict import (  # noqa: E402
    build_inference_rows, entry_list, qualifying_grid,
)

page_setup("Race", icon="🏎️")

version = services.data_version()
if not require_dataset():
    st.stop()
ready, why = services.forecast_ready()
if not ready:
    st.title("Race")
    st.error(f"The forecast needs the raw results as well as the dataset: {why}.")
    st.code("./.venv/bin/python -m src.data.generate_dataset --skip-download",
            language="bash")
    st.stop()

dataset = services.load_dataset(version)
manifest = services.manifest_or_none()
sidebar_models(dataset, manifest)
trials = sidebar_trials()

fc_version = services.forecast_version()
fc_year, fc_through = services.forecast_position(fc_version)
results, profiles = services.load_raw(version)

# The saved retirement model's stage decides whether a grid is passed to it.
stage = manifest.stage if manifest is not None else None

st.title("Race")

# --------------------------------------------------------------------------- #
# Pick a race
# --------------------------------------------------------------------------- #

choice = st.radio(
    "Race", ["The next race", "A completed race (backtest)"],
    horizontal=True, key="race_mode",
)

event: dict
grid: dict[str, float] | None = None
grid_source = "not supplied"
outlook = None
fc_source = ""

if choice == "The next race":
    try:
        pre_outlook, upcoming = services.cached_pre_weekend(
            fc_version, year=fc_year, through_round=fc_through, trials=trials)
    except Exception as exc:  # noqa: BLE001 - e.g. no calendar for a finished season
        st.error(f"Could not set up the next race: {exc}")
        st.stop()
    event = {k: upcoming[k] for k in ("year", "round_number", "event_name", "race_date")}
    st.subheader(f"{event['year']} R{event['round_number']} {event['event_name']}")
    st.caption(
        f"{pd.Timestamp(event['race_date']).date()}"
        f"{' · sprint weekend' if upcoming['has_sprint'] else ''}"
    )

    fetched = st.session_state.get("race_grid")
    if fetched is not None and fetched["key"] != (event["year"], event["round_number"]):
        fetched = None  # a grid for a race that has since been run
    if st.button("Fetch the qualifying grid (one network request)", key="race_fetch",
                 help="Penalties are not applied, so a penalised car is forecast "
                      "from its qualifying slot. Nothing is written."):
        with st.spinner("Loading qualifying ..."):
            try:
                loaded = qualifying_grid(event["year"], event["round_number"])
            except Exception as exc:  # noqa: BLE001 - network and cache both fail here
                loaded = None
                st.error(f"Could not load qualifying: {exc}")
        if loaded is None:
            st.warning("Qualifying has not run, or could not be loaded.")
            st.session_state.pop("race_grid", None)
            fetched = None
        else:
            fetched = {"key": (event["year"], event["round_number"]), "grid": loaded}
            st.session_state["race_grid"] = fetched

    if fetched is not None:
        grid, grid_source = fetched["grid"], "qualifying"
        outlook = services.cached_post_quali(
            fc_version,
            event=(event["year"], event["round_number"], event["event_name"],
                   pd.Timestamp(event["race_date"])),
            grid=tuple(sorted(grid.items())), trials=trials,
        )
        fc_source = ("the **post-qualifying** model on the qualifying grid "
                     "(penalties not yet applied)")
    else:
        outlook = pre_outlook
        fc_source = ("the **pre-weekend** model, fitted without a grid -- a "
                     "prior, not a forecast. Fetch the grid after qualifying "
                     "for the sharper one")
else:
    races = services.race_index(version, dataset)
    # The post-qualifying model wants a training window behind it; the first
    # seasons have too little history to fit honestly.
    first = min(int(races["Year"].min()) + 3, int(races["Year"].max()))
    races = races.loc[races["Year"] >= first].reset_index(drop=True)
    label = st.selectbox("Which completed race", races["label"], index=0, key="race_pick")
    row = races.loc[races["label"] == label].iloc[0]
    event = {
        "year": int(row["Year"]), "round_number": int(row["RoundNumber"]),
        "event_name": str(row["EventName"]),
        "race_date": pd.Timestamp(row[services.ORDER_COL]),
    }
    # The real grid is already in the dataset for a race that has run, so
    # there is no reason to go to the network for it.
    actual = dataset.loc[(dataset["Year"] == event["year"])
                         & (dataset["RoundNumber"] == event["round_number"])]
    grid = {str(r["DriverId"]): float(r["grid_position"])
            for _, r in actual.iterrows() if pd.notna(r.get("grid_position"))} or None
    grid_source = "the race's recorded grid"
    try:
        outlook = services.cached_backtest_race(
            fc_version, year=event["year"], round_number=event["round_number"],
            trials=trials)
    except Exception as exc:  # noqa: BLE001 - a fit failure is the answer here
        st.error(f"Could not forecast this race: {exc}")
    fc_source = ("the post-qualifying model on the real grid, fitted **only on "
                 "the races before this one** -- out-of-sample by construction")

already_run = choice != "The next race"

# --------------------------------------------------------------------------- #
# Finishing order
# --------------------------------------------------------------------------- #

if outlook is not None:
    (st.success if grid is not None else st.info)(f"From {fc_source}.")
    favourite = outlook.sort_values("p_win").iloc[-1]
    items = [
        ("Favourite", str(favourite["Abbreviation"]),
         f"{format_odds(favourite['p_win'])} to win"),
        ("Most expected points", str(outlook.iloc[0]["Abbreviation"]),
         f"{outlook.iloc[0]['exp_points']:.1f} points"),
        ("Cars expected out", f"{outlook['p_dnf'].sum():.1f}",
         f"of {len(outlook)} entries"),
    ]
    if "finished" in outlook.columns:
        winner = outlook.loc[outlook["finished"] == 1]
        if len(winner):
            items.append(("Winner", str(winner.iloc[0]["Abbreviation"]),
                          f"forecast {format_odds(winner.iloc[0]['p_win'])}"))
    metric_row(items)

    left, right = st.columns([2, 3])
    with left:
        fig = charts.ranked_bars(
            outlook, label="Abbreviation", value="exp_points", highlight=3,
            hover=("TeamName",), xtitle="expected points",
            height=max(360, 26 * len(outlook)),
        )
        charts.show(fig)
    with right:
        outlook_table(outlook)
    st.caption(
        "P(out) is the chance of not finishing *or* not starting, from the "
        "attrition stage. A retired car scores nothing, so expected points is "
        "roughly P(finish) x the points its finishing order would earn."
        + (" `result` is the classified position; blank did not finish."
           if "finished" in outlook.columns else "")
        + " **Model** shows which features put each driver where."
    )

# --------------------------------------------------------------------------- #
# Retirement risk
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Attrition stage: who retires")
st.caption(
    "The forecast's first stage, on its own: each car's chance of not "
    "finishing, ranked. This is the saved retirement model, which can be "
    "rescored after the race and logged -- the finishing order above refits "
    "its own copy on the races before this one, so their numbers can differ."
)

if manifest is None:
    st.error("No attrition model has been saved. Use **Refit attrition model** "
             "in the sidebar.")
    st.stop()

st.caption(
    f"Using the saved **{manifest.model}** model at stage **`{stage}`**, trained "
    f"through {manifest.trained_through_year} R{manifest.trained_through_round}. "
    "Change the stage by refitting in the sidebar — a stage is a different "
    "model, not a setting."
)

# --------------------------------------------------------------------------- #
# The post_quali guard
# --------------------------------------------------------------------------- #

allow_missing = False
if stage == "post_quali" and grid is None:
    # Refused, as in the CLI -- but as information rather than an error: on the
    # next-race view before qualifying this is the expected state, not a fault.
    st.info(
        "**The retirement ranking needs the grid.** The saved attrition model "
        "is `post_quali`, and a missing grid slot is imputed to the back of "
        "the field, so every driver would be scored as a back-marker -- "
        "roughly 0.20-0.46 against a 14% base rate. Fetch the qualifying grid "
        "above, or refit at `pre_weekend` in the sidebar for a model fitted "
        "without grid position at all. The finishing order above already "
        "carries its own pre-weekend attrition."
    )
    allow_missing = st.checkbox(
        "Predict anyway, treating every driver as starting last", value=False
    )
    if not allow_missing:
        st.stop()

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #

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
# A fetched grid feeds the forecast whatever the saved stage, but only a
# post_quali retirement model actually uses it -- showing grid beside a
# pre_weekend ranking would imply an effect the model does not have.
show_grid = grid is not None and stage == "post_quali"

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
        "Saving the retirement ranking as a pending prediction is what makes "
        "the next `refresh` able to "
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
