"""What the forecast is made of: its stages, its features, and what each is worth.

The forecast is a pipeline, and this page walks it in order -- attrition, the
finishing order before and after qualifying, then points and the season.  For
each order model it shows three things, from most to least settled:

* **The features and what selection measured them to be worth.**  A removal
  cost in nats a race and a paired t, from the 2020-2023 development races.
  That is the honest measure of a feature's value; it is recorded in
  ``forecast.FEATURE_EVIDENCE`` and refreshed only by ``tune``.
* **The current fit** -- coefficients and per-position scales, point-fitted on
  the same window the forecast uses.  These move week to week.
* **One race, driver by driver** -- each feature's contribution to each car's
  strength.  The model is linear in its standardised features, so the parts
  sum exactly to the strength the forecast ranks by; nothing is approximated.

There is deliberately no permutation importance.  On a three-feature model it
restates the removal costs with more noise, and it invites the one misreading
this page exists to prevent: a strength is pace *conditional on finishing*,
not a driver ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import charts, services  # noqa: E402
from app.common import (  # noqa: E402
    metric_row, page_setup, require_dataset, sidebar_models,
)
from src.models import forecast  # noqa: E402

page_setup("Model", icon="🧩")

version = services.data_version()
if not require_dataset():
    st.stop()
ready, why = services.forecast_ready()
if not ready:
    st.title("Model")
    st.error(f"The forecast needs the raw results as well as the dataset: {why}.")
    st.stop()

dataset = services.load_dataset(version)
manifest = services.manifest_or_none()
sidebar_models(dataset, manifest)

fc_version = services.forecast_version()
year, through = services.forecast_position(fc_version)

st.title("Model")
st.caption(
    "How the forecast is built, which features it uses, and how much each one "
    "is worth. Coefficients are refitted on the data through "
    f"{year} R{through}."
)

# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

STAGES = (
    ("1 · Attrition", "Will the car finish?",
     "Logistic regression on grid position. Its job is a calibrated chance of "
     "retiring, not a ranking."),
    ("2 · Finishing order", "In what order do the finishers cross the line?",
     "Stagewise Plackett-Luce: a winner is drawn, then second from the rest, "
     "each position with its own scale."),
    ("3 · Points", "What is that order worth?",
     "The season's own points table, sprints and the fastest-lap point "
     "included where they existed."),
    ("4 · Season", "Who wins the titles?",
     "The remaining races drawn thousands of times, with each team's pace "
     "drifting as a shared random walk."),
)
for column, (title, question, how) in zip(st.columns(4), STAGES):
    with column, st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"*{question}*\n\n{how}")

st.warning(
    "**A strength is pace conditional on finishing.** Retirements are handled "
    "by stage 1 and left out of stage 2, so a high strength means fast *when "
    "the car gets to the end*. Do not read the order model's strengths as a "
    "driver ranking."
)

# --------------------------------------------------------------------------- #
# Per-stage views
# --------------------------------------------------------------------------- #


def evidence_table(stage: str, model) -> pd.DataFrame:
    """The shipped features with their recorded evidence and current weight."""
    coefficients = model.coefficients()
    rows = []
    for order, (feature, cost, t) in enumerate(forecast.FEATURE_EVIDENCE[stage], 1):
        rows.append({
            "taken": order, "feature": feature,
            # The first feature was measured against an empty model, not by
            # removal, so it gets words rather than a blank that reads as zero.
            "removal cost": "first pick" if cost is None else f"{cost:.2f}",
            "t": "" if t is None else f"{t:.1f}",
            "weight now": float(coefficients.get(feature, float("nan"))),
            "what it measures": services.describe_feature(feature),
        })
    return pd.DataFrame(rows)


def show_evidence(stage: str, model) -> None:
    table = evidence_table(stage, model)
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "taken": st.column_config.NumberColumn(
                "taken", help="Order forward selection added it."),
            "removal cost": st.column_config.TextColumn(
                "removal cost (nats/race)",
                help="Held-out log-likelihood lost per race by removing it. "
                     "The first feature taken was measured against an empty "
                     "model instead, so it has no removal cost."),
            "t": st.column_config.TextColumn("paired t"),
            "what it measures": st.column_config.TextColumn(
                "what it measures", width="large"),
            "weight now": st.column_config.NumberColumn(
                "weight now", format="%+.3f",
                help="Coefficient per standard deviation in the current fit. "
                     "The sign follows the feature's units: a percentile or a "
                     "grid slot is better when *lower*, so a negative weight "
                     "there means 'further forward is faster'."),
        },
    )
    st.caption(
        "Removal cost and t are from forward selection on the 2020-2023 "
        "development races (`References/points_model_results.md` Sec 5), "
        "where a feature had to clear t >= 2 *and* 0.02 nats to be kept. They "
        "are a record, not recomputed; `forecast tune` refreshes them."
    )


def show_scales(model) -> None:
    if model.alpha_ is None:
        st.caption("Standard Plackett-Luce: one scale for every position.")
        return
    scales = pd.DataFrame({
        "position": [f"P{i}" for i in range(1, 13)],
        "scale": model.alpha_[:12],
    })
    fig = charts.bars(scales, x="position", y="scale", ytitle="scale",
                      height=260, reference=1.0, reference_label="winner's scale")
    charts.show(fig)
    st.caption(
        "How sharply the strengths decide each finishing position, relative to "
        "the win. Below 1, that position is more of a lottery among the cars "
        "still unplaced. Standard Plackett-Luce forces one value on all of "
        "them, which left it underconfident about winners (calibration slope "
        "2.83) -- the reason the scales are free."
    )


def show_drivers(explained: pd.DataFrame, model, title: str) -> None:
    parts = {f"c_{f}": f for f in model.features}
    st.markdown(f"**{title}**")
    fig = charts.contribution_bars(
        explained, label="Abbreviation", parts=parts, total="strength",
        height=max(380, 26 * len(explained)),
        xtitle="contribution to log-strength, relative to the field average",
    )
    charts.show(fig)
    st.caption(
        "Each bar is one feature's push on a car's strength, centred on the "
        "field: right is faster than the average car in this race, left "
        "slower. The diamond is the net, which is what the order is drawn "
        "from. A gap of 1.0 between two cars makes the faster one e = 2.7 "
        "times as likely as the slower to be drawn as the winner."
    )
    with st.expander("Feature values"):
        shown = explained[["Abbreviation", "TeamName", *model.features, "strength"]]
        st.dataframe(shown, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in (*model.features, "strength")})


tab_pre, tab_post, tab_dnf = st.tabs(
    ["Before qualifying", "After qualifying", "Attrition stage"]
)

with tab_pre:
    st.caption(
        "Runs every forecast made before there is a grid: the next race until "
        "qualifying, and every race in the championship simulation."
    )
    model, _ = services.cached_order_fit(fc_version, "pre_weekend", year, through)
    spec = forecast.PRE_WEEKEND_SPEC
    metric_row([
        ("Features", f"{len(spec.features)}", "chosen from seventeen candidates"),
        ("Training window", f"{spec.lookback_races} races",
         f"half-life {spec.race_halflife:.0f} races"),
        ("Position scales", f"{spec.n_scales}", "free, beyond the winner's"),
    ])
    show_evidence("pre_weekend", model)
    show_scales(model)
    try:
        explained = services.explain_next_race(fc_version, year, through)
    except Exception as exc:  # noqa: BLE001 - e.g. no calendar left to forecast
        st.info(f"No upcoming race to break down: {exc}")
    else:
        setup = services.cached_season_setup(fc_version, year, through)
        race = setup.races.iloc[0]
        show_drivers(explained, model,
                     f"Next race: {year} R{int(race.RoundNumber)} {race.EventName}")
        st.caption(
            "The feature values are the ones the Championship and Race pages "
            "used. The breakdown uses a single fit on the same window; the "
            "forecast itself averages bootstrap refits, so its order can "
            "differ slightly where two cars are close."
        )

with tab_post:
    st.caption(
        "Runs once there is a grid: after qualifying, and for every backtest of "
        "a completed race."
    )
    model, _ = services.cached_order_fit(fc_version, "post_quali", year, through)
    spec = forecast.POST_QUALI_SPEC
    metric_row([
        ("Features", f"{len(spec.features)}", "grid first, then pace and form"),
        ("Training window", f"{spec.lookback_races} races",
         f"half-life {spec.race_halflife:.0f} races"),
        ("Position scales", f"{spec.n_scales}", "free, beyond the winner's"),
    ])
    show_evidence("post_quali", model)
    show_scales(model)
    races = services.race_index(version, dataset)
    label = st.selectbox("Break down a completed race", races["label"], index=0,
                         key="model_race")
    pick = races.loc[races["label"] == label].iloc[0]
    try:
        explained = services.explain_race(fc_version, int(pick["Year"]),
                                          int(pick["RoundNumber"]))
        before = services.previous_race(dataset, int(pick["Year"]),
                                        int(pick["RoundNumber"]))
        race_model, _ = services.cached_order_fit(fc_version, "post_quali", *before)
    except Exception as exc:  # noqa: BLE001 - e.g. the first race on record
        st.info(f"Cannot break this race down: {exc}")
    else:
        show_drivers(explained, race_model,
                     f"{label} -- fitted on the races before it")

with tab_dnf:
    st.caption(
        "Supplies each car's chance of not finishing. The order model ranks "
        "only the cars that finish, so this is where retirements enter the "
        "forecast."
    )
    if manifest is None:
        st.info("No attrition model saved yet; refit it from the sidebar.")
    else:
        metric_row([
            ("Model", manifest.model, f"stage `{manifest.stage}`"),
            ("Features", f"{len(manifest.features)}",
             ", ".join(manifest.features[:3])
             + (" ..." if len(manifest.features) > 3 else "")),
            ("Base rate", f"{manifest.train_base_rate:.1%}",
             f"{manifest.train_rows} training rows"),
        ])
        st.dataframe(
            pd.DataFrame({
                "feature": manifest.features,
                "what it measures": [services.describe_feature(f)
                                     for f in manifest.features],
            }),
            hide_index=True, width="stretch",
        )
    st.caption(
        "One feature is the measured answer, not a simplification: walked "
        "forward race by race over 2022-2026, a logistic on grid position "
        "alone beat every combination of model family and feature set, and "
        "every feature added past the grid made calibration worse. Its value "
        "to the forecast is the *level* of attrition more than the per-car "
        "differences. **Lab** runs those experiments live; **Data** has the "
        "full feature registry."
    )
