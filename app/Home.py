"""Overview: who wins the titles, what happens next Sunday, and can I trust it?

The three questions the forecast exists to answer, above the fold, with the
caveats beside the numbers rather than a page away.  The championship and the
next race come from ``src.models.forecast`` -- attrition, a stagewise
Plackett-Luce over the finishers, points, then a season -- so they match what
``python -m src.models.forecast season`` and ``next`` print.

Run it from the repository root::

    ./.venv/bin/python -m streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit executes this file as a script rather than importing it as part of
# the package, so the repository root is not on the path by default.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import charts, services  # noqa: E402
from app.common import (  # noqa: E402
    fmt_pct, format_odds, metric_row, page_setup, require_dataset,
    sidebar_models,
)
from src.models import forecast, season  # noqa: E402

page_setup("F1 forecast", icon="🏁")

version = services.data_version()
if not require_dataset():
    st.stop()

dataset = services.load_dataset(version)
manifest = services.manifest_or_none()
sidebar_models(dataset, manifest)

st.title("F1 forecast")

ready, why = services.forecast_ready()
if not ready:
    st.error(f"The forecast needs the raw results as well as the dataset: {why}.")
    st.code("./.venv/bin/python -m src.data.generate_dataset --skip-download",
            language="bash")
    st.stop()

fc_version = services.forecast_version()
year, through = services.forecast_position(fc_version)
st.caption(
    f"{year} after round {through}. Where each driver finishes and who wins "
    "both titles, from a model fitted only on races already run."
)

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #

try:
    drivers, teams = services.cached_championship(
        fc_version, year=year, through_round=through,
        team_sd=float(forecast.DEFAULT_SEASON_NOISE.team_sd),
        trials=season.DEFAULT_TRIALS,
    )
    outlook, event = services.cached_pre_weekend(
        fc_version, year=year, through_round=through, trials=season.DEFAULT_TRIALS,
    )
except Exception as exc:  # noqa: BLE001 - e.g. a finished season with no races left
    st.info(
        f"Nothing left to forecast from {year} R{through}: {exc}. The "
        "**Race** page still backtests any completed race."
    )
    st.stop()

next_fav = outlook.sort_values("p_win").iloc[-1]
metric_row([
    ("Drivers' favourite", str(drivers.iloc[0]["Abbreviation"]),
     f"{format_odds(drivers.iloc[0]['p_champion'])} to win the title"),
    ("Constructors' favourite", str(teams.iloc[0]["TeamName"]),
     f"{format_odds(teams.iloc[0]['p_champion'])}"),
    (f"Next: R{event['round_number']}", str(event["event_name"]).replace(" Grand Prix", " GP"),
     f"{event['race_date'].date()}{' · sprint' if event['has_sprint'] else ''}"),
    ("Next race favourite", str(next_fav["Abbreviation"]),
     f"{format_odds(next_fav['p_win'])} to win"),
])

left, right = st.columns([3, 2])
with left:
    st.subheader("Drivers' title odds")
    contenders = drivers.loc[drivers["p_champion"] >= 0.005].head(8)
    if contenders.empty:
        contenders = drivers.head(3)
    fig = charts.ranked_bars(
        contenders, label="Abbreviation", value="p_champion",
        hover=("TeamName",), xtitle="P(champion)",
        height=max(220, 34 * len(contenders)),
    )
    fig.update_xaxes(tickformat=".0%")
    charts.show(fig)
    st.caption("Drivers with at least a 0.5% chance. **Championship** has the "
               "points bands and the constructors.")
with right:
    st.subheader(f"{event['event_name']}")
    top = outlook.sort_values("p_win", ascending=False).head(5)
    st.dataframe(
        top[["Abbreviation", "TeamName", "p_win", "p_podium", "exp_points"]].rename(
            columns={"Abbreviation": "driver", "TeamName": "team",
                     "p_win": "P(win)", "p_podium": "P(podium)",
                     "exp_points": "exp pts"}),
        hide_index=True, width="stretch",
        column_config={
            "P(win)": st.column_config.ProgressColumn(format="%.3f", min_value=0.0,
                                                      max_value=1.0),
            "P(podium)": st.column_config.ProgressColumn(format="%.3f",
                                                         min_value=0.0, max_value=1.0),
            "exp pts": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption("Before qualifying, so a prior rather than a forecast. **Race** "
               "sharpens it with the grid once qualifying has run.")

# --------------------------------------------------------------------------- #
# Can I trust it?
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("How far to trust it")

fresh, freshness = (services.model_staleness(dataset, manifest)
                    if manifest is not None else (False, "no attrition model saved"))
backtest = services.season_calibration()
coverage = None
if backtest is not None:
    calibration, _ = backtest
    row = calibration.loc[
        (calibration["team_sd"].sub(forecast.DEFAULT_SEASON_NOISE.team_sd).abs() < 1e-9)
        & (calibration["driver_sd"] == 0)
    ].set_index("kind")
    if not row.empty:
        coverage = (row.loc["driver", "coverage80"], row.loc["team", "coverage80"])

# Read from the evaluation report when it has been written; otherwise the
# figures recorded in points_model_results.md when the model was fixed.
slopes = (0.81, 0.95, 1.05)
evaluation = services.order_evaluation()
if evaluation is not None:
    tuned = evaluation.loc[(evaluation["stage"] == "post_quali")
                           & (evaluation["model"] == "selected + tuned")]
    if len(tuned):
        slopes = tuple(float(tuned.iloc[0][c])
                       for c in ("calib_win", "calib_podium", "calib_points"))

with st.container(border=True):
    st.markdown(
        "- **Race probabilities are close to calibrated after qualifying.** "
        "On held-out 2024+ races the calibration slopes were "
        f"{slopes[0]:.2f} for the win, {slopes[1]:.2f} for the podium and "
        f"{slopes[2]:.2f} for the points (1.00 is perfect).\n"
        "- **Before qualifying, read the numbers as a prior.** Without a grid "
        "there is far less to go on, and attrition is near its base rate.\n"
        "- **The title odds are too confident.** The season simulation's 80% "
        "bands held "
        + (f"{fmt_pct(coverage[0])} of drivers' and {fmt_pct(coverage[1])} of "
           "constructors' final totals" if coverage else
           "about 70% of drivers' and 73% of constructors' final totals")
        + " in backtests, so odds beyond 1% and 99% print as `<1%` and `>99%`.\n"
        f"- **Attrition model:** {freshness}."
    )
st.caption("Full scores and their protocol on **Track record**; the features "
           "behind each number on **Model**.")
