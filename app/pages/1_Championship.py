"""Both championships, simulated: who wins the titles, and how sure to be.

The last stage of the forecast.  The pre-weekend finishing-order model is
fitted on the races so far, the attrition model supplies each car's chance of
not finishing, and the rest of the season is drawn ten thousand times with
each team's pace allowed to drift -- because a frozen running order covered
only 48% of constructors' outcomes in its nominal 80% band.  Everything comes
from ``src.models.forecast`` via ``app.services``, so it prints what
``python -m src.models.forecast season`` prints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import charts, services  # noqa: E402
from app.common import (  # noqa: E402
    fmt_pct, format_odds, metric_row, page_setup, require_dataset,
    sidebar_models, sidebar_trials,
)
from src.models import forecast  # noqa: E402

page_setup("Championship", icon="🏆")

if not require_dataset():
    st.stop()
ready, why = services.forecast_ready()
if not ready:
    st.title("Championship")
    st.error(f"The forecast needs the raw results as well as the dataset: {why}.")
    st.code("./.venv/bin/python -m src.data.generate_dataset --skip-download",
            language="bash")
    st.stop()

sidebar_models(services.load_dataset(services.data_version()),
               services.manifest_or_none())
trials = sidebar_trials()

version = services.forecast_version()
year, through = services.forecast_position(version)

st.title("Championship")
st.caption(
    f"{year} after round {through}: the remaining races simulated "
    f"{trials:,} times."
)

shipped = forecast.DEFAULT_SEASON_NOISE.team_sd
with st.expander("Pace drift (advanced)"):
    team_sd = st.slider(
        "Team pace drift", 0.0, 1.0, float(shipped), 0.1, key="fc_team_sd",
        help="How far a team's pace may wander between now and the last "
             "race, in log-strength. 0 freezes today's running order; the "
             "shipped value was chosen on regime-shift seasons.",
    )
    if abs(team_sd - shipped) > 1e-9:
        st.warning(
            f"Shipped default is **{shipped}**, measured by `forecast "
            "backtest`. Anything else is an untested setting."
        )

drivers, teams = services.cached_championship(
    version, year=year, through_round=through, team_sd=float(team_sd),
    trials=int(trials),
)
setup = services.cached_season_setup(version, year, through)

leader = drivers.iloc[0]
runner_up = drivers.iloc[1]
metric_row([
    ("Favourite", str(leader["Abbreviation"]),
     f"{format_odds(leader['p_champion'])} to win the title"),
    ("Closest challenger", str(runner_up["Abbreviation"]),
     f"{format_odds(runner_up['p_champion'])}"),
    ("Points now", f"{leader['points_so_far']:.0f}",
     f"{runner_up['Abbreviation']} on {runner_up['points_so_far']:.0f}"),
    ("Still to run", f"{len(setup.races)} races",
     f"{int(setup.races['has_sprint'].sum())} with a sprint"),
])

board = st.radio("Championship", ["Drivers", "Constructors"], horizontal=True,
                 key="fc_board")
if board == "Drivers":
    table, name_col = drivers, "Abbreviation"
else:
    table, name_col = teams, "TeamName"

# The band is the point: a bar of means would be the first-order answer with
# the uncertainty -- the reason to simulate at all -- thrown away.
fig = charts.interval_bars(
    table, label=name_col, low="p10", high="p90", mid="mean",
    now="points_so_far", height=max(320, 30 * len(table)),
    xtitle="season points",
)
charts.show(fig)
st.caption(
    "Grey band: the 10th to 90th percentile of simulated final points. "
    "Tick: points today. Dot: the mean. The gap between them is what the "
    "model expects the car still to earn."
)

shown = table.copy()
shown["Champion"] = shown["p_champion"].map(format_odds)
shown["Top 3"] = shown["p_top3"].map(format_odds)
columns = [name_col] + (["TeamName"] if board == "Drivers" else []) + [
    "points_so_far", "mean", "p10", "p50", "p90", "Champion", "Top 3",
]
st.dataframe(
    shown[columns].rename(columns={
        "TeamName": "team", "Abbreviation": "driver",
        "points_so_far": "now", "mean": "mean", "p10": "p10",
        "p50": "median", "p90": "p90",
    }),
    hide_index=True, width="stretch",
    column_config={
        "now": st.column_config.NumberColumn(format="%.0f"),
        "mean": st.column_config.NumberColumn(format="%.1f"),
        "p10": st.column_config.NumberColumn(format="%.0f"),
        "median": st.column_config.NumberColumn(format="%.0f"),
        "p90": st.column_config.NumberColumn(format="%.0f"),
    },
)
if (table["TeamName"] if board == "Drivers" else pd.Series(dtype=str)).str.endswith(
    "(no longer racing)"
).any():
    st.caption(
        "A driver who has left the grid keeps the points scored and is "
        "still ranked against everyone in every trial."
    )

st.divider()
st.subheader("How far to trust these odds")
backtest = services.season_calibration()
if backtest is None:
    st.info(
        "No calibration table on disk. Build it once with "
        "`./.venv/bin/python -m src.models.forecast backtest --write` "
        "(minutes, not seconds) and this section fills in."
    )
else:
    calibration, seasons = backtest
    row = calibration.loc[
        (calibration["team_sd"].sub(shipped).abs() < 1e-9)
        & (calibration["driver_sd"] == 0)
    ].set_index("kind")
    if row.empty:
        st.info("The backtest on disk was run without the shipped setting.")
    else:
        metric_row([
            ("Driver 80% band held", fmt_pct(row.loc["driver", "coverage80"]),
             f"should be 80% · {int(row.loc['driver', 'entities'])} driver-seasons"),
            ("Constructor 80% band held", fmt_pct(row.loc["team", "coverage80"]),
             f"should be 80% · {int(row.loc['team', 'entities'])} team-seasons"),
        ])
        st.caption(
            f"Backtested on {', '.join(map(str, seasons))}, forecast from "
            "part-way through each season, at the shipped setting. Both "
            "fall short of 80%, so **the bands above are too narrow and "
            "the extreme odds too confident.** Constructors are the weaker "
            "of the two: team-mates share one pace drift, so nothing "
            "diversifies a team's total. That is why odds beyond 1% and "
            "99% print as `<1%` and `>99%`: read them as 'very unlikely' "
            "and 'very likely', not as a number."
        )

