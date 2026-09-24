"""The finishing-order and championship forecast: who wins the title, and who wins Sunday.

This is the model the retirement model was always a component of -- attrition
from the DNF model, a stagewise Plackett-Luce over the survivors, then points,
then a season.  Everything on this page comes from ``src.models.forecast`` via
``app.services``, so it prints what ``python -m src.models.forecast`` prints.

Three views, in the order a fan asks the questions:

* **Championship** -- the rest of the season simulated ten thousand times.
* **Next race** -- per driver, before qualifying (no network) or after it.
* **Backtest a race** -- any completed race, forecast by a model that never saw
  it, beside what happened.  This is the view that earns the other two trust.

There is no saved model here to go stale, and so no refit button: the forecast
refits from the parquets in seconds, which is why its cache key is those files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import charts, services  # noqa: E402
from app.common import (  # noqa: E402
    fmt_pct, format_odds, metric_row, outlook_table, page_setup,
    require_dataset,
)
from src.models import forecast, season  # noqa: E402

page_setup("Forecast", icon="🏆")

if not require_dataset():
    st.stop()
ready, why = services.forecast_ready()
if not ready:
    st.title("Forecast")
    st.error(f"The forecast needs the raw results as well as the dataset: {why}.")
    st.code("./.venv/bin/python -m src.data.generate_dataset --skip-download",
            language="bash")
    st.stop()

version = services.forecast_version()
year, through = services.forecast_position(version)

st.title("Forecast")
st.caption(
    f"{year} after round {through}. Refits from the data on disk every time it "
    "is asked, so there is no saved model to go stale and nothing to refit here."
)

with st.sidebar:
    st.subheader("Simulation")
    trials = st.select_slider(
        "Trials", options=[2_000, 5_000, 10_000, 20_000],
        value=season.DEFAULT_TRIALS, key="fc_trials",
        help="Simulated seasons. More trials smooth the odds of rare outcomes; "
             "they do not make the model more right.",
    )
    st.caption(
        "The pace-drift setting lives on the Championship tab, beside the "
        "numbers it changes."
    )

tab_title, tab_next, tab_back = st.tabs(
    ["Championship", "Next race", "Backtest a race"]
)

# --------------------------------------------------------------------------- #
# Championship
# --------------------------------------------------------------------------- #

with tab_title:
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
                "part-way through each season, at the shipped setting. Both fall short of 80%, so **the bands above "
                "are too narrow and the extreme odds too confident.** "
                "Constructors are the weaker of the two: team-mates share one "
                "pace drift, so nothing diversifies a team's total. Treat 99% "
                "as 'very likely', not as 99%."
            )

# --------------------------------------------------------------------------- #
# Next race
# --------------------------------------------------------------------------- #

with tab_next:
    outlook, event = services.cached_pre_weekend(
        version, year=year, through_round=through, trials=int(trials)
    )
    st.subheader(f"R{event['round_number']} {event['event_name']}")
    st.caption(
        f"{event['race_date'].date()}"
        f"{' · sprint weekend' if event['has_sprint'] else ''}. "
        "Before qualifying there is no grid, so this is a **different model** "
        "from the post-qualifying one, fitted without grid position -- not the "
        "same model with less confidence. Read it as a prior."
    )
    stage = "pre_weekend"
    grid_note = ""

    fetched = st.session_state.get("fc_grid")
    if fetched is not None and fetched["round"] != event["round_number"]:
        fetched = None  # a grid for a race that has since been run

    st.caption(
        "After qualifying, the grid can be fetched from FastF1: one request, "
        "written nowhere. Penalties are not applied, so a penalised car is "
        "forecast from its qualifying slot."
    )
    if st.button("Fetch the qualifying grid (uses the network)", key="fc_fetch"):
        from src.models.predict import qualifying_grid

        with st.spinner("Loading qualifying ..."):
            try:
                grid = qualifying_grid(event["year"], event["round_number"])
            except Exception as exc:  # noqa: BLE001 - network and cache both fail here
                grid = None
                st.error(f"Could not load qualifying: {exc}")
        if grid is None:
            st.warning("Qualifying has not run, or could not be loaded.")
            st.session_state.pop("fc_grid", None)
        else:
            st.session_state["fc_grid"] = {
                "round": event["round_number"], "grid": grid,
            }
            fetched = st.session_state["fc_grid"]

    if fetched is not None:
        outlook = services.cached_post_quali(
            version,
            event=(event["year"], event["round_number"], event["event_name"],
                   event["race_date"]),
            grid=tuple(sorted(fetched["grid"].items())), trials=int(trials),
        )
        stage = "post_quali"
        grid_note = " · from the qualifying grid"

    (st.success if stage == "post_quali" else st.info)(
        f"Showing the **{stage}** model{grid_note}."
    )

    top = outlook.iloc[0]
    metric_row([
        ("Most likely winner", str(outlook.sort_values("p_win").iloc[-1]["Abbreviation"]),
         format_odds(outlook["p_win"].max())),
        ("Highest expected points", str(top["Abbreviation"]),
         f"{top['exp_points']:.1f} points"),
        ("Cars expected to retire", f"{outlook['p_dnf'].sum():.1f}",
         f"of {len(outlook)} entries"),
    ])

    view = outlook.copy()
    left, right = st.columns([2, 3])
    with left:
        fig = charts.ranked_bars(
            view, label="Abbreviation", value="exp_points", highlight=3,
            hover=("TeamName",), xtitle="expected points",
            height=max(360, 26 * len(view)),
        )
        charts.show(fig)
    with right:
        outlook_table(view)
    st.caption(
        "P(out) is the chance of not finishing *or* not starting, from the "
        "retirement model. A retired car scores nothing, so expected points is "
        "roughly P(finish) x the points its finishing order would earn."
    )

# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

with tab_back:
    dataset = services.load_dataset(services.data_version())
    races = services.race_index(services.data_version(), dataset)
    # The post-qualifying model wants a training window behind it; the first
    # seasons have too little history to fit honestly.
    races = races.loc[races["Year"] >= 2021].reset_index(drop=True)
    label = st.selectbox("Which completed race", races["label"], index=0, key="fc_bt")
    pick = races.loc[races["label"] == label].iloc[0]

    try:
        result = services.cached_backtest_race(
            version, year=int(pick["Year"]), round_number=int(pick["RoundNumber"]),
            trials=int(trials),
        )
    except Exception as exc:  # noqa: BLE001 - a fit failure is the answer here
        st.error(f"Could not backtest this race: {exc}")
        st.stop()

    st.caption(
        "Forecast from the real starting grid by a model fitted **only on "
        "earlier races**, so this is out-of-sample by construction -- unlike "
        "the retirement model's Race weekend page, there is no in-sample "
        "case to guard against."
    )

    winner = result.loc[result["finished"] == 1]
    podium = result.loc[result["finished"].between(1, 3)]
    metric_row([
        ("Winner", str(winner.iloc[0]["Abbreviation"]) if len(winner) else "—",
         f"forecast {format_odds(winner.iloc[0]['p_win'])}" if len(winner) else ""),
        ("Favourite", str(result.sort_values("p_win").iloc[-1]["Abbreviation"]),
         f"forecast {format_odds(result['p_win'].max())} to win"),
        ("Podium's mean P(podium)",
         fmt_pct(podium["p_podium"].mean()) if len(podium) else "—",
         "what the model gave the three who made it"),
    ])

    outlook_table(result)
    st.caption(
        "`result` is the classified position; blank means the car did not "
        "finish. One race is twenty rows of noise: read this view for whether "
        "the *shape* is sensible, and `References/points_model_results.md` "
        "for whether the model is calibrated over many."
    )
