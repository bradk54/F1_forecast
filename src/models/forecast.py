"""The finishing-order and championship model, from the command line.

    ./.venv/bin/python -m src.models.forecast evaluate   # held-out scores, 2024 onward
    ./.venv/bin/python -m src.models.forecast tune       # re-run selection and tuning
    ./.venv/bin/python -m src.models.forecast backtest   # season-sim calibration
    ./.venv/bin/python -m src.models.forecast season     # the championship forecast
    ./.venv/bin/python -m src.models.forecast next       # the next race, per driver
    ./.venv/bin/python -m src.models.forecast race 2026 14   # backtest one race

The sibling of :mod:`src.models.predict`, which runs the retirement model.  This
one runs the model that retirement model was always a component of: attrition
from the DNF model, then a stagewise Plackett-Luce over the survivors, then
points, then a season.

Everything below marked *measured* came out of :func:`src.models.tuning.run_protocol`
on the 2020-2023 development races, and was scored on 2024 onward only after it
was fixed.  ``References/points_model_results.md`` has the tables.  Re-run
``tune`` each winter before trusting any of it for a new season: the DNF work
found the best training window moved with the regulations, and nothing here is
exempt from that.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

import pandas as pd

from src import config
from src.features.order_features import OrderFeatureConfig, add_order_features
from src.models import order_eval, season, tuning
from src.models.order_eval import OrderSpec

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The measured choices
# --------------------------------------------------------------------------- #

#: Measured.  The pre-weekend model, which the season simulation runs on
#: because a race that has not been qualified has no grid.  Forward selection
#: from seventeen candidates took, in order: the team's race pace (lineage-keyed
#: EWMA), the driver's qualifying delta to the team-mate (removing it costs
#: 0.61 nats a race, t = 4.8), and the team's season-to-date qualifying pace
#: (0.31, t = 2.8).  Re-selecting under the tuned settings chose the same three.
PRE_WEEKEND_SPEC = OrderSpec(
    "pre_weekend",
    features=("team_finish_pct_ewma", "driver_mate_grid_delta_ewma",
              "team_grid_pct_season"),
    stage="pre_weekend", l2=0.1, lookback_races=100, truncate=10,
    race_halflife=40.0, n_scales=10, l2_scale=0.1,
)
#: Measured.  Two timescales: a four-race team half-life beside a season-long
#: mean, and a slow driver half-life, because driver skill moves more slowly
#: than a car under development.  Per-race team-mate deltas are clipped at half
#: a field, which trims a penalty or a qualifying accident without discarding
#: the head-to-head (-18.690 -> -18.682 on the development races; heavier clips
#: were worse).
#:
#: The slow driver half-life has a known cost, and it is live: it reaches back
#: into a rookie season.  Antonelli lost the 2025 qualifying head-to-head 3-21
#: and has won eight of fourteen 2026 races, so this model rates him below
#: Russell on pace.  A within-season driver term was tested for exactly this
#: and did not clear the bar before qualifying.  See the results document.
PRE_WEEKEND_FEATURES = OrderFeatureConfig(team_halflife=4.0, driver_halflife=24.0,
                                          mate_delta_clip=0.5)

#: Measured.  The Saturday model.  Grid position, then the team's points-rank in
#: the field (removing it costs 0.39 nats a race, t = 3.1) -- the car's
#: underlying pace, which corrects a grid slot distorted by a penalty or a
#: messy qualifying -- then the driver's recent qualifying form (0.21, t = 2.5).
#:
#: Greedy selection is path-dependent, and the third slot was a tie: with a
#: wider candidate pool, three driver terms came within 0.02 nats of each other
#: (season team-mate delta +0.221, this one +0.214, EWMA delta +0.205) and the
#: season delta was taken.  Fully tuned, this configuration scores -17.481 on
#: the development races against -17.521 for that one, so it is kept -- a tie
#: broken on development data, like every other choice here.
POST_QUALI_SPEC = OrderSpec(
    "post_quali",
    features=("grid_position", "team_rank_in_field", "driver_grid_pct_ewma"),
    stage="post_quali", l2=0.1, lookback_races=100, truncate=10,
    race_halflife=40.0, n_scales=10, l2_scale=0.1,
)
POST_QUALI_FEATURES = OrderFeatureConfig(team_halflife=8.0, driver_halflife=4.0)

#: Measured.  Tree parameters from 24 random-search draws on the development
#: races, for the gradient-boosted comparison (Method 3), fed every post-quali
#: candidate.  The winner is the most constrained configuration the search
#: offered -- two levels deep, fifty trees, a slow learning rate -- and it still
#: trails the tuned linear model by 0.43 nats a race on the same races (-17.91
#: against -17.48).  When the flexible family's best setting is the one closest
#: to linear, the linear model is the answer.  ``tune --ranker`` re-runs it.
BOOSTED_PARAMS: tuple[tuple[str, object], ...] = (
    ("colsample_bytree", 0.8), ("learning_rate", 0.02), ("max_depth", 2),
    ("min_child_weight", 20.0), ("n_estimators", 50), ("reg_lambda", 0.1),
    ("subsample", 0.8),
)

_UNTUNED = dict(l2=1.0, lookback_races=60, truncate=None, race_halflife=None,
                n_scales=0)


def ladder(stage: str) -> list[tuple[OrderSpec, OrderFeatureConfig]]:
    """Every model the held-out evaluation compares, cheapest first.

    Baselines first, then the design brief's rungs as plain untuned
    Plackett-Luce, then the chosen features with default settings (so the
    table separates what selection bought from what tuning bought), then the
    tuned model.  Each has to beat the one above it to justify itself.
    """
    base = OrderFeatureConfig()
    S = OrderSpec
    if stage == "pre_weekend":
        return [
            (S("uniform", stage=stage, deterministic="uniform"), base),
            (S("team order", stage=stage, deterministic="team"), base),
            (S("rung 0: team pace", ("team_finish_pct_ewma",), stage=stage, **_UNTUNED), base),
            (S("rung 1: + driver vs team-mate",
               ("team_finish_pct_ewma", "driver_mate_grid_delta_ewma"),
               stage=stage, **_UNTUNED), base),
            (PRE_WEEKEND_SPEC.with_(name="selected, untuned", **_UNTUNED), PRE_WEEKEND_FEATURES),
            (PRE_WEEKEND_SPEC.with_(name="selected + tuned"), PRE_WEEKEND_FEATURES),
        ]
    specs = [
        (S("grid order", deterministic="grid"), base),
        (S("grid only", ("grid_position",), **_UNTUNED), base),
        (S("rung 2: grid + team + driver",
           ("grid_position", "team_finish_pct_ewma", "driver_mate_grid_delta_ewma"),
           **_UNTUNED), base),
        (POST_QUALI_SPEC.with_(name="selected, untuned", **_UNTUNED), POST_QUALI_FEATURES),
        (POST_QUALI_SPEC.with_(name="selected + tuned"), POST_QUALI_FEATURES),
    ]
    if BOOSTED_PARAMS:
        features = tuple(c for c in tuning.POST_QUALI_CANDIDATES if isinstance(c, str))
        specs.append((POST_QUALI_SPEC.with_(
            name="boosted ranker (tuned)", features=features, family="boosted",
            params=BOOSTED_PARAMS), POST_QUALI_FEATURES))
    return specs


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """The modelling table, the raw results, and the circuit profiles."""
    for path in (config.DNF_DATASET_PATH, config.RACE_RESULTS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; build it with "
                "`python -m src.data.generate_dataset --skip-download`")
    profiles = (pd.read_parquet(config.CIRCUIT_PROFILE_PATH)
                if config.CIRCUIT_PROFILE_PATH.exists() else None)
    return (pd.read_parquet(config.DNF_DATASET_PATH),
            pd.read_parquet(config.RACE_RESULTS_PATH), profiles)


def evaluate_stage(
    dataset: pd.DataFrame, stage: str, *, start_after: str | None,
    end_before: str | None, n_samples: int = 4000,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Score the whole ladder for one stage, plus the attrition alternatives.

    The attrition rows answer whether the DNF model belongs in the forecast at
    all.  Four ways to handle a retirement, all on the tuned ordering model:
    the DNF model's per-car probability; the same attrition level for every
    car; no attrition; and no separate stage at all, with retirements ranked
    last inside Plackett-Luce -- which is what "leave the DNF model out" means.
    """
    dnf = order_eval.walk_forward_dnf(dataset, stage)
    frames: dict[OrderFeatureConfig, pd.DataFrame] = {}
    rows, scores = [], {}
    chosen = PRE_WEEKEND_SPEC if stage == "pre_weekend" else POST_QUALI_SPEC
    chosen_cfg = PRE_WEEKEND_FEATURES if stage == "pre_weekend" else POST_QUALI_FEATURES
    variants = [(spec, cfg, "model") for spec, cfg in ladder(stage)]
    variants += [
        (chosen.with_(name="tuned, flat attrition"), chosen_cfg, "flat"),
        (chosen.with_(name="tuned, no attrition"), chosen_cfg, "none"),
        (chosen.with_(name="tuned, retirements ranked last", ranking="all_starters"),
         chosen_cfg, "none"),
    ]
    for spec, cfg, attrition in variants:
        if cfg not in frames:
            frames[cfg] = add_order_features(dataset, cfg)
        table, sc = order_eval.evaluate(
            frames[cfg], [spec], dnf, attrition=attrition,
            start_after=start_after, end_before=end_before, n_samples=n_samples)
        rows.append(table)
        scores.update(sc)
    return pd.concat(rows), scores


def cmd_evaluate(args) -> int:
    dataset, _, _ = load()
    columns = ["races", "ll10_gain", "rps10", "points_rmse", "spearman",
               "brier_win", "brier_podium", "brier_points",
               "calib_win", "calib_podium", "calib_points"]
    out = []
    for stage in args.stages:
        table, scores = evaluate_stage(dataset, stage, start_after=args.start,
                                       end_before=args.end, n_samples=args.samples)
        print(f"\n=== {stage}: races after {args.start}"
              f"{'' if args.end is None else ' and before ' + args.end} ===")
        print(table[columns].round(4).to_string())
        reference = "grid order" if stage == "post_quali" else "team order"
        for name in ("selected + tuned", "tuned, flat attrition",
                     "tuned, retirements ranked last"):
            for against in (reference, "selected + tuned"):
                if name == against or name not in scores:
                    continue
                ci = order_eval.paired_race_bootstrap(scores[name], scores[against])
                print(f"  rps10 {name!r} - {against!r}: {ci['diff']:+.4f} "
                      f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
        out.append(table.assign(stage=stage).reset_index())
    if args.write:
        path = config.REPORTS_DIR / "points_model_evaluation.csv"
        pd.concat(out).round(5).to_csv(path, index=False)
        print(f"\nwrote {path}")
    return 0


def cmd_tune(args) -> int:
    dataset, _, _ = load()
    build = lambda cfg: add_order_features(dataset, cfg)  # noqa: E731
    for stage in args.stages:
        result = tuning.run_protocol(build, stage)
        print(f"\n=== {stage} ===\n{result.spec}\n{result.feature_config}")
        if args.write:
            base = config.REPORTS_DIR / f"points_model_tuning_{stage}"
            result.selection.to_csv(f"{base}_selection.csv", index=False)
            result.search.to_csv(f"{base}_search.csv", index=False)
    if args.ranker:
        frame = build(POST_QUALI_FEATURES)
        features = tuple(c for c in tuning.POST_QUALI_CANDIDATES if isinstance(c, str))
        best, table = tuning.random_search_ranker(frame, POST_QUALI_SPEC.with_(
            name="boosted", features=features, family="boosted"))
        print("\nboosted ranker:", dict(best.params))
        print(table.head(10).round(4).to_string(index=False))
    return 0


def _season_setup(dataset, results, profiles, year, through_round, offline=True):
    races = season.remaining_races(results, year, through_round)
    if races.empty:
        races = season.remaining_races_from_schedule(year, through_round, offline=offline)
    return season.prepare_season(
        dataset, results, profiles, year=year, through_round=through_round,
        races=races, spec=PRE_WEEKEND_SPEC, feature_config=PRE_WEEKEND_FEATURES)


def cmd_season(args) -> int:
    dataset, results, profiles = load()
    year = args.year or int(dataset["Year"].max())
    through = args.through_round or int(dataset.loc[dataset["Year"] == year, "RoundNumber"].max())
    setup = _season_setup(dataset, results, profiles, year, through, offline=not args.online)
    noise = season.SeasonNoise(args.team_sd, args.driver_sd)
    draws = season.simulate(setup, n_trials=args.trials, noise=noise)
    drivers, teams = season.summarise(draws)
    left = ", ".join(f"R{r.RoundNumber} {r.EventName.replace(' Grand Prix', '')}"
                     f"{' (sprint)' if r.has_sprint else ''}"
                     for r in setup.races.itertuples())
    print(f"{year} after round {through}; {len(setup.races)} to run: {left}")
    print(f"noise: {noise}; {args.trials} trials; parameter draws: {setup.strength.shape[0]}")
    fmt = {"points_so_far": "{:.0f}", "mean": "{:.1f}", "p10": "{:.0f}", "p50": "{:.0f}",
           "p90": "{:.0f}", "p_champion": "{:.3f}", "p_top3": "{:.3f}"}
    for title, table in (("drivers", drivers), ("constructors", teams)):
        shown = table.copy()
        for col, f in fmt.items():
            shown[col] = shown[col].map(f.format)
        print(f"\n--- {title} ---\n{shown.to_string(index=False)}")
    if args.write:
        path = config.REPORTS_DIR / f"season_forecast_{year}.csv"
        pd.concat([drivers.assign(kind="driver"), teams.assign(kind="team")]).round(4).to_csv(
            path, index=False)
        print(f"\nwrote {path}")
    return 0


def _print_outlook(title: str, table: pd.DataFrame) -> None:
    print(title)
    shown = table.copy()
    for col in ("p_win", "p_podium", "p_points", "p_dnf"):
        shown[col] = shown[col].map("{:.3f}".format)
    shown["exp_points"] = shown["exp_points"].map("{:.2f}".format)
    for col in ("grid", "finished"):
        if col in shown.columns:
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    print(shown.to_string(index=False))


def cmd_next(args) -> int:
    dataset, results, profiles = load()
    year = int(dataset["Year"].max())
    through = int(dataset.loc[dataset["Year"] == year, "RoundNumber"].max())
    setup = _season_setup(dataset, results, profiles, year, through, offline=not args.online)
    race = setup.races.iloc[0]
    if args.stage == "pre_weekend":
        _print_outlook(
            f"{year} R{race.RoundNumber} {race.EventName} -- pre-weekend forecast "
            f"(no grid; run with --stage post_quali after qualifying)",
            season.race_outlook(setup, n_samples=args.trials))
        return 0

    from src.models import predict

    grid = predict.qualifying_grid(year, int(race.RoundNumber))
    if grid is None:
        print(f"error: qualifying for {year} R{race.RoundNumber} has not run, or could "
              "not be loaded, so there is no grid.\n  Before qualifying, use the "
              "default --stage pre_weekend: a model fitted without the grid.",
              file=sys.stderr)
        return 3
    table = season.saturday_outlook(
        dataset, results, profiles, year=year, round_number=int(race.RoundNumber),
        race_date=race.RaceDate, event_name=race.EventName, grid=grid,
        spec=POST_QUALI_SPEC, feature_config=POST_QUALI_FEATURES, n_samples=args.trials)
    _print_outlook(f"{year} R{race.RoundNumber} {race.EventName} -- post-qualifying "
                   "forecast (grid = qualifying order; penalties not yet applied)", table)
    return 0


def cmd_race(args) -> int:
    """A completed race, forecast from its real grid by a model that never saw it."""
    dataset, results, profiles = load()
    rows = dataset.loc[(dataset["Year"] == args.year) & (dataset["RoundNumber"] == args.round)]
    if rows.empty:
        print(f"error: {args.year} R{args.round} is not in the dataset; for an "
              "upcoming race use `next`", file=sys.stderr)
        return 1
    grid = dict(zip(rows["DriverId"], rows["grid_position"]))
    table = season.saturday_outlook(
        dataset, results, profiles, year=args.year, round_number=args.round,
        race_date=rows[order_eval.ORDER_COL].iloc[0], event_name=rows["EventName"].iloc[0],
        grid=grid, spec=POST_QUALI_SPEC, feature_config=POST_QUALI_FEATURES,
        n_samples=args.trials)
    finished = pd.to_numeric(rows.set_index("Abbreviation")["ClassifiedPosition"],
                             errors="coerce")
    table["finished"] = table["Abbreviation"].map(finished)
    _print_outlook(f"{args.year} R{args.round} {rows['EventName'].iloc[0]} -- backtest: "
                   "fitted only on earlier races, scored against what happened "
                   "(blank = did not finish)", table)
    return 0


#: Where the season simulation is backtested: every completed season with a
#: full training history, forecast from after rounds 6, 12 and 18 -- early,
#: middle and late, when the question is most and least open.
BACKTEST_SEASONS = (2021, 2022, 2023, 2024, 2025)
BACKTEST_CUTOFFS = (6, 12, 18)


def backtest(
    dataset, results, profiles, noises: Sequence[season.SeasonNoise], *,
    seasons=BACKTEST_SEASONS, cutoffs=BACKTEST_CUTOFFS, n_trials: int = 4000,
) -> pd.DataFrame:
    """Forecast completed seasons from part-way through, under each noise setting.

    Setups are built once per cutoff and reused across noise settings, since
    only the simulation depends on the noise.
    """
    scored = []
    for year in seasons:
        last = int(results.loc[results["Year"] == year, "RoundNumber"].max())
        for cut in cutoffs:
            if cut >= last:
                continue
            setup = _season_setup(dataset, results, profiles, year, cut)
            for noise in noises:
                draws = season.simulate(setup, n_trials=n_trials, noise=noise)
                scored.append(season.score_season(draws, results).assign(
                    team_sd=noise.team_sd, driver_sd=noise.driver_sd))
    return pd.concat(scored, ignore_index=True)


def cmd_backtest(args) -> int:
    dataset, results, profiles = load()
    noises = [season.SeasonNoise(t, d) for t in args.team_sd for d in args.driver_sd]
    scored = backtest(dataset, results, profiles, noises, seasons=args.seasons,
                      n_trials=args.trials)
    summary = (scored.groupby(["team_sd", "driver_sd", "kind"])
               .apply(season.calibration_summary, include_groups=False))
    print(summary.round(3).to_string())
    if args.write:
        path = config.REPORTS_DIR / "season_backtest.csv"
        scored.round(4).to_csv(path, index=False)
        print(f"\nwrote {path}")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.models.forecast",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    stages = dict(nargs="+", default=["pre_weekend", "post_quali"],
                  choices=["pre_weekend", "post_quali"])

    p = sub.add_parser("evaluate", help="score every model on held-out races")
    p.add_argument("--stages", **stages)
    p.add_argument("--start", default=order_eval.TEST_START,
                   help="score races after this date (default: the test period)")
    p.add_argument("--end", default=None)
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--write", action="store_true", help="save the table to Reports/")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("tune", help="re-run feature selection and tuning on dev races")
    p.add_argument("--stages", **stages)
    p.add_argument("--ranker", action="store_true", help="also search the boosted ranker")
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("season", help="simulate the rest of a season")
    p.add_argument("--year", type=int)
    p.add_argument("--through-round", type=int)
    p.add_argument("--trials", type=int, default=season.DEFAULT_TRIALS)
    p.add_argument("--team-sd", type=float, default=0.0)
    p.add_argument("--driver-sd", type=float, default=0.0)
    p.add_argument("--online", action="store_true",
                   help="allow a network lookup for the calendar")
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_season)

    p = sub.add_parser("next", help="forecast the next race, per driver")
    p.add_argument("--stage", choices=["pre_weekend", "post_quali"], default="pre_weekend",
                   help="post_quali fetches the qualifying grid (one API call)")
    p.add_argument("--trials", type=int, default=season.DEFAULT_TRIALS)
    p.add_argument("--online", action="store_true",
                   help="allow a network lookup for the calendar")
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("race", help="backtest one completed race from its real grid")
    p.add_argument("year", type=int)
    p.add_argument("round", type=int)
    p.add_argument("--trials", type=int, default=season.DEFAULT_TRIALS)
    p.set_defaults(func=cmd_race)

    p = sub.add_parser("backtest", help="season-sim calibration on completed seasons")
    p.add_argument("--seasons", type=int, nargs="+", default=list(BACKTEST_SEASONS))
    p.add_argument("--team-sd", type=float, nargs="+", default=[0.0])
    p.add_argument("--driver-sd", type=float, nargs="+", default=[0.0])
    p.add_argument("--trials", type=int, default=4000)
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
