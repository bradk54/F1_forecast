"""Fit, refresh and run the retirement model from a terminal.

Four subcommands, in the order you would use them::

    python -m src.models.predict refresh          # weekly: score, refit, save
    python -m src.models.predict next             # rank the upcoming race
    python -m src.models.predict race 2026 14     # rank a specific race
    python -m src.models.predict status           # how it has been doing

**Predicting a race that has not happened.**  The dataset only contains
completed races, so there is no feature row for an upcoming event.  One is
built by appending placeholder rows for the entry list to the historical frame,
running the ordinary feature builders, and taking those rows back out.

That is safe for one specific reason: every feature in this pipeline is
strictly prior-race.  A placeholder's own outcome is shifted out of its own
features, and there is no later race for it to contaminate.  The leakage
discipline that makes the training numbers trustworthy is the same property
that makes forward inference fall out for free.

**Stage matters, and it is the question people get wrong.**  ``pre_weekend``
excludes grid position; ``post_quali`` includes it, along with the team-mate
grid delta.  Grid position is the single most important feature in the model,
so running before qualifying and running after it are genuinely different
predictions, not the same prediction with more confidence.  See
``References/model_runbook.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from src import config
from src.features import registry
from src.features.build_features import ORDER_COL, build_history_features
from src.features.labels import FINISHED
from src.models import monitor, store
from src.models.train import (
    DEFAULT_LOOKBACK_RACES,
    MODEL_FACTORIES,
    TARGET,
    _select_features,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "random_forest"
DEFAULT_STAGE: registry.Stage = "post_quali"


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def training_window(
    dataset: pd.DataFrame, lookback_races: int | None = DEFAULT_LOOKBACK_RACES
) -> pd.DataFrame:
    """The most recent ``lookback_races`` races, or everything if None."""
    if lookback_races is None:
        return dataset
    races = (
        dataset[["Year", "RoundNumber", ORDER_COL]]
        .drop_duplicates()
        .sort_values(ORDER_COL)
        .tail(lookback_races)
    )
    return dataset.merge(races[["Year", "RoundNumber"]], on=["Year", "RoundNumber"])


def fit_current(
    dataset: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    stage: registry.Stage = DEFAULT_STAGE,
    lookback_races: int | None = DEFAULT_LOOKBACK_RACES,
    target: str = TARGET,
) -> tuple[object, store.Manifest, list[str]]:
    """Fit on the most recent races and build the manifest that describes it."""
    train = training_window(dataset, lookback_races)
    if train.empty:
        raise ValueError("no training rows in the selected window")
    selected, numeric, categorical = _select_features(train, stage, (), ())
    estimator = MODEL_FACTORIES[model](numeric, categorical)
    estimator.fit(train[selected], train[target])

    year, rnd, event = store.latest_race(train)
    manifest = store.Manifest(
        model=model,
        stage=stage,
        lookback_races=lookback_races,
        train_rows=len(train),
        train_base_rate=float(train[target].mean()),
        trained_through_year=year,
        trained_through_round=rnd,
        trained_through_event=event,
        features=list(selected),
        dataset_rows=len(dataset),
        dataset_fingerprint=store.dataset_fingerprint(dataset),
    )
    return estimator, manifest, selected


# --------------------------------------------------------------------------- #
# Building a row for a race that has not run
# --------------------------------------------------------------------------- #


def entry_list(results: pd.DataFrame) -> pd.DataFrame:
    """Who is racing, taken from the most recent completed round.

    An assumption, and a documented one: the grid is treated as unchanged since
    last time out.  Reserve drivers and mid-season swaps break it, and there is
    nothing in the data that can tell you they have.
    """
    races = results.loc[results.get("session_type", "R") == "R"]
    year, rnd, _ = store.latest_race(races)
    last = races.loc[(races["Year"] == year) & (races["RoundNumber"] == rnd)]
    return last[["DriverId", "TeamId"]].drop_duplicates().reset_index(drop=True)


def circuit_for_event(results: pd.DataFrame, event_name: str) -> float | None:
    """The circuit key this event used last time it ran.

    Without it the track-geometry features are all NaN, which the model tolerates
    but should not have to.  Matching on event name is crude and survives a
    circuit being renamed only by accident, so a miss returns None rather than
    guessing.
    """
    prior = results.loc[results["EventName"] == event_name]
    if prior.empty or "circuit_key" not in prior.columns:
        return None
    keys = prior.sort_values(ORDER_COL)["circuit_key"].dropna()
    return float(keys.iloc[-1]) if not keys.empty else None


def build_inference_rows(
    results: pd.DataFrame,
    profiles: pd.DataFrame | None,
    *,
    year: int,
    round_number: int,
    race_date,
    event_name: str,
    entries: pd.DataFrame,
    grid: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Feature rows for a race that has not been run.

    Placeholder rows for the entry list are appended to the **raw results** and
    the ordinary dataset build runs over the whole thing, so the upcoming race
    goes through exactly the same code path as a historical one -- same rolling
    features, same circuit-profile join, same leakage discipline.

    That is safe because every feature is strictly prior-race: a placeholder's
    own outcome is shifted out of its own features, and there is no later race
    for it to contaminate.

    Note the input is ``race_results.parquet``, not ``dnf_dataset.parquet``.
    Feeding the built dataset back in silently produces ``_x``/``_y`` suffixed
    columns, because every builder merges features into a frame that already has
    them, and the model then finds none of the names it was fitted on.
    """
    from src.data.generate_dataset import build_dataset

    placeholder = entries.copy()
    placeholder["Year"] = int(year)
    placeholder["RoundNumber"] = int(round_number)
    placeholder[ORDER_COL] = pd.Timestamp(race_date)
    placeholder["EventName"] = event_name
    placeholder["session_type"] = "R"
    # Outcome columns the labeller and builders require.  Never read for these
    # rows: everything downstream of them is shifted by one race.
    placeholder["Status"] = "Finished"
    placeholder["ClassifiedPosition"] = "1"
    for column, value in (
        ("Points", 0.0), ("Position", np.nan), ("Laps", np.nan),
    ):
        if column in results.columns:
            placeholder[column] = value
    key = circuit_for_event(results, event_name)
    if key is not None:
        placeholder["circuit_key"] = key
    placeholder["GridPosition"] = (
        placeholder["DriverId"].map(grid) if grid else np.nan
    )

    # Asking about a race that has already run is a backtest, and its real rows
    # have to come out first: leaving them in duplicates the driver-race key,
    # which breaks the team-mate join, and any that survived would let the race
    # inform its own features.
    history = results.loc[
        ~((results["Year"] == int(year)) & (results["RoundNumber"] == int(round_number)))
    ]
    combined = pd.concat([history, placeholder], ignore_index=True)
    built, _ = build_dataset(combined, profiles, run_checks=False)
    rows = built.loc[
        (built["Year"] == int(year)) & (built["RoundNumber"] == int(round_number))
    ]
    if rows.empty:
        raise ValueError(
            f"no inference rows produced for {year} R{round_number}; the entry "
            f"list may be empty or the event may already be in the results"
        )
    return rows.reset_index(drop=True)


def next_event(dataset: pd.DataFrame, *, year: int | None = None) -> dict | None:
    """The first scheduled event after the last one in the dataset.

    Reads the calendar from FastF1, so it needs either network access or a warm
    cache.  Returns None when the season is complete.
    """
    import fastf1

    from src.data import ingest

    ingest.configure()
    last_year, last_round, _ = store.latest_race(dataset)
    season = year or last_year
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    upcoming = schedule.loc[schedule["RoundNumber"] > last_round]
    if upcoming.empty:
        return None
    event = upcoming.iloc[0]
    return {
        "year": int(season),
        "round_number": int(event["RoundNumber"]),
        "event_name": str(event["EventName"]),
        "race_date": pd.Timestamp(event["EventDate"]),
    }


def qualifying_grid(year: int, round_number: int) -> dict[str, float] | None:
    """Grid positions from the event's qualifying session, if it has run."""
    from src.data import ingest

    try:
        session = ingest.load_session(
            year, round_number, "Q", laps=False, telemetry=False,
            weather=False, retries=1,
        )
        results = pd.DataFrame(session.results)
        if results.empty or "Position" not in results.columns:
            return None
        grid = {
            str(row["DriverId"]): float(row["Position"])
            for _, row in results.iterrows()
            if pd.notna(row.get("DriverId")) and pd.notna(row.get("Position"))
        }
        return grid or None
    except Exception as exc:  # noqa: BLE001 - qualifying may not have run
        log.info("no qualifying grid for %s R%s: %s", year, round_number, exc)
        return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render(
    rows: pd.DataFrame,
    manifest: store.Manifest,
    header: str,
    *,
    show_grid: bool = True,
) -> str:
    """One line per driver, riskiest first.

    ``show_grid`` is False when the grid was never known: ``add_race_context``
    imputes a missing slot to the back of the field, so printing that number
    would show a real-looking grid position that came from nowhere.
    """
    ordered = rows.sort_values("predicted", ascending=False).reset_index(drop=True)
    lines = [header, f"  model: {manifest.describe()}", ""]
    grid_head = f"{'grid':>6}" if show_grid else f"{'grid':>6}"
    lines.append(f"  {'#':>2}  {'driver':<22}{'team':<20}{grid_head}{'P(dnf)':>9}")
    for i, row in ordered.iterrows():
        grid = row.get("grid_position", np.nan)
        grid_text = (
            f"{int(grid):>6}" if show_grid and pd.notna(grid) else f"{'n/a':>6}"
        )
        lines.append(
            f"  {i + 1:>2}  {str(row['DriverId'])[:21]:<22}"
            f"{str(row.get('TeamId', ''))[:19]:<20}{grid_text}"
            f"{row['predicted']:>9.3f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def load_dataset() -> pd.DataFrame:
    """The built modelling table: what the model is fitted and scored on."""
    if not config.DNF_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"no dataset at {config.DNF_DATASET_PATH}; build it with "
            f"'python -m src.data.generate_dataset'"
        )
    frame = pd.read_parquet(config.DNF_DATASET_PATH)
    frame[ORDER_COL] = pd.to_datetime(frame[ORDER_COL])
    return frame


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Raw results and circuit profiles: what an upcoming race is built from."""
    if not config.RACE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"no results at {config.RACE_RESULTS_PATH}; build them with "
            f"'python -m src.data.generate_dataset'"
        )
    results = pd.read_parquet(config.RACE_RESULTS_PATH)
    results[ORDER_COL] = pd.to_datetime(results[ORDER_COL])
    profiles = (
        pd.read_parquet(config.CIRCUIT_PROFILE_PATH)
        if config.CIRCUIT_PROFILE_PATH.exists()
        else None
    )
    return results, profiles


def cmd_refresh(args) -> int:
    """Score the outstanding prediction, then refit on the latest data."""
    if not args.no_download:
        from src.data import generate_dataset

        seasons = f"{config.FIRST_TELEMETRY_SEASON}-{config.LATEST_SEASON - 1}"
        argv = ["--seasons", seasons]
        if args.offline:
            argv.append("--offline")
        code = generate_dataset.main(argv)
        if code != 0:
            print(f"dataset build failed with exit code {code}; not refitting")
            return code

    dataset = load_dataset()

    # Score first, refit second.  A model scored against data it has just been
    # trained on is not out-of-sample, and this ordering is the whole reason the
    # log can be trusted.
    pending = store.load_pending()
    if pending is not None:
        frame, old = pending
        actual = dataset.loc[
            (dataset["Year"] == frame["Year"].iloc[0])
            & (dataset["RoundNumber"] == frame["RoundNumber"].iloc[0])
        ]
        if actual.empty:
            print(
                f"pending prediction for {frame['Year'].iloc[0]} "
                f"R{frame['RoundNumber'].iloc[0]} is not classified yet; keeping it"
            )
        else:
            merged = actual[["DriverId", TARGET]].merge(
                frame[["DriverId", "predicted"]], on="DriverId", how="inner"
            )
            if merged.empty:
                print("pending prediction shares no drivers with the result; dropping")
                store.clear_pending()
            else:
                row = monitor.score_race(
                    merged[TARGET], merged["predicted"],
                    year=int(actual["Year"].iloc[0]),
                    round_number=int(actual["RoundNumber"].iloc[0]),
                    event=str(actual["EventName"].iloc[0]),
                    race_date=actual[ORDER_COL].iloc[0],
                    stage=old.stage, model=old.model, git_sha=old.git_sha,
                    train_rows=old.train_rows,
                    train_base_rate=old.train_base_rate,
                    lookback_races=old.lookback_races,
                )
                monitor.append_row(row)
                print(
                    f"scored {row['year']} R{row['round']} {row['event']}: "
                    f"observed {row['observed_rate']:.3f}, "
                    f"brier skill {row['brier_skill']:+.4f}, "
                    f"top-2 caught {row['top2_hits']}/2"
                )
                store.clear_pending()

    estimator, manifest, _ = fit_current(
        dataset, model=args.model, stage=args.stage,
        lookback_races=args.lookback,
    )
    model_path, manifest_path = store.save_model(estimator, manifest)
    print(f"\nfitted: {manifest.describe()}")
    print(f"saved:  {model_path.relative_to(config.REPO_ROOT)}")
    print(f"        {manifest_path.relative_to(config.REPO_ROOT)}")

    ok, message = monitor.drift_check(dataset, manifest.train_base_rate)
    print(f"drift:  {message}" if ok else f"\n{message}")
    return 0


def _predict_event(args, event: dict) -> int:
    dataset = load_dataset()
    expected = registry.feature_columns(args.stage, available=dataset.columns)
    estimator, manifest = store.load_model(
        dataset, expected_features=expected, expected_stage=args.stage
    )

    grid = None
    if args.stage == "post_quali" and not args.no_grid:
        grid = qualifying_grid(event["year"], event["round_number"])
        if grid is None and not args.allow_missing_grid:
            # Refusing is the safe default, and the reason is specific: a
            # missing grid does not stay missing.  add_race_context reads a
            # non-positive grid slot as a pit-lane start and moves it to the
            # back of the field, so every driver would be scored as a
            # back-marker and every probability would come out inflated --
            # roughly 0.20-0.46 against a 14% base rate, which looks like a
            # model with an opinion rather than one with no information.
            print(
                f"error: qualifying for {event['year']} R{event['round_number']} "
                f"has not run, or could not be loaded, so grid position is "
                f"unknown.\n\n"
                f"  Grid position is the model's strongest feature, and a "
                f"missing one is imputed to the back of the grid rather than\n"
                f"  left absent, which inflates every prediction. Choose one:\n\n"
                f"    --stage pre_weekend      a model fitted without grid "
                f"position at all (correct before qualifying)\n"
                f"    --allow-missing-grid     proceed anyway, knowing every "
                f"driver is treated as starting last\n"
            )
            return 3

    results, profiles = load_raw()
    rows = build_inference_rows(
        results, profiles,
        year=event["year"], round_number=event["round_number"],
        race_date=event["race_date"], event_name=event["event_name"],
        entries=entry_list(results), grid=grid,
    )
    missing = [c for c in manifest.features if c not in rows.columns]
    if missing:
        raise store.SchemaDriftError(
            f"{len(missing)} feature(s) the model was fitted on were not "
            f"produced for this race ({missing[:5]}...). Refit with "
            "'python -m src.models.predict refresh'."
        )
    rows["predicted"] = estimator.predict_proba(rows[manifest.features])[:, 1]

    already_run = (
        (results["Year"] == event["year"])
        & (results["RoundNumber"] == event["round_number"])
    ).any()
    if already_run:
        trained_through = (
            manifest.trained_through_year,
            manifest.trained_through_round,
        )
        note = "backtest: this race has already run"
        if (event["year"], event["round_number"]) <= trained_through:
            note += (
                " and the model was trained through it, so these numbers are "
                "IN-sample and flatter the model"
            )
        print(f"! {note}.\n")

    header = (
        f"{event['year']} R{event['round_number']} {event['event_name']} "
        f"— {args.stage}"
    )
    print(render(rows, manifest, header, show_grid=grid is not None))

    ok, message = monitor.drift_check(dataset, manifest.train_base_rate)
    print(f"\n  base rate {manifest.train_base_rate:.1%} · "
          + (message if ok else message))

    if not args.no_save:
        keep = ["Year", "RoundNumber", "DriverId", "TeamId", "predicted"]
        store.save_pending(rows[keep], manifest)
        print(f"  prediction saved for scoring at the next refresh")
    return 0


def cmd_next(args) -> int:
    dataset = load_dataset()
    event = next_event(dataset)
    if event is None:
        print("no scheduled event after the last one in the dataset")
        return 1
    return _predict_event(args, event)


def cmd_race(args) -> int:
    dataset = load_dataset()
    import fastf1

    from src.data import ingest

    ingest.configure()
    schedule = fastf1.get_event_schedule(args.year, include_testing=False)
    match = schedule.loc[schedule["RoundNumber"] == args.round]
    if match.empty:
        print(f"{args.year} has no round {args.round}")
        return 1
    event = match.iloc[0]
    return _predict_event(args, {
        "year": args.year, "round_number": args.round,
        "event_name": str(event["EventName"]),
        "race_date": pd.Timestamp(event["EventDate"]),
    })


def cmd_status(args) -> int:
    frame = monitor.read_log()
    if frame.empty:
        print("no scored races yet; the log fills as refresh runs after each race")
        try:
            print(f"\ncurrent model: {store.read_manifest().describe()}")
        except store.ModelStoreError as exc:
            print(f"\n{exc}")
        return 0
    print("=== recent races ===")
    columns = ["year", "round", "event", "n", "observed_rate", "mean_predicted",
               "brier_skill", "roc_auc", "top2_hits", "top2_lift"]
    print(frame.tail(args.window)[columns].to_string(index=False))
    print(f"\n=== rolling over last {args.window} ===")
    print(monitor.rolling_summary(args.window).to_string())
    try:
        print(f"\ncurrent model: {store.read_manifest().describe()}")
    except store.ModelStoreError as exc:
        print(f"\n{exc}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit, refresh and run the retirement model.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stage", default=DEFAULT_STAGE,
                        choices=["pre_weekend", "post_quali"],
                        help="Information available to the model.")
    common.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_FACTORIES))

    p_refresh = sub.add_parser("refresh", parents=[common],
                               help="Score the outstanding prediction, then refit.")
    p_refresh.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_RACES,
                           help="Races of history to train on. 0 means all.")
    p_refresh.add_argument("--no-download", action="store_true",
                           help="Skip the data pull and use the dataset as it is.")
    p_refresh.add_argument("--offline", action="store_true",
                           help="Rebuild from the FastF1 cache without network.")
    p_refresh.set_defaults(func=cmd_refresh)

    p_next = sub.add_parser("next", parents=[common], help="Rank the upcoming race.")
    p_next.add_argument("--no-grid", action="store_true",
                        help="Do not look up qualifying, even at post_quali.")
    p_next.add_argument("--allow-missing-grid", action="store_true",
                        help="Predict at post_quali without a grid, treating "
                             "every driver as starting last.")
    p_next.add_argument("--no-save", action="store_true",
                        help="Do not record this prediction for later scoring.")
    p_next.set_defaults(func=cmd_next)

    p_race = sub.add_parser("race", parents=[common], help="Rank a specific race.")
    p_race.add_argument("year", type=int)
    p_race.add_argument("round", type=int)
    p_race.add_argument("--no-grid", action="store_true")
    p_race.add_argument("--allow-missing-grid", action="store_true")
    p_race.add_argument("--no-save", action="store_true")
    p_race.set_defaults(func=cmd_race)

    p_status = sub.add_parser("status", help="Recent performance from the log.")
    p_status.add_argument("--window", type=int, default=10)
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    if getattr(args, "lookback", None) == 0:
        args.lookback = None
    try:
        return args.func(args)
    except store.StageMismatchError as exc:
        print(f"error: {exc}")
        return 4
    except (store.ModelStoreError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
