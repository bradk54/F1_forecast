"""The championship as a loop over the race model.

Once a race is a sampler, a season is a loop over it: start every trial from the
real standings, draw every remaining race whole -- attrition, then an order
among the survivors, then points -- and read the championship off the final
totals of ten thousand trials.

Four decisions shape the answer, and each is a way of being confidently wrong
if it is made carelessly:

**Future races are forecast from the pre-weekend model.**  No grid exists for a
race that has not been qualified, so the strengths come from the model fitted at
``pre_weekend`` stage, and every remaining race uses the features as they stand
after the last completed round.  That freezes the running order at today's
reading, which is a claim, not a fact -- see the next point.

**The per-race draws must be correlated.**  Independent races understate the
spread of a season badly: a car that is fast in Baku is fast in Singapore, and
an upgrade that works lifts every race after it.  :func:`draw_trial_offsets`
decides how pace may move within a trial.  Its calibration is measured, not
assumed, by :func:`backtest` on completed seasons.

**Parameter uncertainty is not outcome noise.**  Each trial draws its own fitted
model from a Bayesian bootstrap over training races, so the forecast answers
"given what the data can tell us about the coefficients" rather than "given
that they are exactly right".

**The rules are the season's rules.**  Points maps change: fastest-lap bonus
2019-2024 only, sprint scoring changed in 2022.  See
:mod:`src.models.points_table`.

Attrition comes from the existing DNF model at its pre-weekend configuration,
plus the non-start rate the modelling table cannot see (it holds starters
only).  Sprints retire cars about half as often as grands prix (6.3% against
13.5% since 2021), so their attrition is scaled by :data:`SPRINT_ATTRITION_RATIO`.

Assumptions carried over from the DNF runbook, and just as breakable: the entry
list is the last race's entry list, and a future race uses the circuit its event
name used last time.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.features.build_features import ORDER_COL, RACE_KEYS
from src.features.labels import add_race_outcome_labels
from src.features.order_features import OrderFeatureConfig, add_order_features
from src.models import predict
from src.models.order_eval import OrderSpec, fit_order_model
from src.models.points_table import (
    grand_prix_points,
    has_fastest_lap_point,
    sprint_points,
)
from src.models.ranking import POINTS_POSITIONS, sample_positions

log = logging.getLogger(__name__)

#: Sprint retirement rate relative to a grand prix, 2021-2026 (6.3% / 13.5%).
SPRINT_ATTRITION_RATIO = 0.47
#: Bootstrap refits drawn to carry parameter uncertainty into the trials.
DEFAULT_BOOTSTRAP = 40
DEFAULT_TRIALS = 10_000


# --------------------------------------------------------------------------- #
# How pace may move within a trial
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeasonNoise:
    """The scale of pace movement allowed within one simulated season.

    Units are log-strength -- the same units as the model's ``beta . x``.  For
    a sense of scale, the pre-weekend model fitted after 2026 R14 spreads the
    22-car field over about 9 units, fastest to slowest, and puts the front
    seven cars within 1.6 of each other.  The bootstrap already moves a car's
    strength by about 0.23 from trial to trial; these offsets are on top of it.
    """

    team_sd: float = 0.0
    driver_sd: float = 0.0


def draw_trial_offsets(
    rng: np.random.Generator,
    n_trials: int,
    n_races: int,
    team_index: np.ndarray,
    noise: SeasonNoise,
) -> np.ndarray:
    """Pace offsets for every trial, race and driver: ``(trials, races, drivers)``.

    Added to each car's log-strength before the race is drawn.  Returning zeros
    means every remaining race is an independent draw around today's running
    order -- the assumption :func:`backtest` exists to test.

    Args:
        team_index: ``(drivers,)`` index of each driver's team, so team-mates
            can share a draw.
        noise: The scales to draw at.
    """
    n_teams = int(team_index.max()) + 1
    team_step = rng.normal(0.0, noise.team_sd, size=(n_trials, n_races, n_teams))
    # A random walk, not one fixed shift: cumsum lets an upgrade compound, so a
    # race in December carries more spread than next week's, rather than every
    # remaining race sharing a single draw.  Noise 0 gives steps of 0, so the
    # cumsum is 0 too -- the zero-noise contract holds without a special case.
    team_offset = np.cumsum(team_step, axis=1)
    # Broadcast each team's walk out to its cars: team_index[i] names driver
    # i's team, so indexing the last axis with it duplicates that team's
    # column onto every driver who races for it -- team-mates move together.
    #
    # noise.driver_sd is not applied yet -- team movement first; an
    # independent per-driver term on top of it is the natural next step.
    return team_offset[..., team_index]


# --------------------------------------------------------------------------- #
# Setting up a season from a point in time
# --------------------------------------------------------------------------- #


@dataclass
class SeasonSetup:
    """Everything a simulation needs, frozen at the end of ``through_round``."""

    year: int
    through_round: int
    drivers: pd.DataFrame  # one row per simulated entrant, with points_so_far
    others: pd.DataFrame  # drivers with points this season who no longer race
    teams: pd.DataFrame  # one row per constructor, with points_so_far
    races: pd.DataFrame  # remaining rounds, with has_sprint
    strength: np.ndarray  # (bootstrap, races, drivers)
    alpha: np.ndarray | None  # (bootstrap, width) stage scales, if stagewise
    p_dnf: np.ndarray  # (drivers,) grand-prix retirement probability
    p_dns: float  # chance an entrant does not start
    team_index: np.ndarray  # (drivers,) row of ``teams`` each driver scores for


def standings(
    results: pd.DataFrame, year: int, through_round: int
) -> tuple[pd.Series, pd.Series]:
    """Driver and constructor points after ``through_round``, sprints included.

    Summed from ``race_results.parquet``, which reproduces the official tables
    exactly -- 2024 and 2025 were checked to the point.  Constructors are
    keyed on the name raced under at the time, so a driver who changed teams
    mid-season credits each team with the races run for it.
    """
    rows = results.loc[
        (results["Year"] == year) & (results["RoundNumber"] <= through_round)
    ]
    return (
        rows.groupby("DriverId")["Points"].sum(),
        rows.groupby("TeamName")["Points"].sum(),
    )


def remaining_races(
    results: pd.DataFrame, year: int, through_round: int
) -> pd.DataFrame:
    """The rest of a completed season's calendar, read from its own results."""
    rows = results.loc[
        (results["Year"] == year) & (results["RoundNumber"] > through_round)
    ]
    sprint = set(rows.loc[rows["session_type"] == "S", "RoundNumber"])
    gp = rows.loc[rows["session_type"] == "R"]
    races = (
        gp.groupby("RoundNumber")
        .agg(EventName=("EventName", "first"), RaceDate=(ORDER_COL, "first"))
        .reset_index()
    )
    races["has_sprint"] = races["RoundNumber"].isin(sprint)
    return races


def remaining_races_from_schedule(
    year: int, through_round: int, *, offline: bool = True
) -> pd.DataFrame:
    """The rest of a live season's calendar, from FastF1's schedule.

    Offline by default: the schedule is almost always in the cache, and a
    calendar lookup is not worth a request against the hourly limit.
    """
    import fastf1

    from src.data import ingest

    ingest.configure(offline=offline)
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    upcoming = schedule.loc[schedule["RoundNumber"] > through_round]
    return pd.DataFrame(
        {
            "RoundNumber": upcoming["RoundNumber"].astype(int).to_numpy(),
            "EventName": upcoming["EventName"].astype(str).to_numpy(),
            "RaceDate": pd.to_datetime(upcoming["EventDate"]).to_numpy(),
            "has_sprint": [ingest.event_has_sprint(e) for _, e in upcoming.iterrows()],
        }
    )


def _placeholders(
    entries: pd.DataFrame, races: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """Unraced rows for every remaining race, so features can be built for them.

    Every order feature is strictly prior-race, so a placeholder's own blank
    outcome never reaches its features, and each later placeholder sees the
    same history as the first -- the running order frozen after the last
    completed round.  Only the circuit-specific features differ between them.
    """
    blocks = []
    for race in races.itertuples(index=False):
        block = entries.copy()
        block["RoundNumber"] = int(race.RoundNumber)
        block[ORDER_COL] = pd.Timestamp(race.RaceDate)
        block["EventName"] = race.EventName
        block["circuit_key"] = predict.circuit_for_event(history, race.EventName)
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True)
    out["session_type"] = "R"
    out["ClassifiedPosition"] = np.nan
    out["grid_position"] = np.nan
    out["grid_position_pct"] = np.nan
    out["field_size"] = float(len(entries))
    return out


@contextlib.contextmanager
def _quiet(*names: str):
    """Silence warnings the dataset build repeats about history on every call.

    They describe the stored results, not this forecast, and ``predict next``
    already surfaces them once; a backtest would print them per cutoff.
    """
    loggers = [logging.getLogger(n) for n in names]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)


def prepare_season(
    dataset: pd.DataFrame,
    results: pd.DataFrame,
    profiles: pd.DataFrame | None,
    *,
    year: int,
    through_round: int,
    races: pd.DataFrame,
    spec: OrderSpec,
    feature_config: OrderFeatureConfig,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = config.RANDOM_SEED,
) -> SeasonSetup:
    """Fit everything on races up to ``through_round`` and nothing after it.

    Args:
        dataset: The modelling table, grands prix only.
        results: ``race_results.parquet`` -- sprints and non-starters included,
            which the standings and the DNF inference path both need.
        races: The remaining calendar, from :func:`remaining_races` for a
            completed season or :func:`remaining_races_from_schedule` for a
            live one.
        spec: A ``pre_weekend`` order spec; future races have no grid.
        n_boot: Bootstrap refits.  0 uses the point fit alone, with no
            parameter uncertainty.
    """
    if spec.stage != "pre_weekend":
        raise ValueError(
            "a season forecast needs a pre_weekend spec: future "
            f"races have no grid, and {spec.name!r} is {spec.stage}"
        )
    if races.empty:
        raise ValueError(f"no races left in {year} after round {through_round}")

    seen = (results["Year"] < year) | (
        (results["Year"] == year) & (results["RoundNumber"] <= through_round)
    )
    past_results = results.loc[seen]
    past = dataset.loc[
        (dataset["Year"] < year)
        | ((dataset["Year"] == year) & (dataset["RoundNumber"] <= through_round))
    ]

    last = past_results.loc[
        (past_results["Year"] == year)
        & (past_results["RoundNumber"] == through_round)
        & (past_results["session_type"] == "R")
    ]
    entries = (
        last[["DriverId", "Abbreviation", "TeamId", "TeamName"]]
        .drop_duplicates("DriverId")
        .reset_index(drop=True)
    )
    entries["Year"] = year

    frame = add_order_features(
        pd.concat(
            [past, _placeholders(entries, races, past_results)], ignore_index=True
        ),
        feature_config,
    )
    future = frame.loc[
        frame["ClassifiedPosition"].isna()
        & (frame["Year"] == year)
        & (frame["RoundNumber"] > through_round)
    ]

    train, calendar = _training_rows(frame, past, spec)
    blocks = [
        future.loc[future["RoundNumber"] == race.RoundNumber]
        .set_index("DriverId")
        .loc[entries["DriverId"]]
        .reset_index()
        for race in races.itertuples(index=False)
    ]
    strength, alpha = _bootstrap_strengths(train, spec, blocks, n_boot, seed)

    # Attrition: the DNF model at its pre-weekend configuration, through the
    # same inference path ``predict next`` uses, for the next race.  It carries
    # to every remaining race: reliability features barely move, and on the DNF
    # evidence there is almost no pre-weekend signal to lose.
    estimator, _, selected = predict.fit_current(past, stage="pre_weekend")
    first = races.iloc[0]
    with _quiet("src.features.labels", "src.data.generate_dataset"):
        rows = predict.build_inference_rows(
            past_results,
            profiles,
            year=year,
            round_number=int(first.RoundNumber),
            race_date=first.RaceDate,
            event_name=first.EventName,
            entries=entries[["DriverId", "TeamId"]],
        )
    p_dnf = (
        pd.Series(estimator.predict_proba(rows[selected])[:, 1], index=rows["DriverId"])
        .reindex(entries["DriverId"])
        .fillna(float(past["dnf"].mean()))
        .to_numpy()
    )

    with _quiet("src.features.labels"):
        labelled = add_race_outcome_labels(
            past_results.loc[past_results["session_type"] == "R"]
        )
    recent = labelled.merge(calendar[list(RACE_KEYS)], on=list(RACE_KEYS), how="inner")
    p_dns = float(1 - recent["started"].mean()) if len(recent) else 0.0

    driver_pts, team_pts = standings(results, year, through_round)
    entries["points_so_far"] = entries["DriverId"].map(driver_pts).fillna(0.0)
    others = (
        driver_pts.drop(entries["DriverId"], errors="ignore")
        .rename("points_so_far")
        .reset_index()
    )
    others = others.join(
        _latest_names(past_results.loc[past_results["Year"] == year]), on="DriverId"
    )
    team_names = sorted(set(team_pts.index) | set(entries["TeamName"]))
    teams = pd.DataFrame({"TeamName": team_names})
    teams["points_so_far"] = teams["TeamName"].map(team_pts).fillna(0.0)
    team_index = (
        teams.reset_index().set_index("TeamName").loc[entries["TeamName"], "index"]
    )

    return SeasonSetup(
        year=year,
        through_round=through_round,
        drivers=entries,
        others=others,
        teams=teams,
        races=races.reset_index(drop=True),
        strength=strength,
        alpha=alpha,
        p_dnf=p_dnf,
        p_dns=p_dns,
        team_index=team_index.to_numpy(),
    )


def _latest_names(results: pd.DataFrame) -> pd.DataFrame:
    """Each driver's abbreviation and team as of their most recent race.

    Sorted by date first: ``drop_duplicates(keep="last")`` on the raw results
    keeps whichever row happens to sit last, which named Perez's 2026 Cadillac
    seat "Red Bull Racing".
    """
    ordered = results.sort_values([ORDER_COL, "RoundNumber"], kind="mergesort")
    return ordered.drop_duplicates("DriverId", keep="last").set_index("DriverId")[
        ["Abbreviation", "TeamName"]
    ]


def _training_rows(
    frame: pd.DataFrame, past: pd.DataFrame, spec: OrderSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The spec's lookback over completed races, ages counted back from now."""
    calendar = (
        past[[*RACE_KEYS, ORDER_COL]]
        .drop_duplicates()
        .sort_values(ORDER_COL)
        .reset_index(drop=True)
    )
    if spec.lookback_races is not None:
        calendar = calendar.tail(spec.lookback_races)
    calendar["age"] = np.arange(len(calendar), 0, -1)
    train = frame.merge(calendar[[*RACE_KEYS, "age"]], on=list(RACE_KEYS), how="inner")
    train = train.loc[
        pd.to_numeric(train["ClassifiedPosition"], errors="coerce").notna()
    ]
    return train, calendar


def _bootstrap_strengths(
    train: pd.DataFrame,
    spec: OrderSpec,
    blocks: list[pd.DataFrame],
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Refit on Bayesian-bootstrap race weights; strengths for every block.

    The Bayesian bootstrap (Rubin, 1981) reweights whole races with Dirichlet
    weights rather than resampling them: a race drawn twice would otherwise
    merge into one forty-car "race" in the ranking likelihood.  Races are the
    unit because cars in one race share its safety cars and its weather.

    Returns:
        ``(n_boot, len(blocks), cars)`` strengths and ``(n_boot, width)``
        stage scales (``None`` for standard Plackett-Luce).  ``n_boot=0``
        returns the single point fit.
    """
    rng = np.random.default_rng(seed)
    codes = pd.factorize(pd.MultiIndex.from_frame(train[list(RACE_KEYS)]))[0]
    n_races = codes.max() + 1
    weights = (
        [None]
        if n_boot == 0
        else [rng.dirichlet(np.ones(n_races))[codes] * n_races for _ in range(n_boot)]
    )
    strength, alpha = [], []
    for w in weights:
        model = fit_order_model(train, spec, age=train["age"].to_numpy(), weight=w)
        strength.append([model.strength(block) for block in blocks])
        alpha.append(model.alpha_)
    return np.asarray(strength), (None if alpha[0] is None else np.asarray(alpha))


def saturday_outlook(
    dataset: pd.DataFrame,
    results: pd.DataFrame,
    profiles: pd.DataFrame | None,
    *,
    year: int,
    round_number: int,
    race_date,
    event_name: str,
    grid: dict[str, float],
    spec: OrderSpec,
    feature_config: OrderFeatureConfig,
    n_boot: int = DEFAULT_BOOTSTRAP,
    n_samples: int = DEFAULT_TRIALS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """One race, after qualifying: the post-quali model on the real grid.

    Fits on every race strictly before this one -- a race that has already run
    is therefore an honest backtest, never in-sample -- and builds the
    unraced row through the same path as ``predict next``, so the DNF model's
    grid-based attrition and the order model see identical inputs.  Nobody is
    given a non-start chance: a car with a grid slot has, by construction,
    turned up.

    Args:
        grid: ``DriverId -> starting position``.  From qualifying for a live
            race (penalties not yet applied), or from the race's own results
            for a backtest.
    """
    if spec.stage != "post_quali":
        raise ValueError(
            f"{spec.name!r} is {spec.stage}; a Saturday forecast "
            "needs the post_quali spec"
        )
    before = lambda f: (f["Year"] < year) | (  # noqa: E731
        (f["Year"] == year) & (f["RoundNumber"] < round_number)
    )
    past, past_results = dataset.loc[before(dataset)], results.loc[before(results)]
    entries = predict.entry_list(past_results)
    entries = entries.loc[entries["DriverId"].isin(grid)].reset_index(drop=True)

    with _quiet("src.features.labels", "src.data.generate_dataset"):
        rows = predict.build_inference_rows(
            past_results,
            profiles,
            year=year,
            round_number=round_number,
            race_date=race_date,
            event_name=event_name,
            entries=entries,
            grid=grid,
        )
    frame = add_order_features(
        pd.concat([past, rows], ignore_index=True), feature_config
    )
    upcoming = (
        frame.loc[(frame["Year"] == year) & (frame["RoundNumber"] == round_number)]
        .set_index("DriverId")
        .loc[entries["DriverId"]]
        .reset_index()
    )

    train, _ = _training_rows(frame.loc[before(frame)], past, spec)
    strength, alpha = _bootstrap_strengths(train, spec, [upcoming], n_boot, seed)

    estimator, _, selected = predict.fit_current(past, stage="post_quali")
    p_dnf = (
        pd.Series(estimator.predict_proba(rows[selected])[:, 1], index=rows["DriverId"])
        .reindex(entries["DriverId"])
        .to_numpy()
    )

    names = _latest_names(past_results)
    drivers = entries.assign(
        Abbreviation=entries["DriverId"].map(names["Abbreviation"]),
        TeamName=entries["DriverId"].map(names["TeamName"]),
        points_so_far=0.0,
    )
    setup = SeasonSetup(
        year=year,
        through_round=round_number - 1,
        drivers=drivers,
        others=pd.DataFrame(columns=["DriverId", "points_so_far"]),
        teams=pd.DataFrame(
            {"TeamName": sorted(drivers["TeamName"].unique()), "points_so_far": 0.0}
        ),
        races=pd.DataFrame(
            {
                "RoundNumber": [round_number],
                "EventName": [event_name],
                "RaceDate": [pd.Timestamp(race_date)],
                "has_sprint": [False],
            }
        ),
        strength=strength,
        alpha=alpha,
        p_dnf=p_dnf,
        p_dns=0.0,
        team_index=np.zeros(len(drivers), dtype=int),
    )
    out = race_outlook(setup, n_samples=n_samples, seed=seed)
    return out.assign(
        grid=out["Abbreviation"].map(
            drivers.set_index("Abbreviation")["DriverId"].map(grid)
        )
    )


# --------------------------------------------------------------------------- #
# Simulating
# --------------------------------------------------------------------------- #


@dataclass
class SeasonDraws:
    """Final points in every trial."""

    setup: SeasonSetup
    drivers: np.ndarray  # (trials, simulated drivers)
    teams: np.ndarray  # (trials, teams)


def _fastest_lap_bonus(
    strength: np.ndarray, positions: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One point to a top-ten finisher, drawn in proportion to strength.

    A modest model of a noisy award -- in practice it often went to whoever
    pitted late for fresh tyres -- and worth about one point a race to the
    front of the field in the seasons it existed.
    """
    eligible = (positions >= 1) & (positions <= POINTS_POSITIONS)
    z = np.where(eligible, strength + rng.gumbel(size=strength.shape), -np.inf)
    winner = np.argmax(z, axis=1)
    bonus = np.zeros(strength.shape)
    has = eligible.any(axis=1)
    bonus[np.arange(len(winner))[has], winner[has]] = 1.0
    return bonus


def simulate(
    setup: SeasonSetup,
    *,
    n_trials: int = DEFAULT_TRIALS,
    noise: SeasonNoise = SeasonNoise(),
    seed: int = config.RANDOM_SEED,
) -> SeasonDraws:
    """Run every remaining race ``n_trials`` times and total the points."""
    rng = np.random.default_rng(seed)
    n_boot, n_races, n_drivers = setup.strength.shape
    pick = rng.integers(0, n_boot, size=n_trials)
    offsets = draw_trial_offsets(rng, n_trials, n_races, setup.team_index, noise)
    alpha = None if setup.alpha is None else setup.alpha[pick]

    gp_table = grand_prix_points(setup.year, n_drivers)
    sprint_table = sprint_points(setup.year, n_drivers)
    fastest_lap = has_fastest_lap_point(setup.year)
    p_gp = setup.p_dns + (1 - setup.p_dns) * setup.p_dnf
    p_sprint = setup.p_dns + (1 - setup.p_dns) * setup.p_dnf * SPRINT_ATTRITION_RATIO

    earned = np.zeros((n_trials, n_drivers))
    for k, race in enumerate(setup.races.itertuples(index=False)):
        strength = setup.strength[pick, k, :] + offsets[:, k, :]
        if race.has_sprint:
            out = rng.random((n_trials, n_drivers)) < p_sprint[None, :]
            earned += sprint_table[sample_positions(strength, out, alpha, rng)]
        out = rng.random((n_trials, n_drivers)) < p_gp[None, :]
        positions = sample_positions(strength, out, alpha, rng)
        earned += gp_table[positions]
        if fastest_lap:
            earned += _fastest_lap_bonus(strength, positions, rng)

    drivers = setup.drivers["points_so_far"].to_numpy()[None, :] + earned
    membership = np.zeros((n_drivers, len(setup.teams)))
    membership[np.arange(n_drivers), setup.team_index] = 1.0
    teams = setup.teams["points_so_far"].to_numpy()[None, :] + earned @ membership
    return SeasonDraws(setup, drivers, teams)


def _champion_share(points: np.ndarray) -> np.ndarray:
    """Share of trials each column finishes top, ties split evenly."""
    best = points.max(axis=1, keepdims=True)
    tied = points == best
    return (tied / tied.sum(axis=1, keepdims=True)).mean(axis=0)


def _top_k_share(points: np.ndarray, k: int) -> np.ndarray:
    rank = (-points).argsort(axis=1).argsort(axis=1)
    return (rank < k).mean(axis=0)


def summarise(draws: SeasonDraws) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Driver and constructor tables: points now, the distribution, the odds.

    Drivers who scored this season but no longer race -- a mid-season
    replacement -- stay in the table on their fixed total, and are ranked
    against everyone else in every trial.  A standings table that dropped them
    would misstate both the order and the odds of a top-three finish.
    """
    setup = draws.setup

    def table(points: np.ndarray, base: pd.DataFrame) -> pd.DataFrame:
        out = base.copy()
        out["mean"] = points.mean(axis=0)
        for q in (10, 50, 90):
            out[f"p{q}"] = np.percentile(points, q, axis=0)
        out["p_champion"] = _champion_share(points)
        out["p_top3"] = _top_k_share(points, 3)
        return out.sort_values("mean", ascending=False, ignore_index=True)

    columns = ["Abbreviation", "TeamName", "points_so_far"]
    fixed = setup.others["points_so_far"].to_numpy()
    everyone = np.hstack(
        [draws.drivers, np.broadcast_to(fixed, (len(draws.drivers), len(fixed)))]
    )
    base = pd.concat(
        [
            setup.drivers[columns],
            setup.others.reindex(columns=columns).assign(
                TeamName=lambda d: d["TeamName"].fillna("") + " (no longer racing)"
            ),
        ],
        ignore_index=True,
    )
    drivers = table(everyone, base)
    teams = table(draws.teams, setup.teams[["TeamName", "points_so_far"]])
    return drivers, teams


def race_outlook(
    setup: SeasonSetup,
    *,
    race: int = 0,
    n_samples: int = DEFAULT_TRIALS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """One upcoming race, per driver: the chances of each outcome that pays.

    The same draws the season simulation makes, for a single race and with
    every bootstrap refit mixed in, so the probabilities carry parameter
    uncertainty as the championship numbers do.  Pre-weekend by construction:
    a qualified grid would sharpen these considerably, and the post-quali
    model is the one to use once it exists.
    """
    rng = np.random.default_rng(seed)
    n_boot, _, n_drivers = setup.strength.shape
    pick = rng.integers(0, n_boot, size=n_samples)
    strength = setup.strength[pick, race, :]
    alpha = None if setup.alpha is None else setup.alpha[pick]
    p_out = setup.p_dns + (1 - setup.p_dns) * setup.p_dnf
    out = rng.random((n_samples, n_drivers)) < p_out[None, :]
    positions = sample_positions(strength, out, alpha, rng)
    table = grand_prix_points(setup.year, n_drivers)
    frame = setup.drivers[["Abbreviation", "TeamName"]].copy()
    frame["p_win"] = (positions == 1).mean(axis=0)
    frame["p_podium"] = ((positions >= 1) & (positions <= 3)).mean(axis=0)
    frame["p_points"] = ((positions >= 1) & (positions <= POINTS_POSITIONS)).mean(
        axis=0
    )
    frame["p_dnf"] = p_out
    frame["exp_points"] = table[positions].mean(axis=0)
    return frame.sort_values("exp_points", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Backtesting on completed seasons
# --------------------------------------------------------------------------- #


def _crps(samples: np.ndarray, actual: float) -> float:
    """Continuous ranked probability score of a sample against one outcome."""
    x = np.sort(samples)
    n = len(x)
    spread = (2.0 * np.arange(1, n + 1) - n - 1) @ x / (n * n)
    return float(np.abs(x - actual).mean() - spread)


def score_season(draws: SeasonDraws, results: pd.DataFrame) -> pd.DataFrame:
    """Every simulated driver and constructor against where they really finished.

    ``pit`` is the probability integral transform: the share of trials below
    the real total.  Over many entities a calibrated forecast gives a uniform
    PIT, and an 80% interval that holds the truth 80% of the time.  Drivers
    who did not race every remaining round for the same team are flagged
    ``complete=False`` -- a mid-season replacement is not a forecast error --
    and constructors, which a driver swap does not disturb, are the cleaner
    test.
    """
    setup = draws.setup
    season = results.loc[results["Year"] == setup.year]
    final_drivers = season.groupby("DriverId")["Points"].sum()
    final_teams = season.groupby("TeamName")["Points"].sum()
    later = season.loc[
        (season["RoundNumber"] > setup.through_round) & (season["session_type"] == "R")
    ]
    raced = later.groupby(["DriverId", "TeamName"])["RoundNumber"].nunique()

    rows = []
    entities = [
        (
            "driver",
            setup.drivers["DriverId"],
            setup.drivers["TeamName"],
            draws.drivers,
            final_drivers,
        ),
        (
            "team",
            setup.teams["TeamName"],
            setup.teams["TeamName"],
            draws.teams,
            final_teams,
        ),
    ]
    for kind, ids, teams, samples, finals in entities:
        champion = finals.idxmax()
        p_champ = _champion_share(samples)
        for j, (entity, team) in enumerate(zip(ids, teams)):
            actual = float(finals.get(entity, 0.0))
            x = samples[:, j]
            complete = kind == "team" or raced.get((entity, team), 0) == len(
                setup.races
            )
            rows.append(
                {
                    "year": setup.year,
                    "through_round": setup.through_round,
                    "kind": kind,
                    "entity": entity,
                    "complete": bool(complete),
                    "actual": actual,
                    "mean": float(x.mean()),
                    "p10": float(np.percentile(x, 10)),
                    "p90": float(np.percentile(x, 90)),
                    "pit": float((x < actual).mean() + 0.5 * (x == actual).mean()),
                    "crps": _crps(x, actual),
                    "p_champion": float(p_champ[j]),
                    "champion": entity == champion,
                }
            )
    return pd.DataFrame(rows)


def calibration_summary(scored: pd.DataFrame) -> pd.Series:
    """Pooled season-forecast calibration, on complete entities only."""
    s = scored.loc[scored["complete"]]
    inside = (s["actual"] >= s["p10"]) & (s["actual"] <= s["p90"])
    pit = s["pit"].to_numpy()
    return pd.Series(
        {
            "entities": float(len(s)),
            "coverage80": float(inside.mean()),
            "below_p10": float((s["actual"] < s["p10"]).mean()),
            "above_p90": float((s["actual"] > s["p90"]).mean()),
            "pit_sd": float(pit.std()),  # 0.289 for a uniform PIT
            "crps": float(s["crps"].mean()),
            # One championship per (season, cutoff, kind): the Brier score of the
            # whole title distribution, summed over entrants, then averaged.
            "brier_champion": float(
                ((s["p_champion"] - s["champion"]) ** 2)
                .groupby(
                    [s[c] for c in ("year", "through_round", "kind") if c in s.columns]
                )
                .sum()
                .mean()
            ),
        }
    )
