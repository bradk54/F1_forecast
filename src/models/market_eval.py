"""Does the model beat the betting market?  Three questions, three tests.

The protocol is ``References/market_benchmark_plan.md``; this module is its
code.  Everything marked *frozen* was written down before any market data was
pulled, and is not to be tuned against the results:

``Q1`` accuracy
    Winner log-loss ``-log p(actual winner)``, model against market, paired by
    race.  The multinomial proper score, and the one the plan names primary.
    Brier on ``p_win`` over all drivers is secondary.  Grid order (or team order
    before qualifying) and a uniform field are scored beside both so the reader
    can see the scale.
``Q2`` information
    Forecast encompassing (Fair & Shiller 1990), as a conditional logit over each
    race's field::

        P(i wins)  proportional to  exp(a log p_market,i + b log p_model,i)

    ``b > 0`` means the model knows something the market does not.  ``(a, b)``
    are fitted **walk-forward** -- on races before *t*, scoring race *t* -- since
    an in-sample blend weight flatters the blend.
``Q3`` money
    Fractional-Kelly staking on the disagreement, after the spread and the
    exchange fee.  Last, and the plan expects its interval to cross zero.

Every comparison resamples **whole races** (``order_eval.paired_race_bootstrap``),
because cars in one race share a safety car and the weather, and publishes the
**minimum detectable effect** beside it.  With ~45 races a gap under ~0.15 nats
cannot register, so "no significant difference" is read as "too few races", not
"equal".

The market is a benchmark, never an input: nothing here feeds a model feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.features.build_features import ORDER_COL, RACE_KEYS
from src.models import order_eval

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Frozen protocol
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Snapshot:
    """When the market is read, and which model it is read against.

    ``anchor`` is a column of the session clock; ``offset_min`` is signed.
    """

    name: str
    anchor: str
    offset_min: float
    stage: str


#: Frozen.  Even the earliest snapshot favours the market: it has seen the
#: week's news, and the model has seen only prior races.  Qualifying runs an
#: hour, so "30 minutes after it ends" is 90 minutes after it starts.
SNAPSHOTS: tuple[Snapshot, ...] = (
    Snapshot("pre_weekend", "fp1", -60.0, "pre_weekend"),
    Snapshot("post_quali", "quali", 90.0, "post_quali"),
    Snapshot("close", "race", -5.0, "post_quali"),
)
SNAPSHOT_BY_NAME = {s.name: s for s in SNAPSHOTS}

#: Frozen.  Price assigned to a driver the market has not listed (or not traded)
#: by the snapshot, before normalising.  The model is scored on the same field,
#: so the driver may not be dropped.
UNLISTED_PRICE = 0.005
#: A book wider than this is not a quote (bid 1c / ask 99c has a 50c midpoint for a
#: driver nobody expects to win).  Added after a first live run showed such books
#: inflating raw totals to 1.3 and above; it removes garbage, not inconvenient data.
MAX_QUOTE_SPREAD = 0.20
#: Frozen.  A race whose raw listed prices sum outside this range is flagged.
TOTAL_RANGE = (0.95, 1.15)
#: Probabilities are clipped here before any log-loss, for every forecaster alike.
#: Without it the deterministic baselines score minus infinity on one miss.
PROB_FLOOR = 0.005
#: Frozen.  Q2's walk-forward fit starts after this many races; before it the
#: blend is the market alone.  Ridge is toward (a, b) = (1, 0), the market.
MIN_FIT_RACES = 10
BLEND_RIDGE = 1.0
#: Frozen.  Q3: fraction of full Kelly, the model-over-ask edge needed to bet,
#: the half-spread charged where a source quotes no ask, and Kalshi's fee rate
#: (fee = rate x contracts x p x (1 - p), rounded up to the cent).  The fee rate
#: is the published general schedule at writing -- check the exchange's fee page.
KELLY_FRACTION = 0.25
EDGE_THRESHOLD = 0.05
ASSUMED_HALF_SPREAD = 0.01
FEE_RATE = {"kalshi": 0.07, "polymarket": 0.0}

#: Market-favourite cut for the slice tables.
FAVOURITE_P = 0.10


class MarketLeakageError(RuntimeError):
    """A market price from after the race began was about to be scored."""


# --------------------------------------------------------------------------- #
# Market side: prices -> a probability for every driver in the field
# --------------------------------------------------------------------------- #


def snapshot_times(clock: pd.DataFrame, snapshot: Snapshot) -> pd.DataFrame:
    """``Year, RoundNumber, snapshot_ts`` for one rule, refusing a late one.

    Raises:
        MarketLeakageError: Any snapshot at or after lights-out.  Raised, not
        warned, in the spirit of ``detect_target_leakage``: a price taken once
        the race is under way is not a forecast.
    """
    out = clock[["Year", "RoundNumber"]].copy()
    out["snapshot_ts"] = clock[snapshot.anchor] + pd.Timedelta(minutes=snapshot.offset_min)
    out["race_start"] = clock["race"]
    out = out.dropna(subset=["snapshot_ts", "race_start"])
    late = out.loc[out["snapshot_ts"] >= out["race_start"]]
    if not late.empty:
        raise MarketLeakageError(
            f"snapshot {snapshot.name!r} falls at or after the start of "
            f"{len(late)} race(s), e.g. {late.iloc[0][['Year', 'RoundNumber']].tolist()}")
    return out.drop(columns="race_start")


def quote(prices: pd.DataFrame) -> pd.DataFrame:
    """Add ``p`` (the price to use) and ``spread``.

    The bid/ask midpoint where the book is two-sided and sane; otherwise the last
    trade.  A Kalshi book with nobody on a side reads bid 0 / ask 1, and its
    midpoint is 0.5 for a 200-to-1 outsider, so a one-sided book is not a quote.
    """
    out = prices.copy()
    two_sided = ((out["bid"] > 0) & (out["ask"] < 1) & (out["ask"] >= out["bid"])
                 & (out["ask"] - out["bid"] <= MAX_QUOTE_SPREAD))
    out["p"] = np.where(two_sided, (out["bid"] + out["ask"]) / 2, out["price"])
    out["spread"] = np.where(two_sided, out["ask"] - out["bid"], np.nan)
    return out.dropna(subset=["p"])


def market_snapshot(
    prices: pd.DataFrame, field: pd.DataFrame, times: pd.DataFrame,
) -> pd.DataFrame:
    """The market's probability for every driver in every race's field.

    Args:
        prices: :func:`src.data.markets.build_prices` output.
        field: One row per driver per race to be scored: ``Year``,
            ``RoundNumber``, ``DriverId``.
        times: :func:`snapshot_times` output.

    Returns one row per (source, race, driver): ``p_market`` (normalised over the
    field), ``market_raw`` (the price before), ``listed``, ``spread``,
    ``age_min`` (snapshot minus the observation used) and ``raw_total`` (the sum
    of listed prices, for the coverage check).  A source with no observation at
    all before the snapshot has no rows for that race -- the market had not
    opened -- rather than a made-up field.
    """
    quoted = quote(prices).sort_values("ts")
    frames = []
    for source, sub in quoted.groupby("source"):
        left = field.merge(times, on=["Year", "RoundNumber"]).sort_values("snapshot_ts")
        merged = pd.merge_asof(
            left, sub[["Year", "RoundNumber", "DriverId", "ts", "p", "spread"]],
            left_on="snapshot_ts", right_on="ts", by=["Year", "RoundNumber", "DriverId"],
            direction="backward")
        merged["source"] = source
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    if (out["ts"] > out["snapshot_ts"]).any():
        raise MarketLeakageError("a price observation postdates its snapshot")

    keys = ["source", "Year", "RoundNumber"]
    opened = out.groupby(keys)["p"].transform(lambda s: s.notna().any())
    out = out.loc[opened].copy()
    out["listed"] = out["p"].notna()
    out["market_raw"] = out["p"].fillna(UNLISTED_PRICE)
    out["raw_total"] = out["p"].where(out["listed"]).groupby(
        [out[k] for k in keys]).transform("sum")
    out["p_market"] = out["market_raw"] / out.groupby(keys)["market_raw"].transform("sum")
    out["age_min"] = (out["snapshot_ts"] - out["ts"]).dt.total_seconds() / 60.0
    return out[[*keys, "DriverId", "p_market", "market_raw", "listed", "spread",
                "age_min", "raw_total"]].reset_index(drop=True)


def coverage(log_frame: pd.DataFrame) -> pd.DataFrame:
    """Per source and snapshot: races priced, drivers unlisted, races flagged.

    The check that runs before anything is scored.  A race whose listed total
    sits outside :data:`TOTAL_RANGE` is counted in ``flagged`` but kept.
    """
    per_race = (log_frame.groupby(["source", "snapshot", "Year", "RoundNumber"], observed=True)
                .agg(raw_total=("raw_total", "first"), unlisted=("listed", lambda s: (~s).sum()),
                     field=("DriverId", "size")).reset_index())
    per_race["flagged"] = ~per_race["raw_total"].between(*TOTAL_RANGE)
    return (per_race.groupby(["source", "snapshot"])
            .agg(races=("Year", "size"), flagged=("flagged", "sum"),
                 unlisted_drivers=("unlisted", "sum"), mean_total=("raw_total", "mean"))
            .reset_index())


# --------------------------------------------------------------------------- #
# Model side: the walk-forward win probabilities
# --------------------------------------------------------------------------- #


def model_win_probabilities(
    dataset: pd.DataFrame, stage: str, *, start_after: str,
) -> pd.DataFrame:
    """Out-of-sample ``p_win`` for every driver, for the tuned model and its baseline.

    The same path ``forecast evaluate`` scores: a walk-forward Plackett-Luce
    refitted before every race, attrition from the walk-forward DNF model, and
    each race sampled whole.  Every probability therefore predates its race.
    Returns ``p_model``, ``p_baseline`` (grid order after qualifying, team order
    before), ``p_uniform``, and ``won``.
    """
    from src.models import forecast

    spec, cfg = ((forecast.PRE_WEEKEND_SPEC, forecast.PRE_WEEKEND_FEATURES)
                 if stage == "pre_weekend"
                 else (forecast.POST_QUALI_SPEC, forecast.POST_QUALI_FEATURES))
    base_spec = next(s for s, _ in forecast.ladder(stage)
                     if s.name == ("team order" if stage == "pre_weekend" else "grid order"))
    from src.features.order_features import add_order_features

    dnf = order_eval.walk_forward_dnf(dataset, stage)
    frame = add_order_features(dataset, cfg)
    scored = {}
    for label, s in (("p_model", spec), ("p_baseline", base_spec)):
        walk = order_eval.walk_forward_order(frame, s, start_after=start_after, end_before=None)
        comp = order_eval.composite_scores(
            order_eval.attach_attrition(walk.predictions, dnf, "model"), alphas=walk.alphas)
        scored[label] = comp
    out = scored["p_model"][[*RACE_KEYS, "DriverId", "grid_position", "p_win", "won"]].rename(
        columns={"p_win": "p_model"})
    base = scored["p_baseline"][[*RACE_KEYS, "DriverId", "p_win"]].rename(
        columns={"p_win": "p_baseline"})
    out = out.merge(base, on=[*RACE_KEYS, "DriverId"], how="left", validate="one_to_one")
    out["p_uniform"] = 1.0 / out.groupby(list(RACE_KEYS))["DriverId"].transform("size")
    return out


# --------------------------------------------------------------------------- #
# The log: model and market, side by side
# --------------------------------------------------------------------------- #

LOG_COLUMNS = ["origin", "Year", "RoundNumber", "snapshot", "source", "DriverId",
               "p_model", "p_market", "p_baseline", "p_uniform", "market_raw",
               "raw_total", "listed", "spread", "age_min", "grid_position", "won"]


def assemble_log(
    model: pd.DataFrame, market: pd.DataFrame, snapshot: str, origin: str = "backfill",
) -> pd.DataFrame:
    """Join one snapshot's model and market probabilities on the model's field."""
    out = model.merge(market, on=["Year", "RoundNumber", "DriverId"], how="inner",
                      validate="one_to_many")
    out["snapshot"] = snapshot
    out["origin"] = origin
    return out[LOG_COLUMNS]


def build_backfill(
    dataset: pd.DataFrame, prices: pd.DataFrame, clock: pd.DataFrame, *,
    start_after: str = "2024-12-31",
) -> pd.DataFrame:
    """Every priced race from 2025 on, at every snapshot, model beside market."""
    model_by_stage = {stage: model_win_probabilities(dataset, stage, start_after=start_after)
                      for stage in {s.stage for s in SNAPSHOTS}}
    blocks = []
    for snap in SNAPSHOTS:
        model = model_by_stage[snap.stage]
        field = model[["Year", "RoundNumber", "DriverId"]]
        market = market_snapshot(prices, field, snapshot_times(clock, snap))
        if market.empty:
            continue
        blocks.append(assemble_log(model, market, snap.name))
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=LOG_COLUMNS)


def settle(log_frame: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Fill ``won`` for rows whose race has since been classified.

    A prospective row is written with ``won`` blank, like ``predictions.csv``,
    and filled in here once the result exists.
    """
    if "session_type" in results.columns:
        # A sprint also has a classified P1; the market prices the grand prix.
        results = results.loc[results["session_type"] == "R"]
    winners = results.loc[pd.to_numeric(results["ClassifiedPosition"], errors="coerce") == 1,
                          ["Year", "RoundNumber", "DriverId"]].assign(_won=1)
    ran = results[["Year", "RoundNumber"]].drop_duplicates().assign(_ran=1)
    out = log_frame.merge(winners, on=["Year", "RoundNumber", "DriverId"], how="left")
    out = out.merge(ran, on=["Year", "RoundNumber"], how="left")
    settled = out["_ran"].eq(1)
    out.loc[settled, "won"] = out.loc[settled, "_won"].fillna(0)
    return out.drop(columns=["_won", "_ran"])


def read_log() -> pd.DataFrame:
    if not config.MARKET_LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(config.MARKET_LOG_PATH)


def write_log(backfill: pd.DataFrame | None = None, live: pd.DataFrame | None = None) -> None:
    """Rewrite one origin and keep the other as it stands.

    A backfill is recomputable and is replaced wholesale; live rows are claims
    made before a race and are never replaced (see :func:`append_live`).
    """
    old = read_log()
    parts = []
    parts.append(backfill if backfill is not None else old.loc[old["origin"] == "backfill"])
    parts.append(live if live is not None else old.loc[old["origin"] == "live"])
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    config.MARKET_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(config.MARKET_LOG_PATH, index=False)


def append_live(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add prospective rows, refusing to replace a (race, snapshot, source) already logged.

    Letting a later look overwrite an earlier one would let the log be tidied
    after the fact.  Returns the live table and how many incoming rows were new.
    """
    old = read_log()
    live = old.loc[old["origin"] == "live"]
    key = ["Year", "RoundNumber", "snapshot", "source"]
    have = set(map(tuple, live[key].drop_duplicates().to_numpy())) if len(live) else set()
    fresh = rows.loc[[tuple(k) not in have for k in rows[key].to_numpy()]]
    return pd.concat([f for f in (live, fresh) if len(f)], ignore_index=True), len(fresh)


# --------------------------------------------------------------------------- #
# Q1: accuracy
# --------------------------------------------------------------------------- #

FORECASTERS = ("p_model", "p_market", "p_baseline", "p_uniform")


def clip_renormalise(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PROB_FLOOR, None)
    return p / p.sum()


def per_race_scores(log_frame: pd.DataFrame, forecasters=FORECASTERS) -> pd.DataFrame:
    """One row per (source, snapshot, race): winner log-loss and Brier per forecaster.

    Races with no classified winner in the field are skipped.  Columns are
    ``ll_<name>`` and ``brier_<name>``, so a frame can go straight into
    :func:`order_eval.paired_race_bootstrap` with ``metric="ll_p_model"``.
    """
    rows = []
    keys = ["source", "snapshot", "Year", "RoundNumber"]
    for key, race in log_frame.dropna(subset=["won"]).groupby(keys, sort=True, observed=True):
        won = race["won"].to_numpy(dtype=float)
        if won.sum() != 1:
            continue
        row = dict(zip(keys, key))
        for name in forecasters:
            p = clip_renormalise(race[name].to_numpy(dtype=float))
            row[f"ll_{name}"] = float(-np.log(p[won == 1][0]))
            row[f"brier_{name}"] = float(((p - won) ** 2).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def minimum_detectable_effect(diff: pd.Series, t: float = 2.0) -> float:
    """The paired gap, in nats, that would reach ``t`` standard errors with this many races."""
    n = diff.notna().sum()
    return float(t * diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def q1_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean winner log-loss and Brier per forecaster, and model minus market.

    ``model - market`` is negative when the model is more accurate.  The interval
    is a paired bootstrap over whole races; ``mde`` is the smallest gap that
    sample could have detected.
    """
    rows = []
    for (source, snapshot), g in scores.groupby(["source", "snapshot"]):
        row = {"source": source, "snapshot": snapshot, "races": len(g)}
        for name in FORECASTERS:
            row[f"ll_{name[2:]}"] = g[f"ll_{name}"].mean()
        row["brier_model"], row["brier_market"] = g["brier_p_model"].mean(), g["brier_p_market"].mean()
        ci = _paired(g, "ll_p_model", "ll_p_market")
        row.update({"ll_diff": ci["diff"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                    "mde": minimum_detectable_effect(g["ll_p_model"] - g["ll_p_market"])})
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(frame: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    """Paired race bootstrap of ``mean(a) - mean(b)`` over the frame's races."""
    keys = list(RACE_KEYS)
    return order_eval.paired_race_bootstrap(
        frame[keys].assign(m=frame[a].to_numpy()),
        frame[keys].assign(m=frame[b].to_numpy()), "m")


# --------------------------------------------------------------------------- #
# Q2: does the model know something the market does not?
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Blend:
    a: float
    b: float
    se_a: float
    se_b: float
    races: int

    @property
    def t_b(self) -> float:
        return self.b / self.se_b if self.se_b > 0 else float("nan")


def _races_xy(log_frame: pd.DataFrame) -> list[tuple[np.ndarray, int]]:
    """Per race: the ``(n, 2)`` matrix of ``log p_market, log p_model`` and the winner's row."""
    out = []
    for _, race in log_frame.dropna(subset=["won"]).groupby(["Year", "RoundNumber"]):
        won = race["won"].to_numpy(dtype=float)
        if won.sum() != 1:
            continue
        x = np.column_stack([
            np.log(clip_renormalise(race["p_market"].to_numpy(dtype=float))),
            np.log(clip_renormalise(race["p_model"].to_numpy(dtype=float)))])
        out.append((x, int(np.argmax(won))))
    return out


def fit_blend(races: list[tuple[np.ndarray, int]], ridge: float = BLEND_RIDGE) -> Blend:
    """Newton fit of ``(a, b)``, ridged toward the market ``(1, 0)``.

    The log-likelihood of a conditional logit is concave, so Newton from the
    market-only start converges in a handful of steps; the information matrix
    (the within-race covariance of the features, summed) also gives the standard
    errors.
    """
    target = np.array([1.0, 0.0])
    theta = target.copy()
    for _ in range(50):
        grad, hess = ridge * (theta - target), ridge * np.eye(2)
        for x, w in races:
            s = x @ theta
            p = np.exp(s - s.max())
            p /= p.sum()
            mean = p @ x
            grad += mean - x[w]
            centred = x - mean
            hess += (centred * p[:, None]).T @ centred
        step = np.linalg.solve(hess, grad)
        theta = theta - step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(hess)))
    return Blend(float(theta[0]), float(theta[1]), float(se[0]), float(se[1]), len(races))


def blend_loglosses(races: list[tuple[np.ndarray, int]], *, min_fit: int = MIN_FIT_RACES):
    """Walk-forward: per race, the blend's log-loss and the market's.

    Races are in calendar order (``groupby`` sorts them).  Race *t* is scored by
    a blend fitted on races before it, and by the market alone while fewer than
    ``min_fit`` have been seen -- so both columns cover the same races.
    """
    rows = []
    for t, (x, w) in enumerate(races):
        theta = np.array([1.0, 0.0])
        if t >= min_fit:
            fit = fit_blend(races[:t])
            theta = np.array([fit.a, fit.b])
        s = x @ theta
        blend_ll = -(s[w] - np.log(np.exp(s - s.max()).sum()) - s.max())
        rows.append((float(blend_ll), float(-x[w, 0]), t >= min_fit))
    return pd.DataFrame(rows, columns=["ll_blend", "ll_market", "fitted"])


def q2_table(log_frame: pd.DataFrame) -> pd.DataFrame:
    """Per source and snapshot: the full-sample blend and the walk-forward gain.

    ``b`` is the weight on the model's log-probability; ``t_b`` its t-statistic.
    ``a=1, b=0`` is "the market encompasses the model".  ``wf_gain`` is the
    walk-forward blend's log-loss minus the market's over the races where the
    blend was fitted (negative = the blend improved on the crowd), with a
    paired-bootstrap interval.
    """
    rows = []
    for (source, snapshot), g in log_frame.groupby(["source", "snapshot"]):
        races = _races_xy(g)
        if len(races) < 3:
            continue
        fit = fit_blend(races)
        walk = blend_loglosses(races)
        fitted = walk.loc[walk["fitted"]]
        gain = fitted["ll_blend"] - fitted["ll_market"]
        row = {"source": source, "snapshot": snapshot, "races": len(races),
               "a": fit.a, "b": fit.b, "se_b": fit.se_b, "t_b": fit.t_b,
               "wf_races": len(fitted), "wf_gain": float(gain.mean()) if len(gain) else np.nan}
        if len(gain) > 2:
            rng = np.random.default_rng(config.RANDOM_SEED)
            boot = rng.choice(gain.to_numpy(), size=(2000, len(gain))).mean(axis=1)
            row.update(ci_low=float(np.percentile(boot, 2.5)),
                       ci_high=float(np.percentile(boot, 97.5)),
                       mde=minimum_detectable_effect(gain))
        rows.append(row)
    return pd.DataFrame(rows)


def slice_table(log_frame: pd.DataFrame, source: str, snapshot: str) -> pd.DataFrame:
    """Where do model and market disagree usefully?  Brier by slice, paired by race.

    Driver-level slices (favourite against longshot, grid tier) and race-level
    ones (season, early against late).  Each row is mean squared error of
    ``p_win`` for model and market over the drivers in the slice, and the paired
    bootstrap of the difference; negative ``diff`` favours the model.
    """
    g = log_frame.loc[(log_frame["source"] == source) & (log_frame["snapshot"] == snapshot)
                      & log_frame["won"].notna()].copy()
    g["se_model"] = (g["p_model"] - g["won"]) ** 2
    g["se_market"] = (g["p_market"] - g["won"]) ** 2
    grid = pd.to_numeric(g["grid_position"], errors="coerce")
    slices = {
        "market favourite (p >= 10%)": g["p_market"] >= FAVOURITE_P,
        "market longshot (p < 10%)": g["p_market"] < FAVOURITE_P,
        "grid P1-2": grid <= 2,
        "grid P3-10": (grid > 2) & (grid <= 10),
        "grid P11+": grid > 10,
        "rounds 1-8": g["RoundNumber"] <= 8,
        "rounds 9+": g["RoundNumber"] > 8,
        **{f"{y}": g["Year"] == y for y in sorted(g["Year"].unique())},
    }
    rows = []
    for label, mask in slices.items():
        sub = g.loc[mask]
        if sub["RoundNumber"].nunique() + sub["Year"].nunique() < 3 or sub.empty:
            continue
        ci = _paired(sub, "se_model", "se_market")
        rows.append({"slice": label, "driver_races": len(sub), "races": int(ci["races"]),
                     "brier_model": sub["se_model"].mean(), "brier_market": sub["se_market"].mean(),
                     "diff": ci["diff"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Q3: money
# --------------------------------------------------------------------------- #


def kalshi_fee(contracts: float, price: float, rate: float) -> float:
    """Fee in dollars, rounded up to the cent, as the exchange publishes it."""
    return float(np.ceil(rate * contracts * price * (1.0 - price) * 100.0) / 100.0)


def stake_race(
    race: pd.DataFrame, *, source: str, kelly: float = KELLY_FRACTION,
    edge: float = EDGE_THRESHOLD, half_spread: float = ASSUMED_HALF_SPREAD,
) -> float:
    """Profit on a unit bankroll from one race's bets, after spread and fee.

    A bet is a Yes contract bought at the ask (the price plus the observed half
    spread, else ``half_spread``) whenever the model's probability exceeds that
    price by ``edge``.  The stake is ``kelly`` of full Kelly, ``(p - c)/(1 - c)``,
    and the bankroll is reset each race: the question is whether the edge pays,
    not how a compounding path happens to wander.  Several drivers can be backed
    in one race; their stakes are scaled down if they would exceed the bankroll.
    """
    winner = race["won"].to_numpy(dtype=float)
    spread = race["spread"].fillna(2 * half_spread).to_numpy(dtype=float) / 2
    cost = np.clip(race["p_market"].to_numpy() + spread, 0.01, 0.99)
    p = race["p_model"].to_numpy()
    bet = p - cost > edge
    if not bet.any():
        return 0.0
    stake = np.where(bet, kelly * (p - cost) / (1 - cost), 0.0)
    if stake.sum() > 1.0:
        stake = stake / stake.sum()
    contracts = stake / cost
    gross = (contracts * winner - stake).sum()
    fees = sum(kalshi_fee(c, k, FEE_RATE[source]) for c, k in zip(contracts[bet], cost[bet]))
    return float(gross - fees)


def q3_table(log_frame: pd.DataFrame) -> pd.DataFrame:
    """Staking return per source and snapshot: total, drawdown, and a bootstrap interval.

    ``total`` is summed per-race profit on a unit bankroll.  The interval
    resamples races; expect it to cross zero.
    """
    rows = []
    for (source, snapshot), g in log_frame.dropna(subset=["won"]).groupby(["source", "snapshot"]):
        profit = np.array([stake_race(r, source=source)
                           for _, r in g.groupby(["Year", "RoundNumber"]) if r["won"].sum() == 1])
        if not len(profit):
            continue
        rng = np.random.default_rng(config.RANDOM_SEED)
        boot = rng.choice(profit, size=(2000, len(profit))).sum(axis=1)
        curve = profit.cumsum()
        rows.append({"source": source, "snapshot": snapshot, "races": len(profit),
                     "bets_races": int((profit != 0).sum()), "total": profit.sum(),
                     "per_race": profit.mean(),
                     "max_drawdown": float((np.maximum.accumulate(curve) - curve).max()),
                     "ci_low": float(np.percentile(boot, 2.5)),
                     "ci_high": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(rows)
