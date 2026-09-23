"""Walk-forward evaluation of finishing-order models, race by race.

The sibling of :func:`src.models.train.walk_forward_races` for a ranking
target.  Same discipline: every race is scored by a model refitted on races
strictly before it, and nothing after.  Different unit: a ranking model trains
and scores *by race*, because the thing being predicted is an order.

Two layers are scored, deliberately separately:

**The ordering** -- how well Plackett-Luce ranks the cars that finished.  Scored
by the log-likelihood of the observed top-ten among finishers, against the
uniform null.  Closed form, no sampling noise, and the model's own proper
scoring rule, which is why the tuning in :mod:`src.models.tuning` optimises it.

**The composite** -- attrition then ordering, as the forecast is actually used.
Each race is sampled whole (see :func:`src.models.ranking.sample_finishing_positions`)
and the samples are scored on the eleven outcomes that matter for points:
P1 through P10 each, and "outside the points" for everything else, retirements
included.  Metrics:

``rps10``
    Ranked probability score over those eleven ordered outcomes.  The proper
    scoring rule for ordered categories: squared error on the cumulative
    distribution, so predicting P4 for a P3 finish costs less than predicting
    P9.  It lumps P11 with a retirement on purpose -- neither pays -- which
    concentrates the evaluation where the points cliff is.  Lower is better.
``points_mae``
    Mean absolute error of expected points against the points actually scored.
    The decision variable.  Grand-prix position points only; the fastest-lap
    bonus is left to the season simulation.
``brier_win`` / ``brier_podium`` / ``brier_points``, and ``calib_*``
    Whether "a 30% podium" happens 30% of the time.  The finishing-order brief
    predicts Plackett-Luce is right about winners and wrong about the tail;
    the ``calib_points`` slope is the diagnostic that would show it.

Every summary **pools** over driver-races rather than averaging per-race
figures.  A third of races have unusual shapes -- a safety-car lottery, eight
retirements -- and averaging ratios over races lets those set the headline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.features import registry
from src.features.build_features import ORDER_COL, RACE_KEYS
from src.models import train
from src.models.points_table import grand_prix_points
from src.models.ranking import (
    POINTS_POSITIONS,
    PlackettLuce,
    sample_finishing_positions,
    score_orders,
)

log = logging.getLogger(__name__)

#: Races used to choose features and hyperparameters.  Scoring starts in 2020
#: so the first scored race has two seasons of history behind its features and
#: its training window; it ends before 2024 so everything after is untouched.
DEV_START, DEV_END = "2020-01-01", "2024-01-01"
#: Races scored once, at the end, with the choices made on the development
#: period.  Nothing in this range informs any choice.
TEST_START = "2024-01-01"

#: Outcomes the composite is scored on: P1..P10, then everything else.
N_OUTCOMES = POINTS_POSITIONS + 1
DETERMINISTIC_SCALE = 1_000.0


@dataclass(frozen=True)
class OrderSpec:
    """Everything that defines one finishing-order model.

    Args:
        name: Label for reports.
        features: Registry columns forming the log-strength.
        stage: Latest information stage allowed; every feature must be
            knowable by then (checked, not trusted).
        l2: Prior precision on the standardised coefficients.
        lookback_races: Train on the most recent N races; ``None`` expands.
        truncate: Training truncation of the likelihood; ``None`` = full order.
        race_halflife: Optional time decay on training races, in races.
        ranking: ``"finishers"`` fits the order among classified cars only
            (treatment (b): a retirement is not ranked).  ``"all_starters"``
            ranks every starter by ``Position``, retirements last -- treatment
            (a), kept only to measure what it costs.
        deterministic: Skip fitting and order by a column instead: ``"grid"``
            (grid order), ``"team"`` (team pace order) or ``"uniform"``.  The
            named baselines of the brief.
        n_scales / l2_scale: Stagewise position scales; see
            :class:`src.models.ranking.PlackettLuce`.  0 is standard
            Plackett-Luce.
    """

    name: str
    features: tuple[str, ...] = ()
    stage: registry.Stage = "post_quali"
    l2: float = 1.0
    lookback_races: int | None = 60
    truncate: int | None = None
    race_halflife: float | None = None
    ranking: str = "finishers"
    deterministic: str | None = None
    n_scales: int = 0
    l2_scale: float = 1.0
    #: ``"plackett_luce"``, or ``"boosted"`` for the gradient-boosted ranker of
    #: :mod:`src.models.boosted_ranker`, whose tree parameters travel in
    #: ``params`` as sorted ``(name, value)`` pairs so the spec stays hashable.
    family: str = "plackett_luce"
    params: tuple[tuple[str, object], ...] = ()

    def with_(self, **changes) -> "OrderSpec":
        return replace(self, **changes)


def check_stage(spec: OrderSpec) -> None:
    """Refuse a spec whose features are not all knowable at its stage."""
    cutoff = registry.STAGE_ORDER.index(spec.stage)
    late = [f for f in spec.features
            if f in registry.BY_NAME
            and registry.STAGE_ORDER.index(registry.BY_NAME[f].stage) > cutoff]
    if late:
        raise ValueError(
            f"{spec.name}: {late} are not knowable at stage {spec.stage!r}")
    unknown = [f for f in spec.features if f not in registry.BY_NAME]
    if unknown:
        raise ValueError(f"{spec.name}: {unknown} are not registered features")


def ranking_target(frame: pd.DataFrame, ranking: str = "finishers") -> pd.Series:
    """The position each row holds in the ranking, NaN if it is not ranked."""
    if ranking == "finishers":
        return pd.to_numeric(frame["ClassifiedPosition"], errors="coerce")
    if ranking == "all_starters":
        return pd.to_numeric(frame["Position"], errors="coerce")
    raise ValueError(f"unknown ranking {ranking!r}")


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #

_CARRY = ("DriverId", "Abbreviation", "TeamId", "TeamName", "EventName",
          "grid_position", "ClassifiedPosition", "dnf", ORDER_COL)


@dataclass
class OrderWalk:
    """Out-of-sample strengths for every scored race, and the ordering scores."""

    spec: OrderSpec
    predictions: pd.DataFrame
    race_ll: pd.DataFrame
    coefficients: list[pd.Series] = field(default_factory=list)
    #: Stage scales in force for each scored race, keyed by (Year, Round);
    #: empty for a standard Plackett-Luce or a deterministic order.
    alphas: dict[tuple, np.ndarray] = field(default_factory=dict)


def _deterministic_strength(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "uniform":
        return np.zeros(len(frame))
    if kind == "grid":
        key = pd.to_numeric(frame["grid_position"], errors="coerce")
    elif kind == "team":
        key = frame["team_finish_pct_ewma"].fillna(1.0)
    else:
        raise ValueError(f"unknown deterministic order {kind!r}")
    return -DETERMINISTIC_SCALE * key.to_numpy(dtype=float)


def fit_order_model(
    train_rows: pd.DataFrame, spec: OrderSpec, age: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> PlackettLuce:
    """Fit the spec's model on ``train_rows``.

    Args:
        age: Races between each row's race and the race being predicted, 1 for
            the most recent.  Only read when the spec has a time decay.
        weight: Extra per-row weight, constant within a race -- how the season
            simulation bootstraps the fit over races.
    """
    if spec.race_halflife and age is not None:
        decay = 0.5 ** ((np.asarray(age, dtype=float) - 1) / spec.race_halflife)
        weight = decay if weight is None else decay * np.asarray(weight, dtype=float)
    if spec.family == "boosted":
        from src.models.boosted_ranker import DEFAULT_PARAMS, BoostedRanker

        model = BoostedRanker(list(spec.features), {**DEFAULT_PARAMS, **dict(spec.params)},
                              n_scales=spec.n_scales, l2_scale=spec.l2_scale)
    else:
        model = PlackettLuce(list(spec.features), l2=spec.l2, truncate=spec.truncate,
                             n_scales=spec.n_scales, l2_scale=spec.l2_scale)
    return model.fit(train_rows, ranking_target(train_rows, spec.ranking), weight)


def walk_forward_order(
    frame: pd.DataFrame,
    spec: OrderSpec,
    *,
    start_after: str | pd.Timestamp | None = DEV_START,
    end_before: str | pd.Timestamp | None = DEV_END,
    refit_every: int = 1,
    keep_coefficients: bool = False,
) -> OrderWalk:
    """Refit before every race and predict it, the way the model would be run.

    Args:
        frame: Modelling rows carrying the spec's features, from
            :func:`src.features.order_features.add_order_features`.
        start_after / end_before: Which races to *score*.  The training window
            is always measured over the full calendar, so the first scored
            race gets a full history rather than none.
        refit_every: Refit every N races.  1 is the operational cadence.
    """
    check_stage(spec)
    frame = frame.copy()
    frame[ORDER_COL] = pd.to_datetime(frame[ORDER_COL])
    race_id = list(RACE_KEYS)
    calendar = (frame[[*race_id, ORDER_COL]].drop_duplicates()
                .sort_values([ORDER_COL, *race_id], kind="mergesort")
                .reset_index(drop=True))
    calendar["race_index"] = np.arange(len(calendar))
    frame = frame.merge(calendar[[*race_id, "race_index"]], on=race_id, how="left")

    scored = calendar
    if start_after is not None:
        scored = scored.loc[scored[ORDER_COL] > pd.Timestamp(start_after)]
    if end_before is not None:
        scored = scored.loc[scored[ORDER_COL] < pd.Timestamp(end_before)]

    blocks, coefs, alphas = [], [], {}
    model, fitted_at = None, None
    for position, race in enumerate(scored.itertuples(index=False)):
        idx = race.race_index
        test = frame.loc[frame["race_index"] == idx]
        if spec.deterministic:
            strength = _deterministic_strength(test, spec.deterministic)
        else:
            if model is None or position - fitted_at >= refit_every:
                lo = 0 if spec.lookback_races is None else idx - spec.lookback_races
                in_window = (frame["race_index"] < idx) & (frame["race_index"] >= lo)
                train_rows = frame.loc[in_window]
                model = fit_order_model(train_rows, spec,
                                        age=idx - train_rows["race_index"].to_numpy())
                fitted_at = position
                if keep_coefficients and spec.family == "plackett_luce":
                    coefs.append(model.coefficients().rename(
                        f"{race.Year}-{race.RoundNumber}"))
            strength = model.strength(test)
            if model.alpha_ is not None:
                alphas[(race.Year, race.RoundNumber)] = model.alpha_
        block = test[[*race_id, *[c for c in _CARRY if c in test.columns]]].copy()
        block["strength"] = strength
        blocks.append(block)

    predictions = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    if predictions.empty:
        return OrderWalk(spec, predictions, pd.DataFrame(), coefs, alphas)
    race_ll = score_orders(
        predictions, predictions["strength"].to_numpy(),
        ranking_target(predictions, "finishers"), truncate=POINTS_POSITIONS,
        alphas=alphas,
    )
    if spec.deterministic:
        race_ll["log_lik"] = np.nan
    return OrderWalk(spec, predictions, race_ll, coefs, alphas)


def ordering_summary(walk: OrderWalk) -> pd.Series:
    """Pooled ordering scores: top-ten log-likelihood per race, and its gain."""
    ll = walk.race_ll
    return pd.Series({
        "races": float(len(ll)),
        "ll10": float(ll["log_lik"].mean()),
        "ll10_uniform": float(ll["log_lik_uniform"].mean()),
        "ll10_gain": float((ll["log_lik"] - ll["log_lik_uniform"]).mean()),
    })


# --------------------------------------------------------------------------- #
# Attrition
# --------------------------------------------------------------------------- #


def walk_forward_dnf(dataset: pd.DataFrame, stage: registry.Stage) -> pd.DataFrame:
    """Out-of-sample retirement probabilities from the existing DNF model.

    Uses the configuration :mod:`src.models.train` measured best for the
    stage -- a logistic on grid position after qualifying, on the reliability
    pair before it -- refitted before every race on its default window.  The
    training base rate travels alongside as ``p_dnf_flat``: the same attrition
    level with no per-car information, which is what "no DNF model" means.
    """
    result = train.walk_forward_races(
        dataset, stage=stage, model="logistic",
        feature_set=train.default_feature_set(stage),
    )
    out = result.predictions[[*RACE_KEYS, "DriverId", "predicted", "train_base_rate"]]
    return out.rename(columns={"predicted": "p_dnf", "train_base_rate": "p_dnf_flat"})


def attach_attrition(
    predictions: pd.DataFrame, dnf: pd.DataFrame | None, mode: str = "model"
) -> pd.DataFrame:
    """Add the ``p_dnf`` the composite will sample from.

    ``mode`` is ``"model"`` (the DNF model's per-car probability), ``"flat"``
    (the training base rate for every car) or ``"none"`` (nobody retires).
    """
    out = predictions.copy()
    if mode == "none":
        out["p_dnf"] = 0.0
        return out
    if dnf is None:
        raise ValueError(f"attrition mode {mode!r} needs DNF probabilities")
    merged = out.merge(dnf, on=[*RACE_KEYS, "DriverId"], how="left", validate="one_to_one")
    column = "p_dnf" if mode == "model" else "p_dnf_flat"
    fallback = merged["p_dnf_flat"].mean()
    out["p_dnf"] = merged[column].fillna(fallback).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Composite scoring
# --------------------------------------------------------------------------- #


def composite_scores(
    predictions: pd.DataFrame, *, n_samples: int = 4000,
    seed: int = config.RANDOM_SEED,
    alphas: dict[tuple, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Sample every race whole and score each car's outcome distribution.

    Needs ``strength`` and ``p_dnf`` per row.  Returns one row per car per race
    with its eleven outcome probabilities summarised into the metrics in the
    module docstring.
    """
    rng = np.random.default_rng(seed)
    points = np.zeros(N_OUTCOMES + 1)
    points[1:N_OUTCOMES] = grand_prix_points(2026)[1:N_OUTCOMES]   # 25..1, then 0
    outcome_points = points[1:]

    rows = []
    alphas = alphas or {}
    for key, race in predictions.groupby(list(RACE_KEYS), sort=False, observed=True):
        position = sample_finishing_positions(
            race["strength"].to_numpy(dtype=float),
            race["p_dnf"].to_numpy(dtype=float), n_samples, rng,
            alpha=alphas.get(tuple(key)))
        outcome = np.where((position >= 1) & (position <= POINTS_POSITIONS),
                           position, N_OUTCOMES)
        probs = np.stack([(outcome == c).mean(axis=0) for c in range(1, N_OUTCOMES + 1)],
                         axis=1)

        classified = pd.to_numeric(race["ClassifiedPosition"], errors="coerce").to_numpy()
        observed = np.where((classified >= 1) & (classified <= POINTS_POSITIONS),
                            classified, N_OUTCOMES).astype(int)
        cdf_pred = probs.cumsum(axis=1)[:, :POINTS_POSITIONS]
        cdf_obs = (np.arange(1, POINTS_POSITIONS + 1)[None, :] >= observed[:, None])
        expected = probs @ outcome_points

        block = race[[*RACE_KEYS, "DriverId", *[c for c in ("TeamId", "grid_position")
                                                 if c in race.columns]]].copy()
        block["rps10"] = ((cdf_pred - cdf_obs) ** 2).sum(axis=1) / POINTS_POSITIONS
        block["exp_points"] = expected
        block["points"] = outcome_points[observed - 1]
        block["p_win"] = probs[:, 0]
        block["p_podium"] = probs[:, :3].sum(axis=1)
        block["p_points"] = probs[:, :POINTS_POSITIONS].sum(axis=1)
        block["won"] = (observed == 1).astype(int)
        block["podium"] = (observed <= 3).astype(int)
        block["scored"] = (observed <= POINTS_POSITIONS).astype(int)
        # Expected outcome index, for a per-race rank correlation.
        block["exp_outcome"] = probs @ np.arange(1, N_OUTCOMES + 1)
        block["outcome"] = observed
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def composite_summary(scores: pd.DataFrame) -> pd.Series:
    """Pooled composite metrics over every scored car in every race."""
    err = scores["exp_points"] - scores["points"]
    rho = scores.groupby(list(RACE_KEYS), observed=True).apply(
        lambda r: spearmanr(r["exp_outcome"], r["outcome"])[0]
        if r["outcome"].nunique() > 1 else np.nan,
        include_groups=False)
    summary = {
        "races": float(scores.groupby(list(RACE_KEYS)).ngroups),
        "rps10": float(scores["rps10"].mean()),
        "points_mae": float(err.abs().mean()),
        "points_rmse": float(np.sqrt((err ** 2).mean())),
        "spearman": float(rho.mean()),
    }
    for event, prob in (("won", "p_win"), ("podium", "p_podium"), ("scored", "p_points")):
        label = {"won": "win", "podium": "podium", "scored": "points"}[event]
        summary[f"brier_{label}"] = float(((scores[prob] - scores[event]) ** 2).mean())
        summary[f"calib_{label}"] = train.calibration_slope(
            scores[event].to_numpy(), scores[prob].to_numpy())
    return pd.Series(summary)


def paired_race_bootstrap(
    a: pd.DataFrame, b: pd.DataFrame, metric: str = "rps10", *,
    n_boot: int = 2000, seed: int = config.RANDOM_SEED,
) -> dict[str, float]:
    """Interval on ``mean(a) - mean(b)`` that resamples whole races.

    Cars in one race share a safety car, a first-lap pile-up and the weather,
    so the race is the independent unit.  Both frames must score the same
    races; the difference is pooled over cars within each resample.
    """
    keys = list(RACE_KEYS)
    per_a = a.groupby(keys, observed=True)[metric].agg(["sum", "size"])
    per_b = b.groupby(keys, observed=True)[metric].agg(["sum", "size"])
    joined = per_a.join(per_b, lsuffix="_a", rsuffix="_b", how="inner")
    rng = np.random.default_rng(seed)
    n = len(joined)
    pick = rng.integers(0, n, size=(n_boot, n))
    sa, na = joined["sum_a"].to_numpy(), joined["size_a"].to_numpy()
    sb, nb = joined["sum_b"].to_numpy(), joined["size_b"].to_numpy()
    diff = sa[pick].sum(1) / na[pick].sum(1) - sb[pick].sum(1) / nb[pick].sum(1)
    point = sa.sum() / na.sum() - sb.sum() / nb.sum()
    return {"diff": float(point), "ci_low": float(np.percentile(diff, 2.5)),
            "ci_high": float(np.percentile(diff, 97.5)), "races": float(n)}


def evaluate(
    frame: pd.DataFrame,
    specs: Sequence[OrderSpec],
    dnf: pd.DataFrame | None,
    *,
    attrition: str = "model",
    start_after: str | None = DEV_START,
    end_before: str | None = DEV_END,
    n_samples: int = 4000,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Walk every spec forward and score it both ways.

    Returns:
        ``(table, scores)``: one summary row per spec, and each spec's
        per-car composite scores for paired comparisons.
    """
    rows, scores = [], {}
    for spec in specs:
        walk = walk_forward_order(frame, spec, start_after=start_after,
                                  end_before=end_before)
        composite = composite_scores(
            attach_attrition(walk.predictions, dnf, attrition), n_samples=n_samples,
            alphas=walk.alphas)
        scores[spec.name] = composite
        row = pd.concat([ordering_summary(walk),
                         composite_summary(composite).drop("races")])
        row["model"] = spec.name
        rows.append(row)
        log.info("%s: rps10=%.4f ll10_gain=%.3f", spec.name, row["rps10"], row["ll10_gain"])
    table = pd.DataFrame(rows).set_index("model")
    return table, scores
