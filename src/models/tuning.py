"""Feature selection and hyperparameter search, nested inside the walk-forward.

The DNF model was never tuned, and ``points_model_design.md`` says why that was
defensible there -- one feature, one effective degree of freedom -- and why it
stops being defensible the moment the feature count rises.  This module is the
search, and the protocol it follows is the part worth reading:

**Every choice is made on the development races and nothing else.**  Features
are selected and hyperparameters searched on races from 2020 to 2023
(:data:`src.models.order_eval.DEV_START` to ``DEV_END``), each scored by a model
refitted on strictly earlier races.  Races from 2024 onward are touched once, at
the end, by :mod:`src.models.forecast`'s ``evaluate``.  A search that could see
the test races would report its own optimism as skill.

**The objective is the ordering log-likelihood.**  Mean over races of the
log-probability of the observed top ten among finishers -- the model's own
proper scoring rule, closed-form, so two configurations differ by what they
predict rather than by Monte Carlo noise.  The composite metrics (ranked
probability score, expected points) are reported for the chosen models, not
searched on: the DNF stage in front of them is fixed, so a better ordering is a
better composite.

**An improvement has to clear a bar, and the bar is set by races.**  Cars in
one race share a safety car and the weather, so the race is the independent
unit.  A feature joins the set only if its per-race log-likelihood gain over the
current set has a paired t-statistic of at least :data:`MIN_T` *and* a mean of
at least :data:`MIN_GAIN` nats.  The first guards against noise, the second
against a real effect too small to be worth a column.  Forward selection
repeated over a dozen candidates still inflates false inclusions; the held-out
test is what catches that, and :mod:`src.models.forecast` reports the selected
model against the simpler rungs on it for exactly that reason.

Hyperparameters are searched by coordinate descent rather than a full grid: one
knob at a time over a short list of values, two passes.  It is not guaranteed to
find the joint optimum, and it does not need to -- the surface is flat near the
top, and a knob-by-knob table says what each one is worth, which a grid's single
winner does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from src.features.build_features import RACE_KEYS
from src.features.order_features import OrderFeatureConfig
from src.models.order_eval import DEV_END, DEV_START, OrderSpec, walk_forward_order

log = logging.getLogger(__name__)

#: Paired t-statistic, over races, that an addition must reach.
MIN_T = 2.0
#: Mean per-race log-likelihood gain, in nats, that an addition must reach.
MIN_GAIN = 0.02

FrameBuilder = Callable[[OrderFeatureConfig], pd.DataFrame]


def race_scores(
    frame: pd.DataFrame, spec: OrderSpec, *,
    start: str | None = DEV_START, end: str | None = DEV_END,
) -> pd.Series:
    """Out-of-sample top-ten log-likelihood per race, indexed by race."""
    walk = walk_forward_order(frame, spec, start_after=start, end_before=end)
    return walk.race_ll.set_index(list(RACE_KEYS))["log_lik"]


def paired_gain(candidate: pd.Series, incumbent: pd.Series) -> tuple[float, float]:
    """Mean per-race gain of ``candidate`` over ``incumbent``, and its t-statistic."""
    diff = (candidate - incumbent).dropna()
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return float(diff.mean()) if len(diff) else 0.0, 0.0
    return float(diff.mean()), float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))


@dataclass
class SelectionResult:
    """What forward selection chose, and every comparison it made on the way."""

    selected: tuple[str, ...]
    history: pd.DataFrame
    scores: pd.Series = field(repr=False, default=None)


def forward_select(
    frame: pd.DataFrame,
    base: OrderSpec,
    candidates: Sequence[str | tuple[str, ...]],
    *,
    start: str | None = DEV_START,
    end: str | None = DEV_END,
    min_t: float = MIN_T,
    min_gain: float = MIN_GAIN,
    max_steps: int = 12,
) -> SelectionResult:
    """Greedy forward selection over features or groups of features.

    Starts from ``base.features`` (which may be empty only if a candidate can
    stand alone) and at each step adds the candidate with the largest mean
    per-race gain, provided it clears both bars.  A candidate may be a tuple,
    so that features that only make sense together -- an interaction and its
    main effect -- enter together.

    Returns:
        The selected features and a history table with one row per candidate
        per step: its mean gain, t-statistic, and whether it was taken.
    """
    current = tuple(base.features)
    pool = [c if isinstance(c, tuple) else (c,) for c in candidates]
    pool = [c for c in pool if not set(c) <= set(current)]
    incumbent = race_scores(frame, base, start=start, end=end) if current else None
    rows = []
    for step in range(1, max_steps + 1):
        trials = []
        for group in pool:
            spec = base.with_(name="+".join(group), features=current + group)
            scores = race_scores(frame, spec, start=start, end=end)
            if incumbent is None:
                gain, t = float(scores.mean()), np.inf
            else:
                gain, t = paired_gain(scores, incumbent)
            trials.append((gain, t, group, scores))
            rows.append({"step": step, "candidate": "+".join(group), "gain": gain,
                         "t": t, "ll10": float(scores.mean()), "taken": False})
        if not trials:
            break
        gain, t, group, scores = max(trials, key=lambda x: x[0])
        if incumbent is not None and (t < min_t or gain < min_gain):
            log.info("step %d: best %s gain=%.3f t=%.2f -- stop", step, group, gain, t)
            break
        rows[-len(trials) + [g for _, _, g, _ in trials].index(group)]["taken"] = True
        current, incumbent = current + group, scores
        pool = [c for c in pool if c != group]
        log.info("step %d: + %s gain=%.3f t=%.2f", step, group, gain, t)
    return SelectionResult(current, pd.DataFrame(rows), incumbent)


def backward_prune(
    frame: pd.DataFrame, spec: OrderSpec, *,
    start: str | None = DEV_START, end: str | None = DEV_END,
    keep: Iterable[str] = (), min_t: float = MIN_T,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Drop any feature whose removal does not cost a significant amount.

    Forward selection judges each feature against the set *at the time it was
    added*; a later addition can make an earlier one redundant.  This checks
    each against the final set and removes the weakest one that no longer
    clears ``min_t``, repeating until every survivor earns its place.
    """
    current = list(spec.features)
    full = race_scores(frame, spec, start=start, end=end)
    rows = []
    while len(current) > 1:
        trials = []
        for name in current:
            if name in set(keep):
                continue
            reduced = spec.with_(features=tuple(c for c in current if c != name))
            scores = race_scores(frame, reduced, start=start, end=end)
            cost, t = paired_gain(full, scores)
            trials.append((t, name, cost, scores))
            rows.append({"feature": name, "cost_of_removal": cost, "t": t,
                         "n_features": len(current)})
        if not trials:
            break
        t, name, cost, scores = min(trials, key=lambda x: x[0])
        if t >= min_t:
            break
        log.info("prune %s (cost %.3f, t=%.2f)", name, cost, t)
        current.remove(name)
        full = scores
    return tuple(current), pd.DataFrame(rows)


#: The values each hyperparameter is searched over.  Model knobs change the fit;
#: feature knobs change the columns and force a rebuild of the order features.
MODEL_SPACE: dict[str, list] = {
    "lookback_races": [20, 40, 60, 100, None],
    "l2": [0.1, 1.0, 10.0, 100.0],
    "n_scales": [0, 2, 4, 10],
    "l2_scale": [0.1, 1.0, 10.0],
    "truncate": [None, 10, 5],
    "race_halflife": [None, 10.0, 20.0, 40.0],
}
FEATURE_SPACE: dict[str, list] = {
    "team_halflife": [2.0, 4.0, 8.0, 16.0, 32.0],
    "driver_halflife": [4.0, 12.0, 24.0, 48.0],
    "retention_shrinkage": [2.0, 6.0, 20.0],
    "mate_delta_clip": [None, 0.5, 0.3],
}


def coordinate_search(
    build: FrameBuilder,
    spec: OrderSpec,
    feature_config: OrderFeatureConfig,
    *,
    model_space: dict[str, list] | None = None,
    feature_space: dict[str, list] | None = None,
    passes: int = 2,
    start: str | None = DEV_START,
    end: str | None = DEV_END,
) -> tuple[OrderSpec, OrderFeatureConfig, pd.DataFrame]:
    """Tune one knob at a time, keeping each change that raises the objective.

    Args:
        build: Rebuilds the modelling frame for a feature configuration;
            called only when a feature knob changes, and cached.

    Returns:
        The tuned spec and feature configuration, and a table with one row
        per evaluation: knob, value, objective, and whether it was kept.
    """
    model_space = MODEL_SPACE if model_space is None else model_space
    feature_space = FEATURE_SPACE if feature_space is None else feature_space
    frames: dict[OrderFeatureConfig, pd.DataFrame] = {}

    def frame_for(cfg: OrderFeatureConfig) -> pd.DataFrame:
        if cfg not in frames:
            frames[cfg] = build(cfg)
        return frames[cfg]

    def objective(s: OrderSpec, cfg: OrderFeatureConfig) -> float:
        return float(race_scores(frame_for(cfg), s, start=start, end=end).mean())

    best = objective(spec, feature_config)
    rows = [{"pass": 0, "knob": "(start)", "value": "", "ll10": best, "kept": True}]
    for n in range(1, passes + 1):
        changed = False
        for knob, values in [*model_space.items(), *feature_space.items()]:
            is_feature = knob in feature_space
            current = getattr(feature_config if is_feature else spec, knob)
            for value in values:
                if value == current:
                    continue
                if is_feature:
                    cfg, s = replace(feature_config, **{knob: value}), spec
                else:
                    cfg, s = feature_config, spec.with_(**{knob: value})
                if knob == "l2_scale" and not s.n_scales:
                    continue
                score = objective(s, cfg)
                kept = score > best + 1e-4
                rows.append({"pass": n, "knob": knob, "value": str(value),
                             "ll10": score, "kept": kept})
                if kept:
                    best, spec, feature_config, changed = score, s, cfg, True
                    log.info("pass %d: %s=%s -> %.4f", n, knob, value, score)
        if not changed:
            break
    return spec, feature_config, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #

#: Every feature offered to forward selection, by the stage it becomes knowable.
#: Each is a hypothesis with a named mechanism; the existing DNF-era columns are
#: included deliberately, as the contrast their EWMA, lineage-keyed replacements
#: have to beat (5-race windows keyed on ``TeamId``, which resets at a rebrand).
PRE_WEEKEND_CANDIDATES: tuple[str | tuple[str, ...], ...] = (
    # car pace -- H1 (long window), H2 (qualifying vs race pace), H3 (lineage)
    "team_grid_pct_ewma", "team_finish_pct_ewma", "team_grid_pct_season",
    "team_avg_grid_5", "team_avg_finish_5", "team_points_rate_5",
    "team_rank_in_field",
    # races better than it qualifies -- H5
    "team_gain_ewma", "driver_gain_ewma",
    # driver skill against the only control for the car -- H4
    "driver_mate_grid_delta_ewma", "driver_mate_finish_delta_ewma",
    "driver_grid_pct_ewma", "driver_points_rate_5", "driver_avg_grid_5",
    # ...and its within-season counterpart, the drivers' version of H3
    "driver_mate_grid_delta_season", "driver_mate_finish_delta_season",
    # car-circuit suitability, experience
    "team_circuit_grid_residual_3", "driver_starts_at_circuit", "is_new_pairing",
)
#: Added after qualifying -- H6 (grid and its functional form), H7 (circuit).
POST_QUALI_CANDIDATES: tuple[str | tuple[str, ...], ...] = (
    *PRE_WEEKEND_CANDIDATES,
    "grid_position", "log_grid_position", "starts_from_pit_lane",
    "teammate_grid_delta", "grid_x_circuit_retention",
)
CANDIDATES_BY_STAGE = {
    "pre_weekend": PRE_WEEKEND_CANDIDATES,
    "post_quali": POST_QUALI_CANDIDATES,
}

#: Where the search starts.  Stagewise scales are on from the outset because
#: they were measured before any feature work (calibration slope for P(win)
#: 2.83 -> 1.44 on the development races), and selecting features under a
#: model known to be misspecified would select for the misspecification.
START_SPEC = OrderSpec("start", l2=1.0, lookback_races=60, n_scales=10, l2_scale=1.0)


@dataclass
class ProtocolResult:
    """Everything one stage's search produced."""

    stage: str
    spec: OrderSpec
    feature_config: OrderFeatureConfig
    selection: pd.DataFrame
    pruning: pd.DataFrame
    search: pd.DataFrame
    reselection: pd.DataFrame


def run_protocol(
    build: FrameBuilder, stage: str, *,
    feature_config: OrderFeatureConfig | None = None,
    start_spec: OrderSpec = START_SPEC,
) -> ProtocolResult:
    """Select, prune, tune, then select again under the tuned settings.

    The second selection is the stability check: if the tuned hyperparameters
    would have chosen different features, the first selection was an artefact
    of the starting values, and the result says so rather than hiding it.
    """
    cfg = feature_config or OrderFeatureConfig()
    frame = build(cfg)
    base = start_spec.with_(name=stage, stage=stage, features=())
    candidates = CANDIDATES_BY_STAGE[stage]

    first = forward_select(frame, base, candidates)
    pruned, pruning = backward_prune(frame, base.with_(features=first.selected))
    spec, cfg, search = coordinate_search(build, base.with_(features=pruned), cfg)

    again = forward_select(build(cfg), spec.with_(features=()), candidates)
    if set(again.selected) != set(spec.features):
        log.warning("%s: re-selection under tuned settings chose %s, not %s",
                    stage, again.selected, spec.features)
    return ProtocolResult(stage, spec, cfg, first.history, pruning, search, again.history)


#: Where the boosted ranker's random search looks.  Wide on purpose: the point
#: is to give the flexible family a real tuning budget, so that if it loses it
#: loses on its merits.
RANKER_SPACE: dict[str, list] = {
    "n_estimators": [50, 100, 200, 400],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 6],
    "min_child_weight": [1.0, 5.0, 20.0, 50.0],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0],
}


def random_search_ranker(
    frame: pd.DataFrame, spec: OrderSpec, *, n_iter: int = 24,
    space: dict[str, list] | None = None, seed: int = 0,
    start: str | None = DEV_START, end: str | None = DEV_END,
) -> tuple[OrderSpec, pd.DataFrame]:
    """Random search over the boosted ranker's tree parameters, on the dev races.

    Random rather than coordinate search because tree parameters interact
    strongly -- depth and learning rate trade against the number of trees --
    which is exactly where one-knob-at-a-time search goes wrong.
    """
    space = RANKER_SPACE if space is None else space
    rng = np.random.default_rng(seed)
    rows, best, best_spec = [], -np.inf, spec
    for i in range(n_iter):
        params = {k: v[rng.integers(len(v))] for k, v in space.items()}
        candidate = spec.with_(params=tuple(sorted(params.items())))
        score = float(race_scores(frame, candidate, start=start, end=end).mean())
        rows.append({"iter": i, **params, "ll10": score})
        if score > best:
            best, best_spec = score, candidate
            log.info("ranker iter %d: %.4f %s", i, score, params)
    return best_spec, pd.DataFrame(rows).sort_values("ll10", ascending=False)
