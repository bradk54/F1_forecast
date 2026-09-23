"""Gradient-boosted learning-to-rank with a calibrated Plackett-Luce head.

Method 3 of ``References/finishing_order_models.md``, built so that it can be
compared with the linear Plackett-Luce fairly:

1. **Score.**  An ``XGBRanker`` with a pairwise objective learns a flexible
   score from every candidate feature, grouped by race (``RACE_KEYS`` is the
   ``qid``).  Trees find interactions -- grid mattering more at Monaco -- that a
   linear score must be told about.
2. **Calibrate.**  The score becomes the single feature of a *stagewise*
   Plackett-Luce, fitted on the most recent races of the training window, which
   the trees did not see.  That head supplies the temperature and the per-position
   scales, so the output is a proper distribution over orders with the same
   exact sampler as the linear model.

Why this has to beat a *tuned* linear model and not an untuned one: a boosted
ranker brings a dozen hyperparameters, and if it wins against a model given no
tuning budget, nobody can say whether it won on flexibility or on search.
:func:`src.models.tuning.random_search_ranker` gives it its own search on the
same development races.

The repository's prior on this family is poor, and worth stating: on the DNF
problem a random forest calibrated at 0.606 and gradient boosting at 0.212
against 0.915 for a one-feature logistic, because trees turn an ordinal input
into a step function.  The calibration head repairs the global scale of the
score; it cannot repair a score that is flat where it should not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src import config
from src.features.build_features import ORDER_COL, RACE_KEYS
from src.models.ranking import PlackettLuce

#: Share of the training window's races held back to fit the calibration head.
HEAD_FRACTION = 0.25

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 150, "learning_rate": 0.1, "max_depth": 3,
    "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


@dataclass
class BoostedRanker:
    """Pairwise XGBoost ranker, then a stagewise Plackett-Luce on its score."""

    features: Sequence[str]
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    n_scales: int = 10
    l2_scale: float = 0.1
    head_: PlackettLuce = field(default=None, init=False, repr=False)
    ranker_: Any = field(default=None, init=False, repr=False)

    @property
    def alpha_(self) -> np.ndarray | None:
        return None if self.head_ is None else self.head_.alpha_

    def _score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.ranker_.predict(frame[list(self.features)].to_numpy(dtype=float))

    def fit(self, frame: pd.DataFrame, order: pd.Series,
            race_weight: pd.Series | None = None) -> "BoostedRanker":
        """Trees on the older races, the calibration head on the newer ones.

        ``race_weight`` is accepted for interface parity and ignored: the
        pairwise objective takes per-group weights, but a time decay applied
        to the tree fit and not the head would make the two halves disagree
        about which races matter.
        """
        from xgboost import XGBRanker

        order = pd.to_numeric(pd.Series(order, index=frame.index), errors="coerce")
        ranked = frame.loc[order.notna()].assign(_order=order[order.notna()])
        dates = ranked.groupby(list(RACE_KEYS))[ORDER_COL].first().sort_values()
        n_head = max(1, int(round(len(dates) * HEAD_FRACTION)))
        head_races = dates.index[-n_head:]
        is_head = pd.MultiIndex.from_frame(ranked[list(RACE_KEYS)]).isin(head_races)

        trees = ranked.loc[~is_head].sort_values([*RACE_KEYS, "_order"])
        qid = pd.factorize(pd.MultiIndex.from_frame(trees[list(RACE_KEYS)]))[0]
        # Higher relevance is better: the winner of a race of m finishers gets m.
        size = trees.groupby(list(RACE_KEYS))["_order"].transform("size")
        relevance = (size - trees.groupby(list(RACE_KEYS))["_order"]
                     .rank(method="first")).to_numpy()
        self.ranker_ = XGBRanker(objective="rank:pairwise", tree_method="hist",
                                 random_state=config.RANDOM_SEED, **self.params)
        self.ranker_.fit(trees[list(self.features)].to_numpy(dtype=float),
                         relevance, qid=qid)

        head_rows = ranked.loc[is_head].assign(ranker_score=self._score(ranked.loc[is_head]))
        self.head_ = PlackettLuce(["ranker_score"], l2=0.01, truncate=10,
                                  n_scales=self.n_scales, l2_scale=self.l2_scale)
        self.head_.fit(head_rows, head_rows["_order"])
        return self

    def strength(self, frame: pd.DataFrame) -> np.ndarray:
        scored = frame.assign(ranker_score=self._score(frame))
        return self.head_.strength(scored)

    def coefficients(self) -> pd.Series:
        booster = self.ranker_.get_booster()
        gain = booster.get_score(importance_type="gain")
        return pd.Series({name: gain.get(f"f{i}", 0.0)
                          for i, name in enumerate(self.features)}).sort_values(ascending=False)
