"""The walk-forward must never let a race see its own future, and must score honestly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.order_features import add_order_features
from src.models import order_eval as oe
from tests.synthetic import make_order_results


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    from src.data.generate_dataset import build_dataset

    dataset, _ = build_dataset(make_order_results(seed=11), None,
                               circuit_reference=False, run_checks=False)
    return add_order_features(dataset)


SPEC = oe.OrderSpec("test", ("grid_position", "team_grid_pct_ewma"), lookback_races=12)
WINDOW = dict(start_after="2022-06-01", end_before="2023-06-01")


def test_grid_cannot_enter_a_pre_weekend_model() -> None:
    spec = oe.OrderSpec("bad", ("grid_position",), stage="pre_weekend")
    with pytest.raises(ValueError, match="not knowable"):
        oe.check_stage(spec)


def test_unregistered_features_are_refused() -> None:
    with pytest.raises(ValueError, match="not registered"):
        oe.check_stage(oe.OrderSpec("bad", ("made_up_column",)))


def test_a_race_is_predicted_from_its_past_only(frame) -> None:
    """Scramble every result after a race; its prediction must not move."""
    base = oe.walk_forward_order(frame, SPEC, **WINDOW)
    first = base.predictions[["Year", "RoundNumber"]].drop_duplicates().iloc[0]
    later = (frame["Year"] > first.Year) | (
        (frame["Year"] == first.Year) & (frame["RoundNumber"] > first.RoundNumber))
    scrambled = frame.copy()
    rng = np.random.default_rng(0)
    scrambled.loc[later, "ClassifiedPosition"] = rng.permutation(
        scrambled.loc[later, "ClassifiedPosition"].to_numpy())
    again = oe.walk_forward_order(scrambled, SPEC, **WINDOW)
    pick = lambda p: p.loc[(p["Year"] == first.Year) & (p["RoundNumber"] == first.RoundNumber),
                           "strength"].to_numpy()
    np.testing.assert_allclose(pick(base.predictions), pick(again.predictions))


def test_a_model_with_signal_beats_the_uniform_null(frame) -> None:
    walk = oe.walk_forward_order(frame, SPEC, **WINDOW)
    assert oe.ordering_summary(walk)["ll10_gain"] > 1.0


def test_stagewise_scales_travel_with_each_race(frame) -> None:
    walk = oe.walk_forward_order(frame, SPEC.with_(n_scales=3), **WINDOW)
    races = walk.predictions[["Year", "RoundNumber"]].drop_duplicates()
    assert len(walk.alphas) == len(races)
    assert all(a[0] == 1.0 for a in walk.alphas.values())


def test_a_perfect_forecast_scores_zero_rps() -> None:
    race = pd.DataFrame({
        "Year": 2024, "RoundNumber": 1, "DriverId": [f"d{i}" for i in range(12)],
        "ClassifiedPosition": [str(i) for i in range(1, 13)],
        "strength": -1_000.0 * np.arange(12), "p_dnf": 0.0,
    })
    scores = oe.composite_scores(race, n_samples=200)
    assert scores["rps10"].max() == pytest.approx(0.0)
    assert scores["exp_points"].tolist()[:3] == [25.0, 18.0, 15.0]


def test_attrition_modes() -> None:
    preds = pd.DataFrame({"Year": 2024, "RoundNumber": 1, "DriverId": ["a", "b"]})
    dnf = pd.DataFrame({"Year": 2024, "RoundNumber": 1, "DriverId": ["a", "b"],
                        "p_dnf": [0.1, 0.3], "p_dnf_flat": [0.2, 0.2]})
    assert oe.attach_attrition(preds, dnf, "model")["p_dnf"].tolist() == [0.1, 0.3]
    assert oe.attach_attrition(preds, dnf, "flat")["p_dnf"].tolist() == [0.2, 0.2]
    assert oe.attach_attrition(preds, None, "none")["p_dnf"].tolist() == [0.0, 0.0]


def test_paired_bootstrap_sees_a_real_difference() -> None:
    rng = np.random.default_rng(1)
    keys = pd.DataFrame({"Year": 2024, "RoundNumber": np.repeat(np.arange(40), 10)})
    a = keys.assign(rps10=rng.normal(0.10, 0.02, len(keys)))
    b = keys.assign(rps10=a["rps10"] + 0.01 + rng.normal(0, 0.002, len(keys)))
    ci = oe.paired_race_bootstrap(a, b)
    assert ci["ci_high"] < 0 and ci["diff"] == pytest.approx(-0.01, abs=0.002)
