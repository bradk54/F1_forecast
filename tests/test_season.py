"""The season simulation's bookkeeping, checked where the answer is known exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import season
from src.models.order_eval import OrderSpec
from src.models.points_table import grand_prix_points, has_fastest_lap_point, sprint_points


def test_points_rules_change_where_the_sport_changed_them() -> None:
    assert not has_fastest_lap_point(2018)
    assert all(has_fastest_lap_point(y) for y in range(2019, 2025))
    assert not has_fastest_lap_point(2025)
    assert sprint_points(2020).sum() == 0
    assert sprint_points(2021)[1:4].tolist() == [3, 2, 1]
    assert sprint_points(2026)[1:9].tolist() == [8, 7, 6, 5, 4, 3, 2, 1]
    assert grand_prix_points(2026)[[0, 1, 10, 11]].tolist() == [0, 25, 1, 0]


def _setup(*, n_races=1, has_sprint=False, year=2026, p_dnf=0.0) -> season.SeasonSetup:
    """Three cars, two teams, strengths so far apart the order is certain."""
    drivers = pd.DataFrame({"DriverId": ["a", "b", "c"], "Abbreviation": ["A", "B", "C"],
                            "TeamId": ["x", "x", "y"], "TeamName": ["X", "X", "Y"],
                            "points_so_far": [10.0, 0.0, 5.0]})
    races = pd.DataFrame({"RoundNumber": np.arange(1, n_races + 1), "EventName": "E",
                          "RaceDate": pd.Timestamp("2026-10-01"), "has_sprint": has_sprint})
    strength = np.tile([100.0, 0.0, -100.0], (2, n_races, 1))
    return season.SeasonSetup(
        year=year, through_round=5, drivers=drivers,
        others=pd.DataFrame({"DriverId": ["z"], "points_so_far": [1.0],
                             "Abbreviation": ["Z"], "TeamName": ["Y"]}),
        teams=pd.DataFrame({"TeamName": ["X", "Y"], "points_so_far": [10.0, 6.0]}),
        races=races, strength=strength, alpha=None,
        p_dnf=np.full(3, p_dnf), p_dns=0.0, team_index=np.array([0, 0, 1]))


def test_one_certain_race_scores_exactly() -> None:
    draws = season.simulate(_setup(), n_trials=50)
    np.testing.assert_allclose(draws.drivers, np.tile([35.0, 18.0, 20.0], (50, 1)))
    np.testing.assert_allclose(draws.teams, np.tile([53.0, 21.0], (50, 1)))


def test_a_sprint_weekend_pays_twice() -> None:
    draws = season.simulate(_setup(has_sprint=True), n_trials=20)
    np.testing.assert_allclose(draws.drivers[0], [35.0 + 8, 18.0 + 7, 20.0 + 6])


def test_the_fastest_lap_point_exists_only_in_its_seasons() -> None:
    with_bonus = season.simulate(_setup(year=2023), n_trials=200)
    assert with_bonus.drivers.sum(axis=1).mean() == pytest.approx(15 + 58 + 1)
    without = season.simulate(_setup(year=2026), n_trials=200)
    assert without.drivers.sum(axis=1).mean() == pytest.approx(15 + 58)


def test_a_retired_car_scores_nothing() -> None:
    draws = season.simulate(_setup(p_dnf=1.0), n_trials=20)
    np.testing.assert_allclose(draws.drivers[0], [10.0, 0.0, 5.0])


def test_constructors_are_the_sum_of_their_drivers() -> None:
    setup = _setup(n_races=4)
    setup.strength = np.random.default_rng(0).normal(size=setup.strength.shape)
    draws = season.simulate(setup, n_trials=500)
    earned = draws.drivers - setup.drivers["points_so_far"].to_numpy()
    team_x = draws.teams[:, 0] - 10.0
    np.testing.assert_allclose(team_x, earned[:, 0] + earned[:, 1])


def test_championship_odds_sum_to_one() -> None:
    setup = _setup(n_races=3)
    setup.strength = np.zeros_like(setup.strength)
    drivers, teams = season.summarise(season.simulate(setup, n_trials=2000))
    assert drivers["p_champion"].sum() == pytest.approx(1.0, abs=1e-9)
    assert teams["p_champion"].sum() == pytest.approx(1.0, abs=1e-9)


def test_zero_noise_means_no_offsets() -> None:
    """Whatever structure the offsets take, no noise has to mean no movement."""
    rng = np.random.default_rng(0)
    offsets = season.draw_trial_offsets(rng, 100, 6, np.array([0, 0, 1, 1, 2]),
                                        season.SeasonNoise(0.0, 0.0))
    assert offsets.shape == (100, 6, 5)
    np.testing.assert_allclose(offsets, 0.0)


def test_crps_of_a_point_mass_at_the_truth_is_zero() -> None:
    assert season._crps(np.full(100, 42.0), 42.0) == pytest.approx(0.0)
    assert season._crps(np.full(100, 42.0), 40.0) == pytest.approx(2.0)


def test_a_season_forecast_refuses_a_post_quali_model() -> None:
    with pytest.raises(ValueError, match="pre_weekend"):
        season.prepare_season(
            pd.DataFrame(), pd.DataFrame(), None, year=2026, through_round=1,
            races=pd.DataFrame({"RoundNumber": [2]}),
            spec=OrderSpec("post", ("grid_position",)), feature_config=None)


def test_latest_names_follow_the_calendar_not_the_row_order() -> None:
    results = pd.DataFrame({
        "DriverId": ["per", "per"], "Abbreviation": ["PER", "PER"],
        "TeamName": ["Cadillac", "Red Bull Racing"], "RoundNumber": [1, 20],
        "RaceDate": pd.to_datetime(["2026-03-08", "2024-12-08"]),
    })
    assert season._latest_names(results).loc["per", "TeamName"] == "Cadillac"


def _saturday(results: pd.DataFrame) -> pd.DataFrame:
    from src.data.generate_dataset import build_dataset
    from src.features.order_features import OrderFeatureConfig

    dataset, _ = build_dataset(results, None, circuit_reference=False, run_checks=False)
    race = dataset.loc[(dataset["Year"] == 2023) & (dataset["RoundNumber"] == 10)]
    return season.saturday_outlook(
        dataset, results, None, year=2023, round_number=10,
        race_date=race["RaceDate"].iloc[0], event_name=race["EventName"].iloc[0],
        grid=dict(zip(race["DriverId"], race["grid_position"])),
        spec=OrderSpec("post", ("grid_position", "team_grid_pct_ewma"), lookback_races=20),
        feature_config=OrderFeatureConfig(), n_boot=3, n_samples=3000)


def test_a_saturday_forecast_never_sees_its_own_race() -> None:
    """Scramble the race's result: a forecast built from earlier races cannot move."""
    from tests.synthetic import make_order_results

    results = make_order_results(seed=21)
    before = _saturday(results)
    assert before["p_win"].sum() == pytest.approx(1.0, abs=0.01)
    assert before["p_podium"].sum() == pytest.approx(3.0, abs=0.05)

    scrambled = results.copy()
    race = (scrambled["Year"] == 2023) & (scrambled["RoundNumber"] == 10)
    rng = np.random.default_rng(0)
    for column in ("ClassifiedPosition", "Status", "Points", "Position"):
        scrambled.loc[race, column] = rng.permutation(scrambled.loc[race, column].to_numpy())
    after = _saturday(scrambled)
    pd.testing.assert_frame_equal(before, after)


def test_a_saturday_forecast_refuses_a_pre_weekend_model() -> None:
    with pytest.raises(ValueError, match="post_quali"):
        season.saturday_outlook(
            pd.DataFrame(), pd.DataFrame(), None, year=2026, round_number=1,
            race_date="2026-03-08", event_name="E", grid={},
            spec=OrderSpec("pre", ("team_grid_pct_ewma",), stage="pre_weekend"),
            feature_config=None)
