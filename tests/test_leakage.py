"""Leakage is the failure mode that looks like success.

A feature that reads the current race's outcome produces a backtest that is
excellent and a forecast that is worthless, and nothing in the metrics
distinguishes the two.  These tests do two things: confirm the shipped builders
are clean, and confirm the detector can actually catch a leak — a detector that
always returns "clean" would be worse than none at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    build_history_features,
    detect_target_leakage,
    prior_expanding,
    prior_rolling,
    races_since,
)
from src.features.labels import add_race_outcome_labels


def test_shipped_builder_is_clean(labelled_results: pd.DataFrame) -> None:
    report = detect_target_leakage(labelled_results)
    assert report.empty, f"leaking features:\n{report.to_string(index=False)}"


@pytest.mark.parametrize(
    ("year", "round_number"),
    [(2021, 1), (2022, 3), (2022, 12), (2023, 6)],
)
def test_clean_at_every_point_in_the_calendar(
    labelled_results: pd.DataFrame, year: int, round_number: int
) -> None:
    """Season openers matter: they are where cross-season carry-over happens."""
    report = detect_target_leakage(
        labelled_results, flip_year=year, flip_round=round_number
    )
    assert report.empty, f"{year} R{round_number}:\n{report.to_string(index=False)}"


def test_detector_catches_a_deliberate_leak(labelled_results: pd.DataFrame) -> None:
    """A detector that never fires proves nothing.  Plant a leak; catch it."""

    def leaky_builder(frame: pd.DataFrame) -> pd.DataFrame:
        out = build_history_features(frame)
        # The classic mistake: a cumulative sum with no shift, so the current
        # race is inside its own feature.
        out["driver_dnf_cumsum_leaky"] = out.sort_values("RaceDate").groupby(
            "DriverId", observed=True
        )["dnf"].cumsum()
        return out

    report = detect_target_leakage(labelled_results, builder=leaky_builder)
    assert "driver_dnf_cumsum_leaky" in set(report["feature"])


def test_detector_catches_a_same_race_team_aggregate(
    labelled_results: pd.DataFrame,
) -> None:
    """The subtler leak: a team mean that includes the current race."""

    def leaky_builder(frame: pd.DataFrame) -> pd.DataFrame:
        out = build_history_features(frame)
        out["team_dnf_this_race_leaky"] = out.groupby(
            ["Year", "RoundNumber", "TeamId"], observed=True
        )["dnf"].transform("mean")
        return out

    report = detect_target_leakage(labelled_results, builder=leaky_builder)
    assert "team_dnf_this_race_leaky" in set(report["feature"])


def test_prior_rolling_excludes_the_current_row() -> None:
    frame = pd.DataFrame(
        {
            "DriverId": ["a"] * 5,
            "Year": [2022] * 5,
            "RoundNumber": [1, 2, 3, 4, 5],
            "RaceDate": pd.date_range("2022-03-01", periods=5, freq="14D"),
            "dnf": [1, 0, 0, 1, 0],
        }
    )
    rolled = prior_rolling(frame, "DriverId", "dnf", window=3)
    # First race has no history at all.
    assert np.isnan(rolled.iloc[0])
    # Second race sees only the first race's outcome.
    assert rolled.iloc[1] == pytest.approx(1.0)
    # Fourth race sees races 1-3, mean of (1, 0, 0).
    assert rolled.iloc[3] == pytest.approx(1 / 3)
    # Fifth race sees races 2-4, mean of (0, 0, 1).
    assert rolled.iloc[4] == pytest.approx(1 / 3)


def test_prior_expanding_excludes_the_current_row() -> None:
    frame = pd.DataFrame(
        {
            "DriverId": ["a"] * 4,
            "Year": [2022] * 4,
            "RoundNumber": [1, 2, 3, 4],
            "RaceDate": pd.date_range("2022-03-01", periods=4, freq="14D"),
            "dnf": [1, 1, 0, 0],
        }
    )
    expanded = prior_expanding(frame, "DriverId", "dnf")
    assert np.isnan(expanded.iloc[0])
    assert expanded.iloc[1] == pytest.approx(1.0)
    assert expanded.iloc[2] == pytest.approx(1.0)
    assert expanded.iloc[3] == pytest.approx(2 / 3)


def test_races_since_counts_only_prior_races() -> None:
    frame = pd.DataFrame(
        {
            "DriverId": ["a"] * 6,
            "Year": [2022] * 6,
            "RoundNumber": [1, 2, 3, 4, 5, 6],
            "RaceDate": pd.date_range("2022-03-01", periods=6, freq="14D"),
            "dnf": [0, 1, 0, 0, 1, 0],
        }
    )
    gaps = races_since(frame, "DriverId", "dnf")
    # NaN until a retirement has been seen.
    assert np.isnan(gaps.iloc[0]) and np.isnan(gaps.iloc[1])
    assert gaps.iloc[2] == 0.0  # race 2 retired, so zero races since
    assert gaps.iloc[3] == 1.0
    assert gaps.iloc[4] == 2.0
    assert gaps.iloc[5] == 0.0  # race 5 retired


def test_history_crosses_season_boundaries(labelled_results: pd.DataFrame) -> None:
    """A season opener should inherit the previous season's history.

    Manually seeding a new season from the last one, as the older notebooks do,
    is unnecessary when features are ordered by date across the whole dataset.
    """
    featured = build_history_features(labelled_results)
    openers = featured.loc[
        (featured["RoundNumber"] == 1) & (featured["Year"] > featured["Year"].min())
    ]
    assert openers["driver_dnf_rate_10"].notna().mean() > 0.9


def test_features_recover_planted_risk(labelled_results: pd.DataFrame) -> None:
    """The fixture plants latent per-driver risk; the features should find it."""
    featured = build_history_features(labelled_results)
    settled = featured.loc[featured["driver_races_to_date"] >= 10]
    correlation = settled[["driver_dnf_rate_career", "_true_driver_risk"]].corr(
        method="spearman"
    ).iloc[0, 1]
    assert correlation > 0.3


def test_missing_columns_raise_clearly() -> None:
    with pytest.raises(KeyError, match="missing required columns"):
        build_history_features(pd.DataFrame({"dnf": [0, 1]}))


# --------------------------------------------------------------------------- #
# Sprint weekends
# --------------------------------------------------------------------------- #
#
# A sprint round holds two events on two days.  That breaks two assumptions the
# first version of this code made: that (Year, RoundNumber) uniquely identifies
# a race, and that ``RaceDate`` alone orders events.  Both are exercised here.


@pytest.fixture(scope="module")
def sprint_results(labelled_results: pd.DataFrame) -> pd.DataFrame:
    """Rounds 1-4 gain a Saturday sprint alongside the Sunday grand prix."""
    sprints = labelled_results.loc[labelled_results["RoundNumber"] <= 4].copy()
    sprints["session_type"] = "S"
    sprints["RaceDate"] = sprints["RaceDate"] - pd.Timedelta(days=1)
    return pd.concat([labelled_results, sprints], ignore_index=True)


def test_driver_race_key_stays_unique_with_sprints(sprint_results) -> None:
    featured = build_history_features(sprint_results)
    counts = featured.groupby(
        ["Year", "RoundNumber", "session_type", "DriverId"], observed=True
    ).size()
    assert counts.max() == 1


@pytest.mark.parametrize("round_number", [1, 3, 4, 7, 11])
@pytest.mark.parametrize("session", ["R", "S"])
def test_no_leakage_on_sprint_weekends(sprint_results, round_number, session) -> None:
    present = (
        (sprint_results["RoundNumber"] == round_number)
        & (sprint_results["session_type"] == session)
    ).any()
    if not present:
        pytest.skip(f"round {round_number} has no {session} session")
    report = detect_target_leakage(
        sprint_results, flip_round=round_number, flip_session=session
    )
    assert report.empty, report.to_string(index=False)


def test_a_race_may_learn_from_its_own_weekend_sprint(sprint_results) -> None:
    """Saturday informing Sunday is history, not leakage.

    This is the distinction that made the first detector report false
    positives: it flipped the whole round and compared on date alone, so a
    legitimate Saturday-to-Sunday dependency looked like a leak.
    """
    from src.features.labels import MECHANICAL

    frame = sprint_results.copy()
    frame["RaceDate"] = pd.to_datetime(frame["RaceDate"])
    sprint_mask = (
        (frame["Year"] == frame["Year"].max())
        & (frame["RoundNumber"] == 2)
        & (frame["session_type"] == "S")
    )
    assert sprint_mask.any()

    corrupted = frame.copy()
    corrupted.loc[sprint_mask, "dnf"] = 1 - corrupted.loc[sprint_mask, "dnf"]
    corrupted.loc[sprint_mask, "dnf_cause"] = MECHANICAL

    baseline = build_history_features(frame)
    altered = build_history_features(corrupted)

    race_mask = (
        (baseline["Year"] == baseline["Year"].max())
        & (baseline["RoundNumber"] == 2)
        & (baseline["session_type"] == "R")
    )
    changed = (
        baseline.loc[race_mask, "driver_dnf_rate_5"].fillna(-9).to_numpy()
        != altered.loc[race_mask, "driver_dnf_rate_5"].fillna(-9).to_numpy()
    )
    assert changed.any(), "the Sunday race should see Saturday's sprint"


def test_event_ordinal_separates_sprint_from_race(sprint_results) -> None:
    from src.features.build_features import event_ordinal

    ordinals = event_ordinal(sprint_results)
    weekend = sprint_results.assign(_ord=ordinals)
    weekend = weekend.loc[
        (weekend["Year"] == weekend["Year"].max()) & (weekend["RoundNumber"] == 2)
    ]
    sprint_ord = weekend.loc[weekend["session_type"] == "S", "_ord"].unique()
    race_ord = weekend.loc[weekend["session_type"] == "R", "_ord"].unique()
    assert len(sprint_ord) == 1 and len(race_ord) == 1
    assert sprint_ord[0] < race_ord[0]  # Saturday before Sunday


# --------------------------------------------------------------------------- #
# Team competitiveness, team-mate comparison and field composition
# --------------------------------------------------------------------------- #
#
# These read Points, Position and GridPosition -- all of which describe the
# current race -- so they are the builders most able to leak. Each column below
# must depend only on races strictly before the one it sits on.


NEW_TEAM_FEATURES = (
    "team_points_rate_5",
    "team_points_rate_10",
    "team_points_rate_career",
    "team_form_delta",
    "team_avg_finish_5",
    "team_best_finish_5",
    "team_avg_grid_5",
    "team_avg_grid_10",
)

NEW_FIELD_FEATURES = (
    "field_dnf_rate_mean",
    "driver_dnf_rate_vs_field",
    "field_pace_spread",
    "team_rank_in_field",
    "season_progress",
)


def test_new_features_are_all_present(labelled_results: pd.DataFrame) -> None:
    built = build_history_features(labelled_results)
    expected = (
        *NEW_TEAM_FEATURES,
        *NEW_FIELD_FEATURES,
        "teammate_grid_delta",
        "driver_grid_vs_teammate_5",
    )
    missing = [c for c in expected if c not in built.columns]
    assert not missing, missing


def test_team_performance_does_not_read_the_current_race(
    labelled_results: pd.DataFrame,
) -> None:
    """Rewrite every Points value; the prior-race features must not move."""
    baseline = build_history_features(labelled_results)

    perturbed = labelled_results.copy()
    last = perturbed["RaceDate"] == perturbed["RaceDate"].max()
    perturbed.loc[last, "Points"] = 999.0
    after = build_history_features(perturbed)

    for column in NEW_TEAM_FEATURES:
        pd.testing.assert_series_equal(
            baseline.loc[last.to_numpy(), column],
            after.loc[last.to_numpy(), column],
            check_names=False,
            obj=column,
        )


def test_team_points_rate_matches_a_hand_computation() -> None:
    """Two teams, four races: the rolling mean is checked against arithmetic."""
    rows = []
    for rnd, points in enumerate([(10.0, 6.0), (20.0, 2.0), (0.0, 8.0), (4.0, 4.0)], 1):
        for team, pts in zip(("alpha", "beta"), points):
            for i in range(2):  # two cars per team
                rows.append(
                    {
                        "Year": 2022,
                        "RoundNumber": rnd,
                        "RaceDate": pd.Timestamp("2022-03-01")
                        + pd.Timedelta(days=14 * rnd),
                        "session_type": "R",
                        "DriverId": f"{team}_{i}",
                        "TeamId": team,
                        "Status": "Finished",
                        "ClassifiedPosition": "1",
                        "Points": pts / 2,
                        "Position": 1.0,
                        "GridPosition": 1.0,
                    }
                )
    frame = add_race_outcome_labels(pd.DataFrame(rows), warn_on_unmapped=False)
    built = build_history_features(frame)

    alpha = built.loc[built["TeamId"] == "alpha"].sort_values("RoundNumber")
    # Team totals per race are 10, 20, 0, 4.  Round 1 has no prior race.
    assert alpha.loc[alpha["RoundNumber"] == 1, "team_points_rate_5"].isna().all()
    assert alpha.loc[alpha["RoundNumber"] == 2, "team_points_rate_5"].eq(10.0).all()
    assert alpha.loc[alpha["RoundNumber"] == 3, "team_points_rate_5"].eq(15.0).all()
    assert alpha.loc[alpha["RoundNumber"] == 4, "team_points_rate_5"].eq(10.0).all()


def test_teammate_grid_delta_is_antisymmetric(labelled_results: pd.DataFrame) -> None:
    """Two cars, one team: whatever one gains the other loses."""
    built = build_history_features(labelled_results)
    pairs = built.dropna(subset=["teammate_grid_delta"])
    totals = pairs.groupby(
        ["Year", "RoundNumber", "TeamId"], observed=True
    )["teammate_grid_delta"].sum()
    assert np.allclose(totals.to_numpy(), 0.0), "deltas within a team must cancel"


def test_teammate_delta_is_nan_for_a_single_car_entry() -> None:
    """A one-car team has nothing to compare against; NaN, not zero."""
    rows = [
        {
            "Year": 2022, "RoundNumber": rnd,
            "RaceDate": pd.Timestamp("2022-03-01") + pd.Timedelta(days=14 * rnd),
            "session_type": "R", "DriverId": "solo", "TeamId": "lonely",
            "Status": "Finished", "ClassifiedPosition": "1",
            "Points": 1.0, "Position": 1.0, "GridPosition": 5.0,
        }
        for rnd in (1, 2, 3)
    ]
    frame = add_race_outcome_labels(pd.DataFrame(rows), warn_on_unmapped=False)
    built = build_history_features(frame)
    assert built["teammate_grid_delta"].isna().all()


def test_field_context_reads_only_prior_race_features(
    labelled_results: pd.DataFrame,
) -> None:
    """Field aggregates are built from rolling columns, so outcomes cannot move them."""
    baseline = build_history_features(labelled_results)

    perturbed = labelled_results.copy()
    last = perturbed["RaceDate"] == perturbed["RaceDate"].max()
    perturbed.loc[last, "dnf"] = 1 - perturbed.loc[last, "dnf"]
    after = build_history_features(perturbed)

    for column in NEW_FIELD_FEATURES:
        pd.testing.assert_series_equal(
            baseline.loc[last.to_numpy(), column],
            after.loc[last.to_numpy(), column],
            check_names=False,
            obj=column,
        )


def test_season_progress_spans_the_season(labelled_results: pd.DataFrame) -> None:
    built = build_history_features(labelled_results)
    per_year = built.groupby("Year", observed=True)["season_progress"]
    assert per_year.max().eq(1.0).all(), "the final round must be 1.0"
    assert (per_year.min() > 0).all()


def test_team_rank_in_field_is_a_percentile(labelled_results: pd.DataFrame) -> None:
    built = build_history_features(labelled_results)
    rank = built["team_rank_in_field"].dropna()
    assert rank.between(0, 1).all()


def test_new_features_survive_the_full_leakage_detector(
    labelled_results: pd.DataFrame,
) -> None:
    """The shipped detector, run over a frame that now carries the new columns."""
    built = build_history_features(labelled_results)
    report = detect_target_leakage(built)
    flagged = set(report["feature"]) if not report.empty else set()
    overlap = flagged.intersection(
        {*NEW_TEAM_FEATURES, *NEW_FIELD_FEATURES,
         "teammate_grid_delta", "driver_grid_vs_teammate_5"}
    )
    assert not overlap, sorted(overlap)
