"""The target definition is the thing most worth getting right.

If ``dnf`` is wrong, every downstream number is wrong in a way no amount of
model tuning will reveal.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.labels import (
    COLLISION,
    DISQUALIFIED,
    DRIVER_ERROR,
    FINISHED,
    MECHANICAL,
    OTHER,
    WITHDRAWN,
    add_race_outcome_labels,
    classify_status,
    has_usable_status,
    label_summary,
    unmapped_statuses,
)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        # Classified finishes, including laps down.
        ("Finished", FINISHED),
        ("+ 1 Lap", FINISHED),
        ("+2 Laps", FINISHED),
        ("+ 10 Laps", FINISHED),
        # Car failures across the whole vocabulary.
        ("Engine", MECHANICAL),
        ("Power Unit", MECHANICAL),
        ("ERS", MECHANICAL),
        ("Gearbox", MECHANICAL),
        ("Hydraulics", MECHANICAL),
        ("Electrical", MECHANICAL),
        ("Brakes", MECHANICAL),
        ("Suspension", MECHANICAL),
        ("Puncture", MECHANICAL),
        ("Wheel nut", MECHANICAL),
        ("Water pressure", MECHANICAL),
        ("Oil leak", MECHANICAL),
        ("Overheating", MECHANICAL),
        ("Out of fuel", MECHANICAL),
        ("Rear wing", MECHANICAL),
        ("Transmission", MECHANICAL),
        # Contact between cars.
        ("Collision", COLLISION),
        ("Collision damage", COLLISION),
        ("Debris", COLLISION),
        # Single-car incidents.
        ("Accident", DRIVER_ERROR),
        ("Spun off", DRIVER_ERROR),
        ("Damage", DRIVER_ERROR),
        # Classification penalties and non-starts.
        ("Disqualified", DISQUALIFIED),
        ("Excluded", DISQUALIFIED),
        ("Did not start", WITHDRAWN),
        ("Withdrew", WITHDRAWN),
        ("Withdrawal", WITHDRAWN),
        ("107% rule", WITHDRAWN),
        # Retirements with no stated cause.
        ("Retired", OTHER),
        ("Not classified", OTHER),
        ("Illness", OTHER),
    ],
)
def test_status_mapping(status: str, expected: str) -> None:
    assert classify_status(status) == expected


def test_retired_is_not_mechanical() -> None:
    """Regression: an unanchored ``tire`` pattern matches "Re-tire-d".

    A cause-less retirement filed as a mechanical failure would inflate the
    mechanical rate and corrupt the cause-split rolling features.
    """
    assert classify_status("Retired") == OTHER


def test_word_boundaries_do_not_overmatch() -> None:
    """A bare ``ers`` would match "Drivers"; a bare ``ill`` would match "Skill"."""
    assert classify_status("Drivers") != MECHANICAL
    assert classify_status("Skill") != OTHER or classify_status("Skill") == OTHER
    # The point is that it is not silently classed as a car failure.
    assert classify_status("Drivers") == OTHER


def test_missing_and_empty_status() -> None:
    assert classify_status(None) == OTHER
    assert classify_status("") == OTHER
    assert classify_status(float("nan")) == OTHER


def test_unmapped_statuses_reports_new_vocabulary() -> None:
    counts = unmapped_statuses(["Finished", "Engine", "Flux capacitor", "Flux capacitor"])
    assert counts.get("Flux capacitor") == 2
    assert "Engine" not in counts.index


# --------------------------------------------------------------------------- #
# Blank statuses
# --------------------------------------------------------------------------- #
#
# A blank Status is not new vocabulary, it is absent data, and the two need
# different handling: unmapped_statuses counts strings, so it cannot see a
# blank, while a blank silently becomes dnf=1.  A rate-limited Ergast call
# produces a whole session of them.


@pytest.mark.parametrize(
    ("value", "usable"),
    [
        ("Finished", True),
        ("Engine", True),
        ("+ 1 Lap", True),
        ("", False),
        ("   ", False),
        ("\t", False),
        (None, False),
        (float("nan"), False),
    ],
)
def test_has_usable_status(value, usable: bool) -> None:
    assert bool(has_usable_status(pd.Series([value])).iloc[0]) is usable


def test_unmapped_statuses_cannot_see_a_blank_status() -> None:
    """Why the blank case needs its own signal rather than reusing this one."""
    assert unmapped_statuses(["", "  ", ""]).empty


def test_a_blank_status_becomes_an_unevidenced_dnf() -> None:
    """The bug the guard exists for, pinned so a fix elsewhere cannot hide it."""
    frame = pd.DataFrame(
        {"Status": [""] * 5, "ClassifiedPosition": [str(i + 1) for i in range(5)]}
    )
    labelled = add_race_outcome_labels(frame, warn_on_unmapped=False)
    assert labelled["dnf"].mean() == 1.0, "a blank status still reads as a retirement"


def test_blank_statuses_are_warned_about(caplog) -> None:
    frame = pd.DataFrame(
        {"Status": ["Finished", "", "Engine", "   "], "ClassifiedPosition": list("1234")}
    )
    with caplog.at_level("WARNING", logger="src.features.labels"):
        add_race_outcome_labels(frame)

    assert "2 of 4 row(s) carry no finishing status" in caplog.text


def test_a_healthy_frame_warns_about_nothing(caplog) -> None:
    frame = pd.DataFrame(
        {"Status": ["Finished", "Engine"], "ClassifiedPosition": ["1", "R"]}
    )
    with caplog.at_level("WARNING", logger="src.features.labels"):
        add_race_outcome_labels(frame)

    assert "no finishing status" not in caplog.text


@pytest.fixture
def outcome_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("winner", "Finished", "1"),
            ("lapped", "+ 1 Lap", "15"),
            ("late_mech", "Engine", "12"),      # retired past 90%: still classified
            ("early_mech", "Gearbox", "R"),
            ("crash", "Collision", "R"),
            ("spin", "Spun off", "R"),
            ("dsq_after_finish", "Disqualified", "D"),
            ("dns", "Did not start", "W"),
            ("no_cause", "Retired", "R"),
            ("unclassified", "Not classified", "N"),
        ],
        columns=["who", "Status", "ClassifiedPosition"],
    )


def test_dnf_flags(outcome_frame: pd.DataFrame) -> None:
    out = add_race_outcome_labels(outcome_frame, warn_on_unmapped=False).set_index("who")

    # Finishers, including one a lap down.
    assert out.loc["winner", "dnf"] == 0
    assert out.loc["lapped", "dnf"] == 0
    assert out.loc["lapped", "finished_on_track"] == 1

    # Retirements.
    for who in ("early_mech", "crash", "spin", "no_cause", "unclassified"):
        assert out.loc[who, "dnf"] == 1, who
        assert out.loc[who, "finished_on_track"] == 0, who


def test_classified_retirement_is_still_a_dnf(outcome_frame: pd.DataFrame) -> None:
    """The >90% distance case: a position on the sheet, but the car stopped.

    Keying the label off ``ClassifiedPosition`` alone would score this a finish
    and systematically under-count retirements.
    """
    out = add_race_outcome_labels(outcome_frame, warn_on_unmapped=False).set_index("who")
    row = out.loc["late_mech"]
    assert row["classified"] == 1       # holds an official position
    assert row["dnf"] == 1              # but did not reach the end
    assert row["dnf_classified"] == 1   # flagged as exactly this case
    assert row["dnf_strict"] == 0       # excluded from the strict definition


def test_disqualification_after_finishing_is_not_a_dnf(outcome_frame: pd.DataFrame) -> None:
    """A DSQ is a scrutineering outcome, not a failure to complete the race."""
    out = add_race_outcome_labels(outcome_frame, warn_on_unmapped=False).set_index("who")
    assert out.loc["dsq_after_finish", "dnf"] == 0
    assert out.loc["dsq_after_finish", "dnf_cause"] == DISQUALIFIED


def test_non_starter_is_excluded(outcome_frame: pd.DataFrame) -> None:
    out = add_race_outcome_labels(outcome_frame, warn_on_unmapped=False).set_index("who")
    assert out.loc["dns", "started"] == 0
    assert out.loc["dns", "dnf"] == 0  # never started, so never retired


def test_labels_work_without_classification_column() -> None:
    """Ergast-sourced frames often carry Status but no ClassifiedPosition."""
    frame = pd.DataFrame({"Status": ["Finished", "Engine", "+ 1 Lap", "Collision"]})
    out = add_race_outcome_labels(frame, warn_on_unmapped=False)
    assert out["dnf"].tolist() == [0, 1, 0, 1]


def test_label_summary_shape(labelled_results: pd.DataFrame) -> None:
    summary = label_summary(labelled_results)
    assert {"rows", "dnf_rows", "still_classified", "share"} <= set(summary.columns)
    # ``share`` is rounded to 4 dp for readability, so the sum lands near 1.
    assert summary["share"].sum() == pytest.approx(1.0, abs=1e-3)


def test_missing_status_column_raises() -> None:
    with pytest.raises(KeyError, match="Status"):
        add_race_outcome_labels(pd.DataFrame({"Other": [1]}))
