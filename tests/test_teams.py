"""A team that is renamed is still the same team, and the table has to say so.

The tests that matter here are the ones about *refusing*: a lookup that
silently guesses which Lotus you meant, or that quietly hands back an empty
history for a constructor it has never heard of, is worse than no lookup at
all, because the failure is invisible in the modelling frame.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import teams


# --- the table itself -------------------------------------------------------

def test_table_is_internally_consistent() -> None:
    """Every structural invariant at once; the message names what broke."""
    assert teams.validate() == []


def test_lineage_id_matches_its_latest_entry() -> None:
    """A lineage is named after the team currently racing it.

    This is the invariant that stops the naming scheme going stale silently.
    When Aston Martin is next renamed, this fails, and the fix is to rename
    the lineage -- not to leave a lineage called ``aston_martin`` whose newest
    constructor is something else.
    """
    for lineage_id in teams.LINEAGES:
        chain = teams._lineage_entries(lineage_id)
        assert chain[-1].key == lineage_id, (
            f"{lineage_id} now ends at {chain[-1].key}"
        )


def test_every_chain_terminates_in_a_new_constructor() -> None:
    """Follow any lineage far enough back and someone founded it."""
    for lineage_id in teams.LINEAGES:
        oldest = teams._lineage_entries(lineage_id)[0]
        assert oldest.succession == teams.NEW
        assert oldest.succeeds is None


@pytest.mark.parametrize(
    "season, expected",
    [
        (2006, 11),  # Super Aguri's first season, MF1's only one
        (2009, 10),  # Super Aguri gone, Honda is now Brawn
        (2010, 12),  # the three-team intake under the abandoned budget cap
        (2013, 11),  # HRT left off the entry list
        (2015, 10),  # Caterham and Marussia both collapsed over the winter
        (2016, 11),  # Haas arrives
        (2017, 10),  # Manor folds; the grid that held until 2025
        (2026, 11),  # Cadillac arrives
    ],
)
def test_grid_size_matches_the_record(season: int, expected: int) -> None:
    """Counting entries per season is the cheapest check on the whole table.

    A missing constructor, a span off by a year, or a rename recorded as a new
    team all show up here as a grid of the wrong size.
    """
    assert len(teams.active_in(season)) == expected


def test_only_seven_constructors_since_2006_were_genuinely_new() -> None:
    """The premise of the module, asserted.

    Teams are bought and renamed; they are hardly ever founded. If this count
    moves, either the sport did something unusual or the table has recorded a
    rename as a birth -- which is the mistake that resets a rolling history.
    """
    born = {
        e.key
        for e in teams.ENTRIES
        if e.succession == teams.NEW and e.first_season >= 2002
    }
    assert born == {
        "toyota", "super_aguri", "lotus_racing", "virgin", "hrt", "haas", "cadillac",
    }


# --- the Lotus problem ------------------------------------------------------

def test_lotus_without_a_season_is_refused() -> None:
    """Two unrelated teams raced as Lotus in 2011. There is no right guess."""
    with pytest.raises(teams.AmbiguousTeamError):
        teams.resolve("Lotus")


@pytest.mark.parametrize(
    "season, key, lineage",
    [
        (2010, "lotus_racing", "caterham"),
        (2011, "team_lotus", "caterham"),
        (2012, "lotus_f1", "alpine"),
        (2015, "lotus_f1", "alpine"),
    ],
)
def test_lotus_resolves_by_season(season: int, key: str, lineage: str) -> None:
    """The plain name moved between two lineages in 2012, so the season decides.

    Getting this backwards would credit Caterham's record to Alpine's ancestor
    and vice versa.
    """
    entry = teams.resolve("Lotus", season)
    assert (entry.key, entry.lineage_id) == (key, lineage)


def test_the_two_lotus_teams_of_2011_are_told_apart_by_their_full_names() -> None:
    assert teams.resolve("Team Lotus", 2011).lineage_id == "caterham"
    assert teams.resolve("Lotus Renault GP", 2011).lineage_id == "alpine"


# --- resolving real identifiers ---------------------------------------------

@pytest.mark.parametrize(
    "name, season, key",
    [
        # Sponsor-laden display names, as FastF1 reports them.
        ("Oracle Red Bull Racing", 2026, "red_bull"),
        ("Scuderia Ferrari HP", 2026, "ferrari"),
        ("Mercedes-AMG Petronas F1 Team", 2020, "mercedes"),
        ("Stake F1 Team Kick Sauber", 2024, "kick_sauber"),
        ("Aston Martin Aramco Cognizant F1 Team", 2022, "aston_martin"),
        ("Visa Cash App Racing Bulls Formula One Team", 2026, "racing_bulls"),
        ("Atlassian Williams F1 Team", 2026, "williams"),
        ("Audi Revolut F1 Team", 2026, "audi"),
        # Bare constructor ids, as the results backend reports them.
        ("toro_rosso", 2010, "toro_rosso"),
        ("force_india", 2016, "force_india"),
        ("alphatauri", 2021, "alphatauri"),
        # Transitional names, where the sponsor leads and the constructor
        # follows -- the cases the usual word order does not cover.
        ("Racing Point Force India F1 Team", 2018, "force_india"),
        ("Marussia Virgin Racing", 2011, "virgin"),
        ("Manor Marussia", 2015, "marussia"),
    ],
)
def test_resolves_names_as_they_actually_arrive(name: str, season: int, key: str) -> None:
    assert teams.resolve(name, season).key == key


def test_an_engine_supplier_in_the_name_does_not_win() -> None:
    """F1 names the constructor first and the engine after it.

    "McLaren Honda" is a McLaren. Reading it as a Honda would merge Woking's
    record into Brackley's.
    """
    assert teams.resolve("McLaren Honda", 2016).key == "mclaren"
    assert teams.resolve("Aston Martin Aramco Honda", 2026).key == "aston_martin"
    assert teams.resolve("Red Bull Ford", 2026).key == "red_bull"


# --- carrying a name across a rename ----------------------------------------

def test_a_stale_constructor_id_still_resolves() -> None:
    """The failure mode this is really for.

    The results backend has kept a constructor id across a rebrand before, so
    a 2024 row can arrive tagged ``sauber`` years after that name left the
    entry list. Answering "no such team" would drop the row's history.
    """
    assert teams.resolve("sauber", 2024).key == "kick_sauber"
    assert teams.resolve("alfa", 2026).key == "audi"
    assert teams.resolve("rb", 2025).key == "racing_bulls"


def test_a_gap_inside_a_names_use_is_filled() -> None:
    """Sauber raced 1993-2005 and again 2011-2018; the gap is still Hinwil.

    Likewise Renault, whose Enstone team spent 2012-2015 as Lotus.
    """
    assert teams.resolve("sauber", 2008).key == "bmw_sauber"
    assert teams.resolve("renault", 2013).key == "lotus_f1"


def test_a_name_is_never_carried_backwards() -> None:
    """A current name on an old season is a mislabelling, not a rename.

    The 2015 Enstone car was a Lotus. Calling it an Alpine would invent a
    history that the championship, and the car, never had.
    """
    for name, season in [("Alpine", 2015), ("Mercedes", 2006), ("Aston Martin", 2018)]:
        with pytest.raises(teams.UnknownTeamError):
            teams.resolve(name, season)


def test_an_unknown_team_raises_rather_than_returning_nothing() -> None:
    """In a modelling frame, a null lineage is a team with no history."""
    with pytest.raises(teams.UnknownTeamError):
        teams.resolve("Bloggs Grand Prix", 2020)


# --- the lineages the module exists to bridge -------------------------------

@pytest.mark.parametrize(
    "lineage_id, members",
    [
        (
            "aston_martin",
            [("jordan", 2005), ("mf1", 2006), ("spyker", 2007),
             ("force_india", 2015), ("racing_point", 2019), ("aston_martin", 2026)],
        ),
        (
            "audi",
            [("sauber", 2000), ("bmw_sauber", 2008), ("sauber", 2016),
             ("alfa", 2021), ("kick_sauber", 2024), ("audi", 2026)],
        ),
        (
            "alpine",
            [("benetton", 1995), ("renault", 2006), ("lotus_f1", 2013),
             ("renault", 2018), ("alpine", 2026)],
        ),
        (
            "mercedes",
            [("tyrrell", 1990), ("bar", 2002), ("honda", 2007),
             ("brawn", 2009), ("mercedes", 2026)],
        ),
        (
            "racing_bulls",
            [("minardi", 2000), ("toro_rosso", 2010), ("alphatauri", 2021),
             ("rb", 2024), ("racing_bulls", 2026)],
        ),
        ("red_bull", [("stewart", 1998), ("jaguar", 2002), ("red_bull", 2026)]),
    ],
)
def test_lineages_bridge_every_rename(lineage_id: str, members: list) -> None:
    """The whole point: these names all denote one continuous operation."""
    for key, season in members:
        assert teams.resolve(key, season).lineage_id == lineage_id


# --- the grading, which is the part a modeller has to choose ----------------

def test_only_three_successions_broke_the_sporting_record() -> None:
    """Insolvency is rare, and it is the one thing that resets a team's record.

    Brawn, Racing Point and Manor Marussia each continued an operation through
    an administration as a new entity. Everything else in the table is a sale
    or a new coat of paint.
    """
    broken = {
        (e.key, e.first_season)
        for e in teams.ENTRIES
        if e.succession == teams.RECONSTITUTED
    }
    assert broken == {("brawn", 2009), ("racing_point", 2019), ("marussia", 2015)}


def test_era_splits_a_lineage_at_insolvency_and_nowhere_else() -> None:
    """``lineage_era`` is the conservative grouping: what the sport recognises.

    Force India and Racing Point are the same factory and the same people, but
    the championship struck the constructor's points in 2018 and restarted
    them. Grouping on the lineage alone carries that history; grouping on
    ``(lineage_id, lineage_era)`` does not.
    """
    assert teams.era_of(teams.resolve("force_india", 2018)) == 0
    assert teams.era_of(teams.resolve("racing_point", 2019)) == 1
    assert teams.era_of(teams.resolve("aston_martin", 2026)) == 1
    # A rename and a sale both keep the record intact.
    assert teams.era_of(teams.resolve("jordan", 2000)) == 0
    assert teams.era_of(teams.resolve("spyker", 2007)) == 0
    # A lineage that never went under stays in one era throughout.
    assert {teams.era_of(e) for e in teams._lineage_entries("audi")} == {0}


# --- the pandas surface -----------------------------------------------------

def test_attach_lineage_bridges_a_rename_in_a_frame() -> None:
    """The join a feature builder would actually make."""
    frame = pd.DataFrame(
        {
            "Year": [2018, 2019, 2021, 2026],
            "TeamId": ["force_india", "racing_point", "aston_martin", "aston_martin"],
        }
    )
    out = teams.attach_lineage(frame)
    assert list(out["lineage_id"]) == ["aston_martin"] * 4
    assert list(out["lineage_era"]) == [0, 1, 1, 1]
    assert list(out["history_carries"]) == [True, False, True, True]


def test_attach_lineage_keeps_every_row_and_column() -> None:
    frame = pd.DataFrame(
        {"Year": [2026, 2026], "TeamId": ["audi", "cadillac"], "dnf": [0, 1]}
    )
    out = teams.attach_lineage(frame)
    assert len(out) == len(frame)
    assert list(out["dnf"]) == [0, 1]
    assert "Year" in out.columns and "TeamId" in out.columns


def test_attach_lineage_warns_about_an_unknown_team_but_keeps_the_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mirrors circuits.attach_reference: a new team is visible, not silent."""
    frame = pd.DataFrame({"Year": [2026, 2026], "TeamId": ["audi", "Bloggs GP"]})
    with caplog.at_level("WARNING"):
        out = teams.attach_lineage(frame)
    assert "Bloggs GP" in caplog.text
    assert out.loc[0, "lineage_id"] == "audi"
    assert pd.isna(out.loc[1, "lineage_id"])


def test_attach_lineage_can_refuse_instead() -> None:
    """What a build step should do: fail rather than write a null history."""
    frame = pd.DataFrame({"Year": [2026], "TeamId": ["Bloggs GP"]})
    with pytest.raises(teams.UnknownTeamError):
        teams.attach_lineage(frame, strict=True)


def test_attach_lineage_requires_the_columns_it_is_told_to_use() -> None:
    with pytest.raises(KeyError):
        teams.attach_lineage(pd.DataFrame({"Year": [2026]}), team_col="TeamId")


def test_frames_cover_the_table() -> None:
    entries, lineages = teams.entries_frame(), teams.lineage_frame()
    assert len(entries) == len(teams.ENTRIES)
    assert len(lineages) == len(teams.LINEAGES)
    assert set(lineages["lineage_id"]) == set(entries["lineage_id"])
    # Eight lineages raced under more than one name, which is what this module
    # is for. The other eight never changed theirs -- and five of those eight
    # are teams that came and went inside a few seasons.
    multi = set(lineages.loc[lineages["n_names"] > 1, "lineage_id"])
    assert multi == {
        "alpine", "aston_martin", "audi", "caterham",
        "manor", "mercedes", "racing_bulls", "red_bull",
    }


def test_renames_between_is_scoped_to_the_window() -> None:
    renames = teams.renames_between(2006, 2026)
    assert renames["season"].between(2006, 2026).all()
    # The Silverstone team alone changed its constructor name five times.
    assert (renames["lineage_id"] == "aston_martin").sum() == 5
