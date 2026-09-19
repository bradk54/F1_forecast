"""Constructor lineage: which team on today's grid is which team of ten years ago.

Formula One constructors are almost never born and almost never die.  They are
bought, renamed, re-liveried and sold on, while the same few hundred people
keep going to the same building.  The Silverstone factory that races as Aston
Martin today was Jordan in 1991, Midland in 2006, Spyker in 2007, Force India
from 2008 and Racing Point in 2019 -- one continuous operation under six
constructor names.  Six rows in a results table; one team.

That matters here because every team feature in
:mod:`src.features.build_features` is a rolling history keyed on ``TeamId``
(``team_dnf_rate_10``, ``team_points_rate_career``, ``team_races_to_date``).
A rename resets all of them to zero.  The 2021 season opens with Aston Martin
and Alpine apparently having never entered a race, when in fact both had
decades of form; ``is_new_pairing`` fires for twenty untouched driver-team
combinations; ``team_rank_in_field`` ranks two established midfield cars
against a field of strangers.  The reset lands at exactly the moment the
rolling window is most valuable -- the start of a season -- and nothing in the
pipeline reports it, because a team with no history is indistinguishable from
a team that is genuinely new.

This module is the bridge.  It is reference data in the same sense as
:mod:`src.data.circuits`: published fact about the sport that no amount of
telemetry will tell you, hand-entered, with each entry naming its basis.

What it deliberately does *not* do
----------------------------------
It does not decide **for** you that history should carry across a rename.
That is a modelling assumption and a contestable one, so the table grades every
succession rather than flattening it to a boolean:

``rebrand``
    A new name over an unchanged legal entity, factory and staff.  Sauber to
    Alfa Romeo in 2019 was a title-sponsorship deal; the cars were still built
    in Hinwil by the same people, to the same design lineage.  Carrying history
    across a rebrand is close to free.

``takeover``
    Ownership changed and the operation continued.  Red Bull bought Jaguar,
    which had bought Stewart.  The building and most of the staff persist, but
    a new owner usually brings money, and money is the main driver of a team's
    reliability.  History carries, but the level may step.

``reconstituted``
    The operation continued *through* an insolvency, as a new legal entity.
    This is the one to be careful with.  When Force India went into
    administration in July 2018 its FIA entry was not transferred: the
    championship struck the constructor's points and restarted it from zero
    mid-season, and Racing Point's 2020 Sakhir win is officially their *first*.
    The sport itself declines to carry the history across that boundary.

``new``
    No predecessor.  Rare, and worth knowing precisely: across the twenty-one
    seasons covered here only seven constructors were genuinely new -- Toyota
    (2002), Super Aguri (2006), the three 2010 entrants brought in by the
    abandoned budget-cap rules, Haas (2016), and Cadillac (2026).

:func:`attach_lineage` exposes both readings, so a feature can be built on
either and the choice is recorded rather than assumed:

* ``lineage_id`` groups by continuous *operation* -- the factory and the
  people.  Aston Martin's lineage reaches back to Jordan 1991 unbroken.
* ``lineage_era`` additionally splits at every ``reconstituted`` boundary, so
  grouping on ``(lineage_id, lineage_era)`` reproduces what the championship
  itself treats as one continuous record.

Scope
-----
Complete for every constructor that started a World Championship race from
**2006 to 2026** inclusive.  Earlier entries appear where they are the ancestor
of an in-window chain (Tyrrell, Toleman, Benetton, Jordan, Minardi, BAR,
Stewart, Jaguar, Sauber's first spell), so that every chain terminates in a
``new`` constructor rather than trailing off.

Two traps this table exists to catch
------------------------------------
* **"Lotus" in 2011 and 2012 are different teams.**  In 2011 the grid carried
  both Team Lotus (Tony Fernandes, Hingham -- later Caterham) and Lotus Renault
  GP (Enstone -- later Alpine), and in 2012 the Enstone team took the plain
  "Lotus" name the Hingham team had just given up.  A lookup keyed on the
  displayed name alone gets this backwards.  :func:`resolve` refuses to guess:
  a bare ``"Lotus"`` without a season raises :class:`AmbiguousTeamError`.
* **An identifier can outlive the name it stood for.**  The Ergast-compatible
  backend has historically kept a constructor id across a rebrand, so a 2024
  row can arrive tagged ``sauber`` years after the Sauber name left the entry
  list.  :func:`resolve` carries such a name *forward* along its lineage to the
  constructor that was actually racing in the season asked for.  It never
  carries a name backwards, because a current name applied to an old season is
  a mislabelling, not a rename.

Sources
-------
Wikipedia's constructor histories, which are themselves sourced to the FIA
entry lists and contemporaneous reporting: *List of Formula One constructors*,
*Team Enstone*, *Sauber Motorsport*, *Racing Point F1 Team*, *Brawn GP*,
*Scuderia Toro Rosso*, *Racing Bulls*, *Marussia F1*, *Caterham F1*,
*HRT Formula 1 Team*, *Cadillac in Formula One*, *Audi in Formula One*.
Each entry's ``note`` records the specific fact and the year it happened.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

# --- succession grades ------------------------------------------------------
#: No predecessor: a constructor that had never entered a race before.
NEW = "new"
#: A new name over an unchanged entity.  History carries almost for free.
REBRAND = "rebrand"
#: New owner, continuing operation.  History carries; the level may step.
TAKEOVER = "takeover"
#: Continued through insolvency as a new entity.  The championship restarted
#: the constructor's record here, and so should you unless you mean not to.
RECONSTITUTED = "reconstituted"

SUCCESSION_GRADES = (NEW, REBRAND, TAKEOVER, RECONSTITUTED)


class UnknownTeamError(KeyError):
    """No entry matched the name given.

    Raised rather than returning a null lineage, because an unmatched team in
    a modelling frame is silently a team with no history -- the exact failure
    this module exists to prevent.  A genuinely new constructor should be added
    to :data:`ENTRIES`; there is roughly one every decade.
    """


class AmbiguousTeamError(KeyError):
    """The name matched more than one constructor and no season was given.

    The sport really has run two unrelated teams under one name at once (see
    the module docstring on "Lotus"), so this is a question the caller has to
    answer rather than a defect to tune away.
    """


@dataclass(frozen=True)
class TeamEntry:
    """One constructor name, over the seasons it was used.

    A constructor gets a second entry rather than a widened span when it left
    the entry list and came back -- ``sauber`` raced 1993-2005, handed the name
    to BMW Sauber for five seasons, and took it back for 2011-2018.  Treating
    that as one 1993-2018 span would claim the team raced under the Sauber name
    in 2008, which it did not.
    """

    #: Canonical slug.  Chosen to match the Ergast/Jolpica ``constructorId``
    #: where that is known, since ``TeamId`` is the column this joins against.
    #: Matching is never *only* on this string -- see :func:`resolve`.
    key: str
    #: The constructor name as the championship recorded it.
    name: str
    first_season: int
    #: ``None`` means still racing.  Deliberately open-ended rather than set to
    #: the current year, so the table does not go stale every January.
    last_season: int | None
    lineage_id: str
    #: The ``key`` this entry took over from, or ``None`` for a new constructor.
    succeeds: str | None
    succession: str
    #: What happened, and when.  This is the citation.
    note: str

    def covers(self, season: int) -> bool:
        if season < self.first_season:
            return False
        return self.last_season is None or season <= self.last_season

    @property
    def span(self) -> str:
        """``"2008-2018"``, or ``"2026-"`` for a team still racing."""
        if self.last_season is None:
            return f"{self.first_season}-"
        if self.last_season == self.first_season:
            return str(self.first_season)
        return f"{self.first_season}-{self.last_season}"


#: Every constructor entry, oldest first within each lineage.
#:
#: ``lineage_id`` names a lineage after its most recent constructor, because
#: that is the name a reader will recognise in a feature table.  The cost is
#: that it changes when the current team is renamed; :data:`LINEAGES` also
#: carries the factory town, which is the thing that actually persists, and
#: ``test_lineage_id_matches_its_latest_entry`` fails the build if the two ever
#: drift apart.
ENTRIES: tuple[TeamEntry, ...] = (
    # --- Maranello: Ferrari ------------------------------------------------
    TeamEntry(
        "ferrari", "Ferrari", 1950, None, "ferrari", None, NEW,
        "Present at the first World Championship round in 1950 and every "
        "season since; the only constructor never to have been sold.",
    ),
    # --- Woking: McLaren ---------------------------------------------------
    TeamEntry(
        "mclaren", "McLaren", 1966, None, "mclaren", None, NEW,
        "Founded by Bruce McLaren, 1963; first entered as a constructor in "
        "1966.  Ownership has changed hands repeatedly without the entry or "
        "the name ever lapsing.",
    ),
    # --- Grove: Williams ---------------------------------------------------
    TeamEntry(
        "williams", "Williams", 1977, None, "williams", None, NEW,
        "Williams Grand Prix Engineering, founded 1977.  Sold to Dorilton "
        "Capital in August 2020, ending the founding family's involvement, "
        "but the entity, the entry and the name all continued unbroken.",
    ),
    # --- Brackley: Tyrrell -> BAR -> Honda -> Brawn -> Mercedes ------------
    TeamEntry(
        "tyrrell", "Tyrrell", 1970, 1998, "mercedes", None, NEW,
        "Ken Tyrrell's team entered F1 in 1968 running others' cars and built "
        "its own from 1970.",
    ),
    TeamEntry(
        "bar", "BAR", 1999, 2005, "mercedes", "tyrrell", TAKEOVER,
        "British American Tobacco bought Tyrrell in 1997 and raced as British "
        "American Racing from 1999, from a new factory at Brackley that the "
        "team has occupied ever since.",
    ),
    TeamEntry(
        "honda", "Honda", 2006, 2008, "mercedes", "bar", TAKEOVER,
        "Honda took a 45% stake in 2004 and full control at the end of 2005, "
        "converting its engine partnership into a works team.",
    ),
    TeamEntry(
        "brawn", "Brawn", 2009, 2009, "mercedes", "honda", RECONSTITUTED,
        "Honda withdrew in December 2008 and the team was days from closure.  "
        "Ross Brawn led a management buyout for a nominal GBP 1 and it raced "
        "as Brawn GP, winning both championships in its only season.",
    ),
    TeamEntry(
        "mercedes", "Mercedes", 2010, None, "mercedes", "brawn", TAKEOVER,
        "Daimler and Aabar Investments bought 75.1% of Brawn GP in November "
        "2009 and rebranded it Mercedes GP, keeping the Brackley site and "
        "workforce.",
    ),
    # --- Milton Keynes: Stewart -> Jaguar -> Red Bull ----------------------
    TeamEntry(
        "stewart", "Stewart", 1997, 1999, "red_bull", None, NEW,
        "Founded by Jackie and Paul Stewart with Ford backing, 1997.",
    ),
    TeamEntry(
        "jaguar", "Jaguar", 2000, 2004, "red_bull", "stewart", TAKEOVER,
        "Ford bought the team outright in 1999 and rebadged it as its Jaguar "
        "marque for 2000.",
    ),
    TeamEntry(
        "red_bull", "Red Bull", 2005, None, "red_bull", "jaguar", TAKEOVER,
        "Red Bull GmbH bought the team from Ford in November 2004, having "
        "previously been a Sauber sponsor.",
    ),
    # --- Faenza: Minardi -> Toro Rosso -> AlphaTauri -> RB -> Racing Bulls -
    TeamEntry(
        "minardi", "Minardi", 1985, 2005, "racing_bulls", None, NEW,
        "Giancarlo Minardi's team, founded in Faenza in 1979, entered F1 in "
        "1985.",
    ),
    TeamEntry(
        "toro_rosso", "Toro Rosso", 2006, 2019, "racing_bulls", "minardi",
        TAKEOVER,
        "Red Bull bought Minardi from Paul Stoddart in 2005 and renamed it "
        "Scuderia Toro Rosso.  The sale required the team to stay in Faenza, "
        "where it remains.",
    ),
    TeamEntry(
        "alphatauri", "AlphaTauri", 2020, 2023, "racing_bulls", "toro_rosso",
        REBRAND,
        "Renamed on 1 December 2019 to promote Red Bull's fashion label.  "
        "Entity, factory and staff unchanged.",
    ),
    TeamEntry(
        "rb", "RB", 2024, 2024, "racing_bulls", "alphatauri", REBRAND,
        "Rebranded RB for 2024, entering as Visa Cash App RB.",
    ),
    TeamEntry(
        "racing_bulls", "Racing Bulls", 2025, None, "racing_bulls", "rb",
        REBRAND,
        "The RB initialism was expanded to Racing Bulls for 2025.",
    ),
    # --- Enstone: Toleman -> Benetton -> Renault -> Lotus -> Alpine --------
    TeamEntry(
        "toleman", "Toleman", 1981, 1985, "alpine", None, NEW,
        "Toleman Group Motorsport, 1981.  Based at Witney; the move to the "
        "Enstone factory the lineage is named for came in 1992.",
    ),
    TeamEntry(
        "benetton", "Benetton", 1986, 2001, "alpine", "toleman", TAKEOVER,
        "Benetton bought Toleman in 1985.  Two drivers' titles and a "
        "constructors' title in 1994-95.",
    ),
    TeamEntry(
        "renault", "Renault", 2002, 2011, "alpine", "benetton", TAKEOVER,
        "Renault bought Benetton in March 2000; the team ran one more season "
        "under the Benetton name and became Renault F1 Team in 2002.  It was "
        "still the Renault constructor in 2011, when it raced as Lotus "
        "Renault GP under Genii Capital ownership.",
    ),
    TeamEntry(
        "lotus_f1", "Lotus F1", 2012, 2015, "alpine", "renault", REBRAND,
        "Genii-owned and branded for Group Lotus.  The name only became "
        "available when the unrelated Hingham team gave it up to become "
        "Caterham, which is why the same word means two different teams "
        "either side of 2012.",
    ),
    TeamEntry(
        "renault", "Renault", 2016, 2020, "alpine", "lotus_f1", TAKEOVER,
        "Renault repurchased a 90% stake in the Enstone team from Genii at "
        "the end of 2015 and restored the works name.",
    ),
    TeamEntry(
        "alpine", "Alpine", 2021, None, "alpine", "renault", REBRAND,
        "Renamed for 2021 after Renault's Alpine sports-car marque.  A "
        "marketing change: same entity, same factory.",
    ),
    # --- Hinwil: Sauber -> BMW Sauber -> Sauber -> Alfa -> Kick -> Audi ----
    TeamEntry(
        "sauber", "Sauber", 1993, 2005, "audi", None, NEW,
        "Peter Sauber's sports-car team, founded 1970, entered F1 in 1993 "
        "from Hinwil, where the operation has stayed ever since.",
    ),
    TeamEntry(
        "bmw_sauber", "BMW Sauber", 2006, 2010, "audi", "sauber", TAKEOVER,
        "BMW bought a majority stake in 2005.  BMW withdrew after 2009 and "
        "sold back to Peter Sauber, but the 2010 entry still carried the BMW "
        "Sauber constructor name, so the name outlasts the involvement by a "
        "season.",
    ),
    TeamEntry(
        "sauber", "Sauber", 2011, 2018, "audi", "bmw_sauber", TAKEOVER,
        "Independent again under Peter Sauber, running customer Ferrari "
        "power units.",
    ),
    TeamEntry(
        "alfa", "Alfa Romeo", 2019, 2023, "audi", "sauber", REBRAND,
        "A title-sponsorship deal signed in 2018; raced as Alfa Romeo Racing "
        "and then Alfa Romeo F1 Team.  Sauber Motorsport AG remained the "
        "entrant throughout.",
    ),
    TeamEntry(
        "kick_sauber", "Kick Sauber", 2024, 2025, "audi", "alfa", REBRAND,
        "Entered as Stake F1 Team Kick Sauber when the Alfa Romeo deal ended, "
        "carrying over the Stake and Kick sponsorships.",
    ),
    TeamEntry(
        "audi", "Audi", 2026, None, "audi", "kick_sauber", TAKEOVER,
        "Audi AG completed its acquisition of Sauber Motorsport in 2024 and "
        "the team races as a full works entry with Audi's own power unit from "
        "2026.  The FIA's 2026 entry list still names Sauber Motorsport AG as "
        "the entrant company while the entity is renamed -- a reminder that "
        "legal entity, entrant name and constructor name change on three "
        "different schedules.",
    ),
    # --- Silverstone: Jordan -> Midland -> Spyker -> Force India ->
    #     Racing Point -> Aston Martin ---------------------------------------
    TeamEntry(
        "jordan", "Jordan", 1991, 2005, "aston_martin", None, NEW,
        "Eddie Jordan's team, 1991.  Four wins; the entry it established ran "
        "unbroken until the 2018 administration.",
    ),
    TeamEntry(
        "mf1", "MF1", 2006, 2006, "aston_martin", "jordan", TAKEOVER,
        "Alex Shnaider's Midland Group bought Jordan in January 2005 and "
        "raced as MF1 Racing in 2006 under a Russian licence, from Jordan's "
        "Silverstone factory.  Sold on to Spyker Cars in September 2006, and "
        "renamed Spyker MF1 before the season ended.",
    ),
    TeamEntry(
        "spyker", "Spyker", 2007, 2007, "aston_martin", "mf1", TAKEOVER,
        "Spyker Cars ran the team under its own name for one season.",
    ),
    TeamEntry(
        "force_india", "Force India", 2008, 2018, "aston_martin", "spyker",
        TAKEOVER,
        "A consortium led by Vijay Mallya bought the team in October 2007.  "
        "The name covers the whole of 2018, including the second half, when "
        "the rescued team raced as Racing Point Force India.",
    ),
    TeamEntry(
        "racing_point", "Racing Point", 2019, 2020, "aston_martin",
        "force_india", RECONSTITUTED,
        "Force India entered administration in July 2018 and a consortium led "
        "by Lawrence Stroll bought the assets.  The FIA entry was not "
        "transferred: the entry Jordan established in 1991 ended, the "
        "constructor's points were struck mid-season and restarted from zero, "
        "and the 2020 Sakhir win is recorded as the team's first.  This is "
        "the one boundary in the table the championship itself refuses to "
        "carry history across.",
    ),
    TeamEntry(
        "aston_martin", "Aston Martin", 2021, None, "aston_martin",
        "racing_point", REBRAND,
        "Rebranded for 2021 after Lawrence Stroll led an investment in Aston "
        "Martin Lagonda.  Same entity and staff as Racing Point.",
    ),
    # --- Hingham / Leafield: Lotus Racing -> Team Lotus -> Caterham --------
    TeamEntry(
        "lotus_racing", "Lotus Racing", 2010, 2010, "caterham", None, NEW,
        "Tony Fernandes's new team, one of three admitted for 2010 under the "
        "budget-cap rules that were then abandoned.  Unrelated to the Enstone "
        "team despite the name: it licensed 'Lotus' from Group Lotus.",
    ),
    TeamEntry(
        "team_lotus", "Team Lotus", 2011, 2011, "caterham", "lotus_racing",
        REBRAND,
        "Proton terminated the Group Lotus licence, so Fernandes bought the "
        "dormant historic Team Lotus name instead.  For 2011 only, two "
        "different teams on the grid raced as Lotus.",
    ),
    TeamEntry(
        "caterham", "Caterham", 2012, 2014, "caterham", "team_lotus", REBRAND,
        "Fernandes gave up the Lotus name -- which cleared the way for the "
        "Enstone team to take it -- and renamed after his Caterham Cars.  The "
        "team collapsed late in 2014 and its assets were auctioned in 2015.",
    ),
    # --- Dinnington / Banbury: Virgin -> Marussia -> Manor -----------------
    TeamEntry(
        "virgin", "Virgin", 2010, 2011, "manor", None, NEW,
        "Manor Motorsport's entry, branded for Virgin; the second of the 2010 "
        "intake.  Marussia Motors took a stake in 2010 and it raced as "
        "Marussia Virgin Racing in 2011.",
    ),
    TeamEntry(
        "marussia", "Marussia", 2012, 2014, "manor", "virgin", REBRAND,
        "Marussia took majority control and the Virgin branding was dropped.",
    ),
    TeamEntry(
        "marussia", "Manor Marussia", 2015, 2015, "manor", "marussia",
        RECONSTITUTED,
        "The team entered administration in November 2014, missed the last "
        "two races of that season, and was re-established in February 2015 "
        "as a new entity, Manor Grand Prix Racing, after buying the assets.",
    ),
    TeamEntry(
        "manor", "Manor", 2016, 2016, "manor", "marussia", REBRAND,
        "Raced as Manor Racing for 2016, then failed to secure funding in "
        "January 2017 and folded.",
    ),
    # --- Cologne: Toyota ---------------------------------------------------
    TeamEntry(
        "toyota", "Toyota", 2002, 2009, "toyota", None, NEW,
        "A full works entry built from scratch in Cologne, and among the "
        "best-funded teams of its era.  Withdrew in November 2009 without "
        "ever winning a race.",
    ),
    # --- Leafield: Super Aguri --------------------------------------------
    TeamEntry(
        "super_aguri", "Super Aguri", 2006, 2008, "super_aguri", None, NEW,
        "Aguri Suzuki's Honda-backed team, formed at short notice for 2006 "
        "and running year-old Honda chassis.  Withdrew four races into 2008 "
        "when its funding failed.  A satellite of the Brackley team in "
        "everything but the constructor's entry, which was its own.",
    ),
    # --- Madrid / Murcia: HRT ----------------------------------------------
    TeamEntry(
        "hrt", "HRT", 2010, 2012, "hrt", None, NEW,
        "Entered as Campos Meta, the third of the 2010 intake, and taken over "
        "by Jose Ramon Carabante before its first race to become Hispania "
        "Racing, then HRT.  Left off the 2013 entry list when no buyer was "
        "found.",
    ),
    # --- Kannapolis: Haas --------------------------------------------------
    TeamEntry(
        "haas", "Haas", 2016, None, "haas", None, NEW,
        "Gene Haas's team, the first new constructor since the 2010 intake "
        "and the first American entry since 1986.  Built on a customer model: "
        "Ferrari power unit and gearbox, Dallara chassis construction.",
    ),
    # --- Fishers / Silverstone: Cadillac -----------------------------------
    TeamEntry(
        "cadillac", "Cadillac", 2026, None, "cadillac", None, NEW,
        "The eleventh entry, and the first genuinely new constructor since "
        "Haas in 2016.  Bid first as Andretti Cadillac and rejected by the "
        "commercial rights holder in 2024; resubmitted under TWG Motorsports "
        "after TWG Global bought Andretti Global, and accepted for 2026.  "
        "Runs Ferrari power units, with General Motors to supply its own "
        "later.",
    ),
)


#: Lineage metadata.  ``base`` is the town the operation actually lives in,
#: which is the more durable identity -- Wikipedia calls the Enstone lineage
#: "Team Enstone" for exactly this reason -- while ``lineage_id`` uses the
#: recognisable current name.
LINEAGES: dict[str, dict[str, str]] = {
    "ferrari":      {"name": "Ferrari",      "base": "Maranello, Italy"},
    "mclaren":      {"name": "McLaren",      "base": "Woking, UK"},
    "williams":     {"name": "Williams",     "base": "Grove, UK"},
    "mercedes":     {"name": "Mercedes",     "base": "Brackley, UK"},
    "red_bull":     {"name": "Red Bull",     "base": "Milton Keynes, UK"},
    "racing_bulls": {"name": "Racing Bulls", "base": "Faenza, Italy"},
    "alpine":       {"name": "Alpine",       "base": "Enstone, UK"},
    "audi":         {"name": "Audi",         "base": "Hinwil, Switzerland"},
    "aston_martin": {"name": "Aston Martin", "base": "Silverstone, UK"},
    "caterham":     {"name": "Caterham",     "base": "Hingham / Leafield, UK"},
    "manor":        {"name": "Manor",        "base": "Dinnington / Banbury, UK"},
    "toyota":       {"name": "Toyota",       "base": "Cologne, Germany"},
    "super_aguri":  {"name": "Super Aguri",  "base": "Leafield, UK"},
    "hrt":          {"name": "HRT",          "base": "Madrid / Murcia, Spain"},
    "haas":         {"name": "Haas",         "base": "Kannapolis, USA"},
    "cadillac":     {"name": "Cadillac",     "base": "Fishers, USA"},
}


#: Normalised alias -> the constructor keys it may mean.
#:
#: Matching is on whole tokens, never raw substrings, so ``"rb"`` cannot match
#: inside another word.  Nearly every alias resolves to one key; ``"lotus"`` is
#: the exception the sport forced on us and needs a season to resolve.
#:
#: Aliases cover both sides of the join: the Ergast/Jolpica ``constructorId``
#: spellings that arrive as ``TeamId``, and the sponsor-laden display strings
#: that arrive as ``TeamName``.  The sponsor prefixes themselves are not listed
#: -- they change yearly and are stripped by the longest-match rule instead.
ALIASES: dict[str, tuple[str, ...]] = {
    "ferrari": ("ferrari",),
    "mclaren": ("mclaren",),
    "williams": ("williams",),
    "tyrrell": ("tyrrell",),
    "bar": ("bar",),
    "british american racing": ("bar",),
    "honda": ("honda",),
    "honda racing": ("honda",),
    "brawn": ("brawn",),
    "brawn gp": ("brawn",),
    "mercedes": ("mercedes",),
    "mercedes gp": ("mercedes",),
    "mercedes amg": ("mercedes",),
    "stewart": ("stewart",),
    "jaguar": ("jaguar",),
    "red bull": ("red_bull",),
    "minardi": ("minardi",),
    "toro rosso": ("toro_rosso",),
    "str": ("toro_rosso",),
    "alphatauri": ("alphatauri",),
    "alpha tauri": ("alphatauri",),
    "rb": ("rb",),
    "racing bulls": ("racing_bulls",),
    "toleman": ("toleman",),
    "benetton": ("benetton",),
    "renault": ("renault",),
    "lotus renault": ("renault",),
    "lotus f1": ("lotus_f1",),
    "alpine": ("alpine",),
    "sauber": ("sauber",),
    "bmw sauber": ("bmw_sauber",),
    "bmw": ("bmw_sauber",),
    "alfa": ("alfa",),
    "alfa romeo": ("alfa",),
    "kick sauber": ("kick_sauber",),
    "stake": ("kick_sauber",),
    "audi": ("audi",),
    "jordan": ("jordan",),
    "mf1": ("mf1",),
    "midland": ("mf1",),
    "spyker mf1": ("mf1",),
    "spyker": ("spyker",),
    "force india": ("force_india",),
    "racing point": ("racing_point",),
    # One of three transitional double-barrelled names, all of which have to
    # be spelled out.  They are the cases where the usual convention -- the
    # constructor comes first, the sponsor after -- is reversed, so the
    # longest-match rule needs the full string to get them right.  Here it
    # would otherwise read the "Racing Point" in front and date a 2018 row to
    # 2019, the year the constructor actually changed.
    "racing point force india": ("force_india",),
    "aston martin": ("aston_martin",),
    "lotus racing": ("lotus_racing",),
    "team lotus": ("team_lotus",),
    "caterham": ("caterham",),
    "virgin": ("virgin",),
    "marussia": ("marussia",),
    # As above, for 2011: the constructor was still Virgin, the livery was not.
    "marussia virgin": ("virgin",),
    "manor": ("manor",),
    # The third: in 2015 the new owner's name led and the constructor followed.
    "manor marussia": ("marussia",),
    "toyota": ("toyota",),
    "super aguri": ("super_aguri",),
    "aguri": ("super_aguri",),
    "hrt": ("hrt",),
    "hispania": ("hrt",),
    "campos": ("hrt",),
    "haas": ("haas",),
    "cadillac": ("cadillac",),
    # Ambiguous on purpose: two unrelated teams raced as Lotus in 2011, and the
    # plain name moved from one to the other in 2012.
    "lotus": ("lotus_racing", "team_lotus", "lotus_f1"),
}

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def normalise(name: str) -> tuple[str, ...]:
    """Lower-case a team name and split it into comparable tokens.

    ``"Stake F1 Team Kick Sauber"`` becomes
    ``("stake", "f1", "team", "kick", "sauber")``.  Nothing is dropped: the
    noise words are harmless because matching looks for known marques rather
    than trying to decide what a team name is *not*.
    """
    return tuple(t for t in _TOKEN_SPLIT.split(str(name).lower()) if t)


#: Pre-split aliases, longest first, so the longest match is found first.
_ALIAS_TOKENS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = tuple(
    sorted(
        ((normalise(alias), keys) for alias, keys in ALIASES.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def _match_alias(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Find the constructor keys a normalised name refers to.

    Two rules, in order, both of them load-bearing:

    1. **The longest alias wins.**  "Stake F1 Team Kick Sauber" contains both
       ``kick sauber`` and ``sauber``; the longer one is the more specific
       claim and is right.
    2. **On a tie, the earliest wins.**  "McLaren Honda" contains two
       one-token marques, and F1 names its constructor before its engine
       supplier, so position decides it.  The same rule strips sponsor
       prefixes for free -- "Oracle Red Bull Racing" has only one marque in it.
    """
    best: tuple[int, int, tuple[str, ...]] | None = None
    for alias, keys in _ALIAS_TOKENS:
        width = len(alias)
        if width > len(tokens):
            continue
        for start in range(len(tokens) - width + 1):
            if tokens[start : start + width] == alias:
                candidate = (width, -start, keys)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
                break
    return () if best is None else best[2]


def _entries_for(key: str) -> tuple[TeamEntry, ...]:
    return tuple(e for e in ENTRIES if e.key == key)


def _lineage_entries(lineage_id: str) -> tuple[TeamEntry, ...]:
    return tuple(
        sorted(
            (e for e in ENTRIES if e.lineage_id == lineage_id),
            key=lambda e: e.first_season,
        )
    )


def resolve(team: str, season: int | None = None) -> TeamEntry:
    """Find the constructor a name refers to.

    ``team`` may be an Ergast/Jolpica ``constructorId``, a FastF1 ``TeamName``,
    or any reasonable spelling in between; sponsor prefixes and "F1 Team"
    suffixes are handled.

    ``season`` disambiguates, and should be passed whenever it is known.  Two
    behaviours depend on it:

    * A name used by more than one constructor needs it.  Without a season,
      ``"Lotus"`` raises :class:`AmbiguousTeamError` rather than picking one.
    * A name that has since been replaced is carried **forward** along its
      lineage.  ``resolve("sauber", 2024)`` returns the Kick Sauber entry,
      because that is the constructor the Hinwil team entered as in 2024 --
      which is what you want when the backend keeps an id across a rename.
      The reverse never happens: ``resolve("Alpine", 2015)`` raises rather
      than claiming the 2015 Lotus F1 car was an Alpine.

    Raises :class:`UnknownTeamError` if nothing matches, because in a
    modelling frame an unrecognised team is silently a team with no history.
    """
    tokens = normalise(team)
    keys = _match_alias(tokens)
    if not keys:
        raise UnknownTeamError(
            f"no constructor matches {team!r}. If this is a new team, add it "
            f"to src.data.teams.ENTRIES; if it is a new spelling of an "
            f"existing one, add it to ALIASES."
        )

    candidates = [e for key in keys for e in _entries_for(key)]

    if season is None:
        distinct = {e.key for e in candidates}
        if len(distinct) > 1:
            raise AmbiguousTeamError(
                f"{team!r} matches more than one constructor "
                f"({', '.join(sorted(distinct))}) and no season was given. "
                f"Two unrelated teams have raced under one name; pass the "
                f"season to say which you mean."
            )
        return max(candidates, key=lambda e: e.first_season)

    covering = [e for e in candidates if e.covers(season)]
    if len(covering) == 1:
        return covering[0]
    if len(covering) > 1:
        raise AmbiguousTeamError(
            f"{team!r} matches {len(covering)} constructors in {season}: "
            f"{', '.join(sorted(e.key for e in covering))}."
        )

    # No entry carrying this name covers the season.  Follow the lineage --
    # but only from the point the name first appeared.  That admits the two
    # safe cases and refuses the unsafe one:
    #
    #   * a gap *inside* the name's use.  "Sauber" raced 1993-2005 and again
    #     2011-2018, and in between the same Hinwil team raced as BMW Sauber.
    #   * a season *after* the name was retired.  This is the one that matters
    #     in practice: the results backend has kept a constructor id across a
    #     rebrand, so a 2024 row can still say ``sauber``.
    #
    # Going backwards is refused, because a name applied to a season before it
    # existed is a mislabelling rather than a rename: the 2015 Enstone car was
    # a Lotus, and calling it an Alpine would invent history.
    earliest = min(candidates, key=lambda e: e.first_season)
    if season >= earliest.first_season:
        latest = max(candidates, key=lambda e: e.first_season)
        for entry in _lineage_entries(latest.lineage_id):
            if entry.covers(season):
                log.debug(
                    "%r does not name a %d constructor; resolved along its "
                    "lineage to %s.", team, season, entry.name,
                )
                return entry

    raise UnknownTeamError(
        f"{team!r} did not enter the {season} championship. It raced "
        f"{', '.join(e.span for e in candidates)}."
    )


def lineage_of(team: str, season: int | None = None) -> str:
    """The lineage id for a team name.  See :func:`resolve` for the rules."""
    return resolve(team, season).lineage_id


def _predecessor(entry: TeamEntry) -> TeamEntry | None:
    """The entry this one took over from.

    Looked up by key *and* season rather than key alone, because a constructor
    name can be used twice: Manor succeeded ``marussia``, and the Marussia it
    succeeded is the 2015 entity, not the 2012-2014 one.
    """
    if entry.succeeds is None:
        return None
    prior = [
        p
        for p in ENTRIES
        if p.key == entry.succeeds and p.first_season < entry.first_season
    ]
    return max(prior, key=lambda p: p.first_season) if prior else None


def _era_by_entry() -> dict[TeamEntry, int]:
    """Map each entry to its era index within its lineage.

    The counter starts at 0 and steps at every ``reconstituted`` succession, so
    ``(lineage_id, lineage_era)`` is the grouping the championship itself would
    recognise as one continuous record.
    """
    eras: dict[TeamEntry, int] = {}
    for lineage_id in LINEAGES:
        era = 0
        for entry in _lineage_entries(lineage_id):
            if entry.succession == RECONSTITUTED:
                era += 1
            eras[entry] = era
    return eras


_ERAS = _era_by_entry()


def era_of(entry: TeamEntry) -> int:
    """Which continuous-record era of its lineage an entry belongs to."""
    return _ERAS[entry]


def entries_frame() -> pd.DataFrame:
    """Every constructor entry as a DataFrame, oldest first.

    This is the joinable form of the table.  ``*.csv`` is gitignored in this
    repository, so the committed lookup is the generated Markdown at
    ``References/team_lineage.md``; write this frame out yourself if you want
    a CSV that another tool can read.
    """
    frame = pd.DataFrame(
        [
            {
                "constructor_key": e.key,
                "constructor_name": e.name,
                "first_season": e.first_season,
                "last_season": e.last_season,
                "seasons": e.span,
                "lineage_id": e.lineage_id,
                "lineage_name": LINEAGES[e.lineage_id]["name"],
                "lineage_base": LINEAGES[e.lineage_id]["base"],
                "lineage_era": era_of(e),
                "succeeds": e.succeeds,
                "succession": e.succession,
                "history_carries": e.succession != RECONSTITUTED,
                "note": e.note,
            }
            for e in ENTRIES
        ]
    )
    # Nullable, so an open-ended span does not turn the column to float and
    # render a team's last season as "2014.0".
    frame["last_season"] = frame["last_season"].astype("Int64")
    return frame.sort_values(
        ["lineage_id", "first_season"], ignore_index=True
    )


def lineage_frame() -> pd.DataFrame:
    """One row per lineage: what it is called now, and what it used to be."""
    rows = []
    for lineage_id, meta in LINEAGES.items():
        entries = _lineage_entries(lineage_id)
        rows.append(
            {
                "lineage_id": lineage_id,
                "lineage_name": meta["name"],
                "lineage_base": meta["base"],
                "first_season": entries[0].first_season,
                "last_season": entries[-1].last_season,
                "active": entries[-1].last_season is None,
                "n_names": len(entries),
                "n_eras": era_of(entries[-1]) + 1,
                "names": " -> ".join(e.name for e in entries),
            }
        )
    frame = pd.DataFrame(rows)
    frame["last_season"] = frame["last_season"].astype("Int64")
    return frame.sort_values("lineage_id", ignore_index=True)


def attach_lineage(
    frame: pd.DataFrame,
    *,
    team_col: str = "TeamId",
    season_col: str = "Year",
    strict: bool = False,
) -> pd.DataFrame:
    """Join constructor lineage onto a results or modelling frame.

    Adds ``constructor_key``, ``constructor_name``, ``lineage_id``,
    ``lineage_name``, ``lineage_era``, ``succession`` and ``history_carries``.

    The two grouping keys answer different questions, and which one is right is
    a modelling decision this function does not make for you:

    ``lineage_id``
        Group by continuous *operation*.  Aston Martin's history reaches back
        to Jordan in 1991 whatever the championship says about entries.

    ``(lineage_id, lineage_era)``
        Group by continuous *record*, splitting at every insolvency.  This is
        what the sport recognises, and it is the conservative choice.

    Unknown teams get nulls and a warning naming them, mirroring
    :func:`src.data.circuits.attach_reference`: a new constructor should be
    visible in the logs rather than silently acquiring an empty history.  Pass
    ``strict=True`` to raise instead, which is what a build step should do.
    """
    for col in (team_col, season_col):
        if col not in frame.columns:
            raise KeyError(f"{col!r} not in frame; columns: {list(frame.columns)}")

    resolved: dict[tuple[str, int | None], TeamEntry | None] = {}
    unknown: dict[str, str] = {}

    pairs = frame[[team_col, season_col]].drop_duplicates()
    for team, season in pairs.itertuples(index=False):
        if pd.isna(team):
            continue
        year = None if pd.isna(season) else int(season)
        try:
            resolved[(team, year)] = resolve(team, year)
        except (UnknownTeamError, AmbiguousTeamError) as exc:
            resolved[(team, year)] = None
            unknown[f"{team} ({year})"] = str(exc)

    if unknown:
        message = (
            f"No lineage for {len(unknown)} team-season(s): "
            f"{', '.join(sorted(unknown))}. Add them to "
            f"src.data.teams.ENTRIES or ALIASES."
        )
        if strict:
            raise UnknownTeamError(message)
        log.warning("%s Rows keep null lineage columns.", message)

    def _lookup(row: tuple) -> TeamEntry | None:
        team, season = row
        if pd.isna(team):
            return None
        return resolved.get((team, None if pd.isna(season) else int(season)))

    entries = [
        _lookup(row) for row in frame[[team_col, season_col]].itertuples(index=False)
    ]

    out = frame.copy()
    out["constructor_key"] = [e.key if e else None for e in entries]
    out["constructor_name"] = [e.name if e else None for e in entries]
    out["lineage_id"] = [e.lineage_id if e else None for e in entries]
    out["lineage_name"] = [
        LINEAGES[e.lineage_id]["name"] if e else None for e in entries
    ]
    out["lineage_era"] = pd.array(
        [era_of(e) if e else None for e in entries], dtype="Int8"
    )
    out["succession"] = [e.succession if e else None for e in entries]
    out["history_carries"] = pd.array(
        [(e.succession != RECONSTITUTED) if e else None for e in entries],
        dtype="boolean",
    )
    return out


def renames_between(first_season: int, last_season: int) -> pd.DataFrame:
    """Every constructor name change in a season range, newest first.

    Useful as a sanity check before trusting a rolling team feature over a
    window: each row here is a point at which a ``TeamId``-keyed history
    silently restarts.

    The columns are ``from_name`` and ``to_name`` rather than the more natural
    ``from`` and ``to`` because ``from`` is a Python keyword, and a column so
    named is unreachable through ``DataFrame.itertuples``.
    """
    rows = [
        {
            "season": e.first_season,
            "from_name": getattr(_predecessor(e), "name", None),
            "to_name": e.name,
            "lineage_id": e.lineage_id,
            "succession": e.succession,
            "history_carries": e.succession != RECONSTITUTED,
        }
        for e in ENTRIES
        if e.succeeds is not None and first_season <= e.first_season <= last_season
    ]
    return pd.DataFrame(rows).sort_values(
        ["season", "lineage_id"], ascending=[False, True], ignore_index=True
    )


def active_in(season: int) -> tuple[TeamEntry, ...]:
    """Every constructor that contested a given season."""
    return tuple(
        sorted(
            (e for e in ENTRIES if e.covers(season)),
            key=lambda e: e.name,
        )
    )


def validate(entries: Iterable[TeamEntry] = ENTRIES) -> list[str]:
    """Check the table's internal consistency.  Returns a list of problems.

    Run by the tests, and cheap enough to call from a build step.  It catches
    the mistakes that hand-entered reference data actually accumulates: a
    succession pointing at a team that did not exist yet, a lineage whose id
    no longer matches the team currently racing, overlapping spans for one
    name, and a chain that does not terminate in a ``new`` constructor.
    """
    problems: list[str] = []
    entries = tuple(entries)
    keys = {e.key for e in entries}

    for e in entries:
        if e.lineage_id not in LINEAGES:
            problems.append(f"{e.key} {e.span}: lineage {e.lineage_id!r} not in LINEAGES")
        if e.succession not in SUCCESSION_GRADES:
            problems.append(f"{e.key} {e.span}: unknown succession {e.succession!r}")
        if e.last_season is not None and e.last_season < e.first_season:
            problems.append(f"{e.key} {e.span}: ends before it starts")
        if (e.succeeds is None) != (e.succession == NEW):
            problems.append(
                f"{e.key} {e.span}: succession {e.succession!r} disagrees with "
                f"succeeds={e.succeeds!r}"
            )
        if e.succeeds is not None:
            if e.succeeds not in keys:
                problems.append(f"{e.key} {e.span}: succeeds unknown key {e.succeeds!r}")
            else:
                prior = _predecessor(e)
                if prior is None:
                    problems.append(
                        f"{e.key} {e.span}: succeeds {e.succeeds!r}, which never "
                        f"raced earlier"
                    )
                elif prior.lineage_id != e.lineage_id:
                    problems.append(
                        f"{e.key} {e.span}: succeeds {e.succeeds!r} from another "
                        f"lineage"
                    )

    # One name must not describe two overlapping periods.
    for key in sorted(keys):
        spans = sorted(_entries_for(key), key=lambda e: e.first_season)
        for earlier, later in zip(spans, spans[1:]):
            if earlier.last_season is None or earlier.last_season >= later.first_season:
                problems.append(f"{key}: spans {earlier.span} and {later.span} overlap")

    for lineage_id in LINEAGES:
        chain = _lineage_entries(lineage_id)
        if not chain:
            problems.append(f"{lineage_id}: lineage has no entries")
            continue
        if chain[0].succession != NEW:
            problems.append(
                f"{lineage_id}: oldest entry {chain[0].key!r} is "
                f"{chain[0].succession!r}, so the chain does not terminate"
            )
        if chain[-1].key != lineage_id:
            problems.append(
                f"{lineage_id}: named after {lineage_id!r} but its latest "
                f"constructor is {chain[-1].key!r} -- rename the lineage"
            )

    return problems
