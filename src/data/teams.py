"""Constructor lineage: which team names are the same organisation.

Ergast / Jolpica issue a new ``constructorId`` whenever an entry is renamed, so
``TeamId`` changes at every rebrand.  Four organisations on the 2018-2026 grid
have done so, one of them three times:

=============  ===========================================================
lineage        TeamIds, in order
=============  ===========================================================
``enstone``    ``renault`` (-2020) -> ``alpine`` (2021-)
``faenza``     ``toro_rosso`` (-2019) -> ``alphatauri`` (2020-23) -> ``rb``
``hinwil``     ``sauber`` (2018) -> ``alfa`` (2019-23) -> ``sauber`` (2024-25)
               -> ``audi`` (2026-)
``silverstone``  ``force_india`` (2018) -> ``racing_point`` (2019-20)
               -> ``aston_martin`` (2021-)
=============  ===========================================================

Why this matters for a finishing-order model and barely mattered for the DNF
one: **the running order carries across the winter.**  Measured on this
dataset, the correlation of a team's season-mean grid percentile with the
previous season's is 0.79-0.95, including across the 2022 and 2026 regulation
breaks.  A history keyed on ``TeamId`` throws that away four times -- Audi's
2026 "history" would start at zero, although it is the same factory, staff and
wind tunnel that ran as Sauber the year before.  Keyed on lineage, it starts
from Sauber's 2025 form, which is the best available prior.

A name change is not always *only* a name change -- Aston Martin's 2021 car
was a Racing Point with a new badge, while Audi's 2026 entry brought its own
power unit.  Lineage asserts organisational continuity, not an unchanged car;
the winter-carryover numbers above are what justify treating that continuity
as information rather than assuming it.

The lineage is named after the factory because that is the one thing that did
not change, and it will survive the next rebrand without a rename here.
Entries that never changed name are their own lineage.  A genuinely new
entrant (Cadillac, 2026) is its own lineage and correctly starts with no
history.

Sources: the FIA entry lists for each season and the teams' own published
histories.  Entry names are what Ergast / Jolpica report as ``constructorId``.
"""

from __future__ import annotations

import pandas as pd

#: ``TeamId`` -> lineage, for every team that has raced under more than one
#: name since 2018.  Anything not listed is its own lineage.
TEAM_LINEAGE: dict[str, str] = {
    "renault": "enstone",
    "alpine": "enstone",
    "toro_rosso": "faenza",
    "alphatauri": "faenza",
    "rb": "faenza",
    "sauber": "hinwil",
    "alfa": "hinwil",
    "audi": "hinwil",
    "force_india": "silverstone",
    "racing_point": "silverstone",
    "aston_martin": "silverstone",
}


def team_lineage(team_id: object) -> object:
    """The organisation behind a ``TeamId``; unknown ids map to themselves."""
    if team_id is None or (isinstance(team_id, float) and pd.isna(team_id)):
        return team_id
    return TEAM_LINEAGE.get(str(team_id), str(team_id))


def add_team_lineage(frame: pd.DataFrame, *, team_col: str = "TeamId") -> pd.DataFrame:
    """Append ``team_lineage``, the grouping key for cross-season team history."""
    out = frame.copy()
    out["team_lineage"] = out[team_col].map(team_lineage).astype("object")
    return out
