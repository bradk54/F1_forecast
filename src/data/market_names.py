"""Market outcomes -> drivers, and market events -> races.

Prediction markets name drivers however the listing trader typed them:
Polymarket uses a bare surname ("Piastri") for 2025 and a full name for 2024,
Kalshi uses the full name, and the same driver turns up as "Kimi Antonelli",
"Andrea Kimi Antonelli" and "A. K. Antonelli" across sources.  Events are no
better -- "Gran Premio de Mexico", "2025 F1 British Grand Prix" -- so races are
joined on **date**, which every source gets right to within a day.

The failure this module exists to prevent is the quiet one.  A driver dropped
because a name did not match is not an error anywhere downstream: the field just
has one fewer car, the remaining prices are renormalised over it, and every
probability in the race is a little wrong.  So an unmatched name **raises**
(:class:`MarketNameError`), in the spirit of ``DegradedResultsError``, and the
fix is to add the name to :data:`DRIVER_ALIASES` or :data:`IGNORED_OUTCOMES`
below -- one explicit, reviewable table -- rather than to loosen the matcher.

Matching is scoped to **one race's field**, never to all drivers ever.  A bare
surname is unambiguous inside a 20-car grid even when it would not be across a
decade, and scoping it this way means "Verstappen" never has to be told apart
from a Verstappen who is not racing.
"""

from __future__ import annotations

import re

import pandas as pd
from unidecode import unidecode

#: Calendar days a market's stated end date may differ from the race date.
#: Observed: Polymarket's end date is the race date or one day later.
DATE_TOLERANCE_DAYS = 2

#: Outcomes that are not a driver and carry no probability.  Polymarket pads
#: each listing with "Driver A".."Driver J" placeholders and an "Other" bucket.
IGNORED_OUTCOMES: frozenset[str] = frozenset({
    "other", "field", "the field", "any other driver", "no winner", "none",
})
_PLACEHOLDER = re.compile(r"^driver [a-z]$")

#: Normalised market name -> normalised name as it appears in the results'
#: ``FullName``.  Only for names the token matcher cannot reach.
DRIVER_ALIASES: dict[str, str] = {
    "checo perez": "sergio perez",
}


class MarketNameError(ValueError):
    """A market outcome or event could not be tied to a driver or a race."""


def normalise(name: object) -> str:
    """Lower-case, ASCII, punctuation-free, single-spaced."""
    text = unidecode(str(name)).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_ignored(name: object) -> bool:
    key = normalise(name)
    return key in IGNORED_OUTCOMES or bool(_PLACEHOLDER.match(key))


def _tokens_match(market: list[str], full: list[str]) -> bool:
    """Every market token matches a distinct token of the full name.

    A one-letter token is an initial ("A. K. Antonelli") and matches any name
    token starting with it; a longer one must match exactly.  Subset, not
    equality, so a surname or a dropped middle name still resolves.
    """
    remaining = list(full)
    for token in market:
        hit = next((f for f in remaining
                    if f == token or (len(token) == 1 and f.startswith(token))), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def resolve_outcomes(names: list[str], field: pd.DataFrame) -> dict[str, str | None]:
    """Map each market outcome name to a ``DriverId`` in this race's field.

    Args:
        names: Raw outcome names for one race.
        field: One row per driver who started, with ``DriverId`` and ``FullName``.

    Returns:
        ``name -> DriverId``, or ``None`` for an ignored outcome.

    Raises:
        MarketNameError: A name matches no driver, or more than one.  Names are
        collected and raised together so one run reports them all.
    """
    full = {row.DriverId: normalise(row.FullName).split() for row in field.itertuples()}
    resolved: dict[str, str | None] = {}
    problems = []
    for raw in names:
        if is_ignored(raw):
            resolved[raw] = None
            continue
        key = normalise(raw)
        key = DRIVER_ALIASES.get(key, key)
        hits = [d for d, parts in full.items() if _tokens_match(key.split(), parts)]
        if len(hits) == 1:
            resolved[raw] = hits[0]
        else:
            problems.append(f"{raw!r}: " + ("matches no driver in the field"
                                            if not hits else f"ambiguous between {hits}"))
    if problems:
        raise MarketNameError(
            "unresolved market outcomes (add to DRIVER_ALIASES or IGNORED_OUTCOMES in "
            "src/data/market_names.py):\n  " + "\n  ".join(problems))
    return resolved


def match_race(event_date: object, calendar: pd.DataFrame) -> tuple[int, int] | None:
    """The ``(Year, RoundNumber)`` whose race date is nearest ``event_date``.

    ``None`` if nothing is within :data:`DATE_TOLERANCE_DAYS`.  A tie raises:
    two races that close together is a calendar problem to look at, not guess.
    """
    when = pd.Timestamp(event_date).tz_localize(None).normalize()
    gap = (pd.to_datetime(calendar["RaceDate"]).dt.normalize() - when).abs()
    near = calendar.loc[gap <= pd.Timedelta(days=DATE_TOLERANCE_DAYS)]
    if near.empty:
        return None
    best = gap.loc[near.index].sort_values(kind="mergesort")
    if len(best) > 1 and best.iloc[0] == best.iloc[1]:
        raise MarketNameError(f"{when.date()} is equidistant from two races")
    row = calendar.loc[best.index[0]]
    return int(row["Year"]), int(row["RoundNumber"])
