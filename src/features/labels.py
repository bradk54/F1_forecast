"""Turn Formula 1 classification codes into modelling targets.

Two independent fields describe how a driver's race ended, and conflating them
is the most common way to get this label wrong:

``ClassifiedPosition``
    The official *classification*.  Either an integer (the driver holds a
    finishing position) or one of ``R`` retired, ``D`` disqualified,
    ``E`` excluded, ``W`` withdrawn, ``F`` failed to qualify, ``N`` not
    classified.  A driver who retires after covering 90% of the race distance
    is still classified, so ``ClassifiedPosition`` alone under-counts DNFs.

``Status``
    Why the race ended: ``Finished``, ``+1 Lap``, ``Engine``, ``Collision``,
    ``Accident``, ``Gearbox`` and so on.  This carries the cause but not the
    classification.

The two are combined here into an explicit set of columns:

=======================  ====================================================
``started``              Driver took the start (excludes DNS/DNQ/withdrawn).
``classified``           ``ClassifiedPosition`` parses as an integer.
``dnf``                  **Primary target.**  The car stopped before the end.
``dnf_strict``           Retired *and* not officially classified.
``dnf_classified``       Retired late but still classified (the >90% case).
``dnf_cause``            mechanical / collision / driver_error / disqualified /
                         withdrawn / other / finished.
``finished_on_track``    Complement of ``dnf`` among starters; 1 when the car
                         took the chequered flag under its own power.
=======================  ====================================================

A note on disqualification.  A DSQ is a *classification* penalty, usually
applied to a car that completed the race (Hamilton and Leclerc at Austin 2023
for excessive plank wear, for example).  Treating that as a DNF would teach the
model that scrutineering failures are mechanical failures, so ``dnf`` is 0 for
a driver who was disqualified after finishing and 1 only if they also retired.
``dnf_cause`` still records ``disqualified`` so the rows stay findable.

Cause mapping follows Ergast's status vocabulary, where ``Accident`` denotes a
single-car incident and ``Collision`` denotes contact between cars.  Any status
string the rules do not recognise lands in ``other`` and is reported by
:func:`unmapped_statuses`, so new vocabulary never fails silently.

A *blank* status is a different problem, and one these rules cannot solve: with
no cause to read, the row falls into ``other`` and is labelled ``dnf=1`` on no
evidence.  :func:`has_usable_status` is the predicate for "this row has a status
at all"; the pipeline uses it to reject a degraded ingest outright rather than
letting it reach the labels.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Classification codes
# --------------------------------------------------------------------------- #

#: Codes meaning "no official finishing position".
NON_CLASSIFIED_CODES = frozenset({"R", "D", "E", "W", "F", "N"})

#: Codes meaning the driver never took the start.
NON_STARTER_CODES = frozenset({"W", "F"})

# --------------------------------------------------------------------------- #
# Cause taxonomy
# --------------------------------------------------------------------------- #

FINISHED = "finished"
MECHANICAL = "mechanical"
COLLISION = "collision"
DRIVER_ERROR = "driver_error"
DISQUALIFIED = "disqualified"
WITHDRAWN = "withdrawn"
OTHER = "other"

CAUSE_ORDER = (
    FINISHED,
    MECHANICAL,
    COLLISION,
    DRIVER_ERROR,
    DISQUALIFIED,
    WITHDRAWN,
    OTHER,
)

#: Statuses that mean the driver reached the end of the race.  ``+ N Lap(s)``
#: is a classified finish one or more laps down, not a retirement.
_FINISHED_PATTERNS = (
    re.compile(r"^finished$", re.I),
    re.compile(r"^\+\s*\d+\s+laps?$", re.I),
)

#: Ordered (pattern, cause) rules.  The first match wins, so more specific
#: patterns must come first.  Keywords rather than exact strings, because the
#: status vocabulary grows: "Power Unit", "ERS", "Water pressure" and friends
#: all appear over the seasons covered here.
_CAUSE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # --- disqualification / exclusion ---------------------------------------
    (re.compile(r"\b(?:disqualif\w*|excluded)\b", re.I), DISQUALIFIED),
    # --- did not start ------------------------------------------------------
    (
        re.compile(
            r"did not (start|qualify|prequalify)|withdr\w*|"
            r"107%|not restarted|safety concerns",
            re.I,
        ),
        WITHDRAWN,
    ),
    # --- contact between cars ----------------------------------------------
    (re.compile(r"\b(?:collision|debris)\b", re.I), COLLISION),
    # --- single-car incidents ----------------------------------------------
    # "Accident" is Ergast's code for a solo shunt; "Damage" without a
    # collision qualifier usually follows one.
    (re.compile(r"\b(?:accident|spun\s*off)\b|^damage$", re.I), DRIVER_ERROR),
    # --- car systems --------------------------------------------------------
    # Word boundaries matter here.  An unanchored ``tire`` matches "Retired"
    # and an unanchored ``ers`` matches "Drivers", which would quietly file a
    # cause-less retirement as a mechanical failure.
    (
        re.compile(
            # Prefixes: match the stem so inflections are covered.
            r"\b(?:electr|hydraul|overheat|pneumatic|vibrat|mechanic|"
            r"technical|puncture|transmiss)"
            # Whole words and multi-word phrases.
            r"|\b(?:engine|power\s*unit|ers|energy\s*store|turbo|battery|mgu"
            r"|mgu-?[hk]|gearbox|clutch|driveshaft|halfshaft|differential"
            r"|alternator|ignition|injection|distributor|magneto"
            r"|brakes?|suspension|steering|track\s*rod|wheels?|wheel\s*nut"
            r"|tyres?|tires?|radiator|cooling|coolant|water|oil|fuel"
            r"|exhaust|throttle|wing|undertray|bodywork|chassis|seat"
            r"|launch\s*control|handling|pressure|leak|failure|fire"
            r"|out\s*of\s*fuel|engine\s*fire)\b",
            re.I,
        ),
        MECHANICAL,
    ),
    # --- human factors ------------------------------------------------------
    (re.compile(r"\b(?:injur\w*|illness|ill|fatigue|physical)\b", re.I), OTHER),
    # --- generic retirement with no stated cause ---------------------------
    (re.compile(r"^retired$|^not classified$", re.I), OTHER),
)


def _is_finished_status(status: str) -> bool:
    return any(p.match(status.strip()) for p in _FINISHED_PATTERNS)


def classify_status(status: object) -> str:
    """Map a single ``Status`` string onto the cause taxonomy.

    >>> classify_status("Finished")
    'finished'
    >>> classify_status("+ 1 Lap")
    'finished'
    >>> classify_status("Power Unit")
    'mechanical'
    >>> classify_status("Collision damage")
    'collision'
    >>> classify_status("Spun off")
    'driver_error'
    """
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return OTHER
    text = str(status).strip()
    if not text:
        return OTHER
    if _is_finished_status(text):
        return FINISHED
    for pattern, cause in _CAUSE_RULES:
        if pattern.search(text):
            return cause
    return OTHER


def has_usable_status(statuses: object) -> pd.Series:
    """Boolean mask: True where a finishing status is actually present.

    Blank is not a status.  When the Ergast backend rate-limits, FastF1 logs a
    warning, gives up on the result payload and returns a results frame whose
    ``Status`` column is filled with **empty strings** -- not nulls, so
    ``isna()`` reports nothing amiss.  Every such row then classifies as
    :data:`OTHER`, which :func:`add_race_outcome_labels` reads as a retirement,
    so a rate-limited round arrives labelled as a 100% DNF race.

    This is the predicate both the ingest guard and the dataset-level check
    are built on; see :func:`src.data.ingest.extract_results` and
    :func:`src.data.generate_dataset.status_coverage`.
    """
    series = (
        statuses
        if isinstance(statuses, pd.Series)
        else pd.Series(statuses, dtype="object")
    )
    return series.map(
        lambda v: not pd.isna(v) and bool(str(v).strip())
    ).astype(bool)


def unmapped_statuses(statuses: Iterable[object]) -> pd.Series:
    """Count status strings that fell through to ``other``.

    Run this after every ingest.  A new status string appearing in the ``other``
    bucket is a signal to extend :data:`_CAUSE_RULES`, not something to ignore.
    """
    rows = [
        str(s).strip()
        for s in statuses
        if classify_status(s) == OTHER and str(s).strip()
    ]
    return pd.Series(rows, dtype="object").value_counts()


def _parse_classified_position(value: object) -> tuple[bool, str | None]:
    """Return ``(is_classified, code)`` for one ``ClassifiedPosition`` value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False, None
    text = str(value).strip().upper()
    if not text:
        return False, None
    if text.isdigit():
        return True, None
    # Occasionally arrives as "12.0" from a float round-trip.
    try:
        float(text)
    except ValueError:
        return False, text
    return True, None


def add_race_outcome_labels(
    results: pd.DataFrame,
    *,
    status_col: str = "Status",
    classified_col: str = "ClassifiedPosition",
    warn_on_unmapped: bool = True,
) -> pd.DataFrame:
    """Attach every outcome label to a race-results frame.

    Args:
        results: One row per driver per race.  Must carry ``status_col``;
            ``classified_col`` is used when present and inferred from the
            status otherwise (Ergast-sourced frames often lack it).
        status_col: Name of the finishing-status column.
        classified_col: Name of the official-classification column.
        warn_on_unmapped: Log any status strings that fell through to ``other``.

    Returns:
        A copy of ``results`` with the label columns described in the module
        docstring appended.
    """
    if status_col not in results.columns:
        raise KeyError(
            f"{status_col!r} not found; available columns: {list(results.columns)}"
        )

    out = results.copy()
    status = out[status_col]
    usable = has_usable_status(status)

    out["dnf_cause"] = status.map(classify_status).astype(
        pd.CategoricalDtype(categories=CAUSE_ORDER)
    )

    if classified_col in out.columns:
        parsed = out[classified_col].map(_parse_classified_position)
        out["classified"] = [bool(p[0]) for p in parsed]
        codes = [p[1] for p in parsed]
    else:
        # No classification column: a finished-status row is classified.
        out["classified"] = out["dnf_cause"].eq(FINISHED).to_numpy()
        codes = [None] * len(out)
    out["classification_code"] = pd.Series(codes, index=out.index, dtype="object")

    took_flag = out["dnf_cause"].eq(FINISHED).to_numpy()
    code_series = out["classification_code"].fillna("")

    # Never started: withdrawn / failed to qualify, by code or by status.
    out["started"] = ~(
        code_series.isin(NON_STARTER_CODES).to_numpy()
        | out["dnf_cause"].eq(WITHDRAWN).to_numpy()
    )

    # Primary target: the car stopped before the end of the race.  A driver
    # disqualified after taking the flag is not a DNF.
    stopped = out["dnf_cause"].isin([MECHANICAL, COLLISION, DRIVER_ERROR, OTHER]).to_numpy()
    retired_by_code = code_series.eq("R").to_numpy() | code_series.eq("N").to_numpy()
    out["dnf"] = ((stopped | retired_by_code) & out["started"].to_numpy()).astype("int8")

    # Retired but still officially classified: covered enough race distance to
    # be given a position.  These rows are the reason ClassifiedPosition alone
    # is not a sufficient label.
    out["dnf_classified"] = (
        out["dnf"].to_numpy().astype(bool) & out["classified"].to_numpy()
    ).astype("int8")

    # Strictest reading: retired and left without a classification.
    out["dnf_strict"] = (
        out["dnf"].to_numpy().astype(bool) & ~out["classified"].to_numpy()
    ).astype("int8")

    out["finished_on_track"] = (
        took_flag & out["started"].to_numpy()
    ).astype("int8")

    out["started"] = out["started"].astype("int8")
    out["classified"] = out["classified"].astype("int8")

    if warn_on_unmapped:
        # Blank statuses first: they are invisible to unmapped_statuses (which
        # counts vocabulary, and blank is not vocabulary) but each one becomes
        # a spurious dnf=1, so they are the more damaging of the two.
        n_blank = int((~usable).sum())
        if n_blank:
            log.warning(
                "%d of %d row(s) carry no finishing status; every one of them "
                "is labelled dnf=1 on no evidence. A whole race missing its "
                "status usually means the Ergast backend rate-limited the "
                "ingest -- rebuild those rounds rather than trusting these "
                "labels.",
                n_blank,
                len(out),
            )
        leftovers = unmapped_statuses(status)
        if not leftovers.empty:
            log.warning(
                "Statuses mapped to %r (extend _CAUSE_RULES if any are real "
                "causes):\n%s",
                OTHER,
                leftovers.to_string(),
            )

    return out


def label_summary(labelled: pd.DataFrame) -> pd.DataFrame:
    """One-line-per-cause summary, useful as a sanity check after ingest."""
    starters = labelled.loc[labelled["started"] == 1]
    summary = (
        starters.groupby("dnf_cause", observed=False)
        .agg(
            rows=("dnf", "size"),
            dnf_rows=("dnf", "sum"),
            still_classified=("dnf_classified", "sum"),
        )
        .assign(share=lambda d: (d["rows"] / max(len(starters), 1)).round(4))
    )
    return summary
