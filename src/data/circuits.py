"""Circuit reference data that telemetry cannot supply.

Most of what matters about a circuit is measured from the car — see
:mod:`src.features.track_profile`.  Three things are not:

* **Wall proximity.**  A position trace shows where the car went, never how
  much room it had.  Monaco and Silverstone can share a corner radius and
  differ completely in what a mistake there costs.  That is what
  ``is_street_circuit`` and ``runoff_class`` stand in for.
* **Time of day.**  Night races run on cooler track surfaces under artificial
  light.  Both plausibly move retirement risk, neither is visible in X/Y.
* **Altitude.**  Mexico City sits at roughly 2,200 m, where thin air costs the
  turbo and the radiators dearly.  This is a genuine mechanical-failure driver
  and is invisible to a speed trace.

Everything below is a stable, published fact about the venue rather than a
measurement, and each entry names its basis.  Coordinates are deliberately
*not* hand-typed: :func:`fetch_ergast_circuits` pulls them from the Ergast /
Jolpica API, which is the primary source, and the static table carries only
what that API does not provide.

Sources
-------
* Circuit classification (street / permanent / hybrid) and night-race status:
  FIA event regulations and the circuits' own published descriptions.
* Altitude for the two circuits where it is material: Autódromo Hermanos
  Rodríguez (~2,240 m) and Interlagos (~785 m), both widely published figures
  used in F1 cooling and power-unit discussion.
* Coordinates, locality and country: Ergast / Jolpica ``/circuits`` endpoint.

When a circuit is missing from the table the pipeline emits a warning and
fills conservative defaults rather than guessing, so a new venue is visible
instead of silently mislabelled.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

#: Circuit type.  ``street`` means public roads with permanent walls;
#: ``hybrid`` means a permanent circuit using public roads or a parkland layout
#: with limited runoff; ``permanent`` means a purpose-built facility.
STREET = "street"
HYBRID = "hybrid"
PERMANENT = "permanent"

#: Keyed on the Ergast/Jolpica ``circuitId``, which is stable across seasons
#: and is what :func:`fetch_ergast_circuits` returns.
CIRCUIT_REFERENCE: dict[str, dict[str, Any]] = {
    # --- street circuits ---------------------------------------------------
    "monaco":        {"circuit_type": STREET,    "night_race": False, "altitude_m": 15},
    "baku":          {"circuit_type": STREET,    "night_race": False, "altitude_m": -25},
    "marina_bay":    {"circuit_type": STREET,    "night_race": True,  "altitude_m": 15},
    "jeddah":        {"circuit_type": STREET,    "night_race": True,  "altitude_m": 10},
    "miami":         {"circuit_type": STREET,    "night_race": False, "altitude_m": 3},
    "vegas":         {"circuit_type": STREET,    "night_race": True,  "altitude_m": 610},
    "valencia":      {"circuit_type": STREET,    "night_race": False, "altitude_m": 5},
    "sochi":         {"circuit_type": STREET,    "night_race": False, "altitude_m": 5},
    "madring":       {"circuit_type": STREET,    "night_race": False, "altitude_m": 650},
    # --- hybrid / limited-runoff ------------------------------------------
    "albert_park":   {"circuit_type": HYBRID,    "night_race": False, "altitude_m": 10},
    "villeneuve":    {"circuit_type": HYBRID,    "night_race": False, "altitude_m": 13},
    "zandvoort":     {"circuit_type": HYBRID,    "night_race": False, "altitude_m": 5},
    "imola":         {"circuit_type": HYBRID,    "night_race": False, "altitude_m": 37},
    "interlagos":    {"circuit_type": HYBRID,    "night_race": False, "altitude_m": 785},
    "monza":         {"circuit_type": HYBRID,    "night_race": False, "altitude_m": 162},
    # --- permanent circuits ------------------------------------------------
    "bahrain":       {"circuit_type": PERMANENT, "night_race": True,  "altitude_m": 7},
    "losail":        {"circuit_type": PERMANENT, "night_race": True,  "altitude_m": 15},
    "yas_marina":    {"circuit_type": PERMANENT, "night_race": True,  "altitude_m": 5},
    "rodriguez":     {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 2240},
    "red_bull_ring": {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 678},
    "spa":           {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 401},
    "silverstone":   {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 153},
    "hungaroring":   {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 264},
    "catalunya":     {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 109},
    "suzuka":        {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 45},
    "shanghai":      {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 5},
    "americas":      {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 160},
    "ricard":        {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 432},
    "hockenheimring": {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 103},
    "nurburgring":   {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 578},
    "portimao":      {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 92},
    "mugello":       {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 255},
    "istanbul":      {"circuit_type": PERMANENT, "night_race": False, "altitude_m": 130},
    "yeongam":       {"circuit_type": PERMANENT, "night_race": True,  "altitude_m": 5},
}

#: Used when a circuit is absent from the table.  Deliberately the modal case
#: rather than a guess, and always accompanied by a warning.
DEFAULT_REFERENCE: dict[str, Any] = {
    "circuit_type": PERMANENT,
    "night_race": False,
    "altitude_m": float("nan"),
}


def reference_frame() -> pd.DataFrame:
    """The static reference table as a DataFrame keyed on ``circuit_id``."""
    frame = pd.DataFrame.from_dict(CIRCUIT_REFERENCE, orient="index")
    frame.index.name = "circuit_id"
    return frame.reset_index()


def attach_reference(
    frame: pd.DataFrame, *, circuit_id_col: str = "circuit_id"
) -> pd.DataFrame:
    """Join the static reference onto a frame carrying Ergast circuit ids.

    Adds ``circuit_type``, ``night_race``, ``altitude_m`` plus the derived
    ``is_street_circuit`` and ``is_high_altitude`` flags.  Unknown circuits get
    the conservative defaults and a warning naming them, so a new venue shows
    up in the logs rather than quietly acquiring a permanent-circuit label.
    """
    if circuit_id_col not in frame.columns:
        raise KeyError(f"{circuit_id_col!r} not in frame; columns: {list(frame.columns)}")

    out = frame.merge(
        reference_frame().rename(columns={"circuit_id": circuit_id_col}),
        on=circuit_id_col,
        how="left",
    )

    unknown = sorted(out.loc[out["circuit_type"].isna(), circuit_id_col].dropna().unique())
    if unknown:
        log.warning(
            "No reference entry for %d circuit(s): %s. Filling defaults "
            "(%s, night_race=False, altitude unknown). Add them to "
            "CIRCUIT_REFERENCE.",
            len(unknown),
            ", ".join(map(str, unknown)),
            DEFAULT_REFERENCE["circuit_type"],
        )
    out["circuit_type"] = out["circuit_type"].fillna(DEFAULT_REFERENCE["circuit_type"])
    out["night_race"] = out["night_race"].fillna(DEFAULT_REFERENCE["night_race"])

    out["is_street_circuit"] = (out["circuit_type"] == STREET).astype("int8")
    out["is_limited_runoff"] = out["circuit_type"].isin([STREET, HYBRID]).astype("int8")
    out["is_night_race"] = out["night_race"].astype(bool).astype("int8")
    # Thin air costs the turbo and the radiators; 1,500 m is the point at which
    # teams run visibly different cooling packages.
    out["is_high_altitude"] = (
        pd.to_numeric(out["altitude_m"], errors="coerce") > 1500
    ).astype("int8")
    return out


def fetch_ergast_circuits(season: int | None = None) -> pd.DataFrame:
    """Fetch circuit coordinates and identity from Ergast / Jolpica.

    This is the primary source for latitude, longitude, locality and country;
    they are not hand-entered anywhere in this repository.

    Requires network access to the Ergast-compatible API that FastF1 wraps.
    Raises a clear error when that host is unreachable, rather than returning
    a half-populated frame.
    """
    from fastf1.ergast import Ergast  # imported lazily: needs network

    ergast = Ergast(result_type="pandas", auto_cast=True)
    try:
        response = ergast.get_circuits(season=season) if season else ergast.get_circuits()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Could not reach the Ergast/Jolpica API for circuit coordinates. "
            "Check network access to api.jolpi.ca, or supply coordinates "
            "manually."
        ) from exc

    frame = pd.DataFrame(response)
    renames = {
        "circuitId": "circuit_id",
        "circuitName": "circuit_name",
        "lat": "latitude",
        "long": "longitude",
        "locality": "locality",
        "country": "country",
    }
    frame = frame.rename(columns={k: v for k, v in renames.items() if k in frame.columns})
    keep = [c for c in renames.values() if c in frame.columns]
    return frame[keep]
