"""Which features exist, and when each one becomes knowable.

A retirement model is only useful if it can run at the moment you want an
answer, and different questions arrive at different times:

``pre_weekend``
    Known on the Monday before the race: form, reliability history, the
    circuit's measured character, the calendar.  This is the feature set for
    season-long simulation and for pricing a race before anyone has driven.

``post_quali``
    Adds grid position and qualifying pace.  Grid slot is one of the strongest
    single predictors of retirement — the back of the grid both breaks more and
    gets collected at turn one more — so this set should score materially
    better.  It can only be run from Saturday evening.

``race_day``
    Adds the weather that actually occurred.  **This is not available at
    forecast time.**  It is kept because a retrospective model including it
    quantifies how much of retirement risk is weather, which is worth knowing
    even though you cannot use it to predict.  Never report ``race_day`` scores
    as forecast accuracy.

Selecting features by stage rather than by hand is what stops a grid-position
column from quietly wandering into a pre-weekend model.  :func:`feature_columns`
is the single place that decision is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd

Stage = Literal["pre_weekend", "post_quali", "race_day"]
Kind = Literal["numeric", "categorical", "binary"]

#: Stages in the order information arrives.  A model at one stage may use every
#: feature from that stage and all earlier ones.
STAGE_ORDER: tuple[Stage, ...] = ("pre_weekend", "post_quali", "race_day")


@dataclass(frozen=True)
class Feature:
    """One modelling column and everything the pipeline needs to know about it."""

    name: str
    stage: Stage
    kind: Kind
    description: str
    #: False for features that are structural facts rather than history, so a
    #: missing value means a genuine data problem rather than a cold start.
    allows_cold_start_nan: bool = True


def _f(name: str, stage: Stage, kind: Kind, description: str, **kw) -> Feature:
    return Feature(name=name, stage=stage, kind=kind, description=description, **kw)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

FEATURES: tuple[Feature, ...] = (
    # ---- driver reliability history (pre-weekend) --------------------------
    _f("driver_dnf_rate_5", "pre_weekend", "numeric",
       "Driver's retirement rate over their previous 5 races."),
    _f("driver_dnf_rate_10", "pre_weekend", "numeric",
       "Driver's retirement rate over their previous 10 races."),
    _f("driver_dnf_rate_career", "pre_weekend", "numeric",
       "Driver's retirement rate over every prior race in the dataset."),
    _f("driver_mech_dnf_rate_10", "pre_weekend", "numeric",
       "Share of the driver's previous 10 races ending in a car failure."),
    _f("driver_incident_dnf_rate_10", "pre_weekend", "numeric",
       "Share of the driver's previous 10 races ending in a collision or spin."),
    _f("driver_races_since_dnf", "pre_weekend", "numeric",
       "Races since the driver last retired; NaN before their first retirement."),
    _f("driver_races_to_date", "pre_weekend", "numeric",
       "Experience: races started before this one.", allows_cold_start_nan=False),
    _f("driver_points_rate_5", "pre_weekend", "numeric",
       "Mean points over the previous 5 races; largely a proxy for car pace."),
    _f("driver_avg_grid_5", "pre_weekend", "numeric",
       "Mean starting position over the previous 5 races."),

    # ---- team reliability history (pre-weekend) ---------------------------
    _f("team_dnf_rate_5", "pre_weekend", "numeric",
       "Team's per-car retirement rate over its previous 5 races."),
    _f("team_dnf_rate_10", "pre_weekend", "numeric",
       "Team's per-car retirement rate over its previous 10 races."),
    _f("team_mech_dnf_rate_10", "pre_weekend", "numeric",
       "Team's per-car mechanical retirement rate over its previous 10 races."),
    _f("team_dnf_rate_career", "pre_weekend", "numeric",
       "Team's per-car retirement rate over every prior race."),
    _f("teammate_dnf_rate_10", "pre_weekend", "numeric",
       "The other car's own prior-race retirement rate. Two cars share a power "
       "unit and a design office, so the sister car carries information this "
       "driver's record does not."),
    _f("team_races_this_season", "pre_weekend", "numeric",
       "Races the team has completed this season; low values mean a car still "
       "early in its reliability shakedown.", allows_cold_start_nan=False),

    # ---- pairing (pre-weekend) --------------------------------------------
    _f("pair_races_to_date", "pre_weekend", "numeric",
       "Races this driver has started for this team.", allows_cold_start_nan=False),
    _f("is_new_pairing", "pre_weekend", "binary",
       "1 for the first five races of a driver-team combination.",
       allows_cold_start_nan=False),
    _f("pair_dnf_rate_10", "pre_weekend", "numeric",
       "Retirement rate of this driver-team pairing over its previous 10 races."),

    # ---- circuit history (pre-weekend) ------------------------------------
    _f("circuit_dnf_rate_prior", "pre_weekend", "numeric",
       "Field-wide retirement rate at this circuit across all prior visits."),
    _f("circuit_dnf_rate_3", "pre_weekend", "numeric",
       "Field-wide retirement rate over the circuit's previous 3 visits."),
    _f("driver_dnf_rate_at_circuit", "pre_weekend", "numeric",
       "This driver's retirement rate at this circuit, prior visits only."),
    _f("driver_starts_at_circuit", "pre_weekend", "numeric",
       "How many times the driver has started here before.",
       allows_cold_start_nan=False),

    # ---- measured circuit character (pre-weekend) -------------------------
    _f("track_speed_index", "pre_weekend", "numeric",
       "Composite of mean speed, high-speed share and full-throttle share, "
       "z-scored across circuits. Monza high, Monaco low."),
    _f("mechanical_stress_index", "pre_weekend", "numeric",
       "Composite of full-throttle share, braking-zone density, peak "
       "deceleration and gear changes. Hypothesised driver of car failures."),
    _f("incident_exposure_index", "pre_weekend", "numeric",
       "Composite of corner density, low-speed share and corner radius. "
       "Hypothesised driver of collision and driver-error retirements."),
    _f("speed_mean_kph", "pre_weekend", "numeric",
       "Mean speed around a reference lap, distance-weighted."),
    _f("pct_full_throttle", "pre_weekend", "numeric",
       "Share of lap distance at 95% throttle or more; power-unit load."),
    _f("pct_dist_above_250kph", "pre_weekend", "numeric",
       "Share of lap distance above 250 km/h."),
    _f("pct_dist_below_120kph", "pre_weekend", "numeric",
       "Share of lap distance below 120 km/h."),
    _f("corners_per_km", "pre_weekend", "numeric",
       "Corner count divided by lap length."),
    _f("braking_zones_per_km", "pre_weekend", "numeric",
       "Discrete braking zones per kilometre; brake and gearbox load."),
    _f("decel_g_p95", "pre_weekend", "numeric",
       "95th-percentile braking deceleration in g."),
    _f("lat_g_mean", "pre_weekend", "numeric",
       "Mean lateral acceleration in g, from v-squared times curvature."),
    _f("median_corner_radius_m", "pre_weekend", "numeric",
       "Median radius of the non-straight parts of the lap."),
    _f("pct_dist_straight", "pre_weekend", "numeric",
       "Share of lap distance with a radius above 500 m."),
    _f("elevation_range_m", "pre_weekend", "numeric",
       "Highest point minus lowest point around the lap."),
    _f("curvature_asymmetry", "pre_weekend", "numeric",
       "How one-directional the circuit is, from 0 (balanced) to 1 (all one "
       "way); drives one-sided tyre and brake load."),
    _f("n_gear_changes", "pre_weekend", "numeric",
       "Gear changes per lap; transmission duty cycle."),
    _f("lap_length_m", "pre_weekend", "numeric", "Lap distance in metres."),

    # ---- secondary circuit measurements (pre-weekend) ---------------------
    # Not in the recommended starting set, but measured and available.  They
    # correlate heavily with the headline features above, so add them one at a
    # time and watch the ablation rather than throwing them all in at once.
    _f("speed_median_kph", "pre_weekend", "numeric", "Median lap speed."),
    _f("speed_min_kph", "pre_weekend", "numeric",
       "Slowest point of the lap; the speed of the tightest corner."),
    _f("speed_max_kph", "pre_weekend", "numeric", "Top speed reached on the lap."),
    _f("speed_p10_kph", "pre_weekend", "numeric", "10th-percentile lap speed."),
    _f("speed_p90_kph", "pre_weekend", "numeric", "90th-percentile lap speed."),
    _f("speed_std_kph", "pre_weekend", "numeric",
       "Speed dispersion; high on stop-start circuits, low on flowing ones."),
    _f("speed_range_kph", "pre_weekend", "numeric", "Top speed minus slowest speed."),
    _f("pct_braking", "pre_weekend", "numeric",
       "Share of lap distance with the brakes applied."),
    _f("pct_throttle_lift", "pre_weekend", "numeric",
       "Share of lap distance on partial throttle."),
    _f("pct_drs_open", "pre_weekend", "numeric",
       "Share of lap distance with DRS open; a proxy for usable straight."),
    _f("braking_m_per_lap", "pre_weekend", "numeric",
       "Metres per lap spent braking; total brake energy proxy."),
    _f("n_braking_zones", "pre_weekend", "numeric", "Discrete braking zones per lap."),
    _f("n_corners", "pre_weekend", "numeric",
       "Corner count, from the official corner table where available."),
    _f("decel_g_mean", "pre_weekend", "numeric", "Mean braking deceleration in g."),
    _f("decel_g_max", "pre_weekend", "numeric", "Peak braking deceleration in g."),
    _f("lat_g_p95", "pre_weekend", "numeric", "95th-percentile lateral acceleration."),
    _f("lat_g_max", "pre_weekend", "numeric", "Peak lateral acceleration in g."),
    _f("mean_abs_curvature_1pm", "pre_weekend", "numeric",
       "Mean absolute curvature per metre around the lap."),
    _f("min_corner_radius_m", "pre_weekend", "numeric",
       "Tightest radius on the lap; the hairpin."),
    _f("elevation_gain_m", "pre_weekend", "numeric", "Total climb around a lap."),
    _f("max_gradient_pct", "pre_weekend", "numeric", "Steepest gradient on the lap."),
    _f("gear_mean", "pre_weekend", "numeric", "Mean gear held around the lap."),
    _f("rpm_mean", "pre_weekend", "numeric", "Mean engine speed."),
    _f("rpm_p95", "pre_weekend", "numeric", "95th-percentile engine speed."),

    # ---- shorter-window history variants (pre-weekend) --------------------
    _f("driver_mech_dnf_rate_5", "pre_weekend", "numeric",
       "Driver's mechanical retirement rate over their previous 5 races."),
    _f("driver_incident_dnf_rate_5", "pre_weekend", "numeric",
       "Driver's incident retirement rate over their previous 5 races."),
    _f("team_mech_dnf_rate_5", "pre_weekend", "numeric",
       "Team's mechanical retirement rate over its previous 5 races."),
    _f("team_races_to_date", "pre_weekend", "numeric",
       "Races the team has entered before this one.", allows_cold_start_nan=False),
    _f("circuit_races_prior", "pre_weekend", "numeric",
       "Times this circuit has appeared in the dataset before.",
       allows_cold_start_nan=False),

    # ---- circuit reference facts (pre-weekend) ----------------------------
    _f("is_street_circuit", "pre_weekend", "binary",
       "Public roads with permanent walls. Carries the runoff a position trace "
       "cannot see.", allows_cold_start_nan=False),
    _f("is_limited_runoff", "pre_weekend", "binary",
       "Street or hybrid circuit; limited room for a recoverable mistake.",
       allows_cold_start_nan=False),
    _f("is_night_race", "pre_weekend", "binary",
       "Run under floodlights, on a cooler surface.", allows_cold_start_nan=False),
    _f("is_high_altitude", "pre_weekend", "binary",
       "Above 1,500 m, where thin air strains turbo and cooling.",
       allows_cold_start_nan=False),

    # ---- calendar and field (pre-weekend) ---------------------------------
    _f("season_round", "pre_weekend", "numeric",
       "Round number within the season.", allows_cold_start_nan=False),
    _f("is_season_opener", "pre_weekend", "binary",
       "First race of the season, when new-car reliability is at its worst.",
       allows_cold_start_nan=False),
    _f("field_size", "pre_weekend", "numeric",
       "Cars taking the start.", allows_cold_start_nan=False),
    _f("days_since_last_race", "pre_weekend", "numeric",
       "Days since this driver's previous race."),
    _f("regulation_era", "pre_weekend", "categorical",
       "Technical-regulation era; 2022 onward is the ground-effect ruleset.",
       allows_cold_start_nan=False),

    # ---- qualifying-dependent (post-quali) --------------------------------
    _f("grid_position", "post_quali", "numeric",
       "Starting position, with a pit-lane start moved to the back of the grid.",
       allows_cold_start_nan=False),
    _f("grid_position_pct", "post_quali", "numeric",
       "Grid position as a fraction of field size, so seasons with different "
       "entry counts are comparable.", allows_cold_start_nan=False),
    _f("is_back_half_of_grid", "post_quali", "binary",
       "Starting in the slower half of the field.", allows_cold_start_nan=False),
    _f("starts_from_pit_lane", "post_quali", "binary",
       "Started from the pit lane rather than a grid slot.",
       allows_cold_start_nan=False),
    _f("grid_penalty_places", "post_quali", "numeric",
       "Places lost between qualifying and the grid, usually a component "
       "penalty — which is itself a reliability signal."),

    # ---- observed conditions (race day; NOT available at forecast time) ---
    _f("air_temp_c_mean", "race_day", "numeric", "Mean air temperature."),
    _f("track_temp_c_mean", "race_day", "numeric", "Mean track temperature."),
    _f("track_temp_c_max", "race_day", "numeric", "Peak track temperature."),
    _f("humidity_pct_mean", "race_day", "numeric", "Mean relative humidity."),
    _f("wind_speed_ms_mean", "race_day", "numeric", "Mean wind speed."),
    _f("rain_share", "race_day", "numeric",
       "Share of weather samples reporting rainfall."),
    _f("any_rain", "race_day", "binary", "Any rainfall recorded during the race."),
)

BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}


def feature_columns(
    stage: Stage = "post_quali",
    *,
    kinds: Iterable[Kind] | None = None,
    available: Iterable[str] | None = None,
) -> list[str]:
    """Feature names usable at ``stage``.

    Args:
        stage: The latest stage of information the model is allowed to see.
            Features from earlier stages are always included.
        kinds: Restrict to these kinds, e.g. ``("numeric", "binary")`` to skip
            categoricals that need encoding.
        available: Restrict to columns actually present in a DataFrame.  Pass
            ``df.columns`` so a partially built table does not raise.

    Returns:
        Column names, in registry order.

    >>> "grid_position" in feature_columns("pre_weekend")
    False
    >>> "grid_position" in feature_columns("post_quali")
    True
    >>> "rain_share" in feature_columns("post_quali")
    False
    """
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    cutoff = STAGE_ORDER.index(stage)
    allowed_kinds = set(kinds) if kinds else None
    present = set(available) if available is not None else None

    return [
        f.name
        for f in FEATURES
        if STAGE_ORDER.index(f.stage) <= cutoff
        and (allowed_kinds is None or f.kind in allowed_kinds)
        and (present is None or f.name in present)
    ]


def registry_frame() -> pd.DataFrame:
    """The registry as a DataFrame, for the data dictionary and for review."""
    return pd.DataFrame(
        [
            {
                "feature": f.name,
                "stage": f.stage,
                "kind": f.kind,
                "allows_cold_start_nan": f.allows_cold_start_nan,
                "description": f.description,
            }
            for f in FEATURES
        ]
    )


#: Numeric columns that are targets, identifiers, raw inputs or diagnostics
#: rather than features.  Excluded from the audit so its output stays readable
#: and a genuine unregistered feature is not lost among them.
NON_FEATURE_COLUMNS = frozenset(
    {
        # targets and outcome descriptions
        "dnf", "dnf_strict", "dnf_classified", "dnf_mechanical", "dnf_incident",
        "finished_on_track", "classified", "started",
        # raw result fields, superseded by cleaned features
        "Position", "GridPosition", "Points", "Laps", "DriverNumber",
        "QualifyingPosition", "total_laps",
        # identifiers and calendar keys
        "Year", "RoundNumber", "circuit_key", "latitude", "longitude",
        "altitude_m", "night_race",
        # intermediates and data-quality flags
        "team_cars", "has_track_profile", "profile_years",
        "grid_samples", "resample_step_m", "trace_closure_gap_m",
        "reference_lap_time_s", "signed_turning_rad", "turn_direction_sign",
    }
)


def audit_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the registry against a built dataset.

    Reports registered features missing from the frame, and numeric columns in
    the frame that no one registered.  Unregistered columns are the ones to
    look at hardest: without a stage tag nothing stops a column that is only
    knowable on Saturday from being handed to a Monday model.
    """
    registered = set(BY_NAME)
    present = set(frame.columns)
    rows = [
        {"feature": name, "issue": "registered but missing from dataset"}
        for name in sorted(registered - present)
    ]
    numeric = {
        c
        for c in frame.select_dtypes("number").columns
        if not c.startswith("_") and c not in NON_FEATURE_COLUMNS
    }
    rows.extend(
        {"feature": name, "issue": "in dataset but not registered"}
        for name in sorted(numeric - registered)
    )
    return pd.DataFrame(rows, columns=["feature", "issue"])
