"""Characterise a circuit from position and speed telemetry.

The intuition this module encodes is simple: Monza and Monaco are different
places to break a car.  Monza is 80% full throttle with four heavy braking
zones, so it punishes engines and brakes; Monaco is nineteen corners in 3.3 km
with walls on both sides, so it punishes drivers.  Those two failure modes
should not share a single "circuit" dummy variable, and a one-hot per circuit
learns nothing transferable to a new venue.

So instead of naming circuits, the pipeline measures them.  FastF1 exposes the
car's position at roughly 10 Hz (``X``, ``Y``, ``Z``, in tenths of a metre)
alongside speed, throttle, brake and gear.  From a single reference lap that
yields the geometry of the track and how a car is actually driven around it:

* **Curvature.**  Resample the trace onto a uniform distance grid, smooth it,
  and differentiate twice with respect to arc length.  The curvature
  :math:`\\kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^{3/2}` gives the radius of
  every point on the lap, and its sign gives the direction of the turn.
* **Lateral load.**  :math:`a_{lat} = v^2 \\kappa`, the cornering force the
  chassis and tyres actually see.
* **Longitudinal load.**  On a distance grid, :math:`a_{long} = v \\, dv/ds`,
  which picks out braking zones without needing a time axis.
* **Duty cycle.**  Share of the lap at full throttle, share braking, number of
  discrete braking events, gear changes, DRS usage.

Every quantity is a share or a physical unit, so it is comparable across
circuits, across seasons, and across layout changes.  A resurfaced or
reprofiled circuit (Zandvoort's 2021 banking, Melbourne's 2022 reprofile, Yas
Marina's 2021 rework) simply produces a different profile for that year, which
is why profiles are keyed on ``(circuit_key, year)``.

The heavy lifting lives in :func:`build_lap_profile`, a pure function over a
DataFrame.  Nothing in it imports FastF1, so it can be tested without network
access.  :func:`profile_session` is the thin FastF1 adapter on top.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

GRAVITY_MS2 = 9.80665
KPH_TO_MS = 1.0 / 3.6
#: FastF1 position channels arrive in tenths of a metre.
POS_UNITS_PER_M = 10.0
#: DRS channel values that mean the flap is open.
DRS_OPEN_VALUES = frozenset({10, 12, 14})

#: Telemetry channels the profiler can use.  Only X, Y, Speed and Distance are
#: required; the rest degrade gracefully to NaN features when absent.
REQUIRED_CHANNELS = ("X", "Y", "Speed", "Distance")
OPTIONAL_CHANNELS = ("Z", "Throttle", "Brake", "nGear", "RPM", "DRS")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _periodic_savgol(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing that wraps around a closed lap.

    A racing lap is a loop, so the sample after the last one is the first.
    Smoothing without that knowledge puts an artificial kink at the
    start/finish line, which then shows up as a spurious corner.
    """
    from scipy.signal import savgol_filter

    n = len(values)
    if n < 2:
        return values.astype(float)
    window = min(window, n if n % 2 else n - 1)
    if window < 3:
        return values.astype(float)
    if window % 2 == 0:
        window -= 1
    polyorder = min(polyorder, window - 1)
    return savgol_filter(values, window, polyorder, mode="wrap")


def _periodic_gradient(values: np.ndarray, spacing: float) -> np.ndarray:
    """First derivative with wrap-around boundaries.

    ``np.gradient`` uses one-sided differences at the edges.  On a closed loop
    that is wrong by construction, so the array is tiled before differentiating
    and the interior slice is returned.
    """
    n = len(values)
    if n < 3:
        return np.zeros(n)
    pad = min(8, n - 1)
    tiled = np.concatenate([values[-pad:], values, values[:pad]])
    return np.gradient(tiled, spacing)[pad : pad + n]


def compute_curvature(
    x_m: np.ndarray,
    y_m: np.ndarray,
    step_m: float,
    *,
    smooth_window: int = config.CURVATURE_SMOOTH_WINDOW,
    polyorder: int = config.CURVATURE_SMOOTH_POLYORDER,
    closed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed curvature and radius of a lap trace sampled on a distance grid.

    Args:
        x_m, y_m: Position in metres, sampled every ``step_m`` of arc length.
        step_m: Grid spacing in metres.
        smooth_window: Savitzky-Golay window, in samples.  Raw position data is
            noisy enough that an unsmoothed second derivative is mostly
            sampling jitter.
        polyorder: Savitzky-Golay polynomial order.
        closed: Treat the trace as a closed loop.

    Returns:
        ``(kappa, radius)``.  ``kappa`` is signed: its sign distinguishes left
        from right turns, and its integral over a closed lap is
        :math:`\\pm 2\\pi`.  ``radius`` is ``1/|kappa|`` in metres, clipped to a
        finite ceiling on straights.
    """
    if closed:
        xs = _periodic_savgol(np.asarray(x_m, dtype=float), smooth_window, polyorder)
        ys = _periodic_savgol(np.asarray(y_m, dtype=float), smooth_window, polyorder)
        dx = _periodic_gradient(xs, step_m)
        dy = _periodic_gradient(ys, step_m)
        ddx = _periodic_gradient(dx, step_m)
        ddy = _periodic_gradient(dy, step_m)
    else:
        from scipy.signal import savgol_filter

        n = len(x_m)
        win = min(smooth_window, n if n % 2 else n - 1)
        if win >= 3:
            win = win if win % 2 else win - 1
            po = min(polyorder, win - 1)
            xs = savgol_filter(np.asarray(x_m, float), win, po)
            ys = savgol_filter(np.asarray(y_m, float), win, po)
        else:
            xs, ys = np.asarray(x_m, float), np.asarray(y_m, float)
        dx, dy = np.gradient(xs, step_m), np.gradient(ys, step_m)
        ddx, ddy = np.gradient(dx, step_m), np.gradient(dy, step_m)

    denom = np.power(dx * dx + dy * dy, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 1e-12, (dx * ddy - dy * ddx) / denom, 0.0)
        radius = np.where(np.abs(kappa) > 1e-9, 1.0 / np.abs(kappa), np.inf)
    return kappa, np.clip(radius, 0.0, 1e6)


def resample_to_distance_grid(
    tel: pd.DataFrame, step_m: float = config.TRACK_RESAMPLE_STEP_M
) -> pd.DataFrame:
    """Interpolate telemetry onto an evenly spaced distance grid.

    Telemetry samples are spaced in *time*, so a car crawling through Monaco's
    hairpin produces far more samples per metre than one flat out at Monza.
    Averaging over raw samples therefore biases every statistic toward the slow
    parts of the lap.  Resampling by distance removes that bias and makes
    circuits directly comparable.
    """
    missing = [c for c in REQUIRED_CHANNELS if c not in tel.columns]
    if missing:
        raise KeyError(f"telemetry is missing required channels: {missing}")

    frame = tel.dropna(subset=["Distance", "X", "Y", "Speed"]).copy()
    frame = frame.sort_values("Distance")
    # Distance must be strictly increasing for np.interp.
    frame = frame.loc[frame["Distance"].diff().fillna(1.0) > 0]
    if len(frame) < 10:
        raise ValueError(f"too few usable telemetry samples: {len(frame)}")

    dist = frame["Distance"].to_numpy(dtype=float)
    grid = np.arange(dist[0], dist[-1], step_m, dtype=float)
    out = {"Distance": grid}

    for channel in ("X", "Y") + OPTIONAL_CHANNELS + ("Speed",):
        if channel not in frame.columns:
            continue
        raw = frame[channel]
        if raw.dtype == bool:
            values = raw.to_numpy(dtype=float)
            out[channel] = np.interp(grid, dist, values) > 0.5
        elif pd.api.types.is_numeric_dtype(raw):
            values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
            if np.isnan(values).all():
                continue
            # Fill interior gaps so interpolation is not poisoned by NaN.
            values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
            if channel in ("nGear", "DRS"):
                # Discrete channels: nearest-neighbour, not linear.
                idx = np.searchsorted(dist, grid).clip(0, len(values) - 1)
                out[channel] = values[idx]
            else:
                out[channel] = np.interp(grid, dist, values)

    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #


def _geometry_features(grid: pd.DataFrame, step_m: float) -> dict[str, float]:
    x_m = grid["X"].to_numpy(dtype=float) / POS_UNITS_PER_M
    y_m = grid["Y"].to_numpy(dtype=float) / POS_UNITS_PER_M

    # A lap is closed when the last sample lands back near the first.
    gap = float(np.hypot(x_m[-1] - x_m[0], y_m[-1] - y_m[0]))
    closed = gap < max(50.0, 5 * step_m)

    kappa, radius = compute_curvature(x_m, y_m, step_m, closed=closed)
    abs_kappa = np.abs(kappa)
    speed_ms = grid["Speed"].to_numpy(dtype=float) * KPH_TO_MS

    lat_g = speed_ms**2 * abs_kappa / GRAVITY_MS2
    is_straight = radius > config.STRAIGHT_RADIUS_THRESHOLD_M

    # Signed turning integrated around the lap is +-2*pi for a closed circuit.
    # The sign identifies the predominant direction; the imbalance between left
    # and right turning is what loads one side of the car harder than the other.
    turning_rad = float(np.sum(kappa) * step_m)
    left = float(np.sum(kappa[kappa > 0]) * step_m)
    right = float(-np.sum(kappa[kappa < 0]) * step_m)
    total_turn = left + right

    feats: dict[str, float] = {
        "lap_length_m": float(grid["Distance"].iloc[-1] - grid["Distance"].iloc[0]),
        "trace_closure_gap_m": gap,
        "mean_abs_curvature_1pm": float(np.mean(abs_kappa)),
        "median_corner_radius_m": float(np.median(radius[~is_straight]))
        if (~is_straight).any()
        else np.nan,
        "min_corner_radius_m": float(np.min(radius)) if len(radius) else np.nan,
        "pct_dist_straight": float(np.mean(is_straight)),
        "lat_g_mean": float(np.mean(lat_g)),
        "lat_g_p95": float(np.percentile(lat_g, 95)),
        "lat_g_max": float(np.max(lat_g)),
        "signed_turning_rad": turning_rad,
        # Convention-free: 0 means a perfectly balanced circuit, 1 means every
        # corner turns the same way.
        "curvature_asymmetry": float(abs(left - right) / total_turn)
        if total_turn > 0
        else np.nan,
        "turn_direction_sign": float(np.sign(turning_rad)),
    }

    if "Z" in grid.columns:
        z_m = grid["Z"].to_numpy(dtype=float) / POS_UNITS_PER_M
        dz = np.diff(z_m)
        feats["elevation_range_m"] = float(np.ptp(z_m))
        feats["elevation_gain_m"] = float(np.sum(dz[dz > 0]))
        feats["max_gradient_pct"] = float(np.max(np.abs(dz)) / step_m * 100.0)
    else:
        feats["elevation_range_m"] = np.nan
        feats["elevation_gain_m"] = np.nan
        feats["max_gradient_pct"] = np.nan

    return feats


def _speed_features(grid: pd.DataFrame, step_m: float) -> dict[str, float]:
    speed = grid["Speed"].to_numpy(dtype=float)
    speed_ms = speed * KPH_TO_MS

    # a_long = v * dv/ds.  Works directly on the distance grid, no time axis.
    dv_ds = _periodic_gradient(speed_ms, step_m)
    a_long = speed_ms * dv_ds
    decel = -a_long[a_long < 0] / GRAVITY_MS2

    feats: dict[str, float] = {
        "speed_mean_kph": float(np.mean(speed)),
        "speed_median_kph": float(np.median(speed)),
        "speed_std_kph": float(np.std(speed)),
        "speed_min_kph": float(np.min(speed)),
        "speed_max_kph": float(np.max(speed)),
        "speed_p10_kph": float(np.percentile(speed, 10)),
        "speed_p90_kph": float(np.percentile(speed, 90)),
        "speed_range_kph": float(np.max(speed) - np.min(speed)),
        "pct_dist_above_250kph": float(
            np.mean(speed >= config.HIGH_SPEED_THRESHOLD_KPH)
        ),
        "pct_dist_below_120kph": float(
            np.mean(speed <= config.LOW_SPEED_THRESHOLD_KPH)
        ),
        "decel_g_mean": float(np.mean(decel)) if decel.size else np.nan,
        "decel_g_p95": float(np.percentile(decel, 95)) if decel.size else np.nan,
        "decel_g_max": float(np.max(decel)) if decel.size else np.nan,
    }

    if "Throttle" in grid.columns:
        throttle = grid["Throttle"].to_numpy(dtype=float)
        # 104 is FastF1's "no data" sentinel; treat anything above 100 as full.
        full = throttle >= config.FULL_THROTTLE_THRESHOLD
        feats["pct_full_throttle"] = float(np.mean(full))
        feats["pct_throttle_lift"] = float(np.mean((throttle > 5) & ~full))
    else:
        feats["pct_full_throttle"] = np.nan
        feats["pct_throttle_lift"] = np.nan

    if "Brake" in grid.columns:
        brake = grid["Brake"].to_numpy()
        brake = brake.astype(bool) if brake.dtype != bool else brake
        feats["pct_braking"] = float(np.mean(brake))
        feats["n_braking_zones"] = float(
            _count_events(brake, min_run=max(2, int(30.0 / step_m)))
        )
        feats["braking_m_per_lap"] = float(np.sum(brake) * step_m)
    else:
        feats["pct_braking"] = np.nan
        feats["n_braking_zones"] = np.nan
        feats["braking_m_per_lap"] = np.nan

    if "nGear" in grid.columns:
        gear = np.round(grid["nGear"].to_numpy(dtype=float))
        feats["n_gear_changes"] = float(np.sum(np.diff(gear) != 0))
        feats["gear_mean"] = float(np.mean(gear))
    else:
        feats["n_gear_changes"] = np.nan
        feats["gear_mean"] = np.nan

    if "DRS" in grid.columns:
        drs = np.round(grid["DRS"].to_numpy(dtype=float)).astype(int)
        feats["pct_drs_open"] = float(np.mean(np.isin(drs, list(DRS_OPEN_VALUES))))
    else:
        feats["pct_drs_open"] = np.nan

    if "RPM" in grid.columns:
        rpm = grid["RPM"].to_numpy(dtype=float)
        feats["rpm_mean"] = float(np.mean(rpm))
        feats["rpm_p95"] = float(np.percentile(rpm, 95))
    else:
        feats["rpm_mean"] = np.nan
        feats["rpm_p95"] = np.nan

    return feats


def _count_events(flags: np.ndarray, min_run: int = 1) -> int:
    """Count runs of ``True`` at least ``min_run`` samples long.

    Used for braking zones: a single noisy sample of brake pressure is not a
    braking zone, so short runs are discarded.
    """
    if flags.size == 0:
        return 0
    padded = np.concatenate([[False], flags.astype(bool), [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return int(np.sum((ends - starts) >= min_run))


def build_lap_profile(
    tel: pd.DataFrame,
    *,
    step_m: float = config.TRACK_RESAMPLE_STEP_M,
    corners: pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Profile a circuit from one reference lap of telemetry.

    Args:
        tel: Telemetry for a single lap.  Must carry ``X``, ``Y``, ``Speed``
            and ``Distance``; ``Z``, ``Throttle``, ``Brake``, ``nGear``,
            ``RPM`` and ``DRS`` are used when present.
        step_m: Distance-grid resolution in metres.
        corners: Optional corner table from ``Session.get_circuit_info()``.
            Supplies the official corner count, which is a cleaner signal than
            counting curvature peaks.
        metadata: Extra key/value pairs copied straight into the result, e.g.
            ``{"circuit_key": 39, "year": 2024}``.

    Returns:
        A flat dict of features, ready to become one row of the circuit table.
    """
    grid = resample_to_distance_grid(tel, step_m=step_m)
    profile: dict[str, Any] = dict(metadata or {})
    profile.update(_geometry_features(grid, step_m))
    profile.update(_speed_features(grid, step_m))

    lap_km = profile["lap_length_m"] / 1000.0
    if corners is not None and len(corners):
        profile["n_corners"] = float(len(corners))
    else:
        # Fall back to counting sustained curvature peaks.
        x_m = grid["X"].to_numpy(float) / POS_UNITS_PER_M
        y_m = grid["Y"].to_numpy(float) / POS_UNITS_PER_M
        kappa, radius = compute_curvature(x_m, y_m, step_m)
        profile["n_corners"] = float(
            _count_events(
                radius < config.STRAIGHT_RADIUS_THRESHOLD_M,
                min_run=max(2, int(20.0 / step_m)),
            )
        )
    profile["corners_per_km"] = profile["n_corners"] / lap_km if lap_km > 0 else np.nan
    profile["braking_zones_per_km"] = (
        profile["n_braking_zones"] / lap_km
        if lap_km > 0 and not pd.isna(profile["n_braking_zones"])
        else np.nan
    )
    profile["grid_samples"] = float(len(grid))
    profile["resample_step_m"] = float(step_m)
    return profile


# --------------------------------------------------------------------------- #
# Cross-circuit normalisation
# --------------------------------------------------------------------------- #

#: Columns combined into the published composite indices, with their sign.
#: A positive weight means "more of this makes the index larger".
SPEED_INDEX_WEIGHTS = {
    "speed_mean_kph": 1.0,
    "pct_dist_above_250kph": 1.0,
    "pct_full_throttle": 1.0,
    "pct_dist_below_120kph": -1.0,
    "corners_per_km": -1.0,
}

#: Hypothesis, not established fact: these are the circuit traits expected to
#: drive *mechanical* retirements.  Sustained full throttle loads the power
#: unit, repeated heavy braking loads brakes and gearbox, and gear changes
#: load the transmission.  Test it before trusting it.
MECHANICAL_STRESS_WEIGHTS = {
    "pct_full_throttle": 1.0,
    "braking_zones_per_km": 1.0,
    "decel_g_p95": 1.0,
    "n_gear_changes": 1.0,
    "rpm_mean": 0.5,
}

#: Hypothesis for *incident* retirements: tight, twisty, low-radius circuits
#: with little room leave less margin for error.  Combine with the
#: ``is_street_circuit`` flag, which carries the wall proximity this cannot see.
INCIDENT_EXPOSURE_WEIGHTS = {
    "corners_per_km": 1.0,
    "pct_dist_below_120kph": 1.0,
    "median_corner_radius_m": -1.0,
    "pct_dist_straight": -1.0,
}


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def add_composite_indices(
    profiles: pd.DataFrame,
    weight_sets: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Add cross-circuit composite indices to a profile table.

    Each index is the weighted mean of z-scored components, so it is centred on
    zero across the circuits present and reads in standard deviations.  With
    the default weights, Monza sits near the top of ``track_speed_index`` and
    Monaco near the bottom.

    The indices are deliberately transparent linear combinations rather than a
    fitted PCA: a component's contribution stays interpretable, and adding a
    season does not silently rotate the axes.
    """
    weight_sets = weight_sets or {
        "track_speed_index": SPEED_INDEX_WEIGHTS,
        "mechanical_stress_index": MECHANICAL_STRESS_WEIGHTS,
        "incident_exposure_index": INCIDENT_EXPOSURE_WEIGHTS,
    }
    out = profiles.copy()
    for name, weights in weight_sets.items():
        usable = {c: w for c, w in weights.items() if c in out.columns}
        if not usable:
            out[name] = np.nan
            continue
        parts, total = [], 0.0
        for col, weight in usable.items():
            values = pd.to_numeric(out[col], errors="coerce")
            if values.notna().sum() < 2:
                continue
            parts.append(_zscore(values.fillna(values.mean())) * weight)
            total += abs(weight)
        out[name] = (
            sum(parts) / total if parts and total > 0 else pd.Series(np.nan, out.index)
        )
    return out


def aggregate_circuit_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-year profiles into one row per circuit.

    Used as a fallback when a given season has no usable telemetry.  The median
    is preferred over the mean because a single truncated or traffic-affected
    reference lap should not drag the profile with it.
    """
    keys = [k for k in ("circuit_key", "circuit_name") if k in profiles.columns]
    if not keys:
        raise KeyError("profiles must carry circuit_key or circuit_name")
    key = keys[0]
    # The grouping key is often numeric itself, so it has to come out of the
    # aggregated columns or reset_index collides with it.
    numeric = [
        c
        for c in profiles.select_dtypes(include=[np.number]).columns
        if c != key
    ]
    grouped = profiles.groupby(key, dropna=False)
    agg = grouped[numeric].median()
    agg["profile_years"] = grouped.size()
    if "circuit_name" in profiles.columns:
        agg["circuit_name"] = grouped["circuit_name"].agg(
            lambda s: s.dropna().iloc[-1] if s.notna().any() else None
        )
    return agg.reset_index()
