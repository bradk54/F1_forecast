"""Synthetic, FastF1-shaped data for testing without network access.

The F1 timing APIs are unreachable from some environments (and rate-limited
everywhere), so the pipeline is validated against generated data whose schema
matches what FastF1 returns.  Three generators live here:

``make_circuit_telemetry``
    Builds a track from a segment list (straights and constant-radius arcs),
    integrates it into an X/Y/Z trace, then runs a quasi-steady-state lap
    simulation to produce a coherent speed, throttle, brake and gear trace.
    Because the geometry is stated exactly, curvature and lateral-load features
    can be checked against closed-form answers.

``make_fourier_telemetry``
    Same idea, but the layout comes from a Fourier series, so the trace closes
    to machine precision.  This is what exercises the profiler's periodic
    smoothing and wrap-around gradient paths, which an open segment list
    cannot.

``make_race_results``
    A season of driver-race rows carrying the columns ``Session.results``
    provides, including the ``Status`` and ``ClassifiedPosition`` fields the
    labelling code depends on.  Drivers and teams are given latent reliability,
    so rolling features have real signal to recover rather than pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRAVITY = 9.80665


@dataclass(frozen=True)
class Segment:
    """One piece of track: a straight if ``radius`` is None, else an arc.

    ``radius`` is in metres; positive turns left, negative turns right.
    ``length`` is arc length in metres.
    """

    length: float
    radius: float | None = None

    @property
    def angle(self) -> float:
        """Signed heading change through this segment, in radians."""
        return 0.0 if self.radius is None else self.length / self.radius


def build_trace(
    segments: list[Segment], step_m: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a segment list into an ``(x, y, radius)`` trace in metres."""
    xs, ys, radii = [0.0], [0.0], []
    heading = 0.0
    for seg in segments:
        n = max(1, int(round(seg.length / step_m)))
        ds = seg.length / n
        for _ in range(n):
            if seg.radius is None:
                radii.append(np.inf)
            else:
                heading += ds / seg.radius
                radii.append(abs(seg.radius))
            xs.append(xs[-1] + ds * np.cos(heading))
            ys.append(ys[-1] + ds * np.sin(heading))
    # Drop the duplicated closing point so len(x) == len(radius).
    return np.array(xs[:-1]), np.array(ys[:-1]), np.array(radii)


def close_track(segments: list[Segment], tolerance: float = 0.25) -> list[Segment]:
    """Adjust arc lengths so the lap turns through exactly 2*pi.

    The turning-tangent theorem says a simple closed curve turns through
    exactly 2*pi, so a segment list whose signed turning misses that cannot be
    a real lap.  The deficit is spread across the arcs in proportion to how
    much each already turns, which keeps the shape recognisable and leaves the
    radii — and therefore the cornering speeds — untouched.

    This fixes *total turning*, not positional closure: a hand-authored list
    generally still ends some distance from where it started.  The profiler
    detects that through ``trace_closure_gap_m`` and falls back to
    non-periodic smoothing, which is itself worth testing.  For an exactly
    closed trace use :func:`oval` or :func:`make_fourier_telemetry`.

    Raises:
        ValueError: if closing the track would change any arc by more than
            ``tolerance`` of its angle.  A correction that large means the
            layout is nowhere near a lap and the fixture is wrong.
    """
    arcs = [(i, s.angle) for i, s in enumerate(segments) if s.radius is not None]
    if not arcs:
        raise ValueError("track has no arcs to adjust")

    total = sum(a for _, a in arcs)
    abs_total = sum(abs(a) for _, a in arcs)
    target = 2 * np.pi * (np.sign(total) if total != 0 else 1.0)
    deficit = target - total

    out = list(segments)
    for i, angle in arcs:
        adjustment = deficit * (abs(angle) / abs_total)
        if abs(adjustment) > tolerance * abs(angle):
            raise ValueError(
                f"closing the track would change segment {i} by "
                f"{abs(adjustment):.3f} rad (from {angle:.3f} rad); the layout "
                "is too far from a closed lap to fix by scaling"
            )
        radius = segments[i].radius
        assert radius is not None
        out[i] = Segment(abs((angle + adjustment) * radius), radius)
    return out


def simulate_speed(
    radii: np.ndarray,
    step_m: float,
    *,
    v_max_kph: float = 340.0,
    lat_g_max: float = 4.5,
    accel_g: float = 1.2,
    brake_g: float = 5.0,
    passes: int = 3,
) -> np.ndarray:
    """Quasi-steady-state lap speed in km/h.

    Grip-limited cornering speed first, then a forward pass for the traction
    limit and a backward pass for the braking limit, iterated so the closed
    loop converges.  Defaults sit at modern-F1 orders of magnitude.
    """
    v_max = v_max_kph / 3.6
    n = len(radii)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.minimum(v_max, np.sqrt(lat_g_max * GRAVITY * radii))
    v = np.nan_to_num(v, posinf=v_max, nan=v_max)

    a_acc, a_brk = accel_g * GRAVITY, brake_g * GRAVITY
    for _ in range(passes):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_acc * step_m))
        for i in range(n - 1, -1, -1):
            j = (i - 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_brk * step_m))
    return v * 3.6


def _assemble_telemetry(
    dist: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    radii: np.ndarray,
    speed_kph: np.ndarray,
    step_m: float,
    elevation_amplitude_m: float,
    sample_by_time: bool,
    lat_g_max: float = 4.5,
) -> pd.DataFrame:
    """Assemble derived channels into a FastF1-shaped telemetry frame."""
    speed_ms = speed_kph / 3.6
    a_long = speed_ms * np.gradient(speed_ms, step_m)

    # Throttle follows the traction circle.  A tyre has one budget of grip to
    # spend on cornering and driving; whatever cornering takes is unavailable
    # for throttle.  So the driver is flat only where lateral load is near
    # zero, and barely on the pedal mid-hairpin:
    #
    #     throttle ~ sqrt(1 - (a_lat / a_lat_max)^2)
    #
    # Without this a car is at 100% throttle everywhere it is not braking,
    # which would put Monaco's full-throttle share up beside Monza's.
    a_lat_max = lat_g_max * GRAVITY
    with np.errstate(divide="ignore", invalid="ignore"):
        a_lat = np.where(np.isfinite(radii), speed_ms**2 / radii, 0.0)
    grip_left = np.clip(1.0 - (a_lat / a_lat_max) ** 2, 0.0, 1.0)

    brake = a_long < -0.5
    throttle = 100.0 * np.sqrt(grip_left)
    throttle[brake] = 0.0
    throttle = np.clip(throttle, 0.0, 100.0)

    gear = np.clip(np.digitize(speed_kph, [80, 130, 180, 220, 260, 300, 330]) + 1, 1, 8)
    rpm = 6000.0 + 6000.0 * (speed_kph / max(float(speed_kph.max()), 1.0))
    # DRS open on long straights above 280 km/h, using FastF1's "on" code 12.
    drs = np.where((radii > 800) & (speed_kph > 280), 12, 0)

    span = max(float(dist[-1]), 1.0)
    z_m = elevation_amplitude_m / 2 * np.sin(2 * np.pi * dist / span)

    frame = pd.DataFrame(
        {
            "Distance": dist,
            "X": x_m * 10.0,  # FastF1 position channels are in tenths of a metre
            "Y": y_m * 10.0,
            "Z": z_m * 10.0,
            "Speed": speed_kph,
            "Throttle": throttle,
            "Brake": brake,
            "nGear": gear,
            "RPM": rpm,
            "DRS": drs,
        }
    )
    dt = step_m / np.maximum(speed_ms, 1.0)
    frame["Time"] = pd.to_timedelta(np.cumsum(dt), unit="s")

    if sample_by_time:
        # Reproduce the bias real telemetry has toward slow corners: samples
        # are evenly spaced in time, so a hairpin yields far more of them per
        # metre than a straight does.
        elapsed = np.cumsum(dt)
        grid = np.arange(0.0, elapsed[-1], 0.24)  # FastF1's ~240 ms car-data rate
        idx = np.searchsorted(elapsed, grid).clip(0, len(frame) - 1)
        frame = frame.iloc[idx].reset_index(drop=True)
    return frame


def make_circuit_telemetry(
    segments: list[Segment],
    *,
    step_m: float = 1.0,
    elevation_amplitude_m: float = 0.0,
    sample_by_time: bool = True,
    **speed_kwargs,
) -> pd.DataFrame:
    """FastF1-shaped telemetry for one lap of a segment-defined track."""
    x_m, y_m, radii = build_trace(segments, step_m=step_m)
    speed_kph = simulate_speed(radii, step_m, **speed_kwargs)
    dist = np.arange(len(x_m), dtype=float) * step_m
    return _assemble_telemetry(
        dist, x_m, y_m, radii, speed_kph, step_m,
        elevation_amplitude_m, sample_by_time,
        lat_g_max=speed_kwargs.get("lat_g_max", 4.5),
    )


def fourier_circuit(
    coefficients: list[tuple[int, float, float]],
    *,
    base_radius_m: float = 700.0,
    n_points: int = 6000,
) -> tuple[np.ndarray, np.ndarray]:
    """An exactly closed track from a Fourier series, in metres.

    Every term is periodic in the parameter, so the curve returns to its start
    to machine precision.  Low harmonics with small amplitudes give a fast,
    open circuit; more and larger harmonics give a twisty one.
    """
    t = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    r = np.full_like(t, float(base_radius_m))
    for harmonic, amplitude, phase in coefficients:
        r = r + amplitude * np.cos(harmonic * t + phase)
    if (r <= 0).any():
        raise ValueError("amplitudes exceed the base radius; the curve self-intersects")
    return r * np.cos(t), r * np.sin(t)


def make_fourier_telemetry(
    coefficients: list[tuple[int, float, float]],
    *,
    base_radius_m: float = 700.0,
    step_m: float = 1.0,
    elevation_amplitude_m: float = 0.0,
    sample_by_time: bool = True,
    **speed_kwargs,
) -> pd.DataFrame:
    """FastF1-shaped telemetry for an exactly closed Fourier circuit."""
    x_m, y_m = fourier_circuit(coefficients, base_radius_m=base_radius_m)

    # Reparameterise onto an even arc-length grid.
    seg = np.hypot(np.diff(x_m, append=x_m[0]), np.diff(y_m, append=y_m[0]))
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    grid = np.arange(0.0, s[-1], step_m)
    xg, yg = np.interp(grid, s, x_m), np.interp(grid, s, y_m)

    dx, dy = np.gradient(xg, step_m), np.gradient(yg, step_m)
    ddx, ddy = np.gradient(dx, step_m), np.gradient(dy, step_m)
    denom = np.power(dx * dx + dy * dy, 1.5)
    kappa = np.where(denom > 1e-12, (dx * ddy - dy * ddx) / denom, 0.0)
    radii = np.where(np.abs(kappa) > 1e-9, 1.0 / np.abs(kappa), np.inf)

    speed_kph = simulate_speed(radii, step_m, **speed_kwargs)
    return _assemble_telemetry(
        grid, xg, yg, radii, speed_kph, step_m,
        elevation_amplitude_m, sample_by_time,
        lat_g_max=speed_kwargs.get("lat_g_max", 4.5),
    )


# --------------------------------------------------------------------------- #
# Reference circuit shapes
# --------------------------------------------------------------------------- #
#
# Each layout is written as (angle, radius) pairs so the total turning is
# visible by inspection, then converted to segments.  Straights are interleaved.
# The angles net close to 2*pi in one direction, as any real lap must.


def _from_angles(
    corners: list[tuple[float, float]], straights: list[float]
) -> list[Segment]:
    """Interleave straights and arcs given ``(angle_rad, radius_m)`` corners."""
    segments: list[Segment] = []
    for i, (angle, radius) in enumerate(corners):
        if i < len(straights):
            segments.append(Segment(straights[i], None))
        signed_radius = radius if angle > 0 else -radius
        segments.append(Segment(abs(angle) * radius, signed_radius))
    segments.extend(Segment(length, None) for length in straights[len(corners):])
    return close_track(segments)


def monza_like() -> list[Segment]:
    """Long straights, few corners, big radii: a power circuit.

    Roughly 6 km with about three quarters of the lap flat out, which is the
    shape of Monza's problem for an engine.
    """
    corners = [
        (-1.2, 40),    # tight opening chicane
        (1.1, 55),
        (-1.5, 190),   # long sweeper
        (-1.3, 210),
        (-0.9, 60),
        (1.0, 70),
        (-1.4, 240),
        (-1.4, 320),   # long final right-hander
    ]
    straights = [1100, 60, 700, 250, 900, 80, 650, 760]
    return _from_angles(corners, straights)


def monaco_like() -> list[Segment]:
    """Short, tight and relentless: a driver circuit.

    Roughly 2.5 km, fifteen corners, several under a 30 m radius, almost no
    straight worth the name.
    """
    corners = [
        (-1.6, 30), (0.9, 80), (-1.4, 22), (0.7, 60), (-1.8, 15),
        (1.0, 70), (-1.2, 45), (0.6, 100), (-1.5, 25), (0.8, 55),
        (-1.3, 35), (0.5, 90), (-1.7, 18), (0.6, 65), (-1.1, 40),
    ]
    straights = [180, 120, 90, 140, 70, 110, 95, 200, 80, 130, 85, 150, 60, 100, 90]
    return _from_angles(corners, straights)


def oval(radius_m: float = 500.0) -> list[Segment]:
    """A perfect circle: curvature has a closed-form answer of ``1/radius``."""
    return [Segment(2 * np.pi * radius_m, radius_m)]


#: A fast, open Fourier layout — few harmonics, gentle radii.
FAST_FOURIER_COEFFS: list[tuple[int, float, float]] = [(2, 180.0, 0.0), (3, 90.0, 1.1)]

#: A twisty Fourier layout — more harmonics, larger amplitudes, tighter radii.
TWISTY_FOURIER_COEFFS: list[tuple[int, float, float]] = [
    (3, 150.0, 0.0), (5, 110.0, 0.7), (7, 70.0, 2.0), (11, 40.0, 0.3),
]


# --------------------------------------------------------------------------- #
# Race results
# --------------------------------------------------------------------------- #

#: (status, classified-position code).  A None code means the driver was
#: classified and the code is filled in with their finishing position.
_FINISH_STATUSES = [("Finished", None), ("+ 1 Lap", None)]
_DNF_STATUSES = [
    ("Engine", "R"), ("Gearbox", "R"), ("Hydraulics", "R"), ("Power Unit", "R"),
    ("Brakes", "R"), ("Collision", "R"), ("Accident", "R"), ("Spun off", "R"),
]
#: Which of the above are mechanical, so tests can assert on cause splits.
_MECHANICAL_STATUSES = {"Engine", "Gearbox", "Hydraulics", "Power Unit", "Brakes"}


def make_race_results(
    *,
    seasons: tuple[int, ...] = (2022, 2023),
    races_per_season: int = 10,
    n_drivers: int = 12,
    n_circuits: int = 5,
    base_dnf_rate: float = 0.16,
    seed: int = 0,
) -> pd.DataFrame:
    """A FastF1-shaped race-results frame spanning several seasons.

    Each driver and each team carries a latent reliability multiplier, and each
    circuit carries its own attrition multiplier, so a model fitted to this
    data should be able to recover something.  The returned frame also carries
    the multipliers themselves (``_true_*`` columns) so tests can check that
    the rolling features track the truth rather than merely being non-null.
    """
    rng = np.random.default_rng(seed)
    n_teams = max(1, n_drivers // 2)
    drivers = [f"driver_{i:02d}" for i in range(n_drivers)]
    teams = [f"team_{i // 2:02d}" for i in range(n_drivers)]

    driver_risk = rng.uniform(0.4, 1.8, size=n_drivers)
    team_risk = rng.uniform(0.4, 1.8, size=n_teams)
    circuit_risk = rng.uniform(0.6, 1.5, size=n_circuits)
    circuit_keys = [100 + i for i in range(n_circuits)]

    rows = []
    for year in seasons:
        for rnd in range(1, races_per_season + 1):
            date = pd.Timestamp(f"{year}-03-01") + pd.Timedelta(days=14 * rnd)
            ci = (rnd - 1) % n_circuits
            grid = rng.permutation(n_drivers) + 1
            for i, (driver, team) in enumerate(zip(drivers, teams)):
                risk = (
                    base_dnf_rate
                    * driver_risk[i]
                    * team_risk[i // 2]
                    * circuit_risk[ci]
                )
                retired = rng.random() < min(risk, 0.85)
                if retired:
                    status, code = _DNF_STATUSES[rng.integers(len(_DNF_STATUSES))]
                    position, points = np.nan, 0.0
                    classified = code
                    laps = float(rng.integers(1, 50))
                else:
                    status, _ = _FINISH_STATUSES[rng.integers(len(_FINISH_STATUSES))]
                    position = float(i + 1)
                    points = float(max(0.0, 25.0 - 2.5 * i))
                    classified = str(int(position))
                    laps = 57.0
                rows.append(
                    {
                        "Year": year,
                        "RoundNumber": rnd,
                        "RaceDate": date,
                        "EventName": f"Circuit {circuit_keys[ci]} Grand Prix",
                        "circuit_key": circuit_keys[ci],
                        "DriverId": driver,
                        "Abbreviation": f"D{i:02d}",
                        "TeamId": team,
                        "GridPosition": float(grid[i]),
                        "Position": position,
                        "ClassifiedPosition": classified,
                        "Status": status,
                        "Points": points,
                        "Laps": laps,
                        "_true_driver_risk": driver_risk[i],
                        "_true_team_risk": team_risk[i // 2],
                        "_true_circuit_risk": circuit_risk[ci],
                    }
                )
    return pd.DataFrame(rows)
