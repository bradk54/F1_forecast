"""What the sport paid, season by season.

The points map changed twice inside this dataset's window, so a single
hard-coded table is wrong at both ends of it:

* **Fastest-lap bonus.**  One point to the driver who set the fastest lap,
  provided they finished in the top ten -- awarded 2019 through 2024 and
  withdrawn from 2025.  Confirmed against this repository's own results rather
  than taken on trust: the ``Points`` column exceeds the position map by one
  point per race in exactly those six seasons (19-20 a season) and by nothing
  in 2018, 2025 or 2026.
* **Sprints.**  None before 2021; the top three scored 3-2-1 in 2021; the top
  eight have scored 8 down to 1 since 2022.

Not modelled: half points for a race stopped early (Spa 2021 paid 12.5 to the
winner), which cannot be forecast and happened once.
"""

from __future__ import annotations

import numpy as np

GRAND_PRIX_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)

#: Seasons in which the fastest lap earned a bonus point for a top-ten finisher.
FASTEST_LAP_SEASONS = range(2019, 2025)


def grand_prix_points(season: int, n_cars: int = 30) -> np.ndarray:
    """Points indexed by finishing position; index 0 is a retirement.

    The array is long enough to index any classified position -- which scores
    nothing beyond tenth -- and never shorter than the table itself, so a small
    field cannot truncate it.
    """
    table = np.zeros(max(n_cars, len(GRAND_PRIX_POINTS)) + 1)
    table[1:len(GRAND_PRIX_POINTS) + 1] = GRAND_PRIX_POINTS
    return table


def sprint_points(season: int, n_cars: int = 30) -> np.ndarray:
    """Sprint points indexed by finishing position; index 0 is a retirement."""
    paid = () if season < 2021 else (3, 2, 1) if season == 2021 else (8, 7, 6, 5, 4, 3, 2, 1)
    table = np.zeros(max(n_cars, len(paid)) + 1)
    table[1:len(paid) + 1] = paid
    return table


def has_fastest_lap_point(season: int) -> bool:
    """Whether a top-ten finisher could earn one more point for the fastest lap."""
    return season in FASTEST_LAP_SEASONS
