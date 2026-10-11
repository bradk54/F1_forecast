"""Synthetic parquets shaped like a live season, for exercising the dashboard.

The app reads ``dnf_dataset.parquet`` and ``race_results.parquet`` and nothing
else it cannot do without.  This writes both from
:func:`tests.synthetic.make_order_results`, cut part-way through the last
season: the dataset stops at ``through_round`` while the raw results run to
the end of the year, which is how a completed season's calendar is read
offline -- so the championship and next-race views work with no network and
no FastF1 schedule.

Also runnable, to point a real ``streamlit run`` at fake data::

    ./.venv/bin/python -m tests.app_fixture /tmp/f1_fake
    F1_PROCESSED_DIR=/tmp/f1_fake F1_MODELS_DIR=/tmp/f1_fake \\
        F1_REPORTS_DIR=/tmp/f1_fake ./.venv/bin/python -m streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from tests.synthetic import make_order_results

SEASONS = (2023, 2024, 2025, 2026)
THROUGH_ROUND = 8


def write_app_data(directory: Path, *, through_round: int = THROUGH_ROUND) -> tuple[Path, Path]:
    """Write ``(dnf_dataset.parquet, race_results.parquet)`` into ``directory``."""
    from src.data.generate_dataset import build_dataset

    directory.mkdir(parents=True, exist_ok=True)
    results = make_order_results(seasons=SEASONS, races_per_season=12, seed=7)
    last = SEASONS[-1]
    seen = (results["Year"] < last) | (results["RoundNumber"] <= through_round)
    dataset, _ = build_dataset(results.loc[seen], None, circuit_reference=False,
                               run_checks=False)
    dataset_path = directory / "dnf_dataset.parquet"
    results_path = directory / "race_results.parquet"
    dataset.to_parquet(dataset_path)
    results.to_parquet(results_path)
    return dataset_path, results_path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "fake_data")
    for path in write_app_data(out):
        print(f"wrote {path} ({len(pd.read_parquet(path))} rows)")
