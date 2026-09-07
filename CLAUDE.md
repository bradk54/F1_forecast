# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Forecasting Formula 1 race outcomes. There are **two parallel codebases**, and knowing which one a task belongs to is the first thing to work out:

1. **`src/` — the retirement (DNF) prediction pipeline.** A tested Python package, ~4,800 lines: ingest, labelling, feature engineering, and model evaluation. This is where pipeline work belongs.
2. **`Notebooks/` — the points/finishing-position modelling.** Notebook-centric exploratory work on a separate track, with its own feature harness (see below). It does not import `src/`.

They share the `Data/raw` fastf1 cache and nothing else.

## Environment & Commands

- **`.venv/` at the repo root is the environment for `src/` and the tests** (Python 3.12). Use `./.venv/bin/python` — not the pyenv shim.
- Notebooks resolve `jupyter` through pyenv shims (Python 3.11) and load shared imports with `%run ../packages.py`.
- `requirements.txt` is populated and covers both tracks: fastf1, pandas/numpy/scipy, pyarrow, scikit-learn, xgboost, matplotlib/seaborn, beautifulsoup4, Unidecode, tqdm, pytest.

```bash
./.venv/bin/python -m pytest -q                    # full suite
./.venv/bin/python -m src.data.generate_dataset --seasons 2018-2025
./.venv/bin/python -m scripts.write_data_dictionary # regenerate References/data_dictionary.md
```

Run notebooks with `jupyter lab` / `jupyter notebook` from the repo root.

## The Notebook Track (`Notebooks/`)

Directory names are capitalized (`Data/`, `Notebooks/`, `Models/`), unlike the lowercase layout sketched in README.md. All data files (`*.csv`, `*.pkl`, `*.json`, `*.parquet`) are gitignored — data exists only on the local machine and cannot be inspected from git history. `Data/raw` is large (tens of GB: the fastf1 cache plus a multi-GB `fastf1_http_cache.sqlite` holding Ergast responses).

Flow, in notebook number order:

1. **`Notebooks/1_0_web_scrape.ipynb`** — scrapes pitwall.app season and race pages (2000–2025) with BeautifulSoup and saves pickles to `Data/raw/<year>/`.
2. **`Data/raw/` doubles as the fastf1 cache directory** (`fastf1.Cache.enable_cache(RAW_PATH)`), so `Data/raw/<year>/` contains both pitwall pickles and fastf1 session caches (`<date>_<event>/<session>/*.ff1pkl`). Don't delete or restructure it casually.
3. **`Data/processed/<year>/`** — cleaned per-season CSVs: `results.csv`, `qualifying.csv`, `finishes.csv`.
4. **`Notebooks/2_0_Model_Development.ipynb`** — the main modeling notebook. Loads multi-year race results from the fastf1 cache (~2023–2025), builds rolling-window features (points/finish form for driver, team, and driver–team pairing), and trains sklearn regressors plus a Bayesian Poisson points model (PyMC/arviz).

`Notebooks/1_1_EDA_2024_race_results.ipynb` is exploratory analysis of the 2024 season.

### Feature Iteration Infrastructure (in 2_0_Model_Development.ipynb)

The tail of the notebook has a self-contained harness for feature experiments. The intended workflow:

1. Run the **Multi-Year Data Loader** cell to build `multi_raw`/`multi_df` from the fastf1 cache (fast when cached).
2. Toggle features in the **Feature Registry** cell (`active: True/False`) — no other code changes needed.
3. Run **Walk-Forward Eval** (train on seasons < test year, evaluate on test year) for Spearman / MAE / top-3 accuracy across 2022–2024.
4. Run **Feature Ablation** to measure the Spearman drop from removing each feature.
5. Run **Experiment Log** to append scores to a CSV so results persist between sessions.

When adding model features, extend the Feature Registry rather than writing one-off feature code elsewhere in the notebook.

## The DNF Pipeline (`src/`)

Predicts whether a driver retires from a given race. One row per driver per race, 2018 onward — 2018 is the floor because that is where fastf1 car and position telemetry begins.

```
src/config.py                 paths; every directory overridable by env var (F1_DATA_DIR, F1_FASTF1_CACHE, ...)
src/data/ingest.py            the ONLY module that touches the network
src/data/generate_dataset.py  CLI orchestrator; runs the checks that gate a write
src/data/circuits.py          static street/night/altitude reference
src/data/f1_loader.py         standalone self-auditing season loader
src/features/labels.py        Status + ClassifiedPosition -> dnf and the cause taxonomy
src/features/build_features.py  leakage-safe rolling history (driver/team/pairing/circuit)
src/features/track_profile.py   circuit geometry from a reference lap's telemetry
src/features/registry.py      single source of truth for every feature and when it is knowable
src/models/train.py           walk-forward evaluation, ablation, permutation importance
```

Build it with `python -m src.data.generate_dataset --seasons 2018-2025`. Intermediates cache to `Data/processed/{race_results,circuit_profiles}.parquet`; the modelling table lands at `Data/processed/dnf_dataset.parquet`. `--skip-download` reuses those parquets; `--offline` rebuilds from the fastf1 cache without network.

### Rules that hold across this pipeline

- **The registry is authoritative.** Every feature is declared in `src/features/registry.py` with a `Stage` saying when it becomes knowable. Add features there, not ad hoc — `audit_coverage` fails the build for anything in the dataset but not registered.
- **Leakage is checked, not assumed.** `detect_target_leakage` runs before every write and `main()` returns exit 3 if it fires. Rolling features must be strictly prior-race; the helpers in `build_features.py` (`prior_rolling`, `prior_expanding`) exist for this.
- **A load can succeed and still be useless.** `Status` comes from the Ergast backend, not F1 timing, and `Session.load` swallows an Ergast failure — a rate-limited round returns a full grid with a blank `Status`, which the labeller would read as an all-retirement race. `extract_results` rejects that (`DegradedResultsError`); `status_coverage` is the dataset-level backstop and `main()` returns exit 4 rather than writing. Prefer `--offline` when the cache is warm; it is faster and cannot provoke a 429.
- **Sprints inform history but are not modelling rows.** A 100 km sprint and a 305 km grand prix do not share an attrition process.
- **Exit codes:** 1 no data / unreachable, 2 missing cached intermediates, 3 leakage, 4 a race with no finishing status.

## Conventions

- Notebook naming: `<major>_<minor>_description.ipynb` (e.g. `1_0_web_scrape.ipynb`), numbered in pipeline order.
- Notebooks hardcode absolute paths in `RAW_PATH` / `PROCESSED_PATH` constants near the top; update those constants rather than scattering new paths.
- The pitwall.app scraper sleeps 5s between requests — keep rate limiting when extending it.
- In `src/`, paths come from `src/config.py`; never hardcode one. Every directory is overridable by environment variable, which is how tests and alternate cache locations work.
- `src/` code is documented densely and explains *why*, not what — match that when editing. Docstrings carry the domain reasoning (why a DSQ is not a DNF, why profiles are keyed on `(circuit_key, year)`); keep that reasoning with the code.
- Tests do not touch the network. `tests/synthetic.py` generates fastf1-shaped frames with known latent structure, so features can be checked against the truth that generated them.
