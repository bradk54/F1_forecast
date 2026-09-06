# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Forecasting Formula 1 race outcomes. This is a notebook-centric data science project — all real work lives in `Notebooks/`. The `src/` scripts (`generate_dataset.py`, `build_features.py`, `train.py`, `predict.py`) are empty placeholders from the project template; do not assume they contain logic.

## Environment & Commands

- Python 3.11 via pyenv (`jupyter` resolves through pyenv shims). There is no build, lint, or test setup.
- `requirements.txt` is empty. Common imports are centralized in `packages.py` at the repo root, which notebooks load with `%run ../packages.py` (requests, pandas, numpy, BeautifulSoup, matplotlib, seaborn, pickle, glob, json). Notebooks additionally import `fastf1`, `scikit-learn`, `pymc`/`arviz`, `tqdm`, and `unidecode` directly.
- Run notebooks with `jupyter lab` / `jupyter notebook` from the repo root.

## Data Pipeline & Architecture

Directory names are capitalized (`Data/`, `Notebooks/`, `Models/`), unlike the lowercase layout sketched in README.md. All data files (`*.csv`, `*.pkl`, `*.json`, etc.) are gitignored — data exists only on the local machine and cannot be inspected from git history.

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

## Conventions

- Notebook naming: `<major>_<minor>_description.ipynb` (e.g. `1_0_web_scrape.ipynb`), numbered in pipeline order.
- Notebooks hardcode absolute paths in `RAW_PATH` / `PROCESSED_PATH` constants near the top; update those constants rather than scattering new paths.
- The pitwall.app scraper sleeps 5s between requests — keep rate limiting when extending it.
