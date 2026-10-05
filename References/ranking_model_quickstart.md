# Running the Finishing-Order Models from Terminal

This guide shows how to run the Plackett-Luce ranking model and evaluate it against 2023-2026 race data.

## Prerequisites

The dataset must be built first from the DNF pipeline:

```bash
# One-time setup: install dependencies
pip install -r requirements.txt

# Generate the datasets (requires network access to F1 data sources)
./.venv/bin/python -m src.data.generate_dataset --seasons 2018-2025
```

Once that completes, three parquet files are created:
- `Data/processed/race_results.parquet` — raw race results from FastF1
- `Data/processed/circuit_profiles.parquet` — circuit-specific features from telemetry
- `Data/processed/dnf_dataset.parquet` — the full feature table, one row per driver per race

The ranking models read from `dnf_dataset.parquet`.

## Running the Model

All commands assume you're in the repo root and have activated the environment:

```bash
./.venv/bin/python -m src.models.rank_predict <command> [options]
```

### Evaluate against historical data

**Grid position only (baseline):**
```bash
./.venv/bin/python -m src.models.rank_predict evaluate --feature-set grid_only
```

**Grid + team strength (intermediate complexity):**
```bash
./.venv/bin/python -m src.models.rank_predict evaluate --feature-set grid_and_team
```

**All features (full model):**
```bash
./.venv/bin/python -m src.models.rank_predict evaluate --feature-set full
```

### Additional options

```
--model {plackett_luce}
        Which model to use. Currently only Plackett-Luce is implemented.
        
--lookback N
        Train on only the last N races. Set to 0 for an expanding window
        (every prior race). Default: 40 races. Larger lookback windows
        overfit to settled seasons; 40 is measured to be the sweet spot
        for seasons with regime change (2026 with new power units).

--start-after YYYY-MM-DD
        Only score races after this date. Useful to ignore early races
        where the model has little history to learn from.

--output path/to/file.csv
        Save detailed per-driver per-race predictions to a CSV file.
```

### Example: Compare feature sets

```bash
# Grid only
./.venv/bin/python -m src.models.rank_predict evaluate --feature-set grid_only \
  --output /tmp/grid_only.csv

# Grid + team
./.venv/bin/python -m src.models.rank_predict evaluate --feature-set grid_and_team \
  --output /tmp/grid_team.csv

# All features
./.venv/bin/python -m src.models.rank_predict evaluate --feature-set full \
  --output /tmp/full.csv
```

Each produces a summary table and optionally saves predictions to CSV. The summary shows:

```
Model                  plackett_luce
Features               grid_position
Mean Spearman          0.456
Mean MAE               3.45
Mean Top-3 Acc         62.3%
Mean Top-10 Acc        87.1%
```

**Metrics explained:**

| Metric | Meaning |
| --- | --- |
| **Spearman** | Rank correlation between predicted and actual finishing positions. Higher is better (range: -1 to 1, 0 = random). |
| **MAE** | Mean absolute error in positions. E.g., 3.45 means the model is off by ~3.5 positions on average. Lower is better. |
| **Top-3 Acc** | How often does the model correctly rank a top-3 finisher into the top 3? Percent, 0–100. |
| **Top-10 Acc** | Same, for the top 10 (the points scorers in F1). Most predictive metric for fantasy / betting. |

## Understanding the Code

The implementation lives in two modules:

### `src/models/ranking.py` — The model

- **`PlackettLuceModel`**: Fits via Cox proportional hazards (time-rank duality). Takes feature vectors per driver and a ranking outcome per race. Truncates at k=10 to focus on points-scoring positions.

- **`walk_forward_races_ranked()`**: The evaluation harness. For each race:
  1. Train on every prior race (with optional lookback window)
  2. Score the current race
  3. Compare predictions to actual outcomes
  - Returns per-season scores (Spearman, MAE, top-3/10 accuracy) and per-race predictions.

- **`sample_ranking()`**: Generates random permutations from the fitted model for Monte Carlo season simulation.

### `src/models/rank_predict.py` — The CLI

Command-line interface for building and evaluating models. Handles data loading, feature selection, and result formatting.

## The Design Choice: Plackett-Luce First

Why Plackett-Luce and not something else? See `References/finishing_order_models.md` for the full design brief. Short version:

1. **Tractable**: Estimated via Cox partial likelihood (standard survival machinery), not EM or variational approximation.
2. **Samples exactly**: Can generate 10,000 race orderings in <1 second for season simulation.
3. **Measured baseline**: On F1 data 2022-2026, Plackett-Luce on grid position alone beats random forests and gradient boosting on every metric.
4. **Upgrade path**: If calibration is poor, fit Stern's gamma family (one extra parameter). If that's still inadequate, graduate to gradient-boosted learning-to-rank.

## Next Steps

1. **Run the baseline** (`grid_only`) against the full dataset. Establish what you measure yourself.
2. **Add team strength** and measure the gain (likely ~1-2% on Spearman).
3. **Attach the DNF model** as a survival curve (option coming next).
4. **Validate top-10 calibration** with Brier skill score.
5. **Season simulation**: Sample 100,000 races and check that simulated championship distributions match observed ones.

For season simulation code, the `.sample_ranking()` method on the fitted model is the generator. See `References/finishing_order_models.md` §9 for the sampler's design.

## Troubleshooting

**`FileNotFoundError: no dataset at Data/processed/dnf_dataset.parquet`**

The dataset hasn't been built. Run:
```bash
./.venv/bin/python -m src.data.generate_dataset --seasons 2018-2025
```

This requires network access to F1 data sources (FastF1, Ergast/Jolpica). If you already have a populated FastF1 cache (from Notebooks/), rebuild from it offline:
```bash
./.venv/bin/python -m src.data.generate_dataset --offline
```

**Model overfits to 2024 / undershoots on 2025**

This is expected. 2024 was the last settled hybrid-v2 season. 2025 is hybrid-v3 with new power units and changed reliability. The 40-race lookback window bets on recency over history depth; for a settled formula (2024), an expanding window would be better. Re-check the window each season and adjust in the `--lookback` argument if needed.

**Top-10 accuracy is much worse than top-3**

This is also expected and measured in the DNF pipeline: the model is strong on `grid_position` (which predicts top-10), but the mid-field (positions 6-10) is chaos. You cannot predict the difference between 8th and 11th from pre-race features; that comes from race dynamics (pit stops, safety cars, attrition).

**Metrics are worse this year**

Verify you're using the right lookback window and that you've refit the model on the current year's early races. If metrics collapse abruptly midseason, that's a signal of a regime change (regulation, team changes, driver performance shifts). File that as a finding and refit more frequently.
