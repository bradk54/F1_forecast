# Dashboard

A local Streamlit app over the retirement (DNF) pipeline in `src/`.

```bash
./.venv/bin/python -m streamlit run app/Home.py
```

Then open <http://localhost:8501>.

(`.claude/launch.json` is caught by the repository's blanket `*.json` ignore, so
a `dashboard` launch entry there stays local to your machine.)

## What it is, and what it is not

Every number comes from the same function the CLI calls — `walk_forward_races`,
`score_predictions`, `build_inference_rows`, `monitor.drift_check`. The app
holds caching, error handling and charts, and nothing else. If a chart and
`python -m src.models.predict` ever disagree, the app has a bug.

**It reads the pipeline; it does not run it.** The one action that writes to
disk is **Refit and save** in the sidebar, which rewrites
`Models/dnf_model.joblib` and its manifest. Nothing in the app fetches new
races — rebuild the dataset from a terminal:

```bash
./.venv/bin/python -m src.data.generate_dataset --offline
```

Two places reach the network, both behind an explicit click, and neither writes
anything: looking up the next race on the calendar, and fetching a qualifying
grid.

## Pages

| Page | Answers |
| --- | --- |
| **Home** | Is the model current, is it calibrated, has the sport moved? |
| **Race weekend** | Rank a grid by retirement risk; score it once the race has run. |
| **Model health** | Read `Reports/model_log.csv` — skill over time, drift, what the grid buys. |
| **Lab** | Walk-forward, ablation, permutation importance, run live and compared. |
| **Data explorer** | Where attrition lives, feature distributions, circuit profiles, the registry. |

## Filling the performance log

`Reports/model_log.csv` fills one race at a time as `predict refresh` runs after
each weekend. To plot it before that history exists, reconstruct it:

```bash
./.venv/bin/python -m scripts.backfill_model_log --stage post_quali
./.venv/bin/python -m scripts.backfill_model_log --stage pre_weekend
```

A replay refits before every race on strictly prior data, so its rows are honest
out-of-sample scores — but it reads the dataset as it stands today rather than
what was known that week. Those rows are written `source=replay`, they never
overwrite a `source=live` row, and every page that mixes the two says so.

## Two things the pages will not let you do quietly

**Run `post_quali` without a grid.** `grid_position` is the model's strongest
feature and a missing slot is imputed to the back of the field, so every driver
comes out looking like a back-marker and every probability is inflated. The page
refuses and points at `pre_weekend`, which is a model fitted without grid
position at all.

**Read an in-sample backtest as a score.** If the saved model was trained
through the race you picked, the page says so before showing the numbers.

## Conventions

Charts live in `app/charts.py` and share one theme. Colour carries identity,
never a magnitude a bar length already carries; categorical hues are assigned in
fixed order and never cycled; there is never a second y-axis. Add a form there
rather than building a one-off figure in a page.
