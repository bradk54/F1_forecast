# Dashboard

A local Streamlit app over the finishing-order and championship forecast in
`src/` -- and the retirement (DNF) model that is its first stage.

**Double-click `Dashboard.command`** at the repository root. It starts the
server and opens the browser; that Terminal window *is* the server, so closing
it stops the app. Clicking it again while it is already running just reopens the
browser rather than failing on the port.

Or from a terminal, either of:

```bash
./Dashboard.command
./.venv/bin/python -m streamlit run app/Home.py
```

Then open <http://localhost:8501>.

(`.claude/launch.json` is caught by the repository's blanket `*.json` ignore, so
a `dashboard` launch entry there stays local to your machine.)

## What it is, and what it is not

Every number comes from the same function the CLI calls — `forecast.championship`,
`forecast.backtest_race`, `walk_forward_races`, `build_inference_rows`,
`monitor.drift_check`. The app holds caching, error handling and charts, and
nothing else. If a chart and `python -m src.models.forecast` or
`python -m src.models.predict` ever disagree, the app has a bug.

The pages follow the pipeline the forecast runs: attrition, then the finishing
order, then points, then a season. The DNF model is presented as the first
stage, not as a product of its own.

**It reads the pipeline; it does not run it.** The one action that writes to
disk is **Refit attrition model** in the sidebar, which rewrites
`Models/dnf_model.joblib` and its manifest. Nothing in the app fetches new
races — rebuild the dataset from a terminal:

```bash
./.venv/bin/python -m src.data.generate_dataset --offline
```

One place reaches the network, behind an explicit click, and writes nothing:
fetching a qualifying grid on the **Race** page. The next race comes from the
forecast's own calendar, read from the FastF1 schedule offline.

## Pages

| Page | Answers |
| --- | --- |
| **Home** | Who wins the titles, what happens next Sunday, and how far to trust it? |
| **Championship** | Both championships simulated: points bands, title odds, pace drift. |
| **Race** | The next race or any completed one: P(win/podium/points/out) and expected points, then the attrition stage's retirement ranking beneath. |
| **Model** | The pipeline, each stage's features and what selection measured them to be worth, the current coefficients and position scales, and a per-driver breakdown of one race. |
| **Track record** | Held-out finishing-order scores and season calibration, then the attrition log (`Reports/model_log.csv`). |
| **Lab** | Walk-forward, ablation and importance for the attrition model, run live. |
| **Data** | Where attrition lives, feature distributions, circuit profiles, the registry. |

### The sidebar model card

Every page carries the same card: the saved attrition model and whether it is
fresh, both finishing-order models with their window and scales, and the season
noise. The order-model lines are read from `STAGE_MODELS` and
`DEFAULT_SEASON_NOISE` in `src/models/forecast.py`, so the card cannot describe
a model the code no longer runs. The order models refit from the parquets on
every request, so only the attrition model has a staleness to report.

### Reports the pages read but do not build

Two sections fill in only once their report exists, because each is minutes of
simulation and belongs in a terminal:

```bash
./.venv/bin/python -m src.models.forecast evaluate --write   # Track record: held-out scores
./.venv/bin/python -m src.models.forecast backtest --write   # season calibration
```

### Testing

`tests/test_app.py` renders every page on synthetic data with Streamlit's
`AppTest`. To click through the app on that data yourself:

```bash
./.venv/bin/python -m tests.app_fixture /tmp/f1_fake
F1_PROCESSED_DIR=/tmp/f1_fake F1_MODELS_DIR=/tmp/f1_fake F1_REPORTS_DIR=/tmp/f1_fake \
    ./.venv/bin/python -m streamlit run app/Home.py
```

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

**Run a `post_quali` retirement model without a grid.** `grid_position` is the
retirement model's strongest feature and a missing slot is imputed to the back of the field, so every driver
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
