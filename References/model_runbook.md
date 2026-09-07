# DNF model runbook

How to run the retirement model week to week: what the commands do, what the
model assumes, and what to do when something breaks.

**Run everything with `./.venv/bin/python` from the repository root.** Not
`python`, not a conda environment. The CLI needs `fastf1`, which only the venv
has, and the saved model is pickled by the venv's scikit-learn — running it
elsewhere gets you `ModuleNotFoundError: No module named 'fastf1'` in one env
and a wall of `InconsistentVersionWarning` in another.

The notebooks are the exception: they run under the `data_science` conda
environment and read the parquet files rather than importing `src/`.

---

## The weekly loop

Two commands per race weekend.

| When | Command | What it does |
| --- | --- | --- |
| **Tuesday**, after the last race is classified | `predict refresh` | Pulls new results, scores last weekend's prediction into the log, refits, saves |
| **Saturday**, after qualifying | `predict next` | Ranks this weekend's grid by retirement risk |

```bash
# Tuesday
./.venv/bin/python -m src.models.predict refresh

# Saturday, once the grid is set
./.venv/bin/python -m src.models.predict next
```

That is the whole cadence. `refresh` scores **before** it refits, which is what
keeps the log honest — a model scored against data it has just trained on is
not measuring anything.

If you want a number before qualifying, see
[Before qualifying vs after](#before-qualifying-vs-after) — it is a different
model, not the same one with less confidence.

---

## Commands

### `refresh` — the weekly retrain

```bash
./.venv/bin/python -m src.models.predict refresh
./.venv/bin/python -m src.models.predict refresh --offline      # warm cache, no network
./.venv/bin/python -m src.models.predict refresh --no-download   # dataset as-is
./.venv/bin/python -m src.models.predict refresh --lookback 60   # wider training window
./.venv/bin/python -m src.models.predict refresh --lookback 0    # all history
```

In order: rebuild the dataset (network) → score the outstanding prediction into
`Reports/model_log.csv` → fit on the last 40 races → write
`Models/dnf_model.joblib` and `Models/dnf_model.json` → report drift.

Takes a few minutes, almost all of it the data pull. The fit itself is about a
quarter of a second.

### `next` — rank the upcoming race

```bash
./.venv/bin/python -m src.models.predict next
./.venv/bin/python -m src.models.predict next --stage pre_weekend
./.venv/bin/python -m src.models.predict next --no-save
```

Finds the first scheduled event after the last one in the dataset, builds
feature rows for it, and prints drivers from riskiest to safest. The prediction
is saved to `Models/pending_prediction.json` so the next `refresh` can score it
— that is what fills the log. `--no-save` skips that, for a look that should
not enter the record.

### `race` — a specific event, or a backtest

```bash
./.venv/bin/python -m src.models.predict race 2026 14
```

Same as `next` for a named round. If that race has already run, its own rows
are removed before features are built (otherwise the race would inform its own
prediction), and you get a warning. If the model was **trained through** that
race, the warning says so explicitly: those numbers are in-sample and flatter
the model, sometimes dramatically.

### `status` — how it has been doing

```bash
./.venv/bin/python -m src.models.predict status
./.venv/bin/python -m src.models.predict status --window 20
```

Prints the recent rows of the log, a rolling summary, and which model is
currently saved.

---

## Before qualifying vs after

**Yes, it changes, and by a lot.** This is the single most important thing to
get right about running the model.

`grid_position` and `grid_position_pct` are the two strongest features in the
model — above every reliability feature — so a model that has the grid and one
that does not are genuinely different models, not the same prediction at
different confidence.

| | `--stage pre_weekend` | `--stage post_quali` (default) |
| --- | --- | --- |
| When | Any time before qualifying | After qualifying |
| Features | 95 | 101 |
| Uses grid position | No | Yes |
| Uses team-mate grid delta | No | Yes |
| Typical spread | ~0.13–0.19 | ~0.06–0.40 |

The post-quali model separates the field far more, because it knows who is
starting at the back — and starting at the back is where contact happens.

### The trap

If you run the **default** (`post_quali`) **before qualifying has run**, the
command refuses:

```
error: qualifying for 2026 R14 has not run, or could not be loaded, so grid
position is unknown.
```

This refusal is deliberate. A missing grid does not stay missing:
`add_race_context` reads a non-positive grid slot as a pit-lane start and moves
it to the back of the field. Every driver would be scored as a back-marker, and
every probability would come out inflated — roughly 0.20–0.46 against a 14%
base rate. That looks like a model with a strong opinion, when in fact it has
no information at all.

**Before qualifying, use `--stage pre_weekend`.** It is a model fitted without
grid position, so it is honest about what it does not know. `--allow-missing-grid`
exists if you want the post-quali model anyway, but read the paragraph above
first.

> Note: `refresh` fits **one** model, at whichever `--stage` you give it. If you
> want both, run `refresh --stage pre_weekend` early in the week and
> `refresh --stage post_quali --no-download` on Saturday.

---

## What gets saved where

```
Models/dnf_model.joblib          the fitted pipeline, ~2 MB          (gitignored)
Models/dnf_model.json            the manifest                        (gitignored)
Models/pending_prediction.json   last prediction, awaiting its race  (gitignored)
Reports/model_log.csv            one row per scored race             (COMMITTED)
Reports/predictions.csv          one row per driver per prediction   (COMMITTED)
```

**Nothing in `Models/` arrives with a clone.** It is gitignored and rebuilt in a
quarter of a second, so on a fresh checkout run `refresh --no-download` before
anything expects a model to exist. The two files in `Reports/` are the durable
record and do travel with the repository.

Only the **current** model is kept. Two megabytes per race across a season is
forty-odd files nobody opens, and the log already records what each one scored.
Both are rebuilt in a quarter of a second, so they are artefacts rather than
source — which is why they are gitignored and the log is not.

`Reports/model_log.csv` is committed on purpose. It is a few kilobytes, and the
entire point is to read drift out of the diff.

### The manifest

`cat Models/dnf_model.json` answers "what is deployed" without unpickling
anything:

```json
{ "model": "random_forest", "stage": "post_quali", "lookback_races": 40,
  "train_rows": 815, "train_base_rate": 0.1436,
  "trained_through_year": 2026, "trained_through_round": 13,
  "trained_through_event": "Italian Grand Prix",
  "n_features": 101, "sklearn_version": "1.5.1", "git_sha": "b26d800" }
```

It exists so three failures are refused rather than tolerated: a **stale** model
(dataset moved on, model did not), **schema drift** (registry gained a feature),
and **environment drift** (a different scikit-learn pickled it).

---

## The two logs

`Reports/model_log.csv` says **how a weekend went** — one row per race, the
aggregate scores. `Reports/predictions.csv` says **what was claimed about each
car** — one row per driver per prediction, written before the race with `dnf`
and `scored_at` left blank, and filled in by the next `refresh`.

The second exists because the first cannot answer "what did we say about this
driver before Barcelona". It is also the honest record: a row is written at a
point where the answer does not exist yet, so it cannot be quietly revised once
the result is known.

Predicting the same race twice at the same stage **replaces** the earlier rows.
The two stages are kept separately, because a Thursday call and a Saturday call
are different claims, and `refresh` fills in the outcome for both.

```python
import pandas as pd
p = pd.read_csv("Reports/predictions.csv")
scored = p[p.dnf.notna()]
scored.groupby("driver_id").agg(n=("dnf", "size"), mean_pred=("predicted", "mean"),
                                actual=("dnf", "mean"))
```

### Reading the race log

One row per race. Two things will look wrong and are not:

**`roc_auc` is often blank.** It is undefined when nobody retired, which at
current rates is about a third of races. Read it as a rolling mean over five or
ten races — `status` does this for you. A single race's AUC is built from twenty
rows and is noise.

**Skill is measured against the *training* base rate, not the race's own.** The
model cannot know in advance that this particular Sunday would be a 25%
attrition race, so scoring it against 25% would credit it for something it never
predicted.

Columns worth watching:

| Column | What it tells you |
| --- | --- |
| `brier_skill` | Above zero means better than quoting the base rate. Noisy per race. |
| `top2_hits` | Of the two cars flagged, how many actually stopped. What you would act on. |
| `mean_predicted` vs `observed_rate` | A persistent gap is drift, not bad luck. |
| `train_base_rate` | What the model thought normal was when it was fitted. |

---

## Assumptions

Every one of these can be wrong in a way the model will not notice.

1. **The entry list is last race's entry list.** Taken from the most recent
   completed round. A reserve driver, a mid-season swap or a new entry will be
   wrong until that race has been run once.

2. **The circuit is the one this event used last time.** Matched on event name.
   A renamed event or a first-time venue gets no track-geometry features at all
   — the model tolerates NaN, but it is predicting with less than it thinks.

3. **The model trains on the last 40 races.** Roughly two seasons.
   [Measured on held-out 2026](#why-40-races), where it beat using all history.
   It is a tuned parameter, not a constant — see below.

4. **Features are strictly prior-race.** This is what makes predicting an
   unraced event legitimate: a placeholder row's own outcome is shifted out of
   its own features, and there is no later race for it to contaminate. It is
   also what the leakage check enforces on every dataset build.

5. **The target is `dnf`, undivided.** Cause labels (`dnf_mechanical`,
   `dnf_incident`) stop in 2023 — the backend returns a bare `Retired` from then
   on. Cause-specific modelling was tried and could not be resolved; do not
   reopen it without cause labels from another source.

6. **Sprints inform history but are never predicted.** A 100 km sprint and a
   305 km grand prix do not share an attrition process.

### Why 40 races

On held-out 2026 a bounded window beat an expanding one over all nine seasons:

| Window | AUC | Brier skill | Calibration |
| --- | --- | --- | --- |
| All history | 0.559 | +0.021 | 0.59 |
| Last 60 | 0.602 | +0.052 | 0.86 |
| **Last 40** | **0.615** | **+0.049** | 0.75 |
| Last 25 | 0.613 | +0.049 | 0.82 |

It is a **bet, not a free win**: it beats an expanding window on 2026 and loses
on 2024 and 2025, both settled seasons. 2026 changed power-unit regulations and
roughly doubled the retirement rate, so it is the season most likely to reward
recency. The bet is that a short window is the right posture *going into* a
regime change, when you cannot yet know one has begun.

**Re-check it each season** with `--lookback`. In a settled formula the best
window is probably longer.

---

## Drift

`refresh` and `next` both print a drift line:

```
drift:  last 5 races observed 16.4% vs 14.4% trained (+2.0%)
```

Above four percentage points it becomes a warning naming `lookback_races`. It
never fails the run — a regime change is information, not an error, and the
response is to re-tune the window rather than to stop predicting.

This check exists because of 2026. New power units took attrition from 10.3% to
19.7% while the model was still fitted at 14.2%, and nothing in the pipeline
said so; the predictions simply came out too low. Two races of this check would
have caught it.

---

## When things break

### `ModuleNotFoundError: No module named 'fastf1'`

You are not in the venv. Every command in this file starts with
`./.venv/bin/python`, and a bare `python` picks up whatever conda has activated.

### `error: the saved model was fitted at stage 'pre_weekend', but 'post_quali' was requested`

`refresh` fits **one** model, at whichever `--stage` it was given. Either refit
at the stage you want, or ask for the one that exists — the message names both
commands. This is not drift, and refitting "to fix drift" will not help if you
refit at the same wrong stage.

### `InconsistentVersionWarning: Trying to unpickle estimator ... from version X when using version Y`

The model was pickled by a different scikit-learn than the one reading it —
almost always because the command was run outside the venv. It is a warning,
not an error, and the estimator usually still scores correctly, but the
predictions are not guaranteed identical. Run from `./.venv/bin/python`, or
refit:

```bash
./.venv/bin/python -m src.models.predict refresh --no-download
```

### `error: no model at Models/dnf_model.joblib`

Nothing has been fitted yet.

```bash
./.venv/bin/python -m src.models.predict refresh --no-download
```

### `StaleModelError: model was trained through 2026 R13 but the dataset now reaches 2026 R14`

The dataset has new races the model has not seen. Refit:

```bash
./.venv/bin/python -m src.models.predict refresh
```

### `SchemaDriftError: the saved model's features no longer match the registry`

Someone added or removed a feature. The saved model has the old column list.
Refit — this is the intended response, not a bug:

```bash
./.venv/bin/python -m src.models.predict refresh --no-download
```

### `error: qualifying ... has not run`

Expected before Saturday. Use `--stage pre_weekend`, or `--allow-missing-grid`
if you have read [the trap](#the-trap) and want it anyway.

### `status check: FAILED` during a refresh

The data pull was rate-limited and a race came back with no finishing status.
The dataset is **not** written and `refresh` stops without refitting — this is
the guard working. Either retry later, or rebuild from the warm cache:

```bash
./.venv/bin/python -m src.models.predict refresh --offline
```

### `coverage check: FAILED (N race(s) present before and missing now)`

The rebuild covers fewer races than the results it was about to replace, so
**nothing was overwritten**. Races vanish from a pull when the backend cannot
confirm them -- almost always an Ergast rate limit. The rows that survive are
perfectly valid, which is why no other check fires; only this one notices that
a third of the calendar went missing.

Retry once the limit clears, rebuild from the cache with `--offline`, or pass
`--allow-shrink` if you meant to narrow the data.

### `leakage check: FAILED` during a refresh

A feature is reading the current race's outcome. This is a code bug, not an
operational one — do not work around it. The failing feature is named in the
output; fix it in `src/features/build_features.py` and add a case to
`tests/test_leakage.py`.

### `ValueError: Failed to load any schedule data.` / `any API: 500 calls/h`

Ergast allows 500 calls an hour and you have spent them, or the network is
down. FastF1 tries three backends for a calendar and raises only when all three
fail. The season is reported as a failure and the run continues; if *every*
season fails the build stops with instructions rather than a traceback.

The fix is not to wait — it is to stop going to the network at all:

```bash
F1_FASTF1_CACHE="$PWD/Data/raw" \
  ./.venv/bin/python -m src.models.predict refresh --offline
```

**Check `F1_FASTF1_CACHE` first.** The default is `Data/raw/fastf1_cache`, and
if your cache actually lives at `Data/raw` then every session is a cache miss
and a full rebuild will exhaust the hourly limit on its own. `find Data/raw
-name '*.ff1pkl' | wc -l` tells you where the cache really is.

### The pull is slow, or hits HTTP 429

`api.jolpi.ca` rate-limits. A warm cache avoids it entirely:

```bash
F1_FASTF1_CACHE=/path/to/Data/raw \
  ./.venv/bin/python -m src.models.predict refresh --offline
```

### Predictions all look the same, around 0.14

The model has little to separate the field. Check `--stage`: at `pre_weekend`
this is normal and correct. At `post_quali` it suggests the grid did not load —
look for `grid` showing `n/a` in the output.

### Every prediction looks high (0.2–0.45)

Almost always the missing-grid trap: the post-quali model with
`--allow-missing-grid` treats every driver as starting last. Check whether the
`grid` column reads `n/a`.

### `MergeError: Merge keys are not unique`

A race is present twice in the results. Usually a partial pull that appended
rather than replaced. Rebuild the dataset:

```bash
./.venv/bin/python -m src.data.generate_dataset --seasons 2018-2026 --offline
```

---

## What the model is not

Worth stating plainly, because the output looks more authoritative than the
evidence supports.

- **It ranks better than it calibrates.** The ordering is sound; the
  probabilities are compressed toward the base rate. Treat "who is most at
  risk" as the product and the number itself as a rough guide.
- **The evidence is thin.** The ranking lift clears significance on 2024–2026,
  but the Brier improvement only clears it on 2026 — the single season that most
  favours a short training window. Pooled across all three it does not.
- **It is a within-race ranker.** `top2_lift` is the honest measure of use: of
  the two cars flagged, how many stopped. Recent lift is roughly 1.5–1.8× the
  base rate, not a solved problem.

---

## Related

- `CLAUDE.md` — pipeline invariants and the rules that hold across `src/`
- `References/data_dictionary.md` — every feature, and when it becomes knowable
- `src/models/train.py` — walk-forward evaluation, ablation, permutation importance
- `src/models/store.py` — persistence and the manifest checks
- `src/models/monitor.py` — the performance log and drift check
