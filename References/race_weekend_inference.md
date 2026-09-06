# Race-weekend inference: a plan

**Status:** design, not yet built. Hand-written; unlike `data_dictionary.md`, no script
regenerates this file.

**The question this answers:** every race weekend, on a laptop, how do we go from "a race is
coming" to "here is each driver's retirement probability" — without breaking the guarantees
the pipeline already makes, and while leaving code a stranger can read in two years?

---

## 1. What already works

More than a first read suggests. The pipeline is closer to inference-ready than most
research code ever gets, and it is worth naming why before listing what is missing.

| Capability | Where | Why it matters on Sunday |
| --- | --- | --- |
| Repo-relative, env-overridable paths | `src/config.py` | Runs unchanged on the laptop, in CI, in a container |
| Network isolated to one module | `src/data/ingest.py` | One place to add retries, caching and an offline switch |
| Features tagged by when they become knowable | `src/features/registry.py` | A Saturday feature cannot leak into a Monday model |
| Leakage enforced by a detector, not a promise | `build_features.detect_target_leakage` | The build refuses to write when a feature sees its own race |
| Honest metrics, walk-forward by season | `src/models/train.py` | Backtest numbers mean something |
| 125 tests, none touching the network | `tests/` | You can refactor without a live API |
| Not-yet-run races already carried forward | `src/data/f1_loader.py` (`IsFuture`) | Someone already solved half of "what does an unraced row look like" |

The stage registry is the single most valuable asset here. Most forecasting projects
discover the difference between `post_quali` and `race_day` the hard way — by shipping a
model that scores beautifully in backtest and is useless on Thursday. This repo encoded that
distinction before writing a forecast. Build on it; do not route around it.

## 2. The idea the whole design turns on

**Every history feature shifts before it aggregates. Therefore a row for a race that has not
happened is a legal input to the existing feature builder.**

`prior_rolling` computes `s.shift(1).rolling(window).mean()` within each entity. The shift
means a row never contributes to its own feature. Flip that around and it says something
stronger: a row with *no outcome at all* still gets a complete, correct feature vector,
because its features were only ever built from the rows before it.

So the forecast path does not need its own feature code. It needs to append one
outcome-free row per entered driver to the labelled history and call
`build_history_features` — the same function, the same registry, the same leakage detector.

This is the entire design, and it is worth stating as a rule:

> **Never write a second implementation of a feature for inference.** If a feature cannot be
> computed by the training-time code path, that is a fact about the feature, not a reason to
> reimplement it.

The failure mode this avoids is training/serving skew — the model scoring a feature vector
built by different code from the one it learned on. Zinkevich's *Rules of Machine Learning*
puts it as Rule #29: the reliable way to train as you serve is to use the same code to
produce both.[^zinkevich] Sculley and colleagues catalogue the same problem as a leading
source of hidden technical debt in ML systems, and note that the glue code around a model
routinely dwarfs the model itself.[^sculley] A single feature builder is the cheapest
insurance available, and this repo is one small module away from having it.

## 3. Six gaps between here and Sunday

1. **No fitted model is ever saved.** `train.py` evaluates and returns scores. It has no
   `fit_final`, no `__main__`, and no reference to `MODELS_DIR`. Every forecast today would
   silently retrain.
2. **`src/models/predict.py` is empty** (0 bytes).
3. **The DNF ingest path cannot see a future event.** `collect_season_results` walks the
   schedule and calls `load_session(...)` for each round; an unraced round has no session to
   load. Nothing enumerates *upcoming* events on this path.
4. **No entry list.** Knowing a race is next Sunday is not knowing who starts it.
5. **Two loaders, two schemas.** `ingest.py` speaks long-format `RoundNumber` / `Position` /
   `Status`; `f1_loader.py` speaks wide-format `Round` / `RacePosition` / `IsFuture`. Sixteen
   uses of one key against thirty-two of the other. Both are good modules. Together they are
   the single largest threat to the readability you asked about.
6. **No output contract and no scoring loop.** A forecast that is printed and lost teaches
   nothing.

## 4. Proposed shape

Five files. No new top-level concept — `data` ingests, `features` builds, `models` fits and
predicts, `scripts` orchestrates.

```
src/data/schedule.py        NEW   ~150 lines   the calendar and the entry list
src/features/forecast.py    NEW   ~120 lines   assemble the outcome-free rows
src/models/persist.py       NEW   ~100 lines   save/load a fitted pipeline + its card
src/models/train.py         EDIT  +80 lines    fit_final() and a CLI
src/models/predict.py       FILL  ~180 lines   forecast_event() and a CLI
scripts/forecast_weekend.py NEW   ~60 lines    the one command you run
scripts/score_forecasts.py  NEW   ~80 lines    the loop that keeps you honest
Makefile                    NEW   ~30 lines    three verbs
```

### `src/data/schedule.py`

```python
def upcoming_event(after: datetime | None = None) -> pd.Series
def event_by_round(year: int, round_number: int) -> pd.Series
def entry_list(year: int, round_number: int, *, history: pd.DataFrame) -> pd.DataFrame
```

FastF1 already exposes `fastf1.get_events_remaining()`, which builds an `EventSchedule` of
events still to come and accounts for the time of day against the last session of an
event.[^fastf1events] Confirm the exact signature against your installed version — the API
has moved across the 3.x line — but the primitive exists, so `upcoming_event` is a thin,
honest wrapper rather than calendar arithmetic of our own.

`entry_list` matters more than it looks, and it should degrade down a ladder that mirrors
the stage ladder:

| Source | Available | Gives you |
| --- | --- | --- |
| Qualifying classification | Saturday evening | Drivers, teams, **and grid** — the `post_quali` row |
| FP1 classification | Friday | Drivers and teams, including a reserve driver in the car |
| Last completed race's lineup | Always | The `pre_weekend` assumption |
| A hand-written override file | Always | The escape hatch for news the API does not carry |

`f1_loader._latest_lineup` already implements rung three, including a fallback through
earlier races when the most recent is unusable. Lift it rather than rewrite it.

### `src/features/forecast.py`

```python
FORECAST_SENTINELS = {"dnf": pd.NA, "dnf_cause": pd.NA, "started": 1, "is_forecast": True}

def build_forecast_frame(
    labelled_history: pd.DataFrame,   # completed races, already through add_race_outcome_labels
    event: pd.Series,                 # one row from the schedule
    entries: pd.DataFrame,            # DriverId, TeamId, and GridPosition when known
    *,
    stage: registry.Stage = "post_quali",
) -> pd.DataFrame                     # the entry rows only, fully featured
```

It concatenates history and the sentinel-filled entry rows, calls `build_history_features`
on the whole thing, joins the circuit profile, and returns only the rows where
`is_forecast` is True. History goes in so the rolling windows have something to roll over;
it does not come back out.

**Label the history, never the forecast row.** See §6, trap 1 — this is not a stylistic
preference.

### `src/models/persist.py`

`joblib.dump` the fitted `Pipeline`, plus a JSON sidecar recording:

- the git commit the fit ran at, and whether the tree was dirty
- training seasons, row count, and observed base rate
- stage, model key, and **the exact ordered feature list**
- versions of scikit-learn, pandas, numpy and fastf1
- walk-forward scores measured at fit time
- a UTC timestamp

The feature list is the load-bearing entry. At predict time, assert the forecast frame's
columns equal it exactly — same names, same order — and fail loudly otherwise. That single
assertion is your train/serve skew guard, and it costs three lines.

Record the library versions because scikit-learn's own guidance is blunt about it:
pickled estimators are not guaranteed to load across versions, and are unsafe to load from
untrusted sources.[^sklearnpersist] Neither is a problem for a solo local project; both turn
a mystifying failure into a one-line diagnosis eighteen months from now.

### `src/models/predict.py`

```python
def forecast_event(
    dataset: pd.DataFrame,
    event: pd.Series,
    entries: pd.DataFrame,
    model: Pipeline,
    card: dict,
    *,
    stage: registry.Stage = "post_quali",
) -> pd.DataFrame   # DriverId, TeamId, p_dnf, plus provenance columns
```

Refuse `stage="race_day"` unless an explicit `retrospective=True` is passed. The registry
already documents that observed weather is unavailable at forecast time; make the code
enforce what the docstring says.

## 5. The weekly ritual

Three verbs, tied to the rhythm of a race weekend. This is the part that has to survive
contact with a real season.

```make
refresh:   ## Monday — pull last weekend, rebuild, refit, archive
	python -m src.data.generate_dataset --seasons 2018-2026
	python -m src.models.train --stage post_quali --save

forecast:  ## Saturday evening — the honest forecast
	python -m scripts.forecast_weekend --stage post_quali

score:     ## Monday — grade last week's forecast
	python -m scripts.score_forecasts
```

| When | Command | Why then |
| --- | --- | --- |
| Monday | `make refresh` | Last race is in the API; refit on it before it matters |
| Thursday (optional) | `make forecast STAGE=pre_weekend` | The early read; also the season-simulation input |
| Friday (new circuits only) | `make refresh` | FP1 is the first telemetry a brand-new venue ever has — see §6, trap 6 |
| Saturday evening | `make forecast` | Grid is known. This is the forecast worth keeping |
| Monday | `make score` | Join the forecast to what happened; append to the log |

**The `score` step is the one to protect.** It is the least fun and it is the only one that
converts this project from an interesting build into evidence about whether the model works.
A forecast you never grade is a hobby; a forecast you grade every week for a season is a
portfolio piece with a Brier skill number attached to it. Gneiting and Raftery's treatment
of strictly proper scoring rules is the reference for why Brier and log loss are the right
graders here and why accuracy is not: a proper rule is uniquely maximised by reporting your
true belief, which is exactly the property you want when the thing being graded is a
probability.[^gneiting]

### Where output goes

```
Reports/forecasts/2026_17_post_quali.parquet   # full frame, gitignored
Reports/forecasts/2026_17_post_quali.md        # the card, committed
Reports/forecast_log.parquet                    # append-only, one row per forecast
```

`.gitignore` excludes `*.parquet` but not `*.md`. Lean on that deliberately: **commit the
markdown card and your git history becomes a tamper-evident prediction log.** Every forecast
is timestamped by a commit you cannot backdate without effort. Nothing else you can build in
an afternoon does as much for the credibility of the numbers.

## 6. Traps I found in the current code

These are specific, verified against the source, and every one of them fails silently.

**1. A NaN status labels as a retirement.** `classify_status` maps `None`/`NaN` to `OTHER`
(`labels.py:171`), and `OTHER` is in the `stopped` set that sets `dnf = 1`
(`labels.py:269`). So the obvious implementation — append a future row with `Status = NaN`
and run `add_race_outcome_labels` over everything — marks every entered driver as having
already retired. The row's own features survive, because of the shift; the `dnf` column does
not, and any code that later treats the frame as training data inherits a fabricated
positive. *Mitigation: label history first, then concat sentinel rows carrying an explicit
`is_forecast` flag, and have `fit_final` refuse any frame where that flag is set.*

**2. Placeholder rows poison the rolling windows.** Append all eight remaining races at once
and race N+2's "previous 5 races" window spans five *positions*, three of which are
placeholders. `rolling(...).mean()` skips the NaN but the window still consumes the slot, so
a five-race rate quietly averages two races. Worse, `prior_count` uses `cumcount`, which
counts placeholders outright and inflates `driver_races_to_date` — a feature explicitly
tagged `allows_cold_start_nan=False`, meaning a wrong value there is indistinguishable from
a right one. *Mitigation: **forecast exactly one race at a time.** History plus one event's
entry rows, never more. Multi-race lookahead is a different problem — it needs recursive
simulation that samples outcomes for the intervening races — and that is the natural home
for the `pre_weekend` stage once single-race forecasting works.*

**3. An unknown grid silently becomes last place.** In `add_race_context`
(`build_features.py:367`):

```python
grid = grid.where(grid > 0, out["field_size"])
```

`NaN > 0` is False, so a missing grid is replaced by `field_size`. For completed races this
is harmless — grid is always known — and the intent is right: FastF1 codes a pit-lane start
as 0, which genuinely belongs at the back. For a pre-weekend row it asserts "starts last" as
though it were measured. The registry keeps `grid_position` out of a `pre_weekend` model, so
today's blast radius is inspection rather than prediction, but the guard is cheap.
*Mitigation: see trap 4 — this is what the pre-flight check catches.*

**4. `allows_cold_start_nan` is declared and never read.** It appears nowhere outside
`registry.py`. It is a hook someone built for exactly this moment. *Mitigation: add
`registry.check_forecast_frame(frame, stage)`, which fails when any feature with
`allows_cold_start_nan=False` is null and warns on unexpected nulls elsewhere. Run it before
every `predict_proba`. It converts trap 3 and every future variant of it from a silent wrong
number into a refusal to forecast.*

**5. Two schemas will drift.** `Round` versus `RoundNumber` is the visible symptom; the
disease is two modules that both know how to talk to FastF1 and disagree about what a race
is called. They have not bitten yet because they feed different notebooks. They will bite
the moment inference needs the schedule logic that lives in the wide loader and the column
names that live in the long one. *Mitigation, staged: (a) now — put the calendar in
`src/data/schedule.py` and have both loaders import it, so at least the definition of "next
event" is single-sourced; (b) later, when the points model also wants weekly inference —
pick `RoundNumber` as canonical, make `f1_loader` a wide-format **adapter** over the long
frame rather than a parallel ingest, and delete the duplicated session-loading code.* Do not
attempt (b) as part of this work. It is a real refactor and it is not on the critical path
to a first forecast.

**6. A brand-new circuit has no profile until Friday.** Profiles are keyed on
`(circuit_key, year)` and built from a reference lap. `pick_reference_lap` walks
`REFERENCE_SESSION_ORDER = ("Q", "R", "SQ", "S")` (`ingest.py:43`), and for a venue never
visited none of those four sessions exists until the cars run — so the fallback to that
circuit's median across seasons has no circuit to fall back on. This is not hypothetical;
the calendar keeps adding venues. *Mitigation: prepend `FP1` to that tuple for events with
no existing profile. That buys a Friday-evening profile and makes the Friday `make refresh`
row in §5 meaningful. Until it exists, have the forecast emit `has_track_profile = False`
and say so on the card rather than imputing quietly.*

**7. Reserve drivers.** `_latest_lineup` carries forward the previous race's grid. When a
driver is stood down on Thursday, the pre-weekend forecast confidently prices the wrong
person, and their entire feature vector — reliability history, pairing history, circuit
history — belongs to someone else. *Mitigation: the override file at rung four of the entry
ladder, plus a card line naming the lineup's source and date.*

## 7. Keeping it readable in 2027

You asked for this explicitly, so it gets its own section rather than a footnote.

**The docstrings in this repo are the best thing about it.** `train.py` opens by explaining
why accuracy is the wrong metric. `build_features.py` opens by explaining what a shift is
for. `registry.py` opens by explaining why a stage tag exists. These are not comments about
mechanics; they are arguments about design, written to a reader who does not yet know why
the code is shaped this way. That is a genuinely uncommon habit and it is the reason a
stranger could pick this up. **Name it as the house style so it survives your own future
hurry**, and hold new modules to it: every new file opens with the *why*, not the *what*.

Four more rules, in decreasing order of how much trouble they save:

1. **The registry is the contract.** Adding a feature means adding a `Feature` entry. No
   exceptions, including "just for an experiment" — `audit_coverage` exists precisely to
   catch the experiment you forgot to clean up.
2. **One feature builder.** Restated because it is the rule most likely to be broken under
   deadline, and the one whose breach is hardest to detect afterwards.
3. **Every new module gets a test that runs without network.** `tests/synthetic.py` already
   generates FastF1-shaped frames; a forecast-frame test is a dozen lines on top of it.
   Suggested first three: a forecast row carries no `dnf`; its features match those the
   training path produces for the same driver at the same point in history; and
   `check_forecast_frame` rejects a frame with a null `driver_races_to_date`.
4. **Notebooks consume `src/`; they never define pipeline logic.** The migration from
   notebook cells to `src/data/f1_loader.py` already happened once here. It was the right
   call. Do not let it reverse.

## 8. Build order

Sequenced so each step is independently useful and testable. Roughly a weekend of work, and
you can stop after step 4 with something that runs.

| # | Step | Unblocks |
| --- | --- | --- |
| 1 | `persist.py` + `fit_final()` + `train.py` CLI | A saved model. Independently useful today |
| 2 | `registry.check_forecast_frame` | Traps 3 and 4, before they can bite |
| 3 | `schedule.py` — `upcoming_event` and `entry_list` | Knowing who races next |
| 4 | `features/forecast.py` + `predict.py` + `forecast_weekend.py` | **First real forecast** |
| 5 | `score_forecasts.py` + the log | Evidence the thing works |
| 6 | FP1 fallback in `REFERENCE_SESSION_ORDER` | New circuits |
| 7 | Shared calendar; both loaders import it | Trap 5, stage (a) |
| 8 | Recursive multi-race simulation | Season-long forecasting |

## 9. Running it on the laptop, concretely

**Budget.** The cold build is 30–60 minutes and a few GB, per the README. The weekly refresh
is not: one or two results-only sessions (`laps=False, telemetry=False`), a feature rebuild
over roughly 3,400 driver-race rows for 2018–2026, and a `HistGradientBoostingClassifier`
fit on the same. Seconds of compute, and the network cost is one session. This problem is
comfortably laptop-sized and will stay that way — do not let anyone talk you into
infrastructure it does not need.

**Data source.** Ergast shut down at the start of 2025; Jolpica-F1 is the drop-in successor
FastF1 now reads, at `api.jolpi.ca/ergast/f1/`.[^jolpica][^fastf1jolpica] It is run by
volunteers and rate-limited, which is an argument for the offline-first design already in
place: `--offline` rebuilds from cache without touching the network, and `--skip-download`
reuses the parquet intermediates. Prefer both. Fetch once, rebuild many times.

**Scheduling — and a case against automating it yet.** You will want to `cron` this. I would
push back for one season.

A 24-race calendar gives roughly 24 executions a year. That is far too few for an automated
job to become reliable, and more than enough for the calendar's irregularities — sprint
weekends, triple-headers, a Las Vegas race that finishes at 06:00 UTC Sunday — to break your
timing assumptions repeatedly. Meanwhile the failure mode of a broken scheduled job is the
worst one available: a stale forecast that still looks fresh. And the value this project has
for you is mostly in the looking. Run it by hand, watch the numbers move, and build the
intuition that tells you when an output is wrong. Automate the ritual once it has become
boring, which is the only honest signal that it is well understood.

When that day comes, use `launchd` rather than `cron`. Apple has deprecated `cron` on macOS
in favour of `launchd`, and the decisive difference for a laptop is behaviour across sleep:
a `StartCalendarInterval` job whose window passed while the machine was asleep runs when it
wakes, whereas `cron` simply misses the slot.[^launchd] For a job that fires once a week on
a machine that is usually shut, that is the whole ballgame.

## 10. Open questions

Answering these changes what gets built, so they are worth a few minutes before step 1.

1. **Which forecast is the product?** A calibrated per-driver probability, a ranked
   "who is most at risk" list, or a field-level expected retirement count? All three come
   from the same model, but they want different cards, different scoring, and arguably
   different metrics — ranking rewards ROC-AUC, calibration rewards Brier.
2. **Do you want the pre-weekend forecast at all in year one?** Skipping it halves the entry
   ladder and defers the recursive-simulation work, at the cost of the Thursday read.
3. **How much does a wrong forecast cost you?** Nothing operational, so the honest answer
   shapes how much guard-railing is worth building. My instinct: build the pre-flight check
   (cheap, catches real bugs) and skip everything more elaborate.
4. **Points and retirement — one forecast or two?** They share a weekend and an entry list,
   and combining them is what makes trap 5's refactor worth doing. Deciding now tells you
   whether step 7 is optional or load-bearing.

---

### References

[^zinkevich]: Zinkevich, M. *Rules of Machine Learning: Best Practices for ML Engineering.*
Google. Rule #29 on training/serving skew.
<https://developers.google.com/machine-learning/guides/rules-of-ml>

[^sculley]: Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D.,
Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden Technical Debt in
Machine Learning Systems. *Advances in Neural Information Processing Systems, 28.*
<https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>

[^gneiting]: Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules,
Prediction, and Estimation. *Journal of the American Statistical Association, 102*(477),
359–378. <https://doi.org/10.1198/016214506000001437>

[^sklearnpersist]: scikit-learn developers. *Model persistence.* On cross-version load
guarantees and the risks of loading untrusted pickles.
<https://scikit-learn.org/stable/model_persistence.html>

[^fastf1events]: FastF1 developers. *Event Schedule — `fastf1.events`*, including
`get_events_remaining()`. <https://docs.fastf1.dev/events.html>

[^fastf1jolpica]: FastF1 developers. *Jolpica-F1 API Interface.*
<https://docs.fastf1.dev/api_reference/jolpica.html>

[^jolpica]: Jolpica-F1. *Free & Open Source Formula 1 Racing API* — the Ergast-compatible
successor, at `api.jolpi.ca/ergast/f1/`. <https://github.com/jolpica/jolpica-f1>

[^launchd]: Apple Inc. *Daemons and Services Programming Guide: Scheduling Timed Jobs* —
`launchd` over `cron`, and `StartCalendarInterval` behaviour across sleep.
<https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/ScheduledJobs.html>
