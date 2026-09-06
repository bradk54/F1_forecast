f1-forecast-engine
==================

Forecasting Formula 1 outcomes. Two models live here:

- **Points** (`Notebooks/2_0_Model_Development.ipynb`) — how many points will a driver score?
- **Retirement** (`Notebooks/3_0_DNF_Dataset.ipynb`) — will a driver finish the race at all?

---

## Retirement (DNF) prediction

### Quick start

```bash
pip install -r requirements.txt

# Full build: results + telemetry-derived circuit profiles, 2018 onward.
python -m src.data.generate_dataset --seasons 2018-2025

# Rebuild feature logic without touching the network.
python -m src.data.generate_dataset --skip-download

pytest                                    # 125 tests, no network required
python -m scripts.write_data_dictionary   # regenerate References/data_dictionary.md
```

The first build downloads one telemetry session per event and takes 30–60 minutes on a
cold cache. Budget a few GB. It needs egress to `livetiming.formula1.com` and
`api.jolpi.ca`; `src.data.ingest.check_connectivity()` reports clearly if either is
blocked, which some corporate and sandboxed networks do.

### Three ideas the dataset is built around

**1. The circuit is measured, not named.**

A one-hot per venue learns nothing that transfers to a circuit the model has not seen, and
it cannot express why Monza and Monaco break cars differently. So every circuit is
characterised from its position and speed telemetry instead. FastF1 gives the car's X/Y/Z
at ~10 Hz alongside speed, throttle, brake and gear; resampling that onto a uniform
distance grid and differentiating twice with respect to arc length yields the curvature
`κ = (x'y'' − y'x'') / (x'² + y'²)^{3/2}` at every point on the lap, and from it the
corner radii, the lateral load `v²κ`, the straight share, and the direction of every turn.

That produces around forty measured columns per circuit-season, summarised by three
transparent composite indices:

| index | components | reads as |
| --- | --- | --- |
| `track_speed_index` | mean speed, high-speed share, full-throttle share, corner density | Monza high, Monaco low |
| `mechanical_stress_index` | full throttle, braking density, peak deceleration, gear changes | hypothesised driver of **car** failures |
| `incident_exposure_index` | corner density, low-speed share, corner radius | hypothesised driver of **driver** failures |

The two stress indices are hypotheses with hand-chosen weights, not findings. Section 7 of
the notebook tests them by ablation; treat them as something to falsify.

Profiles are keyed on `(circuit_key, year)` because layouts change — Zandvoort gained
banking in 2021, Melbourne was reprofiled in 2022, Yas Marina reworked in 2021. A season
with no usable telemetry falls back to that circuit's median across the seasons that have
it. Three things telemetry cannot see — wall proximity, floodlights, and altitude — come
from a small curated table in `src/data/circuits.py`, each entry sourced.

**2. Leakage is enforced, not assumed.**

Every history feature is shifted within its entity before it is aggregated, so no feature
can see the race it describes. That is checked rather than promised:
`detect_target_leakage` flips one event's outcomes, rebuilds the entire feature table, and
reports any feature that moved at or before that event. The build refuses to write if
anything is found.

The detector is itself tested against deliberately planted leaks — a `cumsum` with no
shift, and a same-race team aggregate — so a clean pass means something. It has already
caught two real bugs here: a team aggregate joined back from the current race, and a
`(Year, RoundNumber)` key that stopped being unique on sprint weekends.

**3. Features are tagged with when they become knowable.**

| stage | available | use for |
| --- | --- | --- |
| `pre_weekend` | Monday before the race | season simulation, early forecasting |
| `post_quali` | Saturday evening | the strongest honest forecast |
| `race_day` | **after the race** | retrospective analysis only |

Grid position is one of the strongest single predictors of retirement and is unknown until
Saturday; observed weather is never available at forecast time at all. `registry.py` makes
that choice once, in code, so a Saturday feature cannot drift into a Monday model.

### The target

`dnf = 1` when the car stopped before the end of the race. Two cases decide whether this is
right, and both are handled explicitly:

- **A driver who retires past 90% distance is still a DNF.** They keep an official
  classification, so `ClassifiedPosition` alone would score it a finish and systematically
  under-count retirements. `dnf_classified` flags exactly these rows.
- **A disqualification after taking the flag is not a DNF.** A DSQ is a scrutineering
  outcome applied to a car that completed the race. `dnf_cause` still records it.

`dnf_cause` splits retirements into `mechanical`, `collision`, `driver_error`,
`disqualified`, `withdrawn` and `other`, so the binary model can become a competing-risks
model without rebuilding anything. Non-starters are dropped: predicting a withdrawal is a
different problem. Sprints inform the rolling history but are not modelling rows — a 100 km
sprint and a 305 km grand prix do not share an attrition process.

Full column-by-column reference: [`References/data_dictionary.md`](References/data_dictionary.md).

### Forecasting an upcoming race

Not built yet. The pipeline is designed for races that have already run, and turning it
into a weekly forecast needs five small modules, not a rewrite — every history feature
shifts before it aggregates, so a row for an unraced event is already a legal input to
the existing feature builder.

The plan, the traps it has to avoid, and the weekly ritual it implies:
[`References/race_weekend_inference.md`](References/race_weekend_inference.md).

### Evaluating it

**Accuracy is the wrong metric and nothing here reports it.** About one car in seven
retires, so "everyone finishes" scores ~86% accuracy while carrying no information. The
headline number is **Brier skill**: how much the forecast improves on always predicting the
base rate. Zero means the model has learnt nothing; negative means it is worse than
nothing.

Evaluation is walk-forward by season — train on every prior season, predict the next, never
the reverse. Confidence intervals resample whole races rather than rows, because one
first-lap pile-up retires several cars at once and driver-race rows inside an event are not
independent.

---

## Layout

```
|- Data                     <- generated, never committed (see .gitignore)
|  |- raw                   <- FastF1 HTTP cache; several GB once warm
|  |- processed             <- race_results / circuit_profiles / dnf_dataset parquet
|
|- Models                   <- trained, serialised models
|
|- Notebooks
|  |- 1_0_web_scrape        <- pitwall.app scraping (earlier approach)
|  |- 1_1_EDA_2024          <- exploratory analysis of the 2024 season
|  |- 2_0_Model_Development <- points forecasting
|  |- 3_0_DNF_Dataset       <- retirement dataset and baselines
|
|- References
|  |- data_dictionary.md    <- generated from the feature registry
|  |- race_weekend_inference.md  <- plan for per-race forecasting (design, not built)
|
|- Reports/figures
|
|- scripts
|  |- write_data_dictionary.py
|
|- src
|  |- config.py             <- repo-relative paths and pipeline constants
|  |- data
|  |  |- ingest.py          <- FastF1 loading; the only module that hits the network
|  |  |- circuits.py        <- curated circuit reference (street / night / altitude)
|  |  |- generate_dataset.py<- CLI orchestrator
|  |- features
|  |  |- labels.py          <- classification codes -> modelling targets
|  |  |- track_profile.py   <- telemetry -> circuit geometry and speed character
|  |  |- build_features.py  <- leakage-safe rolling history
|  |  |- registry.py        <- feature catalogue with stage tags
|  |- models
|     |- train.py           <- baselines, walk-forward evaluation, ablation
|
|- tests
   |- synthetic.py          <- FastF1-shaped fixtures; no network needed
```

## Testing

```bash
pytest -q
```

125 tests, none of which touch the network. The F1 APIs are rate-limited everywhere and
blocked outright on some networks, so the pipeline is validated against generated data whose
schema matches FastF1's. `tests/synthetic.py` builds circuits from segment lists and Fourier
series and runs a quasi-steady-state lap simulation over them, which means the geometry has
closed-form answers: a circle of radius R must return curvature exactly `1/R` and turn
through exactly `2π`.

That fixture validates *plumbing*, not *findings*. Its circuits carry no relationship
between geometry and retirement, so an ablation run against it correctly shows the track
features adding nothing. Only real data can answer whether they help.
