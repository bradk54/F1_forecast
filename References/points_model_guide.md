# The finishing-order and championship model: a guide

How the model works, what it assumes, how to run it, how to keep it current,
and the handful of mathematical ideas it rests on.

Four documents cover this model; read the one that answers your question:

| document | answers |
| --- | --- |
| **this guide** | how does it work, how do I run it, how do I update it |
| `points_model_results.md` | why each choice was made — every number that decided something |
| `finishing_order_models.md` | the literature, and why Plackett-Luce |
| `points_model_design.md` | the original brief and the DNF lessons behind it |

---

## 1. The pipeline, end to end

```
 FastF1 / Jolpica  (network; 500 calls an hour -- see CLAUDE.md)
        |
        v
 src/data/ingest.py ---------> Data/processed/race_results.parquet
        |                       every grand prix AND sprint, 2018 on, with points
        v
 src/features/labels.py        Status + ClassifiedPosition -> dnf, classified, ...
 src/features/build_features.py  leakage-safe rolling history (DNF features)
        |
        v
 Data/processed/dnf_dataset.parquet     one row per starter per grand prix
        |
        +------------------------------+------------------------------------+
        |                              |                                    |
        v                              v                                    |
 ATTRITION                      ORDER FEATURES (at fit time)                |
 the DNF model                  src/features/order_features.py              |
 src/models/train.py            team pace, driver vs team-mate,             |
 logistic on grid (Saturday)    lineage-keyed, EWMA half-lives              |
 or reliability (pre-weekend)          |                                    |
        |                              v                                    |
        |                       ORDERING                                    |
        |                       stagewise Plackett-Luce                     |
        |                       src/models/ranking.py                       |
        |                              |                                    |
        +--------------+---------------+                                    |
                       v                                                    |
              ONE RACE: sample attrition, then an order among survivors     |
              src/models/ranking.py:sample_positions                        |
                       |                                                    |
                       v                                                    |
              POINTS: the season's rules  (src/models/points_table.py)      |
                       |                                                    |
          +------------+-------------+                                      |
          v                          v                                      |
   `forecast next` / `race`    `forecast season`                            |
   per-driver odds for          loop over the remaining calendar,           |
   one race                     10,000 trials -> both championships  <------+
                                src/models/season.py              standings from
                                                                  race_results
```

Evaluation sits beside it: `order_eval.py` walks every model forward race by
race, `tuning.py` searches features and hyperparameters on the development
races, and `season.py:score_season` backtests the championship simulation.

**Nothing is saved between runs.** Every command refits from the parquets in
seconds, so "updating the model" means updating the data (§6).

---

## 2. The four stages

### 2.1 Attrition — will the car finish?

The existing DNF model, unchanged: a logistic regression refitted on the last
40 races. After qualifying it uses grid position alone; before qualifying it
uses the team's and the driver's 10-race retirement rates, which is barely
better than the base rate — and the pipeline says so rather than hiding it.

Two additions for the season simulation:

- **Non-starts.** The modelling table holds starters only, so the chance an
  entrant does not start at all comes from the raw results over the same
  window (0.85% of entries since 2021; 2.3% in 2026).
- **Sprints** retire cars about half as often as grands prix, so a sprint uses
  0.47 × the grand-prix probability.

### 2.2 Ordering — among the survivors, who finishes where?

A **stagewise Plackett-Luce** model (§7). Each car gets a log-strength
`s_i = beta . x_i` from a few features; the winner is drawn from the
survivors, then second place from the rest, and so on, each position at its
own scale.

| | pre-weekend | post-qualifying |
| --- | --- | --- |
| used by | `next`, `season` | `next --stage post_quali`, `race` |
| features | `team_finish_pct_ewma`, `driver_mate_grid_delta_ewma`, `team_grid_pct_season` | `grid_position`, `team_rank_in_field`, `driver_grid_pct_ewma` |
| team / driver half-life | 4 / 24 races | – / 4 races |
| team-mate delta clip | 0.5 | – |
| training window | last 100 races, decay half-life 40 | same |
| likelihood | top 10 of each race, 10 position scales | same |
| priors (`l2`, `l2_scale`) | 0.1, 0.1 | 0.1, 0.1 |

Every value is a measured constant in `src/models/forecast.py`, with its
evidence in the comment above it.

### 2.3 Points

`src/models/points_table.py`. Grand prix 25-18-15-12-10-8-6-4-2-1; a
fastest-lap point for a top-ten finisher from 2019 to 2024 only; sprints
3-2-1 in 2021 and 8 down to 1 since 2022. Retired cars score nothing — in this
dataset no car that retired and stayed classified ever scored.

### 2.4 The season

From the real standings after the last completed round, every remaining race
(and sprint) is drawn whole, 10,000 times. Each trial also draws one of 40
bootstrap refits of the ordering model, so the spread includes uncertainty
about the coefficients, not only race-day luck. Constructors are the sum of
their drivers race by race; a driver who has left the grid keeps their points.

---

## 3. The features

Percentiles run from 0 (front of the field) to 1 (back), so **lower is always
better**, and a 20-car and a 22-car grid are comparable. Every feature reads
prior races only; team history follows the organisation across rebrands
(`src/data/teams.py`).

| feature | stage | what it is | why it is there |
| --- | --- | --- | --- |
| `team_finish_pct_ewma` | pre-weekend | the team's finishing percentile among classified cars, exponentially weighted (half-life 4 races) | car race pace, recent — the first feature selection took |
| `team_grid_pct_season` | pre-weekend | the team's mean grid percentile this season, reset each winter | the car's settled level this year; pairs with the short EWMA to give two timescales |
| `driver_mate_grid_delta_ewma` | pre-weekend | driver's grid percentile minus the team-mate's, weighted over ~24 races, each race clipped at ±0.5 | the only measure of the driver that controls for the car |
| `grid_position` | post-quali | starting slot; pit-lane starts moved to the back | the dominant single signal |
| `team_rank_in_field` | post-quali | the team's rank by 5-race points rate within this race (DNF pipeline column) | "this car is faster than its grid slot" — a penalty, a bad qualifying |
| `driver_grid_pct_ewma` | post-quali | the driver's own grid percentile, half-life 4 races | recent qualifying form |

About twenty more candidates were built, registered and **rejected** by the
selection protocol — racecraft, circuit suitability, `log(grid)`, grid ×
circuit, grid × rain, and the season-reset driver deltas. They stay in the
registry so the evidence can be re-run; `points_model_results.md` §5 has each
verdict.

---

## 4. Assumptions

Each one can be wrong. The right-hand column is how you would find out.

| # | assumption | what breaks it | how you would notice |
| --- | --- | --- | --- |
| 1 | features use prior races only | a builder that forgets to shift | `detect_order_leakage` and the test suite fail |
| 2 | a retirement says nothing about pace | a driver crashing *because* they were slow | strengths of crash-prone drivers look high; always read them with the DNF model attached |
| 3 | the entry list is the last race's | a mid-season swap | the swap's first race is forecast with the old driver |
| 4 | the running order is frozen after the last race | upgrades; a regime change | season backtest coverage (§6.4); a team out-scoring its interval |
| 5 | future races use the circuit the event name last used | a new venue or a renamed event | circuit features fall back to the field mean |
| 6 | the live grid is the qualifying order | grid penalties applied after qualifying | a penalised car forecast from its qualifying slot |
| 7 | team-mates' outcomes are independent within a race | shared strategy calls, shared failures | not yet measured; the research brief's Method 2 tests it |
| 8 | retirements are independent between cars | a first-lap pile-up takes out four | per-race scores are noisier than per-car ones; the race bootstrap accounts for it |
| 9 | sprints share the grand-prix strengths | a car built for one-lap or long-run pace | sprint-weekend errors (not measured separately) |
| 10 | the coefficients hold across the last 100 races | a regulation change | re-run `tune` each winter |
| 11 | a driver's skill is their grid against the team-mate, over ~24 races | a rookie improving; a grid penalty | **known live case**: Antonelli is under-rated in 2026 (results §11) |
| 12 | a team's pace drifts as one random walk shared by both cars, and nothing else moves independently | a season where the order moves faster than `team_sd` allows, or one team-mate moving on their own | **known**: at the shipped `team_sd` 0.5, holdout 80% coverage is still 0.73 (constructors) and 0.70 (drivers); `driver_sd` is not applied yet (results §10) |

---

## 5. Running it

Everything runs from the repository root with the project venv. None of it
touches the network except where marked.

```bash
./.venv/bin/python -m src.models.forecast next                      # next race, before qualifying
./.venv/bin/python -m src.models.forecast next --stage post_quali   # after qualifying (1 API call)
./.venv/bin/python -m src.models.forecast race 2026 14              # backtest one completed race
./.venv/bin/python -m src.models.forecast season --write            # both championships
./.venv/bin/python -m src.models.forecast evaluate --write          # held-out scores
./.venv/bin/python -m src.models.forecast backtest --write          # season-sim calibration
./.venv/bin/python -m src.models.forecast tune --ranker --write     # re-run selection and tuning
```

| command | reads | output |
| --- | --- | --- |
| `next` | the dataset; the calendar from the FastF1 cache | P(win), P(podium), P(points), P(DNF), expected points, per driver |
| `next --stage post_quali` | + the qualifying classification | the same, sharper; refuses if qualifying has not run |
| `race Y R` | a race already in the dataset, and its real grid | the Saturday forecast beside what happened; fitted only on earlier races |
| `season` | the dataset, results, calendar | points now, mean, 10th/50th/90th percentile, P(champion), P(top 3) |
| `evaluate` | the dataset | the held-out tables of the results doc |
| `backtest` | the dataset, results | coverage, PIT and CRPS of season forecasts from completed seasons |
| `tune` | the dataset | chosen features and hyperparameters; tables with `--write` |

`--write` saves CSVs to `Reports/`, which are local artefacts (gitignored).
The durable record is `References/points_model_results.md`.

**Working from a git worktree:** `Data/` is not versioned, so point at the
main checkout's data:

```bash
F1_DATA_DIR=/path/to/main/checkout/Data ./.venv/bin/python -m src.models.forecast season
```

and **never** set it when running `pytest` — the suite must only ever see an
empty data directory.

---

## 6. Keeping it current

### 6.1 Every race weekend

There is no model file to retrain: every command refits in seconds from the
dataset. Keeping the forecast current means keeping the **data** current,
which the DNF pipeline already does:

| when | run |
| --- | --- |
| Tuesday, after the race is classified | `python -m src.models.predict refresh --since 2026` — pulls the new results (~38 API loads), rebuilds the dataset, refits the DNF model |
| then | `python -m src.models.forecast season --write` |
| Thursday | `python -m src.models.forecast next` |
| Saturday, after qualifying | `python -m src.models.forecast next --stage post_quali` |
| after the race | `python -m src.models.forecast race <year> <round>` — how did Saturday's forecast do |

### 6.2 Every winter, or after a regulation change

The measured choices are only as good as the seasons they were measured on.

1. Move the periods forward in `src/models/order_eval.py`: `DEV_START` /
   `DEV_END` to the most recent four settled seasons, `TEST_START` to the
   season after.
2. `forecast tune --ranker --write`. Compare with the constants in
   `forecast.py`.
3. If they changed, update `PRE_WEEKEND_SPEC`, `POST_QUALI_SPEC`, the two
   `OrderFeatureConfig`s and `BOOSTED_PARAMS` — with the new evidence in the
   comment above each, as the current ones have.
4. `forecast evaluate --write` once. Record the tables in the results doc.
5. Bump `LATEST_SEASON` in `src/config.py` in January (the DNF runbook's rule).

**Never tune on the test table.** If a live forecast looks wrong, add a
hypothesis to the development search, as §11 of the results doc records doing.

### 6.3 Adding a feature

1. Build it in `src/features/order_features.py` — strictly prior races (use
   `prior_ewma` / `prior_rolling` / `prior_expanding`), aggregate team features
   at the team-race level on `team_lineage`, percentiles with 0 = front.
2. Register it in `src/features/registry.py` with its **stage** and
   `table="order_frame"`, and add it to `ORDER_FEATURES`.
3. Add it to the candidate lists in `src/models/tuning.py`, with a comment
   naming the hypothesis.
4. `pytest` — the leakage tests run `detect_order_leakage` over every order
   feature.
5. `forecast tune`, then `forecast evaluate`. It ships if selection takes it.

Remember that a feature constant within a race (circuit, weather, calendar)
**has no effect on its own** in Plackett-Luce — build it as an interaction.

### 6.4 Tuning the season noise

The season simulation's correlated pace offsets (`draw_trial_offsets`) have
scales in `SeasonNoise`; the shipped default lives in
`forecast.DEFAULT_SEASON_NOISE` (currently `team_sd=0.5`, `driver_sd=0.0`),
with the evidence for it in the comment above it. To re-tune, choose on
completed development seasons, then check on later ones:

```bash
./.venv/bin/python -m src.models.forecast backtest --seasons 2021 2022 2023 \
    --team-sd 0 0.1 0.2 0.3 0.5 --driver-sd 0
./.venv/bin/python -m src.models.forecast backtest --seasons 2024 2025 \
    --team-sd 0 0.2 0.3 0.5 --driver-sd 0
```

Sweep `--driver-sd` only once `draw_trial_offsets` uses it. It is a no-op today,
and a sweep over it would report identical rows.

Aim for `coverage80` near 0.80, `pit_sd` near 0.289, and the lowest `crps` —
and expect them not to agree. On the calm 2021–2023 seasons `crps` bottomed
out near `team_sd` 0.2–0.3 while coverage kept rising; on 2024–2025 every
metric kept improving through 0.5. The default was chosen for the
regime-shift case (results §10).

---

## 7. The maths, quickly

### 7.1 Plackett-Luce: a race as a chain of softmaxes

Give each car a positive strength `θ_i = exp(s_i)`. The winner is drawn with
probability proportional to strength; remove them and repeat:

```
P(A beats B beats C)  =   θ_A / (θ_A + θ_B + θ_C)   ×   θ_B / (θ_B + θ_C)
```

With `s = (2, 1, 0)`: the first factor is `e² / (e² + e + 1)` = 0.665, the
second `e / (e + 1)` = 0.731, so this exact order has probability 0.486. Two
properties follow directly:

- **It is a distribution over whole orders**, so there is always exactly one
  winner and the positions always add up — which twenty separate per-driver
  models cannot promise.
- **Only differences in strength matter.** Adding the same constant to every
  `s_i` multiplies every `θ` by the same factor, which cancels in every
  fraction. That is why a circuit or weather feature does nothing unless it is
  multiplied by something that differs between cars.

The model is `s_i = β · x_i`: a linear score, like a logistic regression's,
but for "who beats whom" rather than "yes or no".

### 7.2 Stagewise scales: a temperature per position

Standard Plackett-Luce uses the same `s` at every step. The stagewise version
draws position `j` at scale `α_j`:

```
P(car i takes position j | cars left)  =  exp(α_j · s_i) / Σ_left exp(α_j · s_t)
```

with `α_1 = 1`. A small `α_j` flattens the softmax (the result at that position
is closer to a lottery); a large one sharpens it. Fitted on 2019–2023 the
scales fall from 1.00 for the winner to 0.64 for second and 0.19 for tenth:
**the front of a grand prix is far more predictable than the midfield**. It is
the idea Benter (1994) used to price place and show bets in horse racing.

### 7.3 Fitting: maximum likelihood with a prior

The log-likelihood of the observed orders is a sum over races and positions of

```
α_j · s_winner(j)  −  log Σ_left exp(α_j · s_t)
```

and the model maximises it minus `½ · l2 · |β|²` (a Gaussian prior on the
coefficients, and the same on `log α` with `l2_scale`). Its gradient has a
readable form — for each position, **the features of the car that took it,
minus the features the model expected to take it**:

```
∂/∂β  =  Σ  α_j · ( x_winner(j)  −  Σ_left p_t · x_t )
```

For standard PL the problem is concave: one optimum, no restarts. It is solved
with L-BFGS in about 20 ms.

**Truncation** keeps only the first *k* positions of each race in the sum
(here the top ten), spending the fit on where points are decided.

### 7.4 Retirements: factorise, do not rank

```
P(result)  =  P(who survives)  ×  P(order | survivors)
               the DNF model      Plackett-Luce, fitted on finishers only
```

A retirement contributes to the first factor and nothing to the second, so the
ordering model learns pace from cars that finished. Ranking retirements last
instead would teach it that fragile cars are slow, and count reliability
twice. Measured: significantly worse (results §9). The cost to remember: a
fitted strength is **pace given the car finishes**.

### 7.5 Sampling: the Gumbel-max trick

To draw a Plackett-Luce order, add independent Gumbel noise to every
log-strength and sort:

```
order  =  argsort( s_i + g_i ),    g_i = −log(−log(u_i)),   u_i ~ Uniform(0, 1)
```

This is exact, not an approximation: the largest of `s_i + g_i` is car *i*
with probability `exp(s_i) / Σ exp(s_t)`, and the rest follows by the same
argument. The stagewise model draws one position at a time with fresh noise
at each scale. The season simulation draws 10,000 seasons this way in a couple
of seconds.

### 7.6 Exponential weighting and half-lives

An EWMA with half-life *h* weights a race *k* races ago by `0.5^(k/h)`: with
*h* = 4, last race counts twice as much as the race four back. It is how the
features stay recent without a hard window, and the half-lives are tuned.

### 7.7 Shrinkage (empirical Bayes)

A circuit visited three times has a noisy "how much does grid matter here"
estimate. It is pulled toward the field average in proportion to how noisy it
is: `(Σ circuit values + k · field mean) / (n visits + k)`. The split-half
reliability of 0.36 implied *k* ≈ 6 pseudo-visits. (The feature was then
rejected anyway.)

### 7.8 Scoring

- **Log-likelihood in nats.** The log-probability the model gave the order
  that actually happened. Reported as the gain over "every order equally
  likely": the tuned Saturday model gains 8.2 nats a race, meaning it gave the
  real top ten about e^8 ≈ 3,800 times the probability a coin-flip model
  would.
- **Ranked probability score (RPS).** For ordered outcomes. Compare the
  predicted and actual *cumulative* distributions over P1, P2, … P10, "no
  points", and average the squared gaps. Predicting P4 for a P3 finish costs
  little; predicting P9 costs a lot. 0 is perfect; lower is better.
- **Brier score and calibration slope.** For a yes/no event like "podium":
  the mean squared gap between probability and outcome, and the slope of a
  logistic refit of outcomes on predicted log-odds. A slope of 1 means "30%"
  happens 30% of the time; above 1, the model is too timid; below 1, too sure.
- **Squared error for expected points, not absolute error.** Absolute error is
  minimised by the median, and most drivers' median is 0 points, so it rewards
  a model that confidently predicts zero. The mean — which is what "expected
  points" is — is scored by squared error.

### 7.9 Honest evaluation

- **Walk-forward.** Each race is predicted by a model fitted only on races
  before it, the way it would be run.
- **Development and test periods.** Choices are made on 2020–2023 and scored
  once on 2024 onward. A search that can see the test races reports its own
  optimism as skill.
- **Paired tests over races.** Two models are compared race by race, and the
  race — not the car — is the independent unit, because a safety car or a
  first-lap pile-up hits every car at once. A feature needs t ≥ 2.
- **Race bootstrap** for intervals: resample whole races with replacement and
  recompute. **Bayesian bootstrap** for the season simulation's parameter
  uncertainty: refit with random Dirichlet weights on races, one refit per
  trial.

### 7.10 Checking a season forecast

- **Coverage.** Across many forecasts, the actual final total should fall
  inside the 10th–90th percentile band 80% of the time.
- **PIT (probability integral transform).** The share of simulated totals
  below the actual one. For a calibrated forecast it is uniform on [0, 1]
  (standard deviation 0.289); a larger spread means the actual totals keep
  landing in the tails — intervals too narrow.
- **CRPS.** The ranked probability score's continuous cousin: how far, in
  points, the forecast distribution sits from the outcome. Lower is better.

---

## 8. Where things live

| file | holds |
| --- | --- |
| `src/data/teams.py` | constructor lineage |
| `src/features/order_features.py` | the order features; `OrderFeatureConfig`; `detect_order_leakage` |
| `src/features/registry.py` | every feature, its stage, and where it is built |
| `src/models/ranking.py` | Plackett-Luce: likelihood, gradient, fitting, sampling |
| `src/models/order_eval.py` | walk-forward, composite scoring, `DEV_*` / `TEST_START` |
| `src/models/tuning.py` | selection, pruning, coordinate and random search, candidate lists |
| `src/models/boosted_ranker.py` | the gradient-boosted comparison model |
| `src/models/points_table.py` | points rules by season |
| `src/models/season.py` | season and single-race simulation; backtest scoring |
| `src/models/forecast.py` | the CLI and the measured constants |
| `tests/test_{ranking,order_features,order_eval,season}.py` | the model's tests; `tests/synthetic.py:make_order_results` generates known-truth races |
