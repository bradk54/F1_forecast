# The finishing-order and championship model: what was built, and what it measured

The model `points_model_design.md` sketched and `finishing_order_models.md`
argued for, built and scored. This document records the hypotheses, the
protocol, every number that decided something, and — with equal weight — what
did not work.

Everything below is a **measurement on this repository's data** unless it says
otherwise. Code: `src/features/order_features.py`, `src/models/{ranking,
order_eval, tuning, boosted_ranker, points_table, season, forecast}.py`.

---

## 1. The model in one screen

```
for each race:
    1. attrition   DNF_i ~ Bernoulli(p_i)          the existing DNF model, unchanged
    2. ordering    stagewise Plackett-Luce over     log-strength s_i = beta . x_i;
                   the cars that survive            position j drawn at scale alpha_j
    3. points      the season's points map          fastest-lap point 2019-2024 only
season:  the same, looped over the remaining calendar, 10,000 times
```

| | pre-weekend (drives the season sim) | post-qualifying (Saturday) |
| --- | --- | --- |
| features | `team_finish_pct_ewma`, `driver_mate_grid_delta_ewma`, `team_grid_pct_season` | `grid_position`, `team_rank_in_field`, `driver_grid_pct_ewma` |
| training window | last 100 races, time-decay half-life 40 | same |
| likelihood | top-10 truncated, 10 free position scales | same |
| priors | `l2` 0.1 on coefficients, 0.1 on log-scales | same |
| feature half-lives | team 4 races, driver 24; team-mate deltas clipped at 0.5 | driver 4 |
| held-out RPS@10 | **0.1076** vs team order 0.1426 (−25%) | **0.0898** vs grid order 0.1051 (−15%) |
| held-out calibration (win / podium / points) | 0.69 / 0.99 / 1.12 | 0.81 / 0.95 / 1.05 |

The measured choices live as documented constants in `src/models/forecast.py`.

---

## 2. How it was tested

**Two periods, and the second is touched once.** Every feature and every
hyperparameter was chosen on the **development races**, 2020–2023 (83 races).
The **test races**, 2024 to 2026 R14 (62 races), were scored only after the
choices were fixed.

One honest qualification: the test table was scored **twice**. The second
time came after two driver features and a clipping knob were added to the
*development* search. They were prompted by a live-forecast observation (§11),
not by the test table, and they were chosen on development data like
everything else. The pre-weekend test RPS moved from 0.1072 to 0.1076.

**Walked forward race by race.** Every race is predicted by a model refitted on
races strictly before it — the sibling of `walk_forward_races`, scoring an order
rather than a row (`order_eval.walk_forward_order`). A test proves a race's
prediction cannot move when every later result is scrambled.

**The search objective is the model's own proper score.** Mean per-race
log-likelihood of the observed top ten among finishers (`ll10`): closed form, so
two configurations differ by what they predict, not by sampling noise.

**The race is the unit of evidence.** A feature joins only if its per-race gain
has a paired t-statistic ≥ 2 **and** a mean ≥ 0.02 nats. Composite results
carry race-bootstrap intervals. All summaries pool over cars; nothing averages
per-race ratios (see `pool-dont-average` in the project memory).

**Composite metrics** sample whole races (attrition, then order) and score the
eleven outcomes that pay: P1..P10 each, and "outside the points" — which lumps
P11 with a retirement, since neither scores.

| metric | why |
| --- | --- |
| `rps10` | ranked probability score over those eleven outcomes; the headline |
| `points_rmse` | expected points against points scored |
| `brier_*`, `calib_*` | does a "30% podium" happen 30% of the time |

**Expected points are scored with squared error, never absolute error.**
The first run reported `points_mae`, on which *deterministic grid order won*
(3.16 against 3.48–3.91 for every probabilistic model) while losing on every
proper score. MAE is minimised by the median, and a points distribution is
mostly zeros, so MAE rewards confident point forecasts over calibrated ones.
The mean is elicited by squared error.

---

## 3. What the data said before any model

Each hypothesis in §5 was grounded in one of these first.

| measurement (2018–2026) | value |
| --- | --- |
| mean per-race Spearman(grid, finish) among finishers | 0.74 (0.68–0.80 by season) |
| circuits, most and least grid-bound | Suzuka 0.88, Monaco 0.87 … Austin 0.66, Interlagos 0.63 |
| circuit retention, odd visits vs even visits | r = 0.36 |
| finishing-percentile variance explained by team-season / team-race | 58% / 82% |
| team grid percentile autocorrelation, lag 1 → lag 10 (same season) | 0.69 → 0.67 |
| season-mean team order vs previous season | r = 0.79–0.95, incl. 2022 and 2026 |
| team-mate qualifying delta, first half vs second half | r = 0.61 |
| places gained, residualised on grid, first vs second half | r = 0.46 (0.15 raw) |
| sprint retirement rate relative to a grand prix | 0.47 (6.3% vs 13.5%) |

Two data findings that were not hypotheses but mattered:

- **`TeamId` changes at every rebrand.** Four lineages since 2018 (Renault →
  Alpine; Toro Rosso → AlphaTauri → RB; Force India → Racing Point → Aston
  Martin; Sauber → Alfa → Sauber → Audi). Every existing `team_*` feature
  resets there. `src/data/teams.py` maps lineage.
- **`add_team_performance` says `Position` is NaN for a retirement; it is not.**
  Retirements carry back-of-field positions ordered by laps, so
  `team_avg_finish_5` counts them as finishes. The new features use
  `ClassifiedPosition`. The DNF feature is unchanged and flagged here.

---

## 4. The structural finding: one temperature cannot fit a grand prix

Standard Plackett-Luce was **underconfident at the front** (development races,
grid + team + driver features):

| training truncation | calib win | calib podium | calib points | RPS@10 |
| --- | --- | --- | --- | --- |
| full order | 2.82 | 1.89 | 1.16 | 0.1058 |
| top 10 | 2.30 | 1.49 | **1.01** | 0.1026 |
| top 3 | 1.54 | **1.02** | 0.81 | 0.1017 |
| top 1 | **1.21** | 0.84 | 0.73 | 0.1028 |

No single depth calibrates every tier: **the front of a grand prix is far more
predictable than the midfield**, and standard PL has one scale for every
position. Fitting a free scale per position (2019–2023) shows the shape:

```
P1 1.00  P2 0.64  P3 0.53  P4 0.49  P5 0.39  P6 0.39  P7 0.35  P8 0.30  P9 0.27  P10 0.19  P11 0.15  P12+ 0.10
```

This is not a functional-form artefact: with free scales, a `log(grid)` term
gets a coefficient of ~0 (and the wrong sign). The fix is Benter's (1994)
horse-racing correction — and an exponential fit to our profile gives 0.81 /
0.66 for second and third, almost exactly his discounts.

**The stagewise model** (`ranking.PlackettLuce(n_scales=10)`) fits the scales
jointly with the coefficients, with a prior pulling them toward 1:

| development races | standard PL | stagewise |
| --- | --- | --- |
| ll10 gain over uniform (nats/race) | 6.67 | **7.56** |
| RPS@10 | 0.1058 | **0.1015**, diff 95% CI [−0.0056, −0.0028] |
| Brier, win | 0.0359 | **0.0280** |
| calibration, win / podium | 2.83 / 1.91 | **1.44 / 1.07** |

Sampling stays exact: positions are drawn one at a time, each from a fresh
Gumbel draw at its own scale.

**Retirements ranked last** (treatment (a) of the brief) was far worse on the
development races: RPS 0.1236 against 0.1058, calibration 4.67 for P(win).

---

## 5. Hypotheses, and what happened to each

Verdicts are from forward selection on the development races: gain in
nats/race against the final selected set, paired t over 83 races.

| # | hypothesis | verdict | evidence |
| --- | --- | --- | --- |
| H1 | car pace wants a long window | **partly** | tuning chose *two* timescales: a 4-race team EWMA beside a season-long mean |
| H2 | qualifying pace measures the car better than race results | **both** | pre-weekend took race pace (EWMA) *and* season qualifying pace |
| H3 | history should follow the organisation | **supported** | lineage-keyed EWMA taken first; `TeamId`-keyed 5-race windows add nothing beyond it (+0.008 to −0.021) |
| H4 | driver skill = team-mate delta | **supported** | pre-weekend: removing it costs 0.61 nats (t = 4.8); post-quali: a driver term costs 0.21 (t = 2.5) |
| H4b | driver form resets over a winter | **not before quali** | season delta +0.007 (t = 0.17) pre-weekend; tied with two variants post-quali (§6) |
| H5 | some cars/drivers race better than they qualify | **not proven** | consistently positive, never significant: +0.08 to +0.14, t ≤ 1.72 |
| H6 | grid's functional form matters | **rejected** | `log_grid_position` −0.017 (t = −1.66) once stage scales exist |
| H7 | grid matters more at some circuits | **rejected, harmful** | `grid_x_circuit_retention` −0.130 (t = −1.39) |
| — | car–circuit suitability | **rejected, harmful** | `team_circuit_grid_residual_3` −0.032 (t = −2.21) post-quali |
| H8 | the DNF model improves the composite | **structurally yes, per-car barely** | §9 |
| H9 | rain reshuffles the order (race-day, retrospective only) | **rejected** | `grid_x_rain_share` −0.122 (t = −0.92), and −0.93 in the 10 wet races: too few wet races per window to estimate a stable effect |

A general lesson from H7: **anything constant within a race cancels in
Plackett-Luce.** A circuit, weather or calendar feature can only act through
an interaction with something that varies between cars — and the one tested
here was noise.

---

## 6. Feature selection

**Pre-weekend**, from 19 candidates:

| step | taken | gain | t |
| --- | --- | --- | --- |
| 1 | `team_finish_pct_ewma` | — | — |
| 2 | `driver_mate_grid_delta_ewma` | +0.569 | 4.53 |
| 3 | `team_grid_pct_season` | +0.306 | 2.82 |
| stop | best remaining `team_gain_ewma` | +0.127 | 1.72 |

Re-selected under the tuned settings: the same three.

**Post-qualifying**, from 24 candidates:

| step | taken | gain | t |
| --- | --- | --- | --- |
| 1 | `grid_position` | — | — |
| 2 | `team_rank_in_field` | +1.415 | 5.41 |
| 3 | `driver_grid_pct_ewma` | +0.214 | 2.53 |
| stop | best remaining `team_gain_ewma` | +0.139 | 1.41 |

`team_rank_in_field`, an existing DNF-era column (the team's points rank in the
field), beat every new pace feature once grid is known: it says "this car is
faster than its grid slot" — a penalty, a bad qualifying — and points
concentrate on the top of the order, which is where the objective looks.

**Selection is greedy, so it is path-dependent.** With the two driver-season
features added to the pool, step 3 became a three-way tie: season team-mate
delta +0.221, `driver_grid_pct_ewma` +0.214, EWMA team-mate delta +0.205. Fully
tuned, the configuration above scores −17.481 on the development races against
−17.521 for the season-delta version, so it is kept — a tie broken on
development data.

---

## 7. Hyperparameter tuning

Coordinate descent, two passes, every change kept only if it raised the
development objective. Nothing here was ever searched before; every factory in
`train.py` is still hand-set.

| knob | pre-weekend | post-quali | read |
| --- | --- | --- | --- |
| training window | 60 → **100** races | 60 → **100** | more data for the coefficients... |
| time-decay half-life | none → **40** races | none → **40** | ...while recent races count for more |
| coefficient prior `l2` | 1 → **0.1** | 1 → **0.1** | three features need little shrinkage |
| scale prior `l2_scale` | 1 → **0.1** | 1 → **0.1** | let the data set the position scales |
| training truncation | full → **top 10** | full → **top 10** | spend the likelihood where points are |
| team / driver half-life | 8 / 12 → **4 / 24** | driver 12 → **4** | |
| team-mate delta clip | none → **0.5** | n/a | trims a penalty grid slot |
| **total** | +0.168 nats/race | +0.134 | |

The window result is worth contrasting with the DNF model, where 40 races beat
an expanding window going into 2026. Here the long window wins because the
decay does the forgetting: the coefficients ("how much does grid matter") are
structural, while the *features* carry the recency.

---

## 8. Held-out results: 2024 to 2026 R14

**Post-qualifying** (62 races):

| model | ll10 gain | RPS@10 | points RMSE | Brier win | calib win / podium / points |
| --- | --- | --- | --- | --- | --- |
| grid order | — | 0.1051 | 4.47 | 0.0378 | 0.35 / 0.33 / 0.27 |
| grid only (PL, untuned) | 5.98 | 0.1037 | 4.86 | 0.0391 | 3.05 / 2.33 / 1.17 |
| rung 2: grid + team + driver | 7.12 | 0.0952 | 4.56 | 0.0381 | 2.21 / 1.85 / 1.11 |
| selected, untuned | 7.21 | 0.0944 | 4.51 | 0.0377 | 2.54 / 1.95 / 1.11 |
| **selected + tuned** | **8.24** | **0.0898** | **4.32** | **0.0328** | **0.81 / 0.95 / 1.05** |
| boosted ranker (tuned) | 8.00 | 0.0894 | 4.27 | 0.0330 | 0.91 / 1.00 / 1.13 |

Against grid order: RPS −0.0154, 95% CI [−0.0212, −0.0094]. Tuning alone took
RPS from 0.0944 to 0.0898 and repaired calibration on races it never saw.

**Pre-weekend** (62 races):

| model | ll10 gain | RPS@10 | points RMSE | calib win / podium / points |
| --- | --- | --- | --- | --- |
| uniform | — | 0.1774 | 7.15 | 0.39 / 0.30 / 1.33 |
| team order | — | 0.1426 | 5.87 | 0.17 / 0.24 / 0.22 |
| rung 0: team pace | 4.44 | 0.1221 | 5.58 | 2.02 / 1.95 / 1.23 |
| rung 1: + driver vs team-mate | 5.11 | 0.1160 | 5.34 | 1.82 / 1.80 / 1.21 |
| selected, untuned | 5.29 | 0.1141 | 5.27 | 2.04 / 1.97 / 1.22 |
| **selected + tuned** | **6.12** | **0.1076** | **5.03** | **0.69 / 0.99 / 1.12** |

Against team order: RPS −0.0350, CI [−0.0422, −0.0282]. The win calibration
of 0.69 means **pre-weekend win probabilities are overconfident on the test
races** — which is what a model trained on a stable order does when 2024 and
2026 moved it. §10 is the same lesson at season scale.

**The gradient-boosted ranker** (Method 3) got its own 24-draw random search,
every candidate feature, and the same stagewise calibration head. Its best
development configuration was the *most constrained* on offer (depth 2, 50
trees, learning rate 0.02) and still lost by 0.43 nats/race. On the test races
it ties: RPS difference −0.0004, CI [−0.0022, +0.0013]; ll10 −0.245 (t =
−1.51). Plackett-Luce stays — three features against twenty-four, better
points calibration, seven times faster, and the brief's rule that a flexible
family must *beat* a tuned linear one. Averaging the two is an untested option.

---

## 9. Should the DNF model be in it?

Four ways to handle a retirement, all on the tuned ordering model, test races:

| attrition | post-quali RPS | calib points | pre-weekend RPS |
| --- | --- | --- | --- |
| **DNF model, per car** | **0.0898** | **1.05** | 0.1076 |
| same rate for every car | 0.0900 | 1.21 | 0.1075 |
| nobody retires | 0.0912 | 0.67 | 0.1085 |
| retirements ranked last inside PL | 0.0914 | 0.87 | 0.1088 |

- **A separate attrition stage is required.** Ranking retirements last is
  significantly worse (post-quali +0.0016, CI [+0.0009, +0.0024]; pre-weekend
  +0.0012, CI [+0.0001, +0.0023]), and ignoring attrition wrecks points
  calibration (0.67).
- **The DNF model's per-car differentiation is small.** Post-quali it is not
  significant on RPS (flat − model +0.0003, CI [−0.0005, +0.0010]) but it
  calibrates P(points) better (1.05 vs 1.21). Pre-weekend it is
  indistinguishable from the base rate — exactly the DNF work's own finding.
- **The design brief predicted "slightly, and mostly in the back half".**
  Front half of the grid: −0.00008, CI [−0.00109, +0.00119]. Back half:
  −0.00046, CI [−0.00100, +0.00008]. Directionally right, barely.

**Verdict: keep it.** Its value is the calibrated attrition *level* — which
tracks regime changes like 2026's doubled retirement rate through its 40-race
window — and the composition `P(survive) × P(order | survivors)`, which is what
lets the ordering model be fitted on pace alone.

---

## 10. The season simulation

`src/models/season.py`. From the real standings after a round, every remaining
race is drawn whole, 10,000 times, with:

- **pre-weekend strengths** (no grid exists for an unqualified race), features
  frozen after the last completed round;
- **parameter uncertainty** from a Bayesian bootstrap: 40 refits on
  Dirichlet-weighted races, one per trial;
- **attrition** from the DNF model's pre-weekend configuration through
  `predict.build_inference_rows`, plus the non-start rate the modelling table
  cannot see; sprints at 0.47× the retirement rate;
- **season rules**: fastest-lap bonus 2019–2024, sprint scoring by year;
- drivers who have left the grid kept on their fixed totals.

**Independent races are confidently wrong.** Backtested from after rounds 6, 12
and 18 of every season 2021–2025, with each race an independent draw around
the frozen order:

| | nominal 80% interval covers | below p10 / above p90 | PIT sd (uniform 0.289) |
| --- | --- | --- | --- |
| drivers, 2021–23 | 78% | 13% / 9% | 0.311 |
| constructors, 2021–23 | 58% | 28% / 14% | 0.348 |
| drivers, 2024–25 | 55% | 23% / 22% | 0.367 |
| constructors, 2024–25 | **48%** | 25% / 27% | 0.384 |

It gave the eventual constructors' champions **0.7%** (Mercedes, 2021 after R6),
**0.0%** and **1.9%** (McLaren, 2024 after R6 and R12). Where the order held —
Red Bull 2022–23, McLaren 2025 — it was fine. The failure is specific: a
season in which **the running order moves**, which the frozen, independent
simulation cannot represent. Constructors suffer most because a team's total is
two cars sharing one car's pace, so the missing component dominates it.

**The fix is correlated pace offsets within a trial** (`draw_trial_offsets`),
with their scale tuned by this same backtest on 2021–2023 and checked on
2024–2025. That work is in progress; until it lands, treat every season
interval as too narrow.

For the record, the naive 2026 forecast after R14 (independent races — **do
not quote it**): Antonelli 84%, Russell 16%; Mercedes 100% for constructors.

---

## 11. Known limitations

**A long driver memory reaches back into a rookie season.** Antonelli lost the
2025 qualifying head-to-head to Russell 3–21 and the race head-to-head 3–17; in
2026 it is 7–7 and 8–4, with eight wins to two. The pre-weekend model's
24-race driver half-life still weights 2025, so it rates Russell faster
(Baku: Russell 32% to win, Antonelli 11%). A within-season driver term was
tested for exactly this and did not clear the bar before qualifying on
2020–2023 (+0.007, t = 0.17). The model is not tuned to fix one driver;
this is recorded instead.

**Grid position carries penalties.** The dataset has no clean qualifying
position, so a grid penalty reads as slow qualifying (Antonelli started 19th at
2026 R13 and won). The 0.5 clip trims it. The real fix is ingesting qualifying
results — about 186 session loads, a third of an hour's API budget, done once
and cached.

**The entry list is the last race's.** A mid-season swap is invisible until it
has raced once (as the DNF runbook says of its own model).

**Frozen features.** Every future race uses today's features; development
during the rest of the season is exactly what the correlated offsets must
represent.

**Sprints use the grand-prix strengths**, and the fastest-lap point (for
backtests of 2019–2024) goes to a top-ten finisher in proportion to strength —
a rough model of a noisy award worth about one point a race.

---

## 12. Running it

```bash
./.venv/bin/python -m src.models.forecast next       # the next race, pre-weekend
./.venv/bin/python -m src.models.forecast next --stage post_quali   # after qualifying
./.venv/bin/python -m src.models.forecast race 2026 14   # backtest one race
./.venv/bin/python -m src.models.forecast season     # both championships
./.venv/bin/python -m src.models.forecast evaluate   # the §8 tables
./.venv/bin/python -m src.models.forecast backtest   # the §10 table
./.venv/bin/python -m src.models.forecast tune --ranker --write   # §6, §7
```

Nothing here touches the network except the qualifying grid for
`--stage post_quali` (one call) and an optional calendar lookup (`--online`).
`points_model_guide.md` covers running and updating it in full.

---

## 13. What next

1. **Correlated season offsets**, tuned by backtest (in progress).
2. **Qualifying positions**, to separate penalties from pace (budget the loads).
3. **Record season forecasts before the races**, as `Reports/predictions.csv`
   does for retirements: a claim made when the answer did not exist.
4. **Re-run `tune` each winter.** 2026 is a new formula; nothing here is exempt
   from the DNF lesson that the best window moves with the regulations.
