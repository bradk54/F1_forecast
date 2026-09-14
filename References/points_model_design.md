# Designing the points model

Where the project is, what the DNF work actually taught, and how to build the
thing it was always a component of: a probabilistic forecast of where each
driver finishes, per race and across a season.

Nothing here is built yet. It is a design brief, written while the evidence was
fresh.

---

## 1. The model as it stands

The DNF model, in full. There is no more to it than this:

```
logit P(dnf) = -1.8646 + 0.5252 · z(grid_position)

    z = (grid_position - 10.841) / 5.999      # StandardScaler, refit each race
    missing grid -> median (11.0)             # SimpleImputer
```

| grid | P(dnf) |
| --- | --- |
| P1 | 0.061 |
| P5 | 0.085 |
| P10 | 0.126 |
| P15 | 0.182 |
| P20 | 0.257 |
| P22 | 0.292 |

**Pipeline:** `ColumnTransformer([median impute → standard scale])` →
`LogisticRegression`.

**Hyperparameters:** `C=1.0`, `penalty="l2"`, `solver="lbfgs"`, `max_iter=2000`,
`class_weight=None`, `random_state=42`. Refit before every race on a sliding
window of the last **40 races** (~817 rows).

**None of these were tuned, and there is no hyperparameter search anywhere in
the repository.** Every value in every factory in `MODEL_FACTORIES` is
hand-set. That is defensible here and worth stating rather than hiding: with one
feature and 817 rows, an L2 logistic has essentially one degree of freedom, `C`
does almost nothing, and a search would be fitting noise. It stops being
defensible the moment the feature count goes up — see §4.

The other factories carry hand-set values too, unsearched:
`gradient_boosting` (`max_iter=300, learning_rate=0.05, min_samples_leaf=25`),
`random_forest` (`n_estimators=500, min_samples_leaf=15`), `xgboost`
(`n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
reg_lambda=1.0`).

### What it is worth

Honestly: not much on its own, and that is the point of §2.

| | AUC | Brier skill | calibration |
| --- | --- | --- | --- |
| this model | 0.593 | +0.0147 | 0.915 |
| the 101-feature forest it replaced | 0.565 | +0.0057 | 0.606 |
| base rate | — | 0.000 | — |

Retirements cost about **10.2%** of the notional points pool. **81%** of those
forgone points come from P1–10 starters, while only **37%** of retirements do —
so the model is weakest precisely where points are decided (AUC among top-10
starters: **0.553**, i.e. chance). Its job in the points pipeline is to supply a
*calibrated multiplier*, not a ranking, and at a calibration slope of 0.92 it
does that adequately.

---

## 2. Lessons learned

Written down because each one cost something to find.

**Fit the trivial model first, and make it the thing you have to beat.** A
101-feature random forest was beaten on every metric by a logistic regression on
one column — and by *raw grid position as a sort order* on AUC. Nobody had
checked. Any new feature set should be reported against (a) the base rate and
(b) the simplest single-feature model, in the same walk-forward, or the number
means nothing.

**Decide whether you need ranking or calibration, because they are different
jobs and the same model is rarely good at both.** A random forest on one ordinal
column ranks acceptably (AUC 0.555) and calibrates terribly (0.137) because
trees turn a continuous input into a step function. A logistic on 101 correlated
columns ranks acceptably (0.539) and calibrates catastrophically (0.030). For
anything that gets multiplied into an expected value, calibration is the
constraint.

**Model family and feature set are not independent choices.** The matrix in §1
is not separable — the best family changes with the number of features. Re-run
the whole grid when changing either.

**Pool, do not average.** The mean of per-race Brier skill is −0.0139 where the
pooled figure is +0.0002: averaging flips the sign of the headline. A third of
races have no retirements at all, and those punish any positive prediction while
leaving AUC undefined. Reconstruct the totals and take the ratio once.

**Score before you refit.** A model scored against data it has just trained on
measures nothing. The entire validity of `Reports/model_log.csv` rests on this
ordering, which is why `refresh` does it in that order and why it is called out
in three separate places.

**Nothing is stationary.** Attrition ran 19.5% (2018), 10.3% (2024), 19.6%
(2026). A sliding 40-race window beat an expanding one going into the regime
change and *lost* on the settled seasons either side. Re-check the window each
season; do not treat a tuned constant as a finding.

**A successful load can return useless data.** A rate-limited Ergast call
returns a full grid with a blank `Status`, which the labeller would read as an
all-retirement race. Every ingest path needs a "did this actually answer"
check distinct from "did this raise".

**Guards fire because a failure looks like a success.** A partial pull is
indistinguishable from a good one by every check except the one built to catch
it. `--allow-shrink` exists for a loss you can name; using it to get *past* a
guard converts a refused write into real data loss. This was learned the
expensive way.

**Size the prize before optimising.** Ten percent of the points pool was the
ceiling on all DNF work, and most of that ten percent sits in the subset the
model handles worst. Compute the bound first.

**Separate "when is this knowable" from "should the model use it".** The
registry answered the first and was quietly assumed to answer the second.
`FEATURE_SETS` is the second axis, and `grid_only` being *empty* at
`pre_weekend` is information, not an edge case.

---

## 3. The target

Points, not position — but position is the natural latent variable, because
points are a deterministic, non-linear function of it.

```
finishing position -> points:  1st=25, 2nd=18, 3rd=15, 4th=12, 5th=10,
                               6th=8, 7th=6, 8th=4, 9th=2, 10th=1, else 0
                               (+1 for fastest lap where the rules grant it)
```

Two consequences worth internalising:

**The mapping is a cliff, so the interesting probability mass is concentrated.**
The difference between P10 and P11 is the difference between a point and
nothing; the difference between P15 and P16 is nothing at all. A model optimised
for mean absolute position error will happily buy accuracy at P15 with accuracy
at P10, which is exactly backwards. Score on points, or on a ranked probability
score weighted toward the top ten.

**Regressing on points directly is a bad idea.** The distribution is
zero-inflated (over half of all driver-races score nothing), bounded, discrete
and violently non-linear in the latent quantity. Model position — or better,
model the *ordering* — and push it through the points map afterwards.

---

## 4. Feature creation, per race

### The discipline that already exists, and is not optional

Everything in `src/features/` applies unchanged to a points model:

- **Declare every feature in `src/features/registry.py` with its `Stage`.**
  `audit_coverage` fails the build for anything present but unregistered. Add
  features there, never ad hoc.
- **Rolling features must be strictly prior-race.** Use `prior_rolling` /
  `prior_expanding`; `detect_target_leakage` runs before every write and exits 3
  if it fires. A points model has more ways to leak than a DNF model — a
  championship-standings feature computed after the race is the obvious one.
- **Walk forward, refitting before each race.** Season-level splits flatter
  early rounds and penalise late ones.
- **Sprints inform history but are not modelling rows.** A sprint's points map
  is different and its attrition process is different.

### The ladder — build it in this order, and stop when it stops paying

This is the structure the project should follow, crudest first. Each rung must
beat the one below it in the same walk-forward, or it does not ship.

**Rung 0 — team base rate (pre-weekend).** Expected finishing position = the
team's mean finishing position over the last *N* races. Crude, no driver
adjustment, no circuit adjustment. This is the number everything else has to
beat, and on the DNF evidence it will be more competitive than feels reasonable.

**Rung 1 — driver within team (pre-weekend).** Add the driver's mean deviation
from their team-mate. This is the cleanest available estimate of driver skill,
because the team-mate comparison controls for the car — by far the largest
source of variance in F1 — and it is the one comparison the sport makes
naturally.

**Rung 2 — grid position (post-qualifying).** The regime change. On the DNF
evidence grid slot is the dominant single signal, and for finishing position it
will be dramatically stronger: most races finish close to grid order. Model
**position change from grid**, not position, so the model learns the deviation
rather than re-learning the grid.

**Rung 3 — circuit overtaking difficulty.** Position change has very different
variance at Monaco than at Monza. This is already computable from the circuit
profiles: the historical standard deviation of `finish − grid` per
`(circuit_key, year)` is a single strong feature, and the measured geometry in
`src/features/track_profile.py` lets it generalise to a circuit never raced —
which Madrid in 2026 showed is a real requirement, not a hypothetical.

**Rung 4 — race-day conditions.** Rain especially: `rain_share` and `any_rain`
already exist. These are `race_day` stage, so they are **retrospective only**
and must never be reported as forecast accuracy. Useful for quantifying how much
of the residual is weather; useless for Saturday.

### What to be sceptical of

The DNF work is a warning. A hundred plausible engineered features made the
model worse than one obvious one. Before adding anything, ask what it adds *over
grid position and team strength*, and be ready for the answer to be nothing.

Specifically doubtful on current evidence: long rolling windows (the sport moves
faster than they do), interaction terms with no mechanism behind them, and
anything derived from `dnf_cause` (the Jolpica backend returns a bare `Retired`
for every retirement from 2023 on, so those columns are only populated for
2018–2022 and are diagnostics, not features).

---

## 5. The probabilistic model

### Why the obvious approach fails

Predicting each driver's finishing position independently produces incoherent
results: two drivers assigned P1, nobody assigned P7, expected positions that do
not sum to 1+2+…+20. You cannot integrate that into a championship forecast,
because the thing you want — P(driver X wins the title) — depends on the joint
distribution, not twenty marginals.

**A race result is a permutation.** The model has to respect that.

### The generative structure

This composes cleanly with the DNF model, which is the argument for having built
it first:

```
for each race:
    1. attrition      for each driver i:  DNF_i ~ Bernoulli(p_i)
                      p_i from the DNF model, already calibrated

    2. ordering       among the survivors, draw a finishing order from a
                      Plackett-Luce model with strengths θ_i

    3. points         map the order through the points table
                      retirements score nothing (unless classified)
```

**Plackett-Luce** is the right choice and worth understanding rather than
adopting on faith. It says: give each driver a positive strength θ_i; the
probability that driver *i* finishes first is `θ_i / Σ θ_j`; then remove them and
repeat for second place, and so on. Properties that matter here:

- It is a proper distribution over **orderings**, so the permutation constraint
  holds by construction.
- It is a **softmax over a linear score**, so `log θ_i = β·x_i` makes it an
  ordinary regression problem — the features from §4 go straight in.
- It reduces to a ranking loss, so it optimises what you care about (who beats
  whom) rather than squared position error.
- It is cheap to sample, which is what makes §6 tractable.

Its main limitation is the independence-of-irrelevant-alternatives assumption:
it has no notion of "these two are hard to separate because they are team-mates
on the same tyre strategy". Live with it initially; if it bites, the standard
escape is a mixture or a rank-dependent extension.

### Fitting it

`log θ_i = β · x_i` where `x_i` is the feature vector from §4. The likelihood of
an observed finishing order is the product of the softmax terms down the order,
and it is differentiable, so any optimiser will do. Fit it on the same sliding
window and the same walk-forward as everything else.

**Handle retirements explicitly rather than treating a DNF as "last".** A driver
who retires on lap 3 while running second is not evidence that their strength is
low. Censor them: their contribution to the likelihood is that they beat nobody
they were still ahead of when they stopped, and nothing more. Getting this wrong
will systematically understate the strength of unreliable-but-fast cars, which
is precisely the population the DNF model already struggles with.

### Scoring it

Do not reach for accuracy. Use:

- **Ranked probability score** over the finishing-position distribution — the
  natural proper scoring rule for ordered outcomes.
- **Log-likelihood of the observed order** under the fitted Plackett-Luce.
- **Expected-points error**, since that is the decision variable.
- **Calibration of the top-*k* probabilities**: when the model says 70% for a
  podium, does it happen 70% of the time? Bin and plot, exactly as
  `app/common.py:calibration_bins` already does for DNF.

Baselines to beat, in the same walk-forward: finishing order = grid order;
finishing order = team-strength order; and the Rung 0 team base rate.

---

## 6. The season

Once a race is a sampler, a season is a loop over it.

```
for trial in 1..10_000:
    standings = current points
    for race in remaining races:
        draw a result from §5            # attrition, then ordering, then points
        standings += points
    record final standings

→ P(driver X wins the title), P(top three), full distribution over final points
```

Four things that will otherwise be got wrong:

**Correlate the draws.** Independent per-race sampling understates the variance
of the final standings badly. A car that is fast in Baku is fast in Singapore;
a team that is unreliable in September is unreliable in October. Carry a
per-trial team-strength offset and a per-trial reliability offset, drawn once at
the top of the trial and applied to every race in it. Without this the
championship intervals will be far too narrow and the model will look confident
about things it should not be.

**Propagate parameter uncertainty, not just outcome noise.** Re-draw β from its
posterior (or bootstrap the fit across trials). Otherwise the simulation says
"given that my coefficients are exactly right, here is the spread", which is the
wrong question.

**Model the regime, not just the race.** `regulation_era` exists because 2026
was a genuine break. A season simulation that assumes September's car order
holds through December is making a claim, and it should be one you made
deliberately.

**Sanity-check the marginals before believing the joint.** Simulated
season-total points per driver should reproduce the observed distribution when
run over a completed season. If they do not, the joint is wrong no matter how
good the per-race scores look.

---

## 7. Where to start

In order, smallest first:

1. Build **Rung 0** and score it. It is an afternoon, and it sets the bar.
2. Build **Rung 2** (position change from grid, post-qualifying) and confirm it
   beats Rung 0 by a wide margin. If it does not, something is wrong with the
   harness rather than the model.
3. Only then fit a **Plackett-Luce** on those same features and check it beats
   the point-estimate model on ranked probability score. The distribution is
   what makes §6 possible, but it should still have to earn its place.
4. Add the **DNF model as a multiplier**, and measure whether it helps. On the
   §1 numbers it will help slightly and mostly in the back half of the grid.
5. Then, and only then, the **season simulation**.

The existing harness carries over unchanged: `walk_forward_races` for the
evaluation, `Reports/model_log.csv` for the weekly record, the `app/` dashboard
for reading it, and the leakage and coverage guards for keeping it honest.
