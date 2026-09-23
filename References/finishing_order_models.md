# Models for the finishing order

Three ways to put a probability distribution over *who finishes where*, each
one compatible with the pipeline in `src/`, and the reasons to prefer one over
another.

This is a literature and design document. **Nothing in it has been run against
our data** — no numbers below are measurements of our model, and every claim
about what will work here is a hypothesis with a named way to test it. Read it
alongside `References/points_model_design.md`, which sets out the target and
the feature ladder; this document is the §5 that brief gestured at, worked out
properly and argued rather than asserted.

---

## 1. The requirement, stated precisely

The thing being asked for has four parts, and they are not the same part:

1. **Coherence.** A race result is a *permutation*. If Norris finishes first
   then nobody else did. Twenty independent per-driver models cannot express
   this: they will assign P1 to two drivers, P7 to nobody, and expected
   positions that do not sum to 1+2+…+20.
2. **Probability, not a ranking.** A sorted list of predicted positions is not
   a forecast. The output has to be a distribution over orderings from which
   `P(Verstappen podium)`, `P(Hamilton beats Russell)` and
   `E[points]` all fall out as integrals of the same object.
3. **Composability into a season.** The per-race distribution has to be
   *sampleable*, because the season forecast is a Monte Carlo loop over it.
   A model with a beautiful closed-form likelihood and no cheap sampler is
   useless for the actual goal.
4. **Scoreability.** Whatever is built must be comparable, in the same
   walk-forward, against the trivial baselines — grid order, team order, the
   Rung 0 base rate. The DNF work established that this is not a formality:
   a 101-feature random forest lost to one column, and nobody had checked.

Requirement 3 is the binding one, and it eliminates more candidates than
requirement 1 does. Plenty of models produce coherent orderings; far fewer
let you draw ten thousand seasons in under a minute.

### What the sport actually pays

```
Grand prix:  1st=25, 2nd=18, 3rd=15, 4th=12, 5th=10,
             6th=8, 7th=6, 8th=4, 9th=2, 10th=1, 11th and back = 0
Sprint:      1st=8, 2nd=7, ... 8th=1, 9th and back = 0
```

One correction to the points map in `points_model_design.md`, which hedges
with "+1 for fastest lap where the rules grant it". That hedge can now be
resolved, and it matters because our dataset spans the change: **the fastest-lap
bonus point existed from 2019 to 2024 inclusive, and the FIA removed it for
2025**. It did not exist in 2018 either. So over the 2018-2026 window the points
map is time-varying in a way a single hard-coded table will get wrong at both
ends. Key the map on season.

The cliff at P10 is the dominant feature of this function and it should drive
model choice, not just scoring. Half the grid is competing for a payoff that is
identically zero, and the difference between tenth and eleventh is the
difference between a point and nothing. Any method that spends its likelihood
evenly across all twenty positions is spending most of it on a region where the
answer does not matter.

---

## 2. One frame, not a menu

The three methods below are not independent inventions. They are all
**random-utility models**: give each driver a latent quantity, and let the
finishing order be the sort of that quantity.

```
latent race performance    U_i  =  f(x_i)  +  e_i
observed finishing order   =  argsort(U)
```

Everything is a choice about `e_i`, and that single choice determines the whole
character of the model:

| distribution of `e_i` | resulting model | closed-form order probability? |
| --- | --- | --- |
| Gumbel, i.i.d. | **Plackett–Luce** / rank-ordered logit | yes — a product of softmaxes |
| Normal, i.i.d. | Thurstone–Mosteller (ordered probit) | no |
| Normal, correlated / unequal variance | multinomial probit | no |
| Gamma(shape *r*), scale from `f(x_i)` | **Stern's family**, interpolating the two | only at *r*=1 |

This is a theorem, not an analogy. [Yellott (1977)](https://doi.org/10.1016/0022-2496(77)90026-8)
proved that i.i.d. Gumbel errors are the *only* choice that yields Luce's
choice axiom, and [Stern (1990)](https://doi.org/10.1080/01621459.1990.10476235)
showed that ranking *k* independent gamma variables with common shape *r*
recovers Plackett–Luce at *r*=1 and approaches Thurstone as *r*→∞. Equivalently,
in racing terms: [Harville (1973)](https://doi.org/10.1080/01621459.1973.10482425)
showed that independent exponential finishing times give
`P(car j wins) = λ_j / Σλ_i` and a closed form for the full order, which
[Fry, Brighton and Fanzon (2024)](https://arxiv.org/abs/2312.14637) label
*time-rank duality* and apply directly to Formula 1.

Why this frame earns its place here: it tells you **what you are buying and
what you are paying** at each step, instead of leaving you with three
unrelated options and a coin to flip. Gumbel buys you a tractable likelihood
and a two-line sampler, and pays for it with independence of irrelevant
alternatives. Normal buys you correlation and heteroscedasticity, and pays for
it with simulation-based fitting. Gamma buys you a one-parameter dial between
the two.

---

## 3. The retirement problem, which is shared and comes first

Every method below has to answer the same question, and the answer is not
obvious. Getting it wrong contaminates everything downstream, so settle it
before choosing a likelihood.

**A retirement is not a twentieth-place finish.** A driver who stops on lap 3
while running second has told you almost nothing about their pace and a great
deal about their car's reliability. There are three treatments:

**(a) Score DNFs as last.** Wrong, and wrong in a specific direction: it drags
the estimated strength of fast-but-fragile cars toward the back of the grid.
It also double-counts reliability, once in the ordering model and again in the
DNF model that multiplies it — which is precisely the error our architecture
exists to avoid.

**(b) Drop non-finishers and model the order among survivors.** This is what
[van Kesteren and Bergkamp (2023)](https://doi.org/10.1515/jqas-2022-0021) do,
removing 590 of 3,267 driver-races. It is *not* a hack, and for us it is very
likely the right answer, for a reason specific to this repository: **we already
have a separately calibrated model of `P(dnf)`.** The generative story

```
P(order) = P(who survives)  ×  P(order | survivors)
```

factorises exactly, and treatment (b) estimates the second factor on precisely
the population it describes. The two pieces compose without double-counting.

The cost is real and the authors are candid about it: the strength parameter
now means *pace conditional on finishing*, and nothing else. Their own
sensitivity analysis is the cautionary tale — with non-finishers excluded,
Pastor Maldonado, a driver whose defining characteristic was crashing, ranks
6th of all drivers in the era. That is not a bug in their model; it is the
correct answer to a different question than the one you probably meant. If you
adopt (b), you must say so every time you report a strength, and you must never
let a "driver ranking" derived from it escape without the DNF model attached.

**(c) Censor at the point of retirement.** The statistically honest version,
and available to us in a way it was not to van Kesteren: we hold lap-by-lap
position data in the fastf1 cache. A driver who retires from second on lap 3
did beat everyone who was behind them at that moment, and that is genuine
information the other treatments discard. In the likelihood this is an ordinary
right-censored observation.

**Recommendation:** start with (b), because it composes cleanly with the DNF
model we already trust and costs nothing to implement. Treat (c) as the first
upgrade once the harness works, and measure it — the prize is the driver-races
that (b) throws away, which by the attrition figures in `CLAUDE.md` runs
between 10.3% (2024) and 19.7% (2026) of every row. That is not nothing when
the training window is 40 races, and it is largest exactly when the formula is
changing and history is least informative.

One detail the labels already handle and the ordering model must respect:
`dnf_classified` marks cars that retired past 90% distance and *kept an
official position*. Those rows carry a real `Position` that is a genuine
ordering observation, and a `dnf` of 1. Do not let treatment (b) silently drop
them — they belong in the ordering likelihood even though they belong in the
DNF model's positive class too. `dnf_strict` is the flag for a row with no
usable position.

---

## 4. Method 1 — Plackett–Luce, truncated at the points cliff

### What it is

Give driver *i* a strength `θ_i = exp(β·x_i)`. The probability that *i* wins is
`θ_i / Σθ_j`; remove them and repeat for second place, and so on. For an
observed order `i_1 ≻ i_2 ≻ … ≻ i_m`:

```
                 m
P(order)  =  ∏  ─────────────────────────
                k=1   exp(β·x_{i_k})
                    ─────────────────────
                      Σ_{t≥k} exp(β·x_{i_t})
```

The model is due jointly to [Luce (1959)](https://archive.org/details/individualchoice0000luce)
and [Plackett (1975)](https://doi.org/10.2307/2346567); econometrics calls the
same object the rank-ordered or "exploded" logit.

### Why it fits here

- **The likelihood is a product of softmaxes over a linear score**, so it is an
  ordinary regression. Every feature in `src/features/registry.py` goes in
  unchanged, with its `Stage` tag intact. No new feature machinery.
- **The log-likelihood is concave** in β under stated conditions
  ([Dong, Han, Jiang and Xu, 2025](https://arxiv.org/abs/2406.16507), who give
  necessary and sufficient conditions for existence and uniqueness of the MLE
  in exactly our setting — covariates that vary across comparisons, which is
  what `grid_position` is). One optimum, no restarts, no seeds.
- **Sampling is trivial and exact.** Draw a Gumbel per driver, add it to
  `β·x_i`, sort. That is one line of numpy and it makes §7 tractable.
- **It is the measured-best family for this sport.** Henderson and Kirrane
  ([2018](https://doi.org/10.1214/17-BA1048)) built probabilistic F1 forecasts
  on it; van Kesteren and Bergkamp (2023) used a Bayesian multilevel version and
  found it beat the linear-regression-on-position approach that preceded it.

### The fitting trick worth knowing

You do not need to write a custom likelihood. [Allison and Christakis (1994)](https://doi.org/10.2307/270983)
showed that the rank-ordered logit has **the same partial likelihood as a Cox
proportional hazards model**, stratified by comparison set. Map it as:

```
duration  = finishing position       (1 = first across the line)
event     = 1 for a finisher, 0 for a censored retirement
strata    = race id  -> our RACE_KEYS = ("Year", "RoundNumber")
covariates= the registry's feature columns for the stage
```

The Cox risk set at position *k* is exactly the Plackett–Luce denominator, and
the sign works out so that a larger `β·x` means a shorter duration, i.e. a
better finish — so `exp(β·x_i)` is the Plackett–Luce strength directly.
`lifelines.CoxPHFitter(strata=...)` or `scikit-survival` will both fit it, and
both hand you standard errors as a side effect. Neither is currently in
`requirements.txt`; adding one is a smaller change than writing and testing a
bespoke optimiser, though `scipy.optimize.minimize` on the concave
log-likelihood is perhaps fifteen lines if you would rather not take the
dependency.

**One caution, because the convenience invites overclaiming.** The censoring
*mechanism* comes free; the censoring *value* does not. Retirements excluded
from the risk set altogether give you treatment (b). Censoring them at a large
duration — "after all finishers" — quietly gives you treatment (a), because
such a row sits in every denominator and no numerator, which is the likelihood
asserting that everyone who finished beat them. Treatment (c) means censoring
at the position the car was running when it stopped, and that encodes a real
assumption: that a retirement carries no information about pace beyond the
position held at the time. For a blown engine that is reasonable; for a driver
limping to the pits with damage, or one who crashed because they were already
struggling, it is not. The machinery will not tell you which you have. Pick the
encoding deliberately and write down why.

For the pure no-covariate case, `choix` implements Hunter's
([2004](https://doi.org/10.1214/aos/1079120141)) MM algorithm and the I-LSR
spectral method of [Maystre and Grossglauser (2015)](https://papers.nips.cc/paper/2015/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html).
Useful for a sanity check; not sufficient, because we need covariates.

### Truncate it

This is the adaptation the points cliff demands, and it is the part most likely
to be skipped. **Model the top *k* of the order exactly and lump the rest.**
The likelihood keeps the first *k* softmax factors and stops:

```
             k
P(top-k)  =  ∏  exp(β·x_{i_j}) / Σ_{t≥j} exp(β·x_{i_t})
            j=1
```

Set *k*=10 and the likelihood is concentrated exactly where points are decided.
This is not an approximation invented here — it is the "truncated" half of
Henderson and Kirrane's title, and the standard *top-n* partial ranking in the
rankings literature (see [Turner, van Etten, Firth and Kosmidis (2020)](https://doi.org/10.1007/s00180-020-00959-3)
for the distinction between top-*n* and subset rankings, which imply different
denominators and which most software supports only one of). It also happens to
be robust to the part of the data we trust least: the gap between P17 and P18 is
frequently decided by a late pit stop for a fastest-lap attempt that no longer
exists, or by a car nursing a problem to the flag.

### What to be sceptical about

**Independence of irrelevant alternatives, and it is not a technicality.** The
model asserts that the odds of Piastri beating Norris do not depend on who else
is in the race. In Formula 1 that is false in a specific and consequential way:
team-mates share machinery, strategy calls, and a pit crew, so their outcomes
are correlated in a manner PL cannot represent.

The practical consequence is documented and it is the tail, not the head. In
horse racing — the same model, applied for fifty years — **the Harville/PL form
reproduces win probabilities well and systematically misprices place and show**.
The finding is usually traced to Bacon-Shone, Lo and Busche (1992), who report
that Henery's normal and Stern's gamma formulations fit better than Harville's
exponential precisely in the tail; it is a research report rather than a
journal paper, and the result is restated in the survey literature and in Lo
and Bacon-Shone's later work. Treat the direction of the bias as well
established and the exact magnitude as something to re-measure on our own
data rather than inherit. That bias
lands on us badly: our points map runs to *tenth*, and the season forecast
compounds the whole order over twenty-plus races. A model that is right about
who wins and wrong about who finishes eighth is wrong about the constructors'
championship.

So: adopt PL first because it is tractable, concave, and cheap to sample — but
adopt it knowing exactly which number it is expected to get wrong, and put the
diagnostic in from the start. **Bin the predicted P(top-3) and P(points) and
plot observed against predicted**, which is what `app/common.py:calibration_bins`
already does for DNF. If the top-*k* calibration slope comes back materially
below 1, Method 2 is the answer and you will already have the evidence.

---

## 5. Method 2 — Latent pace with correlation (Thurstone / Stern family)

### What it is

Keep the random-utility frame and change the error distribution:

```
U_i  ~  Normal(mu_i, sigma_i^2)        mu_i = beta · x_i
order = argsort(U)                     with Cov(U_i, U_j) != 0 allowed
```

Three things become expressible that Plackett–Luce forbids:

1. **Unequal variance.** A rookie in a midfield car is genuinely more variable
   than Verstappen in a Red Bull. Under PL every driver carries the same
   Gumbel spread and the only way to be unpredictable is to be mediocre.
2. **Correlation.** Team-mates, shared power units, and everyone who chose the
   same tyre strategy are correlated. A block-diagonal `Σ` with a per-team
   term says so directly.
3. **A tunable tail.** Stern's shape parameter *r* is one number that dials
   from PL (*r*=1) toward the normal limit, and it is exactly the knob that
   fixes the place-and-show mispricing described above. Fitting one extra
   scalar is a very cheap way to buy back a known bias.

### Why it fits here

Sampling is *still* trivial — draw from a multivariate normal and `argsort` —
so §7 remains tractable. That is the property that matters most, and this
method keeps it.

What it costs is the likelihood. `P(order)` is an orthant probability with no
closed form, so fitting needs one of:

- **Simulated maximum likelihood.** Draw *S* latent vectors per race, count the
  fraction ordering correctly. Simple, unbiased, slow, and the gradient is
  noisy.
- **Composite / pairwise likelihood.** Replace the full order with all
  `C(20,2) = 190` pairwise comparisons per race, each a probit. Consistent,
  hugely cheaper, and it gives up some efficiency. This is the pragmatic route
  and it is what I would try first.
- **MCMC.** The notebook track already runs PyMC and arviz, so the tooling
  exists on that side — though note that `requirements.txt` does not currently
  list PyMC, so the dependency is informal and would need to be made real
  before this touches `src/`.

Bayesian Thurstonian models for exactly this data shape are well-trodden, with
JAGS and Stan implementations in the psychometrics literature going back to
[Thurstone (1927)](https://doi.org/10.1037/h0070288). The machinery is not
exotic; it is just more expensive than a softmax.

### Where it pays and where it does not

It pays if and only if the residual correlation is real and large. The honest
test is cheap and should be run *before* building any of this: fit Method 1,
take the residuals, and check whether team-mates' residuals correlate. If
that correlation is near zero, Method 2 is a great deal of machinery for
nothing, and the project's own history says to expect exactly that outcome
more often than feels reasonable.

A middle path is worth naming, because it gets most of the benefit for almost
none of the cost: **fit Method 1, then estimate Stern's single shape parameter
*r* by maximising the truncated likelihood over a grid**. One extra scalar, no
new inference machinery, and it directly targets the documented failure mode.
If *r* comes back near 1, Plackett–Luce was right and you have the evidence to
say so. That is a genuinely informative negative result and it costs an
afternoon.

---

## 6. Method 3 — Gradient-boosted ranking with a calibrated softmax head

### What it is

Two stages, deliberately separated:

```
stage 1   s_i = f(x_i)          f = gradient-boosted trees, listwise ranking loss
stage 2   theta_i = exp(alpha * s_i)     alpha fitted by 1-D maximisation of the
                                         Plackett-Luce log-likelihood, held out
```

Stage 1 learns the *ordering* with a flexible non-linear function. Stage 2
turns scores into a probability distribution by fitting a single temperature.
The output is a Plackett–Luce model whose log-strength is a tree ensemble rather
than a linear form — so it inherits PL's exact sampler and its coherence, while
relaxing the linearity.

### Why it fits here

- **`xgboost` is already a dependency** and already a member of
  `MODEL_FACTORIES`. Its `rank:pairwise` and `rank:ndcg` objectives take a
  `qid` group vector, and `RACE_KEYS = ("Year", "RoundNumber")` is that group
  key with no transformation needed.
- **It captures interactions for free.** "Grid position matters more at Monaco
  than Monza" is a product term a linear model must be told about and a tree
  finds on its own. Given that circuit overtaking difficulty is Rung 3 of the feature
  ladder, this is not a hypothetical benefit.
- The theory is not a mismatch: ListNet's top-one probability
  ([Cao, Qin, Liu, Tsai and Li, 2007](https://doi.org/10.1145/1273496.1273513))
  *is* the Plackett–Luce first factor, and LambdaMART
  ([Burges, 2010](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/))
  is the standard engine for it. Learning-to-rank and rank-ordered logit are the
  same subject with two vocabularies.

### Why I would not start here

Three reasons, and the first is the serious one.

**This repository has already run this experiment on the adjacent problem and
lost.** From `CLAUDE.md`: a random forest on the full feature set calibrates at
0.606 and gradient boosting at 0.212, against 0.915 for a logistic regression
on one column. Tree ensembles turn ordinal inputs into step functions, and the
single most important input here — grid position — is ordinal and monotone.
The stage-2 temperature repairs the *global* scale of the scores; it cannot
repair a score function that is locally flat in the wrong places. Expect the
same failure and plan to measure it.

**Ranking metrics are not the objective.** NDCG rewards getting the top of the
list right, which sounds aligned with the points cliff and is not the same
thing. NDCG's discount is logarithmic and the points map is not; optimising one
and reporting the other is how you end up with a model that is excellent at
something nobody asked for. If you use this method, use a listwise loss that
*is* the truncated PL likelihood, rather than an off-the-shelf ranking
objective that merely resembles it.

**Hyperparameters.** `points_model_design.md` is explicit that every value in
every factory is hand-set and unsearched, and that this is defensible only
while the model has one feature and one effective degree of freedom. A boosted
ranker has none of that protection. Adopting Method 3 means adopting a nested
walk-forward search as well, or the scores are not honest.

Method 3 is the right *third* thing to try. It is the wrong first thing,
because if it wins you will not know whether it won on flexibility or on the
extra tuning budget nobody gave the baseline.

---

## 7. What to reject, and why it is tempting

**Per-driver ordinal regression** (proportional odds over the 20 positions) is
the obvious approach, it is what the F1 literature reaches for
([Weissbock and Mills, 2025](https://arxiv.org/abs/2507.10966) use ordinal
logistic on 7,800 driver-weekends to confirm qualifying dominates), and it fails requirement 1 outright. Twenty
separate marginals over positions do not form a permutation. Two drivers get
assigned P1; the expected positions do not sum to 210.

It can be repaired — project the matrix of marginals onto the doubly-stochastic
set with Sinkhorn iterations, then sample a permutation — but you have now built
something with no generative story, no likelihood, and no principled way to
propagate parameter uncertainty into the season simulation of §9. It is
strictly more machinery than Plackett–Luce for strictly less. Skip it.

The same verdict applies, for the same reason, to **regressing on finishing
position directly** and sorting the predictions. It produces an ordering but
never a distribution, so it cannot answer the question that motivated the whole
exercise.

---

## 8. Scoring: how you will know

The metrics in `src/models/train.py` are binary-classification metrics and do
not transfer. A ranking model needs its own set, and the same discipline.

| metric | what it answers | notes |
| --- | --- | --- |
| **Ranked probability score** | is the position distribution right? | The proper scoring rule for ordered categories (Epstein 1969; Murphy 1971 — see references). Squared error on the *cumulative* distribution, so a near-miss is punished less than a far one. |
| **Log-likelihood of the observed order** | is the joint right? | The model's own objective, out of sample. Compare against a uniform-permutation null. |
| **Expected-points error** | is the decision variable right? | The one that actually matters. Pool it, do not average it. |
| **Top-*k* calibration slope** | can you multiply these probabilities? | Reuse `app/common.py:calibration_bins`. This is the diagnostic that decides between Methods 1 and 2. |
| **Pairwise accuracy** | a sanity check | Fraction of driver pairs ordered correctly. Easy to read, easy to over-trust. |

Three rules carry over from the DNF work unchanged, and all three were learned
the expensive way:

- **Pool, do not average.** The mean of per-race skill flipped the sign of the
  headline number last time. Reconstruct totals, take the ratio once.
- **Score before you refit.** The validity of `Reports/model_log.csv` rests
  entirely on this ordering.
- **Beat the trivial model or do not ship.** For this problem the baselines are
  named and non-negotiable: *finishing order = grid order*, *finishing order =
  team-strength order*, and the Rung 0 team base rate. Grid order will be
  hard to beat. Fry et al.'s stepwise regression of finishing position on
  constructor dummies plus a second-driver flag reached R² = 0.39 on the 2022
  season — most of the order, from team identity and nothing else. Our own DNF
  result — one ordinal column beating a hundred engineered features — is the
  same lesson in a different costume.

---

## 9. The season simulation

Once a race is a sampler, a season is a loop:

```
for trial in 1..10_000:
    draw beta from its posterior (or a race-bootstrap of the fit)
    draw a per-trial team-strength offset and reliability offset
    standings = current points
    for race in remaining races:
        survivors = [i for i in grid if not Bernoulli(p_i_dnf)]
        order     = sample_PL(theta[survivors])      # Gumbel + argsort
        standings += points_map(season)(order)
    record final standings
```

Four failure modes, all of which produce a confident and wrong answer:

**Independent draws understate the variance, badly.** A car fast in Baku is
fast in Singapore. Without a per-trial offset carried across every race in the
trial, the championship intervals come out far too narrow and the model will
look certain about things it cannot know. This is the single most likely way
the season forecast ends up embarrassing.

**Outcome noise is not parameter uncertainty.** Re-draw β per trial. Otherwise
the simulation answers "given that my coefficients are exactly right, what is
the spread" — a question nobody asked. A race-level bootstrap of the fit is
the cheap version and requires no Bayesian machinery; `race_bootstrap` in
`train.py` already resamples whole races rather than rows, which is the correct
unit, because a first-lap pile-up is one event and not four.

**The points map is time-varying.** See §1. A simulator with 2024's fastest-lap
point hard-coded will quietly inflate every projection.

**Validate the marginals before believing the joint.** Run the simulator over a
completed season and check that simulated season-point totals reproduce the
observed distribution. If they do not, the joint is wrong regardless of how
good the per-race scores look. This check is cheap and it is the one that
catches the correlation error above.

---

## 10. Where I would start

In order, smallest first. Each step is a day or two and each produces a number
that decides the next one.

1. **Build the ordering harness before any model.** `walk_forward_races` scores
   a binary target one row at a time and cannot be reused as-is; a ranking model
   trains and scores *by race*. The good news is that the group key already
   exists — `RACE_KEYS` — and `registry.py`, `store.py` and the leakage guards
   carry over untouched. Write `walk_forward_races_ranked` as a sibling, not a
   modification.
2. **Fit Plackett–Luce on `grid_position` alone**, finishers only, truncated at
   *k*=10. This is the Method 1 minimum and it is perhaps thirty lines via the
   Cox route. Score it against grid order. If it does not beat grid order,
   stop and fix the harness — the model is not the problem.
3. **Add team strength and the driver-versus-team-mate delta** (Rungs 0 and 1).
   Measure. Expect the gain to be smaller than it feels like it should be.
4. **Attach the DNF model as the survival stage** and measure whether the
   composite beats the ordering model alone. On the §1 numbers of the design
   brief it should help slightly, mostly in the back half of the grid, and it
   is worth knowing by how much before the season simulator depends on it.
5. **Check top-*k* calibration.** This is the decision point. Slope near 1
   means ship Method 1 and move to the season simulator. Slope materially below
   1 means fit Stern's *r* — the cheap middle path in §5 — before reaching for
   correlated normals.
6. **Then, and only then, the season simulation.**

Methods 2 and 3 are upgrades with entry criteria, not alternatives to choose
between today. Method 2 earns its place when the calibration diagnostic fails.
Method 3 earns its place when the linear score is demonstrably the binding
constraint — and it should have to beat a tuned Method 1, not an untuned one.

---

## 11. The critique I would make of this plan

Three things worth holding in mind, offered because the alternative is finding
them out later.

**The ceiling here is lower than it looks.** Grid position explains most of
finishing position, and grid position is *given to you* on Saturday. It is
entirely possible that Plackett–Luce on grid alone lands close to the
achievable maximum and every subsequent rung buys a percent. The DNF work
produced exactly this shape of result and it was the most valuable finding in
the project. Decide now that a flat ablation table is a publishable answer, so
that when it arrives it does not read as failure.

**Conditioning on finishing changes what a "driver rating" means, permanently.**
Under treatment (b) the strength parameter measures pace given that the car
lasted. Any leaderboard derived from it is a leaderboard of conditional pace,
and it will rank a fast crasher implausibly high — that is the Maldonado
result, and it is a property of the estimand, not a bug to patch. Write that
caveat into the docstring on day one, because it will otherwise escape into a
dashboard and be quoted back at you.

**The season simulation is where the errors compound and where they are hardest
to see.** A per-race model that is 2% miscalibrated is fine; the same model run
forward over fifteen races with independent draws will produce championship
probabilities that are confidently wrong, and nothing in the per-race scores
will warn you. The marginal-validation check in §9 is not optional polish. It
is the only thing standing between a working race model and a season forecast
that is wrong in a way nobody notices.

---

## References

**Foundational — the models themselves**

- Luce, R. D. (1959). *Individual Choice Behavior: A Theoretical Analysis.* Wiley.
- Plackett, R. L. (1975). "The Analysis of Permutations." *JRSS Series C*, 24(2), 193-202. https://doi.org/10.2307/2346567
- Thurstone, L. L. (1927). "A Law of Comparative Judgment." *Psychological Review*, 34(4), 273-286. https://doi.org/10.1037/h0070288
- Yellott, J. I. (1977). "The relationship between Luce's choice axiom, Thurstone's theory of comparative judgment, and the double exponential distribution." *Journal of Mathematical Psychology*, 15(2), 109-144. https://doi.org/10.1016/0022-2496(77)90026-8
- Bradley, R. A. and Terry, M. E. (1952). "Rank Analysis of Incomplete Block Designs." *Biometrika*, 39(3/4), 324-345. https://doi.org/10.2307/2334029

**Racing and multi-entry competitions**

- Harville, D. A. (1973). "Assigning Probabilities to the Outcomes of Multi-Entry Competitions." *JASA*, 68(342), 312-316. https://doi.org/10.1080/01621459.1973.10482425
- Stern, H. (1990). "Models for Distributions on Permutations." *JASA*, 85(410), 558-564. https://doi.org/10.1080/01621459.1990.10476235
- Henery, R. J. (1981). "Permutation Probabilities as Models for Horse Races." *JRSS Series B*, 43(1), 86-91. https://doi.org/10.1111/j.2517-6161.1981.tb01153.x
- Lo, V. S. Y. and Bacon-Shone, J. (1994/2008). "Approximating the ordering probabilities of multi-entry competitions by a simple method." *Management Science*. https://doi.org/10.1287/mnsc.1080.0893
- Bacon-Shone, J., Lo, V. S. Y. and Busche, K. (1992). "Logistic analyses of complicated bets." Research Report, Department of Statistics, University of Hong Kong. — the usual citation for Henery and Stern out-fitting Harville in the tail. A research report, not peer-reviewed; the result is restated in the racing-models survey literature.

**Formula 1 specifically**

- Henderson, D. A. and Kirrane, L. J. (2018). "A Comparison of Truncated and Time-Weighted Plackett-Luce Models for Probabilistic Forecasting of Formula One Results." *Bayesian Analysis*, 13(2), 335-358. https://doi.org/10.1214/17-BA1048 — the closest prior art to what we are building. Code: https://github.com/d-a-henderson/F1
- van Kesteren, E.-J. and Bergkamp, T. (2023). "Bayesian analysis of Formula One race results: disentangling driver skill and constructor advantage." *Journal of Quantitative Analysis in Sports*, 19(4), 273-293. https://doi.org/10.1515/jqas-2022-0021 (preprint: https://arxiv.org/abs/2203.08489)
- Fry, J., Brighton, T. and Fanzon, S. (2024). "Faster identification of faster Formula 1 drivers via time-rank duality." https://arxiv.org/abs/2312.14637
- Bell, A., Smith, J., Sabel, C. E. and Jones, K. (2016). "Formula for success: Multilevel modelling of Formula One driver and constructor performance, 1950-2014." *JQAS*, 12(2), 99-112. https://doi.org/10.1515/jqas-2015-0050
- Weissbock, J. and Mills, S. (2025). "Evaluating the Predictive Power of Qualifying Performance in Formula One Grand Prix." https://arxiv.org/abs/2507.10966 — ordinal logistic over ~7,800 driver-weekends; useful as evidence on qualifying, not as a model to copy (see §7).
- Eichenberger, R. and Stadelmann, D. (2009). "Who is the best Formula 1 driver? An economic approach to evaluating talent." *Economic Analysis and Policy*, 39(3), 398-409.

**Estimation and software**

- Hunter, D. R. (2004). "MM algorithms for generalized Bradley-Terry models." *Annals of Statistics*, 32(1), 384-406. https://doi.org/10.1214/aos/1079120141
- Allison, P. D. and Christakis, N. A. (1994). "Logit Models for Sets of Ranked Items." *Sociological Methodology*, 24, 199-228. https://doi.org/10.2307/270983 — the Cox partial-likelihood equivalence.
- Turner, H. L., van Etten, J., Firth, D. and Kosmidis, I. (2020). "Modelling rankings in R: the PlackettLuce package." *Computational Statistics*, 35, 1027-1057. https://doi.org/10.1007/s00180-020-00959-3 (preprint: https://arxiv.org/abs/1810.12068)
- Maystre, L. and Grossglauser, M. (2015). "Fast and Accurate Inference of Plackett-Luce Models." *NIPS 2015*. Python: `choix`, https://pypi.org/project/choix/
- Dong, P., Han, R., Jiang, B. and Xu, Y. (2025). "Statistical ranking with dynamic covariates." https://arxiv.org/abs/2406.16507 — identifiability and MLE existence when covariates vary per comparison.
- Glickman, M. E. and Hennessy, J. (2015). "A stochastic rank ordered logit model for rating multi-competitor games and sports." *JQAS*, 11(3), 131-144. https://doi.org/10.1515/jqas-2015-0012

**Learning to rank**

- Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F. and Li, H. (2007). "Learning to rank: from pairwise approach to listwise approach." *ICML 2007*. https://doi.org/10.1145/1273496.1273513
- Burges, C. J. C. (2010). "From RankNet to LambdaRank to LambdaMART: An Overview." Microsoft Research Technical Report MSR-TR-2010-82.

**Scoring**

- Epstein, E. S. (1969). "A Scoring System for Probability Forecasts of Ranked Categories." *Journal of Applied Meteorology*, 8(6), 985-987. DOI `10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2` (given as plain text: the angle brackets break markdown links).
- Murphy, A. H. (1971). "A Note on the Ranked Probability Score." *Journal of Applied Meteorology*, 10(1), 155-156. DOI `10.1175/1520-0450(1971)010<0155:ANVPOT>2.0.CO;2`

**In this repository**

- `References/points_model_design.md` — the target, the feature ladder, the lessons.
- `References/model_runbook.md` — operational assumptions and failure modes.
- `CLAUDE.md` — the rules that hold across the pipeline.
