# Benchmarking the model against betting markets: a plan

Status: **plan, not yet built.** Written 2026-10-05. Nothing below is a
measurement; every number is a source's claim or an estimate, and says which.

---

## 1. The question, stated precisely

"Does the model beat the market?" hides three different questions, and they
have different answers and different costs.

| # | Question | Test | What a "yes" means |
| --- | --- | --- | --- |
| Q1 | Is the model's forecast **more accurate** than the market's? | Paired log-loss and Brier on the same events | The model could replace the market as a forecast |
| Q2 | Does the model know **anything the market does not**? | Forecast-encompassing regression; a blended forecast | The model adds information even if it loses Q1 |
| Q3 | Could you **profit** from the disagreement after costs? | Staking simulation with spreads and fees | The added information is large enough to survive frictions |

**Expect the market to win Q1, especially before qualifying.** The market
prices everything the model cannot see: upgrade packages, practice long runs,
driver news, weather forecasts, grid penalties announced on Friday. The
runbook already says the pre-weekend DNF model is "the base rate with a faint
reliability tilt"; the finishing-order model is better, but it still knows
only what happened in prior races. The model also carries a known, live
weakness the market will punish: the 24-race driver half-life rates Antonelli
below Russell (see `points_model_results.md` §11).

**Q2 is the question worth building for.** A model can lose on accuracy and
still carry independent signal, and that signal is what a blend, a bet, or a
better model would use. Q3 is a stricter version of Q2 and should come last;
a positive Q3 on ~45 races would be luck as often as skill.

---

## 2. Which markets, and what they cost

Verified against each provider's documentation on 2026-10-05.

| Source | Cost | F1 coverage | History | Verdict |
| --- | --- | --- | --- | --- |
| **Polymarket** | Free; read endpoints need no key | Race winner per GP, drivers' and constructors' champion, season head-to-heads, safety-car / red-flag props | Race-winner markets from late 2024 (Las Vegas GP market launched 2024-11-06); price history at 1-minute fidelity | **Primary source** |
| **Kalshi** | Free; "no authentication headers or API keys are required for the public market data endpoints" | Race winner (`KXF1RACE` series), drivers' champion (`KXF1`) | Candlesticks at 1 min / 1 h / 1 day, with bid/ask at close; settled markets move to `/historical/` | **Second source**; start date to confirm in Phase 0 |
| **Betfair Exchange** | Free BASIC historical tier; free delayed API key | Motor sport markets on the exchange | Since 2016, 1-minute last-traded price, no volume | **Only deep history**, but Betfair does not accept US customers, so an account may be out of reach |
| The Odds API | Free tier exists | **No Formula 1 sport key** | Historical odds on paid plans only | Not usable |
| Pinnacle, Oddschecker, OddsPortal | — | Full bookmaker coverage | — | No free public API; scraping breaches terms. Skip |
| Manifold | Free | Some F1 questions | Full | Play money, thin; optional curiosity only |

Implications:

1. **The comparison window is 2025 onward**, roughly 45 races. The model's
   held-out test period starts in 2024, so 2025-2026 is both out of sample for
   the model and covered by both prediction markets.
2. **Two independent crowds are better than one.** Polymarket (crypto-settled,
   global) and Kalshi (CFTC-regulated, US) draw different traders. Where they
   disagree, that disagreement measures the market's own uncertainty, and
   agreement between them makes "the market" a firmer benchmark.
3. **The network policy matters.** This repository's cloud sandbox blocks
   `gamma-api.polymarket.com`, `clob.polymarket.com` and the Kalshi API (all
   returned 403 at the proxy). Run pulls from a local machine, exactly as the
   fastf1 pulls already are.

---

## 3. What can be compared

Map each market onto something the model already produces.

| Market | Model output | Source in code | Events available | Power |
| --- | --- | --- | --- | --- |
| Race winner | `p_win` per driver | `order_eval.composite_scores`, `season.race_outlook` | ~45 races × ~22 drivers | **Modest**; the main test |
| Drivers' / constructors' champion | Season-sim title share | `season.simulate`, `season.summarise` | 2 seasons, priced daily | **None as a test**; descriptive only |
| Season head-to-heads ("who finishes higher") | Pairwise share from the season sim | `season.simulate` (needs a small helper) | ~10-20 pairs a season | Low; outcomes correlate through teams |
| Podium / top-10 / points | `p_podium`, `p_points` | `composite_scores` | Bookmakers only, not free | Out of scope unless a free source appears |
| Safety car, red flag | Not modelled | — | — | Out of scope |

Race winner is the comparison that can actually decide something. Treat the
championship comparison as a trajectory plot: one outcome per season cannot
test a probability.

---

## 4. Getting the timing right

A comparison is only fair when both forecasts used the same information. Fix
two snapshot times per race, and match each to a model stage:

| Snapshot | Market price taken at | Compared with |
| --- | --- | --- |
| `pre_weekend` | 1 hour before FP1 starts | `PRE_WEEKEND_SPEC` |
| `post_quali` | 30 minutes after qualifying ends | the Saturday spec (grid known) |
| `close` | 5 minutes before the race starts | `post_quali` model, as a stress test |

Rules:

- **Even the pre-FP1 snapshot favours the market.** It has seen the week's
  news; the model has seen only prior races. State this in the results rather
  than pretending the snapshots are equivalent.
- **A snapshot after lights-out is leakage.** Assert it, in the same spirit as
  `detect_target_leakage`, and refuse to score rather than warn.
- **Take the snapshot from price history, not from a re-run today.** The
  model side already comes from a race-by-race walk-forward
  (`walk_forward_order`), so every model probability predates its race.
- **Freeze these times before pulling any data.** Choosing the snapshot after
  seeing which one makes the model look best is the market-benchmark version
  of tuning on the test set.

---

## 5. Turning prices into probabilities

A prediction market lists one Yes/No contract per driver, so a race's prices
need not sum to 1.

1. **Price.** Use the bid/ask midpoint where the source gives it (Kalshi
   candlesticks do); fall back to last trade (Polymarket `prices-history`,
   Betfair BASIC). Record the spread and the time since the last trade.
2. **Normalise.** Divide by the race total. Prediction-market overrounds are
   small, so basic normalisation is defensible; log the raw total per race and
   flag any race outside [0.95, 1.15].
3. **Bookmaker odds, if added later,** carry a real margin and a
   favourite-longshot bias. Use Shin's method rather than normalisation; it
   recovers less biased probabilities (Shin 1993; Štrumbelj 2014; Clarke,
   Kovalchik & Ingram 2017).
4. **Drivers with no market.** A driver the market never listed gets the
   residual mass split evenly; a driver priced but never traded is flagged.
   Do not drop them: the model must be scored on the same field.

Run a calibration check on the market itself. Exchanges have shown a *reverse*
favourite-longshot bias in some sports (Angelini, De Angelis & Singleton 2022),
and if F1 markets misprice longshots, that is exactly where a model could add
information.

---

## 6. The tests

Everything scores per race and resamples whole races, because cars in one race
share a safety car and the weather. `order_eval.paired_race_bootstrap` already
does this and should be reused, not rewritten.

**Q1, accuracy.**

- Primary: winner log-loss, `-log p(actual winner)`, model vs market, paired
  by race. This is the multinomial proper score and matches the repository's
  preference for log-likelihood objectives.
- Secondary: Brier on `p_win` over all drivers; calibration slope via
  `train.calibration_slope`.
- Baselines in the same table: grid order (post-quali), team order
  (pre-weekend), and a uniform field, so the reader can see the scale.

**Q2, information.** The test of forecast encompassing (Fair & Shiller 1990),
in log-odds form. Per driver per race, within a conditional logit over each
race's field:

```
P(i wins) ∝ exp( a · log p_market,i  +  b · log p_model,i )
```

- `b > 0`, significantly, means the model carries information the market lacks.
- `a ≈ 1, b ≈ 0` means the market encompasses the model.
- Fit `(a, b)` by walk-forward too: estimate on races before *t*, score race
  *t*. An in-sample blend weight flatters the blend.

Report the walked-forward blend's log-loss against the market alone. That
number answers "would adding the model improve on the crowd?" directly.

Then **slice the disagreement**: by grid tier (front row, P2-P10, back), by
favourite vs longshot, by early vs late season, by regulation year (2026
power units). A model that only helps on longshots in regime-change seasons
is a real and useful finding.

**Q3, money.** Only after Q2. Stake fractional Kelly (say 0.25) whenever the
model's probability exceeds the ask by a fixed edge threshold, chosen before
looking. Charge the real costs: the spread, and Kalshi's trading fee (published
as a function of price and contracts; take the current schedule from its fee
page). Report return, maximum drawdown, and a bootstrap interval on return.
Expect that interval to cross zero.

### Power, honestly

A rough estimate: if per-race log-loss differences have a standard deviation
near 0.5 nats, detecting a 0.1-nat gap at *t* ≈ 2 needs about
`(2 × 0.5 / 0.1)² ≈ 100` races. With ~45, only a gap above ~0.15 nats will
register. Compute the actual standard deviation in Phase 3 and publish the
minimum detectable effect beside every result, so "no significant difference"
is read as "too few races", not "equal".

---

## 7. Where it lives in the code

Follow the repository's existing rules rather than inventing new ones.

```
src/data/markets.py         the second module allowed to touch the network; Polymarket,
                            Kalshi, (Betfair file loader). Raw JSON cached under
                            Data/raw/markets/<source>/, never re-fetched once settled
src/data/market_names.py    market outcome -> driver_id, market event -> (year, round);
                            an explicit table, Unidecode-normalised, unmatched names raise
src/models/market_eval.py   snapshot selection, normalisation, Q1/Q2/Q3 scoring
src/models/forecast.py      new subcommand: `forecast markets`
tests/test_markets.py       synthetic price histories; no network
Reports/market_log.csv      committed: one row per driver per race per snapshot,
                            model and market probability side by side
```

Guards, by analogy with the ones that already exist:

- **Name matching fails loudly.** "Kimi Antonelli", "Andrea Kimi Antonelli"
  and "A. K. Antonelli" all appear across sources. An unmatched name raises,
  like `DegradedResultsError`, rather than silently dropping a driver.
- **Coverage check.** A race whose normalised market omits a driver who
  started, or whose raw total sits outside the flagged range, is reported
  before scoring.
- **No market data in features.** The market is a benchmark, never an input.
  Feeding market prices to the model would make Q2 circular, and it would make
  the model useless on the races the market prices thinly.
- **Budget the calls.** About 45 races × 22 drivers × 2 sources is ~2,000
  history requests for the backfill. Settled markets cannot change, so cache
  them permanently and fetch only the live race each week, as `--since` does
  for results.
- **CLAUDE.md update.** Amend the "`ingest.py` is the ONLY module that touches
  the network" rule to name `markets.py`, and say why.

---

## 8. Phases and decision gates

| Phase | Work | Gate to continue |
| --- | --- | --- |
| **0. Spike** (an evening) | By hand, pull three 2025 races and one 2026 race from Polymarket and Kalshi. Confirm first-market dates, liquidity, name formats, and whether price history reaches back to the pre-FP1 snapshot | Both sources cover ≥ 30 races with priced fields → continue. Otherwise narrow to one source |
| **1. Pre-register** | Commit this document's snapshot times, metrics, edge threshold and hypotheses *before* the backfill | A commit hash exists that predates the data |
| **2. Ingest** | `markets.py`, `market_names.py`, cache, tests | Every 2025-26 race maps; tests pass offline |
| **3. Q1** | Winner log-loss and Brier, both stages, both sources, with baselines and the minimum detectable effect | — (any result is a result) |
| **4. Q2** | Walked-forward encompassing and blend; slices | If `b` is indistinguishable from zero everywhere, stop before Q3 and write up |
| **5. Q3** | Staking simulation with costs | — |
| **6. Prospective log** | Each Saturday, write market snapshot and model probability to `market_log.csv` before the race, as `predictions.csv` already does | Runs unattended for the rest of 2026 |
| **7. Write-up** | `References/market_benchmark_results.md`, in the style of `points_model_results.md`: hypotheses, protocol, every deciding number, and what failed | — |

Phase 6 matters more than it looks. A backfill can be done carefully and still
be questioned; a prediction logged before the race, beside the market price at
that moment, cannot be revised once the answer exists. It is the same reason
`refresh` scores before refitting.

---

## 9. Hypotheses, written down before the data

1. The market beats the model on winner log-loss before the weekend, by more
   than the minimum detectable effect.
2. After qualifying, the gap narrows, because grid position closes most of the
   information gap.
3. The model nonetheless carries some independent information (`b > 0`), most
   likely on longshots and early in the 2026 regulation change, where the
   market has the least history to lean on.
4. No staking strategy clears zero with confidence after costs.

Being wrong on any of these is informative. Being wrong on 3 in the
pessimistic direction says the model's value is as a component of a points and
championship forecast, which markets do not price race by race, rather than as
a winner forecast.

---

## 10. Sources

- Polymarket, "Get prices history", API documentation:
  <https://docs.polymarket.com/api-reference/markets/get-prices-history>
- Polymarket, Las Vegas Grand Prix Winner market (launch date):
  <https://polymarket.com/event/las-vegas-grand-prix-winner>
- Polymarket, F1 props (market types): <https://polymarket.com/sports/f1/props>
- Kalshi, "Quick Start: Market Data":
  <https://docs.kalshi.com/getting_started/quick_start_market_data>
- Kalshi, "Historical Data": <https://docs.kalshi.com/getting_started/historical_data>
- Kalshi, Bahrain GP race-winner market (`KXF1RACE`):
  <https://kalshi.com/markets/kxf1race/f1-race/kxf1race-bah26>
- Betfair Data Scientists, "Historic Data Site":
  <https://betfair-datascientists.github.io/data/usingHistoricDataSite/>
- Betfair Developer Program, Exchange API FAQs:
  <https://developer.betfair.com/en/exchange-api/faq/>
- The Odds API, sports list (no Formula 1):
  <https://the-odds-api.com/sports-odds-data/sports-apis.html>; historical
  data on paid plans only: <https://the-odds-api.com/historical-odds-data/>
- Shin, H. S. (1993). Measuring the incidence of insider trading in a market
  for state-contingent claims. *Economic Journal*, 103(420), 1141-1153.
- Štrumbelj, E. (2014). On determining probability forecasts from betting
  odds. *International Journal of Forecasting*, 30(4), 934-943.
- Clarke, S., Kovalchik, S., & Ingram, M. (2017). Adjusting bookmaker's odds
  to allow for overround. *American Journal of Sports Science*, 5(6), 45-49.
- Angelini, G., De Angelis, L., & Singleton, C. (2022). Informational
  efficiency and behaviour within in-play prediction markets. *International
  Journal of Forecasting*, 38(1), 282-299.
- Snowberg, E., & Wolfers, J. (2010). Explaining the favorite-longshot bias:
  Is it risk-love or misperceptions? *Journal of Political Economy*, 118(4),
  723-746.
- Fair, R. C., & Shiller, R. J. (1990). Comparing information in forecasts
  from econometric models. *American Economic Review*, 80(3), 375-389.
