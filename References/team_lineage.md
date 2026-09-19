# Formula One constructor lineage, 2006-2026

Which team on today's grid is which team of twenty years ago.

Generated from `src/data/teams.py`. Regenerate with:

```bash
python -m scripts.write_team_lineage
```


## Why this file exists

Constructors are bought and renamed far more often than they are founded.
Across the 21 seasons from 2006 to 2026 there were **25 changes of constructor name** but only **6 genuinely new teams**:
Super Aguri (2006), HRT (2010), Lotus Racing (2010), Virgin (2010), Haas (2016), Cadillac (2026).

Every team feature in `src/features/build_features.py` is a rolling history
keyed on `TeamId` -- `team_dnf_rate_10`, `team_points_rate_career`,
`team_races_to_date`. A rename resets all of them to zero. The 2021 season
opens with Aston Martin and Alpine apparently having never entered a race;
`is_new_pairing` fires for twenty untouched driver-team combinations. The
reset lands at the start of a season, where the rolling window is most
valuable, and nothing reports it, because a team with no history looks
exactly like a team that is genuinely new.


## How to use it

```python
from src.data import teams

teams.resolve("sauber", 2024).name        # 'Kick Sauber'
teams.lineage_of("Force India", 2016)     # 'aston_martin'
frame = teams.attach_lineage(frame)       # joins on TeamId + Year
```

`attach_lineage` adds two grouping keys, and choosing between them is a
modelling decision this table does not make for you:

| key | groups by | Aston Martin's history starts |
| --- | --- | --- |
| `lineage_id` | continuous **operation** -- the factory and the people | 1991, as Jordan |
| `(lineage_id, lineage_era)` | continuous **record**, split at every insolvency | 2018, as Racing Point |

The second is what the championship itself recognises, and is the
conservative choice.


## Succession grades

| grade | meaning | count |
| --- | --- | --- |
| `rebrand` | New name, unchanged entity, factory and staff. | 12 |
| `takeover` | New owner, continuing operation. | 15 |
| `reconstituted` | Continued through an insolvency as a new legal entity. **The championship restarts the constructor's record here.** | 3 |
| `new` | No predecessor. | 16 |

## Lineages

Named after the constructor currently racing them. `base` is the town the
operation actually lives in, which is the more durable identity.

| lineage | base | names | active | seasons |
| --- | --- | --- | --- | --- |
| `alpine` | Enstone, UK | Toleman -> Benetton -> Renault -> Lotus F1 -> Renault -> Alpine | yes | 1981-present |
| `aston_martin` | Silverstone, UK | Jordan -> MF1 -> Spyker -> Force India -> Racing Point -> Aston Martin | yes | 1991-present |
| `audi` | Hinwil, Switzerland | Sauber -> BMW Sauber -> Sauber -> Alfa Romeo -> Kick Sauber -> Audi | yes | 1993-present |
| `cadillac` | Fishers, USA | Cadillac | yes | 2026-present |
| `caterham` | Hingham / Leafield, UK | Lotus Racing -> Team Lotus -> Caterham | no | 2010-2014 |
| `ferrari` | Maranello, Italy | Ferrari | yes | 1950-present |
| `haas` | Kannapolis, USA | Haas | yes | 2016-present |
| `hrt` | Madrid / Murcia, Spain | HRT | no | 2010-2012 |
| `manor` | Dinnington / Banbury, UK | Virgin -> Marussia -> Manor Marussia -> Manor | no | 2010-2016 |
| `mclaren` | Woking, UK | McLaren | yes | 1966-present |
| `mercedes` | Brackley, UK | Tyrrell -> BAR -> Honda -> Brawn -> Mercedes | yes | 1970-present |
| `racing_bulls` | Faenza, Italy | Minardi -> Toro Rosso -> AlphaTauri -> RB -> Racing Bulls | yes | 1985-present |
| `red_bull` | Milton Keynes, UK | Stewart -> Jaguar -> Red Bull | yes | 1997-present |
| `super_aguri` | Leafield, UK | Super Aguri | no | 2006-2008 |
| `toyota` | Cologne, Germany | Toyota | no | 2002-2009 |
| `williams` | Grove, UK | Williams | yes | 1977-present |

## Every constructor, by lineage


### Alpine — Enstone, UK

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 1981-1985 | Toleman | `toleman` | — | `new` | 0 |
| 1986-2001 | Benetton | `benetton` | Toleman | `takeover` | 0 |
| 2002-2011 | Renault | `renault` | Benetton | `takeover` | 0 |
| 2012-2015 | Lotus F1 | `lotus_f1` | Renault | `rebrand` | 0 |
| 2016-2020 | Renault | `renault` | Lotus F1 | `takeover` | 0 |
| 2021- | Alpine | `alpine` | Renault | `rebrand` | 0 |

- **Toleman** (1981-1985) — Toleman Group Motorsport, 1981.  Based at Witney; the move to the Enstone factory the lineage is named for came in 1992.
- **Benetton** (1986-2001) — Benetton bought Toleman in 1985.  Two drivers' titles and a constructors' title in 1994-95.
- **Renault** (2002-2011) — Renault bought Benetton in March 2000; the team ran one more season under the Benetton name and became Renault F1 Team in 2002.  It was still the Renault constructor in 2011, when it raced as Lotus Renault GP under Genii Capital ownership.
- **Lotus F1** (2012-2015) — Genii-owned and branded for Group Lotus.  The name only became available when the unrelated Hingham team gave it up to become Caterham, which is why the same word means two different teams either side of 2012.
- **Renault** (2016-2020) — Renault repurchased a 90% stake in the Enstone team from Genii at the end of 2015 and restored the works name.
- **Alpine** (2021-) — Renamed for 2021 after Renault's Alpine sports-car marque.  A marketing change: same entity, same factory.

### Aston Martin — Silverstone, UK

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 1991-2005 | Jordan | `jordan` | — | `new` | 0 |
| 2006 | MF1 | `mf1` | Jordan | `takeover` | 0 |
| 2007 | Spyker | `spyker` | MF1 | `takeover` | 0 |
| 2008-2018 | Force India | `force_india` | Spyker | `takeover` | 0 |
| 2019-2020 | Racing Point | `racing_point` | Force India | `reconstituted` | 1 |
| 2021- | Aston Martin | `aston_martin` | Racing Point | `rebrand` | 1 |

- **Jordan** (1991-2005) — Eddie Jordan's team, 1991.  Four wins; the entry it established ran unbroken until the 2018 administration.
- **MF1** (2006) — Alex Shnaider's Midland Group bought Jordan in January 2005 and raced as MF1 Racing in 2006 under a Russian licence, from Jordan's Silverstone factory.  Sold on to Spyker Cars in September 2006, and renamed Spyker MF1 before the season ended.
- **Spyker** (2007) — Spyker Cars ran the team under its own name for one season.
- **Force India** (2008-2018) — A consortium led by Vijay Mallya bought the team in October 2007.  The name covers the whole of 2018, including the second half, when the rescued team raced as Racing Point Force India.
- **Racing Point** (2019-2020) — Force India entered administration in July 2018 and a consortium led by Lawrence Stroll bought the assets.  The FIA entry was not transferred: the entry Jordan established in 1991 ended, the constructor's points were struck mid-season and restarted from zero, and the 2020 Sakhir win is recorded as the team's first.  This is the one boundary in the table the championship itself refuses to carry history across.
- **Aston Martin** (2021-) — Rebranded for 2021 after Lawrence Stroll led an investment in Aston Martin Lagonda.  Same entity and staff as Racing Point.

### Audi — Hinwil, Switzerland

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 1993-2005 | Sauber | `sauber` | — | `new` | 0 |
| 2006-2010 | BMW Sauber | `bmw_sauber` | Sauber | `takeover` | 0 |
| 2011-2018 | Sauber | `sauber` | BMW Sauber | `takeover` | 0 |
| 2019-2023 | Alfa Romeo | `alfa` | Sauber | `rebrand` | 0 |
| 2024-2025 | Kick Sauber | `kick_sauber` | Alfa Romeo | `rebrand` | 0 |
| 2026- | Audi | `audi` | Kick Sauber | `takeover` | 0 |

- **Sauber** (1993-2005) — Peter Sauber's sports-car team, founded 1970, entered F1 in 1993 from Hinwil, where the operation has stayed ever since.
- **BMW Sauber** (2006-2010) — BMW bought a majority stake in 2005.  BMW withdrew after 2009 and sold back to Peter Sauber, but the 2010 entry still carried the BMW Sauber constructor name, so the name outlasts the involvement by a season.
- **Sauber** (2011-2018) — Independent again under Peter Sauber, running customer Ferrari power units.
- **Alfa Romeo** (2019-2023) — A title-sponsorship deal signed in 2018; raced as Alfa Romeo Racing and then Alfa Romeo F1 Team.  Sauber Motorsport AG remained the entrant throughout.
- **Kick Sauber** (2024-2025) — Entered as Stake F1 Team Kick Sauber when the Alfa Romeo deal ended, carrying over the Stake and Kick sponsorships.
- **Audi** (2026-) — Audi AG completed its acquisition of Sauber Motorsport in 2024 and the team races as a full works entry with Audi's own power unit from 2026.  The FIA's 2026 entry list still names Sauber Motorsport AG as the entrant company while the entity is renamed -- a reminder that legal entity, entrant name and constructor name change on three different schedules.

### Cadillac — Fishers, USA

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 2026- | Cadillac | `cadillac` | — | `new` | 0 |

- **Cadillac** (2026-) — The eleventh entry, and the first genuinely new constructor since Haas in 2016.  Bid first as Andretti Cadillac and rejected by the commercial rights holder in 2024; resubmitted under TWG Motorsports after TWG Global bought Andretti Global, and accepted for 2026.  Runs Ferrari power units, with General Motors to supply its own later.

### Caterham — Hingham / Leafield, UK

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 2010 | Lotus Racing | `lotus_racing` | — | `new` | 0 |
| 2011 | Team Lotus | `team_lotus` | Lotus Racing | `rebrand` | 0 |
| 2012-2014 | Caterham | `caterham` | Team Lotus | `rebrand` | 0 |

- **Lotus Racing** (2010) — Tony Fernandes's new team, one of three admitted for 2010 under the budget-cap rules that were then abandoned.  Unrelated to the Enstone team despite the name: it licensed 'Lotus' from Group Lotus.
- **Team Lotus** (2011) — Proton terminated the Group Lotus licence, so Fernandes bought the dormant historic Team Lotus name instead.  For 2011 only, two different teams on the grid raced as Lotus.
- **Caterham** (2012-2014) — Fernandes gave up the Lotus name -- which cleared the way for the Enstone team to take it -- and renamed after his Caterham Cars.  The team collapsed late in 2014 and its assets were auctioned in 2015.

### Ferrari — Maranello, Italy

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 1950- | Ferrari | `ferrari` | — | `new` | 0 |

- **Ferrari** (1950-) — Present at the first World Championship round in 1950 and every season since; the only constructor never to have been sold.

### HRT — Madrid / Murcia, Spain

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 2010-2012 | HRT | `hrt` | — | `new` | 0 |

- **HRT** (2010-2012) — Entered as Campos Meta, the third of the 2010 intake, and taken over by Jose Ramon Carabante before its first race to become Hispania Racing, then HRT.  Left off the 2013 entry list when no buyer was found.

### Haas — Kannapolis, USA

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 2016- | Haas | `haas` | — | `new` | 0 |

- **Haas** (2016-) — Gene Haas's team, the first new constructor since the 2010 intake and the first American entry since 1986.  Built on a customer model: Ferrari power unit and gearbox, Dallara chassis construction.

### Manor — Dinnington / Banbury, UK

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 2010-2011 | Virgin | `virgin` | — | `new` | 0 |
| 2012-2014 | Marussia | `marussia` | Virgin | `rebrand` | 0 |
| 2015 | Manor Marussia | `marussia` | Marussia | `reconstituted` | 1 |
| 2016 | Manor | `manor` | Manor Marussia | `rebrand` | 1 |

- **Virgin** (2010-2011) — Manor Motorsport's entry, branded for Virgin; the second of the 2010 intake.  Marussia Motors took a stake in 2010 and it raced as Marussia Virgin Racing in 2011.
- **Marussia** (2012-2014) — Marussia took majority control and the Virgin branding was dropped.
- **Manor Marussia** (2015) — The team entered administration in November 2014, missed the last two races of that season, and was re-established in February 2015 as a new entity, Manor Grand Prix Racing, after buying the assets.
- **Manor** (2016) — Raced as Manor Racing for 2016, then failed to secure funding in January 2017 and folded.

### McLaren — Woking, UK

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 1966- | McLaren | `mclaren` | — | `new` | 0 |

- **McLaren** (1966-) — Founded by Bruce McLaren, 1963; first entered as a constructor in 1966.  Ownership has changed hands repeatedly without the entry or the name ever lapsing.

### Mercedes — Brackley, UK

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 1970-1998 | Tyrrell | `tyrrell` | — | `new` | 0 |
| 1999-2005 | BAR | `bar` | Tyrrell | `takeover` | 0 |
| 2006-2008 | Honda | `honda` | BAR | `takeover` | 0 |
| 2009 | Brawn | `brawn` | Honda | `reconstituted` | 1 |
| 2010- | Mercedes | `mercedes` | Brawn | `takeover` | 1 |

- **Tyrrell** (1970-1998) — Ken Tyrrell's team entered F1 in 1968 running others' cars and built its own from 1970.
- **BAR** (1999-2005) — British American Tobacco bought Tyrrell in 1997 and raced as British American Racing from 1999, from a new factory at Brackley that the team has occupied ever since.
- **Honda** (2006-2008) — Honda took a 45% stake in 2004 and full control at the end of 2005, converting its engine partnership into a works team.
- **Brawn** (2009) — Honda withdrew in December 2008 and the team was days from closure.  Ross Brawn led a management buyout for a nominal GBP 1 and it raced as Brawn GP, winning both championships in its only season.
- **Mercedes** (2010-) — Daimler and Aabar Investments bought 75.1% of Brawn GP in November 2009 and rebranded it Mercedes GP, keeping the Brackley site and workforce.

### Racing Bulls — Faenza, Italy

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 1985-2005 | Minardi | `minardi` | — | `new` | 0 |
| 2006-2019 | Toro Rosso | `toro_rosso` | Minardi | `takeover` | 0 |
| 2020-2023 | AlphaTauri | `alphatauri` | Toro Rosso | `rebrand` | 0 |
| 2024 | RB | `rb` | AlphaTauri | `rebrand` | 0 |
| 2025- | Racing Bulls | `racing_bulls` | RB | `rebrand` | 0 |

- **Minardi** (1985-2005) — Giancarlo Minardi's team, founded in Faenza in 1979, entered F1 in 1985.
- **Toro Rosso** (2006-2019) — Red Bull bought Minardi from Paul Stoddart in 2005 and renamed it Scuderia Toro Rosso.  The sale required the team to stay in Faenza, where it remains.
- **AlphaTauri** (2020-2023) — Renamed on 1 December 2019 to promote Red Bull's fashion label.  Entity, factory and staff unchanged.
- **RB** (2024) — Rebranded RB for 2024, entering as Visa Cash App RB.
- **Racing Bulls** (2025-) — The RB initialism was expanded to Racing Bulls for 2025.

### Red Bull — Milton Keynes, UK

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 1997-1999 | Stewart | `stewart` | — | `new` | 0 |
| 2000-2004 | Jaguar | `jaguar` | Stewart | `takeover` | 0 |
| 2005- | Red Bull | `red_bull` | Jaguar | `takeover` | 0 |

- **Stewart** (1997-1999) — Founded by Jackie and Paul Stewart with Ford backing, 1997.
- **Jaguar** (2000-2004) — Ford bought the team outright in 1999 and rebadged it as its Jaguar marque for 2000.
- **Red Bull** (2005-) — Red Bull GmbH bought the team from Ford in November 2004, having previously been a Sauber sponsor.

### Super Aguri — Leafield, UK

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 2006-2008 | Super Aguri | `super_aguri` | — | `new` | 0 |

- **Super Aguri** (2006-2008) — Aguri Suzuki's Honda-backed team, formed at short notice for 2006 and running year-old Honda chassis.  Withdrew four races into 2008 when its funding failed.  A satellite of the Brackley team in everything but the constructor's entry, which was its own.

### Toyota — Cologne, Germany

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 2002-2009 | Toyota | `toyota` | — | `new` | 0 |

- **Toyota** (2002-2009) — A full works entry built from scratch in Cologne, and among the best-funded teams of its era.  Withdrew in November 2009 without ever winning a race.

### Williams — Grove, UK

| seasons | constructor | key | from | succession | era |
| --- | --- | --- | --- | --- | --- |
| 1977- | Williams | `williams` | — | `new` | 0 |

- **Williams** (1977-) — Williams Grand Prix Engineering, founded 1977.  Sold to Dorilton Capital in August 2020, ending the founding family's involvement, but the entity, the entry and the name all continued unbroken.

## Name changes, newest first

Each row is a point at which a `TeamId`-keyed rolling history silently
restarts.

| season | from | to | lineage | succession | record carries |
| --- | --- | --- | --- | --- | --- |
| 2026 | Kick Sauber | Audi | `audi` | `takeover` | yes |
| 2025 | RB | Racing Bulls | `racing_bulls` | `rebrand` | yes |
| 2024 | Alfa Romeo | Kick Sauber | `audi` | `rebrand` | yes |
| 2024 | AlphaTauri | RB | `racing_bulls` | `rebrand` | yes |
| 2021 | Renault | Alpine | `alpine` | `rebrand` | yes |
| 2021 | Racing Point | Aston Martin | `aston_martin` | `rebrand` | yes |
| 2020 | Toro Rosso | AlphaTauri | `racing_bulls` | `rebrand` | yes |
| 2019 | Force India | Racing Point | `aston_martin` | `reconstituted` | **no** |
| 2019 | Sauber | Alfa Romeo | `audi` | `rebrand` | yes |
| 2016 | Lotus F1 | Renault | `alpine` | `takeover` | yes |
| 2016 | Manor Marussia | Manor | `manor` | `rebrand` | yes |
| 2015 | Marussia | Manor Marussia | `manor` | `reconstituted` | **no** |
| 2012 | Renault | Lotus F1 | `alpine` | `rebrand` | yes |
| 2012 | Team Lotus | Caterham | `caterham` | `rebrand` | yes |
| 2012 | Virgin | Marussia | `manor` | `rebrand` | yes |
| 2011 | BMW Sauber | Sauber | `audi` | `takeover` | yes |
| 2011 | Lotus Racing | Team Lotus | `caterham` | `rebrand` | yes |
| 2010 | Brawn | Mercedes | `mercedes` | `takeover` | yes |
| 2009 | Honda | Brawn | `mercedes` | `reconstituted` | **no** |
| 2008 | Spyker | Force India | `aston_martin` | `takeover` | yes |
| 2007 | MF1 | Spyker | `aston_martin` | `takeover` | yes |
| 2006 | Jordan | MF1 | `aston_martin` | `takeover` | yes |
| 2006 | Sauber | BMW Sauber | `audi` | `takeover` | yes |
| 2006 | BAR | Honda | `mercedes` | `takeover` | yes |
| 2006 | Minardi | Toro Rosso | `racing_bulls` | `takeover` | yes |

## The grid, season by season

| season | n | constructors |
| --- | --- | --- |
| 2006 | 11 | BMW Sauber, Ferrari, Honda, MF1, McLaren, Red Bull, Renault, Super Aguri, Toro Rosso, Toyota, Williams |
| 2007 | 11 | BMW Sauber, Ferrari, Honda, McLaren, Red Bull, Renault, Spyker, Super Aguri, Toro Rosso, Toyota, Williams |
| 2008 | 11 | BMW Sauber, Ferrari, Force India, Honda, McLaren, Red Bull, Renault, Super Aguri, Toro Rosso, Toyota, Williams |
| 2009 | 10 | BMW Sauber, Brawn, Ferrari, Force India, McLaren, Red Bull, Renault, Toro Rosso, Toyota, Williams |
| 2010 | 12 | BMW Sauber, Ferrari, Force India, HRT, Lotus Racing, McLaren, Mercedes, Red Bull, Renault, Toro Rosso, Virgin, Williams |
| 2011 | 12 | Ferrari, Force India, HRT, McLaren, Mercedes, Red Bull, Renault, Sauber, Team Lotus, Toro Rosso, Virgin, Williams |
| 2012 | 12 | Caterham, Ferrari, Force India, HRT, Lotus F1, Marussia, McLaren, Mercedes, Red Bull, Sauber, Toro Rosso, Williams |
| 2013 | 11 | Caterham, Ferrari, Force India, Lotus F1, Marussia, McLaren, Mercedes, Red Bull, Sauber, Toro Rosso, Williams |
| 2014 | 11 | Caterham, Ferrari, Force India, Lotus F1, Marussia, McLaren, Mercedes, Red Bull, Sauber, Toro Rosso, Williams |
| 2015 | 10 | Ferrari, Force India, Lotus F1, Manor Marussia, McLaren, Mercedes, Red Bull, Sauber, Toro Rosso, Williams |
| 2016 | 11 | Ferrari, Force India, Haas, Manor, McLaren, Mercedes, Red Bull, Renault, Sauber, Toro Rosso, Williams |
| 2017 | 10 | Ferrari, Force India, Haas, McLaren, Mercedes, Red Bull, Renault, Sauber, Toro Rosso, Williams |
| 2018 | 10 | Ferrari, Force India, Haas, McLaren, Mercedes, Red Bull, Renault, Sauber, Toro Rosso, Williams |
| 2019 | 10 | Alfa Romeo, Ferrari, Haas, McLaren, Mercedes, Racing Point, Red Bull, Renault, Toro Rosso, Williams |
| 2020 | 10 | Alfa Romeo, AlphaTauri, Ferrari, Haas, McLaren, Mercedes, Racing Point, Red Bull, Renault, Williams |
| 2021 | 10 | Alfa Romeo, AlphaTauri, Alpine, Aston Martin, Ferrari, Haas, McLaren, Mercedes, Red Bull, Williams |
| 2022 | 10 | Alfa Romeo, AlphaTauri, Alpine, Aston Martin, Ferrari, Haas, McLaren, Mercedes, Red Bull, Williams |
| 2023 | 10 | Alfa Romeo, AlphaTauri, Alpine, Aston Martin, Ferrari, Haas, McLaren, Mercedes, Red Bull, Williams |
| 2024 | 10 | Alpine, Aston Martin, Ferrari, Haas, Kick Sauber, McLaren, Mercedes, RB, Red Bull, Williams |
| 2025 | 10 | Alpine, Aston Martin, Ferrari, Haas, Kick Sauber, McLaren, Mercedes, Racing Bulls, Red Bull, Williams |
| 2026 | 11 | Alpine, Aston Martin, Audi, Cadillac, Ferrari, Haas, McLaren, Mercedes, Racing Bulls, Red Bull, Williams |

## Two traps

**"Lotus" in 2011 and 2012 are different teams.** In 2011 the grid carried
both Team Lotus (Hingham, later Caterham) and Lotus Renault GP (Enstone,
later Alpine); in 2012 the Enstone team took the plain name the Hingham team
had just given up. `resolve("Lotus")` without a season raises rather than
guessing.

**An identifier can outlive the name it stood for.** The results backend has
kept a constructor id across a rebrand, so a 2024 row can arrive tagged
`sauber`. `resolve` carries such a name *forward* along its lineage, and
fills a gap inside it (`sauber` in 2008 is BMW Sauber). It never carries a
name backwards: the 2015 Enstone car was a Lotus, not an Alpine.


## Sources

Wikipedia's constructor histories, themselves sourced to the FIA entry lists
and contemporaneous reporting: *List of Formula One constructors*, *Team
Enstone*, *Sauber Motorsport*, *Racing Point F1 Team*, *Brawn GP*, *Scuderia
Toro Rosso*, *Racing Bulls*, *Marussia F1*, *Caterham F1*, *HRT Formula 1
Team*, *Cadillac in Formula One*, *Audi in Formula One*. Each entry's note
above records the specific fact and the year it happened.

