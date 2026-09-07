# Data dictionary: retirement (DNF) prediction dataset

Generated from `src/features/registry.py`. Regenerate with:

```bash
python -m scripts.write_data_dictionary
```


## The target

| column | definition |
| --- | --- |
| `dnf` | **Primary target.** 1 when the car stopped before the end of the race. |
| `dnf_strict` | 1 when the car retired *and* was left unclassified. |
| `dnf_classified` | 1 when the car retired but had covered enough distance to keep a position. |
| `dnf_cause` | `finished` / `mechanical` / `collision` / `driver_error` / `disqualified` / `withdrawn` / `other` |
| `finished_on_track` | 1 when the car took the chequered flag under its own power. |
| `started` | 1 when the driver took the start. Rows with 0 are excluded from the dataset. |
| `classified` | 1 when `ClassifiedPosition` holds an integer. |

Two decisions worth knowing about:

1. **A driver who retires past 90% distance is still a DNF.** They keep an official
   position, so `ClassifiedPosition` alone would score it a finish and systematically
   under-count retirements. `dnf_classified` flags exactly these rows.
2. **A disqualification after finishing is not a DNF.** A DSQ is a scrutineering
   outcome applied to a car that usually completed the race. `dnf_cause` still records
   it, so the rows stay findable.


## Feature stages

Each feature is tagged with when it becomes knowable. A model may use its own stage
and every earlier one.

| stage | available | use for |
| --- | --- | --- |
| `pre_weekend` | Monday before the race | season simulation, early forecasting |
| `post_quali` | Saturday evening | the strongest honest forecast |
| `race_day` | **after the race** | retrospective analysis only — never a forecast |

> Observed weather is `race_day`. Including it in a model and reporting the result as
> forecast accuracy would be wrong; it is kept so you can measure how much of
> retirement risk weather explains.


## `pre_weekend` features (99)

| feature | kind | description |
| --- | --- | --- |
| `driver_dnf_rate_5` | numeric | Driver's retirement rate over their previous 5 races. |
| `driver_dnf_rate_10` | numeric | Driver's retirement rate over their previous 10 races. |
| `driver_dnf_rate_career` | numeric | Driver's retirement rate over every prior race in the dataset. |
| `driver_mech_dnf_rate_10` | numeric | Share of the driver's previous 10 races ending in a car failure. |
| `driver_incident_dnf_rate_10` | numeric | Share of the driver's previous 10 races ending in a collision or spin. |
| `driver_races_since_dnf` | numeric | Races since the driver last retired; NaN before their first retirement. |
| `driver_races_to_date` | numeric | Experience: races started before this one. |
| `driver_points_rate_5` | numeric | Mean points over the previous 5 races; largely a proxy for car pace. |
| `driver_avg_grid_5` | numeric | Mean starting position over the previous 5 races. |
| `team_dnf_rate_5` | numeric | Team's per-car retirement rate over its previous 5 races. |
| `team_dnf_rate_10` | numeric | Team's per-car retirement rate over its previous 10 races. |
| `team_mech_dnf_rate_10` | numeric | Team's per-car mechanical retirement rate over its previous 10 races. |
| `team_dnf_rate_career` | numeric | Team's per-car retirement rate over every prior race. |
| `teammate_dnf_rate_10` | numeric | The other car's own prior-race retirement rate. Two cars share a power unit and a design office, so the sister car carries information this driver's record does not. |
| `team_races_this_season` | numeric | Races the team has completed this season; low values mean a car still early in its reliability shakedown. |
| `team_points_rate_5` | numeric | Team's mean points per race over its previous 5 races. |
| `team_points_rate_10` | numeric | Team's mean points per race over its previous 10 races. |
| `team_points_rate_career` | numeric | Team's mean points per race over every prior race in the dataset. |
| `team_form_delta` | numeric | team_points_rate_5 minus team_points_rate_career; positive when the team is scoring above its own historical rate, which is what a working upgrade looks like before it reaches the standings. |
| `team_avg_finish_5` | numeric | Team's mean finishing position over its previous 5 races, taken over finishers only (a retirement has no position). |
| `team_best_finish_5` | numeric | Team's best finishing position over its previous 5 races. |
| `team_avg_grid_5` | numeric | Team's mean grid position over its previous 5 races: single-lap pace, the cleanest available proxy for car speed. |
| `team_avg_grid_10` | numeric | Team's mean grid position over its previous 10 races. |
| `pair_races_to_date` | numeric | Races this driver has started for this team. |
| `is_new_pairing` | binary | 1 for the first five races of a driver-team combination. |
| `pair_dnf_rate_10` | numeric | Retirement rate of this driver-team pairing over its previous 10 races. |
| `circuit_dnf_rate_prior` | numeric | Field-wide retirement rate at this circuit across all prior visits. |
| `circuit_dnf_rate_3` | numeric | Field-wide retirement rate over the circuit's previous 3 visits. |
| `driver_dnf_rate_at_circuit` | numeric | This driver's retirement rate at this circuit, prior visits only. |
| `driver_starts_at_circuit` | numeric | How many times the driver has started here before. |
| `track_speed_index` | numeric | Composite of mean speed, high-speed share and full-throttle share, z-scored across circuits. Monza high, Monaco low. |
| `mechanical_stress_index` | numeric | Composite of full-throttle share, braking-zone density, peak deceleration and gear changes. Hypothesised driver of car failures. |
| `incident_exposure_index` | numeric | Composite of corner density, low-speed share and corner radius. Hypothesised driver of collision and driver-error retirements. |
| `speed_mean_kph` | numeric | Mean speed around a reference lap, distance-weighted. |
| `pct_full_throttle` | numeric | Share of lap distance at 95% throttle or more; power-unit load. |
| `pct_dist_above_250kph` | numeric | Share of lap distance above 250 km/h. |
| `pct_dist_below_120kph` | numeric | Share of lap distance below 120 km/h. |
| `corners_per_km` | numeric | Corner count divided by lap length. |
| `braking_zones_per_km` | numeric | Discrete braking zones per kilometre; brake and gearbox load. |
| `decel_g_p95` | numeric | 95th-percentile braking deceleration in g. |
| `lat_g_mean` | numeric | Mean lateral acceleration in g, from v-squared times curvature. |
| `median_corner_radius_m` | numeric | Median radius of the non-straight parts of the lap. |
| `pct_dist_straight` | numeric | Share of lap distance with a radius above 500 m. |
| `elevation_range_m` | numeric | Highest point minus lowest point around the lap. |
| `curvature_asymmetry` | numeric | How one-directional the circuit is, from 0 (balanced) to 1 (all one way); drives one-sided tyre and brake load. |
| `n_gear_changes` | numeric | Gear changes per lap; transmission duty cycle. |
| `lap_length_m` | numeric | Lap distance in metres. |
| `speed_median_kph` | numeric | Median lap speed. |
| `speed_min_kph` | numeric | Slowest point of the lap; the speed of the tightest corner. |
| `speed_max_kph` | numeric | Top speed reached on the lap. |
| `speed_p10_kph` | numeric | 10th-percentile lap speed. |
| `speed_p90_kph` | numeric | 90th-percentile lap speed. |
| `speed_std_kph` | numeric | Speed dispersion; high on stop-start circuits, low on flowing ones. |
| `speed_range_kph` | numeric | Top speed minus slowest speed. |
| `pct_braking` | numeric | Share of lap distance with the brakes applied. |
| `pct_throttle_lift` | numeric | Share of lap distance on partial throttle. |
| `pct_drs_open` | numeric | Share of lap distance with DRS open; a proxy for usable straight. |
| `braking_m_per_lap` | numeric | Metres per lap spent braking; total brake energy proxy. |
| `n_braking_zones` | numeric | Discrete braking zones per lap. |
| `n_corners` | numeric | Corner count, from the official corner table where available. |
| `decel_g_mean` | numeric | Mean braking deceleration in g. |
| `decel_g_max` | numeric | Peak braking deceleration in g. |
| `lat_g_p95` | numeric | 95th-percentile lateral acceleration. |
| `lat_g_max` | numeric | Peak lateral acceleration in g. |
| `mean_abs_curvature_1pm` | numeric | Mean absolute curvature per metre around the lap. |
| `min_corner_radius_m` | numeric | Tightest radius on the lap; the hairpin. |
| `elevation_gain_m` | numeric | Total climb around a lap. |
| `max_gradient_pct` | numeric | Steepest gradient on the lap. |
| `gear_mean` | numeric | Mean gear held around the lap. |
| `rpm_mean` | numeric | Mean engine speed. |
| `rpm_p95` | numeric | 95th-percentile engine speed. |
| `driver_mech_dnf_rate_5` | numeric | Driver's mechanical retirement rate over their previous 5 races. |
| `driver_incident_dnf_rate_5` | numeric | Driver's incident retirement rate over their previous 5 races. |
| `team_mech_dnf_rate_5` | numeric | Team's mechanical retirement rate over its previous 5 races. |
| `team_races_to_date` | numeric | Races the team has entered before this one. |
| `circuit_races_prior` | numeric | Times this circuit has appeared in the dataset before. |
| `is_street_circuit` | binary | Public roads with permanent walls. Carries the runoff a position trace cannot see. |
| `is_limited_runoff` | binary | Street or hybrid circuit; limited room for a recoverable mistake. |
| `is_night_race` | binary | Run under floodlights, on a cooler surface. |
| `is_high_altitude` | binary | Above 1,500 m, where thin air strains turbo and cooling. |
| `season_round` | numeric | Round number within the season. |
| `is_season_opener` | binary | First race of the season, when new-car reliability is at its worst. |
| `field_size` | numeric | Cars taking the start. |
| `days_since_last_race` | numeric | Days since this driver's previous race. |
| `regulation_era` | categorical | Technical-regulation era; 2022 onward is the ground-effect ruleset. |
| `field_dnf_rate_last_3` | numeric | Mean retirement rate across the whole grid over the previous 3 races. |
| `field_dnf_rate_last_5` | numeric | Mean retirement rate across the whole grid over the previous 5 races. One number per race, shared by every driver in it. |
| `field_dnf_rate_last_10` | numeric | Mean retirement rate across the whole grid over the previous 10 races. |
| `field_dnf_rate_ewma` | numeric | Exponentially weighted grid retirement rate, 3-race half-life, so last weekend counts for roughly four times a race six weekends ago. |
| `driver_dnf_rate_3` | numeric | Driver's retirement rate over their previous 3 races. |
| `driver_dnf_ewma` | numeric | Exponentially weighted driver retirement rate, 3-race half-life. |
| `driver_dnf_rate_season` | numeric | Driver's retirement rate so far this season only.  Resets at the winter break: a new car is a new reliability question. |
| `team_dnf_rate_3` | numeric | Team's retirement rate over its previous 3 races. |
| `team_dnf_ewma` | numeric | Exponentially weighted team retirement rate, 3-race half-life. |
| `field_dnf_rate_mean` | numeric | Mean of every starter's driver_dnf_rate_10 for this race: how fragile this particular field is on recent form. |
| `driver_dnf_rate_vs_field` | numeric | Driver's 10-race retirement rate minus the field mean; a 20% rate means something different on a fragile grid than on a robust one. |
| `field_pace_spread` | numeric | Standard deviation of team_points_rate_5 across the grid.  A compressed field races closer together, which is where contact comes from. |
| `team_rank_in_field` | numeric | Team's percentile rank by team_points_rate_5 within this race, 0 for the fastest car.  Relative pace travels across eras where the raw points rate does not. |
| `season_progress` | numeric | Round number as a fraction of the season's rounds.  Late-season cars are developed and understood; early-season ones are neither. |

## `post_quali` features (7)

| feature | kind | description |
| --- | --- | --- |
| `teammate_grid_delta` | numeric | This driver's grid slot minus the other car's, negative when ahead.  The team mate is the only genuine control for car quality in the sport: same machinery, same weekend. |
| `driver_grid_vs_teammate_5` | numeric | Rolling mean of teammate_grid_delta over the driver's previous 5 races: much closer to a measure of the driver than raw grid position, which mostly measures the car. |
| `grid_position` | numeric | Starting position, with a pit-lane start moved to the back of the grid. |
| `grid_position_pct` | numeric | Grid position as a fraction of field size, so seasons with different entry counts are comparable. |
| `is_back_half_of_grid` | binary | Starting in the slower half of the field. |
| `starts_from_pit_lane` | binary | Started from the pit lane rather than a grid slot. |
| `grid_penalty_places` | numeric | Places lost between qualifying and the grid, usually a component penalty — which is itself a reliability signal. |

## `race_day` features (7)

| feature | kind | description |
| --- | --- | --- |
| `air_temp_c_mean` | numeric | Mean air temperature. |
| `track_temp_c_mean` | numeric | Mean track temperature. |
| `track_temp_c_max` | numeric | Peak track temperature. |
| `humidity_pct_mean` | numeric | Mean relative humidity. |
| `wind_speed_ms_mean` | numeric | Mean wind speed. |
| `rain_share` | numeric | Share of weather samples reporting rainfall. |
| `any_rain` | binary | Any rainfall recorded during the race. |

## Leakage guarantee

Every history feature is built by shifting within its entity before aggregating, so
no feature can see the race it describes. This is enforced, not asserted:
`src.features.build_features.detect_target_leakage` flips one event's outcomes,
rebuilds the whole feature table, and reports any feature that moved at or before
that event. The dataset build runs it and refuses to write if anything is found.

The detector is itself tested against deliberately planted leaks — a `cumsum` with no
shift, and a same-race team aggregate — so a silent pass means something.

