"""The market benchmark: names, snapshots, normalisation and the three tests.

Nothing here touches the network.  ``get_json`` is replaced where a fetch is
exercised, and price histories are built by hand so the answer is known.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import market_names, markets
from src.models import market_eval

UTC = "UTC"


# --------------------------------------------------------------------------- #
# Names and dates
# --------------------------------------------------------------------------- #

FIELD = pd.DataFrame({
    "DriverId": ["max_verstappen", "norris", "antonelli", "hulkenberg", "sainz"],
    "FullName": ["Max Verstappen", "Lando Norris", "Andrea Kimi Antonelli",
                 "Nico Hülkenberg", "Carlos Sainz"],
})


def test_names_resolve_however_the_listing_typed_them() -> None:
    got = market_names.resolve_outcomes(
        ["Verstappen", "Lando Norris", "Kimi Antonelli", "A. K. Antonelli", "Hulkenberg",
         "Other", "Driver C"], FIELD)
    assert got["Verstappen"] == "max_verstappen"
    assert got["Lando Norris"] == "norris"
    assert got["Kimi Antonelli"] == got["A. K. Antonelli"] == "antonelli"
    assert got["Hulkenberg"] == "hulkenberg"
    assert got["Other"] is None and got["Driver C"] is None


def test_an_unmatched_name_raises_and_reports_every_miss() -> None:
    with pytest.raises(market_names.MarketNameError) as err:
        market_names.resolve_outcomes(["Norris", "Palou", "Rossi"], FIELD)
    assert "Palou" in str(err.value) and "Rossi" in str(err.value)


def test_a_name_matching_two_drivers_is_ambiguous_not_guessed() -> None:
    field = pd.concat([FIELD, pd.DataFrame({"DriverId": ["carlos_sainz_jr"],
                                            "FullName": ["Carlos Sainz Jr"]})])
    with pytest.raises(market_names.MarketNameError, match="ambiguous"):
        market_names.resolve_outcomes(["Carlos Sainz"], field)


def test_events_join_races_on_date_within_tolerance() -> None:
    cal = pd.DataFrame({"Year": [2025, 2025], "RoundNumber": [1, 2],
                        "RaceDate": pd.to_datetime(["2025-03-16", "2025-03-23"])})
    assert market_names.match_race("2025-03-17T00:00:00Z", cal) == (2025, 1)
    assert market_names.match_race("2025-03-22", cal) == (2025, 2)
    assert market_names.match_race("2025-06-01", cal) is None
    with pytest.raises(market_names.MarketNameError, match="equidistant"):
        market_names.match_race("2025-03-19 12:00", pd.DataFrame({
            "Year": [2025, 2025], "RoundNumber": [1, 2],
            "RaceDate": pd.to_datetime(["2025-03-18", "2025-03-20"])}))


def test_only_race_winner_listings_pass_the_polymarket_filter() -> None:
    assert markets.is_polymarket_race_winner("Singapore Grand Prix Winner")
    assert markets.is_polymarket_race_winner("F1: Belgian Grand Prix Winner")
    for other in ("Miami Grand Prix Pole Winner", "China Grand Prix Sprint Winner",
                  "F1 Bahrain Grand Prix: which constructor scores the most points",
                  "F1 Drivers Champion", "Monaco Grand Prix Head-to-Head Matchups"):
        assert not markets.is_polymarket_race_winner(other)


# --------------------------------------------------------------------------- #
# Fetching (get_json replaced)
# --------------------------------------------------------------------------- #


def test_a_polymarket_event_is_normalised_and_skips_placeholders(monkeypatch) -> None:
    calls = []

    def fake(url, params=None):
        calls.append(url)
        return {"history": [{"t": 1_000, "p": 0.2}, {"t": 4_600, "p": 0.25}]}

    monkeypatch.setattr(markets, "get_json", fake)
    event = {"id": 7, "title": "X Grand Prix Winner", "startDate": "2025-03-01T00:00:00Z",
             "endDate": "2025-03-16T00:00:00Z", "closed": True, "markets": [
                 {"groupItemTitle": "Norris", "volumeNum": 10.0,
                  "clobTokenIds": json.dumps(["yes", "no"])},
                 {"groupItemTitle": "Driver A", "volumeNum": 0.0,
                  "clobTokenIds": json.dumps(["a", "b"])}]}
    raw = markets.fetch_polymarket_event(event)
    assert raw["settled"] and raw["race_date"] == "2025-03-16"
    assert [o["name"] for o in raw["outcomes"]] == ["Norris", "Driver A"]
    assert raw["outcomes"][0]["history"][1] == {"ts": 4600, "price": 0.25, "bid": None,
                                               "ask": None, "volume": None}
    assert raw["outcomes"][1]["history"] == []
    assert len(calls) == 3   # 15 days in 7-day chunks; the placeholder cost no request


def test_a_cached_settled_event_is_not_requested_again(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MARKETS_DIR", tmp_path)
    path = markets.cache_path("polymarket", "7")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"settled": True, "race_date": "2025-03-16"}))
    monkeypatch.setattr(markets, "polymarket_events", lambda: [{"id": 7, "endDate": "2025-03-16"}])
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("refetched a settled event"))
    monkeypatch.setattr(markets, "fetch_polymarket_event", boom)
    assert markets.pull(["polymarket"])["cached"] == 1


# --------------------------------------------------------------------------- #
# The long table
# --------------------------------------------------------------------------- #

RESULTS = pd.DataFrame({
    "Year": 2025, "RoundNumber": 1, "RaceDate": pd.Timestamp("2025-03-16"),
    "DriverId": ["norris", "max_verstappen", "antonelli"],
    "FullName": ["Lando Norris", "Max Verstappen", "Andrea Kimi Antonelli"],
})


def _event(source, event_id, race_date, outcomes, volume=100.0):
    return {"source": source, "event_id": event_id, "title": "t", "race_date": race_date,
            "settled": True,
            "outcomes": [{"name": n, "volume": volume, "history": h} for n, h in outcomes]}


def _hist(*points):
    return [{"ts": t, "price": p, "bid": b, "ask": a, "volume": 1.0} for t, p, b, a in points]


def test_build_prices_resolves_names_and_drops_a_non_f1_event() -> None:
    f1 = _event("polymarket", "1", "2025-03-16", [
        ("Norris", _hist((1_000, 0.5, None, None))),
        ("Verstappen", _hist((1_000, 0.3, None, None))),
        ("Kimi Antonelli", _hist((1_000, 0.2, None, None))),
        ("Other", [])])
    indy = _event("polymarket", "2", "2025-03-16", [
        ("Palou", _hist((1_000, 0.5, None, None))), ("Rossi", _hist((1_000, 0.5, None, None)))])
    out = markets.build_prices([f1, indy], RESULTS)
    assert set(out["event_id"]) == {"1"}
    assert set(out["DriverId"]) == {"norris", "max_verstappen", "antonelli"}


def test_build_prices_raises_on_one_stray_name_in_a_real_f1_event() -> None:
    event = _event("kalshi", "1", "2025-03-16", [
        ("Lando Norris", _hist((1_000, 0.5, 0.4, 0.6))),
        ("Max Verstappen", _hist((1_000, 0.3, 0.2, 0.4))),
        ("Mystery Driver", _hist((1_000, 0.2, 0.1, 0.3)))])
    with pytest.raises(market_names.MarketNameError, match="Mystery"):
        markets.build_prices([event], RESULTS)


def test_the_busier_of_two_listings_for_one_race_wins() -> None:
    quiet = _event("polymarket", "q", "2025-03-16", [("Norris", _hist((1, 0.9, None, None)))],
                   volume=1.0)
    busy = _event("polymarket", "b", "2025-03-16", [("Norris", _hist((1, 0.4, None, None)))],
                  volume=500.0)
    assert set(markets.build_prices([quiet, busy], RESULTS)["event_id"]) == {"b"}


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #


def _clock(race="2025-03-16 14:00"):
    t = lambda s: pd.Timestamp(s, tz=UTC)
    return pd.DataFrame({"Year": [2025], "RoundNumber": [1], "fp1": [t("2025-03-14 02:30")],
                         "quali": [t("2025-03-15 05:00")], "race": [t(race)]})


def test_a_snapshot_after_lights_out_is_refused() -> None:
    bad = market_eval.Snapshot("late", "race", 5.0, "post_quali")
    with pytest.raises(market_eval.MarketLeakageError):
        market_eval.snapshot_times(_clock(), bad)
    ok = market_eval.snapshot_times(_clock(), market_eval.SNAPSHOT_BY_NAME["close"])
    assert ok["snapshot_ts"].iloc[0] == pd.Timestamp("2025-03-16 13:55", tz=UTC)


def test_a_one_sided_book_is_not_a_quote() -> None:
    frame = pd.DataFrame({"price": [0.02, 0.10], "bid": [0.0, 0.08], "ask": [0.99, 0.12]})
    out = market_eval.quote(frame)
    assert out["p"].tolist() == pytest.approx([0.02, 0.10])   # last trade, then the midpoint
    assert np.isnan(out["spread"].iloc[0]) and out["spread"].iloc[1] == pytest.approx(0.04)


def _prices():
    ts = lambda s: pd.Timestamp(s, tz=UTC)
    rows = []
    for driver, series in {"norris": [(ts("2025-03-14 00:00"), 0.40), (ts("2025-03-14 09:00"), 0.60),
                                      (ts("2025-03-14 12:00"), 0.90)],   # last is after the snapshot
                           "max_verstappen": [(ts("2025-03-14 00:00"), 0.40)]}.items():
        for when, p in series:
            rows.append({"source": "polymarket", "event_id": "1", "Year": 2025,
                         "RoundNumber": 1, "outcome": driver, "DriverId": driver, "ts": when,
                         "price": p, "bid": np.nan, "ask": np.nan, "volume": np.nan})
    return pd.DataFrame(rows)


def test_the_snapshot_uses_the_last_price_before_it_and_never_one_after() -> None:
    field = RESULTS[["Year", "RoundNumber", "DriverId"]]
    times = market_eval.snapshot_times(_clock(), market_eval.SNAPSHOT_BY_NAME["pre_weekend"])
    out = market_eval.market_snapshot(_prices(), field, times).set_index("DriverId")
    # FP1 is 02:30 on the 14th, so the snapshot is 01:30: only the 00:00 prints qualify.
    assert out.loc["norris", "market_raw"] == pytest.approx(0.40)
    assert out.loc["max_verstappen", "market_raw"] == pytest.approx(0.40)
    assert out.loc["antonelli", "market_raw"] == market_eval.UNLISTED_PRICE
    assert not out.loc["antonelli", "listed"]
    assert out["p_market"].sum() == pytest.approx(1.0)
    assert out.loc["norris", "raw_total"] == pytest.approx(0.80)
    assert out.loc["norris", "age_min"] == pytest.approx(90.0)


def test_a_market_that_had_not_opened_yet_has_no_rows() -> None:
    field = RESULTS[["Year", "RoundNumber", "DriverId"]]
    early = pd.DataFrame({"Year": [2025], "RoundNumber": [1],
                          "snapshot_ts": [pd.Timestamp("2025-03-01", tz=UTC)]})
    assert market_eval.market_snapshot(_prices(), field, early).empty


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _log(n_races=30, informed=True, seed=3):
    """A market that is right on average, and a model that is or is not independent noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        truth = rng.dirichlet(np.ones(6) * 1.5)
        winner = rng.choice(6, p=truth)
        market = truth * np.exp(rng.normal(0, 0.15, 6))
        model = truth * np.exp(rng.normal(0, 0.15, 6)) if informed else rng.dirichlet(np.ones(6))
        for d in range(6):
            rows.append(dict(origin="backfill", Year=2025, RoundNumber=r + 1, snapshot="post_quali",
                             source="kalshi", DriverId=f"d{d}", p_model=model[d] / model.sum(),
                             p_market=market[d] / market.sum(), p_baseline=1 / 6, p_uniform=1 / 6,
                             market_raw=market[d], raw_total=1.0, listed=True, spread=0.02,
                             age_min=30.0, grid_position=d + 1, won=float(d == winner)))
    return pd.DataFrame(rows)


def test_log_loss_is_the_log_of_the_winners_probability() -> None:
    log = _log(2)
    scores = market_eval.per_race_scores(log)
    race = log[log["RoundNumber"] == 1]
    p = race["p_uniform"].to_numpy()
    assert scores["ll_p_uniform"].iloc[0] == pytest.approx(-np.log(p[0]))
    assert (scores["brier_p_uniform"] > 0).all()


def test_a_deterministic_miss_does_not_score_minus_infinity() -> None:
    log = _log(3)
    log["p_baseline"] = np.where(log["DriverId"] == "d0", 1.0, 0.0)
    assert np.isfinite(market_eval.per_race_scores(log)["ll_p_baseline"]).all()


def test_q1_reports_the_gap_and_what_it_could_have_detected() -> None:
    table = market_eval.q1_table(market_eval.per_race_scores(_log(40)))
    row = table.iloc[0]
    assert row["races"] == 40 and row["mde"] > 0
    assert row["ci_low"] <= row["ll_diff"] <= row["ci_high"]


def test_encompassing_finds_information_the_market_lacks() -> None:
    """If the model is the truth and the market is noisy, the blend must lean on the model."""
    rng = np.random.default_rng(5)
    races = []
    for _ in range(200):
        truth = rng.dirichlet(np.ones(6) * 1.5)
        w = int(rng.choice(6, p=truth))
        market = truth * np.exp(rng.normal(0, 0.8, 6))
        races.append((np.column_stack([np.log(market / market.sum()), np.log(truth)]), w))
    fit = market_eval.fit_blend(races)
    assert fit.b > 0.5 and fit.t_b > 3


def test_encompassing_finds_nothing_when_the_model_is_noise() -> None:
    rng = np.random.default_rng(6)
    races = []
    for _ in range(200):
        truth = rng.dirichlet(np.ones(6) * 1.5)
        w = int(rng.choice(6, p=truth))
        noise = rng.dirichlet(np.ones(6))
        races.append((np.column_stack([np.log(truth), np.log(noise)]), w))
    fit = market_eval.fit_blend(races)
    assert abs(fit.t_b) < 3 and fit.a > 0.7


def test_the_walk_forward_blend_never_sees_the_race_it_scores() -> None:
    log = _log(25, informed=True)
    walk = market_eval.blend_loglosses(market_eval._races_xy(log), min_fit=10)
    assert len(walk) == 25 and walk["fitted"].sum() == 15
    # before the minimum history, the blend IS the market
    early = walk.loc[~walk["fitted"]]
    assert np.allclose(early["ll_blend"], early["ll_market"])


def test_q2_table_has_a_row_per_source_and_snapshot() -> None:
    table = market_eval.q2_table(_log(30))
    assert list(table[["source", "snapshot"]].iloc[0]) == ["kalshi", "post_quali"]
    assert {"b", "t_b", "wf_gain", "mde"} <= set(table.columns)


def test_slices_cover_favourites_and_longshots() -> None:
    table = market_eval.slice_table(_log(30), "kalshi", "post_quali")
    labels = set(table["slice"])
    assert "market favourite (p >= 10%)" in labels and "grid P1-2" in labels


# --------------------------------------------------------------------------- #
# Staking
# --------------------------------------------------------------------------- #


def _race(p_model, p_market, won, spread=0.02):
    return pd.DataFrame({"p_model": p_model, "p_market": p_market, "won": won,
                         "spread": spread})


def test_no_bet_inside_the_edge_threshold() -> None:
    race = _race([0.30, 0.70], [0.28, 0.72], [1.0, 0.0])
    assert market_eval.stake_race(race, source="kalshi") == 0.0


def test_a_winning_bet_pays_net_of_spread_and_fee() -> None:
    race = _race([0.50, 0.50], [0.30, 0.70], [1.0, 0.0])
    free = market_eval.stake_race(race, source="polymarket")
    kalshi = market_eval.stake_race(race, source="kalshi")
    assert free > 0 and 0 < kalshi < free
    assert market_eval.kalshi_fee(10, 0.5, 0.07) == pytest.approx(0.18)   # ceil(0.175 -> 0.18)


def test_a_losing_bet_loses_the_stake() -> None:
    race = _race([0.50, 0.50], [0.30, 0.70], [0.0, 1.0])
    assert market_eval.stake_race(race, source="polymarket") < 0


# --------------------------------------------------------------------------- #
# The log
# --------------------------------------------------------------------------- #


def test_a_logged_claim_is_not_overwritten(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MARKET_LOG_PATH", tmp_path / "market_log.csv")
    first = _log(1).assign(origin="live", won=np.nan)
    live, added = market_eval.append_live(first)
    assert added == len(first)
    market_eval.write_log(live=live)
    later = first.assign(p_market=0.5)           # same race/snapshot/source, a "better" look
    live2, added2 = market_eval.append_live(later)
    assert added2 == 0 and live2["p_market"].ne(0.5).all()


def test_settle_fills_won_only_for_classified_races() -> None:
    live = _log(1).assign(origin="live", won=np.nan, RoundNumber=1)
    live["DriverId"] = ["norris", "max_verstappen", "antonelli", "x1", "x2", "x3"]
    results = RESULTS.assign(ClassifiedPosition=["1", "2", "3"])
    out = market_eval.settle(live, results)
    assert out.loc[out["DriverId"] == "norris", "won"].iloc[0] == 1.0
    assert out["won"].isna().sum() == 0
    unraced = market_eval.settle(live.assign(RoundNumber=9), results)
    assert unraced["won"].isna().all()


def test_kalshi_candles_read_both_the_live_and_the_historical_schema(monkeypatch) -> None:
    live = {"candlesticks": [{"end_period_ts": 10, "volume_fp": "5.00",
                              "price": {"close_dollars": "0.2000"},
                              "yes_bid": {"close_dollars": "0.1800"},
                              "yes_ask": {"close_dollars": "0.2200"}}]}
    hist = {"candlesticks": [{"end_period_ts": 10, "volume": "5.00",
                              "price": {"close": "0.2000"}, "yes_bid": {"close": "0.1800"},
                              "yes_ask": {"close": "0.2200"}}]}
    for page in (live, hist):
        monkeypatch.setattr(markets, "get_json", lambda url, params=None, page=page: page)
        assert markets.kalshi_candles("T", 0, 100, 60) == [
            {"ts": 10, "price": 0.2, "bid": 0.18, "ask": 0.22, "volume": 5.0}]


def test_a_settled_kalshi_event_falls_back_to_the_historical_listing(monkeypatch) -> None:
    seen = []

    def fake(url, params=None):
        seen.append(url.rsplit("/", 2)[-2] + "/" + url.rsplit("/", 1)[-1])
        if url.endswith("/historical/markets"):
            return {"markets": [{"ticker": "K-A", "yes_sub_title": "Norris", "volume_fp": "0.00",
                                 "status": "finalized", "open_time": "2025-03-14T14:00:00Z",
                                 "expected_expiration_time": "2025-03-16T06:00:00Z"}]}
        return {"markets": []}

    monkeypatch.setattr(markets, "get_json", fake)
    raw = markets.fetch_kalshi_event({"event_ticker": "K", "title": "t"})
    assert raw["race_date"] == "2025-03-16" and raw["settled"]
    assert raw["outcomes"][0] == {"name": "Norris", "volume": 0.0, "history": []}
