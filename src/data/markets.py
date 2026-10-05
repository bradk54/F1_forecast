"""Pull race-winner prices from Polymarket and Kalshi.

The second module allowed to touch the network (``ingest.py`` is the first), and
for the same reason: everything downstream operates on DataFrames, so scoring is
testable without reaching either exchange.  Both read endpoints are public and
need no key.

Three rules carried over from the FastF1 pulls:

* **Settled markets cannot change, so they are fetched once.**  Each event's raw
  history is cached under ``Data/raw/markets/<source>/<event>.json`` and a cached
  settled event is never requested again.  A backfill is ~45 races x ~22 drivers
  x 2 sources of history requests; a weekly pull is the live race only.
* **Pace the requests.**  Neither API publishes a hard limit, but a 429 partway
  through a backfill looks exactly like a thin market.  Requests are spaced by
  :data:`REQUEST_PAUSE` and a 429 backs off and retries.
* **Never half-write.**  An event is cached only after every outcome's history
  loaded, so a cached event is a complete one.

The cache keeps what the exchange said, on one schema for both sources::

    {"source", "event_id", "title", "race_date", "settled",
     "outcomes": [{"name", "volume", "history": [{"ts", "price", "bid", "ask", "volume"}]}]}

``price`` is the last trade (Polymarket's only series; Kalshi's candle close);
``bid``/``ask`` are Kalshi's quoted close and are ``None`` for Polymarket, whose
history endpoint returns a single price.  Prices are probabilities in [0, 1].
:func:`build_prices` turns the cache into the long table scoring reads, resolving
names and events through :mod:`src.data.market_names`.

The market is a benchmark, never an input.  Nothing here feeds a model feature.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src import config
from src.data import market_names

log = logging.getLogger(__name__)

SOURCES = ("kalshi", "polymarket")

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = "KXF1RACE"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB = "https://clob.polymarket.com"
#: Polymarket's tag id for Formula 1 (the ``formula1`` tag on any F1 event).
POLYMARKET_F1_TAG = 435

#: Seconds between requests, and the backoff schedule after a 429.
REQUEST_PAUSE = 0.25
BACKOFF_SECONDS = (2.0, 8.0, 30.0)
#: Minutes per price point.  Hourly, as the snapshot rules need no finer a clock.
DEFAULT_FIDELITY_MIN = 60

#: A Polymarket event is a race-winner market if its title says so ...
_WINNER = re.compile(r"grand prix.*winner|winner.*grand prix|\bgp\b.*winner", re.I)
#: ... and none of the other things Polymarket lists beside it.
_NOT_RACE_WINNER = re.compile(
    r"pole|sprint|constructor|team|head|fastest|champion|podium|points|lap|finish|"
    r"leave|attend|testing|season", re.I)


class MarketFetchError(RuntimeError):
    """An exchange refused or garbled a request."""


def cache_path(source: str, event_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(event_id))
    return config.MARKETS_DIR / source / f"{safe}.json"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET with a pause before it and a backoff on 429.  Raises ``MarketFetchError``."""
    import requests

    for attempt, wait in enumerate((0.0, *BACKOFF_SECONDS)):
        time.sleep(max(wait, REQUEST_PAUSE))
        try:
            response = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            raise MarketFetchError(f"{url}: {exc}") from exc
        if response.status_code == 429 and attempt < len(BACKOFF_SECONDS):
            log.warning("429 from %s; backing off %.0fs", url, BACKOFF_SECONDS[attempt])
            continue
        if response.status_code != 200:
            raise MarketFetchError(f"{url} -> HTTP {response.status_code}")
        return response.json()
    raise MarketFetchError(f"{url}: still rate-limited after {len(BACKOFF_SECONDS)} retries")


def _ts(value: object) -> int:
    """Unix seconds for a timestamp, reading a naive one as UTC."""
    when = pd.Timestamp(value)
    when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    return int(when.timestamp())


def _num(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if out != out else out


# --------------------------------------------------------------------------- #
# Kalshi
# --------------------------------------------------------------------------- #


def _dollars(block: dict | None, key: str) -> float | None:
    """A candle price.  The live endpoint calls it ``close_dollars``, the historical ``close``."""
    block = block or {}
    return _num(block.get(f"{key}_dollars", block.get(key)))



def kalshi_events() -> list[dict]:
    """Every ``KXF1RACE`` event, settled or open, paged to the end."""
    events, cursor = [], None
    while True:
        params = {"series_ticker": KALSHI_SERIES, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        page = get_json(f"{KALSHI_API}/events", params)
        events += page.get("events", [])
        cursor = page.get("cursor")
        if not cursor or not page.get("events"):
            return events


def kalshi_candles(ticker: str, start: int, end: int, minutes: int) -> list[dict]:
    """Candles for one market, falling back to the historical endpoint on a 404."""
    params = {"start_ts": start, "end_ts": end, "period_interval": minutes}
    try:
        page = get_json(f"{KALSHI_API}/series/{KALSHI_SERIES}/markets/{ticker}/candlesticks",
                        params)
    except MarketFetchError as exc:
        if "HTTP 404" not in str(exc):
            raise
        page = get_json(f"{KALSHI_API}/historical/markets/{ticker}/candlesticks", params)
    rows = []
    for c in page.get("candlesticks", []):
        rows.append({
            "ts": int(c["end_period_ts"]),
            "price": _dollars(c.get("price"), "close"),
            "bid": _dollars(c.get("yes_bid"), "close"),
            "ask": _dollars(c.get("yes_ask"), "close"),
            "volume": _num(c.get("volume_fp", c.get("volume"))),
        })
    return rows


def fetch_kalshi_event(event: dict, minutes: int = DEFAULT_FIDELITY_MIN) -> dict | None:
    """One Kalshi race-winner event with every traded outcome's history."""
    params = {"event_ticker": event["event_ticker"], "limit": 200}
    markets = get_json(f"{KALSHI_API}/markets", params).get("markets", [])
    if not markets:
        # Settled markets leave the live listing for /historical/ (observed: every
        # 2025 event), with the same fields but no occurrence_datetime.
        markets = get_json(f"{KALSHI_API}/historical/markets", params).get("markets", [])
    if not markets:
        return None
    race_when = markets[0].get("occurrence_datetime") or markets[0].get("expected_expiration_time")
    outcomes = []
    for m in markets:
        volume = _num(m.get("volume_fp")) or 0.0
        history = []
        if volume > 0:
            start = _ts(m["open_time"])
            end = _ts(race_when) + 86_400
            history = kalshi_candles(m["ticker"], start, end, minutes)
        outcomes.append({"name": m.get("yes_sub_title") or m.get("title"),
                         "volume": volume, "history": history})
    return {
        "source": "kalshi", "event_id": event["event_ticker"], "title": event.get("title"),
        "race_date": pd.Timestamp(race_when).strftime("%Y-%m-%d"),
        "settled": all(m.get("status") in ("finalized", "settled") for m in markets),
        "outcomes": outcomes,
    }


# --------------------------------------------------------------------------- #
# Polymarket
# --------------------------------------------------------------------------- #


def is_polymarket_race_winner(title: str) -> bool:
    return bool(_WINNER.search(title)) and not _NOT_RACE_WINNER.search(title)


def polymarket_events() -> list[dict]:
    """Every F1-tagged race-winner event, closed and open."""
    found: dict[str, dict] = {}
    for closed in ("true", "false"):
        offset = 0
        while True:
            page = get_json(f"{POLYMARKET_GAMMA}/events", {
                "tag_id": POLYMARKET_F1_TAG, "closed": closed, "limit": 100, "offset": offset})
            for e in page:
                if is_polymarket_race_winner(e.get("title", "")):
                    found[str(e["id"])] = e
            if len(page) < 100:
                break
            offset += 100
    return list(found.values())


def fetch_polymarket_event(event: dict, minutes: int = DEFAULT_FIDELITY_MIN) -> dict | None:
    """One Polymarket event with each driver's Yes-token price history."""
    markets = event.get("markets") or []
    if not markets:
        return None
    start, end = _ts(event["startDate"]), _ts(event["endDate"]) + 86_400
    outcomes = []
    for m in markets:
        name = m.get("groupItemTitle") or m.get("question") or ""
        volume = _num(m.get("volumeNum")) or 0.0
        history = []
        if volume > 0 and not market_names.is_ignored(name):
            token = json.loads(m["clobTokenIds"])[0]
            page = get_json(f"{POLYMARKET_CLOB}/prices-history", {
                "market": token, "startTs": start, "endTs": end, "fidelity": minutes})
            history = [{"ts": int(p["t"]), "price": _num(p["p"]), "bid": None, "ask": None,
                        "volume": None} for p in page.get("history", [])]
        outcomes.append({"name": name, "volume": volume, "history": history})
    return {
        "source": "polymarket", "event_id": str(event["id"]), "title": event.get("title"),
        "race_date": pd.Timestamp(event["endDate"]).strftime("%Y-%m-%d"),
        "settled": bool(event.get("closed")),
        "outcomes": outcomes,
    }


# --------------------------------------------------------------------------- #
# Pulling and caching
# --------------------------------------------------------------------------- #


def read_cache(source: str | None = None) -> list[dict]:
    """Every cached event, optionally for one source."""
    sources = (source,) if source else SOURCES
    out = []
    for s in sources:
        for path in sorted((config.MARKETS_DIR / s).glob("*.json")):
            out.append(json.loads(path.read_text()))
    return out


def pull(
    sources: Iterable[str] = SOURCES, *, since_year: int = 2024, refresh: bool = False,
    minutes: int = DEFAULT_FIDELITY_MIN,
) -> dict[str, int]:
    """Fetch every uncached or unsettled race-winner event; return counts.

    A settled cached event is skipped unless ``refresh``.  Events dated before
    ``since_year`` are skipped without a request: the benchmark window starts
    where both exchanges do.
    """
    counts = {"fetched": 0, "cached": 0, "skipped": 0, "failed": 0}
    for source in sources:
        listing = kalshi_events() if source == "kalshi" else polymarket_events()
        log.info("%s: %d candidate events", source, len(listing))
        for event in listing:
            event_id = str(event.get("event_ticker") or event["id"])
            path = cache_path(source, event_id)
            if path.exists() and not refresh and json.loads(path.read_text()).get("settled"):
                counts["cached"] += 1
                continue
            stated = event.get("endDate") or event.get("last_updated_ts") or ""
            if source == "polymarket" and stated and int(stated[:4]) < since_year:
                counts["skipped"] += 1
                continue
            try:
                raw = (fetch_kalshi_event if source == "kalshi" else fetch_polymarket_event)(
                    event, minutes)
            except MarketFetchError as exc:
                log.error("%s %s: %s", source, event_id, exc)
                counts["failed"] += 1
                continue
            if raw is None or int(raw["race_date"][:4]) < since_year:
                counts["skipped"] += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw))
            counts["fetched"] += 1
    return counts


# --------------------------------------------------------------------------- #
# The long table
# --------------------------------------------------------------------------- #

PRICE_COLUMNS = ["source", "event_id", "Year", "RoundNumber", "outcome", "DriverId",
                 "ts", "price", "bid", "ask", "volume"]


def calendar_from_results(results: pd.DataFrame) -> pd.DataFrame:
    """One row per race: ``Year``, ``RoundNumber``, ``RaceDate``."""
    return (results[["Year", "RoundNumber", "RaceDate"]].drop_duplicates()
            .assign(RaceDate=lambda d: pd.to_datetime(d["RaceDate"]))
            .reset_index(drop=True))


def build_prices(events: list[dict], results: pd.DataFrame) -> pd.DataFrame:
    """Resolve cached events into one row per observation, on the race's field.

    An event whose date matches no race is skipped (a reported non-F1 or
    out-of-window listing).  One that matches a race but whose names mostly fail
    to resolve is skipped as not-F1 -- Polymarket lists an IndyCar race beside the
    Spanish Grand Prix on the same Sunday.  Anything else with an unresolved name
    raises :class:`~src.data.market_names.MarketNameError`.

    Two events for one (source, race) -- Polymarket re-listed Mexico -- keep the
    busier one.
    """
    calendar = calendar_from_results(results)
    fields = {k: g[["DriverId", "FullName"]].drop_duplicates("DriverId")
              for k, g in results.groupby(["Year", "RoundNumber"])}
    chosen: dict[tuple, dict] = {}
    for event in events:
        key = market_names.match_race(event["race_date"], calendar)
        if key is None:
            log.info("%s %s (%s): no race within tolerance", event["source"],
                     event["event_id"], event["race_date"])
            continue
        field = fields[key]
        names = [o["name"] for o in event["outcomes"] if not market_names.is_ignored(o["name"])]
        hits = [n for n in names if _resolvable(n, field)]
        if not names or len(hits) < 0.5 * len(names):
            log.info("%s %s: %d of %d names fit the %s field; not an F1 race",
                     event["source"], event["event_id"], len(hits), len(names), key)
            continue
        slot = (event["source"], *key)
        volume = sum(o["volume"] for o in event["outcomes"])
        if slot not in chosen or volume > chosen[slot][1]:
            chosen[slot] = (event, volume)

    rows = []
    for (source, year, rnd), (event, _) in chosen.items():
        resolved = market_names.resolve_outcomes(
            [o["name"] for o in event["outcomes"]], fields[(year, rnd)])
        for outcome in event["outcomes"]:
            driver = resolved[outcome["name"]]
            if driver is None:
                continue
            for h in outcome["history"]:
                rows.append((source, event["event_id"], year, rnd, outcome["name"], driver,
                             h["ts"], h["price"], h["bid"], h["ask"], h["volume"]))
    frame = pd.DataFrame(rows, columns=PRICE_COLUMNS)
    frame["ts"] = pd.to_datetime(frame["ts"], unit="s", utc=True)
    return frame.sort_values(["source", "Year", "RoundNumber", "DriverId", "ts"],
                             ignore_index=True)


def _resolvable(name: str, field: pd.DataFrame) -> bool:
    try:
        market_names.resolve_outcomes([name], field)
        return True
    except market_names.MarketNameError:
        return False


def save_prices(frame: pd.DataFrame) -> Path:
    config.MARKET_PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.MARKET_PRICES_PATH, index=False)
    return config.MARKET_PRICES_PATH


def load_prices() -> pd.DataFrame:
    if not config.MARKET_PRICES_PATH.exists():
        raise FileNotFoundError(
            f"{config.MARKET_PRICES_PATH} is missing; build it with "
            "`python -m src.models.forecast markets pull`")
    return pd.read_parquet(config.MARKET_PRICES_PATH)


# --------------------------------------------------------------------------- #
# Session clock
# --------------------------------------------------------------------------- #

CLOCK_COLUMNS = ["Year", "RoundNumber", "fp1", "quali", "race"]


def session_clock(years: Iterable[int], *, offline: bool = True) -> pd.DataFrame:
    """UTC start of FP1, qualifying and the race for each round, from the schedule.

    Cached to ``Data/raw/markets/clock_<year>.json`` so a scoring run is
    reproducible without FastF1.  A completed season is read once and kept.
    """
    frames = []
    for year in years:
        path = config.MARKETS_DIR / f"clock_{year}.json"
        if path.exists():
            frames.append(pd.read_json(path, orient="records", convert_dates=False))
            continue
        import fastf1

        from src.data import ingest

        ingest.configure(offline=offline)
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        rows = []
        for _, event in schedule.iterrows():
            by_name = {}
            for n in range(1, 6):
                name, when = event.get(f"Session{n}"), event.get(f"Session{n}DateUtc")
                if isinstance(name, str) and pd.notna(when):
                    by_name[name] = pd.Timestamp(when, tz="UTC").isoformat()
            rows.append({"Year": year, "RoundNumber": int(event["RoundNumber"]),
                         "fp1": by_name.get("Practice 1"), "quali": by_name.get("Qualifying"),
                         "race": by_name.get("Race")})
        frame = pd.DataFrame(rows, columns=CLOCK_COLUMNS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(frame.to_json(orient="records"))
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    for column in ("fp1", "quali", "race"):
        out[column] = pd.to_datetime(out[column], utc=True)
    return out
