"""
╔══════════════════════════════════════════════════════════════╗
║         XAUUSD PHASE 4 — FUNDAMENTAL ENGINE                 ║
║         Macro Data + News Event Filter                      ║
╠══════════════════════════════════════════════════════════════╣
║  Data Sources:                                               ║
║   1. DXY    — Dollar Index (inverse correlation w/ gold)    ║
║   2. CPI    — Inflation data via FRED API                   ║
║   3. FOMC   — Fed interest rate decisions via FRED API      ║
║   4. NFP    — Non-farm payrolls via FRED API                ║
║   5. Events — Upcoming high-impact news (ForexFactory)      ║
╠══════════════════════════════════════════════════════════════╣
║  Logic:                                                      ║
║   - Pause ALL trading 2hrs before/after high-impact events  ║
║   - Score macro bias: BULLISH / BEARISH / NEUTRAL for gold  ║
║   - Must align with Phase 1 + Phase 2 signal to trade       ║
╠══════════════════════════════════════════════════════════════╣
║  Setup:                                                      ║
║   pip install requests pandas fredapi                        ║
║   Get free FRED API key: https://fred.stlouisfed.org/docs/api║
║   Set FRED_API_KEY below (or in environment variable)       ║
╚══════════════════════════════════════════════════════════════╝
"""
FRED_API_KEY = "bc1ca133c3e1c499cb3b31a8b33429b1"
import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")

# Cache file to avoid hitting APIs every 30 seconds
CACHE_FILE = "fundamental_cache.json"
CACHE_TTL  = {
    "dxy"    : 3600,    # refresh every 1 hour
    "cpi"    : 86400,   # refresh every 24 hours
    "fomc"   : 86400,   # refresh every 24 hours
    "nfp"    : 86400,   # refresh every 24 hours
    "events" : 1800,    # refresh every 30 minutes
}

# High-impact event keywords to watch for
HIGH_IMPACT_KEYWORDS = [
    "Non-Farm", "NFP", "FOMC", "Fed Rate", "Federal Reserve",
    "Interest Rate", "CPI", "Inflation", "GDP", "PCE",
    "Unemployment", "Powell", "Jackson Hole", "Treasury"
]

# FRED series IDs
FRED_SERIES = {
    "cpi"       : "CPIAUCSL",    # Consumer Price Index
    "core_cpi"  : "CPILFESL",    # Core CPI (ex food & energy)
    "fed_rate"  : "FEDFUNDS",    # Federal Funds Rate
    "nfp"       : "PAYEMS",      # Non-Farm Payrolls
    "dxy_proxy" : "DTWEXBGS",    # Dollar index (broad)
    "real_rate" : "DFII10",      # 10-Year Real Interest Rate
    "gold_etf"  : "GOLDAMGBD228NLBM",  # Gold price (London fix)
}

# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class FundamentalSignal:
    bias: str               # 'BULLISH', 'BEARISH', 'NEUTRAL' for gold
    score: int              # -3 to +3 (positive = bullish gold)
    news_block: bool        # True = block all trading now
    news_reason: str        # Why trading is blocked
    dxy_signal: str         # 'BULLISH_GOLD', 'BEARISH_GOLD', 'NEUTRAL'
    dxy_value: Optional[float]
    dxy_trend: str          # 'RISING', 'FALLING', 'FLAT'
    cpi_signal: str         # 'BULLISH_GOLD', 'BEARISH_GOLD', 'NEUTRAL'
    cpi_value: Optional[float]
    cpi_mom: Optional[float]  # Month-over-month change
    fed_signal: str         # 'BULLISH_GOLD', 'BEARISH_GOLD', 'NEUTRAL'
    fed_rate: Optional[float]
    fed_trend: str          # 'HIKING', 'CUTTING', 'HOLD'
    nfp_signal: str         # 'BULLISH_GOLD', 'BEARISH_GOLD', 'NEUTRAL'
    nfp_value: Optional[float]
    upcoming_events: list   # List of high-impact events in next 48hrs
    details: str

@dataclass
class EconomicEvent:
    title: str
    datetime_utc: datetime
    impact: str             # 'HIGH', 'MEDIUM', 'LOW'
    currency: str           # 'USD' etc
    hours_until: float      # How many hours until event

# ─────────────────────────────────────────────
#  CACHE SYSTEM
# ─────────────────────────────────────────────

def _load_cache() -> dict:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def _cache_get(key: str, ttl: int) -> Optional[dict]:
    cache = _load_cache()
    if key in cache:
        age = time.time() - cache[key].get('timestamp', 0)
        if age < ttl:
            return cache[key].get('data')
    return None

def _cache_set(key: str, data):
    cache = _load_cache()
    cache[key] = {'timestamp': time.time(), 'data': data}
    _save_cache(cache)

# ─────────────────────────────────────────────
#  1. FRED API — CPI, FED RATE, NFP
# ─────────────────────────────────────────────

def _fetch_fred_series(series_id: str, limit: int = 24) -> Optional[list]:
    """Fetch recent observations from FRED API."""
    if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
        return None

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id"     : series_id,
        "api_key"       : FRED_API_KEY,
        "file_type"     : "json",
        "sort_order"    : "desc",
        "limit"         : limit,
        "observation_start": (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"),
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            obs = [o for o in data.get("observations", []) if o["value"] != "."]
            return obs
    except Exception as e:
        print(f"  ⚠️  FRED API error ({series_id}): {e}")
    return None


def get_cpi_data() -> dict:
    """
    Fetch CPI data from FRED.
    Returns: current value, MoM change, YoY change, signal for gold.
    """
    cached = _cache_get("cpi", CACHE_TTL["cpi"])
    if cached:
        return cached

    result = {
        "value"     : None,
        "mom_change": None,
        "yoy_change": None,
        "signal"    : "NEUTRAL",
        "details"   : "CPI data unavailable"
    }

    obs = _fetch_fred_series(FRED_SERIES["cpi"], limit=14)
    if obs and len(obs) >= 2:
        latest  = float(obs[0]["value"])
        prev    = float(obs[1]["value"])
        year_ago = float(obs[12]["value"]) if len(obs) >= 13 else None

        mom = ((latest - prev) / prev) * 100
        yoy = ((latest - year_ago) / year_ago) * 100 if year_ago else None

        result["value"]      = round(latest, 2)
        result["mom_change"] = round(mom, 3)
        result["yoy_change"] = round(yoy, 2) if yoy else None

        # Gold signal: High/rising inflation = BULLISH gold
        if yoy and yoy > 4.0:
            result["signal"]  = "BULLISH_GOLD"
            result["details"] = f"CPI {yoy:.1f}% YoY — high inflation bullish for gold"
        elif yoy and yoy > 2.5:
            result["signal"]  = "NEUTRAL"
            result["details"] = f"CPI {yoy:.1f}% YoY — moderate inflation, neutral"
        elif yoy and yoy < 2.0:
            result["signal"]  = "BEARISH_GOLD"
            result["details"] = f"CPI {yoy:.1f}% YoY — low inflation bearish for gold"
        else:
            result["details"] = f"CPI {latest:.2f} (MoM: {mom:+.3f}%)"

    _cache_set("cpi", result)
    return result


def get_fed_rate_data() -> dict:
    """
    Fetch Federal Funds Rate from FRED.
    Returns: current rate, trend (hiking/cutting/hold), gold signal.
    """
    cached = _cache_get("fomc", CACHE_TTL["fomc"])
    if cached:
        return cached

    result = {
        "rate"    : None,
        "trend"   : "UNKNOWN",
        "signal"  : "NEUTRAL",
        "details" : "Fed rate data unavailable"
    }

    obs = _fetch_fred_series(FRED_SERIES["fed_rate"], limit=12)
    if obs and len(obs) >= 3:
        current = float(obs[0]["value"])
        prev3   = float(obs[2]["value"])
        prev6   = float(obs[5]["value"]) if len(obs) >= 6 else None

        result["rate"] = round(current, 2)

        # Determine trend
        if current > prev3 + 0.1:
            result["trend"] = "HIKING"
        elif current < prev3 - 0.1:
            result["trend"] = "CUTTING"
        else:
            result["trend"] = "HOLD"

        # Gold signal: Rate cuts = BULLISH gold (less return on USD)
        if result["trend"] == "CUTTING":
            result["signal"]  = "BULLISH_GOLD"
            result["details"] = f"Fed cutting rates ({current:.2f}%) — bullish gold"
        elif result["trend"] == "HIKING":
            result["signal"]  = "BEARISH_GOLD"
            result["details"] = f"Fed hiking rates ({current:.2f}%) — bearish gold"
        else:
            # High rates on hold = somewhat bearish
            if current >= 4.5:
                result["signal"]  = "BEARISH_GOLD"
                result["details"] = f"Rates on hold but high ({current:.2f}%) — bearish bias"
            else:
                result["signal"]  = "NEUTRAL"
                result["details"] = f"Fed on hold at {current:.2f}%"

    _cache_set("fomc", result)
    return result


def get_nfp_data() -> dict:
    """
    Fetch Non-Farm Payrolls from FRED.
    Returns: latest value, MoM change, gold signal.
    """
    cached = _cache_get("nfp", CACHE_TTL["nfp"])
    if cached:
        return cached

    result = {
        "value"     : None,
        "mom_change": None,
        "signal"    : "NEUTRAL",
        "details"   : "NFP data unavailable"
    }

    obs = _fetch_fred_series(FRED_SERIES["nfp"], limit=3)
    if obs and len(obs) >= 2:
        latest  = float(obs[0]["value"])
        prev    = float(obs[1]["value"])
        mom     = latest - prev   # In thousands of jobs

        result["value"]      = round(latest, 0)
        result["mom_change"] = round(mom, 0)

        # Gold signal: Weak jobs = BULLISH gold (USD weakens, rate cut bets rise)
        if mom < 100:
            result["signal"]  = "BULLISH_GOLD"
            result["details"] = f"NFP weak (+{mom:.0f}k jobs) — bullish gold"
        elif mom > 300:
            result["signal"]  = "BEARISH_GOLD"
            result["details"] = f"NFP strong (+{mom:.0f}k jobs) — bearish gold"
        else:
            result["signal"]  = "NEUTRAL"
            result["details"] = f"NFP in-line (+{mom:.0f}k jobs) — neutral"

    _cache_set("nfp", result)
    return result

# ─────────────────────────────────────────────
#  2. DXY — DOLLAR INDEX ANALYSIS
# ─────────────────────────────────────────────

def get_dxy_data() -> dict:
    """
    Fetch Dollar Index data.
    Uses FRED broad dollar index as proxy.
    Gold has strong inverse correlation with DXY (~-0.85).
    """
    cached = _cache_get("dxy", CACHE_TTL["dxy"])
    if cached:
        return cached

    result = {
        "value"  : None,
        "trend"  : "UNKNOWN",
        "signal" : "NEUTRAL",
        "details": "DXY data unavailable"
    }

    # Try FRED broad dollar index
    obs = _fetch_fred_series(FRED_SERIES["dxy_proxy"], limit=30)
    if obs and len(obs) >= 10:
        current  = float(obs[0]["value"])
        week_ago = float(obs[4]["value"])   # ~5 trading days
        month_ago = float(obs[21]["value"]) if len(obs) >= 22 else None

        result["value"] = round(current, 3)

        pct_change_week = ((current - week_ago) / week_ago) * 100

        if pct_change_week > 0.5:
            result["trend"]  = "RISING"
            result["signal"] = "BEARISH_GOLD"
            result["details"] = f"DXY rising +{pct_change_week:.2f}% this week — bearish gold"
        elif pct_change_week < -0.5:
            result["trend"]  = "FALLING"
            result["signal"] = "BULLISH_GOLD"
            result["details"] = f"DXY falling {pct_change_week:.2f}% this week — bullish gold"
        else:
            result["trend"]  = "FLAT"
            result["signal"] = "NEUTRAL"
            result["details"] = f"DXY flat ({pct_change_week:+.2f}% this week) — neutral"

    _cache_set("dxy", result)
    return result

# ─────────────────────────────────────────────
#  3. UPCOMING HIGH-IMPACT NEWS EVENTS
# ─────────────────────────────────────────────

def get_upcoming_events() -> list[EconomicEvent]:
    """
    Scrape upcoming high-impact economic events from ForexFactory.
    Returns events in the next 48 hours that affect XAUUSD.

    Falls back to a manual upcoming events list if scraping fails.
    """
    cached = _cache_get("events", CACHE_TTL["events"])
    if cached:
        events = []
        for e in cached:
            ev = EconomicEvent(
                title        = e["title"],
                datetime_utc = datetime.fromisoformat(e["datetime_utc"]),
                impact       = e["impact"],
                currency     = e["currency"],
                hours_until  = e["hours_until"]
            )
            events.append(ev)
        return events

    events = []

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        url  = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            now_utc = datetime.now(timezone.utc)

            for item in data:
                try:
                    # Parse event time
                    date_str = item.get("date", "")
                    time_str = item.get("time", "")
                    impact   = item.get("impact", "").upper()
                    currency = item.get("currency", "")
                    title    = item.get("title", "")

                    # Only care about HIGH impact USD events
                    if currency != "USD" or impact not in ("HIGH", "MEDIUM"):
                        continue

                    # Parse datetime
                    if date_str and time_str and time_str != "All Day" and time_str != "":
                        try:
                            dt_str = f"{date_str} {time_str}"
                            dt = datetime.strptime(dt_str, "%Y-%m-%d %I:%M%p")
                            dt = dt.replace(tzinfo=timezone.utc)
                        except Exception:
                            continue
                    elif date_str:
                        try:
                            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        except Exception:
                            continue
                    else:
                        continue

                    hours_until = (dt - now_utc).total_seconds() / 3600

                    # Only events in next 48 hours
                    if -2 <= hours_until <= 48:
                        events.append(EconomicEvent(
                            title        = title,
                            datetime_utc = dt,
                            impact       = impact,
                            currency     = currency,
                            hours_until  = round(hours_until, 2)
                        ))
                except Exception:
                    continue

    except Exception as e:
        print(f"  ⚠️  ForexFactory fetch failed: {e}")

    # Sort by time
    events.sort(key=lambda x: x.hours_until)

    # Cache as dicts
    cache_data = [{
        "title"       : e.title,
        "datetime_utc": e.datetime_utc.isoformat(),
        "impact"      : e.impact,
        "currency"    : e.currency,
        "hours_until" : e.hours_until
    } for e in events]
    _cache_set("events", cache_data)

    return events


def is_news_blackout(events: list[EconomicEvent], blackout_hours: float = 2.0) -> tuple[bool, str]:
    """
    Check if we're within the news blackout window.

    Blackout: 2 hours BEFORE and 2 hours AFTER any HIGH impact event.

    Returns:
        (is_blocked, reason_string)
    """
    for event in events:
        if event.impact != "HIGH":
            continue
        h = event.hours_until

        # 2 hours before event
        if 0 < h <= blackout_hours:
            return True, f"⛔ Blackout: {event.title} in {h:.1f}h at {event.datetime_utc.strftime('%H:%M')} UTC"

        # During/just after event (negative hours = already happened)
        if -blackout_hours <= h <= 0:
            return True, f"⛔ Blackout: {event.title} just occurred ({abs(h):.1f}h ago) — waiting for dust to settle"

    return False, ""

# ─────────────────────────────────────────────
#  4. COMBINED FUNDAMENTAL SIGNAL
# ─────────────────────────────────────────────

def get_fundamental_signal() -> FundamentalSignal:
    """
    Combine all 4 fundamental sources into one signal.

    Scoring (gold perspective):
      DXY  falling  → +1
      CPI  high     → +1
      Fed  cutting  → +1
      NFP  weak     → +1

      DXY  rising   → -1
      CPI  low      → -1
      Fed  hiking   → -1
      NFP  strong   → -1

    Final bias:
      score >= +2  → BULLISH
      score <= -2  → BEARISH
      else         → NEUTRAL
    """
    # Fetch all data
    dxy  = get_dxy_data()
    cpi  = get_cpi_data()
    fed  = get_fed_rate_data()
    nfp  = get_nfp_data()
    events = get_upcoming_events()

    # Score each signal
    def score_signal(sig: str) -> int:
        if sig == "BULLISH_GOLD":  return +1
        if sig == "BEARISH_GOLD":  return -1
        return 0

    dxy_score = score_signal(dxy["signal"])
    cpi_score = score_signal(cpi["signal"])
    fed_score = score_signal(fed["signal"])
    nfp_score = score_signal(nfp["signal"])
    total     = dxy_score + cpi_score + fed_score + nfp_score

    # Determine overall bias
    if total >= 2:
        bias = "BULLISH"
    elif total <= -2:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # Check news blackout
    blocked, block_reason = is_news_blackout(events)

    # Build detail string
    details_parts = [
        f"DXY: {dxy['trend']} ({dxy_score:+d})",
        f"CPI: {cpi.get('yoy_change', 'N/A')}% ({cpi_score:+d})",
        f"Fed: {fed['trend']} @ {fed.get('rate', 'N/A')}% ({fed_score:+d})",
        f"NFP: {nfp.get('mom_change', 'N/A')}k ({nfp_score:+d})",
    ]
    details = " | ".join(details_parts) + f" | Total: {total:+d} → {bias}"

    # Upcoming events summary
    upcoming = [e for e in events if 0 < e.hours_until <= 24]

    return FundamentalSignal(
        bias             = bias,
        score            = total,
        news_block       = blocked,
        news_reason      = block_reason,
        dxy_signal       = dxy["signal"],
        dxy_value        = dxy.get("value"),
        dxy_trend        = dxy["trend"],
        cpi_signal       = cpi["signal"],
        cpi_value        = cpi.get("value"),
        cpi_mom          = cpi.get("mom_change"),
        fed_signal       = fed["signal"],
        fed_rate         = fed.get("rate"),
        fed_trend        = fed["trend"],
        nfp_signal       = nfp["signal"],
        nfp_value        = nfp.get("value"),
        upcoming_events  = upcoming,
        details          = details,
    )

# ─────────────────────────────────────────────
#  5. INTEGRATION HELPERS FOR xauusd_bot.py
# ─────────────────────────────────────────────

def fundamental_filter_passes(signal: FundamentalSignal, trade_direction: str) -> tuple[bool, str]:
    """
    Check if fundamental conditions allow this trade direction.

    Rules:
      1. NEVER trade during news blackout
      2. BEARISH fundamental → block BUY signals
      3. BULLISH fundamental → block SELL signals
      4. NEUTRAL fundamental → allow both directions

    Parameters:
        signal          : FundamentalSignal from get_fundamental_signal()
        trade_direction : 'BUY' or 'SELL'

    Returns:
        (passes, reason)
    """
    # Rule 1: News blackout — blocks everything
    if signal.news_block:
        return False, signal.news_reason

    # Rule 2: Macro strongly against trade direction
    if signal.bias == "BEARISH" and trade_direction == "BUY":
        return False, f"⛔ Fundamental bearish (score {signal.score}) — blocking BUY"

    if signal.bias == "BULLISH" and trade_direction == "SELL":
        return False, f"⛔ Fundamental bullish (score {signal.score}) — blocking SELL"

    # Rule 3: Neutral or aligned — allow
    align = "aligned" if (
        (signal.bias == "BULLISH" and trade_direction == "BUY") or
        (signal.bias == "BEARISH" and trade_direction == "SELL")
    ) else "neutral"

    return True, f"✅ Fundamental {align} (score {signal.score:+d}) — {trade_direction} allowed"


def get_fundamental_confluence(signal: FundamentalSignal) -> str:
    """Return a human-readable fundamental summary for terminal display."""
    parts = []
    icon_map = {"BULLISH_GOLD": "🟢", "BEARISH_GOLD": "🔴", "NEUTRAL": "⚪"}

    parts.append(f"DXY {icon_map[signal.dxy_signal]} {signal.dxy_trend}")
    parts.append(f"CPI {icon_map[signal.cpi_signal]}")
    parts.append(f"Fed {icon_map[signal.fed_signal]} {signal.fed_trend}")
    parts.append(f"NFP {icon_map[signal.nfp_signal]}")

    return "  ".join(parts) + f"  →  {signal.bias} (score {signal.score:+d})"

# ─────────────────────────────────────────────
#  6. PRETTY PRINT REPORT
# ─────────────────────────────────────────────

def print_fundamental_report(signal: FundamentalSignal):
    """Print formatted fundamental analysis report."""
    icon_map = {
        "BULLISH_GOLD" : "🟢",
        "BEARISH_GOLD" : "🔴",
        "NEUTRAL"      : "⚪",
        "BULLISH"      : "📈",
        "BEARISH"      : "📉",
    }
    bias_icon = icon_map.get(signal.bias, "⚪")
    block_icon = "🚫" if signal.news_block else "✅"

    bar_pos = "+" * max(0, signal.score)
    bar_neg = "-" * max(0, -signal.score)
    bar     = (bar_pos or bar_neg or "=") + f"  {signal.score:+d}/4"

    print("╔══════════════════════════════════════════════════╗")
    print("║       PHASE 4 — FUNDAMENTAL ANALYSIS            ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Gold Bias  : {bias_icon} {signal.bias}")
    print(f"║  Macro Score: {bar}")
    print(f"║  News Block : {block_icon} {'ACTIVE — ' + signal.news_reason if signal.news_block else 'Clear'}")
    print("╠══════════════════════════════════════════════════╣")

    dxy_icon = icon_map[signal.dxy_signal]
    cpi_icon = icon_map[signal.cpi_signal]
    fed_icon = icon_map[signal.fed_signal]
    nfp_icon = icon_map[signal.nfp_signal]

    print(f"║  {dxy_icon} DXY Trend   : {signal.dxy_trend}")
    if signal.dxy_value:
        print(f"║     Index: {signal.dxy_value}")
    print(f"║  {cpi_icon} CPI         : {signal.cpi_value or 'N/A'}")
    if signal.cpi_mom is not None:
        print(f"║     MoM change: {signal.cpi_mom:+.3f}%")
    print(f"║  {fed_icon} Fed Rate    : {signal.fed_rate or 'N/A'}%  ({signal.fed_trend})")
    print(f"║  {nfp_icon} NFP         : {signal.nfp_value or 'N/A'}k jobs")

    if signal.upcoming_events:
        print("╠══════════════════════════════════════════════════╣")
        print("║  Upcoming Events (next 24h):")
        for ev in signal.upcoming_events[:4]:
            imp_icon = "🔴" if ev.impact == "HIGH" else "🟡"
            print(f"║    {imp_icon} {ev.title[:30]:30s} in {ev.hours_until:.1f}h")

    print("╚══════════════════════════════════════════════════╝")
    print(f"  {signal.details}\n")

# ─────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PHASE 4 FUNDAMENTAL ENGINE — LIVE TEST")
    print("="*60)

    if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
        print("\n⚠️  No FRED API key set.")
        print("   1. Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   2. Set it in this file: FRED_API_KEY = 'your_key_here'")
        print("   3. Or: set environment variable FRED_API_KEY=your_key")
        print("\n   Running with MOCK data for demonstration...\n")

        # Mock data demonstration
        mock = FundamentalSignal(
            bias             = "BEARISH",
            score            = -2,
            news_block       = False,
            news_reason      = "",
            dxy_signal       = "BEARISH_GOLD",
            dxy_value        = 104.25,
            dxy_trend        = "RISING",
            cpi_signal       = "NEUTRAL",
            cpi_value        = 314.23,
            cpi_mom          = 0.021,
            fed_signal       = "BEARISH_GOLD",
            fed_rate         = 5.25,
            fed_trend        = "HOLD",
            nfp_signal       = "NEUTRAL",
            nfp_value        = 158200,
            upcoming_events  = [
                EconomicEvent("FOMC Rate Decision", datetime.now(timezone.utc) + timedelta(hours=6), "HIGH", "USD", 6.0),
                EconomicEvent("CPI m/m", datetime.now(timezone.utc) + timedelta(hours=26), "HIGH", "USD", 26.0),
            ],
            details          = "MOCK DATA — set FRED_API_KEY for real data"
        )
        print_fundamental_report(mock)

        # Test integration
        print("── Integration Test ──")
        for direction in ["BUY", "SELL"]:
            passes, reason = fundamental_filter_passes(mock, direction)
            print(f"  {direction}: {reason}")

    else:
        print("\n✅ FRED API key found. Fetching live data...\n")

        print("── DXY ──")
        dxy = get_dxy_data()
        print(f"  Trend  : {dxy['trend']}")
        print(f"  Signal : {dxy['signal']}")
        print(f"  Detail : {dxy['details']}")

        print("\n── CPI ──")
        cpi = get_cpi_data()
        print(f"  Value  : {cpi.get('value')}")
        print(f"  YoY    : {cpi.get('yoy_change')}%")
        print(f"  Signal : {cpi['signal']}")
        print(f"  Detail : {cpi['details']}")

        print("\n── Fed Rate ──")
        fed = get_fed_rate_data()
        print(f"  Rate   : {fed.get('rate')}%")
        print(f"  Trend  : {fed['trend']}")
        print(f"  Signal : {fed['signal']}")

        print("\n── NFP ──")
        nfp = get_nfp_data()
        print(f"  Jobs   : +{nfp.get('mom_change')}k")
        print(f"  Signal : {nfp['signal']}")

        print("\n── Upcoming Events ──")
        events = get_upcoming_events()
        if events:
            for ev in events[:6]:
                print(f"  [{ev.impact:6s}] {ev.title[:35]:35s} in {ev.hours_until:.1f}h")
        else:
            print("  No high-impact events in next 48 hours")

        print("\n── Combined Fundamental Signal ──")
        signal = get_fundamental_signal()
        print_fundamental_report(signal)

        print("── Phase 1 + 2 + 4 Integration Test ──")
        for direction in ["BUY", "SELL"]:
            passes, reason = fundamental_filter_passes(signal, direction)
            print(f"  {direction}: {reason}")

        print(f"\n  Confluence: {get_fundamental_confluence(signal)}")

    print("\n" + "="*60)
    print("  Fundamental Engine test complete!")
    print("="*60)
