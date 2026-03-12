import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
import time

# ══════════════════════════════════════════════════════════════════════
#
#   PHASE 1 — TECHNICAL ENGINES FOR XAUUSD
#   ----------------------------------------
#   Engine 1A: RSI Filter
#   Engine 1B: ATR Dynamic SL/TP
#   Engine 1C: Multi-Timeframe (MTF) Trend Filter
#   Engine 1D: Session Time Filter
#   Engine 1E: Swing High / Low Detector
#
#   All engines are modular — plug into any strategy
#
# ══════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────┐
# │                     GLOBAL SETTINGS                             │
# └─────────────────────────────────────────────────────────────────┘
SYMBOL       = "XAUUSD"
TIMEFRAME_H1 = mt5.TIMEFRAME_H1
TIMEFRAME_H4 = mt5.TIMEFRAME_H4
TIMEFRAME_D1 = mt5.TIMEFRAME_D1
NUM_CANDLES  = 300
PRICE_DEC    = 2

# EMA Settings
EMA_FAST     = 9
EMA_SLOW     = 21
EMA_TREND    = 50    # H4 trend filter EMA

# ── Engine 1A: RSI Settings ──
RSI_PERIOD      = 14
RSI_OVERBOUGHT  = 70
RSI_OVERSOLD    = 30
RSI_BULL_MIN    = 50    # RSI must be above this for BUY signals
RSI_BEAR_MAX    = 50    # RSI must be below this for SELL signals

# ── Engine 1B: ATR Settings ──
ATR_PERIOD      = 14
ATR_SL_MULT     = 1.5   # Stop Loss = 1.5 x ATR
ATR_TP_MULT     = 3.0   # Take Profit = 3.0 x ATR (2:1 RR)

# ── Engine 1C: MTF Settings ──
MTF_TIMEFRAME   = TIMEFRAME_H4
MTF_EMA_PERIOD  = 50

# ── Engine 1D: Session Settings (UTC hours) ──
SESSIONS = {
    "Sydney"   : {"start": 22, "end": 7,  "trade": False},
    "Tokyo"    : {"start": 0,  "end": 9,  "trade": False},
    "London"   : {"start": 8,  "end": 17, "trade": True},
    "New York" : {"start": 13, "end": 22, "trade": True},
}
# Best trading window = London-NY Overlap
BEST_START_UTC = 13   # 1 PM UTC
BEST_END_UTC   = 17   # 5 PM UTC

# ── Engine 1E: Swing Settings ──
SWING_LOOKBACK = 5    # Bars each side to confirm swing point


# ══════════════════════════════════════════════════════════════════════
# MT5 CONNECTION
# ══════════════════════════════════════════════════════════════════════
def connect_mt5():
    if not mt5.initialize():
        print(f"❌ MT5 failed: {mt5.last_error()}")
        return False
    return True

def get_data(timeframe=TIMEFRAME_H1, n=NUM_CANDLES):
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, n)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df.index = pd.to_datetime(df["time"], unit="s")
    df.index.name = "time"
    df = df.drop(columns=["time"], errors="ignore")
    return df


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1A — RSI FILTER
#   ─────────────────────────────────────────────────────────────────
#   Purpose: Only allow BUY signals when RSI > 50 (bullish momentum)
#            Only allow SELL signals when RSI < 50 (bearish momentum)
#            Block trades when RSI is overbought (>70) or oversold (<30)
#            Also detects RSI divergence (advanced filter)
# ══════════════════════════════════════════════════════════════════════
def calculate_rsi(df, period=RSI_PERIOD):
    """
    RSI = Relative Strength Index
    Measures momentum — how fast and how much price is moving

    Formula:
      RS  = Average Gain / Average Loss over N periods
      RSI = 100 - (100 / (1 + RS))

    Interpretation:
      RSI > 70  = Overbought  → price moved up too fast, may reverse
      RSI < 30  = Oversold    → price moved down too fast, may reverse
      RSI > 50  = Bullish momentum → favor BUY trades
      RSI < 50  = Bearish momentum → favor SELL trades
    """
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)

    # Use EWM (exponential weighted) for Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi
    return df

def rsi_filter(df):
    """
    Returns signal filtered by RSI:
      +1  = RSI confirms BUY  (RSI > 50, not overbought)
      -1  = RSI confirms SELL (RSI < 50, not oversold)
       0  = RSI blocks trade  (overbought/oversold extreme)
    """
    latest_rsi = df["RSI"].iloc[-1]

    if latest_rsi >= RSI_OVERBOUGHT:
        return 0, latest_rsi, "BLOCKED — RSI Overbought (>{})".format(RSI_OVERBOUGHT)
    elif latest_rsi <= RSI_OVERSOLD:
        return 0, latest_rsi, "BLOCKED — RSI Oversold (<{})".format(RSI_OVERSOLD)
    elif latest_rsi > RSI_BULL_MIN:
        return 1, latest_rsi, "CONFIRMED BUY — RSI bullish ({:.1f})".format(latest_rsi)
    elif latest_rsi < RSI_BEAR_MAX:
        return -1, latest_rsi, "CONFIRMED SELL — RSI bearish ({:.1f})".format(latest_rsi)
    else:
        return 0, latest_rsi, "NEUTRAL — RSI at midpoint ({:.1f})".format(latest_rsi)

def detect_rsi_divergence(df, lookback=20):
    """
    RSI Divergence — one of the most powerful reversal signals

    Bullish Divergence: Price makes LOWER LOW but RSI makes HIGHER LOW
    → Bearish momentum weakening → potential BUY opportunity

    Bearish Divergence: Price makes HIGHER HIGH but RSI makes LOWER HIGH
    → Bullish momentum weakening → potential SELL opportunity
    """
    recent = df.tail(lookback)

    price_low1  = recent["close"].iloc[-1]
    price_low2  = recent["close"].min()
    rsi_low1    = recent["RSI"].iloc[-1]
    rsi_low2    = recent["RSI"].min()

    price_high1 = recent["close"].iloc[-1]
    price_high2 = recent["close"].max()
    rsi_high1   = recent["RSI"].iloc[-1]
    rsi_high2   = recent["RSI"].max()

    bull_div = (price_low1 < price_low2) and (rsi_low1 > rsi_low2)
    bear_div = (price_high1 > price_high2) and (rsi_high1 < rsi_high2)

    if bull_div:
        return "BULLISH DIVERGENCE — potential reversal UP"
    elif bear_div:
        return "BEARISH DIVERGENCE — potential reversal DOWN"
    else:
        return "No divergence detected"


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1B — ATR DYNAMIC SL/TP ENGINE
#   ─────────────────────────────────────────────────────────────────
#   Purpose: Calculate Stop Loss and Take Profit based on actual
#            market volatility instead of fixed dollar amounts.
#            During high volatility (news), SL auto-widens.
#            During low volatility, SL is tighter for better RR.
# ══════════════════════════════════════════════════════════════════════
def calculate_atr(df, period=ATR_PERIOD):
    """
    ATR = Average True Range
    Measures how much the market moves on average per candle

    True Range = max of:
      1. Current High - Current Low
      2. |Current High - Previous Close|
      3. |Current Low  - Previous Close|

    ATR = Average of True Range over N periods
    Gold H1 ATR is typically $3–$8 under normal conditions
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low  - close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    df["ATR"] = atr
    return df

def get_atr_levels(df, signal, entry_price):
    """
    Calculate dynamic SL and TP based on ATR

    For BUY:
      SL = entry - (ATR x SL_multiplier)
      TP = entry + (ATR x TP_multiplier)

    For SELL:
      SL = entry + (ATR x SL_multiplier)
      TP = entry - (ATR x TP_multiplier)

    Returns: sl, tp, atr_value, rr_ratio
    """
    atr = df["ATR"].iloc[-1]

    if signal == "BUY":
        sl = round(entry_price - (atr * ATR_SL_MULT), PRICE_DEC)
        tp = round(entry_price + (atr * ATR_TP_MULT), PRICE_DEC)
    elif signal == "SELL":
        sl = round(entry_price + (atr * ATR_SL_MULT), PRICE_DEC)
        tp = round(entry_price - (atr * ATR_TP_MULT), PRICE_DEC)
    else:
        return None, None, atr, 0

    sl_distance = abs(entry_price - sl)
    tp_distance = abs(entry_price - tp)
    rr_ratio    = round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0

    return sl, tp, round(atr, PRICE_DEC), rr_ratio

def atr_volatility_state(df):
    """
    Classify current market volatility using ATR
    Useful for adjusting strategy behavior during news events
    """
    current_atr = df["ATR"].iloc[-1]
    avg_atr     = df["ATR"].tail(50).mean()
    ratio       = current_atr / avg_atr

    if ratio > 2.0:
        return "EXTREME", current_atr, ratio
    elif ratio > 1.5:
        return "HIGH", current_atr, ratio
    elif ratio > 0.8:
        return "NORMAL", current_atr, ratio
    else:
        return "LOW", current_atr, ratio


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1C — MULTI-TIMEFRAME (MTF) TREND FILTER
#   ─────────────────────────────────────────────────────────────────
#   Purpose: Only trade in the direction of the higher timeframe trend.
#            Uses H4 EMA50 to determine macro bias.
#            BUY signals only when H4 price > H4 EMA50 (bullish)
#            SELL signals only when H4 price < H4 EMA50 (bearish)
#            This is the single most powerful filter to add to any strategy.
# ══════════════════════════════════════════════════════════════════════
def get_mtf_trend():
    """
    Fetch H4 data and calculate EMA50 to determine higher timeframe bias

    Returns:
      "BULLISH" — H4 price above EMA50 → only take BUY signals on H1
      "BEARISH" — H4 price below EMA50 → only take SELL signals on H1
      "NEUTRAL" — price at EMA50 → avoid trading
    """
    df_h4 = get_data(timeframe=TIMEFRAME_H4, n=100)
    if df_h4 is None:
        return "UNKNOWN", None, None

    df_h4["EMA50_H4"] = df_h4["close"].ewm(span=MTF_EMA_PERIOD,
                                            adjust=False).mean()

    latest_close = df_h4["close"].iloc[-1]
    latest_ema50 = df_h4["EMA50_H4"].iloc[-1]
    distance     = round(latest_close - latest_ema50, PRICE_DEC)
    pct_from_ema = round((distance / latest_ema50) * 100, 3)

    if latest_close > latest_ema50 * 1.001:   # 0.1% buffer above EMA
        trend = "BULLISH"
    elif latest_close < latest_ema50 * 0.999:  # 0.1% buffer below EMA
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return trend, round(latest_ema50, PRICE_DEC), pct_from_ema

def get_daily_trend():
    """
    Also check Daily EMA200 for the macro trend
    Strongest filter — only trade with the monthly bias
    """
    df_d1 = get_data(timeframe=TIMEFRAME_D1, n=250)
    if df_d1 is None:
        return "UNKNOWN", None

    df_d1["EMA200_D1"] = df_d1["close"].ewm(span=200, adjust=False).mean()

    latest_close  = df_d1["close"].iloc[-1]
    latest_ema200 = df_d1["EMA200_D1"].iloc[-1]

    trend = "BULLISH" if latest_close > latest_ema200 else "BEARISH"
    return trend, round(latest_ema200, PRICE_DEC)

def mtf_filter(signal, h4_trend, daily_trend):
    """
    Combines H4 and Daily trend to filter H1 signals

    Rules:
      BUY  signal + H4 BULLISH + Daily BULLISH → STRONG BUY ✅
      BUY  signal + H4 BULLISH + Daily BEARISH → WEAK BUY ⚠️
      BUY  signal + H4 BEARISH                 → BLOCKED ❌
      SELL signal + H4 BEARISH + Daily BEARISH → STRONG SELL ✅
      SELL signal + H4 BEARISH + Daily BULLISH → WEAK SELL ⚠️
      SELL signal + H4 BULLISH                 → BLOCKED ❌
    """
    if signal == "BUY":
        if h4_trend == "BULLISH" and daily_trend == "BULLISH":
            return "STRONG BUY", True
        elif h4_trend == "BULLISH":
            return "WEAK BUY", True
        else:
            return "BLOCKED — H4 bearish, no BUY allowed", False

    elif signal == "SELL":
        if h4_trend == "BEARISH" and daily_trend == "BEARISH":
            return "STRONG SELL", True
        elif h4_trend == "BEARISH":
            return "WEAK SELL", True
        else:
            return "BLOCKED — H4 bullish, no SELL allowed", False

    return "NO SIGNAL", False


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1D — SESSION TIME FILTER
#   ─────────────────────────────────────────────────────────────────
#   Purpose: Only allow trades during high-probability trading sessions.
#            Avoid Sydney and Tokyo sessions (low volume, choppy Gold).
#            Focus on London open and London-NY overlap.
#            Automatically blocks trades outside trading hours.
# ══════════════════════════════════════════════════════════════════════
def get_current_session():
    """
    Detect which session(s) are currently active based on UTC time

    Gold session characteristics:
      Sydney   (22:00–07:00 UTC) — Low volume, avoid
      Tokyo    (00:00–09:00 UTC) — Moderate, creates Asia range
      London   (08:00–17:00 UTC) — High volume, sets direction
      New York (13:00–22:00 UTC) — Highest volume, major data releases

    Best window: London-NY overlap 13:00–17:00 UTC
    """
    now_utc  = datetime.now(timezone.utc)
    hour_utc = now_utc.hour
    minute   = now_utc.minute

    active_sessions = []

    # Check each session (handle overnight sessions)
    if 22 <= hour_utc or hour_utc < 7:
        active_sessions.append("Sydney")
    if 0 <= hour_utc < 9:
        active_sessions.append("Tokyo")
    if 8 <= hour_utc < 17:
        active_sessions.append("London")
    if 13 <= hour_utc < 22:
        active_sessions.append("New York")

    # Detect overlaps
    overlap = None
    if "Tokyo" in active_sessions and "London" in active_sessions:
        overlap = "Tokyo-London Overlap (08:00-09:00 UTC)"
    if "London" in active_sessions and "New York" in active_sessions:
        overlap = "London-NY Overlap (13:00-17:00 UTC) ⭐ BEST WINDOW"

    return active_sessions, overlap, hour_utc, minute

def session_filter():
    """
    Returns True if current time is within approved trading hours.

    Approved sessions for XAUUSD:
      London open:    08:00–10:00 UTC (strong directional moves)
      London-NY:      13:00–17:00 UTC (highest probability)
      NY morning:     13:00–16:00 UTC (major data releases)

    Blocked sessions:
      Sydney/Tokyo:   22:00–08:00 UTC (choppy, low volume)
      NY close:       17:00–22:00 UTC (fading moves, reversals)
      Weekend:        Friday 22:00 – Sunday 22:00 UTC
    """
    sessions, overlap, hour_utc, minute = get_current_session()
    now_utc  = datetime.now(timezone.utc)

    # Block weekends
    if now_utc.weekday() >= 5:   # 5=Saturday, 6=Sunday
        return False, "WEEKEND — Markets closed"

    # London open window
    if 8 <= hour_utc < 10:
        return True, "London Open — Good trading window"

    # Best window — London-NY overlap
    if BEST_START_UTC <= hour_utc < BEST_END_UTC:
        return True, "London-NY Overlap — BEST trading window ⭐"

    # NY morning (before overlap ends)
    if 13 <= hour_utc < 17:
        return True, "New York Session — Active trading window"

    return False, f"Outside trading hours (UTC {hour_utc:02d}:{minute:02d}) — Avoid"

def get_session_info():
    """Full session information display"""
    sessions, overlap, hour_utc, minute = get_current_session()
    allowed, reason = session_filter()

    return {
        "utc_time"       : f"{hour_utc:02d}:{minute:02d} UTC",
        "active_sessions": sessions if sessions else ["No major session"],
        "overlap"        : overlap or "No overlap currently",
        "trade_allowed"  : allowed,
        "reason"         : reason,
    }


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1E — SWING HIGH / LOW DETECTOR
#   ─────────────────────────────────────────────────────────────────
#   Purpose: Detect significant swing highs and lows in price.
#            Used for: Structure analysis, SL placement, liquidity mapping
#            A swing high = candle with N lower highs on each side
#            A swing low  = candle with N higher lows on each side
# ══════════════════════════════════════════════════════════════════════
def detect_swing_points(df, lookback=SWING_LOOKBACK):
    """
    Detect swing highs and swing lows

    Swing High: A candle whose high is higher than N candles
                on both left and right side
    Swing Low:  A candle whose low is lower than N candles
                on both left and right side

    These are used for:
      - Structure analysis (higher highs = uptrend)
      - Stop loss placement (put SL below last swing low)
      - Liquidity identification (stops cluster at swing points)
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_highs = []
    swing_lows  = []

    for i in range(lookback, n - lookback):
        # Check swing high
        is_swing_high = all(highs[i] > highs[i-j] for j in range(1, lookback+1)) and \
                        all(highs[i] > highs[i+j] for j in range(1, lookback+1))

        # Check swing low
        is_swing_low  = all(lows[i] < lows[i-j] for j in range(1, lookback+1)) and \
                        all(lows[i] < lows[i+j] for j in range(1, lookback+1))

        if is_swing_high:
            swing_highs.append({
                "time" : df.index[i],
                "price": round(highs[i], PRICE_DEC),
                "index": i
            })

        if is_swing_low:
            swing_lows.append({
                "time" : df.index[i],
                "price": round(lows[i], PRICE_DEC),
                "index": i
            })

    return swing_highs, swing_lows

def analyze_structure(swing_highs, swing_lows):
    """
    Determine market structure from swing points

    Uptrend:   Series of Higher Highs (HH) and Higher Lows (HL)
    Downtrend: Series of Lower Lows  (LL) and Lower Highs (LH)
    Ranging:   No clear pattern
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "INSUFFICIENT DATA"

    # Get last 3 swing highs and lows
    recent_highs = [sh["price"] for sh in swing_highs[-3:]]
    recent_lows  = [sl["price"] for sl in swing_lows[-3:]]

    hh = all(recent_highs[i] > recent_highs[i-1]
             for i in range(1, len(recent_highs)))
    hl = all(recent_lows[i]  > recent_lows[i-1]
             for i in range(1, len(recent_lows)))
    ll = all(recent_lows[i]  < recent_lows[i-1]
             for i in range(1, len(recent_lows)))
    lh = all(recent_highs[i] < recent_highs[i-1]
             for i in range(1, len(recent_highs)))

    if hh and hl:
        return "UPTREND — Higher Highs and Higher Lows"
    elif ll and lh:
        return "DOWNTREND — Lower Lows and Lower Highs"
    elif hh and not hl:
        return "WEAKENING UPTREND — HH but no HL"
    elif ll and not lh:
        return "WEAKENING DOWNTREND — LL but no LH"
    else:
        return "RANGING — No clear structure"

def get_sl_from_structure(signal, swing_highs, swing_lows, buffer=0.5):
    """
    Place stop loss beyond the last swing point — professional method

    BUY trade:  SL below the last swing LOW  (+ small buffer)
    SELL trade: SL above the last swing HIGH (+ small buffer)

    This is better than a fixed SL because it respects market structure
    """
    if signal == "BUY" and swing_lows:
        last_swing_low = swing_lows[-1]["price"]
        sl = round(last_swing_low - buffer, PRICE_DEC)
        return sl, f"Below last swing low at {last_swing_low}"

    elif signal == "SELL" and swing_highs:
        last_swing_high = swing_highs[-1]["price"]
        sl = round(last_swing_high + buffer, PRICE_DEC)
        return sl, f"Above last swing high at {last_swing_high}"

    return None, "No swing point found"


# ══════════════════════════════════════════════════════════════════════
#   COMBINED EMA + ALL ENGINES SIGNAL
#   ─────────────────────────────────────────────────────────────────
#   This combines all 5 engines with the EMA crossover strategy
#   to produce a high-quality filtered signal
# ══════════════════════════════════════════════════════════════════════
def get_ema_signal(df):
    """Original EMA 9/21 crossover signal"""
    df["EMA_9"]  = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_21"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    if prev["EMA_9"] < prev["EMA_21"] and latest["EMA_9"] > latest["EMA_21"]:
        return "BUY"
    elif prev["EMA_9"] > prev["EMA_21"] and latest["EMA_9"] < latest["EMA_21"]:
        return "SELL"
    else:
        return "HOLD"

def run_full_analysis():
    """
    Run all 5 engines together and produce a comprehensive signal report
    """
    print("=" * 65)
    print("   XAUUSD — PHASE 1 TECHNICAL ENGINE ANALYSIS")
    print("=" * 65)

    # Connect
    if not connect_mt5():
        return

    # Fetch H1 data
    df = get_data(TIMEFRAME_H1, NUM_CANDLES)
    if df is None:
        print("❌ No data received")
        mt5.shutdown()
        return

    account = mt5.account_info()
    print(f"✅ Connected: {account.login} | Balance: ${account.balance:,.2f}")
    print(f"✅ Symbol   : {SYMBOL}")
    print(f"✅ Candles  : {len(df)} H1 bars loaded")
    print()

    # ── Calculate all indicators ──
    df = calculate_rsi(df)
    df = calculate_atr(df)

    latest       = df.iloc[-1]
    current_price = latest["close"]
    fmt = f"{{:.{PRICE_DEC}f}}"

    # ── Raw EMA Signal ──
    ema_signal = get_ema_signal(df)

    print("─" * 65)
    print(f"  💲 Current Price : {fmt.format(current_price)}")
    print(f"  📡 EMA Signal    : {ema_signal}")
    print("─" * 65)

    # ─────────────────────────────────────────
    # ENGINE 1A — RSI
    # ─────────────────────────────────────────
    print("\n  🔵 ENGINE 1A — RSI FILTER")
    print(f"  {'─'*60}")
    rsi_result, rsi_val, rsi_msg = rsi_filter(df)
    divergence = detect_rsi_divergence(df)
    print(f"  RSI ({RSI_PERIOD})    : {rsi_val:.2f}")
    print(f"  RSI Signal  : {rsi_msg}")
    print(f"  Divergence  : {divergence}")
    print(f"  Overbought  : {RSI_OVERBOUGHT} | Oversold: {RSI_OVERSOLD}")

    # ─────────────────────────────────────────
    # ENGINE 1B — ATR
    # ─────────────────────────────────────────
    print("\n  🟡 ENGINE 1B — ATR DYNAMIC SL/TP")
    print(f"  {'─'*60}")
    sl, tp, atr_val, rr = get_atr_levels(df, ema_signal, current_price)
    vol_state, vol_atr, vol_ratio = atr_volatility_state(df)
    print(f"  ATR ({ATR_PERIOD})    : ${atr_val}")
    print(f"  Volatility  : {vol_state} ({vol_ratio:.2f}x average)")
    if sl and tp:
        print(f"  Stop Loss   : {fmt.format(sl)} (ATR x {ATR_SL_MULT})")
        print(f"  Take Profit : {fmt.format(tp)} (ATR x {ATR_TP_MULT})")
        print(f"  R:R Ratio   : 1:{rr}")
    else:
        print(f"  SL/TP       : N/A (no active signal)")

    # ─────────────────────────────────────────
    # ENGINE 1C — MTF TREND FILTER
    # ─────────────────────────────────────────
    print("\n  🟠 ENGINE 1C — MULTI-TIMEFRAME TREND FILTER")
    print(f"  {'─'*60}")
    h4_trend, h4_ema50, h4_pct = get_mtf_trend()
    d1_trend, d1_ema200        = get_daily_trend()
    mtf_result, allowed        = mtf_filter(ema_signal, h4_trend, d1_trend)
    print(f"  H4 EMA50    : {fmt.format(h4_ema50) if h4_ema50 else 'N/A'} | Trend: {h4_trend} ({h4_pct}% from EMA)")
    print(f"  D1 EMA200   : {fmt.format(d1_ema200) if d1_ema200 else 'N/A'} | Trend: {d1_trend}")
    print(f"  MTF Result  : {mtf_result}")
    print(f"  Trade Allow : {'✅ YES' if allowed else '❌ NO'}")

    # ─────────────────────────────────────────
    # ENGINE 1D — SESSION FILTER
    # ─────────────────────────────────────────
    print("\n  🟣 ENGINE 1D — SESSION TIME FILTER")
    print(f"  {'─'*60}")
    session_info = get_session_info()
    print(f"  UTC Time    : {session_info['utc_time']}")
    print(f"  Sessions    : {', '.join(session_info['active_sessions'])}")
    print(f"  Overlap     : {session_info['overlap']}")
    print(f"  Trade Allow : {'✅ YES' if session_info['trade_allowed'] else '❌ NO'}")
    print(f"  Reason      : {session_info['reason']}")

    # ─────────────────────────────────────────
    # ENGINE 1E — SWING POINTS
    # ─────────────────────────────────────────
    print("\n  🔴 ENGINE 1E — SWING HIGH/LOW & STRUCTURE")
    print(f"  {'─'*60}")
    swing_highs, swing_lows = detect_swing_points(df)
    structure = analyze_structure(swing_highs, swing_lows)
    struct_sl, struct_sl_reason = get_sl_from_structure(
        ema_signal, swing_highs, swing_lows)

    print(f"  Swing Highs : {len(swing_highs)} detected")
    print(f"  Swing Lows  : {len(swing_lows)} detected")
    if swing_highs:
        print(f"  Last Sw.High: {fmt.format(swing_highs[-1]['price'])} at "
              f"{swing_highs[-1]['time'].strftime('%m-%d %H:%M')}")
    if swing_lows:
        print(f"  Last Sw.Low : {fmt.format(swing_lows[-1]['price'])} at "
              f"{swing_lows[-1]['time'].strftime('%m-%d %H:%M')}")
    print(f"  Structure   : {structure}")
    if struct_sl:
        print(f"  Structure SL: {fmt.format(struct_sl)} ({struct_sl_reason})")

    # ─────────────────────────────────────────
    # FINAL COMBINED SIGNAL
    # ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  🏆 FINAL COMBINED SIGNAL — ALL ENGINES")
    print("=" * 65)

    # All filters must pass
    rsi_pass     = rsi_result != 0
    mtf_pass     = allowed
    session_pass = session_info["trade_allowed"]

    engines_passed = sum([rsi_pass, mtf_pass, session_pass])
    engines_total  = 3

    if ema_signal == "HOLD":
        final = "⏳ HOLD — No EMA crossover signal"
        confidence = "N/A"
    elif engines_passed == 3:
        final = f"{'🟢 STRONG ' + ema_signal if mtf_result.startswith('STRONG') else '✅ ' + ema_signal}"
        confidence = "HIGH"
    elif engines_passed == 2:
        final = f"⚠️  WEAK {ema_signal} — some filters failed"
        confidence = "MEDIUM"
    else:
        final = f"❌ BLOCKED — too many filters failed"
        confidence = "LOW — SKIP THIS TRADE"

    print(f"\n  EMA Signal    : {ema_signal}")
    print(f"  RSI Filter    : {'✅ PASS' if rsi_pass else '❌ FAIL'} "
          f"(RSI: {rsi_val:.1f})")
    print(f"  MTF Filter    : {'✅ PASS' if mtf_pass else '❌ FAIL'} "
          f"(H4: {h4_trend})")
    print(f"  Session Filter: {'✅ PASS' if session_pass else '❌ FAIL'} "
          f"({session_info['utc_time']})")
    print(f"\n  Engines Passed: {engines_passed}/{engines_total}")
    print(f"  Confidence    : {confidence}")
    print(f"\n  ▶ FINAL SIGNAL: {final}")

    if ema_signal != "HOLD" and sl and tp:
        print(f"\n  📊 Trade Setup (if taking):")
        print(f"     Entry  : {fmt.format(current_price)}")
        print(f"     SL     : {fmt.format(sl)} (ATR-based)")
        print(f"     TP     : {fmt.format(tp)} (ATR-based)")
        print(f"     R:R    : 1:{rr}")
        if struct_sl:
            print(f"     Alt SL : {fmt.format(struct_sl)} (Structure-based)")

    print("=" * 65)
    mt5.shutdown()
    print("✅ Analysis complete. MT5 disconnected.")


# ══════════════════════════════════════════════════════════════════════
#   CHART — Visualize all engines on one chart
# ══════════════════════════════════════════════════════════════════════
def draw_engine_chart():
    """
    Draw a 3-panel chart showing:
    Panel 1: Price + EMA9/21 + Swing Points
    Panel 2: RSI with overbought/oversold levels
    Panel 3: ATR volatility
    """
    if not connect_mt5():
        return

    df = get_data(TIMEFRAME_H1, 150)
    if df is None:
        mt5.shutdown()
        return

    df = calculate_rsi(df)
    df = calculate_atr(df)
    df["EMA_9"]  = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_21"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # Signals
    df["Position"] = np.where(df["EMA_9"] > df["EMA_21"], 1, -1)
    df["Signal"]   = df["Position"].diff()
    buys  = df[df["Signal"] > 0]
    sells = df[df["Signal"] < 0]

    swing_highs, swing_lows = detect_swing_points(df)

    mt5.shutdown()

    # ── Chart ──
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12),
                                         gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.patch.set_facecolor("#0D1117")
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#8B949E")
        ax.grid(color="#21262D", linestyle="--", linewidth=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    # Panel 1 — Price + EMAs + Swings
    ax1.plot(df.index, df["close"],  color="#8B949E", lw=1.0,
             label="XAUUSD", alpha=0.8)
    ax1.plot(df.index, df["EMA_9"],  color="#58A6FF", lw=2.0,
             label=f"EMA {EMA_FAST}")
    ax1.plot(df.index, df["EMA_21"], color="#E3B341", lw=2.0,
             label=f"EMA {EMA_SLOW}")

    ax1.fill_between(df.index, df["EMA_9"], df["EMA_21"],
                     where=df["EMA_9"] >= df["EMA_21"],
                     alpha=0.08, color="#2EA043")
    ax1.fill_between(df.index, df["EMA_9"], df["EMA_21"],
                     where=df["EMA_9"] < df["EMA_21"],
                     alpha=0.08, color="#DA3633")

    ax1.scatter(buys.index,  buys["EMA_9"],  marker="^",
                color="#2EA043", s=150, zorder=5, label="BUY Signal")
    ax1.scatter(sells.index, sells["EMA_9"], marker="v",
                color="#DA3633", s=150, zorder=5, label="SELL Signal")

    # Swing points
    for sh in swing_highs[-8:]:
        ax1.axhline(y=sh["price"], color="#FF6B6B", lw=0.5,
                    linestyle=":", alpha=0.6)
    for sl in swing_lows[-8:]:
        ax1.axhline(y=sl["price"], color="#51CF66", lw=0.5,
                    linestyle=":", alpha=0.6)

    ax1.set_title(f"{SYMBOL} — Phase 1 Technical Engines | "
                  f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
                  color="#F0F6FC", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (USD)", color="#8B949E")
    ax1.legend(loc="upper left", facecolor="#21262D",
               edgecolor="#30363D", labelcolor="#F0F6FC", fontsize=9)

    # Panel 2 — RSI
    ax2.plot(df.index, df["RSI"], color="#C792EA", lw=1.5, label="RSI(14)")
    ax2.axhline(y=RSI_OVERBOUGHT, color="#DA3633", lw=1.0,
                linestyle="--", alpha=0.8, label=f"Overbought ({RSI_OVERBOUGHT})")
    ax2.axhline(y=RSI_OVERSOLD,   color="#2EA043", lw=1.0,
                linestyle="--", alpha=0.8, label=f"Oversold ({RSI_OVERSOLD})")
    ax2.axhline(y=50, color="#8B949E", lw=0.8, linestyle="--", alpha=0.5)
    ax2.fill_between(df.index, df["RSI"], 70,
                     where=df["RSI"] >= 70, alpha=0.15, color="#DA3633")
    ax2.fill_between(df.index, df["RSI"], 30,
                     where=df["RSI"] <= 30, alpha=0.15, color="#2EA043")
    ax2.set_ylabel("RSI", color="#8B949E")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", facecolor="#21262D",
               edgecolor="#30363D", labelcolor="#F0F6FC", fontsize=8)

    # Panel 3 — ATR
    ax3.plot(df.index, df["ATR"], color="#E3B341", lw=1.5, label="ATR(14)")
    avg_atr = df["ATR"].mean()
    ax3.axhline(y=avg_atr, color="#58A6FF", lw=1.0,
                linestyle="--", alpha=0.7, label=f"Avg ATR ({avg_atr:.2f})")
    ax3.fill_between(df.index, df["ATR"], avg_atr,
                     where=df["ATR"] >= avg_atr, alpha=0.15, color="#E3B341")
    ax3.set_ylabel("ATR", color="#8B949E")
    ax3.set_xlabel("Date & Time", color="#8B949E")
    ax3.legend(loc="upper left", facecolor="#21262D",
               edgecolor="#30363D", labelcolor="#F0F6FC", fontsize=8)

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
#   ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nRunning all Phase 1 Technical Engines...\n")
    run_full_analysis()
    print("\nOpening 3-panel engine chart...")
    draw_engine_chart()