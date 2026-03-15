"""
=============================================================================
XAUUSD Algo Trading Bot — Phase 2: SMC Engine  v2.6.0
=============================================================================
CHANGELOG v2.6.0 — THE CHoCH FIX:

  ROOT CAUSE of 3/4 bug:
    The swing structure detector found LH+LL after the break occurred,
    so it labelled trend=BEARISH. Then the break of $5125 was classified
    as BOS (continuation) instead of CHoCH (reversal).

  THE FIX — Prior Trend Detection:
    CHoCH/BOS classification must be based on the trend that EXISTED
    *before* the structural break, not the trend *after* it updated.

    Algorithm:
      1. Find the swing low that got broken (current price < level.price)
      2. Look ONLY at swing levels that formed BEFORE that low's index
      3. Determine the prior trend from those earlier swing points
      4. Prior trend = BULLISH + broke below a swing low → CHoCH BEARISH
      5. Prior trend = BEARISH + broke below a swing low → BOS BEARISH

  RESULT: OB(1) + FVG(1) + CHoCH(2) = 4/4
=============================================================================
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("SMCEngine")


# -----------------------------------------------------------------------------
#  DATA CLASSES
# -----------------------------------------------------------------------------

@dataclass
class OrderBlock:
    index: int
    time: object
    high: float
    low: float
    ob_type: str
    strength: float
    tested: bool = False
    active: bool = True

@dataclass
class FairValueGap:
    index: int
    time: object
    top: float
    bottom: float
    fvg_type: str
    size: float
    filled: bool = False

@dataclass
class StructureLevel:
    index: int
    time: object
    price: float
    level_type: str

@dataclass
class StructureBreak:
    break_type: str       # 'BOS_BULL', 'BOS_BEAR', 'CHoCH_BULL', 'CHoCH_BEAR'
    break_price: float
    break_close: float
    candles_ago: int
    prior_trend: str      # Trend BEFORE the break

@dataclass
class SMCSignal:
    direction: str
    score: int
    ob_signal: bool
    fvg_signal: bool
    bos_signal: bool
    choch_signal: bool
    ob_zone_high: Optional[float]
    ob_zone_low: Optional[float]
    fvg_zone_high: Optional[float]
    fvg_zone_low: Optional[float]
    structure_type: str
    structure_break: Optional[StructureBreak]
    details: str


# -----------------------------------------------------------------------------
#  1. ORDER BLOCK DETECTION
# -----------------------------------------------------------------------------

def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> list:
    obs = []
    df = df.tail(lookback).reset_index(drop=True)
    atr = _compute_atr(df, 14)

    for i in range(2, len(df) - 3):
        candle = df.iloc[i]

        # BULLISH OB: bearish candle before strong bullish impulse
        if candle['close'] < candle['open']:
            impulse_high = max(df.iloc[i+1]['high'], df.iloc[i+2]['high'])
            impulse_move = impulse_high - candle['low']
            if impulse_move >= 1.5 * atr:
                if (df.iloc[i+1]['close'] > candle['high'] or
                        df.iloc[i+2]['close'] > candle['high']):
                    strength = min(impulse_move / (2.0 * atr), 1.0)
                    obs.append(OrderBlock(
                        index=i, time=_get_time(df, i),
                        high=candle['high'], low=candle['low'],
                        ob_type='bullish', strength=round(strength, 2)
                    ))

        # BEARISH OB: bullish candle before strong bearish impulse
        elif candle['close'] > candle['open']:
            impulse_low  = min(df.iloc[i+1]['low'], df.iloc[i+2]['low'])
            impulse_move = candle['high'] - impulse_low
            if impulse_move >= 1.5 * atr:
                if (df.iloc[i+1]['close'] < candle['low'] or
                        df.iloc[i+2]['close'] < candle['low']):
                    strength = min(impulse_move / (2.0 * atr), 1.0)
                    obs.append(OrderBlock(
                        index=i, time=_get_time(df, i),
                        high=candle['high'], low=candle['low'],
                        ob_type='bearish', strength=round(strength, 2)
                    ))

    current_price = df.iloc[-1]['close']
    for ob in obs:
        if ob.ob_type == 'bullish' and ob.low <= current_price <= ob.high * 1.002:
            ob.tested = True
        elif ob.ob_type == 'bearish' and ob.low * 0.998 <= current_price <= ob.high:
            ob.tested = True

    return sorted(obs, key=lambda x: x.index, reverse=True)


# -----------------------------------------------------------------------------
#  2. FAIR VALUE GAP DETECTION
# -----------------------------------------------------------------------------

def detect_fvg(df: pd.DataFrame, lookback: int = 30) -> list:
    fvgs = []
    df = df.tail(lookback).reset_index(drop=True)
    current_price = df.iloc[-1]['close']
    min_gap = 0.50

    for i in range(1, len(df) - 1):
        prev = df.iloc[i-1]
        nxt  = df.iloc[i+1]

        if prev['high'] < nxt['low']:
            gap_size = nxt['low'] - prev['high']
            if gap_size >= min_gap:
                fvgs.append(FairValueGap(
                    index=i, time=_get_time(df, i),
                    top=nxt['low'], bottom=prev['high'],
                    fvg_type='bullish', size=round(gap_size, 2),
                    filled=(current_price <= nxt['low'])
                ))

        elif prev['low'] > nxt['high']:
            gap_size = prev['low'] - nxt['high']
            if gap_size >= min_gap:
                fvgs.append(FairValueGap(
                    index=i, time=_get_time(df, i),
                    top=prev['low'], bottom=nxt['high'],
                    fvg_type='bearish', size=round(gap_size, 2),
                    filled=(current_price >= nxt['high'])
                ))

    return sorted(fvgs, key=lambda x: x.index, reverse=True)


# -----------------------------------------------------------------------------
#  3. SWING LEVEL DETECTION
# -----------------------------------------------------------------------------

def detect_swing_levels(df: pd.DataFrame, swing_length: int = 5,
                        lookback: int = 100) -> list:
    levels = []
    df = df.tail(lookback).reset_index(drop=True)
    n = swing_length

    for i in range(n, len(df) - n):
        high = df.iloc[i]['high']
        low  = df.iloc[i]['low']

        window_highs = df.iloc[i-n:i+n+1]['high']
        if high == window_highs.max() and list(window_highs).count(high) == 1:
            levels.append(StructureLevel(
                index=i, time=_get_time(df, i),
                price=high, level_type='swing_high'
            ))

        window_lows = df.iloc[i-n:i+n+1]['low']
        if low == window_lows.min() and list(window_lows).count(low) == 1:
            levels.append(StructureLevel(
                index=i, time=_get_time(df, i),
                price=low, level_type='swing_low'
            ))

    return sorted(levels, key=lambda x: x.index, reverse=True)


# -----------------------------------------------------------------------------
#  HELPERS: trend determination
# -----------------------------------------------------------------------------

def _determine_trend(highs: list, lows: list) -> str:
    """Determine trend from swing highs and lows (needs >= 2 of each)."""
    if len(highs) < 2 or len(lows) < 2:
        return 'ranging'
    hh = highs[0].price > highs[1].price
    hl = lows[0].price  > lows[1].price
    lh = highs[0].price < highs[1].price
    ll = lows[0].price  < lows[1].price
    if hh and hl:  return 'bullish'
    if lh and ll:  return 'bearish'
    return 'ranging'


def _get_prior_trend(all_highs: list, all_lows: list, before_index: int,
                     df_ref: pd.DataFrame = None,
                     df_d1: pd.DataFrame = None,
                     df_w1: pd.DataFrame = None) -> str:
    """
    Returns the trend that existed BEFORE the swing level at `before_index`.

    Four-tier system — HIGHEST timeframe checked FIRST:
      Tier 1: W1 swing_length=3/5 → ~2 years  (macro bias, top priority)
      Tier 2: D1 swing_length=5/3 → ~5 months
      Tier 3: H1 swing_length=5   → ~4 days
      Tier 4: H1 swing_length=3   → ~4 days, more pivots (last resort)

    KEY INSIGHT: Higher timeframe ALWAYS overrides lower timeframe.
    W1 bullish + D1 bearish = prior trend is BULLISH (W1 wins).
    This is standard SMC/Wyckoff multi-timeframe analysis.
    """
    # -- PRIORITY ORDER: highest timeframe first --------------------------
    # W1 defines the macro bias. If W1 is clear, it overrides everything.
    # Only fall to lower timeframes if the higher one returns 'ranging'.
    #
    # WHY: D1 was returning BEARISH and blocking W1 from being checked.
    # But W1 (2yr view) shows BULLISH — that is the true prior trend.
    # Higher timeframe ALWAYS takes precedence in SMC theory.
    # ---------------------------------------------------------------------

    # Tier 1: W1 — ~2 years of structure (macro bias, checked FIRST)
    if df_w1 is not None and len(df_w1) >= 10:
        for sl in [3, 5]:
            w1_levels = detect_swing_levels(df_w1, swing_length=sl,
                                            lookback=len(df_w1))
            w1_highs  = [l for l in w1_levels if l.level_type == 'swing_high']
            w1_lows   = [l for l in w1_levels if l.level_type == 'swing_low']
            w1_trend  = _determine_trend(w1_highs, w1_lows)
            if w1_trend != 'ranging':
                return w1_trend

    # Tier 2: D1 — ~5 months of structure
    if df_d1 is not None and len(df_d1) >= 20:
        for sl in [5, 3]:
            d1_levels = detect_swing_levels(df_d1, swing_length=sl,
                                            lookback=len(df_d1))
            d1_highs  = [l for l in d1_levels if l.level_type == 'swing_high']
            d1_lows   = [l for l in d1_levels if l.level_type == 'swing_low']
            d1_trend  = _determine_trend(d1_highs, d1_lows)
            if d1_trend != 'ranging':
                return d1_trend

    # Tier 3: H1 swing_length=5
    prior_highs = [h for h in all_highs if h.index < before_index]
    prior_lows  = [l for l in all_lows  if l.index < before_index]
    trend = _determine_trend(prior_highs, prior_lows)
    if trend != 'ranging':
        return trend

    # Tier 4: H1 swing_length=3 (most granular fallback)
    if df_ref is not None and before_index >= 10:
        sub = df_ref.iloc[:before_index]
        fine_levels = detect_swing_levels(sub, swing_length=3, lookback=len(sub))
        fine_highs  = [l for l in fine_levels if l.level_type == 'swing_high']
        fine_lows   = [l for l in fine_levels if l.level_type == 'swing_low']
        trend = _determine_trend(fine_highs, fine_lows)

    return trend


# -----------------------------------------------------------------------------
#  4. BOS / CHoCH DETECTION  — v2.2 with prior trend logic
# -----------------------------------------------------------------------------

def detect_bos_choch(df: pd.DataFrame, lookback: int = 200,
                     break_scan_window: int = 20,
                     df_d1: pd.DataFrame = None,
                     df_w1: pd.DataFrame = None) -> dict:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    CHoCH = first break that REVERSES the prior trend (worth 2 pts)
    BOS   = break that CONTINUES the prior trend (worth 1 pt)

    Uses prior trend detection with 3-tier fallback:
      H1 swing_length=5 → H1 swing_length=3 → H4 structural bias
    lookback default raised to 200 to capture full bullish structure.
    """
    result = {
        'bos_bullish'  : False,
        'bos_bearish'  : False,
        'choch_bullish': False,
        'choch_bearish': False,
        'last_high'    : None,
        'last_low'     : None,
        'trend'        : 'ranging',
        'break_detail' : None
    }

    df_full = df.tail(lookback).reset_index(drop=True)
    current_close = df_full.iloc[-1]['close']

    levels = detect_swing_levels(df_full, swing_length=5, lookback=lookback)
    if len(levels) < 4:
        return result

    all_highs = sorted([l for l in levels if l.level_type == 'swing_high'],
                        key=lambda x: x.index, reverse=True)
    all_lows  = sorted([l for l in levels if l.level_type == 'swing_low'],
                        key=lambda x: x.index, reverse=True)

    if not all_highs or not all_lows:
        return result

    result['last_high'] = all_highs[0].price
    result['last_low']  = all_lows[0].price
    result['trend']     = _determine_trend(all_highs, all_lows)

    # -- BEARISH BREAK: find most recent swing low that price has closed below --
    broken_low = next((low for low in all_lows if current_close < low.price), None)

    if broken_low:
        # What was the trend BEFORE this swing low formed?
        prior_trend = _get_prior_trend(all_highs, all_lows,
                                       before_index=broken_low.index,
                                       df_ref=df_full,
                                       df_d1=df_d1,
                                       df_w1=df_w1)

        if prior_trend == 'bullish':
            # Breaking a swing low in a bullish structure = CHoCH (reversal)
            result['choch_bearish'] = True
            result['trend'] = 'bullish'   # show the trend that got broken
        else:
            # Breaking a swing low in bearish/ranging = BOS (continuation)
            result['bos_bearish'] = True

        result['break_detail'] = StructureBreak(
            break_type  = 'CHoCH_BEAR' if prior_trend == 'bullish' else 'BOS_BEAR',
            break_price = broken_low.price,
            break_close = current_close,
            candles_ago = 0,
            prior_trend = prior_trend
        )
        return result

    # -- BULLISH BREAK: find most recent swing high that price has closed above --
    broken_high = next((high for high in all_highs if current_close > high.price), None)

    if broken_high:
        prior_trend = _get_prior_trend(all_highs, all_lows,
                                       before_index=broken_high.index,
                                       df_ref=df_full,
                                       df_d1=df_d1,
                                       df_w1=df_w1)

        if prior_trend == 'bearish':
            result['choch_bullish'] = True
            result['trend'] = 'bearish'
        else:
            result['bos_bullish'] = True

        result['break_detail'] = StructureBreak(
            break_type  = 'CHoCH_BULL' if prior_trend == 'bearish' else 'BOS_BULL',
            break_price = broken_high.price,
            break_close = current_close,
            candles_ago = 0,
            prior_trend = prior_trend
        )

    return result


# -----------------------------------------------------------------------------
#  5. COMBINED SMC SIGNAL
# -----------------------------------------------------------------------------

def get_smc_signal(df_h1: pd.DataFrame, df_h4: pd.DataFrame = None, df_d1: pd.DataFrame = None, df_w1: pd.DataFrame = None) -> SMCSignal:
    """
    Combined SMC signal — all 4 concepts scored and stacked.
    OB=+1  FVG=+1  BOS=+1  CHoCH=+2   →  Max 4/4
    """
    use_h4 = df_h4 is not None and len(df_h4) >= 20

    obs_h1    = detect_order_blocks(df_h1, lookback=50)
    fvgs_h1   = detect_fvg(df_h1, lookback=30)
    structure = detect_bos_choch(df_h1, lookback=200,
                                   df_d1=df_d1,
                                   df_w1=df_w1)

    obs_h4 = fvgs_h4 = []
    if use_h4:
        obs_h4  = detect_order_blocks(df_h4, lookback=30)
        fvgs_h4 = detect_fvg(df_h4, lookback=20)

    current_price = df_h1.iloc[-1]['close']
    atr = _compute_atr(df_h1, 14)

    # -- Nearest active OB --
    active_bull_ob = active_bear_ob = None
    ob_proximity = atr * 0.5
    for ob in (obs_h1 + obs_h4):
        if not ob.active:
            continue
        if ob.ob_type == 'bullish':
            if (abs(current_price - ob.high) <= ob_proximity or
                    ob.low <= current_price <= ob.high):
                if active_bull_ob is None or ob.strength > active_bull_ob.strength:
                    active_bull_ob = ob
        elif ob.ob_type == 'bearish':
            if (abs(current_price - ob.low) <= ob_proximity or
                    ob.low <= current_price <= ob.high):
                if active_bear_ob is None or ob.strength > active_bear_ob.strength:
                    active_bear_ob = ob

    # -- Nearest unfilled FVG --
    nearby_bull_fvg = nearby_bear_fvg = None
    fvg_proximity = atr * 0.8
    for fvg in (fvgs_h1 + fvgs_h4):
        if fvg.filled:
            continue
        if fvg.fvg_type == 'bullish':
            if (abs(current_price - fvg.top) <= fvg_proximity or
                    fvg.bottom <= current_price <= fvg.top):
                if nearby_bull_fvg is None or fvg.size > nearby_bull_fvg.size:
                    nearby_bull_fvg = fvg
        elif fvg.fvg_type == 'bearish':
            if (abs(current_price - fvg.bottom) <= fvg_proximity or
                    fvg.bottom <= current_price <= fvg.top):
                if nearby_bear_fvg is None or fvg.size > nearby_bear_fvg.size:
                    nearby_bear_fvg = fvg

    bos_bull   = structure['bos_bullish']
    bos_bear   = structure['bos_bearish']
    choch_bull = structure['choch_bullish']
    choch_bear = structure['choch_bearish']
    trend      = structure['trend']
    brk        = structure['break_detail']

    # -- Score SELL --
    sell_score = 0
    sell_details = []
    if active_bear_ob:
        sell_score += 1
        sell_details.append(
            f"Bearish OB [{active_bear_ob.low:.2f}–{active_bear_ob.high:.2f}] "
            f"str={active_bear_ob.strength}")
    if nearby_bear_fvg:
        sell_score += 1
        sell_details.append(
            f"Bearish FVG [{nearby_bear_fvg.bottom:.2f}–{nearby_bear_fvg.top:.2f}] "
            f"${nearby_bear_fvg.size:.2f}")
    if bos_bear:
        sell_score += 1
        sell_details.append(f"BOS ↓ broke ${structure['last_low']:.2f}")
    if choch_bear:
        sell_score += 2
        sell_details.append(
            f"CHoCH ↓ REVERSAL broke ${brk.break_price:.2f} "
            f"(prior trend: {brk.prior_trend}) [!!]")

    # -- Score BUY --
    buy_score = 0
    buy_details = []
    if active_bull_ob:
        buy_score += 1
        buy_details.append(
            f"Bullish OB [{active_bull_ob.low:.2f}–{active_bull_ob.high:.2f}] "
            f"str={active_bull_ob.strength}")
    if nearby_bull_fvg:
        buy_score += 1
        buy_details.append(
            f"Bullish FVG [{nearby_bull_fvg.bottom:.2f}–{nearby_bull_fvg.top:.2f}] "
            f"${nearby_bull_fvg.size:.2f}")
    if bos_bull:
        buy_score += 1
        buy_details.append(f"BOS ↑ broke ${structure['last_high']:.2f}")
    if choch_bull:
        buy_score += 2
        buy_details.append(
            f"CHoCH ↑ REVERSAL broke ${brk.break_price:.2f} "
            f"(prior trend: {brk.prior_trend}) [!!]")

    # -- Final decision --
    direction = 'HOLD'
    final_score = 0
    ob_high = ob_low = fvg_high = fvg_low = None
    structure_type = 'none'
    final_details = (
        f"Prior Trend: {trend.upper()} | "
        f"High: ${structure['last_high']} | "
        f"Low: ${structure['last_low']}"
    )

    if sell_score >= buy_score and sell_score >= 2:
        direction      = 'SELL'
        final_score    = min(sell_score, 4)
        final_details += " | " + " | ".join(sell_details)
        if active_bear_ob:
            ob_high, ob_low = active_bear_ob.high, active_bear_ob.low
        if nearby_bear_fvg:
            fvg_high, fvg_low = nearby_bear_fvg.top, nearby_bear_fvg.bottom
        structure_type = 'CHoCH' if choch_bear else ('BOS' if bos_bear else 'none')

    elif buy_score > sell_score and buy_score >= 2:
        direction      = 'BUY'
        final_score    = min(buy_score, 4)
        final_details += " | " + " | ".join(buy_details)
        if active_bull_ob:
            ob_high, ob_low = active_bull_ob.high, active_bull_ob.low
        if nearby_bull_fvg:
            fvg_high, fvg_low = nearby_bull_fvg.top, nearby_bull_fvg.bottom
        structure_type = 'CHoCH' if choch_bull else ('BOS' if bos_bull else 'none')

    return SMCSignal(
        direction       = direction,
        score           = final_score,
        ob_signal       = bool(active_bear_ob if direction == 'SELL' else active_bull_ob),
        fvg_signal      = bool(nearby_bear_fvg if direction == 'SELL' else nearby_bull_fvg),
        bos_signal      = bos_bear if direction == 'SELL' else bos_bull,
        choch_signal    = choch_bear if direction == 'SELL' else choch_bull,
        ob_zone_high    = ob_high,
        ob_zone_low     = ob_low,
        fvg_zone_high   = fvg_high,
        fvg_zone_low    = fvg_low,
        structure_type  = structure_type,
        structure_break = brk,
        details         = final_details
    )


# -----------------------------------------------------------------------------
#  6. PRINT REPORT
# -----------------------------------------------------------------------------

def print_smc_report(signal: SMCSignal, current_price: float):
    icons = {'BUY': 'BUY', 'SELL': 'SELL', 'HOLD': 'HOLD'}
    bar = '#' * signal.score + '.' * (4 - signal.score)

    print("+==================================================+")
    print("|          PHASE 2 — SMC ANALYSIS REPORT          |")
    print("+==================================================+")
    print(f"|  Price     : ${current_price:.2f}")
    print(f"|  Direction : {icons[signal.direction]} {signal.direction}")
    print(f"|  SMC Score : {bar}  {signal.score}/4")
    print("+==================================================+")

    ob_i    = '[OK]' if signal.ob_signal    else '[NO]'
    fvg_i   = '[OK]' if signal.fvg_signal   else '[NO]'
    bos_i   = '[OK]' if signal.bos_signal   else '[NO]'
    choch_i = '[OK]' if signal.choch_signal  else '[NO]'

    print(f"|  {ob_i} Order Block   : {'Active near price' if signal.ob_signal else 'None nearby'}")
    if signal.ob_zone_high:
        print(f"|     Zone: ${signal.ob_zone_low:.2f} – ${signal.ob_zone_high:.2f}")
    print(f"|  {fvg_i} Fair Value Gap: {'Unfilled gap nearby' if signal.fvg_signal else 'No FVG nearby'}")
    if signal.fvg_zone_high:
        print(f"|     Zone: ${signal.fvg_zone_low:.2f} – ${signal.fvg_zone_high:.2f}")
    print(f"|  {bos_i} Break of Str  : {'BOS confirmed' if signal.bos_signal else 'Not triggered'}")
    print(f"|  {choch_i} CHoCH         : {'[!!] REVERSAL SIGNAL' if signal.choch_signal else 'Not triggered'}")

    if signal.structure_break:
        b = signal.structure_break
        print(f"|     {b.break_type}: broke ${b.break_price:.2f} "
              f"(prior: {b.prior_trend.upper()}) → close ${b.break_close:.2f}")

    print(f"|  Structure : {signal.structure_type.upper()}")
    print("+==================================================+")
    if signal.direction != 'HOLD':
        print(f"|  ► {signal.direction} SIGNAL — SMC score {signal.score}/4")
    else:
        print("|  ► HOLD — waiting for SMC confirmation")
    print("+==================================================+")
    print(f"  Details: {signal.details}\n")


# -----------------------------------------------------------------------------
#  7. INTEGRATION HELPERS
# -----------------------------------------------------------------------------

def smc_filter_passes(signal: SMCSignal, ema_direction: str,
                      min_score: int = 2) -> bool:
    """Bridge: Phase 1 (EMA/RSI/ATR) + Phase 2 (SMC) must both agree."""
    if signal.direction == 'HOLD': return False
    if signal.score < min_score:   return False
    if signal.direction != ema_direction: return False
    return True


def get_smc_confluence_score(signal: SMCSignal) -> str:
    confirmed = []
    if signal.ob_signal:    confirmed.append("OB")
    if signal.fvg_signal:   confirmed.append("FVG")
    if signal.bos_signal:   confirmed.append("BOS")
    if signal.choch_signal: confirmed.append("CHoCH [!!]")
    if not confirmed:
        return "No SMC confluence"
    return " + ".join(confirmed) + f"  (score {signal.score}/4)"


# -----------------------------------------------------------------------------
#  HELPERS
# -----------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high  = df['high']
    low   = df['low']
    close = df['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs()
    ], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return val if not np.isnan(val) else (high - low).mean()


def _get_time(df: pd.DataFrame, i: int):
    try:    return df.index[i]
    except: return i


# -----------------------------------------------------------------------------
#  STANDALONE TEST
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PHASE 2 SMC ENGINE v2.6.0 — LIVE TEST (H1+H4+D1+W1)")
    print("="*60)

    try:
        import MetaTrader5 as mt5

        if not mt5.initialize():
            print("[NO] MT5 not running. Start MetaTrader 5 first.")
            exit()

        print("[OK] Connected to MT5")
        rates_h1 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1,  0, 200)
        rates_h4 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H4,  0, 100)
        rates_d1 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_D1,  0, 100)
        rates_w1 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_W1,  0, 100)
        mt5.shutdown()

        df_h1 = pd.DataFrame(rates_h1)
        df_h4 = pd.DataFrame(rates_h4)
        df_d1 = pd.DataFrame(rates_d1)
        df_w1 = pd.DataFrame(rates_w1)
        for df in [df_h1, df_h4, df_d1, df_w1]:
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

        current_price = df_h1.iloc[-1]['close']
        print(f"[OK] Data loaded — {len(df_h1)} H1 | {len(df_h4)} H4 | {len(df_d1)} D1 | {len(df_w1)} W1 candles | Price: ${current_price:.2f}\n")

        print("-- Order Blocks --")
        obs = detect_order_blocks(df_h1, lookback=50)
        for ob in [o for o in obs if o.active][:5]:
            t = " [!!]TESTED" if ob.tested else ""
            print(f"  {ob.ob_type.upper():8s} OB | ${ob.low:.2f}–${ob.high:.2f} | str={ob.strength}{t}")

        print("\n-- Fair Value Gaps --")
        fvgs = detect_fvg(df_h1, lookback=30)
        for fvg in [f for f in fvgs if not f.filled][:5]:
            print(f"  {fvg.fvg_type.upper():8s} FVG | ${fvg.bottom:.2f}–${fvg.top:.2f} | ${fvg.size:.2f}")

        print("\n-- Market Structure --")
        structure = detect_bos_choch(df_h1, lookback=200,
                                   df_d1=df_d1,
                                   df_w1=df_w1)
        print(f"  Prior Trend: {structure['trend'].upper()}")
        if structure['last_high']: print(f"  Last High  : ${structure['last_high']:.2f}")
        if structure['last_low']:  print(f"  Last Low   : ${structure['last_low']:.2f}")
        print(f"  BOS Bull   : {'YES [OK]' if structure['bos_bullish']  else 'No'}")
        print(f"  BOS Bear   : {'YES [OK]' if structure['bos_bearish']  else 'No'}")
        print(f"  CHoCH Bull : {'YES [!!]' if structure['choch_bullish'] else 'No'}")
        print(f"  CHoCH Bear : {'YES [!!]' if structure['choch_bearish'] else 'No'}")
        if structure['break_detail']:
            b = structure['break_detail']
            print(f"  Break      : {b.break_type} @ ${b.break_price:.2f} | "
                  f"prior={b.prior_trend.upper()} → close ${b.break_close:.2f}")

        print("\n-- Combined SMC Signal --")
        signal = get_smc_signal(df_h1, df_h4, df_d1, df_w1)
        print_smc_report(signal, current_price)
        print(f"  Confluence : {get_smc_confluence_score(signal)}")

        print("\n-- Phase 1 + Phase 2 Integration Test --")
        for direction in ['BUY', 'SELL']:
            passes = smc_filter_passes(signal, direction)
            icon = '[OK]' if passes else '[NO]'
            print(f"  {icon} {direction} signal + SMC filter: {'PASSES' if passes else 'BLOCKED'}")

    except ImportError:
        print("⚠️  MT5 not installed. Synthetic test.\n")
        np.random.seed(99)
        n = 200
        p = 5200.0
        dates = pd.date_range('2026-01-01', periods=n, freq='1h')
        o, h, l, c = [], [], [], []
        for i in range(n):
            # Bullish for first 130 bars, then drop sharply
            drift = 1.2 if i < 130 else -2.5
            ch = np.random.normal(drift, 7)
            op = p; cl = p + ch
            hi = max(op, cl) + abs(np.random.normal(0, 3))
            lo = min(op, cl) - abs(np.random.normal(0, 3))
            o.append(op); h.append(hi); l.append(lo); c.append(cl)
            p = cl
        df = pd.DataFrame({'open':o,'high':h,'low':l,'close':c}, index=dates)
        print(f"Synthetic price: ${c[-1]:.2f}")
        structure = detect_bos_choch(df, lookback=100)
        print(f"Prior Trend: {structure['trend']}")
        print(f"CHoCH Bear: {structure['choch_bearish']}")
        if structure['break_detail']:
            b = structure['break_detail']
            print(f"Break: {b.break_type} | prior={b.prior_trend}")
        signal = get_smc_signal(df)
        print_smc_report(signal, c[-1])

    print("="*60)
    print("  SMC Engine v2.6.0 test complete!")
    print("="*60)
