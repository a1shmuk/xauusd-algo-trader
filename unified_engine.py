"""
╔══════════════════════════════════════════════════════════════╗
║         XAUUSD PHASE 5 — UNIFIED SIGNAL ENGINE              ║
║         Combines Phase 1 + Phase 2 + Phase 4                ║
╠══════════════════════════════════════════════════════════════╣
║  Scoring System (0–10 points total):                         ║
║   Phase 1 Technical  : 0–3 pts (filters passed)             ║
║   Phase 2 SMC        : 0–4 pts (OB+FVG+BOS+CHoCH)          ║
║   Phase 4 Fundamental: 0–3 pts (macro alignment)            ║
║                                                              ║
║  Trade fires when score >= MIN_UNIFIED_SCORE (default: 6)   ║
║  News blackout always hard-blocks regardless of score        ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

# ── Minimum score to place a trade (optimized in phase5_optimizer.py) ──
MIN_UNIFIED_SCORE = 8   # out of 10

# ── Phase weights ──
WEIGHT_PHASE1 = 3   # max points from technical filters
WEIGHT_PHASE2 = 4   # max points from SMC
WEIGHT_PHASE4 = 3   # max points from fundamental

# ─────────────────────────────────────────────
#  UNIFIED SIGNAL RESULT
# ─────────────────────────────────────────────

@dataclass
class UnifiedSignal:
    direction: str          # 'BUY', 'SELL', 'HOLD'
    total_score: int        # 0–10
    phase1_score: int       # 0–3
    phase2_score: int       # 0–4
    phase4_score: int       # 0–3
    news_block: bool        # hard block regardless of score
    news_reason: str
    confidence: str         # 'HIGH', 'MEDIUM', 'LOW'
    details: str

# ─────────────────────────────────────────────
#  PHASE 1 SCORING (0–3 points)
# ─────────────────────────────────────────────

def score_phase1(rsi_ok: bool, mtf_ok: bool, sess_ok: bool) -> int:
    """
    Convert Phase 1 filter results into a score.
    Each passing filter = 1 point. Max = 3.
    """
    return sum([rsi_ok, mtf_ok, sess_ok])


# ─────────────────────────────────────────────
#  PHASE 2 SCORING (0–4 points)
# ─────────────────────────────────────────────

def score_phase2(df_h1: pd.DataFrame, df_h4: pd.DataFrame, direction: str) -> tuple[int, str]:
    """
    Run SMC engine and return score + details.
    CHoCH = +2, each other signal = +1. Max = 4.

    Returns:
        (score, details_string)
    """
    try:
        from smc_engine import get_smc_signal, get_smc_confluence_score
        signal = get_smc_signal(df_h1, df_h4)

        # Only score if direction matches
        if signal.direction != direction and signal.direction != 'HOLD':
            return 0, f"SMC direction mismatch ({signal.direction} vs {direction})"

        score = min(signal.score, 4)
        details = get_smc_confluence_score(signal)
        return score, details

    except ImportError:
        return 0, "SMC engine not available"
    except Exception as e:
        return 0, f"SMC error: {e}"


# ─────────────────────────────────────────────
#  PHASE 4 SCORING (0–3 points)
# ─────────────────────────────────────────────

def score_phase4(direction: str) -> tuple[int, bool, str, str]:
    """
    Run fundamental engine and return score + news block status.

    Scoring:
      Macro score +2 or +3 and direction aligned  → +3 pts
      Macro score +1 and direction aligned         → +2 pts
      Macro NEUTRAL                                → +1 pt
      Macro against direction                      → 0 pts
      News blackout                                → hard block

    Returns:
        (score, news_block, news_reason, details)
    """
    try:
        from fundamental_engine import get_fundamental_signal, fundamental_filter_passes

        signal = get_fundamental_signal()

        # Hard block for news regardless of score
        if signal.news_block:
            return 0, True, signal.news_reason, signal.details

        passes, reason = fundamental_filter_passes(signal, direction)

        if not passes:
            return 0, False, "", f"Fundamental blocked: {signal.bias} (score {signal.score:+d})"

        # Convert macro score to 0–3 points
        macro = signal.score  # -4 to +4
        aligned = (
            (signal.bias == "BULLISH" and direction == "BUY") or
            (signal.bias == "BEARISH" and direction == "SELL")
        )

        if aligned and abs(macro) >= 2:
            pts = 3
        elif aligned and abs(macro) == 1:
            pts = 2
        elif signal.bias == "NEUTRAL":
            pts = 1
        else:
            pts = 0

        details = f"Fund {signal.bias} ({signal.score:+d}) → {pts}pts"
        return pts, False, "", details

    except ImportError:
        # No fundamental engine — give neutral score
        return 1, False, "", "Fundamental engine not available — neutral"
    except Exception as e:
        return 1, False, "", f"Fundamental error: {e} — neutral"


# ─────────────────────────────────────────────
#  UNIFIED SIGNAL GENERATOR
# ─────────────────────────────────────────────

def get_unified_signal(
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
    ema_direction: str,     # 'BUY' or 'SELL' from EMA crossover
    rsi_ok: bool,
    mtf_ok: bool,
    sess_ok: bool,
    min_score: int = MIN_UNIFIED_SCORE
) -> UnifiedSignal:
    """
    Generate unified signal by combining all 3 phases.

    Parameters:
        df_h1         : H1 OHLC DataFrame
        df_h4         : H4 OHLC DataFrame
        ema_direction : 'BUY' or 'SELL' from EMA crossover
        rsi_ok        : Phase 1 RSI filter result
        mtf_ok        : Phase 1 MTF trend filter result
        sess_ok       : Phase 1 session filter result
        min_score     : Minimum total score to fire trade

    Returns:
        UnifiedSignal with direction, score, and all details
    """

    # ── Phase 1 score ──
    p1_score = score_phase1(rsi_ok, mtf_ok, sess_ok)

    # ── Phase 4 score (check news blackout first) ──
    p4_score, news_block, news_reason, p4_details = score_phase4(ema_direction)

    # Hard block on news — return immediately
    if news_block:
        return UnifiedSignal(
            direction    = "HOLD",
            total_score  = 0,
            phase1_score = p1_score,
            phase2_score = 0,
            phase4_score = 0,
            news_block   = True,
            news_reason  = news_reason,
            confidence   = "BLOCKED",
            details      = news_reason
        )

    # ── Phase 2 score ──
    p2_score, p2_details = score_phase2(df_h1, df_h4, ema_direction)

    # ── Total score ──
    total = p1_score + p2_score + p4_score

    # ── Determine direction and confidence ──
    if total >= min_score:
        direction = ema_direction
        if total >= 8:
            confidence = "HIGH"
        elif total >= 6:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    else:
        direction  = "HOLD"
        confidence = "LOW"

    details = (
        f"P1={p1_score}/3  P2={p2_score}/4  P4={p4_score}/3  "
        f"Total={total}/10  Min={min_score}  "
        f"| {p2_details} | {p4_details}"
    )

    return UnifiedSignal(
        direction    = direction,
        total_score  = total,
        phase1_score = p1_score,
        phase2_score = p2_score,
        phase4_score = p4_score,
        news_block   = False,
        news_reason  = "",
        confidence   = confidence,
        details      = details
    )


# ─────────────────────────────────────────────
#  PRETTY PRINT REPORT
# ─────────────────────────────────────────────

def print_unified_report(signal: UnifiedSignal, current_price: float):
    """Print a formatted unified signal report."""
    bar = "█" * signal.total_score + "░" * (10 - signal.total_score)
    dir_icon = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️ "}.get(signal.direction, "⏸️ ")
    conf_icon = {"HIGH": "🔥", "MEDIUM": "✅", "LOW": "⚠️", "BLOCKED": "🚫"}.get(signal.confidence, "")

    print("╔══════════════════════════════════════════════════╗")
    print("║         PHASE 5 — UNIFIED SIGNAL ENGINE         ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Price      : ${current_price:.2f}")
    print(f"║  Direction  : {dir_icon} {signal.direction}")
    print(f"║  Confidence : {conf_icon} {signal.confidence}")
    print(f"║  Score      : {bar}  {signal.total_score}/10")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Phase 1 Technical  : {'█' * signal.phase1_score + '░' * (3 - signal.phase1_score)}  {signal.phase1_score}/3")
    print(f"║  Phase 2 SMC        : {'█' * signal.phase2_score + '░' * (4 - signal.phase2_score)}  {signal.phase2_score}/4")
    print(f"║  Phase 4 Fundamental: {'█' * signal.phase4_score + '░' * (3 - signal.phase4_score)}  {signal.phase4_score}/3")
    print("╠══════════════════════════════════════════════════╣")
    if signal.news_block:
        print(f"║  🚫 NEWS BLOCK: {signal.news_reason[:45]}")
    elif signal.direction != "HOLD":
        print(f"║  ► {signal.direction} SIGNAL CONFIRMED — score {signal.total_score}/10")
    else:
        print(f"║  ► HOLD — score {signal.total_score}/10 below threshold")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  {signal.details}\n")


# ─────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PHASE 5 UNIFIED ENGINE — LIVE TEST")
    print("="*60)

    try:
        import MetaTrader5 as mt5

        if not mt5.initialize():
            print("❌ MT5 not running. Start MetaTrader 5 first.")
            exit()

        print("✅ Connected to MT5\n")

        rates_h1 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 300)
        rates_h4 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H4, 0, 100)
        mt5.shutdown()

        df_h1 = pd.DataFrame(rates_h1)
        df_h4 = pd.DataFrame(rates_h4)
        for df in [df_h1, df_h4]:
            df['time'] = pd.to_datetime(df['time'], unit='s')

        current_price = df_h1.iloc[-1]['close']
        print(f"Current price: ${current_price:.2f}\n")

        # Simulate Phase 1 results
        close   = df_h1['close']
        rsi_val = 100 - (100 / (1 + (close.diff().clip(lower=0).ewm(14).mean() /
                                      (-close.diff().clip(upper=0).ewm(14).mean()))))
        rsi = rsi_val.iloc[-1]
        ema9  = close.ewm(span=9,  adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]

        ema_direction = "BUY" if ema9 > ema21 else "SELL"
        rsi_ok  = (ema_direction == "BUY"  and 50 <= rsi <= 65) or \
                  (ema_direction == "SELL" and 35 <= rsi <= 50)
        mtf_ok  = True   # simplified for test
        sess_ok = 13 <= datetime.now(timezone.utc).hour <= 17

        print(f"EMA Direction : {ema_direction}")
        print(f"RSI           : {rsi:.1f}  {'✅' if rsi_ok else '❌'}")
        print(f"Session       : {'✅' if sess_ok else '❌'}")
        print()

        # Run unified signal
        signal = get_unified_signal(
            df_h1         = df_h1,
            df_h4         = df_h4,
            ema_direction = ema_direction,
            rsi_ok        = rsi_ok,
            mtf_ok        = mtf_ok,
            sess_ok       = sess_ok,
            min_score     = MIN_UNIFIED_SCORE
        )

        print_unified_report(signal, current_price)

        # Test different thresholds
        print("── Score at different thresholds ──")
        for threshold in [4, 5, 6, 7, 8]:
            fires = signal.total_score >= threshold
            icon  = "✅ TRADE" if fires else "❌ HOLD "
            print(f"  Min score {threshold}: {icon}  (score={signal.total_score})")

    except ImportError:
        print("⚠️  MetaTrader5 not available — showing score breakdown only\n")
        print("  Phase 1 (Technical) : max 3 pts — RSI + MTF + Session")
        print("  Phase 2 (SMC)       : max 4 pts — OB + FVG + BOS + CHoCH")
        print("  Phase 4 (Fundamental): max 3 pts — DXY + CPI + Fed + NFP")
        print("  ─────────────────────────────────────────────")
        print("  Total               : max 10 pts")
        print("  Default threshold   : 6/10 to fire trade")
        print("\n  Install MetaTrader5 and run again for live test.")

    print("="*60)
    print("  Unified Engine test complete!")
    print("="*60)
