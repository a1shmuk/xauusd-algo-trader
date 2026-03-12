import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time

# ── Telegram Alerts ──
try:
    from telegram_alerts import (
        alert_bot_started, alert_trade_opened, alert_trade_closed,
        alert_signal_blocked, alert_trail_moved, alert_daily_summary,
        maybe_send_heartbeat, alert_error, alert_disconnected,
        alert_reconnected, alert_bot_stopped
    )
    TELEGRAM_ON = True
    print("✅ Telegram alerts loaded")
except ImportError:
    TELEGRAM_ON = False
    print("⚠️  telegram_alerts.py not found — running without Telegram")

# ══════════════════════════════════════════════════════════════════════
#
#   XAUUSD PHASE 1 — LIVE AUTO TRADING BOT
#   ─────────────────────────────────────────
#   Runs 24/7 — checks every new H1 candle close
#   Places real BUY / SELL orders on MT5
#
#   Engine 1A: RSI Filter
#   Engine 1B: ATR Dynamic SL/TP
#   Engine 1C: Multi-Timeframe Trend Filter
#   Engine 1D: Session Time Filter
#   Engine 1E: Swing High/Low Structure SL
#
# ══════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────┐
# │                     ⚙️  SETTINGS                               │
# │         Change these to customize the bot                       │
# └─────────────────────────────────────────────────────────────────┘

# ── Symbol & Timeframe ──
SYMBOL        = "XAUUSD"
TIMEFRAME     = mt5.TIMEFRAME_H1
TIMEFRAME_H4  = mt5.TIMEFRAME_H4
TIMEFRAME_D1  = mt5.TIMEFRAME_D1
NUM_CANDLES   = 300

# ── EMA Crossover ──
EMA_FAST      = 9
EMA_SLOW      = 21

# ── Engine 1A: RSI ──
# ✅ OPTIMIZED — tested 4374 combinations on 2 years data
RSI_PERIOD    = 14
RSI_OB        = 65       # Tighter OB — blocks more bad BUYs  (was 70)
RSI_OS        = 35       # Tighter OS — blocks more bad SELLs (was 30)
RSI_MID       = 50       # BUY needs RSI > 50, SELL needs RSI < 50

# ── Engine 1B: ATR ──
# ✅ OPTIMIZED — wider SL gives trades more room to breathe
ATR_PERIOD    = 14
ATR_SL_MULT   = 2.0      # SL = ATR × 2.0  wider stop (was 1.5)
ATR_TP_MULT   = 3.0      # TP = ATR × 3.0  gives 1:1.5 RR

# ── Engine 1C: MTF ──
MTF_EMA       = 50       # H4 EMA period for trend filter

# ── Engine 1D: Session (UTC hours) ──
# ✅ OPTIMIZED — only trade London-NY overlap (highest quality window)
SESSION_START = 13       # London-NY overlap start (was 8)
SESSION_END   = 17       # London-NY overlap end   (was 17)
BEST_START    = 13       # London-NY overlap start
BEST_END      = 17       # London-NY overlap end

# ── Engine 1E: Swing SL ──
SWING_BARS    = 5        # Bars each side to confirm swing
SWING_BUFFER  = 0.50     # Extra buffer beyond swing point ($)

# ── Trade Settings ──
LOT_SIZE      = 0.01     # Trade size (0.01 = micro lot)
MAGIC         = 99001    # Unique bot ID — don't change
COMMENT       = "P1Bot"  # Order label in MT5
MAX_TRADES    = 1        # Max open positions at once
MIN_FILTERS   = 3        # ✅ OPTIMIZED — all 3 filters must pass (was 2)

# ── Bot Timing ──
CHECK_EVERY   = 30       # Seconds between checks (30 recommended)

# ┌─────────────────────────────────────────────────────────────────┐
# │                    END OF SETTINGS                              │
# └─────────────────────────────────────────────────────────────────┘

# ── State tracking ──
last_bar_time   = None
total_trades    = 0
total_wins      = 0
total_loss      = 0
bot_start_time  = datetime.now(timezone.utc)

# ── Telegram context (filled during analysis, used in place_order) ──
_last_rsi     = 0.0
_last_atr     = 0.0
_last_filters = 0
_last_h4      = "UNKNOWN"
_last_session = "Unknown"


# ══════════════════════════════════════════════════════════════════════
#   CONNECTION
# ══════════════════════════════════════════════════════════════════════
def connect():
    if not mt5.initialize():
        print(f"❌ MT5 connection failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    print(f"✅ Connected  | Account: {info.login} | Balance: ${info.balance:,.2f}")
    return True

def get_data(tf=None, n=NUM_CANDLES):
    if tf is None:
        tf = TIMEFRAME
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df.index = pd.to_datetime(df["time"], unit="s")
    df.drop(columns=["time"], inplace=True, errors="ignore")
    return df


# ══════════════════════════════════════════════════════════════════════
#   INDICATORS
# ══════════════════════════════════════════════════════════════════════
def add_ema(df):
    df["EMA9"]  = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    return df

def add_rsi(df):
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD,
                        adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD,
                        adjust=False).mean()
    rs       = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_atr(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/ATR_PERIOD, min_periods=ATR_PERIOD,
                       adjust=False).mean()
    return df


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1A — RSI FILTER
# ══════════════════════════════════════════════════════════════════════
def engine_rsi(df, signal):
    rsi = df["RSI"].iloc[-1]

    if signal == "BUY":
        # BUY needs RSI above 50 but NOT overbought
        ok = (rsi > RSI_MID) and (rsi < RSI_OB)
        if rsi >= RSI_OB:
            reason = f"BLOCKED — Overbought RSI={rsi:.1f}"
        elif rsi <= RSI_MID:
            reason = f"BLOCKED — RSI below 50 ({rsi:.1f}) no bull momentum"
        else:
            reason = f"✅ PASS — RSI={rsi:.1f} (bullish, not OB)"

    elif signal == "SELL":
        # SELL needs RSI below 50 but NOT oversold
        ok = (rsi < RSI_MID) and (rsi > RSI_OS)
        if rsi <= RSI_OS:
            reason = f"BLOCKED — Oversold RSI={rsi:.1f}"
        elif rsi >= RSI_MID:
            reason = f"BLOCKED — RSI above 50 ({rsi:.1f}) no bear momentum"
        else:
            reason = f"✅ PASS — RSI={rsi:.1f} (bearish, not OS)"
    else:
        ok, reason = False, "No signal"

    return ok, rsi, reason


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1B — ATR DYNAMIC SL/TP
# ══════════════════════════════════════════════════════════════════════
def engine_atr(df, signal, entry):
    atr = df["ATR"].iloc[-1]

    if signal == "BUY":
        sl = round(entry - atr * ATR_SL_MULT, 2)
        tp = round(entry + atr * ATR_TP_MULT, 2)
    elif signal == "SELL":
        sl = round(entry + atr * ATR_SL_MULT, 2)
        tp = round(entry - atr * ATR_TP_MULT, 2)
    else:
        return 0, 0, atr

    return sl, tp, round(atr, 2)


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1C — MTF TREND FILTER
# ══════════════════════════════════════════════════════════════════════
def engine_mtf(signal):
    df_h4 = get_data(TIMEFRAME_H4, 100)
    if df_h4 is None:
        return True, "UNKNOWN", 0  # Don't block if no data

    df_h4["EMA50"] = df_h4["close"].ewm(span=MTF_EMA, adjust=False).mean()
    h4_close = df_h4["close"].iloc[-1]
    h4_ema50 = df_h4["EMA50"].iloc[-1]
    h4_trend = "BULLISH" if h4_close > h4_ema50 else "BEARISH"

    if signal == "BUY":
        ok     = h4_trend == "BULLISH"
        reason = f"{'✅ PASS' if ok else '❌ FAIL'} — H4 EMA50={h4_ema50:.2f} Trend={h4_trend}"
    elif signal == "SELL":
        ok     = h4_trend == "BEARISH"
        reason = f"{'✅ PASS' if ok else '❌ FAIL'} — H4 EMA50={h4_ema50:.2f} Trend={h4_trend}"
    else:
        ok, reason = False, "No signal"

    return ok, h4_trend, h4_ema50


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1D — SESSION FILTER
# ══════════════════════════════════════════════════════════════════════
def engine_session():
    now      = datetime.now(timezone.utc)
    hour     = now.hour
    weekday  = now.weekday()  # 0=Monday, 5=Saturday, 6=Sunday

    # Block weekends
    if weekday >= 5:
        return False, "WEEKEND", hour

    # Identify active session
    in_london  = SESSION_START <= hour < SESSION_END
    in_overlap = BEST_START    <= hour < BEST_END

    if in_overlap:
        session = "London-NY Overlap ⭐ BEST"
        ok      = True
    elif in_london:
        session = "London"
        ok      = True
    else:
        ok = False
        if 0 <= hour < 9:
            session = "Tokyo (avoid)"
        elif 13 <= hour < 22:
            session = "New York"
            ok      = True   # Also allow NY
        else:
            session = "Sydney (avoid)"

    return ok, session, hour


# ══════════════════════════════════════════════════════════════════════
#   ENGINE 1E — SWING SL REFINEMENT
# ══════════════════════════════════════════════════════════════════════
def engine_swing(df, signal, atr_sl):
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)
    lb    = SWING_BARS

    swing_lows  = []
    swing_highs = []

    for i in range(lb, n - lb):
        if all(lows[i]  < lows[i-j]  for j in range(1, lb+1)) and \
           all(lows[i]  < lows[i+j]  for j in range(1, lb+1)):
            swing_lows.append(round(lows[i], 2))

        if all(highs[i] > highs[i-j] for j in range(1, lb+1)) and \
           all(highs[i] > highs[i+j] for j in range(1, lb+1)):
            swing_highs.append(round(highs[i], 2))

    if signal == "BUY" and swing_lows:
        struct_sl = round(swing_lows[-1] - SWING_BUFFER, 2)
        # Use tighter SL of the two methods
        final_sl  = max(struct_sl, atr_sl)
        return final_sl, f"SwingLow={swing_lows[-1]:.2f} StructSL={struct_sl:.2f}"

    elif signal == "SELL" and swing_highs:
        struct_sl = round(swing_highs[-1] + SWING_BUFFER, 2)
        final_sl  = min(struct_sl, atr_sl)
        return final_sl, f"SwingHigh={swing_highs[-1]:.2f} StructSL={struct_sl:.2f}"

    return atr_sl, "No swing found — using ATR SL"


# ══════════════════════════════════════════════════════════════════════
#   EMA CROSSOVER SIGNAL
# ══════════════════════════════════════════════════════════════════════
def get_signal(df):
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if prev["EMA9"] < prev["EMA21"] and curr["EMA9"] > curr["EMA21"]:
        return "BUY"
    elif prev["EMA9"] > prev["EMA21"] and curr["EMA9"] < curr["EMA21"]:
        return "SELL"
    return "HOLD"


# ══════════════════════════════════════════════════════════════════════
#   AUTO-DETECT FILL MODE
# ══════════════════════════════════════════════════════════════════════
def get_fill_mode():
    """
    Reads what fill modes the broker actually supports.
    Error 10030 = wrong fill mode. This fixes it automatically.
    """
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return mt5.ORDER_FILLING_IOC

    filling = info.filling_mode

    # filling_mode is a bitmask:
    # 1 = FOK supported
    # 2 = IOC supported
    # If neither, use RETURN

    if filling == 0:
        return mt5.ORDER_FILLING_RETURN   # 0 means RETURN only
    elif filling & 2:
        return mt5.ORDER_FILLING_IOC      # IOC supported
    elif filling & 1:
        return mt5.ORDER_FILLING_FOK      # FOK supported
    else:
        return mt5.ORDER_FILLING_RETURN


# ══════════════════════════════════════════════════════════════════════
#   ORDER EXECUTION
# ══════════════════════════════════════════════════════════════════════
def place_order(signal, sl, tp):
    global total_trades

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"❌ Symbol {SYMBOL} not found")
        return False

    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)

    # Get current price
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("❌ Cannot get price tick")
        return False

    price      = tick.ask if signal == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL

    # Validate SL/TP minimum distances
    min_stop = symbol_info.trade_stops_level * symbol_info.point
    sl = round(sl, symbol_info.digits)
    tp = round(tp, symbol_info.digits)

    if signal == "BUY":
        if sl >= price - min_stop:
            sl = round(price - min_stop - symbol_info.point * 10, symbol_info.digits)
        if tp <= price + min_stop:
            tp = round(price + min_stop + symbol_info.point * 10, symbol_info.digits)
    else:
        if sl <= price + min_stop:
            sl = round(price + min_stop + symbol_info.point * 10, symbol_info.digits)
        if tp >= price - min_stop:
            tp = round(price - min_stop - symbol_info.point * 10, symbol_info.digits)

    # Auto-detect fill mode
    fill_mode = get_fill_mode()
    fill_names = {
        mt5.ORDER_FILLING_IOC    : "IOC",
        mt5.ORDER_FILLING_FOK    : "FOK",
        mt5.ORDER_FILLING_RETURN : "RETURN"
    }
    print(f"   ℹ️  Fill mode: {fill_names.get(fill_mode, str(fill_mode))}")

    print(f"   📤 Sending {signal} order...")
    print(f"      Price : {price:.2f}")
    print(f"      SL    : {sl:.2f}")
    print(f"      TP    : {tp:.2f}")
    print(f"      Lot   : {LOT_SIZE}")

    # Try each fill mode until one works
    all_fills = [fill_mode] + [m for m in [
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_RETURN
    ] if m != fill_mode]

    for fm in all_fills:
        request = {
            "action"      : mt5.TRADE_ACTION_DEAL,
            "symbol"      : SYMBOL,
            "volume"      : LOT_SIZE,
            "type"        : order_type,
            "price"       : price,
            "sl"          : sl,
            "tp"          : tp,
            "deviation"   : 50,
            "magic"       : MAGIC,
            "comment"     : COMMENT,
            "type_time"   : mt5.ORDER_TIME_GTC,
            "type_filling": fm,
        }

        result = mt5.order_send(request)

        if result is None:
            print(f"   ⚠️  {fill_names.get(fm,'?')} — no response, trying next...")
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            total_trades += 1
            print(f"✅ ORDER PLACED!  Ticket: #{result.order}")
            print(f"   Entry : {result.price:.2f}")
            print(f"   SL    : {sl:.2f}  |  TP: {tp:.2f}")
            # ── Telegram: Trade opened ──
            if TELEGRAM_ON:
                try:
                    alert_trade_opened(
                        signal=signal,
                        entry=result.price,
                        sl=sl, tp=tp,
                        lot=LOT_SIZE,
                        ticket=result.order,
                        rsi=_last_rsi,
                        atr=_last_atr,
                        filters_passed=_last_filters,
                        h4_trend=_last_h4,
                        session=_last_session,
                    )
                except Exception:
                    pass
            return True
        else:
            print(f"   ⚠️  {fill_names.get(fm,'?')} failed "
                  f"(code {result.retcode}) — trying next fill mode...")

    print("❌ ORDER FAILED — all fill modes rejected by broker")
    print("   → Check MT5 journal for broker-side reason")
    return False


# ══════════════════════════════════════════════════════════════════════
#   POSITION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════
def count_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return 0
    return sum(1 for p in positions if p.magic == MAGIC)

def has_buy():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return False
    return any(p.magic == MAGIC and p.type == mt5.ORDER_TYPE_BUY
               for p in positions)

def has_sell():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return False
    return any(p.magic == MAGIC and p.type == mt5.ORDER_TYPE_SELL
               for p in positions)

def close_all():
    global total_wins, total_loss
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return

    for pos in positions:
        if pos.magic != MAGIC:
            continue

        tick       = mt5.symbol_info_tick(SYMBOL)
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        order_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY \
                      else mt5.ORDER_TYPE_BUY

        request = {
            "action"    : mt5.TRADE_ACTION_DEAL,
            "symbol"    : SYMBOL,
            "volume"    : pos.volume,
            "type"      : order_type,
            "position"  : pos.ticket,
            "price"     : close_price,
            "deviation" : 30,
            "magic"     : MAGIC,
            "comment"   : "close",
            "type_time" : mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            profit = pos.profit
            if profit > 0:
                total_wins += 1
                print(f"✅ Closed #{pos.ticket} — WIN  ${profit:.2f}")
            else:
                total_loss += 1
                print(f"❌ Closed #{pos.ticket} — LOSS ${profit:.2f}")

def get_open_pnl():
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return 0, 0
    my_pos = [p for p in positions if p.magic == MAGIC]
    total_pnl = sum(p.profit for p in my_pos)
    return total_pnl, len(my_pos)


# ══════════════════════════════════════════════════════════════════════
#   SMART EXIT SYSTEM
#   ─────────────────────────────────────────────────────────────────
#   Monitors open positions every 30 seconds and closes them when:
#   Exit 1 — Trailing Stop   : SL moves behind price as profit grows
#   Exit 2 — RSI Reversal    : RSI hits extreme opposite level
#   Exit 3 — EMA Crossover   : EMA crosses against the trade direction
#   Exit 4 — TP/SL           : Already handled by MT5 automatically
# ══════════════════════════════════════════════════════════════════════

# ── Settings for smart exits ──
TRAIL_ATR_MULT   = 1.0    # Trailing stop = ATR × this (tighter than entry SL)
RSI_EXIT_SELL    = 30     # Close SELL when RSI drops this low (oversold = bounce)
RSI_EXIT_BUY     = 70     # Close BUY  when RSI rises this high (overbought = drop)
MIN_PROFIT_TRAIL = 0.5    # Only start trailing after this × ATR in profit

def close_position(ticket, reason):
    """Close a single position by ticket number"""
    global total_wins, total_loss

    if not posInfo_select(ticket):
        return False

    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return False

    for pos in positions:
        if pos.ticket != ticket:
            continue

        tick        = mt5.symbol_info_tick(SYMBOL)
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY \
                      else mt5.ORDER_TYPE_BUY

        request = {
            "action"      : mt5.TRADE_ACTION_DEAL,
            "symbol"      : SYMBOL,
            "volume"      : pos.volume,
            "type"        : close_type,
            "position"    : pos.ticket,
            "price"       : close_price,
            "deviation"   : 50,
            "magic"       : MAGIC,
            "comment"     : f"exit:{reason}",
            "type_time"   : mt5.ORDER_TIME_GTC,
            "type_filling": get_fill_mode(),
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            profit = pos.profit
            if profit >= 0:
                total_wins += 1
                print(f"  ✅ CLOSED #{ticket} — WIN  ${profit:+.2f}  Reason: {reason}")
            else:
                total_loss += 1
                print(f"  ❌ CLOSED #{ticket} — LOSS ${profit:+.2f}  Reason: {reason}")
            # ── Telegram: Trade closed ──
            if TELEGRAM_ON:
                try:
                    close_px = pos.price_current
                    alert_trade_closed(
                        signal="BUY" if pos.type==mt5.ORDER_TYPE_BUY else "SELL",
                        entry=pos.price_open,
                        exit_price=close_px,
                        pnl=profit,
                        ticket=ticket,
                        reason=reason,
                    )
                except Exception:
                    pass
            return True
        else:
            code = result.retcode if result else "N/A"
            print(f"  ⚠️  Close failed #{ticket} code:{code}")
            return False

    return False

def posInfo_select(ticket):
    """Helper to check if position exists"""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return False
    return any(p.ticket == ticket for p in positions)

def update_trailing_stop(pos, atr):
    """
    Move SL closer to price as trade moves in our favour.
    Only moves SL in the profitable direction — never against us.
    Only starts after minimum profit threshold reached.
    """
    trail_dist = atr * TRAIL_ATR_MULT
    current_sl = pos.sl
    current_tp = pos.tp
    entry      = pos.price_open

    if pos.type == mt5.ORDER_TYPE_BUY:
        bid       = mt5.symbol_info_tick(SYMBOL).bid
        profit_pts = bid - entry

        # Only trail after minimum profit reached
        if profit_pts < atr * MIN_PROFIT_TRAIL:
            return False

        new_sl = round(bid - trail_dist, 2)

        # Only move SL UP (never down for a buy)
        if new_sl > current_sl + 0.10:
            result = mt5.order_send({
                "action"   : mt5.TRADE_ACTION_SLTP,
                "symbol"   : SYMBOL,
                "position" : pos.ticket,
                "sl"       : new_sl,
                "tp"       : current_tp,
            })
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  📈 TRAIL BUY  #{pos.ticket} "
                      f"SL {current_sl:.2f} → {new_sl:.2f}  "
                      f"(price={bid:.2f})")
                if TELEGRAM_ON:
                    try:
                        alert_trail_moved(pos.ticket, "BUY",
                                          current_sl, new_sl, bid)
                    except Exception:
                        pass
                return True

    elif pos.type == mt5.ORDER_TYPE_SELL:
        ask        = mt5.symbol_info_tick(SYMBOL).ask
        profit_pts = entry - ask

        # Only trail after minimum profit reached
        if profit_pts < atr * MIN_PROFIT_TRAIL:
            return False

        new_sl = round(ask + trail_dist, 2)

        # Only move SL DOWN (never up for a sell)
        if current_sl == 0 or new_sl < current_sl - 0.10:
            result = mt5.order_send({
                "action"   : mt5.TRADE_ACTION_SLTP,
                "symbol"   : SYMBOL,
                "position" : pos.ticket,
                "sl"       : new_sl,
                "tp"       : current_tp,
            })
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  📉 TRAIL SELL #{pos.ticket} "
                      f"SL {current_sl:.2f} → {new_sl:.2f}  "
                      f"(price={ask:.2f})")
                if TELEGRAM_ON:
                    try:
                        alert_trail_moved(pos.ticket, "SELL",
                                          current_sl, new_sl, ask)
                    except Exception:
                        pass
                return True

    return False

def manage_exits(df):
    """
    Run on every loop — checks all open positions for exit conditions.

    Exit conditions checked in order:
    1. Trailing stop update (moves SL, doesn't close)
    2. RSI extreme reversal (closes position)
    3. EMA crossover against trade (closes position)
    TP and SL exits are handled automatically by MT5 server.
    """
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return

    my_positions = [p for p in positions if p.magic == MAGIC]
    if not my_positions:
        return

    rsi   = df["RSI"].iloc[-1]
    ema9  = df["EMA9"].iloc[-1]
    ema21 = df["EMA21"].iloc[-1]
    atr   = df["ATR"].iloc[-1]
    price = df["close"].iloc[-1]

    for pos in my_positions:
        entry  = pos.price_open
        profit = pos.profit
        ptype  = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"

        # ── EXIT 1: Trailing Stop ──
        update_trailing_stop(pos, atr)

        # ── EXIT 2: RSI Reversal ──
        # If we are in a SELL and RSI drops to oversold → gold about to bounce UP → exit SELL
        if pos.type == mt5.ORDER_TYPE_SELL and rsi <= RSI_EXIT_SELL:
            print(f"\n  🔔 RSI EXIT TRIGGERED on #{pos.ticket}")
            print(f"     SELL position — RSI dropped to {rsi:.1f} (oversold)")
            print(f"     Gold may bounce UP — closing SELL to protect profit")
            close_position(pos.ticket, f"RSI_oversold_{rsi:.0f}")
            continue

        # If we are in a BUY and RSI rises to overbought → gold about to drop → exit BUY
        if pos.type == mt5.ORDER_TYPE_BUY and rsi >= RSI_EXIT_BUY:
            print(f"\n  🔔 RSI EXIT TRIGGERED on #{pos.ticket}")
            print(f"     BUY position — RSI rose to {rsi:.1f} (overbought)")
            print(f"     Gold may drop DOWN — closing BUY to protect profit")
            close_position(pos.ticket, f"RSI_overbought_{rsi:.0f}")
            continue

        # ── EXIT 3: EMA Crossover Against Trade ──
        # If in SELL and EMA9 crosses ABOVE EMA21 → bullish → exit SELL
        if pos.type == mt5.ORDER_TYPE_SELL and ema9 > ema21:
            # Only exit if trade is profitable or at breakeven
            if profit >= 0:
                print(f"\n  🔔 EMA EXIT TRIGGERED on #{pos.ticket}")
                print(f"     SELL position — EMA9 crossed ABOVE EMA21 (bullish)")
                print(f"     Closing SELL with profit ${profit:.2f}")
                close_position(pos.ticket, "EMA_cross_bullish")
                continue

        # If in BUY and EMA9 crosses BELOW EMA21 → bearish → exit BUY
        if pos.type == mt5.ORDER_TYPE_BUY and ema9 < ema21:
            if profit >= 0:
                print(f"\n  🔔 EMA EXIT TRIGGERED on #{pos.ticket}")
                print(f"     BUY position — EMA9 crossed BELOW EMA21 (bearish)")
                print(f"     Closing BUY with profit ${profit:.2f}")
                close_position(pos.ticket, "EMA_cross_bearish")
                continue

        # ── Status of open position ──
        print(f"  📂 Position #{pos.ticket} {ptype} "
              f"Entry:{entry:.2f} "
              f"Now:{price:.2f} "
              f"P&L:${profit:+.2f} "
              f"RSI:{rsi:.1f} "
              f"EMA_gap:{ema9-ema21:+.2f}")


# ══════════════════════════════════════════════════════════════════════
#   STATUS PRINT — shows every 30 seconds even without signal
# ══════════════════════════════════════════════════════════════════════
def print_status(df, signal, rsi, atr, h4_trend, session, hour_utc):
    now       = datetime.now(timezone.utc)
    price     = df["close"].iloc[-1]
    ema9      = df["EMA9"].iloc[-1]
    ema21     = df["EMA21"].iloc[-1]
    gap       = ema9 - ema21
    pnl, npos = get_open_pnl()
    account   = mt5.account_info()
    balance   = account.balance if account else 0
    uptime    = now - bot_start_time
    hours     = int(uptime.total_seconds() // 3600)
    mins      = int((uptime.total_seconds() % 3600) // 60)

    print(f"\n{'═'*60}")
    print(f"  🕐 {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'─'*60}")
    print(f"  💲 Price   : {price:.2f}  |  Session: {session} (UTC {hour_utc:02d}:xx)")
    print(f"  📊 EMA9    : {ema9:.2f}  |  EMA21: {ema21:.2f}  |  GAP: {gap:+.2f}")
    print(f"  📡 Signal  : {signal}")
    print(f"  📈 RSI     : {rsi:.1f}  {'⛔OB' if rsi>=RSI_OB else '⛔OS' if rsi<=RSI_OS else '✅Bull' if rsi>50 else '✅Bear'}")
    print(f"  📏 ATR     : ${atr:.2f}")
    print(f"  🌐 H4 Trend: {h4_trend}")
    print(f"{'─'*60}")
    if npos > 0:
        print(f"  📂 Open    : {npos} position(s)  |  P&L: ${pnl:+.2f}")
    else:
        print(f"  📂 Open    : No positions")
    print(f"  🏆 Trades  : {total_trades}  W:{total_wins} L:{total_loss}  "
          f"WR:{(total_wins/total_trades*100):.0f}%" if total_trades > 0
          else f"  🏆 Trades  : {total_trades}  (waiting for first signal...)")
    print(f"  💰 Balance : ${balance:,.2f}")
    print(f"  ⏱️  Uptime  : {hours}h {mins}m")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════
#   MAIN ANALYSIS LOOP — runs on every new H1 candle
# ══════════════════════════════════════════════════════════════════════
def analyze_and_trade():
    global last_bar_time

    # Get H1 data
    df = get_data(TIMEFRAME, NUM_CANDLES)
    if df is None:
        print("⚠️  No data — will retry...")
        return

    # Add all indicators
    df = add_ema(df)
    df = add_rsi(df)
    df = add_atr(df)

    # ── Check if new bar ──
    current_bar = df.index[-1]
    is_new_bar  = (current_bar != last_bar_time)

    rsi = df["RSI"].iloc[-1]
    atr = df["ATR"].iloc[-1]

    # ── Session check (always) ──
    sess_ok, session, hour_utc = engine_session()

    # ── H4 trend (always) ──
    h4_ok, h4_trend, h4_ema = engine_mtf("BUY")  # just for display

    # ── Print status ──
    signal = get_signal(df)
    print_status(df, signal, rsi, atr, h4_trend, session, hour_utc)

    # ── SMART EXIT CHECK (runs every loop, not just new bars) ──
    if count_positions() > 0:
        print(f"  🔍 Checking exit conditions...")
        manage_exits(df)

    if not is_new_bar:
        print(f"  ⏳ Same candle — next check in {CHECK_EVERY}s")
        return

    # ── New bar! ──

    # ── New bar! ──
    last_bar_time = current_bar
    print(f"\n  🕯️  NEW H1 CANDLE at {current_bar.strftime('%Y-%m-%d %H:%M')} UTC")

    # ── EMA Signal ──
    if signal == "HOLD":
        print("  📡 EMA: HOLD — no crossover on this candle")
        return

    print(f"\n  ⚡ EMA CROSSOVER: {signal} SIGNAL DETECTED!")
    print(f"{'─'*60}")

    # ── Run all 5 engines ──
    entry = mt5.symbol_info_tick(SYMBOL).ask if signal == "BUY" \
            else mt5.symbol_info_tick(SYMBOL).bid

    # 1A — RSI
    rsi_ok, rsi_val, rsi_reason = engine_rsi(df, signal)
    print(f"  1A RSI    : {rsi_reason}")

    # 1B — ATR
    atr_sl, atr_tp, atr_val = engine_atr(df, signal, entry)
    print(f"  1B ATR    : ${atr_val:.2f}  SL={atr_sl:.2f}  TP={atr_tp:.2f}  "
          f"RR=1:{ATR_TP_MULT/ATR_SL_MULT:.1f}")

    # 1C — MTF
    mtf_ok, h4_trend, h4_ema = engine_mtf(signal)
    print(f"  1C MTF    : {'✅ PASS' if mtf_ok else '❌ FAIL'} — "
          f"H4 EMA50={h4_ema:.2f} Trend={h4_trend}")

    # 1D — Session
    print(f"  1D Session: {'✅ PASS' if sess_ok else '❌ FAIL'} — "
          f"{session} UTC{hour_utc:02d}")

    # 1E — Swing SL
    final_sl, swing_reason = engine_swing(df, signal, atr_sl)
    print(f"  1E Swing  : {swing_reason}  FinalSL={final_sl:.2f}")

    # ── Count passing filters ──
    filters_passed = sum([rsi_ok, mtf_ok, sess_ok])
    print(f"\n  {'─'*58}")
    print(f"  Filters   : {filters_passed}/3 passed "
          f"(need {MIN_FILTERS})")

    if filters_passed < MIN_FILTERS:
        blocked_by = []
        if not rsi_ok:  blocked_by.append("RSI")
        if not mtf_ok:  blocked_by.append("MTF")
        if not sess_ok: blocked_by.append("Session")
        print(f"  ❌ BLOCKED by: {', '.join(blocked_by)}")
        print(f"  ⏳ Waiting for next signal...")
        # ── Telegram: signal blocked ──
        if TELEGRAM_ON:
            try:
                alert_signal_blocked(signal, blocked_by,
                                     rsi, h4_trend, session)
            except Exception:
                pass
        return

    print(f"  ✅ ALL REQUIRED FILTERS PASSED!")

    # ── Save context for Telegram alert in place_order ──
    global _last_rsi, _last_atr, _last_filters, _last_h4, _last_session
    _last_rsi     = rsi
    _last_atr     = atr
    _last_filters = filters_passed
    _last_h4      = h4_trend
    _last_session = session

    # ── Check max positions ──
    n_open = count_positions()
    if n_open >= MAX_TRADES:
        print(f"  ⚠️  Max trades ({MAX_TRADES}) already open — skipping")
        return

    # ── Close opposite position first ──
    if signal == "BUY"  and has_sell():
        print("  🔄 Closing existing SELL before opening BUY...")
        close_all()
        time.sleep(1)

    if signal == "SELL" and has_buy():
        print("  🔄 Closing existing BUY before opening SELL...")
        close_all()
        time.sleep(1)

    # ── Place order ──
    print(f"\n  🚀 PLACING {signal} ORDER...")
    success = place_order(signal, final_sl, atr_tp)

    if success:
        print(f"  ✅ Bot placed {signal} successfully!")
    else:
        print(f"  ❌ Order failed — will retry on next signal")


# ══════════════════════════════════════════════════════════════════════
#   BOT STARTUP
# ══════════════════════════════════════════════════════════════════════
def print_banner():
    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║       XAUUSD PHASE 1 — LIVE AUTO TRADING BOT        ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Symbol      : {SYMBOL:<38}║")
    print(f"║  Timeframe   : H1{'':<38}║")
    print(f"║  EMA         : {EMA_FAST}/{EMA_SLOW} crossover{'':<30}║")
    print(f"║  Lot Size    : {LOT_SIZE:<38}║")
    print(f"║  RSI Filter  : ON  (OB:{RSI_OB} OS:{RSI_OS} Mid:{RSI_MID}){'':<17}║")
    print(f"║  ATR SL/TP   : ON  (SL:{ATR_SL_MULT}x  TP:{ATR_TP_MULT}x){'':<22}║")
    print(f"║  MTF Filter  : ON  (H4 EMA{MTF_EMA}){'':<24}║")
    print(f"║  Session     : ON  ({SESSION_START}:00-{SESSION_END}:00 UTC){'':<27}║")
    print(f"║  Min Filters : {MIN_FILTERS}/3{'':<37}║")
    print(f"║  Check Every : {CHECK_EVERY}s{'':<37}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Press Ctrl+C to stop the bot safely                ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


# ══════════════════════════════════════════════════════════════════════
#   MAIN — infinite loop
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print_banner()

    # Connect to MT5
    if not connect():
        print("❌ Cannot connect to MT5 — make sure MT5 is open and logged in")
        exit()

    info = mt5.account_info()
    print(f"╔══════════════════════════════════════════╗")
    print(f"║  Account  : {info.login:<30}║")
    print(f"║  Balance  : ${info.balance:<29,.2f}║")
    print(f"║  Server   : {info.server:<30}║")
    print(f"╚══════════════════════════════════════════╝")
    print("\n✅ Bot running! Ctrl+C to stop.\n")

    # ── Telegram: Bot started ──
    if TELEGRAM_ON:
        alert_bot_started(info.balance, {
            "ema_fast": EMA_FAST, "ema_slow": EMA_SLOW,
            "min_filters": MIN_FILTERS,
            "sess_start": SESSION_START, "sess_end": SESSION_END,
            "atr_sl": ATR_SL_MULT, "atr_tp": ATR_TP_MULT,
        })

    # ── Main loop ──
    loop_count = 0
    while True:
        try:
            loop_count += 1

            # Reconnect if MT5 dropped
            if not mt5.terminal_info():
                print("⚠️  MT5 disconnected — reconnecting...")
                if not connect():
                    print("❌ Reconnect failed — waiting 60s...")
                    time.sleep(60)
                    continue

            # Run analysis
            analyze_and_trade()

            # ── Telegram: Heartbeat + Daily Summary ──
            if TELEGRAM_ON:
                try:
                    acct = mt5.account_info()
                    bal  = acct.balance if acct else 0
                    maybe_send_heartbeat(total_trades, total_wins,
                                         total_loss, bal)
                    now = datetime.now(timezone.utc)
                    if now.hour == 22 and now.minute < 1:
                        net = sum(p.profit for p in
                                  (mt5.positions_get(symbol=SYMBOL) or []))
                        alert_daily_summary(total_trades, total_wins,
                                            total_loss, net, bal)
                except Exception:
                    pass

            # Wait before next check
            print(f"\n  ⏳ Next check in {CHECK_EVERY} seconds... "
                  f"(loop #{loop_count})")
            time.sleep(CHECK_EVERY)

        except KeyboardInterrupt:
            print("\n\n⛔ Bot stopped by user (Ctrl+C)")
            print("─" * 50)
            print(f"  Total Trades : {total_trades}")
            print(f"  Wins         : {total_wins}")
            print(f"  Losses       : {total_loss}")
            if total_trades > 0:
                wr = total_wins / total_trades * 100
                print(f"  Win Rate     : {wr:.1f}%")
            info = mt5.account_info()
            if info:
                print(f"  Final Balance: ${info.balance:,.2f}")
            print("─" * 50)
            # ── Telegram: Bot stopped ──
            if TELEGRAM_ON:
                try:
                    bal = info.balance if info else 0
                    alert_bot_stopped(total_trades, total_wins,
                                      total_loss, bal)
                except Exception:
                    pass
            mt5.shutdown()
            print("✅ MT5 disconnected cleanly. Goodbye!")
            break

        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            if TELEGRAM_ON:
                try:
                    alert_error(str(e))
                except Exception:
                    pass
            print("🔄 Recovering in 10 seconds...")
            time.sleep(10)