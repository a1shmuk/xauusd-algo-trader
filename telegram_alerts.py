import requests
import MetaTrader5 as mt5
from datetime import datetime, timezone
import time
import threading

# ══════════════════════════════════════════════════════════════════════
#
#   XAUUSD TELEGRAM ALERT SYSTEM
#   ─────────────────────────────────────────────────────────────────
#   Sends rich Telegram notifications for every bot event:
#
#   📥 Trade Opened    — signal, entry, SL, TP, filters passed
#   ✅ Trade Closed TP — profit amount, pips gained
#   🛑 Trade Closed SL — loss amount, what went wrong
#   🔔 RSI Exit        — early close with reason
#   📊 Daily Summary   — sent at 22:00 UTC every day
#   💓 Heartbeat       — bot still alive ping every 6 hours
#   ⚠️  Error Alert     — if bot crashes or loses MT5 connection
#
# ══════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────┐
# │                  🔑 YOUR CREDENTIALS                           │
# └─────────────────────────────────────────────────────────────────┘
BOT_TOKEN = "8686394018:AAGrJJ8UNEXd0HR2ugY0b7P6Gd7EX641dL8"
CHAT_ID   = "6500991930"

# ┌─────────────────────────────────────────────────────────────────┐
# │                  ⚙️  ALERT SETTINGS                            │
# └─────────────────────────────────────────────────────────────────┘
HEARTBEAT_HOURS  = 6      # Send "bot alive" ping every N hours
DAILY_SUMMARY_HR = 22     # UTC hour to send daily summary (22 = 3:30 AM IST)
ALERT_ON_HOLD    = False  # Send message when signal is HOLD (noisy — keep False)

# ── Internal state ──
_last_heartbeat  = None
_last_summary_day = None
_session_start   = datetime.now(timezone.utc)
_trade_log       = []     # List of all trades this session


# ══════════════════════════════════════════════════════════════════════
#   CORE SEND FUNCTION
# ══════════════════════════════════════════════════════════════════════
def send(message: str, silent: bool = False) -> bool:
    """
    Send a Telegram message.
    silent=True = no phone notification (for low-priority messages)
    Returns True if sent successfully.
    """
    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id"                  : CHAT_ID,
        "text"                     : message,
        "parse_mode"               : "HTML",
        "disable_notification"     : silent,
        "disable_web_page_preview" : True,
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            return True
        else:
            print(f"  ⚠️  Telegram error {r.status_code}: {r.text[:100]}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ⚠️  Telegram: No internet connection")
        return False
    except requests.exceptions.Timeout:
        print("  ⚠️  Telegram: Request timed out")
        return False
    except Exception as e:
        print(f"  ⚠️  Telegram error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
#   ALERT 1 — BOT STARTED
# ══════════════════════════════════════════════════════════════════════
def alert_bot_started(balance: float, settings: dict):
    now = datetime.now(timezone.utc)
    msg = (
        f"🚀 <b>XAUUSD Bot Started</b>\n"
        f"{'─' * 28}\n"
        f"🕐 Time     : {now.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"💰 Balance  : <b>${balance:,.2f}</b>\n"
        f"📊 Strategy : EMA {settings.get('ema_fast',9)}/{settings.get('ema_slow',21)} Crossover\n"
        f"🎯 Filters  : {settings.get('min_filters',3)}/3 required\n"
        f"⏰ Session  : {settings.get('sess_start',13)}:00 – {settings.get('sess_end',17)}:00 UTC\n"
        f"📏 ATR SL   : {settings.get('atr_sl',2.0)}× | TP: {settings.get('atr_tp',3.0)}×\n"
        f"{'─' * 28}\n"
        f"✅ Bot is live and watching for signals"
    )
    ok = send(msg)
    if ok:
        print("  📱 Telegram: Bot started alert sent")
    return ok


# ══════════════════════════════════════════════════════════════════════
#   ALERT 2 — TRADE OPENED
# ══════════════════════════════════════════════════════════════════════
def alert_trade_opened(signal: str, entry: float, sl: float, tp: float,
                       lot: float, ticket: int,
                       rsi: float, atr: float,
                       filters_passed: int, h4_trend: str,
                       session: str):
    now      = datetime.now(timezone.utc)
    sl_pips  = abs(entry - sl)
    tp_pips  = abs(tp - entry)
    rr       = round(tp_pips / sl_pips, 1) if sl_pips > 0 else 0
    emoji    = "📈" if signal == "BUY" else "📉"
    color    = "🟢" if signal == "BUY" else "🔴"

    msg = (
        f"{emoji} <b>TRADE OPENED — {signal}</b> {color}\n"
        f"{'─' * 28}\n"
        f"🎫 Ticket   : #{ticket}\n"
        f"💲 Entry    : <b>${entry:.2f}</b>\n"
        f"🛑 Stop Loss: ${sl:.2f}  (${sl_pips:.2f} away)\n"
        f"🎯 Take Prof: ${tp:.2f}  (${tp_pips:.2f} away)\n"
        f"⚖️  Risk:Rew : 1:{rr}\n"
        f"📦 Lot Size : {lot}\n"
        f"{'─' * 28}\n"
        f"📊 RSI      : {rsi:.1f}\n"
        f"📏 ATR      : ${atr:.2f}\n"
        f"🌐 H4 Trend : {h4_trend}\n"
        f"🕐 Session  : {session}\n"
        f"✅ Filters  : {filters_passed}/3 passed\n"
        f"{'─' * 28}\n"
        f"🕐 {now.strftime('%H:%M')} UTC"
    )

    # Log trade for daily summary
    _trade_log.append({
        "type"   : "OPEN",
        "signal" : signal,
        "entry"  : entry,
        "sl"     : sl,
        "tp"     : tp,
        "ticket" : ticket,
        "time"   : now,
    })

    ok = send(msg)
    if ok:
        print(f"  📱 Telegram: Trade opened alert sent (#{ticket})")
    return ok


# ══════════════════════════════════════════════════════════════════════
#   ALERT 3 — TRADE CLOSED (TP / SL / Manual)
# ══════════════════════════════════════════════════════════════════════
def alert_trade_closed(signal: str, entry: float, exit_price: float,
                       pnl: float, ticket: int, reason: str,
                       bars_held: int = 0):
    now     = datetime.now(timezone.utc)
    won     = pnl >= 0
    pips    = abs(exit_price - entry)

    if reason == "TP":
        header = "✅ <b>TAKE PROFIT HIT!</b> 🎉"
        footer = "Target reached — full profit captured"
    elif reason == "SL":
        header = "🛑 <b>STOP LOSS HIT</b>"
        footer = "Stop triggered — loss limited as planned"
    elif "RSI" in reason:
        header = "🔔 <b>RSI EXIT</b>"
        footer = f"Early exit — RSI reversal signal"
    elif "EMA" in reason:
        header = "🔄 <b>EMA CROSSOVER EXIT</b>"
        footer = "Early exit — trend reversed"
    else:
        header = "🚪 <b>TRADE CLOSED</b>"
        footer = f"Reason: {reason}"

    pnl_str  = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    pnl_emoji = "💚" if won else "🔴"

    msg = (
        f"{header}\n"
        f"{'─' * 28}\n"
        f"🎫 Ticket   : #{ticket}\n"
        f"📊 Direction: {signal}\n"
        f"💲 Entry    : ${entry:.2f}\n"
        f"🚪 Exit     : ${exit_price:.2f}\n"
        f"📏 Pips     : {pips:.2f}\n"
        f"{pnl_emoji} P&L      : <b>{pnl_str}</b>\n"
        f"{'─' * 28}\n"
        f"ℹ️  {footer}\n"
        f"🕐 {now.strftime('%H:%M')} UTC"
    )

    # Log for daily summary
    _trade_log.append({
        "type"  : "CLOSE",
        "signal": signal,
        "pnl"   : pnl,
        "reason": reason,
        "ticket": ticket,
        "time"  : now,
        "won"   : won,
    })

    ok = send(msg)
    if ok:
        print(f"  📱 Telegram: Trade closed alert sent (#{ticket} {pnl_str})")
    return ok


# ══════════════════════════════════════════════════════════════════════
#   ALERT 4 — SIGNAL BLOCKED
# ══════════════════════════════════════════════════════════════════════
def alert_signal_blocked(signal: str, blocked_by: list,
                         rsi: float, h4_trend: str, session: str):
    now = datetime.now(timezone.utc)
    msg = (
        f"🚫 <b>SIGNAL BLOCKED — {signal}</b>\n"
        f"{'─' * 28}\n"
        f"EMA crossover detected but filtered out\n"
        f"❌ Blocked by: <b>{', '.join(blocked_by)}</b>\n"
        f"{'─' * 28}\n"
        f"📊 RSI      : {rsi:.1f}\n"
        f"🌐 H4 Trend : {h4_trend}\n"
        f"🕐 Session  : {session}\n"
        f"🕐 {now.strftime('%H:%M')} UTC\n"
        f"<i>Waiting for better conditions...</i>"
    )
    ok = send(msg, silent=True)  # Silent — not critical
    if ok:
        print(f"  📱 Telegram: Blocked signal alert sent")
    return ok


# ══════════════════════════════════════════════════════════════════════
#   ALERT 5 — TRAILING STOP MOVED
# ══════════════════════════════════════════════════════════════════════
def alert_trail_moved(ticket: int, signal: str,
                      old_sl: float, new_sl: float, price: float):
    locked = abs(price - new_sl)
    msg = (
        f"📐 <b>TRAILING STOP MOVED</b>\n"
        f"{'─' * 28}\n"
        f"🎫 #{ticket} {signal}\n"
        f"📍 Old SL : ${old_sl:.2f}\n"
        f"📍 New SL : <b>${new_sl:.2f}</b>\n"
        f"💲 Price  : ${price:.2f}\n"
        f"🔒 Locked : ${locked:.2f} profit secured"
    )
    ok = send(msg, silent=True)  # Silent — informational
    if ok:
        print(f"  📱 Telegram: Trail moved alert sent")
    return ok


# ══════════════════════════════════════════════════════════════════════
#   ALERT 6 — DAILY SUMMARY
# ══════════════════════════════════════════════════════════════════════
def alert_daily_summary(total_trades: int, wins: int, losses: int,
                        net_pnl: float, balance: float):
    global _last_summary_day
    now = datetime.now(timezone.utc)

    # Only send once per day
    today = now.date()
    if _last_summary_day == today:
        return False
    _last_summary_day = today

    wr      = (wins / total_trades * 100) if total_trades > 0 else 0
    wr_bar  = "█" * int(wr / 10) + "░" * (10 - int(wr / 10))
    pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
    grade   = "🟢 Excellent" if wr >= 60 else \
              "🟡 Good"      if wr >= 50 else \
              "🟠 Average"   if wr >= 40 else "🔴 Poor"

    msg = (
        f"📊 <b>DAILY SUMMARY</b>\n"
        f"{'─' * 28}\n"
        f"📅 {now.strftime('%Y-%m-%d')}\n"
        f"{'─' * 28}\n"
        f"🔢 Trades   : {total_trades}\n"
        f"✅ Wins     : {wins}\n"
        f"❌ Losses   : {losses}\n"
        f"📈 Win Rate : {wr:.1f}%  {grade}\n"
        f"   {wr_bar}\n"
        f"{'─' * 28}\n"
        f"💰 Net P&L  : <b>{pnl_str}</b>\n"
        f"🏦 Balance  : ${balance:,.2f}\n"
        f"{'─' * 28}\n"
        f"<i>Next session: 13:00–17:00 UTC</i>"
    )
    ok = send(msg)
    if ok:
        print("  📱 Telegram: Daily summary sent")
    return ok


# ══════════════════════════════════════════════════════════════════════
#   ALERT 7 — HEARTBEAT (bot still alive)
# ══════════════════════════════════════════════════════════════════════
def maybe_send_heartbeat(total_trades: int, wins: int,
                         losses: int, balance: float):
    global _last_heartbeat
    now = datetime.now(timezone.utc)

    # Send every N hours
    if _last_heartbeat is None or \
       (now - _last_heartbeat).total_seconds() >= HEARTBEAT_HOURS * 3600:

        uptime   = now - _session_start
        hours    = int(uptime.total_seconds() // 3600)
        mins     = int((uptime.total_seconds() % 3600) // 60)
        wr       = (wins / total_trades * 100) if total_trades > 0 else 0

        msg = (
            f"💓 <b>Bot Heartbeat</b>\n"
            f"{'─' * 28}\n"
            f"✅ Bot is alive and running\n"
            f"⏱️  Uptime  : {hours}h {mins}m\n"
            f"🕐 Time    : {now.strftime('%H:%M')} UTC\n"
            f"💰 Balance : ${balance:,.2f}\n"
            f"📊 Trades  : {total_trades}  "
            f"W:{wins} L:{losses}"
            + (f"  WR:{wr:.0f}%" if total_trades > 0 else "") + "\n"
            f"<i>Next heartbeat in {HEARTBEAT_HOURS}h</i>"
        )
        ok = send(msg, silent=True)
        if ok:
            _last_heartbeat = now
            print("  📱 Telegram: Heartbeat sent")
        return ok
    return False


# ══════════════════════════════════════════════════════════════════════
#   ALERT 8 — ERROR / DISCONNECT
# ══════════════════════════════════════════════════════════════════════
def alert_error(error_msg: str):
    now = datetime.now(timezone.utc)
    msg = (
        f"⚠️ <b>BOT ERROR</b>\n"
        f"{'─' * 28}\n"
        f"🕐 {now.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"❌ Error: {error_msg}\n"
        f"{'─' * 28}\n"
        f"🔄 Bot will attempt to recover automatically\n"
        f"<i>Check terminal if this repeats</i>"
    )
    ok = send(msg)
    if ok:
        print("  📱 Telegram: Error alert sent")
    return ok


def alert_disconnected():
    msg = (
        f"🔌 <b>MT5 DISCONNECTED</b>\n"
        f"{'─' * 28}\n"
        f"Bot lost connection to MetaTrader 5\n"
        f"🔄 Attempting to reconnect...\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%H:%M')} UTC"
    )
    ok = send(msg)
    if ok:
        print("  📱 Telegram: Disconnect alert sent")
    return ok


def alert_reconnected(balance: float):
    msg = (
        f"✅ <b>MT5 RECONNECTED</b>\n"
        f"{'─' * 28}\n"
        f"Connection restored successfully\n"
        f"💰 Balance : ${balance:,.2f}\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%H:%M')} UTC\n"
        f"<i>Bot resuming normal operation</i>"
    )
    ok = send(msg)
    if ok:
        print("  📱 Telegram: Reconnect alert sent")
    return ok


def alert_bot_stopped(total_trades: int, wins: int,
                      losses: int, balance: float):
    now    = datetime.now(timezone.utc)
    uptime = now - _session_start
    hours  = int(uptime.total_seconds() // 3600)
    mins   = int((uptime.total_seconds() % 3600) // 60)
    wr     = (wins / total_trades * 100) if total_trades > 0 else 0
    net    = balance - 100000  # vs starting balance

    msg = (
        f"⛔ <b>Bot Stopped</b>\n"
        f"{'─' * 28}\n"
        f"🕐 Stopped  : {now.strftime('%H:%M')} UTC\n"
        f"⏱️  Uptime   : {hours}h {mins}m\n"
        f"{'─' * 28}\n"
        f"📊 Session Stats:\n"
        f"   Trades   : {total_trades}\n"
        f"   Wins     : {wins}  Losses: {losses}\n"
        + (f"   Win Rate : {wr:.1f}%\n" if total_trades > 0 else "") +
        f"{'─' * 28}\n"
        f"💰 Balance  : ${balance:,.2f}\n"
        f"<i>Manually stopped via Ctrl+C</i>"
    )
    ok = send(msg)
    if ok:
        print("  📱 Telegram: Bot stopped alert sent")
    return ok


# ══════════════════════════════════════════════════════════════════════
#   TEST FUNCTION — run this file directly to test all alerts
# ══════════════════════════════════════════════════════════════════════
def test_all_alerts():
    print("\n🧪 Testing all Telegram alerts...\n")

    print("1. Testing bot started alert...")
    alert_bot_started(100000.67, {
        "ema_fast": 9, "ema_slow": 21,
        "min_filters": 3, "sess_start": 13,
        "sess_end": 17, "atr_sl": 2.0, "atr_tp": 3.0
    })
    time.sleep(1)

    print("2. Testing trade opened alert...")
    alert_trade_opened(
        signal="BUY", entry=3150.50, sl=3093.30, tp=3264.10,
        lot=0.01, ticket=12345678,
        rsi=54.2, atr=28.60,
        filters_passed=3, h4_trend="BULLISH",
        session="London-NY Overlap ⭐"
    )
    time.sleep(1)

    print("3. Testing trade closed TP alert...")
    alert_trade_closed(
        signal="BUY", entry=3150.50, exit_price=3264.10,
        pnl=85.70, ticket=12345678, reason="TP"
    )
    time.sleep(1)

    print("4. Testing trade closed SL alert...")
    alert_trade_closed(
        signal="SELL", entry=3180.00, exit_price=3237.60,
        pnl=-43.20, ticket=12345679, reason="SL"
    )
    time.sleep(1)

    print("5. Testing signal blocked alert...")
    alert_signal_blocked(
        signal="BUY", blocked_by=["RSI", "Session"],
        rsi=72.4, h4_trend="BEARISH",
        session="Tokyo (avoid)"
    )
    time.sleep(1)

    print("6. Testing trailing stop alert...")
    alert_trail_moved(
        ticket=12345678, signal="BUY",
        old_sl=3093.30, new_sl=3142.80, price=3199.40
    )
    time.sleep(1)

    print("7. Testing heartbeat alert...")
    _last_heartbeat_backup = globals().get("_last_heartbeat")
    globals()["_last_heartbeat"] = None
    maybe_send_heartbeat(5, 3, 2, 100088.45)
    time.sleep(1)

    print("8. Testing daily summary alert...")
    global _last_summary_day
    _last_summary_day = None
    alert_daily_summary(5, 3, 2, 88.45, 100088.45)
    time.sleep(1)

    print("9. Testing error alert...")
    alert_error("MT5 returned no data for XAUUSD H1")
    time.sleep(1)

    print("10. Testing bot stopped alert...")
    alert_bot_stopped(5, 3, 2, 100088.45)

    print("\n✅ All 10 test alerts sent!")
    print("   Check your Telegram — you should have 10 messages")
    print("   If you got them all, the system is working perfectly 🎯")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║       XAUUSD TELEGRAM ALERT SYSTEM — TEST           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Bot Token : ...{BOT_TOKEN[-10:]:<38}║")
    print(f"║  Chat ID   : {CHAT_ID:<42}║")
    print("╚══════════════════════════════════════════════════════╝\n")

    print("Sending test message first...")
    ok = send("🔧 <b>Connection Test</b>\n\nTelegram alerts are working!\nXAUUSD bot is ready to send notifications.")
    if ok:
        print("✅ Connection successful!\n")
        test_all_alerts()
    else:
        print("❌ Failed to connect. Check your Bot Token and Chat ID.")