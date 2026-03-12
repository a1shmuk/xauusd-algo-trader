import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
#
#   XAUUSD PHASE 3 — BACKTEST ENGINE
#   ─────────────────────────────────────────────────────────────────
#   Tests the Phase 1 strategy on 2 years of real XAUUSD H1 data
#   fetched directly from MT5.
#
#   What it measures:
#     • Win Rate, Profit Factor, Expectancy
#     • Sharpe Ratio, Sortino Ratio, Calmar Ratio
#     • Max Drawdown, Recovery Factor
#     • Monthly P&L breakdown
#     • Best / Worst trades
#     • Equity curve visualization
#
# ══════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────┐
# │               ⚙️  BACKTEST SETTINGS                            │
# └─────────────────────────────────────────────────────────────────┘

SYMBOL          = "XAUUSD"
STARTING_BALANCE= 10000.0     # Simulated starting account balance ($)
RISK_PCT        = 1.0         # Risk per trade (% of balance)
YEARS_BACK      = 2           # How many years of data to test

# Strategy parameters (must match live bot)
EMA_FAST        = 9
EMA_SLOW        = 21
RSI_PERIOD      = 14
RSI_OB          = 70
RSI_OS          = 30
RSI_MID         = 50
ATR_PERIOD      = 14
ATR_SL_MULT     = 1.5
ATR_TP_MULT     = 3.0
MTF_EMA         = 50
SESSION_START   = 8
SESSION_END     = 17
MIN_FILTERS     = 2
SWING_BARS      = 5
SPREAD_PIPS     = 0.30        # Typical XAUUSD spread ($)
COMMISSION      = 0.07        # Commission per 0.01 lot round-trip ($)


# ══════════════════════════════════════════════════════════════════════
#   STEP 1 — FETCH HISTORICAL DATA FROM MT5
# ══════════════════════════════════════════════════════════════════════
def fetch_data():
    print("🔌 Connecting to MT5...")
    if not mt5.initialize():
        print(f"❌ MT5 failed: {mt5.last_error()}")
        return None, None

    account = mt5.account_info()
    print(f"✅ Connected: {account.login} | Balance: ${account.balance:,.2f}")

    # Date range
    end_date   = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365 * YEARS_BACK)

    print(f"\n📥 Fetching {YEARS_BACK} years of XAUUSD H1 data...")
    print(f"   From : {start_date.strftime('%Y-%m-%d')}")
    print(f"   To   : {end_date.strftime('%Y-%m-%d')}")

    # H1 data for strategy
    rates_h1 = mt5.copy_rates_range(
        SYMBOL, mt5.TIMEFRAME_H1,
        start_date, end_date
    )

    # H4 data for MTF filter
    rates_h4 = mt5.copy_rates_range(
        SYMBOL, mt5.TIMEFRAME_H4,
        start_date - timedelta(days=30), end_date
    )

    mt5.shutdown()

    if rates_h1 is None or len(rates_h1) < 100:
        print("❌ No H1 data received")
        return None, None

    df_h1 = pd.DataFrame(rates_h1)
    df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
    df_h1.set_index("time", inplace=True)

    df_h4 = pd.DataFrame(rates_h4)
    df_h4["time"] = pd.to_datetime(df_h4["time"], unit="s")
    df_h4.set_index("time", inplace=True)

    print(f"✅ H1 Candles : {len(df_h1):,}")
    print(f"✅ H4 Candles : {len(df_h4):,}")
    print(f"   Date range : {df_h1.index[0].strftime('%Y-%m-%d')} → "
          f"{df_h1.index[-1].strftime('%Y-%m-%d')}")

    return df_h1, df_h4


# ══════════════════════════════════════════════════════════════════════
#   STEP 2 — CALCULATE ALL INDICATORS
# ══════════════════════════════════════════════════════════════════════
def calculate_indicators(df_h1, df_h4):
    print("\n⚙️  Calculating indicators...")

    # ── H1 Indicators ──
    df = df_h1.copy()

    # EMA
    df["EMA9"]  = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # RSI
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD,
                        adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD,
                        adjust=False).mean()
    rs       = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/ATR_PERIOD, min_periods=ATR_PERIOD,
                       adjust=False).mean()

    # EMA crossover signals
    df["EMA_pos"]    = np.where(df["EMA9"] > df["EMA21"], 1, -1)
    df["BullCross"]  = (df["EMA_pos"] == 1) & (df["EMA_pos"].shift(1) == -1)
    df["BearCross"]  = (df["EMA_pos"] == -1) & (df["EMA_pos"].shift(1) == 1)

    # ── H4 EMA50 (MTF filter) — merge onto H1 ──
    df_h4 = df_h4.copy()
    df_h4["EMA50_H4"] = df_h4["close"].ewm(span=MTF_EMA, adjust=False).mean()
    df_h4["H4_Bull"]  = df_h4["close"] > df_h4["EMA50_H4"]

    # Forward-fill H4 data onto H1 timestamps
    df["H4_Bull"] = df_h4["H4_Bull"].reindex(df.index, method="ffill")
    df["H4_Bull"] = df["H4_Bull"].fillna(False)

    # ── Session filter ──
    df["InSession"] = df.index.map(
        lambda t: SESSION_START <= t.hour < SESSION_END
                  and t.weekday() < 5
    )

    # ── Swing High/Low ──
    lb = SWING_BARS
    swing_lows  = []
    swing_highs = []
    highs = df["high"].values
    lows  = df["low"].values

    for i in range(lb, len(df) - lb):
        if all(lows[i] < lows[i-j] for j in range(1, lb+1)) and \
           all(lows[i] < lows[i+j] for j in range(1, lb+1)):
            swing_lows.append(i)
        if all(highs[i] > highs[i-j] for j in range(1, lb+1)) and \
           all(highs[i] > highs[i+j] for j in range(1, lb+1)):
            swing_highs.append(i)

    # Create last swing series
    df["LastSwingLow"]  = np.nan
    df["LastSwingHigh"] = np.nan
    for i in swing_lows:
        df.iloc[i, df.columns.get_loc("LastSwingLow")] = lows[i]
    for i in swing_highs:
        df.iloc[i, df.columns.get_loc("LastSwingHigh")] = highs[i]

    df["LastSwingLow"]  = df["LastSwingLow"].ffill()
    df["LastSwingHigh"] = df["LastSwingHigh"].ffill()

    print(f"✅ Indicators calculated on {len(df):,} candles")
    return df


# ══════════════════════════════════════════════════════════════════════
#   STEP 3 — RUN BACKTEST
# ══════════════════════════════════════════════════════════════════════
def run_backtest(df):
    print("\n🚀 Running backtest...")

    balance  = STARTING_BALANCE
    equity   = STARTING_BALANCE
    trades   = []
    equity_curve = []
    open_trade   = None

    for i in range(50, len(df)):
        row      = df.iloc[i]
        bar_time = df.index[i]
        price    = row["close"]
        rsi      = row["RSI"]
        atr      = row["ATR"]

        # Track equity
        if open_trade:
            # Calculate unrealised P&L
            if open_trade["type"] == "BUY":
                unreal = (price - open_trade["entry"]) * open_trade["lot_size"] * 100
            else:
                unreal = (open_trade["entry"] - price) * open_trade["lot_size"] * 100
            equity = balance + unreal
        else:
            equity = balance

        equity_curve.append({
            "time"   : bar_time,
            "equity" : equity,
            "balance": balance
        })

        # ── Manage open trade ──
        if open_trade:
            hit_sl = False
            hit_tp = False

            if open_trade["type"] == "BUY":
                if row["low"]  <= open_trade["sl"]: hit_sl = True
                if row["high"] >= open_trade["tp"]: hit_tp = True
            else:
                if row["high"] >= open_trade["sl"]: hit_sl = True
                if row["low"]  <= open_trade["tp"]: hit_tp = True

            # RSI exit
            rsi_exit = False
            if open_trade["type"] == "SELL" and rsi <= RSI_OS:
                rsi_exit = True
            if open_trade["type"] == "BUY"  and rsi >= RSI_OB:
                rsi_exit = True

            # EMA crossover exit
            ema_exit = False
            if open_trade["type"] == "SELL" and row["BullCross"]:
                ema_exit = True
            if open_trade["type"] == "BUY"  and row["BearCross"]:
                ema_exit = True

            if hit_tp or hit_sl or rsi_exit or ema_exit:
                # Determine exit price
                if hit_tp:
                    exit_price  = open_trade["tp"]
                    exit_reason = "TP"
                elif hit_sl:
                    exit_price  = open_trade["sl"]
                    exit_reason = "SL"
                elif rsi_exit:
                    exit_price  = price
                    exit_reason = "RSI_exit"
                else:
                    exit_price  = price
                    exit_reason = "EMA_exit"

                # Calculate P&L
                lot   = open_trade["lot_size"]
                if open_trade["type"] == "BUY":
                    pnl = (exit_price - open_trade["entry"]) * lot * 100
                else:
                    pnl = (open_trade["entry"] - exit_price) * lot * 100

                pnl -= COMMISSION  # Deduct commission

                balance += pnl
                open_trade["exit_price"]  = exit_price
                open_trade["exit_time"]   = bar_time
                open_trade["exit_reason"] = exit_reason
                open_trade["pnl"]         = round(pnl, 2)
                open_trade["balance"]     = round(balance, 2)
                open_trade["result"]      = "WIN" if pnl > 0 else "LOSS"
                open_trade["bars_held"]   = i - open_trade["entry_bar"]
                trades.append(open_trade)
                open_trade = None
            continue  # Don't look for new entries while in trade

        # ── Look for new entry ──
        is_bull = row["BullCross"]
        is_bear = row["BearCross"]
        if not is_bull and not is_bear:
            continue

        signal = "BUY" if is_bull else "SELL"

        # Engine 1A — RSI
        if signal == "BUY":
            rsi_ok = (rsi > RSI_MID) and (rsi < RSI_OB)
        else:
            rsi_ok = (rsi < RSI_MID) and (rsi > RSI_OS)

        # Engine 1C — MTF
        mtf_ok = bool(row["H4_Bull"]) if signal == "BUY" \
                 else not bool(row["H4_Bull"])

        # Engine 1D — Session
        sess_ok = bool(row["InSession"])

        # Count filters
        passed = sum([rsi_ok, mtf_ok, sess_ok])
        if passed < MIN_FILTERS:
            continue

        # Engine 1B — ATR SL/TP
        entry = price + SPREAD_PIPS if signal == "BUY" else price - SPREAD_PIPS

        if signal == "BUY":
            atr_sl = entry - atr * ATR_SL_MULT
            atr_tp = entry + atr * ATR_TP_MULT
        else:
            atr_sl = entry + atr * ATR_SL_MULT
            atr_tp = entry - atr * ATR_TP_MULT

        # Engine 1E — Swing SL refinement
        if signal == "BUY" and not np.isnan(row["LastSwingLow"]):
            struct_sl = row["LastSwingLow"] - 0.5
            final_sl  = max(struct_sl, atr_sl)
        elif signal == "SELL" and not np.isnan(row["LastSwingHigh"]):
            struct_sl = row["LastSwingHigh"] + 0.5
            final_sl  = min(struct_sl, atr_sl)
        else:
            final_sl = atr_sl

        # Position sizing — risk fixed % of balance
        sl_dist  = abs(entry - final_sl)
        if sl_dist < 0.1:
            continue
        risk_amt = balance * (RISK_PCT / 100)
        lot_size = round(risk_amt / (sl_dist * 100), 2)
        lot_size = max(0.01, min(lot_size, 1.0))  # Cap between 0.01 and 1.0

        open_trade = {
            "type"       : signal,
            "entry"      : round(entry, 2),
            "entry_time" : bar_time,
            "entry_bar"  : i,
            "sl"         : round(final_sl, 2),
            "tp"         : round(atr_tp, 2),
            "lot_size"   : lot_size,
            "rsi_at_entry": round(rsi, 1),
            "atr_at_entry": round(atr, 2),
            "filters_passed": passed,
            "month"      : bar_time.strftime("%Y-%m"),
        }

    # Close any remaining open trade at last price
    if open_trade:
        last_price = df["close"].iloc[-1]
        lot = open_trade["lot_size"]
        if open_trade["type"] == "BUY":
            pnl = (last_price - open_trade["entry"]) * lot * 100
        else:
            pnl = (open_trade["entry"] - last_price) * lot * 100
        pnl -= COMMISSION
        balance += pnl
        open_trade.update({
            "exit_price"  : round(last_price, 2),
            "exit_time"   : df.index[-1],
            "exit_reason" : "END_OF_DATA",
            "pnl"         : round(pnl, 2),
            "balance"     : round(balance, 2),
            "result"      : "WIN" if pnl > 0 else "LOSS",
            "bars_held"   : len(df) - open_trade["entry_bar"],
        })
        trades.append(open_trade)

    print(f"✅ Backtest complete — {len(trades)} trades simulated")
    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


# ══════════════════════════════════════════════════════════════════════
#   STEP 4 — CALCULATE PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════
def calculate_metrics(trades_df, equity_df):
    if trades_df.empty:
        print("❌ No trades found")
        return {}

    wins   = trades_df[trades_df["result"] == "WIN"]
    losses = trades_df[trades_df["result"] == "LOSS"]

    total        = len(trades_df)
    n_wins       = len(wins)
    n_losses     = len(losses)
    win_rate     = n_wins / total * 100 if total > 0 else 0

    gross_profit = wins["pnl"].sum()   if not wins.empty   else 0
    gross_loss   = losses["pnl"].sum() if not losses.empty else 0
    net_profit   = gross_profit + gross_loss

    profit_factor = abs(gross_profit / gross_loss) \
                    if gross_loss != 0 else float("inf")

    avg_win  = wins["pnl"].mean()   if not wins.empty   else 0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # Drawdown
    equity_vals  = equity_df["equity"].values
    rolling_max  = np.maximum.accumulate(equity_vals)
    drawdowns    = (equity_vals - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()

    # Sharpe Ratio (annualised, using daily returns)
    equity_df["daily_return"] = equity_df["equity"].pct_change()
    daily_returns = equity_df["daily_return"].dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) \
             if daily_returns.std() > 0 else 0

    # Sortino Ratio (only uses downside deviation)
    neg_returns = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / neg_returns.std() * np.sqrt(252)) \
              if len(neg_returns) > 0 and neg_returns.std() > 0 else 0

    # Calmar Ratio
    annual_return = (equity_df["equity"].iloc[-1] / STARTING_BALANCE - 1) \
                    * (365 / YEARS_BACK / 365) * 100
    calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

    # Recovery Factor
    recovery = abs(net_profit / (STARTING_BALANCE * abs(max_drawdown) / 100)) \
               if max_drawdown != 0 else 0

    # Consecutive wins/losses
    results   = trades_df["result"].tolist()
    max_consec_wins  = max_consecutive(results, "WIN")
    max_consec_loss  = max_consecutive(results, "LOSS")

    # Exit reasons breakdown
    exit_counts = trades_df["exit_reason"].value_counts()

    metrics = {
        "Total Trades"      : total,
        "Win Rate"          : round(win_rate, 1),
        "Wins / Losses"     : f"{n_wins} / {n_losses}",
        "Net Profit ($)"    : round(net_profit, 2),
        "Gross Profit ($)"  : round(gross_profit, 2),
        "Gross Loss ($)"    : round(gross_loss, 2),
        "Profit Factor"     : round(profit_factor, 2),
        "Expectancy ($/trade)": round(expectancy, 2),
        "Avg Win ($)"       : round(avg_win, 2),
        "Avg Loss ($)"      : round(avg_loss, 2),
        "Max Drawdown (%)"  : round(max_drawdown, 2),
        "Sharpe Ratio"      : round(sharpe, 2),
        "Sortino Ratio"     : round(sortino, 2),
        "Calmar Ratio"      : round(calmar, 2),
        "Recovery Factor"   : round(recovery, 2),
        "Max Consec. Wins"  : max_consec_wins,
        "Max Consec. Losses": max_consec_loss,
        "Exit: TP"          : exit_counts.get("TP", 0),
        "Exit: SL"          : exit_counts.get("SL", 0),
        "Exit: RSI"         : exit_counts.get("RSI_exit", 0),
        "Exit: EMA"         : exit_counts.get("EMA_exit", 0),
        "Final Balance ($)" : round(equity_df["equity"].iloc[-1], 2),
        "Return (%)"        : round((equity_df["equity"].iloc[-1] /
                               STARTING_BALANCE - 1) * 100, 2),
    }
    return metrics

def max_consecutive(results, target):
    max_c = cur_c = 0
    for r in results:
        if r == target:
            cur_c += 1
            max_c = max(max_c, cur_c)
        else:
            cur_c = 0
    return max_c


# ══════════════════════════════════════════════════════════════════════
#   STEP 5 — PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════
def print_results(metrics, trades_df):
    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║        XAUUSD BACKTEST RESULTS — PHASE 1 BOT        ║")
    print("╠══════════════════════════════════════════════════════╣")

    # Grade each metric
    def grade(key, val):
        grades = {
            "Win Rate"        : [(55,"🟢 Excellent"),(50,"🟡 Good"),
                                  (45,"🟠 Acceptable"),(0,"🔴 Poor")],
            "Profit Factor"   : [(2.0,"🟢 Excellent"),(1.5,"🟡 Good"),
                                  (1.2,"🟠 Acceptable"),(0,"🔴 Poor")],
            "Sharpe Ratio"    : [(2.0,"🟢 Excellent"),(1.5,"🟡 Good"),
                                  (1.0,"🟠 Acceptable"),(0,"🔴 Poor")],
            "Max Drawdown (%)" : [(-5,"🟢 Excellent"),(-10,"🟡 Good"),
                                   (-20,"🟠 Acceptable"),(-100,"🔴 Poor")],
        }
        if key not in grades:
            return ""
        for threshold, label in grades[key]:
            if key == "Max Drawdown (%)" :
                if val >= threshold: return label
            else:
                if val >= threshold: return label
        return ""

    sections = [
        ("📊 OVERVIEW",
         ["Total Trades","Win Rate","Wins / Losses","Return (%)",
          "Final Balance ($)"]),
        ("💰 PROFITABILITY",
         ["Net Profit ($)","Gross Profit ($)","Gross Loss ($)",
          "Profit Factor","Expectancy ($/trade)","Avg Win ($)","Avg Loss ($)"]),
        ("📉 RISK METRICS",
         ["Max Drawdown (%)","Sharpe Ratio","Sortino Ratio",
          "Calmar Ratio","Recovery Factor"]),
        ("🔢 STREAKS",
         ["Max Consec. Wins","Max Consec. Losses"]),
        ("🚪 EXIT BREAKDOWN",
         ["Exit: TP","Exit: SL","Exit: RSI","Exit: EMA"]),
    ]

    for section_name, keys in sections:
        print(f"╠══════════════════════════════════════════════════════╣")
        print(f"║  {section_name:<50}║")
        print(f"╠══════════════════════════════════════════════════════╣")
        for key in keys:
            if key not in metrics:
                continue
            val    = metrics[key]
            g      = grade(key, val) if isinstance(val, (int, float)) else ""
            line   = f"  {key:<24}: {str(val):<12} {g}"
            print(f"║{line:<54}║")

    print("╚══════════════════════════════════════════════════════╝")

    # Strategy assessment
    print("\n📋 STRATEGY ASSESSMENT:")
    pf  = metrics.get("Profit Factor", 0)
    wr  = metrics.get("Win Rate", 0)
    dd  = metrics.get("Max Drawdown (%)", 0)
    sh  = metrics.get("Sharpe Ratio", 0)

    if pf > 1.5 and wr > 50 and dd > -15 and sh > 1.0:
        print("  🟢 STRONG STRATEGY — Ready for live trading with real money")
    elif pf > 1.2 and wr > 45:
        print("  🟡 DECENT STRATEGY — Works but needs optimization first")
    elif pf > 1.0:
        print("  🟠 MARGINAL STRATEGY — Profitable but fragile, optimize first")
    else:
        print("  🔴 WEAK STRATEGY — Not profitable, needs major rework")

    print("\n  What to improve:")
    if wr < 45:
        print("  → Win rate too low — tighten entry filters (MIN_FILTERS=3)")
    if pf < 1.5:
        print("  → Profit factor low — increase ATR_TP_MULT or reduce ATR_SL_MULT")
    if dd < -15:
        print("  → Drawdown too high — reduce LOT_SIZE or add daily loss limit")
    if sh < 1.0:
        print("  → Sharpe too low — add session filter (SESSION filter is critical)")

    # Monthly breakdown
    if not trades_df.empty:
        print("\n📅 MONTHLY P&L:")
        monthly = trades_df.groupby("month")["pnl"].sum().round(2)
        for month, pnl in monthly.items():
            bar   = "█" * int(abs(pnl) / 5)
            color = "+" if pnl >= 0 else "-"
            print(f"  {month}  {color}${abs(pnl):>8.2f}  {bar}")

    # Best and worst trades
    if not trades_df.empty:
        best  = trades_df.loc[trades_df["pnl"].idxmax()]
        worst = trades_df.loc[trades_df["pnl"].idxmin()]
        print(f"\n🏆 Best Trade  : {best['type']} "
              f"on {str(best['entry_time'])[:10]} "
              f"P&L: +${best['pnl']:.2f} "
              f"({best['exit_reason']})")
        print(f"💀 Worst Trade : {worst['type']} "
              f"on {str(worst['entry_time'])[:10]} "
              f"P&L: ${worst['pnl']:.2f} "
              f"({worst['exit_reason']})")


# ══════════════════════════════════════════════════════════════════════
#   STEP 6 — DRAW CHART
# ══════════════════════════════════════════════════════════════════════
def draw_results_chart(trades_df, equity_df, df):
    print("\n📊 Drawing results chart...")

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])   # Equity curve (full width)
    ax2 = fig.add_subplot(gs[1, 0])   # Price chart with signals
    ax3 = fig.add_subplot(gs[1, 1])   # Win/Loss distribution
    ax4 = fig.add_subplot(gs[2, 0])   # Monthly P&L
    ax5 = fig.add_subplot(gs[2, 1])   # Exit reason pie

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#8B949E", labelsize=8)
        ax.grid(color="#21262D", linestyle="--", linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    # ── Panel 1: Equity Curve ──
    ax1.plot(equity_df["time"], equity_df["equity"],
             color="#58A6FF", lw=1.5, label="Equity")
    ax1.plot(equity_df["time"], equity_df["balance"],
             color="#E3B341", lw=1.0, alpha=0.7,
             linestyle="--", label="Balance")
    ax1.axhline(y=STARTING_BALANCE, color="#8B949E",
                lw=0.8, linestyle=":", alpha=0.5)
    ax1.fill_between(equity_df["time"], equity_df["equity"],
                     STARTING_BALANCE,
                     where=equity_df["equity"] >= STARTING_BALANCE,
                     alpha=0.1, color="#2EA043")
    ax1.fill_between(equity_df["time"], equity_df["equity"],
                     STARTING_BALANCE,
                     where=equity_df["equity"] < STARTING_BALANCE,
                     alpha=0.1, color="#DA3633")
    ax1.set_title("Equity Curve", color="#F0F6FC",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("Balance ($)", color="#8B949E")
    ax1.legend(facecolor="#21262D", edgecolor="#30363D",
               labelcolor="#F0F6FC", fontsize=9)

    # ── Panel 2: Price + Trades (last 500 bars) ──
    plot_df = df.tail(500)
    ax2.plot(plot_df.index, plot_df["close"],
             color="#8B949E", lw=0.8, alpha=0.7)
    ax2.plot(plot_df.index, plot_df["EMA9"],
             color="#58A6FF", lw=1.5, label="EMA9")
    ax2.plot(plot_df.index, plot_df["EMA21"],
             color="#E3B341", lw=1.5, label="EMA21")

    # Plot recent trade entries
    if not trades_df.empty:
        recent_trades = trades_df[
            trades_df["entry_time"] >= plot_df.index[0]
        ]
        buys  = recent_trades[recent_trades["type"] == "BUY"]
        sells = recent_trades[recent_trades["type"] == "SELL"]
        if not buys.empty:
            ax2.scatter(buys["entry_time"], buys["entry"],
                        marker="^", color="#2EA043", s=60,
                        zorder=5, label="BUY")
        if not sells.empty:
            ax2.scatter(sells["entry_time"], sells["entry"],
                        marker="v", color="#DA3633", s=60,
                        zorder=5, label="SELL")

    ax2.set_title("Price Chart + Signals (last 500 bars)",
                  color="#F0F6FC", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Price ($)", color="#8B949E")
    ax2.legend(facecolor="#21262D", edgecolor="#30363D",
               labelcolor="#F0F6FC", fontsize=8)

    # ── Panel 3: P&L Distribution ──
    if not trades_df.empty:
        wins   = trades_df[trades_df["pnl"] > 0]["pnl"]
        losses = trades_df[trades_df["pnl"] < 0]["pnl"]
        ax3.hist(wins,   bins=20, color="#2EA043",
                 alpha=0.7, label=f"Wins ({len(wins)})")
        ax3.hist(losses, bins=20, color="#DA3633",
                 alpha=0.7, label=f"Losses ({len(losses)})")
        ax3.axvline(x=0, color="#F0F6FC", lw=1.0, linestyle="--")
        ax3.set_title("P&L Distribution",
                      color="#F0F6FC", fontsize=10, fontweight="bold")
        ax3.set_xlabel("P&L ($)", color="#8B949E")
        ax3.set_ylabel("Count", color="#8B949E")
        ax3.legend(facecolor="#21262D", edgecolor="#30363D",
                   labelcolor="#F0F6FC", fontsize=8)

    # ── Panel 4: Monthly P&L ──
    if not trades_df.empty:
        monthly = trades_df.groupby("month")["pnl"].sum()
        colors  = ["#2EA043" if v >= 0 else "#DA3633"
                   for v in monthly.values]
        ax4.bar(monthly.index, monthly.values,
                color=colors, alpha=0.8, width=0.6)
        ax4.axhline(y=0, color="#8B949E", lw=0.8)
        ax4.set_title("Monthly P&L ($)",
                      color="#F0F6FC", fontsize=10, fontweight="bold")
        ax4.set_ylabel("P&L ($)", color="#8B949E")
        ax4.set_xticklabels(monthly.index, rotation=45,
                             ha="right", fontsize=7)

    # ── Panel 5: Exit Reasons ──
    if not trades_df.empty:
        exit_counts = trades_df["exit_reason"].value_counts()
        pie_colors  = ["#2EA043","#DA3633","#E3B341","#58A6FF","#C792EA"]
        ax5.pie(exit_counts.values,
                labels=exit_counts.index,
                colors=pie_colors[:len(exit_counts)],
                autopct="%1.0f%%",
                textprops={"color": "#F0F6FC", "fontsize": 9},
                pctdistance=0.75)
        ax5.set_title("Exit Reasons",
                      color="#F0F6FC", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"XAUUSD Phase 1 Backtest — {YEARS_BACK} Years | "
        f"Start: ${STARTING_BALANCE:,.0f} | "
        f"Risk: {RISK_PCT}%/trade",
        color="#F0F6FC", fontsize=13, fontweight="bold", y=0.98
    )

    plt.savefig("backtest_results.png", dpi=150,
                bbox_inches="tight",
                facecolor="#0D1117")
    print("✅ Chart saved as backtest_results.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║       XAUUSD PHASE 3 — BACKTEST ENGINE              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Testing  : {YEARS_BACK} years of XAUUSD H1 data{'':<16}║")
    print(f"║  Capital  : ${STARTING_BALANCE:,.0f}{'':<36}║")
    print(f"║  Risk/trade: {RISK_PCT}% of balance{'':<28}║")
    print(f"║  Strategy : EMA {EMA_FAST}/{EMA_SLOW} + {MIN_FILTERS}/3 filters{'':<21}║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Step 1 — Fetch
    df_h1, df_h4 = fetch_data()
    if df_h1 is None:
        print("❌ Failed to get data. Make sure MT5 is open.")
        exit()

    # Step 2 — Indicators
    df = calculate_indicators(df_h1, df_h4)

    # Step 3 — Backtest
    trades_df, equity_df = run_backtest(df)

    if trades_df.empty:
        print("❌ No trades generated. Check your settings.")
        exit()

    # Step 4 — Metrics
    metrics = calculate_metrics(trades_df, equity_df)

    # Step 5 — Print results
    print_results(metrics, trades_df)

    # Step 6 — Save trades to CSV
    trades_df.to_csv("backtest_trades.csv", index=False)
    print(f"\n✅ All {len(trades_df)} trades saved to backtest_trades.csv")

    # Step 7 — Chart
    draw_results_chart(trades_df, equity_df, df)