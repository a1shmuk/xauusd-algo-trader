"""
╔══════════════════════════════════════════════════════════════╗
║         XAUUSD PHASE 5 — UNIFIED OPTIMIZER                  ║
║         Optimizes ALL parameters across all 3 phases        ║
╠══════════════════════════════════════════════════════════════╣
║  Tests every combination of:                                 ║
║   Phase 1: RSI thresholds, ATR multipliers, session times   ║
║   Phase 2: SMC min score threshold                          ║
║   Phase 5: Unified min score threshold                      ║
║                                                              ║
║  Saves: phase5_optimization_results.csv                     ║
║         phase5_optimization_results.png                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone

# ─────────────────────────────────────────────
#  PARAMETER GRID
# ─────────────────────────────────────────────

PARAM_GRID = {
    "rsi_ob"        : [60, 65, 70],
    "rsi_os"        : [30, 35, 40],
    "atr_sl_mult"   : [1.5, 2.0, 2.5],
    "atr_tp_mult"   : [2.5, 3.0, 3.5],
    "sess_start"    : [12, 13, 14],
    "sess_end"      : [17, 18, 19],
    "smc_min_score" : [1, 2, 3],
    "min_unified"   : [5, 6, 7, 8],
}

# ─────────────────────────────────────────────
#  DATA + INDICATORS
# ─────────────────────────────────────────────

def fetch_data():
    print("Connecting to MT5...")
    if not mt5.initialize():
        raise RuntimeError("MT5 not running.")
    print("✅ Connected. Fetching 2 years of data...")
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end   = datetime.now(timezone.utc)
    r_h1  = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1, start, end)
    r_h4  = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H4, start, end)
    mt5.shutdown()
    if r_h1 is None or r_h4 is None:
        raise RuntimeError("Failed to fetch data.")
    df_h1 = pd.DataFrame(r_h1)
    df_h4 = pd.DataFrame(r_h4)
    for df in [df_h1, df_h4]:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
    print(f"✅ {len(df_h1)} H1 candles | {len(df_h4)} H4 candles\n")
    return df_h1, df_h4


def calc_indicators(df_h1, df_h4):
    df = df_h1.copy()
    c  = df['close']
    df['ema9']  = c.ewm(span=9,  adjust=False).mean()
    df['ema21'] = c.ewm(span=21, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - c.shift(1)).abs(),
        (df['low']  - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr']  = tr.ewm(span=14, adjust=False).mean()
    df['hour'] = df.index.hour
    df['cross'] = np.where(
        (df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1)), 1,
        np.where(
            (df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1)), -1, 0
        )
    )
    h4 = df_h4.copy()
    h4['ema50']    = h4['close'].ewm(span=50, adjust=False).mean()
    h4['h4_trend'] = np.where(h4['close'] > h4['ema50'], 1, -1)
    h4_trend = h4['h4_trend'].resample('1h').ffill()
    df = df.join(h4_trend.rename('h4_trend'), how='left')
    df['h4_trend'] = df['h4_trend'].ffill().fillna(0)

    # SMC proxy score
    w = 50
    rh = df['high'].rolling(w).max()
    rl = df['low'].rolling(w).min()
    smc = pd.Series(0, index=df.index)
    smc += ((df['close'] >= rh * 0.998) | (df['close'] <= rl * 1.002)).astype(int)
    body = (df['close'] - df['open']).abs()
    smc += (body > df['close'] * 0.003).astype(int)
    ph = df['high'].rolling(20).max().shift(1)
    pl = df['low'].rolling(20).min().shift(1)
    smc += ((df['close'] > ph) | (df['close'] < pl)).astype(int)
    trend_flip = (df['ema9'] > df['ema21']) != (df['ema9'].shift(5) > df['ema21'].shift(5))
    smc += trend_flip.astype(int)
    df['smc_score'] = smc.clip(0, 4)

    # Fundamental proxy: ATR% as volatility regime
    atr_pct = df['atr'] / df['close'] * 100
    df['fund_score'] = np.where(atr_pct < 0.3, 3,
                       np.where(atr_pct < 0.5, 2,
                       np.where(atr_pct < 0.8, 1, 0)))

    return df.dropna()


# ─────────────────────────────────────────────
#  BACKTEST
# ─────────────────────────────────────────────

def backtest(df, params, balance=10000.0):
    rsi_ob      = params['rsi_ob']
    rsi_os      = params['rsi_os']
    sl_m        = params['atr_sl_mult']
    tp_m        = params['atr_tp_mult']
    ss          = params['sess_start']
    se          = params['sess_end']
    smc_min     = params['smc_min_score']
    min_uni     = params['min_unified']

    trades   = []
    in_trade = False
    entry = sl = tp = d = None

    for i in range(50, len(df)):
        row   = df.iloc[i]
        price = row['close']

        if in_trade:
            hit_tp = (d == 1 and price >= tp) or (d == -1 and price <= tp)
            hit_sl = (d == 1 and price <= sl) or (d == -1 and price >= sl)
            if hit_tp or hit_sl:
                pnl = ((tp if hit_tp else sl) - entry) * d * 0.01 * 100 * 0.1
                balance += pnl
                trades.append(pnl)
                in_trade = False
            continue

        if row['cross'] == 0:
            continue

        d_now = int(row['cross'])
        dir_s = "BUY" if d_now == 1 else "SELL"
        rsi   = row['rsi']
        rsi_ok  = (dir_s == "BUY"  and 50 <= rsi <= rsi_ob) or \
                  (dir_s == "SELL" and rsi_os <= rsi <= 50)
        mtf_ok  = (d_now == 1 and row['h4_trend'] == 1) or \
                  (d_now == -1 and row['h4_trend'] == -1)
        sess_ok = ss <= row['hour'] <= se
        p1 = sum([rsi_ok, mtf_ok, sess_ok])
        p2 = int(row['smc_score']) if int(row['smc_score']) >= smc_min else 0
        p4 = int(row['fund_score'])
        total = p1 + p2 + p4

        if total < min_uni:
            continue

        in_trade = True
        d        = d_now
        entry    = price
        atr      = row['atr']
        sl       = price - atr * sl_m * d
        tp       = price + atr * tp_m * d

    if not trades:
        return {'trades':0,'win_rate':0,'profit_factor':0,'net_pnl':0,'max_dd':0,'sharpe':0,'composite':-999}

    wins  = [p for p in trades if p > 0]
    loss  = [p for p in trades if p <= 0]
    wr    = len(wins) / len(trades) * 100
    gw    = sum(wins) or 0
    gl    = abs(sum(loss)) or 1
    pf    = gw / gl
    net   = sum(trades)
    pnl_a = np.array(trades)
    sh    = float(np.mean(pnl_a) / (np.std(pnl_a) + 1e-9) * np.sqrt(252))

    peak = 0; bal = 0; mdd = 0
    for p in trades:
        bal += p; peak = max(peak, bal)
        mdd = max(mdd, (peak - bal) / (peak + 1e-9) * 100)

    comp = wr * 0.3 + pf * 20 + sh * 10 + net * 0.5 - mdd * 0.5
    return {'trades':len(trades),'win_rate':round(wr,2),'profit_factor':round(pf,3),
            'net_pnl':round(net,2),'max_dd':round(mdd,2),'sharpe':round(sh,3),
            'composite':round(comp,2)}


# ─────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────

def plot_results(df_res, best):
    valid = df_res[df_res['trades'] >= 10]
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0A1628')
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)
    gold = '#F0A500'

    def ax_s(ax, t):
        ax.set_facecolor('#0F2040')
        ax.set_title(t, color='white', fontsize=11)
        ax.tick_params(colors='#8BA0BE', labelsize=9)
        for s in ax.spines.values(): s.set_edgecolor('#162B50')

    ax1 = fig.add_subplot(gs[0,0])
    ax1.hist(valid['win_rate'], bins=30, color=gold, alpha=0.8, edgecolor='none')
    ax_s(ax1, 'Win Rate Distribution')
    if best: ax1.axvline(best['win_rate'], color='#1D9E75', lw=2, ls='--', label=f"Best: {best['win_rate']:.1f}%")
    ax1.legend(fontsize=8, labelcolor='white', facecolor='#0F2040')

    ax2 = fig.add_subplot(gs[0,1])
    ax2.hist(valid['profit_factor'].clip(0,5), bins=30, color='#1D9E75', alpha=0.8, edgecolor='none')
    ax_s(ax2, 'Profit Factor Distribution')
    if best: ax2.axvline(min(best['profit_factor'],5), color=gold, lw=2, ls='--', label=f"Best: {best['profit_factor']:.2f}")
    ax2.legend(fontsize=8, labelcolor='white', facecolor='#0F2040')

    ax3 = fig.add_subplot(gs[0,2])
    sc = ax3.scatter(valid['win_rate'], valid['profit_factor'].clip(0,5),
                     c=valid['composite'], cmap='YlOrRd', alpha=0.4, s=6)
    ax_s(ax3, 'Win Rate vs Profit Factor')
    plt.colorbar(sc, ax=ax3).ax.tick_params(colors='#8BA0BE', labelsize=8)
    if best: ax3.scatter(best['win_rate'], min(best['profit_factor'],5), color='#1D9E75', s=150, zorder=5, marker='*')

    ax4 = fig.add_subplot(gs[1,0])
    mn = valid.groupby('min_unified')['win_rate'].mean()
    ax4.plot(mn.index, mn.values, color='#1D9E75', lw=2, marker='o')
    ax_s(ax4, 'Unified Threshold vs Win Rate')
    ax4.set_xlabel('Min Unified Score', color='#8BA0BE', fontsize=9)

    ax5 = fig.add_subplot(gs[1,1])
    ms = valid.groupby('smc_min_score')['win_rate'].mean()
    ax5.bar(ms.index, ms.values, color=gold, alpha=0.8)
    ax_s(ax5, 'SMC Min Score vs Win Rate')
    ax5.set_xlabel('SMC Min Score', color='#8BA0BE', fontsize=9)

    ax6 = fig.add_subplot(gs[1,2])
    top = valid.nlargest(20, 'composite')
    ax6.barh(range(len(top)), top['composite'], color=gold, alpha=0.8)
    ax_s(ax6, 'Top 20 Composite Scores')
    ax6.set_yticks([])

    fig.suptitle('XAUUSD Phase 5 — Unified Optimizer Results',
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('phase5_optimization_results.png', dpi=150, bbox_inches='tight', facecolor='#0A1628')
    print("✅ Chart saved: phase5_optimization_results.png")
    plt.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    total  = len(combos)

    print("\n" + "="*60)
    print("  XAUUSD PHASE 5 — UNIFIED OPTIMIZER")
    print("="*60)
    print(f"\n  Combinations : {total:,}")
    print(f"  Phases       : Technical + SMC + Fundamental")
    print(f"  Data         : 2 years XAUUSD H1\n")

    try:
        df_h1, df_h4 = fetch_data()
        print("Pre-computing indicators...")
        df = calc_indicators(df_h1, df_h4)
        print(f"✅ Ready — {len(df):,} rows\n")
        print(f"Testing {total:,} combinations...\n")

        results    = []
        best_score = -999
        best_params = None

        for idx, combo in enumerate(combos):
            params  = dict(zip(keys, combo))
            metrics = backtest(df, params)
            metrics.update(params)
            results.append(metrics)

            if metrics['composite'] > best_score and metrics['trades'] >= 10:
                best_score  = metrics['composite']
                best_params = metrics.copy()

            if (idx + 1) % 500 == 0:
                pct = (idx + 1) / total * 100
                bp  = best_params or {}
                print(f"  [{pct:5.1f}%] {idx+1:,}/{total:,}  |  "
                      f"Best WR={bp.get('win_rate',0):.1f}%  "
                      f"PF={bp.get('profit_factor',0):.2f}  "
                      f"Score={best_score:.1f}")

        df_res = pd.DataFrame(results).sort_values('composite', ascending=False)
        df_res.to_csv("phase5_optimization_results.csv", index=False)
        print(f"\n✅ Saved: phase5_optimization_results.csv")

        if best_params:
            print("\n" + "="*60)
            print("  BEST PARAMETERS FOUND")
            print("="*60)
            print(f"  Win Rate      : {best_params['win_rate']:.1f}%")
            print(f"  Profit Factor : {best_params['profit_factor']:.3f}")
            print(f"  Net P&L       : ${best_params['net_pnl']:,.2f}")
            print(f"  Max Drawdown  : {best_params['max_dd']:.2f}%")
            print(f"  Sharpe Ratio  : {best_params['sharpe']:.3f}")
            print(f"  Total Trades  : {best_params['trades']}")
            print(f"\n  ── Copy into xauusd_bot.py ──")
            print(f"  RSI_OB        = {best_params['rsi_ob']}")
            print(f"  RSI_OS        = {best_params['rsi_os']}")
            print(f"  ATR_SL_MULT   = {best_params['atr_sl_mult']}")
            print(f"  ATR_TP_MULT   = {best_params['atr_tp_mult']}")
            print(f"  SESSION_START = {best_params['sess_start']}")
            print(f"  SESSION_END   = {best_params['sess_end']}")
            print(f"  SMC_MIN_SCORE = {best_params['smc_min_score']}")
            print(f"  MIN_UNIFIED   = {best_params['min_unified']}")

        plot_results(df_res, best_params)
        print("\n✅ Phase 5 optimization complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure MetaTrader 5 is running.")
