import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone, timedelta
from itertools import product
import warnings
warnings.filterwarnings("ignore")

SYMBOL           = "XAUUSD"
STARTING_BALANCE = 10000.0
RISK_PCT         = 1.0
YEARS_BACK       = 2
SPREAD_PIPS      = 0.30
COMMISSION       = 0.07

PARAM_GRID = {
    "ema_fast"    : [9],
    "ema_slow"    : [21],
    "rsi_ob"      : [65, 70, 75],
    "rsi_os"      : [25, 30, 35],
    "rsi_mid"     : [45, 50, 55],
    "atr_sl_mult" : [1.0, 1.5, 2.0],
    "atr_tp_mult" : [2.0, 2.5, 3.0],
    "sess_start"  : [8, 10, 13],
    "sess_end"    : [17, 20, 22],
    "min_filters" : [2, 3],
}

def fetch_data():
    print("Connecting to MT5...")
    if not mt5.initialize():
        print(f"MT5 failed: {mt5.last_error()}")
        return None, None
    account = mt5.account_info()
    print(f"Connected: {account.login}")
    end_date   = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365 * YEARS_BACK)
    print(f"Fetching {YEARS_BACK} years XAUUSD H1...")
    rates_h1 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, start_date, end_date)
    rates_h4 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H4, start_date - timedelta(days=30), end_date)
    mt5.shutdown()
    if rates_h1 is None or len(rates_h1) < 100:
        print("No data")
        return None, None
    def to_df(rates):
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df
    df_h1, df_h4 = to_df(rates_h1), to_df(rates_h4)
    print(f"Got {len(df_h1):,} H1 candles")
    return df_h1, df_h4

def prepare_base(df_h1, df_h4):
    print("Preparing base indicators...")
    df = df_h1.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],(df["high"]-prev_close).abs(),(df["low"]-prev_close).abs()],axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df_h4 = df_h4.copy()
    df_h4["EMA50_H4"] = df_h4["close"].ewm(span=50, adjust=False).mean()
    df_h4["H4_Bull"]  = df_h4["close"] > df_h4["EMA50_H4"]
    df["H4_Bull"]     = df_h4["H4_Bull"].reindex(df.index, method="ffill").fillna(False)
    lb = 5
    highs = df["high"].values
    lows  = df["low"].values
    df["LastSwingLow"]  = np.nan
    df["LastSwingHigh"] = np.nan
    for i in range(lb, len(df) - lb):
        if all(lows[i]<lows[i-j] for j in range(1,lb+1)) and all(lows[i]<lows[i+j] for j in range(1,lb+1)):
            df.iloc[i, df.columns.get_loc("LastSwingLow")] = lows[i]
        if all(highs[i]>highs[i-j] for j in range(1,lb+1)) and all(highs[i]>highs[i+j] for j in range(1,lb+1)):
            df.iloc[i, df.columns.get_loc("LastSwingHigh")] = highs[i]
    df["LastSwingLow"]  = df["LastSwingLow"].ffill()
    df["LastSwingHigh"] = df["LastSwingHigh"].ffill()
    print("Base data ready")
    return df

def fast_backtest(df_base, params):
    ef=params["ema_fast"]; es=params["ema_slow"]
    rob=params["rsi_ob"]; ros=params["rsi_os"]; rmd=params["rsi_mid"]
    asl=params["atr_sl_mult"]; atp=params["atr_tp_mult"]
    ss=params["sess_start"]; se=params["sess_end"]; mf=params["min_filters"]
    df = df_base.copy()
    df["EMA_f"]   = df["close"].ewm(span=ef, adjust=False).mean()
    df["EMA_s"]   = df["close"].ewm(span=es, adjust=False).mean()
    df["EMA_pos"] = np.where(df["EMA_f"] > df["EMA_s"], 1, -1)
    bull = (df["EMA_pos"]==1)&(df["EMA_pos"].shift(1)==-1)
    bear = (df["EMA_pos"]==-1)&(df["EMA_pos"].shift(1)==1)
    delta=df["close"].diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    al=loss.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    df["RSI"]=100-(100/(1+ag/al))
    df["InSess"]=df.index.map(lambda t: ss<=t.hour<se and t.weekday()<5)
    balance=STARTING_BALANCE; trades=[]; equity=[STARTING_BALANCE]; open_t=None
    for i in range(50, len(df)):
        row=df.iloc[i]; rsi=row["RSI"]; atr=row["ATR"]
        if open_t:
            ht=hsl=False
            if open_t["t"]=="BUY":
                if row["low"]<=open_t["sl"]: hsl=True
                if row["high"]>=open_t["tp"]: ht=True
            else:
                if row["high"]>=open_t["sl"]: hsl=True
                if row["low"]<=open_t["tp"]: ht=True
            re=(open_t["t"]=="SELL" and rsi<=ros) or (open_t["t"]=="BUY" and rsi>=rob)
            ee=(open_t["t"]=="SELL" and bull.iloc[i]) or (open_t["t"]=="BUY" and bear.iloc[i])
            if ht or hsl or re or ee:
                ep=open_t["tp"] if ht else (open_t["sl"] if hsl else row["close"])
                er="TP" if ht else ("SL" if hsl else ("RSI" if re else "EMA"))
                lot=open_t["lot"]
                pnl=((ep-open_t["e"]) if open_t["t"]=="BUY" else (open_t["e"]-ep))*lot*100-COMMISSION
                balance+=pnl
                trades.append({"pnl":pnl,"result":"WIN" if pnl>0 else "LOSS","exit":er})
                open_t=None
            equity.append(balance)
            continue
        equity.append(balance)
        if not (bull.iloc[i] or bear.iloc[i]): continue
        sig="BUY" if bull.iloc[i] else "SELL"
        rsi_ok=(rsi>rmd and rsi<rob) if sig=="BUY" else (rsi<rmd and rsi>ros)
        mtf_ok=bool(row["H4_Bull"]) if sig=="BUY" else not bool(row["H4_Bull"])
        sess_ok=bool(row["InSess"])
        if sum([rsi_ok,mtf_ok,sess_ok])<mf: continue
        entry=row["close"]+(SPREAD_PIPS if sig=="BUY" else -SPREAD_PIPS)
        if sig=="BUY": sl=entry-atr*asl; tp=entry+atr*atp
        else: sl=entry+atr*asl; tp=entry-atr*atp
        if sig=="BUY" and not np.isnan(row["LastSwingLow"]): sl=max(row["LastSwingLow"]-0.5,sl)
        elif sig=="SELL" and not np.isnan(row["LastSwingHigh"]): sl=min(row["LastSwingHigh"]+0.5,sl)
        sl_dist=abs(entry-sl)
        if sl_dist<0.1: continue
        lot=round(max(0.01,min(balance*(RISK_PCT/100)/(sl_dist*100),1.0)),2)
        open_t={"t":sig,"e":entry,"sl":sl,"tp":tp,"lot":lot}
    if not trades: return None
    t_df=pd.DataFrame(trades)
    wins=t_df[t_df["result"]=="WIN"]; losses=t_df[t_df["result"]=="LOSS"]
    n=len(t_df); wr=len(wins)/n*100
    gp=wins["pnl"].sum() if not wins.empty else 0
    gl=losses["pnl"].sum() if not losses.empty else 0
    pf=abs(gp/gl) if gl!=0 else 9.99
    eq=np.array(equity); rm=np.maximum.accumulate(eq); dd=((eq-rm)/rm*100).min()
    ret=np.diff(eq)/eq[:-1]
    sh=(ret.mean()/ret.std()*np.sqrt(252*24)) if ret.std()>0 else 0
    net=gp+gl; ret_pct=(balance/STARTING_BALANCE-1)*100
    return {
        "params":params,"trades":n,"win_rate":round(wr,1),
        "profit_factor":round(pf,2),"net_profit":round(net,2),
        "return_pct":round(ret_pct,2),"max_drawdown":round(dd,2),
        "sharpe":round(sh,3),
        "exit_tp":t_df[t_df["exit"]=="TP"].shape[0],
        "exit_sl":t_df[t_df["exit"]=="SL"].shape[0],
        "score":round(wr*0.3+pf*20+sh*10+min(ret_pct,50)*0.5+max(dd,-30)*0.5,2)
    }

def run_optimization(df_base):
    keys=list(PARAM_GRID.keys()); values=list(PARAM_GRID.values())
    combos=list(product(*values)); total=len(combos)
    print(f"\nTesting {total} parameter combinations...")
    results=[]
    for idx,combo in enumerate(combos):
        params=dict(zip(keys,combo))
        if params["atr_tp_mult"]<=params["atr_sl_mult"]: continue
        if params["sess_start"]>=params["sess_end"]: continue
        result=fast_backtest(df_base,params)
        if result is None or result["trades"]<10: continue
        results.append(result)
        if (idx+1)%50==0:
            pct=(idx+1)/total*100
            if results:
                best=max(results,key=lambda x:x["score"])
                print(f"  Progress: {idx+1}/{total} ({pct:.0f}%)  Best WR:{best['win_rate']}%  PF:{best['profit_factor']}  Score:{best['score']:.1f}")
    print(f"\nTested {len(results)} valid combinations")
    return pd.DataFrame(results)

def print_results(results_df):
    if results_df.empty:
        print("No results"); return None
    results_df=results_df.sort_values("score",ascending=False)
    top10=results_df.head(10)
    print("\n")
    print("="*65)
    print("  TOP 10 PARAMETER COMBINATIONS (ranked by score)")
    print("="*65)
    print(f"  {'#':<3} {'WR%':<7} {'PF':<6} {'Net$':<9} {'DD%':<8} {'Sh':<6} {'Trades'}")
    print("  "+"-"*55)
    for i,(_,row) in enumerate(top10.iterrows(),1):
        p=row["params"]
        print(f"  {i:<3} {row['win_rate']:<7} {row['profit_factor']:<6} {row['net_profit']:<9.0f} {row['max_drawdown']:<8.1f} {row['sharpe']:<6.2f} {row['trades']}")
    best=results_df.iloc[0]; p=best["params"]
    print("\n")
    print("="*60)
    print("  BEST PARAMETER SET FOUND")
    print("="*60)
    print(f"  Win Rate      : {best['win_rate']}%")
    print(f"  Profit Factor : {best['profit_factor']}")
    print(f"  Net Profit    : ${best['net_profit']:.2f}")
    print(f"  Max Drawdown  : {best['max_drawdown']}%")
    print(f"  Sharpe Ratio  : {best['sharpe']}")
    print(f"  Total Trades  : {best['trades']}")
    print("-"*60)
    print("  COPY THESE INTO xauusd_bot.py:")
    print("-"*60)
    print(f"  RSI_OB        = {p['rsi_ob']}")
    print(f"  RSI_OS        = {p['rsi_os']}")
    print(f"  RSI_MID       = {p['rsi_mid']}")
    print(f"  ATR_SL_MULT   = {p['atr_sl_mult']}")
    print(f"  ATR_TP_MULT   = {p['atr_tp_mult']}")
    print(f"  SESSION_START = {p['sess_start']}")
    print(f"  SESSION_END   = {p['sess_end']}")
    print(f"  MIN_FILTERS   = {p['min_filters']}")
    print("="*60)
    print("\n  IMPROVEMENT vs ORIGINAL:")
    print(f"  Win Rate     : 32.9% -> {best['win_rate']}%  ({best['win_rate']-32.9:+.1f}pp)")
    print(f"  Profit Factor: 1.0   -> {best['profit_factor']}  ({best['profit_factor']-1.0:+.2f})")
    print(f"  Net Profit   : -$50  -> ${best['net_profit']:.0f}  ({best['net_profit']+50:+.0f})")
    return best

def draw_chart(results_df):
    print("\nDrawing optimization chart...")
    fig=plt.figure(figsize=(18,10)); fig.patch.set_facecolor("#0D1117")
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.4,wspace=0.35)
    axes=[fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor("#161B22"); ax.tick_params(colors="#8B949E",labelsize=8)
        ax.grid(color="#21262D",linestyle="--",linewidth=0.5,alpha=0.5)
        for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
    top=results_df.nlargest(200,"score")
    sc=axes[0].scatter(top["win_rate"],top["profit_factor"],c=top["score"],cmap="RdYlGn",alpha=0.7,s=30)
    axes[0].axvline(x=45,color="#DA3633",lw=1,linestyle="--",alpha=0.6)
    axes[0].axhline(y=1.5,color="#DA3633",lw=1,linestyle="--",alpha=0.6)
    axes[0].set_xlabel("Win Rate (%)",color="#8B949E"); axes[0].set_ylabel("Profit Factor",color="#8B949E")
    axes[0].set_title("Win Rate vs Profit Factor",color="#F0F6FC",fontsize=9,fontweight="bold")
    plt.colorbar(sc,ax=axes[0]).ax.tick_params(colors="#8B949E")
    axes[1].scatter(top["max_drawdown"],top["net_profit"],c=top["score"],cmap="RdYlGn",alpha=0.7,s=30)
    axes[1].axhline(y=0,color="#8B949E",lw=1,linestyle="--")
    axes[1].set_xlabel("Max Drawdown (%)",color="#8B949E"); axes[1].set_ylabel("Net Profit ($)",color="#8B949E")
    axes[1].set_title("Drawdown vs Net Profit",color="#F0F6FC",fontsize=9,fontweight="bold")
    for sl_val in PARAM_GRID["atr_sl_mult"]:
        subset=results_df[results_df["params"].apply(lambda p:p["atr_sl_mult"]==sl_val)]
        if not subset.empty: axes[2].hist(subset["win_rate"],bins=12,alpha=0.6,label=f"SL*{sl_val}")
    axes[2].set_xlabel("Win Rate (%)",color="#8B949E"); axes[2].set_ylabel("Count",color="#8B949E")
    axes[2].set_title("ATR SL Mult -> Win Rate",color="#F0F6FC",fontsize=9,fontweight="bold")
    axes[2].legend(facecolor="#21262D",edgecolor="#30363D",labelcolor="#F0F6FC",fontsize=8)
    for mf in PARAM_GRID["min_filters"]:
        subset=results_df[results_df["params"].apply(lambda p:p["min_filters"]==mf)]
        if not subset.empty: axes[3].hist(subset["profit_factor"],bins=12,alpha=0.6,label=f"MinFilt={mf}")
    axes[3].set_xlabel("Profit Factor",color="#8B949E"); axes[3].set_ylabel("Count",color="#8B949E")
    axes[3].set_title("Min Filters -> Profit Factor",color="#F0F6FC",fontsize=9,fontweight="bold")
    axes[3].legend(facecolor="#21262D",edgecolor="#30363D",labelcolor="#F0F6FC",fontsize=8)
    sess_scores={ss:results_df[results_df["params"].apply(lambda p:p["sess_start"]==ss)]["score"].mean() for ss in PARAM_GRID["sess_start"]}
    axes[4].bar([str(k) for k in sess_scores],[v for v in sess_scores.values()],color=["#58A6FF","#E3B341","#2EA043"],alpha=0.8)
    axes[4].set_xlabel("Session Start UTC",color="#8B949E"); axes[4].set_ylabel("Avg Score",color="#8B949E")
    axes[4].set_title("Session Start -> Avg Score",color="#F0F6FC",fontsize=9,fontweight="bold")
    top10=results_df.nlargest(10,"score")
    labels=[f"WR:{r['win_rate']}% PF:{r['profit_factor']}" for _,r in top10.iterrows()]
    colors=["#2EA043" if r["net_profit"]>0 else "#DA3633" for _,r in top10.iterrows()]
    axes[5].barh(range(len(top10)),top10["score"].values,color=colors,alpha=0.8)
    axes[5].set_yticks(range(len(top10))); axes[5].set_yticklabels(labels,fontsize=7,color="#8B949E")
    axes[5].set_xlabel("Score",color="#8B949E"); axes[5].set_title("Top 10 by Score",color="#F0F6FC",fontsize=9,fontweight="bold")
    fig.suptitle(f"XAUUSD Optimizer — {len(results_df)} combos tested",color="#F0F6FC",fontsize=13,fontweight="bold",y=0.98)
    plt.savefig("optimization_results.png",dpi=150,bbox_inches="tight",facecolor="#0D1117")
    print("Chart saved as optimization_results.png")
    plt.show()

if __name__=="__main__":
    total=1
    for v in PARAM_GRID.values(): total*=len(v)
    print(f"\nXAUUSD Strategy Optimizer — testing {total} combinations\n")
    df_h1,df_h4=fetch_data()
    if df_h1 is None:
        print("Make sure MT5 is open"); exit()
    df_base=prepare_base(df_h1,df_h4)
    results_df=run_optimization(df_base)
    if results_df.empty:
        print("No valid results"); exit()
    results_df.to_csv("all_optimization_results.csv",index=False)
    print(f"All results saved to all_optimization_results.csv")
    best=print_results(results_df)
    draw_chart(results_df)
    print("\nNEXT STEPS:")
    print("  1. Copy the settings above into xauusd_bot.py")
    print("  2. Run backtest_engine.py again to confirm improvement")
    print("  3. Run the live bot with optimized settings")