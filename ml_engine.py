"""
╔══════════════════════════════════════════════════════════════╗
║         XAUUSD PHASE 6 — MACHINE LEARNING ENGINE            ║
║         XGBoost Signal Classifier                           ║
╠══════════════════════════════════════════════════════════════╣
║  What it does:                                               ║
║   1. Extracts 25+ features from Phase 1–5 signals           ║
║   2. Trains XGBoost to predict WIN vs LOSE                  ║
║   3. Adds final ML gate before every trade                  ║
║   4. Only takes trades where ML confidence >= threshold     ║
╠══════════════════════════════════════════════════════════════╣
║  Setup:                                                      ║
║   pip install xgboost scikit-learn joblib                   ║
║                                                              ║
║  Usage:                                                      ║
║   python ml_engine.py train   → trains model on 2yr data   ║
║   python ml_engine.py test    → tests on last 6 months      ║
║   python ml_engine.py live    → scores current market       ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional

# ── Try importing ML libraries ──
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, accuracy_score)
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML libraries not installed. Run:")
    print("   pip install xgboost scikit-learn joblib")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

MODEL_FILE     = "ml_model.joblib"
SCALER_FILE    = "ml_scaler.joblib"
FEATURES_FILE  = "ml_features.joblib"

ML_THRESHOLD   = 0.65    # Minimum win probability to allow trade (65%)
TRAIN_MONTHS   = 18      # Months of data to train on
TEST_MONTHS    = 6       # Months held out for testing

# ─────────────────────────────────────────────
#  ML SIGNAL RESULT
# ─────────────────────────────────────────────

@dataclass
class MLSignal:
    allows_trade: bool
    win_probability: float    # 0.0 – 1.0
    confidence: str           # 'HIGH', 'MEDIUM', 'LOW', 'REJECT'
    feature_importances: dict
    details: str

# ─────────────────────────────────────────────
#  1. DATA FETCHING
# ─────────────────────────────────────────────

def fetch_training_data(months: int = 24) -> pd.DataFrame:
    """Fetch XAUUSD H1 data from MT5 for training."""
    print(f"Fetching {months} months of XAUUSD H1 data...")

    if not mt5.initialize():
        raise RuntimeError("MT5 not running.")

    start = datetime.now(timezone.utc) - timedelta(days=months * 30)
    end   = datetime.now(timezone.utc)

    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1, start, end)
    rates_h4 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H4, start, end)
    mt5.shutdown()

    if rates is None:
        raise RuntimeError("Failed to fetch H1 data.")

    df    = pd.DataFrame(rates)
    df_h4 = pd.DataFrame(rates_h4)

    df['time']    = pd.to_datetime(df['time'], unit='s', utc=True)
    df_h4['time'] = pd.to_datetime(df_h4['time'], unit='s', utc=True)

    df.set_index('time', inplace=True)
    df_h4.set_index('time', inplace=True)

    print(f"✅ Loaded {len(df):,} H1 candles")
    return df, df_h4


# ─────────────────────────────────────────────
#  2. FEATURE ENGINEERING (25+ features)
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, df_h4: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 25+ features from raw OHLCV data.
    These represent everything Phase 1–5 looks at,
    converted into numbers XGBoost can learn from.
    """
    f = df.copy()
    c = f['close']
    h = f['high']
    l = f['low']
    o = f['open']

    # ── Phase 1: Technical features ──

    # EMA crossover
    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    f['ema_gap']        = (ema9 - ema21) / c           # normalized gap
    f['ema_gap_change'] = f['ema_gap'].diff()           # momentum of gap
    f['price_vs_ema50'] = (c - ema50) / c              # distance from EMA50

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    f['rsi']        = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    f['rsi_change'] = f['rsi'].diff()                  # RSI momentum

    # ATR and volatility
    prev_c = c.shift(1)
    tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    f['atr']        = tr.ewm(span=14, adjust=False).mean()
    f['atr_pct']    = f['atr'] / c                     # normalized ATR
    f['atr_change'] = f['atr'].diff() / f['atr']       # ATR trend

    # Candle body features
    f['body_size']  = (c - o).abs() / f['atr']        # body as ATR fraction
    f['upper_wick'] = (h - pd.concat([c, o], axis=1).max(axis=1)) / f['atr']
    f['lower_wick'] = (pd.concat([c, o], axis=1).min(axis=1) - l) / f['atr']
    f['is_bullish'] = (c > o).astype(int)

    # Session (hour of day)
    f['hour']       = f.index.hour
    f['in_session'] = ((f['hour'] >= 13) & (f['hour'] <= 17)).astype(int)
    f['day_of_week']= f.index.dayofweek                # 0=Mon, 4=Fri

    # ── Phase 2: SMC-inspired features ──

    # Rolling high/low (Order Block proxy)
    f['dist_to_high20'] = (h.rolling(20).max() - c) / f['atr']
    f['dist_to_low20']  = (c - l.rolling(20).min()) / f['atr']

    # Fair Value Gap proxy
    fvg_bull = (c.shift(2) < c) & ((h.shift(2) - l) < (c - l.shift(2)).abs())
    fvg_bear = (c.shift(2) > c) & ((h.shift(2) - l) > (c - l.shift(2)).abs())
    f['fvg_bull'] = fvg_bull.astype(int)
    f['fvg_bear'] = fvg_bear.astype(int)

    # Structure break proxy
    f['broke_high20'] = (c > h.rolling(20).max().shift(1)).astype(int)
    f['broke_low20']  = (c < l.rolling(20).min().shift(1)).astype(int)

    # ── Phase 4: Macro proxy features ──

    # Momentum over different timeframes (macro regime proxy)
    f['momentum_5']  = c.pct_change(5)                 # 5-bar momentum
    f['momentum_20'] = c.pct_change(20)                # 20-bar momentum
    f['momentum_50'] = c.pct_change(50)                # 50-bar momentum

    # Volatility regime
    f['vol_regime'] = f['atr_pct'].rolling(20).mean()  # rolling vol

    # ── H4 trend feature ──
    h4_ema50 = df_h4['close'].ewm(span=50, adjust=False).mean()
    h4_trend = (df_h4['close'] > h4_ema50).astype(int)
    h4_trend_h1 = h4_trend.resample('1h').ffill()
    f = f.join(h4_trend_h1.rename('h4_bullish'), how='left')
    f['h4_bullish'] = f['h4_bullish'].ffill().fillna(0.5)

    # ── Signal direction ──
    cross = np.where(
        (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1)), 1,
        np.where(
            (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1)), -1, 0
        )
    )
    f['signal'] = cross

    return f.dropna()


# ─────────────────────────────────────────────
#  3. LABEL GENERATION
# ─────────────────────────────────────────────

def generate_labels(df: pd.DataFrame,
                    atr_sl_mult: float = 2.0,
                    atr_tp_mult: float = 3.0) -> pd.DataFrame:
    """
    For each EMA crossover signal, simulate the trade outcome.
    Label = 1 if trade would WIN (hit TP before SL), 0 if LOSE.
    """
    df = df.copy()
    df['label']   = np.nan
    df['trade_id'] = np.nan

    signals = df[df['signal'] != 0].copy()
    trade_id = 0

    for idx, row in signals.iterrows():
        direction = row['signal']   # 1=BUY, -1=SELL
        entry     = row['close']
        atr       = row['atr']

        sl = entry - atr * atr_sl_mult * direction
        tp = entry + atr * atr_tp_mult * direction

        # Look forward up to 50 candles for outcome
        future = df.loc[idx:].iloc[1:51]
        label  = np.nan

        for _, frow in future.iterrows():
            if direction == 1:
                if frow['low']  <= sl: label = 0; break
                if frow['high'] >= tp: label = 1; break
            else:
                if frow['high'] >= sl: label = 0; break
                if frow['low']  <= tp: label = 1; break

        if not np.isnan(label):
            df.loc[idx, 'label']    = label
            df.loc[idx, 'trade_id'] = trade_id
            trade_id += 1

    labeled = df[df['label'].notna()].copy()
    labeled['label'] = labeled['label'].astype(int)
    print(f"✅ Generated {len(labeled):,} labeled trades")
    print(f"   Wins  : {labeled['label'].sum():,}  ({labeled['label'].mean()*100:.1f}%)")
    print(f"   Losses: {(1-labeled['label']).sum():,}  ({(1-labeled['label']).mean()*100:.1f}%)")
    return labeled


# ─────────────────────────────────────────────
#  4. FEATURE SELECTION
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'ema_gap', 'ema_gap_change', 'price_vs_ema50',
    'rsi', 'rsi_change',
    'atr_pct', 'atr_change',
    'body_size', 'upper_wick', 'lower_wick', 'is_bullish',
    'hour', 'in_session', 'day_of_week',
    'dist_to_high20', 'dist_to_low20',
    'fvg_bull', 'fvg_bear',
    'broke_high20', 'broke_low20',
    'momentum_5', 'momentum_20', 'momentum_50',
    'vol_regime', 'h4_bullish',
    'signal'
]


# ─────────────────────────────────────────────
#  5. MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(df_labeled: pd.DataFrame,
                test_size: float = 0.25) -> tuple:
    """
    Train XGBoost classifier on labeled trade data.

    Returns:
        (model, scaler, feature_names, metrics)
    """
    if not ML_AVAILABLE:
        raise ImportError("xgboost and scikit-learn required.")

    # ── Prepare features ──
    available = [c for c in FEATURE_COLS if c in df_labeled.columns]
    X = df_labeled[available].values
    y = df_labeled['label'].values

    print(f"\nTraining on {len(X):,} samples with {len(available)} features...")

    # ── Train/test split (time-based, not random) ──
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ── Scale features ──
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── XGBoost model ──
    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 10,    # prevents overfitting on small samples
        scale_pos_weight  = (y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder = False,
        eval_metric       = 'auc',
        random_state      = 42,
        verbosity         = 0,
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set          = [(X_test, y_test)],
        verbose           = False,
    )

    # ── Evaluate ──
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    accuracy    = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)

    metrics = {
        'accuracy'  : round(accuracy * 100, 2),
        'auc'       : round(auc, 4),
        'train_size': len(X_train),
        'test_size' : len(X_test),
        'features'  : len(available),
        'n_wins_train'  : int(y_train.sum()),
        'n_losses_train': int((1-y_train).sum()),
    }

    print(f"\n{'='*50}")
    print(f"  MODEL TRAINING RESULTS")
    print(f"{'='*50}")
    print(f"  Train samples : {len(X_train):,}")
    print(f"  Test samples  : {len(X_test):,}")
    print(f"  Accuracy      : {accuracy*100:.2f}%")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['LOSE', 'WIN']))

    # ── Feature importance ──
    importances = dict(zip(available,
                           model.feature_importances_))
    top_features = sorted(importances.items(),
                          key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top 10 Features:")
    for feat, imp in top_features:
        bar = "█" * int(imp * 50)
        print(f"    {feat:25s} {bar} {imp:.4f}")

    return model, scaler, available, metrics


# ─────────────────────────────────────────────
#  6. SAVE / LOAD MODEL
# ─────────────────────────────────────────────

def save_model(model, scaler, feature_names: list):
    """Save trained model to disk."""
    joblib.dump(model,         MODEL_FILE)
    joblib.dump(scaler,        SCALER_FILE)
    joblib.dump(feature_names, FEATURES_FILE)
    print(f"\n✅ Model saved:")
    print(f"   {MODEL_FILE}")
    print(f"   {SCALER_FILE}")
    print(f"   {FEATURES_FILE}")


def load_model():
    """Load trained model from disk."""
    if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, FEATURES_FILE]):
        raise FileNotFoundError(
            "Model files not found. Run: python ml_engine.py train"
        )
    model         = joblib.load(MODEL_FILE)
    scaler        = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURES_FILE)
    return model, scaler, feature_names


# ─────────────────────────────────────────────
#  7. LIVE PREDICTION
# ─────────────────────────────────────────────

def predict_trade(df_h1: pd.DataFrame,
                  df_h4: pd.DataFrame,
                  direction: str) -> MLSignal:
    """
    Score the current market conditions.
    Returns MLSignal with win probability.

    Parameters:
        df_h1     : Recent H1 data (at least 100 candles)
        df_h4     : Recent H4 data (at least 60 candles)
        direction : 'BUY' or 'SELL'

    Returns:
        MLSignal with win probability and recommendation
    """
    if not ML_AVAILABLE:
        return MLSignal(
            allows_trade     = True,
            win_probability  = 0.5,
            confidence       = "UNAVAILABLE",
            feature_importances = {},
            details          = "ML libraries not installed — allowing trade"
        )

    try:
        model, scaler, feature_names = load_model()
    except FileNotFoundError:
        return MLSignal(
            allows_trade     = True,
            win_probability  = 0.5,
            confidence       = "UNTRAINED",
            feature_importances = {},
            details          = "Model not trained yet — run: python ml_engine.py train"
        )

    # Engineer features on recent data
    features_df = engineer_features(df_h1.copy(), df_h4.copy())

    if len(features_df) == 0:
        return MLSignal(
            allows_trade     = True,
            win_probability  = 0.5,
            confidence       = "ERROR",
            feature_importances = {},
            details          = "Not enough data for feature engineering"
        )

    # Get latest row
    latest = features_df.iloc[-1]

    # Force signal direction
    dir_val = 1 if direction == "BUY" else -1
    row_data = latest.copy()
    row_data['signal'] = dir_val

    # Extract features in correct order
    X = np.array([[row_data.get(f, 0) for f in feature_names]])
    X = scaler.transform(X)

    # Predict
    prob      = model.predict_proba(X)[0][1]   # P(WIN)
    allows    = prob >= ML_THRESHOLD

    # Confidence level
    if prob >= 0.75:
        confidence = "HIGH"
    elif prob >= ML_THRESHOLD:
        confidence = "MEDIUM"
    elif prob >= 0.50:
        confidence = "LOW"
    else:
        confidence = "REJECT"

    # Feature importances for this prediction
    importances = dict(zip(feature_names, model.feature_importances_))
    top5 = dict(sorted(importances.items(),
                        key=lambda x: x[1], reverse=True)[:5])

    details = (
        f"ML P(WIN)={prob:.3f}  threshold={ML_THRESHOLD}  "
        f"→ {'✅ ALLOWED' if allows else '❌ REJECTED'}"
    )

    return MLSignal(
        allows_trade        = allows,
        win_probability     = round(prob, 4),
        confidence          = confidence,
        feature_importances = top5,
        details             = details
    )


# ─────────────────────────────────────────────
#  8. ML-FILTERED BACKTEST
# ─────────────────────────────────────────────

def ml_backtest(df_labeled: pd.DataFrame,
                model, scaler, feature_names: list,
                threshold: float = ML_THRESHOLD) -> dict:
    """
    Compare baseline strategy vs ML-filtered strategy.
    Shows how many bad trades ML correctly filters out.
    """
    available = [c for c in feature_names if c in df_labeled.columns]
    X = df_labeled[available].values
    y = df_labeled['label'].values
    X_scaled = scaler.transform(X)

    probs     = model.predict_proba(X_scaled)[:, 1]
    ml_filter = probs >= threshold

    baseline_wr  = y.mean() * 100
    ml_wr        = y[ml_filter].mean() * 100 if ml_filter.sum() > 0 else 0
    trades_taken = ml_filter.sum()
    trades_skipped = (~ml_filter).sum()

    # Simulate P&L (simplified: +3 for win, -2 for loss in ATR units)
    base_pnl = sum(3 if yi == 1 else -2 for yi in y)
    ml_pnl   = sum(3 if yi == 1 else -2
                   for yi, f in zip(y, ml_filter) if f)

    results = {
        'baseline_trades'  : len(y),
        'baseline_win_rate': round(baseline_wr, 2),
        'baseline_pnl_pts' : base_pnl,
        'ml_trades'        : int(trades_taken),
        'ml_skipped'       : int(trades_skipped),
        'ml_win_rate'      : round(ml_wr, 2),
        'ml_pnl_pts'       : ml_pnl,
        'win_rate_gain'    : round(ml_wr - baseline_wr, 2),
        'pnl_gain_pts'     : ml_pnl - base_pnl,
    }

    print(f"\n{'='*55}")
    print(f"  ML BACKTEST COMPARISON")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'Baseline':>10} {'ML Filter':>10}")
    print(f"  {'─'*45}")
    print(f"  {'Trades taken':<25} {results['baseline_trades']:>10} {results['ml_trades']:>10}")
    print(f"  {'Win Rate':<25} {results['baseline_win_rate']:>9.1f}% {results['ml_win_rate']:>9.1f}%")
    print(f"  {'P&L (ATR pts)':<25} {results['baseline_pnl_pts']:>10} {results['ml_pnl_pts']:>10}")
    print(f"  {'Trades filtered':<25} {'—':>10} {results['ml_skipped']:>10}")
    print(f"  {'Win Rate gain':<25} {'—':>10} {results['win_rate_gain']:>+9.1f}%")
    print(f"{'='*55}")

    return results


# ─────────────────────────────────────────────
#  9. VISUALIZATION
# ─────────────────────────────────────────────

def plot_results(df_labeled: pd.DataFrame,
                 model, scaler, feature_names: list,
                 metrics: dict):
    """Generate ML results visualization."""
    available = [c for c in feature_names if c in df_labeled.columns]
    X = df_labeled[available].values
    y = df_labeled['label'].values
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0A1628')
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)
    gold  = '#F0A500'
    green = '#1D9E75'
    red   = '#D85A30'

    def ax_s(ax, t):
        ax.set_facecolor('#0F2040')
        ax.set_title(t, color='white', fontsize=11)
        ax.tick_params(colors='#8BA0BE', labelsize=9)
        for s in ax.spines.values():
            s.set_edgecolor('#162B50')

    # 1. Win probability distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(probs[y==1], bins=30, color=green, alpha=0.7,
             label='Actual WIN', edgecolor='none')
    ax1.hist(probs[y==0], bins=30, color=red, alpha=0.7,
             label='Actual LOSE', edgecolor='none')
    ax1.axvline(ML_THRESHOLD, color=gold, lw=2, ls='--',
                label=f'Threshold {ML_THRESHOLD}')
    ax_s(ax1, 'Win Probability Distribution')
    ax1.legend(fontsize=8, labelcolor='white', facecolor='#0F2040')
    ax1.set_xlabel('P(WIN)', color='#8BA0BE', fontsize=9)

    # 2. Feature importance
    ax2 = fig.add_subplot(gs[0, 1])
    importances = model.feature_importances_
    indices = np.argsort(importances)[-12:]
    ax2.barh(range(len(indices)),
             importances[indices], color=gold, alpha=0.8)
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels([available[i] for i in indices],
                         fontsize=8, color='#8BA0BE')
    ax_s(ax2, 'Top Feature Importances')
    ax2.set_xlabel('Importance', color='#8BA0BE', fontsize=9)

    # 3. Win rate at different thresholds
    ax3 = fig.add_subplot(gs[0, 2])
    thresholds = np.arange(0.45, 0.85, 0.05)
    win_rates  = []
    trade_pcts = []
    for t in thresholds:
        mask = probs >= t
        wr   = y[mask].mean() * 100 if mask.sum() > 0 else 0
        pct  = mask.mean() * 100
        win_rates.append(wr)
        trade_pcts.append(pct)
    ax3_twin = ax3.twinx()
    ax3.plot(thresholds, win_rates, color=green, lw=2,
             marker='o', markersize=4, label='Win Rate')
    ax3_twin.plot(thresholds, trade_pcts, color=gold, lw=2,
                  ls='--', marker='s', markersize=4, label='% Trades')
    ax3.axvline(ML_THRESHOLD, color='white', lw=1, ls=':', alpha=0.5)
    ax_s(ax3, 'Threshold vs Win Rate & Trade Volume')
    ax3.set_xlabel('ML Threshold', color='#8BA0BE', fontsize=9)
    ax3.set_ylabel('Win Rate %', color=green, fontsize=9)
    ax3_twin.set_ylabel('% Trades Taken', color=gold, fontsize=9)
    ax3.tick_params(axis='y', colors=green)
    ax3_twin.tick_params(axis='y', colors=gold)

    # 4. Confusion matrix
    ax4 = fig.add_subplot(gs[1, 0])
    preds = (probs >= ML_THRESHOLD).astype(int)
    cm    = confusion_matrix(y, preds)
    im    = ax4.imshow(cm, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Pred LOSE', 'Pred WIN'], color='#8BA0BE', fontsize=9)
    ax4.set_yticklabels(['Actual LOSE', 'Actual WIN'], color='#8BA0BE', fontsize=9)
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white', fontsize=14, fontweight='bold')
    ax_s(ax4, 'Confusion Matrix')

    # 5. Cumulative P&L comparison
    ax5 = fig.add_subplot(gs[1, 1])
    base_cum = np.cumsum([3 if yi == 1 else -2 for yi in y])
    ml_mask  = probs >= ML_THRESHOLD
    ml_cum   = []
    running  = 0
    for yi, mi in zip(y, ml_mask):
        if mi:
            running += 3 if yi == 1 else -2
        ml_cum.append(running)
    ax5.plot(base_cum, color=red,   lw=1.5, alpha=0.8, label='Baseline')
    ax5.plot(ml_cum,   color=green, lw=2,   label='ML Filtered')
    ax5.axhline(0, color='#8BA0BE', lw=0.5, ls='--')
    ax_s(ax5, 'Cumulative P&L (ATR pts)')
    ax5.legend(fontsize=8, labelcolor='white', facecolor='#0F2040')
    ax5.set_xlabel('Trade #', color='#8BA0BE', fontsize=9)

    # 6. Model metrics summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax_s(ax6, 'Model Summary')
    summary = [
        f"Accuracy    : {metrics['accuracy']:.1f}%",
        f"ROC-AUC     : {metrics['auc']:.4f}",
        f"Train size  : {metrics['train_size']:,}",
        f"Test size   : {metrics['test_size']:,}",
        f"Features    : {metrics['features']}",
        f"Threshold   : {ML_THRESHOLD}",
        "",
        f"Baseline WR : see chart",
        f"ML WR       : see chart",
    ]
    for i, line in enumerate(summary):
        ax6.text(0.05, 0.9 - i*0.1, line, transform=ax6.transAxes,
                 color='#8BA0BE' if line else 'white',
                 fontsize=11, fontfamily='monospace')

    fig.suptitle('XAUUSD Phase 6 — ML Engine Results',
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('ml_results.png', dpi=150,
                bbox_inches='tight', facecolor='#0A1628')
    print("✅ Chart saved: ml_results.png")
    plt.close()


# ─────────────────────────────────────────────
#  10. INTEGRATION HELPER FOR xauusd_bot.py
# ─────────────────────────────────────────────

def ml_filter_passes(df_h1: pd.DataFrame,
                     df_h4: pd.DataFrame,
                     direction: str) -> tuple[bool, str]:
    """
    Final ML gate before placing a trade.
    Returns (passes, reason_string).
    """
    signal = predict_trade(df_h1, df_h4, direction)

    if signal.confidence in ("UNAVAILABLE", "UNTRAINED", "ERROR"):
        return True, f"⚠️  ML {signal.confidence} — allowing trade"

    if signal.allows_trade:
        return True, (f"✅ ML approved P(WIN)={signal.win_probability:.1%} "
                      f"[{signal.confidence}]")
    else:
        return False, (f"❌ ML rejected P(WIN)={signal.win_probability:.1%} "
                       f"< threshold {ML_THRESHOLD}")


def print_ml_report(signal: MLSignal):
    """Print formatted ML signal report."""
    conf_icons = {"HIGH": "🔥", "MEDIUM": "✅", "LOW": "⚠️",
                  "REJECT": "❌", "UNAVAILABLE": "⚠️", "UNTRAINED": "⚠️"}
    icon = conf_icons.get(signal.confidence, "⚠️")
    bar  = int(signal.win_probability * 20)
    prob_bar = "█" * bar + "░" * (20 - bar)

    print("╔══════════════════════════════════════════════════╗")
    print("║         PHASE 6 — ML ENGINE VERDICT             ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  P(WIN)     : {prob_bar}  {signal.win_probability:.1%}")
    print(f"║  Confidence : {icon} {signal.confidence}")
    print(f"║  Decision   : {'✅ TRADE ALLOWED' if signal.allows_trade else '❌ TRADE REJECTED'}")
    if signal.feature_importances:
        print("╠══════════════════════════════════════════════════╣")
        print("║  Top drivers:")
        for feat, imp in list(signal.feature_importances.items())[:3]:
            print(f"║    {feat:<20} {imp:.4f}")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  {signal.details}\n")


# ─────────────────────────────────────────────
#  MAIN — train / test / live modes
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    print("\n" + "="*60)
    print(f"  XAUUSD PHASE 6 — ML ENGINE  [{mode.upper()}]")
    print("="*60 + "\n")

    if not ML_AVAILABLE:
        print("❌ Required libraries not installed. Run:")
        print("   pip install xgboost scikit-learn joblib")
        sys.exit(1)

    if mode == "train":
        print("Step 1: Fetching data...")
        df_h1, df_h4 = fetch_training_data(months=24)

        print("\nStep 2: Engineering features...")
        df_feat = engineer_features(df_h1, df_h4)
        print(f"✅ {len(df_feat):,} rows | {len(FEATURE_COLS)} features")

        print("\nStep 3: Generating trade labels...")
        df_labeled = generate_labels(df_feat)

        print("\nStep 4: Training XGBoost model...")
        model, scaler, feat_names, metrics = train_model(df_labeled)

        print("\nStep 5: Running ML backtest comparison...")
        ml_backtest(df_labeled, model, scaler, feat_names)

        print("\nStep 6: Saving model...")
        save_model(model, scaler, feat_names)

        print("\nStep 7: Generating charts...")
        plot_results(df_labeled, model, scaler, feat_names, metrics)

        print("\n" + "="*60)
        print("  ✅ Training complete!")
        print("  Now run: python ml_engine.py live")
        print("="*60)

    elif mode == "live":
        print("Fetching recent market data...")
        df_h1, df_h4 = fetch_training_data(months=2)

        for direction in ["BUY", "SELL"]:
            print(f"\n── Testing {direction} signal ──")
            signal = predict_trade(df_h1, df_h4, direction)
            print_ml_report(signal)

    elif mode == "test":
        print("Loading saved model and running test...")
        model, scaler, feat_names = load_model()
        df_h1, df_h4 = fetch_training_data(months=6)
        df_feat    = engineer_features(df_h1, df_h4)
        df_labeled = generate_labels(df_feat)
        ml_backtest(df_labeled, model, scaler, feat_names)

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python ml_engine.py [train|live|test]")
