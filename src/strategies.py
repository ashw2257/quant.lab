"""
FuturesLab — strategies.py
==========================
Trading signal generation for three strategies.

Strategy 1: Trend Following (TF)
---------------------------------
Hypothesis: Markets exhibiting persistent directional movement may
continue trending, making momentum a useful signal.

LONG conditions (all must be met):
  1. EMA20 > EMA50 (uptrend structure)
  2. Close > rolling 20-day high from prior day (breakout confirmation)
  3. RSI > 50 (momentum confirmation)

SHORT conditions (all must be met):
  1. EMA20 < EMA50 (downtrend structure)
  2. Close < rolling 20-day low from prior day (breakdown confirmation)
  3. RSI < 50 (momentum confirmation)

FLAT: None of the above, or conflicting signals.

Note: This strategy generates signals on day T.
      The signal is shifted by 1 day in backtest.py (T+1 execution).

Strategy 2: Mean Reversion (MR)
---------------------------------
Hypothesis: In non-trending conditions, short-term price deviations from
the rolling mean tend to partially reverse.

LONG: z-score < -ZSCORE_ENTRY_LONG  (price unusually far below mean)
SHORT: z-score > ZSCORE_ENTRY_SHORT  (price unusually far above mean)
EXIT: |z-score| < ZSCORE_EXIT        (price has reverted toward mean)

This strategy should ONLY be applied in SIDEWAYS regime.
Applied indiscriminately, it loses money in trending markets.

Regime-Adaptive Strategy
-------------------------
TRENDING   → apply Trend Following strategy
SIDEWAYS   → apply Mean Reversion strategy
HIGH_VOL   → flat (no position)

All three strategies return a signal series with values:
  +1 = long
  -1 = short
   0 = flat
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    BREAKOUT_PERIOD,
    ZSCORE_WINDOW,
    ZSCORE_ENTRY_LONG,
    ZSCORE_ENTRY_SHORT,
    ZSCORE_EXIT,
)
from src.regimes import REGIME_TRENDING, REGIME_SIDEWAYS, REGIME_HIGH_VOL


# ─────────────────────────────────────────────
# Trend Following Signal
# ─────────────────────────────────────────────

def trend_signal(
    df: pd.DataFrame,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
    breakout_period: int = BREAKOUT_PERIOD,
) -> pd.Series:
    """
    Generate daily trend-following signal (+1, 0, -1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all features (from features.add_all_features()).
    ema_fast, ema_slow, rsi_period, breakout_period : int
        Period parameters (allows override for robustness testing).

    Returns
    -------
    pd.Series
        Signal values: +1 (long), -1 (short), 0 (flat).
        NOT yet shifted for T+1 execution. Shifting happens in backtest.py.
    """
    ema_f  = df[f"EMA{ema_fast}"]
    ema_s  = df[f"EMA{ema_slow}"]
    rsi    = df[f"RSI{rsi_period}"]
    hi     = df[f"Roll_High{breakout_period}"]  # already lagged 1 day in features.py
    lo     = df[f"Roll_Low{breakout_period}"]   # already lagged 1 day in features.py
    close  = df["Close"]

    long_cond  = (ema_f > ema_s) & (close > hi) & (rsi > 50)
    short_cond = (ema_f < ema_s) & (close < lo) & (rsi < 50)

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[long_cond]  =  1.0
    signal[short_cond] = -1.0

    return signal


# ─────────────────────────────────────────────
# Mean Reversion Signal
# ─────────────────────────────────────────────

def mean_reversion_signal(
    df: pd.DataFrame,
    zscore_window: int = ZSCORE_WINDOW,
    entry_long: float = ZSCORE_ENTRY_LONG,
    entry_short: float = ZSCORE_ENTRY_SHORT,
    exit_threshold: float = ZSCORE_EXIT,
) -> pd.Series:
    """
    Generate daily mean-reversion signal (+1, 0, -1).

    Signal logic:
      - Enter LONG  when z-score < -entry_long  (oversold deviation)
      - Enter SHORT when z-score > +entry_short (overbought deviation)
      - Exit         when |z-score| < exit_threshold (reversion to mean)

    The signal is stateful: once entered, it holds until z reverts.
    This is implemented via forward-fill within state transitions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ZScore column.
    zscore_window : int
        Z-score window (used to name column).
    entry_long : float
        Magnitude of z-score to trigger long entry (e.g., 1.5 → z < -1.5).
    entry_short : float
        Magnitude of z-score to trigger short entry (e.g., 1.5 → z > +1.5).
    exit_threshold : float
        |z| below this → exit position.

    Returns
    -------
    pd.Series
        Signal: +1, -1, or 0.
    """
    z = df[f"ZScore{zscore_window}"]
    signal = pd.Series(0, index=df.index, dtype=float)

    pos = 0.0
    signals = []
    for z_val in z:
        if np.isnan(z_val):
            signals.append(0.0)
            continue

        if pos == 0:
            if z_val < -entry_long:
                pos = 1.0
            elif z_val > entry_short:
                pos = -1.0
        else:
            if abs(z_val) < exit_threshold:
                pos = 0.0

        signals.append(pos)

    return pd.Series(signals, index=df.index, dtype=float)


# ─────────────────────────────────────────────
# Regime-Adaptive Strategy
# ─────────────────────────────────────────────

def regime_adaptive_signal(
    df: pd.DataFrame,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
    breakout_period: int = BREAKOUT_PERIOD,
    zscore_window: int = ZSCORE_WINDOW,
    entry_long: float = ZSCORE_ENTRY_LONG,
    entry_short: float = ZSCORE_ENTRY_SHORT,
    exit_threshold: float = ZSCORE_EXIT,
) -> pd.Series:
    """
    Regime-adaptive signal that switches between TF and MR based on regime.

    TRENDING   → trend_signal()
    SIDEWAYS   → mean_reversion_signal()
    HIGH_VOL   → 0 (flat)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and 'Regime' column.

    Returns
    -------
    pd.Series
        Signal: +1, -1, or 0.
    """
    if "Regime" not in df.columns:
        raise ValueError("'Regime' column not found. Run classify_regimes() first.")

    tf_sig = trend_signal(df, ema_fast, ema_slow, rsi_period, breakout_period)
    mr_sig = mean_reversion_signal(df, zscore_window, entry_long, entry_short, exit_threshold)

    regime = df["Regime"]
    signal = pd.Series(0.0, index=df.index)
    signal[regime == REGIME_TRENDING] = tf_sig[regime == REGIME_TRENDING]
    signal[regime == REGIME_SIDEWAYS] = mr_sig[regime == REGIME_SIDEWAYS]
    signal[regime == REGIME_HIGH_VOL] = 0.0

    return signal


# ─────────────────────────────────────────────
# Buy-and-Hold Benchmark
# ─────────────────────────────────────────────

def buy_and_hold_signal(df: pd.DataFrame) -> pd.Series:
    """Always long. Benchmark to compare against."""
    return pd.Series(1.0, index=df.index)


# ─────────────────────────────────────────────
# Signal explanation (for dashboard)
# ─────────────────────────────────────────────

def explain_signal(df: pd.DataFrame, date: pd.Timestamp = None) -> dict:
    """
    Generate a human-readable explanation of the current signal.
    Used by the Streamlit dashboard.

    Returns dict with keys: regime, strategy, signal, reasons
    """
    if date is None:
        row = df.dropna(subset=["Regime"]).iloc[-1]
    else:
        row = df.loc[date]

    regime    = row.get("Regime", "N/A")
    ema_fast  = EMA_FAST
    ema_slow  = EMA_SLOW

    ema_f_val = row.get(f"EMA{ema_fast}", float("nan"))
    ema_s_val = row.get(f"EMA{ema_slow}", float("nan"))
    rsi_val   = row.get(f"RSI{RSI_PERIOD}", float("nan"))
    close     = row.get("Close", float("nan"))
    roll_hi   = row.get(f"Roll_High{BREAKOUT_PERIOD}", float("nan"))
    roll_lo   = row.get(f"Roll_Low{BREAKOUT_PERIOD}", float("nan"))
    zscore    = row.get(f"ZScore{ZSCORE_WINDOW}", float("nan"))
    rvol      = row.get(f"RVOL20", float("nan"))

    reasons = []
    signal_val = 0

    if regime == REGIME_HIGH_VOL:
        strategy = "FLAT (High Volatility)"
        reasons.append(f"✗ High volatility regime: RVOL={rvol:.1%}")
        signal_val = 0

    elif regime == REGIME_TRENDING:
        strategy = "TREND FOLLOWING"
        tf_ok = (ema_f_val > ema_s_val) if not np.isnan(ema_f_val) else False
        bo_ok = (close > roll_hi)       if not np.isnan(roll_hi) else False
        rsi_ok = (rsi_val > 50)          if not np.isnan(rsi_val) else False
        tf_dn = (ema_f_val < ema_s_val) if not np.isnan(ema_f_val) else False
        bo_dn = (close < roll_lo)        if not np.isnan(roll_lo) else False
        ri_dn = (rsi_val < 50)           if not np.isnan(rsi_val) else False

        if tf_ok and bo_ok and rsi_ok:
            signal_val = 1
            reasons.append(f"✓ EMA{ema_fast} ({ema_f_val:.1f}) > EMA{ema_slow} ({ema_s_val:.1f})")
            reasons.append(f"✓ Close ({close:.1f}) > 20d High ({roll_hi:.1f})")
            reasons.append(f"✓ RSI ({rsi_val:.1f}) > 50 — bullish momentum")
        elif tf_dn and bo_dn and ri_dn:
            signal_val = -1
            reasons.append(f"✓ EMA{ema_fast} ({ema_f_val:.1f}) < EMA{ema_slow} ({ema_s_val:.1f})")
            reasons.append(f"✓ Close ({close:.1f}) < 20d Low ({roll_lo:.1f})")
            reasons.append(f"✓ RSI ({rsi_val:.1f}) < 50 — bearish momentum")
        else:
            signal_val = 0
            reasons.append("✗ Trend conditions not fully confirmed")

    elif regime == REGIME_SIDEWAYS:
        strategy = "MEAN REVERSION"
        if not np.isnan(zscore):
            if zscore < -ZSCORE_ENTRY_LONG:
                signal_val = 1
                reasons.append(f"✓ Z-score ({zscore:.2f}) < -{ZSCORE_ENTRY_LONG} — oversold")
            elif zscore > ZSCORE_ENTRY_SHORT:
                signal_val = -1
                reasons.append(f"✓ Z-score ({zscore:.2f}) > +{ZSCORE_ENTRY_SHORT} — overbought")
            else:
                signal_val = 0
                reasons.append(f"✗ Z-score ({zscore:.2f}) — no extreme deviation")
    else:
        strategy = "N/A"

    signal_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}[signal_val]

    return {
        "regime": regime,
        "strategy": strategy,
        "signal": signal_str,
        "signal_val": signal_val,
        "reasons": reasons,
        "ema_fast": ema_f_val,
        "ema_slow": ema_s_val,
        "rsi": rsi_val,
        "atr": row.get(f"ATR{14}", float("nan")),
        "zscore": zscore,
        "rvol": rvol,
        "roll_high": roll_hi,
        "roll_low": roll_lo,
    }
