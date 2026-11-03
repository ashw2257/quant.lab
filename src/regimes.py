"""
FuturesLab — regimes.py
=======================
Market regime classification module.

Three regimes
-------------
1. TRENDING      — market exhibits persistent directional price movement
2. SIDEWAYS      — market is range-bound / mean-reverting
3. HIGH_VOL      — extreme volatility dominates; reduce or avoid exposure

Classification logic
--------------------
The classifier is entirely rule-based — no machine learning, no hidden
Markov models. Any interviewer can understand it on a whiteboard in 60
seconds. The rules use only features calculated from past data.

Step 1: Volatility check (highest priority)
    If RVOL20 > HVOL_THRESHOLD (e.g., 75th percentile of rolling vol):
        → HIGH_VOL regime
    Rationale: Extreme volatility overrides everything else.
               Strategy loses edge during chaotic tape.

Step 2: Trend check
    Compute "trend score" from three components:
    (a) EMA spread signal: EMA20 vs EMA50 normalised distance
    (b) Price confirmation: price vs EMA20 direction
    (c) Momentum confirmation: RSI above/below 50

    If |ema_spread| > TREND_EMA_SPREAD_THRESHOLD
       AND price on correct side of EMA20
       AND RSI confirms direction:
        → TRENDING regime

Step 3: Default
    If neither HIGH_VOL nor TRENDING:
        → SIDEWAYS regime

Threshold decisions (documented in Research_Decision_Log.md)
--------------------------------------------------------------
TREND_EMA_SPREAD_THRESHOLD = 0.005 (0.5%)
  Chosen by examining the distribution of |EMA_spread| on the
  development set (2012-2018). The 40th percentile of |EMA_spread|
  falls near 0.4-0.6% for all three markets. This ensures we classify
  ~40% of days as trending when the EMA spread is meaningful, and ~40%
  as sideways, with the remainder in HIGH_VOL.

HVOL_RVOL_PERCENTILE = 75
  The top quartile of realised volatility identifies genuinely elevated
  risk environments. Tested at 65th and 85th percentiles in robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    TREND_EMA_SPREAD_THRESHOLD,
    HVOL_RVOL_PERCENTILE,
    RSI_BULL_THRESHOLD,
    RSI_BEAR_THRESHOLD,
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    RVOL_PERIOD,
)

# Regime labels
REGIME_TRENDING = "TRENDING"
REGIME_SIDEWAYS = "SIDEWAYS"
REGIME_HIGH_VOL = "HIGH_VOL"

ALL_REGIMES = [REGIME_TRENDING, REGIME_SIDEWAYS, REGIME_HIGH_VOL]


def _compute_hvol_threshold(rvol: pd.Series, percentile: float = HVOL_RVOL_PERCENTILE) -> pd.Series:
    """
    Rolling historical percentile threshold for HIGH_VOL classification.

    Uses an expanding window so the threshold at time T is computed from
    all data up to and including T — no look-ahead bias.

    Parameters
    ----------
    rvol : pd.Series
        Annualised rolling realised volatility.
    percentile : float
        Percentile (0-100) used to define 'high volatility'.

    Returns
    -------
    pd.Series
        Threshold value at each point in time.
    """
    return rvol.expanding(min_periods=252).quantile(percentile / 100)


def classify_regimes(
    df: pd.DataFrame,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
    rvol_period: int = RVOL_PERIOD,
    trend_ema_threshold: float = TREND_EMA_SPREAD_THRESHOLD,
    hvol_percentile: float = HVOL_RVOL_PERCENTILE,
) -> pd.DataFrame:
    """
    Classify each trading day into TRENDING, SIDEWAYS, or HIGH_VOL.

    Input DataFrame must already contain feature columns (from features.py).
    Required columns:
      EMA{ema_fast}, EMA{ema_slow}, EMA_spread,
      RSI{rsi_period}, RVOL{rvol_period}, Close

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with technical features already added.
    (remaining) : various
        Override defaults from config if needed (for robustness testing).

    Returns
    -------
    pd.DataFrame
        df with added columns:
          'HVOL_Threshold' — rolling vol percentile threshold
          'Regime'         — one of TRENDING / SIDEWAYS / HIGH_VOL
          'Regime_Detail'  — human-readable explanation for each day
    """
    df = df.copy()

    ema_f_col   = f"EMA{ema_fast}"
    ema_s_col   = f"EMA{ema_slow}"
    rsi_col     = f"RSI{rsi_period}"
    rvol_col    = f"RVOL{rvol_period}"
    spread_col  = "EMA_spread"

    # Validate required columns
    for col in [ema_f_col, ema_s_col, rsi_col, rvol_col, spread_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Run features.add_all_features() first.")

    # ── Step 1: HIGH_VOL threshold (no look-ahead) ──
    hvol_thresh = _compute_hvol_threshold(df[rvol_col], hvol_percentile)
    df["HVOL_Threshold"] = hvol_thresh

    # ── Classify row-by-row via vectorised conditions ──
    rvol  = df[rvol_col]
    spread = df[spread_col].abs()
    ema_f  = df[ema_f_col]
    ema_s  = df[ema_s_col]
    close  = df["Close"]
    rsi    = df[rsi_col]
    raw_spread = df[spread_col]  # signed

    # HIGH_VOL: rvol exceeds expanding percentile threshold
    is_high_vol = rvol > hvol_thresh

    # TRENDING criteria:
    # (a) |EMA spread| > threshold → meaningful divergence
    spread_ok = spread > trend_ema_threshold
    # (b) Price confirms trend direction
    #     Bullish: close > EMA_fast AND EMA_fast > EMA_slow
    #     Bearish: close < EMA_fast AND EMA_fast < EMA_slow
    bullish = (close > ema_f) & (raw_spread > 0)
    bearish = (close < ema_f) & (raw_spread < 0)
    price_confirms = bullish | bearish
    # (c) Momentum confirmation via RSI
    rsi_bull = rsi > RSI_BULL_THRESHOLD
    rsi_bear = rsi < RSI_BEAR_THRESHOLD
    momentum_confirms = (bullish & rsi_bull) | (bearish & rsi_bear)

    is_trending = (~is_high_vol) & spread_ok & price_confirms & momentum_confirms

    # SIDEWAYS: everything else
    # is_sideways = ~is_high_vol & ~is_trending

    # Assign regime
    regime = pd.Series(REGIME_SIDEWAYS, index=df.index, dtype=str)
    regime[is_trending] = REGIME_TRENDING
    regime[is_high_vol] = REGIME_HIGH_VOL  # HIGH_VOL overrides TRENDING

    df["Regime"] = regime

    # ── Human-readable explanation column ──
    def _detail(row):
        r = row["Regime"]
        if r == REGIME_HIGH_VOL:
            return (
                f"HIGH_VOL: RVOL={row.get(rvol_col, float('nan')):.1%} "
                f"> threshold {row['HVOL_Threshold']:.1%}"
            )
        elif r == REGIME_TRENDING:
            direction = "BULLISH" if row[spread_col] > 0 else "BEARISH"
            return (
                f"TRENDING ({direction}): |EMA_spread|={abs(row[spread_col]):.3%} "
                f"> {trend_ema_threshold:.3%}, RSI={row.get(rsi_col, float('nan')):.1f}"
            )
        else:
            return (
                f"SIDEWAYS: |EMA_spread|={abs(row.get(spread_col, 0)):.3%} "
                f"≤ {trend_ema_threshold:.3%} or momentum not confirmed"
            )

    df["Regime_Detail"] = df.apply(_detail, axis=1)

    return df


def regime_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics by regime.

    Returns a DataFrame with rows = regimes, columns:
      Count, Pct_Days, Mean_Return, Std_Return, Annualized_Vol, Sharpe
    """
    results = []
    total = len(df.dropna(subset=["Regime", "Return"]))

    for regime in ALL_REGIMES:
        mask = df["Regime"] == regime
        sub = df[mask]["Return"].dropna()
        if len(sub) == 0:
            results.append({
                "Regime": regime, "Count": 0, "Pct_Days": 0,
                "Mean_Daily_Return": np.nan, "Std_Daily_Return": np.nan,
                "Ann_Return": np.nan, "Ann_Vol": np.nan, "Sharpe": np.nan,
            })
            continue

        n = len(sub)
        mu = sub.mean()
        sd = sub.std()
        ann_ret = mu * 252
        ann_vol = sd * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

        results.append({
            "Regime": regime,
            "Count": n,
            "Pct_Days": n / total,
            "Mean_Daily_Return": mu,
            "Std_Daily_Return": sd,
            "Ann_Return": ann_ret,
            "Ann_Vol": ann_vol,
            "Sharpe": sharpe,
        })

    return pd.DataFrame(results).set_index("Regime")


def regime_transitions(df: pd.DataFrame) -> int:
    """Count number of regime transitions (consecutive different regimes)."""
    reg = df["Regime"].dropna()
    return (reg != reg.shift(1)).sum() - 1  # subtract initial


def regime_summary(df: pd.DataFrame) -> dict:
    """High-level regime summary for dashboard display."""
    stats = regime_statistics(df)
    n_trans = regime_transitions(df)
    current = df["Regime"].dropna().iloc[-1] if len(df["Regime"].dropna()) > 0 else "N/A"
    return {
        "statistics": stats,
        "transitions": n_trans,
        "current_regime": current,
    }
