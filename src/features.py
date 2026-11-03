"""
FuturesLab — features.py
========================
Technical indicator computation.

All indicators are calculated using only lookback windows of past data.
No look-ahead bias is introduced here — that is enforced in backtest.py
by shifting signals by 1 day before execution.

Indicators implemented
----------------------
Trend:
  ema()              → Exponential Moving Average
  ema_cross_signal() → EMA20/EMA50 crossover signal

Momentum:
  rsi()              → Relative Strength Index (14-day)

Volatility:
  atr()              → Average True Range
  realized_vol()     → Rolling realized volatility (annualised)

Breakout:
  rolling_high()     → N-day rolling maximum of Close
  rolling_low()      → N-day rolling minimum of Close
  breakout_signal()  → Price vs rolling high/low

Standardised deviation:
  rolling_zscore()   → Z-score of Close within rolling window

Why these indicators?
---------------------
EMA20 / EMA50:
  Exponential moving averages weight recent prices more heavily than SMA.
  The 20/50 pair is a widely used benchmark for short-to-medium trend detection.
  Crossover of EMA20 above EMA50 suggests momentum has shifted upward.

RSI-14:
  Developed by Welles Wilder. Measures speed and magnitude of price changes.
  RSI > 50 generally indicates upward momentum (more up-days than down-days).
  RSI < 50 indicates downward momentum. Not an overbought/oversold signal here —
  we use it as a pure momentum confirmation.

ATR-14:
  True Range captures gap risk (overnight moves). ATR14 is a 14-day average.
  Normalised by price, ATR gives a per-unit-price measure of volatility.
  Used for stop-loss placement and position sizing.

Realized Volatility (20-day):
  Standard deviation of daily log returns × √252. The most direct empirical
  measure of recent price fluctuation. Used for regime classification and
  position sizing.

Rolling High/Low:
  20-day rolling maximum/minimum identifies price breakouts — periods when
  the market moves outside its recent range. A key component of trend detection.

Z-Score:
  Standardises Close relative to its rolling mean and standard deviation.
  z > 0 means price is above its recent average. Used in mean-reversion strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    ATR_PERIOD,
    RVOL_PERIOD,
    BREAKOUT_PERIOD,
    ZSCORE_WINDOW,
    TRADING_DAYS_PER_YEAR,
)


# ─────────────────────────────────────────────
# Trend indicators
# ─────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average.

    Formula: EMA_t = α × Price_t + (1 - α) × EMA_{t-1}
    where α = 2 / (span + 1)

    Parameters
    ----------
    series : pd.Series
        Price series (typically Close).
    span : int
        EMA period (number of days).

    Returns
    -------
    pd.Series
        EMA values. First (span-1) values will be NaN.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def ema_spread(df: pd.DataFrame, fast: int = EMA_FAST, slow: int = EMA_SLOW) -> pd.Series:
    """
    Normalised EMA spread: (EMA_fast - EMA_slow) / EMA_slow.

    Positive = fast EMA above slow EMA = bullish tendency.
    Negative = fast EMA below slow EMA = bearish tendency.
    """
    ema_f = ema(df["Close"], fast)
    ema_s = ema(df["Close"], slow)
    return (ema_f - ema_s) / ema_s


def add_emas(df: pd.DataFrame, fast: int = EMA_FAST, slow: int = EMA_SLOW) -> pd.DataFrame:
    """Add EMA_fast, EMA_slow, and EMA_spread columns to df."""
    df = df.copy()
    df[f"EMA{fast}"] = ema(df["Close"], fast)
    df[f"EMA{slow}"] = ema(df["Close"], slow)
    df["EMA_spread"] = (df[f"EMA{fast}"] - df[f"EMA{slow}"]) / df[f"EMA{slow}"]
    return df


# ─────────────────────────────────────────────
# Momentum indicator
# ─────────────────────────────────────────────

def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Relative Strength Index (Wilder smoothing).

    RSI = 100 - 100 / (1 + RS)
    RS  = Average Gain / Average Loss over the period

    Wilder's method uses exponential smoothing with alpha = 1/period.

    Parameters
    ----------
    series : pd.Series
        Close price series.
    period : int
        Lookback period (default 14).

    Returns
    -------
    pd.Series
        RSI values in [0, 100]. First `period` values will be NaN.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder smoothing = EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # When avg_loss = 0: RS → ∞ → RSI = 100 (pure uptrend)
    # When avg_gain = 0: RS = 0 → RSI = 0 (pure downtrend)
    # Avoid dividing by zero while preserving NaN for warm-up period
    rsi_val = pd.Series(np.nan, index=series.index)
    both_valid = avg_gain.notna() & avg_loss.notna()
    both_zero  = both_valid & (avg_gain == 0) & (avg_loss == 0)
    loss_zero  = both_valid & (avg_loss == 0) & (avg_gain != 0)
    gain_zero  = both_valid & (avg_gain == 0) & (avg_loss != 0)
    normal     = both_valid & (avg_loss != 0) & (avg_gain != 0)

    rsi_val[loss_zero]  = 100.0
    rsi_val[gain_zero]  = 0.0
    rsi_val[both_zero]  = 50.0
    rs = avg_gain[normal] / avg_loss[normal]
    rsi_val[normal] = 100 - (100 / (1 + rs))
    return rsi_val


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Add RSI column to df."""
    df = df.copy()
    df[f"RSI{period}"] = rsi(df["Close"], period)
    return df


# ─────────────────────────────────────────────
# Volatility indicators
# ─────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    Average True Range (Wilder smoothing).

    True Range = max(
        High - Low,
        |High - prior Close|,
        |Low  - prior Close|
    )

    ATR = Wilder EMA of True Range over `period` days.

    ATR captures gap risk that a simple High-Low range misses.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain High, Low, Close columns.
    period : int
        Lookback period (default 14).

    Returns
    -------
    pd.Series
        ATR values. First `period` values will be NaN.
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close  = (df["Low"]  - df["Close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def realized_vol(
    series: pd.Series,
    window: int = RVOL_PERIOD,
    annualize: bool = True,
) -> pd.Series:
    """
    Rolling realized volatility.

    Calculated as rolling standard deviation of daily log returns,
    optionally annualised by multiplying by √252.

    Parameters
    ----------
    series : pd.Series
        Log return series.
    window : int
        Rolling window in trading days.
    annualize : bool
        If True, multiply by √252 to produce annualised volatility.

    Returns
    -------
    pd.Series
        Rolling realized volatility.
    """
    rvol = series.rolling(window).std()
    if annualize:
        rvol = rvol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return rvol


def add_volatility(
    df: pd.DataFrame,
    atr_period: int = ATR_PERIOD,
    rvol_period: int = RVOL_PERIOD,
) -> pd.DataFrame:
    """Add ATR, ATR_pct (ATR/Close), and RVOL columns to df."""
    df = df.copy()
    df[f"ATR{atr_period}"] = atr(df, atr_period)
    df[f"ATR{atr_period}_pct"] = df[f"ATR{atr_period}"] / df["Close"]
    df[f"RVOL{rvol_period}"] = realized_vol(df["Return"], rvol_period)
    return df


# ─────────────────────────────────────────────
# Breakout indicators
# ─────────────────────────────────────────────

def rolling_high(series: pd.Series, window: int = BREAKOUT_PERIOD) -> pd.Series:
    """N-day rolling maximum. Uses a 1-day lag to avoid look-ahead bias."""
    return series.shift(1).rolling(window).max()


def rolling_low(series: pd.Series, window: int = BREAKOUT_PERIOD) -> pd.Series:
    """N-day rolling minimum. Uses a 1-day lag to avoid look-ahead bias."""
    return series.shift(1).rolling(window).min()


def add_breakout(df: pd.DataFrame, window: int = BREAKOUT_PERIOD) -> pd.DataFrame:
    """
    Add rolling high/low breakout columns.

    Rolling_High and Rolling_Low are lagged by 1 day so that the breakout
    level used for a signal on day T was observable at end of day T-1.
    'Above_High' = True if today's Close exceeds yesterday's N-day high.
    'Below_Low'  = True if today's Close falls below yesterday's N-day low.
    """
    df = df.copy()
    df[f"Roll_High{window}"] = rolling_high(df["Close"], window)
    df[f"Roll_Low{window}"]  = rolling_low(df["Close"], window)
    df[f"Above_High{window}"] = df["Close"] > df[f"Roll_High{window}"]
    df[f"Below_Low{window}"]  = df["Close"] < df[f"Roll_Low{window}"]
    return df


# ─────────────────────────────────────────────
# Z-score / mean reversion
# ─────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """
    Rolling z-score of a price series.

    z_t = (Close_t - μ_{t,window}) / σ_{t,window}

    Positive z → price above rolling mean (potentially overbought in MR context).
    Negative z → price below rolling mean (potentially oversold in MR context).

    Note: z-score does NOT look ahead — it uses only the past `window` prices.
    """
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)


def add_zscore(df: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    """Add rolling z-score column to df."""
    df = df.copy()
    df[f"ZScore{window}"] = rolling_zscore(df["Close"], window)
    return df


# ─────────────────────────────────────────────
# Full feature pipeline
# ─────────────────────────────────────────────

def add_all_features(
    df: pd.DataFrame,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
    atr_period: int = ATR_PERIOD,
    rvol_period: int = RVOL_PERIOD,
    breakout_period: int = BREAKOUT_PERIOD,
    zscore_window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    Add all technical features to a cleaned OHLCV DataFrame.

    Call once after loading processed data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV with 'Return' column.
    (remaining) : int
        Period parameters for each indicator.

    Returns
    -------
    pd.DataFrame
        df with all feature columns added.
    """
    df = add_emas(df, ema_fast, ema_slow)
    df = add_rsi(df, rsi_period)
    df = add_volatility(df, atr_period, rvol_period)
    df = add_breakout(df, breakout_period)
    df = add_zscore(df, zscore_window)
    return df
