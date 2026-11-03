"""
FuturesLab — backtest.py
========================
Event-driven backtesting engine.

Execution Timing (CRITICAL — no look-ahead bias)
-------------------------------------------------
All signals are shifted by 1 trading day before execution:

    Data available at T (Close price, features computed)
              ↓
    Signal generated at T (using ONLY data up to T)
              ↓
    Order executed at T+1 open (modelled as T+1 Close for daily data)
              ↓
    Return captured from T+1 to T+2 (next bar return)

This means:

    position[t] = signal[t-1]  (the signal generated yesterday)

In code:
    signal_executed = raw_signal.shift(1)

This is the most important safeguard against look-ahead bias in a
daily backtest. Without this shift, the strategy sees tomorrow's
closing price to make today's decision — an impossible feat in practice.

Why T+1 Close (not Open)?
    Daily data from Yahoo Finance gives OHLCV per day. Without intraday
    data, we cannot model the exact T+1 open fill. Using T+1 Close is a
    conservative approximation: it assumes we can trade at the closing
    price of the execution day. This slightly underestimates performance
    for trend-following (which benefits from gap-up opens) and is a
    documented research assumption.

Position Sizing
    Position size from risk.py is applied to the shifted signal:
    effective_position = signal_executed × position_size

Strategy Comparison
-------------------
Four strategies are evaluated for each market:
  'buy_hold'    — always long, full position
  'trend'       — trend following, volatility sized
  'mean_rev'    — mean reversion, volatility sized
  'adaptive'    — regime-adaptive, volatility sized

Return Calculation
    daily_strategy_return = position[t] × market_return[t+1] - cost[t]
    where cost[t] is the transaction cost when position changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.config import (
    EMA_FAST, EMA_SLOW, RSI_PERIOD, ATR_PERIOD,
    BREAKOUT_PERIOD, ZSCORE_WINDOW,
    ZSCORE_ENTRY_LONG, ZSCORE_ENTRY_SHORT, ZSCORE_EXIT,
    ATR_STOP_MULTIPLIER, TOTAL_COST_PER_TRADE_BPS,
    TARGET_ANNUALIZED_VOL, MAX_POSITION_SIZE, HVOL_POSITION_SCALE,
    TRADING_DAYS_PER_YEAR,
)
from src.features import add_all_features
from src.regimes import classify_regimes
from src.strategies import (
    trend_signal,
    mean_reversion_signal,
    regime_adaptive_signal,
    buy_and_hold_signal,
)
from src.risk import compute_position_size, apply_atr_stop, transaction_cost


STRATEGY_NAMES = ["buy_hold", "trend", "mean_rev", "adaptive"]


def prepare_features(
    df: pd.DataFrame,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
    atr_period: int = ATR_PERIOD,
    breakout_period: int = BREAKOUT_PERIOD,
    zscore_window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Add features and regime classification to df."""
    df = add_all_features(
        df,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        atr_period=atr_period,
        breakout_period=breakout_period,
        zscore_window=zscore_window,
    )
    df = classify_regimes(
        df,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
    )
    return df


def run_backtest(
    df: pd.DataFrame,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
    atr_period: int = ATR_PERIOD,
    breakout_period: int = BREAKOUT_PERIOD,
    zscore_window: int = ZSCORE_WINDOW,
    entry_long: float = ZSCORE_ENTRY_LONG,
    entry_short: float = ZSCORE_ENTRY_SHORT,
    exit_threshold: float = ZSCORE_EXIT,
    atr_stop_mult: float = ATR_STOP_MULTIPLIER,
    cost_bps: float = TOTAL_COST_PER_TRADE_BPS,
    target_vol: float = TARGET_ANNUALIZED_VOL,
    max_pos: float = MAX_POSITION_SIZE,
    hvol_scale: float = HVOL_POSITION_SCALE,
    apply_stops: bool = True,
) -> pd.DataFrame:
    """
    Run all four strategies on a single market's data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV data (from data.load_processed()).
    (remaining) : various
        Configuration overrides for robustness testing.

    Returns
    -------
    pd.DataFrame
        Multi-column DataFrame with daily returns for each strategy.
        Columns:
          'market_return'    — raw daily log return
          'pos_buy_hold'     — buy-and-hold position
          'pos_trend'        — trend-following position
          'pos_mean_rev'     — mean-reversion position
          'pos_adaptive'     — adaptive position
          'ret_buy_hold'     — buy-and-hold daily return
          'ret_trend'        — trend daily return
          'ret_mean_rev'     — mean-reversion daily return
          'ret_adaptive'     — adaptive daily return
          'Regime'           — daily regime
    """
    df = prepare_features(
        df,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        atr_period=atr_period,
        breakout_period=breakout_period,
        zscore_window=zscore_window,
    )

    # ── 1. Generate raw signals (at signal day T) ──
    bh_sig  = buy_and_hold_signal(df)
    tf_sig  = trend_signal(df, ema_fast, ema_slow, rsi_period, breakout_period)
    mr_sig  = mean_reversion_signal(df, zscore_window, entry_long, entry_short, exit_threshold)
    ada_sig = regime_adaptive_signal(
        df, ema_fast, ema_slow, rsi_period, breakout_period,
        zscore_window, entry_long, entry_short, exit_threshold,
    )

    # ── 2. Apply ATR stops ──
    if apply_stops:
        tf_sig  = apply_atr_stop(tf_sig,  df, atr_stop_mult)
        mr_sig  = apply_atr_stop(mr_sig,  df, atr_stop_mult)
        ada_sig = apply_atr_stop(ada_sig, df, atr_stop_mult)

    # ── 3. Shift signals by 1 day (T+1 execution — ANTI-LOOKAHEAD) ──
    bh_sig_exec  = bh_sig.shift(1).fillna(0)
    tf_sig_exec  = tf_sig.shift(1).fillna(0)
    mr_sig_exec  = mr_sig.shift(1).fillna(0)
    ada_sig_exec = ada_sig.shift(1).fillna(0)

    # ── 4. Compute position sizes (based on lagged volatility) ──
    pos_size = compute_position_size(df, target_vol=target_vol, max_size=max_pos, hvol_scale=hvol_scale)
    pos_size_exec = pos_size.shift(1).fillna(1.0)  # also lagged

    # ── 5. Effective positions ──
    pos_bh  = bh_sig_exec                          # buy-hold always 1.0
    pos_tf  = tf_sig_exec  * pos_size_exec
    pos_mr  = mr_sig_exec  * pos_size_exec
    pos_ada = ada_sig_exec * pos_size_exec

    # ── 6. Transaction costs ──
    cost_bh  = transaction_cost(pos_bh,  cost_bps)
    cost_tf  = transaction_cost(pos_tf,  cost_bps)
    cost_mr  = transaction_cost(pos_mr,  cost_bps)
    cost_ada = transaction_cost(pos_ada, cost_bps)

    # ── 7. Daily strategy returns ──
    mkt_ret = df["Return"]

    ret_bh  = pos_bh  * mkt_ret - cost_bh
    ret_tf  = pos_tf  * mkt_ret - cost_tf
    ret_mr  = pos_mr  * mkt_ret - cost_mr
    ret_ada = pos_ada * mkt_ret - cost_ada

    # ── 8. Assemble result DataFrame ──
    result = pd.DataFrame({
        "market_return": mkt_ret,
        "pos_buy_hold":  pos_bh,
        "pos_trend":     pos_tf,
        "pos_mean_rev":  pos_mr,
        "pos_adaptive":  pos_ada,
        "ret_buy_hold":  ret_bh,
        "ret_trend":     ret_tf,
        "ret_mean_rev":  ret_mr,
        "ret_adaptive":  ret_ada,
        "Regime":        df["Regime"],
        "Close":         df["Close"],
        "RVOL20":        df.get(f"RVOL{20}", pd.Series(dtype=float)),
        "ATR14":         df.get(f"ATR{14}", pd.Series(dtype=float)),
    }, index=df.index)

    return result


def run_all_markets(
    data: Dict[str, pd.DataFrame],
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Run backtest for all markets.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Market key → cleaned OHLCV DataFrame.
    **kwargs : passed to run_backtest()

    Returns
    -------
    dict[str, pd.DataFrame]
        Market key → backtest result DataFrame.
    """
    results = {}
    for mkt, df in data.items():
        print(f"Running backtest for {mkt}...")
        results[mkt] = run_backtest(df.copy(), **kwargs)
    return results


def split_period(
    df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Slice a DataFrame to a specific date range (inclusive)."""
    return df.loc[start:end]
