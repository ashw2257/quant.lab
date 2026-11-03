"""
FuturesLab — risk.py
=====================
Position sizing and risk management.

Volatility-Based Position Sizing
----------------------------------
Principle: Target a constant level of portfolio risk regardless of market
           volatility. When vol is high, take smaller positions. When vol
           is low, take larger positions.

Formula:
    raw_size = TARGET_ANNUALIZED_VOL / current_rvol
    position_size = clip(raw_size, 0, MAX_POSITION_SIZE)

Where:
    TARGET_ANNUALIZED_VOL = 15% (configurable in config.py)
    current_rvol          = 20-day annualised realized volatility
    MAX_POSITION_SIZE      = 1.5 (caps leverage at 1.5x)

Why this works:
    A market with 30% annual vol gets a 15/30 = 0.5 allocation.
    A market with 10% annual vol gets a 15/10 = 1.5 allocation (capped).
    The resulting strategy vol tracks the 15% target more closely.

High-Volatility Scaling:
    During HIGH_VOL regime, the position size is further multiplied
    by HVOL_POSITION_SCALE (default 0.25), drastically reducing exposure.

ATR-Based Stop Loss:
    stop_distance = ATR14 × ATR_STOP_MULTIPLIER
    For longs:  stop = entry_price - stop_distance
    For shorts: stop = entry_price + stop_distance

    On each day, if price has moved against the position by more than
    stop_distance from the entry price, the position is closed.

Transaction Costs:
    Applied at every trade (entry and exit):
    cost = TOTAL_COST_PER_TRADE_BPS / 10000
    The position return is reduced by this cost per trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    TARGET_ANNUALIZED_VOL,
    MAX_POSITION_SIZE,
    ATR_STOP_MULTIPLIER,
    HVOL_POSITION_SCALE,
    TOTAL_COST_PER_TRADE_BPS,
    RVOL_PERIOD,
    ATR_PERIOD,
)
from src.regimes import REGIME_HIGH_VOL


def compute_position_size(
    df: pd.DataFrame,
    rvol_col: str = None,
    target_vol: float = TARGET_ANNUALIZED_VOL,
    max_size: float = MAX_POSITION_SIZE,
    hvol_scale: float = HVOL_POSITION_SCALE,
) -> pd.Series:
    """
    Volatility-targeted position sizing.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain RVOL column and optionally 'Regime'.
    rvol_col : str
        Name of realized vol column. Defaults to f"RVOL{RVOL_PERIOD}".
    target_vol : float
        Target annualised portfolio volatility (e.g., 0.15 = 15%).
    max_size : float
        Maximum position multiplier (caps leverage).
    hvol_scale : float
        Scaling factor during HIGH_VOL regime.

    Returns
    -------
    pd.Series
        Position size multiplier for each day.
        Range: [0, max_size] normally, [0, hvol_scale × max_size] in HIGH_VOL.
    """
    if rvol_col is None:
        rvol_col = f"RVOL{RVOL_PERIOD}"

    rvol = df[rvol_col].copy()

    # Replace zeros/NaN to avoid division issues
    rvol = rvol.replace(0, np.nan).fillna(target_vol)

    raw_size = target_vol / rvol
    pos_size = raw_size.clip(upper=max_size)

    # Apply HIGH_VOL scaling
    if "Regime" in df.columns:
        is_hvol = df["Regime"] == REGIME_HIGH_VOL
        pos_size[is_hvol] = pos_size[is_hvol] * hvol_scale

    return pos_size


def apply_atr_stop(
    signal: pd.Series,
    df: pd.DataFrame,
    atr_multiplier: float = ATR_STOP_MULTIPLIER,
    atr_col: str = None,
) -> pd.Series:
    """
    Apply ATR-based stop-loss to a signal series.

    Iterates day-by-day maintaining entry price. If price moves adversely
    by more than (atr_multiplier × ATR) from entry, closes position.

    Parameters
    ----------
    signal : pd.Series
        Raw signal (+1, 0, -1).
    df : pd.DataFrame
        Must contain Close and ATR column.
    atr_multiplier : float
        Number of ATRs for stop distance.
    atr_col : str
        ATR column name. Defaults to f"ATR{ATR_PERIOD}".

    Returns
    -------
    pd.Series
        Signal with stops applied.
    """
    if atr_col is None:
        atr_col = f"ATR{ATR_PERIOD}"

    close  = df["Close"].values
    atr    = df[atr_col].values
    raw_sig = signal.values.copy()
    result  = np.zeros(len(raw_sig))

    entry_price   = np.nan
    current_pos   = 0.0
    stop_price    = np.nan

    for i in range(len(raw_sig)):
        new_sig = raw_sig[i]
        c       = close[i]
        a       = atr[i] if not np.isnan(atr[i]) else 0.0

        # Check if stop triggered
        if current_pos == 1.0 and not np.isnan(stop_price):
            if c <= stop_price:
                current_pos = 0.0
                entry_price = np.nan
                stop_price  = np.nan

        elif current_pos == -1.0 and not np.isnan(stop_price):
            if c >= stop_price:
                current_pos = 0.0
                entry_price = np.nan
                stop_price  = np.nan

        # Enter or change position
        if new_sig != current_pos:
            if new_sig == 0.0:
                current_pos = 0.0
                entry_price = np.nan
                stop_price  = np.nan
            else:
                current_pos = new_sig
                entry_price = c
                if new_sig == 1.0:
                    stop_price = c - atr_multiplier * a
                else:
                    stop_price = c + atr_multiplier * a

        result[i] = current_pos

    return pd.Series(result, index=signal.index)


def transaction_cost(
    position: pd.Series,
    cost_bps: float = TOTAL_COST_PER_TRADE_BPS,
) -> pd.Series:
    """
    Compute daily transaction cost from position changes.

    A trade occurs whenever position changes.
    Cost is applied as a fraction of AUM on each trade day.

    Parameters
    ----------
    position : pd.Series
        Position series (includes fractional sizes).
    cost_bps : float
        One-way cost in basis points.

    Returns
    -------
    pd.Series
        Daily cost drag (non-negative), deducted from return.
    """
    cost_pct = cost_bps / 10_000
    pos_change = position.diff().abs()
    cost = pos_change * cost_pct
    cost.iloc[0] = abs(position.iloc[0]) * cost_pct  # initial entry
    return cost
