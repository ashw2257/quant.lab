"""
FuturesLab — evaluation.py
===========================
Performance metrics computation and tabular reporting.

All metrics are computed from daily log-return series.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR, DATA_PROC_DIR


STRATEGY_DISPLAY = {
    "buy_hold": "Buy & Hold",
    "trend":    "Static Trend",
    "mean_rev": "Static Mean-Rev",
    "adaptive": "Regime-Adaptive",
}


# ─────────────────────────────────────────────
# Individual metric functions
# ─────────────────────────────────────────────

def cagr(returns: pd.Series, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Compound Annual Growth Rate from daily log returns.

    CAGR = exp(mean_log_return × 252) - 1
    """
    if returns.dropna().empty:
        return np.nan
    return float(np.exp(returns.dropna().mean() * ann_factor) - 1)


def annualized_vol(returns: pd.Series, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised standard deviation of daily returns."""
    if returns.dropna().empty:
        return np.nan
    return float(returns.dropna().std() * np.sqrt(ann_factor))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualised Sharpe Ratio.

    Sharpe = (Ann_Return - Risk_Free) / Ann_Vol

    We use risk_free = 0 for simplicity (can be set to prevailing T-bill rate).
    """
    c = cagr(returns, ann_factor)
    v = annualized_vol(returns, ann_factor)
    if v == 0 or np.isnan(v):
        return np.nan
    return float((c - risk_free) / v)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualised Sortino Ratio.

    Uses downside deviation (standard deviation of negative returns only).
    Better than Sharpe when return distribution is asymmetric.
    """
    c = cagr(returns, ann_factor)
    downside = returns[returns < 0].dropna()
    if len(downside) == 0:
        return np.nan
    downside_std = downside.std() * np.sqrt(ann_factor)
    if downside_std == 0:
        return np.nan
    return float((c - risk_free) / downside_std)


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown from peak to trough.

    MDD = max(peak - trough) / peak  (in percentage terms, negative)
    """
    cum = np.exp(returns.dropna().cumsum())
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calmar = CAGR / |Max Drawdown|."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    c = cagr(returns, ann_factor)
    return float(c / abs(mdd))


def win_rate(position: pd.Series, returns: pd.Series) -> float:
    """
    Fraction of active trading days where the strategy made money.

    Only considers days where |position| > 0.
    """
    active = position.abs() > 0
    if active.sum() == 0:
        return np.nan
    strat_ret = position * returns
    wins = (strat_ret[active] > 0).sum()
    return float(wins / active.sum())


def trade_count(position: pd.Series) -> int:
    """Number of position changes (entries + exits + reversals)."""
    return int((position.diff().abs() > 0.05).sum())


def avg_trade_return(position: pd.Series, returns: pd.Series) -> float:
    """Average return per trade (entry to exit)."""
    active = position.abs() > 0
    strat_ret = (position * returns)[active]
    if len(strat_ret) == 0:
        return np.nan
    return float(strat_ret.mean())


def exposure(position: pd.Series) -> float:
    """Percentage of time with non-zero position."""
    return float((position.abs() > 0.05).sum() / len(position))


def turnover(position: pd.Series, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised two-way turnover (sum of absolute position changes / periods × 252)."""
    return float(position.diff().abs().sum() / len(position) * ann_factor)


def monthly_returns(returns: pd.Series) -> pd.Series:
    """Resample daily log returns to monthly, return as simple returns."""
    monthly = returns.resample("ME").sum()
    return np.expm1(monthly)


def best_worst_months(returns: pd.Series) -> tuple[float, float]:
    """Return (best month, worst month) as simple returns."""
    mr = monthly_returns(returns)
    if mr.empty:
        return (np.nan, np.nan)
    return float(mr.max()), float(mr.min())


# ─────────────────────────────────────────────
# Full metrics table
# ─────────────────────────────────────────────

def compute_metrics(
    backtest_df: pd.DataFrame,
    strategies: List[str] = None,
) -> pd.DataFrame:
    """
    Compute all performance metrics for each strategy.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output from backtest.run_backtest(). Must have ret_* and pos_* columns.
    strategies : list[str]
        Subset of strategies to evaluate. Defaults to all four.

    Returns
    -------
    pd.DataFrame
        Rows = strategies, Columns = metrics.
    """
    if strategies is None:
        strategies = ["buy_hold", "trend", "mean_rev", "adaptive"]

    rows = []
    for strat in strategies:
        ret_col = f"ret_{strat}"
        pos_col = f"pos_{strat}"

        if ret_col not in backtest_df.columns:
            continue

        ret = backtest_df[ret_col].dropna()
        pos = backtest_df[pos_col]

        bst, wst = best_worst_months(ret)

        rows.append({
            "Strategy":        STRATEGY_DISPLAY.get(strat, strat),
            "CAGR":            cagr(ret),
            "Ann_Vol":         annualized_vol(ret),
            "Sharpe":          sharpe_ratio(ret),
            "Sortino":         sortino_ratio(ret),
            "Max_Drawdown":    max_drawdown(ret),
            "Calmar":          calmar_ratio(ret),
            "Win_Rate":        win_rate(pos, backtest_df["market_return"]),
            "Trades":          trade_count(pos),
            "Avg_Trade_Ret":   avg_trade_return(pos, backtest_df["market_return"]),
            "Exposure":        exposure(pos),
            "Turnover_Ann":    turnover(pos),
            "Best_Month":      bst,
            "Worst_Month":     wst,
        })

    return pd.DataFrame(rows).set_index("Strategy")


def compute_all_markets_metrics(
    results: Dict[str, pd.DataFrame],
    period_start: str = None,
    period_end: str = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute metrics for all markets, optionally over a sub-period.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Market key → backtest result DataFrame.
    period_start, period_end : str
        ISO date strings for slicing.

    Returns
    -------
    dict[str, pd.DataFrame]
        Market key → metrics DataFrame.
    """
    all_metrics = {}
    for mkt, df in results.items():
        sub = df.loc[period_start:period_end] if period_start or period_end else df
        all_metrics[mkt] = compute_metrics(sub)
    return all_metrics


def print_metrics_table(metrics: pd.DataFrame, market: str = "") -> None:
    """Print a formatted metrics table to console."""
    print(f"\n{'='*70}")
    print(f"  Performance Metrics -- {market}")
    print(f"{'='*70}")
    fmt = metrics.copy()
    for col in ["CAGR", "Ann_Vol", "Sharpe", "Sortino", "Max_Drawdown",
                "Calmar", "Win_Rate", "Avg_Trade_Ret", "Exposure",
                "Best_Month", "Worst_Month"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    print(fmt.to_string())
    print()


def save_results(
    all_metrics: Dict[str, pd.DataFrame],
    backtest_results: Dict[str, pd.DataFrame],
    output_dir: Path = DATA_PROC_DIR,
    label: str = "full",
) -> None:
    """Save metrics and equity curves to processed directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mkt, mdf in all_metrics.items():
        fname = output_dir / f"{mkt}_metrics_{label}.csv"
        mdf.to_csv(fname)

    for mkt, bdf in backtest_results.items():
        fname = output_dir / f"{mkt}_backtest_{label}.parquet"
        bdf.to_parquet(fname)

    print(f"Results saved to {output_dir}")


def load_backtest_result(market_key: str, label: str = "full") -> pd.DataFrame:
    """Load a saved backtest result parquet."""
    path = DATA_PROC_DIR / f"{market_key}_backtest_{label}.parquet"
    return pd.read_parquet(path)
