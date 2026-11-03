"""
FuturesLab — run_research.py
=============================
Master pipeline script.

Usage:
    python run_research.py

This script:
1. Downloads market data from Yahoo Finance
2. Cleans and validates data
3. Computes technical features
4. Classifies market regimes
5. Runs all four strategies for all three markets
6. Evaluates performance metrics
7. Runs robustness tests
8. Runs macro event study
9. Saves all results to data/processed/
10. Generates charts to reports/figures/
11. Prints final results table
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import (
    MARKETS, DATA_PROC_DIR, FIGURES_DIR,
    DEV_START, DEV_END, VAL_START, VAL_END, OOS_START, OOS_END,
    ROBUST_EMA_PAIRS, ROBUST_BREAKOUT_PERIODS,
    ROBUST_ZSCORE_THRESHOLDS, ROBUST_COST_BPS,
    EMA_FAST, EMA_SLOW, BREAKOUT_PERIOD, ZSCORE_WINDOW,
    TRADING_DAYS_PER_YEAR,
)
from src.data import run_pipeline, load_processed, data_quality_report
from src.features import add_all_features
from src.regimes import classify_regimes, regime_statistics
from src.backtest import run_all_markets, split_period
from src.evaluation import (
    compute_metrics, compute_all_markets_metrics, print_metrics_table, save_results,
    cagr, sharpe_ratio, max_drawdown, monthly_returns,
)
from src.macro import run_event_study


FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Data Pipeline
# ─────────────────────────────────────────────
def step1_data():
    logger.info("STEP 1: Data acquisition and cleaning")
    data = run_pipeline(force=False)
    for mkt, df in data.items():
        report = data_quality_report(df, mkt)
        logger.info("[%s] %d rows, %s → %s", mkt, report["rows"], report["start"], report["end"])
    return data


# ─────────────────────────────────────────────
# 2. Regime Statistics
# ─────────────────────────────────────────────
def step2_regime_stats(data):
    logger.info("STEP 2: Regime classification statistics")
    regime_stats_all = {}
    for mkt, df in data.items():
        feat_df = add_all_features(df)
        reg_df  = classify_regimes(feat_df)
        stats   = regime_statistics(reg_df)
        regime_stats_all[mkt] = stats
        logger.info("[%s] Regime distribution:\n%s", mkt, stats[["Count", "Pct_Days"]].to_string())
    return regime_stats_all


# ─────────────────────────────────────────────
# 3. Full Backtest
# ─────────────────────────────────────────────
def step3_backtest(data):
    logger.info("STEP 3: Running backtests (all markets, all strategies)")
    results = run_all_markets(data)
    return results


# ─────────────────────────────────────────────
# 4. Metrics — Full Period
# ─────────────────────────────────────────────
def step4_metrics(results):
    logger.info("STEP 4: Computing performance metrics")
    all_metrics = {}
    for mkt, df in results.items():
        metrics = compute_metrics(df)
        all_metrics[mkt] = metrics
        print_metrics_table(metrics, mkt)
    return all_metrics


# ─────────────────────────────────────────────
# 5. Period Breakdown (Dev / Val / OOS)
# ─────────────────────────────────────────────
def step5_period_breakdown(results):
    logger.info("STEP 5: Period breakdown (Dev / Validation / OOS)")
    period_metrics = {}
    for period_name, start, end in [
        ("Development", DEV_START, DEV_END),
        ("Validation",  VAL_START, VAL_END),
        ("OOS",         OOS_START, OOS_END),
    ]:
        pm = compute_all_markets_metrics(results, start, end)
        period_metrics[period_name] = pm
        logger.info("\n── %s (%s → %s) ──", period_name, start, end)
        for mkt, mdf in pm.items():
            print_metrics_table(mdf, f"{mkt} [{period_name}]")
    return period_metrics


# ─────────────────────────────────────────────
# 6. Robustness Tests
# ─────────────────────────────────────────────
def step6_robustness(data):
    logger.info("STEP 6: Robustness testing (parameter sensitivity)")
    robust_records = []

    # EMA pair sensitivity (adaptive strategy, development period)
    for fast, slow in ROBUST_EMA_PAIRS:
        for mkt, df in data.items():
            try:
                from src.backtest import run_backtest
                bt = run_backtest(df.copy(), ema_fast=fast, ema_slow=slow)
                bt_dev = split_period(bt, DEV_START, DEV_END)
                ret = bt_dev["ret_adaptive"].dropna()
                robust_records.append({
                    "test": f"EMA{fast}/{slow}",
                    "market": mkt,
                    "period": "Dev",
                    "CAGR": cagr(ret),
                    "Sharpe": sharpe_ratio(ret),
                    "MaxDD": max_drawdown(ret),
                })
            except Exception as e:
                logger.warning("Robustness EMA %d/%d %s failed: %s", fast, slow, mkt, e)

    # Breakout period sensitivity
    for bp in ROBUST_BREAKOUT_PERIODS:
        for mkt, df in data.items():
            try:
                from src.backtest import run_backtest
                bt = run_backtest(df.copy(), breakout_period=bp)
                bt_dev = split_period(bt, DEV_START, DEV_END)
                ret = bt_dev["ret_adaptive"].dropna()
                robust_records.append({
                    "test": f"Breakout{bp}d",
                    "market": mkt,
                    "period": "Dev",
                    "CAGR": cagr(ret),
                    "Sharpe": sharpe_ratio(ret),
                    "MaxDD": max_drawdown(ret),
                })
            except Exception as e:
                logger.warning("Robustness Breakout %d %s failed: %s", bp, mkt, e)

    # Transaction cost sensitivity
    for cost in ROBUST_COST_BPS:
        for mkt, df in data.items():
            try:
                from src.backtest import run_backtest
                bt = run_backtest(df.copy(), cost_bps=cost)
                bt_dev = split_period(bt, DEV_START, DEV_END)
                ret = bt_dev["ret_adaptive"].dropna()
                robust_records.append({
                    "test": f"Cost{cost:.0f}bps",
                    "market": mkt,
                    "period": "Dev",
                    "CAGR": cagr(ret),
                    "Sharpe": sharpe_ratio(ret),
                    "MaxDD": max_drawdown(ret),
                })
            except Exception as e:
                logger.warning("Robustness Cost %s %s failed: %s", cost, mkt, e)

    robust_df = pd.DataFrame(robust_records)
    if not robust_df.empty:
        robust_df.to_csv(DATA_PROC_DIR / "robustness_results.csv", index=False)
        logger.info("Robustness results saved.")
        print("\n=== ROBUSTNESS TEST SUMMARY ===")
        print(robust_df.to_string(index=False))
    return robust_df


# ─────────────────────────────────────────────
# 7. Macro Event Study
# ─────────────────────────────────────────────
def step7_macro(data):
    logger.info("STEP 7: Macro event study")
    market_returns = {mkt: df["Return"] for mkt, df in data.items()}
    event_results = run_event_study(market_returns, window=1)

    for mkt, ev_dict in event_results.items():
        for ev_name, ev_data in ev_dict.items():
            summary = ev_data["summary"]
            logger.info("[%s] %s event study:\n%s", mkt, ev_name, summary.to_string())
            fname = DATA_PROC_DIR / f"{mkt}_{ev_name}_event_study.csv"
            summary.to_csv(fname)

    return event_results


# ─────────────────────────────────────────────
# 8. Generate Charts
# ─────────────────────────────────────────────
def step8_charts(data, results):
    logger.info("STEP 8: Generating charts")
    _plot_equity_curves(results)
    _plot_regime_timeline(data, results)
    _plot_drawdown(results)
    _plot_monthly_heatmap(results)
    logger.info("Charts saved to %s", FIGURES_DIR)


def _plot_equity_curves(results):
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=False)
    colors = {"buy_hold": "#888888", "trend": "#2196F3", "mean_rev": "#FF9800", "adaptive": "#4CAF50"}
    labels = {"buy_hold": "Buy & Hold", "trend": "Trend", "mean_rev": "Mean-Rev", "adaptive": "Adaptive"}

    for ax, (mkt, df) in zip(axes, results.items()):
        for strat in ["buy_hold", "trend", "mean_rev", "adaptive"]:
            col = f"ret_{strat}"
            if col not in df.columns:
                continue
            equity = np.exp(df[col].dropna().cumsum())
            ax.plot(equity.index, equity.values, label=labels[strat],
                    color=colors[strat], linewidth=1.5 if strat == "adaptive" else 1.0)
        ax.set_title(f"{MARKETS[mkt]['name']} — Equity Curves", fontsize=11, fontweight="bold")
        ax.set_ylabel("Growth of $1")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.suptitle("FuturesLab — Strategy Equity Curves", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved equity_curves.png")


def _plot_regime_timeline(data, results):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    regime_colors = {"TRENDING": "#4CAF50", "SIDEWAYS": "#2196F3", "HIGH_VOL": "#F44336"}

    for ax, (mkt, df) in zip(axes, results.items()):
        price = data[mkt]["Close"]
        ax2 = ax.twinx()
        ax2.plot(price.index, price.values, color="#333333", linewidth=0.7, alpha=0.6, label="Close")
        ax2.set_ylabel("Price", fontsize=8)

        # Color background by regime
        regime = df["Regime"].dropna()
        prev_regime = None
        prev_date = None
        for date, reg in regime.items():
            if reg != prev_regime:
                if prev_regime is not None:
                    ax.axvspan(prev_date, date, alpha=0.25,
                               color=regime_colors.get(prev_regime, "#999999"), label=None)
                prev_regime = reg
                prev_date = date
        if prev_regime is not None:
            ax.axvspan(prev_date, regime.index[-1], alpha=0.25,
                       color=regime_colors.get(prev_regime, "#999999"))

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, alpha=0.4, label=r) for r, c in regime_colors.items()]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
        ax.set_title(f"{MARKETS[mkt]['name']} — Regime Timeline", fontsize=11, fontweight="bold")
        ax.set_yticks([])
        ax.grid(False)

    plt.suptitle("FuturesLab — Market Regimes", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regime_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved regime_timeline.png")


def _plot_drawdown(results):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    colors = {"buy_hold": "#888888", "trend": "#2196F3", "mean_rev": "#FF9800", "adaptive": "#4CAF50"}
    labels = {"buy_hold": "Buy & Hold", "trend": "Trend", "mean_rev": "Mean-Rev", "adaptive": "Adaptive"}

    for ax, (mkt, df) in zip(axes, results.items()):
        for strat in ["buy_hold", "trend", "adaptive"]:
            col = f"ret_{strat}"
            if col not in df.columns:
                continue
            ret = df[col].dropna()
            cum = np.exp(ret.cumsum())
            rolling_max = cum.expanding().max()
            dd = (cum - rolling_max) / rolling_max
            ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color=colors[strat], label=labels[strat])
        ax.set_title(f"{MARKETS[mkt]['name']} — Drawdown", fontsize=11, fontweight="bold")
        ax.set_ylabel("Drawdown")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("FuturesLab — Strategy Drawdowns", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "drawdowns.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved drawdowns.png")


def _plot_monthly_heatmap(results):
    """Monthly return heatmaps for each strategy × market."""
    try:
        import matplotlib.cm as cm
        from matplotlib.colors import TwoSlopeNorm

        for mkt, df in results.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            axes_flat = axes.flatten()
            strats = ["buy_hold", "trend", "mean_rev", "adaptive"]
            strat_names = ["Buy & Hold", "Trend", "Mean-Rev", "Adaptive"]

            for ax, strat, strat_name in zip(axes_flat, strats, strat_names):
                ret = df[f"ret_{strat}"].dropna()
                mr = np.expm1(ret.resample("ME").sum())
                if mr.empty:
                    ax.set_visible(False)
                    continue
                pivot = mr.groupby([mr.index.year, mr.index.month]).mean().unstack()
                pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
                vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01) if pivot.notna().any().any() else 0.05
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", norm=norm)
                ax.set_xticks(range(pivot.shape[1]))
                ax.set_xticklabels(pivot.columns, fontsize=7)
                ax.set_yticks(range(pivot.shape[0]))
                ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
                ax.set_title(strat_name, fontsize=9)
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        v = pivot.iloc[i, j]
                        if not np.isnan(v):
                            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=5.5)
                plt.colorbar(im, ax=ax, shrink=0.6)

            plt.suptitle(f"{MARKETS[mkt]['name']} — Monthly Return Heatmaps", fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{mkt}_monthly_heatmap.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("Saved %s monthly heatmap", mkt)
    except Exception as e:
        logger.warning("Monthly heatmap failed: %s", e)


# ─────────────────────────────────────────────
# 9. Save All Results
# ─────────────────────────────────────────────
def step9_save(results, all_metrics):
    logger.info("STEP 9: Saving all results")
    save_results(all_metrics, results, DATA_PROC_DIR, label="full")

    # Also save summary CSV
    summary_rows = []
    for mkt, mdf in all_metrics.items():
        for strat_name, row in mdf.iterrows():
            r = {"market": mkt, "strategy": strat_name}
            r.update(row.to_dict())
            summary_rows.append(r)
    pd.DataFrame(summary_rows).to_csv(DATA_PROC_DIR / "all_metrics_summary.csv", index=False)
    logger.info("Summary saved to all_metrics_summary.csv")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FuturesLab -- Regime-Aware Futures Research")
    print("=" * 60)

    data = step1_data()
    regime_stats = step2_regime_stats(data)
    results = step3_backtest(data)
    all_metrics = step4_metrics(results)
    period_metrics = step5_period_breakdown(results)
    robust_df = step6_robustness(data)
    event_results = step7_macro(data)
    step8_charts(data, results)
    step9_save(results, all_metrics)

    print("\n" + "=" * 60)
    print("  FuturesLab Research Pipeline Complete")
    print(f"  Charts: {FIGURES_DIR}")
    print(f"  Data:   {DATA_PROC_DIR}")
    print("=" * 60)
