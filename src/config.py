"""
FuturesLab Configuration
========================
Central configuration constants for the entire project.
All parameters are defined here so that sensitivity analysis
and robustness testing can be performed by changing a single file.

IMPORTANT: Once the out-of-sample period is frozen (see OOS_START),
no parameters below should be changed based on OOS observations.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ─────────────────────────────────────────────
# Markets
# ─────────────────────────────────────────────
# Note: Yahoo Finance provides continuous-contract-like ETF/futures proxies.
# ES=F  → E-mini S&P 500 Futures continuous front-month (Yahoo Finance)
# GC=F  → Gold Futures continuous front-month (Yahoo Finance)
# CL=F  → Crude Oil Futures continuous front-month (Yahoo Finance)
# Yahoo Finance rolls to the next contract automatically for =F tickers.
# These are NOT identical to CME-published continuous contracts; they are
# synthetic rolling series stitched at expiry. We document this fully.
MARKETS = {
    "ES": {
        "ticker": "ES=F",
        "name": "S&P 500 E-mini Futures",
        "asset_class": "Equity Index Futures",
        "currency": "USD",
        "point_value": 50,          # USD per index point
        "initial_margin_approx": 12000,  # approximate, varies by broker
        "description": (
            "CME E-mini S&P 500 futures (ES). Each contract is worth "
            "$50 × index level. Initial margin ~$12k, giving ~20:1 leverage "
            "at typical index levels. Highly liquid; benchmark for equity index futures."
        ),
    },
    "GC": {
        "ticker": "GC=F",
        "name": "Gold Futures",
        "asset_class": "Precious Metals Futures",
        "currency": "USD",
        "point_value": 100,         # USD per troy oz × 100 oz/contract
        "initial_margin_approx": 9000,
        "description": (
            "CME COMEX Gold futures (GC). Each contract represents 100 troy oz. "
            "Safe-haven asset; often inversely correlated with USD and equity risk. "
            "Contango typical in normal carry environments."
        ),
    },
    "CL": {
        "ticker": "CL=F",
        "name": "WTI Crude Oil Futures",
        "asset_class": "Energy Futures",
        "currency": "USD",
        "point_value": 1000,        # USD per barrel × 1000 barrels/contract
        "initial_margin_approx": 6000,
        "description": (
            "CME NYMEX WTI Crude Oil futures (CL). Each contract represents "
            "1000 barrels. Highly sensitive to geopolitical events, EIA inventory "
            "reports, and OPEC decisions. Frequently exhibits both contango and "
            "backwardation depending on supply/demand dynamics."
        ),
    },
}

# ─────────────────────────────────────────────
# Data Acquisition
# ─────────────────────────────────────────────
DATA_START = "2012-01-01"     # Beginning of historical dataset
DATA_END   = "2026-08-31"     # End of historical dataset (latest available)

# ─────────────────────────────────────────────
# Out-of-Sample Split Dates
# These dates are FROZEN before any strategy development.
# Do not change these after seeing OOS results.
# ─────────────────────────────────────────────
DEV_START   = "2012-01-01"    # Development / In-sample start
DEV_END     = "2018-12-31"    # Development end
VAL_START   = "2019-01-01"    # Validation start
VAL_END     = "2021-12-31"    # Validation end (includes COVID volatility)
OOS_START   = "2022-01-01"    # Out-of-sample start (FROZEN - do not tune to this)
OOS_END     = "2026-08-31"    # Out-of-sample end

# ─────────────────────────────────────────────
# Technical Indicators (baseline parameters)
# ─────────────────────────────────────────────
EMA_FAST    = 20     # Fast EMA period
EMA_SLOW    = 50     # Slow EMA period
RSI_PERIOD  = 14     # RSI lookback
ATR_PERIOD  = 14     # ATR lookback
RVOL_PERIOD = 20     # Realized volatility window (trading days)
BREAKOUT_PERIOD = 20 # Rolling high/low for breakout detection
ZSCORE_WINDOW   = 20 # Rolling window for z-score calculation

# ─────────────────────────────────────────────
# Regime Classification Thresholds
# Chosen by examining distribution of EMA spread and volatility
# on the development set. Documented in Research_Decision_Log.md.
# ─────────────────────────────────────────────
# EMA spread = (EMA_FAST - EMA_SLOW) / EMA_SLOW
TREND_EMA_SPREAD_THRESHOLD = 0.005   # 0.5% normalized spread → meaningful trend
# Percentile threshold for HIGH_VOLATILITY regime
HVOL_RVOL_PERCENTILE       = 75      # top 25% of rolling vol → HIGH_VOL
# RSI bounds for momentum confirmation
RSI_BULL_THRESHOLD         = 50      # RSI > 50 confirms bullish momentum
RSI_BEAR_THRESHOLD         = 50      # RSI < 50 confirms bearish momentum

# ─────────────────────────────────────────────
# Strategy Parameters
# ─────────────────────────────────────────────
ZSCORE_ENTRY_LONG   = -1.5   # Enter long mean-reversion when z < -1.5
ZSCORE_ENTRY_SHORT  =  1.5   # Enter short mean-reversion when z > +1.5
ZSCORE_EXIT         =  0.3   # Exit when z reverts toward mean (|z| < 0.3)

# ─────────────────────────────────────────────
# Risk Management
# ─────────────────────────────────────────────
TARGET_ANNUALIZED_VOL  = 0.15    # Target portfolio volatility (15% annual)
MAX_POSITION_SIZE      = 1.5     # Maximum leverage / position size multiplier
ATR_STOP_MULTIPLIER    = 2.0     # Stop-loss = entry ± 2 × ATR
HVOL_POSITION_SCALE    = 0.25   # Reduce to 25% normal size in HIGH_VOL regime

# ─────────────────────────────────────────────
# Transaction Cost Model
# Research assumptions — NOT broker-specific live costs.
# ─────────────────────────────────────────────
TRANSACTION_COST_BPS   = 5.0    # One-way cost in basis points (round-trip = 10 bps)
SLIPPAGE_BPS           = 2.0    # Slippage per trade in basis points
TOTAL_COST_PER_TRADE_BPS = TRANSACTION_COST_BPS + SLIPPAGE_BPS  # 7 bps one-way

# ─────────────────────────────────────────────
# Annualization
# ─────────────────────────────────────────────
TRADING_DAYS_PER_YEAR  = 252

# ─────────────────────────────────────────────
# Robustness Test Parameter Grids
# ─────────────────────────────────────────────
ROBUST_EMA_PAIRS = [(15, 40), (20, 50), (30, 60)]
ROBUST_BREAKOUT_PERIODS = [15, 20, 30]
ROBUST_ZSCORE_THRESHOLDS = [1.0, 1.5, 2.0]
ROBUST_COST_BPS = [3.0, 7.0, 15.0]
ROBUST_HVOL_PERCENTILES = [65, 75, 85]
