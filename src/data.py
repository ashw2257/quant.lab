"""
FuturesLab — data.py
====================
Data acquisition, cleaning, and persistence module.

Data Source
-----------
Yahoo Finance via the `yfinance` library.

Tickers used:
  ES=F  →  E-mini S&P 500 Futures (continuous front-month roll, Yahoo Finance)
  GC=F  →  Gold Futures (continuous front-month roll, Yahoo Finance)
  CL=F  →  WTI Crude Oil Futures (continuous front-month roll, Yahoo Finance)

Important caveat on continuous futures:
----------------------------------------
Yahoo Finance's =F tickers represent a synthetic rolling series.
At each contract expiry, Yahoo stitches the front-month contract to the
next.  This is NOT the same as a CME-published Panama-channel back-adjusted
continuous contract.  Price gaps at roll dates remain; there is no backward
price adjustment.

This means:
  1. Price-level comparisons across roll dates may be misleading.
  2. Roll slippage is NOT explicitly modelled in the raw price series.
  3. Open interest is typically that of the front-month contract.
  4. Volume data is available and reflects front-month volume.

For a research project of this scope, this is the industry-standard
publicly available proxy.  We document it fully and do not pretend
otherwise.

Pipeline
--------
download_raw()       → fetch from Yahoo Finance → save to data/raw/
load_raw()           → load saved CSV
clean()              → validate and standardise
save_processed()     → save to data/processed/ as parquet
load_processed()     → load processed parquet
run_pipeline()       → orchestrate the full pipeline
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# yfinance imported lazily so tests that don't need network still pass
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

from src.config import (
    DATA_RAW_DIR,
    DATA_PROC_DIR,
    MARKETS,
    DATA_START,
    DATA_END,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
# Columns we keep in processed output (standardised internal format)
PROCESSED_COLS = ["Open", "High", "Low", "Close", "Volume", "Return"]


# ─────────────────────────────────────────────
# Acquisition
# ─────────────────────────────────────────────

def download_raw(
    market_key: str,
    start: str = DATA_START,
    end: str = DATA_END,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download raw OHLCV data from Yahoo Finance and cache locally.

    Parameters
    ----------
    market_key : str
        Key in MARKETS dict (e.g. 'ES', 'GC', 'CL').
    start : str
        ISO date string for start of data.
    end : str
        ISO date string for end of data.
    force : bool
        If True, re-download even if cached file exists.

    Returns
    -------
    pd.DataFrame
        Raw OHLCV DataFrame.
    """
    if not _YF_AVAILABLE:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    cfg = MARKETS[market_key]
    ticker = cfg["ticker"]
    raw_path = DATA_RAW_DIR / f"{market_key}_raw.csv"

    if raw_path.exists() and not force:
        logger.info("Loading cached raw data for %s from %s", market_key, raw_path)
        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        return df

    logger.info(
        "Downloading %s (%s) from Yahoo Finance [%s → %s]",
        market_key, ticker, start, end,
    )

    # retry logic for intermittent network failures
    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,  # adjusts for dividends/splits where applicable
            )
            break
        except Exception as exc:
            logger.warning("Attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Failed to download {ticker} after 3 attempts.")

    if raw.empty:
        raise ValueError(f"Downloaded data for {ticker} is empty.")

    # Flatten multi-level columns produced by yfinance >= 0.2
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index.name = "Date"
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path)
    logger.info("Saved raw data → %s  (%d rows)", raw_path, len(raw))
    return raw


def load_raw(market_key: str) -> pd.DataFrame:
    """Load cached raw CSV from data/raw/."""
    raw_path = DATA_RAW_DIR / f"{market_key}_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No raw data found for {market_key}. Run download_raw() first."
        )
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ─────────────────────────────────────────────
# Cleaning & Validation
# ─────────────────────────────────────────────

def clean(df: pd.DataFrame, market_key: str) -> pd.DataFrame:
    """
    Validate and clean raw OHLCV data.

    Checks performed (in order):
    1.  Ensure DatetimeIndex.
    2.  Remove duplicate dates (keep first).
    3.  Sort by date ascending.
    4.  Check for and handle missing OHLC values.
    5.  Validate OHLC relationships (High >= max(Open,Close), Low <= min(Open,Close)).
    6.  Flag extreme daily returns (|return| > 20%) for inspection.
    7.  Ensure numeric types.
    8.  Add daily log-return column.
    9.  Drop rows with any remaining NaN in OHLC after forward-fill.
    10. Log summary of all cleaning actions.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data.
    market_key : str
        Market identifier (for logging).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised columns + 'Return'.
    """
    issues: list[str] = []
    original_len = len(df)

    # 1. DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # 2. Duplicate dates
    n_dups = df.index.duplicated().sum()
    if n_dups > 0:
        issues.append(f"Removed {n_dups} duplicate dates (kept first occurrence).")
        df = df[~df.index.duplicated(keep="first")]

    # 3. Sort
    df = df.sort_index()

    # 4. Keep only recognised OHLCV columns
    available_cols = [c for c in REQUIRED_COLS if c in df.columns]
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}. Will proceed with available: {available_cols}.")
    df = df[available_cols].copy()

    # 5. Numeric types
    for col in available_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Missing OHLC — forward fill then drop residual NaN
    n_missing_before = df[["Open", "High", "Low", "Close"]].isna().sum().sum()
    if n_missing_before > 0:
        issues.append(
            f"Found {n_missing_before} missing OHLC values. "
            "Forward-filling (i.e. carry prior close forward)."
        )
        df[["Open", "High", "Low", "Close"]] = (
            df[["Open", "High", "Low", "Close"]].ffill()
        )
    n_missing_after = df[["Open", "High", "Low", "Close"]].isna().sum().sum()
    if n_missing_after > 0:
        n_drop = df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum()
        issues.append(f"Dropped {n_drop} rows with unresolvable missing OHLC values.")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # 7. Missing Volume
    if "Volume" in df.columns:
        n_vol_missing = df["Volume"].isna().sum()
        if n_vol_missing > 0:
            issues.append(
                f"Found {n_vol_missing} missing Volume values. Filling with 0."
            )
            df["Volume"] = df["Volume"].fillna(0)

    # 8. OHLC relationship validation
    invalid_high = df["High"] < df[["Open", "Close"]].max(axis=1)
    invalid_low  = df["Low"]  > df[["Open", "Close"]].min(axis=1)
    n_invalid = (invalid_high | invalid_low).sum()
    if n_invalid > 0:
        issues.append(
            f"Found {n_invalid} rows with invalid OHLC relationships "
            "(High < max(O,C) or Low > min(O,C)). "
            "Correcting: High = max(O,H,C), Low = min(O,L,C)."
        )
        df["High"] = df[["Open", "High", "Close"]].max(axis=1)
        df["Low"]  = df[["Open", "Low",  "Close"]].min(axis=1)

    # 9. Non-positive prices
    non_pos = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    if non_pos.sum() > 0:
        issues.append(
            f"Found {non_pos.sum()} rows with non-positive prices. Dropping."
        )
        df = df[~non_pos]

    # 10. Compute daily log return
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 11. Flag extreme returns (do NOT silently drop)
    extreme = df["Return"].abs() > 0.20
    if extreme.sum() > 0:
        extreme_dates = df.index[extreme].tolist()
        issues.append(
            f"Flagged {extreme.sum()} trading days with |return| > 20%: "
            f"{[str(d.date()) for d in extreme_dates[:10]]} "
            "(retained — likely genuine market events; see CL=F March 2020)."
        )

    # Summary
    final_len = len(df)
    logger.info(
        "[%s] Cleaning complete: %d → %d rows. Issues: %d",
        market_key, original_len, final_len, len(issues),
    )
    for issue in issues:
        logger.warning("[%s] %s", market_key, issue)

    return df


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────

def save_processed(df: pd.DataFrame, market_key: str) -> Path:
    """Save cleaned DataFrame to data/processed/ as parquet."""
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_PROC_DIR / f"{market_key}.parquet"
    df.to_parquet(path)
    logger.info("Saved processed data → %s", path)
    return path


def load_processed(market_key: str) -> pd.DataFrame:
    """Load processed parquet from data/processed/."""
    path = DATA_PROC_DIR / f"{market_key}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found for {market_key}. "
            "Run run_pipeline() first."
        )
    return pd.read_parquet(path)


# ─────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────

def run_pipeline(
    markets: Optional[list[str]] = None,
    start: str = DATA_START,
    end: str = DATA_END,
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end data pipeline for all (or specified) markets.

    Parameters
    ----------
    markets : list[str] | None
        Market keys to process. Defaults to all MARKETS.
    start : str
        ISO date for data start.
    end : str
        ISO date for data end.
    force : bool
        Re-download raw data even if cached.

    Returns
    -------
    dict[str, pd.DataFrame]
        Market key → cleaned DataFrame mapping.
    """
    if markets is None:
        markets = list(MARKETS.keys())

    results = {}
    for mkt in markets:
        logger.info("=== Processing %s ===", mkt)
        raw = download_raw(mkt, start=start, end=end, force=force)
        cleaned = clean(raw, mkt)
        save_processed(cleaned, mkt)
        results[mkt] = cleaned
        logger.info(
            "[%s] Final dataset: %d rows from %s to %s",
            mkt,
            len(cleaned),
            cleaned.index[0].date(),
            cleaned.index[-1].date(),
        )

    return results


# ─────────────────────────────────────────────
# Data Quality Report
# ─────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame, market_key: str) -> dict:
    """
    Generate a concise data-quality summary dict for reporting.
    """
    report = {
        "market": market_key,
        "rows": len(df),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "missing_ohlc": int(df[["Open", "High", "Low", "Close"]].isna().sum().sum()),
        "missing_volume": int(df["Volume"].isna().sum()) if "Volume" in df.columns else "N/A",
        "extreme_returns": int((df["Return"].abs() > 0.20).sum()),
        "negative_prices": int((df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "mean_daily_return": float(df["Return"].mean()),
        "std_daily_return": float(df["Return"].std()),
    }
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = run_pipeline(force=False)
    for key, df in data.items():
        report = data_quality_report(df, key)
        print(f"\n=== {key} Quality Report ===")
        for k, v in report.items():
            print(f"  {k}: {v}")
