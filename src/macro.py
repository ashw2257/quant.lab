"""
FuturesLab — macro.py
=====================
Lightweight macro/fundamental event study module.

Objective
---------
Demonstrate that fundamental macroeconomic information materially affects
futures-market volatility and price movement.

We do NOT attempt to predict macroeconomic releases.
We analyse historical market REACTIONS to known events.

Events studied
--------------
1. Federal Reserve FOMC interest-rate decisions
   Source: Federal Reserve website (scraped from public historical calendar)
   Frequency: ~8 per year
   Relevance: Rate decisions directly affect equity multiples (ES),
              gold (GC, as USD and rates drive gold), and energy (CL).

2. U.S. CPI releases
   Source: BLS (Bureau of Labor Statistics) — scraped from public data
   Frequency: Monthly
   Relevance: CPI surprises drive inflation expectations → asset prices.

3. EIA Weekly Petroleum Status Report (crude oil inventory)
   Source: EIA public API
   Frequency: Weekly (Wednesday)
   Relevance: Primary short-term driver of WTI crude oil (CL) futures.

Event Window Analysis
---------------------
For each event date, compute:
  day -1: return the day before the event (pre-positioning)
  day  0: return on the event day itself
  day +1: return the day after (initial reaction)

Then aggregate across all events:
  mean return per window day
  median return per window day
  standard deviation (volatility) per window day
  fraction of large moves (|return| > 1%)
  mean absolute return

Interpretation
--------------
If the mean absolute return on day 0 is significantly larger than on
day -1 or day +1, the event is causing above-normal price movement.
If returns on day +1 are larger in magnitude than day -1, the event
is likely creating new information that takes time to be fully digested.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# FOMC Decision Dates
# ─────────────────────────────────────────────
# Source: Federal Reserve (https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
# These are the dates of FOMC statement releases (usually afternoon, eastern time).
# For daily data, we treat these as the event date T=0.
# This list covers 2012–2026 (approximate). Compiled from public Fed records.

FOMC_DATES_RAW = [
    # 2012
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20",
    "2012-08-01","2012-09-13","2012-10-24","2012-12-12",
    # 2013
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19",
    "2013-07-31","2013-09-18","2013-10-30","2013-12-18",
    # 2014
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18",
    "2014-07-30","2014-09-17","2014-10-29","2014-12-17",
    # 2015
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17",
    "2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    # 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15",
    "2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    # 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14",
    "2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    # 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13",
    "2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
    "2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    # 2020
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29",
    "2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
    "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15",
    "2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14",
    "2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
    "2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
    "2025-07-30","2025-09-17","2025-11-05","2025-12-17",
    # 2026
    "2026-01-28","2026-03-18","2026-05-06","2026-06-17",
    "2026-07-29",
]

FOMC_DATES = pd.to_datetime(FOMC_DATES_RAW)


# ─────────────────────────────────────────────
# EIA Weekly Petroleum Report Dates
# ─────────────────────────────────────────────
# EIA releases the Weekly Petroleum Status Report every Wednesday (~10:30am ET).
# We generate approximate dates: every Wednesday from 2012 to present.

def _generate_eia_wednesdays(start: str = "2012-01-04", end: str = "2026-08-31") -> pd.DatetimeIndex:
    """Generate all Wednesdays in the given date range as EIA report proxy dates."""
    all_wednesdays = pd.date_range(start=start, end=end, freq="W-WED")
    return all_wednesdays


EIA_DATES = _generate_eia_wednesdays()


# ─────────────────────────────────────────────
# CPI Release Dates
# ─────────────────────────────────────────────
# BLS releases CPI-U for the prior month, typically second or third Tuesday.
# Approximate dates compiled from BLS historical release calendar.
# Source: https://www.bls.gov/cpi/

CPI_DATES_RAW = [
    # 2012
    "2012-01-19","2012-02-16","2012-03-15","2012-04-13","2012-05-15",
    "2012-06-14","2012-07-17","2012-08-15","2012-09-14","2012-10-16",
    "2012-11-15","2012-12-14",
    # 2013
    "2013-01-16","2013-02-21","2013-03-15","2013-04-16","2013-05-16",
    "2013-06-18","2013-07-16","2013-08-15","2013-09-17","2013-10-30",
    "2013-11-20","2013-12-17",
    # 2014
    "2014-01-16","2014-02-20","2014-03-18","2014-04-15","2014-05-15",
    "2014-06-17","2014-07-22","2014-08-19","2014-09-17","2014-10-22",
    "2014-11-20","2014-12-17",
    # 2015
    "2015-01-16","2015-02-26","2015-03-24","2015-04-17","2015-05-22",
    "2015-06-18","2015-07-17","2015-08-19","2015-09-16","2015-10-15",
    "2015-11-17","2015-12-15",
    # 2016
    "2016-01-20","2016-02-19","2016-03-16","2016-04-14","2016-05-17",
    "2016-06-16","2016-07-15","2016-08-16","2016-09-16","2016-10-18",
    "2016-11-17","2016-12-15",
    # 2017
    "2017-01-18","2017-02-15","2017-03-15","2017-04-14","2017-05-12",
    "2017-06-14","2017-07-14","2017-08-11","2017-09-14","2017-10-13",
    "2017-11-15","2017-12-13",
    # 2018
    "2018-01-12","2018-02-14","2018-03-13","2018-04-11","2018-05-10",
    "2018-06-12","2018-07-12","2018-08-10","2018-09-13","2018-10-11",
    "2018-11-14","2018-12-12",
    # 2019
    "2019-01-11","2019-02-13","2019-03-12","2019-04-10","2019-05-10",
    "2019-06-12","2019-07-11","2019-08-13","2019-09-12","2019-10-10",
    "2019-11-13","2019-12-11",
    # 2020
    "2020-01-14","2020-02-13","2020-03-11","2020-04-10","2020-05-12",
    "2020-06-10","2020-07-14","2020-08-12","2020-09-11","2020-10-13",
    "2020-11-12","2020-12-10",
    # 2021
    "2021-01-13","2021-02-10","2021-03-10","2021-04-13","2021-05-12",
    "2021-06-10","2021-07-13","2021-08-11","2021-09-14","2021-10-13",
    "2021-11-10","2021-12-10",
    # 2022
    "2022-01-12","2022-02-10","2022-03-10","2022-04-12","2022-05-11",
    "2022-06-10","2022-07-13","2022-08-10","2022-09-13","2022-10-13",
    "2022-11-10","2022-12-13",
    # 2023
    "2023-01-12","2023-02-14","2023-03-14","2023-04-12","2023-05-10",
    "2023-06-13","2023-07-12","2023-08-10","2023-09-13","2023-10-12",
    "2023-11-14","2023-12-12",
    # 2024
    "2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15",
    "2024-06-12","2024-07-11","2024-08-14","2024-09-11","2024-10-10",
    "2024-11-13","2024-12-11",
    # 2025
    "2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-13",
    "2025-06-11","2025-07-15","2025-08-12",
]

CPI_DATES = pd.to_datetime(CPI_DATES_RAW)


# ─────────────────────────────────────────────
# Event window analysis
# ─────────────────────────────────────────────

def event_window_returns(
    returns: pd.Series,
    event_dates: pd.DatetimeIndex,
    window: int = 1,
    label: str = "event",
) -> pd.DataFrame:
    """
    Compute returns in a symmetric window around each event date.

    For each event date, find the nearest available trading day within
    a tolerance, then extract returns for days -window to +window.

    Parameters
    ----------
    returns : pd.Series
        Daily log return series with DatetimeIndex.
    event_dates : pd.DatetimeIndex
        Dates of macro events.
    window : int
        Days before/after event to include.
    label : str
        Event type label for the output DataFrame.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
          event_date, offset, return, label
    """
    trading_days = returns.index
    records = []

    for ev_date in event_dates:
        # Find nearest trading day (within 2 calendar days)
        diff = (trading_days - ev_date).days
        candidates = trading_days[abs(diff) <= 2]
        if len(candidates) == 0:
            continue
        nearest = candidates[np.argmin(abs((candidates - ev_date).days))]
        idx = returns.index.get_loc(nearest)

        for offset in range(-window, window + 1):
            i = idx + offset
            if 0 <= i < len(returns):
                records.append({
                    "event_date": ev_date,
                    "event_label": label,
                    "offset": offset,
                    "return": returns.iloc[i],
                    "abs_return": abs(returns.iloc[i]),
                    "large_move": abs(returns.iloc[i]) > 0.01,  # >1%
                })

    return pd.DataFrame(records)


def event_study_summary(window_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event-window statistics.

    Parameters
    ----------
    window_df : pd.DataFrame
        Output from event_window_returns().

    Returns
    -------
    pd.DataFrame
        Statistics by offset day: mean_return, median_return, std_return,
        mean_abs_return, pct_large_moves, n_events.
    """
    if window_df.empty:
        return pd.DataFrame()

    summary = (
        window_df.groupby("offset")
        .agg(
            mean_return=("return", "mean"),
            median_return=("return", "median"),
            std_return=("return", "std"),
            mean_abs_return=("abs_return", "mean"),
            pct_large_moves=("large_move", "mean"),
            n_events=("return", "count"),
        )
    )
    return summary


def run_event_study(
    market_returns: Dict[str, pd.Series],
    window: int = 1,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run event study for all markets and event types.

    Parameters
    ----------
    market_returns : dict[str, pd.Series]
        Market key → daily log return series.
    window : int
        Event window ± days.

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        {market: {event_type: summary_df}}
    """
    events = {
        "FOMC": FOMC_DATES,
        "CPI":  CPI_DATES,
        "EIA":  EIA_DATES,
    }

    results = {}
    for mkt, ret_series in market_returns.items():
        results[mkt] = {}
        for ev_name, ev_dates in events.items():
            if mkt != "CL" and ev_name == "EIA":
                continue  # EIA inventory primarily relevant for crude oil
            window_df = event_window_returns(ret_series, ev_dates, window, ev_name)
            summary   = event_study_summary(window_df)
            results[mkt][ev_name] = {
                "window_df": window_df,
                "summary": summary,
            }
        logger.info("Event study complete for %s", mkt)

    return results
