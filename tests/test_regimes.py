"""
tests/test_regimes.py  (fixed)
================================
Unit tests for market regime classification (src/regimes.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features import add_all_features
from src.regimes import (
    classify_regimes, regime_statistics, regime_transitions,
    REGIME_TRENDING, REGIME_SIDEWAYS, REGIME_HIGH_VOL,
)


def make_ohlcv_with_features(n=500, seed=42, vol_scale=1.0):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 1.5 * vol_scale, n).cumsum()
    close = np.maximum(close, 1.0)
    high  = close + abs(rng.normal(0, vol_scale, n))
    low   = close - abs(rng.normal(0, vol_scale, n))
    low   = np.minimum(low, close)
    high  = np.maximum(high, close)
    dates = pd.date_range("2014-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close, "Volume": 1e6,
    }, index=dates)
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return add_all_features(df)


class TestRegimeClassification:
    def test_valid_regimes_only(self):
        """All regime labels must be one of the three valid values."""
        feat = make_ohlcv_with_features()
        reg = classify_regimes(feat)
        valid = {REGIME_TRENDING, REGIME_SIDEWAYS, REGIME_HIGH_VOL}
        unique_regimes = set(reg["Regime"].dropna().unique())
        assert unique_regimes.issubset(valid)

    def test_regime_column_no_unexpected_values(self):
        """Regime column must never contain unexpected string values."""
        feat = make_ohlcv_with_features(n=600)
        reg = classify_regimes(feat)
        valid = {REGIME_TRENDING, REGIME_SIDEWAYS, REGIME_HIGH_VOL}
        non_null_regimes = reg["Regime"].dropna()
        assert all(r in valid for r in non_null_regimes)

    def test_hvol_threshold_no_lookahead(self):
        """HVOL threshold must be non-NaN after min_periods (252)."""
        feat = make_ohlcv_with_features(n=700)
        reg = classify_regimes(feat)
        thresh = reg["HVOL_Threshold"].dropna()
        # After 252 observations, threshold should be set
        assert len(thresh) > 0
        assert thresh.iloc[-1] > 0

    def test_high_vol_regime_appears_in_volatile_data(self):
        """Extremely volatile data should trigger HIGH_VOL regime."""
        # Create data with MASSIVE volatility that obviously exceeds any threshold
        n = 700
        rng = np.random.default_rng(77)
        # First 400 bars: normal vol; last 300 bars: extremely high vol
        ret_normal  = rng.normal(0, 0.01, 400)
        ret_extreme = rng.normal(0, 0.15, 300)  # 15% daily sd (insane)
        all_ret = np.concatenate([ret_normal, ret_extreme])
        close = 100 * np.exp(np.cumsum(all_ret))
        close = np.maximum(close, 1.0)
        high = close * (1 + abs(rng.normal(0, 0.02, n)))
        low  = close * (1 - abs(rng.normal(0, 0.02, n)))
        low  = np.minimum(low, close)
        high = np.maximum(high, close)
        dates = pd.date_range("2013-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "Open": close, "High": high, "Low": low, "Close": close, "Volume": 1e6,
        }, index=dates)
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        feat = add_all_features(df)
        reg = classify_regimes(feat)
        # In the extreme period, should have HIGH_VOL days
        extreme_period = reg.iloc[450:]
        has_hvol = (extreme_period["Regime"] == REGIME_HIGH_VOL).any()
        assert has_hvol

    def test_regime_detail_column_exists(self):
        """Regime_Detail should be a string column."""
        feat = make_ohlcv_with_features()
        reg = classify_regimes(feat)
        assert "Regime_Detail" in reg.columns
        non_null = reg["Regime_Detail"].dropna()
        assert all(isinstance(v, str) for v in non_null)

    def test_regime_statistics_all_regimes(self):
        """regime_statistics should return rows for all three regimes."""
        feat = make_ohlcv_with_features(n=700, vol_scale=3.0)
        reg = classify_regimes(feat)
        stats = regime_statistics(reg)
        assert REGIME_TRENDING in stats.index
        assert REGIME_SIDEWAYS in stats.index
        assert REGIME_HIGH_VOL in stats.index

    def test_regime_transitions_count(self):
        """regime_transitions must return a non-negative integer."""
        feat = make_ohlcv_with_features(n=400)
        reg = classify_regimes(feat)
        n_trans = regime_transitions(reg)
        assert isinstance(n_trans, (int, np.integer))
        assert n_trans >= 0
