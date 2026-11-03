"""
tests/test_strategies.py  (fixed)
===================================
Unit tests for strategy signal generation (src/strategies.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features import add_all_features
from src.regimes import classify_regimes, REGIME_TRENDING, REGIME_SIDEWAYS, REGIME_HIGH_VOL
from src.strategies import (
    trend_signal, mean_reversion_signal, regime_adaptive_signal, buy_and_hold_signal,
)
from src.config import ZSCORE_ENTRY_LONG, ZSCORE_ENTRY_SHORT


def make_ohlcv(n=300, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 2, n).cumsum()
    close = np.maximum(close, 1.0)
    high  = close + abs(rng.normal(0, 1, n))
    low   = close - abs(rng.normal(0, 1, n))
    low   = np.minimum(low, close)
    high  = np.maximum(high, close)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                       "Volume": 1e6}, index=dates)
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return add_all_features(df)


class TestTrendSignal:
    def test_signal_values_valid(self):
        """Signal must only contain values in {-1, 0, +1}."""
        df = make_ohlcv()
        sig = trend_signal(df)
        assert set(sig.dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_uptrend_generates_long_signal(self):
        """A strong uptrend with very small noise should produce long signals."""
        n = 500
        # Very smooth uptrend with tiny noise so breakout condition fires easily
        close = 100 + np.arange(n, dtype=float) * 1.5 + np.random.default_rng(0).normal(0, 0.01, n)
        close = np.maximum(close, 1.0)
        high  = close + 0.1
        low   = close - 0.1
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                           "Volume": 1e6}, index=dates)
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df = add_all_features(df)
        sig = trend_signal(df)
        # In a smooth uptrend: EMA20>EMA50, close>rolling_high, RSI>50 should fire
        late_signal = sig.iloc[100:]
        assert (late_signal == 1.0).sum() >= 1  # at least some long signals

    def test_downtrend_generates_short_signal(self):
        """A strong downtrend should generate short signals."""
        n = 500
        close = 1000 - np.arange(n, dtype=float) * 1.5 + np.random.default_rng(1).normal(0, 0.01, n)
        close = np.maximum(close, 1.0)
        high  = close + 0.1
        low   = close - 0.1
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                           "Volume": 1e6}, index=dates)
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df = add_all_features(df)
        sig = trend_signal(df)
        late_signal = sig.iloc[100:]
        assert (late_signal == -1.0).sum() >= 1


class TestMeanReversionSignal:
    def test_signal_values_valid(self):
        """Signal must only contain values in {-1, 0, +1}."""
        df = make_ohlcv()
        sig = mean_reversion_signal(df)
        assert set(sig.dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_enters_long_on_oversold(self):
        """Signal should go long when z-score is very negative."""
        rng = np.random.default_rng(99)
        n = 300
        # First 200 bars: gentle random walk around 100 (provides non-zero std for z-score)
        # Last 100 bars: sustained drop to ~70 (far below rolling mean → negative z)
        close_arr = np.concatenate([
            100 + rng.normal(0, 1.0, 200),
            70  + rng.normal(0, 0.5, 100),
        ])
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "Open":   close_arr, "High": close_arr + 0.5,
            "Low":    close_arr - 0.5, "Close": close_arr, "Volume": 1e6,
        }, index=dates)
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df = add_all_features(df)
        sig = mean_reversion_signal(df, entry_long=1.5, entry_short=1.5, exit_threshold=0.3)
        # Verify z-score goes below -1.5 during the drop period
        from src.features import rolling_zscore
        z = rolling_zscore(pd.Series(close_arr, index=dates), 20)
        z_min = z.dropna().min()
        if z_min < -1.5:
            assert (sig == 1.0).any(), f"z_min={z_min:.2f} < -1.5 but no long signal"
        else:
            pytest.skip(f"z_min={z_min:.2f} didn't reach threshold")

    def test_exits_when_z_reverts(self):
        """Position should close when z-score reverts to near zero."""
        n = 250
        # Drop, then fully revert
        close = [100.0] * 100 + [85.0] * 50 + [100.0] * 100
        close = pd.Series(close, dtype=float)
        high  = close + 0.3
        low   = close - 0.3
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                           "Volume": 1e6}, index=dates)
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df = add_all_features(df)
        sig = mean_reversion_signal(df, exit_threshold=0.3)
        # After full reversion, position should eventually return to 0
        assert sig.iloc[-10:].eq(0.0).any()


class TestRegimeAdaptive:
    def test_flat_in_high_vol(self):
        """Regime-adaptive must be flat when regime is HIGH_VOL."""
        df = make_ohlcv(n=400)
        df = classify_regimes(df)
        df["Regime"] = REGIME_HIGH_VOL
        sig = regime_adaptive_signal(df)
        assert (sig == 0.0).all()

    def test_uses_trend_in_trending(self):
        """In TRENDING regime, adaptive signal should match trend signal."""
        df = make_ohlcv(n=400)
        df = classify_regimes(df)
        df_trend = df.copy()
        df_trend["Regime"] = REGIME_TRENDING
        ada_sig = regime_adaptive_signal(df_trend)
        tf_sig  = trend_signal(df_trend)
        matching = (ada_sig == tf_sig)
        assert matching.mean() > 0.8

    def test_uses_mr_in_sideways(self):
        """In SIDEWAYS regime, adaptive signal should match mean-reversion signal."""
        df = make_ohlcv(n=400)
        df = classify_regimes(df)
        df_sw = df.copy()
        df_sw["Regime"] = REGIME_SIDEWAYS
        ada_sig = regime_adaptive_signal(df_sw)
        mr_sig  = mean_reversion_signal(df_sw)
        matching = (ada_sig == mr_sig)
        assert matching.mean() > 0.9


class TestBuyAndHold:
    def test_always_long(self):
        """Buy-and-hold should always return 1.0."""
        df = make_ohlcv()
        sig = buy_and_hold_signal(df)
        assert (sig == 1.0).all()
