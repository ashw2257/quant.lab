"""
tests/test_features.py  (fixed)
================================
Unit tests for technical indicator computation (src/features.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ema, rsi, atr, realized_vol, rolling_zscore,
    add_all_features, rolling_high, rolling_low,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_ohlcv(n=200, seed=42, trend="flat"):
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    if trend == "up":
        close = 100 + np.arange(n) * 0.5 + rng.normal(0, 1, n)
    elif trend == "down":
        close = 200 - np.arange(n) * 0.5 + rng.normal(0, 1, n)
    else:
        close = 100 + rng.normal(0, 2, n).cumsum()

    close = np.maximum(close, 1.0)
    high  = close + abs(rng.normal(0, 1, n))
    low   = close - abs(rng.normal(0, 1, n))
    low   = np.minimum(low, close)
    high  = np.maximum(high, close)
    open_ = close + rng.normal(0, 0.5, n)

    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": rng.integers(100000, 1000000, n).astype(float),
    }, index=dates)
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


# ─────────────────────────────────────────────
# EMA tests
# ─────────────────────────────────────────────

class TestEMA:
    def test_constant_series_equals_constant(self):
        """EMA of a constant series must equal that constant."""
        s = pd.Series([100.0] * 100)
        result = ema(s, span=20)
        assert abs(result.dropna().iloc[-1] - 100.0) < 1e-6

    def test_shorter_ema_reacts_faster(self):
        """EMA20 should be closer to recent price than EMA50 during a trend."""
        s = pd.Series([100.0] * 100 + [150.0] * 100)
        ema20 = ema(s, span=20)
        ema50 = ema(s, span=50)
        assert ema20.iloc[-1] > ema50.iloc[-1]

    def test_ema_nan_warmup(self):
        """First (span - 1) values should be NaN."""
        s = pd.Series(list(range(1, 101)), dtype=float)  # fixed: no dtype kwarg in range
        result = ema(s, span=20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()

    def test_ema_alpha_weight(self):
        """α = 2/(span+1) — verify by computing one step manually."""
        span = 3
        # alpha = 0.5; after 3-step warmup EMA=10; next = 0.5*20 + 0.5*10 = 15
        s = pd.Series([10.0, 10.0, 10.0, 20.0])
        result = ema(s, span=span)
        assert abs(result.iloc[-1] - 15.0) < 0.1


# ─────────────────────────────────────────────
# RSI tests
# ─────────────────────────────────────────────

class TestRSI:
    def test_rsi_range(self):
        """RSI must be in [0, 100]."""
        df = make_ohlcv(300)
        result = rsi(df["Close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_constant_up_rsi_high(self):
        """Pure upward series should yield RSI close to 100."""
        # Need enough data for EWM to converge — use 200 bars
        s = pd.Series([100.0 + i for i in range(200)], dtype=float)
        result = rsi(s, 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 90

    def test_constant_down_rsi_low(self):
        """Pure downward series should yield RSI close to 0."""
        s = pd.Series([200.0 - i for i in range(200)], dtype=float)
        result = rsi(s, 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] < 10

    def test_flat_rsi_midrange(self):
        """RSI for flat alternating series should be near 50."""
        vals = [100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(200)]
        s = pd.Series(vals, dtype=float)
        result = rsi(s, 14)
        valid = result.dropna()
        assert abs(valid.iloc[-1] - 50) < 15

    def test_rsi_warmup_nan(self):
        """RSI should be NaN before EWM warms up."""
        s = pd.Series(list(range(1, 101)), dtype=float)
        result = rsi(s, 14)
        assert result.iloc[:14].isna().all()


# ─────────────────────────────────────────────
# ATR tests
# ─────────────────────────────────────────────

class TestATR:
    def test_atr_positive(self):
        """ATR must always be positive (or NaN during warm-up)."""
        df = make_ohlcv(200)
        result = atr(df, 14)
        assert (result.dropna() > 0).all()

    def test_atr_captures_gap(self):
        """ATR should be elevated after a large overnight gap."""
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = [100.0] * n
        high  = [101.0] * n
        low   = [99.0] * n
        # Big gap on day 20: prior close=100, new close=130
        for i in range(20, n):
            close[i] = 130.0
            high[i]  = 131.0
            low[i]   = 129.0
        df = pd.DataFrame({
            "High": high, "Low": low, "Close": close, "Open": close,
        }, index=dates)
        result = atr(df, 14)
        # ATR after the gap (day 25+) should be larger than before the gap (day 10)
        after_gap  = result.dropna().iloc[15:20].mean()
        before_gap = result.dropna().iloc[0:5].mean()
        assert after_gap > before_gap

    def test_atr_warmup_nan(self):
        """ATR NaN warmup: first ~13 values should be NaN (EWM with min_periods=14)."""
        df = make_ohlcv(100)
        result = atr(df, 14)
        # First 13 should be NaN (0-indexed: 0 to 12), index 13 may have value
        assert result.iloc[:13].isna().all()


# ─────────────────────────────────────────────
# Realized Volatility tests
# ─────────────────────────────────────────────

class TestRealizedVol:
    def test_rvol_positive(self):
        """Realized vol must be positive."""
        df = make_ohlcv(200)
        result = realized_vol(df["Return"].dropna(), 20)
        assert (result.dropna() > 0).all()

    def test_rvol_annualisation(self):
        """Annualised vol ≈ daily_std × √252 (loosen tolerance for randomness)."""
        rng = np.random.default_rng(0)
        daily_ret = pd.Series(rng.normal(0, 0.01, 2000))  # large sample for convergence
        result = realized_vol(daily_ret, 252, annualize=True)
        expected_ann = 0.01 * np.sqrt(252)
        # Allow 0.01 tolerance for random variability
        assert abs(result.dropna().iloc[-1] - expected_ann) < 0.01

    def test_rvol_not_annualized(self):
        """Non-annualised vol should approximate daily std."""
        rng = np.random.default_rng(1)
        daily_ret = pd.Series(rng.normal(0, 0.01, 2000))
        result = realized_vol(daily_ret, 252, annualize=False)
        assert abs(result.dropna().iloc[-1] - 0.01) < 0.003


# ─────────────────────────────────────────────
# Rolling Z-Score tests
# ─────────────────────────────────────────────

class TestZScore:
    def test_zscore_mean_zero(self):
        """Z-score should have approximately zero mean over the window."""
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(100, 5, 500))
        result = rolling_zscore(s, 50)
        assert abs(result.dropna().mean()) < 0.1

    def test_zscore_std_approx_one(self):
        """Z-score std should be ≈ 1 over the window."""
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(100, 5, 500))
        result = rolling_zscore(s, 50)
        assert abs(result.dropna().std() - 1.0) < 0.2

    def test_extreme_deviation_large_zscore(self):
        """A price far from the mean should produce a large |z|."""
        base = [100.0] * 100
        s = pd.Series(base + [200.0])  # outlier at end
        result = rolling_zscore(s, 20)
        assert result.dropna().iloc[-1] > 3.0


# ─────────────────────────────────────────────
# Rolling High/Low (breakout) tests
# ─────────────────────────────────────────────

class TestBreakout:
    def test_rolling_high_lagged(self):
        """rolling_high must be lagged — today's value uses yesterday's window."""
        s = pd.Series([float(i) for i in range(1, 101)])
        rh = rolling_high(s, 5)
        # Day index 5: prior 5 days (indices 0-4) → values 1..5 → max=5
        assert rh.iloc[5] == 5.0

    def test_rolling_low_lagged(self):
        """rolling_low must be lagged."""
        s = pd.Series([float(100 - i) for i in range(100)])
        rl = rolling_low(s, 5)
        assert not np.isnan(rl.iloc[5])


# ─────────────────────────────────────────────
# Full feature pipeline
# ─────────────────────────────────────────────

class TestAddAllFeatures:
    def test_adds_expected_columns(self):
        """add_all_features should add all expected feature columns."""
        df = make_ohlcv(300)
        result = add_all_features(df)
        expected_cols = [
            "EMA20", "EMA50", "EMA_spread",
            "RSI14", "ATR14", "ATR14_pct", "RVOL20",
            "Roll_High20", "Roll_Low20", "Above_High20", "Below_Low20",
            "ZScore20",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_inf_values(self):
        """No infinite values should appear in feature columns."""
        df = make_ohlcv(300)
        result = add_all_features(df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not np.isinf(result[numeric_cols].values).any()
