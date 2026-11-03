"""
tests/test_risk.py  (fixed)
============================
Unit tests for risk management (src/risk.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.risk import compute_position_size, apply_atr_stop, transaction_cost
from src.config import (
    TARGET_ANNUALIZED_VOL, MAX_POSITION_SIZE,
    HVOL_POSITION_SCALE, ATR_STOP_MULTIPLIER,
    TOTAL_COST_PER_TRADE_BPS,
)
from src.regimes import REGIME_HIGH_VOL, REGIME_TRENDING, REGIME_SIDEWAYS


def make_risk_df(n=100, rvol=0.20):
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "RVOL20": [rvol] * n,
        "ATR14":  [2.0] * n,
        "Close":  [100.0] * n,
        "Regime": [REGIME_TRENDING] * n,
    }, index=dates)


class TestPositionSizing:
    def test_inverse_volatility(self):
        """Higher vol → smaller position size."""
        df_low  = make_risk_df(rvol=0.10)
        df_high = make_risk_df(rvol=0.40)
        assert compute_position_size(df_low).mean() > compute_position_size(df_high).mean()

    def test_target_vol_at_target(self):
        """When RVOL equals TARGET_VOL, position size should be ~1.0."""
        df = make_risk_df(rvol=TARGET_ANNUALIZED_VOL)
        size = compute_position_size(df).mean()
        assert abs(size - 1.0) < 0.01

    def test_max_cap_respected(self):
        """Position size should never exceed MAX_POSITION_SIZE."""
        df = make_risk_df(rvol=0.001)  # very low vol → huge raw size
        size = compute_position_size(df)
        assert (size <= MAX_POSITION_SIZE + 1e-9).all()

    def test_hvol_regime_reduces_size(self):
        """HIGH_VOL regime should reduce position size by HVOL_POSITION_SCALE."""
        df = make_risk_df(rvol=TARGET_ANNUALIZED_VOL)
        df["Regime"] = REGIME_HIGH_VOL
        size_hvol = compute_position_size(df).mean()
        expected = 1.0 * HVOL_POSITION_SCALE
        assert abs(size_hvol - expected) < 0.01

    def test_non_negative(self):
        """Position size should always be non-negative."""
        df = make_risk_df(rvol=0.30)
        assert (compute_position_size(df) >= 0).all()


class TestATRStop:
    def test_long_stop_triggers(self):
        """
        A long entry followed by a large price drop should trigger the stop.
        Signal: enter long on day 0, then go flat from day 5 onwards.
        Price: 100 on day 0-4, then drops to 80 (stop = 100 - 4 = 96).
        Stop fires on day 5; since signal is also 0 from day 5, position = 0.
        """
        atr_val = 2.0
        n = 30
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        close_vals = [100.0] * 5 + [80.0] * 25
        df = pd.DataFrame({
            "Close": close_vals,
            "ATR14": [atr_val] * n,
            "Open":  close_vals, "High": close_vals, "Low": close_vals,
        }, index=dates)
        # Signal: long for first 5 days, then flat
        sig_vals = [1.0] * 5 + [0.0] * 25
        signal = pd.Series(sig_vals, index=dates)
        result = apply_atr_stop(signal, df, atr_multiplier=ATR_STOP_MULTIPLIER)
        # After price drops to 80 (below stop at 96), position should be 0
        assert (result.iloc[6:] == 0.0).all()

    def test_short_stop_triggers(self):
        """
        A short entry followed by a large price rise should trigger the stop.
        """
        atr_val = 2.0
        n = 30
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        close_vals = [100.0] * 5 + [120.0] * 25
        df = pd.DataFrame({
            "Close": close_vals,
            "ATR14": [atr_val] * n,
            "Open":  close_vals, "High": close_vals, "Low": close_vals,
        }, index=dates)
        sig_vals = [-1.0] * 5 + [0.0] * 25
        signal = pd.Series(sig_vals, index=dates)
        result = apply_atr_stop(signal, df, atr_multiplier=ATR_STOP_MULTIPLIER)
        assert (result.iloc[6:] == 0.0).all()

    def test_no_stop_when_price_flat(self):
        """No stop should trigger when price stays flat after entry."""
        n = 50
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "Close": [100.0] * n,
            "ATR14": [2.0] * n,
            "Open":  [100.0] * n, "High": [101.0] * n, "Low": [99.0] * n,
        }, index=dates)
        signal = pd.Series([1.0] * n, index=dates)
        result = apply_atr_stop(signal, df, atr_multiplier=ATR_STOP_MULTIPLIER)
        assert (result == 1.0).all()


class TestTransactionCost:
    def test_non_negative(self):
        """Transaction costs must always be non-negative."""
        rng = np.random.default_rng(0)
        n = 50
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        pos = pd.Series(rng.choice([-1.0, 0.0, 1.0], n).astype(float), index=dates)
        cost = transaction_cost(pos, cost_bps=TOTAL_COST_PER_TRADE_BPS)
        assert (cost >= 0).all()

    def test_no_cost_when_position_unchanged(self):
        """No cost on days with no position change (after initial entry)."""
        n = 20
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        pos = pd.Series([1.0] * n, index=dates)
        cost = transaction_cost(pos, cost_bps=TOTAL_COST_PER_TRADE_BPS)
        # Only first day has a cost
        assert (cost.iloc[1:] == 0.0).all()

    def test_cost_proportional_to_position_change(self):
        """Larger position change → larger cost."""
        n = 5
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        pos_small = pd.Series([0.0, 0.5, 0.5, 0.5, 0.5], index=dates)
        pos_large = pd.Series([0.0, 1.5, 1.5, 1.5, 1.5], index=dates)
        cost_small = transaction_cost(pos_small, 10).sum()
        cost_large = transaction_cost(pos_large, 10).sum()
        assert cost_large > cost_small
