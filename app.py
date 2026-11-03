"""
FuturesLab — app.py
====================
Streamlit dashboard for FuturesLab research.

Pages:
  1. Market Overview   — price, regime, current signal
  2. Strategy          — active strategy with WHY explanation
  3. Backtest          — equity curves, drawdown, performance table
  4. Regime + Macro    — regime timeline, event study

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.config import MARKETS, DATA_PROC_DIR, OOS_START, OOS_END, TRADING_DAYS_PER_YEAR
from src.data import load_processed
from src.features import add_all_features
from src.regimes import classify_regimes, regime_statistics
from src.strategies import explain_signal
from src.evaluation import compute_metrics, cagr, max_drawdown, sharpe_ratio
from src.macro import (
    FOMC_DATES, CPI_DATES, EIA_DATES,
    event_window_returns, event_study_summary,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FuturesLab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REGIME_COLORS = {
    "TRENDING":  "#4CAF50",
    "SIDEWAYS":  "#2196F3",
    "HIGH_VOL":  "#F44336",
}

STRATEGY_COLORS = {
    "Buy & Hold":      "#888888",
    "Static Trend":    "#2196F3",
    "Static Mean-Rev": "#FF9800",
    "Regime-Adaptive": "#4CAF50",
}


# ─────────────────────────────────────────────
# Data Loading (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_market_data():
    """Load and prepare all market data with features."""
    data = {}
    for mkt in MARKETS:
        try:
            df = load_processed(mkt)
            df = add_all_features(df)
            df = classify_regimes(df)
            data[mkt] = df
        except FileNotFoundError:
            pass
    return data


@st.cache_data(show_spinner=False)
def load_backtest_data():
    """Load pre-computed backtest results."""
    results = {}
    for mkt in MARKETS:
        try:
            path = DATA_PROC_DIR / f"{mkt}_backtest_full.parquet"
            if path.exists():
                results[mkt] = pd.read_parquet(path)
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def build_sidebar():
    st.sidebar.image("https://via.placeholder.com/200x60?text=FuturesLab", width=200)
    st.sidebar.title("FuturesLab")
    st.sidebar.caption("Regime-Aware Futures Strategy Research")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Market Overview", "🎯 Strategy", "📈 Backtest", "🌐 Regime & Macro"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**Research Parameters**
- OOS Start: `2022-01-01`
- Markets: ES, GC, CL
- Data: Yahoo Finance

**Risk Model**
- Target Vol: 15%
- Max Leverage: 1.5×
- ATR Stop: 2×ATR14
""")
    return page


# ─────────────────────────────────────────────
# Page 1: Market Overview
# ─────────────────────────────────────────────

def page_market_overview(data):
    st.title("📊 Market Overview")
    st.caption("Latest market snapshot for each futures market. Data from Yahoo Finance (continuous contract proxies).")

    if not data:
        st.error("No processed data found. Run `python run_research.py` first.")
        return

    # Top metric row
    cols = st.columns(len(data))
    for col, (mkt, df) in zip(cols, data.items()):
        last = df.dropna(subset=["Close"]).iloc[-1]
        regime = last.get("Regime", "N/A")
        ret = last.get("Return", np.nan)
        rvol = last.get("RVOL20", np.nan)
        atr  = last.get("ATR14", np.nan)
        rsi  = last.get("RSI14", np.nan)

        regime_color = REGIME_COLORS.get(regime, "#999999")
        col.metric(
            label=f"**{MARKETS[mkt]['name']}**",
            value=f"${last['Close']:.2f}",
            delta=f"{ret:.2%}" if not np.isnan(ret) else "N/A",
        )
        col.markdown(
            f"<span style='background:{regime_color};color:white;padding:2px 8px;"
            f"border-radius:4px;font-weight:bold'>{regime}</span>",
            unsafe_allow_html=True,
        )
        col.caption(f"RVOL: {rvol:.1%} | ATR: {atr:.2f} | RSI: {rsi:.1f}")

    st.markdown("---")

    # Price charts with regime color bands
    selected_mkt = st.selectbox(
        "Select Market", list(data.keys()),
        format_func=lambda k: MARKETS[k]["name"],
    )
    df = data[selected_mkt]

    # Lookback selector
    lookback = st.slider("Lookback (trading days)", 60, len(df), 500, step=50)
    plot_df = df.tail(lookback)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=["Price + EMA + Regime", "RSI", "Volume"],
        vertical_spacing=0.04,
    )

    # Price
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"],
        name="Price", showlegend=False,
        increasing_line_color="#4CAF50", decreasing_line_color="#F44336",
    ), row=1, col=1)

    ema_fast_col = "EMA20"
    ema_slow_col = "EMA50"
    if ema_fast_col in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[ema_fast_col],
                                  name="EMA20", line=dict(color="#2196F3", width=1.5)), row=1, col=1)
    if ema_slow_col in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[ema_slow_col],
                                  name="EMA50", line=dict(color="#FF9800", width=1.5, dash="dot")), row=1, col=1)

    # Regime coloring via shapes
    if "Regime" in plot_df.columns:
        reg = plot_df["Regime"].dropna()
        for regime_name, color in REGIME_COLORS.items():
            mask = reg == regime_name
            if not mask.any():
                continue
            starts = reg[mask & (~mask.shift(1, fill_value=False))].index
            ends   = reg[mask & (~mask.shift(-1, fill_value=False))].index
            for s, e in zip(starts, ends):
                fig.add_vrect(
                    x0=s, x1=e,
                    fillcolor=color, opacity=0.07,
                    layer="below", line_width=0,
                    row=1, col=1,
                )

    # RSI
    if "RSI14" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI14"],
                                  name="RSI14", line=dict(color="#9C27B0", width=1.2)), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="red", dash="dot", width=1), row=2, col=1)
        fig.add_hline(y=50, line=dict(color="gray", dash="dot", width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="green", dash="dot", width=1), row=2, col=1)

    # Volume
    if "Volume" in plot_df.columns:
        colors = ["#4CAF50" if plot_df["Close"].iloc[i] >= plot_df["Open"].iloc[i] else "#F44336"
                  for i in range(len(plot_df))]
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"],
                              marker_color=colors, name="Volume", showlegend=False), row=3, col=1)

    fig.update_layout(
        height=700,
        title=f"{MARKETS[selected_mkt]['name']} — Price Chart",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Regime distribution
    st.subheader("Regime Distribution (Full History)")
    if "Regime" in df.columns:
        reg_counts = df["Regime"].value_counts(normalize=True).reset_index()
        reg_counts.columns = ["Regime", "Fraction"]
        fig_reg = px.bar(
            reg_counts, x="Regime", y="Fraction",
            color="Regime", color_discrete_map=REGIME_COLORS,
            title="Fraction of Trading Days by Regime",
            text_auto=".1%",
        )
        fig_reg.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_reg, use_container_width=True)


# ─────────────────────────────────────────────
# Page 2: Strategy
# ─────────────────────────────────────────────

def page_strategy(data):
    st.title("🎯 Strategy Signal")
    st.caption("Current strategy and signal with full explanation based on live indicators.")

    if not data:
        st.error("No data. Run pipeline first.")
        return

    mkt = st.selectbox("Market", list(data.keys()), format_func=lambda k: MARKETS[k]["name"])
    df  = data[mkt]

    # Date selector
    max_date = df.dropna(subset=["Regime"]).index[-1]
    sel_date = st.date_input("Signal date", value=max_date.date(),
                              min_value=df.index[0].date(), max_value=max_date.date())
    sel_ts = pd.Timestamp(sel_date)
    try:
        row_idx = df.index.asof(sel_ts)
        signal_info = explain_signal(df, row_idx)
    except Exception as e:
        st.error(f"Could not retrieve signal: {e}")
        return

    # Main signal box
    regime = signal_info["regime"]
    signal = signal_info["signal"]
    strategy = signal_info["strategy"]

    regime_col = REGIME_COLORS.get(regime, "#999999")
    signal_col = "#4CAF50" if signal == "LONG" else ("#F44336" if signal == "SHORT" else "#888888")

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div style='background:{regime_col};color:white;padding:12px;border-radius:8px;"
        f"text-align:center'><h2 style='margin:0'>REGIME<br/>{regime}</h2></div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div style='background:#333;color:white;padding:12px;border-radius:8px;"
        f"text-align:center'><h2 style='margin:0'>STRATEGY<br/>{strategy}</h2></div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<div style='background:{signal_col};color:white;padding:12px;border-radius:8px;"
        f"text-align:center'><h2 style='margin:0'>SIGNAL<br/>{signal}</h2></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Signal Rationale")
    for reason in signal_info["reasons"]:
        if reason.startswith("✓"):
            st.success(reason)
        elif reason.startswith("✗"):
            st.warning(reason)
        else:
            st.info(reason)

    # Indicator table
    st.markdown("---")
    st.subheader("Current Indicators")
    ind_data = {
        "EMA20":      f"{signal_info['ema_fast']:.2f}" if not np.isnan(signal_info['ema_fast']) else "N/A",
        "EMA50":      f"{signal_info['ema_slow']:.2f}" if not np.isnan(signal_info['ema_slow']) else "N/A",
        "RSI14":      f"{signal_info['rsi']:.1f}"      if not np.isnan(signal_info['rsi'])      else "N/A",
        "ATR14":      f"{signal_info['atr']:.2f}"      if not np.isnan(signal_info['atr'])      else "N/A",
        "20d RVOL":   f"{signal_info['rvol']:.1%}"     if not np.isnan(signal_info['rvol'])     else "N/A",
        "Z-Score":    f"{signal_info['zscore']:.2f}"   if not np.isnan(signal_info['zscore'])   else "N/A",
        "20d High":   f"{signal_info['roll_high']:.2f}" if not np.isnan(signal_info['roll_high']) else "N/A",
        "20d Low":    f"{signal_info['roll_low']:.2f}"  if not np.isnan(signal_info['roll_low'])  else "N/A",
    }
    cols = st.columns(4)
    for i, (k, v) in enumerate(ind_data.items()):
        cols[i % 4].metric(k, v)


# ─────────────────────────────────────────────
# Page 3: Backtest
# ─────────────────────────────────────────────

def page_backtest(data, backtest_results):
    st.title("📈 Backtest Results")
    st.caption("All backtests use T+1 execution (signal generated at close T, executed at T+1). OOS period: 2022–2026.")

    if not backtest_results:
        st.warning("Backtest results not found. Run `python run_research.py` first.")
        return

    # Selectors
    col1, col2, col3 = st.columns(3)
    mkt = col1.selectbox("Market", list(backtest_results.keys()),
                          format_func=lambda k: MARKETS[k]["name"])
    period = col2.selectbox("Period", ["Full", "Development (2012-2018)",
                                        "Validation (2019-2021)", "OOS (2022-2026)"])
    show_strats = col3.multiselect(
        "Strategies",
        ["Buy & Hold", "Static Trend", "Static Mean-Rev", "Regime-Adaptive"],
        default=["Buy & Hold", "Static Trend", "Regime-Adaptive"],
    )

    bt_df = backtest_results[mkt].copy()

    period_map = {
        "Full": (None, None),
        "Development (2012-2018)": ("2012-01-01", "2018-12-31"),
        "Validation (2019-2021)":  ("2019-01-01", "2021-12-31"),
        "OOS (2022-2026)":        ("2022-01-01", "2026-08-31"),
    }
    start, end = period_map[period]
    if start:
        bt_df = bt_df.loc[start:end]

    # Strategy column mapping
    strat_col_map = {
        "Buy & Hold":      "ret_buy_hold",
        "Static Trend":    "ret_trend",
        "Static Mean-Rev": "ret_mean_rev",
        "Regime-Adaptive": "ret_adaptive",
    }

    # Equity Curves
    st.subheader("Equity Curves (Growth of \$1)")
    fig_eq = go.Figure()
    for name in show_strats:
        col = strat_col_map.get(name)
        if col and col in bt_df.columns:
            ret = bt_df[col].dropna()
            equity = np.exp(ret.cumsum())
            fig_eq.add_trace(go.Scatter(
                x=equity.index, y=equity.values,
                name=name, line=dict(color=STRATEGY_COLORS.get(name), width=2),
            ))
    # Add OOS marker
    if start is None:
        fig_eq.add_vline(x=pd.Timestamp("2022-01-01"), line_dash="dash",
                          line_color="gray", annotation_text="OOS Start")
    fig_eq.update_layout(height=400, hovermode="x unified",
                          yaxis_title="Portfolio Value", xaxis_title="")
    st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown
    st.subheader("Drawdown")
    fig_dd = go.Figure()
    for name in show_strats:
        col = strat_col_map.get(name)
        if col and col in bt_df.columns:
            ret = bt_df[col].dropna()
            cum = np.exp(ret.cumsum())
            rolling_max = cum.expanding().max()
            dd = (cum - rolling_max) / rolling_max
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values * 100,
                name=name, fill="tozeroy",
                line=dict(color=STRATEGY_COLORS.get(name), width=1.5),
                opacity=0.6,
            ))
    fig_dd.update_layout(height=300, yaxis_title="Drawdown (%)",
                          hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)

    # Performance table
    st.subheader("Performance Metrics")
    try:
        metrics = compute_metrics(bt_df)
        display_cols = ["CAGR", "Ann_Vol", "Sharpe", "Sortino", "Max_Drawdown",
                         "Calmar", "Win_Rate", "Trades", "Exposure"]
        display = metrics[[c for c in display_cols if c in metrics.columns]].copy()
        fmt = {
            "CAGR": "{:.1%}", "Ann_Vol": "{:.1%}", "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}", "Max_Drawdown": "{:.1%}", "Calmar": "{:.2f}",
            "Win_Rate": "{:.1%}", "Exposure": "{:.1%}",
        }
        for col, f in fmt.items():
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f.format(x) if isinstance(x, float) else x)
        st.dataframe(display, use_container_width=True)
    except Exception as e:
        st.error(f"Could not compute metrics: {e}")


# ─────────────────────────────────────────────
# Page 4: Regime + Macro
# ─────────────────────────────────────────────

def page_regime_macro(data):
    st.title("🌐 Regime Analysis & Macro Events")

    if not data:
        st.error("No data available.")
        return

    mkt = st.selectbox("Market", list(data.keys()),
                        format_func=lambda k: MARKETS[k]["name"])
    df = data[mkt]

    tab1, tab2 = st.tabs(["Regime Analysis", "Macro Event Study"])

    with tab1:
        st.subheader("Regime Statistics")
        if "Regime" in df.columns:
            stats = regime_statistics(df)
            display_stats = stats[["Count", "Pct_Days", "Ann_Return", "Ann_Vol", "Sharpe"]].copy()
            display_stats["Pct_Days"]  = display_stats["Pct_Days"].map("{:.1%}".format)
            display_stats["Ann_Return"] = display_stats["Ann_Return"].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "N/A")
            display_stats["Ann_Vol"]   = display_stats["Ann_Vol"].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "N/A")
            display_stats["Sharpe"]    = display_stats["Sharpe"].map(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
            st.dataframe(display_stats, use_container_width=True)

            # Regime over time
            st.subheader("Regime Timeline (recent 3 years)")
            recent = df.tail(756)  # ~3 years
            if "Regime" in recent.columns:
                regime_num = recent["Regime"].map(
                    {"TRENDING": 1, "SIDEWAYS": 0, "HIGH_VOL": -1}
                ).fillna(0)
                fig_rt = go.Figure()
                for regime_name, val in [("TRENDING", 1), ("SIDEWAYS", 0), ("HIGH_VOL", -1)]:
                    mask = recent["Regime"] == regime_name
                    fig_rt.add_trace(go.Scatter(
                        x=recent.index[mask],
                        y=[val] * mask.sum(),
                        mode="markers",
                        name=regime_name,
                        marker=dict(color=REGIME_COLORS[regime_name], size=3),
                    ))
                fig_rt.update_layout(height=200, yaxis=dict(
                    tickvals=[-1, 0, 1], ticktext=["HIGH_VOL", "SIDEWAYS", "TRENDING"]
                ), hovermode="x")
                st.plotly_chart(fig_rt, use_container_width=True)

    with tab2:
        st.subheader("Macro Event Study")
        st.caption("""
        Analysing market returns in a ±1 day window around major macro events.
        **Hypothesis:** Major releases cause above-normal price volatility on the event day.
        """)

        event_type = st.selectbox(
            "Event Type",
            ["FOMC Rate Decisions", "CPI Releases", "EIA Inventory (CL only)"],
        )

        event_map = {
            "FOMC Rate Decisions": FOMC_DATES,
            "CPI Releases": CPI_DATES,
            "EIA Inventory (CL only)": EIA_DATES,
        }
        ev_dates = event_map[event_type]

        if event_type == "EIA Inventory (CL only)" and mkt != "CL":
            st.info("EIA inventory reports are most relevant for Crude Oil (CL). Switch market to CL for best results.")

        try:
            returns = df["Return"].dropna()
            window_df = event_window_returns(returns, ev_dates, window=1, label=event_type)
            if window_df.empty:
                st.warning("No events found within the data range.")
            else:
                summary = event_study_summary(window_df)
                st.markdown("**Average Market Behavior Around Event (window = ±1 trading day)**")

                # Bar chart
                fig_ev = go.Figure()
                fig_ev.add_trace(go.Bar(
                    x=["Day -1", "Day 0 (Event)", "Day +1"],
                    y=summary["mean_return"].values * 100,
                    name="Mean Return (%)",
                    marker_color=["#2196F3", "#FF5722", "#4CAF50"],
                ))
                fig_ev.add_trace(go.Scatter(
                    x=["Day -1", "Day 0 (Event)", "Day +1"],
                    y=summary["mean_abs_return"].values * 100,
                    name="Mean |Return| (%)",
                    line=dict(color="#9C27B0", dash="dot", width=2),
                    yaxis="y2",
                ))
                fig_ev.update_layout(
                    title=f"{event_type} — Event Window Returns ({mkt})",
                    height=350,
                    yaxis=dict(title="Mean Return (%)"),
                    yaxis2=dict(title="Mean |Return| (%)", overlaying="y", side="right"),
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_ev, use_container_width=True)

                # Summary table
                display_ev = summary.copy()
                for c in ["mean_return", "median_return", "std_return", "mean_abs_return", "pct_large_moves"]:
                    if c in display_ev.columns:
                        display_ev[c] = display_ev[c].map(lambda x: f"{x:.3%}" if not np.isnan(x) else "N/A")
                display_ev.index = ["Day -1", "Day 0 (Event)", "Day +1"]
                st.dataframe(display_ev, use_container_width=True)

                n_events = summary["n_events"].iloc[1] if len(summary) > 1 else 0
                st.caption(f"Based on {n_events} events in the dataset.")
        except Exception as e:
            st.error(f"Event study error: {e}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    data = load_market_data()
    backtest_results = load_backtest_data()

    page = build_sidebar()

    if page == "📊 Market Overview":
        page_market_overview(data)
    elif page == "🎯 Strategy":
        page_strategy(data)
    elif page == "📈 Backtest":
        page_backtest(data, backtest_results)
    elif page == "🌐 Regime & Macro":
        page_regime_macro(data)


if __name__ == "__main__":
    main()
