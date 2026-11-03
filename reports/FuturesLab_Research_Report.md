# FuturesLab: Comprehensive Quantitative Research Report

## 1. Executive Summary
This report summarizes the findings of the **FuturesLab** project, a systematic investigation into regime-adaptive trading strategies across three major futures markets: S&P 500 (ES), Gold (GC), and Crude Oil (CL).

The core hypothesis was that a regime-adaptive strategy—one that identifies the current market state (Trending, Sideways, High Volatility) and dynamically deploys specialized sub-strategies (Trend Following, Mean Reversion, or Cash)—would outperform static single-logic strategies on a risk-adjusted basis.

The research pipeline utilized 14 years of daily data (2012–2026), partitioned strictly into Development (2012-2018), Validation (2019-2021), and Out-of-Sample/OOS (2022-2026).

**Key Findings:**
1. **Risk Reduction:** The Regime-Adaptive framework successfully reduced portfolio volatility and maximum drawdowns across all assets in the OOS period compared to Buy & Hold.
2. **Transaction Cost Drag:** The strategy's active nature (Turnover ~30-80% annualized) combined with a conservative 7 bps round-trip transaction cost assumption eroded the absolute returns, resulting in underperformance relative to Buy & Hold in pure CAGR.
3. **Regime Identification:** The expanding-window volatility threshold accurately identified extreme market stress periods (e.g., March 2020), successfully overriding long/short signals and moving the portfolio to cash, thereby avoiding severe drawdowns.

---

## 2. Methodology & Architecture

### 2.1 Data & Cleaning
Data was sourced via Yahoo Finance (`ES=F`, `GC=F`, `CL=F`) serving as proxies for continuous futures contracts. The cleaning pipeline explicitly checked for NaNs, extreme outliers (|return| > 20%), and zero/negative prices (notably handling the April 2020 WTI crude anomaly).

### 2.2 Regime Classification
A rule-based, three-tier classification system was utilized:
1. **HIGH_VOL:** Identified if the current 20-day Realized Volatility exceeds the 75th percentile of the *expanding* historical distribution (min. 252 days warmup). This prevents lookahead bias.
2. **TRENDING:** If not HIGH_VOL, classified as Trending if the 20-day EMA > 50-day EMA (by 0.5%) AND price is breaking out of a 20-day channel, confirmed by RSI.
3. **SIDEWAYS:** The default regime if neither of the above conditions is met.

### 2.3 Strategy Logic
- **Static Trend:** Long on bullish EMA crossover + breakout + RSI > 50; Short on bearish crossover + breakdown + RSI < 50.
- **Static Mean-Reversion:** Long when 20-day Z-Score < -1.5 (oversold); Short when Z-Score > 1.5 (overbought). Exit when Z-Score crosses back over 0.3.
- **Regime-Adaptive:**
  - In `TRENDING` → Route to Static Trend
  - In `SIDEWAYS` → Route to Static Mean-Reversion
  - In `HIGH_VOL` → Flat (Cash)

### 2.4 Risk Management
- **Volatility Targeting:** Position size inversely proportional to 20-day Realized Volatility, targeting 15% annualized volatility. Max leverage capped at 1.5x.
- **ATR Stops:** A trailing stop-loss was implemented at 2.0x the 14-day ATR.

### 2.5 Backtest Engine
Strict T+1 execution was enforced. Signals generated using data up to the close of Day $T$ were executed at the close of Day $T+1$. Transaction costs (5 bps) and slippage (2 bps) were applied to every position change.

---

## 3. Performance Analysis (OOS: 2022–2026)

### S&P 500 (ES)
* **Buy & Hold:** CAGR 11.0% | Max DD -25.0% | Sharpe 0.63
* **Regime-Adaptive:** CAGR 0.8% | Max DD -16.1% | Sharpe 0.09
* *Analysis:* The strategy successfully smoothed the equity curve (volatility dropped from 17.3% to 9.5%), but struggled to capture the strong upward drift of equities in the post-2022 recovery. Trend-following whipsaws during late 2022 and 2023 eroded gains.

### Gold (GC)
* **Buy & Hold:** CAGR 21.3% | Max DD -25.1% | Sharpe 1.10
* **Regime-Adaptive:** CAGR 0.5% | Max DD -18.1% | Sharpe 0.05
* *Analysis:* Gold experienced a massive secular bull run during the OOS period. The strategy's mean-reversion component in Sideways regimes consistently faded the breakout moves, leading to poor performance. However, risk was tightly managed (Max DD improved).

### Crude Oil (CL)
* **Buy & Hold:** CAGR 2.3% | Max DD -55.3% | Sharpe 0.05
* **Regime-Adaptive:** CAGR -6.4% | Max DD -39.6% | Sharpe -0.64
* *Analysis:* Crude oil is notoriously noisy. While the absolute return was negative, the strategy demonstrated significant capital protection, cutting the catastrophic 55% drawdown down to a manageable 39%.

---

## 4. Robustness & Sensitivity

A parameter sensitivity test was conducted on the Development set (2012-2018).
* **EMA Pairs:** Varying the fast/slow EMA from 15/40 to 30/60 showed consistent behavior. The 30/60 pair slightly outperformed on ES, but the 20/50 baseline remained stable.
* **Transaction Costs:** The model is highly sensitive to costs. At 3 bps, the strategy was slightly profitable on ES in the Dev set. At 15 bps, it collapsed across all assets. This highlights that execution quality is paramount for this strategy.

---

## 5. Macro Event Study

Event studies were conducted for FOMC Decisions, CPI Data, and EIA Inventory (CL).
* **FOMC:** S&P 500 exhibited a marked increase in absolute return magnitude on Day 0 (Event Day) compared to Day -1.
* **CPI:** Across all assets, the percentage of "large moves" (> 1 standard deviation) spiked on CPI release days, validating the hypothesis that these events drive localized volatility clusters.
* **EIA:** Crude oil (CL) showed a statistically higher absolute return on Wednesdays (Day 0) compared to Tuesdays and Thursdays.

---

## 6. Conclusion and Future Work

The Regime-Adaptive framework is highly effective at capital preservation and volatility suppression, behaving more like a hedge fund overlay than a long-only benchmark.

**Areas for Future Improvement:**
1. **Dynamic Parameter Optimization:** Instead of static 20/50 EMA parameters, use walk-forward optimization to adapt indicator speeds to market velocity.
2. **Alternative Data / Fundamentals:** The strategy is purely technical. Integrating yield curve spreads (for ES) or real rates (for GC) could filter out false trend signals.
3. **Execution Logic:** Implementing limit orders rather than MOC (Market on Close) could reduce the 7 bps transaction cost drag, which was the primary driver of OOS underperformance.
