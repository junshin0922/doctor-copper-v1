# Doctor Copper: Macro Regime Indicator

A quantitative macro regime model that uses copper and global
macro signals to classify market states and forecast next-month
S&P 500 returns.

2026.03.02

---

## Acknowledgements
This project was developed with assistance from Claude (AI) for coding support, and debugging.

## Overview

Built a 3-state regime model (Risk-ON / Neutral / Risk-OFF) using
five macro signals:
- Copper price
- Copper/Gold ratio
- DXY (US Dollar Index)
- US 10-Year Treasury Yield
- WTI Crude Oil

Each signal is transformed into a 3-month smoothed z-score and
combined into a weighted composite risk score to classify the
current macro environment.

---

## Key Results

| Regime   | N  | Avg Next-Month Return | Median  | Hit Rate |
|----------|----|-----------------------|---------|----------|
| Risk-ON  | 61 | **+1.72%**            | +1.82%  | 68.85%   |
| Neutral  | 62 | +1.04%                | +1.53%  | 66.13%   |
| Risk-OFF | 62 | +0.37%                | +1.02%  | 59.68%   |

- **Risk-ON vs Risk-OFF spread: +1.35%/month** across 185 observations (2006–2026)
- Directional ordering holds perfectly: Risk-ON > Neutral > Risk-OFF

### Statistical Test (Risk-ON vs Risk-OFF)

```
T-statistic : 1.44
P-value     : 0.1535
Result      : Not statistically significant at the 5% level
```

The spread is economically meaningful but does not clear the 5%
significance threshold, primarily due to sample size (~61 obs per
regime). Walk-forward validation in V2 will address this.

---

## Methodology

### 1. Signal Construction

Five macro signals, each transformed via 3-month rolling mean → global z-score:

| Signal | Direction | Weight |
|--------|-----------|--------|
| Copper return | Risk-ON proxy | +0.35 |
| Copper/Gold ratio return | Growth vs safety | +0.25 |
| WTI crude return | Demand confirmation | +0.15 |
| DXY return | Dollar strength = risk-off | −0.15 |
| US 10Y yield change | Tightening conditions | −0.10 |

```
risk_score = 0.35×copper_z + 0.25×cg_ratio_z + 0.15×wti_z
           − 0.15×dxy_z − 0.10×us10y_z
```

Final score smoothed with a 2-month rolling mean.

### 2. Regime Classification

Regimes assigned via historical percentile thresholds:
- **Risk-ON**: score ≥ 67th percentile
- **Neutral**: 33rd–67th percentile
- **Risk-OFF**: score ≤ 33rd percentile

### 3. Predictive Signal Analysis

Lag correlation analysis (copper return vs next-month S&P 500):

| Lag | Correlation |
|-----|-------------|
| 1 month | +0.126 (strongest short-term) |
| 9 months | +0.112 |
| 12 months | −0.110 (reversal) |

Short-to-medium lags show positive predictive power;
12-month lag reverses — consistent with momentum-to-mean-reversion dynamics.

### 4. Direction Prediction (Logistic Regression)

Attempted binary direction forecasting (S&P up/down next month):

```
Test accuracy : 0.5818
Baseline      : 0.6364  (majority class)
Delta         : −0.055
```

Linear classification underperforms the majority-class baseline,
suggesting that raw directional prediction is not the model's
strength — regime-based return differentiation is more robust.

### 5. Validation

- Directional ordering verified across 185 monthly observations
- No random train/test split; time-based split used throughout

---

## Limitations

- **Weights and thresholds are in-sample** — designed on the full
  dataset, introducing look-ahead bias
- **p-value 0.15** — spread not significant at 5%; partly a sample
  size constraint (~61 obs per regime)
- **Single market** — validated on S&P 500 only; cross-market
  generalizability untested in V1

---

## Files

- `fetch-data.py` — data fetching script
- `01_eda.ipynb` — full analysis notebook
- `master_monthly.csv` — processed data

---

## Tech Stack

Python, Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn, yfinance

---

## Next Steps (V2)

V2 addresses V1's core limitations:

1. **Walk-forward validation** — parameters estimated on rolling
   train windows, tested on unseen periods; eliminates in-sample bias
2. **KOSPI extension** — applies the same signal to Korean equities
   to test cross-market portability of the copper regime framework
3. **V1 vs V2 comparison** — explicit OOS performance comparison
   to quantify the cost of in-sample optimization
