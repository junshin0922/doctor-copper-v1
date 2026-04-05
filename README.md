# Doctor Copper: Macro Regime Indicator

A quantitative macro regime model that uses copper and global 
macro signals to classify market states and forecast next-month 
S&P 500 returns.

## Overview

Built a 3-state regime model (Risk-ON / Neutral / Risk-OFF) using 
five macro signals:
- Copper price
- Copper/Gold ratio
- DXY (US Dollar Index)
- US 10-Year Treasury Yield
- WTI Crude Oil

Each signal is transformed into a smoothed z-score and combined 
into a weighted composite risk score to classify the current 
macro environment.

## Key Results

- Risk-ON months: **+1.72% avg next-month S&P 500 return**
- Risk-OFF months: **+0.37% avg next-month S&P 500 return**
- Spread: **+1.35%** across 186 monthly observations (2006–2026)
- Hit rates: Risk-ON 68.85% | Neutral 66.13% | Risk-OFF 59.68%

## Methodology

1. Fetch monthly macro data via `yfinance`
2. Engineer smoothed z-score features for each signal
3. Construct weighted composite risk score
4. Classify into regime states using percentile thresholds
5. Validate directional ordering across 186 observations

## Files

- `fetch-data.py` — data fetching script
- `01_eda.ipynb` — full analysis notebook

## Tech Stack

Python, Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn, yfinance

## Next Steps (V2)

V2 implements walk-forward validation to eliminate look-ahead bias 
and extends the analysis to KOSPI to test cross-market signal 
portability.
