# NSE Signal Lab

An end-to-end quantitative equity & derivatives screener, backtesting engine, and analytics dashboard covering **NSE 500**, **NIFTY 50**, and **210+ NSE Futures & Options (F&O)** constituents.

---

## Architecture Overview

```
nse 500/
├── src/                          # Modular core quantitative engine
│   ├── indicators.py             # Heikin-Ashi, Ichimoku Kinko Hyo, Relative Strength
│   ├── patterns.py               # Reversal streak detection, order levels (1R/2R)
│   ├── scoring.py                # Multi-factor 0-100 quality scoring & ranking
│   ├── backtest.py               # Event-driven walk-forward backtesting simulator
│   └── data.py                   # Constituents parser & batch market data fetching
├── tests/                        # Comprehensive unit test suite (pytest)
│   ├── test_indicators.py        # Validates HA formulas, resampling, Ichimoku
│   ├── test_patterns.py          # Validates streak counts & trade level math
│   ├── test_scoring.py           # Validates score bounds & monotonicity
│   └── test_backtest.py          # Validates simulated executions & metrics
├── dashboard/
│   └── app.py                    # Streamlit & Plotly interactive web application
├── run_backtest.py               # CLI strategy backtester & performance tear sheet
├── HAweekly.py                   # Nifty 50 weekly scanner
├── HAmonthly.py                  # NSE 500 monthly scanner
├── futures_weekly.py             # F&O weekly scanner with lot sizes & INR risk
├── futures_monthly.py            # F&O monthly scanner with lot sizes & INR risk
└── requirements.txt              # Dependencies
```

---

## 1. Quick Start

### Install Dependencies:
```powershell
python -m pip install -r requirements.txt
```

### Run Unit Tests (`pytest`):
```powershell
pytest tests/ -v
```
*(Runs 12 unit tests verifying indicator formulas, streak counting, risk math, and backtesting metrics).*

---

## 2. Interactive Web Dashboard

Launch the Streamlit analytics lab:
```powershell
streamlit run dashboard/app.py
```

- **Multi-Strategy Scans**: Run Heikin-Ashi reversals, Ichimoku Cloud breakouts, and Mansfield Relative Strength (RS) scans.
- **Visual Analytics**: Interactive Plotly candlestick charts with Entry, Stop Loss, and 1R/2R Target lines.
- **Built-in Strategy Backtester**: Test historical Win Rate %, Profit Factor, Expectancy, and Equity Curves directly from the dashboard expander.

---

## 3. Quantitative Backtester (CLI)

Run walk-forward event-driven simulations over historical bars:

```powershell
# Backtest NIFTY 50 weekly reversals over 4 years with 1R target:
python run_backtest.py --universe nifty50 --timeframe weekly --target 1.0

# Backtest NSE Futures with 2R target:
python run_backtest.py --universe futures --timeframe weekly --target 2.0

# Backtest a single stock (e.g. Reliance):
python run_backtest.py --symbol RELIANCE.NS --timeframe weekly
```

### Metrics Reported:
- **Total Trades**, **Win Rate %**, **Loss Rate %**
- **Profit Factor** ($\text{Gross Win} / \text{Gross Loss}$)
- **Expectancy** ($R$ per trade)
- **Cumulative $R$ curve & Max Drawdown ($R$)**
- **Detailed Trade Log CSV export**

---

## 4. Standalone Scanner Scripts

```powershell
# NIFTY 50 Weekly:
python HAweekly.py --date 2026-08-28

# NSE 500 Monthly:
python HAmonthly.py --month 2026-08

# NSE Futures (F&O) Weekly:
python futures_weekly.py --date 2026-08-28

# NSE Futures (F&O) Monthly:
python futures_monthly.py --month 2026-08
```
