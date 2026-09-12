"""Unit tests for src/backtest.py."""

import pandas as pd
import pytest

from src.backtest import ReversalBacktester


def test_backtest_target_hit():
    """Verifies backtester simulates trade breakout and Target 1 hit."""
    dates = pd.date_range("2026-01-01", periods=8, freq="B")
    # Bars 0, 1, 2: Red candles (bearish streak)
    # Bar 3: Green reversal candle (High=100, Low=90, Risk=10 -> Target=110)
    # Bar 4: Triggers entry (High=102, Low=95)
    # Bar 5: Hits Target (High=112, Low=98)
    data = {
        "Open": [120.0, 115.0, 110.0, 92.0, 98.0, 105.0, 108.0, 106.0],
        "High": [122.0, 116.0, 111.0, 100.0, 102.0, 112.0, 110.0, 108.0],
        "Low": [114.0, 109.0, 105.0, 90.0, 95.0, 98.0, 102.0, 104.0],
        "Close": [115.0, 110.0, 106.0, 98.0, 101.0, 111.0, 106.0, 107.0],
        "HA_Bullish": [False, False, False, True, True, True, False, True],
        "HA_Bearish": [True, True, True, False, False, False, True, False],
    }
    df = pd.DataFrame(data, index=dates)

    backtester = ReversalBacktester(streak_min=3, target_multiple=1.0, max_holding_bars=5)
    trades = backtester.backtest_series(df, symbol="TEST_LONG", direction="long")

    assert not trades.empty
    assert len(trades) == 1
    trade = trades.iloc[0]

    assert trade["Symbol"] == "TEST_LONG"
    assert trade["Direction"] == "Long"
    assert trade["Entry Price"] == 100.0
    assert trade["SL Price"] == 90.0
    assert trade["Target Price"] == 110.0
    assert trade["Outcome"] == "TARGET_HIT"
    assert trade["PnL (R)"] == 1.0


def test_calculate_metrics():
    """Verifies statistical metric calculations."""
    trade_log = pd.DataFrame([
        {"PnL (R)": 1.0, "Holding Bars": 3},
        {"PnL (R)": 1.0, "Holding Bars": 2},
        {"PnL (R)": -1.0, "Holding Bars": 4},
    ])

    metrics = ReversalBacktester.calculate_metrics(trade_log)

    assert metrics["Total Trades"] == 3
    assert metrics["Winning Trades"] == 2
    assert metrics["Losing Trades"] == 1
    assert metrics["Win Rate %"] == pytest.approx(66.67, 0.01)
    assert metrics["Profit Factor"] == pytest.approx(2.0)  # 2.0 gross profit / 1.0 gross loss
    assert metrics["Expectancy (R)"] == pytest.approx(0.33, 0.01)  # (1 + 1 - 1) / 3

