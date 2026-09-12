"""Unit tests for src/patterns.py."""

import pandas as pd
import pytest

from src.patterns import (
    count_prior_streak,
    compute_trade_levels,
    detect_reversal_pattern,
)


@pytest.fixture
def reversal_pattern_df():
    """Generates a frame with 4 red candles followed by 1 green reversal candle."""
    dates = pd.date_range("2026-08-01", periods=5, freq="B")
    data = {
        "Open": [100.0, 98.0, 96.0, 94.0, 92.0],
        "High": [101.0, 99.0, 97.0, 95.0, 97.0],
        "Low": [97.0, 95.0, 93.0, 91.0, 91.5],
        "Close": [97.5, 95.5, 93.5, 91.5, 96.5],
        "HA_Bullish": [False, False, False, False, True],
        "HA_Bearish": [True, True, True, True, False],
    }
    return pd.DataFrame(data, index=dates)


def test_count_prior_streak(reversal_pattern_df):
    """Verifies that count_prior_streak counts exact consecutive prior red bars."""
    count, j = count_prior_streak(reversal_pattern_df, pos=4, streak_type="bearish")
    assert count == 4
    assert j == -1

    # At pos=3 (4th candle), prior bearish count should be 3
    count_3, _ = count_prior_streak(reversal_pattern_df, pos=3, streak_type="bearish")
    assert count_3 == 3

    # Checking bullish streak on red candles should yield 0
    count_bull, _ = count_prior_streak(reversal_pattern_df, pos=4, streak_type="bullish")
    assert count_bull == 0


def test_compute_trade_levels_long():
    """Verifies Long trade level formulas."""
    candle = pd.Series({"High": 500.0, "Low": 470.0})
    levels = compute_trade_levels(candle, is_bullish=True, market_lot=100)

    assert levels["Entry"] == 500.0
    assert levels["SL"] == 470.0
    assert levels["Risk"] == 30.0
    assert levels["Target 1"] == 530.0  # 500 + 30
    assert levels["Target 2"] == 560.0  # 500 + 60
    assert levels["Risk/Lot (INR)"] == 3000.0  # 30 * 100


def test_compute_trade_levels_short():
    """Verifies Short trade level formulas."""
    candle = pd.Series({"High": 500.0, "Low": 470.0})
    levels = compute_trade_levels(candle, is_bullish=False, market_lot=200)

    assert levels["Entry"] == 470.0
    assert levels["SL"] == 500.0
    assert levels["Risk"] == 30.0
    assert levels["Target 1"] == 440.0  # 470 - 30
    assert levels["Target 2"] == 410.0  # 470 - 60
    assert levels["Risk/Lot (INR)"] == 6000.0  # 30 * 200


def test_detect_reversal_pattern(reversal_pattern_df):
    """Verifies Bullish reversal detection on qualifying 4 red -> 1 green setup."""
    bull_sig, bear_sig = detect_reversal_pattern(reversal_pattern_df, pos=4, streak_min=3)

    assert bull_sig is not None
    assert bear_sig is None
    assert bull_sig["Signal Type"] == "Bullish Reversal (Long)"
    assert bull_sig["Streak Count"] == 4
    assert bull_sig["Entry"] == 97.0
    assert bull_sig["SL"] == 91.5

