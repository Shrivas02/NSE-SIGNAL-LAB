"""Unit tests for src/indicators.py."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    compute_heikin_ashi,
    resample_ohlc,
    compute_ichimoku,
    compute_relative_strength,
)


@pytest.fixture
def sample_ohlc_df():
    """Generates a synthetic 10-day OHLC DataFrame."""
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    data = {
        "Open": [100.0, 102.0, 101.0, 99.0, 97.0, 95.0, 94.0, 96.0, 98.0, 100.0],
        "High": [105.0, 104.0, 103.0, 100.0, 98.0, 97.0, 98.0, 101.0, 102.0, 105.0],
        "Low": [98.0, 100.0, 98.0, 96.0, 94.0, 93.0, 92.0, 95.0, 97.0, 99.0],
        "Close": [102.0, 101.0, 99.0, 97.0, 95.0, 94.0, 96.0, 98.0, 100.0, 103.0],
        "Volume": [1000, 1100, 1200, 900, 950, 1050, 1300, 1400, 1500, 1600],
    }
    return pd.DataFrame(data, index=dates)


def test_heikin_ashi_calculation(sample_ohlc_df):
    """Verifies that Heikin-Ashi formula correctly computes HA OHLC values."""
    ha = compute_heikin_ashi(sample_ohlc_df)

    assert not ha.empty
    assert len(ha) == len(sample_ohlc_df)
    assert set(["HA_Open", "HA_High", "HA_Low", "HA_Close", "HA_Bullish", "HA_Bearish"]).issubset(ha.columns)

    # First bar HA_Close = (100 + 105 + 98 + 102) / 4 = 101.25
    assert ha["HA_Close"].iloc[0] == pytest.approx(101.25)

    # First bar HA_Open = (100 + 102) / 2 = 101.0
    assert ha["HA_Open"].iloc[0] == pytest.approx(101.0)

    # Second bar HA_Open = (HA_Open[0] + HA_Close[0]) / 2 = (101.0 + 101.25) / 2 = 101.125
    assert ha["HA_Open"].iloc[1] == pytest.approx(101.125)

    # HA_High should always be >= max(HA_Open, HA_Close)
    assert (ha["HA_High"] >= ha["HA_Open"]).all()
    assert (ha["HA_High"] >= ha["HA_Close"]).all()

    # HA_Low should always be <= min(HA_Open, HA_Close)
    assert (ha["HA_Low"] <= ha["HA_Open"]).all()
    assert (ha["HA_Low"] <= ha["HA_Close"]).all()


def test_heikin_ashi_empty_and_missing_columns():
    """Verifies safe error handling for empty dataframes or missing columns."""
    assert compute_heikin_ashi(pd.DataFrame()).empty
    assert compute_heikin_ashi(None).empty

    incomplete_df = pd.DataFrame({"Open": [100.0], "Close": [105.0]})
    with pytest.raises(ValueError, match="missing required OHLC columns"):
        compute_heikin_ashi(incomplete_df)


def test_resample_ohlc_weekly(sample_ohlc_df):
    """Verifies weekly resampling aggregates properly."""
    resampled = resample_ohlc(sample_ohlc_df, timeframe="weekly")

    assert not resampled.empty
    assert "Open" in resampled.columns
    assert "High" in resampled.columns
    assert "Low" in resampled.columns
    assert "Close" in resampled.columns


def test_compute_ichimoku(sample_ohlc_df):
    """Verifies Ichimoku lines are computed."""
    # Extend series to 60 days so 52-period Senkou B has enough points
    dates = pd.date_range("2026-01-01", periods=65, freq="B")
    long_df = pd.DataFrame({
        "Open": np.linspace(100, 150, 65),
        "High": np.linspace(105, 155, 65),
        "Low": np.linspace(95, 145, 65),
        "Close": np.linspace(102, 152, 65),
    }, index=dates)

    ichi = compute_ichimoku(long_df)
    assert "Tenkan_Sen" in ichi.columns
    assert "Kijun_Sen" in ichi.columns
    assert "Senkou_Span_A" in ichi.columns
    assert "Senkou_Span_B" in ichi.columns
    assert "Kumo_Top" in ichi.columns
    assert "Kumo_Bottom" in ichi.columns

