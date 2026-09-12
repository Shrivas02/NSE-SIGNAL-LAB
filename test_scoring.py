"""Unit tests for src/scoring.py."""

import pandas as pd
import pytest

from src.scoring import score_and_rank_signals


def test_score_and_rank_signals_bounds_and_ranking():
    """Verifies that scores are bounded in [0, 100] and ranks are monotonic."""
    df = pd.DataFrame({
        "Stock": ["STOCK_A", "STOCK_B", "STOCK_C"],
        "Open": [100.0, 200.0, 50.0],
        "High": [110.0, 205.0, 52.0],
        "Low": [98.0, 195.0, 48.0],
        "Close": [110.0, 201.0, 49.0],  # A closed at high (+10%), B closed small gain, C closed weak
        "Streak Count": [5, 3, 4],
        "Entry": [110.0, 205.0, 52.0],
        "SL": [98.0, 195.0, 48.0],
        "Risk": [12.0, 10.0, 4.0],
    })

    ranked = score_and_rank_signals(df, is_bullish=True)

    assert not ranked.empty
    assert len(ranked) == 3
    assert "Score" in ranked.columns
    assert "Rank" in ranked.columns

    # Verify score bounds
    assert (ranked["Score"] >= 0.0).all()
    assert (ranked["Score"] <= 100.0).all()

    # Verify ranking monotonicity
    assert list(ranked["Rank"]) == [1, 2, 3]
    assert ranked["Score"].iloc[0] >= ranked["Score"].iloc[1]
    assert ranked["Score"].iloc[1] >= ranked["Score"].iloc[2]

    # Stock A (closed at absolute high with 10% gain) should be Rank 1
    assert ranked["Stock"].iloc[0] == "STOCK_A"


def test_score_and_rank_empty():
    """Verifies safe empty dataframe handling."""
    assert score_and_rank_signals(pd.DataFrame()).empty
    assert score_and_rank_signals(None).empty

