"""Quantitative Signal Scoring and Ranking Module.

Computes a multi-factor Quality Score (0 to 100) for trading setups:
- Close Position (35%): Measures buyer/seller dominance into bar close
- Candle Move % (30%): Momentum magnitude of the reversal bar
- Exhaustion Streak (20%): Length of prior consecutive counter-trend streak
- Risk Tightness (15%): Normalizes stop loss percentage buffer
"""

import numpy as np
import pandas as pd


def score_and_rank_signals(df: pd.DataFrame, is_bullish: bool = True) -> pd.DataFrame:
    """Scores and ranks setups from best (Rank 1) to worst based on multi-factor heuristic.

    Args:
        df: DataFrame of identified signals.
        is_bullish: True for Long signals, False for Short breakdowns.

    Returns:
        Sorted DataFrame with Score (0-100), Move %, Close Pos %, Risk %, and Rank (1..N).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    close_col = "Actual Close" if "Actual Close" in out.columns else "Close"
    open_col = "Actual Open" if "Actual Open" in out.columns else "Open"
    high_col = "Actual High" if "Actual High" in out.columns else "High"
    low_col = "Actual Low" if "Actual Low" in out.columns else "Low"
    streak_col = "Streak Count" if "Streak Count" in out.columns else ("Bearish Months" if "Bearish Months" in out.columns else "Bearish Days")

    candle_range = out[high_col] - out[low_col]

    if is_bullish:
        # Move %: (Close - Open) / Open * 100
        out["Move %"] = ((out[close_col] - out[open_col]) / out[open_col] * 100.0).round(2)
        # Close Pos %: (Close - Low) / (High - Low) * 100 (100% means closed at high)
        out["Close Pos %"] = np.where(
            candle_range > 0,
            ((out[close_col] - out[low_col]) / candle_range * 100.0).round(2),
            100.0,
        )
    else:
        # Downside Move %: (Open - Close) / Open * 100
        out["Move %"] = ((out[open_col] - out[close_col]) / out[open_col] * 100.0).round(2)
        # Downside Close Pos %: (High - Close) / (High - Low) * 100 (100% means closed at bottom)
        out["Close Pos %"] = np.where(
            candle_range > 0,
            ((out[high_col] - out[close_col]) / candle_range * 100.0).round(2),
            100.0,
        )

    # Risk %: Risk / Entry * 100
    if "Risk" in out.columns and "Entry" in out.columns:
        out["Risk %"] = ((out["Risk"] / out["Entry"]) * 100.0).round(2)
    elif "SL" in out.columns and "Entry" in out.columns:
        out["Risk %"] = (np.abs(out["Entry"] - out["SL"]) / out["Entry"] * 100.0).round(2)
    else:
        out["Risk %"] = 5.0

    # 1. Close Position Score (0 to 35 pts)
    s_close = (np.clip(out["Close Pos %"], 0.0, 100.0) / 100.0) * 35.0

    # 2. Move / Momentum Score (0 to 30 pts)
    max_move = out["Move %"].max()
    if pd.notna(max_move) and max_move > 0:
        s_move = np.clip(out["Move %"] / max_move, 0.0, 1.0) * 30.0
    else:
        s_move = 15.0

    # 3. Exhaustion Streak Score (0 to 20 pts): 3 count = 10 pts, 6+ count = 20 pts
    if streak_col in out.columns:
        s_streak = np.clip((out[streak_col] / 6.0) * 20.0, 10.0, 20.0)
    else:
        s_streak = 10.0

    # 4. Risk Profile Score (0 to 15 pts): Rewards manageable risk (< 15%)
    max_risk = out["Risk %"].max()
    if pd.notna(max_risk) and max_risk > 0:
        s_risk = np.clip((1.0 - (out["Risk %"] / (max_risk * 1.5))) * 15.0, 3.0, 15.0)
    else:
        s_risk = 10.0

    # Total Score (0 to 100)
    out["Score"] = (s_close + s_move + s_streak + s_risk).round(1)

    # Sort descending by Score, then Move %
    out = out.sort_values(by=["Score", "Move %"], ascending=[False, False]).reset_index(drop=True)
    if "Rank" in out.columns:
        out = out.drop(columns=["Rank"])
    out.insert(0, "Rank", range(1, len(out) + 1))

    return out

