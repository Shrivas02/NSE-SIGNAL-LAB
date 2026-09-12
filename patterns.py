"""Candlestick and Heikin-Ashi Pattern Detection Module.

Identifies reversal structures:
- Bullish Reversal (3+ consecutive Bearish candles -> 1 Bullish reversal candle)
- Bearish Reversal (3+ consecutive Bullish candles -> 1 Bearish reversal candle)
Computes dynamic trade order parameters (Entry, Stop Loss, 1R / 2R Targets, and Lot Risk).
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd


def count_prior_streak(
    frame: pd.DataFrame,
    pos: int,
    streak_type: str = "bearish",
    col: Optional[str] = None,
) -> Tuple[int, int]:
    """Counts consecutive prior candles of the specified type preceding position pos.

    Args:
        frame: DataFrame with boolean candle flags.
        pos: Index position of the current reversal candle.
        streak_type: 'bearish' or 'bullish'.
        col: Optional column name. Defaults to 'HA_Bearish' / 'HA_Bullish' if present,
             else falls back to 'Bearish' / 'Bullish'.

    Returns:
        Tuple of (streak_count, last_evaluated_index_j).
    """
    if pos <= 0 or frame.empty or pos >= len(frame):
        return 0, pos - 1

    if col is None:
        if "HA_Bearish" in frame.columns and "HA_Bullish" in frame.columns:
            flag_col = "HA_Bearish" if streak_type.lower() == "bearish" else "HA_Bullish"
        else:
            flag_col = "Bearish" if streak_type.lower() == "bearish" else "Bullish"
    else:
        flag_col = col

    if flag_col not in frame.columns:
        raise KeyError(f"Column '{flag_col}' not found in frame columns: {list(frame.columns)}")

    count = 0
    j = pos - 1

    # Loop backward while candles match the streak condition
    while j >= 0:
        val = frame[flag_col].iloc[j]
        if bool(val) and not pd.isna(val):
            count += 1
            j -= 1
        else:
            break

    return count, j


def compute_trade_levels(
    signal_candle: pd.Series,
    is_bullish: bool = True,
    market_lot: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes exact execution trade levels for long or short reversal setups.

    Long Setup:
        - Entry: High of reversal candle (breakout above high)
        - Stop Loss (SL): Low of reversal candle
        - Risk (1R): Entry - SL
        - Target 1: Entry + 1R
        - Target 2: Entry + 2R

    Short Setup:
        - Entry: Low of reversal candle (breakdown below low)
        - Stop Loss (SL): High of reversal candle
        - Risk (1R): SL - Entry
        - Target 1: Entry - 1R
        - Target 2: Entry - 2R

    Args:
        signal_candle: Series with 'High' and 'Low'.
        is_bullish: True for Long reversal, False for Short breakdown.
        market_lot: Optional F&O contract lot size.

    Returns:
        Dictionary of trade levels.
    """
    high = float(signal_candle["High"])
    low = float(signal_candle["Low"])
    lot_val = float(market_lot) if market_lot is not None and pd.notna(market_lot) else None

    if is_bullish:
        entry = round(high, 2)
        sl = round(low, 2)
        risk = round(entry - sl, 2)
        if risk <= 0:
            return {
                "Entry": entry, "SL": sl, "Risk": 0.0,
                "Target 1": np.nan, "Target 2": np.nan,
                "Risk/Lot (INR)": np.nan,
            }
        t1 = round(entry + risk, 2)
        t2 = round(entry + 2 * risk, 2)
        risk_lot = round(risk * lot_val, 2) if lot_val else np.nan
        return {
            "Entry": entry,
            "SL": sl,
            "Risk": risk,
            "Target 1": t1,
            "Target 2": t2,
            "Risk/Lot (INR)": risk_lot,
        }
    else:
        entry = round(low, 2)
        sl = round(high, 2)
        risk = round(sl - entry, 2)
        if risk <= 0:
            return {
                "Entry": entry, "SL": sl, "Risk": 0.0,
                "Target 1": np.nan, "Target 2": np.nan,
                "Risk/Lot (INR)": np.nan,
            }
        t1 = round(entry - risk, 2)
        t2 = round(entry - 2 * risk, 2)
        risk_lot = round(risk * lot_val, 2) if lot_val else np.nan
        return {
            "Entry": entry,
            "SL": sl,
            "Risk": risk,
            "Target 1": t1,
            "Target 2": t2,
            "Risk/Lot (INR)": risk_lot,
        }


def detect_reversal_pattern(
    frame: pd.DataFrame,
    pos: int,
    streak_min: int = 3,
    market_lot: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Detects both Bullish and Bearish reversal patterns at the specified position.

    Args:
        frame: DataFrame containing OHLC and HA candle flags.
        pos: Target position index to evaluate.
        streak_min: Minimum number of consecutive prior opposite candles (default 3).
        market_lot: Optional F&O lot size.

    Returns:
        Tuple of (bullish_signal_dict, bearish_signal_dict).
    """
    if pos < streak_min or pos >= len(frame):
        return None, None

    candle = frame.iloc[pos]
    date_val = frame.index[pos]

    bullish_flag = candle.get("HA_Bullish", candle.get("Bullish", False))
    bearish_flag = candle.get("HA_Bearish", candle.get("Bearish", False))

    bullish_sig = None
    bearish_sig = None

    # 1. Bullish Setup: Current candle is Bullish, preceded by >= streak_min Bearish candles
    if bool(bullish_flag):
        streak_count, j = count_prior_streak(frame, pos, streak_type="bearish")
        if streak_count >= streak_min:
            first_idx = frame.index[j + 1]
            trade = compute_trade_levels(candle, is_bullish=True, market_lot=market_lot)
            bullish_sig = {
                "Signal Type": "Bullish Reversal (Long)",
                "Signal Date": date_val,
                "Streak Count": streak_count,
                "Streak Type": "Bearish",
                "First Streak Date": first_idx,
                "Open": float(candle["Open"]),
                "High": float(candle["High"]),
                "Low": float(candle["Low"]),
                "Close": float(candle["Close"]),
                **trade,
            }
            if "HA_Close" in candle:
                bullish_sig.update({
                    "HA Open": float(candle["HA_Open"]),
                    "HA High": float(candle["HA_High"]),
                    "HA Low": float(candle["HA_Low"]),
                    "HA Close": float(candle["HA_Close"]),
                })

    # 2. Bearish Setup: Current candle is Bearish, preceded by >= streak_min Bullish candles
    if bool(bearish_flag):
        streak_count, j = count_prior_streak(frame, pos, streak_type="bullish")
        if streak_count >= streak_min:
            first_idx = frame.index[j + 1]
            trade = compute_trade_levels(candle, is_bullish=False, market_lot=market_lot)
            bearish_sig = {
                "Signal Type": "Bearish Reversal (Short)",
                "Signal Date": date_val,
                "Streak Count": streak_count,
                "Streak Type": "Bullish",
                "First Streak Date": first_idx,
                "Open": float(candle["Open"]),
                "High": float(candle["High"]),
                "Low": float(candle["Low"]),
                "Close": float(candle["Close"]),
                **trade,
            }
            if "HA_Close" in candle:
                bearish_sig.update({
                    "HA Open": float(candle["HA_Open"]),
                    "HA High": float(candle["HA_High"]),
                    "HA Low": float(candle["HA_Low"]),
                    "HA Close": float(candle["HA_Close"]),
                })

    return bullish_sig, bearish_sig

