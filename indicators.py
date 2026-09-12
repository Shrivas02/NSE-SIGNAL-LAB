"""Technical and Quantitative Indicators Module.

Provides mathematical algorithms for:
- Heikin-Ashi candle generation (recursive smoothing)
- Multi-timeframe OHLC resampling
- Ichimoku Kinko Hyo equilibrium cloud calculations
- Mansfield / Index Relative Strength (RS)
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Heikin-Ashi smoothed candles from a standard OHLC dataframe.

    Formulas:
        HA_Close = (Open + High + Low + Close) / 4
        HA_Open[0] = (Open[0] + Close[0]) / 2
        HA_Open[i] = (HA_Open[i-1] + HA_Close[i-1]) / 2
        HA_High = max(High, HA_Open, HA_Close)
        HA_Low = min(Low, HA_Open, HA_Close)

    Args:
        df: DataFrame containing 'Open', 'High', 'Low', 'Close' columns.

    Returns:
        DataFrame with original OHLC plus 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close',
        'HA_Bullish', and 'HA_Bearish'.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"DataFrame missing required OHLC columns: {missing}")

    frame = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(frame) == 0:
        return pd.DataFrame()

    ha = pd.DataFrame(index=frame.index)
    ha_close = (frame["Open"] + frame["High"] + frame["Low"] + frame["Close"]) / 4.0

    n = len(frame)
    ha_open = np.zeros(n, dtype=float)
    ha_open[0] = (frame["Open"].iloc[0] + frame["Close"].iloc[0]) / 2.0

    # Recursive computation of HA_Open
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha["HA_Open"] = ha_open
    ha["HA_Close"] = ha_close.values
    ha["HA_High"] = np.maximum.reduce([frame["High"].values, ha_open, ha_close.values])
    ha["HA_Low"] = np.minimum.reduce([frame["Low"].values, ha_open, ha_close.values])

    ha["HA_Bullish"] = ha["HA_Close"] > ha["HA_Open"]
    ha["HA_Bearish"] = ha["HA_Close"] < ha["HA_Open"]

    # Retain raw price bars for realistic execution and order levels
    ha["Open"] = frame["Open"]
    ha["High"] = frame["High"]
    ha["Low"] = frame["Low"]
    ha["Close"] = frame["Close"]
    if "Volume" in frame.columns:
        ha["Volume"] = frame["Volume"]

    return ha


def resample_ohlc(df: pd.DataFrame, timeframe: str = "weekly") -> pd.DataFrame:
    """Resamples daily OHLC data into standard weekly or monthly bars.

    Args:
        df: Daily OHLC dataframe with DatetimeIndex.
        timeframe: 'daily', 'weekly' (ends Friday: W-FRI), or 'monthly' (month-end: ME).

    Returns:
        Resampled OHLC DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["Open", "Close"]).copy()
    df.index = pd.to_datetime(df.index)

    tf_lower = timeframe.lower().strip()
    rule_map = {
        "daily": None,
        "d": None,
        "weekly": "W-FRI",
        "w": "W-FRI",
        "w-fri": "W-FRI",
        "monthly": "ME",
        "m": "ME",
        "me": "ME",
    }
    if tf_lower not in rule_map:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use 'daily', 'weekly', or 'monthly'.")

    rule = rule_map[tf_lower]
    if rule is None:
        return df[["Open", "High", "Low", "Close"]].copy()

    agg_dict = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in df.columns:
        agg_dict["Volume"] = "sum"

    resampled = df.resample(rule).agg(agg_dict).dropna(subset=["Open", "Close"])
    return resampled


def compute_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Computes Ichimoku Kinko Hyo lines and equilibrium cloud components.

    Components:
        - Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 over 9 periods
        - Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 over 26 periods
        - Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted forward by displacement
        - Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 over 52 periods shifted forward
        - Chikou Span (Lagging Span): Close shifted backward by displacement

    Returns:
        DataFrame with original OHLC + Ichimoku indicators and Kumo cloud status flags.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    frame = df.copy()
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]

    # Tenkan-sen (Conversion Line)
    tenkan = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2.0

    # Kijun-sen (Base Line)
    kijun = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2.0

    # Senkou Span A
    span_a = (tenkan + kijun) / 2.0

    # Senkou Span B
    span_b = (high.rolling(window=senkou_b_period).max() + low.rolling(window=senkou_b_period).min()) / 2.0

    frame["Tenkan_Sen"] = tenkan
    frame["Kijun_Sen"] = kijun
    frame["Senkou_Span_A"] = span_a.shift(displacement)
    frame["Senkou_Span_B"] = span_b.shift(displacement)
    frame["Chikou_Span"] = close.shift(-displacement)

    # Cloud boundaries and relationships
    frame["Kumo_Top"] = np.maximum(frame["Senkou_Span_A"], frame["Senkou_Span_B"])
    frame["Kumo_Bottom"] = np.minimum(frame["Senkou_Span_A"], frame["Senkou_Span_B"])
    frame["Above_Kumo"] = frame["Close"] > frame["Kumo_Top"]
    frame["Below_Kumo"] = frame["Close"] < frame["Kumo_Bottom"]
    frame["Inside_Kumo"] = (frame["Close"] >= frame["Kumo_Bottom"]) & (frame["Close"] <= frame["Kumo_Top"])
    frame["TK_Bullish_Cross"] = (frame["Tenkan_Sen"] > frame["Kijun_Sen"]) & (
        frame["Tenkan_Sen"].shift(1) <= frame["Kijun_Sen"].shift(1)
    )

    return frame


def compute_relative_strength(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback: int = 60,
) -> pd.DataFrame:
    """Computes relative price performance of a stock against an index benchmark.

    Ratio = Stock Close / Benchmark Close
    RS Score = Percentage change of Ratio over lookback periods.
    """
    if stock_df.empty or benchmark_df.empty:
        return pd.DataFrame()

    aligned = pd.DataFrame(index=stock_df.index)
    aligned["Stock_Close"] = stock_df["Close"]
    aligned["Bench_Close"] = benchmark_df["Close"].reindex(stock_df.index).ffill()

    aligned = aligned.dropna()
    if len(aligned) < lookback + 1:
        return pd.DataFrame()

    aligned["RS_Ratio"] = aligned["Stock_Close"] / aligned["Bench_Close"]
    aligned["RS_Momentum"] = (
        (aligned["RS_Ratio"] - aligned["RS_Ratio"].shift(lookback)) / aligned["RS_Ratio"].shift(lookback) * 100.0
    )

    return aligned

