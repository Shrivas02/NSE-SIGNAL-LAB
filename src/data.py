"""Data Ingestion and Management Utilities Module.

Handles:
- Loading constituent lists (Nifty 50, Nifty 500, F&O list)
- Downloading historical OHLC from Yahoo Finance
- Safe extraction of single-ticker frames from multi-index containers
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import yfinance as yf


def load_universe(file_path: Path) -> pd.DataFrame:
    """Loads and standardizes an equity constituent CSV file.

    Args:
        file_path: Path to constituent CSV file.

    Returns:
        Cleaned DataFrame with 'Symbol', 'Company Name', 'Industry', and 'YahooSymbol'.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Constituents file not found: {path}")

    df = pd.read_csv(path)
    required = {"Company Name", "Industry", "Symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in constituent file {path.name}: {missing}")

    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df[df["Symbol"].ne("")].drop_duplicates("Symbol").reset_index(drop=True)

    if "YahooSymbol" not in df.columns:
        df["YahooSymbol"] = df["Symbol"] + ".NS"

    return df


def extract_ticker_ohlc(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Safely extracts a single ticker's OHLC DataFrame from yfinance batch downloads."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            frame = raw[ticker].copy()
        elif ticker in raw.columns.get_level_values(1):
            frame = raw.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = raw.copy()

    needed = ["Open", "High", "Low", "Close"]
    if not all(c in frame.columns for c in needed):
        return pd.DataFrame()

    return frame[needed].dropna(how="all")


def fetch_batch_ohlc(
    tickers: List[str],
    period: str = "max",
    interval: str = "1d",
    threads: bool = True,
) -> pd.DataFrame:
    """Downloads cash-market historical OHLC from Yahoo Finance in batches."""
    return yf.download(
        tickers=list(tickers),
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        threads=threads,
        progress=False,
    )

