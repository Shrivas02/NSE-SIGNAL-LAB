import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf

"""NSE 500 Monthly Heikin-Ashi Reversal Scanner.

Scans for:
  - Bullish Reversals: >=3 consecutive red HA monthly candles followed by 1 green HA candle
  - Bearish Reversals: >=3 consecutive green HA monthly candles followed by 1 red HA candle
"""

SIGNAL_MONTH = "2026-08"
BASE_DIR = Path(__file__).resolve().parent
CONSTITUENTS_FILE = BASE_DIR / "ind_nifty500list.csv"
BATCH_SIZE = 50
YF_PERIOD = "max"



def load_symbols():
    if not CONSTITUENTS_FILE.exists():
        raise FileNotFoundError(f"Constituents file not found: {CONSTITUENTS_FILE}")

    df = pd.read_csv(CONSTITUENTS_FILE)
    required = {"Company Name", "Industry", "Symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in constituent file: {missing}")

    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df[df["Symbol"].ne("")].drop_duplicates("Symbol")
    df["YahooSymbol"] = df["Symbol"] + ".NS"
    return df


def compute_heikin_ashi(df):
    """Calculates Heikin-Ashi candles from regular OHLC dataframe."""
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    ha = pd.DataFrame(index=df.index)
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0

    ha_open = np.zeros(len(df))
    ha_open[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha["HA_Open"] = ha_open
    ha["HA_Close"] = ha_close.values
    ha["HA_High"] = np.maximum.reduce([df["High"].values, ha_open, ha_close.values])
    ha["HA_Low"] = np.minimum.reduce([df["Low"].values, ha_open, ha_close.values])

    ha["HA_Bullish"] = ha["HA_Close"] > ha["HA_Open"]
    ha["HA_Bearish"] = ha["HA_Close"] < ha["HA_Open"]

    # Keep original OHLC for actual trade execution
    ha["Open"] = df["Open"]
    ha["High"] = df["High"]
    ha["Low"] = df["Low"]
    ha["Close"] = df["Close"]
    return ha


def monthly_ohlc(daily):
    daily = daily.dropna(how="all").copy()
    if daily.empty:
        return pd.DataFrame()

    daily.index = pd.to_datetime(daily.index)
    monthly = daily.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna(subset=["Open", "Close"])

    return compute_heikin_ashi(monthly)


def count_prior_streak(frame, pos, streak_type="bearish"):
    """Counts consecutive prior monthly HA candles of the specified type."""
    count = 0
    j = pos - 1
    flag_col = "HA_Bearish" if streak_type == "bearish" else "HA_Bullish"

    while j >= 0:
        prev_period = frame.index[j].to_period("M")
        next_period = frame.index[j + 1].to_period("M")
        if (next_period - prev_period).n != 1:
            break
        candle = frame.iloc[j]
        if bool(candle[flag_col]):
            count += 1
            j -= 1
        else:
            break
    return count, j


def find_monthly_signals(monthly, signal_month):
    """Detects both Bullish and Bearish Heikin-Ashi reversals for the given month."""
    if monthly.empty or len(monthly) < 4:
        return None, None

    monthly = monthly.copy()
    monthly["YM"] = monthly.index.to_period("M").astype(str)

    rows = monthly[monthly["YM"] == signal_month]
    if rows.empty:
        return None, None

    signal_idx = rows.index[-1]
    signal = monthly.loc[signal_idx]
    pos = monthly.index.get_loc(signal_idx)

    bullish_sig = None
    bearish_sig = None

    # 1. Bullish Reversal Check (Prior 3+ Red HA -> Current 1 Green HA)
    if bool(signal["HA_Bullish"]):
        bearish_count, j = count_prior_streak(monthly, pos, streak_type="bearish")
        if bearish_count >= 3:
            first_bearish = monthly.index[j + 1]
            entry = round(float(signal["High"]), 2)
            sl = round(float(signal["Low"]), 2)
            risk = round(entry - sl, 2)
            t1 = round(entry + risk, 2) if risk > 0 else np.nan
            t2 = round(entry + 2 * risk, 2) if risk > 0 else np.nan

            bullish_sig = {
                "Signal Type": "Bullish Reversal (Long)",
                "Streak Count": bearish_count,
                "Streak Type": "Bearish Months",
                "First Streak Month": first_bearish.strftime("%Y-%m"),
                "Reversal Month": signal_idx.strftime("%Y-%m"),
                "Reversal Date": signal_idx.strftime("%Y-%m-%d"),
                "HA Open": round(float(signal["HA_Open"]), 2),
                "HA High": round(float(signal["HA_High"]), 2),
                "HA Low": round(float(signal["HA_Low"]), 2),
                "HA Close": round(float(signal["HA_Close"]), 2),
                "Actual Open": round(float(signal["Open"]), 2),
                "Actual High": round(float(signal["High"]), 2),
                "Actual Low": round(float(signal["Low"]), 2),
                "Actual Close": round(float(signal["Close"]), 2),
                "Entry": entry,
                "SL": sl,
                "Risk": risk,
                "Target 1": t1,
                "Target 2": t2,
            }

    # 2. Bearish Reversal Check (Prior 3+ Green HA -> Current 1 Red HA)
    if bool(signal["HA_Bearish"]):
        bullish_count, j = count_prior_streak(monthly, pos, streak_type="bullish")
        if bullish_count >= 3:
            first_bullish = monthly.index[j + 1]
            entry = round(float(signal["Low"]), 2)
            sl = round(float(signal["High"]), 2)
            risk = round(sl - entry, 2)
            t1 = round(entry - risk, 2) if risk > 0 else np.nan
            t2 = round(entry - 2 * risk, 2) if risk > 0 else np.nan

            bearish_sig = {
                "Signal Type": "Bearish Reversal (Short)",
                "Streak Count": bullish_count,
                "Streak Type": "Bullish Months",
                "First Streak Month": first_bullish.strftime("%Y-%m"),
                "Reversal Month": signal_idx.strftime("%Y-%m"),
                "Reversal Date": signal_idx.strftime("%Y-%m-%d"),
                "HA Open": round(float(signal["HA_Open"]), 2),
                "HA High": round(float(signal["HA_High"]), 2),
                "HA Low": round(float(signal["HA_Low"]), 2),
                "HA Close": round(float(signal["HA_Close"]), 2),
                "Actual Open": round(float(signal["Open"]), 2),
                "Actual High": round(float(signal["High"]), 2),
                "Actual Low": round(float(signal["Low"]), 2),
                "Actual Close": round(float(signal["Close"]), 2),
                "Entry": entry,
                "SL": sl,
                "Risk": risk,
                "Target 1": t1,
                "Target 2": t2,
            }

    return bullish_sig, bearish_sig


def score_and_rank_signals(df, is_bullish=True):
    if df.empty:
        return df
    df = df.copy()

    if is_bullish:
        # Move % = (Close - Open) / Open * 100
        df["Move %"] = ((df["Actual Close"] - df["Actual Open"]) / df["Actual Open"] * 100).round(2)
        # Risk % = Risk / Entry * 100
        df["Risk %"] = ((df["Risk"] / df["Entry"]) * 100).round(2)
        # Close Pos % = (Close - Low) / (High - Low) * 100
        candle_range = df["Actual High"] - df["Actual Low"]
        df["Close Pos %"] = np.where(
            candle_range > 0,
            ((df["Actual Close"] - df["Actual Low"]) / candle_range * 100).round(2),
            100.0,
        )
    else:
        # Downward Move % = (Open - Close) / Open * 100
        df["Move %"] = ((df["Actual Open"] - df["Actual Close"]) / df["Actual Open"] * 100).round(2)
        # Risk % = Risk / Entry * 100
        df["Risk %"] = ((df["Risk"] / df["Entry"]) * 100).round(2)
        # Downside Close Pos % = (High - Close) / (High - Low) * 100
        candle_range = df["Actual High"] - df["Actual Low"]
        df["Close Pos %"] = np.where(
            candle_range > 0,
            ((df["Actual High"] - df["Actual Close"]) / candle_range * 100).round(2),
            100.0,
        )

    # Scoring (0-100):
    s_close = (df["Close Pos %"] / 100.0) * 35.0
    max_move = df["Move %"].max()
    s_move = np.where(
        max_move > 0,
        np.clip(df["Move %"] / max_move, 0, 1) * 30.0,
        15.0,
    )
    s_count = np.clip((df["Streak Count"] / 6.0) * 20.0, 10.0, 20.0)
    max_risk = df["Risk %"].max()
    s_risk = np.where(
        max_risk > 0,
        np.clip((1.0 - (df["Risk %"] / (max_risk * 1.5))) * 15.0, 3.0, 15.0),
        10.0,
    )

    df["Score"] = (s_close + s_move + s_count + s_risk).round(1)
    df = df.sort_values(by=["Score", "Move %"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def download_batch(tickers):
    return yf.download(
        tickers=tickers,
        period=YF_PERIOD,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )


def extract_ticker_frame(raw, ticker):
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


def month_label(signal_month):
    dt = pd.Period(signal_month, freq="M").to_timestamp()
    return dt.strftime("%B %Y")


def main():
    parser = argparse.ArgumentParser(description="NSE 500 Monthly Heikin-Ashi Reversal Scanner")
    parser.add_argument("--month", default=SIGNAL_MONTH, help="YYYY-MM target month (e.g. 2026-08)")
    args = parser.parse_args()
    target_month = args.month
    label = month_label(target_month)

    universe = load_symbols()
    bullish_results = []
    bearish_results = []
    failed = []
    tickers = universe["YahooSymbol"].tolist()

    print("==========================================================")
    print("  NSE 500 MONTHLY HEIKIN-ASHI REVERSAL SCANNER")
    print(f"  Target Month: {label} ({target_month})")
    print(f"  Universe: {len(universe)} NSE 500 stocks")
    print("==========================================================")
    print("Downloading historical cash-market data in batches...")

    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start : start + BATCH_SIZE]
        print(f"Processing {start + 1}-{min(start + BATCH_SIZE, len(tickers))} of {len(tickers)}...")

        try:
            raw = download_batch(batch)
        except Exception as e:
            failed.extend(batch)
            print(f"Batch failed {start}:{start + len(batch)}: {e}")
            continue

        for yahoo_symbol in batch:
            row = universe.loc[universe["YahooSymbol"] == yahoo_symbol].iloc[0]
            stock = row["Symbol"]
            company = row["Company Name"]
            industry = row["Industry"]

            try:
                stock_daily = extract_ticker_frame(raw, yahoo_symbol)
                if stock_daily.empty:
                    failed.append(yahoo_symbol)
                    continue

                m_ohlc = monthly_ohlc(stock_daily)
                b_sig, s_sig = find_monthly_signals(m_ohlc, target_month)
                if b_sig:
                    bullish_results.append({
                        "Stock": stock,
                        "Company Name": company,
                        "Industry": industry,
                        **b_sig,
                    })
                if s_sig:
                    bearish_results.append({
                        "Stock": stock,
                        "Company Name": company,
                        "Industry": industry,
                        **s_sig,
                    })
            except Exception as e:
                failed.append(yahoo_symbol)
                print(f"Failed {yahoo_symbol}: {e}")

        time.sleep(1)

    bullish_df = score_and_rank_signals(pd.DataFrame(bullish_results), is_bullish=True)
    bearish_df = score_and_rank_signals(pd.DataFrame(bearish_results), is_bullish=False)

    out_stem = f"NSE500_Monthly_{target_month}_HA_Reversal_Scan"
    out_xlsx = BASE_DIR / f"{out_stem}.xlsx"
    out_csv = BASE_DIR / f"{out_stem}.csv"

    # Combine for the main summary sheet
    combined_frames = []
    if not bullish_df.empty:
        combined_frames.append(bullish_df)
    if not bearish_df.empty:
        combined_frames.append(bearish_df)

    summary_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()

    cols_order = [
        "Signal Type", "Rank", "Stock", "Company Name", "Industry", "Score", "Streak Count",
        "Move %", "Close Pos %", "Risk %", "Streak Type", "First Streak Month",
        "Reversal Month", "Reversal Date",
        "Entry", "SL", "Risk", "Target 1", "Target 2",
        "HA Open", "HA High", "HA Low", "HA Close",
        "Actual Open", "Actual High", "Actual Low", "Actual Close",
    ]
    if not summary_df.empty:
        summary_df = summary_df[[c for c in cols_order if c in summary_df.columns]]
    if not bullish_df.empty:
        bullish_df = bullish_df[[c for c in cols_order if c in bullish_df.columns]]
    if not bearish_df.empty:
        bearish_df = bearish_df[[c for c in cols_order if c in bearish_df.columns]]

    summary_df.to_csv(out_csv, index=False)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="All Signals", index=False)
        bullish_df.to_excel(writer, sheet_name="Bullish Reversals (Long)", index=False)
        bearish_df.to_excel(writer, sheet_name="Bearish Reversals (Short)", index=False)
        universe.to_excel(writer, sheet_name="NSE 500 Universe", index=False)
        if failed:
            pd.DataFrame({"YahooSymbol": sorted(set(failed))}).to_excel(
                writer, sheet_name="Download Failures", index=False
            )

    print(f"\n================ MONTHLY HEIKIN-ASHI SCAN ({label.upper()}) ================")
    print(f"Bullish Long Reversals (3+ Red -> 1 Green): {len(bullish_df)}")
    print(f"Bearish Short Reversals (3+ Green -> 1 Red): {len(bearish_df)}")
    print(f"Saved Excel: {out_xlsx.name}")
    print(f"Saved CSV:   {out_csv.name}")

    if not bullish_df.empty:
        print("\n>>> BULLISH REVERSALS (LONG) <<<")
        print(bullish_df[[
            "Rank", "Stock", "Company Name", "Score", "Streak Count", "Move %", "Close Pos %", "Risk %",
            "Entry", "SL", "Target 1", "Target 2"
        ]].to_string(index=False))

    if not bearish_df.empty:
        print("\n>>> BEARISH REVERSALS (SHORT) <<<")
        print(bearish_df[[
            "Rank", "Stock", "Company Name", "Score", "Streak Count", "Move %", "Close Pos %", "Risk %",
            "Entry", "SL", "Target 1", "Target 2"
        ]].to_string(index=False))

    if bullish_df.empty and bearish_df.empty:
        print(f"\nNo qualifying Heikin-Ashi reversals found for {label}.")

    if failed:
        print(f"\nDownload/processing failures: {len(set(failed))}")


if __name__ == "__main__":
    main()
