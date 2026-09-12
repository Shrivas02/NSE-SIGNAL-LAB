"""CLI Runner for Quantitative Strategy Backtesting.

Usage:
    python run_backtest.py --universe nifty50 --timeframe weekly
    python run_backtest.py --universe futures --timeframe monthly --target 2.0
    python run_backtest.py --symbol RELIANCE.NS --timeframe weekly
"""

import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

from src.backtest import ReversalBacktester
from src.data import load_universe, extract_ticker_ohlc, fetch_batch_ohlc
from src.indicators import resample_ohlc

BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="NSE Reversal Strategy Backtester")
    parser.add_argument("--universe", default="nifty50", choices=["nifty50", "futures", "nifty500"],
                        help="Target universe to backtest across")
    parser.add_argument("--symbol", default=None, help="Specific single ticker (e.g. RELIANCE.NS)")
    parser.add_argument("--timeframe", default="weekly", choices=["daily", "weekly", "monthly"],
                        help="Candle timeframe")
    parser.add_argument("--period", default="4y", help="Historical lookback period (e.g. 2y, 4y, max)")
    parser.add_argument("--streak", type=int, default=3, help="Minimum consecutive counter-trend streak")
    parser.add_argument("--target", type=float, default=1.0, help="Target R-multiple (1.0 = 1R, 2.0 = 2R)")
    parser.add_argument("--holding", type=int, default=12, help="Max holding period in bars")
    parser.add_argument("--direction", default="both", choices=["both", "long", "short"], help="Trade direction")
    args = parser.parse_args()

    print("==========================================================")
    print("  NSE QUANTITATIVE STRATEGY BACKTESTING ENGINE")
    print(f"  Timeframe: {args.timeframe.upper()} | Lookback: {args.period} | Streak: >={args.streak} | Target: {args.target}R")
    print("==========================================================")

    backtester = ReversalBacktester(
        streak_min=args.streak,
        target_multiple=args.target,
        max_holding_bars=args.holding,
    )

    if args.symbol:
        tickers = [args.symbol if args.symbol.endswith(".NS") or args.symbol.startswith("^") else f"{args.symbol}.NS"]
    else:
        file_map = {
            "nifty50": BASE_DIR / "ind_nifty50list.csv",
            "futures": BASE_DIR / "ind_nifty_futures_list.csv",
            "nifty500": BASE_DIR / "ind_nifty500list.csv",
        }
        uni_df = load_universe(file_map[args.universe])
        tickers = uni_df["YahooSymbol"].tolist()

    print(f"Downloading historical data for {len(tickers)} symbols...")
    raw = fetch_batch_ohlc(tickers, period=args.period, interval="1d")

    all_trades = []
    for ticker in tickers:
        df = extract_ticker_ohlc(raw, ticker)
        if df.empty or len(df) < 50:
            continue

        resampled = resample_ohlc(df, timeframe=args.timeframe)
        if len(resampled) < 20:
            continue

        sym_name = ticker.replace(".NS", "")
        trades = backtester.backtest_series(resampled, symbol=sym_name, direction=args.direction)
        if not trades.empty:
            all_trades.append(trades)

    if not all_trades:
        print("\nNo trades executed during the backtest window.")
        return

    combined_trades = pd.concat(all_trades, ignore_index=True)
    metrics = backtester.calculate_metrics(combined_trades)

    print("\n================ BACKTEST PERFORMANCE SUMMARY ================")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print("==============================================================")

    # Save detailed trade log
    out_file = BASE_DIR / f"backtest_trades_{args.timeframe}_{args.target}R.csv"
    combined_trades.to_csv(out_file, index=False)
    print(f"\nDetailed trade log ({len(combined_trades)} trades) saved to: {out_file.name}")

    # Preview top winning trades
    print("\nSample Trade Log (Last 5 Trades):")
    sample_cols = ["Symbol", "Direction", "Signal Date", "Outcome", "PnL %", "PnL (R)", "Holding Bars"]
    print(combined_trades[sample_cols].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()

