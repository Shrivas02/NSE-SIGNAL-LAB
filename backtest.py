"""Quantitative Backtesting Simulation Engine.

Simulates historical event-driven execution of Heikin-Ashi and Candlestick
reversal setups without lookahead bias:
- Models realistic order triggers (breakout above High for Long, breakdown below Low for Short)
- Evaluates Target 1 (1R) / Target 2 (2R) vs Stop Loss (SL) bar-by-bar
- Calculates quantitative performance metrics: Win Rate %, Profit Factor,
  Expectancy (R/trade), Cumulative R curve, and Max Drawdown.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.indicators import compute_heikin_ashi
from src.patterns import detect_reversal_pattern


class ReversalBacktester:
    """Historical backtester for multi-timeframe reversal breakout strategies."""

    def __init__(
        self,
        streak_min: int = 3,
        target_multiple: float = 1.0,
        max_holding_bars: int = 12,
        trigger_bars: int = 2,
    ):
        """
        Args:
            streak_min: Minimum consecutive prior red/green bars required to qualify (default 3).
            target_multiple: Target R-multiple (1.0 = 1R Target, 2.0 = 2R Target).
            max_holding_bars: Maximum duration to hold open trade before market exit timeout.
            trigger_bars: Maximum bars to wait for breakout entry trigger before expiring.
        """
        self.streak_min = streak_min
        self.target_multiple = target_multiple
        self.max_holding_bars = max_holding_bars
        self.trigger_bars = trigger_bars

    def backtest_series(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        direction: str = "both",
    ) -> pd.DataFrame:
        """Runs bar-by-bar backtest over historical price series.

        Args:
            df: DataFrame containing OHLC bars (with DatetimeIndex).
            symbol: Ticker symbol string.
            direction: 'both', 'long', or 'short'.

        Returns:
            DataFrame containing individual trade logs.
        """
        if df is None or len(df) < (self.streak_min + 5):
            return pd.DataFrame()

        # Compute Heikin-Ashi if not already present
        if "HA_Bullish" not in df.columns:
            frame = compute_heikin_ashi(df)
        else:
            frame = df.copy()

        trades: List[Dict[str, Any]] = []
        n_bars = len(frame)

        # Iterate bar-by-bar
        for i in range(self.streak_min, n_bars - 1):
            bullish_sig, bearish_sig = detect_reversal_pattern(
                frame, i, streak_min=self.streak_min
            )

            # 1. Evaluate Long Setup
            if bullish_sig and direction.lower() in ("both", "long"):
                trade = self._simulate_long_trade(frame, i, bullish_sig, symbol)
                if trade:
                    trades.append(trade)

            # 2. Evaluate Short Setup
            if bearish_sig and direction.lower() in ("both", "short"):
                trade = self._simulate_short_trade(frame, i, bearish_sig, symbol)
                if trade:
                    trades.append(trade)

        if not trades:
            return pd.DataFrame()

        trade_df = pd.DataFrame(trades)
        return trade_df

    def _simulate_long_trade(
        self,
        frame: pd.DataFrame,
        sig_idx: int,
        sig: Dict[str, Any],
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """Simulates execution and exit for a Long reversal setup."""
        entry_price = sig["Entry"]
        sl_price = sig["SL"]
        risk = sig["Risk"]
        if risk <= 0:
            return None

        target_price = round(entry_price + self.target_multiple * risk, 2)
        n_bars = len(frame)

        # Step 1: Wait for entry breakout trigger
        triggered = False
        trigger_idx = None
        for step in range(1, self.trigger_bars + 1):
            curr_idx = sig_idx + step
            if curr_idx >= n_bars:
                break
            bar = frame.iloc[curr_idx]
            if bar["High"] >= entry_price:
                triggered = True
                trigger_idx = curr_idx
                break

        if not triggered or trigger_idx is None:
            return None  # Setup expired without trigger

        # Step 2: Walk forward from trigger bar to evaluate SL vs Target
        entry_date = frame.index[trigger_idx]
        outcome = "TIMEOUT"
        exit_price = frame.iloc[min(trigger_idx + self.max_holding_bars, n_bars - 1)]["Close"]
        exit_date = frame.index[min(trigger_idx + self.max_holding_bars, n_bars - 1)]
        holding_bars = 0

        for step in range(trigger_idx, min(trigger_idx + self.max_holding_bars, n_bars)):
            bar = frame.iloc[step]
            holding_bars += 1

            # Check Stop Loss first (conservative risk modeling)
            if bar["Low"] <= sl_price:
                outcome = "STOP_LOSS"
                exit_price = sl_price
                exit_date = frame.index[step]
                break

            # Check Target Hit
            if bar["High"] >= target_price:
                outcome = "TARGET_HIT"
                exit_price = target_price
                exit_date = frame.index[step]
                break

        pnl_cash = round(exit_price - entry_price, 2)
        pnl_pct = round((pnl_cash / entry_price) * 100.0, 2)
        pnl_r = round(pnl_cash / risk, 2)

        return {
            "Symbol": symbol,
            "Direction": "Long",
            "Signal Date": sig["Signal Date"],
            "Entry Date": entry_date,
            "Exit Date": exit_date,
            "Streak Count": sig["Streak Count"],
            "Entry Price": entry_price,
            "SL Price": sl_price,
            "Target Price": target_price,
            "Exit Price": exit_price,
            "Risk": risk,
            "Outcome": outcome,
            "PnL %": pnl_pct,
            "PnL (R)": pnl_r,
            "Holding Bars": holding_bars,
        }

    def _simulate_short_trade(
        self,
        frame: pd.DataFrame,
        sig_idx: int,
        sig: Dict[str, Any],
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """Simulates execution and exit for a Short breakdown setup."""
        entry_price = sig["Entry"]
        sl_price = sig["SL"]
        risk = sig["Risk"]
        if risk <= 0:
            return None

        target_price = round(entry_price - self.target_multiple * risk, 2)
        n_bars = len(frame)

        # Step 1: Wait for entry breakdown trigger
        triggered = False
        trigger_idx = None
        for step in range(1, self.trigger_bars + 1):
            curr_idx = sig_idx + step
            if curr_idx >= n_bars:
                break
            bar = frame.iloc[curr_idx]
            if bar["Low"] <= entry_price:
                triggered = True
                trigger_idx = curr_idx
                break

        if not triggered or trigger_idx is None:
            return None

        # Step 2: Walk forward from trigger bar
        entry_date = frame.index[trigger_idx]
        outcome = "TIMEOUT"
        exit_price = frame.iloc[min(trigger_idx + self.max_holding_bars, n_bars - 1)]["Close"]
        exit_date = frame.index[min(trigger_idx + self.max_holding_bars, n_bars - 1)]
        holding_bars = 0

        for step in range(trigger_idx, min(trigger_idx + self.max_holding_bars, n_bars)):
            bar = frame.iloc[step]
            holding_bars += 1

            # Check Stop Loss first (High breaks above SL)
            if bar["High"] >= sl_price:
                outcome = "STOP_LOSS"
                exit_price = sl_price
                exit_date = frame.index[step]
                break

            # Check Target Hit (Low breaks below Target)
            if bar["Low"] <= target_price:
                outcome = "TARGET_HIT"
                exit_price = target_price
                exit_date = frame.index[step]
                break

        pnl_cash = round(entry_price - exit_price, 2)
        pnl_pct = round((pnl_cash / entry_price) * 100.0, 2)
        pnl_r = round(pnl_cash / risk, 2)

        return {
            "Symbol": symbol,
            "Direction": "Short",
            "Signal Date": sig["Signal Date"],
            "Entry Date": entry_date,
            "Exit Date": exit_date,
            "Streak Count": sig["Streak Count"],
            "Entry Price": entry_price,
            "SL Price": sl_price,
            "Target Price": target_price,
            "Exit Price": exit_price,
            "Risk": risk,
            "Outcome": outcome,
            "PnL %": pnl_pct,
            "PnL (R)": pnl_r,
            "Holding Bars": holding_bars,
        }

    @staticmethod
    def calculate_metrics(trade_log: pd.DataFrame) -> Dict[str, Any]:
        """Computes statistical performance metrics from a trade log."""
        if trade_log is None or trade_log.empty:
            return {
                "Total Trades": 0,
                "Win Rate %": 0.0,
                "Profit Factor": 0.0,
                "Expectancy (R)": 0.0,
                "Total PnL (R)": 0.0,
                "Max Drawdown (R)": 0.0,
                "Avg Holding Bars": 0.0,
            }

        total = len(trade_log)
        wins = trade_log[trade_log["PnL (R)"] > 0]
        losses = trade_log[trade_log["PnL (R)"] < 0]

        win_rate = round((len(wins) / total) * 100.0, 2)
        gross_profit = wins["PnL (R)"].sum()
        gross_loss = abs(losses["PnL (R)"].sum())

        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        expectancy = round(trade_log["PnL (R)"].mean(), 2)
        total_r = round(trade_log["PnL (R)"].sum(), 2)

        # Cumulative R curve and Max Drawdown
        cum_r = trade_log["PnL (R)"].cumsum()
        peak = cum_r.cummax()
        drawdown = peak - cum_r
        max_dd = round(drawdown.max(), 2) if not drawdown.empty else 0.0

        avg_bars = round(trade_log["Holding Bars"].mean(), 1)

        return {
            "Total Trades": total,
            "Winning Trades": len(wins),
            "Losing Trades": len(losses),
            "Win Rate %": win_rate,
            "Profit Factor": profit_factor,
            "Expectancy (R)": expectancy,
            "Total PnL (R)": total_r,
            "Max Drawdown (R)": max_dd,
            "Avg Holding Bars": avg_bars,
        }

