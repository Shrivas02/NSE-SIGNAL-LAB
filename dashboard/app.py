from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import plotly.graph_objects as go
    import yfinance as yf
    from src.backtest import ReversalBacktester
    from src.indicators import resample_ohlc
except ImportError as exc:
    st.error(f"Missing dashboard dependency: {exc.name}. Run: pip install -r requirements.txt")
    st.stop()

CONFIGS = {
    "Ichimoku Daily / NSE 500": {
        "ichimoku": True,
        "timeframe": "daily",
        "date_label": "Signal date",
        "universe": "ind_nifty500list.csv",
        "universe_label": "NSE 500",
    },
    "Ichimoku Weekly / NSE 500": {
        "ichimoku": True,
        "timeframe": "weekly",
        "date_label": "Signal week",
        "universe": "ind_nifty500list.csv",
        "universe_label": "NSE 500",
    },
    "Ichimoku Monthly / NSE 500": {
        "ichimoku": True,
        "timeframe": "monthly",
        "date_label": "Signal month",
        "universe": "ind_nifty500list.csv",
        "universe_label": "NSE 500",
    },
    "RS Monthly": {
        "rs": True,
        "timeframe": "monthly",
        "date_label": "Latest completed month",
        "default_date": "Latest",
        "universe": "ind_nifty500list.csv",
        "universe_label": "NSE 500",
    },
    "RS Weekly": {
        "rs": True,
        "timeframe": "weekly",
        "date_label": "Latest completed week",
        "default_date": "Latest",
        "universe": "ind_nifty500list.csv",
        "universe_label": "NSE 500",
    },
    "RS Daily": {
        "rs": True,
        "timeframe": "daily",
        "date_label": "Latest completed day",
        "default_date": "Latest",
        "universe": "ind_nifty500list.csv",
        "universe_label": "NSE 500",
    },
    "RS Monthly / NIFTY 50": {
        "rs": True,
        "timeframe": "monthly",
        "date_label": "Latest completed month",
        "default_date": "Latest",
        "universe": "ind_nifty50list.csv",
        "universe_label": "NIFTY 50",
    },
    "RS Weekly / NIFTY 50": {
        "rs": True,
        "timeframe": "weekly",
        "date_label": "Latest completed week",
        "default_date": "Latest",
        "universe": "ind_nifty50list.csv",
        "universe_label": "NIFTY 50",
    },
    "NSE 500 / Monthly": {
        "script": BASE_DIR / "HAmonthly.py",
        "pattern": "NSE500_Monthly_*_HA_Reversal_Scan.csv",
        "date_label": "Signal month",
        "default_date": "2026-08",
        "date_format": "YYYY-MM",
        "universe": "ind_nifty500list.csv",
    },
    "NIFTY 50 / Weekly": {
        "script": BASE_DIR / "HAweekly.py",
        "pattern": "NIFTY50_Weekly_*_HA_Reversal_Scan.csv",
        "date_label": "Week date",
        "default_date": "2026-08-28",
        "date_format": "YYYY-MM-DD",
        "universe": "ind_nifty50list.csv",
    },
    "NSE Futures / Monthly": {
        "script": BASE_DIR / "futures_monthly.py",
        "pattern": "NSE_Futures_Monthly_*_HA_Reversal_Scan.csv",
        "date_label": "Signal month",
        "default_date": "2026-08",
        "date_format": "YYYY-MM",
        "universe": "ind_nifty_futures_list.csv",
    },
    "NSE Futures / Weekly": {
        "script": BASE_DIR / "futures_weekly.py",
        "pattern": "NIFTY_Futures_Weekly_*_HA_Reversal_Scan.csv",
        "date_label": "Week date",
        "default_date": "2026-08-28",
        "date_format": "YYYY-MM-DD",
        "universe": "ind_nifty_futures_list.csv",
    },
}

st.set_page_config(page_title="NSE Signal Lab", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1400px; padding-top: 2rem; }
    [data-testid="stMetricValue"] { color: #0b6e4f; }
    .signal-note { padding: 0.8rem 1rem; background: #eef7f2; border-left: 4px solid #0b6e4f; }
    </style>
    """,
    unsafe_allow_html=True,
)


def newest_report(config):
    reports = list(BASE_DIR.glob(config["pattern"]))
    return max(reports, key=lambda path: path.stat().st_mtime) if reports else None


def load_report(config):
    report = newest_report(config)
    if report is None:
        return pd.DataFrame(), None
    return pd.read_csv(report), report


def run_scan(config, target):
    if config["date_format"] == "YYYY-MM":
        command = [sys.executable, str(config["script"]), "--month", target]
    else:
        command = [sys.executable, str(config["script"]), "--date", target]
    result = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, timeout=900)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "The scan failed.")
    return load_report(config)


def compute_heikin_ashi(frame):
    if frame.empty:
        return pd.DataFrame()
    frame = frame[["Open", "High", "Low", "Close"]].dropna().copy()
    ha_close = (frame["Open"] + frame["High"] + frame["Low"] + frame["Close"]) / 4
    ha_open = pd.Series(index=frame.index, dtype=float)
    ha_open.iloc[0] = (frame["Open"].iloc[0] + frame["Close"].iloc[0]) / 2
    for position in range(1, len(frame)):
        ha_open.iloc[position] = (ha_open.iloc[position - 1] + ha_close.iloc[position - 1]) / 2
    return pd.DataFrame({"HA_Close": ha_close, "HA_Open": ha_open})


def resample_heikin_ashi(frame, timeframe):
    rule = {"daily": "D", "weekly": "W-FRI", "monthly": "ME"}[timeframe]
    grouped = frame.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last",
    }).dropna(subset=["Open", "Close"])
    return compute_heikin_ashi(grouped)


def extract_download_frame(raw, ticker):
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            raw = raw[ticker]
        elif ticker in raw.columns.get_level_values(1):
            raw = raw.xs(ticker, axis=1, level=1)
        else:
            return pd.DataFrame()
    required = ["Open", "High", "Low", "Close"]
    return raw[required].dropna(how="all") if all(column in raw for column in required) else pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_download(tickers, period):
    return yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )


def download_relative_strength(config, lookback, target_date):
    universe = pd.read_csv(BASE_DIR / config["universe"])
    universe["Symbol"] = universe["Symbol"].astype(str).str.strip()
    universe = universe[universe["Symbol"].ne("")].drop_duplicates("Symbol").copy()
    universe["YahooSymbol"] = universe["Symbol"] + ".NS"
    tickers = universe["YahooSymbol"].tolist()
    history_period = {"daily": "2y", "weekly": "4y", "monthly": "15y"}[config["timeframe"]]
    reference = cached_download(("^NSEI",), history_period)
    reference = extract_download_frame(reference, "^NSEI")
    reference_ha = resample_heikin_ashi(reference, config["timeframe"])
    if len(reference_ha) <= lookback:
        raise ValueError("Not enough NIFTY history for the selected RS lookback.")

    period_frequency = {"daily": "D", "weekly": "W-FRI", "monthly": "M"}[config["timeframe"]]
    target_period = pd.Timestamp(target_date).to_period(period_frequency)
    matching_positions = [
        position for position, period in enumerate(reference_ha.index.to_period(period_frequency))
        if period == target_period
    ]
    latest_position = matching_positions[-1] if matching_positions else len(reference_ha) - 1
    if latest_position == len(reference_ha) - 1:
        latest_position -= 1
    if latest_position <= lookback:
        raise ValueError("The selected date does not have enough history for this lookback.")
    latest_period = reference_ha.index[latest_position]
    reference_current = reference_ha["HA_Close"].iloc[latest_position]
    reference_previous = reference_ha["HA_Close"].iloc[latest_position - lookback]
    reference_prior_current = reference_ha["HA_Close"].iloc[latest_position - 1]
    reference_prior_previous = reference_ha["HA_Close"].iloc[latest_position - 1 - lookback]
    results = []

    progress = st.progress(0, text="Downloading NSE 500 data...")
    batch_size = 100
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start : start + batch_size]
        raw = cached_download(tuple(batch), history_period)
        progress.progress(min((start + len(batch)) / len(tickers), 1.0), text=f"Downloaded {min(start + len(batch), len(tickers))} of {len(tickers)} stocks")
        for ticker in batch:
            stock_frame = extract_download_frame(raw, ticker)
            stock_ha = resample_heikin_ashi(stock_frame, config["timeframe"])
            if len(stock_ha) <= lookback:
                continue
            aligned = stock_ha["HA_Close"].reindex(reference_ha.index)
            if latest_period not in aligned.index or pd.isna(aligned.loc[latest_period]):
                continue
            current_position = aligned.index.get_loc(latest_period)
            current_periods = [latest_position, latest_position - lookback]
            prior_periods = [latest_position - 1, latest_position - 1 - lookback]
            if min(current_periods + prior_periods) < 0:
                continue
            stock_current = aligned.iloc[current_periods[0]]
            stock_previous = aligned.iloc[current_periods[1]]
            stock_prior_current = aligned.iloc[prior_periods[0]]
            stock_prior_previous = aligned.iloc[prior_periods[1]]
            if pd.isna(stock_previous) or pd.isna(stock_prior_current) or pd.isna(stock_prior_previous):
                continue
            current_rs = (stock_current / stock_previous) / (reference_current / reference_previous) - 1
            previous_rs = (stock_prior_current / stock_prior_previous) / (reference_prior_current / reference_prior_previous) - 1
            if previous_rs <= 0 < current_rs:
                row = universe.loc[universe["YahooSymbol"] == ticker].iloc[0]
                results.append({
                    "Stock": row["Symbol"],
                    "Sector": row.get("Industry", ""),
                    "Previous RS": round(float(previous_rs), 2),
                    "Current RS": round(float(current_rs), 2),
                })
    progress.empty()
    result = pd.DataFrame(results)
    result.attrs["checked_period"] = latest_period.strftime("%Y-%m-%d")
    return result.sort_values("Current RS", ascending=False).reset_index(drop=True) if not result.empty else result


def calculate_ichimoku(monthly, conversion_periods=9, base_periods=26, span_b_periods=52, displacement=26):
    result = monthly.copy()
    result["Conversion Line"] = (
        result["High"].rolling(conversion_periods).max()
        + result["Low"].rolling(conversion_periods).min()
    ) / 2
    result["Base Line"] = (
        result["High"].rolling(base_periods).max()
        + result["Low"].rolling(base_periods).min()
    ) / 2
    result["Leading Span A"] = (result["Conversion Line"] + result["Base Line"]) / 2
    result["Leading Span B"] = (
        result["High"].rolling(span_b_periods).max()
        + result["Low"].rolling(span_b_periods).min()
    ) / 2
    visible_shift = displacement - 1
    spans = result[["Leading Span A", "Leading Span B"]]
    raw_cloud_top = spans.max(axis=1, skipna=False)
    raw_cloud_bottom = spans.min(axis=1, skipna=False)
    result["Cloud Top"] = raw_cloud_top.shift(visible_shift)
    result["Cloud Bottom"] = raw_cloud_bottom.shift(visible_shift)
    result["Lagging Cloud Top"] = raw_cloud_top.shift(visible_shift * 2)
    result["Lagging Cloud Bottom"] = raw_cloud_bottom.shift(visible_shift * 2)
    return result


def download_ichimoku(config, target_date):
    universe = pd.read_csv(BASE_DIR / config["universe"])
    universe["Symbol"] = universe["Symbol"].astype(str).str.strip()
    universe = universe[universe["Symbol"].ne("")].drop_duplicates("Symbol").copy()
    universe["YahooSymbol"] = universe["Symbol"] + ".NS"
    timeframe = config["timeframe"]
    history_period = {"daily": "5y", "weekly": "15y", "monthly": "15y"}[timeframe]
    reference = cached_download(("^NSEI",), history_period)
    reference_frame = extract_download_frame(reference, "^NSEI")
    reference_periods = resample_heikin_ashi(reference_frame, timeframe)
    period_frequency = {"daily": "D", "weekly": "W-FRI", "monthly": "M"}[timeframe]
    period = pd.Timestamp(target_date).to_period(period_frequency)
    matching = [position for position, value in enumerate(reference_periods.index.to_period(period_frequency)) if value == period]
    if not matching:
        raise ValueError(f"No {timeframe} market data is available for {period}.")
    target_position = matching[-1]
    if target_position >= len(reference_periods) - 1:
        target_position -= 1
    minimum_history = 52 + (26 - 1) * (2 if timeframe == "daily" else 1) - 1
    if target_position < minimum_history:
        raise ValueError("The selected period does not have enough history for Ichimoku Span B and displacement.")
    target_index = reference_periods.index[target_position]
    results = []
    tickers = universe["YahooSymbol"].tolist()
    progress = st.progress(0, text=f"Downloading {timeframe} data for Ichimoku scan...")

    for start in range(0, len(tickers), 100):
        batch = tickers[start : start + 100]
        raw = cached_download(tuple(batch), history_period)
        progress.progress(min((start + len(batch)) / len(tickers), 1.0), text=f"Downloaded {min(start + len(batch), len(tickers))} of {len(tickers)} stocks")
        for ticker in batch:
            daily = extract_download_frame(raw, ticker)
            if daily.empty:
                continue
            periods = daily.resample({"daily": "D", "weekly": "W-FRI", "monthly": "ME"}[timeframe]).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
            ichimoku = calculate_ichimoku(periods)
            if target_index not in ichimoku.index:
                continue
            position = ichimoku.index.get_loc(target_index)
            minimum_history = 52 + (26 - 1) * (2 if timeframe == "daily" else 1) - 1
            if position < minimum_history or position == 0:
                continue
            current = ichimoku.iloc[position]
            previous = ichimoku.iloc[position - 1]
            if pd.isna(current["Cloud Top"]) or pd.isna(previous["Cloud Top"]):
                continue
            crossed_up_this_period = (
                previous["Close"] <= previous["Cloud Top"]
                and current["Close"] > current["Cloud Top"]
            )
            prior_cloud_rows = ichimoku.iloc[:position].dropna(subset=["Cloud Top"])
            was_above_before = bool((prior_cloud_rows["Close"] > prior_cloud_rows["Cloud Top"]).any())
            row = universe.loc[universe["YahooSymbol"] == ticker].iloc[0]
            common = {
                "Stock": row["Symbol"],
                "Sector": row.get("Industry", ""),
                "Signal Period": target_index.strftime("%Y-%m-%d"),
                "Previous Close": round(float(previous["Close"]), 2),
                "Period Close": round(float(current["Close"]), 2),
                "Previous Cloud Top": round(float(previous["Cloud Top"]), 2),
                "Cloud Top": round(float(current["Cloud Top"]), 2),
                "Cloud Bottom": round(float(current["Cloud Bottom"]), 2),
                "Distance From Cloud %": round(float((current["Close"] / current["Cloud Top"] - 1) * 100), 2),
            }
            if crossed_up_this_period:
                signal_type = "Recovery above cloud" if was_above_before else "First upward cloud breakout"
                results.append({"Signal": signal_type, **common})
            if not pd.isna(current["Lagging Cloud Top"]) and not pd.isna(previous["Lagging Cloud Top"]):
                lagging_crossed_up = (
                    previous["Close"] <= previous["Lagging Cloud Top"]
                    and current["Close"] > current["Lagging Cloud Top"]
                )
                if lagging_crossed_up:
                    results.append({
                        **common,
                        "Signal": "Lagging span breakout above cloud",
                        "Previous Cloud Top": round(float(previous["Lagging Cloud Top"]), 2),
                        "Cloud Top": round(float(current["Lagging Cloud Top"]), 2),
                        "Cloud Bottom": round(float(current["Lagging Cloud Bottom"]), 2),
                        "Distance From Cloud %": round(float((current["Close"] / current["Lagging Cloud Top"] - 1) * 100), 2),
                    })
    progress.empty()
    result = pd.DataFrame(results)
    result.attrs["checked_period"] = target_index.strftime("%Y-%m-%d")
    if result.empty:
        return result
    return result.sort_values(["Signal", "Distance From Cloud %"], ascending=[True, False]).reset_index(drop=True)


def save_ichimoku_excel(frame, checked_period):
    output_path = BASE_DIR / f"Ichimoku_NSE500_{checked_period}.xlsx"
    columns = [
        "Stock", "Sector", "Signal", "Signal Period", "Previous Close", "Period Close",
        "Previous Cloud Top", "Cloud Top", "Cloud Bottom", "Distance From Cloud %",
    ]
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        frame[columns].to_excel(writer, index=False, sheet_name="Ichimoku Signals")
        worksheet = writer.sheets["Ichimoku Signals"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = cell.font.copy(bold=True)
    return output_path


def save_rs_excel(frame, config, checked_period):
    label = config["universe_label"].replace(" ", "")
    filename = f"RS_{label}_{config['timeframe'].title()}_{checked_period}.xlsx"
    output_path = BASE_DIR / filename
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        frame[["Stock", "Sector", "Previous RS", "Current RS"]].to_excel(writer, index=False, sheet_name="RS Crossovers")
        worksheet = writer.sheets["RS Crossovers"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = cell.font.copy(bold=True)
        worksheet.column_dimensions["A"].width = 18
        worksheet.column_dimensions["B"].width = 32
        worksheet.column_dimensions["C"].width = 16
        worksheet.column_dimensions["D"].width = 16
        for row in worksheet.iter_rows(min_row=2, min_col=3, max_col=4):
            for cell in row:
                cell.number_format = "0.00"
    return output_path


def metric_value(frame, column, value):
    if column not in frame:
        return value
    return int((frame[column] == value).sum())


st.title("NSE Signal Lab")
st.caption("Heikin-Ashi reversal research for NSE cash and futures universes")

with st.sidebar:
    st.header("Research controls")
    mode = st.selectbox("Universe and timeframe", list(CONFIGS))
    config = CONFIGS[mode]
    if config.get("ichimoku"):
        if config["timeframe"] == "monthly":
            available_periods = pd.period_range(end=pd.Timestamp.today(), periods=180, freq="M")[::-1]
            selected_period = st.selectbox(config["date_label"], available_periods, format_func=lambda value: value.strftime("%B %Y"))
            target = selected_period.start_time.strftime("%Y-%m-%d")
        else:
            target = st.date_input(config["date_label"], value=pd.Timestamp.today().date()).strftime("%Y-%m-%d")
        st.caption(f"Uses {config['timeframe']} candles and Ichimoku defaults: 9 / 26 / 52 / 26.")
    elif config.get("rs"):
        if config["timeframe"] == "monthly":
            available_months = pd.period_range(end=pd.Timestamp.today(), periods=180, freq="M")[::-1]
            selected_month = st.selectbox(
                config["date_label"],
                available_months,
                format_func=lambda month: month.strftime("%B %Y"),
            )
            target = selected_month.start_time.strftime("%Y-%m-%d")
        else:
            target_date = st.date_input(config["date_label"], value=pd.Timestamp.today().date())
            target = target_date.strftime("%Y-%m-%d")
        default_lookback = {
            "daily": 123,
            "weekly": 52,
            "monthly": 12,
        }[config["timeframe"]]
        lookback = st.number_input("RS lookback periods", min_value=1, max_value=500, value=default_lookback, step=1)
        unit = {"daily": "trading days", "weekly": "weeks", "monthly": "months"}[config["timeframe"]]
        st.caption(f"Checks all {config['universe_label']} stocks against NIFTY using completed Heikin-Ashi candles.")
        st.caption(f"Lookback: compares performance with {lookback} {unit} earlier.")
    else:
        target = st.text_input(config["date_label"], value=config["default_date"])
        lookback = 123
        st.caption(f"Expected format: {config['date_format']}")
    run_button = st.button("Run fresh scan", type="primary", use_container_width=True)
    st.divider()
    st.caption("Data source: Yahoo Finance")

if run_button:
    with st.spinner("Downloading market data and calculating signals..."):
        try:
            if config.get("ichimoku"):
                signals = download_ichimoku(config, target)
                checked_period = signals.attrs.get("checked_period", "latest")
                report_path = save_ichimoku_excel(signals, checked_period) if not signals.empty else None
            elif config.get("rs"):
                signals = download_relative_strength(config, int(lookback), target)
                checked_period = signals.attrs.get("checked_period", "latest")
                report_path = save_rs_excel(signals, config, checked_period) if not signals.empty else None
            else:
                signals, report_path = run_scan(config, target.strip())
            st.session_state["signals"] = signals
            st.session_state["report_path"] = report_path
            st.session_state["scan_label"] = f"{mode}:{target}"
            st.success(f"Scan complete: {len(signals):,} signals")
        except subprocess.TimeoutExpired:
            st.error("The scan exceeded 15 minutes. Try a smaller universe or run the script directly.")
        except Exception as exc:
            st.error(str(exc))

scan_label = f"{mode}:{target}"
if "signals" in st.session_state and st.session_state.get("scan_label") == scan_label:
    signals = st.session_state["signals"]
    report_path = st.session_state.get("report_path")
else:
    signals, report_path = (pd.DataFrame(), None) if config.get("rs") or config.get("ichimoku") else load_report(config)
    st.session_state["signals"] = signals
    st.session_state["report_path"] = report_path
    st.session_state["scan_label"] = scan_label

if signals.empty:
    if config.get("ichimoku"):
        checked_period = signals.attrs.get("checked_period", "the selected month")
        st.info(f"No upward Ichimoku cloud event was found for {checked_period}.")
    elif config.get("rs"):
        checked_period = signals.attrs.get("checked_period", "the latest completed period")
        period_label = {"daily": "day", "weekly": "week", "monthly": "month"}[config["timeframe"]]
        st.info(f"No {config['universe_label']} stock crossed above the RS zero line during {checked_period}. The scan checks only the latest completed {period_label}.")
    else:
        st.warning("No saved report is available for this scanner. Use Run fresh scan to create one.")
    st.stop()

bullish_count = metric_value(signals, "Signal Type", "Bullish Reversal (Long)")
bearish_count = metric_value(signals, "Signal Type", "Bearish Reversal (Short)")
max_score = signals["Score"].max() if "Score" in signals else 0

first_row = st.columns(4)
first_row[0].metric("Signals", f"{len(signals):,}")
if config.get("ichimoku"):
    first_row[1].metric("Events", "Upward only")
    first_row[2].metric("Universe checked", config["universe_label"])
    first_row[3].metric("Timeframe", "Monthly")
elif config.get("rs"):
    first_row[1].metric("Crossover", "Above zero")
    first_row[2].metric("Universe checked", config["universe_label"])
    first_row[3].metric("Timeframe", config["timeframe"].title())
else:
    first_row[1].metric("Long reversals", f"{bullish_count:,}")
    first_row[2].metric("Short reversals", f"{bearish_count:,}")
    first_row[3].metric("Best score", f"{max_score:.1f}")

if report_path:
    st.caption(f"Loaded report: {report_path.name}")

note = f"Showing upward Ichimoku events for {config['universe_label']} using completed monthly candles."
if config.get("rs"):
    note = f"Showing {config['universe_label']} stocks with Relative Strength crossing above zero on the latest completed candle."
if not config.get("rs") and not config.get("ichimoku"):
    note = f"Displaying {config.get('universe_label', 'market')} Heikin-Ashi reversal signals ranked by setup score."
st.markdown(f'<div class="signal-note">{note}</div>', unsafe_allow_html=True)


signals = signals.copy()
with st.expander("Filter signals", expanded=True):
    filters = st.columns(3)
    if config.get("ichimoku"):
        type_column = "Signal"
        signal_types = filters[0].multiselect("Event", sorted(signals[type_column].dropna().unique()), default=sorted(signals[type_column].dropna().unique()))
        industries = filters[1].multiselect("Sector", sorted(signals["Sector"].dropna().unique()), default=[])
        industry_column = "Sector"
        minimum_score = 0.0
    elif config.get("rs"):
        filters[0].info("Only upward zero-line crossovers are shown.")
        industries = []
        industry_column = "Industry"
        minimum_score = 0.0
        type_column = "Stock"
        signal_types = signals["Stock"].tolist()
    else:
        type_column = "Signal Type"
        signal_types = filters[0].multiselect("Signal", sorted(signals[type_column].dropna().unique()), default=sorted(signals[type_column].dropna().unique()))
        industries = filters[1].multiselect("Industry", sorted(signals["Industry"].dropna().unique()), default=[])
        industry_column = "Industry"
        minimum_score = filters[2].slider("Minimum score", 0.0, 100.0, 0.0, 0.5)

filtered = signals[signals[type_column].isin(signal_types)]
if industries:
    filtered = filtered[filtered[industry_column].isin(industries)]
if "Score" in filtered:
    filtered = filtered[filtered["Score"] >= minimum_score]

left, right = st.columns([1.45, 1])
with left:
    st.subheader("Ranked signals")
    display_columns = [
        "Rank", "Stock", "Company Name", "Industry", "Signal Type", "Score", "Streak Count",
        "Move %", "Risk %", "Entry", "SL", "Target 1", "Target 2",
    ]
    if config.get("ichimoku"):
        display_columns = [
            "Stock", "Sector", "Signal", "Signal Period", "Previous Close", "Period Close",
            "Previous Cloud Top", "Cloud Top", "Cloud Bottom", "Distance From Cloud %",
        ]
    elif config.get("rs"):
        display_columns = ["Stock", "Sector", "Previous RS", "Current RS"]
    display_columns = [column for column in display_columns if column in filtered]
    st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)
    download_name = f"Ichimoku_{config['universe_label'].replace(' ', '')}_{config['timeframe']}_signals.csv" if config.get("ichimoku") else (f"RS_{config['universe_label'].replace(' ', '')}_{config['timeframe']}_crossovers.csv" if config.get("rs") else "nse_signal_results.csv")
    st.download_button("Download filtered CSV", filtered.to_csv(index=False), download_name, "text/csv")

with right:
    st.subheader("Signal detail")
    if filtered.empty:
        st.info("No signals match the current filters.")
    elif config.get("ichimoku"):
        selected_stock = st.selectbox("Select stock", filtered["Stock"].tolist())
        selected = filtered[filtered["Stock"] == selected_stock].iloc[0]
        st.metric("Event", selected["Signal"])
        st.metric("Distance from cloud", f"{selected['Distance From Cloud %']:.2f}%")
        st.caption(f"Sector: {selected['Sector']}")
    elif config.get("rs"):
        selected_stock = st.selectbox("Select stock", filtered["Stock"].tolist())
        selected = filtered[filtered["Stock"] == selected_stock].iloc[0]
        st.metric("Previous RS", f"{selected['Previous RS']:.2f}")
        st.metric("Current RS", f"{selected['Current RS']:.2f}")
        st.caption(f"Sector: {selected['Sector']}")
        st.caption("RS values are decimals. Example: 0.24 means approximately 24% relative outperformance.")
    else:
        selected_stock = st.selectbox("Select stock", filtered["Stock"].tolist())
        selected = filtered[filtered["Stock"] == selected_stock].iloc[0]
        detail_columns = ["Signal Type", "Score", "Streak Count", "Entry", "SL", "Risk", "Target 1", "Target 2", "Reversal Date"]
        details = {column: selected[column] for column in detail_columns if column in selected and pd.notna(selected[column])}
        st.json(details)

        if st.checkbox("Load price chart", value=False):
            ticker = f"{selected_stock}.NS"
            history = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
            if not history.empty:
                if isinstance(history.columns, pd.MultiIndex):
                    history.columns = history.columns.get_level_values(0)
                chart = go.Figure(go.Candlestick(x=history.index, open=history["Open"], high=history["High"], low=history["Low"], close=history["Close"]))
                chart.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("No price history returned for this symbol.")

st.divider()
with st.expander("Strategy Backtester", expanded=False):
    st.markdown("Run a walk-forward simulation to evaluate **Win Rate %**, **Profit Factor**, **Expectancy (R)**, and **Max Drawdown**.")
    bt_cols = st.columns(5)
    bt_universe = bt_cols[0].selectbox("Universe", ["NIFTY 50", "NSE Futures", "Selected Stock Only"])
    bt_timeframe = bt_cols[1].selectbox("Timeframe", ["weekly", "monthly", "daily"])
    bt_period = bt_cols[2].selectbox("Lookback", ["2y", "4y", "max"], index=1)
    bt_target = bt_cols[3].selectbox("Target R", [1.0, 1.5, 2.0], index=0)
    bt_direction = bt_cols[4].selectbox("Direction", ["both", "long", "short"])

    if st.button("Run Strategy Backtest", type="primary"):
        with st.spinner("Running historical backtest simulation..."):

            bt_engine = ReversalBacktester(streak_min=3, target_multiple=bt_target, max_holding_bars=12)
            if bt_universe == "Selected Stock Only":
                test_tickers = [f"{selected_stock}.NS"] if "selected_stock" in locals() else ["RELIANCE.NS"]
            elif bt_universe == "NIFTY 50":
                uni_df = pd.read_csv(BASE_DIR / "ind_nifty50list.csv")
                test_tickers = (uni_df["Symbol"] + ".NS").tolist()
            else:
                uni_df = pd.read_csv(BASE_DIR / "ind_nifty_futures_list.csv")
                test_tickers = uni_df["YahooSymbol"].tolist() if "YahooSymbol" in uni_df.columns else (uni_df["Symbol"] + ".NS").tolist()

            raw_bt = yf.download(test_tickers, period=bt_period, interval="1d", auto_adjust=False, progress=False)
            all_sim_trades = []
            for t_sym in test_tickers:
                if isinstance(raw_bt.columns, pd.MultiIndex):
                    if t_sym in raw_bt.columns.get_level_values(0):
                        stock_df = raw_bt[t_sym].dropna(how="all")
                    elif t_sym in raw_bt.columns.get_level_values(1):
                        stock_df = raw_bt.xs(t_sym, axis=1, level=1).dropna(how="all")
                    else:
                        continue
                else:
                    stock_df = raw_bt.dropna(how="all")
                if stock_df.empty or len(stock_df) < 50:
                    continue

                resampled_df = resample_ohlc(stock_df, timeframe=bt_timeframe)
                if len(resampled_df) < 20:
                    continue

                t_res = bt_engine.backtest_series(resampled_df, symbol=t_sym.replace(".NS", ""), direction=bt_direction)
                if not t_res.empty:
                    all_sim_trades.append(t_res)

            if all_sim_trades:
                sim_trades_df = pd.concat(all_sim_trades, ignore_index=True)
                sim_metrics = bt_engine.calculate_metrics(sim_trades_df)

                m_cols = st.columns(6)
                m_cols[0].metric("Total Trades", sim_metrics["Total Trades"])
                m_cols[1].metric("Win Rate %", f"{sim_metrics['Win Rate %']}%")
                m_cols[2].metric("Profit Factor", sim_metrics["Profit Factor"])
                m_cols[3].metric("Expectancy (R)", f"{sim_metrics['Expectancy (R)']} R")
                m_cols[4].metric("Total PnL (R)", f"{sim_metrics['Total PnL (R)']} R")
                m_cols[5].metric("Max Drawdown (R)", f"{sim_metrics['Max Drawdown (R)']} R")

                # Cumulative Equity Curve (R)
                sim_trades_df["Cumulative R"] = sim_trades_df["PnL (R)"].cumsum()
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(
                    y=sim_trades_df["Cumulative R"],
                    mode="lines",
                    line=dict(color="#0b6e4f", width=2),
                    name="Cumulative R",
                ))
                eq_fig.update_layout(
                    title="Cumulative Equity Curve (R-Multiples)",
                    xaxis_title="Trade Number",
                    yaxis_title="Total R Accumulated",
                    height=360,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(eq_fig, use_container_width=True)

                st.subheader("Simulated Trade Log")
                st.dataframe(sim_trades_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Backtest Trade Log (CSV)",
                    sim_trades_df.to_csv(index=False),
                    f"backtest_{bt_timeframe}_{bt_target}R.csv",
                    "text/csv",
                )
            else:
                st.info("No trades triggered during the selected historical period.")

