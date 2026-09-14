from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh


# ============================================================
# OP.exe
# One-minute FVG setup detector and multi-timeframe analyzer
# ============================================================

st.set_page_config(
    page_title="OP.exe",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIMEFRAMES = {
    "1m": {"period": "7d", "interval": "1m", "resample": None},
    "5m": {"period": "60d", "interval": "5m", "resample": None},
    "15m": {"period": "60d", "interval": "15m", "resample": None},
    "30m": {"period": "60d", "interval": "30m", "resample": None},
    "1H": {"period": "730d", "interval": "60m", "resample": None},
    "4H": {"period": "730d", "interval": "60m", "resample": "4h"},
    "1D": {"period": "5y", "interval": "1d", "resample": None},
    "1W": {"period": "10y", "interval": "1wk", "resample": None},
}

TF_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1H": 60,
    "4H": 240,
    "1D": 1440,
    "1W": 10080,
}

# Yahoo Finance continuous-futures proxies.
# Exact TradingView and Topstep contract prices may differ.
SYMBOL_ALIASES = {
    "MNQ": "NQ=F",
    "NQ": "NQ=F",
    "MES": "ES=F",
    "ES": "ES=F",
    "M2K": "RTY=F",
    "RTY": "RTY=F",
    "MYM": "YM=F",
    "YM": "YM=F",
    "MGC": "GC=F",
    "GC": "GC=F",
    "MCL": "CL=F",
    "CL": "CL=F",
    "SPX": "^GSPC",
}


@dataclass
class FVG:
    direction: str
    bottom: float
    midpoint: float
    top: float
    created_at: pd.Timestamp
    filled: bool


@dataclass
class Analysis:
    timeframe: str
    signal: str
    direction: str
    confidence: float
    score: float
    price: Optional[float]
    target: Optional[float]
    stop: Optional[float]
    risk_points: Optional[float]
    reward_points: Optional[float]
    rr: Optional[float]
    fvg: Optional[FVG]
    boundary_touched: bool
    confirmation: bool
    setup_formed: bool
    freshness: str
    reasons: list[str]
    warnings: list[str]


def clean_symbol(raw_symbol: str) -> str:
    symbol = raw_symbol.upper().strip().replace(" ", "")
    return SYMBOL_ALIASES.get(symbol, symbol)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    frame = df.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in frame.columns
        ]

    rename_map = {}

    for column in frame.columns:
        name = str(column).strip().lower()

        if name == "open":
            rename_map[column] = "Open"
        elif name == "high":
            rename_map[column] = "High"
        elif name == "low":
            rename_map[column] = "Low"
        elif name == "close":
            rename_map[column] = "Close"
        elif name == "volume":
            rename_map[column] = "Volume"

    frame = frame.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close"]

    if not all(column in frame.columns for column in required):
        return pd.DataFrame()

    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0

    frame = frame[["Open", "High", "Low", "Close", "Volume"]]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=required)

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(
            frame.index,
            utc=True,
            errors="coerce",
        )
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")

    frame = frame[
        ~frame.index.isna()
    ]

    frame = frame[
        ~frame.index.duplicated(keep="last")
    ].sort_index()

    return frame


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df

    return (
        df.resample(
            rule,
            label="right",
            closed="right",
        )
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )


@st.cache_data(ttl=20, show_spinner=False)
def download_yahoo(
    symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    frame = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=True,
        progress=False,
        threads=False,
    )

    return normalize_ohlcv(frame)


@st.cache_data(ttl=10, show_spinner=False)
def download_live_bridge(
    url: str,
    token: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    headers = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "timeframe": timeframe,
        },
        headers=headers,
        timeout=12,
    )

    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict):
        candles = payload.get("candles", [])
    else:
        candles = payload

    frame = pd.DataFrame(candles)

    rename_map = {
        "time": "Datetime",
        "datetime": "Datetime",
        "timestamp": "Datetime",
        "date": "Datetime",
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }

    frame = frame.rename(
        columns={
            column: rename_map.get(
                str(column).strip().lower(),
                column,
            )
            for column in frame.columns
        }
    )

    if "Datetime" not in frame.columns:
        raise ValueError(
            "Live endpoint response requires a time, datetime, "
            "timestamp, or date field."
        )

    frame["Datetime"] = pd.to_datetime(
        frame["Datetime"],
        utc=True,
        errors="coerce",
    )

    frame = frame.set_index("Datetime")
    return normalize_ohlcv(frame)


def read_tradingview_csv(uploaded_file) -> pd.DataFrame:
    frame = pd.read_csv(
        io.BytesIO(uploaded_file.getvalue())
    )

    lowercase_columns = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    time_column = None

    for candidate in [
        "time",
        "datetime",
        "timestamp",
        "date",
    ]:
        if candidate in lowercase_columns:
            time_column = lowercase_columns[candidate]
            break

    if time_column is None:
        raise ValueError(
            "The TradingView CSV requires a time, datetime, "
            "timestamp, or date column."
        )

    frame[time_column] = pd.to_datetime(
        frame[time_column],
        utc=True,
        errors="coerce",
    )

    frame = frame.set_index(time_column)
    return normalize_ohlcv(frame)


def calculate_atr(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / length,
        adjust=False,
    ).mean()


def calculate_rsi(
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / length,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / length,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss.replace(
        0,
        np.nan,
    )

    return 100 - (
        100 / (1 + relative_strength)
    )


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    frame["SMA20"] = frame["Close"].rolling(20).mean()
    frame["SMA50"] = frame["Close"].rolling(50).mean()
    frame["ATR"] = calculate_atr(frame)
    frame["RSI"] = calculate_rsi(frame["Close"])
    frame["VolumeMA20"] = frame["Volume"].rolling(20).mean()

    # Prior structure excludes the current candle.
    frame["PriorSwingHigh"] = (
        frame["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    frame["PriorSwingLow"] = (
        frame["Low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    return frame


def find_fvgs(
    df: pd.DataFrame,
    lookback: int = 300,
) -> list[FVG]:
    frame = df.tail(lookback)
    gaps: list[FVG] = []

    for current_index in range(2, len(frame)):
        first_candle = frame.iloc[current_index - 2]
        third_candle = frame.iloc[current_index]
        created_at = frame.index[current_index]

        # Bullish FVG:
        # third candle's low is above first candle's high.
        if third_candle["Low"] > first_candle["High"]:
            bottom = float(first_candle["High"])
            top = float(third_candle["Low"])

            future = frame.iloc[current_index + 1 :]

            filled = (
                bool((future["Low"] <= bottom).any())
                if not future.empty
                else False
            )

            gaps.append(
                FVG(
                    direction="bullish",
                    bottom=bottom,
                    midpoint=(bottom + top) / 2,
                    top=top,
                    created_at=created_at,
                    filled=filled,
                )
            )

        # Bearish FVG:
        # third candle's high is below first candle's low.
        if third_candle["High"] < first_candle["Low"]:
            bottom = float(third_candle["High"])
            top = float(first_candle["Low"])

            future = frame.iloc[current_index + 1 :]

            filled = (
                bool((future["High"] >= top).any())
                if not future.empty
                else False
            )

            gaps.append(
                FVG(
                    direction="bearish",
                    bottom=bottom,
                    midpoint=(bottom + top) / 2,
                    top=top,
                    created_at=created_at,
                    filled=filled,
                )
            )

    return gaps


def select_nearest_fvg(
    df: pd.DataFrame,
    direction: str,
) -> Optional[FVG]:
    current_price = float(df["Close"].iloc[-1])

    valid_gaps = [
        gap
        for gap in find_fvgs(df)
        if gap.direction == direction
        and not gap.filled
    ]

    if not valid_gaps:
        return None

    recent_gaps = valid_gaps[-30:]

    return min(
        recent_gaps,
        key=lambda gap: abs(
            current_price - gap.midpoint
        ),
    )


def is_green_hammer(candle: pd.Series) -> bool:
    candle_range = max(
        float(candle["High"] - candle["Low"]),
        1e-12,
    )

    body = abs(
        float(candle["Close"] - candle["Open"])
    )

    lower_wick = (
        min(candle["Open"], candle["Close"])
        - candle["Low"]
    )

    upper_wick = (
        candle["High"]
        - max(candle["Open"], candle["Close"])
    )

    return bool(
        candle["Close"] > candle["Open"]
        and lower_wick >= max(
            body * 1.8,
            candle_range * 0.45,
        )
        and upper_wick <= candle_range * 0.25
    )


def is_red_shooting_star(
    candle: pd.Series,
) -> bool:
    candle_range = max(
        float(candle["High"] - candle["Low"]),
        1e-12,
    )

    body = abs(
        float(candle["Close"] - candle["Open"])
    )

    upper_wick = (
        candle["High"]
        - max(candle["Open"], candle["Close"])
    )

    lower_wick = (
        min(candle["Open"], candle["Close"])
        - candle["Low"]
    )

    return bool(
        candle["Close"] < candle["Open"]
        and upper_wick >= max(
            body * 1.8,
            candle_range * 0.45,
        )
        and lower_wick <= candle_range * 0.25
    )


def confirmation_present(
    df: pd.DataFrame,
    direction: str,
) -> tuple[bool, str]:
    if len(df) < 2:
        return False, "Insufficient candles"

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    if direction == "bullish":
        two_bullish = bool(
            previous["Close"] > previous["Open"]
            and latest["Close"] > latest["Open"]
        )

        if is_green_hammer(latest):
            return True, "Green hammer"

        if two_bullish:
            return True, "Two consecutive bullish candles"

        return False, "No bullish confirmation"

    two_bearish = bool(
        previous["Close"] < previous["Open"]
        and latest["Close"] < latest["Open"]
    )

    if is_red_shooting_star(latest):
        return True, "Red shooting-star candle"

    if two_bearish:
        return True, "Two consecutive bearish candles"

    return False, "No bearish confirmation"


def fvg_boundary_touched(
    df: pd.DataFrame,
    gap: Optional[FVG],
    direction: str,
) -> bool:
    if gap is None:
        return False

    latest = df.iloc[-1]
    current_atr = float(latest["ATR"])

    gap_size = max(
        gap.top - gap.bottom,
        1e-12,
    )

    tolerance = max(
        current_atr * 0.05,
        gap_size * 0.10,
    )

    if direction == "bullish":
        # User's LONG rule:
        # touch the bottom bullish FVG boundary.
        return bool(
            latest["Low"] <= gap.bottom + tolerance
            and latest["High"] >= gap.bottom - tolerance
        )

    # User's SHORT rule:
    # touch the top bearish FVG boundary.
    return bool(
        latest["High"] >= gap.top - tolerance
        and latest["Low"] <= gap.top + tolerance
    )


def data_freshness(
    df: pd.DataFrame,
    timeframe: str,
) -> tuple[str, float]:
    if df.empty:
        return "MISSING", float("inf")

    latest_time = df.index[-1]

    if latest_time.tzinfo is None:
        latest_time = latest_time.tz_localize("UTC")
    else:
        latest_time = latest_time.tz_convert("UTC")

    age_minutes = (
        pd.Timestamp.now(tz="UTC") - latest_time
    ).total_seconds() / 60

    allowed_age = TF_MINUTES[timeframe] * 2.5

    # Allow for closed markets and higher-timeframe candles.
    if timeframe == "1D":
        allowed_age = 2160

    elif timeframe == "1W":
        allowed_age = 12960

    status = (
        "CURRENT"
        if age_minutes <= allowed_age
        else "DELAYED"
    )

    return status, age_minutes


def determine_trade_levels(
    df: pd.DataFrame,
    direction: str,
    gap: Optional[FVG],
) -> tuple[float, float]:
    latest = df.iloc[-1]
    price = float(latest["Close"])
    current_atr = float(latest["ATR"])
    recent = df.tail(80)

    if direction == "bullish":
        possible_highs = recent.loc[
            recent["High"] > price,
            "High",
        ]

        if not possible_highs.empty:
            target = float(possible_highs.max())
        else:
            target = price + current_atr * 2

        if gap:
            structural_stop = (
                gap.bottom - current_atr * 0.20
            )
        else:
            structural_stop = float(
                recent["Low"].tail(10).min()
            )

        stop = min(
            structural_stop,
            price - current_atr * 0.35,
        )

        return target, stop

    possible_lows = recent.loc[
        recent["Low"] < price,
        "Low",
    ]

    if not possible_lows.empty:
        target = float(possible_lows.min())
    else:
        target = price - current_atr * 2

    if gap:
        structural_stop = (
            gap.top + current_atr * 0.20
        )
    else:
        structural_stop = float(
            recent["High"].tail(10).max()
        )

    stop = max(
        structural_stop,
        price + current_atr * 0.35,
    )

    return target, stop


def analyze_timeframe(
    raw_df: pd.DataFrame,
    timeframe: str,
    catalyst: str,
    strict_mode: bool,
) -> Analysis:
    reasons: list[str] = []
    warnings: list[str] = []

    if raw_df.empty or len(raw_df) < 55:
        one_minute_signal = (
            "SETUP NOT FORMED"
            if timeframe == "1m"
            else "WAIT"
        )

        return Analysis(
            timeframe=timeframe,
            signal=one_minute_signal,
            direction="neutral",
            confidence=0,
            score=0,
            price=None,
            target=None,
            stop=None,
            risk_points=None,
            reward_points=None,
            rr=None,
            fvg=None,
            boundary_touched=False,
            confirmation=False,
            setup_formed=False,
            freshness="MISSING",
            reasons=["Insufficient candle history"],
            warnings=[
                "No reliable signal can be calculated."
            ],
        )

    df = add_indicators(raw_df).dropna(
        subset=["SMA20", "SMA50", "ATR"]
    )

    if len(df) < 2:
        return analyze_timeframe(
            pd.DataFrame(),
            timeframe,
            catalyst,
            strict_mode,
        )

    latest = df.iloc[-1]
    price = float(latest["Close"])
    current_atr = max(float(latest["ATR"]), 1e-12)

    bull_score = 0.0
    bear_score = 0.0

    freshness, age_minutes = data_freshness(
        df,
        timeframe,
    )

    if freshness == "DELAYED":
        warnings.append(
            f"Latest candle is approximately "
            f"{age_minutes:.1f} minutes old."
        )

    bullish_gap = select_nearest_fvg(
        df,
        "bullish",
    )

    bearish_gap = select_nearest_fvg(
        df,
        "bearish",
    )

    long_touch = fvg_boundary_touched(
        df,
        bullish_gap,
        "bullish",
    )

    short_touch = fvg_boundary_touched(
        df,
        bearish_gap,
        "bearish",
    )

    long_confirmation, long_confirmation_name = (
        confirmation_present(
            df,
            "bullish",
        )
    )

    short_confirmation, short_confirmation_name = (
        confirmation_present(
            df,
            "bearish",
        )
    )

    # --------------------------------------------------------
    # Trend evidence
    # --------------------------------------------------------

    if (
        latest["Close"] > latest["SMA20"]
        and latest["SMA20"] > latest["SMA50"]
    ):
        bull_score += 2.0
        reasons.append(
            "Price, SMA20 and SMA50 are bullish"
        )

    elif (
        latest["Close"] < latest["SMA20"]
        and latest["SMA20"] < latest["SMA50"]
    ):
        bear_score += 2.0
        reasons.append(
            "Price, SMA20 and SMA50 are bearish"
        )

    elif latest["SMA20"] > latest["SMA50"]:
        bull_score += 0.75
        reasons.append("SMA trend leans bullish")

    elif latest["SMA20"] < latest["SMA50"]:
        bear_score += 0.75
        reasons.append("SMA trend leans bearish")

    # --------------------------------------------------------
    # Break of structure
    # --------------------------------------------------------

    if latest["Close"] > latest["PriorSwingHigh"]:
        bull_score += 2.0
        reasons.append("Bullish break of structure")

    elif latest["Close"] < latest["PriorSwingLow"]:
        bear_score += 2.0
        reasons.append("Bearish break of structure")

    # --------------------------------------------------------
    # Liquidity sweeps
    # --------------------------------------------------------

    prior_ten_high = (
        df["High"]
        .shift(1)
        .rolling(10)
        .max()
        .iloc[-1]
    )

    prior_ten_low = (
        df["Low"]
        .shift(1)
        .rolling(10)
        .min()
        .iloc[-1]
    )

    if (
        latest["Low"] < prior_ten_low
        and latest["Close"] > prior_ten_low
    ):
        bull_score += 1.25
        reasons.append(
            "Sell-side liquidity sweep and reclaim"
        )

    if (
        latest["High"] > prior_ten_high
        and latest["Close"] < prior_ten_high
    ):
        bear_score += 1.25
        reasons.append(
            "Buy-side liquidity sweep and rejection"
        )

    # --------------------------------------------------------
    # Displacement and relative volume
    # --------------------------------------------------------

    candle_body = abs(
        float(latest["Close"] - latest["Open"])
    )

    displacement = (
        candle_body >= current_atr * 0.75
    )

    volume_confirmed = bool(
        latest["VolumeMA20"] > 0
        and latest["Volume"]
        >= latest["VolumeMA20"] * 1.20
    )

    if (
        displacement
        and latest["Close"] > latest["Open"]
    ):
        bull_score += (
            1.25 if volume_confirmed else 0.75
        )
        reasons.append("Bullish displacement")

    if (
        displacement
        and latest["Close"] < latest["Open"]
    ):
        bear_score += (
            1.25 if volume_confirmed else 0.75
        )
        reasons.append("Bearish displacement")

    # --------------------------------------------------------
    # Momentum support
    # --------------------------------------------------------

    if 52 <= latest["RSI"] <= 72:
        bull_score += 0.50
        reasons.append("Momentum supports LONG")

    elif 28 <= latest["RSI"] <= 48:
        bear_score += 0.50
        reasons.append("Momentum supports SHORT")

    # --------------------------------------------------------
    # Fresh FVG proximity
    # --------------------------------------------------------

    if bullish_gap:
        bullish_distance = abs(
            price - bullish_gap.midpoint
        )

        if bullish_distance <= current_atr * 1.5:
            bull_score += 1.0
            reasons.append(
                "Price is near a fresh bullish FVG"
            )

    if bearish_gap:
        bearish_distance = abs(
            price - bearish_gap.midpoint
        )

        if bearish_distance <= current_atr * 1.5:
            bear_score += 1.0
            reasons.append(
                "Price is near a fresh bearish FVG"
            )

    # --------------------------------------------------------
    # User's exact one-minute FVG rules
    # --------------------------------------------------------

    if long_touch:
        bull_score += 2.5
        reasons.append(
            "Bullish FVG bottom boundary touched"
        )

    if short_touch:
        bear_score += 2.5
        reasons.append(
            "Bearish FVG top boundary touched"
        )

    if long_confirmation:
        bull_score += 1.75
        reasons.append(long_confirmation_name)

    if short_confirmation:
        bear_score += 1.75
        reasons.append(short_confirmation_name)

    # --------------------------------------------------------
    # Manually verified catalyst
    # --------------------------------------------------------

    if catalyst == "Bullish":
        bull_score += 1.0
        reasons.append("Verified bullish catalyst")

    elif catalyst == "Bearish":
        bear_score += 1.0
        reasons.append("Verified bearish catalyst")

    net_score = bull_score - bear_score
    total_score = bull_score + bear_score

    threshold = 4.75 if strict_mode else 4.0
    conflict_limit = 3.0 if strict_mode else 3.75

    direction = "neutral"
    selected_gap = None
    selected_touch = False
    selected_confirmation = False
    setup_formed = False

    # --------------------------------------------------------
    # One-minute output
    # --------------------------------------------------------

    if timeframe == "1m":
        bullish_setup = bool(
            bullish_gap is not None
            and long_touch
            and long_confirmation
            and net_score >= threshold
            and bear_score < conflict_limit
        )

        bearish_setup = bool(
            bearish_gap is not None
            and short_touch
            and short_confirmation
            and net_score <= -threshold
            and bull_score < conflict_limit
        )

        if bullish_setup:
            signal = "SETUP FORMED — LONG"
            direction = "bullish"
            selected_gap = bullish_gap
            selected_touch = True
            selected_confirmation = True
            setup_formed = True

        elif bearish_setup:
            signal = "SETUP FORMED — SHORT"
            direction = "bearish"
            selected_gap = bearish_gap
            selected_touch = True
            selected_confirmation = True
            setup_formed = True

        else:
            signal = "SETUP NOT FORMED"

            if long_touch and not long_confirmation:
                warnings.append(
                    "Bullish boundary touched, but the "
                    "required confirmation has not formed."
                )

            elif short_touch and not short_confirmation:
                warnings.append(
                    "Bearish boundary touched, but the "
                    "required confirmation has not formed."
                )

            elif long_confirmation and not long_touch:
                warnings.append(
                    "Bullish confirmation exists, but the "
                    "bullish FVG bottom was not touched."
                )

            elif short_confirmation and not short_touch:
                warnings.append(
                    "Bearish confirmation exists, but the "
                    "bearish FVG top was not touched."
                )

            else:
                warnings.append(
                    "FVG touch and candle confirmation "
                    "have not formed together."
                )

    # --------------------------------------------------------
    # Higher-timeframe output
    # --------------------------------------------------------

    else:
        if (
            net_score >= threshold
            and bear_score < conflict_limit
        ):
            signal = "LONG"
            direction = "bullish"
            selected_gap = bullish_gap
            selected_touch = long_touch
            selected_confirmation = long_confirmation

        elif (
            net_score <= -threshold
            and bull_score < conflict_limit
        ):
            signal = "SHORT"
            direction = "bearish"
            selected_gap = bearish_gap
            selected_touch = short_touch
            selected_confirmation = short_confirmation

        else:
            signal = "WAIT"

    # --------------------------------------------------------
    # Block delayed lower-timeframe decisions
    # --------------------------------------------------------

    if freshness == "DELAYED":
        if timeframe == "1m":
            signal = "SETUP NOT FORMED"
            direction = "neutral"
            setup_formed = False
            selected_gap = None
            warnings.append(
                "The setup was blocked because the "
                "one-minute data is delayed."
            )

        elif timeframe == "5m":
            signal = "WAIT"
            direction = "neutral"
            warnings.append(
                "The signal was blocked because the "
                "five-minute data is delayed."
            )

    target = None
    stop = None
    risk_points = None
    reward_points = None
    rr_value = None

    actionable = (
        signal in {
            "LONG",
            "SHORT",
            "SETUP FORMED — LONG",
            "SETUP FORMED — SHORT",
        }
    )

    if actionable:
        target, stop = determine_trade_levels(
            df,
            direction,
            selected_gap,
        )

        risk_points = abs(price - stop)
        reward_points = abs(target - price)

        if risk_points > 0:
            rr_value = reward_points / risk_points

        if rr_value is None or rr_value < 1.20:
            warnings.append(
                "Projected reward-to-risk is below 1.20."
            )

            if strict_mode:
                if timeframe == "1m":
                    signal = "SETUP NOT FORMED"
                    direction = "neutral"
                    setup_formed = False
                else:
                    signal = "WAIT"
                    direction = "neutral"

    if total_score > 0:
        confidence = min(
            95.0,
            50.0
            + abs(net_score)
            / max(total_score, 1)
            * 35.0,
        )
    else:
        confidence = 0.0

    if signal in {
        "WAIT",
        "SETUP NOT FORMED",
    }:
        confidence = min(confidence, 69.0)

    return Analysis(
        timeframe=timeframe,
        signal=signal,
        direction=direction,
        confidence=round(confidence, 1),
        score=round(net_score, 2),
        price=price,
        target=target,
        stop=stop,
        risk_points=risk_points,
        reward_points=reward_points,
        rr=rr_value,
        fvg=selected_gap,
        boundary_touched=selected_touch,
        confirmation=selected_confirmation,
        setup_formed=setup_formed,
        freshness=freshness,
        reasons=reasons,
        warnings=warnings,
    )


def format_price(
    value: Optional[float],
) -> str:
    if value is None or pd.isna(value):
        return "—"

    return f"{value:,.2f}"


def build_chart(
    df: pd.DataFrame,
    timeframe: str,
    analysis: Analysis,
) -> go.Figure:
    display = add_indicators(df).tail(180)

    chart = go.Figure()

    chart.add_trace(
        go.Candlestick(
            x=display.index,
            open=display["Open"],
            high=display["High"],
            low=display["Low"],
            close=display["Close"],
            name="Candles",
            increasing_line_color="#00D084",
            decreasing_line_color="#FF4B4B",
        )
    )

    chart.add_trace(
        go.Scatter(
            x=display.index,
            y=display["SMA20"],
            name="SMA 20",
            line={
                "color": "#4DA3FF",
                "width": 1.4,
            },
        )
    )

    chart.add_trace(
        go.Scatter(
            x=display.index,
            y=display["SMA50"],
            name="SMA 50",
            line={
                "color": "#FFB020",
                "width": 1.4,
            },
        )
    )

    visible_gaps = [
        gap
        for gap in find_fvgs(display, 180)
        if not gap.filled
    ][-8:]

    for gap in visible_gaps:
        if gap.direction == "bullish":
            color = "rgba(0,208,132,0.14)"
        else:
            color = "rgba(255,75,75,0.14)"

        chart.add_hrect(
            y0=gap.bottom,
            y1=gap.top,
            fillcolor=color,
            line_width=0,
        )

        chart.add_hline(
            y=gap.midpoint,
            line_dash="dot",
            line_color="#9AA0A6",
            opacity=0.55,
        )

    if analysis.target is not None:
        chart.add_hline(
            y=analysis.target,
            line_color="#00D084",
            line_dash="dash",
            annotation_text="Swing target",
        )

    if analysis.stop is not None:
        chart.add_hline(
            y=analysis.stop,
            line_color="#FF4B4B",
            line_dash="dash",
            annotation_text="Structural stop",
        )

    chart.update_layout(
        template="plotly_dark",
        height=620,
        title=(
            f"{timeframe} candles, SMAs and fresh FVGs"
        ),
        xaxis_rangeslider_visible=False,
        margin={
            "l": 20,
            "r": 20,
            "t": 55,
            "b": 20,
        },
    )

    return chart


# ============================================================
# INTERFACE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #273244;
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("OP.exe")
st.caption(
    "One-minute FVG setup detector and multi-timeframe "
    "market-structure analysis"
)

with st.sidebar:
    st.header("Market")

    symbol_input = st.text_input(
        "Ticker",
        value="MNQ",
        help=(
            "Examples: MNQ, MES, M2K, MGC, AAPL, "
            "NVDA, QQQ, SPY, BTC-USD"
        ),
    )

    yahoo_symbol = clean_symbol(symbol_input)

    data_source = st.selectbox(
        "Candle-data source",
        [
            "Yahoo Finance",
            "TradingView CSV",
            "Live candle bridge",
        ],
    )

    catalyst_selection = st.selectbox(
        "Verified catalyst",
        [
            "Neutral or unknown",
            "Bullish",
            "Bearish",
        ],
        help=(
            "Only select bullish or bearish after "
            "confirming a real catalyst."
        ),
    )

    catalyst = {
        "Neutral or unknown": "Neutral",
        "Bullish": "Bullish",
        "Bearish": "Bearish",
    }[catalyst_selection]

    strict_mode = st.toggle(
        "Strict filtering",
        value=True,
        help=(
            "Requires stronger confluence and produces "
            "more WAIT or SETUP NOT FORMED results."
        ),
    )

    auto_refresh = st.toggle(
        "Automatically refresh",
        value=False,
    )

    refresh_seconds = st.select_slider(
        "Refresh every",
        options=[15, 30, 60, 120, 300],
        value=30,
        disabled=not auto_refresh,
        format_func=lambda value: f"{value} seconds",
    )

    if (
        auto_refresh
        and data_source != "TradingView CSV"
    ):
        st_autorefresh(
            interval=refresh_seconds * 1000,
            key="opexe_auto_refresh",
        )

    uploaded_csv = None
    uploaded_timeframe = "1m"
    bridge_url = ""
    bridge_token = ""

    if data_source == "TradingView CSV":
        uploaded_csv = st.file_uploader(
            "TradingView CSV file",
            type=["csv"],
        )

        uploaded_timeframe = st.selectbox(
            "CSV candle timeframe",
            list(TIMEFRAMES.keys()),
        )

    elif data_source == "Live candle bridge":
        bridge_url = st.text_input(
            "HTTPS candle endpoint",
            placeholder=(
                "https://your-domain.com/candles"
            ),
        )

        bridge_token = st.text_input(
            "Private bearer token",
            type="password",
        )

    st.divider()
    st.header("Risk model")

    contracts = st.number_input(
        "Number of contracts",
        min_value=1,
        max_value=100,
        value=3,
    )

    point_value = st.number_input(
        "Dollar value per point per contract",
        min_value=0.01,
        value=2.00,
        step=0.25,
        help=(
            "Verify this with your broker. "
            "MNQ is commonly $2 per index point."
        ),
    )

    maximum_risk = st.number_input(
        "Maximum combined planned loss",
        min_value=1.0,
        value=120.0,
        step=10.0,
    )

    if st.button(
        "Refresh analysis",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.warning(
        "OP.exe does not place orders. Always verify "
        "price, contract, stop and target before trading."
    )


# ============================================================
# LOAD DATA
# ============================================================

frames: dict[str, pd.DataFrame] = {}
data_errors: list[str] = []

with st.spinner(
    "Loading candles and calculating OP.exe signals..."
):
    try:
        if data_source == "Yahoo Finance":
            for timeframe, settings in TIMEFRAMES.items():
                frame = download_yahoo(
                    yahoo_symbol,
                    settings["period"],
                    settings["interval"],
                )

                if settings["resample"]:
                    frame = resample_ohlcv(
                        frame,
                        settings["resample"],
                    )

                frames[timeframe] = frame

        elif data_source == "TradingView CSV":
            if uploaded_csv is None:
                st.info(
                    "Upload your TradingView CSV file "
                    "to begin."
                )
                st.stop()

            source_frame = read_tradingview_csv(
                uploaded_csv
            )

            frames[uploaded_timeframe] = source_frame

            if uploaded_timeframe == "1m":
                resample_rules = {
                    "5m": "5min",
                    "15m": "15min",
                    "30m": "30min",
                    "1H": "1h",
                    "4H": "4h",
                    "1D": "1D",
                    "1W": "1W",
                }

                for timeframe, rule in (
                    resample_rules.items()
                ):
                    frames[timeframe] = (
                        resample_ohlcv(
                            source_frame,
                            rule,
                        )
                    )

        elif data_source == "Live candle bridge":
            if not bridge_url:
                st.info(
                    "Enter your HTTPS candle endpoint."
                )
                st.stop()

            for timeframe in TIMEFRAMES:
                try:
                    frames[timeframe] = (
                        download_live_bridge(
                            bridge_url,
                            bridge_token,
                            symbol_input,
                            timeframe,
                        )
                    )

                except Exception as error:
                    data_errors.append(
                        f"{timeframe}: {error}"
                    )

    except Exception as error:
        st.error(
            f"Candle data could not be loaded: {error}"
        )
        st.stop()


# ============================================================
# ANALYSIS
# ============================================================

analyses: dict[str, Analysis] = {}

for timeframe in TIMEFRAMES:
    frame = frames.get(
        timeframe,
        pd.DataFrame(),
    )

    analyses[timeframe] = analyze_timeframe(
        frame,
        timeframe,
        catalyst,
        strict_mode,
    )


# Block one-minute setups that directly oppose 15-minute
# directional context.
one_minute = analyses["1m"]
fifteen_minute = analyses["15m"]

if (
    one_minute.signal == "SETUP FORMED — LONG"
    and fifteen_minute.signal == "SHORT"
):
    one_minute.signal = "SETUP NOT FORMED"
    one_minute.direction = "neutral"
    one_minute.setup_formed = False
    one_minute.target = None
    one_minute.stop = None
    one_minute.risk_points = None
    one_minute.reward_points = None
    one_minute.rr = None

    one_minute.warnings.append(
        "Potential LONG setup blocked by directly "
        "opposing 15-minute market structure."
    )

elif (
    one_minute.signal == "SETUP FORMED — SHORT"
    and fifteen_minute.signal == "LONG"
):
    one_minute.signal = "SETUP NOT FORMED"
    one_minute.direction = "neutral"
    one_minute.setup_formed = False
    one_minute.target = None
    one_minute.stop = None
    one_minute.risk_points = None
    one_minute.reward_points = None
    one_minute.rr = None

    one_minute.warnings.append(
        "Potential SHORT setup blocked by directly "
        "opposing 15-minute market structure."
    )


# ============================================================
# RESULTS
# ============================================================

st.subheader(
    f"{symbol_input.upper()} analysis"
)

if (
    data_source == "Yahoo Finance"
    and yahoo_symbol
    != symbol_input.upper().strip().replace(" ", "")
):
    st.info(
        f"Yahoo mapping: {symbol_input.upper()} → "
        f"{yahoo_symbol}. This is a continuous-futures "
        "proxy and may differ from your exact TradingView "
        "or Topstep contract."
    )

if one_minute.signal == "SETUP FORMED — LONG":
    st.success(
        f"🟢 1-MINUTE: SETUP FORMED — LONG "
        f"({one_minute.confidence:.0f}% evidence)"
    )

elif one_minute.signal == "SETUP FORMED — SHORT":
    st.error(
        f"🔴 1-MINUTE: SETUP FORMED — SHORT "
        f"({one_minute.confidence:.0f}% evidence)"
    )

else:
    st.warning(
        f"🟡 1-MINUTE: SETUP NOT FORMED "
        f"({one_minute.confidence:.0f}% evidence)"
    )

st.subheader("Higher-timeframe direction")

higher_timeframes = [
    "5m",
    "15m",
    "30m",
    "1H",
    "4H",
    "1D",
    "1W",
]

metric_columns = st.columns(
    len(higher_timeframes)
)

for column, timeframe in zip(
    metric_columns,
    higher_timeframes,
):
    result = analyses[timeframe]

    column.metric(
        label=timeframe,
        value=result.signal,
        delta=(
            f"{result.confidence:.0f}% evidence"
        ),
    )


table_rows = []

for timeframe, result in analyses.items():
    table_rows.append(
        {
            "Timeframe": timeframe,
            "Result": result.signal,
            "Score": result.score,
            "Confidence": (
                f"{result.confidence:.1f}%"
            ),
            "Price": format_price(result.price),
            "Target": format_price(result.target),
            "Stop": format_price(result.stop),
            "R:R": (
                f"{result.rr:.2f}"
                if result.rr is not None
                else "—"
            ),
            "Data": result.freshness,
        }
    )

st.dataframe(
    pd.DataFrame(table_rows),
    hide_index=True,
    use_container_width=True,
)

if data_errors:
    with st.expander("Candle-feed errors"):
        for error in data_errors:
            st.warning(error)


# ============================================================
# CHART AND DETAILS
# ============================================================

left_column, right_column = st.columns(
    [2.2, 1]
)

with left_column:
    selected_timeframe = st.selectbox(
        "Chart timeframe",
        list(TIMEFRAMES.keys()),
    )

    selected_frame = frames.get(
        selected_timeframe,
        pd.DataFrame(),
    )

    selected_analysis = analyses[
        selected_timeframe
    ]

    if selected_frame.empty:
        st.warning(
            f"No {selected_timeframe} candles "
            "are available."
        )

    else:
        st.plotly_chart(
            build_chart(
                selected_frame,
                selected_timeframe,
                selected_analysis,
            ),
            use_container_width=True,
        )

with right_column:
    selected_analysis = analyses[
        selected_timeframe
    ]

    st.subheader("Selected analysis")

    st.metric(
        "Result",
        selected_analysis.signal,
    )

    st.metric(
        "Price",
        format_price(
            selected_analysis.price
        ),
    )

    st.metric(
        "Swing target",
        format_price(
            selected_analysis.target
        ),
    )

    st.metric(
        "Structural stop",
        format_price(
            selected_analysis.stop
        ),
    )

    st.metric(
        "Projected R:R",
        (
            f"{selected_analysis.rr:.2f}"
            if selected_analysis.rr is not None
            else "—"
        ),
    )

    if selected_analysis.risk_points:
        estimated_risk = (
            selected_analysis.risk_points
            * float(point_value)
            * int(contracts)
        )

        st.metric(
            "Estimated combined risk",
            f"${estimated_risk:,.2f}",
        )

        if estimated_risk > maximum_risk:
            st.error(
                f"Estimated risk exceeds your "
                f"${maximum_risk:,.2f} limit."
            )

        else:
            st.success(
                "Estimated risk is within your "
                "selected limit."
            )

    st.write(
        "Data status:",
        selected_analysis.freshness,
    )

    with st.expander(
        "Evidence",
        expanded=True,
    ):
        if selected_analysis.reasons:
            for reason in (
                selected_analysis.reasons
            ):
                st.write(f"• {reason}")

        else:
            st.write(
                "No strong directional evidence."
            )

    if selected_analysis.warnings:
        with st.expander(
            "Warnings",
            expanded=True,
        ):
            for warning in (
                selected_analysis.warnings
            ):
                st.warning(warning)


# ============================================================
# ONE-MINUTE CHECKLIST
# ============================================================

st.divider()
st.subheader("One-minute setup checks")

one_minute = analyses["1m"]

one_minute_checks = {
    "Enough candle history": (
        one_minute.price is not None
    ),
    "Relevant fresh FVG selected": (
        one_minute.fvg is not None
    ),
    "Correct FVG boundary touched": (
        one_minute.boundary_touched
    ),
    "Required candle confirmation": (
        one_minute.confirmation
    ),
    "Candle data current": (
        one_minute.freshness == "CURRENT"
    ),
    "Complete setup formed": (
        one_minute.setup_formed
    ),
}

check_columns = st.columns(2)

for index, (name, passed) in enumerate(
    one_minute_checks.items()
):
    icon = "✅" if passed else "❌"

    check_columns[index % 2].write(
        f"{icon} {name}"
    )

if one_minute.fvg:
    st.write(
        f"Selected FVG: "
        f"{one_minute.fvg.bottom:,.2f}–"
        f"{one_minute.fvg.top:,.2f}"
    )

    st.write(
        f"FVG midpoint: "
        f"{one_minute.fvg.midpoint:,.2f}"
    )

st.subheader("Strategy rules")

strategy_rules = [
    (
        "Mark strong, fresh and unfilled "
        "one-minute FVGs."
    ),
    (
        "Prefer FVGs supported by 15-minute "
        "context and market structure."
    ),
    (
        "LONG requires price to touch the "
        "bullish FVG bottom boundary."
    ),
    (
        "SHORT requires price to touch the "
        "bearish FVG top boundary."
    ),
    "Wick touches count.",
    (
        "LONG confirmation is a green hammer "
        "or two consecutive bullish candles."
    ),
    (
        "SHORT confirmation is a red shooting "
        "star or two consecutive bearish candles."
    ),
    (
        "LONG targets the relevant swing high."
    ),
    (
        "SHORT targets the relevant swing low."
    ),
    (
        "Do not take a setup generated from "
        "delayed one-minute candles."
    ),
]

for rule in strategy_rules:
    st.write(f"• {rule}")

st.divider()

st.caption(
    "OP.exe is a research and decision-support tool. "
    "It does not guarantee accuracy or profitability and "
    "does not place trades. Test with fees and slippage, "
    "then forward-test before using real or funded capital."
)
