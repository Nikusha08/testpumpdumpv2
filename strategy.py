"""
strategy.py - Signal generation logic

The core setup is intentionally price-action first:
- only closed candles are used
- live trading and backtesting share the same setup rules
- funding/OI add confidence, but do not create untestable entries by themselves
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from data import (
    get_klines,
    get_funding_rate,
    is_oi_diverging,
    get_open_interest_history,
)
from indicators import (
    calc_rsi,
    calc_atr,
    is_volume_spike,
    get_volume_ratio,
    find_resistance_levels,
    nearest_resistance,
    price_near_resistance,
    detect_liquidity_sweep,
)

logger = logging.getLogger(__name__)


class Config:
    MIN_PUMP_PCT: float = 25.0
    VOLUME_MULTIPLIER: float = 2.5
    RSI_OVERBOUGHT: float = 78.0
    RESISTANCE_TOLERANCE_PCT: float = 2.0
    MAX_RESISTANCE_DISTANCE_PCT: float = 8.0
    MIN_FUNDING_RATE: float = 0.0005
    MIN_SCORE: int = 4
    ATR_PERIOD: int = 14
    ATR_SL_MULT: float = 2.0
    ATR_TP1_MULT: float = 2.0
    ATR_TP2_MULT: float = 4.0
    MIN_ATR_PCT: float = 0.004
    MIN_CANDLES: int = 30


@dataclass
class Signal:
    symbol: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    rsi: float
    funding_rate: Optional[float]
    open_interest: Optional[float]
    pump_percent: float
    resistance_4h: Optional[float]
    resistance_1d: Optional[float]
    volume_ratio: float
    oi_divergence: bool
    liquidity_sweep: bool
    score: int
    core_score: int
    bonus_score: int
    df_4h: pd.DataFrame = field(repr=False, default=None)


@dataclass
class SetupMetrics:
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    rsi: float
    pump_percent: float
    resistance_4h: Optional[float]
    resistance_1d: Optional[float]
    volume_ratio: float
    volume_spike: bool
    liquidity_sweep: bool
    score: int


def _ok_df(df: Optional[pd.DataFrame], symbol: str, min_len: int = 25) -> bool:
    if df is None or len(df) < min_len:
        return False
    if df[["open", "high", "low", "close", "volume"]].isnull().any().any():
        logger.debug(f"{symbol}: NaN in OHLCV")
        return False
    if (df["close"] <= 0).any():
        logger.debug(f"{symbol}: non-positive close prices")
        return False
    return True


def _ok_levels(entry: float, sl: float, tp1: float, tp2: float) -> bool:
    if any(v <= 0 or not math.isfinite(v) for v in [entry, sl, tp1, tp2]):
        return False
    if sl <= entry:
        return False
    if tp1 >= entry:
        return False
    if tp2 >= tp1:
        return False
    if (sl - entry) / entry < 0.002:
        return False
    return True


def build_daily_frame(df_4h: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Resample 4H candles into fully closed daily candles.
    The current UTC day is excluded because it is still forming intraday.
    """
    if df_4h is None or df_4h.empty:
        return None

    daily = (
        df_4h.resample("1D", label="left", closed="left")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
    )
    if daily.empty:
        return None

    current_day_start = df_4h.index[-1].floor("1D")
    if daily.index[-1] == current_day_start:
        daily = daily.iloc[:-1]

    return daily if not daily.empty else None


def evaluate_price_action_setup(
    symbol: str,
    pump_pct: float,
    df_4h: Optional[pd.DataFrame],
    df_1d: Optional[pd.DataFrame] = None,
) -> Optional[SetupMetrics]:
    """
    Shared setup used by both live scanning and backtests.
    Entries require:
    - a large 24H pump
    - overbought RSI
    - price near confirmed resistance
    - a reversal trigger (volume spike or liquidity sweep)
    """
    if pump_pct < Config.MIN_PUMP_PCT:
        return None
    if not _ok_df(df_4h, symbol, Config.MIN_CANDLES):
        return None

    price = float(df_4h["close"].iloc[-1])
    rsi = calc_rsi(df_4h["close"])
    if not math.isfinite(rsi):
        logger.debug(f"{symbol}: RSI not finite ({rsi})")
        return None

    atr = calc_atr(df_4h, Config.ATR_PERIOD)
    if not math.isfinite(atr) or atr <= 0:
        logger.debug(f"{symbol}: ATR invalid ({atr})")
        return None
    if atr / price < Config.MIN_ATR_PCT:
        logger.debug(f"{symbol}: ATR too small ({atr/price:.4%})")
        return None

    entry = price
    sl = entry + Config.ATR_SL_MULT * atr
    tp1 = entry - Config.ATR_TP1_MULT * atr
    tp2 = entry - Config.ATR_TP2_MULT * atr
    if not _ok_levels(entry, sl, tp1, tp2):
        logger.debug(f"{symbol}: invalid trade levels")
        return None

    if df_1d is None:
        df_1d = build_daily_frame(df_4h)

    levels_4h = find_resistance_levels(df_4h, min_touches=2)
    res_4h = nearest_resistance(price, levels_4h, Config.MAX_RESISTANCE_DISTANCE_PCT)
    near_4h = price_near_resistance(price, res_4h, Config.RESISTANCE_TOLERANCE_PCT)

    res_1d = None
    near_1d = False
    if _ok_df(df_1d, symbol, min_len=10):
        levels_1d = find_resistance_levels(df_1d, min_touches=2)
        res_1d = nearest_resistance(price, levels_1d, Config.MAX_RESISTANCE_DISTANCE_PCT)
        near_1d = price_near_resistance(price, res_1d, Config.RESISTANCE_TOLERANCE_PCT)

    near_resistance = near_4h or near_1d
    rsi_overbought = rsi >= Config.RSI_OVERBOUGHT
    volume_ratio = get_volume_ratio(df_4h)
    volume_spike = is_volume_spike(df_4h, Config.VOLUME_MULTIPLIER)
    sweep = detect_liquidity_sweep(df_4h)
    reversal_trigger = volume_spike or sweep

    if not near_resistance:
        return None
    if not rsi_overbought:
        return None
    if not reversal_trigger:
        return None

    score = 1
    score += int(rsi_overbought)
    score += int(near_resistance)
    score += int(volume_spike)
    score += int(sweep)

    if score < Config.MIN_SCORE:
        return None

    return SetupMetrics(
        entry=entry,
        stop_loss=sl,
        tp1=tp1,
        tp2=tp2,
        rsi=rsi,
        pump_percent=pump_pct,
        resistance_4h=res_4h,
        resistance_1d=res_1d,
        volume_ratio=volume_ratio,
        volume_spike=volume_spike,
        liquidity_sweep=sweep,
        score=score,
    )


def analyze_symbol(symbol: str, pump_pct: float) -> Optional[Signal]:
    """
    Runs full analysis pipeline on one symbol.
    Uses only closed candles for the entry setup.
    """
    df_4h = get_klines(symbol, "4h", limit=120, closed_only=True)
    if df_4h is None:
        return None

    df_1d = get_klines(symbol, "1d", limit=90, closed_only=True)
    setup = evaluate_price_action_setup(symbol, pump_pct, df_4h, df_1d)
    if setup is None:
        return None

    funding = get_funding_rate(symbol)
    oi_df = get_open_interest_history(symbol, period="1h", limit=2)
    oi_value = None
    if oi_df is not None and not oi_df.empty:
        raw_oi = float(oi_df["sumOpenInterestValue"].iloc[-1])
        if math.isfinite(raw_oi) and raw_oi > 0:
            oi_value = raw_oi

    oi_div = is_oi_diverging(symbol)

    bonus_score = 0
    if funding is not None and funding >= Config.MIN_FUNDING_RATE:
        bonus_score += 1
    if oi_div:
        bonus_score += 1

    total_score = setup.score + bonus_score
    logger.info(
        f"{symbol}: core={setup.score}/5 bonus={bonus_score}/2 total={total_score}/7"
    )

    return Signal(
        symbol=symbol,
        entry=setup.entry,
        stop_loss=setup.stop_loss,
        tp1=setup.tp1,
        tp2=setup.tp2,
        rsi=setup.rsi,
        funding_rate=funding,
        open_interest=oi_value,
        pump_percent=setup.pump_percent,
        resistance_4h=setup.resistance_4h,
        resistance_1d=setup.resistance_1d,
        volume_ratio=setup.volume_ratio,
        oi_divergence=oi_div,
        liquidity_sweep=setup.liquidity_sweep,
        score=total_score,
        core_score=setup.score,
        bonus_score=bonus_score,
        df_4h=df_4h,
    )
