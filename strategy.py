"""
strategy.py — улучшенная логика сигналов
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from data import (
    get_klines, get_funding_rate, is_oi_diverging, get_open_interest_history
)
from indicators import (
    calc_rsi, calc_atr, is_volume_spike, get_volume_ratio,
    find_resistance_levels, nearest_resistance, price_near_resistance,
    detect_liquidity_sweep, find_volume_profile_levels, detect_rsi_divergence,
    is_pin_bar, is_volume_climax, is_weak_after_pump, get_macd, detect_macd_divergence
)

logger = logging.getLogger(__name__)

class Config:
    # Пользовательские настройки (через env)
    MIN_PUMP_PCT = float(os.getenv("MIN_PUMP_PCT", "25"))
    MIN_SCORE = float(os.getenv("MIN_SCORE", "4.0"))
    SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SEC", "300"))

    # Внутренние настройки (оптимизированы)
    VOLUME_MULTIPLIER = 3.0
    RSI_OVERBOUGHT = 75.0
    RESISTANCE_TOLERANCE = 3.0
    MIN_FUNDING_RATE = 0.0005
    ATR_PERIOD = 14
    ATR_SL_MULT = 2.0
    ATR_TP1_MULT = 2.0
    ATR_TP2_MULT = 4.0
    MIN_ATR_PCT = 0.003
    MIN_OI_VALUE = 1_500_000
    USE_MULTI_TF = True

    # Веса факторов (подобраны по бэктестам)
    WEIGHTS = {
        'pump': 0.5,
        'volume_spike': 0.5,
        'rsi': 1.0,
        'resistance': 1.5,
        'funding': 1.0,
        'oi_divergence': 1.0,
        'liquidity_sweep': 0.8,
        'pin_bar_4h': 1.2,
        'pin_bar_1h': 0.8,
        'rsi_divergence': 1.0,
        'macd_divergence': 1.0,
        'volume_climax': 0.7,
        'weak_after_pump': 0.5,
        'confirm_1h': 1.0,
        'confirm_15m': 1.5,
    }

@dataclass
class Signal:
    symbol: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    rsi: float
    funding_rate: float
    open_interest: float
    pump_percent: float
    resistance_4h: Optional[float]
    resistance_1d: Optional[float]
    volume_ratio: float
    oi_divergence: bool
    liquidity_sweep: bool
    score: float
    confidence: int
    df_4h: pd.DataFrame = field(repr=False, default=None)
    pin_bar_4h: bool = False
    pin_bar_1h: bool = False
    rsi_divergence: bool = False
    macd_divergence: bool = False
    volume_climax: bool = False
    weak_after_pump: bool = False

def _ok_df(df, symbol, min_len):
    if df is None or len(df) < min_len:
        return False
    if df[["open","high","low","close","volume"]].isnull().any().any():
        return False
    return (df["close"] > 0).all()

def _ok_levels(entry, sl, tp1, tp2):
    if any(v<=0 or not math.isfinite(v) for v in [entry,sl,tp1,tp2]):
        return False
    if sl <= entry:
        return False
    if tp1 >= entry or tp2 >= tp1:
        return False
    if (sl - entry) / entry < 0.002:
        return False
    return True

def analyze_symbol(symbol: str, pump_pct: float) -> Optional[Signal]:
    if pump_pct < Config.MIN_PUMP_PCT:
        return None

    df_4h = get_klines(symbol, "4h", limit=200)
    if not _ok_df(df_4h, symbol, 50):
        return None

    df_1h = get_klines(symbol, "1h", limit=100) if Config.USE_MULTI_TF else None
    df_15m = get_klines(symbol, "15m", limit=100) if Config.USE_MULTI_TF else None

    price = float(df_4h["close"].iloc[-1])
    rsi_4h = calc_rsi(df_4h["close"])
    if not math.isfinite(rsi_4h):
        return None

    atr = calc_atr(df_4h, Config.ATR_PERIOD)
    if not math.isfinite(atr) or atr / price < Config.MIN_ATR_PCT:
        return None

    entry = price
    sl = entry + Config.ATR_SL_MULT * atr
    tp1 = entry - Config.ATR_TP1_MULT * atr
    tp2 = entry - Config.ATR_TP2_MULT * atr
    if not _ok_levels(entry, sl, tp1, tp2):
        return None

    funding = get_funding_rate(symbol)
    if funding is None or funding < Config.MIN_FUNDING_RATE:
        return None

    oi_df = get_open_interest_history(symbol, period="1h", limit=2)
    if oi_df is None or oi_df.empty:
        return None
    oi_value = float(oi_df["sumOpenInterestValue"].iloc[-1])
    if oi_value < Config.MIN_OI_VALUE:
        return None

    # --- Основные фильтры (4H) ---
    vol_spike = is_volume_spike(df_4h, Config.VOLUME_MULTIPLIER)
    levels_4h = find_resistance_levels(df_4h, min_touches=2)
    res_4h = nearest_resistance(price, levels_4h)
    near_res = price_near_resistance(price, res_4h, Config.RESISTANCE_TOLERANCE)
    oi_div = is_oi_diverging(symbol)
    sweep = detect_liquidity_sweep(df_4h)

    # --- Новые фильтры (4H) ---
    vp_levels = find_volume_profile_levels(df_4h, num_levels=2)
    near_vp = any(price_near_resistance(price, lvl, 1.5) for lvl in vp_levels)
    pin_4h = is_pin_bar(df_4h)
    weak = is_weak_after_pump(df_4h)

    rsi_values = [calc_rsi(df_4h['close'].iloc[:i+1]) for i in range(len(df_4h))]
    rsi_div = detect_rsi_divergence(df_4h['high'], rsi_values)

    macd, sig, _ = get_macd(df_4h)
    macd_div = detect_macd_divergence(df_4h, macd, sig)

    vol_climax = is_volume_climax(df_4h)

    # --- Подтверждение на 1H и 15m ---
    confirm_1h = False
    confirm_15m = False
    pin_1h = False

    if Config.USE_MULTI_TF and df_1h is not None and len(df_1h) >= 20:
        rsi_1h = calc_rsi(df_1h["close"])
        if rsi_1h >= Config.RSI_OVERBOUGHT - 5:
            confirm_1h = True
        pin_1h = is_pin_bar(df_1h)

    if Config.USE_MULTI_TF and df_15m is not None and len(df_15m) >= 20:
        rsi_15m = calc_rsi(df_15m["close"])
        if rsi_15m >= 70 and (is_pin_bar(df_15m) or is_bearish_engulfing(df_15m)):
            confirm_15m = True

    # --- Взвешенная оценка ---
    factors = {
        'pump': Config.WEIGHTS['pump'] * (min(pump_pct / Config.MIN_PUMP_PCT, 2.0)),
        'volume_spike': Config.WEIGHTS['volume_spike'] if vol_spike else 0,
        'rsi': Config.WEIGHTS['rsi'] * ((rsi_4h - 70) / 20) if rsi_4h > 70 else 0,
        'resistance': Config.WEIGHTS['resistance'] if near_res else (Config.WEIGHTS['resistance']*0.5 if near_vp else 0),
        'funding': Config.WEIGHTS['funding'] * (funding / Config.MIN_FUNDING_RATE) if funding else 0,
        'oi_divergence': Config.WEIGHTS['oi_divergence'] if oi_div else 0,
        'liquidity_sweep': Config.WEIGHTS['liquidity_sweep'] if sweep else 0,
        'pin_bar_4h': Config.WEIGHTS['pin_bar_4h'] if pin_4h else 0,
        'pin_bar_1h': Config.WEIGHTS['pin_bar_1h'] if pin_1h else 0,
        'rsi_divergence': Config.WEIGHTS['rsi_divergence'] if rsi_div else 0,
        'macd_divergence': Config.WEIGHTS['macd_divergence'] if macd_div else 0,
        'volume_climax': Config.WEIGHTS['volume_climax'] if vol_climax else 0,
        'weak_after_pump': Config.WEIGHTS['weak_after_pump'] if weak else 0,
    }
    if Config.USE_MULTI_TF:
        factors['confirm_1h'] = Config.WEIGHTS['confirm_1h'] if confirm_1h else 0
        factors['confirm_15m'] = Config.WEIGHTS['confirm_15m'] if confirm_15m else 0

    score = sum(factors.values())
    max_possible = sum(v for v in Config.WEIGHTS.values())
    confidence = min(100, int(score / max_possible * 100)) if max_possible > 0 else 0

    if score < Config.MIN_SCORE:
        logger.debug(f"{symbol}: score {score:.1f} < {Config.MIN_SCORE}")
        return None

    if Config.USE_MULTI_TF and not (confirm_1h or confirm_15m) and score < Config.MIN_SCORE + 1:
        return None

    logger.info(f"✅ {symbol}: score={score:.1f} confidence={confidence}%")

    return Signal(
        symbol=symbol, entry=entry, stop_loss=sl, tp1=tp1, tp2=tp2,
        rsi=rsi_4h, funding_rate=funding, open_interest=oi_value,
        pump_percent=pump_pct, resistance_4h=res_4h, resistance_1d=None,
        volume_ratio=get_volume_ratio(df_4h), oi_divergence=oi_div,
        liquidity_sweep=sweep, score=score, confidence=confidence,
        df_4h=df_4h,
        pin_bar_4h=pin_4h, pin_bar_1h=pin_1h,
        rsi_divergence=rsi_div, macd_divergence=macd_div,
        volume_climax=vol_climax, weak_after_pump=weak,
    )
