"""
indicators.py — расширенный набор индикаторов
"""

import numpy as np
import pandas as pd
import math
from typing import Optional

# ---------- RSI ----------
def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    closes = closes.dropna()
    if len(closes) < period + 1:
        return float("nan")
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    avg_loss_safe = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if math.isfinite(val) else float("nan")

# ---------- ATR ----------
def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    df = df.dropna(subset=["high", "low", "close"])
    if len(df) < period + 1:
        return float("nan")
    high = df["high"]
    low = df["low"]
    prev = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    val = float(atr.iloc[-1])
    return val if math.isfinite(val) and val > 0 else float("nan")

# ---------- Volume ----------
def is_volume_spike(df: pd.DataFrame, multiplier: float = 3.0, lookback: int = 20) -> bool:
    if len(df) < lookback + 1:
        return False
    avg = df["volume"].iloc[-(lookback + 1):-1].mean()
    last = df["volume"].iloc[-1]
    return last >= avg * multiplier if avg > 0 else False

def get_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    if len(df) < lookback + 1:
        return 1.0
    avg = df["volume"].iloc[-(lookback + 1):-1].mean()
    last = df["volume"].iloc[-1]
    return last / avg if avg > 0 else 1.0

# ---------- Resistance ----------
def find_resistance_levels(df: pd.DataFrame, lookback: int = 50, min_touches: int = 2, tolerance_pct: float = 0.5) -> list[float]:
    if df is None or len(df) < 10:
        return []
    highs = df["high"].values[-lookback:]
    n = len(highs)
    if n < 6:
        return []
    candidates = []
    for i in range(2, n - 2):
        h = highs[i]
        if (h > highs[i-1] and h > highs[i-2] and
                h > highs[i+1] and h > highs[i+2]):
            candidates.append(h)
    candidates.append(float(np.max(highs)))
    if not candidates:
        return []
    validated = []
    for level in candidates:
        if level <= 0:
            continue
        tol = level * (tolerance_pct / 100)
        lower = level - tol
        upper = level + tol
        touches = int(np.sum((highs >= lower) & (highs <= upper)))
        if touches >= min_touches:
            validated.append((level, touches))
    if not validated:
        return sorted(set(round(c, 6) for c in candidates))
    validated.sort(key=lambda x: x[0])
    merged = []
    i = 0
    while i < len(validated):
        level, touches = validated[i]
        cluster = [level]
        cluster_touches = touches
        j = i + 1
        while j < len(validated) and validated[j][0] <= level * 1.01:
            cluster.append(validated[j][0])
            cluster_touches += validated[j][1]
            j += 1
        best = max(cluster, key=lambda l: sum(1 for h in highs if l*(1-tolerance_pct/100) <= h <= l*(1+tolerance_pct/100)))
        merged.append(round(best, 8))
        i = j
    return sorted(set(merged))

def nearest_resistance(price: float, levels: list[float]) -> Optional[float]:
    if not levels:
        return None
    above = [l for l in levels if l >= price * 0.97]
    if above:
        return min(above, key=lambda x: abs(x - price))
    return min(levels, key=lambda x: abs(x - price))

def price_near_resistance(price: float, resistance: Optional[float], tolerance_pct: float = 3.0) -> bool:
    if resistance is None or resistance <= 0:
        return False
    diff = abs(price - resistance) / resistance * 100
    return diff <= tolerance_pct

# ---------- Liquidity sweep ----------
def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback + 2:
        return False
    recent = df.iloc[-(lookback + 2):-1]
    last = df.iloc[-1]
    recent_high = recent["high"].max()
    swept_high = float(last["high"]) > recent_high
    closed_below = float(last["close"]) < recent_high
    candle_range = float(last["high"]) - float(last["low"])
    if candle_range <= 0:
        return False
    upper_wick = float(last["high"]) - max(float(last["open"]), float(last["close"]))
    large_wick = (upper_wick / candle_range) > 0.5
    return swept_high and closed_below and large_wick

# ---------- Pump ----------
def calc_pump_percent(df_4h: pd.DataFrame, candles_back: int = 6) -> float:
    if df_4h is None or len(df_4h) < candles_back + 1:
        return 0.0
    start = float(df_4h["close"].iloc[-(candles_back + 1)])
    end = float(df_4h["close"].iloc[-1])
    if start <= 0:
        return 0.0
    return (end / start - 1) * 100

# ---------- Volume Profile ----------
def find_volume_profile_levels(df: pd.DataFrame, num_levels: int = 3, bins: int = 50) -> list[float]:
    if df.empty or len(df) < 10:
        return []
    price_min = df['low'].min()
    price_max = df['high'].max()
    if price_max <= price_min:
        return []
    bin_width = (price_max - price_min) / bins
    bins_list = np.arange(price_min, price_max + bin_width, bin_width)
    volume_by_bin = np.zeros(len(bins_list)-1)
    for i in range(len(df)):
        price = df['close'].iloc[i]
        vol = df['volume'].iloc[i]
        bin_idx = int((price - price_min) / bin_width)
        if 0 <= bin_idx < len(volume_by_bin):
            volume_by_bin[bin_idx] += vol
    top_bins = np.argsort(volume_by_bin)[-num_levels:][::-1]
    levels = [bins_list[bin] + bin_width/2 for bin in top_bins if volume_by_bin[bin] > 0]
    return sorted(levels)

# ---------- Divergence ----------
def detect_rsi_divergence(price_highs: pd.Series, rsi_values: list, lookback: int = 10) -> bool:
    if len(price_highs) < lookback or len(rsi_values) < lookback:
        return False
    price_peaks = price_highs.iloc[-lookback:].values
    rsi_peaks = rsi_values[-lookback:]
    price_max_idx = np.argmax(price_peaks)
    rsi_max_idx = np.argmax(rsi_peaks)
    if price_max_idx != len(price_peaks)-1:
        return False
    prev_price_max = max(price_peaks[:price_max_idx]) if price_max_idx > 0 else price_peaks[0]
    prev_rsi_max = max(rsi_peaks[:rsi_max_idx]) if rsi_max_idx > 0 else rsi_peaks[0]
    return price_peaks[-1] > prev_price_max and rsi_peaks[-1] < prev_rsi_max

# ---------- MACD ----------
def get_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def detect_macd_divergence(df: pd.DataFrame, macd_line: pd.Series, signal_line: pd.Series) -> bool:
    if len(df) < 5 or len(macd_line) < 5:
        return False
    price_high = df['high'].iloc[-1] / df['high'].iloc[-2] - 1
    macd_change = (macd_line.iloc[-1] - macd_line.iloc[-2]) / (abs(macd_line.iloc[-2])+1e-9)
    return price_high > 0.01 and macd_change < -0.01

# ---------- Candlestick patterns ----------
def is_pin_bar(df: pd.DataFrame, lookback: int = 1) -> bool:
    if len(df) < lookback:
        return False
    last = df.iloc[-1]
    body = abs(last['close'] - last['open'])
    upper_wick = last['high'] - max(last['open'], last['close'])
    lower_wick = min(last['open'], last['close']) - last['low']
    if body == 0:
        return upper_wick > lower_wick * 2 and upper_wick > 0.5 * (last['high'] - last['low'])
    return upper_wick > body * 2 and upper_wick > lower_wick

def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    if prev['close'] <= prev['open']:
        return False
    return (curr['close'] < curr['open'] and
            curr['open'] >= prev['close'] and
            curr['close'] <= prev['open'])

def is_volume_climax(df: pd.DataFrame, multiplier: float = 2.5) -> bool:
    if len(df) < 20:
        return False
    avg_vol = df['volume'].iloc[-20:-1].mean()
    last_vol = df['volume'].iloc[-1]
    last_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    return (last_vol > avg_vol * multiplier and
            last_close <= prev_close)

def is_weak_after_pump(df: pd.DataFrame, lookback: int = 4) -> bool:
    if len(df) < lookback:
        return True
    recent_high = df['high'].iloc[-lookback:-1].max()
    last_high = df['high'].iloc[-1]
    return last_high < recent_high
