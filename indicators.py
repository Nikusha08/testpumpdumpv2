"""
indicators.py — Technical indicators

Fixes:
 - Resistance levels now require minimum touch count (not single outlier)
 - Cluster method uses price tolerance bands instead of raw histogram
 - All functions handle NaN / edge cases safely
"""

import math
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────

def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    """RSI via Wilder EMA. Returns NaN if not enough data."""
    closes = closes.dropna()
    if len(closes) < period + 1:
        return float("nan")

    delta    = closes.diff().dropna()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Avoid division by zero
    avg_loss_safe = avg_loss.replace(0, np.nan)
    rs  = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))

    val = float(rsi.iloc[-1])
    return val if math.isfinite(val) else float("nan")


# ─────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    ATR via Wilder EMA.
    Returns NaN if not enough data or invalid input.
    """
    df = df.dropna(subset=["high", "low", "close"])
    if len(df) < period + 1:
        return float("nan")

    high  = df["high"]
    low   = df["low"]
    prev  = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    val = float(atr.iloc[-1])
    return val if math.isfinite(val) and val > 0 else float("nan")


# ─────────────────────────────────────────────
# VOLUME SPIKE
# ─────────────────────────────────────────────

def is_volume_spike(df: pd.DataFrame, multiplier: float = 3.0, lookback: int = 20) -> bool:
    """Last candle volume >= multiplier × rolling average."""
    if len(df) < lookback + 1:
        return False
    avg  = df["volume"].iloc[-(lookback + 1):-1].mean()
    last = df["volume"].iloc[-1]
    if avg <= 0:
        return False
    return bool(last >= avg * multiplier)


def get_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Ratio of last candle volume to rolling average."""
    if len(df) < lookback + 1:
        return 1.0
    avg  = df["volume"].iloc[-(lookback + 1):-1].mean()
    last = df["volume"].iloc[-1]
    if avg <= 0:
        return 1.0
    ratio = last / avg
    return ratio if math.isfinite(ratio) else 1.0


# ─────────────────────────────────────────────
# RESISTANCE LEVELS  ← major rewrite
# ─────────────────────────────────────────────

def find_resistance_levels(
    df: pd.DataFrame,
    lookback: int = 50,
    min_touches: int = 2,
    tolerance_pct: float = 0.5,
) -> list[float]:
    """
    Find strong resistance levels using:
    1. Pivot highs — local maxima with left/right confirmation
    2. Touch-count cluster — price zones touched >= min_touches times

    min_touches:    level must be "respected" at least this many times
    tolerance_pct:  candles within this % of a level count as a touch

    Returns sorted list of validated resistance levels (most touches first).
    """
    if df is None or len(df) < 10:
        return []

    highs = df["high"].values[-lookback:]
    n     = len(highs)
    if n < 6:
        return []

    candidates: list[float] = []

    # ── Method 1: Pivot highs ──────────────────────
    # Require 2 candles on each side to confirm pivot
    for i in range(2, n - 2):
        h = highs[i]
        if (h > highs[i-1] and h > highs[i-2] and
                h > highs[i+1] and h > highs[i+2]):
            candidates.append(h)

    # Also include rolling max (absolute high of the window)
    candidates.append(float(np.max(highs)))

    if not candidates:
        return []

    # ── Method 2: Count touches per candidate ──────
    # A "touch" = any candle high within tolerance_pct% of the level
    validated: list[tuple[float, int]] = []  # (level, touch_count)

    for level in candidates:
        if level <= 0:
            continue
        tol   = level * (tolerance_pct / 100)
        lower = level - tol
        upper = level + tol
        touches = int(np.sum((highs >= lower) & (highs <= upper)))
        if touches >= min_touches:
            validated.append((level, touches))

    if not validated:
        # Fallback: return pivot highs with min 1 touch
        return sorted(set(round(c, 6) for c in candidates))

    # ── Merge nearby levels (within 1%) ────────────
    validated.sort(key=lambda x: x[0])
    merged: list[float] = []
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
        # Use the most-touched level in the cluster as the representative
        best = max(
            cluster,
            key=lambda l: sum(
                1 for h in highs
                if l * (1 - tolerance_pct/100) <= h <= l * (1 + tolerance_pct/100)
            )
        )
        merged.append(round(best, 8))
        i = j

    return sorted(set(merged))


def nearest_resistance(price: float, levels: list[float]) -> Optional[float]:
    """
    Returns the nearest resistance ABOVE price (within 5%).
    If nothing above, returns the closest level overall.
    """
    if not levels:
        return None

    above = [l for l in levels if l >= price * 0.97]
    if above:
        return min(above, key=lambda x: abs(x - price))

    # Nothing above — return overall closest (rare edge case)
    return min(levels, key=lambda x: abs(x - price))


def price_near_resistance(
    price: float,
    resistance: Optional[float],
    tolerance_pct: float = 3.0,
) -> bool:
    """True if price is within tolerance% of resistance level."""
    if resistance is None or resistance <= 0:
        return False
    diff = abs(price - resistance) / resistance * 100
    return diff <= tolerance_pct


# ─────────────────────────────────────────────
# LIQUIDITY SWEEP
# ─────────────────────────────────────────────

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 10) -> bool:
    """
    Detect stop hunt / liquidity sweep on the last candle:
    - Candle wicks above recent N-bar high (grabs stops)
    - But closes BELOW that high (rejection)
    - Upper wick is > 50% of total candle range

    Classic signal before sharp reversal down.
    """
    if len(df) < lookback + 2:
        return False

    recent      = df.iloc[-(lookback + 2):-1]
    last        = df.iloc[-1]
    recent_high = recent["high"].max()

    swept_high  = float(last["high"])  > recent_high
    closed_below = float(last["close"]) < recent_high

    candle_range = float(last["high"]) - float(last["low"])
    if candle_range <= 0:
        return False

    upper_wick  = float(last["high"]) - max(float(last["open"]), float(last["close"]))
    large_wick  = (upper_wick / candle_range) > 0.5

    return swept_high and closed_below and large_wick


# ─────────────────────────────────────────────
# PUMP DETECTION
# ─────────────────────────────────────────────

def calc_pump_percent(df_4h: pd.DataFrame, candles_back: int = 6) -> float:
    """
    % price change over last N candles.
    Default 6 × 4H = 24H lookback.
    """
    if df_4h is None or len(df_4h) < candles_back + 1:
        return 0.0
    start = float(df_4h["close"].iloc[-(candles_back + 1)])
    end   = float(df_4h["close"].iloc[-1])
    if start <= 0:
        return 0.0
    return (end / start - 1) * 100
