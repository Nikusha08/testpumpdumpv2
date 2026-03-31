"""
strategy.py — Signal generation logic

Fixes:
 - MIN_FUNDING_RATE raised to 0.0005 (0.05%) — eliminates low-funding noise
 - OI parsing now relies on validated data.py (sumOpenInterestValue)
 - All numeric inputs validated before scoring
 - Resistance requires min_touches=2 (no single-outlier pivots)
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


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

class Config:
    # Pump filter
    MIN_PUMP_PCT: float       = 25.0

    # Volume spike multiplier vs 20-candle avg
    VOLUME_MULTIPLIER: float  = 3.0

    # RSI overbought on 4H
    RSI_OVERBOUGHT: float     = 75.0

    # Price must be within X% of resistance
    RESISTANCE_TOLERANCE_PCT  = 3.0

    # Funding rate: raised from 0.0001 to 0.0005 (0.05%)
    # Below this = normal market, not extreme long crowding
    MIN_FUNDING_RATE: float   = 0.0005

    # Minimum confirmed filters to emit signal
    MIN_SCORE: int            = 3

    # ATR settings
    ATR_PERIOD: int           = 14
    ATR_SL_MULT: float        = 2.0
    ATR_TP1_MULT: float       = 2.0
    ATR_TP2_MULT: float       = 4.0

    # Validation gates
    MIN_ATR_PCT: float        = 0.003   # ATR must be >= 0.3% of price
    MIN_CANDLES: int          = 25      # minimum candles needed
    MIN_OI_VALUE: float       = 1_500_000  # minimum OI $1.5M — filters junk coins


# ─────────────────────────────────────────────
# SIGNAL DATA CLASS
# ─────────────────────────────────────────────

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
    score: int
    df_4h: pd.DataFrame = field(repr=False, default=None)


# ─────────────────────────────────────────────
# VALIDATION HELPERS
# ─────────────────────────────────────────────

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
    if sl <= entry:          # short: SL must be ABOVE entry
        return False
    if tp1 >= entry:         # short: TP must be BELOW entry
        return False
    if tp2 >= tp1:
        return False
    if (sl - entry) / entry < 0.002:  # < 0.2% distance — too tight
        return False
    return True


# ─────────────────────────────────────────────
# MAIN SIGNAL GENERATOR
# ─────────────────────────────────────────────

def analyze_symbol(symbol: str, pump_pct: float) -> Optional[Signal]:
    """
    Runs full analysis pipeline on one symbol.
    Returns Signal or None.
    """

    # ── Gate 1: pump threshold ─────────────────────
    if pump_pct < Config.MIN_PUMP_PCT:
        return None

    # ── Gate 2: enough 4H candles ─────────────────
    df_4h = get_klines(symbol, "4h", limit=100)
    if not _ok_df(df_4h, symbol, Config.MIN_CANDLES):
        return None

    price = float(df_4h["close"].iloc[-1])

    # ── Gate 3: RSI must be a real number ──────────
    rsi = calc_rsi(df_4h["close"])
    if not math.isfinite(rsi):
        logger.debug(f"{symbol}: RSI not finite ({rsi})")
        return None

    # ── Gate 4: ATR must be meaningful ────────────
    atr = calc_atr(df_4h, Config.ATR_PERIOD)
    if not math.isfinite(atr) or atr <= 0:
        logger.debug(f"{symbol}: ATR invalid ({atr})")
        return None
    if atr / price < Config.MIN_ATR_PCT:
        logger.debug(f"{symbol}: ATR too small ({atr/price:.4%})")
        return None

    # ── Gate 5: trade levels must be valid ─────────
    entry = price
    sl    = entry + Config.ATR_SL_MULT  * atr
    tp1   = entry - Config.ATR_TP1_MULT * atr
    tp2   = entry - Config.ATR_TP2_MULT * atr

    if not _ok_levels(entry, sl, tp1, tp2):
        logger.debug(f"{symbol}: invalid trade levels")
        return None

    # ── Gate 6: must have real funding rate ────────
    # None = symbol has no perpetual funding = not a perp = skip
    funding = get_funding_rate(symbol)
    if funding is None:
        logger.debug(f"{symbol}: no funding rate — not a perp")
        return None

    # ── Gate 7: must have real OI data ────────────
    oi_df = get_open_interest_history(symbol, period="1h", limit=2)
    if oi_df is None or oi_df.empty:
        logger.debug(f"{symbol}: no OI data")
        return None

    oi_value = float(oi_df["sumOpenInterestValue"].iloc[-1])
    if not math.isfinite(oi_value) or oi_value <= 0:
        logger.debug(f"{symbol}: OI value invalid ({oi_value})")
        return None
    if oi_value < Config.MIN_OI_VALUE:
        logger.debug(f"{symbol}: OI too small (${oi_value:,.0f} < $2M)")
        return None

    # ── SCORE SYSTEM ────────────────────────────────
    score   = 0
    reasons = []

    # 1. Pump ✅
    score += 1
    reasons.append(f"pump={pump_pct:.1f}%")

    # 2. Volume spike
    vol_ratio = get_volume_ratio(df_4h)
    if is_volume_spike(df_4h, Config.VOLUME_MULTIPLIER):
        score += 1
        reasons.append(f"vol={vol_ratio:.1f}x")

    # 3. RSI overbought
    if rsi >= Config.RSI_OVERBOUGHT:
        score += 1
        reasons.append(f"RSI={rsi:.1f}")

    # 4. Near strong resistance (requires 2+ touches)
    levels_4h = find_resistance_levels(df_4h, min_touches=2)
    res_4h    = nearest_resistance(price, levels_4h)
    near_4h   = price_near_resistance(price, res_4h, Config.RESISTANCE_TOLERANCE_PCT)

    df_1d  = get_klines(symbol, "1d", limit=60)
    res_1d = None
    near_1d = False
    if _ok_df(df_1d, symbol, min_len=10):
        levels_1d = find_resistance_levels(df_1d, min_touches=2)
        res_1d    = nearest_resistance(price, levels_1d)
        near_1d   = price_near_resistance(price, res_1d, Config.RESISTANCE_TOLERANCE_PCT)

    if near_4h or near_1d:
        score += 1
        reasons.append(f"res={'4H' if near_4h else '1D'}")

    # 5. High funding rate (0.05%+ threshold — real crowding)
    if funding >= Config.MIN_FUNDING_RATE:
        score += 1
        reasons.append(f"funding={funding*100:.3f}%")

    # 6. OI divergence (price up, OI down)
    oi_div = is_oi_diverging(symbol)
    if oi_div:
        score += 1
        reasons.append("OI_div")

    # 7. Liquidity sweep
    sweep = detect_liquidity_sweep(df_4h)
    if sweep:
        score += 1
        reasons.append("sweep")

    # ── Final score gate ───────────────────────────
    if score < Config.MIN_SCORE:
        logger.debug(f"{symbol}: score {score}/{Config.MIN_SCORE} — [{', '.join(reasons)}]")
        return None

    logger.info(f"✅ {symbol}: score={score}/7 [{', '.join(reasons)}]")

    return Signal(
        symbol=symbol,
        entry=entry,
        stop_loss=sl,
        tp1=tp1,
        tp2=tp2,
        rsi=rsi,
        funding_rate=funding,
        open_interest=oi_value,
        pump_percent=pump_pct,
        resistance_4h=res_4h,
        resistance_1d=res_1d,
        volume_ratio=vol_ratio,
        oi_divergence=oi_div,
        liquidity_sweep=sweep,
        score=score,
        df_4h=df_4h,
    )
