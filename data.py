"""
data.py — Binance API layer

Fixes:
 - HTTP session reuse (keep-alive)
 - Retry with exponential backoff on 429 / 5xx
 - Correct OI field names validated against live API
 - Strict symbol filtering (only liquid USDT perps)
 - Rate limit headroom via adaptive sleep
"""

import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com"   # USDT-margined futures

# ─────────────────────────────────────────────
# HTTP SESSION — reused across all requests
# ─────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,        # 1.5s, 3s, 6s, 12s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

_session = _make_session()


def _get(url: str, params: dict = None, timeout: int = 10) -> Optional[dict | list]:
    """
    Shared GET with rate-limit awareness.
    Returns parsed JSON or None on failure.
    """
    try:
        resp = _session.get(url, params=params, timeout=timeout)

        # Binance rate limit header — back off if close to limit
        used_weight = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", 0))
        if used_weight > 1000:
            wait = 2.0 + (used_weight - 1000) / 100
            logger.warning(f"Rate limit weight={used_weight}, sleeping {wait:.1f}s")
            time.sleep(wait)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10))
            logger.warning(f"429 Too Many Requests — sleeping {retry_after}s")
            time.sleep(retry_after)
            return None

        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code} for {url} params={params}")
            return None

        return resp.json()

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout: {url}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Connection error: {e}")
        return None
    except Exception as e:
        logger.error(f"_get error: {e}")
        return None


# ─────────────────────────────────────────────
# MARKET SCANNER — only real liquid perps
# ─────────────────────────────────────────────

def get_24h_tickers(min_volume_usdt: float = 5_000_000) -> list[dict]:
    """
    Returns 24h ticker data for liquid USDT-margined perpetuals.
    Filters:
    - 24h quote volume >= min_volume_usdt
    - Symbol ends with USDT
    - Not a delivery/quarterly contract (no underscore in name)
    - Price > 0 (active market)
    """
    data = _get(f"{BASE_URL}/fapi/v1/ticker/24hr")
    if not data:
        logger.error("get_24h_tickers: empty response")
        return []

    filtered: list[dict] = []
    for t in data:
        sym = t.get("symbol", "")
        try:
            quote_vol = float(t.get("quoteVolume", 0))
            price     = float(t.get("lastPrice", 0))
            change    = float(t.get("priceChangePercent", 0))
        except (ValueError, TypeError):
            continue

        if not sym.endswith("USDT"):
            continue
        if "_" in sym:           # quarterly/delivery: BTCUSDT_240329
            continue
        if price <= 0:
            continue
        if quote_vol < min_volume_usdt:
            continue

        filtered.append({
            "symbol": sym,
            "quoteVolume": quote_vol,
            "lastPrice": price,
            "priceChangePercent": change,
        })

    logger.info(f"get_24h_tickers: {len(filtered)} liquid perps found")
    return filtered


def get_futures_symbols(min_volume_usdt: float = 5_000_000) -> list[str]:
    """
    Returns USDT-margined perpetual futures symbols filtered by liquidity.
    """
    tickers = get_24h_tickers(min_volume_usdt=min_volume_usdt)
    return [t["symbol"] for t in tickers]


def get_24h_change(symbol: str) -> Optional[float]:
    """Returns 24h price change percent."""
    data = _get(f"{BASE_URL}/fapi/v1/ticker/24hr", {"symbol": symbol})
    if not data:
        return None
    try:
        return float(data["priceChangePercent"])
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"get_24h_change({symbol}): {e}")
        return None


def get_latest_prices(symbols: Optional[Iterable[str]] = None) -> dict[str, float]:
    """
    Returns latest prices for symbols.
    If symbols is None, returns all symbols.
    """
    data = _get(f"{BASE_URL}/fapi/v1/ticker/price")
    if not data:
        return {}

    want = None
    if symbols is not None:
        want = {s for s in symbols}

    prices: dict[str, float] = {}
    try:
        if isinstance(data, dict):
            # Single symbol response
            sym = data.get("symbol")
            price = float(data.get("price", 0))
            if sym and (want is None or sym in want) and price > 0:
                prices[sym] = price
            return prices

        for item in data:
            sym = item.get("symbol")
            if want is not None and sym not in want:
                continue
            try:
                price = float(item.get("price", 0))
            except (TypeError, ValueError):
                continue
            if sym and price > 0:
                prices[sym] = price
        return prices
    except Exception as e:
        logger.warning(f"get_latest_prices: {e}")
        return {}


# ─────────────────────────────────────────────
# KLINES (OHLCV)
# ─────────────────────────────────────────────

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]


def get_klines(symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Binance Futures.
    Returns clean DataFrame or None if data is invalid.
    """
    data = _get(
        f"{BASE_URL}/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit}
    )
    if not data or not isinstance(data, list) or len(data) < 5:
        logger.debug(f"get_klines({symbol},{interval}): empty/short response")
        return None

    try:
        df = pd.DataFrame(data, columns=_KLINE_COLS)
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        df = df[["open", "high", "low", "close", "volume", "quote_volume"]]

        # Drop rows with NaN or zero prices
        df = df.dropna()
        df = df[df["close"] > 0]
        df = df[df["high"] >= df["low"]]    # sanity: high >= low

        if len(df) < 5:
            logger.debug(f"get_klines({symbol},{interval}): too few valid rows after clean")
            return None

        return df

    except Exception as e:
        logger.warning(f"get_klines({symbol},{interval}) parse error: {e}")
        return None


def get_historical_klines(
    symbol: str,
    interval: str,
    start_str: str,
    end_str: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Paginated historical klines for backtesting."""
    try:
        start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
        end_ts   = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)

        all_rows = []
        current  = start_ts

        while current < end_ts:
            data = _get(f"{BASE_URL}/fapi/v1/klines", {
                "symbol":    symbol,
                "interval":  interval,
                "startTime": current,
                "endTime":   end_ts,
                "limit":     1500,
            })
            if not data:
                break
            all_rows.extend(data)
            current = data[-1][0] + 1
            time.sleep(0.15)

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows, columns=_KLINE_COLS)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df = df[~df.index.duplicated(keep="first")]
        return df

    except Exception as e:
        logger.error(f"get_historical_klines({symbol}): {e}")
        return None


# ─────────────────────────────────────────────
# FUNDING RATE
# ─────────────────────────────────────────────

def get_funding_rate(symbol: str) -> Optional[float]:
    """
    Returns current funding rate.
    Endpoint: /fapi/v1/premiumIndex
    Field: lastFundingRate (string float, e.g. "0.00050000")
    Returns None if symbol has no funding (not a perp).
    """
    data = _get(f"{BASE_URL}/fapi/v1/premiumIndex", {"symbol": symbol})
    if not data:
        return None

    try:
        rate_str = data.get("lastFundingRate", "")
        if rate_str == "" or rate_str is None:
            return None
        rate = float(rate_str)
        # Sanity: funding rate should be between -2% and +2%
        if not (-0.02 <= rate <= 0.02):
            logger.warning(f"{symbol}: suspicious funding rate {rate:.6f}, treating as None")
            return None
        return rate
    except (ValueError, TypeError) as e:
        logger.warning(f"get_funding_rate({symbol}): parse error {e}, data={data}")
        return None


# ─────────────────────────────────────────────
# OPEN INTEREST
# ─────────────────────────────────────────────

def get_open_interest_history(
    symbol: str,
    period: str = "1h",
    limit: int = 6,
) -> Optional[pd.DataFrame]:
    """
    Fetch open interest history.
    Endpoint: /futures/data/openInterestHist
    Fields returned:
        symbol, sumOpenInterest, sumOpenInterestValue, timestamp

    sumOpenInterest      — number of contracts
    sumOpenInterestValue — value in USDT  ← this is what we use
    """
    data = _get(f"{BASE_URL}/futures/data/openInterestHist", {
        "symbol": symbol,
        "period": period,
        "limit":  limit,
    })
    if not data or not isinstance(data, list) or len(data) == 0:
        logger.debug(f"get_open_interest_history({symbol}): empty")
        return None

    try:
        df = pd.DataFrame(data)

        # Validate required fields exist
        required = {"sumOpenInterest", "sumOpenInterestValue", "timestamp"}
        if not required.issubset(df.columns):
            logger.warning(f"get_open_interest_history({symbol}): missing fields, got {list(df.columns)}")
            return None

        df["timestamp"]            = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["sumOpenInterest"]      = pd.to_numeric(df["sumOpenInterest"],      errors="coerce")
        df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
        df.set_index("timestamp", inplace=True)
        df = df.dropna(subset=["sumOpenInterest", "sumOpenInterestValue"])

        if len(df) == 0:
            return None

        return df

    except Exception as e:
        logger.warning(f"get_open_interest_history({symbol}): {e}")
        return None


def get_open_interest_value(symbol: str) -> Optional[float]:
    """Quick snapshot of current OI in USDT."""
    df = get_open_interest_history(symbol, period="5m", limit=1)
    if df is None or df.empty:
        return None
    val = float(df["sumOpenInterestValue"].iloc[-1])
    return val if val > 0 else None


def is_oi_diverging(symbol: str) -> bool:
    """
    OI divergence: price rising while OI falling over last 6 × 1H candles.
    Bearish signal — buyers exhausted, forced longs unwinding.
    """
    try:
        oi_df  = get_open_interest_history(symbol, period="1h", limit=6)
        klines = get_klines(symbol, "1h", limit=6)

        if oi_df is None or klines is None or len(oi_df) < 2 or len(klines) < 2:
            return False

        price_chg = klines["close"].iloc[-1] / klines["close"].iloc[0] - 1
        # Use USDT value (not contract count) — more accurate for divergence
        oi_chg    = oi_df["sumOpenInterestValue"].iloc[-1] / oi_df["sumOpenInterestValue"].iloc[0] - 1

        # Price up > 1%, OI down > 1%
        diverging = price_chg > 0.01 and oi_chg < -0.01
        if diverging:
            logger.info(f"{symbol}: OI divergence — price {price_chg:+.2%}, OI {oi_chg:+.2%}")
        return diverging

    except Exception as e:
        logger.warning(f"is_oi_diverging({symbol}): {e}")
        return False
