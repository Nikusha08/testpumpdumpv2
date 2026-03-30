
---

## 3. `data.py` – полная версия (с историческими OI/funding и кэшем)

```python
"""
data.py — Binance API layer with caching and historical data.
"""

import time
import logging
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
BASE_URL = "https://fapi.binance.com"

# ---------- HTTP Session ----------
_session = None
_cache = {}  # {key: {'data': ..., 'ts': ...}}

def _make_session():
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def _get(url: str, params: dict = None, timeout: int = 10, ttl: int = 60) -> Optional[Any]:
    """GET with caching (ttl seconds)."""
    global _session
    if _session is None:
        _session = _make_session()

    key = (url, frozenset(params.items()) if params else None)
    now = time.time()
    if key in _cache and now - _cache[key]['ts'] < ttl:
        return _cache[key]['data']

    try:
        resp = _session.get(url, params=params, timeout=timeout)
        # Rate limit backoff
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

        resp.raise_for_status()
        data = resp.json()
        if ttl > 0:
            _cache[key] = {'data': data, 'ts': now}
        return data
    except Exception as e:
        logger.warning(f"_get error: {e}")
        return None

# ---------- Ticker (one call) ----------
def get_all_24h_changes() -> Dict[str, float]:
    """Returns dict {symbol: priceChangePercent} for all USDT perps."""
    data = _get(f"{BASE_URL}/fapi/v1/ticker/24hr", ttl=30)
    if not data:
        return {}
    result = {}
    for t in data:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT") or "_" in sym:
            continue
        try:
            pct = float(t.get("priceChangePercent", 0))
            result[sym] = pct
        except:
            pass
    return result

def get_futures_symbols(min_volume_usdt: float = 5_000_000) -> list[str]:
    """Return liquid USDT perps (volume filter)."""
    data = _get(f"{BASE_URL}/fapi/v1/ticker/24hr", ttl=30)
    if not data:
        return []
    symbols = []
    for t in data:
        sym = t.get("symbol", "")
        try:
            quote_vol = float(t.get("quoteVolume", 0))
            price = float(t.get("lastPrice", 0))
        except:
            continue
        if not sym.endswith("USDT") or "_" in sym:
            continue
        if price <= 0 or quote_vol < min_volume_usdt:
            continue
        symbols.append(sym)
    return symbols

# ---------- KLINES ----------
_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]

def get_klines(symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
    data = _get(f"{BASE_URL}/fapi/v1/klines",
                {"symbol": symbol, "interval": interval, "limit": limit},
                ttl=15)  # 15s cache
    if not data or len(data) < 5:
        return None
    try:
        df = pd.DataFrame(data, columns=_KLINE_COLS)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df = df[df["close"] > 0]
        return df if len(df) >= 5 else None
    except Exception as e:
        logger.warning(f"get_klines({symbol}): {e}")
        return None

def get_historical_klines(symbol: str, interval: str, start_str: str, end_str: str) -> Optional[pd.DataFrame]:
    """Paginated historical klines."""
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)
    all_rows = []
    current = start_ts
    while current < end_ts:
        data = _get(f"{BASE_URL}/fapi/v1/klines", {
            "symbol": symbol, "interval": interval,
            "startTime": current, "endTime": end_ts, "limit": 1500
        }, ttl=0)  # no cache for historical
        if not data:
            break
        all_rows.extend(data)
        current = data[-1][0] + 1
        time.sleep(0.1)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=_KLINE_COLS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df

# ---------- FUNDING (historical) ----------
def get_funding_rate(symbol: str) -> Optional[float]:
    """Current funding rate."""
    data = _get(f"{BASE_URL}/fapi/v1/premiumIndex", {"symbol": symbol}, ttl=30)
    if not data:
        return None
    try:
        rate = float(data.get("lastFundingRate", 0))
        return rate if -0.02 <= rate <= 0.02 else None
    except:
        return None

def get_historical_funding(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Funding rate history between timestamps (ms)."""
    data = _get(f"{BASE_URL}/fapi/v1/fundingInfo", {"symbol": symbol, "limit": 1000}, ttl=0)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    df = df[(df["fundingTime"] >= pd.Timestamp(start_ts, unit="ms", utc=True)) &
            (df["fundingTime"] <= pd.Timestamp(end_ts, unit="ms", utc=True))]
    return df.set_index("fundingTime")

# ---------- OPEN INTEREST (historical) ----------
def get_open_interest_history(symbol: str, period: str = "1h", limit: int = 6) -> Optional[pd.DataFrame]:
    """Recent OI history (max 1000)."""
    data = _get(f"{BASE_URL}/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "limit": limit},
                ttl=30)
    if not data:
        return None
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"])
    return df.set_index("timestamp")

def get_historical_oi(symbol: str, start_ts: int, end_ts: int, period: str = "1h") -> pd.DataFrame:
    """Paginated OI history."""
    all_rows = []
    current = start_ts
    while current < end_ts:
        data = _get(f"{BASE_URL}/futures/data/openInterestHist", {
            "symbol": symbol, "period": period,
            "startTime": current, "endTime": end_ts, "limit": 1000
        }, ttl=0)
        if not data:
            break
        all_rows.extend(data)
        current = data[-1]["timestamp"] + 1
        time.sleep(0.1)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"])
    return df.set_index("timestamp")

# ---------- OI divergence (current) ----------
def is_oi_diverging(symbol: str) -> bool:
    oi_df = get_open_interest_history(symbol, period="1h", limit=6)
    klines = get_klines(symbol, "1h", limit=6)
    if oi_df is None or klines is None or len(oi_df) < 2 or len(klines) < 2:
        return False
    price_chg = klines["close"].iloc[-1] / klines["close"].iloc[0] - 1
    oi_chg = oi_df["sumOpenInterestValue"].iloc[-1] / oi_df["sumOpenInterestValue"].iloc[0] - 1
    return price_chg > 0.01 and oi_chg < -0.01