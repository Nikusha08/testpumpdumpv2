"""
backtest.py — Backtesting engine with historical OI and funding.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data import get_historical_klines, get_historical_funding, get_historical_oi
from indicators import (
    calc_rsi, calc_atr, find_resistance_levels, nearest_resistance,
    price_near_resistance, is_volume_spike, detect_liquidity_sweep,
    calc_pump_percent, get_volume_ratio, is_volume_climax,
    is_weak_after_pump, is_pin_bar, detect_rsi_divergence, get_macd,
    detect_macd_divergence
)
from strategy import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COMMISSION_PCT = 0.08

@dataclass
class Trade:
    entry_time: pd.Timestamp
    symbol: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    exit_price: float
    result: str
    pnl_pct: float
    rr: float

def get_funding_at_time(symbol: str, ts: pd.Timestamp) -> float:
    start = int(ts.timestamp() * 1000) - 2 * 3600 * 1000
    end = int(ts.timestamp() * 1000) + 2 * 3600 * 1000
    df = get_historical_funding(symbol, start, end)
    if df.empty:
        return 0.0
    df = df[df.index <= ts]
    if df.empty:
        return 0.0
    return float(df.iloc[-1]["fundingRate"])

def get_oi_at_time(symbol: str, ts: pd.Timestamp) -> float:
    start = int(ts.timestamp() * 1000) - 2 * 3600 * 1000
    end = int(ts.timestamp() * 1000) + 2 * 3600 * 1000
    df = get_historical_oi(symbol, start, end, period="1h")
    if df.empty:
        return 0.0
    df = df[df.index <= ts]
    if df.empty:
        return 0.0
    return float(df.iloc[-1]["sumOpenInterestValue"])

def oi_divergence_at_time(symbol: str, ts: pd.Timestamp) -> bool:
    end = int(ts.timestamp() * 1000)
    start = end - 6 * 3600 * 1000
    oi_df = get_historical_oi(symbol, start, end, period="1h")
    klines = get_historical_klines(symbol, "1h",
                                   pd.Timestamp(start, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S"),
                                   pd.Timestamp(end, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S"))
    if oi_df.empty or klines is None or len(oi_df) < 2 or len(klines) < 2:
        return False
    price_chg = klines["close"].iloc[-1] / klines["close"].iloc[0] - 1
    oi_chg = oi_df["sumOpenInterestValue"].iloc[-1] / oi_df["sumOpenInterestValue"].iloc[0] - 1
    return price_chg > 0.01 and oi_chg < -0.01

def run_backtest(symbol: str, start: str, end: str, interval: str = "4h") -> list[Trade]:
    logger.info(f"Loading klines for {symbol} [{start} → {end}]")
    df = get_historical_klines(symbol, interval, start, end)
    if df is None or len(df) < 100:
        logger.error("Not enough data")
        return []

    trades = []
    warmup = 60

    for i in range(warmup, len(df)):
        window = df.iloc[:i+1].copy()
        last = window.iloc[-1]
        current_price = float(last["close"])
        ts = window.index[-1]

        # 1. Pump
        pump_pct = calc_pump_percent(window, candles_back=6)
        if pump_pct < Config.MIN_PUMP_PCT:
            continue

        # 2. Volume spike
        if not is_volume_spike(window, Config.VOLUME_MULTIPLIER):
            continue

        # 3. RSI
        rsi = calc_rsi(window["close"])
        if rsi < Config.RSI_OVERBOUGHT:
            continue

        # 4. Resistance
        levels = find_resistance_levels(window, min_touches=2)
        resistance = nearest_resistance(current_price, levels)
        if not price_near_resistance(current_price, resistance, Config.RESISTANCE_TOLERANCE):
            continue

        # 5. Funding at that time
        funding = get_funding_at_time(symbol, ts)
        if funding < Config.MIN_FUNDING_RATE:
            continue

        # 6. OI divergence
        if not oi_divergence_at_time(symbol, ts):
            continue

        # 7. Liquidity sweep (4H)
        if not detect_liquidity_sweep(window):
            continue

        # 8. Additional filters (optional – you can enable more)
        # pin_4h = is_pin_bar(window)
        # weak = is_weak_after_pump(window)
        # vol_climax = is_volume_climax(window)
        # We'll keep it simple for backtest, but you can add them as conditions

        # Trade levels
        atr = calc_atr(window, Config.ATR_PERIOD)
        entry = current_price
        sl = entry + Config.ATR_SL_MULT * atr
        tp1 = entry - Config.ATR_TP1_MULT * atr
        tp2 = entry - Config.ATR_TP2_MULT * atr
        risk = sl - entry

        # Simulate forward
        result = "OPEN"
        exit_price = current_price
        be_hit = False
        current_sl = sl
        future = df.iloc[i+1 : i+61]

        for _, candle in future.iterrows():
            high = candle["high"]
            low = candle["low"]

            # TP2 first
            if low <= tp2:
                result = "TP2"
                exit_price = tp2
                break

            # TP1 hit -> move SL to entry
            if not be_hit and low <= tp1:
                be_hit = True
                current_sl = entry

            # SL check
            if high >= current_sl:
                result = "BE" if be_hit else "SL"
                exit_price = current_sl
                break

        raw_pnl = (entry - exit_price) / entry * 100
        pnl_pct = raw_pnl - COMMISSION_PCT
        rr = (entry - exit_price) / risk if risk > 0 else 0

        trades.append(Trade(
            entry_time=ts, symbol=symbol, entry=entry, stop_loss=sl, tp1=tp1, tp2=tp2,
            exit_price=exit_price, result=result, pnl_pct=pnl_pct, rr=rr
        ))

    return trades

def compute_metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {}
    closed = [t for t in trades if t.result != "OPEN"]
    if not closed:
        return {"total": len(trades), "closed": 0}
    winners = [t for t in closed if t.pnl_pct > 0]
    losers = [t for t in closed if t.pnl_pct < 0]
    gross_profit = sum(t.pnl_pct for t in winners)
    gross_loss = abs(sum(t.pnl_pct for t in losers))
    equity = np.cumsum([t.pnl_pct for t in closed])
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity))
    metrics = {
        "total_signals": len(trades),
        "closed_trades": len(closed),
        "winners": len(winners),
        "losers": len(losers),
        "winrate_pct": len(winners)/len(closed)*100,
        "profit_factor": gross_profit/gross_loss if gross_loss>0 else float("inf"),
        "avg_win_pct": np.mean([t.pnl_pct for t in winners]) if winners else 0,
        "avg_loss_pct": np.mean([t.pnl_pct for t in losers]) if losers else 0,
        "avg_rr": np.mean([t.rr for t in closed]),
        "max_drawdown_pct": max_dd,
        "total_pnl_pct": float(equity[-1]) if len(equity)>0 else 0,
    }
    return metrics

def print_metrics(metrics: dict, symbol: str):
    print("\n" + "═"*50)
    print(f"  BACKTEST RESULTS — {symbol}")
    print("═"*50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    print("═"*50)

def plot_equity_curve(trades: list[Trade], symbol: str, save_path: str = "backtest_equity.png"):
    closed = [t for t in trades if t.result != "OPEN"]
    if not closed:
        return
    timestamps = [t.entry_time for t in closed]
    equity = np.cumsum([t.pnl_pct for t in closed])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor="#0d0d0d")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0d0d0d")
    ax1.plot(timestamps, equity, color="#00e5ff", linewidth=2)
    ax1.fill_between(timestamps, equity, 0, alpha=0.15, color="#00e5ff")
    ax1.axhline(0, color="#555", linestyle="--")
    ax1.set_title(f"{symbol} — Equity Curve", color="white")
    colors = {"TP2": "#00e676", "TP1": "#69f0ae", "SL": "#ff1744", "BE": "#ffd600"}
    ax2.bar(timestamps, [t.pnl_pct for t in closed], color=[colors.get(t.result, "#888") for t in closed], width=3)
    ax2.axhline(0, color="#555", linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="#0d0d0d")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    trades = run_backtest(args.symbol, args.start, args.end)
    if not trades:
        print("No trades.")
        sys.exit(0)
    metrics = compute_metrics(trades)
    print_metrics(metrics, args.symbol)
    plot_equity_curve(trades, args.symbol)
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(f"backtest_{args.symbol}.csv", index=False)