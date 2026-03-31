"""
backtest.py — Backtesting engine
Runs pump reversal strategy on historical data and reports metrics.

Usage:
    python backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data import get_historical_klines, get_funding_rate, is_oi_diverging
from indicators import (
    calc_rsi,
    calc_atr,
    find_resistance_levels,
    nearest_resistance,
    price_near_resistance,
    is_volume_spike,
    detect_liquidity_sweep,
    calc_pump_percent,
)
from strategy import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Commission per trade (entry + exit), taker fee 0.04% × 2
COMMISSION_PCT = 0.08


# ─────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class Trade:
    entry_time: pd.Timestamp
    symbol: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    exit_price: float
    result: str        # 'TP1', 'TP2', 'SL', 'BE', 'OPEN'
    pnl_pct: float
    rr: float          # risk:reward achieved


# ─────────────────────────────────────────────
# STRATEGY REPLAY
# ─────────────────────────────────────────────

def run_backtest(
    symbol: str,
    start: str,
    end: str,
    interval: str = "4h",
    rsi_thresh: float = Config.RSI_OVERBOUGHT,
    min_pump: float = Config.MIN_PUMP_PCT,
    vol_mult: float = Config.VOLUME_MULTIPLIER,
    res_tol: float = Config.RESISTANCE_TOLERANCE_PCT,
    atr_period: int = Config.ATR_PERIOD,
    sl_mult: float = Config.ATR_SL_MULT,
    tp1_mult: float = Config.ATR_TP1_MULT,
    tp2_mult: float = Config.ATR_TP2_MULT,
    use_live_funding: bool = False,
    use_live_oi: bool = False,
) -> list[Trade]:
    """
    Runs strategy on historical 4H klines and simulates trades.
    Returns list of Trade results.
    """
    logger.info(f"Loading klines for {symbol} [{start} → {end}]")
    df = get_historical_klines(symbol, interval, start, end)

    if df is None or len(df) < 100:
        logger.error("Not enough data for backtest")
        return []

    trades = []
    warmup = 60  # candles needed before first signal

    for i in range(warmup, len(df)):
        window = df.iloc[:i + 1].copy()
        last = window.iloc[-1]
        current_price = float(last["close"])

        # ── Filter 1: Pump ──────────────────────────
        pump_pct = calc_pump_percent(window, candles_back=6)
        if pump_pct < min_pump:
            continue

        # ── Filter 2: Volume spike ──────────────────
        if not is_volume_spike(window, vol_mult):
            continue

        # ── Filter 3: RSI ───────────────────────────
        rsi = calc_rsi(window["close"])
        if rsi < rsi_thresh:
            continue

        # ── Filter 4: Resistance ────────────────────
        levels = find_resistance_levels(window, min_touches=2)
        resistance = nearest_resistance(current_price, levels)
        if not price_near_resistance(current_price, resistance, res_tol):
            continue

        # ── Filter 5: Funding rate (match live strategy) ──
        if use_live_funding:
            funding = get_funding_rate(symbol)
            if funding is None or funding < Config.MIN_FUNDING_RATE:
                continue

        # ── Filter 6: OI divergence ─────────────────
        if use_live_oi:
            if not is_oi_diverging(symbol):
                continue

        # ── Filter 7: Liquidity sweep ───────────────
        if not detect_liquidity_sweep(window):
            continue

        # ── Calculate trade levels ──────────────────
        atr = calc_atr(window, atr_period)
        entry = current_price
        sl = entry + sl_mult * atr
        tp1 = entry - tp1_mult * atr
        tp2 = entry - tp2_mult * atr
        risk = sl - entry

        # ── Simulate forward with BE after TP1 ──────
        result     = "OPEN"
        exit_price = current_price
        be_hit     = False
        current_sl = sl  # sl moves to BE after TP1

        future = df.iloc[i + 1: i + 61]
        for _, fc in future.iterrows():
            high = fc["high"]
            low  = fc["low"]

            hit_sl  = high >= current_sl
            hit_tp2 = low <= tp2
            hit_tp1 = low <= tp1

            # Conservative order: SL/BE first, then TP2
            if hit_sl:
                result = "SL" if not be_hit else "BE"
                exit_price = current_sl
                break
            if hit_tp2:
                result = "TP2"
                exit_price = tp2
                break

            # TP1 hit — move SL to breakeven (no exit)
            if hit_tp1 and not be_hit:
                be_hit     = True
                current_sl = entry  # breakeven

        # ── PnL with commission ─────────────────────
        raw_pnl = (entry - exit_price) / entry * 100
        pnl_pct = raw_pnl - COMMISSION_PCT
        rr = (entry - exit_price) / risk if risk > 0 else 0

        trades.append(Trade(
            entry_time=window.index[-1],
            symbol=symbol,
            entry=entry,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            exit_price=exit_price,
            result=result,
            pnl_pct=pnl_pct,
            rr=rr,
        ))

    return trades


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(trades: list[Trade]) -> dict:
    """Calculates backtest performance metrics."""
    if not trades:
        return {}

    closed = [t for t in trades if t.result != "OPEN"]
    if not closed:
        return {"total": len(trades), "closed": 0}

    winners = [t for t in closed if t.pnl_pct > 0]
    losers  = [t for t in closed if t.pnl_pct < 0]
    # BE = breakeven, small loss due to commission only

    gross_profit = sum(t.pnl_pct for t in winners)
    gross_loss = abs(sum(t.pnl_pct for t in losers))

    # Equity curve
    equity = np.cumsum([t.pnl_pct for t in closed])
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = float(np.max(drawdown))

    metrics = {
        "total_signals": len(trades),
        "closed_trades": len(closed),
        "winners": len(winners),
        "losers": len(losers),
        "winrate_pct": len(winners) / len(closed) * 100,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_win_pct": np.mean([t.pnl_pct for t in winners]) if winners else 0,
        "avg_loss_pct": np.mean([t.pnl_pct for t in losers]) if losers else 0,
        "avg_rr": np.mean([t.rr for t in closed]),
        "max_drawdown_pct": max_dd,
        "total_pnl_pct": float(equity[-1]) if len(equity) > 0 else 0,
    }

    return metrics


def print_metrics(metrics: dict, symbol: str):
    print("\n" + "═" * 50)
    print(f"  BACKTEST RESULTS — {symbol}")
    print("═" * 50)
    print(f"  Total signals:    {metrics.get('total_signals', 0)}")
    print(f"  Closed trades:    {metrics.get('closed_trades', 0)}")
    print(f"  Win rate:         {metrics.get('winrate_pct', 0):.1f}%")
    print(f"  Profit factor:    {metrics.get('profit_factor', 0):.2f}")
    print(f"  Avg win:          +{metrics.get('avg_win_pct', 0):.2f}%")
    print(f"  Avg loss:         {metrics.get('avg_loss_pct', 0):.2f}%")
    print(f"  Avg R:R achieved: {metrics.get('avg_rr', 0):.2f}")
    print(f"  Max drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Total PnL:        {metrics.get('total_pnl_pct', 0):.2f}%")
    print(f"  (incl. 0.08% commission per trade)")
    print("═" * 50 + "\n")


# ─────────────────────────────────────────────
# EQUITY CURVE CHART
# ─────────────────────────────────────────────

def plot_equity_curve(trades: list[Trade], symbol: str, save_path: str = "backtest_equity.png"):
    closed = [t for t in trades if t.result != "OPEN"]
    if not closed:
        return

    timestamps = [t.entry_time for t in closed]
    equity = np.cumsum([t.pnl_pct for t in closed])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor="#0d0d0d")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Equity curve
    ax1.plot(timestamps, equity, color="#00e5ff", linewidth=2, label="Equity %")
    ax1.fill_between(timestamps, equity, 0, alpha=0.15, color="#00e5ff")
    ax1.axhline(0, color="#555", linewidth=1, linestyle="--")
    ax1.set_title(f"{symbol} — Backtest Equity Curve", color="white", fontsize=13, pad=12)
    ax1.set_ylabel("Cumulative PnL %", color="#aaa")
    ax1.tick_params(colors="#aaa")
    ax1.legend(facecolor="#1a1a1a", labelcolor="white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Trade results bar chart
    colors = {"TP2": "#00e676", "TP1": "#69f0ae", "SL": "#ff1744", "OPEN": "#888"}
    bar_colors = [colors.get(t.result, "#888") for t in closed]
    ax2.bar(timestamps, [t.pnl_pct for t in closed], color=bar_colors, width=3)
    ax2.axhline(0, color="#555", linewidth=1, linestyle="--")
    ax2.set_title("Individual Trade PnL %", color="white", fontsize=11, pad=8)
    ax2.set_ylabel("PnL %", color="#aaa")
    ax2.tick_params(colors="#aaa")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    logger.info(f"Equity curve saved: {save_path}")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pump Reversal Backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to backtest")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--use-live-funding",
        action="store_true",
        help="Use live funding filter (not historical; may skew results)",
    )
    parser.add_argument(
        "--use-live-oi",
        action="store_true",
        help="Use live OI divergence filter (not historical; may skew results)",
    )
    args = parser.parse_args()

    trades = run_backtest(
        args.symbol,
        args.start,
        args.end,
        use_live_funding=args.use_live_funding,
        use_live_oi=args.use_live_oi,
    )

    if not trades:
        print("No trades generated.")
        sys.exit(0)

    metrics = compute_metrics(trades)
    print_metrics(metrics, args.symbol)
    plot_equity_curve(trades, args.symbol)

    # Save trades to CSV
    rows = [
        {
            "date": t.entry_time,
            "symbol": t.symbol,
            "entry": t.entry,
            "stop_loss": t.stop_loss,
            "tp1": t.tp1,
            "tp2": t.tp2,
            "exit_price": t.exit_price,
            "result": t.result,
            "pnl_pct": round(t.pnl_pct, 3),
            "rr": round(t.rr, 2),
        }
        for t in trades
    ]
    df_out = pd.DataFrame(rows)
    df_out.to_csv(f"backtest_{args.symbol}.csv", index=False)
    print(f"Trades saved to backtest_{args.symbol}.csv")
