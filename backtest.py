"""
backtest.py - Backtesting engine

Supports:
- single-symbol backtests
- portfolio backtests across many symbols
- summary reports close to the Telegram format used by the bot
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import get_futures_market_snapshot, get_historical_klines
from indicators import calc_pump_percent
from strategy import build_daily_frame, evaluate_price_action_setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    exit_price: float
    result: str
    pnl_pct: float
    rr: float
    bars_held: int


def _net_return(entry_exec: float, exit_exec: float, fee_pct: float) -> float:
    gross = (entry_exec - exit_exec) / entry_exec
    return gross - (2 * fee_pct)


def _winrate_bar(winrate: float, width: int = 16) -> str:
    filled = round(winrate / 100 * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _trade_label(trade: Optional[Trade]) -> str:
    if trade is None:
        return "—"
    sign = "+" if trade.pnl_pct >= 0 else ""
    return f"{trade.symbol} {sign}{trade.pnl_pct:.1f}% ({trade.result})"


def simulate_trade(
    symbol: str,
    entry_time: pd.Timestamp,
    future: pd.DataFrame,
    entry: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    fee_pct: float,
    slippage_pct: float,
) -> Trade:
    """
    Conservative short simulation.

    Result labels are user-facing and mutually exclusive:
    - TP2: final target hit
    - TP1: TP1 hit, TP2 not hit, remaining size closed on timeout
    - BU: TP1 hit, remaining size stopped at break-even
    - SL: stop loss hit before TP1
    - TIME: no TP1/TP2/SL before timeout
    """
    entry_exec = entry * (1 - slippage_pct)
    risk_per_trade = max((stop_loss * (1 + slippage_pct) - entry_exec) / entry_exec, 1e-9)

    took_tp1 = False
    realized_return = 0.0
    remaining_size = 1.0
    exit_price = float(future["close"].iloc[-1])
    exit_time = future.index[-1]

    for bars_held, (ts, candle) in enumerate(future.iterrows(), start=1):
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        if not took_tp1:
            if high >= stop_loss:
                exit_exec = stop_loss * (1 + slippage_pct)
                realized_return = remaining_size * _net_return(entry_exec, exit_exec, fee_pct)
                return Trade(
                    entry_time=entry_time,
                    exit_time=ts,
                    symbol=symbol,
                    entry=entry,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2,
                    exit_price=exit_exec,
                    result="SL",
                    pnl_pct=realized_return * 100,
                    rr=realized_return / risk_per_trade,
                    bars_held=bars_held,
                )

            if low <= tp1:
                tp1_exec = tp1 * (1 + slippage_pct)
                realized_return += 0.5 * _net_return(entry_exec, tp1_exec, fee_pct)
                remaining_size = 0.5
                took_tp1 = True

                if high >= entry:
                    be_exec = entry * (1 + slippage_pct)
                    realized_return += remaining_size * _net_return(entry_exec, be_exec, fee_pct)
                    return Trade(
                        entry_time=entry_time,
                        exit_time=ts,
                        symbol=symbol,
                        entry=entry,
                        stop_loss=stop_loss,
                        tp1=tp1,
                        tp2=tp2,
                        exit_price=be_exec,
                        result="BU",
                        pnl_pct=realized_return * 100,
                        rr=realized_return / risk_per_trade,
                        bars_held=bars_held,
                    )

                if low <= tp2:
                    tp2_exec = tp2 * (1 + slippage_pct)
                    realized_return += remaining_size * _net_return(entry_exec, tp2_exec, fee_pct)
                    return Trade(
                        entry_time=entry_time,
                        exit_time=ts,
                        symbol=symbol,
                        entry=entry,
                        stop_loss=stop_loss,
                        tp1=tp1,
                        tp2=tp2,
                        exit_price=tp2_exec,
                        result="TP2",
                        pnl_pct=realized_return * 100,
                        rr=realized_return / risk_per_trade,
                        bars_held=bars_held,
                    )

        else:
            if high >= entry:
                be_exec = entry * (1 + slippage_pct)
                realized_return += remaining_size * _net_return(entry_exec, be_exec, fee_pct)
                return Trade(
                    entry_time=entry_time,
                    exit_time=ts,
                    symbol=symbol,
                    entry=entry,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2,
                    exit_price=be_exec,
                    result="BU",
                    pnl_pct=realized_return * 100,
                    rr=realized_return / risk_per_trade,
                    bars_held=bars_held,
                )

            if low <= tp2:
                tp2_exec = tp2 * (1 + slippage_pct)
                realized_return += remaining_size * _net_return(entry_exec, tp2_exec, fee_pct)
                return Trade(
                    entry_time=entry_time,
                    exit_time=ts,
                    symbol=symbol,
                    entry=entry,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2,
                    exit_price=tp2_exec,
                    result="TP2",
                    pnl_pct=realized_return * 100,
                    rr=realized_return / risk_per_trade,
                    bars_held=bars_held,
                )

        exit_price = close
        exit_time = ts

    timeout_exec = exit_price * (1 + slippage_pct)
    realized_return += remaining_size * _net_return(entry_exec, timeout_exec, fee_pct)
    result = "TP1" if took_tp1 else "TIME"
    bars_held = len(future)
    return Trade(
        entry_time=entry_time,
        exit_time=exit_time,
        symbol=symbol,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        exit_price=timeout_exec,
        result=result,
        pnl_pct=realized_return * 100,
        rr=realized_return / risk_per_trade,
        bars_held=bars_held,
    )


def run_backtest(
    symbol: str,
    start: str,
    end: str,
    interval: str = "4h",
    fee_bps: float = 4.0,
    slippage_bps: float = 3.0,
    max_hold_bars: int = 24,
) -> list[Trade]:
    logger.info(f"Loading klines for {symbol} [{start} -> {end}]")
    df = get_historical_klines(symbol, interval, start, end)

    if df is None or len(df) < 120:
        logger.warning(f"{symbol}: not enough data for backtest")
        return []

    fee_pct = fee_bps / 10000
    slippage_pct = slippage_bps / 10000
    trades: list[Trade] = []
    warmup = 60
    i = warmup

    while i < len(df) - 1:
        window = df.iloc[:i + 1].copy()
        pump_pct = calc_pump_percent(window, candles_back=6)
        df_1d = build_daily_frame(window)
        setup = evaluate_price_action_setup(symbol, pump_pct, window, df_1d)

        if setup is None:
            i += 1
            continue

        future = df.iloc[i + 1:i + 1 + max_hold_bars]
        if future.empty:
            break

        trade = simulate_trade(
            symbol=symbol,
            entry_time=window.index[-1],
            future=future,
            entry=setup.entry,
            stop_loss=setup.stop_loss,
            tp1=setup.tp1,
            tp2=setup.tp2,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
        )
        trades.append(trade)
        i += max(1, trade.bars_held)

    return trades


def run_portfolio_backtest(
    start: str,
    end: str,
    max_symbols: int = 0,
    min_volume_usdt: float = 5_000_000,
    fee_bps: float = 4.0,
    slippage_bps: float = 3.0,
    max_hold_bars: int = 24,
) -> tuple[list[Trade], list[str]]:
    """
    Runs the backtest across many symbols and aggregates all trades.
    Symbols come from the current liquid futures universe.
    """
    market = get_futures_market_snapshot(min_volume_usdt=min_volume_usdt)
    symbols = [item["symbol"] for item in market]
    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    logger.info(f"Portfolio backtest: {len(symbols)} symbols [{start} -> {end}]")
    all_trades: list[Trade] = []
    active_symbols: list[str] = []

    for idx, symbol in enumerate(symbols, start=1):
        logger.info(f"[{idx}/{len(symbols)}] Backtesting {symbol}")
        trades = run_backtest(
            symbol=symbol,
            start=start,
            end=end,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_hold_bars=max_hold_bars,
        )
        if trades:
            all_trades.extend(trades)
            active_symbols.append(symbol)

    all_trades.sort(key=lambda t: (t.entry_time, t.symbol))
    return all_trades, active_symbols


def compute_metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {}

    pnl_series = np.array([t.pnl_pct for t in trades], dtype=float)
    equity = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity

    winners = [t for t in trades if t.pnl_pct > 0]
    losers = [t for t in trades if t.pnl_pct < 0]
    best_trade = max(trades, key=lambda t: t.pnl_pct, default=None)
    worst_trade = min(trades, key=lambda t: t.pnl_pct, default=None)

    gross_profit = sum(t.pnl_pct for t in winners)
    gross_loss = abs(sum(t.pnl_pct for t in losers))
    decisive = len(winners) + len(losers)
    winrate = (len(winners) / decisive * 100) if decisive > 0 else 0.0

    return {
        "total_trades": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "winrate_pct": winrate,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_win_pct": np.mean([t.pnl_pct for t in winners]) if winners else 0,
        "avg_loss_pct": np.mean([t.pnl_pct for t in losers]) if losers else 0,
        "avg_rr": np.mean([t.rr for t in trades]) if trades else 0,
        "max_drawdown_pct": float(np.max(drawdown)) if len(drawdown) else 0,
        "total_pnl_pct": float(np.sum(pnl_series)),
        "avg_bars_held": np.mean([t.bars_held for t in trades]) if trades else 0,
        "tp2_count": sum(1 for t in trades if t.result == "TP2"),
        "tp1_count": sum(1 for t in trades if t.result == "TP1"),
        "be_count": sum(1 for t in trades if t.result == "BU"),
        "sl_count": sum(1 for t in trades if t.result == "SL"),
        "time_count": sum(1 for t in trades if t.result == "TIME"),
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "best_trade_str": _trade_label(best_trade),
        "worst_trade_str": _trade_label(worst_trade),
    }


def backtest_verdict(metrics: dict) -> str:
    if not metrics:
        return "⚪️ Недостаточно данных"
    if metrics["winrate_pct"] >= 55 and metrics["profit_factor"] >= 1.5:
        return "🟢 Стратегия прибыльная"
    if metrics["winrate_pct"] >= 45 and metrics["profit_factor"] >= 1.0:
        return "🟡 Стратегия в плюсе"
    return "🔴 Нужна донастройка"


def format_backtest_report(
    metrics: dict,
    generated_at: Optional[datetime] = None,
    title: str = "БЭКТЕСТ РЕЗУЛЬТАТЫ",
    subtitle: Optional[str] = None,
) -> str:
    generated_at = generated_at or datetime.utcnow()
    now_str = generated_at.strftime("%d.%m.%Y %H:%M UTC")
    bar = _winrate_bar(metrics.get("winrate_pct", 0))
    verdict = backtest_verdict(metrics)

    lines = [
        f"📊 <b>{title}</b>",
        f"🕐 {now_str}",
    ]
    if subtitle:
        lines.append(f"🪙 <b>{subtitle}</b>")

    lines.extend([
        "━━━━━━━━━━━━━━━━━━━━",
        "",
        "🏆 <b>WIN RATE</b>",
        f"<code>{bar}</code>  <b>{metrics.get('winrate_pct', 0):.1f}%</b>",
        "",
        f"🏅 TP2 закрыто:  <b>{metrics.get('tp2_count', 0)}</b>",
        f"✅ TP1 закрыто:  <b>{metrics.get('tp1_count', 0)}</b>",
        f"🔄 Безубыток:    <b>{metrics.get('be_count', 0)}</b>",
        f"❌ SL сработало: <b>{metrics.get('sl_count', 0)}</b>",
        f"⏱ Тайм-аут:     <b>{metrics.get('time_count', 0)}</b>",
        f"📊 Всего:        <b>{metrics.get('total_trades', 0)}</b>",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "💰 <b>P&amp;L</b>",
        f"📉 Итого P&amp;L:      <b>{metrics.get('total_pnl_pct', 0):+.2f}%</b>",
        f"📈 Средний выигрыш: <b>+{metrics.get('avg_win_pct', 0):.2f}%</b>",
        f"📉 Средний стоп:    <b>{metrics.get('avg_loss_pct', 0):.2f}%</b>",
        f"⚖️ Risk/Reward:     <b>{metrics.get('avg_rr', 0):.2f}</b>",
        f"📊 Profit Factor:   <b>{metrics.get('profit_factor', 0):.2f}</b>",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "⭐️ <b>РЕКОРДЫ</b>",
        f"🥇 Лучший:  {metrics.get('best_trade_str', '—')}",
        f"💀 Худший:  {metrics.get('worst_trade_str', '—')}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        verdict,
    ])
    return "\n".join(lines)


def print_metrics(metrics: dict, label: str):
    print("\n" + "=" * 54)
    print(f"  BACKTEST RESULTS - {label}")
    print("=" * 54)
    print(f"  Total trades:      {metrics.get('total_trades', 0)}")
    print(f"  Win rate:          {metrics.get('winrate_pct', 0):.1f}%")
    print(f"  Profit factor:     {metrics.get('profit_factor', 0):.2f}")
    print(f"  Avg win:           +{metrics.get('avg_win_pct', 0):.2f}%")
    print(f"  Avg loss:          {metrics.get('avg_loss_pct', 0):.2f}%")
    print(f"  Avg R multiple:    {metrics.get('avg_rr', 0):.2f}")
    print(f"  Max drawdown:      {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Total PnL:         {metrics.get('total_pnl_pct', 0):.2f}%")
    print(f"  TP2 / TP1 / BU:    {metrics.get('tp2_count', 0)} / {metrics.get('tp1_count', 0)} / {metrics.get('be_count', 0)}")
    print(f"  SL / TIME:         {metrics.get('sl_count', 0)} / {metrics.get('time_count', 0)}")
    print(f"  Best:              {metrics.get('best_trade_str', '—')}")
    print(f"  Worst:             {metrics.get('worst_trade_str', '—')}")
    print("=" * 54 + "\n")


def plot_equity_curve(trades: list[Trade], label: str, save_path: str = "backtest_equity.png"):
    if not trades:
        return

    timestamps = [t.exit_time for t in trades]
    equity = np.cumsum(np.array([t.pnl_pct for t in trades], dtype=float))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor="#0d0d0d")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    ax1.plot(timestamps, equity, color="#00e5ff", linewidth=2, label="Equity %")
    ax1.fill_between(timestamps, equity, 0, alpha=0.15, color="#00e5ff")
    ax1.axhline(0, color="#555", linewidth=1, linestyle="--")
    ax1.set_title(f"{label} - Backtest Equity Curve", color="white", fontsize=13, pad=12)
    ax1.set_ylabel("Cumulative PnL %", color="#aaa")
    ax1.tick_params(colors="#aaa")
    ax1.legend(facecolor="#1a1a1a", labelcolor="white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    colors = {
        "TP2": "#00e676",
        "TP1": "#69f0ae",
        "BU": "#00c853",
        "SL": "#ff1744",
        "TIME": "#888",
    }
    ax2.bar(
        timestamps,
        [t.pnl_pct for t in trades],
        color=[colors.get(t.result, "#888") for t in trades],
        width=3,
    )
    ax2.axhline(0, color="#555", linewidth=1, linestyle="--")
    ax2.set_title("Trade PnL %", color="white", fontsize=11, pad=8)
    ax2.set_ylabel("PnL %", color="#aaa")
    ax2.tick_params(colors="#aaa")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    logger.info(f"Equity curve saved: {save_path}")


def export_trades_csv(trades: list[Trade], out_path: str):
    df_out = pd.DataFrame([
        {
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "symbol": t.symbol,
            "entry": t.entry,
            "stop_loss": t.stop_loss,
            "tp1": t.tp1,
            "tp2": t.tp2,
            "exit_price": round(t.exit_price, 8),
            "result": t.result,
            "pnl_pct": round(t.pnl_pct, 3),
            "rr": round(t.rr, 2),
            "bars_held": t.bars_held,
        }
        for t in trades
    ])
    df_out.to_csv(out_path, index=False)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Pump Reversal Backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to backtest, or ALL")
    parser.add_argument("--all", action="store_true", help="Run portfolio backtest across many symbols")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--fee-bps", type=float, default=4.0, help="Round-trip fee assumption in basis points")
    parser.add_argument("--slippage-bps", type=float, default=3.0, help="Entry/exit slippage in basis points")
    parser.add_argument("--max-symbols", type=int, default=0, help="Limit portfolio backtest universe (0 = all)")
    args = parser.parse_args()

    run_all = args.all or args.symbol.upper() == "ALL"
    if run_all:
        trades, active_symbols = run_portfolio_backtest(
            start=args.start,
            end=args.end,
            max_symbols=args.max_symbols,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )
        label = f"ALL ({len(active_symbols)} symbols)"
        out_path = "backtest_ALL.csv"
    else:
        trades = run_backtest(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )
        label = args.symbol
        out_path = f"backtest_{args.symbol}.csv"

    if not trades:
        print("No trades generated.")
        sys.exit(0)

    metrics = compute_metrics(trades)
    print_metrics(metrics, label)
    plot_equity_curve(trades, label)
    export_trades_csv(trades, out_path)
    print(format_backtest_report(metrics, title="БЭКТЕСТ РЕЗУЛЬТАТЫ", subtitle=label))
    print(f"Trades saved to {out_path}")
