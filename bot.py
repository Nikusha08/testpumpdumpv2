"""
bot.py — Main entry point
Telegram bot: scanner, chart generator, signal sender, CSV logger.

Environment variables:
    TG_TOKEN           — Telegram bot token
    TG_CHAT            — Telegram chat ID
    SCAN_INTERVAL_SEC  — scan frequency seconds (default: 300)
    MIN_PUMP_PCT       — override pump threshold (default: 25)
    MIN_SCORE          — override min score (default: 3)
"""

import csv
import io
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests

from data import get_24h_tickers, get_latest_prices
from strategy import analyze_symbol, Signal, Config
from indicators import calc_rsi
from paper_trading import (
    open_paper_trade,
    update_open_trades,
    get_paper_report,
    reset_paper_account,
    get_open_trade_symbols,
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log"),
    ],
)
logger = logging.getLogger("bot")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TG_TOKEN        = os.environ.get("TG_TOKEN", "")
TG_CHAT         = os.environ.get("TG_CHAT", "")
SCAN_INTERVAL   = int(os.environ.get("SCAN_INTERVAL_SEC", "300"))
SIGNALS_CSV     = Path("signals.csv")
SIGNAL_COOLDOWN  = 7200
COOLDOWN_FILE    = Path("cooldown.json")
_signal_cache: dict[str, float] = {}


def _load_cooldown():
    """Load cooldown state from disk on startup."""
    global _signal_cache
    try:
        if COOLDOWN_FILE.exists():
            import json
            data = json.loads(COOLDOWN_FILE.read_text())
            now  = time.time()
            # Only keep entries that are still within cooldown window
            _signal_cache = {k: v for k, v in data.items() if now - v < SIGNAL_COOLDOWN}
            logger.info(f"Cooldown loaded: {len(_signal_cache)} active symbols")
    except Exception as e:
        logger.warning(f"_load_cooldown: {e}")


def _save_cooldown():
    """Save cooldown state to disk."""
    try:
        import json
        COOLDOWN_FILE.write_text(json.dumps(_signal_cache))
    except Exception as e:
        logger.warning(f"_save_cooldown: {e}")

if os.environ.get("MIN_PUMP_PCT"):
    Config.MIN_PUMP_PCT = float(os.environ["MIN_PUMP_PCT"])
if os.environ.get("MIN_SCORE"):
    Config.MIN_SCORE = int(os.environ["MIN_SCORE"])

_signal_cache: dict[str, float] = {}


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

def tg_send_message(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT:
        logger.warning("TG_TOKEN/TG_CHAT not set")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TG_CHAT,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"tg_send_message: {e}")
        return False


def tg_send_photo(image_bytes: bytes, caption: str = "") -> bool:
    if not TG_TOKEN or not TG_CHAT:
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        resp = requests.post(
            url,
            data={"chat_id": TG_CHAT, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"tg_send_photo: {e}")
        return False


# ─────────────────────────────────────────────
# CHART GENERATOR
# ─────────────────────────────────────────────

BG      = "#0d0d0d"
GREEN   = "#00e676"
RED     = "#ff1744"
YELLOW  = "#ffd600"
ORANGE  = "#ff9800"
PURPLE  = "#e040fb"
CYAN    = "#00e5ff"
WHITE   = "#ffffff"
GREY    = "#555555"


def _price_fmt(price: float) -> str:
    """Format price nicely regardless of magnitude."""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.5f}"
    else:
        return f"{price:.8f}"


def generate_chart(signal: Signal) -> Optional[bytes]:
    """
    Generates a dark-themed chart with:
    - Candlestick bars (last 60 × 4H candles)
    - Entry / SL / TP1 / TP2 horizontal lines with labels
    - Resistance levels
    - RSI subplot
    - Volume bars
    """
    try:
        df = signal.df_4h.iloc[-60:].copy()
        n  = len(df)

        if n < 10:
            logger.warning(f"generate_chart: too few candles ({n})")
            return None

        # Validate all OHLCV values before drawing
        if df[["open", "high", "low", "close", "volume"]].isnull().any().any():
            logger.warning("generate_chart: NaN in candle data")
            return None

        # ── Layout ─────────────────────────────────
        fig = plt.figure(figsize=(14, 10), facecolor=BG)
        gs  = fig.add_gridspec(3, 1, height_ratios=[3, 0.8, 0.8], hspace=0.08)

        ax_price  = fig.add_subplot(gs[0])
        ax_vol    = fig.add_subplot(gs[1], sharex=ax_price)
        ax_rsi    = fig.add_subplot(gs[2], sharex=ax_price)

        for ax in [ax_price, ax_vol, ax_rsi]:
            ax.set_facecolor(BG)
            ax.tick_params(colors="#888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a2a")

        x = np.arange(n)

        # ── Candlesticks ───────────────────────────
        for i in range(n):
            row   = df.iloc[i]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = GREEN if c >= o else RED

            # Wick
            ax_price.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)

            # Body
            body_h = abs(c - o)
            body_y = min(o, c)
            if body_h < (h - l) * 0.01:  # doji — thin line
                body_h = (h - l) * 0.015
            rect = mpatches.Rectangle(
                (i - 0.38, body_y), 0.76, body_h,
                linewidth=0, facecolor=color, zorder=3
            )
            ax_price.add_patch(rect)

        # ── Horizontal trade levels ─────────────────
        trade_levels = [
            (signal.stop_loss, RED,       "--", 2.0, f"SL  {_price_fmt(signal.stop_loss)}"),
            (signal.entry,     WHITE,     "--", 1.8, f"Entry  {_price_fmt(signal.entry)}"),
            (signal.entry,     "#ffd600", ":",  1.2, f"BE  {_price_fmt(signal.entry)}"),
            (signal.tp1,       GREEN,     ":",  1.5, f"TP1  {_price_fmt(signal.tp1)}"),
            (signal.tp2,       "#00c853", ":",  2.0, f"TP2  {_price_fmt(signal.tp2)}"),
        ]

        for price_level, color, ls, lw, label in trade_levels:
            ax_price.axhline(price_level, color=color, linewidth=lw,
                             linestyle=ls, alpha=0.95, zorder=4)
            ax_price.text(
                n + 0.3, price_level, label,
                color=color, fontsize=7.5, va="center", fontweight="bold"
            )

        # ── Resistance zones (жирные, с зоной ±0.5%) ──
        res_levels = []
        if signal.resistance_4h:
            res_levels.append((signal.resistance_4h, "#ff9800", "4H"))
        if signal.resistance_1d:
            res_levels.append((signal.resistance_1d, "#ff5722", "1D"))

        for res_price, res_color, res_tf in res_levels:
            # Зона ±0.5%
            zone_h = res_price * 0.005
            ax_price.axhspan(
                res_price - zone_h, res_price + zone_h,
                alpha=0.18, color=res_color, zorder=2
            )
            # Центральная линия
            ax_price.axhline(res_price, color=res_color, linewidth=2.0,
                             linestyle="-", alpha=0.9, zorder=5)
            # Подпись с таймфреймом
            ax_price.text(
                n + 0.3, res_price,
                f"Res {res_tf}  {_price_fmt(res_price)}",
                color=res_color, fontsize=8, va="center", fontweight="bold"
            )

        # ── Zone shading ────────────────────────────
        ax_price.axhspan(signal.entry, signal.stop_loss, alpha=0.06, color=RED,   zorder=1)
        ax_price.axhspan(signal.tp2,   signal.entry,     alpha=0.04, color=GREEN, zorder=1)

        # ── Price axis limits ───────────────────────
        all_prices = list(df["high"]) + list(df["low"]) + [signal.stop_loss, signal.tp2]
        if signal.resistance_4h: all_prices.append(signal.resistance_4h)
        if signal.resistance_1d: all_prices.append(signal.resistance_1d)
        price_min  = min(all_prices) * 0.994
        price_max  = max(all_prices) * 1.006
        ax_price.set_ylim(price_min, price_max)
        ax_price.set_xlim(-1, n + 10)
        ax_price.set_ylabel("Price (USDT)", color="#888", fontsize=9)

        # ── Title + таймфрейм ───────────────────────
        ax_price.set_title(
            f"SHORT SIGNAL  ·  {signal.symbol}  ·  4H Chart  ·  Score {signal.score}/7  ·  Pump +{signal.pump_percent:.1f}%",
            color=WHITE, fontsize=12, fontweight="bold", pad=10
        )

        # Таймфрейм в углу
        ax_price.text(
            0.01, 0.97, "4H", transform=ax_price.transAxes,
            color="#888", fontsize=10, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", alpha=0.7)
        )

        # ── Volume bars ─────────────────────────────
        vol_avg = df["volume"].mean()
        for i in range(n):
            row   = df.iloc[i]
            color = GREEN if row["close"] >= row["open"] else RED
            alpha = 0.9 if row["volume"] > vol_avg * 2.5 else 0.5
            ax_vol.bar(i, row["volume"], color=color, alpha=alpha, width=0.76)

        ax_vol.axhline(vol_avg, color=GREY, linewidth=0.8, linestyle="--")
        ax_vol.set_ylabel("Vol", color="#888", fontsize=8)
        ax_vol.set_yticks([])

        # ── RSI line ────────────────────────────────
        rsi_values = []
        for i in range(n):
            window = df["close"].iloc[max(0, i - 14): i + 1]
            v = calc_rsi(window)
            rsi_values.append(v if not (np.isnan(v) or np.isinf(v)) else 50.0)

        ax_rsi.plot(x, rsi_values, color=PURPLE, linewidth=1.5, zorder=3)
        ax_rsi.axhline(75, color=RED,   linewidth=0.8, linestyle="--", alpha=0.6)
        ax_rsi.axhline(70, color=ORANGE, linewidth=0.6, linestyle=":",  alpha=0.5)
        ax_rsi.axhline(30, color=GREEN,  linewidth=0.6, linestyle=":",  alpha=0.5)
        ax_rsi.fill_between(x, rsi_values, 75,
                            where=[v > 75 for v in rsi_values],
                            alpha=0.2, color=RED, zorder=2)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", color="#888", fontsize=8)
        ax_rsi.text(n + 0.3, signal.rsi, f"{signal.rsi:.0f}",
                    color=PURPLE, fontsize=8, va="center", fontweight="bold")

        # ── X axis labels (shared) ──────────────────
        step      = max(1, n // 8)
        tick_pos  = list(range(0, n, step))
        tick_labs = [df.index[i].strftime("%m/%d\n%H:%M") for i in tick_pos]
        ax_rsi.set_xticks(tick_pos)
        ax_rsi.set_xticklabels(tick_labs, color="#888", fontsize=7)
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_vol.get_xticklabels(),   visible=False)

        # ── Render ──────────────────────────────────
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor=BG)
        plt.close()
        buf.seek(0)
        return buf.read()

    except Exception as e:
        logger.error(f"generate_chart({signal.symbol}): {e}", exc_info=True)
        return None


# ─────────────────────────────────────────────
# MESSAGE FORMATTER
# ─────────────────────────────────────────────

def format_signal_message(signal: Signal) -> str:
    res_4h = _price_fmt(signal.resistance_4h) if signal.resistance_4h else "—"
    res_1d = _price_fmt(signal.resistance_1d) if signal.resistance_1d else "—"

    return (
        f"⚡ <b>PUMP REVERSAL SIGNAL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 <b>Coin:</b> #{signal.symbol}\n"
        f"📍 <b>Entry:</b> <code>{_price_fmt(signal.entry)}</code>\n"
        f"🛑 <b>Stop Loss:</b> <code>{_price_fmt(signal.stop_loss)}</code>\n"
        f"✅ <b>TP1:</b> <code>{_price_fmt(signal.tp1)}</code>\n"
        f"🎯 <b>TP2:</b> <code>{_price_fmt(signal.tp2)}</code>\n"
        f"⚠️ <b>При TP1 → стоп в безубыток</b> <code>{_price_fmt(signal.entry)}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Pump 24H:</b> +{signal.pump_percent:.1f}%\n"
        f"📊 <b>RSI (4H):</b> {signal.rsi:.1f}\n"
        f"💰 <b>Funding:</b> {signal.funding_rate * 100:.4f}%\n"
        f"📦 <b>Open Interest:</b> ${signal.open_interest:,.0f}\n"
        f"🔀 <b>Volume:</b> {signal.volume_ratio:.1f}x avg\n"
        f"{'🔻' if signal.oi_divergence else '➖'} <b>OI Divergence:</b> {'YES' if signal.oi_divergence else 'NO'}\n"
        f"{'🔫' if signal.liquidity_sweep else '➖'} <b>Liq Sweep:</b> {'YES' if signal.liquidity_sweep else 'NO'}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏔 <b>Res 4H:</b> <code>{res_4h}</code>\n"
        f"🏔 <b>Res 1D:</b> <code>{res_1d}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⭐ <b>Score:</b> {signal.score}/7\n"
        f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )


# ─────────────────────────────────────────────
# CSV LOGGER
# ─────────────────────────────────────────────

CSV_HEADERS = [
    "date", "symbol", "entry", "stop_loss", "tp1", "tp2",
    "rsi", "funding", "open_interest", "pump_percent", "score"
]


def log_signal_to_csv(signal: Signal):
    file_exists = SIGNALS_CSV.exists()
    try:
        with open(SIGNALS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if not file_exists:
                w.writeheader()
            w.writerow({
                "date":          datetime.now(timezone.utc).isoformat(),
                "symbol":        signal.symbol,
                "entry":         round(signal.entry, 8),
                "stop_loss":     round(signal.stop_loss, 8),
                "tp1":           round(signal.tp1, 8),
                "tp2":           round(signal.tp2, 8),
                "rsi":           round(signal.rsi, 2),
                "funding":       round(signal.funding_rate, 6),
                "open_interest": round(signal.open_interest, 2),
                "pump_percent":  round(signal.pump_percent, 2),
                "score":         signal.score,
            })
    except Exception as e:
        logger.error(f"log_signal_to_csv: {e}")


# ─────────────────────────────────────────────
# COOLDOWN
# ─────────────────────────────────────────────

def is_on_cooldown(symbol: str) -> bool:
    return (time.time() - _signal_cache.get(symbol, 0)) < SIGNAL_COOLDOWN


def mark_signalled(symbol: str):
    _signal_cache[symbol] = time.time()
    _save_cooldown()


# ─────────────────────────────────────────────
# MARKET SCAN
# ─────────────────────────────────────────────

def scan_market():
    logger.info("━━━ Starting scan ━━━")
    t0 = time.time()

    tickers = get_24h_tickers(min_volume_usdt=5_000_000)
    if not tickers:
        logger.warning("No tickers received from Binance")
        return

    tickers.sort(key=lambda x: x.get("quoteVolume", 0), reverse=True)
    tickers = tickers[:250]
    logger.info(f"Scanning {len(tickers)} symbols")

    # Fast pre-filter: collect pumped coins
    pumped = []
    for t in tickers:
        pct = t.get("priceChangePercent")
        if pct is not None and pct >= Config.MIN_PUMP_PCT:
            pumped.append((t["symbol"], pct))

    pumped.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Pumped ≥{Config.MIN_PUMP_PCT}%: {len(pumped)}")

    sent = 0
    for symbol, pump_pct in pumped:
        if is_on_cooldown(symbol):
            continue
        try:
            signal = analyze_symbol(symbol, pump_pct)
            if signal is None:
                continue

            logger.info(f"🚨 {symbol} score={signal.score}/7")

            chart = generate_chart(signal)
            msg   = format_signal_message(signal)

            if chart:
                tg_send_photo(chart, caption=msg)
            else:
                tg_send_message(msg)

            log_signal_to_csv(signal)
            mark_signalled(symbol)

            # Открываем paper trade по этому сигналу
            paper_trade = open_paper_trade(signal)
            if paper_trade:
                logger.info(f"Paper trade opened: {symbol} x{paper_trade['leverage']}")

            sent += 1
            time.sleep(1)

        except Exception as e:
            logger.error(f"{symbol}: {e}", exc_info=True)
        finally:
            time.sleep(0.15)

    logger.info(f"━━━ Done: {sent} signals in {time.time()-t0:.1f}s ━━━")

    # Обновляем открытые paper позиции текущими ценами
    try:
        open_syms = get_open_trade_symbols()
        if open_syms:
            current_prices = get_latest_prices(open_syms)
            if current_prices:
                update_open_trades(current_prices)
    except Exception as e:
        logger.warning(f"Paper update error: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# TELEGRAM POLLING — listen for commands
# ─────────────────────────────────────────────

_last_update_id = 0


def tg_get_updates(offset: int = 0) -> list:
    """Long-poll Telegram for new messages."""
    try:
        url  = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates"
        resp = requests.get(url, params={"offset": offset, "timeout": 20}, timeout=25)
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e:
        logger.warning(f"tg_get_updates: {e}")
        return []


def tg_reply(chat_id: int, text: str):
    """Send reply to a specific chat (not just the main channel)."""
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception as e:
        logger.warning(f"tg_reply: {e}")


def tg_reply_photo(chat_id: int, image_bytes: bytes, caption: str = ""):
    """Send photo reply to a specific chat."""
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=15,
        )
    except Exception as e:
        logger.warning(f"tg_reply_photo: {e}")


# ─────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────

def _simulate_outcome(row: dict) -> tuple[str, bool]:
    """
    Fetch candles STARTING from signal date (historical).
    Checks which level was hit first: SL, TP2, TP1, or BE.
    After TP1 is hit — SL moves to entry (breakeven).
    Returns: (result, tp1_hit)
    result: 'TP2', 'SL', 'BE', or 'OPEN'
    """
    try:
        symbol   = row["symbol"]
        entry    = float(row["entry"])
        sl       = float(row["stop_loss"])
        tp1      = float(row["tp1"])
        tp2      = float(row["tp2"])
        sig_time = pd.Timestamp(row["date"])
        if sig_time.tzinfo is None:
            sig_time = sig_time.tz_localize("UTC")
        else:
            sig_time = sig_time.tz_convert("UTC")

        start_str = sig_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str   = (sig_time + pd.Timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")

        from data import get_historical_klines
        df = get_historical_klines(symbol, "4h", start_str=start_str, end_str=end_str)

        if df is None or df.empty:
            return "OPEN"

        future = df.iloc[1:61]
        if future.empty:
            return "OPEN"

        be_hit     = False
        tp1_hit    = False
        current_sl = sl  # SL moves to entry after TP1

        for _, candle in future.iterrows():
            high = candle["high"]
            low  = candle["low"]

            hit_sl  = high >= current_sl
            hit_tp2 = low <= tp2
            hit_tp1 = low <= tp1

            # Conservative priority: SL/BE first, then TP2
            if hit_sl:
                return ("BE" if be_hit else "SL", tp1_hit or be_hit or hit_tp1)

            if hit_tp2:
                return ("TP2", True)

            # TP1 hit — move SL to breakeven (no exit)
            if hit_tp1 and not be_hit:
                be_hit     = True
                tp1_hit    = True
                current_sl = entry  # breakeven

        return ("OPEN", tp1_hit)

    except Exception as e:
        logger.warning(f"_simulate_outcome: {e}")
        return ("OPEN", False)


def _winrate_bar(winrate: float, width: int = 16) -> str:
    """Generates a text progress bar for winrate."""
    filled = round(winrate / 100 * width)
    empty  = width - filled
    bar    = "█" * filled + "░" * empty
    return bar


def _best_worst(trades: list[dict]) -> tuple[str, str]:
    """Returns (best_trade_str, worst_trade_str) with pnl %."""
    winners = [t for t in trades if t["result"] == "TP2"]
    losers  = [t for t in trades if t["result"] == "SL"]

    best  = "—"
    worst = "—"

    if winners:
        b = max(winners, key=lambda t: t.get("pnl", 0))
        pnl = b.get("pnl", 0)
        best = f"{b['symbol']} +{pnl:.1f}% ({b['result']})"
    if losers:
        w = min(losers, key=lambda t: t.get("pnl", 0))
        pnl = w.get("pnl", 0)
        worst = f"{w['symbol']} {pnl:.1f}% (SL)"

    return best, worst


def handle_backtest(chat_id: int, args: list[str]):
    """
    /backtest — анализирует все сигналы из signals.csv
    Симулирует исходы (TP1/TP2/SL) и выдаёт итоговый отчёт.
    """
    tg_reply(chat_id, "⏳ <b>Считаю результаты...</b>\nАнализирую все сигналы из истории, подожди немного.")

    try:
        # ── Читаем signals.csv ─────────────────────
        if not SIGNALS_CSV.exists():
            tg_reply(chat_id, "⚠️ Файл signals.csv пустой — бот ещё не отправил ни одного сигнала.")
            return

        rows = []
        with open(SIGNALS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            tg_reply(chat_id, "⚠️ Сигналов в истории нет.")
            return

        total = len(rows)

        # ── Симулируем исходы ──────────────────────
        results = []
        for row in rows:
            outcome, tp1_hit = _simulate_outcome(row)
            results.append({**row, "result": outcome, "tp1_hit": tp1_hit})
            time.sleep(0.1)  # не спамим API

        # ── Считаем статистику ─────────────────────
        tp2_cnt  = sum(1 for r in results if r["result"] == "TP2")
        tp1_cnt  = sum(1 for r in results if r.get("tp1_hit"))
        sl_cnt   = sum(1 for r in results if r["result"] == "SL")
        be_cnt   = sum(1 for r in results if r["result"] == "BE")
        open_cnt = sum(1 for r in results if r["result"] == "OPEN")

        wins   = tp2_cnt
        closed = wins + sl_cnt + be_cnt

        winrate = (wins / closed * 100) if closed > 0 else 0.0
        bar     = _winrate_bar(winrate)

        # P&L с учётом комиссии 0.08% на сделку
        COMMISSION = 0.08
        pnl_list = []
        for r in results:
            try:
                entry = float(r["entry"])
                sl    = float(r["stop_loss"])
                tp1   = float(r["tp1"])
                tp2   = float(r["tp2"])
                risk  = abs(sl - entry)
                if r["result"] == "TP2":
                    pnl = (entry - tp2) / entry * 100 - COMMISSION
                elif r["result"] == "TP1":
                    pnl = (entry - tp1) / entry * 100 - COMMISSION
                elif r["result"] == "SL":
                    pnl = -risk / entry * 100 - COMMISSION
                elif r["result"] == "BE":
                    pnl = -COMMISSION  # вышли в ноль, только комиссия
                else:
                    pnl = 0.0
                r["pnl"] = pnl
                pnl_list.append(pnl)
            except Exception:
                r["pnl"] = 0.0

        total_pnl = sum(pnl_list) if pnl_list else 0.0
        avg_win   = sum(p for p in pnl_list if p > 0) / max(1, wins)
        avg_loss  = sum(p for p in pnl_list if p < 0) / max(1, sl_cnt)
        gross_win = sum(p for p in pnl_list if p > 0)
        gross_los = abs(sum(p for p in pnl_list if p < 0))
        pf        = gross_win / gross_los if gross_los > 0 else float("inf")
        rr        = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        best, worst = _best_worst(results)

        now_str = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")

        # ── Вердикт ────────────────────────────────
        if winrate >= 55 and pf >= 1.5:
            verdict = "🟢 Стратегия прибыльная"
        elif winrate >= 45 and pf >= 1.0:
            verdict = "🟡 Стратегия в плюсе"
        else:
            verdict = "🔴 Нужна донастройка"

        # ── Форматируем отчёт ──────────────────────
        msg = (
            f"📊 <b>БЭКТЕСТ РЕЗУЛЬТАТЫ</b>\n"
            f"🕐 {now_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"🏆 <b>WIN RATE</b>\n"
            f"<code>{bar}</code>  <b>{winrate:.1f}%</b>\n"
            f"\n"
            f"🏅 TP2 закрыто:  <b>{tp2_cnt}</b>\n"
            f"✅ TP1 достигнуто: <b>{tp1_cnt}</b>\n"
            f"❌ SL сработало: <b>{sl_cnt}</b>\n"
            f"🔄 Безубыток:    <b>{be_cnt}</b>\n"
            f"⏱ Тайм-аут:     <b>{open_cnt}</b>\n"
            f"📊 Всего:        <b>{total}</b>\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>P&amp;L</b>\n"
            f"📉 Итого P&amp;L:      <b>{total_pnl:+.2f}%</b>\n"
            f"📈 Средний выигрыш: <b>+{avg_win:.2f}%</b>\n"
            f"📉 Средний стоп:    <b>{avg_loss:.2f}%</b>\n"
            f"⚖️ Risk/Reward:     <b>{rr:.2f}</b>\n"
            f"📊 Profit Factor:   <b>{pf:.2f}</b>\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⭐️ <b>РЕКОРДЫ</b>\n"
            f"🥇 Лучший:  {best}\n"
            f"💀 Худший:  {worst}\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{verdict}"
        )

        tg_reply(chat_id, msg)

    except Exception as e:
        logger.error(f"handle_backtest: {e}", exc_info=True)
        tg_reply(chat_id, f"❌ Ошибка: {e}")


def handle_paper(chat_id: int, args: list[str]):
    """
    /paper — показывает статистику paper trading
    /paper reset — сбросить счёт
    """
    if args and args[0].lower() == "reset":
        reset_paper_account()
        tg_reply(chat_id,
            "🔄 <b>Paper Trading сброшен</b>\n"
            f"Новый баланс: ${200:.2f}\n"
            "Все сделки удалены."
        )
        return

    report = get_paper_report()
    tg_reply(chat_id, report)


def handle_status(chat_id: int):
    """/status — показывает текущие настройки бота и статистику сигналов."""
    # Count signals from CSV
    total_signals = 0
    if SIGNALS_CSV.exists():
        try:
            with open(SIGNALS_CSV) as f:
                total_signals = sum(1 for line in f) - 1  # minus header
        except Exception:
            pass

    active_cooldowns = sum(
        1 for sym, t in _signal_cache.items()
        if time.time() - t < SIGNAL_COOLDOWN
    )

    tg_reply(chat_id,
        f"🤖 <b>Bot Status</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ Scan interval:  {SCAN_INTERVAL}s\n"
        f"📈 Min pump:       {Config.MIN_PUMP_PCT}%\n"
        f"⭐ Min score:      {Config.MIN_SCORE}/7\n"
        f"📊 RSI threshold:  {Config.RSI_OVERBOUGHT}\n"
        f"💰 Min funding:    {Config.MIN_FUNDING_RATE*100:.3f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📬 Total signals:  {total_signals}\n"
        f"⏳ On cooldown:    {active_cooldowns} symbols\n"
        f"🕐 Time (UTC):     {datetime.utcnow().strftime('%H:%M:%S')}"
    )


def handle_help(chat_id: int):
    """/help — список команд."""
    tg_reply(chat_id,
        "🤖 <b>Pump Reversal Bot — Команды</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "/backtest — бэктест по всем сигналам\n\n"
        "/paper — paper trading статистика\n"
        "/paper reset — сбросить paper счёт\n\n"
        "/status — настройки и статистика бота\n\n"
        "/help — эта справка\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Сигналы приходят автоматически каждые 5 минут."
    )


def process_update(update: dict):
    """Parse and route incoming Telegram update."""
    try:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return

        chat_id = msg["chat"]["id"]
        text    = msg.get("text", "").strip()

        if not text.startswith("/"):
            return

        parts   = text.split()
        command = parts[0].lower().split("@")[0]  # handle /cmd@botname
        args    = parts[1:]

        logger.info(f"Command from {chat_id}: {command} {args}")

        if command == "/backtest":
            threading.Thread(
                target=handle_backtest,
                args=(chat_id, args),
                daemon=True
            ).start()
        elif command == "/status":
            handle_status(chat_id)
        elif command == "/paper":
            threading.Thread(
                target=handle_paper,
                args=(chat_id, args),
                daemon=True
            ).start()
        elif command == "/help" or command == "/start":
            handle_help(chat_id)
        else:
            tg_reply(chat_id, "Неизвестная команда. Напиши /help")

    except Exception as e:
        logger.error(f"process_update: {e}", exc_info=True)


# ─────────────────────────────────────────────
# POLLING LOOP (runs in background thread)
# ─────────────────────────────────────────────

def polling_loop():
    """Continuously poll Telegram for new commands."""
    global _last_update_id
    logger.info("Polling loop started")

    while True:
        try:
            updates = tg_get_updates(offset=_last_update_id + 1)
            for upd in updates:
                _last_update_id = upd["update_id"]
                process_update(upd)
        except Exception as e:
            logger.warning(f"polling_loop: {e}")
            time.sleep(5)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not TG_TOKEN:
        logger.error("TG_TOKEN not set")
        return
    if not TG_CHAT:
        logger.error("TG_CHAT not set")
        return

    logger.info(f"Starting — interval={SCAN_INTERVAL}s, pump={Config.MIN_PUMP_PCT}%, score={Config.MIN_SCORE}/7")

    # Restore cooldown state from previous session
    _load_cooldown()

    # Start command polling in background thread
    t = threading.Thread(target=polling_loop, daemon=True)
    t.start()

    tg_send_message(
        f"🤖 <b>Pump Reversal Bot — STARTED</b>\n"
        f"Scan interval: {SCAN_INTERVAL}s\n"
        f"Min pump: {Config.MIN_PUMP_PCT}%\n"
        f"Min score: {Config.MIN_SCORE}/7\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Команды:\n"
        f"/backtest PEPEUSDT — бэктест\n"
        f"/status — статус бота\n"
        f"/help — справка\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Жду сигналов..."
    )

    # Main scanner loop
    while True:
        try:
            scan_market()
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error(f"Main loop: {e}", exc_info=True)
            time.sleep(30)

        logger.info(f"Next scan in {SCAN_INTERVAL}s")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
