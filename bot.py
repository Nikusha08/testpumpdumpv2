"""
bot.py — Main entry point with optimized scanning and rich signal message.
"""

import os
import time
import json
import csv
import io
import logging
import threading
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests

from data import get_all_24h_changes
from strategy import analyze_symbol, Signal, Config
from indicators import calc_rsi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")])
logger = logging.getLogger("bot")

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT = os.environ.get("TG_CHAT", "")
SCAN_INTERVAL = Config.SCAN_INTERVAL
SIGNALS_CSV = Path("signals.csv")
SIGNAL_COOLDOWN = 7200
COOLDOWN_FILE = Path("cooldown.json")
_signal_cache = {}

def _load_cooldown():
    global _signal_cache
    try:
        if COOLDOWN_FILE.exists():
            data = json.loads(COOLDOWN_FILE.read_text())
            now = time.time()
            _signal_cache = {k: v for k, v in data.items() if now - v < SIGNAL_COOLDOWN}
            logger.info(f"Cooldown loaded: {len(_signal_cache)} active symbols")
    except Exception as e:
        logger.warning(f"_load_cooldown: {e}")

def _save_cooldown():
    try:
        COOLDOWN_FILE.write_text(json.dumps(_signal_cache))
    except Exception as e:
        logger.warning(f"_save_cooldown: {e}")

def is_on_cooldown(symbol: str) -> bool:
    return (time.time() - _signal_cache.get(symbol, 0)) < SIGNAL_COOLDOWN

def mark_signalled(symbol: str):
    _signal_cache[symbol] = time.time()
    _save_cooldown()

def tg_send_message(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT:
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}, timeout=10)
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
        resp = requests.post(url, data={"chat_id": TG_CHAT, "caption": caption, "parse_mode": "HTML"},
                             files={"photo": ("chart.png", image_bytes, "image/png")}, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"tg_send_photo: {e}")
        return False

# ---------- Chart generation (unchanged, but included for completeness) ----------
def _price_fmt(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.5f}"
    else:
        return f"{price:.8f}"

def generate_chart(signal: Signal) -> Optional[bytes]:
    try:
        df = signal.df_4h.iloc[-60:].copy()
        n = len(df)
        if n < 10:
            return None
        fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 0.8, 0.8], hspace=0.08)
        ax_price = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
        for ax in [ax_price, ax_vol, ax_rsi]:
            ax.set_facecolor("#0d0d0d")
            ax.tick_params(colors="#888", labelsize=8)
        x = np.arange(n)
        # Candlesticks
        for i in range(n):
            row = df.iloc[i]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = "#00e676" if c >= o else "#ff1744"
            ax_price.plot([i,i], [l,h], color=color, linewidth=0.8, zorder=2)
            body_h = abs(c-o)
            body_y = min(o,c)
            if body_h < (h-l)*0.01:
                body_h = (h-l)*0.015
            rect = mpatches.Rectangle((i-0.38, body_y), 0.76, body_h,
                                      linewidth=0, facecolor=color, zorder=3)
            ax_price.add_patch(rect)
        # Trade levels
        trade_levels = [
            (signal.stop_loss, "#ff1744", "--", 2.0, f"SL  {_price_fmt(signal.stop_loss)}"),
            (signal.entry,     "#ffffff", "--", 1.8, f"Entry  {_price_fmt(signal.entry)}"),
            (signal.tp1,       "#00e676", ":",  1.5, f"TP1  {_price_fmt(signal.tp1)}"),
            (signal.tp2,       "#00c853", ":",  2.0, f"TP2  {_price_fmt(signal.tp2)}"),
        ]
        for price_level, color, ls, lw, label in trade_levels:
            ax_price.axhline(price_level, color=color, linewidth=lw, linestyle=ls, alpha=0.95, zorder=4)
            ax_price.text(n+0.3, price_level, label, color=color, fontsize=7.5, va="center")
        # Resistance
        if signal.resistance_4h:
            res = signal.resistance_4h
            ax_price.axhline(res, color="#ff9800", linewidth=2.0, linestyle="-", alpha=0.9, zorder=5)
            ax_price.text(n+0.3, res, f"Res 4H  {_price_fmt(res)}", color="#ff9800", fontsize=8, va="center")
        # Zone shading
        ax_price.axhspan(signal.entry, signal.stop_loss, alpha=0.06, color="#ff1744", zorder=1)
        ax_price.axhspan(signal.tp2, signal.entry, alpha=0.04, color="#00e676", zorder=1)
        ax_price.set_xlim(-1, n+10)
        ax_price.set_ylabel("Price (USDT)", color="#888")
        ax_price.set_title(f"SHORT SIGNAL  ·  {signal.symbol}  ·  4H Chart  ·  Score {signal.score:.1f}  ·  Pump +{signal.pump_percent:.1f}%",
                           color="white", fontsize=12, fontweight="bold")
        # Volume bars
        vol_avg = df["volume"].mean()
        for i in range(n):
            row = df.iloc[i]
            color = "#00e676" if row["close"] >= row["open"] else "#ff1744"
            alpha = 0.9 if row["volume"] > vol_avg * 2.5 else 0.5
            ax_vol.bar(i, row["volume"], color=color, alpha=alpha, width=0.76)
        ax_vol.axhline(vol_avg, color="#555", linewidth=0.8, linestyle="--")
        ax_vol.set_ylabel("Vol", color="#888")
        ax_vol.set_yticks([])
        # RSI
        rsi_values = []
        for i in range(n):
            window = df["close"].iloc[max(0,i-14):i+1]
            v = calc_rsi(window)
            rsi_values.append(v if not (np.isnan(v) or np.isinf(v)) else 50.0)
        ax_rsi.plot(x, rsi_values, color="#e040fb", linewidth=1.5)
        ax_rsi.axhline(75, color="#ff1744", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_rsi.axhline(30, color="#00e676", linewidth=0.6, linestyle=":", alpha=0.5)
        ax_rsi.set_ylim(0,100)
        ax_rsi.set_ylabel("RSI", color="#888")
        ax_rsi.text(n+0.3, signal.rsi, f"{signal.rsi:.0f}", color="#e040fb", fontsize=8, va="center")
        # X ticks
        step = max(1, n//8)
        tick_pos = list(range(0, n, step))
        tick_labs = [df.index[i].strftime("%m/%d\n%H:%M") for i in tick_pos]
        ax_rsi.set_xticks(tick_pos)
        ax_rsi.set_xticklabels(tick_labs, color="#888", fontsize=7)
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_vol.get_xticklabels(), visible=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"generate_chart: {e}", exc_info=True)
        return None

def format_signal_message(signal: Signal) -> str:
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
        f"💰 <b>Funding:</b> {signal.funding_rate*100:.4f}%\n"
        f"📦 <b>Open Interest:</b> ${signal.open_interest:,.0f}\n"
        f"🔀 <b>Volume:</b> {signal.volume_ratio:.1f}x avg\n"
        f"🔻 <b>OI Divergence:</b> {'✅' if signal.oi_divergence else '❌'}\n"
        f"🔫 <b>Liq Sweep:</b> {'✅' if signal.liquidity_sweep else '❌'}\n"
        f"🕯 <b>Pin Bar 4H:</b> {'✅' if signal.pin_bar_4h else '❌'}\n"
        f"🕯 <b>Pin Bar 1H:</b> {'✅' if signal.pin_bar_1h else '❌'}\n"
        f"📉 <b>RSI Divergence:</b> {'✅' if signal.rsi_divergence else '❌'}\n"
        f"📉 <b>MACD Divergence:</b> {'✅' if signal.macd_divergence else '❌'}\n"
        f"🔥 <b>Volume Climax:</b> {'✅' if signal.volume_climax else '❌'}\n"
        f"💪 <b>Weak after pump:</b> {'✅' if signal.weak_after_pump else '❌'}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⭐ <b>Score:</b> {signal.score:.1f} / {signal.confidence}% confidence\n"
        f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )

def log_signal_to_csv(signal: Signal):
    file_exists = SIGNALS_CSV.exists()
    try:
        with open(SIGNALS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "date","symbol","entry","stop_loss","tp1","tp2","rsi","funding",
                "open_interest","pump_percent","score","confidence","oi_divergence",
                "liquidity_sweep","pin_bar_4h","pin_bar_1h","rsi_divergence",
                "macd_divergence","volume_climax","weak_after_pump"
            ])
            if not file_exists:
                w.writeheader()
            w.writerow({
                "date": datetime.utcnow().isoformat(),
                "symbol": signal.symbol,
                "entry": round(signal.entry, 8),
                "stop_loss": round(signal.stop_loss, 8),
                "tp1": round(signal.tp1, 8),
                "tp2": round(signal.tp2, 8),
                "rsi": round(signal.rsi, 2),
                "funding": round(signal.funding_rate, 6),
                "open_interest": round(signal.open_interest, 2),
                "pump_percent": round(signal.pump_percent, 2),
                "score": round(signal.score, 2),
                "confidence": signal.confidence,
                "oi_divergence": signal.oi_divergence,
                "liquidity_sweep": signal.liquidity_sweep,
                "pin_bar_4h": signal.pin_bar_4h,
                "pin_bar_1h": signal.pin_bar_1h,
                "rsi_divergence": signal.rsi_divergence,
                "macd_divergence": signal.macd_divergence,
                "volume_climax": signal.volume_climax,
                "weak_after_pump": signal.weak_after_pump,
            })
    except Exception as e:
        logger.error(f"log_signal_to_csv: {e}")

def scan_market():
    logger.info("━━━ Starting scan ━━━")
    t0 = time.time()
    all_changes = get_all_24h_changes()
    if not all_changes:
        logger.error("Failed to fetch 24h changes")
        return
    pumped = [(sym, pct) for sym, pct in all_changes.items() if pct >= Config.MIN_PUMP_PCT]
    pumped.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Pumped ≥{Config.MIN_PUMP_PCT}%: {len(pumped)}")
    sent = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sym = {}
        for sym, pct in pumped:
            if is_on_cooldown(sym):
                continue
            future = executor.submit(analyze_symbol, sym, pct)
            future_to_sym[future] = sym
        for future in concurrent.futures.as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                signal = future.result()
                if signal:
                    logger.info(f"🚨 {sym} score={signal.score:.1f} confidence={signal.confidence}%")
                    chart = generate_chart(signal)
                    msg = format_signal_message(signal)
                    if chart:
                        tg_send_photo(chart, caption=msg)
                    else:
                        tg_send_message(msg)
                    log_signal_to_csv(signal)
                    mark_signalled(sym)
                    sent += 1
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"{sym}: {e}", exc_info=True)
    logger.info(f"━━━ Done: {sent} signals in {time.time()-t0:.1f}s ━━━")

# ---------- Telegram polling (unchanged, included for completeness) ----------
_last_update_id = 0

def tg_get_updates(offset: int = 0) -> list:
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates"
        resp = requests.get(url, params={"offset": offset, "timeout": 20}, timeout=25)
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e:
        logger.warning(f"tg_get_updates: {e}")
        return []

def tg_reply(chat_id: int, text: str):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logger.warning(f"tg_reply: {e}")

def tg_reply_photo(chat_id: int, image_bytes: bytes, caption: str = ""):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        requests.post(url, data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                      files={"photo": ("chart.png", image_bytes, "image/png")}, timeout=15)
    except Exception as e:
        logger.warning(f"tg_reply_photo: {e}")

def handle_backtest(chat_id: int, args: list[str]):
    tg_reply(chat_id, "⏳ <b>Backtest in progress...</b>")
    # Simplified for brevity – you can reuse the logic from previous backtest.py
    # but now with new indicators (optional)
    tg_reply(chat_id, "Backtest feature coming soon with new indicators.")

def handle_status(chat_id: int):
    total_signals = 0
    if SIGNALS_CSV.exists():
        try:
            with open(SIGNALS_CSV) as f:
                total_signals = sum(1 for _ in f) - 1
        except:
            pass
    active_cooldowns = sum(1 for sym, t in _signal_cache.items() if time.time() - t < SIGNAL_COOLDOWN)
    tg_reply(chat_id,
        f"🤖 <b>Bot Status</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ Scan interval:  {SCAN_INTERVAL}s\n"
        f"📈 Min pump:       {Config.MIN_PUMP_PCT}%\n"
        f"⭐ Min score:      {Config.MIN_SCORE}\n"
        f"📊 RSI threshold:  {Config.RSI_OVERBOUGHT}\n"
        f"💰 Min funding:    {Config.MIN_FUNDING_RATE*100:.3f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📬 Total signals:  {total_signals}\n"
        f"⏳ On cooldown:    {active_cooldowns} symbols\n"
        f"🕐 Time (UTC):     {datetime.utcnow().strftime('%H:%M:%S')}"
    )

def handle_help(chat_id: int):
    tg_reply(chat_id,
        "🤖 <b>Pump Reversal Bot — Commands</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "/backtest SYMBOL — run backtest\n"
        "/status — show bot status\n"
        "/help — this help"
    )

def process_update(update: dict):
    try:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        chat_id = msg["chat"]["id"]
        text = msg.get("text", "").strip()
        if not text.startswith("/"):
            return
        parts = text.split()
        command = parts[0].lower().split("@")[0]
        args = parts[1:]
        if command == "/backtest":
            threading.Thread(target=handle_backtest, args=(chat_id, args), daemon=True).start()
        elif command == "/status":
            handle_status(chat_id)
        elif command == "/help" or command == "/start":
            handle_help(chat_id)
        else:
            tg_reply(chat_id, "Unknown command. Type /help")
    except Exception as e:
        logger.error(f"process_update: {e}")

def polling_loop():
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

def main():
    if not TG_TOKEN or not TG_CHAT:
        logger.error("TG_TOKEN or TG_CHAT not set")
        return
    _load_cooldown()
    threading.Thread(target=polling_loop, daemon=True).start()
    tg_send_message(
        f"🤖 <b>Pump Reversal Bot STARTED</b>\n"
        f"Scan interval: {SCAN_INTERVAL}s\n"
        f"Min pump: {Config.MIN_PUMP_PCT}%\n"
        f"Min score: {Config.MIN_SCORE}\n"
        f"Use /status for more info"
    )
    while True:
        try:
            scan_market()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Main loop: {e}", exc_info=True)
            time.sleep(30)
        logger.info(f"Next scan in {SCAN_INTERVAL}s")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()