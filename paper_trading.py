"""
paper_trading.py — Paper Trading Simulator

Симулирует торговлю по сигналам бота с виртуальным депозитом.
- Стартовый депозит: $200
- Плечо: случайное x3, x5, x7, x10
- Риск на сделку: 2% от текущего депозита
- Учитывает ликвидацию
- Сохраняет все сделки в paper_trades.csv
"""

import csv
import json
import logging
import math
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PAPER_FILE      = Path("paper_account.json")
PAPER_TRADES    = Path("paper_trades.csv")
START_BALANCE   = 200.0      # стартовый депозит $
RISK_PCT        = 0.02       # 2% риска на сделку
COMMISSION_PCT  = 0.0008     # 0.08% комиссия (taker × 2)
MAX_OPEN_TRADES = 3          # максимум одновременных позиций

# Плечо — веса: x10 редко, x3 часто
LEVERAGE_OPTIONS = [3, 3, 5, 5, 7, 10]


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class PaperTrade:
    id: int
    symbol: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    leverage: int
    position_usdt: float   # размер позиции в USDT
    risk_usdt: float       # максимальный риск в $
    liq_price: float       # цена ликвидации
    opened_at: str
    status: str            # 'OPEN', 'TP1', 'TP2', 'SL', 'LIQ', 'BE'
    closed_at: str = ""
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    exit_price: float = 0.0


@dataclass
class PaperAccount:
    balance: float = START_BALANCE
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    be_count: int = 0
    liq_count: int = 0
    total_pnl: float = 0.0
    peak_balance: float = START_BALANCE
    open_trades: list = None

    def __post_init__(self):
        if self.open_trades is None:
            self.open_trades = []


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def load_account() -> PaperAccount:
    """Load account state from disk."""
    try:
        if PAPER_FILE.exists():
            data = json.loads(PAPER_FILE.read_text())
            acc = PaperAccount(
                balance       = data.get("balance", START_BALANCE),
                total_trades  = data.get("total_trades", 0),
                wins          = data.get("wins", 0),
                losses        = data.get("losses", 0),
                be_count      = data.get("be_count", 0),
                liq_count     = data.get("liq_count", 0),
                total_pnl     = data.get("total_pnl", 0.0),
                peak_balance  = data.get("peak_balance", START_BALANCE),
                open_trades   = data.get("open_trades", []),
            )
            return acc
    except Exception as e:
        logger.warning(f"load_account: {e}")
    return PaperAccount()


def save_account(acc: PaperAccount):
    """Save account state to disk."""
    try:
        data = {
            "balance":      acc.balance,
            "total_trades": acc.total_trades,
            "wins":         acc.wins,
            "losses":       acc.losses,
            "be_count":     acc.be_count,
            "liq_count":    acc.liq_count,
            "total_pnl":    acc.total_pnl,
            "peak_balance": acc.peak_balance,
            "open_trades":  acc.open_trades,
        }
        PAPER_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"save_account: {e}")


def get_open_trade_symbols() -> list[str]:
    """Returns list of symbols with open paper trades."""
    acc = load_account()
    return [t.get("symbol") for t in acc.open_trades if t.get("symbol")]


def log_trade_to_csv(trade: PaperTrade):
    """Append closed trade to CSV."""
    file_exists = PAPER_TRADES.exists()
    try:
        with open(PAPER_TRADES, "a", newline="") as f:
            fields = ["id", "symbol", "entry", "stop_loss", "tp1", "tp2",
                      "leverage", "position_usdt", "risk_usdt", "liq_price",
                      "opened_at", "closed_at", "status", "pnl_usdt", "pnl_pct", "exit_price"]
            w = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                w.writeheader()
            w.writerow({k: getattr(trade, k) for k in fields})
    except Exception as e:
        logger.warning(f"log_trade_to_csv: {e}")


# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────

def calc_liq_price(entry: float, leverage: int) -> float:
    """
    Liquidation price for SHORT position.
    Simplified: entry × (1 + 1/leverage × 0.9)
    (0.9 — maintenance margin buffer)
    """
    return entry * (1 + (1 / leverage) * 0.9)


def open_paper_trade(signal) -> Optional[dict]:
    """
    Open a new paper trade from a signal.
    Returns trade dict or None if skipped.
    """
    acc = load_account()

    # Не открываем больше MAX_OPEN_TRADES позиций
    if len(acc.open_trades) >= MAX_OPEN_TRADES:
        logger.info(f"Paper: max open trades ({MAX_OPEN_TRADES}) reached, skip {signal.symbol}")
        return None

    # Не дублируем символ
    open_symbols = [t["symbol"] for t in acc.open_trades]
    if signal.symbol in open_symbols:
        logger.info(f"Paper: {signal.symbol} already open, skip")
        return None

    # Выбираем плечо случайно
    leverage = random.choice(LEVERAGE_OPTIONS)

    # Риск в $ = 2% от баланса
    risk_usdt = acc.balance * RISK_PCT

    # Размер позиции = риск / (|entry - sl| / entry) × leverage
    sl_dist_pct = abs(signal.entry - signal.stop_loss) / signal.entry
    if sl_dist_pct <= 0:
        return None

    position_usdt = (risk_usdt / sl_dist_pct) * leverage
    position_usdt = min(position_usdt, acc.balance * leverage)  # не больше баланс × плечо

    # Цена ликвидации
    liq_price = calc_liq_price(signal.entry, leverage)

    trade = {
        "id":            acc.total_trades + 1,
        "symbol":        signal.symbol,
        "entry":         signal.entry,
        "stop_loss":     signal.stop_loss,
        "tp1":           signal.tp1,
        "tp2":           signal.tp2,
        "leverage":      leverage,
        "position_usdt": round(position_usdt, 2),
        "risk_usdt":     round(risk_usdt, 2),
        "liq_price":     round(liq_price, 6),
        "opened_at":     datetime.utcnow().isoformat(),
        "status":        "OPEN",
        "be_active":     False,  # становится True после TP1
    }

    acc.open_trades.append(trade)
    acc.total_trades += 1
    save_account(acc)

    logger.info(f"Paper OPEN: {signal.symbol} x{leverage} pos=${position_usdt:.0f} risk=${risk_usdt:.2f} liq={liq_price:.6f}")
    return trade


def update_open_trades(current_prices: dict[str, float]):
    """
    Check all open trades against current prices.
    Called during each market scan.
    current_prices: {symbol: price}
    """
    acc = load_account()
    if not acc.open_trades:
        return

    still_open = []
    for t in acc.open_trades:
        sym   = t["symbol"]
        price = current_prices.get(sym)
        if price is None:
            still_open.append(t)
            continue

        entry    = t["entry"]
        sl       = t["stop_loss"]
        tp1      = t["tp1"]
        tp2      = t["tp2"]
        liq      = t["liq_price"]
        be_active = t.get("be_active", False)
        current_sl = entry if be_active else sl  # после TP1 — BE

        closed   = False
        status   = "OPEN"
        exit_p   = price
        pnl_usdt = 0.0

        # SHORT: прибыль когда цена падает
        # Ликвидация — приоритет
        if price >= liq:
            status   = "LIQ"
            exit_p   = liq
            pnl_usdt = -t["risk_usdt"] * t["leverage"] * 0.9  # почти весь маржин
            closed   = True

        # TP2
        elif price <= tp2:
            status   = "TP2"
            exit_p   = tp2
            pnl_pct  = (entry - tp2) / entry
            pnl_usdt = t["position_usdt"] * pnl_pct - t["position_usdt"] * COMMISSION_PCT
            closed   = True

        # TP1 — активируем BE
        elif price <= tp1 and not be_active:
            t["be_active"] = True
            t["stop_loss"] = entry  # визуально обновляем
            logger.info(f"Paper {sym}: TP1 hit, SL moved to BE ({entry:.6f})")

        # SL или BE
        elif price >= current_sl and current_sl > 0:
            status   = "BE" if be_active else "SL"
            exit_p   = current_sl
            pnl_pct  = (entry - current_sl) / entry
            pnl_usdt = t["position_usdt"] * pnl_pct - t["position_usdt"] * COMMISSION_PCT
            closed   = True

        if closed:
            pnl_pct_final = (entry - exit_p) / entry * 100

            # Обновляем баланс
            acc.balance += pnl_usdt
            acc.balance  = max(acc.balance, 0)
            acc.total_pnl += pnl_usdt
            acc.peak_balance = max(acc.peak_balance, acc.balance)

            if status in ("TP1", "TP2"):
                acc.wins += 1
            elif status == "BE":
                acc.be_count += 1
            elif status == "LIQ":
                acc.liq_count += 1
            else:
                acc.losses += 1

            trade_obj = PaperTrade(
                id=t["id"], symbol=sym, entry=entry,
                stop_loss=t["stop_loss"], tp1=tp1, tp2=tp2,
                leverage=t["leverage"], position_usdt=t["position_usdt"],
                risk_usdt=t["risk_usdt"], liq_price=liq,
                opened_at=t["opened_at"],
                status=status,
                closed_at=datetime.utcnow().isoformat(),
                pnl_usdt=round(pnl_usdt, 4),
                pnl_pct=round(pnl_pct_final, 2),
                exit_price=exit_p,
            )
            log_trade_to_csv(trade_obj)
            logger.info(f"Paper CLOSE: {sym} {status} pnl=${pnl_usdt:.2f} balance=${acc.balance:.2f}")
        else:
            still_open.append(t)

    acc.open_trades = still_open
    save_account(acc)


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def get_paper_report() -> str:
    """Returns formatted paper trading report for Telegram."""
    acc = load_account()

    balance    = acc.balance
    start      = START_BALANCE
    total_pnl  = balance - start
    pnl_pct    = (balance / start - 1) * 100
    total      = acc.total_trades
    wins       = acc.wins
    losses     = acc.losses
    be_cnt     = acc.be_count
    liq_cnt    = acc.liq_count
    closed     = wins + losses + be_cnt + liq_cnt
    winrate    = (wins / closed * 100) if closed > 0 else 0.0
    drawdown   = (acc.peak_balance - balance) / acc.peak_balance * 100 if acc.peak_balance > 0 else 0.0
    open_cnt   = len(acc.open_trades)

    # Balance bar
    bar_filled = round(min(balance / start, 2.0) / 2.0 * 16)
    bar = "█" * bar_filled + "░" * (16 - bar_filled)

    pnl_icon = "📈" if total_pnl >= 0 else "📉"
    verdict = "🟢 В плюсе" if balance > start else "🔴 В минусе"

    open_lines = ""
    if acc.open_trades:
        open_lines = "\n━━━━━━━━━━━━━━━━━━━━\n📋 <b>Открытые позиции:</b>\n"
        for t in acc.open_trades:
            be_mark = " 🔄BE" if t.get("be_active") else ""
            open_lines += f"  • {t['symbol']} x{t['leverage']}{be_mark} | риск ${t['risk_usdt']:.2f}\n"

    return (
        f"📋 <b>PAPER TRADING</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 <b>Баланс:</b> ${balance:.2f}\n"
        f"<code>{bar}</code>  {pnl_pct:+.1f}%\n"
        f"{pnl_icon} <b>PnL:</b> ${total_pnl:+.2f}\n"
        f"📉 <b>Просадка:</b> {drawdown:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Всего сделок:</b> {total}\n"
        f"✅ <b>Профит:</b> {wins}\n"
        f"❌ <b>Стоп:</b> {losses}\n"
        f"🔄 <b>Безубыток:</b> {be_cnt}\n"
        f"💀 <b>Ликвидация:</b> {liq_cnt}\n"
        f"⏳ <b>Открытых:</b> {open_cnt}\n"
        f"🏆 <b>Win rate:</b> {winrate:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{verdict}"
        f"{open_lines}"
    )


def reset_paper_account():
    """Reset paper account to initial state."""
    acc = PaperAccount()
    save_account(acc)
    if PAPER_TRADES.exists():
        PAPER_TRADES.unlink()
    logger.info("Paper account reset")
