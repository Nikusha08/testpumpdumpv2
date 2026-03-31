# Pump Reversal Bot 🚀

SHORT reversal strategy bot for crypto futures trading.
Detects pump exhaustion and sends signals to Telegram.

## Strategy Logic

Signal requires **minimum 3 of 7 filters**:

| # | Filter | Threshold |
|---|--------|-----------|
| 1 | 24H Pump | ≥ 25% |
| 2 | Volume Spike | ≥ 3× average |
| 3 | RSI Overbought | ≥ 75 (4H) |
| 4 | Near Resistance | ≤ 3% from level (4H or 1D) |
| 5 | High Funding Rate | ≥ 0.05% |
| 6 | OI Divergence | Price ↑ while OI ↓ |
| 7 | Liquidity Sweep | Wick above recent highs |

## Project Structure

```
bot.py          — Main loop, Telegram sender, chart generator
data.py         — Binance API (klines, funding, open interest)
strategy.py     — Signal logic, score system, trade levels
indicators.py   — RSI, ATR, resistance, volume, sweep detection
backtest.py     — Historical backtesting engine
requirements.txt
signals.csv     — Auto-generated signal log
```

## Setup

### 1. Clone & Install

```bash
git clone <your-repo>
cd pump_reversal_bot
pip install -r requirements.txt
```

### 2. Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TG_TOKEN` | Telegram bot token (from @BotFather) | ✅ |
| `TG_CHAT` | Telegram chat/channel ID | ✅ |
| `SCAN_INTERVAL_SEC` | Scan frequency in seconds (default: 300) | ➖ |
| `MIN_PUMP_PCT` | Override minimum pump % (default: 25) | ➖ |
| `MIN_SCORE` | Override minimum signal score (default: 3) | ➖ |

### 3. Run Locally

```bash
export TG_TOKEN="your_token_here"
export TG_CHAT="your_chat_id"
python bot.py
```

## Deploy on Railway

1. Push code to GitHub
2. Create new project on [Railway](https://railway.app)
3. Connect your GitHub repo
4. Add environment variables:
   - `TG_TOKEN`
   - `TG_CHAT`
5. Railway auto-detects `requirements.txt` and runs `python bot.py`

**Start command** (set in Railway settings):
```
python bot.py
```

## Backtesting

```bash
# Run backtest on BTCUSDT for full 2024
python backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31

# Run on altcoin
python backtest.py --symbol SOLUSDT --start 2024-06-01 --end 2024-12-31
```

Output:
- Metrics printed to console
- `backtest_BTCUSDT.csv` — all trades
- `backtest_equity.png` — equity curve chart

## Signal Format (Telegram)

```
⚡ PUMP REVERSAL SIGNAL
━━━━━━━━━━━━━━━━━━━━
🪙 Coin: #SOLUSDT
📍 Entry: 185.4200
🛑 Stop Loss: 191.8500
✅ TP1: 179.0000
🎯 TP2: 172.5000
━━━━━━━━━━━━━━━━━━━━
📈 Pump 24H: +38.2%
📊 RSI (4H): 82.4
💰 Funding Rate: 0.0821%
📦 Open Interest: $1,234,500
🔀 Volume Ratio: 5.2x
🔻 OI Divergence: YES
🔫 Liquidity Sweep: YES
━━━━━━━━━━━━━━━━━━━━
⭐ Signal Score: 6/7
```

## Signals CSV

All signals saved to `signals.csv`:
```
date, symbol, entry, stop_loss, tp1, tp2, rsi, funding, open_interest, pump_percent, score
```

## Risk Warning

⚠️ This bot is for **educational and research purposes**.
Crypto futures trading involves significant risk of loss.
Always use proper position sizing and risk management.
