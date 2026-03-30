# Pump Reversal Bot 🚀

Short reversal strategy bot for crypto futures trading.
Detects pump exhaustion and sends high-quality short signals to Telegram.

## Key Features

- **Multi-timeframe confirmation** (4H, 1H, 15m) – increases reliability
- **Volume Profile & advanced resistance levels** – precise entries
- **RSI / MACD divergences** – early reversal detection
- **Candlestick patterns** (pin bars, engulfing)
- **Volume climax detection** – exhaustion signals
- **Weighted score & confidence** – filters out weak setups
- **Historical backtesting** with realistic OI/funding data
- **Parallel scanning** – up to 10 symbols simultaneously
- **Configurable via environment variables** (only essential ones)

## Signal Criteria

The bot uses a **weighted scoring system** (optimized defaults):

| Factor | Weight | Description |
|--------|--------|-------------|
| Pump (24H) | 0.5 | Must be ≥ 25% |
| Volume spike | 0.5 | Last volume ≥3x avg |
| RSI overbought (4H) | 1.0 | RSI ≥ 75 |
| Near resistance | 1.5 | Within 3% of 4H/1D level |
| High funding rate | 1.0 | ≥0.05% |
| OI divergence | 1.0 | Price up, OI down |
| Liquidity sweep | 0.8 | Wick above recent highs |
| Pin bar (4H) | 1.2 | Long upper wick |
| Pin bar (1H) | 0.8 | Confirm on lower TF |
| RSI divergence | 1.0 | Bearish divergence |
| MACD divergence | 1.0 | Bearish divergence |
| Volume climax | 0.7 | Huge volume, price stalls |
| Weak after pump | 0.5 | No new highs |
| 1H confirmation | 1.0 | RSI ≥ 70 on 1H |
| 15m confirmation | 1.5 | Strong reversal pattern |

**Minimum score threshold: 4.0** (adjustable via env)

## Installation

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables (see below)
4. Run: `python bot.py`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TG_TOKEN` | Telegram bot token | Required |
| `TG_CHAT` | Telegram chat ID | Required |
| `SCAN_INTERVAL_SEC` | Scan frequency (seconds) | 300 |
| `MIN_PUMP_PCT` | Minimum 24H pump % | 25 |
| `MIN_SCORE` | Minimum weighted score | 4.0 |

## Backtesting

```bash
python backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31