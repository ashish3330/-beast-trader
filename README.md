# D.R.A.G.O.N — Deep Regime-Adaptive Generative Order Navigator

**Autonomous AI Trading Agent | 6 Symbols | 4-Timeframe Intelligence | Self-Learning**

An autonomous trading agent that watches M1/M5/M15/H1 candles in real-time, makes entry/exit/SL/TP decisions using multi-timeframe intelligence, and learns from every trade to get smarter over time.

---

## System Overview

```
Market Ticks (500ms)
    ↓
MTF Intelligence Engine (H1 + M15 + M5 + M1)
    ├── Volume profile (directional, climax, dry-up)
    ├── Swing structure (HH/HL/LH/LL detection)
    ├── Momentum quality (acceleration, exhaustion)
    ├── Order flow (buy/sell pressure, absorption)
    ├── Smart SL (M15 swing levels, not blind ATR)
    ├── Smart TP (structural resistance/support)
    ├── Entry quality score (0-100)
    └── Exit urgency (0-1.0)
    ↓
Momentum Scorer (11 components + exhaustion penalty)
    ↓
ML Meta-Label Filter (33 features, LightGBM, AUC 0.74-0.80)
    ↓
MasterBrain (10+ gates)
    ├── MTF confluence (≥1/4 TFs must agree)
    ├── MTF entry quality (≥25/100)
    ├── Circuit breaker (2 losses = 4h pause)
    ├── Win cooldown (1h rest per symbol after profit)
    ├── Symbol blacklist (3 consecutive losses = 24h ban)
    ├── Correlation filter (no XAUUSD + XAGUSD simultaneously)
    ├── Net directional cap (max 3 same direction)
    ├── Equity slope scaling
    ├── Anti-martingale (+30% on winning streak)
    └── Learning engine adaptive risk
    ↓
Executor
    ├── 3-sub positions (50%@2R, 30%@3R, 20%@trail)
    ├── Single position for trend-followers (BTCUSD)
    ├── Force-single if lot can't split (small account safety)
    └── Per-symbol progressive trailing locks
    ↓
Position Management
    ├── Per-symbol trailing SL (grid-search optimized)
    ├── MTF exit urgency (≥0.7 + profit = close)
    ├── RSI divergence exit (early reversal detection)
    ├── Momentum decay (give-back protection)
    ├── Time decay (20/40/60 bar tiers)
    └── Weekend protection (close <1.5R Friday 20:00 UTC)
    ↓
Learning Engine
    ├── SQLite trade journal (every MT5 deal synced)
    ├── Rolling 10-trade PF per symbol → adaptive risk multiplier
    ├── Daily auto-retrain ML models (only deploy if AUC improves)
    └── Save all 4-TF candles to cache for retraining
```

---

## Symbols

| Symbol | Category | ATR SL | ML | Trail Profile |
|--------|----------|--------|-----|---------------|
| XAUUSD | Gold | 0.5x | ON (AUC 0.776) | Progressive 0.3R→0.6R→1.0R |
| XAGUSD | Gold | 1.5x | ON (AUC 0.803) | Progressive 0.4R→0.6R→1.0R |
| BTCUSD | Crypto | 1.5x | OFF | Original BE+lock (trend-follower) |
| NAS100.r | Index | 1.0x | ON (AUC 0.740) | Progressive 0.3R→0.6R→1.0R |
| JPN225ft | Index | 1.0x | OFF | Progressive 0.15R→0.6R→1.0R |
| USDJPY | Forex | 2.0x | ON (AUC 0.744) | Progressive 0.15R→0.6R→1.0R |

---

## ML Models

**33-feature LightGBM meta-label classifier** per symbol.

Features include:
- Score components (long/short/chosen/margin)
- Indicator context (ADX, BBW, RSI, SuperTrend, MACD, EMA alignment)
- Multi-timeframe (M15 RSI, EMA alignment, ATR ratio)
- Momentum persistence (1/3/5 bar returns, consecutive candles)
- Reversal detection (RSI divergence, distance from 20-bar high/low)
- Volatility (ATR percentile, BB squeeze, ATR expansion)
- Time (hour-of-day, day-of-week cyclical encoding)
- Microstructure (spread/ATR ratio, recent win streak)

Tuned hyperparams per symbol (learning rate, regularization, feature/bagging fraction).

Auto-retrain daily at 04:00 UTC — **only deploys if test AUC improves**, otherwise keeps existing model.

---

## Risk Management

| Parameter | Value |
|-----------|-------|
| Risk per trade | 1.2% equity (scaled 0.5-1.2% by confidence) |
| Max exposure | 4.0% total |
| Max positions | 4 simultaneous |
| DD halve risk | 6% |
| DD warn | 10% |
| DD emergency close | 15% |
| Circuit breaker | 2 consecutive losses = 4h pause |
| Win cooldown | 1h rest per symbol after profit |
| Anti-martingale | +30% risk on positive equity slope |
| Learning adaptive | 0.5x-1.3x per symbol based on rolling PF |

---

## Dashboard

Vue.js real-time dashboard at `http://localhost:8888`

- **Header**: Balance, Equity, Float P&L, Daily P&L (real-time), Positions, Session
- **Scanner**: 6 symbol cards with scores, MTF confluence dots, entry quality gauge, gate status, position P&L, exit urgency
- **Chart**: TradingView lightweight-charts with M1/M5/M15/H1 + EMA overlays
- **Intelligence**: MTF 4-TF direction grid, MasterBrain status, signal breakdown, volume/swing/momentum indicators
- **Performance**: Equity curve, WR/PF/Sharpe/DD stats, R-multiple histogram
- **Trade Log**: MT5 deal history with pagination, real P&L

---

## File Structure

```
beast-trader/
├── run.py                      # Entry point — starts all components
├── config.py                   # All configuration (risk, symbols, trails, thresholds)
├── agent/
│   ├── brain.py                # Main decision loop (1s cycle)
│   ├── scalp_brain.py          # M5 scalp brain
│   ├── master_brain.py         # 10+ gate approval system
│   ├── exit_intelligence.py    # 6 exit strategies
│   ├── mtf_intelligence.py     # 4-TF market monitoring (1690 lines)
│   └── learning_engine.py      # Trade journal + adaptive risk + auto-retrain
├── execution/
│   └── executor.py             # Order execution, 3-sub architecture, trailing SL
├── signals/
│   └── momentum_scorer.py      # 11-component scoring system
├── models/
│   ├── signal_model.py         # LightGBM meta-label (33 features)
│   ├── vol_model.py            # Volatility model for dynamic SL
│   ├── deep_model.py           # CNN+LSTM ensemble (future use)
│   └── saved/                  # Trained model files
├── data/
│   ├── tick_streamer.py        # MT5 bridge, SharedState, candle building
│   ├── feature_engine.py       # Feature computation
│   └── trade_journal.db        # SQLite learning journal
├── dashboard/
│   ├── app.py                  # Flask-SocketIO backend
│   └── vue_app.py              # Vue.js frontend (83KB)
├── backtest/
│   ├── dragon_backtest.py      # Main backtest engine
│   └── dragon_3sub_backtest.py # 3-sub position backtest
└── train_meta_labels.py        # Offline ML training script
```

---

## Setup

```bash
# Prerequisites: MT5 running via Wine on macOS, Python 3.11+

# Install dependencies
pip install flask flask-socketio lightgbm numpy pandas scikit-learn mt5linux torch

# Train ML models
python3 -B train_meta_labels.py

# Run agent
python3 -B run.py

# Dashboard
open http://localhost:8888
```

## Auto-Start (macOS launchd)

```
com.dragon.trader          — Agent (auto-restart on failure)
com.dragon.bridge-tick     — MT5 tick streamer (port 18813)
com.dragon.bridge-dashboard — MT5 dashboard bridge (port 18814)
com.dragon.mt5             — MetaTrader 5 application
```

---

## Account

- **Demo**: 25035146, VantageInternational-Demo
- **Starting**: $1,000

---

## Backtest Results (365 days, 1.2% risk)

| Symbol | PF | WR | Return |
|--------|-----|-----|--------|
| BTCUSD | 3.84 | 23.3% | +1,441,530% |
| JPN225ft | 2.02 | 30.4% | +359% |
| XAUUSD | 1.69 | 38.2% | +142% |
| NAS100.r | 1.55 | 39.0% | +62% |
| XAGUSD | 1.48 | 44.8% | +182% |
| USDJPY | 1.36 | 43.9% | +24% |
| **Portfolio** | **3.84** | | |

*Note: Backtest uses single-position H1 scoring. Live system uses MTF intelligence + 3-sub positions which cannot be backtested on historical data.*

---

## Version History

- **v1.0** (Beast): 4 symbols, basic scoring, single position
- **v2.0** (Dragon): 6 symbols, MasterBrain, 3-sub, progressive locks
- **v3.0** (Current): MTF Intelligence, 33-feature ML, learning engine, smart SL/TP, volume/swing/momentum/order flow analysis
