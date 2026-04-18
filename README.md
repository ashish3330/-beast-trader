# D.R.A.G.O.N — Deep Regime-Adaptive Generative Order Navigator

**Autonomous AI Trading Agent | 10 Symbols | 4-Timeframe Intelligence | 42-Feature ML | Self-Learning**

An autonomous trading agent that watches M1/M5/M15/H1 candles in real-time, makes entry/exit/SL/TP decisions using multi-timeframe intelligence, and learns from every trade to get smarter over time.

---

## System Flow

```
Market Ticks (500ms poll from MT5 via rpyc bridge)
    │
    ▼
TickStreamer ── builds M1/M5/M15/H1 candles ── stores in SharedState
    │
    ▼
AgentBrain (1s decision cycle, per symbol)
    │
    ├─[1] Session Filter ── non-crypto outside 06-22 UTC? skip
    │
    ├─[2] H1 Indicators ── EMA(15/40/80), SuperTrend, MACD, RSI, ATR, BBW
    │
    ├─[3] Momentum Scoring ── 11-component system (0-10 per direction)
    │       ├── EMA stack alignment
    │       ├── SuperTrend confirmation
    │       ├── MACD crossover + histogram
    │       ├── RSI position
    │       ├── Engulfing / Pin bar patterns
    │       ├── Heikin Ashi momentum
    │       ├── Structure trend (HH/HL/LH/LL)
    │       ├── Donchian breakout
    │       └── Bollinger Band breakout
    │       × session alpha multiplier × day-of-week multiplier
    │
    ├─[4] Regime Detection ── BBW + ADX → trending/ranging/volatile/low_vol
    │       └── Per-symbol adaptive MIN_SCORE threshold
    │           (e.g. XAGUSD trending=5.5, ranging=8.5)
    │
    ├─[5] Direction ── score >= adaptive_min? → LONG/SHORT/FLAT
    │       └── FLAT = no entry (score too weak)
    │
    ├─[6] M15 Alignment ── M15 EMA+SuperTrend must not oppose
    │
    ├─[7] ML Meta-Label Filter (42 features, 3-model LightGBM ensemble)
    │       ├── H1 features: scores, ADX, BBW, RSI, ATR%, EMA, MACD
    │       ├── M15 features: RSI, EMA align, ATR ratio, MACD, ADX, BB position
    │       ├── M5 features: RSI, EMA align, ATR ratio, momentum, consec candles
    │       ├── Cross-TF: mtf_agreement (-1 to +1)
    │       ├── Momentum: 1/3/5 bar returns, consecutive candles
    │       ├── Regime: vol percentile, ATR change, BB squeeze
    │       ├── Reversal: RSI divergence, distance from 20-bar high/low
    │       └── Time: hour/dow cyclical encoding, spread ratio
    │       → confidence >= 0.50 passes (symbol-specific AUC 0.67-0.83)
    │
    ├─[8] MasterBrain (12+ gate approval)
    │       ├── Circuit breaker (2 losses → 4h pause)
    │       ├── Win cooldown (1h rest after profit)
    │       ├── Symbol blacklist (3 consecutive losses → 24h ban)
    │       ├── Score floor (absolute minimum 4.0)
    │       ├── MTF confluence (≥1/4 TFs agree)
    │       ├── MTF entry quality (≥20/100)
    │       ├── Correlation filter (no XAUUSD + XAGUSD together)
    │       ├── Net directional cap (max 3 same direction)
    │       ├── Portfolio risk (VaR, concentration limits)
    │       ├── Equity slope scaling
    │       ├── Anti-martingale (+30% on winning streak)
    │       └── Learning engine adaptive risk (0.5x-1.3x per symbol PF)
    │       → Outputs: approved + dynamic risk_pct (0.5%-1.2%)
    │
    ├─[9] MTF Intelligence (2917-line engine, 21 institutional features)
    │       ├── 4-TF analysis (H1/M15/M5/M1)
    │       ├── Volume profile, swing structure, momentum quality
    │       ├── Order flow, liquidity zones, Fibonacci levels
    │       ├── Session overlap, candle patterns, RSI divergence
    │       ├── Volatility cycle, correlation regime, mean reversion
    │       ├── Smart SL (swing-based, not blind ATR)
    │       ├── Smart TP (structural resistance/support)
    │       ├── SL/TP scaling by regime:
    │       │     Strong trend: SL -15%, TP +30%
    │       │     Ranging:      SL +15%, TP -25%
    │       │     Volatile:     SL +25%, TP +20%
    │       └── entry_quality < 20 → reject
    │
    └─[10] EXECUTE
            ├── Lot size = equity × risk_pct / SL_distance
            ├── 3-sub positions: 50%@2R, 30%@3R, 20%@trail
            │   (force-single if lot can't split on small account)
            ├── Per-symbol progressive trailing locks
            ├── Spread spike check (delay if 2x normal)
            └── Requote retry (up to 3x with fresh price)
```

### Exit Flow (every 1s cycle on open positions)

```
EquityGuardian (real-time P&L monitor)
    ├── Daily drawdown emergency (1.5x limit → close all)
    ├── Sharp loss cut (>2% equity change in <10min)
    ├── Profit giveback protection (60%+ of peak)
    ├── Stale loser cut (>1% equity for >2 hours)
    ├── Portfolio heat (>5% with 3+ positions → cut worst)
    └── Rapid loss circuit breaker (3 cuts/day → close all)

ExitIntelligence (smart exit rules)
    ├── Momentum decay (peak ≥1.5R, gave back 60%)
    ├── RSI divergence (price vs RSI disagreement)
    ├── M15 opposing strength (profit-aware thresholds)
    ├── Time decay (20/40/60 bar tiers)
    ├── Breakeven protection (was >1R, now near entry)
    └── Weekend protection (Friday 20:00 UTC, non-crypto <1.5R)

Trailing SL (per-symbol grid-optimized profiles)
    ├── orig:  BE@0.5R → lock@1R → trail 1.5x/1.0x/0.7x ATR
    ├── aggr:  lock@0.8R → trail from 1.5R
    ├── prog:  lock@0.3R → 0.6R → 0.33R → trail from 2R
    └── tight: BE@0.3R → lock@0.5R → trail from 1.5R

MTF Exit Urgency
    └── urgency ≥ 0.7 AND in profit → close
```

### Learning Feedback Loop (background, 30s cycle)

```
LearningEngine
    ├── Sync MT5 deals to SQLite journal (dedup by ticket)
    ├── Per-symbol rolling PF → adaptive risk multiplier
    │     PF > 2.0 → 1.3x risk (press winners)
    │     PF < 0.7 → 0.5x risk (survival mode)
    ├── Daily stats collection (midnight UTC)
    ├── Cache all 4-TF candles to disk (04:00 UTC)
    └── Auto-retrain ML models daily (04:00 UTC)
          └── Only deploy if new AUC > old AUC
```

---

## Symbols (10)

| Symbol | Category | SL | Trail | ML | AUC | Backtest PF |
|--------|----------|-----|-------|-----|------|-------------|
| BTCUSD | Crypto | 1.0x | prog | ON | 0.675 | 5.30 |
| JPN225ft | Index | 0.5x | aggr | ON | 0.676 | 2.73 |
| XAGUSD | Gold | 2.0x | aggr | ON | 0.817 | 2.44 |
| XAUUSD | Gold | 2.5x | orig | ON | 0.825 | 2.18 |
| USDCHF | Forex | 2.5x | orig | ON | 0.830 | 1.77 |
| USDJPY | Forex | 2.0x | prog | ON | 0.792 | 1.79 |
| NAS100.r | Index | 2.0x | prog | ON | 0.802 | 1.75 |
| EURJPY | Forex | 2.0x | prog | ON | 0.823 | 1.35 |
| USDCAD | Forex | 2.5x | tight | OFF | — | 1.47 |
| EURAUD | Forex | 2.5x | prog | OFF | — | 1.10 |

**Per-symbol MIN_SCORE thresholds** (grid-searched, regime-adaptive):
- Each symbol has 4 thresholds: trending, ranging, volatile, low_vol
- Example: XAGUSD trending=5.5, ranging=8.5

---

## ML Models

**42-feature LightGBM ensemble** (3 models per symbol, AUC-weighted stacking):
- Model A: Tuned LightGBM (per-symbol hyperparams)
- Model B: High-regularization LightGBM (diversity)
- Model C: ExtraTreesClassifier (further diversity)

**Real multi-timeframe features** — trained on actual M15 + M5 candle data from MT5:
- H1 (21 features): scores, indicators, regime, time, microstructure
- M15 (6 features): RSI, EMA align, ATR ratio, MACD hist, ADX, BB position
- M5 (5 features): RSI, EMA align, ATR ratio, momentum, consecutive candles
- Cross-TF (1 feature): mtf_agreement across H1/M15/M5
- Momentum (4 features): 1/3/5 bar returns, consecutive candles
- Reversal (5 features): RSI divergence, high/low distance, ATR change, BB squeeze

Auto-retrain daily at 04:00 UTC — **only deploys if test AUC improves**.

---

## Risk Management

| Parameter | Value |
|-----------|-------|
| Risk per trade | 0.5-1.2% equity (MasterBrain-scaled by confidence) |
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

- **Header**: Balance, Equity, Float P&L, Daily P&L, Positions, Session
- **Scanner**: 10 symbol cards with scores, MTF confluence, entry quality, gate status, position P&L
- **Chart**: TradingView lightweight-charts with M1/M5/M15/H1 + EMA overlays
- **Intelligence**: MTF 4-TF direction grid, MasterBrain status, signal breakdown
- **Performance**: Equity curve, WR/PF/Sharpe/DD stats, R-multiple histogram
- **Trade Log**: MT5 deal history with pagination

---

## File Structure

```
beast-trader/
├── run.py                          # Entry point — wires all components
├── config.py                       # Single source of truth (risk, symbols, trails, ML, thresholds)
├── agent/
│   ├── brain.py                    # Main decision loop (1s cycle)
│   ├── scalp_brain.py              # M5 scalp brain (hybrid mode)
│   ├── master_brain.py             # 12+ gate approval system
│   ├── exit_intelligence.py        # 6 smart exit strategies
│   ├── equity_guardian.py          # Real-time P&L protection
│   ├── mtf_intelligence.py         # 4-TF market analysis (2917 lines, 21 features)
│   ├── learning_engine.py          # Trade journal + adaptive risk + auto-retrain
│   └── portfolio_risk.py           # VaR, correlation matrix, concentration limits
├── execution/
│   └── executor.py                 # Order execution, 3-sub architecture, trailing SL
├── signals/
│   └── momentum_scorer.py          # 11-component scoring system
├── models/
│   ├── signal_model.py             # 42-feature LightGBM meta-label ensemble
│   ├── vol_model.py                # Volatility model for dynamic SL
│   └── saved/                      # Trained model files (.pkl)
├── data/
│   ├── tick_streamer.py            # MT5 bridge, SharedState, candle building
│   ├── feature_engine.py           # Feature computation
│   └── beast.db                    # SQLite learning journal
├── dashboard/
│   ├── app.py                      # Flask-SocketIO backend
│   └── vue_app.py                  # Vue.js frontend
├── backtest/
│   ├── dragon_backtest.py          # Main backtest engine (all 24 symbols)
│   └── grid_tune.py               # Per-symbol parameter optimizer
└── tests/
    └── test_all.py                 # 65 unit tests
```

---

## Setup

```bash
# Prerequisites: MT5 running via Wine on macOS, Python 3.11+
pip install flask flask-socketio lightgbm numpy pandas scikit-learn rpyc python-dotenv

# Train ML models (or auto-trains on first run)
python3 -B run.py --train

# Run agent
python3 -B run.py

# Dashboard
open http://localhost:8888
```

## Auto-Start (macOS launchd)

```
com.dragon.trader          — Agent (auto-restart on failure)
com.dragon.bridge-tick     — MT5 rpyc bridge (port 18813)
com.dragon.bridge-dashboard — MT5 dashboard bridge (port 18814)
com.dragon.mt5             — MetaTrader 5 application
```

---

## Account

- **Demo**: 25035146, VantageInternational-Demo, ~$995
- **Starting**: $1,000

---

## Backtest Results (365 days, 0.8% risk, real spreads, no slippage)

| Symbol | PF | WR% | Return | DD% | ML |
|--------|-----|------|--------|-----|-----|
| BTCUSD | 5.30 | 21.8% | +130,741% | 4.4% | ON |
| JPN225ft | 2.73 | 25.9% | +557% | 8.6% | ON |
| XAGUSD | 2.44 | 57.3% | +392% | 7.1% | ON |
| XAUUSD | 2.18 | 45.1% | +203% | 12.1% | ON |
| USDCHF | 1.77 | 47.3% | +20% | 4.8% | ON |
| USDJPY | 1.79 | 73.5% | +26% | 6.2% | ON |
| NAS100.r | 1.75 | 49.2% | +109% | 5.7% | ON |
| USDCAD | 1.47 | 52.7% | +19% | 3.5% | OFF |
| EURJPY | 1.35 | 66.4% | +15% | 8.6% | ON |
| EURAUD | 1.10 | 69.0% | +3% | 6.0% | OFF |
| **Portfolio** | **5.04** | | | | |

---

## Version History

- **v1.0** (Beast): 4 symbols, basic scoring, single position
- **v2.0** (Dragon): 6 symbols, MasterBrain, 3-sub, progressive locks
- **v3.0**: MTF Intelligence, 33-feature ML, learning engine, smart SL/TP
- **v3.5** (Current): 10 symbols, 42-feature real MTF ML (M15+M5), grid-tuned forex, per-symbol ML ON/OFF, Equity Guardian, HA exit tested+rejected
