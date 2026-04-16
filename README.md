# B.E.A.S.T — ML-Enhanced Autonomous Trading System

**8,077 lines of code | 12 trained ML models | 4 symbols | Hybrid swing + scalp**

An autonomous trading agent that combines proven rule-based scoring with machine learning intelligence. Runs on MetaTrader 5 via Wine bridge on macOS. Real-time tick streaming, adaptive strategy selection, and a JARVIS-themed monitoring dashboard.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        BEAST TRADER                             │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│  TICK DATA  │  INTELLIGENCE │  EXECUTION   │    DASHBOARD       │
│             │               │              │                    │
│ tick_stream │ momentum_scor │ executor.py  │ app.py (Flask)     │
│ er.py       │ er.py (H1)   │ risk sizing  │ WebSocket push     │
│ 500ms poll  │ scalp_scorer  │ trailing SL  │ TradingView charts │
│ M1/M5/M15/  │ .py (M5)     │ reversal     │ Market scanner     │
│ H1 candles  │ signal_model  │ close        │ Intelligence panel │
│ SQLite tick  │ .py (ML)     │              │ Performance panel  │
│ storage     │ vol_model.py  │              │                    │
│             │ feature_eng   │              │                    │
│             │ ine.py (29f)  │              │                    │
├─────────────┴──────────────┴──────────────┴────────────────────┤
│                     AGENT BRAINS                                │
│  brain.py (H1 swing, 1s cycle)  │  scalp_brain.py (M5, 500ms) │
│  Regime-adaptive MIN_SCORE      │  London+NY overlap only      │
│  ML meta-label filter (AUC 0.8) │  M1 micro-timing             │
│  Tick momentum gate             │  H1 trend bias gate           │
│  Vol model dynamic SL           │  Max 2 scalps/session         │
└─────────────────────────────────┴──────────────────────────────┘
```

---

## Components

### 1. Tick Data Layer (`data/`)

#### `data/tick_streamer.py` — Real-Time Market Data Engine
- Connects to MT5 via rpyc bridge (port 18813)
- Polls bid/ask/volume every **500ms** for all symbols
- Builds OHLC candles in real-time across 4 timeframes: **M1, M5, M15, H1**
- Maintains rolling window of **500 candles** per symbol per timeframe
- Computes live indicators on every candle close: **EMA(20/50/200), ATR(14), RSI(14), MACD(12/26/9), SuperTrend, Bollinger Bands(20,2), ADX(14), VWAP**
- Stores raw ticks in **SQLite** for ML training data collection
- Thread-safe `SharedState` class — all components read from the same state
- Provides `get_tick()`, `get_tick_history()`, `get_candles()`, `get_indicators()` methods

#### `data/feature_engine.py` — ML Feature Generator
- Generates **29 normalized features** from tick and candle data:
  - **Price features (4):** Returns at 1m, 5m, 15m, 1h horizons
  - **Volatility features (3):** Rolling vol at 5m, 15m, 1h
  - **Momentum features (3):** Rate of change at 3 timeframes
  - **Volume features (3):** Tick count ratio, volume delta, VWAP deviation
  - **Microstructure features (3):** Bid-ask spread, spread percentile, tick direction
  - **Technical features (7):** RSI, MACD histogram, BB position, SuperTrend distance, EMA(20/50/200) distances from price
  - **Regime features (3):** Volatility percentile, ADX, trend strength score
  - **Other (3):** ATR normalized, body/range ratio, consecutive candle count
- Rolling Z-score normalization (500-bar deque window)
- Batch generation mode for ML training with forward-return labels
- Live generation mode for real-time prediction

### 2. Signals Layer (`signals/`)

#### `signals/momentum_scorer.py` — H1 Enriched Scoring System (Proven)
The core signal generator, ported from a system with **backtested PF 1.6-2.6** across multiple symbols.

**11-Component Scoring (max ~14 points):**

| # | Component | Max Points | How It Works |
|---|-----------|-----------|--------------|
| 1 | EMA Stack | 1.5 | Stack order + separation strength + VWAP alignment + triple EMA order |
| 2 | SuperTrend | 1.5 | Direction + distance/ATR scaling (far from ST = stronger signal) |
| 3 | MACD Crossover | 1.5 | Line vs signal + acceleration + fresh cross detection (last 3 bars) |
| 4 | MACD Histogram | 1.0 | Expanding/contracting momentum + 5-bar average comparison + fade penalty |
| 5 | RSI | 1.0 | Sweet spot zones (45-65 long, 35-55 short) + momentum direction + pullback bonus + OB/OS penalty |
| 6 | Candlestick Patterns | 2.0 | Engulfing (body quality ratio) + pin bars (wick length) + volume confirmation on patterns |
| 7 | Heikin Ashi | 1.0 | Trend direction + consecutive HA bars (3+ streak) + body expansion detection |
| 8 | Structure Trend | 1.5 | Higher highs/lows + ADX trend strength confirmation + DI+/DI- directional |
| 9 | Breakout | 2.5 | Donchian channel break + Bollinger squeeze release + distance past channel + volume surge |
| 10 | Momentum Velocity | 0.5 | 3-bar rate of change (ROC) |
| 11 | Trend Persistence | 0.5 | Consecutive candle count in same direction |

**Extra Indicators Computed:** ADX, DI+/DI-, VWAP (20-bar proxy), Volume SMA, ROC3, Consecutive count

**Per-Symbol Indicator Params** (from proven sim.py baseline):
- XAUUSD: EMA 15/30/60, SuperTrend 2.0/7, MACD 5/26/4, ATR 7
- BTCUSD: EMA 20/50/60, SuperTrend 3.5/10, MACD 12/26/9, ATR 10
- NAS100/GER40: EMA 15/40/80, SuperTrend 2.5/10, MACD 8/21/7, ATR 10

#### `signals/scalp_scorer.py` — M5 Fast Scoring
- Faster indicators: **EMA 8/21/50, SuperTrend 1.5/7, MACD 5/13/4, ATR 10**
- Reuses same `_score()` function — different indicator values from faster params
- `_m1_micro_direction()`: EMA(3) vs EMA(8) on M1 candles for entry timing precision
- `MIN_SCALP_SCORE = 5.0` (stricter than swing — M5 signals are noisier)

### 3. ML Models (`models/`)

#### `models/signal_model.py` — Meta-Label Trade Filter (LightGBM)
**Not a direction predictor — a trade quality filter.**

Instead of predicting "will price go up?" (coin flip at 51%), it predicts "given that the scoring system says LONG, will THIS specific trade be profitable?"

**Training Process:**
1. Fetch 50,000 H1 candles from MT5 history
2. Run `_compute_indicators()` + `_score()` on every bar
3. Collect signals where score >= MIN_SCORE (4.0)
4. Simulate forward outcome for each signal (ATR-based SL, trailing stop logic)
5. Label: y=1 if trade was profitable, y=0 if loss
6. Build 21 meta-features per signal
7. Train LightGBM with walk-forward validation (70/15/15 split)

**21 Meta-Features:**
`long_score, short_score, chosen_score, direction, adx, bb_width, atr_percentile, rsi, supertrend_dist, ema_alignment, macd_hist_norm, vol_percentile, trend_structure, hour_sin, hour_cos, dow_sin, dow_cos, spread_atr_ratio, recent_win_streak, score_margin, score_vs_threshold`

**Results (walk-forward validation):**

| Symbol | Test AUC | Accuracy | Precision@60% | Filtered PF | Pass Rate |
|--------|----------|----------|---------------|-------------|-----------|
| XAUUSD | 0.800 | 72.7% | 69.3% | 1.79 | 34.1% |
| BTCUSD | 0.799 | 72.1% | 67.4% | 1.64 | 33.9% |
| NAS100 | 0.801 | 73.5% | 68.0% | 1.68 | 31.1% |
| GER40 | 0.803 | 72.3% | 68.7% | 1.71 | 33.8% |

The filter rejects ~66% of signals and keeps only high-quality setups.

#### `models/vol_model.py` — Volatility Prediction (LightGBM Regressor)
Predicts `ATR_next_4bars / ATR_current` — how much will volatility expand or contract?

**Used for dynamic SL sizing:**
- Predicted expansion > 1.0 → widen SL (avoid getting stopped before the move)
- Predicted contraction < 1.0 → tighten SL (lock profits faster)

**22 Features:** ATR percentiles (20/50/100 lookbacks), cyclical hour+DOW encoding, BB width + percentile, range ratios, ATR change ratios, close vs BB bands, volume ratio, body/range ratio

**Results:** R² 0.27-0.58 across symbols. Hour-of-day is the dominant feature (volatility is highly session-dependent).

### 4. Agent Brains (`agent/`)

#### `agent/brain.py` — H1 Swing Decision Engine
Main decision loop running every **1 second**:

```
Every 1s cycle:
  1. Read equity, DD% from SharedState
  2. Daily loss check (2% limit)
  3. Emergency DD check (15% → close all)
  4. For each symbol:
     a. Session filter (non-crypto: 06-22 UTC only)
     b. Get H1 candles, compute indicators + score
     c. Determine regime (trending/ranging/volatile/low_vol)
     d. Apply regime-adaptive MIN_SCORE:
        - trending: 3.5 (take more trades)
        - ranging: 6.0 (very selective)
        - volatile: 5.0 (moderate)
        - low_vol: 4.5 (standard)
     e. Check M15 alignment (must agree with H1 direction)
     f. Check tick momentum (last 5 ticks must not oppose)
     g. ML meta-label filter (confidence > 0.4 to pass)
     h. Risk check (1% per trade, 3% total)
     i. If all pass → ENTER via executor
  5. Manage open positions:
     - Trailing SL (moderate profile)
     - M15 reversal detection → close immediately
  6. Update SharedState for dashboard
```

**Swing Mode:** One position per symbol. Hold until:
- Trailing SL locks profit and gets hit
- Signal reverses direction (close and flip)
- M15 shows reversal with score >= 5.0

**Graceful Degradation:** If ML model has AUC < 0.55, auto-disables to pure scoring mode. System still profitable without ML (just less selective).

#### `agent/scalp_brain.py` — M5 Scalp Decision Engine
Parallel brain running every **500ms**:

- Uses M5 candles with fast indicators (EMA 8/21/50)
- M1 micro-timing for precise entries
- H1 trend bias gate (never scalp against the swing trend)
- **London+NY overlap only** (13:00-17:00 UTC)
- Max **2 scalps per symbol per session**
- Risk: **0.5%** per trade (half of swing)
- SL: **1.5x ATR(M5)**
- TP: **2R hard target**
- Tight trailing: BE@0.5R, lock@1R, trail 0.7xATR@1.5R, trail 0.5xATR@2R

### 5. Execution (`execution/`)

#### `execution/executor.py` — Trade Execution + Position Management
- **Risk-based lot sizing:** `risk_amount / (sl_points × tick_value)` — never exceeds 1% equity
- **Dynamic SL from vol model:** `sl_mult = base × vol_prediction` (wider before vol expansion)
- **Moderate swing trailing profile:**

| Profit | Action | ATR Multiplier |
|--------|--------|----------------|
| +0.5R | SL → breakeven | — |
| +1.0R | SL locks +0.5R | — |
| +1.5R | Trail starts | 2.0× ATR |
| +2.5R | Tighten trail | 1.5× ATR |
| +4.0R | Tighter trail | 1.0× ATR |
| +6.0R | Very tight | 0.7× ATR |

- **MT5 specifics:** `action:6` for SLTP modify, `type_filling:1` (IOC), all values cast to `float()` for rpyc compatibility
- **Broker min stop distance** enforcement
- Separate magic numbers: swing (8100-8130), scalp (8200-8230)

### 6. Dashboard (`dashboard/`)

#### `dashboard/app.py` — JARVIS Real-Time Trading Terminal
**1,650 lines** | Flask-SocketIO | WebSocket push | TradingView charts

**3 Update Tiers:**
- **Tick (500ms):** Prices, bid/ask, spread, sparklines, equity, positions, P&L
- **Chart (1s):** OHLC candles + EMA/SuperTrend/BB overlays + volume
- **Stats (5s):** Scores, gates, regime, ML confidence, trade log, performance

**Layout:**
```
┌──────────────────────────────────────────────────────────────┐
│ HEADER: Logo + Mode + Balance/Equity/P&L + Actions + Clock   │
├────────────────────────────────┬─────────────────────────────┤
│ TICK CHART (60%)               │ MARKET SCANNER (40%)        │
│ TradingView candlesticks       │ 4 symbol cards:             │
│ M1/M5/M15/H1 toggle           │   Price + direction arrow   │
│ EMA/SuperTrend overlays        │   H1 score bar              │
│ Volume histogram               │   ML confidence bar         │
│ Entry/exit markers             │   Gate dots (TF/OFI/VOL/REG)│
│ SL/TP price lines              │   Position P&L (red/green)  │
│                                │   Mini sparkline            │
├────────────────────────────────┼─────────────────────────────┤
│ INTELLIGENCE (50%)             │ PERFORMANCE (50%)           │
│ Regime badges per symbol       │ Equity curve chart          │
│ Score breakdown (11 components)│ Rolling WR/PF/Sharpe        │
│ Feature importance             │ Trade distribution histogram│
│ Recent trade log               │ Daily P&L chart             │
└────────────────────────────────┴─────────────────────────────┘
```

**JARVIS Theme:** Dark blue (#020810), cyan accents (#00f0ff), Orbitron/Rajdhani fonts, scanline animation, arc reactor, holographic borders, angular clip-path corners

**3 Dedicated MT5 Bridges:**
- Port 18812: Agent brain
- Port 18813: Tick streamer
- Port 18814: Dashboard (never competes with agent)

### 7. Backtesting (`backtest/`)

#### `backtest/engine.py` — Full Backtest Engine
- Complete momentum scoring (self-contained, no import dependencies)
- Swing mode with trailing SL (moderate profile)
- Transaction costs: spread + 1pt slippage per side
- Risk-based lot sizing
- Session filter
- Optional meta-label filter function
- Reports: PF, Sharpe, max DD, avg R, win rate, exit reason breakdown

#### `backtest/mirror_backtest.py` — Live Setup Replica
Mirrors the exact live configuration:
- Regime-adaptive MIN_SCORE
- ML filter simulation (score-based pass probability)
- 1.5x ATR SL minimum
- Real broker spreads
- All 23 available symbols

#### `backtest/cost_model.py` — Transaction Cost Model
- Spread per symbol (from real broker data)
- Slippage: 1 point per side
- Round-trip cost calculation

### 8. Configuration (`config.py`)

```python
# Symbols
SYMBOLS = {XAUUSD, BTCUSD, NAS100.r, GER40.r}

# Risk Management
MAX_RISK_PER_TRADE_PCT = 1.0%     # per trade
MAX_TOTAL_EXPOSURE_PCT = 3.0%     # all positions combined
DAILY_LOSS_LIMIT_PCT = 2.0%       # stop trading for the day
DD_REDUCE_SIZE = 5.0%             # halve position size
DD_PAUSE_ENTRIES = 10.0%          # stop new entries
DD_EMERGENCY_CLOSE = 15.0%       # close everything

# Trading Mode
TRADING_MODE = "hybrid"           # swing + scalp simultaneously

# Scalp Settings
SCALP_SESSION = 13:00-17:00 UTC   # London+NY overlap only
SCALP_MAX_PER_SESSION = 2         # per symbol
SCALP_RISK = 0.5%                 # half of swing risk
```

---

## Account

- **Login:** 25035146
- **Broker:** VantageInternational-Demo
- **Starting Balance:** $1,000
- **Leverage:** 1:500

---

## Infrastructure

| Component | Port | Auto-Restart |
|-----------|------|-------------|
| Beast Agent | — | launchd `com.beast.trader` (KeepAlive) |
| Tick Streamer | 18813 | Part of agent process |
| Dashboard | 8888 | Part of agent process |
| Bridge (Agent) | 18812 | Wine process |
| Bridge (Streamer) | 18813 | launchd `com.apexquant.bridge2` |
| Bridge (Dashboard) | 18814 | launchd `com.beast.bridge3` |

**Critical:** Always start with `python3 -B run.py` (no bytecode cache).

---

## How to Run

```bash
# Start everything
cd /Users/ashish/Documents/beast-trader
python3 -B run.py

# Or via launchd (auto-restart)
launchctl load ~/Library/LaunchAgents/com.beast.trader.plist

# Dashboard
open http://127.0.0.1:8888

# Backtest
python3 -B backtest/mirror_backtest.py
python3 -B backtest/full_scan.py

# Force retrain models
python3 -B run.py --train
```

---

## File Structure

```
beast-trader/
├── run.py                      # Entry point — starts everything
├── config.py                   # All settings, symbols, risk limits
├── .env                        # MT5 credentials
├── requirements.txt            # Dependencies
│
├── agent/
│   ├── brain.py                # H1 swing decision engine (1s cycle)
│   └── scalp_brain.py          # M5 scalp engine (500ms cycle)
│
├── signals/
│   ├── momentum_scorer.py      # Proven 11-point scoring (H1)
│   └── scalp_scorer.py         # Fast M5 scoring
│
├── models/
│   ├── signal_model.py         # Meta-label ML filter (LightGBM)
│   ├── vol_model.py            # Volatility prediction
│   └── saved/                  # 12 trained model files (.pkl)
│
├── data/
│   ├── tick_streamer.py        # Real-time tick + candle engine
│   └── feature_engine.py       # 29-feature ML pipeline
│
├── execution/
│   └── executor.py             # Risk-based execution + trailing SL
│
├── dashboard/
│   └── app.py                  # JARVIS WebSocket dashboard
│
├── backtest/
│   ├── engine.py               # Full backtest engine
│   ├── mirror_backtest.py      # Live setup replica
│   ├── full_scan.py            # All-symbol scanner
│   └── cost_model.py           # Transaction costs
│
└── logs/
    ├── beast.log               # Agent decisions
    ├── beast_stdout.log         # Full output
    └── ticks.db                # SQLite tick storage
```

---

## Dependencies

```
mt5linux        # MT5 bridge for macOS
rpyc            # Remote procedure calls
flask           # Web framework
flask-socketio  # WebSocket support
lightgbm        # ML models
numpy pandas    # Data processing
scikit-learn    # ML utilities
```
