"""
Dragon Trader — Tick-Level ML Trading Agent.
Account: 25035146, $1,000, VantageInternational-Demo.
MT5 bridge: localhost:18813 (rpyc via Wine).
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ═══ MT5 CREDENTIALS ═══
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "25035146"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "C1f%R5*C")
MT5_SERVER = os.getenv("MT5_SERVER", "VantageInternational-Demo")
MT5_HOST = "localhost"
MT5_PORT = 18813  # Separate bridge from ApexQuant (18812)


@dataclass
class SymbolConfig:
    symbol: str
    magic: int
    category: str = "Forex"
    digits: int = 5
    tick_value: float = 1.0      # per lot per point — updated at runtime
    volume_min: float = 0.01
    volume_max: float = 10.0
    volume_step: float = 0.01


# ═══ 11 SYMBOLS (V5 backtested, all PF > 1.5) ═══
SYMBOLS: Dict[str, SymbolConfig] = {
    # Gold
    "XAUUSD":   SymbolConfig("XAUUSD",   8100, "Gold",   2),
    "XAGUSD":   SymbolConfig("XAGUSD",   8140, "Gold",   3),
    # Crypto
    "BTCUSD":   SymbolConfig("BTCUSD",   8110, "Crypto", 2),
    "ETHUSD":   SymbolConfig("ETHUSD",   8230, "Crypto", 2),   # NEW: PF 4.76, $1243
    # Indices
    "NAS100.r": SymbolConfig("NAS100.r", 8120, "Index",  2),
    "JPN225ft": SymbolConfig("JPN225ft", 8150, "Index",  2),
    "SP500.r":  SymbolConfig("SP500.r",  8190, "Index",  2),   # NEW: PF 5.15, $953
    "GER40.r":  SymbolConfig("GER40.r",  8200, "Index",  2),   # NEW: PF 1.91, $249
    # Forex
    "USDCAD":   SymbolConfig("USDCAD",   8180, "Forex",  5),
    "EURJPY":   SymbolConfig("EURJPY",   8210, "Forex",  3),   # NEW: PF 2.79, $451
    "EURUSD":   SymbolConfig("EURUSD",   8220, "Forex",  5),   # NEW: PF 1.82, $313 (LONG only)
    "USDJPY":   SymbolConfig("USDJPY",   8160, "Forex",  3),   # RE-ADDED: PF 2.64, $272 (LONG only)
}

# Per-symbol ML meta-label toggle (Round 6 backtest with retrained models)
# ML ON: symbols where ML filter improves PF (verified per-symbol comparison)
# ML OFF: symbols where ML over-filters good signals (reduces trade count + PF)
DRAGON_ML_ENABLED = {
    "XAUUSD":   True,    # AUC 0.801
    "XAGUSD":   True,    # AUC 0.802
    "BTCUSD":   False,   # weak model, trend needs all signals
    "ETHUSD":   False,   # NEW: no model yet
    "NAS100.r": True,    # AUC 0.799
    "JPN225ft": False,   # AUC 0.665 too weak
    "USDJPY":   False,   # no model retrained for V5 yet
    "USDCAD":   False,   # ML over-filters
    "SP500.r":  False,   # NEW: no model yet
    "GER40.r":  False,   # NEW: no model yet
    "EURJPY":   False,   # NEW: no model yet
    "EURUSD":   False,   # NEW: no model yet
}

# ═══ DRAGON RISK MANAGEMENT (aggressive but survivable — demo phase) ═══
# 90-day PF 1.72 (recent market harder) — stay aggressive but not suicidal
# Compound growth sim: 0.8% risk = $1K → $7.3K/year (630%) with ~30% peak DD
MAX_RISK_PER_TRADE_PCT = 0.8        # 0.8% — aligned with backtest (was 1.2%, too aggressive for $748 account)
MAX_TOTAL_EXPOSURE_PCT = 4.0       # 4.0% total (allows 4 full positions)
DAILY_LOSS_LIMIT_PCT = 3.0         # 3% daily loss warning
MAX_POSITIONS = 4                  # max 4 simultaneous
DD_REDUCE_THRESHOLD = 6.0          # halve risk at 6% DD
DD_PAUSE_THRESHOLD = 10.0          # warn at 10% DD
DD_EMERGENCY_CLOSE = 8.0           # close everything at 8% DD ($740 account — 15% was too high)

# ═══ HARD KILL SWITCHES (cannot be bypassed) ═══
DAILY_HARD_STOP_PCT = 2.0          # HARD STOP: close all + halt trading if daily loss > 2% of start equity
WEEKLY_HARD_STOP_PCT = 5.0         # HARD STOP: close all + halt trading if weekly loss > 5% of start equity

# ═══ PER-SYMBOL RISK CAP (override MAX_RISK for specific symbols) ═══
SYMBOL_RISK_CAP: Dict[str, float] = {
    "BTCUSD": 0.4,                 # HALVED: high variance asset
    "USDCAD": 2.4,                 # 3x Forex
    "EURJPY": 2.4,                 # 3x Forex
    "EURUSD": 2.4,                 # 3x Forex
    "USDJPY": 2.4,                 # 3x Forex
}

# ═══ TICK STREAMING ═══
TICK_INTERVAL_MS = 500             # poll ticks every 500ms
CANDLE_WINDOW = 500                # keep last 500 candles per TF
TIMEFRAMES = [1, 5, 15, 60]       # M1, M5, M15, H1

# ═══ ML MODEL ═══
MODEL_DIR = Path(__file__).parent / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CONFIDENCE_THRESHOLD = 0.60        # min probability to trade
PREDICTION_HORIZON_TF = 15         # M15 forward return for labels

# ═══ TRADING MODE ═══
TRADING_MODE = "hybrid"            # "swing", "scalp", or "hybrid" (both)

# ═══ TRAILING SL — DEFAULT AGGRESSIVE profile ═══
TRAIL_STEPS = [
    (8.0, "trail", 0.3),
    (4.0, "trail", 0.5),
    (2.0, "trail", 0.8),
    (1.5, "lock",  0.7),     # lock 0.7R at 1.5R
    (1.0, "lock",  0.4),     # lock 0.4R at 1.0R
    (0.7, "lock",  0.2),     # lock 0.2R at 0.7R — fills the BE→1R gap
    (0.5, "be",    0.0),     # breakeven at 0.5R
]

# ═══ TRAILING SL — PER-SYMBOL OVERRIDE (V5: tighter profit locks) ═══
# Problem: 39% of trades were tiny wins ($0-2) because locks were too small
# Fix: bigger locks at every level, earlier BE, profit ratchet in executor
SYMBOL_TRAIL_OVERRIDE: Dict[str, list] = {
    "XAUUSD": [  # Gold: wider ATR, need room but lock hard once profitable
        (8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8),
        (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.7, "lock", 0.2), (0.5, "be", 0.0),
    ],
    "XAGUSD": [  # Silver: similar to gold
        (4.0, "trail", 0.3), (2.0, "trail", 0.5), (1.5, "trail", 0.8),
        (1.0, "lock", 0.5), (0.7, "lock", 0.3), (0.4, "be", 0.0),
    ],
    "BTCUSD": [  # Crypto: volatile, lock profits aggressively
        (8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8),
        (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.7, "lock", 0.2), (0.5, "be", 0.0),
    ],
    "ETHUSD": [  # Crypto: same as BTC
        (8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8),
        (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.7, "lock", 0.2), (0.5, "be", 0.0),
    ],
    "NAS100.r": [  # Index: was 294 tiny wins — lock harder
        (4.0, "trail", 0.3), (2.0, "trail", 0.5), (1.5, "trail", 0.8),
        (1.0, "lock", 0.5), (0.7, "lock", 0.3), (0.4, "be", 0.0),
    ],
    "JPN225ft": [  # Japan: lock more, fill the 0.5-1.0R gap
        (8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8),
        (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.7, "lock", 0.2), (0.5, "be", 0.0),
    ],
    "USDCAD": [  # Forex: tight profile
        (4.0, "trail", 0.3), (2.0, "trail", 0.5), (1.5, "trail", 0.8),
        (1.0, "lock", 0.5), (0.7, "lock", 0.3), (0.4, "be", 0.0),
    ],
    "SP500.r": [  # S&P: same as NAS
        (4.0, "trail", 0.3), (2.0, "trail", 0.5), (1.5, "trail", 0.8),
        (1.0, "lock", 0.5), (0.7, "lock", 0.3), (0.4, "be", 0.0),
    ],
    "GER40.r": [  # DAX: same as NAS
        (4.0, "trail", 0.3), (2.0, "trail", 0.5), (1.5, "trail", 0.8),
        (1.0, "lock", 0.5), (0.7, "lock", 0.3), (0.4, "be", 0.0),
    ],
    "EURJPY": [  # Forex: tight
        (4.0, "trail", 0.3), (2.0, "trail", 0.5), (1.5, "trail", 0.8),
        (1.0, "lock", 0.5), (0.7, "lock", 0.3), (0.4, "be", 0.0),
    ],
    "EURUSD": [  # Forex: tight
        (4.0, "trail", 0.3), (2.0, "trail", 0.5), (1.5, "trail", 0.8),
        (1.0, "lock", 0.5), (0.7, "lock", 0.3), (0.4, "be", 0.0),
    ],
    "USDJPY": [  # Forex: tight
        (4.0, "trail", 0.3), (2.0, "trail", 0.5), (1.5, "trail", 0.8),
        (1.0, "lock", 0.5), (0.7, "lock", 0.3), (0.4, "be", 0.0),
    ],
}

# ═══ TRAILING SL — Sub2 RUNNER (wider for big moves, but still locks profit) ═══
SUB2_TRAIL_STEPS = [
    (10.0, "trail", 0.3),            # ultra-tight at 10R
    (8.0, "trail", 0.4),             # tight at 8R
    (6.0, "trail", 0.5),             # at 6R
    (4.0, "trail", 0.7),             # at 4R
    (2.0, "trail", 0.8),             # at 2R
    (1.5, "lock",  0.7),             # lock 0.7R at 1.5R (was 0.5R)
    (1.0, "lock",  0.4),             # lock 0.4R at 1.0R (was 0.2R at 0.8R)
]

# ═══ SCALP CONFIG ═══
SCALP_ENABLED = True
SCALP_RISK_PCT = 0.2              # 0.2% equity per scalp trade (was 0.5)
SCALP_ATR_MULT = 1.5              # SL = 1.5x ATR(M5)
SCALP_MAGIC_OFFSET = 100          # scalp magic = base magic + 100
SCALP_SESSION_START = 13           # scalp session 13:00 UTC
SCALP_SESSION_END = 17             # scalp session 17:00 UTC
SCALP_MAX_PER_SESSION = 2          # max 2 scalps per symbol per session

# ═══ SCALP TRAILING SL (tight profile) ═══
SCALP_TRAIL_STEPS = [
    # (profit_R, action, atr_multiplier_or_lock_R)
    (2.0, "trail", 0.5),             # trail 0.5x ATR at 2R
    (1.5, "trail", 0.7),             # trail 0.7x ATR at 1.5R
    (1.0, "lock",  0.3),             # lock 0.3R at 1R (was 0.5R)
    (0.7, "be",    0.0),             # BE at 0.7R (was 0.5R — more room)
]

# ═══ SESSION FILTER ═══
SESSION_START_UTC = 6              # non-crypto default: 06:00 UTC
SESSION_END_UTC = 22               # non-crypto default: 22:00 UTC

# Per-symbol session overrides (start_utc, end_utc)
# JPN225ft: Asian session starts at 00:00 UTC — default 06:00 misses best hours
# XAUUSD/XAGUSD: London+NY overlap 06-22 is optimal, keep default
SYMBOL_SESSION_OVERRIDE: Dict[str, Tuple[int, int]] = {
    "JPN225ft": (0, 22),           # include Asian session (00-22 UTC)
}

# ═══ ATR SL ═══
ATR_SL_MULTIPLIER = 1.5           # SL = 1.5x ATR default (was 3.0 — KEY FIX for PF)

# Per-symbol ATR SL multiplier overrides (grid search + baseline backtest)
SYMBOL_ATR_SL_OVERRIDE: Dict[str, float] = {
    "XAUUSD":   3.0,    # V5 tune: PF=3.35
    "XAGUSD":   2.5,    # V5 tune: PF=3.33
    "BTCUSD":   3.0,    # V5 tune: PF=4.55
    "ETHUSD":   3.0,    # V5 tune: PF=4.76
    "NAS100.r": 3.0,    # V5 tune: PF=8.66
    "JPN225ft": 2.0,    # V5 tune: PF=5.12
    "USDCAD":   0.5,    # V5 tune: PF=2.30
    "SP500.r":  2.5,    # NEW: default index SL
    "GER40.r":  2.5,    # NEW: default index SL
    "EURJPY":   2.5,    # NEW: default forex SL
    "EURUSD":   2.5,    # NEW: default forex SL
    "USDJPY":   2.5,    # NEW: default forex SL
}

# ═══ SMART ENTRY — Per-Symbol Intelligence Mode ═══
# Backtested: cherry-pick best strategy per symbol (avg PF 2.28 vs 2.19 base)
SMART_ENTRY_MODE: Dict[str, Dict[str, bool]] = {
    # adaptive_trail: scale trail by current ATR vs 50-bar avg ATR
    # fresh_momentum: require MACD acceleration + RSI not exhausted
    "XAUUSD":   {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 2.18)
    "XAGUSD":   {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 2.44)
    "BTCUSD":   {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 5.30)
    "NAS100.r": {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 1.75)
    "USDJPY":   {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 1.79)
    "JPN225ft": {"adaptive_trail": True,  "fresh_momentum": True},   # BOTH: PF 2.73→2.98
    "USDCAD":   {"adaptive_trail": False, "fresh_momentum": True},   # FR.MOM: PF 1.47→1.52
}

# ═══ DASHBOARD ═══
DASHBOARD_PORT = 8888
STARTING_BALANCE = 1000.0

# ═══ SQLITE ═══
DB_PATH = Path(__file__).parent / "data" / "beast.db"

# ═══ DRAGON-SPECIFIC CONSTANTS ═══
DRAGON_MIN_SCORE_BASELINE = 7.0    # minimum score for H1 swing entry

# ═══ M15 PRIMARY SCORING (lower thresholds — M15 has 4x more signals) ═══
DRAGON_M15_MIN_SCORE_BASELINE = 6.0
DRAGON_M15_SYMBOL_MIN_SCORE: Dict[str, Dict[str, float]] = {
    "XAUUSD":   {"trending": 5.5, "ranging": 6.5, "volatile": 6.0, "low_vol": 6.0},
    "XAGUSD":   {"trending": 5.5, "ranging": 6.0, "volatile": 6.5, "low_vol": 6.5},
    "BTCUSD":   {"trending": 6.0, "ranging": 7.0, "volatile": 6.5, "low_vol": 6.5},  # raised +1.0: anti-churn
    "NAS100.r": {"trending": 6.0, "ranging": 6.0, "volatile": 6.0, "low_vol": 6.0},
    "JPN225ft": {"trending": 6.0, "ranging": 6.0, "volatile": 6.5, "low_vol": 6.5},
    "USDJPY":   {"trending": 6.0, "ranging": 6.5, "volatile": 6.5, "low_vol": 6.5},
    "USDCAD":   {"trending": 6.0, "ranging": 6.0, "volatile": 6.0, "low_vol": 6.0},
}

# ═══ MEAN-REVERSION STRATEGY (fires when momentum is flat) ═══
MR_ENABLED = True
MR_MIN_SCORE = 3.0              # Lower bar: BB touch(1) + RSI(1) + EMA dist(1) = 3
MR_RISK_DISCOUNT = 0.7          # 70% of momentum risk (less conviction)
MR_SL_ATR_MULT = 1.0            # Tighter SL (1.0 ATR vs 1.5-2.5 for momentum)
MR_TRAIL_STEPS = [
    (1.5, "trail", 0.5),        # Trail at 1.5R (quick exit for reversion)
    (1.0, "lock", 0.3),         # Lock 0.3R at 1R
    (0.5, "be", 0.0),           # BE at 0.5R
]
# Only fire MR in ranging/low_vol regimes (not trending/volatile)
MR_ALLOWED_REGIMES = {"ranging", "low_vol"}

# ═══ INDUSTRY ENTRY GATES (per-symbol, from V2 backtest validation) ═══
INDUSTRY_GATES_ENABLED: Dict[str, bool] = {
    "XAUUSD":   False,  # mega: PF=1.66 (gates OFF better in walk-forward)
    "XAGUSD":   True,   # mega: PF=2.33 WF=4.76
    "BTCUSD":   False,  # mega: PF=4.05 (gates hurt crypto)
    "NAS100.r": False,  # gates kill NAS (PF 1.01→1.75 OFF)
    "JPN225ft": True,   # mega: PF=2.42 WF=3.37
    "USDJPY":   True,   # mega: PF=2.39 WF=2.22
    "USDCAD":   False,  # gates hurt CAD (PF 1.23→1.29 OFF)
}

# ═══ PRIMARY TIMEFRAME (M15 for entries, H1 for bias) ═══
PRIMARY_TF = 15                 # M15 = primary signal timeframe
BIAS_TF = 60                    # H1 = directional bias only
EVAL_ON_CANDLE_CLOSE = True     # Only score on new M15 candle (not every 500ms)

# Per-symbol regime MIN_SCORE overrides (from grid search optimization)
# Each tested 15-25 combinations, picked highest PF with >= 15 trades
DRAGON_SYMBOL_MIN_SCORE: Dict[str, Dict[str, float]] = {
    "XAUUSD":   {"trending": 6.0, "ranging": 7.0, "volatile": 6.5, "low_vol": 6.5},  # mega WF PF=2.01 188tr
    "XAGUSD":   {"trending": 7.0, "ranging": 7.5, "volatile": 7.2, "low_vol": 7.2},  # PF=2.05 gates ON 139tr
    "BTCUSD":   {"trending": 7.0, "ranging": 8.0, "volatile": 7.5, "low_vol": 7.5},  # raised +1.0: was churning 76 trades/day with -$222/week
    "NAS100.r": {"trending": 7.0, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},  # PF=1.75 gates OFF 132tr
    "JPN225ft": {"trending": 7.0, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},  # mega WF PF=3.37 128tr
    "USDJPY":   {"trending": 7.5, "ranging": 8.0, "volatile": 7.8, "low_vol": 7.8},  # mega WF PF=2.22 104tr
    "USDCAD":   {"trending": 7.0, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},  # PF=1.29 gates OFF 146tr
}
DRAGON_SCALP_MIN_SCORE = 6.5       # minimum score for scalp entry
DRAGON_CONFIDENCE_FLOOR = 0.56     # ML meta-label floor (tuned: XAUUSD WR 37.3%→38.3% at 0.56)
DRAGON_MAX_CONSECUTIVE_LOSSES = 4  # blacklist symbol after 4 consecutive losses (was 3)
DRAGON_BLACKLIST_HOURS = 12        # hours to ban symbol after consecutive losses (was 24)
DRAGON_EQUITY_SLOPE_WINDOW = 20    # trades to measure equity slope
DRAGON_STANDBY_HOURS = 4           # hours of no favorable conditions before standby
DRAGON_RISK_SCALE_MIN = 0.5        # min risk % (floor for low confidence — tuned from 0.4)
DRAGON_RISK_SCALE_MAX = 1.2        # max risk % (tuned from 1.0 — PF 2.73 at 1.2%, DD still 19.4%)
DRAGON_LOSS_DAY_RISK_MULT = 0.5    # halve risk after losing day

# ═══════════════════════════════════════════════════════════════════
#  V5 SIGNAL QUALITY SCORING (0-100 scale, raw H1, NO blending)
# ═══════════════════════════════════════════════════════════════════
# Raw H1 score / divisor * 100 → normalized 0-100
# 12.0 = practical ceiling (11 components, ~12 in strong trends)
SIGNAL_QUALITY_DIVISOR = 12.0

# Per-regime minimum signal_quality (0-100) for entry
# V5 tuned: 45% beats 50% by $212/90d (+25%). Trail handles exit quality.
SIGNAL_QUALITY_THRESHOLDS: Dict[str, int] = {
    "trending": 45,    # 5.4 raw
    "ranging":  45,    # 5.4 raw
    "volatile": 45,    # 5.4 raw
    "low_vol":  45,    # 5.4 raw
}

# Per-symbol quality override (where optimal differs from default)
SIGNAL_QUALITY_SYMBOL: Dict[str, Dict[str, int]] = {
    # All symbols use default 45% — backtest showed 45% beats 50% by 25%
    # Trail system handles exit quality, entry just needs directional conviction
}

# MTF high-conviction override: skip M15 gate if signal_quality >= this
MTF_OVERRIDE_QUALITY = 75  # 9.0 raw — monster signal, M15 doesn't matter

# Conviction sizing on 0-100 scale (replaces old CONVICTION_SIZING)
CONVICTION_SIZING_V2: Dict[str, float] = {
    "80+":   1.5,    # 150% risk — PF 16.21 at this level
    "65-80": 1.2,    # 120% risk
    "55-65": 1.0,    # standard
    "<55":   0.6,    # reduced (only reachable via near-miss or reversal)
}

# ═══ DIRECTION BIAS (backtest-proven edge) ═══
# Restrict symbols to directions with PF > 1.5 (skip marginal/losing direction)
# None = both directions allowed
DIRECTION_BIAS: Dict[str, str] = {
    "XAUUSD":   "LONG",     # LONG PF=2.54 vs SHORT PF=1.02
    "USDCAD":   "SHORT",    # SHORT PF=1.09 vs LONG PF=0.87
    "EURUSD":   "LONG",     # LONG PF=2.27 vs SHORT PF=1.31
    "USDJPY":   "LONG",     # LONG PF=3.43 vs SHORT PF=1.32
    # Both directions: XAGUSD, BTCUSD, NAS100.r, JPN225ft, SP500.r, GER40.r, EURJPY
}

# ═══ CONVICTION-BASED POSITION SIZING ═══
# Score 9+  → PF 16.21 (monster edge) → max size
# Score 8-9 → PF 2.31 (solid) → full size
# Score 7-8 → PF 5.76 (good) → standard size
# Score 6-7 → PF 1.30 (marginal) → reduced size
CONVICTION_SIZING: Dict[str, float] = {
    "9+":  1.5,    # 150% of base risk — high conviction, proven PF 16.21
    "8-9": 1.2,    # 120% of base risk
    "7-8": 1.0,    # 100% standard
    "6-7": 0.6,    # 60% reduced — marginal edge
}

# ═══ TOXIC HOURS — block entries during consistently losing hours ═══
# H01-04: low liquidity noise (but exempt crypto + JPN)
# H07-H08 REMOVED from toxic — was blocking all forex/gold right after session open
TOXIC_HOURS_UTC: set = {1, 2, 3, 4}
# Per-symbol overrides: some symbols trade well during "toxic" hours
TOXIC_HOUR_EXEMPT: Dict[str, set] = {
    "BTCUSD":   {1, 2, 3, 4},  # crypto 24/7
    "JPN225ft": {1, 2, 3, 4},  # Asian index prime hours
}

# ═══ PULLBACK ENTRY — wait for retrace before entering ═══
# Instead of entering at signal bar close, require price to pull back
# towards the signal direction before entering (better fill, higher WR)
PULLBACK_ENTRY_ENABLED = True    # regime-adaptive: ON in trending/volatile, OFF in low_vol/ranging
PULLBACK_ATR_RETRACE = 0.2
PULLBACK_MAX_WAIT_BARS = 3
PULLBACK_REGIMES = {"trending", "volatile"}  # only wait for pullback in these regimes

# ═══ CORRELATION PAIRS ═══
# Won't open simultaneous positions in both symbols if correlation >= threshold
CORRELATION_PAIRS: Dict[Tuple[str, str], float] = {
    ("XAUUSD", "XAGUSD"): 0.85,
    ("NAS100.r", "JPN225ft"): 0.60,
}

# ═══ RL LEARNING — per-symbol toggle + tuned params (576 backtest combos) ═══
RL_ENABLED_SYMBOLS = {
    "XAUUSD",   # PF 1.85→2.19, DD 23.3%→7.7%
    "JPN225ft", # PF 2.36→3.51, DD 5.5%→2.6%
    "USDJPY",   # PF 1.55→2.05, DD 4.2%→2.5%
    "USDCAD",   # PF 1.53→1.69, DD 3.3%→3.1%
    "XAGUSD",   # PF 2.33→2.51, DD 9.0%→9.6%
    "NAS100.r", # PF 2.18→2.24, DD 5.3%→2.9%
    # NOT: BTCUSD — RL kills trend trades, keep pure scoring
}
# Per-symbol RL params (grid-tuned)
RL_SYMBOL_PARAMS: Dict[str, Dict] = {
    "XAUUSD":   {"lookback": 20, "boost_max": 1.2},
    "XAGUSD":   {"lookback": 10, "boost_max": 1.2},
    "NAS100.r": {"lookback": 10, "boost_max": 1.3},
    "JPN225ft": {"lookback": 30, "boost_max": 1.5},
    "USDJPY":   {"lookback": 10, "boost_max": 1.4},
    "USDCAD":   {"lookback": 30, "boost_max": 1.5},
}
