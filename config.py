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


# ═══ 6 SYMBOLS ═══
SYMBOLS: Dict[str, SymbolConfig] = {
    "XAUUSD":   SymbolConfig("XAUUSD",   8100, "Gold",   2),
    "XAGUSD":   SymbolConfig("XAGUSD",   8140, "Gold",   3),
    "BTCUSD":   SymbolConfig("BTCUSD",   8110, "Crypto", 2),
    "NAS100.r": SymbolConfig("NAS100.r", 8120, "Index",  2),
    "JPN225ft": SymbolConfig("JPN225ft", 8150, "Index",  2),
    "USDJPY":   SymbolConfig("USDJPY",   8160, "Forex",  3),
}

# Per-symbol ML meta-label toggle (Round 6 backtest with retrained models)
# ML ON: symbols where ML filter improves PF (verified per-symbol comparison)
# ML OFF: symbols where ML over-filters good signals (reduces trade count + PF)
DRAGON_ML_ENABLED = {
    "XAUUSD":   True,    # ON PF 1.69 vs OFF 1.41 — ML wins (AUC 0.742)
    "XAGUSD":   True,    # ON PF 1.48 vs OFF 1.10 — ML wins big (AUC 0.777)
    "BTCUSD":   False,   # ON PF 2.82 vs OFF 2.97 — OFF wins (trend needs all signals)
    "NAS100.r": False,   # ON PF 1.35 vs OFF 1.65 — OFF wins
    "JPN225ft": False,   # ON PF 2.09 vs OFF 2.10 — OFF wins (marginal)
    "USDJPY":   False,   # model max output 0.534 — blocks everything, OFF is better
}

# ═══ DRAGON RISK MANAGEMENT (aggressive but survivable — demo phase) ═══
# 90-day PF 1.72 (recent market harder) — stay aggressive but not suicidal
# Compound growth sim: 0.8% risk = $1K → $7.3K/year (630%) with ~30% peak DD
MAX_RISK_PER_TRADE_PCT = 1.0       # 1.0% equity per trade (balanced growth + survival)
MAX_TOTAL_EXPOSURE_PCT = 4.0       # 4.0% total (allows 4 full positions)
DAILY_LOSS_LIMIT_PCT = 3.0         # 3% daily loss warning
MAX_POSITIONS = 4                  # max 4 simultaneous
DD_REDUCE_THRESHOLD = 6.0          # halve risk at 6% DD
DD_PAUSE_THRESHOLD = 10.0          # warn at 10% DD
DD_EMERGENCY_CLOSE = 15.0          # close everything at 15% DD

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
    (8.0, "trail", 0.5),
    (4.0, "trail", 0.7),
    (2.0, "trail", 1.0),
    (1.5, "trail", 1.5),
    (0.8, "lock",  0.2),
]

# ═══ TRAILING SL — PER-SYMBOL OVERRIDE (backtested) ═══
# XAUUSD: lock 0.10R at 0.3R → PF 1.30→1.96, WR 47→69%, DD 19→8%
SYMBOL_TRAIL_OVERRIDE: Dict[str, list] = {
    "XAUUSD": [
        (6.0, "trail", 0.5),             # tight trail at 6R
        (4.0, "trail", 0.7),             # trail at 4R
        (2.0, "trail", 1.0),             # trail at 2R
        (1.5, "trail", 1.5),             # trail at 1.5R (runners breathe)
        (1.0, "lock",  0.33),            # lock 0.33R at 1R
        (0.6, "lock",  0.20),            # lock 0.20R at 0.6R
        (0.3, "lock",  0.10),            # lock 0.10R at 0.3R
        # PF=2.05 WR=70.5% DD=5.7% — progressive 1:3 ratio locks
    ],
    "XAGUSD": [
        (6.0, "trail", 0.5),
        (4.0, "trail", 0.7),
        (2.0, "trail", 1.0),
        (1.5, "trail", 1.5),
        (1.0, "lock",  0.33),
        (0.6, "lock",  0.20),
        (0.3, "lock",  0.10),
    ],
}

# ═══ TRAILING SL — Sub2 RUNNER (still wider than Sub0/1) ═══
SUB2_TRAIL_STEPS = [
    (10.0, "trail", 0.3),            # ultra-tight at 10R
    (8.0, "trail", 0.5),             # tight at 8R
    (6.0, "trail", 0.7),             # at 6R
    (4.0, "trail", 1.0),             # moderate at 4R
    (2.0, "trail", 1.5),             # at 2R
    (1.5, "trail", 2.0),             # wider at 1.5R
    (0.8, "lock",  0.15),            # tiny lock at 0.8R
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
    "XAUUSD":   2.5,              # baseline test: PF 1.73→2.00, gives gold room
    "NAS100.r": 1.0,              # grid search: tighter SL, PF 1.50→1.68
    "JPN225ft": 1.0,              # grid search: tighter SL, PF 1.67→2.02
    "USDJPY":   2.0,              # baseline test: PF 1.25→1.80, wider works better
}

# ═══ DASHBOARD ═══
DASHBOARD_PORT = 8888
STARTING_BALANCE = 1000.0

# ═══ SQLITE ═══
DB_PATH = Path(__file__).parent / "data" / "beast.db"

# ═══ DRAGON-SPECIFIC CONSTANTS ═══
DRAGON_MIN_SCORE_BASELINE = 7.0    # minimum score for any swing entry

# Per-symbol regime MIN_SCORE overrides (from grid search optimization)
# Each tested 15-25 combinations, picked highest PF with >= 15 trades
DRAGON_SYMBOL_MIN_SCORE: Dict[str, Dict[str, float]] = {
    "XAUUSD":   {"trending": 6.5, "ranging": 8.5, "volatile": 7.0, "low_vol": 7.0},  # PF 1.26→1.52
    "XAGUSD":   {"trending": 6.0, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0},  # PF 2.34 (optimal=current)
    "BTCUSD":   {"trending": 5.0, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0},  # PF 3.21→3.38
    "NAS100.r": {"trending": 6.0, "ranging": 8.5, "volatile": 7.0, "low_vol": 7.0},  # PF 1.50→1.68
    "JPN225ft": {"trending": 6.0, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0},  # PF 1.67→2.02
    "USDJPY":   {"trending": 6.5, "ranging": 8.5, "volatile": 7.0, "low_vol": 7.0},  # PF 1.42→1.51
}
DRAGON_SCALP_MIN_SCORE = 6.5       # minimum score for scalp entry
DRAGON_CONFIDENCE_FLOOR = 0.52     # ML meta-label minimum probability (was 0.65 — blocked 100% of signals)
DRAGON_MAX_CONSECUTIVE_LOSSES = 3  # blacklist symbol after 3 consecutive losses
DRAGON_BLACKLIST_HOURS = 24        # hours to ban symbol after consecutive losses
DRAGON_EQUITY_SLOPE_WINDOW = 20    # trades to measure equity slope
DRAGON_STANDBY_HOURS = 4           # hours of no favorable conditions before standby
DRAGON_RISK_SCALE_MIN = 0.4        # min risk % (floor for low confidence)
DRAGON_RISK_SCALE_MAX = 1.0        # max risk % (matches MAX_RISK_PER_TRADE_PCT)
DRAGON_LOSS_DAY_RISK_MULT = 0.5    # halve risk after losing day

# ═══ CORRELATION PAIRS ═══
# Won't open simultaneous positions in both symbols if correlation >= threshold
CORRELATION_PAIRS: Dict[Tuple[str, str], float] = {
    ("XAUUSD", "XAGUSD"): 0.85,
    ("NAS100.r", "JPN225ft"): 0.60,
}
