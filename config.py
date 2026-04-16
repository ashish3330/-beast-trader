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

# Per-symbol ML meta-label toggle (based on backtest: ML ON vs OFF)
# ML ON helps: XAGUSD, XAUUSD — filter improves PF significantly
# ML OFF better: JPN225ft, BTCUSD, NAS100.r, USDJPY — filter over-rejects good signals
DRAGON_ML_ENABLED = {
    "XAUUSD":   True,    # PF 1.44 ON vs 1.34 OFF
    "XAGUSD":   True,    # PF 2.39 ON vs 1.76 OFF — big difference
    "BTCUSD":   False,   # PF 3.17 ON vs 3.06 OFF — similar, but more trades OFF
    "NAS100.r": False,   # PF 1.50 ON vs 1.64 OFF
    "JPN225ft": False,   # PF 1.62 ON vs 2.40 OFF — ML over-filters
    "USDJPY":   False,   # PF 1.44 ON vs 1.50 OFF
}

# ═══ DRAGON RISK MANAGEMENT ═══
MAX_RISK_PER_TRADE_PCT = 0.3       # 0.3% equity per trade (was 1.0)
MAX_TOTAL_EXPOSURE_PCT = 1.5       # 1.5% total across all open (was 3.0)
DAILY_LOSS_LIMIT_PCT = 1.0         # stop after 1% daily loss (was 2.0)
MAX_POSITIONS = 3                  # max 3 simultaneous (was 4)
DD_REDUCE_THRESHOLD = 3.0          # halve lot at 3% DD (was 5.0)
DD_PAUSE_THRESHOLD = 5.0           # no new entries at 5% DD (was 10.0)
DD_EMERGENCY_CLOSE = 8.0           # close everything at 8% DD (was 15.0)

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

# ═══ TRAILING SL (moderate profile — swing) ═══
TRAIL_STEPS = [
    # (profit_R, action, atr_multiplier_or_lock_R)
    (6.0, "trail", 0.7),
    (4.0, "trail", 1.0),
    (2.5, "trail", 1.5),
    (1.5, "trail", 2.0),
    (1.0, "lock",  0.5),
    (0.5, "be",    0.0),
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
    # TP = 2R hard target; trailing gets tighter as profit grows
    (2.0, "trail", 0.5),          # trail 0.5x ATR at 2R
    (1.5, "trail", 0.7),          # trail 0.7x ATR at 1.5R
    (1.0, "lock",  0.5),          # lock 0.5R profit at 1R
    (0.5, "be",    0.0),          # break-even at 0.5R
]

# ═══ SESSION FILTER ═══
SESSION_START_UTC = 6              # non-crypto: 06:00 UTC
SESSION_END_UTC = 22               # non-crypto: 22:00 UTC

# ═══ ATR SL ═══
ATR_SL_MULTIPLIER = 1.5           # SL = 1.5x ATR (was 3.0 — KEY FIX for PF)

# ═══ DASHBOARD ═══
DASHBOARD_PORT = 8888
STARTING_BALANCE = 1000.0

# ═══ SQLITE ═══
DB_PATH = Path(__file__).parent / "data" / "beast.db"

# ═══ DRAGON-SPECIFIC CONSTANTS ═══
DRAGON_MIN_SCORE_BASELINE = 7.0    # minimum score for any swing entry
DRAGON_SCALP_MIN_SCORE = 6.5       # minimum score for scalp entry
DRAGON_CONFIDENCE_FLOOR = 0.65     # ML meta-label minimum probability
DRAGON_MAX_CONSECUTIVE_LOSSES = 3  # blacklist symbol after 3 consecutive losses
DRAGON_BLACKLIST_HOURS = 24        # hours to ban symbol after consecutive losses
DRAGON_EQUITY_SLOPE_WINDOW = 20    # trades to measure equity slope
DRAGON_STANDBY_HOURS = 4           # hours of no favorable conditions before standby
DRAGON_RISK_SCALE_MIN = 0.1        # min risk % (scaled by confidence)
DRAGON_RISK_SCALE_MAX = 0.5        # max risk % (scaled by confidence)
DRAGON_LOSS_DAY_RISK_MULT = 0.5    # halve risk after losing day

# ═══ CORRELATION PAIRS ═══
# Won't open simultaneous positions in both symbols if correlation >= threshold
CORRELATION_PAIRS: Dict[Tuple[str, str], float] = {
    ("XAUUSD", "XAGUSD"): 0.85,
    ("NAS100.r", "JPN225ft"): 0.60,
}
