"""
Beast Trader — Tick-Level ML Trading Agent.
Account: 25035146, $1,000, VantageInternational-Demo.
MT5 bridge: localhost:18813 (rpyc via Wine).
"""
import os
from dataclasses import dataclass, field
from typing import Dict
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


# ═══ 4 SYMBOLS ═══
SYMBOLS: Dict[str, SymbolConfig] = {
    "XAUUSD":   SymbolConfig("XAUUSD",   8100, "Gold",   2),
    "BTCUSD":   SymbolConfig("BTCUSD",   8110, "Crypto", 2),
    "NAS100.r": SymbolConfig("NAS100.r", 8120, "Index",  2),
    "GER40.r":  SymbolConfig("GER40.r",  8130, "Index",  2),
}

# ═══ RISK MANAGEMENT ═══
MAX_RISK_PER_TRADE_PCT = 1.0       # 1% equity per trade
MAX_TOTAL_EXPOSURE_PCT = 3.0       # 3% total across all open
DAILY_LOSS_LIMIT_PCT = 2.0         # stop after 2% daily loss
MAX_POSITIONS = 4                  # one per symbol
DD_REDUCE_THRESHOLD = 5.0          # halve lot at 5% DD
DD_PAUSE_THRESHOLD = 10.0          # no new entries at 10% DD
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
SCALP_RISK_PCT = 0.5              # 0.5% equity per scalp trade
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
ATR_SL_MULTIPLIER = 3.0           # SL = 3x ATR minimum

# ═══ DASHBOARD ═══
DASHBOARD_PORT = 8888
STARTING_BALANCE = 1000.0

# ═══ SQLITE ═══
DB_PATH = Path(__file__).parent / "data" / "beast.db"
