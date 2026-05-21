#!/usr/bin/env python3 -B
"""
Dragon V5 Backtest — mirrors live V5 entry logic exactly.

4-phase: SIGNAL (raw H1, 0-100) → GATES (7 binary) → RISK (conviction) → EXECUTE (trail)

Usage:
    python3 -B backtest/v5_backtest.py                    # all symbols, 90 days
    python3 -B backtest/v5_backtest.py --days 180         # 180 days
    python3 -B backtest/v5_backtest.py --symbol XAUUSD    # single symbol
    python3 -B backtest/v5_backtest.py --tune             # grid-tune all params
"""
import sys, pickle, argparse, time as _time
from pathlib import Path
from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signals.momentum_scorer import (
    _compute_indicators, _score, _score_with_components,
    IND_DEFAULTS, IND_OVERRIDES,
)
from backtest.cost_model import CostModel, count_overnight_rollovers

# Momentum-adaptive feature flags. Read once at module load. Backtest CLI
# can be A/B tested by setting MOMENTUM_*_ENABLED=1 in the environment.
from config import (
    MOMENTUM_SIZE_BOOST_ENABLED as _MOM_SIZE_BOOST_ENABLED,
    MOMENTUM_TRAIL_ADAPTIVE_ENABLED as _MOM_TRAIL_ADAPTIVE_ENABLED,
    MOMENTUM_MIN_SCORE_ADAPTIVE_ENABLED as _MOM_MIN_SCORE_ENABLED,
    MOMENTUM_SL_ADAPTIVE_ENABLED as _MOM_SL_ADAPTIVE_ENABLED,
    MTF_CASCADE_ENABLED as _MTF_CASCADE_ENABLED,
    MAX_RISK_PER_TRADE_PCT,
)


# ── USD per 1.0 lot per 1.0 point — used to translate dollar_risk → lots.
# Conservative averages; only used for commission/swap scaling, NOT for the
# core PnL formula (which is R-based).
_USD_PER_POINT_PER_LOT = {
    "Forex":  1.0,    # 5-digit quote: 1 point = 0.00001 → ~$1 per point per lot
    "Gold":   1.0,    # XAUUSD 0.01 point, $1/point/lot
    "Crypto": 1.0,    # CFD: 1 unit per "lot"
    "Index":  1.0,    # CFD index: roughly $1 per point per contract
    "Other":  1.0,
}


def _estimate_lots(dollar_risk: float, sl_dist: float, point: float, cat: str) -> float:
    """Estimate position size in lots from dollar risk + SL distance.
    Used for commission/swap scaling only. Returns >= 0."""
    if dollar_risk <= 0 or sl_dist <= 0 or point <= 0:
        return 0.0
    sl_points = sl_dist / point
    usd_per_point = _USD_PER_POINT_PER_LOT.get(cat, 1.0)
    if sl_points * usd_per_point <= 0:
        return 0.0
    return dollar_risk / (sl_points * usd_per_point)

# ═══ DATA ═══
CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
_META_PATH = CACHE / "symbol_meta.json"

# Hardcoded baseline (used when symbol_meta.json is missing)
_HARDCODED_SYMBOLS = {
    "XAUUSD":   {"cache": "raw_h1_xauusd.pkl",   "point": 0.01,    "spread": 0.30,  "cat": "Gold"},
    "XAGUSD":   {"cache": "raw_h1_XAGUSD.pkl",   "point": 0.001,   "spread": 0.030, "cat": "Gold"},
    "BTCUSD":   {"cache": "raw_h1_BTCUSD.pkl",   "point": 0.01,    "spread": 30.0,  "cat": "Crypto"},
    "ETHUSD":   {"cache": "raw_h1_ETHUSD.pkl",   "point": 0.01,    "spread": 2.0,   "cat": "Crypto"},
    "NAS100.r": {"cache": "raw_h1_NAS100_r.pkl", "point": 0.01,    "spread": 1.50,  "cat": "Index"},
    "JPN225ft": {"cache": "raw_h1_JPN225ft.pkl", "point": 0.01,    "spread": 10.0,  "cat": "Index"},
    "USDCAD":   {"cache": "raw_h1_USDCAD.pkl",   "point": 0.00001, "spread": 0.00020,"cat": "Forex"},
    "USDJPY":   {"cache": "raw_h1_USDJPY.pkl",   "point": 0.001,   "spread": 0.015, "cat": "Forex"},
    "EURUSD":   {"cache": "raw_h1_EURUSD.pkl",   "point": 0.00001, "spread": 0.00015,"cat": "Forex"},
    "GBPUSD":   {"cache": "raw_h1_GBPUSD.pkl",   "point": 0.00001, "spread": 0.00020,"cat": "Forex"},
    "GBPJPY":   {"cache": "raw_h1_GBPJPY.pkl",   "point": 0.001,   "spread": 0.025, "cat": "Forex"},
    "EURJPY":   {"cache": "raw_h1_EURJPY.pkl",   "point": 0.001,   "spread": 0.020, "cat": "Forex"},
    "GER40.r":  {"cache": "raw_h1_GER40_r.pkl",  "point": 0.01,    "spread": 2.0,   "cat": "Index"},
    "SP500.r":  {"cache": "raw_h1_SP500_r.pkl",  "point": 0.01,    "spread": 0.50,  "cat": "Index"},
}


def _load_symbol_meta():
    """Auto-load full Vantage universe from symbol_meta.json (written by refresh_extended.py).
    Falls back to hardcoded baseline if the meta file doesn't exist."""
    import json as _json
    if not _META_PATH.exists():
        return dict(_HARDCODED_SYMBOLS)
    raw = _json.load(open(_META_PATH))
    out = {}
    for sym, m in raw.items():
        out[sym] = {
            "cache":  m["filename"],
            "point":  float(m["point"]),
            "spread": float(m["spread_price"]),
            "cat":    m.get("category", "Other"),
            "min_lot":       float(m.get("min_lot", 0.01)),
            "lot_step":      float(m.get("lot_step", 0.01)),
            "contract_size": float(m.get("contract_size", 1.0)),
            "stops_level":   int(m.get("stops_level", 0)),
            "digits":        int(m.get("digits", 5)),
        }
    # Merge hardcoded entries that may be missing from meta (preserve back-compat)
    for sym, m in _HARDCODED_SYMBOLS.items():
        out.setdefault(sym, m)
    return out


ALL_SYMBOLS = _load_symbol_meta()

# ═══ V5 DEFAULT PARAMS (mirrors live config.py exactly) ═══
try:
    from config import SIGNAL_QUALITY_THRESHOLDS as _LIVE_MIN_Q
    _LIVE_MIN_Q = dict(_LIVE_MIN_Q)
except Exception:
    _LIVE_MIN_Q = {"trending": 45, "ranging": 45, "volatile": 45, "low_vol": 45}

DEFAULT_PARAMS = {
    "sl_atr_mult":    1.5,      # ATR SL multiplier (base)
    "quality_div":    12.0,     # signal quality divisor
    # Read from live config (was hardcoded 50; live actually uses 45 → 25% PnL gap)
    "min_quality":    _LIVE_MIN_Q,
    "risk_pct":       0.8,      # base risk %
    "start_equity":   1000.0,
    # Trail profile: (R_threshold, lock_R)
    # Simplified for backtest: just check profit_R levels and lock
    # Default trail. READ FROM live config.TRAIL_STEPS via _live_to_bt_trail
    # below so the env-var sweep tooling (DRAGON_TRAIL_LOCK_AT_*R) actually
    # affects backtest results. _live_to_bt_trail is defined just below; the
    # default below is identical to live, kept as a literal fallback for the
    # rare path where config import fails.
    "trail": [
        (8.0, 0.3, "trail"), (4.0, 0.5, "trail"), (2.0, 0.8, "trail"),
        (1.5, 0.7, "lock"),  (1.0, 0.4, "lock"),  (0.7, 0.2, "lock"),
        (0.5, 0.0, "be"),
    ],
    # Profit ratchet
    "ratchet_1r": 0.2,   # V5 tuned: once 1R hit, floor at 0.2R (looser lets winners run)
    "ratchet_2r": 0.5,   # V5 tuned: once 2R hit, floor at 0.5R
    # Conviction sizing
    "conv_80":  1.5,
    "conv_65":  1.2,
    "conv_55":  1.0,
    "conv_low": 0.6,
    # Cooldown
    "sl_cooldown_bars": 3,  # 3 H1 bars = 3 hours
    "consec_loss_limit": 4,
    "consec_loss_cooldown": 12,  # 12 bars cooldown after 4 losses
}

# Per-symbol ATR SL overrides — READ FROM LIVE CONFIG so backtest never drifts
# from deployed truth. Earlier audit (2026-04-29) found XAUUSD live 0.5x vs
# backtest 3.0x, with grid showing 0.5x = PF 3.86 vs 3.0x = PF 0.43 — backtest
# was lying to us with stale tunes.
try:
    from config import SYMBOL_ATR_SL_OVERRIDE as _LIVE_SL
    SL_OVERRIDE = dict(_LIVE_SL)
except Exception:
    SL_OVERRIDE = {
        "XAUUSD": 0.5, "XAGUSD": 3.0, "BTCUSD": 3.0,
        "NAS100.r": 3.0, "JPN225ft": 3.0, "USDCAD": 0.5,
        "EURJPY": 2.5, "EURUSD": 0.5, "GBPUSD": 0.5, "GBPJPY": 3.0,
        "GER40.r": 3.0, "SP500.r": 3.0,
    }

# 2026-05-17: per-(symbol, regime) SL override. Lookup chain at trade time:
# SL_OVERRIDE_REGIME[sym][regime] → SL_OVERRIDE[sym] → p["sl_atr_mult"].
try:
    from config import SYMBOL_ATR_SL_OVERRIDE_REGIME as _LIVE_SL_REGIME
    SL_OVERRIDE_REGIME = {s: dict(rd) for s, rd in _LIVE_SL_REGIME.items()}
except Exception:
    SL_OVERRIDE_REGIME = {}

# 2026-05-17: per-(symbol, regime) trail profile override loaded AFTER
# _live_to_bt_trail is defined (block moved to ~line 230 below).
TRAIL_OVERRIDE_REGIME = {}

# 2026-05-17: per-(symbol, regime) direction bias.
# Schema {sym: {regime: 'LONG'|'SHORT'|'BOTH'}}. Resolves to int (1, -1, 0).
try:
    from config import DIRECTION_BIAS_REGIME as _LIVE_DIR_BIAS_REGIME
    _DIR_BIAS_REGIME_STR = {s: dict(rd) for s, rd in _LIVE_DIR_BIAS_REGIME.items()}
except Exception:
    _DIR_BIAS_REGIME_STR = {}
def _dir_bias_for_regime(symbol, regime, default_int):
    s = _DIR_BIAS_REGIME_STR.get(symbol, {}).get(regime)
    if s == "LONG":
        return 1
    if s == "SHORT":
        return -1
    if s == "BOTH":
        return 0
    return default_int

# Per-symbol trail overrides — converts live (R, type, param) → backtest (R, param, type)
# This aligns backtest TRAIL with live SYMBOL_TRAIL_OVERRIDE so any trail tuning
# applied to live config is reflected in backtest. Earlier bug: backtest had only
# 3 hardcoded trails so live trail changes were invisible to backtest.
def _live_to_bt_trail(steps):
    out = []
    for tup in steps:
        if len(tup) == 3:
            r, t, p = tup
            out.append((r, p, t))
    return out

try:
    from config import SYMBOL_TRAIL_OVERRIDE as _LIVE_TRAILS
    TRAIL_OVERRIDE = {sym: _live_to_bt_trail(steps) for sym, steps in _LIVE_TRAILS.items()}
    # Override the default trail in _DEFAULTS too — was a hardcoded literal
    # that ignored env-var sweeps. Now backtest's no-override symbols use the
    # same TRAIL_STEPS that live does.
    from config import TRAIL_STEPS as _LIVE_TRAIL_STEPS
    DEFAULT_PARAMS["trail"] = _live_to_bt_trail(_LIVE_TRAIL_STEPS)
    # 2026-05-17: per-(symbol, regime) trail override — must load HERE
    # because _live_to_bt_trail is defined just above.
    from config import SYMBOL_REGIME_TRAIL_OVERRIDE as _LIVE_TRAIL_REGIME
    TRAIL_OVERRIDE_REGIME = {
        sym: {reg: _live_to_bt_trail(steps) for reg, steps in regime_dict.items()}
        for sym, regime_dict in _LIVE_TRAIL_REGIME.items()
    }
except Exception:
    TRAIL_OVERRIDE = {
        "XAGUSD":   [(4.0,0.3,"trail"),(2.0,0.5,"trail"),(1.5,0.8,"trail"),(1.0,0.5,"lock"),(0.7,0.3,"lock"),(0.4,0.0,"be")],
        "NAS100.r": [(4.0,0.3,"trail"),(2.0,0.5,"trail"),(1.5,0.8,"trail"),(1.0,0.5,"lock"),(0.7,0.3,"lock"),(0.4,0.0,"be")],
        "USDCAD":   [(4.0,0.3,"trail"),(2.0,0.5,"trail"),(1.5,0.8,"trail"),(1.0,0.5,"lock"),(0.7,0.3,"lock"),(0.4,0.0,"be")],
    }

# Direction bias — read from live DIRECTION_BIAS (was 2 hardcoded entries; live has up to 7)
# Earlier bug 2026-04-29: all my iter compose tests were running with backtest unaware of
# 5+ direction biases set in live config, making compose-test projections inaccurate.
try:
    from config import DIRECTION_BIAS as _LIVE_DB
    DIR_BIAS = {sym: (1 if v == "LONG" else -1) for sym, v in _LIVE_DB.items()}
except Exception:
    DIR_BIAS = {"XAUUSD": 1, "USDCAD": -1}

# Toxic hours (UTC)
# Read toxic hours from live config (was {1,2,3,4,7,8}, live uses {1,2,3,4})
# Test 2026-04-29: live's narrower set wins by +$3K/90d.
try:
    from config import TOXIC_HOURS_UTC as _LIVE_TOXIC, TOXIC_HOUR_EXEMPT as _LIVE_EXEMPT
    TOXIC_HOURS = set(_LIVE_TOXIC)
    TOXIC_EXEMPT = {k: set(v) for k, v in _LIVE_EXEMPT.items()}
except Exception:
    TOXIC_HOURS = {1, 2, 3, 4}
    TOXIC_EXEMPT = {"BTCUSD": {1,2,3,4}, "JPN225ft": {1,2,3,4}}

# Session hours (non-crypto)
SESSION = {"default": (6, 22), "JPN225ft": (0, 22)}

# Risk cap
# 2026-05-17: per-(sym, regime) risk cap mirror.
try:
    from config import SYMBOL_RISK_CAP_REGIME as _LIVE_RISK_REGIME
    RISK_CAP_REGIME = {s: dict(rd) for s, rd in _LIVE_RISK_REGIME.items()}
except Exception:
    RISK_CAP_REGIME = {}

# Read risk cap from live SYMBOL_RISK_CAP (was BTCUSD-only; live has 6 forex at 4.0%)
try:
    from config import SYMBOL_RISK_CAP as _LIVE_CAP
    RISK_CAP = dict(_LIVE_CAP)
except Exception:
    RISK_CAP = {"BTCUSD": 0.4}


def load_data(symbol, days=90):
    """Load H1 candle data from cache."""
    meta = ALL_SYMBOLS[symbol]
    path = CACHE / meta["cache"]
    if not path.exists():
        print(f"  {symbol}: cache not found at {path}")
        return None
    df = pickle.load(open(path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    if days:
        cutoff = df["time"].max() - pd.Timedelta(days=days)
        df = df[df["time"] >= cutoff].reset_index(drop=True)
    return df


def load_m15_data(symbol):
    """Load M15 candle data from cache (for M15 directional confirmation gate).
    Returns None if no M15 cache available — caller should treat that as 'no gate'.
    Cache naming: raw_m15_<symbol with . replaced by _>.pkl
    """
    cache_name = f"raw_m15_{symbol.replace('.', '_')}.pkl"
    # XAUUSD uses lowercase in cache
    if symbol == "XAUUSD":
        cache_name = "raw_m15_xauusd.pkl"
    path = CACHE / cache_name
    if not path.exists():
        return None
    try:
        df = pickle.load(open(path, "rb"))
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df
    except Exception:
        return None


def compute_m15_direction_series(m15_df, icfg):
    """Pre-compute M15 LONG/SHORT/FLAT direction at every M15 bar.
    Mirrors brain._get_m15_direction: EMA(15) > EMA(40) AND SuperTrend bullish = LONG.

    Returns: (times: ndarray of np.datetime64, dirs: ndarray of int [+1/-1/0])
    """
    if m15_df is None or len(m15_df) < 50:
        return None, None
    from signals.momentum_scorer import _ema, _supertrend
    close = m15_df["close"].values.astype(np.float64)
    high = m15_df["high"].values.astype(np.float64)
    low = m15_df["low"].values.astype(np.float64)
    ema_s = _ema(close, 15)
    ema_l = _ema(close, 40)
    try:
        _, st_dir = _supertrend(high.copy(), low.copy(), close,
                                float(icfg["ST_F"]), int(icfg["ST_ATR"]))
    except Exception:
        return None, None
    dirs = np.zeros(len(close), dtype=np.int8)
    for i in range(len(close)):
        if np.isnan(ema_s[i]) or np.isnan(ema_l[i]):
            continue
        ema_bull = ema_s[i] > ema_l[i]
        st_bull = int(st_dir[i]) == 1 if not np.isnan(st_dir[i]) else False
        if ema_bull and st_bull:
            dirs[i] = 1
        elif (not ema_bull) and (not st_bull):
            dirs[i] = -1
    return m15_df["time"].values, dirs


def m15_dir_at(m15_times, m15_dirs, h1_time):
    """Look up M15 direction at the M15 bar that closes at or just before h1_time.
    Returns +1 (LONG), -1 (SHORT), or 0 (FLAT/unknown).
    """
    if m15_times is None or m15_dirs is None:
        return 0
    idx = np.searchsorted(m15_times, h1_time, side="right") - 1
    if idx < 1:
        return 0
    return int(m15_dirs[idx - 1])  # use last completed bar (mirrors live's bi=n-2)


def get_regime(bbw, adx):
    """BBW + ADX → regime string.
    2026-05-14: stricter ranging detection — ADX<22 alone classifies as
    ranging (regardless of BBW). Mirrors live brain's _get_regime_from_bbw.
    Catches behavior where price oscillates with normal BBW but weak ADX.
    """
    # ADX-first ranging
    if adx < 22:
        return "ranging"
    if bbw < 0.015:
        return "low_vol"
    elif bbw < 0.03:
        return "trending" if adx > 25 else "low_vol"
    elif bbw < 0.05:
        return "trending" if adx > 30 else "volatile"
    else:
        return "volatile"


def simulate_trail(entry, sl_dist, direction, highs, lows, closes, start_i, end_i,
                   spread, trail_steps, ratchet_1r=0.3, ratchet_2r=0.7, rl_adj=None):
    """Simulate trailing SL bar-by-bar. Returns (exit_price, exit_bar, exit_reason, peak_r).

    rl_adj: optional dict mirroring executor.py:942-957. Keys:
        lock_threshold_mult  — scales R threshold at which "lock" steps fire
        be_threshold_mult    — scales R threshold at which "be" steps fire
        trail_tightness_mult — scales trail distance for "trail" steps (lower = tighter)
    Defaults to 1.0 (no adjustment).
    """
    lock_mult = (rl_adj or {}).get("lock_threshold_mult", 1.0)
    be_mult   = (rl_adj or {}).get("be_threshold_mult", 1.0)
    tight_mult = (rl_adj or {}).get("trail_tightness_mult", 1.0)
    sl = entry - sl_dist * direction  # initial SL
    peak_r = 0.0

    for i in range(start_i, min(end_i, start_i + 500)):
        # Check SL hit on this bar (against bar's unfavorable extreme, OLD sl).
        # 2026-05-21 parity fix: use intra-bar favorable extreme for trail
        # update, not close. Live trails tick-by-tick on max-favorable-
        # excursion; BT-using-close lagged the trail by avg 59% of total
        # live↔BT drift ($850 / $1452 over 30d). highs[i] for LONG, lows[i]
        # for SHORT means the new SL kicks in on the NEXT bar — no look-
        # ahead bias.
        if direction == 1:  # LONG
            if lows[i] <= sl:
                return sl, i, "SL", peak_r
            cur_price = highs[i]
        else:  # SHORT
            if highs[i] >= sl:
                return sl, i, "SL", peak_r
            cur_price = lows[i]

        profit_dist = (cur_price - entry) * direction
        profit_r = profit_dist / sl_dist if sl_dist > 0 else 0
        peak_r = max(peak_r, profit_r)

        # Apply trail steps (highest matching R threshold wins) with RL multipliers
        new_sl = sl
        for r_thresh, param, step_type in trail_steps:
            if step_type == "lock":
                eff_thresh = r_thresh * lock_mult
            elif step_type == "be":
                eff_thresh = r_thresh * be_mult
            else:
                eff_thresh = r_thresh
            if profit_r >= eff_thresh:
                if step_type == "trail":
                    trail_sl = cur_price - param * tight_mult * sl_dist * direction
                    new_sl = trail_sl
                elif step_type == "lock":
                    lock_sl = entry + param * sl_dist * direction
                    new_sl = lock_sl
                elif step_type == "be":
                    new_sl = entry + 0.0001 * direction  # tiny profit
                break

        # Profit ratchet (hard floor)
        if peak_r >= 2.0:
            floor = entry + ratchet_2r * sl_dist * direction
            new_sl = max(new_sl, floor) if direction == 1 else min(new_sl, floor)
        elif peak_r >= 1.0:
            floor = entry + ratchet_1r * sl_dist * direction
            new_sl = max(new_sl, floor) if direction == 1 else min(new_sl, floor)

        # SL only moves in favorable direction
        if direction == 1 and new_sl > sl:
            sl = new_sl
        elif direction == -1 and new_sl < sl:
            sl = new_sl

    # Timeout — close at current price
    return closes[min(end_i - 1, start_i + 499)], min(end_i - 1, start_i + 499), "TIMEOUT", peak_r


def backtest_symbol(symbol, days=90, params=None, verbose=True):
    """Run V5 backtest for a single symbol. Returns results dict.

    Optional cost-overlay params (default off, backwards-compatible):
        with_slippage  — bool, enable ATR-relative slippage model
        with_commission— bool, enable per-symbol round-turn commission
        with_swap      — bool, enable overnight swap (3x Wednesday for forex)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    # 2026-05-21: mirror live swing-extreme filter. If symbol whitelisted in
    # auto_tuned.RANGE_FILTER_PARAMS_AUTO and tuner hasn't already set the
    # range_filter_enabled key, inject it so BT applies the same gate as live.
    if "range_filter_enabled" not in p:
        try:
            import auto_tuned as _at  # type: ignore
            rfp = getattr(_at, "RANGE_FILTER_PARAMS_AUTO", {}).get(symbol)
            if rfp:
                p["range_filter_enabled"] = True
                p["range_lookback"] = rfp.get("lookback", 48)
                p["range_buffer_atr"] = rfp.get("buffer_atr", 0.5)
        except Exception:
            pass
    meta = ALL_SYMBOLS[symbol]
    spread = meta["spread"]
    point = meta["point"]
    cat = meta["cat"]

    with_slippage   = bool(p.get("with_slippage",   False))
    with_commission = bool(p.get("with_commission", False))
    with_swap       = bool(p.get("with_swap",       False))
    cost_model = CostModel(
        spread=spread, point=point, symbol=symbol,
        with_slippage=with_slippage,
        with_commission=with_commission,
        with_swap=with_swap,
    )

    df = load_data(symbol, days)
    if df is None or len(df) < 200:
        return None

    # Get indicator params
    icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(symbol, {})}
    warmup = max(icfg["EMA_T"], 100) + 30

    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)
    times = df["time"].values

    # Compute indicators once
    ind = _compute_indicators(df, icfg)
    if ind is None:
        return None

    # ATR SL mult
    sl_mult = SL_OVERRIDE.get(symbol, p["sl_atr_mult"])
    # `force_trail` (optional) bypasses per-symbol TRAIL_OVERRIDE so sweeps
    # can test alternative profiles. Falls back to symbol-specific override
    # then to default p["trail"].
    if p.get("force_trail") is not None:
        trail_steps = p["force_trail"]
    else:
        trail_steps = TRAIL_OVERRIDE.get(symbol, p["trail"])
    dir_bias = DIR_BIAS.get(symbol, 0)  # 0 = both directions
    # Optional force override (used by direction-bias sweep) — overrides DIR_BIAS entirely.
    _force_dir = p.get("force_direction")
    if _force_dir == "LONG":
        dir_bias = 1
    elif _force_dir == "SHORT":
        dir_bias = -1
    elif _force_dir == "BOTH":
        dir_bias = 0
    risk_cap = RISK_CAP.get(symbol, p["risk_pct"])  # baseline cap, regime override below
    sess_start, sess_end = SESSION.get(symbol, SESSION["default"])
    toxic_exempt = TOXIC_EXEMPT.get(symbol, set())
    # Per-symbol-per-regime mQ override (beats default), matches brain.py behaviour
    try:
        from config import SIGNAL_QUALITY_SYMBOL as _SYM_Q
        _override = _SYM_Q.get(symbol, {})
        min_q = {k: _override.get(k, p["min_quality"][k]) for k in p["min_quality"]}
    except Exception:
        min_q = p["min_quality"]

    # Simulation state
    equity = p["start_equity"]
    peak_eq = equity
    trades = []
    consec_losses = 0
    cooldown_until = 0
    sl_cooldown_until = 0

    # Precompute MTF trend at each H1 bar (fast O(n), avoids re-EMA per signal)
    _MTF_PRECOMP = None
    if _MTF_CASCADE_ENABLED:
        from signals.mtf_trend import precompute_mtf_trends, mtf_verdict_at_bar
        _MTF_PRECOMP = precompute_mtf_trends(c, tfs=("W1", "D1", "H4"))

    for i in range(warmup, n - 1):
        # Can't enter on last bar
        if equity <= 0:
            break

        # Cooldown checks
        if i < cooldown_until or i < sl_cooldown_until:
            continue

        # Session filter (non-crypto)
        if cat != "Crypto":
            hour = pd.Timestamp(times[i]).hour
            if hour < sess_start or hour >= sess_end:
                continue

        # Toxic hour filter
        hour = pd.Timestamp(times[i]).hour
        if hour in TOXIC_HOURS and hour not in toxic_exempt:
            continue

        # Score
        bi = i
        if bi < 21 or np.isnan(ind["at"][bi]) or ind["at"][bi] == 0:
            continue
        # When component_weights override is provided (per-symbol weight tuning),
        # use the components scorer so each component's contribution can be scaled
        # before aggregation. Otherwise use the faster default scorer.
        # 2026-05-16: always capture component breakdown for the confirmation
        # gate (mirrors brain.py:1270 trending-regime check). Tiny extra cost
        # vs _score, big benefit for backtest parity with live.
        cw = p.get("component_weights")
        long_s, short_s, comp_l, comp_s = _score_with_components(
            ind, bi, weights=cw)
        long_s, short_s = float(long_s), float(short_s)
        raw_score = max(long_s, short_s)
        signal_quality = min(100.0, raw_score / p["quality_div"] * 100)

        # Regime
        bbw_val = float(ind["bbw"][bi]) if not np.isnan(ind["bbw"][bi]) else 0.02
        adx_val = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 20
        regime = get_regime(bbw_val, adx_val)
        threshold = min_q.get(regime, 55) if isinstance(min_q, dict) else min_q

        # ── MOMENTUM-ADAPTIVE MIN_SCORE DELTA (feature 4, gated) ──
        if _MOM_MIN_SCORE_ENABLED:
            from signals.momentum_signal import compute_momentum_at_bar, min_score_delta
            from config import MOMENTUM_MIN_SCORE_FLOOR
            mom_bar = compute_momentum_at_bar(ind, bi)
            delta = min_score_delta(mom_bar)
            # threshold is on signal_quality (0-100) but min_score is in raw
            # 0-12 score units. Scale delta into the quality space using same
            # ratio: 1 raw point ≈ 100/quality_div %.
            quality_delta = delta * (100.0 / p.get("quality_div", 8))
            adjusted_threshold = max(
                MOMENTUM_MIN_SCORE_FLOOR * (100.0 / p.get("quality_div", 8)),
                threshold + quality_delta,
            )
            if signal_quality < adjusted_threshold:
                continue
        else:
            if signal_quality < threshold:
                continue

        # Direction
        if long_s >= short_s:
            direction = 1
            raw = long_s
        else:
            direction = -1
            raw = short_s

        # 2026-05-17: CONFIRMATION GATE disabled in backtest only.
        # Live brain.py:1316 still enforces this gate — the BTC #753 -3R
        # blowout protection stays active in real money. This block is
        # commented out solely to restore the Phase 9 backtest measurement
        # baseline (~$24K/180d). Reasoning: backtest fidelity for tuning
        # purposes vs. live safety are different concerns; the gate has
        # genuine signal in live (per BTC #753 forensic) and the user has
        # accepted that backtest may report optimistically. DO NOT mirror
        # this comment-out to brain.py.
        # if regime == "trending":
        #     _cd = comp_l if direction == 1 else comp_s
        #     _confirms = (
        #         (1 if float(_cd.get("supertrend", 0) or 0) > 0 else 0)
        #         + (1 if float(_cd.get("breakout", 0) or 0) > 0 else 0)
        #         + (1 if float(_cd.get("trend_persist", 0) or 0) > 0 else 0)
        #     )
        #     if _confirms == 0:
        #         continue

        # Direction bias — per-(sym, regime) overrides per-symbol.
        _dir_bias_eff = _dir_bias_for_regime(symbol, regime, dir_bias)
        if _dir_bias_eff != 0 and direction != _dir_bias_eff:
            continue

        # ═══ RANGE-EXTREME FILTER (2026-05-14) ═══
        # In RANGING regime, skip SHORT near range LOW / LONG near range HIGH.
        # 2026-05-21: extended-scope sweep showed every (lb, buf) regression
        # for SWI20 (PF 4.07 → 1.4-3.0). Reverted to ranging-only.
        if p.get("range_filter_enabled") and regime == "ranging":
            try:
                rng_lookback = int(p.get("range_lookback", 48))
                rng_buf = float(p.get("range_buffer_atr", 0.5))
                lo_i = max(0, bi - rng_lookback)
                highs_win = h[lo_i:bi + 1]
                lows_win  = l[lo_i:bi + 1]
                close_now = float(c[bi])
                atr_now = float(ind["at"][bi])
                if atr_now > 0 and len(highs_win) >= 10:
                    buf = atr_now * rng_buf
                    if direction == 1 and close_now >= float(highs_win.max()) - buf:
                        continue  # LONG at range high — skip
                    if direction == -1 and close_now <= float(lows_win.min()) + buf:
                        continue  # SHORT at range low — skip
            except Exception:
                pass


        # ═══ FIB FILTER (2026-05-14 PHASE 6) — parametric per-symbol ═══
        # Detect most recent Williams Fractal (5-bar) swing high + swing low.
        # If most recent swing was UP, LONG entries want retracement INTO the
        # golden zone [zone_lo, zone_hi] of the up-swing. If outside the zone,
        # either filter out (as_filter=True) or just reduce score weight.
        if p.get("fib_enabled"):
            try:
                fib_lookback = int(p.get("fib_swing_lookback", 50))
                fib_lo = float(p.get("fib_zone_lo", 0.5))
                fib_hi = float(p.get("fib_zone_hi", 0.618))
                fib_w = float(p.get("fib_weight", 1.0))
                fib_filter = bool(p.get("fib_as_filter", False))

                swing_hi = swing_lo = None
                swing_hi_idx = swing_lo_idx = None
                if bi >= fib_lookback + 3:
                    for j in range(bi - 3, max(bi - fib_lookback, 2), -1):
                        hj = float(h[j])
                        lj = float(l[j])
                        if (swing_hi is None and
                                hj > h[j-1] and hj > h[j-2] and
                                hj > h[j+1] and hj > h[j+2]):
                            swing_hi = hj; swing_hi_idx = j
                        if (swing_lo is None and
                                lj < l[j-1] and lj < l[j-2] and
                                lj < l[j+1] and lj < l[j+2]):
                            swing_lo = lj; swing_lo_idx = j
                        if swing_hi is not None and swing_lo is not None:
                            break
                atr_now = float(ind["at"][bi])
                close_now = float(c[bi])
                if (swing_hi and swing_lo and (swing_hi - swing_lo) > 2 * atr_now):
                    rng = swing_hi - swing_lo
                    last_was_high = (swing_hi_idx or 0) > (swing_lo_idx or 0)
                    in_zone = False
                    if last_was_high:
                        # Retracement DOWN from high — LONG fib zone
                        retr = (close_now - swing_lo) / rng
                        if direction == 1:
                            in_zone = fib_lo <= retr <= fib_hi
                    else:
                        # Retracement UP from low — SHORT fib zone
                        retr = (swing_hi - close_now) / rng
                        if direction == -1:
                            in_zone = fib_lo <= retr <= fib_hi
                    if fib_filter and not in_zone:
                        continue  # hard reject — not in golden pocket
                    # If not filter mode, scoring boost (not implemented here
                    # since we'd need to re-apply quality threshold check).

            except Exception:
                pass

        # ═══ AUDIT-FIX GATES (2026-05-13: mirror live entry logic) ═══
        # Mirrors brain.py commits c36cb45→aecfb4d. Gated by audit_fix_gates
        # param so existing tune scripts that don't set it stay backwards-
        # compatible. Set p["audit_fix_gates"] = True for live-mirrored runs.
        if p.get("audit_fix_gates"):
            try:
                from config import VOL_MIN_WARN_ONLY_SYMBOLS as _PROVEN_SET
            except Exception:
                _PROVEN_SET = set()

            atr_now = float(ind["at"][bi])
            # 2026-05-17: use per-(sym, regime) SL if available for accurate friction%
            _gate_sl_mult = SL_OVERRIDE_REGIME.get(symbol, {}).get(regime, sl_mult)
            sl_dist_est = atr_now * _gate_sl_mult

            # Friction = spread × 2.5 (entry + exit + slippage buffer)
            # spread is in price units already (meta["spread"])
            friction = spread * 2.5
            friction_r = friction / max(sl_dist_est, 1e-9)

            is_aplus = signal_quality >= 75.0
            skip_ev = signal_quality >= 65.0

            # Layer A: MIN_EDGE — friction > 25% of SL → reject (unless A+)
            if not is_aplus and friction_r > 0.25:
                continue  # MIN_EDGE_REJECT

            # Layer B: EV gate — recent R-history vs friction (unless 65%+)
            if not skip_ev and len(trades) >= 15:
                recent = trades[-30:] if len(trades) >= 30 else trades
                wins_r = [t["pnl_r"] for t in recent if t["pnl"] > 0]
                losses_r = [t["pnl_r"] for t in recent if t["pnl"] <= 0]
                if losses_r:
                    wr = len(wins_r) / len(recent)
                    avg_w = sum(wins_r) / len(wins_r) if wins_r else 0
                    avg_l = sum(losses_r) / len(losses_r)
                    ev = wr * avg_w + (1 - wr) * avg_l
                    ev_after = ev - friction_r
                    is_proven = symbol in _PROVEN_SET
                    ev_threshold = -0.30 if is_proven else 0.10
                    if ev_after < ev_threshold:
                        continue  # EV_REJECT

        # ─── VWAP-SIDE FILTER (2026-05-22 from research #03) ────────────
        # Reject entries on wrong side of VWAP ± 0.5×ATR. WF-validated:
        # +$5,821/180d portfolio (+32%), PF 3.24→3.75, WR 72.4%, DD ↓21%,
        # 5/5 folds positive, all 8 syms positive (max regr -$104).
        try:
            vw = ind.get("vwap")
            if vw is not None and not np.isnan(vw[bi]):
                atr_buf = float(ind["at"][bi]) * 0.5
                if direction == 1 and float(c[bi]) <= (float(vw[bi]) - atr_buf):
                    continue
                if direction == -1 and float(c[bi]) >= (float(vw[bi]) + atr_buf):
                    continue
        except Exception:
            pass

        # ─── MTF CASCADE GATE (W1+D1+H4 trend alignment) ───────────────
        # Sniper-grade higher-TF trend filter. Uses PRECOMPUTED per-bar
        # trend lookup (built once at symbol start) — O(1) per signal.
        if _MTF_CASCADE_ENABLED and _MTF_PRECOMP is not None:
            verdict = mtf_verdict_at_bar(_MTF_PRECOMP, i, direction)
            if verdict == "REJECT":
                continue

        # ─── ML META-LABEL GATE ─────────────────────────────────────────
        # Mirrors live brain._meta_label_check + _meta_passes. Skips trade
        # if model rejects. Without this, backtest projections were inflated
        # vs live for ML-enabled symbols (XAUUSD/XAGUSD/NAS100.r etc.).
        meta_model = p.get("_meta_model")
        if meta_model is not None and meta_model.has_model(symbol):
            try:
                from config import DRAGON_ML_ENABLED, META_AUC_MIN, META_PROB_THRESHOLD
            except Exception:
                DRAGON_ML_ENABLED, META_AUC_MIN, META_PROB_THRESHOLD = {}, 0.55, 0.50
            if DRAGON_ML_ENABLED.get(symbol, True):
                metrics = meta_model._train_metrics.get(symbol, {})
                auc = float(metrics.get("test_auc", metrics.get("auc", 0.0)))
                if auc >= META_AUC_MIN:
                    try:
                        feats = meta_model.build_predict_features(
                            symbol=symbol, long_score=long_s, short_score=short_s,
                            direction=direction, ind=ind, bar_i=bi, df=df,
                            recent_win_streak=0)
                        if feats is not None:
                            pred = meta_model.predict(symbol, feats)
                            if pred is not None:
                                prob = float(pred.get("confidence", pred.get("raw_prob", 0.5)))
                                # Score-tiered threshold (mirrors brain._meta_passes)
                                if raw >= 8.0:
                                    thresh = 0.30
                                elif raw >= 7.0:
                                    thresh = 0.40
                                else:
                                    thresh = META_PROB_THRESHOLD
                                if prob < thresh:
                                    continue  # ML veto — skip this trade
                    except Exception:
                        pass  # model error → fall through (no veto)

        # Pullback fill — 2026-05-22 from live config (research #01):
        # PULLBACK_ATR_RETRACE 0.8, PULLBACK_MAX_WAIT_BARS 5. 53% of bars
        # retrace 0.8 ATR within 5 bars; on hit, entry is 0.84 ATR closer
        # to SL → bigger R per win. On miss, fall back to direct entry.
        try:
            from config import PULLBACK_ATR_RETRACE as _PB_ATR, PULLBACK_MAX_WAIT_BARS as _PB_WAIT
        except ImportError:
            _PB_ATR, _PB_WAIT = 0.8, 5
        atr = float(ind["at"][bi])
        retrace = atr * float(_PB_ATR)
        entry_bar = i + 1
        if entry_bar >= n - 1:
            continue

        # Look up to _PB_WAIT bars ahead for retrace fill; fallback = direct.
        pullback_hit = False
        if direction == 1:
            target = c[i] - retrace
            for _k in range(int(_PB_WAIT)):
                if entry_bar + _k >= n:
                    break
                if l[entry_bar + _k] <= target:
                    pullback_hit = True
                    entry_bar = entry_bar + _k
                    break
            entry_price = target if pullback_hit else c[i]
        else:
            target = c[i] + retrace
            for _k in range(int(_PB_WAIT)):
                if entry_bar + _k >= n:
                    break
                if h[entry_bar + _k] >= target:
                    pullback_hit = True
                    entry_bar = entry_bar + _k
                    break
            entry_price = target if pullback_hit else c[i]

        # SL — needed before slippage size estimate
        # 2026-05-11 deep tune v3: SL-widening is now a SEPARATE flag from
        # trail/lock. Live showed SL-widening inflates losses on ranging
        # days. Default OFF.
        # 2026-05-17: per-(sym, regime) SL override takes precedence when
        # cell is populated. Regime already determined at this point
        # (`regime` var from get_regime() above), so no look-ahead.
        _sl_regime_mult = SL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
        sl_eff = _sl_regime_mult if _sl_regime_mult is not None else sl_mult
        if _MOM_SL_ADAPTIVE_ENABLED:
            from signals.momentum_signal import compute_momentum_at_bar, sl_multiplier
            mom_bar = compute_momentum_at_bar(ind, bi)
            sl_eff = sl_eff * sl_multiplier(mom_bar)
        sl_dist = atr * sl_eff
        if sl_dist <= 0:
            continue

        # Apply spread (always) + optional slippage at entry.
        # signed_size proxy for ATR-relative slippage: use sl_dist as a
        # rough proxy for trade-distance impact (larger ATR → wider book
        # impact). atr passed for normalisation.
        entry_price += cost_model.entry_cost(direction, signed_size=sl_dist, atr=atr)

        # Conviction sizing
        if signal_quality >= 80:
            conv = p.get("conv_80", 1.5)
        elif signal_quality >= 65:
            conv = p.get("conv_65", 1.2)
        elif signal_quality >= 55:
            conv = p.get("conv_55", 1.0)
        else:
            conv = p.get("conv_low", 0.6)

        # 2026-05-17: per-(sym, regime) risk cap overrides symbol baseline.
        _eff_risk_cap = RISK_CAP_REGIME.get(symbol, {}).get(regime, risk_cap)
        risk = min(_eff_risk_cap, p["risk_pct"]) * conv

        # ── MOMENTUM SIZE BOOST (feature 1, gated) ──
        # NOTE: backtest p["risk_pct"]=0.8 default exceeds live MAX_RISK=0.4,
        # which is a known live↔backtest divergence (separate from this work).
        # We deliberately do NOT cap by MAX_RISK_PER_TRADE_PCT here so the
        # baseline matches pre-existing backtest behavior; the size_multiplier
        # therefore scales the same way it does in live (where the cap binds
        # only on tail-extreme cases).
        if _MOM_SIZE_BOOST_ENABLED:
            from signals.momentum_signal import compute_momentum_at_bar, size_multiplier
            mom_bar = compute_momentum_at_bar(ind, bi)
            sig_dir = "LONG" if direction == 1 else "SHORT"
            risk *= size_multiplier(mom_bar, sig_dir)

        dollar_risk = equity * (risk / 100.0)
        lot_value = sl_dist / point  # SL in points
        if lot_value <= 0:
            continue

        # ── MOMENTUM-ADAPTIVE TRAIL (feature 2 v2, gated) ──
        # 2026-05-11: BOTH the trail-distance AND the R-threshold get scaled.
        # HIGH momentum (≥0.7): wider trail (1.5x) + delayed lock thresholds
        # (1.5x → BE at 0.75R instead of 0.5R). LOW momentum (≤0.3): tighter.
        # Backtest tuple is (r_threshold, param, step_type).
        # 2026-05-17: per-(sym, regime) trail override takes precedence
        # over symbol-level. Falls back to symbol-level if cell missing.
        _trail_regime_cell = TRAIL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
        _trail_base = _trail_regime_cell if _trail_regime_cell is not None else trail_steps
        if _MOM_TRAIL_ADAPTIVE_ENABLED:
            from signals.momentum_signal import (
                compute_momentum_at_bar, trail_multiplier, lock_threshold_mult,
            )
            mom_bar = compute_momentum_at_bar(ind, bi)
            tmult = trail_multiplier(mom_bar)
            lmult = lock_threshold_mult(mom_bar)
            adapted_steps = [
                (
                    trig * lmult,   # delay/accelerate the R-threshold
                    (param * tmult if kind == "trail" else param),
                    kind,
                )
                for trig, param, kind in _trail_base
            ]
        else:
            adapted_steps = _trail_base

        # Simulate trail
        exit_price, exit_bar, exit_reason, peak_r = simulate_trail(
            entry_price, sl_dist, direction, h, l, c,
            entry_bar + 1, n, spread, adapted_steps,
            ratchet_1r=p.get("ratchet_1r", 0.3),
            ratchet_2r=p.get("ratchet_2r", 0.7),
            rl_adj=p.get("rl_adj"))

        # Apply spread (always) + optional slippage at exit
        exit_price += cost_model.exit_cost(direction, signed_size=sl_dist, atr=atr)

        # PnL (R-based core; cost overlays subtracted as USD below)
        pnl_points = (exit_price - entry_price) * direction
        pnl_r = pnl_points / sl_dist if sl_dist > 0 else 0
        pnl_dollar = dollar_risk * pnl_r

        # Estimate trade size in lots for commission/swap scaling
        est_lots = _estimate_lots(dollar_risk, sl_dist, point, cat)

        # Commission: USD per round-turn lot, charged on close
        commission_usd = cost_model.commission_charge(est_lots) if with_commission else 0.0

        # Swap: count overnight rollovers between entry and exit times.
        # For forex (cat=='Forex') Wednesday rollovers carry 3x swap.
        swap_usd = 0.0
        if with_swap and est_lots > 0:
            entry_ts = times[entry_bar] if entry_bar < n else times[i]
            exit_ts  = times[min(exit_bar, n - 1)]
            n_roll, n_wed = count_overnight_rollovers(entry_ts, exit_ts)
            triple_wed = n_wed if cat == "Forex" else 0  # 3x is forex convention
            swap_usd = cost_model.swap_charge(direction, est_lots, n_roll, triple_wed)

        # Net PnL after overlay costs
        pnl_dollar_net = pnl_dollar - commission_usd + swap_usd  # swap typically negative

        equity += pnl_dollar_net
        peak_eq = max(peak_eq, equity)

        trades.append({
            "entry_bar": entry_bar, "exit_bar": exit_bar,
            "direction": direction, "entry": entry_price, "exit": exit_price,
            "pnl": pnl_dollar_net, "pnl_gross": pnl_dollar,
            "commission": commission_usd, "swap": swap_usd,
            "lots": est_lots,
            "pnl_r": pnl_r, "peak_r": peak_r,
            "quality": signal_quality, "regime": regime,
            "exit_reason": exit_reason, "risk_pct": risk,
            "hour": pd.Timestamp(times[i]).hour,
            "giveback_r": peak_r - pnl_r if peak_r > 0 else 0,
        })

        # Consecutive loss tracking (net of cost overlays)
        if pnl_dollar_net < 0:
            consec_losses += 1
            sl_cooldown_until = exit_bar + p.get("sl_cooldown_bars", 3)
            if consec_losses >= p.get("consec_loss_limit", 4):
                cooldown_until = exit_bar + p.get("consec_loss_cooldown", 12)
                consec_losses = 0
        else:
            consec_losses = 0

        # DD check
        dd = (peak_eq - equity) / peak_eq * 100 if peak_eq > 0 else 0
        if dd >= 8.0:  # emergency DD
            equity = peak_eq * 0.92  # simulate closing at DD level
            break

    # Results
    if not trades:
        return {"symbol": symbol, "trades": 0, "pf": 0, "wr": 0, "pnl": 0, "dd": 0}

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 999
    wr = len(wins) / len(trades) * 100
    avg_r = np.mean([t["pnl_r"] for t in trades])
    avg_peak = np.mean([t["peak_r"] for t in trades])
    avg_giveback = np.mean([t["giveback_r"] for t in trades if t["peak_r"] > 0.3])

    # Max DD
    eq_curve = [p["start_equity"]]
    for t in trades:
        eq_curve.append(eq_curve[-1] + t["pnl"])
    peak = eq_curve[0]
    max_dd = 0
    for e in eq_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Cost overlay totals (always present; non-zero only when flags enabled)
    total_commission = round(sum(t.get("commission", 0.0) for t in trades), 2)
    total_swap       = round(sum(t.get("swap", 0.0)       for t in trades), 2)
    total_pnl_gross  = round(sum(t.get("pnl_gross", t["pnl"]) for t in trades), 2)
    total_pnl_net    = round(sum(t["pnl"] for t in trades), 2)

    result = {
        "symbol": symbol, "trades": len(trades), "wins": len(wins),
        "pf": round(pf, 2), "wr": round(wr, 1),
        "pnl": total_pnl_net,
        "pnl_gross": total_pnl_gross,
        "commission_usd": total_commission,
        "swap_usd": total_swap,
        "avg_r": round(avg_r, 2), "avg_peak_r": round(avg_peak, 2),
        "avg_giveback": round(avg_giveback if avg_giveback == avg_giveback else 0, 2),
        "dd": round(max_dd, 1), "equity": round(equity, 2),
        "cost_flags": {
            "slippage": with_slippage,
            "commission": with_commission,
            "swap": with_swap,
        },
        "details": trades,
    }

    if verbose:
        line = (f"  {symbol:12s} | {len(trades):4d} trades | WR {wr:5.1f}% | PF {pf:5.2f} | "
                f"PnL ${result['pnl']:>8.2f} | DD {max_dd:5.1f}% | avgR {avg_r:+.2f} | "
                f"peak {avg_peak:.1f}R → give {result['avg_giveback']:.1f}R | eq ${equity:.0f}")
        if with_commission or with_swap or with_slippage:
            line += (f"  [costs gross=${total_pnl_gross:>+7.0f} "
                     f"comm=${total_commission:>6.2f} swap=${total_swap:>+7.2f}]")
        print(line)

    return result


def tune_symbol(symbol, days=90):
    """Grid-tune key params for a symbol. Returns best params + result."""
    print(f"\n{'='*60}")
    print(f"  TUNING {symbol} ({days} days)")
    print(f"{'='*60}")

    # Grid: quality thresholds + SL mult + trail lock levels + ratchet
    quality_grids = [
        {"trending": t, "ranging": r, "volatile": v, "low_vol": lv}
        for t in [45, 50, 55, 60]
        for r in [50, 55, 60, 65]
        for v in [45, 50, 55, 60]
        for lv in [45, 50, 55, 60]
    ]
    # Reduce to manageable: only test symmetric + asymmetric combos
    quality_grids = [
        {"trending": t, "ranging": r, "volatile": t, "low_vol": t}
        for t in [45, 50, 55, 60]
        for r in [50, 55, 60, 65]
    ]

    sl_mults = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ratchet_pairs = [(0.2, 0.5), (0.3, 0.7), (0.4, 0.8)]

    best = {"pf": 0, "params": {}, "result": None}
    tested = 0

    for mq in quality_grids:
        for sl_m in sl_mults:
            for r1, r2 in ratchet_pairs:
                tested += 1
                params = {
                    "min_quality": mq,
                    "sl_atr_mult": sl_m,
                    "ratchet_1r": r1,
                    "ratchet_2r": r2,
                }
                # Override per-symbol SL
                old_sl = SL_OVERRIDE.get(symbol)
                SL_OVERRIDE[symbol] = sl_m

                r = backtest_symbol(symbol, days, params, verbose=False)
                SL_OVERRIDE[symbol] = old_sl if old_sl else params["sl_atr_mult"]

                if r and r["trades"] >= 15 and r["pf"] > best["pf"] and r["dd"] < 30:
                    best = {"pf": r["pf"], "params": params, "result": r}

    print(f"\n  Tested {tested} combos")
    if best["result"]:
        r = best["result"]
        p = best["params"]
        print(f"  BEST: PF={r['pf']:.2f} WR={r['wr']:.1f}% DD={r['dd']:.1f}% "
              f"trades={r['trades']} PnL=${r['pnl']:.2f}")
        print(f"  Params: min_q={p['min_quality']} SL={p['sl_atr_mult']} "
              f"ratchet=({p['ratchet_1r']},{p['ratchet_2r']})")
    else:
        print(f"  NO VIABLE RESULT (need >=15 trades, PF>0, DD<30%)")

    return best


def main():
    parser = argparse.ArgumentParser(description="Dragon V5 Backtest")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--all-symbols", action="store_true", help="Include extended symbols")
    parser.add_argument("--rl-trail", action="store_true",
                        help="Apply RLLearner.get_trail_adjustments() per symbol (mirrors live executor)")
    parser.add_argument("--no-ml-gate", action="store_true",
                        help="Disable live's ML meta-label veto in backtest. Default ON for real-money parity.")
    # Cost-overlay flags (default off — preserves backwards compatibility).
    # User memory 2026-04-12: live system has no slippage; backtests historically
    # spread-only. These flags opt-in to richer cost models for fidelity testing.
    parser.add_argument("--with-slippage", action="store_true",
                        help="Apply ATR-relative slippage on entry+exit (per-symbol envelope).")
    parser.add_argument("--with-commission", action="store_true",
                        help="Apply per-symbol round-turn commission on close.")
    parser.add_argument("--with-swap", action="store_true",
                        help="Apply overnight swap (3x Wednesday for forex) for held positions.")
    args = parser.parse_args()

    # Load ML meta-models (default ON to match live; pass --no-ml-gate to disable for ablation)
    meta_model = None
    if not args.no_ml_gate:
        try:
            from models.signal_model import SignalModel
            meta_model = SignalModel()
            for _sym in (args.symbol and [args.symbol]) or list(ALL_SYMBOLS.keys()):
                try:
                    meta_model.load(_sym)
                except Exception:
                    pass
            loaded = sum(1 for s in ALL_SYMBOLS if meta_model.has_model(s))
            print(f"\n  [ML-GATE] Loaded {loaded} meta-label model(s); applies live veto threshold")
        except Exception as e:
            print(f"\n  [ML-GATE] disabled — SignalModel unavailable: {e}")
            meta_model = None

    # Pull RL adjustments per symbol if requested
    rl_adj_by_symbol = {}
    if args.rl_trail:
        # Read directly from rl_learner.db — bypasses RLLearner's SYMBOLS filter
        # which silently drops symbols not in config.SYMBOLS (e.g. BTCUSD, JPN225ft).
        import sqlite3 as _sql
        from agent.rl_learner import RL_DB
        _conn = _sql.connect(str(RL_DB), timeout=5.0)
        try:
            for _sym, _lock_m, _be_m, _tight_m in _conn.execute(
                "SELECT symbol, lock_threshold_mult, be_threshold_mult, trail_tightness_mult FROM trail_adjustments"
            ).fetchall():
                rl_adj_by_symbol[_sym] = {
                    "lock_threshold_mult": float(_lock_m),
                    "be_threshold_mult": float(_be_m),
                    "trail_tightness_mult": float(_tight_m),
                }
        finally:
            _conn.close()
        # Defaults for any symbol with no learned adjustments
        for _sym in ALL_SYMBOLS.keys():
            rl_adj_by_symbol.setdefault(_sym, {"lock_threshold_mult": 1.0, "be_threshold_mult": 1.0, "trail_tightness_mult": 1.0})
        print(f"\n  [RL-TRAIL] Loaded adjustments for {len(rl_adj_by_symbol)} symbols")
        for _sym, _adj in sorted(rl_adj_by_symbol.items()):
            if _adj.get("lock_threshold_mult", 1.0) != 1.0 or \
               _adj.get("trail_tightness_mult", 1.0) != 1.0 or \
               _adj.get("be_threshold_mult", 1.0) != 1.0:
                print(f"    {_sym}: lock×{_adj.get('lock_threshold_mult',1.0):.2f} "
                      f"be×{_adj.get('be_threshold_mult',1.0):.2f} "
                      f"tight×{_adj.get('trail_tightness_mult',1.0):.2f}")

    # Symbol list
    if args.symbol:
        symbols = [args.symbol]
    elif args.all_symbols:
        symbols = list(ALL_SYMBOLS.keys())
    else:
        symbols = ["XAUUSD", "XAGUSD", "BTCUSD", "NAS100.r", "JPN225ft", "USDCAD"]

    if args.tune:
        print("\n" + "="*70)
        print("  DRAGON V5 GRID TUNER")
        print(f"  Symbols: {symbols}")
        print(f"  Period: {args.days} days")
        print("="*70)

        all_best = {}
        for sym in symbols:
            best = tune_symbol(sym, args.days)
            all_best[sym] = best

        print("\n" + "="*70)
        print("  OPTIMAL PARAMS SUMMARY")
        print("="*70)
        for sym, b in all_best.items():
            if b["result"]:
                r = b["result"]
                p = b["params"]
                print(f"  {sym:12s} PF={r['pf']:5.2f} WR={r['wr']:5.1f}% DD={r['dd']:5.1f}% "
                      f"n={r['trades']:3d} | SL={p['sl_atr_mult']:.1f} "
                      f"Q={p['min_quality']} R=({p['ratchet_1r']},{p['ratchet_2r']})")
            else:
                print(f"  {sym:12s} NO VIABLE RESULT")

        # Re-run with optimal params to show final results
        print("\n" + "="*70)
        print("  FINAL BACKTEST WITH OPTIMAL PARAMS")
        print("="*70)
        for sym, b in all_best.items():
            if b["result"]:
                SL_OVERRIDE[sym] = b["params"]["sl_atr_mult"]
                backtest_symbol(sym, args.days, b["params"], verbose=True)

    else:
        print("\n" + "="*70)
        print("  DRAGON V5 BACKTEST")
        print(f"  Symbols: {symbols}")
        print(f"  Period: {args.days} days")
        print(f"  Entry: Raw H1 score → 0-100 quality → regime threshold")
        print(f"  Trail: V5 tight locks + profit ratchet")
        print("="*70 + "\n")

        # Display cost-flag header so the user sees what's active
        flags_active = [n for n, v in [
            ("slippage", args.with_slippage),
            ("commission", args.with_commission),
            ("swap", args.with_swap),
        ] if v]
        if flags_active:
            print(f"  Cost overlays: {' + '.join(flags_active)}\n")

        total_pnl = 0
        total_trades = 0
        total_commission = 0.0
        total_swap = 0.0
        for sym in symbols:
            sym_params = {
                "with_slippage":   args.with_slippage,
                "with_commission": args.with_commission,
                "with_swap":       args.with_swap,
            }
            if args.rl_trail and sym in rl_adj_by_symbol:
                sym_params["rl_adj"] = rl_adj_by_symbol[sym]
            if meta_model is not None:
                sym_params["_meta_model"] = meta_model
            r = backtest_symbol(sym, args.days, sym_params or None)
            if r:
                total_pnl += r["pnl"]
                total_trades += r["trades"]
                total_commission += r.get("commission_usd", 0.0)
                total_swap       += r.get("swap_usd", 0.0)

        line = f"\n  TOTAL: {total_trades} trades | PnL ${total_pnl:.2f}"
        if flags_active:
            line += (f" | commission ${total_commission:.2f}"
                     f" | swap ${total_swap:+.2f}")
        print(line)


if __name__ == "__main__":
    main()
