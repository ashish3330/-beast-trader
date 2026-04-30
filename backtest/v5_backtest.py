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

from signals.momentum_scorer import _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES

# ═══ DATA ═══
CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

ALL_SYMBOLS = {
    "XAUUSD":   {"cache": "raw_h1_xauusd.pkl",   "point": 0.01,    "spread": 0.30,  "cat": "Gold"},
    "XAGUSD":   {"cache": "raw_h1_XAGUSD.pkl",   "point": 0.001,   "spread": 0.030, "cat": "Gold"},
    "BTCUSD":   {"cache": "raw_h1_BTCUSD.pkl",   "point": 0.01,    "spread": 30.0,  "cat": "Crypto"},
    "ETHUSD":   {"cache": "raw_h1_ETHUSD.pkl",   "point": 0.01,    "spread": 2.0,   "cat": "Crypto"},
    "NAS100.r": {"cache": "raw_h1_NAS100_r.pkl", "point": 0.01,    "spread": 1.50,  "cat": "Index"},
    "JPN225ft": {"cache": "raw_h1_JPN225ft.pkl", "point": 0.01,    "spread": 10.0,  "cat": "Index"},
    "USDCAD":   {"cache": "raw_h1_USDCAD.pkl",   "point": 0.00001, "spread": 0.00020,"cat": "Forex"},
    # Extended symbols for discovery
    "USDJPY":   {"cache": "raw_h1_USDJPY.pkl",   "point": 0.001,   "spread": 0.015, "cat": "Forex"},
    "EURUSD":   {"cache": "raw_h1_EURUSD.pkl",   "point": 0.00001, "spread": 0.00015,"cat": "Forex"},
    "GBPUSD":   {"cache": "raw_h1_GBPUSD.pkl",   "point": 0.00001, "spread": 0.00020,"cat": "Forex"},
    "GBPJPY":   {"cache": "raw_h1_GBPJPY.pkl",   "point": 0.001,   "spread": 0.025, "cat": "Forex"},
    "EURJPY":   {"cache": "raw_h1_EURJPY.pkl",   "point": 0.001,   "spread": 0.020, "cat": "Forex"},
    "XAGUSD":   {"cache": "raw_h1_XAGUSD.pkl",   "point": 0.001,   "spread": 0.030, "cat": "Gold"},
    "GER40.r":  {"cache": "raw_h1_GER40_r.pkl",  "point": 0.01,    "spread": 2.0,   "cat": "Index"},
    "SP500.r":  {"cache": "raw_h1_SP500_r.pkl",  "point": 0.01,    "spread": 0.50,  "cat": "Index"},
}

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
    "trail": [
        (8.0, 0.3, "trail"), (4.0, 0.5, "trail"), (2.0, 0.8, "trail"),
        (1.5, 0.7, "lock"),  (1.0, 0.4, "lock"),  (0.5, 0.0, "be"),
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
    """BBW + ADX → regime string."""
    if bbw < 0.015:
        return "ranging" if adx <= 20 else "low_vol"
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
        # Check SL hit on this bar
        if direction == 1:  # LONG
            if lows[i] <= sl:
                return sl, i, "SL", peak_r
            cur_price = closes[i]
        else:  # SHORT
            if highs[i] >= sl:
                return sl, i, "SL", peak_r
            cur_price = closes[i]

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
    """Run V5 backtest for a single symbol. Returns results dict."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    meta = ALL_SYMBOLS[symbol]
    spread = meta["spread"]
    point = meta["point"]
    cat = meta["cat"]

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
    trail_steps = TRAIL_OVERRIDE.get(symbol, p["trail"])
    dir_bias = DIR_BIAS.get(symbol, 0)  # 0 = both directions
    risk_cap = RISK_CAP.get(symbol, p["risk_pct"])
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
        long_s, short_s = _score(ind, bi)
        long_s, short_s = float(long_s), float(short_s)
        raw_score = max(long_s, short_s)
        signal_quality = min(100.0, raw_score / p["quality_div"] * 100)

        # Regime
        bbw_val = float(ind["bbw"][bi]) if not np.isnan(ind["bbw"][bi]) else 0.02
        adx_val = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 20
        regime = get_regime(bbw_val, adx_val)
        threshold = min_q.get(regime, 55) if isinstance(min_q, dict) else min_q

        if signal_quality < threshold:
            continue

        # Direction
        if long_s >= short_s:
            direction = 1
            raw = long_s
        else:
            direction = -1
            raw = short_s

        # Direction bias
        if dir_bias != 0 and direction != dir_bias:
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

        # Pullback: check if next bar retraces 0.2 ATR
        atr = float(ind["at"][bi])
        retrace = atr * 0.2
        entry_bar = i + 1
        if entry_bar >= n - 1:
            continue

        # Simple pullback check: does next bar's low (LONG) or high (SHORT) retrace?
        if direction == 1:
            pullback_hit = l[entry_bar] <= c[i] - retrace
            entry_price = c[i] - retrace if pullback_hit else c[i]
        else:
            pullback_hit = h[entry_bar] >= c[i] + retrace
            entry_price = c[i] + retrace if pullback_hit else c[i]

        # Apply spread cost at entry
        entry_price += (spread / 2) * direction

        # SL
        sl_dist = atr * sl_mult
        if sl_dist <= 0:
            continue

        # Conviction sizing
        if signal_quality >= 80:
            conv = p.get("conv_80", 1.5)
        elif signal_quality >= 65:
            conv = p.get("conv_65", 1.2)
        elif signal_quality >= 55:
            conv = p.get("conv_55", 1.0)
        else:
            conv = p.get("conv_low", 0.6)

        risk = min(risk_cap, p["risk_pct"]) * conv
        dollar_risk = equity * (risk / 100.0)
        lot_value = sl_dist / point  # SL in points
        if lot_value <= 0:
            continue

        # Simulate trail
        exit_price, exit_bar, exit_reason, peak_r = simulate_trail(
            entry_price, sl_dist, direction, h, l, c,
            entry_bar + 1, n, spread, trail_steps,
            ratchet_1r=p.get("ratchet_1r", 0.3),
            ratchet_2r=p.get("ratchet_2r", 0.7),
            rl_adj=p.get("rl_adj"))

        # Apply spread at exit
        exit_price -= (spread / 2) * direction

        # PnL
        pnl_points = (exit_price - entry_price) * direction
        pnl_r = pnl_points / sl_dist if sl_dist > 0 else 0
        pnl_dollar = dollar_risk * pnl_r

        equity += pnl_dollar
        peak_eq = max(peak_eq, equity)

        trades.append({
            "entry_bar": entry_bar, "exit_bar": exit_bar,
            "direction": direction, "entry": entry_price, "exit": exit_price,
            "pnl": pnl_dollar, "pnl_r": pnl_r, "peak_r": peak_r,
            "quality": signal_quality, "regime": regime,
            "exit_reason": exit_reason, "risk_pct": risk,
            "hour": pd.Timestamp(times[i]).hour,
            "giveback_r": peak_r - pnl_r if peak_r > 0 else 0,
        })

        # Consecutive loss tracking
        if pnl_dollar < 0:
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

    result = {
        "symbol": symbol, "trades": len(trades), "wins": len(wins),
        "pf": round(pf, 2), "wr": round(wr, 1),
        "pnl": round(sum(t["pnl"] for t in trades), 2),
        "avg_r": round(avg_r, 2), "avg_peak_r": round(avg_peak, 2),
        "avg_giveback": round(avg_giveback if avg_giveback == avg_giveback else 0, 2),
        "dd": round(max_dd, 1), "equity": round(equity, 2),
        "details": trades,
    }

    if verbose:
        print(f"  {symbol:12s} | {len(trades):4d} trades | WR {wr:5.1f}% | PF {pf:5.2f} | "
              f"PnL ${result['pnl']:>8.2f} | DD {max_dd:5.1f}% | avgR {avg_r:+.2f} | "
              f"peak {avg_peak:.1f}R → give {result['avg_giveback']:.1f}R | eq ${equity:.0f}")

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

        total_pnl = 0
        total_trades = 0
        for sym in symbols:
            sym_params = {}
            if args.rl_trail and sym in rl_adj_by_symbol:
                sym_params["rl_adj"] = rl_adj_by_symbol[sym]
            if meta_model is not None:
                sym_params["_meta_model"] = meta_model
            r = backtest_symbol(sym, args.days, sym_params or None)
            if r:
                total_pnl += r["pnl"]
                total_trades += r["trades"]

        print(f"\n  TOTAL: {total_trades} trades | PnL ${total_pnl:.2f}")


if __name__ == "__main__":
    main()
