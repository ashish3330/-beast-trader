#!/usr/bin/env python3 -B
"""Canonical TREND-book reconstruction engine (shared by the per-symbol tuners).

Faithful to scripts/_trend_exit_tune_h1.py: a DAILY (D1) 3-EMA-ensemble signal
drives entry/flip; the exit stack (ATR stop + Chandelier trail + profit-lock +
peak-giveback, plus an optional ATR-distance TP) is checked per H1 bar (~12x
finer than D1, close to the live 60s cadence). Returns a structured trade list
with entry/exit prices, pnl_R, and MAE_R so tuners score on one shared engine.

IMPORTANT (see backtest/tune/DATA_AUDIT.md for the full parity write-up): this
engine uses a FIXED atr_stop * ATR stop, exactly like the tuners. LIVE tightens
the stop to cap risk at TREND_MAX_RISK_PCT (gold ~0.2xATR, ~15x tighter) and
scales the lock/giveback ACTIVATION to that capped sl_dist, so live exits far
earlier than this engine. Do NOT read absolute pnl_R here as a live P/L forecast;
use it to RANK exit-parameter variants on identical data.

Usage:
    from backtest.tune.trend_engine import load, run_symbol, summarize
    m = load("JPN225ft")                       # merged D1-signal + H1 frame
    trades = run_symbol("JPN225ft", frame=m)   # uses config per-symbol params
    print(summarize(trades))
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `config` importable no matter where the engine is run/imported from
# (repo root = two levels up from backtest/tune/). Without this the config
# per-symbol EMA/exit params silently fall back to generic defaults.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

# Defaults mirror scripts/_trend_exit_tune_h1.py / config.py TREND_* globals.
DEFAULT_EMA_PAIRS = [(16, 64), (32, 128), (64, 256)]
DEFAULT_MIN_ABS = 0.34
DEFAULT_ATR_P = 20
DEFAULT_ATR_STOP = 3.0
DEFAULT_TRAIL_LOOKBACK = 22
DEFAULT_TP_ATR = 999.0        # 999 == "no TP" (tuner convention)


def _token(sym):
    return sym.replace(".", "_")


def _naive(s):
    s = pd.to_datetime(s, utc=True)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        try:
            return s.dt.tz_convert(None)
        except (TypeError, AttributeError):
            return s


def _cfg_ema_pairs(sym):
    try:
        from config import trend_ema_pairs
        return list(trend_ema_pairs(sym))
    except Exception:
        return DEFAULT_EMA_PAIRS


def _cfg_exit_params(sym):
    """(TRAIL, LOCK, GIVEBACK, ACT) from config per-symbol table, else defaults."""
    try:
        from config import trend_exit_params
        return tuple(trend_exit_params(sym))
    except Exception:
        return (2.5, 0.6, 0.30, 0.3)


def d1_context(sym, ema_pairs=None, min_abs=DEFAULT_MIN_ABS,
               atr_p=DEFAULT_ATR_P, trail_lookback=DEFAULT_TRAIL_LOOKBACK):
    """Build the daily context: effective-from date (prior completed D1 -> next
    day, no look-ahead), signal (-1/0/+1), ATR, and the Chandelier hh/ll windows.
    Matches _trend_exit_tune_h1.d1_context but per-symbol EMA pairs."""
    ema_pairs = ema_pairs or _cfg_ema_pairs(sym)
    d = pickle.load(open(CACHE / ("raw_d1_" + _token(sym) + ".pkl"), "rb"))
    d = d.copy()
    d["time"] = _naive(d["time"])
    d = d.sort_values("time").reset_index(drop=True)
    c, h, l = d["close"], d["high"], d["low"]
    sig = pd.Series(0.0, index=d.index)
    for f, s in ema_pairs:
        sig = sig + np.sign(c.ewm(span=f).mean() - c.ewm(span=s).mean())
    sig = (sig / len(ema_pairs)).apply(
        lambda v: 0 if abs(v) < min_abs else (1 if v > 0 else -1))
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_p, adjust=False).mean()
    out = pd.DataFrame({
        "eff": d["time"].dt.normalize() + pd.Timedelta(days=1),
        "sig": sig.astype(int), "atr": atr,
        "hh": h.rolling(trail_lookback).max(),
        "ll": l.rolling(trail_lookback).min()})
    return out.dropna().reset_index(drop=True)


def load(sym, ema_pairs=None, **d1_kw):
    """Merge the H1 price frame with the daily signal context (merge_asof
    backward on eff). Returns the frame the simulator consumes."""
    h1 = pickle.load(open(CACHE / ("raw_h1_" + _token(sym) + ".pkl"), "rb"))
    h1 = h1.copy()
    h1["time"] = _naive(h1["time"])
    h1 = h1.sort_values("time").reset_index(drop=True)
    d1 = d1_context(sym, ema_pairs=ema_pairs, **d1_kw)
    h1["time"] = h1["time"].astype("datetime64[ns]")
    d1["eff"] = d1["eff"].astype("datetime64[ns]")
    m = pd.merge_asof(h1, d1, left_on="time", right_on="eff",
                      direction="backward").dropna(
        subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)
    return m


def spread_cost_rt(m):
    """Round-trip spread cost as a fraction of price (same as the tuner)."""
    if "spread" not in m.columns or len(m) == 0:
        return 0.0
    px = float(m["close"].iloc[-1])
    if px <= 0:
        return 0.0
    return 2.0 * (float(np.nanmedian(m["spread"].values)) * 0.01) / px


def simulate(m, TR, LK, GB, ACT, ATR_STOP=DEFAULT_ATR_STOP,
             TP=DEFAULT_TP_ATR, cost=0.0):
    """Per-H1-bar exit simulation. Returns a list of trade dicts:
      {dir, t_in, t_out, entry, exit, reason, ret, pnl_R, mae_R, mfe_R}
    ret  = signed fractional price return minus round-trip cost.
    pnl_R = ret expressed in R, where 1R = ATR_STOP*ATR(entry)/entry (the stop
            distance as a fraction of price) — i.e. profit relative to the risk
            actually taken at entry.
    Faithful to _trend_exit_tune_h1.simulate; adds bookkeeping + reason/MAE/MFE."""
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    tm = m["time"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    pos = 0
    entry = sl = tp = peak = 0.0
    e_atr = 0.0            # ATR at entry -> defines 1R
    e_i = 0               # entry bar index
    worst = 0.0           # max adverse excursion (price points, positive)
    best = 0.0            # max favourable excursion (== peak)
    blocked = 0
    trades = []

    def _close(ex_price, reason, t_idx):
        nonlocal pos
        ret = ((ex_price - entry) / entry) * pos - cost
        r_unit = (ATR_STOP * e_atr) / entry if entry > 0 and e_atr > 0 else 0.0
        pnl_R = ret / r_unit if r_unit > 0 else 0.0
        mae_R = (worst / (ATR_STOP * e_atr)) if e_atr > 0 else 0.0
        mfe_R = (best / (ATR_STOP * e_atr)) if e_atr > 0 else 0.0
        trades.append({
            "dir": pos, "t_in": tm[e_i], "t_out": tm[t_idx],
            "entry": float(entry), "exit": float(ex_price), "reason": reason,
            "ret": float(ret), "pnl_R": float(pnl_R),
            "mae_R": float(mae_R), "mfe_R": float(mfe_R)})
        pos = 0

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blocked and s != blocked:
                blocked = 0
            if s != 0 and s != blocked:
                pos = s; entry = o[t]; peak = 0.0; e_atr = a; e_i = t
                worst = 0.0; best = 0.0
                sl = entry - ATR_STOP * a if s == 1 else entry + ATR_STOP * a
                tp = None if TP >= 999 else (
                    (entry + TP * a) if s == 1 else (entry - TP * a))
            continue
        # flip on daily signal reversal (open of the flip bar)
        if s != 0 and s != pos:
            _close(o[t], "flip", t)
            blocked = 0
            pos = s; entry = o[t]; peak = 0.0; e_atr = a; e_i = t
            worst = 0.0; best = 0.0
            sl = entry - ATR_STOP * a if pos == 1 else entry + ATR_STOP * a
            tp = None if TP >= 999 else (
                (entry + TP * a) if pos == 1 else (entry - TP * a))
            continue
        if pos == 1:
            worst = max(worst, entry - l[t])
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else -1e18
            ex = ereason = None
            if l[t] <= sl:
                ex, ereason = sl, "stop/trail"
            elif gb > -1e17 and l[t] <= gb:
                ex, ereason = gb, "giveback"; blocked = pos
            elif tp is not None and h[t] >= tp:
                ex, ereason = tp, "tp"
            if ex is not None:
                _close(ex, ereason, t)
            else:
                peak = max(peak, h[t] - entry); best = peak
        else:
            worst = max(worst, h[t] - entry)
            sl = min(sl, ll[t] + TR * a)
            if peak >= ACT * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else 1e18
            ex = ereason = None
            if h[t] >= sl:
                ex, ereason = sl, "stop/trail"
            elif gb < 1e17 and h[t] >= gb:
                ex, ereason = gb, "giveback"; blocked = pos
            elif tp is not None and l[t] <= tp:
                ex, ereason = tp, "tp"
            if ex is not None:
                _close(ex, ereason, t)
            else:
                peak = max(peak, entry - l[t]); best = peak
    return trades


def run_symbol(sym, frame=None, tr=None, lk=None, gb=None, act=None,
               atr_stop=DEFAULT_ATR_STOP, tp_atr=DEFAULT_TP_ATR,
               ema_pairs=None, cost=None):
    """Convenience: load (if needed) + simulate with config per-symbol exit
    params (override any of tr/lk/gb/act to sweep). Returns the trade list."""
    if frame is None:
        frame = load(sym, ema_pairs=ema_pairs)
    d_tr, d_lk, d_gb, d_act = _cfg_exit_params(sym)
    tr = d_tr if tr is None else tr
    lk = d_lk if lk is None else lk
    gb = d_gb if gb is None else gb
    act = d_act if act is None else act
    cost = spread_cost_rt(frame) if cost is None else cost
    return simulate(frame, tr, lk, gb, act, ATR_STOP=atr_stop, TP=tp_atr, cost=cost)


def summarize(trades):
    """Aggregate metrics from a trade list."""
    if not trades:
        return {"n": 0, "total_R": 0.0, "total_ret": 0.0, "pf": 0.0,
                "win_rate": 0.0, "max_dd_R": 0.0, "avg_mae_R": 0.0}
    R = np.array([t["pnl_R"] for t in trades])
    ret = np.array([t["ret"] for t in trades])
    win = R[R > 0].sum(); loss = -R[R < 0].sum()
    eq = np.cumsum(R)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    return {
        "n": len(trades),
        "total_R": float(R.sum()),
        "total_ret": float(ret.sum()),
        "pf": float(win / loss) if loss > 0 else 99.0,
        "win_rate": float((R > 0).mean()),
        "max_dd_R": dd,
        "avg_mae_R": float(np.mean([t["mae_R"] for t in trades])),
    }


if __name__ == "__main__":
    SYM = "JPN225ft"
    frame = load(SYM)
    trades = run_symbol(SYM, frame=frame)
    s = summarize(trades)
    print("[trend_engine smoke test] %s" % SYM)
    print("  H1 bars merged : %d  (%s -> %s)" % (
        len(frame), str(frame["time"].iloc[0])[:10], str(frame["time"].iloc[-1])[:10]))
    print("  exit params    : TR/LK/GB/ACT = %s" % (_cfg_exit_params(SYM),))
    print("  n_trades       : %d" % s["n"])
    print("  total_R        : %.2f" % s["total_R"])
    print("  PF / WR        : %.2f / %.0f%%" % (s["pf"], 100 * s["win_rate"]))
    print("  max_dd_R       : %.2f   avg_mae_R: %.2f" % (s["max_dd_R"], s["avg_mae_R"]))
    if trades:
        from collections import Counter
        print("  exit reasons   : %s" % dict(Counter(t["reason"] for t in trades)))
