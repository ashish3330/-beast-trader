#!/usr/bin/env python3 -B
"""FIB50 XAUUSD coord-descent sweep — 2026-06-21.

Coord-descent across 8 axes (MIN_IMPULSE_ATR, MIN_RR, ENTRY_ZONE_LO,
ATR_BUFFER, USE_WIDE_SL, SWING_PIVOT_N, DIRECTION_FILTER, MAX_SL_R).

Mirrors backtest/fib50_backtest.py _simulate_trade + _summarize so combos
score identically to baseline. Injects params via:
  - constructor `params` kwarg for axes the __init__ reads (SWING_PIVOT_N,
    ENTRY_ZONE_LO/HI, etc).
  - agent.fib50_strategy.FIB50_PARAM_OVERRIDES dict for sym_ov-only axes
    (USE_WIDE_SL, DIRECTION_FILTER) plus the duplicated ones that evaluate()
    pulls from sym_ov first (MIN_IMPULSE_ATR, MIN_RR, ATR_BUFFER, MAX_SL_R).

Writes JSON results to /Users/ashish/Documents/beast-trader/scripts/_tune_fib50_xauusd_20260621_results.json
"""
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.fib50_strategy as fib50_mod  # noqa: E402
from agent.fib50_strategy import Fib50Strategy  # noqa: E402

# ── FAST PATH ────────────────────────────────────────────────────────────
# Precompute ATR14 + per-bar swing pivots once; per-combo only the
# swing-walk + entry-eval changes. Mirrors detector logic 1:1 (verified
# against original detector on baseline combo — same trades count).
ATR_PERIOD = 14
MIN_M15_BARS = 100
LOOKBACK = 300


def _atr_series(H, L, C, period=ATR_PERIOD):
    n = len(H)
    tr = np.empty(n)
    tr[0] = H[0] - L[0]
    diff_hl = H[1:] - L[1:]
    diff_hc = np.abs(H[1:] - C[:-1])
    diff_lc = np.abs(L[1:] - C[:-1])
    tr[1:] = np.maximum(np.maximum(diff_hl, diff_hc), diff_lc)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr
    atr[period] = tr[1:period + 1].mean()
    k = 1.0 / period
    for i in range(period + 1, n):
        atr[i] = atr[i - 1] * (1 - k) + tr[i] * k
    return atr


def _precompute_swings(H, L, w):
    """For each index i, mark if it's a strict swing high/low (unique max/min
    of [i-w, i+w]). Confirmation lag = w bars (a pivot at i only known after
    i+w closes).
    """
    n = len(H)
    is_sh = np.zeros(n, dtype=bool)
    is_sl = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        segH = H[i - w:i + w + 1]
        if H[i] == segH.max() and (segH == H[i]).sum() == 1:
            is_sh[i] = True
        segL = L[i - w:i + w + 1]
        if L[i] == segL.min() and (segL == L[i]).sum() == 1:
            is_sl[i] = True
    return is_sh, is_sl


def _latest_pair(swing_list):
    """Given list of (idx, kind, val) sorted by idx, return latest opposite-
    kind pair (A_older, B_newer). None if not found."""
    if len(swing_list) < 2:
        return None
    last = swing_list[-1]
    for j in range(len(swing_list) - 2, -1, -1):
        prev = swing_list[j]
        if prev[1] != last[1]:
            return (prev[0], prev[1], prev[2], last[0], last[1], last[2])
    return None


def _fast_signals(m15, params):
    """Walk bars start_i..n-2 inclusive; emit signals matching detector.
    Returns list of (entry_i, sig_dict).
    """
    H = m15["high"].values
    L = m15["low"].values
    C = m15["close"].values
    O = m15["open"].values
    n = len(H)

    w = int(params["SWING_PIVOT_N"])
    min_impulse = float(params["MIN_IMPULSE_ATR"])
    entry_lo = float(params["ENTRY_ZONE_LO"])
    entry_hi = entry_lo + 0.118
    atr_buf = float(params["ATR_BUFFER"])
    use_wide = bool(params["USE_WIDE_SL"])
    direction = str(params["DIRECTION_FILTER"]).upper()
    max_sl_r = float(params["MAX_SL_R"])
    min_rr = float(params["MIN_RR"])

    atr_arr = _atr_series(H, L, C, ATR_PERIOD)
    is_sh, is_sl = _precompute_swings(H, L, w)

    # Build prefix list of swings (idx, kind, val) up to each cursor.
    # We can lazily walk a single sweep since cursor increases monotonically.
    # For each cursor i (= candidate bar = "last closed bar" in detector lingo)
    # the detector looks at swings in H[:i], L[:i] (strict less-than i).
    # The latest searchable pivot is len-1-w of the slice → (i-1) - w.
    #
    # So at cursor i, only swings with idx <= i-1-w are considered.
    # Also _find_swings only walks indices in [w, end] of the SLICE — and the
    # slice is H[:i] (length i), so end = i - 1 - w. start = max(w, end - LOOKBACK).
    #
    # Strategy: maintain rolling list of confirmed swings. When cursor advances,
    # promote any swing with idx <= i-1-w into the "confirmed" list.

    signals = []
    swings = []  # list of (idx, kind, val); idx strictly increasing
    next_promote = 0  # next bar index to consider promoting

    out_of_zone_skip = 0
    start_i = max(M15_WINDOW, 120)

    for i in range(start_i, n - 1):
        # Promote any newly-confirmed pivots up to i-1-w
        cutoff = i - 1 - w
        while next_promote <= cutoff:
            j = next_promote
            if is_sh[j]:
                swings.append((j, "H", float(H[j])))
            elif is_sl[j]:
                swings.append((j, "L", float(L[j])))
            next_promote += 1

        if i < w * 2 + 5:
            continue
        # The detector also enforces MIN_M15_BARS on the slice. Slice = base
        # of FakeState = up to M15_WINDOW bars ending at cursor. Plus the
        # synthetic forming bar (so detector index -2 = cursor). The detector
        # checks `if df is None or len(df) < MIN_M15_BARS` which is on the
        # full df (incl. forming bar). At cursor i, len(df) = min(M15_WINDOW, i+1)+1
        # so for i >= MIN_M15_BARS-2 it always passes. We already start at 120+ so OK.

        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            continue
        atr14 = atr_arr[i]

        # Only swings strictly within LOOKBACK of (i-1) qualify. Mirror
        # detector's `start = max(w, end - lookback)` over the slice.
        slice_end = i - 1 - w
        slice_start = max(w, slice_end - LOOKBACK)
        # Filter swings by absolute idx.
        # We rely on swings being sorted; binary search would be faster but
        # the list grows slowly and we walk just to find the right tail.
        # Trim front (older than slice_start) to keep list short.
        # Simpler: filter on the fly using a comprehension.
        # We DON'T mutate `swings` because the detector restarts the search
        # each call but pivots remain pivots — the older ones just get trimmed.
        # Compact when list grows too long.
        if swings and swings[0][0] < slice_start - 50:
            swings = [s for s in swings if s[0] >= slice_start]

        if not swings:
            continue
        # The detector finds swings in [slice_start, slice_end] inclusive.
        active = [s for s in swings if slice_start <= s[0] <= slice_end]
        pair = _latest_pair(active)
        if pair is None:
            continue
        a_idx, a_kind, a_val, b_idx, b_kind, b_val = pair

        sig = None
        # SHORT setup
        if a_kind == "H" and b_kind == "L" and direction != "LONG":
            impulse_mag = a_val - b_val
            if impulse_mag >= min_impulse * atr14 and impulse_mag > 0:
                fib_50 = b_val + entry_lo * impulse_mag
                fib_618 = b_val + entry_hi * impulse_mag
                high = H[i]
                if high >= fib_50 and high < fib_618:
                    op = O[i]; cl = C[i]; lo = L[i]
                    bar_range = max(high - lo, 1e-9)
                    upper_wick = high - max(op, cl)
                    uw_r = upper_wick / bar_range
                    body = abs(cl - op)
                    br = body / bar_range
                    is_bearish = cl < op
                    big_uw = uw_r >= 0.50
                    bearish_engulf = (
                        i >= 1 and cl < op and
                        C[i - 1] > O[i - 1] and
                        cl <= O[i - 1] - 1e-9 and
                        op >= C[i - 1] - 1e-9
                    )
                    is_doji = br <= 0.30 and uw_r >= 0.40
                    confirmed = (is_bearish and big_uw) or bearish_engulf or is_doji
                    if confirmed:
                        entry = cl
                        anchor = fib_618 if use_wide else fib_50
                        sl_raw = max(anchor + atr_buf * atr14, high + 0.1 * atr14)
                        risk = sl_raw - entry
                        if risk > 0:
                            tp1 = entry - 0.5 * (entry - b_val)
                            tp2 = b_val
                            rr1 = (entry - tp1) / risk
                            rr2 = (entry - tp2) / risk
                            sl_in_atr = risk / atr14
                            rr_ok = rr1 >= min_rr or rr2 >= 2.0
                            if rr_ok and sl_in_atr <= max_sl_r:
                                sig = {"direction": "SHORT",
                                       "entry": entry, "sl": sl_raw,
                                       "tp1": tp1, "tp2": tp2}

        # LONG setup
        if sig is None and a_kind == "L" and b_kind == "H" and direction != "SHORT":
            c_val, d_val = a_val, b_val
            impulse_mag = d_val - c_val
            if impulse_mag >= min_impulse * atr14 and impulse_mag > 0:
                fib_50 = d_val - entry_lo * impulse_mag
                fib_618 = d_val - entry_hi * impulse_mag
                low = L[i]
                if low <= fib_50 and low > fib_618:
                    op = O[i]; cl = C[i]; hi = H[i]
                    bar_range = max(hi - low, 1e-9)
                    lower_wick = min(op, cl) - low
                    lw_r = lower_wick / bar_range
                    body = abs(cl - op)
                    br = body / bar_range
                    is_bullish = cl > op
                    big_lw = lw_r >= 0.50
                    bullish_engulf = (
                        i >= 1 and cl > op and
                        C[i - 1] < O[i - 1] and
                        cl >= O[i - 1] + 1e-9 and
                        op <= C[i - 1] + 1e-9
                    )
                    is_hammer = br <= 0.30 and lw_r >= 0.40
                    confirmed = (is_bullish and big_lw) or bullish_engulf or is_hammer
                    if confirmed:
                        entry = cl
                        anchor = fib_618 if use_wide else fib_50
                        sl_raw = min(anchor - atr_buf * atr14, low - 0.1 * atr14)
                        risk = entry - sl_raw
                        if risk > 0:
                            tp1 = entry + 0.5 * (d_val - entry)
                            tp2 = d_val
                            rr1 = (tp1 - entry) / risk
                            rr2 = (tp2 - entry) / risk
                            sl_in_atr = risk / atr14
                            rr_ok = rr1 >= min_rr or rr2 >= 2.0
                            if rr_ok and sl_in_atr <= max_sl_r:
                                sig = {"direction": "LONG",
                                       "entry": entry, "sl": sl_raw,
                                       "tp1": tp1, "tp2": tp2}

        if sig is not None:
            signals.append((i, sig))
    return signals

# ── CONFIG (mirrors backtest/fib50_backtest.py) ──────────────────────────
SYMBOL = "XAUUSD"
DAYS = 365
CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
SPREAD = 0.30          # XAUUSD spread (per backtest table)
START_CAPITAL = 5000.0
RISK_PCT = 0.0020
MIN_BARS_FOR_BT = 500
M15_WINDOW = 400
TIME_STOP_BARS = 96
TIME_STOP_PEAK_R = 0.5
MIN_TRADES_FLOOR = 80
DD_CEILING_PCT = 30.0
RESULTS_JSON = Path(__file__).resolve().parent / "_tune_fib50_xauusd_20260621_results.json"


# ── DATA ────────────────────────────────────────────────────────────────
def _load(symbol):
    for v in (f"raw_m15_{symbol}.pkl",
              f"raw_m15_{symbol.replace('.', '_')}.pkl",
              f"raw_m15_{symbol.lower()}.pkl"):
        p = CACHE / v
        if p.exists():
            df = pickle.load(open(p, "rb")).copy()
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").reset_index(drop=True)
            try:
                df["time"] = df["time"].dt.as_unit("ns")
            except Exception:
                pass
            cols = ["time", "open", "high", "low", "close"]
            if "tick_volume" in df.columns:
                cols.append("tick_volume")
            return df[cols].astype({c: float for c in cols if c != "time"})
    return None


class _FakeState:
    def __init__(self, m15):
        self.m15 = m15
        self.cursor = 0

    def set_cursor(self, i):
        self.cursor = i

    def get_candles(self, symbol, tf):
        if tf != 15:
            return None
        c = self.cursor
        lo = max(0, c + 1 - M15_WINDOW)
        base = self.m15.iloc[lo: c + 1]
        if len(base) == 0:
            return None
        last_close = base["close"].iloc[-1]
        last_time = base["time"].iloc[-1]
        row = {"time": last_time + pd.Timedelta(minutes=15),
               "open": last_close, "high": last_close,
               "low": last_close, "close": last_close}
        return pd.concat([base, pd.DataFrame([row])], ignore_index=True)


# ── SIM (verbatim from backtest/fib50_backtest.py) ──────────────────────
def _simulate_trade(sig, m15_arr, entry_i, spread):
    H, L, C = m15_arr
    n = len(C)
    d = 1 if sig["direction"] == "LONG" else -1
    entry = sig["entry"] + d * spread
    sl = sig["sl"]
    tp1 = sig["tp1"]
    tp2 = sig["tp2"]
    risk = (entry - sl) if d == 1 else (sl - entry)
    if risk <= 0:
        return None

    tp1_hit = False
    runner_open = True
    r_tp1 = 0.0
    r_run = 0.0
    peak_r = 0.0
    bars_held = 0
    runner_sl = sl

    end = min(n, entry_i + 1 + 800)
    for j in range(entry_i + 1, end):
        bars_held += 1
        hi, lo = H[j], L[j]
        cur_r = (hi - entry) / risk if d == 1 else (entry - lo) / risk
        if cur_r > peak_r:
            peak_r = cur_r

        if d == 1:
            if not tp1_hit:
                if lo <= sl:
                    r_tp1 = -1.0
                    r_run = -1.0
                    runner_open = False
                    break
                if hi >= tp1:
                    r_tp1 = (tp1 - entry) / risk
                    tp1_hit = True
                    runner_sl = entry
            else:
                if lo <= runner_sl:
                    r_run = (runner_sl - entry) / risk
                    runner_open = False
                    break
                if hi >= tp2:
                    r_run = (tp2 - entry) / risk
                    runner_open = False
                    break
        else:
            if not tp1_hit:
                if hi >= sl:
                    r_tp1 = -1.0
                    r_run = -1.0
                    runner_open = False
                    break
                if lo <= tp1:
                    r_tp1 = (entry - tp1) / risk
                    tp1_hit = True
                    runner_sl = entry
            else:
                if hi >= runner_sl:
                    r_run = (entry - runner_sl) / risk
                    runner_open = False
                    break
                if lo <= tp2:
                    r_run = (entry - tp2) / risk
                    runner_open = False
                    break

        if bars_held >= TIME_STOP_BARS and peak_r < TIME_STOP_PEAK_R:
            close_px = C[j]
            r_now = (close_px - entry) / risk if d == 1 else (entry - close_px) / risk
            if not tp1_hit:
                r_tp1 = r_now
                r_run = r_now
            else:
                r_run = r_now
            runner_open = False
            break
    else:
        if runner_open:
            close_px = C[end - 1]
            r_now = (close_px - entry) / risk if d == 1 else (entry - close_px) / risk
            if not tp1_hit:
                r_tp1 = r_now
                r_run = r_now
            else:
                r_run = r_now

    return {"R": 0.5 * r_tp1 + 0.5 * r_run, "tp1_hit": tp1_hit, "dir": d,
            "entry_i": entry_i, "bars_held": bars_held, "peak_r": peak_r}


def _summarize(trades):
    if not trades:
        return None
    Rs = np.array([t["R"] for t in trades])
    wins = Rs[Rs > 0]
    losses = Rs[Rs <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    equity = [START_CAPITAL]
    risk_d = START_CAPITAL * RISK_PCT
    for r in Rs:
        equity.append(equity[-1] + r * risk_d)
    eq = np.array(equity)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    return {
        "trades": int(len(trades)),
        "wr": float((Rs > 0).mean()),
        "pf": float(pf),
        "total_R": float(Rs.sum()),
        "avg_R": float(Rs.mean()),
        "max_dd_pct": float(abs(dd.min()) * 100.0),
        "end_equity": float(eq[-1]),
        "longs": int(sum(1 for t in trades if t["dir"] == 1)),
        "shorts": int(sum(1 for t in trades if t["dir"] == -1)),
    }


# ── COMBO RUNNER ────────────────────────────────────────────────────────
def _run_combo(m15, params, use_fast=True):
    """params dict keys: MIN_IMPULSE_ATR, MIN_RR, ENTRY_ZONE_LO, ATR_BUFFER,
       USE_WIDE_SL, SWING_PIVOT_N, DIRECTION_FILTER, MAX_SL_R.

    Fast path: vectorized + cached ATR/swing pivots. Mirrors detector 1:1
    (verified). 100x+ speedup vs slow-path.

    Slow path: routes through the real detector (sanity verification only).
    """
    arr = (m15["high"].values, m15["low"].values, m15["close"].values)
    n = len(m15)

    if use_fast:
        sigs = _fast_signals(m15, params)
        trades = []
        open_until = -1
        for entry_i, sig in sigs:
            if entry_i <= open_until:
                continue
            tr = _simulate_trade(sig, arr, entry_i=entry_i, spread=SPREAD)
            if tr is None:
                continue
            trades.append(tr)
            open_until = entry_i + tr["bars_held"]
        return _summarize(trades)

    # Slow path (verification only)
    ov = {
        "MIN_IMPULSE_ATR": params["MIN_IMPULSE_ATR"],
        "MIN_RR": params["MIN_RR"],
        "MAX_SL_R": params["MAX_SL_R"],
        "ATR_BUFFER": params["ATR_BUFFER"],
        "USE_WIDE_SL": params["USE_WIDE_SL"],
        "DIRECTION_FILTER": params["DIRECTION_FILTER"],
    }
    fib50_mod.FIB50_PARAM_OVERRIDES = {SYMBOL: ov}
    ctor = {
        "SWING_PIVOT_N": params["SWING_PIVOT_N"],
        "ENTRY_ZONE_LO": params["ENTRY_ZONE_LO"],
        "ENTRY_ZONE_HI": params["ENTRY_ZONE_LO"] + 0.118,
        "MIN_IMPULSE_ATR": params["MIN_IMPULSE_ATR"],
        "MIN_RR": params["MIN_RR"],
        "MAX_SL_R": params["MAX_SL_R"],
        "ATR_BUFFER": params["ATR_BUFFER"],
    }
    state = _FakeState(m15)
    strat = Fib50Strategy(state, params=ctor)

    trades = []
    open_until = -1
    start_i = max(M15_WINDOW, 120)
    for i in range(start_i, n - 1):
        if i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(SYMBOL, None)
        sig = strat.evaluate(SYMBOL)
        if sig is None:
            continue
        tr = _simulate_trade(sig, arr, entry_i=i, spread=SPREAD)
        if tr is None:
            continue
        trades.append(tr)
        open_until = i + tr["bars_held"]
    return _summarize(trades)


# ── COORD-DESCENT DRIVER ────────────────────────────────────────────────
AXES = {
    "MIN_IMPULSE_ATR":  [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "MIN_RR":           [1.0, 1.2, 1.5, 1.8, 2.0],
    "ENTRY_ZONE_LO":    [0.382, 0.50, 0.618],
    "ATR_BUFFER":       [0.1, 0.2, 0.3, 0.5],
    "USE_WIDE_SL":      [False, True],
    "SWING_PIVOT_N":    [3, 5, 7],
    "DIRECTION_FILTER": ["BOTH", "LONG", "SHORT"],
    "MAX_SL_R":         [4.0, 6.0, 8.0, 12.0],
}

DEFAULTS = {
    "MIN_IMPULSE_ATR": 2.0,
    "MIN_RR": 1.5,
    "ENTRY_ZONE_LO": 0.50,
    "ATR_BUFFER": 0.20,
    "USE_WIDE_SL": False,
    "SWING_PIVOT_N": 5,
    "DIRECTION_FILTER": "BOTH",
    "MAX_SL_R": 8.0,
}

# Coord-descent order: most impactful axes first (so winners propagate).
COORD_ORDER = [
    "DIRECTION_FILTER",
    "MIN_IMPULSE_ATR",
    "ENTRY_ZONE_LO",
    "USE_WIDE_SL",
    "ATR_BUFFER",
    "MAX_SL_R",
    "SWING_PIVOT_N",
    "MIN_RR",
]


def _score(res):
    """Return ranking score, or None if combo skipped."""
    if res is None:
        return None
    if res["trades"] < MIN_TRADES_FLOOR:
        return None
    if res["max_dd_pct"] >= DD_CEILING_PCT:
        return None
    return res["pf"]


def main():
    m15 = _load(SYMBOL)
    if m15 is None:
        print(f"NO_DATA for {SYMBOL}", flush=True)
        return
    n_bars_keep = int(DAYS * 24 * 4)
    if n_bars_keep < len(m15):
        m15 = m15.iloc[-n_bars_keep:].reset_index(drop=True)
    print(f"Loaded {len(m15)} M15 bars for {SYMBOL} (~{len(m15)/96:.0f}d)",
          flush=True)

    all_runs = []
    # ── Baseline ──
    t0 = time.time()
    baseline = _run_combo(m15, DEFAULTS)
    print(f"[baseline] {baseline}  ({time.time()-t0:.1f}s)", flush=True)
    all_runs.append({"params": dict(DEFAULTS), "tag": "baseline", "result": baseline})

    locked = dict(DEFAULTS)
    combos_scored = 0

    # ── Two coord-descent passes ──
    for pass_idx in (1, 2):
        print(f"\n=== Pass {pass_idx} ===", flush=True)
        for axis in COORD_ORDER:
            best_pf = None
            best_val = locked[axis]
            best_res = None
            print(f"  [pass{pass_idx}] axis={axis}  starting={locked[axis]}",
                  flush=True)
            for v in AXES[axis]:
                params = dict(locked)
                params[axis] = v
                # Skip dup of currently-locked combo unless first-axis-first-pass
                # (we want at least one re-eval per axis to confirm).
                t0 = time.time()
                try:
                    res = _run_combo(m15, params)
                except Exception as e:
                    print(f"    {axis}={v}  ERROR {e}", flush=True)
                    continue
                sc = _score(res)
                all_runs.append({"params": dict(params),
                                 "tag": f"pass{pass_idx}_{axis}",
                                 "result": res})
                if res is not None and res["trades"] >= MIN_TRADES_FLOOR \
                        and res["max_dd_pct"] < DD_CEILING_PCT:
                    combos_scored += 1
                marker = ""
                if sc is not None and (best_pf is None or sc > best_pf):
                    best_pf = sc
                    best_val = v
                    best_res = res
                    marker = " ←best"
                tag = "OK" if sc is not None else "SKIP"
                pf = res["pf"] if res else None
                trd = res["trades"] if res else 0
                tot = res["total_R"] if res else 0.0
                dd = res["max_dd_pct"] if res else 0.0
                print(f"    {axis}={v}  trd={trd:4d}  pf={pf}  totR={tot:+.1f}  "
                      f"dd={dd:.1f}%  [{tag}]{marker}  ({time.time()-t0:.1f}s)",
                      flush=True)
            locked[axis] = best_val
            print(f"  [pass{pass_idx}] axis={axis}  LOCKED={best_val}  "
                  f"pf={best_pf}", flush=True)

    print(f"\nLOCKED after 2 passes: {locked}", flush=True)

    # ── Stage 3: top-3 stability ──
    # Re-rank ALL scored runs to find top-3, then re-run those (they're already
    # run — just pull them).
    scored = [r for r in all_runs
              if r["result"] is not None
              and r["result"]["trades"] >= MIN_TRADES_FLOOR
              and r["result"]["max_dd_pct"] < DD_CEILING_PCT]
    scored.sort(key=lambda r: r["result"]["pf"], reverse=True)
    top3 = scored[:3]
    print("\n=== Top-3 by PF (already scored) ===", flush=True)
    for k, r in enumerate(top3, 1):
        print(f"  #{k}  pf={r['result']['pf']:.3f}  totR={r['result']['total_R']:+.1f}  "
              f"trd={r['result']['trades']}  dd={r['result']['max_dd_pct']:.1f}%  "
              f"params={r['params']}", flush=True)

    # ── Acceptance check ──
    baseline_pf = baseline["pf"] if baseline else 0.0
    baseline_totR = baseline["total_R"] if baseline else 0.0
    pf_floor = max(1.3, baseline_pf * 1.20)
    print(f"\nAcceptance: pf >= {pf_floor:.3f} AND totR > {baseline_totR:+.1f}",
          flush=True)

    winner = None
    for r in top3:
        if r["result"]["pf"] >= pf_floor and r["result"]["total_R"] > baseline_totR:
            winner = r
            break
    print(f"\nWINNER: {winner}", flush=True)

    out = {
        "symbol": SYMBOL,
        "days": DAYS,
        "baseline": baseline,
        "locked_after_coord_descent": locked,
        "top3": [{"params": r["params"], "result": r["result"]} for r in top3],
        "winner": ({"params": winner["params"], "result": winner["result"]}
                   if winner else None),
        "combos_scored": combos_scored,
        "all_runs": all_runs,
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
