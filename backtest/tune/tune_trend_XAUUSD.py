#!/usr/bin/env python3 -B
"""Hard-tune the LIVE TREND book for XAUUSD  (re-runnable, committed).

    python3 -B backtest/tune/tune_trend_XAUUSD.py

WHAT THIS TUNES
    The TREND ("7th strategy") book = agent/brain.py::_process_trend +
    agent/trend_follower.py. Daily (D1) 3-EMA ensemble signal (MIN_ABS_SIGNAL
    gates the agreement), SL = ATR_STOP*ATR(ATR_PERIOD), TP = TP_ATR*ATR, and an
    intraday exit stack (chandelier trail TRAIL*ATR over a 22-bar HH/LL window +
    profit-lock LOCK once profit >= ACT + peak-giveback GIVEBACK + signal-flip).

    The exit mechanics are copied faithfully from the VALIDATED reference sims
    scripts/_trend_exit_tune.py and scripts/_trend_exit_tune_h1.py (same causal
    ordering: set SL/giveback from the prior bar's peak, check THIS bar's exits,
    then update the peak — so one bar can never both set the peak and exit on its
    own pullback). This file adds: (a) total-R objective, (b) full param coverage
    (signal + risk + exits), (c) one-at-a-time WF sweeps + a combined WF verdict.

DATA / GRANULARITY  (honest caveat, printed at runtime)
    Signal + exits both run on the D1 cache raw_d1_XAUUSD.pkl (full history,
    ~3000 bars / ~11.6 yr). The XAU H1 cache (raw_h1_xauusd.pkl) is only ~1500
    bars (~3 months) — FAR too short for a full-history exit tune — so intrabar
    exits are approximated from D1 daily extremes (bar.high/low), exactly as the
    validated D1 reference tuner does. This is the documented fallback. A D1
    daily-extreme fill is slightly optimistic on same-bar stop-vs-target
    ordering; we use adverse-first ordering (stop/giveback before TP) so the sim
    never flatters itself.

OBJECTIVE
    total R  (sum of per-trade R, R = pnl_points / sl_dist, sl_dist = ATR_STOP*ATR
    at entry, minus round-trip spread cost expressed in R). PF is the robustness
    cross-check. Walk-forward 60/40 chronological: pick each param's best on the
    first 60% (TRAIN); KEEP it only if it is >= neutral (>= baseline) on the last
    40% (TEST). SHIP_NONE otherwise. Combined best-config is WF-validated too.
"""
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = "XAUUSD"
CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
D1_FILE = CACHE / "raw_d1_XAUUSD.pkl"
H1_FILE = CACHE / "raw_h1_xauusd.pkl"          # lowercase for XAU
OUT_JSON = Path(__file__).with_name("results_trend_XAUUSD.json")

TRAIL_LOOKBACK = 22        # chandelier HH/LL window (D1 bars) — live default, fixed
WF_SPLIT = 0.60            # 60% train / 40% test chronological
CHURN_MULT = 1.4           # best's test trade count > this x baseline = churn artifact, reject

# ── LIVE BASELINE (config.py + TREND_EXIT_PER_SYMBOL["XAUUSD"]) ──
BASELINE = {
    "ATR_STOP":   3.0,     # TREND_ATR_STOP
    "ATR_PERIOD": 20,      # TREND_ATR_PERIOD
    "TP":         6.0,     # TREND_TP_ATR   (None = trail-only)
    "TRAIL":      2.5,     # TREND_TRAIL_ATR / exit TRAIL (shared lever for XAU)
    "MIN_ABS":    0.34,    # TREND_MIN_ABS_SIGNAL
    "EMA_PAIRS":  [(16, 64), (32, 128), (64, 256)],  # TREND_EMA_PAIRS
    "LOCK":       0.5,     # exit LOCK
    "GIVEBACK":   0.30,    # exit GIVEBACK
    "ACT":        0.5,     # exit ACT
}

# ── ONE-AT-A-TIME sweep grids (from the task spec) ──
SWEEPS = {
    "ATR_STOP":   [2.0, 2.5, 3.0, 3.5, 4.0],
    "ATR_PERIOD": [14, 20, 28],
    "TP":         [4.0, 5.0, 6.0, 8.0, 10.0, None],     # None = trail-only
    "TRAIL":      [2.0, 2.5, 3.0, 3.5],
    "MIN_ABS":    [0.20, 0.34, 0.50, 0.67],
    "EMA_PAIRS":  [
        [(16, 64), (32, 128), (64, 256)],               # current
        [(16, 96), (32, 160), (64, 256)],               # wider (task-requested)
        [(20, 80), (40, 160), (80, 320)],               # slower alt
        [(12, 48), (24, 96), (48, 192)],                # faster alt
    ],
    # exit small grids around current
    "LOCK":       [0.4, 0.5, 0.6, 0.7],
    "GIVEBACK":   [0.20, 0.30, 0.40, 0.50],
    "ACT":        [0.3, 0.5, 0.7],
}

# param -> pretty printable current value
def _fmt(v):
    if isinstance(v, list):
        return "/".join("%d-%d" % (a, b) for a, b in v)
    return str(v)


def _naive(s):
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def load_d1():
    df = pickle.load(open(D1_FILE, "rb"))
    df["time"] = _naive(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def _atr(h, l, c, n):
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def _signal(close, pairs, min_abs):
    sig = pd.Series(0.0, index=close.index)
    for f, s in pairs:
        sig = sig + np.sign(close.ewm(span=f).mean() - close.ewm(span=s).mean())
    sig /= len(pairs)
    return sig.apply(lambda v: 0 if abs(v) < min_abs else (1 if v > 0 else -1))


def simulate(df, p, cost_pts):
    """Faithful copy of scripts/_trend_exit_tune.py::simulate, but returns per-trade
    R (= pnl_points / sl_dist_at_entry, minus round-trip cost in R). sl_dist =
    ATR_STOP*ATR at the entry bar. cost_pts = round-trip spread in price points."""
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    atr_arr = _atr(df["high"], df["low"], df["close"], int(p["ATR_PERIOD"])).values
    sig = _signal(df["close"], p["EMA_PAIRS"], float(p["MIN_ABS"])).values
    hh = df["high"].rolling(TRAIL_LOOKBACK).max().values
    ll = df["low"].rolling(TRAIL_LOOKBACK).min().values

    STOP = float(p["ATR_STOP"]); TR = float(p["TRAIL"])
    LK = float(p["LOCK"]); AC = float(p["ACT"]); GB = float(p["GIVEBACK"])
    TP = p["TP"]   # float or None

    trades = []
    pos = 0
    entry = sl = peak = 0.0
    sl_dist = 0.0
    tp = None
    blocked = 0
    n = len(df)
    start = 260   # EMA(256) + ATR warmup

    def _bank(exit_px, direction):
        pnl = (exit_px - entry) * direction
        return (pnl - cost_pts) / sl_dist if sl_dist > 0 else 0.0

    for t in range(start, n):
        a = atr_arr[t - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        s = int(sig[t - 1])          # prior close signal -> act at this open

        if pos == 0:
            if blocked and s != blocked:
                blocked = 0
            if s != 0 and s != blocked:
                pos = s
                entry = o[t]
                peak = 0.0
                sl_dist = STOP * a
                sl = entry - sl_dist if s == 1 else entry + sl_dist
                tp = None if TP is None else ((entry + TP * a) if s == 1 else (entry - TP * a))
            continue

        # 1) flip on signal reversal -> exit at open, re-enter opposite
        if s != 0 and s != pos:
            trades.append(_bank(o[t], pos))
            pos = s
            entry = o[t]; peak = 0.0; blocked = 0
            sl_dist = STOP * a
            sl = entry - sl_dist if pos == 1 else entry + sl_dist
            tp = None if TP is None else ((entry + TP * a) if pos == 1 else (entry - TP * a))
            continue

        # 2) causal exit checks (prior-bar peak sets SL/giveback; then update peak)
        if pos == 1:
            chand = hh[t - 1] - TR * a
            sl = max(sl, chand)
            if peak >= AC * a:
                sl = max(sl, entry + LK * peak)
            gb_level = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= AC * a) else -1e18
            exit_px = None
            if l[t] <= sl:
                exit_px = sl
            elif gb_level > -1e17 and l[t] <= gb_level:
                exit_px = gb_level; blocked = pos
            elif tp is not None and h[t] >= tp:
                exit_px = tp
            if exit_px is not None:
                trades.append(_bank(exit_px, pos)); pos = 0
            else:
                peak = max(peak, h[t] - entry)
        else:
            chand = ll[t - 1] + TR * a
            sl = min(sl, chand)
            if peak >= AC * a:
                sl = min(sl, entry - LK * peak)
            gb_level = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= AC * a) else 1e18
            exit_px = None
            if h[t] >= sl:
                exit_px = sl
            elif gb_level < 1e17 and h[t] >= gb_level:
                exit_px = gb_level; blocked = pos
            elif tp is not None and l[t] <= tp:
                exit_px = tp
            if exit_px is not None:
                trades.append(_bank(exit_px, pos)); pos = 0
            else:
                peak = max(peak, entry - l[t])
    return trades


def metrics(rets):
    if not rets:
        return {"n": 0, "R": 0.0, "pf": 0.0, "wr": 0.0}
    r = np.array(rets)
    wins = r[r > 0].sum(); losses = -r[r < 0].sum()
    pf = float(wins / losses) if losses > 0 else (99.0 if wins > 0 else 0.0)
    return {"n": len(r), "R": float(r.sum()), "pf": pf, "wr": float((r > 0).mean())}


def _run(df, p, cost_pts):
    return metrics(simulate(df, p, cost_pts))


def main():
    df = load_d1()
    # H1 depth check (why we fall back to D1 extremes)
    try:
        h1 = pickle.load(open(H1_FILE, "rb"))
        h1_n = len(h1)
        h1_span = "%s -> %s" % (str(_naive(h1["time"]).iloc[0])[:10],
                                str(_naive(h1["time"]).iloc[-1])[:10])
    except Exception:
        h1_n, h1_span = 0, "missing"

    # round-trip cost in price points (spread-only, no slippage, per live model)
    med_spread_pts = float(np.nanmedian(df["spread"].values)) if "spread" in df else 0.0
    cost_pts = max(2.0 * med_spread_pts * 0.01, 0.02)   # points; 0.01/pt for gold

    split = int(len(df) * WF_SPLIT)
    df_tr = df.iloc[:split].reset_index(drop=True)
    df_te = df.iloc[split - 260:].reset_index(drop=True)   # keep 260-bar warmup lead-in
    split_date = str(df["time"].iloc[split])[:10]

    print("=" * 92)
    print("TREND hard-tune  %s   (objective = total R;  PF = robustness check)" % SYMBOL)
    print("=" * 92)
    print("D1 signal+exit cache : %s   bars=%d   %s -> %s"
          % (D1_FILE.name, len(df), str(df["time"].iloc[0])[:10], str(df["time"].iloc[-1])[:10]))
    print("H1 exit cache        : %s   bars=%d   %s" % (H1_FILE.name, h1_n, h1_span))
    print("!! H1 is only ~%d bars (~3 mo) — TOO SHORT for a full-history exit tune." % h1_n)
    print("   Exits approximated from D1 DAILY EXTREMES (validated fallback, adverse-first).")
    print("round-trip cost      : %.3f pts   WF split 60/40 @ %s" % (cost_pts, split_date))
    print("-" * 92)

    base_tr = _run(df_tr, BASELINE, cost_pts)
    base_te = _run(df_te, BASELINE, cost_pts)
    base_full = _run(df, BASELINE, cost_pts)
    print("BASELINE  full: R=%7.2f  PF=%5.2f  n=%3d  wr=%.2f  |  train R=%7.2f PF=%.2f n=%d  |  test R=%7.2f PF=%.2f n=%d"
          % (base_full["R"], base_full["pf"], base_full["n"], base_full["wr"],
             base_tr["R"], base_tr["pf"], base_tr["n"], base_te["R"], base_te["pf"], base_te["n"]))
    print("-" * 92)
    print("PER-PARAM one-at-a-time.  SHIP requires ALL of:")
    print("  (WF)    best-on-train has test_R >= baseline test_R")
    print("  (PF)    best-on-train has test_PF >= baseline test_PF  (total-R alone is denominator-")
    print("          biased toward tighter stops/TP; a real edge must also hold PF)")
    print("  (CHURN) best's test trade count <= %.1fx baseline (a lift from 3-6x more trades on" % CHURN_MULT)
    print("          D1 daily-extreme fills is churn, not edge)")
    print("%-11s %-12s %-12s %8s %8s %8s %8s %6s %6s   %s"
          % ("param", "current", "best", "trBase", "trBest", "teBase", "teBest", "teN", "tePF", "verdict"))

    per_param = []
    NEUTRAL_EPS = 1e-6
    for param, grid in SWEEPS.items():
        cur = BASELINE[param]
        rows = []
        for val in grid:
            p = deepcopy(BASELINE); p[param] = val
            rows.append((val, _run(df_tr, p, cost_pts), _run(df_te, p, cost_pts)))
        # best on TRAIN by total R (the requested objective)
        best_val, best_tr, best_te = max(rows, key=lambda r: r[1]["R"])
        cur_row = next((r for r in rows if r[0] == cur), None)
        cur_tr = cur_row[1] if cur_row else base_tr
        cur_te = cur_row[2] if cur_row else base_te

        # grid-edge flag (numeric grids only): best sits at min/max => optimum wants
        # to leave the grid = a fill/rescaling artifact, not an interior optimum.
        numeric = all(not isinstance(v, list) for v in grid) and param != "EMA_PAIRS"
        at_edge = numeric and best_val in (min(grid, key=lambda x: (x is None, x)),
                                           max(grid, key=lambda x: (x is None, x)))

        wf_ok_p = best_te["R"] >= cur_te["R"] - NEUTRAL_EPS
        pf_ok_p = best_te["pf"] >= cur_te["pf"] - 0.01
        churn = cur_te["n"] > 0 and best_te["n"] > CHURN_MULT * cur_te["n"]

        if best_val == cur:
            verdict, ship_val = "SHIP_NONE (baseline already best on train)", cur
        elif not wf_ok_p:
            verdict, ship_val = "SHIP_NONE (train-only, fails WF test)", cur
        elif churn:
            verdict, ship_val = ("SHIP_NONE (CHURN artifact: %dx trades)"
                                 % round(best_te["n"] / max(1, cur_te["n"])), cur)
        elif not pf_ok_p:
            verdict, ship_val = ("SHIP_NONE (R-rescale only: test PF %.2f<%.2f)"
                                 % (best_te["pf"], cur_te["pf"]), cur)
        elif at_edge:
            verdict, ship_val = ("SHIP_NONE (tight grid-edge = D1-fill artifact; needs H1)", cur)
        else:
            verdict, ship_val = "SHIP -> %s" % _fmt(best_val), best_val

        print("%-11s %-12s %-12s %8.2f %8.2f %8.2f %8.2f %6d %6.2f   %s"
              % (param, _fmt(cur), _fmt(best_val),
                 cur_tr["R"], best_tr["R"], cur_te["R"], best_te["R"],
                 best_te["n"], best_te["pf"], verdict))
        per_param.append({
            "param": param,
            "current": _fmt(cur),
            "best": _fmt(best_val),
            "ship_value": _fmt(ship_val),
            "total_R_train_base": round(cur_tr["R"], 3),
            "total_R_train_best": round(best_tr["R"], 3),
            "total_R_test_base": round(cur_te["R"], 3),
            "total_R_test_best": round(best_te["R"], 3),
            "n_test_base": cur_te["n"], "n_train_best": best_tr["n"], "n_test_best": best_te["n"],
            "pf_test_base": round(cur_te["pf"], 2), "pf_test_best": round(best_te["pf"], 2),
            "grid_edge": bool(at_edge),
            "verdict": verdict,
            "_ship_raw": ship_val,   # internal (real python value)
        })

    # ── COMBINED best-config = baseline + every SHIP param, then WF-validate ──
    combined = deepcopy(BASELINE)
    shipped = []
    for r in per_param:
        if r["verdict"].startswith("SHIP ->"):
            combined[r["param"]] = r["_ship_raw"]
            shipped.append("%s=%s" % (r["param"], r["best"]))

    comb_tr = _run(df_tr, combined, cost_pts)
    comb_te = _run(df_te, combined, cost_pts)
    comb_full = _run(df, combined, cost_pts)
    wf_ok = (comb_tr["R"] >= base_tr["R"] - NEUTRAL_EPS) and (comb_te["R"] >= base_te["R"] - NEUTRAL_EPS)

    print("-" * 92)
    print("COMBINED shipped params: %s" % (", ".join(shipped) if shipped else "(none)"))
    print("COMBINED  full: R=%7.2f  PF=%5.2f  n=%3d  wr=%.2f  |  train R=%7.2f PF=%.2f  |  test R=%7.2f PF=%.2f"
          % (comb_full["R"], comb_full["pf"], comb_full["n"], comb_full["wr"],
             comb_tr["R"], comb_tr["pf"], comb_te["R"], comb_te["pf"]))
    print("COMBINED vs BASELINE  train dR=%+.2f  test dR=%+.2f   WF_OK=%s"
          % (comb_tr["R"] - base_tr["R"], comb_te["R"] - base_te["R"], wf_ok))
    ship_combined = wf_ok and bool(shipped)
    print("VERDICT: %s" % ("SHIP combined config" if ship_combined
                           else ("SHIP_NONE — combined has no WF edge over baseline"
                                 if not shipped else "SHIP_NONE — combined fails WF on test")))
    print("=" * 92)

    # strip internal key
    for r in per_param:
        r.pop("_ship_raw", None)

    result = {
        "symbol": SYMBOL,
        "objective": "total_R (pnl_points / sl_dist), PF = robustness check",
        "data": {
            "d1_file": D1_FILE.name, "d1_bars": len(df),
            "d1_span": [str(df["time"].iloc[0])[:10], str(df["time"].iloc[-1])[:10]],
            "h1_file": H1_FILE.name, "h1_bars": h1_n,
            "exit_granularity": "D1 daily extremes (H1 too short ~3mo; validated fallback)",
            "cost_pts": round(cost_pts, 4),
            "wf_split": WF_SPLIT, "split_date": split_date,
        },
        "baseline": {
            "params": {k: _fmt(v) for k, v in BASELINE.items()},
            "total_R": round(base_full["R"], 3), "PF": round(base_full["pf"], 2),
            "n": base_full["n"],
            "total_R_train": round(base_tr["R"], 3), "total_R_test": round(base_te["R"], 3),
            "PF_train": round(base_tr["pf"], 2), "PF_test": round(base_te["pf"], 2),
        },
        "per_param": per_param,
        "combined": {
            "params": {k: _fmt(v) for k, v in combined.items()},
            "shipped": shipped,
            "total_R": round(comb_full["R"], 3), "PF": round(comb_full["pf"], 2),
            "n": comb_full["n"],
            "total_R_train": round(comb_tr["R"], 3), "total_R_test": round(comb_te["R"], 3),
            "PF_train": round(comb_tr["pf"], 2), "PF_test": round(comb_te["pf"], 2),
            "wf_ok": bool(wf_ok),
            "ship": bool(ship_combined),
        },
        "notes": [
            "Signal + exits both simulated on D1 (raw_d1_XAUUSD.pkl, full history).",
            "H1 XAU cache is ~%d bars (~3 months) — too short; exits use D1 daily extremes." % h1_n,
            "Exit logic copied from validated scripts/_trend_exit_tune.py (causal peak ordering, adverse-first).",
            "Objective = total R; WF 60/40 chronological; SHIP a param only if best-on-train is >= baseline on test.",
            "TRAIL is a shared lever: config TREND_TRAIL_ATR and TREND_EXIT_PER_SYMBOL['XAUUSD']['TRAIL'] are both 2.5.",
            "Distrust universal lifts: check n — a change helping every fold equally at higher trade count = churn/cost artifact.",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float))
    print("wrote %s" % OUT_JSON)


if __name__ == "__main__":
    main()
