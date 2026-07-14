#!/usr/bin/env python3 -B
"""Hard-tune EVERY parameter of the live TREND book for NAS100.r (magic +6000).

Faithful to agent/brain.py::_process_trend + agent/trend_follower.py:
  - D1 3-EMA ensemble signal (default 16/64, 32/128, 64/256), MIN_ABS gate (all/most agree)
  - Entry on signal edge; SL = ATR_STOP x ATR(ATR_PERIOD); TP = TP_ATR x ATR (or None)
  - H1 intraday exits: Chandelier trail (TRAIL x ATR / 22d hh-ll) + profit-lock (LOCK @ ACT)
    + peak-giveback reversal (GIVEBACK) + D1 signal-flip exit.
  - Exits checked per H1 bar (~12x finer, close to live 60s cadence) via merge_asof of
    the prior completed D1 context onto H1 bars.  [mirrors scripts/_trend_exit_tune_h1.py
    and the validated reconstruction nas_emergency_cap.py]

CHURN ARTIFACT GUARD (central to this file)
-------------------------------------------
The naive cadence re-enters the SAME direction on the very next H1 bar after a
trail/TP stop (D1 signal is still long), producing thousands of micro round-trips
(observed 4638 trades / PF 16.87 = FAKE). The realistic / live cadence blocks
re-entry in a direction until the D1 signal FLIPS to the opposite side. Every
sweep cell below is evaluated CHURN-FREE (block-until-flip) and the churn count is
reported alongside so the artifact can never leak into a verdict.

METHOD
------
- Full H1 history, CHURN-FREE cadence, objective = total R (pnl_pts / entry_sl_dist).
- Sweep ONE parameter at a time off the live baseline; then combine the shipped winners.
- Walk-forward 60/40 chronological. SHIP a value only if it is best-on-TRAIN total-R
  AND its TEST total-R is >= the baseline TEST total-R (>= neutral out-of-sample).
- Combined config is WF-validated the same way (train AND test >= baseline).

Writes results_trend_NAS100_r.json next to this file. Does NOT touch config.py/brain.py.
Run: python3 -B backtest/tune/tune_trend_NAS100_r.py
"""
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
SYM = "NAS100.r"
OUT = Path(__file__).resolve().parent / "results_trend_NAS100_r.json"

SPREAD_PTS = 1.5      # NAS100.r round-trip entry worsen (points) — matches reconstruction
TRAIL_LB = 22         # Chandelier hh/ll lookback (D1 bars) — fixed live constant

# ── LIVE BASELINE (config.TREND_* + TREND_EXIT_PER_SYMBOL["NAS100.r"]) ──
BASE = {
    "EMA_PAIRS": [(16, 64), (32, 128), (64, 256)],
    "MIN_ABS": 0.34,
    "ATR_P": 20,
    "ATR_STOP": 3.0,
    "TP_ATR": 6.0,
    "TR": 2.5,     # TRAIL_ATR
    "LK": 0.6,     # LOCK
    "GB": 0.35,    # GIVEBACK
    "ACT": 0.5,    # LOCK_ACTIVATE (x ATR)
}

# EMA-pair alternatives (default + 2 alts)
EMA_ALTS = {
    "default_16/64": [(16, 64), (32, 128), (64, 256)],
    "wide_slow_16/96": [(16, 96), (32, 160), (64, 256)],   # ETH winner shape
    "fast_8/32": [(8, 32), (16, 64), (32, 128)],
}

# ── one-at-a-time grids ──
GRIDS = {
    "ATR_STOP": [2.0, 2.5, 3.0, 3.5, 4.0],
    "ATR_P":    [14, 20, 28],
    "TP_ATR":   [4.0, 5.0, 6.0, 8.0, 10.0, None],
    "TR":       [2.0, 2.5, 3.0, 3.5],
    "MIN_ABS":  [0.20, 0.34, 0.50, 0.67],
    "EMA_PAIRS": list(EMA_ALTS.keys()),           # handled specially
    "LK":       [0.4, 0.5, 0.6, 0.7, 0.8],
    "GB":       [0.25, 0.35, 0.50, 0.65],
    "ACT":      [0.3, 0.5, 0.7],
}


def _naive(s):
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def d1_context(ema_pairs, min_abs, atr_p):
    d = pickle.load(open(C / ("raw_d1_" + SYM.replace(".", "_") + ".pkl"), "rb"))
    d["time"] = _naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
    c, h, l = d["close"], d["high"], d["low"]
    sig = pd.Series(0.0, index=d.index)
    for f, s in ema_pairs:
        sig = sig + np.sign(c.ewm(span=f).mean() - c.ewm(span=s).mean())
    sig = (sig / len(ema_pairs)).apply(
        lambda v: 0 if abs(v) < min_abs else (1 if v > 0 else -1))
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_p, adjust=False).mean()
    out = pd.DataFrame({"eff": d["time"].dt.normalize() + pd.Timedelta(days=1),
                        "sig": sig.astype(int), "atr": atr,
                        "hh": h.rolling(TRAIL_LB).max(), "ll": l.rolling(TRAIL_LB).min()})
    return out.dropna().reset_index(drop=True)


def load_h1():
    h1 = pickle.load(open(C / ("raw_h1_" + SYM.replace(".", "_") + ".pkl"), "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    h1["time"] = h1["time"].astype("datetime64[ns]")
    return h1


_H1 = None
def build(p):
    """Merge prior-completed D1 context (for these signal params) onto every H1 bar."""
    global _H1
    if _H1 is None:
        _H1 = load_h1()
    d1 = d1_context(p["EMA_PAIRS"], p["MIN_ABS"], p["ATR_P"])
    d1["eff"] = d1["eff"].astype("datetime64[ns]")
    m = pd.merge_asof(_H1, d1, left_on="time", right_on="eff", direction="backward").dropna(
        subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)
    return m


def simulate(m, p, churn_free=True):
    """Per-trade pnl in R. churn_free=True blocks same-dir re-entry until the D1
    signal flips to the opposite side (live cadence). churn_free=False = naive
    (re-enter next bar) = the FAKE churn artifact, kept only for the guard ratio."""
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    tm = m["time"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    ATR_STOP = p["ATR_STOP"]; TP_ATR = p["TP_ATR"]
    TR = p["TR"]; LK = p["LK"]; GB = p["GB"]; ACT = p["ACT"]
    pos = 0; entry = sl = 0.0; tp = None; peak = 0.0; sl_dist = 0.0
    ent_t = None; blk_dir = 0
    trades = []

    def _open(t, s):
        nonlocal pos, entry, peak, ent_t, sl_dist, sl, tp
        a = atr[t]
        pos = s; entry = o[t]; peak = 0.0; ent_t = tm[t]
        sl_dist = ATR_STOP * a
        sl = entry - sl_dist if s == 1 else entry + sl_dist
        tp = None if TP_ATR is None else ((entry + TP_ATR * a) if s == 1 else (entry - TP_ATR * a))

    def close(exit_px, reason):
        nonlocal pos
        d = pos
        pnl_pts = (exit_px - entry) * d - SPREAD_PTS
        trades.append({"entry_t": str(ent_t)[:19], "dir": d,
                       "pnl_R": pnl_pts / sl_dist, "exit_reason": reason})
        pos = 0

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blk_dir and s != blk_dir:          # clear block only when signal flips away
                blk_dir = 0
            if s != 0 and s != blk_dir:
                _open(t, s)
            continue
        # flip on daily signal reversal (a genuine flip, never churn)
        if s != 0 and s != pos:
            close(o[t], "FLIP")
            blk_dir = 0
            _open(t, s)
            continue
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else -1e18
            ex = None; rsn = None
            if l[t] <= sl:
                ex = sl; rsn = "SL/TRAIL"
            elif gb > -1e17 and l[t] <= gb:
                ex = gb; rsn = "GIVEBACK"
            elif tp is not None and h[t] >= tp:
                ex = tp; rsn = "TP"
            if ex is not None:
                close(ex, rsn)
                if churn_free:
                    blk_dir = 1                    # block re-long until D1 flips short
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR * a)
            if peak >= ACT * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else 1e18
            ex = None; rsn = None
            if h[t] >= sl:
                ex = sl; rsn = "SL/TRAIL"
            elif gb < 1e17 and h[t] >= gb:
                ex = gb; rsn = "GIVEBACK"
            elif tp is not None and l[t] <= tp:
                ex = tp; rsn = "TP"
            if ex is not None:
                close(ex, rsn)
                if churn_free:
                    blk_dir = -1                   # block re-short until D1 flips long
            else:
                peak = max(peak, entry - l[t])
    return trades


def metrics(trades):
    R = np.array([t["pnl_R"] for t in trades]) if trades else np.array([])
    if len(R) == 0:
        return {"n": 0, "R": 0.0, "PF": 0.0, "WR": 0.0}
    w = R[R > 0].sum(); ls = -R[R < 0].sum()
    pf = float(w / ls) if ls > 0 else float("inf")
    return {"n": len(R), "R": float(R.sum()),
            "PF": (round(pf, 2) if np.isfinite(pf) else 999.0),
            "WR": round(float((R > 0).mean()) * 100, 1)}


def wf(m, p):
    """Full + walk-forward 60/40 churn-free metrics for a param set."""
    tr = simulate(m, p, churn_free=True)
    n = len(tr)
    split = int(n * 0.60)
    full = metrics(tr)
    train = metrics(tr[:split])
    test = metrics(tr[split:])
    churn_n = len(simulate(m, p, churn_free=False))
    return {"full": full, "train": train, "test": test,
            "churn_n": churn_n,
            "churn_ratio": round(churn_n / n, 1) if n else 0.0}


def eval_params(p):
    return wf(build(p), p)


def main():
    d1c = d1_context(BASE["EMA_PAIRS"], BASE["MIN_ABS"], BASE["ATR_P"])
    m0 = build(BASE)
    print("=" * 78)
    print(f"TREND hard-tune  {SYM}   (CHURN-FREE cadence = truth)")
    print(f"H1 bars merged: {len(m0)}   D1 signal rows: {len(d1c)}")
    print(f"date range: {str(m0['time'].iloc[0])[:10]} -> {str(m0['time'].iloc[-1])[:10]}")
    print("=" * 78)

    base = eval_params(BASE)
    b_full, b_tr, b_te = base["full"], base["train"], base["test"]
    print(f"BASELINE (churn-free): n={b_full['n']}  fullR={b_full['R']:+.2f} PF={b_full['PF']} "
          f"WR={b_full['WR']}%")
    print(f"  churn (naive re-entry) n={base['churn_n']}  ratio={base['churn_ratio']}x  <-- FAKE")
    print(f"  WF60/40  trainR={b_tr['R']:+.2f} PF={b_tr['PF']} (n={b_tr['n']}) | "
          f"testR={b_te['R']:+.2f} PF={b_te['PF']} (n={b_te['n']})")

    results = {
        "symbol": SYM,
        "method": ("Full H1 history, CHURN-FREE cadence (block same-dir re-entry until D1 "
                   "signal flips). Objective total R. WF 60/40: ship best-on-train only if "
                   "test R >= baseline test R. Churn (naive re-entry) reported as the FAKE "
                   "artifact for every cell."),
        "baseline": {"params": {k: (list(map(list, v)) if k == "EMA_PAIRS" else v)
                                for k, v in BASE.items()},
                     "full": b_full, "train": b_tr, "test": b_te,
                     "churn_n": base["churn_n"], "churn_ratio": base["churn_ratio"]},
        "per_param": [],
    }

    shipped = {}   # param -> shipped value (for combined)
    for param, grid in GRIDS.items():
        print("\n" + "-" * 78)
        print(f"SWEEP {param}   (baseline value = "
              f"{'default_16/64' if param == 'EMA_PAIRS' else BASE[param]})")
        print(f"{'value':>16} | {'trR':>8} {'trPF':>6} | {'teR':>8} {'tePF':>6} | "
              f"{'fullR':>8} {'fullPF':>6} {'n':>4} {'churn':>6}")
        cells = []
        best = None  # (train_R, value, cellmetrics)
        for val in grid:
            p = deepcopy(BASE)
            if param == "EMA_PAIRS":
                p["EMA_PAIRS"] = EMA_ALTS[val]
            else:
                p[param] = val
            r = eval_params(p)
            cell = {"value": (val if param != "EMA_PAIRS"
                              else {"name": val, "pairs": list(map(list, EMA_ALTS[val]))}),
                    "train_R": round(r["train"]["R"], 2), "train_PF": r["train"]["PF"],
                    "test_R": round(r["test"]["R"], 2), "test_PF": r["test"]["PF"],
                    "full_R": round(r["full"]["R"], 2), "full_PF": r["full"]["PF"],
                    "n": r["full"]["n"], "churn_n": r["churn_n"],
                    "churn_ratio": r["churn_ratio"]}
            cells.append(cell)
            vlabel = val if param != "EMA_PAIRS" else val
            mark = ""
            if param == "EMA_PAIRS":
                is_base = (val == "default_16/64")
            elif param == "TP_ATR":
                is_base = (val == BASE[param])
            else:
                is_base = (abs(val - BASE[param]) < 1e-9)
            if is_base:
                mark = " <base"
            print(f"{str(vlabel):>16} | {r['train']['R']:>8.2f} {r['train']['PF']:>6} | "
                  f"{r['test']['R']:>8.2f} {r['test']['PF']:>6} | "
                  f"{r['full']['R']:>8.2f} {r['full']['PF']:>6} {r['full']['n']:>4} "
                  f"{r['churn_n']:>6}{mark}")
            if best is None or r["train"]["R"] > best[0]:
                best = (r["train"]["R"], val, cell)

        best_val = best[1]; best_cell = best[2]
        # SHIP: best-on-train must be non-baseline AND test >= baseline test (neutral)
        if param == "EMA_PAIRS":
            is_base_best = (best_val == "default_16/64")
        elif param == "TP_ATR":
            is_base_best = (best_val == BASE[param])
        else:
            is_base_best = (abs(best_val - BASE[param]) < 1e-9)
        test_ok = best_cell["test_R"] >= b_te["R"] - 1e-9
        train_beats = best_cell["train_R"] > b_tr["R"] + 1e-9
        if is_base_best or not train_beats:
            verdict = "SHIP_NONE"; reason = "baseline is best-on-train (or not beaten)"
        elif not test_ok:
            verdict = "SHIP_NONE"; reason = f"test {best_cell['test_R']:+.2f} < baseline {b_te['R']:+.2f} (curve-fit)"
        else:
            verdict = "SHIP"; reason = "beats baseline on train AND >= neutral on test"
            shipped[param] = best_val
        print(f"  -> best-on-train = {best_val}  verdict={verdict}  ({reason})")
        results["per_param"].append({
            "param": param, "baseline_value": ("default_16/64" if param == "EMA_PAIRS" else BASE[param]),
            "grid": [str(g) for g in grid], "cells": cells,
            "best_on_train": (best_val if param != "EMA_PAIRS" else best_val),
            "verdict": verdict, "reason": reason})

    # ── COMBINED: baseline overlaid with every shipped winner, WF-validated ──
    print("\n" + "=" * 78)
    combo = deepcopy(BASE)
    combo_desc = {}
    for param, val in shipped.items():
        if param == "EMA_PAIRS":
            combo["EMA_PAIRS"] = EMA_ALTS[val]; combo_desc["EMA_PAIRS"] = val
        else:
            combo[param] = val; combo_desc[param] = val
    print(f"COMBINED shipped winners: {combo_desc if combo_desc else '(none — all SHIP_NONE)'}")
    cr = eval_params(combo)
    c_full, c_tr, c_te = cr["full"], cr["train"], cr["test"]
    train_ok = c_tr["R"] >= b_tr["R"] - 1e-9
    test_ok = c_te["R"] >= b_te["R"] - 1e-9
    combo_ship = bool(combo_desc) and train_ok and test_ok
    print(f"  trainR={c_tr['R']:+.2f} PF={c_tr['PF']} (base {b_tr['R']:+.2f}) -> "
          f"{'OK' if train_ok else 'WORSE'}")
    print(f"  testR ={c_te['R']:+.2f} PF={c_te['PF']} (base {b_te['R']:+.2f}) -> "
          f"{'OK' if test_ok else 'WORSE'}")
    print(f"  fullR ={c_full['R']:+.2f} PF={c_full['PF']} (base {b_full['R']:+.2f})  "
          f"n={c_full['n']} churn={cr['churn_n']}")
    print(f"  COMBINED VERDICT = {'SHIP' if combo_ship else 'SHIP_NONE'}")

    results["combined"] = {
        "shipped_params": combo_desc,
        "params": {k: (list(map(list, v)) if k == "EMA_PAIRS" else v) for k, v in combo.items()},
        "train": c_tr, "test": c_te, "full": c_full,
        "churn_n": cr["churn_n"], "churn_ratio": cr["churn_ratio"],
        "vs_baseline": {"train_R_delta": round(c_tr["R"] - b_tr["R"], 2),
                        "test_R_delta": round(c_te["R"] - b_te["R"], 2),
                        "full_R_delta": round(c_full["R"] - b_full["R"], 2)},
        "verdict": "SHIP" if combo_ship else "SHIP_NONE",
    }
    results["notes"] = [
        "CHURN-FREE cadence is truth: block same-dir re-entry until the D1 signal flips.",
        f"Naive re-entry churns to ~{base['churn_ratio']}x the trades (baseline "
        f"{base['churn_n']} vs {b_full['n']}) with an inflated PF — the known FAKE artifact.",
        "SPREAD_PTS=1.5 round-trip; exits checked every H1 bar; SL_dist = ATR_STOP x ATR at entry.",
        "WF 60/40 chronological; ship only if train beats baseline AND test >= baseline (neutral).",
        "Deliverable does NOT modify config.py/brain.py — apply winners manually if desired.",
    ]

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
