#!/usr/bin/env python3 -B
"""ROLLING-WF validation of a candidate ENTRY GATE vs baseline (2026-07-10).
Takes a symbol + a gate spec, runs 4 ANCHORED rolling walk-forward folds on deep
H1, and reports baseline vs gated per-fold OOS PF/return/trades. A gate SHIPS only
if it beats baseline on RETURN in >=3/4 folds AND reduces (or ~holds) trade count
AND never turns a positive-baseline fold negative. Reuses the entry-research sim.

Usage: python3 -B scripts/_trend_entry_validate.py SYMBOL GATEJSON
  e.g. ... ETHUSD '{"ADX":25}'
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("er", str(Path(__file__).resolve().parent / "_trend_entry_research.py"))
er = importlib.util.module_from_spec(spec); spec.loader.exec_module(er)


def folds(m, k=4):
    n = len(m); half = n // 2; step = (n - half) // k
    out = []
    for i in range(k):
        a = half + i * step
        b = n if i == k - 1 else half + (i + 1) * step
        out.append((m.iloc[:a].reset_index(drop=True), m.iloc[a:b].reset_index(drop=True)))
    return out


def yrs(seg):
    return max((seg["time"].iloc[-1] - seg["time"].iloc[0]).days / 365.25, 0.1)


def main():
    sym = sys.argv[1]
    gate = json.loads(sys.argv[2])
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import trend_exit_params
    tr, lk, gb, ac = trend_exit_params(sym)
    ex = {"TRAIL": tr, "LOCK": lk, "GIVEBACK": gb, "ACT": ac}
    m = er.build(sym)
    px = float(m["close"].iloc[-1])
    cost = 2.0 * (float(np.nanmedian(m["spread"].values)) * 0.01) / px
    fs = folds(m, 4)
    beats = 0; safe = True; rows = []
    for i, (is_seg, oos) in enumerate(fs):
        y = yrs(oos)
        bo = er.stats(er.simulate(oos, ex, {}, cost), y)
        go = er.stats(er.simulate(oos, ex, gate, cost), y)
        rows.append({"fold": i, "base": bo, "gate": go})
        if go["ret"] > bo["ret"]:
            beats += 1
        if bo["ret"] > 0 and go["ret"] < 0:      # turned a winning fold into a loss
            safe = False
    # trade count: total gated <= total baseline (real filter)
    tot_b = sum(r["base"]["n"] for r in rows)
    tot_g = sum(r["gate"]["n"] for r in rows)
    ship = beats >= 3 and safe and tot_g <= tot_b * 1.1
    print(json.dumps({"symbol": sym, "gate": gate, "folds_beaten": beats, "safe": safe,
                      "base_trades": tot_b, "gate_trades": tot_g, "SHIP": ship,
                      "per_fold": [{"fold": r["fold"],
                                    "base_ret": round(r["base"]["ret"], 3), "base_n": r["base"]["n"],
                                    "gate_ret": round(r["gate"]["ret"], 3), "gate_n": r["gate"]["n"]}
                                   for r in rows]}, default=float))


if __name__ == "__main__":
    main()
