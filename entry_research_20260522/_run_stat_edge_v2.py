#!/usr/bin/env python3 -B
"""V2 — relax BL n_min to find more BL cells; also keep WL strict.

Tests several variants:
  BL_n15_pf1.0  — original
  BL_n10_pf1.0  — relax sample
  BL_n10_pf0.8  — more aggressive (only really-bad cells)
  BL_n20_pf1.2  — conservative-conservative
"""
import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, load_data  # noqa: E402
import backtest.v5_backtest as v5
import inspect

SYMBOLS = [
    "DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY",
    "EURUSD", "US2000.r", "UKOUSD", "JPN225ft",
]
DAYS = 360

VARIANTS = [
    # (label, n_min, pf_max)
    ("BL_n15_pf1.0", 15, 1.0),
    ("BL_n10_pf1.0", 10, 1.0),
    ("BL_n10_pf0.8", 10, 0.8),
    ("BL_n20_pf1.2", 20, 1.2),
    ("BL_n12_pf0.9", 12, 0.9),
]

OUT_DIR = Path(__file__).resolve().parent


# ---- patched backtest_symbol that returns trades ----
src = inspect.getsource(v5.backtest_symbol)
new_src = src.replace(
    "    return result",
    "    result['_trades'] = trades\n    return result",
    1,
)
ns = dict(v5.__dict__)
exec(new_src, ns)
patched = ns["backtest_symbol"]


def trades_metrics(trades):
    if not trades:
        return {"n": 0, "wins": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "avg_r": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gw / gl if gl > 0 else (999.0 if gw > 0 else 0.0)
    wr = len(wins) / len(trades) * 100
    avg_r = float(np.mean([t.get("pnl_r", 0.0) for t in trades]))
    pnl = sum(t["pnl"] for t in trades)
    return {"n": len(trades), "wins": len(wins),
            "wr": round(wr, 1), "pf": round(pf, 2),
            "avg_r": round(avg_r, 2), "pnl": round(pnl, 2)}


def build_cell_map(trades):
    cells = defaultdict(list)
    for t in trades:
        cells[(t["hour"], t["dow"])].append(t)
    return {k: trades_metrics(v) for k, v in cells.items()}


def apply_blacklist(trades, cell_map, n_min, pf_max):
    keep = []
    for t in trades:
        c = cell_map.get((t["hour"], t["dow"]))
        if c is None:
            keep.append(t); continue
        if c["n"] >= n_min and c["pf"] < pf_max:
            continue
        keep.append(t)
    return keep


def walk_forward(trades, n_min, pf_max, n_folds=5):
    trades_sorted = sorted(trades, key=lambda x: x["ts"])
    N = len(trades_sorted)
    if N < n_folds * 3:
        return []
    folds = []
    chunk = N // n_folds
    for k in range(1, n_folds):
        train = trades_sorted[: k * chunk]
        test  = trades_sorted[k * chunk:(k + 1) * chunk]
        if not test:
            continue
        cm = build_cell_map(train)
        base_m = trades_metrics(test)
        bl_m = trades_metrics(apply_blacklist(test, cm, n_min, pf_max))
        folds.append({
            "fold": k,
            "base_pnl": base_m["pnl"], "bl_pnl": bl_m["pnl"],
            "delta": round(bl_m["pnl"] - base_m["pnl"], 2),
            "bl_n":   bl_m["n"], "bl_pf": bl_m["pf"],
        })
    return folds


def run():
    sym_trades = {}
    for sym in SYMBOLS:
        r = patched(sym, days=DAYS, params=None, verbose=False)
        if r is None or r.get("trades", 0) == 0:
            print(f"{sym}: no trades"); continue
        df = load_data(sym, DAYS)
        if df is None:
            print(f"{sym}: no data"); continue
        times = df["time"].values
        n = len(times)
        trades = []
        for t in r["_trades"]:
            eb = t.get("entry_bar")
            if eb is None or eb >= n:
                continue
            ts = pd.Timestamp(times[eb])
            t2 = dict(t); t2["hour"] = ts.hour; t2["dow"] = int(ts.dayofweek); t2["ts"] = ts
            trades.append(t2)
        sym_trades[sym] = trades

    print(f"{'variant':16s}  {'sym':10s}  {'base':>10s}  {'BL':>10s}  {'delta':>9s}  "
          f"{'wf_avgΔ':>9s}  {'wf_pf':>6s}  {'pos/N':>6s}  ship?")

    portfolio = {v[0]: {"base": 0.0, "bl": 0.0, "wf_pass_syms": 0,
                       "ship_syms": []} for v in VARIANTS}
    out_payload = {"variants": {}}

    for var_label, n_min, pf_max in VARIANTS:
        v_data = {}
        for sym, trades in sym_trades.items():
            cm = build_cell_map(trades)
            base = trades_metrics(trades)
            bl   = trades_metrics(apply_blacklist(trades, cm, n_min, pf_max))
            wf   = walk_forward(trades, n_min, pf_max, n_folds=5)
            full_delta = round(bl["pnl"] - base["pnl"], 2)
            if wf:
                deltas = [f["delta"] for f in wf]
                pfs    = [f["bl_pf"] for f in wf if f["bl_n"] > 0]
                wf_avg_delta = round(float(np.mean(deltas)), 2)
                wf_avg_pf    = round(float(np.mean(pfs)) if pfs else 0.0, 2)
                pos_folds    = sum(1 for d in deltas if d > 0)
                n_folds      = len(deltas)
            else:
                wf_avg_delta = wf_avg_pf = 0.0
                pos_folds = n_folds = 0
            ship = (full_delta >= 30
                    and wf_avg_pf > 1.5
                    and pos_folds >= max(3, math.ceil(0.6 * n_folds)))
            print(f"{var_label:16s}  {sym:10s}  {base['pnl']:>10.0f}  "
                  f"{bl['pnl']:>10.0f}  {full_delta:>+9.2f}  "
                  f"{wf_avg_delta:>+9.2f}  {wf_avg_pf:>6.2f}  "
                  f"{pos_folds}/{n_folds:<3d}  {ship}")
            portfolio[var_label]["base"] += base["pnl"]
            portfolio[var_label]["bl"]   += bl["pnl"]
            if ship:
                portfolio[var_label]["ship_syms"].append(sym)
            v_data[sym] = {
                "base_pnl": base["pnl"], "bl_pnl": bl["pnl"],
                "delta": full_delta,
                "wf_avg_delta": wf_avg_delta, "wf_avg_pf": wf_avg_pf,
                "pos_folds": pos_folds, "n_folds": n_folds,
                "ship": ship,
            }
        out_payload["variants"][var_label] = {
            "params": {"n_min": n_min, "pf_max": pf_max},
            "symbols": v_data,
        }
        print()

    print("=" * 78)
    print("PORTFOLIO-LEVEL BY VARIANT")
    print("=" * 78)
    for v, p in portfolio.items():
        delta = p["bl"] - p["base"]
        print(f"  {v:16s} base=${p['base']:>10.0f}  BL=${p['bl']:>10.0f}  "
              f"delta=${delta:+,.2f}  ship_syms={p['ship_syms']}")
        out_payload["variants"][v]["portfolio"] = {
            "base_pnl": round(p["base"], 2),
            "bl_pnl":   round(p["bl"],   2),
            "delta":    round(delta, 2),
            "ship_syms": p["ship_syms"],
            "ship_count": len(p["ship_syms"]),
        }

    out_json = OUT_DIR / "10_stat_edge_variants.json"
    json.dump(out_payload, open(out_json, "w"), indent=2, default=str)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    run()
