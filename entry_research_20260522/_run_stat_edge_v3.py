#!/usr/bin/env python3 -B
"""V3 — evaluate ship rule at the PORTFOLIO level.

Walk-forward 5-fold across the combined trade timeline. Train (cell map)
built from each symbol's first k/5 of trades chronologically; test is
the next 1/5 slice across ALL symbols pooled.

Ship rule (portfolio): Δ ≥ $30 AND WF avg PF > 1.5 AND ≥3/5 folds positive.
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

import backtest.v5_backtest as v5
from backtest.v5_backtest import load_data  # noqa: E402
import inspect

SYMBOLS = [
    "DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY",
    "EURUSD", "US2000.r", "UKOUSD", "JPN225ft",
]
DAYS = 360

VARIANTS = [
    ("WL_n15_pf2.0",  "wl", 15, 2.0),
    ("BL_n15_pf1.0",  "bl", 15, 1.0),
    ("BL_n10_pf1.0",  "bl", 10, 1.0),
    ("BL_n10_pf0.8",  "bl", 10, 0.8),
    ("BL_n12_pf0.9",  "bl", 12, 0.9),
    ("BL_n20_pf1.2",  "bl", 20, 1.2),
]

OUT_DIR = Path(__file__).resolve().parent

# patched backtest_symbol
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


def apply_filter(trades, cell_maps_by_sym, kind, n_min, pf_max=None, pf_min=None):
    """kind: 'bl' (block cells with n>=n_min AND pf<pf_max)
            'wl' (allow only cells with n>=n_min AND pf>=pf_min)"""
    keep = []
    for t in trades:
        cm = cell_maps_by_sym.get(t["sym"], {})
        c = cm.get((t["hour"], t["dow"]))
        if kind == "bl":
            if c is None:
                keep.append(t); continue
            if c["n"] >= n_min and c["pf"] < pf_max:
                continue
            keep.append(t)
        else:  # wl
            if c is None:
                continue  # unknown cell → block
            if c["n"] >= n_min and c["pf"] >= pf_min:
                keep.append(t)
    return keep


def main():
    print("Loading 360d trades for 8 symbols...")
    sym_trades = {}
    for sym in SYMBOLS:
        r = patched(sym, days=DAYS, params=None, verbose=False)
        if r is None or r.get("trades", 0) == 0:
            continue
        df = load_data(sym, DAYS)
        if df is None:
            continue
        times = df["time"].values
        n = len(times)
        trades = []
        for t in r["_trades"]:
            eb = t.get("entry_bar")
            if eb is None or eb >= n:
                continue
            ts = pd.Timestamp(times[eb])
            t2 = dict(t)
            t2["sym"]  = sym
            t2["hour"] = ts.hour
            t2["dow"]  = int(ts.dayofweek)
            t2["ts"]   = ts
            trades.append(t2)
        sym_trades[sym] = trades
        print(f"  {sym:12s}  n={len(trades)}")

    # Pool trades for portfolio WF, but cell maps are still PER-SYMBOL.
    all_trades = []
    for ts in sym_trades.values():
        all_trades.extend(ts)
    all_trades.sort(key=lambda x: x["ts"])
    N = len(all_trades)
    print(f"\nPortfolio total trades: {N}")

    results = {}
    for label, kind, n_min, threshold in VARIANTS:
        # Full set in-sample
        cell_maps_full = {sym: build_cell_map(ts) for sym, ts in sym_trades.items()}
        base = trades_metrics(all_trades)
        if kind == "bl":
            after = apply_filter(all_trades, cell_maps_full, "bl", n_min, pf_max=threshold)
        else:
            after = apply_filter(all_trades, cell_maps_full, "wl", n_min, pf_min=threshold)
        after_m = trades_metrics(after)
        full_delta = round(after_m["pnl"] - base["pnl"], 2)

        # WF 5-fold portfolio
        folds = []
        chunk = N // 5
        for k in range(1, 5):
            train = all_trades[: k * chunk]
            test  = all_trades[k * chunk:(k + 1) * chunk]
            # per-symbol cell maps from train slice
            train_by_sym = defaultdict(list)
            for t in train:
                train_by_sym[t["sym"]].append(t)
            cm_train = {sym: build_cell_map(ts) for sym, ts in train_by_sym.items()}
            tb = trades_metrics(test)
            if kind == "bl":
                ta = apply_filter(test, cm_train, "bl", n_min, pf_max=threshold)
            else:
                ta = apply_filter(test, cm_train, "wl", n_min, pf_min=threshold)
            tam = trades_metrics(ta)
            folds.append({
                "fold": k,
                "test_n": tb["n"], "after_n": tam["n"],
                "test_pnl": tb["pnl"], "after_pnl": tam["pnl"],
                "after_pf": tam["pf"],
                "delta": round(tam["pnl"] - tb["pnl"], 2),
            })

        deltas = [f["delta"] for f in folds]
        pfs    = [f["after_pf"] for f in folds if f["after_n"] > 0]
        wf_avg_delta = round(float(np.mean(deltas)), 2)
        wf_avg_pf    = round(float(np.mean(pfs)) if pfs else 0.0, 2)
        pos_folds    = sum(1 for d in deltas if d > 0)

        ship = (full_delta >= 30
                and wf_avg_pf > 1.5
                and pos_folds >= 3)

        print(f"\n{label:16s}  n_min={n_min}  thr={threshold}")
        print(f"  in-sample:  base PnL=${base['pnl']:>10,.2f}  "
              f"after PnL=${after_m['pnl']:>10,.2f}  delta=${full_delta:+,.2f}")
        print(f"  cells affected: removed {base['n']-after_m['n']} of {base['n']} trades")
        print(f"  WF 4-fold: avgΔ=${wf_avg_delta:+,.2f}  avgPF={wf_avg_pf:.2f}  "
              f"pos={pos_folds}/4")
        for f in folds:
            print(f"    fold{f['fold']}: test n={f['test_n']:4d} → "
                  f"after n={f['after_n']:4d}  Δ=${f['delta']:>+8.2f}  "
                  f"pf={f['after_pf']:.2f}")
        print(f"  ship={ship}  (Δ>=30: {full_delta>=30}, "
              f"WF_PF>1.5: {wf_avg_pf>1.5}, pos>=3: {pos_folds>=3})")
        results[label] = {
            "n_min": n_min, "threshold": threshold,
            "base_pnl": base["pnl"], "after_pnl": after_m["pnl"],
            "delta": full_delta,
            "in_sample_n_removed": base["n"] - after_m["n"],
            "wf_avg_delta": wf_avg_delta,
            "wf_avg_pf":    wf_avg_pf,
            "wf_pos_folds": pos_folds, "wf_n_folds": len(folds),
            "wf_folds":     folds,
            "ship":         ship,
        }

    out = OUT_DIR / "10_stat_edge_portfolio_wf.json"
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
