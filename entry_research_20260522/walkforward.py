#!/usr/bin/env python3 -B
"""5-fold walk-forward validation for top pullback variants.

Splits 180 days into 5 contiguous folds of 36 days each. Runs each variant
on each fold independently — checks stability across periods, not just
in-sample fit.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Patch v5_backtest.load_data to filter to a date range
import backtest.v5_backtest as _v5
_orig_load = _v5.load_data

class _Fold:
    start_off = None
    end_off = None

def _patched_load(symbol, days=90):
    df = _orig_load(symbol, days=180)  # always load 180 days
    if df is None:
        return None
    if _Fold.start_off is None or _Fold.end_off is None:
        return df
    tmax = df["time"].max()
    cutoff_end = tmax - pd.Timedelta(days=_Fold.start_off)
    cutoff_start = tmax - pd.Timedelta(days=_Fold.end_off)
    df = df[(df["time"] >= cutoff_start) & (df["time"] <= cutoff_end)].reset_index(drop=True)
    return df

_v5.load_data = _patched_load

# Also patch the imported reference in pullback_bt_v2
from entry_research_20260522 import pullback_bt_v2 as _pbv2
_pbv2.load_data = _patched_load
from entry_research_20260522.pullback_bt_v2 import backtest_v2

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
# Top variants to walk-forward
VARIANTS = ["baseline", "deep_05_3bar", "deep_05_5bar", "deep_07_5bar",
            "deep_07_8bar", "deep_03_3bar"]

# 5 folds: each 36 days, going from most-recent-back
FOLDS = [
    ("F1_recent", 0, 36),      # most recent 36 days
    ("F2", 36, 72),
    ("F3", 72, 108),
    ("F4", 108, 144),
    ("F5_oldest", 144, 180),
]

all_results = {}
t0 = time.time()
for variant in VARIANTS:
    print(f"\n=== {variant} ===")
    all_results[variant] = {}
    for fold_name, start_off, end_off in FOLDS:
        _Fold.start_off = start_off
        _Fold.end_off = end_off
        fold_total = 0.0
        per_sym = {}
        for sym in SYMBOLS:
            r = backtest_v2(sym, days=180, params={"pullback_mode": variant})
            if r is None:
                per_sym[sym] = {"error": "no_data"}; continue
            per_sym[sym] = {"trades": r["trades"], "pf": float(r["pf"]),
                            "pnl": float(r["pnl"])}
            fold_total += r["pnl"]
        all_results[variant][fold_name] = {"total_pnl": round(fold_total, 2),
                                           "symbols": per_sym}
        print(f"  {fold_name}: ${fold_total:.2f}")

# Summary: per-variant per-fold total
print(f"\n{'='*70}")
print(f"{'Variant':<22}", end="")
for f in FOLDS:
    print(f"{f[0]:>13}", end="")
print(f"{'AvgPF≥1.5?':>14}")
for variant in VARIANTS:
    print(f"{variant:<22}", end="")
    pfs_sum = []
    for f in FOLDS:
        v = all_results[variant][f[0]]["total_pnl"]
        print(f"{v:>12.2f}", end=" ")
    # Compute avg PF across folds & syms
    pf_vals = []
    for f in FOLDS:
        for sym in SYMBOLS:
            d = all_results[variant][f[0]]["symbols"].get(sym, {})
            if isinstance(d, dict) and d.get("trades", 0) >= 5:
                pf_vals.append(d["pf"])
    avg_pf = np.mean(pf_vals) if pf_vals else 0
    print(f"   {avg_pf:.2f}")

out = Path(__file__).parent / "walkforward_results.json"
json.dump(all_results, open(out, "w"), indent=2, default=float)
print(f"\nwrote {out} ({time.time()-t0:.1f}s)")
