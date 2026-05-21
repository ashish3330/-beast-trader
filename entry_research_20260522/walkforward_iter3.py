#!/usr/bin/env python3 -B
"""Walk-forward validation for iter3 top candidates."""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest.v5_backtest as _v5
_orig_load = _v5.load_data
class _Fold:
    start_off = None
    end_off = None
def _patched_load(symbol, days=90):
    df = _orig_load(symbol, days=180)
    if df is None: return None
    if _Fold.start_off is None or _Fold.end_off is None: return df
    tmax = df["time"].max()
    cutoff_end = tmax - pd.Timedelta(days=_Fold.start_off)
    cutoff_start = tmax - pd.Timedelta(days=_Fold.end_off)
    return df[(df["time"] >= cutoff_start) & (df["time"] <= cutoff_end)].reset_index(drop=True)
_v5.load_data = _patched_load

from entry_research_20260522 import pullback_bt_v2 as _pbv2
_pbv2.load_data = _patched_load
MODE_CONFIG = _pbv2.MODE_CONFIG
MODE_CONFIG["deep_08_5bar"] = (5, True, 1.0)
MODE_CONFIG["deep_08_8bar"] = (8, True, 1.0)
MODE_CONFIG["deep_06_5bar"] = (5, True, 1.0)

_orig_tgt = _pbv2._entry_target
def _ext_tgt(mode, direction, c, h, l, bi, ema20_s, ema50_s, ind, atr, p):
    close_now = float(c[bi])
    if mode in ("deep_06_5bar", "deep_06_8bar"):
        retr = atr * 0.6
        return close_now - retr if direction == 1 else close_now + retr
    if mode in ("deep_08_5bar", "deep_08_8bar"):
        retr = atr * 0.8
        return close_now - retr if direction == 1 else close_now + retr
    return _orig_tgt(mode, direction, c, h, l, bi, ema20_s, ema50_s, ind, atr, p)
_pbv2._entry_target = _ext_tgt

backtest_v2 = _pbv2.backtest_v2

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
VARIANTS = ["baseline", "deep_05_3bar", "deep_07_5bar", "deep_08_5bar", "deep_08_8bar", "deep_07_8bar"]
FOLDS = [("F1_recent", 0, 36), ("F2", 36, 72), ("F3", 72, 108), ("F4", 108, 144), ("F5_oldest", 144, 180)]

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
            per_sym[sym] = {"trades": r["trades"], "pf": float(r["pf"]), "pnl": float(r["pnl"])}
            fold_total += r["pnl"]
        all_results[variant][fold_name] = {"total_pnl": round(fold_total, 2), "symbols": per_sym}
        print(f"  {fold_name}: ${fold_total:.2f}")

# Summary
print(f"\n{'='*70}")
print(f"{'Variant':<22}", end="")
for f in FOLDS: print(f"{f[0]:>13}", end="")
print(f"{'AvgPF':>8}")
for variant in VARIANTS:
    print(f"{variant:<22}", end="")
    for f in FOLDS:
        v = all_results[variant][f[0]]["total_pnl"]
        print(f"{v:>12.2f}", end=" ")
    pf_vals = []
    for f in FOLDS:
        for sym in SYMBOLS:
            d = all_results[variant][f[0]]["symbols"].get(sym, {})
            if isinstance(d, dict) and d.get("trades", 0) >= 5:
                pf_vals.append(d["pf"])
    avg_pf = np.mean(pf_vals) if pf_vals else 0
    print(f"  {avg_pf:.2f}")

# Per-symbol per-fold delta
baseline = all_results["baseline"]
print("\n--- Per-symbol Δ pass rate (folds positive vs baseline) ---")
for variant in VARIANTS:
    if variant == "baseline": continue
    print(f"\n{variant}:")
    for sym in SYMBOLS:
        folds_pos = 0; folds_with_data = 0
        for f, _, _ in FOLDS:
            vd = all_results[variant][f]["symbols"].get(sym, {})
            bd = baseline[f]["symbols"].get(sym, {})
            if isinstance(vd, dict) and isinstance(bd, dict) and "pnl" in vd and "pnl" in bd:
                folds_with_data += 1
                if vd["pnl"] - bd["pnl"] > 0:
                    folds_pos += 1
        print(f"  {sym:<12} {folds_pos}/{folds_with_data} folds positive")

out = Path(__file__).parent / "walkforward_iter3.json"
json.dump(all_results, open(out, "w"), indent=2, default=float)
print(f"\nwrote {out} ({time.time()-t0:.1f}s)")
