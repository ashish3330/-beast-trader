#!/usr/bin/env python3 -B
"""Final WF on top 8 + 10 OOS, deep_08_5bar specifically (the winner)."""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest.v5_backtest as _v5
_orig_load = _v5.load_data
class _Fold:
    start_off = None; end_off = None
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
MODE_CONFIG["deep_07_5bar"] = (5, True, 1.0)
_orig_tgt = _pbv2._entry_target
def _ext_tgt(mode, direction, c, h, l, bi, ema20_s, ema50_s, ind, atr, p):
    close_now = float(c[bi])
    if mode == "deep_08_5bar":
        retr = atr * 0.8
        return close_now - retr if direction == 1 else close_now + retr
    return _orig_tgt(mode, direction, c, h, l, bi, ema20_s, ema50_s, ind, atr, p)
_pbv2._entry_target = _ext_tgt
backtest_v2 = _pbv2.backtest_v2

# All 18 live symbols (8 IS + 10 OOS)
ALL_LIVE = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft",
            "BTCUSD", "ETHUSD", "USDCAD", "GBPUSD", "GBPJPY", "USDJPY",
            "NAS100.r", "GER40.r", "SP500.r", "XAGUSD"]
VARIANTS = ["baseline", "deep_07_5bar", "deep_08_5bar"]
FOLDS = [("F1_recent", 0, 36), ("F2", 36, 72), ("F3", 72, 108), ("F4", 108, 144), ("F5_oldest", 144, 180)]

all_results = {}
t0 = time.time()
for variant in VARIANTS:
    all_results[variant] = {}
    for fold_name, start_off, end_off in FOLDS:
        _Fold.start_off = start_off; _Fold.end_off = end_off
        fold_total = 0.0; per_sym = {}
        for sym in ALL_LIVE:
            r = backtest_v2(sym, days=180, params={"pullback_mode": variant})
            if r is None: continue
            per_sym[sym] = {"trades": r["trades"], "pf": float(r["pf"]), "pnl": float(r["pnl"])}
            fold_total += r["pnl"]
        all_results[variant][fold_name] = {"total_pnl": round(fold_total, 2), "symbols": per_sym}

# Summary
print(f"{'Variant':<22}", end="")
for f in FOLDS: print(f"{f[0]:>13}", end="")
print(f"{'Total':>13}")
for variant in VARIANTS:
    print(f"{variant:<22}", end="")
    total = 0
    for f in FOLDS:
        v = all_results[variant][f[0]]["total_pnl"]
        total += v
        print(f"{v:>12.2f}", end=" ")
    print(f"  {total:>10.2f}")

# Per-symbol per-fold wins
print("\n--- Per-symbol fold-wins (deep_08_5bar vs baseline) ---")
for sym in ALL_LIVE:
    wins = 0; data = 0
    for f, _, _ in FOLDS:
        vd = all_results["deep_08_5bar"][f]["symbols"].get(sym, {})
        bd = all_results["baseline"][f]["symbols"].get(sym, {})
        if isinstance(vd, dict) and "pnl" in vd and isinstance(bd, dict) and "pnl" in bd:
            data += 1
            if vd["pnl"] - bd["pnl"] > 0: wins += 1
    print(f"  {sym:<12} {wins}/{data}")

out = Path(__file__).parent / "walkforward_final.json"
json.dump(all_results, open(out, "w"), indent=2, default=float)
print(f"\nwrote {out} ({time.time()-t0:.1f}s)")
