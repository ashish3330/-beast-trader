#!/usr/bin/env python3 -B
"""Iter 2: pullback as FILL improvement (not signal gate)."""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from entry_research_20260522.pullback_bt_v2 import backtest_v2

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
DAYS = 180
VARIANTS = [
    "baseline",
    "deep_03_3bar", "deep_05_3bar", "deep_05_5bar", "deep_05_8bar",
    "deep_07_5bar", "deep_07_8bar", "deep_10_5bar", "deep_10_8bar",
    "fib50_fill", "fib50_fill_5b",
    "ema20_fill", "ema20_fill_5b",
    "pullback_size_boost", "pullback_size_boost_hi",
    "no_fallback_05_5b", "no_fallback_07_5b",
]

results = {}
t0 = time.time()
for variant in VARIANTS:
    print(f"\n=== {variant} ===")
    results[variant] = {}
    for sym in SYMBOLS:
        r = backtest_v2(sym, days=DAYS, params={"pullback_mode": variant})
        if r is None:
            results[variant][sym] = {"error": "no_data"}; continue
        results[variant][sym] = {"trades": r["trades"], "pf": float(r["pf"]),
                                  "wr": float(r["wr"]), "pnl": float(r["pnl"]),
                                  "dd": float(r["dd"]), "pb_rate": float(r["pb_rate"])}
        print(f"  {sym:12s} trades={r['trades']:4d} pf={r['pf']:5.2f} wr={r['wr']:5.1f}% "
              f"pnl=${r['pnl']:9.2f} pb={r['pb_rate']:.0f}%")

baseline = results["baseline"]
total_baseline_pnl = sum(baseline[s]["pnl"] for s in SYMBOLS if "pnl" in baseline[s])
print(f"\n{'='*70}")
print(f"BASELINE TOTAL: ${total_baseline_pnl:.2f}")
print(f"{'='*70}")
print(f"{'Variant':<26} {'Total':<12} {'Δ':<12} {'Wins ≥+$30':<12}")
for variant in VARIANTS:
    if variant == "baseline": continue
    total = sum(results[variant][s]["pnl"] for s in SYMBOLS if "pnl" in results[variant][s])
    delta = total - total_baseline_pnl
    wins = sum(1 for s in SYMBOLS if "pnl" in results[variant][s] and
               (results[variant][s]["pnl"] - baseline[s]["pnl"]) >= 30)
    print(f"{variant:<26} ${total:<11.2f} ${delta:<+11.2f} {wins}/{len(SYMBOLS)}")

out = Path(__file__).parent / "iter2_results.json"
json.dump({"days": DAYS, "results": results, "elapsed": round(time.time() - t0, 1)},
          open(out, "w"), indent=2, default=float)
print(f"\nwrote {out} ({time.time() - t0:.1f}s)")
