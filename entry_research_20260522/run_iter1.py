#!/usr/bin/env python3 -B
"""Iter 1: run baseline + 5 standard variants across top 8 symbols, 180d."""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from entry_research_20260522.pullback_bt import backtest_with_pullback

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
DAYS = 180
VARIANTS = ["baseline", "fib_50", "fib_618", "ema20", "ema50", "bb_mid", "order_block", "trendline"]

results = {}
t0 = time.time()
for variant in VARIANTS:
    print(f"\n=== {variant} ===")
    results[variant] = {}
    for sym in SYMBOLS:
        r = backtest_with_pullback(sym, days=DAYS, params={"pullback_mode": variant})
        if r is None:
            results[variant][sym] = {"error": "no_data"}; continue
        results[variant][sym] = {"trades": r["trades"], "pf": float(r["pf"]),
                                  "wr": float(r["wr"]), "pnl": float(r["pnl"]), "dd": float(r["dd"])}
        print(f"  {sym:12s} trades={r['trades']:4d} pf={r['pf']:5.2f} wr={r['wr']:5.1f}% pnl=${r['pnl']:9.2f}")

# Compute deltas
baseline = results["baseline"]
total_baseline_pnl = sum(baseline[s]["pnl"] for s in SYMBOLS if "pnl" in baseline[s])
print(f"\n{'='*60}")
print(f"BASELINE TOTAL: ${total_baseline_pnl:.2f}")
print(f"{'='*60}")
print(f"{'Variant':<15} {'Total':<12} {'Δ':<12} {'Symbols ≥+$30':<10}")
for variant in VARIANTS:
    if variant == "baseline": continue
    total = sum(results[variant][s]["pnl"] for s in SYMBOLS if "pnl" in results[variant][s])
    delta = total - total_baseline_pnl
    wins = sum(1 for s in SYMBOLS if "pnl" in results[variant][s] and
               (results[variant][s]["pnl"] - baseline[s]["pnl"]) >= 30)
    print(f"{variant:<15} ${total:<11.2f} ${delta:<+11.2f} {wins}/{len(SYMBOLS)}")

out = Path(__file__).parent / "iter1_results.json"
json.dump({"days": DAYS, "results": results, "elapsed": round(time.time() - t0, 1)},
          open(out, "w"), indent=2, default=float)
print(f"\nwrote {out} ({time.time() - t0:.1f}s)")
