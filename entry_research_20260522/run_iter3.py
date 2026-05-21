#!/usr/bin/env python3 -B
"""Iter 3: tighter variants + sensitivity check.

- Conservative: deep_05_3bar at 0.5 ATR (less price improvement, more fills)
- Investigate: per-symbol best fit
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from entry_research_20260522.pullback_bt_v2 import backtest_v2, MODE_CONFIG

# Test additional intermediate depths
SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
DAYS = 180

# We need to add finer-grained variants. Patch MODE_CONFIG inline.
MODE_CONFIG["deep_04_3bar"] = (3, True, 1.0)
MODE_CONFIG["deep_04_5bar"] = (5, True, 1.0)
MODE_CONFIG["deep_06_5bar"] = (5, True, 1.0)
MODE_CONFIG["deep_06_8bar"] = (8, True, 1.0)
MODE_CONFIG["deep_08_5bar"] = (5, True, 1.0)
MODE_CONFIG["deep_08_8bar"] = (8, True, 1.0)

# Need to add the entry-target lookup for these. Patch pullback_bt_v2._entry_target
import entry_research_20260522.pullback_bt_v2 as _pbv2

_orig_tgt = _pbv2._entry_target
def _extended_tgt(mode, direction, c, h, l, bi, ema20_s, ema50_s, ind, atr, p):
    close_now = float(c[bi])
    if mode == "deep_04_3bar" or mode == "deep_04_5bar":
        retr = atr * 0.4
        return close_now - retr if direction == 1 else close_now + retr
    if mode in ("deep_06_5bar", "deep_06_8bar"):
        retr = atr * 0.6
        return close_now - retr if direction == 1 else close_now + retr
    if mode in ("deep_08_5bar", "deep_08_8bar"):
        retr = atr * 0.8
        return close_now - retr if direction == 1 else close_now + retr
    return _orig_tgt(mode, direction, c, h, l, bi, ema20_s, ema50_s, ind, atr, p)
_pbv2._entry_target = _extended_tgt

VARIANTS = [
    "baseline",
    "deep_04_3bar", "deep_04_5bar",
    "deep_05_3bar", "deep_05_5bar",
    "deep_06_5bar", "deep_06_8bar",
    "deep_07_5bar", "deep_07_8bar",
    "deep_08_5bar", "deep_08_8bar",
]

results = {}
t0 = time.time()
for variant in VARIANTS:
    print(f"\n=== {variant} ===")
    results[variant] = {}
    for sym in SYMBOLS:
        r = _pbv2.backtest_v2(sym, days=DAYS, params={"pullback_mode": variant})
        if r is None:
            results[variant][sym] = {"error": "no_data"}; continue
        results[variant][sym] = {"trades": r["trades"], "pf": float(r["pf"]),
                                  "wr": float(r["wr"]), "pnl": float(r["pnl"]),
                                  "dd": float(r["dd"]), "pb_rate": float(r["pb_rate"])}
        print(f"  {sym:12s} trades={r['trades']:4d} pf={r['pf']:5.2f} pnl=${r['pnl']:9.2f} pb={r['pb_rate']:.0f}%")

baseline = results["baseline"]
total_baseline_pnl = sum(baseline[s]["pnl"] for s in SYMBOLS if "pnl" in baseline[s])
print(f"\n{'='*70}")
print(f"BASELINE TOTAL: ${total_baseline_pnl:.2f}")
print(f"{'='*70}")
print(f"{'Variant':<18} {'Total':<12} {'Δ':<12} {'Wins ≥+$30':<12} avg_pb%")
for variant in VARIANTS:
    if variant == "baseline": continue
    total = sum(results[variant][s]["pnl"] for s in SYMBOLS if "pnl" in results[variant][s])
    delta = total - total_baseline_pnl
    wins = sum(1 for s in SYMBOLS if "pnl" in results[variant][s] and
               (results[variant][s]["pnl"] - baseline[s]["pnl"]) >= 30)
    avg_pb = sum(results[variant][s].get("pb_rate", 0) for s in SYMBOLS if "pb_rate" in results[variant][s])
    avg_pb /= len(SYMBOLS)
    print(f"{variant:<18} ${total:<11.2f} ${delta:<+11.2f} {wins}/{len(SYMBOLS):<6} {avg_pb:5.1f}%")

out = Path(__file__).parent / "iter3_results.json"
json.dump({"days": DAYS, "results": results, "elapsed": round(time.time() - t0, 1)},
          open(out, "w"), indent=2, default=float)
print(f"\nwrote {out} ({time.time() - t0:.1f}s)")
