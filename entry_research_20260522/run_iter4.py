#!/usr/bin/env python3 -B
"""Iter 4: stress test with REALISTIC fallback (next_open, not stale c[i]).

If the edge persists under realistic fallback, it's a real edge.
If the edge disappears, it was data-mining the stale c[i] fallback.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from entry_research_20260522.pullback_bt_v2 import backtest_v2, MODE_CONFIG
import entry_research_20260522.pullback_bt_v2 as _pbv2

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

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
DAYS = 180
VARIANTS = ["baseline", "deep_05_3bar", "deep_07_5bar", "deep_08_5bar", "deep_08_8bar"]

print("=" * 70)
print("WITH REALISTIC FALLBACK (next_open):")
print("=" * 70)
results = {}
for variant in VARIANTS:
    print(f"\n=== {variant} ===")
    results[variant] = {}
    for sym in SYMBOLS:
        r = _pbv2.backtest_v2(sym, days=DAYS, params={
            "pullback_mode": variant, "fallback": "next_open"
        })
        if r is None: continue
        results[variant][sym] = {"trades": r["trades"], "pf": float(r["pf"]),
                                  "pnl": float(r["pnl"]), "pb_rate": float(r["pb_rate"])}
        print(f"  {sym:12s} trades={r['trades']:4d} pf={r['pf']:5.2f} pnl=${r['pnl']:9.2f} pb={r['pb_rate']:.0f}%")

base = results["baseline"]
base_total = sum(base[s]["pnl"] for s in SYMBOLS if s in base)
print(f"\n{'='*70}")
print(f"BASELINE (next_open fallback) TOTAL: ${base_total:.2f}")
print(f"{'='*70}")
print(f"{'Variant':<18} {'Total':<14} {'Δ vs base':<14} {'Wins ≥+$30':<12}")
for variant in VARIANTS:
    if variant == "baseline": continue
    total = sum(results[variant][s]["pnl"] for s in SYMBOLS if s in results[variant])
    delta = total - base_total
    wins = sum(1 for s in SYMBOLS if s in results[variant] and s in base and
               (results[variant][s]["pnl"] - base[s]["pnl"]) >= 30)
    print(f"{variant:<18} ${total:<13.2f} ${delta:<+13.2f} {wins}/{len(SYMBOLS)}")

out = Path(__file__).parent / "iter4_results.json"
json.dump({"days": DAYS, "results": results}, open(out, "w"), indent=2, default=float)
print(f"\nwrote {out}")
