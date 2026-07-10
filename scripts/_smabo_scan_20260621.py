#!/usr/bin/env python3 -B
"""SMABO baseline scan across all cached symbols.

Runs default SMABO params on each sym, ranks by PF. Identifies which syms
are worth hard-tuning vs which are baseline anti-edge.

Honors SMABO_PARAM_OVERRIDES if present — so XAU/BTC will show the TUNED
results, not defaults.

CLI:
    python3 -B scripts/_smabo_scan_20260621.py [--days N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest.sma_breakout_backtest as bt  # noqa: E402

# All cached syms — names match the cache file stems.
ALL_SYMS = [
    # Live universe (8) — XAU/BTC tuned, EUR blacklisted, others untested
    "XAUUSD", "BTCUSD", "EURUSD", "DJ30.r", "SP500.r",
    "GER40.r", "UK100.r", "US2000.r",
    # Extras with 50K+ bars (cached but not in live SYMBOLS)
    "AUDJPY", "CHFJPY", "ETHUSD", "JPN225ft", "NAS100.r",
    "SPI200.r", "USDCAD", "USOUSD", "XAGUSD", "XPTUSD.r",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--out", type=str,
                        default="scripts/_smabo_scan_20260621.json")
    args = parser.parse_args()

    print(f"SMABO BASELINE SCAN — {args.days}d window — {len(ALL_SYMS)} syms")
    print("(note: XAU/BTC use SHIPPED tuned params from SMABO_PARAM_OVERRIDES)\n")
    print(f"{'Symbol':<12} {'Bars':>6} {'Trd':>5} {'WR':>6} {'PF':>6} "
          f"{'totalR':>9} {'DD%':>6} {'verdict'}")
    print("-" * 90)

    results = []
    t0 = time.time()
    for sym in ALL_SYMS:
        try:
            r, _ = bt.backtest_symbol(sym, days=args.days)
        except Exception as e:
            print(f"{sym:<12} ERROR: {e}")
            continue
        if r.get("status") != "OK":
            print(f"{sym:<12} {r.get('status','??')}")
            results.append({**r, "verdict": "NO_DATA"})
            continue
        pf = r["pf"]; tr = r["total_R"]; dd = r["max_dd_pct"]
        n = r["trades"]
        if pf >= 1.30 and dd < 25:
            verdict = "DEPLOY"
        elif pf >= 1.10 and tr > 0:
            verdict = "STRONG_LEAD"
        elif pf >= 1.00 and tr > 0:
            verdict = "MARGINAL"
        elif pf >= 0.90:
            verdict = "TUNE_CANDIDATE"
        else:
            verdict = "ANTI_EDGE"
        print(f"{sym:<12} {r['bars']:>6} {n:>5} {r['wr']*100:>5.1f}% "
              f"{pf:>6.2f} {tr:>+9.1f} {dd:>5.1f}%  {verdict}")
        results.append({**r, "verdict": verdict})

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # Save + summary
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {args.out}\n")

    # Rank
    valid = [r for r in results if r.get("status") == "OK"]
    valid.sort(key=lambda r: r["pf"], reverse=True)
    print("═══ RANKED LEADERBOARD ═══")
    for r in valid:
        sym = r["symbol"]; pf = r["pf"]; tr = r["total_R"]
        n = r["trades"]; dd = r["max_dd_pct"]; v = r["verdict"]
        print(f"  {sym:<12} PF {pf:>5.2f}  R {tr:>+8.1f}  trd {n:>4}  DD {dd:>5.1f}%  → {v}")

    # Recommendations
    tune_candidates = [r["symbol"] for r in valid
                       if r["verdict"] in ("STRONG_LEAD", "MARGINAL", "TUNE_CANDIDATE")
                       and r["symbol"] not in ("XAUUSD", "BTCUSD")]  # already tuned
    print(f"\n→ HARD-TUNE TARGETS: {tune_candidates}")


if __name__ == "__main__":
    main()
