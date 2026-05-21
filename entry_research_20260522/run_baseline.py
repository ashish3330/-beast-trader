#!/usr/bin/env python3 -B
"""Baseline: run v5_backtest.backtest_symbol on top 8 symbols, 180d, no pullback variants."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
DAYS = 180

out = {}
total_pnl = 0.0
total_trades = 0
for sym in SYMBOLS:
    r = backtest_symbol(sym, days=DAYS, params=None, verbose=False)
    if r is None:
        out[sym] = {"error": "no_data"}
        continue
    out[sym] = {"trades": r["trades"], "pf": r["pf"], "wr": r["wr"], "pnl": r["pnl"], "dd": r["dd"]}
    total_pnl += r["pnl"]
    total_trades += r["trades"]
    print(f"  {sym:12s} trades={r['trades']:4d} pf={r['pf']:5.2f} wr={r['wr']:5.1f}% pnl=${r['pnl']:9.2f} dd={r['dd']:.1f}%")

print(f"TOTAL pnl=${total_pnl:.2f} trades={total_trades}")
OUT = Path(__file__).parent / "baseline.json"
json.dump({"days": DAYS, "symbols": out, "total_pnl": round(total_pnl, 2), "total_trades": total_trades}, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
