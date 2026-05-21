#!/usr/bin/env python3 -B
"""Profile baseline trade quality/regime/exit distribution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.v5_backtest import backtest_symbol
from collections import Counter

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
for sym in SYMBOLS:
    r = backtest_symbol(sym, days=180, params=None, verbose=False)
    if r is None or r.get("trades", 0) == 0:
        continue
    exits = Counter(t["exit_reason"] for t in r["details"])
    regimes = Counter(t["regime"] for t in r["details"])
    avg_r_by_regime = {}
    for reg in regimes:
        trs = [t for t in r["details"] if t["regime"] == reg]
        avg_r_by_regime[reg] = round(sum(t["pnl_r"] for t in trs)/len(trs), 2)
    print(f"\n{sym} ({r['trades']} trades, PnL ${r['pnl']:.0f}):")
    print(f"  Exits: {dict(exits)}")
    print(f"  Regimes: {dict(regimes)}  avg_R: {avg_r_by_regime}")
