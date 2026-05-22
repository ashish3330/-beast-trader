#!/usr/bin/env python3 -B
"""Verify the shipped auto_tuned.py changes produce the predicted Δ vs prior baseline.

This must be run AFTER auto_tuned.py has been edited.
"""
import sys
import importlib
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import auto_tuned as at; importlib.reload(at)
import config as cfg; importlib.reload(cfg)
import backtest.v5_backtest as bt; importlib.reload(bt)

# Per-symbol days (XAU/JPY have shallow cache)
DAYS = {"XAUUSD": 29, "SP500.r": 180, "USDJPY": 28}

print("=== Verify shipped config ===")
for sym in ["XAUUSD", "SP500.r", "USDJPY"]:
    r = bt.backtest_symbol(sym, days=DAYS[sym], verbose=False)
    print(f"{sym}: trades={r['trades']:3d} PF={r['pf']:6.2f} WR={r['wr']:.1f}% PnL=${r['pnl']:+,.0f} DD={r['dd']:.1f}%")
print()
print("Compare vs tuner-reported numbers:")
print("  XAUUSD predicted stacked: trades=59 PF=7.14 PnL=$+752")
print("  SP500.r predicted stacked: trades=321 PF=8.08 PnL=$+3,117,770")
print("  USDJPY predicted stacked: trades=45 PF=2.93 PnL=$+116")
