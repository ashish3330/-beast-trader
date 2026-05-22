#!/usr/bin/env python3 -B
"""Check cache depth per symbol."""
import sys
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))
from backtest.v5_backtest import load_data
for sym in ["XAUUSD", "SP500.r", "USDJPY"]:
    df = load_data(sym, days=None)
    if df is None or df.empty:
        print(f"{sym}: NO DATA")
        continue
    print(f"{sym}: n={len(df)} from {df['time'].min()} to {df['time'].max()} ({(df['time'].max()-df['time'].min()).days}d span)")
