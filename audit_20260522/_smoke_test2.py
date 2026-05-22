#!/usr/bin/env python3 -B
"""Smoke test #2 — verify _DIR_BIAS_REGIME_STR overlay actually affects the BT."""
import sys
import importlib
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import config as cfg
importlib.reload(cfg)
import backtest.v5_backtest as bt
importlib.reload(bt)

print("Initial _DIR_BIAS_REGIME_STR for XAUUSD:", bt._DIR_BIAS_REGIME_STR.get("XAUUSD"))
r0 = bt.backtest_symbol("XAUUSD", days=180, verbose=False)
print(f"baseline:        trades={r0['trades']} PnL=${r0['pnl']:+.0f}")

# Force ALL regimes to LONG (block all shorts on XAUUSD)
bt._DIR_BIAS_REGIME_STR["XAUUSD"] = {
    "trending": "LONG", "ranging": "LONG", "volatile": "LONG", "low_vol": "LONG"
}
print("After overlay:", bt._DIR_BIAS_REGIME_STR.get("XAUUSD"))
r1 = bt.backtest_symbol("XAUUSD", days=180, verbose=False)
print(f"all-LONG overlay: trades={r1['trades']} PnL=${r1['pnl']:+.0f}")

# now just low_vol = LONG
bt._DIR_BIAS_REGIME_STR["XAUUSD"] = {"low_vol": "LONG"}
print("After overlay low_vol only:", bt._DIR_BIAS_REGIME_STR.get("XAUUSD"))
r2 = bt.backtest_symbol("XAUUSD", days=180, verbose=False)
print(f"low_vol=LONG:     trades={r2['trades']} PnL=${r2['pnl']:+.0f}")

# Distribution by regime
import sqlite3
print("\nNow looking at signal stats — count regimes in 180d window:")
import pickle
import pandas as pd
import numpy as np
from backtest.v5_backtest import load_data, _compute_indicators, get_regime, IND_DEFAULTS, IND_OVERRIDES
df = load_data("XAUUSD", days=180)
icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get("XAUUSD", {})}
ind = _compute_indicators(df, icfg)
regs = []
for i in range(100, len(df)-1):
    bbw = float(ind["bbw"][i]) if not np.isnan(ind["bbw"][i]) else 0.02
    adx = float(ind["adx"][i]) if not np.isnan(ind["adx"][i]) else 20
    regs.append(get_regime(bbw, adx))
from collections import Counter
print("Regime counts in 180d:", Counter(regs))
