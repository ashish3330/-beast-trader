#!/usr/bin/env python3 -B
"""Trace SP500.r delta breakdown."""
import sys, importlib
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))
import auto_tuned as at; importlib.reload(at)
import config as cfg; importlib.reload(cfg)
import backtest.v5_backtest as bt; importlib.reload(bt)

print("SIGNAL_QUALITY_SYMBOL[SP500.r]:", cfg.SIGNAL_QUALITY_SYMBOL.get("SP500.r"))
print("RANGE_FILTER_PARAMS_AUTO[SP500.r]:", getattr(at, "RANGE_FILTER_PARAMS_AUTO", {}).get("SP500.r"))
print("TOXIC_HOURS_PER_SYMBOL_AUTO[SP500.r]:", getattr(at, "TOXIC_HOURS_PER_SYMBOL_AUTO", {}).get("SP500.r"))
print()
r = bt.backtest_symbol("SP500.r", days=180, verbose=False)
print(f"BT SP500.r: trades={r['trades']} PF={r['pf']} PnL=${r['pnl']:+,.0f}")

# Test: force min_q={volatile:35} via params
r2 = bt.backtest_symbol("SP500.r", days=180,
                       params={"min_quality": {"trending":28,"ranging":28,"volatile":35,"low_vol":28}},
                       verbose=False)
print(f"BT SP500.r min_q vol=35 via params: trades={r2['trades']} PF={r2['pf']} PnL=${r2['pnl']:+,.0f}")
