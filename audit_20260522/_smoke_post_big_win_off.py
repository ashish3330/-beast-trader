#!/usr/bin/env python3 -B
"""Verify patched BT == original BT when POST_BIG_WIN_COOLDOWN_ENABLED=False."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import importlib.util


def _load(label, path):
    spec = importlib.util.spec_from_file_location(label, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


patched = _load("v5_backtest_patched",
                os.path.join(ROOT, "audit_20260522", "_v5_backtest_patched.py"))
original = _load("v5_backtest_original",
                 os.path.join(ROOT, "backtest", "v5_backtest.py"))

import config as _cfg
_cfg.POST_BIG_WIN_COOLDOWN_ENABLED = False

for sym in ["XAUUSD", "EURUSD"]:
    r_o = original.backtest_symbol(sym, days=30, verbose=False)
    r_p = patched.backtest_symbol(sym, days=30, verbose=False)
    same = (
        r_o["trades"] == r_p["trades"]
        and abs(r_o["pnl"] - r_p["pnl"]) < 0.01
        and abs(r_o["pf"] - r_p["pf"]) < 0.01
    )
    print(f"{sym}: orig({r_o['trades']}t/${r_o['pnl']:+.2f}/pf{r_o['pf']:.2f}) "
          f"== patched({r_p['trades']}t/${r_p['pnl']:+.2f}/pf{r_p['pf']:.2f})? "
          f"{'YES' if same else 'NO  <<<<<'}")
