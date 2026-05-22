#!/usr/bin/env python3 -B
"""Smoke test: load the patched v5_backtest module and run 30d on XAUUSD + EURUSD
to verify the POST_BIG_WIN_COOLDOWN patch doesn't break anything.

Compares against original v5_backtest run with same params.
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import importlib.util


def _load(label, path):
    spec = importlib.util.spec_from_file_location(label, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


print("Loading PATCHED backtest module …")
patched = _load("v5_backtest_patched",
                os.path.join(ROOT, "audit_20260522", "_v5_backtest_patched.py"))

print("Loading ORIGINAL backtest module …")
original = _load("v5_backtest_original",
                 os.path.join(ROOT, "backtest", "v5_backtest.py"))

# Force POST_BIG_WIN_COOLDOWN_ENABLED=True for the smoke test even if config
# default is False. We do this AFTER module load so the imports already happened.
# Both modules read config at function entry, so we monkey-patch sys.modules['config'].
import config as _cfg
_cfg.POST_BIG_WIN_COOLDOWN_ENABLED = True


def _summarize(label, r):
    if not r:
        print(f"  {label}: NO RESULT")
        return
    print(f"  {label}: trades={r['trades']:3d}  PF={r['pf']:.2f}  "
          f"WR={r['wr']:.1f}%  PnL=${r['pnl']:+.2f}  DD={r['dd']:.1f}%")


for sym in ["XAUUSD", "EURUSD"]:
    print(f"\n=== {sym} (30 days) ===")
    r_orig = original.backtest_symbol(sym, days=30, verbose=False)
    r_patch = patched.backtest_symbol(sym, days=30, verbose=False)
    _summarize("ORIGINAL", r_orig)
    _summarize("PATCHED ", r_patch)
    if r_orig and r_patch:
        delta_n = r_patch["trades"] - r_orig["trades"]
        delta_pnl = r_patch["pnl"] - r_orig["pnl"]
        print(f"  delta:    Δtrades={delta_n:+d}  ΔPnL=${delta_pnl:+.2f}")
        if r_patch["trades"] > r_orig["trades"]:
            print(f"  WARN: patched has MORE trades than original — "
                  f"cooldown should reduce count")

print("\nSMOKE TEST COMPLETE")
