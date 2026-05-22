#!/usr/bin/env python3 -B
"""Smoke test the entry tuner _bt_one worker."""
import sys
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audit_20260522"))

from per_sym_entry_tune import _bt_one, _bt_baseline

# Test 1: baseline
print("baseline XAUUSD:", _bt_baseline("XAUUSD"))

# Test 2: dir_bias overlay
args = ("XAUUSD", {"low_vol": "LONG"}, None, set(), None, None)
print("dir_bias low_vol=LONG:", _bt_one(args))

# Test 3: min_q overlay
args = ("XAUUSD", {}, {"low_vol": 60}, set(), None, None)
print("min_q low_vol=60:", _bt_one(args))

# Test 4: toxic overlay
args = ("XAUUSD", {}, None, {17, 18}, None, None)
print("toxic h17_18:", _bt_one(args))

# Test 5: range filter overlay
args = ("XAUUSD", {}, None, set(), (72, 0.7), None)
print("range_filter (72, 0.7):", _bt_one(args))

# Test 6: WF fold 1 (XAUUSD has fold_d=5 now)
for sym in ["XAUUSD", "SP500.r", "USDJPY"]:
    for fold in [1, 3, 5]:
        args = (sym, {}, None, set(), None, fold)
        r = _bt_one(args)
        if "err" in r:
            print(f"WF {sym} fold {fold}: ERR={r['err'][:80]}")
        else:
            print(f"WF {sym} fold {fold}: trades={r['trades']:3d} PnL=${r['pnl']:+.2f} PF={r['pf']:.2f}")
