#!/usr/bin/env python3 -B
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/ashish/Documents/beast-trader")))
sys.path.insert(0, str(Path("/Users/ashish/Documents/beast-trader/audit_20260522")))
from per_sym_entry_tune import _bt_one
# Run all 21 XAUUSD candidates serially
from per_sym_entry_tune import _gen_candidates
cands = _gen_candidates("XAUUSD")
for i, c in enumerate(cands):
    args = ("XAUUSD", c["dir_bias_cell"], c["min_q_cell"], c["toxic_set"], c["range_filter"], None)
    r = _bt_one(args)
    if "err" in r:
        print(f"{i:2d}. {c['tag']:38s} ERR: {r['err'][:100]}")
    else:
        print(f"{i:2d}. {c['tag']:38s} trades={r['trades']:3d} PnL=${r['pnl']:+.0f} PF={r['pf']:.2f}")
