#!/usr/bin/env python3 -B
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/ashish/Documents/beast-trader")))
sys.path.insert(0, str(Path("/Users/ashish/Documents/beast-trader/audit_20260522")))
from per_sym_entry_tune import _gen_candidates
for s in ["XAUUSD", "SP500.r", "USDJPY"]:
    c = _gen_candidates(s)
    print(f"{s}: {len(c)} candidates")
    for i, x in enumerate(c):
        print(f"  {i:2d}. {x['tag']}")
