#!/usr/bin/env python3 -B
"""Smoke test: baseline + 1 trial."""
import sys, time
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "per_symbol_tune_20260522" / "_work"))

from tune_ethusd import _baseline_only, _job_runner, TRAIL_PROFILES

print("=== ETHUSD smoke test ===")
t0 = time.time()
print("1. Baseline ...")
base = _baseline_only()
print(f"   baseline={base}   ({time.time()-t0:.1f}s)")

t1 = time.time()
print("\n2. One sample trial ...")
trial = {
    "sl": 1.5, "trail": "_ETH_LIVE", "pb_atr": 0.8, "pb_wait": 5,
    "vwap": "0.5_default", "mq": 33, "dir_trend": "BOTH", "dir_vol": "BOTH",
}
res = _job_runner((0, trial, 180))
print(f"   trial result={res}   ({time.time()-t1:.1f}s)")

t2 = time.time()
print("\n3. Trial with VWAP disabled ...")
trial2 = dict(trial); trial2["vwap"] = "0.0_disabled"
res2 = _job_runner((1, trial2, 180))
print(f"   trial result={res2}   ({time.time()-t2:.1f}s)")

t3 = time.time()
print("\n4. Trial with SHORT trending bias ...")
trial3 = dict(trial); trial3["dir_trend"] = "SHORT"; trial3["dir_vol"] = "LONG"
res3 = _job_runner((2, trial3, 180))
print(f"   trial result={res3}   ({time.time()-t3:.1f}s)")

print(f"\nTotal: {time.time()-t0:.1f}s")
