"""Smoke test for usousd_runner worker plumbing."""
import sys
sys.path.insert(0, '/Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/_work')
from usousd_runner import _bt_worker, TRAIL_PROFILES

tests = [
    ("baseline", None, {}),
    ("sl_test",  None, {"sl_mult": 2.0}),
    ("vwap_dis", None, {"vwap_buf": "disabled"}),
    ("vwap_07",  None, {"vwap_buf": 0.7}),
    ("combo",    None, {
        "sl_mult": 2.0, "force_trail": TRAIL_PROFILES["WIDE_RUNNER"],
        "pb_atr": 0.5, "pb_wait": 4, "vwap_buf": 0.3, "min_quality": 33}),
    ("fold5",    5,    {"sl_mult": 2.0}),
    ("fold1",    1,    {}),
    ("trail",    None, {"force_trail": TRAIL_PROFILES["TIGHT_LOCK"]}),
    ("mq30",     None, {"min_quality": 30}),
    ("pb_combo", None, {"pb_atr": 1.0, "pb_wait": 8}),
]
for tag, fold, ovr in tests:
    t, r = _bt_worker((tag, fold, ovr))
    if "error" in r:
        print(f"FAIL {tag}: {r['error'][:120]}")
    else:
        print(f"OK   {tag:9}: pnl=${r['pnl']:+.0f} pf={r['pf']:.2f} n={r['trades']:3d} t={r['took']}s")
