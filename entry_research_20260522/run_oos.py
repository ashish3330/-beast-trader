#!/usr/bin/env python3 -B
"""OOS test: check the top variant on 10 OTHER live symbols not in our 8."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from entry_research_20260522.pullback_bt_v2 import backtest_v2, MODE_CONFIG
import entry_research_20260522.pullback_bt_v2 as _pbv2

MODE_CONFIG["deep_08_5bar"] = (5, True, 1.0)
MODE_CONFIG["deep_07_5bar"] = (5, True, 1.0)

_orig_tgt = _pbv2._entry_target
def _ext_tgt(mode, direction, c, h, l, bi, ema20_s, ema50_s, ind, atr, p):
    close_now = float(c[bi])
    if mode == "deep_08_5bar":
        retr = atr * 0.8
        return close_now - retr if direction == 1 else close_now + retr
    return _orig_tgt(mode, direction, c, h, l, bi, ema20_s, ema50_s, ind, atr, p)
_pbv2._entry_target = _ext_tgt

# 10 other live symbols not in our top-8
OOS_SYMBOLS = ["BTCUSD", "ETHUSD", "USDCAD", "GBPUSD", "GBPJPY", "USDJPY",
               "NAS100.r", "GER40.r", "SP500.r", "XAGUSD"]
DAYS = 180

print(f"OOS test on {len(OOS_SYMBOLS)} symbols not in our top 8.")
print(f"{'Symbol':<12} {'base_pnl':<10} {'d07_5_pnl':<12} {'d08_5_pnl':<12} {'Δ_d08_5':<10} {'base_pf':<8} {'d08_5_pf':<8}")
print("=" * 80)
base_total = 0; d07_total = 0; d08_total = 0
wins_07 = 0; wins_08 = 0
for sym in OOS_SYMBOLS:
    rb = _pbv2.backtest_v2(sym, days=DAYS, params={"pullback_mode": "baseline"})
    r7 = _pbv2.backtest_v2(sym, days=DAYS, params={"pullback_mode": "deep_07_5bar"})
    r8 = _pbv2.backtest_v2(sym, days=DAYS, params={"pullback_mode": "deep_08_5bar"})
    if not (rb and r7 and r8):
        print(f"{sym:<12} <no data>")
        continue
    base_total += rb["pnl"]; d07_total += r7["pnl"]; d08_total += r8["pnl"]
    delta_8 = r8["pnl"] - rb["pnl"]
    if r7["pnl"] - rb["pnl"] >= 30: wins_07 += 1
    if delta_8 >= 30: wins_08 += 1
    print(f"{sym:<12} ${rb['pnl']:<9.0f} ${r7['pnl']:<11.0f} ${r8['pnl']:<11.0f} ${delta_8:<+9.0f} {rb['pf']:<8.2f} {r8['pf']:<8.2f}")
print("=" * 80)
print(f"{'TOTAL':<12} ${base_total:<9.0f} ${d07_total:<11.0f} ${d08_total:<11.0f}")
print(f"Δ deep_07_5bar: ${d07_total - base_total:+.2f}  wins {wins_07}/{len(OOS_SYMBOLS)}")
print(f"Δ deep_08_5bar: ${d08_total - base_total:+.2f}  wins {wins_08}/{len(OOS_SYMBOLS)}")
