#!/usr/bin/env python3 -B
"""Show BT regime distribution + per-(dir, regime) PnL contribution for the 3 targets."""
import sys
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))
import numpy as np
from collections import Counter, defaultdict
from backtest.v5_backtest import (
    load_data, _compute_indicators, get_regime,
    IND_DEFAULTS, IND_OVERRIDES, backtest_symbol,
)

# For each symbol, show regime distribution
for sym in ["XAUUSD", "SP500.r", "USDJPY"]:
    df = load_data(sym, days=180)
    if df is None or df.empty:
        print(f"{sym}: no data")
        continue
    icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(sym, {})}
    ind = _compute_indicators(df, icfg)
    cnt = Counter()
    for i in range(100, len(df) - 1):
        bbw = float(ind["bbw"][i]) if not np.isnan(ind["bbw"][i]) else 0.02
        adx = float(ind["adx"][i]) if not np.isnan(ind["adx"][i]) else 20
        cnt[get_regime(bbw, adx)] += 1
    print(f"{sym} regime distribution (180d): {dict(cnt)}")

# Run BT and get per-trade dir+regime breakdown by injecting a trade collector
# (we can use the result trade list)
print("\n--- Per-trade BT regime/direction breakdown ---")
for sym in ["XAUUSD", "SP500.r", "USDJPY"]:
    r = backtest_symbol(sym, days=180, verbose=False)
    if r is None:
        print(f"{sym}: no result")
        continue
    trades = r.get("details", [])
    if not trades:
        print(f"{sym}: trades={r.get('trades')} pnl={r.get('pnl')} (no per-trade list)")
        continue
    buckets = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'r': 0.0, 'wins': 0})
    for t in trades:
        d = t.get('direction')
        if d == 1: d = 'LONG'
        elif d == -1: d = 'SHORT'
        reg = t.get('regime', '?')
        key = (d, reg)
        b = buckets[key]
        b['n'] += 1
        b['pnl'] += t.get('pnl', 0)
        b['r'] += t.get('pnl_r', 0)
        if t.get('pnl', 0) > 0:
            b['wins'] += 1
    print(f"\n=== {sym} BT trades by (dir, regime) ===")
    for k, v in sorted(buckets.items()):
        wr = 100 * v['wins'] / v['n']
        print(f"  {k}: n={v['n']:3d} pnl=${v['pnl']:+7.2f} R={v['r']:+6.2f} WR={wr:.0f}%")
    # Hour breakdown
    hr_buckets = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        h = t.get('hour')
        b = hr_buckets[h]
        b['n'] += 1
        b['pnl'] += t.get('pnl', 0)
        if t.get('pnl', 0) > 0:
            b['wins'] += 1
    print(f"  -- by hour --")
    for h in sorted(hr_buckets.keys()):
        v = hr_buckets[h]
        wr = 100 * v['wins'] / v['n']
        marker = "BAD" if v['pnl'] < -1 else ("ok" if v['pnl'] > 1 else "")
        print(f"  h={h:02d}: n={v['n']:3d} pnl=${v['pnl']:+7.2f} WR={wr:4.0f}% {marker}")
