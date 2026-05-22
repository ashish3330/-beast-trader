#!/usr/bin/env python3 -B
"""Quick analyzer for XAU/SP/JPY journal — entry pattern divergence helper."""
import sqlite3
from collections import Counter
con = sqlite3.connect('data/trade_journal.db')
cur = con.cursor()
for sym in ['XAUUSD', 'SP500.r', 'USDJPY']:
    print(f"\n=== {sym} ALL trades by (dir, regime) since 2026-04-23 ===")
    cur.execute("SELECT direction, regime, pnl, r_multiple FROM trades WHERE symbol=? AND timestamp >= '2026-04-23'", (sym,))
    rows = cur.fetchall()
    buckets = {}
    for d, r, pnl, rm in rows:
        key = (d, r or '')
        b = buckets.setdefault(key, {'n': 0, 'pnl': 0.0, 'r': 0.0, 'wins': 0})
        b['n'] += 1
        b['pnl'] += pnl or 0
        b['r'] += rm or 0
        if (pnl or 0) > 0: b['wins'] += 1
    for k, b in sorted(buckets.items()):
        wr = 100 * b['wins'] / b['n']
        print(f"  {k}: n={b['n']:3d} pnl=${b['pnl']:+7.2f} R={b['r']:+6.2f} WR={wr:4.0f}%")
    print(f"\n  -- by hour --")
    cur.execute("SELECT session_hour, direction, regime, pnl, r_multiple FROM trades WHERE symbol=? AND timestamp >= '2026-04-23'", (sym,))
    hr_buckets = {}
    for h, d, r, pnl, rm in cur.fetchall():
        b = hr_buckets.setdefault(h, {'n': 0, 'pnl': 0, 'wins': 0})
        b['n'] += 1
        b['pnl'] += pnl or 0
        if (pnl or 0) > 0: b['wins'] += 1
    for h, b in sorted(hr_buckets.items()):
        wr = 100 * b['wins'] / b['n']
        marker = "  BAD" if b['pnl'] < -1 else ("  ok" if b['pnl'] > 1 else "")
        print(f"  h={h:02d}: n={b['n']:3d} pnl=${b['pnl']:+7.2f} WR={wr:4.0f}% {marker}")
