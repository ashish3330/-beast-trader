#!/usr/bin/env python3
"""SR BT on symbols NOT currently in SR's live universe.

Run: python3 scripts/sr_candidates_bt.py
"""
import sys
sys.path.insert(0, '.')
import backtest.sweep_reclaim_backtest as sr_bt

# Currently SR-covered (skip): DJ30 JPN225 SPI200 XAU BTC ETH EURUSD UK100 XPT CHFJPY USOUSD
# Candidates SR does NOT consider today:
CANDIDATES = [
    # Active SR blacklist
    "USDJPY", "SWI20.r", "AUDJPY", "USDCAD", "XAGUSD",
    # Commented out of SYMBOLS dict
    "NAS100.r", "SP500.r", "US2000.r",
    # Dormant / removed
    "GER40.r", "UKOUSD", "HK50.r", "EURAUD",
    "COPPER-Cr", "GAS-Cr", "NG-Cr",
    # Extra liquid pairs not previously considered
    "GBPUSD", "GBPJPY", "AUDUSD", "EURJPY", "USDCHF",
]

results = []
for sym in CANDIDATES:
    try:
        r, _trades = sr_bt.backtest_symbol(sym)
        results.append(r)
    except Exception as e:
        results.append({'symbol': sym, 'status': f'ERR: {e}', 'trades': 0, 'pf': 0, 'total_R': 0})

# Filter to symbols with enough trades + sort by PF
ok = [r for r in results if r.get('status') == 'OK' and r.get('trades', 0) >= 50]
ok.sort(key=lambda x: -x['pf'])
err = [r for r in results if r.get('status') != 'OK']
thin = [r for r in results if r.get('status') == 'OK' and r.get('trades', 0) < 50]

print(f"\n{'=' * 110}")
print(f"  SR CANDIDATES — symbols NOT currently considered by live SR")
print(f"{'=' * 110}")
print(f"\nOK (n_trades >= 50, sorted by PF):\n")
print(f"  {'Sym':<10} {'Trd':>4} {'L/S':>7} {'WR':>5} {'TP1%':>5} {'PF':>5} {'avgR':>6} {'totR':>7} {'mDD%':>5}")
for r in ok:
    print(f"  {r['symbol']:<10} {r['trades']:>4} {r['longs']:>3}/{r['shorts']:<3} "
          f"{r['wr']*100:>4.1f}% {r['tp1_rate']*100:>4.0f}% "
          f"{r['pf']:>5.2f} {r['avg_R']:>+5.2f} {r['total_R']:>+7.1f} "
          f"{r['max_dd_pct']:>4.1f}%")

print(f"\nThin (n < 50):")
for r in thin:
    print(f"  {r['symbol']:<10} n={r['trades']} PF={r['pf']:.2f}")

print(f"\nERROR / no data:")
for r in err:
    print(f"  {r['symbol']:<10} {r.get('status', '?')}")

# Top 3 by PF
print(f"\n{'=' * 110}\n  TOP 3 BY PF\n{'=' * 110}")
for r in ok[:3]:
    print(f"  {r['symbol']:<10}  PF={r['pf']:.2f}  trades={r['trades']}  WR={r['wr']*100:.1f}%  totalR={r['total_R']:+.1f}  mDD={r['max_dd_pct']:.1f}%")
