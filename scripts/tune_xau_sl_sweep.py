#!/usr/bin/env python3
"""XAU SL multiplier sweep — 2026-06-10.

Sweeps SYMBOL_ATR_SL_OVERRIDE['XAUUSD'] in {1.0, 1.2, 1.5, 1.8, 2.0, 2.5}
over 180d via v5_backtest. Patches the live config dict in-memory before
each run (no file mutation). Reports PF/WR/PnL/DD/avgR/max_loss per combo.

Run:  python3 scripts/tune_xau_sl_sweep.py
"""
import sys
import json
sys.path.insert(0, '.')

import config as _cfg
import backtest.v5_backtest as v5

SL_GRID = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
DAYS_FOLDS = [60, 180, 360]
SYMBOL = 'XAUUSD'

orig_sl = v5.SL_OVERRIDE.get(SYMBOL)
all_results = {}  # {days: [{sl_mult, n, pf, wr, pnl, ...}]}

for days in DAYS_FOLDS:
    print(f"\n{'=' * 78}")
    print(f"  XAU SL sweep | {days}d window | grid={SL_GRID}")
    print(f"{'=' * 78}\n")
    fold_results = []
    for sl_mult in SL_GRID:
        v5.SL_OVERRIDE[SYMBOL] = sl_mult
        try:
            r = v5.backtest_symbol(SYMBOL, days=days, verbose=False)
            if r is None or not isinstance(r, dict):
                print(f"  SL={sl_mult}: NULL")
                fold_results.append({'sl_mult': sl_mult, 'pf': None})
                continue
            row = {
                'sl_mult': sl_mult,
                'n': r.get('n_trades', r.get('trades', 0)),
                'pf': r.get('pf', 0),
                'wr': r.get('wr', 0),
                'pnl': r.get('pnl', r.get('pnl_usd', 0)),
                'avg_r': r.get('avg_r', r.get('avg_R', 0)),
            }
            fold_results.append(row)
            print(f"  SL={sl_mult:<4}  n={row['n']:<4} PF={row['pf']:<6.2f} WR={row['wr']:<5.1%} PnL=${row['pnl']:<7.2f} avgR={row['avg_r']:<5.2f}")
        except Exception as e:
            print(f"  SL={sl_mult}: ERROR {e}")
            fold_results.append({'sl_mult': sl_mult, 'error': str(e), 'pf': None})
    all_results[days] = fold_results

# Restore
if orig_sl is not None:
    v5.SL_OVERRIDE[SYMBOL] = orig_sl
print(f"\n--- restored v5.SL_OVERRIDE['{SYMBOL}'] = {orig_sl} ---")

# Cross-fold robustness: for each SL, rank in each fold
print(f"\n{'=' * 78}\n  CROSS-FOLD ROBUSTNESS\n{'=' * 78}")
print(f"  PF per fold + rank (1=best). A robust SL ranks top-3 across ALL folds.\n")
print(f"  {'SL':<6}", end='')
for d in DAYS_FOLDS:
    print(f"{'PF_' + str(d) + 'd':<10}{'rank':<6}", end='')
print(f"{'avg_PF':<8}{'avg_rank':<10}{'robust':<10}")
# Compute rank per fold
ranks_per_sl = {sl: [] for sl in SL_GRID}
pfs_per_sl = {sl: [] for sl in SL_GRID}
for days, fold in all_results.items():
    valid = [r for r in fold if r.get('pf') is not None]
    sorted_fold = sorted(valid, key=lambda x: -x['pf'])
    rank_map = {r['sl_mult']: i + 1 for i, r in enumerate(sorted_fold)}
    for r in fold:
        if r.get('pf') is not None:
            ranks_per_sl[r['sl_mult']].append(rank_map[r['sl_mult']])
            pfs_per_sl[r['sl_mult']].append(r['pf'])

# Sort SL grid by avg rank
sl_summary = []
for sl in SL_GRID:
    if not pfs_per_sl[sl]:
        continue
    avg_pf = sum(pfs_per_sl[sl]) / len(pfs_per_sl[sl])
    avg_rank = sum(ranks_per_sl[sl]) / len(ranks_per_sl[sl])
    max_rank = max(ranks_per_sl[sl])
    robust = 'YES' if max_rank <= 3 else 'NO'
    sl_summary.append((sl, avg_pf, avg_rank, max_rank, robust))
sl_summary.sort(key=lambda x: x[2])  # by avg_rank

for sl, avg_pf, avg_rank, max_rank, robust in sl_summary:
    print(f"  {sl:<6}", end='')
    for days in DAYS_FOLDS:
        fold = next(f for f in all_results[days] if f['sl_mult'] == sl)
        if fold.get('pf') is None:
            print(f"{'NULL':<10}{'-':<6}", end='')
        else:
            r = ranks_per_sl[sl][DAYS_FOLDS.index(days)]
            print(f"{fold['pf']:<10.2f}{r:<6}", end='')
    print(f"{avg_pf:<8.2f}{avg_rank:<10.2f}{robust:<10}")

print(f"\nBEST ROBUST PICK: SL_ATR_MULT = {sl_summary[0][0] if sl_summary and sl_summary[0][4] == 'YES' else 'none — current is fine'}")

with open('/tmp/xau_sl_sweep_wf.json', 'w') as f:
    json.dump({'grid': SL_GRID, 'folds': DAYS_FOLDS,
               'per_fold': all_results, 'summary': sl_summary}, f, indent=2, default=str)
print(f"Saved to /tmp/xau_sl_sweep_wf.json\n")
