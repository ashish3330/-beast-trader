"""2026-05-23 hard tune XAUUSD with all live safety patches active.
Coordinate descent: SL × trail × mq × pullback × VWAP × direction_bias.
"""
import sys, json, time
from multiprocessing import Pool, cpu_count
sys.path.insert(0, '/Users/ashish/Documents/beast-trader')

import auto_tuned as _at
TRAIL_PROFILES = {
    '_TIGHT_LOCK': _at._TIGHT_LOCK,
    '_AGGR_LOCK': _at._AGGR_LOCK,
    '_RUNNER_NO_BE': _at._RUNNER_NO_BE,
    '_WIDE_RUNNER': _at._WIDE_RUNNER,
    '_RANGE_TIGHT': _at._RANGE_TIGHT,
    '_TREND_LOOSE': _at._TREND_LOOSE,
    '_WIDE_RUNNER_BE07': _at._WIDE_RUNNER_BE07,
}

def run(args):
    label, params = args
    from backtest.v5_backtest import backtest_symbol
    p = {'risk_pct': 2.0, **params}
    try:
        r = backtest_symbol('XAUUSD', days=180, params=p, verbose=False)
        if not r: return label, None
        return label, {'trades': r['trades'], 'pf': r['pf'], 'pnl': r['pnl'], 'avg_r': r.get('avg_r', 0)}
    except Exception as e:
        return label, {'error': str(e)[:60]}

# Phase A: independent sweeps
phase_a = []
for sl in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]:
    phase_a.append((f'sl_{sl}', {'sl_atr_mult': sl}))
for tn, tp in TRAIL_PROFILES.items():
    phase_a.append((f'tr_{tn}', {'force_trail': tp}))
for mq in [22, 25, 28, 30, 33, 35, 38, 40]:
    phase_a.append((f'mq_{mq}', {'min_quality': mq}))
for d in ['LONG', 'SHORT', None]:
    if d: phase_a.append((f'dir_{d}', {'force_direction': d}))

t0 = time.time()
with Pool(min(8, cpu_count()-2)) as pool:
    res_a = dict(pool.map(run, phase_a))

print(f'Phase A done in {time.time()-t0:.1f}s')
# Print top-3 per dim
for prefix in ['sl_', 'tr_', 'mq_', 'dir_']:
    items = [(l, r) for l, r in res_a.items() if l.startswith(prefix) and r and 'error' not in r]
    items.sort(key=lambda x: -x[1].get('pnl', 0))
    print(f'\n{prefix} top-3:')
    for l, r in items[:3]:
        print(f'  {l:25s}  t={r["trades"]:4d}  PF={r["pf"]:5.2f}  pnl=${r["pnl"]:>9,.0f}  avgR={r["avg_r"]:+.2f}')

# Phase B: top-2 SL × top-2 trail × top-2 mq + best dir
def top2(prefix):
    items = [(l, r) for l, r in res_a.items() if l.startswith(prefix) and r and 'error' not in r]
    items.sort(key=lambda x: -x[1].get('pnl', 0))
    return [(l, r) for l, r in items[:2]]

sl_top = top2('sl_')
tr_top = top2('tr_')
mq_top = top2('mq_')
dir_top = top2('dir_')

phase_b = []
for sl_lbl, _ in sl_top:
    sl_v = float(sl_lbl[3:])
    for tn_lbl, _ in tr_top:
        tn = tn_lbl[3:]
        for mq_lbl, _ in mq_top:
            mq_v = int(mq_lbl[3:])
            for d_lbl, _ in dir_top:
                d = d_lbl[4:]
                phase_b.append((f'B_sl{sl_v}_tr{tn}_mq{mq_v}_d{d}', {
                    'sl_atr_mult': sl_v,
                    'force_trail': TRAIL_PROFILES[tn],
                    'min_quality': mq_v,
                    'force_direction': d,
                }))

print(f'\nPhase B: {len(phase_b)} configs')
with Pool(min(8, cpu_count()-2)) as pool:
    res_b = dict(pool.map(run, phase_b))

sorted_b = sorted([(l, r) for l, r in res_b.items() if r and 'error' not in r], key=lambda x: -x[1].get('pnl', 0))
print('\nPhase B top-5:')
for l, r in sorted_b[:5]:
    print(f'  {l}')
    print(f'    t={r["trades"]:4d}  PF={r["pf"]:5.2f}  pnl=${r["pnl"]:>9,.0f}  avgR={r["avg_r"]:+.2f}')

# Phase C: WF top-3
print('\nPhase C — 5-fold WF on top-3:')
wf_tasks = []
for l, r in sorted_b[:3]:
    parts = l.split('_')
    # Parse back: B_sl0.7_tr_TIGHT_LOCK_mq25_dLONG → sl=0.7, tr=_TIGHT_LOCK, mq=25, d=LONG
    sl_v = float(parts[1][2:])
    # trail name might contain underscores, find positions
    mq_idx = next(i for i, p in enumerate(parts) if p.startswith('mq'))
    d_idx = next(i for i, p in enumerate(parts) if p.startswith('d') and i > mq_idx)
    tn = '_' + '_'.join(parts[3:mq_idx])
    mq_v = int(parts[mq_idx][2:])
    d = parts[d_idx][1:]
    params = {'sl_atr_mult': sl_v, 'force_trail': TRAIL_PROFILES[tn], 'min_quality': mq_v, 'force_direction': d}
    for fold_days in [60, 90, 120, 150, 180]:
        wf_tasks.append((f'{l}|f{fold_days}', {**params, '__days': fold_days}))

def run_wf(args):
    label, params = args
    days = params.pop('__days')
    from backtest.v5_backtest import backtest_symbol
    p = {'risk_pct': 2.0, **params}
    try:
        r = backtest_symbol('XAUUSD', days=days, params=p, verbose=False)
        if not r: return label, None
        return label, {'trades': r['trades'], 'pf': r['pf'], 'pnl': r['pnl']}
    except Exception as e:
        return label, {'error': str(e)[:60]}

with Pool(min(8, cpu_count()-2)) as pool:
    res_c = dict(pool.map(run_wf, wf_tasks))

for l, r in sorted_b[:3]:
    folds = [res_c.get(f'{l}|f{d}') for d in [60, 90, 120, 150, 180]]
    valid = [f for f in folds if f and 'error' not in f]
    pos = sum(1 for f in valid if (f.get('pnl', 0) or 0) > 0)
    avg_pf = sum(f.get('pf', 0) for f in valid) / max(1, len(valid))
    pass_flag = '✓ SHIP' if pos >= 3 and avg_pf >= 1.5 else '✗ HOLD'
    print(f'  {l}: WF {pos}/5 avg_pf={avg_pf:.2f} {pass_flag}')

print(f'\nTotal runtime: {time.time()-t0:.1f}s')
