"""Mid-tier per-symbol tune: USDJPY, GBPUSD, EURJPY.
Same methodology as _continue_tune.py — phase A/B/C with WF validation.
"""
import sys, os, json, time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '/Users/ashish/Documents/beast-trader')

import auto_tuned as _at
TRAIL_PROFILES = {
    '_TIGHT_LOCK':       _at._TIGHT_LOCK,
    '_WIDE_RUNNER':      _at._WIDE_RUNNER,
    '_RANGE_TIGHT':      _at._RANGE_TIGHT,
    '_TREND_LOOSE':      _at._TREND_LOOSE,
    '_AGGR_LOCK':        _at._AGGR_LOCK,
    '_RUNNER_NO_BE':     _at._RUNNER_NO_BE,
    '_WIDE_RUNNER_BE07': _at._WIDE_RUNNER_BE07,
}

def run_bt(args):
    label, symbol, days, params = args
    from backtest.v5_backtest import backtest_symbol
    try:
        r = backtest_symbol(symbol, days=days, params=params, verbose=False)
        if r is None:
            return label, None
        return label, {
            'trades': r.get('trades', 0),
            'pf': r.get('pf', 0),
            'pnl': r.get('pnl', 0),
            'wr': r.get('win_rate', 0),
            'dd': r.get('max_dd', r.get('dd', 0)),
        }
    except Exception as e:
        return label, {'error': str(e)}


def tune(symbol, sl_grid, mqs, days=180):
    t0 = time.time()
    _, baseline = run_bt(('baseline', symbol, days, None))
    if baseline is None or baseline.get('error'):
        return {'symbol': symbol, 'error': 'no baseline'}

    tasks = []
    for sl in sl_grid:
        tasks.append((f'sl_{sl}', symbol, days, {'sl_atr_mult': sl}))
    for tn in TRAIL_PROFILES:
        tasks.append((f'tr_{tn}', symbol, days, {'force_trail': TRAIL_PROFILES[tn]}))
    for mq in mqs:
        tasks.append((f'mq_{mq}', symbol, days, {'min_quality': mq}))

    with Pool(min(8, cpu_count()-2)) as pool:
        results_a = dict(pool.map(run_bt, tasks))

    def top_by(prefix, k=2):
        items = [(lbl, r) for lbl, r in results_a.items() if lbl.startswith(prefix) and r and not r.get('error')]
        items.sort(key=lambda x: -(x[1].get('pnl', 0) or 0))
        return items[:k]

    sl_top, trl_top, mq_top = top_by('sl_'), top_by('tr_'), top_by('mq_')

    # Phase B
    phase_b_configs = []
    for sl_lbl, _ in sl_top:
        sl_v = float(sl_lbl[3:])
        for tn_lbl, _ in trl_top:
            tn = tn_lbl[3:]
            for mq_lbl, _ in mq_top:
                mq_v = float(mq_lbl[3:])
                phase_b_configs.append({'sl': sl_v, 'trail': tn, 'mq': mq_v})

    b_tasks = []
    for idx, cfg in enumerate(phase_b_configs):
        b_tasks.append((f'B{idx}', symbol, days, {
            'sl_atr_mult': cfg['sl'],
            'force_trail': TRAIL_PROFILES[cfg['trail']],
            'min_quality': cfg['mq'],
        }))

    with Pool(min(8, cpu_count()-2)) as pool:
        results_b = dict(pool.map(run_bt, b_tasks))

    b_results = []
    for idx, cfg in enumerate(phase_b_configs):
        r = results_b.get(f'B{idx}')
        if r and not r.get('error'):
            b_results.append((idx, cfg, r))
    b_results.sort(key=lambda x: -(x[2].get('pnl', 0) or 0))
    if not b_results:
        return {'symbol': symbol, 'baseline': baseline, 'error': 'phase B empty'}

    # Phase C WF
    wf_tasks = []
    for idx, cfg, _ in b_results[:3]:
        params = {'sl_atr_mult': cfg['sl'], 'force_trail': TRAIL_PROFILES[cfg['trail']], 'min_quality': cfg['mq']}
        for fold_days in [60, 90, 120, 150, 180]:
            wf_tasks.append((f'B{idx}|f{fold_days}', symbol, fold_days, params))

    with Pool(min(8, cpu_count()-2)) as pool:
        wf_results = dict(pool.map(run_bt, wf_tasks))

    wf_summary = []
    for idx, cfg, b_res in b_results[:3]:
        folds = [wf_results.get(f'B{idx}|f{d}', None) for d in [60, 90, 120, 150, 180]]
        valid = [f for f in folds if f and not f.get('error')]
        pos = sum(1 for f in valid if (f.get('pnl', 0) or 0) > 0)
        avg_pf = sum((f.get('pf', 0) or 0) for f in valid) / max(1, len(valid))
        passed = pos >= 3 and avg_pf >= 1.3
        wf_summary.append({
            'config': cfg, 'b_result': b_res, 'folds': folds,
            'pos_count': pos, 'avg_pf': avg_pf, 'pass': passed,
        })

    winner = next((w for w in wf_summary if w['pass']), None)
    delta = (winner['b_result'].get('pnl', 0) - baseline.get('pnl', 0)) if winner else 0

    return {
        'symbol': symbol,
        'baseline': baseline,
        'phase_a_sl_top': sl_top,
        'phase_a_trail_top': trl_top,
        'phase_a_mq_top': mq_top,
        'phase_c_wf': wf_summary,
        'winner': winner,
        'delta_vs_baseline': delta,
        'ship': bool(winner) and delta >= 50,
        'elapsed': time.time() - t0,
    }


SYMBOL_CONFIGS = {
    'USDJPY': {'sl_grid': [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0], 'mqs': [25, 28, 30, 33, 35, 38, 40, 43, 45]},
    'GBPUSD': {'sl_grid': [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0], 'mqs': [25, 28, 30, 33, 35, 38, 40, 43]},
    'EURJPY': {'sl_grid': [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0], 'mqs': [25, 28, 30, 33, 35, 38, 40, 43]},
}

if __name__ == '__main__':
    out_dir = '/Users/ashish/Documents/beast-trader/per_symbol_tune_20260522'
    for sym, cfg in SYMBOL_CONFIGS.items():
        print(f'\n=== {sym} ===')
        result = tune(sym, sl_grid=cfg['sl_grid'], mqs=cfg['mqs'], days=180)
        with open(f'{out_dir}/{sym}.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        w = result.get('winner')
        if w:
            c = w['config']
            print(f"  WINNER: sl={c['sl']} trail={c['trail']} mq={c['mq']} | Δ=${result['delta_vs_baseline']:.0f} | WF {w['pos_count']}/5 PF {w['avg_pf']:.2f}")
        else:
            print(f"  NO WINNER (baseline ${result.get('baseline', {}).get('pnl', 0):.0f})")
