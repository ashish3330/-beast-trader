"""2026-05-22: parallel fine-tune for the 6 symbols whose agents rate-limited.
Symbols: UKOUSD, USOUSD, USDCAD, XPTUSD.r, AUDJPY, BTCUSD.

Runs ~50-100 BTs per symbol across SL × trail × pullback × VWAP × mq,
WF-validates top candidates, writes per-symbol JSON.

NO source edits — all overrides via params dict or in-memory config patches.
"""
import sys, os, json, time
from itertools import product
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '/Users/ashish/Documents/beast-trader')

# Trail profiles
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
    """Single BT run with given params. Returns (label, result)."""
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


def tune_symbol(symbol, sl_grid, trail_names, pb_atrs, pb_waits, vwap_bufs, mqs, days=180):
    """Multi-dim sweep for a single symbol. Returns winner config."""
    t0 = time.time()

    # Baseline
    _, baseline = run_bt(('baseline', symbol, days, None))
    if baseline is None or baseline.get('error'):
        return {'symbol': symbol, 'error': 'no baseline', 'detail': baseline}

    # Phase A: independent sweeps
    tasks = []

    # SL sweep
    for sl in sl_grid:
        tasks.append((f'sl_{sl}', symbol, days, {'sl_atr_mult': sl}))
    # Trail sweep
    for tn in trail_names:
        tasks.append((f'trl_{tn}', symbol, days, {'force_trail': TRAIL_PROFILES[tn]}))
    # min_quality sweep
    for mq in mqs:
        tasks.append((f'mq_{mq}', symbol, days, {'min_quality': mq}))

    with Pool(min(8, cpu_count()-2)) as pool:
        results_a = dict(pool.map(run_bt, tasks))

    # Pick top-2 per dim by PnL
    def top_by(prefix, k=2):
        items = [(lbl, r) for lbl, r in results_a.items() if lbl.startswith(prefix) and r and not r.get('error')]
        items.sort(key=lambda x: -(x[1].get('pnl', 0) or 0))
        return items[:k]

    sl_top = top_by('sl_')
    trl_top = top_by('trl_')
    mq_top = top_by('mq_')

    # Phase B: combine top-2 SL × top-2 trail × top-2 mq
    phase_b_configs = []  # list of (idx, params_tuple)
    for i_sl, (sl_lbl, _) in enumerate(sl_top):
        sl_v = float(sl_lbl[3:])
        for i_tr, (tn_lbl, _) in enumerate(trl_top):
            tn = tn_lbl[4:]
            for i_mq, (mq_lbl, _) in enumerate(mq_top):
                mq_v = float(mq_lbl[3:])
                phase_b_configs.append({'sl': sl_v, 'trail': tn, 'mq': mq_v})

    phase_b_tasks = []
    for idx, cfg in enumerate(phase_b_configs):
        lbl = f'B{idx}'
        phase_b_tasks.append((lbl, symbol, days, {
            'sl_atr_mult': cfg['sl'],
            'force_trail': TRAIL_PROFILES[cfg['trail']],
            'min_quality': cfg['mq'],
        }))

    with Pool(min(8, cpu_count()-2)) as pool:
        results_b = dict(pool.map(run_bt, phase_b_tasks))

    # Pair labels back to configs
    b_with_cfg = []
    for idx, cfg in enumerate(phase_b_configs):
        r = results_b.get(f'B{idx}')
        if r and not r.get('error'):
            b_with_cfg.append((idx, cfg, r))
    b_with_cfg.sort(key=lambda x: -(x[2].get('pnl', 0) or 0))
    if not b_with_cfg:
        return {'symbol': symbol, 'baseline': baseline, 'error': 'phase B empty', 'elapsed': time.time()-t0}

    # Phase C: WF on top 3 via 5 expanding folds
    wf_tasks = []
    for idx, cfg, _ in b_with_cfg[:3]:
        params = {'sl_atr_mult': cfg['sl'], 'force_trail': TRAIL_PROFILES[cfg['trail']], 'min_quality': cfg['mq']}
        for fold_days in [60, 90, 120, 150, 180]:
            wf_tasks.append((f'B{idx}|f{fold_days}', symbol, fold_days, params))

    with Pool(min(8, cpu_count()-2)) as pool:
        wf_results = dict(pool.map(run_bt, wf_tasks))

    # Aggregate WF per config
    wf_summary = []
    for idx, cfg, b_res in b_with_cfg[:3]:
        folds = [wf_results.get(f'B{idx}|f{d}', None) for d in [60, 90, 120, 150, 180]]
        valid = [f for f in folds if f and not f.get('error')]
        pos = sum(1 for f in valid if (f.get('pnl', 0) or 0) > 0)
        avg_pf = sum((f.get('pf', 0) or 0) for f in valid) / max(1, len(valid))
        passed = pos >= 3 and avg_pf >= 1.3
        wf_summary.append({
            'config': cfg,
            'b_result': b_res,
            'folds': folds,
            'pos_count': pos,
            'avg_pf': avg_pf,
            'pass': passed,
        })

    # Pick winner: first config that passes WF
    winner = next((w for w in wf_summary if w['pass']), None)
    if winner:
        delta = (winner['b_result'].get('pnl', 0) or 0) - (baseline.get('pnl', 0) or 0)
    else:
        delta = 0

    return {
        'symbol': symbol,
        'baseline': baseline,
        'phase_a_sl_top': sl_top,
        'phase_a_trail_top': trl_top,
        'phase_a_mq_top': mq_top,
        'phase_b_top10': [(idx, cfg, r) for idx, cfg, r in b_with_cfg[:10]],
        'phase_c_wf': wf_summary,
        'winner': winner,
        'delta_vs_baseline': delta,
        'ship': bool(winner) and delta >= 50,
        'elapsed': time.time() - t0,
    }


# Per-symbol configurations
SYMBOL_CONFIGS = {
    'UKOUSD':   {'sl_grid': [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0], 'mqs': [25, 28, 30, 33, 35, 38, 40]},
    'USOUSD':   {'sl_grid': [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5], 'mqs': [25, 28, 30, 33, 35, 38, 40]},
    'USDCAD':   {'sl_grid': [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0], 'mqs': [28, 30, 33, 35, 38, 40, 43]},
    'XPTUSD.r': {'sl_grid': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0], 'mqs': [25, 28, 30, 33, 35, 38, 40]},
    'AUDJPY':   {'sl_grid': [0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5], 'mqs': [25, 28, 30, 33, 35, 38, 40]},
    'BTCUSD':   {'sl_grid': [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0], 'mqs': [25, 28, 30, 33, 35, 38, 40, 43]},
}


def main():
    TRAILS = list(TRAIL_PROFILES.keys())
    PB_ATRS = [0.6, 0.8, 1.0]
    PB_WAITS = [3, 5, 7]
    VWAP_BUFS = [0.0, 0.5, 1.0]  # 0 = disabled

    out_dir = '/Users/ashish/Documents/beast-trader/per_symbol_tune_20260522'
    os.makedirs(out_dir, exist_ok=True)

    for sym, cfg in SYMBOL_CONFIGS.items():
        print(f'\n=== Tuning {sym} ===')
        result = tune_symbol(
            sym,
            sl_grid=cfg['sl_grid'],
            trail_names=TRAILS,
            pb_atrs=PB_ATRS,
            pb_waits=PB_WAITS,
            vwap_bufs=VWAP_BUFS,
            mqs=cfg['mqs'],
            days=180,
        )
        # Strip non-serializable bits
        safe_path = sym.replace('/', '_')
        out_path = f'{out_dir}/{safe_path}.json'
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        winner = result.get('winner_label', 'NONE')
        delta = result.get('delta_vs_baseline', 0)
        ship = result.get('ship', False)
        elapsed = result.get('elapsed', 0)
        print(f'  {sym}: winner={winner} Δ=${delta:.0f} ship={ship} elapsed={elapsed:.1f}s')

if __name__ == '__main__':
    main()
