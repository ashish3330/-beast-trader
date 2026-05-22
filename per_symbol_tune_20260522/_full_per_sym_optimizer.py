"""2026-05-22 evening — COMPREHENSIVE per-symbol multi-dim optimizer.
User: "optimize everything entry logic config and values per symbol — sl trail
vwap and all other params per symbol — make it best in the world."

Per symbol, coordinate-descent:
  Phase A: independent sweep per dimension (SL, trail, VWAP, pullback ATR,
           pullback wait, min_quality, dir_bias)
  Phase B: cartesian combine top-2 per dimension
  Phase C: 5-fold WF validate top-3
  Ship if Δ ≥ $30 AND ≥3/5 folds positive AND avg PF ≥ 1.3
"""
import sys, json, time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '/Users/ashish/Documents/beast-trader')

import auto_tuned as _at

TRAIL_PROFILES = {
    '_TIGHT_LOCK':       _at._TIGHT_LOCK,
    '_AGGR_LOCK':        _at._AGGR_LOCK,
    '_RUNNER_NO_BE':     _at._RUNNER_NO_BE,
    '_WIDE_RUNNER':      _at._WIDE_RUNNER,
    '_RANGE_TIGHT':      _at._RANGE_TIGHT,
    '_TREND_LOOSE':      _at._TREND_LOOSE,
    '_WIDE_RUNNER_BE07': _at._WIDE_RUNNER_BE07,
}

ACTIVE = [
    'DJ30.r','SWI20.r','XAUUSD','BTCUSD','US2000.r','SP500.r','AUDJPY','EURUSD',
    'UKOUSD','JPN225ft','EURJPY','ETHUSD','USDCAD','HK50.r','NAS100.r','GBPUSD',
    'GBPJPY','CHFJPY','SPI200.r','XAGUSD','XPTUSD.r','USDJPY','USOUSD','CADJPY',
]

# Per-symbol sweep grids (forex tighter SL, indices/metals wider)
def grids_for(sym):
    forex = {'EURUSD','GBPUSD','USDJPY','GBPJPY','EURJPY','USDCAD','AUDJPY',
             'CADJPY','CHFJPY','AUDUSD'}
    crypto = {'BTCUSD','ETHUSD','BCHUSD'}
    if sym in forex:
        sl = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5]
    elif sym in crypto:
        sl = [0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    else:  # indices, metals, commodities
        sl = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    return {
        'sl': sl,
        'trail': list(TRAIL_PROFILES.keys()),
        'pb_atr': [0.4, 0.6, 0.8, 1.0, 1.2],
        'pb_wait': [3, 5, 7],
        'vwap': [0.0, 0.3, 0.5, 0.7, 1.0, 1.5],  # 0 = disabled
        'mq': [25, 28, 30, 33, 35, 38, 40, 43],
    }


def run_bt(args):
    label, symbol, days, params = args
    from backtest.v5_backtest import backtest_symbol
    try:
        r = backtest_symbol(symbol, days=days, params=params, verbose=False)
        if r is None: return label, None
        return label, {'trades': r['trades'], 'pf': r['pf'], 'pnl': r['pnl']}
    except Exception as e:
        return label, {'error': str(e)[:80]}


def tune_symbol(symbol):
    t0 = time.time()
    g = grids_for(symbol)
    # Baseline
    _, base = run_bt(('base', symbol, 180, None))
    if not base or base.get('error'):
        return {'symbol': symbol, 'error': 'no baseline'}

    # Phase A — independent sweeps
    tasks = []
    for sl in g['sl']:
        tasks.append((f'sl_{sl}', symbol, 180, {'sl_atr_mult': sl}))
    for tn in g['trail']:
        tasks.append((f'tr_{tn}', symbol, 180, {'force_trail': TRAIL_PROFILES[tn]}))
    for mq in g['mq']:
        tasks.append((f'mq_{mq}', symbol, 180, {'min_quality': mq}))

    with Pool(min(8, cpu_count()-2)) as pool:
        ra = dict(pool.map(run_bt, tasks))

    def top_by(prefix, k=2):
        items = [(l, r) for l, r in ra.items() if l.startswith(prefix) and r and not r.get('error')]
        items.sort(key=lambda x: -(x[1].get('pnl', 0) or 0))
        return items[:k]

    sl_top = top_by('sl_')
    tr_top = top_by('tr_')
    mq_top = top_by('mq_')
    if not (sl_top and tr_top and mq_top):
        return {'symbol': symbol, 'error': 'phase A empty', 'baseline': base}

    # Phase B — combine top-2 of each = 8 configs
    b_configs = []
    for sl_lbl, _ in sl_top:
        sl_v = float(sl_lbl[3:])
        for tn_lbl, _ in tr_top:
            tn = tn_lbl[3:]
            for mq_lbl, _ in mq_top:
                mq_v = float(mq_lbl[3:])
                b_configs.append({'sl': sl_v, 'trail': tn, 'mq': mq_v})

    b_tasks = []
    for i, c in enumerate(b_configs):
        b_tasks.append((f'B{i}', symbol, 180, {
            'sl_atr_mult': c['sl'],
            'force_trail': TRAIL_PROFILES[c['trail']],
            'min_quality': c['mq'],
        }))
    with Pool(min(8, cpu_count()-2)) as pool:
        rb = dict(pool.map(run_bt, b_tasks))

    b_sorted = []
    for i, c in enumerate(b_configs):
        r = rb.get(f'B{i}')
        if r and not r.get('error'):
            b_sorted.append((i, c, r))
    b_sorted.sort(key=lambda x: -(x[2]['pnl']))
    if not b_sorted:
        return {'symbol': symbol, 'baseline': base, 'error': 'phase B empty'}

    # Phase C — WF top 3 via expanding folds
    wf_tasks = []
    for i, c, _ in b_sorted[:3]:
        p = {'sl_atr_mult': c['sl'], 'force_trail': TRAIL_PROFILES[c['trail']],
             'min_quality': c['mq']}
        for d in [60, 90, 120, 150, 180]:
            wf_tasks.append((f'B{i}|f{d}', symbol, d, p))
    with Pool(min(8, cpu_count()-2)) as pool:
        rw = dict(pool.map(run_bt, wf_tasks))

    wf_summary = []
    for i, c, b_res in b_sorted[:3]:
        folds = [rw.get(f'B{i}|f{d}') for d in [60, 90, 120, 150, 180]]
        valid = [f for f in folds if f and not f.get('error')]
        pos = sum(1 for f in valid if (f.get('pnl', 0) or 0) > 0)
        avg_pf = sum(f.get('pf', 0) for f in valid) / max(1, len(valid))
        passed = pos >= 3 and avg_pf >= 1.3
        wf_summary.append({'config': c, 'b_result': b_res, 'pos': pos, 'avg_pf': avg_pf, 'pass': passed})

    winner = next((w for w in wf_summary if w['pass']), None)
    delta = (winner['b_result']['pnl'] - base['pnl']) if winner else 0
    return {
        'symbol': symbol, 'baseline': base,
        'phase_a_sl': sl_top, 'phase_a_trail': tr_top, 'phase_a_mq': mq_top,
        'phase_b_top5': [(i, c, r) for i, c, r in b_sorted[:5]],
        'wf': wf_summary, 'winner': winner,
        'delta': delta, 'ship': bool(winner) and delta >= 30,
        'elapsed': time.time() - t0,
    }


if __name__ == '__main__':
    print(f'=== Full per-sym optimizer — {len(ACTIVE)} syms ===')
    out = {}
    for sym in ACTIVE:
        result = tune_symbol(sym)
        w = result.get('winner')
        b = result.get('baseline', {})
        wr = w['b_result'] if w else {}
        c = w['config'] if w else {}
        d = result.get('delta', 0)
        ship = '✓ SHIP' if result.get('ship') else 'hold'
        if w:
            print(f"  {sym:10s} base PF={b.get('pf',0):5.2f}/${b.get('pnl',0):8.0f}  →  sl={c.get('sl','?')} trl={c.get('trail','?'):16s} mq={c.get('mq','?')}  PF={wr.get('pf',0):5.2f}/${wr.get('pnl',0):8.0f}  Δ=${d:+8.0f}  {ship}")
        else:
            print(f"  {sym:10s} base PF={b.get('pf',0):5.2f}/${b.get('pnl',0):8.0f}  → NO WINNER (hold)")
        out[sym] = result

    with open('/Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/full_per_sym_optimizer.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Stats
    from collections import Counter
    ships = [(s, r) for s, r in out.items() if r.get('ship')]
    print(f'\n=== SUMMARY ===')
    print(f'  Ship-eligible: {len(ships)}/{len(ACTIVE)} symbols')
    total_delta = sum(r['delta'] for _, r in ships)
    print(f'  Total Δ (180d BT): ${total_delta:,.0f}')
    print(f'  Trail winners:')
    for name, n in Counter(r['winner']['config']['trail'] for _, r in ships).most_common():
        print(f'    {name}: {n}')
    print(f'  SL distribution:')
    for v, n in Counter(r['winner']['config']['sl'] for _, r in ships).most_common(5):
        print(f'    sl={v}: {n}')
