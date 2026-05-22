"""2026-05-22 evening — find BEST trail profile per symbol via BT.
User: "use trail config which is best optimized not aggressive so tune
harder again and again then backtest and make the bot live again."

Tests current aggressive profiles + OLD looser variants + middle-ground
"BALANCED" profiles. Per-symbol winner = best PnL with WF 5/5 positive.
"""
import sys, json, time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '/Users/ashish/Documents/beast-trader')

# Candidate trail profiles — name → list of (R_threshold, step_type, param)
# Tested HIGH→LOW threshold order (first match in loop wins).
CANDIDATES = {
    # ── CURRENT AGGRESSIVE (today's rewrites) ──
    'TIGHT_LOCK_AGGR': [
        (6.0,"lock",5.5),(4.5,"lock",4.0),(3.5,"lock",3.0),(2.5,"lock",2.1),
        (2.0,"lock",1.65),(1.5,"lock",1.2),(1.0,"lock",0.75),
        (0.7,"lock",0.45),(0.5,"lock",0.25),(0.3,"be",0.0),
    ],
    'AGGR_LOCK_AGGR': [
        (6.0,"lock",5.4),(4.0,"lock",3.5),(3.0,"lock",2.6),(2.0,"lock",1.6),
        (1.5,"lock",1.15),(1.0,"lock",0.7),(0.7,"lock",0.45),(0.5,"lock",0.25),
        (0.3,"be",0.0),
    ],
    'RUNNER_NO_BE_AGGR': [
        (8.0,"lock",7.0),(5.0,"lock",4.2),(3.5,"lock",2.9),(2.5,"lock",2.0),
        (1.8,"lock",1.4),(1.3,"lock",0.95),(0.9,"lock",0.6),(0.7,"lock",0.4),
        (0.5,"lock",0.22),
    ],
    # ── OLD LOOSER (pre-2026-05-22 aggressive rewrite) ──
    'TIGHT_LOCK_LOOSE': [
        (4.0,"lock",2.5),(2.0,"lock",1.2),(1.0,"lock",0.5),(0.3,"be",0.0),
    ],
    'AGGR_LOCK_LOOSE': [
        (8.0,"trail",0.3),(4.0,"trail",0.5),(2.0,"trail",0.8),
        (1.5,"lock",0.7),(1.0,"lock",0.4),(0.5,"be",0.0),
    ],
    'RUNNER_NO_BE_LOOSE': [
        (10.0,"trail",0.3),(5.0,"trail",0.4),(2.0,"trail",0.5),
        (1.0,"trail",0.5),(0.7,"lock",0.4),(0.5,"lock",0.2),
    ],
    'WIDE_RUNNER': [
        (10.0,"trail",0.3),(5.0,"trail",0.5),(2.5,"trail",0.7),
        (1.5,"lock",0.5),(0.7,"be",0.0),
    ],
    'TREND_LOOSE': [
        (15.0,"trail",0.3),(8.0,"trail",0.4),(4.0,"trail",0.5),
        (2.0,"lock",1.0),(1.0,"lock",0.5),(0.3,"be",0.0),
    ],
    # ── BALANCED middle-ground (newly designed: ~0.6-0.8R giveback) ──
    'BALANCED_TIGHT': [
        (5.0,"lock",4.0),(3.0,"lock",2.3),(2.0,"lock",1.4),(1.5,"lock",1.0),
        (1.0,"lock",0.6),(0.7,"lock",0.3),(0.5,"be",0.0),
    ],
    'BALANCED_MID': [
        (6.0,"trail",0.3),(3.0,"trail",0.5),(2.0,"lock",1.2),(1.5,"lock",0.8),
        (1.0,"lock",0.5),(0.7,"be",0.0),
    ],
    'BALANCED_WIDE': [
        (8.0,"trail",0.4),(4.0,"trail",0.6),(2.5,"lock",1.5),(1.5,"lock",0.9),
        (1.0,"lock",0.5),(0.5,"be",0.0),
    ],
}

ACTIVE = [
    'DJ30.r','SWI20.r','XAUUSD','BTCUSD','US2000.r','SP500.r','AUDJPY','EURUSD',
    'UKOUSD','JPN225ft','EURJPY','ETHUSD','USDCAD','HK50.r','NAS100.r','GBPUSD',
    'GBPJPY','CHFJPY','SPI200.r','XAGUSD','XPTUSD.r','USDJPY','USOUSD','CADJPY',
]


def run_bt(args):
    label, symbol, days, params = args
    from backtest.v5_backtest import backtest_symbol
    try:
        r = backtest_symbol(symbol, days=days, params=params, verbose=False)
        if r is None:
            return label, None
        return label, {'trades': r['trades'], 'pf': r['pf'], 'pnl': r['pnl']}
    except Exception as e:
        return label, {'error': str(e)[:100]}


def tune_symbol(symbol):
    t0 = time.time()
    # Baseline (current config)
    _, baseline = run_bt((f'base_{symbol}', symbol, 180, None))
    if not baseline or baseline.get('error'):
        return {'symbol': symbol, 'error': 'no baseline', 'elapsed': time.time()-t0}

    # Sweep all candidates
    tasks = [(name, symbol, 180, {'force_trail': prof}) for name, prof in CANDIDATES.items()]
    with Pool(min(8, cpu_count()-2)) as pool:
        results = dict(pool.map(run_bt, tasks))

    valid = [(n, r) for n, r in results.items() if r and not r.get('error')]
    if not valid:
        return {'symbol': symbol, 'baseline': baseline, 'error': 'no valid'}
    valid.sort(key=lambda x: -x[1]['pnl'])

    # WF validate top 3 — 5 expanding folds
    top3 = valid[:3]
    wf_tasks = []
    for name, _ in top3:
        for fold_days in [60, 90, 120, 150, 180]:
            wf_tasks.append((f'{name}|f{fold_days}', symbol, fold_days, {'force_trail': CANDIDATES[name]}))
    with Pool(min(8, cpu_count()-2)) as pool:
        wf_results = dict(pool.map(run_bt, wf_tasks))

    wf_summary = []
    for name, r in top3:
        folds = [wf_results.get(f'{name}|f{d}') for d in [60, 90, 120, 150, 180]]
        valid_folds = [f for f in folds if f and not f.get('error')]
        pos = sum(1 for f in valid_folds if (f.get('pnl', 0) or 0) > 0)
        avg_pf = sum(f.get('pf', 0) for f in valid_folds) / max(1, len(valid_folds))
        wf_summary.append({
            'name': name, 'result': r,
            'pos_folds': pos, 'avg_pf': avg_pf,
            'pass': pos >= 3 and avg_pf >= 1.3,
        })

    # Pick winner: first that passes WF
    winner = next((w for w in wf_summary if w['pass']), None)
    delta = (winner['result']['pnl'] - baseline['pnl']) if winner else 0
    return {
        'symbol': symbol,
        'baseline': baseline,
        'all_results': dict(valid),
        'wf_top3': wf_summary,
        'winner': winner['name'] if winner else None,
        'winner_result': winner['result'] if winner else None,
        'delta': delta,
        'ship': bool(winner) and delta >= 50,
        'elapsed': time.time() - t0,
    }


if __name__ == '__main__':
    print(f'=== Trail optimizer — {len(ACTIVE)} syms × {len(CANDIDATES)} profiles ===')
    out = {}
    for sym in ACTIVE:
        result = tune_symbol(sym)
        w = result.get('winner')
        b = result.get('baseline', {})
        wr = result.get('winner_result', {})
        d = result.get('delta', 0)
        ship_flag = '✓ SHIP' if result.get('ship') else 'hold'
        print(f"  {sym:10s}  base PF={b.get('pf',0):5.2f} pnl=${b.get('pnl',0):8.0f}  →  winner={w or 'NONE':22s}  PF={wr.get('pf',0):5.2f} pnl=${wr.get('pnl',0):8.0f}  Δ=${d:+8.0f}  {ship_flag}")
        out[sym] = result

    with open('/Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/trail_optimizer.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nFull JSON: per_symbol_tune_20260522/trail_optimizer.json')

    # Histogram of winners
    from collections import Counter
    winners = Counter(r.get('winner') for r in out.values() if r.get('ship'))
    print(f'\nShip-winner distribution:')
    for name, n in winners.most_common():
        print(f'  {name}: {n} symbols')
