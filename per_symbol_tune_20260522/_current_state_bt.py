"""2026-05-22 parallel BT mirror of current live setup.
Runs backtest_symbol on all active symbols at 180d using the current
SL/trail/min_quality/pullback/VWAP/dir_bias config in auto_tuned + config.

LIMITATION: BT does NOT model these live-only safety layers:
- AVG_WIN_LOSS_CAP (dollar cap on losses)
- POST_BIG_WIN 5h cooldown
- Score-tier marginal trail dispatch
- PEAK_GIVEBACK circuit breaker
- EarlyLossCut tiered exits

BT reflects: entry gates + trail profile + ratchet + per-sym SL/mq overrides.
Live PnL should be LOWER than BT in normal volatility, HIGHER in catastrophic
moves (where safety layers save losses), with much TIGHTER avg_loss.
"""
import sys, json, time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '/Users/ashish/Documents/beast-trader')

ACTIVE_SYMBOLS = [
    'DJ30.r','SWI20.r','XAUUSD','BTCUSD','US2000.r','SP500.r','AUDJPY','EURUSD',
    'UKOUSD','JPN225ft','EURJPY','ETHUSD','USDCAD','HK50.r','NAS100.r','GBPUSD',
    'GBPJPY','CHFJPY','SPI200.r','XAGUSD','XPTUSD.r','USDJPY','USOUSD','CADJPY',
]


def run_one(symbol):
    from backtest.v5_backtest import backtest_symbol
    t0 = time.time()
    try:
        r = backtest_symbol(symbol, days=180, verbose=False)
        if r is None:
            return symbol, {'error': 'no data', 'elapsed': time.time()-t0}
        return symbol, {
            'trades': r.get('trades', 0),
            'pf': r.get('pf', 0),
            'pnl': r.get('pnl', 0),
            'wr': r.get('win_rate', 0),
            'avg_r': r.get('avg_r', 0),
            'elapsed': time.time() - t0,
        }
    except Exception as e:
        return symbol, {'error': str(e)[:100], 'elapsed': time.time()-t0}


if __name__ == '__main__':
    print(f'=== Current-state parallel BT — {len(ACTIVE_SYMBOLS)} symbols × 180d ===')
    print(f'Workers: {min(cpu_count()-2, 8)}\n')
    t0 = time.time()
    with Pool(min(cpu_count()-2, 8)) as pool:
        results = dict(pool.map(run_one, ACTIVE_SYMBOLS))
    elapsed = time.time() - t0
    print(f'Done in {elapsed:.1f}s\n')

    # Sort by PF
    valid = [(s, r) for s, r in results.items() if 'error' not in r]
    errors = [(s, r) for s, r in results.items() if 'error' in r]
    valid.sort(key=lambda x: -x[1]['pf'])

    total_trades = sum(r['trades'] for _, r in valid)
    total_pnl    = sum(r['pnl']    for _, r in valid)
    avg_pf       = sum(r['pf']     for _, r in valid) / max(1, len(valid))

    print(f'{"symbol":12} {"trades":>6} {"PF":>7} {"WR":>5} {"PnL ($BT)":>14} {"avg_R":>6}  status')
    print('-' * 75)
    for sym, r in valid:
        flag = '⚠'  if r['pf'] < 1.5 else ('✗' if r['pf'] < 1.0 else '✓')
        print(f'{sym:12} {r["trades"]:>6d} {r["pf"]:>7.2f} {r["wr"]*100:>4.0f}% ${r["pnl"]:>13,.0f} {r["avg_r"]:>+5.2f}  {flag}')

    print('-' * 75)
    print(f'{"TOTAL":12} {total_trades:>6d} {avg_pf:>7.2f}       ${total_pnl:>13,.0f}')

    if errors:
        print(f'\nErrors ({len(errors)}):')
        for s, r in errors:
            print(f'  {s}: {r["error"]}')

    # Save JSON
    out = {
        'timestamp': time.time(),
        'elapsed': elapsed,
        'per_symbol': {s: r for s, r in valid},
        'errors': {s: r for s, r in errors},
        'summary': {
            'total_trades': total_trades,
            'total_pnl_compound_bt': total_pnl,
            'avg_pf': avg_pf,
            'symbols_passing': sum(1 for _, r in valid if r['pf'] >= 1.5),
            'symbols_marginal': sum(1 for _, r in valid if 1.0 <= r['pf'] < 1.5),
            'symbols_negative': sum(1 for _, r in valid if r['pf'] < 1.0),
        },
    }
    with open('/Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/current_state_bt.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nFull JSON: per_symbol_tune_20260522/current_state_bt.json')
