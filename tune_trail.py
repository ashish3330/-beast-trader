"""Hardcore trail ATR multiplier grid search — single symbol mode."""
import numpy as np, sys, pickle, pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, '/Users/ashish/Documents/beast-trader')
from signals.momentum_scorer import _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES
import backtest.dragon_backtest as bt
from config import SYMBOL_ATR_SL_OVERRIDE, ATR_SL_MULTIPLIER

CACHE = Path('/Users/ashish/Documents/xauusd-trading-bot/cache')

SYMS = {
    'XAUUSD':   ('raw_h1_xauusd.pkl',   0.01,  1.0,   0.33,  'Gold',   True),
    'BTCUSD':   ('raw_h1_BTCUSD.pkl',   0.01,  0.01,  17.0,  'Crypto', False),
    'XAGUSD':   ('raw_h1_XAGUSD.pkl',   0.001, 50.0,  0.025, 'Gold',   True),
    'NAS100.r': ('raw_h1_NAS100_r.pkl', 0.01,  1.0,   2.5,   'Index',  False),
    'JPN225ft': ('raw_h1_JPN225ft.pkl', 1.0,   0.01,  15.0,  'Index',  False),
    'USDJPY':   ('raw_h1_USDJPY.pkl',   0.001, 7.14,  0.016, 'Forex',  False),
}

# Pre-compute indicators for target symbol
SYM = sys.argv[1] if len(sys.argv) > 1 else 'XAUUSD'
cf, pt, tv, spread, cat, ml = SYMS[SYM]
df = pickle.load(open(CACHE / cf, 'rb'))
if not pd.api.types.is_datetime64_any_dtype(df['time']):
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
icfg = dict(IND_DEFAULTS); icfg.update(IND_OVERRIDES.get(SYM, {}))
cutoff = df['time'].max() - pd.Timedelta(days=365)
start_idx = max(int(df[df['time'] >= cutoff].index[0]), icfg['EMA_T'] + 30)
IND = _compute_indicators(df, icfg)
N = IND['n']
SL_M = SYMBOL_ATR_SL_OVERRIDE.get(SYM, ATR_SL_MULTIPLIER)


def run(trail_steps):
    eq = 1000; peak = 1000; max_dd = 0; wins = 0; gp = 0; gl = 0; nt = 0
    in_trade = False; d = 0; entry = 0; pos_sl = 0; sl_dist = 0; lot = 0
    np.random.seed(42)
    for i in range(start_idx, N):
        atr = float(IND['at'][i])
        if atr == 0: continue
        bt2 = df['time'].iloc[i]
        if cat != 'Crypto' and hasattr(bt2, 'hour') and (bt2.hour >= 22 or bt2.hour < 6): continue
        if in_trade:
            if (d == 1 and IND['l'][i] <= pos_sl) or (d == -1 and IND['h'][i] >= pos_sl):
                pnl = d * (pos_sl - entry) / pt * tv * lot - spread / pt * tv * lot
                eq += pnl
                if pnl > 0: gp += pnl; wins += 1
                else: gl += abs(pnl)
                nt += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
                in_trade = False; continue
            cur = float(IND['c'][i])
            pr = ((cur - entry) * d) / sl_dist if sl_dist > 0 else 0
            ns = None
            for th, ac, pa in trail_steps:
                if pr >= th:
                    if ac == 'trail': ns = cur - pa * atr * d
                    elif ac == 'lock': ns = entry + pa * sl_dist * d
                    elif ac == 'be': ns = entry
                    break
            if ns:
                if d == 1 and ns > pos_sl: pos_sl = ns
                elif d == -1 and ns < pos_sl: pos_sl = ns
        bi = i - 1
        if bi < 21: continue
        ls, ss = _score(IND, bi)
        reg = bt.get_regime(IND, bi); am = bt.get_adaptive_min_score(reg, SYM)
        buy = ls >= am; sell = ss >= am
        if not buy and not sell: continue
        nd = 1 if (buy and (not sell or ls >= ss)) else -1
        if ml and np.random.random() > min(1.0, max(ls, ss) / 10.0): continue
        if in_trade and nd != d:
            pnl = d * (float(IND['c'][i]) - entry) / pt * tv * lot - spread / pt * tv * lot
            eq += pnl
            if pnl > 0: gp += pnl; wins += 1
            else: gl += abs(pnl)
            nt += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
            in_trade = False
        if not in_trade:
            d = nd; sl_dist = max(atr * SL_M, atr * 1.5)
            ra = eq * 0.01; pv = sl_dist / pt * tv
            lot = max(ra / pv, 0.01) if pv > 0 else 0.01
            entry = float(IND['o'][i]) + spread / 2 * d
            pos_sl = entry - sl_dist * d; in_trade = True
    pf = gp / gl if gl > 0 else 999
    dd = max_dd / peak * 100 if peak else 0
    wr = wins / nt * 100 if nt else 0
    return pf, wr, dd, nt


# Symbol-specific grid configs
CONFIGS = {
    'XAUUSD': {
        'locks': [(1.5, 'trail', 1.5), (1.0, 'lock', 0.33), (0.6, 'lock', 0.20), (0.3, 'lock', 0.10)],
        'levels': [(6.0, '6R'), (4.0, '4R'), (2.0, '2R')],
        'ranges': {'6R': [0.3, 0.5, 0.7], '4R': [0.3, 0.5, 0.7, 1.0], '2R': [0.7, 1.0, 1.3, 1.5, 2.0]},
    },
    'BTCUSD': {
        'locks': [(1.5, 'trail', 2.0), (1.0, 'lock', 0.5), (0.5, 'be', 0.0)],
        'levels': [(6.0, '6R'), (4.0, '4R'), (2.5, '2.5R')],
        'ranges': {'6R': [0.3, 0.5, 0.7, 1.0], '4R': [0.5, 0.7, 1.0, 1.5], '2.5R': [1.0, 1.5, 2.0, 2.5, 3.0]},
    },
    'XAGUSD': {
        'locks': [(1.5, 'trail', 1.5), (1.0, 'lock', 0.33), (0.6, 'lock', 0.20), (0.4, 'lock', 0.20)],
        'levels': [(6.0, '6R'), (4.0, '4R'), (2.0, '2R')],
        'ranges': {'6R': [0.3, 0.5, 0.7], '4R': [0.3, 0.5, 0.7, 1.0], '2R': [0.7, 1.0, 1.3, 1.5, 2.0]},
    },
    'NAS100.r': {
        'locks': [(1.5, 'trail', 1.5), (1.0, 'lock', 0.33), (0.6, 'lock', 0.20), (0.3, 'lock', 0.15)],
        'levels': [(6.0, '6R'), (4.0, '4R'), (2.0, '2R')],
        'ranges': {'6R': [0.3, 0.5, 0.7], '4R': [0.3, 0.5, 0.7, 1.0], '2R': [0.7, 1.0, 1.3, 1.5, 2.0]},
    },
    'JPN225ft': {
        'locks': [(1.5, 'trail', 1.5), (1.0, 'lock', 0.33), (0.6, 'lock', 0.20), (0.15, 'lock', 0.05)],
        'levels': [(6.0, '6R'), (4.0, '4R'), (2.0, '2R')],
        'ranges': {'6R': [0.3, 0.5, 0.7], '4R': [0.3, 0.5, 0.7, 1.0], '2R': [0.7, 1.0, 1.3, 1.5, 2.0]},
    },
    'USDJPY': {
        'locks': [(1.5, 'trail', 1.5), (1.0, 'lock', 0.33), (0.6, 'lock', 0.20), (0.15, 'lock', 0.10)],
        'levels': [(6.0, '6R'), (4.0, '4R'), (2.0, '2R')],
        'ranges': {'6R': [0.3, 0.5, 0.7], '4R': [0.3, 0.5, 0.7, 1.0], '2R': [0.7, 1.0, 1.3, 1.5, 2.0]},
    },
}

cfg = CONFIGS[SYM]
locks = cfg['locks']
levels = cfg['levels']
ranges = cfg['ranges']

level_options = [[(lbl, v) for v in ranges[lbl]] for _, lbl in levels]
best = (0, '', None)
results = []

for combo in product(*level_options):
    trail = []
    for (th, lbl), (_, val) in zip(levels, combo):
        trail.append((th, 'trail', val))
    trail.extend(locks)
    pf, wr, dd, nt = run(trail)
    if nt >= 30:
        tag = ' '.join(f'{lbl}={v}' for (lbl, v) in combo)
        results.append((pf, wr, dd, nt, tag, trail))
        if pf > best[0]:
            best = (pf, tag, trail, wr, dd, nt)

print(f'{SYM} - Top 10 Trail ATR Combos (365d)')
print('=' * 70)
for pf, wr, dd, nt, tag, _ in sorted(results, key=lambda x: -x[0])[:10]:
    print(f'  {tag} -> PF={pf:.2f} WR={wr:.1f}% DD={dd:.1f}% N={nt}')
if best[0] > 0:
    print(f'  ** BEST: {best[1]} -> PF={best[0]:.2f} WR={best[3]:.1f}% DD={best[4]:.1f}% N={best[5]}')
    print(f'  steps: {best[2]}')
