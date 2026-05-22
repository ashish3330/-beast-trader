#!/usr/bin/env python3 -B
"""Check BT-computed SL vs live-implied SL for outlier trades."""
import pickle, sys
sys.path.insert(0, '/Users/ashish/Documents/beast-trader')
from signals.momentum_scorer import _compute_indicators, IND_DEFAULTS, IND_OVERRIDES
from backtest.v5_backtest import SL_OVERRIDE, SL_OVERRIDE_REGIME, get_regime
import pandas as pd
import numpy as np
import sqlite3

JDB = '/Users/ashish/Documents/beast-trader/data/trade_journal.db'
con = sqlite3.connect(JDB)
con.row_factory = sqlite3.Row
trade_ids = [647, 695, 716, 718, 726, 231, 250, 255, 263, 300]
rows = con.execute(f"SELECT * FROM trades WHERE id IN ({','.join(str(i) for i in trade_ids)})").fetchall()

for r in rows:
    t = dict(r)
    sym = t['symbol']
    fn_map = {'XAGUSD':'raw_h1_XAGUSD.pkl','XAUUSD':'raw_h1_XAUUSD.pkl',
              'USDJPY':'raw_h1_USDJPY.pkl','EURJPY':'raw_h1_EURJPY.pkl',
              'SP500.r':'raw_h1_SP500_r.pkl'}
    if sym not in fn_map:
        continue
    df = pickle.load(open(f'/Users/ashish/Documents/xauusd-trading-bot/cache/{fn_map[sym]}','rb'))
    icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(sym, {})}
    ind = _compute_indicators(df, icfg)
    target = pd.Timestamp(t['timestamp'])
    if target.tz is None:
        target = target.tz_localize('UTC')
    times = df['time'].values.astype('datetime64[ns]')
    idx = np.searchsorted(times, np.datetime64(target.to_numpy(),'ns'), side='right') - 1
    if idx < 30:
        continue
    atr_bt = float(ind['at'][idx])
    bbw = float(ind['bbw'][idx]) if not np.isnan(ind['bbw'][idx]) else 0.02
    adx = float(ind['adx'][idx]) if not np.isnan(ind['adx'][idx]) else 20
    regime = get_regime(bbw, adx)
    sl_mult = (SL_OVERRIDE_REGIME.get(sym, {}).get(regime)
               or SL_OVERRIDE.get(sym)
               or 1.5)
    sl_dist_bt = atr_bt * sl_mult
    # Implied live SL_DIST: |exit - entry| / r_multiple (when SL hit)
    if t['r_multiple'] and t['r_multiple'] != 0:
        sl_dist_live_implied = abs((t['exit_price'] - t['entry_price']) / t['r_multiple'])
    else:
        sl_dist_live_implied = float('nan')
    print(f"id={t['id']:>4} {sym:8s} regime={regime:8s} live_r={t['r_multiple']:>7.2f}  ATR_bt={atr_bt:.4f}  sl_mult={sl_mult}  sl_dist_bt={sl_dist_bt:.4f}  sl_dist_live_implied={sl_dist_live_implied:.4f}  ratio_bt/live={sl_dist_bt/sl_dist_live_implied if sl_dist_live_implied else 0:.2f}")
