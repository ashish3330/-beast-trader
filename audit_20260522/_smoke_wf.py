#!/usr/bin/env python3 -B
"""Debug WF fold result_none issue."""
import sys
import importlib
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import config as cfg
importlib.reload(cfg)
import backtest.v5_backtest as bt
importlib.reload(bt)

import pandas as pd

# Mimic fold-1 load_data
orig_load = bt.load_data
def load_data_fold(sym, _ignored_days=None):
    df = orig_load(sym, days=None)
    if df is None or df.empty:
        return df
    end = df["time"].max()
    fold_n = 1
    num = 5
    fold_d = 36
    offset_end = (num - fold_n) * fold_d
    offset_start = offset_end + fold_d
    t_end = end - pd.Timedelta(days=offset_end)
    t_start = end - pd.Timedelta(days=offset_start)
    df2 = df[(df["time"] > t_start) & (df["time"] <= t_end)].reset_index(drop=True)
    print(f"  fold 1: t_start={t_start} t_end={t_end} n_bars={len(df2)} (original {len(df)})")
    return df2

bt.load_data = load_data_fold
r = bt.backtest_symbol("XAUUSD", days=None, verbose=False)
print(f"Fold 1 result: {r}")
