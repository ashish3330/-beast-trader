#!/usr/bin/env python3 -B
import sys, importlib
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))
import pandas as pd
import config as cfg; importlib.reload(cfg)
import backtest.v5_backtest as bt; importlib.reload(bt)
orig = bt.load_data
df_full = orig("XAUUSD", days=None)
print("XAUUSD full df:", len(df_full), "from", df_full['time'].min(), "to", df_full['time'].max())
print("warmup needed:", max(100, 30) + 30, "bars; we have 12-day window ≈", 12*24, "bars")

for fold_n in [1, 3, 5]:
    fold_d = 12
    slide = 4
    t_min = df_full["time"].min()
    t_start = t_min + pd.Timedelta(days=(fold_n-1)*slide)
    t_end = t_start + pd.Timedelta(days=fold_d)
    df = df_full[(df_full["time"] >= t_start) & (df_full["time"] < t_end)].reset_index(drop=True)
    print(f"  fold {fold_n}: n={len(df)} {t_start} .. {t_end}")
