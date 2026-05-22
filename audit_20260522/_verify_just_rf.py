#!/usr/bin/env python3 -B
"""Verify just the RF effect on XAUUSD (toxic invisible to BT)."""
import sys, importlib
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))
import auto_tuned as at; importlib.reload(at)
import config as cfg; importlib.reload(cfg)
import backtest.v5_backtest as bt; importlib.reload(bt)

# Run WF folds with the shipped config (rf=(96,1.0) only — toxic doesn't apply in BT)
import pandas as pd
orig_load = bt.load_data
def make_fold_loader(fold_n, slide_step=3, fold_d=15):
    def load_data_fold(sym, _ignored_days=None):
        df = orig_load(sym, days=None)
        if df is None or df.empty:
            return df
        t_min = df["time"].min()
        t_start = t_min + pd.Timedelta(days=(fold_n - 1) * slide_step)
        t_end = t_start + pd.Timedelta(days=fold_d)
        df = df[(df["time"] >= t_start) & (df["time"] < t_end)].reset_index(drop=True)
        return df
    return load_data_fold

for sym in ["XAUUSD", "SP500.r", "USDJPY"]:
    # Reset
    importlib.reload(at); importlib.reload(cfg); importlib.reload(bt)
    print(f"\n=== {sym} WF folds (shipped auto_tuned only — toxic not modeled in BT) ===")
    if sym == "SP500.r":
        days_full = 180
        slide = 36
        fold_d = 36
        mode = "disjoint"
    else:
        days_full = None
        slide = 3
        fold_d = 15
        mode = "sliding"
    for f_id in range(1, 6):
        importlib.reload(at); importlib.reload(cfg); importlib.reload(bt)
        if mode == "sliding":
            bt.load_data = make_fold_loader(f_id, slide, fold_d)
        else:
            def load_disjoint(sym2, _ignored=None, _f=f_id):
                df = orig_load(sym2, days=None)
                if df is None or df.empty: return df
                end = df["time"].max()
                offset_end = (5 - _f) * 36
                offset_start = offset_end + 36
                t_end = end - pd.Timedelta(days=offset_end)
                t_start = end - pd.Timedelta(days=offset_start)
                df = df[(df["time"] > t_start) & (df["time"] <= t_end)].reset_index(drop=True)
                return df
            bt.load_data = load_disjoint
        r = bt.backtest_symbol(sym, days=None, verbose=False)
        if r:
            print(f"  fold {f_id}: trades={r['trades']:3d} PF={r['pf']:5.2f} PnL=${r['pnl']:+,.0f}")
        else:
            print(f"  fold {f_id}: None")
        bt.load_data = orig_load
