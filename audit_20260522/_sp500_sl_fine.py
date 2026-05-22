#!/usr/bin/env python3 -B
"""Fine-grained SL sweep around 0.7 to confirm winner robustness.
Also: dir_bias[volatile]=None standalone WF + composability check."""
import sys
from pathlib import Path
ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))
from multiprocessing import Pool
import importlib
import pandas as pd

WF_NUM_FOLDS = 5
WF_FOLD_DAYS = 36

def _bt(args):
    """args = (label, sl, dir_bias_cell, fold_id)"""
    label, sl, dir_bias_cell, fold_id = args
    import auto_tuned as _at
    importlib.reload(_at)
    import config as _cfg
    importlib.reload(_cfg)
    import backtest.v5_backtest as _bt
    importlib.reload(_bt)
    if sl is not None:
        _bt.SL_OVERRIDE['SP500.r'] = float(sl)
        _bt.SL_OVERRIDE_REGIME.pop('SP500.r', None)
    if dir_bias_cell is not None:
        _bt._DIR_BIAS_REGIME_STR['SP500.r'] = dict(dir_bias_cell)
    if fold_id is not None:
        orig_load = _bt.load_data
        fold_n = int(fold_id)
        num = WF_NUM_FOLDS
        fold_d = WF_FOLD_DAYS
        def load_data_fold(sym, _ignored=None):
            df = orig_load(sym, days=None)
            end = df["time"].max()
            offset_end = (num - fold_n) * fold_d
            offset_start = offset_end + fold_d
            t_end = end - pd.Timedelta(days=offset_end)
            t_start = end - pd.Timedelta(days=offset_start)
            df = df[(df["time"] > t_start) & (df["time"] <= t_end)].reset_index(drop=True)
            return df
        _bt.load_data = load_data_fold
        r = _bt.backtest_symbol("SP500.r", days=None, params={"risk_pct": 2.0}, verbose=False)
    else:
        r = _bt.backtest_symbol("SP500.r", days=180, params={"risk_pct": 2.0}, verbose=False)
    if r is None: return {"label": label, "err": "null"}
    return {"label": label, "trades": r["trades"], "pf": r["pf"], "pnl": r["pnl"], "dd": r["dd"], "wr": r["wr"]}

def main():
    # Phase 1: fine-grained SL sweep — full 180d
    print("\n=== Fine SL sweep (180d) ===")
    sl_vals = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
    jobs = [(f"sl={s}", s, None, None) for s in sl_vals]
    with Pool(6) as p:
        for r in p.imap(_bt, jobs):
            if "err" in r:
                print(f"  {r['label']}: ERR")
            else:
                print(f"  {r['label']:10s}  n={r['trades']:4d} PF={r['pf']:5.2f} PnL=${r['pnl']:+10,.0f} DD={r['dd']:.2f}% WR={r['wr']}%")

    # Phase 2: WF for sl=0.6, 0.7, 0.8 (around winner)
    print("\n=== WF for sl ∈ {0.6, 0.7, 0.8} ===")
    wf_jobs = []
    for sl in [0.6, 0.7, 0.8]:
        for fold in range(1, WF_NUM_FOLDS + 1):
            wf_jobs.append((f"sl={sl}_f{fold}", sl, None, fold))
    with Pool(6) as p:
        wf_results = list(p.imap(_bt, wf_jobs))
    # Group by sl
    by_sl = {}
    for r in wf_results:
        sl_key = r["label"].split("_")[0]
        by_sl.setdefault(sl_key, []).append(r)
    for sl_key, folds in by_sl.items():
        total = sum(f["pnl"] for f in folds)
        pos = sum(1 for f in folds if f["pnl"] > 0)
        avg_pf = sum(f["pf"] for f in folds) / len(folds)
        print(f"  {sl_key}  WF {pos}/5  avg_pf={avg_pf:.2f}  total=${total:+,.0f}")
        for f in folds:
            print(f"    n={f['trades']:3d} PF={f['pf']:5.2f} PnL=${f['pnl']:+,.0f} WR={f['wr']}%")

    # Phase 3: Is dir_bias[volatile]=None composable with sl=0.7?
    print("\n=== Compositions (180d) ===")
    compos = [
        ("baseline (current live)", None, None),
        ("sl=0.7", 0.7, None),
        ("dir_bias[volatile]=None", None, {'ranging': 'SHORT'}),  # drop volatile cell
        ("sl=0.7 + dir_bias[volatile]=None", 0.7, {'ranging': 'SHORT'}),
        ("sl=0.7 + dir_bias[volatile]=BOTH", 0.7, {'volatile': 'BOTH', 'ranging': 'SHORT'}),
    ]
    jobs = [(label, sl, db, None) for label, sl, db in compos]
    with Pool(6) as p:
        for r in p.imap(_bt, jobs):
            if "err" in r:
                print(f"  {r['label']:45s} ERR")
            else:
                print(f"  {r['label']:45s}  n={r['trades']:4d} PF={r['pf']:5.2f} PnL=${r['pnl']:+10,.0f} DD={r['dd']:.2f}% WR={r['wr']}%")

if __name__ == "__main__":
    main()
