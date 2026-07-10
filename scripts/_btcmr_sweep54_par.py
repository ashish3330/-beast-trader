#!/usr/bin/env python3 -B
"""Parallel 54-config BTCMR sweep (CONFIRM=0, SL_ATR=1 fixed).
Runs full 365d + fold3-of-4 per config via override dicts (env-free workers).
Writes incremental JSONL to /tmp/btcmr_sweep54_sl1_par.jsonl
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = "/tmp/btcmr_sweep54_sl1_par.jsonl"

# Scrub any BTCMR_ env so override dicts win
for k in list(os.environ):
    if k.startswith("BTCMR_"):
        del os.environ[k]

CONFIGS = []
for bb in (2.0, 2.5, 3.0):
    for rlo, rhi in ((10.0, 90.0), (20.0, 80.0), (30.0, 70.0)):
        for adx in (20.0, 25.0, 30.0):
            for ts in (16, 24):
                CONFIGS.append({
                    "CONFIRM": 0, "SL_ATR": 1,
                    "BB_MULT": bb, "RSI_LOW": rlo, "RSI_HIGH": rhi,
                    "ADX_MAX": adx, "TIME_STOP_BARS": ts,
                })

TASKS = []
for idx, cfg in enumerate(CONFIGS):
    TASKS.append((idx, "full", cfg))
    TASKS.append((idx, "fold3", cfg))


def work(task):
    idx, mode, cfg = task
    for k in list(os.environ):
        if k.startswith("BTCMR_"):
            del os.environ[k]
    import _btcmr_run as runner
    if mode == "full":
        res = runner.run("BTCUSD", days=365, override=cfg)
    else:
        res = runner.run("BTCUSD", days=365, fold=3, folds=4, override=cfg)
    return {"idx": idx, "mode": mode, "cfg": cfg, "res": res}


if __name__ == "__main__":
    with open(OUT, "w") as f:
        pass
    with mp.Pool(6) as pool:
        n_done = 0
        with open(OUT, "a") as f:
            for row in pool.imap_unordered(work, TASKS):
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
                print(f"[{n_done}/{len(TASKS)}] idx={row['idx']} {row['mode']} "
                      f"pf={row['res'].get('pf')} trades={row['res'].get('trades')}",
                      flush=True)
    print("SWEEP COMPLETE", flush=True)
