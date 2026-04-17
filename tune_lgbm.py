"""
Dragon Trader — LightGBM Hyperparameter Tuning Script.
Two-phase grid search: Phase 1 (lr x leaves x depth), Phase 2 (regularization).
"""
import os, sys, pickle, time, logging, numpy as np, pandas as pd
from itertools import product

os.environ["PYTHONUNBUFFERED"] = "1"

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, "/Users/ashish/Documents/beast-trader")

CACHE_DIR = "/Users/ashish/Documents/xauusd-trading-bot/cache"
SYMBOL_CACHE = {
    "XAUUSD": f"{CACHE_DIR}/raw_h1_xauusd.pkl",
    "XAGUSD": f"{CACHE_DIR}/raw_h1_XAGUSD.pkl",
}

def p(*args, **kwargs):
    print(*args, **kwargs, flush=True)


class MockMT5:
    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        path = SYMBOL_CACHE.get(symbol)
        if not path:
            return None
        with open(path, "rb") as f:
            df = pickle.load(f)
        result = np.zeros(len(df), dtype=[
            ("time", "i8"), ("open", "f8"), ("high", "f8"),
            ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
            ("spread", "i4"), ("real_volume", "i8"),
        ])
        if hasattr(df["time"].iloc[0], "timestamp"):
            result["time"] = df["time"].apply(lambda x: int(x.timestamp())).values
        else:
            result["time"] = df["time"].values.astype("int64") // 10**9
        for c in ["open", "high", "low", "close"]:
            result[c] = df[c].values
        result["tick_volume"] = df["tick_volume"].values
        result["spread"] = df["spread"].values
        result["real_volume"] = df["real_volume"].values
        return result


import models.signal_model as sm
import lightgbm as lgb

_orig_lgb_train = lgb.train


def make_patched(overrides):
    def patched(params, *a, **kw):
        params.update(overrides)
        return _orig_lgb_train(params, *a, **kw)
    return patched


mock = MockMT5()

# ══════════════════════════════════════════════════════════════
# PHASE 1: lr x leaves x depth
# ══════════════════════════════════════════════════════════════
P1 = list(product(
    [0.01, 0.03, 0.05, 0.08, 0.1],    # lr
    [15, 31, 63, 127],                  # leaves
    [3, 5, 7, -1],                      # depth
))
p(f"Phase 1: {len(P1)} combos x 2 symbols")
p("=" * 90)

phase1_results = {}

for sym in ["XAUUSD", "XAGUSD"]:
    p(f"\n{'='*90}")
    p(f"  {sym} — Phase 1")
    p(f"{'='*90}")

    rows = []
    best_auc = 0
    t0 = time.time()

    for i, (lr, lv, dp) in enumerate(P1):
        lgb.train = make_patched({"learning_rate": lr, "num_leaves": lv, "max_depth": dp})
        try:
            m = sm.SignalModel()
            met = m.train(sym, mock, None)
            if met and met.get("status") == "ok":
                auc = met["test_auc"]
                pf = met["filtered_pf"]
                prec = met["precision_at_conf"]
                trees = met["n_trees"]
                rows.append({"lr": lr, "lv": lv, "dp": dp, "auc": auc, "pf": pf, "prec": prec, "trees": trees})
                tag = ""
                if auc > best_auc:
                    best_auc = auc
                    tag = " <<< BEST"
                if tag or (i+1) % 20 == 0:
                    p(f"  [{i+1:3d}/{len(P1)}] lr={lr:.2f} lv={lv:3d} dp={dp:2d}  AUC={auc:.4f}  PF={pf:.2f}  Prec={prec:.3f}  T={trees}{tag}")
        except Exception as e:
            pass
        finally:
            lgb.train = _orig_lgb_train

    elapsed = time.time() - t0
    p(f"\n  {sym} Phase 1 done in {elapsed:.0f}s ({len(rows)} successful)")

    rows.sort(key=lambda x: x["auc"], reverse=True)
    phase1_results[sym] = rows

    p(f"\n  TOP 15 for {sym}:")
    p(f"  {'lr':>5}  {'lv':>4}  {'dp':>3}  {'AUC':>7}  {'PF':>7}  {'Prec':>6}  {'Trees':>5}")
    for r in rows[:15]:
        p(f"  {r['lr']:5.2f}  {r['lv']:4d}  {r['dp']:3d}  {r['auc']:7.4f}  {r['pf']:7.2f}  {r['prec']:6.3f}  {r['trees']:5d}")

# ══════════════════════════════════════════════════════════════
# PHASE 2: Regularization on Phase 1 winner
# ══════════════════════════════════════════════════════════════
P2 = list(product(
    [0.5, 0.7, 0.9],    # feature_fraction
    [0.5, 0.7, 0.9],    # bagging_fraction
    [0.0, 0.1, 1.0],    # lambda_l1
    [0.0, 1.0, 5.0],    # lambda_l2
))

p(f"\n{'='*90}")
p(f"  PHASE 2: Regularization ({len(P2)} combos x 2 symbols)")
p(f"{'='*90}")

for sym in ["XAUUSD", "XAGUSD"]:
    if not phase1_results.get(sym):
        continue
    b = phase1_results[sym][0]
    lr, lv, dp = b["lr"], b["lv"], b["dp"]
    p(f"\n  {sym}: Base = lr={lr} lv={lv} dp={dp} AUC={b['auc']:.4f}")

    rows = []
    best_auc = 0
    t0 = time.time()

    for i, (ff, bf, l1, l2) in enumerate(P2):
        lgb.train = make_patched({
            "learning_rate": lr, "num_leaves": lv, "max_depth": dp,
            "feature_fraction": ff, "bagging_fraction": bf,
            "lambda_l1": l1, "lambda_l2": l2,
        })
        try:
            m = sm.SignalModel()
            met = m.train(sym, mock, None)
            if met and met.get("status") == "ok":
                auc = met["test_auc"]
                pf = met["filtered_pf"]
                prec = met["precision_at_conf"]
                trees = met["n_trees"]
                rows.append({"ff": ff, "bf": bf, "l1": l1, "l2": l2,
                             "auc": auc, "pf": pf, "prec": prec, "trees": trees})
                tag = ""
                if auc > best_auc:
                    best_auc = auc
                    tag = " <<< BEST"
                if tag or (i+1) % 20 == 0:
                    p(f"    [{i+1:3d}/{len(P2)}] ff={ff:.1f} bf={bf:.1f} l1={l1:.1f} l2={l2:.1f}  "
                      f"AUC={auc:.4f}  PF={pf:.2f}  Prec={prec:.3f}{tag}")
        except Exception:
            pass
        finally:
            lgb.train = _orig_lgb_train

    elapsed = time.time() - t0
    p(f"\n  {sym} Phase 2 done in {elapsed:.0f}s")

    rows.sort(key=lambda x: x["auc"], reverse=True)

    p(f"\n  TOP 10 FINAL for {sym} (lr={lr} lv={lv} dp={dp}):")
    p(f"  {'ff':>4}  {'bf':>4}  {'l1':>4}  {'l2':>4}  {'AUC':>7}  {'PF':>7}  {'Prec':>6}  {'Trees':>5}")
    for r in rows[:10]:
        p(f"  {r['ff']:4.1f}  {r['bf']:4.1f}  {r['l1']:4.1f}  {r['l2']:4.1f}  "
          f"{r['auc']:7.4f}  {r['pf']:7.2f}  {r['prec']:6.3f}  {r['trees']:5d}")

    if rows:
        best = rows[0]
        p(f"\n  *** OPTIMAL {sym} ***")
        p(f"    learning_rate:    {lr}")
        p(f"    num_leaves:       {lv}")
        p(f"    max_depth:        {dp}")
        p(f"    feature_fraction: {best['ff']}")
        p(f"    bagging_fraction: {best['bf']}")
        p(f"    lambda_l1:        {best['l1']}")
        p(f"    lambda_l2:        {best['l2']}")
        p(f"    AUC:              {best['auc']:.4f}")
        p(f"    Filtered PF:      {best['pf']:.2f}")
        p(f"    Precision@conf:   {best['prec']:.3f}")

p(f"\n{'='*90}")
p("TUNING COMPLETE")
p(f"{'='*90}")
