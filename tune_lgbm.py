"""
Dragon Trader — LightGBM Hyperparameter Tuning Script.
Monkey-patches the params dict in SignalModel.train() to test
different hyperparameter combinations for XAUUSD and XAGUSD.
"""
import pickle, sys, time, logging, numpy as np, pandas as pd
from itertools import product

logging.basicConfig(level=logging.WARNING)  # suppress training logs

sys.path.insert(0, "/Users/ashish/Documents/beast-trader")

CACHE_DIR = "/Users/ashish/Documents/xauusd-trading-bot/cache"
SYMBOL_CACHE = {
    "XAUUSD": f"{CACHE_DIR}/raw_h1_xauusd.pkl",
    "XAGUSD": f"{CACHE_DIR}/raw_h1_XAGUSD.pkl",
}


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


# Hyperparameter grid
GRID = {
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
    "num_leaves":       [15, 31, 63, 127],
    "max_depth":        [3, 5, 7, -1],
    "feature_fraction": [0.5, 0.7, 0.9],
    "bagging_fraction": [0.5, 0.7, 0.9],
    "lambda_l1":        [0.0, 0.1, 1.0],
    "lambda_l2":        [0.0, 1.0, 5.0],
}

# Phase 1: Coarse grid on lr, leaves, depth (most impactful)
PHASE1_GRID = list(product(
    GRID["learning_rate"],
    GRID["num_leaves"],
    GRID["max_depth"],
))

print(f"Phase 1: {len(PHASE1_GRID)} combos x 2 symbols = {len(PHASE1_GRID)*2} runs")
print("=" * 80)

import models.signal_model as sm
import lightgbm as lgb

# Store original train to monkey-patch
_orig_train_fn = lgb.train


def make_patched_lgb_train(override_params):
    """Create a patched lgb.train that injects override params."""
    def patched_train(params, *args, **kwargs):
        params.update(override_params)
        return _orig_train_fn(params, *args, **kwargs)
    return patched_train


mock = MockMT5()
results = {}

for sym in ["XAUUSD", "XAGUSD"]:
    print(f"\n{'='*80}")
    print(f"  TUNING {sym}  —  Phase 1: lr x leaves x depth")
    print(f"{'='*80}")

    sym_results = []
    best_auc = 0
    best_combo = None

    for i, (lr, leaves, depth) in enumerate(PHASE1_GRID):
        override = {
            "learning_rate": lr,
            "num_leaves": leaves,
            "max_depth": depth,
        }

        # Monkey-patch lgb.train
        lgb.train = make_patched_lgb_train(override)

        try:
            model = sm.SignalModel()
            metrics = model.train(sym, mock, None)

            if metrics and metrics.get("status") == "ok":
                auc = metrics["test_auc"]
                pf = metrics["filtered_pf"]
                prec = metrics["precision_at_conf"]
                trees = metrics["n_trees"]

                sym_results.append({
                    "lr": lr, "leaves": leaves, "depth": depth,
                    "auc": auc, "pf": pf, "prec": prec, "trees": trees,
                })

                marker = ""
                if auc > best_auc:
                    best_auc = auc
                    best_combo = (lr, leaves, depth)
                    marker = " *** NEW BEST ***"

                if (i + 1) % 10 == 0 or marker:
                    print(f"  [{i+1:3d}/{len(PHASE1_GRID)}] lr={lr:.2f} leaves={leaves:3d} depth={depth:2d}  "
                          f"AUC={auc:.4f}  PF={pf:.2f}  Prec={prec:.3f}  Trees={trees}{marker}")
        except Exception as e:
            pass
        finally:
            lgb.train = _orig_train_fn

    # Sort by AUC
    sym_results.sort(key=lambda x: x["auc"], reverse=True)
    results[sym] = sym_results

    print(f"\n  TOP 10 for {sym}:")
    print(f"  {'lr':>5s}  {'leaves':>6s}  {'depth':>5s}  {'AUC':>6s}  {'PF':>6s}  {'Prec':>5s}  {'Trees':>5s}")
    for r in sym_results[:10]:
        print(f"  {r['lr']:5.2f}  {r['leaves']:6d}  {r['depth']:5d}  "
              f"{r['auc']:6.4f}  {r['pf']:6.2f}  {r['prec']:5.3f}  {r['trees']:5d}")

# Phase 2: Fine-tune regularization on top-1 from Phase 1
print(f"\n{'='*80}")
print("  PHASE 2: Fine-tune regularization (feature_fraction, bagging, L1, L2)")
print(f"{'='*80}")

PHASE2_GRID = list(product(
    GRID["feature_fraction"],
    GRID["bagging_fraction"],
    GRID["lambda_l1"],
    GRID["lambda_l2"],
))

print(f"Phase 2: {len(PHASE2_GRID)} combos x 2 symbols = {len(PHASE2_GRID)*2} runs")

for sym in ["XAUUSD", "XAGUSD"]:
    if not results.get(sym):
        continue

    best_p1 = results[sym][0]
    base_lr = best_p1["lr"]
    base_leaves = best_p1["leaves"]
    base_depth = best_p1["depth"]

    print(f"\n  {sym}: Phase 1 winner — lr={base_lr}, leaves={base_leaves}, depth={base_depth}, AUC={best_p1['auc']:.4f}")
    print(f"  Testing {len(PHASE2_GRID)} regularization combos...")

    p2_results = []
    best_auc = 0

    for i, (ff, bf, l1, l2) in enumerate(PHASE2_GRID):
        override = {
            "learning_rate": base_lr,
            "num_leaves": base_leaves,
            "max_depth": base_depth,
            "feature_fraction": ff,
            "bagging_fraction": bf,
            "lambda_l1": l1,
            "lambda_l2": l2,
        }

        lgb.train = make_patched_lgb_train(override)

        try:
            model = sm.SignalModel()
            metrics = model.train(sym, mock, None)

            if metrics and metrics.get("status") == "ok":
                auc = metrics["test_auc"]
                pf = metrics["filtered_pf"]
                prec = metrics["precision_at_conf"]
                trees = metrics["n_trees"]

                p2_results.append({
                    "lr": base_lr, "leaves": base_leaves, "depth": base_depth,
                    "ff": ff, "bf": bf, "l1": l1, "l2": l2,
                    "auc": auc, "pf": pf, "prec": prec, "trees": trees,
                })

                marker = ""
                if auc > best_auc:
                    best_auc = auc
                    marker = " *** NEW BEST ***"

                if (i + 1) % 15 == 0 or marker:
                    print(f"    [{i+1:3d}/{len(PHASE2_GRID)}] ff={ff:.1f} bf={bf:.1f} l1={l1:.1f} l2={l2:.1f}  "
                          f"AUC={auc:.4f}  PF={pf:.2f}  Prec={prec:.3f}{marker}")
        except Exception:
            pass
        finally:
            lgb.train = _orig_train_fn

    p2_results.sort(key=lambda x: x["auc"], reverse=True)

    print(f"\n  TOP 10 FINAL for {sym}:")
    print(f"  {'lr':>5s}  {'leaves':>6s}  {'depth':>5s}  {'ff':>4s}  {'bf':>4s}  {'l1':>4s}  {'l2':>4s}  "
          f"{'AUC':>6s}  {'PF':>6s}  {'Prec':>5s}  {'Trees':>5s}")
    for r in p2_results[:10]:
        print(f"  {r['lr']:5.2f}  {r['leaves']:6d}  {r['depth']:5d}  "
              f"{r['ff']:4.1f}  {r['bf']:4.1f}  {r['l1']:4.1f}  {r['l2']:4.1f}  "
              f"{r['auc']:6.4f}  {r['pf']:6.2f}  {r['prec']:5.3f}  {r['trees']:5d}")

    if p2_results:
        best = p2_results[0]
        print(f"\n  OPTIMAL {sym}:")
        print(f"    learning_rate:    {best['lr']}")
        print(f"    num_leaves:       {best['leaves']}")
        print(f"    max_depth:        {best['depth']}")
        print(f"    feature_fraction: {best['ff']}")
        print(f"    bagging_fraction: {best['bf']}")
        print(f"    lambda_l1:        {best['l1']}")
        print(f"    lambda_l2:        {best['l2']}")
        print(f"    AUC:              {best['auc']:.4f}")
        print(f"    Filtered PF:      {best['pf']:.2f}")
        print(f"    Precision@conf:   {best['prec']:.3f}")

print("\n" + "=" * 80)
print("TUNING COMPLETE")
print("=" * 80)
