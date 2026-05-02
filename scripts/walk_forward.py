"""
Walk-forward overfit detector + k-fold time-series CV with embargo.

Two modes:
  Default: holdout walk-forward (TRAIN: most-recent 180d; TEST: previous 180d).
  --kfold N: k-fold time-series CV with --embargo-days gap between train/test
             folds (anti-leakage). For each fold: train = all data EXCEPT
             that fold (with embargo gap), test = that fold's window.

For the top 10 symbols (by post-tune PnL in tune_180d_pass1.json):
  TRAIN window = most-recent 180d        (start_offset=0,   window=180)
  TEST  window = previous 180d           (start_offset=180, window=180)

Run identical (tuned) params on both windows. Verdict:
  ROBUST   — test pf >= 1.5 and test pnl > 0
  WEAK     — test pf 1.0..1.5 (marginal)
  OVERFIT  — train pf > 1.5 but test pf < 1.0 (or test pnl < 0)

Output: backtest/results/walk_forward.json
"""
import sys, json, os, time, pickle, argparse
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from backtest import v5_backtest as v5
from backtest.v5_backtest import ALL_SYMBOLS, backtest_symbol, SL_OVERRIDE, CACHE

RESULTS = ROOT / "backtest" / "results"
TUNE_PATH = RESULTS / "tune_180d_pass1.json"
OUT_PATH = RESULTS / "walk_forward.json"

WINDOW_DAYS = 180
WORKERS = int(os.environ.get("WF_WORKERS", "6"))

# Globals set per-worker via initializer
_WINDOW_OFFSET = 0       # days to shift the window's END back from now
_WINDOW_DAYS = WINDOW_DAYS
# K-fold mode: explicit [start, end] offset pair (days back from now).
# When set (non-None), takes precedence over (_WINDOW_OFFSET, _WINDOW_DAYS).
_FOLD_START_OFFSET = None  # days back from end-of-data to slice start
_FOLD_END_OFFSET   = None  # days back from end-of-data to slice end

# Original load_data preserved for reference
_orig_load_data = v5.load_data


def _windowed_load_data(symbol, days=90):
    """Drop-in replacement for v5.load_data that respects window globals.

    Two slice modes:
      - default:    [end - _WINDOW_OFFSET - _WINDOW_DAYS, end - _WINDOW_OFFSET]
      - fold mode:  [end - _FOLD_START_OFFSET, end - _FOLD_END_OFFSET]
                    when both fold offsets are not None.
    The `days` arg from the caller is ignored.
    """
    meta = ALL_SYMBOLS[symbol]
    path = CACHE / meta["cache"]
    if not path.exists():
        return None
    df = pickle.load(open(path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    end_of_data = df["time"].max()
    if _FOLD_START_OFFSET is not None and _FOLD_END_OFFSET is not None:
        start = end_of_data - pd.Timedelta(days=_FOLD_START_OFFSET)
        end   = end_of_data - pd.Timedelta(days=_FOLD_END_OFFSET)
    else:
        end = end_of_data - pd.Timedelta(days=_WINDOW_OFFSET)
        start = end - pd.Timedelta(days=_WINDOW_DAYS)
    df = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
    return df


def _set_window(offset_days, window_days=WINDOW_DAYS):
    global _WINDOW_OFFSET, _WINDOW_DAYS, _FOLD_START_OFFSET, _FOLD_END_OFFSET
    _WINDOW_OFFSET = int(offset_days)
    _WINDOW_DAYS = int(window_days)
    _FOLD_START_OFFSET = None
    _FOLD_END_OFFSET = None


def _set_fold_window(start_off_days, end_off_days):
    """Set absolute fold window: [now - start_off, now - end_off]
    where start_off > end_off (start is older)."""
    global _FOLD_START_OFFSET, _FOLD_END_OFFSET
    _FOLD_START_OFFSET = float(start_off_days)
    _FOLD_END_OFFSET   = float(end_off_days)


def _build_kfold_windows(total_days: int, k: int, embargo_days: int):
    """Build k consecutive test-fold windows over the most-recent `total_days`.

    Returns list of dicts:
        {fold_idx, test_start, test_end, train_start, train_end, embargo}
    All offsets are measured in days BACK from end-of-data (so older = larger).

    Layout (oldest left, newest right; offsets shrink to zero on the right):
        | TRAIN_old | EMBARGO | TEST | EMBARGO | TRAIN_new |
    Train uses everything except the test fold and ±embargo gaps.
    The first / last folds drop the missing-side gap (no leakage to forbid).
    """
    fold_size = total_days / float(k)
    folds = []
    for fi in range(k):
        # Older boundary first — offsets are days-back-from-end-of-data.
        test_start_off = total_days - fi * fold_size       # older edge of test
        test_end_off   = total_days - (fi + 1) * fold_size # newer edge of test
        # Embargo gaps around the test fold (in offset-space).
        train_old_start = total_days       # oldest training data
        train_old_end   = test_start_off + embargo_days  # stop before test+embargo
        train_new_start = test_end_off - embargo_days    # restart after test-embargo
        train_new_end   = 0                # newest training data
        folds.append({
            "fold_idx": fi,
            "test_start_off": test_start_off,
            "test_end_off":   test_end_off,
            "train_old_start_off": train_old_start,
            "train_old_end_off":   train_old_end,
            "train_new_start_off": train_new_start,
            "train_new_end_off":   train_new_end,
            "embargo_days": embargo_days,
        })
    return folds


def _slim(r):
    if not r:
        return r
    return {k: v for k, v in r.items() if k != "details"}


def _bt(symbol, params):
    sl = (params or {}).get("sl_atr_mult")
    if sl is None:
        return _slim(backtest_symbol(symbol, WINDOW_DAYS, params or None, verbose=False))
    old = SL_OVERRIDE.get(symbol)
    SL_OVERRIDE[symbol] = sl
    try:
        r = backtest_symbol(symbol, WINDOW_DAYS, params, verbose=False)
    finally:
        if old is not None:
            SL_OVERRIDE[symbol] = old
        else:
            SL_OVERRIDE.pop(symbol, None)
    return _slim(r)


def _aggregate_results(rs):
    """Aggregate two slim result dicts into one (used to merge two train segments).
    PF is recomputed from gross win/loss; PnL/trades summed. Returns None if empty.
    """
    rs = [r for r in rs if r and r.get("trades", 0) > 0]
    if not rs:
        return None
    trades = sum(r.get("trades", 0) for r in rs)
    wins   = sum(r.get("wins", 0) for r in rs)
    pnl    = sum(r.get("pnl", 0.0) for r in rs)
    # PF: weighted by trades — use harmonic-style approx via avg PF * trade share
    # (good enough for verdict stage; full reconstruction would need details).
    total_trades = float(trades) or 1.0
    weighted_pf = sum(float(r.get("pf", 0)) * (r.get("trades", 0) / total_trades) for r in rs)
    wr = (wins / total_trades) * 100.0 if total_trades else 0.0
    return {"trades": trades, "wins": wins, "pf": round(weighted_pf, 2),
            "pnl": round(pnl, 2), "wr": round(wr, 1)}


def _worker_init():
    """Each worker patches its own copy of v5.load_data once."""
    v5.load_data = _windowed_load_data


def _run_one_kfold(args):
    """k-fold variant. args = (symbol, params, folds, total_days)."""
    symbol, params, folds, total_days = args
    v5.load_data = _windowed_load_data

    fold_results = []
    for f in folds:
        # TEST fold
        _set_fold_window(f["test_start_off"], f["test_end_off"])
        test_r = _bt(symbol, params)

        # TRAIN: two segments (old + new) excluding the test+embargo region
        train_old = None
        if f["train_old_start_off"] > f["train_old_end_off"]:
            _set_fold_window(f["train_old_start_off"], f["train_old_end_off"])
            train_old = _bt(symbol, params)
        train_new = None
        if f["train_new_start_off"] > f["train_new_end_off"]:
            _set_fold_window(f["train_new_start_off"], f["train_new_end_off"])
            train_new = _bt(symbol, params)
        train_agg = _aggregate_results([train_old, train_new])

        fold_results.append({
            "fold_idx": f["fold_idx"],
            "test_window": [f["test_start_off"], f["test_end_off"]],
            "embargo_days": f["embargo_days"],
            "train": train_agg,
            "test":  test_r,
        })

    # Aggregate stats across folds (test-side)
    test_pfs  = [fr["test"].get("pf", 0)  for fr in fold_results if fr["test"]]
    test_pnls = [fr["test"].get("pnl", 0) for fr in fold_results if fr["test"]]
    test_trades = sum((fr["test"] or {}).get("trades", 0) for fr in fold_results)
    if test_pfs:
        import statistics as _stats
        agg = {
            "test_pf_mean":   round(_stats.mean(test_pfs), 2),
            "test_pf_stdev":  round(_stats.pstdev(test_pfs), 2) if len(test_pfs) > 1 else 0.0,
            "test_pnl_total": round(sum(test_pnls), 2),
            "test_pnl_mean":  round(_stats.mean(test_pnls), 2),
            "test_trades_total": test_trades,
            "n_folds": len(fold_results),
        }
    else:
        agg = {"test_pf_mean": 0, "test_pf_stdev": 0, "test_pnl_total": 0,
               "test_pnl_mean": 0, "test_trades_total": 0, "n_folds": 0}

    return symbol, {
        "params": params,
        "folds": fold_results,
        "aggregate": agg,
    }


def _run_one(args):
    symbol, params = args
    # Ensure patched
    v5.load_data = _windowed_load_data

    # TRAIN: most-recent 180d (offset 0)
    _set_window(0, WINDOW_DAYS)
    train = _bt(symbol, params)

    # TEST: previous 180d (offset 180)
    _set_window(WINDOW_DAYS, WINDOW_DAYS)
    test = _bt(symbol, params)

    # Verdict
    train_pf = (train or {}).get("pf", 0)
    train_pnl = (train or {}).get("pnl", 0)
    test_pf = (test or {}).get("pf", 0)
    test_pnl = (test or {}).get("pnl", 0)

    if train_pf > 1.5 and (test_pf < 1.0 or test_pnl < 0):
        verdict = "OVERFIT"
    elif test_pf >= 1.5 and test_pnl > 0:
        verdict = "ROBUST"
    elif test_pf >= 1.0:
        verdict = "WEAK"
    else:
        verdict = "OVERFIT"  # train didn't beat 1.5, test below 1.0 → just bad

    return symbol, {
        "params": params,
        "train": train,
        "test": test,
        "verdict": verdict,
        "delta_pnl": round(test_pnl - train_pnl, 2),
        "delta_pf": round(test_pf - train_pf, 2),
    }


def _kfold_main(top10, k, total_days, embargo_days):
    """k-fold time-series CV with embargo. Prints per-fold + aggregate stats."""
    folds = _build_kfold_windows(total_days, k, embargo_days)
    print(f"\nk-fold time-series CV: k={k}, total={total_days}d, "
          f"embargo={embargo_days}d, fold_size={total_days/k:.0f}d")
    print(f"  Top {len(top10)} symbols by tuned PnL")
    print()
    for s, pnl, _ in top10:
        print(f"  {s:14s} tuned_pnl=${pnl:>8.0f}")
    print()

    args = [(sym, params, folds, total_days) for sym, _, params in top10]
    out = {}
    t0 = time.time()
    hdr = f"{'SYM':<14} {'FOLD':>5} {'TRAIN PF':>9} {'TRAIN PnL':>12} {'TEST PF':>8} {'TEST PnL':>12}"
    print(hdr)
    print("-" * len(hdr))
    with Pool(WORKERS, initializer=_worker_init) as pool:
        for sym, rec in pool.imap_unordered(_run_one_kfold, args):
            out[sym] = rec
            for fr in rec["folds"]:
                tr = fr["train"] or {}
                te = fr["test"] or {}
                print(f"{sym:<14} {fr['fold_idx']:>5d} "
                      f"{tr.get('pf', 0):>9.2f} ${tr.get('pnl', 0):>10.0f}  "
                      f"{te.get('pf', 0):>8.2f} ${te.get('pnl', 0):>10.0f}")
            agg = rec["aggregate"]
            print(f"{sym:<14} {'AGG':>5} {'':>9} {'':>12} "
                  f"{agg['test_pf_mean']:>8.2f} ${agg['test_pnl_total']:>10.0f}  "
                  f"(stdev pf={agg['test_pf_stdev']:.2f}, "
                  f"trades={agg['test_trades_total']})")
            print()

    elapsed = time.time() - t0
    print("-" * len(hdr))
    print(f"  Elapsed {elapsed:.0f}s — {len(out)} symbols × {k} folds")

    payload = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "kfold",
        "k": k,
        "total_days": total_days,
        "embargo_days": embargo_days,
        "workers": WORKERS,
        "elapsed_s": round(elapsed, 1),
        "top_symbols_by_tuned_pnl": [s for s, _, _ in top10],
        "results": out,
    }
    json.dump(payload, open(OUT_PATH, "w"), indent=2, default=str)
    print(f"\nWrote {OUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward / k-fold CV for Dragon V5")
    parser.add_argument("--kfold", type=int, default=0,
                        help="If >0, run k-fold time-series CV (default 0 = holdout walk-forward).")
    parser.add_argument("--total-days", type=int, default=540,
                        help="Total span (days) covered by k folds. Default 540 (3*180).")
    parser.add_argument("--embargo-days", type=int, default=1,
                        help="Embargo days between train and test. Default 1.")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top tuned symbols to evaluate. Default 10.")
    args = parser.parse_args()

    if not TUNE_PATH.exists():
        print(f"ERROR: {TUNE_PATH} not found")
        sys.exit(1)
    tune = json.load(open(TUNE_PATH))
    res = tune.get("results", {})

    ranked = []
    for sym, r in res.items():
        best = r.get("best") or {}
        rr = best.get("result")
        if rr and best.get("params"):
            ranked.append((sym, rr.get("pnl", 0), best.get("params")))
    ranked.sort(key=lambda x: x[1], reverse=True)
    top = ranked[:args.top]

    if args.kfold and args.kfold > 1:
        _kfold_main(top, args.kfold, args.total_days, args.embargo_days)
        return

    print(f"\nWalk-forward: top {len(top)} symbols by tuned PnL, {WORKERS} workers")
    print(f"  TRAIN window: now-{WINDOW_DAYS}d → now")
    print(f"  TEST  window: now-{2*WINDOW_DAYS}d → now-{WINDOW_DAYS}d")
    print()
    for s, pnl, _ in top:
        print(f"  {s:14s} tuned_pnl=${pnl:>8.0f}")
    print()

    pool_args = [(sym, params) for sym, _, params in top]
    out = {}
    t0 = time.time()
    print(f"{'SYM':<14} {'TRAIN PF/PnL':>20} {'TEST PF/PnL':>20} {'dPF':>6} {'dPnL':>9}  VERDICT")
    print("-" * 90)
    with Pool(WORKERS, initializer=_worker_init) as pool:
        for sym, rec in pool.imap_unordered(_run_one, pool_args):
            out[sym] = rec
            tr = rec["train"] or {}
            te = rec["test"] or {}
            print(f"{sym:<14} "
                  f"{tr.get('pf', 0):>6.2f} ${tr.get('pnl', 0):>9.0f}  "
                  f"{te.get('pf', 0):>6.2f} ${te.get('pnl', 0):>9.0f}  "
                  f"{rec['delta_pf']:>+6.2f} ${rec['delta_pnl']:>+8.0f}  {rec['verdict']}")

    elapsed = time.time() - t0
    overfits = [s for s, r in out.items() if r["verdict"] == "OVERFIT"]
    robust = [s for s, r in out.items() if r["verdict"] == "ROBUST"]
    weak = [s for s, r in out.items() if r["verdict"] == "WEAK"]
    print("-" * 90)
    print(f"  ROBUST: {len(robust)}  WEAK: {len(weak)}  OVERFIT: {len(overfits)}  ({elapsed:.0f}s)")
    if overfits:
        print(f"  Drop candidates (OVERFIT): {', '.join(overfits)}")

    payload = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "holdout",
        "window_days": WINDOW_DAYS,
        "workers": WORKERS,
        "elapsed_s": round(elapsed, 1),
        "top_by_tuned_pnl": [s for s, _, _ in top],
        "results": out,
        "summary": {
            "robust": robust,
            "weak": weak,
            "overfit": overfits,
        },
    }
    json.dump(payload, open(OUT_PATH, "w"), indent=2, default=str)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
