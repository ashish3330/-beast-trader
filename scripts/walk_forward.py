"""
Walk-forward overfit detector for pass-1 tuned params.

For the top 10 symbols (by post-tune PnL in tune_180d_pass1.json):
  TRAIN window = most-recent 180d        (start_offset=0,   window=180)
  TEST  window = previous 180d           (start_offset=180, window=180)

Run identical (tuned) params on both windows. Verdict:
  ROBUST   — test pf >= 1.5 and test pnl > 0
  WEAK     — test pf 1.0..1.5 (marginal)
  OVERFIT  — train pf > 1.5 but test pf < 1.0 (or test pnl < 0)

Output: backtest/results/walk_forward.json
"""
import sys, json, os, time, pickle
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

# Original load_data preserved for reference
_orig_load_data = v5.load_data


def _windowed_load_data(symbol, days=90):
    """Drop-in replacement for v5.load_data that respects _WINDOW_OFFSET.

    Slices: [end - offset - window, end - offset]
    The `days` arg from the caller is ignored — we always use _WINDOW_DAYS.
    """
    meta = ALL_SYMBOLS[symbol]
    path = CACHE / meta["cache"]
    if not path.exists():
        return None
    df = pickle.load(open(path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    end = df["time"].max() - pd.Timedelta(days=_WINDOW_OFFSET)
    start = end - pd.Timedelta(days=_WINDOW_DAYS)
    df = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
    return df


def _set_window(offset_days, window_days=WINDOW_DAYS):
    global _WINDOW_OFFSET, _WINDOW_DAYS
    _WINDOW_OFFSET = int(offset_days)
    _WINDOW_DAYS = int(window_days)


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


def _worker_init():
    """Each worker patches its own copy of v5.load_data once."""
    v5.load_data = _windowed_load_data


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


def main():
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
    top10 = ranked[:10]

    print(f"\nWalk-forward: top 10 symbols by tuned PnL, {WORKERS} workers")
    print(f"  TRAIN window: now-{WINDOW_DAYS}d → now")
    print(f"  TEST  window: now-{2*WINDOW_DAYS}d → now-{WINDOW_DAYS}d")
    print()
    for s, pnl, _ in top10:
        print(f"  {s:14s} tuned_pnl=${pnl:>8.0f}")
    print()

    args = [(sym, params) for sym, _, params in top10]
    out = {}
    t0 = time.time()
    print(f"{'SYM':<14} {'TRAIN PF/PnL':>20} {'TEST PF/PnL':>20} {'ΔPF':>6} {'ΔPnL':>9}  VERDICT")
    print("-" * 90)
    with Pool(WORKERS, initializer=_worker_init) as pool:
        for sym, rec in pool.imap_unordered(_run_one, args):
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
        "window_days": WINDOW_DAYS,
        "workers": WORKERS,
        "elapsed_s": round(elapsed, 1),
        "top10_by_tuned_pnl": [s for s, _, _ in top10],
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
