#!/usr/bin/env python3 -B
"""3-fold walk-forward validation for JPN225ft entry-sweep winner.

Winner params (frozen from in-sample 90d tune):
  MIN_SCORE = 35
  PULLBACK_ATR_RETRACE = 0.5
  PULLBACK_MAX_WAIT_BARS = 1

Folds (non-overlapping, 60d each):
  fold 1: days -180 → -120
  fold 2: days -120 → -60
  fold 3: days -60  → 0

Decision: SHIP if 2/3 or 3/3 folds have pnl > 0. NULL otherwise.

Mechanism: monkey-patch backtest.v5_backtest.load_data to slice cache
to a [t_max - end_days, t_max - start_days] window, then call
backtest_symbol(SYMBOL, days=None, params=...).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pickle  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from backtest import v5_backtest  # noqa: E402
from backtest.v5_backtest import DEFAULT_PARAMS, ALL_SYMBOLS, CACHE  # noqa: E402

SYMBOL = "JPN225ft"
OUT = ROOT / "backtest" / "results" / "entry_sweep_20260614" / f"{SYMBOL}_wf.json"

# Winner params
WINNER_MIN_SCORE = 35
WINNER_RETRACE = 0.5
WINNER_WAIT = 1

# Folds: (label, end_days_ago, start_days_ago) — slice = [t_max - end, t_max - start]
# fold 1: -180 to -120 → start_days=180, end_days=120
# fold 2: -120 to -60  → start_days=120, end_days=60
# fold 3: -60  to 0    → start_days=60,  end_days=0
FOLDS = [
    {"name": "fold1_-180_-120", "start_days": 180, "end_days": 120},
    {"name": "fold2_-120_-60",  "start_days": 120, "end_days": 60},
    {"name": "fold3_-60_0",     "start_days": 60,  "end_days": 0},
]

# Preload meta model (mirrors entry_sweep behaviour)
META = None
try:
    from models.signal_model import SignalModel
    _m = SignalModel(); _m.load(SYMBOL)
    if _m.has_model(SYMBOL):
        META = _m
except Exception:
    META = None


def make_window_loader(start_days, end_days):
    """Return a load_data shim that slices to the fold window.
    window = [t_max - start_days, t_max - end_days)
    """
    def _load(symbol, days=90):
        meta = ALL_SYMBOLS[symbol]
        path = CACHE / meta["cache"]
        if not path.exists():
            print(f"  {symbol}: cache not found at {path}")
            return None
        df = pickle.load(open(path, "rb"))
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        t_max = df["time"].max()
        lo = t_max - pd.Timedelta(days=start_days)
        hi = t_max - pd.Timedelta(days=end_days)
        df = df[(df["time"] >= lo) & (df["time"] < hi)].reset_index(drop=True)
        return df
    return _load


def run_fold(fold):
    """Run one fold with the frozen winner params."""
    # Patch SIGNAL_QUALITY_SYMBOL for this fold
    orig_q = dict(config.SIGNAL_QUALITY_SYMBOL.get(SYMBOL, {}))
    config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
        "trending": WINNER_MIN_SCORE,
        "ranging": WINNER_MIN_SCORE,
        "volatile": WINNER_MIN_SCORE,
        "low_vol": WINNER_MIN_SCORE,
    }

    p = {**DEFAULT_PARAMS, "audit_fix_gates": True}
    p["pullback_atr_retrace"] = WINNER_RETRACE
    p["pullback_max_wait"] = WINNER_WAIT
    if META is not None:
        p["_meta_model"] = META

    # Patch load_data
    orig_load = v5_backtest.load_data
    v5_backtest.load_data = make_window_loader(fold["start_days"], fold["end_days"])

    try:
        r = v5_backtest.backtest_symbol(SYMBOL, days=None, params=p, verbose=False)
    except Exception as e:
        r = {"error": str(e)}
    finally:
        v5_backtest.load_data = orig_load
        config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = orig_q

    if not r or "error" in r:
        return {
            "fold": fold["name"],
            "trades": 0, "pf": 0, "pnl": 0, "wr": 0, "dd": 0,
            "error": (r or {}).get("error", "no result"),
        }
    return {
        "fold": fold["name"],
        "trades": int(r.get("trades", 0)),
        "pf": round(float(r.get("pf", 0) or 0), 3),
        "pnl": round(float(r.get("pnl", 0) or 0), 2),
        "wr": round(float(r.get("wr", 0) or 0), 1),
        "dd": round(float(r.get("dd", 0) or 0), 2),
    }


def main():
    t0 = time.time()
    print(f"=== WF VALIDATION {SYMBOL} (3x60d non-overlapping) ===")
    print(f"Winner: MIN_SCORE={WINNER_MIN_SCORE} RETRACE={WINNER_RETRACE} WAIT={WINNER_WAIT}")

    fold_results = []
    for i, fold in enumerate(FOLDS, 1):
        print(f"\n[Fold {i}/3] {fold['name']} ...")
        r = run_fold(fold)
        fold_results.append(r)
        print(f"  trades={r['trades']} pf={r['pf']} pnl=${r['pnl']} wr={r.get('wr', 0)}% dd={r.get('dd', 0)}")

    pos = sum(1 for r in fold_results if r["pnl"] > 0)
    decision = "SHIP" if pos >= 2 else "NULL"
    reason = f"{pos}/3 folds pnl>0"

    out = {
        "symbol": SYMBOL,
        "winner_params": {
            "MIN_SCORE": WINNER_MIN_SCORE,
            "PULLBACK_ATR_RETRACE": WINNER_RETRACE,
            "PULLBACK_MAX_WAIT_BARS": WINNER_WAIT,
        },
        "folds": fold_results,
        "folds_positive": pos,
        "folds_run": len(fold_results),
        "wf_decision": decision,
        "wf_reason": reason,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {OUT}")
    print(f"DECISION: {decision} — {reason}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
