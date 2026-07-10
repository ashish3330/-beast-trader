#!/usr/bin/env python3 -B
"""5-fold STRICT walk-forward validation for DJ30.r / momentum.

FROZEN winner_config (no per-fold re-tuning):
  SL_ATR_MULT      = 0.7
  PULLBACK_RETRACE = 0.3
  PULLBACK_WAIT    = 1
  SIGNAL_QUALITY   = {trending:45, ranging:50, volatile:45, low_vol:55}
  DIRECTION_BIAS   = SHORT

Folds (non-overlapping 90d, covering days -450..0):
  fold 1: days -450 -> -360
  fold 2: days -360 -> -270
  fold 3: days -270 -> -180
  fold 4: days -180 -> -90
  fold 5: days  -90 ->   0

STRICT SHIP gate:
  folds_pf_ok    >= 4    (PF >= 1.5 in >=4 folds)
  folds_pnl_ok   >= 4    (PnL > 0 in >=4 folds)
NULL otherwise.

Mechanism: monkey-patch backtest.v5_backtest.load_data to slice cache to a
[t_max - start_days, t_max - end_days) window per fold; patch SL_OVERRIDE,
SIGNAL_QUALITY_SYMBOL, DIR_BIAS; pass force_direction + pullback retrace/wait
in params; preload meta-model for ML parity.
"""
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import pickle  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from backtest import v5_backtest  # noqa: E402
from backtest.v5_backtest import (  # noqa: E402
    DEFAULT_PARAMS, ALL_SYMBOLS, CACHE, SL_OVERRIDE, DIR_BIAS,
)

SYMBOL = "DJ30.r"
STRATEGY = "momentum"
OUT_DIR = ROOT / "backtest" / "results" / "wf_strict5_dj30r_winner_20260617"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"{SYMBOL}_momentum_5fold_strict.json"

# ============================================================
# Winner config (FROZEN)
# ============================================================
WINNER_SL_ATR_MULT = 0.7
WINNER_PULLBACK_RETRACE = 0.3
WINNER_PULLBACK_WAIT = 1
WINNER_SIGNAL_QUALITY = {
    "trending": 45,
    "ranging": 50,
    "volatile": 45,
    "low_vol": 55,
}
WINNER_DIR_BIAS = "SHORT"
DIR_MAP = {"LONG": 1, "SHORT": -1, "BOTH": 0}

# Folds: non-overlapping 90d, days -450..0
FOLDS = [
    {"name": "fold1_-450_-360", "start_days": 450, "end_days": 360},
    {"name": "fold2_-360_-270", "start_days": 360, "end_days": 270},
    {"name": "fold3_-270_-180", "start_days": 270, "end_days": 180},
    {"name": "fold4_-180_-90",  "start_days": 180, "end_days":  90},
    {"name": "fold5_-90_-0",    "start_days":  90, "end_days":   0},
]

# STRICT gate constants (user spec — no max-fold-loss constraint)
PF_THRESHOLD = 1.5
PF_OK_MIN = 4
PNL_OK_MIN = 4

# Snapshot originals so live config is not contaminated
_ORIG_SL = SL_OVERRIDE.get(SYMBOL, None)
_ORIG_DIR = DIR_BIAS.get(SYMBOL, None)
_ORIG_Q_CFG = copy.deepcopy(config.SIGNAL_QUALITY_SYMBOL.get(SYMBOL, {}))


def apply_winner_config():
    SL_OVERRIDE[SYMBOL] = float(WINNER_SL_ATR_MULT)
    config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = dict(WINNER_SIGNAL_QUALITY)
    DIR_BIAS[SYMBOL] = DIR_MAP[WINNER_DIR_BIAS]


def restore_originals():
    if _ORIG_SL is not None:
        SL_OVERRIDE[SYMBOL] = _ORIG_SL
    elif SYMBOL in SL_OVERRIDE:
        del SL_OVERRIDE[SYMBOL]
    if _ORIG_DIR is not None:
        DIR_BIAS[SYMBOL] = _ORIG_DIR
    elif SYMBOL in DIR_BIAS:
        del DIR_BIAS[SYMBOL]
    config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = copy.deepcopy(_ORIG_Q_CFG)


# Preload meta model (ML parity with live; DJ30.r not in ML_BYPASS_SYMBOLS)
META = None
try:
    from models.signal_model import SignalModel
    _m = SignalModel(); _m.load(SYMBOL)
    if _m.has_model(SYMBOL):
        META = _m
        print(f"  [ML] loaded {SYMBOL} meta-model", flush=True)
    else:
        print(f"  [ML] no meta-model for {SYMBOL}", flush=True)
except Exception as e:
    print(f"  [ML] meta-model load failed: {e}", flush=True)
    META = None


def make_window_loader(start_days, end_days):
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
    p = {**DEFAULT_PARAMS, "audit_fix_gates": True}
    p["sl_atr_mult"] = float(WINNER_SL_ATR_MULT)
    p["pullback_atr_retrace"] = float(WINNER_PULLBACK_RETRACE)
    p["pullback_atr_wait"] = int(WINNER_PULLBACK_WAIT)   # advisory; v5 may not honor
    p["force_direction"] = WINNER_DIR_BIAS               # hard override
    p["min_quality"] = dict(WINNER_SIGNAL_QUALITY)
    if META is not None:
        p["_meta_model"] = META

    orig_load = v5_backtest.load_data
    v5_backtest.load_data = make_window_loader(fold["start_days"], fold["end_days"])
    try:
        r = v5_backtest.backtest_symbol(SYMBOL, days=None, params=p, verbose=False)
    except Exception as e:
        r = {"error": str(e)}
    finally:
        v5_backtest.load_data = orig_load

    if not r or "error" in r:
        return {
            "fold": fold["name"],
            "trades": 0, "pf": 0.0, "pnl": 0.0, "wr": 0.0, "dd": 0.0,
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
    print(f"=== STRICT 5-FOLD WF {SYMBOL} ({STRATEGY}, 5x90d non-overlapping -450..0) ===")
    print(f"Winner: SL={WINNER_SL_ATR_MULT}  PB_RETRACE={WINNER_PULLBACK_RETRACE}  "
          f"PB_WAIT={WINNER_PULLBACK_WAIT}  DIR={WINNER_DIR_BIAS}  Q={WINNER_SIGNAL_QUALITY}")
    print(f"Gate (STRICT): PF>={PF_THRESHOLD} in >={PF_OK_MIN}/5  AND  PnL>0 in >={PNL_OK_MIN}/5")

    # Sanity log on cache span
    meta = ALL_SYMBOLS[SYMBOL]
    cache_path = CACHE / meta["cache"]
    df_all = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df_all["time"]):
        df_all["time"] = pd.to_datetime(df_all["time"], unit="s", utc=True)
    span_days = (df_all["time"].max() - df_all["time"].min()).days
    print(f"  Cache: {cache_path.name}  rows={len(df_all)}  span={span_days}d  "
          f"(latest={df_all['time'].max().date()})")

    apply_winner_config()
    try:
        fold_results = []
        for i, fold in enumerate(FOLDS, 1):
            print(f"\n[Fold {i}/5] {fold['name']} ...", flush=True)
            r = run_fold(fold)
            fold_results.append(r)
            err = r.get("error", "")
            print(f"  trades={r['trades']} pf={r['pf']} pnl=${r['pnl']} "
                  f"wr={r.get('wr',0)}% dd={r.get('dd',0)}"
                  + (f"  ERROR={err}" if err else ""), flush=True)
    finally:
        restore_originals()

    # Gate eval
    folds_pf_ok = sum(1 for r in fold_results if r["pf"] >= PF_THRESHOLD)
    folds_pnl_ok = sum(1 for r in fold_results if r["pnl"] > 0)

    pf_pass = folds_pf_ok >= PF_OK_MIN
    pnl_pass = folds_pnl_ok >= PNL_OK_MIN
    decision = "SHIP" if (pf_pass and pnl_pass) else "NULL"

    fail_reasons = []
    if not pf_pass:
        fail_reasons.append(f"folds_pf_ok={folds_pf_ok}/5 (<{PF_OK_MIN})")
    if not pnl_pass:
        fail_reasons.append(f"folds_pnl_ok={folds_pnl_ok}/5 (<{PNL_OK_MIN})")
    reason = (
        f"folds_pf_ok={folds_pf_ok}/5, folds_pnl_ok={folds_pnl_ok}/5, "
        + ("PASS" if decision == "SHIP" else "FAIL: " + "; ".join(fail_reasons))
    )

    per_fold_lines = "; ".join(
        f"{r['fold']}: pnl=${r['pnl']} pf={r['pf']} trades={r['trades']}"
        for r in fold_results
    )

    out = {
        "symbol": SYMBOL,
        "strategy": STRATEGY,
        "winner_config": {
            "SL_ATR_MULT": WINNER_SL_ATR_MULT,
            "PULLBACK_RETRACE": WINNER_PULLBACK_RETRACE,
            "PULLBACK_WAIT": WINNER_PULLBACK_WAIT,
            "SIGNAL_QUALITY": WINNER_SIGNAL_QUALITY,
            "DIRECTION_BIAS": WINNER_DIR_BIAS,
        },
        "folds": fold_results,
        "folds_run": len(fold_results),
        "folds_pf_ok": folds_pf_ok,
        "folds_pnl_ok": folds_pnl_ok,
        "decision": decision,
        "decision_reason": reason,
        "per_fold_summary": per_fold_lines,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {OUT}")
    print(f"DECISION: {decision}")
    print(f"REASON: {reason}")
    print(f"PER-FOLD: {per_fold_lines}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
