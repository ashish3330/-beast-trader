#!/usr/bin/env python3 -B
"""5-fold STRICT walk-forward validation for BTCUSD with FROZEN winner config.

Winner config (FROZEN — no per-fold re-tuning):
  SL_ATR_MULT       = 1.25
  PULLBACK_RETRACE  = 0.1   (ATR multiplier for pullback check)
  PULLBACK_WAIT     = 1     (bars to wait — already default in v5_backtest)
  SIGNAL_QUALITY    = {trending:60, ranging:75, volatile:55, low_vol:60}
  DIRECTION_BIAS    = BOTH

Folds (non-overlapping 90d):
  fold 1: days -450 → -360
  fold 2: days -360 → -270
  fold 3: days -270 → -180
  fold 4: days -180 → -90
  fold 5: days  -90 →   0

STRICT SHIP gate:
  folds_pf_ok  >= 4 (PF >= 1.5 in >=4 folds)
  AND folds_pnl_ok >= 4 (PnL > 0 in >=4 folds)
  NULL otherwise.

Pullback retrace is hardcoded to 0.2*ATR in v5_backtest. We patch the module
text-replacement once at import time so retrace = WINNER_PULLBACK_RETRACE
(0.1 * ATR) without touching production code on disk.
"""
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
    DEFAULT_PARAMS,
    ALL_SYMBOLS,
    CACHE,
)

SYMBOL = "BTCUSD"
STRATEGY = "momentum"
OUT_DIR = ROOT / "backtest" / "results" / "wf_strict5_btcusd_20260617"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"{SYMBOL}_{STRATEGY}_5fold_strict_winner_20260617.json"

# ── Winner config (frozen) ───────────────────────────────────────────────
WINNER_SL_ATR_MULT = 1.25
WINNER_PULLBACK_RETRACE = 0.1   # ATR multiplier
WINNER_PULLBACK_WAIT = 1        # bars (already default)
WINNER_SIGNAL_QUALITY = {
    "trending": 60,
    "ranging": 75,
    "volatile": 55,
    "low_vol": 60,
}
WINNER_DIR_BIAS = "BOTH"

# Folds — 5 non-overlapping 90d windows covering -450 → 0
FOLDS = [
    {"name": "fold1_-450_-360", "start_days": 450, "end_days": 360},
    {"name": "fold2_-360_-270", "start_days": 360, "end_days": 270},
    {"name": "fold3_-270_-180", "start_days": 270, "end_days": 180},
    {"name": "fold4_-180_-90",  "start_days": 180, "end_days":  90},
    {"name": "fold5_-90_0",     "start_days":  90, "end_days":   0},
]

# STRICT gate constants
PF_THRESHOLD = 1.5
PF_OK_MIN = 4
PNL_OK_MIN = 4

# ─── Monkey-patch pullback retrace at module level ──────────────────────
# v5_backtest line ~827 hardcodes `retrace = atr * 0.2`. We can't change the
# source on disk, but we CAN replace the runtime function. Easier: patch the
# constant inline by string-substituting + exec'ing the affected function.
# Simpler still: just write a small env-var hook the script reads.
# Cleanest approach: re-read the source, replace `atr * 0.2` with
# `atr * <WINNER>`, exec into v5_backtest namespace, replacing the
# backtest_symbol function.
import re  # noqa: E402

_src = (ROOT / "backtest" / "v5_backtest.py").read_text()
_pat = re.compile(r"retrace = atr \* 0\.2")
_n = len(_pat.findall(_src))
assert _n == 1, f"expected 1 'retrace = atr * 0.2' occurrence, found {_n}"
_new_src = _pat.sub(f"retrace = atr * {WINNER_PULLBACK_RETRACE}", _src)
# Exec replaced source into the v5_backtest module namespace so backtest_symbol
# picks up the new retrace constant.
exec(compile(_new_src, str(ROOT / "backtest" / "v5_backtest.py"), "exec"),
     v5_backtest.__dict__)
# After exec, re-import refs we use:
from backtest.v5_backtest import DEFAULT_PARAMS, ALL_SYMBOLS, CACHE  # noqa: E402,F811

# Preload meta model
META = None
try:
    from models.signal_model import SignalModel
    _m = SignalModel(); _m.load(SYMBOL)
    if _m.has_model(SYMBOL):
        META = _m
except Exception:
    META = None


def _load_btcusd_long_h1():
    """Resample M15 → H1 for max coverage (~529 days)."""
    m15_path = CACHE / "raw_m15_BTCUSD.pkl"
    if not m15_path.exists():
        raise RuntimeError(f"M15 cache not found at {m15_path}")
    m15 = pickle.load(open(m15_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(m15["time"]):
        m15["time"] = pd.to_datetime(m15["time"], unit="s", utc=True)
    m15 = m15.set_index("time").sort_index()
    agg = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "tick_volume": "sum",
        "spread": "max",
        "real_volume": "sum",
    }
    h1 = m15.resample("1h", label="left", closed="left").agg(agg).dropna(
        subset=["open", "high", "low", "close"]
    )
    h1 = h1.reset_index()
    return h1


_H1_LONG = None


def make_window_loader(start_days, end_days):
    def _load(symbol, days=90):
        global _H1_LONG
        if symbol != SYMBOL:
            meta = ALL_SYMBOLS[symbol]
            path = CACHE / meta["cache"]
            if not path.exists():
                return None
            df = pickle.load(open(path, "rb"))
            if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            return df
        if _H1_LONG is None:
            _H1_LONG = _load_btcusd_long_h1()
        df = _H1_LONG
        t_max = df["time"].max()
        lo = t_max - pd.Timedelta(days=start_days)
        hi = t_max - pd.Timedelta(days=end_days)
        df = df[(df["time"] >= lo) & (df["time"] < hi)].reset_index(drop=True)
        return df
    return _load


def run_fold(fold):
    p = {**DEFAULT_PARAMS, "audit_fix_gates": True}
    p["sl_atr_mult"] = WINNER_SL_ATR_MULT
    p["min_quality"] = dict(WINNER_SIGNAL_QUALITY)
    p["force_direction"] = WINNER_DIR_BIAS  # BOTH → allows long + short
    if META is not None:
        p["_meta_model"] = META

    # Override SL_OVERRIDE for BTCUSD to our winner mult
    sl_save = v5_backtest.SL_OVERRIDE.get(SYMBOL, None)
    v5_backtest.SL_OVERRIDE[SYMBOL] = WINNER_SL_ATR_MULT

    # Clear SIGNAL_QUALITY_SYMBOL[BTCUSD] so p["min_quality"] sticks
    sq_save = None
    sq_had = False
    if hasattr(config, "SIGNAL_QUALITY_SYMBOL"):
        sq_had = True
        sq_save = dict(config.SIGNAL_QUALITY_SYMBOL)
        config.SIGNAL_QUALITY_SYMBOL.pop(SYMBOL, None)

    orig_load = v5_backtest.load_data
    v5_backtest.load_data = make_window_loader(
        fold["start_days"], fold["end_days"]
    )

    try:
        r = v5_backtest.backtest_symbol(
            SYMBOL, days=None, params=p, verbose=False
        )
    except Exception as e:
        import traceback
        r = {"error": f"{e}\n{traceback.format_exc()}"}
    finally:
        v5_backtest.load_data = orig_load
        if sl_save is None:
            v5_backtest.SL_OVERRIDE.pop(SYMBOL, None)
        else:
            v5_backtest.SL_OVERRIDE[SYMBOL] = sl_save
        if sq_had:
            config.SIGNAL_QUALITY_SYMBOL.clear()
            config.SIGNAL_QUALITY_SYMBOL.update(sq_save)

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
    print(f"=== STRICT 5-FOLD WF {SYMBOL} ({STRATEGY}) — winner_20260617 ===")
    print(
        f"Winner: SL={WINNER_SL_ATR_MULT} PB_RETRACE={WINNER_PULLBACK_RETRACE} "
        f"PB_WAIT={WINNER_PULLBACK_WAIT} SQ={WINNER_SIGNAL_QUALITY} "
        f"DIR={WINNER_DIR_BIAS}"
    )
    print(
        f"Gate: PF>={PF_THRESHOLD} in >={PF_OK_MIN}/5 AND "
        f"PnL>0 in >={PNL_OK_MIN}/5"
    )

    # Check cache span first
    h1 = _load_btcusd_long_h1()
    span_days = (h1["time"].max() - h1["time"].min()).days
    print(f"\nBTCUSD H1 cache span: {span_days} days ({len(h1)} bars)")
    print(f"  range: {h1['time'].min()} -> {h1['time'].max()}")

    fold_results = []
    for i, fold in enumerate(FOLDS, 1):
        print(f"\n[Fold {i}/5] {fold['name']} ...")
        r = run_fold(fold)
        fold_results.append(r)
        err = r.get("error", "")
        print(
            f"  trades={r['trades']} pf={r['pf']} pnl=${r['pnl']} "
            f"wr={r.get('wr', 0)}% dd={r.get('dd', 0)}"
            + (f"  ERROR={err[:120]}" if err else "")
        )

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
