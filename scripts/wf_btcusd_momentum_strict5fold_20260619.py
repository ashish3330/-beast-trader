#!/usr/bin/env python3 -B
"""5-fold STRICT walk-forward validation for BTCUSD / momentum (2026-06-19).

Winner config (FROZEN — no per-fold re-tuning):
  SL_ATR_MULT = 3
  SIGNAL_QUALITY = {trending:30, ranging:42, volatile:35, low_vol:30}
  DIRECTION_BIAS = NO_FORCE          (== BOTH; no bias enforced)
  PULLBACK_RETRACE = 0               (immediate market fill)
  PULLBACK_WAIT = 1                  (BT pullback path runs on next bar only — already 1-bar)
  EXTRA_TOXIC_HOURS = [8, 12, 21]    (added to TOXIC_HOURS_PER_SYMBOL[BTCUSD])
  ZONE_GATE = OFF                    (no FIB / discount-premium zone filter)

Folds (non-overlapping 90d, holdout -90 → 0):
  fold 1: days -450 → -360
  fold 2: days -360 → -270
  fold 3: days -270 → -180
  fold 4: days -180 → -90
  fold 5: days -90  → 0   (NOTE: holdout used as 5th fold to cover -450 → 0)

STRICT SHIP gate:
  folds_pf_ok    >= 4 (PF >= 1.5 in >=4 folds)
  folds_pnl_ok   >= 4 (PnL > 0 in >=4 folds)

DATA SOURCE NOTE (parity with strict5fold 20260614):
  raw_h1_BTCUSD.pkl ships with only ~500 H1 bars (~20 days). To get 450+ days
  of H1 data we resample M15 → H1 inside the load_data shim.
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
from backtest.v5_backtest import (  # noqa: E402
    DEFAULT_PARAMS,
    ALL_SYMBOLS,
    CACHE,
)

SYMBOL = "BTCUSD"
STRATEGY = "momentum"
OUT_DIR = ROOT / "backtest" / "results" / "wf_strict5_btcusd_20260619"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"{SYMBOL}_{STRATEGY}_5fold_strict.json"

# ── Winner config (frozen) ───────────────────────────────────────────────
WINNER_SL_ATR_MULT = 3.0
WINNER_SIGNAL_QUALITY = {
    "trending": 30,
    "ranging": 42,
    "volatile": 35,
    "low_vol": 30,
}
WINNER_DIR_BIAS = "NO_FORCE"           # both directions allowed
WINNER_PULLBACK_RETRACE = 0.0          # immediate market fill
WINNER_PULLBACK_WAIT = 1               # already 1 bar in v5_backtest
WINNER_EXTRA_TOXIC_HOURS = {8, 12, 21}
WINNER_ZONE_GATE_OFF = True            # fib zone filter OFF

# Folds: 5 x 90d, covering -450d → 0 (using holdout as 5th fold per user spec)
FOLDS = [
    {"name": "fold1_-450_-360", "start_days": 450, "end_days": 360},
    {"name": "fold2_-360_-270", "start_days": 360, "end_days": 270},
    {"name": "fold3_-270_-180", "start_days": 270, "end_days": 180},
    {"name": "fold4_-180_-90",  "start_days": 180, "end_days":  90},
    {"name": "fold5_-90_-0",    "start_days":  90, "end_days":   0},
]

# STRICT gate constants
PF_THRESHOLD = 1.5
PF_OK_MIN = 4
PNL_OK_MIN = 4

# Preload meta model (mirrors live ML META gate)
META = None
try:
    from models.signal_model import SignalModel
    _m = SignalModel()
    _m.load(SYMBOL)
    if _m.has_model(SYMBOL):
        META = _m
except Exception:
    META = None


# ─── Build a long-history H1 dataframe by resampling M15 → H1 ───
def _load_btcusd_long_h1():
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
        hi = t_max - pd.Timedelta(days=end_days) if end_days > 0 else t_max
        df = df[(df["time"] >= lo) & (df["time"] < hi)].reset_index(drop=True)
        return df
    return _load


def run_fold(fold):
    p = {**DEFAULT_PARAMS, "audit_fix_gates": True}
    p["sl_atr_mult"] = WINNER_SL_ATR_MULT
    p["min_quality"] = dict(WINNER_SIGNAL_QUALITY)
    # NO_FORCE => let DIR_BIAS pass-through (no override). Don't set force_direction.
    p["pullback_atr_retrace"] = WINNER_PULLBACK_RETRACE
    if WINNER_ZONE_GATE_OFF:
        p["fib_filter"] = False
        p["use_fib_zone"] = False
    if META is not None:
        p["_meta_model"] = META

    # Patch SL_OVERRIDE so live BTCUSD=3.0 doesn't squash our value
    sl_save = v5_backtest.SL_OVERRIDE.get(SYMBOL, None)
    v5_backtest.SL_OVERRIDE[SYMBOL] = WINNER_SL_ATR_MULT

    # Clear per-symbol min-Q override so winner SQ dict sticks
    sq_save = None
    sq_had = False
    if hasattr(config, "SIGNAL_QUALITY_SYMBOL"):
        sq_had = True
        sq_save = dict(config.SIGNAL_QUALITY_SYMBOL)
        config.SIGNAL_QUALITY_SYMBOL.pop(SYMBOL, None)

    # Patch TOXIC_HOURS_PER_SYMBOL[BTCUSD] with extra toxic hours (UNION semantics)
    tox_save = v5_backtest.TOXIC_HOURS_PER_SYMBOL.get(SYMBOL, set())
    tox_save_copy = set(tox_save) if tox_save else set()
    merged = set(tox_save_copy) | set(WINNER_EXTRA_TOXIC_HOURS)
    v5_backtest.TOXIC_HOURS_PER_SYMBOL[SYMBOL] = merged

    # Patch DIR_BIAS — NO_FORCE means clear any per-sym bias
    dir_save = v5_backtest.DIR_BIAS.get(SYMBOL, 0)
    v5_backtest.DIR_BIAS[SYMBOL] = 0

    orig_load = v5_backtest.load_data
    v5_backtest.load_data = make_window_loader(
        fold["start_days"], fold["end_days"]
    )

    try:
        r = v5_backtest.backtest_symbol(
            SYMBOL, days=None, params=p, verbose=False
        )
    except Exception as e:
        r = {"error": str(e)}
    finally:
        v5_backtest.load_data = orig_load
        if sl_save is None:
            v5_backtest.SL_OVERRIDE.pop(SYMBOL, None)
        else:
            v5_backtest.SL_OVERRIDE[SYMBOL] = sl_save
        if sq_had:
            config.SIGNAL_QUALITY_SYMBOL.clear()
            config.SIGNAL_QUALITY_SYMBOL.update(sq_save)
        if tox_save_copy:
            v5_backtest.TOXIC_HOURS_PER_SYMBOL[SYMBOL] = tox_save_copy
        else:
            v5_backtest.TOXIC_HOURS_PER_SYMBOL.pop(SYMBOL, None)
        v5_backtest.DIR_BIAS[SYMBOL] = dir_save

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
    print(f"=== STRICT 5-FOLD WF {SYMBOL} ({STRATEGY}, 5x90d non-overlapping) ===")
    print(
        f"Winner: SL={WINNER_SL_ATR_MULT} "
        f"SQ={WINNER_SIGNAL_QUALITY} "
        f"DIR_BIAS={WINNER_DIR_BIAS} "
        f"PB_RETRACE={WINNER_PULLBACK_RETRACE} "
        f"PB_WAIT={WINNER_PULLBACK_WAIT} "
        f"EXTRA_TOXIC={sorted(WINNER_EXTRA_TOXIC_HOURS)} "
        f"ZONE_GATE=OFF"
    )
    print(
        f"Gate: PF>={PF_THRESHOLD} in >={PF_OK_MIN}/5 "
        f"AND PnL>0 in >={PNL_OK_MIN}/5"
    )

    fold_results = []
    for i, fold in enumerate(FOLDS, 1):
        print(f"\n[Fold {i}/5] {fold['name']} ...")
        r = run_fold(fold)
        fold_results.append(r)
        err = r.get("error", "")
        print(
            f"  trades={r['trades']} pf={r['pf']} pnl=${r['pnl']} "
            f"wr={r.get('wr', 0)}% dd={r.get('dd', 0)}"
            + (f"  ERROR={err}" if err else "")
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
            "SIGNAL_QUALITY": WINNER_SIGNAL_QUALITY,
            "DIRECTION_BIAS": WINNER_DIR_BIAS,
            "PULLBACK_RETRACE": WINNER_PULLBACK_RETRACE,
            "PULLBACK_WAIT": WINNER_PULLBACK_WAIT,
            "EXTRA_TOXIC_HOURS": sorted(WINNER_EXTRA_TOXIC_HOURS),
            "ZONE_GATE": "OFF",
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
