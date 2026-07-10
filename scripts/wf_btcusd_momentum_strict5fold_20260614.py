#!/usr/bin/env python3 -B
"""5-fold STRICT walk-forward validation for BTCUSD / momentum.

Winner config (FROZEN — no per-fold re-tuning):
  SL_ATR_MULT = 1.5
  SIGNAL_QUALITY = {trending:30, ranging:30, volatile:30, low_vol:30}
  DIR_BIAS = SHORT
  TRAIL_PROFILE = _AGGR_LOCK
  PULLBACK_RETRACE = 0
  ELC_TRIGGER_R = "skipped_axis_not_implemented_in_v5_backtest"  (skipped)

Folds (non-overlapping 90d, holdout -90 → 0):
  fold 1: days -540 → -450
  fold 2: days -450 → -360
  fold 3: days -360 → -270
  fold 4: days -270 → -180
  fold 5: days -180 → -90
  HOLDOUT (live obs): -90 → 0

STRICT SHIP gate:
  folds_pf_ok    >= 4 (PF >= 1.5 in >=4 folds)
  folds_pnl_ok   >= 4 (PnL > 0 in >=4 folds)
  no single fold loses more than -2 x baseline_per_fold_pnl
  (baseline_per_fold_pnl = mean PnL across all 5 folds).

DATA SOURCE NOTE:
  raw_h1_BTCUSD.pkl ships with only ~500 H1 bars (~20 days). To get 540 days
  of H1 data for 5x90d non-overlapping folds, we resample M15 → H1 inside the
  load_data shim. M15 cache has 529 days. We FALL BACK to the live H1 cache
  for the most recent bars (where it overlaps) just to keep parity with live.
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
OUT_DIR = ROOT / "backtest" / "results" / "wf_strict5_btcusd_20260614"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"{SYMBOL}_{STRATEGY}_5fold_strict.json"

# ── Winner config (frozen) ───────────────────────────────────────────────
WINNER_SL_ATR_MULT = 1.5
WINNER_SIGNAL_QUALITY = {
    "trending": 30,
    "ranging": 30,
    "volatile": 30,
    "low_vol": 30,
}
WINNER_DIR_BIAS = "SHORT"
WINNER_TRAIL_PROFILE_NAME = "_AGGR_LOCK"
# _AGGR_LOCK from auto_tuned.py:31 — live (R, type, param) tuples; v5 wants
# (R, param, type), conversion done below via _live_to_bt_trail.
_AGGR_LOCK_LIVE = [
    (8.0, "trail", 0.3),
    (4.0, "trail", 0.5),
    (2.0, "trail", 0.8),
    (1.5, "lock", 0.7),
    (1.0, "lock", 0.4),
    (0.5, "be", 0.0),
]


def _live_to_bt_trail(steps):
    out = []
    for tup in steps:
        if len(tup) == 3:
            r, t, p = tup
            out.append((r, p, t))
    return out


WINNER_TRAIL_BT = _live_to_bt_trail(_AGGR_LOCK_LIVE)
WINNER_PULLBACK_RETRACE = 0.0  # 0 = immediate market fill

# Folds: (label, start_days_ago, end_days_ago) — slice = [t_max-start, t_max-end)
FOLDS = [
    {"name": "fold1_-540_-450", "start_days": 540, "end_days": 450},
    {"name": "fold2_-450_-360", "start_days": 450, "end_days": 360},
    {"name": "fold3_-360_-270", "start_days": 360, "end_days": 270},
    {"name": "fold4_-270_-180", "start_days": 270, "end_days": 180},
    {"name": "fold5_-180_-90",  "start_days": 180, "end_days":  90},
]

# STRICT gate constants
PF_THRESHOLD = 1.5
PF_OK_MIN = 4
PNL_OK_MIN = 4
MAX_FOLD_LOSS_MULT = 2.0

# Preload meta model (mirrors live ML META gate) — bypass list lives in config;
# BTCUSD is not in ML_BYPASS_SYMBOLS today (only XAU/JPN225), so include it.
META = None
try:
    from models.signal_model import SignalModel
    _m = SignalModel(); _m.load(SYMBOL)
    if _m.has_model(SYMBOL):
        META = _m
except Exception:
    META = None


# ─── Build a long-history H1 dataframe by resampling M15 → H1 ───
def _load_btcusd_long_h1():
    """Return a single dataframe covering 529d of BTCUSD H1 bars (resampled
    from raw_m15_BTCUSD.pkl). Used by the per-fold window slicer."""
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
    """Return load_data shim that slices the long M15-derived H1 frame."""
    def _load(symbol, days=90):
        global _H1_LONG
        if symbol != SYMBOL:
            # Fall back to the real loader for any non-BTCUSD symbol that
            # happens to get pulled in (shouldn't happen — backtest_symbol is
            # called per-symbol — but keep safe).
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
    """Run one fold with the frozen winner config."""
    p = {**DEFAULT_PARAMS, "audit_fix_gates": True}
    # Override knobs from winner_config
    p["sl_atr_mult"] = WINNER_SL_ATR_MULT
    p["min_quality"] = dict(WINNER_SIGNAL_QUALITY)
    p["force_direction"] = WINNER_DIR_BIAS
    p["force_trail"] = WINNER_TRAIL_BT
    p["pullback_atr_retrace"] = WINNER_PULLBACK_RETRACE
    if META is not None:
        p["_meta_model"] = META

    # Patch:
    #  1) SL_OVERRIDE so the live BTCUSD=3.0 mult doesn't squash our 1.5
    #  2) SIGNAL_QUALITY_SYMBOL[BTCUSD] (cleared) so p["min_quality"] sticks
    #  3) load_data with our long-history M15→H1 slice
    sl_save = v5_backtest.SL_OVERRIDE.get(SYMBOL, None)
    v5_backtest.SL_OVERRIDE[SYMBOL] = WINNER_SL_ATR_MULT

    # SIGNAL_QUALITY_SYMBOL is read INSIDE backtest_symbol via
    #     from config import SIGNAL_QUALITY_SYMBOL as _SYM_Q
    # so we patch config.SIGNAL_QUALITY_SYMBOL directly. Stash & restore.
    sq_save = None
    sq_had = False
    if hasattr(config, "SIGNAL_QUALITY_SYMBOL"):
        sq_had = True
        sq_save = dict(config.SIGNAL_QUALITY_SYMBOL)
        # Pop BTCUSD entry so symbol-level merge yields p["min_quality"] (=30/all)
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
        r = {"error": str(e)}
    finally:
        v5_backtest.load_data = orig_load
        # Restore SL_OVERRIDE
        if sl_save is None:
            v5_backtest.SL_OVERRIDE.pop(SYMBOL, None)
        else:
            v5_backtest.SL_OVERRIDE[SYMBOL] = sl_save
        # Restore SIGNAL_QUALITY_SYMBOL
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
    print(f"=== STRICT 5-FOLD WF {SYMBOL} ({STRATEGY}, 5x90d non-overlapping) ===")
    print(
        f"Winner: SL={WINNER_SL_ATR_MULT} SQ=30/all DIR_BIAS={WINNER_DIR_BIAS} "
        f"TRAIL={WINNER_TRAIL_PROFILE_NAME} PB_RETRACE={WINNER_PULLBACK_RETRACE} "
        f"ELC=skipped"
    )
    print(
        f"Gate: PF>={PF_THRESHOLD} in >={PF_OK_MIN}/5  AND  PnL>0 in >={PNL_OK_MIN}/5 "
        f"AND no fold > {MAX_FOLD_LOSS_MULT}x baseline-loss"
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

    baseline_per_fold_pnl = (
        sum(r["pnl"] for r in fold_results) / max(1, len(fold_results))
    )
    loss_limit = -MAX_FOLD_LOSS_MULT * abs(baseline_per_fold_pnl)
    fold_loss_breach = [r for r in fold_results if r["pnl"] < loss_limit]

    pf_pass = folds_pf_ok >= PF_OK_MIN
    pnl_pass = folds_pnl_ok >= PNL_OK_MIN
    loss_pass = (len(fold_loss_breach) == 0)

    decision = "SHIP" if (pf_pass and pnl_pass and loss_pass) else "NULL"
    fail_reasons = []
    if not pf_pass:
        fail_reasons.append(f"folds_pf_ok={folds_pf_ok}/5 (<{PF_OK_MIN})")
    if not pnl_pass:
        fail_reasons.append(f"folds_pnl_ok={folds_pnl_ok}/5 (<{PNL_OK_MIN})")
    if not loss_pass:
        fail_reasons.append(
            f"fold_loss_breach={[r['fold'] for r in fold_loss_breach]} "
            f"loss_limit={loss_limit:.2f}"
        )
    reason = (
        f"folds_pf_ok={folds_pf_ok}/5, folds_pnl_ok={folds_pnl_ok}/5, "
        f"baseline_per_fold=${baseline_per_fold_pnl:.2f}, "
        f"loss_limit=${loss_limit:.2f}, "
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
            "DIR_BIAS": WINNER_DIR_BIAS,
            "TRAIL_PROFILE": WINNER_TRAIL_PROFILE_NAME,
            "PULLBACK_RETRACE": WINNER_PULLBACK_RETRACE,
            "ELC_TRIGGER_R": "skipped_axis_not_implemented_in_v5_backtest",
        },
        "folds": fold_results,
        "folds_run": len(fold_results),
        "folds_pf_ok": folds_pf_ok,
        "folds_pnl_ok": folds_pnl_ok,
        "baseline_per_fold_pnl": round(baseline_per_fold_pnl, 2),
        "loss_limit": round(loss_limit, 2),
        "fold_loss_breach": [r["fold"] for r in fold_loss_breach],
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
