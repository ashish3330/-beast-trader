#!/usr/bin/env python3 -B
"""5-fold STRICT walk-forward validation for DJ30.r momentum winner_config.

Frozen winner_config (from backtest/results/deepest_sweep_momentum/DJ30.r.json):
    SL_ATR_MULT      = 2.5
    SIGNAL_QUALITY   = 40    (applied symmetric across all 4 regimes)
    DIR_BIAS         = None  (no override — natural DIR_BIAS_REGIME_AUTO['DJ30.r']
                              cell 'ranging': 'LONG' still applies; this matches
                              the deepest_sweep configuration that produced the
                              winner row)
    TRAIL_PROFILE    = "_TIGHT_LOCK"
    PULLBACK_RETRACE = 0.4
    ELC_TRIGGER_R    = None  (BT does NOT simulate ELC — inert)

DJ30.r uses raw_h1_DJ30_r.pkl (5249-day span), so 5x180d = 900d window fits.
Folds (non-overlapping, 180d each, day offsets from end-of-data):
    fold 1: -900 → -720
    fold 2: -720 → -540
    fold 3: -540 → -360
    fold 4: -360 → -180
    fold 5: -180 →    0
(Holdout -90→0 overlaps the tail of fold 5 — the user asked for the 3yr-cache
larger split which uses the full 5x180d. The 90d holdout is reserved by live.)

SHIP gate (STRICT):
    folds_pf_ok   >= 4    (PF >= 1.5 in 4 of 5 folds)
    folds_pnl_ok  >= 4    (PnL > 0 in 4 of 5 folds)
    no single fold loses more than -2 × baseline_per_fold_pnl
NULL otherwise.

Mechanism mirrors scripts/wf_xauusd_winner_20260614.py:
  - monkey-patch backtest.v5_backtest.load_data to slice cache to fold window
  - patch config.SIGNAL_QUALITY_SYMBOL['DJ30.r'] uniform across regimes
  - patch bt.SL_OVERRIDE['DJ30.r'] = 2.5
  - patch bt.TRAIL_OVERRIDE['DJ30.r'] = _TIGHT_LOCK (BT-format tuples)
  - pass pullback_atr_retrace=0.4 in params
  - leave DIR_BIAS_REGIME_AUTO intact (DIR_BIAS=None means "no extra override")
  - preload ML meta-model (parity with live + deepest sweep)
"""
import sys
import json
import time
import copy
import pickle
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import auto_tuned  # noqa: E402
from backtest import v5_backtest as bt  # noqa: E402
from backtest.v5_backtest import ALL_SYMBOLS, CACHE, backtest_symbol  # noqa: E402

SYMBOL = "DJ30.r"
STRATEGY = "momentum"

# ---- frozen winner_config ----
WINNER_SL_ATR_MULT = 2.5
WINNER_SIGNAL_QUALITY = 40
WINNER_DIR_BIAS = None  # null
WINNER_TRAIL_PROFILE = "_TIGHT_LOCK"
WINNER_PULLBACK_RETRACE = 0.4
WINNER_ELC_TRIGGER_R = None  # null; BT does not sim ELC anyway

# Baseline (from deepest_sweep_momentum/DJ30.r.json winner row on 1095d).
# Used to derive a per-fold expected PnL for the STRICT max-fold-loss gate.
WINNER_WINDOW_DAYS = 1095
WINNER_WINDOW_PNL = 5785.01      # 1095d winner PnL
FOLD_DAYS = 180
BASELINE_PER_FOLD_PNL = WINNER_WINDOW_PNL * (FOLD_DAYS / WINNER_WINDOW_DAYS)  # ~951
MAX_FOLD_LOSS = -2.0 * BASELINE_PER_FOLD_PNL  # ~-1902 (any fold below this fails STRICT)

# Folds: non-overlapping 180d each, day offsets from end-of-data
FOLDS = [
    {"idx": 1, "start_off": 900, "end_off": 720},
    {"idx": 2, "start_off": 720, "end_off": 540},
    {"idx": 3, "start_off": 540, "end_off": 360},
    {"idx": 4, "start_off": 360, "end_off": 180},
    {"idx": 5, "start_off": 180, "end_off": 0},
]

OUT_DIR = ROOT / "backtest" / "results" / "deepest_sweep_momentum"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / f"{SYMBOL}_wf_strict.json"


# -----------------------------------------------------------------------------
# Windowed load_data (slice cache to [t_max - start_off, t_max - end_off))
# -----------------------------------------------------------------------------
_ORIG_LOAD = bt.load_data


def make_window_loader(start_off, end_off):
    def _load(symbol, days=None):
        meta = ALL_SYMBOLS[symbol]
        path = CACHE / meta["cache"]
        if not path.exists():
            return None
        df = pickle.load(open(path, "rb"))
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        t_max = df["time"].max()
        lo = t_max - pd.Timedelta(days=start_off)
        hi = t_max - pd.Timedelta(days=end_off)
        return df[(df["time"] >= lo) & (df["time"] < hi)].reset_index(drop=True)
    return _load


# -----------------------------------------------------------------------------
# Trail profile lookup (live → BT tuple format)
# -----------------------------------------------------------------------------
TRAIL_PROFILES_LIVE = {
    "_TIGHT_LOCK":   auto_tuned._TIGHT_LOCK,
    "_WIDE_RUNNER":  auto_tuned._WIDE_RUNNER,
    "_RANGE_TIGHT":  auto_tuned._RANGE_TIGHT,
    "_TREND_LOOSE":  auto_tuned._TREND_LOOSE,
    "_AGGR_LOCK":    auto_tuned._AGGR_LOCK,
    "_RUNNER_NO_BE": auto_tuned._RUNNER_NO_BE,
    "_FOREX_LOOSE":  auto_tuned._FOREX_LOOSE,
}
TRAIL_PROFILES_BT = {
    name: bt._live_to_bt_trail(steps) for name, steps in TRAIL_PROFILES_LIVE.items()
}
WINNER_TRAIL_BT = TRAIL_PROFILES_BT[WINNER_TRAIL_PROFILE]


# -----------------------------------------------------------------------------
# Optional ML meta-model (parity with deepest_sweep + live)
# -----------------------------------------------------------------------------
META_MODEL = None
try:
    from models.signal_model import SignalModel
    _m = SignalModel()
    _m.load(SYMBOL)
    if _m.has_model(SYMBOL):
        META_MODEL = _m
except Exception as e:
    print(f"  [ML] load failed: {e}")
print(f"  [ML] meta-model loaded: {META_MODEL is not None}")


# -----------------------------------------------------------------------------
# Per-fold runner
# -----------------------------------------------------------------------------
def run_fold(fold):
    # Snapshot mutable state so we can restore cleanly between folds
    saved_load = bt.load_data
    saved_sq = copy.deepcopy(config.SIGNAL_QUALITY_SYMBOL.get(SYMBOL, {}))
    saved_sl = bt.SL_OVERRIDE.get(SYMBOL, "__MISSING__")
    saved_tr = bt.TRAIL_OVERRIDE.get(SYMBOL, "__MISSING__")

    try:
        # 1) windowed cache
        bt.load_data = make_window_loader(fold["start_off"], fold["end_off"])
        # 2) signal quality (symmetric across all 4 regimes)
        config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
            "trending": WINNER_SIGNAL_QUALITY,
            "ranging":  WINNER_SIGNAL_QUALITY,
            "volatile": WINNER_SIGNAL_QUALITY,
            "low_vol":  WINNER_SIGNAL_QUALITY,
        }
        # 3) SL ATR mult
        bt.SL_OVERRIDE[SYMBOL] = WINNER_SL_ATR_MULT
        # 4) trail profile
        bt.TRAIL_OVERRIDE[SYMBOL] = WINNER_TRAIL_BT
        # 5) params: pullback retrace + ML model (DIR_BIAS=None → no force_direction)
        params = {"pullback_atr_retrace": float(WINNER_PULLBACK_RETRACE)}
        if META_MODEL is not None:
            params["_meta_model"] = META_MODEL

        t0 = time.time()
        try:
            r = backtest_symbol(SYMBOL, days=None, params=params, verbose=False)
        except Exception as e:
            return {
                "fold": fold["idx"],
                "window": [fold["start_off"], fold["end_off"]],
                "error": str(e),
                "seconds": round(time.time() - t0, 1),
                "pnl": 0.0, "pf": 0.0, "trades": 0,
            }
        if r is None:
            return {
                "fold": fold["idx"],
                "window": [fold["start_off"], fold["end_off"]],
                "error": "no-result",
                "seconds": round(time.time() - t0, 1),
                "pnl": 0.0, "pf": 0.0, "trades": 0,
            }
        return {
            "fold": fold["idx"],
            "window": [fold["start_off"], fold["end_off"]],
            "pnl": round(float(r.get("pnl", 0.0)), 2),
            "pf": round(float(r.get("pf", 0.0) or 0.0), 3),
            "trades": int(r.get("trades", 0)),
            "wr": round(float(r.get("wr", 0.0) or 0.0), 1),
            "max_dd": round(float(r.get("dd", 0.0) or 0.0), 2),
            "seconds": round(time.time() - t0, 1),
        }
    finally:
        # restore
        bt.load_data = saved_load
        config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = saved_sq
        if saved_sl == "__MISSING__":
            bt.SL_OVERRIDE.pop(SYMBOL, None)
        else:
            bt.SL_OVERRIDE[SYMBOL] = saved_sl
        if saved_tr == "__MISSING__":
            bt.TRAIL_OVERRIDE.pop(SYMBOL, None)
        else:
            bt.TRAIL_OVERRIDE[SYMBOL] = saved_tr


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()

    # Sanity: cache range
    meta = ALL_SYMBOLS[SYMBOL]
    path = CACHE / meta["cache"]
    df_all = pickle.load(open(path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df_all["time"]):
        df_all["time"] = pd.to_datetime(df_all["time"], unit="s", utc=True)
    span_days = (df_all["time"].max() - df_all["time"].min()).days
    print(f"{SYMBOL} cache: {df_all['time'].min()} -> {df_all['time'].max()}  "
          f"({len(df_all)} bars, {span_days} days)")

    print(f"\n=== 5-fold STRICT WF — {STRATEGY} / {SYMBOL} ===")
    print(f"winner_config: SL={WINNER_SL_ATR_MULT} Q={WINNER_SIGNAL_QUALITY} "
          f"DIR={WINNER_DIR_BIAS} TRAIL={WINNER_TRAIL_PROFILE} "
          f"PB={WINNER_PULLBACK_RETRACE} ELC={WINNER_ELC_TRIGGER_R}")
    print(f"fold size: {FOLD_DAYS}d  total span: {FOLDS[0]['start_off']}d -> {FOLDS[-1]['end_off']}d")
    print(f"baseline per fold (winner $5785 / 1095d × 180d): ${BASELINE_PER_FOLD_PNL:.2f}")
    print(f"max single-fold loss allowed (-2×baseline): ${MAX_FOLD_LOSS:.2f}\n")

    results = []
    for f in FOLDS:
        print(f"  Fold {f['idx']}: [-{f['start_off']}d, -{f['end_off']}d) ...", flush=True)
        r = run_fold(f)
        results.append(r)
        if "error" in r and r["error"]:
            print(f"    ERROR: {r['error']}")
        else:
            print(f"    pnl=${r['pnl']:+.2f}  pf={r['pf']:.3f}  n={r['trades']}  "
                  f"wr={r.get('wr', 0):.1f}%  dd={r.get('max_dd', 0):.2f}%  ({r['seconds']}s)")

    # ---- decision gates ----
    valid = [r for r in results if "error" not in r]
    folds_run = len(valid)
    folds_pnl_ok = sum(1 for r in valid if r["pnl"] > 0.0)
    folds_pf_ok  = sum(1 for r in valid if r["pf"] >= 1.5)
    worst_loss = min((r["pnl"] for r in valid), default=0.0)
    max_loss_breach = worst_loss < MAX_FOLD_LOSS  # True = any fold lost more than -2x

    failed_gates = []
    if folds_pf_ok < 4:
        failed_gates.append(f"folds_pf_ok={folds_pf_ok} (<4 required @ PF>=1.5)")
    if folds_pnl_ok < 4:
        failed_gates.append(f"folds_pnl_ok={folds_pnl_ok} (<4 required @ PnL>0)")
    if max_loss_breach:
        failed_gates.append(
            f"worst_fold_pnl=${worst_loss:.2f} breaches max_loss=${MAX_FOLD_LOSS:.2f}"
        )

    decision = "SHIP" if not failed_gates else "NULL"
    if decision == "SHIP":
        reason = (f"STRICT PASSED — pf_ok={folds_pf_ok}/5 pnl_ok={folds_pnl_ok}/5 "
                  f"worst_loss=${worst_loss:.2f} (>{MAX_FOLD_LOSS:.2f}); "
                  + " ".join(f"F{r['fold']}: pnl=${r['pnl']:+.2f}/pf={r['pf']:.2f}/n={r['trades']}" for r in valid))
    else:
        reason = ("STRICT FAILED — " + "; ".join(failed_gates)
                  + " :: per-fold: "
                  + " ".join(f"F{r['fold']}: pnl=${r['pnl']:+.2f}/pf={r['pf']:.2f}/n={r['trades']}" for r in valid))

    payload = {
        "strategy": STRATEGY,
        "symbol": SYMBOL,
        "winner_config": {
            "SL_ATR_MULT":      WINNER_SL_ATR_MULT,
            "SIGNAL_QUALITY":   WINNER_SIGNAL_QUALITY,
            "DIR_BIAS":         WINNER_DIR_BIAS,
            "TRAIL_PROFILE":    WINNER_TRAIL_PROFILE,
            "PULLBACK_RETRACE": WINNER_PULLBACK_RETRACE,
            "ELC_TRIGGER_R":    WINNER_ELC_TRIGGER_R,
        },
        "fold_size_days": FOLD_DAYS,
        "baseline_per_fold_pnl": round(BASELINE_PER_FOLD_PNL, 2),
        "max_fold_loss_allowed": round(MAX_FOLD_LOSS, 2),
        "folds": results,
        "folds_run": folds_run,
        "folds_pf_ok": folds_pf_ok,
        "folds_pnl_ok": folds_pnl_ok,
        "worst_fold_pnl": round(worst_loss, 2),
        "decision": decision,
        "decision_reason": reason,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_FILE.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n=== DECISION: {decision} ===")
    print(f"  {reason}")
    print(f"\nWritten: {OUT_FILE}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    return payload


if __name__ == "__main__":
    main()
