#!/usr/bin/env python3 -B
"""5-fold STRICT walk-forward validation for JPN225ft / momentum.

Winner config (FROZEN — no per-fold re-tuning):
  SL_ATR_MULT = 1.5
  SIGNAL_QUALITY = NO_CHANGE (live config)
  DIR_BIAS = NO_CHANGE (live config)
  TRAIL_PROFILE = NO_CHANGE (live config)
  PULLBACK_RETRACE = 0.1
  ELC_TRIGGER_R = NO_CHANGE (live config)

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

Mechanism: monkey-patch backtest.v5_backtest.load_data to slice cache
to a [t_max - start_days, t_max - end_days) window per fold.
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
OUT_DIR = ROOT / "backtest" / "results" / "wf_strict5_20260614"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"{SYMBOL}_momentum_5fold_strict.json"

# Winner config (frozen)
WINNER_SL_ATR_MULT = 1.5
WINNER_PULLBACK_RETRACE = 0.1
# NO_CHANGE keys -> inherit live config / DEFAULT_PARAMS

# Folds: (label, start_days_ago, end_days_ago) — slice = [t_max-start, t_max-end)
FOLDS = [
    {"name": "fold1_-540_-450", "start_days": 540, "end_days": 450},
    {"name": "fold2_-450_-360", "start_days": 450, "end_days": 360},
    {"name": "fold3_-360_-270", "start_days": 360, "end_days": 270},
    {"name": "fold4_-270_-180", "start_days": 270, "end_days": 180},
    {"name": "fold5_-180_-90",  "start_days": 180, "end_days": 90},
]

# STRICT gate constants
PF_THRESHOLD = 1.5
PF_OK_MIN = 4
PNL_OK_MIN = 4
MAX_FOLD_LOSS_MULT = 2.0  # no fold may lose more than 2 x |baseline_per_fold|

# Preload meta model (mirrors entry_sweep / live behaviour)
META = None
try:
    from models.signal_model import SignalModel
    _m = SignalModel(); _m.load(SYMBOL)
    if _m.has_model(SYMBOL):
        META = _m
except Exception:
    META = None


def make_window_loader(start_days, end_days):
    """Return load_data shim slicing cache to [t_max-start, t_max-end)."""
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
    """Run one fold with the frozen winner config."""
    # Build param dict honoring winner_config + NO_CHANGE inheritance
    p = {**DEFAULT_PARAMS, "audit_fix_gates": True}
    p["sl_atr_mult"] = WINNER_SL_ATR_MULT
    p["pullback_atr_retrace"] = WINNER_PULLBACK_RETRACE
    # PULLBACK_MAX_WAIT_BARS: inherit per-symbol/global default (NO_CHANGE)
    # SIGNAL_QUALITY: NO_CHANGE -> do NOT patch config.SIGNAL_QUALITY_SYMBOL
    # DIR_BIAS: NO_CHANGE
    # TRAIL_PROFILE: NO_CHANGE (DEFAULT_PARAMS["trail"] already mirrors live)
    # ELC_TRIGGER_R: NO_CHANGE
    if META is not None:
        p["_meta_model"] = META

    # Patch load_data with fold window
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
    print(f"=== STRICT 5-FOLD WF {SYMBOL} (momentum, 5x90d non-overlapping) ===")
    print(f"Winner: SL_ATR_MULT={WINNER_SL_ATR_MULT} PB_RETRACE={WINNER_PULLBACK_RETRACE}")
    print(f"        SIGNAL_QUALITY/DIR_BIAS/TRAIL/ELC = NO_CHANGE (live config)")
    print(f"Gate:   PF>={PF_THRESHOLD} in >={PF_OK_MIN}/5  AND  PnL>0 in >={PNL_OK_MIN}/5")
    print(f"        AND  no fold loses > {MAX_FOLD_LOSS_MULT}x baseline_per_fold_pnl")

    fold_results = []
    for i, fold in enumerate(FOLDS, 1):
        print(f"\n[Fold {i}/5] {fold['name']} ...")
        r = run_fold(fold)
        fold_results.append(r)
        err = r.get("error", "")
        print(f"  trades={r['trades']} pf={r['pf']} pnl=${r['pnl']} "
              f"wr={r.get('wr',0)}% dd={r.get('dd',0)}"
              + (f"  ERROR={err}" if err else ""))

    # Gate eval
    folds_pf_ok = sum(1 for r in fold_results if r["pf"] >= PF_THRESHOLD)
    folds_pnl_ok = sum(1 for r in fold_results if r["pnl"] > 0)

    baseline_per_fold_pnl = sum(r["pnl"] for r in fold_results) / max(1, len(fold_results))
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
        + (f"PASS" if decision == "SHIP" else "FAIL: " + "; ".join(fail_reasons))
    )

    per_fold_lines = "; ".join(
        f"{r['fold']}: pnl=${r['pnl']} pf={r['pf']} trades={r['trades']}"
        for r in fold_results
    )

    out = {
        "symbol": SYMBOL,
        "strategy": "momentum",
        "winner_config": {
            "SL_ATR_MULT": WINNER_SL_ATR_MULT,
            "SIGNAL_QUALITY": "NO_CHANGE",
            "DIR_BIAS": "NO_CHANGE",
            "TRAIL_PROFILE": "NO_CHANGE",
            "PULLBACK_RETRACE": WINNER_PULLBACK_RETRACE,
            "ELC_TRIGGER_R": "NO_CHANGE",
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
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
