#!/usr/bin/env python3 -B
"""
MTF Confluence research driver — READ-ONLY.

Tests 5 variants of the higher-timeframe confluence gate against the
existing baseline on 8 symbols × 180-day backtest, plus walk-forward
validation across 5 folds.

Variants:
  BASELINE          — current mtf_verdict_at_bar (REJECT only when 2+ opposed)
  REQUIRE_3_OF_3    — REJECT unless ALL three TFs aligned (W1==D1==H4==dir)
  ADD_M15           — same as 3-of-3 but adds a 4th gate using H1 EMA(20/50/200)
                       stack as M15-confirmation proxy; needs 4-of-4
  REJECT_AGAINST_W1 — REJECT whenever W1 opposes entry direction (W1 FLAT ok)
  WEEKLY_BIAS_ONLY  — only trade with W1 trend (W1 FLAT also rejected)

Output:
  entry_research_20260522/04_mtf_confluence.json
  entry_research_20260522/04_mtf_confluence.md

Constraints honoured:
  - READ-ONLY: never writes to live config, auto_tuned.py, etc.
  - Overrides via params + monkey-patch of signals.mtf_trend module only.
  - python3 -B caller.
"""
from __future__ import annotations
import json, sys, time, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

OUT_DIR = ROOT / "entry_research_20260522"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY",
    "EURUSD", "US2000.r", "UKOUSD", "JPN225ft",
]
DAYS = 180
N_FOLDS = 5
FOLD_DAYS = DAYS // N_FOLDS  # 36 days/fold

# ----------------------------------------------------------------------
# Bring in the backtest + mtf module ONCE; monkey-patch verdict logic per
# variant. Variants share precomputed trend arrays (cheap O(n)).
# ----------------------------------------------------------------------
from backtest import v5_backtest as bt  # noqa: E402
from signals import mtf_trend as _mtf  # noqa: E402

# Capture originals so we can restore between variants.
_ORIG_VERDICT  = _mtf.mtf_verdict_at_bar
_ORIG_PRECOMP  = _mtf.precompute_mtf_trends
_ORIG_LOAD     = bt.load_data


# ----------------------------------------------------------------------
# Walk-forward: replace load_data with a window-aware version. Each fold
# is 36d; 5 sequential, non-overlapping folds spanning the same 180d span
# that the in-sample run uses. We resample by truncating max-time per fold.
# ----------------------------------------------------------------------
def _make_windowed_load(end_days_back: int, window_days: int):
    """Return load_data variant: keeps bars in (max_time - end_days_back - window_days,
    max_time - end_days_back]."""

    def _load(symbol, days=90):  # signature compatible
        meta = bt.ALL_SYMBOLS.get(symbol)
        if not meta:
            return None
        path = bt.CACHE / meta["cache"]
        if not path.exists():
            return None
        df = pickle.load(open(path, "rb"))
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        tmax = df["time"].max()
        upper = tmax - pd.Timedelta(days=end_days_back)
        lower = upper - pd.Timedelta(days=window_days)
        # Use bars in (lower, upper]; also include enough warmup BEFORE
        # lower so indicators settle. Add 30 days warmup.
        warmup_lower = lower - pd.Timedelta(days=30)
        df = df[(df["time"] >= warmup_lower) & (df["time"] <= upper)].reset_index(drop=True)
        return df

    return _load


# ----------------------------------------------------------------------
# Variant verdict patches
# ----------------------------------------------------------------------
def _verdict_require_3_of_3(precomputed, bar_idx, entry_direction):
    aligned = opposed = flat = 0
    total = len(precomputed)
    for tf, dirs in precomputed.items():
        if bar_idx >= len(dirs):
            flat += 1
            continue
        d = dirs[bar_idx]
        if d == 0:
            flat += 1
        elif d == entry_direction:
            aligned += 1
        else:
            opposed += 1
    if aligned == total:
        return "SNIPER"
    return "REJECT"  # anything short of full alignment is rejected


def _verdict_reject_against_w1(precomputed, bar_idx, entry_direction):
    # Reject only when W1 is explicitly opposite.
    w1 = precomputed.get("W1")
    if w1 is not None and bar_idx < len(w1):
        if w1[bar_idx] != 0 and w1[bar_idx] != entry_direction:
            return "REJECT"
    # Otherwise fall back to baseline behaviour for D1/H4 mix.
    return _ORIG_VERDICT(precomputed, bar_idx, entry_direction)


def _verdict_weekly_bias_only(precomputed, bar_idx, entry_direction):
    w1 = precomputed.get("W1")
    if w1 is None or bar_idx >= len(w1):
        return "REJECT"
    return "STRONG" if w1[bar_idx] == entry_direction else "REJECT"


def _verdict_add_m15(precomputed, bar_idx, entry_direction):
    # 4-of-4 with M15 (key "M15" added at precompute time).
    aligned = opposed = flat = 0
    total = len(precomputed)
    for tf, dirs in precomputed.items():
        if bar_idx >= len(dirs):
            flat += 1
            continue
        d = dirs[bar_idx]
        if d == 0:
            flat += 1
        elif d == entry_direction:
            aligned += 1
        else:
            opposed += 1
    if aligned == total:
        return "SNIPER"
    return "REJECT"


def _precompute_with_m15(h1_closes, tfs=("W1", "D1", "H4")):
    """Same as original precompute, plus an extra 'M15' synthetic key.

    Because we operate on the H1 candle stream (no true M15 in cache for
    all symbols), the M15 confirmation gate proxies the entry-TF EMA
    stack: EMA(20) > EMA(50) > EMA(200) on H1 closes. This mirrors
    "3 EMAs aligned on entry timeframe" — Elder's third screen.
    """
    out = _ORIG_PRECOMP(h1_closes, tfs=tfs)

    n = len(h1_closes)
    if n < 50:
        out["M15"] = np.zeros(n, dtype=np.int8)
        return out
    arr = np.asarray(h1_closes, dtype=float)

    def _ema_full(a, p):
        alpha = 2.0 / (p + 1)
        e = np.empty_like(a)
        e[0] = a[0]
        for i in range(1, len(a)):
            e[i] = alpha * a[i] + (1 - alpha) * e[i - 1]
        return e

    e20 = _ema_full(arr, 20)
    e50 = _ema_full(arr, 50)
    p200 = min(200, n - 1)
    e200 = _ema_full(arr, p200)
    long_mask = (e20 > e50) & (e50 > e200)
    short_mask = (e20 < e50) & (e50 < e200)
    out["M15"] = np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(np.int8)
    return out


# ----------------------------------------------------------------------
# Variant registry. Each entry sets up + tears down the monkey-patches.
# ----------------------------------------------------------------------
def _apply_variant(name: str):
    """Patch signals.mtf_trend & the local import in v5_backtest."""
    # v5_backtest imports verdict/precompute inside the per-symbol loop
    # via `from signals.mtf_trend import precompute_mtf_trends, mtf_verdict_at_bar`
    # so swapping the module-level attrs is sufficient (re-import gets the
    # new attributes on every backtest_symbol call).
    if name == "BASELINE":
        _mtf.mtf_verdict_at_bar = _ORIG_VERDICT
        _mtf.precompute_mtf_trends = _ORIG_PRECOMP
    elif name == "REQUIRE_3_OF_3":
        _mtf.mtf_verdict_at_bar = _verdict_require_3_of_3
        _mtf.precompute_mtf_trends = _ORIG_PRECOMP
    elif name == "ADD_M15":
        _mtf.mtf_verdict_at_bar = _verdict_add_m15
        _mtf.precompute_mtf_trends = _precompute_with_m15
    elif name == "REJECT_AGAINST_W1":
        _mtf.mtf_verdict_at_bar = _verdict_reject_against_w1
        _mtf.precompute_mtf_trends = _ORIG_PRECOMP
    elif name == "WEEKLY_BIAS_ONLY":
        _mtf.mtf_verdict_at_bar = _verdict_weekly_bias_only
        _mtf.precompute_mtf_trends = _ORIG_PRECOMP
    else:
        raise ValueError(f"unknown variant {name}")


def _restore():
    _mtf.mtf_verdict_at_bar = _ORIG_VERDICT
    _mtf.precompute_mtf_trends = _ORIG_PRECOMP
    bt.load_data = _ORIG_LOAD


# ----------------------------------------------------------------------
# Mirror live params (audit_fix_gates + cost overlays) for fidelity.
# ----------------------------------------------------------------------
def _bt_params():
    return {
        **bt.DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": False,   # spread-only per project memory
        "with_commission": True,
        "with_swap": True,
    }


def _safe_run(symbol, days):
    try:
        return bt.backtest_symbol(symbol, days=days, params=_bt_params(), verbose=False)
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "trades": 0,
                "pf": 0, "pnl": 0, "dd": 0, "wr": 0}


def _summarise(r):
    if not r:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "pnl": 0.0, "dd": 0.0}
    return {
        "trades": int(r.get("trades", 0)),
        "pf":     float(r.get("pf", 0)),
        "wr":     float(r.get("wr", 0)),
        "pnl":    float(r.get("pnl", 0)),
        "dd":     float(r.get("dd", 0)),
    }


def _run_inscope(variant, symbols, days):
    """Single 180d pass (all 8 symbols) under the named variant."""
    _apply_variant(variant)
    bt.load_data = _ORIG_LOAD  # full window
    out = {}
    for sym in symbols:
        r = _safe_run(sym, days)
        out[sym] = _summarise(r)
    return out


def _run_walkforward(variant, symbols, n_folds, fold_days):
    """5 non-overlapping folds, each fold_days long, anchored at the most
    recent candle. Fold 0 = newest, fold N = oldest."""
    _apply_variant(variant)
    per_symbol = {}
    for sym in symbols:
        folds = []
        for fold_i in range(n_folds):
            end_back = fold_i * fold_days
            bt.load_data = _make_windowed_load(end_back, fold_days)
            r = _safe_run(sym, fold_days)
            folds.append(_summarise(r))
        per_symbol[sym] = folds
    bt.load_data = _ORIG_LOAD
    return per_symbol


def main():
    t0 = time.time()
    variants = [
        "BASELINE",
        "REQUIRE_3_OF_3",
        "ADD_M15",
        "REJECT_AGAINST_W1",
        "WEEKLY_BIAS_ONLY",
    ]

    print(f"[mtf-confluence] symbols={len(SYMBOLS)} days={DAYS} "
          f"folds={N_FOLDS}x{FOLD_DAYS}d variants={len(variants)}")

    out = {
        "meta": {
            "days": DAYS, "n_folds": N_FOLDS, "fold_days": FOLD_DAYS,
            "symbols": SYMBOLS, "variants": variants,
            "ship_criteria": {
                "per_symbol_delta_min_usd": 30.0,
                "wf_avg_pf_min": 1.5,
                "wf_positive_folds_min": 3,
            },
        },
        "inscope_180d": {},
        "walkforward": {},
    }

    for v in variants:
        v_t0 = time.time()
        print(f"\n[mtf-confluence] === variant: {v} ===")
        in_scope = _run_inscope(v, SYMBOLS, DAYS)
        for sym, r in in_scope.items():
            print(f"  IS  {sym:10s} trades={r['trades']:4d} pf={r['pf']:>5.2f} "
                  f"wr={r['wr']:>5.1f}% pnl=${r['pnl']:>+8.2f} dd={r['dd']:>5.1f}%")
        out["inscope_180d"][v] = in_scope

        wf = _run_walkforward(v, SYMBOLS, N_FOLDS, FOLD_DAYS)
        # Compact WF print
        for sym, folds in wf.items():
            pfs = [f["pf"] for f in folds]
            pnls = [f["pnl"] for f in folds]
            pos = sum(1 for p in pnls if p > 0)
            print(f"  WF  {sym:10s} pfs={['%.2f'%p for p in pfs]} "
                  f"pnls={['%+0.0f'%p for p in pnls]} pos={pos}/{N_FOLDS}")
        out["walkforward"][v] = wf
        print(f"[mtf-confluence] variant {v} took {time.time()-v_t0:.1f}s")

    _restore()

    # Decisions per symbol per variant -- ship if Δ ≥ $30 vs baseline AND
    # WF avg pf > 1.5 AND ≥3/5 folds positive.
    decisions = {}
    baseline = out["inscope_180d"]["BASELINE"]
    for v in variants:
        if v == "BASELINE":
            continue
        per_sym = {}
        for sym in SYMBOLS:
            base_pnl = baseline.get(sym, {}).get("pnl", 0.0)
            var_pnl  = out["inscope_180d"][v].get(sym, {}).get("pnl", 0.0)
            delta = var_pnl - base_pnl
            wf_folds = out["walkforward"][v].get(sym, [])
            wf_pfs   = [f["pf"] for f in wf_folds if f["trades"] > 0]
            wf_avg_pf = float(np.mean(wf_pfs)) if wf_pfs else 0.0
            wf_pos   = sum(1 for f in wf_folds if f["pnl"] > 0)
            ship = (delta >= 30.0 and wf_avg_pf > 1.5 and wf_pos >= 3)
            per_sym[sym] = {
                "base_pnl": round(base_pnl, 2),
                "var_pnl":  round(var_pnl, 2),
                "delta":    round(delta, 2),
                "wf_avg_pf": round(wf_avg_pf, 2),
                "wf_positive_folds": int(wf_pos),
                "ship": bool(ship),
            }
        decisions[v] = per_sym
    out["decisions"] = decisions

    out_json = OUT_DIR / "04_mtf_confluence.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[mtf-confluence] wrote {out_json}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
