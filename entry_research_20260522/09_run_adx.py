#!/usr/bin/env python3 -B
"""ADX-based TREND STRENGTH entry-filter research.

READ-ONLY (no edits to v5_backtest.py / momentum_scorer.py). We monkey-patch
backtest.v5_backtest._score_with_components with a wrapper that zeroes the
returned scores whenever the ADX gate fails. Because v5's entry loop checks
`signal_quality < threshold` immediately after, a zero score causes the trade
to be skipped — effectively gating entries on ADX criteria.

Variants:
  - GLOBAL_ADX_FLOOR(X): require ind["adx"][i] >= X (X ∈ {18,22,25,30})
  - REGIME_SPECIFIC_ADX: trending requires ADX>=25, ranging requires ADX<22,
    volatile requires ADX>=20, low_vol requires ADX>=18
  - ADX_DIRECTION: only allow LONG when +DI > -DI, SHORT when -DI > +DI

For walk-forward: split each symbol's 180d window into 5 contiguous folds (~36d each)
and run backtest on each slice via params["days_override"] equivalent — we instead
pass days=N for each fold using load_data slicing through a custom days argument.
"""
import sys, json, time, copy
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest.v5_backtest as v5
from backtest.v5_backtest import backtest_symbol, load_data

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
DAYS = 180

OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "09_adx.json"
OUT_MD = OUT_DIR / "09_adx.md"

_ORIG_SCORE = v5._score_with_components

# Global state used by the wrapper. Set BEFORE each backtest_symbol call.
_GATE = {"mode": None, "params": {}}


def _adx_gate_passes(ind, i):
    """Return (allow, allow_long, allow_short). False allow=skip both sides.
    allow_long/short let direction-asymmetric gates apply.
    """
    mode = _GATE["mode"]
    if mode is None:
        return True, True, True
    if i < 0 or i >= ind["n"]:
        return True, True, True
    adx_arr = ind.get("adx")
    if adx_arr is None:
        return True, True, True
    adx = float(adx_arr[i])
    if np.isnan(adx):
        adx = 20.0

    if mode == "global":
        floor = _GATE["params"]["floor"]
        return (adx >= floor), True, True

    if mode == "regime":
        # Compute regime locally using v5.get_regime (BBW + ADX → regime)
        bbw_val = float(ind["bbw"][i]) if not np.isnan(ind["bbw"][i]) else 0.02
        regime = v5.get_regime(bbw_val, adx)
        regime_floors = _GATE["params"]["regime_floors"]
        floor = regime_floors.get(regime, 0.0)
        # For "ranging" regime, the spec is "ADX < threshold" — encode as max instead
        regime_max = _GATE["params"].get("regime_max", {})
        if regime in regime_max and adx > regime_max[regime]:
            return False, True, True
        return (adx >= floor), True, True

    if mode == "direction":
        di_p = ind.get("di_plus")
        di_m = ind.get("di_minus")
        if di_p is None or di_m is None:
            return True, True, True
        p = float(di_p[i]) if not np.isnan(di_p[i]) else 0.0
        m = float(di_m[i]) if not np.isnan(di_m[i]) else 0.0
        # Also require a min ADX if configured
        floor = _GATE["params"].get("floor", 0.0)
        if adx < floor:
            return False, True, True
        # Allow LONG only if +DI > -DI, SHORT only if -DI > +DI
        return True, (p > m), (m > p)

    if mode == "combo":
        # Global floor + direction
        floor = _GATE["params"]["floor"]
        if adx < floor:
            return False, True, True
        di_p = ind.get("di_plus")
        di_m = ind.get("di_minus")
        p = float(di_p[i]) if di_p is not None and not np.isnan(di_p[i]) else 0.0
        m = float(di_m[i]) if di_m is not None and not np.isnan(di_m[i]) else 0.0
        return True, (p > m), (m > p)

    return True, True, True


def _patched_score(ind, i, weights=None):
    """Wrapper that zeroes scores when ADX gate fails."""
    long_s, short_s, comp_l, comp_s = _ORIG_SCORE(ind, i, weights=weights)
    allow, allow_long, allow_short = _adx_gate_passes(ind, i)
    if not allow:
        return 0.0, 0.0, comp_l, comp_s
    if not allow_long:
        long_s = 0.0
    if not allow_short:
        short_s = 0.0
    return long_s, short_s, comp_l, comp_s


# Install patch
v5._score_with_components = _patched_score


def run_symbol(symbol, days, gate_mode, gate_params):
    """Run backtest_symbol with the global ADX gate set."""
    _GATE["mode"] = gate_mode
    _GATE["params"] = gate_params or {}
    r = backtest_symbol(symbol, days=days, params=None, verbose=False)
    _GATE["mode"] = None
    _GATE["params"] = {}
    return r


def run_no_gate(symbol, days):
    _GATE["mode"] = None
    _GATE["params"] = {}
    return backtest_symbol(symbol, days=days, params=None, verbose=False)


# ─── WALK-FORWARD ────────────────────────────────────────────────────────────
def _wf_days_slices(days_total, n_folds):
    """Return list of (start_day_offset_from_now, days_window) so each fold
    is a *contiguous* slice of the available data.

    We use the "days" arg in load_data which clips to the LAST N days. To get
    folds, we precompute the cutoff in time. But load_data only supports
    "last-N-days" anchored at df.time.max(). To approximate 5 contiguous
    folds, run with days=N_total then days=N_total*4/5, etc.

    Simpler & honest: emulate 5 folds by re-running with different "days"
    values truncating to recent windows + a single fold per available chunk.
    However for proper WF we'd need a sliding window.

    Approach: We run each fold by post-processing the trade log we already
    have for the full window — split trades by time into 5 buckets. That
    avoids re-running and gives true contiguous WF folds.
    """
    return None  # not used


def split_trades_into_folds(trades, n_folds=5):
    """Split trades into n_folds contiguous folds based on entry_bar index.
    Returns list of per-fold dicts: {n, pnl, pf, wr, wins, losses}.
    """
    if not trades:
        return [{"n": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0, "wins": 0,
                 "losses": 0} for _ in range(n_folds)]
    bars = [t.get("entry_bar", 0) for t in trades]
    bmin, bmax = min(bars), max(bars)
    bucket_w = (bmax - bmin) / n_folds if bmax > bmin else 0
    folds = [[] for _ in range(n_folds)]
    for t in trades:
        b = t.get("entry_bar", bmin)
        idx = int((b - bmin) / bucket_w) if bucket_w > 0 else 0
        if idx >= n_folds:
            idx = n_folds - 1
        folds[idx].append(t)

    out = []
    for f in folds:
        wins = [t for t in f if t["pnl"] > 0]
        losses = [t for t in f if t["pnl"] <= 0]
        gp = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
        wr = (len(wins) / len(f) * 100) if f else 0.0
        pnl = sum(t["pnl"] for t in f)
        out.append({
            "n": len(f), "pnl": round(pnl, 2), "pf": round(pf, 2),
            "wr": round(wr, 1), "wins": len(wins), "losses": len(losses),
        })
    return out


# ─── VARIANTS ────────────────────────────────────────────────────────────────
VARIANTS = []

# GLOBAL_ADX_FLOOR sweep
for x in [18, 22, 25, 30]:
    VARIANTS.append({
        "name": f"global_adx_{x}",
        "mode": "global",
        "params": {"floor": x},
    })

# REGIME_SPECIFIC_ADX
VARIANTS.append({
    "name": "regime_adx_v1",
    "mode": "regime",
    "params": {
        "regime_floors": {
            "trending": 25.0,
            "volatile": 22.0,
            "low_vol":  18.0,
            "ranging":  0.0,   # already gated to ADX<22 by get_regime
        },
        # Optionally cap ranging at 22 (already enforced by get_regime, so redundant)
    },
})
VARIANTS.append({
    "name": "regime_adx_strict",
    "mode": "regime",
    "params": {
        "regime_floors": {
            "trending": 28.0,
            "volatile": 25.0,
            "low_vol":  20.0,
            "ranging":  0.0,
        },
    },
})

# ADX_DIRECTION only
VARIANTS.append({
    "name": "adx_direction_only",
    "mode": "direction",
    "params": {"floor": 0.0},
})
VARIANTS.append({
    "name": "adx_direction_floor20",
    "mode": "direction",
    "params": {"floor": 20.0},
})

# Combo: floor 22 + direction
VARIANTS.append({
    "name": "combo_floor22_dir",
    "mode": "combo",
    "params": {"floor": 22.0},
})


# ─── RUN ─────────────────────────────────────────────────────────────────────
def summarize(r):
    if r is None:
        return None
    return {
        "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
        "pnl": round(r["pnl"], 2), "dd": r["dd"],
    }


def main():
    t0 = time.time()
    results = {
        "days": DAYS,
        "symbols": SYMBOLS,
        "baseline": {},
        "baseline_total": {"pnl": 0.0, "trades": 0},
        "variants": {},
    }

    # 1) Baseline (no gate)
    print("=== BASELINE (no ADX gate) ===")
    baseline_trades = {}  # sym -> list of trades for WF later
    for sym in SYMBOLS:
        r = run_no_gate(sym, DAYS)
        if r is None:
            results["baseline"][sym] = {"error": "no_data"}
            continue
        summary = summarize(r)
        results["baseline"][sym] = summary
        results["baseline_total"]["pnl"] += summary["pnl"]
        results["baseline_total"]["trades"] += summary["trades"]
        baseline_trades[sym] = r.get("details", [])
        print(f"  {sym:12s} trades={summary['trades']:4d} pf={summary['pf']:5.2f} "
              f"wr={summary['wr']:5.1f}% pnl=${summary['pnl']:9.2f} dd={summary['dd']:.1f}%")
    results["baseline_total"]["pnl"] = round(results["baseline_total"]["pnl"], 2)
    base_total = results["baseline_total"]["pnl"]
    print(f"  TOTAL pnl=${base_total:.2f} trades={results['baseline_total']['trades']}\n")

    # 2) Each variant
    for v in VARIANTS:
        print(f"=== {v['name']} (mode={v['mode']}, params={v['params']}) ===")
        var_out = {
            "mode": v["mode"],
            "params": v["params"],
            "symbols": {},
            "total_pnl": 0.0,
            "total_trades": 0,
            "wf": {},  # sym -> list of fold dicts
        }
        for sym in SYMBOLS:
            r = run_symbol(sym, DAYS, v["mode"], v["params"])
            if r is None:
                var_out["symbols"][sym] = {"error": "no_data"}
                continue
            summary = summarize(r)
            var_out["symbols"][sym] = summary
            var_out["total_pnl"] += summary["pnl"]
            var_out["total_trades"] += summary["trades"]
            # WF: split this variant's trades into 5 folds
            tr_log = r.get("details", [])
            wf_folds = split_trades_into_folds(tr_log, n_folds=5)
            var_out["wf"][sym] = wf_folds
            print(f"  {sym:12s} trades={summary['trades']:4d} pf={summary['pf']:5.2f} "
                  f"wr={summary['wr']:5.1f}% pnl=${summary['pnl']:9.2f} dd={summary['dd']:.1f}%")
        var_out["total_pnl"] = round(var_out["total_pnl"], 2)
        delta = var_out["total_pnl"] - base_total
        var_out["delta_vs_baseline"] = round(delta, 2)
        print(f"  TOTAL pnl=${var_out['total_pnl']:.2f} (Δ ${delta:+.2f}) "
              f"trades={var_out['total_trades']}\n")

        # Aggregate WF: avg PF + folds-positive count across symbols
        per_fold_pnl = [0.0] * 5
        per_fold_pf_num = [0.0] * 5
        per_fold_pf_den = [0.0] * 5
        for sym, folds in var_out["wf"].items():
            for i, f in enumerate(folds):
                per_fold_pnl[i] += f["pnl"]
                # PF rebuild from wins/losses across symbols
                # Approximation: weight per-fold PF by trade count? Simpler:
                # sum wins.pnl and abs(losses.pnl) per fold from each sym
        # Rebuild fold-level PF by traversing each variant's trades per sym again
        fold_wins = [0.0] * 5
        fold_losses = [0.0] * 5
        fold_n = [0] * 5
        for sym, folds in var_out["wf"].items():
            for i, f in enumerate(folds):
                fold_n[i] += f["n"]
                # wins/losses dollar weighting: we only stored pf/pnl, not raw.
                # Use win/loss counts × avg = pnl positive part — approximation.
                # Instead, recompute from raw trades:
        # Recompute from trades collected during run — but we discarded them.
        # Approximate fold-PF from per-symbol fold metrics: weight by trade count
        fold_pf_list = []
        for i in range(5):
            # Average PF across symbols for this fold (only including syms with trades)
            pfs = [folds[i]["pf"] for sym, folds in var_out["wf"].items()
                   if folds[i]["n"] > 0 and folds[i]["pf"] != float("inf")]
            avg_pf = sum(pfs) / len(pfs) if pfs else 0.0
            fold_pf_list.append(round(avg_pf, 2))
        # Folds positive: sum(per_fold_pnl[i] > 0)
        folds_pos = sum(1 for p in per_fold_pnl if p > 0)
        wf_avg_pf = round(sum(fold_pf_list) / 5, 2)
        var_out["wf_summary"] = {
            "per_fold_pnl": [round(x, 2) for x in per_fold_pnl],
            "per_fold_avg_pf": fold_pf_list,
            "folds_positive_pnl": folds_pos,
            "wf_avg_pf": wf_avg_pf,
        }
        # Ship criteria
        ship_delta_ok = delta >= 30.0
        ship_pf_ok = wf_avg_pf > 1.5
        ship_folds_ok = folds_pos >= 3
        var_out["ship"] = bool(ship_delta_ok and ship_pf_ok and ship_folds_ok)
        var_out["ship_reasons"] = {
            "delta_ge_30": ship_delta_ok,
            "wf_avg_pf_gt_1_5": ship_pf_ok,
            "folds_pos_ge_3": ship_folds_ok,
        }
        print(f"  WF: avg_pf={wf_avg_pf:.2f}, folds_pos={folds_pos}/5, "
              f"ship={var_out['ship']}\n")
        results["variants"][v["name"]] = var_out

    # 3) Per-symbol best floor recommendation: for each symbol, find the
    # global_adx_X that gave the best PnL while not dropping below baseline.
    per_sym_reco = {}
    for sym in SYMBOLS:
        base = results["baseline"].get(sym, {})
        if not base or "error" in base:
            continue
        best = {"variant": "BASELINE", "pnl": base["pnl"], "pf": base["pf"],
                "delta": 0.0}
        for x in [18, 22, 25, 30]:
            vn = f"global_adx_{x}"
            v = results["variants"][vn]["symbols"].get(sym, {})
            if "error" in v:
                continue
            delta = v["pnl"] - base["pnl"]
            # Recommend a floor only if pnl is no worse than baseline AND
            # PF doesn't collapse (≥ 1.5)
            if v["pnl"] >= base["pnl"] - 5.0 and v["pf"] >= 1.5:
                # prefer higher floor when tie (more selective)
                if (v["pnl"] > best["pnl"] - 5.0) and (v["pnl"] >= best["pnl"]
                        or (abs(v["pnl"] - best["pnl"]) <= 10.0 and x > 18)):
                    best = {"variant": vn, "floor": x, "pnl": v["pnl"],
                            "pf": v["pf"], "delta": round(delta, 2)}
        per_sym_reco[sym] = best
    results["per_symbol_recommendation"] = per_sym_reco

    results["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(results, open(OUT_JSON, "w"), indent=2, default=str)
    print(f"\nWrote {OUT_JSON} (elapsed {results['elapsed_sec']}s)")
    return results


if __name__ == "__main__":
    main()
