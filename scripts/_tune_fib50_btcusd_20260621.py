#!/usr/bin/env python3 -B
"""
Coord-descent sweep for FIB50 strategy on BTCUSD (365d M15 cache).

Strategy: coord-descent over the 8 axes listed in the task. Each pass walks
all axes; the best value (by PF, subject to MIN_TRADES floor) becomes the
new locked value. Pass 2 starts from Pass 1 winners. Final stage re-runs
top-3 combined configs for stability.

Injection: the detector reads ENTRY_ZONE_LO/HI, SWING_PIVOT_N, ATR_BUFFER,
MAX_SL_R, MIN_RR, MIN_IMPULSE_ATR from constructor params; but the
USE_WIDE_SL and DIRECTION_FILTER toggles are read ONLY from the per-symbol
override dict (FIB50_PARAM_OVERRIDES). We patch both: the constructor
params dict on the Fib50Strategy instance AND patch the module-level
override dict so all axes take effect.

Output: backtest/results/fib50_tune_btcusd_20260621.json
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from agent import fib50_strategy as fib50_mod  # noqa: E402
from backtest import fib50_backtest as bt  # noqa: E402

SYMBOL = "BTCUSD"
DAYS = 365
MIN_TRADES_FLOOR = 40
DD_CEILING_PCT = 30.0

OUT_DIR = ROOT / "backtest" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / f"fib50_tune_{SYMBOL.lower()}_20260621.json"

# ── Axes ──────────────────────────────────────────────────────────────────
AXES = {
    "MIN_IMPULSE_ATR":  [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "MIN_RR":           [1.0, 1.2, 1.5, 1.8, 2.0],
    "ENTRY_ZONE_LO":    [0.382, 0.50, 0.618],
    "ATR_BUFFER":       [0.1, 0.2, 0.3, 0.5],
    "USE_WIDE_SL":      [False, True],
    "SWING_PIVOT_N":    [3, 5, 7],
    "DIRECTION_FILTER": ["BOTH", "LONG", "SHORT"],
    "MAX_SL_R":         [4.0, 6.0, 8.0, 12.0],
}

# Defaults that mirror current live config (FIB50_PARAM_OVERRIDES empty for BTCUSD)
DEFAULTS = {
    "MIN_IMPULSE_ATR":  2.0,
    "MIN_RR":           1.5,
    "ENTRY_ZONE_LO":    0.50,
    "ATR_BUFFER":       0.20,
    "USE_WIDE_SL":      False,
    "SWING_PIVOT_N":    5,
    "DIRECTION_FILTER": "BOTH",
    "MAX_SL_R":         8.0,
}

# Coord-descent order (most impactful first)
AXIS_ORDER = [
    "MIN_IMPULSE_ATR",
    "DIRECTION_FILTER",
    "USE_WIDE_SL",
    "MIN_RR",
    "MAX_SL_R",
    "ENTRY_ZONE_LO",
    "ATR_BUFFER",
    "SWING_PIVOT_N",
]


# ── Inject + run one combo ────────────────────────────────────────────────
def _params_to_overrides(params):
    """Translate axis dict → FIB50_PARAM_OVERRIDES entry the detector reads."""
    return {
        "MIN_IMPULSE_ATR":  float(params["MIN_IMPULSE_ATR"]),
        "MIN_RR":           float(params["MIN_RR"]),
        "MAX_SL_R":         float(params["MAX_SL_R"]),
        "ATR_BUFFER":       float(params["ATR_BUFFER"]),
        "USE_WIDE_SL":      bool(params["USE_WIDE_SL"]),
        "DIRECTION_FILTER": str(params["DIRECTION_FILTER"]),
    }


# We also need ENTRY_ZONE_LO/HI and SWING_PIVOT_N to take effect via
# the constructor — but the detector creates a fresh Fib50Strategy inside
# backtest_symbol(). So monkey-patch the class constructor defaults too.
_ORIG_INIT = fib50_mod.Fib50Strategy.__init__


def _make_patched_init(zone_lo, swing_n):
    """Wrap Fib50Strategy.__init__ so every instance created during this
    combo inherits the per-combo ENTRY_ZONE_LO/HI + SWING_PIVOT_N."""
    def _patched(self, state, params=None):
        p = dict(params or {})
        p.setdefault("ENTRY_ZONE_LO", float(zone_lo))
        p.setdefault("ENTRY_ZONE_HI", float(zone_lo) + 0.118)
        p.setdefault("SWING_PIVOT_N", int(swing_n))
        _ORIG_INIT(self, state, p)
    return _patched


def run_combo(params):
    """Run one combo through the BT. Returns summary dict (or None on error)."""
    # 1. Patch the per-symbol override dict
    saved_overrides = copy.deepcopy(fib50_mod.FIB50_PARAM_OVERRIDES)
    fib50_mod.FIB50_PARAM_OVERRIDES = {
        SYMBOL: _params_to_overrides(params),
    }

    # 2. Patch class constructor to inject ENTRY_ZONE / SWING_PIVOT_N
    fib50_mod.Fib50Strategy.__init__ = _make_patched_init(
        params["ENTRY_ZONE_LO"], params["SWING_PIVOT_N"]
    )

    try:
        summary, _ = bt.backtest_symbol(SYMBOL, days=DAYS)
    except Exception as e:
        summary = {"status": f"ERROR: {e}"}
    finally:
        # 3. Restore
        fib50_mod.FIB50_PARAM_OVERRIDES = saved_overrides
        fib50_mod.Fib50Strategy.__init__ = _ORIG_INIT

    return summary


def is_acceptable(summary):
    """MIN_TRADES floor + DD sanity. Used to filter, not to score."""
    if not summary or summary.get("status") != "OK":
        return False
    if summary.get("trades", 0) < MIN_TRADES_FLOOR:
        return False
    if summary.get("max_dd_pct", 999) >= DD_CEILING_PCT:
        return False
    return True


def score(summary):
    """Score = PF (subject to floor). Tie-break by total_R."""
    if not is_acceptable(summary):
        return (-999.0, -999.0)
    return (summary["pf"], summary["total_R"])


# ── Coord descent ─────────────────────────────────────────────────────────
def coord_descent(start_params, results_log):
    """One full pass through AXIS_ORDER. Returns updated params + best summary."""
    locked = dict(start_params)
    best_summary = None
    for axis in AXIS_ORDER:
        values = AXES[axis]
        axis_results = []
        for v in values:
            trial = dict(locked)
            trial[axis] = v
            t0 = time.time()
            summary = run_combo(trial)
            dt = time.time() - t0
            sc = score(summary)
            row = {
                "axis": axis, "value": v, "params": dict(trial),
                "summary": summary, "score": sc, "dt_sec": dt,
            }
            axis_results.append(row)
            results_log.append(row)
            status = summary.get("status", "?")
            tr = summary.get("trades", 0)
            pf = summary.get("pf", float("nan"))
            tR = summary.get("total_R", float("nan"))
            dd = summary.get("max_dd_pct", float("nan"))
            print(f"   axis={axis:<18} val={str(v):<7} "
                  f"status={status:<14} trades={tr:>4} pf={pf:>6.2f} "
                  f"totR={tR:>+7.1f} dd={dd:>5.1f}% sc={sc[0]:>6.2f}/{sc[1]:>+7.1f} "
                  f"[{dt:.1f}s]", flush=True)

        # Pick best by score; if no acceptable result, keep prior value
        acceptable = [r for r in axis_results if is_acceptable(r["summary"])]
        if acceptable:
            best = max(acceptable, key=lambda r: r["score"])
            old_val = locked[axis]
            locked[axis] = best["value"]
            best_summary = best["summary"]
            print(f"   >> axis {axis} locked: {old_val} -> {best['value']}  "
                  f"PF={best_summary['pf']:.2f} totR={best_summary['total_R']:+.1f}",
                  flush=True)
        else:
            print(f"   >> axis {axis} KEPT default {locked[axis]} (no acceptable)",
                  flush=True)
    return locked, best_summary


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("=" * 100)
    print(f"FIB50 coord-descent — {SYMBOL} {DAYS}d")
    print("=" * 100)

    # Baseline (defaults, no overrides)
    print("\n[BASELINE] running with defaults...")
    baseline = run_combo(DEFAULTS)
    print(f"   baseline: trades={baseline.get('trades',0)} "
          f"pf={baseline.get('pf','?')} totR={baseline.get('total_R','?')} "
          f"dd={baseline.get('max_dd_pct','?')}%")

    results_log = []

    # Pass 1: from defaults
    print("\n" + "-" * 100)
    print("[PASS 1] coord-descent from defaults")
    print("-" * 100)
    p1_params, p1_summary = coord_descent(DEFAULTS, results_log)

    # Pass 2: from pass 1 winners
    print("\n" + "-" * 100)
    print("[PASS 2] coord-descent from Pass 1 winners")
    print("-" * 100)
    p2_params, p2_summary = coord_descent(p1_params, results_log)

    # Top-3 stable re-run
    print("\n" + "-" * 100)
    print("[STABILITY] re-run top-3 distinct param sets")
    print("-" * 100)
    # Collect all acceptable combos from results_log, dedupe by params signature
    seen = {}
    for row in results_log:
        if not is_acceptable(row["summary"]):
            continue
        key = json.dumps(row["params"], sort_keys=True, default=str)
        if key not in seen or row["score"] > seen[key]["score"]:
            seen[key] = row
    # Add pass winners explicitly
    for p in (p1_params, p2_params):
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in seen:
            s = run_combo(p)
            if is_acceptable(s):
                seen[key] = {"params": dict(p), "summary": s,
                             "score": score(s), "axis": "(pass-winner)", "value": "-",
                             "dt_sec": 0.0}
    ranked = sorted(seen.values(), key=lambda r: r["score"], reverse=True)
    top3 = ranked[:3]
    stability = []
    for i, r in enumerate(top3, 1):
        # Re-run to confirm determinism (should be identical, but logs it)
        s = run_combo(r["params"])
        sc = score(s)
        stability.append({"rank": i, "params": r["params"],
                          "first": r["summary"], "rerun": s, "rerun_score": sc})
        print(f"   #{i} pf_first={r['summary']['pf']:.2f} pf_rerun={s.get('pf','?')} "
              f"totR_first={r['summary']['total_R']:+.1f} "
              f"totR_rerun={s.get('total_R','?')} params={r['params']}",
              flush=True)

    # Choose best: highest rerun PF (subject to floor); tiebreak total_R
    final_best = None
    for s_row in stability:
        if is_acceptable(s_row["rerun"]):
            if final_best is None or s_row["rerun_score"] > final_best["rerun_score"]:
                final_best = s_row

    combos_tried = sum(1 for r in results_log if is_acceptable(r["summary"])) + len(stability)

    out = {
        "symbol": SYMBOL,
        "days": DAYS,
        "min_trades_floor": MIN_TRADES_FLOOR,
        "dd_ceiling_pct": DD_CEILING_PCT,
        "baseline": baseline,
        "pass1_winner": {"params": p1_params, "summary": p1_summary},
        "pass2_winner": {"params": p2_params, "summary": p2_summary},
        "stability": stability,
        "final_best": final_best,
        "combos_scored": combos_tried,
        "combos_total_attempts": len(results_log) + len(stability),
        "elapsed_sec": time.time() - t_start,
        "results_log": results_log,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 100)
    print(f"DONE in {out['elapsed_sec']:.1f}s — wrote {OUT_FILE}")
    print(f"Scored combos (acceptable): {combos_tried}  "
          f"Total attempts: {out['combos_total_attempts']}")
    if final_best:
        fb = final_best["rerun"]
        print(f"FINAL BEST: PF={fb['pf']:.2f}  totR={fb['total_R']:+.1f}  "
              f"trades={fb['trades']}  WR={fb['wr']*100:.1f}%  "
              f"DD={fb['max_dd_pct']:.1f}%")
        print(f"PARAMS: {final_best['params']}")
    else:
        print("FINAL BEST: NONE — no combo passed acceptance.")
    print("=" * 100)


if __name__ == "__main__":
    main()
