#!/usr/bin/env python3 -B
"""
Coordinate-descent entry-param sweep for BTCUSD (90d M15).

Axes:
  1. MIN_SCORE       (mapped to quality 0-100 = raw/12*100); grid: raw [5.5,6,6.5,7,7.5]
                     -> quality [45.83, 50.0, 54.17, 58.33, 62.5]
  2. PULLBACK_ATR_RETRACE [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
  3. PULLBACK_MAX_WAIT_BARS [1, 2, 3]
  4. VWAP_TOO_FAR    SKIPPED (BTCUSD not in VWAP_GATE_SYMBOLS, param doesn't exist)

Output: backtest/results/entry_sweep_20260614/BTCUSD.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

SYMBOL = "BTCUSD"
DAYS = 90
OUT_DIR = ROOT / "backtest" / "results" / "entry_sweep_20260614"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / f"{SYMBOL}.json"

# Grids
RAW_SCORES = [5.5, 6.0, 6.5, 7.0, 7.5]
RETRACES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
WAITS = [1, 2, 3]

# Default (current) quality dict for BTCUSD per auto_tuned.py
BASELINE_QUALITY = {"trending": 30, "ranging": 32, "volatile": 35, "low_vol": 30}
BASELINE_RETRACE = 0.2  # global default (BTCUSD not in PULLBACK_CONFIG_PER_SYMBOL)
BASELINE_WAIT = 1


def raw_to_quality_dict(raw):
    """Map raw MIN_SCORE to quality dict (uniform across regimes).

    quality = raw / 12 * 100 (matches v5_backtest.py:596 formula).
    """
    q = raw / 12.0 * 100.0
    return {"trending": q, "ranging": q, "volatile": q, "low_vol": q}


def run(retrace, wait, quality_dict):
    """Run one BT combo. Monkey-patches config.SIGNAL_QUALITY_SYMBOL so the
    per-sym overlay in v5_backtest.py:536-538 stamps our test thresholds.
    """
    # Patch SIGNAL_QUALITY_SYMBOL for this symbol
    orig_sym_q = config.SIGNAL_QUALITY_SYMBOL.get(SYMBOL)
    config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = dict(quality_dict)
    try:
        p = {
            **DEFAULT_PARAMS,
            "audit_fix_gates": True,
            "with_slippage": True,
            "with_commission": True,
            "with_swap": True,
            "pullback_atr_retrace": retrace,
            "pullback_max_wait": int(wait),
            "min_quality": dict(quality_dict),  # also pass through p
        }
        try:
            r = backtest_symbol(SYMBOL, days=DAYS, params=p, verbose=False)
        except Exception as e:
            return {"error": str(e), "trades": 0, "pnl": 0, "pf": 0}
    finally:
        # restore
        if orig_sym_q is None:
            config.SIGNAL_QUALITY_SYMBOL.pop(SYMBOL, None)
        else:
            config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = orig_sym_q
    if not r:
        return {"trades": 0, "pnl": 0, "pf": 0}
    return {
        "trades": r.get("trades", 0),
        "pnl": float(r.get("pnl", 0)),
        "pf": float(r.get("pf", 0)),
        "wr": float(r.get("wr", 0)),
        "dd": float(r.get("dd", 0)),
        "avg_r": float(r.get("avg_r", 0)),
    }


def best_of(results, min_trades=10, fallback_min_trades=5):
    """Pick highest-pnl result with trades >= min_trades. If none qualify,
    fall back to highest-pnl with trades >= fallback_min_trades."""
    qualified = [(v, r) for v, r in results if r.get("trades", 0) >= min_trades]
    if not qualified:
        qualified = [(v, r) for v, r in results if r.get("trades", 0) >= fallback_min_trades]
    if not qualified:
        # final fallback: anything with trades > 0
        qualified = [(v, r) for v, r in results if r.get("trades", 0) > 0]
    if not qualified:
        return None, None
    qualified.sort(key=lambda x: -x[1]["pnl"])
    return qualified[0]


def main():
    global_t0 = time.time()
    print(f"\n=== BTCUSD entry-sweep ({DAYS}d) ===")
    print(f"Baseline: quality={BASELINE_QUALITY} retrace={BASELINE_RETRACE} wait={BASELINE_WAIT}")

    state = {
        "symbol": SYMBOL,
        "days": DAYS,
        "baseline": {"quality": BASELINE_QUALITY, "retrace": BASELINE_RETRACE, "wait": BASELINE_WAIT},
        "axis_traces": [],
    }

    # -------- Baseline --------
    print("\n[baseline] running...")
    t0 = time.time()
    base_r = run(BASELINE_RETRACE, BASELINE_WAIT, BASELINE_QUALITY)
    print(f"  baseline: pnl={base_r['pnl']:.2f} pf={base_r['pf']:.2f} trades={base_r['trades']} ({time.time()-t0:.1f}s)")
    state["baseline_result"] = base_r

    if "error" in base_r:
        print(f"  ERROR in baseline: {base_r['error']}")
        OUT_FILE.write_text(json.dumps(state, indent=2, default=str))
        return state

    cur_retrace = BASELINE_RETRACE
    cur_wait = BASELINE_WAIT
    cur_quality = dict(BASELINE_QUALITY)
    cur_raw = None  # tracks which RAW grid value is currently best (None = baseline mixed)

    TIME_BUDGET = 25 * 60  # 25 min hard cap

    # -------- Axis 1: MIN_SCORE (uniform raw -> quality dict) --------
    print("\n[axis1] MIN_SCORE (raw -> quality)")
    axis1 = {"param": "MIN_SCORE", "tested": [], "best_value": None}
    results1 = []
    for raw in RAW_SCORES:
        if time.time() - global_t0 > TIME_BUDGET:
            print(f"  TIME BUDGET EXCEEDED — aborting axis1")
            break
        q = raw_to_quality_dict(raw)
        t0 = time.time()
        r = run(cur_retrace, cur_wait, q)
        dt = time.time() - t0
        print(f"  raw={raw} (q={q['trending']:.1f}): pnl={r['pnl']:.2f} pf={r['pf']:.2f} trades={r['trades']} ({dt:.1f}s)")
        axis1["tested"].append({"value": raw, "pnl": r["pnl"], "pf": r["pf"], "trades": r["trades"]})
        results1.append((raw, r))

    # also test BASELINE_QUALITY as the "current" anchor — included via baseline_result
    best_val, best_r = best_of(results1)
    if best_val is not None and best_r["pnl"] > base_r["pnl"]:
        cur_raw = best_val
        cur_quality = raw_to_quality_dict(best_val)
        print(f"  -> AXIS1 winner: raw={best_val} pnl={best_r['pnl']:.2f} pf={best_r['pf']:.2f} (beats baseline {base_r['pnl']:.2f})")
        axis1["best_value"] = best_val
    else:
        print(f"  -> AXIS1 keeps BASELINE quality (none beat baseline pnl={base_r['pnl']:.2f})")
        # Encode baseline as "best_value" for axis trace; use avg of baseline (raw equiv)
        # baseline avg quality 31.75 -> raw 3.81; but to satisfy schema, use a real raw if any candidate had trades
        if best_val is not None:
            axis1["best_value"] = best_val  # the best of grid for record (but didn't beat baseline)
        else:
            axis1["best_value"] = RAW_SCORES[0]  # placeholder
    state["axis_traces"].append(axis1)

    # -------- Axis 2: PULLBACK_ATR_RETRACE --------
    print("\n[axis2] PULLBACK_ATR_RETRACE")
    axis2 = {"param": "PULLBACK_ATR_RETRACE", "tested": [], "best_value": None}
    results2 = []
    for ret in RETRACES:
        if time.time() - global_t0 > TIME_BUDGET:
            print(f"  TIME BUDGET EXCEEDED — aborting axis2")
            break
        t0 = time.time()
        r = run(ret, cur_wait, cur_quality)
        dt = time.time() - t0
        print(f"  retrace={ret}: pnl={r['pnl']:.2f} pf={r['pf']:.2f} trades={r['trades']} ({dt:.1f}s)")
        axis2["tested"].append({"value": ret, "pnl": r["pnl"], "pf": r["pf"], "trades": r["trades"]})
        results2.append((ret, r))

    best_val2, best_r2 = best_of(results2)
    if best_val2 is not None:
        cur_retrace = best_val2
        axis2["best_value"] = best_val2
        print(f"  -> AXIS2 winner: retrace={best_val2} pnl={best_r2['pnl']:.2f}")
    else:
        axis2["best_value"] = BASELINE_RETRACE
    state["axis_traces"].append(axis2)

    # -------- Axis 3: PULLBACK_MAX_WAIT_BARS --------
    print("\n[axis3] PULLBACK_MAX_WAIT_BARS")
    axis3 = {"param": "PULLBACK_MAX_WAIT_BARS", "tested": [], "best_value": None}
    results3 = []
    for w in WAITS:
        if time.time() - global_t0 > TIME_BUDGET:
            print(f"  TIME BUDGET EXCEEDED — aborting axis3")
            break
        t0 = time.time()
        r = run(cur_retrace, w, cur_quality)
        dt = time.time() - t0
        print(f"  wait={w}: pnl={r['pnl']:.2f} pf={r['pf']:.2f} trades={r['trades']} ({dt:.1f}s)")
        axis3["tested"].append({"value": w, "pnl": r["pnl"], "pf": r["pf"], "trades": r["trades"]})
        results3.append((w, r))

    best_val3, best_r3 = best_of(results3)
    if best_val3 is not None:
        cur_wait = best_val3
        axis3["best_value"] = best_val3
        print(f"  -> AXIS3 winner: wait={best_val3} pnl={best_r3['pnl']:.2f}")
    else:
        axis3["best_value"] = BASELINE_WAIT
    state["axis_traces"].append(axis3)

    # -------- Confirm winner combo --------
    print(f"\n[confirm] retrace={cur_retrace} wait={cur_wait} quality={cur_quality}")
    t0 = time.time()
    winner_r = run(cur_retrace, cur_wait, cur_quality)
    print(f"  winner: pnl={winner_r['pnl']:.2f} pf={winner_r['pf']:.2f} trades={winner_r['trades']} ({time.time()-t0:.1f}s)")

    state["winner"] = {
        "retrace": cur_retrace,
        "wait": cur_wait,
        "quality": cur_quality,
        "raw_min_score": cur_raw,
        "result": winner_r,
    }
    state["lift_pnl"] = winner_r["pnl"] - base_r["pnl"]
    state["elapsed_s"] = round(time.time() - global_t0, 1)

    OUT_FILE.write_text(json.dumps(state, indent=2, default=str))
    print(f"\nSaved: {OUT_FILE}")
    print(f"Lift: ${state['lift_pnl']:.2f}  Total: {state['elapsed_s']}s")
    return state


if __name__ == "__main__":
    main()
