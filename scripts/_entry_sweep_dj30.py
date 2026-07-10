#!/usr/bin/env python3 -B
"""
Coordinate-descent entry-param sweep for DJ30.r (90d).

Sweeps in order: MIN_SCORE (as quality threshold) -> PULLBACK_ATR_RETRACE ->
PULLBACK_MAX_WAIT_BARS. VWAP_TOO_FAR is skipped (no BT implementation per recon).

Writes results JSON to backtest/results/entry_sweep_20260614/DJ30.r.json
"""
import json
import sys
import time
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

OUT_DIR = ROOT / "backtest" / "results" / "entry_sweep_20260614"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "DJ30.r"
DAYS = 90

# Convert raw-score MIN_SCORE grid to quality (raw/12*100). Round to int.
MIN_SCORE_GRID = [5.5, 6.0, 6.5, 7.0, 7.5]
QUALITY_GRID = [int(round(s / 12 * 100)) for s in MIN_SCORE_GRID]  # [46,50,54,58,62]

PULLBACK_RETRACE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
PULLBACK_WAIT_GRID = [1, 2, 3]


def _load_meta():
    try:
        from models.signal_model import SignalModel
        mm = SignalModel()
        mm.load(SYMBOL)
        if mm.has_model(SYMBOL):
            return mm
    except Exception:
        pass
    return None


_MM = _load_meta()


def run_bt(quality_uniform=None, pullback_retrace=None, pullback_wait=None):
    """
    Run a single BT with the requested overrides.

    quality_uniform: int or None. If int, monkey-patch config.SIGNAL_QUALITY_SYMBOL[SYMBOL]
                     to a uniform per-regime dict so the lookup at v5_backtest.py:537 uses it.
                     If None, restore the original per-sym auto_tuned values.
    pullback_retrace / pullback_wait: pass-through into params dict.
    """
    # Backup + patch SIGNAL_QUALITY_SYMBOL for this symbol
    orig = config.SIGNAL_QUALITY_SYMBOL.get(SYMBOL, None)
    try:
        if quality_uniform is not None:
            config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
                "trending": quality_uniform,
                "ranging": quality_uniform,
                "volatile": quality_uniform,
                "low_vol": quality_uniform,
            }
        # else: leave whatever auto_tuned merged in
        p = copy.deepcopy(DEFAULT_PARAMS)
        if pullback_retrace is not None:
            p["pullback_atr_retrace"] = float(pullback_retrace)
        if pullback_wait is not None:
            p["pullback_max_wait"] = int(pullback_wait)
        if _MM is not None:
            p["_meta_model"] = _MM
        try:
            return backtest_symbol(SYMBOL, days=DAYS, params=p, verbose=False)
        except Exception as e:
            return {"error": str(e)}
    finally:
        # Restore
        if orig is None:
            config.SIGNAL_QUALITY_SYMBOL.pop(SYMBOL, None)
        else:
            config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = orig


def short_r(r):
    if not r or "error" in (r or {}):
        return {"trades": 0, "pf": 0.0, "pnl": 0.0, "wr": 0.0, "dd": 0.0, "error": (r or {}).get("error") if r else "no-result"}
    return {
        "trades": int(r.get("trades", 0)),
        "pf": float(r.get("pf", 0.0)),
        "pnl": float(r.get("pnl", 0.0)),
        "wr": float(r.get("wr", 0.0)),
        "dd": float(r.get("dd", 0.0)),
    }


def pick_best(tested, baseline_pnl):
    """Pick the value with highest pnl among those with trades >= 10. Skip <5 trades."""
    eligible = [t for t in tested if t["trades"] >= 10]
    if not eligible:
        return None
    eligible.sort(key=lambda t: -t["pnl"])
    return eligible[0]


def main():
    t0 = time.time()
    state = {
        "symbol": SYMBOL,
        "days": DAYS,
        "min_score_grid": MIN_SCORE_GRID,
        "quality_grid": QUALITY_GRID,
        "pullback_retrace_grid": PULLBACK_RETRACE_GRID,
        "pullback_wait_grid": PULLBACK_WAIT_GRID,
        "vwap_too_far_skipped": True,
        "vwap_skip_reason": "param does not exist in BT (recon)",
        "axes": [],
    }

    # ── BASELINE ──
    print(f"[{time.time()-t0:6.1f}s] Baseline DJ30.r 90d (no overrides) ...", flush=True)
    base = run_bt()
    base_short = short_r(base)
    state["baseline"] = base_short
    print(f"           baseline: pnl={base_short['pnl']:.2f} pf={base_short['pf']:.2f} n={base_short['trades']}", flush=True)
    baseline_pnl = base_short["pnl"]
    baseline_pf = base_short["pf"]
    baseline_trades = base_short["trades"]

    # Track current-best params
    cur = {"min_quality": None, "pullback_retrace": None, "pullback_wait": None}
    cur_pnl = baseline_pnl

    # ── AXIS 1: MIN_SCORE (as quality threshold) ──
    print(f"\n[{time.time()-t0:6.1f}s] AXIS MIN_SCORE: {MIN_SCORE_GRID} -> qual {QUALITY_GRID}", flush=True)
    axis1 = {"param": "MIN_SCORE", "tested": [], "best_value": None}
    for ms, q in zip(MIN_SCORE_GRID, QUALITY_GRID):
        r = run_bt(quality_uniform=q)
        rs = short_r(r)
        axis1["tested"].append({
            "value": float(ms),
            "quality": int(q),
            "pnl": rs["pnl"], "pf": rs["pf"], "trades": rs["trades"],
        })
        print(f"           MIN_SCORE={ms} (q={q}): pnl={rs['pnl']:.2f} pf={rs['pf']:.2f} n={rs['trades']}", flush=True)
    # Decide axis-1 winner: best pnl among trades>=10
    eligible1 = [t for t in axis1["tested"] if t["trades"] >= 10]
    if eligible1:
        best1 = max(eligible1, key=lambda t: t["pnl"])
        if best1["pnl"] > cur_pnl:
            cur["min_quality"] = int(best1["quality"])
            cur_pnl = best1["pnl"]
            axis1["best_value"] = best1["value"]
            print(f"           >> ship MIN_SCORE={best1['value']} (q={best1['quality']}) pnl={best1['pnl']:.2f}", flush=True)
        else:
            axis1["best_value"] = None
            print(f"           >> no MIN_SCORE beats baseline ({baseline_pnl:.2f}); keep default", flush=True)
    else:
        axis1["best_value"] = None
        print("           >> no eligible MIN_SCORE candidates (trades<10)", flush=True)
    state["axes"].append(axis1)

    # ── AXIS 2: PULLBACK_ATR_RETRACE ──
    print(f"\n[{time.time()-t0:6.1f}s] AXIS PULLBACK_ATR_RETRACE: {PULLBACK_RETRACE_GRID} (with q={cur['min_quality']})", flush=True)
    axis2 = {"param": "PULLBACK_ATR_RETRACE", "tested": [], "best_value": None}
    for rt in PULLBACK_RETRACE_GRID:
        r = run_bt(quality_uniform=cur["min_quality"], pullback_retrace=rt,
                   pullback_wait=cur["pullback_wait"])
        rs = short_r(r)
        axis2["tested"].append({
            "value": float(rt),
            "pnl": rs["pnl"], "pf": rs["pf"], "trades": rs["trades"],
        })
        print(f"           retrace={rt}: pnl={rs['pnl']:.2f} pf={rs['pf']:.2f} n={rs['trades']}", flush=True)
    eligible2 = [t for t in axis2["tested"] if t["trades"] >= 10]
    if eligible2:
        best2 = max(eligible2, key=lambda t: t["pnl"])
        if best2["pnl"] > cur_pnl:
            cur["pullback_retrace"] = float(best2["value"])
            cur_pnl = best2["pnl"]
            axis2["best_value"] = best2["value"]
            print(f"           >> ship retrace={best2['value']} pnl={best2['pnl']:.2f}", flush=True)
        else:
            axis2["best_value"] = None
            print(f"           >> no retrace beats cur_pnl ({cur_pnl:.2f}); keep default", flush=True)
    else:
        axis2["best_value"] = None
        print("           >> no eligible retrace candidates", flush=True)
    state["axes"].append(axis2)

    # ── AXIS 3: PULLBACK_MAX_WAIT_BARS ──
    print(f"\n[{time.time()-t0:6.1f}s] AXIS PULLBACK_MAX_WAIT_BARS: {PULLBACK_WAIT_GRID} (retrace={cur['pullback_retrace']}, q={cur['min_quality']})", flush=True)
    axis3 = {"param": "PULLBACK_MAX_WAIT_BARS", "tested": [], "best_value": None}
    for w in PULLBACK_WAIT_GRID:
        r = run_bt(quality_uniform=cur["min_quality"],
                   pullback_retrace=cur["pullback_retrace"],
                   pullback_wait=w)
        rs = short_r(r)
        axis3["tested"].append({
            "value": int(w),
            "pnl": rs["pnl"], "pf": rs["pf"], "trades": rs["trades"],
        })
        print(f"           wait={w}: pnl={rs['pnl']:.2f} pf={rs['pf']:.2f} n={rs['trades']}", flush=True)
    eligible3 = [t for t in axis3["tested"] if t["trades"] >= 10]
    if eligible3:
        best3 = max(eligible3, key=lambda t: t["pnl"])
        if best3["pnl"] > cur_pnl:
            cur["pullback_wait"] = int(best3["value"])
            cur_pnl = best3["pnl"]
            axis3["best_value"] = best3["value"]
            print(f"           >> ship wait={best3['value']} pnl={best3['pnl']:.2f}", flush=True)
        else:
            axis3["best_value"] = None
            print(f"           >> no wait beats cur_pnl ({cur_pnl:.2f}); keep default", flush=True)
    else:
        axis3["best_value"] = None
        print("           >> no eligible wait candidates", flush=True)
    state["axes"].append(axis3)

    # ── AXIS 4 (VWAP_TOO_FAR) is skipped ──
    state["axes"].append({"param": "VWAP_TOO_FAR", "tested": [], "best_value": None,
                          "skipped_reason": "no BT implementation"})

    # ── FINAL CONFIRM ──
    print(f"\n[{time.time()-t0:6.1f}s] FINAL combo: q={cur['min_quality']} retrace={cur['pullback_retrace']} wait={cur['pullback_wait']}", flush=True)
    winner = run_bt(quality_uniform=cur["min_quality"],
                    pullback_retrace=cur["pullback_retrace"],
                    pullback_wait=cur["pullback_wait"])
    winner_short = short_r(winner)
    state["winner_combo"] = {
        "min_quality_uniform": cur["min_quality"],
        "pullback_retrace": cur["pullback_retrace"],
        "pullback_wait": cur["pullback_wait"],
    }
    state["winner"] = winner_short
    print(f"           winner: pnl={winner_short['pnl']:.2f} pf={winner_short['pf']:.2f} n={winner_short['trades']}", flush=True)

    # ── DECISION ──
    lift_pnl = winner_short["pnl"] - baseline_pnl
    if (lift_pnl > 30 and winner_short["trades"] >= 15 and
            winner_short["pf"] >= max(1.2, baseline_pf - 0.1)):
        decision = "SHIP"
    elif 5 < lift_pnl <= 30 or (abs(winner_short["trades"] - baseline_trades) <= 5 and lift_pnl > 5):
        decision = "MARGINAL"
    else:
        decision = "NULL"
    state["lift_pnl"] = round(lift_pnl, 2)
    state["decision"] = decision
    state["elapsed_s"] = round(time.time() - t0, 1)
    print(f"\n           DECISION: {decision}  lift=${lift_pnl:.2f}  elapsed={state['elapsed_s']}s", flush=True)

    out = OUT_DIR / f"{SYMBOL}.json"
    with open(out, "w") as f:
        json.dump(state, f, indent=2)
    print(f"           written: {out}", flush=True)
    return state


if __name__ == "__main__":
    main()
