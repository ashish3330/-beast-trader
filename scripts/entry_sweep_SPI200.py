#!/usr/bin/env python3 -B
"""
Coordinate-descent entry-param sweep for SPI200.r (2026-06-14).
Axes: MIN_SCORE (via min_quality) -> PULLBACK_ATR_RETRACE -> PULLBACK_MAX_WAIT_BARS.
VWAP_TOO_FAR skipped (no BT implementation).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as _cfg  # noqa: E402
from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

SYMBOL = "SPI200.r"
DAYS = 90
OUT = ROOT / "backtest" / "results" / "entry_sweep_20260614" / f"{SYMBOL}.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Live baseline per-regime quality for SPI200.r (auto_tuned.py SIGNAL_QUALITY_SYMBOL_AUTO)
BASELINE_Q = {"trending": 40, "ranging": 40, "volatile": 50, "low_vol": 40}
# Recon-spec grid: MIN_SCORE raw 5.5..7.5 -> quality raw/12*100
# 5.5 -> 45.83, 6.0 -> 50.0, 6.5 -> 54.17, 7.0 -> 58.33, 7.5 -> 62.5
MIN_SCORE_GRID = [5.5, 6.0, 6.5, 7.0, 7.5]
def raw_to_q(r):
    return round(r / 12.0 * 100, 2)

PB_RETRACE_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
PB_WAIT_GRID = [1, 2, 3]

# Live baseline per PULLBACK_CONFIG_PER_SYMBOL[SPI200.r]
BASELINE_PB_R = 0.0
BASELINE_PB_W = 1

# Live SL override (from auto_tuned SYMBOL_ATR_SL_OVERRIDE_AUTO)
LIVE_SL = _cfg.SYMBOL_ATR_SL_OVERRIDE.get(SYMBOL, DEFAULT_PARAMS["sl_atr_mult"])

# Per-axis result store
axis_traces = []


def run_bt(min_q_dict, pb_r, pb_w, label=""):
    params = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        # Stay close to live: no slippage (memory: live has no slippage)
        "with_slippage": False,
        "with_commission": False,
        "with_swap": False,
        "min_quality": dict(min_q_dict),
        "pullback_atr_retrace": float(pb_r),
        "pullback_max_wait": int(pb_w),
        "sl_atr_mult": float(LIVE_SL),
    }
    # Temporarily blank out config.SIGNAL_QUALITY_SYMBOL[SYMBOL] so v5_backtest's
    # per-sym overlay at line 538 does NOT re-replace our injected min_quality.
    save_q = _cfg.SIGNAL_QUALITY_SYMBOL.get(SYMBOL)
    _cfg.SIGNAL_QUALITY_SYMBOL[SYMBOL] = dict(min_q_dict)
    try:
        r = backtest_symbol(SYMBOL, days=DAYS, params=params, verbose=False)
    except Exception as e:
        print(f"  ERR {label}: {e}")
        r = None
    finally:
        if save_q is None:
            _cfg.SIGNAL_QUALITY_SYMBOL.pop(SYMBOL, None)
        else:
            _cfg.SIGNAL_QUALITY_SYMBOL[SYMBOL] = save_q
    return r


def fmt(r):
    if not r:
        return "FAIL"
    return f"trades={r.get('trades',0)} pf={r.get('pf',0):.2f} pnl=${r.get('pnl',0):.2f}"


def keep_or_skip(r, min_trades=10):
    if not r:
        return False
    return r.get("trades", 0) >= min_trades


def main():
    t0 = time.time()
    print(f"=== Entry sweep {SYMBOL} {DAYS}d ===")
    print(f"Baseline q={BASELINE_Q} pb_r={BASELINE_PB_R} pb_w={BASELINE_PB_W} sl={LIVE_SL}")

    # ---- Baseline ----
    bl = run_bt(BASELINE_Q, BASELINE_PB_R, BASELINE_PB_W, "BASELINE")
    print(f"BASELINE: {fmt(bl)}")
    if not bl:
        # cannot continue without baseline
        out = {
            "symbol": SYMBOL,
            "decision": "NULL",
            "reason": "baseline BT failed",
            "axis_traces": [],
        }
        OUT.write_text(json.dumps(out, indent=2))
        return out

    base_pnl = bl["pnl"]; base_pf = bl["pf"]; base_trades = bl["trades"]
    cur = {
        "q": dict(BASELINE_Q),
        "pb_r": BASELINE_PB_R,
        "pb_w": BASELINE_PB_W,
        "pnl": base_pnl,
        "pf": base_pf,
        "trades": base_trades,
    }

    # ---- Axis 1: MIN_SCORE (uniform-per-regime override) ----
    axis1 = {"param": "MIN_SCORE", "tested": [], "best_value": None}
    print("\n--- Axis 1: MIN_SCORE ---")
    best_pnl = cur["pnl"]; best_val = None
    for raw in MIN_SCORE_GRID:
        q_val = raw_to_q(raw)
        q_dict = {k: q_val for k in BASELINE_Q}
        r = run_bt(q_dict, cur["pb_r"], cur["pb_w"], f"min_score={raw}")
        print(f"  raw={raw} q={q_val} -> {fmt(r)}")
        rec = {
            "value": float(raw),
            "trades": int(r.get("trades", 0)) if r else 0,
            "pf": float(r.get("pf", 0.0)) if r else 0.0,
            "pnl": float(r.get("pnl", 0.0)) if r else 0.0,
        }
        axis1["tested"].append(rec)
        if r and r.get("trades", 0) >= 5 and r["pnl"] > best_pnl:
            best_pnl = r["pnl"]; best_val = raw
            cur_best_q = q_dict
            cur_best_r = r
    if best_val is not None:
        cur["q"] = cur_best_q
        cur["pnl"] = cur_best_r["pnl"]
        cur["pf"] = cur_best_r["pf"]
        cur["trades"] = cur_best_r["trades"]
        axis1["best_value"] = float(best_val)
        print(f"  >>> Axis1 winner raw={best_val} pnl=${best_pnl:.2f}")
    else:
        axis1["best_value"] = float(0.0)  # marker: no improvement
        print(f"  >>> Axis1: no improvement, keep baseline")
    axis_traces.append(axis1)

    # ---- Axis 2: PULLBACK_ATR_RETRACE ----
    axis2 = {"param": "PULLBACK_ATR_RETRACE", "tested": [], "best_value": None}
    print("\n--- Axis 2: PULLBACK_ATR_RETRACE ---")
    best_pnl = cur["pnl"]; best_val = None; cur_best_r = None
    for pb_r in PB_RETRACE_GRID:
        r = run_bt(cur["q"], pb_r, cur["pb_w"], f"pb_r={pb_r}")
        print(f"  pb_r={pb_r} -> {fmt(r)}")
        rec = {
            "value": float(pb_r),
            "trades": int(r.get("trades", 0)) if r else 0,
            "pf": float(r.get("pf", 0.0)) if r else 0.0,
            "pnl": float(r.get("pnl", 0.0)) if r else 0.0,
        }
        axis2["tested"].append(rec)
        if r and r.get("trades", 0) >= 5 and r["pnl"] > best_pnl:
            best_pnl = r["pnl"]; best_val = pb_r; cur_best_r = r
    if best_val is not None:
        cur["pb_r"] = best_val
        cur["pnl"] = cur_best_r["pnl"]
        cur["pf"] = cur_best_r["pf"]
        cur["trades"] = cur_best_r["trades"]
        axis2["best_value"] = float(best_val)
        print(f"  >>> Axis2 winner pb_r={best_val} pnl=${best_pnl:.2f}")
    else:
        axis2["best_value"] = float(cur["pb_r"])
        print(f"  >>> Axis2: no improvement, keep {cur['pb_r']}")
    axis_traces.append(axis2)

    # ---- Axis 3: PULLBACK_MAX_WAIT_BARS ----
    axis3 = {"param": "PULLBACK_MAX_WAIT_BARS", "tested": [], "best_value": None}
    print("\n--- Axis 3: PULLBACK_MAX_WAIT_BARS ---")
    best_pnl = cur["pnl"]; best_val = None; cur_best_r = None
    for pb_w in PB_WAIT_GRID:
        r = run_bt(cur["q"], cur["pb_r"], pb_w, f"pb_w={pb_w}")
        print(f"  pb_w={pb_w} -> {fmt(r)}")
        rec = {
            "value": float(pb_w),
            "trades": int(r.get("trades", 0)) if r else 0,
            "pf": float(r.get("pf", 0.0)) if r else 0.0,
            "pnl": float(r.get("pnl", 0.0)) if r else 0.0,
        }
        axis3["tested"].append(rec)
        if r and r.get("trades", 0) >= 5 and r["pnl"] > best_pnl:
            best_pnl = r["pnl"]; best_val = pb_w; cur_best_r = r
    if best_val is not None:
        cur["pb_w"] = best_val
        cur["pnl"] = cur_best_r["pnl"]
        cur["pf"] = cur_best_r["pf"]
        cur["trades"] = cur_best_r["trades"]
        axis3["best_value"] = float(best_val)
        print(f"  >>> Axis3 winner pb_w={best_val} pnl=${best_pnl:.2f}")
    else:
        axis3["best_value"] = float(cur["pb_w"])
        print(f"  >>> Axis3: no improvement, keep {cur['pb_w']}")
    axis_traces.append(axis3)

    # ---- Confirm final combo ----
    print("\n--- CONFIRM final combo ---")
    final_r = run_bt(cur["q"], cur["pb_r"], cur["pb_w"], "FINAL")
    print(f"FINAL: q={cur['q']} pb_r={cur['pb_r']} pb_w={cur['pb_w']} -> {fmt(final_r)}")

    win_pnl = final_r["pnl"] if final_r else cur["pnl"]
    win_pf = final_r["pf"] if final_r else cur["pf"]
    win_trades = final_r["trades"] if final_r else cur["trades"]
    lift = win_pnl - base_pnl

    # Decision logic
    if lift > 30 and win_trades >= 15 and win_pf >= max(1.2, base_pf - 0.1):
        decision = "SHIP"
        reason = (f"lift_pnl={lift:.2f} > $30, trades={win_trades} >= 15, "
                  f"pf={win_pf:.2f} >= max(1.2, {base_pf:.2f}-0.1)")
    elif lift > 5 or (win_trades > 0 and abs(win_trades - base_trades) <= max(2, base_trades*0.2)):
        decision = "MARGINAL"
        reason = f"lift_pnl={lift:.2f}, trades={win_trades} (baseline={base_trades})"
    else:
        decision = "NULL"
        reason = (f"lift_pnl={lift:.2f} <= $5 or trades={win_trades} < 10")

    # Convert winning q dict back to raw equivalent for schema compactness.
    # Use ranging value (uniform for sweep cases).
    qv = list(cur["q"].values())[0]
    raw_min = round(qv * 12 / 100, 2)

    out = {
        "symbol": SYMBOL,
        "days": DAYS,
        "elapsed_s": round(time.time() - t0, 1),
        "baseline": {"pnl": base_pnl, "pf": base_pf, "trades": base_trades,
                     "q": BASELINE_Q, "pb_r": BASELINE_PB_R, "pb_w": BASELINE_PB_W},
        "winner": {
            "pnl": win_pnl, "pf": win_pf, "trades": win_trades,
            "q_dict": cur["q"],
            "raw_min_score_equiv": raw_min,
            "pb_r": cur["pb_r"], "pb_w": cur["pb_w"],
        },
        "lift_pnl": lift,
        "decision": decision,
        "reason": reason,
        "axis_traces": axis_traces,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}")
    print(f"DECISION: {decision} ({reason})")
    return out


if __name__ == "__main__":
    main()
