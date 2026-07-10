#!/usr/bin/env python3 -B
"""Coordinate-descent entry-param sweep for JPN225ft, 90d.

Axes: MIN_SCORE (via SIGNAL_QUALITY_SYMBOL uniform per-regime),
      PULLBACK_ATR_RETRACE, PULLBACK_MAX_WAIT_BARS.

VWAP_TOO_FAR skipped — not implemented in BT.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

SYMBOL = "JPN225ft"
DAYS = 90
OUT = ROOT / "backtest" / "results" / "entry_sweep_20260614" / f"{SYMBOL}.json"

# Preload meta model (mirrors tune_pullback.py behaviour)
META = None
try:
    from models.signal_model import SignalModel
    _m = SignalModel(); _m.load(SYMBOL)
    if _m.has_model(SYMBOL):
        META = _m
except Exception:
    META = None


def run(quality_thresh=None, pullback_retrace=None, pullback_wait=None, label=""):
    """Single BT with given overrides.

    quality_thresh: int — uniform per-regime threshold to write into config.SIGNAL_QUALITY_SYMBOL[SYMBOL]
    pullback_retrace: float — override
    pullback_wait: int — override
    """
    # Reset and patch the per-symbol quality dict (config-level — read at BT call time)
    orig_q = dict(config.SIGNAL_QUALITY_SYMBOL.get(SYMBOL, {}))
    if quality_thresh is not None:
        config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
            "trending": int(quality_thresh),
            "ranging": int(quality_thresh),
            "volatile": int(quality_thresh),
            "low_vol": int(quality_thresh),
        }

    p = {**DEFAULT_PARAMS, "audit_fix_gates": True}
    if pullback_retrace is not None:
        p["pullback_atr_retrace"] = float(pullback_retrace)
    if pullback_wait is not None:
        p["pullback_max_wait"] = int(pullback_wait)
    if META is not None:
        p["_meta_model"] = META

    try:
        r = backtest_symbol(SYMBOL, days=DAYS, params=p, verbose=False)
    except Exception as e:
        r = {"error": str(e)}
    finally:
        config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = orig_q

    if not r or "error" in r:
        return {"label": label, "trades": 0, "pf": 0, "pnl": 0, "error": r.get("error") if r else "none"}
    return {
        "label": label,
        "trades": int(r.get("trades", 0)),
        "pf": round(float(r.get("pf", 0) or 0), 3),
        "pnl": round(float(r.get("pnl", 0) or 0), 2),
        "wr": round(float(r.get("wr", 0) or 0), 1),
        "dd": round(float(r.get("dd", 0) or 0), 2),
    }


def best_of(records, min_trades=10):
    """Pick highest pnl among records with trades >= min_trades."""
    elig = [r for r in records if r["trades"] >= min_trades]
    if not elig:
        # fallback: less-strict floor of 5 trades
        elig = [r for r in records if r["trades"] >= 5]
    if not elig:
        return None
    return max(elig, key=lambda r: r["pnl"])


def main():
    t0 = time.time()
    print(f"=== ENTRY SWEEP {SYMBOL} ({DAYS}d) ===")

    # Current per-sym defaults:
    # quality: uniform 35 per regime (auto_tuned.py:158)
    # pullback retrace: 0.2 (no entry in PULLBACK_CONFIG_PER_SYMBOL — global default)
    # pullback wait: 1 (global default)
    cur_quality = 35
    cur_retrace = 0.20
    cur_wait = 1

    # ---- BASELINE ----
    baseline = run(quality_thresh=cur_quality, pullback_retrace=cur_retrace, pullback_wait=cur_wait, label="baseline")
    print(f"BASELINE: q={cur_quality} retrace={cur_retrace} wait={cur_wait} -> "
          f"trades={baseline['trades']} pf={baseline['pf']} pnl=${baseline['pnl']}")

    axis_traces = []

    # ---- AXIS 1: MIN_SCORE (= per-regime quality threshold) ----
    # Recon grid: [5.5, 6.0, 6.5, 7.0, 7.5] raw → quality 45.8, 50, 54.2, 58.3, 62.5
    # We'll test on the 0-100 quality scale. Current 35 — add lower options to sweep too.
    score_grid = [30, 35, 40, 45, 50, 55, 60]
    score_records = []
    for q in score_grid:
        r = run(quality_thresh=q, pullback_retrace=cur_retrace, pullback_wait=cur_wait, label=f"q={q}")
        r["value"] = q
        score_records.append(r)
        print(f"  Q={q:>2}: trades={r['trades']:>3} pf={r['pf']:.2f} pnl=${r['pnl']:>8.2f} wr={r['wr']}%")
    best_q = best_of(score_records, min_trades=10)
    if best_q:
        cur_quality = int(best_q["value"])
        print(f"  >> BEST quality: {cur_quality} (pnl ${best_q['pnl']})")
    axis_traces.append({
        "param": "MIN_SCORE",
        "tested": [{"value": float(r["value"]), "pnl": float(r["pnl"]), "pf": float(r["pf"]), "trades": int(r["trades"])} for r in score_records],
        "best_value": float(cur_quality),
    })

    # ---- AXIS 2: PULLBACK_ATR_RETRACE ----
    retrace_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    retrace_records = []
    for rt in retrace_grid:
        r = run(quality_thresh=cur_quality, pullback_retrace=rt, pullback_wait=cur_wait, label=f"rt={rt}")
        r["value"] = rt
        retrace_records.append(r)
        print(f"  RT={rt:.2f}: trades={r['trades']:>3} pf={r['pf']:.2f} pnl=${r['pnl']:>8.2f}")
    best_rt = best_of(retrace_records, min_trades=10)
    if best_rt:
        cur_retrace = float(best_rt["value"])
        print(f"  >> BEST retrace: {cur_retrace} (pnl ${best_rt['pnl']})")
    axis_traces.append({
        "param": "PULLBACK_ATR_RETRACE",
        "tested": [{"value": float(r["value"]), "pnl": float(r["pnl"]), "pf": float(r["pf"]), "trades": int(r["trades"])} for r in retrace_records],
        "best_value": float(cur_retrace),
    })

    # ---- AXIS 3: PULLBACK_MAX_WAIT_BARS ----
    wait_grid = [1, 2, 3]
    wait_records = []
    for w in wait_grid:
        r = run(quality_thresh=cur_quality, pullback_retrace=cur_retrace, pullback_wait=w, label=f"w={w}")
        r["value"] = w
        wait_records.append(r)
        print(f"  W={w}: trades={r['trades']:>3} pf={r['pf']:.2f} pnl=${r['pnl']:>8.2f}")
    best_w = best_of(wait_records, min_trades=10)
    if best_w:
        cur_wait = int(best_w["value"])
        print(f"  >> BEST wait: {cur_wait} (pnl ${best_w['pnl']})")
    axis_traces.append({
        "param": "PULLBACK_MAX_WAIT_BARS",
        "tested": [{"value": float(r["value"]), "pnl": float(r["pnl"]), "pf": float(r["pf"]), "trades": int(r["trades"])} for r in wait_records],
        "best_value": float(cur_wait),
    })

    # ---- FINAL CONFIRM ----
    winner = run(quality_thresh=cur_quality, pullback_retrace=cur_retrace, pullback_wait=cur_wait, label="winner")
    print(f"\nWINNER: q={cur_quality} retrace={cur_retrace} wait={cur_wait} -> "
          f"trades={winner['trades']} pf={winner['pf']} pnl=${winner['pnl']}")

    # ---- DECISION ----
    lift = float(winner["pnl"]) - float(baseline["pnl"])
    decision = "NULL"
    reason = ""
    if lift > 30 and winner["trades"] >= 15 and winner["pf"] >= max(1.2, float(baseline["pf"]) - 0.1):
        decision = "SHIP"
        reason = f"lift ${lift:.2f} clears $30 + trades {winner['trades']} >=15 + PF {winner['pf']:.2f}"
    elif lift > 5 and winner["trades"] >= 10:
        decision = "MARGINAL"
        reason = f"lift ${lift:.2f} in (5,30] or trades close to baseline"
    else:
        decision = "NULL"
        reason = f"lift ${lift:.2f} <= $5 or trades {winner['trades']} <10 or PF guard failed"

    out = {
        "symbol": SYMBOL,
        "days": DAYS,
        "elapsed_s": round(time.time() - t0, 1),
        "baseline": baseline,
        "winner": winner,
        "winner_params": {
            "MIN_SCORE": cur_quality,
            "PULLBACK_ATR_RETRACE": cur_retrace,
            "PULLBACK_MAX_WAIT_BARS": cur_wait,
        },
        "axis_traces": axis_traces,
        "lift_pnl": round(lift, 2),
        "decision": decision,
        "decision_reason": reason,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {OUT}")
    print(f"DECISION: {decision} — {reason}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
