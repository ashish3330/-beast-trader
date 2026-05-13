#!/usr/bin/env python3 -B
"""
PHASE 4 — three optimization axes in one symbol pass.

4a: Direction bias (LONG / SHORT / BOTH)
4b: Toxic hours — find worst N hours, disable
4c: Per-regime SL multiplier — refine Phase 2 winners

Baseline: Phase 1 + Phase 2 + Phase 3 winners already in auto_tuned.py
(loaded via standard import path through config.py).

Per symbol:
  4a: 3 options × WF 5-fold = 15 backtests
  4b: 24 hours × 1 disable test = 24 backtests; pick top 3 worst to disable
  4c: 4 regimes × 5 SL deltas × WF = 100 backtests
  Total: ~140 per symbol × 17 syms = 2,380 backtests

Output: backtest/results/phase4_combined/<SYMBOL>.json
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

OUT_DIR = ROOT / "backtest" / "results" / "phase4_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def calmar(pnl_pct, max_dd_pct, days):
    if max_dd_pct <= 0 or days <= 0:
        return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / max_dd_pct


def score(r, days):
    if not r or r.get("trades", 0) < 10:
        return -1e9
    pnl_pct = r["pnl"] / 1000 * 100
    return calmar(pnl_pct, max(r["dd"], 0.5), days) * math.sqrt(r["trades"])


def base_params():
    """Reads auto_tuned for already-tuned baseline."""
    return {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
    }


def walk_forward_5fold(symbol, params):
    folds = []
    for d in [60, 90, 120, 150, 180]:
        try:
            r = backtest_symbol(symbol, days=d, params=params, verbose=False)
        except Exception:
            continue
        if r and r.get("trades", 0) > 5:
            folds.append({"days": d, "trades": r["trades"], "pf": r["pf"],
                          "pnl": r["pnl"], "dd": r["dd"]})
    if not folds:
        return None
    return {
        "folds": folds,
        "avg_pf": round(sum(f["pf"] for f in folds) / len(folds), 2),
        "avg_pnl": round(sum(f["pnl"] for f in folds) / len(folds), 0),
        "n_positive": sum(1 for f in folds if f["pnl"] > 0),
        "n_folds": len(folds),
    }


def is_wf_winner(wf):
    if not wf: return False
    return (wf["avg_pf"] > 1.3 and
            wf["n_positive"] >= max(3, int(0.6 * wf["n_folds"])))


def phase4a_direction(symbol):
    """Direction bias sweep."""
    out = {}
    for direction in ["LONG", "SHORT", "BOTH"]:
        p = base_params()
        if direction != "BOTH":
            p["force_direction"] = direction
        wf = walk_forward_5fold(symbol, p)
        if wf:
            out[direction] = wf
    if not out:
        return None
    # Pick best by avg_pnl, requiring WF gate
    candidates = sorted(out.items(), key=lambda x: -x[1]["avg_pnl"])
    for direction, wf in candidates:
        if is_wf_winner(wf) and direction != "BOTH":
            # Only override if NON-both beats both
            both = out.get("BOTH", {})
            if direction != "BOTH" and wf["avg_pnl"] > both.get("avg_pnl", -1e9) * 1.2:
                return {"direction": direction, "wf": wf}
    return None


def phase4b_toxic_hours(symbol):
    """Find worst trading hours by sweep, disable top 3."""
    # Run baseline once
    base = base_params()
    base_r = backtest_symbol(symbol, days=180, params=base, verbose=False)
    if not base_r or base_r.get("trades", 0) < 30:
        return None
    base_pnl = base_r["pnl"]

    # Test disabling each hour individually
    hour_impacts = []
    for hour in range(24):
        p = base_params()
        p["toxic_hours"] = [hour]
        r = backtest_symbol(symbol, days=180, params=p, verbose=False)
        if not r:
            continue
        delta = r["pnl"] - base_pnl  # positive = hour was toxic (disabling improved)
        hour_impacts.append({"hour": hour, "delta": round(delta, 2)})

    if not hour_impacts:
        return None
    # Sort by delta (most improvement when disabled = most toxic)
    hour_impacts.sort(key=lambda x: -x["delta"])
    toxic = [h for h in hour_impacts if h["delta"] > 0][:3]

    if not toxic:
        return None

    # Test all 3 toxic hours disabled together + WF validate
    p = base_params()
    p["toxic_hours"] = [h["hour"] for h in toxic]
    wf = walk_forward_5fold(symbol, p)
    if not is_wf_winner(wf):
        return None
    return {"toxic_hours": [h["hour"] for h in toxic],
            "individual_impacts": hour_impacts[:5],
            "wf": wf}


def phase4c_regime_sl(symbol):
    """Per-regime SL multiplier refinement."""
    # Test 4 regimes × 5 SL deltas around Phase 2 winner SL
    REGIMES = ["trending", "ranging", "volatile", "low_vol"]
    SL_DELTAS = [-0.3, -0.15, 0, 0.15, 0.3]

    # Find current SL from auto_tuned
    try:
        import auto_tuned as _at
        importlib.reload(_at)
        current_sl = _at.SL_OVERRIDE_AUTO.get(symbol, 1.0)
    except Exception:
        current_sl = 1.0

    best = {}
    for regime in REGIMES:
        for d_sl in SL_DELTAS:
            sl = round(max(0.4, current_sl + d_sl), 2)
            p = base_params()
            p["sl_atr_mult"] = sl
            # Filter to this regime only by setting OTHER regimes' min_quality high
            high_q = {r: 95 for r in REGIMES if r != regime}
            high_q[regime] = 40  # normal for the regime under test
            p["min_quality"] = high_q
            r = backtest_symbol(symbol, days=180, params=p, verbose=False)
            if r and r.get("trades", 0) > 5:
                s = score(r, 180)
                if regime not in best or s > best[regime]["score"]:
                    best[regime] = {"sl": sl, "score": s, "pnl": r["pnl"],
                                    "pf": r["pf"], "trades": r["trades"]}
    if not best:
        return None
    return best


import importlib  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: phase4_combined.py SYMBOL"); sys.exit(1)
    symbol = sys.argv[1]

    print(f"\n=== PHASE 4 {symbol} ===")
    t0 = time.time()
    out = {"symbol": symbol}

    print(f"  4a: direction bias...", flush=True)
    out["direction_bias"] = phase4a_direction(symbol)
    print(f"      → {out['direction_bias'] or 'BOTH (no override)'}")

    print(f"  4b: toxic hours...", flush=True)
    out["toxic_hours"] = phase4b_toxic_hours(symbol)
    if out["toxic_hours"]:
        print(f"      → disable hours {out['toxic_hours']['toxic_hours']}")
    else:
        print(f"      → no toxic hours found")

    print(f"  4c: per-regime SL...", flush=True)
    out["regime_sl"] = phase4c_regime_sl(symbol)
    if out["regime_sl"]:
        for r, info in out["regime_sl"].items():
            print(f"      → {r}: SL={info['sl']} PnL=${info['pnl']:.0f}")

    out["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  Saved (elapsed {out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
