#!/usr/bin/env python3 -B
"""
PHASE 3a — quality threshold × Kelly sizing tune.

For each symbol:
  - Sweep per-regime quality thresholds (5 deltas around default)
  - Sweep risk_pct (Kelly-style sizing) — 7 levels
  - Apply Phase 1 + Phase 2 best SL/TP/trail as baseline
  - 5-fold walk-forward validate top 3

Grid: 5 quality deltas × 7 risk levels = 35 combos per symbol.
Tiny grid but high-leverage parameters.

Output: backtest/results/phase3a_quality_kelly/<SYMBOL>.json
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

PHASE1_DIR = ROOT / "backtest" / "results" / "full_tune_20260513"
PHASE2_DIR = ROOT / "backtest" / "results" / "phase2_hard_tune"
OUT_DIR = ROOT / "backtest" / "results" / "phase3a_quality_kelly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_DELTAS = [-10, -5, 0, +5, +10]   # applied to each regime threshold
RISK_PCT_VALUES = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]

DEFAULT_QUALITY = {"trending": 40, "ranging": 42, "volatile": 45, "low_vol": 40}


def calmar(pnl_pct, max_dd_pct, days):
    if max_dd_pct <= 0 or days <= 0:
        return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / max_dd_pct


def score(r, days):
    if not r or r.get("trades", 0) < 15:
        return -1e9
    pnl_pct = r["pnl"] / 1000 * 100
    dd = max(r["dd"], 0.5)
    return calmar(pnl_pct, dd, days) * math.sqrt(r["trades"])


def get_baseline_params(symbol):
    """Pull SL/TP from Phase 2 winner if present, else Phase 1."""
    p2_file = PHASE2_DIR / f"{symbol}.json"
    if p2_file.exists():
        d = json.loads(p2_file.read_text())
        w = d.get("phase2_winner")
        if w:
            return {"sl": w["sl"], "tp_r": w["tp_r"], "source": "phase2"}
    p1_file = PHASE1_DIR / f"{symbol}.json"
    if p1_file.exists():
        d = json.loads(p1_file.read_text())
        b = d.get("best")
        if b:
            return {"sl": b["sl"], "tp_r": b["tp_r"], "source": "phase1"}
    return {"sl": 1.0, "tp_r": [1.5, 2.5, 4.0], "source": "default"}


def run_combo(symbol, days, base, quality, risk_pct):
    p = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
        "sl_atr_mult": base["sl"],
        "sub_tp_r": base["tp_r"],
        "min_quality": quality,
        "risk_pct": risk_pct,
    }
    try:
        return backtest_symbol(symbol, days=days, params=p, verbose=False)
    except Exception:
        return None


def walk_forward_5fold(symbol, base, quality, risk_pct):
    folds = []
    for d in [60, 90, 120, 150, 180]:
        r = run_combo(symbol, d, base, quality, risk_pct)
        if r and r.get("trades", 0) > 5:
            folds.append({
                "days": d, "trades": r["trades"], "pf": r["pf"],
                "pnl": r["pnl"], "dd": r["dd"],
            })
    return folds


def main():
    if len(sys.argv) < 2:
        print("Usage: phase3a_quality_kelly.py SYMBOL"); sys.exit(1)
    symbol = sys.argv[1]
    days = 180
    base = get_baseline_params(symbol)
    print(f"\n=== PHASE 3a {symbol} (baseline from {base['source']}: SL={base['sl']}) ===")

    results = []
    n = len(QUALITY_DELTAS) * len(RISK_PCT_VALUES)
    print(f"  Grid: {n} combos")
    t0 = time.time()

    for d_q in QUALITY_DELTAS:
        quality = {k: v + d_q for k, v in DEFAULT_QUALITY.items()}
        for rp in RISK_PCT_VALUES:
            r = run_combo(symbol, days, base, quality, rp)
            if r and r.get("trades", 0) > 0:
                results.append({
                    "q_delta": d_q,
                    "quality": quality,
                    "risk_pct": rp,
                    "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
                    "pnl": r["pnl"], "dd": r["dd"],
                    "score": round(score(r, days), 1),
                })

    if not results:
        (OUT_DIR / f"{symbol}.json").write_text(json.dumps({"symbol": symbol, "winner": None}))
        print(f"  NO RESULTS"); return

    results.sort(key=lambda x: -x["score"])
    top3 = results[:3]
    print(f"  TOP 3:")
    for i, c in enumerate(top3):
        print(f"    #{i+1}: q={c['q_delta']:+d} risk={c['risk_pct']}% "
              f"PnL=${c['pnl']:.0f} PF={c['pf']:.2f} n={c['trades']} DD={c['dd']:.1f}%")

    # Walk-forward top 3
    for c in top3:
        wf = walk_forward_5fold(symbol, base, c["quality"], c["risk_pct"])
        c["wf_folds"] = wf
        if wf:
            c["wf_avg_pf"] = round(sum(f["pf"] for f in wf) / len(wf), 2)
            c["wf_avg_pnl"] = round(sum(f["pnl"] for f in wf) / len(wf), 0)
            c["wf_n_positive"] = sum(1 for f in wf if f["pnl"] > 0)
            c["wf_n_folds"] = len(wf)

    # Pick winner: WF avg PF > 1.3 AND ≥3/5 positive folds
    winner = None
    for c in top3:
        if (c.get("wf_avg_pf", 0) > 1.3 and
                c.get("wf_n_positive", 0) >= max(3, int(0.6 * c.get("wf_n_folds", 5)))):
            winner = c
            break

    out = {
        "symbol": symbol, "days": days,
        "baseline": base,
        "elapsed_s": round(time.time() - t0, 1),
        "tested": n,
        "top3": top3,
        "winner": winner,
    }
    if winner:
        print(f"\n  WINNER: q={winner['q_delta']:+d} risk={winner['risk_pct']}% "
              f"WF avg PF={winner['wf_avg_pf']} folds {winner['wf_n_positive']}/{winner['wf_n_folds']}+")
    else:
        print(f"\n  no winner passed strict WF")

    (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
