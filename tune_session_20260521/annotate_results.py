#!/usr/bin/env python3 -B
"""Post-process metals_crypto_sl_trail.json to add recommendations + caveats."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "metals_crypto_sl_trail.json"
d = json.loads(OUT.read_text())

recommendations = {}
for sym_key, base in d["baselines"].items():
    if base.get("status") != "OK":
        recommendations[sym_key] = {
            "action": "KEEP_BASELINE",
            "reason": f"INSUFFICIENT_DATA: baseline trades {base.get('trades', 0)} < 20",
        }
        continue

    win = d["winners"].get(sym_key)
    base_pnl = base["pnl"]

    if not win:
        recommendations[sym_key] = {
            "action": "KEEP_BASELINE",
            "reason": f"No grid config beat baseline PnL ${base_pnl:.2f} by ≥$40 with WF gates (avg_pf ≥1.2, ≥3/5 folds positive).",
            "baseline_pnl": round(base_pnl, 2),
            "baseline_pf": round(base["pf"], 2),
            "baseline_n": base["trades"],
        }
        continue

    # Check for caveats on the winner
    folds = win["wf_folds"]
    distinct_pnls = len({round(f["pnl"], 2) for f in folds})
    distinct_n = len({f["n"] for f in folds})
    degraded_wf = distinct_pnls == 1 and distinct_n == 1
    # Stale edge = MOST RECENT (60d) fold is negative. Single intermediate
    # losing fold is acceptable noise.
    recent_neg = len(folds) >= 1 and folds[0]["pnl"] < 0
    cavs = []
    if degraded_wf:
        cavs.append("WF_DEGRADED: all 5 folds return identical results (data cache too short)")
    if recent_neg:
        neg_folds = [f"{f['days']}d=${f['pnl']:+.0f}" for f in folds if f["pnl"] < 0]
        cavs.append(f"RECENT_FOLD_NEGATIVE: {','.join(neg_folds)}")
    if win["best_pnl"] > 50000:
        cavs.append(f"COMPOUND_BLOWUP: PnL ${win['best_pnl']:+.0f} suggests runaway equity compounding (start $1K → end {1000+win['best_pnl']:.0f})")

    # Conservative deployment decision
    deploy = True
    deploy_reason = "Beats baseline by ≥$40, WF passes."
    if recent_neg and not degraded_wf:
        deploy = False
        deploy_reason = "Recent 60d/90d folds negative — edge is stale. Do not deploy."
    if win["best_pnl"] > 50000:
        deploy = False
        deploy_reason = "Backtest PnL suggests compounding blow-up. Re-tune with fixed equity sizing before deploy."

    recommendations[sym_key] = {
        "action": "DEPLOY" if deploy else "INVESTIGATE",
        "best_sl": win["best_sl"],
        "best_trail_name": win["best_trail_name"],
        "delta": win["delta"],
        "wf_avg_pf": win["wf_avg_pf"],
        "wf_pos_folds": win["wf_pos_folds"],
        "caveats": cavs,
        "deploy_reason": deploy_reason,
    }

d["recommendations"] = recommendations

# Symbols to re-tune = those with winning deploy-recommended action
retune = [k for k, r in recommendations.items() if r["action"] == "DEPLOY"]
investigate = [k for k, r in recommendations.items() if r["action"] == "INVESTIGATE"]
keep = [k for k, r in recommendations.items() if r["action"] == "KEEP_BASELINE"]
d["summary"] = {
    "deploy": retune,
    "investigate": investigate,
    "keep_baseline": keep,
}

OUT.write_text(json.dumps(d, indent=2, default=str))
print("annotated -> ", OUT)
print(json.dumps(d["summary"], indent=2))
for sym_key, r in recommendations.items():
    print(f"  {sym_key:8s} {r['action']:14s} {r.get('deploy_reason','')}")
    for c in r.get("caveats", []):
        print(f"           - {c}")
