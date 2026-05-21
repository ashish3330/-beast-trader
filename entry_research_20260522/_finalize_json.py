#!/usr/bin/env python3 -B
"""Merge variants and portfolio_wf into 10_stat_edge.json (final deliverable)."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
main_p   = D / "10_stat_edge.json"
var_p    = D / "10_stat_edge_variants.json"
port_p   = D / "10_stat_edge_portfolio_wf.json"

main_d = json.load(open(main_p))
var_d  = json.load(open(var_p))
port_d = json.load(open(port_p))

# Merge in variants/portfolio_wf as new top-level keys
main_d["variants_per_symbol"] = var_d.get("variants", {})
main_d["portfolio_wf"]        = port_d

# Determine final recommended variant. None pass ship rule. Document that.
# Recommend the one with best in-sample delta AND most consistent positive WF folds.
best_label = None
best = None
for label, info in port_d.items():
    if not isinstance(info, dict) or "delta" not in info:
        continue
    delta = info["delta"]
    pos   = info["wf_pos_folds"]
    pf    = info["wf_avg_pf"]
    score = delta + 100 * pos + 50 * pf
    if best is None or score > best:
        best = score
        best_label = label

main_d["recommendation"] = {
    "ship_any": False,
    "reason": ("No variant passes WF ship rule (>=3/5 positive folds). "
               "Cell maps require ~80% of 360d trade history before any "
               "cell reaches n_min=10-15. Folds 1-3 of WF produce zero "
               "BL trades because train slice too thin to identify cells."),
    "best_variant_by_score": best_label,
    "ship_rule": "delta>=30 AND wf_avg_pf>1.5 AND pos_folds>=3/4",
}

json.dump(main_d, open(main_p, "w"), indent=2, default=str)
print(f"merged → {main_p}")
