#!/usr/bin/env python3 -B
"""Compute ship-eligibility per variant per symbol from walk-forward results.

Criteria:
    1. Δ ≥ $30 per symbol (across full 180d)
    2. WF avg PF > 1.5 (per-symbol average across folds with ≥5 trades)
    3. ≥ 3/5 folds positive Δ vs baseline
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WF = json.load(open(ROOT / "entry_research_20260522/walkforward_results.json"))
ITER2 = json.load(open(ROOT / "entry_research_20260522/iter2_results.json"))

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
FOLDS = ["F1_recent", "F2", "F3", "F4", "F5_oldest"]
VARIANTS = list(WF.keys())

print(f"{'Variant':<22}  Total Δ    Sym ≥+$30   AvgPF    Folds≥0    SHIP")
print("=" * 80)
for variant in VARIANTS:
    if variant == "baseline":
        continue

    # 180d delta vs baseline (from iter2)
    total_delta = 0
    sym_wins = 0  # symbols with Δ ≥ +$30 over 180d
    for sym in SYMBOLS:
        bv = ITER2["results"]["baseline"][sym]["pnl"]
        vv = ITER2["results"][variant][sym]["pnl"]
        total_delta += (vv - bv)
        if (vv - bv) >= 30:
            sym_wins += 1

    # PF per fold per symbol
    pf_vals = []
    for f in FOLDS:
        for sym in SYMBOLS:
            d = WF[variant][f]["symbols"].get(sym, {})
            if isinstance(d, dict) and d.get("trades", 0) >= 5:
                pf_vals.append(d["pf"])
    avg_pf = np.mean(pf_vals) if pf_vals else 0

    # Folds with positive total delta vs baseline
    folds_positive = 0
    for f in FOLDS:
        vt = WF[variant][f]["total_pnl"]
        bt = WF["baseline"][f]["total_pnl"]
        if (vt - bt) > 0:
            folds_positive += 1

    eligible = (sym_wins >= 4 and avg_pf > 1.5 and folds_positive >= 3)
    flag = "SHIP" if eligible else "REJECT"
    print(f"{variant:<22}  ${total_delta:+9.2f}   {sym_wins}/{len(SYMBOLS):<10} {avg_pf:.2f}   {folds_positive}/{len(FOLDS)}     {flag}")

# Per-symbol detail for top candidate
print("\n--- Per-symbol Δ (180d) for top candidate ---")
for variant in VARIANTS:
    if variant == "baseline": continue
    print(f"\n{variant}:")
    for sym in SYMBOLS:
        bv = ITER2["results"]["baseline"][sym]["pnl"]
        vv = ITER2["results"][variant][sym]["pnl"]
        delta = vv - bv
        folds_pos = sum(1 for f in FOLDS
                        if (WF[variant][f]["symbols"].get(sym, {}).get("pnl", 0) -
                            WF["baseline"][f]["symbols"].get(sym, {}).get("pnl", 0)) > 0)
        mark = "✓" if delta >= 30 else "·" if abs(delta) < 30 else "✗"
        print(f"  {sym:<12} {mark} Δ=${delta:+8.2f}  folds_positive={folds_pos}/5")
