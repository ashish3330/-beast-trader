#!/usr/bin/env python3 -B
"""Hard-tune XAU SMABO for MAX PROFIT: grid MIN_RR x ADX_MIN, full window +
4-fold walk-forward. Rank by total_R subject to robustness (recent folds +) and
DD cap. Emits a ranked table + the recommended ship config."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._smabo_regime_run import run  # noqa: E402

SYM = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
MIN_RR = [1.5, 2.0, 2.5, 3.0]
ADX_MIN = [0, 16, 20, 24, 28]


def _set(rr, adx):
    os.environ[f"SMABO_{SYM}_MIN_RR"] = str(rr)
    os.environ[f"SMABO_{SYM}_ADX_MIN"] = str(adx)


def _clear():
    for k in ("MIN_RR", "ADX_MIN"):
        os.environ.pop(f"SMABO_{SYM}_{k}", None)


rows = []
for rr in MIN_RR:
    for adx in ADX_MIN:
        _set(rr, adx)
        full = run(SYM, days=1200)
        if full.get("status") != "OK" or full.get("trades", 0) < 150:
            continue
        folds = [run(SYM, days=1200, fold=f, folds=4) for f in range(4)]
        favg = [f.get("avg_R", 0) for f in folds if f.get("status") == "OK"]
        fpf = [f.get("pf", 0) for f in folds if f.get("status") == "OK"]
        rows.append({
            "rr": rr, "adx": adx,
            "pf": round(full["pf"], 3), "totalR": round(full["total_R"], 1),
            "wr": round(full["wr"] * 100), "dd": round(full["max_dd_pct"], 1),
            "trades": full["trades"],
            "min_fold_pf": round(min(fpf), 2) if fpf else 0,
            "recent_ok": bool(len(favg) == 4 and favg[2] > 0 and favg[3] > 0),
            "folds_pos": sum(1 for a in favg if a > 0),
        })
_clear()

# Robust set: recent folds positive, no fold PF < 0.85, DD <= 12.
robust = [r for r in rows if r["recent_ok"] and r["min_fold_pf"] >= 0.85 and r["dd"] <= 12]
robust.sort(key=lambda r: r["totalR"], reverse=True)
rows.sort(key=lambda r: r["totalR"], reverse=True)

print(f"{'rr':>5} {'adx':>4} {'PF':>6} {'totalR':>8} {'WR%':>4} {'DD%':>5} "
      f"{'trd':>4} {'minFoldPF':>9} {'foldsPos':>8} {'recent':>6}")
print("-" * 70)
for r in rows[:12]:
    print(f"{r['rr']:>5} {r['adx']:>4} {r['pf']:>6} {r['totalR']:>8} {r['wr']:>4} "
          f"{r['dd']:>5} {r['trades']:>4} {r['min_fold_pf']:>9} {r['folds_pos']:>8} "
          f"{str(r['recent_ok']):>6}")
print("-" * 70)
if robust:
    w = robust[0]
    print(f"SHIP (max profit + robust): MIN_RR={w['rr']} ADX_MIN={w['adx']} "
          f"-> PF={w['pf']} totalR={w['totalR']} WR={w['wr']}% DD={w['dd']}% "
          f"foldsPos={w['folds_pos']}/4")
else:
    print("NO robust config cleared the gate.")
