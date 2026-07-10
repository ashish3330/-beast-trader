#!/usr/bin/env python3 -B
"""Hard entry-filter sweep for SMABO: ADX_MIN x ATR_PCT_MIN x HTF_ALIGN.

Goal: fix the DECAYED entry edge (recent walk-forward folds negative) by
NOT taking breakout signals in the wrong regime — rather than tuning exits.
Runs full-window + 4 non-overlapping folds per config. Ranks by recency:
requires BOTH recent folds (2 AND 3) positive and no full-window regression.

Emits ranked JSON per symbol.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._smabo_regime_run import run  # noqa: E402

SYMBOLS = ["BTCUSD", "XAUUSD"]
ADX_MIN = [0, 18, 20, 22, 25]
ATR_PCT_MIN = [0.0, 0.4]
HTF_ALIGN = [0, 1]
FILTER_KEYS = ("ADX_MIN", "ATR_PCT_MIN", "HTF_ALIGN")


def _clear(sym):
    for k in FILTER_KEYS:
        os.environ.pop(f"SMABO_{sym}_{k}", None)


def _metrics(sym, days=365, fold=None, folds=None):
    r = run(sym, days=days, fold=fold, folds=folds)
    if r.get("status") != "OK":
        return None
    return {"pf": r["pf"], "avg_R": r["avg_R"], "trades": r["trades"],
            "dd": r["max_dd_pct"], "wr": r["wr"]}


def evaluate(sym, adx, atr, htf):
    _clear(sym)
    if adx:
        os.environ[f"SMABO_{sym}_ADX_MIN"] = str(adx)
    if atr:
        os.environ[f"SMABO_{sym}_ATR_PCT_MIN"] = str(atr)
    if htf:
        os.environ[f"SMABO_{sym}_HTF_ALIGN"] = str(htf)
    full = _metrics(sym)
    folds = [_metrics(sym, fold=f, folds=4) for f in range(4)]
    _clear(sym)
    if full is None or any(f is None for f in folds):
        return None
    f2, f3 = folds[2]["avg_R"], folds[3]["avg_R"]
    return {
        "adx_min": adx, "atr_pct_min": atr, "htf_align": htf,
        "full_pf": round(full["pf"], 3), "full_avg_r": round(full["avg_R"], 4),
        "full_trades": full["trades"], "full_dd": round(full["dd"], 2),
        "fold2_avg_r": round(f2, 4), "fold3_avg_r": round(f3, 4),
        "recent_avg_r": round((f2 + f3) / 2, 4),
        "recent_both_pos": bool(f2 > 0 and f3 > 0),
        "min_fold_pf": round(min(x["pf"] for x in folds), 3),
        "fold_pfs": [round(x["pf"], 3) for x in folds],
    }


def main():
    out = {}
    for sym in SYMBOLS:
        base = evaluate(sym, 0, 0.0, 0)
        rows = []
        for adx in ADX_MIN:
            for atr in ATR_PCT_MIN:
                for htf in HTF_ALIGN:
                    if adx == 0 and atr == 0.0 and htf == 0:
                        continue
                    r = evaluate(sym, adx, atr, htf)
                    if r:
                        rows.append(r)
        # Ship-worthy = both recent folds positive, no full-window PF regression
        # vs baseline, healthy trade count, contained DD.
        base_pf = base["full_pf"] if base else 1.0
        worthy = [r for r in rows if r["recent_both_pos"]
                  and r["full_pf"] >= base_pf - 0.02
                  and r["full_trades"] >= 120
                  and r["full_dd"] <= 15.0]
        worthy.sort(key=lambda r: (r["recent_avg_r"], r["full_pf"]), reverse=True)
        rows.sort(key=lambda r: (r["recent_avg_r"], r["full_pf"]), reverse=True)
        out[sym] = {"baseline": base, "worthy_top": worthy[:5], "all_top": rows[:8]}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
