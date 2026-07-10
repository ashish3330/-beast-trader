#!/usr/bin/env python3
"""
HARD ENTRY-SIDE TUNE for UK100.r — 2026-06-12
Sweeps: SL_ATR_MULT (global + per-regime), MIN_QUALITY per regime,
        DIRECTION_BIAS, TOXIC_HOURS_UTC, ML_ENABLED.

Mirrors live config first, then sweeps.
WF: days=60 vs days=180.
"""
import sys
import os
import json
import time
from copy import deepcopy

sys.path.insert(0, "/Users/ashish/Documents/beast-trader")
os.chdir("/Users/ashish/Documents/beast-trader")

import backtest.v5_backtest as v5

SYMBOL = "UK100.r"

# ---- Mirror LIVE cfg ----
# Live SL_OVERRIDE['UK100.r'] = 1.75 (per config.py line 129)
# UK100 NOT in SL_OVERRIDE_REGIME currently
# DIRECTION_BIAS: UK100 not set => BOTH (0)
# SIGNAL_QUALITY_SYMBOL['UK100.r'] = {trending:60, ranging:65, volatile:60, low_vol:95}
# TOXIC_HOURS_UTC = {1,2,3,4}
# ML: DRAGON_ML_ENABLED.get(UK100, True) — check
print(f"[mirror] SL_OVERRIDE[{SYMBOL}] = {v5.SL_OVERRIDE.get(SYMBOL)}")
print(f"[mirror] SL_OVERRIDE_REGIME[{SYMBOL}] = {v5.SL_OVERRIDE_REGIME.get(SYMBOL)}")
print(f"[mirror] DIR_BIAS[{SYMBOL}] = {v5.DIR_BIAS.get(SYMBOL, 0)}")
print(f"[mirror] TOXIC_HOURS = {v5.TOXIC_HOURS}")
print(f"[mirror] TOXIC_HOURS_PER_SYMBOL[{SYMBOL}] = {v5.TOXIC_HOURS_PER_SYMBOL.get(SYMBOL)}")

# Snapshots for restoration after each sweep
SL_BASE = v5.SL_OVERRIDE.get(SYMBOL, 1.75)
SL_REG_BASE = deepcopy(v5.SL_OVERRIDE_REGIME.get(SYMBOL, {}))
DIR_BASE = v5.DIR_BIAS.get(SYMBOL, 0)
TOXIC_BASE = set(v5.TOXIC_HOURS)
TOXIC_SYM_BASE = set(v5.TOXIC_HOURS_PER_SYMBOL.get(SYMBOL, set()))


def run(days, params=None, tag="", with_details=False):
    """Run BT and extract key metrics."""
    try:
        r = v5.backtest_symbol(SYMBOL, days=days, params=params, verbose=False)
    except Exception as e:
        return {"err": str(e), "n": 0, "pf": 0, "pnl": 0, "wr": 0, "dd": 0, "tag": tag}
    if r is None:
        return {"err": "None", "n": 0, "pf": 0, "pnl": 0, "wr": 0, "dd": 0, "tag": tag}
    n = r.get("trades", 0) if isinstance(r.get("trades"), int) else len(r.get("trades", []) or [])
    pf = r.get("pf", 0)
    pnl = r.get("pnl") or r.get("pnl_usd") or 0
    wr = r.get("wr", 0)
    dd = r.get("dd") or r.get("max_dd") or r.get("dd_pct") or 0
    avg_r = r.get("avg_r") or r.get("avg_R") or 0
    out = {
        "tag": tag,
        "days": days,
        "n": int(n) if n is not None else 0,
        "pf": round(float(pf or 0), 3),
        "pnl": round(float(pnl), 2),
        "wr": round(float(wr or 0), 1),
        "dd": round(float(dd or 0), 2),
        "avg_r": round(float(avg_r or 0), 3),
    }
    if with_details:
        out["_details"] = r.get("details") or {}
    return out


# ════════════════════════════════════════════════════════
# BASELINE (mirror live)
# ════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"BASELINE — live mirror SL={SL_BASE}, DIR={DIR_BASE}, TOXIC={TOXIC_BASE}")
print("=" * 72)
b60 = run(60, tag="baseline_60d")
b180 = run(180, tag="baseline_180d")
print(f"  60d : {b60}")
print(f"  180d: {b180}")

cache_bound = (b60["n"] == b180["n"] and b60["pf"] == b180["pf"])
print(f"  cache_bound (60==180)? {cache_bound}")


# ════════════════════════════════════════════════════════
# SWEEP 1: SL_ATR_MULT (global)
# ════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SWEEP 1: SL_ATR_MULT global")
print("=" * 72)
sl_results = []
for sl in [0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
    v5.SL_OVERRIDE[SYMBOL] = sl
    r60 = run(60, tag=f"SL={sl}_60d")
    r180 = run(180, tag=f"SL={sl}_180d")
    print(f"  SL={sl:>4}  60d: n={r60['n']:>3} pf={r60['pf']:>5} pnl=${r60['pnl']:>8}  |  180d: n={r180['n']:>3} pf={r180['pf']:>5} pnl=${r180['pnl']:>8}")
    sl_results.append({"sl": sl, "r60": r60, "r180": r180})
v5.SL_OVERRIDE[SYMBOL] = SL_BASE


# ════════════════════════════════════════════════════════
# SWEEP 2: SL_ATR_MULT per-regime
# Quick scan: keep other regimes at baseline (1.75 / no override), sweep ONE regime at a time
# ════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SWEEP 2: SL_ATR_MULT per regime (one regime swept at a time)")
print("=" * 72)
sl_regime_results = {}
REGIMES = ["trending", "ranging", "volatile", "low_vol"]
for regime in REGIMES:
    sl_regime_results[regime] = []
    for sl in [0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        v5.SL_OVERRIDE_REGIME[SYMBOL] = {regime: sl}
        r60 = run(60, tag=f"SL_REG[{regime}]={sl}_60d")
        r180 = run(180, tag=f"SL_REG[{regime}]={sl}_180d")
        print(f"  {regime[:8]:>8} SL={sl:>4}  60d: n={r60['n']:>3} pf={r60['pf']:>5} pnl=${r60['pnl']:>8}  |  180d: n={r180['n']:>3} pf={r180['pf']:>5} pnl=${r180['pnl']:>8}")
        sl_regime_results[regime].append({"sl": sl, "r60": r60, "r180": r180})
    v5.SL_OVERRIDE_REGIME[SYMBOL] = deepcopy(SL_REG_BASE)


# ════════════════════════════════════════════════════════
# SWEEP 3: MIN_QUALITY per regime
# Sweep ONE regime at a time, others at live baseline
# ════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SWEEP 3: MIN_QUALITY per regime (live baseline UK100: tr60/ra65/vol60/lv95)")
print("=" * 72)
LIVE_MQ = {"trending": 60, "ranging": 65, "volatile": 60, "low_vol": 95}
mq_results = {}
for regime in REGIMES:
    mq_results[regime] = []
    for q in [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]:
        mq = dict(LIVE_MQ)
        mq[regime] = q
        params = {"min_quality": mq}
        r60 = run(60, params=params, tag=f"MQ[{regime}]={q}_60d")
        r180 = run(180, params=params, tag=f"MQ[{regime}]={q}_180d")
        print(f"  {regime[:8]:>8} Q={q:>3}  60d: n={r60['n']:>3} pf={r60['pf']:>5} pnl=${r60['pnl']:>8}  |  180d: n={r180['n']:>3} pf={r180['pf']:>5} pnl=${r180['pnl']:>8}")
        mq_results[regime].append({"q": q, "r60": r60, "r180": r180})


# ════════════════════════════════════════════════════════
# SWEEP 4: DIRECTION_BIAS
# ════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SWEEP 4: DIRECTION_BIAS")
print("=" * 72)
dir_results = []
for direction in ["LONG", "SHORT", "BOTH"]:
    params = {"force_direction": direction}
    r60 = run(60, params=params, tag=f"DIR={direction}_60d")
    r180 = run(180, params=params, tag=f"DIR={direction}_180d")
    print(f"  DIR={direction:>5}  60d: n={r60['n']:>3} pf={r60['pf']:>5} pnl=${r60['pnl']:>8}  |  180d: n={r180['n']:>3} pf={r180['pf']:>5} pnl=${r180['pnl']:>8}")
    dir_results.append({"dir": direction, "r60": r60, "r180": r180})


# ════════════════════════════════════════════════════════
# SWEEP 5: TOXIC_HOURS — per-hour PF grid
# Run a baseline trade list, group by hour, identify PF<0.5 hours with n>=4
# To approximate this, we run with each candidate hour blocked individually + measure delta.
# ════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SWEEP 5: TOXIC_HOURS — block one hour at a time, see PF delta")
print("=" * 72)
# First get hourly distribution from baseline 180d trades
v5.SL_OVERRIDE[SYMBOL] = SL_BASE
v5.SL_OVERRIDE_REGIME[SYMBOL] = deepcopy(SL_REG_BASE)
r_base = v5.backtest_symbol(SYMBOL, days=180, verbose=False)
hour_stats = {}
if r_base and r_base.get("details"):
    for tr in r_base["details"]:
        h = tr.get("hour")
        if h is None:
            continue
        h = int(h)
        hour_stats.setdefault(h, {"n": 0, "wins": 0, "gross_win": 0.0, "gross_loss": 0.0, "pnl": 0.0})
        pnl = float(tr.get("pnl") or 0)
        hour_stats[h]["n"] += 1
        hour_stats[h]["pnl"] += pnl
        if pnl > 0:
            hour_stats[h]["wins"] += 1
            hour_stats[h]["gross_win"] += pnl
        else:
            hour_stats[h]["gross_loss"] += abs(pnl)

print("  per-hour distribution (from baseline 180d trades):")
toxic_candidates = []
for h in sorted(hour_stats.keys()):
    s = hour_stats[h]
    pf = s["gross_win"] / s["gross_loss"] if s["gross_loss"] > 0 else 99.0
    wr = s["wins"] / s["n"] * 100 if s["n"] else 0
    flag = ""
    if s["n"] >= 4 and pf < 0.5:
        flag = " <-- TOXIC CANDIDATE"
        toxic_candidates.append(h)
    print(f"    h={h:>2}  n={s['n']:>3} pf={pf:>5.2f} wr={wr:>5.1f}% pnl=${s['pnl']:>7.2f}{flag}")
print(f"  toxic candidates (n>=4, pf<0.5): {toxic_candidates}")


# ════════════════════════════════════════════════════════
# SWEEP 6: ML_ENABLED
# ════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SWEEP 6: ML_ENABLED")
print("=" * 72)
ml_results = []
# Patch DRAGON_ML_ENABLED for UK100
from config import DRAGON_ML_ENABLED as _DML
ml_orig = _DML.get(SYMBOL, True)
for ml_on in [True, False]:
    _DML[SYMBOL] = ml_on
    r60 = run(60, tag=f"ML={ml_on}_60d")
    r180 = run(180, tag=f"ML={ml_on}_180d")
    print(f"  ML={ml_on}  60d: n={r60['n']:>3} pf={r60['pf']:>5} pnl=${r60['pnl']:>8}  |  180d: n={r180['n']:>3} pf={r180['pf']:>5} pnl=${r180['pnl']:>8}")
    ml_results.append({"ml": ml_on, "r60": r60, "r180": r180})
_DML[SYMBOL] = ml_orig


# ════════════════════════════════════════════════════════
# DUMP all results to JSON
# ════════════════════════════════════════════════════════
out = {
    "symbol": SYMBOL,
    "baseline": {"60d": b60, "180d": b180},
    "cache_bound": cache_bound,
    "sweep_sl_global": sl_results,
    "sweep_sl_regime": sl_regime_results,
    "sweep_min_quality": mq_results,
    "sweep_direction": dir_results,
    "sweep_toxic_hours_per_hour": {str(k): v for k, v in hour_stats.items()},
    "toxic_candidates": toxic_candidates,
    "sweep_ml": ml_results,
}
out_path = f"/tmp/uk100_tune_{int(time.time())}.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[done] results dumped to {out_path}")
