#!/usr/bin/env python3
"""Confirmation runs for UK100.r winners — combined effect + WF."""
import sys, os, json
from copy import deepcopy

sys.path.insert(0, "/Users/ashish/Documents/beast-trader")
os.chdir("/Users/ashish/Documents/beast-trader")

import backtest.v5_backtest as v5

SYMBOL = "UK100.r"
SL_BASE = v5.SL_OVERRIDE.get(SYMBOL, 1.75)
SL_REG_BASE = deepcopy(v5.SL_OVERRIDE_REGIME.get(SYMBOL, {}))


def run(days, params=None, tag=""):
    r = v5.backtest_symbol(SYMBOL, days=days, params=params, verbose=False)
    if r is None:
        return {"tag": tag, "n": 0, "pf": 0, "pnl": 0, "wr": 0, "dd": 0}
    n = r.get("trades", 0) if isinstance(r.get("trades"), int) else 0
    return {
        "tag": tag, "days": days, "n": n,
        "pf": round(float(r.get("pf") or 0), 3),
        "pnl": round(float(r.get("pnl") or 0), 2),
        "wr": round(float(r.get("wr") or 0), 1),
        "dd": round(float(r.get("dd") or 0), 2),
    }


def reset():
    v5.SL_OVERRIDE[SYMBOL] = SL_BASE
    v5.SL_OVERRIDE_REGIME[SYMBOL] = deepcopy(SL_REG_BASE)


print("=" * 72)
print("CONFIRMATION: combine winners")
print("=" * 72)

# A: BASELINE
reset()
print("\n[A] BASELINE (mirror live)")
print("  60d :", run(60))
print("  180d:", run(180))

# B: DIR=SHORT only
reset()
print("\n[B] DIR=SHORT only (others live)")
print("  60d :", run(60, params={"force_direction": "SHORT"}))
print("  180d:", run(180, params={"force_direction": "SHORT"}))

# C: SL volatile=3.0 only
reset()
v5.SL_OVERRIDE_REGIME[SYMBOL] = {"volatile": 3.0}
print("\n[C] SL_regime[volatile]=3.0 (others live)")
print("  60d :", run(60))
print("  180d:", run(180))
reset()

# D: SL volatile=3.0 + DIR=SHORT
v5.SL_OVERRIDE_REGIME[SYMBOL] = {"volatile": 3.0}
print("\n[D] SL_regime[volatile]=3.0 + DIR=SHORT")
print("  60d :", run(60, params={"force_direction": "SHORT"}))
print("  180d:", run(180, params={"force_direction": "SHORT"}))
reset()

# E: SL global=3.0 + DIR=SHORT
v5.SL_OVERRIDE[SYMBOL] = 3.0
print("\n[E] SL_OVERRIDE=3.0 global + DIR=SHORT")
print("  60d :", run(60, params={"force_direction": "SHORT"}))
print("  180d:", run(180, params={"force_direction": "SHORT"}))
reset()

# F: SL global=2.5 + DIR=SHORT (less extreme SL)
v5.SL_OVERRIDE[SYMBOL] = 2.5
print("\n[F] SL_OVERRIDE=2.5 global + DIR=SHORT")
print("  60d :", run(60, params={"force_direction": "SHORT"}))
print("  180d:", run(180, params={"force_direction": "SHORT"}))
reset()

# G: SL volatile=2.5 + DIR=SHORT (less extreme regime)
v5.SL_OVERRIDE_REGIME[SYMBOL] = {"volatile": 2.5}
print("\n[G] SL_regime[volatile]=2.5 + DIR=SHORT")
print("  60d :", run(60, params={"force_direction": "SHORT"}))
print("  180d:", run(180, params={"force_direction": "SHORT"}))
reset()

# H: DIR=SHORT + block hour 17
import config as _cfg
orig_per_sym = _cfg.TOXIC_HOURS_PER_SYMBOL.get(SYMBOL, set())
_cfg.TOXIC_HOURS_PER_SYMBOL[SYMBOL] = set(orig_per_sym) | {17}
v5.TOXIC_HOURS_PER_SYMBOL[SYMBOL] = set(orig_per_sym) | {17}
print("\n[H] DIR=SHORT + block_hour=17")
print("  60d :", run(60, params={"force_direction": "SHORT"}))
print("  180d:", run(180, params={"force_direction": "SHORT"}))

# I: Add hour 17 + 10 (10 had n=3 pnl<0 — borderline, sample thin)
v5.TOXIC_HOURS_PER_SYMBOL[SYMBOL] = set(orig_per_sym) | {17, 10}
print("\n[I] DIR=SHORT + block_hours=17,10")
print("  60d :", run(60, params={"force_direction": "SHORT"}))
print("  180d:", run(180, params={"force_direction": "SHORT"}))

# Reset toxic
v5.TOXIC_HOURS_PER_SYMBOL[SYMBOL] = set(orig_per_sym) if orig_per_sym else set()
_cfg.TOXIC_HOURS_PER_SYMBOL.pop(SYMBOL, None) if not orig_per_sym else None

# Cross-window sanity check: do the toxic candidates show up in BOTH 60d and 180d data?
print("\n" + "=" * 72)
print("Per-hour distribution — 60d window (sanity check)")
print("=" * 72)
reset()
r60 = v5.backtest_symbol(SYMBOL, days=60, verbose=False)
hr60 = {}
for tr in r60.get("details", []):
    h = int(tr.get("hour", -1))
    if h < 0: continue
    hr60.setdefault(h, {"n": 0, "pnl": 0, "gw": 0, "gl": 0, "w": 0})
    pnl = float(tr.get("pnl") or 0)
    hr60[h]["n"] += 1
    hr60[h]["pnl"] += pnl
    if pnl > 0:
        hr60[h]["w"] += 1
        hr60[h]["gw"] += pnl
    else:
        hr60[h]["gl"] += abs(pnl)
for h in sorted(hr60):
    s = hr60[h]
    pf = s["gw"] / s["gl"] if s["gl"] > 0 else 99
    print(f"    h={h:>2}  n={s['n']:>3} pf={pf:>5.2f} pnl=${s['pnl']:>7.2f}")
