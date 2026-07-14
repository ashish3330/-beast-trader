#!/usr/bin/env python3 -B
"""Hard-tune the TREND book's TRAIL INTERNALS (per symbol) — the trail MECHANICS
the first-wave scalar sweeps did NOT cover.

SCOPE (see the campaign task):
  1. TREND_TRAIL_LOOKBACK  (config.py:761, =22 D1 bars) — the Chandelier
     highest-high / lowest-low window. NEVER swept before. Grid {10,15,22,30,44}
     per symbol. PRIMARY target (changes trail geometry without necessarily
     churning).
  2. The PROFIT-LOCK RATCHET interaction: ACT (activate at N x ATR) x LOCK
     (lock FRAC of peak). Grid over ACT x LOCK per symbol.
  3. GIVEBACK binding — global TREND_GIVEBACK_FRAC vs the per-symbol override.

WHICH PARAM BINDS LIVE (audited in agent/brain.py::_process_trend):
  - config.trend_exit_params(sym) returns the per-symbol (TRAIL, LOCK, GIVEBACK,
    ACT) from TREND_EXIT_PER_SYMBOL. ALL 5 campaign symbols are IN that table, so
    for them the PER-SYMBOL LOCK / ACT / GIVEBACK bind. The globals
    TREND_LOCK_FRAC(0.6) / TREND_LOCK_ACTIVATE_ATR(0.3) / TREND_GIVEBACK_FRAC(0.30)
    bind ONLY for symbols absent from the table. => We tune the EFFECTIVE
    (per-symbol) ratchet, and GIVEBACK's effective binding is the per-symbol value.
  - TREND_TRAIL_LOOKBACK is a GLOBAL (not per-symbol): brain.py passes
    TREND_TRAIL_LOOKBACK into _trend_chandelier for every symbol. So the "best
    lookback per symbol" here is advisory (live uses one global value); we report
    per-symbol optima so a global choice can be made from the trustworthy syms.

ENGINE: reuses backtest/tune/trend_engine.py — load()/d1_context() (this is where
TRAIL_LOOKBACK lives, via the rolling hh/ll window) and summarize(). Scoring uses
a CHURN-FREE cadence (block same-dir re-entry until the D1 signal flips) with the
live TP = TREND_TP_ATR (6.0) and round-trip spread cost. This exactly reproduces
the first-wave scalar-tuner baselines (verified: NAS100.r n=32 / PF~1.68), so
these TRAIL-internal verdicts stack with the scalar verdicts on the same ruler.
trend_engine.simulate (which re-enters after stop/trail = churn-prone) is run as a
CROSS-CHECK and its trade count is reported as churn_n / churn_ratio.

METHOD (honest WF, mandatory guards):
  - WF 60/40 chronological split of the merged H1 frame.
  - One param at a time, then the ACT x LOCK ratchet grid, off the live baseline.
  - SHIP a cell ONLY if ALL hold:
      * best-on-TRAIN total_R AND train_R > baseline train_R
      * test_R >= baseline test_R           (>= neutral OUT-OF-SAMPLE, both folds)
      * test_PF >= baseline test_PF          (PF-hold — no PF collapse)
      * full/ train/ test trade count <= 1.4 x baseline           (CHURN guard)
      * best value is NOT at a grid edge     (reject monotonic-to-edge artifact)
  - DATA TRUST (backtest/tune/DATA_AUDIT.md): only JPN225ft & NAS100.r H1 are
    trustworthy for WF. XAU (91d shallow) / BTC (242 bars) / ETH (63d stale) are
    run for reference only and CANNOT ship (verdict forced to NO_SHIP_DATA).

Writes results_trend_TRAIL.json next to this file. Does NOT touch config.py/brain.py.
Run: python3 -B backtest/tune/tune_trend_TRAIL.py
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import trend_engine as TE  # noqa: E402

OUT = _HERE / "results_trend_TRAIL.json"

SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "JPN225ft", "NAS100.r"]
TP_ATR = 6.0            # live TREND_TP_ATR (first-wave campaign used this)
BASE_LOOKBACK = 22     # live TREND_TRAIL_LOOKBACK
LOOKBACK_GRID = [10, 15, 22, 30, 44]
RATCHET_ACT = [0.3, 0.5, 0.7]
RATCHET_LOCK = [0.4, 0.5, 0.6, 0.7]   # 0.7 = documented churn cliff (kept to prove it; guarded out)
COUNT_GUARD = 1.4      # cell trade count must be <= this x baseline (churn guard)

# DATA_AUDIT.md verdicts — only JPN225ft & NAS100.r H1 are WF-trustworthy.
DATA_TRUST = {
    "XAUUSD":   {"trust": "SPOT_CHECK", "can_ship": False,
                 "reason": "H1 only 91d/1500 bars — far too shallow for rolling WF."},
    "BTCUSD":   {"trust": "BLOCK",      "can_ship": False,
                 "reason": "H1 242 bars/9d — any tune is noise (SHIP_NONE trap)."},
    "ETHUSD":   {"trust": "IS_ONLY",    "can_ship": False,
                 "reason": "H1 deep but 63d stale; OOS half ends 2026-05-12, misses current regime."},
    "JPN225ft": {"trust": "TRUST",      "can_ship": True,
                 "reason": "21k H1 bars, 5d stale — safe to walk-forward."},
    "NAS100.r": {"trust": "TRUST",      "can_ship": True,
                 "reason": "50k H1 bars, 6d stale — safe to walk-forward."},
}


# ── churn-free cadence (block same-dir re-entry until D1 flips) = campaign truth ──
def sim_churnfree(m, TR, LK, GB, ACT, ATR_STOP=3.0, TP=TP_ATR, cost=0.0):
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    pos = 0; entry = sl = tp = peak = e_atr = 0.0; blk = 0
    R = []

    def close(px):
        nonlocal pos
        ret = ((px - entry) / entry) * pos - cost
        r_unit = (ATR_STOP * e_atr) / entry if entry > 0 and e_atr > 0 else 0.0
        R.append(ret / r_unit if r_unit > 0 else 0.0); pos = 0

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blk and s != blk:
                blk = 0
            if s != 0 and s != blk:
                pos = s; entry = o[t]; peak = 0.0; e_atr = a
                sl = entry - ATR_STOP * a if s == 1 else entry + ATR_STOP * a
                tp = None if TP >= 999 else ((entry + TP * a) if s == 1 else (entry - TP * a))
            continue
        if s != 0 and s != pos:
            close(o[t]); blk = 0
            pos = s; entry = o[t]; peak = 0.0; e_atr = a
            sl = entry - ATR_STOP * a if pos == 1 else entry + ATR_STOP * a
            tp = None if TP >= 999 else ((entry + TP * a) if pos == 1 else (entry - TP * a))
            continue
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else -1e18
            ex = None
            if l[t] <= sl:
                ex = sl
            elif gb > -1e17 and l[t] <= gb:
                ex = gb
            elif tp is not None and h[t] >= tp:
                ex = tp
            if ex is not None:
                close(ex); blk = 1
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR * a)
            if peak >= ACT * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else 1e18
            ex = None
            if h[t] >= sl:
                ex = sl
            elif gb < 1e17 and h[t] >= gb:
                ex = gb
            elif tp is not None and l[t] <= tp:
                ex = tp
            if ex is not None:
                close(ex); blk = -1
            else:
                peak = max(peak, entry - l[t])
    return np.array(R)


def _metrics(R):
    if len(R) == 0:
        return {"n": 0, "R": 0.0, "PF": 0.0, "WR": 0.0}
    w = R[R > 0].sum(); ls = -R[R < 0].sum()
    pf = float(w / ls) if ls > 0 else 99.0
    return {"n": int(len(R)), "R": round(float(R.sum()), 2),
            "PF": round(pf, 2), "WR": round(float((R > 0).mean()) * 100, 1)}


def wf(m, TR, LK, GB, ACT, cost):
    """WF 60/40 churn-free metrics + churn-prone (trend_engine) cross-check count."""
    R = sim_churnfree(m, TR, LK, GB, ACT, TP=TP_ATR, cost=cost)
    n = len(R); split = int(n * 0.60)
    full = _metrics(R); train = _metrics(R[:split]); test = _metrics(R[split:])
    churn = TE.simulate(m, TR, LK, GB, ACT, TP=TP_ATR, cost=cost)  # reused engine (churn-prone)
    return {"full": full, "train": train, "test": test,
            "churn_n": len(churn),
            "churn_ratio": round(len(churn) / n, 1) if n else 0.0}


def _passes(cell, base, is_edge):
    """Mandatory guard stack. Returns (ship_bool, reason)."""
    bt, be = base["train"], base["test"]
    if cell["train"]["R"] <= bt["R"] + 1e-9:
        return False, "train does not beat baseline"
    if cell["test"]["R"] < be["R"] - 1e-9:
        return False, "test_R %+.2f < baseline %+.2f (curve-fit)" % (cell["test"]["R"], be["R"])
    if cell["test"]["PF"] < be["PF"] - 1e-9:
        return False, "test_PF %.2f < baseline %.2f (PF collapse)" % (cell["test"]["PF"], be["PF"])
    for fold in ("full", "train", "test"):
        if base[fold]["n"] > 0 and cell[fold]["n"] > COUNT_GUARD * base[fold]["n"] + 1e-9:
            return False, "%s trades %d > %.0fx baseline %d (churn)" % (
                fold, cell[fold]["n"], COUNT_GUARD, base[fold]["n"])
    if is_edge:
        return False, "best at grid edge (monotonic-to-edge artifact)"
    return True, "beats train + neutral test + PF-hold + no churn + interior"


def tune_symbol(sym):
    trust = DATA_TRUST[sym]
    can_ship = trust["can_ship"]
    tr0, lk0, gb0, act0 = TE._cfg_exit_params(sym)   # per-symbol effective baseline
    # baseline frame (live lookback 22) + its spread cost
    fb = TE.load(sym, trail_lookback=BASE_LOOKBACK)
    cost = TE.spread_cost_rt(fb)
    base = wf(fb, tr0, lk0, gb0, act0, cost)
    span = (str(fb["time"].iloc[0])[:10], str(fb["time"].iloc[-1])[:10])
    print("\n" + "=" * 82)
    print("%-9s  trust=%-10s can_ship=%s   H1 bars=%d  %s->%s" % (
        sym, trust["trust"], can_ship, len(fb), span[0], span[1]))
    print("  baseline TR/LK/GB/ACT=%.1f/%.2f/%.2f/%.1f  LB=%d  cost=%.5f" % (
        tr0, lk0, gb0, act0, BASE_LOOKBACK, cost))
    print("  BASE  full n=%d R=%+.2f PF=%.2f | train R=%+.2f PF=%.2f n=%d | test R=%+.2f PF=%.2f n=%d | churn=%dx" % (
        base["full"]["n"], base["full"]["R"], base["full"]["PF"],
        base["train"]["R"], base["train"]["PF"], base["train"]["n"],
        base["test"]["R"], base["test"]["PF"], base["test"]["n"], base["churn_ratio"]))

    params = []

    # ── PARAM 1: TREND_TRAIL_LOOKBACK ──
    print("  -- SWEEP TREND_TRAIL_LOOKBACK %s --" % LOOKBACK_GRID)
    lb_cells = []; best = None
    for lb in LOOKBACK_GRID:
        f = TE.load(sym, trail_lookback=lb)
        c = wf(f, tr0, lk0, gb0, act0, cost)
        c["value"] = lb
        lb_cells.append(c)
        tag = " <base" if lb == BASE_LOOKBACK else ""
        print("    LB=%2d | tr R=%+7.2f PF=%5.2f | te R=%+7.2f PF=%5.2f | full n=%3d churn=%2dx%s" % (
            lb, c["train"]["R"], c["train"]["PF"], c["test"]["R"], c["test"]["PF"],
            c["full"]["n"], c["churn_ratio"], tag))
        if best is None or c["train"]["R"] > best["train"]["R"]:
            best = c
    bv = best["value"]
    is_edge = bv in (LOOKBACK_GRID[0], LOOKBACK_GRID[-1])
    if not can_ship:
        verdict, reason = "NO_SHIP_DATA", trust["reason"]
    elif bv == BASE_LOOKBACK:
        verdict, reason = "SHIP_NONE", "baseline lookback is best-on-train"
    else:
        ok, reason = _passes(best, base, is_edge)
        verdict = "SHIP" if ok else "SHIP_NONE"
    print("    -> best LB=%d  verdict=%s (%s)" % (bv, verdict, reason))
    params.append({"name": "TREND_TRAIL_LOOKBACK", "current": BASE_LOOKBACK, "best": bv,
                   "verdict": verdict, "reason": reason,
                   "train_R": best["train"]["R"], "test_R": best["test"]["R"],
                   "train_PF": best["train"]["PF"], "test_PF": best["test"]["PF"],
                   "grid": LOOKBACK_GRID,
                   "cells": [{"LB": c["value"], "train_R": c["train"]["R"], "train_PF": c["train"]["PF"],
                              "test_R": c["test"]["R"], "test_PF": c["test"]["PF"],
                              "full_n": c["full"]["n"], "churn_ratio": c["churn_ratio"]} for c in lb_cells]})

    # ── PARAM 2: PROFIT-LOCK RATCHET grid  (ACT x LOCK) at baseline lookback ──
    print("  -- RATCHET GRID  ACT %s  x  LOCK %s  (GB=%.2f fixed) --" % (RATCHET_ACT, RATCHET_LOCK, gb0))
    grid_cells = []; gbest = None
    for act in RATCHET_ACT:
        row = []
        for lk in RATCHET_LOCK:
            c = wf(fb, tr0, lk, gb0, act, cost)
            c["act"] = act; c["lock"] = lk
            grid_cells.append({"ACT": act, "LOCK": lk,
                               "train_R": c["train"]["R"], "train_PF": c["train"]["PF"],
                               "test_R": c["test"]["R"], "test_PF": c["test"]["PF"],
                               "full_n": c["full"]["n"], "churn_ratio": c["churn_ratio"]})
            row.append("%+6.2f/%2dx" % (c["train"]["R"], c["churn_ratio"]))
            if gbest is None or c["train"]["R"] > gbest["train"]["R"]:
                gbest = c
        print("    ACT=%.1f | " % act + " | ".join("LK%.1f %s" % (RATCHET_LOCK[i], row[i])
                                                    for i in range(len(RATCHET_LOCK))))
    g_act, g_lk = gbest["act"], gbest["lock"]
    g_edge = (g_act in (RATCHET_ACT[0], RATCHET_ACT[-1])) or (g_lk in (RATCHET_LOCK[0], RATCHET_LOCK[-1]))
    is_base_cell = (abs(g_act - act0) < 1e-9 and abs(g_lk - lk0) < 1e-9)
    if not can_ship:
        gverdict, greason = "NO_SHIP_DATA", trust["reason"]
    elif is_base_cell:
        gverdict, greason = "SHIP_NONE", "baseline (ACT=%.1f,LOCK=%.1f) is best-on-train" % (act0, lk0)
    else:
        ok, greason = _passes(gbest, base, g_edge)
        gverdict = "SHIP" if ok else "SHIP_NONE"
    print("    -> best ratchet ACT=%.1f LOCK=%.1f verdict=%s (%s)" % (g_act, g_lk, gverdict, greason))
    ratchet_best = {"ACT": g_act, "LOCK": g_lk, "current_ACT": act0, "current_LOCK": lk0,
                    "verdict": gverdict, "reason": greason,
                    "train_R": gbest["train"]["R"], "test_R": gbest["test"]["R"],
                    "train_PF": gbest["train"]["PF"], "test_PF": gbest["test"]["PF"],
                    "grid": grid_cells}
    params.append({"name": "RATCHET_ACTxLOCK", "current": [act0, lk0], "best": [g_act, g_lk],
                   "verdict": gverdict, "reason": greason,
                   "train_R": gbest["train"]["R"], "test_R": gbest["test"]["R"],
                   "train_PF": gbest["train"]["PF"], "test_PF": gbest["test"]["PF"]})

    # ── PARAM 3: GIVEBACK binding confirmation (no tune — just report which binds) ──
    from config import (TREND_GIVEBACK_FRAC, TREND_EXIT_PER_SYMBOL)
    in_table = sym in TREND_EXIT_PER_SYMBOL
    eff_gb = TREND_EXIT_PER_SYMBOL[sym]["GIVEBACK"] if in_table else TREND_GIVEBACK_FRAC
    binds = "per-symbol TREND_EXIT_PER_SYMBOL[%s].GIVEBACK" % sym if in_table else "global TREND_GIVEBACK_FRAC"
    params.append({"name": "TREND_GIVEBACK_FRAC", "current": eff_gb, "best": eff_gb,
                   "verdict": "BINDING_CONFIRM",
                   "reason": "effective GIVEBACK for %s = %.2f, bound by %s (global default=%.2f)" % (
                       sym, eff_gb, binds, TREND_GIVEBACK_FRAC),
                   "train_R": base["train"]["R"], "test_R": base["test"]["R"]})

    return {"symbol": sym, "data_trust": trust["trust"], "can_ship": can_ship,
            "trust_reason": trust["reason"],
            "baseline": {"TR": tr0, "LK": lk0, "GB": gb0, "ACT": act0, "LOOKBACK": BASE_LOOKBACK,
                         "full": base["full"], "train": base["train"], "test": base["test"],
                         "churn_ratio": base["churn_ratio"]},
            "params": params, "ratchet_grid_best": ratchet_best}


def main():
    print("=" * 82)
    print("TREND TRAIL-INTERNALS hard-tune  (LOOKBACK + ratchet ACTxLOCK + GIVEBACK binding)")
    print("engine: trend_engine.load/summarize + churn-free cadence + live TP=%.1f" % TP_ATR)
    print("=" * 82)
    per_symbol = []
    for sym in SYMBOLS:
        try:
            per_symbol.append(tune_symbol(sym))
        except Exception as e:
            print("  %s ERROR: %s" % (sym, e))
            per_symbol.append({"symbol": sym, "error": str(e),
                               "data_trust": DATA_TRUST[sym]["trust"], "can_ship": False})

    notes = [
        "COMPONENT = trail internals (TRAIL_LOOKBACK, profit-lock ratchet ACTxLOCK, GIVEBACK "
        "binding). Complements the first-wave scalar sweeps of TRAIL_ATR/LOCK/GIVEBACK/ACT.",
        "BINDING: all 5 symbols are in TREND_EXIT_PER_SYMBOL, so their effective LOCK/ACT/GIVEBACK "
        "are the PER-SYMBOL values (globals TREND_LOCK_FRAC/ACTIVATE_ATR/GIVEBACK_FRAC bind only "
        "for symbols absent from that table). TREND_TRAIL_LOOKBACK is a GLOBAL applied to every "
        "symbol by brain.py::_process_trend, so per-symbol optima here are advisory for a single "
        "global choice.",
        "SCORING: churn-free cadence (block same-dir re-entry until D1 flips) + live TP=6.0 + "
        "round-trip spread cost — reproduces the first-wave scalar baselines (NAS100.r n=32, "
        "PF~1.68) so verdicts are on one ruler. trend_engine.simulate is the churn-prone "
        "cross-check; its trade count is reported as churn_ratio.",
        "GUARDS (all mandatory to SHIP): best-on-train, test_R>=baseline (neutral OOS), "
        "test_PF>=baseline (PF-hold), full/train/test trade count <=1.4x baseline (churn cliff — "
        "LOCK>=0.7 documented 19x-25x churn on ETH), reject grid-edge (monotonic artifact).",
        "DATA TRUST (DATA_AUDIT.md): only JPN225ft & NAS100.r H1 are WF-trustworthy. XAU (91d), "
        "BTC (242 bars), ETH (63d stale) run for reference only, verdict forced NO_SHIP_DATA.",
        "Backtest R is a RELATIVE RANKING only — live risk-capped SL + early-armed giveback "
        "dominate live P/L (see DATA_AUDIT.md sec 3). Does NOT modify config.py/brain.py.",
    ]
    out = {"component": "trail", "per_symbol": per_symbol, "notes": notes}
    OUT.write_text(json.dumps(out, indent=2))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
