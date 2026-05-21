#!/usr/bin/env python3 -B
"""Per-(symbol, regime) min_quality tune — task 2026-05-21.

Constraints:
  - READ-ONLY on repo files (no auto_tuned.py writes).
  - All injection in-memory via mutating config.SIGNAL_QUALITY_SYMBOL.
  - Live bot is running; we only touch our own process state.

Method:
  Stage A   : 180d baseline per symbol with current config.
              Split trades by regime → per-cell PnL/PF/n.
  Stage B   : for each (sym, regime) with cell-n >= 5, sweep
              min_quality ∈ {25, 28, 30, 32, 35, 38, 40}.
              Winner cell = highest 180d cell-restricted PnL among candidates
              whose OVERALL (full-symbol) PF >= 1.8.
              Δ vs baseline cell-PnL must be >= $30.
  Stage C   : 5-fold walk-forward at expanding windows
              [60, 90, 120, 150, 180]d on candidate. Pass = >=3/5 folds
              with positive OVERALL PnL on the symbol.
  Ship gate : Δ >= $30 AND WF passed.

Parallelism: multiprocessing.Pool over (sym, regime, candidate_mq) jobs.

Output: tune_session_20260521/regime_min_quality.json
"""
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Live universe (config.py SYMBOLS dict at session start)
LIVE_SYMBOLS = [
    "DJ30.r", "JPN225ft", "SPI200.r", "SWI20.r", "US2000.r",
    "XAUUSD", "BTCUSD", "ETHUSD", "XAGUSD",
    "AUDJPY", "EURUSD",
    "NAS100.r", "SP500.r", "UK100.r", "XPTUSD.r",
    "USDCAD", "USDJPY", "CHFJPY", "USOUSD",
]

REGIMES = ("trending", "ranging", "volatile", "low_vol")
MQ_GRID = [25, 28, 30, 32, 35, 38, 40]

MIN_CELL_TRADES = 5
MIN_LIFT_USD = 30.0
MIN_OVERALL_PF = 1.8
WF_FOLDS = [60, 90, 120, 150, 180]
MIN_WF_POSITIVE = 3       # >= 3/5 positive folds
TUNE_DAYS = 180

OUT_DIR = ROOT / "tune_session_20260521"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "regime_min_quality.json"
DETAIL_JSON = OUT_DIR / "regime_min_quality_full.json"

N_WORKERS = max(2, min(8, (cpu_count() or 4)))


# ──────────────────────────────────────────────────────────────────────
# Worker primitives — each call mutates THIS process's config dict, runs
# one BT, restores original dict. Safe under fork-based multiprocessing
# because each worker has its own memory image.
# ──────────────────────────────────────────────────────────────────────
def _bt_with_override(symbol: str, days: int, override_dict: dict):
    """Run one backtest with config.SIGNAL_QUALITY_SYMBOL[symbol] = override_dict.
    Restores prior value before returning."""
    import config as cfg
    import backtest.v5_backtest as bt
    prev = cfg.SIGNAL_QUALITY_SYMBOL.get(symbol)
    try:
        cfg.SIGNAL_QUALITY_SYMBOL[symbol] = dict(override_dict)
        r = bt.backtest_symbol(symbol, days=days, verbose=False)
    finally:
        if prev is None:
            cfg.SIGNAL_QUALITY_SYMBOL.pop(symbol, None)
        else:
            cfg.SIGNAL_QUALITY_SYMBOL[symbol] = prev
    return r


def _flt(x):
    """Coerce numpy float / int → builtin so json.dumps serializes as number."""
    try:
        return float(x)
    except Exception:
        return x


def _baseline_worker(symbol):
    """Stage A: full backtest, split by regime. Return baseline dict + current quality."""
    import config as cfg
    import backtest.v5_backtest as bt
    quality = cfg.SIGNAL_QUALITY_SYMBOL.get(symbol)
    if isinstance(quality, dict):
        quality = dict(quality)
    else:
        quality = dict(cfg.SIGNAL_QUALITY_THRESHOLDS)
    try:
        r = bt.backtest_symbol(symbol, days=TUNE_DAYS, verbose=False)
    except Exception as e:
        return symbol, {"error": repr(e), "quality": quality}
    cells = {reg: {"n": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0} for reg in REGIMES}
    for t in r.get("details", []) or []:
        reg = t.get("regime")
        if reg not in cells:
            continue
        cells[reg]["n"] += 1
    for reg in REGIMES:
        sub = [t for t in r.get("details", []) or [] if t.get("regime") == reg]
        n = len(sub)
        if n == 0:
            continue
        gp = sum(t["pnl"] for t in sub if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in sub if t["pnl"] < 0) or 0.01
        cells[reg]["n"] = n
        cells[reg]["pnl"] = round(_flt(sum(t["pnl"] for t in sub)), 2)
        cells[reg]["pf"] = round(_flt(gp / gl), 2)
        wins = sum(1 for t in sub if t["pnl"] > 0)
        cells[reg]["wr"] = round(100.0 * wins / n, 1)
    summary = {
        "total_pnl": _flt(r.get("pnl", 0)),
        "total_pf": _flt(r.get("pf", 0)),
        "total_trades": r.get("trades", 0),
        "cells": cells,
        "quality": quality,
    }
    return symbol, summary


def _sweep_cell_job(args):
    """Worker for one (sym, regime, candidate_mq) test.
    Returns dict with cell PnL/PF and overall PF for the candidate."""
    symbol, regime, baseline_quality, candidate_mq = args
    override = dict(baseline_quality)
    override[regime] = candidate_mq
    try:
        r = _bt_with_override(symbol, TUNE_DAYS, override)
    except Exception as e:
        return {"symbol": symbol, "regime": regime, "mq": candidate_mq, "error": repr(e)}
    if not r or r.get("trades", 0) == 0:
        return {"symbol": symbol, "regime": regime, "mq": candidate_mq,
                "cell_n": 0, "cell_pnl": 0.0, "cell_pf": 0.0,
                "overall_pf": 0.0, "overall_pnl": 0.0, "overall_trades": 0}
    cell_trades = [t for t in r.get("details", []) if t.get("regime") == regime]
    n = len(cell_trades)
    cell_pnl = round(sum(t["pnl"] for t in cell_trades), 2) if n else 0.0
    gp = sum(t["pnl"] for t in cell_trades if t["pnl"] > 0)
    gl = sum(abs(t["pnl"]) for t in cell_trades if t["pnl"] < 0) or 0.01
    cell_pf = round(gp / gl, 2) if n else 0.0
    return {
        "symbol": symbol, "regime": regime, "mq": candidate_mq,
        "cell_n": n, "cell_pnl": _flt(cell_pnl), "cell_pf": _flt(cell_pf),
        "overall_pf": _flt(r.get("pf", 0)), "overall_pnl": _flt(r.get("pnl", 0)),
        "overall_trades": r.get("trades", 0),
    }


def _wf_job(args):
    """Worker for one (sym, override, fold_days) walk-forward backtest."""
    symbol, override, fold_days = args
    try:
        r = _bt_with_override(symbol, fold_days, override)
    except Exception as e:
        return {"symbol": symbol, "days": fold_days, "error": repr(e),
                "pnl": 0, "pf": 0, "n": 0}
    if not r or r.get("trades", 0) == 0:
        return {"symbol": symbol, "days": fold_days, "pnl": 0.0, "pf": 0.0, "n": 0}
    return {"symbol": symbol, "days": fold_days,
            "pnl": _flt(r.get("pnl", 0)), "pf": _flt(r.get("pf", 0)),
            "n": r.get("trades", 0)}


# ──────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"\n{'='*72}")
    print(f"  Per-(symbol, regime) min_quality tune  —  {datetime.now().isoformat(timespec='seconds')}")
    print(f"{'='*72}")
    print(f"  Live universe: {len(LIVE_SYMBOLS)} symbols × {len(REGIMES)} regimes")
    print(f"  MQ grid: {MQ_GRID}")
    print(f"  Min cell n: {MIN_CELL_TRADES}   Min lift: ${MIN_LIFT_USD}")
    print(f"  Overall PF gate: >= {MIN_OVERALL_PF}   WF positive folds: >= {MIN_WF_POSITIVE}/5")
    print(f"  Workers: {N_WORKERS}\n")

    # ── STAGE A: baseline ──────────────────────────────────────────────
    print("[A] Baseline 180d per symbol (parallel)...")
    with Pool(N_WORKERS) as pool:
        baseline_results = dict(pool.map(_baseline_worker, LIVE_SYMBOLS))
    for sym in LIVE_SYMBOLS:
        b = baseline_results[sym]
        if "error" in b:
            print(f"  {sym:12s}: ERROR {b['error']}")
            continue
        cells_s = " ".join(
            f"{r[:3]}=${b['cells'][r]['pnl']:.0f}/n{b['cells'][r]['n']}/PF{b['cells'][r]['pf']}"
            for r in REGIMES if b['cells'][r]['n'] > 0
        )
        print(f"  {sym:12s} total ${b['total_pnl']:>8.0f} "
              f"PF {b['total_pf']:<5} n{b['total_trades']:>4} | {cells_s}")
    print(f"  baseline elapsed {time.time() - t0:.0f}s\n")

    # ── STAGE B: cell sweeps ───────────────────────────────────────────
    sweep_jobs = []
    for sym in LIVE_SYMBOLS:
        b = baseline_results[sym]
        if "error" in b:
            continue
        for reg in REGIMES:
            if b["cells"][reg]["n"] < MIN_CELL_TRADES:
                continue
            for mq in MQ_GRID:
                sweep_jobs.append((sym, reg, b["quality"], mq))
    print(f"[B] Cell sweeps: {len(sweep_jobs)} backtests "
          f"({sum(1 for s in LIVE_SYMBOLS for r in REGIMES if baseline_results[s].get('cells',{}).get(r,{}).get('n',0) >= MIN_CELL_TRADES)} qualifying cells × {len(MQ_GRID)})")
    tB = time.time()
    sweep_results = []
    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_sweep_cell_job, sweep_jobs), 1):
            sweep_results.append(res)
            if i % 50 == 0 or i == len(sweep_jobs):
                print(f"  sweep {i}/{len(sweep_jobs)} ({time.time()-tB:.0f}s)")
    print(f"  sweep elapsed {time.time() - tB:.0f}s\n")

    # group sweep results by (sym, regime)
    grouped = {}
    for r in sweep_results:
        if "error" in r:
            continue
        key = (r["symbol"], r["regime"])
        grouped.setdefault(key, []).append(r)

    # ── Pick best candidate per cell ───────────────────────────────────
    print("[B'] Picking winners (cell_pnl maximizer, overall_pf >= 1.8, Δ >= $30)...")
    candidates_to_validate = {}
    cell_records = {}   # (sym, regime) → dict that will go to final JSON
    for sym in LIVE_SYMBOLS:
        b = baseline_results[sym]
        if "error" in b:
            continue
        cell_records.setdefault(sym, {})
        for reg in REGIMES:
            base_cell = b["cells"][reg]
            current_mq = b["quality"].get(reg)
            n_base = base_cell["n"]
            base_pnl = base_cell["pnl"]
            entry = {
                "current": current_mq,
                "winner": current_mq,
                "delta_pnl": 0,
                "wf_passed": False,
                "recommend_ship": False,
                "reason": "",
                "baseline_cell_pnl": base_pnl,
                "baseline_cell_n": n_base,
                "baseline_cell_pf": base_cell["pf"],
                "grid_results": [],
            }
            if n_base < MIN_CELL_TRADES:
                entry["reason"] = f"baseline cell n={n_base} < {MIN_CELL_TRADES}"
                cell_records[sym][reg] = entry
                continue
            cell_cands = grouped.get((sym, reg), [])
            entry["grid_results"] = [
                {"mq": c["mq"], "cell_n": c["cell_n"], "cell_pnl": c["cell_pnl"],
                 "cell_pf": c["cell_pf"], "overall_pf": c["overall_pf"],
                 "overall_pnl": c["overall_pnl"], "overall_trades": c["overall_trades"]}
                for c in sorted(cell_cands, key=lambda x: x["mq"])
            ]
            # Filter: cell_n >= 5 AND overall_pf >= 1.8
            valid = [c for c in cell_cands
                     if c["cell_n"] >= MIN_CELL_TRADES and c["overall_pf"] >= MIN_OVERALL_PF]
            if not valid:
                entry["reason"] = "no candidate satisfies overall_pf >= 1.8 + cell_n >= 5"
                cell_records[sym][reg] = entry
                continue
            best = max(valid, key=lambda c: c["cell_pnl"])
            delta = round(best["cell_pnl"] - base_pnl, 2)
            entry["winner"] = best["mq"]
            entry["delta_pnl"] = delta
            entry["best_overall_pf"] = best["overall_pf"]
            entry["best_cell_pf"] = best["cell_pf"]
            entry["best_cell_n"] = best["cell_n"]
            if delta < MIN_LIFT_USD:
                entry["reason"] = f"Δ ${delta:.0f} < ${MIN_LIFT_USD}"
                cell_records[sym][reg] = entry
                # Still no WF — winner same as current effectively (small gain)
                continue
            # Will run WF
            override = dict(b["quality"])
            override[reg] = best["mq"]
            candidates_to_validate[(sym, reg)] = {
                "override": override, "best": best, "entry": entry,
                "baseline_pnl": base_pnl,
            }
            cell_records[sym][reg] = entry

    print(f"  candidates to WF-validate: {len(candidates_to_validate)}\n")

    # ── STAGE C: walk-forward 5 folds (ordered Pool.map for identity) ──
    print(f"[C] Walk-forward 5 folds × {len(candidates_to_validate)} cells = "
          f"{5 * len(candidates_to_validate)} backtests")
    tC = time.time()
    wf_buckets = {(s, r): [] for (s, r) in candidates_to_validate}
    sequential_wf_jobs = []
    cell_idx_for_job = []
    for (sym, reg), info in candidates_to_validate.items():
        for d in WF_FOLDS:
            sequential_wf_jobs.append((sym, info["override"], d))
            cell_idx_for_job.append((sym, reg))
    if sequential_wf_jobs:
        with Pool(N_WORKERS) as pool:
            results = pool.map(_wf_job, sequential_wf_jobs)
        for cell_key, r in zip(cell_idx_for_job, results):
            wf_buckets[cell_key].append(r)
    print(f"  walk-forward elapsed {time.time() - tC:.0f}s\n")

    # ── Apply WF acceptance ────────────────────────────────────────────
    ship_count = 0
    for (sym, reg), info in candidates_to_validate.items():
        folds = wf_buckets[(sym, reg)]
        pos_folds = sum(1 for f in folds if f.get("pnl", 0) > 0)
        avg_pf = round(sum(f.get("pf", 0) for f in folds) / max(1, len(folds)), 2)
        entry = cell_records[sym][reg]
        entry["wf_folds"] = folds
        entry["wf_pos_folds"] = pos_folds
        entry["wf_avg_pf"] = avg_pf
        if pos_folds >= MIN_WF_POSITIVE:
            entry["wf_passed"] = True
            entry["recommend_ship"] = True
            entry["reason"] = f"OK Δ${entry['delta_pnl']:.0f} WF {pos_folds}/5"
            ship_count += 1
        else:
            entry["wf_passed"] = False
            entry["recommend_ship"] = False
            entry["reason"] = f"WF FAIL {pos_folds}/5 positive"

    # ── Build output ───────────────────────────────────────────────────
    # Compact ship-form per spec
    compact = {}
    for sym in LIVE_SYMBOLS:
        compact[sym] = {}
        for reg in REGIMES:
            e = cell_records.get(sym, {}).get(reg)
            if e is None:
                continue
            compact[sym][reg] = {
                "current": e["current"],
                "winner": e["winner"],
                "delta_pnl": e["delta_pnl"],
                "wf_passed": e["wf_passed"],
                "recommend_ship": e["recommend_ship"],
            }

    OUT_JSON.write_text(json.dumps(compact, indent=2, default=str))

    # Detailed JSON keeps grids + WF folds for audit
    full = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
        "live_symbols": LIVE_SYMBOLS,
        "mq_grid": MQ_GRID,
        "acceptance": {
            "min_cell_trades": MIN_CELL_TRADES,
            "min_lift_usd": MIN_LIFT_USD,
            "min_overall_pf": MIN_OVERALL_PF,
            "wf_folds": WF_FOLDS,
            "min_wf_positive": MIN_WF_POSITIVE,
        },
        "baselines": baseline_results,
        "cells": cell_records,
        "ship_count": ship_count,
    }
    DETAIL_JSON.write_text(json.dumps(full, indent=2, default=str))

    print(f"\n{'='*72}")
    print(f"  SHIP-WORTHY CELLS: {ship_count}")
    print(f"{'='*72}")
    if ship_count:
        print(f"  {'Symbol':12} {'Regime':10} {'mq base→new':14} {'Δ':>10} {'WF':>6}")
        for sym in LIVE_SYMBOLS:
            for reg in REGIMES:
                e = cell_records.get(sym, {}).get(reg)
                if e and e["recommend_ship"]:
                    print(f"  {sym:12} {reg:10} "
                          f"{e['current']:>3}→{e['winner']:<3}        "
                          f"${e['delta_pnl']:>+7.0f}   "
                          f"{e['wf_pos_folds']}/5")
    print(f"\n  compact JSON  : {OUT_JSON}")
    print(f"  detailed JSON : {DETAIL_JSON}")
    print(f"  total elapsed : {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
