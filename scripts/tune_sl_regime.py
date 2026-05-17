#!/usr/bin/env python3 -B
"""
Per-(symbol, regime) SL ATR multiplier deep tune.

Plumbing committed 2026-05-17: backtest/v5_backtest.py + agent/brain.py +
execution/executor.py + config.py.SYMBOL_ATR_SL_OVERRIDE_REGIME all read
the new per-cell override. Schema (sym → regime → float). Populated here.

Method (mirrors tune_regime_deep.py):
  Stage A — baseline backtest per symbol; split trades by regime.
  Stage B — for each (sym, regime) cell with >= MIN_TRADES, sweep SL_GRID.
  Stage C — 5-fold WF on top candidate per cell.
  Cross-symbol verifier — full-universe backtest with ALL accepted winners
                          applied at once. Reject any winner that causes
                          the universe-total to drop below the no-winner
                          baseline (catches interaction effects).

Output:
  backtest/results/sl_regime_tune_<date>/_summary.json

Usage:
  python3 -B scripts/tune_sl_regime.py
"""
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LIVE_SYMBOLS = [
    "AUDJPY", "BTCUSD", "DJ30.r", "EURUSD", "JPN225ft", "SPI200.r",
    "SWI20.r", "US2000.r", "XAUUSD", "NAS100.r", "SP500.r", "UK100.r",
    "XPTUSD.r", "USDCAD", "USDJPY", "CHFJPY", "USOUSD",
]

REGIMES = ("trending", "ranging", "volatile", "low_vol")

# SL ATR multiplier grid. Span ~0.5 (tight, anti-noise) → 3.0 (wide,
# for volatile assets). Mirrors phase9 SL_GRID coverage.
SL_GRID = [0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

MIN_TRADES_PER_CELL = 25
MIN_LIFT_USD = 30.0
MIN_WF_PF = 1.4
MIN_WF_POSITIVE_FOLDS = 3

WF_FOLDS = [60, 90, 120, 150, 180]
TUNE_DAYS = int(os.getenv("TUNE_DAYS", "180"))
N_WORKERS = max(2, min(10, os.cpu_count() or 4))

DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "backtest" / "results" / f"sl_regime_tune_{DATE_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _backtest_with_sl_regime(symbol, days, sl_regime_override):
    """Run a single-symbol backtest with SYMBOL_ATR_SL_OVERRIDE_REGIME[sym]
    patched to sl_regime_override (a {regime: mult} dict)."""
    import importlib
    import config as cfg
    importlib.reload(cfg)
    cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
    cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME[symbol] = dict(sl_regime_override)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    return bt.backtest_symbol(symbol, days=days, verbose=False)


def _baseline_per_cell(symbol):
    import backtest.v5_backtest as bt
    r = bt.backtest_symbol(symbol, days=TUNE_DAYS, verbose=False)
    cells = {reg: {"pnl": 0.0, "pf": 0.0, "wr": 0.0, "n": 0} for reg in REGIMES}
    if r.get("trades", 0) == 0:
        return r, cells
    for t in r.get("details", []):
        reg = t.get("regime", "")
        if reg in cells:
            cells[reg]["n"] = cells[reg].get("n", 0) + 1
            cells[reg]["_trades"] = cells[reg].get("_trades", [])
            cells[reg]["_trades"].append(t)
    for reg, bucket in cells.items():
        if bucket["n"] == 0:
            continue
        trades = bucket.pop("_trades", [])
        bucket["pnl"] = round(sum(t["pnl"] for t in trades), 2)
        wins = [t for t in trades if t["pnl"] > 0]
        bucket["wr"] = round(100.0 * len(wins) / bucket["n"], 1) if bucket["n"] else 0
        gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in trades if t["pnl"] < 0) or 0.01
        bucket["pf"] = round(gp / gl, 2)
    return r, cells


def _sweep_cell(args):
    symbol, target_regime, baseline_cell_pnl = args
    candidates = []
    for sl_mult in SL_GRID:
        override = {target_regime: sl_mult}  # only this regime cell
        try:
            r = _backtest_with_sl_regime(symbol, days=TUNE_DAYS,
                                         sl_regime_override=override)
        except Exception:
            continue
        cell_trades = [t for t in r.get("details", [])
                       if t.get("regime") == target_regime]
        if len(cell_trades) < 8:
            continue
        cell_pnl = round(sum(t["pnl"] for t in cell_trades), 2)
        gp = sum(t["pnl"] for t in cell_trades if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell_trades if t["pnl"] < 0) or 0.01
        cell_pf = round(gp / gl, 2)
        wins = sum(1 for t in cell_trades if t["pnl"] > 0)
        candidates.append({
            "sl_mult": sl_mult, "cell_pnl": cell_pnl, "cell_pf": cell_pf,
            "n": len(cell_trades),
            "wr": round(100.0 * wins / len(cell_trades), 1),
        })
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c["cell_pnl"])
    if best["cell_pnl"] - baseline_cell_pnl < MIN_LIFT_USD:
        return None
    return {
        "symbol": symbol, "regime": target_regime,
        "baseline_pnl": baseline_cell_pnl,
        "best": best, "all": candidates,
    }


def _wf_validate(symbol, target_regime, sl_mult):
    fold_results = []
    for d in WF_FOLDS:
        try:
            r = _backtest_with_sl_regime(symbol, days=d,
                                         sl_regime_override={target_regime: sl_mult})
        except Exception:
            return None
        cell_trades = [t for t in r.get("details", [])
                       if t.get("regime") == target_regime]
        if not cell_trades:
            fold_results.append({"days": d, "pnl": 0, "pf": 0, "n": 0})
            continue
        gp = sum(t["pnl"] for t in cell_trades if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell_trades if t["pnl"] < 0) or 0.01
        fold_results.append({
            "days": d, "pnl": round(sum(t["pnl"] for t in cell_trades), 2),
            "pf": round(gp / gl, 2), "n": len(cell_trades),
        })
    avg_pf = round(sum(f["pf"] for f in fold_results)
                   / max(1, len(fold_results)), 2)
    pos = sum(1 for f in fold_results if f["pnl"] > 0)
    return {"folds": fold_results, "avg_pf": avg_pf, "pos_folds": pos}


def _full_universe_verifier(combined_overrides):
    """Run a single-symbol backtest on each LIVE_SYMBOL with the FULL
    combined overrides dict applied. Sum total PnL. Compare to baseline.
    Returns (total_pnl, per_symbol). Sequential — workers can't share patch state.
    """
    import importlib
    import config as cfg
    importlib.reload(cfg)
    cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
    for s, rd in combined_overrides.items():
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME.setdefault(s, {}).update(rd)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    per_sym = {}
    total = 0
    for sym in LIVE_SYMBOLS:
        r = bt.backtest_symbol(sym, days=TUNE_DAYS, verbose=False)
        per_sym[sym] = r.get("pnl", 0)
        total += r.get("pnl", 0)
    return round(total, 2), per_sym


def main():
    print(f"\nPer-(symbol, regime) SL ATR-mult deep tune — {DATE_TAG}")
    print(f"Live universe: {len(LIVE_SYMBOLS)} symbols × {len(REGIMES)} regimes")
    print(f"SL grid: {SL_GRID}")
    print(f"Min cell n={MIN_TRADES_PER_CELL}, min lift=${MIN_LIFT_USD}, "
          f"WF PF >= {MIN_WF_PF}, pos folds >= {MIN_WF_POSITIVE_FOLDS}/5")
    print(f"Workers: {N_WORKERS}, tune days: {TUNE_DAYS}")
    print(f"Output: {OUT_DIR}\n")

    t0 = time.time()

    # ── STAGE A: baseline ────────────────────────────────────────────
    print("[A] Baseline per-cell measurement...")
    baselines = {}
    baseline_total = 0
    for sym in LIVE_SYMBOLS:
        r, cells = _baseline_per_cell(sym)
        baselines[sym] = {
            "total_pnl": r.get("pnl", 0),
            "total_trades": r.get("trades", 0),
            "cells": cells,
        }
        baseline_total += r.get("pnl", 0)
        cell_summary = " ".join(
            f"{reg[:3]}=${cells[reg]['pnl']:.0f}/n{cells[reg]['n']}"
            for reg in REGIMES if cells[reg]["n"] > 0
        )
        print(f"  {sym:12s} total ${r.get('pnl', 0):>7.0f} "
              f"n{r.get('trades', 0):4d} | {cell_summary}")
    print(f"  baseline elapsed {time.time() - t0:.0f}s "
          f"| universe total ${baseline_total:.0f}\n")

    # ── STAGE B: cell sweeps ─────────────────────────────────────────
    sweep_jobs = []
    for sym in LIVE_SYMBOLS:
        for reg in REGIMES:
            cell = baselines[sym]["cells"][reg]
            if cell["n"] < MIN_TRADES_PER_CELL:
                continue
            sweep_jobs.append((sym, reg, cell["pnl"]))
    print(f"[B] Sweeping {len(sweep_jobs)} qualifying cells × {len(SL_GRID)} SL"
          f" values = {len(sweep_jobs) * len(SL_GRID)} backtests")
    candidates = []
    if sweep_jobs:
        with Pool(N_WORKERS) as pool:
            for i, result in enumerate(pool.imap_unordered(_sweep_cell, sweep_jobs), 1):
                if result is not None:
                    candidates.append(result)
                if i % 5 == 0 or i == len(sweep_jobs):
                    print(f"  sweep {i}/{len(sweep_jobs)} ({time.time() - t0:.0f}s)")
    print(f"  [B] candidates: {len(candidates)}\n")

    # ── STAGE C: WF validation ────────────────────────────────────────
    print(f"[C] Walk-forward 5-fold...")
    wf_winners = []
    for c in candidates:
        wf = _wf_validate(c["symbol"], c["regime"], c["best"]["sl_mult"])
        if wf is None:
            continue
        if wf["avg_pf"] >= MIN_WF_PF and wf["pos_folds"] >= MIN_WF_POSITIVE_FOLDS:
            wf_winners.append({**c, "wf": wf})
            print(f"  ✓ {c['symbol']} {c['regime']} SL={c['best']['sl_mult']} "
                  f"Δ${c['best']['cell_pnl'] - c['baseline_pnl']:+.0f} "
                  f"WF PF {wf['avg_pf']} pos {wf['pos_folds']}/5")
        else:
            print(f"  ✗ {c['symbol']} {c['regime']} SL={c['best']['sl_mult']} "
                  f"WF FAIL PF {wf['avg_pf']} pos {wf['pos_folds']}/5")

    # ── STAGE D: cross-symbol verifier ────────────────────────────────
    print(f"\n[D] Cross-symbol verifier — full-universe backtest with all "
          f"{len(wf_winners)} winners applied...")
    combined = {}
    for w in wf_winners:
        combined.setdefault(w["symbol"], {})[w["regime"]] = w["best"]["sl_mult"]
    combined_total, combined_per_sym = _full_universe_verifier(combined)
    universe_delta = combined_total - baseline_total
    print(f"  Universe baseline: ${baseline_total:.0f}")
    print(f"  Universe w/winners: ${combined_total:.0f}")
    print(f"  Universe delta:    ${universe_delta:+.0f}")

    # If universe regresses, find and drop the worst-offender winners one-by-one
    accepted_winners = list(wf_winners)
    if universe_delta < 0:
        print(f"  REGRESSION — dropping winners one at a time to find culprits...")
        while accepted_winners and universe_delta < 0:
            worst = None
            worst_delta = -1e9
            for w in accepted_winners:
                test_combined = {}
                for ww in accepted_winners:
                    if ww is w:
                        continue
                    test_combined.setdefault(ww["symbol"], {})[ww["regime"]] = \
                        ww["best"]["sl_mult"]
                t_total, _ = _full_universe_verifier(test_combined)
                gain_from_dropping = t_total - combined_total
                if gain_from_dropping > worst_delta:
                    worst_delta = gain_from_dropping
                    worst = w
            if worst is None or worst_delta <= 0:
                break
            print(f"  drop {worst['symbol']} {worst['regime']} → "
                  f"+${worst_delta:.0f}")
            accepted_winners = [w for w in accepted_winners if w is not worst]
            combined = {}
            for ww in accepted_winners:
                combined.setdefault(ww["symbol"], {})[ww["regime"]] = \
                    ww["best"]["sl_mult"]
            combined_total, combined_per_sym = _full_universe_verifier(combined)
            universe_delta = combined_total - baseline_total
            print(f"  recheck: universe ${combined_total:.0f} "
                  f"Δ${universe_delta:+.0f}")

    # ── OUTPUT ────────────────────────────────────────────────────────
    summary = {
        "captured_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "live_symbols": LIVE_SYMBOLS,
        "tune_days": TUNE_DAYS,
        "sl_grid": SL_GRID,
        "acceptance": {
            "min_trades_per_cell": MIN_TRADES_PER_CELL,
            "min_lift_usd": MIN_LIFT_USD,
            "min_wf_pf": MIN_WF_PF,
            "min_wf_positive_folds": MIN_WF_POSITIVE_FOLDS,
        },
        "baseline_total": baseline_total,
        "candidates": candidates,
        "wf_winners": wf_winners,
        "accepted_winners": accepted_winners,
        "combined_overrides": combined,
        "combined_universe_total": combined_total,
        "combined_universe_delta": universe_delta,
        "combined_per_sym": combined_per_sym,
    }
    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  saved: {summary_path}")
    print(f"  total elapsed {time.time() - t0:.0f}s")

    if accepted_winners:
        print("\n" + "=" * 70)
        print("ACCEPTED WINNERS (post cross-symbol verification)")
        print("=" * 70)
        print(f"{'Symbol':12} {'Regime':10} {'SL_mult':>8} "
              f"{'Δcell':>8} {'WF PF':>6} {'folds':>6}")
        for w in sorted(accepted_winners,
                        key=lambda x: -(x["best"]["cell_pnl"] - x["baseline_pnl"])):
            d = w["best"]["cell_pnl"] - w["baseline_pnl"]
            print(f"  {w['symbol']:10s} {w['regime']:10s} "
                  f"{w['best']['sl_mult']:>7.2f}   ${d:>+6.0f}   "
                  f"{w['wf']['avg_pf']:>4.2f}    {w['wf']['pos_folds']}/5")
        print(f"\nUniverse total: ${baseline_total:.0f} → "
              f"${combined_total:.0f}  Δ${universe_delta:+.0f}\n")
    else:
        print("\nNo winners passed all gates.\n")


if __name__ == "__main__":
    main()
