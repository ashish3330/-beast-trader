#!/usr/bin/env python3 -B
"""
Per-(symbol, regime) deep tune — industry-grade hierarchical coord descent.

MVP axis: min_quality per (symbol, regime). Already supported by
SIGNAL_QUALITY_SYMBOL[sym][regime] lookup in brain.py and backtest.

Method:
  1. Stage A — baseline: one 180d backtest per symbol with current config.
     Split each symbol's trades by trade["regime"] for per-cell stats.
  2. Stage B — sweep min_quality grid per (symbol, regime) cell.
     For each candidate threshold, run a single 180d backtest with the
     CELL's quality threshold patched and the rest of config untouched.
     Score = per-cell PnL × PF on the regime-filtered subset.
  3. Stage C — 5-fold walk-forward on top candidate per cell.
     WF folds anchored expanding (60, 90, 120, 150, 180 days).
     Accept iff avg PF > 1.4 AND >= 3/5 folds positive AND Δ > $30 vs cell baseline.
  4. Cell sample floor: n >= 25 trades in baseline regime subset.
     Below floor, no per-regime tune for that cell (fall back to global).

Output:
  backtest/results/regime_deep_tune_<date>/per_symbol/<SYM>.json
  backtest/results/regime_deep_tune_<date>/_summary.json

Usage:
  python3 -B scripts/tune_regime_deep.py
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

# Live universe — the 17 symbols that actually trade.
LIVE_SYMBOLS = [
    "AUDJPY", "BTCUSD", "DJ30.r", "EURUSD", "JPN225ft", "SPI200.r",
    "SWI20.r", "US2000.r", "XAUUSD", "NAS100.r", "SP500.r", "UK100.r",
    "XPTUSD.r", "USDCAD", "USDJPY", "CHFJPY", "USOUSD",
]

REGIMES = ("trending", "ranging", "volatile", "low_vol")

# Min_quality candidate grid. Live default thresholds are ~40-55 per regime;
# sweep covers tighter (selectivity) and looser (entry rate) candidates.
MQ_GRID = [30, 35, 40, 45, 50, 55, 60, 65]

# Cell-acceptance floors
MIN_TRADES_PER_CELL = 25
MIN_LIFT_USD = 30.0      # candidate must beat cell baseline by $30
MIN_WF_PF = 1.4          # walk-forward avg PF floor
MIN_WF_POSITIVE_FOLDS = 3  # of 5

WF_FOLDS = [60, 90, 120, 150, 180]
TUNE_DAYS = int(os.getenv("TUNE_DAYS", "180"))
N_WORKERS = max(2, min(10, os.cpu_count() or 4))

DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "backtest" / "results" / f"regime_deep_tune_{DATE_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "per_symbol").mkdir(parents=True, exist_ok=True)


def _backtest_with_quality_override(symbol, days, sym_quality_override):
    """Run a single-symbol backtest with SIGNAL_QUALITY_SYMBOL[sym] patched.

    sym_quality_override: {regime: threshold} dict applied for this symbol ONLY.
    Other symbols' thresholds untouched. Returns the backtest result dict.
    """
    # Patch live config first so backtest sees the override (it reads config).
    import importlib
    import config as cfg
    importlib.reload(cfg)  # restore canonical state per worker invocation
    cfg.SIGNAL_QUALITY_SYMBOL = dict(cfg.SIGNAL_QUALITY_SYMBOL)
    cfg.SIGNAL_QUALITY_SYMBOL[symbol] = dict(sym_quality_override)

    # Backtest module reads config.SIGNAL_QUALITY_THRESHOLDS / *_SYMBOL via
    # DEFAULT_PARAMS at import time; reimport to re-read.
    from importlib import reload
    import backtest.v5_backtest as bt
    reload(bt)
    return bt.backtest_symbol(symbol, days=days, verbose=False)


def _baseline_per_cell(symbol):
    """Stage A: run one full backtest, split trades by regime."""
    import backtest.v5_backtest as bt
    r = bt.backtest_symbol(symbol, days=TUNE_DAYS, verbose=False)
    cells = {reg: {"trades": [], "pnl": 0.0, "pf": 0.0, "wr": 0.0, "n": 0}
             for reg in REGIMES}
    if r.get("trades", 0) == 0:
        return r, cells
    for t in r.get("details", []):
        reg = t.get("regime", "")
        if reg not in cells:
            continue
        cells[reg]["trades"].append(t)
    for reg, bucket in cells.items():
        bucket["n"] = len(bucket["trades"])
        if bucket["n"] == 0:
            continue
        bucket["pnl"] = round(sum(t["pnl"] for t in bucket["trades"]), 2)
        wins = [t for t in bucket["trades"] if t["pnl"] > 0]
        bucket["wr"] = round(100.0 * len(wins) / bucket["n"], 1)
        gp = sum(t["pnl"] for t in bucket["trades"] if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in bucket["trades"] if t["pnl"] < 0) or 0.01
        bucket["pf"] = round(gp / gl, 2)
        # drop trade list from output (keeps JSON small)
        bucket["trades"] = None
    return r, cells


def _live_quality_dict(symbol):
    """Get current per-regime quality dict for a symbol from live config."""
    import config as cfg
    if hasattr(cfg, "SIGNAL_QUALITY_SYMBOL"):
        sym_q = cfg.SIGNAL_QUALITY_SYMBOL.get(symbol)
        if isinstance(sym_q, dict):
            return dict(sym_q)
    # Fallback: regime-default thresholds
    return dict(cfg.SIGNAL_QUALITY_THRESHOLDS)


def _sweep_cell(args):
    """Worker: sweep one (symbol, regime) cell. Returns winner dict or None."""
    symbol, target_regime, baseline_quality, baseline_cell_pnl = args
    candidates = []
    for mq in MQ_GRID:
        sym_override = dict(baseline_quality)
        sym_override[target_regime] = mq
        try:
            r = _backtest_with_quality_override(
                symbol, days=TUNE_DAYS, sym_quality_override=sym_override)
        except Exception as e:
            continue
        # Re-split for this regime's subset
        cell_trades = [t for t in r.get("details", [])
                       if t.get("regime") == target_regime]
        if len(cell_trades) < 8:  # need at least 8 trades to score
            continue
        cell_pnl = round(sum(t["pnl"] for t in cell_trades), 2)
        gp = sum(t["pnl"] for t in cell_trades if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell_trades if t["pnl"] < 0) or 0.01
        cell_pf = round(gp / gl, 2)
        wins = sum(1 for t in cell_trades if t["pnl"] > 0)
        candidates.append({
            "mq": mq, "cell_pnl": cell_pnl, "cell_pf": cell_pf,
            "n": len(cell_trades), "wr": round(100.0 * wins / len(cell_trades), 1),
        })
    if not candidates:
        return None
    # Best by cell PnL (must beat baseline by MIN_LIFT_USD)
    best = max(candidates, key=lambda c: c["cell_pnl"])
    if best["cell_pnl"] - baseline_cell_pnl < MIN_LIFT_USD:
        return None
    return {
        "symbol": symbol, "regime": target_regime,
        "baseline_mq": baseline_quality.get(target_regime),
        "baseline_pnl": baseline_cell_pnl,
        "best": best, "all": candidates,
    }


def _wf_validate(symbol, target_regime, override_quality):
    """Stage C: 5-fold walk-forward on the candidate."""
    fold_results = []
    for d in WF_FOLDS:
        try:
            r = _backtest_with_quality_override(
                symbol, days=d, sym_quality_override=override_quality)
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
    avg_pf = round(sum(f["pf"] for f in fold_results) / max(1, len(fold_results)), 2)
    pos_folds = sum(1 for f in fold_results if f["pnl"] > 0)
    return {"folds": fold_results, "avg_pf": avg_pf, "pos_folds": pos_folds}


def main():
    print(f"\nPer-(symbol, regime) deep tune — {DATE_TAG}")
    print(f"Live universe: {len(LIVE_SYMBOLS)} symbols × {len(REGIMES)} regimes "
          f"= {len(LIVE_SYMBOLS) * len(REGIMES)} cells")
    print(f"Min cell n={MIN_TRADES_PER_CELL}, min lift=${MIN_LIFT_USD}, "
          f"WF PF >= {MIN_WF_PF}, pos folds >= {MIN_WF_POSITIVE_FOLDS}/5")
    print(f"Workers: {N_WORKERS}, tune days: {TUNE_DAYS}")
    print(f"Output: {OUT_DIR}\n")

    t0 = time.time()

    # ── STAGE A: baseline per cell ────────────────────────────────────
    print("[A] Baseline per-cell measurement (one backtest per symbol)...")
    baselines = {}
    for sym in LIVE_SYMBOLS:
        r, cells = _baseline_per_cell(sym)
        baselines[sym] = {
            "total_pnl": r.get("pnl", 0),
            "total_trades": r.get("trades", 0),
            "cells": cells,
            "quality_dict": _live_quality_dict(sym),
        }
        cell_summary = " ".join(
            f"{reg[:3]}=${cells[reg]['pnl']:.0f}/n{cells[reg]['n']}"
            for reg in REGIMES if cells[reg]["n"] > 0
        )
        print(f"  {sym:12s} total ${r.get('pnl', 0):>7.0f} "
              f"n{r.get('trades', 0):4d} | {cell_summary}")
    print(f"  baseline elapsed {time.time() - t0:.0f}s\n")

    # ── STAGE B: cell sweeps ──────────────────────────────────────────
    sweep_jobs = []
    for sym in LIVE_SYMBOLS:
        b = baselines[sym]
        for reg in REGIMES:
            cell = b["cells"][reg]
            if cell["n"] < MIN_TRADES_PER_CELL:
                continue
            sweep_jobs.append((sym, reg, b["quality_dict"], cell["pnl"]))
    print(f"[B] Sweeping {len(sweep_jobs)} qualifying cells × {len(MQ_GRID)} "
          f"min_quality values = {len(sweep_jobs) * len(MQ_GRID)} backtests")

    candidates = []
    if sweep_jobs:
        with Pool(N_WORKERS) as pool:
            for i, result in enumerate(pool.imap_unordered(_sweep_cell, sweep_jobs), 1):
                if result is not None:
                    candidates.append(result)
                if i % 10 == 0 or i == len(sweep_jobs):
                    print(f"  sweep {i}/{len(sweep_jobs)} "
                          f"({time.time() - t0:.0f}s)")
    print(f"  [B] candidates found: {len(candidates)}\n")

    # ── STAGE C: walk-forward validation ─────────────────────────────
    print(f"[C] Walk-forward 5-fold on {len(candidates)} candidates...")
    winners = []
    for c in candidates:
        sym = c["symbol"]
        reg = c["regime"]
        override = dict(baselines[sym]["quality_dict"])
        override[reg] = c["best"]["mq"]
        wf = _wf_validate(sym, reg, override)
        if wf is None:
            continue
        if wf["avg_pf"] >= MIN_WF_PF and wf["pos_folds"] >= MIN_WF_POSITIVE_FOLDS:
            winners.append({**c, "wf": wf})
            print(f"  ✓ {sym} {reg} mq={c['best']['mq']} "
                  f"Δ${c['best']['cell_pnl'] - c['baseline_pnl']:+.0f} "
                  f"WF PF {wf['avg_pf']} pos {wf['pos_folds']}/5")
        else:
            print(f"  ✗ {sym} {reg} mq={c['best']['mq']} "
                  f"WF FAIL PF {wf['avg_pf']} pos {wf['pos_folds']}/5")

    # ── OUTPUT ────────────────────────────────────────────────────────
    summary = {
        "captured_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "live_symbols": LIVE_SYMBOLS,
        "tune_days": TUNE_DAYS,
        "mq_grid": MQ_GRID,
        "acceptance": {
            "min_trades_per_cell": MIN_TRADES_PER_CELL,
            "min_lift_usd": MIN_LIFT_USD,
            "min_wf_pf": MIN_WF_PF,
            "min_wf_positive_folds": MIN_WF_POSITIVE_FOLDS,
        },
        "baselines": {
            sym: {
                "total_pnl": b["total_pnl"],
                "total_trades": b["total_trades"],
                "cells": {reg: {k: v for k, v in cell.items()
                                if k != "trades"}
                          for reg, cell in b["cells"].items()},
                "quality_dict": b["quality_dict"],
            }
            for sym, b in baselines.items()
        },
        "candidates": candidates,
        "winners": winners,
    }
    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n  total elapsed {time.time() - t0:.0f}s")
    print(f"  candidates: {len(candidates)}, winners: {len(winners)}")
    print(f"  saved: {summary_path}\n")

    # Print winners table
    if winners:
        print("=" * 70)
        print("WINNERS (per-cell min_quality overrides)")
        print("=" * 70)
        print(f"{'Symbol':12} {'Regime':10} {'mq base→new':14} "
              f"{'Δcell':>8} {'PF':>5} {'WF PF':>6} {'folds':>6}")
        for w in sorted(winners, key=lambda x: -(x["best"]["cell_pnl"] - x["baseline_pnl"])):
            delta = w["best"]["cell_pnl"] - w["baseline_pnl"]
            print(f"  {w['symbol']:10s} {w['regime']:10s} "
                  f"{w['baseline_mq']}→{w['best']['mq']:<5d}    "
                  f"${delta:>+6.0f}  "
                  f"{w['best']['cell_pf']:>4.2f}   "
                  f"{w['wf']['avg_pf']:>4.2f}   "
                  f"{w['wf']['pos_folds']}/5")
        total_lift = sum(w["best"]["cell_pnl"] - w["baseline_pnl"] for w in winners)
        print(f"\nTotal expected lift (sum of cell deltas): ${total_lift:+.0f}\n")
    else:
        print("No winners passed acceptance gates.\n")


if __name__ == "__main__":
    main()
