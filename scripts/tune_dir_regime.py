#!/usr/bin/env python3 -B
"""
Per-(symbol, regime) direction bias deep tune.

Plumbed 2026-05-17: config.DIRECTION_BIAS_REGIME, brain.py + backtest read
it with chain (sym, regime) → (sym) → both. 'BOTH' explicitly opens both
sides in that regime even when symbol-level DIRECTION_BIAS restricts.

Method:
  Stage A — baseline per-cell.
  Stage B — for each qualifying cell, try {'LONG', 'SHORT', 'BOTH'}, pick
            best by cell-PnL.
  Stage C — 5-fold WF on top candidate.
  Stage D — cross-symbol full-universe verifier.

Output:
  backtest/results/dir_regime_tune_<date>/_summary.json
"""
import json, os, sys, time
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
DIR_GRID = ("LONG", "SHORT", "BOTH")

MIN_TRADES_PER_CELL = 25
MIN_LIFT_USD = 30.0
MIN_WF_PF = 1.4
MIN_WF_POSITIVE_FOLDS = 3
WF_FOLDS = [60, 90, 120, 150, 180]
TUNE_DAYS = int(os.getenv("TUNE_DAYS", "180"))
N_WORKERS = max(2, min(10, os.cpu_count() or 4))

DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "backtest" / "results" / f"dir_regime_tune_{DATE_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _backtest_with_dir_override(symbol, days, dir_regime_override):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    cfg.DIRECTION_BIAS_REGIME = dict(cfg.DIRECTION_BIAS_REGIME)
    cfg.DIRECTION_BIAS_REGIME[symbol] = dict(dir_regime_override)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    return bt.backtest_symbol(symbol, days=days, verbose=False)


def _baseline_per_cell(symbol):
    import backtest.v5_backtest as bt
    r = bt.backtest_symbol(symbol, days=TUNE_DAYS, verbose=False)
    cells = {reg: {"pnl": 0.0, "pf": 0.0, "wr": 0.0, "n": 0, "long_pnl": 0, "short_pnl": 0, "long_n": 0, "short_n": 0} for reg in REGIMES}
    if r.get("trades", 0) == 0:
        return r, cells
    for t in r.get("details", []):
        reg = t.get("regime", "")
        if reg in cells:
            d = t.get("direction", 0)
            cells[reg]["n"] += 1
            if d == 1:
                cells[reg]["long_n"] += 1
                cells[reg]["long_pnl"] += t["pnl"]
            else:
                cells[reg]["short_n"] += 1
                cells[reg]["short_pnl"] += t["pnl"]
            cells[reg]["pnl"] += t["pnl"]
    for reg, bucket in cells.items():
        for k in ("pnl", "long_pnl", "short_pnl"):
            bucket[k] = round(bucket[k], 2)
    return r, cells


def _sweep_cell(args):
    symbol, target_regime, baseline_cell_pnl = args
    candidates = []
    for d in DIR_GRID:
        override = {target_regime: d}
        try:
            r = _backtest_with_dir_override(symbol, days=TUNE_DAYS, dir_regime_override=override)
        except Exception:
            continue
        cell_trades = [t for t in r.get("details", []) if t.get("regime") == target_regime]
        if len(cell_trades) < 5:
            continue
        cell_pnl = round(sum(t["pnl"] for t in cell_trades), 2)
        gp = sum(t["pnl"] for t in cell_trades if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell_trades if t["pnl"] < 0) or 0.01
        candidates.append({
            "dir": d, "cell_pnl": cell_pnl,
            "cell_pf": round(gp / gl, 2),
            "n": len(cell_trades),
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


def _wf_validate(symbol, target_regime, dir_choice):
    folds = []
    for d_days in WF_FOLDS:
        try:
            r = _backtest_with_dir_override(symbol, days=d_days, dir_regime_override={target_regime: dir_choice})
        except Exception:
            return None
        cell_trades = [t for t in r.get("details", []) if t.get("regime") == target_regime]
        if not cell_trades:
            folds.append({"days": d_days, "pnl": 0, "pf": 0, "n": 0})
            continue
        gp = sum(t["pnl"] for t in cell_trades if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell_trades if t["pnl"] < 0) or 0.01
        folds.append({"days": d_days, "pnl": round(sum(t["pnl"] for t in cell_trades), 2),
                      "pf": round(gp / gl, 2), "n": len(cell_trades)})
    avg_pf = round(sum(f["pf"] for f in folds) / max(1, len(folds)), 2)
    pos = sum(1 for f in folds if f["pnl"] > 0)
    return {"folds": folds, "avg_pf": avg_pf, "pos_folds": pos}


def _full_universe_verifier(combined):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    cfg.DIRECTION_BIAS_REGIME = dict(cfg.DIRECTION_BIAS_REGIME)
    for s, rd in combined.items():
        cfg.DIRECTION_BIAS_REGIME.setdefault(s, {}).update(rd)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    total = 0
    per_sym = {}
    for sym in LIVE_SYMBOLS:
        r = bt.backtest_symbol(sym, days=TUNE_DAYS, verbose=False)
        per_sym[sym] = r.get("pnl", 0)
        total += r.get("pnl", 0)
    return round(total, 2), per_sym


def main():
    print(f"\nPer-(symbol, regime) direction-bias tune — {DATE_TAG}")
    print(f"Workers: {N_WORKERS}, tune days: {TUNE_DAYS}\n")
    t0 = time.time()

    print("[A] Baseline...")
    baselines = {}
    baseline_total = 0
    for sym in LIVE_SYMBOLS:
        r, cells = _baseline_per_cell(sym)
        baselines[sym] = {"total_pnl": r.get("pnl", 0), "total_trades": r.get("trades", 0), "cells": cells}
        baseline_total += r.get("pnl", 0)
        cell_summary = " ".join(
            f"{reg[:3]}=${cells[reg]['pnl']:.0f}/L{cells[reg]['long_n']}S{cells[reg]['short_n']}"
            for reg in REGIMES if cells[reg]["n"] > 0
        )
        print(f"  {sym:12s} ${r.get('pnl', 0):>7.0f} n{r.get('trades', 0):4d} | {cell_summary}")
    print(f"  baseline {time.time() - t0:.0f}s | universe ${baseline_total:.0f}\n")

    sweep_jobs = []
    for sym in LIVE_SYMBOLS:
        for reg in REGIMES:
            cell = baselines[sym]["cells"][reg]
            if cell["n"] < MIN_TRADES_PER_CELL:
                continue
            sweep_jobs.append((sym, reg, cell["pnl"]))
    print(f"[B] Sweeping {len(sweep_jobs)} cells × 3 directions = {len(sweep_jobs) * 3} backtests")
    candidates = []
    if sweep_jobs:
        with Pool(N_WORKERS) as pool:
            for i, result in enumerate(pool.imap_unordered(_sweep_cell, sweep_jobs), 1):
                if result is not None:
                    candidates.append(result)
                if i % 5 == 0 or i == len(sweep_jobs):
                    print(f"  sweep {i}/{len(sweep_jobs)} ({time.time() - t0:.0f}s)")
    print(f"  [B] candidates: {len(candidates)}\n")

    print(f"[C] WF validation...")
    winners = []
    for c in candidates:
        wf = _wf_validate(c["symbol"], c["regime"], c["best"]["dir"])
        if wf is None:
            continue
        if wf["avg_pf"] >= MIN_WF_PF and wf["pos_folds"] >= MIN_WF_POSITIVE_FOLDS:
            winners.append({**c, "wf": wf})
            print(f"  ✓ {c['symbol']} {c['regime']} → {c['best']['dir']:5s} "
                  f"Δ${c['best']['cell_pnl'] - c['baseline_pnl']:+.0f} "
                  f"WF PF {wf['avg_pf']} {wf['pos_folds']}/5")
        else:
            print(f"  ✗ {c['symbol']} {c['regime']} → {c['best']['dir']:5s} "
                  f"WF FAIL PF {wf['avg_pf']} {wf['pos_folds']}/5")

    print(f"\n[D] Full-universe verifier on {len(winners)} winners...")
    combined = {}
    for w in winners:
        combined.setdefault(w["symbol"], {})[w["regime"]] = w["best"]["dir"]
    combined_total, per_sym = _full_universe_verifier(combined)
    delta = combined_total - baseline_total
    print(f"  Universe baseline ${baseline_total:.0f} → ${combined_total:.0f}  Δ${delta:+.0f}")

    accepted = list(winners)
    if delta < 0:
        print(f"  REGRESSION — dropping worst-offender iteratively...")
        while accepted and delta < 0:
            best_drop = None
            best_gain = -1e9
            for w in accepted:
                test = {}
                for ww in accepted:
                    if ww is w:
                        continue
                    test.setdefault(ww["symbol"], {})[ww["regime"]] = ww["best"]["dir"]
                t_total, _ = _full_universe_verifier(test)
                gain = t_total - combined_total
                if gain > best_gain:
                    best_gain = gain
                    best_drop = w
            if best_drop is None or best_gain <= 0:
                break
            print(f"  drop {best_drop['symbol']} {best_drop['regime']} → +${best_gain:.0f}")
            accepted = [w for w in accepted if w is not best_drop]
            combined = {}
            for ww in accepted:
                combined.setdefault(ww["symbol"], {})[ww["regime"]] = ww["best"]["dir"]
            combined_total, per_sym = _full_universe_verifier(combined)
            delta = combined_total - baseline_total
            print(f"  recheck ${combined_total:.0f} Δ${delta:+.0f}")

    summary = {
        "captured_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "baseline_total": baseline_total,
        "candidates": candidates,
        "wf_winners": winners,
        "accepted_winners": accepted,
        "combined_overrides": combined,
        "combined_universe_total": combined_total,
        "combined_universe_delta": delta,
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  saved: {OUT_DIR}/_summary.json")
    print(f"  total elapsed {time.time() - t0:.0f}s")

    if accepted:
        print("\n" + "=" * 70)
        print("ACCEPTED WINNERS")
        print("=" * 70)
        for w in sorted(accepted, key=lambda x: -(x["best"]["cell_pnl"] - x["baseline_pnl"])):
            d = w["best"]["cell_pnl"] - w["baseline_pnl"]
            print(f"  {w['symbol']:10s} {w['regime']:10s} → {w['best']['dir']:5s}  "
                  f"Δ${d:>+6.0f}  WF PF {w['wf']['avg_pf']}  {w['wf']['pos_folds']}/5")
        print(f"\nUniverse ${baseline_total:.0f} → ${combined_total:.0f}  Δ${delta:+.0f}\n")
    else:
        print("\nNo winners.\n")


if __name__ == "__main__":
    main()
