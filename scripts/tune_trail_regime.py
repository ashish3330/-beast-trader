#!/usr/bin/env python3 -B
"""
Per-(symbol, regime) trail profile deep tune.

Plumbed 2026-05-17: backtest reads TRAIL_OVERRIDE_REGIME (from
config.SYMBOL_REGIME_TRAIL_OVERRIDE + auto_tuned.TRAIL_OVERRIDE_REGIME_AUTO).
Live executor.py:387 already reads same dict.

Profiles are concrete (R, type, param) tuple lists, NOT names. They are
the 7 baseline profiles from tune_regime_trails.py + DEFAULT (live).
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

# Backtest tuple is (R_threshold, type, param) — same as live config (after _live_to_bt_trail mapping).
# Use the LIVE-shape tuples; backtest converts via _live_to_bt_trail in the loader.
PROFILES = {
    "ULTRA_TIGHT":  [(2.0, "lock", 1.5), (1.0, "lock", 0.7), (0.5, "lock", 0.2), (0.2, "be", 0.0)],
    "TIGHT_LOCK":   [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "DEFAULT":      [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.6),
                     (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "lock", 0.15), (0.3, "be", 0.0)],
    "TREND_LOOSE":  [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5),
                     (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "RANGE_TIGHT":  [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)],
    "AGGR_LOCK":    [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8),
                     (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "WIDE_RUNNER":  [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7),
                     (1.5, "lock", 0.5), (0.7, "be", 0.0)],
}

MIN_TRADES_PER_CELL = 25
MIN_LIFT_USD = 30.0
MIN_WF_PF = 1.4
MIN_WF_POSITIVE_FOLDS = 3
WF_FOLDS = [60, 90, 120, 150, 180]
TUNE_DAYS = int(os.getenv("TUNE_DAYS", "180"))
N_WORKERS = max(2, min(10, os.cpu_count() or 4))

DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "backtest" / "results" / f"trail_regime_tune_{DATE_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _backtest_with_trail_override(symbol, days, trail_regime_override):
    """trail_regime_override: {regime: [live-shape (R, type, param) tuples]}"""
    import importlib
    import config as cfg
    importlib.reload(cfg)
    cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
    cfg.SYMBOL_REGIME_TRAIL_OVERRIDE[symbol] = dict(trail_regime_override)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    return bt.backtest_symbol(symbol, days=days, verbose=False)


def _baseline_per_cell(symbol):
    import backtest.v5_backtest as bt
    r = bt.backtest_symbol(symbol, days=TUNE_DAYS, verbose=False)
    cells = {reg: {"pnl": 0.0, "n": 0} for reg in REGIMES}
    for t in r.get("details", []):
        reg = t.get("regime", "")
        if reg in cells:
            cells[reg]["n"] += 1
            cells[reg]["pnl"] += t["pnl"]
    for reg in cells:
        cells[reg]["pnl"] = round(cells[reg]["pnl"], 2)
    return r, cells


def _sweep_cell(args):
    symbol, target_regime, baseline_cell_pnl = args
    candidates = []
    for name, profile in PROFILES.items():
        override = {target_regime: profile}
        try:
            r = _backtest_with_trail_override(symbol, days=TUNE_DAYS, trail_regime_override=override)
        except Exception:
            continue
        cell_trades = [t for t in r.get("details", []) if t.get("regime") == target_regime]
        if len(cell_trades) < 8:
            continue
        cell_pnl = round(sum(t["pnl"] for t in cell_trades), 2)
        gp = sum(t["pnl"] for t in cell_trades if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell_trades if t["pnl"] < 0) or 0.01
        candidates.append({
            "profile_name": name, "profile": profile,
            "cell_pnl": cell_pnl, "cell_pf": round(gp / gl, 2),
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


def _wf_validate(symbol, target_regime, profile):
    folds = []
    for d_days in WF_FOLDS:
        try:
            r = _backtest_with_trail_override(symbol, days=d_days, trail_regime_override={target_regime: profile})
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
    cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
    for s, rd in combined.items():
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE.setdefault(s, {}).update(rd)
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
    print(f"\nPer-(symbol, regime) trail tune — {DATE_TAG}")
    print(f"Profiles: {list(PROFILES)}")
    print(f"Workers: {N_WORKERS}, tune days: {TUNE_DAYS}\n")
    t0 = time.time()

    print("[A] Baseline...")
    baselines = {}
    baseline_total = 0
    for sym in LIVE_SYMBOLS:
        r, cells = _baseline_per_cell(sym)
        baselines[sym] = {"total_pnl": r.get("pnl", 0), "cells": cells}
        baseline_total += r.get("pnl", 0)
        print(f"  {sym:12s} ${r.get('pnl', 0):>7.0f}  cells={[(reg, cells[reg]['n']) for reg in REGIMES if cells[reg]['n'] > 0]}")
    print(f"  baseline {time.time() - t0:.0f}s  universe ${baseline_total:.0f}\n")

    sweep_jobs = []
    for sym in LIVE_SYMBOLS:
        for reg in REGIMES:
            cell = baselines[sym]["cells"][reg]
            if cell["n"] < MIN_TRADES_PER_CELL:
                continue
            sweep_jobs.append((sym, reg, cell["pnl"]))
    print(f"[B] Sweeping {len(sweep_jobs)} cells × {len(PROFILES)} profiles = {len(sweep_jobs)*len(PROFILES)} backtests")
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
        wf = _wf_validate(c["symbol"], c["regime"], c["best"]["profile"])
        if wf is None:
            continue
        if wf["avg_pf"] >= MIN_WF_PF and wf["pos_folds"] >= MIN_WF_POSITIVE_FOLDS:
            winners.append({**c, "wf": wf})
            print(f"  ✓ {c['symbol']} {c['regime']} → {c['best']['profile_name']:12s} "
                  f"Δ${c['best']['cell_pnl'] - c['baseline_pnl']:+.0f} "
                  f"WF PF {wf['avg_pf']} {wf['pos_folds']}/5")
        else:
            print(f"  ✗ {c['symbol']} {c['regime']} → {c['best']['profile_name']:12s} "
                  f"WF FAIL PF {wf['avg_pf']} {wf['pos_folds']}/5")

    print(f"\n[D] Full-universe verifier on {len(winners)} winners...")
    combined = {}
    for w in winners:
        combined.setdefault(w["symbol"], {})[w["regime"]] = w["best"]["profile"]
    combined_total, per_sym = _full_universe_verifier(combined)
    delta = combined_total - baseline_total
    print(f"  Universe ${baseline_total:.0f} → ${combined_total:.0f}  Δ${delta:+.0f}")

    accepted = list(winners)
    if delta < 0:
        print(f"  REGRESSION — iterative drop...")
        while accepted and delta < 0:
            best_drop = None
            best_gain = -1e9
            for w in accepted:
                test = {}
                for ww in accepted:
                    if ww is w:
                        continue
                    test.setdefault(ww["symbol"], {})[ww["regime"]] = ww["best"]["profile"]
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
                combined.setdefault(ww["symbol"], {})[ww["regime"]] = ww["best"]["profile"]
            combined_total, per_sym = _full_universe_verifier(combined)
            delta = combined_total - baseline_total

    summary = {
        "captured_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "baseline_total": baseline_total,
        "candidates": candidates, "wf_winners": winners,
        "accepted_winners": accepted, "combined_overrides": combined,
        "combined_universe_total": combined_total,
        "combined_universe_delta": delta,
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  saved: {OUT_DIR}/_summary.json")

    if accepted:
        print("\n" + "=" * 70)
        print("ACCEPTED WINNERS")
        print("=" * 70)
        for w in sorted(accepted, key=lambda x: -(x["best"]["cell_pnl"] - x["baseline_pnl"])):
            d = w["best"]["cell_pnl"] - w["baseline_pnl"]
            print(f"  {w['symbol']:10s} {w['regime']:10s} → {w['best']['profile_name']:12s}  "
                  f"Δ${d:>+6.0f}  WF PF {w['wf']['avg_pf']}  {w['wf']['pos_folds']}/5")
        print(f"\nUniverse ${baseline_total:.0f} → ${combined_total:.0f}  Δ${delta:+.0f}\n")


if __name__ == "__main__":
    main()
