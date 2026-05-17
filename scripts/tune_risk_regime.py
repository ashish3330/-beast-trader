#!/usr/bin/env python3 -B
"""
Per-(symbol, regime) risk-per-trade cap deep tune.

Plumbed 2026-05-17 (config.SYMBOL_RISK_CAP_REGIME + backtest mirror).
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
RISK_GRID = [0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.0]

MIN_TRADES_PER_CELL = 25
MIN_LIFT_USD = 30.0
MIN_WF_PF = 1.4
MIN_WF_POSITIVE_FOLDS = 3
WF_FOLDS = [60, 90, 120, 150, 180]
TUNE_DAYS = int(os.getenv("TUNE_DAYS", "180"))
N_WORKERS = max(2, min(10, os.cpu_count() or 4))
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "backtest" / "results" / f"risk_regime_tune_{DATE_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _backtest_with_risk_override(symbol, days, risk_regime_override):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    cfg.SYMBOL_RISK_CAP_REGIME = dict(cfg.SYMBOL_RISK_CAP_REGIME)
    cfg.SYMBOL_RISK_CAP_REGIME[symbol] = dict(risk_regime_override)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    return bt.backtest_symbol(symbol, days=days, verbose=False)


def _baseline(symbol):
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
    symbol, target_regime, baseline_pnl = args
    cands = []
    for r_val in RISK_GRID:
        override = {target_regime: r_val}
        try:
            r = _backtest_with_risk_override(symbol, days=TUNE_DAYS, risk_regime_override=override)
        except Exception:
            continue
        cell_trades = [t for t in r.get("details", []) if t.get("regime") == target_regime]
        if len(cell_trades) < 8:
            continue
        cell_pnl = round(sum(t["pnl"] for t in cell_trades), 2)
        gp = sum(t["pnl"] for t in cell_trades if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell_trades if t["pnl"] < 0) or 0.01
        cands.append({"risk": r_val, "cell_pnl": cell_pnl,
                      "cell_pf": round(gp / gl, 2), "n": len(cell_trades)})
    if not cands:
        return None
    best = max(cands, key=lambda c: c["cell_pnl"])
    if best["cell_pnl"] - baseline_pnl < MIN_LIFT_USD:
        return None
    return {"symbol": symbol, "regime": target_regime, "baseline_pnl": baseline_pnl, "best": best, "all": cands}


def _wf(symbol, regime, risk):
    folds = []
    for d in WF_FOLDS:
        try:
            r = _backtest_with_risk_override(symbol, days=d, risk_regime_override={regime: risk})
        except Exception:
            return None
        cell = [t for t in r.get("details", []) if t.get("regime") == regime]
        if not cell:
            folds.append({"days": d, "pnl": 0, "pf": 0, "n": 0})
            continue
        gp = sum(t["pnl"] for t in cell if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell if t["pnl"] < 0) or 0.01
        folds.append({"days": d, "pnl": round(sum(t["pnl"] for t in cell), 2),
                      "pf": round(gp / gl, 2), "n": len(cell)})
    avg_pf = round(sum(f["pf"] for f in folds) / max(1, len(folds)), 2)
    pos = sum(1 for f in folds if f["pnl"] > 0)
    return {"folds": folds, "avg_pf": avg_pf, "pos_folds": pos}


def _verifier(combined):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    cfg.SYMBOL_RISK_CAP_REGIME = dict(cfg.SYMBOL_RISK_CAP_REGIME)
    for s, rd in combined.items():
        cfg.SYMBOL_RISK_CAP_REGIME.setdefault(s, {}).update(rd)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    t = 0
    per = {}
    for sym in LIVE_SYMBOLS:
        r = bt.backtest_symbol(sym, days=TUNE_DAYS, verbose=False)
        per[sym] = r.get("pnl", 0)
        t += r.get("pnl", 0)
    return round(t, 2), per


def main():
    print(f"\nPer-(symbol, regime) risk-cap tune — {DATE_TAG}\n")
    t0 = time.time()
    print("[A] Baseline...")
    baselines = {}
    btot = 0
    for sym in LIVE_SYMBOLS:
        r, cells = _baseline(sym)
        baselines[sym] = {"total_pnl": r.get("pnl", 0), "cells": cells}
        btot += r.get("pnl", 0)
        print(f"  {sym:12s} ${r.get('pnl', 0):>7.0f}")
    print(f"  baseline {time.time() - t0:.0f}s universe ${btot:.0f}\n")

    jobs = []
    for sym in LIVE_SYMBOLS:
        for reg in REGIMES:
            cell = baselines[sym]["cells"][reg]
            if cell["n"] < MIN_TRADES_PER_CELL:
                continue
            jobs.append((sym, reg, cell["pnl"]))
    print(f"[B] {len(jobs)} cells × {len(RISK_GRID)} risks = {len(jobs)*len(RISK_GRID)} backtests")
    cands = []
    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_sweep_cell, jobs), 1):
            if res:
                cands.append(res)
            if i % 5 == 0 or i == len(jobs):
                print(f"  sweep {i}/{len(jobs)} ({time.time() - t0:.0f}s)")
    print(f"  [B] candidates: {len(cands)}\n")

    print(f"[C] WF...")
    winners = []
    for c in cands:
        wf = _wf(c["symbol"], c["regime"], c["best"]["risk"])
        if wf is None:
            continue
        if wf["avg_pf"] >= MIN_WF_PF and wf["pos_folds"] >= MIN_WF_POSITIVE_FOLDS:
            winners.append({**c, "wf": wf})
            print(f"  ✓ {c['symbol']} {c['regime']} risk={c['best']['risk']} Δ${c['best']['cell_pnl']-c['baseline_pnl']:+.0f} WF PF {wf['avg_pf']} {wf['pos_folds']}/5")
        else:
            print(f"  ✗ {c['symbol']} {c['regime']} risk={c['best']['risk']} WF FAIL")

    print(f"\n[D] Verifier on {len(winners)}...")
    combined = {}
    for w in winners:
        combined.setdefault(w["symbol"], {})[w["regime"]] = w["best"]["risk"]
    ctot, _ = _verifier(combined)
    delta = ctot - btot
    print(f"  Universe ${btot:.0f} → ${ctot:.0f}  Δ${delta:+.0f}")

    accepted = list(winners)
    while accepted and delta < 0:
        best_drop = None
        best_gain = -1e9
        for w in accepted:
            test = {}
            for ww in accepted:
                if ww is w:
                    continue
                test.setdefault(ww["symbol"], {})[ww["regime"]] = ww["best"]["risk"]
            t_total, _ = _verifier(test)
            g = t_total - ctot
            if g > best_gain:
                best_gain = g
                best_drop = w
        if best_drop is None or best_gain <= 0:
            break
        print(f"  drop {best_drop['symbol']} {best_drop['regime']} → +${best_gain:.0f}")
        accepted = [w for w in accepted if w is not best_drop]
        combined = {}
        for ww in accepted:
            combined.setdefault(ww["symbol"], {})[ww["regime"]] = ww["best"]["risk"]
        ctot, _ = _verifier(combined)
        delta = ctot - btot

    summary = {
        "captured_at": datetime.now().isoformat(),
        "baseline": btot, "candidates": cands, "wf_winners": winners,
        "accepted": accepted, "combined": combined, "ctot": ctot, "delta": delta,
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if accepted:
        print("\n=== ACCEPTED ===")
        for w in sorted(accepted, key=lambda x: -(x["best"]["cell_pnl"] - x["baseline_pnl"])):
            d = w["best"]["cell_pnl"] - w["baseline_pnl"]
            print(f"  {w['symbol']:10s} {w['regime']:10s} risk={w['best']['risk']:.1f}  Δ${d:+.0f}  WF PF {w['wf']['avg_pf']}  {w['wf']['pos_folds']}/5")
        print(f"\nUniverse ${btot:.0f} → ${ctot:.0f}  Δ${delta:+.0f}\n")


if __name__ == "__main__":
    main()
