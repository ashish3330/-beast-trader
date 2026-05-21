#!/usr/bin/env python3 -B
"""
SWI20.r focused optimization — sweep SL × trail Cartesian with wider-BE profiles.

User complaint 2026-05-18: SWI20 LONG opened at 13240.50, peaked at +0.43R,
trail's BE@0.2R fired, price reverted, exit at BE (~$0). $8 unrealized
profit reverted because trail was too tight on the BE step.

Test BE thresholds: 0.2 (current ULTRA_TIGHT) up to 0.7R.
Test SL multipliers: 0.7 to 3.0.

For each (SL, trail) combo:
  - Run 180d backtest
  - Score = PnL × PF over SWI20.r trades
  - 5-fold WF validate top combo
"""
import json, os, sys, time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SYM = "XAUUSD"
SL_GRID = [0.7, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
TRAIL_PROFILES = {
    "ULTRA_TIGHT_BE02":     [(2.0, "lock", 1.5), (1.0, "lock", 0.7), (0.5, "lock", 0.2), (0.2, "be", 0.0)],
    "TIGHT_LOCK_BE03":      [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "DEFAULT_BE03":         [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.6),
                             (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "lock", 0.15), (0.3, "be", 0.0)],
    "WIDER_BE05":           [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "lock", 1.0),
                             (1.0, "lock", 0.5), (0.5, "be", 0.0)],
    "MUCH_WIDER_BE07":      [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "lock", 1.0),
                             (1.0, "lock", 0.5), (0.7, "be", 0.0)],
    "WIDE_RUNNER_BE07":     [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7),
                             (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    "RUNNER_NO_BE":         [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.0, "trail", 0.7),
                             (1.0, "trail", 0.8)],  # pure trail, no BE
}

WF_FOLDS = [60, 90, 120, 150, 180]
TUNE_DAYS = int(os.getenv("TUNE_DAYS", "180"))
MIN_TRADES = 15
MIN_WF_PF = 1.4
MIN_WF_POSITIVE_FOLDS = 3

DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "backtest" / "results" / f"xauusd_optim_{DATE_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bt(days, sl_mult, trail):
    """Run SWI20 backtest with both SL and trail overridden across ALL regimes."""
    import importlib
    import config as cfg
    importlib.reload(cfg)
    # SL: apply at per-symbol level (works across all regimes)
    cfg.SYMBOL_ATR_SL_OVERRIDE = dict(cfg.SYMBOL_ATR_SL_OVERRIDE)
    cfg.SYMBOL_ATR_SL_OVERRIDE[SYM] = sl_mult
    # Clear per-regime SL so it doesn't override the per-symbol value
    cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
    cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME[SYM] = {}
    # Trail: apply to all 4 regimes
    cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
    cfg.SYMBOL_REGIME_TRAIL_OVERRIDE[SYM] = {
        "trending": trail, "ranging": trail, "volatile": trail, "low_vol": trail,
    }
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    return bt.backtest_symbol(SYM, days=days, verbose=False)


def _sweep_one(args):
    sl, name, trail = args
    try:
        r = _bt(TUNE_DAYS, sl, trail)
    except Exception as e:
        return None
    n = r.get("trades", 0)
    if n < MIN_TRADES:
        return None
    return {
        "sl_mult": sl, "trail_name": name,
        "pnl": r.get("pnl", 0), "pf": r.get("pf", 0),
        "wr": r.get("wr", 0), "n": n,
        "dd": r.get("dd", 0), "avg_r": r.get("avg_r", 0),
        "avg_peak_r": r.get("avg_peak_r", 0),
        "avg_giveback": r.get("avg_giveback", 0),
    }


def _wf(sl, trail):
    folds = []
    for d in WF_FOLDS:
        try:
            r = _bt(d, sl, trail)
        except Exception:
            return None
        folds.append({"days": d, "pnl": r.get("pnl", 0), "pf": r.get("pf", 0), "n": r.get("trades", 0)})
    avg_pf = round(sum(f["pf"] for f in folds) / max(1, len(folds)), 2)
    pos = sum(1 for f in folds if f["pnl"] > 0)
    return {"folds": folds, "avg_pf": avg_pf, "pos_folds": pos}


def main():
    print(f"\nSWI20.r SL × Trail Cartesian optimization — {DATE_TAG}")
    print(f"SL grid: {SL_GRID}  ({len(SL_GRID)} values)")
    print(f"Trail profiles: {list(TRAIL_PROFILES)}  ({len(TRAIL_PROFILES)} profiles)")
    print(f"Total: {len(SL_GRID) * len(TRAIL_PROFILES)} backtests on {TUNE_DAYS}d window\n")
    t0 = time.time()

    # Baseline (current config)
    r0 = _bt(TUNE_DAYS, 3.0, TRAIL_PROFILES["ULTRA_TIGHT_BE02"])
    print(f"[A] Baseline (current: SL=3.0, ULTRA_TIGHT_BE02): {r0.get('trades', 0)} trades, "
          f"PnL ${r0.get('pnl', 0)}, PF {r0.get('pf', 0)}, "
          f"avg_peak {r0.get('avg_peak_r', 0)}R, giveback {r0.get('avg_giveback', 0)}R\n")

    print("[B] Cartesian sweep...")
    jobs = [(sl, name, trail) for sl in SL_GRID for name, trail in TRAIL_PROFILES.items()]
    workers = max(2, min(8, os.cpu_count() or 4))
    results = []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_sweep_one, jobs), 1):
            if res is not None:
                results.append(res)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)")
    print(f"  [B] {len(results)} valid results\n")

    # Sort by PnL desc, top 10
    top10 = sorted(results, key=lambda x: -x["pnl"])[:10]
    print("[C] Top 10 by PnL:")
    print(f"  {'SL':>5} {'Trail':22} {'Trades':>7} {'WR%':>6} {'PF':>6} {'PnL':>10} {'Peak':>6} {'Give':>6}")
    for r in top10:
        print(f"  {r['sl_mult']:>5.2f} {r['trail_name']:22} "
              f"{r['n']:>7d} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
              f"${r['pnl']:>+8.0f} {r['avg_peak_r']:>5.2f}R {r['avg_giveback']:>5.2f}R")
    print()

    # WF validate top 5 of those
    print("[D] WF validation on top 5...")
    winners = []
    for r in top10[:5]:
        wf = _wf(r["sl_mult"], TRAIL_PROFILES[r["trail_name"]])
        if wf is None:
            continue
        ok = wf["avg_pf"] >= MIN_WF_PF and wf["pos_folds"] >= MIN_WF_POSITIVE_FOLDS
        flag = "✓" if ok else "✗"
        print(f"  {flag} SL={r['sl_mult']} {r['trail_name']:22} "
              f"180d PnL ${r['pnl']:>+.0f} WF PF {wf['avg_pf']} {wf['pos_folds']}/5")
        if ok:
            winners.append({**r, "wf": wf})

    if winners:
        best = winners[0]
        print(f"\nBEST: SL={best['sl_mult']} {best['trail_name']} "
              f"PnL ${best['pnl']:+.0f} PF {best['pf']} WF PF {best['wf']['avg_pf']}")
        print(f"  vs baseline PnL ${r0.get('pnl', 0):+.0f}: "
              f"Δ${best['pnl'] - r0.get('pnl', 0):+.0f}")
        # Save trail profile dict for application
        out = {
            "best_sl": best["sl_mult"], "best_trail_name": best["trail_name"],
            "best_trail_profile": TRAIL_PROFILES[best["trail_name"]],
            "baseline_pnl": r0.get("pnl", 0), "best_pnl": best["pnl"],
            "all_results": results, "winners": winners,
        }
        (OUT_DIR / "_summary.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  saved: {OUT_DIR}/_summary.json")
    else:
        print("\nNo winners passed WF gates.")

    print(f"  total elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
