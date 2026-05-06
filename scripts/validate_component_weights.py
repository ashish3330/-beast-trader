#!/usr/bin/env python3 -B
"""
Validate the per-symbol component weights against the LONGER 180d window.

The tuner (tune_component_weights.py) optimised on 60d to capture the recent
regime. This script re-runs each (symbol, weighted) backtest on 180d and
compares to the unweighted (1.0 across the board) baseline. Anything that
backtests worse on 180d than it did at unit weights is suspicious — may
have overfit the 60d tuning window.

Output:
  backtest/results/component_weights_validate.json
  Stdout: per-symbol pnl/pf delta with REGRESSED flag for outliers.
"""
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, ALL_SYMBOLS  # noqa: E402

RESULTS = ROOT / "backtest" / "results"
AUTO_DICT_PATH = RESULTS / "component_weights_auto_dict.py"
VALIDATE_OUT = RESULTS / "component_weights_validate.json"

DAYS = int(os.environ.get("VAL_DAYS", "180"))
WORKERS = int(os.environ.get("VAL_WORKERS", str(min(4, cpu_count()))))


def _load_auto() -> dict:
    spec = importlib.util.spec_from_file_location("cw", AUTO_DICT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "COMPONENT_WEIGHTS_AUTO", {})


def _eval(job):
    symbol, weights, label = job
    try:
        params = {"component_weights": weights} if weights else None
        r = backtest_symbol(symbol, days=DAYS, params=params, verbose=False)
        return {
            "symbol": symbol, "label": label,
            "pnl": r.get("pnl", 0), "pf": r.get("pf", 0),
            "dd": r.get("dd", 0), "trades": r.get("trades", 0),
        }
    except Exception as e:
        return {"symbol": symbol, "label": label, "error": str(e)[:140],
                "pnl": 0, "pf": 0, "dd": 0, "trades": 0}


def main():
    auto = _load_auto()
    if not auto:
        print("No COMPONENT_WEIGHTS_AUTO found — nothing to validate")
        sys.exit(1)

    symbols = sorted(auto.keys())
    print(f"\nValidating {len(symbols)} symbols × ({DAYS}d baseline + tuned) = {len(symbols)*2} backtests")
    print(f"Workers: {WORKERS}\n")

    # Build full weights dict per symbol (defaults filled with 1.0)
    full_weights = {s: {**{c: 1.0 for c in [
        "ema_stack","supertrend","macd_signal","macd_hist","rsi","candle_pattern",
        "heikin_ashi","structure","breakout","momentum_vel","trend_persist"
    ]}, **auto[s]} for s in symbols}

    jobs = []
    for s in symbols:
        jobs.append((s, None, "BASE"))
        jobs.append((s, full_weights[s], "TUNED"))

    t0 = time.time()
    results: dict = {s: {} for s in symbols}
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval, jobs), 1):
            results[r["symbol"]][r["label"]] = r

    # Compare
    summary = {"better": [], "worse": [], "neutral": []}
    total_base = 0.0; total_tuned = 0.0
    for s in symbols:
        b = results[s].get("BASE", {}); t = results[s].get("TUNED", {})
        b_pnl = float(b.get("pnl", 0)); t_pnl = float(t.get("pnl", 0))
        b_pf = float(b.get("pf", 0)); t_pf = float(t.get("pf", 0))
        total_base += b_pnl; total_tuned += t_pnl
        delta = t_pnl - b_pnl
        record = {
            "symbol": s, "base_pnl": round(b_pnl, 2), "tuned_pnl": round(t_pnl, 2),
            "base_pf": round(b_pf, 2), "tuned_pf": round(t_pf, 2),
            "delta_pnl": round(delta, 2),
            "trades_base": int(b.get("trades", 0)),
            "trades_tuned": int(t.get("trades", 0)),
        }
        if delta > 5:
            summary["better"].append(record)
        elif delta < -50:
            summary["worse"].append(record)
        else:
            summary["neutral"].append(record)

    json.dump({"days": DAYS, "elapsed_s": round(time.time() - t0, 1),
               "results": results, "summary": summary,
               "total_base_pnl": round(total_base, 2),
               "total_tuned_pnl": round(total_tuned, 2)},
              open(VALIDATE_OUT, "w"), indent=2, default=str)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    print(f"  BASE  total ${total_base:+.2f}")
    print(f"  TUNED total ${total_tuned:+.2f}")
    print(f"  DELTA       ${total_tuned - total_base:+.2f}")
    print(f"  Better: {len(summary['better'])}  Worse: {len(summary['worse'])}  Neutral: {len(summary['neutral'])}")
    print(f"\nWORST regressions (>$50 down vs baseline) — candidates to drop:")
    for r in sorted(summary["worse"], key=lambda x: x["delta_pnl"])[:10]:
        print(f"  {r['symbol']:10s} pnl ${r['base_pnl']:>8.2f}->${r['tuned_pnl']:>8.2f}  "
              f"delta ${r['delta_pnl']:>+8.2f}  pf {r['base_pf']:.2f}->{r['tuned_pf']:.2f}  "
              f"n={r['trades_base']}->{r['trades_tuned']}")
    print(f"\nBEST improvements (>$5 up):")
    for r in sorted(summary["better"], key=lambda x: x["delta_pnl"], reverse=True)[:10]:
        print(f"  {r['symbol']:10s} pnl ${r['base_pnl']:>8.2f}->${r['tuned_pnl']:>8.2f}  "
              f"delta ${r['delta_pnl']:>+8.2f}  pf {r['base_pf']:.2f}->{r['tuned_pf']:.2f}")


if __name__ == "__main__":
    main()
