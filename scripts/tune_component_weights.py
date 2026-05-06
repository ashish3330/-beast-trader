#!/usr/bin/env python3 -B
"""
Per-symbol component-weight tuner.

For each symbol × scoring component, run a backtest at 5 weight levels
(0.0, 0.5, 1.0, 1.5, 2.0) holding all other components at 1.0. Pick the
weight that maximises a robust objective (pnl × pf / sqrt(dd)) per
component per symbol.

Why univariate (not full grid):
  - 11-dim grid at 5 levels = 5^11 = 48M combos / sym → infeasible.
  - Univariate keeps each component's optimum interpretable + statistically
    robust given limited sample size per backtest.
  - Live RL fine-tunes from these starting weights post-deploy.

Output:
  backtest/results/component_weights_auto_dict.py
    COMPONENT_WEIGHTS_AUTO = {
        symbol: {component: best_weight, ...},
        ...
    }

Trader picks them up via rl_learner._weights initialisation (separate commit).
"""
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
RESULTS.mkdir(exist_ok=True, parents=True)

DAYS = int(os.environ.get("TUNE_DAYS", "60"))   # 60d default — recent regime
WORKERS = int(os.environ.get("TUNE_WORKERS", str(min(4, cpu_count()))))
LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]
COMPONENTS = [
    "ema_stack", "supertrend", "macd_signal", "macd_hist",
    "rsi", "candle_pattern", "heikin_ashi", "structure",
    "breakout", "momentum_vel", "trend_persist",
]

_env_syms = os.environ.get("TUNE_SYMBOLS", "").strip()
if _env_syms:
    wanted = {s.strip() for s in _env_syms.split(",") if s.strip()}
    SYMBOLS = sorted(s for s in ALL_SYMBOLS.keys() if s in wanted)
else:
    SYMBOLS = sorted(ALL_SYMBOLS.keys())


def _score(metrics: dict) -> float:
    """Robust objective. Higher = better."""
    pnl = float(metrics.get("pnl", 0))
    pf = float(metrics.get("pf", 0)) or 0.01
    dd = float(metrics.get("dd", 0)) or 0.01
    n = int(metrics.get("trades", 0))
    if n < 5:
        return -1e9                 # disqualify tiny samples
    return pnl * pf / (max(dd, 1.0) ** 0.5)


def _eval_one(job: tuple) -> dict:
    """Run one backtest with weights {comp: level, others: 1.0}."""
    symbol, comp, level = job
    weights = {c: 1.0 for c in COMPONENTS}
    weights[comp] = level
    try:
        r = backtest_symbol(
            symbol, days=DAYS,
            params={"component_weights": weights},
            verbose=False,
        )
        return {
            "symbol": symbol, "component": comp, "level": level,
            "pnl": r.get("pnl", 0), "pf": r.get("pf", 0),
            "dd": r.get("dd", 0), "trades": r.get("trades", 0),
            "score": _score(r),
        }
    except Exception as e:
        return {
            "symbol": symbol, "component": comp, "level": level,
            "pnl": 0, "pf": 0, "dd": 0, "trades": 0, "score": -1e9,
            "error": str(e)[:120],
        }


def main() -> None:
    jobs = [(s, c, lv) for s in SYMBOLS for c in COMPONENTS for lv in LEVELS]
    print(f"\nComponent-weight tune: {len(SYMBOLS)} symbols × {len(COMPONENTS)} components "
          f"× {len(LEVELS)} levels = {len(jobs)} backtests")
    print(f"Window: {DAYS}d  Workers: {WORKERS}\n")

    t0 = time.time()
    results = []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_one, jobs), 1):
            results.append(r)
            if i % 100 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(jobs) - i)
                print(f"  [{i:>4}/{len(jobs)}] elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m")

    # Pick best weight per (symbol, component)
    best: dict = {s: {c: 1.0 for c in COMPONENTS} for s in SYMBOLS}
    summary: list = []
    for sym in SYMBOLS:
        for comp in COMPONENTS:
            cands = [r for r in results
                     if r["symbol"] == sym and r["component"] == comp]
            if not cands:
                continue
            base = next((r for r in cands if r["level"] == 1.0), None)
            best_r = max(cands, key=lambda r: r["score"])
            if best_r["level"] != 1.0 and base and best_r["score"] > base["score"] * 1.03:
                best[sym][comp] = best_r["level"]
                summary.append({
                    "symbol": sym, "component": comp,
                    "from": 1.0, "to": best_r["level"],
                    "base_pnl": base["pnl"], "best_pnl": best_r["pnl"],
                    "base_pf": base["pf"], "best_pf": best_r["pf"],
                    "trades": best_r["trades"],
                })

    # Write JSON of every backtest + auto_dict for synthesize
    json.dump(
        {"days": DAYS, "elapsed_s": round(time.time() - t0, 1),
         "results": results, "summary": summary},
        open(RESULTS / "component_weights_tune.json", "w"),
        indent=2, default=str,
    )
    auto_path = RESULTS / "component_weights_auto_dict.py"
    with open(auto_path, "w") as f:
        f.write('"""AUTO-GENERATED by scripts/tune_component_weights.py.\n')
        f.write(f'Window: {DAYS}d  Symbols: {len(SYMBOLS)}  Acceptance: best beats base by >3%.\n"""\n\n')
        f.write("COMPONENT_WEIGHTS_AUTO = {\n")
        for sym in sorted(best):
            non_default = {c: v for c, v in best[sym].items() if v != 1.0}
            if not non_default:
                continue
            f.write(f"    {sym!r}: {{\n")
            for c, v in sorted(non_default.items()):
                f.write(f"        {c!r}: {v},\n")
            f.write("    },\n")
        f.write("}\n")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    print(f"  Wrote {auto_path}")
    print(f"  Improvements found on {len(summary)} (symbol, component) pairs")
    if summary:
        print(f"  Top 10 by PnL lift:")
        for s in sorted(summary, key=lambda x: x["best_pnl"] - x["base_pnl"], reverse=True)[:10]:
            print(f"    {s['symbol']:10s} {s['component']:15s} "
                  f"{s['from']}->{s['to']:.1f}  pnl ${s['base_pnl']:.0f}->${s['best_pnl']:.0f}  "
                  f"pf {s['base_pf']:.2f}->{s['best_pf']:.2f}  n={s['trades']}")


if __name__ == "__main__":
    main()
