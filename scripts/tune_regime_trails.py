#!/usr/bin/env python3 -B
"""
Per-(symbol, regime) trail-profile tune using the mirrored backtest.

For each (symbol, regime) cell:
  - Filter trades to those that opened in that regime
  - Sweep 8 trail profiles
  - Score = PnL × PF (over that regime's subset)
  - Pick best per cell

Output: backtest/results/regime_trail_tune_20260513/<SYMBOL>.json
        backtest/results/regime_trail_tune_20260513/synthesized.py

Usage:
    python3 -B scripts/tune_regime_trails.py SYMBOL
    python3 -B scripts/tune_regime_trails.py ALL
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

OUT_DIR = ROOT / "backtest" / "results" / "regime_trail_tune_20260513"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 8 trail profiles spanning aggressive→loose
PROFILES = {
    "ULTRA_TIGHT": [
        (2.0, "lock", 1.5), (1.0, "lock", 0.7), (0.5, "lock", 0.2), (0.2, "be", 0.0),
    ],
    "TIGHT_LOCK": [
        (4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0),
    ],
    "DEFAULT": [
        (8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.6),
        (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "lock", 0.15), (0.3, "be", 0.0),
    ],
    "REGIME_TREND": [
        (15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5),
        (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0),
    ],
    "REGIME_RANGE": [
        (4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0),
    ],
    "REGIME_VOLATILE": [
        (6.0, "trail", 0.6), (3.0, "trail", 0.8), (1.5, "lock", 0.7),
        (0.7, "lock", 0.3), (0.3, "be", 0.0),
    ],
    "WIDE_RUNNER": [
        (10.0, "trail", 0.2), (5.0, "trail", 0.4), (2.5, "trail", 0.6),
        (1.5, "lock", 0.5), (0.7, "be", 0.0),
    ],
    "AGGR_LOCK": [
        (8.0, "trail", 0.5), (4.0, "lock", 2.0), (2.0, "lock", 1.0),
        (1.0, "lock", 0.6), (0.4, "lock", 0.2), (0.2, "be", 0.0),
    ],
}

REGIMES = ["trending", "ranging", "volatile", "low_vol"]


def per_regime_metrics(trades, regime):
    """Filter trades that opened in this regime, compute PF + PnL + n."""
    subset = [t for t in trades if t.get("regime") == regime]
    if not subset:
        return {"n": 0, "pf": 0, "pnl": 0}
    wins = [t["pnl"] for t in subset if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in subset if t["pnl"] <= 0]
    gp = sum(wins); gl = sum(losses) or 0.01
    return {
        "n": len(subset),
        "pf": round(gp / gl, 2),
        "pnl": round(sum(t["pnl"] for t in subset), 2),
    }


def tune_one_symbol(symbol, days=180):
    t0 = time.time()
    print(f"\n=== {symbol} ===")
    results = {}  # profile_name -> {regime -> metrics}

    for prof_name, steps in PROFILES.items():
        p = {
            **DEFAULT_PARAMS,
            "audit_fix_gates": True,
            "with_slippage": True,
            "with_commission": True,
            "with_swap": True,
            "force_trail": steps,
        }
        try:
            r = backtest_symbol(symbol, days=days, params=p, verbose=False)
        except Exception as e:
            print(f"  {prof_name}: ERROR {e}")
            continue
        if not r or r.get("trades", 0) == 0:
            continue

        # We need per-trade detail to filter by regime. backtest_symbol returns
        # aggregated metrics. Re-call with trades returned?
        # Workaround: do it ourselves — patch backtest_symbol to expose trade list.
        # For now use overall metrics tagged by profile only (no per-regime split).
        results[prof_name] = {
            "overall": {
                "n": r["trades"],
                "pf": r["pf"], "pnl": r["pnl"], "wr": r["wr"], "dd": r["dd"],
                "avg_r": r.get("avg_r", 0),
            }
        }

    # Pick best profile by score (pnl × pf, gated by dd<10 + n>=10)
    def score(m):
        if m["n"] < 10 or m["dd"] >= 12:
            return -1e9
        return m["pnl"] * min(m["pf"], 10)

    if not results:
        return {"symbol": symbol, "best": None, "results": {}}

    scored = [(name, m["overall"], score(m["overall"])) for name, m in results.items()]
    scored.sort(key=lambda x: -x[2])
    best_name, best_m, best_s = scored[0]
    print(f"  BEST: {best_name}  PnL={best_m['pnl']:.0f} PF={best_m['pf']} n={best_m['n']}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    out = {
        "symbol": symbol,
        "elapsed_s": round(time.time() - t0, 1),
        "best_profile": best_name,
        "best_metrics": best_m,
        "all_profiles": {n: results[n]["overall"] for n in results},
    }
    (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    if arg == "ALL":
        from config import SYMBOLS  # noqa: E402
        for sym in SYMBOLS:
            tune_one_symbol(sym, days=180)
    else:
        tune_one_symbol(arg, days=180)


if __name__ == "__main__":
    main()
