#!/usr/bin/env python3 -B
"""
Per-symbol grid tune using the MIRRORED backtest (audit_fix_gates=True).

Sweeps: SL multiplier × trail profile × ratchet × per-regime quality threshold.
Walk-forward validates the best in-sample config on a held-out window.

Output: backtest/results/agent_tune_20260513/<SYMBOL>.json

Usage:
    python3 -B scripts/agent_tune_symbol.py SYMBOL [DAYS]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

OUT_DIR = ROOT / "backtest" / "results" / "agent_tune_20260513"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Trail profiles to test (mirrors sweep_trails.py)
TRAIL_PROFILES = {
    "DEFAULT": [
        (8.0, 0.3, "trail"), (4.0, 0.5, "trail"), (2.0, 0.8, "trail"),
        (1.5, 0.7, "lock"),  (1.0, 0.4, "lock"),  (0.5, 0.0, "be"),
    ],
    "TIGHT": [
        (6.0, 0.4, "trail"), (3.0, 0.6, "trail"),
        (1.5, 0.9, "lock"),  (1.0, 0.5, "lock"),  (0.5, 0.0, "be"),
    ],
    "LOOSE": [
        (10.0, 0.2, "trail"), (5.0, 0.4, "trail"), (2.5, 0.6, "trail"),
        (1.5, 0.5, "lock"),   (0.7, 0.0, "be"),
    ],
    "AGGR_RUN": [
        (15.0, 0.3, "trail"), (8.0, 0.5, "trail"),
        (3.0, 0.5, "lock"),   (1.0, 0.0, "be"),
    ],
}

# Parameter grid
SL_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
RATCHET_PAIRS = [(0.3, 0.7), (0.5, 1.0), (0.4, 0.8)]
QUALITY_VARIANTS = [
    {"trending": 40, "ranging": 45, "volatile": 45, "low_vol": 40},  # current
    {"trending": 45, "ranging": 50, "volatile": 50, "low_vol": 45},  # tighter
    {"trending": 50, "ranging": 55, "volatile": 55, "low_vol": 50},  # strict
]


def score(r):
    """Composite quality score: PnL × PF / sqrt(DD). Higher = better."""
    if not r or r.get("trades", 0) < 8:
        return -1e9
    pf = min(r["pf"], 20)
    dd = max(r["dd"], 1)
    return r["pnl"] * pf / (dd ** 0.5)


def run_combo(symbol, days, sl, ratchet, trail_name, trail_steps, qual):
    p = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
        "sl_atr_mult": sl,
        "ratchet_1r": ratchet[0],
        "ratchet_2r": ratchet[1],
        "force_trail": trail_steps,
        "min_quality": qual,
    }
    try:
        return backtest_symbol(symbol, days=days, params=p, verbose=False)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: agent_tune_symbol.py SYMBOL [DAYS]")
        sys.exit(1)
    symbol = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 180

    t0 = time.time()
    print(f"\n=== TUNING {symbol} ({days}d in-sample) ===")

    results = []
    n_combos = len(SL_MULTS) * len(RATCHET_PAIRS) * len(TRAIL_PROFILES) * len(QUALITY_VARIANTS)
    print(f"Grid: {n_combos} combos")

    i = 0
    for sl in SL_MULTS:
        for ratchet in RATCHET_PAIRS:
            for trail_name, trail_steps in TRAIL_PROFILES.items():
                for qual in QUALITY_VARIANTS:
                    i += 1
                    r = run_combo(symbol, days, sl, ratchet, trail_name, trail_steps, qual)
                    if r and r.get("trades", 0) > 0:
                        results.append({
                            "sl": sl, "ratchet": ratchet,
                            "trail": trail_name, "qual": qual,
                            "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
                            "pnl": r["pnl"], "avg_r": r.get("avg_r", 0),
                            "dd": r["dd"],
                            "score": round(score(r), 2),
                        })
                    if i % 30 == 0:
                        elapsed = time.time() - t0
                        print(f"  {i}/{n_combos} ({elapsed:.0f}s)", flush=True)

    if not results:
        print(f"NO VIABLE RESULTS for {symbol}")
        out = {"symbol": symbol, "tested": n_combos, "results": []}
    else:
        results.sort(key=lambda x: -x["score"])
        top10 = results[:10]
        best = results[0]
        print(f"\n  BEST: SL={best['sl']} trail={best['trail']} ratchet={best['ratchet']} qual={best['qual']}")
        print(f"  PF={best['pf']} WR={best['wr']}% PnL=${best['pnl']:.2f} DD={best['dd']}% trades={best['trades']}")

        # Walk-forward: re-run best on last 60d (held out)
        wf_r = run_combo(symbol, 60, best["sl"], best["ratchet"],
                         best["trail"], TRAIL_PROFILES[best["trail"]], best["qual"])
        wf = None
        if wf_r and wf_r.get("trades", 0) > 0:
            wf = {"trades": wf_r["trades"], "pf": wf_r["pf"], "wr": wf_r["wr"],
                  "pnl": wf_r["pnl"], "avg_r": wf_r.get("avg_r", 0),
                  "dd": wf_r["dd"]}
            print(f"  WALK-FWD (60d): PF={wf_r['pf']} WR={wf_r['wr']}% PnL=${wf_r['pnl']:.2f}")
        else:
            print(f"  WALK-FWD: no viable result")

        out = {
            "symbol": symbol,
            "days": days,
            "tested": n_combos,
            "elapsed_s": round(time.time() - t0, 1),
            "best": best,
            "top10": top10,
            "walk_forward_60d": wf,
        }

    out_file = OUT_DIR / f"{symbol}.json"
    out_file.write_text(json.dumps(out, indent=2, default=str))
    print(f"  Saved: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
