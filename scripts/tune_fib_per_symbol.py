#!/usr/bin/env python3 -B
"""
PHASE 6 — PER-SYMBOL FIB TUNE (entry filter + TP exit).

Tests fib as both:
  A. ENTRY FILTER — soft boost score when entering in golden zone
  B. EXIT TP — TP1 = max(R*2, fib_1.272 of swing)

Per-symbol params:
  swing_lookback: 30, 50, 75, 100 bars
  zone_lo:        0.382, 0.500  (lower golden bound)
  zone_hi:        0.618, 0.65, 0.786 (upper golden bound)
  weight:         0.0 (off), 0.5, 1.0, 1.5
  use_as_filter:  True/False (hard reject vs soft boost)

Total combos: 4 × 2 × 3 × 4 × 2 = 192 per symbol → with 5-fold WF on top 3.

Decision rule:
  - Per-symbol WINNER if WF avg PF > 1.3 AND ≥3/5 folds positive
                       AND delta vs no-fib baseline > +$30
  - Else: KEEP no-fib for that symbol

Output: backtest/results/fib_per_symbol_20260514/<SYMBOL>.json
"""
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

OUT_DIR = ROOT / "backtest" / "results" / "fib_per_symbol_20260514"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SWING_LOOKBACK = [30, 50, 75, 100]
ZONE_LO = [0.382, 0.500]
ZONE_HI = [0.618, 0.65, 0.786]
WEIGHT = [0.5, 1.0, 1.5]
USE_AS_FILTER = [False, True]


def calmar(pnl_pct, dd_pct, days):
    if dd_pct <= 0 or days <= 0: return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / dd_pct


def score(r, days):
    if not r or r.get("trades", 0) < 10: return -1e9
    return calmar(r["pnl"] / 1000 * 100, max(r["dd"], 0.5), days) * math.sqrt(r["trades"])


def run_combo(symbol, days, lookback, lo, hi, w, as_filter, enabled=True):
    p = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
        "fib_enabled": enabled,
        "fib_swing_lookback": lookback,
        "fib_zone_lo": lo,
        "fib_zone_hi": hi,
        "fib_weight": w,
        "fib_as_filter": as_filter,
    }
    try:
        return backtest_symbol(symbol, days=days, params=p, verbose=False)
    except Exception:
        return None


def walk_forward_5fold(symbol, lookback, lo, hi, w, as_filter):
    folds = []
    for d in [60, 90, 120, 150, 180]:
        r = run_combo(symbol, d, lookback, lo, hi, w, as_filter)
        if r and r.get("trades", 0) > 5:
            folds.append({"days": d, "trades": r["trades"], "pf": r["pf"],
                          "pnl": r["pnl"], "dd": r["dd"]})
    if not folds: return None
    return {"folds": folds,
            "avg_pf": round(sum(f["pf"] for f in folds) / len(folds), 2),
            "avg_pnl": round(sum(f["pnl"] for f in folds) / len(folds), 0),
            "n_positive": sum(1 for f in folds if f["pnl"] > 0),
            "n_folds": len(folds)}


def main():
    if len(sys.argv) < 2: print("Usage: tune_fib_per_symbol.py SYMBOL"); sys.exit(1)
    symbol = sys.argv[1]
    days = 180
    t0 = time.time()
    print(f"\n=== PHASE 6 FIB {symbol} ({days}d) ===")

    baseline = run_combo(symbol, days, 50, 0.5, 0.618, 0, False, enabled=False)
    base_pnl = baseline["pnl"] if baseline else 0
    print(f"  Baseline (no fib): PnL=${base_pnl:.0f} PF={baseline['pf'] if baseline else 0:.2f}")

    results = []
    total = (len(SWING_LOOKBACK) * len(ZONE_LO) * len(ZONE_HI)
             * len(WEIGHT) * len(USE_AS_FILTER))
    i = 0
    for lb in SWING_LOOKBACK:
        for lo in ZONE_LO:
            for hi in ZONE_HI:
                if lo >= hi: continue
                for w in WEIGHT:
                    for f in USE_AS_FILTER:
                        i += 1
                        r = run_combo(symbol, days, lb, lo, hi, w, f)
                        if r and r.get("trades", 0) > 0:
                            results.append({
                                "lookback": lb, "lo": lo, "hi": hi,
                                "weight": w, "as_filter": f,
                                "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
                                "pnl": r["pnl"], "dd": r["dd"],
                                "score": round(score(r, days), 1),
                                "delta": round(r["pnl"] - base_pnl, 1),
                            })
                        if i % 50 == 0:
                            print(f"  {i}/{total} ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        (OUT_DIR / f"{symbol}.json").write_text(json.dumps({"symbol": symbol, "winner": None}))
        return

    results.sort(key=lambda x: -x["score"])
    top3 = results[:3]
    print(f"  TOP 3:")
    for c in top3:
        print(f"    lb={c['lookback']} zone=[{c['lo']},{c['hi']}] w={c['weight']} "
              f"filter={c['as_filter']} PnL=${c['pnl']:.0f} Δ=${c['delta']:+.0f}")

    for c in top3:
        wf = walk_forward_5fold(symbol, c["lookback"], c["lo"], c["hi"],
                                 c["weight"], c["as_filter"])
        c["wf"] = wf

    winner = None
    for c in top3:
        wf = c.get("wf") or {}
        if (wf.get("avg_pf", 0) > 1.3 and
                wf.get("n_positive", 0) >= max(3, int(0.6 * wf.get("n_folds", 5))) and
                c["delta"] > 30):
            winner = c; break

    out = {"symbol": symbol, "days": days, "tested": total,
           "elapsed_s": round(time.time() - t0, 1),
           "baseline_pnl": base_pnl, "top3": top3, "winner": winner}
    if winner:
        wf = winner["wf"]
        print(f"  WINNER: lb={winner['lookback']} zone=[{winner['lo']},{winner['hi']}] "
              f"w={winner['weight']} filter={winner['as_filter']} "
              f"WF PF={wf['avg_pf']} folds {wf['n_positive']}/{wf['n_folds']}+ "
              f"Δ=${winner['delta']:+.0f}")
    else:
        print(f"  NO WINNER passed (Δ>$30 + WF gate)")
    (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
