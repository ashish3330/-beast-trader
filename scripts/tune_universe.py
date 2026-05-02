"""
Parallel per-symbol tuner — Vantage universe.

Reads symbol list from cache/symbol_meta.json, runs grid search per symbol in worker
processes, saves best params to backtest/results/tune_<days>d.json.

Smart-search strategy:
  PASS 1 (coarse): wide grid, find region of viability
  PASS 2 (fine):   refine around PASS 1 winner ±1 tick

Also writes a baseline_<days>d.json snapshot before any tuning so we can compare honestly.
"""
import sys, json, os, time, copy
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import (
    ALL_SYMBOLS, backtest_symbol, SL_OVERRIDE, DEFAULT_PARAMS,
)

RESULTS = ROOT / "backtest" / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

DAYS = int(os.environ.get("TUNE_DAYS", "180"))
WORKERS = int(os.environ.get("TUNE_WORKERS", str(min(8, cpu_count()))))
PASS_NAME = os.environ.get("TUNE_PASS", "pass1")

# Coarse grid (PASS 1): ~96 combos / symbol
COARSE_QUALITY = [
    {"trending": t, "ranging": r, "volatile": t, "low_vol": t}
    for t in (40, 45, 50, 55)
    for r in (45, 50, 55, 60)
]
COARSE_SL = (1.0, 1.5, 2.0, 2.5)
COARSE_RATCHET = ((0.2, 0.5), (0.3, 0.7))


def _slim(r):
    """Drop trade-by-trade detail to keep JSON small."""
    if not r: return r
    return {k: v for k, v in r.items() if k != "details"}


def _bt(symbol, params):
    """One backtest call. If params has sl_atr_mult, override SL for this run only."""
    sl = params.get("sl_atr_mult")
    if sl is None:
        return backtest_symbol(symbol, DAYS, params or None, verbose=False)
    old = SL_OVERRIDE.get(symbol)
    SL_OVERRIDE[symbol] = sl
    try:
        r = backtest_symbol(symbol, DAYS, params, verbose=False)
    finally:
        if old is not None:
            SL_OVERRIDE[symbol] = old
        else:
            SL_OVERRIDE.pop(symbol, None)
    return r


def tune_symbol(symbol):
    """Run coarse grid for one symbol, return {best, baseline, tested}."""
    t0 = time.time()
    # Baseline = current production params (reads SL_OVERRIDE / SIGNAL_QUALITY etc. from live config)
    baseline = _slim(_bt(symbol, {}))

    best = {"pf": 0, "score": -1e9, "params": None, "result": None}
    tested = 0
    for mq in COARSE_QUALITY:
        for sl in COARSE_SL:
            for r1, r2 in COARSE_RATCHET:
                tested += 1
                params = {
                    "min_quality": mq,
                    "sl_atr_mult": sl,
                    "ratchet_1r": r1,
                    "ratchet_2r": r2,
                }
                r = _bt(symbol, params)
                if not r or r.get("trades", 0) < 15:
                    continue
                if r["dd"] > 25:
                    continue
                # Score = PnL × PF / sqrt(DD) — favor PnL but penalize DD
                pf = r["pf"] if r["pf"] < 99 else 5
                score = r["pnl"] * pf / max(r["dd"], 1) ** 0.5
                if score > best["score"]:
                    best = {"pf": pf, "score": score, "params": params, "result": _slim(r)}

    elapsed = time.time() - t0
    return {
        "symbol": symbol,
        "elapsed_s": round(elapsed, 1),
        "tested": tested,
        "baseline": baseline,
        "best": best,
    }


def main():
    symbols = sorted(ALL_SYMBOLS.keys())
    print(f"\nTuning {len(symbols)} symbols × {DAYS}d, {WORKERS} workers")
    print(f"Grid: {len(COARSE_QUALITY)} mq × {len(COARSE_SL)} sl × {len(COARSE_RATCHET)} ratchet "
          f"= {len(COARSE_QUALITY)*len(COARSE_SL)*len(COARSE_RATCHET)} combos/sym\n")

    t0 = time.time()
    results = {}
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(tune_symbol, symbols), 1):
            sym = r["symbol"]
            base = r["baseline"]
            best = r["best"]
            base_pnl = (base or {}).get("pnl", 0)
            base_pf = (base or {}).get("pf", 0)
            if best["result"]:
                br = best["result"]
                bp = best["params"]
                tag = "↑" if br["pnl"] > base_pnl else "·"
                print(f"  [{i:>2}/{len(symbols)}] {sym:14s} {tag} BASE pf={base_pf:5.2f} pnl=${base_pnl:>7.0f}  "
                      f"BEST pf={br['pf']:5.2f} pnl=${br['pnl']:>7.0f} dd={br['dd']:>4.1f}% n={br['trades']:>4}  "
                      f"sl={bp['sl_atr_mult']} mq={bp['min_quality']['trending']}/{bp['min_quality']['ranging']}  "
                      f"({r['elapsed_s']:.0f}s)")
            else:
                print(f"  [{i:>2}/{len(symbols)}] {sym:14s} ✗ NO VIABLE  base pf={base_pf:.2f} pnl=${base_pnl:.0f}  ({r['elapsed_s']:.0f}s)")
            results[sym] = r

    out_path = RESULTS / f"tune_{DAYS}d_{PASS_NAME}.json"
    json.dump({
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "days": DAYS,
        "workers": WORKERS,
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }, open(out_path, "w"), indent=2, default=str)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Wrote {out_path}")


if __name__ == "__main__":
    main()
