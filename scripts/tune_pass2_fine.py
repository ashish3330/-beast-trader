"""
Pass 2 — REFINED per-symbol grid centered on each symbol's pass-1 best.

Reads backtest/results/tune_180d_pass1.json. For each symbol with a viable
pass-1 best, builds a focused grid around that best:

  sl: [base-0.5, base-0.25, base, base+0.25, base+0.5]   clamp 0.5..3.5
  mq_trending: [t-3, t, t+3]                              clamp 35..65
  mq_ranging:  [r-3, r, r+3]                              clamp 35..65
  ratchet_pairs: [(0.2,0.5), (0.3,0.6), (0.3,0.7), (0.4,0.7), (0.4,0.8)]

= 5 × 3 × 3 × 5 = 225 combos / sym.

Score = pnl × pf / sqrt(max(dd,1)).
Acceptance: pass2 score must beat pass1 score by > 3%.  Otherwise pass1
best is carried forward (so apply_tuned_params still gets a viable best).

Output: backtest/results/tune_180d_pass2.json (same shape as pass1).
"""
import sys, json, os, time
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import (
    backtest_symbol, SL_OVERRIDE,
)

RESULTS = ROOT / "backtest" / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

DAYS = int(os.environ.get("TUNE_DAYS", "180"))
WORKERS = int(os.environ.get("TUNE_WORKERS", "6"))
PASS1_PATH = RESULTS / f"tune_{DAYS}d_pass1.json"
OUT_PATH = RESULTS / f"tune_{DAYS}d_pass2.json"

ACCEPT_LIFT_PCT = 3.0  # pass2 must beat pass1 score by >3%

RATCHET_PAIRS = [
    (0.2, 0.5),
    (0.3, 0.6),
    (0.3, 0.7),
    (0.4, 0.7),
    (0.4, 0.8),
]


def _slim(r):
    if not r:
        return r
    return {k: v for k, v in r.items() if k != "details"}


def _bt(symbol, params):
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


def _score(r):
    if not r:
        return -1e9
    pf = r["pf"] if r["pf"] < 99 else 5
    return r["pnl"] * pf / max(r["dd"], 1) ** 0.5


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _build_grid(p1_params):
    base_sl = float(p1_params["sl_atr_mult"])
    sl_list = sorted({round(_clamp(base_sl + d, 0.5, 3.5), 2)
                      for d in (-0.5, -0.25, 0.0, 0.25, 0.5)})

    base_t = int(p1_params["min_quality"]["trending"])
    base_r = int(p1_params["min_quality"]["ranging"])
    t_list = sorted({_clamp(base_t + d, 35, 65) for d in (-3, 0, 3)})
    r_list = sorted({_clamp(base_r + d, 35, 65) for d in (-3, 0, 3)})

    grid = []
    for sl in sl_list:
        for t in t_list:
            for r in r_list:
                for r1, r2 in RATCHET_PAIRS:
                    grid.append({
                        "min_quality": {
                            "trending": int(t),
                            "ranging": int(r),
                            "volatile": int(t),
                            "low_vol": int(t),
                        },
                        "sl_atr_mult": sl,
                        "ratchet_1r": r1,
                        "ratchet_2r": r2,
                    })
    return grid


def tune_symbol(args):
    symbol, p1_entry = args
    t0 = time.time()
    p1_best = (p1_entry or {}).get("best") or {}
    p1_params = p1_best.get("params")
    p1_score = p1_best.get("score", -1e9)
    p1_result = p1_best.get("result")
    baseline = (p1_entry or {}).get("baseline")  # carry pass1's baseline so apply_tuned_params still has it

    if not p1_params:
        return {
            "symbol": symbol,
            "elapsed_s": round(time.time() - t0, 1),
            "tested": 0,
            "baseline": baseline,
            "pass1_best": p1_best,
            "best": {"pf": 0, "score": -1e9, "params": None, "result": None},
            "lift_pct": 0.0,
            "accepted_pass2": False,
            "note": "no pass1 best",
        }

    grid = _build_grid(p1_params)

    best = {"pf": 0, "score": -1e9, "params": None, "result": None}
    tested = 0
    for params in grid:
        tested += 1
        r = _bt(symbol, params)
        if not r or r.get("trades", 0) < 15:
            continue
        if r["dd"] > 25:
            continue
        s = _score(r)
        pf = r["pf"] if r["pf"] < 99 else 5
        if s > best["score"]:
            best = {"pf": pf, "score": s, "params": params, "result": _slim(r)}

    # Acceptance gate: must beat pass1 by >3%
    p2_score = best["score"]
    if p1_score > 0:
        lift_pct = (p2_score - p1_score) / abs(p1_score) * 100.0
    else:
        lift_pct = 100.0 if p2_score > p1_score else 0.0

    accepted = (best["result"] is not None) and (lift_pct > ACCEPT_LIFT_PCT)
    if not accepted:
        # Carry pass1 best forward so downstream apply still works
        final = {
            "pf": p1_best.get("pf", 0),
            "score": p1_score,
            "params": p1_params,
            "result": p1_result,
        }
    else:
        final = best

    return {
        "symbol": symbol,
        "elapsed_s": round(time.time() - t0, 1),
        "tested": tested,
        "baseline": baseline,
        "pass1_best": {"score": p1_score, "params": p1_params, "result": p1_result},
        "pass2_raw_best": best,
        "best": final,
        "lift_pct": round(lift_pct, 2),
        "accepted_pass2": accepted,
    }


def main():
    if not PASS1_PATH.exists():
        print(f"NOT FOUND: {PASS1_PATH}")
        sys.exit(1)
    pass1 = json.load(open(PASS1_PATH))
    p1_results = pass1["results"]
    _env_syms = os.environ.get("TUNE_SYMBOLS", "").strip()
    if _env_syms:
        wanted = {s.strip() for s in _env_syms.split(",") if s.strip()}
        symbols = sorted(s for s in p1_results.keys() if s in wanted)
    else:
        symbols = sorted(p1_results.keys())

    print(f"\nPass 2 fine-tune: {len(symbols)} symbols × {DAYS}d, {WORKERS} workers")
    print(f"Per-symbol grid: 5 sl × 3 mq_t × 3 mq_r × 5 ratchet = up to 225 combos/sym")
    print(f"Acceptance: pass2 score must beat pass1 by > {ACCEPT_LIFT_PCT}%\n")

    args = [(s, p1_results[s]) for s in symbols]

    t0 = time.time()
    results = {}
    accepted_count = 0
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(tune_symbol, args), 1):
            sym = r["symbol"]
            p1 = r["pass1_best"]
            best = r["best"]
            br = best.get("result") or {}
            bp = best.get("params") or {}
            tag = "ACC" if r["accepted_pass2"] else "---"
            if accepted_count is not None and r["accepted_pass2"]:
                accepted_count += 1
            p1_pnl = (p1.get("result") or {}).get("pnl", 0)
            p1_pf = (p1.get("result") or {}).get("pf", 0)
            if br:
                print(f"  [{i:>2}/{len(symbols)}] {sym:14s} {tag} "
                      f"P1 pf={p1_pf:5.2f} pnl=${p1_pnl:>7.0f}  "
                      f"P2 pf={br.get('pf', 0):5.2f} pnl=${br.get('pnl', 0):>7.0f} dd={br.get('dd', 0):>4.1f}% n={br.get('trades', 0):>4}  "
                      f"sl={bp.get('sl_atr_mult')} mq={bp.get('min_quality', {}).get('trending')}/{bp.get('min_quality', {}).get('ranging')} r={bp.get('ratchet_1r')}/{bp.get('ratchet_2r')}  "
                      f"lift={r['lift_pct']:+.1f}%  ({r['elapsed_s']:.0f}s)")
            else:
                print(f"  [{i:>2}/{len(symbols)}] {sym:14s} no viable  ({r['elapsed_s']:.0f}s)")
            results[sym] = r

    # Targeted re-tuning: merge with existing pass2 so untouched symbols survive
    merged = results
    if _env_syms and OUT_PATH.exists():
        try:
            existing = json.load(open(OUT_PATH))
            existing_results = existing.get("results", {})
            existing_results.update(results)
            merged = existing_results
            print(f"\nMerged {len(results)} re-tuned symbols into existing "
                  f"{len(existing_results)} symbols at {OUT_PATH.name}")
        except Exception as e:
            print(f"\nWARN: merge with existing pass2 failed ({e}) — overwriting")
    json.dump({
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "days": DAYS,
        "workers": WORKERS,
        "elapsed_s": round(time.time() - t0, 1),
        "accept_lift_pct": ACCEPT_LIFT_PCT,
        "accepted_count": accepted_count,
        "results": merged,
    }, open(OUT_PATH, "w"), indent=2, default=str)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Wrote {OUT_PATH}")
    print(f"  Accepted pass2 improvements on {accepted_count}/{len(symbols)} symbols")


if __name__ == "__main__":
    main()
