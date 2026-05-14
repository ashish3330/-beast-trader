#!/usr/bin/env python3 -B
"""
PHASE 7b coordinator — parallel per-symbol indicator tune across 17-sym live universe.

Forks one worker per (symbol, combo) using multiprocessing.Pool.
Saves per-symbol JSON + writes aggregate winners file.
"""
import json, math, os, sys, time, traceback
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "backtest" / "results" / "phase7b_indicators"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_SYMBOLS = [
    "XAUUSD", "XAGUSD", "BTCUSD",
    "DJ30.r", "GER40.r", "HK50.r", "JPN225ft", "SPI200.r",
    "SWI20.r", "US2000.r",
    "COPPER-Cr", "GAS-Cr", "NG-Cr", "UKOUSD",
    "AUDJPY", "EURAUD", "EURUSD",
]

EMA_S_GRID = [8, 15, 20]
EMA_L_GRID = [30, 40, 50]
MACD_F_GRID = [5, 8, 12]
ST_F_GRID = [2.0, 2.5, 3.0]
ATR_LEN_GRID = [7, 10, 14]

WF_FOLDS = [60, 90, 120, 150, 180]


def _calmar(pnl_pct, dd_pct, days):
    if dd_pct <= 0 or days <= 0: return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / dd_pct


def _score(r, days):
    if not r or r.get("trades", 0) < 10: return -1e9
    return _calmar(r["pnl"] / 1000 * 100, max(r["dd"], 0.5), days) * math.sqrt(r["trades"])


def _build_params(symbol):
    from backtest.v5_backtest import DEFAULT_PARAMS
    p = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
    }
    try:
        import auto_tuned as _at
        rfp = getattr(_at, "RANGE_FILTER_PARAMS_AUTO", {}).get(symbol)
        if rfp:
            p["range_filter_enabled"] = True
            p["range_lookback"] = rfp.get("lookback", 48)
            p["range_buffer_atr"] = rfp.get("buffer_atr", 0.5)
        fp = getattr(_at, "FIB_PARAMS_AUTO", {}).get(symbol)
        if fp:
            p["fib_enabled"] = True
            p["fib_swing_lookback"] = fp.get("lookback", 50)
            p["fib_zone_lo"] = fp.get("zone_lo", 0.5)
            p["fib_zone_hi"] = fp.get("zone_hi", 0.618)
            p["fib_as_filter"] = fp.get("as_filter", True)
    except Exception:
        pass
    return p


def _worker_run(task):
    """Run a single (symbol, combo, days) backtest with overridden indicator params."""
    symbol, combo, days = task
    es, el, mf, stf, atrl = combo
    try:
        from backtest.v5_backtest import backtest_symbol
        from signals import momentum_scorer
        orig = momentum_scorer.IND_OVERRIDES.get(symbol, {}).copy()
        momentum_scorer.IND_OVERRIDES[symbol] = {
            **momentum_scorer.IND_DEFAULTS, **orig,
            "EMA_S": es, "EMA_L": el, "MACD_F": mf, "ST_F": stf, "ATR_LEN": atrl,
        }
        try:
            r = backtest_symbol(symbol, days=days, params=_build_params(symbol), verbose=False)
        finally:
            if orig:
                momentum_scorer.IND_OVERRIDES[symbol] = orig
            elif symbol in momentum_scorer.IND_OVERRIDES:
                del momentum_scorer.IND_OVERRIDES[symbol]
        return (symbol, combo, days, r)
    except Exception:
        return (symbol, combo, days, {"error": traceback.format_exc()[-300:]})


def _baseline_run(task):
    symbol, days = task
    try:
        from backtest.v5_backtest import backtest_symbol
        r = backtest_symbol(symbol, days=days, params=_build_params(symbol), verbose=False)
        return (symbol, days, r)
    except Exception:
        return (symbol, days, {"error": traceback.format_exc()[-200:]})


def main():
    t0 = time.time()
    days = 180
    workers = max(2, min(10, os.cpu_count() or 4))

    print(f"\n{'='*70}")
    print(f"  PHASE 7b — per-symbol indicator param tune")
    print(f"  Universe: {len(LIVE_SYMBOLS)} symbols × 243 grid + 5-fold WF on top 3")
    print(f"  Workers: {workers}  Days: {days}")
    print(f"{'='*70}\n", flush=True)

    # Step 1: baselines
    print("[1/3] Computing baselines (current IND_OVERRIDES per symbol)...", flush=True)
    base_tasks = [(s, days) for s in LIVE_SYMBOLS]
    with Pool(workers) as pool:
        base_results = pool.map(_baseline_run, base_tasks)
    baselines = {}
    for sym, _, r in base_results:
        if r and not r.get("error"):
            baselines[sym] = r
            print(f"  {sym:<12} base PnL=${r.get('pnl', 0):+.0f} PF={r.get('pf', 0):.2f} N={r.get('trades', 0)}", flush=True)
        else:
            print(f"  {sym:<12} ERROR: {r.get('error','')[:80] if r else 'no result'}", flush=True)
            baselines[sym] = {"pnl": 0, "pf": 0, "trades": 0}

    # Step 2: grid sweep
    combos = []
    for es in EMA_S_GRID:
        for el in EMA_L_GRID:
            if el <= es: continue
            for mf in MACD_F_GRID:
                for stf in ST_F_GRID:
                    for atrl in ATR_LEN_GRID:
                        combos.append((es, el, mf, stf, atrl))
    print(f"\n[2/3] Grid sweep: {len(combos)} combos × {len(LIVE_SYMBOLS)} symbols = "
          f"{len(combos)*len(LIVE_SYMBOLS)} backtests", flush=True)

    sweep_tasks = []
    for sym in LIVE_SYMBOLS:
        for combo in combos:
            sweep_tasks.append((sym, combo, days))

    sweep_results_by_sym = {s: [] for s in LIVE_SYMBOLS}
    t_sweep = time.time()
    with Pool(workers) as pool:
        for i, (sym, combo, _, r) in enumerate(
            pool.imap_unordered(_worker_run, sweep_tasks, chunksize=4), 1
        ):
            if r and not r.get("error") and r.get("trades", 0) > 0:
                sweep_results_by_sym[sym].append({
                    "combo": combo, "trades": r["trades"], "pf": r["pf"],
                    "wr": r["wr"], "pnl": r["pnl"], "dd": r["dd"],
                    "score": round(_score(r, days), 1),
                })
            if i % 200 == 0:
                elapsed = time.time() - t_sweep
                eta = elapsed / i * (len(sweep_tasks) - i)
                print(f"  sweep {i}/{len(sweep_tasks)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    # Step 3: 5-fold WF on top 3 per symbol
    print(f"\n[3/3] 5-fold WF on top 3 per symbol...", flush=True)
    wf_tasks = []
    top3_by_sym = {}
    for sym, results in sweep_results_by_sym.items():
        if not results:
            top3_by_sym[sym] = []
            continue
        results.sort(key=lambda x: -x["score"])
        # Filter: keep only combos with Δ>$0 from baseline
        base_pnl = baselines.get(sym, {}).get("pnl", 0)
        positive = [r for r in results if r["pnl"] > base_pnl]
        top3 = positive[:3] if positive else results[:3]
        top3_by_sym[sym] = top3
        for c in top3:
            for d in WF_FOLDS:
                wf_tasks.append((sym, c["combo"], d))

    print(f"  WF tasks: {len(wf_tasks)} ({len(WF_FOLDS)}-fold × top 3 × symbols with top3)", flush=True)
    wf_results = {}
    t_wf = time.time()
    with Pool(workers) as pool:
        for i, (sym, combo, d, r) in enumerate(
            pool.imap_unordered(_worker_run, wf_tasks, chunksize=4), 1
        ):
            key = (sym, combo)
            if key not in wf_results:
                wf_results[key] = []
            if r and not r.get("error") and r.get("trades", 0) > 5:
                wf_results[key].append({"days": d, "trades": r["trades"],
                                          "pf": r["pf"], "pnl": r["pnl"], "dd": r["dd"]})
            if i % 50 == 0:
                print(f"  wf {i}/{len(wf_tasks)}  ({time.time()-t_wf:.0f}s)", flush=True)

    # Aggregate winners
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}\n", flush=True)

    winners = {}
    for sym in LIVE_SYMBOLS:
        top3 = top3_by_sym.get(sym, [])
        base_pnl = baselines.get(sym, {}).get("pnl", 0)
        chosen = None
        for c in top3:
            folds = wf_results.get((sym, c["combo"]), [])
            if not folds: continue
            avg_pf = sum(f["pf"] for f in folds) / len(folds)
            n_pos = sum(1 for f in folds if f["pnl"] > 0)
            delta = c["pnl"] - base_pnl
            c["wf"] = {"folds": folds, "avg_pf": round(avg_pf, 2),
                       "n_positive": n_pos, "n_folds": len(folds),
                       "delta_vs_baseline": round(delta, 1)}
            if (avg_pf > 1.3 and
                    n_pos >= max(3, int(0.6 * len(folds))) and
                    delta > 50):
                chosen = c
                break

        out = {"symbol": sym, "baseline_pnl": base_pnl,
               "n_tested": len(sweep_results_by_sym.get(sym, [])),
               "top3": top3, "winner": chosen}
        (OUT_DIR / f"{sym.replace('/', '_')}.json").write_text(
            json.dumps(out, indent=2, default=str))

        if chosen:
            es, el, mf, stf, atrl = chosen["combo"]
            winners[sym] = {
                "EMA_S": es, "EMA_L": el, "MACD_F": mf,
                "ST_F": stf, "ATR_LEN": atrl,
                "delta": chosen["wf"]["delta_vs_baseline"],
                "wf_pf": chosen["wf"]["avg_pf"],
                "wf_positive": f"{chosen['wf']['n_positive']}/{chosen['wf']['n_folds']}",
            }
            print(f"  {sym:<12} WINNER  ES={es} EL={el} MF={mf} STF={stf} ATR={atrl} "
                  f"Δ=${chosen['wf']['delta_vs_baseline']:+.0f} WF PF={chosen['wf']['avg_pf']} "
                  f"folds {chosen['wf']['n_positive']}/{chosen['wf']['n_folds']}")
        else:
            print(f"  {sym:<12} no winner (top3 failed Δ>$50 + WF gate)")

    summary = {
        "phase": "7b",
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "universe": LIVE_SYMBOLS,
        "baselines": {s: baselines[s].get("pnl", 0) for s in LIVE_SYMBOLS},
        "winners": winners,
        "winner_count": len(winners),
        "total_delta": round(sum(w["delta"] for w in winners.values()), 1),
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n{'='*70}")
    print(f"  TOTAL WINNERS: {len(winners)}/{len(LIVE_SYMBOLS)}")
    print(f"  TOTAL Δ: ${summary['total_delta']:+.0f}")
    print(f"  ELAPSED: {summary['elapsed_s']:.0f}s")
    print(f"  Saved to: {OUT_DIR}")
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
