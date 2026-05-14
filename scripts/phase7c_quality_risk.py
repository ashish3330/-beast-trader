#!/usr/bin/env python3 -B
"""
PHASE 7c — refine per-symbol (min_quality × risk_pct) with denser grid + 5-fold WF.

Phase 3a tuned these with a coarse grid. This pass refines with smaller
step sizes around the current per-symbol winner.

Axes:
  min_quality: 45, 50, 55, 60, 65, 70   (6)
  risk_pct:    0.4, 0.6, 0.8, 1.0, 1.5  (5)
  = 30 combos × 17 symbols = 510 backtests + 255 WF

Output: backtest/results/phase7c_quality_risk/<SYMBOL>.json + _summary.json
"""
import json, math, os, sys, time, traceback
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "backtest" / "results" / "phase7c_quality_risk"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_SYMBOLS = [
    "XAUUSD", "XAGUSD", "BTCUSD",
    "DJ30.r", "GER40.r", "HK50.r", "JPN225ft", "SPI200.r",
    "SWI20.r", "US2000.r",
    "COPPER-Cr", "GAS-Cr", "NG-Cr", "UKOUSD",
    "AUDJPY", "EURAUD", "EURUSD",
]

MIN_Q_GRID = [45, 50, 55, 60, 65, 70]
RISK_GRID = [0.4, 0.6, 0.8, 1.0, 1.5]
WF_FOLDS = [60, 90, 120, 150, 180]


def _calmar(pnl_pct, dd_pct, days):
    if dd_pct <= 0 or days <= 0: return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / dd_pct


def _score(r, days):
    if not r or r.get("trades", 0) < 10: return -1e9
    return _calmar(r["pnl"] / 1000 * 100, max(r["dd"], 0.5), days) * math.sqrt(r["trades"])


def _build_params(symbol, mq, risk):
    from backtest.v5_backtest import DEFAULT_PARAMS
    p = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
        "min_quality": mq,
        "risk_pct": risk,
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


def _worker(task):
    sym, mq, risk, days = task
    try:
        from backtest.v5_backtest import backtest_symbol
        r = backtest_symbol(sym, days=days, params=_build_params(sym, mq, risk), verbose=False)
        return sym, mq, risk, days, r
    except Exception:
        return sym, mq, risk, days, {"error": traceback.format_exc()[-200:]}


def main():
    t0 = time.time()
    days = 180
    workers = max(2, min(10, os.cpu_count() or 4))

    print(f"\n{'='*70}\n  PHASE 7c — per-symbol min_quality × risk_pct refine\n"
          f"  Universe: {len(LIVE_SYMBOLS)} | Grid: {len(MIN_Q_GRID)*len(RISK_GRID)} per sym\n"
          f"  Workers: {workers}\n{'='*70}\n", flush=True)

    # Baselines = current auto_tuned per-symbol values (45 MQ + 0.8 risk default if no override)
    print("[1/3] Baselines...", flush=True)
    import auto_tuned as _at
    sq_auto = getattr(_at, "SIGNAL_QUALITY_SYMBOL_AUTO", {})
    rp_auto = getattr(_at, "SYMBOL_RISK_PCT_OVERRIDE_AUTO", {})
    base_tasks = []
    for s in LIVE_SYMBOLS:
        mq = sq_auto.get(s, 45)
        rp = rp_auto.get(s, 0.8)
        base_tasks.append((s, mq, rp, days))
    baselines = {}
    with Pool(workers) as pool:
        for sym, mq, risk, _, r in pool.imap_unordered(_worker, base_tasks):
            if r and not r.get("error"):
                baselines[sym] = {"pnl": r["pnl"], "pf": r["pf"], "trades": r["trades"],
                                   "mq": mq, "risk": risk}
                print(f"  {sym:<12} base PnL=${r['pnl']:+.0f} N={r['trades']} (mq={mq} rp={risk})", flush=True)
            else:
                baselines[sym] = {"pnl": 0, "pf": 0, "trades": 0, "mq": mq, "risk": risk}

    # Grid sweep
    sweep_tasks = []
    for sym in LIVE_SYMBOLS:
        for mq in MIN_Q_GRID:
            for risk in RISK_GRID:
                sweep_tasks.append((sym, mq, risk, days))
    print(f"\n[2/3] Grid sweep: {len(sweep_tasks)} backtests", flush=True)

    results_by_sym = {s: [] for s in LIVE_SYMBOLS}
    t_s = time.time()
    with Pool(workers) as pool:
        for i, (sym, mq, risk, _, r) in enumerate(
            pool.imap_unordered(_worker, sweep_tasks, chunksize=4), 1
        ):
            if r and not r.get("error") and r.get("trades", 0) > 0:
                results_by_sym[sym].append({
                    "mq": mq, "risk": risk, "trades": r["trades"],
                    "pf": r["pf"], "wr": r["wr"], "pnl": r["pnl"], "dd": r["dd"],
                    "score": round(_score(r, days), 1),
                })
            if i % 100 == 0:
                elapsed = time.time() - t_s
                eta = elapsed / i * (len(sweep_tasks) - i)
                print(f"  sweep {i}/{len(sweep_tasks)} elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    # WF on top 3
    print("\n[3/3] 5-fold WF on top 3...", flush=True)
    wf_tasks = []
    top3_by_sym = {}
    for sym, results in results_by_sym.items():
        base_pnl = baselines.get(sym, {}).get("pnl", 0)
        results.sort(key=lambda x: -x["score"])
        positive = [r for r in results if r["pnl"] > base_pnl]
        top3 = positive[:3] if positive else results[:3]
        top3_by_sym[sym] = top3
        for c in top3:
            for d in WF_FOLDS:
                wf_tasks.append((sym, c["mq"], c["risk"], d))

    wf_results = {}
    with Pool(workers) as pool:
        for i, (sym, mq, risk, d, r) in enumerate(
            pool.imap_unordered(_worker, wf_tasks, chunksize=4), 1
        ):
            key = (sym, mq, risk)
            wf_results.setdefault(key, [])
            if r and not r.get("error") and r.get("trades", 0) > 5:
                wf_results[key].append({"days": d, "trades": r["trades"],
                                         "pf": r["pf"], "pnl": r["pnl"], "dd": r["dd"]})

    # Winners
    print(f"\n{'='*70}\n  RESULTS\n{'='*70}\n", flush=True)
    winners = {}
    for sym in LIVE_SYMBOLS:
        top3 = top3_by_sym.get(sym, [])
        base_pnl = baselines.get(sym, {}).get("pnl", 0)
        chosen = None
        for c in top3:
            folds = wf_results.get((sym, c["mq"], c["risk"]), [])
            if not folds: continue
            avg_pf = sum(f["pf"] for f in folds) / len(folds)
            n_pos = sum(1 for f in folds if f["pnl"] > 0)
            delta = c["pnl"] - base_pnl
            c["wf"] = {"avg_pf": round(avg_pf, 2), "n_positive": n_pos,
                       "n_folds": len(folds), "delta": round(delta, 1)}
            if (avg_pf > 1.3 and n_pos >= max(3, int(0.6 * len(folds))) and delta > 50):
                chosen = c
                break

        out = {"symbol": sym, "baseline": baselines.get(sym, {}),
               "top3": top3, "winner": chosen}
        (OUT_DIR / f"{sym.replace('/', '_')}.json").write_text(json.dumps(out, indent=2, default=str))
        if chosen:
            winners[sym] = {"mq": chosen["mq"], "risk": chosen["risk"],
                            "delta": chosen["wf"]["delta"],
                            "wf_pf": chosen["wf"]["avg_pf"],
                            "wf_positive": f"{chosen['wf']['n_positive']}/{chosen['wf']['n_folds']}"}
            print(f"  {sym:<12} WINNER  mq={chosen['mq']} rp={chosen['risk']} "
                  f"Δ=${chosen['wf']['delta']:+.0f} WF PF={chosen['wf']['avg_pf']} "
                  f"{chosen['wf']['n_positive']}/{chosen['wf']['n_folds']}")
        else:
            print(f"  {sym:<12} no winner")

    summary = {
        "phase": "7c", "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "winners": winners, "winner_count": len(winners),
        "total_delta": round(sum(w["delta"] for w in winners.values()), 1),
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  TOTAL WINNERS: {len(winners)}  Δ=${summary['total_delta']:+.0f}  ({summary['elapsed_s']:.0f}s)\n",
          flush=True)


if __name__ == "__main__":
    main()
