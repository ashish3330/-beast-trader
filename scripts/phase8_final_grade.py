#!/usr/bin/env python3 -B
"""
PHASE 8 — Final pre-real-money grade + SL refinement.

(A) Full backtest report per symbol at 180d (recent) + 360d (extended).
    Mirrors live: audit_fix_gates ON, slippage/commission/swap ON, reads
    auto_tuned RANGE/FIB overrides.

(B) Per-symbol global SL_MULT refinement. 8-value grid × 17 symbols × 5-fold
    walk-forward. Decision gate: Δ>$30 vs current AND avg PF>1.3 AND
    ≥3/5 folds positive.

Output:
  backtest/results/phase8_final_grade/_baseline_grade.json (per-symbol 180/360)
  backtest/results/phase8_final_grade/_sl_winners.json
  backtest/results/phase8_final_grade/<SYMBOL>.json
"""
import json, math, os, sys, time, traceback
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "backtest" / "results" / "phase8_final_grade"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_SYMBOLS = [
    "XAUUSD", "XAGUSD", "BTCUSD",
    "DJ30.r", "GER40.r", "HK50.r", "JPN225ft", "SPI200.r",
    "SWI20.r", "US2000.r",
    "COPPER-Cr", "GAS-Cr", "NG-Cr", "UKOUSD",
    "AUDJPY", "EURAUD", "EURUSD",
]

SL_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
WF_FOLDS = [60, 90, 120, 150, 180]


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


def _calmar(pnl_pct, dd_pct, days):
    if dd_pct <= 0 or days <= 0: return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / dd_pct


def _score(r, days):
    if not r or r.get("trades", 0) < 10: return -1e9
    return _calmar(r["pnl"] / 1000 * 100, max(r["dd"], 0.5), days) * math.sqrt(r["trades"])


def _baseline_worker(task):
    sym, days = task
    try:
        from backtest.v5_backtest import backtest_symbol
        r = backtest_symbol(sym, days=days, params=_build_params(sym), verbose=False)
        return sym, days, r
    except Exception:
        return sym, days, {"error": traceback.format_exc()[-200:]}


def _sl_worker(task):
    sym, sl, days = task
    try:
        from backtest import v5_backtest
        orig_sl = v5_backtest.SL_OVERRIDE.get(sym)
        v5_backtest.SL_OVERRIDE[sym] = sl
        try:
            r = v5_backtest.backtest_symbol(sym, days=days, params=_build_params(sym), verbose=False)
        finally:
            if orig_sl is not None:
                v5_backtest.SL_OVERRIDE[sym] = orig_sl
            elif sym in v5_backtest.SL_OVERRIDE:
                del v5_backtest.SL_OVERRIDE[sym]
        return sym, sl, days, r
    except Exception:
        return sym, sl, days, {"error": traceback.format_exc()[-200:]}


def main():
    t0 = time.time()
    workers = max(2, min(10, os.cpu_count() or 4))

    print(f"\n{'='*70}\n  PHASE 8 — Final pre-real-money grade\n  Workers: {workers}\n{'='*70}\n", flush=True)

    # ─── (A) 180d + 360d baseline grade ───
    print(f"[A] Baseline grade ({len(LIVE_SYMBOLS)} symbols × 2 horizons)...", flush=True)
    base_tasks = []
    for sym in LIVE_SYMBOLS:
        base_tasks.append((sym, 180))
        base_tasks.append((sym, 360))
    base_results = {}
    with Pool(workers) as pool:
        for sym, days, r in pool.imap_unordered(_baseline_worker, base_tasks):
            base_results.setdefault(sym, {})[days] = r

    print(f"\n  {'SYM':<12} {'180d_N':>6} {'180d_PnL':>10} {'180d_PF':>6} {'180d_WR':>6} {'180d_DD':>7} "
          f" {'360d_N':>6} {'360d_PnL':>10} {'360d_PF':>6} {'360d_WR':>6} {'360d_DD':>7}", flush=True)
    print("  " + "-" * 116, flush=True)
    grade = []
    for sym in LIVE_SYMBOLS:
        r180 = base_results.get(sym, {}).get(180) or {}
        r360 = base_results.get(sym, {}).get(360) or {}
        line = (f"  {sym:<12} "
                f"{r180.get('trades', 0):>6} ${r180.get('pnl', 0):>+8.0f} {r180.get('pf', 0):>6.2f} "
                f"{r180.get('wr', 0):>5.0f}% {r180.get('dd', 0):>6.1f}% "
                f" {r360.get('trades', 0):>6} ${r360.get('pnl', 0):>+8.0f} {r360.get('pf', 0):>6.2f} "
                f"{r360.get('wr', 0):>5.0f}% {r360.get('dd', 0):>6.1f}%")
        print(line, flush=True)
        grade.append({
            "symbol": sym,
            "180d": {k: r180.get(k) for k in ("trades", "pnl", "pf", "wr", "dd", "avg_r")},
            "360d": {k: r360.get(k) for k in ("trades", "pnl", "pf", "wr", "dd", "avg_r")},
        })

    tot_180 = sum(g["180d"].get("pnl") or 0 for g in grade)
    tot_360 = sum(g["360d"].get("pnl") or 0 for g in grade)
    n_180 = sum(g["180d"].get("trades") or 0 for g in grade)
    n_360 = sum(g["360d"].get("trades") or 0 for g in grade)
    n_profitable_180 = sum(1 for g in grade if (g["180d"].get("pnl") or 0) > 0)
    n_profitable_360 = sum(1 for g in grade if (g["360d"].get("pnl") or 0) > 0)
    print(f"\n  TOTAL 180d: N={n_180}  PnL=${tot_180:+.0f}  "
          f"{n_profitable_180}/{len(LIVE_SYMBOLS)} symbols profitable", flush=True)
    print(f"  TOTAL 360d: N={n_360}  PnL=${tot_360:+.0f}  "
          f"{n_profitable_360}/{len(LIVE_SYMBOLS)} symbols profitable", flush=True)

    (OUT_DIR / "_baseline_grade.json").write_text(json.dumps({
        "phase": "8A_grade", "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totals": {"180d_pnl": tot_180, "180d_trades": n_180, "180d_profitable": n_profitable_180,
                   "360d_pnl": tot_360, "360d_trades": n_360, "360d_profitable": n_profitable_360},
        "per_symbol": grade,
    }, indent=2, default=str))

    # ─── (B) Per-symbol SL refinement with WF gate ───
    print(f"\n[B] Per-symbol global SL_MULT refinement", flush=True)
    print(f"    Grid: {SL_GRID}  ({len(SL_GRID)} values × {len(LIVE_SYMBOLS)} sym = "
          f"{len(SL_GRID)*len(LIVE_SYMBOLS)} sweep + {3*len(LIVE_SYMBOLS)*len(WF_FOLDS)} WF)", flush=True)

    sl_tasks = []
    for sym in LIVE_SYMBOLS:
        for sl in SL_GRID:
            sl_tasks.append((sym, sl, 180))

    sweep_by_sym = {s: [] for s in LIVE_SYMBOLS}
    t_s = time.time()
    with Pool(workers) as pool:
        for i, (sym, sl, _, r) in enumerate(
            pool.imap_unordered(_sl_worker, sl_tasks, chunksize=2), 1
        ):
            if r and not r.get("error") and r.get("trades", 0) > 0:
                sweep_by_sym[sym].append({
                    "sl": sl, "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
                    "pnl": r["pnl"], "dd": r["dd"], "score": round(_score(r, 180), 1),
                })
            if i % 20 == 0:
                print(f"    sweep {i}/{len(sl_tasks)} ({time.time()-t_s:.0f}s)", flush=True)

    # Pick top 3 per symbol, WF on each
    wf_tasks = []
    top3_by_sym = {}
    for sym, results in sweep_by_sym.items():
        if not results:
            top3_by_sym[sym] = []
            continue
        # Get current SL (from auto_tuned + config)
        import auto_tuned as _at
        from backtest import v5_backtest as _bt
        current_sl = _at.SL_OVERRIDE_AUTO.get(sym, _bt.SL_OVERRIDE.get(sym, 1.5))
        results.sort(key=lambda x: -x["score"])
        # Filter: keep only combos with Δ vs current
        base_pnl = base_results.get(sym, {}).get(180, {}).get("pnl", 0) or 0
        top3 = [r for r in results if r["pnl"] > base_pnl + 30][:3]
        top3_by_sym[sym] = top3
        for c in top3:
            for d in WF_FOLDS:
                wf_tasks.append((sym, c["sl"], d))

    print(f"    WF tasks: {len(wf_tasks)} ({3} top × {len(WF_FOLDS)} folds × {sum(1 for v in top3_by_sym.values() if v)} symbols)", flush=True)
    wf_results = {}
    t_wf = time.time()
    with Pool(workers) as pool:
        for i, (sym, sl, days, r) in enumerate(
            pool.imap_unordered(_sl_worker, wf_tasks, chunksize=2), 1
        ):
            key = (sym, sl)
            wf_results.setdefault(key, [])
            if r and not r.get("error") and r.get("trades", 0) > 5:
                wf_results[key].append({"days": days, "trades": r["trades"],
                                         "pf": r["pf"], "pnl": r["pnl"], "dd": r["dd"]})

    # Winners
    print(f"\n[B] WINNERS (passed WF gate):", flush=True)
    winners = {}
    for sym in LIVE_SYMBOLS:
        top3 = top3_by_sym.get(sym, [])
        base_pnl = base_results.get(sym, {}).get(180, {}).get("pnl", 0) or 0
        chosen = None
        for c in top3:
            folds = wf_results.get((sym, c["sl"]), [])
            if not folds: continue
            avg_pf = sum(f["pf"] for f in folds) / len(folds)
            n_pos = sum(1 for f in folds if f["pnl"] > 0)
            delta = c["pnl"] - base_pnl
            c["wf"] = {"avg_pf": round(avg_pf, 2), "n_positive": n_pos,
                       "n_folds": len(folds), "delta": round(delta, 1)}
            if (avg_pf > 1.3 and n_pos >= max(3, int(0.6 * len(folds))) and delta > 30):
                chosen = c
                break

        if chosen:
            import auto_tuned as _at
            from backtest import v5_backtest as _bt
            current_sl = _at.SL_OVERRIDE_AUTO.get(sym, _bt.SL_OVERRIDE.get(sym, 1.5))
            winners[sym] = {
                "old_sl": current_sl,
                "new_sl": chosen["sl"],
                "delta": chosen["wf"]["delta"],
                "wf_pf": chosen["wf"]["avg_pf"],
                "wf_positive": f"{chosen['wf']['n_positive']}/{chosen['wf']['n_folds']}",
            }
            print(f"  {sym:<12} SL {current_sl} → {chosen['sl']}  Δ=${chosen['wf']['delta']:+.0f}  "
                  f"WF PF={chosen['wf']['avg_pf']}  {chosen['wf']['n_positive']}/{chosen['wf']['n_folds']}", flush=True)

    if not winners:
        print(f"  No winners — current SL tuning is already optimal.", flush=True)

    (OUT_DIR / "_sl_winners.json").write_text(json.dumps(winners, indent=2, default=str))

    # Per-symbol full file
    for sym in LIVE_SYMBOLS:
        out = {
            "symbol": sym,
            "180d_baseline": base_results.get(sym, {}).get(180),
            "360d_baseline": base_results.get(sym, {}).get(360),
            "sl_top3": top3_by_sym.get(sym, []),
            "sl_winner": winners.get(sym),
        }
        (OUT_DIR / f"{sym.replace('/', '_')}.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"\n{'='*70}", flush=True)
    print(f"  Phase 8 complete. elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"  Saved: {OUT_DIR}", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
