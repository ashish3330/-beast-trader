#!/usr/bin/env python3 -B
"""
PHASE 9 — Hard tune of 9 A-grade symbols for $1.2K real-money setup.

Universe (all positive on both 180d AND 360d):
  US2000.r, SWI20.r, JPN225ft, DJ30.r, SPI200.r,
  XAUUSD, EURUSD, AUDJPY, BTCUSD

Multi-axis grid per symbol:
  SL_MULT:     0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0   (8)
  EMA_S:       8, 12, 15, 20                              (4)
  EMA_L:       30, 40, 50                                 (3, filter el>es)
  MACD_F:      5, 8, 12                                   (3)
  ATR_LEN:     7, 10, 14                                  (3)
  ST_F:        2.0, 2.5, 3.0                              (3)
  min_quality: 40, 55, 70                                 (3)
= ~5,184 combos per symbol × 9 = ~47K backtests
+ 5-fold WF on top 3 per symbol = ~135

WF gate (real-money grade):
  - avg PF > 1.5
  - ≥3/5 folds positive
  - Δ > $30 vs current baseline
"""
import json, math, os, sys, time, traceback
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "backtest" / "results" / "phase9_hard_tune"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "US2000.r", "SWI20.r", "JPN225ft", "DJ30.r", "SPI200.r",
    "XAUUSD", "EURUSD", "AUDJPY", "BTCUSD",
]

SL_GRID = [0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
EMA_S_GRID = [8, 12, 15, 20]
EMA_L_GRID = [30, 40, 50]
MACD_F_GRID = [5, 8, 12]
ATR_LEN_GRID = [7, 10, 14]
ST_F_GRID = [2.0, 2.5, 3.0]
MIN_Q_GRID = [40, 55, 70]
WF_FOLDS = [60, 90, 120, 150, 180]


def _build_params(min_q):
    from backtest.v5_backtest import DEFAULT_PARAMS
    return {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
        "min_quality": min_q,
    }


def _calmar(pnl_pct, dd_pct, days):
    if dd_pct <= 0 or days <= 0: return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / dd_pct


def _score(r, days):
    if not r or r.get("trades", 0) < 10: return -1e9
    return _calmar(r["pnl"] / 1000 * 100, max(r["dd"], 0.5), days) * math.sqrt(r["trades"])


def _worker(task):
    sym, sl, es, el, mf, atrl, stf, mq, days = task
    try:
        from backtest import v5_backtest
        from signals import momentum_scorer
        orig_sl = v5_backtest.SL_OVERRIDE.get(sym)
        orig_ind = momentum_scorer.IND_OVERRIDES.get(sym, {}).copy()
        v5_backtest.SL_OVERRIDE[sym] = sl
        momentum_scorer.IND_OVERRIDES[sym] = {
            **momentum_scorer.IND_DEFAULTS, **orig_ind,
            "EMA_S": es, "EMA_L": el, "MACD_F": mf,
            "ATR_LEN": atrl, "ST_F": stf,
        }
        try:
            r = v5_backtest.backtest_symbol(sym, days=days,
                                             params=_build_params(mq), verbose=False)
        finally:
            if orig_sl is not None:
                v5_backtest.SL_OVERRIDE[sym] = orig_sl
            elif sym in v5_backtest.SL_OVERRIDE:
                del v5_backtest.SL_OVERRIDE[sym]
            if orig_ind:
                momentum_scorer.IND_OVERRIDES[sym] = orig_ind
            elif sym in momentum_scorer.IND_OVERRIDES:
                del momentum_scorer.IND_OVERRIDES[sym]
        return sym, (sl, es, el, mf, atrl, stf, mq), days, r
    except Exception:
        return sym, (sl, es, el, mf, atrl, stf, mq), days, {"error": traceback.format_exc()[-200:]}


def main():
    t0 = time.time()
    workers = max(2, min(10, os.cpu_count() or 4))

    print(f"\n{'='*70}\n  PHASE 9 — Hard tune of {len(SYMBOLS)} A-grade symbols\n{'='*70}", flush=True)
    print(f"  Workers: {workers}", flush=True)

    # Baselines
    print(f"\n[1] Baselines (current config)...", flush=True)
    base_results = {}
    from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS
    for sym in SYMBOLS:
        # Use current auto_tuned values implicitly
        p = {**DEFAULT_PARAMS, "audit_fix_gates": True,
             "with_slippage": True, "with_commission": True, "with_swap": True}
        r = backtest_symbol(sym, days=180, params=p, verbose=False)
        base_results[sym] = r
        print(f"  {sym:<12} N={r.get('trades',0):>4}  PnL=${r.get('pnl',0):>+7.0f}  "
              f"PF={r.get('pf',0):.2f}  WR={r.get('wr',0):.0f}%  DD={r.get('dd',0):.1f}%", flush=True)
    base_total = sum(r.get('pnl', 0) for r in base_results.values())
    print(f"  TOTAL baseline 180d: ${base_total:+.0f}", flush=True)

    # Build combos
    combos = []
    for sl in SL_GRID:
        for es in EMA_S_GRID:
            for el in EMA_L_GRID:
                if el <= es: continue
                for mf in MACD_F_GRID:
                    for atrl in ATR_LEN_GRID:
                        for stf in ST_F_GRID:
                            for mq in MIN_Q_GRID:
                                combos.append((sl, es, el, mf, atrl, stf, mq))
    print(f"\n[2] Grid: {len(combos)} combos/sym × {len(SYMBOLS)} syms = "
          f"{len(combos)*len(SYMBOLS)} backtests", flush=True)

    sweep_tasks = []
    for sym in SYMBOLS:
        for c in combos:
            sweep_tasks.append((sym, *c, 180))

    sweep_by_sym = {s: [] for s in SYMBOLS}
    t_s = time.time()
    with Pool(workers) as pool:
        for i, (sym, combo, _, r) in enumerate(
            pool.imap_unordered(_worker, sweep_tasks, chunksize=16), 1
        ):
            if r and not r.get("error") and r.get("trades", 0) > 0:
                sweep_by_sym[sym].append({
                    "combo": combo, "trades": r["trades"], "pf": r["pf"],
                    "wr": r["wr"], "pnl": r["pnl"], "dd": r["dd"],
                    "score": round(_score(r, 180), 1),
                })
            if i % 1000 == 0:
                elapsed = time.time() - t_s
                eta = elapsed / i * (len(sweep_tasks) - i)
                print(f"  sweep {i}/{len(sweep_tasks)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    # WF on top 3
    print(f"\n[3] 5-fold walk-forward on top 3 per symbol...", flush=True)
    wf_tasks = []
    top3_by_sym = {}
    for sym, results in sweep_by_sym.items():
        if not results:
            top3_by_sym[sym] = []
            continue
        base_pnl = base_results.get(sym, {}).get("pnl", 0) or 0
        results.sort(key=lambda x: -x["score"])
        positive = [r for r in results if r["pnl"] > base_pnl + 30]
        top3 = positive[:3] if positive else results[:3]
        top3_by_sym[sym] = top3
        for c in top3:
            for d in WF_FOLDS:
                wf_tasks.append((sym, *c["combo"], d))

    print(f"  WF tasks: {len(wf_tasks)}", flush=True)
    wf_results = {}
    t_wf = time.time()
    with Pool(workers) as pool:
        for i, (sym, combo, d, r) in enumerate(
            pool.imap_unordered(_worker, wf_tasks, chunksize=4), 1
        ):
            key = (sym, combo)
            wf_results.setdefault(key, [])
            if r and not r.get("error") and r.get("trades", 0) > 5:
                wf_results[key].append({"days": d, "trades": r["trades"],
                                         "pf": r["pf"], "pnl": r["pnl"], "dd": r["dd"]})
            if i % 50 == 0:
                print(f"  wf {i}/{len(wf_tasks)} ({time.time()-t_wf:.0f}s)", flush=True)

    # Winners
    print(f"\n{'='*70}\n  WINNERS\n{'='*70}\n", flush=True)
    winners = {}
    new_total = 0
    for sym in SYMBOLS:
        top3 = top3_by_sym.get(sym, [])
        base_pnl = base_results.get(sym, {}).get("pnl", 0) or 0
        chosen = None
        for c in top3:
            folds = wf_results.get((sym, c["combo"]), [])
            if not folds: continue
            avg_pf = sum(f["pf"] for f in folds) / len(folds)
            n_pos = sum(1 for f in folds if f["pnl"] > 0)
            delta = c["pnl"] - base_pnl
            c["wf"] = {"avg_pf": round(avg_pf, 2), "n_positive": n_pos,
                       "n_folds": len(folds), "delta": round(delta, 1)}
            if (avg_pf > 1.5 and
                    n_pos >= max(3, int(0.6 * len(folds))) and
                    delta > 30):
                chosen = c
                break

        if chosen:
            sl, es, el, mf, atrl, stf, mq = chosen["combo"]
            winners[sym] = {
                "sl_mult": sl, "EMA_S": es, "EMA_L": el,
                "MACD_F": mf, "ATR_LEN": atrl, "ST_F": stf, "min_quality": mq,
                "old_pnl": round(base_pnl, 0), "new_pnl": round(chosen["pnl"], 0),
                "delta": chosen["wf"]["delta"],
                "wf_pf": chosen["wf"]["avg_pf"],
                "wf_positive": f"{chosen['wf']['n_positive']}/{chosen['wf']['n_folds']}",
                "trades": chosen["trades"], "dd": chosen["dd"],
            }
            new_total += chosen["pnl"]
            print(f"  {sym:<12} SL={sl} ES={es} EL={el} MF={mf} ATR={atrl} STF={stf} MQ={mq}", flush=True)
            print(f"             ${base_pnl:+.0f} → ${chosen['pnl']:+.0f}  Δ=${chosen['wf']['delta']:+.0f}  "
                  f"WF PF={chosen['wf']['avg_pf']}  {chosen['wf']['n_positive']}/{chosen['wf']['n_folds']}", flush=True)
        else:
            new_total += base_pnl
            print(f"  {sym:<12} NO WINNER — keeping current config (PnL ${base_pnl:+.0f})", flush=True)

    print(f"\n  TOTAL BASELINE: ${base_total:+.0f}", flush=True)
    print(f"  TOTAL POST-TUNE: ${new_total:+.0f}", flush=True)
    print(f"  TOTAL LIFT: ${new_total-base_total:+.0f}", flush=True)

    out = {
        "phase": "9_hard_tune",
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "baselines": {s: {"pnl": base_results.get(s, {}).get("pnl"),
                          "pf": base_results.get(s, {}).get("pf"),
                          "trades": base_results.get(s, {}).get("trades")} for s in SYMBOLS},
        "winners": winners,
        "total_baseline": base_total, "total_new": new_total,
        "total_lift": new_total - base_total,
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Saved: {OUT_DIR}/_summary.json  elapsed={time.time()-t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    main()
