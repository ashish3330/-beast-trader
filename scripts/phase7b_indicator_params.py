#!/usr/bin/env python3 -B
"""
PHASE 7b — per-symbol indicator param tune.

Tunes base indicator params (EMAs, MACD, ATR, SuperTrend) per symbol.
These directly affect scoring — different symbols may need different
EMA periods, MACD speeds, ATR lookbacks.

Per-symbol axes (focused grid to avoid combinatorial explosion):
  EMA_S:    8, 15, 20         (3)
  EMA_L:    30, 40, 50        (3)
  MACD_F:   5, 8, 12          (3)
  ST_F:     2.0, 2.5, 3.0     (3)
  ATR_LEN:  7, 10, 14         (3)
  = 243 combos × 17 symbols = 4131 backtests

5-fold WF on top 3 each.

Output: backtest/results/phase7b_indicators/<SYMBOL>.json
"""
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402
from signals.momentum_scorer import IND_DEFAULTS, IND_OVERRIDES  # noqa: E402

OUT_DIR = ROOT / "backtest" / "results" / "phase7b_indicators"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMA_S = [8, 15, 20]
EMA_L = [30, 40, 50]
MACD_F = [5, 8, 12]
ST_F = [2.0, 2.5, 3.0]
ATR_LEN = [7, 10, 14]


def calmar(pnl_pct, dd_pct, days):
    if dd_pct <= 0 or days <= 0: return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / dd_pct


def score(r, days):
    if not r or r.get("trades", 0) < 10: return -1e9
    return calmar(r["pnl"] / 1000 * 100, max(r["dd"], 0.5), days) * math.sqrt(r["trades"])


def run_combo(symbol, days, ema_s, ema_l, macd_f, st_f, atr_len):
    # Override IND_OVERRIDES for this symbol temporarily by setting custom icfg
    orig_overrides = IND_OVERRIDES.get(symbol, {}).copy()
    IND_OVERRIDES[symbol] = {
        **IND_DEFAULTS, **orig_overrides,
        "EMA_S": ema_s, "EMA_L": ema_l,
        "MACD_F": macd_f, "ST_F": st_f, "ATR_LEN": atr_len,
    }
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
    try:
        result = backtest_symbol(symbol, days=days, params=p, verbose=False)
    except Exception:
        result = None
    # Restore
    if orig_overrides:
        IND_OVERRIDES[symbol] = orig_overrides
    elif symbol in IND_OVERRIDES:
        del IND_OVERRIDES[symbol]
    return result


def walk_forward_5fold(symbol, ema_s, ema_l, macd_f, st_f, atr_len):
    folds = []
    for d in [60, 90, 120, 150, 180]:
        r = run_combo(symbol, d, ema_s, ema_l, macd_f, st_f, atr_len)
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
    if len(sys.argv) < 2: print("Usage: phase7b_indicator_params.py SYMBOL"); sys.exit(1)
    symbol = sys.argv[1]
    days = 180
    t0 = time.time()
    print(f"\n=== PHASE 7b {symbol} (indicator tune, {days}d) ===")

    baseline = backtest_symbol(symbol, days=days, params={
        **DEFAULT_PARAMS, "audit_fix_gates": True,
        "with_slippage": True, "with_commission": True, "with_swap": True,
    }, verbose=False)
    base_pnl = baseline["pnl"] if baseline else 0
    print(f"  Baseline (current IND_OVERRIDES): PnL=${base_pnl:.0f}")

    results = []
    total = len(EMA_S) * len(EMA_L) * len(MACD_F) * len(ST_F) * len(ATR_LEN)
    i = 0
    for es in EMA_S:
        for el in EMA_L:
            if el <= es: continue
            for mf in MACD_F:
                for stf in ST_F:
                    for atrl in ATR_LEN:
                        i += 1
                        r = run_combo(symbol, days, es, el, mf, stf, atrl)
                        if r and r.get("trades", 0) > 0:
                            results.append({
                                "ema_s": es, "ema_l": el, "macd_f": mf,
                                "st_f": stf, "atr_len": atrl,
                                "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
                                "pnl": r["pnl"], "dd": r["dd"],
                                "score": round(score(r, days), 1),
                                "delta": round(r["pnl"] - base_pnl, 1),
                            })
                        if i % 30 == 0:
                            print(f"  {i}/{total} ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        (OUT_DIR / f"{symbol}.json").write_text(json.dumps({"symbol": symbol, "winner": None}))
        return

    results.sort(key=lambda x: -x["score"])
    top3 = results[:3]
    print(f"  TOP 3:")
    for c in top3:
        print(f"    ES={c['ema_s']} EL={c['ema_l']} MF={c['macd_f']} STF={c['st_f']} ATR={c['atr_len']} "
              f"PnL=${c['pnl']:.0f} Δ=${c['delta']:+.0f}")

    for c in top3:
        c["wf"] = walk_forward_5fold(symbol, c["ema_s"], c["ema_l"],
                                      c["macd_f"], c["st_f"], c["atr_len"])

    winner = None
    for c in top3:
        wf = c.get("wf") or {}
        if (wf.get("avg_pf", 0) > 1.3 and
                wf.get("n_positive", 0) >= max(3, int(0.6 * wf.get("n_folds", 5))) and
                c["delta"] > 50):
            winner = c; break

    out = {"symbol": symbol, "days": days, "tested": total,
           "elapsed_s": round(time.time() - t0, 1),
           "baseline_pnl": base_pnl, "top3": top3, "winner": winner}
    if winner:
        wf = winner["wf"]
        print(f"  WINNER: ES={winner['ema_s']} EL={winner['ema_l']} MF={winner['macd_f']} "
              f"STF={winner['st_f']} ATR={winner['atr_len']} Δ=${winner['delta']:+.0f} "
              f"WF PF={wf['avg_pf']} folds {wf['n_positive']}/{wf['n_folds']}")
    else:
        print(f"  no winner (Δ>$50 + WF gate)")
    (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
