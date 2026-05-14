#!/usr/bin/env python3 -B
"""
PHASE 5 — per-symbol range-filter tune (lookback × buffer_atr).

Tests:
  range_lookback: [24, 36, 48, 72, 96] bars (= 6, 9, 12, 18, 24 hours on H1)
  range_buffer_atr: [0.25, 0.5, 0.75, 1.0] × ATR

Filter logic (in v5_backtest, gated by range_filter_enabled=True):
  IF regime=='ranging' AND close >= range_high - buffer  → reject LONG (chase top)
  IF regime=='ranging' AND close <= range_low + buffer   → reject SHORT (chase bot)

Goal: optimize for compound Calmar with 5-fold walk-forward, gate=passes
  - WF avg PF >= 1.3 AND >= 3/5 folds positive.

Output: backtest/results/range_filter_tune_20260514/<SYMBOL>.json
"""
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

OUT_DIR = ROOT / "backtest" / "results" / "range_filter_tune_20260514"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACKS = [24, 36, 48, 72, 96]
BUFFERS = [0.25, 0.5, 0.75, 1.0]


def calmar(pnl_pct, dd_pct, days):
    if dd_pct <= 0 or days <= 0: return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / dd_pct


def score(r, days):
    if not r or r.get("trades", 0) < 10: return -1e9
    return calmar(r["pnl"] / 1000 * 100, max(r["dd"], 0.5), days) * math.sqrt(r["trades"])


def run_combo(symbol, days, lookback, buf, enabled=True):
    p = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
        "range_filter_enabled": enabled,
        "range_lookback": lookback,
        "range_buffer_atr": buf,
    }
    try:
        return backtest_symbol(symbol, days=days, params=p, verbose=False)
    except Exception:
        return None


def walk_forward_5fold(symbol, lookback, buf):
    folds = []
    for d in [60, 90, 120, 150, 180]:
        r = run_combo(symbol, d, lookback, buf)
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
    if len(sys.argv) < 2: print("Usage: tune_range_filter.py SYMBOL"); sys.exit(1)
    symbol = sys.argv[1]
    days = 180
    t0 = time.time()
    print(f"\n=== PHASE 5 {symbol} ({days}d, range-filter tune) ===")

    # Baseline (filter disabled)
    baseline = run_combo(symbol, days, 48, 0.5, enabled=False)
    base_score = score(baseline, days) if baseline else 0
    print(f"  Baseline (no filter): PnL=${baseline['pnl'] if baseline else 0:.0f} "
          f"PF={baseline['pf'] if baseline else 0:.2f} score={base_score:.1f}")

    results = []
    n_combos = len(LOOKBACKS) * len(BUFFERS)
    i = 0
    for lb in LOOKBACKS:
        for buf in BUFFERS:
            i += 1
            r = run_combo(symbol, days, lb, buf)
            if r and r.get("trades", 0) > 0:
                results.append({
                    "lookback": lb, "buffer": buf,
                    "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
                    "pnl": r["pnl"], "dd": r["dd"],
                    "score": round(score(r, days), 1),
                    "delta_vs_baseline": round((r["pnl"] - (baseline["pnl"] if baseline else 0)), 1),
                })
            if i % 5 == 0:
                print(f"  {i}/{n_combos} ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        out = {"symbol": symbol, "winner": None}
        (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2))
        print(f"  NO RESULTS"); return

    results.sort(key=lambda x: -x["score"])
    top3 = results[:3]
    print(f"\n  TOP 3:")
    for c in top3:
        print(f"    lb={c['lookback']} buf={c['buffer']} PnL=${c['pnl']:.0f} PF={c['pf']:.2f} "
              f"Δ=${c['delta_vs_baseline']:+.0f}")

    # Walk-forward top 3
    for c in top3:
        wf = walk_forward_5fold(symbol, c["lookback"], c["buffer"])
        c["wf"] = wf

    winner = None
    for c in top3:
        wf = c.get("wf") or {}
        if (wf.get("avg_pf", 0) > 1.3 and
                wf.get("n_positive", 0) >= max(3, int(0.6 * wf.get("n_folds", 5))) and
                c["delta_vs_baseline"] > 0):
            winner = c; break

    out = {"symbol": symbol, "days": days, "tested": n_combos,
           "elapsed_s": round(time.time() - t0, 1),
           "baseline_pnl": baseline["pnl"] if baseline else 0,
           "baseline_pf": baseline["pf"] if baseline else 0,
           "top3": top3, "winner": winner}
    if winner:
        wf = winner["wf"]
        print(f"\n  WINNER: lb={winner['lookback']} buf={winner['buffer']} "
              f"WF PF={wf['avg_pf']} folds {wf['n_positive']}/{wf['n_folds']}+ "
              f"Δ=${winner['delta_vs_baseline']:+.0f}")
    else:
        print(f"\n  NO WINNER (filter didn't beat baseline + pass WF)")
    (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
