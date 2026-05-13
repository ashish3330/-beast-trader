#!/usr/bin/env python3 -B
"""
PHASE 3b — per-symbol component weight tune.

For each symbol, sweeps weights on the 11 momentum scoring components.
Focused approach: sweep the TOP 3 highest-baseline-impact components
together; keep others at 1.0. Avoids the 5^11 explosion.

Components (max raw contribution):
  ema_stack(2.5), supertrend(1.5), macd_signal(1.5), macd_hist(1.0),
  rsi(1.0), candle_pattern(2.0), heikin_ashi(1.0), structure(1.5),
  breakout(2.5), momentum_vel(0.5), trend_persist(0.5)

Top-3 by leverage: ema_stack, breakout, candle_pattern (each max 2.0-2.5).
Sweep weights [0.5, 0.75, 1.0, 1.25, 1.5] on these three.
= 5^3 = 125 combos per symbol.

Plus 5-fold walk-forward validation on top 3.

Output: backtest/results/phase3b_component_weights/<SYMBOL>.json
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

PHASE1_DIR = ROOT / "backtest" / "results" / "full_tune_20260513"
PHASE2_DIR = ROOT / "backtest" / "results" / "phase2_hard_tune"
OUT_DIR = ROOT / "backtest" / "results" / "phase3b_component_weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP3_COMPONENTS = ["ema_stack", "breakout", "candle_pattern"]
WEIGHT_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]


def calmar(pnl_pct, max_dd_pct, days):
    if max_dd_pct <= 0 or days <= 0:
        return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / max_dd_pct


def score(r, days):
    if not r or r.get("trades", 0) < 15:
        return -1e9
    pnl_pct = r["pnl"] / 1000 * 100
    dd = max(r["dd"], 0.5)
    return calmar(pnl_pct, dd, days) * math.sqrt(r["trades"])


def get_baseline_params(symbol):
    p2 = PHASE2_DIR / f"{symbol}.json"
    if p2.exists():
        d = json.loads(p2.read_text())
        w = d.get("phase2_winner")
        if w: return {"sl": w["sl"], "tp_r": w["tp_r"]}
    p1 = PHASE1_DIR / f"{symbol}.json"
    if p1.exists():
        d = json.loads(p1.read_text())
        b = d.get("best")
        if b: return {"sl": b["sl"], "tp_r": b["tp_r"]}
    return {"sl": 1.0, "tp_r": [1.5, 2.5, 4.0]}


def run_combo(symbol, days, base, weights):
    p = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
        "sl_atr_mult": base["sl"],
        "sub_tp_r": base["tp_r"],
        "component_weights": weights,
    }
    try:
        return backtest_symbol(symbol, days=days, params=p, verbose=False)
    except Exception:
        return None


def walk_forward_5fold(symbol, base, weights):
    folds = []
    for d in [60, 90, 120, 150, 180]:
        r = run_combo(symbol, d, base, weights)
        if r and r.get("trades", 0) > 5:
            folds.append({
                "days": d, "trades": r["trades"], "pf": r["pf"],
                "pnl": r["pnl"], "dd": r["dd"],
            })
    return folds


def main():
    if len(sys.argv) < 2:
        print("Usage: phase3b_component_weights.py SYMBOL"); sys.exit(1)
    symbol = sys.argv[1]
    days = 180
    base = get_baseline_params(symbol)
    print(f"\n=== PHASE 3b {symbol} (baseline SL={base['sl']}) ===")

    results = []
    n = len(WEIGHT_VALUES) ** 3
    print(f"  Grid: {n} combos")
    t0 = time.time()
    i = 0
    for w1 in WEIGHT_VALUES:
        for w2 in WEIGHT_VALUES:
            for w3 in WEIGHT_VALUES:
                i += 1
                weights = {
                    TOP3_COMPONENTS[0]: w1,
                    TOP3_COMPONENTS[1]: w2,
                    TOP3_COMPONENTS[2]: w3,
                }
                r = run_combo(symbol, days, base, weights)
                if r and r.get("trades", 0) > 0:
                    results.append({
                        "weights": weights,
                        "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
                        "pnl": r["pnl"], "dd": r["dd"],
                        "score": round(score(r, days), 1),
                    })
                if i % 25 == 0:
                    print(f"  {i}/{n} ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        (OUT_DIR / f"{symbol}.json").write_text(json.dumps({"symbol": symbol, "winner": None}))
        print(f"  NO RESULTS"); return

    results.sort(key=lambda x: -x["score"])
    top3 = results[:3]
    print(f"\n  TOP 3:")
    for c in top3:
        w = c["weights"]
        print(f"    ema={w['ema_stack']} breakout={w['breakout']} pattern={w['candle_pattern']} "
              f"PnL=${c['pnl']:.0f} PF={c['pf']:.2f} n={c['trades']}")

    # WF top 3
    for c in top3:
        wf = walk_forward_5fold(symbol, base, c["weights"])
        c["wf_folds"] = wf
        if wf:
            c["wf_avg_pf"] = round(sum(f["pf"] for f in wf) / len(wf), 2)
            c["wf_avg_pnl"] = round(sum(f["pnl"] for f in wf) / len(wf), 0)
            c["wf_n_positive"] = sum(1 for f in wf if f["pnl"] > 0)
            c["wf_n_folds"] = len(wf)

    winner = None
    for c in top3:
        if (c.get("wf_avg_pf", 0) > 1.3 and
                c.get("wf_n_positive", 0) >= max(3, int(0.6 * c.get("wf_n_folds", 5)))):
            winner = c; break

    out = {
        "symbol": symbol, "days": days, "baseline": base,
        "elapsed_s": round(time.time() - t0, 1),
        "tested": n,
        "top3": top3, "winner": winner,
    }
    if winner:
        w = winner["weights"]
        print(f"\n  WINNER: ema={w['ema_stack']} brk={w['breakout']} pat={w['candle_pattern']} "
              f"WF PF={winner['wf_avg_pf']} folds {winner['wf_n_positive']}/{winner['wf_n_folds']}+")
    else:
        print(f"\n  no winner passed WF")

    (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
