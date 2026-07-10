#!/usr/bin/env python3 -B
"""Unified FIB50 walk-forward validator — invoke per (symbol, params).

CLI:
    python3 -B scripts/_tune_fib50_wf_unified_20260621.py BTCUSD '{"DIRECTION_FILTER":"SHORT",...}'
    python3 -B scripts/_tune_fib50_wf_unified_20260621.py EURUSD '{"DIRECTION_FILTER":"SHORT",...}'

Reuses backtest/fib50_backtest.py helpers directly. Splits the 365d window
into 5 contiguous folds and runs Fib50Strategy with the supplied params dict
on each fold independently. No look-ahead — bars before the fold start are
allowed as indicator warmup (just like the live detector), but trades only
fire for bars inside the fold range.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.fib50_strategy import Fib50Strategy  # noqa: E402
import backtest.fib50_backtest as bt  # noqa: E402

DAYS = 365
N_FOLDS = 5


def run_fold(m15, params, symbol, fold_lo, fold_hi, warmup):
    """Run BT on bar range [fold_lo, fold_hi) using params."""
    H = m15["high"].values
    L = m15["low"].values
    C = m15["close"].values
    n = len(m15)
    spread = bt.SPREAD.get(symbol, 0.0002)
    state = bt._FakeState(m15)
    strat = Fib50Strategy(state, params=params)

    trades = []
    open_until = -1
    start_i = max(fold_lo, warmup)
    end_i = min(fold_hi, n - 1)
    for i in range(start_i, end_i):
        if i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(symbol, None)
        sig = strat.evaluate(symbol)
        if sig is None:
            continue
        tr = bt._simulate_trade(sig, (H, L, C), entry_i=i, spread=spread)
        if tr is None:
            continue
        trades.append(tr)
        open_until = i + tr["bars_held"]

    if not trades:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "total_R": 0.0,
                "max_dd_pct": 0.0, "avg_R": 0.0}

    Rs = np.array([t["R"] for t in trades])
    wins = Rs[Rs > 0]
    losses = Rs[Rs <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    equity = [bt.START_CAPITAL]
    risk_d = bt.START_CAPITAL * bt.RISK_PCT
    for r in Rs:
        equity.append(equity[-1] + r * risk_d)
    eq = np.array(equity)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    return {
        "trades": int(len(trades)),
        "pf": float(pf),
        "wr": float((Rs > 0).mean()),
        "total_R": float(Rs.sum()),
        "avg_R": float(Rs.mean()),
        "max_dd_pct": float(abs(dd.min()) * 100.0),
        "trade_Rs": Rs.tolist(),
    }


def main():
    symbol = sys.argv[1]
    params = json.loads(sys.argv[2])
    out_json = sys.argv[3] if len(sys.argv) > 3 else (
        f"scripts/_tune_fib50_{symbol.lower()}_20260621_wf_winner.json"
    )

    m15 = bt._load(symbol, "m15")
    if m15 is None:
        print(f"NO_DATA for {symbol}")
        sys.exit(1)
    n_keep = int(DAYS * 24 * 4)
    if n_keep < len(m15):
        m15 = m15.iloc[-n_keep:].reset_index(drop=True)
    n = len(m15)
    print(f"Loaded {symbol}: {n} bars (~{n/96:.0f}d)")
    print(f"Winner params: {params}")

    warmup = max(bt.M15_WINDOW, 120)
    usable_lo = warmup
    usable_hi = n - 1
    total_usable = usable_hi - usable_lo
    fold_size = total_usable // N_FOLDS

    folds = []
    all_Rs = []
    pass_count = 0
    for k in range(N_FOLDS):
        f_lo = usable_lo + k * fold_size
        f_hi = (usable_lo + (k + 1) * fold_size) if k < N_FOLDS - 1 else usable_hi
        s = run_fold(m15, params, symbol, f_lo, f_hi, warmup)
        passed = (s["pf"] >= 1.3 and s["total_R"] > 0)
        if passed:
            pass_count += 1
        all_Rs.extend(s.get("trade_Rs", []))
        s_print = {k_: v for k_, v in s.items() if k_ != "trade_Rs"}
        print(f"  Fold {k+1}: {s_print}  pass={passed}")
        folds.append({"fold": k+1, "lo": f_lo, "hi": f_hi, "summary": s_print, "pass": passed})

    # Combined pool
    if all_Rs:
        Rs = np.array(all_Rs)
        wins = Rs[Rs > 0]
        losses = Rs[Rs <= 0]
        pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
        equity = [bt.START_CAPITAL]
        risk_d = bt.START_CAPITAL * bt.RISK_PCT
        for r in Rs:
            equity.append(equity[-1] + r * risk_d)
        eq = np.array(equity)
        dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
        combined = {
            "trades": int(len(Rs)),
            "pf": float(pf),
            "wr": float((Rs > 0).mean()),
            "total_R": float(Rs.sum()),
            "avg_R": float(Rs.mean()),
            "max_dd_pct": float(abs(dd.min()) * 100.0),
        }
    else:
        combined = {"trades": 0, "pf": 0.0, "wr": 0.0, "total_R": 0.0,
                    "avg_R": 0.0, "max_dd_pct": 0.0}

    n_pos_folds = sum(1 for f in folds if f["summary"]["total_R"] > 0)
    wf_passed = pass_count >= 4  # strict 4-of-5
    print(f"\nCombined: PF={combined['pf']:.2f} R={combined['total_R']:+.1f} "
          f"trd={combined['trades']} DD={combined['max_dd_pct']:.1f}% "
          f"folds_pos={n_pos_folds}/5 strict_pass={pass_count}/5")
    out = {
        "symbol": symbol,
        "winning_params": params,
        "n_folds": N_FOLDS,
        "fold_summaries": folds,
        "combined": combined,
        "pass_count_strict": pass_count,
        "n_positive_folds": n_pos_folds,
        "wf_passed_strict": wf_passed,
    }
    Path(out_json).write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
