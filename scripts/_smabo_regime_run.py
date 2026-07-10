#!/usr/bin/env python3 -B
"""Regime-conditional TP sweep runner for SMABO. Emits ONE JSON line.

Reuses the real backtest sim (backtest.sma_breakout_backtest) verbatim — no
re-implementation, so zero parity drift. Regime-TP params are passed via env
(SMABO_<SYM>_<KEY>) which agent/sma_breakout.py reads directly.

  Full window:   python3 -B scripts/_smabo_regime_run.py --symbol BTCUSD --days 365
  WF fold i/k:   ... --folds 4 --fold 0        (non-overlapping chronological)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest import sma_breakout_backtest as bt  # noqa: E402


def run(symbol, days=None, fold=None, folds=None):
    m15 = bt._load(symbol, "m15")
    if m15 is None:
        return {"status": "NO_DATA"}
    if days and days > 0:
        keep = int(days * 24 * 4)
        if keep < len(m15):
            m15 = m15.iloc[-keep:].reset_index(drop=True)
    if folds and folds > 1 and fold is not None:
        n = len(m15)
        w = n // folds
        lo = fold * w
        hi = n if fold == folds - 1 else (fold + 1) * w
        m15 = m15.iloc[lo:hi].reset_index(drop=True)
    n = len(m15)
    if n < bt.MIN_BARS_FOR_BT:
        return {"status": f"INSUFFICIENT_DATA ({n})"}
    h1 = bt._resample_h1(m15)
    spread = bt.SPREAD.get(symbol, 0.0002)
    state = bt._FakeState(m15, h1)
    strat = bt.SMABreakoutStrategy(state)
    arr = (m15["high"].values, m15["low"].values, m15["close"].values)
    trades = []
    open_until = -1
    for i in range(max(800, 60), n - 1):
        if i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(symbol, None)
        sig = strat.evaluate(symbol)
        if sig is None:
            continue
        tr = bt._simulate_trade(sig, arr, entry_i=i, spread=spread)
        if tr is None:
            continue
        trades.append(tr)
        open_until = i + tr["bars_held"]
    return bt._summarize(symbol, trades, n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    a = ap.parse_args()
    r = run(a.symbol, a.days, a.fold, a.folds)
    keys = ("status", "trades", "wr", "tp1_rate", "pf", "avg_R",
            "total_R", "max_dd_pct", "max_cons_loss")
    out = {"symbol": a.symbol}
    for k in keys:
        if k in r:
            v = r[k]
            out[k] = round(v, 4) if isinstance(v, float) else v
    print(json.dumps(out))
