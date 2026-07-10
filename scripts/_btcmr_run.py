#!/usr/bin/env python3 -B
"""Backtest runner for the BTC mean-reversion strategy (agent/btc_mean_reversion).
Emits ONE JSON line. Supports --days and --fold i/--folds k (walk-forward).

Fill model: enter at signal-bar close, simulate forward bar-by-bar to TP1 (mean)
or SL, with a time-stop. SL assumed first if both touched in one bar (conservative).
Spread charged on entry+exit.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest import sma_breakout_backtest as bt   # noqa: E402  (reuse _load)
from agent import btc_mean_reversion as mr          # noqa: E402

SPREAD = float(os.getenv("BTCMR_SPREAD", "5.0"))     # BTC $ spread cost per side


def simulate(m15, override):
    H = m15["high"].values
    L = m15["low"].values
    C = m15["close"].values
    n = len(m15)
    time_stop = int(mr._param("TIME_STOP_BARS", override))
    trades = []
    open_until = -1
    start = max(60, 40)
    for i in range(start, n - 2):
        if i <= open_until:
            continue
        sig = mr.evaluate(m15, i, override)
        if sig is None:
            continue
        entry = sig["entry"]
        sl = sig["sl"]
        tp1 = sig["tp1"]
        is_long = sig["direction"] == "LONG"
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        exit_px = None
        exit_j = None
        for j in range(i + 1, min(i + 1 + time_stop, n)):
            hi, lo = H[j], L[j]
            if is_long:
                if lo <= sl:
                    exit_px, exit_j = sl, j
                    break
                if hi >= tp1:
                    exit_px, exit_j = tp1, j
                    break
            else:
                if hi >= sl:
                    exit_px, exit_j = sl, j
                    break
                if lo <= tp1:
                    exit_px, exit_j = tp1, j
                    break
        if exit_px is None:                       # time-stop at last bar's close
            exit_j = min(i + time_stop, n - 1)
            exit_px = C[exit_j]
        gross = (exit_px - entry) if is_long else (entry - exit_px)
        gross -= SPREAD                            # entry+exit spread cost
        r = gross / risk
        trades.append({"R": float(r), "dir": sig["direction"],
                       "tp1_hit": bool(abs(exit_px - tp1) < 1e-6)})
        open_until = exit_j
    return trades


def summarize(sym, trades, n):
    if not trades:
        return {"symbol": sym, "status": "OK", "trades": 0}
    R = np.array([t["R"] for t in trades])
    wins = R[R > 0]
    losses = R[R <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    eq = np.concatenate([[0.0], np.cumsum(R)])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak)
    return {
        "symbol": sym, "status": "OK", "trades": len(trades),
        "wr": round(float((R > 0).mean()), 4),
        "tp1_rate": round(float(np.mean([t["tp1_hit"] for t in trades])), 4),
        "pf": round(float(pf), 4), "avg_R": round(float(R.mean()), 4),
        "total_R": round(float(R.sum()), 4),
        "max_dd_R": round(float(abs(dd.min())), 4),
    }


def run(sym, days=None, fold=None, folds=None, override=None):
    m15 = bt._load(sym, "m15")
    if m15 is None:
        return {"status": "NO_DATA"}
    if days and days > 0:
        keep = int(days * 24 * 4)
        if keep < len(m15):
            m15 = m15.iloc[-keep:].reset_index(drop=True)
    if folds and folds > 1 and fold is not None:
        w = len(m15) // folds
        lo = fold * w
        hi = len(m15) if fold == folds - 1 else (fold + 1) * w
        m15 = m15.iloc[lo:hi].reset_index(drop=True)
    trades = simulate(m15, override or {})
    return summarize(sym, trades, len(m15))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSD")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.symbol, a.days, a.fold, a.folds)))
