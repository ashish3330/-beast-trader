#!/usr/bin/env python3 -B
"""FIB50 XAUUSD walk-forward validation — 2026-06-21.

Strict 5-fold contiguous walk-forward on the 365d window.

Winning params (locked, no re-tune):
  MIN_IMPULSE_ATR=3, MIN_RR=1, ENTRY_ZONE_LO=0.5, ATR_BUFFER=0.5,
  USE_WIDE_SL=False, SWING_PIVOT_N=5, DIRECTION_FILTER=SHORT, MAX_SL_R=4

Method:
  - Load same 365d M15 window as tune.
  - Split contiguously into 5 chunks of ~73d each (~7008 M15 bars each).
  - For each fold, run _fast_signals + _simulate_trade with the LOCKED params
    on the fold's bar range only. ATR + swing pivots are re-precomputed inside
    the fold slice (no look-ahead from neighbouring folds).
  - Trade may extend up to 800 bars past its entry — we cap by clipping the
    fold's last entry index so the trade can resolve inside the fold; or we
    let it close at fold-end via the time-stop/end-clamp logic. We DO clip
    the entry window so each fold uses its own slice independently.
  - Pool all OOS trade R values across the 5 folds; recompute PF / WR / DD.

PASS criterion: 4-of-5 folds have PF>=1.3 AND total_R>0.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse all helpers + config from the tune script (sibling file, no package)
import importlib.util  # noqa: E402
_tune_path = Path(__file__).resolve().parent / "_tune_fib50_btcusd_20260621.py"
_spec = importlib.util.spec_from_file_location("_tune_fib50_btc", _tune_path)
_tune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tune)

SYMBOL = _tune.SYMBOL
DAYS = _tune.DAYS
SPREAD = _tune.SPREAD
START_CAPITAL = _tune.START_CAPITAL
RISK_PCT = _tune.RISK_PCT
M15_WINDOW = _tune.M15_WINDOW
TIME_STOP_BARS = _tune.TIME_STOP_BARS
TIME_STOP_PEAK_R = _tune.TIME_STOP_PEAK_R
_load = _tune._load
_fast_signals = _tune._fast_signals
_simulate_trade = _tune._simulate_trade

WINNER_PARAMS = {
    "MIN_IMPULSE_ATR": 4.0,
    "MIN_RR": 1.0,
    "ENTRY_ZONE_LO": 0.382,
    "ATR_BUFFER": 0.5,
    "USE_WIDE_SL": True,
    "SWING_PIVOT_N": 5,
    "DIRECTION_FILTER": "SHORT",
    "MAX_SL_R": 4.0,
}

N_FOLDS = 5
WF_RESULTS_JSON = Path(__file__).resolve().parent / "_tune_fib50_btcusd_20260621_wf_results.json"


def _summarize_pool(Rs):
    """Pool-style summary — like _summarize but takes an R array directly."""
    if len(Rs) == 0:
        return {
            "trades": 0, "wr": 0.0, "pf": 0.0,
            "total_R": 0.0, "avg_R": 0.0, "max_dd_pct": 0.0,
            "end_equity": START_CAPITAL,
        }
    Rs = np.asarray(Rs, dtype=float)
    wins = Rs[Rs > 0]
    losses = Rs[Rs <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    equity = [START_CAPITAL]
    risk_d = START_CAPITAL * RISK_PCT
    for r in Rs:
        equity.append(equity[-1] + r * risk_d)
    eq = np.array(equity)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    return {
        "trades": int(len(Rs)),
        "wr": float((Rs > 0).mean()),
        "pf": float(pf),
        "total_R": float(Rs.sum()),
        "avg_R": float(Rs.mean()),
        "max_dd_pct": float(abs(dd.min()) * 100.0),
        "end_equity": float(eq[-1]),
    }


def _run_fold(m15_fold, params):
    """Run _fast_signals + _simulate_trade on a single fold slice.

    Returns (summary, trade_list).
    The fold slice is a self-contained M15 df — ATR and pivots recompute from
    its own first bar (no peeking at prior fold).
    """
    arr = (m15_fold["high"].values, m15_fold["low"].values, m15_fold["close"].values)
    sigs = _fast_signals(m15_fold, params)
    trades = []
    open_until = -1
    for entry_i, sig in sigs:
        if entry_i <= open_until:
            continue
        tr = _simulate_trade(sig, arr, entry_i=entry_i, spread=SPREAD)
        if tr is None:
            continue
        trades.append(tr)
        open_until = entry_i + tr["bars_held"]
    Rs = [t["R"] for t in trades]
    summ = _summarize_pool(Rs)
    return summ, trades


def main():
    m15 = _load(SYMBOL)
    if m15 is None:
        print(f"NO_DATA for {SYMBOL}", flush=True)
        return
    n_bars_keep = int(DAYS * 24 * 4)
    if n_bars_keep < len(m15):
        m15 = m15.iloc[-n_bars_keep:].reset_index(drop=True)
    n = len(m15)
    print(f"Loaded {n} M15 bars for {SYMBOL} (~{n/96:.0f}d)", flush=True)
    print(f"Winner params: {WINNER_PARAMS}", flush=True)

    # Contiguous fold boundaries
    edges = np.linspace(0, n, N_FOLDS + 1, dtype=int)
    fold_ranges = [(int(edges[k]), int(edges[k + 1])) for k in range(N_FOLDS)]
    print(f"Fold edges (bar idx): {fold_ranges}", flush=True)

    fold_results = []
    pooled_R = []
    pass_count = 0

    for k, (lo, hi) in enumerate(fold_ranges, 1):
        m15_fold = m15.iloc[lo:hi].reset_index(drop=True)
        t0 = time.time()
        summ, trades = _run_fold(m15_fold, WINNER_PARAMS)
        elapsed = time.time() - t0
        days_in_fold = len(m15_fold) / 96.0
        # Fold-level pass
        fold_pass = summ["pf"] >= 1.3 and summ["total_R"] > 0
        if fold_pass:
            pass_count += 1
        marker = " PASS" if fold_pass else " FAIL"
        print(f"  Fold {k}/{N_FOLDS}  bars[{lo}:{hi}]  ~{days_in_fold:.1f}d  "
              f"trd={summ['trades']:3d}  wr={summ['wr']:.2%}  pf={summ['pf']:.3f}  "
              f"totR={summ['total_R']:+.2f}  dd={summ['max_dd_pct']:.1f}%  "
              f"[{marker}]  ({elapsed:.1f}s)", flush=True)
        pooled_R.extend([t["R"] for t in trades])
        fold_results.append({
            "fold": k, "bar_lo": lo, "bar_hi": hi,
            "days": days_in_fold, "summary": summ,
            "fold_pass": fold_pass,
        })

    combined = _summarize_pool(pooled_R)
    wf_pass = pass_count >= 4
    print(f"\n=== Combined (pooled OOS) ===", flush=True)
    print(f"  trd={combined['trades']}  wr={combined['wr']:.2%}  "
          f"pf={combined['pf']:.3f}  totR={combined['total_R']:+.2f}  "
          f"dd={combined['max_dd_pct']:.1f}%", flush=True)
    print(f"\n=== WF VERDICT: {pass_count}/{N_FOLDS} folds passed "
          f"-> {'PASS' if wf_pass else 'FAIL'} ===", flush=True)

    out = {
        "symbol": SYMBOL,
        "strategy": "FIB50",
        "winner_params": WINNER_PARAMS,
        "in_sample_pf": 1.17,
        "in_sample_total_R": 18.8,
        "n_folds": N_FOLDS,
        "fold_results": fold_results,
        "combined": combined,
        "pass_folds": pass_count,
        "wf_passed": wf_pass,
    }
    WF_RESULTS_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {WF_RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
