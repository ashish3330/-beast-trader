#!/usr/bin/env python3 -B
"""M1 SCALPER backtest v2 — VECTORIZED + industry-research features.

Mean-reversion band fade (research #2) gated by session hours + ADX regime +
ATR-expansion (research #1), targeting the mean (clears cost), fixed R exit +
time-stop (567k-study: fixed beats trailing). REAL spread charged both sides.

All indicators precomputed as arrays (fast). Env params SCALP_<KEY>.
Emits ONE JSON line. --fold i / --folds k for walk-forward.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest import sma_breakout_backtest as bt   # noqa: E402  (_load)

SPREAD_PRICE = {"XAUUSD": 0.23, "BTCUSD": 16.94}


def _p(key, d):
    v = os.getenv(f"SCALP_{key}")
    try:
        return float(v) if v is not None else float(d)
    except Exception:
        return float(d)


def _wilder(s, period):
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


def run(sym, days=None, fold=None, folds=None):
    period = int(_p("PERIOD", 20))
    bb_mult = _p("BB_MULT", 2.0)
    rsi_p = int(_p("RSI_PERIOD", 2))
    rsi_lo, rsi_hi = _p("RSI_LOW", 10), _p("RSI_HIGH", 90)
    sl_atr = _p("SL_ATR", 1.0)
    tp_atr = _p("TP_ATR", 1.0)
    tp_mean = _p("TP_MEAN", 1) >= 1        # 1 = target the mean (default), 0 = fixed TP_ATR
    time_stop = int(_p("TIME_STOP", 12))
    adx_max = _p("ADX_MAX", 20)            # only fade when ADX < this (ranging)
    atr_expand = _p("ATR_EXPAND", 1) >= 1  # require ATR > its MA (live vol)
    h_start, h_end = int(_p("H_START", 0)), int(_p("H_END", 24))  # session hours (bar TZ)
    spread = _p("SPREAD", SPREAD_PRICE.get(sym, 0.5))

    m1 = bt._load(sym, "m1")
    if m1 is None:
        return {"status": "NO_DATA"}
    if days and days > 0:
        keep = int(days * 24 * 60)
        if keep < len(m1):
            m1 = m1.iloc[-keep:].reset_index(drop=True)
    if folds and folds > 1 and fold is not None:
        w = len(m1) // folds
        lo, hi = fold * w, (len(m1) if fold == folds - 1 else (fold + 1) * w)
        m1 = m1.iloc[lo:hi].reset_index(drop=True)
    n = len(m1)
    if n < period + 100:
        return {"status": f"INSUFFICIENT ({n})"}

    close = m1["close"]
    high, low = m1["high"], m1["low"]
    hour = pd.to_datetime(m1["time"]).dt.hour.values

    # ── vectorized indicators ──
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    lower = (mid - bb_mult * sd).values
    upper = (mid + bb_mult * sd).values
    midv = mid.values
    prev_c = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14).values
    atr_ma = pd.Series(atr).rolling(20).mean().values
    # RSI(rsi_p)
    d = close.diff()
    rsi = (100 - 100 / (1 + _wilder(d.clip(lower=0), rsi_p) / _wilder(-d.clip(upper=0), rsi_p).replace(0, np.nan))).values
    # ADX(14)
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_s = _wilder(tr, 14)
    pdi = 100 * _wilder(pd.Series(plus_dm), 14) / atr_s
    mdi = 100 * _wilder(pd.Series(minus_dm), 14) / atr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = _wilder(dx.fillna(0), 14).values

    H, L, C = high.values, low.values, close.values
    trades = []
    open_until = -1
    start = max(period + 30, 40)
    for i in range(start, n - 1):
        if i <= open_until:
            continue
        if not (h_start <= hour[i] < h_end):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(midv[i]):
            continue
        if adx[i] >= adx_max:                      # regime: skip trends
            continue
        if atr_expand and (not np.isfinite(atr_ma[i]) or a <= atr_ma[i]):
            continue
        c = C[i]
        direction = None
        if c < lower[i] and rsi[i] < rsi_lo:
            direction = "LONG"
        elif c > upper[i] and rsi[i] > rsi_hi:
            direction = "SHORT"
        if direction is None:
            continue
        entry = c
        if direction == "LONG":
            sl = entry - sl_atr * a
            tp = midv[i] if tp_mean else entry + tp_atr * a
            if tp <= entry:
                continue
        else:
            sl = entry + sl_atr * a
            tp = midv[i] if tp_mean else entry - tp_atr * a
            if tp >= entry:
                continue
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        exit_px, exit_j = None, None
        for j in range(i + 1, min(i + 1 + time_stop, n)):
            if direction == "LONG":
                if L[j] <= sl:
                    exit_px, exit_j = sl, j; break
                if H[j] >= tp:
                    exit_px, exit_j = tp, j; break
            else:
                if H[j] >= sl:
                    exit_px, exit_j = sl, j; break
                if L[j] <= tp:
                    exit_px, exit_j = tp, j; break
        if exit_px is None:
            exit_j = min(i + time_stop, n - 1)
            exit_px = C[exit_j]
        gross = (exit_px - entry) if direction == "LONG" else (entry - exit_px)
        gross -= spread
        trades.append(gross / risk)
        open_until = exit_j

    if not trades:
        return {"symbol": sym, "status": "OK", "trades": 0}
    R = np.array(trades)
    wins, losses = R[R > 0], R[R <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    eq = np.concatenate([[0.0], np.cumsum(R)])
    dd = float(abs((eq - np.maximum.accumulate(eq)).min()))
    return {"symbol": sym, "status": "OK", "trades": len(R),
            "per_day": round(len(R) / max(n / (24 * 60), 1), 1),
            "wr": round(float((R > 0).mean()), 4), "pf": round(float(pf), 4),
            "avg_R": round(float(R.mean()), 4), "total_R": round(float(R.sum()), 1),
            "max_dd_R": round(dd, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.symbol, a.days, a.fold, a.folds)))
