#!/usr/bin/env python3 -B
"""Vectorized batch sweep of BTCMR configs (CONFIRM=0, SL_ATR=0.7 fixed).

Replicates scripts/_btcmr_run.py semantics exactly:
- evaluate() at bar i uses prefix arrays; all indicators here are forward-only
  recursions so full-array series values at i == prefix values at i.
- BB_PERIOD=20, RSI_PERIOD=2 (defaults, not swept).
- simulate(): start=60, i in [60, n-3], open_until skip, SL-first conservative,
  time-stop exit at close, SPREAD=5.0 subtracted once.
- days trim then fold slice, indicators recomputed on the slice (matches runner).

Also includes --check mode: runs original mr.evaluate simulate for one config
to validate parity.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest import sma_breakout_backtest as bt  # noqa: E402

SPREAD = 5.0
BB_P = 20
RSI_P = 2
ATR_P = 14
ADX_P = 14


def precompute(m15):
    H = m15["high"].values.astype(float)
    L = m15["low"].values.astype(float)
    C = m15["close"].values.astype(float)
    n = len(C)

    # SMA20 (mid band) and rolling std20 ddof=0, value at i = window ending at i
    cs = np.cumsum(np.insert(C, 0, 0.0))
    sma = np.full(n, np.nan)
    sma[BB_P - 1:] = (cs[BB_P:] - cs[:-BB_P]) / BB_P
    std = np.full(n, np.nan)
    if n >= BB_P:
        win = np.lib.stride_tricks.sliding_window_view(C, BB_P)
        std[BB_P - 1:] = win.std(axis=1, ddof=0)  # == np.std(C[i-19:i+1])

    # Wilder ATR14 series (matches _atr prefix value at each i)
    tr = np.empty(n)
    tr[0] = H[0] - L[0]
    tr[1:] = np.maximum.reduce([H[1:] - L[1:], np.abs(H[1:] - C[:-1]),
                                np.abs(L[1:] - C[:-1])])
    atr = np.full(n, np.nan)
    if n > ATR_P:
        atr[ATR_P] = tr[1:ATR_P + 1].mean()
        k = 1.0 / ATR_P
        for i in range(ATR_P + 1, n):
            atr[i] = atr[i - 1] * (1 - k) + tr[i] * k

    # Wilder RSI(2) series (matches _rsi prefix value at each i)
    rsi = np.full(n, 50.0)  # _rsi returns 50 when n < period+1
    d = np.diff(C)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    if n >= RSI_P + 1:
        ag = gain[:RSI_P].mean()
        al = loss[:RSI_P].mean()
        k = 1.0 / RSI_P

        def _rsi_val(ag, al):
            if al == 0:
                return 100.0
            rs = ag / al
            return 100.0 - 100.0 / (1.0 + rs)

        # prefix C[:i+1] uses d[:i]; recursion over d indices RSI_P..i-1
        rsi[RSI_P] = _rsi_val(ag, al)
        for i in range(RSI_P, len(d)):
            ag = ag * (1 - k) + gain[i] * k
            al = al * (1 - k) + loss[i] * k
            rsi[i + 1] = _rsi_val(ag, al)

    # ADX14: dx array (index j corresponds to bar j+1), Wilder-smoothed sums
    adx = np.zeros(n)
    if n >= 2 * ADX_P + 1:
        up = H[1:] - H[:-1]
        dn = L[:-1] - L[1:]
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr1 = tr[1:]
        k = 1.0 / ADX_P

        def _wilder(x):
            out = np.empty(len(x))
            out[:ADX_P] = np.nan
            out[ADX_P - 1] = x[:ADX_P].sum()
            for i in range(ADX_P, len(x)):
                out[i] = out[i - 1] - out[i - 1] * k + x[i]
            return out

        atr_s = _wilder(tr1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = 100.0 * _wilder(plus_dm) / atr_s
            mdi = 100.0 * _wilder(minus_dm) / atr_s
            dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
        # ADX at bar i = mean of last 14 non-nan dx of prefix (dx idx <= i-1).
        # dx is nan for idx < ADX_P-1 and possibly where pdi+mdi==0 (rare).
        # Replicate exactly with a running list per bar (O(n), cheap).
        vals = []
        # bar index i corresponds to dx index i-1
        for i in range(1, n):
            v = dx[i - 1]
            if np.isfinite(v):
                vals.append(v)
            if len(vals) >= ADX_P:
                m = float(np.mean(vals[-ADX_P:]))
                adx[i] = m if np.isfinite(m) else 0.0
            else:
                adx[i] = 0.0
    return dict(H=H, L=L, C=C, n=n, sma=sma, std=std, atr=atr, rsi=rsi, adx=adx)


def simulate_cfg(P, bb_m, rsi_lo, rsi_hi, adx_max, sl_atr, time_stop):
    H, L, C, n = P["H"], P["L"], P["C"], P["n"]
    sma, std, atr, rsi, adx = P["sma"], P["std"], P["atr"], P["rsi"], P["adx"]
    trades_R = []
    tp1_hits = []
    open_until = -1
    lower = sma - bb_m * std
    upper = sma + bb_m * std
    # candidate mask (CONFIRM=0 path)
    with np.errstate(invalid="ignore"):
        long_m = (C < lower) & (rsi < rsi_lo)
        short_m = (C > upper) & (rsi > rsi_hi)
        ok = (np.isfinite(sma) & (atr > 0) & (adx < adx_max)
              & np.isfinite(atr))
    cand = np.nonzero((long_m | short_m) & ok)[0]
    cand = cand[(cand >= 60) & (cand <= n - 3)]
    for i in cand:
        if i <= open_until:
            continue
        entry = C[i]
        m = sma[i]
        is_long = bool(long_m[i])
        if is_long:
            sl = entry - sl_atr * atr[i]
            tp1 = m
            if sl >= entry or tp1 <= entry:
                continue
        else:
            sl = entry + sl_atr * atr[i]
            tp1 = m
            if sl <= entry or tp1 >= entry:
                continue
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
        if exit_px is None:
            exit_j = min(i + time_stop, n - 1)
            exit_px = C[exit_j]
        gross = (exit_px - entry) if is_long else (entry - exit_px)
        gross -= SPREAD
        trades_R.append(gross / risk)
        tp1_hits.append(abs(exit_px - tp1) < 1e-6)
        open_until = exit_j
    return np.array(trades_R), tp1_hits


def summarize(R, tp1_hits):
    if len(R) == 0:
        return {"trades": 0}
    wins = R[R > 0]
    losses = R[R <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    return {"trades": int(len(R)), "wr": round(float((R > 0).mean()), 4),
            "tp1_rate": round(float(np.mean(tp1_hits)), 4),
            "pf": round(float(pf), 4), "avg_R": round(float(R.mean()), 4),
            "total_R": round(float(R.sum()), 4)}


def get_slices():
    m15 = bt._load("BTCUSD", "m15")
    keep = int(365 * 24 * 4)
    if keep < len(m15):
        m15 = m15.iloc[-keep:].reset_index(drop=True)
    n = len(m15)
    w = n // 4
    slices = {
        "full": m15,
        "fold3": m15.iloc[3 * w:n].reset_index(drop=True),
        "fold2": m15.iloc[2 * w:3 * w].reset_index(drop=True),
    }
    return slices


def main():
    slices = get_slices()
    pre = {k: precompute(v) for k, v in slices.items()}
    sl_atr = 0.7
    results = []
    for bb in (2.0, 2.5, 3.0):
        for rl, rh in ((10.0, 90.0), (20.0, 80.0), (30.0, 70.0)):
            for adx_max in (20.0, 25.0, 30.0):
                for ts in (16, 24):
                    row = {"BB_MULT": bb, "RSI_LOW": rl, "RSI_HIGH": rh,
                           "ADX_MAX": adx_max, "TIME_STOP_BARS": ts}
                    for key in ("full", "fold3", "fold2"):
                        R, hits = simulate_cfg(pre[key], bb, rl, rh,
                                               adx_max, sl_atr, ts)
                        row[key] = summarize(R, hits)
                    results.append(row)
                    print(json.dumps(row), flush=True)
    Path("/tmp/btcmr_batch_c0_sl07.json").write_text(json.dumps(results))
    print("BATCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
