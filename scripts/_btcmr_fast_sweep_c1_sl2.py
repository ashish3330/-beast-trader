#!/usr/bin/env python3 -B
"""Fast exact-replica sweep of scripts/_btcmr_run.py for CONFIRM=1, SL_ATR=2.

Indicators (SMA/std/ATR/ADX/RSI) in agent/btc_mean_reversion.py are forward
recursions computed on prefixes C[:i+1]; a single full-array pass yields
identical values at each i. All swept params (BB_MULT, RSI_LOW/HIGH, ADX_MAX,
TIME_STOP_BARS) are pure thresholds, so one precompute per dataset serves all
54 configs. Simulate loop is copied verbatim from the runner semantics.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest import sma_breakout_backtest as bt  # noqa: E402

SPREAD = float(os.getenv("BTCMR_SPREAD", "5.0"))
BB_P = 20
RSI_P = 2
PER = 14  # ATR/ADX period


def precompute(m15):
    H = m15["high"].values.astype(float)
    L = m15["low"].values.astype(float)
    C = m15["close"].values.astype(float)
    n = len(C)

    # --- SMA(BB_P) exact (_sma cumsum formula) ---
    cs = np.cumsum(np.insert(C, 0, 0.0))
    mid = np.full(n, np.nan)
    if n >= BB_P:
        mid[BB_P - 1:] = (cs[BB_P:] - cs[:-BB_P]) / BB_P

    # --- rolling std ddof=0, bit-identical to np.std on each slice ---
    sd = np.full(n, np.nan)
    for i in range(BB_P - 1, n):
        sd[i] = np.std(C[i - BB_P + 1:i + 1], ddof=0)

    # --- Wilder ATR full pass (atr[i] == _atr(prefix i)) ---
    tr = np.empty(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
    atr = np.zeros(n)
    if n >= PER + 1:
        atr[PER] = tr[1:PER + 1].mean()
        k = 1.0 / PER
        for i in range(PER + 1, n):
            atr[i] = atr[i - 1] * (1 - k) + tr[i] * k
    # prefix with n < PER+1 -> _atr returns 0.0; atr[:PER] left 0

    # --- Wilder ADX: replicate _adx per prefix ---
    up = H[1:] - H[:-1]
    dn = L[:-1] - L[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr1 = np.maximum.reduce([H[1:] - L[1:], np.abs(H[1:] - C[:-1]),
                             np.abs(L[1:] - C[:-1])])
    k = 1.0 / PER

    def _wilder(x):
        out = np.empty(len(x))
        out[:PER] = np.nan
        out[PER - 1] = x[:PER].sum()
        for i in range(PER, len(x)):
            out[i] = out[i - 1] - out[i - 1] * k + x[i]
        return out

    atr_s = _wilder(tr1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * _wilder(plus_dm) / atr_s
        mdi = 100.0 * _wilder(minus_dm) / atr_s
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    valid = np.isfinite(dx)
    vidx = np.where(valid)[0]           # indices into dx (length n-1)
    dxv = dx[vidx]
    # adx_at[i] = _adx(H[:i+1],...) ; prefix diff array = dx[:i]
    adx = np.zeros(n)
    for i in range(n):
        if i + 1 < 2 * PER + 1:
            adx[i] = 0.0
            continue
        cnt = np.searchsorted(vidx, i)  # valid entries with index < i
        if cnt < PER:
            adx[i] = 0.0
            continue
        v = dxv[cnt - PER:cnt].mean()
        adx[i] = float(v) if np.isfinite(v) else 0.0

    # --- Wilder RSI(2) full pass ---
    rsi = np.full(n, 50.0)
    d = np.diff(C)
    if n >= RSI_P + 1:
        gain = np.where(d > 0, d, 0.0)
        loss = np.where(d < 0, -d, 0.0)
        ag = gain[:RSI_P].mean()
        al = loss[:RSI_P].mean()
        kk = 1.0 / RSI_P
        rsi[RSI_P] = 100.0 if al == 0 else float(100.0 - 100.0 / (1.0 + ag / al))
        for j in range(RSI_P, len(d)):
            ag = ag * (1 - kk) + gain[j] * kk
            al = al * (1 - kk) + loss[j] * kk
            rsi[j + 1] = 100.0 if al == 0 else float(100.0 - 100.0 / (1.0 + ag / al))
    return dict(H=H, L=L, C=C, n=n, mid=mid, sd=sd, atr=atr, adx=adx, rsi=rsi)


def run_config(pre, bb_m, rsi_lo, rsi_hi, adx_max, sl_atr, time_stop):
    H, L, C = pre["H"], pre["L"], pre["C"]
    n = pre["n"]
    mid, sd, atr, adx, rsi = pre["mid"], pre["sd"], pre["atr"], pre["adx"], pre["rsi"]

    lower = mid - bb_m * sd
    upper = mid + bb_m * sd
    ok = np.isfinite(mid) & (atr > 0) & (adx < adx_max)
    idx = np.arange(n)
    gate = idx >= max(BB_P, 30) + 2
    prev_out_lo = np.zeros(n, dtype=bool)
    prev_out_hi = np.zeros(n, dtype=bool)
    prev_out_lo[1:] = C[:-1] < lower[:-1]
    prev_out_hi[1:] = C[:-1] > upper[:-1]
    long_sig = prev_out_lo & (C > lower) & (rsi < rsi_lo)
    short_sig = prev_out_hi & (C < upper) & (rsi > rsi_hi)
    cand = ok & gate & (long_sig | short_sig)
    cand_idx = np.where(cand[60:n - 2])[0] + 60

    trades = []
    open_until = -1
    for i in cand_idx:
        if i <= open_until:
            continue
        is_long = bool(long_sig[i])
        entry = float(C[i])
        m = float(mid[i])
        if is_long:
            sl = entry - sl_atr * float(atr[i])
            tp1 = m
            if sl >= entry or tp1 <= entry:
                continue
        else:
            sl = entry + sl_atr * float(atr[i])
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
        r = gross / risk
        trades.append({"R": float(r),
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


def main():
    m15 = bt._load("BTCUSD", "m15")
    keep = int(365 * 24 * 4)
    if keep < len(m15):
        m15 = m15.iloc[-keep:].reset_index(drop=True)
    full_df = m15
    w = len(full_df) // 4
    fold2_df = full_df.iloc[2 * w:3 * w].reset_index(drop=True)
    fold3_df = full_df.iloc[3 * w:].reset_index(drop=True)

    pre_full = precompute(full_df)
    pre_f2 = precompute(fold2_df)
    pre_f3 = precompute(fold3_df)

    SL_ATR = 2.0
    results = []
    for bb in (2.0, 2.5, 3.0):
        for rl, rh in ((10.0, 90.0), (20.0, 80.0), (30.0, 70.0)):
            for adx in (20.0, 25.0, 30.0):
                for ts in (16, 24):
                    row = {"bb": bb, "rl": rl, "rh": rh, "adx": adx, "ts": ts}
                    for name, pre in (("full", pre_full), ("fold2", pre_f2),
                                      ("fold3", pre_f3)):
                        tr = run_config(pre, bb, rl, rh, adx, SL_ATR, ts)
                        row[name] = summarize("BTCUSD", tr, pre["n"])
                    results.append(row)
                    print(json.dumps(row), flush=True)
    with open("/tmp/btcmr_fast_c1_sl2.json", "w") as f:
        json.dump(results, f, indent=1)
    print("SWEEP_DONE", len(results))


if __name__ == "__main__":
    main()
