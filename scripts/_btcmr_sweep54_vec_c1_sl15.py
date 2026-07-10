#!/usr/bin/env python3 -B
"""Vectorized 54-config BTCMR sweep. CONFIRM=1, SL_ATR=1.5 fixed.

Exact-parity reimplementation of agent/btc_mean_reversion.evaluate + the
scripts/_btcmr_run.py fill model, with indicators precomputed ONCE per dataset
(the originals recompute Wilder chains per bar = O(n^2), ~30 min/run).

All Wilder recursions (RSI2 / ATR14 / ADX14 dx-chain) are prefix-stable, so a
single sequential pass gives bit-identical values to per-bar recomputation from
index 0. Validated against brute-force mr.evaluate before sweeping.

Output: /tmp/btcmr_sweep54_c1_sl15_vec.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for k in list(os.environ):
    if k.startswith("BTCMR_"):
        del os.environ[k]

from backtest import sma_breakout_backtest as bt   # noqa: E402
from agent import btc_mean_reversion as mr         # noqa: E402

SPREAD = 5.0
BB_P = 20
RSI_P = 2
SL_ATR = 1.5
CONFIRM = 1


def precompute(m15):
    H = m15["high"].values.astype(float)
    L = m15["low"].values.astype(float)
    C = m15["close"].values.astype(float)
    n = len(C)

    # SMA20 (mid band)
    cs = np.cumsum(np.insert(C, 0, 0.0))
    mid = np.full(n, np.nan)
    if n >= BB_P:
        mid[BB_P - 1:] = (cs[BB_P:] - cs[:-BB_P]) / BB_P

    # rolling population std over BB_P (two-pass per window for numerical parity
    # with np.std; windowed via stride tricks -> identical fp path as np.std? Use
    # direct per-window np.std for exactness -- still fast with sliding_window_view)
    from numpy.lib.stride_tricks import sliding_window_view
    sd = np.full(n, np.nan)
    if n >= BB_P:
        w = sliding_window_view(C, BB_P)
        sd[BB_P - 1:] = w.std(axis=1, ddof=0)

    # Wilder RSI(2), sequential (prefix-stable vs per-bar recompute)
    rsi = np.full(n, 50.0)
    d = np.diff(C)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    if n >= RSI_P + 1:
        ag = gain[:RSI_P].mean()
        al = loss[:RSI_P].mean()
        k = 1.0 / RSI_P
        # value at bar i uses d[:i]; first defined at i = RSI_P
        rsi[RSI_P] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
        for j in range(RSI_P, len(d)):
            ag = ag * (1 - k) + gain[j] * k
            al = al * (1 - k) + loss[j] * k
            rsi[j + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)

    # Wilder ATR14, sequential
    per = 14
    atr = np.zeros(n)
    if n >= per + 1:
        tr = np.empty(n)
        tr[0] = H[0] - L[0]
        tr[1:] = np.maximum.reduce([H[1:] - L[1:],
                                    np.abs(H[1:] - C[:-1]),
                                    np.abs(L[1:] - C[:-1])])
        a = tr[1:per + 1].mean()
        atr[per] = a
        k = 1.0 / per
        for j in range(per + 1, n):
            a = a * (1 - k) + tr[j] * k
            atr[j] = a

    # ADX14 per _adx(): dx chain sequential, ADX at bar i = mean of last 14
    # non-nan dx among dx[:i] (dx index j corresponds to bar j+1), 0 if <14 valid
    # or i+1 < 2*per+1.
    adx = np.zeros(n)
    if n >= 2:
        up = H[1:] - H[:-1]
        dn = L[:-1] - L[1:]
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr1 = np.maximum.reduce([H[1:] - L[1:],
                                 np.abs(H[1:] - C[:-1]),
                                 np.abs(L[1:] - C[:-1])])
        k = 1.0 / per

        def _wilder(x):
            out = np.empty(len(x))
            out[:per] = np.nan
            if len(x) >= per:
                out[per - 1] = x[:per].sum()
                for j in range(per, len(x)):
                    out[j] = out[j - 1] - out[j - 1] * k + x[j]
            return out

        atr_s = _wilder(tr1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = 100.0 * _wilder(plus_dm) / atr_s
            mdi = 100.0 * _wilder(minus_dm) / atr_s
            dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
        valid = np.isfinite(dx)
        vidx = np.nonzero(valid)[0]
        vval = dx[vidx]
        pref = np.concatenate([[0.0], np.cumsum(vval)])
        # count of valid dx entries with index < i  (dx has length n-1)
        cnt_lt = np.searchsorted(vidx, np.arange(n))  # cnt_lt[i] = #valid dx[:i]
        for i in range(n):
            if i + 1 < 2 * per + 1:
                adx[i] = 0.0
                continue
            c = cnt_lt[i]
            if c < per:
                adx[i] = 0.0
                continue
            m = (pref[c] - pref[c - per]) / per
            adx[i] = m if np.isfinite(m) else 0.0

    return {"H": H, "L": L, "C": C, "n": n,
            "mid": mid, "sd": sd, "rsi": rsi, "atr": atr, "adx": adx}


def signal_at(pc, i, bb_m, rsi_lo, rsi_hi, adx_max):
    """Mirror mr.evaluate for CONFIRM=1, returning (dir, entry, sl, tp1) or None."""
    if i < max(BB_P, 30) + 2:
        return None
    mid = pc["mid"]; sd = pc["sd"]; C = pc["C"]
    m = mid[i]
    if not np.isfinite(m):
        return None
    upper = m + bb_m * sd[i]
    lower = m - bb_m * sd[i]
    if pc["atr"][i] <= 0:
        return None
    if pc["adx"][i] >= adx_max:
        return None
    r = pc["rsi"][i]
    prev = C[i - 1]
    m_prev = mid[i - 1]
    sd_prev = sd[i - 1]
    lower_prev = m_prev - bb_m * sd_prev
    upper_prev = m_prev + bb_m * sd_prev
    close = C[i]
    direction = None
    if prev < lower_prev and close > lower and r < rsi_lo:
        direction = "LONG"
    elif prev > upper_prev and close < upper and r > rsi_hi:
        direction = "SHORT"
    if direction is None:
        return None
    entry = close
    a = SL_ATR * pc["atr"][i]
    if direction == "LONG":
        sl = entry - a
        tp1 = m
        if sl >= entry or tp1 <= entry:
            return None
    else:
        sl = entry + a
        tp1 = m
        if sl <= entry or tp1 >= entry:
            return None
    return (direction, entry, sl, tp1)


def simulate_fast(pc, bb_m, rsi_lo, rsi_hi, adx_max, time_stop):
    H, L, C, n = pc["H"], pc["L"], pc["C"], pc["n"]
    trades = []
    open_until = -1
    i = 60
    while i < n - 2:
        if i <= open_until:
            i += 1
            continue
        sig = signal_at(pc, i, bb_m, rsi_lo, rsi_hi, adx_max)
        if sig is None:
            i += 1
            continue
        direction, entry, sl, tp1 = sig
        is_long = direction == "LONG"
        risk = abs(entry - sl)
        if risk <= 0:
            i += 1
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
        trades.append(gross / risk)
        open_until = exit_j
        i += 1
    return trades


def summarize(trades):
    if not trades:
        return {"trades": 0, "pf": None, "avg_R": None, "total_R": 0.0, "wr": None}
    R = np.array(trades)
    wins = R[R > 0]
    losses = R[R <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    return {"trades": len(R), "pf": round(float(pf), 4),
            "avg_R": round(float(R.mean()), 4),
            "total_R": round(float(R.sum()), 4),
            "wr": round(float((R > 0).mean()), 4)}


def validate(m15):
    """Brute-force parity check vs mr.evaluate on a 3000-bar slice."""
    sl = m15.iloc[:3000].reset_index(drop=True)
    pc = precompute(sl)
    ov = {"CONFIRM": 1, "SL_ATR": 1.5, "BB_MULT": 2.0, "RSI_LOW": 20.0,
          "RSI_HIGH": 80.0, "ADX_MAX": 25.0, "TIME_STOP_BARS": 16}
    mism = 0
    checked = 0
    for i in range(60, len(sl) - 2):
        ref = mr.evaluate(sl, i, ov)
        fast = signal_at(pc, i, 2.0, 20.0, 80.0, 25.0)
        if (ref is None) != (fast is None):
            mism += 1
            if mism <= 5:
                print(f"  MISMATCH presence @i={i}: ref={ref is not None} fast={fast is not None}",
                      flush=True)
            continue
        if ref is not None:
            checked += 1
            if (ref["direction"] != fast[0]
                    or abs(ref["entry"] - fast[1]) > 1e-9
                    or abs(ref["sl"] - fast[2]) > 1e-9
                    or abs(ref["tp1"] - fast[3]) > 1e-9):
                mism += 1
                if mism <= 5:
                    print(f"  MISMATCH values @i={i}: {ref} vs {fast}", flush=True)
    print(f"VALIDATION: {checked} signals compared, {mism} mismatches", flush=True)
    return mism == 0


def main():
    m15 = bt._load("BTCUSD", "m15")
    if m15 is None:
        print(json.dumps({"status": "NO_DATA"}))
        return
    keep = 365 * 24 * 4
    if keep < len(m15):
        m15 = m15.iloc[-keep:].reset_index(drop=True)
    print(f"bars: {len(m15)}", flush=True)

    if not validate(m15):
        print("PARITY FAILED — aborting", flush=True)
        sys.exit(2)

    n = len(m15)
    w = n // 4
    datasets = {
        "full": m15,
        "fold2": m15.iloc[2 * w:3 * w].reset_index(drop=True),
        "fold3": m15.iloc[3 * w:].reset_index(drop=True),
    }
    pcs = {k: precompute(v) for k, v in datasets.items()}
    print("precompute done", flush=True)

    results = []
    for bb in (2.0, 2.5, 3.0):
        for rlo, rhi in ((10.0, 90.0), (20.0, 80.0), (30.0, 70.0)):
            for adxm in (20.0, 25.0, 30.0):
                for ts in (16, 24):
                    row = {"BB_MULT": bb, "RSI_LOW": rlo, "RSI_HIGH": rhi,
                           "ADX_MAX": adxm, "TIME_STOP_BARS": ts}
                    for name, pc in pcs.items():
                        tr = simulate_fast(pc, bb, rlo, rhi, adxm, ts)
                        row[name] = summarize(tr)
                    results.append(row)
                    print(f"cfg bb={bb} rl={rlo} adx={adxm} ts={ts} "
                          f"full_pf={row['full']['pf']} full_n={row['full']['trades']} "
                          f"f3_avgR={row['fold3']['avg_R']}", flush=True)

    with open("/tmp/btcmr_sweep54_c1_sl15_vec.json", "w") as f:
        json.dump(results, f, indent=1)
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
