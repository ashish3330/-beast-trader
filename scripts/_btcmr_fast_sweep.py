#!/usr/bin/env python3 -B
"""Fast exact replica of scripts/_btcmr_run.py for the 54-config sweep.
Computes per-bar indicator arrays ONCE per data slice (causal recursions
identical to agent.sma_breakout helpers + agent.btc_mean_reversion), then
applies each config as threshold masks + trade loop.

Replicates exactly:
  - evaluate(): warm-up gate i >= max(BB_PERIOD,30)+2, SMA/std(ddof=0) bands,
    Wilder ATR/ADX/RSI seeded at slice start, CONFIRM=0 path, tp1/sl sanity.
  - simulate(): start=60, SL-first fill, time-stop close exit, SPREAD=5 total,
    open_until re-entry block, i range to n-3.
  - run(): --days tail-keep, fold slicing (last fold takes remainder).
Validated against _btcmr_run.py output before use.
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
PERIOD = 14


def indicator_arrays(H, L, C):
    """Per-bar causal indicator values identical to slice-based helpers."""
    n = len(C)
    # --- SMA20 + rolling std (ddof=0) of last 20 closes ---
    cs = np.cumsum(np.insert(C, 0, 0.0))
    mid = np.full(n, np.nan)
    mid[BB_P - 1:] = (cs[BB_P:] - cs[:-BB_P]) / BB_P
    cs2 = np.cumsum(np.insert(C * C, 0, 0.0))
    var = np.full(n, np.nan)
    var[BB_P - 1:] = (cs2[BB_P:] - cs2[:-BB_P]) / BB_P - mid[BB_P - 1:] ** 2
    sd = np.sqrt(np.maximum(var, 0.0))
    # exact per-bar std to match np.std of the window (fp-safe recompute)
    # cumsum float error is ~1e-9 on BTC prices^2; refine windows directly:
    sd_exact = np.full(n, np.nan)
    for i in range(BB_P - 1, n):
        sd_exact[i] = np.std(C[i - BB_P + 1:i + 1], ddof=0)
    sd = sd_exact
    mid_exact = np.full(n, np.nan)
    for i in range(BB_P - 1, n):
        mid_exact[i] = C[i - BB_P + 1:i + 1].mean()
    mid = mid_exact

    # --- Wilder ATR14 (seed = mean tr[1:15], recursion) ---
    tr = np.empty(n)
    tr[0] = H[0] - L[0]
    tr[1:] = np.maximum.reduce([H[1:] - L[1:], np.abs(H[1:] - C[:-1]),
                                np.abs(L[1:] - C[:-1])])
    atr = np.full(n, np.nan)
    if n > PERIOD:
        atr[PERIOD] = tr[1:PERIOD + 1].mean()
        k = 1.0 / PERIOD
        for i in range(PERIOD + 1, n):
            atr[i] = atr[i - 1] * (1 - k) + tr[i] * k

    # --- Wilder ADX14: per-bar = mean of last 14 non-NaN dx with dx-idx <= i-1 ---
    up = H[1:] - H[:-1]
    dn = L[:-1] - L[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr1 = tr[1:]
    k = 1.0 / PERIOD

    def _wilder(x):
        out = np.empty(len(x))
        out[:PERIOD] = np.nan
        out[PERIOD - 1] = x[:PERIOD].sum()
        for i in range(PERIOD, len(x)):
            out[i] = out[i - 1] - out[i - 1] * k + x[i]
        return out

    atr_s = _wilder(tr1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * _wilder(plus_dm) / atr_s
        mdi = 100.0 * _wilder(minus_dm) / atr_s
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    finite_pos = np.where(~np.isnan(dx))[0]          # keeps +inf like original
    fvals = dx[finite_pos]
    fpre = np.concatenate([[0.0], np.cumsum(fvals)])
    adx = np.zeros(n)
    # counts of non-nan dx entries with position <= i-1
    counts = np.searchsorted(finite_pos, np.arange(n) - 1, side="right")
    for i in range(n):
        if i < 2 * PERIOD + 1:
            adx[i] = 0.0
            continue
        c = counts[i]
        if c < PERIOD:
            adx[i] = 0.0
            continue
        m = (fpre[c] - fpre[c - PERIOD]) / PERIOD
        adx[i] = m if np.isfinite(m) else 0.0

    # --- Wilder RSI2 per-bar (seed = first 2 diffs, recursion) ---
    rsi = np.full(n, 50.0)
    d = np.diff(C)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    if n >= RSI_P + 1:
        ag = gain[:RSI_P].mean()
        al = loss[:RSI_P].mean()
        kr = 1.0 / RSI_P
        # value at bar index RSI_P (i.e. after seed, before recursion steps)
        def _val(ag, al):
            if al == 0:
                return 100.0
            return 100.0 - 100.0 / (1.0 + ag / al)
        rsi[RSI_P] = _val(ag, al)
        for j in range(RSI_P, len(d)):
            ag = ag * (1 - kr) + gain[j] * kr
            al = al * (1 - kr) + loss[j] * kr
            rsi[j + 1] = _val(ag, al)
    return mid, sd, atr, adx, rsi


def simulate_fast(H, L, C, mid, sd, atr, adx, cfg_rsi, cfg):
    n = len(C)
    bb_m = cfg["BB_MULT"]
    rsi_lo = cfg["RSI_LOW"]
    rsi_hi = cfg["RSI_HIGH"]
    adx_max = cfg["ADX_MAX"]
    sl_atr = cfg["SL_ATR"]
    time_stop = int(cfg["TIME_STOP_BARS"])
    warm = max(BB_P, 30) + 2

    upper = mid + bb_m * sd
    lower = mid - bb_m * sd
    ok = np.arange(n) >= max(60, warm)
    ok &= np.isfinite(mid) & (atr > 0) & ~np.isnan(atr) & (adx < adx_max)
    long_m = ok & (C < lower) & (cfg_rsi < rsi_lo)
    short_m = ok & ~long_m & (C > upper) & (cfg_rsi > rsi_hi)
    cand = np.where(long_m | short_m)[0]

    trades_R = []
    open_until = -1
    for i in cand:
        if i <= open_until or i > n - 3:
            continue
        is_long = bool(long_m[i])
        entry = C[i]
        if is_long:
            sl = entry - sl_atr * atr[i]
            tp1 = mid[i]
            if sl >= entry or tp1 <= entry:
                continue
        else:
            sl = entry + sl_atr * atr[i]
            tp1 = mid[i]
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
        open_until = exit_j
    return trades_R


def summarize(R):
    if not R:
        return {"trades": 0}
    R = np.array(R)
    wins = R[R > 0]
    losses = R[R <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    return {"trades": len(R), "wr": round(float((R > 0).mean()), 4),
            "pf": round(float(pf), 4), "avg_R": round(float(R.mean()), 4),
            "total_R": round(float(R.sum()), 4)}


def main():
    m15_all = bt._load("BTCUSD", "m15")
    keep = int(365 * 24 * 4)
    if keep < len(m15_all):
        m15_all = m15_all.iloc[-keep:].reset_index(drop=True)

    slices = {"full": m15_all}
    w = len(m15_all) // 4
    slices["fold3"] = m15_all.iloc[3 * w:].reset_index(drop=True)
    slices["fold2"] = m15_all.iloc[2 * w:3 * w].reset_index(drop=True)

    ind = {}
    for name, df in slices.items():
        H = df["high"].values.astype(float)
        L = df["low"].values.astype(float)
        C = df["close"].values.astype(float)
        ind[name] = (H, L, C) + indicator_arrays(H, L, C)
        print(f"indicators ready: {name} n={len(df)}", file=sys.stderr, flush=True)

    configs = []
    for bb in (2.0, 2.5, 3.0):
        for rlo, rhi in ((10.0, 90.0), (20.0, 80.0), (30.0, 70.0)):
            for adxm in (20.0, 25.0, 30.0):
                for ts in (16, 24):
                    configs.append({"CONFIRM": 0, "SL_ATR": 1.0, "BB_MULT": bb,
                                    "RSI_LOW": rlo, "RSI_HIGH": rhi,
                                    "ADX_MAX": adxm, "TIME_STOP_BARS": ts})

    out = []
    for cfg in configs:
        row = {"cfg": cfg}
        for name in ("full", "fold3", "fold2"):
            H, L, C, mid, sd, atr, adx, rsi = ind[name]
            R = simulate_fast(H, L, C, mid, sd, atr, adx, rsi, cfg)
            row[name] = summarize(R)
        out.append(row)
        print(json.dumps(row), flush=True)
    with open("/tmp/btcmr_fast_sweep.json", "w") as f:
        json.dump(out, f, indent=1)
    print("FAST SWEEP COMPLETE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
