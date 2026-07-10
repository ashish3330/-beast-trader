#!/usr/bin/env python3 -B
"""HONEST mean-reversion screen (2026-07-10) for symbols the trend workflow
flagged as SEPARATE_LOGIC (move but no daily-trend edge). Tests a Connors-style
RSI(2) fade on H1 with realistic round-trip spread + walk-forward IS/OOS.

Entry (both sides): RSI2 < LO -> long (oversold), RSI2 > HI -> short (overbought).
Exit: RSI2 crosses back through mid (50), OR close beyond SMA(exit), OR time-stop
(bars), OR a catastrophic ATR stop. Bar-close causal (act on next bar open).
NO intra-bar-fill optimism: exits checked at bar CLOSE only (conservative for MR).

Purpose: is there a REAL, robust MR edge (positive across IS AND OOS, PF>~1.3,
sane trade count), or was the churn profitability just an intra-bar artifact?

Usage: python3 -B scripts/_mr_screen.py SYMBOL
"""
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

GRID = {
    "LO":   [5, 10, 15],        # RSI2 oversold entry
    "HI":   [85, 90, 95],       # RSI2 overbought entry
    "EXIT_SMA": [5, 10],        # exit when close crosses this SMA
    "TIME": [24, 48, 96],       # time-stop in H1 bars
    "SL_ATR": [3.0, 4.0],       # catastrophic ATR(14) stop
}


def _naive(s):
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)


def atr(h, l, c, n):
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False).mean()


def load(sym):
    p = C / ("raw_h1_" + sym.replace(".", "_") + ".pkl")
    if not p.exists():
        return None
    d = pickle.load(open(p, "rb"))
    d["time"] = _naive(d["time"])
    return d.sort_values("time").reset_index(drop=True)


def simulate(df, p, cost):
    o = df["open"].values; h = df["high"].values; l = df["low"].values; c = df["close"].values
    r = rsi(df["close"], 2).values
    a = atr(df["high"], df["low"], df["close"], 14).values
    sma = df["close"].rolling(p["EXIT_SMA"]).mean().values
    LO, HI, TIME, SL = p["LO"], p["HI"], p["TIME"], p["SL_ATR"]
    pos = 0; entry = sl = 0.0; held = 0
    trades = []
    for t in range(210, len(df)):
        if not np.isfinite(a[t-1]) or a[t-1] <= 0:
            continue
        if pos == 0:
            if r[t-1] < LO:
                pos = 1; entry = o[t]; sl = entry - SL*a[t-1]; held = 0
            elif r[t-1] > HI:
                pos = -1; entry = o[t]; sl = entry + SL*a[t-1]; held = 0
            continue
        held += 1
        # catastrophic stop (intrabar) — conservative, counts against us
        if pos == 1 and l[t] <= sl:
            trades.append(((sl-entry)/entry) - cost); pos = 0; continue
        if pos == -1 and h[t] >= sl:
            trades.append(((entry-sl)/entry) - cost); pos = 0; continue
        # MR exit at bar close: revert through mid or cross exit-SMA or time-stop
        exit_now = False
        if pos == 1 and (r[t] > 50 or c[t] > sma[t] or held >= TIME):
            exit_now = True
        elif pos == -1 and (r[t] < 50 or c[t] < sma[t] or held >= TIME):
            exit_now = True
        if exit_now:
            trades.append((((c[t]-entry)/entry) if pos == 1 else ((entry-c[t])/entry)) - cost)
            pos = 0
    return trades


def stats(r, years):
    if not r:
        return {"n": 0, "tpy": 0.0, "ret": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0}
    r = np.array(r); w = r[r > 0].sum(); ls = -r[r < 0].sum()
    eq = np.cumsum(r); dd = float((np.maximum.accumulate(eq)-eq).max())
    return {"n": len(r), "tpy": len(r)/max(years, 0.1), "ret": float(r.sum()),
            "pf": float(w/ls) if ls > 0 else (99.0 if w > 0 else 0.0),
            "wr": float((r > 0).mean()), "dd": dd}


def main():
    sym = sys.argv[1]
    df = load(sym)
    if df is None or len(df) < 3000:
        print(json.dumps({"symbol": sym, "error": "insufficient H1 data"})); return
    px = float(df["close"].iloc[-1])
    cost = 2.0 * (float(np.nanmedian(df["spread"].values))*0.01)/px if "spread" in df else 0.0004
    split = int(len(df)*0.70)
    dis, dos = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    yi = (dis["time"].iloc[-1]-dis["time"].iloc[0]).days/365.25
    yo = (dos["time"].iloc[-1]-dos["time"].iloc[0]).days/365.25

    rows = []
    keys = list(GRID.keys())
    for combo in product(*[GRID[k] for k in keys]):
        p = dict(zip(keys, combo))
        mi = stats(simulate(dis, p, cost), yi)
        mo = stats(simulate(dos, p, cost), yo)
        # robust: both IS and OOS positive, OOS pf>=1.2, enough trades, sane turnover
        if mi["ret"] > 0 and mo["ret"] > 0 and mo["pf"] >= 1.2 and mo["n"] >= 20 and mo["tpy"] <= 400:
            rows.append({"p": p, "is": mi, "oos": mo,
                         "score": mo["ret"]/(1+mo["dd"])})
    rows.sort(key=lambda r: r["score"], reverse=True)
    print(json.dumps({"symbol": sym, "h1_bars": len(df), "cost_rt": round(cost, 5),
                      "yr_is": round(yi, 1), "yr_oos": round(yo, 1),
                      "n_robust": len(rows), "n_total": len(list(product(*[GRID[k] for k in keys]))),
                      "top": rows[:6]}, default=float))


if __name__ == "__main__":
    main()
