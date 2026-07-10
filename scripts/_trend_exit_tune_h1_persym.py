#!/usr/bin/env python3 -B
"""PER-SYMBOL intraday (H1) hard-tune of the TREND exit model (2026-07-09).

Now that all 5 trend symbols have deep H1 history, tune each one on INTRADAY bars
(exits checked hourly ~ close to the live 60s cadence) instead of the artifact-
prone D1. Daily signal (prior completed D1) drives entry/flip; the exit stack
(chandelier + profit-lock SL + peak-giveback + 3xATR stop + TP) is simulated per
H1 bar with realistic round-trip spread. Walk-forward IS(70%)/OOS(30%).

CHURN-CLIFF GUARD: a daily trend book should trade ~10-40x/yr. Configs whose
turnover explodes (the profit-lock tapped intraday + re-entry) are intra-bar-fill
ARTIFACTS, not edge — so the winner is picked ONLY from the realistic-turnover
region (<= MAX_TPY trades/yr). Both the realistic winner and the raw (unfiltered)
top are reported so the churn cliff stays visible.

Usage: python3 -B scripts/_trend_exit_tune_h1_persym.py SYMBOL
"""
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
EMA_PAIRS = [(16, 64), (32, 128), (64, 256)]
MIN_ABS = 0.34
ATR_P = 20
ATR_STOP = 3.0
MAX_TPY = 40.0            # realistic-turnover cap (trades/year); above = churn artifact

GRID = {
    "GIVEBACK": [0.25, 0.35, 0.50, 1.01],
    "LOCK":     [0.5, 0.6, 0.7],
    "ACT":      [0.3, 0.5],
    "TRAIL":    [1.5, 2.0, 2.5, 3.0],
    "TP":       [6.0, 999.0],
}


def _naive(s):
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def d1_context(sym):
    d = pickle.load(open(C / ("raw_d1_" + sym.replace(".", "_") + ".pkl"), "rb"))
    d["time"] = _naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
    c, h, l = d["close"], d["high"], d["low"]
    sig = pd.Series(0.0, index=d.index)
    for f, s in EMA_PAIRS:
        sig = sig + np.sign(c.ewm(span=f).mean() - c.ewm(span=s).mean())
    sig = (sig / len(EMA_PAIRS)).apply(lambda v: 0 if abs(v) < MIN_ABS else (1 if v > 0 else -1))
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_P, adjust=False).mean()
    return pd.DataFrame({"eff": (d["time"].dt.normalize() + pd.Timedelta(days=1)).astype("datetime64[ns]"),
                         "sig": sig.astype(int), "atr": atr,
                         "hh": h.rolling(22).max(), "ll": l.rolling(22).min()}).dropna().reset_index(drop=True)


def build(sym):
    h1 = pickle.load(open(C / ("raw_h1_" + sym.replace(".", "_") + ".pkl"), "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    h1["time"] = h1["time"].astype("datetime64[ns]")
    d1 = d1_context(sym)
    m = pd.merge_asof(h1, d1, left_on="time", right_on="eff", direction="backward")
    return m.dropna(subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)


def simulate(m, p, cost):
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    GB, LK, AC, TR, TP = p["GIVEBACK"], p["LOCK"], p["ACT"], p["TRAIL"], p["TP"]
    pos = 0; entry = sl = tp = peak = 0.0; blocked = 0
    trades = []
    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blocked and s != blocked:
                blocked = 0
            if s != 0 and s != blocked:
                pos = s; entry = o[t]; peak = 0.0
                sl = entry - ATR_STOP * a if s == 1 else entry + ATR_STOP * a
                tp = None if TP >= 999 else ((entry + TP * a) if s == 1 else (entry - TP * a))
            continue
        if s != 0 and s != pos:
            trades.append(((o[t] - entry) / entry) * pos - cost)
            pos = s; entry = o[t]; peak = 0.0; blocked = 0
            sl = entry - ATR_STOP * a if pos == 1 else entry + ATR_STOP * a
            tp = None if TP >= 999 else ((entry + TP * a) if pos == 1 else (entry - TP * a))
            continue
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if peak >= AC * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= AC * a) else -1e18
            ex = None
            if l[t] <= sl:
                ex = sl
            elif gb > -1e17 and l[t] <= gb:
                ex = gb; blocked = pos
            elif tp is not None and h[t] >= tp:
                ex = tp
            if ex is not None:
                trades.append(((ex - entry) / entry) * pos - cost); pos = 0
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR * a)
            if peak >= AC * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= AC * a) else 1e18
            ex = None
            if h[t] >= sl:
                ex = sl
            elif gb < 1e17 and h[t] >= gb:
                ex = gb; blocked = pos
            elif tp is not None and l[t] <= tp:
                ex = tp
            if ex is not None:
                trades.append(((ex - entry) / entry) * pos - cost); pos = 0
            else:
                peak = max(peak, entry - l[t])
    return trades


def met(r, years):
    if not r:
        return {"n": 0, "tpy": 0.0, "ret": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0}
    r = np.array(r); w = r[r > 0].sum(); ls = -r[r < 0].sum()
    eq = np.cumsum(r); dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"n": len(r), "tpy": len(r) / max(years, 0.1), "ret": float(r.sum()),
            "pf": float(w / ls) if ls > 0 else (99.0 if w > 0 else 0.0),
            "wr": float((r > 0).mean()), "dd": dd}


def main():
    sym = sys.argv[1]
    m = build(sym)
    px = float(m["close"].iloc[-1])
    cost = 2.0 * (float(np.nanmedian(m["spread"].values)) * 0.01) / px if px > 0 else 0.0004
    split = int(len(m) * 0.70)
    mis, moos = m.iloc[:split].reset_index(drop=True), m.iloc[split:].reset_index(drop=True)
    yr_is = (mis["time"].iloc[-1] - mis["time"].iloc[0]).days / 365.25
    yr_oos = (moos["time"].iloc[-1] - moos["time"].iloc[0]).days / 365.25

    rows = []
    keys = list(GRID.keys())
    for combo in product(*[GRID[k] for k in keys]):
        p = dict(zip(keys, combo))
        mi = met(simulate(mis, p, cost), yr_is)
        mo = met(simulate(moos, p, cost), yr_oos)
        rows.append({"p": p, "is": mi, "oos": mo})

    base_p = {"GIVEBACK": 0.30, "LOCK": 0.6, "ACT": 0.3, "TRAIL": 2.0, "TP": 6.0}
    base = {"p": base_p, "is": met(simulate(mis, base_p, cost), yr_is),
            "oos": met(simulate(moos, base_p, cost), yr_oos)}

    # realistic-turnover region only (both IS & OOS below the churn cap), robust,
    # ranked by OOS return / (1+DD) then OOS PF.
    def ok(r):
        return (r["is"]["tpy"] <= MAX_TPY and r["oos"]["tpy"] <= MAX_TPY
                and r["oos"]["ret"] > 0 and r["is"]["ret"] > 0 and r["oos"]["n"] >= 5)
    real = [r for r in rows if ok(r)]
    real.sort(key=lambda r: (r["oos"]["ret"] / (1 + r["oos"]["dd"]), r["oos"]["pf"]), reverse=True)
    raw = sorted(rows, key=lambda r: r["oos"]["ret"], reverse=True)

    print(json.dumps({"symbol": sym, "h1_bars": len(m), "cost_rt": round(cost, 5),
                      "yr_is": round(yr_is, 1), "yr_oos": round(yr_oos, 1),
                      "split": str(m["time"].iloc[split])[:10], "max_tpy": MAX_TPY,
                      "baseline": base, "realistic_top": real[:6],
                      "raw_top_churn": raw[:3], "n_realistic": len(real)}, default=float))


if __name__ == "__main__":
    main()
