#!/usr/bin/env python3 -B
"""ETHUSD intraday (H1) validation of the TREND exit tune (2026-07-09).

The D1 tuner's optimum ran monotonically to the tightest grid edge (TR1.0/LK0.9)
— a D1 intra-bar fill artifact. ETH is the only trend symbol with deep H1 history
(50k bars), so re-run the exit sweep checking exits every H1 bar (~12x finer,
close to the live 60s cadence). If the optimum is INTERIOR here, tighter-is-better
was an artifact; if it still runs to the edge, tight really is better for ETH.

Daily signal (prior completed D1) drives entry/flip; exits (chandelier + profit-
lock + peak-giveback + 3xATR stop) checked per H1 bar. Realistic round-trip spread.
"""
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
EMA_PAIRS = [(16, 64), (32, 128), (64, 256)]
MIN_ABS = 0.34
ATR_P = 20
ATR_STOP = 3.0
GB, ACT, TP = 0.35, 0.3, 999.0        # fixed; sweep the two key levers
TRAILS = [1.0, 1.5, 2.0, 2.5, 3.0]
LOCKS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _naive(s):
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def d1_context():
    d = pickle.load(open(C / "raw_d1_ETHUSD.pkl", "rb"))
    d["time"] = _naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
    c, h, l = d["close"], d["high"], d["low"]
    sig = pd.Series(0.0, index=d.index)
    for f, s in EMA_PAIRS:
        sig = sig + np.sign(c.ewm(span=f).mean() - c.ewm(span=s).mean())
    sig = (sig / len(EMA_PAIRS)).apply(lambda v: 0 if abs(v) < MIN_ABS else (1 if v > 0 else -1))
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_P, adjust=False).mean()
    out = pd.DataFrame({"eff": d["time"].dt.normalize() + pd.Timedelta(days=1),
                        "sig": sig.astype(int), "atr": atr,
                        "hh": h.rolling(22).max(), "ll": l.rolling(22).min()})
    return out.dropna().reset_index(drop=True)


def build():
    h1 = pickle.load(open(C / "raw_h1_ETHUSD.pkl", "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    d1 = d1_context()
    h1["time"] = h1["time"].astype("datetime64[ns]")
    d1["eff"] = d1["eff"].astype("datetime64[ns]")
    m = pd.merge_asof(h1, d1, left_on="time", right_on="eff", direction="backward").dropna(
        subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)
    return m


def simulate(m, TR, LK, cost):
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
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
        if s != 0 and s != pos:                    # flip on daily signal reversal
            trades.append(((o[t] - entry) / entry) * pos - cost)
            pos = s; entry = o[t]; peak = 0.0; blocked = 0
            sl = entry - ATR_STOP * a if pos == 1 else entry + ATR_STOP * a
            tp = None if TP >= 999 else ((entry + TP * a) if pos == 1 else (entry - TP * a))
            continue
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else -1e18
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
            if peak >= ACT * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else 1e18
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


def met(r):
    if not r:
        return (0, 0.0, 0.0, 0.0, 0.0)
    r = np.array(r); w = r[r > 0].sum(); ls = -r[r < 0].sum()
    eq = np.cumsum(r); dd = float((np.maximum.accumulate(eq) - eq).max())
    return (len(r), float(r.sum()), float(w / ls) if ls > 0 else 99.0, float((r > 0).mean()), dd)


def main():
    m = build()
    px = float(m["close"].iloc[-1])
    cost = 2.0 * (float(np.nanmedian(m["spread"].values)) * 0.01) / px
    split = int(len(m) * 0.70)
    mis, moos = m.iloc[:split].reset_index(drop=True), m.iloc[split:].reset_index(drop=True)
    print("ETHUSD H1  bars=%d  cost_rt=%.4f  split=%s" % (len(m), cost, str(m["time"].iloc[split])[:10]))
    print("grid: rows=TRAIL, cols=LOCK  — cell = OOS_ret / OOS_pf / n  (IS_ret in parens)")
    print("        " + "".join("  LK%.1f        " % lk for lk in LOCKS))
    for TR in TRAILS:
        cells = []
        for LK in LOCKS:
            ni, ri, pi, wi, di = met(simulate(mis, TR, LK, cost))
            no, ro, po, wo, do = met(simulate(moos, TR, LK, cost))
            cells.append("%5.1f/%4.1f/%3d(%4.1f)" % (ro, po, no, ri))
        print("TR%.1f  " % TR + "  ".join(cells))


if __name__ == "__main__":
    main()
