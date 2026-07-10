#!/usr/bin/env python3 -B
"""FOREX STRATEGY RESEARCH (2026-07-10) — trend (EMA-cross) and simple RSI-MR both
FAILED on FX. Test the industry-standard FX approaches that are genuinely DIFFERENT,
honestly, on H1 with realistic round-trip spread + walk-forward IS(70%)/OOS(30%):

  1) DONCHIAN volatility breakout (Turtle) — enter on N-bar high/low breakout,
     exit on M-bar opposite channel OR chandelier ATR trail OR catastrophic stop.
  2) LONDON session breakout — Asian-range (00:00-07:00 server) breakout traded in
     the London window, flat by NY close; ATR stop.
  3) BOLLINGER mean-reversion + ADX regime filter — fade BB(20,k) extremes ONLY
     when ADX<thr (ranging); exit at midband / time-stop.

Bar-CLOSE causal, no intra-bar-fill optimism. Reports per (pair, strategy) the
best WF-robust config (IS&OOS positive, OOS PF>=1.2, sane turnover). This finds
whether ANY industry FX strategy has a real edge before we build/deploy it.

Usage: python3 -B scripts/_fx_research.py SYMBOL [strategy]   (strategy: donchian|london|bbands|all)
"""
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")


def _naive(s):
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def load(sym):
    p = C / ("raw_h1_" + sym.replace(".", "_") + ".pkl")
    if not p.exists():
        return None
    d = pickle.load(open(p, "rb"))
    d["time"] = _naive(d["time"])
    return d.sort_values("time").reset_index(drop=True)


def atr(h, l, c, n):
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def adx(h, l, c, n):
    up = h.diff(); dn = -l.diff()
    plus = ((up > dn) & (up > 0)) * up
    minus = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atrn = tr.ewm(alpha=1.0/n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1.0/n, adjust=False).mean() / atrn.replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1.0/n, adjust=False).mean() / atrn.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1.0/n, adjust=False).mean().fillna(0)


# ── STRATEGY SIMS (return list of trade returns, fraction) ──
def sim_donchian(df, p, cost):
    o, h, l, c = [df[x].values for x in ("open", "high", "low", "close")]
    a = atr(df["high"], df["low"], df["close"], 14).values
    N, M, SL, TR = p["N"], p["M"], p["SL"], p["TR"]
    hh = df["high"].rolling(N).max().shift(1).values
    ll = df["low"].rolling(N).min().shift(1).values
    xh = df["high"].rolling(M).max().shift(1).values
    xl = df["low"].rolling(M).min().shift(1).values
    pos = 0; entry = sl = peak = 0.0; trades = []
    for t in range(N + 2, len(df)):
        if not np.isfinite(a[t-1]) or a[t-1] <= 0:
            continue
        if pos == 0:
            if c[t-1] > hh[t-1]:
                pos = 1; entry = o[t]; sl = entry - SL*a[t-1]; peak = 0.0
            elif c[t-1] < ll[t-1]:
                pos = -1; entry = o[t]; sl = entry + SL*a[t-1]; peak = 0.0
            continue
        if pos == 1:
            if l[t] <= sl: trades.append((sl-entry)/entry - cost); pos = 0; continue
            peak = max(peak, h[t]-entry)
            sl = max(sl, (df["high"].iloc[max(0,t-N):t].max()) - TR*a[t-1])
            if c[t] < xl[t]: trades.append((c[t]-entry)/entry - cost); pos = 0
        else:
            if h[t] >= sl: trades.append((entry-sl)/entry - cost); pos = 0; continue
            peak = max(peak, entry-l[t])
            sl = min(sl, (df["low"].iloc[max(0,t-N):t].min()) + TR*a[t-1])
            if c[t] > xh[t]: trades.append((entry-c[t])/entry - cost); pos = 0
    return trades


def sim_london(df, p, cost):
    t = df["time"]; hr = t.dt.hour.values
    o, h, l, c = [df[x].values for x in ("open", "high", "low", "close")]
    a = atr(df["high"], df["low"], df["close"], 14).values
    day = t.dt.date.values
    BUF, SL, TP = p["BUF"], p["SL"], p["TP"]
    pos = 0; entry = sl = tp = 0.0; trades = []
    asia_hi = asia_lo = None; cur_day = None
    for i in range(20, len(df)):
        d = day[i]
        if d != cur_day:                      # new day → reset Asian range
            cur_day = d; asia_hi = asia_lo = None
        if 0 <= hr[i] < 7:                     # Asian session: build range
            asia_hi = h[i] if asia_hi is None else max(asia_hi, h[i])
            asia_lo = l[i] if asia_lo is None else min(asia_lo, l[i])
            continue
        if pos == 0:
            if 7 <= hr[i] < 12 and asia_hi and a[i-1] > 0:   # London breakout window
                rng = asia_hi - asia_lo
                if c[i] > asia_hi + BUF*rng:
                    pos = 1; entry = c[i]; sl = entry - SL*a[i-1]; tp = entry + TP*a[i-1]
                elif c[i] < asia_lo - BUF*rng:
                    pos = -1; entry = c[i]; sl = entry + SL*a[i-1]; tp = entry - TP*a[i-1]
            continue
        # manage: flat by NY close (hr>=21) or SL/TP
        if pos == 1:
            if l[i] <= sl: trades.append((sl-entry)/entry - cost); pos = 0
            elif h[i] >= tp: trades.append((tp-entry)/entry - cost); pos = 0
            elif hr[i] >= 21: trades.append((c[i]-entry)/entry - cost); pos = 0
        else:
            if h[i] >= sl: trades.append((entry-sl)/entry - cost); pos = 0
            elif l[i] <= tp: trades.append((entry-tp)/entry - cost); pos = 0
            elif hr[i] >= 21: trades.append((entry-c[i])/entry - cost); pos = 0
    return trades


def sim_bbands(df, p, cost):
    o, h, l, c = [df[x].values for x in ("open", "high", "low", "close")]
    a = atr(df["high"], df["low"], df["close"], 14).values
    N, K, ADXT, TIME, SL = p["N"], p["K"], p["ADXT"], p["TIME"], p["SL"]
    ma = df["close"].rolling(N).mean(); sd = df["close"].rolling(N).std()
    up = (ma + K*sd).values; dn = (ma - K*sd).values; mid = ma.values
    ax = adx(df["high"], df["low"], df["close"], 14).values
    pos = 0; entry = sl = 0.0; held = 0; trades = []
    for t in range(N+2, len(df)):
        if not np.isfinite(a[t-1]) or a[t-1] <= 0:
            continue
        if pos == 0:
            if ax[t-1] < ADXT:                 # only fade in ranging regime
                if c[t-1] < dn[t-1]:
                    pos = 1; entry = o[t]; sl = entry - SL*a[t-1]; held = 0
                elif c[t-1] > up[t-1]:
                    pos = -1; entry = o[t]; sl = entry + SL*a[t-1]; held = 0
            continue
        held += 1
        if pos == 1:
            if l[t] <= sl: trades.append((sl-entry)/entry - cost); pos = 0; continue
            if c[t] >= mid[t] or held >= TIME: trades.append((c[t]-entry)/entry - cost); pos = 0
        else:
            if h[t] >= sl: trades.append((entry-sl)/entry - cost); pos = 0; continue
            if c[t] <= mid[t] or held >= TIME: trades.append((entry-c[t])/entry - cost); pos = 0
    return trades


STRATS = {
    "donchian": (sim_donchian, {"N": [20, 40, 55], "M": [10, 20], "SL": [2.0, 3.0], "TR": [2.5, 3.5]}),
    "london":   (sim_london, {"BUF": [0.0, 0.1, 0.25], "SL": [1.0, 1.5, 2.0], "TP": [1.5, 2.0, 3.0]}),
    "bbands":   (sim_bbands, {"N": [20], "K": [2.0, 2.5], "ADXT": [20, 25], "TIME": [12, 24], "SL": [2.0, 3.0]}),
}


def stats(r, years):
    if not r:
        return {"n": 0, "tpy": 0.0, "ret": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0}
    r = np.array(r); w = r[r > 0].sum(); ls = -r[r < 0].sum()
    eq = np.cumsum(r); dd = float((np.maximum.accumulate(eq)-eq).max())
    return {"n": len(r), "tpy": len(r)/max(years, 0.1), "ret": float(r.sum()),
            "pf": float(w/ls) if ls > 0 else (99.0 if w > 0 else 0.0),
            "wr": float((r > 0).mean()), "dd": dd}


def run_strat(df, name, cost, yi, yo, split):
    fn, grid = STRATS[name]
    dis, dos = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    keys = list(grid.keys()); rows = []
    for combo in product(*[grid[k] for k in keys]):
        p = dict(zip(keys, combo))
        mi = stats(fn(dis, p, cost), yi); mo = stats(fn(dos, p, cost), yo)
        if mi["ret"] > 0 and mo["ret"] > 0 and mo["pf"] >= 1.2 and mo["n"] >= 20 and mo["tpy"] <= 600:
            rows.append({"strategy": name, "p": p, "is": mi, "oos": mo, "score": mo["ret"]/(1+mo["dd"])})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def main():
    sym = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    df = load(sym)
    if df is None or len(df) < 5000:
        print(json.dumps({"symbol": sym, "error": "insufficient H1"})); return
    px = float(df["close"].iloc[-1])
    # FX cost: spread field is in POINTS; point size depends on digits (JPY pairs
    # ~3-digit -> 0.001, other FX ~5-digit -> 0.00001). Floor/cap to a realistic
    # major-pair band so garbage spread data can't break the sim.
    is_jpy = "JPY" in sym or px > 20
    point = 0.001 if is_jpy else 0.00001
    med = float(np.nanmedian(df["spread"].values)) if "spread" in df else 15.0
    if not np.isfinite(med) or med <= 0:
        med = 15.0
    cost = 2.0 * med * point / px
    cost = min(max(cost, 0.00008), 0.0004)      # realistic FX round-trip band
    split = int(len(df)*0.70)
    yi = (df["time"].iloc[split-1]-df["time"].iloc[0]).days/365.25
    yo = (df["time"].iloc[-1]-df["time"].iloc[split]).days/365.25
    names = list(STRATS) if which == "all" else [which]
    out = {}
    for n in names:
        rows = run_strat(df, n, cost, yi, yo, split)
        out[n] = {"n_robust": len(rows), "top": rows[:3]}
    print(json.dumps({"symbol": sym, "h1_bars": len(df), "cost_rt": round(cost, 5),
                      "yr_is": round(yi, 1), "yr_oos": round(yo, 1), "strategies": out}, default=float))


if __name__ == "__main__":
    main()
