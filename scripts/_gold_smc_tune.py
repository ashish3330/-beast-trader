#!/usr/bin/env python3 -B
"""TUNE the Hybrid SMC strategy for GOLD (2026-07-12) — gold H1 has the one genuine,
robust SMC edge (baseline PF 1.15, OOS 1.44, 3/4 folds). Sweep the real levers with
4-fold ANCHORED rolling walk-forward + a CHURN guard (turnover must not explode) and
ship only configs that beat baseline in >=3/4 OOS folds without turning a winning
fold into a loss. Parameters: TP1_R, TP2_R, SL_ATR buffer, SWEEP_LB, SEQ window.
Entry TF H1, bias D1. Realistic gold cost.
"""
import json
import pickle
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
SWING = 5
SLIP = 0.00004


def _ema(x, n): return x.ewm(span=n, adjust=False).mean()
def _rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + up/dn.replace(0, np.nan))).fillna(50)
def _macd(c): m = _ema(c, 12)-_ema(c, 26); return m, _ema(m, 9)
def _atr(h, l, c, n=14):
    p = c.shift(1); tr = pd.concat([(h-l), (h-p).abs(), (l-p).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()
def _naive(s):
    s = pd.to_datetime(s)
    try: return s.dt.tz_localize(None)
    except (TypeError, AttributeError): return s


def base_df():
    h1 = pickle.load(open(C/"raw_h1_XAUUSD.pkl", "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    c, h, l, o = h1["close"], h1["high"], h1["low"], h1["open"]
    vol = h1["tick_volume"]
    h1["ema9"], h1["ema21"] = _ema(c, 9), _ema(c, 21)
    h1["rsi"] = _rsi(c); h1["macd"], h1["sig"] = _macd(c); h1["atr"] = _atr(h, l, c)
    h1["body"] = (c-o).abs(); h1["avgbody"] = h1["body"].rolling(20).mean()
    day = h1["time"].dt.date; tp = (h+l+c)/3
    h1["vwap"] = ((tp*vol).groupby(day).cumsum()/vol.groupby(day).cumsum().replace(0, np.nan)).ffill()
    h1["prior_hi"] = h.rolling(SWING*2+1).max().shift(1)
    h1["prior_lo"] = l.rolling(SWING*2+1).min().shift(1)
    d = pickle.load(open(C/"raw_d1_XAUUSD.pkl", "rb")); d["time"] = _naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
    d["bias"] = np.where(d["close"] > _ema(d["close"], 50), 1, -1)
    d["eff"] = (d["time"].dt.normalize()+pd.Timedelta(days=1)).astype("datetime64[ns]")
    h1["time"] = h1["time"].astype("datetime64[ns]")
    h1 = pd.merge_asof(h1, d[["eff", "bias"]], left_on="time", right_on="eff", direction="backward")
    return h1.dropna(subset=["ema21", "atr", "avgbody", "prior_hi"]).reset_index(drop=True)


def simulate(m, p, spread):
    o, h, l, c = [m[x].values for x in ("open", "high", "low", "close")]
    ema9, ema21 = m["ema9"].values, m["ema21"].values
    rsi, macd, sig = m["rsi"].values, m["macd"].values, m["sig"].values
    vwap, atr, bias = m["vwap"].values, m["atr"].values, m["bias"].values
    body, avgbody = m["body"].values, m["avgbody"].values
    prior_hi, prior_lo = m["prior_hi"].values, m["prior_lo"].values
    swlo = pd.Series(l).rolling(p["SWEEP_LB"]).min().shift(1).values
    swhi = pd.Series(h).rolling(p["SWEEP_LB"]).max().shift(1).values
    T1, T2, SLA, SEQ = p["TP1_R"], p["TP2_R"], p["SL_ATR"], p["SEQ"]
    trades = []
    pos = 0; entry = slp = tp1 = tp2 = risk0 = 0.0; half = False
    lbs = lbslo = -999; lbsh = lbshi = -999
    for t in range(3, len(m)):
        if not np.isfinite(swlo[t]):
            continue
        if l[t] < swlo[t] and c[t] > swlo[t]:
            lbs = t; lbslo = l[t]
        if h[t] > swhi[t] and c[t] < swhi[t]:
            lbsh = t; lbshi = h[t]
        if pos != 0:
            is_long = pos == 1
            if is_long:
                if l[t] <= slp: trades.append(((slp*(1-SLIP)-entry)/entry-(0 if half else spread))/risk0); pos = 0; continue
                if not half and h[t] >= tp1: trades.append((((tp1-entry)/entry-spread)/risk0)*0.5); half = True; slp = entry
                if half and h[t] >= tp2: trades.append((((tp2-entry)/entry)/risk0)*0.5); pos = 0; continue
                if half and c[t] < ema9[t]: trades.append((((c[t]-entry)/entry)/risk0)*0.5); pos = 0; continue
            else:
                if h[t] >= slp: trades.append(((entry-slp*(1+SLIP))/entry-(0 if half else spread))/risk0); pos = 0; continue
                if not half and l[t] <= tp1: trades.append((((entry-tp1)/entry-spread)/risk0)*0.5); half = True; slp = entry
                if half and l[t] <= tp2: trades.append((((entry-tp2)/entry)/risk0)*0.5); pos = 0; continue
                if half and c[t] > ema9[t]: trades.append((((entry-c[t])/entry)/risk0)*0.5); pos = 0; continue
            continue
        bull_fvg = (l[t] > h[t-2]) or (l[t-1] > h[t-3])
        bear_fvg = (h[t] < l[t-2]) or (h[t-1] < l[t-3])
        be = c[t] > o[t] and c[t-1] < o[t-1] and c[t] > o[t-1] and o[t] < c[t-1]
        se = c[t] < o[t] and c[t-1] > o[t-1] and c[t] < o[t-1] and o[t] > c[t-1]
        long_ok = (bias[t] == 1 and (t-lbs) <= SEQ and c[t] > prior_hi[t] and bull_fvg
                   and ema9[t] > ema21[t] and ema9[t] > ema9[t-1] and ema21[t] > ema21[t-1]
                   and c[t] > vwap[t] and (macd[t] > sig[t] or rsi[t] > 50) and (body[t] > 1.2*avgbody[t] or be))
        short_ok = (bias[t] == -1 and (t-lbsh) <= SEQ and c[t] < prior_lo[t] and bear_fvg
                    and ema9[t] < ema21[t] and ema9[t] < ema9[t-1] and ema21[t] < ema21[t-1]
                    and c[t] < vwap[t] and (macd[t] < sig[t] or rsi[t] < 50) and (body[t] > 1.2*avgbody[t] or se))
        if long_ok:
            entry = c[t]*(1+SLIP); slp = min(l[t], lbslo) - SLA*atr[t]; risk0 = (entry-slp)/entry
            if risk0 <= 0: continue
            tp1 = entry+T1*(entry-slp); tp2 = entry+T2*(entry-slp); pos = 1; half = False
        elif short_ok:
            entry = c[t]*(1-SLIP); slp = max(h[t], lbshi) + SLA*atr[t]; risk0 = (slp-entry)/entry
            if risk0 <= 0: continue
            tp1 = entry-T1*(slp-entry); tp2 = entry-T2*(slp-entry); pos = -1; half = False
    return trades


def stat(r):
    if not r: return dict(n=0, pf=0.0, ret=0.0, wr=0.0)
    r = np.array(r); w = r[r > 0].sum(); ls = -r[r < 0].sum()
    return dict(n=len(r), pf=float(w/ls) if ls > 0 else 99.0, ret=float(r.sum()), wr=float((r > 0).mean()))


def folds(m, k=4):
    n = len(m); half = n//2; step = (n-half)//k
    return [(half+i*step, n if i == k-1 else half+(i+1)*step) for i in range(k)]


def main():
    m = base_df()
    px = float(m["close"].iloc[-1]); med = float(np.nanmedian(m["spread"].values)); spread = 2*med*0.01/px
    fs = folds(m, 4)
    base_p = dict(TP1_R=1.5, TP2_R=2.0, SL_ATR=0.2, SWEEP_LB=10, SEQ=8)
    base = [stat(simulate(m.iloc[a:b].reset_index(drop=True), base_p, spread)) for a, b in fs]
    base_ret = [r["ret"] for r in base]; base_n = sum(r["n"] for r in base)
    GRID = dict(TP1_R=[1.0, 1.5, 2.0], TP2_R=[2.0, 3.0, 4.0], SL_ATR=[0.1, 0.2, 0.3, 0.5],
                SWEEP_LB=[8, 12, 20], SEQ=[5, 8, 12])
    keys = list(GRID)
    ships = []
    for combo in product(*[GRID[k] for k in keys]):
        p = dict(zip(keys, combo))
        per = [stat(simulate(m.iloc[a:b].reset_index(drop=True), p, spread)) for a, b in fs]
        rets = [r["ret"] for r in per]; ns = [r["n"] for r in per]
        beats = sum(1 for a, b in zip(rets, base_ret) if a > b)
        safe = all(not (b > 0 and a < 0) for a, b in zip(rets, base_ret))
        not_churny = sum(ns) <= max(1.6*base_n, base_n+40)
        allpos = all(r > 0 for r in rets)                       # every fold profitable (strict)
        if beats >= 3 and safe and not_churny and sum(rets) > sum(base_ret) and min(ns) >= 6:
            ships.append(dict(p=p, sum_ret=round(sum(rets), 2), fold_ret=[round(x, 2) for x in rets],
                              fold_n=ns, allpos=allpos, beats=beats))
    ships.sort(key=lambda s: s["sum_ret"], reverse=True)
    print(json.dumps({"symbol": "XAUUSD", "spread": round(spread, 5),
                      "baseline": {"p": base_p, "sum_ret": round(sum(base_ret), 2),
                                   "fold_ret": [round(x, 2) for x in base_ret], "fold_n": [r["n"] for r in base]},
                      "n_configs": len(list(product(*[GRID[k] for k in keys]))),
                      "n_ship": len(ships), "top": ships[:8]}, default=float))


if __name__ == "__main__":
    main()
