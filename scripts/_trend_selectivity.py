#!/usr/bin/env python3 -B
"""TREND SELECTIVITY tuner (2026-07-10) — "PF 6.91" discipline: take FEWER, higher-
conviction trades to raise Profit Factor. The 3-EMA signal already needs ALL 3
pairs to agree, so the conviction lever is TREND STRENGTH, not agreement.

Sweeps two daily conviction gates on top of the existing signal + deployed
per-symbol exit model, on H1 (real intraday exit path), walk-forward IS/OOS:
  ADX_MIN   — only enter when daily ADX(14) >= thr (skip weak/choppy trends)
  SLOPE_MIN — only enter when |slow-EMA slope| over 10 bars >= thr*ATR (trend moving)

Reports, per symbol, PF + trade count for each conviction level so we can pick the
selectivity that MAXIMIZES OOS PF (accepting fewer trades). Realistic spread,
bar-close causal.

Usage: python3 -B scripts/_trend_selectivity.py SYMBOL
"""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
EMA_PAIRS = [(16, 64), (32, 128), (64, 256)]
MIN_ABS = 0.34
ATR_P = 20


def _naive(s):
    s = pd.to_datetime(s)
    try: return s.dt.tz_localize(None)
    except (TypeError, AttributeError): return s


def _atr(h, l, c, n):
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False).mean()


def _adx(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    plus = ((up > dn) & (up > 0)) * up
    minus = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1.0/n, adjust=False).mean()
    pdi = 100*plus.ewm(alpha=1.0/n, adjust=False).mean()/a.replace(0, np.nan)
    mdi = 100*minus.ewm(alpha=1.0/n, adjust=False).mean()/a.replace(0, np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1.0/n, adjust=False).mean().fillna(0)


def d1_ctx(sym, params):
    d = pickle.load(open(C/("raw_d1_"+sym.replace(".", "_")+".pkl"), "rb"))
    d["time"] = _naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
    c, h, l = d["close"], d["high"], d["low"]
    sig = pd.Series(0.0, index=d.index)
    for f, s in EMA_PAIRS:
        sig = sig + np.sign(c.ewm(span=f).mean() - c.ewm(span=s).mean())
    sig = (sig/len(EMA_PAIRS)).apply(lambda v: 0 if abs(v) < MIN_ABS else (1 if v > 0 else -1))
    atr = _atr(h, l, c, ATR_P)
    slow = c.ewm(span=256).mean()
    slope = (slow - slow.shift(10)).abs() / atr.replace(0, np.nan)
    adx = _adx(h, l, c, 14)
    lb = int(params["LOOKBACK"])
    return pd.DataFrame({"eff": (d["time"].dt.normalize()+pd.Timedelta(days=1)).astype("datetime64[ns]"),
                         "sig": sig.astype(int), "atr": atr, "adx": adx, "slope": slope.fillna(0),
                         "hh": h.rolling(lb).max(), "ll": l.rolling(lb).min()}).dropna().reset_index(drop=True)


def build(sym, params):
    h1 = pickle.load(open(C/("raw_h1_"+sym.replace(".", "_")+".pkl"), "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    h1["time"] = h1["time"].astype("datetime64[ns]")
    d1 = d1_ctx(sym, params); d1["eff"] = d1["eff"].astype("datetime64[ns]")
    m = pd.merge_asof(h1, d1, left_on="time", right_on="eff", direction="backward")
    return m.dropna(subset=["sig", "atr", "hh", "ll", "adx", "slope"]).reset_index(drop=True)


def simulate(m, ex, ADX_MIN, SLOPE_MIN, cost):
    o, h, l, c = [m[x].values for x in ("open", "high", "low", "close")]
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    adx = m["adx"].values; slope = m["slope"].values
    hh = m["hh"].values; ll = m["ll"].values
    STOP, TR, LK, GB, AC = ex["STOP"], ex["TRAIL"], ex["LOCK"], ex["GIVEBACK"], ex["ACT"]
    pos = 0; entry = sl = peak = 0.0; blocked = 0; trades = []
    for t in range(len(m)):
        a = atr[t]
        if a <= 0: continue
        s = int(sig[t])
        conviction = adx[t] >= ADX_MIN and slope[t] >= SLOPE_MIN
        if pos == 0:
            if blocked and s != blocked: blocked = 0
            if s != 0 and s != blocked and conviction:      # CONVICTION GATE
                pos = s; entry = o[t]; peak = 0.0
                sl = entry - STOP*a if s == 1 else entry + STOP*a
            continue
        if s != 0 and s != pos:                              # flip
            trades.append(((o[t]-entry)/entry)*pos - cost)
            pos = 0; blocked = 0
            continue
        if pos == 1:
            sl = max(sl, hh[t] - TR*a)
            if peak >= AC*a: sl = max(sl, entry + LK*peak)
            gb = entry + peak*(1-GB) if (GB < 1 and peak >= AC*a) else -1e18
            if l[t] <= sl: trades.append((sl-entry)/entry - cost); pos = 0
            elif gb > -1e17 and l[t] <= gb: trades.append((gb-entry)/entry - cost); pos = 0; blocked = 1
            else: peak = max(peak, h[t]-entry)
        else:
            sl = min(sl, ll[t] + TR*a)
            if peak >= AC*a: sl = min(sl, entry - LK*peak)
            gb = entry - peak*(1-GB) if (GB < 1 and peak >= AC*a) else 1e18
            if h[t] >= sl: trades.append((entry-sl)/entry - cost); pos = 0
            elif gb < 1e17 and h[t] >= gb: trades.append((entry-gb)/entry - cost); pos = 0; blocked = -1
            else: peak = max(peak, entry-l[t])
    return trades


def stats(r, yrs):
    if not r: return {"n": 0, "tpy": 0.0, "ret": 0.0, "pf": 0.0, "wr": 0.0}
    r = np.array(r); w = r[r > 0].sum(); ls = -r[r < 0].sum()
    return {"n": len(r), "tpy": len(r)/max(yrs, 0.1), "ret": float(r.sum()),
            "pf": float(w/ls) if ls > 0 else (99.0 if w > 0 else 0.0), "wr": float((r > 0).mean())}


def main():
    sym = sys.argv[1]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import trend_exit_params
    tr, lk, gb, ac = trend_exit_params(sym)
    ex = {"STOP": 3.0, "LOOKBACK": 22, "TRAIL": tr, "LOCK": lk, "GIVEBACK": gb, "ACT": ac}
    m = build(sym, ex)
    px = float(m["close"].iloc[-1])
    cost = 2.0*(float(np.nanmedian(m["spread"].values))*0.01)/px if "spread" in m else 0.0002
    split = int(len(m)*0.70)
    mis, mos = m.iloc[:split].reset_index(drop=True), m.iloc[split:].reset_index(drop=True)
    yi = (mis["time"].iloc[-1]-mis["time"].iloc[0]).days/365.25
    yo = (mos["time"].iloc[-1]-mos["time"].iloc[0]).days/365.25
    print(f"{sym}  H1={len(m)}  OOS={yo:.1f}y  (exit={ex['TRAIL']}/{ex['LOCK']}/{ex['GIVEBACK']}/{ex['ACT']})")
    print(f"  {'ADX_MIN':>7} {'SLOPE':>5} | {'IS pf/n':>12} | {'OOS pf':>6} {'OOS ret':>8} {'OOS n':>5} {'OOS wr':>6} {'tpy':>4}")
    grid = []
    for adxm in [0, 15, 20, 25, 30]:
        for slm in [0.0, 0.5, 1.0]:
            mi = stats(simulate(mis, ex, adxm, slm, cost), yi)
            mo = stats(simulate(mos, ex, adxm, slm, cost), yo)
            grid.append((adxm, slm, mi, mo))
            print(f"  {adxm:>7} {slm:>5} | {mi['pf']:>6.2f}/{mi['n']:>4} | {mo['pf']:>6.2f} {mo['ret']:>+8.2f} {mo['n']:>5} {mo['wr']:>5.0%} {mo['tpy']:>4.0f}")
    # best OOS PF with >= 8 OOS trades AND IS pf>1 (robust)
    ok = [g for g in grid if g[3]["n"] >= 8 and g[2]["pf"] > 1.0 and g[3]["ret"] > 0]
    ok.sort(key=lambda g: g[3]["pf"], reverse=True)
    if ok:
        b = ok[0]
        print(f"  >>> BEST robust: ADX_MIN={b[0]} SLOPE_MIN={b[1]} -> OOS pf={b[3]['pf']:.2f} ret={b[3]['ret']:+.2f} n={b[3]['n']}")
    else:
        print("  >>> no robust conviction gate beats baseline")


if __name__ == "__main__":
    main()
