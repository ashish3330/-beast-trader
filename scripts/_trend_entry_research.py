#!/usr/bin/env python3 -B
"""TREND ENTRY-QUALITY RESEARCH (2026-07-10) — the exits work (+$96 peak-giveback);
the ENTRIES are weak (32% live WR). Test entry FILTERS on top of the all-3-EMA
signal + deployed per-symbol exits, per symbol, walk-forward IS(70%)/OOS(30%) on
deep H1. Each filter tested INDEPENDENTLY vs baseline so we can see what actually
lifts PF at a sane trade count. Realistic spread, bar-close causal.

Filters (daily, evaluated at the signal bar):
  ADX      ADX(14) >= thr           (trend strength)
  SLOPE    |EMA256 slope/ATR| >= thr (trend actually moving)
  WEEKLY   weekly EMA(8) vs EMA(21) agrees with the daily direction (HTF align)
  DIST     |close-EMA256|/ATR <= thr (skip over-extended / late entries)
  VOLBAND  ATR/price percentile in [lo,hi] (skip dead + blowout vol regimes)

Usage: python3 -B scripts/_trend_entry_research.py SYMBOL  -> JSON
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
STOP = 3.0
LB = 22


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


def d1_ctx(sym):
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
    dist = (c - slow).abs() / atr.replace(0, np.nan)
    volp = (atr / c).rolling(252, min_periods=60).rank(pct=True)   # vol percentile
    # weekly HTF alignment
    dw = d.set_index("time")["close"].resample("1W").last().dropna()
    wk = (np.sign(dw.ewm(span=8).mean() - dw.ewm(span=21).mean()))
    wk = wk.reindex(d["time"].dt.normalize().values, method="ffill") if False else wk
    # map weekly sign back to daily by asof
    wser = wk.reset_index(); wser.columns = ["time", "wk"]
    out = pd.DataFrame({"time": d["time"], "sig": sig.astype(int), "atr": atr,
                        "hh": h.rolling(LB).max(), "ll": l.rolling(LB).min(),
                        "adx": adx, "slope": slope.fillna(0), "dist": dist.fillna(0),
                        "volp": volp.fillna(0.5)})
    out = pd.merge_asof(out.sort_values("time"), wser.sort_values("time"), on="time", direction="backward")
    out["wk"] = out["wk"].fillna(0)
    out["eff"] = (out["time"].dt.normalize()+pd.Timedelta(days=1)).astype("datetime64[ns]")
    return out.dropna(subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)


def build(sym):
    h1 = pickle.load(open(C/("raw_h1_"+sym.replace(".", "_")+".pkl"), "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    h1["time"] = h1["time"].astype("datetime64[ns]")
    d1 = d1_ctx(sym); d1["eff"] = d1["eff"].astype("datetime64[ns]")
    m = pd.merge_asof(h1, d1[["eff", "sig", "atr", "hh", "ll", "adx", "slope", "dist", "volp", "wk"]],
                      left_on="time", right_on="eff", direction="backward")
    return m.dropna(subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)


def simulate(m, ex, gate, cost):
    o, h, l, c = [m[x].values for x in ("open", "high", "low", "close")]
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    adx = m["adx"].values; slope = m["slope"].values; dist = m["dist"].values
    volp = m["volp"].values; wk = m["wk"].values
    TR, LK, GB, AC = ex["TRAIL"], ex["LOCK"], ex["GIVEBACK"], ex["ACT"]
    pos = 0; entry = sl = peak = 0.0; blocked = 0; trades = []
    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        # conviction gate on ENTRY only
        ok = (adx[t] >= gate.get("ADX", 0) and slope[t] >= gate.get("SLOPE", 0)
              and dist[t] <= gate.get("DIST", 1e9)
              and gate.get("VLO", 0.0) <= volp[t] <= gate.get("VHI", 1.0))
        if gate.get("WEEKLY") and s != 0:
            ok = ok and (int(wk[t]) == s)
        if pos == 0:
            if blocked and s != blocked:
                blocked = 0
            if s != 0 and s != blocked and ok:
                pos = s; entry = o[t]; peak = 0.0
                sl = entry - STOP*a if s == 1 else entry + STOP*a
            continue
        if s != 0 and s != pos:
            trades.append(((o[t]-entry)/entry)*pos - cost); pos = 0; blocked = 0
            continue
        if pos == 1:
            sl = max(sl, hh[t]-TR*a)
            if peak >= AC*a: sl = max(sl, entry+LK*peak)
            gb = entry+peak*(1-GB) if (GB < 1 and peak >= AC*a) else -1e18
            if l[t] <= sl: trades.append((sl-entry)/entry - cost); pos = 0
            elif gb > -1e17 and l[t] <= gb: trades.append((gb-entry)/entry - cost); pos = 0; blocked = 1
            else: peak = max(peak, h[t]-entry)
        else:
            sl = min(sl, ll[t]+TR*a)
            if peak >= AC*a: sl = min(sl, entry-LK*peak)
            gb = entry-peak*(1-GB) if (GB < 1 and peak >= AC*a) else 1e18
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
    ex = {"TRAIL": tr, "LOCK": lk, "GIVEBACK": gb, "ACT": ac}
    m = build(sym)
    px = float(m["close"].iloc[-1])
    cost = 2.0*(float(np.nanmedian(m["spread"].values))*0.01)/px if "spread" in m else 0.0002
    split = int(len(m)*0.70)
    mis, mos = m.iloc[:split].reset_index(drop=True), m.iloc[split:].reset_index(drop=True)
    yi = (mis["time"].iloc[-1]-mis["time"].iloc[0]).days/365.25
    yo = (mos["time"].iloc[-1]-mos["time"].iloc[0]).days/365.25

    # candidate gates (each independent). baseline = no gate.
    gates = {"baseline": {}}
    for v in [20, 25, 30]: gates[f"ADX{v}"] = {"ADX": v}
    for v in [0.3, 0.5, 0.8]: gates[f"SLOPE{v}"] = {"SLOPE": v}
    for v in [1.5, 2.5, 4.0]: gates[f"DIST{v}"] = {"DIST": v}
    gates["WEEKLY"] = {"WEEKLY": True}
    gates["VOL20-90"] = {"VLO": 0.2, "VHI": 0.9}
    gates["VOL10-80"] = {"VLO": 0.1, "VHI": 0.8}

    base = None; rows = []
    for name, g in gates.items():
        mi = stats(simulate(mis, ex, g, cost), yi)
        mo = stats(simulate(mos, ex, g, cost), yo)
        r = {"gate": name, "is": mi, "oos": mo}
        if name == "baseline": base = r
        rows.append(r)
    # winners: OOS PF > baseline*1.15 AND IS PF > baseline_IS AND OOS n>=8 AND both positive
    b_ispf = base["is"]["pf"]; b_oospf = base["oos"]["pf"]; b_oosret = base["oos"]["ret"]
    wins = [r for r in rows if r["gate"] != "baseline"
            and r["oos"]["ret"] > 0 and r["is"]["ret"] > 0 and r["oos"]["n"] >= 8
            and r["oos"]["pf"] >= b_oospf*1.15 and r["is"]["pf"] >= b_ispf
            and r["oos"]["ret"] >= b_oosret*0.9]
    wins.sort(key=lambda r: r["oos"]["pf"], reverse=True)
    print(json.dumps({"symbol": sym, "h1": len(m), "cost": round(cost, 5),
                      "baseline": base, "candidates": rows, "winners": wins[:5]}, default=float))


if __name__ == "__main__":
    main()
