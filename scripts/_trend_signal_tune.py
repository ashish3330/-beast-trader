#!/usr/bin/env python3 -B
"""HARDEST TUNE — the ENTRY SIGNAL itself (2026-07-11). All session I held the
3-EMA ensemble fixed (16/64, 32/128, 64/256). This sweeps the signal construction
+ regime split, per symbol, with 4-fold ANCHORED rolling walk-forward on deep H1,
using the deployed per-symbol exits. Ships a config ONLY if it beats the CURRENT
signal on RETURN in >=3/4 OOS folds AND every fold has >= MIN_TRADES (no thin
overfit cells) AND doesn't turn a winning fold into a loss.

Dimensions swept:
  EMA_SET   : the 3 crossover pairs (speed of the trend definition)
  MIN_ABS   : agreement threshold (0.34 = all-3-agree; lower = 2/3 ok)
  REGIME    : optionally require ADX(14)>=thr (trending-only) — a market-condition
              gate applied to the SIGNAL, validated per-fold with sample guards.

Usage: python3 -B scripts/_trend_signal_tune.py SYMBOL  -> JSON
"""
import json
import pickle
import sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
ATR_P = 20
STOP = 3.0
LB = 22
MIN_TRADES = 6            # per-fold OOS floor — reject thin overfit cells
CUR_EMA = [(16, 64), (32, 128), (64, 256)]
CUR_MINABS = 0.34

EMA_SETS = {
    "cur":   [(16, 64), (32, 128), (64, 256)],
    "fast":  [(8, 32), (16, 64), (32, 128)],
    "slow":  [(32, 128), (64, 256), (128, 400)],
    "wide":  [(16, 96), (32, 160), (64, 256)],
    "dual":  [(20, 100), (50, 200)],
}
MINABS_SET = [0.34, 0.65, 0.99]     # note: 3 pairs -> {0.33,1.0}; 2 pairs -> {0,0.5,1.0}
ADX_SET = [0, 20, 25]


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


def d1_ctx(sym, ema_set, min_abs, adx_min):
    d = pickle.load(open(C/("raw_d1_"+sym.replace(".", "_")+".pkl"), "rb"))
    d["time"] = _naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
    c, h, l = d["close"], d["high"], d["low"]
    sig = pd.Series(0.0, index=d.index)
    for f, s in ema_set:
        sig = sig + np.sign(c.ewm(span=f).mean() - c.ewm(span=s).mean())
    sig = (sig/len(ema_set)).apply(lambda v: 0 if abs(v) < min_abs else (1 if v > 0 else -1))
    if adx_min > 0:
        adx = _adx(h, l, c, 14)
        sig = sig.where(adx >= adx_min, 0)          # regime: trending-only signal
    atr = _atr(h, l, c, ATR_P)
    return pd.DataFrame({"time": d["time"], "sig": sig.astype(int), "atr": atr,
                         "hh": h.rolling(LB).max(), "ll": l.rolling(LB).min(),
                         "eff": (d["time"].dt.normalize()+pd.Timedelta(days=1)).astype("datetime64[ns]")
                         }).dropna().reset_index(drop=True)


def build(sym, ema_set, min_abs, adx_min):
    h1 = pickle.load(open(C/("raw_h1_"+sym.replace(".", "_")+".pkl"), "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    h1["time"] = h1["time"].astype("datetime64[ns]")
    d1 = d1_ctx(sym, ema_set, min_abs, adx_min)
    m = pd.merge_asof(h1, d1[["eff", "sig", "atr", "hh", "ll"]], left_on="time",
                      right_on="eff", direction="backward")
    return m.dropna(subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)


def simulate(m, ex, cost):
    o, h, l, c = [m[x].values for x in ("open", "high", "low", "close")]
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    TR, LK, GB, AC = ex["TRAIL"], ex["LOCK"], ex["GIVEBACK"], ex["ACT"]
    pos = 0; entry = sl = peak = 0.0; blocked = 0; trades = []
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


def stats(r):
    if not r: return {"n": 0, "ret": 0.0, "pf": 0.0, "wr": 0.0}
    r = np.array(r); w = r[r > 0].sum(); ls = -r[r < 0].sum()
    return {"n": len(r), "ret": float(r.sum()),
            "pf": float(w/ls) if ls > 0 else (99.0 if w > 0 else 0.0), "wr": float((r > 0).mean())}


def folds(m, k=4):
    n = len(m); half = n//2; step = (n-half)//k
    return [(m.iloc[:(half+i*step)].reset_index(drop=True),
             m.iloc[(half+i*step):(n if i == k-1 else half+(i+1)*step)].reset_index(drop=True))
            for i in range(k)]


def eval_cfg(sym, ema_set, min_abs, adx_min, ex, cost_ref):
    m = build(sym, ema_set, min_abs, adx_min)
    if len(m) < 5000:
        return None
    per = []
    for is_seg, oos in folds(m, 4):
        per.append(stats(simulate(oos, ex, cost_ref)))
    return per


def main():
    sym = sys.argv[1]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import trend_exit_params
    tr, lk, gb, ac = trend_exit_params(sym)
    ex = {"TRAIL": tr, "LOCK": lk, "GIVEBACK": gb, "ACT": ac}
    # reference cost from current-signal build
    m0 = build(sym, CUR_EMA, CUR_MINABS, 0)
    px = float(m0["close"].iloc[-1])
    cost = 2.0*(float(np.nanmedian(m0["spread"].values))*0.01)/px

    cur = eval_cfg(sym, CUR_EMA, CUR_MINABS, 0, ex, cost)
    cur_rets = [f["ret"] for f in cur]

    results = []
    for name, es in EMA_SETS.items():
        for ma in MINABS_SET:
            for adx in ADX_SET:
                if name == "cur" and ma == CUR_MINABS and adx == 0:
                    continue
                per = eval_cfg(sym, es, ma, adx, ex, cost)
                if per is None:
                    continue
                rets = [f["ret"] for f in per]
                ns = [f["n"] for f in per]
                beats = sum(1 for a, b in zip(rets, cur_rets) if a > b)
                safe = all(not (b > 0 and a < 0) for a, b in zip(rets, cur_rets))
                thick = all(n >= MIN_TRADES for n in ns)          # no thin overfit fold
                # CHURN GUARD: a signal change must NOT explode turnover (intra-bar
                # -fill inflates high-frequency configs). Cap at 1.5x current trades.
                cur_total = sum(f["n"] for f in cur)
                not_churny = sum(ns) <= max(1.5 * cur_total, cur_total + 20)
                ship = beats >= 3 and safe and thick and not_churny and sum(rets) > sum(cur_rets)
                results.append({"ema": name, "min_abs": ma, "adx": adx,
                                "beats": beats, "safe": safe, "thick": thick, "SHIP": ship,
                                "sum_ret": round(sum(rets), 3), "fold_ret": [round(x, 3) for x in rets],
                                "fold_n": ns})
    ships = [r for r in results if r["SHIP"]]
    ships.sort(key=lambda r: r["sum_ret"], reverse=True)
    print(json.dumps({"symbol": sym, "cur_sum_ret": round(sum(cur_rets), 3),
                      "cur_fold_ret": [round(x, 3) for x in cur_rets],
                      "cur_fold_n": [f["n"] for f in cur],
                      "n_configs": len(results), "n_ship": len(ships),
                      "ships": ships[:5]}, default=float))


if __name__ == "__main__":
    main()
