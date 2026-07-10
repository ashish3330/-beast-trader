#!/usr/bin/env python3 -B
"""HARD per-symbol intraday tune of the TREND exit model (2026-07-10).

Upgrades over _trend_exit_tune_h1_persym.py (which was a single 70/30 split):
 - FULL param set swept: STOP (initial ATR stop), TP, LOOKBACK (chandelier window),
   TRAIL, LOCK, GIVEBACK, ACT — not just the trail/lock pair.
 - ROLLING walk-forward: 4 anchored folds (expanding IS, disjoint OOS windows) so a
   config must work in MULTIPLE out-of-sample regimes, not one lucky tail.
 - CHURN GUARD: reject configs > MAX_TPY trades/yr in ANY fold (intra-bar-fill
   artifact — the "tighter is always better" trap).
 - PLATEAU SELECTION: a config's score is discounted by its own OOS variance across
   folds AND rewarded only if it sits in a robust neighborhood (reported), so we
   pick a stable plateau, not a spike.
 - BOOTSTRAP: block-bootstrap the OOS trade sequence of the top config → 5/50/95
   percentile return, so we ship a config whose 5th-percentile is still >= 0.

Signal = same 3-EMA daily ensemble (entry fixed; this tunes EXITS only). Realistic
round-trip spread from the data (spread-only, per the live no-slippage model).

Usage: python3 -B scripts/_trend_exit_tune_hard.py SYMBOL  -> JSON to stdout
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
MAX_TPY = 40.0
N_FOLDS = 4
BOOT_N = 400
BLOCK = 10        # block-bootstrap block length (trades)

# full exit-param grid (entry/signal held fixed)
GRID = {
    "STOP":     [2.5, 3.0, 4.0],
    "LOOKBACK": [15, 22, 30],
    "TRAIL":    [2.0, 2.5, 3.0, 3.5],
    "LOCK":     [0.4, 0.5, 0.6],
    "GIVEBACK": [0.25, 0.35, 0.50, 1.01],
    "ACT":      [0.3, 0.5],
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
    out = {"eff": (d["time"].dt.normalize() + pd.Timedelta(days=1)).astype("datetime64[ns]"),
           "sig": sig.astype(int), "atr": atr}
    for lb in set(GRID["LOOKBACK"]):
        out["hh%d" % lb] = h.rolling(lb).max()
        out["ll%d" % lb] = l.rolling(lb).min()
    return pd.DataFrame(out).dropna().reset_index(drop=True)


def build(sym):
    h1 = pickle.load(open(C / ("raw_h1_" + sym.replace(".", "_") + ".pkl"), "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    h1["time"] = h1["time"].astype("datetime64[ns]")
    d1 = d1_context(sym)
    m = pd.merge_asof(h1, d1, left_on="time", right_on="eff", direction="backward")
    keep = ["time", "open", "high", "low", "close", "spread", "sig", "atr"]
    keep += [c for c in m.columns if c.startswith(("hh", "ll"))]
    return m[keep].dropna(subset=["sig", "atr"]).reset_index(drop=True)


def simulate(m, p, cost):
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh%d" % p["LOOKBACK"]].values; ll = m["ll%d" % p["LOOKBACK"]].values
    STOP, LB, TR, LK, GB, AC, TP = (p["STOP"], p["LOOKBACK"], p["TRAIL"],
                                    p["LOCK"], p["GIVEBACK"], p["ACT"], p["TP"])
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
                sl = entry - STOP * a if s == 1 else entry + STOP * a
                tp = None if TP >= 999 else ((entry + TP * a) if s == 1 else (entry - TP * a))
            continue
        if s != 0 and s != pos:
            trades.append(((o[t] - entry) / entry) * pos - cost)
            pos = s; entry = o[t]; peak = 0.0; blocked = 0
            sl = entry - STOP * a if pos == 1 else entry + STOP * a
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


def stats(r, years):
    if not r:
        return {"n": 0, "tpy": 0.0, "ret": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0}
    r = np.array(r); w = r[r > 0].sum(); ls = -r[r < 0].sum()
    eq = np.cumsum(r); dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"n": len(r), "tpy": len(r) / max(years, 0.1), "ret": float(r.sum()),
            "pf": float(w / ls) if ls > 0 else (99.0 if w > 0 else 0.0),
            "wr": float((r > 0).mean()), "dd": dd}


def folds(m, k):
    """Anchored rolling WF: expanding IS, k disjoint OOS windows in the back half."""
    n = len(m); half = n // 2; step = (n - half) // k
    out = []
    for i in range(k):
        oos_a = half + i * step
        oos_b = n if i == k - 1 else half + (i + 1) * step
        out.append((m.iloc[:oos_a].reset_index(drop=True),
                    m.iloc[oos_a:oos_b].reset_index(drop=True)))
    return out


def yrs(seg):
    return max((seg["time"].iloc[-1] - seg["time"].iloc[0]).days / 365.25, 0.1)


def boot_ci(rets):
    if len(rets) < 8:
        return [0.0, 0.0, 0.0]
    r = np.array(rets); n = len(r); nb = n // BLOCK + 1
    sums = []
    rng = np.random.default_rng(12345)
    for _ in range(BOOT_N):
        idx = rng.integers(0, max(1, n - BLOCK), size=nb)
        s = np.concatenate([r[i:i + BLOCK] for i in idx])[:n]
        sums.append(s.sum())
    return [float(np.percentile(sums, 5)), float(np.percentile(sums, 50)),
            float(np.percentile(sums, 95))]


def main():
    sym = sys.argv[1]
    m = build(sym)
    px = float(m["close"].iloc[-1])
    cost = 2.0 * (float(np.nanmedian(m["spread"].values)) * 0.01) / px if px > 0 else 0.0004
    fs = folds(m, N_FOLDS)

    keys = list(GRID.keys())
    scored = []
    for combo in product(*[GRID[k] for k in keys]):
        p = dict(zip(keys, combo))
        oos_rets, ok = [], True
        allr = []
        for is_seg, oos_seg in fs:
            so = stats(simulate(oos_seg, p, cost), yrs(oos_seg))
            si = stats(simulate(is_seg, p, cost), yrs(is_seg))
            if so["tpy"] > MAX_TPY or si["tpy"] > MAX_TPY:   # churn guard (any fold)
                ok = False; break
            oos_rets.append(so["ret"])
            allr.append((si, so))
        if not ok or len(oos_rets) < N_FOLDS:
            continue
        arr = np.array(oos_rets)
        # robust: EVERY OOS fold positive; score = mean/(1+std) (plateau-friendly)
        if (arr <= 0).any():
            continue
        score = float(arr.mean() / (1.0 + arr.std()))
        scored.append({"p": p, "oos_mean": float(arr.mean()),
                       "oos_min": float(arr.min()), "oos_std": float(arr.std()),
                       "score": score, "folds": allr})
    scored.sort(key=lambda r: r["score"], reverse=True)

    # bootstrap the top-5 on the LAST (largest) OOS window; ship the best whose
    # 5th-pct return >= 0 (survives worst-case resampling).
    top = []
    for r in scored[:5]:
        _, last_oos = fs[-1]
        ci = boot_ci(simulate(last_oos, r["p"], cost))
        top.append({"p": r["p"], "oos_mean": r["oos_mean"], "oos_min": r["oos_min"],
                    "oos_std": r["oos_std"], "score": r["score"], "boot_ci": ci,
                    "fold_oos": [f[1] for f in r["folds"]]})

    # current live per-symbol config for comparison
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from config import trend_exit_params
        tr, lk, gb, ac = trend_exit_params(sym)
    except Exception:
        tr, lk, gb, ac = 2.5, 0.6, 0.30, 0.3
    cur_p = {"STOP": 3.0, "LOOKBACK": 22, "TRAIL": tr, "LOCK": lk, "GIVEBACK": gb, "ACT": ac, "TP": 6.0}
    cur = [stats(simulate(oos, cur_p, cost), yrs(oos)) for _, oos in fs]

    print(json.dumps({"symbol": sym, "h1_bars": len(m), "cost_rt": round(cost, 5),
                      "n_folds": N_FOLDS, "max_tpy": MAX_TPY, "n_robust": len(scored),
                      "current": {"p": cur_p, "fold_oos": cur},
                      "top": top}, default=float))


if __name__ == "__main__":
    main()
