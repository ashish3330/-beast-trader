#!/usr/bin/env python3 -B
"""Scan ALL available symbols with V5 backtest logic."""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.momentum_scorer import _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES
from backtest.v5_backtest import simulate_trail, get_regime

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

ALL = {
    "XAUUSD":    {"f":"raw_h1_xauusd.pkl",   "sp":0.30,   "cat":"Gold"},
    "XAGUSD":    {"f":"raw_h1_XAGUSD.pkl",   "sp":0.030,  "cat":"Gold"},
    "BTCUSD":    {"f":"raw_h1_BTCUSD.pkl",   "sp":30.0,   "cat":"Crypto"},
    "ETHUSD":    {"f":"raw_h1_ETHUSD.pkl",   "sp":2.0,    "cat":"Crypto"},
    "NAS100.r":  {"f":"raw_h1_NAS100_r.pkl", "sp":1.50,   "cat":"Index"},
    "SP500.r":   {"f":"raw_h1_SP500_r.pkl",  "sp":0.50,   "cat":"Index"},
    "JPN225ft":  {"f":"raw_h1_JPN225ft.pkl", "sp":10.0,   "cat":"Index"},
    "GER40.r":   {"f":"raw_h1_GER40_r.pkl",  "sp":2.0,    "cat":"Index"},
    "UK100.r":   {"f":"raw_h1_UK100_r.pkl",  "sp":2.0,    "cat":"Index"},
    "EURUSD":    {"f":"raw_h1_EURUSD.pkl",   "sp":0.00015,"cat":"Forex"},
    "GBPUSD":    {"f":"raw_h1_GBPUSD.pkl",   "sp":0.00020,"cat":"Forex"},
    "USDJPY":    {"f":"raw_h1_USDJPY.pkl",   "sp":0.015,  "cat":"Forex"},
    "USDCAD":    {"f":"raw_h1_USDCAD.pkl",   "sp":0.00020,"cat":"Forex"},
    "USDCHF":    {"f":"raw_h1_USDCHF.pkl",   "sp":0.00020,"cat":"Forex"},
    "EURJPY":    {"f":"raw_h1_EURJPY.pkl",   "sp":0.020,  "cat":"Forex"},
    "GBPJPY":    {"f":"raw_h1_GBPJPY.pkl",   "sp":0.025,  "cat":"Forex"},
    "AUDUSD":    {"f":"raw_h1_AUDUSD.pkl",   "sp":0.00018,"cat":"Forex"},
    "NZDUSD":    {"f":"raw_h1_NZDUSD.pkl",   "sp":0.00020,"cat":"Forex"},
    "AUDJPY":    {"f":"raw_h1_AUDJPY.pkl",   "sp":0.020,  "cat":"Forex"},
    "EURAUD":    {"f":"raw_h1_EURAUD.pkl",   "sp":0.00030,"cat":"Forex"},
    "EURCHF":    {"f":"raw_h1_EURCHF.pkl",   "sp":0.00025,"cat":"Forex"},
    "EURGBP":    {"f":"raw_h1_EURGBP.pkl",   "sp":0.00020,"cat":"Forex"},
    "GBPCHF":    {"f":"raw_h1_GBPCHF.pkl",   "sp":0.00030,"cat":"Forex"},
    "COPPER-Cr": {"f":"raw_h1_COPPER-Cr.pkl","sp":0.50,   "cat":"Commodity"},
    "UKOUSD":    {"f":"raw_h1_UKOUSD.pkl",   "sp":0.05,   "cat":"Commodity"},
}

trail = [(8.0,0.3,"trail"),(4.0,0.5,"trail"),(2.0,0.8,"trail"),
         (1.5,0.7,"lock"),(1.0,0.4,"lock"),(0.7,0.2,"lock"),(0.5,0.0,"be")]
days = 90
results = []

print(f"{'Sym':14s} {'Cat':9s} {'N':>4s} {'WR':>5s} {'PF':>6s} {'PnL':>9s} {'DD':>5s} {'aR':>5s} {'Pk':>4s} {'Gb':>4s} {'D':2s}")
print("-" * 80)

for sym, meta in sorted(ALL.items()):
    path = CACHE / meta["f"]
    if not path.exists():
        continue
    try:
        df = pickle.load(open(path, "rb"))
    except Exception:
        print(f"{sym:14s} CORRUPT DATA — skipped")
        continue
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    df = df[df["time"] >= cutoff].reset_index(drop=True)
    if len(df) < 200:
        continue

    icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(sym, {})}
    warmup = max(icfg.get("EMA_T", 80), 100) + 30
    ind = _compute_indicators(df, icfg)
    if ind is None:
        continue

    c = df["close"].values.astype(float)
    h2 = df["high"].values.astype(float)
    l2 = df["low"].values.astype(float)
    n = len(df)
    times = df["time"].values
    sp = meta["sp"]
    cat = meta["cat"]

    eq = 1000.0
    pk_eq = eq
    trades = []
    cd = 0
    cl = 0

    for i in range(warmup, n - 1):
        if eq <= 0 or i < cd:
            continue
        hr = pd.Timestamp(times[i]).hour
        if cat not in ("Crypto",) and (hr < 6 or hr >= 22):
            continue
        if hr in (1, 2, 3, 4) and cat not in ("Crypto",):
            continue
        bi = i
        if bi < 21 or np.isnan(ind["at"][bi]) or ind["at"][bi] == 0:
            continue
        ls, ss = _score(ind, bi)
        ls, ss = float(ls), float(ss)
        raw = max(ls, ss)
        sq = min(100, raw / 12 * 100)
        if sq < 45:
            continue
        d = 1 if ls >= ss else -1
        atr = float(ind["at"][bi])
        sl_d = atr * 2.5
        if sl_d <= 0:
            continue
        eb = i + 1
        if eb >= n - 1:
            continue
        ep = c[i] + (sp / 2) * d
        conv = 1.5 if sq >= 80 else 1.2 if sq >= 65 else 1.0
        risk = 0.8 * conv
        dr = eq * (risk / 100)
        xp, xb, xr, pr = simulate_trail(ep, sl_d, d, h2, l2, c, eb + 1, n, sp, trail, 0.2, 0.5)
        xp -= (sp / 2) * d
        pnl_r = ((xp - ep) * d) / sl_d if sl_d > 0 else 0
        pnl_d = dr * pnl_r
        eq += pnl_d
        pk_eq = max(pk_eq, eq)
        trades.append({"pnl": pnl_d, "r": pnl_r, "pk": pr, "d": d,
                        "gb": pr - pnl_r if pr > 0 else 0})
        if pnl_d < 0:
            cl += 1
            if cl >= 4:
                cd = xb + 12
                cl = 0
        else:
            cl = 0
        if pk_eq > 0 and (pk_eq - eq) / pk_eq * 100 >= 8:
            break

    if not trades:
        continue

    w = [t for t in trades if t["pnl"] > 0]
    gw = sum(t["pnl"] for t in w)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gw / gl if gl > 0 else 999
    wr = len(w) / len(trades) * 100
    ar = np.mean([t["r"] for t in trades])
    ap = np.mean([t["pk"] for t in trades])
    gb_list = [t["gb"] for t in trades if t["pk"] > 0.3]
    ag = np.mean(gb_list) if gb_list else 0
    tp = sum(t["pnl"] for t in trades)

    eqc = [1000.0]
    for t in trades:
        eqc.append(eqc[-1] + t["pnl"])
    p2 = eqc[0]
    mdd = 0
    for e in eqc:
        p2 = max(p2, e)
        mdd = max(mdd, (p2 - e) / p2 * 100 if p2 > 0 else 0)

    lp = sum(t["pnl"] for t in trades if t["d"] == 1)
    sp2 = sum(t["pnl"] for t in trades if t["d"] == -1)
    dn = "L" if lp > sp2 * 2 else "S" if sp2 > lp * 2 else "B"
    tag = " **" if pf >= 1.5 and len(trades) >= 15 and mdd < 15 else ""

    print(f"{sym:14s} {cat:9s} {len(trades):>4d} {wr:>4.0f}% {pf:>6.2f} ${tp:>8.2f} {mdd:>4.1f}% {ar:>+4.2f} {ap:>3.1f}R {ag:>3.1f}R {dn}{tag}")
    results.append({"sym": sym, "cat": cat, "n": len(trades), "pf": pf,
                     "wr": wr, "pnl": tp, "dd": mdd, "dir": dn})

print()
print("** = DEPLOYABLE (PF>=1.5, n>=15, DD<15%)")
good = sorted([r for r in results if r["pf"] >= 1.5 and r["n"] >= 15 and r["dd"] < 15],
               key=lambda x: -x["pf"])
bad = sorted([r for r in results if not (r["pf"] >= 1.5 and r["n"] >= 15 and r["dd"] < 15)],
              key=lambda x: -x["pf"])

print(f"\nDEPLOYABLE ({len(good)}):")
total = 0
for r in good:
    total += r["pnl"]
    print(f"  {r['sym']:14s} PF={r['pf']:>5.2f} n={r['n']:>3d} WR={r['wr']:>4.0f}% "
          f"DD={r['dd']:>4.1f}% PnL=${r['pnl']:>8.2f} dir={r['dir']}")
print(f"  TOTAL PnL: ${total:.2f}")

print(f"\nREJECTED ({len(bad)}):")
for r in bad:
    print(f"  {r['sym']:14s} PF={r['pf']:>5.2f} n={r['n']:>3d} WR={r['wr']:>4.0f}% "
          f"DD={r['dd']:>4.1f}% PnL=${r['pnl']:>8.2f}")
