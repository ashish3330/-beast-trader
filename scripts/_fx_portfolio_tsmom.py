#!/usr/bin/env python3 -B
"""FX PORTFOLIO time-series momentum (AQR/Moskowitz) — the canonical FX edge:
single pairs are weak, but a DIVERSIFIED vol-targeted basket can have real
portfolio Sharpe. Resample H1->daily for all FX pairs, per-pair TSMOM signal,
vol-scale to a target, sum across the basket. Report portfolio Sharpe IS vs OOS
(the honest scale-invariant metric). Costs charged per position flip.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP",
         "EURJPY","GBPJPY","AUDJPY","EURCHF","EURAUD","EURCAD","EURNZD","GBPCHF",
         "GBPAUD","CADJPY","CHFJPY","NZDJPY"]
TARGET_VOL = 0.10        # per-pair annualized vol target
COST = 0.00015           # ~1.5bp round-trip per flip


def _naive(s):
    s = pd.to_datetime(s)
    try: return s.dt.tz_localize(None)
    except (TypeError, AttributeError): return s


def daily_close(sym):
    p = C / ("raw_h1_" + sym + ".pkl")
    if not p.exists(): return None
    d = pickle.load(open(p, "rb")); d["time"] = _naive(d["time"])
    d = d.set_index("time").sort_index()
    return d["close"].resample("1D").last().dropna()


def signal(close):
    sig = pd.Series(0.0, index=close.index)
    for f, s in [(16, 64), (32, 128), (64, 256)]:
        sig = sig + np.sign(close.ewm(span=f).mean() - close.ewm(span=s).mean())
    return (sig / 3.0)


def pair_returns(sym):
    close = daily_close(sym)
    if close is None or len(close) < 300: return None
    ret = np.log(close).diff()
    vol = ret.ewm(com=33).std() * np.sqrt(252)
    sig = signal(close)
    pos = (sig.shift(1) * TARGET_VOL / vol.shift(1)).clip(-3, 3)   # vol-scaled, prior-day
    flips = pos.diff().abs().fillna(0)
    strat = pos * ret - flips * COST
    return strat.dropna()


def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 20 and r.std() > 0 else 0.0


def main():
    streams = {}
    for s in PAIRS:
        r = pair_returns(s)
        if r is not None: streams[s] = r
    if not streams:
        print("no data"); return
    port = pd.concat(streams.values(), axis=1).mean(axis=1).dropna()   # equal-weight basket
    n = len(port); split = int(n * 0.70)
    port_is, port_oos = port.iloc[:split], port.iloc[split:]
    print(f"FX PORTFOLIO TSMOM — {len(streams)} pairs, {n} days")
    print(f"  Full   Sharpe: {sharpe(port):.2f}")
    print(f"  IS     Sharpe: {sharpe(port_is):.2f}  ({port_is.index[0].date()}..{port_is.index[-1].date()})")
    print(f"  OOS    Sharpe: {sharpe(port_oos):.2f}  ({port_oos.index[0].date()}..{port_oos.index[-1].date()})")
    print(f"  OOS ann.return: {port_oos.mean()*252*100:.1f}%   OOS ann.vol: {port_oos.std()*np.sqrt(252)*100:.1f}%")
    # per-pair OOS Sharpe (diversification check)
    print("  per-pair OOS Sharpe:")
    ps = {s: sharpe(r.iloc[int(len(r)*0.70):]) for s, r in streams.items()}
    for s, v in sorted(ps.items(), key=lambda x: -x[1]):
        print(f"    {s:8s} {v:+.2f}")


if __name__ == "__main__":
    main()
