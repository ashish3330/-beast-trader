#!/usr/bin/env python3 -B
"""Diversified volatility-targeted TIME-SERIES TREND-FOLLOWING backtest (D1).

Research-spec (Moskowitz-Ooi-Pedersen 2012 / AQR Century-of-Evidence 2017):
 - Signal: ensemble of 3 EMA crossovers (16/64, 32/128, 64/256), combined sign.
 - Vol targeting: each instrument scaled to ~target annualized vol (EWMA vol).
 - Portfolio: sum of vol-scaled positions across the whole basket (diversification
   is the edge). Realistic per-flip cost charged. Metrics scaled to a vol target.
 - Correlation-aware: equities & forex clusters down-weighted so they aren't 4x.

Honest: Sharpe is scale-invariant (the real number); CAGR/DD reported at a chosen
portfolio vol. Out-of-sample = first-half vs second-half + per-instrument.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
BASKET = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "USOUSD",
          "SP500.r", "NAS100.r", "US2000.r", "DJ30.r", "GER40.r", "UK100.r", "JPN225ft",
          "EURUSD", "USDJPY", "USDCAD", "AUDJPY"]
# correlation clusters → weight so each cluster ~1 bet (equities 6, forex 4)
CLUSTER = {"SP500.r": "eq", "NAS100.r": "eq", "US2000.r": "eq", "DJ30.r": "eq",
           "GER40.r": "eq", "UK100.r": "eq", "JPN225ft": "eq",
           "EURUSD": "fx", "USDJPY": "fx", "USDCAD": "fx", "AUDJPY": "fx"}
TARGET_VOL = 0.10        # per-instrument annualized vol target
PORT_VOL = 0.12          # portfolio vol target for CAGR/DD reporting
MAX_LEV = 3.0            # cap per-instrument leverage
COST_PER_FLIP = 0.0015   # ~0.15% round-trip cost when a position reverses (generous)


def load(sym):
    p = C / ("raw_d1_" + sym.replace(".", "_") + ".pkl")
    if not p.exists():
        return None
    df = pickle.load(open(p, "rb"))
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")["close"].sort_index()


def signal(close):
    """3-speed EMA-crossover ensemble → combined sign in [-1,1]."""
    sig = pd.Series(0.0, index=close.index)
    for fast, slow in [(16, 64), (32, 128), (64, 256)]:
        s = np.sign(close.ewm(span=fast).mean() - close.ewm(span=slow).mean())
        sig = sig + s
    return (sig / 3.0)


def instrument_stream(sym):
    close = load(sym)
    if close is None or len(close) < 300:
        return None
    ret = np.log(close).diff()
    vol = ret.ewm(com=33).std() * np.sqrt(252)      # annualized EWMA vol
    sig = signal(close)
    # vol-scaled exposure, using YESTERDAY's signal+vol (no look-ahead)
    exp = (sig.shift(1) * (TARGET_VOL / vol.shift(1)).clip(-MAX_LEV, MAX_LEV))
    cw = 1.0 / (6 if CLUSTER.get(sym) == "eq" else (4 if CLUSTER.get(sym) == "fx" else 1))
    exp = exp * cw
    # cost: charge when exposure sign flips
    flip = (np.sign(exp) != np.sign(exp.shift(1))) & (exp != 0)
    cost = flip.astype(float) * COST_PER_FLIP
    pnl = exp * ret - cost
    return pnl.rename(sym)


def metrics(daily, label):
    daily = daily.dropna()
    if len(daily) < 50 or daily.std() == 0:
        print(f"  {label}: insufficient"); return
    sharpe = daily.mean() / daily.std() * np.sqrt(252)
    scale = PORT_VOL / (daily.std() * np.sqrt(252))     # scale to portfolio vol target
    d = daily * scale
    eq = (1 + d).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(d)) - 1
    dd = (eq / eq.cummax() - 1).min()
    wr = (daily > 0).mean()
    yrs = len(d) / 252
    print(f"  {label:<22} Sharpe={sharpe:+.2f}  CAGR={cagr*100:+5.1f}%  "
          f"maxDD={dd*100:5.1f}%  WR={wr*100:.0f}%  years={yrs:.1f}")


def main():
    all_streams = {}
    for s in BASKET:
        st = instrument_stream(s)
        if st is not None:
            all_streams[s] = st

    TREND_CORE = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "USOUSD",
                  "SP500.r", "NAS100.r", "DJ30.r", "JPN225ft"]
    GOOD5 = ["XAUUSD", "BTCUSD", "ETHUSD", "JPN225ft", "NAS100.r"]
    FITTABLE = ["ETHUSD", "BTCUSD", "JPN225ft"]   # only ones sizable on $2.3k
    for name, basket in [("GOOD-5 (needs \$10k+ account)", GOOD5),
                         ("FITTABLE-3 (safe on \$2.3k: ETH+BTC+JPN)", FITTABLE)]:
        streams = {s: all_streams[s] for s in basket if s in all_streams}
        port = pd.DataFrame(streams).mean(axis=1, skipna=True)
        print(f"=== {name} — {len(streams)} instruments ===")
        metrics(port, "FULL SAMPLE")
        half = len(port) // 2
        metrics(port.iloc[:half], "1st half (OOS-A)")
        metrics(port.iloc[half:], "2nd half (OOS-B)")
        metrics(port.iloc[-1260:], "last ~5 years")
        print()

    print("=== PER-INSTRUMENT trend backtest (standalone, vol-targeted, cost-charged) ===")
    TC = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "USOUSD",
          "SP500.r", "NAS100.r", "DJ30.r", "JPN225ft"]
    for s in TC:
        if s in all_streams:
            metrics(all_streams[s], s)


if __name__ == "__main__":
    main()
