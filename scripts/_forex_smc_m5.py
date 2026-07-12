#!/usr/bin/env python3 -B
"""SMC strategy backtest on the INTENDED entry TF: M5 entry + M15 bias (2026-07-12).
Reuses the exact strategy logic (sweep->BOS->FVG->EMA/VWAP/MACD/RSI->engulfing,
TP1 1.5R/TP2 2R, EMA9 trail) from _forex_smc_backtest.py — only the data source
changes to M5/M15. This is the strategy on (near) its designed scalping timeframe.
Usage: python3 -B scripts/_forex_smc_m5.py SYMBOL
"""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

_P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("smc", str(_P / "_forex_smc_backtest.py"))
smc = importlib.util.module_from_spec(spec); spec.loader.exec_module(smc)
C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")


def build_m5(sym):
    fn = C / ("raw_m5_" + sym.replace(".", "_") + ".pkl")
    if not fn.exists():
        return None
    h1 = pickle.load(open(fn, "rb"))              # M5 entry bars
    h1["time"] = smc._naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    c, h, l, o = h1["close"], h1["high"], h1["low"], h1["open"]
    vol = h1["tick_volume"] if "tick_volume" in h1 else pd.Series(1.0, index=h1.index)
    h1["ema9"], h1["ema21"] = smc._ema(c, 9), smc._ema(c, 21)
    h1["rsi"] = smc._rsi(c); h1["macd"], h1["sig"] = smc._macd(c)
    h1["atr"] = smc._atr(h, l, c)
    h1["body"] = (c - o).abs(); h1["avgbody"] = h1["body"].rolling(20).mean()
    day = h1["time"].dt.date; tp = (h + l + c) / 3
    h1["_cumtpv"] = (tp * vol).groupby(day).cumsum()
    h1["_cumv"] = vol.groupby(day).cumsum().replace(0, np.nan)
    h1["vwap"] = (h1["_cumtpv"] / h1["_cumv"]).ffill()
    S = smc.SWING
    h1["prior_hi"] = h.rolling(S*2+1).max().shift(1)
    h1["prior_lo"] = l.rolling(S*2+1).min().shift(1)
    h1["sweep_lo"] = l.rolling(smc.SWEEP_LB).min().shift(1)
    h1["sweep_hi"] = h.rolling(smc.SWEEP_LB).max().shift(1)
    # M15 bias
    mfn = C / ("raw_m15_" + sym.replace(".", "_") + ".pkl")
    if mfn.exists():
        d = pickle.load(open(mfn, "rb"))
        d["time"] = smc._naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
        d["bias"] = np.where(d["close"] > smc._ema(d["close"], 50), 1, -1)
        d["eff"] = d["time"].astype("datetime64[ns]")
        h1["time"] = h1["time"].astype("datetime64[ns]")
        h1 = pd.merge_asof(h1, d[["eff", "bias"]], left_on="time", right_on="eff", direction="backward")
    else:
        h1["bias"] = np.where(c > smc._ema(c, 200), 1, -1)
    return h1.dropna(subset=["ema21", "atr", "avgbody", "prior_hi", "sweep_lo"]).reset_index(drop=True)


def main():
    sym = sys.argv[1]
    m = build_m5(sym)
    if m is None or len(m) < 5000:
        print(json.dumps({"symbol": sym, "error": "no/short M5 data"})); return
    px = float(m["close"].iloc[-1]); is_jpy = "JPY" in sym or px > 20
    point = 0.001 if is_jpy else 0.00001
    med = float(np.nanmedian(m["spread"].values)) if "spread" in m else 15
    spread = min(max(2*med*point/px, 0.00008), 0.0004)
    yrs = (m["time"].iloc[-1]-m["time"].iloc[0]).days/365.25
    sp = int(len(m)*0.7)
    full = smc.stats(smc.simulate(m, spread), yrs)
    isr = smc.stats(smc.simulate(m.iloc[:sp].reset_index(drop=True), spread), yrs*0.7)
    oos = smc.stats(smc.simulate(m.iloc[sp:].reset_index(drop=True), spread), yrs*0.3)
    print(json.dumps({"symbol": sym, "bars": len(m), "yrs": round(yrs, 2), "spread": round(spread, 5),
                      "full": full, "is": isr, "oos": oos}, default=float))


if __name__ == "__main__":
    main()
