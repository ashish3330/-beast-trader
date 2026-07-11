#!/usr/bin/env python3 -B
"""HYBRID SMC + MOMENTUM/BREAKOUT strategy — backtest on all FX pairs (2026-07-12).

Faithful implementation of the user's spec, on H1 entry TF + D1 bias (the broker
gives no 1m/5m FX data — this is the lower-frequency proxy of the 1m/5m scalp).

LONG (all must be true):
  1. HTF bias bullish: D1 close > D1 EMA50 (structural up-bias)
  2. Liquidity sweep: bar low < prior N-bar low (took liquidity) AND closes back above it
  3. BOS: close breaks the prior swing high
  4. FVG/OB: a bullish FVG exists (gap: low[t] > high[t-2]) near entry
  5. Indicators: EMA9>EMA21 both sloping up; price > VWAP(daily); (MACD>signal OR RSI>50)
  6. Candle: bullish engulfing OR body > 1.2x avg body, closing above the swept low
  Entry at bar close. SL = swept-low - buffer. TP1 1.5R, TP2 2.0R (50/50), EMA9 trail on runner.
SHORT = mirror.

Costs: round-trip spread (per-pair from data) + slippage. Walk-forward IS/OOS.
Usage: python3 -B scripts/_forex_smc_backtest.py SYMBOL
"""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
SWING = 5            # fractal lookback for swing hi/lo
SWEEP_LB = 10        # liquidity lookback
SLIP_FRAC = 0.00005  # FX slippage per fill
TP1_R, TP2_R = 1.5, 2.0


def _naive(s):
    s = pd.to_datetime(s)
    try: return s.dt.tz_localize(None)
    except (TypeError, AttributeError): return s


def _ema(x, n): return x.ewm(span=n, adjust=False).mean()
def _rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + up/dn.replace(0, np.nan))).fillna(50)
def _macd(c):
    m = _ema(c, 12) - _ema(c, 26); return m, _ema(m, 9)
def _atr(h, l, c, n=14):
    p = c.shift(1); tr = pd.concat([(h-l), (h-p).abs(), (l-p).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def build(sym):
    h1 = pickle.load(open(C/("raw_h1_"+sym.replace(".", "_")+".pkl"), "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    c, h, l, o = h1["close"], h1["high"], h1["low"], h1["open"]
    vol = h1["tick_volume"] if "tick_volume" in h1 else pd.Series(1.0, index=h1.index)
    h1["ema9"], h1["ema21"] = _ema(c, 9), _ema(c, 21)
    h1["rsi"] = _rsi(c); h1["macd"], h1["sig"] = _macd(c)
    h1["atr"] = _atr(h, l, c)
    h1["body"] = (c - o).abs(); h1["avgbody"] = h1["body"].rolling(20).mean()
    # daily-reset VWAP
    day = h1["time"].dt.date
    tp = (h + l + c) / 3
    h1["_cumtpv"] = (tp * vol).groupby(day).cumsum()
    h1["_cumv"] = vol.groupby(day).cumsum().replace(0, np.nan)
    h1["vwap"] = (h1["_cumtpv"] / h1["_cumv"]).ffill()
    # swing hi/lo (fractal)
    h1["swing_hi"] = h[(h.shift(SWING) < h) & (h.shift(-0) <= h)].rolling(1).max()
    h1["prior_hi"] = h.rolling(SWING*2+1, center=False).max().shift(1)
    h1["prior_lo"] = l.rolling(SWING*2+1, center=False).min().shift(1)
    h1["sweep_lo"] = l.rolling(SWEEP_LB).min().shift(1)
    h1["sweep_hi"] = h.rolling(SWEEP_LB).max().shift(1)
    # D1 bias
    d = pickle.load(open(C/("raw_d1_"+sym.replace(".", "_")+".pkl"), "rb")) if (C/("raw_d1_"+sym.replace(".", "_")+".pkl")).exists() else None
    if d is not None:
        d["time"] = _naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
        d["bias"] = np.where(d["close"] > _ema(d["close"], 50), 1, -1)
        d["eff"] = (d["time"].dt.normalize()+pd.Timedelta(days=1)).astype("datetime64[ns]")
        h1["time"] = h1["time"].astype("datetime64[ns]")
        h1 = pd.merge_asof(h1, d[["eff", "bias"]], left_on="time", right_on="eff", direction="backward")
    else:
        # fallback: H1 EMA200 bias
        h1["bias"] = np.where(c > _ema(c, 200), 1, -1)
    return h1.dropna(subset=["ema21", "atr", "avgbody", "prior_hi", "sweep_lo"]).reset_index(drop=True)


def simulate(m, spread):
    o, h, l, c = [m[x].values for x in ("open", "high", "low", "close")]
    ema9, ema21 = m["ema9"].values, m["ema21"].values
    rsi, macd, sig = m["rsi"].values, m["macd"].values, m["sig"].values
    vwap, atr, bias = m["vwap"].values, m["atr"].values, m["bias"].values
    body, avgbody = m["body"].values, m["avgbody"].values
    prior_hi, prior_lo = m["prior_hi"].values, m["prior_lo"].values
    sweep_lo, sweep_hi = m["sweep_lo"].values, m["sweep_hi"].values
    cost = spread + 2*SLIP_FRAC
    trades = []
    pos = 0; entry = sl = tp1 = tp2 = 0.0; risk0 = 0.0; half = False
    SEQ = 8                                        # bars a sweep stays "active" for entry
    last_bull_sweep = last_bull_sweep_lo = -999
    last_bear_sweep = last_bear_sweep_hi = -999
    for t in range(3, len(m)):
        # ── update liquidity-sweep memory (sweep = took liquidity then reclaimed) ──
        if l[t] < sweep_lo[t] and c[t] > sweep_lo[t]:
            last_bull_sweep = t; last_bull_sweep_lo = l[t]
        if h[t] > sweep_hi[t] and c[t] < sweep_hi[t]:
            last_bear_sweep = t; last_bear_sweep_hi = h[t]
        if pos != 0:
            is_long = pos == 1
            # TP2 / SL / EMA9 trail exit at bar close for runner
            if is_long:
                if l[t] <= sl:
                    trades.append(((sl*(1-SLIP_FRAC)-entry)/entry - (0 if half else spread))/risk0); pos = 0; continue
                if not half and h[t] >= tp1:
                    trades.append((((tp1-entry)/entry - spread)/risk0)*0.5); half = True; sl = entry  # BE after TP1
                if half and h[t] >= tp2:
                    trades.append((((tp2-entry)/entry)/risk0)*0.5); pos = 0; continue
                if c[t] < ema9[t] and half:      # EMA9 trail exit on runner
                    trades.append((((c[t]-entry)/entry)/risk0)*0.5); pos = 0; continue
            else:
                if h[t] >= sl:
                    trades.append(((entry-sl*(1+SLIP_FRAC))/entry - (0 if half else spread))/risk0); pos = 0; continue
                if not half and l[t] <= tp1:
                    trades.append((((entry-tp1)/entry - spread)/risk0)*0.5); half = True; sl = entry
                if half and l[t] <= tp2:
                    trades.append((((entry-tp2)/entry)/risk0)*0.5); pos = 0; continue
                if c[t] > ema9[t] and half:
                    trades.append((((entry-c[t])/entry)/risk0)*0.5); pos = 0; continue
            continue
        # recent bullish/bearish FVG (gap) within last 3 bars = OB/FVG reaction proxy
        bull_fvg = (l[t] > h[t-2]) or (l[t-1] > h[t-3]) if t >= 3 else False
        bear_fvg = (h[t] < l[t-2]) or (h[t-1] < l[t-3]) if t >= 3 else False
        bull_engulf = c[t] > o[t] and c[t-1] < o[t-1] and c[t] > o[t-1] and o[t] < c[t-1]
        bear_engulf = c[t] < o[t] and c[t-1] > o[t-1] and c[t] < o[t-1] and o[t] > c[t-1]
        # ── LONG setup: recent sweep -> BOS -> reaction + confirmation ──
        long_ok = (bias[t] == 1
                   and (t - last_bull_sweep) <= SEQ                         # recent liquidity sweep
                   and c[t] > prior_hi[t]                                    # BOS (break prior high)
                   and bull_fvg                                             # FVG/OB reaction
                   and ema9[t] > ema21[t] and ema9[t] > ema9[t-1] and ema21[t] > ema21[t-1]
                   and c[t] > vwap[t]
                   and (macd[t] > sig[t] or rsi[t] > 50)
                   and (body[t] > 1.2*avgbody[t] or bull_engulf))
        short_ok = (bias[t] == -1
                    and (t - last_bear_sweep) <= SEQ
                    and c[t] < prior_lo[t]
                    and bear_fvg
                    and ema9[t] < ema21[t] and ema9[t] < ema9[t-1] and ema21[t] < ema21[t-1]
                    and c[t] < vwap[t]
                    and (macd[t] < sig[t] or rsi[t] < 50)
                    and (body[t] > 1.2*avgbody[t] or bear_engulf))
        if long_ok:
            entry = c[t]*(1+SLIP_FRAC); sl = min(l[t], last_bull_sweep_lo) - 0.2*atr[t]
            risk0 = (entry - sl)/entry
            if risk0 <= 0: continue
            tp1 = entry + TP1_R*(entry-sl); tp2 = entry + TP2_R*(entry-sl); pos = 1; half = False
        elif short_ok:
            entry = c[t]*(1-SLIP_FRAC); sl = max(h[t], last_bear_sweep_hi) + 0.2*atr[t]
            risk0 = (sl - entry)/entry
            if risk0 <= 0: continue
            tp1 = entry - TP1_R*(sl-entry); tp2 = entry - TP2_R*(sl-entry); pos = -1; half = False
    return trades


def stats(r, yrs):
    if not r: return {"n": 0, "pf": 0.0, "wr": 0.0, "expR": 0.0, "netR": 0.0, "tpy": 0.0}
    r = np.array(r); w = r[r > 0].sum(); ls = -r[r < 0].sum()
    return {"n": len(r), "pf": round(float(w/ls) if ls > 0 else 99.0, 2),
            "wr": round(float((r > 0).mean()), 3), "expR": round(float(r.mean()), 3),
            "netR": round(float(r.sum()), 1), "tpy": round(len(r)/max(yrs, 0.1), 1)}


def main():
    sym = sys.argv[1]
    m = build(sym)
    if len(m) < 5000:
        print(json.dumps({"symbol": sym, "error": "insufficient data"})); return
    px = float(m["close"].iloc[-1])
    is_jpy = "JPY" in sym or px > 20
    point = 0.001 if is_jpy else 0.00001
    med = float(np.nanmedian(m["spread"].values)) if "spread" in m else 15
    spread = min(max(2*med*point/px, 0.00008), 0.0004)
    yrs = (m["time"].iloc[-1]-m["time"].iloc[0]).days/365.25
    sp = int(len(m)*0.7)
    full = stats(simulate(m, spread), yrs)
    isr = stats(simulate(m.iloc[:sp].reset_index(drop=True), spread), yrs*0.7)
    oos = stats(simulate(m.iloc[sp:].reset_index(drop=True), spread), yrs*0.3)
    print(json.dumps({"symbol": sym, "bars": len(m), "spread": round(spread, 5),
                      "full": full, "is": isr, "oos": oos}, default=float))


if __name__ == "__main__":
    main()
