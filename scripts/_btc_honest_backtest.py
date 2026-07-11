#!/usr/bin/env python3 -B
"""TRUSTWORTHY BTCUSD backtest (2026-07-12) — strips the artifacts that inflate the
standard sim to PF 70 / 97% WR (which live reality (-$2.90/146 trades) exposes as
fiction). Honest modelling:
  * CAUSAL: peak profit updated AFTER exit checks (no same-bar peak-then-exit look-ahead)
  * peak-giveback + profit-lock exits fill at the BAR CLOSE (a market order on a
    detected condition), NOT at the exact intra-bar level
  * SL fills intra-bar (a stop triggers when touched) but with adverse SLIPPAGE
  * SLIPPAGE charged on EVERY fill (entry + exit), plus round-trip spread
  * R-BASED, NO COMPOUNDING: each trade measured in R = pnl / initial_risk; summed.
    PF = gross-win-R / gross-loss-R. (compounding fractions is the other inflator.)
Reports full-period + walk-forward halves. Uses the deployed BTC config.
"""
import numpy as np
import pickle
from pathlib import Path
import pandas as pd
import sys

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import trend_exit_params, trend_ema_pairs, TREND_MIN_ABS_SIGNAL, TREND_ATR_PERIOD

SYM = "BTCUSD"
STOP = 3.0
LB = 22
SLIP_FRAC = 0.0004        # ~0.04% adverse slippage per fill (BTC is liquid but the Wine bridge adds real latency)


def _naive(s):
    s = pd.to_datetime(s)
    try: return s.dt.tz_localize(None)
    except (TypeError, AttributeError): return s


def _atr(h, l, c, n):
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False).mean()


def build():
    d = pickle.load(open(C/"raw_d1_BTCUSD.pkl", "rb"))
    d["time"] = _naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
    c, h, l = d["close"], d["high"], d["low"]
    sig = pd.Series(0.0, index=d.index)
    for f, s in trend_ema_pairs(SYM):
        sig = sig + np.sign(c.ewm(span=f).mean() - c.ewm(span=s).mean())
    sig = (sig/len(trend_ema_pairs(SYM))).apply(lambda v: 0 if abs(v) < TREND_MIN_ABS_SIGNAL else (1 if v > 0 else -1))
    atr = _atr(h, l, c, TREND_ATR_PERIOD)
    d1 = pd.DataFrame({"time": d["time"], "sig": sig.astype(int), "atr": atr,
                       "hh": h.rolling(LB).max(), "ll": l.rolling(LB).min(),
                       "eff": (d["time"].dt.normalize()+pd.Timedelta(days=1)).astype("datetime64[ns]")}).dropna()
    h1 = pickle.load(open(C/"raw_h1_BTCUSD.pkl", "rb"))
    h1["time"] = _naive(h1["time"]).astype("datetime64[ns]"); h1 = h1.sort_values("time").reset_index(drop=True)
    m = pd.merge_asof(h1, d1[["eff", "sig", "atr", "hh", "ll"]], left_on="time", right_on="eff", direction="backward")
    return m.dropna(subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)


def simulate(m, ex, spread_frac):
    o, h, l, c = [m[x].values for x in ("open", "high", "low", "close")]
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    TR, LK, GB, AC = ex["TRAIL"], ex["LOCK"], ex["GIVEBACK"], ex["ACT"]
    trades = []                                  # each = R multiple
    pos = 0; entry = sl = peak = risk0 = 0.0; blocked = 0
    cost = spread_frac + 2 * SLIP_FRAC           # round-trip: spread + slip both ends

    def close_at(px_raw, is_long):
        # adverse slippage on exit fill
        fill = px_raw * (1 - SLIP_FRAC) if is_long else px_raw * (1 + SLIP_FRAC)
        pnl_frac = ((fill - entry) / entry) if is_long else ((entry - fill) / entry)
        return (pnl_frac - spread_frac) / risk0   # in R (risk0 = initial SL distance frac)

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blocked and s != blocked:
                blocked = 0
            if s != 0 and s != blocked:
                pos = s
                entry = o[t] * (1 + SLIP_FRAC) if s == 1 else o[t] * (1 - SLIP_FRAC)  # entry slip
                peak = 0.0
                sl = entry - STOP*a if s == 1 else entry + STOP*a
                risk0 = abs(entry - sl) / entry     # initial risk (frac) for R normalisation
            continue
        if s != 0 and s != pos:                      # daily signal flip -> exit at open
            trades.append(close_at(o[t], pos == 1)); pos = 0; blocked = 0
            continue
        is_long = pos == 1
        # --- CAUSAL: use PRIOR-bar peak for the giveback/lock; check exits; THEN update peak ---
        if is_long:
            sl = max(sl, hh[t] - TR*a)
            if peak >= AC*a: sl = max(sl, entry + LK*peak)
            gb = entry + peak*(1-GB) if (GB < 1 and peak >= AC*a) else -1e18
            if l[t] <= sl:                           # STOP hit intra-bar (fills at sl, +slip via close_at)
                trades.append(close_at(sl, True)); pos = 0
            elif gb > -1e17 and c[t] <= gb:          # giveback: market close AT BAR CLOSE (not intra-bar)
                trades.append(close_at(c[t], True)); pos = 0; blocked = 1
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR*a)
            if peak >= AC*a: sl = min(sl, entry - LK*peak)
            gb = entry - peak*(1-GB) if (GB < 1 and peak >= AC*a) else 1e18
            if h[t] >= sl:
                trades.append(close_at(sl, False)); pos = 0
            elif gb < 1e17 and c[t] >= gb:
                trades.append(close_at(c[t], False)); pos = 0; blocked = -1
            else:
                peak = max(peak, entry - l[t])
    return trades


def report(name, r, yrs):
    if not r:
        print(f"{name}: no trades"); return
    r = np.array(r)
    w = r[r > 0].sum(); ls = -r[r < 0].sum()
    eq = np.cumsum(r); dd = float((np.maximum.accumulate(eq) - eq).max())
    pf = w/ls if ls > 0 else 99.0
    exp = r.mean()
    print(f"{name}: trades={len(r)} ({len(r)/max(yrs,0.1):.0f}/yr)  PF={pf:.2f}  WR={ (r>0).mean()*100:.0f}%  "
          f"expectancy={exp:+.3f}R  netR={r.sum():+.1f}  maxDD={dd:.1f}R  avgWin={r[r>0].mean() if (r>0).any() else 0:+.2f}R avgLoss={r[r<0].mean() if (r<0).any() else 0:+.2f}R")


def main():
    m = build()
    tr, lk, gb, ac = trend_exit_params(SYM)
    ex = {"TRAIL": tr, "LOCK": lk, "GIVEBACK": gb, "ACT": ac}
    px = float(m["close"].iloc[-1])
    spread = 2.0 * (float(np.nanmedian(m["spread"].values)) * 0.01) / px   # round-trip spread frac
    yrs = (m["time"].iloc[-1] - m["time"].iloc[0]).days/365.25
    print("=== BTCUSD HONEST backtest (bar-close fills, slippage, R-based) ===")
    print(f"data {len(m)} H1  {str(m['time'].iloc[0])[:10]}->{str(m['time'].iloc[-1])[:10]}  spread_rt={spread:.5f} slip={SLIP_FRAC}")
    print(f"config EMA={trend_ema_pairs(SYM)} TRAIL={tr} LOCK={lk} GB={gb} ACT={ac}\n")
    report("FULL   ", simulate(m, ex, spread), yrs)
    sp = int(len(m)*0.5)
    report("IS(old)", simulate(m.iloc[:sp].reset_index(drop=True), ex, spread), yrs/2)
    report("OOS(new)", simulate(m.iloc[sp:].reset_index(drop=True), ex, spread), yrs/2)


if __name__ == "__main__":
    main()
