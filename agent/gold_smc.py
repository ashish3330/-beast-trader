#!/usr/bin/env python3 -B
"""GOLD SMC strategy detector (2026-07-12) — the one validated new edge from the
SMC research: Hybrid SMC + Momentum/Breakout on XAUUSD H1 (D1 bias). BASELINE
params (NOT the tuned ones — the tune overfit to the recent gold bull; the
baseline is PF 1.15 and positive across BOTH halves of an 11.7yr history).

Pure function: evaluate(h1_df, params) -> signal dict or None. Identical logic to
scripts/_forex_smc_backtest.py so live == backtest. The brain calls this once per
new H1 bar on the last CLOSED bar (iloc[-2]).

Signal (LONG mirror SHORT) — 4-of-5 confluence + regime gate (2026-07-18):
  MANDATORY: bias bullish (D1 close > D1 EMA50) + recent liquidity sweep (took a
  low, reclaimed) within SEQ bars + EMA9>EMA21 both sloping up.
  Plus >= MIN_CONFL(4) of the 5 OPTIONAL confluences: [BOS (close>prior swing high),
  bullish FVG, close>VWAP, (MACD>signal OR RSI>50), strong/engulfing candle].
  Plus REGIME GATE (orthogonal churn-cutter, keeps the post-2020 edge that drives
  the PF gain while keeping churn <=1.4x vs the old all-5-AND): ADX>=ADX_MIN AND the
  ATR is not in the bottom ATR_PCT_MIN of its trailing ATR_PCT_WIN-bar range.
Returns entry (bar close), SL (swept-low - 0.2*ATR), TP1 (1.5R), TP2 (2R).
"""
import numpy as np
import pandas as pd

DEFAULTS = {"SWING": 5, "SWEEP_LB": 10, "SEQ": 8, "SL_ATR": 0.2,
            "TP1_R": 1.5, "TP2_R": 2.0, "BIAS_EMA": 50,
            # ── 4-of-5 confluence + regime gate (2026-07-18) ──
            # bias + sweep + EMA-cross stay MANDATORY; require >= MIN_CONFL of the
            # 5 optional confluences [BOS, FVG, close-vs-VWAP, MACD/RSI, strong/engulf].
            # Regime gate cuts the extra churn: trade only when ADX>=ADX_MIN AND the
            # ATR is not in the bottom (ATR_PCT_MIN) of its trailing ATR_PCT_WIN window.
            "MIN_CONFL": 4, "ADX_N": 14, "ADX_MIN": 16.0,
            "ATR_PCT_WIN": 500, "ATR_PCT_MIN": 0.30}


def _ema(x, n): return x.ewm(span=n, adjust=False).mean()
def _rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + up/dn.replace(0, np.nan))).fillna(50)
def _macd(c): m = _ema(c, 12) - _ema(c, 26); return m, _ema(m, 9)
def _atr(h, l, c, n=14):
    p = c.shift(1); tr = pd.concat([(h-l), (h-p).abs(), (l-p).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()
def _adx(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    p = c.shift(1); tr = pd.concat([(h-l), (h-p).abs(), (l-p).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1/n, adjust=False).mean() / atr
    mdi = 100 * minus.ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(0)


def evaluate(h1, d1_bias, params=None):
    """h1: XAU H1 OHLC (+tick_volume), ascending, >= ~300 bars. d1_bias: +1/-1
    (D1 close vs D1 EMA50). Returns signal on the last CLOSED bar or None."""
    p = dict(DEFAULTS, **(params or {}))
    if h1 is None or len(h1) < 260:
        return None
    c, h, l, o = h1["close"], h1["high"], h1["low"], h1["open"]
    vol = h1["tick_volume"] if "tick_volume" in h1 else pd.Series(1.0, index=h1.index)
    ema9, ema21 = _ema(c, 9), _ema(c, 21)
    rsi = _rsi(c); macd, sigl = _macd(c); atr = _atr(h, l, c)
    adx = _adx(h, l, c, int(p["ADX_N"]))
    atr_pct = atr.rolling(int(p["ATR_PCT_WIN"]), min_periods=100).rank(pct=True)
    body = (c - o).abs(); avgbody = body.rolling(20).mean()
    # daily-reset VWAP
    day = pd.to_datetime(h1["time"]).dt.date if "time" in h1 else pd.Series(range(len(h1))) // 24
    tp = (h + l + c) / 3
    vwap = ((tp*vol).groupby(day).cumsum() / vol.groupby(day).cumsum().replace(0, np.nan)).ffill()
    S = int(p["SWING"])
    prior_hi = h.rolling(S*2+1).max().shift(1)
    prior_lo = l.rolling(S*2+1).min().shift(1)
    sweep_lo = l.rolling(int(p["SWEEP_LB"])).min().shift(1)
    sweep_hi = h.rolling(int(p["SWEEP_LB"])).max().shift(1)

    n = len(h1)
    T = n - 2                      # evaluate the last CLOSED bar
    if T < 5 or not np.isfinite(atr.iloc[T]) or atr.iloc[T] <= 0:
        return None
    # find the most recent sweep within SEQ bars ending at T
    SEQ = int(p["SEQ"])
    lbs = lbslo = None; lbsh = lbshi = None
    for t in range(max(3, T-SEQ), T+1):
        if l.iloc[t] < sweep_lo.iloc[t] and c.iloc[t] > sweep_lo.iloc[t]:
            lbs, lbslo = t, l.iloc[t]
        if h.iloc[t] > sweep_hi.iloc[t] and c.iloc[t] < sweep_hi.iloc[t]:
            lbsh, lbshi = t, h.iloc[t]

    def g(s, i): return float(s.iloc[i])
    bull_fvg = (g(l, T) > g(h, T-2)) or (g(l, T-1) > g(h, T-3))
    bear_fvg = (g(h, T) < g(l, T-2)) or (g(h, T-1) < g(l, T-3))
    be = g(c, T) > g(o, T) and g(c, T-1) < g(o, T-1) and g(c, T) > g(o, T-1) and g(o, T) < g(c, T-1)
    se = g(c, T) < g(o, T) and g(c, T-1) > g(o, T-1) and g(c, T) < g(o, T-1) and g(o, T) > g(c, T-1)
    a = g(atr, T)
    # ── regime gate (orthogonal churn-cutter): ADX trend strength + ATR not-dead ──
    apct = g(atr_pct, T)
    regime_ok = (g(adx, T) >= p["ADX_MIN"]
                 and (not np.isfinite(apct) or apct >= p["ATR_PCT_MIN"]))
    mc = int(p["MIN_CONFL"])
    # 5 OPTIONAL confluences: BOS, FVG, close-vs-VWAP, MACD/RSI, strong/engulf candle
    long_confl = (int(g(c, T) > g(prior_hi, T)) + int(bull_fvg) + int(g(c, T) > g(vwap, T))
                  + int(g(macd, T) > g(sigl, T) or g(rsi, T) > 50)
                  + int(g(body, T) > 1.2*g(avgbody, T) or be))
    short_confl = (int(g(c, T) < g(prior_lo, T)) + int(bear_fvg) + int(g(c, T) < g(vwap, T))
                   + int(g(macd, T) < g(sigl, T) or g(rsi, T) < 50)
                   + int(g(body, T) > 1.2*g(avgbody, T) or se))
    # MANDATORY: bias + liquidity sweep + EMA9/21 cross both sloping. + >=4-of-5 + regime.
    long_ok = (d1_bias == 1 and lbs is not None
               and g(ema9, T) > g(ema21, T) and g(ema9, T) > g(ema9, T-1) and g(ema21, T) > g(ema21, T-1)
               and regime_ok and long_confl >= mc)
    short_ok = (d1_bias == -1 and lbsh is not None
                and g(ema9, T) < g(ema21, T) and g(ema9, T) < g(ema9, T-1) and g(ema21, T) < g(ema21, T-1)
                and regime_ok and short_confl >= mc)
    entry = g(c, T)
    if long_ok:
        sl = min(g(l, T), lbslo) - p["SL_ATR"] * a
        risk = entry - sl
        if risk <= 0: return None
        return {"direction": 1, "entry": entry, "sl": sl,
                "tp1": entry + p["TP1_R"]*risk, "tp2": entry + p["TP2_R"]*risk,
                "bar_time": str(h1["time"].iloc[T]) if "time" in h1 else T}
    if short_ok:
        sl = max(g(h, T), lbshi) + p["SL_ATR"] * a
        risk = sl - entry
        if risk <= 0: return None
        return {"direction": -1, "entry": entry, "sl": sl,
                "tp1": entry - p["TP1_R"]*risk, "tp2": entry - p["TP2_R"]*risk,
                "bar_time": str(h1["time"].iloc[T]) if "time" in h1 else T}
    return None
