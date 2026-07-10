#!/usr/bin/env python3 -B
"""TREND-FOLLOWER (7th strategy) — daily time-series trend signal + ATR.

3-speed EMA-crossover ensemble on Daily bars → net direction. Pure function;
identical logic to scripts/_trend_backtest.py so live matches the validated test.
The brain rebalances once/day: flip/close on signal change, wide 3xATR tail stop.
"""
import numpy as np
import pandas as pd


def _atr(high, low, close, period):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def conviction(d1, atr_period=20):
    """Daily trend-strength: (ADX(14), slope) where slope = |EMA256 - EMA256[-10]|
    / ATR. Used by the per-symbol selectivity gate to skip weak/choppy trends."""
    if d1 is None or len(d1) < 260:
        return 0.0, 0.0
    h, l, c = d1["high"], d1["low"], d1["close"]
    # ADX(14)
    up = h.diff(); dn = -l.diff()
    plus = ((up > dn) & (up > 0)) * up
    minus = ((dn > up) & (dn > 0)) * dn
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    a14 = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1.0 / 14, adjust=False).mean() / a14.replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1.0 / 14, adjust=False).mean() / a14.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = float(dx.ewm(alpha=1.0 / 14, adjust=False).mean().iloc[-1])
    slow = c.ewm(span=256).mean()
    atr = float(_atr(h, l, c, atr_period).iloc[-1])
    slope = abs(float(slow.iloc[-1] - slow.iloc[-11])) / atr if atr > 0 else 0.0
    if not np.isfinite(adx):
        adx = 0.0
    return adx, slope


def chandelier_stop(d1, direction, params):
    """Chandelier trailing stop from Daily bars (protect an open trend position):
      long  -> highest-high(lookback) - mult*ATR
      short -> lowest-low(lookback)   + mult*ATR
    Returns the stop price (float) or None. Same _atr as the entry signal so the
    trail is consistent with the sizing stop."""
    atr_p = int(params.get("ATR_PERIOD", 20))
    lb = int(params.get("TRAIL_LOOKBACK", 22))
    mult = float(params.get("TRAIL_ATR", 2.5))
    if d1 is None or len(d1) < max(atr_p, lb) + 2 or direction not in (1, -1):
        return None
    high, low, close = d1["high"], d1["low"], d1["close"]
    atr = float(_atr(high, low, close, atr_p).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None
    if direction == 1:
        return float(high.iloc[-lb:].max()) - mult * atr
    return float(low.iloc[-lb:].min()) + mult * atr


def evaluate(d1, params):
    """d1: Daily OHLC DataFrame (open/high/low/close [+time]) ascending.
    Returns {direction(+1/-1/0), signal, atr, close} or None."""
    pairs = params.get("EMA_PAIRS", [(16, 64), (32, 128), (64, 256)])
    min_abs = float(params.get("MIN_ABS_SIGNAL", 0.34))
    atr_p = int(params.get("ATR_PERIOD", 20))
    if d1 is None or len(d1) < 260:
        return None
    close, high, low = d1["close"], d1["high"], d1["low"]
    sig = 0.0
    for f, s in pairs:
        sig += float(np.sign(close.ewm(span=f).mean().iloc[-1]
                             - close.ewm(span=s).mean().iloc[-1]))
    sig /= len(pairs)
    direction = 0 if abs(sig) < min_abs else (1 if sig > 0 else -1)
    atr = float(_atr(high, low, close, atr_p).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None
    return {"direction": direction, "signal": float(sig),
            "atr": atr, "close": float(close.iloc[-1])}
