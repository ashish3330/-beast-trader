#!/usr/bin/env python3 -B
"""BTCMR — BTC-specific MEAN-REVERSION strategy (M15).

Rationale: the SMA8/20 breakout (SMABO) works on BTC only in trends; in the
recent chop regime BTC whipsaws and the breakout bleeds. BTC spends most of its
time RANGING between H4 levels, so a mean-reversion fade of stretched moves fits
its actual behaviour. Independent magic range so it never collides with SMABO.

Signal (M15, on the last CLOSED bar):
  Regime : ADX(14) < ADX_MAX  (only fade when NOT strongly trending)
  LONG   : close < lower Bollinger(BB_PERIOD, BB_MULT) AND RSI(RSI_PERIOD) < RSI_LOW
  SHORT  : close > upper Bollinger              AND RSI > RSI_HIGH
  SL     : entry -/+ SL_ATR * ATR14  (beyond the stretch)
  TP1    : middle band (SMA of BB_PERIOD) = the mean  (reversion target)
  TP2    : opposite band (full reversion)
  Exit   : also a time-stop after TIME_STOP_BARS bars if neither TP nor SL hit.

Params are env/dict overridable (BTCMR_<KEY>) for tuners. Pure-function eval so
the backtest and the live brain share identical logic.
"""
import os

import numpy as np

# Reuse validated indicator helpers from the breakout module.
from agent.sma_breakout import _sma, _atr, _adx

DEFAULTS = {
    "BB_PERIOD": 20, "BB_MULT": 2.2,
    "RSI_PERIOD": 2, "RSI_LOW": 10.0, "RSI_HIGH": 90.0,
    "ADX_MAX": 25.0,
    "SL_ATR": 1.5,
    "TIME_STOP_BARS": 16,
    "MIN_RR": 0.0,          # 0 = accept MR's natural (often <1) R:R
    "CONFIRM": 0.0,         # 1 = require close back INSIDE band (reversal confirmed)
}


def _param(key, override):
    ev = os.getenv(f"BTCMR_{key}")
    if ev is not None:
        try:
            return float(ev)
        except Exception:
            pass
    if override and key in override:
        return float(override[key])
    return float(DEFAULTS[key])


def _rsi(close, period):
    """Wilder RSI — latest value. period=2 = Connors-style fast oscillator."""
    n = len(close)
    if n < period + 1:
        return 50.0
    d = np.diff(close)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag = gain[:period].mean()
    al = loss[:period].mean()
    k = 1.0 / period
    for i in range(period, len(d)):
        ag = ag * (1 - k) + gain[i] * k
        al = al * (1 - k) + loss[i] * k
    if al == 0:
        return 100.0
    rs = ag / al
    return float(100.0 - 100.0 / (1.0 + rs))


def evaluate(m15, i, override=None):
    """Inspect closed bar index i of an OHLC DataFrame. Return sig dict or None.
    m15 must have columns open/high/low/close and be time-sorted ascending."""
    bb_p = int(_param("BB_PERIOD", override))
    bb_m = _param("BB_MULT", override)
    rsi_p = int(_param("RSI_PERIOD", override))
    rsi_lo = _param("RSI_LOW", override)
    rsi_hi = _param("RSI_HIGH", override)
    adx_max = _param("ADX_MAX", override)
    sl_atr = _param("SL_ATR", override)

    if i < max(bb_p, 30) + 2:
        return None
    C = m15["close"].values[:i + 1]
    H = m15["high"].values[:i + 1]
    L = m15["low"].values[:i + 1]
    close = float(C[-1])

    mid = _sma(C, bb_p)
    m = float(mid[-1])
    if not np.isfinite(m):
        return None
    sd = float(np.std(C[-bb_p:], ddof=0))
    upper = m + bb_m * sd
    lower = m - bb_m * sd

    atr14 = _atr(H, L, C, 14)
    if atr14 <= 0:
        return None
    adx14 = _adx(H, L, C, 14)
    if adx14 >= adx_max:          # too trendy to fade
        return None
    rsi = _rsi(C, rsi_p)
    confirm = _param("CONFIRM", override) >= 1.0

    direction = None
    if confirm:
        # Reversal-confirmed: prior bar closed OUTSIDE the band, this bar closes
        # back INSIDE it — the fade only fires once price turns, not on the touch.
        prev = float(C[-2])
        m_prev = float(mid[-2])
        sd_prev = float(np.std(C[-bb_p - 1:-1], ddof=0))
        lower_prev = m_prev - bb_m * sd_prev
        upper_prev = m_prev + bb_m * sd_prev
        if prev < lower_prev and close > lower and rsi < rsi_lo:
            direction = "LONG"
        elif prev > upper_prev and close < upper and rsi > rsi_hi:
            direction = "SHORT"
    else:
        if close < lower and rsi < rsi_lo:
            direction = "LONG"
        elif close > upper and rsi > rsi_hi:
            direction = "SHORT"
    if direction is None:
        return None

    entry = close
    if direction == "LONG":
        sl = entry - sl_atr * atr14
        tp1 = m                       # revert to mean
        tp2 = upper                   # full reversion
        if sl >= entry or tp1 <= entry:
            return None
    else:
        sl = entry + sl_atr * atr14
        tp1 = m
        tp2 = lower
        if sl <= entry or tp1 >= entry:
            return None

    return {
        "direction": direction,
        "entry": float(entry), "sl": float(sl),
        "tp1": float(tp1), "tp2": float(tp2),
        "atr14": float(atr14), "adx14": float(adx14), "rsi": float(rsi),
        "mid": float(m), "upper": float(upper), "lower": float(lower),
        "reason": (f"{direction} MR close={close:.2f} "
                   f"{'<L' if direction=='LONG' else '>U'} band, RSI{rsi_p}={rsi:.1f}, ADX={adx14:.1f}"),
    }
