#!/usr/bin/env python3 -B
"""CRYPTO TSMOM-LF — long/flat daily trend on BTCUSD/ETHUSD (2026-07-08).

Time-series momentum, LONG-ONLY (crypto short side is structurally weak).
D1 closed bars: enter when close>SMA200 AND close>close[90] AND close makes a
new 20-day Donchian(close) high. Broker SL = entry - 3.0xATR(20). No TP.
Exit = 3.5xATR Chandelier trail from highest CLOSE since entry (up-only),
OR D1 close < SMA200 (close next open).

Pure functions; identical math to scripts/_crypto_tsmom_backtest.py so live
matches the validated test. Mirrors agent/trend_follower.py style.

Sizing is fixed min-lot (0.01) on both symbols. BTC additionally carries a
vol gate — implied min-lot SL risk (SL_ATR_MULT x ATR x $/pt x min_lot) must
be <= CT_BTC_MAX_RISK_PCT of equity, else SKIP the entry (ETH keeps trading).
Swap drag is real (~ -25%/yr BTC, -38%/yr ETH, interest-on-price mode) — the
trail exists to cap holding time in chop; do not widen it casually.
"""
import numpy as np
import pandas as pd


def _atr(high, low, close, period):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def evaluate(d1, params):
    """d1: Daily OHLC DataFrame (open/high/low/close [+time]) ascending,
    LAST ROW = most recent CLOSED D1 bar (caller must drop the forming bar).

    Returns None if insufficient data, else a state dict:
      entry        bool  — all entry conditions true on this closed bar
      close, atr   float — last closed close / ATR(20)
      sl           float — initial broker SL if entering (close - 3.0xATR)
      sma200       float — trend filter level (exit_flat when close < sma200)
      exit_flat    bool  — close < SMA200 → close any open position next open
      don_high     float — Donchian(20) highest close incl. current bar
    """
    sma_p = int(params.get("SMA_PERIOD", 200))
    roc_p = int(params.get("ROC_PERIOD", 90))
    don_p = int(params.get("DON_PERIOD", 20))
    atr_p = int(params.get("ATR_PERIOD", 20))
    sl_mult = float(params.get("SL_ATR_MULT", 3.0))
    if d1 is None or len(d1) < max(sma_p, roc_p, don_p, atr_p) + 5:
        return None
    close, high, low = d1["close"], d1["high"], d1["low"]
    c = float(close.iloc[-1])
    sma = float(close.rolling(sma_p).mean().iloc[-1])
    roc_ref = float(close.iloc[-1 - roc_p])
    don_high = float(close.iloc[-don_p:].max())
    atr = float(_atr(high, low, close, atr_p).iloc[-1])
    if not (np.isfinite(atr) and atr > 0 and np.isfinite(sma)):
        return None
    entry = bool(c > sma and c > roc_ref and c >= don_high)
    return {
        "entry": entry,
        "close": c,
        "atr": atr,
        "sl": c - sl_mult * atr,
        "sma200": sma,
        "exit_flat": bool(c < sma),
        "don_high": don_high,
    }


def chandelier_stop(highest_close, atr, prev_stop, params):
    """Up-only Chandelier trail. highest_close = max D1 close since entry,
    atr = current ATR(20), prev_stop = last broker SL (or the initial SL).
    Returns the new stop — never lower than prev_stop."""
    mult = float(params.get("TRAIL_ATR_MULT", 3.5))
    lvl = highest_close - mult * atr
    if prev_stop is None:
        return lvl
    return max(float(prev_stop), lvl)


def btc_vol_gate_ok(atr, dollars_per_point_min_lot, equity, params):
    """True if implied min-lot SL risk <= max % of equity (BTC only)."""
    sl_mult = float(params.get("SL_ATR_MULT", 3.0))
    max_pct = float(params.get("BTC_MAX_RISK_PCT", 3.0))
    risk = sl_mult * atr * dollars_per_point_min_lot
    return risk <= equity * max_pct / 100.0
