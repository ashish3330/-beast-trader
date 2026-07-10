#!/usr/bin/env python3 -B
"""GOLD_H4 — D1-regime / H4-trigger long-only Donchian breakout on XAUUSD.

Pure functions; identical math to scripts/_gold_h4_backtest.py so live matches
the validated test. Mirrors agent/trend_follower.py style.

REGIME (D1):  long-only allowed when D1 close > SMA200 AND SMA50 > SMA200.
ENTRY (H4):   close breaks above the highest H4 close of the prior 20 bars
              (Donchian-20 on closes). Initial SL = entry - 2.0 x ATR(14, H4).
EXIT:         chandelier trail = highest H4 close since entry - 3.0 x ATR(14,H4)
              (ratchet up only), plus HARD exit when D1 close < SMA200.
SKIP-GATE:    at min lot (0.01 = $1 per $1.00 gold move) a 2xATR stop can be
              $45-70 = 1.9-3.0% of a ~$2.3K account. If stop-risk at min lot
              exceeds MAX_RISK_USD the signal is SKIPPED (size by skipping,
              never by tightening the stop). Max 1 gold position (brain-side).

Params are FROZEN (Donchian 20 / 2xATR SL / 3xATR trail) — no per-session or
per-regime variants; that surface has a history of overfit in this repo.
"""
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "DONCHIAN_N": 20,          # lookback of prior H4 closes for breakout
    "ATR_PERIOD": 14,          # H4 ATR (Wilder)
    "SL_ATR_MULT": 2.0,        # initial stop distance
    "TRAIL_ATR_MULT": 3.0,     # chandelier trail distance
    "SMA_FAST_D1": 50,
    "SMA_SLOW_D1": 200,
    "MAX_RISK_USD": 70.0,      # 3.0% of $2355 — SKIP signal if min-lot risk exceeds this
    "USD_PER_UNIT_MIN_LOT": 1.0,  # XAUUSD @0.01 lot (1 oz): $1 per $1.00 price move
}


def _p(params, key):
    return params.get(key, DEFAULT_PARAMS[key]) if params else DEFAULT_PARAMS[key]


def _atr(high, low, close, period):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _d1_from(h4, d1):
    """Daily close series: prefer the real D1 feed, else resample H4."""
    if d1 is not None and len(d1) > 0:
        c = d1["close"]
        if "time" in d1.columns:
            c = pd.Series(d1["close"].values,
                          index=pd.to_datetime(d1["time"]).dt.normalize())
        return c
    t = pd.to_datetime(h4["time"]) if "time" in h4.columns else h4.index
    return pd.Series(h4["close"].values, index=t).resample("1D").last().dropna()


def regime_ok(h4, params=None, d1=None):
    """True when D1 close > SMA200 AND SMA50 > SMA200 (uses last COMPLETED
    daily close — caller must pass data without a partial daily bar)."""
    dc = _d1_from(h4, d1)
    slow_n = int(_p(params, "SMA_SLOW_D1"))
    if len(dc) < slow_n + 5:
        return False
    fast = dc.rolling(int(_p(params, "SMA_FAST_D1"))).mean().iloc[-1]
    slow = dc.rolling(slow_n).mean().iloc[-1]
    return bool(np.isfinite(fast) and np.isfinite(slow)
                and dc.iloc[-1] > slow and fast > slow)


def hard_exit(h4, params=None, d1=None):
    """HARD exit for an open long: D1 close < SMA200 (fast/slow cross NOT
    required to exit — only the close<SMA200 leg)."""
    dc = _d1_from(h4, d1)
    slow_n = int(_p(params, "SMA_SLOW_D1"))
    if len(dc) < slow_n + 5:
        return False
    slow = dc.rolling(slow_n).mean().iloc[-1]
    return bool(np.isfinite(slow) and dc.iloc[-1] < slow)


def trail_stop(highest_close_since_entry, atr, params=None):
    """Chandelier stop level. Caller ratchets: new_sl = max(old_sl, this)."""
    return float(highest_close_since_entry) - float(_p(params, "TRAIL_ATR_MULT")) * float(atr)


def evaluate(h4, params=None, d1=None):
    """h4: H4 OHLC DataFrame (time/open/high/low/close) ascending, iloc[-1] =
    latest COMPLETED H4 bar. d1: optional real D1 frame (else resampled).
    Returns a BUY signal dict or None (None also when skip-gate trips)."""
    n = int(_p(params, "DONCHIAN_N"))
    atr_p = int(_p(params, "ATR_PERIOD"))
    if h4 is None or len(h4) < max(n + 2, atr_p + 2):
        return None
    close, high, low = h4["close"], h4["high"], h4["low"]

    if not regime_ok(h4, params, d1):
        return None

    donchian = float(close.iloc[-(n + 1):-1].max())      # prior N closes, excl current
    c = float(close.iloc[-1])
    if not (c > donchian):
        return None

    atr = float(_atr(high, low, close, atr_p).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None

    sl_dist = float(_p(params, "SL_ATR_MULT")) * atr
    risk_usd = sl_dist * float(_p(params, "USD_PER_UNIT_MIN_LOT"))
    if risk_usd > float(_p(params, "MAX_RISK_USD")):
        return None                                       # SKIP — never tighten

    return {
        "action": "BUY",
        "entry": c,
        "sl": c - sl_dist,
        "atr": atr,
        "donchian": donchian,
        "risk_usd_min_lot": risk_usd,
        "trail_atr_mult": float(_p(params, "TRAIL_ATR_MULT")),
    }
