#!/usr/bin/env python3 -B
"""INDICES MEAN-REVERSION (IMR) — long-only RSI(2)+IBS dip-buying above SMA200.

Universe: SP500.r / US2000.r / JPN225ft (D1). One decision per day on the last
CLOSED daily bar (iloc[-2] — the live -1 bar is still forming), fills at next
open. Broker SL = entry - 6.0xATR(14): DISASTER stop only — never tightened,
never trailed. Exits (first wins, executed next open): RSI2>65 OR close>SMA5
OR 7-trading-day time-stop.

Backtest: scripts/_indices_mr_backtest.py (identical math — parity asserted
there). Validated 2026-07-08: basket PF 1.66 / WR 69%, OOS(>=2023) PF 2.30 /
WR 73% / Sharpe 1.74, real costs (spread+slip+swap, live-verified specs).
NOTE: pre-2023 SP500 edge is ~zero after costs (PF 1.02) — see backtest header.

PROHIBITED (research contract): shorts, tightening the stop, removing the
time-stop, intraday evaluation, adding params after losses.
"""
import numpy as np
import pandas as pd

DEFAULTS = {
    "RSI_ENTRY": 15.0,     # RSI(2) oversold entry threshold
    "IBS_ENTRY": 0.30,     # internal-bar-strength entry threshold
    "RSI_EXIT": 65.0,      # RSI(2) exit threshold
    "SMA_TREND": 200,      # regime filter
    "SMA_EXIT": 5,         # mean target
    "ATR_PERIOD": 14,
    "SL_ATR": 6.0,         # disaster stop distance
    "TIME_STOP_DAYS": 7,   # trading days (D1 bars) held, then exit next open
}


def _rsi(close, n):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def _atr(high, low, close, n):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def evaluate(d1, params=None):
    """d1: Daily OHLC(+time) DataFrame ascending. Entry check on the last
    CLOSED bar (iloc[-2]). Returns sig dict or None. LONG only."""
    p = dict(DEFAULTS)
    if params:
        p.update({k: params[k] for k in DEFAULTS if k in params})
    need = int(p["SMA_TREND"]) + 10
    if d1 is None or len(d1) < need:
        return None
    close, high, low = d1["close"], d1["high"], d1["low"]
    i = len(d1) - 2                      # last closed D1 bar

    c_i = float(close.iloc[i])
    c_prev = float(close.iloc[i - 1])
    sma200 = float(close.rolling(int(p["SMA_TREND"])).mean().iloc[i])
    rng = float(high.iloc[i] - low.iloc[i])
    if rng <= 0:
        return None
    ibs = (c_i - float(low.iloc[i])) / rng
    rsi2 = float(_rsi(close, 2).iloc[i])
    atr = float(_atr(high, low, close, int(p["ATR_PERIOD"])).iloc[i])
    if not (np.isfinite(sma200) and np.isfinite(rsi2) and np.isfinite(atr) and atr > 0):
        return None

    if not (c_i > sma200 and rsi2 < float(p["RSI_ENTRY"])
            and ibs < float(p["IBS_ENTRY"]) and c_i < c_prev):
        return None

    sl = c_i - float(p["SL_ATR"]) * atr   # brain re-anchors off the actual fill
    bar_t = d1["time"].iloc[i] if "time" in d1.columns else d1.index[i]
    return {
        "direction": "LONG", "entry": c_i, "sl": float(sl),
        "sl_atr_mult": float(p["SL_ATR"]), "atr14": atr,
        "rsi2": rsi2, "ibs": float(ibs), "sma200": sma200,
        "bar_time": bar_t,
        "reason": (f"LONG idx-MR dip: RSI2={rsi2:.0f}<{p['RSI_ENTRY']:.0f} "
                   f"IBS={ibs:.2f}<{p['IBS_ENTRY']:.2f} c>SMA200 c<c[1]"),
    }


def should_exit(d1, bars_held, params=None):
    """Exit check on the last CLOSED bar. bars_held = closed D1 bars since the
    entry fill. Returns reason string (brain closes at/near next open) or None.
    The 6xATR broker SL handles the intraday disaster case on its own."""
    p = dict(DEFAULTS)
    if params:
        p.update({k: params[k] for k in DEFAULTS if k in params})
    if d1 is None or len(d1) < int(p["SMA_EXIT"]) + 5:
        return None
    close = d1["close"]
    i = len(d1) - 2
    rsi2 = float(_rsi(close, 2).iloc[i])
    sma5 = float(close.rolling(int(p["SMA_EXIT"])).mean().iloc[i])
    c_i = float(close.iloc[i])
    if np.isfinite(rsi2) and rsi2 > float(p["RSI_EXIT"]):
        return f"RSI2={rsi2:.0f}>{p['RSI_EXIT']:.0f}"
    if np.isfinite(sma5) and c_i > sma5:
        return "close>SMA5"
    if bars_held >= int(p["TIME_STOP_DAYS"]):
        return f"time-stop {bars_held}d"
    return None
