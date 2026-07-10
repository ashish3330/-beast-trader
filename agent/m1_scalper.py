#!/usr/bin/env python3 -B
"""M1 SCALPER — live detector. Research-backed mean-reversion band fade, gated by
session hours + ADX regime + ATR-expansion, targeting the mean. Fixed R stop +
time-stop (enforced by the brain). Own magic range (+5000). XAU only.

Backtest: scripts/_scalper_run.py. Shipped params (2026-07-07 hard tune, PF 1.43,
4/4 walk-forward folds +): session 7-20, ADX<18, RSI(2) 5/95, BB(20,2.0),
SL 1.0xATR, TP=mean, time-stop 10 bars. Pure-function eval — identical logic to
the backtest so live matches.
"""
import numpy as np
import pandas as pd

DEFAULTS = {
    "PERIOD": 20, "BB_MULT": 2.0, "RSI_PERIOD": 2, "RSI_LOW": 5.0, "RSI_HIGH": 95.0,
    "SL_ATR": 1.0, "ADX_MAX": 18.0, "H_START": 7, "H_END": 20,
}


def _wilder(s, p):
    return s.ewm(alpha=1.0 / p, adjust=False).mean()


def evaluate(m1_df, override=None):
    """m1_df: OHLC(+time) M1 DataFrame ascending. Returns sig dict or None.
    Evaluates the last CLOSED bar (iloc[-2]); the live -1 is still forming."""
    p = dict(DEFAULTS)
    if override:
        p.update({k: override[k] for k in DEFAULTS if k in override})
    period = int(p["PERIOD"])
    if m1_df is None or len(m1_df) < period + 40:
        return None
    df = m1_df
    i = len(df) - 2  # last closed M1 bar
    if i < period + 30:
        return None

    close = df["close"]
    high, low = df["high"], df["low"]
    hour = int(pd.to_datetime(df["time"].iloc[i]).hour) if "time" in df.columns \
        else int(df.index[i].hour)
    if not (int(p["H_START"]) <= hour < int(p["H_END"])):
        return None

    c_i = float(close.iloc[i])
    win = close.iloc[max(0, i - period + 1):i + 1]
    mid = float(win.mean())
    sd = float(win.std(ddof=0))
    if not np.isfinite(mid) or sd <= 0:
        return None
    bb = float(p["BB_MULT"])
    lower, upper = mid - bb * sd, mid + bb * sd

    prev_c = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr_s = _wilder(tr, 14)
    atr = float(atr_s.iloc[i])
    atr_ma = float(atr_s.rolling(20).mean().iloc[i]) if i >= 20 else np.nan
    if not np.isfinite(atr) or atr <= 0:
        return None
    if not (np.isfinite(atr_ma) and atr > atr_ma):   # ATR-expansion gate
        return None

    # ADX(14)
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pdi = 100 * _wilder(plus_dm, 14) / atr_s
    mdi = 100 * _wilder(minus_dm, 14) / atr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = float(_wilder(dx.fillna(0), 14).iloc[i])
    if adx >= float(p["ADX_MAX"]):          # only fade in ranges
        return None

    # RSI
    d = close.diff()
    rsi = float((100 - 100 / (1 + _wilder(d.clip(lower=0), int(p["RSI_PERIOD"]))
                / _wilder(-d.clip(upper=0), int(p["RSI_PERIOD"])).replace(0, np.nan))).iloc[i])

    direction = None
    if c_i < lower and rsi < float(p["RSI_LOW"]):
        direction = "LONG"
    elif c_i > upper and rsi > float(p["RSI_HIGH"]):
        direction = "SHORT"
    if direction is None:
        return None

    entry = c_i
    sl_atr = float(p["SL_ATR"])
    if direction == "LONG":
        sl = entry - sl_atr * atr
        tp = mid                       # target the mean
        if tp <= entry or sl >= entry:
            return None
    else:
        sl = entry + sl_atr * atr
        tp = mid
        if tp >= entry or sl <= entry:
            return None

    bar_t = df["time"].iloc[i] if "time" in df.columns else df.index[i]
    return {
        "direction": direction, "entry": float(entry), "sl": float(sl),
        "tp1": float(tp), "tp2": float(tp),   # single target (both legs same TP)
        "atr14": float(atr), "adx14": float(adx), "rsi": float(rsi),
        "mid": float(mid), "bar_time": bar_t,
        "reason": f"{direction} M1-scalp {'<L' if direction=='LONG' else '>U'} "
                  f"RSI{int(p['RSI_PERIOD'])}={rsi:.0f} ADX={adx:.0f} h{hour}",
    }
