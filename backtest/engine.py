"""
Beast Trader — Backtest Engine.
Swing mode: enter on momentum score signal, hold until trailing SL hit or reversal.
Full trailing SL logic (moderate profile), risk-based lot sizing, transaction costs.
"""
import pickle
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow running standalone or as module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_model import CostModel

# ═══════════════════════════════════════════════════════════
# Import scoring system from momentum_scorer (same indicators)
# We replicate _compute_indicators and _score inline to avoid
# import issues with orchestration/core dependencies.
# ═══════════════════════════════════════════════════════════

# ═══ INDICATOR PARAMS ═══
IND_DEFAULTS = {
    "EMA_S": 15, "EMA_L": 40, "EMA_T": 80,
    "ST_F": 2.5, "ST_ATR": 10,
    "MACD_F": 8, "MACD_SL": 21, "MACD_SIG": 7,
    "ATR_LEN": 14,
}

IND_OVERRIDES = {
    "XAUUSD":   {"EMA_S": 15, "EMA_L": 30, "EMA_T": 60, "ST_F": 2.0, "ST_ATR": 7,
                 "MACD_F": 5, "MACD_SL": 26, "MACD_SIG": 4, "ATR_LEN": 7},
    "NAS100.r": {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 2.5, "ST_ATR": 10,
                 "MACD_F": 8, "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "GER40.r":  {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 2.5, "ST_ATR": 10,
                 "MACD_F": 8, "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "BTCUSD":   {"EMA_S": 20, "EMA_L": 50, "EMA_T": 60, "ST_F": 3.5, "ST_ATR": 10,
                 "MACD_F": 12, "MACD_SL": 26, "MACD_SIG": 9, "ATR_LEN": 10},
}

MIN_SCORE = 4.0

REGIME_PARAMS = {
    "trending":  (3.0, 2.0, 3.5, 6.0),
    "ranging":   (2.5, 1.5, 2.5, 4.0),
    "low_vol":   (2.5, 1.5, 2.5, 4.0),
    "high_vol":  (3.5, 1.5, 2.5, 4.0),
}
DEFAULT_PARAMS = (3.0, 1.5, 2.5, 5.0)

# Trailing SL steps (moderate profile from config.py)
# (profit_R_threshold, action, param)
# Walked highest-first: first match wins
TRAIL_STEPS = [
    (6.0, "trail", 0.7),
    (4.0, "trail", 1.0),
    (2.5, "trail", 1.5),
    (1.5, "trail", 2.0),
    (1.0, "lock",  0.5),
    (0.5, "be",    0.0),
]

# ═══ SYMBOL CONFIGS ═══
CACHE_DIR = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

SYMBOL_CFGS = {
    "XAUUSD":   {"cache": "raw_h1_xauusd.pkl",   "point": 0.01,  "tv": 1.0,    "spread": 0.30, "lot_min": 0.01, "cat": "Gold"},
    "BTCUSD":   {"cache": "raw_h1_BTCUSD.pkl",    "point": 0.01,  "tv": 0.01,   "spread": 30.0, "lot_min": 0.01, "cat": "Crypto"},
    "NAS100.r": {"cache": "raw_h1_NAS100_r.pkl",  "point": 0.01,  "tv": 0.01,   "spread": 2.0,  "lot_min": 0.10, "cat": "Index"},
    "GER40.r":  {"cache": "raw_h1_GER40_r.pkl",   "point": 0.01,  "tv": 0.0117, "spread": 2.0,  "lot_min": 0.10, "cat": "Index"},
}


# ═══════════════════════════════════════════════════════════
# NUMPY INDICATORS (copied from momentum_scorer.py to be self-contained)
# ═══════════════════════════════════════════════════════════

def _ema(a, p):
    al = 2 / (p + 1)
    o = np.empty_like(a, dtype=np.float64); o[0] = a[0]
    for i in range(1, len(a)):
        o[i] = al * a[i] + (1 - al) * o[i - 1]
    return o

def _atr(h, l, c, p):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return _ema(tr, p)

def _macd(c, f, s, sig):
    ml = _ema(c, f) - _ema(c, s)
    sl = _ema(ml, sig)
    return ml, sl, ml - sl

def _rsi(c, p=14):
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0, d, 0.0)
    lo = np.where(d < 0, -d, 0.0)
    eg = _ema(g, p); el = _ema(lo, p)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 100 - 100 / (1 + np.where(el > 0, eg / el, 100.0))

def _supertrend(h, l, c, fac, alen):
    nn = len(c); av = _atr(h, l, c, alen)
    hl2 = (h + l) / 2
    u = hl2 + fac * av; lo2 = hl2 - fac * av
    st = np.full(nn, np.nan); d = np.ones(nn, dtype=np.int32)
    for i in range(1, nn):
        plb = lo2[i - 1] if not np.isnan(lo2[i - 1]) else lo2[i]
        pub = u[i - 1] if not np.isnan(u[i - 1]) else u[i]
        if lo2[i] < plb: lo2[i] = plb
        if u[i] > pub: u[i] = pub
        ps = st[i - 1] if not np.isnan(st[i - 1]) else u[i]
        if ps == pub:
            if c[i] > u[i]:  st[i] = lo2[i]; d[i] = 1
            else:             st[i] = u[i];   d[i] = -1
        else:
            if c[i] < lo2[i]: st[i] = u[i];   d[i] = -1
            else:             st[i] = lo2[i]; d[i] = 1
    return st, d

def _ha(o, h, l, c):
    nn = len(c); hc = (o + h + l + c) / 4
    ho = np.empty(nn, dtype=np.float64); ho[0] = (o[0] + c[0]) / 2
    for i in range(1, nn):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return ho, hc

def _bb(c, ln=20, ns=2.0):
    w = np.full_like(c, np.nan)
    bbu = np.full_like(c, np.nan); bbl = np.full_like(c, np.nan)
    for i in range(ln - 1, len(c)):
        win = c[i - ln + 1:i + 1]
        m = np.mean(win); s = np.std(win)
        bbu[i] = m + ns * s; bbl[i] = m - ns * s
        w[i] = (bbu[i] - bbl[i]) / m * 100 if m > 0 else 0
    return bbu, bbl, w

def _donch(h, l, ln=20):
    u = np.full_like(h, np.nan); lo = np.full_like(l, np.nan)
    for i in range(ln - 1, len(h)):
        u[i] = np.max(h[i - ln + 1:i + 1]); lo[i] = np.min(l[i - ln + 1:i + 1])
    return u, lo


def _compute_indicators(df, icfg):
    """Precompute all indicators on the dataframe."""
    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    n = len(c)

    es = _ema(c, icfg["EMA_S"])
    el = _ema(c, icfg["EMA_L"])
    et = _ema(c, icfg["EMA_T"])
    at = _atr(h, l, c, icfg["ATR_LEN"])
    ml, ms, mh = _macd(c, icfg["MACD_F"], icfg["MACD_SL"], icfg["MACD_SIG"])
    stl, std = _supertrend(h.copy(), l.copy(), c, icfg["ST_F"], icfg["ST_ATR"])
    hao, hac = _ha(o, h, l, c)
    rs = _rsi(c, 14)
    bbu, bbl, bbw = _bb(c, 20, 2.0)
    dcu, dcl = _donch(h, l, 20)

    # Candle patterns
    be = np.zeros(n); se = np.zeros(n)
    bp = np.zeros(n); sp = np.zeros(n)
    for i in range(1, n):
        if c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] <= c[i - 1] and c[i] >= o[i - 1]: be[i] = 1
        if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] >= c[i - 1] and c[i] <= o[i - 1]: se[i] = 1
        body = abs(c[i] - o[i]); fr = h[i] - l[i]
        if fr > 0:
            lw = min(c[i], o[i]) - l[i]; uw = h[i] - max(c[i], o[i])
            if lw > body * 2 and lw > fr * 0.6: bp[i] = 1
            if uw > body * 2 and uw > fr * 0.6: sp[i] = 1

    # Structure trend
    st_ = np.zeros(n)
    for i in range(20, n):
        ch = np.max(h[i - 10:i + 1]); ph = np.max(h[i - 20:i - 9])
        cl = np.min(l[i - 10:i + 1]); pl = np.min(l[i - 20:i - 9])
        if ch > ph and cl > pl: st_[i] = 1
        elif ch < ph and cl < pl: st_[i] = -1

    # Breakout
    bkl = np.zeros(n); bks = np.zeros(n)
    for i in range(2, n):
        if np.isnan(dcu[i - 1]) or np.isnan(bbw[i]): continue
        sq = (not np.isnan(bbw[i - 1])) and bbw[i - 1] < 3.0 and bbw[i] > bbw[i - 1]
        if c[i] > dcu[i - 1] and (sq or c[i] > bbu[i]): bkl[i] = 1
        if c[i] < dcl[i - 1] and (sq or c[i] < bbl[i]): bks[i] = 1

    # ADX
    adx = np.full(n, np.nan, dtype=np.float64)
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    for j in range(1, n):
        up = h[j] - h[j-1]; down = l[j-1] - l[j]
        plus_di[j] = up if up > down and up > 0 else 0
        minus_di[j] = down if down > up and down > 0 else 0
    sm_plus = _ema(np.nan_to_num(plus_di), 14)
    sm_minus = _ema(np.nan_to_num(minus_di), 14)
    sm_atr = _atr(h, l, c, 14)
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = np.where(sm_atr > 0, 100 * sm_plus / sm_atr, 0)
        di_minus = np.where(sm_atr > 0, 100 * sm_minus / sm_atr, 0)
        dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = _ema(dx, 14)

    # VWAP proxy
    vwap = np.full(n, np.nan, dtype=np.float64)
    vol = df["tick_volume"].values.astype(np.float64) if "tick_volume" in df.columns else np.ones(n)
    for j in range(20, n):
        pv = np.sum(c[j-20:j] * vol[j-20:j])
        sv = np.sum(vol[j-20:j])
        vwap[j] = pv / sv if sv > 0 else c[j]

    # Volume SMA
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    for j in range(20, n):
        vol_sma[j] = np.mean(vol[j-20:j])

    # 3-bar momentum
    roc3 = np.zeros(n, dtype=np.float64)
    for j in range(3, n):
        if c[j-3] > 0: roc3[j] = (c[j] - c[j-3]) / c[j-3] * 100

    # Consecutive candle count
    consec = np.zeros(n, dtype=np.float64)
    for j in range(1, n):
        if c[j] > o[j] and c[j-1] > o[j-1]: consec[j] = consec[j-1] + 1
        elif c[j] < o[j] and c[j-1] < o[j-1]: consec[j] = consec[j-1] - 1
        else: consec[j] = 1 if c[j] > o[j] else -1

    return {"c": c, "o": o, "h": h, "l": l, "n": n,
            "es": es, "el": el, "et": et, "at": at,
            "ml": ml, "ms": ms, "mh": mh,
            "stl": stl, "hao": hao, "hac": hac, "rs": rs, "bbw": bbw,
            "be": be, "se": se, "bp": bp, "sp": sp,
            "st": st_, "bkl": bkl, "bks": bks,
            "adx": adx, "di_plus": di_plus, "di_minus": di_minus,
            "vwap": vwap, "vol": vol, "vol_sma": vol_sma,
            "roc3": roc3, "consec": consec,
            "bbu": bbu, "bbl": bbl, "dcu": dcu, "dcl": dcl}


def _score(ind, i):
    """Multi-indicator scoring (max ~14 points per side). Exact copy from momentum_scorer."""
    sl = ss = 0.0
    p = ind["c"][i]; o = ind["o"][i]; h = ind["h"][i]; l = ind["l"][i]
    atr = ind["at"][i] if not np.isnan(ind["at"][i]) else 1.0

    # 1. EMA STACK (0-1.5)
    if p > ind["et"][i] and ind["es"][i] > ind["el"][i]: sl += 0.5
    if p < ind["et"][i] and ind["es"][i] < ind["el"][i]: ss += 0.5
    ema_sep = abs(ind["es"][i] - ind["el"][i])
    if ema_sep > 0.5 * atr:
        if ind["es"][i] > ind["el"][i]: sl += 0.25
        else: ss += 0.25
    if not np.isnan(ind["vwap"][i]):
        if p > ind["vwap"][i] and ind["es"][i] > ind["el"][i]: sl += 0.25
        if p < ind["vwap"][i] and ind["es"][i] < ind["el"][i]: ss += 0.25
    if ind["es"][i] > ind["el"][i] > ind["et"][i]: sl += 0.5
    if ind["es"][i] < ind["el"][i] < ind["et"][i]: ss += 0.5

    # 2. SUPERTREND (0-1.5)
    if not np.isnan(ind["stl"][i]):
        if p > ind["stl"][i]:
            sl += 0.5
            st_dist = (p - ind["stl"][i]) / atr
            if st_dist > 1.0: sl += 0.5
            if st_dist > 2.0: sl += 0.5
        if p < ind["stl"][i]:
            ss += 0.5
            st_dist = (ind["stl"][i] - p) / atr
            if st_dist > 1.0: ss += 0.5
            if st_dist > 2.0: ss += 0.5

    # 3. MACD LINE vs SIGNAL (0-1.5)
    if ind["ml"][i] > ind["ms"][i]:
        sl += 0.5
        if i > 1 and (ind["ml"][i] - ind["ms"][i]) > (ind["ml"][i-1] - ind["ms"][i-1]):
            sl += 0.5
    if ind["ml"][i] < ind["ms"][i]:
        ss += 0.5
        if i > 1 and (ind["ms"][i] - ind["ml"][i]) > (ind["ms"][i-1] - ind["ml"][i-1]):
            ss += 0.5
    if i > 3:
        for k in range(1, 4):
            if ind["ml"][i-k] <= ind["ms"][i-k] and ind["ml"][i] > ind["ms"][i]:
                sl += 0.5; break
            if ind["ml"][i-k] >= ind["ms"][i-k] and ind["ml"][i] < ind["ms"][i]:
                ss += 0.5; break

    # 4. MACD HISTOGRAM (0-1.0)
    if ind["mh"][i] > 0: sl += 0.25
    if ind["mh"][i] < 0: ss += 0.25
    if i > 1:
        if ind["mh"][i] > 0 and ind["mh"][i] > ind["mh"][i-1]: sl += 0.25
        if ind["mh"][i] < 0 and ind["mh"][i] < ind["mh"][i-1]: ss += 0.25
    if i > 5:
        mh_avg = np.mean([ind["mh"][i-j] for j in range(5)])
        if ind["mh"][i] > mh_avg and ind["mh"][i] > 0: sl += 0.25
        if ind["mh"][i] < mh_avg and ind["mh"][i] < 0: ss += 0.25
    if i > 1:
        if ind["mh"][i] > 0 and ind["mh"][i] < ind["mh"][i-1] * 0.7: sl -= 0.25
        if ind["mh"][i] < 0 and ind["mh"][i] > ind["mh"][i-1] * 0.7: ss -= 0.25

    # 5. RSI (0-1.0)
    rsi = ind["rs"][i]
    if not np.isnan(rsi):
        if 45 < rsi < 65: sl += 0.25
        if 35 < rsi < 55: ss += 0.25
        if i > 1 and not np.isnan(ind["rs"][i-1]):
            if rsi > ind["rs"][i-1] and rsi < 70: sl += 0.25
            if rsi < ind["rs"][i-1] and rsi > 30: ss += 0.25
        if i > 5:
            rsi_5ago = ind["rs"][i-5]
            if not np.isnan(rsi_5ago):
                if rsi_5ago > 70 and 45 < rsi < 65: sl += 0.25
                if rsi_5ago < 30 and 35 < rsi < 55: ss += 0.25
        if rsi > 75: sl -= 0.25
        if rsi < 25: ss -= 0.25

    # 6. CANDLESTICK PATTERNS (0-2.0)
    body = abs(p - o)
    full_range = h - l if h > l else 0.001
    body_ratio = body / full_range
    if ind["be"][i]:
        sl += 1.0
        if body_ratio > 0.6: sl += 0.5
    if ind["se"][i]:
        ss += 1.0
        if body_ratio > 0.6: ss += 0.5
    if ind["bp"][i]:
        sl += 0.5
        lower_wick = min(p, o) - l
        if lower_wick > body * 2.5: sl += 0.5
    if ind["sp"][i]:
        ss += 0.5
        upper_wick = h - max(p, o)
        if upper_wick > body * 2.5: ss += 0.5
    if not np.isnan(ind["vol_sma"][i]) and ind["vol_sma"][i] > 0:
        vol_ratio = ind["vol"][i] / ind["vol_sma"][i]
        if (ind["be"][i] or ind["bp"][i]) and vol_ratio > 1.3: sl += 0.5
        if (ind["se"][i] or ind["sp"][i]) and vol_ratio > 1.3: ss += 0.5

    # 7. HEIKIN ASHI (0-1.0)
    if ind["hac"][i] > ind["hao"][i]: sl += 0.25
    if ind["hac"][i] < ind["hao"][i]: ss += 0.25
    if i > 2:
        ha_streak_bull = all(ind["hac"][i-k] > ind["hao"][i-k] for k in range(3))
        ha_streak_bear = all(ind["hac"][i-k] < ind["hao"][i-k] for k in range(3))
        if ha_streak_bull: sl += 0.5
        if ha_streak_bear: ss += 0.5
    if i > 1:
        ha_body_now = abs(ind["hac"][i] - ind["hao"][i])
        ha_body_prev = abs(ind["hac"][i-1] - ind["hao"][i-1])
        if ha_body_now > ha_body_prev * 1.2 and ind["hac"][i] > ind["hao"][i]: sl += 0.25
        if ha_body_now > ha_body_prev * 1.2 and ind["hac"][i] < ind["hao"][i]: ss += 0.25

    # 8. STRUCTURE TREND (0-1.5)
    if ind["st"][i] == 1: sl += 0.5
    if ind["st"][i] == -1: ss += 0.5
    if not np.isnan(ind["adx"][i]):
        if ind["adx"][i] > 25:
            if ind["st"][i] == 1: sl += 0.5
            if ind["st"][i] == -1: ss += 0.5
        if ind["adx"][i] > 40:
            if ind["di_plus"][i] > ind["di_minus"][i]: sl += 0.5
            if ind["di_minus"][i] > ind["di_plus"][i]: ss += 0.5
    if ind["st"][i] == 0 and not np.isnan(ind["adx"][i]) and ind["adx"][i] < 20:
        sl -= 0.25; ss -= 0.25

    # 9. BREAKOUT (0-2.5)
    if ind["bkl"][i]:
        sl += 1.0
        if not np.isnan(ind["bbw"][i]) and i > 1 and not np.isnan(ind["bbw"][i-1]):
            if ind["bbw"][i-1] < 2.0: sl += 0.5
        if not np.isnan(ind["dcu"][i-1]):
            break_dist = (p - ind["dcu"][i-1]) / atr
            if break_dist > 0.5: sl += 0.5
        if not np.isnan(ind["vol_sma"][i]) and ind["vol_sma"][i] > 0:
            if ind["vol"][i] > 1.5 * ind["vol_sma"][i]: sl += 0.5
    if ind["bks"][i]:
        ss += 1.0
        if not np.isnan(ind["bbw"][i]) and i > 1 and not np.isnan(ind["bbw"][i-1]):
            if ind["bbw"][i-1] < 2.0: ss += 0.5
        if not np.isnan(ind["dcl"][i-1]):
            break_dist = (ind["dcl"][i-1] - p) / atr
            if break_dist > 0.5: ss += 0.5
        if not np.isnan(ind["vol_sma"][i]) and ind["vol_sma"][i] > 0:
            if ind["vol"][i] > 1.5 * ind["vol_sma"][i]: ss += 0.5

    # 10. MOMENTUM VELOCITY (0-0.5)
    if ind["roc3"][i] > 0.3: sl += 0.25
    if ind["roc3"][i] < -0.3: ss += 0.25
    if ind["roc3"][i] > 0.6: sl += 0.25
    if ind["roc3"][i] < -0.6: ss += 0.25

    # 11. TREND PERSISTENCE (0-0.5)
    if ind["consec"][i] >= 3: sl += 0.25
    if ind["consec"][i] <= -3: ss += 0.25
    if ind["consec"][i] >= 5: sl += 0.25
    if ind["consec"][i] <= -5: ss += 0.25

    sl = max(0, sl)
    ss = max(0, ss)
    return sl, ss


def mean_reversion_score(ind, i):
    '''Returns (long_score, short_score) for mean-reversion strategy.
    ind: dict of numpy arrays (same as momentum_scorer uses: rs, bbw, bbu, bbl, at, c, o, h, l, es, el, et, adx, vol, vol_sma, consec)
    i: bar index
    '''
    sl = ss = 0.0
    p = ind["c"][i]
    atr = ind["at"][i] if not np.isnan(ind["at"][i]) else 1e-10

    # --- 1. RSI EXTREME (0-2) ---
    rsi = ind["rs"][i]
    if not np.isnan(rsi):
        if rsi < 25:
            sl += 2.0
        elif rsi < 30:
            sl += 1.0
        if rsi > 75:
            ss += 2.0
        elif rsi > 70:
            ss += 1.0

    # --- 2. BOLLINGER BAND TOUCH (0-2) ---
    bbu = ind["bbu"][i]
    bbl = ind["bbl"][i]
    if not np.isnan(bbl) and not np.isnan(bbu):
        if p < bbl:
            dist_below = (bbl - p) / atr
            if dist_below > 1.0:
                sl += 2.0
            else:
                sl += 1.0
        if p > bbu:
            dist_above = (p - bbu) / atr
            if dist_above > 1.0:
                ss += 2.0
            else:
                ss += 1.0

    # --- 3. VOLUME EXHAUSTION (0-2) ---
    # Volume spike at price extreme suggests capitulation / reversal
    vol = ind["vol"][i]
    vol_avg = ind["vol_sma"][i]
    if not np.isnan(vol_avg) and vol_avg > 0:
        vol_ratio = vol / vol_avg
        # Only score if price is also at an extreme (RSI or BB)
        price_extreme_long = (not np.isnan(rsi) and rsi < 35) or (not np.isnan(bbl) and p <= bbl)
        price_extreme_short = (not np.isnan(rsi) and rsi > 65) or (not np.isnan(bbu) and p >= bbu)
        if vol_ratio > 2.0:
            if price_extreme_long:
                sl += 2.0
            if price_extreme_short:
                ss += 2.0
        elif vol_ratio > 1.5:
            if price_extreme_long:
                sl += 1.0
            if price_extreme_short:
                ss += 1.0

    # --- 4. CONSECUTIVE CANDLES (0-2) ---
    # Bearish streak = LONG mean-reversion, bullish streak = SHORT mean-reversion
    consec = ind["consec"][i]
    if consec <= -7:
        sl += 2.0
    elif consec <= -5:
        sl += 1.0
    if consec >= 7:
        ss += 2.0
    elif consec >= 5:
        ss += 1.0

    # --- 5. DISTANCE FROM EMA (0-2) ---
    # Use EMA short (es) as the mean reference
    ema = ind["es"][i]
    if not np.isnan(ema):
        dist_from_ema = (p - ema) / atr
        if dist_from_ema < -2.0:
            sl += 2.0
        elif dist_from_ema < -1.5:
            sl += 1.0
        if dist_from_ema > 2.0:
            ss += 2.0
        elif dist_from_ema > 1.5:
            ss += 1.0

    # --- REGIME FILTER: ADX > 30 = strong trend, reduce by 50% ---
    adx = ind["adx"][i]
    if not np.isnan(adx) and adx > 30:
        sl *= 0.5
        ss *= 0.5

    # Clamp to [0, 10]
    sl = min(10.0, max(0.0, sl))
    ss = min(10.0, max(0.0, ss))
    return sl, ss


# ═══════════════════════════════════════════════════════════
# TRADE RECORD
# ═══════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    symbol: str
    direction: int           # 1=LONG, -1=SHORT
    entry_price: float
    exit_price: float
    entry_time: object       # datetime
    exit_time: object
    sl_dist: float           # initial SL distance
    lot_size: float
    pnl_usd: float
    pnl_r: float             # PnL in R multiples
    exit_reason: str         # "trailing_sl", "reversal", "end_of_data"
    max_r: float = 0.0       # best R achieved during trade


# ═══════════════════════════════════════════════════════════
# BACKTEST RESULT
# ═══════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    symbol: str
    trades: int
    wins: int
    win_rate: float
    profit_factor: float
    sharpe: float
    max_dd_pct: float
    avg_r: float
    total_return_pct: float
    final_equity: float
    gross_profit: float
    gross_loss: float
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[TradeRecord] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════

def classify_regime(bbw):
    if bbw is None or np.isnan(bbw):
        return "unknown"
    if bbw > 4.0:
        return "high_vol"
    elif bbw < 1.5:
        return "low_vol"
    elif bbw < 3.0:
        return "ranging"
    return "trending"


# ═══════════════════════════════════════════════════════════
# M15 REVERSAL DETECTION (simulated from H1 if M15 not available)
# ═══════════════════════════════════════════════════════════

def detect_m15_reversal(ind, i, direction):
    """
    Simulated M15 reversal detection using H1 data.
    Checks for momentum exhaustion on the completed bar.
    Returns True if reversal detected against current direction.
    """
    if i < 5:
        return False

    # Heikin Ashi reversal: HA flipped direction for 2+ bars
    ha_now = ind["hac"][i] > ind["hao"][i]  # True = bullish HA
    ha_prev = ind["hac"][i-1] > ind["hao"][i-1]

    if direction == 1 and not ha_now and not ha_prev:
        # Two consecutive bearish HA bars while long
        # Additional confirmation: MACD histogram turning negative
        if ind["mh"][i] < 0 and ind["mh"][i-1] < ind["mh"][i-2]:
            return True

    if direction == -1 and ha_now and ha_prev:
        # Two consecutive bullish HA bars while short
        if ind["mh"][i] > 0 and ind["mh"][i-1] > ind["mh"][i-2]:
            return True

    return False


# ═══════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════

def run_backtest(
    symbol: str,
    days: int = 365,
    start_equity: float = 1000.0,
    risk_pct: float = 1.0,
    meta_filter: Optional[Callable] = None,
    use_m15_reversal: bool = True,
) -> BacktestResult:
    """
    Run swing-mode backtest on a single symbol.

    Args:
        symbol: symbol name matching SYMBOL_CFGS keys
        days: lookback period in days
        start_equity: starting account balance in USD
        risk_pct: risk per trade as % of equity
        meta_filter: optional callable(ind, i, direction) -> bool
                     Return True to ACCEPT the signal, False to reject.
        use_m15_reversal: use M15 reversal detection for exits
    """
    scfg = SYMBOL_CFGS[symbol]
    cache_path = CACHE_DIR / scfg["cache"]
    if not cache_path.exists():
        print(f"[{symbol}] Cache file not found: {cache_path}")
        return BacktestResult(symbol=symbol, trades=0, wins=0, win_rate=0,
                              profit_factor=0, sharpe=0, max_dd_pct=0, avg_r=0,
                              total_return_pct=0, final_equity=start_equity,
                              gross_profit=0, gross_loss=0)

    df = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    pt = scfg["point"]
    tv = scfg["tv"]
    cat = scfg["cat"]
    lot_min = scfg["lot_min"]

    # Cost model: spread + 1 point slippage per side
    cost = CostModel(spread=scfg["spread"], slippage_pts=1.0, point=pt)

    # Indicator config
    icfg = dict(IND_DEFAULTS)
    icfg.update(IND_OVERRIDES.get(symbol, {}))

    # Compute cutoff
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    start_idx = max(int(df[df["time"] >= cutoff].index[0]), icfg["EMA_T"] + 30)

    # Compute indicators on full dataset
    ind = _compute_indicators(df, icfg)
    n = ind["n"]

    # State
    eq = start_equity
    peak = start_equity
    max_dd = 0.0
    equity_curve = [eq]

    # Trade tracking
    trades: List[TradeRecord] = []
    gross_p = 0.0
    gross_l = 0.0

    # Position state
    in_trade = False
    d = 0           # direction: 1=LONG, -1=SHORT
    entry = 0.0
    pos_sl = 0.0
    sl_dist = 0.0
    lot = 0.0
    entry_time = None
    max_r = 0.0

    def _close_trade(exit_price, exit_time, reason):
        """Close current position and record trade."""
        nonlocal eq, peak, max_dd, in_trade, gross_p, gross_l

        # Apply exit cost
        adj_exit = exit_price + cost.exit_cost(d)
        pnl_price = d * (adj_exit - entry)
        pnl_pts = pnl_price / pt
        pnl_usd = pnl_pts * tv * lot
        pnl_r = pnl_price / sl_dist if sl_dist > 0 else 0.0

        eq += pnl_usd
        if pnl_usd > 0:
            gross_p += pnl_usd
        else:
            gross_l += abs(pnl_usd)

        peak = max(peak, eq)
        dd = peak - eq
        max_dd = max(max_dd, dd)

        trades.append(TradeRecord(
            symbol=symbol, direction=d,
            entry_price=entry, exit_price=adj_exit,
            entry_time=entry_time, exit_time=exit_time,
            sl_dist=sl_dist, lot_size=lot,
            pnl_usd=pnl_usd, pnl_r=pnl_r,
            exit_reason=reason, max_r=max_r,
        ))
        equity_curve.append(eq)
        in_trade = False

    for i in range(start_idx, n):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0:
            continue

        bar_time = df["time"].iloc[i]
        bar_hour = bar_time.hour if hasattr(bar_time, "hour") else 12

        # Session filter: non-crypto skip 22-06 UTC
        if cat != "Crypto" and (bar_hour >= 22 or bar_hour < 6):
            continue

        # ══════ MANAGE OPEN POSITION ══════
        if in_trade:
            # Check SL hit by bar's high/low
            if (d == 1 and ind["l"][i] <= pos_sl) or (d == -1 and ind["h"][i] >= pos_sl):
                _close_trade(pos_sl, bar_time, "trailing_sl")
                continue

            # M15 reversal detection
            if use_m15_reversal and detect_m15_reversal(ind, i, d):
                _close_trade(float(ind["c"][i]), bar_time, "m15_reversal")
                continue

            # Trail SL based on current bar close
            cur = float(ind["c"][i])
            profit_dist = (cur - entry) * d
            profit_r = profit_dist / sl_dist if sl_dist > 0 else 0
            max_r = max(max_r, profit_r)

            new_sl = None
            for r_threshold, step_type, param in TRAIL_STEPS:
                if profit_r >= r_threshold:
                    if step_type == "trail":
                        trail_dist = param * atr_val
                        new_sl = cur - trail_dist * d
                        # Floor: never trail below lock level when above 1.5R
                        if profit_r >= 1.5:
                            floor = entry + 0.5 * sl_dist * d
                            if d == 1:
                                new_sl = max(new_sl, floor)
                            else:
                                new_sl = min(new_sl, floor)
                    elif step_type == "lock":
                        new_sl = entry + param * sl_dist * d
                    elif step_type == "be":
                        new_sl = entry + 2 * pt * d
                    break

            if new_sl is not None:
                if d == 1 and new_sl > pos_sl:
                    pos_sl = new_sl
                elif d == -1 and new_sl < pos_sl:
                    pos_sl = new_sl

        # ══════ SCORE (on completed bar = i-1) ══════
        bi = i - 1
        if bi < 21:
            continue

        ls, ss = _score(ind, bi)
        buy = ls >= MIN_SCORE
        sell = ss >= MIN_SCORE
        if not buy and not sell:
            continue

        new_dir = 1 if (buy and (not sell or ls >= ss)) else -1

        # Meta-label filter: reject if filter says no
        if meta_filter is not None:
            if not meta_filter(ind, bi, new_dir):
                continue

        # ══════ SIGNAL REVERSAL — close old, open new ══════
        if in_trade and new_dir != d:
            _close_trade(float(ind["c"][i]), bar_time, "reversal")

        # ══════ ENTRY (only if flat) ══════
        if not in_trade:
            d = new_dir

            # Regime-adaptive SL
            bbw_val = float(ind["bbw"][bi]) if not np.isnan(ind["bbw"][bi]) else None
            regime = classify_regime(bbw_val)
            sl_mult = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]

            sl_dist = max(atr_val * sl_mult, atr_val * 1.0)  # min 1x ATR

            # SuperTrend reference for SL
            if not np.isnan(ind["stl"][bi]):
                st = float(ind["stl"][bi])
                if d == 1 and st < ind["o"][i]:
                    st_sd = ind["o"][i] - st
                    if st_sd > atr_val * 1.0:
                        sl_dist = max(st_sd, atr_val * 1.0)

            # Entry price with cost
            raw_entry = float(ind["o"][i])
            entry = raw_entry + cost.entry_cost(d)
            pos_sl = entry - sl_dist * d

            # Risk-based lot sizing: 1% equity / (sl_points * tick_value)
            risk_amount = eq * (risk_pct / 100.0)
            sl_pts = sl_dist / pt
            if tv > 0 and sl_pts > 0:
                lot = risk_amount / (sl_pts * tv)
            else:
                lot = lot_min

            # Clamp to min lot, round to 2 decimals
            lot = max(lot_min, lot)
            lot = round(int(lot / lot_min) * lot_min, 2)

            entry_time = bar_time
            in_trade = True
            max_r = 0.0

    # Close any open position at end of data
    if in_trade:
        _close_trade(float(ind["c"][n - 1]), df["time"].iloc[n - 1], "end_of_data")

    # ══════ COMPUTE STATS ══════
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl_usd > 0)
    win_rate = (wins / n_trades * 100) if n_trades else 0
    pf = gross_p / gross_l if gross_l > 0 else (999.0 if gross_p > 0 else 0)
    dd_pct = (max_dd / peak * 100) if peak > 0 else 0
    ret_pct = (eq - start_equity) / start_equity * 100
    avg_r = np.mean([t.pnl_r for t in trades]) if trades else 0

    # Sharpe ratio (annualized from per-trade returns)
    if n_trades > 1:
        trade_returns = [t.pnl_usd / start_equity for t in trades]
        mean_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns)
        # Approximate: assume ~250 trades/year
        trades_per_year = min(n_trades * (365 / days), 500)
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return BacktestResult(
        symbol=symbol, trades=n_trades, wins=wins,
        win_rate=round(win_rate, 1),
        profit_factor=round(pf, 2),
        sharpe=round(sharpe, 2),
        max_dd_pct=round(dd_pct, 1),
        avg_r=round(avg_r, 2),
        total_return_pct=round(ret_pct, 1),
        final_equity=round(eq, 2),
        gross_profit=round(gross_p, 2),
        gross_loss=round(gross_l, 2),
        equity_curve=equity_curve,
        trade_log=trades,
    )


# ═══════════════════════════════════════════════════════════
# MULTI-SYMBOL PORTFOLIO BACKTEST
# ═══════════════════════════════════════════════════════════

def run_portfolio(
    symbols: Optional[List[str]] = None,
    days: int = 365,
    start_equity: float = 1000.0,
    risk_pct: float = 1.0,
    meta_filter: Optional[Callable] = None,
) -> Dict[str, BacktestResult]:
    """Run backtest on multiple symbols independently."""
    if symbols is None:
        symbols = list(SYMBOL_CFGS.keys())

    results = {}
    for sym in symbols:
        if sym not in SYMBOL_CFGS:
            print(f"[{sym}] Not in SYMBOL_CFGS, skipping")
            continue
        results[sym] = run_backtest(sym, days=days, start_equity=start_equity,
                                     risk_pct=risk_pct, meta_filter=meta_filter)
    return results


# ═══════════════════════════════════════════════════════════
# PRETTY PRINT
# ═══════════════════════════════════════════════════════════

def print_results(results: Dict[str, BacktestResult], days: int, start_eq: float):
    """Print formatted results table."""
    print("=" * 110)
    print("  BEAST TRADER BACKTEST: Swing Mode | Trail SL (moderate) | Risk-based lots | Spread+Slip costs")
    print(f"  ${start_eq:.0f} | {days}d | MIN_SCORE {MIN_SCORE} | Session filter (non-crypto skip 22-06 UTC)")
    print("=" * 110)
    print(f"\n{'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Sharpe':>7} {'AvgR':>7} "
          f"{'Return%':>9} {'DD%':>7} {'Final$':>10}")
    print("-" * 110)

    total_gp = total_gl = 0.0
    for sym, r in results.items():
        total_gp += r.gross_profit
        total_gl += r.gross_loss
        print(f"{sym:<12} {r.trades:>7} {r.win_rate:>6.1f}% {r.profit_factor:>7.2f} "
              f"{r.sharpe:>7.2f} {r.avg_r:>6.2f}R "
              f"{r.total_return_pct:>8.1f}% {r.max_dd_pct:>6.1f}% ${r.final_equity:>9.2f}")

    print("-" * 110)
    port_pf = total_gp / total_gl if total_gl > 0 else 0
    print(f"{'PORTFOLIO':<12} {'':>7} {'':>7} {port_pf:>7.2f}")
    print("=" * 110)

    # Exit reason breakdown
    print(f"\n{'Symbol':<12} {'SL_exit':>8} {'Reversal':>9} {'M15_rev':>8} {'EOD':>6}")
    print("-" * 50)
    for sym, r in results.items():
        sl_exits = sum(1 for t in r.trade_log if t.exit_reason == "trailing_sl")
        rev_exits = sum(1 for t in r.trade_log if t.exit_reason == "reversal")
        m15_exits = sum(1 for t in r.trade_log if t.exit_reason == "m15_reversal")
        eod_exits = sum(1 for t in r.trade_log if t.exit_reason == "end_of_data")
        print(f"{sym:<12} {sl_exits:>8} {rev_exits:>9} {m15_exits:>8} {eod_exits:>6}")
    print()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beast Trader Backtest Engine")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--equity", type=float, default=1000.0, help="Starting equity")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade (%%)")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to test")
    parser.add_argument("--no-m15", action="store_true", help="Disable M15 reversal exits")
    args = parser.parse_args()

    t0 = time.time()

    symbols = [args.symbol] if args.symbol else None

    if args.symbol:
        result = run_backtest(
            args.symbol, days=args.days, start_equity=args.equity,
            risk_pct=args.risk, use_m15_reversal=not args.no_m15,
        )
        print_results({args.symbol: result}, args.days, args.equity)
    else:
        results = run_portfolio(
            symbols=symbols, days=args.days,
            start_equity=args.equity, risk_pct=args.risk,
        )
        print_results(results, args.days, args.equity)

    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")
