"""
Momentum Strategy — Full multi-indicator scoring (ported from proven sim.py).

11-point scoring: EMA stack, SuperTrend, MACD, RSI, engulfing, pin bars,
Heikin Ashi, structure trend, Donchian/BB breakout.
ATR-based SL with SuperTrend as reference. Regime-adaptive TP.
Min score 3.5 to enter (proven threshold).
"""
import numpy as np
import pandas as pd
# Standalone — no dependency on ApexQuant orchestration
try:
    from orchestration.context import CandidateSignal
    from core.config import SymbolConfig
except ImportError:
    CandidateSignal = None
    SymbolConfig = None

# ═══ INDICATOR PARAMS (proven baseline from sim.py) ═══
IND_DEFAULTS = {
    "EMA_S": 15, "EMA_L": 40, "EMA_T": 80,
    "ST_F": 2.5, "ST_ATR": 10,
    "MACD_F": 8, "MACD_SL": 21, "MACD_SIG": 7,
    "ATR_LEN": 14,
}

# Per-symbol overrides (from BASELINE_IND in sim.py)
IND_OVERRIDES = {
    "XAUUSD":   {"EMA_S": 15, "EMA_L": 30, "EMA_T": 60, "ST_F": 2.0, "ST_ATR": 7,
                 "MACD_F": 5, "MACD_SL": 26, "MACD_SIG": 4, "ATR_LEN": 7},
    "NAS100.r": {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 2.5, "ST_ATR": 10,
                 "MACD_F": 8, "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "GER40.r":  {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 2.5, "ST_ATR": 10,
                 "MACD_F": 8, "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "SP500.r":  {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 2.5, "ST_ATR": 10,
                 "MACD_F": 8, "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "UK100.r":  {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 2.5, "ST_ATR": 10,
                 "MACD_F": 8, "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "XAGUSD":   {"EMA_S": 8,  "EMA_L": 40, "EMA_T": 80, "ST_F": 3.5, "ST_ATR": 14,
                 "MACD_F": 12, "MACD_SL": 21, "MACD_SIG": 9, "ATR_LEN": 14},
    "BTCUSD":   {"EMA_S": 20, "EMA_L": 50, "EMA_T": 60, "ST_F": 3.5, "ST_ATR": 10,
                 "MACD_F": 12, "MACD_SL": 26, "MACD_SIG": 9, "ATR_LEN": 10},
    "ETHUSD":   {"EMA_S": 20, "EMA_L": 50, "EMA_T": 60, "ST_F": 3.5, "ST_ATR": 10,
                 "MACD_F": 12, "MACD_SL": 26, "MACD_SIG": 9, "ATR_LEN": 10},
    "JPN225ft": {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 2.5, "ST_ATR": 10,
                 "MACD_F": 8,  "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "USDJPY":   {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 3.0, "ST_ATR": 10,
                 "MACD_F": 8,  "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "GBPUSD":   {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 3.0, "ST_ATR": 10,
                 "MACD_F": 8,  "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "EURUSD":   {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 3.0, "ST_ATR": 10,
                 "MACD_F": 8,  "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
    "AUDJPY":   {"EMA_S": 15, "EMA_L": 40, "EMA_T": 80, "ST_F": 3.0, "ST_ATR": 10,
                 "MACD_F": 8,  "MACD_SL": 21, "MACD_SIG": 7, "ATR_LEN": 10},
}

MIN_SCORE = 4.0  # swing mode — score picks direction, trailing SL is the edge

# ═══ REGIME-DEPENDENT SL/TP ═══
REGIME_PARAMS = {
    "trending":  (3.0, 2.0, 3.5, 6.0),   # 3× ATR SL — survives intrabar noise
    "ranging":   (2.5, 1.5, 2.5, 4.0),   # 2.5× ATR
    "low_vol":   (2.5, 1.5, 2.5, 4.0),   # 2.5× ATR
    "high_vol":  (3.5, 1.5, 2.5, 4.0),   # 3.5× ATR — volatile needs most room
}
DEFAULT_PARAMS = (1.0, 1.5, 2.5, 5.0)


# ═══ NUMPY INDICATORS (from sim.py — battle-tested) ═══
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
        # Bullish/bearish engulfing
        if c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] <= c[i - 1] and c[i] >= o[i - 1]: be[i] = 1
        if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] >= c[i - 1] and c[i] <= o[i - 1]: se[i] = 1
        # Pin bars
        body = abs(c[i] - o[i]); fr = h[i] - l[i]
        if fr > 0:
            lw = min(c[i], o[i]) - l[i]; uw = h[i] - max(c[i], o[i])
            if lw > body * 2 and lw > fr * 0.6: bp[i] = 1
            if uw > body * 2 and uw > fr * 0.6: sp[i] = 1

    # Structure trend (higher highs/lows)
    st_ = np.zeros(n)
    for i in range(20, n):
        ch = np.max(h[i - 10:i + 1]); ph = np.max(h[i - 20:i - 9])
        cl = np.min(l[i - 10:i + 1]); pl = np.min(l[i - 20:i - 9])
        if ch > ph and cl > pl: st_[i] = 1
        elif ch < ph and cl < pl: st_[i] = -1

    # Breakout (Donchian + BB squeeze)
    bkl = np.zeros(n); bks = np.zeros(n)
    for i in range(2, n):
        if np.isnan(dcu[i - 1]) or np.isnan(bbw[i]): continue
        sq = (not np.isnan(bbw[i - 1])) and bbw[i - 1] < 3.0 and bbw[i] > bbw[i - 1]
        if c[i] > dcu[i - 1] and (sq or c[i] > bbu[i]): bkl[i] = 1
        if c[i] < dcl[i - 1] and (sq or c[i] < bbl[i]): bks[i] = 1

    # ── Additional enriched indicators ──
    # ADX (Average Directional Index) — trend strength
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

    # VWAP proxy (volume-weighted average price, rolling 20 bars)
    vwap = np.full(n, np.nan, dtype=np.float64)
    vol = df["tick_volume"].values.astype(np.float64) if "tick_volume" in df.columns else np.ones(n)
    for j in range(20, n):
        pv = np.sum(c[j-20:j] * vol[j-20:j])
        sv = np.sum(vol[j-20:j])
        vwap[j] = pv / sv if sv > 0 else c[j]

    # Volume SMA for surge detection
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    for j in range(20, n):
        vol_sma[j] = np.mean(vol[j-20:j])

    # 3-bar momentum (rate of change)
    roc3 = np.zeros(n, dtype=np.float64)
    for j in range(3, n):
        if c[j-3] > 0: roc3[j] = (c[j] - c[j-3]) / c[j-3] * 100

    # Consecutive candle count (same direction)
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
    """Industry-grade multi-indicator scoring (max ~14 points per side).

    Each component is enriched beyond simple binary checks:
    - EMA: checks stack order + separation strength + VWAP alignment
    - SuperTrend: checks direction + distance (far = stronger)
    - MACD: checks crossover + histogram acceleration + divergence
    - RSI: checks zone + momentum direction + divergence
    - Patterns: checks candle quality + body/wick ratio + volume confirmation
    - Heikin Ashi: checks trend + consecutive bars
    - Structure: checks swing consistency + ADX trend strength
    - Breakout: checks channel break + squeeze release + volume surge
    """
    sl = ss = 0.0
    p = ind["c"][i]; o = ind["o"][i]; h = ind["h"][i]; l = ind["l"][i]
    atr = ind["at"][i] if not np.isnan(ind["at"][i]) else 1.0

    # ── 1. EMA STACK (0-1.5) ──
    # Base: price vs trend + short vs long
    if p > ind["et"][i] and ind["es"][i] > ind["el"][i]: sl += 0.5
    if p < ind["et"][i] and ind["es"][i] < ind["el"][i]: ss += 0.5
    # Bonus: EMA separation strength (distance between short & long > 0.5 ATR)
    ema_sep = abs(ind["es"][i] - ind["el"][i])
    if ema_sep > 0.5 * atr:
        if ind["es"][i] > ind["el"][i]: sl += 0.25
        else: ss += 0.25
    # Bonus: price above/below VWAP confirms institutional direction
    if not np.isnan(ind["vwap"][i]):
        if p > ind["vwap"][i] and ind["es"][i] > ind["el"][i]: sl += 0.25
        if p < ind["vwap"][i] and ind["es"][i] < ind["el"][i]: ss += 0.25
    # Bonus: all 3 EMAs in order (short > long > trend for bull)
    if ind["es"][i] > ind["el"][i] > ind["et"][i]: sl += 0.5
    if ind["es"][i] < ind["el"][i] < ind["et"][i]: ss += 0.5

    # ── 2. SUPERTREND (0-1.5) ──
    if not np.isnan(ind["stl"][i]):
        if p > ind["stl"][i]:
            sl += 0.5
            # Bonus: further from SuperTrend = stronger trend
            st_dist = (p - ind["stl"][i]) / atr
            if st_dist > 1.0: sl += 0.5
            if st_dist > 2.0: sl += 0.5
        if p < ind["stl"][i]:
            ss += 0.5
            st_dist = (ind["stl"][i] - p) / atr
            if st_dist > 1.0: ss += 0.5
            if st_dist > 2.0: ss += 0.5

    # ── 3. MACD LINE vs SIGNAL (0-1.5) ──
    if ind["ml"][i] > ind["ms"][i]:
        sl += 0.5
        # Bonus: MACD line accelerating away from signal
        if i > 1 and (ind["ml"][i] - ind["ms"][i]) > (ind["ml"][i-1] - ind["ms"][i-1]):
            sl += 0.5
    if ind["ml"][i] < ind["ms"][i]:
        ss += 0.5
        if i > 1 and (ind["ms"][i] - ind["ml"][i]) > (ind["ms"][i-1] - ind["ml"][i-1]):
            ss += 0.5
    # Bonus: fresh crossover (just crossed in last 3 bars)
    if i > 3:
        for k in range(1, 4):
            if ind["ml"][i-k] <= ind["ms"][i-k] and ind["ml"][i] > ind["ms"][i]:
                sl += 0.5; break
            if ind["ml"][i-k] >= ind["ms"][i-k] and ind["ml"][i] < ind["ms"][i]:
                ss += 0.5; break

    # ── 4. MACD HISTOGRAM (0-1.0) ──
    if ind["mh"][i] > 0: sl += 0.25
    if ind["mh"][i] < 0: ss += 0.25
    # Bonus: histogram increasing (accelerating momentum)
    if i > 1:
        if ind["mh"][i] > 0 and ind["mh"][i] > ind["mh"][i-1]: sl += 0.25
        if ind["mh"][i] < 0 and ind["mh"][i] < ind["mh"][i-1]: ss += 0.25
    # Bonus: histogram above/below its own 5-bar average
    if i > 5:
        mh_avg = np.mean([ind["mh"][i-j] for j in range(5)])
        if ind["mh"][i] > mh_avg and ind["mh"][i] > 0: sl += 0.25
        if ind["mh"][i] < mh_avg and ind["mh"][i] < 0: ss += 0.25
    # Penalty: histogram shrinking (momentum fading)
    if i > 1:
        if ind["mh"][i] > 0 and ind["mh"][i] < ind["mh"][i-1] * 0.7: sl -= 0.25
        if ind["mh"][i] < 0 and ind["mh"][i] > ind["mh"][i-1] * 0.7: ss -= 0.25

    # ── 5. RSI (0-1.0) ──
    rsi = ind["rs"][i]
    if not np.isnan(rsi):
        # Sweet spot: 45-65 for longs (trending up, not overbought)
        if 45 < rsi < 65: sl += 0.25
        if 35 < rsi < 55: ss += 0.25
        # Bonus: RSI rising for longs, falling for shorts
        if i > 1 and not np.isnan(ind["rs"][i-1]):
            if rsi > ind["rs"][i-1] and rsi < 70: sl += 0.25
            if rsi < ind["rs"][i-1] and rsi > 30: ss += 0.25
        # Bonus: RSI pulled back from extreme (reset = new move)
        if i > 5:
            rsi_5ago = ind["rs"][i-5]
            if not np.isnan(rsi_5ago):
                if rsi_5ago > 70 and 45 < rsi < 65: sl += 0.25  # pulled back from OB
                if rsi_5ago < 30 and 35 < rsi < 55: ss += 0.25  # pulled back from OS
        # Penalty: overbought/oversold in trade direction
        if rsi > 75: sl -= 0.25
        if rsi < 25: ss -= 0.25

    # ── 6. CANDLESTICK PATTERNS (0-2.0) ──
    body = abs(p - o)
    full_range = h - l if h > l else 0.001
    body_ratio = body / full_range
    # Engulfing (strong: body > 60% of range)
    if ind["be"][i]:
        sl += 1.0
        if body_ratio > 0.6: sl += 0.5  # strong body
    if ind["se"][i]:
        ss += 1.0
        if body_ratio > 0.6: ss += 0.5
    # Pin bar (bonus: long wick > 2x body)
    if ind["bp"][i]:
        sl += 0.5
        lower_wick = min(p, o) - l
        if lower_wick > body * 2.5: sl += 0.5  # very long wick
    if ind["sp"][i]:
        ss += 0.5
        upper_wick = h - max(p, o)
        if upper_wick > body * 2.5: ss += 0.5
    # Volume confirmation on pattern
    if not np.isnan(ind["vol_sma"][i]) and ind["vol_sma"][i] > 0:
        vol_ratio = ind["vol"][i] / ind["vol_sma"][i]
        if (ind["be"][i] or ind["bp"][i]) and vol_ratio > 1.3: sl += 0.5
        if (ind["se"][i] or ind["sp"][i]) and vol_ratio > 1.3: ss += 0.5

    # ── 7. HEIKIN ASHI (0-1.0) ──
    if ind["hac"][i] > ind["hao"][i]: sl += 0.25
    if ind["hac"][i] < ind["hao"][i]: ss += 0.25
    # Bonus: consecutive HA bars in same direction
    if i > 2:
        ha_streak_bull = all(ind["hac"][i-k] > ind["hao"][i-k] for k in range(3))
        ha_streak_bear = all(ind["hac"][i-k] < ind["hao"][i-k] for k in range(3))
        if ha_streak_bull: sl += 0.5
        if ha_streak_bear: ss += 0.5
    # Bonus: HA body expanding (momentum building)
    if i > 1:
        ha_body_now = abs(ind["hac"][i] - ind["hao"][i])
        ha_body_prev = abs(ind["hac"][i-1] - ind["hao"][i-1])
        if ha_body_now > ha_body_prev * 1.2 and ind["hac"][i] > ind["hao"][i]: sl += 0.25
        if ha_body_now > ha_body_prev * 1.2 and ind["hac"][i] < ind["hao"][i]: ss += 0.25

    # ── 8. STRUCTURE TREND (0-1.5) ──
    if ind["st"][i] == 1: sl += 0.5
    if ind["st"][i] == -1: ss += 0.5
    # Bonus: ADX confirms trend strength
    if not np.isnan(ind["adx"][i]):
        if ind["adx"][i] > 25:  # strong trend
            if ind["st"][i] == 1: sl += 0.5
            if ind["st"][i] == -1: ss += 0.5
        if ind["adx"][i] > 40:  # very strong trend
            if ind["di_plus"][i] > ind["di_minus"][i]: sl += 0.5
            if ind["di_minus"][i] > ind["di_plus"][i]: ss += 0.5
    # Penalty: no structure + low ADX = choppy market
    if ind["st"][i] == 0 and not np.isnan(ind["adx"][i]) and ind["adx"][i] < 20:
        sl -= 0.25; ss -= 0.25

    # ── 9. BREAKOUT (0-2.5) ──
    if ind["bkl"][i]:
        sl += 1.0
        # Bonus: BB squeeze before breakout (compression → explosion)
        if not np.isnan(ind["bbw"][i]) and i > 1 and not np.isnan(ind["bbw"][i-1]):
            if ind["bbw"][i-1] < 2.0: sl += 0.5  # tight squeeze
        # Bonus: broke above Donchian by > 0.5 ATR
        if not np.isnan(ind["dcu"][i-1]):
            break_dist = (p - ind["dcu"][i-1]) / atr
            if break_dist > 0.5: sl += 0.5
        # Volume surge on breakout
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

    # ── 10. MOMENTUM VELOCITY (0-0.5) — 3-bar rate of change ──
    if ind["roc3"][i] > 0.3: sl += 0.25   # strong 3-bar up move
    if ind["roc3"][i] < -0.3: ss += 0.25
    if ind["roc3"][i] > 0.6: sl += 0.25   # very strong
    if ind["roc3"][i] < -0.6: ss += 0.25

    # ── 11. TREND PERSISTENCE (0-0.5) — consecutive candles ──
    if ind["consec"][i] >= 3: sl += 0.25    # 3+ bullish candles in a row
    if ind["consec"][i] <= -3: ss += 0.25
    if ind["consec"][i] >= 5: sl += 0.25    # 5+ very persistent
    if ind["consec"][i] <= -5: ss += 0.25

    # Clamp negatives
    sl = max(0, sl)
    ss = max(0, ss)

    return sl, ss


def generate(mt5, sym: str, cfg: SymbolConfig, df: pd.DataFrame, regime: str,
             intelligence: dict = None) -> list:
    """Generate signals using full multi-indicator scoring + intelligence gates.

    intelligence dict (optional, from agent coordinator):
        vol_profile: VolumeProfile dataclass
        ofi_z: float (-3 to +3)
        tf_analysis: TFAnalysis dataclass (h4_trend, h1_trend, alignment, strength)
        patterns: PatternResult dataclass
        cross_asset: float (-1 to +1)
        session_mult: float (0.1 to 1.3)
    """
    signals = []
    if df is None or len(df) < 100:
        return signals

    si = mt5.symbol_info(sym)
    if si is None:
        return signals
    pt = si.point

    # Get indicator config for this symbol
    icfg = dict(IND_DEFAULTS)
    icfg.update(IND_OVERRIDES.get(sym, {}))

    # Compute all indicators
    ind = _compute_indicators(df, icfg)
    n = ind["n"]
    bi = n - 2  # completed bar

    if bi < 21 or np.isnan(ind["at"][bi]) or ind["at"][bi] == 0:
        return signals

    # ═══ MULTI-INDICATOR SCORING ═══
    long_score, short_score = _score(ind, bi)

    buy = long_score >= MIN_SCORE
    sell = cfg.allow_sell and short_score >= MIN_SCORE

    if not buy and not sell:
        return signals

    direction = "long" if (buy and (not sell or long_score >= short_score)) else "short"

    # ═══ INTELLIGENCE GATES — reject bad trades before entry ═══
    if intelligence:
        # Gate 1: H4 trend must not oppose direction
        tf = intelligence.get("tf_analysis")
        if tf and hasattr(tf, "h4_trend"):
            h4 = int(tf.h4_trend)
            if direction == "long" and h4 == -1:
                return signals  # don't buy against H4 downtrend
            if direction == "short" and h4 == 1:
                return signals  # don't sell against H4 uptrend

        # Gate 2: OFI must not strongly oppose direction
        ofi_z = intelligence.get("ofi_z", 0)
        if direction == "long" and ofi_z < -1.5:
            return signals  # institutional flow is selling
        if direction == "short" and ofi_z > 1.5:
            return signals  # institutional flow is buying

        # Gate 3: Breakout needs volume confirmation
        is_breakout = bool(ind["bkl"][bi]) if direction == "long" else bool(ind["bks"][bi])
        vp = intelligence.get("vol_profile")
        if is_breakout and vp and hasattr(vp, "surge"):
            if not vp.surge and not vp.vol_increasing:
                return signals  # false breakout — no volume

        # Gate 4: Strict ranging regime needs pattern/breakout confirmation
        # (low_vol is OK — it just means quiet, not necessarily ranging)
        if regime == "ranging":
            has_pattern = bool(ind["be"][bi] or ind["se"][bi] or ind["bp"][bi] or ind["sp"][bi])
            has_breakout = bool(ind["bkl"][bi] or ind["bks"][bi])
            if not has_pattern and not has_breakout:
                return signals  # momentum in ranging without confirmation

    # undo direction pick — let the rest of the function handle it
    buy = long_score >= MIN_SCORE
    sell = cfg.allow_sell and short_score >= MIN_SCORE

    direction = "long" if (buy and (not sell or long_score >= short_score)) else "short"
    raw_score = long_score if direction == "long" else short_score

    # Normalize to 0-100 for confidence
    score_pct = min(100, raw_score / 11.0 * 100)

    # ═══ DYNAMIC ATR-BASED SL ═══
    # Cast all numpy values to plain Python float for rpyc bridge compatibility
    atr_val = float(ind["at"][bi])
    sl_atr_mult, tp1_r, tp2_r, trail_r = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)

    # SuperTrend as SL reference (proven in sim.py)
    st_level = float(ind["stl"][bi]) if not np.isnan(ind["stl"][bi]) else None

    # ATR-based SL distance
    sl_dist_atr = float(atr_val * sl_atr_mult)
    # Safety cap from config
    sl_dist_cap = float(cfg.sl_points * pt)
    sl_dist = float(min(sl_dist_atr, sl_dist_cap))

    # Ensure minimum SL = 3× ATR (H1 bars move 2-3× ATR intrabar in live)
    min_sl_dist = float(atr_val * 3.0)
    sl_dist = float(max(sl_dist, min_sl_dist))

    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        return signals

    is_breakout = bool(ind["bkl"][bi]) if direction == "long" else bool(ind["bks"][bi])
    # Breakout bonus on TP (from sim.py)
    brk_bonus = float(sl_dist * 0.5) if is_breakout else 0.0

    if direction == "long":
        entry = float(tick.ask)
        if st_level and st_level < entry:
            st_sl_dist = entry - st_level
            if st_sl_dist < sl_dist_atr:
                sl_dist = float(max(st_sl_dist, min_sl_dist))
        sl = float(entry - sl_dist)
        tp1 = float(entry + tp1_r * sl_dist + brk_bonus)
        tp2 = float(entry + tp2_r * sl_dist + brk_bonus)
        trail = float(entry + trail_r * sl_dist + brk_bonus)
    else:
        entry = float(tick.bid)
        if st_level and st_level > entry:
            st_sl_dist = st_level - entry
            if st_sl_dist < sl_dist_atr:
                sl_dist = float(max(st_sl_dist, min_sl_dist))
        sl = float(entry + sl_dist)
        tp1 = float(entry - tp1_r * sl_dist - brk_bonus)
        tp2 = float(entry - tp2_r * sl_dist - brk_bonus)
        trail = float(entry - trail_r * sl_dist - brk_bonus)

    rr = float(tp1_r)
    sl_dist = float(sl_dist)

    signals.append(CandidateSignal(
        symbol=sym, strategy="momentum", direction=direction,
        timeframe=60, raw_score=float(score_pct), confidence=float(score_pct / 100),
        entry=float(entry), sl=float(sl), tp1=float(tp1), tp2=float(tp2), trail_tp=float(trail),
        sl_dist=sl_dist, rr_ratio=rr, regime=regime,
        metadata={
            "long_score": float(round(long_score, 1)),
            "short_score": float(round(short_score, 1)),
            "atr_val": float(round(atr_val, 5)),
            "sl_atr_mult": float(sl_atr_mult),
            "sl_dist": float(round(sl_dist, 5)),
            "st_level": float(round(st_level, 5)) if st_level else None,
            "is_breakout": bool(is_breakout),
            "regime": regime,
            "tp1_r": float(tp1_r), "tp2_r": float(tp2_r), "trail_r": float(trail_r),
        },
    ))
    return signals
