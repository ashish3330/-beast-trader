"""
Mean-Reversion Scorer — fires when momentum is flat and price is at extremes.

Designed to complement momentum strategy:
- Momentum profits in trending markets
- Mean-reversion profits in ranging/low-vol markets
- Together: always have something working → higher trade frequency

Score components (0-10 max):
1. RSI extreme (0-2): oversold/overbought reversal signal
2. Bollinger Band touch (0-2): price at statistical extreme
3. Volume exhaustion (0-2): high volume at extreme = capitulation
4. Consecutive candles (0-2): extended streak = reversion due
5. Distance from EMA (0-2): price far from mean = rubber band effect

REGIME FILTER: ADX > 30 halves all scores (don't fight strong trends).
"""
import numpy as np


def score(ind, i):
    """Returns (long_score, short_score) for mean-reversion strategy.

    Args:
        ind: dict of numpy arrays from _compute_indicators()
        i: bar index to evaluate

    Returns:
        (float, float): long and short mean-reversion scores (0-10)
    """
    sl = ss = 0.0
    p = ind["c"][i]
    atr = ind["at"][i] if not np.isnan(ind["at"][i]) else 1e-10

    # 1. RSI EXTREME (0-2)
    rsi = ind["rs"][i]
    if not np.isnan(rsi):
        if rsi < 25: sl += 2.0
        elif rsi < 30: sl += 1.0
        elif rsi < 35: sl += 0.5
        if rsi > 75: ss += 2.0
        elif rsi > 70: ss += 1.0
        elif rsi > 65: ss += 0.5

    # 2. BOLLINGER BAND TOUCH (0-2)
    bbu = ind["bbu"][i]
    bbl = ind["bbl"][i]
    if not np.isnan(bbl) and not np.isnan(bbu):
        if p < bbl:
            dist_below = (bbl - p) / atr
            sl += min(2.0, 1.0 + dist_below * 0.5)
        if p > bbu:
            dist_above = (p - bbu) / atr
            ss += min(2.0, 1.0 + dist_above * 0.5)

    # 3. VOLUME EXHAUSTION (0-2)
    vol = ind["vol"][i]
    vol_avg = ind["vol_sma"][i]
    if not np.isnan(vol_avg) and vol_avg > 0:
        vol_ratio = vol / vol_avg
        price_extreme_long = (not np.isnan(rsi) and rsi < 35) or (not np.isnan(bbl) and p <= bbl)
        price_extreme_short = (not np.isnan(rsi) and rsi > 65) or (not np.isnan(bbu) and p >= bbu)
        if vol_ratio > 2.0:
            if price_extreme_long: sl += 2.0
            if price_extreme_short: ss += 2.0
        elif vol_ratio > 1.5:
            if price_extreme_long: sl += 1.0
            if price_extreme_short: ss += 1.0

    # 4. CONSECUTIVE CANDLES (0-2)
    consec = ind["consec"][i]
    if consec <= -7: sl += 2.0
    elif consec <= -5: sl += 1.0
    elif consec <= -3: sl += 0.5
    if consec >= 7: ss += 2.0
    elif consec >= 5: ss += 1.0
    elif consec >= 3: ss += 0.5

    # 5. DISTANCE FROM EMA (0-2)
    ema = ind["es"][i]
    if not np.isnan(ema):
        dist_from_ema = (p - ema) / atr
        if dist_from_ema < -2.0: sl += 2.0
        elif dist_from_ema < -1.5: sl += 1.0
        elif dist_from_ema < -1.0: sl += 0.5
        if dist_from_ema > 2.0: ss += 2.0
        elif dist_from_ema > 1.5: ss += 1.0
        elif dist_from_ema > 1.0: ss += 0.5

    # REGIME FILTER: strong trend = halve scores
    adx = ind["adx"][i]
    if not np.isnan(adx) and adx > 30:
        sl *= 0.5
        ss *= 0.5

    return min(10.0, max(0.0, sl)), min(10.0, max(0.0, ss))
