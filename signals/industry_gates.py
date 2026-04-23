"""
Industry-grade entry gates — shared between backtest and live brain.
Ported from backtest/dragon_backtest_v2.py (2026-04-20 audit).

HARD gates: binary block (Elder opposing, squeeze wrong dir)
SOFT gates: quality scoring (ER, z-score, ROC, squeeze, fresh mom, regime composite)
"""
import numpy as np
from signals.momentum_scorer import _ema, _macd, _atr


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR FUNCTIONS (vectorized, computed once per candle set)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_keltner(close, high, low, period=20, mult=1.5):
    ema = _ema(close, period)
    atr = _atr(high, low, close, period)
    return ema + mult * atr, ema - mult * atr


def compute_ttm_squeeze(close, high, low, bb_period=20, bb_mult=2.0,
                        kc_period=20, kc_mult=1.5):
    n = len(close)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    for i in range(bb_period - 1, n):
        win = close[i - bb_period + 1:i + 1]
        m = np.mean(win)
        s = np.std(win)
        bb_upper[i] = m + bb_mult * s
        bb_lower[i] = m - bb_mult * s

    kc_upper, kc_lower = compute_keltner(close, high, low, kc_period, kc_mult)

    squeeze_on = np.zeros(n, dtype=bool)
    for i in range(max(bb_period, kc_period), n):
        if not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i]):
            squeeze_on[i] = (bb_lower[i] > kc_lower[i]) and (bb_upper[i] < kc_upper[i])

    sma20 = np.full(n, np.nan)
    for i in range(19, n):
        sma20[i] = np.mean(close[i-19:i+1])

    donch_mid = np.full(n, np.nan)
    for i in range(19, n):
        donch_mid[i] = (np.max(high[i-19:i+1]) + np.min(low[i-19:i+1])) / 2

    delta = np.full(n, np.nan)
    for i in range(19, n):
        if not np.isnan(donch_mid[i]) and not np.isnan(sma20[i]):
            delta[i] = close[i] - (donch_mid[i] + sma20[i]) / 2

    momentum = np.full(n, np.nan)
    for i in range(38, n):
        y = delta[i-19:i+1]
        if np.any(np.isnan(y)):
            continue
        x = np.arange(20)
        slope = np.polyfit(x, y, 1)[0]
        momentum[i] = slope * 10 + y[-1]

    return squeeze_on, momentum


def compute_efficiency_ratio(close, period=10):
    n = len(close)
    er = np.full(n, np.nan)
    for i in range(period, n):
        direction = abs(close[i] - close[i - period])
        volatility = sum(abs(close[j] - close[j-1]) for j in range(i - period + 1, i + 1))
        er[i] = direction / volatility if volatility > 0 else 0
    return er


def compute_choppiness(high, low, close, period=14):
    n = len(close)
    chop = np.full(n, np.nan)
    atr1 = np.zeros(n)
    for i in range(1, n):
        atr1[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr1[0] = high[0] - low[0]

    for i in range(period, n):
        atr_sum = np.sum(atr1[i - period + 1:i + 1])
        hi = np.max(high[i - period + 1:i + 1])
        lo = np.min(low[i - period + 1:i + 1])
        hl_range = hi - lo
        if hl_range > 0:
            chop[i] = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
    return chop


def compute_zscore(close, lookback=100):
    n = len(close)
    zscore = np.full(n, np.nan)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 0 else 0

    for i in range(lookback, n):
        win = returns[i - lookback + 1:i + 1]
        mu = np.mean(win)
        sigma = np.std(win)
        if sigma > 0:
            zscore[i] = (returns[i] - mu) / sigma
    return zscore


def compute_dual_roc(close, fast_period=12, slow_period=21):
    n = len(close)
    roc_fast = np.full(n, np.nan)
    roc_slow = np.full(n, np.nan)
    for i in range(fast_period, n):
        if close[i - fast_period] > 0:
            roc_fast[i] = (close[i] - close[i - fast_period]) / close[i - fast_period] * 100
    for i in range(slow_period, n):
        if close[i - slow_period] > 0:
            roc_slow[i] = (close[i] - close[i - slow_period]) / close[i - slow_period] * 100
    return roc_fast, roc_slow


def compute_elder_impulse(close, period=13):
    n = len(close)
    ema13 = _ema(close, period)
    _, _, macd_hist = _macd(close, 12, 26, 9)
    impulse = np.zeros(n, dtype=np.int32)

    for i in range(1, n):
        ema_rising = ema13[i] > ema13[i-1]
        ema_falling = ema13[i] < ema13[i-1]
        hist_rising = macd_hist[i] > macd_hist[i-1]
        hist_falling = macd_hist[i] < macd_hist[i-1]

        if ema_rising and hist_rising:
            impulse[i] = 1
        elif ema_falling and hist_falling:
            impulse[i] = -1

    return impulse


def compute_regime_composite(adx, chop, er, bbw):
    n = len(adx)
    regime_score = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        s = 0
        if not np.isnan(adx[i]) and adx[i] > 25:
            s += 1
        if not np.isnan(chop[i]) and chop[i] < 50:
            s += 1
        if not np.isnan(er[i]) and er[i] > 0.4:
            s += 1
        if not np.isnan(bbw[i]) and 1.5 <= bbw[i] < 4.0:
            s += 1
        regime_score[i] = s
    return regime_score


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL GATE INDICATORS (call once per symbol per candle update)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gate_indicators(ind):
    """Compute all gate indicators from momentum_scorer indicator dict.
    Returns dict with: er, zscore, roc_fast, roc_slow, impulse, squeeze_on, squeeze_mom, regime_comp.
    """
    close = ind["c"]
    high = ind["h"]
    low = ind["l"]

    squeeze_on, squeeze_mom = compute_ttm_squeeze(close, high, low)
    er = compute_efficiency_ratio(close, 10)
    chop = compute_choppiness(high, low, close, 14)
    zscore = compute_zscore(close, 100)
    roc_fast, roc_slow = compute_dual_roc(close, 12, 21)
    impulse = compute_elder_impulse(close, 13)
    regime_comp = compute_regime_composite(ind["adx"], chop, er, ind["bbw"])

    return {
        "er": er, "zscore": zscore, "roc_fast": roc_fast, "roc_slow": roc_slow,
        "impulse": impulse, "squeeze_on": squeeze_on, "squeeze_mom": squeeze_mom,
        "regime_comp": regime_comp,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY GATE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_entry_gates(i, direction, score, ind, extras, category="Forex"):
    """
    Industry-grade entry gate evaluation. Returns (pass, reason, quality_ratio) tuple.
    Direction: 1=LONG, -1=SHORT (or "LONG"/"SHORT" strings).

    HARD gates: binary block (Elder opposing, squeeze wrong direction)
    SOFT gates: quality multiplier (ER, z-score, ROC, squeeze, fresh mom, regime composite)
    """
    # Normalize direction to int
    if isinstance(direction, str):
        direction = 1 if direction == "LONG" else -1

    er = extras["er"]
    zscore = extras["zscore"]
    roc_fast = extras["roc_fast"]
    roc_slow = extras["roc_slow"]
    impulse = extras["impulse"]
    squeeze_on = extras["squeeze_on"]
    squeeze_mom = extras["squeeze_mom"]
    regime_comp = extras["regime_comp"]

    is_trending_asset = category in ("Crypto", "Index", "Gold")

    # ═══ HARD GATES (binary block) ═══

    # Hard Gate 1: Elder Impulse — 2+ consecutive opposing bars
    if i > 1 and score < 9.0:
        if direction == 1 and impulse[i] == -1 and impulse[i-1] == -1:
            return False, "elder_red_consecutive", 0
        if direction == -1 and impulse[i] == 1 and impulse[i-1] == 1:
            return False, "elder_green_consecutive", 0

    # Hard Gate 2: Squeeze fired in WRONG direction
    if i > 1 and squeeze_on[i-1] and not squeeze_on[i]:
        if not np.isnan(squeeze_mom[i]):
            if direction == 1 and squeeze_mom[i] < -1.0:
                return False, "squeeze_fire_wrong_dir", 0
            if direction == -1 and squeeze_mom[i] > 1.0:
                return False, "squeeze_fire_wrong_dir", 0

    # ═══ SOFT GATES (quality scoring) ═══
    quality_points = 0.0
    max_points = 0.0

    # Soft 1: Efficiency Ratio (0-2 points)
    max_points += 2.0
    if not np.isnan(er[i]):
        if er[i] >= 0.5: quality_points += 2.0
        elif er[i] >= 0.3: quality_points += 1.5
        elif er[i] >= 0.15: quality_points += 1.0
        elif er[i] >= 0.08: quality_points += 0.5

    # Soft 2: Z-score (0-2 points)
    max_points += 2.0
    if not np.isnan(zscore[i]):
        abs_z = abs(zscore[i])
        z_chasing = (direction == 1 and zscore[i] > 0) or (direction == -1 and zscore[i] < 0)
        if z_chasing:
            if is_trending_asset:
                if abs_z < 2.0: quality_points += 2.0
                elif abs_z < 2.5: quality_points += 1.5
                elif abs_z < 3.0: quality_points += 1.0
                else: quality_points += 0.5
            else:
                if abs_z < 1.5: quality_points += 2.0
                elif abs_z < 2.0: quality_points += 1.5
                elif abs_z < 2.5: quality_points += 0.75
                elif abs_z < 3.0: quality_points += 0.25
        else:
            quality_points += 2.0

    # Soft 3: Dual ROC alignment (0-2 points)
    max_points += 2.0
    if not np.isnan(roc_fast[i]) and not np.isnan(roc_slow[i]):
        roc_confirms = 0
        if direction == 1:
            if roc_fast[i] > 0: roc_confirms += 1
            if roc_slow[i] > 0: roc_confirms += 1
        else:
            if roc_fast[i] < 0: roc_confirms += 1
            if roc_slow[i] < 0: roc_confirms += 1
        quality_points += roc_confirms

    # Soft 4: TTM Squeeze state (0-2 points)
    max_points += 2.0
    if squeeze_on[i]:
        quality_points += 0.5
    else:
        quality_points += 1.5
        if i > 3 and any(squeeze_on[i-k] for k in range(1, 4)):
            if not np.isnan(squeeze_mom[i]):
                if (direction == 1 and squeeze_mom[i] > 0) or (direction == -1 and squeeze_mom[i] < 0):
                    quality_points += 0.5

    # Soft 5: Fresh momentum (0-2 points)
    max_points += 2.0
    rsi = ind["rs"][i] if not np.isnan(ind["rs"][i]) else 50

    macd_fresh = 0
    if i > 1:
        mh_now = ind["mh"][i]; mh_prev = ind["mh"][i-1]
        if direction == 1 and mh_now > mh_prev: macd_fresh = 1
        if direction == -1 and mh_now < mh_prev: macd_fresh = 1

    rsi_sweet = 0
    if direction == 1 and 40 < rsi < 70: rsi_sweet = 1
    if direction == -1 and 30 < rsi < 60: rsi_sweet = 1

    vol_surge = 0
    if not np.isnan(ind["vol_sma"][i]) and ind["vol_sma"][i] > 0:
        if ind["vol"][i] > 1.3 * ind["vol_sma"][i]: vol_surge = 1

    quality_points += min(2.0, macd_fresh + rsi_sweet + vol_surge * 0.5)

    # Soft 6: Regime composite (0-2 points)
    max_points += 2.0
    rc = regime_comp[i]
    quality_points += min(2.0, rc * 0.5)

    # ═══ QUALITY THRESHOLD ═══
    quality_ratio = quality_points / max_points if max_points > 0 else 0

    if score >= 9.0: threshold = 0.50
    elif score >= 8.0: threshold = 0.55
    elif score >= 7.5: threshold = 0.60
    else: threshold = 0.65

    if quality_ratio < threshold:
        return False, "quality_low", quality_ratio

    return True, "ok", quality_ratio
