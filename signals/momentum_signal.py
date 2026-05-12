"""
Momentum signal — single source of truth for the four momentum-adaptive
features (size boost, trail tightness, pyramid, MIN_SCORE delta).

Reads existing indicators (ADX, ATR, EMAs) — no new computation. Returns
a normalized score 0-1, a regime class, and a directional bias.

Use:
    from signals.momentum_signal import compute_momentum

    mom = compute_momentum(indicators, candles_h1=df)
    if mom["score"] > 0.7 and mom["regime"] == "TRENDING_HARD":
        # explosive move — boost size, tighten trail, allow pyramid
        ...

Both the live brain and the backtest call this with the same dict shape so
the signal is identical in-sample and live.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("dragon.momentum")

# ── Tunables ────────────────────────────────────────────────────────────
# ADX bands. Standard interpretation:
#   <20 = no trend, 20-25 = developing, 25-50 = strong, >50 = extreme.
ADX_NEUTRAL = 25.0
ADX_STRONG = 35.0
ADX_DEAD = 15.0

# ATR ratio bands. atr_ratio = current ATR / 30-bar ATR mean.
ATR_RATIO_HIGH = 1.3        # vol expansion
ATR_RATIO_LOW = 0.7         # vol contraction
ATR_RATIO_EXTREME = 1.8     # blow-off / event spike

# ROC lookback (in H1 candles). 4h = 4 candles.
ROC_PERIOD_H1 = 4
ROC_NORMALIZE = 2.0         # cap |roc/atr| at this multiple before scaling

# Score weights (must sum to 1.0)
W_ADX = 0.4
W_ROC = 0.3
W_ATR = 0.3


def _safe(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return default
    return float(v)


def _atr_ratio(candles_h1: Optional[pd.DataFrame], current_atr: float) -> float:
    """Current ATR vs 30-bar ATR mean. Returns 1.0 if not computable."""
    if current_atr <= 0:
        return 1.0
    if candles_h1 is None or len(candles_h1) < 30:
        return 1.0
    try:
        # Per-bar ATR proxy: high - low. Cheaper than rolling Wilder.
        hl = (candles_h1["high"] - candles_h1["low"]).tail(30)
        if len(hl) < 30:
            return 1.0
        baseline = float(hl.mean())
        if baseline <= 0:
            return 1.0
        return min(3.0, max(0.1, current_atr / baseline))
    except Exception:
        return 1.0


def _roc_normalized(candles_h1: Optional[pd.DataFrame], atr: float) -> float:
    """4-bar ROC divided by ATR. Magnitude only — direction is separate.
    Returns 0-1 where 1.0 = ROC of 2*ATR over 4 bars (very fast move).
    """
    if candles_h1 is None or len(candles_h1) < ROC_PERIOD_H1 + 1 or atr <= 0:
        return 0.0
    try:
        c = candles_h1["close"]
        roc = float(c.iloc[-1] - c.iloc[-1 - ROC_PERIOD_H1])
        magnitude = abs(roc) / atr
        return min(1.0, magnitude / ROC_NORMALIZE)
    except Exception:
        return 0.0


def _direction_from_emas(indicators: dict) -> str:
    """LONG / SHORT / FLAT from EMA stack alignment."""
    e20 = _safe(indicators, "ema20")
    e50 = _safe(indicators, "ema50")
    e200 = _safe(indicators, "ema200")
    if e20 == 0 or e50 == 0 or e200 == 0:
        return "FLAT"
    if e20 > e50 > e200:
        return "LONG"
    if e20 < e50 < e200:
        return "SHORT"
    return "FLAT"


def compute_momentum(
    indicators: dict,
    candles_h1: Optional[pd.DataFrame] = None,
) -> dict:
    """Return {score, regime, direction, components} for a symbol.

    Args:
        indicators: dict with at least adx, atr, ema20/50/200.
        candles_h1: optional H1 DataFrame for ROC + ATR-ratio. If None,
            those components are conservative (assume neutral).

    Returns:
        dict with keys:
            score:     0-1 momentum strength (regime-blind magnitude)
            regime:    TRENDING_HARD | TRENDING_SOFT | RANGING | VOLATILE | DEAD
            direction: LONG | SHORT | FLAT
            components: {adx_n, roc_n, atr_ratio} for debugging
    """
    if not isinstance(indicators, dict):
        return {"score": 0.0, "regime": "DEAD", "direction": "FLAT", "components": {}}

    adx = _safe(indicators, "adx", default=ADX_NEUTRAL)
    atr = _safe(indicators, "atr")

    # ── Component 1: ADX normalized to 0-1 ──
    # 0 at ADX_DEAD, 1.0 at ADX_STRONG, linear between.
    adx_n = max(0.0, min(1.0, (adx - ADX_DEAD) / (ADX_STRONG - ADX_DEAD)))

    # ── Component 2: ROC magnitude normalized by ATR ──
    roc_n = _roc_normalized(candles_h1, atr)

    # ── Component 3: ATR-ratio normalized ──
    atr_ratio = _atr_ratio(candles_h1, atr)
    # Map atr_ratio to 0-1 with peak at ATR_RATIO_HIGH, decay above EXTREME.
    if atr_ratio < ATR_RATIO_LOW:
        atr_n = 0.0
    elif atr_ratio < ATR_RATIO_HIGH:
        atr_n = (atr_ratio - ATR_RATIO_LOW) / (ATR_RATIO_HIGH - ATR_RATIO_LOW)
    elif atr_ratio < ATR_RATIO_EXTREME:
        atr_n = 1.0
    else:
        # extreme volatility — likely blow-off or event spike, score back down.
        atr_n = max(0.0, 1.0 - (atr_ratio - ATR_RATIO_EXTREME) / 1.0)

    score = W_ADX * adx_n + W_ROC * roc_n + W_ATR * atr_n

    # ── Regime classification ──
    if adx >= ADX_STRONG and atr_ratio >= ATR_RATIO_HIGH:
        regime = "TRENDING_HARD"
    elif adx >= ADX_NEUTRAL:
        regime = "TRENDING_SOFT"
    elif adx <= ADX_DEAD and atr_ratio <= ATR_RATIO_LOW:
        regime = "DEAD"
    elif atr_ratio >= ATR_RATIO_EXTREME:
        regime = "VOLATILE"
    else:
        regime = "RANGING"

    direction = _direction_from_emas(indicators)

    return {
        "score": round(score, 3),
        "regime": regime,
        "direction": direction,
        "components": {
            "adx": round(adx, 1),
            "atr_ratio": round(atr_ratio, 2),
            "adx_n": round(adx_n, 3),
            "roc_n": round(roc_n, 3),
            "atr_n": round(atr_n, 3),
        },
    }


# ── Convenience helpers used by the 4 features ─────────────────────────

def size_multiplier(mom: dict, signal_direction: str) -> float:
    """Feature 1: size up on confirmed momentum aligned with the entry signal."""
    if mom["regime"] == "TRENDING_HARD" and mom["direction"] == signal_direction:
        return 1.30
    if mom["regime"] == "TRENDING_SOFT" and mom["direction"] == signal_direction:
        return 1.15
    if mom["regime"] in ("DEAD", "VOLATILE"):
        return 0.70
    return 1.0


def trail_multiplier(mom: dict) -> float:
    """Feature 2 (v2): WIDER trail when momentum strong (let winners run),
    tighter when weak (lock fast on fake bursts).

    Earlier version was reversed — locked the burst at peak and got stopped
    on minor retraces, defeating the trend-follower logic. Live evidence
    (ETHUSD trades 2026-05-10): trades hit ~0.5R, BE locked, retrace closed
    at +0.05R. Wider trail in high momentum should let those run to 2-3R.
    """
    if mom["score"] >= 0.7:
        return 1.5   # widen — let the trend extend
    if mom["score"] <= 0.3:
        return 0.8   # tighten — fake burst, lock fast
    return 1.0


def sl_multiplier(mom: dict) -> float:
    """Adaptive initial SL distance. HIGH momentum gets wider SL (less
    likely to get stopped on noise inside the trend). LOW momentum gets
    tighter SL (capital efficiency on dud setups)."""
    if mom["score"] >= 0.7:
        return 1.3   # wider initial stop
    if mom["score"] <= 0.3:
        return 0.85  # tighter — give it less room since edge is weak
    return 1.0


def lock_threshold_mult(mom: dict) -> float:
    """Delay BE/lock thresholds when momentum strong.

    2026-05-12: tried reducing 1.5→1.2 because live showed BE missed on
    0.6-0.7R retraces. But v4 walk-forward came in at -4.3% vs baseline —
    the 1.5x lock delay IS the edge. Reverting. Live losses today were
    partly slippage on wide-spread Asian-session symbols (JPN225ft, etc),
    not the lock mult. Keep 1.5x.
    """
    if mom["score"] >= 0.7:
        return 1.5
    if mom["score"] <= 0.3:
        return 0.8
    return 1.0


def pyramid_allowed(mom: dict, position_direction: str) -> bool:
    """Feature 3: pyramid only when strong and still aligned."""
    return (
        mom["regime"] in ("TRENDING_HARD", "TRENDING_SOFT")
        and mom["direction"] == position_direction
        and mom["score"] >= 0.65
    )


def compute_momentum_at_bar(ind: dict, bi: int) -> dict:
    """Backtest-mode momentum compute at a specific bar index.

    Uses the v5_backtest indicator-dict layout: arrays keyed by short codes
    (`at`, `adx`, `es`, `el`, `et`, `c`). Returns the same shape as
    compute_momentum() so the helper functions size_multiplier() etc. work
    identically in both paths.
    """
    if bi < 30:
        return {"score": 0.0, "regime": "DEAD", "direction": "FLAT", "components": {}}

    try:
        adx = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else ADX_NEUTRAL
        atr = float(ind["at"][bi]) if not np.isnan(ind["at"][bi]) else 0.0
        c = ind["c"]
        es = ind["es"][bi]
        el = ind["el"][bi]
        et = ind["et"][bi]
    except (KeyError, IndexError, TypeError):
        return {"score": 0.0, "regime": "DEAD", "direction": "FLAT", "components": {}}

    # ROC magnitude over last 4 bars (H1)
    if atr > 0 and bi >= ROC_PERIOD_H1:
        roc = abs(float(c[bi] - c[bi - ROC_PERIOD_H1])) / atr
        roc_n = min(1.0, roc / ROC_NORMALIZE)
    else:
        roc_n = 0.0

    # ATR ratio: current ATR vs 30-bar ATR mean
    if atr > 0 and bi >= 30:
        try:
            atr_window = ind["at"][bi - 29 : bi + 1]
            atr_mean = float(np.nanmean(atr_window))
            atr_ratio = atr / atr_mean if atr_mean > 0 else 1.0
        except Exception:
            atr_ratio = 1.0
    else:
        atr_ratio = 1.0

    adx_n = max(0.0, min(1.0, (adx - ADX_DEAD) / (ADX_STRONG - ADX_DEAD)))
    if atr_ratio < ATR_RATIO_LOW:
        atr_n = 0.0
    elif atr_ratio < ATR_RATIO_HIGH:
        atr_n = (atr_ratio - ATR_RATIO_LOW) / (ATR_RATIO_HIGH - ATR_RATIO_LOW)
    elif atr_ratio < ATR_RATIO_EXTREME:
        atr_n = 1.0
    else:
        atr_n = max(0.0, 1.0 - (atr_ratio - ATR_RATIO_EXTREME) / 1.0)

    score = W_ADX * adx_n + W_ROC * roc_n + W_ATR * atr_n

    if adx >= ADX_STRONG and atr_ratio >= ATR_RATIO_HIGH:
        regime = "TRENDING_HARD"
    elif adx >= ADX_NEUTRAL:
        regime = "TRENDING_SOFT"
    elif adx <= ADX_DEAD and atr_ratio <= ATR_RATIO_LOW:
        regime = "DEAD"
    elif atr_ratio >= ATR_RATIO_EXTREME:
        regime = "VOLATILE"
    else:
        regime = "RANGING"

    # EMA-stack direction. Backtest uses es/el/et as short/long/trend EMA.
    if es > el > et:
        direction = "LONG"
    elif es < el < et:
        direction = "SHORT"
    else:
        direction = "FLAT"

    return {
        "score": round(score, 3),
        "regime": regime,
        "direction": direction,
        "components": {
            "adx": round(adx, 1),
            "atr_ratio": round(atr_ratio, 2),
        },
    }


def min_score_delta(mom: dict) -> float:
    """Feature 4: relax/tighten MIN_SCORE based on regime.

    Floor enforced by caller: never let MIN_SCORE drop below 6.0.
    """
    if mom["regime"] == "TRENDING_HARD":
        return -0.5
    if mom["regime"] in ("RANGING", "DEAD"):
        return +1.0
    if mom["regime"] == "VOLATILE":
        return +0.5
    return 0.0
