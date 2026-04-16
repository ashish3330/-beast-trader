"""
Beast Trader — M5 Scalp Scorer.

Reuses _compute_indicators and _score from momentum_scorer.py with
M5-tuned indicator params (faster EMAs, tighter SuperTrend, quicker MACD).
Adds M1 micro-direction check for entry timing.

MIN_SCALP_SCORE = 5.0 (higher bar than swing — scalps need conviction).
"""
import numpy as np
import logging

from signals.momentum_scorer import _compute_indicators, _score, _ema, _supertrend

log = logging.getLogger("beast.scalp_scorer")

# ═══ M5 SCALP INDICATOR PARAMS ═══
SCALP_IND_DEFAULTS = {
    "EMA_S": 8, "EMA_L": 21, "EMA_T": 50,
    "ST_F": 1.5, "ST_ATR": 7,
    "MACD_F": 5, "MACD_SL": 13, "MACD_SIG": 4,
    "ATR_LEN": 10,
}

# Per-symbol overrides for scalping (tighter params)
SCALP_IND_OVERRIDES = {
    "XAUUSD":   {"EMA_S": 8, "EMA_L": 21, "EMA_T": 50, "ST_F": 1.5, "ST_ATR": 7,
                 "MACD_F": 5, "MACD_SL": 13, "MACD_SIG": 4, "ATR_LEN": 10},
    "BTCUSD":   {"EMA_S": 8, "EMA_L": 21, "EMA_T": 50, "ST_F": 1.5, "ST_ATR": 7,
                 "MACD_F": 5, "MACD_SL": 13, "MACD_SIG": 4, "ATR_LEN": 10},
    "NAS100.r": {"EMA_S": 8, "EMA_L": 21, "EMA_T": 50, "ST_F": 1.5, "ST_ATR": 7,
                 "MACD_F": 5, "MACD_SL": 13, "MACD_SIG": 4, "ATR_LEN": 10},
    "GER40.r":  {"EMA_S": 8, "EMA_L": 21, "EMA_T": 50, "ST_F": 1.5, "ST_ATR": 7,
                 "MACD_F": 5, "MACD_SL": 13, "MACD_SIG": 4, "ATR_LEN": 10},
}

MIN_SCALP_SCORE = 5.0


def scalp_compute_indicators(df, symbol=None):
    """Compute indicators on M5 candles using scalp params."""
    icfg = dict(SCALP_IND_DEFAULTS)
    if symbol:
        icfg.update(SCALP_IND_OVERRIDES.get(symbol, {}))
    return _compute_indicators(df, icfg)


def scalp_score(ind, i):
    """Score using the same proven scoring function, but on M5 data with scalp params."""
    return _score(ind, i)


def _m1_micro_direction(state, symbol):
    """
    Check M1 micro-direction for scalp entry timing.
    Uses EMA(3) vs EMA(8) on M1 candles.
    Returns "LONG", "SHORT", or "FLAT".
    """
    m1_df = state.get_candles(symbol, 1)
    if m1_df is None or len(m1_df) < 20:
        return "FLAT"

    try:
        close = m1_df["close"].values.astype(np.float64)
        ema3 = _ema(close, 3)
        ema8 = _ema(close, 8)

        # Last completed bar
        bi = len(close) - 2
        if bi < 1:
            return "FLAT"

        e3 = float(ema3[bi])
        e8 = float(ema8[bi])

        # Also check that the cross is recent (within last 3 M1 bars)
        # and EMA3 is pulling away from EMA8
        if e3 > e8:
            return "LONG"
        elif e3 < e8:
            return "SHORT"
        else:
            return "FLAT"
    except Exception as e:
        log.warning("[%s] M1 micro-direction failed: %s", symbol, e)
        return "FLAT"
