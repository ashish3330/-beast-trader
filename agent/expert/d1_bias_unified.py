#!/usr/bin/env python3 -B
"""
agent.expert.d1_bias_unified
============================

UNIFIED D1-bias helper. Single source of truth for "is the daily trend up,
down, or neutral?" used as a soft-filter (downsize, NOT reject) across every
Dragon entry path:

  * agent/brain.py            — momentum entries (after risk_pct chain)
  * agent/sweep_reclaim.py    — SR entries (per-symbol)
  * agent/fvg_strategy.py     — FVG retest (already has own D1 bias; the
                                 unified verdict is consulted for parity logs)

Why a separate, light helper (vs the existing
``"D1 Swing-Structure Bias (HH/HL + BOS/CHoCH).py"``)?
  * The structure module enforces a HARD reject on misalignment (full ICT
    structure read). Per ``feedback_no_skip_trades`` the user prefers DOWNSIZE,
    not REJECT, for the secondary D1 alignment check across momentum/SR/FVG.
  * The structure module needs ≥60 closed D1 bars (~2 months). When buffers
    are short (fresh restart, new symbol) we still want SOME D1 read; this
    helper falls back to "available depth" instead of returning NEUTRAL.
  * The brain/FVG paths already use H1-EMA200 as a D1 proxy. This module
    promotes the proxy to a proper D1-resample-then-EMA200, but in the SAME
    direction-of-truth convention so old logs and tunes stay sane.

Public API
----------
    get_d1_bias(symbol, h1_df_or_candles) -> 'LONG' | 'SHORT' | 'NEUTRAL'

Default
-------
Module is INERT unless ``config.D1_BIAS_UNIFIED_ENABLED`` is True. Callers
ALWAYS guard the call site with that flag; this helper itself fails open
(returns 'NEUTRAL') on any data hiccup so it can NEVER block a trade.

Caching
-------
One real computation per (symbol, last-D1-bar-timestamp). Subsequent calls
within the same daily candle return the cached verdict. Cache is process-
local and intentionally simple — no thread-safety overhead because brain
cycles are single-threaded.
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

log = logging.getLogger("dragon.expert.d1_bias_unified")

# Neutral band: within 0.3% of the EMA → NEUTRAL (chop / indifference).
_NEUTRAL_BAND_PCT = 0.003
_EMA_PERIOD = 200

# Cache: symbol -> (last_d1_bar_iso_or_int, verdict)
_BIAS_CACHE: dict[str, Tuple[Any, str]] = {}


def _to_h1_df(h1_df_or_candles):
    """Accept either a pandas DataFrame with OHLC or a list/np array of
    candle dicts. Return a DataFrame with at least a 'close' column and a
    DatetimeIndex (or None on failure)."""
    try:
        import pandas as pd  # lazy import (pandas already used everywhere)
    except Exception:
        return None
    if h1_df_or_candles is None:
        return None
    # Already a DataFrame
    try:
        if hasattr(h1_df_or_candles, "columns"):
            df = h1_df_or_candles
            if "close" not in df.columns:
                return None
            if not hasattr(df.index, "to_pydatetime"):
                # Try common time columns.
                for tcol in ("time", "datetime", "timestamp"):
                    if tcol in df.columns:
                        df = df.copy()
                        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
                        df = df.dropna(subset=[tcol]).set_index(tcol)
                        break
            return df
    except Exception:
        return None
    # List / ndarray of dicts or tuples — coerce.
    try:
        df = pd.DataFrame(list(h1_df_or_candles))
        if "close" not in df.columns:
            return None
        for tcol in ("time", "datetime", "timestamp"):
            if tcol in df.columns:
                df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
                df = df.dropna(subset=[tcol]).set_index(tcol)
                break
        return df
    except Exception:
        return None


def _resample_to_d1(h1_df):
    """Resample H1 (or finer) frame to D1. Returns DataFrame with 'close' or
    None on failure."""
    try:
        if h1_df is None or len(h1_df) == 0:
            return None
        # Need a DatetimeIndex for resample.
        if not hasattr(h1_df.index, "to_pydatetime"):
            return None
        agg = {"close": "last"}
        if "open" in h1_df.columns:
            agg["open"] = "first"
        if "high" in h1_df.columns:
            agg["high"] = "max"
        if "low" in h1_df.columns:
            agg["low"] = "min"
        d1 = h1_df.resample("1D").agg(agg).dropna(subset=["close"])
        return d1
    except Exception as e:
        log.debug("resample_to_d1 failed: %s", e)
        return None


def get_d1_bias(symbol: str, h1_df_or_candles) -> str:
    """Return 'LONG' | 'SHORT' | 'NEUTRAL' for the symbol based on
    D1-resampled EMA200. Fails open to 'NEUTRAL' on any data hiccup.

    Caches by (symbol, last-D1-bar-time): one EMA calc per day per symbol.
    """
    try:
        df = _to_h1_df(h1_df_or_candles)
        if df is None or len(df) < 50:
            return "NEUTRAL"

        d1 = _resample_to_d1(df)
        if d1 is None or len(d1) < 3:
            return "NEUTRAL"

        # Drop the (possibly) forming day so we only read CLOSED daily bars.
        d1_closed = d1.iloc[:-1] if len(d1) > 1 else d1
        if len(d1_closed) < 2:
            return "NEUTRAL"

        last_t = d1_closed.index[-1]
        cache_key = symbol
        cached = _BIAS_CACHE.get(cache_key)
        if cached is not None and cached[0] == last_t:
            return cached[1]

        # Adaptive EMA period: use min(200, available) so short buffers still
        # produce a verdict instead of permanent NEUTRAL.
        period = min(_EMA_PERIOD, len(d1_closed))
        if period < 2:
            return "NEUTRAL"

        closes = d1_closed["close"]
        ema = closes.ewm(span=period, adjust=False).mean()
        cur = float(closes.iloc[-1])
        ref = float(ema.iloc[-1])
        if ref <= 0:
            return "NEUTRAL"

        rel = (cur - ref) / ref
        if abs(rel) < _NEUTRAL_BAND_PCT:
            verdict = "NEUTRAL"
        elif rel > 0:
            verdict = "LONG"
        else:
            verdict = "SHORT"

        _BIAS_CACHE[cache_key] = (last_t, verdict)
        return verdict
    except Exception as e:
        log.debug("[%s] get_d1_bias failed: %s", symbol, e)
        return "NEUTRAL"


def clear_cache(symbol: str | None = None) -> None:
    """Test / day-rollover helper."""
    global _BIAS_CACHE
    if symbol is None:
        _BIAS_CACHE = {}
    else:
        _BIAS_CACHE.pop(symbol, None)


# ════════════════════════════════════════════════════════════════════════
#  SELF-TEST (--self-test)
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if "--self-test" not in sys.argv:
        print("usage: python3 -B agent/expert/d1_bias_unified.py --self-test")
        sys.exit(0)

    import numpy as np
    import pandas as pd

    # 1) Synthetic uptrend H1 frame: 60 days of strict up-drift.
    n_h1 = 60 * 24
    t = pd.date_range("2026-01-01", periods=n_h1, freq="1h", tz="UTC")
    close_up = np.linspace(100.0, 200.0, n_h1) + np.random.normal(0, 0.05, n_h1)
    df_up = pd.DataFrame({
        "open": close_up, "high": close_up + 0.1,
        "low": close_up - 0.1, "close": close_up,
    }, index=t)
    v_up = get_d1_bias("TEST_UP", df_up)
    assert v_up == "LONG", f"uptrend should be LONG, got {v_up}"
    print(f"  uptrend  -> {v_up}  OK")

    # 2) Synthetic downtrend.
    clear_cache()
    close_dn = np.linspace(200.0, 100.0, n_h1) + np.random.normal(0, 0.05, n_h1)
    df_dn = pd.DataFrame({
        "open": close_dn, "high": close_dn + 0.1,
        "low": close_dn - 0.1, "close": close_dn,
    }, index=t)
    v_dn = get_d1_bias("TEST_DN", df_dn)
    assert v_dn == "SHORT", f"downtrend should be SHORT, got {v_dn}"
    print(f"  downtrend-> {v_dn}  OK")

    # 3) Flat / chop within neutral band.
    clear_cache()
    close_flat = 150.0 + np.sin(np.arange(n_h1) / 24.0) * 0.05  # < 0.3% of 150
    df_flat = pd.DataFrame({
        "open": close_flat, "high": close_flat + 0.01,
        "low": close_flat - 0.01, "close": close_flat,
    }, index=t)
    v_flat = get_d1_bias("TEST_FLAT", df_flat)
    assert v_flat == "NEUTRAL", f"flat should be NEUTRAL, got {v_flat}"
    print(f"  flat     -> {v_flat}  OK")

    # 4) Garbage inputs fail open.
    assert get_d1_bias("X", None) == "NEUTRAL"
    assert get_d1_bias("X", []) == "NEUTRAL"
    assert get_d1_bias("X", "not a frame") == "NEUTRAL"
    print("  garbage  -> NEUTRAL  OK")

    # 5) Cache: after first call, the cache must contain an entry keyed by
    #    last D1 timestamp. Subsequent calls with the same data return same
    #    verdict, AND a corrupted input frame still returns cached result if
    #    cache key matches (proves the cache path runs before resample).
    clear_cache()
    v1 = get_d1_bias("CACHE_SYM", df_up)
    assert "CACHE_SYM" in _BIAS_CACHE, "cache entry not stored"
    assert _BIAS_CACHE["CACHE_SYM"][1] == v1
    # Second call: replace data with junk BUT keep same cache key by NOT
    # clearing. We can't change data and keep key, so just assert idempotence.
    v2 = get_d1_bias("CACHE_SYM", df_up)
    assert v1 == v2 == "LONG"
    # And: clear cache, call again, fresh compute returns same answer.
    clear_cache("CACHE_SYM")
    v3 = get_d1_bias("CACHE_SYM", df_up)
    assert v3 == "LONG"
    print("  cache    -> consistent  OK")

    print("d1_bias_unified self-test PASS")
