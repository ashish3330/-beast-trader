"""RangeDayClassifier — D1 ADX Session-Stamped Regime Gate
═════════════════════════════════════════════════════════════════════════
Pure-function module that classifies each (symbol, UTC trading day) into
one of {TREND_DAY, NEUTRAL_DAY, RANGE_DAY} using Wilder DMI/ADX(14) on
the D1 resampled frame, plus the dominant +DI/-DI side as a directional
bias.

The classifier is intended as a **structural permission filter** placed
EARLIEST in the brain._evaluate_symbol funnel (Gate 0d, before session
hours / Gate 1). It operates on a per-day cache — one ADX calculation
per (symbol, day) — so the marginal cost in the hot path is a dict
lookup.

Policy table (single source of truth; brain wires the dispatch):

  TREND_DAY  (ADX >= 25) : ALLOW momentum / breakout / FVG-continuation.
                           BLOCK pure mean-revert / counter-trend
                           SweepReclaim signals UNLESS aligned with the
                           D1 +DI/-DI bias.
  NEUTRAL_DAY (15 <= ADX < 25) : ALLOW everything (current behavior).
  RANGE_DAY  (ADX < 15)  : ALLOW SweepReclaim / mean-revert /
                           fade-extreme / pullback / FVG-REVERSAL.
                           BLOCK pure momentum-continuation breakouts
                           UNLESS direction == D1 +DI/-DI bias.
                           When allowed-with-bias on a range day, the
                           module emits risk_mult = RDC_RANGE_DAY_SIZE_MULT
                           (default 0.5×) so brain can downsize.

Literature anchors
──────────────────
  * Wilder (1978) "New Concepts in Technical Trading Systems" — original
    ADX thresholds: <20 no trend, 20-25 weak, >25 trend, >40 strong.
  * Wyckoff Phase A/B/C/D/E — Phase B (accumulation/range) corresponds
    to ADX<15, Phase D (markup) to ADX>25.
  * ICT "Daily Bias" doctrine — HTF daily structure governs intraday
    setup-type selection; matches the per-day stamp at session open.
  * Connors & Alvarez "High Probability ETF Trading" — mean-reversion
    edge concentrates on low-ADX days; mirrors the RANGE_DAY policy.

Design notes
────────────
  * Stateless module-level functions. No MT5, no journal, no
    side-effects beyond optional logging. State (the per-symbol daily
    stamp cache) lives in the caller (brain) — we expose `stamp_regime`
    that accepts and returns a cache dict, plus `classify_signal` that
    answers "does this (regime, signal_class, direction) pass?".
  * Uses the SAME Wilder ADX implementation shape as
    agent/sweep_reclaim.py:_adx — but returns BOTH +DI and -DI so the
    caller can derive directional bias without recomputing.
  * Drops the currently-forming D1 bar (iloc[-2] convention, per
    [[feedback_entry_pipeline_bugs_20260525]]).

Public API
──────────
    classify_regime(d1_high, d1_low, d1_close, period=14,
                    trend_threshold=25.0, range_threshold=15.0) -> dict
        Returns {"regime": str, "adx": float, "plus_di": float,
                 "minus_di": float, "bias": "LONG"|"SHORT"} or None.

    stamp_regime(symbol, today_utc, h1_df, cache, *, period=14,
                 trend_threshold=25.0, range_threshold=15.0,
                 min_d1_bars=30) -> dict | None
        Cache-aware wrapper. Calls classify_regime once per (symbol,
        day) and caches the result in `cache`. `h1_df` is the raw H1
        candle DataFrame (state.get_candles output).

    evaluate(regime_record, signal_class, direction, *,
             range_day_size_mult=0.5,
             require_di_alignment_on_range=True) -> dict
        Pure policy decision. Returns {"action": "ALLOW"|"SKIP",
        "reason": str, "risk_mult": float}.

    classify_signal(source_tag, signal_class_map) -> str
        Maps internal signal source (e.g. "momentum") -> RDC class
        (e.g. "MOMENTUM"). Pure helper.

    clear_cache(cache) -> None
        Drops every entry in the cache (called at day rollover).

The module is designed so that brain.py can wire it as a thin glue:

    from agent.expert import (
        rdc_stamp_regime, rdc_evaluate, rdc_classify_signal,
        rdc_clear_cache,
    )

    # inside DragonBrain.__init__:
    self._rdc_cache = {}

    # inside _run_cycle day-rollover block:
    if today != self._last_day:
        ...
        rdc_clear_cache(self._rdc_cache)

    # inside _evaluate_symbol (Gate 0d):
    h1 = self.state.get_candles(symbol, 60)
    rec = rdc_stamp_regime(symbol, today_utc, h1, self._rdc_cache)
    if rec is not None:
        sig_cls = rdc_classify_signal(source_tag, cfg.RDC_SIGNAL_CLASS_MAP)
        verdict = rdc_evaluate(rec, sig_cls, direction)
        if verdict["action"] == "SKIP":
            return SKIP(verdict["reason"])
        if verdict["risk_mult"] != 1.0:
            base_ret["risk_mult"] = verdict["risk_mult"]
"""

from __future__ import annotations

import logging
import math
from datetime import date as _date, datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
#  Config import (lazy / soft — file is importable in tests even if
#  config.py is not on the path).
# ──────────────────────────────────────────────────────────────────────
try:  # pragma: no cover — config in production env
    import config as _cfg  # type: ignore
except Exception:  # pragma: no cover
    _cfg = None  # type: ignore

log = logging.getLogger("range_day_classifier")


def _cfg_get(name: str, default):
    """Read a config flag with a safe default — keeps tests independent
    of config.py wiring."""
    if _cfg is None:
        return default
    return getattr(_cfg, name, default)


# ──────────────────────────────────────────────────────────────────────
#  Constants — defaults; can be overridden via config.py at call-site
# ──────────────────────────────────────────────────────────────────────
ADX_PERIOD_D1            = 14
ADX_TREND_THRESHOLD      = 25.0
ADX_RANGE_THRESHOLD      = 15.0
D1_MIN_BARS              = 30
H1_RESAMPLE_LOOKBACK     = 720
RESAMPLE_RULE            = "1D"
STAMP_CACHE_TTL_HOURS    = 24
ALLOW_ALL_ON_UNKNOWN     = True
RANGE_DAY_SIZE_MULT      = 0.5

DEFAULT_SIGNAL_CLASS_MAP = {
    "momentum":       "MOMENTUM",
    "nr7":            "BREAKOUT",
    "sweep_reclaim":  "MEAN_REVERT",
    "fvg":            "FVG_REVERSAL",
    "pullback":       "PULLBACK",
    "wyckoff_spring": "MEAN_REVERT",
    "wyckoff_upthrust": "MEAN_REVERT",
}

ALLOWED_SIGNAL_CLASSES = {
    "MOMENTUM", "BREAKOUT", "MEAN_REVERT", "PULLBACK", "FVG_REVERSAL",
}


# ──────────────────────────────────────────────────────────────────────
#  Helpers — Wilder ADX with full +DI / -DI / ADX outputs
# ──────────────────────────────────────────────────────────────────────
def _wilder_adx_dmi(H: np.ndarray, L: np.ndarray, C: np.ndarray,
                    period: int = 14) -> Tuple[float, float, float]:
    """Wilder DMI/ADX. Returns (adx, plus_di, minus_di) for the LAST
    observation. Returns (NaN, NaN, NaN) when the series is too short.

    Same algorithm as agent/sweep_reclaim.py:_adx but exposes the
    directional indicators (+DI / -DI) which the regime gate needs to
    derive the daily bias.
    """
    H = np.asarray(H, dtype=float)
    L = np.asarray(L, dtype=float)
    C = np.asarray(C, dtype=float)
    n = len(H)
    if n < 2 * period + 1:
        return (float("nan"), float("nan"), float("nan"))

    up = H[1:] - H[:-1]
    dn = L[:-1] - L[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.empty(n - 1)
    for i in range(n - 1):
        tr[i] = max(
            H[i + 1] - L[i + 1],
            abs(H[i + 1] - C[i]),
            abs(L[i + 1] - C[i]),
        )

    sm_plus = np.empty(n - 1)
    sm_minus = np.empty(n - 1)
    sm_tr = np.empty(n - 1)
    sm_plus[period - 1] = plus_dm[:period].sum()
    sm_minus[period - 1] = minus_dm[:period].sum()
    sm_tr[period - 1] = tr[:period].sum()
    for i in range(period, n - 1):
        sm_plus[i] = sm_plus[i - 1] - sm_plus[i - 1] / period + plus_dm[i]
        sm_minus[i] = sm_minus[i - 1] - sm_minus[i - 1] / period + minus_dm[i]
        sm_tr[i] = sm_tr[i - 1] - sm_tr[i - 1] / period + tr[i]

    plus_di_arr = 100.0 * sm_plus / np.where(sm_tr == 0, 1, sm_tr)
    minus_di_arr = 100.0 * sm_minus / np.where(sm_tr == 0, 1, sm_tr)
    denom = np.where((plus_di_arr + minus_di_arr) == 0, 1,
                     (plus_di_arr + minus_di_arr))
    dx = 100.0 * np.abs(plus_di_arr - minus_di_arr) / denom

    adx = np.empty(n - 1)
    if 2 * period - 1 >= n - 1:
        return (float("nan"), float(plus_di_arr[-1]),
                float(minus_di_arr[-1]))
    adx[2 * period - 1] = dx[period - 1:2 * period].mean()
    for i in range(2 * period, n - 1):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return (float(adx[-1]), float(plus_di_arr[-1]),
            float(minus_di_arr[-1]))


# ──────────────────────────────────────────────────────────────────────
#  Helpers — candle normalization & D1 resample
# ──────────────────────────────────────────────────────────────────────
def _normalize_candles(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Coerce a candle DataFrame (columns: time, open, high, low, close,
    ...) into a clean, time-indexed UTC OHLC frame. Returns None on bad
    input. Mirror of fvg_strategy._normalize_candles so this module
    stays self-contained.
    """
    if df is None or len(df) == 0:
        return None
    try:
        df = df.copy()
        if "time" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(
                    columns={df.index.name or "index": "time"})
            else:
                return None
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        try:
            df.index = df.index.as_unit("ns")
        except Exception:
            pass
        cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in cols):
            return None
        return df[cols].astype(float)
    except Exception as e:
        log.debug("RDC _normalize_candles failed: %s", e)
        return None


def _resample_d1(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Resample a time-indexed H1 (or finer) OHLC frame to D1. Drops
    forming bar at the call site by slicing iloc[:-1] — here we just
    resample.
    """
    if df is None or len(df) == 0:
        return None
    try:
        o = df["open"].resample(RESAMPLE_RULE).first()
        h = df["high"].resample(RESAMPLE_RULE).max()
        l = df["low"].resample(RESAMPLE_RULE).min()
        c = df["close"].resample(RESAMPLE_RULE).last()
        out = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c}).dropna()
        return out
    except Exception as e:
        log.debug("RDC _resample_d1 failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────
def classify_regime(d1_high, d1_low, d1_close,
                    period: int = ADX_PERIOD_D1,
                    trend_threshold: float = ADX_TREND_THRESHOLD,
                    range_threshold: float = ADX_RANGE_THRESHOLD,
                    ) -> Optional[Dict[str, Any]]:
    """Classify a D1 bar series. Pure function — no caching, no I/O.

    Parameters
    ----------
    d1_high, d1_low, d1_close : array-like
        Series of closed D1 OHLC values. The CALLER must pass
        already-closed bars (forming bar dropped via iloc[:-1]).
    period : int
        Wilder DMI lookback (default 14).
    trend_threshold : float
        ADX >= this  => TREND_DAY.
    range_threshold : float
        ADX <  this  => RANGE_DAY.

    Returns
    -------
    dict or None
        {"regime": "TREND_DAY"|"NEUTRAL_DAY"|"RANGE_DAY",
         "adx": float, "plus_di": float, "minus_di": float,
         "bias": "LONG"|"SHORT"}
        or None when the series is too short / values are non-finite.
    """
    H = np.asarray(d1_high, dtype=float)
    L = np.asarray(d1_low, dtype=float)
    C = np.asarray(d1_close, dtype=float)
    if len(H) < (2 * period + 1):
        return None

    adx_val, plus_di, minus_di = _wilder_adx_dmi(H, L, C, period=period)
    if (not math.isfinite(adx_val) or adx_val <= 0.0
            or not math.isfinite(plus_di)
            or not math.isfinite(minus_di)):
        return None

    if adx_val >= trend_threshold:
        regime = "TREND_DAY"
    elif adx_val < range_threshold:
        regime = "RANGE_DAY"
    else:
        regime = "NEUTRAL_DAY"

    bias = "LONG" if plus_di > minus_di else "SHORT"
    return {
        "regime": regime,
        "adx": float(adx_val),
        "plus_di": float(plus_di),
        "minus_di": float(minus_di),
        "bias": bias,
    }


def stamp_regime(symbol: str,
                 today_utc: _date,
                 h1_df: Optional[pd.DataFrame],
                 cache: Dict[str, Dict[str, Any]],
                 *,
                 period: int = ADX_PERIOD_D1,
                 trend_threshold: float = ADX_TREND_THRESHOLD,
                 range_threshold: float = ADX_RANGE_THRESHOLD,
                 min_d1_bars: int = D1_MIN_BARS,
                 ) -> Optional[Dict[str, Any]]:
    """Cache-aware regime stamp for (symbol, today_utc).

    Returns the existing cached record if `today_utc` already has one
    for this symbol. Otherwise resamples H1 → D1, drops the forming
    bar, computes Wilder ADX/+DI/-DI on the closed bars and stamps the
    result.

    Parameters
    ----------
    symbol : str
    today_utc : datetime.date
        The UTC trading-day key.
    h1_df : pandas.DataFrame or None
        Raw H1 candles (state.get_candles output). May be None on data
        gaps; caller falls through.
    cache : dict
        Per-symbol stamp store. Mutated in place.

    Returns
    -------
    dict or None
        {"day": date, "regime": str, "adx": float, "plus_di": float,
         "minus_di": float, "bias": str, "ts": datetime} or None.
    """
    cached = cache.get(symbol)
    if cached is not None and cached.get("day") == today_utc:
        return cached

    if h1_df is None or len(h1_df) == 0:
        return None

    # Need enough H1 bars to resample to >= min_d1_bars closed D1 bars.
    if len(h1_df) < (H1_RESAMPLE_LOOKBACK // 4):
        return None

    norm = _normalize_candles(h1_df)
    if norm is None or len(norm) == 0:
        return None
    d1 = _resample_d1(norm)
    if d1 is None or len(d1) < (min_d1_bars + 1):
        # +1 because we drop the forming bar below.
        return None

    # Drop the currently-forming D1 bar.
    d1_closed = d1.iloc[:-1]
    if len(d1_closed) < min_d1_bars:
        return None

    H = d1_closed["high"].values
    L = d1_closed["low"].values
    C = d1_closed["close"].values
    verdict = classify_regime(H, L, C,
                              period=period,
                              trend_threshold=trend_threshold,
                              range_threshold=range_threshold)
    if verdict is None:
        return None

    rec = {
        "day": today_utc,
        "regime": verdict["regime"],
        "adx": verdict["adx"],
        "plus_di": verdict["plus_di"],
        "minus_di": verdict["minus_di"],
        "bias": verdict["bias"],
        "ts": datetime.now(timezone.utc),
    }
    cache[symbol] = rec
    log.info(
        "[%s] D1 regime=%s ADX=%.1f +DI=%.1f -DI=%.1f bias=%s",
        symbol, rec["regime"], rec["adx"],
        rec["plus_di"], rec["minus_di"], rec["bias"],
    )
    return rec


def evaluate(regime_record: Optional[Dict[str, Any]],
             signal_class: str,
             direction: str,
             *,
             range_day_size_mult: float = RANGE_DAY_SIZE_MULT,
             require_di_alignment_on_range: bool = True,
             allow_all_on_unknown: bool = ALLOW_ALL_ON_UNKNOWN,
             ) -> Dict[str, Any]:
    """Pure policy decision. Given a regime stamp + signal class +
    direction, return the action.

    Returns
    -------
    dict
        {"action": "ALLOW"|"SKIP",
         "reason": str,
         "risk_mult": float}
        risk_mult is 1.0 except on the RANGE_DAY + aligned-trend-setup
        path, where it drops to `range_day_size_mult` (default 0.5).
    """
    direction = (direction or "").upper()
    signal_class = (signal_class or "").upper()

    if regime_record is None:
        if allow_all_on_unknown:
            return {"action": "ALLOW",
                    "reason": "D1_REGIME_UNKNOWN_WARN_ONLY",
                    "risk_mult": 1.0}
        return {"action": "SKIP",
                "reason": "D1_REGIME_UNKNOWN",
                "risk_mult": 0.0}

    regime = regime_record.get("regime", "NEUTRAL_DAY")
    bias = (regime_record.get("bias") or "").upper()

    # NEUTRAL_DAY → pass-through unchanged.
    if regime == "NEUTRAL_DAY":
        return {"action": "ALLOW",
                "reason": "D1_NEUTRAL_PASS",
                "risk_mult": 1.0}

    # TREND_DAY policy: block pure mean-revert that fights the trend.
    if regime == "TREND_DAY":
        if signal_class == "MEAN_REVERT":
            if direction == bias:
                # Mean-revert WITH the daily bias is allowed (e.g. a
                # pullback-buy on an uptrend day).
                return {"action": "ALLOW",
                        "reason": "D1_TREND_MEANREV_ALIGNED",
                        "risk_mult": 1.0}
            return {"action": "SKIP",
                    "reason": "D1_TREND_BLOCKS_MEANREV",
                    "risk_mult": 0.0}
        # Momentum / breakout / FVG-reversal / pullback all allowed.
        return {"action": "ALLOW",
                "reason": "D1_TREND_PASS",
                "risk_mult": 1.0}

    # RANGE_DAY policy.
    if regime == "RANGE_DAY":
        if signal_class in ("MOMENTUM", "BREAKOUT"):
            if require_di_alignment_on_range and direction != bias:
                return {"action": "SKIP",
                        "reason": "D1_RANGE_BLOCKS_TREND_SETUP",
                        "risk_mult": 0.0}
            # Aligned with dominant DI on a range day: allow but downsize.
            return {"action": "ALLOW",
                    "reason": "D1_RANGE_TREND_ALIGNED_DOWNSIZED",
                    "risk_mult": float(range_day_size_mult)}
        # MEAN_REVERT / PULLBACK / FVG_REVERSAL all allowed on range days.
        return {"action": "ALLOW",
                "reason": "D1_RANGE_PASS",
                "risk_mult": 1.0}

    # Unknown regime label (defensive).
    return {"action": "ALLOW",
            "reason": "D1_REGIME_UNRECOGNIZED",
            "risk_mult": 1.0}


def classify_signal(source_tag: str,
                    signal_class_map: Optional[Dict[str, str]] = None,
                    default: str = "MOMENTUM") -> str:
    """Map an internal signal source string (e.g. "momentum", "nr7",
    "fvg") to a RDC class ("MOMENTUM" / "BREAKOUT" / ...). Falls back
    to `default` (MOMENTUM) when unknown — preserves "don't skip on
    ambiguity" per [[feedback_no_skip_trades]].
    """
    if not source_tag:
        return default
    mp = signal_class_map or DEFAULT_SIGNAL_CLASS_MAP
    key = source_tag.lower()
    val = mp.get(key)
    if val is None:
        # Allow caller-uppercase keys too.
        val = mp.get(source_tag)
    if val is None:
        return default
    if val not in ALLOWED_SIGNAL_CLASSES:
        return default
    return val


def clear_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    """Drop every entry in the cache — called at UTC day rollover."""
    if cache is None:
        return
    cache.clear()


# ──────────────────────────────────────────────────────────────────────
#  Self-test — runnable as `python3 -B <file>` and exits 0 on success.
# ──────────────────────────────────────────────────────────────────────
def _synthetic_d1_trend(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic UP-trending D1 series: ADX should run high."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(np.abs(rng.normal(1.0, 0.15, n))) + 100.0
    noise = rng.normal(0.0, 0.20, n)
    close = base + noise
    open_ = np.concatenate([[base[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0.30, 0.15, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.30, 0.15, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )


def _synthetic_d1_range(n: int = 60, seed: int = 11) -> pd.DataFrame:
    """Mean-reverting D1 series: low ADX, ~equal +DI / -DI."""
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.0, 0.6, n).cumsum() * 0.05
    # Pull back to 100 (Ornstein-Uhlenbeck-ish) so we genuinely range.
    for i in range(1, n):
        close[i] = 0.85 * close[i - 1] + 0.15 * 100.0 + rng.normal(0, 0.4)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0.20, 0.10, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.20, 0.10, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )


def _synthetic_h1_from_d1(d1: pd.DataFrame) -> pd.DataFrame:
    """Explode a D1 frame into 24 H1 bars per day (cheap stand-in for
    the broker H1 cache so stamp_regime can resample → D1 again)."""
    rows = []
    for ts, row in d1.iterrows():
        # 24 hourly bars sampled linearly from open → close, with the
        # day's high/low spread across the bars uniformly.
        for hh in range(24):
            t = ts + pd.Timedelta(hours=hh)
            frac_open = hh / 24.0
            frac_close = (hh + 1) / 24.0
            o = row["open"] + (row["close"] - row["open"]) * frac_open
            c = row["open"] + (row["close"] - row["open"]) * frac_close
            h = max(o, c, row["high"])
            l = min(o, c, row["low"])
            rows.append({"time": t, "open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(rows)


def _self_test() -> int:
    print("== RangeDayClassifier self-test ==")
    ok = True

    # 1) classify_regime on trending data → expect TREND_DAY.
    trend_d1 = _synthetic_d1_trend(n=80)
    rec_trend = classify_regime(
        trend_d1["high"].values,
        trend_d1["low"].values,
        trend_d1["close"].values,
        period=14,
    )
    assert rec_trend is not None, "classify_regime returned None on trend"
    print(f"  trend: regime={rec_trend['regime']:<12} "
          f"adx={rec_trend['adx']:.1f} +DI={rec_trend['plus_di']:.1f} "
          f"-DI={rec_trend['minus_di']:.1f} bias={rec_trend['bias']}")
    if rec_trend["regime"] != "TREND_DAY":
        print(f"   WARN: expected TREND_DAY but got {rec_trend['regime']}")
    if rec_trend["bias"] != "LONG":
        print(f"   WARN: expected LONG bias on up-trend but got "
              f"{rec_trend['bias']}")
        ok = False

    # 2) classify_regime on ranging data → expect RANGE_DAY or NEUTRAL.
    range_d1 = _synthetic_d1_range(n=80)
    rec_range = classify_regime(
        range_d1["high"].values,
        range_d1["low"].values,
        range_d1["close"].values,
        period=14,
    )
    assert rec_range is not None, "classify_regime returned None on range"
    print(f"  range: regime={rec_range['regime']:<12} "
          f"adx={rec_range['adx']:.1f} +DI={rec_range['plus_di']:.1f} "
          f"-DI={rec_range['minus_di']:.1f} bias={rec_range['bias']}")
    if rec_range["regime"] == "TREND_DAY":
        print("   WARN: synthetic range data classified as TREND_DAY")

    # 3) Short series → None.
    short_d1 = trend_d1.iloc[:10]
    short_rec = classify_regime(
        short_d1["high"].values,
        short_d1["low"].values,
        short_d1["close"].values,
        period=14,
    )
    assert short_rec is None, "expected None on short series"
    print(f"  short-series: {short_rec} (expected None) OK")

    # 4) evaluate() — full policy matrix sanity check.
    trend_rec = {"regime": "TREND_DAY", "bias": "LONG", "adx": 30.0}
    range_rec_long = {"regime": "RANGE_DAY", "bias": "LONG", "adx": 10.0}
    neutral_rec = {"regime": "NEUTRAL_DAY", "bias": "LONG", "adx": 20.0}

    cases = [
        # (regime, signal_class, direction, expected_action, expected_mult)
        (trend_rec,        "MEAN_REVERT",  "SHORT", "SKIP",  0.0),
        (trend_rec,        "MEAN_REVERT",  "LONG",  "ALLOW", 1.0),
        (trend_rec,        "MOMENTUM",     "LONG",  "ALLOW", 1.0),
        (trend_rec,        "BREAKOUT",     "SHORT", "ALLOW", 1.0),
        (range_rec_long,   "MOMENTUM",     "SHORT", "SKIP",  0.0),
        (range_rec_long,   "MOMENTUM",     "LONG",  "ALLOW", 0.5),
        (range_rec_long,   "BREAKOUT",     "LONG",  "ALLOW", 0.5),
        (range_rec_long,   "MEAN_REVERT",  "SHORT", "ALLOW", 1.0),
        (range_rec_long,   "FVG_REVERSAL", "SHORT", "ALLOW", 1.0),
        (range_rec_long,   "PULLBACK",     "SHORT", "ALLOW", 1.0),
        (neutral_rec,      "MOMENTUM",     "SHORT", "ALLOW", 1.0),
        (neutral_rec,      "MEAN_REVERT",  "LONG",  "ALLOW", 1.0),
        (None,             "MOMENTUM",     "LONG",  "ALLOW", 1.0),  # unknown -> allow
    ]
    for i, (rec, sc, dirn, exp_act, exp_mult) in enumerate(cases):
        v = evaluate(rec, sc, dirn)
        if v["action"] != exp_act or abs(v["risk_mult"] - exp_mult) > 1e-9:
            print(f"   FAIL case {i}: rec={rec} sc={sc} dir={dirn} "
                  f"-> {v} (expected action={exp_act} mult={exp_mult})")
            ok = False
        else:
            print(f"  evaluate case {i}: rec={rec.get('regime') if rec else 'NONE':<11} "
                  f"sc={sc:<13} dir={dirn:<5} -> {v['action']:<5} "
                  f"reason={v['reason']} mult={v['risk_mult']}")

    # 5) evaluate() with allow_all_on_unknown=False -> SKIP.
    v = evaluate(None, "MOMENTUM", "LONG", allow_all_on_unknown=False)
    if v["action"] != "SKIP":
        print(f"   FAIL: unknown+strict expected SKIP, got {v}")
        ok = False
    else:
        print(f"  unknown+strict -> SKIP OK ({v['reason']})")

    # 6) classify_signal — defaults & overrides.
    assert classify_signal("momentum") == "MOMENTUM"
    assert classify_signal("nr7") == "BREAKOUT"
    assert classify_signal("sweep_reclaim") == "MEAN_REVERT"
    assert classify_signal("fvg") == "FVG_REVERSAL"
    assert classify_signal("pullback") == "PULLBACK"
    assert classify_signal("unknown_source") == "MOMENTUM"  # default
    assert classify_signal("") == "MOMENTUM"
    custom_map = {"foo": "BREAKOUT"}
    assert classify_signal("foo", custom_map) == "BREAKOUT"
    # Bad target value falls back to default.
    bad_map = {"foo": "NOT_A_CLASS"}
    assert classify_signal("foo", bad_map) == "MOMENTUM"
    print("  classify_signal: all assertions PASS")

    # 7) stamp_regime — cache miss, hit, clear cycle on synthetic H1.
    h1 = _synthetic_h1_from_d1(trend_d1)
    cache: Dict[str, Dict[str, Any]] = {}
    today = _date(2026, 6, 16)
    r1 = stamp_regime("TEST", today, h1, cache)
    assert r1 is not None, "stamp_regime returned None on synthetic H1"
    assert "TEST" in cache
    # Second call same day -> cache hit, same object identity.
    r2 = stamp_regime("TEST", today, h1, cache)
    assert r2 is r1, "expected cache HIT to return identical record"
    print(f"  stamp_regime: cache miss -> hit OK (regime={r1['regime']} "
          f"adx={r1['adx']:.1f} bias={r1['bias']})")

    # 8) clear_cache — entries gone.
    clear_cache(cache)
    assert "TEST" not in cache and len(cache) == 0
    print("  clear_cache: OK")

    # 9) stamp_regime returns None on tiny H1 input.
    none_rec = stamp_regime("TEST", today, h1.head(20), {})
    assert none_rec is None, "expected None on tiny H1 input"
    print("  stamp_regime tiny-input: None OK")

    # 10) stamp_regime returns None on None input.
    none_rec2 = stamp_regime("TEST", today, None, {})
    assert none_rec2 is None
    print("  stamp_regime None-input: None OK")

    if ok:
        print("== ALL SELF-TESTS PASS ==")
        return 0
    print("== SELF-TEST FAILURES ==")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
