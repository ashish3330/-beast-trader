"""Dynamic SL/TP — ATR + Structure + Regime (DynamicExitPlanner)
─────────────────────────────────────────────────────────────────
Pure-function planner that computes an `ExitPlan` (sl_dist, tp1_dist,
tp2_dist, runner_dist, source_tags) for a candidate entry AFTER all
entry gates pass but BEFORE executor places the trade.

Algorithm summary
─────────────────
  1. Structural SL = max(ATR_SL_MULT × ATR, last swing extreme + buffer),
     clamped to [ATR_FLOOR_MULT*ATR, ATR_CAP_MULT*ATR]
     Buffer = STRUCT_BUFFER_ATR × ATR + spread × SPREAD_BUFFER_MULT
     (Wyckoff "stop beyond the spring" + ICT stop-hunt avoidance.)
  2. TP1 = fixed 1.5R partial (spec).
  3. TP2 priority cascade:
       A. D1 swing extreme (resampled from H1) if within D1_SWING_MAX_R
       B. Nearest H4 supply (SHORT) / demand (LONG) zone within H4_ZONE_MAX_R
       C. Fixed 3R fallback
     Then regime-tilted via REGIME_TP_MULT and clamped to
     [max(TP1*1.2, sl*1.8), sl*5.0] so it's always > TP1 and never silly.
  4. Runner (Sub2) = 5R fixed (mirrors legacy Single mode TP).

References
──────────
  * Lewis Borsellino — "The Day Trader" (stops beyond protected swing wicks).
  * ICT (Inner Circle Trader) — premium/discount arrays, stop hunt theory.
  * Sam Seiden — "drop-base-rally / rally-base-drop" supply/demand clustering.
  * Bourgade & Hassani arXiv:2009.08821 — fixed-R exits leak EV when
    structural draws exist.

Module is pure functions + thin convenience class. Takes pandas OHLC
frames + scalars, returns an ExitPlan dataclass. No MT5, no executor
side effects, no logging beyond optional debug.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
#  Soft config import — keeps the module standalone-testable
# ──────────────────────────────────────────────────────────────────────
try:  # pragma: no cover
    import config as _cfg  # type: ignore
except Exception:  # pragma: no cover
    _cfg = None  # type: ignore

log = logging.getLogger("dynamic_exit")


def _cfg_get(name: str, default):
    if _cfg is None:
        return default
    return getattr(_cfg, name, default)


# ──────────────────────────────────────────────────────────────────────
#  Constants — mirror the spec, all overridable from config.py
# ──────────────────────────────────────────────────────────────────────
# Structural lookbacks
M15_SWING_LB_DEF       = 20
H4_SWING_LB_DEF        = 40
D1_SWING_LB_DEF        = 30
H4_BARS_PER            = 4    # H1 → H4 aggregation factor
D1_BARS_PER            = 24   # H1 → D1 aggregation factor
# ATR clamps / buffer
ATR_FLOOR_MULT_DEF     = 1.0
ATR_CAP_MULT_DEF       = 3.0
STRUCT_BUFFER_ATR_DEF  = 0.25
SPREAD_BUFFER_MULT_DEF = 1.5
# TP geometry
TP1_R_DEF              = 1.5
TP2_R_FLOOR_DEF        = 1.8
TP2_R_CAP_DEF          = 5.0
TP2_FALLBACK_R_DEF     = 3.0
D1_SWING_MAX_R_DEF     = 5.0
H4_ZONE_MAX_R_DEF      = 5.0
# Regime tilt — TP2 only, SL stays structure-anchored
REGIME_TP_MULT_DEF = {
    "trending": 1.20,
    "volatile": 1.15,
    "ranging":  0.80,
    "low_vol":  0.90,
    "unknown":  1.00,
}
# Supply/demand zone detection
ZONE_CLUSTER_ATR_DEF   = 0.5
ZONE_MIN_TOUCHES_DEF   = 2
# Runner / Sub2
RUNNER_R_DEF           = 5.0


# ──────────────────────────────────────────────────────────────────────
#  Result type
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ExitPlan:
    """Distances are POSITIVE absolute price units. Executor converts to
    levels: long_sl = entry - sl_dist, short_sl = entry + sl_dist."""
    sl_dist:        float
    tp1_dist:       float
    tp2_dist:       float
    runner_dist:    float
    # Provenance / telemetry
    tp2_source:     str    = "FALLBACK_3R"
    atr_sl_dist:    float  = 0.0
    struct_sl_used: bool   = False
    regime_mult:    float  = 1.0
    regime:         str    = "unknown"
    source_tags:    List[str] = field(default_factory=list)
    # Sanity — raw R-multiples for log/journal cross-ref
    tp1_R:          float  = 1.5
    tp2_R:          float  = 3.0
    runner_R:       float  = 5.0
    # Used inputs
    atr:            float  = 0.0
    spread:         float  = 0.0
    symbol:         str    = ""
    direction:      str    = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
#  Helpers — DataFrame normalisation + OHLC aggregation
# ──────────────────────────────────────────────────────────────────────
def _normalize_ohlc(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return a frame with DatetimeIndex + lowercase OHLC cols, or None."""
    if df is None or len(df) == 0:
        return None
    try:
        out = df.copy()
        out.columns = [str(c).lower() for c in out.columns]
        if not isinstance(out.index, pd.DatetimeIndex):
            if "time" in out.columns:
                out["time"] = pd.to_datetime(out["time"], utc=True)
                out = out.set_index("time").sort_index()
            else:
                # Best effort — wrap a default range index
                out.index = pd.RangeIndex(len(out))
        for c in ("open", "high", "low", "close"):
            if c not in out.columns:
                return None
        return out
    except Exception as e:  # pragma: no cover
        log.debug("dynamic_exit._normalize_ohlc failed: %s", e)
        return None


def aggregate_ohlc(h1_df: pd.DataFrame, bars_per: int) -> Optional[pd.DataFrame]:
    """Aggregate H1 OHLC into bars_per-sized chunks (H4 = 4, D1 = 24).

    Uses index-position-based grouping (no calendar resample) so it
    works with synthetic frames and with frames that don't start on a
    session boundary. Drops the currently-forming partial bar.
    """
    df = _normalize_ohlc(h1_df)
    if df is None or len(df) < bars_per + 1:
        return None
    try:
        n = len(df)
        usable = (n // bars_per) * bars_per
        if usable <= 0:
            return None
        sub = df.iloc[:usable].copy()
        # Group every bars_per consecutive bars
        grp = np.arange(usable) // bars_per
        o = sub["open"].groupby(grp).first()
        h = sub["high"].groupby(grp).max()
        l = sub["low"].groupby(grp).min()
        c = sub["close"].groupby(grp).last()
        # Use start-index timestamp for grouping label
        try:
            tidx = sub.index[(np.arange(len(o)) * bars_per).astype(int)]
        except Exception:
            tidx = pd.RangeIndex(len(o))
        out = pd.DataFrame(
            {"open": o.values, "high": h.values, "low": l.values, "close": c.values},
            index=tidx,
        )
        return out
    except Exception as e:  # pragma: no cover
        log.debug("dynamic_exit.aggregate_ohlc failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────
#  Structural extreme detection
# ──────────────────────────────────────────────────────────────────────
def find_swing_extreme(df: pd.DataFrame, direction: str,
                       lookback: int = 20) -> Optional[float]:
    """Return the protective swing extreme PRICE for an entry in the
    given direction, scanning the most recent `lookback` CLOSED bars
    (uses iloc[-2] convention so the live forming bar is excluded).

    LONG  → lowest low in the window (stop sits below)
    SHORT → highest high in the window (stop sits above)
    """
    df = _normalize_ohlc(df)
    if df is None or len(df) < 3:
        return None
    try:
        # Exclude the last (potentially-forming) bar
        window = df.iloc[-(lookback + 1):-1]
        if len(window) == 0:
            return None
        if direction.upper() == "LONG":
            return float(window["low"].min())
        elif direction.upper() == "SHORT":
            return float(window["high"].max())
        else:
            return None
    except Exception as e:  # pragma: no cover
        log.debug("find_swing_extreme failed: %s", e)
        return None


def find_next_swing_extreme(df: pd.DataFrame, direction: str,
                            lookback: int = 30,
                            entry: float = 0.0) -> Optional[float]:
    """Return the DISTANCE from `entry` to the next significant swing
    extreme in the trade direction (LONG → swing-high above, SHORT →
    swing-low below). None if no extreme lies in the trade direction.

    `lookback` is in bars of the supplied frame (already resampled).
    """
    df = _normalize_ohlc(df)
    if df is None or len(df) < 3:
        return None
    try:
        window = df.iloc[-(lookback + 1):-1]
        if len(window) == 0:
            return None
        if direction.upper() == "LONG":
            ext = float(window["high"].max())
            d = ext - entry
        elif direction.upper() == "SHORT":
            ext = float(window["low"].min())
            d = entry - ext
        else:
            return None
        if d <= 0 or not math.isfinite(d):
            return None
        return d
    except Exception as e:  # pragma: no cover
        log.debug("find_next_swing_extreme failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────
#  H4 supply/demand zone detection
# ──────────────────────────────────────────────────────────────────────
def _find_pivots(values: np.ndarray, n: int = 2, kind: str = "high") -> List[int]:
    """Indices of fractal pivots (n bars on each side strictly less/greater).
    `kind` is 'high' for swing highs, 'low' for swing lows."""
    idx: List[int] = []
    if values is None or len(values) < (2 * n + 1):
        return idx
    for i in range(n, len(values) - n):
        v = values[i]
        left = values[i - n:i]
        right = values[i + 1:i + 1 + n]
        if kind == "high":
            if v >= left.max() and v >= right.max() and v > left.min() and v > right.min():
                idx.append(i)
        else:
            if v <= left.min() and v <= right.min() and v < left.max() and v < right.max():
                idx.append(i)
    return idx


def find_h4_zone(h4_df: pd.DataFrame, direction: str, entry: float,
                 atr: float, cluster_atr: float = 0.5,
                 min_touches: int = 2, lookback: int = 40) -> Optional[float]:
    """Return DISTANCE from `entry` to nearest H4 supply (for SHORT) or
    demand (for LONG) zone within the lookback window. A "zone" is a
    cluster of swing extremes within `cluster_atr * atr` of each other
    with at least `min_touches` touches.

    LONG  → demand-zone-above (target = lowest cluster of swing highs
                              that lies ABOVE entry)
    SHORT → supply-zone-below (target = highest cluster of swing lows
                               that lies BELOW entry)

    NOTE: For a momentum/trend-follow entry, the *target* zone is the
    one in the direction of the trade — i.e. swing highs above for
    LONG, swing lows below for SHORT.
    """
    df = _normalize_ohlc(h4_df)
    if df is None or len(df) < 5:
        return None
    if atr is None or atr <= 0:
        return None
    try:
        window = df.iloc[-(lookback + 1):-1]
        if len(window) < (2 * 2 + 1):
            return None
        if direction.upper() == "LONG":
            highs = window["high"].values
            piv_idx = _find_pivots(highs, n=2, kind="high")
            pivots = [float(highs[i]) for i in piv_idx if highs[i] > entry]
        elif direction.upper() == "SHORT":
            lows = window["low"].values
            piv_idx = _find_pivots(lows, n=2, kind="low")
            pivots = [float(lows[i]) for i in piv_idx if lows[i] < entry]
        else:
            return None
        if not pivots:
            return None
        # Cluster pivots within cluster_atr * atr
        pivots.sort()  # ascending price
        tol = cluster_atr * atr
        clusters: List[List[float]] = []
        cur: List[float] = [pivots[0]]
        for p in pivots[1:]:
            if abs(p - cur[-1]) <= tol:
                cur.append(p)
            else:
                clusters.append(cur)
                cur = [p]
        clusters.append(cur)
        # Filter by min_touches
        valid = [c for c in clusters if len(c) >= min_touches]
        if not valid:
            return None
        # Choose nearest cluster in trade direction
        if direction.upper() == "LONG":
            # Want smallest cluster mean ABOVE entry
            means = [float(np.mean(c)) for c in valid]
            target = min(means)
            d = target - entry
        else:
            # Want largest cluster mean BELOW entry → smallest distance
            means = [float(np.mean(c)) for c in valid]
            target = max(means)
            d = entry - target
        if d <= 0 or not math.isfinite(d):
            return None
        return float(d)
    except Exception as e:  # pragma: no cover
        log.debug("find_h4_zone failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────
#  Symbol/regime ATR-SL multiplier lookup
# ──────────────────────────────────────────────────────────────────────
def lookup_atr_sl_mult(symbol: str, regime: str, default: float = 2.0) -> float:
    """Mirror config.SYMBOL_ATR_SL_OVERRIDE_REGIME / SYMBOL_ATR_SL_OVERRIDE
    when present; otherwise fall back to `default` (2.0)."""
    try:
        per_regime = _cfg_get("SYMBOL_ATR_SL_OVERRIDE_REGIME", None)
        if isinstance(per_regime, dict):
            sym_d = per_regime.get(symbol)
            if isinstance(sym_d, dict):
                v = sym_d.get(regime) or sym_d.get("default")
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        per_sym = _cfg_get("SYMBOL_ATR_SL_OVERRIDE", None)
        if isinstance(per_sym, dict):
            v = per_sym.get(symbol)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        atr_sl_mult = _cfg_get("ATR_SL_MULT", default)
        if isinstance(atr_sl_mult, (int, float)) and atr_sl_mult > 0:
            return float(atr_sl_mult)
    except Exception:  # pragma: no cover
        pass
    return float(default)


# ──────────────────────────────────────────────────────────────────────
#  Tiny utility — clamp
# ──────────────────────────────────────────────────────────────────────
def _clamp(x: float, lo: float, hi: float) -> float:
    if hi < lo:
        hi = lo
    return max(lo, min(hi, x))


# ──────────────────────────────────────────────────────────────────────
#  Main planner — pure function
# ──────────────────────────────────────────────────────────────────────
def compute_plan(symbol: str,
                 direction: str,
                 entry: float,
                 atr: float,
                 regime: str,
                 h1_df: Optional[pd.DataFrame],
                 m15_df: Optional[pd.DataFrame],
                 spread: float = 0.0) -> ExitPlan:
    """Build an `ExitPlan` for a fresh entry.

    Caller responsibility: only invoke after all entry gates pass + only
    when DYNAMIC_EXIT_ENABLED is True for this (symbol, strategy).
    """
    direction_u = (direction or "").upper()
    if direction_u not in ("LONG", "SHORT"):
        raise ValueError(f"dynamic_exit.compute_plan: bad direction={direction!r}")
    if atr is None or atr <= 0 or not math.isfinite(atr):
        raise ValueError(f"dynamic_exit.compute_plan: bad atr={atr!r}")
    if entry is None or entry <= 0 or not math.isfinite(entry):
        raise ValueError(f"dynamic_exit.compute_plan: bad entry={entry!r}")
    spread = max(0.0, float(spread or 0.0))

    # ── Resolve config knobs (with safe defaults) ────────────────────
    m15_lb       = int(_cfg_get("DE_M15_SWING_LB", M15_SWING_LB_DEF))
    h4_lb        = int(_cfg_get("DE_H4_SWING_LB", H4_SWING_LB_DEF))
    d1_lb        = int(_cfg_get("DE_D1_SWING_LB", D1_SWING_LB_DEF))
    atr_floor    = float(_cfg_get("DE_ATR_FLOOR_MULT", ATR_FLOOR_MULT_DEF))
    atr_cap      = float(_cfg_get("DE_ATR_CAP_MULT", ATR_CAP_MULT_DEF))
    buf_atr      = float(_cfg_get("DE_STRUCT_BUFFER_ATR", STRUCT_BUFFER_ATR_DEF))
    buf_spread   = float(_cfg_get("DE_SPREAD_BUFFER_MULT", SPREAD_BUFFER_MULT_DEF))
    tp1_r        = float(_cfg_get("DE_TP1_R", TP1_R_DEF))
    tp2_r_floor  = float(_cfg_get("DE_TP2_R_FLOOR", TP2_R_FLOOR_DEF))
    tp2_r_cap    = float(_cfg_get("DE_TP2_R_CAP", TP2_R_CAP_DEF))
    tp2_fallback = float(_cfg_get("DE_TP2_FALLBACK_R", TP2_FALLBACK_R_DEF))
    d1_max_r     = float(_cfg_get("DE_D1_SWING_MAX_R", D1_SWING_MAX_R_DEF))
    h4_max_r     = float(_cfg_get("DE_H4_ZONE_MAX_R", H4_ZONE_MAX_R_DEF))
    cluster_atr  = float(_cfg_get("DE_ZONE_CLUSTER_ATR", ZONE_CLUSTER_ATR_DEF))
    min_touches  = int(_cfg_get("DE_ZONE_MIN_TOUCHES", ZONE_MIN_TOUCHES_DEF))
    runner_r     = float(_cfg_get("DE_RUNNER_R", RUNNER_R_DEF))
    regime_map   = _cfg_get("DE_REGIME_TP_MULT", REGIME_TP_MULT_DEF) or REGIME_TP_MULT_DEF

    tags: List[str] = []

    # ── 1) Structural SL ─────────────────────────────────────────────
    atr_sl_mult = lookup_atr_sl_mult(symbol, regime)
    atr_sl_dist = atr * atr_sl_mult

    struct_sl_level = find_swing_extreme(m15_df, direction_u, lookback=m15_lb)
    struct_used = False
    if struct_sl_level is not None and math.isfinite(struct_sl_level):
        buffer = atr * buf_atr + spread * buf_spread
        if direction_u == "LONG":
            struct_sl_dist = (entry - struct_sl_level) + buffer
        else:
            struct_sl_dist = (struct_sl_level - entry) + buffer
        # Only use structural SL if it's actually beyond the ATR SL and on
        # the correct side of entry (positive distance).
        if struct_sl_dist > 0 and math.isfinite(struct_sl_dist):
            sl_dist = max(atr_sl_dist, struct_sl_dist)
            if struct_sl_dist > atr_sl_dist:
                struct_used = True
                tags.append("STRUCT_SL")
            else:
                tags.append("ATR_SL")
        else:
            sl_dist = atr_sl_dist
            tags.append("ATR_SL")
    else:
        sl_dist = atr_sl_dist
        tags.append("ATR_SL")

    sl_dist = _clamp(sl_dist, atr * atr_floor, atr * atr_cap)

    # ── 2) TP1: fixed R partial ──────────────────────────────────────
    tp1_dist = sl_dist * tp1_r

    # ── 3) TP2: structure cascade ────────────────────────────────────
    tp2_src = "FALLBACK_3R"
    h4_arr: Optional[pd.DataFrame] = None
    d1_arr: Optional[pd.DataFrame] = None
    try:
        h4_arr = aggregate_ohlc(h1_df, H4_BARS_PER)
    except Exception:
        h4_arr = None
    try:
        d1_arr = aggregate_ohlc(h1_df, D1_BARS_PER)
    except Exception:
        d1_arr = None

    d1_target_dist = None
    if d1_arr is not None:
        d1_target_dist = find_next_swing_extreme(d1_arr, direction_u,
                                                  lookback=d1_lb,
                                                  entry=entry)

    tp2_raw: float
    if d1_target_dist is not None and d1_target_dist <= sl_dist * d1_max_r:
        tp2_raw = d1_target_dist
        tp2_src = "D1_SWING"
        tags.append("D1_SWING")
    else:
        h4_zone_dist = None
        if h4_arr is not None:
            h4_zone_dist = find_h4_zone(h4_arr, direction_u, entry, atr,
                                         cluster_atr=cluster_atr,
                                         min_touches=min_touches,
                                         lookback=h4_lb)
        if h4_zone_dist is not None and h4_zone_dist <= sl_dist * h4_max_r:
            tp2_raw = h4_zone_dist
            tp2_src = "H4_ZONE"
            tags.append("H4_ZONE")
        else:
            tp2_raw = sl_dist * tp2_fallback
            tp2_src = "FALLBACK_3R"
            tags.append("FALLBACK_3R")

    # ── 4) Regime tilt on TP2 only ───────────────────────────────────
    regime_key = (regime or "unknown").lower()
    regime_mult = float(regime_map.get(regime_key, regime_map.get("unknown", 1.0)))
    tp2_dist = tp2_raw * regime_mult

    # ── 5) Clamp TP2 to [max(TP1*1.2, sl*1.8), sl*5.0] ───────────────
    lo = max(tp1_dist * 1.2, sl_dist * tp2_r_floor)
    hi = sl_dist * tp2_r_cap
    tp2_dist = _clamp(tp2_dist, lo, hi)

    # ── 6) Runner (Sub2) — 5R fixed ──────────────────────────────────
    runner_dist = sl_dist * runner_r

    plan = ExitPlan(
        sl_dist=float(sl_dist),
        tp1_dist=float(tp1_dist),
        tp2_dist=float(tp2_dist),
        runner_dist=float(runner_dist),
        tp2_source=tp2_src,
        atr_sl_dist=float(atr_sl_dist),
        struct_sl_used=bool(struct_used),
        regime_mult=float(regime_mult),
        regime=regime_key,
        source_tags=tags,
        tp1_R=float(tp1_dist / sl_dist) if sl_dist > 0 else float(tp1_r),
        tp2_R=float(tp2_dist / sl_dist) if sl_dist > 0 else float(tp2_fallback),
        runner_R=float(runner_dist / sl_dist) if sl_dist > 0 else float(runner_r),
        atr=float(atr),
        spread=float(spread),
        symbol=symbol or "",
        direction=direction_u,
    )

    if bool(_cfg_get("DE_LOG_PLANS", True)):
        log.debug(
            "DynamicExit plan symbol=%s dir=%s entry=%.5f atr=%.5f regime=%s "
            "sl=%.5f tp1=%.5f tp2=%.5f runner=%.5f src=%s tags=%s",
            symbol, direction_u, entry, atr, regime_key,
            sl_dist, tp1_dist, tp2_dist, runner_dist,
            tp2_src, ",".join(tags),
        )
    return plan


# ──────────────────────────────────────────────────────────────────────
#  Thin convenience class for the planner — matches the wiring spec
# ──────────────────────────────────────────────────────────────────────
class DynamicExitPlanner:
    """Brain holds an instance as `self._dynamic_exit`. Constructor
    takes only `state` (for state.get_candles(...)). The planner does
    NOT call state itself — brain passes the H1/M15 frames in — but the
    state handle is kept for forward-compat helpers (e.g. precomputed
    regime cache)."""

    def __init__(self, state=None):
        self.state = state

    # Exposed for tests
    @staticmethod
    def compute_plan(symbol: str,
                     direction: str,
                     entry: float,
                     atr: float,
                     regime: str,
                     h1_df: Optional[pd.DataFrame],
                     m15_df: Optional[pd.DataFrame],
                     spread: float = 0.0) -> ExitPlan:
        return compute_plan(symbol, direction, entry, atr, regime,
                            h1_df, m15_df, spread=spread)


# ──────────────────────────────────────────────────────────────────────
#  Synthetic self-test
# ──────────────────────────────────────────────────────────────────────
def _synth_h1(n_days: int = 60, start_px: float = 100.0,
              seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nbars = n_days * 24
    idx = pd.date_range("2026-01-01", periods=nbars, freq="1h", tz="UTC")
    drift = np.linspace(0, 8.0, nbars)
    noise = rng.normal(0, 0.4, nbars).cumsum() * 0.5
    base = start_px + drift + noise
    # Add periodic swing peaks/troughs every ~36h
    for i in range(36, nbars, 36):
        if (i // 36) % 2 == 0:
            base[i] += 3.0
        else:
            base[i] -= 3.0
    close = base
    open_ = np.concatenate([[start_px], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, nbars))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, nbars))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                      index=idx)
    df.index.name = "time"
    return df


def _synth_m15(h1: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nbars = len(h1) * 4
    idx = pd.date_range(h1.index[0], periods=nbars, freq="15min", tz="UTC")
    h1_close = h1["close"].values
    closes = np.repeat(h1_close, 4) + rng.normal(0, 0.15, nbars).cumsum() * 0.02
    open_ = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(open_, closes) + np.abs(rng.normal(0, 0.08, nbars))
    low = np.minimum(open_, closes) - np.abs(rng.normal(0, 0.08, nbars))
    final_px = closes[-1]
    # Embed a clear swing low ~1.5% below for LONG protective stop
    sl_idx = nbars - 28
    low[sl_idx] = final_px - max(0.5, 0.005 * final_px)
    high[sl_idx] = min(high[sl_idx], final_px - 0.10)
    # Make sure open/close don't violate high/low at the manipulated bar
    for arr in (open_, closes):
        arr[sl_idx] = max(low[sl_idx], min(high[sl_idx], arr[sl_idx]))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": closes},
                      index=idx)
    df.index.name = "time"
    return df


def _run_self_test() -> int:
    print("DynamicExitPlanner self-test starting...")
    h1 = _synth_h1(n_days=60)
    m15 = _synth_m15(h1)
    entry = float(h1["close"].iloc[-1])
    # Crude ATR(14) on H1
    atr_h1 = float((h1["high"] - h1["low"]).rolling(14).mean().iloc[-2])
    print(f"  entry={entry:.4f}  atr_h1={atr_h1:.4f}")

    # ── Test 1: LONG happy-path ──────────────────────────────────
    plan_l = compute_plan(symbol="TEST", direction="LONG",
                          entry=entry, atr=atr_h1, regime="trending",
                          h1_df=h1, m15_df=m15, spread=0.05)
    print(f"  LONG plan: sl={plan_l.sl_dist:.4f} tp1={plan_l.tp1_dist:.4f} "
          f"tp2={plan_l.tp2_dist:.4f} runner={plan_l.runner_dist:.4f} "
          f"src={plan_l.tp2_source} tags={plan_l.source_tags}")
    assert plan_l.sl_dist > 0, "LONG sl_dist must be positive"
    assert plan_l.tp1_dist > 0, "LONG tp1_dist must be positive"
    assert plan_l.tp2_dist > plan_l.tp1_dist, "TP2 must exceed TP1"
    assert plan_l.runner_dist >= plan_l.tp2_dist, "Runner must >= TP2"
    assert plan_l.tp1_R == pytest_isclose(plan_l.tp1_dist / plan_l.sl_dist), \
        "TP1 R-mult mismatch"
    # SL clamps
    assert plan_l.sl_dist >= atr_h1 * 1.0 - 1e-9, "SL below floor"
    assert plan_l.sl_dist <= atr_h1 * 3.0 + 1e-9, "SL above cap"
    # TP2 clamps
    assert plan_l.tp2_dist >= plan_l.sl_dist * 1.8 - 1e-9, "TP2 below floor"
    assert plan_l.tp2_dist <= plan_l.sl_dist * 5.0 + 1e-9, "TP2 above cap"

    # ── Test 2: SHORT happy-path ────────────────────────────────
    plan_s = compute_plan(symbol="TEST", direction="SHORT",
                          entry=entry, atr=atr_h1, regime="ranging",
                          h1_df=h1, m15_df=m15, spread=0.05)
    print(f"  SHORT plan: sl={plan_s.sl_dist:.4f} tp1={plan_s.tp1_dist:.4f} "
          f"tp2={plan_s.tp2_dist:.4f} runner={plan_s.runner_dist:.4f} "
          f"src={plan_s.tp2_source} tags={plan_s.source_tags}")
    assert plan_s.sl_dist > 0, "SHORT sl_dist must be positive"
    assert plan_s.tp2_dist > plan_s.tp1_dist, "SHORT TP2 must exceed TP1"
    # Ranging regime should pull TP2 in (mult=0.80)
    assert plan_s.regime_mult == 0.80, f"Ranging mult wrong: {plan_s.regime_mult}"

    # ── Test 3: Unknown regime falls back to 1.00 ───────────────
    plan_u = compute_plan(symbol="TEST", direction="LONG",
                          entry=entry, atr=atr_h1, regime="space_alien",
                          h1_df=h1, m15_df=m15, spread=0.0)
    assert plan_u.regime_mult == 1.00, f"Unknown regime fallback failed: {plan_u.regime_mult}"

    # ── Test 4: Aggregation helpers ─────────────────────────────
    h4 = aggregate_ohlc(h1, H4_BARS_PER)
    d1 = aggregate_ohlc(h1, D1_BARS_PER)
    assert h4 is not None and len(h4) > 0, "H4 aggregation produced empty frame"
    assert d1 is not None and len(d1) > 0, "D1 aggregation produced empty frame"
    # 60 days × 24 h = 1440 H1 bars → 360 H4 bars, 60 D1 bars
    assert len(h4) == 360, f"H4 expected 360 bars got {len(h4)}"
    assert len(d1) == 60, f"D1 expected 60 bars got {len(d1)}"
    print(f"  H4 bars={len(h4)} D1 bars={len(d1)}")

    # ── Test 5: Swing-extreme detector — finds lowest low ───────
    sw = find_swing_extreme(m15, "LONG", lookback=40)
    assert sw is not None and sw < entry, "LONG swing extreme should be below entry"
    sh = find_swing_extreme(m15, "SHORT", lookback=40)
    assert sh is not None and sh > 0, "SHORT swing extreme should be a valid high"

    # ── Test 6: Bad input rejection ─────────────────────────────
    rejected = 0
    for bad_dir in ("WAT", "", None):
        try:
            compute_plan("X", bad_dir, entry, atr_h1, "trending", h1, m15)
        except ValueError:
            rejected += 1
    assert rejected == 3, f"bad direction not rejected: {rejected}/3"
    try:
        compute_plan("X", "LONG", entry, -1.0, "trending", h1, m15)
        assert False, "negative ATR should raise"
    except ValueError:
        pass
    try:
        compute_plan("X", "LONG", 0.0, atr_h1, "trending", h1, m15)
        assert False, "zero entry should raise"
    except ValueError:
        pass

    # ── Test 7: Planner class wrapper ───────────────────────────
    planner = DynamicExitPlanner(state=None)
    plan_c = planner.compute_plan("TEST", "LONG", entry, atr_h1, "volatile",
                                   h1, m15, spread=0.10)
    assert plan_c.regime_mult == 1.15, "Volatile regime mult wrong"

    # ── Test 8: ExitPlan.to_dict serialisable ───────────────────
    d = plan_c.to_dict()
    assert "sl_dist" in d and "tp2_source" in d, "to_dict missing keys"
    assert isinstance(d["source_tags"], list), "source_tags not list"

    # ── Test 9: Missing H1 frame → still returns plan (fallback) ─
    plan_nh = compute_plan("TEST", "LONG", entry, atr_h1, "trending",
                            None, m15, spread=0.0)
    assert plan_nh.tp2_source == "FALLBACK_3R", \
        f"Without H1 should fall back to 3R, got {plan_nh.tp2_source}"

    # ── Test 10: spread buffer materially widens SL when present ─
    p_zero = compute_plan("TEST", "LONG", entry, atr_h1, "unknown",
                           h1, m15, spread=0.0)
    p_high = compute_plan("TEST", "LONG", entry, atr_h1, "unknown",
                           h1, m15, spread=atr_h1 * 0.5)  # huge spread
    # Either equal (structural SL not selected) or wider with spread
    assert p_high.sl_dist >= p_zero.sl_dist - 1e-9, \
        "Spread buffer should not tighten SL"

    print("DynamicExitPlanner self-test PASSED.")
    return 0


def pytest_isclose(b: float, tol: float = 1e-6):
    """Tiny shim so 'a == pytest_isclose(b)' is a tolerant equality."""
    class _C:
        def __eq__(self, a):  # noqa: D401
            return abs(float(a) - float(b)) < tol
        def __repr__(self):
            return f"isclose({b})"
    return _C()


if __name__ == "__main__":
    sys.exit(_run_self_test())
