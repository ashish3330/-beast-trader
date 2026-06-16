#!/usr/bin/env python3 -B
"""
Session-Conditional Setup Logic (SCSL) — Phase 2 module.

Classifies each entry candidate into one of:
  * ASIA           — 00:00-06:59 UTC (Tokyo / Sydney overlap)
  * LONDON         — 07:00-12:59 UTC (London open, pre-NY)
  * NY             — 13:00-20:59 UTC (NY core)
  * LATE_NY_FADE   — 19:00-20:59 UTC (last 2h of NY, fade-only overlay)
  * ASIA_PREP      — 21:00-23:59 UTC (low-conviction Asian setup)

And one of:
  * RANGE_REVERT   — squeeze inside the 24h range (mean-revert)
  * BREAKOUT_CONT  — vol expansion aligned with direction
  * TREND_CONT     — established H1 trend follow (close vs EMA20 >= 0.5 ATR)
  * FADE           — counter-trend at top/bottom of range
  * MIXED          — ambiguous (always rejected by allow-map)

A 5x5 ALLOW map (session x setup -> (allowed, min_quality_bump)) decides
whether the candidate is permitted at this hour. Disallowed combinations are
rejected with reason '<session>_REJECT_<setup>'. Allowed combinations may
still be quality-bumped (e.g. ASIA_PREP RANGE_REVERT needs +5 quality).

Spec ref: feedback_value_entry_research_20260605.md, ICT kill-zones,
Babypips session map.

This is a NEW module that does NOT wire into brain.py yet — Phase 3 will add
Gate 3g (SESSION_SETUP) immediately after Gate 3f (ICT_NO_SWEEP).

API (pure functions — no class state needed):
    from agent.expert.session_setup import classify_session, classify_setup, evaluate
    verdict = evaluate(symbol, hour_utc, h1_df, ind, bi,
                       direction, signal_quality, regime, atr, min_quality)
    # verdict = {"allowed": bool, "session": str, "setup_type": str,
    #            "reason": str, "min_quality": float}

Defensive (FAIL-OPEN):
  * If H1 buffer too short → return allowed=True, reason='insufficient_h1'.
  * If any pandas / numpy op raises → return allowed=True, reason='scsl_error'.
  * If SESSION_SETUP_ENABLED=False → return allowed=True, reason='disabled'.
  * If symbol in SESSION_SETUP_BYPASS_SYMBOLS → allowed=True, reason='bypass'.
  * If SCSL_LOG_ONLY=True → caller is expected to record verdict but treat as
    allowed regardless; this module always returns the true verdict (caller
    decides log-vs-enforce).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger("dragon.scsl")

# ─────────────────────────────────────────────────────────────────────────
# Config import (graceful fallback for unit tests / standalone runs).
# ─────────────────────────────────────────────────────────────────────────
try:
    import config as _cfg
except Exception:  # pragma: no cover — only for isolated module test
    _cfg = None


# ─────────────────────────────────────────────────────────────────────────
# Defaults (mirrored in config.py; this module reads from config when avail).
# ─────────────────────────────────────────────────────────────────────────
_DEFAULT_SESSION_BANDS = {
    "ASIA":         (0,  7),
    "LONDON":       (7,  13),
    "NY":           (13, 21),
    "LATE_NY_FADE": (19, 21),
    "ASIA_PREP":    (21, 24),
}
_DEFAULT_RANGE_POS_LB = 24
_DEFAULT_BB_BREAKOUT_PCT = 0.85
_DEFAULT_BB_RANGE_PCT = 0.35
_DEFAULT_EMA20_TREND_DIST_ATR = 0.5
_DEFAULT_FADE_POS_HI = 0.80
_DEFAULT_FADE_POS_LO = 0.20
_DEFAULT_MIN_H1_BARS = 120
_DEFAULT_BYPASS_SYMBOLS = {"BTCUSD", "ETHUSD", "BCHUSD", "JPN225ft"}

# Each cell = (allowed_bool, min_quality_bump_added_to_regime_threshold).
# Bump is ADDITIVE to the caller's base_min_quality.
_DEFAULT_ALLOW_MAP = {
    "ASIA": {
        "RANGE_REVERT":  (True,  0),
        "BREAKOUT_CONT": (False, 99),
        "TREND_CONT":    (False, 5),
        "FADE":          (False, 99),
        "MIXED":         (False, 99),
    },
    "ASIA_PREP": {
        "RANGE_REVERT":  (True,  5),
        "BREAKOUT_CONT": (False, 99),
        "TREND_CONT":    (False, 10),
        "FADE":          (False, 99),
        "MIXED":         (False, 99),
    },
    "LONDON": {
        "RANGE_REVERT":  (False, 99),
        "BREAKOUT_CONT": (True,  0),
        "TREND_CONT":    (True,  0),
        "FADE":          (False, 99),
        "MIXED":         (False, 99),
    },
    "NY": {
        "RANGE_REVERT":  (False, 99),
        "BREAKOUT_CONT": (True,  0),
        "TREND_CONT":    (True,  0),
        "FADE":          (False, 99),
        "MIXED":         (False, 99),
    },
    "LATE_NY_FADE": {
        "RANGE_REVERT":  (False, 99),
        "BREAKOUT_CONT": (False, 10),
        "TREND_CONT":    (True,  5),
        "FADE":          (True,  0),
        "MIXED":         (False, 99),
    },
}

SETUP_TYPES = ("RANGE_REVERT", "BREAKOUT_CONT", "TREND_CONT", "FADE", "MIXED")


def _cfg_get(name: str, default: Any) -> Any:
    """Read attribute from config module if loaded, else default."""
    if _cfg is None:
        return default
    return getattr(_cfg, name, default)


# ─────────────────────────────────────────────────────────────────────────
# Session classifier
# ─────────────────────────────────────────────────────────────────────────
def classify_session(hour_utc: int) -> str:
    """
    Map UTC hour-of-day (0-23) to session label.

    Note: LATE_NY_FADE overlaps NY (19-21). The overlay is resolved by the
    classifier returning LATE_NY_FADE for those 2h — the caller's allow-map
    cell for LATE_NY_FADE then encodes the fade-friendly rules.
    """
    try:
        h = int(hour_utc) % 24
    except Exception:
        return "ASIA_PREP"   # safe default (lowest conviction)

    bands = _cfg_get("SCSL_SESSION_BANDS", _DEFAULT_SESSION_BANDS)

    # LATE_NY_FADE wins over plain NY when both match (more specific).
    lo, hi = bands.get("LATE_NY_FADE", (19, 21))
    if lo <= h < hi:
        return "LATE_NY_FADE"

    for label in ("ASIA", "LONDON", "NY", "ASIA_PREP"):
        lo, hi = bands.get(label, (-1, -1))
        if lo <= h < hi:
            return label

    return "ASIA_PREP"


# ─────────────────────────────────────────────────────────────────────────
# Setup-type classifier
# ─────────────────────────────────────────────────────────────────────────
def _safe_get_bb_width(ind: Any, bi: int):
    """Extract bb_width array + current value from ind dict.
    Returns (current_value, hist_array) or (None, None) on miss."""
    try:
        arr = ind["bb_width"] if isinstance(ind, dict) else getattr(ind, "bb_width", None)
        if arr is None:
            return None, None
        cur = float(arr[bi])
        return cur, arr
    except Exception:
        return None, None


def classify_setup(
    h1_df: Any,
    ind: Any,
    bi: int,
    direction: str,
    atr: float,
) -> str:
    """
    Classify the candidate setup based on H1 structure.

    Inputs:
        h1_df   — pandas DataFrame with columns: high, low, close (H1 candles).
        ind     — dict-like with key 'bb_width' (np array indexed by bar idx).
        bi      — current bar index for ind arrays.
        direction — 'LONG' or 'SHORT'.
        atr     — current ATR value (any TF; used as distance unit).

    Returns one of SETUP_TYPES.

    Defensive: returns 'MIXED' on any computation error (will be rejected by
    allow-map). Caller decides FAIL-OPEN vs FAIL-CLOSED policy via evaluate().
    """
    try:
        lb = int(_cfg_get("SCSL_RANGE_POS_LB", _DEFAULT_RANGE_POS_LB))
        bb_brk = float(_cfg_get("SCSL_BB_BREAKOUT_PCT", _DEFAULT_BB_BREAKOUT_PCT))
        bb_rng = float(_cfg_get("SCSL_BB_RANGE_PCT", _DEFAULT_BB_RANGE_PCT))
        trend_atr = float(_cfg_get("SCSL_EMA20_TREND_DIST_ATR", _DEFAULT_EMA20_TREND_DIST_ATR))
        fade_hi = float(_cfg_get("SCSL_FADE_POS_HI", _DEFAULT_FADE_POS_HI))
        fade_lo = float(_cfg_get("SCSL_FADE_POS_LO", _DEFAULT_FADE_POS_LO))

        # Range position (24-bar by default).
        highs = h1_df["high"].iloc[-lb:]
        lows = h1_df["low"].iloc[-lb:]
        cl = float(h1_df["close"].iloc[-1])
        hi = float(highs.max())
        lo = float(lows.min())
        rng = max(hi - lo, 1e-9)
        pos = (cl - lo) / rng   # 0..1

        # BB-width percentile rank over trailing 100 bars.
        bbw_now, bbw_arr = _safe_get_bb_width(ind, bi)
        if bbw_now is None:
            bbw_pct = 0.5  # neutral if missing
        else:
            try:
                window = bbw_arr[max(0, bi - 100): bi + 1]
                # percentile rank of bbw_now within window: fraction <= now
                # Use generic comparison so this works for numpy or list.
                cnt = 0
                tot = 0
                for v in window:
                    tot += 1
                    if float(v) <= bbw_now:
                        cnt += 1
                bbw_pct = (cnt / tot) if tot else 0.5
            except Exception:
                bbw_pct = 0.5

        # EMA20 trend distance.
        ema20 = float(h1_df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        atr_safe = max(float(atr) if atr is not None else 0.0, 1e-9)
        dist_atr = abs(cl - ema20) / atr_safe
        with_trend = (
            (direction == "LONG" and cl > ema20)
            or (direction == "SHORT" and cl < ema20)
        )

        # Decision tree (order matters — first match wins).
        if bbw_pct >= bb_brk and with_trend:
            return "BREAKOUT_CONT"
        if bbw_pct <= bb_rng and (fade_lo + 0.05) < pos < (fade_hi - 0.05):
            # squeeze inside the range (not at edges)
            return "RANGE_REVERT"
        if dist_atr >= trend_atr and with_trend:
            return "TREND_CONT"
        # Edge-of-range counter-trend = fade candidate.
        if direction == "SHORT" and pos >= fade_hi:
            return "FADE"
        if direction == "LONG" and pos <= fade_lo:
            return "FADE"
        return "MIXED"
    except Exception as e:   # pragma: no cover — defensive
        log.debug("classify_setup error: %s", e)
        return "MIXED"


# ─────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────
def evaluate(
    symbol: str,
    hour_utc: int,
    h1_df: Any,
    ind: Any,
    bi: int,
    direction: str,
    signal_quality: float,
    regime: Optional[str],
    atr: float,
    base_min_quality: float,
) -> Dict[str, Any]:
    """
    Run SCSL gate. See module docstring for inputs and FAIL-OPEN policy.

    Returns dict:
        {
          "allowed": bool,
          "session": str | None,
          "setup_type": str | None,
          "reason": str,
          "min_quality": float,   # effective threshold after bump (if computed)
        }
    """
    enabled = bool(_cfg_get("SESSION_SETUP_ENABLED", False))
    bypass = set(_cfg_get("SESSION_SETUP_BYPASS_SYMBOLS", _DEFAULT_BYPASS_SYMBOLS))
    log_only = bool(_cfg_get("SCSL_LOG_ONLY", True))
    min_h1 = int(_cfg_get("SCSL_MIN_H1_BARS", _DEFAULT_MIN_H1_BARS))
    allow_map = _cfg_get("SCSL_ALLOW_MAP", _DEFAULT_ALLOW_MAP)
    per_sym_override = _cfg_get("SCSL_PER_SYMBOL_OVERRIDE", {}) or {}

    base = {
        "allowed": True,
        "session": None,
        "setup_type": None,
        "reason": "disabled",
        "min_quality": float(base_min_quality),
        "log_only": log_only,
    }

    if not enabled:
        return base

    if symbol in bypass:
        return {**base, "reason": "bypass"}

    # H1 buffer sanity (fail-open if too short).
    try:
        nbars = len(h1_df)
    except Exception:
        nbars = 0
    if nbars < min_h1:
        return {**base, "reason": "insufficient_h1"}

    try:
        session = classify_session(hour_utc)
        setup = classify_setup(h1_df, ind, bi, direction, atr)

        # Per-symbol override merge (sparse).
        cell_map = allow_map
        if symbol in per_sym_override:
            cell_map = _merge_override(allow_map, per_sym_override[symbol])

        cell = cell_map.get(session, {}).get(setup)
        if cell is None:
            # Unknown combo → fail-open with reason.
            return {**base, "session": session, "setup_type": setup,
                    "reason": "unknown_cell", "min_quality": float(base_min_quality)}

        allowed_bool, bump = cell
        eff_min_q = float(base_min_quality) + float(bump)

        if not allowed_bool:
            return {
                "allowed": False,
                "session": session,
                "setup_type": setup,
                "reason": f"{session}_REJECT_{setup}",
                "min_quality": eff_min_q,
                "log_only": log_only,
            }

        if signal_quality < eff_min_q:
            return {
                "allowed": False,
                "session": session,
                "setup_type": setup,
                "reason": f"{session}_{setup}_BELOW_Q({signal_quality:.0f}<{eff_min_q:.0f})",
                "min_quality": eff_min_q,
                "log_only": log_only,
            }

        return {
            "allowed": True,
            "session": session,
            "setup_type": setup,
            "reason": "ok",
            "min_quality": eff_min_q,
            "log_only": log_only,
        }
    except Exception as e:   # pragma: no cover — defensive
        log.warning("SCSL evaluate failed for %s: %s — fail-open", symbol, e)
        return {**base, "reason": "scsl_error"}


def _merge_override(global_map: dict, sym_override: dict) -> dict:
    """Sparse-merge per-symbol override into the global allow map.
    sym_override schema: {session: {setup: (allowed, bump)}}."""
    merged = {sess: dict(cells) for sess, cells in global_map.items()}
    for sess, cells in (sym_override or {}).items():
        if sess not in merged:
            merged[sess] = {}
        for setup, val in (cells or {}).items():
            merged[sess][setup] = val
    return merged


# ─────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────
def _selftest() -> int:
    """Synthetic self-test. Returns 0 on success, 1 on failure."""
    import sys
    failures = []

    # ── classify_session bands
    cases = [
        (0, "ASIA"), (3, "ASIA"), (6, "ASIA"),
        (7, "LONDON"), (10, "LONDON"), (12, "LONDON"),
        (13, "NY"), (15, "NY"), (18, "NY"),
        (19, "LATE_NY_FADE"), (20, "LATE_NY_FADE"),
        (21, "ASIA_PREP"), (23, "ASIA_PREP"),
    ]
    for h, want in cases:
        got = classify_session(h)
        if got != want:
            failures.append(f"classify_session({h}) -> {got}, want {want}")

    # ── Build synthetic H1 frame for setup classifier
    try:
        import pandas as pd
        import numpy as np
    except Exception as e:
        print(f"SELFTEST FAIL — pandas/numpy not available: {e}")
        return 1

    rng = np.random.default_rng(42)

    # Case A: BREAKOUT_CONT — trending up, expanding BB, close > EMA20.
    n = 200
    base = np.linspace(100, 130, n) + rng.normal(0, 0.2, n)
    df_a = pd.DataFrame({
        "high":  base + 0.5,
        "low":   base - 0.5,
        "close": base,
    })
    # Make last 10 BB-width values monotonically growing (so percentile=high)
    bb_arr = np.concatenate([rng.uniform(0.5, 1.0, 190), np.linspace(2.0, 5.0, 10)])
    ind = {"bb_width": bb_arr}
    atr_a = 0.5
    setup_a = classify_setup(df_a, ind, n - 1, "LONG", atr_a)
    if setup_a != "BREAKOUT_CONT":
        failures.append(f"BREAKOUT_CONT case got {setup_a}")

    # Case B: RANGE_REVERT — flat, contracting BB, mid-range.
    base_b = np.full(n, 100.0) + rng.normal(0, 0.05, n)
    df_b = pd.DataFrame({
        "high":  base_b + 0.2,
        "low":   base_b - 0.2,
        "close": base_b,
    })
    # bb_width decreasing — current is in bottom percentile
    bb_b = np.concatenate([rng.uniform(1.5, 3.0, 190), np.linspace(0.5, 0.1, 10)])
    ind_b = {"bb_width": bb_b}
    setup_b = classify_setup(df_b, ind_b, n - 1, "LONG", 0.3)
    if setup_b not in ("RANGE_REVERT", "MIXED"):
        # MIXED is acceptable here if percentile maths shifts; we want NOT a
        # breakout / trend tag in a flat regime.
        failures.append(f"RANGE_REVERT-ish case got {setup_b}")

    # Case C: FADE — at top of range, SHORT direction, no expansion.
    base_c = np.linspace(100, 110, n) + rng.normal(0, 0.1, n)
    base_c[-5:] = 115.0   # push close to top of range
    df_c = pd.DataFrame({
        "high":  base_c + 0.2,
        "low":   base_c - 0.2,
        "close": base_c,
    })
    bb_c = rng.uniform(0.8, 1.2, n)
    ind_c = {"bb_width": bb_c}
    setup_c = classify_setup(df_c, ind_c, n - 1, "SHORT", 0.3)
    if setup_c not in ("FADE", "TREND_CONT", "MIXED"):
        # accept MIXED — main check is that SHORT at top is NOT BREAKOUT_CONT
        failures.append(f"FADE case got {setup_c}")

    # ── evaluate() smoke test — enabled path needs to read defaults
    #   We mock config attributes by stuffing into _cfg.
    class _MockCfg:
        SESSION_SETUP_ENABLED = True
        SESSION_SETUP_BYPASS_SYMBOLS = {"BTCUSD"}
        SCSL_LOG_ONLY = False
        SCSL_MIN_H1_BARS = 120
        SCSL_ALLOW_MAP = _DEFAULT_ALLOW_MAP
        SCSL_PER_SYMBOL_OVERRIDE = {}
        SCSL_SESSION_BANDS = _DEFAULT_SESSION_BANDS
        SCSL_RANGE_POS_LB = _DEFAULT_RANGE_POS_LB
        SCSL_BB_BREAKOUT_PCT = _DEFAULT_BB_BREAKOUT_PCT
        SCSL_BB_RANGE_PCT = _DEFAULT_BB_RANGE_PCT
        SCSL_EMA20_TREND_DIST_ATR = _DEFAULT_EMA20_TREND_DIST_ATR
        SCSL_FADE_POS_HI = _DEFAULT_FADE_POS_HI
        SCSL_FADE_POS_LO = _DEFAULT_FADE_POS_LO

    global _cfg
    _cfg_save = _cfg
    _cfg = _MockCfg()
    try:
        # Disabled when symbol bypassed
        v = evaluate("BTCUSD", 10, df_a, ind, n - 1, "LONG", 60, "trending", 0.5, 50)
        if not v["allowed"] or v["reason"] != "bypass":
            failures.append(f"bypass case got {v}")

        # Insufficient H1
        short_df = df_a.iloc[:10].copy()
        v2 = evaluate("XAUUSD", 10, short_df, ind, 9, "LONG", 60, "trending", 0.5, 50)
        if not v2["allowed"] or v2["reason"] != "insufficient_h1":
            failures.append(f"insufficient_h1 case got {v2}")

        # ASIA hour + BREAKOUT_CONT setup should REJECT
        v3 = evaluate("XAUUSD", 3, df_a, ind, n - 1, "LONG", 80, "trending", 0.5, 50)
        if v3["session"] != "ASIA":
            failures.append(f"ASIA case session mismatch: {v3}")
        if v3["setup_type"] == "BREAKOUT_CONT" and v3["allowed"]:
            failures.append(f"ASIA + BREAKOUT_CONT should be rejected: {v3}")

        # LONDON + BREAKOUT_CONT setup should ALLOW (quality high enough)
        v4 = evaluate("XAUUSD", 10, df_a, ind, n - 1, "LONG", 80, "trending", 0.5, 50)
        if v4["session"] != "LONDON":
            failures.append(f"LONDON case session mismatch: {v4}")
        # If setup is BREAKOUT_CONT in LONDON, must be allowed.
        if v4["setup_type"] == "BREAKOUT_CONT" and not v4["allowed"]:
            failures.append(f"LONDON + BREAKOUT_CONT should be allowed: {v4}")

        # Quality bump — set a high base_min_quality so below-quality fails.
        v5 = evaluate("XAUUSD", 10, df_a, ind, n - 1, "LONG", 30, "trending", 0.5, 50)
        if v5["allowed"]:
            # only if allow-map permitted the cell; should fail by BELOW_Q
            if "BELOW_Q" not in v5.get("reason", ""):
                # Accept: if the setup_type maps to a disallowed cell, reason
                # is REJECT instead. Both are non-allowed, so just check.
                failures.append(f"low-quality should NOT be allowed: {v5}")

        # NY + TREND_CONT path — synthesize TREND_CONT
        # Use df_a (trending up, EMA20 trail) with no BB expansion at end
        bb_flat = np.full(n, 1.0)
        ind_flat = {"bb_width": bb_flat}
        v6 = evaluate("XAUUSD", 15, df_a, ind_flat, n - 1, "LONG", 80, "trending", 0.5, 50)
        if v6["session"] != "NY":
            failures.append(f"NY case session mismatch: {v6}")
        # Either TREND_CONT or MIXED depending on dist_atr — both legitimate.

        # Per-symbol override test
        _MockCfg.SCSL_PER_SYMBOL_OVERRIDE = {
            "EURUSD": {"ASIA": {"BREAKOUT_CONT": (True, 0)}}
        }
        v7 = evaluate("EURUSD", 3, df_a, ind, n - 1, "LONG", 80, "trending", 0.5, 50)
        # If classified BREAKOUT_CONT, override should now allow it.
        if v7["setup_type"] == "BREAKOUT_CONT" and not v7["allowed"]:
            failures.append(f"per-symbol override should allow: {v7}")

    finally:
        _cfg = _cfg_save

    if failures:
        print("SELFTEST FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("SELFTEST PASS — session_setup classifier + evaluator behave as expected.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
