"""
agent.expert.day_type_classifier
════════════════════════════════
Dalton / Steidlmayer market-profile day-type classifier.

Each session has a CHARACTER, and the same strategy mix performs very
differently across day types:

    • TREND_UP / TREND_DOWN  — one-directional auction, range expansion,
                                low overlap of bar bodies. Momentum LONG
                                (or SHORT) entries shine; counter-trend
                                fades get steamrolled.
    • NORMAL                  — balanced auction, range contained near
                                Initial Balance, mid-bar standard
                                deviation low. SR / mean-reversion
                                edges out momentum.
    • DOUBLE_DIST             — two separate value areas built within
                                one session (chop). Fades at extremes
                                work; breakouts fail.
    • UNKNOWN                 — not enough bars / data hiccup → caller
                                should fall back to neutral behaviour.

Heuristic (light, deterministic, no sklearn / no scipy):

    1. INITIAL BALANCE       = range of first ``initial_balance_bars``
                                H1 bars of the session.
    2. SESSION RANGE         = max(highs) − min(lows).
    3. IB_EXTENSION_RATIO    = SESSION_RANGE / IB_RANGE.
    4. NET_MOVE              = close[-1] − open[0].
    5. TREND_STRENGTH        = abs(NET_MOVE) / SESSION_RANGE.
    6. MID_DISPERSION        = stdev(bar mids) / max(ATR14, 1e-9).
    7. BIMODALITY            = fraction of bars whose mid sits closer to
                                EITHER session high OR session low than
                                to session midpoint (≥ 0.65 ⇒ double dist).

Decision tree (cheap, robust):

    if not enough bars            → UNKNOWN
    elif IB_EXTENSION_RATIO ≥ 2.0 and TREND_STRENGTH ≥ 0.55:
        TREND_UP   if NET_MOVE > 0 else TREND_DOWN
    elif BIMODALITY ≥ 0.65 and IB_EXTENSION_RATIO ≥ 1.4:
        DOUBLE_DIST
    elif IB_EXTENSION_RATIO ≤ 1.5:
        NORMAL
    else:
        UNKNOWN

All thresholds are conservative defaults from Dalton "Mind Over Markets"
+ Steidlmayer "Markets and Market Logic"; tuneable via classify_day_type
kwargs.

Public API
──────────
    classify_day_type(highs, lows, closes, opens, atr14,
                      initial_balance_bars=2, ...) -> dict
        Returns a verdict dict::
            {
              "day_type":  str,                # TREND_UP / ... / UNKNOWN
              "ib_range":  float,
              "session_range": float,
              "ib_ext_ratio": float,
              "net_move":  float,
              "trend_strength": float,
              "mid_dispersion": float,
              "bimodality": float,
              "n_bars": int,
              "reason": str,                  # short tag
            }

    apply_day_type_routing(verdict, signal_source, direction, raw_score,
                           size_mult, cfg=None) -> dict
        Pure helper used by orchestrator. Returns dict::
            {
              "score_delta":   float,   # add to raw_score (momentum boost/gate)
              "size_mult_tilt": float,  # multiplicative size adjustment
              "reject":        bool,    # True if should REJECT (we set False
                                          for [[feedback_no_skip_trades]] —
                                          callers may choose to honour it)
              "reason":        str,
            }

Self-test
─────────
    python3 -B agent/expert/day_type_classifier.py --self-test

Fail-open contract: every public function catches exceptions and returns
a UNKNOWN / neutral verdict so a data hiccup never blocks the brain.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Sequence

log = logging.getLogger("dragon.expert.day_type_classifier")


# ── Default thresholds (Dalton / Steidlmayer literature) ───────────────
DEFAULT_TREND_IB_EXT = 2.0     # session range ≥ 2× IB ⇒ extension day
DEFAULT_TREND_STRENGTH = 0.55  # |net move| / range ≥ 0.55 ⇒ trend
DEFAULT_DOUBLE_DIST_BIMOD = 0.65
DEFAULT_DOUBLE_DIST_IB_EXT = 1.4
DEFAULT_NORMAL_IB_EXT = 1.5

DAY_TYPES = ("TREND_UP", "TREND_DOWN", "NORMAL", "DOUBLE_DIST", "UNKNOWN")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _stdev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def _bimodality_fraction(mids: Sequence[float], hi: float, lo: float) -> float:
    """Fraction of bar mid-points that sit closer to EITHER session high
    or session low than to the session midpoint. ≥ 0.65 ⇒ two distinct
    value areas (double distribution day in Steidlmayer's lexicon)."""
    if not mids or hi <= lo:
        return 0.0
    mid_px = 0.5 * (hi + lo)
    span = hi - lo
    if span <= 0:
        return 0.0
    near_extremes = 0
    for m in mids:
        d_hi = abs(m - hi)
        d_lo = abs(m - lo)
        d_mid = abs(m - mid_px)
        # "near extreme" if closer to hi/lo than to mid AND within 30%
        # of span from that extreme.
        if min(d_hi, d_lo) < d_mid and min(d_hi, d_lo) <= 0.30 * span:
            near_extremes += 1
    return near_extremes / float(len(mids))


# ─────────────────────────────────────────────────────────────────────────
#  Main classifier
# ─────────────────────────────────────────────────────────────────────────
def classify_day_type(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    opens: Sequence[float],
    atr14: float,
    initial_balance_bars: int = 2,
    *,
    trend_ib_ext: float = DEFAULT_TREND_IB_EXT,
    trend_strength: float = DEFAULT_TREND_STRENGTH,
    bimod_thr: float = DEFAULT_DOUBLE_DIST_BIMOD,
    bimod_ib_ext: float = DEFAULT_DOUBLE_DIST_IB_EXT,
    normal_ib_ext: float = DEFAULT_NORMAL_IB_EXT,
) -> Dict[str, Any]:
    """Classify the current session as TREND_UP/TREND_DOWN/NORMAL/
    DOUBLE_DIST/UNKNOWN given H1 OHLC arrays for the session-so-far.

    Fail-open: any exception → UNKNOWN. The orchestrator is expected to
    treat UNKNOWN as a no-op pass-through (no score/size adjustment).
    """
    try:
        n = min(len(highs), len(lows), len(closes), len(opens))
        ib = max(1, int(initial_balance_bars or 2))
        # Need at least IB + 1 bars to have any extension to measure.
        if n < ib + 1:
            return {
                "day_type": "UNKNOWN",
                "reason": "INSUFFICIENT_BARS",
                "n_bars": n,
                "ib_range": 0.0,
                "session_range": 0.0,
                "ib_ext_ratio": 0.0,
                "net_move": 0.0,
                "trend_strength": 0.0,
                "mid_dispersion": 0.0,
                "bimodality": 0.0,
            }

        # Take just the last n bars (caller may pass full series).
        H = [_safe_float(x) for x in highs[-n:]]
        L = [_safe_float(x) for x in lows[-n:]]
        C = [_safe_float(x) for x in closes[-n:]]
        O = [_safe_float(x) for x in opens[-n:]]
        atr = max(_safe_float(atr14), 1e-9)

        ib_high = max(H[:ib])
        ib_low = min(L[:ib])
        ib_range = max(ib_high - ib_low, 1e-9)

        sess_high = max(H)
        sess_low = min(L)
        sess_range = max(sess_high - sess_low, 1e-9)

        ib_ext = sess_range / ib_range
        net_move = C[-1] - O[0]
        trend_str = abs(net_move) / sess_range
        mids = [0.5 * (h + l) for h, l in zip(H, L)]
        mid_disp = _stdev(mids) / atr
        bimod = _bimodality_fraction(mids, sess_high, sess_low)

        # Decision tree
        day_type = "UNKNOWN"
        reason = "AMBIGUOUS"
        if ib_ext >= trend_ib_ext and trend_str >= trend_strength:
            day_type = "TREND_UP" if net_move > 0 else "TREND_DOWN"
            reason = "IB_EXT_AND_TREND_STR"
        elif bimod >= bimod_thr and ib_ext >= bimod_ib_ext:
            day_type = "DOUBLE_DIST"
            reason = "BIMODAL_VALUE_AREAS"
        elif ib_ext <= normal_ib_ext:
            day_type = "NORMAL"
            reason = "CONTAINED_IN_IB"

        return {
            "day_type": day_type,
            "reason": reason,
            "n_bars": n,
            "ib_range": ib_range,
            "session_range": sess_range,
            "ib_ext_ratio": ib_ext,
            "net_move": net_move,
            "trend_strength": trend_str,
            "mid_dispersion": mid_disp,
            "bimodality": bimod,
        }
    except Exception as e:  # pragma: no cover — fail-open
        log.debug("classify_day_type failed: %s", e)
        return {
            "day_type": "UNKNOWN",
            "reason": "EXCEPTION:%s" % type(e).__name__,
            "n_bars": 0,
            "ib_range": 0.0,
            "session_range": 0.0,
            "ib_ext_ratio": 0.0,
            "net_move": 0.0,
            "trend_strength": 0.0,
            "mid_dispersion": 0.0,
            "bimodality": 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────
#  Signal-class helpers (mirror RangeDayClassifier.classify_signal style)
# ─────────────────────────────────────────────────────────────────────────
_MOMENTUM_SOURCES = {"momentum", "breakout", "nr7", "scalp", "fvg"}
_SR_SOURCES = {"sr", "sweep_reclaim", "mean_revert", "fade", "wyckoff"}


def _signal_class(signal_source: Optional[str]) -> str:
    s = (signal_source or "").strip().lower()
    if s in _MOMENTUM_SOURCES:
        return "MOMENTUM"
    if s in _SR_SOURCES:
        return "SR"
    return "MOMENTUM"  # default — most signals are trend-following


# ─────────────────────────────────────────────────────────────────────────
#  Routing helper (consumed by ExpertGate)
# ─────────────────────────────────────────────────────────────────────────
def apply_day_type_routing(
    verdict: Dict[str, Any],
    signal_source: Optional[str],
    direction: Optional[str],
    raw_score: float = 0.0,
    size_mult: float = 1.0,
    *,
    score_boost: float = 1.0,
    score_penalty: float = 1.0,
    normal_sr_boost: float = 1.2,
    normal_momentum_drag: float = 0.8,
    double_dist_drag: float = 0.7,
) -> Dict[str, Any]:
    """Translate a day_type verdict into score/size adjustments per the
    Dalton playbook. NEVER returns reject=True (honours
    [[feedback_no_skip_trades]] — we warn/downsize only). The caller
    composes ``score_delta`` and ``size_mult_tilt`` into whatever scoring
    pipeline it owns.
    """
    out = {
        "score_delta": 0.0,
        "size_mult_tilt": 1.0,
        "reject": False,
        "reason": "neutral",
    }
    try:
        dt = (verdict or {}).get("day_type") or "UNKNOWN"
        if dt == "UNKNOWN":
            return out

        cls = _signal_class(signal_source)
        dirn = (direction or "").upper()

        if dt == "TREND_UP":
            if cls == "MOMENTUM":
                if dirn == "LONG":
                    out["score_delta"] = +abs(score_boost)
                    out["reason"] = "TREND_UP_BOOST_LONG"
                elif dirn == "SHORT":
                    out["score_delta"] = -abs(score_penalty)
                    out["reason"] = "TREND_UP_GATE_SHORT"
        elif dt == "TREND_DOWN":
            if cls == "MOMENTUM":
                if dirn == "SHORT":
                    out["score_delta"] = +abs(score_boost)
                    out["reason"] = "TREND_DOWN_BOOST_SHORT"
                elif dirn == "LONG":
                    out["score_delta"] = -abs(score_penalty)
                    out["reason"] = "TREND_DOWN_GATE_LONG"
        elif dt == "NORMAL":
            if cls == "SR":
                out["size_mult_tilt"] = float(normal_sr_boost)
                out["reason"] = "NORMAL_FAVOR_SR"
            elif cls == "MOMENTUM":
                out["size_mult_tilt"] = float(normal_momentum_drag)
                out["reason"] = "NORMAL_REDUCE_MOMENTUM"
        elif dt == "DOUBLE_DIST":
            out["size_mult_tilt"] = float(double_dist_drag)
            out["reason"] = "DOUBLE_DIST_CHOP_TAX"

        return out
    except Exception as e:  # pragma: no cover — fail-open
        log.debug("apply_day_type_routing failed: %s", e)
        return out


# ─────────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────────
def _self_test() -> int:
    """Minimal smoke tests against synthetic OHLC arrays."""
    fail = 0

    # 1) TREND_UP: steady up-bars with strong extension.
    H = [10.5, 10.8, 11.2, 11.7, 12.3, 12.9, 13.4, 13.9]
    L = [10.0, 10.3, 10.7, 11.2, 11.7, 12.3, 12.8, 13.3]
    O = [10.1, 10.5, 10.8, 11.3, 11.8, 12.4, 12.9, 13.4]
    C = [10.4, 10.7, 11.1, 11.6, 12.2, 12.8, 13.3, 13.8]
    v = classify_day_type(H, L, C, O, atr14=0.5)
    if v["day_type"] != "TREND_UP":
        print("FAIL TREND_UP: got %s (%s)" % (v["day_type"], v))
        fail += 1
    else:
        print("PASS TREND_UP   reason=%s ib_ext=%.2f trend=%.2f" %
              (v["reason"], v["ib_ext_ratio"], v["trend_strength"]))

    # 2) TREND_DOWN: mirror.
    H2 = [13.9, 13.4, 12.9, 12.3, 11.7, 11.2, 10.8, 10.5]
    L2 = [13.3, 12.8, 12.3, 11.7, 11.2, 10.7, 10.3, 10.0]
    O2 = [13.8, 13.3, 12.8, 12.2, 11.6, 11.1, 10.7, 10.4]
    C2 = [13.4, 12.9, 12.4, 11.8, 11.3, 10.8, 10.5, 10.1]
    v2 = classify_day_type(H2, L2, C2, O2, atr14=0.5)
    if v2["day_type"] != "TREND_DOWN":
        print("FAIL TREND_DOWN: got %s (%s)" % (v2["day_type"], v2))
        fail += 1
    else:
        print("PASS TREND_DOWN reason=%s ib_ext=%.2f trend=%.2f" %
              (v2["reason"], v2["ib_ext_ratio"], v2["trend_strength"]))

    # 3) NORMAL: bars contained near IB.
    H3 = [10.5, 10.6, 10.55, 10.62, 10.58, 10.61, 10.6, 10.57]
    L3 = [10.3, 10.32, 10.35, 10.31, 10.34, 10.33, 10.30, 10.32]
    O3 = [10.4, 10.45, 10.42, 10.5, 10.45, 10.5, 10.45, 10.5]
    C3 = [10.45, 10.5, 10.48, 10.46, 10.5, 10.45, 10.5, 10.45]
    v3 = classify_day_type(H3, L3, C3, O3, atr14=0.3)
    if v3["day_type"] != "NORMAL":
        print("FAIL NORMAL: got %s (%s)" % (v3["day_type"], v3))
        fail += 1
    else:
        print("PASS NORMAL     reason=%s ib_ext=%.2f" %
              (v3["reason"], v3["ib_ext_ratio"]))

    # 4) DOUBLE_DIST: bars cluster near 2 separate areas, IB extension is
    #    moderate (>= 1.4) but trend strength low (net move ~ 0).
    H4 = [10.5, 10.6, 10.55, 12.5, 12.6, 12.55, 12.55, 12.5]
    L4 = [10.3, 10.32, 10.35, 12.3, 12.32, 12.35, 12.30, 12.32]
    O4 = [10.4, 10.45, 10.42, 12.4, 12.45, 12.42, 12.45, 12.4]
    C4 = [10.45, 10.5, 10.48, 12.46, 12.5, 12.45, 12.5, 10.45]
    v4 = classify_day_type(H4, L4, C4, O4, atr14=0.3)
    if v4["day_type"] != "DOUBLE_DIST":
        print("FAIL DOUBLE_DIST: got %s (%s)" % (v4["day_type"], v4))
        fail += 1
    else:
        print("PASS DOUBLE_DIST reason=%s bimod=%.2f ib_ext=%.2f" %
              (v4["reason"], v4["bimodality"], v4["ib_ext_ratio"]))

    # 5) UNKNOWN: too few bars.
    v5 = classify_day_type([10.0], [9.9], [9.95], [9.92], atr14=0.1)
    if v5["day_type"] != "UNKNOWN":
        print("FAIL UNKNOWN: got %s" % v5["day_type"])
        fail += 1
    else:
        print("PASS UNKNOWN    reason=%s" % v5["reason"])

    # 6) Routing TREND_UP × momentum LONG → +score boost.
    r = apply_day_type_routing(v, "momentum", "LONG")
    if not (r["score_delta"] > 0 and r["reject"] is False):
        print("FAIL routing TREND_UP/LONG: %s" % r)
        fail += 1
    else:
        print("PASS routing TREND_UP/momentum/LONG  +%.1f reason=%s" %
              (r["score_delta"], r["reason"]))

    # 7) Routing TREND_UP × momentum SHORT → -score penalty.
    r2 = apply_day_type_routing(v, "momentum", "SHORT")
    if not (r2["score_delta"] < 0 and r2["reject"] is False):
        print("FAIL routing TREND_UP/SHORT: %s" % r2)
        fail += 1
    else:
        print("PASS routing TREND_UP/momentum/SHORT %.1f reason=%s" %
              (r2["score_delta"], r2["reason"]))

    # 8) Routing NORMAL × SR → 1.2× size, neutral score.
    r3 = apply_day_type_routing(v3, "sr", "LONG")
    if not (abs(r3["size_mult_tilt"] - 1.2) < 1e-6 and r3["score_delta"] == 0.0):
        print("FAIL routing NORMAL/SR: %s" % r3)
        fail += 1
    else:
        print("PASS routing NORMAL/SR              x%.2f reason=%s" %
              (r3["size_mult_tilt"], r3["reason"]))

    # 9) Routing NORMAL × momentum → 0.8× size.
    r4 = apply_day_type_routing(v3, "momentum", "LONG")
    if not (abs(r4["size_mult_tilt"] - 0.8) < 1e-6):
        print("FAIL routing NORMAL/momentum: %s" % r4)
        fail += 1
    else:
        print("PASS routing NORMAL/momentum        x%.2f reason=%s" %
              (r4["size_mult_tilt"], r4["reason"]))

    # 10) Routing DOUBLE_DIST → 0.7× regardless.
    r5 = apply_day_type_routing(v4, "momentum", "LONG")
    if not (abs(r5["size_mult_tilt"] - 0.7) < 1e-6):
        print("FAIL routing DOUBLE_DIST: %s" % r5)
        fail += 1
    else:
        print("PASS routing DOUBLE_DIST            x%.2f reason=%s" %
              (r5["size_mult_tilt"], r5["reason"]))

    # 11) UNKNOWN routing → neutral.
    r6 = apply_day_type_routing(v5, "momentum", "LONG")
    if not (r6["size_mult_tilt"] == 1.0 and r6["score_delta"] == 0.0):
        print("FAIL routing UNKNOWN: %s" % r6)
        fail += 1
    else:
        print("PASS routing UNKNOWN                neutral")

    # 12) fail-open on bad inputs.
    bad = classify_day_type(None, None, None, None, atr14=float("nan"))  # type: ignore[arg-type]
    if bad.get("day_type") != "UNKNOWN":
        print("FAIL fail-open: %s" % bad)
        fail += 1
    else:
        print("PASS fail-open  reason=%s" % bad["reason"])

    print("")
    if fail:
        print("SELF-TEST FAILED: %d failure(s)" % fail)
        return 1
    print("SELF-TEST PASSED: 12/12")
    return 0


if __name__ == "__main__":
    import sys as _sys

    if "--self-test" in _sys.argv:
        _sys.exit(_self_test())
    print("Use --self-test to run the smoke tests.")
