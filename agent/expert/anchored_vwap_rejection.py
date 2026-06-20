"""
agent/expert/anchored_vwap_rejection.py
══════════════════════════════════════════════════════════════════════════
Anchored VWAP Rejection — momentum score booster (+1)

Theory
──────
Anchored VWAP from today's session open (or N bars back) is a widely-watched
institutional fair-value benchmark. When price probes the AVWAP, takes liquidity
across it intrabar, then closes back on the side of the prevailing trend
WITHOUT a full breach, that's a high-quality confirmation of institutional
respect — "value defended". Brian Shannon (Anchored VWAPs, 2022) treats these
rejection prints as the cleanest re-entry signal for trend continuation.

This module DETECTS that rejection and exposes it as a `+1` booster that
brain.py adds to the raw momentum score BEFORE the quality-threshold check,
*if* the booster flag is enabled AND a fresh rejection (≤ lookback bars old)
in the candidate trade direction is found.

Design intent
─────────────
  • Detector is decoupled — pure-function, no I/O, no side effects.
  • Direction-aware:
      LONG  rejection = wick PIERCED VWAP from above (low ≤ vwap)
                        AND close ≥ vwap (defended)
                        AND no full body close BELOW vwap in lookback.
      SHORT rejection = symmetric (high ≥ vwap, close ≤ vwap, no body
                        close above vwap).
  • Freshness gated — only fires if rejection happened within the last
    `lookback` bars (default 5). Stale rejections add no edge.
  • Fail-OPEN — bad/short data returns (False, None) so the brain just
    skips the booster, never blocks the trade on a data hiccup.

Anchoring
─────────
Two modes supported:
  1. Session anchor (default in compute_anchored_vwap when `session_open_idx`
     is provided) — cumulative since UTC 00:00 of the latest bar's day.
     Aligns with Brian Shannon's "Anchored at session open" methodology.
  2. Rolling N-bar anchor — cumulative over the last `anchor_bars` (default
     24, which is one D1 worth of H1 bars or roughly a 6-hour M15 window
     depending on input TF). Used as a fallback / drop-in for symbols
     where the brain has no session timestamp at hand.

Both modes compute  Σ(typical_price · volume) / Σ(volume)  from the anchor
point forward, where typical_price = (high + low + close) / 3.

Public API
──────────
  • compute_anchored_vwap(highs, lows, closes, volumes,
                          anchor_bars=24, session_open_idx=None) -> np.ndarray
  • detect_vwap_rejection(highs, lows, closes, vwap_series, direction,
                          lookback=5) -> (bool, freshness_bars | None)
  • evaluate_avwap_booster(highs, lows, closes, volumes, direction, *,
                           anchor_bars=24, lookback=5,
                           session_open_idx=None) -> dict
    Convenience: bundles compute + detect + returns the score-bump amount
    + freshness for brain integration.

Config flags (defined in config.py, defensively imported by brain.py)
────────────────────────────────────────────────────────────────────
  • ANCHORED_VWAP_BOOSTER_ENABLED  default False
  • ANCHOR_BARS_DEFAULT             default 24
  • ANCHORED_VWAP_LOOKBACK_BARS     default 5
  • ANCHORED_VWAP_BOOST_AMOUNT      default 1.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Module defaults — brain.py overrides via config flags when calling
# ─────────────────────────────────────────────────────────────────────────
ANCHOR_BARS_DEFAULT       = 24      # one D1 (24 H1 bars) — Brian Shannon default
LOOKBACK_BARS_DEFAULT     = 5       # rejection freshness window
BOOST_AMOUNT_DEFAULT      = 1.0     # +1 to raw_score when rejection fresh

VERDICT_BOOST       = "BOOST"
VERDICT_NO_BOOST    = "NO_BOOST"
VERDICT_SKIP_NODATA = "SKIP_NO_DATA"


# ─────────────────────────────────────────────────────────────────────────
# compute_anchored_vwap — cumulative typical·volume / cumulative volume
# ─────────────────────────────────────────────────────────────────────────
def compute_anchored_vwap(highs: Any,
                          lows: Any,
                          closes: Any,
                          volumes: Any,
                          anchor_bars: int = ANCHOR_BARS_DEFAULT,
                          session_open_idx: Optional[int] = None
                          ) -> np.ndarray:
    """Return per-bar AVWAP series, NaN before the anchor index.

    Parameters
    ──────────
      highs, lows, closes, volumes : array-like (numpy / list / pd.Series)
      anchor_bars                  : when `session_open_idx` is None, anchor
                                     N bars before the LAST bar (rolling).
      session_open_idx             : absolute index into the arrays where the
                                     anchor begins (e.g., session open bar).
                                     When provided, takes precedence over
                                     `anchor_bars`.

    Returns
    ───────
      np.ndarray same length as inputs. NaN for bars before the anchor.
      All-NaN array on data error (fail-open contract — callers detect
      via np.isnan() and skip the booster).
    """
    try:
        h = np.asarray(highs,   dtype=np.float64)
        l = np.asarray(lows,    dtype=np.float64)
        c = np.asarray(closes,  dtype=np.float64)
        v = np.asarray(volumes, dtype=np.float64)
    except Exception:
        return np.full(0, np.nan, dtype=np.float64)

    n = len(c)
    if n == 0 or len(h) != n or len(l) != n or len(v) != n:
        return np.full(max(n, 0), np.nan, dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)

    # Resolve anchor index
    if session_open_idx is not None:
        try:
            ai = int(session_open_idx)
        except Exception:
            ai = max(0, n - int(anchor_bars))
    else:
        try:
            ab = int(anchor_bars)
        except Exception:
            ab = ANCHOR_BARS_DEFAULT
        if ab < 1:
            ab = ANCHOR_BARS_DEFAULT
        ai = max(0, n - ab)

    if ai < 0:
        ai = 0
    if ai >= n:
        return out  # anchor past end → all NaN

    typical = (h + l + c) / 3.0
    cum_pv = 0.0
    cum_v  = 0.0
    for i in range(ai, n):
        vi = v[i]
        # Sanitize: treat NaN / negative volume as zero contribution
        if not (vi == vi) or vi < 0.0:
            vi = 0.0
        cum_pv += typical[i] * vi
        cum_v  += vi
        if cum_v > 0.0:
            out[i] = cum_pv / cum_v
        # else: leave NaN — no volume yet

    return out


# ─────────────────────────────────────────────────────────────────────────
# detect_vwap_rejection — LONG / SHORT wick-rejection with freshness
# ─────────────────────────────────────────────────────────────────────────
def detect_vwap_rejection(highs: Any,
                          lows: Any,
                          closes: Any,
                          vwap_series: Any,
                          direction: Optional[str],
                          lookback: int = LOOKBACK_BARS_DEFAULT
                          ) -> Tuple[bool, Optional[int]]:
    """Find the most recent VWAP rejection in trade direction.

    LONG rejection criteria (any of the last `lookback` bars):
      • low[i]   ≤ vwap[i]                (price touched/broke from above)
      • close[i] >  vwap[i]               (closed back above — defended)
      • NO bar in [i, latest] has  close < vwap  AND  open < vwap
        (no full-body trend reversal since the rejection wick)

    SHORT rejection criteria (symmetric):
      • high[i]  ≥ vwap[i]
      • close[i] <  vwap[i]
      • NO bar in [i, latest] has  close > vwap  AND  open > vwap

    Returns
    ───────
      (True, freshness_bars)  — rejection found, `freshness_bars` = number
                                of completed bars since the rejection
                                (0 = the latest closed bar).
      (False, None)           — no rejection found or data unusable.
    """
    if direction not in ("LONG", "SHORT", "long", "short"):
        return (False, None)
    dir_up = direction.upper() == "LONG"

    try:
        h = np.asarray(highs,  dtype=np.float64)
        l = np.asarray(lows,   dtype=np.float64)
        c = np.asarray(closes, dtype=np.float64)
        w = np.asarray(vwap_series, dtype=np.float64)
    except Exception:
        return (False, None)

    n = len(c)
    if n == 0 or len(h) != n or len(l) != n or len(w) != n:
        return (False, None)

    try:
        lb = int(lookback)
    except Exception:
        lb = LOOKBACK_BARS_DEFAULT
    if lb < 1:
        lb = 1
    if lb > n:
        lb = n

    # Walk from MOST RECENT backwards — first match wins (freshest)
    # We need an open-series for the trend-reversal guard; reconstruct
    # from close[i-1] when not provided. (Caller passes OHLC arrays in
    # most contexts — we approximate opens as previous closes here so
    # this function can also work with close-only inputs.)
    # Since brain calls with full OHLC, we use highs/lows endpoints as
    # surrogate for body when needed:
    #   body_below_vwap  = (close < vwap) AND (high  < vwap)  (full bar below)
    #   body_above_vwap  = (close > vwap) AND (low   > vwap)  (full bar above)
    for back in range(0, lb):
        i = n - 1 - back
        if i < 0:
            break
        wi = w[i]
        if not (wi == wi):  # NaN guard
            continue
        ci, hi, li = c[i], h[i], l[i]

        if dir_up:
            touched = li <= wi
            defended = ci > wi
            if not (touched and defended):
                continue
            # No full-body bear bar BELOW vwap between rejection bar and now
            reversed_ = False
            for j in range(i, n):
                wj = w[j]
                if not (wj == wj):
                    continue
                if c[j] < wj and h[j] < wj:
                    reversed_ = True
                    break
            if reversed_:
                continue
            return (True, back)
        else:
            touched = hi >= wi
            defended = ci < wi
            if not (touched and defended):
                continue
            reversed_ = False
            for j in range(i, n):
                wj = w[j]
                if not (wj == wj):
                    continue
                if c[j] > wj and l[j] > wj:
                    reversed_ = True
                    break
            if reversed_:
                continue
            return (True, back)

    return (False, None)


# ─────────────────────────────────────────────────────────────────────────
# evaluate_avwap_booster — convenience for brain integration
# ─────────────────────────────────────────────────────────────────────────
def evaluate_avwap_booster(highs: Any,
                           lows: Any,
                           closes: Any,
                           volumes: Any,
                           direction: Optional[str],
                           *,
                           anchor_bars: int = ANCHOR_BARS_DEFAULT,
                           lookback: int = LOOKBACK_BARS_DEFAULT,
                           boost_amount: float = BOOST_AMOUNT_DEFAULT,
                           session_open_idx: Optional[int] = None
                           ) -> Dict[str, Any]:
    """Compute AVWAP, detect rejection, return verdict dict.

    Always returns a dict (never raises). Shape::

        {
            "verdict":    "BOOST" | "NO_BOOST" | "SKIP_NO_DATA",
            "boost":      float,           # 0.0 unless BOOST
            "freshness":  int | None,      # bars since rejection
            "vwap_now":   float | None,    # latest AVWAP value
            "reason":     str,
        }
    """
    out: Dict[str, Any] = {
        "verdict":   VERDICT_SKIP_NODATA,
        "boost":     0.0,
        "freshness": None,
        "vwap_now":  None,
        "reason":    "",
    }

    # Fail-OPEN on any None / un-arrayable input
    if highs is None or lows is None or closes is None or volumes is None:
        out["reason"] = "AVWAP_BAD_INPUT"
        return out
    try:
        h = np.asarray(highs,   dtype=np.float64)
        l = np.asarray(lows,    dtype=np.float64)
        c = np.asarray(closes,  dtype=np.float64)
        v = np.asarray(volumes, dtype=np.float64)
    except Exception:
        out["reason"] = "AVWAP_BAD_INPUT"
        return out

    try:
        n = len(c)
    except TypeError:
        out["reason"] = "AVWAP_BAD_INPUT"
        return out

    # Need enough bars to have an anchor + at least 2 bars for a wick test
    try:
        min_bars = max(int(anchor_bars), int(lookback)) + 2
    except Exception:
        min_bars = ANCHOR_BARS_DEFAULT + 2
    if n < min_bars or len(h) != n or len(l) != n or len(v) != n:
        out["reason"] = "AVWAP_SHORT_FRAME"
        return out

    if direction not in ("LONG", "SHORT", "long", "short"):
        out["reason"] = "AVWAP_BAD_DIRECTION"
        return out

    try:
        vwap = compute_anchored_vwap(
            h, l, c, v,
            anchor_bars=anchor_bars,
            session_open_idx=session_open_idx,
        )
    except Exception:
        out["reason"] = "AVWAP_COMPUTE_FAIL"
        return out

    if vwap is None or len(vwap) != n:
        out["reason"] = "AVWAP_COMPUTE_FAIL"
        return out

    last_w = float(vwap[-1]) if (vwap[-1] == vwap[-1]) else None
    out["vwap_now"] = last_w
    if last_w is None:
        out["reason"] = "AVWAP_NAN"
        return out

    try:
        found, freshness = detect_vwap_rejection(
            h, l, c, vwap, direction, lookback=lookback,
        )
    except Exception:
        out["reason"] = "AVWAP_DETECT_FAIL"
        return out

    if found:
        try:
            out["boost"] = float(boost_amount)
        except Exception:
            out["boost"] = BOOST_AMOUNT_DEFAULT
        out["verdict"] = VERDICT_BOOST
        out["freshness"] = int(freshness) if freshness is not None else 0
        out["reason"] = "AVWAP_REJECTION"
        return out

    out["verdict"] = VERDICT_NO_BOOST
    out["reason"] = "AVWAP_NO_REJECTION"
    return out


__all__ = [
    "ANCHOR_BARS_DEFAULT",
    "LOOKBACK_BARS_DEFAULT",
    "BOOST_AMOUNT_DEFAULT",
    "VERDICT_BOOST",
    "VERDICT_NO_BOOST",
    "VERDICT_SKIP_NODATA",
    "compute_anchored_vwap",
    "detect_vwap_rejection",
    "evaluate_avwap_booster",
]


# ═════════════════════════════════════════════════════════════════════════
# Self-test  —  exit 0 on pass, 1 on fail. Run via:
#     python3 -B agent/expert/anchored_vwap_rejection.py --self-test
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    if "--self-test" not in sys.argv:
        print("usage: python3 anchored_vwap_rejection.py --self-test")
        sys.exit(0)

    failures = []

    # ── compute_anchored_vwap basic shape & math ────────────────────────
    # Flat price 100, vol 1 → AVWAP should equal 100 from anchor onward.
    h = [101.0] * 30
    l = [99.0]  * 30
    c = [100.0] * 30
    v = [1.0]   * 30
    w = compute_anchored_vwap(h, l, c, v, anchor_bars=10)
    if not (np.isnan(w[0]) and abs(w[-1] - 100.0) < 1e-9):
        failures.append(f"flat AVWAP wrong: w[0]={w[0]}, w[-1]={w[-1]}")

    # Rising price → AVWAP rises but lags price (cumulative average effect)
    closes_up = [100.0 + i * 0.5 for i in range(30)]
    highs_up  = [x + 0.5 for x in closes_up]
    lows_up   = [x - 0.5 for x in closes_up]
    w2 = compute_anchored_vwap(highs_up, lows_up, closes_up, v, anchor_bars=20)
    if not (w2[-1] < closes_up[-1] and w2[-1] > closes_up[-20]):
        failures.append(f"rising AVWAP not in band: vwap={w2[-1]}, "
                        f"close={closes_up[-1]}, anchor_close={closes_up[-20]}")

    # Empty arrays → empty array
    w3 = compute_anchored_vwap([], [], [], [], anchor_bars=10)
    if len(w3) != 0:
        failures.append(f"empty input should return empty, got len={len(w3)}")

    # Mismatched lengths → all-NaN
    w4 = compute_anchored_vwap([1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3])
    if not (len(w4) == 3 and all(np.isnan(w4))):
        failures.append(f"mismatched lengths should be all-NaN, got {w4}")

    # session_open_idx overrides anchor_bars
    w5 = compute_anchored_vwap(h, l, c, v, anchor_bars=5, session_open_idx=0)
    if np.isnan(w5[0]):
        failures.append("session_open_idx=0 should anchor at bar 0")

    # ── detect_vwap_rejection — LONG case ───────────────────────────────
    # Build a scenario: price trending up, latest bar wicks below VWAP, closes
    # back above. Use a long uptrend so VWAP < current close.
    n = 30
    base_c = np.array([100.0 + i * 0.3 for i in range(n)], dtype=np.float64)
    base_h = base_c + 0.2
    base_l = base_c - 0.2
    base_v = np.ones(n, dtype=np.float64)
    vw = compute_anchored_vwap(base_h, base_l, base_c, base_v, anchor_bars=20)
    # On the latest bar, push the LOW down through VWAP but keep close above.
    base_l[-1] = vw[-1] - 0.5
    found, fresh = detect_vwap_rejection(base_h, base_l, base_c, vw, "LONG", lookback=5)
    if not (found and fresh == 0):
        failures.append(f"LONG rejection on latest bar should fire, got found={found} fresh={fresh}")

    # SHORT case — downtrend, wick above VWAP, close below.
    base_c_d = np.array([100.0 - i * 0.3 for i in range(n)], dtype=np.float64)
    base_h_d = base_c_d + 0.2
    base_l_d = base_c_d - 0.2
    vw_d = compute_anchored_vwap(base_h_d, base_l_d, base_c_d, base_v, anchor_bars=20)
    base_h_d[-1] = vw_d[-1] + 0.5
    found, fresh = detect_vwap_rejection(base_h_d, base_l_d, base_c_d, vw_d, "SHORT", lookback=5)
    if not (found and fresh == 0):
        failures.append(f"SHORT rejection should fire, got found={found} fresh={fresh}")

    # No rejection — flat price never touches VWAP wick-wise (high/low straddle but close stays above)
    flat_c = np.array([100.0] * 30, dtype=np.float64)
    flat_h = flat_c + 0.05
    flat_l = flat_c - 0.05
    vw_flat = compute_anchored_vwap(flat_h, flat_l, flat_c, base_v, anchor_bars=20)
    # Here low = 99.95, vwap = 100.0, so low < vwap (touched) AND close == vwap.
    # close > vwap is FALSE so should NOT fire as LONG rejection.
    found, fresh = detect_vwap_rejection(flat_h, flat_l, flat_c, vw_flat, "LONG", lookback=5)
    if found:
        failures.append(f"flat 'close == vwap' should NOT be rejection, got found={found}")

    # Trend-reversal guard: stale LONG rejection invalidated by later bear close below vwap.
    # Construct: rejection at i = n-5 (touched then closed back), then i = n-2
    # closes BELOW vwap (full-body bear). Expect found = False.
    stale_c = np.array([100.0 + i * 0.3 for i in range(n)], dtype=np.float64)
    stale_h = stale_c + 0.2
    stale_l = stale_c - 0.2
    stale_v = np.ones(n, dtype=np.float64)
    vw_s = compute_anchored_vwap(stale_h, stale_l, stale_c, stale_v, anchor_bars=20)
    # Plant rejection wick at i = n-5
    stale_l[n - 5] = vw_s[n - 5] - 0.5
    # Plant full-body bear at i = n-2 (close + high both below vwap)
    stale_c[n - 2] = vw_s[n - 2] - 1.5
    stale_h[n - 2] = vw_s[n - 2] - 0.5
    found, fresh = detect_vwap_rejection(stale_h, stale_l, stale_c, vw_s, "LONG", lookback=10)
    if found:
        failures.append(f"stale LONG rejection (later body close < vwap) should INVALIDATE, "
                        f"got found={found} fresh={fresh}")

    # Bad direction → (False, None) — no booster on garbage input
    found, fresh = detect_vwap_rejection(stale_h, stale_l, stale_c, vw_s, "WEIRD", lookback=5)
    if found or fresh is not None:
        failures.append(f"bad direction should return (False, None), got ({found}, {fresh})")

    # ── evaluate_avwap_booster end-to-end ───────────────────────────────
    # LONG rejection — should BOOST
    boost_c = np.array([100.0 + i * 0.3 for i in range(40)], dtype=np.float64)
    boost_h = boost_c + 0.2
    boost_l = boost_c - 0.2
    boost_v = np.ones(40, dtype=np.float64)
    # Rejection on the latest bar
    _vw = compute_anchored_vwap(boost_h, boost_l, boost_c, boost_v, anchor_bars=24)
    boost_l[-1] = _vw[-1] - 0.5
    res = evaluate_avwap_booster(boost_h, boost_l, boost_c, boost_v, "LONG",
                                 anchor_bars=24, lookback=5)
    if res["verdict"] != VERDICT_BOOST or res["boost"] != 1.0:
        failures.append(f"end-to-end LONG should BOOST, got {res}")

    # No rejection — flat → NO_BOOST verdict (data is OK)
    plain_c = np.array([100.0] * 40, dtype=np.float64)
    plain_h = plain_c + 0.1
    plain_l = plain_c + 0.05      # low stays ABOVE 100 → never touches AVWAP
    plain_v = np.ones(40, dtype=np.float64)
    res = evaluate_avwap_booster(plain_h, plain_l, plain_c, plain_v, "LONG")
    if res["verdict"] != VERDICT_NO_BOOST or res["boost"] != 0.0:
        failures.append(f"no-rejection should NO_BOOST, got {res}")

    # Short frame → SKIP_NO_DATA
    res = evaluate_avwap_booster([1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 1, 1], "LONG")
    if res["verdict"] != VERDICT_SKIP_NODATA:
        failures.append(f"short frame should SKIP, got {res}")

    # None input → SKIP_NO_DATA (fail-open)
    try:
        res = evaluate_avwap_booster(None, None, None, None, "LONG")
        if res["verdict"] != VERDICT_SKIP_NODATA:
            failures.append(f"None inputs should SKIP, got {res}")
    except Exception as e:
        failures.append(f"None inputs raised: {e!r}")

    # Bad direction → SKIP_NO_DATA
    res = evaluate_avwap_booster(boost_h, boost_l, boost_c, boost_v, "BAD")
    if res["verdict"] != VERDICT_SKIP_NODATA:
        failures.append(f"bad direction should SKIP, got {res}")

    # Custom boost_amount honored
    res = evaluate_avwap_booster(boost_h, boost_l, boost_c, boost_v, "LONG",
                                 boost_amount=2.5)
    if res["verdict"] != VERDICT_BOOST or res["boost"] != 2.5:
        failures.append(f"custom boost_amount=2.5 not honored, got {res}")

    if failures:
        print("anchored_vwap_rejection SELF-TEST FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)

    print("anchored_vwap_rejection self-test OK (15 cases)")
    sys.exit(0)
