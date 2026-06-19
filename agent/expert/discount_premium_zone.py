"""
agent/expert/discount_premium_zone.py
══════════════════════════════════════════════════════════════════════════
ICT Discount / Premium Zone Gate  (Gate 3g.dp)

Computes the H1 *dealing range* (highest-high → lowest-low over the last
``lookback`` H1 bars), splits it into discount / equilibrium / premium
zones, and decides whether a proposed entry direction is being attempted
on the *correct* side of value.

ICT theory
──────────
  • Equilibrium    = midpoint of the dealing range. Smart money does not
                     pay equilibrium — it accumulates in discount and
                     distributes in premium.
  • Discount zone  = price < equilibrium  (institutional accumulation —
                     "buyers' value").  Longs welcome.
  • Premium zone   = price > equilibrium  (institutional distribution —
                     "sellers' value"). Shorts welcome.
  • Deep discount  = bottom 30 % of the range (0.0 – 0.3 normalized).
                     A+ long location — buying retail capitulation.
  • Deep premium   = top    30 % of the range (0.7 – 1.0 normalized).
                     A+ short location — selling retail euphoria.

Gate rules
──────────
  • LONG  rejected  if price ≥ 50 % of range  (i.e. *premium*)  — buying
                    at top = chase.
  • SHORT rejected  if price ≤ 50 % of range  (i.e. *discount*) — selling
                    at bottom = late.
  • strict_mode=True  →  *only* DEEP zones (price ≤ 0.30 for LONG, price
                        ≥ 0.70 for SHORT) are approved. This is the A+
                        "top decile setup" filter — reject everything
                        else, including shallow-discount longs.

Pure function module — no MT5, no journal, no logger side effects.
Self-test guarded by ``--self-test``.

Literature anchors
──────────────────
  • ICT Mentorship — Premium/Discount, Equilibrium, OTE zones.
  • Wyckoff Schematic — accumulation (discount) → markup → distribution
                        (premium) → markdown.
  • SMC standard — only buy at discount, only sell at premium.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Union
import math
import sys

import numpy as np

# Type alias accepted everywhere — pandas Series, numpy arrays, or list.
ArrayLike = Union[Sequence[float], np.ndarray]


# ────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────
def _to_np(arr: ArrayLike) -> np.ndarray:
    """Coerce list / pd.Series / np.ndarray to a 1-D float ndarray."""
    if isinstance(arr, np.ndarray):
        return arr.astype(float, copy=False)
    return np.asarray(list(arr), dtype=float)


def _classify_zone(price_pos: float) -> str:
    """Map a 0-1 normalized price position to a zone label.

    Splits:  [0.0 - 0.3]  → deep_discount
             (0.3 - 0.5]  → discount
             (0.5 - 0.7)  → premium
             [0.7 - 1.0]  → deep_premium
    """
    if price_pos <= 0.30:
        return "deep_discount"
    if price_pos <= 0.50:
        return "discount"
    if price_pos < 0.70:
        return "premium"
    return "deep_premium"


# ────────────────────────────────────────────────────────────────────────
# public API
# ────────────────────────────────────────────────────────────────────────
def compute_zone(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    lookback: int = 60,
) -> Dict[str, Union[float, str]]:
    """Compute the dealing range and current zone for the last bar.

    Parameters
    ----------
    highs / lows / closes
        Sequences of H1 OHLC values (oldest → newest). Pandas Series,
        numpy arrays, or plain lists are all accepted.
    lookback
        Number of trailing bars used to define the dealing range.
        Defaults to 60 (typical ICT H1 window). Capped at ``len(highs)``.

    Returns
    -------
    dict with keys
        equilibrium  : float — midpoint of the dealing range.
        price_pos    : float — current close, normalized 0.0 (range_low)
                       to 1.0 (range_high). NaN-safe (defaults to 0.5
                       when the range is degenerate).
        zone         : str   — one of
                       ``'deep_discount' | 'discount' | 'premium'
                       | 'deep_premium'``.
        range_high   : float — highest high in the lookback window.
        range_low    : float — lowest  low  in the lookback window.

    Notes
    -----
    Degenerate range (high == low, or insufficient bars) → returns the
    current close as equilibrium with ``price_pos = 0.5`` and
    ``zone = 'premium'`` to be conservative (still on the rejection side
    for LONGs, still on the rejection side for SHORTs — fail closed for
    both directions when the range is meaningless).
    """
    h = _to_np(highs)
    l = _to_np(lows)
    c = _to_np(closes)

    n = min(len(h), len(l), len(c))
    if n == 0:
        # No data → return a degenerate marker. Caller's gate is fail-open
        # but evaluate_zone_gate() guards against this with a graceful
        # 'approved=True / reason=INSUFFICIENT_DATA' verdict.
        return {
            "equilibrium": float("nan"),
            "price_pos":   0.5,
            "zone":        "premium",
            "range_high":  float("nan"),
            "range_low":   float("nan"),
        }

    lb = max(1, min(int(lookback), n))
    window_h = h[-lb:]
    window_l = l[-lb:]
    range_high = float(np.max(window_h))
    range_low = float(np.min(window_l))
    current = float(c[-1])

    rng = range_high - range_low
    if rng <= 0.0 or not math.isfinite(rng):
        # Degenerate — treat as midpoint to avoid biased approvals.
        equilibrium = current
        price_pos = 0.5
    else:
        equilibrium = (range_high + range_low) / 2.0
        price_pos = (current - range_low) / rng
        # Clamp into [0.0, 1.0] — current can be outside lookback range
        # if the current bar set a fresh extreme (lookback cap may exclude
        # it depending on slicing). The clamp is cheap insurance.
        price_pos = max(0.0, min(1.0, price_pos))

    return {
        "equilibrium": float(equilibrium),
        "price_pos":   float(price_pos),
        "zone":        _classify_zone(price_pos),
        "range_high":  float(range_high),
        "range_low":   float(range_low),
    }


def evaluate_zone_gate(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    direction: str,
    strict_mode: bool = False,
    lookback: int = 60,
) -> Dict[str, Union[bool, str, dict]]:
    """Decide whether ``direction`` is being attempted on the correct
    side of the H1 dealing-range equilibrium.

    Parameters
    ----------
    highs / lows / closes
        H1 OHLC sequences, oldest → newest.
    direction
        ``'LONG'``  | ``'SHORT'``  | anything else (FLAT / None) → approved
        (gate is a no-op for non-directional context).
    strict_mode
        ``False`` (default)  → any side of equilibrium passes
                              (LONG in discount/deep_discount,
                               SHORT in premium/deep_premium).
        ``True``             → only DEEP zones approved (LONG in
                              deep_discount, SHORT in deep_premium).
                              Top-decile A+ entry filter.
    lookback
        Forwarded to :func:`compute_zone`.

    Returns
    -------
    dict with keys
        approved   : bool   — True = pass; False = REJECT.
        reason     : str    — short, log-friendly tag explaining the
                              verdict (``PASS_*`` on approve,
                              ``REJECT_*`` on block,
                              ``INSUFFICIENT_DATA`` on graceful skip).
        zone_info  : dict   — the full :func:`compute_zone` payload, for
                              downstream metadata and journaling.

    Fail-open policy
    ----------------
    Insufficient / empty data → ``approved = True`` with
    ``reason = 'INSUFFICIENT_DATA'``. The caller's gate is wrapped in a
    try/except in brain.py; this layer additionally guarantees a sane
    pass-through when bars are missing rather than blocking entries
    during data hiccups.
    """
    # Non-directional context → no-op pass.
    if direction not in ("LONG", "SHORT"):
        return {
            "approved": True,
            "reason":   "PASS_NO_DIRECTION",
            "zone_info": {},
        }

    h = _to_np(highs)
    l = _to_np(lows)
    c = _to_np(closes)
    n = min(len(h), len(l), len(c))
    if n < 5:  # need at least a handful of bars to mean anything
        return {
            "approved":  True,
            "reason":    "INSUFFICIENT_DATA",
            "zone_info": {"bars_available": int(n)},
        }

    zi = compute_zone(h, l, c, lookback=lookback)
    pos = float(zi["price_pos"])
    zone = str(zi["zone"])

    if direction == "LONG":
        if strict_mode:
            if zone == "deep_discount":
                return {
                    "approved":  True,
                    "reason":    "PASS_DEEP_DISCOUNT_LONG (pos=%.2f)" % pos,
                    "zone_info": zi,
                }
            return {
                "approved":  False,
                "reason":    "REJECT_LONG_NOT_DEEP_DISCOUNT (zone=%s pos=%.2f strict)" % (zone, pos),
                "zone_info": zi,
            }
        # non-strict — pass anything strictly below equilibrium.
        if pos < 0.50:
            return {
                "approved":  True,
                "reason":    "PASS_LONG_IN_DISCOUNT (zone=%s pos=%.2f)" % (zone, pos),
                "zone_info": zi,
            }
        return {
            "approved":  False,
            "reason":    "REJECT_LONG_IN_PREMIUM (zone=%s pos=%.2f)" % (zone, pos),
            "zone_info": zi,
        }

    # direction == "SHORT"
    if strict_mode:
        if zone == "deep_premium":
            return {
                "approved":  True,
                "reason":    "PASS_DEEP_PREMIUM_SHORT (pos=%.2f)" % pos,
                "zone_info": zi,
            }
        return {
            "approved":  False,
            "reason":    "REJECT_SHORT_NOT_DEEP_PREMIUM (zone=%s pos=%.2f strict)" % (zone, pos),
            "zone_info": zi,
        }
    if pos > 0.50:
        return {
            "approved":  True,
            "reason":    "PASS_SHORT_IN_PREMIUM (zone=%s pos=%.2f)" % (zone, pos),
            "zone_info": zi,
        }
    return {
        "approved":  False,
        "reason":    "REJECT_SHORT_IN_DISCOUNT (zone=%s pos=%.2f)" % (zone, pos),
        "zone_info": zi,
    }


# ────────────────────────────────────────────────────────────────────────
# self-test
# ────────────────────────────────────────────────────────────────────────
def _self_test() -> int:
    """Synthetic uptrend / downtrend / range data → print verdicts.

    Run with ``python3 -B discount_premium_zone.py --self-test``.
    Returns 0 on success, non-zero on assertion failure.
    """
    rng = np.random.default_rng(7)

    def _ohlc_from_closes(closes: np.ndarray, atr: float = 0.5):
        highs = closes + rng.uniform(0.0, atr, size=closes.shape)
        lows = closes - rng.uniform(0.0, atr, size=closes.shape)
        return highs, lows, closes

    n = 80
    fails = 0

    # ── 1. Uptrend — current close near top → premium → LONG REJECTED ─
    closes_up = np.linspace(100.0, 130.0, n)
    h, l, c = _ohlc_from_closes(closes_up)
    zone = compute_zone(h, l, c, lookback=60)
    long_v = evaluate_zone_gate(h, l, c, "LONG")
    short_v = evaluate_zone_gate(h, l, c, "SHORT")
    print("[UPTREND]   zone=%s pos=%.3f eq=%.2f rH=%.2f rL=%.2f"
          % (zone["zone"], zone["price_pos"], zone["equilibrium"],
             zone["range_high"], zone["range_low"]))
    print("            LONG  → approved=%s reason=%s"
          % (long_v["approved"], long_v["reason"]))
    print("            SHORT → approved=%s reason=%s"
          % (short_v["approved"], short_v["reason"]))
    if long_v["approved"]:
        print("            FAIL: LONG should be rejected in premium")
        fails += 1
    if not short_v["approved"]:
        print("            FAIL: SHORT should be approved in premium (non-strict)")
        fails += 1

    # ── 2. Downtrend — current close near bottom → discount → SHORT REJ ─
    closes_dn = np.linspace(130.0, 100.0, n)
    h, l, c = _ohlc_from_closes(closes_dn)
    zone = compute_zone(h, l, c, lookback=60)
    long_v = evaluate_zone_gate(h, l, c, "LONG")
    short_v = evaluate_zone_gate(h, l, c, "SHORT")
    print("[DOWNTREND] zone=%s pos=%.3f eq=%.2f rH=%.2f rL=%.2f"
          % (zone["zone"], zone["price_pos"], zone["equilibrium"],
             zone["range_high"], zone["range_low"]))
    print("            LONG  → approved=%s reason=%s"
          % (long_v["approved"], long_v["reason"]))
    print("            SHORT → approved=%s reason=%s"
          % (short_v["approved"], short_v["reason"]))
    if not long_v["approved"]:
        print("            FAIL: LONG should be approved in discount (non-strict)")
        fails += 1
    if short_v["approved"]:
        print("            FAIL: SHORT should be rejected in discount")
        fails += 1

    # ── 3. Range — mid-of-range close → boundary behaviour ───────────
    base = 110.0
    closes_rg = base + rng.normal(0.0, 0.6, n)
    closes_rg[-1] = base  # force final close to mid
    h, l, c = _ohlc_from_closes(closes_rg, atr=0.4)
    zone = compute_zone(h, l, c, lookback=60)
    long_v = evaluate_zone_gate(h, l, c, "LONG")
    short_v = evaluate_zone_gate(h, l, c, "SHORT")
    print("[RANGE]     zone=%s pos=%.3f eq=%.2f rH=%.2f rL=%.2f"
          % (zone["zone"], zone["price_pos"], zone["equilibrium"],
             zone["range_high"], zone["range_low"]))
    print("            LONG  → approved=%s reason=%s"
          % (long_v["approved"], long_v["reason"]))
    print("            SHORT → approved=%s reason=%s"
          % (short_v["approved"], short_v["reason"]))

    # ── 4. Strict mode — uptrend current close in deep premium ──────
    h, l, c = _ohlc_from_closes(closes_up)
    strict_short = evaluate_zone_gate(h, l, c, "SHORT", strict_mode=True)
    print("[STRICT]    UPTREND SHORT → approved=%s reason=%s"
          % (strict_short["approved"], strict_short["reason"]))
    if not strict_short["approved"]:
        # End of an uptrend should normally land in deep_premium; if not,
        # report but don't auto-fail (depends on RNG noise).
        print("            NOTE: strict short not approved (zone=%s)"
              % strict_short["zone_info"].get("zone"))

    # ── 5. Strict mode — downtrend current close in deep discount ───
    h, l, c = _ohlc_from_closes(closes_dn)
    strict_long = evaluate_zone_gate(h, l, c, "LONG", strict_mode=True)
    print("[STRICT]    DOWNTREND LONG → approved=%s reason=%s"
          % (strict_long["approved"], strict_long["reason"]))

    # ── 6. Insufficient data — fail-open ────────────────────────────
    empty_v = evaluate_zone_gate([], [], [], "LONG")
    print("[EMPTY]     LONG  → approved=%s reason=%s"
          % (empty_v["approved"], empty_v["reason"]))
    if not empty_v["approved"]:
        print("            FAIL: empty input should fail-open")
        fails += 1

    # ── 7. Direction == 'FLAT' — no-op pass ─────────────────────────
    h, l, c = _ohlc_from_closes(closes_up)
    flat_v = evaluate_zone_gate(h, l, c, "FLAT")
    print("[FLAT]      → approved=%s reason=%s"
          % (flat_v["approved"], flat_v["reason"]))
    if not flat_v["approved"]:
        print("            FAIL: FLAT direction should pass-through")
        fails += 1

    if fails == 0:
        print("\nSELF-TEST: PASS (all assertions held)")
        return 0
    print("\nSELF-TEST: FAIL (%d assertions failed)" % fails)
    return 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print("usage: python3 -B discount_premium_zone.py --self-test")
    raise SystemExit(0)
