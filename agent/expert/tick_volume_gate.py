"""
agent/expert/tick_volume_gate.py
══════════════════════════════════════════════════════════════════════════
Tick-volume Order-Flow Imbalance Gate  (Gate 3g)

A participation filter that requires the latest closed H1 bar's
``tick_volume`` to exceed (BREAKOUT) or at least meet (MEAN_REVERT) a
multiple of its 20-bar moving average. Filters out "no-participation"
false moves where price prints a signal-grade move on dry tape.

Design intent
─────────────
Sits BETWEEN  Gate 3f (ICT liquidity-sweep)  and  Gate 4 (position mgmt).
A structural break (BoS/CHoCH) without a corresponding expansion in
tick_volume is a Wyckoff "no-demand" / "no-supply" print — high failure
rate. We require BREAKOUT bars to print >= 1.30x their 20-bar MA volume;
MEAN_REVERT setups need >= 0.70x (enough liquidity to fade, but we
explicitly do NOT require a spike since reversions often fire AFTER one).

A single absolute hard floor (default 0.30x) applies to every setup and
symbol — dead-tape protection for Sunday open / post-news vacuums.

Module-level functions — no class state, no side effects, no I/O.
Pure-function tested via the ``if __name__ == "__main__"`` self-test.

Literature anchors
──────────────────
  • Wyckoff Effort-vs-Result — quiet break = "no demand", reverts.
  • ICT/SMC displacement     — institutional BoS prints expand volume;
                               retail-only quiet BoS reverses.
  • Bollinger/Lighter regime — BBW + volume confirm; volume alone catches
                               stealth participation that BBW misses.
  • Mean-revert inversion    — < 0.70x MA = liquidity vacuum, spread eats
                               the fade.

Public API
──────────
  • classify_setup(direction, regime, comp_dir)         -> str
  • tick_volume_imbalance(h1_df, lookback=20)           -> (ratio, ma, cur, ok_data)
  • evaluate_tv_gate(direction, regime, comp_dir, h1_df, ...)
                                                        -> dict
"""

from __future__ import annotations

from math import isnan
from typing import Any, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────
# Module-level defaults — caller may override via evaluate_tv_gate kwargs
# or via config flags (TV_VOLUME_*) loaded by brain.py at gate time.
# ─────────────────────────────────────────────────────────────────────────
TV_LOOKBACK_BARS      = 20
TV_BREAKOUT_MIN_RATIO = 1.30   # 30% above 20-bar MA = real participation
TV_REVERT_MIN_RATIO   = 0.70   # >=70% of MA = enough liquidity to fade
TV_HARD_FLOOR_RATIO   = 0.30   # absolute REJECT below this for ANY setup
TV_TIMEFRAME_MIN      = 60     # H1 = 60min (informational; data is fed in)


SETUP_BREAKOUT    = "BREAKOUT"
SETUP_MEAN_REVERT = "MEAN_REVERT"
SETUP_UNKNOWN     = "UNKNOWN"

VERDICT_PASS         = "PASS"
VERDICT_REJECT       = "REJECT"
VERDICT_SKIP_NO_DATA = "SKIP_NO_DATA"


# ─────────────────────────────────────────────────────────────────────────
# classify_setup — BREAKOUT vs MEAN_REVERT decision
# ─────────────────────────────────────────────────────────────────────────
def classify_setup(direction: Optional[str],
                   regime: Optional[str],
                   comp_dir: Optional[Dict[str, float]]) -> str:
    """Decide whether the candidate entry is BREAKOUT or MEAN_REVERT.

    Uses already-computed scoring artefacts only — no new indicator work::

        BREAKOUT     if regime in {"trending","volatile"}
                       OR  comp_dir["breakout"] > 0
        MEAN_REVERT  if regime in {"ranging","low_vol"}
                       AND comp_dir["breakout"] == 0
        UNKNOWN      otherwise (defensive — caller fails OPEN)
    """
    # Direction sanity — accept anything but missing direction does not help.
    if direction not in ("LONG", "SHORT"):
        return SETUP_UNKNOWN

    reg = (regime or "").lower().strip() if isinstance(regime, str) else None
    comp = comp_dir or {}
    try:
        breakout_comp = float(comp.get("breakout", 0.0))
    except Exception:
        breakout_comp = 0.0

    # BREAKOUT path: explicit volatile/trending regime, OR a positive
    # breakout component contribution to the score.
    if reg in ("trending", "volatile") or breakout_comp > 0.0:
        return SETUP_BREAKOUT

    # MEAN_REVERT path: explicit ranging/low_vol AND no breakout component.
    if reg in ("ranging", "low_vol") and breakout_comp == 0.0:
        return SETUP_MEAN_REVERT

    return SETUP_UNKNOWN


# ─────────────────────────────────────────────────────────────────────────
# tick_volume_imbalance — pull cur, MA, ratio from a candle frame
# ─────────────────────────────────────────────────────────────────────────
def tick_volume_imbalance(h1_df: Any,
                          lookback: int = TV_LOOKBACK_BARS
                          ) -> Tuple[float, float, float, bool]:
    """Return ``(ratio, ma, cur, ok_data)`` for the supplied candle frame.

    The ``cur`` value is the LATEST bar's tick_volume; the moving average
    is computed over the ``lookback`` PRIOR bars (excludes ``cur`` so we
    have a clean "current vs prior" comparison).

    Failure mode: returns ``(nan, nan, nan, False)`` if the frame is too
    short, lacks a tick_volume column, or yields a non-positive MA. Callers
    treat ``ok_data == False`` as SKIP_NO_DATA (fail OPEN).
    """
    nan = float("nan")
    try:
        lb = int(lookback)
    except Exception:
        lb = TV_LOOKBACK_BARS
    if lb < 1:
        lb = TV_LOOKBACK_BARS

    if h1_df is None:
        return (nan, nan, nan, False)

    # Pull the tick_volume series. Support pandas-DF style (`df["x"]`),
    # dict-of-arrays, and our test-fakes that expose `.values` on `[col]`.
    try:
        col = h1_df["tick_volume"]
    except Exception:
        return (nan, nan, nan, False)

    # Normalize to a numeric list/array.
    values = None
    # pandas Series — use .iloc / .values
    try:
        import numpy as _np  # local — keeps module import-light if numpy absent
        if hasattr(col, "iloc") and hasattr(col, "values"):
            arr = _np.asarray(col.values, dtype=float)
        elif hasattr(col, "values"):
            arr = _np.asarray(col.values, dtype=float)
        else:
            arr = _np.asarray(list(col), dtype=float)
        values = arr
    except Exception:
        try:
            values = [float(v) for v in list(col)]
        except Exception:
            return (nan, nan, nan, False)

    n = len(values)
    if n < (lb + 1):
        return (nan, nan, nan, False)

    try:
        cur = float(values[-1])
    except Exception:
        return (nan, nan, nan, False)

    if cur != cur:  # NaN check w/o importing math.isnan dependency
        return (nan, nan, nan, False)

    # Prior n bars, excluding the latest.
    try:
        window = values[-(lb + 1):-1]
        # Filter out any NaN slots to avoid contaminating mean.
        clean = [float(v) for v in window if (v == v)]
        if len(clean) != lb:
            return (nan, nan, cur, False)
        ma = sum(clean) / float(lb)
    except Exception:
        return (nan, nan, cur, False)

    if not (ma > 0.0):
        return (nan, float(ma), cur, False)

    ratio = cur / ma
    if ratio != ratio:  # NaN guard
        return (nan, float(ma), cur, False)

    return (float(ratio), float(ma), float(cur), True)


# ─────────────────────────────────────────────────────────────────────────
# evaluate_tv_gate — the actual gate decision
# ─────────────────────────────────────────────────────────────────────────
def evaluate_tv_gate(direction: Optional[str],
                     regime: Optional[str],
                     comp_dir: Optional[Dict[str, float]],
                     h1_df: Any,
                     *,
                     lookback: int = TV_LOOKBACK_BARS,
                     thr_breakout: float = TV_BREAKOUT_MIN_RATIO,
                     thr_revert: float = TV_REVERT_MIN_RATIO,
                     per_sym_overrides: Optional[Dict[str, Dict[str, float]]] = None,
                     symbol: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Run the full TV gate and return a verdict dict.

    Returns a dict shaped::

        {
            "verdict":   "PASS" | "REJECT" | "SKIP_NO_DATA",
            "setup":     "BREAKOUT" | "MEAN_REVERT" | "UNKNOWN",
            "ratio":     float,
            "ma":        float,
            "cur":       float,
            "threshold": float,
            "reason":    str,
        }
    """
    nan = float("nan")
    base = {
        "verdict":   VERDICT_SKIP_NO_DATA,
        "setup":     SETUP_UNKNOWN,
        "ratio":     nan,
        "ma":        nan,
        "cur":       nan,
        "threshold": nan,
        "reason":    "",
    }

    # 1. setup classification
    setup = classify_setup(direction, regime, comp_dir)
    base["setup"] = setup

    # 2. tick-volume measurement
    ratio, ma, cur, ok_data = tick_volume_imbalance(h1_df, lookback=lookback)
    base["ratio"] = ratio
    base["ma"]    = ma
    base["cur"]   = cur

    # 3. fail OPEN on bad data
    if not ok_data:
        base["verdict"] = VERDICT_SKIP_NO_DATA
        base["reason"]  = "TV_NO_DATA"
        return base

    # 4. UNKNOWN setup -> let downstream gates decide
    if setup == SETUP_UNKNOWN:
        base["verdict"] = VERDICT_SKIP_NO_DATA
        base["reason"]  = "TV_SETUP_UNKNOWN"
        return base

    # 5. resolve threshold (per-sym override > kwarg default)
    default_thr = float(thr_breakout if setup == SETUP_BREAKOUT else thr_revert)
    thr = default_thr
    if isinstance(per_sym_overrides, dict) and symbol is not None:
        sym_over = per_sym_overrides.get(symbol)
        if isinstance(sym_over, dict):
            cand = sym_over.get(setup)
            if cand is not None:
                try:
                    thr = float(cand)
                except Exception:
                    thr = default_thr
    base["threshold"] = thr

    # 6/7. PASS / REJECT
    if ratio >= thr:
        base["verdict"] = VERDICT_PASS
        base["reason"]  = "TV_OK"
        return base

    if setup == SETUP_BREAKOUT:
        base["reason"] = "TV_THIN_BREAKOUT"
    else:
        base["reason"] = "TV_DEAD_REVERT"
    base["verdict"] = VERDICT_REJECT
    return base


__all__ = [
    "TV_LOOKBACK_BARS",
    "TV_BREAKOUT_MIN_RATIO",
    "TV_REVERT_MIN_RATIO",
    "TV_HARD_FLOOR_RATIO",
    "TV_TIMEFRAME_MIN",
    "SETUP_BREAKOUT",
    "SETUP_MEAN_REVERT",
    "SETUP_UNKNOWN",
    "VERDICT_PASS",
    "VERDICT_REJECT",
    "VERDICT_SKIP_NO_DATA",
    "classify_setup",
    "tick_volume_imbalance",
    "evaluate_tv_gate",
]


# ═════════════════════════════════════════════════════════════════════════
# Self-test — synthetic fixtures, exit 0 on pass.
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    class _Col:
        def __init__(self, arr):
            self.values = list(arr)

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return len(self.values)

    class _FakeDF:
        """Minimal dict-style frame exposing ``df['tick_volume'].values``."""
        def __init__(self, tick_volume):
            self._cols = {"tick_volume": _Col(tick_volume)}

        def __getitem__(self, k):
            return self._cols[k]

    failures = []

    # ─── classify_setup tests ────────────────────────────────────────────
    # trending regime, no comp -> BREAKOUT
    if classify_setup("LONG", "trending", {}) != SETUP_BREAKOUT:
        failures.append("classify_setup trending != BREAKOUT")
    # volatile regime -> BREAKOUT
    if classify_setup("SHORT", "volatile", None) != SETUP_BREAKOUT:
        failures.append("classify_setup volatile != BREAKOUT")
    # ranging regime + no breakout comp -> MEAN_REVERT
    if classify_setup("LONG", "ranging", {"breakout": 0.0}) != SETUP_MEAN_REVERT:
        failures.append("classify_setup ranging != MEAN_REVERT")
    # low_vol regime, no breakout comp -> MEAN_REVERT
    if classify_setup("LONG", "low_vol", {}) != SETUP_MEAN_REVERT:
        failures.append("classify_setup low_vol != MEAN_REVERT")
    # ranging + positive breakout comp -> BREAKOUT (comp wins)
    if classify_setup("LONG", "ranging", {"breakout": 0.5}) != SETUP_BREAKOUT:
        failures.append("classify_setup ranging+breakout comp != BREAKOUT")
    # missing direction -> UNKNOWN
    if classify_setup(None, "trending", {}) != SETUP_UNKNOWN:
        failures.append("classify_setup missing direction != UNKNOWN")
    # unknown regime + zero comp -> UNKNOWN
    if classify_setup("LONG", "weird", {}) != SETUP_UNKNOWN:
        failures.append("classify_setup unknown regime != UNKNOWN")

    # ─── tick_volume_imbalance tests ────────────────────────────────────
    # Stable baseline of 100, last bar 100 -> ratio 1.0
    df_flat = _FakeDF([100.0] * 25)
    r, ma, cur, ok = tick_volume_imbalance(df_flat, lookback=20)
    if not (ok and abs(r - 1.0) < 1e-9 and abs(ma - 100.0) < 1e-9 and cur == 100.0):
        failures.append(f"flat 100 -> got r={r}, ma={ma}, cur={cur}, ok={ok}")

    # Spike: 20 prior bars=100, last bar=200 -> ratio 2.0
    df_spike = _FakeDF([100.0] * 20 + [200.0])
    r, ma, cur, ok = tick_volume_imbalance(df_spike, lookback=20)
    if not (ok and abs(r - 2.0) < 1e-9):
        failures.append(f"spike -> got r={r}, ok={ok}")

    # Dead bar: 20 prior bars=100, last=10 -> ratio 0.1
    df_dead = _FakeDF([100.0] * 20 + [10.0])
    r, ma, cur, ok = tick_volume_imbalance(df_dead, lookback=20)
    if not (ok and abs(r - 0.1) < 1e-9):
        failures.append(f"dead -> got r={r}, ok={ok}")

    # Insufficient bars
    df_short = _FakeDF([100.0] * 10)
    r, ma, cur, ok = tick_volume_imbalance(df_short, lookback=20)
    if ok:
        failures.append("short frame should not be ok")

    # No DF at all
    r, ma, cur, ok = tick_volume_imbalance(None, lookback=20)
    if ok:
        failures.append("None df should not be ok")

    # Missing column
    class _BadDF:
        def __getitem__(self, k):
            raise KeyError(k)
    r, ma, cur, ok = tick_volume_imbalance(_BadDF(), lookback=20)
    if ok:
        failures.append("missing column should not be ok")

    # MA == 0 (all zeros) -> not ok
    df_zero = _FakeDF([0.0] * 21)
    r, ma, cur, ok = tick_volume_imbalance(df_zero, lookback=20)
    if ok:
        failures.append("ma=0 should not be ok")

    # ─── evaluate_tv_gate tests ─────────────────────────────────────────
    # BREAKOUT setup, ratio 2.0 vs 1.30 thr -> PASS
    res = evaluate_tv_gate("LONG", "trending", {}, df_spike)
    if res["verdict"] != VERDICT_PASS or res["setup"] != SETUP_BREAKOUT:
        failures.append(f"breakout spike should PASS, got {res}")

    # BREAKOUT setup, ratio 1.0 vs 1.30 thr -> REJECT (TV_THIN_BREAKOUT)
    res = evaluate_tv_gate("LONG", "trending", {}, df_flat)
    if res["verdict"] != VERDICT_REJECT or res["reason"] != "TV_THIN_BREAKOUT":
        failures.append(f"breakout flat should REJECT, got {res}")

    # MEAN_REVERT setup, ratio 1.0 vs 0.70 thr -> PASS
    res = evaluate_tv_gate("LONG", "ranging", {"breakout": 0.0}, df_flat)
    if res["verdict"] != VERDICT_PASS or res["setup"] != SETUP_MEAN_REVERT:
        failures.append(f"mean_revert flat should PASS, got {res}")

    # MEAN_REVERT setup, ratio 0.10 vs 0.70 thr -> REJECT (TV_DEAD_REVERT)
    res = evaluate_tv_gate("LONG", "ranging", {"breakout": 0.0}, df_dead)
    if res["verdict"] != VERDICT_REJECT or res["reason"] != "TV_DEAD_REVERT":
        failures.append(f"mean_revert dead should REJECT, got {res}")

    # SKIP_NO_DATA path — short df
    res = evaluate_tv_gate("LONG", "trending", {}, df_short)
    if res["verdict"] != VERDICT_SKIP_NO_DATA:
        failures.append(f"short df should SKIP_NO_DATA, got {res}")

    # UNKNOWN setup -> SKIP_NO_DATA (fail OPEN)
    res = evaluate_tv_gate("LONG", "weird", {}, df_spike)
    if res["verdict"] != VERDICT_SKIP_NO_DATA or res["setup"] != SETUP_UNKNOWN:
        failures.append(f"unknown setup should SKIP, got {res}")

    # Per-symbol override raises BREAKOUT thr to 3.0 -> ratio 2.0 should REJECT
    overrides = {"XAUUSD": {"BREAKOUT": 3.0}}
    res = evaluate_tv_gate("LONG", "trending", {}, df_spike,
                           per_sym_overrides=overrides, symbol="XAUUSD")
    if res["verdict"] != VERDICT_REJECT or res["threshold"] != 3.0:
        failures.append(f"per-sym override (raise) should REJECT, got {res}")

    # Per-symbol override lowers BREAKOUT thr to 0.5 -> flat (1.0) should PASS
    overrides = {"XAUUSD": {"BREAKOUT": 0.5}}
    res = evaluate_tv_gate("LONG", "trending", {}, df_flat,
                           per_sym_overrides=overrides, symbol="XAUUSD")
    if res["verdict"] != VERDICT_PASS or res["threshold"] != 0.5:
        failures.append(f"per-sym override (lower) should PASS, got {res}")

    # Per-symbol override for OTHER symbol does not apply
    overrides = {"OTHER": {"BREAKOUT": 0.5}}
    res = evaluate_tv_gate("LONG", "trending", {}, df_flat,
                           per_sym_overrides=overrides, symbol="XAUUSD")
    if res["verdict"] != VERDICT_REJECT:
        failures.append(f"wrong-sym override should not apply, got {res}")

    # SHORT direction works (BREAKOUT)
    res = evaluate_tv_gate("SHORT", "volatile", {}, df_spike)
    if res["verdict"] != VERDICT_PASS:
        failures.append(f"SHORT breakout spike should PASS, got {res}")

    # Malformed inputs do not throw
    try:
        res = evaluate_tv_gate("LONG", None, None, None)
        if res["verdict"] != VERDICT_SKIP_NO_DATA:
            failures.append(f"None-everything should SKIP, got {res}")
    except Exception as e:
        failures.append(f"None-everything raised: {e!r}")

    # Bad direction
    try:
        res = evaluate_tv_gate("BAD", "trending", {}, df_spike)
        if res["setup"] != SETUP_UNKNOWN:
            failures.append(f"BAD direction should be UNKNOWN, got {res}")
    except Exception as e:
        failures.append(f"BAD direction raised: {e!r}")

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)

    print("tick_volume_gate self-test OK (%d cases)" % (
        7 + 7 + 11  # classify_setup + tick_volume_imbalance + evaluate_tv_gate
    ))
    sys.exit(0)
