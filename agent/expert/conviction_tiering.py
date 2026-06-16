"""
agent/expert/conviction_tiering.py
══════════════════════════════════════════════════════════════════════════
Conviction Tiering A+/B+/B (Phase 1 spec)

Replaces the legacy CONVICTION_SIZING_V2 4-bucket continuous-multiplier
ladder with a discrete 3-tier gate-and-size classifier.

Design intent
─────────────
A trade signal that survives the MasterBrain portfolio-gate (line 2528
in brain.py) is classified into exactly one of {A+, B+, B, FAIL}:

  • A+  → raw_score ≥ 9.0, SQ ≥ 75, n_strong ≥ 6, HTF aligned, M15 ok,
          structural print (ICT sweep+reclaim OR Wyckoff absorption),
          regime ∈ {trending, volatile}.
          size_mult = 2.0×   (joins the existing min-chain de-stack)

  • B+  → raw_score ∈ [7, 9), SQ ≥ 60, n_strong ≥ 4, (HTF ok OR M15 ok).
          size_mult = 1.0×   (base risk)

  • B   → raw_score ∈ [6, 7) — marginal — SKIP (size_mult = 0).

  • FAIL → raw_score < 6.0 — SKIP.

The classifier is **stateless** and **module-level** so it is REPL-testable.
It NEVER calls anything that has side effects (no MT5 queries, no journal
writes, no logger calls). All inputs are passed in by brain.py.

Min-chain rule (preserved)
──────────────────────────
brain.py keeps `adaptive_mult = min(conv_mult, rl_mult, sess_dow_mult,
vrp_mult)` unchanged. SYMBOL_RISK_CAP + MAX_RISK_PER_TRADE_PCT still
clamp at the existing line in PHASE 3 — so a 2.0× A+ on a 0.5%-capped
symbol cannot exceed 0.5%.

Literature anchors
──────────────────
  • ICT sweep+reclaim   → already wired (detect_liquidity_sweep, Gate 3f).
  • Wyckoff Spring/UTAD → high-volume absorption at swept extreme,
                          approximated by OrderFlowIntel.absorption AND
                          ob_present.
  • HTF alignment       → MTF cascade W1/D1/H4 ≥ 2-of-3 (Raschke).
  • n_strong ≥ 6        → Bourgade & Hassani (arXiv 2009.08821): only
                          count components contributing ≥0.5 (full-weight
                          signal); half-weights are the laggard tail.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────
# Local copy of the ICT sweep+reclaim detector
#
# This is intentionally duplicated from brain.py:151 so the module has no
# circular-import dependency on the brain package. The algorithm is the
# same as detect_liquidity_sweep(); if brain.py ever changes its
# definition, update this copy in lock-step.
# ─────────────────────────────────────────────────────────────────────────
def _detect_liquidity_sweep_local(highs, lows, closes, direction,
                                  lookback: int = 24, n: int = 5) -> bool:
    try:
        import numpy as _np
        h = _np.asarray(highs, dtype=float)
        l = _np.asarray(lows, dtype=float)
        c = _np.asarray(closes, dtype=float)
    except Exception:
        return False
    N = len(c)
    if N < (n + 2) or lookback < 1 or n < 1:
        return False
    start = max(n, N - lookback)
    if direction == "LONG":
        for i in range(start, N):
            try:
                prior_min = float(l[i - n:i].min())
                if float(l[i]) < prior_min and float(c[i]) > prior_min:
                    return True
            except Exception:
                continue
        return False
    elif direction == "SHORT":
        for i in range(start, N):
            try:
                prior_max = float(h[i - n:i].max())
                if float(h[i]) > prior_max and float(c[i]) < prior_max:
                    return True
            except Exception:
                continue
        return False
    return False


# ─────────────────────────────────────────────────────────────────────────
# Config loader (deferred so test harness can monkey-patch)
# ─────────────────────────────────────────────────────────────────────────
def _cfg(name: str, default):
    """Read attribute from config.py, fall back to default if missing.

    Deferred import so unit tests can stub the module before this runs.
    """
    try:
        import config as _config  # type: ignore
        return getattr(_config, name, default)
    except Exception:
        return default


def _apply_symbol_overrides(symbol: str, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """Merge global thresholds with optional per-symbol overrides.

    CONV_TIER_PER_SYMBOL_OVERRIDES = {symbol: {threshold_name: value, ...}}
    """
    overrides_all = _cfg("CONV_TIER_PER_SYMBOL_OVERRIDES", {}) or {}
    sym_over = overrides_all.get(symbol) if isinstance(overrides_all, dict) else None
    if isinstance(sym_over, dict):
        merged = dict(thresholds)
        merged.update(sym_over)
        return merged
    return thresholds


# ─────────────────────────────────────────────────────────────────────────
# Main entry — classify_conviction
# ─────────────────────────────────────────────────────────────────────────
def classify_conviction(
    symbol: str,
    direction: str,
    raw_score: float,
    signal_quality: float,
    comp_long: Optional[Dict[str, float]],
    comp_short: Optional[Dict[str, float]],
    mtf_aligned: int,
    m15_dir: Optional[str],
    h1_df: Any,
    ind: Optional[Dict[str, Any]] = None,
    bi: Optional[Dict[str, Any]] = None,
    order_flow_intel: Any = None,
    regime: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify a (symbol, direction) signal into A+ / B+ / B / FAIL.

    Returns a dict shaped::

        {
            "tier":      "A+" | "B+" | "B" | "FAIL",
            "size_mult": float,        # 0.0 => SKIP
            "reason":    str,          # short tag for logs/dashboard
            "scorecard": {             # diagnostic dump
                "rs":         float,
                "sq":         float,
                "n_strong":   int,
                "n_active":   int,
                "mtf_align":  int,
                "m15_ok":     bool,
                "ob":         bool,
                "wyckoff":    bool,
                "regime":     str | None,
            },
        }
    """
    # ── Pull thresholds (with per-symbol override merge) ────────────────
    thr: Dict[str, Any] = {
        "APLUS_RAW_MIN":          _cfg("CONV_APLUS_RAW_MIN", 9.0),
        "APLUS_SQ_MIN":           _cfg("CONV_APLUS_SQ_MIN", 75),
        "APLUS_STRONG_MIN":       _cfg("CONV_APLUS_STRONG_MIN", 6),
        "APLUS_ALLOWED_REGIMES":  _cfg("CONV_APLUS_ALLOWED_REGIMES",
                                       {"trending", "volatile"}),
        "APLUS_SIZE_MULT":        _cfg("CONV_APLUS_SIZE_MULT", 2.0),
        "BPLUS_RAW_MIN":          _cfg("CONV_BPLUS_RAW_MIN", 7.0),
        "BPLUS_SQ_MIN":           _cfg("CONV_BPLUS_SQ_MIN", 60),
        "BPLUS_STRONG_MIN":       _cfg("CONV_BPLUS_STRONG_MIN", 4),
        "BPLUS_SIZE_MULT":        _cfg("CONV_BPLUS_SIZE_MULT", 1.0),
        "B_RAW_MIN":              _cfg("CONV_B_RAW_MIN", 6.0),
        "HTF_MIN_ALIGNED":        _cfg("CONV_HTF_MIN_ALIGNED", 2),
        "OB_LOOKBACK":            _cfg("CONV_OB_LOOKBACK_BARS", 24),
        "OB_FRACTAL_N":           _cfg("CONV_OB_FRACTAL_N", 5),
        "REQUIRE_STRUCTURAL_APLUS": _cfg("CONV_REQUIRE_STRUCTURAL_APLUS", True),
        "BPLUS_REQUIRE_HTF_OR_M15": _cfg("CONV_BPLUS_REQUIRE_HTF_OR_M15", True),
    }
    thr = _apply_symbol_overrides(symbol, thr)

    # ── Normalize inputs ───────────────────────────────────────────────
    try:
        rs = float(raw_score)
    except Exception:
        rs = 0.0
    try:
        sq = float(signal_quality)
    except Exception:
        sq = 0.0
    try:
        mtf_align = int(mtf_aligned or 0)
    except Exception:
        mtf_align = 0

    # Pick the component dict for this direction.
    if direction == "LONG":
        comp = comp_long or {}
    elif direction == "SHORT":
        comp = comp_short or {}
    else:
        comp = {}

    # n_active = components > 0; n_strong = components >= 0.5
    n_active = 0
    n_strong = 0
    for v in comp.values():
        try:
            fv = float(v)
        except Exception:
            continue
        if fv > 0.0:
            n_active += 1
            if fv >= 0.5:
                n_strong += 1

    m15_ok = bool(m15_dir is not None and m15_dir == direction)
    htf_ok = bool(mtf_align >= int(thr["HTF_MIN_ALIGNED"]))

    # ── Structural print: ICT sweep+reclaim ────────────────────────────
    ob_present = False
    if h1_df is not None:
        try:
            highs = h1_df["high"].values
            lows = h1_df["low"].values
            closes = h1_df["close"].values
            ob_present = _detect_liquidity_sweep_local(
                highs, lows, closes, direction,
                lookback=int(thr["OB_LOOKBACK"]),
                n=int(thr["OB_FRACTAL_N"]),
            )
        except Exception:
            ob_present = False

    # ── Wyckoff Spring/UTAD proxy (absorption at swept extreme) ────────
    wyckoff_present = False
    if order_flow_intel is not None:
        try:
            of = order_flow_intel.evaluate(symbol)
            absorption = False
            if isinstance(of, dict):
                absorption = bool(of.get("absorption", False))
            wyckoff_present = bool(absorption and ob_present)
        except Exception:
            wyckoff_present = False

    structural_ok = bool(ob_present or wyckoff_present)

    allowed_regimes = thr["APLUS_ALLOWED_REGIMES"]
    regime_ok_for_aplus = (
        allowed_regimes is None
        or (regime is not None and regime in allowed_regimes)
    )

    scorecard = {
        "rs":        rs,
        "sq":        sq,
        "n_strong":  n_strong,
        "n_active":  n_active,
        "mtf_align": mtf_align,
        "m15_ok":    m15_ok,
        "htf_ok":    htf_ok,
        "ob":        bool(ob_present),
        "wyckoff":   bool(wyckoff_present),
        "regime":    regime,
    }

    # ── Decision tree ──────────────────────────────────────────────────
    require_struct = bool(thr["REQUIRE_STRUCTURAL_APLUS"])
    aplus_structural_pass = (structural_ok if require_struct else True)

    if (rs >= float(thr["APLUS_RAW_MIN"])
            and sq >= float(thr["APLUS_SQ_MIN"])
            and n_strong >= int(thr["APLUS_STRONG_MIN"])
            and htf_ok
            and m15_ok
            and aplus_structural_pass
            and regime_ok_for_aplus):
        return {
            "tier":      "A+",
            "size_mult": float(thr["APLUS_SIZE_MULT"]),
            "reason":    "raw>=9 + structural + HTF + M15",
            "scorecard": scorecard,
        }

    bplus_gate_pass = (
        (htf_ok or m15_ok)
        if bool(thr["BPLUS_REQUIRE_HTF_OR_M15"])
        else True
    )

    if (rs >= float(thr["BPLUS_RAW_MIN"])
            and sq >= float(thr["BPLUS_SQ_MIN"])
            and n_strong >= int(thr["BPLUS_STRONG_MIN"])
            and bplus_gate_pass):
        return {
            "tier":      "B+",
            "size_mult": float(thr["BPLUS_SIZE_MULT"]),
            "reason":    "raw 7-9 + most gates",
            "scorecard": scorecard,
        }

    if rs >= float(thr["B_RAW_MIN"]):
        return {
            "tier":      "B",
            "size_mult": 0.0,
            "reason":    "marginal raw 6-7 - SKIP",
            "scorecard": scorecard,
        }

    return {
        "tier":      "FAIL",
        "size_mult": 0.0,
        "reason":    "below floor",
        "scorecard": scorecard,
    }


# ═════════════════════════════════════════════════════════════════════════
# Self-test — synthetic fixtures, exit 0 on pass.
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    class _FakeOrderFlow:
        def __init__(self, absorption: bool):
            self._absorption = absorption

        def evaluate(self, symbol: str):
            return {"absorption": self._absorption}

    class _FakeH1DF:
        """Minimal dict-style wrapper exposing .values for high/low/close."""
        def __init__(self, highs, lows, closes):
            class _Col:
                def __init__(self, arr):
                    self.values = arr
            self._cols = {"high": _Col(highs), "low": _Col(lows), "close": _Col(closes)}

        def __getitem__(self, k):
            return self._cols[k]

    def _h1_with_long_sweep():
        """Construct H1 OHLC where the last bar sweeps below a recent low
        and reclaims it (LONG sweep+reclaim pattern)."""
        # 30 bars: gentle range, then last bar dips below min(low[-6:-1])
        # and closes back above it.
        highs  = [110.0] * 30
        lows   = [100.0] * 30
        closes = [105.0] * 30
        # plant a recent prior swing low at index 25 (well above 90)
        lows[25]   = 100.0
        # last bar: low pierces below 100, close back above 100
        lows[-1]   = 99.0
        closes[-1] = 100.5
        highs[-1]  = 101.0
        return _FakeH1DF(highs, lows, closes)

    def _h1_no_sweep():
        highs  = [110.0] * 30
        lows   = [100.0] * 30
        closes = [105.0] * 30
        return _FakeH1DF(highs, lows, closes)

    # 11 components — n_strong = sum(v >= 0.5)
    def comp_strong(n_strong: int) -> Dict[str, float]:
        d = {f"c{i}": (1.0 if i < n_strong else 0.2) for i in range(11)}
        return d

    failures = []

    # ── Case 1: A+ happy path ──
    res = classify_conviction(
        symbol="XAUUSD",
        direction="LONG",
        raw_score=9.5,
        signal_quality=80,
        comp_long=comp_strong(7),
        comp_short={},
        mtf_aligned=3,
        m15_dir="LONG",
        h1_df=_h1_with_long_sweep(),
        ind={},
        bi={},
        order_flow_intel=_FakeOrderFlow(absorption=True),
        regime="trending",
    )
    if res["tier"] != "A+" or res["size_mult"] != 2.0:
        failures.append(f"Case1 A+ expected, got {res}")

    # ── Case 2: raw 8.0 (B+ band) with good gates ──
    res = classify_conviction(
        symbol="XAUUSD",
        direction="LONG",
        raw_score=8.0,
        signal_quality=70,
        comp_long=comp_strong(5),
        comp_short={},
        mtf_aligned=2,
        m15_dir="LONG",
        h1_df=_h1_no_sweep(),
        ind={},
        bi={},
        order_flow_intel=None,
        regime="trending",
    )
    if res["tier"] != "B+" or res["size_mult"] != 1.0:
        failures.append(f"Case2 B+ expected, got {res}")

    # ── Case 3: raw 6.5 marginal → B SKIP ──
    res = classify_conviction(
        symbol="XAUUSD",
        direction="LONG",
        raw_score=6.5,
        signal_quality=55,
        comp_long=comp_strong(3),
        comp_short={},
        mtf_aligned=1,
        m15_dir="LONG",
        h1_df=_h1_no_sweep(),
        regime="ranging",
    )
    if res["tier"] != "B" or res["size_mult"] != 0.0:
        failures.append(f"Case3 B expected, got {res}")

    # ── Case 4: raw 5.0 → FAIL ──
    res = classify_conviction(
        symbol="XAUUSD",
        direction="LONG",
        raw_score=5.0,
        signal_quality=40,
        comp_long=comp_strong(2),
        comp_short={},
        mtf_aligned=0,
        m15_dir=None,
        h1_df=_h1_no_sweep(),
        regime=None,
    )
    if res["tier"] != "FAIL" or res["size_mult"] != 0.0:
        failures.append(f"Case4 FAIL expected, got {res}")

    # ── Case 5: A+ raw OK but n_strong=4 (< 6) demotes to B+ ──
    res = classify_conviction(
        symbol="XAUUSD",
        direction="LONG",
        raw_score=9.5,
        signal_quality=80,
        comp_long=comp_strong(4),       # only 4 strong components
        comp_short={},
        mtf_aligned=3,
        m15_dir="LONG",
        h1_df=_h1_with_long_sweep(),
        order_flow_intel=_FakeOrderFlow(absorption=True),
        regime="trending",
    )
    # raw 9.5 + SQ 80 + n_strong 4 → A+ fails (needs 6); falls to B+
    # B+ needs raw ≥7 + SQ ≥60 + n_strong ≥4 + (htf_ok or m15_ok) → PASS.
    if res["tier"] != "B+":
        failures.append(f"Case5 B+ expected (A+ demoted by n_strong), got {res}")

    # ── Case 6: A+ raw OK but ranging regime → demoted to B+ ──
    res = classify_conviction(
        symbol="XAUUSD",
        direction="LONG",
        raw_score=9.5,
        signal_quality=80,
        comp_long=comp_strong(7),
        comp_short={},
        mtf_aligned=3,
        m15_dir="LONG",
        h1_df=_h1_with_long_sweep(),
        order_flow_intel=_FakeOrderFlow(absorption=True),
        regime="ranging",       # NOT in default {trending, volatile}
    )
    if res["tier"] != "B+":
        failures.append(f"Case6 ranging-demote-to-B+ expected, got {res}")

    # ── Case 7: A+ raw OK but no structural print + no absorption → B+ ──
    res = classify_conviction(
        symbol="XAUUSD",
        direction="LONG",
        raw_score=9.5,
        signal_quality=80,
        comp_long=comp_strong(7),
        comp_short={},
        mtf_aligned=3,
        m15_dir="LONG",
        h1_df=_h1_no_sweep(),   # no ICT print
        order_flow_intel=None,
        regime="trending",
    )
    if res["tier"] != "B+":
        failures.append(f"Case7 no-structural demote-to-B+ expected, got {res}")

    # ── Case 8: SHORT direction works symmetrically ──
    # Build a SHORT sweep: last bar pierces above recent max and reclaims down.
    highs  = [110.0] * 30
    lows   = [100.0] * 30
    closes = [105.0] * 30
    highs[-1]  = 111.0           # pierces above prior max 110
    closes[-1] = 109.5           # reclaims back below
    lows[-1]   = 108.0
    h1_short = _FakeH1DF(highs, lows, closes)
    res = classify_conviction(
        symbol="XAUUSD",
        direction="SHORT",
        raw_score=9.2,
        signal_quality=78,
        comp_long={},
        comp_short=comp_strong(6),
        mtf_aligned=2,
        m15_dir="SHORT",
        h1_df=h1_short,
        order_flow_intel=_FakeOrderFlow(absorption=True),
        regime="volatile",
    )
    if res["tier"] != "A+":
        failures.append(f"Case8 SHORT A+ expected, got {res}")

    # ── Case 9: scorecard shape sanity ──
    sc = res["scorecard"]
    required = {"rs", "sq", "n_strong", "n_active", "mtf_align",
                "m15_ok", "ob", "wyckoff", "regime"}
    if not required.issubset(sc.keys()):
        failures.append(f"Case9 scorecard missing keys: {required - set(sc.keys())}")

    # ── Case 10: malformed inputs do not throw ──
    try:
        res = classify_conviction(
            symbol="?", direction="?",
            raw_score="garbage",            # type: ignore[arg-type]
            signal_quality=None,            # type: ignore[arg-type]
            comp_long=None, comp_short=None,
            mtf_aligned=None,               # type: ignore[arg-type]
            m15_dir=None,
            h1_df=None,
            order_flow_intel=None,
            regime=None,
        )
        if res["tier"] not in {"A+", "B+", "B", "FAIL"}:
            failures.append(f"Case10 bad-input fallback tier malformed: {res}")
    except Exception as e:
        failures.append(f"Case10 raised: {e!r}")

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)

    print("conviction_tiering self-test OK (10 cases)")
    sys.exit(0)
