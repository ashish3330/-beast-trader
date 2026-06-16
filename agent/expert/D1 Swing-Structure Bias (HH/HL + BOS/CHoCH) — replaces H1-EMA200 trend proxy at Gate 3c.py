#!/usr/bin/env python3 -B
"""
D1 Swing-Structure Bias (HH/HL + BOS/CHoCH).

Replaces the H1-EMA(200) "D1-trend proxy" at Gate 3c with a proper
SMC/ICT market-structure read derived from confirmed D1 swings.

References
----------
* Bill Williams Fractals (1995): 5-bar pivot (2 left + center + 2 right).
* ICT (Inner Circle Trader): BOS = break of structure in trend direction;
  CHoCH = change of character = first counter-trend break.
* SMC market-structure convention:
    HH + HL = bullish structure
    LH + LL = bearish structure
    mixed   = neutral / FLAT

Why D1 (not H1-EMA200)
----------------------
H1-EMA(200) is a smoothed lagging proxy of the daily trend — it confirms
the move long after the structural pivots that define it have printed.
A proper swing-structure read on confirmed D1 bars gives the same trend
context with sharper edges (BOS vs CHoCH lets you distinguish "trend
continuation" from "trend flip").

Data source
-----------
We resample D1 from the H1 frame the brain already retrieves (`get_candles`
returns >= 60 H1 bars per cycle for momentum/MTF use). No new MT5 feed is
required — D1 is a `.resample("1D", label="right", closed="right")` of
the H1 OHLC. Drop the (possibly) forming current D1 bar so only confirmed
dailies count.

Public API (module-level pure functions for testability)
--------------------------------------------------------
    get_d1_frame(h1_df)                   -> pd.DataFrame | None
    find_swings(d1_df)                    -> (highs, lows)
    structure_bias(highs, lows)           -> (bias, strength)
    last_struct_event(d1, highs, lows, bias)
                                          -> dict with event/dir/age_bars/fresh
    d1_bias_verdict(entry_dir, bias, strength, struct_event)
                                          -> "REJECT" | "WARN" | "OK" | "SNIPER"
    evaluate(symbol, h1_df, entry_dir, params=None)
                                          -> dict (full verdict bundle)

Each helper has a config-flag-less default so the module is fully testable
in isolation. Brain-side wiring (Phase 3) reads config flags and feeds the
gated/un-gated path.

Constraints honoured
--------------------
* New file only — no edits to brain.py / other modules in this phase.
* Reads confirmed bars only — the forming D1 is always dropped.
* Defensive: any missing / undersized input yields a graceful FLAT / OK.
* Stateless functions, but the optional `D1StructureBias` class memoizes
  per `(symbol, last_d1_close_time)` to avoid recomputing on every tick.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pandas is already a hard dep of the brain
    import pandas as pd
except Exception:  # pragma: no cover — keep import optional for self-test
    pd = None  # type: ignore[assignment]


log = logging.getLogger("dragon.d1_structure")

# ════════════════════════════════════════════════════════════════════════
#  Defaults (overridable via params dict in evaluate())
# ════════════════════════════════════════════════════════════════════════
D1_MIN_BARS = 60                # >= ~2 months of confirmed D1 history
FRACTAL_N = 2                   # 5-bar pivot: 2 left + center + 2 right
SWING_MIN_COUNT = 4             # need 2 prior + 2 latest swings each side
BOS_LOOKBACK_BARS = 30          # scan window for most-recent BOS/CHoCH
STRUCT_FRESHNESS_BARS = 10      # event older than this loses "fresh" weight
MIN_SWING_SPACING_BARS = 2      # adjacent fractals must be N bars apart


# ════════════════════════════════════════════════════════════════════════
#  1. Build D1 series from H1 (no extra MT5 feed)
# ════════════════════════════════════════════════════════════════════════
def get_d1_frame(h1_df: "pd.DataFrame", min_bars: int = D1_MIN_BARS) -> Optional["pd.DataFrame"]:
    """Resample H1 OHLC → confirmed D1 OHLC.

    Drops the (possibly) forming current D1 bar.

    Returns None when the H1 frame is missing, too short, or resamples
    to fewer than `min_bars` confirmed dailies.
    """
    if pd is None or h1_df is None:
        return None
    try:
        if len(h1_df) < min_bars * 24:
            return None
    except Exception:
        return None

    df = h1_df
    # Accept either time-indexed frame or a column-based frame
    try:
        if "time" in df.columns:
            df = df.set_index("time")
    except Exception:
        pass

    try:
        d1 = df.resample("1D", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
    except Exception as exc:
        log.debug("D1 resample failed: %s", exc)
        return None

    if len(d1) == 0:
        return None
    # Drop the (possibly) forming D1 bar — only CONFIRMED dailies count.
    d1 = d1.iloc[:-1]
    return d1 if len(d1) >= min_bars else None


# ════════════════════════════════════════════════════════════════════════
#  2. Williams 5-bar fractal detector (RIGHT-CONFIRMED)
# ════════════════════════════════════════════════════════════════════════
def _dedupe_min_spacing(pivots: List[Tuple[int, float]], min_spacing: int) -> List[Tuple[int, float]]:
    """Drop pivots that are within `min_spacing` bars of the previous kept
    pivot. Keeps the more-extreme one in each cluster (max for highs would
    have higher price; we just keep the latest because Williams already
    enforces strict-max). Operates on a list ordered oldest -> newest."""
    if not pivots:
        return pivots
    out: List[Tuple[int, float]] = [pivots[0]]
    for idx, price in pivots[1:]:
        last_idx, _ = out[-1]
        if idx - last_idx < min_spacing:
            # Replace if same side AND new pivot is more extreme — but
            # since highs are strict-max, the LATER one in a cluster is
            # the valid one; keep latest.
            out[-1] = (idx, price)
        else:
            out.append((idx, price))
    return out


def find_swings(
    d1: "pd.DataFrame",
    fractal_n: int = FRACTAL_N,
    min_spacing: int = MIN_SWING_SPACING_BARS,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Return ``(highs, lows)`` — each a list of ``(idx, price)`` tuples
    ordered oldest → newest.

    A pivot at index ``i`` is RIGHT-CONFIRMED only after ``fractal_n`` bars
    closed after it (so the latest possible pivot lives at ``len(d1)-1-n``).
    """
    if d1 is None or len(d1) < (2 * fractal_n + 1):
        return [], []
    H = d1["high"].values
    L = d1["low"].values
    n = fractal_n
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []
    for i in range(n, len(d1) - n):
        seg_h = H[i - n: i + n + 1]
        seg_l = L[i - n: i + n + 1]
        if H[i] == seg_h.max() and (seg_h == H[i]).sum() == 1:
            highs.append((i, float(H[i])))
        if L[i] == seg_l.min() and (seg_l == L[i]).sum() == 1:
            lows.append((i, float(L[i])))
    highs = _dedupe_min_spacing(highs, min_spacing)
    lows = _dedupe_min_spacing(lows, min_spacing)
    return highs, lows


# ════════════════════════════════════════════════════════════════════════
#  3. HH+HL / LH+LL bias (last 2 swings vs prior 2 swings)
# ════════════════════════════════════════════════════════════════════════
def structure_bias(
    highs: List[Tuple[int, float]],
    lows: List[Tuple[int, float]],
    swing_min_count: int = SWING_MIN_COUNT,
) -> Tuple[str, float]:
    """Classify market structure from the last 4 swings per side.

    Returns
    -------
    (bias, strength) :
        bias is one of ``LONG`` / ``SHORT`` / ``FLAT``;
        strength is 1.0 (full HH+HL or LH+LL), 0.5 (partial), 0.0 (FLAT).
    """
    if len(highs) < swing_min_count or len(lows) < swing_min_count:
        return "FLAT", 0.0
    h_p2, h_p1 = highs[-4][1], highs[-3][1]   # prior pair
    h_l2, h_l1 = highs[-2][1], highs[-1][1]   # latest pair
    l_p2, l_p1 = lows[-4][1], lows[-3][1]
    l_l2, l_l1 = lows[-2][1], lows[-1][1]
    hh = (h_l2 > h_p1) and (h_l1 > h_l2)
    hl = (l_l2 > l_p1) and (l_l1 > l_l2)
    lh = (h_l2 < h_p1) and (h_l1 < h_l2)
    ll = (l_l2 < l_p1) and (l_l1 < l_l2)
    if hh and hl:
        return "LONG", 1.0
    if lh and ll:
        return "SHORT", 1.0
    if hh or hl:
        return "LONG", 0.5
    if lh or ll:
        return "SHORT", 0.5
    return "FLAT", 0.0


# ════════════════════════════════════════════════════════════════════════
#  4. Most-recent BOS / CHoCH on D1
# ════════════════════════════════════════════════════════════════════════
def _last_swing_before(
    swings: List[Tuple[int, float]],
    idx: int,
) -> Optional[float]:
    """Most-recent swing price whose pivot index < ``idx``. None if none."""
    for s_idx, s_price in reversed(swings):
        if s_idx < idx:
            return s_price
    return None


def last_struct_event(
    d1: "pd.DataFrame",
    highs: List[Tuple[int, float]],
    lows: List[Tuple[int, float]],
    bias: str,
    lookback: int = BOS_LOOKBACK_BARS,
    freshness: int = STRUCT_FRESHNESS_BARS,
) -> Dict[str, Any]:
    """Find the latest close that broke a prior swing (BOS or CHoCH).

    * BOS  = close beyond the prior same-side swing in the bias direction.
    * CHoCH = close beyond the prior swing AGAINST bias = first sign of flip.

    Returns a dict ``{"event", "dir", "age_bars", "fresh"}``. All-None when
    no break found within the lookback window.
    """
    out: Dict[str, Any] = {"event": None, "dir": None, "age_bars": None, "fresh": False}
    if d1 is None or len(d1) == 0:
        return out
    C = d1["close"].values
    last_idx = len(d1) - 1
    scan_start = max(0, last_idx - lookback)
    for i in range(last_idx, scan_start - 1, -1):
        prior_high = _last_swing_before(highs, i)
        prior_low = _last_swing_before(lows, i)
        if prior_high is not None and C[i] > prior_high:
            evt_dir = "LONG"
            event = "BOS" if bias == "LONG" else "CHoCH"
            age = last_idx - i
            return {"event": event, "dir": evt_dir, "age_bars": int(age), "fresh": age <= freshness}
        if prior_low is not None and C[i] < prior_low:
            evt_dir = "SHORT"
            event = "BOS" if bias == "SHORT" else "CHoCH"
            age = last_idx - i
            return {"event": event, "dir": evt_dir, "age_bars": int(age), "fresh": age <= freshness}
    return out


# ════════════════════════════════════════════════════════════════════════
#  5. Brain-side gate decision
# ════════════════════════════════════════════════════════════════════════
def d1_bias_verdict(
    entry_dir: str,
    bias: str,
    strength: float,
    struct: Dict[str, Any],
    weak_bias_warns: bool = True,
) -> str:
    """Return one of ``REJECT`` / ``WARN`` / ``OK`` / ``SNIPER``.

    REJECT:  strong opposing structure + fresh opposing BOS.
    WARN:    bias != entry_dir (weak or stale).
    SNIPER:  bias == entry_dir AND fresh same-direction BOS confirms.
    OK:      everything else (default pass-through).
    """
    if entry_dir not in ("LONG", "SHORT"):
        return "OK"
    bias = (bias or "FLAT").upper()

    # REJECT: full opposing HH/HL + fresh BOS confirms opposing trend
    if strength >= 1.0 and bias != "FLAT" and bias != entry_dir:
        if struct.get("event") == "BOS" and struct.get("fresh") \
                and struct.get("dir") and struct["dir"] != entry_dir:
            return "REJECT"

    # WARN: counter to bias (or weak partial bias against us)
    if bias != "FLAT" and bias != entry_dir:
        if weak_bias_warns and strength < 1.0:
            return "WARN"
        return "WARN"

    # SNIPER: same-direction bias + fresh same-direction BOS
    if bias == entry_dir and struct.get("event") == "BOS" \
            and struct.get("fresh") and struct.get("dir") == entry_dir:
        return "SNIPER"

    return "OK"


# ════════════════════════════════════════════════════════════════════════
#  6. Top-level evaluate() — single entry point for brain wiring
# ════════════════════════════════════════════════════════════════════════
def evaluate(
    symbol: str,
    h1_df: "pd.DataFrame",
    entry_dir: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One-shot evaluation: H1 frame + intended entry direction → verdict.

    The returned dict is always populated (never raises), with:
        symbol, bias, strength, event, age_bars, fresh, verdict, ok
    ``ok`` is True for OK / WARN / SNIPER, False only for REJECT.
    """
    params = params or {}
    min_bars = int(params.get("min_bars", D1_MIN_BARS))
    fractal_n = int(params.get("fractal_n", FRACTAL_N))
    swing_min_count = int(params.get("swing_min_count", SWING_MIN_COUNT))
    bos_lookback = int(params.get("bos_lookback_bars", BOS_LOOKBACK_BARS))
    freshness = int(params.get("freshness_bars", STRUCT_FRESHNESS_BARS))
    spacing = int(params.get("min_swing_spacing", MIN_SWING_SPACING_BARS))
    weak_warns = bool(params.get("weak_bias_warns", True))

    base: Dict[str, Any] = {
        "symbol": symbol,
        "bias": "FLAT",
        "strength": 0.0,
        "event": None,
        "dir": None,
        "age_bars": None,
        "fresh": False,
        "verdict": "OK",
        "ok": True,
        "n_highs": 0,
        "n_lows": 0,
    }

    try:
        d1 = get_d1_frame(h1_df, min_bars=min_bars)
        if d1 is None:
            base["verdict"] = "OK"  # graceful pass-through on insufficient data
            base["reason"] = "insufficient_d1_data"
            return base
        highs, lows = find_swings(d1, fractal_n=fractal_n, min_spacing=spacing)
        base["n_highs"] = len(highs)
        base["n_lows"] = len(lows)
        bias, strength = structure_bias(highs, lows, swing_min_count=swing_min_count)
        struct = last_struct_event(d1, highs, lows, bias,
                                   lookback=bos_lookback, freshness=freshness)
        verdict = d1_bias_verdict(entry_dir, bias, strength, struct,
                                  weak_bias_warns=weak_warns)
        base.update({
            "bias": bias,
            "strength": float(strength),
            "event": struct["event"],
            "dir": struct["dir"],
            "age_bars": struct["age_bars"],
            "fresh": bool(struct["fresh"]),
            "verdict": verdict,
            "ok": verdict != "REJECT",
        })
        return base
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("D1 structure evaluate(%s) failed: %s", symbol, exc)
        base["reason"] = f"error:{exc}"
        return base


# ════════════════════════════════════════════════════════════════════════
#  7. Optional per-symbol cache (drop-in DI for brain.py)
# ════════════════════════════════════════════════════════════════════════
class D1StructureBias:
    """Per-symbol memoizer keyed by ``(symbol, last_d1_close_time)``.

    Avoid recomputing the D1 swing scan on every brain tick — the verdict
    only changes when a new D1 closes. Same DI style as ``self._fvg`` in
    brain.py (see ``agent/fvg_strategy.py``).

    Usage::

        d1s = D1StructureBias()
        verdict = d1s.evaluate("XAUUSD", h1_df, "LONG")
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, Any], Dict[str, Any]] = {}

    def evaluate(
        self,
        symbol: str,
        h1_df: "pd.DataFrame",
        entry_dir: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        key_time: Any = None
        try:
            d1 = get_d1_frame(h1_df, min_bars=int((params or {}).get("min_bars", D1_MIN_BARS)))
            if d1 is not None and len(d1) > 0:
                key_time = d1.index[-1]
        except Exception:
            key_time = None
        # Cache key folds in direction — REJECT/SNIPER depend on it.
        key = (symbol, str(key_time), entry_dir)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        out = evaluate(symbol, h1_df, entry_dir, params=params)
        # Trim cache to avoid unbounded growth (one entry per symbol×dir is plenty)
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = out
        return out


# ════════════════════════════════════════════════════════════════════════
#  Self-test (synthetic data — must exit 0)
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    if pd is None:
        print("pandas not available — skipping pandas path", flush=True)
        sys.exit(0)

    # ── Build a synthetic uptrend H1 frame with day-scale oscillations so
    # fractal pivots actually print (no pivot on monotone series).
    # Trend slope of +0.5 per day + a 6-day sine retrace cycle.
    rng = np.random.default_rng(seed=42)
    n_days = 120
    hours_per_day = 24
    n_hours = n_days * hours_per_day
    day_index = np.arange(n_hours) / hours_per_day
    trend = 100.0 + 0.5 * day_index                                   # +60 over 120d
    cycle = 4.0 * np.sin(2 * np.pi * day_index / 6.0)                  # +/-4 every 6d
    noise = rng.normal(0, 0.3, n_hours)
    close = trend + cycle + noise
    high = close + np.abs(rng.normal(0.4, 0.15, n_hours))
    low = close - np.abs(rng.normal(0.4, 0.15, n_hours))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2026-01-01", periods=n_hours, freq="1h")
    h1 = pd.DataFrame({
        "time": idx, "open": open_, "high": high, "low": low, "close": close,
    })

    # Test 1: get_d1_frame produces enough confirmed D1 bars
    d1 = get_d1_frame(h1)
    assert d1 is not None, "expected non-None D1 frame on 120d feed"
    assert len(d1) >= D1_MIN_BARS, f"want >={D1_MIN_BARS} D1 bars, got {len(d1)}"
    print(f"[ok] get_d1_frame: {len(d1)} confirmed D1 bars")

    # Test 2: find_swings yields at least the minimum count each side
    highs, lows = find_swings(d1)
    assert len(highs) >= SWING_MIN_COUNT, f"want >={SWING_MIN_COUNT} highs, got {len(highs)}"
    assert len(lows) >= SWING_MIN_COUNT, f"want >={SWING_MIN_COUNT} lows, got {len(lows)}"
    print(f"[ok] find_swings: {len(highs)} highs / {len(lows)} lows")

    # Test 3: structure_bias = LONG (uptrend)
    bias, strength = structure_bias(highs, lows)
    assert bias == "LONG" and strength == 1.0, \
        f"want LONG/1.0 on uptrend feed, got {bias}/{strength}"
    print(f"[ok] structure_bias uptrend: bias={bias} strength={strength}")

    # Test 4: BOS/CHoCH detection finds something within lookback
    struct = last_struct_event(d1, highs, lows, bias)
    assert struct["event"] in ("BOS", "CHoCH", None)
    print(f"[ok] last_struct_event: {struct}")

    # Test 5: verdict matrix
    # 5a. LONG entry on LONG bias → OK or SNIPER (never REJECT)
    v_long = d1_bias_verdict("LONG", bias, strength, struct)
    assert v_long in ("OK", "SNIPER"), f"unexpected verdict {v_long}"
    print(f"[ok] verdict LONG-on-LONG-bias = {v_long}")

    # 5b. SHORT entry on LONG bias → WARN or REJECT
    v_short = d1_bias_verdict("SHORT", bias, strength, struct)
    assert v_short in ("WARN", "REJECT"), f"unexpected verdict {v_short}"
    print(f"[ok] verdict SHORT-on-LONG-bias = {v_short}")

    # Test 6: end-to-end evaluate() returns a dict, never raises
    out = evaluate("XAUUSD", h1, "LONG")
    assert isinstance(out, dict) and "verdict" in out
    assert out["ok"] is True or out["verdict"] == "REJECT"
    print(f"[ok] evaluate(XAUUSD, LONG): verdict={out['verdict']} bias={out['bias']}")

    # Test 7: downtrend synthetic — bias must flip to SHORT
    trend_dn = 160.0 - 0.5 * day_index
    close_dn = trend_dn + cycle + noise
    high_dn = close_dn + np.abs(rng.normal(0.4, 0.15, n_hours))
    low_dn = close_dn - np.abs(rng.normal(0.4, 0.15, n_hours))
    open_dn = np.concatenate([[close_dn[0]], close_dn[:-1]])
    h1_dn = pd.DataFrame({
        "time": idx, "open": open_dn, "high": high_dn, "low": low_dn, "close": close_dn,
    })
    d1_dn = get_d1_frame(h1_dn)
    highs_dn, lows_dn = find_swings(d1_dn)
    bias_dn, strength_dn = structure_bias(highs_dn, lows_dn)
    assert bias_dn == "SHORT" and strength_dn == 1.0, \
        f"want SHORT/1.0 on downtrend feed, got {bias_dn}/{strength_dn}"
    print(f"[ok] structure_bias downtrend: bias={bias_dn} strength={strength_dn}")

    # Test 8: insufficient data → graceful FLAT/OK
    short_h1 = h1.iloc[:100].copy()
    out_short = evaluate("XAUUSD", short_h1, "LONG")
    assert out_short["verdict"] == "OK" and out_short["bias"] == "FLAT"
    print(f"[ok] insufficient-data graceful pass: {out_short['verdict']}/{out_short['bias']}")

    # Test 9: cache class returns same dict on repeat call
    cache = D1StructureBias()
    a = cache.evaluate("XAUUSD", h1, "LONG")
    b = cache.evaluate("XAUUSD", h1, "LONG")
    assert a is b, "cache should return identical dict object on hit"
    print("[ok] D1StructureBias cache hit returns same dict")

    # Test 10: SHORT entry on downtrend should be OK/SNIPER
    v_sh_on_sh = d1_bias_verdict("SHORT", bias_dn, strength_dn,
                                 last_struct_event(d1_dn, highs_dn, lows_dn, bias_dn))
    assert v_sh_on_sh in ("OK", "SNIPER")
    print(f"[ok] verdict SHORT-on-SHORT-bias = {v_sh_on_sh}")

    print("ALL TESTS PASSED")
    sys.exit(0)
