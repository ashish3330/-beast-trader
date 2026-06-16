#!/usr/bin/env python3 -B
"""
SetupInvalidator — per-setup-type pre-defined invalidation watcher.
═══════════════════════════════════════════════════════════════════════════

PHILOSOPHY
──────────
Every open position carries a *setup_type* + *invalidation_spec* written at
entry time. Each brain cycle, we watch H1 CLOSED bars: if the structural
condition that made the entry valid has been violated, CLOSE the position
at market with reason ``INVALID_<SETUP>``.

The component is the post-entry counterpart to ICT_NO_SWEEP (Gate 3f in
brain.py): same structural philosophy, opposite side of the trade lifecycle.
It runs ADJACENT to ``manage_trailing_sl`` but BEFORE ``ExitIntelligence``
and ``EarlyLossCut`` so a hard-structural-fail takes precedence over
trail/peak logic.

SETUP TAXONOMY (literature refs)
────────────────────────────────
    WYCKOFF_SPRING (Wyckoff 1931; Weis 2013)
        entry   : LONG after sweep of accumulation low + reclaim
        invalid : H1 close < spring_low (= entry_swept_level)
        time    : 6 H1 bars (~1 trading session)

    WYCKOFF_UPTHRUST (mirror)
        invalid : H1 close > upthrust_high
        time    : 6

    ORDER_BLOCK_LONG / ORDER_BLOCK_SHORT (ICT — Huddleston 2017)
        LONG  invalid : H1 close < ob_low  - 0.10 * ATR14_H1
        SHORT invalid : H1 close > ob_high + 0.10 * ATR14_H1
        time          : 4

    FVG_RETEST (existing agent/fvg_strategy.py)
        LONG  invalid : H1 close < fvg_bot - 0.10 * ATR
        SHORT invalid : H1 close > fvg_top + 0.10 * ATR
        time          : per FVG_TIME_STOP_HOURS_PER_SYMBOL (default 6)

    BREAKOUT (Darvas 1960; Williams 1979)
        invalid : no follow-through in N=3 H1 bars
            LONG  follow-through := high >= breakout_level + 0.25*ATR14_H1
            SHORT follow-through := low  <= breakout_level - 0.25*ATR14_H1
        time    : 3 (== the same N)

    SWEEP_RECLAIM (existing agent/sweep_reclaim.py)
        invalid : H1 close beyond swept_level by > 0.10 * ATR (sweep failed)
        time    : 6

    MOMENTUM_CONTINUATION (legacy default)
        no structural invalidation level → time-only
        time    : 8

DATA MODEL
──────────
At entry, brain writes to ``entry_metadata[sym]``::

    {
        "setup_type":         "WYCKOFF_SPRING" | "ORDER_BLOCK_LONG" | ...,
        "invalidation_level": float | None,    # structural price
        "atr_h1_entry":       float,           # ATR14 at entry, for buffer
        "entry_h1_bar_t":     int,             # epoch sec of entry H1 bar
        "time_inval_bars":    int,             # per-setup default (override)
        "progress_anchor_px": float,           # entry price
        "min_progress_r":     float,           # default 0.3R
        "direction":          "LONG" | "SHORT",
        "stop_dist":          float,           # 1R in price units
        # populated by us:
        "last_invcheck_t":    int,
    }

DESIGN: pure module-level functions only — no class state, no MT5, no
journal-writing side effects. Calling brain code is responsible for invoking
``executor.close_position`` and writing journal rows. This module merely
returns ``InvalidationDecision`` dicts:

    {
        "should_close": bool,
        "reason":       str,            # e.g. "INVALID_WYCKOFF_SPRING_STRUCT"
        "kind":         str,            # "struct" | "breakout_no_ft" | "time"
        "level":        float | None,   # the broken level (if struct)
        "close_px":     float | None,   # the H1 close that breached
        "bars_since":   int,            # bars since entry H1 bar
    }

Or ``None`` if no action.

Defensive: every public function fail-OPEN (returns ``None``) on any data
error. The caller never crashes because of this watcher.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# ── Constants (mirrored in config.py for hot-tuning) ─────────────────────
INVAL_STRUCT_BUFFER_ATR_DEFAULT = 0.10
BREAKOUT_FT_BARS_DEFAULT        = 3
BREAKOUT_FT_ATR_MULT_DEFAULT    = 0.25
DEFAULT_MIN_PROGRESS_R          = 0.30

# Per-setup-type H1 time-stop in bars. Mirrors config.SETUP_TIME_INVAL_BARS.
SETUP_TIME_INVAL_BARS_DEFAULT: Dict[str, int] = {
    "WYCKOFF_SPRING":        6,
    "WYCKOFF_UPTHRUST":      6,
    "ORDER_BLOCK_LONG":      4,
    "ORDER_BLOCK_SHORT":     4,
    "FVG_RETEST":            6,
    "BREAKOUT":              3,
    "SWEEP_RECLAIM":         6,
    "MOMENTUM_CONTINUATION": 8,
}

LONG_SETUPS = {
    "WYCKOFF_SPRING", "ORDER_BLOCK_LONG", "FVG_RETEST",
    "BREAKOUT", "SWEEP_RECLAIM", "MOMENTUM_CONTINUATION",
}
SHORT_SETUPS = {
    "WYCKOFF_UPTHRUST", "ORDER_BLOCK_SHORT",
}


# ─────────────────────────────────────────────────────────────────────────
#  Helpers — pure data manipulation
# ─────────────────────────────────────────────────────────────────────────
def _row(h1: Any, idx: int) -> Optional[Mapping[str, Any]]:
    """Read an H1 row at index ``idx`` (negative supported), framework-agnostic.

    Accepts pandas DataFrames (uses ``.iloc[idx]``) or plain lists of dicts.
    Returns ``None`` on any error.
    """
    if h1 is None:
        return None
    try:
        # pandas DataFrame
        if hasattr(h1, "iloc"):
            return h1.iloc[idx]
        # list-like of dicts
        return h1[idx]
    except Exception:  # noqa: BLE001 — defensive
        return None


def _ohlc(row: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float, int]]:
    """Return ``(open, high, low, close, time)`` as floats/ints, or ``None``."""
    try:
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        t = int(row["time"])
        return o, h, l, c, t
    except Exception:  # noqa: BLE001
        return None


def _len(h1: Any) -> int:
    try:
        return int(len(h1))
    except Exception:
        return 0


def bars_since_entry_h1(meta: Mapping[str, Any], h1: Any) -> int:
    """Count CLOSED H1 bars whose time > entry_h1_bar_t.

    Returns 0 if metadata is incomplete or the H1 buffer is shorter than 2.
    """
    entry_t = int(meta.get("entry_h1_bar_t", 0) or 0)
    if entry_t <= 0:
        return 0
    n = _len(h1)
    if n < 2:
        return 0
    # iterate the closed portion only (last bar may still be forming)
    count = 0
    # We use range from n-2 downwards so most-recent-closed first → break early.
    for i in range(n - 2, -1, -1):
        row = _row(h1, i)
        if row is None:
            break
        try:
            t = int(row["time"])
        except Exception:
            break
        if t > entry_t:
            count += 1
        else:
            break
    return count


# ─────────────────────────────────────────────────────────────────────────
#  Building invalidation specs at ENTRY time
# ─────────────────────────────────────────────────────────────────────────
def build_invalidation_spec(
    *,
    setup_type: str,
    direction: str,
    entry_price: float,
    stop_dist: float,
    atr_h1_entry: float,
    entry_h1_bar_t: int,
    invalidation_level: Optional[float] = None,
    time_inval_bars: Optional[int] = None,
    min_progress_r: Optional[float] = None,
    per_symbol_override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose the ``entry_metadata[sym]`` payload at entry time.

    Brain.py is expected to call this once per fill and merge the dict into
    its existing ``self._entry_metadata[sym]``. This guarantees the field
    schema the rest of this module relies on.

    ``per_symbol_override`` accepts ``{"time_inval_bars": int,
    "min_progress_r": float}`` from
    ``SETUP_INVALIDATOR_PER_SYMBOL_OVERRIDES[symbol]``.
    """
    setup_type = str(setup_type or "MOMENTUM_CONTINUATION").upper()
    direction  = str(direction or "LONG").upper()

    # Time-stop default (per-setup) — caller may override.
    t_default = SETUP_TIME_INVAL_BARS_DEFAULT.get(setup_type, 8)
    if time_inval_bars is None:
        time_inval_bars = t_default
    if min_progress_r is None:
        min_progress_r = DEFAULT_MIN_PROGRESS_R

    # Per-symbol override has the final word.
    if per_symbol_override:
        if "time_inval_bars" in per_symbol_override:
            try:
                time_inval_bars = int(per_symbol_override["time_inval_bars"])
            except Exception:
                pass
        if "min_progress_r" in per_symbol_override:
            try:
                min_progress_r = float(per_symbol_override["min_progress_r"])
            except Exception:
                pass

    return {
        "setup_type":         setup_type,
        "invalidation_level": (
            float(invalidation_level) if invalidation_level is not None else None
        ),
        "atr_h1_entry":       float(atr_h1_entry) if atr_h1_entry else 0.0,
        "entry_h1_bar_t":     int(entry_h1_bar_t) if entry_h1_bar_t else 0,
        "time_inval_bars":    int(time_inval_bars),
        "progress_anchor_px": float(entry_price),
        "min_progress_r":     float(min_progress_r),
        "direction":          direction,
        "stop_dist":          float(stop_dist) if stop_dist else 0.0,
        "last_invcheck_t":    0,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Per-position evaluation (the hot path)
# ─────────────────────────────────────────────────────────────────────────
def evaluate_invalidation_for_position(
    *,
    meta: Mapping[str, Any],
    h1: Any,
    peak_profit_r: Optional[float] = None,
    inval_buffer_atr: float = INVAL_STRUCT_BUFFER_ATR_DEFAULT,
    breakout_ft_bars: int = BREAKOUT_FT_BARS_DEFAULT,
    breakout_ft_atr_mult: float = BREAKOUT_FT_ATR_MULT_DEFAULT,
) -> Optional[Dict[str, Any]]:
    """Decide if a single position should be closed.

    Returns ``None`` if no action. Otherwise a decision dict (see module
    docstring). Caller mutates ``meta["last_invcheck_t"]`` itself after a
    decision — we do NOT mutate the input mapping here so the function
    stays pure & repeatable in tests.

    Precedence:
      1) STRUCTURAL break (highest)
      2) BREAKOUT no-follow-through
      3) TIME-based no-progress (lowest)
    """
    if not isinstance(meta, Mapping):
        return None
    setup_type = str(meta.get("setup_type") or "").upper()
    if not setup_type:
        return None
    direction = str(meta.get("direction") or "LONG").upper()

    n = _len(h1)
    if n < 3:
        return None
    last_closed = _row(h1, -2)
    if last_closed is None:
        return None
    parsed = _ohlc(last_closed)
    if parsed is None:
        return None
    _o, _h, _l, close, t_now = parsed

    # Caller may dedupe via last_invcheck_t — we still defend here too.
    last_t = int(meta.get("last_invcheck_t", 0) or 0)
    if t_now <= last_t:
        return None

    atr_e = float(meta.get("atr_h1_entry", 0.0) or 0.0)

    # ─── (1) STRUCTURAL price invalidation ──────────────────────────────
    lvl = meta.get("invalidation_level")
    if lvl is not None:
        try:
            lvl_f = float(lvl)
            buf   = float(inval_buffer_atr) * atr_e
            broken = (
                (direction == "LONG"  and close < lvl_f - buf) or
                (direction == "SHORT" and close > lvl_f + buf)
            )
            if broken:
                return {
                    "should_close": True,
                    "reason":       f"INVALID_{setup_type}_STRUCT",
                    "kind":         "struct",
                    "level":        lvl_f,
                    "close_px":     close,
                    "bars_since":   bars_since_entry_h1(meta, h1),
                    "t_now":        t_now,
                }
        except Exception:
            pass  # fall through to other checks on malformed level

    # ─── (2) BREAKOUT follow-through ────────────────────────────────────
    if setup_type.startswith("BREAKOUT"):
        bs = bars_since_entry_h1(meta, h1)
        N  = max(1, int(breakout_ft_bars))
        if bs >= N:
            try:
                anchor = float(meta.get("progress_anchor_px") or 0.0)
                if anchor > 0 and atr_e > 0:
                    sign = 1 if direction == "LONG" else -1
                    ft_required = anchor + sign * float(breakout_ft_atr_mult) * atr_e
                    # Pull last-N CLOSED bars (exclude the still-forming live bar).
                    start = n - 1 - N      # inclusive
                    end   = n - 1          # exclusive (drops the live bar)
                    if start < 0:
                        start = 0
                    last_high = None
                    last_low  = None
                    for i in range(start, end):
                        row = _row(h1, i)
                        if row is None:
                            continue
                        try:
                            hi = float(row["high"])
                            lo = float(row["low"])
                        except Exception:
                            continue
                        if last_high is None or hi > last_high:
                            last_high = hi
                        if last_low is None or lo < last_low:
                            last_low = lo
                    if last_high is not None and last_low is not None:
                        ok = (
                            (direction == "LONG"  and last_high >= ft_required) or
                            (direction == "SHORT" and last_low  <= ft_required)
                        )
                        if not ok:
                            return {
                                "should_close": True,
                                "reason":       "INVALID_BREAKOUT_NO_FT",
                                "kind":         "breakout_no_ft",
                                "level":        ft_required,
                                "close_px":     close,
                                "bars_since":   bs,
                                "t_now":        t_now,
                            }
            except Exception:
                pass

    # ─── (3) TIME-based no-progress invalidation ────────────────────────
    try:
        bs = bars_since_entry_h1(meta, h1)
        t_lim = int(meta.get("time_inval_bars", 0) or 0)
        if t_lim > 0 and bs >= t_lim:
            min_pr = float(meta.get("min_progress_r", DEFAULT_MIN_PROGRESS_R) or 0.0)
            mfe = float(peak_profit_r) if peak_profit_r is not None else 0.0
            if mfe < min_pr:
                return {
                    "should_close": True,
                    "reason":       f"INVALID_{setup_type}_TIME",
                    "kind":         "time",
                    "level":        None,
                    "close_px":     close,
                    "bars_since":   bs,
                    "t_now":        t_now,
                    "mfe_r":        mfe,
                }
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────
#  Setup-type derivation from momentum gate trail
# ─────────────────────────────────────────────────────────────────────────
def derive_momentum_setup_type(
    *,
    ict_sweep_detected: bool,
    bias_direction: str,
    range_extreme_exit: bool,
) -> str:
    """Map a momentum-strategy gate trail to a setup_type tag.

    Per spec: ICT sweep + bias trend → WYCKOFF_SPRING/UPTHRUST; pure breakout
    (no sweep, range_extreme exit) → BREAKOUT; else MOMENTUM_CONTINUATION.
    """
    bd = str(bias_direction or "").upper()
    if ict_sweep_detected:
        if bd in ("LONG", "BULL", "BULLISH", "UP"):
            return "WYCKOFF_SPRING"
        if bd in ("SHORT", "BEAR", "BEARISH", "DOWN"):
            return "WYCKOFF_UPTHRUST"
        # Side ambiguous → still tag as continuation rather than wyckoff.
        return "MOMENTUM_CONTINUATION"
    if range_extreme_exit:
        return "BREAKOUT"
    return "MOMENTUM_CONTINUATION"


# ─────────────────────────────────────────────────────────────────────────
#  Portfolio-level orchestration (brain calls this)
# ─────────────────────────────────────────────────────────────────────────
def evaluate_invalidations(
    *,
    positions: Mapping[str, Any],
    entry_metadata: Dict[str, Dict[str, Any]],
    get_candles,                       # callable: get_candles(sym, "H1") -> df
    watched_magics: Optional[Iterable[str]] = None,
    exclude_setups: Optional[Iterable[str]] = None,
    check_interval_sec: int = 60,
    inval_buffer_atr: float = INVAL_STRUCT_BUFFER_ATR_DEFAULT,
    breakout_ft_bars: int = BREAKOUT_FT_BARS_DEFAULT,
    breakout_ft_atr_mult: float = BREAKOUT_FT_ATR_MULT_DEFAULT,
    peak_profit_r_lookup=None,         # callable: f(symbol) -> float | None
    now_sec: Optional[int] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Loop all open positions, return a list of ``(symbol, decision)``.

    Pure orchestration — does NOT close positions or write the journal. The
    brain wraps the result with ``executor.close_position`` + journal calls.

    ``positions`` mapping shape: ``{symbol: position_obj}`` where
    ``position_obj.opened_via`` is a magic-tag string ('momentum', 'fvg', …).
    If your position object stores it differently, pass a thin adapter.

    Mutates ``entry_metadata[sym]["last_invcheck_t"]`` to dedupe per H1 bar.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    if not positions:
        return out

    watched = set(watched_magics or ("momentum", "fvg", "sr"))
    excludes = set(exclude_setups or ())

    for sym, pos in positions.items():
        try:
            meta = entry_metadata.get(sym) if isinstance(entry_metadata, dict) else None
            if not meta or "setup_type" not in meta:
                continue
            opened_via = getattr(pos, "opened_via", None) or (
                pos.get("opened_via") if isinstance(pos, Mapping) else None
            )
            if opened_via is not None and str(opened_via) not in watched:
                continue
            st = str(meta.get("setup_type") or "").upper()
            if not st or st in excludes:
                continue

            # check_interval_sec debounce — caller may set last_invcheck_t to the
            # last wall-clock probe; we re-use it as a coarse cooldown too.
            if now_sec is not None and check_interval_sec > 0:
                last = int(meta.get("last_invcheck_t", 0) or 0)
                if last and (int(now_sec) - last) < int(check_interval_sec):
                    # Allow H1-bar dedupe still — but skip the heavy candle fetch.
                    continue

            try:
                h1 = get_candles(sym, "H1")
            except Exception:
                h1 = None
            if h1 is None or _len(h1) < 3:
                continue

            peak_r = None
            if peak_profit_r_lookup is not None:
                try:
                    peak_r = peak_profit_r_lookup(sym)
                except Exception:
                    peak_r = None

            decision = evaluate_invalidation_for_position(
                meta=meta,
                h1=h1,
                peak_profit_r=peak_r,
                inval_buffer_atr=inval_buffer_atr,
                breakout_ft_bars=breakout_ft_bars,
                breakout_ft_atr_mult=breakout_ft_atr_mult,
            )
            if decision and decision.get("should_close"):
                # Mark to dedupe — both H1 bar time + wall clock.
                t_now = int(decision.get("t_now", 0) or 0)
                if t_now > 0:
                    meta["last_invcheck_t"] = t_now
                elif now_sec is not None:
                    meta["last_invcheck_t"] = int(now_sec)
                out.append((sym, decision))
        except Exception:
            # Per-symbol fail-open — never break the loop.
            continue

    return out


# ─────────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import sys
    import time

    failures: List[str] = []

    def _assert(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"  OK  {name}")
        else:
            failures.append(f"{name}: {detail}")
            print(f"  FAIL {name}: {detail}")

    def make_h1(bars: List[Tuple[int, float, float, float, float]]) -> List[Dict[str, Any]]:
        """bars = list of (time, open, high, low, close)."""
        return [
            {"time": t, "open": o, "high": h, "low": l, "close": c}
            for (t, o, h, l, c) in bars
        ]

    print("SetupInvalidator self-test")

    # ── Test 1: WYCKOFF_SPRING — long, structural break ─────────────────
    # Spring low @ 100.00, ATR = 1.0, buffer = 0.10. Close < 100 - 0.1 = 99.9.
    entry_t = 1_000_000
    bars = [
        (entry_t - 3600,    100.5, 101.0, 100.0, 100.8),
        (entry_t,           100.8, 101.2, 100.3, 101.0),  # entry bar
        (entry_t + 3600,    101.0, 101.1,  99.7,  99.6),  # last CLOSED — break!
        (entry_t + 7200,     99.6,  99.8,  99.4,  99.7),  # still-forming
    ]
    h1 = make_h1(bars)
    meta = build_invalidation_spec(
        setup_type="WYCKOFF_SPRING", direction="LONG",
        entry_price=101.0, stop_dist=1.0, atr_h1_entry=1.0,
        entry_h1_bar_t=entry_t, invalidation_level=100.0,
    )
    d = evaluate_invalidation_for_position(meta=meta, h1=h1, peak_profit_r=0.0)
    _assert("T1 wyckoff_spring struct break",
            d is not None and d["kind"] == "struct"
            and d["reason"] == "INVALID_WYCKOFF_SPRING_STRUCT",
            detail=str(d))

    # ── Test 2: WYCKOFF_SPRING — close >= level - buf → no action ───────
    bars2 = [
        (entry_t - 3600,    100.5, 101.0, 100.0, 100.8),
        (entry_t,           100.8, 101.2, 100.3, 101.0),
        (entry_t + 3600,    101.0, 101.1,  99.85, 99.95),  # close > 99.9
        (entry_t + 7200,     99.95, 100.0, 99.9, 99.97),
    ]
    meta2 = build_invalidation_spec(
        setup_type="WYCKOFF_SPRING", direction="LONG",
        entry_price=101.0, stop_dist=1.0, atr_h1_entry=1.0,
        entry_h1_bar_t=entry_t, invalidation_level=100.0,
    )
    d2 = evaluate_invalidation_for_position(meta=meta2, h1=make_h1(bars2), peak_profit_r=0.0)
    _assert("T2 wyckoff inside buffer → no close", d2 is None, detail=str(d2))

    # ── Test 3: WYCKOFF_UPTHRUST — short side struct break ──────────────
    bars3 = [
        (entry_t - 3600,    100.0, 100.5,  99.5,  99.8),
        (entry_t,            99.8, 100.0,  99.0,  99.2),
        (entry_t + 3600,     99.2,  100.5, 99.2,  100.5),  # break (close > 100.1)
        (entry_t + 7200,     100.5, 100.6, 100.3, 100.4),
    ]
    meta3 = build_invalidation_spec(
        setup_type="WYCKOFF_UPTHRUST", direction="SHORT",
        entry_price=99.2, stop_dist=1.0, atr_h1_entry=1.0,
        entry_h1_bar_t=entry_t, invalidation_level=100.0,
    )
    d3 = evaluate_invalidation_for_position(meta=meta3, h1=make_h1(bars3), peak_profit_r=0.0)
    _assert("T3 upthrust short struct break",
            d3 is not None and d3["reason"] == "INVALID_WYCKOFF_UPTHRUST_STRUCT",
            detail=str(d3))

    # ── Test 4: ORDER_BLOCK_LONG — close < ob_low - 0.10*ATR ────────────
    meta4 = build_invalidation_spec(
        setup_type="ORDER_BLOCK_LONG", direction="LONG",
        entry_price=2000.0, stop_dist=10.0, atr_h1_entry=5.0,
        entry_h1_bar_t=entry_t, invalidation_level=1995.0,
    )
    # buffer = 0.10*5 = 0.5. Trigger when close < 1994.5.
    bars4 = [
        (entry_t - 3600, 2001, 2002, 2000, 2000.5),
        (entry_t,        2000, 2001, 1999, 2000),
        (entry_t + 3600, 2000, 2000, 1993, 1994.0),   # break
        (entry_t + 7200, 1994, 1995, 1992, 1993),
    ]
    d4 = evaluate_invalidation_for_position(meta=meta4, h1=make_h1(bars4), peak_profit_r=0.0)
    _assert("T4 order_block_long struct break",
            d4 is not None and d4["kind"] == "struct"
            and d4["reason"] == "INVALID_ORDER_BLOCK_LONG_STRUCT",
            detail=str(d4))

    # ── Test 5: BREAKOUT no follow-through ──────────────────────────────
    # anchor 100, ATR 1.0, ft_required = 100 + 0.25 = 100.25 for LONG
    meta5 = build_invalidation_spec(
        setup_type="BREAKOUT", direction="LONG",
        entry_price=100.0, stop_dist=1.0, atr_h1_entry=1.0,
        entry_h1_bar_t=entry_t, invalidation_level=None,
    )
    bars5 = [
        (entry_t - 3600, 99.5,  100.0,  99.4,  99.8),
        (entry_t,        99.8,  100.05, 99.7,  100.0),     # entry
        (entry_t + 3600, 100.0, 100.10, 99.85, 100.05),    # closed 1
        (entry_t + 7200, 100.05,100.15, 99.95, 100.10),    # closed 2
        (entry_t +10800, 100.10,100.20, 99.95, 100.15),    # closed 3 (max high 100.20 < 100.25)
        (entry_t +14400, 100.15,100.20, 100.10, 100.15),   # still-forming
    ]
    d5 = evaluate_invalidation_for_position(meta=meta5, h1=make_h1(bars5), peak_profit_r=0.1)
    _assert("T5 breakout no follow-through",
            d5 is not None and d5["kind"] == "breakout_no_ft"
            and d5["reason"] == "INVALID_BREAKOUT_NO_FT",
            detail=str(d5))

    # ── Test 6: BREAKOUT WITH follow-through → no action ────────────────
    # Note: peak_r >= 0.3R, else the TIME branch (BREAKOUT default = 3 bars,
    # which equals bars_since=3) would still fire on no-progress.
    bars6 = list(bars5)
    bars6[-2] = (entry_t + 10800, 100.10, 100.30, 100.00, 100.28)  # high 100.30 >= 100.25
    meta6 = build_invalidation_spec(
        setup_type="BREAKOUT", direction="LONG",
        entry_price=100.0, stop_dist=1.0, atr_h1_entry=1.0,
        entry_h1_bar_t=entry_t, invalidation_level=None,
    )
    d6 = evaluate_invalidation_for_position(meta=meta6, h1=make_h1(bars6), peak_profit_r=0.5)
    _assert("T6 breakout WITH follow-through → no close", d6 is None, detail=str(d6))

    # ── Test 7: TIME-based no-progress invalidation ─────────────────────
    # MOMENTUM_CONTINUATION → time_inval_bars=8, peak_r=0.1 < 0.3 → fire.
    meta7 = build_invalidation_spec(
        setup_type="MOMENTUM_CONTINUATION", direction="LONG",
        entry_price=100.0, stop_dist=1.0, atr_h1_entry=1.0,
        entry_h1_bar_t=entry_t, invalidation_level=None,
    )
    bars7 = [(entry_t - 3600, 99.5, 100, 99.4, 99.8), (entry_t, 99.8, 100.05, 99.7, 100.0)]
    for i in range(1, 11):
        bars7.append((entry_t + 3600 * i, 100.0, 100.05, 99.95, 100.02))
    d7 = evaluate_invalidation_for_position(meta=meta7, h1=make_h1(bars7), peak_profit_r=0.1)
    _assert("T7 time no-progress fires",
            d7 is not None and d7["kind"] == "time"
            and d7["reason"] == "INVALID_MOMENTUM_CONTINUATION_TIME",
            detail=str(d7))

    # ── Test 8: TIME — peak_r >= min_progress_r → no action ─────────────
    meta8 = build_invalidation_spec(
        setup_type="MOMENTUM_CONTINUATION", direction="LONG",
        entry_price=100.0, stop_dist=1.0, atr_h1_entry=1.0,
        entry_h1_bar_t=entry_t, invalidation_level=None,
    )
    d8 = evaluate_invalidation_for_position(meta=meta8, h1=make_h1(bars7), peak_profit_r=0.5)
    _assert("T8 time WITH progress → no close", d8 is None, detail=str(d8))

    # ── Test 9: bars_since_entry_h1 ─────────────────────────────────────
    bs = bars_since_entry_h1({"entry_h1_bar_t": entry_t}, make_h1(bars7))
    # bars7 has entry bar at idx 1; last CLOSED is idx -2. Bars > entry_t (closed): 1..10 minus the last forming one.
    # Closed bars after entry: bars7[2..-2] = 9 bars (since bars7 length is 12 here).
    _assert("T9 bars_since_entry_h1 > 0", bs > 0, detail=f"bs={bs}, total bars={len(bars7)}")

    # ── Test 10: derive_momentum_setup_type ─────────────────────────────
    s = derive_momentum_setup_type(ict_sweep_detected=True, bias_direction="LONG", range_extreme_exit=False)
    _assert("T10a wyckoff_spring derive", s == "WYCKOFF_SPRING", detail=s)
    s = derive_momentum_setup_type(ict_sweep_detected=True, bias_direction="SHORT", range_extreme_exit=False)
    _assert("T10b wyckoff_upthrust derive", s == "WYCKOFF_UPTHRUST", detail=s)
    s = derive_momentum_setup_type(ict_sweep_detected=False, bias_direction="LONG", range_extreme_exit=True)
    _assert("T10c breakout derive", s == "BREAKOUT", detail=s)
    s = derive_momentum_setup_type(ict_sweep_detected=False, bias_direction="LONG", range_extreme_exit=False)
    _assert("T10d momentum_continuation derive", s == "MOMENTUM_CONTINUATION", detail=s)

    # ── Test 11: portfolio-level evaluate_invalidations ─────────────────
    class FakePos:
        def __init__(self, magic): self.opened_via = magic

    pos_map = {"XAUUSD": FakePos("momentum"), "SCALP_SYM": FakePos("scalp")}
    em = {
        "XAUUSD": build_invalidation_spec(
            setup_type="WYCKOFF_SPRING", direction="LONG",
            entry_price=101.0, stop_dist=1.0, atr_h1_entry=1.0,
            entry_h1_bar_t=entry_t, invalidation_level=100.0,
        ),
        "SCALP_SYM": build_invalidation_spec(
            setup_type="WYCKOFF_SPRING", direction="LONG",
            entry_price=101.0, stop_dist=1.0, atr_h1_entry=1.0,
            entry_h1_bar_t=entry_t, invalidation_level=100.0,
        ),
    }

    def fake_candles(sym, tf):
        return make_h1(bars)   # the bars from T1 that DO break

    decisions = evaluate_invalidations(
        positions=pos_map,
        entry_metadata=em,
        get_candles=fake_candles,
        watched_magics={"momentum", "fvg", "sr"},
        exclude_setups=set(),
        now_sec=int(time.time()),
    )
    _assert("T11 portfolio: 1 decision (momentum only, scalp excluded)",
            len(decisions) == 1 and decisions[0][0] == "XAUUSD",
            detail=str(decisions))

    # ── Test 12: SETUP_INVALIDATOR_EXCLUDE_SETUPS skips matching type ───
    decisions2 = evaluate_invalidations(
        positions=pos_map,
        entry_metadata=em,
        get_candles=fake_candles,
        watched_magics={"momentum", "fvg", "sr"},
        exclude_setups={"WYCKOFF_SPRING"},
        now_sec=int(time.time()) + 100,  # bump past check_interval
    )
    _assert("T12 exclude_setups skips it", len(decisions2) == 0, detail=str(decisions2))

    # ── Test 13: per-symbol override increases time_inval_bars ──────────
    over = {"time_inval_bars": 99, "min_progress_r": 0.9}
    meta13 = build_invalidation_spec(
        setup_type="MOMENTUM_CONTINUATION", direction="LONG",
        entry_price=100.0, stop_dist=1.0, atr_h1_entry=1.0,
        entry_h1_bar_t=entry_t, invalidation_level=None,
        per_symbol_override=over,
    )
    _assert("T13a override time_inval_bars",
            meta13["time_inval_bars"] == 99, detail=str(meta13))
    _assert("T13b override min_progress_r",
            abs(meta13["min_progress_r"] - 0.9) < 1e-9, detail=str(meta13))

    # ── Test 14: STRUCT precedence over TIME — both would fire ──────────
    bars14 = [(entry_t - 3600, 99.5, 100, 99.4, 99.8), (entry_t, 99.8, 100.05, 99.7, 100.0)]
    for i in range(1, 11):
        bars14.append((entry_t + 3600 * i, 99.5, 99.7, 99.0, 99.0))   # all below 99.9
    meta14 = build_invalidation_spec(
        setup_type="WYCKOFF_SPRING", direction="LONG",
        entry_price=100.0, stop_dist=1.0, atr_h1_entry=1.0,
        entry_h1_bar_t=entry_t, invalidation_level=100.0,
    )
    d14 = evaluate_invalidation_for_position(meta=meta14, h1=make_h1(bars14), peak_profit_r=0.0)
    _assert("T14 struct precedence over time",
            d14 is not None and d14["kind"] == "struct",
            detail=str(d14))

    # ── Test 15: malformed h1 → fail-open None ──────────────────────────
    d15 = evaluate_invalidation_for_position(
        meta=meta14, h1=[{"foo": 1}], peak_profit_r=0.0
    )
    _assert("T15 malformed h1 fails open (None)", d15 is None, detail=str(d15))

    # ── Summary ─────────────────────────────────────────────────────────
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll tests passed.")
    sys.exit(0)
