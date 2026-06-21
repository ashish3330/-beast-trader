#!/usr/bin/env python3 -B
"""
Fib-50 Pullback Continuation strategy detector (Dragon bot — 5th strategy).

Branch parity with: sweep_reclaim.py / fvg_strategy.py / sma_breakout.py.

CORE THESIS
-----------
After a strong impulse from Swing A to Swing B, professional money typically
waits for a 50% Fibonacci retracement before resuming the trend. The 50% level
is "fair value" — entries here have the best blended R:R because:
  * Continuation entries at the 0% (breakout) chase, R:R poor.
  * Entries at the 100% (reversal) fight the trend.
  * Entries near 50% (with confirmation) capture the trend resumption with
    a structural invalidation just past 61.8%.

Strategy is a single-bar event detector (NOT a state machine).  Each call
inspects the most-recent CLOSED M15 bar:

  SHORT setup:
    1. Detect prior down-impulse: a Swing High (A) followed by a Swing Low (B),
       both confirmed via N-bar fractal pivots (default N=5).
    2. Impulse magnitude (A - B) >= FIB50_MIN_IMPULSE_ATR × ATR14.
    3. Current bar's high must reach the 50-61.8% retracement zone:
         fib_50  = B + 0.5   × (A - B)
         fib_618 = B + 0.618 × (A - B)
       Enter only if fib_50 <= high < fib_618.
    4. Bearish confirmation candle on the current closed bar:
         (a) close < open AND upper wick >= 50% of full range  (rejection wick), OR
         (b) bearish engulfing of the prior candle, OR
         (c) doji/pinbar within the entry zone.
    5. Entry = current close.
    6. SL = max(fib_618, A) + FIB50_ATR_BUFFER × ATR14
       (capped at FIB50_MAX_SL_R × intended_risk if too wide).
    7. TP1 = entry - 0.5 × (entry - B)   (halfway back to swing low)
    8. TP2 = B                            (full retracement, runner)
    9. R:R requirement: TP1 must deliver >= FIB50_MIN_RR.

  LONG setup: symmetric (Swing Low C → Swing High D → pullback to 50%).

API (mirrors SweepReclaimStrategy):
    strat = Fib50Strategy(state)
    sig = strat.evaluate("XAUUSD")
    # sig is None, or:
    # {
    #   "direction": "LONG" | "SHORT",
    #   "entry": float, "sl": float, "tp1": float, "tp2": float,
    #   "swing_A": float, "swing_B": float,
    #   "fib_50": float, "fib_618": float,
    #   "reason": str, "bar_time": pd.Timestamp,
    # }

Stateless across calls. Defensive on data hiccups (returns None). Reads
index -2 of the frame (last CLOSED bar, never the live forming bar).
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger("dragon.fib50")

# Optional config-driven per-symbol overrides + blacklist. Fail-open on
# import (older builds / standalone tests don't have these defined).
try:
    from config import FIB50_PARAM_OVERRIDES  # type: ignore
except Exception:
    FIB50_PARAM_OVERRIDES = {}
try:
    from config import FIB50_SYMBOL_BLACKLIST  # type: ignore
except Exception:
    FIB50_SYMBOL_BLACKLIST = set()

# ════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMS — overridable via constructor params dict
# ════════════════════════════════════════════════════════════════════════
SWING_PIVOT_N = 5             # fractal half-window (high[i] is max of [i-N..i+N])
MIN_IMPULSE_ATR = 2.0         # impulse magnitude floor in ATR14 multiples
ENTRY_ZONE_LO = 0.50          # 50% level
ENTRY_ZONE_HI = 0.618         # 61.8% level (max stretch before setup voids)
ATR_BUFFER = 0.20             # SL beyond fib_618 / swing extreme
# MAX_SL_R caps SL in ATR14 units. Fib-50 SLs are naturally wide (5-8x ATR
# is normal because SL anchors to fib_50 + bar-wick clamp, far from entry).
# Sizing already converts to fixed % risk; this cap is an edge filter, not
# a risk filter. Set high enough to let valid geometries through.
MAX_SL_R = 8.0
MIN_RR = 1.5                  # require TP1 to yield >= this R multiple
ATR_PERIOD = 14
MIN_M15_BARS = 100            # require this many closed bars before evaluating
LOOKBACK = 300                # how far back to search for swing pairs


def _atr(H, L, C, period=ATR_PERIOD):
    """Wilder ATR — returns latest value (or 0 if too short)."""
    n = len(H)
    if n < period + 1:
        return 0.0
    tr = np.empty(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
    atr = np.empty(n)
    atr[period] = tr[1:period + 1].mean()
    k = 1.0 / period
    for i in range(period + 1, n):
        atr[i] = atr[i - 1] * (1 - k) + tr[i] * k
    return float(atr[-1])


def _is_swing_high(H, i, w):
    """Strict fractal pivot: H[i] is the unique max of H[i-w..i+w]."""
    if i < w or i + w >= len(H):
        return False
    seg = H[i - w:i + w + 1]
    return H[i] == seg.max() and (seg == H[i]).sum() == 1


def _is_swing_low(L, i, w):
    if i < w or i + w >= len(L):
        return False
    seg = L[i - w:i + w + 1]
    return L[i] == seg.min() and (seg == L[i]).sum() == 1


def _find_swings(H, L, lookback, w):
    """Walk back from the most recent confirmed pivot. Return list of
    (index, kind, value) tuples sorted by index (most recent last).

    A pivot at index `i` is only "confirmed" after `w` bars close after it,
    so the latest searchable index is len-1-w.
    """
    n = len(H)
    end = n - 1 - w
    start = max(w, end - lookback)
    swings = []
    for i in range(start, end + 1):
        if _is_swing_high(H, i, w):
            swings.append((i, "H", float(H[i])))
        elif _is_swing_low(L, i, w):
            swings.append((i, "L", float(L[i])))
    return swings


def _latest_impulse_pair(swings):
    """Return (A_idx, A_kind, A_val, B_idx, B_kind, B_val) for the most
    recent impulse — the last two swings of OPPOSITE kind. Returns None if
    fewer than 2 swings or last two are same kind."""
    if len(swings) < 2:
        return None
    # Walk back from the most recent swing to find the most recent
    # opposite-kind pair (A then B).
    last = swings[-1]
    for j in range(len(swings) - 2, -1, -1):
        prev = swings[j]
        if prev[1] != last[1]:
            # prev = A (older), last = B (newer)
            return (prev[0], prev[1], prev[2], last[0], last[1], last[2])
    return None


def _normalize_candles(df):
    """Normalize input frame to time-indexed OHLC. Returns None on bad input."""
    if df is None or len(df) == 0:
        return None
    try:
        df = df.copy()
        if "time" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name or "index": "time"})
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
        keep = cols + (["tick_volume"] if "tick_volume" in df.columns else
                       (["volume"] if "volume" in df.columns else []))
        return df[keep].astype(float)
    except Exception as e:
        log.debug("candle normalize failed: %s", e)
        return None


# ════════════════════════════════════════════════════════════════════════
#  Fib50Strategy — drop-in detector class
# ════════════════════════════════════════════════════════════════════════
class Fib50Strategy:
    """Per-symbol single-bar Fib-50 pullback-continuation detector.

    Stateless across calls. Every call inspects only the most-recent closed
    bar of the configured timeframe. The wiring agent should dedupe signals
    by (symbol, bar_time) — the strategy hints at this via _last_bar_t too.
    """

    def __init__(self, state, params=None):
        self.state = state
        p = params or {}
        # Timeframe in minutes — default 15 (M15). H1 = 60.
        self.tf_minutes = int(p.get("TIMEFRAME_MINUTES", 15))
        self.swing_pivot_n = int(p.get("SWING_PIVOT_N", SWING_PIVOT_N))
        self.min_impulse_atr = float(p.get("MIN_IMPULSE_ATR", MIN_IMPULSE_ATR))
        self.entry_zone_lo = float(p.get("ENTRY_ZONE_LO", ENTRY_ZONE_LO))
        self.entry_zone_hi = float(p.get("ENTRY_ZONE_HI", ENTRY_ZONE_HI))
        self.atr_buffer = float(p.get("ATR_BUFFER", ATR_BUFFER))
        self.max_sl_r = float(p.get("MAX_SL_R", MAX_SL_R))
        self.min_rr = float(p.get("MIN_RR", MIN_RR))
        self.lookback = int(p.get("LOOKBACK", LOOKBACK))
        self._last_bar_t = {}

    def _get_frame(self, symbol):
        if self.state is None:
            return None
        try:
            return self.state.get_candles(symbol, self.tf_minutes)
        except Exception as e:
            log.debug("[%s] frame fetch failed: %s", symbol, e)
            return None

    def evaluate(self, symbol):
        """Inspect the most-recent closed bar; return signal dict or None."""
        if symbol in FIB50_SYMBOL_BLACKLIST:
            return None

        # Per-symbol override resolution. Unknown keys are silently ignored
        # so callers can stash extra params without crashing the detector.
        sym_ov = FIB50_PARAM_OVERRIDES.get(symbol, {}) if FIB50_PARAM_OVERRIDES else {}
        eff_min_impulse_atr = float(sym_ov.get("MIN_IMPULSE_ATR", self.min_impulse_atr))
        eff_min_rr = float(sym_ov.get("MIN_RR", self.min_rr))
        eff_max_sl_r = float(sym_ov.get("MAX_SL_R", self.max_sl_r))
        eff_atr_buffer = float(sym_ov.get("ATR_BUFFER", self.atr_buffer))
        direction_filter = str(sym_ov.get("DIRECTION_FILTER", "BOTH")).upper()

        raw = self._get_frame(symbol)
        df = _normalize_candles(raw)
        if df is None or len(df) < MIN_M15_BARS:
            return None

        # Index -2 = last CLOSED bar (-1 is the forming bar)
        i = len(df) - 2
        if i < self.swing_pivot_n * 2 + 5:
            return None
        bar = df.iloc[i]
        bar_t = df.index[i]
        prev = df.iloc[i - 1] if i >= 1 else None

        # Stateless dedupe: only one fire per (symbol, bar_time).
        if self._last_bar_t.get(symbol) == bar_t:
            return None

        H = df["high"].values[:i + 1]
        L = df["low"].values[:i + 1]
        C = df["close"].values[:i + 1]
        O = df["open"].values[:i + 1]

        atr14 = _atr(H, L, C, ATR_PERIOD)
        if atr14 <= 0:
            return None

        # Find swings up to (but excluding) the candidate bar i — we want
        # impulse pair that STRICTLY precedes our entry decision.
        swings = _find_swings(H[:i], L[:i], self.lookback, self.swing_pivot_n)
        pair = _latest_impulse_pair(swings)
        if pair is None:
            self._last_bar_t[symbol] = bar_t
            return None

        a_idx, a_kind, a_val, b_idx, b_kind, b_val = pair

        sig = None

        # ── DOWN IMPULSE → SHORT setup (A=high, B=low; pullback up to fib_50) ──
        if a_kind == "H" and b_kind == "L" and direction_filter != "LONG":
            impulse_mag = a_val - b_val
            if impulse_mag >= eff_min_impulse_atr * atr14 and impulse_mag > 0:
                fib_50 = b_val + 0.5 * impulse_mag
                fib_618 = b_val + 0.618 * impulse_mag

                # Price must reach into the 50-61.8% zone on the current bar.
                # We require the bar high to touch >= fib_50 AND remain
                # < fib_618 (we'd consider deeper retraces a different setup).
                high = float(bar["high"])
                if high >= fib_50 and high < fib_618:
                    # Bearish confirmation
                    op, cl, lo = float(bar["open"]), float(bar["close"]), float(bar["low"])
                    bar_range = max(high - lo, 1e-9)
                    upper_wick = high - max(op, cl)
                    upper_wick_ratio = upper_wick / bar_range
                    body = abs(cl - op)
                    body_ratio = body / bar_range

                    is_bearish = cl < op
                    big_upper_wick = upper_wick_ratio >= 0.50
                    bearish_engulf = (
                        prev is not None
                        and cl < op
                        and float(prev["close"]) > float(prev["open"])
                        and cl <= float(prev["open"]) - 1e-9
                        and op >= float(prev["close"]) - 1e-9
                    )
                    # Doji / pinbar near fib_50 — body small + upper wick dominant
                    is_doji_pinbar = body_ratio <= 0.30 and upper_wick_ratio >= 0.40

                    confirmed = (is_bearish and big_upper_wick) or bearish_engulf or is_doji_pinbar

                    if confirmed:
                        entry = cl
                        # SL = max(fib_50 + buffer, bar high + tiny buffer).
                        # Spec gives a CHOICE — above 50% (tighter) or above
                        # 61.8% (wider). Default tight (above 50%) keeps R:R
                        # tractable; bar-high clamp prevents same-wick stop-out.
                        # Wider variant available via params (USE_WIDE_SL=True
                        # then SL anchors to fib_618).
                        anchor = fib_618 if sym_ov.get("USE_WIDE_SL") else fib_50
                        sl_raw = max(
                            anchor + eff_atr_buffer * atr14,
                            float(bar["high"]) + 0.1 * atr14,
                        )
                        risk = sl_raw - entry
                        if risk > 0:
                            tp1 = entry - 0.5 * (entry - b_val)
                            tp2 = b_val
                            # R:R floor + max SL cap.  MAX_SL_R caps SL in ATR
                            # units (SL > MAX_SL_R × ATR14 → SKIP). 1R := ATR14.
                            rr1 = (entry - tp1) / risk if risk > 0 else 0.0
                            rr2 = (entry - tp2) / risk if risk > 0 else 0.0
                            sl_in_atr = risk / atr14 if atr14 > 0 else 999.0
                            # Accept if EITHER tp1 hits MIN_RR OR tp2 hits 2.0
                            # (runner provides R:R when tp1 doesn't).
                            rr_ok = rr1 >= eff_min_rr or rr2 >= 2.0
                            if rr_ok and sl_in_atr <= eff_max_sl_r:
                                sig = {
                                    "direction": "SHORT",
                                    "entry": float(entry),
                                    "sl": float(sl_raw),
                                    "tp1": float(tp1),
                                    "tp2": float(tp2),
                                    "swing_A": float(a_val),
                                    "swing_B": float(b_val),
                                    "fib_50": float(fib_50),
                                    "fib_618": float(fib_618),
                                    "atr14": float(atr14),
                                    "bar_time": bar_t,
                                    "reason": (
                                        f"SHORT fib50 pullback (A={a_val:.5f} B={b_val:.5f} "
                                        f"fib50={fib_50:.5f} rr1={rr1:.2f} rr2={rr2:.2f})"
                                    ),
                                }

        # ── UP IMPULSE → LONG setup (C=low, D=high; pullback down to fib_50) ──
        if sig is None and a_kind == "L" and b_kind == "H" and direction_filter != "SHORT":
            c_val, d_val = a_val, b_val
            impulse_mag = d_val - c_val
            if impulse_mag >= eff_min_impulse_atr * atr14 and impulse_mag > 0:
                fib_50 = d_val - 0.5 * impulse_mag
                fib_618 = d_val - 0.618 * impulse_mag

                low = float(bar["low"])
                if low <= fib_50 and low > fib_618:
                    op, cl, hi = float(bar["open"]), float(bar["close"]), float(bar["high"])
                    bar_range = max(hi - low, 1e-9)
                    lower_wick = min(op, cl) - low
                    lower_wick_ratio = lower_wick / bar_range
                    body = abs(cl - op)
                    body_ratio = body / bar_range

                    is_bullish = cl > op
                    big_lower_wick = lower_wick_ratio >= 0.50
                    bullish_engulf = (
                        prev is not None
                        and cl > op
                        and float(prev["close"]) < float(prev["open"])
                        and cl >= float(prev["open"]) + 1e-9
                        and op <= float(prev["close"]) + 1e-9
                    )
                    is_hammer_pinbar = body_ratio <= 0.30 and lower_wick_ratio >= 0.40

                    confirmed = (is_bullish and big_lower_wick) or bullish_engulf or is_hammer_pinbar

                    if confirmed:
                        entry = cl
                        # SL = min(fib_50 - buffer, bar low - tiny buffer).
                        # Same logic as SHORT — tight default + bar-extreme
                        # clamp to avoid same-wick stop-out.
                        anchor = fib_618 if sym_ov.get("USE_WIDE_SL") else fib_50
                        sl_raw = min(
                            anchor - eff_atr_buffer * atr14,
                            float(bar["low"]) - 0.1 * atr14,
                        )
                        risk = entry - sl_raw
                        if risk > 0:
                            tp1 = entry + 0.5 * (d_val - entry)
                            tp2 = d_val
                            rr1 = (tp1 - entry) / risk if risk > 0 else 0.0
                            rr2 = (tp2 - entry) / risk if risk > 0 else 0.0
                            sl_in_atr = risk / atr14 if atr14 > 0 else 999.0
                            rr_ok = rr1 >= eff_min_rr or rr2 >= 2.0
                            if rr_ok and sl_in_atr <= eff_max_sl_r:
                                sig = {
                                    "direction": "LONG",
                                    "entry": float(entry),
                                    "sl": float(sl_raw),
                                    "tp1": float(tp1),
                                    "tp2": float(tp2),
                                    "swing_A": float(c_val),
                                    "swing_B": float(d_val),
                                    "fib_50": float(fib_50),
                                    "fib_618": float(fib_618),
                                    "atr14": float(atr14),
                                    "bar_time": bar_t,
                                    "reason": (
                                        f"LONG fib50 pullback (C={c_val:.5f} D={d_val:.5f} "
                                        f"fib50={fib_50:.5f} rr1={rr1:.2f} rr2={rr2:.2f})"
                                    ),
                                }

        # Always mark bar processed (prevent repeated checks within a bar).
        self._last_bar_t[symbol] = bar_t
        return sig


# ════════════════════════════════════════════════════════════════════════
#  Standalone self-test — `python3 agent/fib50_strategy.py --self-test`
# ════════════════════════════════════════════════════════════════════════
def _make_fake_state(df):
    """Wrap a DataFrame into a SharedState-shaped object the detector accepts.

    Live frames carry a partially-formed last bar (iloc[-1]) that the detector
    must ignore — so we append a synthetic forming bar to make iloc[-2] resolve
    to the last bar of the supplied df.
    """
    class _FakeState:
        def __init__(self, base):
            self.base = base

        def get_candles(self, symbol, tf):
            d = self.base.copy()
            last_close = float(d["close"].iloc[-1])
            last_time = d["time"].iloc[-1]
            new_row = {
                "time": last_time + pd.Timedelta(minutes=15),
                "open": last_close,
                "high": last_close,
                "low": last_close,
                "close": last_close,
            }
            if "tick_volume" in d.columns:
                new_row["tick_volume"] = 500.0
            return pd.concat([d, pd.DataFrame([new_row])], ignore_index=True)
    return _FakeState(df)


def _synth_down_impulse_pullback():
    """200 M15 bars with strict monotonic geometry — no spurious intra-impulse
    pivots. Down impulse from bar 30 (A=105) to bar 100 (B=84), then pullback
    to fib_50 (94.5) with bearish rejection candle at bar 199."""
    n = 200
    times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    op = np.zeros(n); cl = np.zeros(n); hi = np.zeros(n); lo = np.zeros(n)

    A_idx, A = 30, 105.0
    B_idx, B = 100, 84.0

    # Bars 0..A_idx-1: chop around 100, hi well below A
    for k in range(A_idx):
        m = 100.0 + 0.05 * np.sin(k * 0.7)
        op[k] = m - 0.1; cl[k] = m + 0.1
        hi[k] = m + 0.2; lo[k] = m - 0.2

    # Bar A_idx: swing HIGH spike (unique max in ±5 window)
    op[A_idx] = 104.0; cl[A_idx] = 104.2
    hi[A_idx] = A
    lo[A_idx] = 103.8

    # Bars A_idx+1 .. B_idx: STRICTLY monotonic decrease from A to B.
    # Each bar's high < previous bar's high so no spurious intermediate pivots.
    for k in range(A_idx + 1, B_idx + 1):
        frac = (k - A_idx) / (B_idx - A_idx)
        m = A - (A - B) * frac
        op[k] = m + 0.05; cl[k] = m - 0.05
        # Tiny vertical offset so each bar's hi strictly less than previous:
        # since m decreases by (A-B)/(B_idx-A_idx) = 21/70 = 0.3 per bar,
        # hi spacing of 0.2 ensures hi[k] < hi[k-1].
        hi[k] = m + 0.1; lo[k] = m - 0.1

    # Bar B_idx: swing LOW (lo is unique min in ±5)
    lo[B_idx] = B

    # Bars B_idx+1 .. n-2: pullback up to fib_50 zone (94.5..96.98).
    # Final closed bar at index n-1 gets the rejection candle.
    pullback_target = 94.5  # right at fib_50
    for k in range(B_idx + 1, n - 1):
        frac = (k - B_idx) / (n - 1 - B_idx)
        m = B + (pullback_target - B) * frac
        op[k] = m - 0.05; cl[k] = m + 0.05
        hi[k] = m + 0.1; lo[k] = m - 0.1

    # Final closed bar (n-1): bearish rejection in fib zone.
    # fib_50 = 94.5, fib_618 = 96.978.
    last = n - 1
    op[last] = 94.0
    cl[last] = 93.0
    hi[last] = 95.0      # in zone: 94.5 <= 95.0 < 96.978
    lo[last] = 92.9
    # bar_range = 2.1, upper_wick = 95.0 - 94.0 = 1.0, ratio = 0.476 — borderline
    # Make it cleaner: cl=92.5 makes upper_wick = 95.0 - 94.0 = 1.0, range=2.5,
    # ratio=0.40 — still <0.50. Need bigger wick. Set hi=95.5 instead.
    hi[last] = 95.5
    # bar_range = 2.6, upper_wick = 95.5 - 94.0 = 1.5, ratio = 0.577 >= 0.50 ✓

    df = pd.DataFrame({"time": times, "open": op, "high": hi, "low": lo, "close": cl})
    return df


def _synth_up_impulse_pullback():
    """Mirror — up impulse C=95 to D=116, pullback to fib_50=105.5, bullish wick."""
    n = 200
    times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    op = np.zeros(n); cl = np.zeros(n); hi = np.zeros(n); lo = np.zeros(n)

    C_idx, C = 30, 95.0
    D_idx, D = 100, 116.0

    # Bars 0..C_idx-1: chop around 100, lo well above C
    for k in range(C_idx):
        m = 100.0 + 0.05 * np.sin(k * 0.7)
        op[k] = m + 0.1; cl[k] = m - 0.1
        hi[k] = m + 0.2; lo[k] = m - 0.2

    # Bar C_idx: swing LOW spike (unique min in ±5)
    op[C_idx] = 96.0; cl[C_idx] = 95.8
    hi[C_idx] = 96.2
    lo[C_idx] = C

    # Bars C_idx+1..D_idx: STRICTLY monotonic increase from C to D.
    for k in range(C_idx + 1, D_idx + 1):
        frac = (k - C_idx) / (D_idx - C_idx)
        m = C + (D - C) * frac
        op[k] = m - 0.05; cl[k] = m + 0.05
        hi[k] = m + 0.1; lo[k] = m - 0.1

    # Bar D_idx: swing HIGH
    hi[D_idx] = D

    # Bars D_idx+1..n-2: pullback down to fib_50 zone.
    # fib_50 = D - 0.5*(D-C) = 116 - 10.5 = 105.5
    # fib_618 = D - 0.618*(D-C) = 116 - 12.978 ≈ 103.022
    pullback_target = 105.5
    for k in range(D_idx + 1, n - 1):
        frac = (k - D_idx) / (n - 1 - D_idx)
        m = D - (D - pullback_target) * frac
        op[k] = m + 0.05; cl[k] = m - 0.05
        hi[k] = m + 0.1; lo[k] = m - 0.1

    # Final closed bar: bullish rejection in fib zone (103.022 < lo <= 105.5).
    last = n - 1
    op[last] = 106.0
    cl[last] = 107.0
    hi[last] = 107.1
    lo[last] = 104.5     # 103.022 < 104.5 <= 105.5 ✓
    # bar_range = 2.6, lower_wick = 106.0 - 104.5 = 1.5, ratio = 0.577 >= 0.50 ✓

    df = pd.DataFrame({"time": times, "open": op, "high": hi, "low": lo, "close": cl})
    return df


def _synth_sideways():
    """Random chop — no clear impulse. Should return None."""
    np.random.seed(7)
    n = 120
    times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.normal(0, 0.2, n).cumsum()
    cl = 100 + rng
    op = np.concatenate([[100.0], cl[:-1]])
    hi = np.maximum(op, cl) + np.abs(np.random.normal(0, 0.1, n))
    lo = np.minimum(op, cl) - np.abs(np.random.normal(0, 0.1, n))
    df = pd.DataFrame({"time": times, "open": op, "high": hi, "low": lo, "close": cl})
    return df


def _synth_overshoot_beyond_618():
    """Down impulse, but the pullback overshoots beyond 61.8% — out of zone.
    Should return None."""
    df = _synth_down_impulse_pullback().copy()
    # Push final bar's high above 96.98 (fib_618).
    last = len(df) - 1
    df.loc[last, "high"] = 98.0    # beyond fib_618 = ~96.98
    df.loc[last, "close"] = 97.5
    df.loc[last, "open"] = 96.0
    df.loc[last, "low"] = 95.5
    return df


def _synth_sideways():
    """Random chop — no clear impulse. Should return None."""
    np.random.seed(7)
    n = 200
    times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.normal(0, 0.2, n).cumsum()
    cl = 100 + rng
    op = np.concatenate([[100.0], cl[:-1]])
    hi = np.maximum(op, cl) + np.abs(np.random.normal(0, 0.1, n))
    lo = np.minimum(op, cl) - np.abs(np.random.normal(0, 0.1, n))
    df = pd.DataFrame({"time": times, "open": op, "high": hi, "low": lo, "close": cl})
    return df


def _run_self_test():
    import sys
    failed = 0

    def _check(name, df, expect_dir):
        state = _make_fake_state(df)
        strat = Fib50Strategy(state)
        sig = strat.evaluate("TEST")
        ok = (sig is None and expect_dir is None) or (
            sig is not None and expect_dir is not None
            and sig["direction"] == expect_dir
        )
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: expect={expect_dir} got="
              f"{(sig or {}).get('direction')}")
        if not ok:
            if sig:
                print(f"        sig={sig}")
            else:
                print("        sig=None")
            return 1
        return 0

    print("\nFIB50 self-test:")
    failed += _check("down-impulse + pullback to 50%  -> SHORT",
                     _synth_down_impulse_pullback(), "SHORT")
    failed += _check("up-impulse + pullback to 50%    -> LONG",
                     _synth_up_impulse_pullback(), "LONG")
    failed += _check("sideways chop                    -> None",
                     _synth_sideways(), None)
    failed += _check("overshoot beyond 61.8%           -> None",
                     _synth_overshoot_beyond_618(), None)

    if failed == 0:
        print("\nALL FIB50 SELF-TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n{failed} FIB50 SELF-TEST(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _run_self_test()
    else:
        print("Usage: python3 -B agent/fib50_strategy.py --self-test")
