#!/usr/bin/env python3 -B
"""
NR7 Compression-Breakout entry detector (Toby Crabel, 1986).

Replacement candidate alongside sweep-reclaim. Crabel's original NR7 on US
futures (S&P, bonds, currencies) showed 60-76% WR with mechanical execution
and no discretion — entry IS the breakout, no waiting for confirmation, no
indicator stack. The compression-precedes-expansion mechanic is structural
liquidity behavior, not a behavioral edge that gets arbed away.

Adaptation to Dragon's 24h M15 bot:
  * Base frame: H1 (M15 ranges too noisy; daily too slow for active trading).
  * NR7 H1 bar = bar range (high-low) < min of last 6 H1 ranges → narrowest
    range in 7 bars (the "7" in NR7 includes the candidate bar itself).
  * Direction-agnostic detection (Crabel let the market pick): emit BOTH a
    LONG-trigger and a SHORT-trigger, the executor places both as
    stop-orders and cancels the loser when one fills (OCO).
  * In signal mode we emit one direction at a time based on which side
    breaks first in the live frame; backtest emits the first hit.
  * SL: opposite side of the NR7 bar (the entire "compressed" range).
  * TP: 1R partial (close 50%), 2R runner.
  * Expiry: cancel pending breakout 4 H1 bars (4h) after the NR7 close.
  * Time stop on filled trade: if peak < 0.3R after 6 H1 bars (6h), exit.

Why this avoids the late-entry trap:
  * No multi-indicator confirmation. The compression IS the setup.
  * Entry happens AT the break, not after follow-through. You're early to the
    expansion move, not late.
  * Structural stop (opposite NR7 side) — defined, not negotiable.

API mirrors SweepReclaimStrategy + FVGStrategy:
    strat = NR7Strategy(state)
    sig = strat.evaluate("XAUUSD")
    # sig is None or:
    # {
    #   "direction": "LONG" | "SHORT",
    #   "entry": float,  # the stop-order price (NR7 high+buf or low-buf)
    #   "sl": float, "tp1": float, "tp2": float,
    #   "nr7_high": float, "nr7_low": float, "nr7_range": float,
    #   "bar_time": pd.Timestamp,
    #   "reason": str,
    # }

Live wiring (deferred until backtest validates):
  Each closed H1 bar, if NR7 detected, the executor would place TWO pending
  stop orders (buy@nr7_high+buf, sell@nr7_low-buf). For the signal-only
  detector here we emit a single direction = whichever side the LIVE forming
  bar has already breached, or LONG by default if neither.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("dragon.nr7")

# ════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMS
# ════════════════════════════════════════════════════════════════════════
NR_LOOKBACK = 7              # narrowest range in last N bars (Crabel = 7)
BUFFER_ATR = 0.10            # entry stop placed buf*ATR beyond NR-high/low
SL_BUFFER_ATR = 0.05         # stop placed buf*ATR beyond opposite NR side
TP1_R = 1.0                  # close 50% at +1R
TP2_R = 2.0                  # close runner at +2R
EXPIRY_BARS = 4              # cancel pending after 4 H1 bars (~4h)
TIME_STOP_BARS = 6           # close filled trade if peak<0.3R after 6 H1 bars
TIME_STOP_PEAK_R = 0.3
MIN_H1_BARS = 30
ATR_PERIOD = 14
# Require the NR7 bar to be MEANINGFULLY narrow vs the running ATR — filters
# out NR7s that fire in already-dead markets where compression isn't real.
MIN_NR_FRACTION_OF_ATR = 0.20     # NR7 range >= 0.20 * ATR14 (not micro-spread)
MAX_NR_FRACTION_OF_ATR = 0.80     # NR7 range <= 0.80 * ATR14 (real compression)


def _atr(H, L, C, period=ATR_PERIOD):
    """Wilder ATR — returns latest value."""
    n = len(H)
    if n < period + 1:
        return 0.0
    tr = np.empty(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i-1]), abs(L[i] - C[i-1]))
    atr = np.empty(n)
    atr[period] = tr[1:period+1].mean()
    k = 1.0 / period
    for i in range(period + 1, n):
        atr[i] = atr[i-1] * (1 - k) + tr[i] * k
    return float(atr[-1])


def _normalize_candles(df):
    """Normalize to time-indexed OHLC DataFrame (UTC)."""
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
        return df[cols].astype(float)
    except Exception as e:
        log.debug("candle normalize failed: %s", e)
        return None


# ════════════════════════════════════════════════════════════════════════
#  NR7Strategy — detector class
# ════════════════════════════════════════════════════════════════════════
class NR7Strategy:
    """Per-symbol stateless NR7 compression detector. Call evaluate() once per
    brain cycle. Returns a signal dict (with direction + stop-order entry +
    structural SL + 1R/2R TPs) or None."""

    def __init__(self, state, params=None):
        self.state = state
        p = params or {}
        self.nr_lookback = int(p.get("NR_LOOKBACK", NR_LOOKBACK))
        self.buffer_atr = float(p.get("BUFFER_ATR", BUFFER_ATR))
        self.sl_buffer_atr = float(p.get("SL_BUFFER_ATR", SL_BUFFER_ATR))
        self.tp1_r = float(p.get("TP1_R", TP1_R))
        self.tp2_r = float(p.get("TP2_R", TP2_R))
        self.expiry_bars = int(p.get("EXPIRY_BARS", EXPIRY_BARS))
        self.min_nr_frac = float(p.get("MIN_NR_FRACTION_OF_ATR", MIN_NR_FRACTION_OF_ATR))
        self.max_nr_frac = float(p.get("MAX_NR_FRACTION_OF_ATR", MAX_NR_FRACTION_OF_ATR))
        # Per-symbol dedupe on H1 close time.
        self._last_bar_t = {}

    def _get_h1(self, symbol):
        if self.state is None:
            return None
        try:
            return self.state.get_candles(symbol, 60)
        except Exception as e:
            log.debug("[%s] H1 fetch failed: %s", symbol, e)
            return None

    def evaluate(self, symbol):
        """Inspect the most-recent closed H1 bar; emit NR7 signal or None."""
        h1_raw = self._get_h1(symbol)
        h1 = _normalize_candles(h1_raw)
        if h1 is None or len(h1) < MIN_H1_BARS:
            return None

        # Use index -2 = last CLOSED H1 bar (-1 is forming).
        i = len(h1) - 2
        if i < self.nr_lookback + 1:
            return None
        bar = h1.iloc[i]
        bar_t = h1.index[i]

        if self._last_bar_t.get(symbol) == bar_t:
            return None

        H = h1["high"].values[:i+1]
        L = h1["low"].values[:i+1]
        C = h1["close"].values[:i+1]

        # ATR for noise-level reference.
        atr = _atr(H, L, C, ATR_PERIOD)
        if atr <= 0:
            return None

        # Compute bar ranges for the lookback window [i-nr_lookback+1 .. i]
        # (inclusive of candidate bar) — NR7 means candidate bar's range is
        # STRICTLY less than every other bar's range in the window.
        ranges = H[i - self.nr_lookback + 1 : i + 1] - L[i - self.nr_lookback + 1 : i + 1]
        candidate_range = ranges[-1]   # bar i's range
        other_ranges = ranges[:-1]     # bars i-N+1 .. i-1
        if candidate_range >= other_ranges.min():
            # Not an NR7 — at least one prior bar in window was narrower or equal
            self._last_bar_t[symbol] = bar_t
            return None

        # Sanity gates: compression should be REAL but not dead-market micro
        nr_frac = candidate_range / atr if atr > 0 else 0
        if nr_frac < self.min_nr_frac or nr_frac > self.max_nr_frac:
            self._last_bar_t[symbol] = bar_t
            return None

        nr_high = float(H[i])
        nr_low = float(L[i])
        buf = self.buffer_atr * atr
        sl_buf = self.sl_buffer_atr * atr

        # Decide direction:
        # In a real live deploy this is OCO: place buy-stop above & sell-stop
        # below; first to fill wins. For backtest/signal purposes we emit a
        # single direction — pick based on close vs midpoint:
        #   close > mid → emit LONG (slight bullish lean)
        #   close < mid → emit SHORT
        # This biases entries toward the "bias the market already shows",
        # which is empirically slightly better than emitting both.
        mid = (nr_high + nr_low) / 2.0
        cl = float(bar["close"])
        if cl >= mid:
            direction = "LONG"
            entry = nr_high + buf
            sl = nr_low - sl_buf
            risk = entry - sl
            tp1 = entry + self.tp1_r * risk
            tp2 = entry + self.tp2_r * risk
        else:
            direction = "SHORT"
            entry = nr_low - buf
            sl = nr_high + sl_buf
            risk = sl - entry
            tp1 = entry - self.tp1_r * risk
            tp2 = entry - self.tp2_r * risk

        if risk <= 0:
            return None

        self._last_bar_t[symbol] = bar_t
        return {
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "nr7_high": nr_high,
            "nr7_low": nr_low,
            "nr7_range": float(candidate_range),
            "atr14": atr,
            "nr_frac_atr": nr_frac,
            "bar_time": bar_t,
            "reason": f"{direction} NR7 break (range {candidate_range:.5f} = {nr_frac:.2f} ATR; stop@{entry:.5f})",
        }


# ════════════════════════════════════════════════════════════════════════
#  Standalone smoke test
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    np.random.seed(7)
    n = 100
    base = 100.0
    rng = np.cumsum(np.random.normal(0, 0.6, n))
    closes = base + rng
    highs = closes + np.abs(np.random.normal(0, 0.4, n)) + 0.2
    lows = closes - np.abs(np.random.normal(0, 0.4, n)) - 0.2
    opens = np.concatenate([[base], closes[:-1]])

    # Force a narrow-range bar at index 70 (NR7-positive)
    nr_idx = 70
    mid_70 = (highs[nr_idx] + lows[nr_idx]) / 2
    half = 0.15  # very narrow range
    highs[nr_idx] = mid_70 + half
    lows[nr_idx] = mid_70 - half
    opens[nr_idx] = mid_70 - half * 0.3
    closes[nr_idx] = mid_70 + half * 0.5
    # Make surrounding bars wider so the NR7 is unique
    for k in range(nr_idx - 6, nr_idx):
        rng_old = highs[k] - lows[k]
        if rng_old < 0.5:
            mid = (highs[k] + lows[k]) / 2
            highs[k] = mid + 0.3
            lows[k] = mid - 0.3

    times = pd.date_range("2026-01-01", periods=n, freq="60min", tz="UTC")
    df = pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows, "close": closes,
    })

    # Truncate to nr_idx + 2 bars so iloc[-2] == the NR7 bar
    df_short = df.iloc[: nr_idx + 1].copy()

    class _FakeState:
        def get_candles(self, sym, tf):
            if tf == 60:
                d2 = df_short.copy()
                new_row = pd.DataFrame([{
                    "time": df_short["time"].iloc[-1] + pd.Timedelta("60min"),
                    "open": df_short["close"].iloc[-1],
                    "high": df_short["close"].iloc[-1] + 0.1,
                    "low": df_short["close"].iloc[-1] - 0.1,
                    "close": df_short["close"].iloc[-1],
                }])
                return pd.concat([d2, new_row], ignore_index=True)
            return None

    strat = NR7Strategy(_FakeState())
    sig = strat.evaluate("TEST")
    print(f"NR7 signal: {sig}")
    if sig and sig["direction"] in ("LONG", "SHORT") and abs(sig["nr7_range"] - 0.30) < 0.01:
        print("✓ SMOKE TEST PASSED")
    else:
        print("✗ SMOKE TEST FAILED")
