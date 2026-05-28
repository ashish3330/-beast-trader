#!/usr/bin/env python3 -B
"""
LIVE ICT Liquidity-Sweep + FVG-Retest strategy module (Dragon bot).

Branch: experiment/new-entry-exit-logic.

This is the LIVE counterpart to the validated backtest in
``backtest/ict_fvg_backtest.py`` (+176R / 42% WR over 2yr). It runs the SAME
sequential state machine, but instead of scanning a 2-year DataFrame in one
pass it is called once per brain cycle with the rolling live candle window and
maintains the sweep / FVG state per symbol across calls.

Strategy (mirrors the backtest exactly):
  Bias    : Daily — price above/below the 200 EMA → long/short bias only.
  Liquidity: 4H — price sweeps a prior swing high/low (false breakout) then
             closes back inside.
  Trigger : 15M — a 3-candle FVG (imbalance) forms in the reversal direction;
            price retraces into the gap; enter at the gap MIDPOINT.

  Entry LONG : sweep BELOW a 4H swing low, reverse up, bullish FVG
               (low[c3] > high[c1]); fill when price retraces to gap midpoint
               (low <= mid).
  Entry SHORT: sweep ABOVE a 4H swing high, reverse down, bearish FVG
               (high[c3] < low[c1]); fill when price retraces to gap midpoint
               (high >= mid).

  Exits (returned for the wiring agent to place broker-side):
    SL  = the swept liquidity level (sweep extreme), +/- one spread.
    TP1 = 1.5 x stop distance  (close 50%)
    TP2 = 3.0 x stop distance  (close remaining 50%)
    Time stop: close entire position if TP1 not hit within TIME_STOP_HOURS.
    (Entry-candle breach rule from the backtest is an exit-side concern; this
     module only produces the entry signal + levels. The wiring agent /
     executor owns the breach / time-stop bookkeeping. DISABLE_BREACH matched
     the validated run — see the backtest header.)

────────────────────────────────────────────────────────────────────────────
LIVE DATA NOTE (important):
The bot keeps only CANDLE_WINDOW (=500) bars per (symbol, timeframe) in
SharedState. 500 M15 bars is ~5 trading days — far short of the 200 *daily*
bars a true Daily-200-EMA wants. So the Daily bias here is derived adaptively:

  * Base / swing / FVG frame : M15 from SharedState (resampled M15 -> 4H).
  * Daily bias               : resample M15 -> D1 and EMA over the *available*
                               daily depth (period = min(DAILY_EMA_PERIOD,
                               n_daily)). If too few daily bars exist we fall
                               back to H1 candles (state.get_candles(sym, 60))
                               and use their EMA200 as the daily-trend proxy —
                               this matches the convention already used in
                               brain.py's TREND_FILTER (H1 EMA200 == long-term
                               trend surrogate).

When run STANDALONE against the offline M15 cache pickle (50k bars / 2yr) the
full Daily-200-EMA is available, so the smoke test reproduces backtest-grade
bias. In live mode the adaptive period keeps the module functional on the
short rolling window instead of returning None forever.
────────────────────────────────────────────────────────────────────────────

API:
    strat = FVGStrategy(state)              # state = SharedState (or None)
    sig = strat.evaluate("NAS100.r")        # call every brain cycle
    # sig is None, or:
    # {"direction": "LONG"|"SHORT", "entry": float, "sl": float,
    #  "tp1": float, "tp2": float, "swept_level": float, "reason": str}

The wiring agent should call ``evaluate(symbol)`` once per cycle per symbol.
A non-None return means: open ``direction`` at market (entry ~= the returned
``entry`` midpoint, fill at current price), SL at ``sl``, scale out 50% at
``tp1`` and the rest at ``tp2``, and arm a TIME_STOP_HOURS time stop.

Constraints honoured:
  * NEW file only — no existing bot file is touched/imported destructively.
  * Defensive: missing / short candle data -> return None.
  * Closed bars only — detection uses index n-2 (the forming bar n-1 is never
    read for signals).
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("dragon.fvg")

# ════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMS — importable / overridable by the tuning agent.
#  (Mirror the backtest defaults; the tuner can patch these module attrs or
#   pass a `params` dict to FVGStrategy.)
# ════════════════════════════════════════════════════════════════════════
SWING_LOOKBACK = 3          # symmetric fractal half-window on the 4H frame
SWING_MEMORY = 20           # recent 4H swing levels kept live as sweep targets
SWEEP_TO_FVG_BARS = 12      # M15 bars to form the reversal FVG after a sweep
SETUP_EXPIRY_BARS_15M = 24  # M15 bars to wait for the midpoint fill (~6h)
TIME_STOP_HOURS = 4.0       # close all if TP1 not hit within this window
TP1_R = 1.5                 # TP1 R multiple (close 50%)
TP2_R = 3.0                 # TP2 R multiple (close runner 50%)
DAILY_EMA_PERIOD = 200      # daily bias EMA period (adaptive when short)

# Minimum bars we need on the base M15 frame before even trying.
MIN_M15_BARS = 60
# Resample rules.
H4_RULE = "4h"
D1_RULE = "1D"
# Per-asset spread (price units) for SL placement — mirrors the backtest.
SPREAD = {
    "EURUSD": 0.00015, "GBPUSD": 0.00020, "USDJPY": 0.015, "USDCAD": 0.00020,
    "USDCHF": 0.00020, "AUDJPY": 0.020, "EURJPY": 0.020, "GBPJPY": 0.025,
    "XAUUSD": 0.30, "XAGUSD": 0.030, "BTCUSD": 30.0, "ETHUSD": 2.0,
    "NAS100.r": 1.50, "SP500.r": 0.50, "DJ30.r": 2.0, "US2000.r": 0.50,
    "GER40.r": 2.0, "UK100.r": 2.0, "JPN225ft": 10.0, "SPI200.r": 2.0,
    "SWI20.r": 3.0, "XPTUSD.r": 1.0, "USOUSD": 0.05,
}
DEFAULT_SPREAD = 0.0002


# ════════════════════════════════════════════════════════════════════════
#  Pure helpers (no external deps beyond numpy/pandas).
# ════════════════════════════════════════════════════════════════════════
def _ema(arr, period):
    """Recursive EMA — identical math to the backtest's _ema."""
    arr = np.asarray(arr, dtype=float)
    out = np.empty(len(arr))
    if len(arr) == 0:
        return out
    k = 2.0 / (period + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _resample(df, rule):
    """OHLC resample from a time-indexed base DataFrame."""
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    return out


def _normalize_candles(df):
    """Accept a SharedState / cache candle DataFrame (columns:
    time, open, high, low, close[, tick_volume, ...]) and return a clean
    time-indexed OHLC frame (UTC, ns resolution), sorted. Returns None on bad
    input."""
    if df is None or len(df) == 0:
        return None
    try:
        df = df.copy()
        if "time" not in df.columns:
            # Already time-indexed? try to use the index.
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
    except Exception as e:  # pragma: no cover - defensive
        log.debug("candle normalize failed: %s", e)
        return None


def _atr_at(H, L, C, i, period=14):
    """Wilder ATR at closed bar `i` (warms up over ~4*period prior bars).
    Used only for the momentum-sizing tilt — cheap because it runs on a fill,
    not every bar."""
    start = max(1, i - 4 * period)
    atr = None
    k = 1.0 / period
    for j in range(start, i + 1):
        tr = max(H[j] - L[j], abs(H[j] - C[j - 1]), abs(L[j] - C[j - 1]))
        atr = tr if atr is None else atr * (1 - k) + tr * k
    return atr if atr is not None else 0.0


def _find_swings(h4, lookback):
    """Confirmed 4H swing highs/lows via a symmetric fractal of `lookback`
    bars each side. Returns (highs, lows) as lists of (time_ns, level).

    Identical to the backtest's _find_swings."""
    highs, lows = [], []
    H = h4["high"].values
    L = h4["low"].values
    T = h4.index
    n = len(h4)
    w = lookback
    for i in range(w, n - w):
        seg_h = H[i - w:i + w + 1]
        seg_l = L[i - w:i + w + 1]
        if H[i] == seg_h.max() and (seg_h == H[i]).sum() == 1:
            highs.append((T[i].value, H[i]))
        if L[i] == seg_l.min() and (seg_l == L[i]).sum() == 1:
            lows.append((T[i].value, L[i]))
    return highs, lows


# ════════════════════════════════════════════════════════════════════════
#  FVGStrategy — live state machine.
# ════════════════════════════════════════════════════════════════════════
class FVGStrategy:
    """Per-symbol persistent ICT sweep+FVG-retest signal generator.

    Call ``evaluate(symbol)`` once per brain cycle. State (pending sweeps,
    pending FVGs, last-processed bar time) is kept per symbol so the same
    sequential logic as the backtest plays out across discrete live calls.
    """

    def __init__(self, state, params=None):
        """state : a SharedState-like object exposing
                     get_candles(symbol, tf) -> DataFrame | None
                   (tf in minutes: 15 = M15, 60 = H1). May be None for the
                   standalone/offline mode where candles are injected directly.
        params : optional dict to override any of the module-level tunables
                 (SWING_LOOKBACK, SWING_MEMORY, SWEEP_TO_FVG_BARS,
                  SETUP_EXPIRY_BARS_15M, TIME_STOP_HOURS, TP1_R, TP2_R,
                  DAILY_EMA_PERIOD).
        """
        self.state = state
        p = params or {}
        self.swing_lookback = int(p.get("SWING_LOOKBACK", SWING_LOOKBACK))
        self.swing_memory = int(p.get("SWING_MEMORY", SWING_MEMORY))
        self.sweep_to_fvg_bars = int(p.get("SWEEP_TO_FVG_BARS", SWEEP_TO_FVG_BARS))
        self.setup_expiry_bars = int(p.get("SETUP_EXPIRY_BARS_15M", SETUP_EXPIRY_BARS_15M))
        self.time_stop_hours = float(p.get("TIME_STOP_HOURS", TIME_STOP_HOURS))
        self.tp1_r = float(p.get("TP1_R", TP1_R))
        self.tp2_r = float(p.get("TP2_R", TP2_R))
        self.daily_ema_period = int(p.get("DAILY_EMA_PERIOD", DAILY_EMA_PERIOD))

        # Per-symbol persistent state.
        #   long_sweep / short_sweep : {"swept_level", "sweep_t"} awaiting FVG
        #   long_fvg   / short_fvg   : {"mid","top","bot","swept_level","fvg_t"}
        #                              awaiting midpoint fill
        #   last_bar_t               : ns time of the last CLOSED bar processed
        self._st = {}

    # ── state helpers ──────────────────────────────────────────────────
    def _sym_state(self, symbol):
        s = self._st.get(symbol)
        if s is None:
            s = {"long_sweep": None, "short_sweep": None,
                 "long_fvg": None, "short_fvg": None, "last_bar_t": None}
            self._st[symbol] = s
        return s

    def reset(self, symbol=None):
        """Drop pending state (e.g. after a fill, or on restart)."""
        if symbol is None:
            self._st.clear()
        else:
            self._st.pop(symbol, None)

    # ── data pulls ─────────────────────────────────────────────────────
    def _get_m15(self, symbol):
        if self.state is None:
            return None
        try:
            return self.state.get_candles(symbol, 15)
        except Exception as e:  # pragma: no cover - defensive
            log.debug("[%s] get_candles M15 failed: %s", symbol, e)
            return None

    def _get_h1(self, symbol):
        if self.state is None:
            return None
        try:
            return self.state.get_candles(symbol, 60)
        except Exception as e:  # pragma: no cover - defensive
            log.debug("[%s] get_candles H1 failed: %s", symbol, e)
            return None

    # ── bias ───────────────────────────────────────────────────────────
    def _daily_bias(self, symbol, m15):
        """Return +1 (long-only), -1 (short-only) or 0 (no bias) for `now`,
        computed on the most recent CLOSED daily bar.

        Primary: resample base M15 -> D1, EMA over min(period, n_daily).
        Fallback (too few daily bars): H1 EMA200 trend proxy (brain convention).
        """
        try:
            d1 = _resample(m15, D1_RULE)
            # Use only CLOSED daily bars: the last row may be a forming day.
            if len(d1) >= 3:
                closes = d1["close"].values[:-1]  # drop forming day
                period = min(self.daily_ema_period, len(closes))
                if period >= 2:
                    ema = _ema(closes, period)
                    if closes[-1] > ema[-1]:
                        return 1
                    if closes[-1] < ema[-1]:
                        return -1
                    return 0
        except Exception as e:
            log.debug("[%s] D1 bias failed: %s", symbol, e)

        # Fallback: H1 EMA200 proxy (matches brain.py TREND_FILTER convention).
        h1 = _normalize_candles(self._get_h1(symbol))
        if h1 is not None and len(h1) >= 30:
            try:
                period = min(200, len(h1))
                ema = _ema(h1["close"].values, period)
                cur = float(h1["close"].values[-1])
                if cur > ema[-1]:
                    return 1
                if cur < ema[-1]:
                    return -1
            except Exception as e:
                log.debug("[%s] H1 bias fallback failed: %s", symbol, e)
        return 0

    # ── main entry point ───────────────────────────────────────────────
    def evaluate(self, symbol):
        """Run the sweep -> FVG -> midpoint-fill state machine for `symbol`
        on newly-closed M15 bars. Returns a signal dict or None.

        Idempotent within a candle: if no new closed bar has appeared since
        the last call, state is left untouched and None is returned (a fill
        can only happen on a freshly-closed bar)."""
        raw = self._get_m15(symbol)
        m15 = _normalize_candles(raw)
        if m15 is None or len(m15) < MIN_M15_BARS:
            return None

        st = self._sym_state(symbol)

        # Build the 4H swing set from the available M15 history.
        try:
            h4 = _resample(m15, H4_RULE)
        except Exception as e:
            log.debug("[%s] 4H resample failed: %s", symbol, e)
            return None
        if len(h4) < (2 * self.swing_lookback + 2):
            return None
        # Drop the (possibly) forming 4H bar before fractal detection so a
        # swing is only "confirmed" once its right-side window has closed.
        swing_highs, swing_lows = _find_swings(h4.iloc[:-1], self.swing_lookback)

        bias = self._daily_bias(symbol, m15)

        # Base arrays (M15). We process CLOSED bars only: the live frame's
        # last row (index -1) is the forming bar, so the last CLOSED bar is
        # index n-2. We advance through any closed bars not yet seen.
        t_ns = m15.index.view("int64")
        o = m15["open"].values
        h = m15["high"].values
        l = m15["low"].values
        c = m15["close"].values
        n = len(m15)

        last_closed_idx = n - 2  # never read the forming bar n-1
        if last_closed_idx < 2:
            return None

        # Determine the first new closed-bar index to process.
        if st["last_bar_t"] is None:
            # First ever call for this symbol: only process the most recent
            # closed bar (avoid replaying ancient history live — that would
            # fire a stale fill). Standalone smoke test overrides via
            # evaluate_series().
            start = last_closed_idx
        else:
            # find first index whose time > last processed closed-bar time
            start = np.searchsorted(t_ns, st["last_bar_t"], side="right")
            if start < 2:
                start = 2

        signal = None
        for i in range(start, last_closed_idx + 1):
            sig = self._step(symbol, st, bias, i, t_ns, o, h, l, c, n,
                             swing_highs, swing_lows)
            st["last_bar_t"] = int(t_ns[i])
            if sig is not None:
                signal = sig
                break  # one signal per evaluate; brain re-calls next cycle
        return signal

    # ── one-bar transition (mirrors the backtest's per-bar block) ───────
    def _step(self, symbol, st, bias, i, T, O, H, L, C, n,
              swing_highs, swing_lows):
        """Advance the state machine by one closed bar `i`. Returns a signal
        dict if a midpoint fill triggered on this bar, else None."""
        t = int(T[i])

        # Live sweep-target sets: 4H swings confirmed at/before this bar's time.
        live_highs = [lvl for (tt, lvl) in swing_highs if tt <= t][-self.swing_memory:]
        live_lows = [lvl for (tt, lvl) in swing_lows if tt <= t][-self.swing_memory:]

        # ======================= LONG side =======================
        if bias == 1:
            # (1) FILL: pending bullish FVG -> enter on retrace to midpoint.
            if st["long_fvg"] is not None:
                fvg = st["long_fvg"]
                age = np.searchsorted(T, fvg["fvg_t"], side="right")
                # age in bars since the FVG bar:
                bars_since = i - (np.searchsorted(T, fvg["fvg_t"], side="left"))
                if bars_since > self.setup_expiry_bars:
                    st["long_fvg"] = None
                elif L[i] <= fvg["mid"]:
                    sm = self._size_mult(1, i, C, H, L)
                    sig = self._make_signal(symbol, 1, fvg["mid"],
                                            fvg["swept_level"], "long FVG midpoint fill", sm)
                    st["long_fvg"] = None
                    if sig is not None:
                        return sig
            # (2) FVG FORM: after a sweep, first bullish 3-candle gap.
            if st["long_fvg"] is None and st["long_sweep"] is not None:
                sw = st["long_sweep"]
                bars_since = i - (np.searchsorted(T, sw["sweep_t"], side="left"))
                if bars_since > self.sweep_to_fvg_bars:
                    st["long_sweep"] = None
                elif L[i] > H[i - 2]:  # bullish FVG: low3 > high1
                    top, bot = L[i], H[i - 2]
                    st["long_fvg"] = {"mid": (top + bot) / 2.0, "top": top, "bot": bot,
                                      "swept_level": sw["swept_level"], "fvg_t": t}
                    st["long_sweep"] = None
            # (3) SWEEP: low pierces a swing low then closes back above it.
            if st["long_sweep"] is None and st["long_fvg"] is None and live_lows:
                below = [lvl for lvl in live_lows if L[i] < lvl <= C[i]]
                if below:
                    swept_level = max(below)
                    st["long_sweep"] = {"swept_level": swept_level, "sweep_t": t}

        # ======================= SHORT side =======================
        if bias == -1:
            if st["short_fvg"] is not None:
                fvg = st["short_fvg"]
                bars_since = i - (np.searchsorted(T, fvg["fvg_t"], side="left"))
                if bars_since > self.setup_expiry_bars:
                    st["short_fvg"] = None
                elif H[i] >= fvg["mid"]:
                    sm = self._size_mult(-1, i, C, H, L)
                    sig = self._make_signal(symbol, -1, fvg["mid"],
                                            fvg["swept_level"], "short FVG midpoint fill", sm)
                    st["short_fvg"] = None
                    if sig is not None:
                        return sig
            if st["short_fvg"] is None and st["short_sweep"] is not None:
                sw = st["short_sweep"]
                bars_since = i - (np.searchsorted(T, sw["sweep_t"], side="left"))
                if bars_since > self.sweep_to_fvg_bars:
                    st["short_sweep"] = None
                elif H[i] < L[i - 2]:  # bearish FVG: high3 < low1
                    top, bot = L[i - 2], H[i]
                    st["short_fvg"] = {"mid": (top + bot) / 2.0, "top": top, "bot": bot,
                                       "swept_level": sw["swept_level"], "fvg_t": t}
                    st["short_sweep"] = None
            if st["short_sweep"] is None and st["short_fvg"] is None and live_highs:
                above = [lvl for lvl in live_highs if H[i] > lvl >= C[i]]
                if above:
                    swept_level = min(above)
                    st["short_sweep"] = {"swept_level": swept_level, "sweep_t": t}

        return None

    # ── momentum-aligned dynamic sizing (validated +23% R, DD unchanged) ──
    def _size_mult(self, direction, i, C, H, L):
        """Bounded size multiplier from direction-aligned 16-bar (~4h) ROC
        normalised by M15 ATR14:  clip(1.0 + 0.30*roc_dir, 0.60, 1.40).

        Validated 2026-05-29 on 180d/7-sym FVG basket: +23.3% size-weighted R
        on identical TOTAL risk, max-consec-R DD 15.16 vs flat 15.30 (no DD
        cost). Daily-EMA-distance (no corr) and FVG-displacement (anti-
        predictive) were tested and REJECTED. Falls back to 1.0 (base size) on
        a short window or degenerate ATR — never blocks the trade
        (feedback_no_skip_trades)."""
        try:
            if i < 16:
                return 1.0
            atr = _atr_at(H, L, C, i, 14)
            if not np.isfinite(atr) or atr <= 0:
                return 1.0
            roc_dir = direction * (C[i] - C[i - 16]) / atr
            return float(max(0.60, min(1.40, 1.0 + 0.30 * roc_dir)))
        except Exception:
            return 1.0

    # ── signal builder (SL/TP placement mirrors the backtest _simulate) ──
    def _make_signal(self, symbol, direction, entry_px, swept_level, reason,
                     size_mult=1.0):
        """Build the entry signal dict with SL/TP1/TP2 levels. Applies the
        same degenerate-stop guard as the backtest (reject if the stop is
        smaller than max(3*spread, 5bp of price))."""
        spread = SPREAD.get(symbol, DEFAULT_SPREAD)
        if direction == 1:
            sl = swept_level - spread
            stop_dist = entry_px - sl
        else:
            sl = swept_level + spread
            stop_dist = sl - entry_px

        min_stop = max(3.0 * spread, 0.0005 * abs(entry_px))
        if stop_dist < min_stop or stop_dist <= 0:
            log.debug("[%s] rejected degenerate stop: dist=%.6f min=%.6f",
                      symbol, stop_dist, min_stop)
            return None

        tp1 = entry_px + direction * self.tp1_r * stop_dist
        tp2 = entry_px + direction * self.tp2_r * stop_dist
        sig = {
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": float(entry_px),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "swept_level": float(swept_level),
            "time_stop_hours": self.time_stop_hours,
            "size_mult": float(size_mult),
            "reason": f"ICT sweep+FVG: {reason} (R={stop_dist:.5f})",
        }
        log.info("[%s] FVG SIGNAL %s entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f swept=%.5f sizex=%.2f",
                 symbol, sig["direction"], entry_px, sl, tp1, tp2, swept_level, size_mult)
        return sig

    # ── offline replay (smoke test / validation only) ───────────────────
    def evaluate_series(self, symbol, m15_df, full_history=True):
        """Replay the full M15 history bar-by-bar (for the standalone smoke
        test). Returns a list of all signals that would have fired. NOT used
        live — live uses evaluate() on the rolling window.

        This recomputes bias/swings ONCE over the whole frame (acceptable for
        an offline proof; the live path recomputes per cycle on the short
        window)."""
        m15 = _normalize_candles(m15_df)
        if m15 is None or len(m15) < MIN_M15_BARS:
            return []
        self.reset(symbol)
        st = self._sym_state(symbol)

        h4 = _resample(m15, H4_RULE)
        swing_highs, swing_lows = _find_swings(h4, self.swing_lookback)

        # Full Daily-200-EMA bias mapped forward onto each base bar (exactly
        # like the backtest), since the offline pickle has full history.
        d1 = _resample(m15, D1_RULE).copy()
        period = min(self.daily_ema_period, len(d1))
        d1["ema"] = _ema(d1["close"].values, period)
        d1["bias"] = np.where(d1["close"] > d1["ema"], 1, -1)
        bias_series = d1["bias"].reindex(m15.index, method="ffill").fillna(0)

        t_ns = m15.index.view("int64")
        o = m15["open"].values
        h = m15["high"].values
        l = m15["low"].values
        c = m15["close"].values
        n = len(m15)

        signals = []
        for i in range(2, n - 1):  # closed bars only (skip forming bar n-1)
            bias = int(bias_series.iloc[i])
            sig = self._step(symbol, st, bias, i, t_ns, o, h, l, c, n,
                             swing_highs, swing_lows)
            if sig is not None:
                sig = dict(sig)
                sig["bar_time"] = str(m15.index[i])
                signals.append(sig)
        return signals


# ════════════════════════════════════════════════════════════════════════
#  Standalone smoke test.
# ════════════════════════════════════════════════════════════════════════
def _smoke(symbol="NAS100.r"):
    """Load a cached M15 pickle and print every signal the strategy fires."""
    logging.basicConfig(level=logging.WARNING,
                        format="%(name)s %(levelname)s %(message)s")
    cache = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
    fn_variants = [
        f"raw_m15_{symbol}.pkl",
        f"raw_m15_{symbol.replace('.', '_')}.pkl",
        f"raw_m15_{symbol.lower()}.pkl",
    ]
    path = None
    for v in fn_variants:
        p = cache / v
        if p.exists():
            path = p
            break
    if path is None:
        print(f"[smoke] no M15 cache for {symbol} (tried {fn_variants})")
        return
    print(f"[smoke] loading {path}")
    df = pickle.load(open(path, "rb"))
    print(f"[smoke] {len(df)} M15 bars")

    strat = FVGStrategy(state=None)  # offline: candles injected directly
    sigs = strat.evaluate_series(symbol, df)
    print(f"[smoke] {symbol}: {len(sigs)} signals fired over full history")
    longs = sum(1 for s in sigs if s["direction"] == "LONG")
    shorts = len(sigs) - longs
    print(f"[smoke]   LONG={longs}  SHORT={shorts}")
    for s in sigs[:5]:
        print(f"[smoke]   {s['bar_time']}  {s['direction']:<5} "
              f"entry={s['entry']:.2f} sl={s['sl']:.2f} "
              f"tp1={s['tp1']:.2f} tp2={s['tp2']:.2f} swept={s['swept_level']:.2f}")
    if len(sigs) > 5:
        print(f"[smoke]   ... (+{len(sigs) - 5} more)")
    print(f"[smoke]   last: {sigs[-1]['bar_time'] if sigs else 'n/a'}")

    # ── also exercise the LIVE path on a short rolling window ──
    full = _normalize_candles(df)
    if full is not None and len(full) > 600:
        print("\n[smoke] live-path check on a 500-bar rolling window:")
        live_state = _FakeState(full.iloc[-500:].reset_index())
        live_strat = FVGStrategy(state=live_state)
        live_sig = live_strat.evaluate(symbol)
        print(f"[smoke]   evaluate() on latest 500 M15 bars -> "
              f"{live_sig if live_sig else 'None (no fresh setup at window edge — expected)'}")


class _FakeState:
    """Minimal SharedState stand-in for the smoke test's live-path check."""
    def __init__(self, m15_df):
        self._m15 = m15_df

    def get_candles(self, symbol, tf):
        if tf == 15:
            return self._m15.copy()
        return None


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "NAS100.r"
    _smoke(sym)
