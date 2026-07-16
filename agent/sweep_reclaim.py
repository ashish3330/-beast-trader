#!/usr/bin/env python3 -B
"""
Liquidity-Sweep + Reclaim entry detector (Dragon bot).

REPLACEMENT for the broken momentum-score-then-confirm paradigm. The momentum
stack stacked 11 lagging indicators and required N-of-K agreement: by the time
agreement printed, the move was over and we entered at swing extremes (the
"buy swing high / sell swing low" pathology proven 2026-06-05 from journal —
94 EarlyLossCut_T1 fires for -$270, 30% of LONG / 20% of SHORT trades went
straight to SL with no peak).

Research basis (cited in feedback_value_entry_research_20260605.md):
  * Bourgade & Hassani (arXiv 2009.08821) — N-of-K confirmation on lagging
    indicators is mathematically required to lag by ~½ slowest period.
  * Bouchaud et al. (arXiv 2501.16772) — markets trend on medium term and
    mean-revert on long term; by confirmation, you're in persist-regime where
    R:R is worst.
  * Adam Grimes / Linda Raschke / Toby Crabel — all explicitly avoid indicator-
    stack entries; favor structural/event-based triggers.
  * Daily Price Action sweep-reclaim writeup, ICT Liquidity-Sweep tutorials —
    public retail validation of the 60-75% WR profile when filters applied.

Strategy (single-bar event detector — NOT a state machine):
  For each symbol, look at the most recent CLOSED M15 bar (index -2 of the
  live frame). Define:
    swing_L = fractal-confirmed lowest low in last `SWING_LOOKBACK` closed bars
              (strict pivot: low[i] < low[i±w] for w in 1..SWING_WIN)
    swing_H = symmetric high

  Bullish sweep-reclaim (LONG signal):
    1. bar.low < swing_L                     (price wicked below the level)
    2. bar.close > swing_L                   (reclaimed back inside range)
    3. (swing_L - bar.low) >= MIN_WICK_ATR * ATR14   (real sweep, not noise)
    4. bar.close > bar.open                  (bullish reclaim candle)
    5. (optional) volume[i] > VOL_MULT * vol_SMA20   (absorption signature)
    6. Regime gate: H1 ADX14 < ADX_REGIME_MAX        (no strong trend = sweep
                                                       is reversal, not
                                                       continuation)

  Bearish sweep-reclaim: symmetric.

  Levels emitted to executor:
    entry = bar.close (immediate-fill at market on signal arrival)
    sl    = bar.low (LONG) - SL_BUFFER_ATR * ATR14    (just beyond sweep wick)
    risk  = entry - sl
    tp1   = entry + TP1_R * risk          (default 1.0R — close 50%)
    tp2   = entry + TP2_R * risk          (default 2.0R — close 50% runner)
    time_stop = TIME_STOP_BARS closed M15 bars after entry (if peak < 0.3R)

  Why this avoids the late-entry trap:
    - Entry IS the reversal print (reclaim close), which is the lowest
      probable price in the new bullish micro-structure.
    - Stop sits 0.1 ATR beyond the sweep wick — the tightest possible
      structural stop. If price re-breaks the swept level, the thesis is
      dead. Fast exit, small loss, defined invalidation.
    - R:R is mechanically asymmetric in our favor (stop ~0.5 ATR, target 1R
      partial + 2R runner = blended R/R ~1.5).
    - No indicator confirmation required. The sweep IS the confirmation.

API (mirrors FVGStrategy for drop-in wiring):
    strat = SweepReclaimStrategy(state)
    sig = strat.evaluate("XAUUSD")
    # sig is None, or:
    # {
    #   "direction": "LONG" | "SHORT",
    #   "entry": float, "sl": float, "tp1": float, "tp2": float,
    #   "swept_level": float, "wick_atr_mult": float,
    #   "reason": str,
    #   "bar_time": pd.Timestamp,
    # }

The wiring agent should call evaluate(symbol) once per brain cycle (per closed
M15 bar). When sig is non-None, decide based on SR_TRADE_LIVE config flag:
  - False: log the signal only (observation mode)
  - True : open trade via executor.open_trade_explicit with SR magic offset

Constraints honoured:
  * NEW file only — does not modify any existing module.
  * Closed bars only — reads index -2 (the live forming bar is never used).
  * Defensive: missing / short data -> return None.
  * Stateless across calls (no per-symbol state required — single-bar event).
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("dragon.sweep_reclaim")

# Optional per-symbol overrides / blacklist — try-import so the detector keeps
# working even if these aren't defined in config (older builds, isolated tests).
try:
    from config import SR_PARAM_OVERRIDES  # type: ignore
except Exception:
    SR_PARAM_OVERRIDES = {}
try:
    from config import SR_SYMBOL_BLACKLIST  # type: ignore
except Exception:
    SR_SYMBOL_BLACKLIST = set()

# ── EQH/EQL Liquidity-Pool Filter (optional, default OFF) ───────────────
# Lifts win rate by ranking sweeps that take out a cluster of equal highs
# or lows ("liquidity pools" institutions target) above sweeps of isolated
# pivots. When ENABLED:
#   * STRICT=False : non-cluster sweeps are downsized (size_mult=0.6)
#   * STRICT=True  : non-cluster sweeps are rejected outright (return None)
# Fail-open: any import / runtime error leaves the signal untouched.
try:
    from config import EQH_EQL_FILTER_ENABLED  # type: ignore
except Exception:
    EQH_EQL_FILTER_ENABLED = False
try:
    from config import EQH_EQL_STRICT  # type: ignore
except Exception:
    EQH_EQL_STRICT = False
try:
    from agent.expert.eqh_eql_detector import (  # type: ignore
        find_eqh_eql as _find_eqh_eql,
        is_sweeping_eqh_eql as _is_sweeping_eqh_eql,
    )
except Exception:
    _find_eqh_eql = None
    _is_sweeping_eqh_eql = None

# ════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMS — overridable via constructor params dict
# ════════════════════════════════════════════════════════════════════════
SWING_LOOKBACK = 20         # how far back to search for swing pivots (closed bars)
SWING_WIN = 3               # strict-pivot half-window (low[i] < low[i±1..3])
MIN_WICK_ATR = 0.25         # sweep wick must be >= 0.25 * ATR14 (real sweep)
SL_BUFFER_ATR = 0.10        # stop = wick extreme +/- 0.10 * ATR14
TP1_R = 2.0                 # 2026-07-17 SR-expand tune: 1.3->2.0 (all per-sym tunes converged here)
TP2_R = 2.0                 # 2026-07-17 SR-expand tune: 1.3->2.0 (equal TPs = 100% close at +2.0R)
TIME_STOP_BARS = 12         # close if peak < 0.3R after 12 closed M15 bars (3h)
TIME_STOP_PEAK_R = 0.3
ADX_REGIME_MAX = 25         # 2026-07-17 SR-expand tune: 35->25 (calmer regime). skip if H1 ADX14 > 25 (sweep =
                            #                          continuation not reversal)
VOL_CONFIRM = False         # require volume[i] > 1.3x SMA20 (off by default —
                            #                                   many feeds have
                            #                                   tick_volume only)
VOL_MULT = 1.3
MIN_M15_BARS = 60           # need at least this many closed M15 bars
MIN_H1_BARS = 30            # need this many H1 for ADX
ATR_PERIOD = 14
ADX_PERIOD = 14


def _atr(H, L, C, period=ATR_PERIOD):
    """Wilder ATR over a numpy array — returns the latest value (or 0 if too short)."""
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


def _adx(H, L, C, period=ADX_PERIOD):
    """Wilder ADX — returns latest value (or 0 if too short)."""
    n = len(H)
    if n < 2 * period + 1:
        return 0.0
    up = H[1:] - H[:-1]
    dn = L[:-1] - L[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.empty(n - 1)
    for i in range(n - 1):
        tr[i] = max(H[i+1] - L[i+1], abs(H[i+1] - C[i]), abs(L[i+1] - C[i]))
    # Wilder smoothing
    sm_plus = np.empty(n - 1); sm_minus = np.empty(n - 1); sm_tr = np.empty(n - 1)
    sm_plus[period-1] = plus_dm[:period].sum()
    sm_minus[period-1] = minus_dm[:period].sum()
    sm_tr[period-1] = tr[:period].sum()
    for i in range(period, n - 1):
        sm_plus[i] = sm_plus[i-1] - sm_plus[i-1]/period + plus_dm[i]
        sm_minus[i] = sm_minus[i-1] - sm_minus[i-1]/period + minus_dm[i]
        sm_tr[i] = sm_tr[i-1] - sm_tr[i-1]/period + tr[i]
    plus_di = 100 * sm_plus / np.where(sm_tr == 0, 1, sm_tr)
    minus_di = 100 * sm_minus / np.where(sm_tr == 0, 1, sm_tr)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    # ADX = Wilder-smoothed DX
    adx = np.empty(n - 1)
    adx[2*period-1] = dx[period-1:2*period].mean()
    for i in range(2*period, n - 1):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return float(adx[-1])


def _is_swing_low(L, i, w):
    """Strict fractal pivot low: L[i] < L[i-w..i-1] and L[i] < L[i+1..i+w]."""
    if i < w or i + w >= len(L):
        return False
    seg = L[i-w:i+w+1]
    return L[i] == seg.min() and (seg == L[i]).sum() == 1


def _is_swing_high(H, i, w):
    if i < w or i + w >= len(H):
        return False
    seg = H[i-w:i+w+1]
    return H[i] == seg.max() and (seg == H[i]).sum() == 1


def _last_swing_low(L, lookback, w):
    """Most recent confirmed swing-low within last `lookback` closed bars."""
    n = len(L)
    # Search from most recent CONFIRMED pivot backward.
    # A pivot at index i is confirmed only after `w` bars have closed after it.
    end = n - 1 - w
    start = max(w, end - lookback)
    for i in range(end, start - 1, -1):
        if _is_swing_low(L, i, w):
            return L[i], i
    return None, None


def _last_swing_high(H, lookback, w):
    n = len(H)
    end = n - 1 - w
    start = max(w, end - lookback)
    for i in range(end, start - 1, -1):
        if _is_swing_high(H, i, w):
            return H[i], i
    return None, None


def _normalize_candles(df):
    """Normalize to time-indexed OHLC DataFrame (UTC). Returns None on bad input."""
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
#  SweepReclaimStrategy — drop-in detector class
# ════════════════════════════════════════════════════════════════════════
class SweepReclaimStrategy:
    """Per-symbol single-bar liquidity-sweep + reclaim signal detector.

    Stateless across calls — every call inspects only the most-recent closed
    M15 bar. The wiring agent should de-duplicate signals by bar time (don't
    fire twice on the same bar). Each call returns either None (no signal) or
    a signal dict (see module docstring for shape).
    """

    def __init__(self, state, params=None):
        self.state = state
        p = params or {}
        self.swing_lookback = int(p.get("SWING_LOOKBACK", SWING_LOOKBACK))
        self.swing_win = int(p.get("SWING_WIN", SWING_WIN))
        self.min_wick_atr = float(p.get("MIN_WICK_ATR", MIN_WICK_ATR))
        self.sl_buffer_atr = float(p.get("SL_BUFFER_ATR", SL_BUFFER_ATR))
        self.tp1_r = float(p.get("TP1_R", TP1_R))
        self.tp2_r = float(p.get("TP2_R", TP2_R))
        self.time_stop_bars = int(p.get("TIME_STOP_BARS", TIME_STOP_BARS))
        self.adx_regime_max = float(p.get("ADX_REGIME_MAX", ADX_REGIME_MAX))
        self.vol_confirm = bool(p.get("VOL_CONFIRM", VOL_CONFIRM))
        self.vol_mult = float(p.get("VOL_MULT", VOL_MULT))
        # Per-symbol last-evaluated bar time — used by the wiring agent
        # to dedupe (the strategy itself is stateless).
        self._last_bar_t = {}

    def _get_m15(self, symbol):
        if self.state is None:
            return None
        try:
            return self.state.get_candles(symbol, 15)
        except Exception as e:
            log.debug("[%s] M15 fetch failed: %s", symbol, e)
            return None

    def _get_h1(self, symbol):
        if self.state is None:
            return None
        try:
            return self.state.get_candles(symbol, 60)
        except Exception as e:
            log.debug("[%s] H1 fetch failed: %s", symbol, e)
            return None

    def evaluate(self, symbol):
        """Inspect the most-recent closed M15 bar; return signal or None."""
        # Early return: blacklisted symbols never fire (surgical kill switch).
        if symbol in SR_SYMBOL_BLACKLIST:
            return None

        # Per-symbol override resolution. The supported keys are intentionally
        # narrow — they map to gates that DO exist in this detector. Unknown
        # keys are silently ignored so callers can stash extra params without
        # crashing the detector.
        sym_ov = SR_PARAM_OVERRIDES.get(symbol, {}) if SR_PARAM_OVERRIDES else {}
        # ATR_EXPANSION_MIN — re-uses min_wick_atr semantics (sweep magnitude
        # threshold expressed as multiple of ATR14). BODY_RATIO_MIN /
        # WICK_RATIO_MIN tighten the reclaim candle quality. HTF_REQUIRED toggles
        # the H1 ADX gate. DAILY_LOSS_KILL_R is a soft cap surfaced to the
        # caller via the signal payload (no internal kill — observation only).
        eff_min_wick_atr = float(sym_ov.get("ATR_EXPANSION_MIN", self.min_wick_atr))
        body_ratio_min = float(sym_ov.get("BODY_RATIO_MIN", 0.0))
        wick_ratio_min = float(sym_ov.get("WICK_RATIO_MIN", 0.0))
        htf_required = bool(sym_ov.get("HTF_REQUIRED", True))
        daily_loss_kill_r = float(sym_ov.get("DAILY_LOSS_KILL_R", 0.0))
        direction_filter = str(sym_ov.get("DIRECTION_FILTER", "BOTH")).upper()

        m15_raw = self._get_m15(symbol)
        m15 = _normalize_candles(m15_raw)
        if m15 is None or len(m15) < MIN_M15_BARS:
            return None

        # Use index -2 = last CLOSED bar (-1 is forming).
        i = len(m15) - 2
        if i < self.swing_win + 2:
            return None
        bar = m15.iloc[i]
        bar_t = m15.index[i]

        # Dedupe: only fire once per bar per symbol (wiring agent uses
        # this hint via _last_bar_t too).
        if self._last_bar_t.get(symbol) == bar_t:
            return None

        H = m15["high"].values[:i + 1]
        L = m15["low"].values[:i + 1]
        C = m15["close"].values[:i + 1]

        atr14 = _atr(H, L, C, ATR_PERIOD)
        if atr14 <= 0:
            return None

        # Regime gate: skip if H1 ADX > threshold (strong trend regime).
        # Override HTF_REQUIRED=False bypasses the gate entirely for symbols
        # where the H1-ADX filter has been empirically harmful.
        if htf_required:
            h1_raw = self._get_h1(symbol)
            h1 = _normalize_candles(h1_raw)
            if h1 is not None and len(h1) >= MIN_H1_BARS:
                adx = _adx(h1["high"].values, h1["low"].values, h1["close"].values, ADX_PERIOD)
                if adx > self.adx_regime_max:
                    return None
            # If H1 absent we proceed (graceful) — better to miss the gate
            # than never fire.

        # Find swing levels using bars STRICTLY BEFORE the candidate bar i.
        # Pivot confirmation needs `swing_win` bars on EACH side, so we look
        # for pivots in [w, i - w - 1]. The "last_swing" helpers already
        # respect this when given the truncated array.
        L_for_pivot = L[:i]  # exclude the candidate bar itself
        H_for_pivot = H[:i]
        sw_low, sw_low_idx = _last_swing_low(L_for_pivot, self.swing_lookback, self.swing_win)
        sw_high, sw_high_idx = _last_swing_high(H_for_pivot, self.swing_lookback, self.swing_win)

        sig = None

        # ── BULLISH SWEEP-RECLAIM (LONG) ───────────────────────────────
        if sw_low is not None and direction_filter != "SHORT":
            wick = sw_low - float(bar["low"])
            bar_range = max(float(bar["high"]) - float(bar["low"]), 1e-9)
            body = abs(float(bar["close"]) - float(bar["open"]))
            body_ratio = body / bar_range
            # For a bullish reclaim, the "lower wick" = open-low (if bullish bar).
            lower_wick = float(min(bar["open"], bar["close"])) - float(bar["low"])
            wick_ratio = lower_wick / bar_range
            if (
                float(bar["low"]) < sw_low
                and float(bar["close"]) > sw_low
                and wick >= eff_min_wick_atr * atr14
                and float(bar["close"]) > float(bar["open"])
                and body_ratio >= body_ratio_min
                and wick_ratio >= wick_ratio_min
            ):
                if self.vol_confirm and "tick_volume" in m15.columns and i >= 20:
                    vol_sma = float(m15["tick_volume"].iloc[i-20:i].mean())
                    if vol_sma > 0 and float(m15["tick_volume"].iloc[i]) < self.vol_mult * vol_sma:
                        return None
                entry = float(bar["close"])
                sl = float(bar["low"]) - self.sl_buffer_atr * atr14
                risk = entry - sl
                if risk <= 0:
                    return None
                sig = {
                    "direction": "LONG",
                    "entry": entry,
                    "sl": sl,
                    "tp1": entry + self.tp1_r * risk,
                    "tp2": entry + self.tp2_r * risk,
                    "swept_level": float(sw_low),
                    "wick_atr_mult": wick / atr14,
                    "atr14": atr14,
                    "bar_time": bar_t,
                    "daily_loss_kill_r": daily_loss_kill_r,
                    "reason": f"LONG sweep-reclaim of swing-L {sw_low:.5f} (wick {wick/atr14:.2f} ATR)",
                }

        # ── BEARISH SWEEP-RECLAIM (SHORT) ──────────────────────────────
        if sig is None and sw_high is not None and direction_filter != "LONG":
            wick = float(bar["high"]) - sw_high
            bar_range_s = max(float(bar["high"]) - float(bar["low"]), 1e-9)
            body_s = abs(float(bar["close"]) - float(bar["open"]))
            body_ratio_s = body_s / bar_range_s
            upper_wick = float(bar["high"]) - float(max(bar["open"], bar["close"]))
            wick_ratio_s = upper_wick / bar_range_s
            if (
                float(bar["high"]) > sw_high
                and float(bar["close"]) < sw_high
                and wick >= eff_min_wick_atr * atr14
                and float(bar["close"]) < float(bar["open"])
                and body_ratio_s >= body_ratio_min
                and wick_ratio_s >= wick_ratio_min
            ):
                if self.vol_confirm and "tick_volume" in m15.columns and i >= 20:
                    vol_sma = float(m15["tick_volume"].iloc[i-20:i].mean())
                    if vol_sma > 0 and float(m15["tick_volume"].iloc[i]) < self.vol_mult * vol_sma:
                        return None
                entry = float(bar["close"])
                sl = float(bar["high"]) + self.sl_buffer_atr * atr14
                risk = sl - entry
                if risk <= 0:
                    return None
                sig = {
                    "direction": "SHORT",
                    "entry": entry,
                    "sl": sl,
                    "tp1": entry - self.tp1_r * risk,
                    "tp2": entry - self.tp2_r * risk,
                    "swept_level": float(sw_high),
                    "wick_atr_mult": wick / atr14,
                    "atr14": atr14,
                    "bar_time": bar_t,
                    "daily_loss_kill_r": daily_loss_kill_r,
                    "reason": f"SHORT sweep-reclaim of swing-H {sw_high:.5f} (wick {wick/atr14:.2f} ATR)",
                }

        # ── EQH/EQL Liquidity-Pool Filter ──────────────────────────────
        # Applied AFTER sweep detection, BEFORE signal return. Sweeps that
        # take out a cluster of equal highs/lows get full size; isolated-
        # pivot sweeps are downsized (or rejected in STRICT mode).
        # Fail-open: any error leaves sig unmodified.
        if sig is not None and EQH_EQL_FILTER_ENABLED \
                and _find_eqh_eql is not None and _is_sweeping_eqh_eql is not None:
            try:
                eqh_eql = _find_eqh_eql(
                    H, L,
                    lookback=50,
                    tolerance_atr=0.10,
                    min_touches=2,
                    atr=atr14,
                    closes=C,
                )
                # Only EQH clusters matter for SHORT (sweep of highs);
                # only EQL clusters matter for LONG (sweep of lows).
                want = "EQL" if sig["direction"] == "LONG" else "EQH"
                hit = _is_sweeping_eqh_eql(
                    sig["swept_level"],
                    eqh_eql,
                    tolerance_atr=0.20,
                    atr=atr14,
                    cluster_type=want,
                )
                if hit:
                    sig["eqh_eql_cluster"] = True
                    sig["size_mult"] = float(sig.get("size_mult", 1.0))
                    sig["reason"] += f" + {want}-cluster"
                else:
                    sig["eqh_eql_cluster"] = False
                    if EQH_EQL_STRICT:
                        log.debug("[%s] EQH/EQL STRICT reject — no cluster at %.5f",
                                  symbol, sig["swept_level"])
                        sig = None
                    else:
                        sig["size_mult"] = 0.6 * float(sig.get("size_mult", 1.0))
                        sig["reason"] += f" (no {want} cluster — 0.6× size)"
            except Exception as e:
                log.debug("[%s] EQH/EQL filter error (fail-open): %s", symbol, e)

        # Mark bar processed regardless (prevents repeated checks within a bar).
        self._last_bar_t[symbol] = bar_t
        return sig


# ════════════════════════════════════════════════════════════════════════
#  Standalone smoke test — `python3 agent/sweep_reclaim.py`
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """Build a synthetic 200-bar M15 frame containing one bullish sweep+reclaim,
    confirm the detector fires correctly."""
    np.random.seed(42)
    n = 200
    base = 100.0
    rng = np.cumsum(np.random.normal(0, 0.5, n))
    closes = base + rng
    highs = closes + np.abs(np.random.normal(0, 0.3, n))
    lows = closes - np.abs(np.random.normal(0, 0.3, n))
    opens = np.concatenate([[base], closes[:-1]])

    # Inject a swing-low at index 170 (inside the 20-bar lookback from sweep at 180),
    # then sweep+reclaim at index 180.
    swing_idx = 170
    lows[swing_idx] = 90.0
    highs[swing_idx] = 91.0
    closes[swing_idx] = 90.5
    opens[swing_idx] = 90.8
    # Ensure adjacent bars don't break the pivot
    for off in [-3, -2, -1, 1, 2, 3]:
        lows[swing_idx + off] = max(lows[swing_idx + off], 91.5)
        highs[swing_idx + off] = max(highs[swing_idx + off], 92.0)

    # The sweep bar at 180:
    lows[180] = 89.5            # wicks below 90.0
    closes[180] = 91.0          # reclaims above 90.0
    opens[180] = 90.0           # bullish candle (close > open)
    highs[180] = 91.2
    # Ensure no LOWER swing low appears between swing_idx and the sweep bar
    for k in range(swing_idx + 1, 180):
        lows[k] = max(lows[k], 91.5)

    times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows, "close": closes,
        "tick_volume": np.full(n, 1000.0),
    })

    class _FakeState:
        def get_candles(self, symbol, tf):
            if tf == 15:
                # Mimic the live frame: forming bar appended at the end.
                d2 = df.copy()
                # Add a forming bar
                new_row = pd.DataFrame([{
                    "time": times[-1] + pd.Timedelta("15min"),
                    "open": closes[-1], "high": closes[-1] + 0.1,
                    "low": closes[-1] - 0.1, "close": closes[-1],
                    "tick_volume": 500.0,
                }])
                return pd.concat([d2, new_row], ignore_index=True)
            elif tf == 60:
                # Provide H1 candles for ADX gate
                df_h1 = df.copy().iloc[::4].reset_index(drop=True)
                return df_h1
            return None

    strat = SweepReclaimStrategy(_FakeState())
    sig = strat.evaluate("TEST")
    print(f"signal at -2 bar: {sig}")
    # Move the "now" forward by truncating to before index 181 so -2 == 180
    # is already represented; the above call evaluates -2 of full frame (=199),
    # so the synthetic detection at 180 won't be reached. Instead, evaluate a
    # truncated frame that ends right after the sweep bar.
    df_short = df.iloc[:181].copy()  # 181 bars (0..180) + 1 forming = 182; i=-2 == 180 (sweep)
    class _S2:
        def get_candles(self, s, tf):
            if tf == 15:
                d2 = df_short.copy()
                new_row = pd.DataFrame([{
                    "time": df_short["time"].iloc[-1] + pd.Timedelta("15min"),
                    "open": df_short["close"].iloc[-1],
                    "high": df_short["close"].iloc[-1] + 0.1,
                    "low": df_short["close"].iloc[-1] - 0.1,
                    "close": df_short["close"].iloc[-1],
                    "tick_volume": 500.0,
                }])
                return pd.concat([d2, new_row], ignore_index=True)
            return df_short.iloc[::4].reset_index(drop=True) if tf == 60 else None
    strat2 = SweepReclaimStrategy(_S2())
    sig2 = strat2.evaluate("TEST")
    print(f"signal targeting bar 180 (the sweep): {sig2}")
    if sig2 and sig2["direction"] == "LONG" and abs(sig2["swept_level"] - 90.0) < 0.01:
        print("✓ SMOKE TEST PASSED")
    else:
        print("✗ SMOKE TEST FAILED — adjust synthetic bar geometry")
