#!/usr/bin/env python3 -B
"""
Wyckoff Spring + Upthrust Detector (Dragon bot).

Higher-conviction superset of sweep_reclaim. The sweep_reclaim detector fires
on any liquidity-sweep + reclaim of a swing extreme; Wyckoff additionally
requires:

  (1) Validated TRADING RANGE prior to the event:
        - ATR contraction (TR height <= RANGE_MAX_HEIGHT_ATR * ATR14(H1)),
        - min duration (>= RANGE_MIN_DURATION H1 bars),
        - low-volatility bodies (median |close-open|/ATR <= RANGE_BODY_ATR_MEDIAN).
  (2) Multi-touch SUPPORT/RESISTANCE level: >= MIN_TOUCHES wicks/closes within
      LEVEL_BAND_ATR * ATR14(H1) of the level BEFORE the spring/upthrust.
  (3) Optional TEST BAR: post-event lower-volume retest that holds above the
      spring low (or below the upthrust high). When TEST_REQUIRED=True, the
      test bar is the entry trigger; otherwise immediate reclaim is entry.

References:
  * Wyckoff "Method" (1931).
  * Hank Pruden — "The Three Skills of Top Trading".
  * Tom Williams — Volume Spread Analysis (VSA).
  * Anna Coulling — "Volume Price Analysis".
  * Bruce Fraser — Pruden-lineage Wyckoff webinars.

Strategy (single-bar event detector — NOT a state machine):
  For each symbol, look at the most recent CLOSED M15 bar (index -2 of the
  live frame). Validate H1 range + ADX + multi-touch level. Then:

  Bullish SPRING (LONG):
    1. M15 bar.low < tr_bot                       (probe below TR support)
    2. M15 bar.close >= tr_bot                    (reclaim back inside TR)
    3. (tr_bot - bar.low) >= SPRING_WICK_ATR_MIN * ATR14(M15)
    4. bar.close > bar.open                       (bullish reclaim candle)
    5. (optional) test bar within TEST_LOOKAHEAD_BARS with lower volume that
       holds above tr_bot - TEST_HOLD_BUFFER_ATR * ATR14(M15).

  Bearish UPTHRUST (SHORT): symmetric — probe above tr_top + reclaim back below.

  Levels emitted to executor (mirror SR shape for drop-in wiring):
    entry = bar.close (immediate reclaim) OR test.close (if TEST_REQUIRED)
    sl    = bar.low  - SL_BUFFER_ATR * ATR14(M15)   (LONG)
            bar.high + SL_BUFFER_ATR * ATR14(M15)   (SHORT)
    tp1   = entry +/- TP1_R * (entry - sl)
    tp2   = entry +/- TP2_R * (entry - sl)
    time_stop = TIME_STOP_BARS closed M15 bars (close if peak_R < 0.3)

API (mirrors SweepReclaimStrategy for drop-in wiring):
    strat = WyckoffSpringUpthrustStrategy(state)
    sig = strat.evaluate("XAUUSD")
    # sig is None, or:
    # {
    #   "direction": "LONG" | "SHORT",
    #   "mode": "SPRING" | "UPTHRUST",
    #   "entry": float, "sl": float, "tp1": float, "tp2": float,
    #   "swept_level": float, "wick_atr_mult": float, "atr14": float,
    #   "bar_time": pd.Timestamp,
    #   "reason": str,
    # }

The wiring agent should call evaluate(symbol) once per brain cycle (per closed
M15 bar). When sig is non-None, decide based on WYCKOFF_TRADE_LIVE config flag:
  - False: log the signal only (observation mode)
  - True : open trade via executor.open_trade_explicit with Wyckoff magic offset

Constraints honoured:
  * NEW file only — does not modify any existing module.
  * Closed bars only — reads index -2 (the live forming bar is never used).
  * Defensive: missing / short data -> return None.
  * Module-level helper functions are pure (no class state needed for testing).
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("dragon.wyckoff_spring")

# Optional per-symbol overrides / blacklist / whitelist — try-import so the
# detector keeps working even if these aren't defined in config (older builds,
# isolated tests, smoke tests).
try:
    from config import WYCKOFF_PARAM_OVERRIDES  # type: ignore
except Exception:
    WYCKOFF_PARAM_OVERRIDES = {}
try:
    from config import WYCKOFF_SYMBOL_BLACKLIST  # type: ignore
except Exception:
    WYCKOFF_SYMBOL_BLACKLIST = set()
try:
    from config import WYCKOFF_SYMBOL_WHITELIST  # type: ignore
except Exception:
    WYCKOFF_SYMBOL_WHITELIST = None  # None = all symbols allowed


# ════════════════════════════════════════════════════════════════════════
#  TUNABLE DEFAULTS — overridable via constructor params dict / per-symbol
# ════════════════════════════════════════════════════════════════════════
TF_PRIMARY            = 15        # M15 trigger frame
TF_RANGE              = 60        # H1 frame for trading-range validation

RANGE_LOOKBACK_BARS   = 48        # H1 bars (~2 days) used to detect the TR
RANGE_MIN_DURATION    = 20        # need >= N H1 bars of TR before the spring
RANGE_MAX_HEIGHT_ATR  = 4.0       # TR top-to-bottom <= N * ATR14(H1)
RANGE_BODY_ATR_MEDIAN = 0.6       # median |close-open|/ATR14(H1) <= N

LEVEL_BAND_ATR        = 0.20      # support/resistance band width (ATR mult)
MIN_TOUCHES           = 2         # >= N prior touches inside band

SPRING_WICK_ATR_MIN   = 0.30      # spring wick below level >= N * ATR14(M15)
SPRING_RECLAIM_PCT    = 1.00      # close must be back inside range (>= level)
UPTHRUST_WICK_ATR_MIN = 0.30      # symmetric

TEST_REQUIRED         = True      # require a test bar (vs immediate reclaim)
TEST_LOOKAHEAD_BARS   = 4         # M15 bars; test must appear within N bars
TEST_VOL_RATIO_MAX    = 0.70      # test bar volume <= N * spring bar volume
TEST_HOLD_BUFFER_ATR  = 0.10      # test extreme must hold within N*ATR

HTF_TREND_FILTER      = "BLOCK_AGAINST_DAILY"  # OFF | BLOCK_AGAINST_DAILY | STRICT
DAILY_EMA_PERIOD      = 50
ADX_REGIME_MAX        = 30        # H1 ADX must be <= N (no runaway trend)

COOLDOWN_BARS_AFTER   = 24        # M15 bars: suppress re-fire (6h)
SL_BUFFER_ATR         = 0.15      # stop = spring_low - N*ATR14(M15)
TP1_R                 = 1.5       # first partial @ +1.5R
TP2_R                 = 3.0       # runner @ +3R
TIME_STOP_BARS        = 16        # close if peak_R < 0.3 after 16 M15 bars (4h)
TIME_STOP_PEAK_R      = 0.3

MIN_M15_BARS          = 80
MIN_H1_BARS           = 60
ATR_PERIOD            = 14
ADX_PERIOD            = 14


# ════════════════════════════════════════════════════════════════════════
#  PURE HELPER FUNCTIONS — Wilder ATR / ADX / EMA / candle normalize
# ════════════════════════════════════════════════════════════════════════
def _atr(H, L, C, period=ATR_PERIOD):
    """Wilder ATR over a numpy array. Returns the latest scalar (or 0 on too-short)."""
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


def _atr_series(H, L, C, period=ATR_PERIOD):
    """Wilder ATR — returns full numpy series (NaN before warmup)."""
    n = len(H)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    tr = np.empty(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i-1]), abs(L[i] - C[i-1]))
    out[period] = tr[1:period+1].mean()
    k = 1.0 / period
    for i in range(period + 1, n):
        out[i] = out[i-1] * (1 - k) + tr[i] * k
    return out


def _adx(H, L, C, period=ADX_PERIOD):
    """Wilder ADX — returns latest scalar (or 0 on too-short)."""
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
    sm_plus = np.empty(n - 1)
    sm_minus = np.empty(n - 1)
    sm_tr = np.empty(n - 1)
    sm_plus[period-1] = plus_dm[:period].sum()
    sm_minus[period-1] = minus_dm[:period].sum()
    sm_tr[period-1] = tr[:period].sum()
    for i in range(period, n - 1):
        sm_plus[i] = sm_plus[i-1] - sm_plus[i-1] / period + plus_dm[i]
        sm_minus[i] = sm_minus[i-1] - sm_minus[i-1] / period + minus_dm[i]
        sm_tr[i] = sm_tr[i-1] - sm_tr[i-1] / period + tr[i]
    plus_di = 100 * sm_plus / np.where(sm_tr == 0, 1, sm_tr)
    minus_di = 100 * sm_minus / np.where(sm_tr == 0, 1, sm_tr)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = np.empty(n - 1)
    if 2 * period - 1 >= len(dx):
        return 0.0
    adx[2*period-1] = dx[period-1:2*period].mean()
    for i in range(2*period, n - 1):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return float(adx[-1])


def _ema(series, period):
    """EMA of a 1-D numpy/pandas series. Returns numpy array."""
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    k = 2.0 / (period + 1.0)
    # Seed with first value (or SMA of first `period` if available).
    if n >= period:
        out[period - 1] = arr[:period].mean()
        for i in range(period, n):
            out[i] = arr[i] * k + out[i-1] * (1 - k)
    else:
        out[0] = arr[0]
        for i in range(1, n):
            out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out


def _normalize_candles(df):
    """Normalize to time-indexed OHLC(+volume) DataFrame (UTC). None on bad input."""
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
#  PURE DETECTION HELPERS — testable without class instance
# ════════════════════════════════════════════════════════════════════════
def validate_trading_range(h1_window, atr_h1,
                           max_height_atr=RANGE_MAX_HEIGHT_ATR,
                           body_atr_median=RANGE_BODY_ATR_MEDIAN,
                           min_duration=RANGE_MIN_DURATION):
    """Return (tr_top, tr_bot, ok). ok=False if window isn't a valid TR.

    Validates:
      * window length >= min_duration
      * (tr_top - tr_bot) <= max_height_atr * atr_h1
      * median |close-open|/atr_h1 <= body_atr_median
    """
    if h1_window is None or len(h1_window) < min_duration:
        return None, None, False
    if atr_h1 is None or atr_h1 <= 0:
        return None, None, False
    tr_top = float(h1_window["high"].max())
    tr_bot = float(h1_window["low"].min())
    tr_height = tr_top - tr_bot
    if tr_height > max_height_atr * atr_h1:
        return tr_top, tr_bot, False
    body_ratio = float(((h1_window["close"] - h1_window["open"]).abs() / atr_h1).median())
    if body_ratio > body_atr_median:
        return tr_top, tr_bot, False
    return tr_top, tr_bot, True


def count_level_touches(h1_window, level, atr_h1, band_atr=LEVEL_BAND_ATR, side="low"):
    """Count how many H1 bars in the window touched the level within band.

    side="low"  -> count bars whose LOW is within band_atr*atr_h1 of level (support)
    side="high" -> count bars whose HIGH is within band_atr*atr_h1 of level (resistance)
    """
    if h1_window is None or len(h1_window) == 0 or atr_h1 <= 0:
        return 0
    band = band_atr * atr_h1
    if side == "low":
        return int(((h1_window["low"] - level).abs() <= band).sum())
    elif side == "high":
        return int(((h1_window["high"] - level).abs() <= band).sum())
    return 0


def detect_d1_trend(h1, ema_period=DAILY_EMA_PERIOD):
    """Resample H1->D1 and return ('up'|'down'|'flat', d1_close, ema_d1_last).

    Uses last D1 close vs D1 EMA50 + mean slope of last 3 EMA points.
    """
    if h1 is None or len(h1) < ema_period * 24 // 24:  # generous floor
        return "flat", np.nan, np.nan
    try:
        d1 = h1.resample("1D").agg({
            "open": "first", "high": "max", "low": "min", "close": "last"
        }).dropna()
        if len(d1) < ema_period + 3:
            return "flat", float(d1["close"].iloc[-1]) if len(d1) else np.nan, np.nan
        ema_d1 = _ema(d1["close"].values, ema_period)
        d1_close = float(d1["close"].iloc[-1])
        ema_last = float(ema_d1[-1])
        # Mean slope over last 3 points of EMA
        slope_window = ema_d1[-3:]
        if np.any(np.isnan(slope_window)):
            return "flat", d1_close, ema_last
        slope = float(np.diff(slope_window).mean())
        if d1_close > ema_last and slope > 0:
            return "up", d1_close, ema_last
        if d1_close < ema_last and slope < 0:
            return "down", d1_close, ema_last
        return "flat", d1_close, ema_last
    except Exception as e:
        log.debug("D1 trend detection failed: %s", e)
        return "flat", np.nan, np.nan


def find_test_bar(m15, i_event, direction, event_extreme, atr_m15,
                  lookahead=TEST_LOOKAHEAD_BARS,
                  vol_ratio_max=TEST_VOL_RATIO_MAX,
                  hold_buffer_atr=TEST_HOLD_BUFFER_ATR):
    """Locate a valid Wyckoff TEST bar following the spring/upthrust event.

    Returns the test bar (pd.Series) or None.

    direction="LONG":
      * test bar low >= event_extreme - hold_buffer_atr * atr_m15
      * test bar tick_volume <= vol_ratio_max * event_bar tick_volume
    direction="SHORT":
      * test bar high <= event_extreme + hold_buffer_atr * atr_m15
      * test bar tick_volume <= vol_ratio_max * event_bar tick_volume

    Uses CLOSED bars only — never inspects past index len(m15)-2.
    """
    if m15 is None:
        return None
    n = len(m15)
    if i_event + 1 >= n - 1:
        return None  # no closed bars after event yet
    event_bar = m15.iloc[i_event]
    event_vol = float(event_bar.get("tick_volume", 0.0)) if "tick_volume" in m15.columns else 0.0
    # Limit search range to CLOSED bars only -> [i_event+1, n-2]
    end = min(i_event + 1 + lookahead, n - 1)
    for k in range(i_event + 1, end):
        b = m15.iloc[k]
        if direction == "LONG":
            held = float(b["low"]) >= event_extreme - hold_buffer_atr * atr_m15
        elif direction == "SHORT":
            held = float(b["high"]) <= event_extreme + hold_buffer_atr * atr_m15
        else:
            return None
        if not held:
            return None  # break of extreme = test failed
        if event_vol > 0 and "tick_volume" in m15.columns:
            vol_ok = float(b["tick_volume"]) <= vol_ratio_max * event_vol
        else:
            vol_ok = True
        if vol_ok:
            return b
    return None


# ════════════════════════════════════════════════════════════════════════
#  WyckoffSpringUpthrustStrategy — drop-in detector class
# ════════════════════════════════════════════════════════════════════════
class WyckoffSpringUpthrustStrategy:
    """Per-symbol single-bar Wyckoff Spring/Upthrust detector.

    Mirrors SweepReclaimStrategy's API surface so brain.py wiring is a drop-in.
    Stateless across calls in the sense that the detector itself never relies
    on past evaluate() results — but maintains a per-symbol dedupe map
    (_last_bar_t) and cooldown map (_cooldown_until) so calls within the same
    M15 bar / within the cooldown window return None without recomputing.
    """

    def __init__(self, state, params=None):
        self.state = state
        p = params or {}
        self.range_lookback_bars   = int(p.get("RANGE_LOOKBACK_BARS", RANGE_LOOKBACK_BARS))
        self.range_min_duration    = int(p.get("RANGE_MIN_DURATION", RANGE_MIN_DURATION))
        self.range_max_height_atr  = float(p.get("RANGE_MAX_HEIGHT_ATR", RANGE_MAX_HEIGHT_ATR))
        self.range_body_atr_median = float(p.get("RANGE_BODY_ATR_MEDIAN", RANGE_BODY_ATR_MEDIAN))
        self.level_band_atr        = float(p.get("LEVEL_BAND_ATR", LEVEL_BAND_ATR))
        self.min_touches           = int(p.get("MIN_TOUCHES", MIN_TOUCHES))
        self.spring_wick_atr_min   = float(p.get("SPRING_WICK_ATR_MIN", SPRING_WICK_ATR_MIN))
        self.spring_reclaim_pct    = float(p.get("SPRING_RECLAIM_PCT", SPRING_RECLAIM_PCT))
        self.upthrust_wick_atr_min = float(p.get("UPTHRUST_WICK_ATR_MIN", UPTHRUST_WICK_ATR_MIN))
        self.test_required         = bool(p.get("TEST_REQUIRED", TEST_REQUIRED))
        self.test_lookahead_bars   = int(p.get("TEST_LOOKAHEAD_BARS", TEST_LOOKAHEAD_BARS))
        self.test_vol_ratio_max    = float(p.get("TEST_VOL_RATIO_MAX", TEST_VOL_RATIO_MAX))
        self.test_hold_buffer_atr  = float(p.get("TEST_HOLD_BUFFER_ATR", TEST_HOLD_BUFFER_ATR))
        self.htf_trend_filter      = str(p.get("HTF_TREND_FILTER", HTF_TREND_FILTER))
        self.daily_ema_period      = int(p.get("DAILY_EMA_PERIOD", DAILY_EMA_PERIOD))
        self.adx_regime_max        = float(p.get("ADX_REGIME_MAX", ADX_REGIME_MAX))
        self.cooldown_bars_after   = int(p.get("COOLDOWN_BARS_AFTER", COOLDOWN_BARS_AFTER))
        self.sl_buffer_atr         = float(p.get("SL_BUFFER_ATR", SL_BUFFER_ATR))
        self.tp1_r                 = float(p.get("TP1_R", TP1_R))
        self.tp2_r                 = float(p.get("TP2_R", TP2_R))
        self.time_stop_bars        = int(p.get("TIME_STOP_BARS", TIME_STOP_BARS))
        # Per-symbol dedupe + cooldown (the strategy is otherwise stateless).
        self._last_bar_t = {}
        self._cooldown_until = {}

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

    def _resolve_params(self, symbol):
        """Resolve per-symbol overrides from config.WYCKOFF_PARAM_OVERRIDES."""
        sym_ov = WYCKOFF_PARAM_OVERRIDES.get(symbol, {}) if WYCKOFF_PARAM_OVERRIDES else {}
        return {
            "SPRING_WICK_ATR_MIN":   float(sym_ov.get("SPRING_WICK_ATR_MIN",   self.spring_wick_atr_min)),
            "UPTHRUST_WICK_ATR_MIN": float(sym_ov.get("UPTHRUST_WICK_ATR_MIN", self.upthrust_wick_atr_min)),
            "TP1_R":                 float(sym_ov.get("TP1_R",                 self.tp1_r)),
            "TP2_R":                 float(sym_ov.get("TP2_R",                 self.tp2_r)),
            "SL_BUFFER_ATR":         float(sym_ov.get("SL_BUFFER_ATR",         self.sl_buffer_atr)),
            "MIN_TOUCHES":           int(sym_ov.get("MIN_TOUCHES",             self.min_touches)),
            "ADX_REGIME_MAX":        float(sym_ov.get("ADX_REGIME_MAX",        self.adx_regime_max)),
            "TEST_REQUIRED":         bool(sym_ov.get("TEST_REQUIRED",          self.test_required)),
        }

    def evaluate(self, symbol):
        """Inspect the most-recent closed M15 bar; return signal dict or None."""
        # ── Surgical kill switches ─────────────────────────────────────
        if symbol in WYCKOFF_SYMBOL_BLACKLIST:
            return None
        if WYCKOFF_SYMBOL_WHITELIST is not None and symbol not in WYCKOFF_SYMBOL_WHITELIST:
            return None

        params = self._resolve_params(symbol)

        # ── Data fetch + normalize ─────────────────────────────────────
        m15 = _normalize_candles(self._get_m15(symbol))
        h1  = _normalize_candles(self._get_h1(symbol))
        if m15 is None or h1 is None:
            return None
        if len(m15) < MIN_M15_BARS or len(h1) < MIN_H1_BARS:
            return None

        # ── Bar selection ─────────────────────────────────────────────
        # The last CLOSED M15 bar = index -2 (anchor bar for dedupe + cooldown).
        # When TEST_REQUIRED=True, the SPRING event itself must have happened
        # earlier so the test bar can print AFTER it (within TEST_LOOKAHEAD_BARS)
        # — both event AND test must be CLOSED bars. So:
        #   * event-candidate index = len(m15) - 2 - test_lookahead_bars
        #   * test bars come from [event+1 .. -2]  (all closed)
        # When TEST_REQUIRED=False, event-candidate index = -2 (immediate reclaim).
        n_total = len(m15)
        anchor_i = n_total - 2          # used for dedupe + cooldown
        if anchor_i < 4:
            return None
        anchor_t = m15.index[anchor_i]

        # Per-bar dedupe on the anchor bar (the brain calls us once per closed
        # M15 bar; we only fire at most one signal per anchor).
        if self._last_bar_t.get(symbol) == anchor_t:
            return None

        # Cooldown gate (per-symbol, set on prior signal).
        cooldown_t = self._cooldown_until.get(symbol)
        if cooldown_t is not None and anchor_t < cooldown_t:
            return None

        # Choose the event-candidate bar.
        if params["TEST_REQUIRED"]:
            # Event must precede the test bar(s). The test search window is
            # [event+1 .. event+test_lookahead_bars], all of which must be CLOSED
            # (i.e. <= n_total-2). So event_i_max = n_total-2-1 (need at least
            # one test slot). We anchor on the bar that's exactly
            # `test_lookahead_bars` back from the latest closed bar so the test
            # window is fully available.
            event_i = anchor_i - self.test_lookahead_bars
            if event_i < 4:
                return None
        else:
            event_i = anchor_i

        bar = m15.iloc[event_i]
        bar_t = m15.index[event_i]

        # ── ATR(M15) + ATR(H1) computed at event_i (no look-ahead) ────
        H_m15 = m15["high"].values[:event_i + 1]
        L_m15 = m15["low"].values[:event_i + 1]
        C_m15 = m15["close"].values[:event_i + 1]
        atr_m15 = _atr(H_m15, L_m15, C_m15, ATR_PERIOD)
        atr_h1  = _atr(h1["high"].values, h1["low"].values, h1["close"].values, ATR_PERIOD)
        if atr_m15 <= 0 or atr_h1 <= 0:
            return None

        # ── (A) Validate H1 trading range over RANGE_LOOKBACK_BARS ────
        h1_window = h1.iloc[-self.range_lookback_bars:]
        tr_top, tr_bot, range_ok = validate_trading_range(
            h1_window, atr_h1,
            max_height_atr=self.range_max_height_atr,
            body_atr_median=self.range_body_atr_median,
            min_duration=self.range_min_duration,
        )
        if not range_ok:
            return None

        # ── ADX regime gate (H1 ADX <= max) ───────────────────────────
        adx_h1 = _adx(h1["high"].values, h1["low"].values, h1["close"].values, ADX_PERIOD)
        if adx_h1 > params["ADX_REGIME_MAX"]:
            return None

        # ── (B) Multi-touch level counting ────────────────────────────
        touches_sup = count_level_touches(h1_window, tr_bot, atr_h1,
                                          band_atr=self.level_band_atr, side="low")
        touches_res = count_level_touches(h1_window, tr_top, atr_h1,
                                          band_atr=self.level_band_atr, side="high")
        spring_candidate   = touches_sup >= params["MIN_TOUCHES"]
        upthrust_candidate = touches_res >= params["MIN_TOUCHES"]

        # ── (C) HTF trend filter ──────────────────────────────────────
        if self.htf_trend_filter in ("BLOCK_AGAINST_DAILY", "STRICT"):
            trend, _, _ = detect_d1_trend(h1, self.daily_ema_period)
            if self.htf_trend_filter == "BLOCK_AGAINST_DAILY":
                # Spring (LONG) is invalid in a D1 downtrend.
                # Upthrust (SHORT) is invalid in a D1 uptrend.
                if spring_candidate and trend == "down":
                    spring_candidate = False
                if upthrust_candidate and trend == "up":
                    upthrust_candidate = False
            elif self.htf_trend_filter == "STRICT":
                # Spring needs accumulation context (NOT downtrend AND ideally up/flat).
                if spring_candidate and trend != "up":
                    spring_candidate = False
                if upthrust_candidate and trend != "down":
                    upthrust_candidate = False

        if not spring_candidate and not upthrust_candidate:
            return None

        # ── (D) Check candidate M15 bar for SPRING (LONG) ─────────────
        sig = None
        if spring_candidate:
            wick = tr_bot - float(bar["low"])
            if (
                float(bar["low"]) < tr_bot
                and float(bar["close"]) >= tr_bot * self.spring_reclaim_pct
                and wick >= params["SPRING_WICK_ATR_MIN"] * atr_m15
                and float(bar["close"]) > float(bar["open"])
            ):
                # Determine entry: test bar (if required) or immediate reclaim.
                entry_price = float(bar["close"])
                event_extreme = float(bar["low"])
                if params["TEST_REQUIRED"]:
                    test = find_test_bar(
                        m15, event_i, "LONG", event_extreme, atr_m15,
                        lookahead=self.test_lookahead_bars,
                        vol_ratio_max=self.test_vol_ratio_max,
                        hold_buffer_atr=self.test_hold_buffer_atr,
                    )
                    if test is None:
                        # No valid test bar yet — return None (caller re-checks next bar).
                        self._last_bar_t[symbol] = anchor_t
                        return None
                    entry_price = float(test["close"])
                sl = event_extreme - params["SL_BUFFER_ATR"] * atr_m15
                risk = entry_price - sl
                if risk <= 0:
                    self._last_bar_t[symbol] = anchor_t
                    return None
                sig = {
                    "direction": "LONG",
                    "mode": "SPRING",
                    "entry": entry_price,
                    "sl": sl,
                    "tp1": entry_price + params["TP1_R"] * risk,
                    "tp2": entry_price + params["TP2_R"] * risk,
                    "swept_level": tr_bot,
                    "tr_top": tr_top,
                    "tr_bot": tr_bot,
                    "wick_atr_mult": wick / atr_m15,
                    "atr14": atr_m15,
                    "adx_h1": adx_h1,
                    "touches_support": touches_sup,
                    "bar_time": bar_t,
                    "time_stop_bars": self.time_stop_bars,
                    "reason": (
                        f"SPRING reclaim of TR-bot {tr_bot:.5f} "
                        f"(wick {wick/atr_m15:.2f} ATR, touches={touches_sup}, ADX {adx_h1:.1f})"
                    ),
                }

        # ── (D') Check candidate M15 bar for UPTHRUST (SHORT) ─────────
        if sig is None and upthrust_candidate:
            wick = float(bar["high"]) - tr_top
            if (
                float(bar["high"]) > tr_top
                and float(bar["close"]) <= tr_top
                and wick >= params["UPTHRUST_WICK_ATR_MIN"] * atr_m15
                and float(bar["close"]) < float(bar["open"])
            ):
                entry_price = float(bar["close"])
                event_extreme = float(bar["high"])
                if params["TEST_REQUIRED"]:
                    test = find_test_bar(
                        m15, event_i, "SHORT", event_extreme, atr_m15,
                        lookahead=self.test_lookahead_bars,
                        vol_ratio_max=self.test_vol_ratio_max,
                        hold_buffer_atr=self.test_hold_buffer_atr,
                    )
                    if test is None:
                        self._last_bar_t[symbol] = anchor_t
                        return None
                    entry_price = float(test["close"])
                sl = event_extreme + params["SL_BUFFER_ATR"] * atr_m15
                risk = sl - entry_price
                if risk <= 0:
                    self._last_bar_t[symbol] = anchor_t
                    return None
                sig = {
                    "direction": "SHORT",
                    "mode": "UPTHRUST",
                    "entry": entry_price,
                    "sl": sl,
                    "tp1": entry_price - params["TP1_R"] * risk,
                    "tp2": entry_price - params["TP2_R"] * risk,
                    "swept_level": tr_top,
                    "tr_top": tr_top,
                    "tr_bot": tr_bot,
                    "wick_atr_mult": wick / atr_m15,
                    "atr14": atr_m15,
                    "adx_h1": adx_h1,
                    "touches_resistance": touches_res,
                    "bar_time": bar_t,
                    "time_stop_bars": self.time_stop_bars,
                    "reason": (
                        f"UPTHRUST reclaim of TR-top {tr_top:.5f} "
                        f"(wick {wick/atr_m15:.2f} ATR, touches={touches_res}, ADX {adx_h1:.1f})"
                    ),
                }

        # Mark anchor bar processed regardless (prevents repeated checks within a bar).
        self._last_bar_t[symbol] = anchor_t

        # Arm cooldown when a signal fires.
        if sig is not None:
            self._cooldown_until[symbol] = anchor_t + pd.Timedelta(minutes=15 * self.cooldown_bars_after)
            # Stamp the actual event time on the signal for journal/dashboard.
            sig["event_bar_time"] = bar_t

        return sig


# ════════════════════════════════════════════════════════════════════════
#  Standalone smoke test — `python3 -B agent/wyckoff_spring.py`
# ════════════════════════════════════════════════════════════════════════
def _build_synthetic_spring_frame(n_h1=60, atr_target=1.0, seed=42):
    """Build synthetic M15 + H1 frames containing a valid Wyckoff Spring.

    Layout:
      * 60 H1 bars of tight consolidation between support ~99.5 and resistance ~101.5.
        Multiple touches of support at index 10, 25, 40, 55 H1 bars.
      * M15 has 4x the bars (~240). The last *closed* M15 bar = spring event:
        wicks below support (low ~99.0), reclaims (close ~100.0), bullish.
      * One additional M15 bar after the spring = the test bar (holds above
        99.0, low volume).
      * One more M15 bar = the forming bar (never inspected).
    Returns (m15_df, h1_df).
    """
    np.random.seed(seed)

    # ── H1 frame ──────────────────────────────────────────────────────
    n = n_h1
    support = 99.5
    resistance = 101.5
    mid = (support + resistance) / 2.0

    # Random walk inside the range with low amplitude (forces ATR contraction).
    # Random oscillation inside the range. Bodies are tight; wicks wider so
    # ATR(H1) ends up ~0.5+ (so 4*ATR >= TR-height ~ 1.96 and the range gate
    # passes). Bodies clamped <0.3 to keep median body/ATR <= 0.6.
    closes_h1 = mid + 0.3 * np.sin(np.linspace(0, 6 * np.pi, n)) + 0.05 * np.random.randn(n)
    opens_h1 = np.concatenate([[mid], closes_h1[:-1]])
    highs_h1 = np.maximum(opens_h1, closes_h1) + 0.30 + 0.15 * np.abs(np.random.randn(n))
    lows_h1  = np.minimum(opens_h1, closes_h1) - 0.30 - 0.15 * np.abs(np.random.randn(n))

    # Force multi-touches of support at specific bars
    for idx in [10, 25, 40, 55]:
        lows_h1[idx] = support + 0.02  # within band
        highs_h1[idx] = max(highs_h1[idx], lows_h1[idx] + 0.5)
    # Force multi-touches of resistance.
    for idx in [15, 30, 45]:
        highs_h1[idx] = resistance - 0.02
        lows_h1[idx] = min(lows_h1[idx], highs_h1[idx] - 0.5)
    # Clamp body magnitudes; keep median |close-open|/atr <= 0.6.
    for i in range(n):
        if abs(closes_h1[i] - opens_h1[i]) > 0.2:
            mid_co = (closes_h1[i] + opens_h1[i]) / 2.0
            closes_h1[i] = mid_co + 0.05
            opens_h1[i] = mid_co - 0.05
        highs_h1[i] = max(highs_h1[i], opens_h1[i], closes_h1[i])
        lows_h1[i] = min(lows_h1[i], opens_h1[i], closes_h1[i])

    times_h1 = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    h1 = pd.DataFrame({
        "time": times_h1,
        "open": opens_h1, "high": highs_h1, "low": lows_h1, "close": closes_h1,
        "tick_volume": np.full(n, 500.0),
    })

    # ── M15 frame: 4x the bars + 3 extra (spring + test + forming) ────
    n_m15 = n * 4
    # Map H1 closes to flat M15 bars then inject spring at end.
    closes_m15 = np.repeat(closes_h1, 4) + 0.02 * np.random.randn(n_m15)
    opens_m15 = np.concatenate([[closes_m15[0]], closes_m15[:-1]])
    highs_m15 = np.maximum(opens_m15, closes_m15) + 0.05
    lows_m15  = np.minimum(opens_m15, closes_m15) - 0.05

    # Append 6 more M15 bars so the SPRING sits at index -6 (event_i =
    # anchor_i - test_lookahead_bars = -2 - 4 = -6) with a valid TEST at -5:
    #   idx n_m15+0 = SPRING bar (wicks below 99.5, reclaims to 100.0, bullish)
    #   idx n_m15+1 = TEST bar (low volume, holds above 99.0)
    #   idx n_m15+2..4 = filler closed bars (continued reclaim — held above)
    #   idx n_m15+5 = forming bar (never inspected; -1 in live frame)
    spring_o = 99.6
    spring_l = 99.0   # wick 0.5 below support (well above min)
    spring_c = 100.0  # reclaim, bullish
    spring_h = 100.1

    test_o = 100.0
    test_l = 99.45    # holds above (support - 0.10*atr) easily
    test_c = 100.1
    test_h = 100.2

    # Three filler closed bars between test and forming — all hold above
    # support so they don't break the structural thesis.
    filler_o = [100.1, 100.15, 100.2]
    filler_h = [100.25, 100.3, 100.35]
    filler_l = [99.95, 100.0, 100.05]
    filler_c = [100.15, 100.2, 100.25]

    forming_o = 100.25
    forming_l = 100.15
    forming_c = 100.3
    forming_h = 100.35

    opens_m15  = np.concatenate([opens_m15,  [spring_o, test_o] + filler_o + [forming_o]])
    highs_m15  = np.concatenate([highs_m15,  [spring_h, test_h] + filler_h + [forming_h]])
    lows_m15   = np.concatenate([lows_m15,   [spring_l, test_l] + filler_l + [forming_l]])
    closes_m15 = np.concatenate([closes_m15, [spring_c, test_c] + filler_c + [forming_c]])

    n_total = n_m15 + 6
    times_m15 = pd.date_range("2026-01-01", periods=n_total, freq="15min", tz="UTC")
    # Test bar has LOW volume (< 0.7 * spring volume).
    vol_m15 = np.concatenate([
        np.full(n_m15, 800.0),
        [1200.0, 400.0, 600.0, 600.0, 600.0, 500.0],
    ])
    m15 = pd.DataFrame({
        "time": times_m15,
        "open": opens_m15, "high": highs_m15, "low": lows_m15, "close": closes_m15,
        "tick_volume": vol_m15,
    })
    return m15, h1


def _build_synthetic_upthrust_frame(n_h1=60, seed=99):
    """Mirror of spring frame — upthrust at end."""
    np.random.seed(seed)
    n = n_h1
    support = 99.5
    resistance = 101.5
    mid = (support + resistance) / 2.0

    closes_h1 = mid + 0.3 * np.cos(np.linspace(0, 6 * np.pi, n)) + 0.05 * np.random.randn(n)
    opens_h1 = np.concatenate([[mid], closes_h1[:-1]])
    # Match the spring frame: wider wicks for ATR(H1) ~ 0.5+
    highs_h1 = np.maximum(opens_h1, closes_h1) + 0.30 + 0.15 * np.abs(np.random.randn(n))
    lows_h1  = np.minimum(opens_h1, closes_h1) - 0.30 - 0.15 * np.abs(np.random.randn(n))

    for idx in [10, 25, 40, 55]:
        highs_h1[idx] = resistance - 0.02
        lows_h1[idx] = min(lows_h1[idx], highs_h1[idx] - 0.5)
    for idx in [15, 30, 45]:
        lows_h1[idx] = support + 0.02
        highs_h1[idx] = max(highs_h1[idx], lows_h1[idx] + 0.5)
    for i in range(n):
        if abs(closes_h1[i] - opens_h1[i]) > 0.2:
            mid_co = (closes_h1[i] + opens_h1[i]) / 2.0
            closes_h1[i] = mid_co + 0.05
            opens_h1[i] = mid_co - 0.05
        highs_h1[i] = max(highs_h1[i], opens_h1[i], closes_h1[i])
        lows_h1[i] = min(lows_h1[i], opens_h1[i], closes_h1[i])

    times_h1 = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    h1 = pd.DataFrame({
        "time": times_h1,
        "open": opens_h1, "high": highs_h1, "low": lows_h1, "close": closes_h1,
        "tick_volume": np.full(n, 500.0),
    })

    n_m15 = n * 4
    closes_m15 = np.repeat(closes_h1, 4) + 0.02 * np.random.randn(n_m15)
    opens_m15 = np.concatenate([[closes_m15[0]], closes_m15[:-1]])
    highs_m15 = np.maximum(opens_m15, closes_m15) + 0.05
    lows_m15  = np.minimum(opens_m15, closes_m15) - 0.05

    upthrust_o = 101.4
    upthrust_h = 102.0    # wick 0.5 above resistance
    upthrust_c = 101.0    # reclaim back below, bearish
    upthrust_l = 100.9

    test_o = 101.0
    test_h = 101.55       # holds below (resistance + 0.10*atr) easily
    test_c = 100.9
    test_l = 100.8

    # Filler bars maintaining the bearish bias
    filler_o = [100.9, 100.85, 100.8]
    filler_h = [101.0,  100.95, 100.9]
    filler_l = [100.7,  100.65, 100.6]
    filler_c = [100.85, 100.8,  100.75]

    forming_o = 100.75
    forming_h = 100.85
    forming_c = 100.7
    forming_l = 100.65

    opens_m15  = np.concatenate([opens_m15,  [upthrust_o, test_o] + filler_o + [forming_o]])
    highs_m15  = np.concatenate([highs_m15,  [upthrust_h, test_h] + filler_h + [forming_h]])
    lows_m15   = np.concatenate([lows_m15,   [upthrust_l, test_l] + filler_l + [forming_l]])
    closes_m15 = np.concatenate([closes_m15, [upthrust_c, test_c] + filler_c + [forming_c]])

    n_total = n_m15 + 6
    times_m15 = pd.date_range("2026-01-01", periods=n_total, freq="15min", tz="UTC")
    vol_m15 = np.concatenate([
        np.full(n_m15, 800.0),
        [1200.0, 400.0, 600.0, 600.0, 600.0, 500.0],
    ])
    m15 = pd.DataFrame({
        "time": times_m15,
        "open": opens_m15, "high": highs_m15, "low": lows_m15, "close": closes_m15,
        "tick_volume": vol_m15,
    })
    return m15, h1


class _FakeState:
    def __init__(self, m15_df, h1_df):
        self._m15 = m15_df
        self._h1 = h1_df

    def get_candles(self, symbol, tf):
        if tf == 15:
            return self._m15.copy()
        if tf == 60:
            return self._h1.copy()
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    failures = []

    # ── Test 1: SPRING with test bar should fire LONG ─────────────────
    m15, h1 = _build_synthetic_spring_frame()
    state = _FakeState(m15, h1)
    strat = WyckoffSpringUpthrustStrategy(state)
    sig = strat.evaluate("TEST")
    print(f"[T1 SPRING+test]  signal: {sig}")
    if sig is None:
        failures.append("T1: expected SPRING LONG signal, got None")
    elif sig["direction"] != "LONG" or sig["mode"] != "SPRING":
        failures.append(f"T1: expected LONG/SPRING, got {sig['direction']}/{sig['mode']}")
    elif abs(sig["swept_level"] - 99.5) > 0.3:
        failures.append(f"T1: swept_level {sig['swept_level']} not near 99.5")
    elif sig["sl"] >= sig["entry"]:
        failures.append(f"T1: SL {sig['sl']} >= entry {sig['entry']}")
    elif sig["tp1"] <= sig["entry"] or sig["tp2"] <= sig["tp1"]:
        failures.append(f"T1: TP order wrong tp1={sig['tp1']} tp2={sig['tp2']} entry={sig['entry']}")
    else:
        print("    PASS: SPRING LONG fires with correct geometry")

    # ── Test 2: UPTHRUST with test bar should fire SHORT ──────────────
    m15u, h1u = _build_synthetic_upthrust_frame()
    state2 = _FakeState(m15u, h1u)
    strat2 = WyckoffSpringUpthrustStrategy(state2)
    sig2 = strat2.evaluate("TEST2")
    print(f"[T2 UPTHRUST+test] signal: {sig2}")
    if sig2 is None:
        failures.append("T2: expected UPTHRUST SHORT signal, got None")
    elif sig2["direction"] != "SHORT" or sig2["mode"] != "UPTHRUST":
        failures.append(f"T2: expected SHORT/UPTHRUST, got {sig2['direction']}/{sig2['mode']}")
    elif sig2["sl"] <= sig2["entry"]:
        failures.append(f"T2: SL {sig2['sl']} <= entry {sig2['entry']}")
    elif sig2["tp1"] >= sig2["entry"] or sig2["tp2"] >= sig2["tp1"]:
        failures.append(f"T2: TP order wrong tp1={sig2['tp1']} tp2={sig2['tp2']} entry={sig2['entry']}")
    else:
        print("    PASS: UPTHRUST SHORT fires with correct geometry")

    # ── Test 3: dedupe — calling twice on the same frame yields None ──
    sig3 = strat.evaluate("TEST")
    print(f"[T3 dedupe]        signal: {sig3}")
    if sig3 is not None:
        failures.append("T3: expected None on repeat-evaluate (same bar dedupe)")
    else:
        print("    PASS: dedupe returns None on second call")

    # ── Test 4: empty / None state returns None gracefully ────────────
    strat4 = WyckoffSpringUpthrustStrategy(None)
    sig4 = strat4.evaluate("ANY")
    print(f"[T4 none state]    signal: {sig4}")
    if sig4 is not None:
        failures.append("T4: expected None when state=None")
    else:
        print("    PASS: graceful None on missing state")

    # ── Test 5: validate_trading_range pure helper ────────────────────
    m15t, h1t = _build_synthetic_spring_frame()
    h1n = _normalize_candles(h1t)
    atr_h1 = _atr(h1n["high"].values, h1n["low"].values, h1n["close"].values)
    tr_top, tr_bot, ok = validate_trading_range(h1n.iloc[-48:], atr_h1)
    print(f"[T5 TR validate]   top={tr_top:.4f} bot={tr_bot:.4f} ok={ok}")
    if not ok:
        failures.append("T5: expected TR to validate True on synthetic consolidation")
    elif tr_top <= tr_bot:
        failures.append("T5: tr_top <= tr_bot")
    else:
        print("    PASS: TR validates correctly")

    # ── Test 6: find_test_bar locates the test bar ────────────────────
    m15n = _normalize_candles(m15t)
    # Synthetic frame layout: [...base, SPRING, TEST, filler, filler, filler, forming]
    # so spring is at index -6, test at -5.
    spring_idx = len(m15n) - 6
    atr_m15v = _atr(m15n["high"].values, m15n["low"].values, m15n["close"].values)
    test = find_test_bar(m15n, spring_idx, "LONG",
                         float(m15n["low"].iloc[spring_idx]), atr_m15v)
    print(f"[T6 find_test_bar] test bar: {None if test is None else test.to_dict()}")
    if test is None:
        failures.append("T6: expected to find a test bar after spring event")
    else:
        print("    PASS: test bar found")

    # ── Test 7: TEST_REQUIRED=False should still fire on reclaim alone ─
    # Build a fresh frame where the spring sits at -2 (no test+filler trail).
    m15_t7, h1_t7 = _build_synthetic_spring_frame()
    # Truncate the trailing test+filler bars so the spring (originally at -6)
    # becomes the -2 anchor. Spring is the 1st of 6 trailing bars => keep up
    # to index -5 then re-append a forming bar.
    m15_t7 = m15_t7.iloc[:-5].copy().reset_index(drop=True)
    # Append a single forming bar so anchor (-2) = spring.
    last = m15_t7.iloc[-1]
    forming = pd.DataFrame([{
        "time": last["time"] + pd.Timedelta("15min"),
        "open": last["close"], "high": last["close"] + 0.1,
        "low": last["close"] - 0.1, "close": last["close"] + 0.05,
        "tick_volume": 500.0,
    }])
    m15_t7 = pd.concat([m15_t7, forming], ignore_index=True)
    strat7 = WyckoffSpringUpthrustStrategy(_FakeState(m15_t7, h1_t7),
                                           params={"TEST_REQUIRED": False})
    sig7 = strat7.evaluate("TEST7")
    print(f"[T7 no test req]   signal: {None if sig7 is None else sig7['mode']}")
    if sig7 is None:
        failures.append("T7: expected SPRING signal with TEST_REQUIRED=False")
    else:
        print("    PASS: fires without test bar requirement")

    # ── Summary ────────────────────────────────────────────────────────
    print()
    if failures:
        print(f"FAILED ({len(failures)} failures):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    else:
        print("ALL SMOKE TESTS PASSED")
        raise SystemExit(0)
