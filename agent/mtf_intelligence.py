"""
Dragon Trader -- Multi-Timeframe Market Intelligence Engine.

Analyzes M1/M5/M15/H1 candles simultaneously to produce:
  - Confluence score (0-4): how many TFs agree on direction
  - Entry quality score (0-100): composite multi-TF entry rating
  - Optimal SL: dynamic stop-loss from swing structure, not just ATR
  - Optimal TP: dynamic take-profit from resistance/support levels
  - Exit urgency (0-1.0): real-time multi-TF exit signal
  - Regime classification: trending/ranging/volatile/breakout

Called every brain cycle (~1s) per symbol. Production-grade:
NaN guards on every array access, fallback defaults, try/except
around every TF so one bad TF never crashes the whole analysis.
"""
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("dragon.mtf")

# Minimum bars required per timeframe
_MIN_BARS = {1: 20, 5: 30, 15: 50, 60: 80}

# =====================================================================
#  NUMPY INDICATOR HELPERS (self-contained, no external deps)
# =====================================================================

def _ema(a, p):
    """Exponential moving average."""
    if len(a) < 1:
        return np.array([0.0])
    al = 2.0 / (p + 1)
    o = np.empty_like(a, dtype=np.float64)
    o[0] = a[0]
    for i in range(1, len(a)):
        o[i] = al * a[i] + (1 - al) * o[i - 1]
    return o


def _atr(h, l, c, p):
    """Average true range."""
    n = len(c)
    if n < 2:
        return np.array([max(h[0] - l[0], 0.0001)])
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return _ema(tr, p)


def _rsi(c, p=14):
    """Relative strength index."""
    if len(c) < p + 1:
        return np.full(len(c), 50.0)
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0, d, 0.0)
    lo = np.where(d < 0, -d, 0.0)
    eg = _ema(g, p)
    el = _ema(lo, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(el > 0, eg / el, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(c, fast=12, slow=26, sig=9):
    """MACD line, signal, histogram."""
    ml = _ema(c, fast) - _ema(c, slow)
    sl = _ema(ml, sig)
    return ml, sl, ml - sl


def _adx(h, l, c, p=14):
    """Average directional index. Returns (adx_array, di_plus, di_minus)."""
    n = len(c)
    if n < p + 2:
        return np.full(n, 25.0), np.full(n, 0.0), np.full(n, 0.0)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
    sm_plus = _ema(plus_dm, p)
    sm_minus = _ema(minus_dm, p)
    atr_v = _atr(h, l, c, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        di_p = np.where(atr_v > 0, 100.0 * sm_plus / atr_v, 0.0)
        di_m = np.where(atr_v > 0, 100.0 * sm_minus / atr_v, 0.0)
        dsum = di_p + di_m
        dx = np.where(dsum > 0, 100.0 * np.abs(di_p - di_m) / dsum, 0.0)
    adx_arr = _ema(dx, p)
    return adx_arr, di_p, di_m


def _supertrend(h, l, c, fac=2.5, alen=10):
    """SuperTrend indicator. Returns (st_line, direction: +1 bull, -1 bear)."""
    n = len(c)
    if n < alen + 2:
        return np.full(n, c[-1] if n > 0 else 0.0), np.ones(n, dtype=int)
    av = _atr(h, l, c, alen)
    hl2 = (h + l) / 2.0
    upper = hl2 + fac * av
    lower = hl2 - fac * av
    st = np.zeros(n)
    d = np.ones(n, dtype=int)
    st[0] = upper[0]
    for i in range(1, n):
        plb = lower[i - 1]
        pub = upper[i - 1]
        if lower[i] < plb:
            lower[i] = plb
        if upper[i] > pub:
            upper[i] = pub
        ps = st[i - 1]
        if ps == pub:
            if c[i] > upper[i]:
                st[i] = lower[i]; d[i] = 1
            else:
                st[i] = upper[i]; d[i] = -1
        else:
            if c[i] < lower[i]:
                st[i] = upper[i]; d[i] = -1
            else:
                st[i] = lower[i]; d[i] = 1
    return st, d


def _safe(arr, idx, default=0.0):
    """Safe array access with NaN guard."""
    if arr is None or idx < 0 or idx >= len(arr):
        return default
    v = arr[idx]
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return default
    return float(v)


def _safe_slice(arr, start, end):
    """Safe array slice with bounds check."""
    if arr is None or len(arr) == 0:
        return np.array([])
    start = max(0, start)
    end = min(len(arr), end)
    if start >= end:
        return np.array([])
    return arr[start:end]


# =====================================================================
#  TF ANALYSIS CONTAINERS
# =====================================================================

class _TFResult:
    """Analysis result for a single timeframe."""
    __slots__ = ("direction", "strength", "detail")

    def __init__(self, direction="FLAT", strength=0.0, detail=None):
        self.direction = direction        # LONG / SHORT / FLAT
        self.strength = float(strength)   # 0.0 - 1.0
        self.detail = detail or {}


# =====================================================================
#  MTF INTELLIGENCE ENGINE
# =====================================================================

class MTFIntelligence:
    """Multi-Timeframe Market Intelligence for Dragon Trader.

    Usage:
        mtf = MTFIntelligence(shared_state)
        result = mtf.analyze("XAUUSD")
        # result["confluence"]     -> 0-4
        # result["entry_quality"]  -> 0-100
        # result["optimal_sl"]     -> price distance
        # result["optimal_tp"]     -> price distance
        # result["exit_urgency"]   -> 0.0-1.0
        # result["regime"]         -> "trending" / "ranging" / "volatile" / "breakout"
    """

    # Cache TTL per symbol (avoid recomputing every 1s call when data hasn't changed)
    _CACHE_TTL = 1.5  # seconds

    def __init__(self, state):
        self.state = state
        self._cache = {}  # symbol -> (result_dict, timestamp)

    # -----------------------------------------------------------------
    #  PUBLIC API
    # -----------------------------------------------------------------

    def analyze(self, symbol: str) -> dict:
        """Full multi-timeframe analysis for a symbol.

        Returns dict with all fields populated. Never raises -- returns
        safe defaults on any failure.
        """
        # Check cache
        cached = self._cache.get(symbol)
        if cached:
            result, ts = cached
            if time.time() - ts < self._CACHE_TTL:
                return result

        default = self._default_result()
        try:
            result = self._analyze_impl(symbol)
        except Exception as e:
            log.error("[%s] MTF analyze failed: %s", symbol, e)
            result = default

        self._cache[symbol] = (result, time.time())
        return result

    # -----------------------------------------------------------------
    #  CORE IMPLEMENTATION
    # -----------------------------------------------------------------

    def _analyze_impl(self, symbol: str) -> dict:
        """Internal analysis -- may raise."""
        # Load all TF candles
        candles = {}
        for tf in [1, 5, 15, 60]:
            df = self.state.get_candles(symbol, tf)
            if df is not None and len(df) >= _MIN_BARS.get(tf, 20):
                candles[tf] = df

        if 60 not in candles:
            # H1 is mandatory -- can't do anything without it
            return self._default_result()

        # ---------- Per-TF analysis ----------
        h1 = self._analyze_h1(candles.get(60))
        m15 = self._analyze_m15(candles.get(15))
        m5 = self._analyze_m5(candles.get(5))
        m1 = self._analyze_m1(candles.get(1))

        # ---------- Confluence ----------
        dirs = [h1.direction, m15.direction, m5.direction, m1.direction]
        confluence, dominant_dir = self._compute_confluence(dirs)

        # ---------- Regime ----------
        regime = self._detect_regime(candles.get(60), h1)

        # ---------- Entry Quality Score ----------
        entry_quality = self._compute_entry_quality(h1, m15, m5, m1, dominant_dir)

        # ---------- Smart SL ----------
        optimal_sl = self._compute_smart_sl(candles, h1, dominant_dir, symbol)

        # ---------- Smart TP ----------
        optimal_tp = self._compute_smart_tp(
            candles, h1, dominant_dir, optimal_sl, confluence, symbol)

        # ---------- Exit Urgency ----------
        exit_urgency = self._compute_exit_urgency(candles, h1, m15, m5, symbol)

        return {
            "confluence": confluence,
            "entry_quality": entry_quality,
            "optimal_sl": optimal_sl,
            "optimal_tp": optimal_tp,
            "exit_urgency": exit_urgency,
            "regime": regime,
            "h1_dir": h1.direction,
            "m15_dir": m15.direction,
            "m5_dir": m5.direction,
            "m1_dir": m1.direction,
            "h1_strength": h1.strength,
            "m15_strength": m15.strength,
            "m5_strength": m5.strength,
            "m1_strength": m1.strength,
            "h1_detail": h1.detail,
            "m15_detail": m15.detail,
            "m5_detail": m5.detail,
            "m1_detail": m1.detail,
        }

    # =================================================================
    #  1. H1 ANALYSIS -- Primary Trend Direction + Strength
    # =================================================================

    def _analyze_h1(self, df: Optional[pd.DataFrame]) -> _TFResult:
        """H1: ADX trend strength, EMA alignment (8/21/50), SuperTrend."""
        if df is None or len(df) < 80:
            return _TFResult()
        try:
            c = df["close"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2  # completed bar

            # EMAs
            ema8 = _ema(c, 8)
            ema21 = _ema(c, 21)
            ema50 = _ema(c, 50)

            # ADX
            adx_arr, di_p, di_m = _adx(h, l, c, 14)
            adx_val = _safe(adx_arr, bi, 25.0)
            dip = _safe(di_p, bi, 0.0)
            dim = _safe(di_m, bi, 0.0)

            # SuperTrend
            st_line, st_dir = _supertrend(h.copy(), l.copy(), c, 2.5, 10)
            st_d = int(st_dir[bi]) if bi < len(st_dir) else 0

            # ATR for normalization
            atr_arr = _atr(h, l, c, 14)
            atr_val = _safe(atr_arr, bi, 0.0001)

            # EMA alignment score
            e8 = _safe(ema8, bi)
            e21 = _safe(ema21, bi)
            e50 = _safe(ema50, bi)
            price = _safe(c, bi)

            # --- Direction ---
            bull_signals = 0
            bear_signals = 0

            # EMA stack
            if e8 > e21 > e50:
                bull_signals += 2
            elif e8 < e21 < e50:
                bear_signals += 2
            elif e8 > e21:
                bull_signals += 1
            elif e8 < e21:
                bear_signals += 1

            # Price vs EMA50
            if price > e50:
                bull_signals += 1
            elif price < e50:
                bear_signals += 1

            # SuperTrend
            if st_d == 1:
                bull_signals += 1
            elif st_d == -1:
                bear_signals += 1

            # ADX directional
            if adx_val > 20:
                if dip > dim:
                    bull_signals += 1
                elif dim > dip:
                    bear_signals += 1

            # Direction determination
            if bull_signals >= 3 and bull_signals > bear_signals:
                direction = "LONG"
            elif bear_signals >= 3 and bear_signals > bull_signals:
                direction = "SHORT"
            else:
                direction = "FLAT"

            # --- Strength (0-1) ---
            # ADX component (0-0.4): >40 = very strong, >25 = strong
            adx_strength = min(adx_val / 50.0, 1.0) * 0.4

            # EMA separation (0-0.3): distance between fast/slow EMAs relative to ATR
            ema_sep = abs(e8 - e50) / max(atr_val, 0.0001)
            ema_strength = min(ema_sep / 3.0, 1.0) * 0.3

            # Agreement (0-0.3): how many sub-signals agree
            total_signals = bull_signals + bear_signals
            agree_ratio = max(bull_signals, bear_signals) / max(total_signals, 1)
            agree_strength = agree_ratio * 0.3

            strength = min(adx_strength + ema_strength + agree_strength, 1.0)

            detail = {
                "adx": round(adx_val, 1),
                "di_plus": round(dip, 1),
                "di_minus": round(dim, 1),
                "ema_stack": "BULL" if e8 > e21 > e50 else "BEAR" if e8 < e21 < e50 else "MIXED",
                "supertrend": "BULL" if st_d == 1 else "BEAR",
                "atr": round(atr_val, 5),
            }
            return _TFResult(direction, strength, detail)

        except Exception as e:
            log.warning("H1 analysis error: %s", e)
            return _TFResult()

    # =================================================================
    #  2. M15 ANALYSIS -- Intermediate Momentum
    # =================================================================

    def _analyze_m15(self, df: Optional[pd.DataFrame]) -> _TFResult:
        """M15: EMA cross (8/21), MACD histogram direction, RSI zone."""
        if df is None or len(df) < 50:
            return _TFResult()
        try:
            c = df["close"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2

            ema8 = _ema(c, 8)
            ema21 = _ema(c, 21)
            ml, sl, hist = _macd(c, 12, 26, 9)
            rsi_arr = _rsi(c, 14)
            atr_arr = _atr(h, l, c, 14)

            e8 = _safe(ema8, bi)
            e21 = _safe(ema21, bi)
            macd_h = _safe(hist, bi)
            macd_h_prev = _safe(hist, bi - 1)
            rsi_val = _safe(rsi_arr, bi, 50.0)
            atr_val = _safe(atr_arr, bi, 0.0001)

            bull = 0
            bear = 0

            # EMA cross
            if e8 > e21:
                bull += 1
            elif e8 < e21:
                bear += 1

            # MACD histogram direction
            if macd_h > 0:
                bull += 1
                if macd_h > macd_h_prev:
                    bull += 1  # accelerating
            elif macd_h < 0:
                bear += 1
                if macd_h < macd_h_prev:
                    bear += 1  # accelerating

            # RSI zone
            if rsi_val > 55:
                bull += 1
            elif rsi_val < 45:
                bear += 1

            # Price vs EMA21
            price = _safe(c, bi)
            if price > e21:
                bull += 1
            elif price < e21:
                bear += 1

            if bull >= 3 and bull > bear:
                direction = "LONG"
            elif bear >= 3 and bear > bull:
                direction = "SHORT"
            else:
                direction = "FLAT"

            # Strength
            total = bull + bear
            agree = max(bull, bear) / max(total, 1)
            # MACD histogram magnitude as % of ATR
            macd_mag = min(abs(macd_h) / max(atr_val * 0.1, 0.0001), 1.0)
            strength = min(agree * 0.5 + macd_mag * 0.3 + (0.2 if rsi_val > 60 or rsi_val < 40 else 0.0), 1.0)

            detail = {
                "ema_cross": "BULL" if e8 > e21 else "BEAR",
                "macd_hist": round(macd_h, 6),
                "macd_accel": "UP" if macd_h > macd_h_prev else "DOWN",
                "rsi": round(rsi_val, 1),
            }
            return _TFResult(direction, strength, detail)

        except Exception as e:
            log.warning("M15 analysis error: %s", e)
            return _TFResult()

    # =================================================================
    #  3. M5 ANALYSIS -- Micro Momentum
    # =================================================================

    def _analyze_m5(self, df: Optional[pd.DataFrame]) -> _TFResult:
        """M5: 3-bar ROC, tick velocity (close-to-close rate), EMA(5) slope."""
        if df is None or len(df) < 30:
            return _TFResult()
        try:
            c = df["close"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2

            atr_arr = _atr(h, l, c, 14)
            atr_val = _safe(atr_arr, bi, 0.0001)

            # 3-bar rate of change
            roc3 = 0.0
            if bi >= 3 and c[bi - 3] > 0:
                roc3 = (c[bi] - c[bi - 3]) / c[bi - 3] * 10000  # in basis points

            # Tick velocity: average close-to-close change over last 5 bars
            velocity = 0.0
            if bi >= 5:
                changes = np.diff(c[bi - 5:bi + 1])
                velocity = float(np.mean(changes))

            # EMA(5) slope (normalized by ATR)
            ema5 = _ema(c, 5)
            slope = 0.0
            if bi >= 2:
                slope = (_safe(ema5, bi) - _safe(ema5, bi - 2)) / max(atr_val, 0.0001)

            # RSI(7) for micro overbought/oversold
            rsi7 = _rsi(c, 7)
            rsi_val = _safe(rsi7, bi, 50.0)

            # Aggregate direction
            bull = 0
            bear = 0

            if roc3 > 5:
                bull += 1
            elif roc3 < -5:
                bear += 1

            if velocity > 0:
                bull += 1
            elif velocity < 0:
                bear += 1

            if slope > 0.1:
                bull += 1
            elif slope < -0.1:
                bear += 1

            if bull >= 2 and bull > bear:
                direction = "LONG"
            elif bear >= 2 and bear > bull:
                direction = "SHORT"
            else:
                direction = "FLAT"

            # Strength: magnitude of momentum
            roc_strength = min(abs(roc3) / 30.0, 1.0)  # 30 bps = full strength
            slope_strength = min(abs(slope) / 0.5, 1.0)
            strength = min((roc_strength * 0.5 + slope_strength * 0.5), 1.0)

            # Exhaustion detection: RSI(7) extreme + ROC slowing
            is_exhausted = False
            if direction == "LONG" and rsi_val > 80:
                is_exhausted = True
            elif direction == "SHORT" and rsi_val < 20:
                is_exhausted = True
            # Check if ROC is decelerating
            roc_prev = 0.0
            if bi >= 4 and c[bi - 4] > 0:
                roc_prev = (c[bi - 1] - c[bi - 4]) / c[bi - 4] * 10000
            is_decelerating = (direction == "LONG" and roc3 < roc_prev) or \
                              (direction == "SHORT" and roc3 > roc_prev)

            detail = {
                "roc3_bps": round(roc3, 1),
                "velocity": round(velocity, 6),
                "ema5_slope": round(slope, 3),
                "rsi7": round(rsi_val, 1),
                "exhausted": is_exhausted,
                "decelerating": is_decelerating,
            }
            return _TFResult(direction, strength, detail)

        except Exception as e:
            log.warning("M5 analysis error: %s", e)
            return _TFResult()

    # =================================================================
    #  4. M1 ANALYSIS -- Entry Timing
    # =================================================================

    def _analyze_m1(self, df: Optional[pd.DataFrame]) -> _TFResult:
        """M1: last 5 bars momentum, buying/selling pressure, spread pattern."""
        if df is None or len(df) < 20:
            return _TFResult()
        try:
            c = df["close"].values.astype(np.float64)
            o = df["open"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2

            # Last 5 completed bars momentum
            lookback = min(5, bi)
            if lookback < 3:
                return _TFResult()

            start = bi - lookback + 1
            segment_c = c[start:bi + 1]
            segment_o = o[start:bi + 1]
            segment_h = h[start:bi + 1]
            segment_l = l[start:bi + 1]

            # Net momentum: sum of (close - open) for last 5 bars
            bar_moves = segment_c - segment_o
            net_momentum = float(np.sum(bar_moves))

            # Buying vs selling pressure (wick analysis)
            # Buying pressure = close - low (bulls pushed up from low)
            # Selling pressure = high - close (bears pushed down from high)
            buy_pressure = float(np.sum(segment_c - segment_l))
            sell_pressure = float(np.sum(segment_h - segment_c))
            total_pressure = buy_pressure + sell_pressure

            if total_pressure > 0:
                pressure_ratio = buy_pressure / total_pressure  # 0.5 = neutral, >0.5 = bullish
            else:
                pressure_ratio = 0.5

            # Count bullish vs bearish bars
            bull_bars = int(np.sum(segment_c > segment_o))
            bear_bars = int(np.sum(segment_c < segment_o))

            # Average bar size (for normalization)
            avg_bar = float(np.mean(np.abs(bar_moves)))
            if avg_bar < 1e-10:
                avg_bar = 1e-10

            # Direction
            bull = 0
            bear = 0

            if net_momentum > avg_bar * 0.5:
                bull += 1
            elif net_momentum < -avg_bar * 0.5:
                bear += 1

            if pressure_ratio > 0.55:
                bull += 1
            elif pressure_ratio < 0.45:
                bear += 1

            if bull_bars >= 3:
                bull += 1
            elif bear_bars >= 3:
                bear += 1

            if bull >= 2 and bull > bear:
                direction = "LONG"
            elif bear >= 2 and bear > bull:
                direction = "SHORT"
            else:
                direction = "FLAT"

            # Strength
            momentum_strength = min(abs(net_momentum) / (avg_bar * 3.0), 1.0)
            pressure_strength = abs(pressure_ratio - 0.5) * 2.0  # 0-1
            strength = min(momentum_strength * 0.6 + pressure_strength * 0.4, 1.0)

            detail = {
                "net_momentum": round(net_momentum, 6),
                "pressure_ratio": round(pressure_ratio, 3),
                "bull_bars": bull_bars,
                "bear_bars": bear_bars,
            }
            return _TFResult(direction, strength, detail)

        except Exception as e:
            log.warning("M1 analysis error: %s", e)
            return _TFResult()

    # =================================================================
    #  CONFLUENCE
    # =================================================================

    def _compute_confluence(self, dirs: list) -> tuple:
        """Count how many TFs agree. Returns (count, dominant_direction)."""
        long_count = sum(1 for d in dirs if d == "LONG")
        short_count = sum(1 for d in dirs if d == "SHORT")

        if long_count > short_count:
            return long_count, "LONG"
        elif short_count > long_count:
            return short_count, "SHORT"
        else:
            # Tie or all FLAT -- use H1 as tiebreaker
            if dirs[0] != "FLAT":
                return max(long_count, short_count), dirs[0]
            return 0, "FLAT"

    # =================================================================
    #  REGIME DETECTION
    # =================================================================

    def _detect_regime(self, h1_df: pd.DataFrame, h1_result: _TFResult) -> str:
        """Classify market regime from H1 data.

        Returns: "trending", "ranging", "volatile", "breakout"
        """
        try:
            c = h1_df["close"].values.astype(np.float64)
            h = h1_df["high"].values.astype(np.float64)
            l = h1_df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2

            adx_val = h1_result.detail.get("adx", 25.0)
            atr_arr = _atr(h, l, c, 14)
            atr_val = _safe(atr_arr, bi, 0.0001)

            # Bollinger Band Width for volatility regime
            bb_width = 0.0
            if bi >= 20:
                window = c[bi - 19:bi + 1]
                mean_p = float(np.mean(window))
                std_p = float(np.std(window))
                if mean_p > 0:
                    bb_width = (2.0 * 2.0 * std_p) / mean_p * 100  # percentage

            # Donchian channel breakout check
            is_breakout = False
            if bi >= 20:
                dc_high = float(np.max(h[bi - 20:bi]))
                dc_low = float(np.min(l[bi - 20:bi]))
                if c[bi] > dc_high or c[bi] < dc_low:
                    is_breakout = True

            # ATR ratio: current ATR vs 50-bar average ATR
            atr_ratio = 1.0
            if bi >= 50:
                avg_atr = float(np.mean(atr_arr[max(0, bi - 50):bi]))
                if avg_atr > 0:
                    atr_ratio = atr_val / avg_atr

            # Classification
            if is_breakout and atr_ratio > 1.3:
                return "breakout"
            elif adx_val > 25 and h1_result.strength > 0.5:
                return "trending"
            elif bb_width > 4.0 or atr_ratio > 1.5:
                return "volatile"
            elif adx_val < 20 and bb_width < 2.0:
                return "ranging"
            elif adx_val >= 20:
                return "trending"
            else:
                return "ranging"

        except Exception as e:
            log.warning("Regime detection error: %s", e)
            return "ranging"

    # =================================================================
    #  ENTRY QUALITY SCORE (0-100)
    # =================================================================

    def _compute_entry_quality(self, h1, m15, m5, m1, dominant_dir) -> float:
        """0-100 composite entry quality score.

        H1 trend alignment:          0-25
        M15 momentum confirmation:   0-25
        M5 micro-timing:             0-25
        M1 tick flow:                0-25
        """
        if dominant_dir == "FLAT":
            return 0.0

        score = 0.0

        # --- H1: Trend alignment (0-25) ---
        if h1.direction == dominant_dir:
            # Base: 10 for agreement
            h1_score = 10.0
            # Strength bonus: up to 10
            h1_score += h1.strength * 10.0
            # ADX bonus: >30 = extra 5
            adx = h1.detail.get("adx", 0)
            if adx > 30:
                h1_score += 5.0
            elif adx > 40:
                h1_score += 5.0  # total 10 extra for very strong
            score += min(h1_score, 25.0)
        elif h1.direction == "FLAT":
            # FLAT H1 = some credit for not opposing
            score += 5.0

        # --- M15: Momentum confirmation (0-25) ---
        if m15.direction == dominant_dir:
            m15_score = 10.0
            m15_score += m15.strength * 10.0
            # MACD acceleration bonus
            if m15.detail.get("macd_accel") == ("UP" if dominant_dir == "LONG" else "DOWN"):
                m15_score += 5.0
            score += min(m15_score, 25.0)
        elif m15.direction == "FLAT":
            score += 3.0

        # --- M5: Micro-timing (0-25) ---
        # Best entry = M5 shows pullback recovery, not exhaustion
        if m5.direction == dominant_dir:
            m5_score = 8.0
            m5_score += m5.strength * 7.0
            # Bonus: NOT exhausted (entering fresh momentum, not the tail end)
            if not m5.detail.get("exhausted", False):
                m5_score += 5.0
            # Bonus: NOT decelerating
            if not m5.detail.get("decelerating", False):
                m5_score += 5.0
            score += min(m5_score, 25.0)
        elif m5.direction == "FLAT":
            # FLAT M5 can be a consolidation before continuation -- small credit
            score += 5.0
        else:
            # M5 opposing = possible pullback entry (can be good if H1+M15 agree)
            # Give partial credit for pullback setup
            if h1.direction == dominant_dir and m15.direction == dominant_dir:
                score += 8.0  # pullback into trend -- decent entry

        # --- M1: Tick flow (0-25) ---
        if m1.direction == dominant_dir:
            m1_score = 8.0
            m1_score += m1.strength * 7.0
            # Buying/selling pressure confirmation
            pr = m1.detail.get("pressure_ratio", 0.5)
            if dominant_dir == "LONG" and pr > 0.6:
                m1_score += 5.0
            elif dominant_dir == "SHORT" and pr < 0.4:
                m1_score += 5.0
            # Bar count confirmation
            if dominant_dir == "LONG" and m1.detail.get("bull_bars", 0) >= 4:
                m1_score += 5.0
            elif dominant_dir == "SHORT" and m1.detail.get("bear_bars", 0) >= 4:
                m1_score += 5.0
            score += min(m1_score, 25.0)
        elif m1.direction == "FLAT":
            score += 3.0

        return round(min(score, 100.0), 1)

    # =================================================================
    #  SMART SL CALCULATOR
    # =================================================================

    def _compute_smart_sl(self, candles: dict, h1: _TFResult,
                          direction: str, symbol: str) -> float:
        """Dynamic SL based on multi-TF structure.

        Uses M15 swing high/low as primary SL reference,
        checks M5 support/resistance zones, applies ATR floor.
        Returns SL distance in price points (positive number).
        """
        if direction == "FLAT":
            return 0.0

        # Get ATR from H1 as baseline
        h1_atr = h1.detail.get("atr", 0.0)
        if h1_atr <= 0:
            # Compute from candles
            df60 = candles.get(60)
            if df60 is not None and len(df60) > 15:
                arr = _atr(df60["high"].values.astype(np.float64),
                           df60["low"].values.astype(np.float64),
                           df60["close"].values.astype(np.float64), 14)
                h1_atr = _safe(arr, len(arr) - 1, 0.0001)
            else:
                h1_atr = 0.0001

        # ---- M15 swing SL ----
        m15_sl = self._find_swing_sl(candles.get(15), direction, lookback=20)

        # ---- M5 support/resistance zone ----
        m5_sl = self._find_sr_zone_sl(candles.get(5), direction, lookback=30)

        # ---- ATR-based floor ----
        # Minimum SL = 1.0x H1 ATR (don't go tighter than this)
        atr_floor = h1_atr * 1.0

        # ---- Combine ----
        # Take the LARGER of M15 swing and M5 SR, but at least ATR floor
        candidates = []
        if m15_sl > 0:
            candidates.append(m15_sl)
        if m5_sl > 0:
            candidates.append(m5_sl)

        if candidates:
            structural_sl = max(candidates)
            # Don't let structural SL exceed 3x ATR (cap for sanity)
            structural_sl = min(structural_sl, h1_atr * 3.0)
            optimal_sl = max(structural_sl, atr_floor)
        else:
            # No structural levels found -- fall back to 1.5x ATR
            optimal_sl = h1_atr * 1.5

        # Broker minimum SL distance (approximate: 10 points for gold, tiny for forex)
        # This is a safety floor; the executor will enforce actual broker minimums
        broker_min = h1_atr * 0.3
        optimal_sl = max(optimal_sl, broker_min)

        return round(float(optimal_sl), 6)

    def _find_swing_sl(self, df: Optional[pd.DataFrame], direction: str,
                       lookback: int = 20) -> float:
        """Find M15 swing high/low for SL placement.

        For LONG: SL below the most recent swing low.
        For SHORT: SL above the most recent swing high.
        Returns price DISTANCE (not absolute level).
        """
        if df is None or len(df) < lookback + 5:
            return 0.0
        try:
            c = df["close"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2
            price = c[bi]

            if direction == "LONG":
                # Find swing lows in last N bars
                # A swing low: l[i] < l[i-1] and l[i] < l[i+1] (and l[i-2], l[i+2] for stronger)
                swing_lows = []
                search_start = max(2, bi - lookback)
                for i in range(search_start, bi - 1):
                    if l[i] <= l[i - 1] and l[i] <= l[i + 1]:
                        # Confirm with wider context
                        if i >= 2 and i + 2 < n:
                            if l[i] <= l[i - 2] and l[i] <= l[i + 2]:
                                swing_lows.append(float(l[i]))
                        else:
                            swing_lows.append(float(l[i]))

                if swing_lows:
                    # Use the most recent swing low (closest to current price)
                    recent_low = swing_lows[-1]
                    distance = price - recent_low
                    if distance > 0:
                        return distance

                # Fallback: use the low of the lookback window
                window_low = float(np.min(l[max(0, bi - lookback):bi + 1]))
                distance = price - window_low
                return max(distance, 0.0)

            else:  # SHORT
                swing_highs = []
                search_start = max(2, bi - lookback)
                for i in range(search_start, bi - 1):
                    if h[i] >= h[i - 1] and h[i] >= h[i + 1]:
                        if i >= 2 and i + 2 < n:
                            if h[i] >= h[i - 2] and h[i] >= h[i + 2]:
                                swing_highs.append(float(h[i]))
                        else:
                            swing_highs.append(float(h[i]))

                if swing_highs:
                    recent_high = swing_highs[-1]
                    distance = recent_high - price
                    if distance > 0:
                        return distance

                window_high = float(np.max(h[max(0, bi - lookback):bi + 1]))
                distance = window_high - price
                return max(distance, 0.0)

        except Exception as e:
            log.warning("Swing SL error: %s", e)
            return 0.0

    def _find_sr_zone_sl(self, df: Optional[pd.DataFrame], direction: str,
                         lookback: int = 30) -> float:
        """Find M5 support/resistance zone for SL reference.

        Uses clustering of recent lows (for longs) or highs (for shorts)
        to identify zones where price has bounced.
        Returns price DISTANCE.
        """
        if df is None or len(df) < lookback + 5:
            return 0.0
        try:
            c = df["close"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2
            price = c[bi]

            atr_arr = _atr(h, l, c, 14)
            atr_val = _safe(atr_arr, bi, 0.0001)
            zone_tolerance = atr_val * 0.5  # cluster tolerance

            start = max(0, bi - lookback)

            if direction == "LONG":
                # Cluster recent lows
                lows = l[start:bi + 1].copy()
                lows.sort()
                # Find densest cluster of lows below current price
                best_level = 0.0
                best_count = 0
                for ref_low in lows:
                    if ref_low >= price:
                        continue
                    count = int(np.sum(np.abs(lows - ref_low) <= zone_tolerance))
                    if count > best_count:
                        best_count = count
                        best_level = float(ref_low)
                if best_count >= 2 and best_level > 0:
                    distance = price - best_level
                    if distance > 0:
                        return distance
            else:  # SHORT
                highs = h[start:bi + 1].copy()
                highs.sort()
                best_level = 0.0
                best_count = 0
                for ref_high in highs[::-1]:
                    if ref_high <= price:
                        continue
                    count = int(np.sum(np.abs(highs - ref_high) <= zone_tolerance))
                    if count > best_count:
                        best_count = count
                        best_level = float(ref_high)
                if best_count >= 2 and best_level > 0:
                    distance = best_level - price
                    if distance > 0:
                        return distance

            return 0.0

        except Exception as e:
            log.warning("SR zone SL error: %s", e)
            return 0.0

    # =================================================================
    #  SMART TP CALCULATOR
    # =================================================================

    def _compute_smart_tp(self, candles: dict, h1: _TFResult,
                          direction: str, sl_dist: float,
                          confluence: int, symbol: str) -> float:
        """Dynamic TP based on multi-TF structure levels and mean move.

        Uses next M15/H1 resistance/support, historical mean move,
        and scales by confluence.
        Returns TP distance in price points (positive number).
        """
        if direction == "FLAT" or sl_dist <= 0:
            return 0.0

        h1_atr = h1.detail.get("atr", 0.0)
        if h1_atr <= 0:
            h1_atr = sl_dist  # fallback

        # ---- 1. Next structural level (M15 + H1) ----
        m15_target = self._find_next_level(candles.get(15), direction, lookback=40)
        h1_target = self._find_next_level(candles.get(60), direction, lookback=30)

        structural_tp = 0.0
        if m15_target > 0 and h1_target > 0:
            # Use the closer target (more conservative)
            structural_tp = min(m15_target, h1_target)
        elif m15_target > 0:
            structural_tp = m15_target
        elif h1_target > 0:
            structural_tp = h1_target

        # ---- 2. Historical mean move ----
        mean_move = self._compute_mean_move(candles.get(60), direction, lookback=50)

        # ---- 3. ATR-based R:R target ----
        # Minimum acceptable: 1.5x SL (R:R >= 1.5)
        min_tp = sl_dist * 1.5

        # ---- 4. Confluence scaling ----
        # 4/4 TFs agree = let it run (wider TP), 2/4 = tighter
        confluence_mult = {
            0: 1.0,
            1: 1.0,
            2: 1.2,
            3: 1.5,
            4: 2.0,
        }.get(confluence, 1.0)

        # ---- Combine ----
        candidates = []
        if structural_tp > 0:
            candidates.append(structural_tp)
        if mean_move > 0:
            candidates.append(mean_move)

        if candidates:
            # Use median of structural and mean-move targets
            base_tp = float(np.median(candidates))
        else:
            # No structural data: use 2x SL as default
            base_tp = sl_dist * 2.0

        # Apply confluence scaling
        scaled_tp = base_tp * confluence_mult

        # Ensure minimum R:R
        optimal_tp = max(scaled_tp, min_tp)

        # Cap at 5x ATR (don't set unrealistic targets)
        optimal_tp = min(optimal_tp, h1_atr * 5.0)

        return round(float(optimal_tp), 6)

    def _find_next_level(self, df: Optional[pd.DataFrame], direction: str,
                         lookback: int = 40) -> float:
        """Find next resistance (for longs) or support (for shorts) level.

        Returns DISTANCE to that level, or 0 if not found.
        """
        if df is None or len(df) < lookback + 5:
            return 0.0
        try:
            c = df["close"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2
            price = c[bi]

            start = max(2, bi - lookback)

            if direction == "LONG":
                # Find swing highs above current price
                levels = []
                for i in range(start, bi - 1):
                    if h[i] >= h[i - 1] and h[i] >= h[i + 1]:
                        if h[i] > price:
                            levels.append(float(h[i]))
                if levels:
                    # Nearest resistance above
                    nearest = min(levels)
                    return nearest - price
            else:  # SHORT
                levels = []
                for i in range(start, bi - 1):
                    if l[i] <= l[i - 1] and l[i] <= l[i + 1]:
                        if l[i] < price:
                            levels.append(float(l[i]))
                if levels:
                    nearest = max(levels)
                    return price - nearest

            return 0.0

        except Exception as e:
            log.warning("Next level error: %s", e)
            return 0.0

    def _compute_mean_move(self, df: Optional[pd.DataFrame], direction: str,
                           lookback: int = 50) -> float:
        """Compute mean directional move distance over recent H1 bars.

        Measures average impulse move (close-to-close of trending sequences).
        Returns distance in price points.
        """
        if df is None or len(df) < lookback + 5:
            return 0.0
        try:
            c = df["close"].values.astype(np.float64)
            n = len(c)
            bi = n - 2
            start = max(0, bi - lookback)

            # Find trending sequences (3+ bars in same direction)
            moves = []
            i = start + 1
            while i <= bi:
                # Start of a move
                if direction == "LONG":
                    if c[i] > c[i - 1]:
                        move_start = c[i - 1]
                        j = i
                        while j <= bi and c[j] > c[j - 1]:
                            j += 1
                        move_end = c[j - 1]
                        bars = j - i
                        if bars >= 2:  # at least 2-bar trending move
                            moves.append(move_end - move_start)
                        i = j
                    else:
                        i += 1
                else:
                    if c[i] < c[i - 1]:
                        move_start = c[i - 1]
                        j = i
                        while j <= bi and c[j] < c[j - 1]:
                            j += 1
                        move_end = c[j - 1]
                        bars = j - i
                        if bars >= 2:
                            moves.append(move_start - move_end)
                        i = j
                    else:
                        i += 1

            if moves:
                return float(np.mean(moves))
            return 0.0

        except Exception as e:
            log.warning("Mean move error: %s", e)
            return 0.0

    # =================================================================
    #  EXIT URGENCY (0 - 1.0)
    # =================================================================

    def _compute_exit_urgency(self, candles: dict, h1: _TFResult,
                              m15: _TFResult, m5: _TFResult,
                              symbol: str) -> float:
        """Real-time multi-TF exit detection.

        0.0 = hold, 1.0 = exit immediately.

        Signals checked:
          - M5 momentum divergence
          - M15 structure break
          - H1 candle reversal pattern
        """
        # We need to know position direction. Since this is called generically,
        # check both directions and return the max urgency for the current regime.
        # The caller knows the position direction and can filter.
        #
        # For a generic analysis, we compute urgency for a LONG position.
        # The brain should flip interpretation for SHORT positions.

        urgency = 0.0

        # ---- 1. M5 Momentum Divergence (0-0.35) ----
        m5_div = self._check_m5_divergence(candles.get(5))
        urgency += m5_div * 0.35

        # ---- 2. M15 Structure Break (0-0.35) ----
        m15_break = self._check_m15_structure_break(candles.get(15))
        urgency += m15_break * 0.35

        # ---- 3. H1 Reversal Pattern (0-0.30) ----
        h1_reversal = self._check_h1_reversal(candles.get(60))
        urgency += h1_reversal * 0.30

        return round(min(float(urgency), 1.0), 3)

    def _check_m5_divergence(self, df: Optional[pd.DataFrame]) -> float:
        """Check M5 for momentum divergence.

        Price making higher highs but MACD making lower highs (bearish divergence)
        or price making lower lows but MACD making higher lows (bullish divergence).

        Returns 0.0 (no divergence) to 1.0 (clear divergence).
        """
        if df is None or len(df) < 30:
            return 0.0
        try:
            c = df["close"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2

            ml, sl, hist = _macd(c, 12, 26, 9)

            # Look at last 10 bars for divergence
            window = 10
            if bi < window + 2:
                return 0.0

            seg_h = h[bi - window:bi + 1]
            seg_l = l[bi - window:bi + 1]
            seg_hist = hist[bi - window:bi + 1]

            # Split into two halves
            half = window // 2
            first_half_h = seg_h[:half]
            second_half_h = seg_h[half:]
            first_half_l = seg_l[:half]
            second_half_l = seg_l[half:]
            first_half_hist = seg_hist[:half]
            second_half_hist = seg_hist[half:]

            score = 0.0

            # Bearish divergence: price higher highs, MACD lower highs
            if float(np.max(second_half_h)) > float(np.max(first_half_h)):
                if float(np.max(second_half_hist)) < float(np.max(first_half_hist)):
                    score = max(score, 0.7)

            # Bullish divergence: price lower lows, MACD higher lows
            if float(np.min(second_half_l)) < float(np.min(first_half_l)):
                if float(np.min(second_half_hist)) > float(np.min(first_half_hist)):
                    score = max(score, 0.7)

            # Weaker signal: MACD histogram shrinking while price extends
            hist_now = _safe(hist, bi)
            hist_3ago = _safe(hist, bi - 3)
            price_now = c[bi]
            price_3ago = c[bi - 3]

            if hist_now > 0 and hist_3ago > 0:
                # Bullish histogram shrinking while price still rising
                if hist_now < hist_3ago * 0.6 and price_now > price_3ago:
                    score = max(score, 0.4)
            elif hist_now < 0 and hist_3ago < 0:
                # Bearish histogram shrinking while price still falling
                if hist_now > hist_3ago * 0.6 and price_now < price_3ago:
                    score = max(score, 0.4)

            return float(score)

        except Exception as e:
            log.warning("M5 divergence error: %s", e)
            return 0.0

    def _check_m15_structure_break(self, df: Optional[pd.DataFrame]) -> float:
        """Check M15 for structure break (swing low broken for longs, swing high broken for shorts).

        Returns 0.0 (no break) to 1.0 (clear break).
        """
        if df is None or len(df) < 30:
            return 0.0
        try:
            c = df["close"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2

            # Find recent swing lows and highs (last 15 bars)
            lookback = min(15, bi - 2)
            start = bi - lookback

            swing_lows = []
            swing_highs = []
            for i in range(start + 1, bi):
                if i < 1 or i + 1 >= n:
                    continue
                if l[i] <= l[i - 1] and l[i] <= l[i + 1]:
                    swing_lows.append(float(l[i]))
                if h[i] >= h[i - 1] and h[i] >= h[i + 1]:
                    swing_highs.append(float(h[i]))

            score = 0.0

            # Bearish structure break: current close below most recent swing low
            if swing_lows:
                recent_low = swing_lows[-1]
                if c[bi] < recent_low:
                    score = max(score, 0.8)
                elif c[bi] < recent_low * 1.001:  # within 0.1% of breaking
                    score = max(score, 0.3)

            # Bullish structure break: current close above most recent swing high
            if swing_highs:
                recent_high = swing_highs[-1]
                if c[bi] > recent_high:
                    score = max(score, 0.8)
                elif c[bi] > recent_high * 0.999:
                    score = max(score, 0.3)

            return float(score)

        except Exception as e:
            log.warning("M15 structure break error: %s", e)
            return 0.0

    def _check_h1_reversal(self, df: Optional[pd.DataFrame]) -> float:
        """Check H1 for reversal candlestick patterns.

        Looks for:
          - Engulfing patterns
          - Pin bars (hammer / shooting star)
          - Large-body reversals after trend

        Returns 0.0 (no reversal) to 1.0 (strong reversal signal).
        """
        if df is None or len(df) < 10:
            return 0.0
        try:
            c = df["close"].values.astype(np.float64)
            o = df["open"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2

            if bi < 3:
                return 0.0

            score = 0.0

            body = abs(c[bi] - o[bi])
            full_range = h[bi] - l[bi]
            if full_range < 1e-10:
                return 0.0
            body_ratio = body / full_range

            prev_body = abs(c[bi - 1] - o[bi - 1])
            prev_range = h[bi - 1] - l[bi - 1]

            # Bearish engulfing
            if (c[bi - 1] > o[bi - 1] and  # prev was bullish
                c[bi] < o[bi] and           # current is bearish
                o[bi] >= c[bi - 1] and      # opens above prev close
                c[bi] <= o[bi - 1]):        # closes below prev open
                score = max(score, 0.7)
                if body_ratio > 0.6:
                    score = max(score, 0.9)

            # Bullish engulfing
            if (c[bi - 1] < o[bi - 1] and
                c[bi] > o[bi] and
                o[bi] <= c[bi - 1] and
                c[bi] >= o[bi - 1]):
                score = max(score, 0.7)
                if body_ratio > 0.6:
                    score = max(score, 0.9)

            # Shooting star (bearish pin bar)
            upper_wick = h[bi] - max(c[bi], o[bi])
            lower_wick = min(c[bi], o[bi]) - l[bi]
            if upper_wick > body * 2.0 and upper_wick > full_range * 0.6:
                score = max(score, 0.6)

            # Hammer (bullish pin bar)
            if lower_wick > body * 2.0 and lower_wick > full_range * 0.6:
                score = max(score, 0.6)

            # Large body reversal: current bar body > 2x previous bar body,
            # and direction changed
            if prev_body > 0:
                if body > prev_body * 2.0:
                    prev_bull = c[bi - 1] > o[bi - 1]
                    curr_bull = c[bi] > o[bi]
                    if prev_bull != curr_bull:
                        score = max(score, 0.5)

            return float(score)

        except Exception as e:
            log.warning("H1 reversal error: %s", e)
            return 0.0

    # =================================================================
    #  DEFAULT RESULT
    # =================================================================

    @staticmethod
    def _default_result() -> dict:
        """Safe default when analysis fails."""
        return {
            "confluence": 0,
            "entry_quality": 0.0,
            "optimal_sl": 0.0,
            "optimal_tp": 0.0,
            "exit_urgency": 0.0,
            "regime": "ranging",
            "h1_dir": "FLAT",
            "m15_dir": "FLAT",
            "m5_dir": "FLAT",
            "m1_dir": "FLAT",
            "h1_strength": 0.0,
            "m15_strength": 0.0,
            "m5_strength": 0.0,
            "m1_strength": 0.0,
            "h1_detail": {},
            "m15_detail": {},
            "m5_detail": {},
            "m1_detail": {},
        }
