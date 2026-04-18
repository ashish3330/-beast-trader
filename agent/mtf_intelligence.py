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
from datetime import datetime, timezone
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


def _bollinger_bands(c, period=20, num_std=2.0):
    """Bollinger Bands: returns (upper, middle, lower, bandwidth, %b)."""
    n = len(c)
    if n < period:
        mid = np.full(n, c[-1] if n > 0 else 0.0)
        return mid, mid, mid, np.zeros(n), np.full(n, 0.5)
    mid = np.empty(n, dtype=np.float64)
    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)
    bw = np.zeros(n, dtype=np.float64)
    pctb = np.full(n, 0.5, dtype=np.float64)
    for i in range(n):
        if i < period - 1:
            mid[i] = upper[i] = lower[i] = c[i]
            continue
        window = c[i - period + 1:i + 1]
        m = float(np.mean(window))
        s = float(np.std(window))
        mid[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s
        band_range = upper[i] - lower[i]
        bw[i] = band_range / m if m > 0 else 0.0
        pctb[i] = (c[i] - lower[i]) / band_range if band_range > 0 else 0.5
    return upper, mid, lower, bw, pctb


# =====================================================================
#  SESSION / TIME CONSTANTS
# =====================================================================

# Market sessions in UTC hours (start, end)
_SESSION_ASIA_START = 0    # Tokyo 00:00 UTC (09:00 JST)
_SESSION_ASIA_END = 8      # 08:00 UTC
_SESSION_LONDON_START = 7  # 07:00 UTC
_SESSION_LONDON_END = 16   # 16:00 UTC
_SESSION_NY_START = 12     # 12:00 UTC (08:00 EST)
_SESSION_NY_END = 21       # 21:00 UTC

# Overlap windows
_LONDON_NY_OVERLAP_START = 12  # 12:00 UTC
_LONDON_NY_OVERLAP_END = 16    # 16:00 UTC
_ASIA_LONDON_OVERLAP_START = 7
_ASIA_LONDON_OVERLAP_END = 8

# Round-number levels for common instruments
_ROUND_LEVELS = {
    "XAUUSD": [50, 25],     # e.g. 2300, 2325, 2350 ...
    "XAGUSD": [1.0, 0.5],
    "US30": [500, 250, 100],
    "US100": [500, 250, 100],
    "US500": [100, 50, 25],
    "GER40": [500, 250, 100],
    "EURUSD": [0.01, 0.005],
    "GBPUSD": [0.01, 0.005],
    "USDJPY": [1.0, 0.5],
    "AUDUSD": [0.01, 0.005],
    "USDCAD": [0.01, 0.005],
    "NZDUSD": [0.01, 0.005],
}

# Fibonacci ratios
_FIB_LEVELS = (0.236, 0.382, 0.500, 0.618, 0.786)


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

        # ---------- Deep Market Monitoring ----------
        vol_h1 = self._analyze_volume_profile(candles.get(60), dominant_dir)
        vol_m15 = self._analyze_volume_profile(candles.get(15), dominant_dir)
        swing_h1 = self._detect_swing_structure(candles.get(60))
        swing_m15 = self._detect_swing_structure(candles.get(15), lookback=30)
        momentum = self._analyze_momentum_quality(candles.get(60))
        order_flow = self._analyze_order_flow(candles.get(5))

        # ---------- Institutional Intelligence ----------
        liquidity = self._detect_liquidity_zones(candles, symbol, dominant_dir)
        fibonacci = self._compute_fibonacci_levels(candles, dominant_dir)
        session_ctx = self._detect_session_context()
        candle_patterns_h1 = self._detect_candle_patterns(candles.get(60), "H1")
        candle_patterns_m15 = self._detect_candle_patterns(candles.get(15), "M15")
        mtf_divergence = self._detect_mtf_divergence(candles)
        vol_cycle = self._detect_volatility_cycle(candles)
        time_weight = self._compute_time_weight(session_ctx)

        # ---------- Additional Intelligence (4 new) ----------
        corr_regime = self._detect_correlation_regime(symbol, dominant_dir)
        best_tf = self._detect_best_timeframe(symbol)
        m1_noise = self._filter_m1_noise(symbol)
        mean_rev = self._detect_mean_reversion(symbol, dominant_dir)

        # Boost/penalize entry quality based on deep analysis
        deep_bonus = 0
        if vol_h1["vol_trend"] == ("bullish" if dominant_dir == "LONG" else "bearish"):
            deep_bonus += 5  # volume confirms direction
        if vol_h1["climax"]:
            deep_bonus -= 10  # climax = potential reversal
        if momentum["exhaustion"]:
            deep_bonus -= 10  # exhaustion = don't enter
        if momentum["decel"]:
            deep_bonus -= 5  # slowing momentum
        if order_flow["absorption"]:
            deep_bonus -= 8  # big players absorbing = reversal setup
        if swing_h1["structure"] == ("uptrend" if dominant_dir == "LONG" else "downtrend"):
            deep_bonus += 5  # structure confirms

        # --- NEW: Institutional bonuses/penalties ---

        # Liquidity zone proximity: penalize entries AT liquidity (reversal risk)
        # but boost entries NEAR liquidity in direction of magnet
        if liquidity["at_liquidity"]:
            deep_bonus -= 8  # right at a major level = reversal risk
        elif liquidity["proximity"] > 0.5:
            # Near a zone but not at it -- check if magnet pulls in our direction
            if dominant_dir == "LONG" and liquidity["magnet_above"] > liquidity["magnet_below"]:
                deep_bonus += 4  # magnet pulling in our direction
            elif dominant_dir == "SHORT" and liquidity["magnet_below"] > liquidity["magnet_above"]:
                deep_bonus += 4

        # Fibonacci alignment: price near key fib level in our direction
        fib_atr = h1.detail.get("atr", 0.0001)
        if fibonacci["nearest_fib_dist"] < fib_atr * 0.5:
            deep_bonus += 3  # price at fib level = institutional interest

        # Candle pattern confirmation
        patterns_dir_match = False
        for cp in [candle_patterns_h1, candle_patterns_m15]:
            if dominant_dir == "LONG" and cp["bullish_count"] > 0:
                best_str = max((p["strength"] for p in cp["patterns"]
                               if p["direction"] == "LONG"), default=0)
                deep_bonus += int(best_str * 8)  # up to +8
                patterns_dir_match = True
            elif dominant_dir == "SHORT" and cp["bearish_count"] > 0:
                best_str = max((p["strength"] for p in cp["patterns"]
                               if p["direction"] == "SHORT"), default=0)
                deep_bonus += int(best_str * 8)
                patterns_dir_match = True
            # Opposing patterns = penalty
            if dominant_dir == "LONG" and cp["bearish_count"] > 0:
                deep_bonus -= 5
            elif dominant_dir == "SHORT" and cp["bullish_count"] > 0:
                deep_bonus -= 5

        # MTF divergence: regular divergence against us = big penalty
        div_combined = mtf_divergence["combined"]
        if dominant_dir == "LONG" and "bearish" in div_combined and "hidden" not in div_combined:
            deep_bonus -= 12  # regular bearish divergence = reversal coming
        elif dominant_dir == "SHORT" and "bullish" in div_combined and "hidden" not in div_combined:
            deep_bonus -= 12
        # Hidden divergence IN our direction = continuation signal = bonus
        if dominant_dir == "LONG" and div_combined == "hidden_bullish":
            deep_bonus += 7
        elif dominant_dir == "SHORT" and div_combined == "hidden_bearish":
            deep_bonus += 7

        # Volatility cycle: squeeze about to break in our direction = excellent entry
        if vol_cycle["squeeze"] and vol_cycle["breakout_dir"] == dominant_dir:
            deep_bonus += 10  # squeeze breakout in our direction
        elif vol_cycle["squeeze"] and vol_cycle["breakout_dir"] != "FLAT" \
                and vol_cycle["breakout_dir"] != dominant_dir:
            deep_bonus -= 8  # squeeze breaking against us
        if vol_cycle["expansion"] and vol_cycle["breakout_dir"] == dominant_dir:
            deep_bonus += 5  # expansion confirming our direction

        # Session quality: scale bonus by time weight
        if session_ctx["overlap"] == "london_ny":
            deep_bonus += 5  # best trading window
        elif session_ctx["session"] == "asian":
            deep_bonus -= 3  # ranging session for most pairs

        # --- 4 new features bonuses ---
        # Correlation regime: all assets same direction = systemic, less reliable for individuals
        if corr_regime["all_same_dir"]:
            deep_bonus -= 5  # herd move, signal less individual

        # Adaptive TF: boost if best TF agrees with signal TF (H1)
        if best_tf["best_tf"] == 60:
            deep_bonus += 3  # H1 is cleanest TF right now
        elif best_tf["h1_score"] < 0.3:
            deep_bonus -= 5  # H1 is noisy, signal unreliable

        # M1 noise: penalize if M1 is noisy
        if m1_noise["noise_level"] > 0.5:
            deep_bonus -= 5  # M1 is noise, micro-timing unreliable

        # Mean reversion: if MR setup opposes our trend entry, warning
        if mean_rev["setup"] and mean_rev["opposes_trend"]:
            deep_bonus -= 10  # entering trend but MR says reversal imminent
        elif mean_rev["setup"] and not mean_rev["opposes_trend"] and mean_rev["strength"] > 0.5:
            deep_bonus += 5  # MR pullback in our trend direction

        entry_quality = max(0, min(100, entry_quality + deep_bonus))

        # Apply time weight as a multiplier
        entry_quality = round(entry_quality * time_weight, 1)

        # Boost exit urgency on deep signals
        if momentum["exhaustion"] and exit_urgency < 0.5:
            exit_urgency = max(exit_urgency, 0.5)
        if vol_h1["climax"] and exit_urgency < 0.4:
            exit_urgency = max(exit_urgency, 0.4)

        # --- NEW: Exit urgency from institutional signals ---
        # Regular divergence against position = exit signal
        if div_combined in ("regular_bearish", "regular_bullish"):
            div_urgency = 0.3 + mtf_divergence["strength"] * 0.4
            exit_urgency = max(exit_urgency, div_urgency)

        # Candle reversal patterns on H1 against position
        for cp in [candle_patterns_h1]:
            opposing = [p for p in cp["patterns"]
                       if (p["direction"] == "SHORT" and dominant_dir == "LONG") or
                          (p["direction"] == "LONG" and dominant_dir == "SHORT")]
            if opposing:
                best_rev = max(p["strength"] for p in opposing)
                exit_urgency = max(exit_urgency, best_rev * 0.6)

        # At liquidity zone against direction = potential reversal
        if liquidity["at_liquidity"]:
            exit_urgency = max(exit_urgency, 0.35)

        # Volatility expansion breaking against us
        if vol_cycle["expansion"] and vol_cycle["breakout_dir"] != "FLAT" \
                and vol_cycle["breakout_dir"] != dominant_dir:
            exit_urgency = max(exit_urgency, 0.5)

        exit_urgency = round(min(float(exit_urgency), 1.0), 3)

        # --- Enhance SL/TP with fib levels ---
        if fibonacci["fib_cluster_sl"] > 0 and optimal_sl > 0:
            # Blend fib SL: if fib level is close to structural SL, use fib (more precise)
            fib_sl = fibonacci["fib_cluster_sl"]
            if 0.5 * optimal_sl < fib_sl < 2.0 * optimal_sl:
                optimal_sl = round((optimal_sl * 0.6 + fib_sl * 0.4), 6)

        if fibonacci["fib_cluster_tp"] > 0 and optimal_tp > 0:
            fib_tp = fibonacci["fib_cluster_tp"]
            if 0.5 * optimal_tp < fib_tp < 2.0 * optimal_tp:
                optimal_tp = round((optimal_tp * 0.5 + fib_tp * 0.5), 6)

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
            # Deep market monitoring
            "volume_h1": vol_h1,
            "volume_m15": vol_m15,
            "swing_h1": swing_h1,
            "swing_m15": swing_m15,
            "momentum": momentum,
            "order_flow": order_flow,
            # Institutional intelligence
            "liquidity": liquidity,
            "fibonacci": fibonacci,
            "session": session_ctx,
            "candle_patterns_h1": candle_patterns_h1,
            "candle_patterns_m15": candle_patterns_m15,
            "mtf_divergence": mtf_divergence,
            "volatility_cycle": vol_cycle,
            "time_weight": time_weight,
            "correlation_regime": corr_regime,
            "best_timeframe": best_tf,
            "m1_noise": m1_noise,
            "mean_reversion": mean_rev,
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
    #  DEEP MARKET MONITORING
    # =================================================================

    def _analyze_volume_profile(self, df, direction):
        """Analyze volume across price levels — institutional footprint."""
        if df is None or len(df) < 30:
            return {"vol_trend": "flat", "vol_ratio": 1.0, "climax": False, "dry_up": False}

        try:
            vol = df["tick_volume"].values.astype(float)
            close = df["close"].values.astype(float)
            n = len(vol)
            bi = n - 2

            # Volume trend: rising or falling
            vol_sma5 = np.mean(vol[max(0, bi-4):bi+1])
            vol_sma20 = np.mean(vol[max(0, bi-19):bi+1])
            vol_ratio = vol_sma5 / vol_sma20 if vol_sma20 > 0 else 1.0

            # Volume climax: current bar > 2x average (institutional activity)
            climax = float(vol[bi]) > vol_sma20 * 2.0

            # Volume dry-up: current < 0.5x average (breakout setup)
            dry_up = float(vol[bi]) < vol_sma20 * 0.5

            # Directional volume: are up-bars on higher volume than down-bars?
            up_vol = 0; dn_vol = 0; up_cnt = 0; dn_cnt = 0
            for i in range(max(0, bi-9), bi+1):
                if close[i] > close[i-1] if i > 0 else True:
                    up_vol += vol[i]; up_cnt += 1
                else:
                    dn_vol += vol[i]; dn_cnt += 1

            avg_up = up_vol / up_cnt if up_cnt > 0 else 0
            avg_dn = dn_vol / dn_cnt if dn_cnt > 0 else 0

            if direction == "LONG":
                vol_trend = "bullish" if avg_up > avg_dn * 1.3 else "bearish" if avg_dn > avg_up * 1.3 else "flat"
            else:
                vol_trend = "bearish" if avg_dn > avg_up * 1.3 else "bullish" if avg_up > avg_dn * 1.3 else "flat"

            return {"vol_trend": vol_trend, "vol_ratio": round(vol_ratio, 2),
                    "climax": climax, "dry_up": dry_up}
        except Exception:
            return {"vol_trend": "flat", "vol_ratio": 1.0, "climax": False, "dry_up": False}

    def _detect_swing_structure(self, df, lookback=50):
        """Detect swing highs and swing lows — market structure."""
        if df is None or len(df) < lookback:
            return {"swing_highs": [], "swing_lows": [], "structure": "unknown",
                    "last_hh": False, "last_hl": False, "last_lh": False, "last_ll": False}

        try:
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            n = len(high)
            start = max(0, n - lookback)

            # Find swing points (3-bar confirmed)
            swing_highs = []
            swing_lows = []
            for i in range(start + 2, n - 1):
                if high[i-1] > high[i-2] and high[i-1] > high[i]:
                    swing_highs.append((i-1, float(high[i-1])))
                if low[i-1] < low[i-2] and low[i-1] < low[i]:
                    swing_lows.append((i-1, float(low[i-1])))

            # Market structure: HH+HL = uptrend, LH+LL = downtrend
            last_hh = False; last_hl = False; last_lh = False; last_ll = False
            structure = "unknown"

            if len(swing_highs) >= 2:
                last_hh = swing_highs[-1][1] > swing_highs[-2][1]
                last_lh = swing_highs[-1][1] < swing_highs[-2][1]

            if len(swing_lows) >= 2:
                last_hl = swing_lows[-1][1] > swing_lows[-2][1]
                last_ll = swing_lows[-1][1] < swing_lows[-2][1]

            if last_hh and last_hl:
                structure = "uptrend"
            elif last_lh and last_ll:
                structure = "downtrend"
            elif last_hh and last_ll:
                structure = "expanding"
            elif last_lh and last_hl:
                structure = "contracting"
            else:
                structure = "sideways"

            return {"swing_highs": swing_highs[-5:], "swing_lows": swing_lows[-5:],
                    "structure": structure, "last_hh": last_hh, "last_hl": last_hl,
                    "last_lh": last_lh, "last_ll": last_ll}
        except Exception:
            return {"swing_highs": [], "swing_lows": [], "structure": "unknown",
                    "last_hh": False, "last_hl": False, "last_lh": False, "last_ll": False}

    def _analyze_momentum_quality(self, df):
        """Deeper momentum analysis — acceleration, deceleration, exhaustion."""
        if df is None or len(df) < 20:
            return {"accel": 0, "decel": False, "exhaustion": False, "impulse_strength": 0}

        try:
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            n = len(close)
            bi = n - 2

            # Momentum acceleration: ROC of ROC
            if bi >= 5:
                roc_now = (close[bi] - close[bi-3]) / close[bi-3] * 100 if close[bi-3] != 0 else 0
                roc_prev = (close[bi-3] - close[bi-6]) / close[bi-6] * 100 if bi >= 6 and close[bi-6] != 0 else 0
                accel = roc_now - roc_prev  # positive = accelerating
            else:
                accel = 0; roc_now = 0

            # Deceleration: momentum slowing (still positive but decreasing)
            decel = abs(roc_now) > 0.1 and abs(accel) > 0.05 and (roc_now * accel < 0)

            # Exhaustion: big candles with wicks (rejection)
            body = abs(close[bi] - close[bi-1]) if bi > 0 else 0
            total_range = high[bi] - low[bi]
            wick_ratio = 1 - (body / total_range) if total_range > 0 else 0
            exhaustion = wick_ratio > 0.6 and total_range > np.mean(high[bi-10:bi] - low[bi-10:bi]) * 1.5

            # Impulse strength: how clean is the move (body / range ratio over last 5 bars)
            bodies = np.abs(close[bi-4:bi+1] - np.roll(close, 1)[bi-4:bi+1]) if bi >= 4 else np.array([0])
            ranges = high[bi-4:bi+1] - low[bi-4:bi+1] if bi >= 4 else np.array([1])
            impulse_strength = float(np.mean(bodies / np.maximum(ranges, 0.0001)))

            return {"accel": round(accel, 4), "decel": decel, "exhaustion": exhaustion,
                    "impulse_strength": round(impulse_strength, 3)}
        except Exception:
            return {"accel": 0, "decel": False, "exhaustion": False, "impulse_strength": 0}

    def _analyze_order_flow(self, df):
        """Analyze buying vs selling pressure from candle microstructure."""
        if df is None or len(df) < 10:
            return {"buy_pressure": 0.5, "sell_pressure": 0.5, "delta": 0, "absorption": False}

        try:
            close = df["close"].values.astype(float)
            opn = df["open"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            vol = df["tick_volume"].values.astype(float) if "tick_volume" in df else np.ones(len(close))
            n = len(close)

            # Last 10 bars: estimate buying vs selling volume from candle structure
            buy_vol = 0; sell_vol = 0
            for i in range(max(0, n-10), n):
                rng = high[i] - low[i]
                if rng == 0: continue
                # Buy volume approximation: (close - low) / range * volume
                buy_frac = (close[i] - low[i]) / rng
                sell_frac = (high[i] - close[i]) / rng
                buy_vol += buy_frac * vol[i]
                sell_vol += sell_frac * vol[i]

            total = buy_vol + sell_vol
            buy_pressure = buy_vol / total if total > 0 else 0.5
            sell_pressure = sell_vol / total if total > 0 else 0.5
            delta = buy_pressure - sell_pressure  # positive = net buying

            # Absorption: high volume but no price movement (big players absorbing)
            recent_vol = np.mean(vol[-5:])
            avg_vol = np.mean(vol[-20:]) if len(vol) >= 20 else recent_vol
            price_move = abs(close[-1] - close[-5]) if len(close) >= 5 else 0
            avg_move = np.mean(np.abs(np.diff(close[-20:]))) * 5 if len(close) >= 20 else price_move or 1
            absorption = recent_vol > avg_vol * 1.5 and price_move < avg_move * 0.3

            return {"buy_pressure": round(buy_pressure, 3), "sell_pressure": round(sell_pressure, 3),
                    "delta": round(delta, 3), "absorption": absorption}
        except Exception:
            return {"buy_pressure": 0.5, "sell_pressure": 0.5, "delta": 0, "absorption": False}

    # =================================================================
    #  7. LIQUIDITY ZONES (prev day H/L, round numbers, weekly open/close)
    # =================================================================

    def _detect_liquidity_zones(self, candles: dict, symbol: str,
                                direction: str) -> dict:
        """Detect institutional liquidity zones.

        Zones: previous day high/low, round-number levels, weekly open/close.
        Returns dict with zone list, proximity score, and magnet/reversal flags.
        """
        default = {"zones": [], "proximity": 0.0, "nearest_dist": 999999.0,
                   "at_liquidity": False, "magnet_above": 0.0, "magnet_below": 0.0}
        try:
            h1_df = candles.get(60)
            if h1_df is None or len(h1_df) < 30:
                return default

            c = h1_df["close"].values.astype(np.float64)
            h = h1_df["high"].values.astype(np.float64)
            l = h1_df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2
            price = float(c[bi])
            atr_arr = _atr(h, l, c, 14)
            atr_val = _safe(atr_arr, bi, 0.0001)

            zones = []  # list of (level, label, strength)

            # --- Previous day high/low ---
            # Approximate: last 24 H1 bars = 1 day
            if bi >= 48:
                prev_day_h = h[bi - 48:bi - 24]
                prev_day_l = l[bi - 48:bi - 24]
                if len(prev_day_h) > 0:
                    pdh = float(np.max(prev_day_h))
                    pdl = float(np.min(prev_day_l))
                    zones.append((pdh, "prev_day_high", 1.0))
                    zones.append((pdl, "prev_day_low", 1.0))
            elif bi >= 24:
                prev_day_h = h[max(0, bi - 48):bi - 12]
                prev_day_l = l[max(0, bi - 48):bi - 12]
                if len(prev_day_h) > 0:
                    pdh = float(np.max(prev_day_h))
                    pdl = float(np.min(prev_day_l))
                    zones.append((pdh, "prev_day_high", 0.8))
                    zones.append((pdl, "prev_day_low", 0.8))

            # --- Weekly open/close ---
            # Approximate: last 120 H1 bars = 5 trading days
            if bi >= 120:
                weekly_open = float(c[bi - 120])
                zones.append((weekly_open, "weekly_open", 0.7))
            if bi >= 24:
                # Today's open (approximate)
                today_open = float(c[max(0, bi - 24)])
                zones.append((today_open, "session_open", 0.6))

            # --- Round number levels ---
            sym_base = symbol.replace(".", "").replace("#", "").upper()
            increments = None
            for key, inc in _ROUND_LEVELS.items():
                if key in sym_base:
                    increments = inc
                    break
            if increments is None:
                # Auto-detect: use magnitude-based rounds
                if price > 1000:
                    increments = [100, 50]
                elif price > 100:
                    increments = [10, 5]
                elif price > 10:
                    increments = [1, 0.5]
                else:
                    increments = [0.01, 0.005]

            for inc in increments:
                if inc <= 0:
                    continue
                base = round(price / inc) * inc
                for offset_mult in range(-3, 4):
                    level = base + offset_mult * inc
                    dist = abs(level - price)
                    if dist < atr_val * 5:  # within 5 ATR range
                        strength = 0.9 if offset_mult == 0 else 0.6
                        zones.append((level, f"round_{level:.4f}", strength))

            # --- Compute proximity and magnet analysis ---
            if not zones:
                return default

            nearest_dist = 999999.0
            magnet_above = 0.0
            magnet_below = 0.0
            zone_list = []

            for level, label, strength in zones:
                dist = abs(level - price)
                norm_dist = dist / atr_val if atr_val > 0 else 999.0
                zone_list.append({
                    "level": round(level, 5),
                    "label": label,
                    "distance": round(dist, 6),
                    "atr_dist": round(norm_dist, 2),
                    "strength": strength,
                })
                if dist < nearest_dist:
                    nearest_dist = dist

                # Magnet effect: nearby zones pull price toward them
                if norm_dist < 3.0:
                    pull = strength / max(norm_dist, 0.1)
                    if level > price:
                        magnet_above = max(magnet_above, pull)
                    else:
                        magnet_below = max(magnet_below, pull)

            proximity = min(1.0, atr_val / max(nearest_dist, 0.0001))
            at_liquidity = nearest_dist < atr_val * 0.3

            # Sort by distance, keep closest 10
            zone_list.sort(key=lambda z: z["distance"])
            zone_list = zone_list[:10]

            return {
                "zones": zone_list,
                "proximity": round(proximity, 3),
                "nearest_dist": round(nearest_dist, 6),
                "at_liquidity": at_liquidity,
                "magnet_above": round(magnet_above, 3),
                "magnet_below": round(magnet_below, 3),
            }

        except Exception as e:
            log.warning("Liquidity zones error: %s", e)
            return default

    # =================================================================
    #  8. FIBONACCI RETRACEMENT
    # =================================================================

    def _compute_fibonacci_levels(self, candles: dict, direction: str) -> dict:
        """Auto-detect last major swing and compute Fibonacci levels.

        Returns fib levels as price values plus alignment score for TP/SL.
        """
        default = {"levels": {}, "swing_high": 0.0, "swing_low": 0.0,
                   "trend_fib": "none", "nearest_fib": 0.0, "nearest_fib_dist": 999999.0,
                   "fib_cluster_sl": 0.0, "fib_cluster_tp": 0.0}
        try:
            h1_df = candles.get(60)
            if h1_df is None or len(h1_df) < 40:
                return default

            c = h1_df["close"].values.astype(np.float64)
            h = h1_df["high"].values.astype(np.float64)
            l = h1_df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2
            price = float(c[bi])

            # Find last major swing: use last 60 H1 bars
            lookback = min(60, bi)
            seg_h = h[bi - lookback:bi + 1]
            seg_l = l[bi - lookback:bi + 1]

            swing_high = float(np.max(seg_h))
            swing_low = float(np.min(seg_l))
            swing_range = swing_high - swing_low

            if swing_range < 1e-10:
                return default

            # Determine if this is an upswing or downswing
            high_idx = int(np.argmax(seg_h))
            low_idx = int(np.argmin(seg_l))

            # Compute fibs from the major swing
            fib_levels = {}
            if high_idx > low_idx:
                # Upswing: retracement down from high
                trend_fib = "upswing"
                for ratio in _FIB_LEVELS:
                    level = swing_high - ratio * swing_range
                    fib_levels[f"fib_{ratio:.3f}"] = round(level, 5)
                # Extensions
                fib_levels["fib_1.000"] = round(swing_low, 5)
                fib_levels["fib_1.272"] = round(swing_high - 1.272 * swing_range, 5)
                fib_levels["fib_1.618"] = round(swing_high - 1.618 * swing_range, 5)
            else:
                # Downswing: retracement up from low
                trend_fib = "downswing"
                for ratio in _FIB_LEVELS:
                    level = swing_low + ratio * swing_range
                    fib_levels[f"fib_{ratio:.3f}"] = round(level, 5)
                fib_levels["fib_1.000"] = round(swing_high, 5)
                fib_levels["fib_1.272"] = round(swing_low + 1.272 * swing_range, 5)
                fib_levels["fib_1.618"] = round(swing_low + 1.618 * swing_range, 5)

            # Find nearest fib to current price
            nearest_fib = 0.0
            nearest_fib_dist = 999999.0
            for label, level in fib_levels.items():
                d = abs(level - price)
                if d < nearest_fib_dist:
                    nearest_fib_dist = d
                    nearest_fib = level

            # Fib-based SL and TP suggestions
            atr_arr = _atr(h, l, c, 14)
            atr_val = _safe(atr_arr, bi, 0.0001)
            fib_cluster_sl = 0.0
            fib_cluster_tp = 0.0

            if direction == "LONG":
                # SL: nearest fib level below price
                sl_fibs = [lv for lv in fib_levels.values() if lv < price]
                if sl_fibs:
                    fib_cluster_sl = price - max(sl_fibs)
                # TP: nearest fib level above price
                tp_fibs = [lv for lv in fib_levels.values() if lv > price]
                if tp_fibs:
                    fib_cluster_tp = min(tp_fibs) - price
            elif direction == "SHORT":
                sl_fibs = [lv for lv in fib_levels.values() if lv > price]
                if sl_fibs:
                    fib_cluster_sl = min(sl_fibs) - price
                tp_fibs = [lv for lv in fib_levels.values() if lv < price]
                if tp_fibs:
                    fib_cluster_tp = price - max(tp_fibs)

            return {
                "levels": fib_levels,
                "swing_high": round(swing_high, 5),
                "swing_low": round(swing_low, 5),
                "trend_fib": trend_fib,
                "nearest_fib": round(nearest_fib, 5),
                "nearest_fib_dist": round(nearest_fib_dist, 6),
                "fib_cluster_sl": round(max(fib_cluster_sl, 0.0), 6),
                "fib_cluster_tp": round(max(fib_cluster_tp, 0.0), 6),
            }

        except Exception as e:
            log.warning("Fibonacci levels error: %s", e)
            return default

    # =================================================================
    #  9. MARKET SESSION OVERLAP DETECTION
    # =================================================================

    def _detect_session_context(self) -> dict:
        """Detect current market session and overlaps.

        Returns session info, overlap status, and time-based scoring adjustments.
        """
        default = {"session": "off_hours", "overlap": "none",
                   "session_score": 0.0, "minutes_to_close": 999,
                   "avoid_entry": False, "reason": ""}
        try:
            now = datetime.now(timezone.utc)
            hour = now.hour
            minute = now.minute
            weekday = now.weekday()  # 0=Monday

            # Weekend check
            if weekday >= 5:
                return {**default, "session": "weekend", "avoid_entry": True,
                        "reason": "weekend_market_closed"}

            # Determine active sessions
            in_asia = _SESSION_ASIA_START <= hour < _SESSION_ASIA_END
            in_london = _SESSION_LONDON_START <= hour < _SESSION_LONDON_END
            in_ny = _SESSION_NY_START <= hour < _SESSION_NY_END

            # Determine overlaps
            in_london_ny = _LONDON_NY_OVERLAP_START <= hour < _LONDON_NY_OVERLAP_END
            in_asia_london = _ASIA_LONDON_OVERLAP_START <= hour < _ASIA_LONDON_OVERLAP_END

            # Session naming
            if in_london_ny:
                session = "london_ny_overlap"
                overlap = "london_ny"
            elif in_asia_london:
                session = "asia_london_overlap"
                overlap = "asia_london"
            elif in_london:
                session = "london"
                overlap = "none"
            elif in_ny:
                session = "new_york"
                overlap = "none"
            elif in_asia:
                session = "asian"
                overlap = "none"
            else:
                session = "off_hours"
                overlap = "none"

            # Session quality score (0-1): how favorable is this time for trading
            session_score_map = {
                "london_ny_overlap": 1.0,   # Best: highest volume + volatility
                "asia_london_overlap": 0.8,  # Good: Europe waking up
                "london": 0.85,             # Strong: London is major hub
                "new_york": 0.8,            # Good: US session
                "asian": 0.5,               # Ranging: low vol for most pairs
                "off_hours": 0.2,           # Avoid: thin liquidity
                "weekend": 0.0,
            }
            session_score = session_score_map.get(session, 0.3)

            # Minutes to session close (relevant session)
            minutes_to_close = 999
            if in_london and not in_ny:
                minutes_to_close = (_SESSION_LONDON_END - hour) * 60 - minute
            elif in_ny:
                minutes_to_close = (_SESSION_NY_END - hour) * 60 - minute
            elif in_asia:
                minutes_to_close = (_SESSION_ASIA_END - hour) * 60 - minute

            # Avoid entry within 5 min of session close
            avoid_entry = False
            reason = ""
            if 0 < minutes_to_close <= 5:
                avoid_entry = True
                reason = f"session_closing_in_{minutes_to_close}m"
            elif session == "off_hours":
                avoid_entry = True
                reason = "off_hours_thin_liquidity"

            # Avoid entry right at major session opens (first 2 min = spread spike)
            session_opens = [_SESSION_LONDON_START, _SESSION_NY_START, _SESSION_ASIA_START]
            for open_hour in session_opens:
                if hour == open_hour and minute < 2:
                    avoid_entry = True
                    reason = f"session_open_spread_spike_{open_hour}UTC"
                    break

            return {
                "session": session,
                "overlap": overlap,
                "session_score": round(session_score, 2),
                "minutes_to_close": minutes_to_close,
                "avoid_entry": avoid_entry,
                "reason": reason,
            }

        except Exception as e:
            log.warning("Session context error: %s", e)
            return default

    # =================================================================
    #  10. CANDLE PATTERN RECOGNITION (M15 + H1)
    # =================================================================

    def _detect_candle_patterns(self, df: Optional[pd.DataFrame],
                                tf_label: str = "H1") -> dict:
        """Detect institutional-grade candlestick patterns.

        Patterns: engulfing, pin bar, inside bar, morning/evening star.
        Returns pattern list with direction and strength.
        """
        default = {"patterns": [], "bullish_count": 0, "bearish_count": 0,
                   "net_signal": 0.0}
        if df is None or len(df) < 10:
            return default
        try:
            c = df["close"].values.astype(np.float64)
            o = df["open"].values.astype(np.float64)
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2  # last completed bar

            if bi < 3:
                return default

            patterns = []

            # Helper: bar properties
            def _bar(i):
                body = c[i] - o[i]
                abs_body = abs(body)
                rng = h[i] - l[i]
                upper_wick = h[i] - max(c[i], o[i])
                lower_wick = min(c[i], o[i]) - l[i]
                is_bull = body > 0
                body_ratio = abs_body / rng if rng > 0 else 0
                return body, abs_body, rng, upper_wick, lower_wick, is_bull, body_ratio

            b0 = _bar(bi)      # current completed
            b1 = _bar(bi - 1)  # previous
            b2 = _bar(bi - 2)  # 2 bars ago

            body0, abs_body0, rng0, uw0, lw0, bull0, br0 = b0
            body1, abs_body1, rng1, uw1, lw1, bull1, br1 = b1
            body2, abs_body2, rng2, uw2, lw2, bull2, br2 = b2

            # --- ENGULFING ---
            # Bullish engulfing
            if not bull1 and bull0 and o[bi] <= c[bi - 1] and c[bi] >= o[bi - 1]:
                strength = 0.8 if br0 > 0.6 else 0.6
                # Stronger if preceded by downtrend (3 bearish bars before)
                if bi >= 4:
                    prior_bearish = sum(1 for k in range(bi - 4, bi - 1) if c[k] < o[k])
                    if prior_bearish >= 2:
                        strength = min(strength + 0.15, 1.0)
                patterns.append({"name": "bullish_engulfing", "direction": "LONG",
                                 "strength": round(strength, 2), "tf": tf_label})

            # Bearish engulfing
            if bull1 and not bull0 and o[bi] >= c[bi - 1] and c[bi] <= o[bi - 1]:
                strength = 0.8 if br0 > 0.6 else 0.6
                if bi >= 4:
                    prior_bullish = sum(1 for k in range(bi - 4, bi - 1) if c[k] > o[k])
                    if prior_bullish >= 2:
                        strength = min(strength + 0.15, 1.0)
                patterns.append({"name": "bearish_engulfing", "direction": "SHORT",
                                 "strength": round(strength, 2), "tf": tf_label})

            # --- PIN BAR (Hammer / Shooting Star) ---
            if rng0 > 0:
                # Bullish pin bar (hammer): long lower wick, small body at top
                if lw0 > abs_body0 * 2.0 and lw0 > rng0 * 0.6 and uw0 < rng0 * 0.15:
                    strength = 0.7 if lw0 > abs_body0 * 3.0 else 0.55
                    patterns.append({"name": "bullish_pin_bar", "direction": "LONG",
                                     "strength": round(strength, 2), "tf": tf_label})

                # Bearish pin bar (shooting star): long upper wick, small body at bottom
                if uw0 > abs_body0 * 2.0 and uw0 > rng0 * 0.6 and lw0 < rng0 * 0.15:
                    strength = 0.7 if uw0 > abs_body0 * 3.0 else 0.55
                    patterns.append({"name": "bearish_pin_bar", "direction": "SHORT",
                                     "strength": round(strength, 2), "tf": tf_label})

            # --- INSIDE BAR ---
            if h[bi] <= h[bi - 1] and l[bi] >= l[bi - 1]:
                # Inside bar = compression / breakout pending
                # Direction comes from breakout, but flag it
                strength = 0.5
                if rng0 < rng1 * 0.5:
                    strength = 0.65  # very tight compression
                patterns.append({"name": "inside_bar", "direction": "FLAT",
                                 "strength": round(strength, 2), "tf": tf_label})

            # --- MORNING STAR (bullish 3-bar reversal) ---
            if bi >= 3:
                # Bar -2: bearish, Bar -1: small body (doji-like), Bar 0: bullish
                if not bull2 and bull0:
                    # Middle bar should be small relative to the other two
                    if abs_body1 < abs_body2 * 0.4 and abs_body1 < abs_body0 * 0.4:
                        # Close of bar 0 should be > midpoint of bar -2
                        mid2 = (o[bi - 2] + c[bi - 2]) / 2.0
                        if c[bi] > mid2:
                            strength = 0.75
                            patterns.append({"name": "morning_star", "direction": "LONG",
                                             "strength": strength, "tf": tf_label})

            # --- EVENING STAR (bearish 3-bar reversal) ---
            if bi >= 3:
                if bull2 and not bull0:
                    if abs_body1 < abs_body2 * 0.4 and abs_body1 < abs_body0 * 0.4:
                        mid2 = (o[bi - 2] + c[bi - 2]) / 2.0
                        if c[bi] < mid2:
                            strength = 0.75
                            patterns.append({"name": "evening_star", "direction": "SHORT",
                                             "strength": strength, "tf": tf_label})

            # --- THREE WHITE SOLDIERS / THREE BLACK CROWS ---
            if bi >= 3:
                if all(c[bi - k] > o[bi - k] for k in range(3)):
                    # 3 bullish bars with higher closes
                    if c[bi] > c[bi - 1] > c[bi - 2]:
                        patterns.append({"name": "three_white_soldiers", "direction": "LONG",
                                         "strength": 0.7, "tf": tf_label})
                if all(c[bi - k] < o[bi - k] for k in range(3)):
                    if c[bi] < c[bi - 1] < c[bi - 2]:
                        patterns.append({"name": "three_black_crows", "direction": "SHORT",
                                         "strength": 0.7, "tf": tf_label})

            # Summarize
            bullish_count = sum(1 for p in patterns if p["direction"] == "LONG")
            bearish_count = sum(1 for p in patterns if p["direction"] == "SHORT")
            bull_str = sum(p["strength"] for p in patterns if p["direction"] == "LONG")
            bear_str = sum(p["strength"] for p in patterns if p["direction"] == "SHORT")
            net_signal = (bull_str - bear_str) / max(bull_str + bear_str, 1.0)

            return {
                "patterns": patterns,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "net_signal": round(net_signal, 3),
            }

        except Exception as e:
            log.warning("Candle pattern (%s) error: %s", tf_label, e)
            return default

    # =================================================================
    #  11. MULTI-TF RSI DIVERGENCE (H1 + M15)
    # =================================================================

    def _detect_mtf_divergence(self, candles: dict) -> dict:
        """Detect RSI divergence across H1 and M15.

        Regular divergence = reversal signal.
        Hidden divergence = trend continuation signal.
        """
        default = {"h1_divergence": "none", "m15_divergence": "none",
                   "combined": "none", "strength": 0.0}
        try:
            result = {}
            for tf_key, tf_label in [(60, "h1"), (15, "m15")]:
                df = candles.get(tf_key)
                if df is None or len(df) < 40:
                    result[tf_label] = "none"
                    continue

                c = df["close"].values.astype(np.float64)
                h = df["high"].values.astype(np.float64)
                l_arr = df["low"].values.astype(np.float64)
                n = len(c)
                bi = n - 2
                rsi_arr = _rsi(c, 14)

                if bi < 20:
                    result[tf_label] = "none"
                    continue

                # Find swing highs/lows in last 20 bars
                lookback = 20
                start = bi - lookback

                price_swing_highs = []
                price_swing_lows = []
                rsi_swing_highs = []
                rsi_swing_lows = []

                for i in range(start + 1, bi):
                    if i < 1 or i + 1 >= n:
                        continue
                    # Swing high
                    if h[i] >= h[i - 1] and h[i] >= h[i + 1]:
                        price_swing_highs.append((i, float(h[i])))
                        rsi_swing_highs.append((i, _safe(rsi_arr, i, 50.0)))
                    # Swing low
                    if l_arr[i] <= l_arr[i - 1] and l_arr[i] <= l_arr[i + 1]:
                        price_swing_lows.append((i, float(l_arr[i])))
                        rsi_swing_lows.append((i, _safe(rsi_arr, i, 50.0)))

                div_type = "none"

                # Need at least 2 swings to compare
                if len(price_swing_highs) >= 2 and len(rsi_swing_highs) >= 2:
                    ph1, ph2 = price_swing_highs[-2][1], price_swing_highs[-1][1]
                    rh1, rh2 = rsi_swing_highs[-2][1], rsi_swing_highs[-1][1]

                    # Regular bearish: price HH, RSI LH
                    if ph2 > ph1 and rh2 < rh1:
                        div_type = "regular_bearish"
                    # Hidden bullish: price LH, RSI HH (trend continuation for longs)
                    elif ph2 < ph1 and rh2 > rh1:
                        div_type = "hidden_bullish"

                if len(price_swing_lows) >= 2 and len(rsi_swing_lows) >= 2:
                    pl1, pl2 = price_swing_lows[-2][1], price_swing_lows[-1][1]
                    rl1, rl2 = rsi_swing_lows[-2][1], rsi_swing_lows[-1][1]

                    # Regular bullish: price LL, RSI HL
                    if pl2 < pl1 and rl2 > rl1:
                        if div_type == "none":
                            div_type = "regular_bullish"
                        else:
                            div_type = "conflicting"
                    # Hidden bearish: price HL, RSI LL (trend continuation for shorts)
                    elif pl2 > pl1 and rl2 < rl1:
                        if div_type == "none":
                            div_type = "hidden_bearish"
                        elif div_type not in ("hidden_bearish",):
                            div_type = "conflicting"

                result[tf_label] = div_type

            h1_div = result.get("h1", "none")
            m15_div = result.get("m15", "none")

            # Combined assessment
            combined = "none"
            strength = 0.0

            if h1_div != "none" and m15_div != "none":
                if h1_div == m15_div:
                    combined = h1_div
                    strength = 0.9  # both TFs agree = very strong
                elif "regular" in h1_div and "regular" in m15_div:
                    combined = "conflicting"
                    strength = 0.3
                else:
                    combined = h1_div  # H1 takes precedence
                    strength = 0.6
            elif h1_div != "none":
                combined = h1_div
                strength = 0.7
            elif m15_div != "none":
                combined = m15_div
                strength = 0.5

            return {
                "h1_divergence": h1_div,
                "m15_divergence": m15_div,
                "combined": combined,
                "strength": round(strength, 2),
            }

        except Exception as e:
            log.warning("MTF divergence error: %s", e)
            return default

    # =================================================================
    #  12. VOLATILITY EXPANSION/CONTRACTION CYCLE (BB Squeeze)
    # =================================================================

    def _detect_volatility_cycle(self, candles: dict) -> dict:
        """Detect Bollinger Band squeeze and expansion states.

        Squeeze = low vol contraction, precedes breakout.
        Expansion = breakout in progress.
        Returns state, direction bias, and squeeze duration.
        """
        default = {"state": "normal", "squeeze": False, "squeeze_bars": 0,
                   "expansion": False, "breakout_dir": "FLAT",
                   "bb_width": 0.0, "bb_pctb": 0.5, "keltner_squeeze": False}
        try:
            h1_df = candles.get(60)
            if h1_df is None or len(h1_df) < 30:
                return default

            c = h1_df["close"].values.astype(np.float64)
            h = h1_df["high"].values.astype(np.float64)
            l = h1_df["low"].values.astype(np.float64)
            n = len(c)
            bi = n - 2

            if bi < 25:
                return default

            # Bollinger Bands (20, 2)
            bb_upper, bb_mid, bb_lower, bb_bw, bb_pctb = _bollinger_bands(c, 20, 2.0)

            curr_bw = _safe(bb_bw, bi, 0.02)
            curr_pctb = _safe(bb_pctb, bi, 0.5)

            # Keltner Channel (20, 1.5) for squeeze detection
            kc_mid = _ema(c, 20)
            atr_arr = _atr(h, l, c, 10)
            kc_upper = np.zeros(n, dtype=np.float64)
            kc_lower = np.zeros(n, dtype=np.float64)
            for i in range(n):
                kc_upper[i] = _safe(kc_mid, i) + 1.5 * _safe(atr_arr, i, 0.0001)
                kc_lower[i] = _safe(kc_mid, i) - 1.5 * _safe(atr_arr, i, 0.0001)

            # Squeeze: BB inside Keltner Channel
            keltner_squeeze = (_safe(bb_lower, bi) > _safe(kc_lower, bi) and
                               _safe(bb_upper, bi) < _safe(kc_upper, bi))

            # Count squeeze duration
            squeeze_bars = 0
            if keltner_squeeze:
                for j in range(bi, max(0, bi - 30), -1):
                    bb_l = _safe(bb_lower, j)
                    bb_u = _safe(bb_upper, j)
                    kc_l = _safe(kc_lower, j)
                    kc_u = _safe(kc_upper, j)
                    if bb_l > kc_l and bb_u < kc_u:
                        squeeze_bars += 1
                    else:
                        break

            # BB width percentile: compare current width to last 50 bars
            bw_lookback = min(50, bi)
            bw_history = bb_bw[bi - bw_lookback:bi + 1]
            bw_history = bw_history[~np.isnan(bw_history)]
            if len(bw_history) > 5:
                bw_percentile = float(np.sum(bw_history < curr_bw)) / len(bw_history)
            else:
                bw_percentile = 0.5

            # State classification
            squeeze = keltner_squeeze or bw_percentile < 0.15
            expansion = bw_percentile > 0.85

            # Breakout direction from squeeze
            breakout_dir = "FLAT"
            if squeeze or expansion:
                # Use momentum direction during squeeze
                ml, sl, hist = _macd(c, 12, 26, 9)
                macd_val = _safe(hist, bi)
                if macd_val > 0:
                    breakout_dir = "LONG"
                elif macd_val < 0:
                    breakout_dir = "SHORT"

            if squeeze:
                state = "squeeze"
            elif expansion:
                state = "expansion"
            elif bw_percentile < 0.3:
                state = "contracting"
            elif bw_percentile > 0.7:
                state = "expanding"
            else:
                state = "normal"

            return {
                "state": state,
                "squeeze": squeeze,
                "squeeze_bars": squeeze_bars,
                "expansion": expansion,
                "breakout_dir": breakout_dir,
                "bb_width": round(curr_bw * 100, 3),  # percentage
                "bb_pctb": round(curr_pctb, 3),
                "keltner_squeeze": keltner_squeeze,
            }

        except Exception as e:
            log.warning("Volatility cycle error: %s", e)
            return default

    # =================================================================
    #  13. TIME-WEIGHTED ENTRY SCORING
    # =================================================================

    def _compute_time_weight(self, session_ctx: dict) -> float:
        """Weight entry quality by time context.

        Returns multiplier 0.0 - 1.0 to apply to entry quality.
        1.0 = optimal time, 0.0 = do not enter.
        """
        try:
            if session_ctx.get("avoid_entry", False):
                return 0.1  # near-zero but not hard block (user rule: never skip)

            session_score = session_ctx.get("session_score", 0.5)
            minutes_to_close = session_ctx.get("minutes_to_close", 999)

            weight = session_score

            # Penalize entries close to session close (last 15 min)
            if 0 < minutes_to_close <= 15:
                weight *= max(0.3, minutes_to_close / 15.0)

            # Penalize entries in first 5 min of session (spread stabilization)
            now = datetime.now(timezone.utc)
            minute = now.minute
            hour = now.hour
            session_opens = [_SESSION_LONDON_START, _SESSION_NY_START]
            for open_hour in session_opens:
                if hour == open_hour and minute < 5:
                    weight *= max(0.5, minute / 5.0)
                    break

            # Bonus for round-hour entries (institutional order flow clusters)
            if minute <= 2 or minute >= 58:
                weight = min(weight * 1.1, 1.0)

            return round(max(0.0, min(1.0, weight)), 3)

        except Exception as e:
            log.warning("Time weight error: %s", e)
            return 0.7  # safe default

    # =================================================================
    # =================================================================
    #  CORRELATION REGIME (cross-asset from available symbols)
    # =================================================================

    def _detect_correlation_regime(self, symbol, dominant_dir):
        """Detect if all symbols are moving together (risk-on/risk-off regime).
        High correlation = systemic move, individual signal less reliable."""
        try:
            returns = {}
            for sym in ["XAUUSD", "BTCUSD", "NAS100.r", "USDJPY", "XAGUSD", "JPN225ft"]:
                df = self.state.get_candles(sym, 60)
                if df is not None and len(df) >= 20:
                    c = df["close"].values.astype(float)
                    ret = (c[-1] - c[-10]) / c[-10] if c[-10] != 0 else 0
                    returns[sym] = ret

            if len(returns) < 3:
                return {"regime": "unknown", "avg_corr": 0, "risk_mode": "neutral", "all_same_dir": False}

            # Check if all symbols moving same direction
            dirs = [1 if r > 0 else -1 for r in returns.values()]
            up = sum(1 for d in dirs if d > 0)
            dn = sum(1 for d in dirs if d < 0)
            total = len(dirs)
            all_same = up >= total * 0.8 or dn >= total * 0.8

            # Risk mode
            gold_up = returns.get("XAUUSD", 0) > 0
            btc_up = returns.get("BTCUSD", 0) > 0
            nas_up = returns.get("NAS100.r", 0) > 0
            jpy_dn = returns.get("USDJPY", 0) > 0  # USD/JPY up = risk-on

            if nas_up and btc_up and jpy_dn and not gold_up:
                risk_mode = "risk_on"
            elif gold_up and not nas_up and not jpy_dn:
                risk_mode = "risk_off"
            else:
                risk_mode = "neutral"

            # Simple average absolute return as correlation proxy
            vals = list(returns.values())
            avg_abs = sum(abs(v) for v in vals) / len(vals) if vals else 0

            return {
                "regime": risk_mode,
                "avg_move": round(avg_abs * 100, 3),
                "risk_mode": risk_mode,
                "all_same_dir": all_same,
                "up_pct": round(up / total * 100),
                "returns": {k: round(v * 100, 3) for k, v in returns.items()},
            }
        except Exception:
            return {"regime": "unknown", "avg_move": 0, "risk_mode": "neutral", "all_same_dir": False}

    # =================================================================
    #  ADAPTIVE TIMEFRAME (which TF is most reliable right now)
    # =================================================================

    def _detect_best_timeframe(self, symbol):
        """Auto-detect which timeframe has the cleanest signals based on
        recent signal-to-noise ratio (body/range ratio over last 10 bars)."""
        try:
            tf_scores = {}
            for tf in [1, 5, 15, 60]:
                df = self.state.get_candles(symbol, tf)
                if df is None or len(df) < 15:
                    continue
                c = df["close"].values.astype(float)
                o = df["open"].values.astype(float)
                h = df["high"].values.astype(float)
                l = df["low"].values.astype(float)
                n = len(c)

                # Signal-to-noise: average body/range over last 10 bars
                bodies = np.abs(c[-10:] - o[-10:])
                ranges = h[-10:] - l[-10:]
                ranges = np.maximum(ranges, 0.0001)
                snr = float(np.mean(bodies / ranges))

                # Trend consistency: how many bars agree with direction
                direction = 1 if c[-1] > c[-5] else -1
                agrees = sum(1 for i in range(-5, 0) if (c[i] > c[i - 1]) == (direction > 0))
                consistency = agrees / 5.0

                # Combined score
                tf_scores[tf] = round(snr * 0.6 + consistency * 0.4, 3)

            best_tf = max(tf_scores, key=tf_scores.get) if tf_scores else 60
            return {
                "best_tf": best_tf,
                "tf_scores": tf_scores,
                "h1_score": tf_scores.get(60, 0),
                "m15_score": tf_scores.get(15, 0),
                "m5_score": tf_scores.get(5, 0),
                "m1_score": tf_scores.get(1, 0),
            }
        except Exception:
            return {"best_tf": 60, "tf_scores": {}, "h1_score": 0, "m15_score": 0, "m5_score": 0, "m1_score": 0}

    # =================================================================
    #  M1 MICROSTRUCTURE NOISE FILTER
    # =================================================================

    def _filter_m1_noise(self, symbol):
        """Filter M1 signals by spread pattern and tick frequency.
        High spread + low tick frequency = noise, not signal."""
        try:
            df = self.state.get_candles(symbol, 1)
            if df is None or len(df) < 10:
                return {"noise_level": 0.5, "spread_stable": True, "tick_active": True}

            h = df["high"].values.astype(float)
            l = df["low"].values.astype(float)
            c = df["close"].values.astype(float)
            v = df["tick_volume"].values.astype(float) if "tick_volume" in df else np.ones(len(c))

            # Spread proxy from M1 candles (high-low vs body)
            ranges = h[-10:] - l[-10:]
            bodies = np.abs(c[-10:] - np.roll(c, 1)[-10:])
            spread_ratio = float(np.mean(ranges / np.maximum(bodies, 0.0001)))

            # Tick volume trend
            vol_recent = float(np.mean(v[-5:])) if len(v) >= 5 else 0
            vol_avg = float(np.mean(v[-20:])) if len(v) >= 20 else vol_recent or 1
            tick_ratio = vol_recent / vol_avg if vol_avg > 0 else 1

            # Noise level: high spread ratio + low volume = noisy
            noise = 0.0
            if spread_ratio > 3.0:
                noise += 0.3  # very wide spreads relative to movement
            if tick_ratio < 0.5:
                noise += 0.3  # low tick activity
            if spread_ratio > 5.0:
                noise += 0.2  # extreme noise

            return {
                "noise_level": round(min(1.0, noise), 2),
                "spread_stable": spread_ratio < 3.0,
                "tick_active": tick_ratio > 0.5,
                "spread_ratio": round(spread_ratio, 2),
                "tick_ratio": round(tick_ratio, 2),
            }
        except Exception:
            return {"noise_level": 0.5, "spread_stable": True, "tick_active": True}

    # =================================================================
    #  MEAN REVERSION DETECTION (counter-trend scalp signal)
    # =================================================================

    def _detect_mean_reversion(self, symbol, dominant_dir):
        """Detect mean reversion setups: BB band touch + RSI extreme.
        These are counter-trend opportunities — only valid in ranging regimes."""
        try:
            df = self.state.get_candles(symbol, 60)
            if df is None or len(df) < 30:
                return {"setup": False, "direction": "FLAT", "strength": 0, "trigger": ""}

            c = df["close"].values.astype(float)
            h = df["high"].values.astype(float)
            l = df["low"].values.astype(float)
            n = len(c)
            bi = n - 2

            # Bollinger Bands
            sma20 = float(np.mean(c[bi - 19:bi + 1]))
            std20 = float(np.std(c[bi - 19:bi + 1]))
            if std20 == 0:
                return {"setup": False, "direction": "FLAT", "strength": 0, "trigger": ""}
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            price = float(c[bi])

            # RSI
            rsi_arr = _rsi(c, 14)
            rsi = float(rsi_arr[bi]) if bi < len(rsi_arr) and not np.isnan(rsi_arr[bi]) else 50

            # Check for mean reversion setup
            setup = False
            direction = "FLAT"
            strength = 0.0
            trigger = ""

            # Bearish MR: price at/above upper BB + RSI > 70
            if price >= bb_upper * 0.998 and rsi > 70:
                setup = True
                direction = "SHORT"
                strength = min(1.0, (rsi - 70) / 20 + (price - bb_upper) / std20 * 0.5)
                trigger = f"BB upper touch + RSI {rsi:.0f}"

            # Bullish MR: price at/below lower BB + RSI < 30
            elif price <= bb_lower * 1.002 and rsi < 30:
                setup = True
                direction = "LONG"
                strength = min(1.0, (30 - rsi) / 20 + (bb_lower - price) / std20 * 0.5)
                trigger = f"BB lower touch + RSI {rsi:.0f}"

            # If MR direction opposes dominant trend, it's a warning
            # If MR direction agrees (rare — trend pullback to BB), it's strong
            return {
                "setup": setup,
                "direction": direction,
                "strength": round(strength, 2),
                "trigger": trigger,
                "bb_upper": round(bb_upper, 5),
                "bb_lower": round(bb_lower, 5),
                "rsi": round(rsi, 1),
                "opposes_trend": setup and direction != dominant_dir,
            }
        except Exception:
            return {"setup": False, "direction": "FLAT", "strength": 0, "trigger": ""}

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
            # Institutional intelligence defaults
            "liquidity": {"zones": [], "proximity": 0.0, "nearest_dist": 999999.0,
                          "at_liquidity": False, "magnet_above": 0.0, "magnet_below": 0.0},
            "fibonacci": {"levels": {}, "swing_high": 0.0, "swing_low": 0.0,
                          "trend_fib": "none", "nearest_fib": 0.0,
                          "nearest_fib_dist": 999999.0,
                          "fib_cluster_sl": 0.0, "fib_cluster_tp": 0.0},
            "session": {"session": "unknown", "overlap": "none",
                        "session_score": 0.5, "minutes_to_close": 999,
                        "avoid_entry": False, "reason": ""},
            "candle_patterns_h1": {"patterns": [], "bullish_count": 0,
                                   "bearish_count": 0, "net_signal": 0.0},
            "candle_patterns_m15": {"patterns": [], "bullish_count": 0,
                                    "bearish_count": 0, "net_signal": 0.0},
            "mtf_divergence": {"h1_divergence": "none", "m15_divergence": "none",
                               "combined": "none", "strength": 0.0},
            "volatility_cycle": {"state": "normal", "squeeze": False,
                                 "squeeze_bars": 0, "expansion": False,
                                 "breakout_dir": "FLAT", "bb_width": 0.0,
                                 "bb_pctb": 0.5, "keltner_squeeze": False},
            "time_weight": 0.7,
            "correlation_regime": {"regime": "unknown", "avg_move": 0, "risk_mode": "neutral", "all_same_dir": False},
            "best_timeframe": {"best_tf": 60, "tf_scores": {}},
            "m1_noise": {"noise_level": 0.5, "spread_stable": True, "tick_active": True},
            "mean_reversion": {"setup": False, "direction": "FLAT", "strength": 0, "trigger": ""},
        }
