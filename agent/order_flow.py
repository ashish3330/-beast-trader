"""
Beast Trader — Order Flow Intelligence.
Detects institutional activity from tick volume patterns on M5/H1 candles.
Provides volume imbalance, absorption, exhaustion, delta divergence,
VWAP, point of control, and high-volume node analysis.
"""
import time
import logging
from typing import Dict, Any, Optional, List

import numpy as np

log = logging.getLogger("beast.orderflow")


class OrderFlowIntel:
    """Institutional order flow detection from tick volume patterns."""

    # Cache TTL in seconds (runs in 500ms brain loop, cache 5s)
    CACHE_TTL = 5.0

    # Tuning constants
    VWAP_LOOKBACK = 20          # bars for VWAP
    POC_LOOKBACK = 50           # bars for point of control
    VOL_PROFILE_BINS = 30       # price bins for volume profile
    HIGH_VOL_TOP_N = 5          # top N high volume nodes to return
    ABSORPTION_VOL_PCTL = 80    # volume percentile for "high volume"
    ABSORPTION_BODY_PCTL = 20   # body-size percentile for "no movement"
    EXHAUSTION_VOL_PCTL = 95    # extreme volume threshold
    DELTA_LOOKBACK = 10         # bars for delta divergence detection
    IMBALANCE_LOOKBACK = 10     # bars for volume imbalance

    def __init__(self, state):
        """
        Args:
            state: SharedState instance from tick_streamer.
        """
        self.state = state
        self._cache: Dict[str, Dict[str, Any]] = {}   # symbol -> {result, ts}

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def get_flow_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Main entry point. Returns order flow intelligence for a symbol.

        Returns:
            {
                "bias": float,          # -1 to +1 (positive = institutional buying)
                "absorption": bool,     # big players absorbing at this level
                "exhaustion": bool,     # climax volume — reversal likely
                "vwap": float,          # volume-weighted average price
                "poc": float,           # point of control (highest volume price)
                "delta": float,         # cumulative delta (buy_vol - sell_vol)
                "high_vol_zones": [{"price": float, "volume": int}, ...],
            }
            or None if insufficient data.
        """
        # Check cache
        cached = self._cache.get(symbol)
        if cached and (time.time() - cached["ts"]) < self.CACHE_TTL:
            return cached["result"]

        result = self._compute(symbol)
        if result is not None:
            self._cache[symbol] = {"result": result, "ts": time.time()}
        return result

    # ═══════════════════════════════════════════════════════════════
    # CORE COMPUTATION
    # ═══════════════════════════════════════════════════════════════

    def _compute(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Compute all order flow metrics from M5 + H1 candles."""
        m5 = self.state.get_candles(symbol, 5)
        h1 = self.state.get_candles(symbol, 60)

        if m5 is None or len(m5) < self.POC_LOOKBACK:
            return None

        # Extract arrays from M5
        o5 = m5["open"].values.astype(np.float64)
        h5 = m5["high"].values.astype(np.float64)
        l5 = m5["low"].values.astype(np.float64)
        c5 = m5["close"].values.astype(np.float64)
        v5 = m5["tick_volume"].values.astype(np.float64)

        # H1 arrays (may be shorter — used for multi-TF confirmation)
        if h1 is not None and len(h1) >= 20:
            oh = h1["open"].values.astype(np.float64)
            ch = h1["close"].values.astype(np.float64)
            vh = h1["tick_volume"].values.astype(np.float64)
        else:
            oh = ch = vh = None

        # ── Volume Imbalance ──
        imbalance = self._volume_imbalance(o5, c5, v5)

        # ── Absorption Detection ──
        absorption = self._detect_absorption(o5, h5, l5, c5, v5)

        # ── Exhaustion Detection ──
        exhaustion = self._detect_exhaustion(h5, l5, c5, v5)

        # ── Delta (cumulative buy vol - sell vol) ──
        delta, delta_divergence = self._compute_delta(o5, c5, v5)

        # ── VWAP ──
        vwap = self._compute_vwap(h5, l5, c5, v5)

        # ── Point of Control ──
        poc = self._compute_poc(h5, l5, c5, v5)

        # ── High Volume Zones ──
        high_vol_zones = self._compute_high_vol_zones(h5, l5, c5, v5)

        # ── Multi-TF Confirmation (H1) ──
        h1_imbalance = 0.0
        if oh is not None:
            h1_imbalance = self._volume_imbalance(oh, ch, vh)

        # ── Composite Bias ──
        bias = self._composite_bias(
            imbalance, absorption, exhaustion, delta,
            delta_divergence, h1_imbalance, c5, vwap,
        )

        return {
            "bias": round(float(np.clip(bias, -1.0, 1.0)), 4),
            "absorption": bool(absorption),
            "exhaustion": bool(exhaustion),
            "vwap": round(float(vwap), 6),
            "poc": round(float(poc), 6),
            "delta": round(float(delta), 2),
            "high_vol_zones": high_vol_zones,
        }

    # ═══════════════════════════════════════════════════════════════
    # VOLUME IMBALANCE
    # ═══════════════════════════════════════════════════════════════

    def _volume_imbalance(self, open_: np.ndarray, close: np.ndarray,
                          volume: np.ndarray) -> float:
        """
        Compare buy volume vs sell volume over recent bars.
        Bullish bar (close > open) volume = buy volume.
        Bearish bar (close < open) volume = sell volume.
        Returns ratio in [-1, +1].
        """
        n = min(self.IMBALANCE_LOOKBACK, len(close))
        o = open_[-n:]
        c = close[-n:]
        v = volume[-n:]

        bull_mask = c > o
        bear_mask = c < o

        buy_vol = np.sum(v[bull_mask])
        sell_vol = np.sum(v[bear_mask])
        total = buy_vol + sell_vol

        if total == 0:
            return 0.0

        # Normalized: +1 = all buying, -1 = all selling
        return float((buy_vol - sell_vol) / total)

    # ═══════════════════════════════════════════════════════════════
    # ABSORPTION DETECTION
    # ═══════════════════════════════════════════════════════════════

    def _detect_absorption(self, open_: np.ndarray, high: np.ndarray,
                           low: np.ndarray, close: np.ndarray,
                           volume: np.ndarray) -> bool:
        """
        Absorption: high volume but no price movement.
        Big players absorbing aggressive orders without letting price move.
        Look at last 3 bars — if any has high volume + tiny body, absorption = True.
        """
        n = len(close)
        if n < 20:
            return False

        body = np.abs(close - open_)
        candle_range = high - low

        # Use percentiles from last 50 bars for context
        lookback = min(50, n)
        vol_thresh = np.percentile(volume[-lookback:], self.ABSORPTION_VOL_PCTL)
        # Body threshold relative to range — absorption = body < 20% of range
        range_recent = candle_range[-lookback:]
        body_thresh = np.percentile(range_recent, self.ABSORPTION_BODY_PCTL)

        # Check last 3 bars
        for i in range(-3, 0):
            if (volume[i] >= vol_thresh and
                    body[i] <= body_thresh and
                    candle_range[i] > 0):
                # Confirm: body is less than 30% of the candle range
                body_ratio = body[i] / candle_range[i]
                if body_ratio < 0.30:
                    return True

        return False

    # ═══════════════════════════════════════════════════════════════
    # EXHAUSTION DETECTION
    # ═══════════════════════════════════════════════════════════════

    def _detect_exhaustion(self, high: np.ndarray, low: np.ndarray,
                           close: np.ndarray, volume: np.ndarray) -> bool:
        """
        Exhaustion / climax volume: extreme volume at price extremes.
        Highest volume bar in last 20 bars coincides with highest high or lowest low.
        """
        n = len(close)
        lookback = min(20, n)

        v_window = volume[-lookback:]
        h_window = high[-lookback:]
        l_window = low[-lookback:]

        vol_thresh = np.percentile(v_window, self.EXHAUSTION_VOL_PCTL)

        # Check last 3 bars for climax
        for i in range(-3, 0):
            if volume[i] < vol_thresh:
                continue

            # Price at extreme of recent range?
            bar_idx = lookback + i  # index into window
            if bar_idx < 0:
                continue

            at_high = h_window[bar_idx] >= np.max(h_window) * 0.999
            at_low = l_window[bar_idx] <= np.min(l_window) * 1.001

            if at_high or at_low:
                return True

        return False

    # ═══════════════════════════════════════════════════════════════
    # DELTA (Cumulative Buy - Sell Volume)
    # ═══════════════════════════════════════════════════════════════

    def _compute_delta(self, open_: np.ndarray, close: np.ndarray,
                       volume: np.ndarray):
        """
        Cumulative delta over recent bars.
        Also detect delta divergence: price trending one way, delta trending opposite.

        Returns:
            (delta, divergence_flag)
            delta: float cumulative buy - sell volume
            divergence_flag: +1 if bullish divergence (price down, delta up),
                             -1 if bearish divergence (price up, delta down),
                              0 if no divergence
        """
        n = min(self.DELTA_LOOKBACK, len(close))
        o = open_[-n:]
        c = close[-n:]
        v = volume[-n:]

        # Per-bar delta: bullish bar = +volume, bearish = -volume
        # Doji (open == close): split proportionally using high/low not available,
        # so assign zero
        bar_delta = np.where(c > o, v, np.where(c < o, -v, 0.0))
        cum_delta = float(np.sum(bar_delta))

        # Divergence detection: compare price trend vs delta trend
        # Price trend: linear slope of close
        divergence = 0
        if n >= 5:
            x = np.arange(n, dtype=np.float64)
            # Price slope
            price_slope = self._linreg_slope(x, c)
            # Delta slope (cumulative delta over time)
            cum_delta_arr = np.cumsum(bar_delta)
            delta_slope = self._linreg_slope(x, cum_delta_arr)

            # Divergence: price and delta moving in opposite directions
            if price_slope > 0 and delta_slope < 0:
                # Price up, smart money selling — bearish divergence
                divergence = -1
            elif price_slope < 0 and delta_slope > 0:
                # Price down, smart money buying — bullish divergence
                divergence = 1

        return cum_delta, divergence

    # ═══════════════════════════════════════════════════════════════
    # VWAP
    # ═══════════════════════════════════════════════════════════════

    def _compute_vwap(self, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, volume: np.ndarray) -> float:
        """Volume-weighted average price from last VWAP_LOOKBACK bars."""
        n = min(self.VWAP_LOOKBACK, len(close))
        h = high[-n:]
        l = low[-n:]
        c = close[-n:]
        v = volume[-n:]

        typical = (h + l + c) / 3.0
        total_vol = np.sum(v)

        if total_vol == 0:
            return float(c[-1])

        return float(np.sum(typical * v) / total_vol)

    # ═══════════════════════════════════════════════════════════════
    # POINT OF CONTROL
    # ═══════════════════════════════════════════════════════════════

    def _compute_poc(self, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, volume: np.ndarray) -> float:
        """
        Point of control: price level with highest volume in last POC_LOOKBACK bars.
        Uses volume profile binning.
        """
        n = min(self.POC_LOOKBACK, len(close))
        h = high[-n:]
        l = low[-n:]
        c = close[-n:]
        v = volume[-n:]

        price_low = float(np.min(l))
        price_high = float(np.max(h))

        if price_high <= price_low:
            return float(c[-1])

        # Build volume profile: distribute each bar's volume across its range
        bins = self.VOL_PROFILE_BINS
        bin_edges = np.linspace(price_low, price_high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        vol_profile = np.zeros(bins, dtype=np.float64)

        for i in range(n):
            bar_low = l[i]
            bar_high = h[i]
            bar_vol = v[i]

            if bar_high <= bar_low or bar_vol == 0:
                # Point bar — assign all volume to closest bin
                idx = int((c[i] - price_low) / (price_high - price_low) * (bins - 1))
                idx = max(0, min(bins - 1, idx))
                vol_profile[idx] += bar_vol
                continue

            # Find bins this bar spans
            low_bin = np.searchsorted(bin_edges, bar_low, side="right") - 1
            high_bin = np.searchsorted(bin_edges, bar_high, side="left")
            low_bin = max(0, low_bin)
            high_bin = min(bins - 1, high_bin)

            # Distribute volume proportionally across bins
            span = high_bin - low_bin + 1
            if span > 0:
                vol_per_bin = bar_vol / span
                vol_profile[low_bin:high_bin + 1] += vol_per_bin

        # POC = bin center with highest volume
        poc_idx = int(np.argmax(vol_profile))
        return float(bin_centers[poc_idx])

    # ═══════════════════════════════════════════════════════════════
    # HIGH VOLUME ZONES
    # ═══════════════════════════════════════════════════════════════

    def _compute_high_vol_zones(self, high: np.ndarray, low: np.ndarray,
                                close: np.ndarray,
                                volume: np.ndarray) -> List[Dict[str, Any]]:
        """
        High volume nodes: peaks in volume profile.
        Returns top N zones sorted by volume descending.
        """
        n = min(self.POC_LOOKBACK, len(close))
        h = high[-n:]
        l = low[-n:]
        c = close[-n:]
        v = volume[-n:]

        price_low = float(np.min(l))
        price_high = float(np.max(h))

        if price_high <= price_low:
            return [{"price": round(float(c[-1]), 6), "volume": int(np.sum(v))}]

        bins = self.VOL_PROFILE_BINS
        bin_edges = np.linspace(price_low, price_high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        vol_profile = np.zeros(bins, dtype=np.float64)

        for i in range(n):
            bar_low = l[i]
            bar_high = h[i]
            bar_vol = v[i]

            if bar_high <= bar_low or bar_vol == 0:
                idx = int((c[i] - price_low) / (price_high - price_low) * (bins - 1))
                idx = max(0, min(bins - 1, idx))
                vol_profile[idx] += bar_vol
                continue

            low_bin = np.searchsorted(bin_edges, bar_low, side="right") - 1
            high_bin = np.searchsorted(bin_edges, bar_high, side="left")
            low_bin = max(0, low_bin)
            high_bin = min(bins - 1, high_bin)

            span = high_bin - low_bin + 1
            if span > 0:
                vol_per_bin = bar_vol / span
                vol_profile[low_bin:high_bin + 1] += vol_per_bin

        # Find local peaks in volume profile
        # A peak is a bin with higher volume than its neighbors
        peaks = []
        for i in range(1, bins - 1):
            if vol_profile[i] > vol_profile[i - 1] and vol_profile[i] > vol_profile[i + 1]:
                peaks.append((float(bin_centers[i]), float(vol_profile[i])))

        # If no peaks found, use top bins by volume
        if not peaks:
            top_idx = np.argsort(vol_profile)[::-1][:self.HIGH_VOL_TOP_N]
            peaks = [(float(bin_centers[i]), float(vol_profile[i])) for i in top_idx
                     if vol_profile[i] > 0]

        # Sort by volume descending, take top N
        peaks.sort(key=lambda x: x[1], reverse=True)
        peaks = peaks[:self.HIGH_VOL_TOP_N]

        return [{"price": round(p, 6), "volume": int(v)} for p, v in peaks]

    # ═══════════════════════════════════════════════════════════════
    # COMPOSITE BIAS
    # ═══════════════════════════════════════════════════════════════

    def _composite_bias(self, imbalance: float, absorption: bool,
                        exhaustion: bool, delta: float,
                        delta_divergence: int, h1_imbalance: float,
                        close: np.ndarray, vwap: float) -> float:
        """
        Combine all signals into a single bias score [-1, +1].

        Weights:
          - Volume imbalance (M5): 0.25
          - Delta direction:       0.25
          - H1 imbalance:          0.20
          - Price vs VWAP:         0.15
          - Delta divergence:      0.15

        Modifiers:
          - Absorption dampens bias toward 0 (big player absorbing = range)
          - Exhaustion pushes bias toward reversal
        """
        # Normalize delta to [-1, 1] using tanh
        delta_norm = float(np.tanh(delta / max(1.0, abs(delta) * 0.5))) if delta != 0 else 0.0

        # Price vs VWAP: above = bullish, below = bearish
        price = float(close[-1])
        if vwap > 0:
            price_vs_vwap = np.clip((price - vwap) / (vwap * 0.001 + 1e-10), -1.0, 1.0)
        else:
            price_vs_vwap = 0.0

        # Weighted sum
        bias = (
            0.25 * imbalance +
            0.25 * delta_norm +
            0.20 * h1_imbalance +
            0.15 * float(price_vs_vwap) +
            0.15 * float(delta_divergence)
        )

        # Absorption modifier: dampen bias (big player holding the level)
        if absorption:
            bias *= 0.4

        # Exhaustion modifier: push toward reversal
        if exhaustion:
            bias *= -0.5

        return float(np.clip(bias, -1.0, 1.0))

    # ═══════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _linreg_slope(x: np.ndarray, y: np.ndarray) -> float:
        """Simple linear regression slope using numpy."""
        n = len(x)
        if n < 2:
            return 0.0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return 0.0
        return float(np.sum((x - x_mean) * (y - y_mean)) / denom)
