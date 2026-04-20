"""
Dragon Trader — Fair Value Gap (FVG) Detector.
Detects institutional imbalance zones where price moved so fast it left a gap
between candle wicks. Price tends to return and fill these gaps.

FVG Types:
  Bullish FVG: Bar[i-2] high < Bar[i] low (gap up — buyers overwhelmed sellers)
  Bearish FVG: Bar[i-2] low > Bar[i] high (gap down — sellers overwhelmed buyers)

Usage:
  - Entry: Enter LONG when price pulls back INTO a bullish FVG (buy the discount)
  - Entry: Enter SHORT when price rallies INTO a bearish FVG (sell the premium)
  - SL: Place behind the FVG (if gap fills completely, thesis is broken)
  - TP: Opposite FVG or next liquidity zone

Timeframes: H1 (primary), M15 (confirmation), M5 (precision entry)
"""
import time
import logging
import numpy as np
from typing import Dict, List, Optional

log = logging.getLogger("dragon.fvg")

# FVG must be at least this % of ATR to be significant
MIN_FVG_SIZE_ATR = 0.3
# FVGs older than this many bars are expired
MAX_FVG_AGE_BARS = 50
# Cache TTL
CACHE_TTL_S = 2.0


class FVGDetector:
    """Detects and tracks Fair Value Gaps across multiple timeframes."""

    def __init__(self, state):
        self.state = state
        self._cache: Dict[str, dict] = {}  # symbol -> {result, ts}

    def get_fvg_signal(self, symbol: str, direction: str, current_price: float) -> dict:
        """
        Get FVG-based intelligence for entry/exit decisions.

        Returns:
            {
                "has_entry_fvg": bool,      # price is inside a FVG that supports our direction
                "entry_fvg_strength": float, # 0-1, how significant the FVG is
                "nearest_bull_fvg": dict,    # nearest unfilled bullish FVG below
                "nearest_bear_fvg": dict,    # nearest unfilled bearish FVG above
                "fvg_sl_price": float,       # suggested SL based on FVG boundary
                "fvg_tp_price": float,       # suggested TP at opposite FVG
                "fvg_bias": float,           # -1 to +1 based on FVG balance
                "active_fvgs": list,         # all unfilled FVGs nearby
                "mtf_fvg_confluence": bool,  # H1 + M15 FVGs align
            }
        """
        cached = self._cache.get(symbol)
        if cached and (time.time() - cached["ts"]) < CACHE_TTL_S:
            return cached["result"]

        result = self._compute(symbol, direction, current_price)
        self._cache[symbol] = {"result": result, "ts": time.time()}
        return result

    def _compute(self, symbol: str, direction: str, price: float) -> dict:
        default = {
            "has_entry_fvg": False, "entry_fvg_strength": 0.0,
            "nearest_bull_fvg": None, "nearest_bear_fvg": None,
            "fvg_sl_price": 0.0, "fvg_tp_price": 0.0,
            "fvg_bias": 0.0, "active_fvgs": [],
            "mtf_fvg_confluence": False,
        }

        # Detect FVGs on H1
        h1_fvgs = self._detect_fvgs(symbol, 60)
        # Detect FVGs on M15 for confluence
        m15_fvgs = self._detect_fvgs(symbol, 15)

        if not h1_fvgs:
            return default

        # Filter active (unfilled) FVGs
        active = [f for f in h1_fvgs if not self._is_filled(f, symbol, 60)]
        if not active:
            return default

        # Compute ATR for normalization
        atr = self._get_atr(symbol)
        if atr <= 0:
            return default

        # Find nearest bull/bear FVGs relative to current price
        bull_fvgs = [f for f in active if f["type"] == "bullish"]
        bear_fvgs = [f for f in active if f["type"] == "bearish"]

        nearest_bull = None
        nearest_bear = None
        if bull_fvgs:
            # Nearest bullish FVG below price (support zone)
            below = [f for f in bull_fvgs if f["top"] <= price]
            if below:
                nearest_bull = min(below, key=lambda f: price - f["top"])
            else:
                nearest_bull = min(bull_fvgs, key=lambda f: abs(f["mid"] - price))

        if bear_fvgs:
            # Nearest bearish FVG above price (resistance zone)
            above = [f for f in bear_fvgs if f["bottom"] >= price]
            if above:
                nearest_bear = min(above, key=lambda f: f["bottom"] - price)
            else:
                nearest_bear = min(bear_fvgs, key=lambda f: abs(f["mid"] - price))

        # Check if price is currently INSIDE an FVG (ideal entry zone)
        has_entry_fvg = False
        entry_strength = 0.0
        for f in active:
            if f["bottom"] <= price <= f["top"]:
                # Price is inside this FVG
                if (direction == "LONG" and f["type"] == "bullish") or \
                   (direction == "SHORT" and f["type"] == "bearish"):
                    has_entry_fvg = True
                    entry_strength = min(1.0, f["size"] / atr)
                    break

        # SL suggestion: behind the FVG that supports entry
        fvg_sl = 0.0
        if direction == "LONG" and nearest_bull:
            fvg_sl = nearest_bull["bottom"] - atr * 0.2  # below FVG with buffer
        elif direction == "SHORT" and nearest_bear:
            fvg_sl = nearest_bear["top"] + atr * 0.2  # above FVG with buffer

        # TP suggestion: next opposing FVG (price will seek to fill it)
        fvg_tp = 0.0
        if direction == "LONG" and nearest_bear:
            fvg_tp = nearest_bear["mid"]  # target: fill the bearish gap above
        elif direction == "SHORT" and nearest_bull:
            fvg_tp = nearest_bull["mid"]  # target: fill the bullish gap below

        # FVG bias: more unfilled bull FVGs below = bullish bias (buy the dips)
        bull_count = len(bull_fvgs)
        bear_count = len(bear_fvgs)
        total = bull_count + bear_count
        if total > 0:
            fvg_bias = (bull_count - bear_count) / total  # -1 to +1
        else:
            fvg_bias = 0.0

        # MTF confluence: check if M15 has a supporting FVG in same direction
        mtf_confluence = False
        if m15_fvgs:
            m15_active = [f for f in m15_fvgs if not self._is_filled(f, symbol, 15)]
            for f in m15_active:
                if f["bottom"] <= price <= f["top"]:
                    if (direction == "LONG" and f["type"] == "bullish") or \
                       (direction == "SHORT" and f["type"] == "bearish"):
                        mtf_confluence = True
                        break

        # Build active list for dashboard
        active_list = []
        for f in active[:10]:  # top 10 nearest
            active_list.append({
                "type": f["type"],
                "top": round(f["top"], 5),
                "bottom": round(f["bottom"], 5),
                "mid": round(f["mid"], 5),
                "size_atr": round(f["size"] / atr, 2) if atr > 0 else 0,
                "age_bars": f["age"],
            })

        return {
            "has_entry_fvg": has_entry_fvg,
            "entry_fvg_strength": round(entry_strength, 2),
            "nearest_bull_fvg": self._fvg_to_dict(nearest_bull) if nearest_bull else None,
            "nearest_bear_fvg": self._fvg_to_dict(nearest_bear) if nearest_bear else None,
            "fvg_sl_price": round(fvg_sl, 5),
            "fvg_tp_price": round(fvg_tp, 5),
            "fvg_bias": round(fvg_bias, 2),
            "active_fvgs": active_list,
            "mtf_fvg_confluence": mtf_confluence,
        }

    def _detect_fvgs(self, symbol: str, tf_minutes: int) -> list:
        """Detect Fair Value Gaps on a given timeframe."""
        df = self.state.get_candles(symbol, tf_minutes)
        if df is None or len(df) < 10:
            return []

        try:
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            c = df["close"].values.astype(np.float64)
            n = len(h)

            # ATR for minimum gap size filter
            tr = np.maximum(h[1:] - l[1:],
                            np.maximum(np.abs(h[1:] - c[:-1]),
                                       np.abs(l[1:] - c[:-1])))
            atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
            min_gap = atr * MIN_FVG_SIZE_ATR

            fvgs = []
            # Start from bar 2 (need bars i-2, i-1, i)
            for i in range(2, n):
                age = n - 1 - i
                if age > MAX_FVG_AGE_BARS:
                    continue

                # Bullish FVG: bar[i-2] high < bar[i] low
                # The gap between bar[i-2] candle top and bar[i] candle bottom
                # Bar[i-1] is the impulse candle that created the gap
                if l[i] > h[i-2]:
                    gap_size = l[i] - h[i-2]
                    if gap_size >= min_gap:
                        fvgs.append({
                            "type": "bullish",
                            "top": float(l[i]),       # upper boundary = bar[i] low
                            "bottom": float(h[i-2]),   # lower boundary = bar[i-2] high
                            "mid": float((l[i] + h[i-2]) / 2),
                            "size": float(gap_size),
                            "age": age,
                            "bar_idx": i,
                        })

                # Bearish FVG: bar[i-2] low > bar[i] high
                if h[i] < l[i-2]:
                    gap_size = l[i-2] - h[i]
                    if gap_size >= min_gap:
                        fvgs.append({
                            "type": "bearish",
                            "top": float(l[i-2]),      # upper boundary = bar[i-2] low
                            "bottom": float(h[i]),      # lower boundary = bar[i] high
                            "mid": float((l[i-2] + h[i]) / 2),
                            "size": float(gap_size),
                            "age": age,
                            "bar_idx": i,
                        })

            # Sort by distance to current price (most recent first)
            fvgs.sort(key=lambda f: f["age"])
            return fvgs

        except Exception as e:
            log.debug("[%s] FVG detection error: %s", symbol, e)
            return []

    def _is_filled(self, fvg: dict, symbol: str, tf_minutes: int) -> bool:
        """Check if price has filled (touched both sides of) the FVG since it was created."""
        df = self.state.get_candles(symbol, tf_minutes)
        if df is None:
            return False

        try:
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            bar_idx = fvg["bar_idx"]
            n = len(h)

            # Check bars after the FVG was created
            for j in range(bar_idx + 1, n):
                if fvg["type"] == "bullish":
                    # Bullish FVG is filled when price dips below the gap top (into the gap)
                    # AND then goes below the bottom (completely filled)
                    if l[j] <= fvg["bottom"]:
                        return True
                else:
                    # Bearish FVG filled when price goes above the gap bottom into the gap
                    # AND above the top (completely filled)
                    if h[j] >= fvg["top"]:
                        return True

            return False
        except Exception:
            return False

    def _get_atr(self, symbol: str) -> float:
        """Get H1 ATR."""
        df = self.state.get_candles(symbol, 60)
        if df is None or len(df) < 15:
            return 0.0
        try:
            h = df["high"].values.astype(np.float64)
            l = df["low"].values.astype(np.float64)
            c = df["close"].values.astype(np.float64)
            tr = np.maximum(h[1:] - l[1:],
                            np.maximum(np.abs(h[1:] - c[:-1]),
                                       np.abs(l[1:] - c[:-1])))
            return float(np.mean(tr[-14:]))
        except Exception:
            return 0.0

    @staticmethod
    def _fvg_to_dict(fvg: dict) -> dict:
        return {
            "type": fvg["type"],
            "top": round(fvg["top"], 5),
            "bottom": round(fvg["bottom"], 5),
            "mid": round(fvg["mid"], 5),
            "age": fvg["age"],
        }
