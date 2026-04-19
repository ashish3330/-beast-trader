"""
Dragon Trader — Smart Entry Intelligence.
Three institutional-grade entry filters that run AFTER MasterBrain approves:

1. M5 Pullback Confirmation — Don't chase, buy the dip
2. USD Strength Index — Don't fight the dollar
3. Volume Confirmation — Don't enter on thin air

Each returns a multiplier (0.0 = block, 0.5 = reduce risk, 1.0 = normal, 1.2 = boost).
"""
import time
import logging
import numpy as np
from datetime import datetime, timezone

log = logging.getLogger("dragon.smart_entry")


class SmartEntry:
    """Institutional-grade entry intelligence layer."""

    def __init__(self, state):
        self.state = state
        self._pullback_state = {}   # symbol -> {waiting_since, best_pullback_depth}
        self._usd_cache = {"strength": 0.0, "ts": 0}
        self._vol_cache = {}        # symbol -> {avg_vol, ts}

    def evaluate(self, symbol, direction, atr_val, category="Forex"):
        """
        Run all smart entry checks. Returns dict:
          approved: bool
          risk_mult: float (0.5-1.3 multiplier on risk)
          reason: str
          details: dict
        """
        checks = {}

        # 1. M5 Pullback (forex/gold only — trend-followers like BTC/indices don't benefit)
        if category in ("Forex", "Gold"):
            pb = self._check_pullback(symbol, direction, atr_val)
        else:
            pb = {"mult": 1.0, "state": "skip_category"}
        checks["pullback"] = pb

        # 2. USD Strength (forex/gold only)
        usd = self._check_usd_strength(symbol, direction, category)
        checks["usd"] = usd

        # 3. Volume confirmation (all categories)
        vol = self._check_volume(symbol, direction)
        checks["volume"] = vol

        # Combine: average the multipliers, but hard-block if any is 0
        mults = [pb["mult"], usd["mult"], vol["mult"]]
        if any(m == 0 for m in mults):
            blocker = [k for k, v in checks.items() if v["mult"] == 0][0]
            return {
                "approved": False,
                "risk_mult": 0.0,
                "reason": f"SmartEntry_{blocker}",
                "details": checks,
            }

        # Weighted combination (pullback matters most)
        combined = pb["mult"] * 0.5 + usd["mult"] * 0.25 + vol["mult"] * 0.25
        combined = max(0.5, min(1.3, combined))

        return {
            "approved": True,
            "risk_mult": round(combined, 2),
            "reason": "pass",
            "details": checks,
        }

    # ═══════════════════════════════════════════════════════════════
    #  1. M5 PULLBACK CONFIRMATION
    # ═══════════════════════════════════════════════════════════════

    def _check_pullback(self, symbol, direction, atr_val):
        """
        Institutional entry: don't chase extended moves.
        Check if M5 shows a pullback-and-recovery pattern.

        States:
        - EXTENDED: M5 price far from M5 EMA → wait for pullback
        - PULLBACK: M5 pulled back to EMA zone → watch for recovery
        - RECOVERY: M5 bouncing off EMA in trade direction → ENTER NOW (best entry)
        - NEUTRAL: M5 near EMA, no clear pattern → enter normally
        """
        try:
            m5_df = self.state.get_candles(symbol, 5)
            if m5_df is None or len(m5_df) < 30:
                return {"mult": 1.0, "state": "no_data"}

            c = m5_df["close"].values.astype(np.float64)
            h = m5_df["high"].values.astype(np.float64)
            l = m5_df["low"].values.astype(np.float64)
            n = len(c)

            # M5 EMA(8) and EMA(21)
            ema8 = self._ema(c, 8)
            ema21 = self._ema(c, 21)

            if np.isnan(ema8[-1]) or np.isnan(ema21[-1]):
                return {"mult": 1.0, "state": "ema_nan"}

            price = c[-1]
            ema_mid = (ema8[-1] + ema21[-1]) / 2
            atr = max(atr_val if atr_val > 0 else abs(price * 0.001), 1e-10)

            # Distance from EMA zone (in ATR units)
            dist = (price - ema_mid) / atr
            if direction == "SHORT":
                dist = -dist  # positive = extended in trade direction

            # Check last 6 M5 bars for pullback pattern
            recent_lows = l[-6:]
            recent_highs = h[-6:]
            recent_close = c[-6:]

            if direction == "LONG":
                # Pullback = price dipped below EMA then recovering
                dipped = any(recent_lows[i] < ema21[-6+i] for i in range(len(recent_lows)) if not np.isnan(ema21[-6+i]))
                recovering = c[-1] > ema8[-1] and c[-1] > c[-2]
                extended = dist > 1.5  # price > 1.5 ATR above EMA
            else:
                dipped = any(recent_highs[i] > ema21[-6+i] for i in range(len(recent_highs)) if not np.isnan(ema21[-6+i]))
                recovering = c[-1] < ema8[-1] and c[-1] < c[-2]
                extended = dist > 1.5

            # Consecutive bars in trade direction (momentum confirmation)
            consec = 0
            for i in range(n-1, max(n-7, 0), -1):
                if (direction == "LONG" and c[i] > c[i-1]) or \
                   (direction == "SHORT" and c[i] < c[i-1]):
                    consec += 1
                else:
                    break

            if extended and not dipped:
                # Price is extended, no pullback yet — risky entry
                if dist > 2.0:
                    # Severely extended — BLOCK entirely, don't chase
                    return {"mult": 0.0, "state": "severely_extended", "dist": round(dist, 2)}
                return {"mult": 0.7, "state": "extended", "dist": round(dist, 2)}

            if dipped and recovering:
                # IDEAL: pulled back and now recovering in our direction
                return {"mult": 1.3, "state": "recovery", "dist": round(dist, 2)}

            if dipped and not recovering:
                # Pulled back but not yet recovering — wait
                return {"mult": 0.8, "state": "pullback_wait", "dist": round(dist, 2)}

            if abs(dist) < 0.5:
                # Near EMA — neutral, fine to enter
                return {"mult": 1.0, "state": "neutral", "dist": round(dist, 2)}

            if consec >= 4:
                # 4+ consecutive bars in direction — might be overextended
                return {"mult": 0.8, "state": "momentum_stretch", "consec": consec}

            return {"mult": 1.0, "state": "ok", "dist": round(dist, 2)}

        except Exception as e:
            log.debug("Pullback check error %s: %s", symbol, e)
            return {"mult": 1.0, "state": "error"}

    # ═══════════════════════════════════════════════════════════════
    #  2. USD STRENGTH INDEX
    # ═══════════════════════════════════════════════════════════════

    def _check_usd_strength(self, symbol, direction, category):
        """
        Compute real-time USD strength from available USD pairs.
        Don't go long gold when USD is strengthening.
        Don't go long USDJPY when USD is weakening.

        USD strength = average H1 return of USD-long pairs minus USD-short pairs.
        """
        try:
            # Only relevant for forex and gold
            if category not in ("Forex", "Gold"):
                return {"mult": 1.0, "usd": 0.0, "reason": "not_forex"}

            # Cache for 60s
            now = time.time()
            if now - self._usd_cache["ts"] < 60:
                usd_str = self._usd_cache["strength"]
            else:
                usd_str = self._compute_usd_strength()
                self._usd_cache = {"strength": usd_str, "ts": now}

            # How does USD strength affect this symbol?
            # USD-long pairs: USDJPY, USDCAD, USDCHF (USD is base)
            # USD-short pairs: XAUUSD, XAGUSD, EURUSD, GBPUSD (USD is quote)
            # Inverse: EURJPY, EURAUD (no direct USD)

            usd_long_syms = {"USDJPY", "USDCAD", "USDCHF"}
            usd_short_syms = {"XAUUSD", "XAGUSD", "EURUSD", "GBPUSD"}
            no_usd = {"EURJPY", "EURAUD", "BTCUSD", "NAS100.r", "JPN225ft"}

            if symbol in no_usd:
                return {"mult": 1.0, "usd": round(usd_str, 3), "reason": "no_usd_pair"}

            # For USD-long pairs: strong USD = LONG is good
            # For USD-short pairs: strong USD = LONG is bad (gold falls when USD rises)
            if symbol in usd_long_syms:
                alignment = usd_str if direction == "LONG" else -usd_str
            elif symbol in usd_short_syms:
                alignment = -usd_str if direction == "LONG" else usd_str
            else:
                return {"mult": 1.0, "usd": round(usd_str, 3), "reason": "unknown_pair"}

            # alignment > 0 = USD supports our trade, < 0 = USD opposes
            if alignment < -0.6:
                # Strong USD opposition — BLOCK gold/silver, heavy reduce forex
                if category == "Gold":
                    return {"mult": 0.0, "usd": round(usd_str, 3), "align": round(alignment, 3),
                            "reason": "USD_BLOCK_GOLD"}
                return {"mult": 0.4, "usd": round(usd_str, 3), "align": round(alignment, 3)}
            elif alignment < -0.3:
                # USD opposing — reduce risk (heavier for gold)
                m = 0.5 if category == "Gold" else 0.7
                return {"mult": m, "usd": round(usd_str, 3), "align": round(alignment, 3)}
            elif alignment > 0.5:
                # Strong USD support — boost
                return {"mult": 1.3, "usd": round(usd_str, 3), "align": round(alignment, 3)}
            elif alignment > 0.3:
                # Moderate USD support
                return {"mult": 1.2, "usd": round(usd_str, 3), "align": round(alignment, 3)}
            else:
                return {"mult": 1.0, "usd": round(usd_str, 3), "align": round(alignment, 3)}

        except Exception as e:
            log.debug("USD strength error: %s", e)
            return {"mult": 1.0, "usd": 0.0, "reason": "error"}

    def _compute_usd_strength(self):
        """
        Compute USD strength from H1 candles of available pairs.
        Returns float: positive = USD strengthening, negative = USD weakening.
        Normalized to roughly -1.0 to +1.0 range.
        """
        usd_returns = []

        # USD is base currency (USDJPY, USDCAD, USDCHF) — price up = USD strong
        for sym in ["USDJPY", "USDCAD", "USDCHF"]:
            ret = self._get_h1_return(sym, bars=5)
            if ret is not None:
                usd_returns.append(ret)

        # USD is quote currency (XAUUSD, EURUSD) — price up = USD weak
        for sym in ["XAUUSD", "XAGUSD"]:
            ret = self._get_h1_return(sym, bars=5)
            if ret is not None:
                usd_returns.append(-ret)  # invert

        if not usd_returns:
            return 0.0

        # Average and normalize (typical H1 move is 0.1-0.5%)
        avg_ret = np.mean(usd_returns)
        # Scale to -1..+1 range (0.3% move = strength 1.0)
        return float(np.clip(avg_ret / 0.003, -1.0, 1.0))

    def _get_h1_return(self, symbol, bars=5):
        """Get H1 return over last N bars. Returns percentage or None."""
        try:
            h1 = self.state.get_candles(symbol, 60)
            if h1 is None or len(h1) < bars + 1:
                return None
            c = h1["close"].values.astype(np.float64)
            if c[-bars-1] == 0:
                return None
            return (c[-1] - c[-bars-1]) / c[-bars-1]
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════
    #  3. VOLUME CONFIRMATION
    # ═══════════════════════════════════════════════════════════════

    def _check_volume(self, symbol, direction):
        """
        Check if volume supports the entry.
        - Signal bar volume > 1.2x average → strong confirmation
        - Signal bar volume < 0.6x average → weak, reduce risk
        - Volume trend (last 5 bars increasing in direction) → boost
        """
        try:
            h1 = self.state.get_candles(symbol, 60)
            if h1 is None or len(h1) < 30:
                return {"mult": 1.0, "reason": "no_data"}

            vol = h1["tick_volume"].values.astype(np.float64) if "tick_volume" in h1.columns else None
            if vol is None or len(vol) < 30:
                # Try 'volume' column
                if "volume" in h1.columns:
                    vol = h1["volume"].values.astype(np.float64)
                else:
                    return {"mult": 1.0, "reason": "no_volume_col"}

            c = h1["close"].values.astype(np.float64)

            # Average volume (last 20 bars)
            avg_vol = np.mean(vol[-21:-1])  # exclude current forming bar
            if avg_vol <= 0:
                return {"mult": 1.0, "reason": "zero_avg_vol"}

            # Signal bar volume (last completed bar)
            signal_vol = vol[-2]  # completed bar, not forming
            vol_ratio = signal_vol / avg_vol

            # Directional volume: was the signal bar a bullish or bearish volume bar?
            bar_bullish = c[-2] > c[-3]  # completed bar closed higher than previous
            vol_aligned = (direction == "LONG" and bar_bullish) or \
                          (direction == "SHORT" and not bar_bullish)

            # Volume trend (are last 3 bars showing increasing volume?)
            vol_trend = 0
            if len(vol) >= 5:
                recent_vol = vol[-4:-1]  # last 3 completed bars
                if recent_vol[-1] > recent_vol[-2] > recent_vol[-3]:
                    vol_trend = 1  # increasing
                elif recent_vol[-1] < recent_vol[-2] < recent_vol[-3]:
                    vol_trend = -1  # decreasing

            # Scoring — base multiplier from volume ratio + alignment
            if vol_ratio > 1.5 and vol_aligned:
                base_mult = 1.2
            elif vol_ratio > 1.2 and vol_aligned:
                base_mult = 1.1
            elif vol_ratio < 0.5:
                base_mult = 0.65
            elif vol_ratio < 0.7 and not vol_aligned:
                base_mult = 0.75
            elif vol_ratio > 1.2 and not vol_aligned:
                base_mult = 0.65
            else:
                base_mult = 1.0

            # Volume TREND boost — increasing volume in direction = conviction
            if vol_trend == 1 and vol_aligned:
                base_mult *= 1.1   # rising volume confirming direction
            elif vol_trend == -1:
                base_mult *= 0.9   # fading volume = weakening move
            # Decreasing vol against direction is actually good (selling exhaustion)
            elif vol_trend == -1 and not vol_aligned:
                base_mult *= 1.05

            base_mult = max(0.5, min(1.3, base_mult))
            return {"mult": round(base_mult, 2), "ratio": round(vol_ratio, 2),
                    "aligned": vol_aligned, "trend": vol_trend}

        except Exception as e:
            log.debug("Volume check error %s: %s", symbol, e)
            return {"mult": 1.0, "reason": "error"}

    # ═══════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _ema(arr, period):
        out = np.full_like(arr, np.nan)
        if len(arr) < period:
            return out
        out[period - 1] = np.mean(arr[:period])
        m = 2.0 / (period + 1)
        for i in range(period, len(arr)):
            out[i] = arr[i] * m + out[i - 1] * (1 - m)
        return out
