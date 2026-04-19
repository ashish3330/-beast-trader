"""
Beast Trader — Price Action Pattern Learner.
Detects classical chart patterns on H1 candles, records outcomes,
and provides a pattern-based directional bias for entry decisions.

Patterns detected:
  - Double top / double bottom
  - Head & shoulders / inverse H&S
  - Rising / falling wedge
  - Bull / bear flag
  - Inside bar
  - Pin bar (hammer / shooting star)

All detection uses numpy only. Results cached 60s per symbol.
Outcomes stored in trade_journal.db → pattern_outcomes table.
"""
import time
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS, DB_PATH

log = logging.getLogger("beast.pattern_learner")

JOURNAL_DB = DB_PATH.parent / "trade_journal.db"
JOURNAL_DB.parent.mkdir(parents=True, exist_ok=True)

# Cache expiry
_CACHE_TTL = 60  # seconds


class PatternLearner:
    """Detects price action patterns, learns from outcomes, provides bias."""

    def __init__(self, state):
        self.state = state
        # Cache: symbol -> {"patterns": [...], "ts": float}
        self._cache: Dict[str, dict] = {}
        self._init_db()

    # ─── DB ──────────────────────────────────────────────────────
    def _init_db(self):
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    pnl REAL,
                    r_multiple REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_po_sym_pat
                ON pattern_outcomes (symbol, pattern_type)
            """)
            conn.commit()
            conn.close()
            log.info("Pattern outcomes table ready: %s", JOURNAL_DB)
        except Exception as e:
            log.error("pattern_outcomes init error: %s", e)

    # ─── PUBLIC API ──────────────────────────────────────────────

    def detect_patterns(self, symbol: str) -> List[dict]:
        """Detect all active patterns on H1 candles for *symbol*.

        Returns list of dicts:
            {"pattern": str, "direction": "LONG"|"SHORT", "strength": float 0-1}
        """
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and now - cached["ts"] < _CACHE_TTL:
            return cached["patterns"]

        h1 = self.state.get_candles(symbol, 60)
        if h1 is None or len(h1) < 30:
            return []

        o = h1["open"].values.astype(float)
        h = h1["high"].values.astype(float)
        l = h1["low"].values.astype(float)
        c = h1["close"].values.astype(float)

        patterns: List[dict] = []

        # Run each detector
        patterns.extend(self._detect_double_top_bottom(h, l, c))
        patterns.extend(self._detect_head_shoulders(h, l, c))
        patterns.extend(self._detect_wedge(h, l, c))
        patterns.extend(self._detect_flag(o, h, l, c))
        patterns.extend(self._detect_inside_bar(o, h, l, c))
        patterns.extend(self._detect_pin_bar(o, h, l, c))

        self._cache[symbol] = {"patterns": patterns, "ts": now}
        return patterns

    def record_pattern_outcome(self, symbol: str, pattern: str,
                               direction: str, entry_price: float,
                               pnl: float, r_multiple: float = 0.0):
        """Record the outcome of a detected pattern after the trade closes."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            conn.execute("""
                INSERT INTO pattern_outcomes
                    (timestamp, symbol, pattern_type, direction, entry_price, pnl, r_multiple)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                symbol, pattern, direction, entry_price, pnl, r_multiple,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.error("record_pattern_outcome error: %s", e)

    def get_pattern_signal(self, symbol: str) -> dict:
        """Return current pattern bias for *symbol*.

        Returns:
            active_patterns : list[dict]  — currently detected patterns
            pattern_bias    : float       — -1 (bearish) to +1 (bullish)
            confidence      : float       — 0..1 based on sample size
        """
        active = self.detect_patterns(symbol)
        if not active:
            return {"active_patterns": [], "pattern_bias": 0.0, "confidence": 0.0}

        # Load historical win rates per pattern type from DB
        wr_map = self._load_pattern_winrates(symbol)

        weighted_bias = 0.0
        weight_sum = 0.0
        min_confidence = 1.0

        for p in active:
            ptype = p["pattern"]
            pdir = p["direction"]   # LONG or SHORT
            strength = p["strength"]

            stats = wr_map.get(ptype)
            if stats and stats["total"] >= 3:
                wr = stats["wins"] / stats["total"]
                # Bias: +1 when 100% win long, -1 when 100% win short
                # Scale from raw WR (0-1) to bias (-1 to +1)
                raw_bias = (wr - 0.5) * 2.0  # positive if > 50% WR
                if pdir == "SHORT":
                    raw_bias = -raw_bias
                conf = min(1.0, stats["total"] / 20.0)  # saturates at 20 samples
            else:
                # No history — use pattern direction with low confidence
                raw_bias = 0.3 if pdir == "LONG" else -0.3
                conf = 0.1

            weighted_bias += raw_bias * strength
            weight_sum += strength
            min_confidence = min(min_confidence, conf)

        if weight_sum > 0:
            pattern_bias = np.clip(weighted_bias / weight_sum, -1.0, 1.0)
        else:
            pattern_bias = 0.0

        # Overall confidence: geometric mean of per-pattern confidence and sample depth
        confidence = round(float(min_confidence), 2)

        return {
            "active_patterns": active,
            "pattern_bias": round(float(pattern_bias), 3),
            "confidence": confidence,
        }

    # ─── INTERNAL: load win rates ────────────────────────────────

    def _load_pattern_winrates(self, symbol: str) -> Dict[str, dict]:
        """Load win/loss counts per pattern_type for *symbol*."""
        result: Dict[str, dict] = {}
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            rows = conn.execute("""
                SELECT pattern_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       AVG(r_multiple) as avg_r
                FROM pattern_outcomes
                WHERE symbol = ?
                GROUP BY pattern_type
            """, (symbol,)).fetchall()
            conn.close()
            for ptype, total, wins, avg_r in rows:
                result[ptype] = {
                    "total": int(total),
                    "wins": int(wins),
                    "avg_r": float(avg_r or 0),
                }
        except Exception as e:
            log.debug("load_pattern_winrates error: %s", e)
        return result

    # ═══════════════════════════════════════════════════════════════
    #  PATTERN DETECTORS
    # ═══════════════════════════════════════════════════════════════

    def _detect_double_top_bottom(self, h, l, c) -> List[dict]:
        """Two highs (or lows) within 0.3% of each other, 5-20 bars apart."""
        patterns = []
        n = len(h)
        lookback = min(n, 50)

        # --- Double top ---
        highs = h[-lookback:]
        for i in range(lookback - 1, 4, -1):
            peak1 = highs[i]
            for j in range(i - 5, max(i - 21, -1), -1):
                if j < 0:
                    break
                peak2 = highs[j]
                pct_diff = abs(peak1 - peak2) / ((peak1 + peak2) / 2) * 100
                if pct_diff <= 0.3:
                    # Confirm: valley between peaks is lower
                    valley = np.min(l[-lookback:][j:i + 1])
                    if valley < min(peak1, peak2) * 0.998:
                        # Recent — only report if peak1 is in last 3 bars
                        if i >= lookback - 3:
                            strength = 1.0 - pct_diff / 0.3  # closer = stronger
                            patterns.append({
                                "pattern": "double_top",
                                "direction": "SHORT",
                                "strength": round(float(np.clip(strength, 0.3, 1.0)), 2),
                            })
                            break
            else:
                continue
            break

        # --- Double bottom ---
        lows = l[-lookback:]
        for i in range(lookback - 1, 4, -1):
            trough1 = lows[i]
            for j in range(i - 5, max(i - 21, -1), -1):
                if j < 0:
                    break
                trough2 = lows[j]
                pct_diff = abs(trough1 - trough2) / ((trough1 + trough2) / 2) * 100
                if pct_diff <= 0.3:
                    peak = np.max(h[-lookback:][j:i + 1])
                    if peak > max(trough1, trough2) * 1.002:
                        if i >= lookback - 3:
                            strength = 1.0 - pct_diff / 0.3
                            patterns.append({
                                "pattern": "double_bottom",
                                "direction": "LONG",
                                "strength": round(float(np.clip(strength, 0.3, 1.0)), 2),
                            })
                            break
            else:
                continue
            break

        return patterns

    def _detect_head_shoulders(self, h, l, c) -> List[dict]:
        """Head & shoulders: 3 peaks, middle highest. Inverse: 3 troughs, middle lowest.
        Scans last 40 bars for the structure."""
        patterns = []
        n = len(h)
        if n < 30:
            return patterns
        lookback = min(n, 40)

        # Find local peaks (H1 candle highs)
        highs = h[-lookback:]
        peaks = self._find_local_extrema(highs, order=3, mode="max")

        if len(peaks) >= 3:
            # Take last 3 peaks
            p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
            h1_v, h2_v, h3_v = highs[p1], highs[p2], highs[p3]

            # Head must be the highest, shoulders roughly equal
            if h2_v > h1_v and h2_v > h3_v:
                shoulder_diff = abs(h1_v - h3_v) / ((h1_v + h3_v) / 2) * 100
                head_prominence = (h2_v - max(h1_v, h3_v)) / h2_v * 100
                if shoulder_diff < 1.0 and head_prominence > 0.2:
                    # Must be recent — right shoulder in last 5 bars
                    if p3 >= lookback - 5:
                        patterns.append({
                            "pattern": "head_shoulders",
                            "direction": "SHORT",
                            "strength": round(float(np.clip(head_prominence / 2.0, 0.3, 1.0)), 2),
                        })

        # Inverse H&S — 3 troughs, middle lowest
        lows = l[-lookback:]
        troughs = self._find_local_extrema(lows, order=3, mode="min")

        if len(troughs) >= 3:
            t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
            l1_v, l2_v, l3_v = lows[t1], lows[t2], lows[t3]

            if l2_v < l1_v and l2_v < l3_v:
                shoulder_diff = abs(l1_v - l3_v) / ((l1_v + l3_v) / 2) * 100
                head_prominence = (min(l1_v, l3_v) - l2_v) / l2_v * 100
                if shoulder_diff < 1.0 and head_prominence > 0.2:
                    if t3 >= lookback - 5:
                        patterns.append({
                            "pattern": "inv_head_shoulders",
                            "direction": "LONG",
                            "strength": round(float(np.clip(head_prominence / 2.0, 0.3, 1.0)), 2),
                        })

        return patterns

    def _detect_wedge(self, h, l, c) -> List[dict]:
        """Converging trendlines: higher lows + lower highs = symmetrical/contracting.
        Rising wedge (bearish): higher highs + higher lows, highs rising slower.
        Falling wedge (bullish): lower highs + lower lows, lows falling slower."""
        patterns = []
        n = len(h)
        if n < 20:
            return patterns

        window = min(n, 25)
        highs = h[-window:]
        lows = l[-window:]
        x = np.arange(window, dtype=float)

        # Fit linear regression to highs and lows
        h_slope, h_intercept = np.polyfit(x, highs, 1)
        l_slope, l_intercept = np.polyfit(x, lows, 1)

        # Residual check — trendlines must fit reasonably well
        h_resid = np.std(highs - (h_slope * x + h_intercept)) / np.mean(highs)
        l_resid = np.std(lows - (l_slope * x + l_intercept)) / np.mean(lows)
        if h_resid > 0.01 or l_resid > 0.01:
            return patterns  # too noisy

        # Converging: slopes moving toward each other
        converging = (h_slope < l_slope) or (h_slope < 0 and l_slope > 0)

        # Normalise slopes to price level for comparison
        price = np.mean(c[-window:])
        h_slope_pct = h_slope / price * 100
        l_slope_pct = l_slope / price * 100

        if converging:
            spread_start = float(highs[0] - lows[0])
            spread_end = float(highs[-1] - lows[-1])
            if spread_end < spread_start * 0.7 and spread_end > 0:
                # Narrowing at least 30%
                if h_slope_pct > 0 and l_slope_pct > 0:
                    # Rising wedge — bearish
                    patterns.append({
                        "pattern": "rising_wedge",
                        "direction": "SHORT",
                        "strength": round(float(np.clip(1.0 - spread_end / spread_start, 0.3, 1.0)), 2),
                    })
                elif h_slope_pct < 0 and l_slope_pct < 0:
                    # Falling wedge — bullish
                    patterns.append({
                        "pattern": "falling_wedge",
                        "direction": "LONG",
                        "strength": round(float(np.clip(1.0 - spread_end / spread_start, 0.3, 1.0)), 2),
                    })
                else:
                    # Symmetrical — direction = breakout of last close vs mid
                    mid = (highs[-1] + lows[-1]) / 2
                    if c[-1] > mid:
                        patterns.append({
                            "pattern": "sym_wedge",
                            "direction": "LONG",
                            "strength": 0.4,
                        })
                    else:
                        patterns.append({
                            "pattern": "sym_wedge",
                            "direction": "SHORT",
                            "strength": 0.4,
                        })

        return patterns

    def _detect_flag(self, o, h, l, c) -> List[dict]:
        """Bull/bear flag: sharp move (pole) then tight consolidation (flag).
        Pole: 5-bar move > 2x recent ATR.
        Flag: next 3-8 bars with range < 50% of pole height."""
        patterns = []
        n = len(c)
        if n < 25:
            return patterns

        # ATR for reference
        tr = np.maximum(
            h[-20:] - l[-20:],
            np.maximum(np.abs(h[-20:] - np.roll(c[-20:], 1)[1:20]),
                       np.abs(l[-20:] - np.roll(c[-20:], 1)[1:20]))
        ) if n >= 21 else h[-20:] - l[-20:]
        atr = float(np.mean(tr))
        if atr <= 0:
            return patterns

        # Check for pole ending 3-8 bars ago, flag = last 3-8 bars
        for flag_len in range(3, 9):
            if n < flag_len + 6:
                continue
            pole_end = n - flag_len
            pole_start = max(0, pole_end - 5)

            pole_move = c[pole_end - 1] - c[pole_start]
            pole_height = abs(pole_move)

            if pole_height < 2.0 * atr:
                continue  # pole not impulsive enough

            flag_high = np.max(h[-flag_len:])
            flag_low = np.min(l[-flag_len:])
            flag_range = flag_high - flag_low

            if flag_range < pole_height * 0.5:
                # Flag is consolidating within the pole
                strength = float(np.clip(1.0 - flag_range / pole_height, 0.3, 1.0))
                if pole_move > 0:
                    patterns.append({
                        "pattern": "bull_flag",
                        "direction": "LONG",
                        "strength": round(strength, 2),
                    })
                else:
                    patterns.append({
                        "pattern": "bear_flag",
                        "direction": "SHORT",
                        "strength": round(strength, 2),
                    })
                break  # one flag detection is enough

        return patterns

    def _detect_inside_bar(self, o, h, l, c) -> List[dict]:
        """Current bar contained within previous bar's range."""
        patterns = []
        n = len(h)
        if n < 2:
            return patterns

        if h[-1] <= h[-2] and l[-1] >= l[-2]:
            # Inside bar — direction from parent candle
            parent_bull = c[-2] > o[-2]
            # Inside bar is a pause; bias follows parent direction
            direction = "LONG" if parent_bull else "SHORT"
            # Strength: how small is inner vs outer
            outer_range = h[-2] - l[-2]
            inner_range = h[-1] - l[-1]
            if outer_range > 0:
                compression = 1.0 - inner_range / outer_range
                patterns.append({
                    "pattern": "inside_bar",
                    "direction": direction,
                    "strength": round(float(np.clip(compression, 0.3, 1.0)), 2),
                })

        return patterns

    def _detect_pin_bar(self, o, h, l, c) -> List[dict]:
        """Pin bar: long wick > 2x body. Signals reversal."""
        patterns = []
        n = len(h)
        if n < 2:
            return patterns

        body = abs(c[-1] - o[-1])
        total_range = h[-1] - l[-1]
        if total_range <= 0 or body <= 0:
            return patterns

        upper_wick = h[-1] - max(o[-1], c[-1])
        lower_wick = min(o[-1], c[-1]) - l[-1]

        # Bullish pin bar: long lower wick (hammer)
        if lower_wick > 2.0 * body and lower_wick > upper_wick * 1.5:
            wick_ratio = lower_wick / body
            patterns.append({
                "pattern": "pin_bar_bull",
                "direction": "LONG",
                "strength": round(float(np.clip(wick_ratio / 5.0, 0.3, 1.0)), 2),
            })

        # Bearish pin bar: long upper wick (shooting star)
        if upper_wick > 2.0 * body and upper_wick > lower_wick * 1.5:
            wick_ratio = upper_wick / body
            patterns.append({
                "pattern": "pin_bar_bear",
                "direction": "SHORT",
                "strength": round(float(np.clip(wick_ratio / 5.0, 0.3, 1.0)), 2),
            })

        return patterns

    # ─── UTILITIES ───────────────────────────────────────────────

    @staticmethod
    def _find_local_extrema(data, order=3, mode="max"):
        """Find indices of local maxima or minima in *data*.
        *order* = number of bars on each side to compare."""
        indices = []
        n = len(data)
        for i in range(order, n - order):
            if mode == "max":
                if all(data[i] >= data[i - j] for j in range(1, order + 1)) and \
                   all(data[i] >= data[i + j] for j in range(1, order + 1)):
                    indices.append(i)
            else:
                if all(data[i] <= data[i - j] for j in range(1, order + 1)) and \
                   all(data[i] <= data[i + j] for j in range(1, order + 1)):
                    indices.append(i)
        return indices
