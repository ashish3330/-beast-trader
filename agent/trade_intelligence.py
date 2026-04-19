"""
Dragon Trader — Trade Intelligence Engine.
Learns from every closed trade and makes the agent smarter over time.

Capabilities:
1. Trade Pattern Memory (SQLite-backed) — fingerprint every trade, query historical edge
2. Score Velocity Tracker — rate of score change per symbol (momentum building/fading)
3. Cross-Symbol Momentum — when correlated symbols agree on direction
4. Smart Re-Entry Signal — track post-SL price action for re-entry decisions

Runs in the brain's hot path (~500ms cycle) — all methods are cached and lightweight.
"""
import time
import logging
import sqlite3
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, SYMBOLS

log = logging.getLogger("dragon.trade_intel")

JOURNAL_DB = DB_PATH.parent / "trade_journal.db"

# ═══ SCORE BUCKETS ═══
SCORE_BUCKETS = [(6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 100.0)]

def _score_bucket(score: float) -> str:
    """Map a score to its bucket label."""
    for lo, hi in SCORE_BUCKETS:
        if lo <= score < hi:
            return f"{lo:.0f}-{hi:.0f}"
    if score >= 9.0:
        return "9-100"
    return "0-6"

# ═══ CROSS-SYMBOL GROUPINGS ═══
_USD_GROUP = ["XAUUSD", "XAGUSD", "USDJPY", "USDCHF", "USDCAD"]
_USD_INVERSE = {"XAUUSD", "XAGUSD"}  # gold rises when USD falls

_JPY_GROUP = ["USDJPY", "EURJPY", "JPN225ft"]

_RISK_GROUP = ["BTCUSD", "NAS100.r", "XAUUSD"]

CROSS_GROUPS: Dict[str, List[Tuple[str, List[str], set]]] = {}
# Build lookup: symbol -> list of (group_name, members, inverse_set)
for sym in SYMBOLS:
    groups = []
    if sym in _USD_GROUP:
        groups.append(("USD", _USD_GROUP, _USD_INVERSE))
    if sym in _JPY_GROUP:
        groups.append(("JPY", _JPY_GROUP, set()))
    if sym in _RISK_GROUP:
        groups.append(("RISK", _RISK_GROUP, set()))
    CROSS_GROUPS[sym] = groups

# ═══ RE-ENTRY TRACKER CONSTANTS ═══
REENTRY_LOOKBACK_BARS = 15       # check price 15 bars after SL hit
REENTRY_CONTINUE_THRESHOLD = 0.5  # price moved 0.5x ATR in our direction = re-enter
REENTRY_COOLDOWN_S = 300          # 5 min cooldown between re-entries


class TradeIntelligence:
    """Learns from trades and enriches entry decisions."""

    def __init__(self, state, learning_engine):
        self.state = state
        self._learning = learning_engine

        # ── Pattern edge cache (refreshed every 60s from DB) ──
        self._pattern_cache: Dict[str, dict] = {}   # cache_key -> edge_dict
        self._pattern_cache_ts: float = 0
        self._PATTERN_CACHE_TTL = 60.0  # refresh from DB every 60s

        # ── Score velocity ring buffers (from learning_engine._score_history) ──
        self._velocity_cache: Dict[str, dict] = {}  # symbol -> {velocity, accel, state, ts}
        self._VELOCITY_CACHE_TTL = 2.0  # recompute every 2s

        # ── Cross momentum cache ──
        self._cross_cache: Dict[str, dict] = {}
        self._cross_cache_ts: float = 0
        self._CROSS_CACHE_TTL = 5.0

        # ── Re-entry tracker ──
        self._sl_events: Dict[str, dict] = {}  # symbol -> {direction, price, time, atr}
        self._reentry_cache: Dict[str, dict] = {}  # symbol -> {should, reason, confidence, ts}
        self._REENTRY_CACHE_TTL = 10.0

        # ── Ensure DB table exists ──
        self._init_db()

        log.info("TradeIntelligence initialized (pattern_edge + velocity + cross_momentum + reentry)")

    # ═══════════════════════════════════════════════════════════════
    #  DB INITIALIZATION
    # ═══════════════════════════════════════════════════════════════

    def _init_db(self):
        """Ensure trade_patterns table exists in trade_journal.db."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=5.0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    score_bucket TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    hour_utc INTEGER NOT NULL,
                    day_of_week INTEGER NOT NULL,
                    m15_alignment TEXT NOT NULL,
                    was_winner INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    r_multiple REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tp_lookup
                ON trade_patterns (symbol, direction, regime, score_bucket)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("TradeIntelligence DB init error: %s", e)

    # ═══════════════════════════════════════════════════════════════
    #  1. TRADE PATTERN MEMORY
    # ═══════════════════════════════════════════════════════════════

    def record_pattern(self, symbol: str, direction: str, score: float,
                       regime: str, m15_dir: str, pnl: float, r_multiple: float):
        """Record a closed trade's fingerprint for pattern learning.
        Called from brain._record_trade_result() after each closed trade."""
        now = datetime.now(timezone.utc)
        bucket = _score_bucket(score)
        was_winner = 1 if pnl > 0 else 0

        # Determine M15 alignment relative to trade direction
        if m15_dir == direction:
            m15_align = "agree"
        elif m15_dir == "FLAT":
            m15_align = "flat"
        else:
            m15_align = "oppose"

        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=5.0)
            conn.execute("""
                INSERT INTO trade_patterns
                (timestamp, symbol, direction, score_bucket, regime, hour_utc,
                 day_of_week, m15_alignment, was_winner, pnl, r_multiple)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now.isoformat(), symbol, direction, bucket, regime,
                now.hour, now.weekday(), m15_align,
                was_winner, round(pnl, 2), round(r_multiple, 3),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("[%s] Pattern record error: %s", symbol, e)

        # Invalidate cache so next query picks up new data
        self._pattern_cache_ts = 0

    def get_pattern_edge(self, symbol: str, direction: str, regime: str,
                         hour: int = -1, score_bucket: str = "") -> dict:
        """Query historical edge for matching trade pattern.
        Returns: {win_rate, sample_size, avg_r_multiple, confidence}

        Uses cached data — DB query runs at most once per 60s.
        """
        # Refresh cache if stale
        now = time.time()
        if now - self._pattern_cache_ts > self._PATTERN_CACHE_TTL:
            self._refresh_pattern_cache()

        # Build cache key — primary match is symbol + direction + regime + bucket
        key = f"{symbol}|{direction}|{regime}|{score_bucket}"
        cached = self._pattern_cache.get(key)
        if cached:
            return dict(cached)

        # Fallback: broader match (symbol + direction only)
        fallback_key = f"{symbol}|{direction}||"
        cached = self._pattern_cache.get(fallback_key)
        if cached:
            return dict(cached)

        return {"win_rate": 0.5, "sample_size": 0, "avg_r_multiple": 0.0, "confidence": 0.0}

    def _refresh_pattern_cache(self):
        """Load pattern statistics from DB into memory cache.
        Runs once per 60s — groups by (symbol, direction, regime, score_bucket)."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=5.0)
            rows = conn.execute("""
                SELECT symbol, direction, regime, score_bucket,
                       COUNT(*) as cnt,
                       SUM(was_winner) as wins,
                       AVG(r_multiple) as avg_r
                FROM trade_patterns
                GROUP BY symbol, direction, regime, score_bucket
            """).fetchall()

            # Also aggregate broader (symbol + direction) for fallback
            broad_rows = conn.execute("""
                SELECT symbol, direction,
                       COUNT(*) as cnt,
                       SUM(was_winner) as wins,
                       AVG(r_multiple) as avg_r
                FROM trade_patterns
                GROUP BY symbol, direction
            """).fetchall()
            conn.close()

            cache = {}
            for sym, d, regime, bucket, cnt, wins, avg_r in rows:
                wr = wins / cnt if cnt > 0 else 0.5
                # Confidence: sigmoid-like ramp from 0 at 3 samples to 1.0 at 30+
                confidence = min(1.0, max(0.0, (cnt - 3) / 27.0)) if cnt >= 3 else 0.0
                key = f"{sym}|{d}|{regime}|{bucket}"
                cache[key] = {
                    "win_rate": round(wr, 3),
                    "sample_size": cnt,
                    "avg_r_multiple": round(avg_r or 0, 3),
                    "confidence": round(confidence, 3),
                }

            for sym, d, cnt, wins, avg_r in broad_rows:
                wr = wins / cnt if cnt > 0 else 0.5
                confidence = min(1.0, max(0.0, (cnt - 3) / 27.0)) if cnt >= 3 else 0.0
                key = f"{sym}|{d}||"
                cache[key] = {
                    "win_rate": round(wr, 3),
                    "sample_size": cnt,
                    "avg_r_multiple": round(avg_r or 0, 3),
                    "confidence": round(confidence, 3),
                }

            self._pattern_cache = cache
            self._pattern_cache_ts = time.time()
        except Exception as e:
            log.debug("Pattern cache refresh error: %s", e)

    # ═══════════════════════════════════════════════════════════════
    #  2. SCORE VELOCITY TRACKER
    # ═══════════════════════════════════════════════════════════════

    def get_score_velocity(self, symbol: str) -> dict:
        """Get rate of score change for a symbol.
        Returns: {velocity, acceleration, state}

        Uses learning_engine._score_history (already maintained every 1s).
        Cached per symbol with 2s TTL.
        """
        now = time.time()
        cached = self._velocity_cache.get(symbol)
        if cached and now - cached.get("ts", 0) < self._VELOCITY_CACHE_TTL:
            return {k: v for k, v in cached.items() if k != "ts"}

        result = {"velocity": 0.0, "acceleration": 0.0, "state": "stable"}

        # Read from learning engine's score history (already in memory)
        sh = self._learning._score_history.get(symbol, [])
        if len(sh) < 5:
            self._velocity_cache[symbol] = {**result, "ts": now}
            return result

        # Last 5 observations for velocity (smoothed)
        recent = sh[-5:]
        scores = [s["best"] for s in recent]
        times = [s["t"] for s in recent]

        # Time span in minutes
        dt_min = (times[-1] - times[0]) / 60.0
        if dt_min < 0.01:
            self._velocity_cache[symbol] = {**result, "ts": now}
            return result

        # Velocity = score change per minute (linear regression slope)
        import numpy as np
        t_arr = np.array([(t - times[0]) / 60.0 for t in times])
        s_arr = np.array(scores)
        if len(t_arr) >= 3:
            coeffs = np.polyfit(t_arr, s_arr, 1)
            velocity = float(coeffs[0])  # score points per minute
        else:
            velocity = (scores[-1] - scores[0]) / dt_min

        # Acceleration: compare velocity of last 5 vs previous 5
        accel = 0.0
        if len(sh) >= 10:
            older = sh[-10:-5]
            older_scores = [s["best"] for s in older]
            older_times = [s["t"] for s in older]
            older_dt = (older_times[-1] - older_times[0]) / 60.0
            if older_dt > 0.01:
                older_vel = (older_scores[-1] - older_scores[0]) / older_dt
                accel = velocity - older_vel

        # State classification
        if velocity > 0.1:
            state = "building"
        elif velocity < -0.1:
            state = "fading"
        else:
            state = "stable"

        result = {
            "velocity": round(velocity, 4),
            "acceleration": round(accel, 4),
            "state": state,
        }
        self._velocity_cache[symbol] = {**result, "ts": now}
        return result

    # ═══════════════════════════════════════════════════════════════
    #  3. CROSS-SYMBOL MOMENTUM
    # ═══════════════════════════════════════════════════════════════

    def get_cross_momentum(self, symbol: str) -> dict:
        """Check if correlated symbols agree on direction.
        Returns: {alignment_score (0-1), aligned_symbols}

        Cached with 5s TTL.
        """
        now = time.time()
        if now - self._cross_cache_ts < self._CROSS_CACHE_TTL and symbol in self._cross_cache:
            return dict(self._cross_cache[symbol])

        result = {"alignment_score": 0.0, "aligned_symbols": []}

        groups = CROSS_GROUPS.get(symbol, [])
        if not groups:
            self._cross_cache[symbol] = result
            return result

        # Get current dominant direction for this symbol from learning engine
        obs = self._learning._market_obs.get(symbol, {})
        my_dir = obs.get("dominant_dir", "FLAT")
        if my_dir == "FLAT":
            self._cross_cache[symbol] = result
            return result

        # Check each group this symbol belongs to
        all_aligned = []
        total_members = 0
        aligned_count = 0

        for group_name, members, inverse_set in groups:
            for other in members:
                if other == symbol:
                    continue
                if other not in SYMBOLS:
                    continue
                total_members += 1

                other_obs = self._learning._market_obs.get(other, {})
                other_dir = other_obs.get("dominant_dir", "FLAT")
                if other_dir == "FLAT":
                    continue

                # Handle inverse relationships
                # For USD group: if WE are in inverse set (gold), our LONG = USD weak
                # If OTHER is in inverse set, their LONG = USD weak
                my_effective = my_dir
                other_effective = other_dir

                # Check if directions are aligned considering inversions
                i_am_inverse = symbol in inverse_set
                other_is_inverse = other in inverse_set

                if i_am_inverse != other_is_inverse:
                    # One is inverse, one is not — aligned means opposite directions
                    if my_dir != other_dir:
                        aligned_count += 1
                        all_aligned.append(other)
                else:
                    # Both same type — aligned means same direction
                    if my_dir == other_dir:
                        aligned_count += 1
                        all_aligned.append(other)

        alignment = aligned_count / total_members if total_members > 0 else 0.0
        # Deduplicate aligned symbols (a symbol can appear in multiple groups)
        unique_aligned = list(set(all_aligned))

        result = {
            "alignment_score": round(alignment, 3),
            "aligned_symbols": unique_aligned,
        }
        self._cross_cache[symbol] = result
        self._cross_cache_ts = now
        return result

    # ═══════════════════════════════════════════════════════════════
    #  4. SMART RE-ENTRY SIGNAL
    # ═══════════════════════════════════════════════════════════════

    def record_stoploss(self, symbol: str, direction: str, price: float, atr: float = 0.0):
        """Record that a position was stopped out. Called from executor on SL close.
        Stores the SL event so we can track post-exit price action."""
        self._sl_events[symbol] = {
            "direction": direction,
            "price": price,
            "atr": atr,
            "time": time.time(),
        }
        # Clear any existing re-entry signal
        self._reentry_cache.pop(symbol, None)
        log.info("[%s] SL event recorded: dir=%s price=%.5f atr=%.5f",
                 symbol, direction, price, atr)

    def check_reentry(self, symbol: str) -> dict:
        """Check if we should re-enter after a stop-loss.
        Returns: {should_reenter, reason, confidence}

        Logic: 5-15 bars after SL, did price continue in our original direction?
        If yes → wrong SL, should re-enter. If no → correct exit, stay away.
        """
        now = time.time()

        # Check cache
        cached = self._reentry_cache.get(symbol)
        if cached and now - cached.get("ts", 0) < self._REENTRY_CACHE_TTL:
            return {k: v for k, v in cached.items() if k != "ts"}

        result = {"should_reenter": False, "reason": "no_sl_event", "confidence": 0.0}

        sl_event = self._sl_events.get(symbol)
        if not sl_event:
            return result

        elapsed = now - sl_event["time"]

        # Too early (< 2 min) or too late (> 30 min) — not actionable
        if elapsed < 120:
            result["reason"] = "too_early"
            self._reentry_cache[symbol] = {**result, "ts": now}
            return result
        if elapsed > 1800:
            # Clean up old events
            self._sl_events.pop(symbol, None)
            result["reason"] = "expired"
            self._reentry_cache[symbol] = {**result, "ts": now}
            return result

        # Get current price
        tick = self.state.get_tick(symbol)
        if not tick:
            self._reentry_cache[symbol] = {**result, "ts": now}
            return result

        current_price = float(tick.bid) if hasattr(tick, 'bid') else float(tick.get("ltp", 0))
        sl_price = sl_event["price"]
        sl_dir = sl_event["direction"]
        atr = sl_event.get("atr", 0)

        if current_price <= 0 or sl_price <= 0:
            self._reentry_cache[symbol] = {**result, "ts": now}
            return result

        # Calculate price movement since SL
        if sl_dir == "LONG":
            move = current_price - sl_price  # positive = price went up (our direction)
        else:
            move = sl_price - current_price  # positive = price went down (our direction)

        # Normalize by ATR if available, else by price
        if atr > 0:
            move_atr = move / atr
        else:
            move_atr = move / (sl_price * 0.001)  # rough normalization

        if move_atr > REENTRY_CONTINUE_THRESHOLD:
            # Price continued in our direction — SL was wrong
            confidence = min(1.0, move_atr / 2.0)  # higher move = higher confidence
            result = {
                "should_reenter": True,
                "reason": "price_continued",
                "confidence": round(confidence, 3),
            }
        elif move_atr < -REENTRY_CONTINUE_THRESHOLD:
            # Price reversed — SL was correct
            result = {
                "should_reenter": False,
                "reason": "reversal_confirmed",
                "confidence": round(min(1.0, abs(move_atr) / 2.0), 3),
            }
        else:
            result = {
                "should_reenter": False,
                "reason": "inconclusive",
                "confidence": 0.0,
            }

        self._reentry_cache[symbol] = {**result, "ts": now}
        return result

    # ═══════════════════════════════════════════════════════════════
    #  COMBINED ENTRY INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════

    def get_entry_intelligence(self, symbol: str, direction: str,
                               score: float, regime: str) -> dict:
        """Combine all 4 intelligence signals into a single entry assessment.
        Called by brain before opening a trade.

        Returns:
            {
                "pattern_edge": {"wr": 0.72, "samples": 15, "confidence": 0.85},
                "score_velocity": {"velocity": 0.3, "state": "building"},
                "cross_momentum": {"alignment": 0.8, "aligned": [...]},
                "reentry": {"active": False},
                "combined_boost": 1.15,  # multiplier for risk sizing (0.5-1.5)
            }
        """
        bucket = _score_bucket(score)

        # 1. Pattern edge
        edge = self.get_pattern_edge(symbol, direction, regime, score_bucket=bucket)
        pattern = {
            "wr": edge["win_rate"],
            "samples": edge["sample_size"],
            "confidence": edge["confidence"],
        }

        # 2. Score velocity
        vel = self.get_score_velocity(symbol)
        velocity = {
            "velocity": vel["velocity"],
            "state": vel["state"],
        }

        # 3. Cross-symbol momentum
        cross = self.get_cross_momentum(symbol)
        cross_mom = {
            "alignment": cross["alignment_score"],
            "aligned": cross["aligned_symbols"],
        }

        # 4. Re-entry check
        reentry = self.check_reentry(symbol)
        reentry_info = {
            "active": reentry["should_reenter"],
            "reason": reentry.get("reason", ""),
        }

        # ═══ COMBINED BOOST CALCULATION ═══
        boost = 1.0

        # Pattern edge boost/penalty
        if edge["sample_size"] >= 10:
            if edge["win_rate"] > 0.6:
                boost *= 1.1
            elif edge["win_rate"] < 0.35:
                boost *= 0.7

        # Score velocity boost
        if vel["state"] == "building":
            boost *= 1.1

        # Cross-symbol momentum boost
        if cross["alignment_score"] > 0.7:
            boost *= 1.1

        # Cap at 0.5-1.5
        boost = max(0.5, min(1.5, boost))
        boost = round(boost, 3)

        return {
            "pattern_edge": pattern,
            "score_velocity": velocity,
            "cross_momentum": cross_mom,
            "reentry": reentry_info,
            "combined_boost": boost,
        }
