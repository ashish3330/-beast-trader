"""
Dragon Trader — Price Level Memory.
Learns which price levels cause bounces vs breakouts for each symbol.
Builds persistent per-symbol level maps from trade outcomes.

Capabilities:
1. Records price levels from trade outcomes (entry, SL, TP)
2. Builds per-symbol level map with bounce/break counts + strength
3. Round number intelligence — learns symbol-specific round number behavior
4. Clusters nearby levels (within 0.1%) to avoid duplicates
5. Caches aggressively — refreshes from DB every 120s max
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
from config import DB_PATH, SYMBOLS

log = logging.getLogger("dragon.levels")

JOURNAL_DB = DB_PATH.parent / "trade_journal.db"
JOURNAL_DB.parent.mkdir(parents=True, exist_ok=True)

# Clustering threshold: levels within 0.1% are merged
CLUSTER_PCT = 0.001

# Cache TTL
CACHE_TTL_SEC = 120

# Round number definitions per category
ROUND_LEVEL_DEFS = {
    "Gold":   [("100s", 100), ("50s", 50), ("10s", 10)],
    "Crypto": [("1000s", 1000), ("500s", 500), ("100s", 100)],
    "Index":  [("1000s", 1000), ("500s", 500), ("100s", 100)],
    "Forex":  [("100pips", 0.01), ("50pips", 0.005), ("10pips", 0.001)],
}

# Map symbols to categories for round number definitions
def _get_category(symbol: str) -> str:
    cfg = SYMBOLS.get(symbol)
    if cfg:
        return cfg.category
    return "Forex"


class LevelMemory:
    """Learns price levels from trade outcomes. Provides level intelligence."""

    def __init__(self):
        # Per-symbol cache: symbol -> {"levels": [...], "round_stats": {...}, "ts": float}
        self._cache: Dict[str, dict] = {}
        self._init_db()

    # ═══════════════════════════════════════════════════════════════
    #  DATABASE INIT
    # ═══════════════════════════════════════════════════════════════

    def _init_db(self):
        """Create price_levels and round_level_stats tables if not exists."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_levels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    level_price REAL NOT NULL,
                    bounce_count INTEGER DEFAULT 0,
                    break_count INTEGER DEFAULT 0,
                    last_touched TEXT,
                    strength REAL DEFAULT 0.5
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_levels_symbol
                ON price_levels (symbol)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS round_level_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    level_type TEXT NOT NULL,
                    bounce_count INTEGER DEFAULT 0,
                    break_count INTEGER DEFAULT 0,
                    bounce_pct REAL DEFAULT 0.5,
                    sample_size INTEGER DEFAULT 0,
                    UNIQUE(symbol, level_type)
                )
            """)
            conn.commit()
            conn.close()
            log.info("Price level memory initialized: %s", JOURNAL_DB)
        except Exception as e:
            log.error("Failed to init price level tables: %s", e)

    # ═══════════════════════════════════════════════════════════════
    #  RECORD EVENTS
    # ═══════════════════════════════════════════════════════════════

    def record_level_event(self, symbol: str, price: float, event_type: str):
        """Record a price level event.

        event_type: "bounce", "break", "sl_hit", "tp_hit"
        - bounce / sl_hit → increments bounce_count (price held)
        - break / tp_hit  → increments break_count (price broke through)
        """
        if price <= 0:
            return

        is_bounce = event_type in ("bounce", "sl_hit")
        is_break = event_type in ("break", "tp_hit")
        if not is_bounce and not is_break:
            log.warning("Unknown level event type: %s", event_type)
            return

        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            now_iso = datetime.now(timezone.utc).isoformat()

            # Find existing level within cluster distance
            existing = self._find_cluster_level(conn, symbol, price)

            if existing:
                level_id, level_price, bc, brc = existing
                # Merge price toward weighted average
                total = bc + brc + 1
                merged_price = (level_price * (total - 1) + price) / total
                if is_bounce:
                    bc += 1
                else:
                    brc += 1
                strength = bc / (bc + brc) if (bc + brc) > 0 else 0.5
                conn.execute("""
                    UPDATE price_levels
                    SET level_price=?, bounce_count=?, break_count=?,
                        last_touched=?, strength=?
                    WHERE id=?
                """, (round(merged_price, 6), bc, brc, now_iso, round(strength, 4), level_id))
            else:
                # New level
                bc = 1 if is_bounce else 0
                brc = 1 if is_break else 0
                strength = bc / (bc + brc) if (bc + brc) > 0 else 0.5
                conn.execute("""
                    INSERT INTO price_levels
                        (symbol, level_price, bounce_count, break_count, last_touched, strength)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (symbol, round(price, 6), bc, brc, now_iso, round(strength, 4)))

            # Update round number stats if price is near a round number
            self._update_round_stats(conn, symbol, price, is_bounce)

            conn.commit()
            conn.close()

            # Invalidate cache for this symbol
            self._cache.pop(symbol, None)

            log.debug("LEVEL %s: %s %.5f %s (bounce=%d break=%d)",
                      symbol, event_type, price,
                      "clustered" if existing else "new",
                      bc, brc)

        except Exception as e:
            log.error("Failed to record level event: %s", e)

    def record_trade_outcome(self, symbol: str, entry_price: float,
                             exit_price: float, sl_price: float,
                             tp_price: float, exit_reason: str):
        """Record level events from a completed trade.

        Logic:
        - Entry price is always recorded as a level touch
        - If SL hit: the SL price acted as a breakout level (price went through it)
          AND the entry price was a bounce (price came back from entry direction)
        - If TP hit: the TP price acted as a breakout (price pushed through to target)
          AND the entry price area was a breakout (momentum carried through)
        - If trailed out: entry was a bounce level if profitable, break if not
        """
        if entry_price <= 0:
            return

        exit_lower = exit_reason.lower() if exit_reason else ""

        if "sl" in exit_lower or "stop" in exit_lower:
            # SL hit — price reversed against us
            # The SL price level was broken through
            self.record_level_event(symbol, sl_price if sl_price > 0 else exit_price, "break")
            # Entry area acted as temporary support/resistance that failed
            self.record_level_event(symbol, entry_price, "break")

        elif "tp" in exit_lower or "take" in exit_lower:
            # TP hit — price pushed through to target
            self.record_level_event(symbol, tp_price if tp_price > 0 else exit_price, "break")
            # Entry area was a launchpad (breakout)
            self.record_level_event(symbol, entry_price, "break")

        elif "trail" in exit_lower or "lock" in exit_lower or "be" in exit_lower:
            # Trailed out — check if profitable
            if exit_price > entry_price:
                # Profitable trail: entry was a good bounce level
                self.record_level_event(symbol, entry_price, "bounce")
            else:
                # Unprofitable trail: entry broke down
                self.record_level_event(symbol, entry_price, "break")
            # Exit price itself is a level where price reversed (bounce)
            self.record_level_event(symbol, exit_price, "bounce")

        else:
            # Generic close — record exit as a bounce (price stopped there)
            self.record_level_event(symbol, exit_price, "bounce")
            self.record_level_event(symbol, entry_price, "bounce")

    # ═══════════════════════════════════════════════════════════════
    #  LEVEL INTELLIGENCE QUERY
    # ═══════════════════════════════════════════════════════════════

    def get_level_intelligence(self, symbol: str, current_price: float) -> dict:
        """Get full level intelligence for a symbol at current price.

        Returns dict with nearest support/resistance, strengths,
        round number bias, and all learned levels nearby.
        """
        result = {
            "nearest_support": 0.0,
            "nearest_resistance": 0.0,
            "support_strength": 0.0,
            "resistance_strength": 0.0,
            "round_level_bias": 0.0,
            "at_learned_level": False,
            "learned_levels": [],
        }

        if current_price <= 0:
            return result

        levels = self._get_cached_levels(symbol)
        round_stats = self._get_cached_round_stats(symbol)

        if not levels:
            # Still compute round level bias even without learned levels
            result["round_level_bias"] = self._compute_round_bias(
                symbol, current_price, round_stats)
            return result

        # Separate into support (below price) and resistance (above price)
        # A level is support if strength > 0.5 (more bounces than breaks)
        # A level is resistance if strength > 0.5 AND above price
        # But for nearest support/resistance, we just find closest below/above
        # with sufficient touches
        supports = []
        resistances = []
        at_level_threshold = current_price * CLUSTER_PCT  # within 0.1%

        for lv in levels:
            lp = lv["price"]
            touches = lv["bounce_count"] + lv["break_count"]
            if touches < 2:
                continue  # need at least 2 touches to be meaningful

            dist = abs(lp - current_price)
            if dist < at_level_threshold:
                result["at_learned_level"] = True

            if lp < current_price:
                supports.append(lv)
            elif lp > current_price:
                resistances.append(lv)

        # Nearest support: closest below with highest strength
        if supports:
            # Sort by distance (closest first), weighted by strength
            supports.sort(key=lambda x: abs(x["price"] - current_price))
            best_sup = supports[0]
            result["nearest_support"] = best_sup["price"]
            result["support_strength"] = best_sup["strength"]

        # Nearest resistance: closest above with highest strength
        if resistances:
            resistances.sort(key=lambda x: abs(x["price"] - current_price))
            best_res = resistances[0]
            result["nearest_resistance"] = best_res["price"]
            result["resistance_strength"] = best_res["strength"]

        # Round level bias
        result["round_level_bias"] = self._compute_round_bias(
            symbol, current_price, round_stats)

        # All learned levels within 2% of current price (for dashboard / brain)
        range_pct = 0.02
        nearby = []
        for lv in levels:
            if abs(lv["price"] - current_price) / current_price <= range_pct:
                touches = lv["bounce_count"] + lv["break_count"]
                if touches >= 2:
                    ltype = "support" if lv["strength"] >= 0.5 else "resistance"
                    nearby.append({
                        "price": lv["price"],
                        "type": ltype,
                        "strength": lv["strength"],
                        "touches": touches,
                    })
        # Sort by strength descending
        nearby.sort(key=lambda x: x["strength"], reverse=True)
        result["learned_levels"] = nearby[:20]  # cap at 20 levels

        return result

    # ═══════════════════════════════════════════════════════════════
    #  ROUND NUMBER INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════

    def _update_round_stats(self, conn, symbol: str, price: float, is_bounce: bool):
        """Update round_level_stats if price is near a round number."""
        category = _get_category(symbol)
        defs = ROUND_LEVEL_DEFS.get(category, ROUND_LEVEL_DEFS["Forex"])

        for level_type, step in defs:
            if step <= 0:
                continue
            nearest_round = round(price / step) * step
            dist_pct = abs(price - nearest_round) / price if price > 0 else 1.0

            # Only count if within 0.15% of a round number
            if dist_pct > 0.0015:
                continue

            # Upsert round_level_stats
            row = conn.execute(
                "SELECT bounce_count, break_count FROM round_level_stats "
                "WHERE symbol=? AND level_type=?",
                (symbol, level_type)
            ).fetchone()

            if row:
                bc, brc = row
                if is_bounce:
                    bc += 1
                else:
                    brc += 1
                total = bc + brc
                bounce_pct = bc / total if total > 0 else 0.5
                conn.execute("""
                    UPDATE round_level_stats
                    SET bounce_count=?, break_count=?, bounce_pct=?, sample_size=?
                    WHERE symbol=? AND level_type=?
                """, (bc, brc, round(bounce_pct, 4), total, symbol, level_type))
            else:
                bc = 1 if is_bounce else 0
                brc = 1 if not is_bounce else 0
                total = bc + brc
                bounce_pct = bc / total if total > 0 else 0.5
                conn.execute("""
                    INSERT INTO round_level_stats
                        (symbol, level_type, bounce_count, break_count, bounce_pct, sample_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (symbol, level_type, bc, brc, round(bounce_pct, 4), total))

    def _compute_round_bias(self, symbol: str, price: float,
                            round_stats: Dict[str, dict]) -> float:
        """Compute round level bias: -1 (breakout bias) to +1 (bounce bias).

        Checks if current price is near a round number and what the
        historical bounce/break ratio is for this symbol at that type of level.
        """
        if price <= 0:
            return 0.0

        category = _get_category(symbol)
        defs = ROUND_LEVEL_DEFS.get(category, ROUND_LEVEL_DEFS["Forex"])

        best_bias = 0.0
        best_proximity = 1.0  # lower = closer to round number

        for level_type, step in defs:
            if step <= 0:
                continue
            nearest_round = round(price / step) * step
            dist_pct = abs(price - nearest_round) / price if price > 0 else 1.0

            # Only relevant if within 0.3% of round number
            if dist_pct > 0.003:
                continue

            stats = round_stats.get(level_type)
            if stats and stats["sample_size"] >= 5:
                # bounce_pct > 0.5 means bounce bias, < 0.5 means break bias
                # Scale to -1..+1 range
                bias = (stats["bounce_pct"] - 0.5) * 2.0
                # Weight by proximity (closer = stronger signal)
                proximity_weight = 1.0 - (dist_pct / 0.003)
                weighted_bias = bias * proximity_weight

                if abs(weighted_bias) > abs(best_bias):
                    best_bias = weighted_bias
                    best_proximity = dist_pct

        return round(max(-1.0, min(1.0, best_bias)), 3)

    # ═══════════════════════════════════════════════════════════════
    #  CLUSTERING
    # ═══════════════════════════════════════════════════════════════

    def _find_cluster_level(self, conn, symbol: str, price: float) -> Optional[tuple]:
        """Find an existing level within CLUSTER_PCT of the given price.
        Returns (id, level_price, bounce_count, break_count) or None.
        """
        # Price range for clustering: price +/- 0.1%
        lo = price * (1 - CLUSTER_PCT)
        hi = price * (1 + CLUSTER_PCT)

        row = conn.execute(
            "SELECT id, level_price, bounce_count, break_count FROM price_levels "
            "WHERE symbol=? AND level_price BETWEEN ? AND ? "
            "ORDER BY ABS(level_price - ?) LIMIT 1",
            (symbol, lo, hi, price)
        ).fetchone()

        return row if row else None

    def merge_close_levels(self, symbol: str):
        """Merge all levels within CLUSTER_PCT of each other for a symbol.
        Called periodically to keep the level map clean.
        """
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            rows = conn.execute(
                "SELECT id, level_price, bounce_count, break_count FROM price_levels "
                "WHERE symbol=? ORDER BY level_price",
                (symbol,)
            ).fetchall()

            if len(rows) < 2:
                conn.close()
                return

            merged = []
            i = 0
            while i < len(rows):
                cluster = [rows[i]]
                j = i + 1
                while j < len(rows):
                    dist_pct = abs(rows[j][1] - cluster[0][1]) / cluster[0][1]
                    if dist_pct <= CLUSTER_PCT:
                        cluster.append(rows[j])
                        j += 1
                    else:
                        break

                if len(cluster) > 1:
                    # Merge: weighted average price, sum counts
                    total_touches = sum(r[2] + r[3] for r in cluster)
                    if total_touches > 0:
                        avg_price = sum(r[1] * (r[2] + r[3]) for r in cluster) / total_touches
                    else:
                        avg_price = np.mean([r[1] for r in cluster])
                    total_bc = sum(r[2] for r in cluster)
                    total_brc = sum(r[3] for r in cluster)
                    strength = total_bc / (total_bc + total_brc) if (total_bc + total_brc) > 0 else 0.5
                    now_iso = datetime.now(timezone.utc).isoformat()

                    # Keep the first ID, delete the rest
                    keep_id = cluster[0][0]
                    conn.execute("""
                        UPDATE price_levels
                        SET level_price=?, bounce_count=?, break_count=?,
                            strength=?, last_touched=?
                        WHERE id=?
                    """, (round(avg_price, 6), total_bc, total_brc,
                          round(strength, 4), now_iso, keep_id))

                    for r in cluster[1:]:
                        conn.execute("DELETE FROM price_levels WHERE id=?", (r[0],))
                    merged.append(keep_id)

                i = j

            conn.commit()
            conn.close()

            if merged:
                self._cache.pop(symbol, None)
                log.debug("LEVEL MERGE %s: consolidated %d clusters", symbol, len(merged))

        except Exception as e:
            log.error("Level merge error for %s: %s", symbol, e)

    # ═══════════════════════════════════════════════════════════════
    #  CACHING
    # ═══════════════════════════════════════════════════════════════

    def _get_cached_levels(self, symbol: str) -> List[dict]:
        """Get levels for symbol, refreshing from DB if cache is stale."""
        cached = self._cache.get(symbol)
        if cached and (time.time() - cached["ts"]) < CACHE_TTL_SEC:
            return cached["levels"]

        levels = self._load_levels_from_db(symbol)
        round_stats = self._load_round_stats_from_db(symbol)

        self._cache[symbol] = {
            "levels": levels,
            "round_stats": round_stats,
            "ts": time.time(),
        }
        return levels

    def _get_cached_round_stats(self, symbol: str) -> Dict[str, dict]:
        """Get round stats for symbol, using cache."""
        cached = self._cache.get(symbol)
        if cached and (time.time() - cached["ts"]) < CACHE_TTL_SEC:
            return cached.get("round_stats", {})

        # Force full cache refresh
        self._get_cached_levels(symbol)
        cached = self._cache.get(symbol, {})
        return cached.get("round_stats", {})

    def _load_levels_from_db(self, symbol: str) -> List[dict]:
        """Load all price levels for a symbol from SQLite."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            rows = conn.execute(
                "SELECT level_price, bounce_count, break_count, last_touched, strength "
                "FROM price_levels WHERE symbol=? ORDER BY level_price",
                (symbol,)
            ).fetchall()
            conn.close()

            return [
                {
                    "price": r[0],
                    "bounce_count": r[1],
                    "break_count": r[2],
                    "last_touched": r[3],
                    "strength": r[4],
                }
                for r in rows
            ]
        except Exception as e:
            log.error("Failed to load levels for %s: %s", symbol, e)
            return []

    def _load_round_stats_from_db(self, symbol: str) -> Dict[str, dict]:
        """Load round level stats for a symbol from SQLite."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            rows = conn.execute(
                "SELECT level_type, bounce_count, break_count, bounce_pct, sample_size "
                "FROM round_level_stats WHERE symbol=?",
                (symbol,)
            ).fetchall()
            conn.close()

            return {
                r[0]: {
                    "bounce_count": r[1],
                    "break_count": r[2],
                    "bounce_pct": r[3],
                    "sample_size": r[4],
                }
                for r in rows
            }
        except Exception as e:
            log.error("Failed to load round stats for %s: %s", symbol, e)
            return {}

    # ═══════════════════════════════════════════════════════════════
    #  STATS / DASHBOARD
    # ═══════════════════════════════════════════════════════════════

    def get_all_stats(self) -> dict:
        """Get level memory stats for dashboard."""
        stats = {}
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            for sym in SYMBOLS:
                row = conn.execute(
                    "SELECT COUNT(*), SUM(bounce_count + break_count) "
                    "FROM price_levels WHERE symbol=?",
                    (sym,)
                ).fetchone()
                level_count = row[0] or 0
                total_touches = row[1] or 0

                round_rows = conn.execute(
                    "SELECT level_type, bounce_pct, sample_size "
                    "FROM round_level_stats WHERE symbol=? AND sample_size >= 5",
                    (sym,)
                ).fetchall()

                round_intel = {}
                for lt, bp, ss in round_rows:
                    round_intel[lt] = {
                        "bounce_pct": round(bp * 100, 1),
                        "sample_size": ss,
                    }

                stats[sym] = {
                    "levels": level_count,
                    "total_touches": total_touches,
                    "round_intel": round_intel,
                }
            conn.close()
        except Exception as e:
            log.error("Failed to get level stats: %s", e)
        return stats

    def prune_stale_levels(self, max_age_days: int = 60, min_touches: int = 2):
        """Remove levels not touched in max_age_days with fewer than min_touches.
        Called periodically to keep DB clean.
        """
        try:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            result = conn.execute(
                "DELETE FROM price_levels WHERE last_touched < ? "
                "AND (bounce_count + break_count) < ?",
                (cutoff, min_touches)
            )
            deleted = result.rowcount
            conn.commit()
            conn.close()

            if deleted > 0:
                self._cache.clear()
                log.info("LEVEL PRUNE: removed %d stale levels (>%dd, <%d touches)",
                         deleted, max_age_days, min_touches)
        except Exception as e:
            log.error("Level prune error: %s", e)
