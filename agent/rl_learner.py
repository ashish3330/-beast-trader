"""
Dragon Trader — Reinforcement Learner.
Learns scoring weights and exit rules from actual trade outcomes.

Safety:
- Weight changes capped at ±5% per update cycle
- Hard bounds: 0.3x - 2.5x of original weight (never goes crazy)
- Minimum 20 trades before any adjustment
- Auto-revert if rolling PF drops below 0.8
- All changes logged + persisted to SQLite for audit

Scoring weight learning:
- After each closed trade, records which scoring components were active
- Computes per-component win rate
- Components that predict winners get higher weight
- Components that predict losers get lower weight

Exit rule learning:
- Tracks R-multiple at exit for each exit reason
- If SL exits avg -1R but trailing exits avg +3R, learns to hold longer
- Adjusts trail lock/BE thresholds based on actual outcomes
"""
import time
import logging
import sqlite3
import threading
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS, DB_PATH

log = logging.getLogger("dragon.rl")

RL_DB = DB_PATH.parent / "rl_learner.db"

# The 11 scoring components (from momentum_scorer.py)
SCORE_COMPONENTS = [
    "ema_stack",       # EMA alignment (max 1.5)
    "supertrend",      # SuperTrend direction (max 1.5)
    "macd_signal",     # MACD line vs signal (max 1.5)
    "macd_hist",       # MACD histogram (max 1.0)
    "rsi",             # RSI sweet zone (max 1.0)
    "candle_pattern",  # Engulfing/pin bar (max 2.0)
    "heikin_ashi",     # HA color/body (max 1.0)
    "structure",       # HH/HL structure + ADX (max 1.5)
    "breakout",        # Donchian/BB breakout (max 2.5)
    "momentum_vel",    # ROC velocity (max 0.5)
    "trend_persist",   # Consecutive bars (max 0.5)
]

# Default weights (1.0 = no change from original scoring)
DEFAULT_WEIGHTS = {c: 1.0 for c in SCORE_COMPONENTS}

# Safety bounds
WEIGHT_MIN = 0.3    # never reduce a component below 30%
WEIGHT_MAX = 2.5    # never boost above 250%
MAX_CHANGE = 0.05   # max ±5% change per update
MIN_TRADES = 20     # need 20+ trades before adjusting
PF_REVERT = 0.8     # revert to defaults if PF drops below this


class RLLearner:
    """Reinforcement learner for scoring weights and exit rules."""

    def __init__(self, state):
        self.state = state
        self._lock = threading.Lock()

        # Current learned weights per symbol
        self._weights: Dict[str, Dict[str, float]] = {}
        for sym in SYMBOLS:
            self._weights[sym] = dict(DEFAULT_WEIGHTS)

        # Trade outcome tracking
        self._trade_outcomes: Dict[str, list] = {sym: [] for sym in SYMBOLS}

        # Exit rule tracking
        self._exit_outcomes: Dict[str, list] = {}  # exit_reason -> [r_multiples]

        # Trail parameter adjustments per symbol
        self._trail_adjustments: Dict[str, Dict[str, float]] = {}
        for sym in SYMBOLS:
            self._trail_adjustments[sym] = {
                "lock_threshold_mult": 1.0,   # multiply lock R-threshold
                "be_threshold_mult": 1.0,     # multiply BE R-threshold
                "trail_tightness_mult": 1.0,  # multiply trail ATR distance
            }

        # Performance tracking
        self._rolling_pf: Dict[str, float] = {sym: 1.0 for sym in SYMBOLS}
        self._reverted: Dict[str, bool] = {sym: False for sym in SYMBOLS}

        # Init DB
        self._init_db()
        self._load_state()

    def _init_db(self):
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS score_weights (
                    symbol TEXT NOT NULL,
                    component TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    win_count INTEGER DEFAULT 0,
                    loss_count INTEGER DEFAULT 0,
                    avg_r_win REAL DEFAULT 0,
                    avg_r_loss REAL DEFAULT 0,
                    updated TEXT,
                    PRIMARY KEY (symbol, component)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exit_learning (
                    symbol TEXT NOT NULL,
                    exit_reason TEXT NOT NULL,
                    count INTEGER DEFAULT 0,
                    avg_r REAL DEFAULT 0,
                    best_r REAL DEFAULT 0,
                    worst_r REAL DEFAULT 0,
                    updated TEXT,
                    PRIMARY KEY (symbol, exit_reason)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trail_adjustments (
                    symbol TEXT NOT NULL PRIMARY KEY,
                    lock_threshold_mult REAL DEFAULT 1.0,
                    be_threshold_mult REAL DEFAULT 1.0,
                    trail_tightness_mult REAL DEFAULT 1.0,
                    updated TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rl_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    action TEXT,
                    detail TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("RL DB init error: %s", e)

    def _load_state(self):
        """Load learned weights from DB on startup."""
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            rows = conn.execute("SELECT symbol, component, weight FROM score_weights").fetchall()
            for sym, comp, w in rows:
                if sym in self._weights and comp in self._weights.get(sym, {}):
                    self._weights[sym][comp] = float(w)

            rows = conn.execute(
                "SELECT symbol, lock_threshold_mult, be_threshold_mult, trail_tightness_mult "
                "FROM trail_adjustments"
            ).fetchall()
            for sym, lock_m, be_m, trail_m in rows:
                if sym in self._trail_adjustments:
                    self._trail_adjustments[sym] = {
                        "lock_threshold_mult": float(lock_m),
                        "be_threshold_mult": float(be_m),
                        "trail_tightness_mult": float(trail_m),
                    }
            conn.close()

            loaded = sum(1 for sym in self._weights for c, w in self._weights[sym].items() if w != 1.0)
            if loaded:
                log.info("RL Learner loaded %d adjusted weights from DB", loaded)
        except Exception as e:
            log.debug("RL load state error: %s", e)

    # ═══════════════════════════════════════════════════════════════
    #  SCORING WEIGHT API (called by brain)
    # ═══════════════════════════════════════════════════════════════

    def should_skip_entry(self, symbol: str, regime: str, hour: int,
                          direction: int) -> tuple:
        """Check if RL has learned this setup is a loser. Uses trade journal DB.
        Returns (skip: bool, reason: str).
        Persists across restarts — reads from actual trade history."""
        from config import RL_ENABLED_SYMBOLS, RL_SYMBOL_PARAMS
        if symbol not in RL_ENABLED_SYMBOLS:
            return False, ""

        try:
            params = RL_SYMBOL_PARAMS.get(symbol, {})
            lookback = params.get("lookback", 20)

            # Use in-memory outcomes first (faster), fall back to DB
            outcomes = self._trade_outcomes.get(symbol, [])

            # If not enough in memory, load from trade journal DB
            if len(outcomes) < lookback:
                try:
                    from config import DB_PATH
                    journal_db = DB_PATH.parent / "trade_journal.db"
                    conn = sqlite3.connect(str(journal_db), timeout=5.0)
                    rows = conn.execute(
                        "SELECT pnl, direction, session_hour, exit_reason "
                        "FROM trades WHERE symbol=? ORDER BY id DESC LIMIT ?",
                        (symbol, lookback * 2)
                    ).fetchall()
                    conn.close()
                    if rows:
                        outcomes = [{"pnl": r[0], "won": r[0] > 0,
                                     "dir": 1 if r[1] == "LONG" else -1,
                                     "hour": r[2] or 12,
                                     "regime": "unknown"} for r in rows]
                except Exception:
                    pass

            if len(outcomes) < lookback:
                return False, ""

            recent = outcomes[-lookback:]

            # Regime WR (from in-memory only — regime not stored in journal)
            regime_trades = [t for t in recent if t.get("regime") == regime]
            if len(regime_trades) >= 5:
                r_wr = sum(1 for t in regime_trades if t["won"]) / len(regime_trades)
                if r_wr < 0.15 and len(regime_trades) >= 8:
                    return True, f"RL: {regime} WR={r_wr:.0%} ({len(regime_trades)} trades)"

            # Hour WR — require more evidence before blocking
            hour_trades = [t for t in recent if t.get("hour") == hour]
            if len(hour_trades) >= 8:
                h_wr = sum(1 for t in hour_trades if t["won"]) / len(hour_trades)
                if h_wr < 0.15:
                    return True, f"RL: hour {hour} WR={h_wr:.0%} ({len(hour_trades)} trades)"

            # Regime+direction combo — require more evidence
            rd_trades = [t for t in recent if t.get("regime") == regime and t.get("dir") == direction]
            if len(rd_trades) >= 8:
                rd_wr = sum(1 for t in rd_trades if t["won"]) / len(rd_trades)
                if rd_wr < 0.15:
                    return True, f"RL: {regime}+{'L' if direction==1 else 'S'} WR={rd_wr:.0%}"

            return False, ""
        except Exception:
            return False, ""

    def get_risk_multiplier(self, symbol: str, regime: str, hour: int) -> float:
        """Get RL-learned risk multiplier based on regime/hour performance.
        Returns 0.6-1.4 (from config per-symbol params)."""
        from config import RL_ENABLED_SYMBOLS, RL_SYMBOL_PARAMS
        if symbol not in RL_ENABLED_SYMBOLS:
            return 1.0

        params = RL_SYMBOL_PARAMS.get(symbol, {})
        boost_max = params.get("boost_max", 1.4)
        outcomes = self._trade_outcomes.get(symbol, [])
        if len(outcomes) < params.get("lookback", 20):
            return 1.0

        recent = outcomes[-params.get("lookback", 20):]
        mult = 1.0

        # Regime WR
        regime_trades = [t for t in recent if t.get("regime") == regime]
        if len(regime_trades) >= 3:
            r_wr = sum(1 for t in regime_trades if t["won"]) / len(regime_trades)
            if r_wr > 0.55:
                mult *= 1.0 + (r_wr - 0.5) * 1.0
            elif r_wr < 0.40:
                mult *= max(0.5, 1.0 - (0.5 - r_wr) * 1.0)

        # Hour WR
        hour_trades = [t for t in recent if t.get("hour") == hour]
        if len(hour_trades) >= 3:
            h_wr = sum(1 for t in hour_trades if t["won"]) / len(hour_trades)
            if h_wr > 0.55:
                mult *= 1.0 + (h_wr - 0.5) * 0.8
            elif h_wr < 0.40:
                mult *= max(0.6, 1.0 - (0.5 - h_wr) * 0.6)

        # Rolling PF
        gp = sum(t["pnl"] for t in recent if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in recent if t["pnl"] < 0) or 0.01
        rpf = gp / gl
        if rpf < 0.7:
            mult *= 0.5
        elif rpf > 2.5:
            mult *= 1.2

        return max(0.6, min(boost_max, mult))

    def get_score_weights(self, symbol: str) -> Dict[str, float]:
        """Get current learned weights for a symbol's scoring components.
        Brain multiplies each component score by these weights."""
        with self._lock:
            return dict(self._weights.get(symbol, DEFAULT_WEIGHTS))

    def get_trail_adjustments(self, symbol: str) -> Dict[str, float]:
        """Get trail parameter adjustments for executor."""
        with self._lock:
            return dict(self._trail_adjustments.get(symbol, {
                "lock_threshold_mult": 1.0,
                "be_threshold_mult": 1.0,
                "trail_tightness_mult": 1.0,
            }))

    # ═══════════════════════════════════════════════════════════════
    #  RECORD TRADE OUTCOME (called after each closed trade)
    # ═══════════════════════════════════════════════════════════════

    def record_outcome(self, symbol: str, direction: str, pnl: float,
                       r_multiple: float, score: float, regime: str,
                       exit_reason: str, score_components: Optional[Dict[str, float]] = None):
        """Record a trade outcome for learning.

        Args:
            score_components: dict of {component_name: component_score} at entry time.
                              If None, learning is skipped for this trade.
        """
        won = pnl > 0

        with self._lock:
            # Track per-symbol outcomes
            self._trade_outcomes.setdefault(symbol, []).append({
                "won": won, "pnl": pnl, "r": r_multiple,
                "score": score, "regime": regime,
                "exit_reason": exit_reason,
                "components": score_components or {},
                "ts": time.time(),
            })
            # Keep last 100
            if len(self._trade_outcomes[symbol]) > 100:
                self._trade_outcomes[symbol] = self._trade_outcomes[symbol][-100:]

            # Track exit outcomes
            key = f"{symbol}_{exit_reason}"
            self._exit_outcomes.setdefault(key, []).append(r_multiple)
            if len(self._exit_outcomes[key]) > 50:
                self._exit_outcomes[key] = self._exit_outcomes[key][-50:]

        # Try to learn (needs enough data)
        self._maybe_update_weights(symbol)
        self._maybe_update_exits(symbol)

    # ═══════════════════════════════════════════════════════════════
    #  WEIGHT LEARNING
    # ═══════════════════════════════════════════════════════════════

    def _maybe_update_weights(self, symbol: str):
        """Update scoring weights based on accumulated outcomes."""
        outcomes = self._trade_outcomes.get(symbol, [])
        if len(outcomes) < MIN_TRADES:
            return

        # Check rolling PF first — revert if strategy is failing
        recent = outcomes[-MIN_TRADES:]
        gross_p = sum(t["pnl"] for t in recent if t["pnl"] > 0)
        gross_l = sum(abs(t["pnl"]) for t in recent if t["pnl"] < 0) or 0.01
        rolling_pf = gross_p / gross_l
        self._rolling_pf[symbol] = rolling_pf

        if rolling_pf < PF_REVERT and not self._reverted.get(symbol, False):
            # Strategy failing — revert to defaults
            log.warning("[%s] RL REVERT: PF %.2f < %.2f — reverting weights to defaults",
                        symbol, rolling_pf, PF_REVERT)
            with self._lock:
                self._weights[symbol] = dict(DEFAULT_WEIGHTS)
                self._reverted[symbol] = True
            self._audit(symbol, "REVERT", f"PF={rolling_pf:.2f} < {PF_REVERT}")
            self._persist_weights(symbol)
            return

        if rolling_pf >= 1.0:
            self._reverted[symbol] = False  # reset revert flag when profitable

        # Compute per-component win rate from recent trades that have component data
        trades_with_components = [t for t in recent if t.get("components")]
        if len(trades_with_components) < 10:
            return  # not enough component data

        component_stats = {}
        for comp in SCORE_COMPONENTS:
            wins = 0
            losses = 0
            avg_r_win = []
            avg_r_loss = []
            for t in trades_with_components:
                comp_val = t["components"].get(comp, 0)
                if comp_val > 0:  # component was active (contributed to score)
                    if t["won"]:
                        wins += 1
                        avg_r_win.append(t["r"])
                    else:
                        losses += 1
                        avg_r_loss.append(t["r"])
            total = wins + losses
            if total >= 5:  # need 5+ trades where component was active
                component_stats[comp] = {
                    "wr": wins / total,
                    "wins": wins, "losses": losses,
                    "avg_r_win": float(np.mean(avg_r_win)) if avg_r_win else 0,
                    "avg_r_loss": float(np.mean(avg_r_loss)) if avg_r_loss else 0,
                }

        if not component_stats:
            return

        # Adjust weights based on component performance
        changes = {}
        with self._lock:
            for comp, stats in component_stats.items():
                old_w = self._weights[symbol].get(comp, 1.0)
                wr = stats["wr"]

                # Target: boost components that predict winners, reduce losers
                if wr > 0.55:
                    # Component predicts winners — small boost
                    delta = min(MAX_CHANGE, (wr - 0.5) * 0.2)
                elif wr < 0.40:
                    # Component predicts losers — small reduction
                    delta = max(-MAX_CHANGE, (wr - 0.5) * 0.2)
                else:
                    delta = 0  # neutral

                if delta != 0:
                    new_w = old_w + delta
                    new_w = max(WEIGHT_MIN, min(WEIGHT_MAX, new_w))
                    if new_w != old_w:
                        self._weights[symbol][comp] = new_w
                        changes[comp] = {"old": old_w, "new": new_w, "wr": wr,
                                         "n": stats["wins"] + stats["losses"]}

        if changes:
            detail = "; ".join(f"{c}: {v['old']:.2f}→{v['new']:.2f} (WR={v['wr']:.0%}, n={v['n']})"
                               for c, v in changes.items())
            log.info("[%s] RL WEIGHT UPDATE (PF=%.2f): %s", symbol, rolling_pf, detail)
            self._audit(symbol, "WEIGHT_UPDATE", detail)
            self._persist_weights(symbol)

    # ═══════════════════════════════════════════════════════════════
    #  EXIT RULE LEARNING
    # ═══════════════════════════════════════════════════════════════

    def _maybe_update_exits(self, symbol: str):
        """Learn from exit outcomes — adjust trail parameters."""
        # Compare SL exits vs trailing exits
        sl_key = f"{symbol}_SL"
        trail_key = f"{symbol}_trailing"

        sl_outcomes = self._exit_outcomes.get(sl_key, [])
        trail_outcomes = self._exit_outcomes.get(trail_key, [])

        if len(sl_outcomes) < 5 or len(trail_outcomes) < 5:
            return

        avg_sl_r = float(np.mean(sl_outcomes[-20:]))
        avg_trail_r = float(np.mean(trail_outcomes[-20:]))

        with self._lock:
            adj = self._trail_adjustments.get(symbol, {
                "lock_threshold_mult": 1.0,
                "be_threshold_mult": 1.0,
                "trail_tightness_mult": 1.0,
            })

            changed = False

            # If SL exits are very negative (avg < -0.8R) but trailing exits are positive,
            # the trail is working but SL is too tight — widen slightly
            if avg_sl_r < -0.8 and avg_trail_r > 1.0:
                # Trailing is profitable but SL keeps hitting — let trades breathe more
                old = adj["be_threshold_mult"]
                adj["be_threshold_mult"] = min(1.5, old + 0.05)
                adj["lock_threshold_mult"] = min(1.5, adj["lock_threshold_mult"] + 0.03)
                if adj["be_threshold_mult"] != old:
                    changed = True
                    log.info("[%s] RL EXIT: SL avg %.2fR, trail avg %.2fR → widen BE/lock thresholds",
                             symbol, avg_sl_r, avg_trail_r)

            # If trailing exits are barely profitable (avg < 0.5R), trail is too tight
            elif avg_trail_r < 0.5 and avg_trail_r > 0:
                old = adj["trail_tightness_mult"]
                adj["trail_tightness_mult"] = min(1.5, old + 0.05)
                if adj["trail_tightness_mult"] != old:
                    changed = True
                    log.info("[%s] RL EXIT: trail avg only %.2fR → loosen trail (x%.2f)",
                             symbol, avg_trail_r, adj["trail_tightness_mult"])

            # If trailing exits are very profitable (avg > 3R), trail is working great
            elif avg_trail_r > 3.0:
                old = adj["trail_tightness_mult"]
                adj["trail_tightness_mult"] = max(0.7, old - 0.03)
                if adj["trail_tightness_mult"] != old:
                    changed = True
                    log.info("[%s] RL EXIT: trail avg %.2fR — excellent, tighten slightly (x%.2f)",
                             symbol, avg_trail_r, adj["trail_tightness_mult"])

            if changed:
                self._trail_adjustments[symbol] = adj
                self._persist_trail(symbol)
                self._audit(symbol, "EXIT_UPDATE",
                            f"SL_avg={avg_sl_r:.2f}R trail_avg={avg_trail_r:.2f}R "
                            f"lock_m={adj['lock_threshold_mult']:.2f} "
                            f"be_m={adj['be_threshold_mult']:.2f} "
                            f"trail_m={adj['trail_tightness_mult']:.2f}")

    # ═══════════════════════════════════════════════════════════════
    #  PERSISTENCE
    # ═══════════════════════════════════════════════════════════════

    def _persist_weights(self, symbol: str):
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            ts = datetime.now(timezone.utc).isoformat()
            for comp, w in self._weights.get(symbol, {}).items():
                conn.execute("""
                    INSERT OR REPLACE INTO score_weights (symbol, component, weight, updated)
                    VALUES (?, ?, ?, ?)
                """, (symbol, comp, w, ts))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("RL persist weights error: %s", e)

    def _persist_trail(self, symbol: str):
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            ts = datetime.now(timezone.utc).isoformat()
            adj = self._trail_adjustments.get(symbol, {})
            conn.execute("""
                INSERT OR REPLACE INTO trail_adjustments
                (symbol, lock_threshold_mult, be_threshold_mult, trail_tightness_mult, updated)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, adj.get("lock_threshold_mult", 1.0),
                  adj.get("be_threshold_mult", 1.0),
                  adj.get("trail_tightness_mult", 1.0), ts))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("RL persist trail error: %s", e)

    def _audit(self, symbol: str, action: str, detail: str):
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            conn.execute("""
                INSERT INTO rl_audit_log (timestamp, symbol, action, detail)
                VALUES (?, ?, ?, ?)
            """, (datetime.now(timezone.utc).isoformat(), symbol, action, detail))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    #  STATUS (for dashboard)
    # ═══════════════════════════════════════════════════════════════

    def get_status(self) -> dict:
        with self._lock:
            status = {}
            for sym in SYMBOLS:
                weights = self._weights.get(sym, {})
                adjusted = {c: w for c, w in weights.items() if w != 1.0}
                trail = self._trail_adjustments.get(sym, {})
                trail_adjusted = {k: v for k, v in trail.items() if v != 1.0}
                n_trades = len(self._trade_outcomes.get(sym, []))
                if adjusted or trail_adjusted or n_trades > 0:
                    status[sym] = {
                        "weight_adjustments": adjusted,
                        "trail_adjustments": trail_adjusted,
                        "rolling_pf": round(self._rolling_pf.get(sym, 1.0), 2),
                        "trades_tracked": n_trades,
                        "reverted": self._reverted.get(sym, False),
                    }
            return status
