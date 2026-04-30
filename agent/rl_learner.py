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
            # 2026-04-29: persist raw trade outcomes so weight learning survives restarts.
            # Without this, _trade_outcomes is in-memory only, MIN_TRADES=20 threshold
            # never reached because every restart wipes the accumulator.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    won INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    r_multiple REAL NOT NULL,
                    score REAL,
                    regime TEXT,
                    exit_reason TEXT,
                    components_json TEXT,
                    peak_r REAL DEFAULT 0,
                    giveback_r REAL DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_outcomes_sym_ts ON trade_outcomes(symbol, ts)")
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("RL DB init error: %s", e)

    def _ensure_symbol(self, sym: str):
        """Lazy-init in-memory state for a symbol so we never silently drop persisted data.
        Called by _load_state for any symbol that appears in the DB but isn't in SYMBOLS."""
        if sym not in self._weights:
            self._weights[sym] = dict(DEFAULT_WEIGHTS)
        if sym not in self._trade_outcomes:
            self._trade_outcomes[sym] = []
        if sym not in self._trail_adjustments:
            self._trail_adjustments[sym] = {
                "lock_threshold_mult": 1.0,
                "be_threshold_mult": 1.0,
                "trail_tightness_mult": 1.0,
            }
        if sym not in self._rolling_pf:
            self._rolling_pf[sym] = 1.0
        if sym not in self._reverted:
            self._reverted[sym] = False

    def _load_state(self):
        """Load all persisted RL state from DB on startup.

        Loads every symbol present in the DB regardless of current config.SYMBOLS.
        Bug fixed 2026-05-01: previously trade_outcomes loop iterated `for sym in SYMBOLS`,
        and score_weights had `if sym in self._weights`, so any symbol that had been
        traded historically but was no longer in the active list (e.g. BTCUSD) had its
        learned state silently discarded on restart. Now we lazy-init via _ensure_symbol
        and log any orphans so the operator can see stranded state.
        """
        orphans = set()
        active = set(SYMBOLS) if isinstance(SYMBOLS, dict) else set(SYMBOLS)

        # ── score_weights ──
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            rows = conn.execute("SELECT symbol, component, weight FROM score_weights").fetchall()
            for sym, comp, w in rows:
                self._ensure_symbol(sym)
                if comp in self._weights[sym]:
                    self._weights[sym][comp] = float(w)
                if sym not in active:
                    orphans.add(sym)

            # ── trail_adjustments ──
            rows = conn.execute(
                "SELECT symbol, lock_threshold_mult, be_threshold_mult, trail_tightness_mult "
                "FROM trail_adjustments"
            ).fetchall()
            for sym, lock_m, be_m, trail_m in rows:
                self._ensure_symbol(sym)
                self._trail_adjustments[sym] = {
                    "lock_threshold_mult": float(lock_m),
                    "be_threshold_mult": float(be_m),
                    "trail_tightness_mult": float(trail_m),
                }
                if sym not in active:
                    orphans.add(sym)
            conn.close()

            loaded = sum(1 for sym in self._weights for c, w in self._weights[sym].items() if w != 1.0)
            if loaded:
                log.info("RL Learner loaded %d adjusted weights from DB", loaded)
        except Exception as e:
            log.debug("RL load state error: %s", e)

        # ── trade_outcomes (last 100 per symbol, all symbols in DB) ──
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            db_symbols = [r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM trade_outcomes"
            ).fetchall()]
            for sym in db_symbols:
                rows = conn.execute(
                    "SELECT ts, won, pnl, r_multiple, score, regime, exit_reason, "
                    "components_json, peak_r, giveback_r "
                    "FROM trade_outcomes WHERE symbol=? "
                    "ORDER BY ts DESC LIMIT 100",
                    (sym,)
                ).fetchall()
                if not rows:
                    continue
                self._ensure_symbol(sym)
                outcomes = []
                for ts, won, pnl, rm, score, regime, exit_reason, comps_json, peak_r, gb_r in reversed(rows):
                    components = {}
                    if comps_json:
                        try: components = json.loads(comps_json)
                        except Exception: pass
                    outcomes.append({
                        "won": bool(won), "pnl": float(pnl), "r": float(rm),
                        "score": float(score or 0), "regime": str(regime or ""),
                        "exit_reason": str(exit_reason or ""),
                        "components": components,
                        "peak_r": float(peak_r or 0),
                        "giveback_r": float(gb_r or 0),
                        "ts": float(ts),
                    })
                self._trade_outcomes[sym] = outcomes
                if sym not in active:
                    orphans.add(sym)
            conn.close()
            total = sum(len(v) for v in self._trade_outcomes.values())
            if total > 0:
                with_comps = sum(1 for s in self._trade_outcomes.values() for o in s if o.get("components"))
                log.info("RL Learner loaded %d trade outcomes from DB (%d with components)",
                         total, with_comps)
        except Exception as e:
            log.warning("RL trade_outcomes load failed: %s", e)

        if orphans:
            log.warning("RL Learner: %d orphan symbol(s) in DB not in current SYMBOLS — "
                        "state preserved but inactive: %s", len(orphans), sorted(orphans))

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
                       exit_reason: str, score_components: Optional[Dict[str, float]] = None,
                       peak_r: float = 0.0):
        """Record a trade outcome for learning.

        Args:
            score_components: dict of {component_name: component_score} at entry time.
                              If None, weight learning is skipped for this trade.
            peak_r: peak R-multiple reached before exit (for giveback analysis).
        """
        won = pnl > 0
        giveback_r = peak_r - r_multiple if peak_r > 0 else 0
        ts_now = time.time()

        with self._lock:
            self._ensure_symbol(symbol)
            # Track per-symbol outcomes
            self._trade_outcomes.setdefault(symbol, []).append({
                "won": won, "pnl": pnl, "r": r_multiple,
                "score": score, "regime": regime,
                "exit_reason": exit_reason,
                "components": score_components or {},
                "peak_r": peak_r,
                "giveback_r": giveback_r,
                "ts": ts_now,
            })

        # 2026-04-29 fix: persist outcome to DB so it survives restart.
        # Without this, learning never accumulates the MIN_TRADES=20 threshold.
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=5.0)
            conn.execute(
                "INSERT INTO trade_outcomes (ts, symbol, won, pnl, r_multiple, score, regime, "
                "exit_reason, components_json, peak_r, giveback_r) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts_now, symbol, 1 if won else 0, float(pnl), float(r_multiple),
                 float(score), str(regime or ""), str(exit_reason or ""),
                 json.dumps(score_components) if score_components else None,
                 float(peak_r), float(giveback_r))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("[%s] RL trade_outcomes persist failed: %s", symbol, e)
            # Keep last 100
            if len(self._trade_outcomes[symbol]) > 100:
                self._trade_outcomes[symbol] = self._trade_outcomes[symbol][-100:]

            # Track exit outcomes (keyed by simplified exit reason)
            exit_key = "SL" if "sl" in exit_reason.lower() else exit_reason.split("[")[0].strip()
            key = f"{symbol}_{exit_key}"
            self._exit_outcomes.setdefault(key, []).append(r_multiple)
            if len(self._exit_outcomes[key]) > 50:
                self._exit_outcomes[key] = self._exit_outcomes[key][-50:]

            # Track giveback (peak vs exit) for trail tightness learning
            gb_key = f"_giveback_{symbol}"
            self._exit_outcomes.setdefault(gb_key, []).append(giveback_r)
            if len(self._exit_outcomes[gb_key]) > 50:
                self._exit_outcomes[gb_key] = self._exit_outcomes[gb_key][-50:]

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
            # Strategy failing — revert BOTH weights AND trail adjustments.
            # Bug fixed 2026-05-01: previously only weights reset, leaving stale
            # learned trail multipliers active for re-entry — silent corruption.
            log.warning("[%s] RL REVERT: PF %.2f < %.2f — reverting weights AND trail to defaults",
                        symbol, rolling_pf, PF_REVERT)
            with self._lock:
                self._weights[symbol] = dict(DEFAULT_WEIGHTS)
                self._trail_adjustments[symbol] = {
                    "lock_threshold_mult": 1.0,
                    "be_threshold_mult": 1.0,
                    "trail_tightness_mult": 1.0,
                }
                self._reverted[symbol] = True
            self._audit(symbol, "REVERT", f"PF={rolling_pf:.2f} < {PF_REVERT} (weights+trail reset)")
            self._persist_weights(symbol)
            self._persist_trail(symbol)
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
        """Learn from exit outcomes — adjust trail parameters based on actual giveback."""
        outcomes = self._trade_outcomes.get(symbol, [])
        if len(outcomes) < 10:
            return

        recent = outcomes[-20:]

        # Compute avg giveback (peak profit vs actual exit)
        givebacks = [o.get("giveback_r", 0) for o in recent if o.get("peak_r", 0) > 0.5]
        avg_giveback = float(np.mean(givebacks)) if givebacks else 0

        # Compute avg exit R for winning trades
        win_rs = [o["r"] for o in recent if o["won"]]
        avg_win_r = float(np.mean(win_rs)) if win_rs else 0

        # Compute avg peak R for all trades that had positive peak
        peak_rs = [o.get("peak_r", 0) for o in recent if o.get("peak_r", 0) > 0.3]
        avg_peak = float(np.mean(peak_rs)) if peak_rs else 0

        with self._lock:
            adj = self._trail_adjustments.get(symbol, {
                "lock_threshold_mult": 1.0,
                "be_threshold_mult": 1.0,
                "trail_tightness_mult": 1.0,
            })

            changed = False

            # RULE 1: High giveback (>60% of peak given back) → tighten trail
            # This is the "profit reversal" problem — trades go into profit then lose it
            if avg_giveback > 0.6 * avg_peak and avg_peak > 0.5 and len(givebacks) >= 5:
                old = adj["trail_tightness_mult"]
                adj["trail_tightness_mult"] = max(0.6, old - 0.05)
                adj["lock_threshold_mult"] = max(0.7, adj["lock_threshold_mult"] - 0.03)
                if adj["trail_tightness_mult"] != old:
                    changed = True
                    log.info("[%s] RL EXIT: high giveback %.2fR (peak=%.2fR) → TIGHTEN trail (x%.2f)",
                             symbol, avg_giveback, avg_peak, adj["trail_tightness_mult"])

            # RULE 2: Low giveback (<20% of peak) and good win R → trail is optimal, keep
            elif avg_giveback < 0.2 * avg_peak and avg_win_r > 1.0 and len(givebacks) >= 5:
                # Trail is working well — nudge toward default
                if adj["trail_tightness_mult"] < 0.95:
                    adj["trail_tightness_mult"] = min(1.0, adj["trail_tightness_mult"] + 0.02)
                    changed = True
                    log.info("[%s] RL EXIT: low giveback, good wins (%.2fR) → relax slightly (x%.2f)",
                             symbol, avg_win_r, adj["trail_tightness_mult"])

            # RULE 3: Lots of tiny wins (avg win < 0.3R) → trail too tight, cutting winners
            elif avg_win_r < 0.3 and avg_win_r > 0 and len(win_rs) >= 5:
                old = adj["trail_tightness_mult"]
                adj["trail_tightness_mult"] = min(1.5, old + 0.05)
                if adj["trail_tightness_mult"] != old:
                    changed = True
                    log.info("[%s] RL EXIT: tiny wins avg %.2fR → LOOSEN trail (x%.2f)",
                             symbol, avg_win_r, adj["trail_tightness_mult"])

            # RULE 4: Big wins (avg > 3R) → excellent, tighten to capture even more
            elif avg_win_r > 3.0 and len(win_rs) >= 5:
                old = adj["trail_tightness_mult"]
                adj["trail_tightness_mult"] = max(0.7, old - 0.03)
                if adj["trail_tightness_mult"] != old:
                    changed = True
                    log.info("[%s] RL EXIT: big wins %.2fR → tighten to lock more (x%.2f)",
                             symbol, avg_win_r, adj["trail_tightness_mult"])

            if changed:
                self._trail_adjustments[symbol] = adj
                self._persist_trail(symbol)
                self._audit(symbol, "EXIT_UPDATE",
                            f"giveback={avg_giveback:.2f}R peak={avg_peak:.2f}R "
                            f"win_r={avg_win_r:.2f}R n={len(recent)} "
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
            # Surface every symbol with state (active or orphaned) so operators
            # can see stranded data after a SYMBOLS list change.
            active = set(SYMBOLS) if isinstance(SYMBOLS, dict) else set(SYMBOLS)
            all_syms = set(self._weights) | set(self._trail_adjustments) | set(self._trade_outcomes)
            for sym in all_syms:
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
                        "active": sym in active,
                    }
            return status
