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
# 2026-05-12: lowered 20→12 so RL adapts to current regime within ~12 trades
# instead of ~20. Safe because (a) MAX_CHANGE still caps per-update at 5%,
# (b) component_stats still needs 5+ per-component trades to be considered,
# (c) PF_REVERT auto-reverts if rolling PF drops below 0.7.
MIN_TRADES = 12
PF_REVERT = 0.8     # revert to defaults if PF drops below this
# 2026-05-12: throttle weight updates — observed 150 events/7d on US2000.r
# (WEIGHT_UPDATE→REVERT→WEIGHT_UPDATE flapping). Cap to one update per
# symbol per 4 hours to let the new weights actually take effect before
# re-tuning. Revert is exempt (always allowed for safety).
UPDATE_THROTTLE_SEC = 4 * 3600


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

        # Equity high-water mark (DD-aware risk scaling)
        self._peak_equity: float = 0.0
        self._peak_ts: float = 0.0

        # Per-regime cache: {symbol: {regime: {component: weight}}}
        # Sparse — only filled when a (symbol, regime, component) cell has
        # enough samples; otherwise get_weights falls back to global.
        self._regime_weights: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._regime_trail: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Health telemetry — number of weight updates this run, last update ts
        self._n_updates: int = 0
        self._n_regime_updates: int = 0
        self._last_health_log_ts: float = 0.0
        # Per-symbol throttle clocks for WEIGHT/EXIT updates — stops flapping
        self._last_update_ts: Dict[str, float] = {}

        # Init DB
        self._init_db()
        self._load_state()
        self._load_peak_equity()
        self._load_regime_state()

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
            # 2026-05-12: equity high-water mark — DD-aware risk scaling.
            # Survives restart so the protect curve doesn't reset to zero DD
            # every time the trader bounces.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_peak (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    peak_equity REAL NOT NULL,
                    peak_ts REAL NOT NULL,
                    updated TEXT
                )
            """)
            # 2026-05-12: per-regime component weights — same symbol behaves
            # differently in trending vs ranging vs volatile vs low_vol.
            # Sparse by design: 5+ component samples within regime to override
            # the global weight; falls back to global otherwise.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_weights (
                    symbol TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    component TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    win_count INTEGER DEFAULT 0,
                    loss_count INTEGER DEFAULT 0,
                    avg_r_win REAL DEFAULT 0,
                    avg_r_loss REAL DEFAULT 0,
                    updated TEXT,
                    PRIMARY KEY (symbol, regime, component)
                )
            """)
            # Per-regime trail adjustments — same idea, regime-conditional
            # trail tightness.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_trail_adjustments (
                    symbol TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    lock_threshold_mult REAL DEFAULT 1.0,
                    be_threshold_mult REAL DEFAULT 1.0,
                    trail_tightness_mult REAL DEFAULT 1.0,
                    updated TEXT,
                    PRIMARY KEY (symbol, regime)
                )
            """)
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

        2026-05-06: also seeds component weights from the auto_dict produced by
        scripts/tune_component_weights.py if the DB row is at default (1.0). Live
        RL fine-tunes from there. DB values always win over the seed — once RL
        has moved a weight, the seed is no longer authoritative.
        """
        orphans = set()
        active = set(SYMBOLS) if isinstance(SYMBOLS, dict) else set(SYMBOLS)

        # ── seed default component weights from tuned auto_dict (if present) ──
        seed_weights: dict = {}
        try:
            from pathlib import Path as _P
            import importlib.util as _iu
            seed_path = _P(__file__).resolve().parent.parent / "backtest" / "results" / "component_weights_auto_dict.py"
            if seed_path.exists():
                spec = _iu.spec_from_file_location("cw_auto", seed_path)
                mod = _iu.module_from_spec(spec); spec.loader.exec_module(mod)
                seed_weights = getattr(mod, "COMPONENT_WEIGHTS_AUTO", {}) or {}
                if seed_weights:
                    log.info("[RL] seeding component weights from auto_dict: %d symbols",
                             len(seed_weights))
        except Exception as e:
            log.warning("[RL] component_weights_auto_dict load failed: %s", e)

        # ── score_weights ──
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            rows = conn.execute("SELECT symbol, component, weight FROM score_weights").fetchall()
            db_weights = {(s, c): float(w) for s, c, w in rows}

            # Apply seed first to all known symbols
            for sym, comps in seed_weights.items():
                self._ensure_symbol(sym)
                for comp, w in comps.items():
                    if comp in self._weights[sym]:
                        self._weights[sym][comp] = float(w)
                if sym not in active:
                    orphans.add(sym)

            # Then DB rows override the seed (so live RL adjustments stick).
            # Skip rows where DB still holds the default 1.0 — let the seed be authoritative.
            for (sym, comp), w in db_weights.items():
                if abs(w - 1.0) < 1e-6 and sym in seed_weights and comp in seed_weights.get(sym, {}):
                    continue   # seed wins
                self._ensure_symbol(sym)
                if comp in self._weights[sym]:
                    self._weights[sym][comp] = w
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

    def get_trail_adjustments(self, symbol: str, regime: Optional[str] = None) -> Dict[str, float]:
        """Get trail parameter adjustments for executor.

        If regime is provided AND we have learned a regime-specific trail
        for (symbol, regime), return that — else fall back to the symbol-level
        learned trail (which itself falls back to defaults if untouched).
        """
        with self._lock:
            base = dict(self._trail_adjustments.get(symbol, {
                "lock_threshold_mult": 1.0,
                "be_threshold_mult": 1.0,
                "trail_tightness_mult": 1.0,
            }))
            if regime:
                reg_overlay = self._regime_trail.get(symbol, {}).get(regime)
                if reg_overlay:
                    base.update(reg_overlay)
            return base

    def get_weights(self, symbol: str, regime: Optional[str] = None) -> Dict[str, float]:
        """Get per-component learned weight multipliers for the scorer.

        Layered lookup:
          1. Start from symbol-level global weights (well-sampled).
          2. If regime provided, overlay any regime-specific weights that
             have been learned for (symbol, regime, component).
        Falls back gracefully — never returns less than the global view.
        Returns empty dict for unknown symbol (scorer treats as default 1.0).
        """
        with self._lock:
            base = dict(self._weights.get(symbol, {}))
            if regime:
                reg_cells = self._regime_weights.get(symbol, {}).get(regime, {})
                if reg_cells:
                    # Regime cells override per-component on a cell-by-cell basis
                    base.update(reg_cells)
            return base

    # ═══════════════════════════════════════════════════════════════
    #  INTELLIGENT RISK CONTROLS (2026-05-12)
    #  Drawdown-aware sizing + streak damper + per-symbol edge score.
    #  These run AS A MULTIPLIER on top of the existing regime/hour
    #  risk multiplier — never override, only protect.
    # ═══════════════════════════════════════════════════════════════

    def _load_regime_state(self):
        """Restore per-regime weights + trail from DB on startup."""
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=5.0)
            for sym, reg, comp, w in conn.execute(
                "SELECT symbol, regime, component, weight FROM regime_weights"
            ).fetchall():
                self._regime_weights.setdefault(sym, {}).setdefault(reg, {})[comp] = float(w)
            n_cells = sum(len(c) for s in self._regime_weights.values() for c in s.values())
            if n_cells:
                log.info("[RL] regime weights loaded: %d cells across %d syms",
                         n_cells, len(self._regime_weights))
            for sym, reg, l_m, be_m, t_m in conn.execute(
                "SELECT symbol, regime, lock_threshold_mult, be_threshold_mult, "
                "trail_tightness_mult FROM regime_trail_adjustments"
            ).fetchall():
                self._regime_trail.setdefault(sym, {})[reg] = {
                    "lock_threshold_mult": float(l_m),
                    "be_threshold_mult": float(be_m),
                    "trail_tightness_mult": float(t_m),
                }
            conn.close()
        except Exception as e:
            log.debug("RL regime state load failed: %s", e)

    def _load_peak_equity(self):
        """Restore peak-equity from DB so DD scaling survives restarts."""
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=5.0)
            row = conn.execute(
                "SELECT peak_equity, peak_ts FROM equity_peak WHERE id=1"
            ).fetchone()
            conn.close()
            if row:
                self._peak_equity = float(row[0] or 0)
                self._peak_ts = float(row[1] or 0)
                log.info("[RL] equity peak restored: $%.2f", self._peak_equity)
        except Exception as e:
            log.debug("RL peak_equity load failed: %s", e)

    def _persist_peak_equity(self):
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=5.0)
            conn.execute("""
                INSERT INTO equity_peak (id, peak_equity, peak_ts, updated)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    peak_equity = excluded.peak_equity,
                    peak_ts = excluded.peak_ts,
                    updated = excluded.updated
            """, (self._peak_equity, self._peak_ts,
                  datetime.now(timezone.utc).isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("RL peak_equity persist failed: %s", e)

    def get_equity_dd_multiplier(self, current_equity: float) -> float:
        """Scale risk down as equity drops from peak. Protects the account
        when streaks turn bad — the single most important survival lever.

        Updates peak on the fly. Returns multiplier in [0.15, 1.0]:
          DD <  3%:  1.00  (normal)
          DD <  6%:  0.75  (caution — wait for recovery)
          DD < 10%:  0.50  (defensive — preserve capital)
          DD < 15%:  0.30  (survival mode)
          DD >= 15%: 0.15  (effectively halted)
        """
        try:
            ce = float(current_equity or 0)
        except Exception:
            return 1.0
        if ce <= 0:
            return 1.0
        with self._lock:
            if ce > self._peak_equity:
                self._peak_equity = ce
                self._peak_ts = time.time()
                self._persist_peak_equity()
            if self._peak_equity <= 0:
                return 1.0
            dd = (self._peak_equity - ce) / self._peak_equity
        # Step function — sharp protection at meaningful DDs.
        if dd < 0.03:  return 1.0
        if dd < 0.06:  return 0.75
        if dd < 0.10:  return 0.50
        if dd < 0.15:  return 0.30
        return 0.15

    def get_streak_multiplier(self, symbol: str) -> float:
        """Damp risk after consecutive losses on this symbol.

        Faster-adapting than rolling PF — captures hot/cold streaks within
        3 trades vs PF needing 12+. Returns [0.5, 1.0]:
          last 3 = 3 losses → 0.50
          last 3 = 2 losses → 0.75
          else              → 1.00
        """
        outcomes = self._trade_outcomes.get(symbol, [])
        if len(outcomes) < 3:
            return 1.0
        last3 = outcomes[-3:]
        losses = sum(1 for o in last3 if not o.get("won", False))
        if losses >= 3:
            return 0.50
        if losses == 2:
            return 0.75
        return 1.0

    def get_quality_threshold_bonus(self, symbol: str) -> int:
        """Auto-tighten entry quality threshold for symbols that are bleeding.

        Uses TWO signals (take the max):
          A. Recent rolling PF (last 10 trades — fast adapt):
             PF >= 1.0  →  0pp
             PF 0.7-1.0 → +5pp
             PF < 0.7   → +10pp
          B. Session-bleed override (last 5 trades):
             4+ losses out of 5 → +15pp (force ultra-picky)
             3 losses out of 5  → +10pp

        Why 10-trade window (was 30): older wins were diluting today's
        bleed. DJ30.r had rolling PF 3.82 from history but lost 8 trades
        in a row today — old system gave it 0pp, new system reacts
        within 5-10 trades.
        """
        outcomes = self._trade_outcomes.get(symbol, [])
        if len(outcomes) < 5:
            return 0

        bonus = 0

        # Signal A: rolling PF on last 10 trades
        if len(outcomes) >= 8:
            recent10 = outcomes[-10:]
            gp = sum(o["pnl"] for o in recent10 if o["pnl"] > 0)
            gl = sum(abs(o["pnl"]) for o in recent10 if o["pnl"] < 0) or 0.01
            pf10 = gp / gl
            if pf10 < 0.7:
                bonus = max(bonus, 10)
            elif pf10 < 1.0:
                bonus = max(bonus, 5)

        # Signal B: session-bleed override on last 5 trades
        last5 = outcomes[-5:]
        losses5 = sum(1 for o in last5 if not o.get("won"))
        if losses5 >= 4:
            bonus = max(bonus, 15)
        elif losses5 >= 3:
            bonus = max(bonus, 10)

        return bonus

    def get_edge_score(self, symbol: str) -> float:
        """Composite per-symbol edge score for capital allocation.

        Uses TIME-WEIGHTED blend so recent reality dominates:
          - last 10 trades  weighted 0.7
          - last 30 trades  weighted 0.3
        That way today's bleed shows up immediately (not diluted by
        last month's wins).

        Returns [0, 1] where 1.0 = highest conviction this symbol earns.
        """
        outcomes = self._trade_outcomes.get(symbol, [])
        n = len(outcomes)
        if n < 8:
            return 0.5  # neutral until enough data

        def _pf_wr(window):
            gp = sum(o["pnl"] for o in window if o["pnl"] > 0)
            gl = sum(abs(o["pnl"]) for o in window if o["pnl"] < 0) or 0.01
            pf = gp / gl
            wr = sum(1 for o in window if o.get("won")) / max(1, len(window))
            return pf, wr

        recent10 = outcomes[-10:]
        recent30 = outcomes[-30:] if n >= 30 else outcomes
        pf10, wr10 = _pf_wr(recent10)
        pf30, wr30 = _pf_wr(recent30)
        # Time-weighted PF + WR (recent dominates)
        pf = 0.7 * pf10 + 0.3 * pf30
        wr = 0.7 * wr10 + 0.3 * wr30
        conf = min(1.0, (n / 50.0) ** 0.5)
        # Normalize PF: 1.0 = 0.5 (neutral), 2.0 = 0.83, 3.0+ = ~0.95
        pf_score = max(0.0, min(1.0, (pf - 0.5) / 2.5))
        edge = (0.55 * pf_score + 0.35 * wr + 0.10 * conf)
        return round(edge, 3)

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

        # 2026-04-29: persist trade_outcomes to DB so it survives restart.
        # 2026-05-04: factored exit_outcomes tracking out of the except block
        # (indentation bug — was only running when persist FAILED, leaving
        # exit_learning table empty for the life of the project).
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

        # Always: keep in-memory window bounded
        with self._lock:
            if len(self._trade_outcomes[symbol]) > 100:
                self._trade_outcomes[symbol] = self._trade_outcomes[symbol][-100:]

            # Always: track per-exit-reason outcomes for the trail RL rules
            exit_key = "SL" if "sl" in exit_reason.lower() else exit_reason.split("[")[0].strip()
            key = f"{symbol}_{exit_key}"
            self._exit_outcomes.setdefault(key, []).append(r_multiple)
            if len(self._exit_outcomes[key]) > 50:
                self._exit_outcomes[key] = self._exit_outcomes[key][-50:]

            # Always: track giveback for trail tightness learning
            gb_key = f"_giveback_{symbol}"
            self._exit_outcomes.setdefault(gb_key, []).append(giveback_r)
            if len(self._exit_outcomes[gb_key]) > 50:
                self._exit_outcomes[gb_key] = self._exit_outcomes[gb_key][-50:]

        # Persist per-exit-reason rolling stats to exit_learning table.
        # Schema: (symbol, exit_reason, count, avg_r, best_r, worst_r, updated).
        # The brain / learning_engine can read this to decide which exits actually
        # produce profit per symbol — currently 0 rows because nothing wrote here.
        try:
            samples = list(self._exit_outcomes.get(key, []))
            if samples:
                count = len(samples)
                avg_r = float(sum(samples) / count)
                best_r = float(max(samples))
                worst_r = float(min(samples))
                conn = sqlite3.connect(str(RL_DB), timeout=5.0)
                conn.execute(
                    "INSERT OR REPLACE INTO exit_learning "
                    "(symbol, exit_reason, count, avg_r, best_r, worst_r, updated) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (symbol, exit_key, count, avg_r, best_r, worst_r,
                     datetime.now(timezone.utc).isoformat())
                )
                conn.commit()
                conn.close()
        except Exception as e:
            log.debug("[%s] exit_learning persist failed: %s", symbol, e)

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

        # Check rolling PF first — revert if strategy is failing (always allowed,
        # bypasses throttle so we never delay a safety reset).
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

        # Throttle: don't WEIGHT_UPDATE more than once per UPDATE_THROTTLE_SEC
        # per symbol. Prevents the 150-events/7d flapping observed on US2000.r
        # where weights bounced between adjustment and revert continuously.
        now = time.time()
        last = self._last_update_ts.get(symbol, 0)
        if (now - last) < UPDATE_THROTTLE_SEC:
            return

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
            self._n_updates += 1
            self._last_update_ts[symbol] = now

        # After global update, also tune per-regime weights wherever there's
        # enough samples (sparse but real edge — same symbol behaves
        # differently in trending vs ranging vs volatile vs low_vol).
        self._maybe_update_regime_weights(symbol, trades_with_components)

    def _maybe_update_regime_weights(self, symbol: str, trades_with_components: list):
        """Tune per-regime weights when a regime cell has enough samples.

        For each regime present in the recent window, compute per-component
        WR using ONLY trades from that regime. Update weight only if 5+
        samples in (regime, component). Bounded by same MAX_CHANGE / WEIGHT_MIN
        / WEIGHT_MAX as global weights. Persisted to regime_weights table.
        """
        by_regime: Dict[str, list] = {}
        for t in trades_with_components:
            reg = t.get("regime", "")
            if reg:
                by_regime.setdefault(reg, []).append(t)
        if not by_regime:
            return

        changes_all: Dict[str, Dict[str, dict]] = {}
        for regime, reg_trades in by_regime.items():
            if len(reg_trades) < 10:
                continue  # need enough regime samples
            # Per-component stats within this regime
            for comp in SCORE_COMPONENTS:
                wins = losses = 0
                for t in reg_trades:
                    if t["components"].get(comp, 0) > 0:
                        if t["won"]:
                            wins += 1
                        else:
                            losses += 1
                total = wins + losses
                if total < 5:
                    continue
                wr = wins / total
                # Same delta logic as global, half-magnitude (regime is finer)
                if wr > 0.55:
                    delta = min(MAX_CHANGE, (wr - 0.5) * 0.15)
                elif wr < 0.40:
                    delta = max(-MAX_CHANGE, (wr - 0.5) * 0.15)
                else:
                    delta = 0
                if delta == 0:
                    continue
                with self._lock:
                    sym_cells = self._regime_weights.setdefault(symbol, {}).setdefault(regime, {})
                    # start from the GLOBAL learned weight, not the previous regime cell —
                    # so a regime cell expresses a delta from the global view.
                    base_w = self._weights.get(symbol, {}).get(comp, 1.0)
                    cur_reg = sym_cells.get(comp, base_w)
                    new_w = cur_reg + delta
                    new_w = max(WEIGHT_MIN, min(WEIGHT_MAX, new_w))
                    if abs(new_w - cur_reg) < 1e-6:
                        continue
                    sym_cells[comp] = new_w
                changes_all.setdefault(regime, {})[comp] = {
                    "old": cur_reg, "new": new_w, "wr": wr, "n": total,
                }

        if changes_all:
            for regime, comps in changes_all.items():
                detail = "; ".join(
                    f"{c}: {v['old']:.2f}→{v['new']:.2f} (WR={v['wr']:.0%}, n={v['n']})"
                    for c, v in comps.items()
                )
                log.info("[%s][%s] RL REGIME WEIGHT UPDATE: %s",
                         symbol, regime, detail)
                self._audit(symbol, f"REGIME_WEIGHT_UPDATE:{regime}", detail)
                self._persist_regime_weights(symbol, regime, comps)
            self._n_regime_updates += 1

    def _persist_regime_weights(self, symbol: str, regime: str, stats: dict):
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            ts = datetime.now(timezone.utc).isoformat()
            sym_cells = self._regime_weights.get(symbol, {}).get(regime, {})
            for comp, w in sym_cells.items():
                s = stats.get(comp, {}) if stats else {}
                conn.execute("""
                    INSERT INTO regime_weights
                        (symbol, regime, component, weight,
                         win_count, loss_count, avg_r_win, avg_r_loss, updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, regime, component) DO UPDATE SET
                        weight = excluded.weight,
                        win_count = COALESCE(excluded.win_count, regime_weights.win_count),
                        loss_count = COALESCE(excluded.loss_count, regime_weights.loss_count),
                        updated = excluded.updated
                """, (symbol, regime, comp, w,
                      int(s.get("n", 0)) - int(s.get("losses", 0)) if s else None,
                      int(s.get("losses", 0)) if s else None,
                      None, None, ts))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("RL persist regime weights error: %s", e)

    # ═══════════════════════════════════════════════════════════════
    #  HEALTH MONITOR (called periodically by brain)
    # ═══════════════════════════════════════════════════════════════

    def health_summary(self, force_log: bool = False) -> dict:
        """Return a compact RL health snapshot. Logs once per hour by default.

        Surfaces: equity peak/DD, total tracked trades, # of weight cells
        adjusted (global + regime), # of trail-adjusted symbols, # of recent
        learning events this run. Operators can read the log to verify RL
        is actually learning (not silently stuck).
        """
        now = time.time()
        should_log = force_log or (now - self._last_health_log_ts) > 3600
        with self._lock:
            n_total_trades = sum(len(o) for o in self._trade_outcomes.values())
            n_weight_adj = sum(
                1 for s in self._weights for c, w in self._weights[s].items() if w != 1.0
            )
            n_regime_cells = sum(
                len(c) for s in self._regime_weights.values() for c in s.values()
            )
            n_trail_adj = sum(
                1 for s in self._trail_adjustments
                if any(v != 1.0 for v in self._trail_adjustments[s].values())
            )
            peak = self._peak_equity
            n_updates = self._n_updates
            n_regime_updates = self._n_regime_updates
            # Per-symbol revert count
            n_reverted = sum(1 for s, r in self._reverted.items() if r)

        snap = {
            "peak_equity": round(peak, 2),
            "tracked_trades": n_total_trades,
            "weight_cells_adjusted": n_weight_adj,
            "regime_cells_adjusted": n_regime_cells,
            "trail_symbols_adjusted": n_trail_adj,
            "reverted_symbols": n_reverted,
            "weight_updates_this_run": n_updates,
            "regime_updates_this_run": n_regime_updates,
        }
        if should_log:
            log.info(
                "[RL HEALTH] peak=$%.2f | trades=%d | weight_cells=%d | "
                "regime_cells=%d | trail_syms=%d | reverted=%d | "
                "updates_run=%d (regime=%d)",
                peak, n_total_trades, n_weight_adj, n_regime_cells,
                n_trail_adj, n_reverted, n_updates, n_regime_updates,
            )
            self._last_health_log_ts = now
        return snap

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

    def _persist_weights(self, symbol: str, component_stats: Optional[Dict] = None):
        """Persist learned weights + supporting evidence (win_count, loss_count, avg_R).
        Bug fixed 2026-05-06: prior version only saved (symbol, component, weight),
        leaving win_count/loss_count/avg_r_win/avg_r_loss at 0 for the life of the
        project — the table couldn't be audited and any tuner reading it saw
        "no evidence" for every component. component_stats is the dict computed
        in _maybe_update_weights; passing it through preserves the full record.
        """
        try:
            conn = sqlite3.connect(str(RL_DB), timeout=10.0)
            ts = datetime.now(timezone.utc).isoformat()
            stats = component_stats or {}
            for comp, w in self._weights.get(symbol, {}).items():
                s = stats.get(comp, {})
                conn.execute("""
                    INSERT INTO score_weights
                        (symbol, component, weight, win_count, loss_count,
                         avg_r_win, avg_r_loss, updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, component) DO UPDATE SET
                        weight = excluded.weight,
                        win_count = COALESCE(excluded.win_count, score_weights.win_count),
                        loss_count = COALESCE(excluded.loss_count, score_weights.loss_count),
                        avg_r_win = COALESCE(excluded.avg_r_win, score_weights.avg_r_win),
                        avg_r_loss = COALESCE(excluded.avg_r_loss, score_weights.avg_r_loss),
                        updated = excluded.updated
                """, (symbol, comp, w,
                      int(s.get("wins", 0)) if s else None,
                      int(s.get("losses", 0)) if s else None,
                      float(s.get("avg_r_win", 0)) if s else None,
                      float(s.get("avg_r_loss", 0)) if s else None,
                      ts))
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
