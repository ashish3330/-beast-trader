"""
Dragon Trader — Agent Brain (Hybrid: Rule-Based Scoring + ML Meta-Label + MasterBrain Gating).

Decision loop (~1s cycle):
  1. Read ticks from SharedState
  2. Build H1 candles (done by tick_streamer)
  3. Compute momentum scores (_score from momentum_scorer)
  4. Pick direction from higher score side if >= MIN_SCORE
  5. Gate checks: session, M15 alignment, position management
  6. Optional ML meta-label filter (skip entry if model says < 0.50)
  7. MasterBrain gate: evaluate_entry() for risk scaling + approval
  8. Risk checks (warn only, never block)
  9. Execute via Executor with MasterBrain-scaled lot sizing
 10. Manage positions: trailing SL, M15 reversal exit, intelligent exits
 11. Update SharedState for dashboard

The scoring system is the PRIMARY signal — always runs.
The ML meta-label is OPTIONAL — degrades gracefully to pure scoring.
The MasterBrain is OPTIONAL — degrades gracefully if not provided.
"""
import time
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SYMBOLS, TICK_INTERVAL_MS,
    MAX_RISK_PER_TRADE_PCT, MAX_TOTAL_EXPOSURE_PCT,
    DAILY_LOSS_LIMIT_PCT,
    DD_REDUCE_THRESHOLD, DD_PAUSE_THRESHOLD, DD_EMERGENCY_CLOSE,
    DAILY_HARD_STOP_PCT, WEEKLY_HARD_STOP_PCT,
    SESSION_START_UTC, SESSION_END_UTC, STARTING_BALANCE,
    ATR_SL_MULTIPLIER, MODEL_DIR,
    DRAGON_MIN_SCORE_BASELINE, DRAGON_CONFIDENCE_FLOOR,
    DRAGON_ML_ENABLED,
    SYMBOL_SESSION_OVERRIDE, SYMBOL_ATR_SL_OVERRIDE,
    DRAGON_SYMBOL_MIN_SCORE,
    SMART_ENTRY_MODE,
    COOLDOWN_BROKER_CLOSE_SECS, COOLDOWN_SL_HIT_SECS,
    COOLDOWN_WIN_SECS, COOLDOWN_LOSS_SECS,
)
from data.tick_streamer import SharedState
from execution.executor import Executor

# ── Momentum scorer internals (proven scoring system) ──
from signals.momentum_scorer import (
    _compute_indicators, _score, _score_with_components,
    IND_DEFAULTS, IND_OVERRIDES, MIN_SCORE, REGIME_PARAMS,
)
from signals.industry_gates import compute_gate_indicators, evaluate_entry_gates
from data.feature_engine import FeatureEngine

# ── RL + intelligence modules (optional — graceful degradation) ──
try:
    from agent.rl_learner import RLLearner
except ImportError:
    RLLearner = None
try:
    from agent.pattern_learner import PatternLearner
except ImportError:
    PatternLearner = None
try:
    from agent.order_flow import OrderFlowIntel
except ImportError:
    OrderFlowIntel = None
try:
    from agent.level_memory import LevelMemory
except ImportError:
    LevelMemory = None
try:
    from agent.fvg_detector import FVGDetector
except ImportError:
    FVGDetector = None

log = logging.getLogger("dragon.brain")

# ═══ CONSTANTS ═══
CYCLE_INTERVAL_S = 0.5           # 500ms decision cycle — real-time scoring
META_PROB_THRESHOLD = 0.48       # V5: lowered from 0.50 (0.499 was displayed as 0.50 but rejected)
META_AUC_MIN = 0.55              # minimum AUC to trust meta-label
H1_MIN_BARS = 100                # minimum H1 bars for scoring
M15_MIN_BARS = 50                # minimum M15 bars for direction check


class AgentBrain:
    """Hybrid trading agent: rule-based scoring + optional ML meta-label + MasterBrain gating."""

    def __init__(self, state: SharedState, mt5, executor: Executor,
                 meta_model=None, master_brain=None, exit_intelligence=None,
                 learning_engine=None, mtf_intelligence=None, equity_guardian=None,
                 smart_entry=None, calendar_filter=None, trade_intelligence=None,
                 rl_learner=None, pattern_learner=None, order_flow=None,
                 level_memory=None, fvg_detector=None,
                 alerter=None, metrics=None):
        """
        Args:
            state: SharedState from tick_streamer (thread-safe).
            mt5: MT5 connection (rpyc bridge).
            executor: Executor for order management.
            meta_model: Optional ML meta-label model (SignalModel or similar).
                        If None or fails validation, runs pure scoring mode.
            master_brain: Optional MasterBrain for entry gating + risk scaling.
                          If None, falls back to fixed MAX_RISK_PER_TRADE_PCT.
            exit_intelligence: Optional ExitIntelligence for smart exits.
                               If None, uses standard trailing SL only.
        """
        self.state = state
        self.mt5 = mt5
        self.executor = executor
        self.running = False
        self._thread = None
        self._cycle = int(0)
        self._mt5_degraded_streak = int(0)   # consecutive cycles failing on transport errors
        self._daily_start_equity = float(0.0)
        self._daily_loss = float(0.0)
        self._last_day = None
        self._trade_log = []

        # ── HARD KILL SWITCH STATE ──
        self._weekly_start_equity = float(0.0)
        self._weekly_start_day = None          # Monday date for weekly reset
        # Kill switch — restored from SQLite (trade_journal.db) so a restart
        # can't bypass a triggered hard stop. Real-money safety: SharedState
        # is in-memory only and would lose the flag on every process restart.
        self._init_kill_switch_table()
        ks_saved = self._load_kill_switch() or {}
        self._kill_switch_active = bool(ks_saved.get("active", False))
        self._kill_switch_reason = str(ks_saved.get("reason", ""))
        self._kill_switch_until = None
        _tripped_iso = ks_saved.get("tripped_at_iso")
        if isinstance(_tripped_iso, str):
            try:
                self._kill_switch_tripped_at = datetime.fromisoformat(_tripped_iso)
            except ValueError:
                self._kill_switch_tripped_at = None
        else:
            self._kill_switch_tripped_at = _tripped_iso
        self._kill_switch_tripped_loss = float(ks_saved.get("tripped_loss", 0.0))
        if self._kill_switch_active:
            log.warning("KILL SWITCH RESTORED FROM DB: %s, tripped at %s with loss %.2f%%",
                        self._kill_switch_reason, self._kill_switch_tripped_at,
                        self._kill_switch_tripped_loss)

        # ── Equity baselines: restore from DB so kill switches survive restart ──
        self._init_equity_state_table()
        self._saved_equity_state = self._load_equity_state() or {}

        # ── Entry metadata cache: symbol → {score, regime, direction, entry_price, ts} ──
        # Used by learning engine deal sync to attach brain metadata to SL/TP exits.
        # Persisted to trade_journal.db so the 12+ daily restarts don't lose risk_pct
        # (which would otherwise zero out r_multiple in deal sync).
        self._entry_metadata: Dict[str, dict] = {}
        self._init_entry_metadata_table()
        self._load_entry_metadata()

        # ── Candle-close tracking: only re-score when new candle appears ──
        self._last_candle_time: Dict[str, float] = {}
        self._last_scores: Dict[str, dict] = {}  # cached scores for dashboard between candles

        # ── 2026-05-17: entry-rate guards ──
        # _last_entry_bar: H1-bar bucket on which we last took an entry per
        # symbol. Backtest baseline 83 BTC trades/180d vs live 204 scaled =
        # 2.5x over-fire (EVAL_ON_CANDLE_CLOSE=False since 2026-05-13 means
        # live re-scores every ~2s; without dedup the same H1 setup can
        # fire many entries before the bar rolls).
        # _score_hist: last 5 raw_scores per symbol for the LATE_MOMENTUM
        # gate. Today's BTC trades fired because score jumped 5.5 → 7.9 in
        # 2 seconds (07:30:33 → 07:30:35) — that's a late-confirm trap.
        self._last_entry_bar: Dict[str, int] = {}
        self._score_hist: Dict[str, list] = {}

        # ── MasterBrain (optional Dragon gating) ──
        self._master_brain = master_brain
        self._exit_intelligence = exit_intelligence
        self._learning_engine = learning_engine
        self._mtf = mtf_intelligence
        self._guardian = equity_guardian
        self._smart_entry = smart_entry
        self._calendar = calendar_filter
        self._trade_intel = trade_intelligence

        # ── RL + Intelligence modules (optional) ──
        self._rl_learner = rl_learner
        self._pattern_learner = pattern_learner
        self._order_flow = order_flow
        self._level_memory = level_memory
        self._fvg = fvg_detector

        # ── Observability (optional, never blocks trading) ──
        # Lazy fallback to module-level Alerter so hooks don't NPE if run.py
        # didn't pass instances (e.g. CLI tools, tests).
        if alerter is None:
            try:
                from agent.alerting import get_default_alerter
                alerter = get_default_alerter()
            except Exception:
                alerter = None
        if metrics is None:
            try:
                from agent.metrics import get_default_metrics
                metrics = get_default_metrics()
            except Exception:
                metrics = None
        self._alerter = alerter
        self._metrics = metrics

        # ── ML Meta-Label (optional enhancement) ──
        self._meta_model = meta_model
        self.ml_enabled = False
        self._validate_meta_model()

        # ── Feature Engine for ML features ──
        self._feature_engine = FeatureEngine(state)

        # ── Win streak tracking for meta-label ──
        self._recent_win_streak = int(0)

        # ── Tick momentum delay tracking ──
        self._tick_delayed = {}    # symbol -> True if delayed last cycle

        # ── Re-entry cooldown: restore from SharedState (survives restarts) ──
        # Drops expired entries on load so the dict never grows stale.
        saved_cooldowns = self.state.get_agent_state().get("sl_cooldowns", {})
        self._sl_cooldown: Dict[str, float] = {
            k: float(v) for k, v in saved_cooldowns.items() if float(v) > time.time()
        }
        self._cooldown_reason: Dict[str, str] = {}
        # 2026-05-11: per-cooldown direction-block ("BOTH" | "LONG" | "SHORT").
        # Not persisted across restarts — defaults to safer "BOTH" on cold start.
        self._cooldown_blocked: Dict[str, str] = {s: "BOTH" for s in self._sl_cooldown}
        if self._sl_cooldown:
            active = {s: f"{(v - time.time())/60:.0f}min" for s, v in self._sl_cooldown.items()}
            log.info("Restored cooldowns from state: %s", active)

        # ── Indicator cache (recompute every cycle per symbol) ──
        self._ind_cache = {}       # symbol -> (indicators_dict, timestamp)
        self._ind_cache_ttl = 0.25  # 250ms cache — near-instant scoring

        # ── Pullback entry: deferred signals waiting for retrace ──
        # 2026-05-22: Persisted to SQLite (parallel to entry_metadata).
        # Today's PULLBACK_MAX_WAIT_BARS=5 means up to 5 H1 bars (~5h) of wait.
        # Without persistence, a bot restart mid-wait silently drops the
        # signal — at 0.8 ATR retrace + 5h max wait this is now a real risk.
        self._pending_pullback: Dict[str, dict] = {}  # symbol -> {direction, score, atr, risk_pct, signal_price, bars_waited, ...}
        self._init_pending_pullback_table()
        self._load_pending_pullback()

        # ── Last close time per symbol: force pullback on re-entry ──
        self._last_close_time: Dict[str, float] = {}

    # ═══════════════════════════════════════════════════════════════
    #  COOLDOWN MANAGEMENT (single source of truth)
    # ═══════════════════════════════════════════════════════════════

    def _arm_cooldown(self, symbol: str, secs: int, reason: str,
                       blocked_direction: str = "BOTH") -> float:
        """Arm a re-entry cooldown.

        blocked_direction:
          "BOTH" — block LONG and SHORT (default — losses, manual closes)
          "LONG" — block only LONG (used when a winning LONG closed; SHORT free)
          "SHORT" — symmetric

        max(existing, now+secs) so a longer cooldown can never be undercut by
        a shorter one. If the new cooldown is BOTH, it always wins; if the new
        is directional and the existing is BOTH, BOTH stays.
        """
        now = time.time()
        new_expiry = now + max(0, int(secs))
        cur_expiry = self._sl_cooldown.get(symbol, 0.0)
        cur_blocked = self._cooldown_blocked.get(symbol, "BOTH")
        chosen = max(cur_expiry, new_expiry)
        was_active = cur_expiry > now

        # Direction merge: BOTH dominates a directional entry.
        if was_active and cur_blocked == "BOTH":
            blocked = "BOTH"
        elif blocked_direction == "BOTH":
            blocked = "BOTH"
        else:
            blocked = blocked_direction

        self._sl_cooldown[symbol] = chosen
        self._cooldown_reason[symbol] = reason
        self._cooldown_blocked[symbol] = blocked
        live = {s: v for s, v in self._sl_cooldown.items() if v > now}
        self.state.update_agent("sl_cooldowns", live)
        if not was_active:
            log.info("[%s] COOLDOWN ARMED: %s for %dm blocks=%s",
                     symbol, reason, int((chosen - now) / 60), blocked)
        return chosen

    def _cooldown_active(self, symbol: str, direction: str = None):
        """Return (active, mins_left, reason).

        If `direction` is given ('LONG'/'SHORT'), only return active=True when
        that direction is blocked. If None (legacy callers), active=True for
        any blocked direction.

        Auto-deletes expired entries.
        """
        now = time.time()
        expiry = self._sl_cooldown.get(symbol, 0.0)
        if expiry <= now:
            if symbol in self._sl_cooldown:
                self._sl_cooldown.pop(symbol, None)
                self._cooldown_reason.pop(symbol, None)
                self._cooldown_blocked.pop(symbol, None)
            return False, 0.0, ""
        blocked = self._cooldown_blocked.get(symbol, "BOTH")
        if direction is not None and blocked != "BOTH" and blocked != direction:
            # Cooldown is directional but our direction isn't blocked.
            return False, 0.0, ""
        return True, (expiry - now) / 60.0, self._cooldown_reason.get(symbol, "")

    # ═══════════════════════════════════════════════════════════════
    #  ENTRY METADATA PERSISTENCE (survives restarts so deal sync
    #  attaches correct risk_pct → r_multiple instead of always 0)
    # ═══════════════════════════════════════════════════════════════

    def _entry_metadata_db(self):
        from config import DB_PATH
        return DB_PATH.parent / "trade_journal.db"

    def _init_entry_metadata_table(self):
        import sqlite3
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS entry_metadata "
                "(symbol TEXT PRIMARY KEY, payload TEXT NOT NULL, ts REAL NOT NULL)")
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("entry_metadata table init failed: %s", e)

    def _load_entry_metadata(self):
        """Restore entry metadata on startup. Drops rows >24h old."""
        import sqlite3, json
        try:
            cutoff = time.time() - 259200  # 72h (was 24h — too tight for slow trades, RL got corrupt data)
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            rows = conn.execute(
                "SELECT symbol, payload, ts FROM entry_metadata WHERE ts > ?",
                (cutoff,)).fetchall()
            conn.execute("DELETE FROM entry_metadata WHERE ts <= ?", (cutoff,))
            conn.commit()
            conn.close()
            for sym, payload, ts in rows:
                try:
                    self._entry_metadata[sym] = json.loads(payload)
                except Exception:
                    pass
            if rows:
                log.info("Restored entry_metadata for %d symbols from DB", len(rows))
                self.state.update_agent("entry_metadata", dict(self._entry_metadata))
        except Exception as e:
            log.warning("entry_metadata load failed: %s", e)

    def _persist_entry_metadata(self, symbol: str, meta: dict):
        """Write entry metadata to disk on every entry. Idempotent."""
        import sqlite3, json
        try:
            safe = {k: v for k, v in meta.items()
                    if isinstance(v, (str, int, float, bool, type(None), list, dict))}
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "INSERT OR REPLACE INTO entry_metadata (symbol, payload, ts) VALUES (?, ?, ?)",
                (symbol, json.dumps(safe), time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("[%s] entry_metadata persist failed: %s", symbol, e)

    # ═══════════════════════════════════════════════════════════════
    #  PENDING_PULLBACK PERSISTENCE (2026-05-22 audit finding #3)
    #  Restart between signal-arm and fill = lost signal. With
    #  PULLBACK_MAX_WAIT_BARS=5 (5 H1 bars ~ 5h) and the 12+ daily restart
    #  cadence, persistence is required to keep deferred signals alive.
    # ═══════════════════════════════════════════════════════════════

    def _init_pending_pullback_table(self):
        import sqlite3
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS pending_pullback "
                "(symbol TEXT PRIMARY KEY, payload TEXT NOT NULL, ts REAL NOT NULL)")
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("pending_pullback table init failed: %s", e)

    def _load_pending_pullback(self):
        """Restore pending pullback signals on startup.
        Prunes entries older than PULLBACK_MAX_WAIT_BARS * 1h (5h default).
        H1 wait is approximated by wall-clock since arm time."""
        import sqlite3, json
        try:
            from config import PULLBACK_MAX_WAIT_BARS as _PB_WAIT
            max_age = float(_PB_WAIT) * 3600.0  # H1 bars → seconds
        except Exception:
            max_age = 5 * 3600.0
        try:
            cutoff = time.time() - max_age
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            rows = conn.execute(
                "SELECT symbol, payload, ts FROM pending_pullback WHERE ts > ?",
                (cutoff,)).fetchall()
            conn.execute("DELETE FROM pending_pullback WHERE ts <= ?", (cutoff,))
            conn.commit()
            conn.close()
            restored = 0
            for sym, payload, ts in rows:
                try:
                    self._pending_pullback[sym] = json.loads(payload)
                    restored += 1
                except Exception:
                    pass
            if restored:
                log.info("Restored pending_pullback for %d symbols from DB", restored)
        except Exception as e:
            log.warning("pending_pullback load failed: %s", e)

    def _persist_pending_pullback(self, symbol: str, pb: dict):
        """Write a pending pullback to disk on arm + on each cycle's bars_waited bump.
        Idempotent — uses INSERT OR REPLACE."""
        import sqlite3, json
        try:
            safe = {k: v for k, v in pb.items()
                    if isinstance(v, (str, int, float, bool, type(None), list, dict))}
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "INSERT OR REPLACE INTO pending_pullback (symbol, payload, ts) VALUES (?, ?, ?)",
                (symbol, json.dumps(safe), time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("[%s] pending_pullback persist failed: %s", symbol, e)

    def _delete_pending_pullback(self, symbol: str):
        """Remove a pending pullback from disk (after fill, fallback, or cancel)."""
        import sqlite3
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute("DELETE FROM pending_pullback WHERE symbol = ?", (symbol,))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("[%s] pending_pullback delete failed: %s", symbol, e)

    def _init_kill_switch_table(self):
        """Single-row table holding the current kill_switch state."""
        import sqlite3
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kill_switch "
                "(id INTEGER PRIMARY KEY CHECK (id=1), payload TEXT NOT NULL, ts REAL NOT NULL)")
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("kill_switch table init failed: %s", e)

    def _load_kill_switch(self):
        """Read persisted kill_switch state on startup."""
        import sqlite3, json
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            row = conn.execute("SELECT payload FROM kill_switch WHERE id=1").fetchone()
            conn.close()
            if row and row[0]:
                return json.loads(row[0])
        except Exception as e:
            log.warning("kill_switch load failed: %s", e)
        return {}

    def _persist_kill_switch(self):
        """Write current kill_switch state to disk. Called after every state change."""
        import sqlite3, json
        try:
            payload = {
                "active": bool(self._kill_switch_active),
                "reason": str(self._kill_switch_reason or ""),
                "tripped_at_iso": (self._kill_switch_tripped_at.isoformat()
                                   if self._kill_switch_tripped_at is not None
                                   and hasattr(self._kill_switch_tripped_at, "isoformat")
                                   else self._kill_switch_tripped_at),
                "tripped_loss": float(self._kill_switch_tripped_loss),
            }
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "INSERT OR REPLACE INTO kill_switch (id, payload, ts) VALUES (1, ?, ?)",
                (json.dumps(payload), time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("kill_switch persist failed: %s", e)

    # ═══════════════════════════════════════════════════════════════
    #  EQUITY STATE PERSISTENCE
    #  Without this, _daily_start_equity / _weekly_start_equity / peak_equity
    #  get re-baselined to current equity on every restart, which makes the
    #  daily/weekly hard-stop kill switches blind to losses that occurred
    #  before the restart. Same-day/same-week restarts must restore baselines.
    # ═══════════════════════════════════════════════════════════════

    def _init_equity_state_table(self):
        import sqlite3
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_equity_state "
                "(id INTEGER PRIMARY KEY CHECK (id=1), payload TEXT NOT NULL, ts REAL NOT NULL)")
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("agent_equity_state table init failed: %s", e)

    def _load_equity_state(self):
        import sqlite3, json
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            row = conn.execute("SELECT payload FROM agent_equity_state WHERE id=1").fetchone()
            conn.close()
            if row and row[0]:
                return json.loads(row[0])
        except Exception as e:
            log.warning("agent_equity_state load failed: %s", e)
        return {}

    def _persist_equity_state(self):
        import sqlite3, json
        try:
            payload = {
                "peak_equity": float(self.state.get_agent_state().get("peak_equity", 0.0)),
                "daily_start_equity": float(self._daily_start_equity),
                "daily_start_day_iso": (self._last_day.isoformat()
                                        if self._last_day is not None else None),
                "weekly_start_equity": float(self._weekly_start_equity),
                "weekly_start_monday_iso": (self._weekly_start_day.isoformat()
                                            if self._weekly_start_day is not None else None),
            }
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "INSERT OR REPLACE INTO agent_equity_state (id, payload, ts) VALUES (1, ?, ?)",
                (json.dumps(payload), time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("agent_equity_state persist failed: %s", e)

    # ═══════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════════════

    def start(self):
        """Start the agent brain in a background thread."""
        self.running = True
        self.state.update_agent("running", True)
        mode = "hybrid" if self.ml_enabled else "scoring_only"
        if self._master_brain:
            mode = "dragon_" + mode
        self.state.update_agent("mode", mode)

        equity = float(self.state.get_agent_state().get("equity", STARTING_BALANCE))
        today = datetime.now(timezone.utc).date()
        current_monday = today - __import__('datetime').timedelta(days=today.weekday())

        # Restore baselines if persisted state matches current day/week — otherwise fresh.
        # Without this, mid-day restarts re-baseline to current (depressed) equity and
        # the daily/weekly hard-stop kill switches go blind to pre-restart losses.
        saved = getattr(self, "_saved_equity_state", {}) or {}
        from datetime import date as _date
        try:
            saved_day = _date.fromisoformat(saved["daily_start_day_iso"]) if saved.get("daily_start_day_iso") else None
        except Exception:
            saved_day = None
        try:
            saved_monday = _date.fromisoformat(saved["weekly_start_monday_iso"]) if saved.get("weekly_start_monday_iso") else None
        except Exception:
            saved_monday = None

        if saved_day == today and float(saved.get("daily_start_equity", 0)) > 0:
            self._daily_start_equity = float(saved["daily_start_equity"])
            log.info("RESTORED daily_start_equity=$%.2f (same day %s)",
                     self._daily_start_equity, today.isoformat())
        else:
            self._daily_start_equity = float(equity)
        self._last_day = today

        if saved_monday == current_monday and float(saved.get("weekly_start_equity", 0)) > 0:
            self._weekly_start_equity = float(saved["weekly_start_equity"])
            log.info("RESTORED weekly_start_equity=$%.2f (same week starting %s)",
                     self._weekly_start_equity, current_monday.isoformat())
        else:
            self._weekly_start_equity = float(equity)
        self._weekly_start_day = current_monday

        # Restore peak_equity to SharedState (which is in-memory only) so DD reads correctly.
        saved_peak = float(saved.get("peak_equity", 0.0))
        if saved_peak > equity:
            self.state.update_agent("peak_equity", saved_peak)
            log.info("RESTORED peak_equity=$%.2f (current equity $%.2f)", saved_peak, equity)
        else:
            self.state.update_agent("peak_equity", float(equity))

        self._persist_equity_state()
        self._kill_switch_active = self._kill_switch_active  # preserved from __init__ load
        self._kill_switch_reason = self._kill_switch_reason
        self._kill_switch_until = None

        self._thread = threading.Thread(
            target=self._decision_loop, daemon=True, name="AgentBrain")
        self._thread.start()
        log.info("Dragon brain started (cycle=%.1fs, mode=%s, BASE_MIN_SCORE=%.1f, regime_adaptive=True, master_brain=%s)",
                 CYCLE_INTERVAL_S,
                 "HYBRID" if self.ml_enabled else "SCORING_ONLY",
                 MIN_SCORE,
                 "ENABLED" if self._master_brain else "DISABLED")

    def stop(self):
        """Stop the agent brain."""
        self.running = False
        self.state.update_agent("running", False)
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Dragon brain stopped after %d cycles", self._cycle)

    # ═══════════════════════════════════════════════════════════════
    #  META-LABEL VALIDATION
    # ═══════════════════════════════════════════════════════════════

    def _validate_meta_model(self):
        """Enable ML meta-label only if model is loaded and AUC > threshold."""
        if self._meta_model is None:
            self.ml_enabled = False
            log.info("No meta-label model provided — running pure scoring mode")
            return

        try:
            # Try loading saved models
            self._meta_model.load()
            has_any = False
            for sym in SYMBOLS:
                if self._meta_model.has_model(sym):
                    metrics = self._meta_model._train_metrics.get(sym, {})
                    auc = float(metrics.get("test_auc", metrics.get("auc", 0.0)))
                    if auc >= META_AUC_MIN:
                        has_any = True
                        log.info("[%s] Meta-label model loaded (AUC=%.3f >= %.2f)",
                                 sym, auc, META_AUC_MIN)
                    else:
                        log.warning("[%s] Meta-label AUC=%.3f < %.2f — disabled for this symbol",
                                    sym, auc, META_AUC_MIN)

            if has_any:
                self.ml_enabled = True
                log.info("Meta-label filter ENABLED (at least one symbol passed AUC check)")
            else:
                self.ml_enabled = False
                log.info("No symbol passed meta-label AUC check — pure scoring mode")
        except Exception as e:
            self.ml_enabled = False
            log.warning("Meta-label validation failed: %s — pure scoring mode", e)

    # ═══════════════════════════════════════════════════════════════
    #  MAIN DECISION LOOP
    # ═══════════════════════════════════════════════════════════════

    def _decision_loop(self):
        """Main loop: every ~1s, evaluate all symbols.

        Transport errors (MT5 bridge dropped, rpyc stream closed) are handled
        as DEGRADED state — single-line warning, no traceback, no entry signals
        are emitted while degraded. Watchdog's crash_loop pattern keys off
        traceback density so we deliberately avoid spamming stack traces for
        recoverable network conditions.
        """
        from execution.mt5_client import MT5Unavailable, _TRANSPORT_ERRORS
        while self.running:
            loop_start = time.time()
            self._cycle += 1

            try:
                self._run_cycle()
                # Successful cycle clears any prior degraded state.
                if self._mt5_degraded_streak:
                    log.info("MT5 RECOVERED after %d degraded cycles", self._mt5_degraded_streak)
                    self._mt5_degraded_streak = 0
                    try:
                        self.state.update_agent("mt5_degraded_streak", 0)
                    except Exception:
                        pass
            except MT5Unavailable as e:
                self._mt5_degraded_streak += 1
                try:
                    self.state.update_agent("mt5_degraded_streak", self._mt5_degraded_streak)
                except Exception:
                    pass
                # Log once per 20 cycles while degraded to avoid log spam.
                if self._mt5_degraded_streak == 1 or self._mt5_degraded_streak % 20 == 0:
                    log.warning("MT5 DEGRADED cycle %d (streak=%d): %s",
                                self._cycle, self._mt5_degraded_streak, e)
            except _TRANSPORT_ERRORS as e:
                # Escaped the facade (e.g. rpyc netref accessed after drop).
                self._mt5_degraded_streak += 1
                try:
                    self.state.update_agent("mt5_degraded_streak", self._mt5_degraded_streak)
                except Exception:
                    pass
                if self._mt5_degraded_streak == 1 or self._mt5_degraded_streak % 20 == 0:
                    log.warning("MT5 transport hiccup cycle %d (streak=%d): %s: %s",
                                self._cycle, self._mt5_degraded_streak,
                                type(e).__name__, e)
            except Exception as e:
                log.error("Dragon brain cycle %d error: %s", self._cycle, e, exc_info=True)

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, CYCLE_INTERVAL_S - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _run_cycle(self):
        """Single cycle: kill switch → guardian → daily reset → DD check → process symbols → exits → manage → dashboard."""
        # ── DRIFT DETECTOR: refresh per-symbol live PF/WR state every 5 min ──
        # Cheap (single DB query), in-process (no separate cron needed). Brain
        # reads symbol_drift_state inside master_brain.calculate_swing_risk()
        # and _get_adaptive_min_score() each cycle, so this keeps both fresh.
        if self._cycle % 600 == 1:  # 600 cycles × 0.5s = 5 min, +1 to fire on 1st cycle
            try:
                from agent import drift_detector
                drift_detector.update_all()
            except Exception as e:
                log.debug("drift refresh failed: %s", e)

        # ── EQUITY GUARDIAN: real-time P&L monitoring (runs FIRST every cycle) ──
        if self._guardian:
            try:
                self._guardian.monitor()
            except Exception as e:
                log.warning("Guardian error: %s", e)

        # ── Daily + Weekly reset at midnight UTC ──
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()
        if today != self._last_day:
            self._last_day = today
            equity = float(self.state.get_agent_state().get("equity", STARTING_BALANCE))
            self._daily_start_equity = float(equity)
            self._daily_loss = float(0.0)
            log.info("New trading day — daily start equity: $%.2f", equity)

            # Reset daily kill switch at new day
            if self._kill_switch_active and self._kill_switch_reason == "daily":
                self._kill_switch_active = False
                self._kill_switch_reason = ""
                self._kill_switch_until = None
                log.info("KILL SWITCH RESET — new trading day, resuming trading")
                self._persist_kill_switch()

            # Weekly reset: Monday 00:00 UTC
            from datetime import timedelta
            current_monday = today - timedelta(days=today.weekday())
            if self._weekly_start_day is None or current_monday > self._weekly_start_day:
                self._weekly_start_day = current_monday
                self._weekly_start_equity = float(equity)
                log.info("New trading week — weekly start equity: $%.2f", equity)
                # Reset weekly kill switch on new week
                if self._kill_switch_active and self._kill_switch_reason == "weekly":
                    self._kill_switch_active = False
                    self._kill_switch_reason = ""
                    self._kill_switch_until = None
                    log.info("KILL SWITCH RESET — new trading week, resuming trading")
                    self._persist_kill_switch()

            # Persist new baselines so a same-day restart restores them.
            self._persist_equity_state()

        # ── Current account state (thread-safe read) ──
        agent = self.state.get_agent_state()
        equity = float(agent.get("equity", STARTING_BALANCE))
        dd_pct = float(agent.get("dd_pct", 0.0))

        # ── Daily loss ──
        daily_pnl = equity - self._daily_start_equity
        daily_loss_pct = float(
            abs(daily_pnl) / self._daily_start_equity * 100
            if daily_pnl < 0 and self._daily_start_equity > 0
            else 0.0
        )
        self._daily_loss = daily_loss_pct

        # ── Weekly loss ──
        weekly_pnl = equity - self._weekly_start_equity
        weekly_loss_pct = float(
            abs(weekly_pnl) / self._weekly_start_equity * 100
            if weekly_pnl < 0 and self._weekly_start_equity > 0
            else 0.0
        )

        # ═══════════════════════════════════════════════════════════════
        #  HARD KILL SWITCH — UPSTREAM OF ALL TRADING LOGIC
        #  Nothing can bypass this. Checked every cycle.
        # ═══════════════════════════════════════════════════════════════

        # Check daily hard stop
        # Guard against bogus 0-equity readings (mt5linux returns 0 when RPC times out)
        if (not self._kill_switch_active and daily_loss_pct >= DAILY_HARD_STOP_PCT
                and equity > 0 and self._daily_start_equity > 0):
            log.critical(
                "DAILY KILL SWITCH TRIGGERED: loss %.2f%% >= %.1f%% "
                "(equity $%.2f, day start $%.2f) — CLOSING ALL, NO NEW TRADES UNTIL TOMORROW",
                daily_loss_pct, DAILY_HARD_STOP_PCT, equity, self._daily_start_equity)
            self._kill_switch_active = True
            self._kill_switch_reason = "daily"
            self._kill_switch_until = None  # resets at next day boundary
            self._kill_switch_tripped_at = now_utc
            self._kill_switch_tripped_loss = float(daily_loss_pct)
            self._persist_kill_switch()
            self.executor.close_all("DailyKillSwitch")

        # Check weekly hard stop (same 0-equity guard)
        if (not self._kill_switch_active and weekly_loss_pct >= WEEKLY_HARD_STOP_PCT
                and equity > 0 and self._weekly_start_equity > 0):
            log.critical(
                "WEEKLY KILL SWITCH TRIGGERED: loss %.2f%% >= %.1f%% "
                "(equity $%.2f, week start $%.2f) — CLOSING ALL, NO NEW TRADES UNTIL NEXT MONDAY",
                weekly_loss_pct, WEEKLY_HARD_STOP_PCT, equity, self._weekly_start_equity)
            self._kill_switch_active = True
            self._kill_switch_tripped_at = now_utc
            self._kill_switch_tripped_loss = float(weekly_loss_pct)
            self._kill_switch_reason = "weekly"
            self._kill_switch_until = None  # resets at next Monday boundary
            self._persist_kill_switch()
            self.executor.close_all("WeeklyKillSwitch")

        # If kill switch is active: manage existing positions ONLY, skip all new trade logic
        if self._kill_switch_active:
            # Log every 60 cycles (~30 seconds) so we know it's alive.
            # Show TRIP-TIME loss (sticky) and reset time, not current loss — the kill switch
            # stays armed until the period boundary regardless of equity recovery.
            if self._cycle % 60 == 0:
                trip_str = (self._kill_switch_tripped_at.strftime("%H:%M UTC")
                            if self._kill_switch_tripped_at else "unknown")
                if self._kill_switch_reason == "daily":
                    reset_str = "00:00 UTC (next day)"
                else:
                    reset_str = "Mon 00:00 UTC"
                log.warning("KILL SWITCH ARMED (%s): tripped %s at -%.2f%% — "
                            "no new entries until %s. Current eq=$%.2f.",
                            self._kill_switch_reason, trip_str,
                            self._kill_switch_tripped_loss, reset_str, equity)

            # Still manage trailing SL for any positions that survived (or were re-opened manually)
            for symbol in SYMBOLS:
                try:
                    if self.executor.has_position(symbol):
                        self.executor.manage_trailing_sl(symbol)
                except Exception as e:
                    log.warning("[%s] Kill switch trailing SL error: %s", symbol, e)

            # Update dashboard state so user sees kill switch is active
            self.state.update_agent("cycle", int(self._cycle))
            self.state.update_agent("profit", float(equity - self._daily_start_equity))
            self.state.update_agent("dd_pct", float(dd_pct))
            self.state.update_agent("daily_loss", float(daily_loss_pct))
            self.state.update_agent("weekly_loss", float(weekly_loss_pct))
            self.state.update_agent("kill_switch", {
                "active": True,
                "reason": self._kill_switch_reason,
                "daily_loss_pct": float(daily_loss_pct),
                "weekly_loss_pct": float(weekly_loss_pct),
                "tripped_at_iso": (self._kill_switch_tripped_at.isoformat()
                                   if self._kill_switch_tripped_at is not None
                                   and hasattr(self._kill_switch_tripped_at, 'isoformat')
                                   else self._kill_switch_tripped_at),
                "tripped_loss": float(self._kill_switch_tripped_loss),
            })
            self.state.update_agent("positions", self.executor.get_positions_info())

            eq_hist = list(self.state.get_agent_state().get("equity_history", []))
            eq_hist.append({"time": time.time(), "equity": float(equity)})
            if len(eq_hist) > 2000:
                eq_hist = eq_hist[-2000:]
            self.state.update_agent("equity_history", eq_hist)
            return  # HARD STOP — skip entire trading pipeline

        # ═══ EMERGENCY DD CHECK ═══
        if dd_pct >= DD_EMERGENCY_CLOSE:
            log.critical("EMERGENCY DD %.1f%% >= %.1f%% — CLOSING ALL",
                         dd_pct, DD_EMERGENCY_CLOSE)
            self.executor.close_all("EmergencyDD")
            time.sleep(10)
            return

        # ═══ WARMUP: skip first 5 cycles after restart (let data stabilize) ═══
        if self._cycle <= 5:
            log.info("WARMUP cycle %d/5 — scoring only, no entries", self._cycle)
            for symbol in SYMBOLS:
                try:
                    # Still score for dashboard but don't enter
                    h1_df = self.state.get_candles(symbol, 60)
                    if h1_df is not None and len(h1_df) >= H1_MIN_BARS:
                        ind = self._get_indicators(symbol, h1_df)
                except Exception:
                    pass
            self.state.update_agent("cycle", int(self._cycle))
            return

        # ═══ PUSH RL TRAIL ADJUSTMENTS + CURRENT REGIME TO EXECUTOR ═══
        # 2026-05-14: moved BEFORE entry processing so trail management runs
        # FIRST. Previously trail ran AFTER entries — race condition allowed
        # the same cycle to:
        #   1. Open a new DJ30 LONG (entry signal fired first)
        #   2. Close the old DJ30 LONG via PEAK_GIVEBACK (trail loop ran second)
        # Result: two positions briefly open + immediate re-entry after big win.
        if self._rl_learner:
            for sym in SYMBOLS:
                try:
                    adj = self._rl_learner.get_trail_adjustments(sym)
                    self.executor.set_rl_trail_adjustments(sym, adj)
                except Exception:
                    pass
        # Push current regime (from previous cycle's _last_scores) so executor
        # uses regime-conditional trail profile.
        for sym in SYMBOLS:
            try:
                last = self._last_scores.get(sym)
                if last and last.get("regime"):
                    self.executor.set_current_regime(sym, last["regime"])
            except Exception:
                pass
        if self._rl_learner:
            try:
                self._rl_learner.health_summary()
            except Exception:
                pass

        # ═══ MANAGE TRAILING SL + MTF EXIT + M15 REVERSAL EXIT ═══
        # MUST RUN BEFORE _process_symbol so positions that should close
        # (peak-giveback, hard-dollar-cap, early-loss-cut) complete BEFORE
        # any new entry signal for the same symbol fires this cycle.
        manage_symbols = set(SYMBOLS)
        try:
            broker_positions = self.mt5.positions_get() or []
            for p in broker_positions:
                sym = getattr(p, "symbol", None)
                if sym:
                    manage_symbols.add(sym)
        except Exception:
            pass

        for symbol in manage_symbols:
            try:
                had_pos_before = self.executor.has_position(symbol)
                if had_pos_before:
                    self.executor.manage_trailing_sl(symbol)
                    self._check_m15_reversal_exit(symbol)
                # 2026-05-14: if trail-management closed the position THIS
                # cycle, arm the cooldown immediately so the same-cycle
                # entry loop sees it. Otherwise cooldown arms next cycle
                # via external-close detection — too late.
                if had_pos_before and not self.executor.has_position(symbol):
                    # 2026-05-22 audit fix: _peak_profit_r is popped during
                    # close_position(). Read _last_close_peak_r instead, which
                    # is captured BEFORE the pop. Falls back to live dict for
                    # the rare case the position closed via TP without going
                    # through executor.close_position().
                    last_peak = (
                        float(getattr(self.executor, "_last_close_peak_r", {}).get(symbol, 0.0))
                        or float(getattr(self.executor, "_peak_profit_r", {}).get(symbol, 0.0))
                    )
                    was_win = float(last_peak) >= 0.5
                    if was_win:
                        # 2026-05-22 post-big-win 5h cooldown — user rule:
                        # "after a great win on a symbol, don't trade for 5h".
                        # Detect great win by peak_r OR last_close_pnl_dollars.
                        try:
                            from config import (
                                POST_BIG_WIN_COOLDOWN_ENABLED,
                                POST_BIG_WIN_COOLDOWN_SECS,
                                POST_BIG_WIN_R_THRESHOLD,
                                POST_BIG_WIN_DOLLAR_THRESHOLD,
                            )
                        except Exception:
                            POST_BIG_WIN_COOLDOWN_ENABLED = False
                            POST_BIG_WIN_COOLDOWN_SECS = 18000
                            POST_BIG_WIN_R_THRESHOLD = 1.5
                            POST_BIG_WIN_DOLLAR_THRESHOLD = 3.0
                        last_pnl = float(getattr(self.executor, "_last_close_pnl", {}).get(symbol, 0.0))
                        is_big_win = (
                            float(last_peak) >= float(POST_BIG_WIN_R_THRESHOLD)
                            or last_pnl >= float(POST_BIG_WIN_DOLLAR_THRESHOLD)
                        )
                        closed_dir = (self.executor._directions.get(symbol, "BOTH")
                                      if hasattr(self.executor, "_directions") else "BOTH")
                        if POST_BIG_WIN_COOLDOWN_ENABLED and is_big_win:
                            log.info(
                                "[%s] BIG WIN: peak_r=%.2f pnl=$%.2f → 5h cooldown both directions",
                                symbol, float(last_peak), float(last_pnl))
                            self._arm_cooldown(
                                symbol, int(POST_BIG_WIN_COOLDOWN_SECS),
                                f"BIG_WIN_peak={last_peak:.2f}R_pnl=${last_pnl:.2f}",
                                blocked_direction="BOTH")
                        else:
                            # Small-win path: existing same-direction short cooldown
                            self._arm_cooldown(symbol, COOLDOWN_WIN_SECS,
                                               f"WIN_{closed_dir}_trail_close",
                                               blocked_direction=closed_dir)
                    else:
                        cd_secs = COOLDOWN_LOSS_SECS
                        try:
                            import auto_tuned as _at  # type: ignore
                            cd_secs = getattr(_at, "COOLDOWN_LOSS_OVERRIDE_AUTO", {}).get(
                                symbol, COOLDOWN_LOSS_SECS)
                        except Exception:
                            pass
                        # 2026-05-22 LOSS-STREAK: track this loss + check if
                        # ≥N losses on this symbol within window → extended cooldown.
                        # Today's SWI20 had 5 losses in 12h with no streak guard.
                        try:
                            from config import (
                                LOSS_STREAK_COOLDOWN_ENABLED, LOSS_STREAK_COUNT,
                                LOSS_STREAK_WINDOW_SECS, LOSS_STREAK_COOLDOWN_SECS,
                            )
                        except Exception:
                            LOSS_STREAK_COOLDOWN_ENABLED = False
                            LOSS_STREAK_COUNT, LOSS_STREAK_WINDOW_SECS, LOSS_STREAK_COOLDOWN_SECS = 2, 14400, 18000
                        if LOSS_STREAK_COOLDOWN_ENABLED:
                            if not hasattr(self, "_loss_history"):
                                self._loss_history = {}
                            now_ts = time.time()
                            hist = self._loss_history.setdefault(symbol, [])
                            hist.append(now_ts)
                            # prune entries outside window
                            cutoff = now_ts - float(LOSS_STREAK_WINDOW_SECS)
                            self._loss_history[symbol] = [t for t in hist if t >= cutoff]
                            n_recent = len(self._loss_history[symbol])
                            if n_recent >= int(LOSS_STREAK_COUNT):
                                log.warning(
                                    "[%s] LOSS_STREAK: %d losses in last %dm → %dh BOTH cooldown",
                                    symbol, n_recent,
                                    int(LOSS_STREAK_WINDOW_SECS / 60),
                                    int(LOSS_STREAK_COOLDOWN_SECS / 3600))
                                cd_secs = max(cd_secs, int(LOSS_STREAK_COOLDOWN_SECS))
                                reason = f"LOSS_STREAK_{n_recent}_in_{int(LOSS_STREAK_WINDOW_SECS/60)}m"
                            else:
                                reason = "LOSS_trail_close"
                        else:
                            reason = "LOSS_trail_close"
                        self._arm_cooldown(symbol, cd_secs, reason,
                                           blocked_direction="BOTH")
            except Exception as e:
                log.warning("[%s] Trailing/exit error: %s", symbol, e)

        # ═══ INTELLIGENT EXITS (Dragon ExitIntelligence) — runs before entries
        if self._exit_intelligence:
            try:
                self._exit_intelligence.evaluate_exits()
            except Exception as e:
                log.warning("ExitIntelligence error: %s", e)

        # ═══ PROCESS EACH SYMBOL (entries) — runs AFTER exits ═══
        # Any cooldowns armed by close_position() above will be respected by
        # _process_symbol's gate check. Same-cycle "close + immediate re-entry"
        # race condition (DJ30 2026-05-14 12:30) is fixed by this ordering.
        scores_for_dashboard = {}
        for symbol in SYMBOLS:
            try:
                result = self._process_symbol(symbol, equity, dd_pct, daily_loss_pct)
                if result:
                    self._last_scores[symbol] = result
                    scores_for_dashboard[symbol] = result
            except Exception as e:
                log.error("[%s] Process error: %s", symbol, e)

        # ═══ MARKET OBSERVATION (learning engine watches patterns) ═══
        if self._learning_engine:
            try:
                self._learning_engine.observe_market()
            except Exception:
                pass

        # ═══ ENRICH SCORES (V5 defaults for dashboard) ═══
        for symbol, r in scores_for_dashboard.items():
            r.setdefault("regime", "unknown")
            r.setdefault("m15_dir", "flat")
            r.setdefault("signal_quality", 0)
            r.setdefault("min_quality", 55)

        # ═══ UPDATE DASHBOARD STATE ═══
        self.state.update_agent("cycle", int(self._cycle))
        # Don't overwrite balance — tick streamer sets it from MT5 account_info
        self.state.update_agent("profit", float(equity - self._daily_start_equity))
        self.state.update_agent("dd_pct", float(dd_pct))
        prev_peak = float(self.state.get_agent_state().get("peak_equity", equity))
        new_peak = max(equity, prev_peak)
        self.state.update_agent("peak_equity", float(new_peak))
        if new_peak > prev_peak:
            self._persist_equity_state()
        self.state.update_agent("daily_loss", float(daily_loss_pct))
        self.state.update_agent("weekly_loss", float(weekly_loss_pct))
        self.state.update_agent("kill_switch", {
            "active": False,
            "reason": "",
            "daily_loss_pct": float(daily_loss_pct),
            "weekly_loss_pct": float(weekly_loss_pct),
        })
        self.state.update_agent("positions", self.executor.get_positions_info())
        self.state.update_agent("trade_log", list(self._trade_log[-50:]))
        self.state.update_agent("scores", scores_for_dashboard)

        # MTF intelligence data for dashboard
        if self._mtf:
            mtf_status = {}
            for sym in SYMBOLS:
                try:
                    m = self._mtf.analyze(sym)
                    liq = m.get("liquidity", {})
                    fib = m.get("fibonacci", {})
                    mtf_status[sym] = {
                        "confluence": m.get("confluence", 0),
                        "entry_quality": m.get("entry_quality", 0),
                        "exit_urgency": m.get("exit_urgency", 0),
                        "optimal_sl": m.get("optimal_sl", 0),
                        "optimal_tp": m.get("optimal_tp", 0),
                        "h1_dir": m.get("h1_dir", "FLAT"),
                        "m15_dir": m.get("m15_dir", "FLAT"),
                        "m5_dir": m.get("m5_dir", "FLAT"),
                        "m1_dir": m.get("m1_dir", "FLAT"),
                        # Liquidity zones (real-time)
                        "liquidity_zones": liq.get("zones", [])[:5],
                        "at_liquidity": liq.get("at_liquidity", False),
                        "magnet_above": liq.get("magnet_above", 0),
                        "magnet_below": liq.get("magnet_below", 0),
                        # Fibonacci levels
                        "fib_levels": fib.get("levels", {}),
                        "fib_sl": fib.get("fib_cluster_sl", 0),
                        "fib_tp": fib.get("fib_cluster_tp", 0),
                    }
                    # FVG data
                    if self._fvg:
                        try:
                            h1_df = self.state.get_candles(sym, 60)
                            if h1_df is not None and len(h1_df) > 0:
                                cur_p = float(h1_df["close"].values[-1])
                                fvg_data = self._fvg.get_fvg_signal(sym, m.get("h1_dir", "FLAT"), cur_p)
                                mtf_status[sym]["fvg_active"] = fvg_data.get("active_fvgs", [])[:5]
                                mtf_status[sym]["fvg_bias"] = fvg_data.get("fvg_bias", 0)
                                mtf_status[sym]["has_entry_fvg"] = fvg_data.get("has_entry_fvg", False)
                        except Exception:
                            pass
                except Exception:
                    pass
            self.state.update_agent("mtf_intelligence", mtf_status)

        mode = "hybrid" if self.ml_enabled else "scoring_only"
        if self._master_brain:
            mode = "dragon_" + mode
        self.state.update_agent("mode", mode)

        # ML confidence per symbol
        ml_conf = {}
        if self._meta_model and hasattr(self._meta_model, '_train_metrics'):
            for sym in SYMBOLS:
                met = self._meta_model._train_metrics.get(sym, {})
                ml_conf[sym] = {"auc": met.get("test_auc", 0), "enabled": self.ml_enabled}
        self.state.update_agent("model_confidence", ml_conf)

        # Feature importance
        if self._meta_model and hasattr(self._meta_model, 'feature_importance'):
            self.state.update_agent("feature_importance", dict(self._meta_model.feature_importance))

        # MasterBrain status for dashboard
        if self._master_brain and hasattr(self._master_brain, 'get_status'):
            try:
                self.state.update_agent("master_brain_status", self._master_brain.get_status())
            except Exception as e:
                log.warning("MasterBrain get_status failed: %s", e)

        # Portfolio risk periodic update (correlation matrix + VaR, runs hourly internally)
        if self._master_brain and hasattr(self._master_brain, 'portfolio_risk') and self._master_brain.portfolio_risk:
            try:
                self._master_brain.portfolio_risk.periodic_update()
            except Exception:
                pass

        # Equity history for curve
        eq_hist = list(self.state.get_agent_state().get("equity_history", []))
        eq_hist.append({"time": time.time(), "equity": float(equity)})
        if len(eq_hist) > 2000:
            eq_hist = eq_hist[-2000:]
        self.state.update_agent("equity_history", eq_hist)

    # ═══════════════════════════════════════════════════════════════
    #  SYMBOL PROCESSING
    # ═══════════════════════════════════════════════════════════════

    def _process_symbol(self, symbol, equity, dd_pct, daily_loss_pct):
        """
        V5 Entry Logic — 4 clean phases: SIGNAL → GATES → RISK → EXECUTE.
        Returns score info dict for dashboard, or None.
        """
        from config import (
            SIGNAL_QUALITY_DIVISOR, SIGNAL_QUALITY_THRESHOLDS,
            MTF_OVERRIDE_QUALITY, CONVICTION_SIZING_V2,
            DIRECTION_BIAS, TOXIC_HOURS_UTC, TOXIC_HOUR_EXEMPT,
            SYMBOL_RISK_CAP, PULLBACK_ENTRY_ENABLED,
            PULLBACK_ATR_RETRACE, PULLBACK_MAX_WAIT_BARS,
            EVAL_ON_CANDLE_CLOSE, PRIMARY_TF,
        )
        try:
            from config import TOXIC_HOURS_PER_SYMBOL
        except ImportError:
            TOXIC_HOURS_PER_SYMBOL = {}
        cfg = SYMBOLS[symbol]
        hour_utc = int(datetime.now(timezone.utc).hour)

        # Helper: build return dict with standard fields
        def _ret(ls, ss, sq, mq, direction, gate, **extra):
            d = {"long_score": ls, "short_score": ss,
                 "signal_quality": sq, "min_quality": mq,
                 "direction": direction, "gate": gate}
            d.update(extra)
            return d

        # ══════════════════════════════════════════
        #  PRE-CHECK: Pullback fill (deferred signal)
        # ══════════════════════════════════════════
        if symbol in self._pending_pullback and not self.executor.has_position(symbol):
            pb = self._pending_pullback[symbol]
            pb["bars_waited"] += 1
            tick = self.state.get_tick(symbol)
            cur_price = float(tick.bid) if tick and hasattr(tick, 'bid') else 0
            if cur_price > 0:
                filled = ((pb["direction"] == "LONG" and cur_price <= pb["entry_target"]) or
                          (pb["direction"] == "SHORT" and cur_price >= pb["entry_target"]))
                if filled:
                    log.info("[%s] PULLBACK FILLED: %s at %.5f (signal=%.5f)",
                             symbol, pb["direction"], cur_price, pb["signal_price"])
                    d, rs, rp, sa = pb["direction"], pb["score"], pb["risk_pct"], pb["atr"]
                    comp_l, comp_s = pb["comp_long"], pb["comp_short"]
                    self._pending_pullback.pop(symbol)
                    self._delete_pending_pullback(symbol)
                    success = self.executor.open_trade(symbol, d, sa, risk_pct=rp, score=rs, regime=pb.get("regime"))
                    if success:
                        self._log_trade(symbol, d, rs, "ENTRY_PULLBACK")
                        self._last_entry_bar[symbol] = int(time.time() // 3600)
                        ep = self.executor._entry_prices.get(symbol, 0)
                        if self._alerter is not None:
                            try:
                                self._alerter.position_open(symbol, d, float(rp), float(ep))
                            except Exception:
                                pass
                        self._entry_metadata[symbol] = {
                            "score": float(rs), "regime": pb.get("regime", ""),
                            "direction": d, "entry_price": float(ep),
                            "risk_pct": float(rp), "m15_dir": pb.get("m15_dir", "FLAT"),
                            "meta_prob": pb.get("meta_prob", 0.0),
                            "score_components": comp_l if d == "LONG" else comp_s,
                            "ts": time.time(),
                        }
                        self.state.update_agent("entry_metadata", dict(self._entry_metadata))
                        self._persist_entry_metadata(symbol, self._entry_metadata[symbol])
                    return _ret(0, 0, pb.get("signal_quality", 0), 0, d,
                                "PULLBACK_ENTERED" if success else "PULLBACK_FAILED")
                # 2026-05-22: per-symbol pullback wait override
                try:
                    from config import PULLBACK_MAX_WAIT_BARS_PER_SYMBOL as _PB_WAIT_PS
                    _pb_wait_eff = int(_PB_WAIT_PS.get(symbol, PULLBACK_MAX_WAIT_BARS))
                except Exception:
                    _pb_wait_eff = PULLBACK_MAX_WAIT_BARS
                if pb["bars_waited"] >= _pb_wait_eff:
                    # 2026-05-16: fallback to direct entry instead of skipping.
                    # Old behavior (skip) caused 136/136 expiry rate that disabled the
                    # feature. Mirror backtest's pullback-or-signal-close semantics so
                    # live↔backtest converge. Per feedback_no_skip_trades: never miss
                    # a signal — at worst take it at a slightly later price.
                    log.info("[%s] PULLBACK EXPIRED after %d bars — fallback to direct entry",
                             symbol, pb["bars_waited"])
                    d, rs, rp, sa = pb["direction"], pb["score"], pb["risk_pct"], pb["atr"]
                    comp_l, comp_s = pb["comp_long"], pb["comp_short"]
                    self._pending_pullback.pop(symbol)
                    self._delete_pending_pullback(symbol)
                    success = self.executor.open_trade(symbol, d, sa, risk_pct=rp, score=rs, regime=pb.get("regime"))
                    if success:
                        self._log_trade(symbol, d, rs, "ENTRY_PULLBACK_FALLBACK")
                        self._last_entry_bar[symbol] = int(time.time() // 3600)
                        ep = self.executor._entry_prices.get(symbol, 0)
                        if self._alerter is not None:
                            try:
                                self._alerter.position_open(symbol, d, float(rp), float(ep))
                            except Exception:
                                pass
                        self._entry_metadata[symbol] = {
                            "score": float(rs), "regime": pb.get("regime", ""),
                            "direction": d, "entry_price": float(ep),
                            "risk_pct": float(rp), "m15_dir": pb.get("m15_dir", "FLAT"),
                            "meta_prob": pb.get("meta_prob", 0.0),
                            "score_components": comp_l if d == "LONG" else comp_s,
                            "ts": time.time(),
                        }
                        self.state.update_agent("entry_metadata", dict(self._entry_metadata))
                        self._persist_entry_metadata(symbol, self._entry_metadata[symbol])
                    return _ret(0, 0, pb.get("signal_quality", 0), 0, d,
                                "PULLBACK_FALLBACK" if success else "PULLBACK_FAILED")
            return _ret(0, 0, pb.get("signal_quality", 0), 0,
                        pb["direction"], "PULLBACK_WAIT",
                        pullback_target=pb["entry_target"], bars_waited=pb["bars_waited"])

        # ══════════════════════════════════════════
        #  PRE-CHECK: Detect broker-side close → arm cooldown
        # ══════════════════════════════════════════
        # ALWAYS pop the marker once consumed — independent of cooldown state.
        # Earlier bug: pop was gated on `symbol not in self._sl_cooldown`, so
        # after the first cooldown was set, subsequent closes never re-armed
        # and the marker stuck around forever (USDCAD re-entered 1s after TP).
        ext_closes = getattr(self.executor, '_external_close_time', {})
        ext_time = ext_closes.pop(symbol, 0) if ext_closes else 0
        if ext_time > 0:
            self._last_close_time[symbol] = ext_time
            # 2026-05-21: forward executor's immediately-journaled external
            # close to RL learner so absent_loss_count + score_weights update.
            try:
                ext_pnl_map = getattr(self.executor, "_external_close_pnl", {}) or {}
                if symbol in ext_pnl_map and self._rl_learner is not None:
                    _pnl = float(ext_pnl_map.pop(symbol, 0))
                    _r = float(getattr(self.executor, "_external_close_r", {}).pop(symbol, 0))
                    _dir = (getattr(self.executor, '_external_close_direction', {}) or {}).get(symbol, "FLAT")
                    _peak_r = float(getattr(self.executor, '_last_close_peak_r', {}).pop(symbol, 0) or 0)
                    _components = (self._entry_metadata.get(symbol, {}) or {}).get("score_components", None)
                    _score = float((self._entry_metadata.get(symbol, {}) or {}).get("score", 0))
                    _regime = str((self._entry_metadata.get(symbol, {}) or {}).get("regime", ""))
                    self._rl_learner.record_outcome(
                        symbol=symbol, direction=_dir, pnl=_pnl, r_multiple=_r,
                        score=_score, regime=_regime,
                        exit_reason="ExternalClose_Immediate",
                        score_components=_components, peak_r=_peak_r,
                    )
                    log.info("[%s] RL recorded external close: pnl=$%.2f r=%.2fR peak=%.2fR",
                             symbol, _pnl, _r, _peak_r)
            except Exception as _e:
                log.debug("[%s] RL external-close record failed: %s", symbol, _e)
            # Asymmetric cooldown: short same-direction-only after WIN, long
            # both-directions after LOSS. Executor exposes both via
            # _external_close_direction + _external_close_was_win.
            ext_dirs = getattr(self.executor, '_external_close_direction', {})
            ext_wins = getattr(self.executor, '_external_close_was_win', {})
            closed_dir = (ext_dirs.pop(symbol, "FLAT") if ext_dirs else "FLAT") or "FLAT"
            was_win = bool(ext_wins.pop(symbol, False) if ext_wins else False)
            if was_win and closed_dir in ("LONG", "SHORT"):
                # WIN: block only the same direction for the short window.
                self._arm_cooldown(symbol, COOLDOWN_WIN_SECS,
                                    f"WIN_{closed_dir}", blocked_direction=closed_dir)
            else:
                # LOSS or unknown: both directions, longer window.
                # 2026-05-13: per-symbol cooldown override from Phase 2 tune
                _cd_secs = COOLDOWN_LOSS_SECS
                try:
                    import auto_tuned as _at  # type: ignore
                    _cd_secs = getattr(_at, "COOLDOWN_LOSS_OVERRIDE_AUTO", {}).get(
                        symbol, COOLDOWN_LOSS_SECS)
                except Exception:
                    pass
                self._arm_cooldown(symbol, _cd_secs,
                                    "LOSS" if closed_dir != "FLAT" else "BROKER_CLOSE",
                                    blocked_direction="BOTH")

        # ══════════════════════════════════════════
        #  PRE-CHECK: re-entry cooldown gate (both-directions only here)
        # ══════════════════════════════════════════
        # If the cooldown is BOTH-direction (post-loss), block early to save
        # cycles. Directional cooldowns (post-win same-direction-only) defer
        # the block to after direction resolution — opposite direction is
        # allowed to proceed and may approve.
        blocked = self._cooldown_blocked.get(symbol, "BOTH")
        active, mins_left, cd_reason = self._cooldown_active(symbol)
        if active and blocked == "BOTH" and not self.executor.has_position(symbol):
            if self._pending_pullback.pop(symbol, None) is not None:
                self._delete_pending_pullback(symbol)
            return _ret(0, 0, 0, 0, "FLAT", "SL_COOLDOWN",
                        cooldown_mins=round(mins_left, 1))

        # 2026-05-14: just-closed guard — block re-entry for 180 seconds
        # after ANY close on this symbol. Was 30s; bumped after GAS-Cr re-entered
        # 33s after GuardianSharpLoss close → second close at HardDollarCap for
        # -$24 (total -$36 in 4 min). The 30s window only covered same-cycle
        # races; sharp-loss patterns need a real cooldown that doesn't depend on
        # brain._arm_cooldown firing in time.
        just_closed_ts = getattr(self.executor, "_just_closed", {}).get(symbol, 0)
        JUST_CLOSED_HOLD_S = 180.0
        if just_closed_ts and (time.time() - just_closed_ts) < JUST_CLOSED_HOLD_S:
            secs_left = JUST_CLOSED_HOLD_S - (time.time() - just_closed_ts)
            return _ret(0, 0, 0, 0, "FLAT", "JUST_CLOSED",
                        cooldown_mins=round(secs_left / 60, 2))

        # ══════════════════════════════════════════════
        #  PHASE 1: SIGNAL (raw H1 score, no blending)
        # ══════════════════════════════════════════════

        # 1a. Get H1 candles
        h1_df = self.state.get_candles(symbol, 60)
        if h1_df is None or len(h1_df) < H1_MIN_BARS:
            return _ret(0, 0, 0, 0, "FLAT", "NO_H1_DATA")

        # 1b. Candle-close gate: skip re-scoring if candle hasn't changed
        if EVAL_ON_CANDLE_CLOSE and not self.executor.has_position(symbol):
            primary_df = self.state.get_candles(symbol, PRIMARY_TF)
            if primary_df is not None and len(primary_df) > 0:
                latest_time = float(primary_df["time"].iloc[-1].timestamp()
                                    if hasattr(primary_df["time"].iloc[-1], "timestamp")
                                    else primary_df["time"].iloc[-1])
                prev_time = self._last_candle_time.get(symbol, 0)
                if latest_time == prev_time:
                    return self._last_scores.get(symbol)
                self._last_candle_time[symbol] = latest_time

        # 1c. Compute raw H1 scores (NO MTF blending, NO multipliers)
        ind = self._get_indicators(symbol, h1_df)
        if ind is None:
            return None
        n = int(ind["n"])
        bi = n - 2
        if bi < 21 or np.isnan(ind["at"][bi]) or float(ind["at"][bi]) == 0.0:
            return _ret(0, 0, 0, 0, "FLAT", "INSUFFICIENT_IND")

        # Detect regime FIRST so we can pass it to the RL weight lookup —
        # per-regime cells override the global weight when learned.
        regime = self._get_regime_from_bbw(ind, bi)

        # Pass per-symbol+regime learned component weights so the RL system
        # actually influences scoring (was a no-op for the life of the project).
        rl_weights = None
        if self._rl_learner is not None:
            try:
                rl_weights = self._rl_learner.get_weights(symbol, regime=regime)
            except Exception:
                rl_weights = None
        long_score, short_score, comp_long, comp_short = _score_with_components(
            ind, bi, weights=rl_weights)
        long_score = float(long_score)
        short_score = float(short_score)
        atr_val = float(ind["at"][bi])

        # 1d. Normalize to 0-100 scale
        raw_score = max(long_score, short_score)
        signal_quality = min(100.0, raw_score / SIGNAL_QUALITY_DIVISOR * 100)

        try:
            from config import SIGNAL_QUALITY_SYMBOL
            sym_q = SIGNAL_QUALITY_SYMBOL.get(symbol, {})
            min_quality = float(sym_q.get(regime, SIGNAL_QUALITY_THRESHOLDS.get(regime, 45)))
        except Exception:
            min_quality = float(SIGNAL_QUALITY_THRESHOLDS.get(regime, 45))

        # RL ADAPTIVE THRESHOLD: auto-tighten quality bar on bleeding symbols.
        # PF < 0.7 → +10pp pickier; PF 0.7-1.0 → +5pp. No effect when earning.
        # 2026-05-13: proven-edge symbols (vol_min whitelist) bypass q_bonus.
        # They already have MIN_EDGE + EV gates protecting them; stacking the
        # streak/PF tighten on top was over-conservative — observed XAGUSD
        # signal at 50% quality blocked by 55% threshold (base 40% + 15pp
        # streak bonus) for hours. EV gate handles the negative-edge case.
        if self._rl_learner is not None:
            try:
                from config import VOL_MIN_WARN_ONLY_SYMBOLS as _PROVEN_TH
                if symbol not in _PROVEN_TH:
                    bonus = int(self._rl_learner.get_quality_threshold_bonus(symbol))
                    if bonus > 0:
                        min_quality = min(95.0, min_quality + bonus)
            except Exception:
                pass

        # 1f. Direction from higher score
        if long_score >= short_score and signal_quality >= min_quality:
            direction = "LONG"
            raw_score = long_score
        elif short_score > long_score and signal_quality >= min_quality:
            direction = "SHORT"
            raw_score = short_score
        else:
            direction = "FLAT"
            self._log_decision(symbol, long_score, short_score,
                               "FLAT", "BELOW_MIN", None, None,
                               "FLAT (quality %.0f%% < %.0f%%, regime=%s)" % (signal_quality, min_quality, regime))
            return _ret(long_score, short_score, signal_quality, min_quality,
                        "FLAT", "BELOW_MIN_SCORE", atr=atr_val, regime=regime)

        # 1g. CONFIRMATION GATE (2026-05-16) — trending regime requires at least
        # one of {supertrend, breakout, trend_persist} to be > 0. BTC trade #753
        # (2026-05-15 20:35 UTC) was a score-7.89 SHORT entry in trending regime
        # with ALL THREE of those confirming signals at 0. ema_stack/structure/
        # candle_pattern (lagging signals) had carried the raw score. Lost -3R
        # vs intended -1R SL because price reversed *into* the entry zone before
        # the soft-cut fired. Block this exact pattern in trending regime; do
        # not gate in low_vol/ranging regimes where these indicators legitimately
        # stay silent and entries rely on mean-reversion.
        if regime == "trending":
            _comp_dir = comp_long if direction == "LONG" else comp_short
            try:
                _confirms = sum(
                    1 for _k in ("supertrend", "breakout", "trend_persist")
                    if float(_comp_dir.get(_k, 0) or 0) > 0
                )
            except Exception:
                _confirms = 1  # fail-open on missing component data
            if _confirms == 0:
                self._log_decision(symbol, long_score, short_score, direction,
                                   "CONFIRM_MISSING", None, None,
                                   "trending+no confirms (supertrend/breakout/trend_persist all 0)")
                return _ret(long_score, short_score, signal_quality, min_quality,
                            "FLAT", "CONFIRM_MISSING", atr=atr_val, regime=regime)

        # ══════════════════════════════════════════════
        #  PHASE 2: GATES (7 binary checks)
        # ══════════════════════════════════════════════

        base_ret = dict(long_score=long_score, short_score=short_score,
                        signal_quality=signal_quality, min_quality=min_quality,
                        atr=atr_val, regime=regime)

        # Gate 0a (2026-05-17): NO PYRAMID.
        # Existing cooldown logic at Gate 2a uses `not has_position` so it
        # was SKIPPING the cooldown check when a position was open — which
        # let new entries fire on top of existing ones. Live evidence
        # 2026-05-17: BTC Position A opened SHORT @ 77739.36 (IST 07:30:35);
        # Position B opened SHORT @ 77888.94 (IST 08:26:19, 56 min later)
        # while A was STILL open. Both lost. Cooldown is for post-close
        # re-entry timing; this gate handles the concurrent-position case.
        if self.executor.has_position(symbol):
            self._log_decision(symbol, long_score, short_score, direction,
                               "POSITION_OPEN", None, None,
                               "SKIP (position already open)")
            return {**base_ret, "direction": direction, "gate": "POSITION_OPEN"}

        # Gate 0b (2026-05-17): BAR DEDUP.
        # Block re-entry on the same H1 bar where we already opened a trade.
        # Live per-tick scoring with EVAL_ON_CANDLE_CLOSE=False can fire
        # entry, hit EarlyLossCut, then re-fire on the same bar within
        # minutes once cooldown expires (COOLDOWN_LOSS_SECS=2700s < H1=3600s).
        # Backtest 180d shows 83 BTC trades; live 30d scaled = ~204 → 2.5x
        # inflation — bar-dedup closes that gap structurally.
        cur_h1_bucket = int(time.time() // 3600)
        last_h1_bucket = self._last_entry_bar.get(symbol, 0)
        if last_h1_bucket == cur_h1_bucket:
            self._log_decision(symbol, long_score, short_score, direction,
                               "BAR_REENTRY", None, None,
                               "SKIP (entry already taken on this H1 bar)")
            return {**base_ret, "direction": direction, "gate": "BAR_REENTRY"}

        # Gate 0c (2026-05-17): LATE_MOMENTUM.
        # Block entries where raw_score jumped sharply across cycles. Today's
        # BTC pattern: score 5.5 rejected by EV_REJECT for 5+ min (negative
        # expectancy at that score), then jumped to 7.9 in 2 seconds and
        # bypassed EV via the high-conviction tier. That score-bypass is
        # mathematically correct in isolation, but means we ONLY enter at
        # the late peak — by definition after the move has matured. Both of
        # today's BTC SHORTs entered this way and reversed.
        # Heuristic: if raw_score >= 7.0 NOW but ANY of the last 3 scores
        # was < 6.0, this is a late-confirm trap — block it.
        prev_scores = list(self._score_hist.get(symbol, []))
        self._score_hist[symbol] = (prev_scores + [raw_score])[-5:]
        LATE_HIGH = 7.0
        LATE_LOW = 6.0
        if (raw_score >= LATE_HIGH and prev_scores and
                any(s < LATE_LOW for s in prev_scores[-3:])):
            self._log_decision(symbol, long_score, short_score, direction,
                               "LATE_MOMENTUM", None, None,
                               "SKIP (raw_score %.1f after recent <%.1f — late confirm)"
                               % (raw_score, LATE_LOW))
            return {**base_ret, "direction": direction, "gate": "LATE_MOMENTUM"}

        # Gate 1: Session hours (non-crypto)
        if cfg.category != "Crypto":
            sess_start, sess_end = SYMBOL_SESSION_OVERRIDE.get(
                symbol, (SESSION_START_UTC, SESSION_END_UTC))
            if hour_utc < sess_start or hour_utc >= sess_end:
                return {**base_ret, "direction": direction, "gate": "SESSION"}

        # Gate 2: Direction bias — now ADAPTIVE.
        # 2026-05-14: was hard-reject. Live evidence (474 DIR_BIAS rejects /
        # 0 forex trades today) showed static bias blocks symbols when market
        # regime flips. Now:
        #   1. Compute rolling 30-trade WR for OPPOSITE direction from RL DB
        #   2. If opposite WR >= 60% over 10+ trades → override bias (market
        #      reversed, opposite direction is now winning)
        #   3. Else if signal quality >= 75 → A+ bypass also override (high
        #      conviction trumps static bias)
        #   4. Else block as before
        # 2026-05-17: per-(symbol, regime) direction bias takes precedence.
        # 'BOTH' explicitly means no restriction in this regime even if the
        # symbol-level DIRECTION_BIAS says otherwise.
        try:
            from config import DIRECTION_BIAS_REGIME
        except Exception:
            DIRECTION_BIAS_REGIME = {}
        _regime_bias = DIRECTION_BIAS_REGIME.get(symbol, {}).get(regime)
        if _regime_bias == "BOTH":
            allowed_dir = None
        elif _regime_bias in ("LONG", "SHORT"):
            allowed_dir = _regime_bias
        else:
            allowed_dir = DIRECTION_BIAS.get(symbol)
        if allowed_dir and direction != allowed_dir:
            override = False
            override_reason = ""
            # Check 1: rolling WR of OPPOSITE direction
            try:
                if self._rl_learner is not None:
                    outcomes = self._rl_learner._trade_outcomes.get(symbol, [])
                    if len(outcomes) >= 10:
                        recent = outcomes[-30:]
                        # Filter by direction. Note: outcomes don't always have 'dir'
                        # field; use 'direction' or fallback to all
                        sig_dir = "LONG" if direction == "LONG" else "SHORT"
                        same_dir = [o for o in recent
                                    if o.get("direction", sig_dir) == sig_dir]
                        if len(same_dir) >= 10:
                            wr = sum(1 for o in same_dir if o.get("won")) / len(same_dir)
                            if wr >= 0.60:
                                override = True
                                override_reason = f"rolling WR {wr:.0%} on {sig_dir}"
            except Exception:
                pass
            # Check 2: A+ signal override
            if not override and signal_quality >= 75.0:
                override = True
                override_reason = f"A+ quality {signal_quality:.0f}%"
            if not override:
                self._log_decision(symbol, long_score, short_score,
                                   direction, "DIR_BIAS", None, None,
                                   "SKIP (%s only %s)" % (symbol, allowed_dir))
                return {**base_ret, "direction": direction, "gate": "DIR_BIAS"}
            else:
                log.info("[%s] DIR_BIAS OVERRIDE (%s vs %s): %s",
                         symbol, direction, allowed_dir, override_reason)

        # Gate 2a: Directional cooldown (post-win same-direction-only).
        # The early both-directions cooldown was already checked above; this
        # is the directional variant — block only when our resolved direction
        # matches the blocked side.
        dir_active, dir_mins, dir_reason = self._cooldown_active(symbol, direction=direction)
        if dir_active and not self.executor.has_position(symbol):
            self._log_decision(symbol, long_score, short_score,
                               direction, "COOLDOWN", None, None,
                               "SKIP (%s %dmin)" % (dir_reason, int(dir_mins)))
            return {**base_ret, "direction": direction, "gate": "COOLDOWN"}

        # Gate 2b: RL skip-entry — only fires when RL has strong negative evidence
        # (≥8 trades in this regime/hour with WR < 15%). Self-gates by lookback so
        # symbols without enough history pass through cleanly.
        if self._rl_learner is not None:
            try:
                dir_int = 1 if direction == "LONG" else -1
                rl_skip, rl_reason = self._rl_learner.should_skip_entry(
                    symbol, regime, hour_utc, dir_int)
                if rl_skip:
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "RL_SKIP", None, None,
                                       "SKIP (%s)" % rl_reason)
                    return {**base_ret, "direction": direction,
                            "gate": "RL_SKIP", "rl_reason": rl_reason}
            except Exception as e:
                log.debug("[%s] RL skip-entry error: %s", symbol, e)

        # Gate 3: Toxic hours
        exempt = TOXIC_HOUR_EXEMPT.get(symbol, set())
        per_sym_toxic = TOXIC_HOURS_PER_SYMBOL.get(symbol, set())
        global_toxic = (hour_utc in TOXIC_HOURS_UTC and hour_utc not in exempt)
        sym_toxic = hour_utc in per_sym_toxic
        if global_toxic or sym_toxic:
            label = "TOXIC_HOUR_SYM" if sym_toxic else "TOXIC_HOUR"
            self._log_decision(symbol, long_score, short_score,
                               direction, label, None, None,
                               "SKIP (H%02d toxic)" % hour_utc)
            return {**base_ret, "direction": direction, "gate": label}

        # Gate 3b: News calendar (high-impact event window).
        # Default: warn-only (per "Never skip trades — warn only" memory rule).
        # Per-symbol opt-in to hard-skip via config.CALENDAR_HARD_BLOCK_SYMBOLS.
        if self._calendar:
            try:
                cal_skip, cal_reason = self._calendar.should_skip_entry(symbol)
                if cal_skip:
                    from config import CALENDAR_HARD_BLOCK_SYMBOLS
                    if symbol in CALENDAR_HARD_BLOCK_SYMBOLS:
                        self._log_decision(symbol, long_score, short_score,
                                           direction, "CALENDAR", None, None,
                                           f"SKIP ({cal_reason})")
                        return {**base_ret, "direction": direction, "gate": "CALENDAR"}
                    log.warning("[%s] news event window (%s) — entering anyway (no-skip rule)",
                                symbol, cal_reason)
            except Exception as e:
                log.debug("[%s] calendar check error: %s", symbol, e)

        # Gate 3d: FVG (Fair Value Gap) confluence. Warn if the H1+M15 FVG bias
        # strongly opposes our direction. Pre-existing FVGDetector that was built
        # but only used by the dashboard. Default warn-only per no-skip rule.
        if self._fvg:
            try:
                fvg_info = self._fvg.get_fvg_signal(symbol,
                                                    direction,
                                                    float(h1_df["close"].iloc[-1]))
                fvg_bias = float(fvg_info.get("fvg_bias", 0.0))
                # bias > 0 = bullish FVG dominance; < 0 = bearish.
                # Misalignment: long signal with bearish bias, or vice versa.
                if abs(fvg_bias) >= 0.5:
                    if (direction == "LONG" and fvg_bias < -0.5) or \
                       (direction == "SHORT" and fvg_bias > 0.5):
                        log.warning("[%s] FVG bias %+.2f opposes %s — entering anyway (no-skip)",
                                    symbol, fvg_bias, direction)
            except Exception as e:
                log.debug("[%s] fvg check error: %s", symbol, e)

        # Gate 3c: Long-term trend filter — H1 EMA(200) as D1-trend proxy.
        # 200 H1 bars ≈ 8.3 days, captures intermediate trend without needing
        # a new D1/H4 candle stream. Reject (or warn) entries counter to the
        # long-term trend. Audit 2026-05-06 estimated ~40% of historical
        # entries were counter-trend.
        try:
            from config import TREND_FILTER_HARD_BLOCK_SYMBOLS
            if len(h1_df) >= 200:
                ema200 = h1_df["close"].ewm(span=200, adjust=False).mean().iloc[-1]
                cur_price = float(h1_df["close"].iloc[-1])
                trend_long = cur_price > ema200
                trend_short = cur_price < ema200
                misaligned = (
                    (direction == "LONG" and trend_short) or
                    (direction == "SHORT" and trend_long)
                )
                if misaligned:
                    if symbol in TREND_FILTER_HARD_BLOCK_SYMBOLS:
                        self._log_decision(symbol, long_score, short_score,
                                           direction, "TREND_FILTER", None, None,
                                           f"SKIP (counter-trend vs H1 EMA200 {ema200:.4f})")
                        return {**base_ret, "direction": direction, "gate": "TREND_FILTER"}
                    log.warning("[%s] %s counter to H1 EMA200 trend — entering anyway (no-skip rule)",
                                symbol, direction)
        except Exception as e:
            log.debug("[%s] trend filter error: %s", symbol, e)

        # Gate 3d: MTF CASCADE (W1 + D1 + H4 alignment — sniper grade)
        # Aggregated from H1 candles (no separate data feed needed). Reject
        # if 2+ higher TFs oppose entry. Track verdict for risk multiplier.
        mtf_verdict = "OK"
        mtf_aligned = 0
        try:
            from config import MTF_CASCADE_ENABLED
            if MTF_CASCADE_ENABLED:
                from signals.mtf_trend import mtf_cascade
                h1_df_for_mtf = self.state.get_candles(symbol, 60)
                if h1_df_for_mtf is not None and len(h1_df_for_mtf) >= 30:
                    # 2026-05-14 audit fix: W1 (168 H1 bars/period) needs ≥200 H1
                    # bars to be meaningful; <200 reduces W1 to FLAT and silently
                    # passes counter-trend. Run reduced cascade (D1, H4 only) when
                    # buffer is short — and log it.
                    bars = len(h1_df_for_mtf)
                    tfs_used = ("W1", "D1", "H4") if bars >= 200 else ("D1", "H4")
                    if bars < 200:
                        log.debug("[%s] MTF cascade reduced to %s (only %d H1 bars)",
                                  symbol, tfs_used, bars)
                    mtf = mtf_cascade(h1_df_for_mtf, direction, tfs=tfs_used)
                    mtf_verdict = mtf["verdict"]
                    mtf_aligned = mtf["aligned"]
                    if mtf_verdict == "REJECT":
                        self._log_decision(symbol, long_score, short_score,
                                           direction, "MTF_CASCADE", None, None,
                                           f"SKIP (MTF opposed: {mtf['tfs']})")
                        return {**base_ret, "direction": direction,
                                "gate": "MTF_CASCADE", "mtf_verdict": "REJECT"}
        except Exception as e:
            log.debug("[%s] MTF cascade error: %s", symbol, e)

        # Gate 3b: Range-extreme filter (2026-05-14)
        # In RANGING regime, reject SHORT near range LOW / LONG near range HIGH.
        # 2026-05-21: tried extending to all regimes after live SWI20 top-buy
        # losses, but 90d sweep showed EVERY (lookback, buffer) combo HURT
        # SWI20 PnL ($195 → $29-$106) — winning trades enter near tops too.
        # Reverted to ranging-only.
        if regime == "ranging":
            rf_lookback = 48
            rf_buffer = 0.5
            try:
                import auto_tuned as _at  # type: ignore
                params = getattr(_at, "RANGE_FILTER_PARAMS_AUTO", {}).get(symbol)
                if params:
                    rf_lookback = params.get("lookback", 48)
                    rf_buffer = params.get("buffer_atr", 0.5)
                else:
                    rf_lookback = None
            except Exception:
                rf_lookback = None
            if rf_lookback is not None:
                at_extreme, dist_ratio = self._is_at_range_extreme(
                    ind, bi, direction, lookback=rf_lookback, buffer_atr=rf_buffer)
                if at_extreme:
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "RANGE_EXTREME", None, None,
                                       "SKIP (%s at range extreme — %.2f ATR from boundary)"
                                       % (direction, dist_ratio))
                    return {**base_ret, "direction": direction, "gate": "RANGE_EXTREME"}

        # Gate 3b2: VWAP-SIDE FILTER (2026-05-22 from research #03 + per-sym tune)
        # Per-symbol buffer override; 0.0 = disable.
        try:
            from config import VWAP_BUFFER_PER_SYMBOL as _VW_BUF_PS
        except Exception:
            _VW_BUF_PS = {}
        _vw_buf_mult = float(_VW_BUF_PS.get(symbol, 0.5))
        try:
            vw_arr = ind.get("vwap") if isinstance(ind, dict) else None
            at_arr = ind.get("at") if isinstance(ind, dict) else None
            if _vw_buf_mult > 0 and vw_arr is not None and at_arr is not None and bi < len(vw_arr):
                vw = float(vw_arr[bi])
                atr_v = float(at_arr[bi])
                if not (vw != vw) and atr_v > 0:  # NaN check via x!=x
                    cur_c = float(ind["c"][bi])
                    atr_buf = atr_v * _vw_buf_mult
                    if direction == "LONG" and cur_c <= (vw - atr_buf):
                        self._log_decision(symbol, long_score, short_score,
                                           direction, "VWAP_WRONG_SIDE", None, None,
                                           "SKIP (LONG %.5f below VWAP %.5f - 0.5ATR %.5f)"
                                           % (cur_c, vw, atr_buf))
                        return {**base_ret, "direction": direction, "gate": "VWAP_WRONG_SIDE"}
                    if direction == "SHORT" and cur_c >= (vw + atr_buf):
                        self._log_decision(symbol, long_score, short_score,
                                           direction, "VWAP_WRONG_SIDE", None, None,
                                           "SKIP (SHORT %.5f above VWAP %.5f + 0.5ATR %.5f)"
                                           % (cur_c, vw, atr_buf))
                        return {**base_ret, "direction": direction, "gate": "VWAP_WRONG_SIDE"}
        except Exception:
            pass

        # Gate 3c: FIB ZONE FILTER (2026-05-14 PHASE 6)
        # Per-symbol — only active for symbols where 5-fold WF proved benefit.
        # Currently: COPPER-Cr [0.382, 0.786], SWI20.r [0.5, 0.65].
        # Detects most recent Williams Fractal swing — entry must be inside
        # golden-pocket retracement zone of last significant swing.
        try:
            import auto_tuned as _at  # type: ignore
            fib_params = getattr(_at, "FIB_PARAMS_AUTO", {}).get(symbol)
            if fib_params and fib_params.get("as_filter"):
                in_zone, retr = self._is_in_fib_zone(
                    ind, bi, direction,
                    lookback=fib_params.get("lookback", 50),
                    zone_lo=fib_params.get("zone_lo", 0.5),
                    zone_hi=fib_params.get("zone_hi", 0.618))
                if in_zone is False:  # explicit False = filter triggered; None = no swing
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "FIB_ZONE", None, None,
                                       "SKIP (%s outside fib zone — retr %.2f)"
                                       % (direction, retr or 0))
                    return {**base_ret, "direction": direction, "gate": "FIB_ZONE"}
        except Exception:
            pass

        # Gate 4 (REMOVED 2026-05-22 per entry_pipeline_audit.md finding #4):
        # Position HOLD/REVERSAL branches were unreachable. Gate 0a (line 1450)
        # already returns POSITION_OPEN on `has_position(symbol)`. By the time
        # we reach here, `has_position` must be False — meaning MT5 has no
        # matching-magic positions for this symbol — and so
        # `get_position_direction` always returns "FLAT". The HOLD_SWING and
        # REVERSAL_DISABLED branches were dead and the two extra rpyc
        # `positions_get` round-trips were pure overhead.

        # Gate 5: MTF confirmation (M15 agrees OR high conviction override)
        # 2026-05-13 tightened: was passing any FLAT M15 on trending symbols
        # (Crypto/Index) regardless of quality — covered ~40% of bleeding
        # entries on DJ30/US2000. Now FLAT M15 also requires quality >= 70.
        m15_dir = self._get_m15_direction(symbol)
        m15_agrees = (m15_dir == direction)
        m15_flat = (m15_dir == "FLAT")
        # 2026-05-13: align M15 FLAT bypass with A+ tier (65% threshold)
        m15_pass = (m15_agrees or
                    signal_quality >= MTF_OVERRIDE_QUALITY or
                    (m15_flat and signal_quality >= 65))
        if not m15_pass:
            self._log_decision(symbol, long_score, short_score,
                               direction, "M15_DISAGREE", m15_dir, None,
                               "SKIP (M15=%s != %s)" % (m15_dir, direction))
            return {**base_ret, "direction": direction, "gate": "M15_DISAGREE",
                    "m15_dir": m15_dir}

        # Gate 6: Meta-label ML filter
        # 2026-05-14 audit fix: pass signal_quality (0-100% normalized) instead
        # of raw_score. raw_score is RL-weighted and has no upper bound — a
        # symbol with RL ema_stack=2.5× could trigger the "very high conviction"
        # threshold at quality 60%, while a symbol with depressed weights never
        # reaches it at quality 90%. Quality is the correct conviction signal.
        meta_prob = self._meta_label_check(symbol, direction, ind, bi)
        meta_pass = self._meta_passes(symbol, meta_prob, score=signal_quality)
        if not meta_pass:
            self._log_decision(symbol, long_score, short_score,
                               direction, "META_REJECT", m15_dir, meta_prob,
                               "SKIP (meta=%.2f)" % (meta_prob or 0))
            return {**base_ret, "direction": direction, "gate": "META_REJECT",
                    "m15_dir": m15_dir, "meta_prob": meta_prob}

        # Gate 7: MasterBrain
        # 2026-05-13 Phase 3a: per-symbol Kelly-tuned risk_pct override
        # Walk-forward validated (5-fold) base risk. SYMBOL_RISK_CAP still
        # acts as a ceiling — never go above it.
        risk_pct = SYMBOL_RISK_CAP.get(symbol, MAX_RISK_PER_TRADE_PCT)
        try:
            import auto_tuned as _at  # type: ignore
            kelly_risk = getattr(_at, "SYMBOL_RISK_PCT_OVERRIDE_AUTO", {}).get(symbol)
            if kelly_risk is not None:
                risk_pct = min(float(kelly_risk), risk_pct)
        except Exception:
            pass
        master_info = {}
        if self._master_brain:
            try:
                entry_eval = self._master_brain.evaluate_entry(
                    symbol=symbol, direction=direction, score=raw_score,
                    regime=regime, meta_prob=meta_prob, m15_dir=m15_dir)
                if not entry_eval.get("approved", True):
                    reject = entry_eval.get("reason", "master_reject")
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "MASTER_REJECT", m15_dir, meta_prob,
                                       "SKIP (Master: %s)" % reject)
                    return {**base_ret, "direction": direction, "gate": "MASTER_REJECT",
                            "m15_dir": m15_dir, "meta_prob": meta_prob,
                            "master_reason": reject}
                risk_pct = float(entry_eval.get("risk_pct", risk_pct))
                master_info = entry_eval
            except Exception as e:
                log.warning("[%s] MasterBrain error: %s — default risk", symbol, e)

        # ══════════════════════════════════════════════
        #  PHASE 3: RISK (position sizing)
        # ══════════════════════════════════════════════

        # 3a. Compute all adaptive multipliers (do NOT stack — pick the most
        # conservative). Per real-money rule "no multiplier stacks". Lost $135
        # in a prior incident from compound stacking.
        # Conviction (signal-quality based)
        if signal_quality >= 80:
            conv_mult = CONVICTION_SIZING_V2.get("80+", 1.5)
        elif signal_quality >= 65:
            conv_mult = CONVICTION_SIZING_V2.get("65-80", 1.2)
        elif signal_quality >= 55:
            conv_mult = CONVICTION_SIZING_V2.get("55-65", 1.0)
        else:
            conv_mult = CONVICTION_SIZING_V2.get("<55", 0.6)

        # Session / DOW
        dow = datetime.now(timezone.utc).weekday()
        sess_mult = self._get_session_multiplier(symbol, hour_utc)
        dow_mult = {0: 0.92, 1: 1.05, 2: 1.05, 3: 1.03, 4: 0.90}.get(dow, 1.0)
        sess_dow_mult = sess_mult * dow_mult  # session+DOW treated as one signal

        # RL learner (per-symbol regime/hour win-rate). 1.0 when insufficient data.
        rl_mult = 1.0
        if self._rl_learner is not None:
            try:
                rl_mult = float(self._rl_learner.get_risk_multiplier(symbol, regime, hour_utc))
            except Exception as e:
                log.debug("[%s] RL risk_multiplier error: %s", symbol, e)

        # DE-STACK: take the MIN of the upside multipliers (most conservative).
        # Real-money safety — never compound boosts. Adaptive sizing still works
        # because any single multiplier dropping below 1.0 still reduces risk.
        adaptive_mult = min(conv_mult, rl_mult, sess_dow_mult)
        risk_pct *= adaptive_mult
        log.info("[%s] ADAPTIVE x%.2f = min(conv=%.2f rl=%.2f sess*dow=%.2f) → risk=%.3f%%",
                 symbol, adaptive_mult, conv_mult, rl_mult, sess_dow_mult, risk_pct)

        # PROTECT: equity-DD-aware + loss-streak damper. These ONLY reduce risk
        # (never boost) and protect the account when conditions deteriorate
        # faster than the rolling PF window catches. Critical for survival on
        # a small account where one bad streak can blow capital.
        if self._rl_learner is not None:
            try:
                cur_equity = 0.0
                peak_equity = 0.0
                if self.state is not None:
                    try:
                        ast_ = self.state.get_agent_state()
                        cur_equity = float(ast_.get("equity", 0) or 0)
                        peak_equity = float(ast_.get("peak_equity", 0) or 0)
                    except Exception:
                        pass
                # Bootstrap RL peak with brain's tracked peak so DD scaling is
                # accurate immediately after restart, not after first new high.
                if peak_equity > cur_equity:
                    self._rl_learner.get_equity_dd_multiplier(peak_equity)
                dd_mult = float(self._rl_learner.get_equity_dd_multiplier(cur_equity))
                streak_mult = float(self._rl_learner.get_streak_multiplier(symbol))
                # Edge-score-weighted sizing: never above 1.0 (never boost),
                # scales 0.6x at zero-edge → 1.0x at full-edge. Concentrates
                # capital on symbols with proven recent edge.
                edge_score = float(self._rl_learner.get_edge_score(symbol))
                edge_mult = 0.60 + 0.40 * edge_score
                protect_mult = min(dd_mult, streak_mult, edge_mult)
                if protect_mult < 1.0:
                    risk_pct *= protect_mult
                    log.info("[%s] PROTECT x%.2f (dd=%.2f streak=%.2f edge=%.2f(%.2f) eq=$%.0f) → risk=%.3f%%",
                             symbol, protect_mult, dd_mult, streak_mult, edge_mult,
                             edge_score, cur_equity, risk_pct)
            except Exception as e:
                log.debug("[%s] PROTECT mult error: %s", symbol, e)

        # 3a-PORTFOLIO: HRP + vol-target + VaR-cap. Single multiplier ≤ 1.0
        # that scales risk down for: correlated stacks (HRP), high-vol regimes
        # (vol target 8%/yr), and book-hot conditions (VaR-breach halves).
        if self._master_brain and hasattr(self._master_brain, 'portfolio_risk') \
                and self._master_brain.portfolio_risk is not None:
            try:
                pf_factor = self._master_brain.portfolio_risk.get_portfolio_sizing_factor(
                    symbol, direction)
                if pf_factor.get("factor", 1.0) < 1.0:
                    risk_pct *= float(pf_factor["factor"])
                    log.info("[%s] PORTFOLIO x%.2f (%s) → risk=%.3f%%",
                             symbol, pf_factor["factor"], pf_factor.get("reason", ""), risk_pct)
            except Exception as e:
                log.debug("[%s] portfolio sizing factor error: %s", symbol, e)

        # 3b. DD reduction (downside safety — always applies)
        if dd_pct >= DD_REDUCE_THRESHOLD:
            risk_pct *= 0.5
            log.info("[%s] DD %.1f%% — risk halved to %.3f%%", symbol, dd_pct, risk_pct)

        # 3c. Clamp — also lower the upper bound from x1.5 to x1.0 to prevent
        # any single multiplier path from exceeding configured MAX_RISK_PER_TRADE.
        risk_pct = max(0.1, min(risk_pct, MAX_RISK_PER_TRADE_PCT))

        # ══════════════════════════════════════════════════════════════
        #  GATE: COST + EXPECTED VALUE FILTER (industry-grade pre-trade)
        # ══════════════════════════════════════════════════════════════
        # A+ BYPASS: signal quality >= 75% means the technical setup is
        # exceptionally strong — these are the trades we MUST take.
        # Skip both MIN_EDGE and EV gates for them. Their high score
        # statistically overrides costs and recent EV noise.
        #
        # Tiered policy:
        #   quality >= 75%  → skip MIN_EDGE + EV (A+ — never miss)
        #   quality 65-75%  → skip EV only (good setup, cost still checked)
        #   quality < 65%   → full gate stack
        #
        # Layer A: MIN_EDGE — friction vs SL distance (structural cost cap)
        # Layer B: EV-GATE — expected value vs friction (statistical edge)
        is_aplus = signal_quality >= 75.0
        skip_ev = signal_quality >= 65.0
        if is_aplus:
            log.info("[%s] A+ BYPASS: quality %.0f%% — MIN_EDGE/EV gates skipped",
                     symbol, signal_quality)
        try:
            from config import ATR_SL_MULTIPLIER, SYMBOL_ATR_SL_OVERRIDE, SYMBOL_ATR_SL_OVERRIDE_REGIME
            # 2026-05-17: per-(symbol, regime) SL override first, then per-symbol, then global.
            _regime_sl = SYMBOL_ATR_SL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
            if _regime_sl is not None:
                sl_mult_base = float(_regime_sl)
            else:
                sl_mult_base = float(SYMBOL_ATR_SL_OVERRIDE.get(symbol, ATR_SL_MULTIPLIER))
            sl_dist_est = atr_val * sl_mult_base
            tick = self.state.get_tick(symbol)
            spread = float(tick.ask - tick.bid) if tick and hasattr(tick, 'bid') else 0.0
            friction = spread * 2.5  # entry + exit + slippage buffer
            friction_pct = friction / max(sl_dist_est, 1e-9)
            friction_r = friction / max(sl_dist_est, 1e-9)  # in R units (cost as fraction of 1R move)

            # Layer A: structural cost cap — BYPASS for A+ signals (quality >= 75%)
            #
            # 2026-05-17: tiered friction threshold for high-conviction signals.
            # USDJPY scoring 8.2 LONG was being blocked every cycle (~6K hits in
            # current log) at fixed 25% friction. A-grade signals (raw_score >=
            # MIN_EDGE_HIGH_CONV_SCORE) now use a 1.5x relaxed cap (37.5%) —
            # high score statistically overrides slightly worse cost ratios.
            # Normal-grade signals keep the strict 25% cap.
            from config import (
                MIN_EDGE_FRICTION_PCT,
                MIN_EDGE_FRICTION_PCT_HIGH_CONV,
                MIN_EDGE_HIGH_CONV_SCORE,
            )
            is_high_conv = raw_score >= MIN_EDGE_HIGH_CONV_SCORE
            friction_cap = (MIN_EDGE_FRICTION_PCT_HIGH_CONV
                            if is_high_conv else MIN_EDGE_FRICTION_PCT)
            if not is_aplus and friction_pct > friction_cap:
                tier = "HIGH-CONV" if is_high_conv else "normal"
                self._log_decision(symbol, long_score, short_score,
                                   direction, "MIN_EDGE_REJECT", m15_dir, meta_prob,
                                   "SKIP (friction %.0f%% > %.0f%% of SL — cost > edge, %s tier)"
                                   % (friction_pct * 100, friction_cap * 100, tier))
                return {**base_ret, "direction": direction, "gate": "MIN_EDGE_REJECT",
                        "m15_dir": m15_dir, "meta_prob": meta_prob}
            if is_high_conv and friction_pct > MIN_EDGE_FRICTION_PCT:
                # Would have been blocked under default cap; log the bypass for audit.
                log.info("[%s] MIN_EDGE HIGH-CONV PASS: raw_score=%.1f friction=%.0f%% "
                         "> default %.0f%% but <= %.0f%% cap",
                         symbol, raw_score, friction_pct * 100,
                         MIN_EDGE_FRICTION_PCT * 100, friction_cap * 100)

            # Layer B: statistical EV check — BYPASS for quality >= 65%
            # AND for proven-edge symbols (longer grace period).
            # Sub-65% quality still gets full EV gate (need cost-vs-edge proof).
            from config import VOL_MIN_WARN_ONLY_SYMBOLS as _PROVEN
            if not skip_ev and self._rl_learner is not None:
                ev_r, wr, avg_w, avg_l, n_ev = self._rl_learner.get_expected_value_r(symbol)
                if n_ev >= 15:
                    ev_after_cost = ev_r - friction_r
                    is_proven = symbol in _PROVEN
                    threshold = -0.30 if is_proven else 0.10
                    if ev_after_cost < threshold:
                        self._log_decision(symbol, long_score, short_score,
                                           direction, "EV_REJECT", m15_dir, meta_prob,
                                           "SKIP (EV %.2fR - cost %.2fR = %.2fR < %.2fR; "
                                           "WR=%.0f%% n=%d%s)"
                                           % (ev_r, friction_r, ev_after_cost, threshold,
                                              wr * 100, n_ev, " [proven]" if is_proven else ""))
                        return {**base_ret, "direction": direction, "gate": "EV_REJECT",
                                "m15_dir": m15_dir, "meta_prob": meta_prob}
        except Exception as e:
            log.debug("[%s] MIN_EDGE/EV check error: %s", symbol, e)

        # ══════════════════════════════════════════════
        #  PHASE 4: EXECUTE
        # ══════════════════════════════════════════════

        smart_atr = float(atr_val)

        # Pullback entry:
        #  - Always pullback on re-entry (recently closed → don't chase same move)
        #  - Regime-adaptive for fresh entries: trending/volatile only
        from config import PULLBACK_REGIMES
        last_close = self._last_close_time.get(symbol, 0)
        is_reentry = (time.time() - last_close) < 3600  # closed within last hour
        use_pullback = PULLBACK_ENTRY_ENABLED and (regime in PULLBACK_REGIMES or is_reentry)
        if use_pullback and symbol not in self._pending_pullback:
            tick = self.state.get_tick(symbol)
            signal_price = float(tick.bid) if tick and hasattr(tick, 'bid') else 0
            if signal_price > 0:
                # 2026-05-22 per-symbol pullback retrace override
                try:
                    from config import PULLBACK_ATR_RETRACE_PER_SYMBOL as _PB_ATR_PS
                    _pb_atr_eff = float(_PB_ATR_PS.get(symbol, PULLBACK_ATR_RETRACE))
                except Exception:
                    _pb_atr_eff = PULLBACK_ATR_RETRACE
                retrace = atr_val * _pb_atr_eff
                target = signal_price - retrace if direction == "LONG" else signal_price + retrace
                self._pending_pullback[symbol] = {
                    "direction": direction, "score": raw_score, "atr": smart_atr,
                    "risk_pct": risk_pct, "signal_price": signal_price,
                    "entry_target": target, "bars_waited": 0,
                    "regime": regime, "m15_dir": m15_dir, "meta_prob": meta_prob,
                    "comp_long": comp_long, "comp_short": comp_short,
                    "signal_quality": signal_quality,
                }
                # 2026-05-22 audit finding #3: persist so restart mid-wait
                # doesn't drop deferred signals.
                self._persist_pending_pullback(symbol, self._pending_pullback[symbol])
                log.info("[%s] PULLBACK: %s quality=%.0f%% signal=%.5f target=%.5f",
                         symbol, direction, signal_quality, signal_price, target)
                return {**base_ret, "direction": direction, "gate": "PULLBACK_WAIT",
                        "m15_dir": m15_dir, "meta_prob": meta_prob,
                        "pullback_target": target}

        # Direct entry (fallback if pullback disabled or already pending)
        # 2026-05-12: pass raw_score so executor can scale TP per conviction.
        # 2026-05-17: pass regime so executor can use per-(sym, regime) SL.
        success = self.executor.open_trade(symbol, direction, smart_atr,
                                            risk_pct=risk_pct, score=raw_score,
                                            regime=regime)

        if success:
            self._log_trade(symbol, direction, raw_score, "ENTRY")
            # 2026-05-17: BAR_REENTRY gate uses this to block same-bar duplicates
            self._last_entry_bar[symbol] = int(time.time() // 3600)
            entry_price = self.executor._entry_prices.get(symbol, 0)
            if self._alerter is not None:
                try:
                    self._alerter.position_open(symbol, direction, float(risk_pct), float(entry_price))
                except Exception:
                    pass
            entry_components = comp_long if direction == "LONG" else comp_short
            # 2026-05-22: persist sl_dist + atr_at_entry so trail survives
            # bot restart even if ATR has since shrunk (DJ30 trail-freeze bug).
            _sl_dist_at_entry = float(self.executor._entry_sl_dist.get(symbol, 0)) if hasattr(self.executor, "_entry_sl_dist") else 0.0
            self._entry_metadata[symbol] = {
                "score": float(raw_score), "regime": str(regime),
                "direction": str(direction), "entry_price": float(entry_price),
                "risk_pct": float(risk_pct),
                "m15_dir": str(m15_dir) if m15_dir else "FLAT",
                "meta_prob": float(meta_prob) if meta_prob is not None else 0.0,
                "score_components": entry_components, "ts": time.time(),
                "sl_dist_at_entry": _sl_dist_at_entry,
                "atr_at_entry": float(atr_val) if atr_val else 0.0,
            }
            self.state.update_agent("entry_metadata", dict(self._entry_metadata))
            self._persist_entry_metadata(symbol, self._entry_metadata[symbol])

            log.info("V5 ENTRY: %s %s quality=%.0f%% (raw=%.1f) risk=%.2f%% regime=%s M15=%s",
                     symbol, direction, signal_quality, raw_score, risk_pct, regime, m15_dir)

        return {**base_ret, "direction": direction,
                "gate": "ENTERED" if success else "EXEC_FAILED",
                "m15_dir": m15_dir, "meta_prob": meta_prob,
                "risk_pct": risk_pct, "master_info": master_info}

    # ═══════════════════════════════════════════════════════════════
    #  (V5: old _process_symbol code removed — 700+ lines replaced above)
    # ═══════════════════════════════════════════════════════════════

    def _record_trade_result(self, symbol, reason="unknown"):
        """Record a trade result with MasterBrain when a position closes."""
        # Track close time for re-entry pullback logic
        self._last_close_time[symbol] = time.time()

        if not self._master_brain:
            return

        try:
            # Get PnL info from executor before the position is removed
            # Aggregate across all subs (3-sub architecture)
            positions = self.executor.get_positions_info()
            pnl = float(0.0)
            direction = "FLAT"
            for p in positions:
                if p["symbol"] == symbol and p.get("mode") == "swing":
                    pnl += float(p.get("pnl", 0.0))
                    # type is BUY/SELL from MT5, map to LONG/SHORT
                    side = str(p.get("type", "")).upper()
                    direction = "LONG" if side == "BUY" else "SHORT" if side == "SELL" else direction

            self._master_brain.record_trade_result(
                symbol=symbol,
                direction=direction,
                pnl=pnl,
            )

            # R-multiple = PnL / actual dollar risk on the position.
            # 2026-05-14: was equity × MAX_RISK_PER_TRADE_PCT — the INTENDED max
            # risk before protect/portfolio multipliers shrink it (~$1 on demo).
            # Produced r_mult=-12R for trades that were really -0.7R. Now reads
            # executor._last_close_dollar_risk (snapshotted at close before pop).
            r_mult = 0.0
            try:
                actual_risk = float(getattr(self.executor, "_last_close_dollar_risk", {}).get(symbol, 0) or 0)
                if actual_risk > 0:
                    r_mult = pnl / actual_risk
                else:
                    equity = float(self.state.get_agent_state().get("equity", 1000))
                    dollar_risk = equity * (MAX_RISK_PER_TRADE_PCT / 100.0)
                    r_mult = pnl / dollar_risk if dollar_risk > 0 else 0
                r_mult = max(-10.0, min(10.0, r_mult))  # clamp per 2026-04-29 policy
            except Exception:
                equity = float(self.state.get_agent_state().get("equity", 1000))
                dollar_risk = equity * (MAX_RISK_PER_TRADE_PCT / 100.0)
                r_mult = pnl / dollar_risk if dollar_risk > 0 else 0

            # Record to learning engine for adaptive risk
            if self._learning_engine:
                entry_price = self.executor._entry_prices.get(symbol, 0)
                meta = self._entry_metadata.get(symbol, {})
                entry_score = meta.get("score", 0.0)
                entry_regime = meta.get("regime", "")
                entry_risk = meta.get("risk_pct", 0.0)
                self._learning_engine.record_trade(
                    symbol=symbol, direction=direction, pnl=pnl,
                    entry_price=entry_price, r_multiple=r_mult,
                    exit_reason=reason,
                    score=entry_score,
                    regime=entry_regime,
                    risk_pct=entry_risk,
                )

            # SL-hit cooldown: block re-entry after a losing exit.
            # Routed through _arm_cooldown so the broker-close path and this
            # path can never desync (max-of-existing semantics).
            is_sl_exit = "sl" in reason.lower() or pnl < 0
            if is_sl_exit:
                # SL/manual-loss → block both directions for the full window.
                self._arm_cooldown(symbol, COOLDOWN_LOSS_SECS,
                                   f"SL_HIT(pnl={pnl:.2f},{reason})",
                                   blocked_direction="BOTH")

            # Record to trade intelligence for pattern learning
            if self._trade_intel:
                m15_dir = self._get_m15_direction(symbol)
                regime_now = self._get_regime(symbol)
                # Use the score from the last entry log if available
                last_score = 0.0
                for t in reversed(self._trade_log):
                    if t["symbol"] == symbol and t["action"] == "ENTRY":
                        last_score = t["score"]
                        break
                self._trade_intel.record_pattern(
                    symbol=symbol, direction=direction, score=last_score,
                    regime=regime_now, m15_dir=m15_dir,
                    pnl=pnl, r_multiple=r_mult,
                )

            # Record to RL learner for scoring weight learning
            if self._rl_learner:
                try:
                    regime_now = self._get_regime(symbol)
                    last_score = 0.0
                    last_direction = direction
                    for t in reversed(self._trade_log):
                        if t["symbol"] == symbol and t["action"] == "ENTRY":
                            last_score = t["score"]
                            break
                    # Get cached score components from entry
                    cached_components = self._entry_metadata.get(symbol, {}).get("score_components", None)
                    # Pass peak_r so giveback can be computed and exit-learning fires.
                    # Without this, _maybe_update_exits sees giveback=0 and never tightens trail.
                    peak_r_for_rl = 0.0
                    if self.executor is not None:
                        peak_r_for_rl = float(getattr(self.executor, "_last_close_peak_r", {}).get(symbol, 0.0) or 0.0)
                    self._rl_learner.record_outcome(
                        symbol=symbol, direction=last_direction, pnl=pnl,
                        r_multiple=r_mult, score=last_score, regime=regime_now,
                        exit_reason=reason, score_components=cached_components,
                        peak_r=peak_r_for_rl,
                    )
                except Exception as e:
                    log.debug("[%s] RL learner record failed: %s", symbol, e)

            log.info("[%s] Trade recorded: pnl=%.2f reason=%s", symbol, pnl, reason)
        except Exception as e:
            log.warning("[%s] Record trade result failed: %s", symbol, e)

    # ═══════════════════════════════════════════════════════════════
    #  SCORING — uses momentum_scorer internals
    # ═══════════════════════════════════════════════════════════════

    def _get_indicators(self, symbol, h1_df):
        """Compute or return cached indicators for a symbol's H1 candles."""
        now = time.time()
        cached = self._ind_cache.get(symbol)
        if cached:
            ind, ts = cached
            if now - ts < self._ind_cache_ttl:
                return ind

        # Get per-symbol indicator config
        icfg = dict(IND_DEFAULTS)
        icfg.update(IND_OVERRIDES.get(symbol, {}))

        try:
            ind = _compute_indicators(h1_df, icfg)
            self._ind_cache[symbol] = (ind, now)
            return ind
        except Exception as e:
            log.warning("[%s] Indicator computation failed: %s", symbol, e)
            return None

    # ═══════════════════════════════════════════════════════════════
    #  M15 DIRECTION CHECK
    # ═══════════════════════════════════════════════════════════════

    def _get_m15_direction(self, symbol):
        """
        Determine M15 direction using a lightweight check:
        EMA(15) vs EMA(40) + SuperTrend on M15 candles.
        Returns "LONG", "SHORT", or "FLAT".
        """
        m15_df = self.state.get_candles(symbol, 15)
        if m15_df is None or len(m15_df) < M15_MIN_BARS:
            return "FLAT"

        try:
            close = m15_df["close"].values.astype(np.float64)
            high = m15_df["high"].values.astype(np.float64)
            low = m15_df["low"].values.astype(np.float64)
            n = len(close)

            # EMA cross
            from signals.momentum_scorer import _ema, _supertrend
            ema_s = _ema(close, 15)
            ema_l = _ema(close, 40)

            # SuperTrend direction
            icfg = dict(IND_DEFAULTS)
            icfg.update(IND_OVERRIDES.get(symbol, {}))
            _, st_dir = _supertrend(
                high.copy(), low.copy(), close,
                float(icfg["ST_F"]), int(icfg["ST_ATR"])
            )

            # Last completed bar
            bi = n - 2
            if bi < 1:
                return "FLAT"

            ema_bull = float(ema_s[bi]) > float(ema_l[bi])
            st_bull = int(st_dir[bi]) == 1

            if ema_bull and st_bull:
                return "LONG"
            elif not ema_bull and not st_bull:
                return "SHORT"
            else:
                return "FLAT"
        except Exception as e:
            log.warning("[%s] M15 direction check failed: %s", symbol, e)
            return "FLAT"

    # ═══════════════════════════════════════════════════════════════
    #  M15 REVERSAL EXIT
    # ═══════════════════════════════════════════════════════════════

    def _check_m15_reversal_exit(self, symbol):
        """
        DISABLED 2026-04-21: M15 reversal exit had 0% win rate over 166 trades,
        losing $946 in 7 days. M15 is too noisy — kills winners before
        trailing SL can work. Let trail/SL handle all exits.
        """
        return  # DISABLED — 0% WR, -$946/week across all symbols

        m15_dir = self._get_m15_direction(symbol)
        if m15_dir == "FLAT":
            return

        if (current_dir == "LONG" and m15_dir == "SHORT") or \
           (current_dir == "SHORT" and m15_dir == "LONG"):
            exit_pnl = self._get_position_pnl(symbol)

            # DON'T kill runners — if trade is > 1.5R profit, let trailing SL handle
            # (tightened from 2R — exit faster on smaller wins when M15 flips)
            entry_key = symbol
            sl_dist = self.executor._entry_sl_dist.get(entry_key, 0)
            entry_price = self.executor._entry_prices.get(entry_key, 0)
            if sl_dist > 0 and entry_price > 0:
                tick = self.state.get_tick(symbol)
                if tick:
                    cur = float(tick.bid) if hasattr(tick, 'bid') else tick.get("ltp", 0)
                    if current_dir == "LONG":
                        profit_r = (cur - entry_price) / sl_dist
                    else:
                        profit_r = (entry_price - cur) / sl_dist
                    if profit_r > 1.5:
                        log.debug("[%s] M15 flip but %.1fR profit — trailing SL handles", symbol, profit_r)
                        return  # let trailing SL protect this winner

            log.info("[%s] M15 REVERSAL EXIT: position=%s, M15=%s, pnl=%.2f",
                     symbol, current_dir, m15_dir, exit_pnl)
            closed = self.executor.close_position(symbol, "M15ReversalExit")
            if closed:
                self._record_trade_result(symbol, reason="m15_reversal_exit")
                self._log_trade(symbol, current_dir, 0.0, "M15_EXIT", pnl=exit_pnl)
                if self._alerter is not None:
                    try:
                        self._alerter.position_close(
                            symbol, current_dir, float(exit_pnl), 0.0,
                            "m15_reversal_exit",
                        )
                    except Exception:
                        pass

    # ═══════════════════════════════════════════════════════════════
    #  META-LABEL FILTER
    # ═══════════════════════════════════════════════════════════════

    def _meta_label_check(self, symbol, direction, ind, bi):
        """
        Query ML meta-label model: "Is this signal likely profitable?"
        Returns probability (0-1) or None if model not available.

        Uses SignalModel.build_predict_features() to construct the full
        21-feature vector that matches training, then calls predict().
        """
        if not self.ml_enabled or self._meta_model is None:
            return None

        # Per-symbol ML toggle from backtest optimization
        if not DRAGON_ML_ENABLED.get(symbol, True):
            return None  # ML disabled for this symbol — pure scoring mode

        if not self._meta_model.has_model(symbol):
            return None

        # Check AUC for this specific symbol
        metrics = self._meta_model._train_metrics.get(symbol, {})
        auc = float(metrics.get("test_auc", metrics.get("auc", 0.0)))
        if auc < META_AUC_MIN:
            return None

        try:
            # Build full meta-features using SignalModel.build_predict_features()
            features = self._build_meta_features(symbol, direction, ind, bi)
            if features is None:
                return None

            prediction = self._meta_model.predict(symbol, features)
            if prediction is None:
                return None

            # predict() returns {"confidence": prob, "take_trade": bool, "raw_prob": prob}
            prob = float(prediction.get("confidence", prediction.get("raw_prob", 0.5)))
            return float(prob)
        except Exception as e:
            log.warning("[%s] Meta-label prediction failed: %s", symbol, e)
            return None

    def _meta_passes(self, symbol, meta_prob, score=0):
        """
        Smart ML filter. Scales threshold by signal_quality (0-100%):
        - Quality >= 75 (very high conviction): only block if ML < 0.25
        - Quality 60-75 (high): block if ML < 0.35
        - Quality < 60 (normal): block if ML < 0.43

        2026-05-14 audit fix: thresholds rebased from raw_score (RL-weighted,
        unbounded) to signal_quality (normalized %). Previously a symbol with
        RL ema_stack=2.5× would trigger the "very high conviction" tier at
        quality 60%, while a symbol with depressed weights never reached it
        at quality 90%.

        Also tracks ML block rate — if ML blocks > 80% of signals for a symbol
        over 50+ evaluations, auto-bypasses ML (model is broken for this symbol).
        """
        if meta_prob is None:
            return True  # No model = pure scoring, always pass

        prob = float(meta_prob)

        # Dynamic threshold based on signal quality
        if score >= 75:
            threshold = 0.25  # very high conviction — only block on hard rejection
        elif score >= 60:
            threshold = 0.35
        else:
            threshold = 0.43

        # Track ML block rate per symbol
        if not hasattr(self, '_ml_eval_count'):
            self._ml_eval_count = {}
            self._ml_block_count = {}

        self._ml_eval_count[symbol] = self._ml_eval_count.get(symbol, 0) + 1
        if prob < threshold:
            self._ml_block_count[symbol] = self._ml_block_count.get(symbol, 0) + 1

        # Auto-bypass: if ML blocks > 80% over 50+ evaluations, it's broken
        evals = self._ml_eval_count.get(symbol, 0)
        blocks = self._ml_block_count.get(symbol, 0)
        if evals >= 50 and blocks / evals > 0.80:
            if evals % 100 == 50:  # log once per 100 evals
                log.warning("[%s] ML auto-bypass: blocked %d/%d (%.0f%%) — model too conservative",
                            symbol, blocks, evals, blocks / evals * 100)
            return True  # bypass broken ML

        return prob >= threshold

    def _build_meta_features(self, symbol, direction, ind, bi):
        """
        Build the full 21-feature dict for meta-label prediction using
        SignalModel.build_predict_features(). This ensures the live features
        match exactly what the model was trained on.
        """
        try:
            # Convert direction string to int (+1/-1) for build_predict_features
            dir_int = int(1) if direction == "LONG" else int(-1)

            # Get scores for the current bar
            long_score, short_score = _score(ind, bi)
            long_score = float(long_score)
            short_score = float(short_score)

            # Get the H1 dataframe for time features
            h1_df = self.state.get_candles(symbol, 60)
            if h1_df is None or len(h1_df) < H1_MIN_BARS:
                return None

            # Get M15/M5 candle data for real MTF ML features
            m15_df = self.state.get_candles(symbol, 15)
            m5_df = self.state.get_candles(symbol, 5)

            # Use SignalModel's own feature builder for exact match with training
            features = self._meta_model.build_predict_features(
                symbol=symbol,
                long_score=long_score,
                short_score=short_score,
                direction=dir_int,
                ind=ind,
                bar_i=bi,
                df=h1_df,
                recent_win_streak=int(self._recent_win_streak),
                m15_df=m15_df,
                m5_df=m5_df,
            )

            # Cast all values to float for rpyc safety
            for k in features:
                features[k] = float(features[k])

            return features
        except Exception as e:
            log.warning("[%s] build_predict_features failed: %s", symbol, e)
            return None

    # ═══════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════

    def _get_atr(self, symbol):
        """Get ATR from shared state indicators."""
        ind = self.state.get_indicators(symbol)
        if ind and "atr" in ind:
            return float(ind["atr"])
        return float(0.0)

    def _get_position_pnl(self, symbol):
        """Get total PnL for a symbol's open positions from MT5 (before closing)."""
        try:
            positions = self.executor.get_positions_info()
            total_pnl = sum(float(p.get("pnl", 0)) for p in positions
                           if p["symbol"] == symbol and p.get("mode") == "swing")
            return total_pnl
        except Exception:
            return 0.0

    def _get_regime(self, symbol):
        """Determine market regime from H1 indicators."""
        ind = self.state.get_indicators(symbol)
        if not ind:
            return "unknown"

        adx = float(ind.get("adx", 25))
        st_dir = int(ind.get("supertrend_dir", 0))
        rsi = float(ind.get("rsi", 50))

        if adx > 30:
            if st_dir > 0:
                return "trending_up"
            else:
                return "trending_down"
        elif adx < 15:
            return "ranging"
        else:
            if rsi > 60:
                return "mild_bullish"
            elif rsi < 40:
                return "mild_bearish"
            return "neutral"

    def _get_regime_from_bbw(self, ind, bi):
        """
        Determine regime from Bollinger Band Width at bar index bi.
        Returns one of: "trending", "ranging", "volatile", "low_vol".

        2026-05-14: stricter ranging detection. Was: BBW<1.5 + ADX<20.
        New rule: ADX<22 alone classifies as ranging (regardless of BBW)
        because XAUUSD on 05-13/14 had BBW>1.5 but oscillated 22 pts over
        30h — that's RANGING by behavior even though BBW was nominal.
        Mean-reversion regime triggers different trail / SR-zone logic.
        """
        try:
            bbw_val = float(ind["bbw"][bi])
            if np.isnan(bbw_val):
                return "low_vol"
            adx_val = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 25.0

            # NEW: ADX-first ranging detection
            if adx_val < 22:
                return "ranging"  # weak directional movement = ranging

            if bbw_val < 1.5:
                return "low_vol"  # tight + ADX>=22 = slow drift
            elif bbw_val < 3.0:
                return "trending" if adx_val > 25 else "low_vol"
            elif bbw_val < 5.0:
                return "trending" if adx_val > 30 else "volatile"
            else:
                return "volatile"
        except Exception:
            return "low_vol"

    def _is_at_range_extreme(self, ind, bi, direction, lookback=48, buffer_atr=0.5):
        """RANGE-AWARE FILTER (2026-05-14).
        Returns (True, distance_ratio) if at extreme, else (False, _).
        """
        try:
            highs = ind["h"][max(0, bi - lookback):bi + 1]
            lows = ind["l"][max(0, bi - lookback):bi + 1]
            close = float(ind["c"][bi])
            atr = float(ind["at"][bi])
            if atr <= 0 or len(highs) < 10:
                return False, 0
            recent_high = float(np.max(highs))
            recent_low = float(np.min(lows))
            buf = atr * buffer_atr
            if direction == "LONG" and close >= recent_high - buf:
                return True, (recent_high - close) / atr
            if direction == "SHORT" and close <= recent_low + buf:
                return True, (close - recent_low) / atr
            return False, 0
        except Exception:
            return False, 0

    def _is_in_fib_zone(self, ind, bi, direction, lookback=50,
                         zone_lo=0.5, zone_hi=0.618):
        """FIB ENTRY FILTER (2026-05-14 PHASE 6).

        Detects most recent Williams Fractal swing (5-bar pivot) on entry TF.
        Computes retracement of current close vs swing range.
        Entry must be inside [zone_lo, zone_hi] to pass.

        Returns:
            (True, retr)  — inside fib zone, entry allowed
            (False, retr) — outside zone, entry blocked
            (None, 0)     — couldn't compute (insufficient data / no swing)
        """
        try:
            if bi < lookback + 3:
                return None, 0
            h = ind["h"]; l = ind["l"]; c = ind["c"]
            atr = float(ind["at"][bi])
            close_now = float(c[bi])
            if atr <= 0:
                return None, 0
            swing_hi = swing_lo = None
            swing_hi_idx = swing_lo_idx = None
            for j in range(bi - 3, max(bi - lookback, 2), -1):
                hj = float(h[j]); lj = float(l[j])
                if (swing_hi is None and
                        hj > h[j-1] and hj > h[j-2] and
                        hj > h[j+1] and hj > h[j+2]):
                    swing_hi = hj; swing_hi_idx = j
                if (swing_lo is None and
                        lj < l[j-1] and lj < l[j-2] and
                        lj < l[j+1] and lj < l[j+2]):
                    swing_lo = lj; swing_lo_idx = j
                if swing_hi is not None and swing_lo is not None:
                    break
            if swing_hi is None or swing_lo is None:
                return None, 0
            if (swing_hi - swing_lo) <= 2 * atr:
                return None, 0  # swing too small to matter
            rng = swing_hi - swing_lo
            last_was_high = (swing_hi_idx or 0) > (swing_lo_idx or 0)
            if last_was_high:
                # Up-swing complete; LONG wants entry on retracement DOWN
                retr = (close_now - swing_lo) / rng
                if direction == "LONG":
                    return (zone_lo <= retr <= zone_hi), retr
                # SHORT here is counter-trend — let through (or let other gates handle)
                return None, retr
            else:
                # Down-swing complete; SHORT wants entry on retracement UP
                retr = (swing_hi - close_now) / rng
                if direction == "SHORT":
                    return (zone_lo <= retr <= zone_hi), retr
                return None, retr
        except Exception:
            return None, 0

    # ── SESSION ALPHA MULTIPLIERS (from microstructure audit) ──
    _SESSION_MULTS = {
        "XAUUSD":   {7:1.15, 8:1.15, 13:1.20, 14:1.20, 15:1.10, 10:0.90, 11:0.90, 20:0.85, 21:0.85},
        "XAGUSD":   {7:1.15, 8:1.15, 13:1.20, 14:1.20, 15:1.10, 10:0.90, 11:0.90, 20:0.85, 21:0.85},
        "NAS100.r": {13:1.15, 14:1.20, 15:1.10, 18:0.90, 19:0.90, 20:0.85},
        "JPN225ft": {0:1.15, 1:1.15, 2:1.10, 7:1.10, 8:1.10, 11:0.85, 12:0.85},
        "USDJPY":   {0:1.15, 1:1.15, 2:1.10, 13:1.10, 14:1.10, 10:0.90, 11:0.90, 20:0.85},
    }

    def _get_session_multiplier(self, symbol, hour_utc):
        # Prefer learned multiplier from observer (needs 10+ trades per hour)
        if self._learning_engine:
            learned = self._learning_engine.get_learned_session_mult(symbol, hour_utc)
            if learned != 1.0:
                return learned
        # Fallback to hardcoded microstructure defaults
        table = self._SESSION_MULTS.get(symbol)
        return table.get(hour_utc, 1.0) if table else 1.0

    def _get_adaptive_min_score(self, regime, symbol=None):
        """
        Dragon-level adaptive MIN_SCORE by market regime + per-symbol overrides.
        Per-symbol overrides in DRAGON_SYMBOL_MIN_SCORE take priority.
        Missed-signal feedback: if many near-misses, slightly lower threshold (max -0.5).
        """
        # Check per-symbol override first
        if symbol and symbol in DRAGON_SYMBOL_MIN_SCORE:
            sym_scores = DRAGON_SYMBOL_MIN_SCORE[symbol]
            if regime in sym_scores:
                base = float(sym_scores[regime])
            else:
                base = float(DRAGON_MIN_SCORE_BASELINE)
        else:
            regime_min_scores = {
                "trending": float(6.0),
                "ranging":  float(8.0),
                "volatile": float(7.0),
                "low_vol":  float(7.0),
            }
            base = float(regime_min_scores.get(regime, DRAGON_MIN_SCORE_BASELINE))

        # Missed-signal feedback: ease threshold if many near-misses in this regime
        if self._learning_engine and symbol:
            try:
                missed = self._learning_engine.get_missed_signals(symbol)
                # Count near-misses in current regime from last hour
                now = time.time()
                recent_missed = [m for m in missed
                                 if m.get("regime") == regime and now - m.get("t", 0) < 3600]
                if len(recent_missed) >= 10:
                    # 10+ near-misses in same regime within 1h → lower by 0.5 (max)
                    base -= 0.5
                elif len(recent_missed) >= 5:
                    # 5+ → lower by 0.3
                    base -= 0.3
            except Exception:
                pass

        # Drift-aware tightening: when the live drift_detector flags a symbol,
        # raise the quality bar so we only enter on genuinely strong setups.
        # Layered with the risk-multiplier in master_brain — bleeders get both
        # a smaller size AND a higher score gate.
        if symbol:
            try:
                from agent import drift_detector
                _, drift_state = drift_detector.get_risk_multiplier(symbol)
                if drift_state == "HEAVY":
                    base += 1.5
                elif drift_state == "LIGHT":
                    base += 0.5
            except Exception:
                pass

        # ── MOMENTUM-ADAPTIVE MIN_SCORE DELTA (feature 4, gated) ──
        try:
            from config import (
                MOMENTUM_MIN_SCORE_ADAPTIVE_ENABLED,
                MOMENTUM_MIN_SCORE_FLOOR,
            )
            if MOMENTUM_MIN_SCORE_ADAPTIVE_ENABLED and symbol:
                from signals.momentum_signal import compute_momentum, min_score_delta
                ind = self.state.get_indicators(symbol) if self.state else {}
                df = self.state.get_candles(symbol, 60) if self.state else None
                mom = compute_momentum(ind or {}, df)
                base += min_score_delta(mom)
                # Floor at config-defined floor (always >= 6.0)
                return max(MOMENTUM_MIN_SCORE_FLOOR, base)
        except Exception as e:
            log.debug("momentum min_score_delta failed for %s: %s", symbol, e)

        # Floor the result at the absolute MIN_SCORE 6.0 baseline (memory rule:
        # below 6.0 produced 307 trades / PF~1.0). Drift can only add, never subtract.
        return max(float(DRAGON_MIN_SCORE_BASELINE), base)

    # ═══════════════════════════════════════════════════════════════
    #  LOGGING
    # ═══════════════════════════════════════════════════════════════

    def _log_decision(self, symbol, long_score, short_score,
                      direction, gate, m15_dir, meta_prob, action_str):
        """Structured decision log line."""
        meta_str = ("%.2f" % meta_prob) if meta_prob is not None else "N/A"
        m15_str = str(m15_dir) if m15_dir else "N/A"
        log.info(
            "DECISION: %s | L=%.1f S=%.1f | DIR=%s | M15=%s | META=%s | GATE=%s | %s",
            symbol, float(long_score), float(short_score),
            direction, m15_str, meta_str, gate, action_str
        )
        # ── Dashboard hook (lazy import to avoid circular deps; never blocks) ──
        try:
            from dashboard import v2_api as _v2  # type: ignore
            payload = {
                "ts": time.time(),
                "symbol": str(symbol),
                "long_score": float(long_score),
                "short_score": float(short_score),
                "direction": str(direction),
                "gate": str(gate),
                "reason": str(action_str),
                "m15_dir": m15_str,
                "regime": str(self._get_regime(symbol)) if hasattr(self, "_get_regime") else "",
                "meta_prob": float(meta_prob) if meta_prob is not None else None,
            }
            _v2.push_decision(payload)
        except Exception:
            pass

    def _log_trade(self, symbol, direction, score, action, pnl=None):
        """Log trade for dashboard display and update win streak."""
        # Get real PnL from MT5 positions BEFORE they're closed
        if pnl is None and action != "ENTRY":
            pnl = self._get_position_pnl(symbol)

        entry = {
            "timestamp": str(datetime.now(timezone.utc).strftime("%H:%M:%S")),
            "symbol": str(symbol),
            "direction": str(direction).lower(),
            "score": float(round(score, 1)),
            "action": str(action),
            "pnl": float(round(pnl or 0.0, 2)),
            "regime": str(self._get_regime(symbol)),
        }
        self._trade_log.append(entry)

        # Update win streak from closed positions for meta-label feature
        if action in ("M15_EXIT", "REVERSAL") or action.startswith("INTEL_EXIT"):
            last_pnl = float(pnl or 0.0)
            if last_pnl > 0:
                self._recent_win_streak = int(max(self._recent_win_streak + 1, 1))
            elif last_pnl < 0:
                self._recent_win_streak = int(min(self._recent_win_streak - 1, -1))

        log.info("[%s] Trade logged: %s %s score=%.1f",
                 symbol, action, direction, score)
