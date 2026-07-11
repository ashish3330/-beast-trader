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
from datetime import datetime, timedelta, timezone
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
    STREAK_COOLDOWN_MULT, POST_BIG_WIN_SECS, BIG_WIN_R_TRIGGER,
    ATTEMPT_BACKOFF_BASE_SECS, ATTEMPT_BACKOFF_CAP_SECS,
)

# ── 2026-06-05: NEW entry-side gates (VWAP / 3% daily-kill / news blackout) ──
# Constants land via a parallel agent's config patch. Import defensively so a
# half-deployed config (this file shipped, config not yet) fails OPEN (gates
# disabled) instead of crashing the brain. Once both files are deployed, the
# values from config win.
try:
    from config import VWAP_GATE_ENABLED
except ImportError:
    VWAP_GATE_ENABLED = False
try:
    from config import VWAP_GATE_SYMBOLS
except ImportError:
    VWAP_GATE_SYMBOLS = set()
try:
    from config import DAILY_LOSS_KILL_ENABLED
except ImportError:
    DAILY_LOSS_KILL_ENABLED = False
try:
    from config import DAILY_LOSS_KILL_PCT
except ImportError:
    DAILY_LOSS_KILL_PCT = 3.0
try:
    from config import NEWS_BLACKOUT_ENABLED
except ImportError:
    NEWS_BLACKOUT_ENABLED = False
# 2026-06-16: ICT-style liquidity-sweep gate (Gate 3f) — see config.py.
# Defensive import so a half-deployed config doesn't crash the brain.
try:
    from config import ICT_SWEEP_REQUIRED_FOR_MOMENTUM
except ImportError:
    ICT_SWEEP_REQUIRED_FOR_MOMENTUM = False
try:
    from config import ICT_SWEEP_LOOKBACK_BARS
except ImportError:
    ICT_SWEEP_LOOKBACK_BARS = 24
try:
    from config import ICT_SWEEP_FRACTAL_N
except ImportError:
    ICT_SWEEP_FRACTAL_N = 5
try:
    from config import DISCOUNT_PREMIUM_GATE_ENABLED
except ImportError:
    DISCOUNT_PREMIUM_GATE_ENABLED = False
try:
    from config import DISCOUNT_PREMIUM_LOOKBACK_BARS
except ImportError:
    DISCOUNT_PREMIUM_LOOKBACK_BARS = 60
try:
    from config import DISCOUNT_PREMIUM_STRICT_MODE
except ImportError:
    DISCOUNT_PREMIUM_STRICT_MODE = False

# 2026-06-21: Anchored-VWAP rejection booster (Brian Shannon, 2022).
# Pure-function detector — fails open on missing flag / data. Default OFF.
try:
    from config import ANCHORED_VWAP_BOOSTER_ENABLED
except ImportError:
    ANCHORED_VWAP_BOOSTER_ENABLED = False
try:
    from config import ANCHOR_BARS_DEFAULT as _AVWAP_ANCHOR_BARS
except ImportError:
    _AVWAP_ANCHOR_BARS = 24
try:
    from config import ANCHORED_VWAP_LOOKBACK_BARS as _AVWAP_LOOKBACK
except ImportError:
    _AVWAP_LOOKBACK = 5
try:
    from config import ANCHORED_VWAP_BOOST_AMOUNT as _AVWAP_BOOST_AMT
except ImportError:
    _AVWAP_BOOST_AMT = 1.0
try:
    from agent.expert.anchored_vwap_rejection import (
        evaluate_avwap_booster as _evaluate_avwap_booster,
    )
except Exception:
    _evaluate_avwap_booster = None

# 2026-06-16: ExpertGate orchestrator — sequences the 11 expert components
# (news_v2, range_day, d1_struct, SCSL, OB, Wyckoff, TV, conviction,
# ASAT/dynamic_sltp, setup_invalidator). Defensive import: a half-built
# expert package fails open (orchestrator becomes None → brain bypasses).
try:
    from config import EXPERT_MODE_ENABLED
except ImportError:
    EXPERT_MODE_ENABLED = False
try:
    from agent.expert.orchestrator import ExpertGate as _ExpertGate
except Exception:
    _ExpertGate = None
try:
    from agent.expert import (
        evaluate_setup_invalidations as _evaluate_setup_invalidations,
        enforce_pre_event_flatten as _enforce_pre_event_flatten,
    )
except Exception:
    _evaluate_setup_invalidations = None
    _enforce_pre_event_flatten = None

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
META_AUC_MIN = 0.65              # 2026-06-04 CTO audit B11: was 0.55; live AUCs cluster 0.53-0.68 = barely better than random. All 945 recent meta_prob predictions live in [0.435, 0.559] — models emit near-constants. UK100 emitted 0.446 for 194 trades straight. Raising to 0.65 disables the half-random models; top earners (XAU/DJ30) bypass ML anyway. Retrain to push AUC >0.70 before re-enabling.
H1_MIN_BARS = 100                # minimum H1 bars for scoring
M15_MIN_BARS = 50                # minimum M15 bars for direction check


# ─────────────────────────────────────────────────────────────────────────
# ICT liquidity-sweep helper (2026-06-16) — module-level so it's unit-
# testable from a REPL/script without instantiating the brain.
#
# detect_liquidity_sweep(highs, lows, closes, direction, lookback, n)
#   highs/lows/closes : np.ndarray or list of H1 OHLC values, oldest→newest
#   direction         : "LONG" or "SHORT"
#   lookback          : how many recent bars to scan for a sweep+reclaim
#   n                 : fractal pivot half-width (5 = compare vs prior 5-bar
#                       swing low/high). Sweep candidate bar at index i is
#                       checked against the swing extreme over bars [i-n, i-1].
# Returns True if a stop-hunt-then-reclaim pattern was found within the
# lookback window:
#   LONG  : ∃ i s.t. low[i]  < min(low[i-n..i-1])  AND close[i] > that prior min
#   SHORT : ∃ i s.t. high[i] > max(high[i-n..i-1]) AND close[i] < that prior max
# This is the classic ICT sweep+reclaim pattern (stop hunt below recent
# swing low / above recent swing high, followed by a reclaim close).
# ─────────────────────────────────────────────────────────────────────────
def detect_liquidity_sweep(highs, lows, closes, direction,
                           lookback=24, n=5):
    try:
        import numpy as _np
        h = _np.asarray(highs, dtype=float)
        l = _np.asarray(lows, dtype=float)
        c = _np.asarray(closes, dtype=float)
    except Exception:
        return False
    N = len(c)
    if N < (n + 2) or lookback < 1 or n < 1:
        return False
    # Scan from oldest bar in lookback window to most recent bar.
    # i ranges over the last `lookback` bars but must have at least `n`
    # bars of history available for the swing reference.
    start = max(n, N - lookback)
    if direction == "LONG":
        for i in range(start, N):
            try:
                prior_min = float(l[i - n:i].min())
                if float(l[i]) < prior_min and float(c[i]) > prior_min:
                    return True
            except Exception:
                continue
        return False
    elif direction == "SHORT":
        for i in range(start, N):
            try:
                prior_max = float(h[i - n:i].max())
                if float(h[i]) > prior_max and float(c[i]) < prior_max:
                    return True
            except Exception:
                continue
        return False
    return False


class AgentBrain:
    """Hybrid trading agent: rule-based scoring + optional ML meta-label + MasterBrain gating."""

    def __init__(self, state: SharedState, mt5, executor: Executor,
                 meta_model=None, master_brain=None, exit_intelligence=None,
                 learning_engine=None, mtf_intelligence=None, equity_guardian=None,
                 smart_entry=None, calendar_filter=None, trade_intelligence=None,
                 rl_learner=None, pattern_learner=None, order_flow=None,
                 level_memory=None, fvg_detector=None,
                 fvg_strategy=None, fvg_whitelist=None,
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
        # 2026-06-18 Tier 1 #1: give executor a weak-ref back to the brain so
        # open_trade() can check `_is_strategy_killed()` and `_brain_ref`
        # the per-strategy kill switch. Defensive: only set if executor exists.
        try:
            if executor is not None:
                executor._brain_ref = self
        except Exception:
            pass
        self.running = False
        self._thread = None
        self._cycle = int(0)
        self._mt5_degraded_streak = int(0)   # consecutive cycles failing on transport errors
        self._daily_start_equity = float(0.0)
        self._daily_loss = float(0.0)
        self._last_day = None
        self._trade_log = []

        # ── 2026-06-05: 3% daily-loss kill state (FTMO/FundedNext/Topstep std) ──
        # Sticky per-UTC-day. Reset by _run_cycle daily reset block when
        # `today != self._last_day`. Not DB-persisted (intra-day only; if
        # process restarts mid-day the _compute_daily_loss_pct() helper still
        # reads journal-truthful PnL on first check after restart and re-arms
        # if still below the threshold).
        self._day_kill_fired_today = False
        self._day_kill_until = None

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

        # ── 2026-06-18 Tier 1 #1: per-strategy kill-switch table init ──
        # Independent of the master kill_switch above. Single row per
        # strategy (momentum/fvg/sr). Auto-trip rules driven by
        # MasterBrain._daily_strategy_r (item #6). DEFAULT OFF for first
        # 48h via STRATEGY_KILL_SWITCH_ENABLED so the table populates +
        # auto-trip events log without enforcing.
        self._strategy_kill_cache: Dict[str, bool] = {}
        self._strategy_kill_cache_until: float = 0.0
        try:
            self._init_strategy_kill_switch_table()
        except Exception as _ksk_e:
            log.warning("strategy_kill_switch init failed (fail-open): %s", _ksk_e)

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

        # ── ICT Liquidity-Sweep + FVG-Retest strategy (separate magic range) ──
        # Independent of the momentum book: own SL/TP, own 0.25% base risk
        # (×momentum-ROC size tilt), own time stop. Whitelist-gated to the 7
        # recent-180d-positive symbols. Yields to momentum (skips if the
        # momentum book already holds the symbol) to avoid hedged exposure.
        self._fvg_strategy = fvg_strategy
        self._fvg_whitelist = set(fvg_whitelist or [])
        # Per-symbol cooldown after an FVG close (prevents instant re-fire on
        # the same setup). Keyed symbol -> epoch expiry.
        self._fvg_cooldown: Dict[str, float] = {}
        # Tracks open->closed transitions per symbol so a broker-side TP/SL
        # close (not just our time stop) arms the post-close cooldown.
        self._fvg_was_open: Dict[str, bool] = {}

        # ── 2026-06-16: ExpertGate orchestrator (single-gate wrapper around
        # the 11 expert components — news_v2, range_day, d1_struct, SCSL,
        # OB, Wyckoff, TV, conviction, ASAT/dynamic_sltp, setup_invalidator).
        # Each sub-component is still gated by its own enable flag, so this
        # is a no-op until the user flips the individual flags. The
        # orchestrator slot in _process_symbol fires between Gate 3f and
        # Gate 4 (see EXPERT_MODE block there).
        self._expert_gate = None
        if EXPERT_MODE_ENABLED and _ExpertGate is not None:
            try:
                self._expert_gate = _ExpertGate(brain=self)
                log.info("ExpertGate orchestrator initialised "
                         "(EXPERT_MODE_ENABLED=True)")
            except Exception as e:
                log.warning("ExpertGate init failed (degrading to legacy): %s", e)
                self._expert_gate = None

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
        # 2026-05-29: SharedState.agent_state is IN-MEMORY only (wiped each
        # process start), so the restore above never actually recovers anything
        # across restarts — every restart let symbols re-enter inside an active
        # cooldown. Make cooldowns DB-durable (trade_journal.db) so the 12+ daily
        # restarts no longer wipe them. This also restores the blocked DIRECTION
        # (was defaulting to BOTH on cold start).
        self._init_cooldown_table()
        self._load_cooldowns()
        if self._sl_cooldown:
            active = {s: f"{(v - time.time())/60:.0f}min" for s, v in self._sl_cooldown.items()}
            log.info("Restored cooldowns: %s", active)
        # 2026-05-29 cooldown redesign: per-(symbol,direction) attempt-strike
        # counter for exponential backoff. Increments on a losing close in that
        # direction, resets on a winning close. Restored from state so a restart
        # doesn't reset the backoff mid-cascade.
        _saved_strikes = self.state.get_agent_state().get("attempt_strikes", {})
        self._attempt_strikes: Dict[str, int] = {str(k): int(v) for k, v in _saved_strikes.items()}

        # ── Indicator cache (recompute every cycle per symbol) ──
        self._ind_cache = {}       # symbol -> (indicators_dict, timestamp)
        self._ind_cache_ttl = 0.25  # 250ms cache — near-instant scoring

        # ── Pullback entry: deferred signals waiting for retrace ──
        self._pending_pullback: Dict[str, dict] = {}  # symbol -> {direction, score, atr, risk_pct, signal_price, bars_waited, ...}

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
        self._persist_cooldowns()   # DB-durable so a restart can't wipe it
        # 2026-06-03 CTO audit (A6): ATTEMPT_BACKOFF_* arms were invisible
        # because `if not was_active` suppressed them (they always fire after
        # a LOSS arm). Always log BACKOFF strikes so escalation is observable.
        is_backoff = "ATTEMPT_BACKOFF" in reason
        if not was_active or is_backoff:
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

    # ── Cooldown DB persistence (durable across restarts) ──
    def _init_cooldown_table(self):
        import sqlite3
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cooldowns "
                "(symbol TEXT PRIMARY KEY, expiry REAL NOT NULL, "
                " blocked TEXT NOT NULL, reason TEXT)")
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("cooldown table init failed: %s", e)

    def _load_cooldowns(self):
        """Restore non-expired cooldowns on startup so restarts don't wipe them."""
        import sqlite3
        try:
            now = time.time()
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            rows = conn.execute(
                "SELECT symbol, expiry, blocked, reason FROM cooldowns WHERE expiry > ?",
                (now,)).fetchall()
            conn.execute("DELETE FROM cooldowns WHERE expiry <= ?", (now,))
            conn.commit()
            conn.close()
            for sym, expiry, blocked, reason in rows:
                self._sl_cooldown[sym] = float(expiry)
                self._cooldown_blocked[sym] = str(blocked or "BOTH")
                self._cooldown_reason[sym] = str(reason or "")
        except Exception as e:
            log.warning("cooldown load failed: %s", e)

    def _persist_cooldowns(self):
        """Write current (future) cooldowns to DB. Called on every arm."""
        import sqlite3
        try:
            now = time.time()
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute("DELETE FROM cooldowns")
            for sym, exp in self._sl_cooldown.items():
                if exp > now:
                    conn.execute(
                        "INSERT OR REPLACE INTO cooldowns "
                        "(symbol, expiry, blocked, reason) VALUES (?, ?, ?, ?)",
                        (sym, float(exp), self._cooldown_blocked.get(sym, "BOTH"),
                         self._cooldown_reason.get(sym, "")))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("cooldown persist failed: %s", e)

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
    #  TIER 1 #1 — PER-STRATEGY KILL SWITCH (DB-persisted, cached)
    #
    #  3-row table keyed by strategy (momentum/fvg/sr). Mirrors the
    #  master `kill_switch` table convention above. Auto-trip rules
    #  driven by MasterBrain (per-strategy daily R-cap, item #6) but
    #  the brain owns the cheap per-cycle read.
    # ═══════════════════════════════════════════════════════════════

    _STRATEGY_NAMES = ("momentum", "fvg", "sr")

    def _init_strategy_kill_switch_table(self):
        """3-row strategy_kill_switch table. Idempotent. Honours fail-open.

        Schema:
          strategy           TEXT PRIMARY KEY  ('momentum'|'fvg'|'sr')
          active             INTEGER  0/1
          reason             TEXT
          tripped_at_iso     TEXT     UTC iso8601
          daily_pnl_r        REAL     tracked by MasterBrain
          auto_reset_at_iso  TEXT     UTC iso8601 (NULL = manual reset only)
          updated_at         INTEGER  unix ts
        """
        import sqlite3
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS strategy_kill_switch ("
                " strategy TEXT PRIMARY KEY,"
                " active INTEGER NOT NULL DEFAULT 0,"
                " reason TEXT,"
                " tripped_at_iso TEXT,"
                " daily_pnl_r REAL DEFAULT 0.0,"
                " auto_reset_at_iso TEXT,"
                " updated_at INTEGER NOT NULL"
                ")")
            now = int(time.time())
            for s in self._STRATEGY_NAMES:
                conn.execute(
                    "INSERT OR IGNORE INTO strategy_kill_switch"
                    " (strategy, active, updated_at) VALUES (?, 0, ?)",
                    (s, now))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("strategy_kill_switch init failed: %s", e)

    def _refresh_strategy_kill_cache(self):
        """30s TTL cache prevents per-tick DB hits."""
        import sqlite3
        try:
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            rows = conn.execute(
                "SELECT strategy, active, auto_reset_at_iso FROM strategy_kill_switch"
            ).fetchall()
            conn.close()
        except Exception:
            return  # leave cache stale on error
        now_iso = datetime.now(timezone.utc).isoformat()
        cache: Dict[str, bool] = {}
        for r in rows:
            try:
                strat = str(r[0])
                active = bool(int(r[1] or 0))
                reset_iso = r[2]
                if active and reset_iso and isinstance(reset_iso, str) and reset_iso <= now_iso:
                    active = False  # auto-reset window expired
                cache[strat] = active
            except Exception:
                continue
        self._strategy_kill_cache = cache
        self._strategy_kill_cache_until = time.time() + 30.0

    def _is_strategy_killed(self, strategy: str) -> bool:
        """Cheap (cached) check used by per-strategy entry gates.

        Default OFF for first 48h via STRATEGY_KILL_SWITCH_ENABLED so the
        table populates without enforcement. Returns False in shadow mode
        even if a row is active.

        Honours fail-open: any DB/cache exception returns False.
        """
        try:
            if time.time() > getattr(self, "_strategy_kill_cache_until", 0.0):
                self._refresh_strategy_kill_cache()
            try:
                from config import STRATEGY_KILL_SWITCH_ENABLED as _ENF
            except Exception:
                _ENF = False
            if not _ENF:
                return False  # shadow mode
            return bool(self._strategy_kill_cache.get(strategy, False))
        except Exception:
            return False

    def _trip_strategy_kill(self, strategy: str, reason: str,
                            auto_reset_hrs: float = 6.0) -> None:
        """Trip a strategy kill switch. Called by MasterBrain when per-strategy
        daily R-cap fires. Always safe to call (best-effort DB write)."""
        import sqlite3
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            reset_iso = (datetime.now(timezone.utc) +
                         timedelta(hours=float(auto_reset_hrs))).isoformat()
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=5.0)
            conn.execute(
                "INSERT INTO strategy_kill_switch"
                " (strategy, active, reason, tripped_at_iso, auto_reset_at_iso, updated_at)"
                " VALUES (?, 1, ?, ?, ?, ?)"
                " ON CONFLICT(strategy) DO UPDATE SET"
                "   active=1, reason=excluded.reason,"
                "   tripped_at_iso=excluded.tripped_at_iso,"
                "   auto_reset_at_iso=excluded.auto_reset_at_iso,"
                "   updated_at=excluded.updated_at",
                (strategy, str(reason), now_iso, reset_iso, int(time.time())))
            conn.commit()
            conn.close()
            log.warning("STRATEGY_KILL %s: %s (auto-reset %s)",
                        strategy, reason, reset_iso)
            # Force cache refresh on next read
            self._strategy_kill_cache_until = 0.0
        except Exception as e:
            log.warning("strategy_kill trip failed for %s: %s", strategy, e)

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

            # 2026-06-05: reset 3% daily-loss kill flag at UTC day rollover.
            # New day = fresh budget. Logged so we can audit re-arm timing.
            if self._day_kill_fired_today:
                log.info("DAILY-LOSS KILL RESET — new trading day, halt cleared")
            self._day_kill_fired_today = False
            self._day_kill_until = None

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

        # ════════════════════════════════════════════════════════════════
        # 2026-06-05: 3% DAILY-LOSS KILL — industry-standard prop-firm rule
        # ════════════════════════════════════════════════════════════════
        # Research: FTMO / FundedNext / Topstep all enforce a 3-5% daily loss
        # cap as the bright-line risk control. The legacy DAILY_HARD_STOP_PCT
        # (40%) was effectively off — over the past 14 demo days the
        # EmergencyDD path fired 13× with an avg of -5.29R/day before any
        # halt logic engaged. Independent of the legacy daily kill so we
        # can ship without rewriting that path.
        if DAILY_LOSS_KILL_ENABLED:
            try:
                _day_loss_pct = self._compute_daily_loss_pct()
                if _day_loss_pct >= DAILY_LOSS_KILL_PCT and not self._day_kill_fired_today:
                    log.error(
                        "DAILY-LOSS KILL: -%.2f%% >= -%.1f%% — closing all + halt new entries",
                        _day_loss_pct, DAILY_LOSS_KILL_PCT)
                    try:
                        self.executor.close_all("DailyLossKill_3pct")
                    except Exception as _ce:
                        log.error("DailyLossKill close_all failed: %s", _ce)
                    self._day_kill_fired_today = True
                    # Resume next UTC day — _run_cycle daily-reset block clears
                    # the flag at the day rollover. Stored for dashboard view.
                    from datetime import timedelta as _td
                    self._day_kill_until = now_utc.date() + _td(days=1)
                    try:
                        self.state.update_agent("day_kill_fired_today", True)
                        self.state.update_agent("day_kill_loss_pct", float(_day_loss_pct))
                    except Exception:
                        pass
            except Exception as e:
                log.warning("Daily-loss check failed: %s", e)

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
            # 2026-06-19: include FVG/SR positions — has_position() only covers swing magics,
            # so pure-FVG/pure-SR tickets were never trailed during kill-switch (same root
            # bug as GER40.r SR -5.31R bypass).
            for symbol in SYMBOLS:
                try:
                    if (self.executor.has_position(symbol)
                            or self.executor.has_fvg_position(symbol)
                            or self.executor.has_sr_position(symbol)
                            or self.executor.has_smabo_position(symbol)
                            or self.executor.has_fib50_position(symbol)):
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
        syms_with_pos = None   # book snapshot (None = fetch failed → probe all)
        try:
            broker_positions = self.mt5.positions_get() or []
            syms_with_pos = {getattr(p, "symbol", None) for p in broker_positions}
            for p in broker_positions:
                sym = getattr(p, "symbol", None)
                # 2026-07-08: ONLY augment with symbols that are in SYMBOLS. AUX
                # symbols (trend/IMR: ETH/NAS/JPN/SP500/US2000) must NOT enter the
                # momentum manage loop — manage_trailing_sl's orphan path would
                # match their +6000/+7000 legs and close them with momentum
                # ELC/PeakGiveback logic. They self-manage via _process_trend's
                # Chandelier trail / _process_imr's detector exit.
                if sym and sym in SYMBOLS:
                    manage_symbols.add(sym)
        except Exception:
            pass

        # 2026-06-04: drain stale CLOSE intents (failed order_send retries).
        # Runs once per brain cycle so a CHFJPY-style 3h-delayed close is
        # bounded to ~30s of brain-cycle latency.
        try:
            self.executor.drain_close_intents()
        except Exception as e:
            log.debug("drain_close_intents failed: %s", e)

        # ── 2026-06-16 EXPERT_MODE — pre-event news flatten (T-30m close
        # of tier-1 exposure) + setup-invalidator (structural-fail closes
        # for tagged setups). Both run BEFORE manage_trailing_sl so any
        # flatten/invalidation close completes before the trail loop tries
        # to update SL on the same position. Fail-OPEN: errors degrade
        # silently — the legacy trail + exit_intelligence pipeline catches
        # whatever this layer misses.
        if EXPERT_MODE_ENABLED:
            # Pre-event flatten (news v2)
            try:
                if (_enforce_pre_event_flatten is not None
                        and getattr(self, "_expert_gate", None) is not None
                        and bool(getattr(self.__class__, "_news_flatten_enabled", None)
                                 or True)):
                    from config import NEWS_FLATTEN_ENABLED as _nfe
                    if _nfe:
                        _enforce_pre_event_flatten(self)
            except Exception as e:
                log.debug("news pre-event flatten failed: %s", e)

            # Setup-invalidator — structural fail closes for tagged setups.
            try:
                from config import (SETUP_INVALIDATOR_ENABLED as _sie,
                                    SETUP_INVALIDATOR_LIVE_CLOSE as _silc)
            except Exception:
                _sie = False
                _silc = False
            if _sie and _evaluate_setup_invalidations is not None:
                try:
                    pos_map = {}
                    try:
                        all_pos = self.mt5.positions_get() or []
                        for p in all_pos:
                            sym = getattr(p, "symbol", None)
                            if sym:
                                pos_map[sym] = p
                    except Exception:
                        pos_map = {}

                    def _peak_r_lookup(sym):
                        try:
                            return float(self.executor._peak_profit_r.get(sym, 0.0))
                        except Exception:
                            return None

                    def _get_h1(sym, _tf):
                        try:
                            return self.state.get_candles(sym, 60)
                        except Exception:
                            return None

                    decisions = _evaluate_setup_invalidations(
                        positions=pos_map,
                        entry_metadata=self._entry_metadata,
                        get_candles=_get_h1,
                        peak_profit_r_lookup=_peak_r_lookup,
                        now_sec=int(time.time()),
                    )
                    for sym, decision in (decisions or []):
                        if not decision or not decision.get("close"):
                            continue
                        reason = decision.get("reason", "SETUP_INVAL")
                        if _silc:
                            try:
                                self.executor.close_position(sym, reason=reason)
                                log.info("[%s] SETUP_INVAL CLOSE: %s", sym, reason)
                            except Exception as e:
                                log.warning("[%s] setup-inval close failed: %s",
                                            sym, e)
                        else:
                            log.info("[%s] SETUP_INVAL WARN (live_close=False): %s",
                                     sym, reason)
                except Exception as e:
                    log.debug("setup-invalidator loop failed: %s", e)

        for symbol in manage_symbols:
            try:
                # 2026-07-08: skip the per-symbol has_* bridge probes entirely for
                # symbols the book snapshot shows are FLAT — no position can need
                # trailing. Cuts idle-cycle bridge reads (the contention source).
                # Only applied when the snapshot is trustworthy (fetch succeeded).
                if syms_with_pos is not None and symbol not in syms_with_pos:
                    continue
                # 2026-06-19: include FVG/SR positions. has_position() only matches
                # the swing magic range — pure-FVG/pure-SR tickets used to never
                # enter manage_trailing_sl, causing the GER40.r SR -5.31R full-SL
                # bypass on 2026-06-18. Mirrors the 2028f2a scalp fix.
                had_pos_before = (self.executor.has_position(symbol)
                                  or self.executor.has_fvg_position(symbol)
                                  or self.executor.has_sr_position(symbol)
                                  or self.executor.has_smabo_position(symbol)
                                  or self.executor.has_fib50_position(symbol))
                if had_pos_before:
                    self.executor.manage_trailing_sl(symbol)
                    self._check_m15_reversal_exit(symbol)
                # 2026-05-14: if trail-management closed the position THIS
                # cycle, arm the cooldown immediately so the same-cycle
                # entry loop sees it. Otherwise cooldown arms next cycle
                # via external-close detection — too late.
                # 2026-06-19: re-check with the same OR so cooldown doesn't
                # false-trip when an SR/FVG ticket is still open.
                still_has_pos = (self.executor.has_position(symbol)
                                 or self.executor.has_fvg_position(symbol)
                                 or self.executor.has_sr_position(symbol)
                                 or self.executor.has_smabo_position(symbol)
                                 or self.executor.has_fib50_position(symbol))
                if had_pos_before and not still_has_pos:
                    last_peak = (self.executor._peak_profit_r.get(symbol, 0.0)
                                 if hasattr(self.executor, "_peak_profit_r") else 0.0)
                    was_win = float(last_peak) >= 0.5
                    if was_win:
                        # Same-direction-only short cooldown after a win
                        closed_dir = (self.executor._directions.get(symbol, "BOTH")
                                      if hasattr(self.executor, "_directions") else "BOTH")
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
                        self._arm_cooldown(symbol, cd_secs,
                                           "LOSS_trail_close", blocked_direction="BOTH")
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
        # 2026-06-05 PM: MOMENTUM_ENABLED + per-symbol whitelist gate.
        # Background: journal split showed momentum WINS on XAU/DJ30/SPI200/JPN225
        # (+$134 net /14d) but LOSES catastrophically on FX/silver/small-cap
        # (-$200 net). Disabling entirely killed both. Whitelist surgically.
        # See feedback_value_entry_research_20260605.md + journal analysis.
        try:
            from config import MOMENTUM_ENABLED, MOMENTUM_SYMBOL_WHITELIST
        except Exception:
            MOMENTUM_ENABLED = True
            MOMENTUM_SYMBOL_WHITELIST = None   # no gate (legacy fallback)
        if MOMENTUM_ENABLED:
            for symbol in SYMBOLS:
                if MOMENTUM_SYMBOL_WHITELIST and symbol not in MOMENTUM_SYMBOL_WHITELIST:
                    continue   # symbol is empirically a momentum loser — skip
                try:
                    result = self._process_symbol(symbol, equity, dd_pct, daily_loss_pct)
                    if result:
                        self._last_scores[symbol] = result
                        scores_for_dashboard[symbol] = result
                except Exception as e:
                    log.error("[%s] Process error: %s", symbol, e)

        # ═══ RESILIENT CANDLE BACKSTOP — keeps SharedState fed even when the
        #     Wine/MT5 tick bridge freezes, so a bridge stall can NEVER halt
        #     trades. Polls fresh candles via the resilient executor client
        #     (fresh request/response — reliable where the streaming bridge flaps).
        try:
            self._ensure_fresh_candles()
        except Exception as e:
            log.debug("fresh-candle backstop error: %s", e)

        # ═══ ICT FVG STRATEGY (independent book, runs after momentum) ═══
        try:
            self._process_fvg_whitelist(equity)
        except Exception as e:
            log.error("FVG whitelist error: %s", e)

        # ═══ SWEEP-RECLAIM STRATEGY (replacement for momentum, signal-only by default) ═══
        try:
            self._process_sweep_reclaim(equity)
        except Exception as e:
            log.error("SR strategy error: %s", e)

        # ═══ SMA-BREAKOUT STRATEGY (4th independent book — default OFF) ═══
        try:
            self._process_sma_breakout(equity)
        except Exception as e:
            log.error("SMABO strategy error: %s", e)

        # ═══ FIB-50 PULLBACK CONTINUATION (5th independent book — default OFF) ═══
        try:
            self._process_fib50(equity)
        except Exception as e:
            log.error("FIB50 strategy error: %s", e)

        # ═══ M1 SCALPER (6th independent book — XAU-only) ═══
        try:
            self._process_scalper(equity)
        except Exception as e:
            log.error("SCALPER strategy error: %s", e)

        # ═══ TREND-FOLLOWER (7th book — diversified daily trend, the robust core) ═══
        try:
            self._process_trend(equity)
        except Exception as e:
            log.error("TREND strategy error: %s", e)

        # ═══ INDICES MEAN-REVERSION (8th book — validated, account-appropriate) ═══
        try:
            self._process_imr(equity)
        except Exception as e:
            log.error("IMR strategy error: %s", e)

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
    #  ICT FVG STRATEGY (independent magic range, own risk/exits)
    # ═══════════════════════════════════════════════════════════════

    def _process_fvg_whitelist(self, equity):
        """Run the ICT liquidity-sweep + FVG-retest strategy for the whitelist.

        Independent of the momentum book:
          * own magics (base+1000/+1001) → never collides with momentum sizing
          * own base risk (FVG_RISK_PCT 0.25%) × validated momentum-ROC size
            tilt carried on the signal (sig["size_mult"], 0.60–1.40)
          * own SL/TP1/TP2 (placed broker-side by open_trade_explicit) + a 6h
            time stop + BE-move on the runner (manage_fvg_position)
        Yields to momentum: if the momentum book already holds the symbol we
        skip the FVG entry to avoid hedged / doubled exposure on one symbol.
        Whitelist-gated to the 7 recent-180d-positive symbols.
        """
        if not self._fvg_strategy or not self._fvg_whitelist:
            return
        try:
            from config import FVG_ENABLED, FVG_RISK_PCT, FVG_MAX_CONCURRENT
        except Exception:
            return
        if not FVG_ENABLED:
            return
        now = time.time()
        FVG_POST_CLOSE_COOLDOWN = 900  # 15min — let a fresh setup form

        # 1) Manage existing FVG positions FIRST (time stop + BE) so a close
        #    frees a concurrency slot before the entry pass this cycle. Also
        #    detect any close (incl. broker-side TP/SL) to arm a cooldown.
        for sym in self._fvg_whitelist:
            had_prev = self._fvg_was_open.get(sym, False)
            try:
                self.executor.manage_fvg_position(sym)
            except Exception as e:
                log.warning("[FVG %s] manage error: %s", sym, e)
            try:
                is_open = self.executor.has_fvg_position(sym)
            except Exception:
                is_open = had_prev
            if had_prev and not is_open:
                self._fvg_cooldown[sym] = now + FVG_POST_CLOSE_COOLDOWN
                # 2026-06-22: per-(FVG, sym) consec-loss counter.
                # Use peak_r proxy (won = peak_r >= 0.5). MasterBrain handles
                # the threshold + halt arming.
                try:
                    peak_r = float(getattr(self.executor, "_peak_profit_r",
                                           {}).get(sym, 0.0))
                    won = peak_r >= 0.5
                    if self._master_brain:
                        self._master_brain.record_strategy_symbol_close(
                            "fvg", sym, won=won)
                except Exception as _e:
                    log.debug("[FVG %s] strat-sym close record err: %s", sym, _e)
            self._fvg_was_open[sym] = is_open

        # 2) Concurrency count AFTER management.
        n_open = sum(1 for s in self._fvg_whitelist if self._fvg_was_open.get(s))

        # 3) Entry pass.
        for sym in self._fvg_whitelist:
            try:
                if n_open >= FVG_MAX_CONCURRENT:
                    break
                if self._fvg_was_open.get(sym):
                    continue                                 # already in an FVG trade
                if float(self._fvg_cooldown.get(sym, 0)) > now:
                    continue                                 # post-close cooldown
                if self.executor.has_position(sym):
                    continue                                 # yield to momentum book
                # 2026-06-22: per-(FVG, sym) 10-consec-loss halt gate
                if (self._master_brain
                        and self._master_brain.is_strategy_symbol_halted("fvg", sym)):
                    continue
                strat = (self._fvg_strategy.get(sym)
                         if isinstance(self._fvg_strategy, dict)
                         else self._fvg_strategy)
                if strat is None:
                    continue
                sig = strat.evaluate(sym)
                if not sig:
                    continue
                sm = float(sig.get("size_mult", 1.0))
                risk_pct = FVG_RISK_PCT * sm

                # 2026-06-21 D1_BIAS_UNIFIED parity check (downsize, NOT reject).
                # FVG already enforces its own daily-EMA200 bias internally
                # (fvg_strategy._daily_bias) — by the time we get a sig here
                # it ALREADY agrees with the FVG-module bias. We still consult
                # the unified verdict (D1-resample-then-EMA200) so misalignment
                # between the two bias sources downsizes the trade rather than
                # silently disagreeing. Same downsize as momentum/SR.
                try:
                    from config import (D1_BIAS_UNIFIED_ENABLED,
                                        D1_BIAS_UNIFIED_DOWNSIZE)
                    if D1_BIAS_UNIFIED_ENABLED:
                        from agent.expert.d1_bias_unified import get_d1_bias
                        _h1 = self.state.get_candles(sym, 60) if self.state else None
                        _d1b = get_d1_bias(sym, _h1)
                        if _d1b in ("LONG", "SHORT") and _d1b != sig["direction"]:
                            _dn = float(D1_BIAS_UNIFIED_DOWNSIZE)
                            if 0.0 < _dn < 1.0:
                                risk_pct = max(0.05, risk_pct * _dn)
                                log.info("[FVG %s] D1_BIAS counter (bias=%s dir=%s) "
                                         "→ x%.2f risk=%.3f%%",
                                         sym, _d1b, sig["direction"], _dn, risk_pct)
                except Exception as e:
                    log.debug("[FVG %s] D1_BIAS_UNIFIED skipped: %s", sym, e)

                # 2026-06-17: explicit strategy_name="FVG" (was relying on default)
                ok = self.executor.open_trade_explicit(
                    sym, sig["direction"], sig["entry"], sig["sl"],
                    sig["tp1"], sig["tp2"], risk_pct=risk_pct,
                    strategy_name="FVG")
                if ok:
                    n_open += 1
                    self._fvg_was_open[sym] = True
                    log.info("[FVG %s] ENTERED %s risk=%.3f%% (%.2f%% × %.2f) %s",
                             sym, sig["direction"], risk_pct, FVG_RISK_PCT, sm,
                             sig.get("reason", ""))
            except Exception as e:
                log.error("[FVG %s] entry error: %s", sym, e)

    # ═══════════════════════════════════════════════════════════════
    #  SWEEP-RECLAIM STRATEGY (value-entry replacement for momentum)
    # ═══════════════════════════════════════════════════════════════
    def _process_sweep_reclaim(self, equity):
        """Run the sweep-reclaim signal detector across the full symbol universe.

        Replaces the broken momentum-score+confirm paradigm with a single-bar
        event detector that enters at the structural inflection (the reclaim
        close) with a structural stop. See agent/sweep_reclaim.py docstring
        for full rationale + research citations.

        Operates in TWO modes (config flag SR_TRADE_LIVE):
          False: SIGNAL-ONLY — logs every detected setup but does NOT trade.
                 Use for the first 24-48h to compare predictions vs reality
                 before committing capital.
          True : LIVE — opens trades via executor.open_trade_explicit using
                 the SR_MAGIC_OFFSET book (independent of momentum + FVG).

        Both modes also de-dupe signals per (symbol, bar_time) so we never
        process the same closed bar twice.
        """
        try:
            from config import (SR_ENABLED, SR_TRADE_LIVE, SR_RISK_PCT,
                                SR_MAX_CONCURRENT, SR_POST_CLOSE_COOLDOWN_SECS)
        except Exception:
            return
        if not SR_ENABLED:
            return

        # Lazy-init the detector + bookkeeping on first call.
        if not hasattr(self, "_sr_strategy") or self._sr_strategy is None:
            try:
                from agent.sweep_reclaim import SweepReclaimStrategy
                self._sr_strategy = SweepReclaimStrategy(self.state)
                self._sr_was_open = {}
                self._sr_cooldown = {}
                self._sr_signal_dedupe = {}   # per-symbol last bar_time fired
                log.info("[SR] SweepReclaimStrategy initialized (TRADE_LIVE=%s)",
                         SR_TRADE_LIVE)
            except Exception as e:
                log.error("[SR] init failed: %s", e)
                self._sr_strategy = None
                return

        now = time.time()

        # 2026-06-22: SR close detection — arm cooldown + record per-(SR, sym)
        # consec-loss. Previously _sr_was_open was set on entry but never
        # reset on close, leaving the SR cooldown latch perpetually broken.
        for sym in list(self._sr_was_open.keys()):
            try:
                had_prev = self._sr_was_open.get(sym, False)
                is_open = self.executor.has_sr_position(sym)
                if had_prev and not is_open:
                    self._sr_cooldown[sym] = now + SR_POST_CLOSE_COOLDOWN_SECS
                    try:
                        peak_r = float(getattr(self.executor, "_peak_profit_r",
                                               {}).get(sym, 0.0))
                        won = peak_r >= 0.5
                        if self._master_brain:
                            self._master_brain.record_strategy_symbol_close(
                                "sr", sym, won=won)
                    except Exception as _e:
                        log.debug("[SR %s] strat-sym close record err: %s", sym, _e)
                self._sr_was_open[sym] = is_open
            except Exception as e:
                log.debug("[SR %s] close-detect err: %s", sym, e)

        # 1) Detect signals across the full symbol universe.
        for sym in SYMBOLS:
            try:
                # Skip if cooldown active.
                if float(self._sr_cooldown.get(sym, 0)) > now:
                    continue
                # 2026-06-22: per-(SR, sym) 10-consec-loss halt gate
                if (self._master_brain
                        and self._master_brain.is_strategy_symbol_halted("sr", sym)):
                    continue
                sig = self._sr_strategy.evaluate(sym)
                if not sig:
                    continue
                bar_t = sig.get("bar_time")
                # Per-symbol per-bar dedupe (don't re-fire on same closed bar)
                if self._sr_signal_dedupe.get(sym) == bar_t:
                    continue
                self._sr_signal_dedupe[sym] = bar_t

                # Always LOG the signal regardless of mode.
                log.info("[SR %s] SIGNAL %s entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f swept=%.5f wick=%.2fATR  %s",
                         sym, sig["direction"], sig["entry"], sig["sl"],
                         sig["tp1"], sig["tp2"], sig["swept_level"],
                         sig.get("wick_atr_mult", 0),
                         "[SIGNAL-ONLY]" if not SR_TRADE_LIVE else "[LIVE]")

                if not SR_TRADE_LIVE:
                    continue   # observation mode

                # LIVE MODE — open the trade via executor.
                # Concurrency cap check.
                n_open = sum(1 for s in self._sr_was_open if self._sr_was_open.get(s))
                if n_open >= SR_MAX_CONCURRENT:
                    log.info("[SR %s] SKIP: max concurrent SR (%d) reached", sym, n_open)
                    continue

                # Yield to momentum + FVG to avoid stacked exposure on one symbol.
                if self.executor.has_position(sym):
                    log.info("[SR %s] SKIP: existing position (momentum or other)", sym)
                    continue

                # 2026-06-17 BUGFIX: pass strategy_name="SR" + SR magic offsets.
                # Prior code defaulted to strategy_name="FVG" + FVG offsets, so
                # every SR trade was tagged "FVG" in MT5 comment, used FVG magic
                # range, and tracked under _fvg_entry_time. This is the source
                # of the "FVG showing in comment for SR trades" data debt + the
                # journal gate=fvg_or_sr ambiguity.
                from config import SR_MAGIC_OFFSET as _SR_OFF

                # 2026-06-21 D1_BIAS_UNIFIED soft-filter (downsize, NOT reject).
                # Same convention as momentum + FVG. Never blocks the trade.
                _sr_risk = SR_RISK_PCT
                try:
                    from config import (D1_BIAS_UNIFIED_ENABLED,
                                        D1_BIAS_UNIFIED_DOWNSIZE)
                    if D1_BIAS_UNIFIED_ENABLED:
                        from agent.expert.d1_bias_unified import get_d1_bias
                        _h1 = self.state.get_candles(sym, 60) if self.state else None
                        _d1b = get_d1_bias(sym, _h1)
                        if _d1b in ("LONG", "SHORT") and _d1b != sig["direction"]:
                            _dn = float(D1_BIAS_UNIFIED_DOWNSIZE)
                            if 0.0 < _dn < 1.0:
                                _sr_risk = max(0.05, _sr_risk * _dn)
                                log.info("[SR %s] D1_BIAS counter (bias=%s dir=%s) "
                                         "→ x%.2f risk=%.3f%%",
                                         sym, _d1b, sig["direction"], _dn, _sr_risk)
                except Exception as e:
                    log.debug("[SR %s] D1_BIAS_UNIFIED skipped: %s", sym, e)

                ok = self.executor.open_trade_explicit(
                    sym, sig["direction"], sig["entry"], sig["sl"],
                    sig["tp1"], sig["tp2"], risk_pct=_sr_risk,
                    magic_offsets=(_SR_OFF, _SR_OFF + 1),
                    strategy_name="SR")
                if ok:
                    self._sr_was_open[sym] = True
                    log.info("[SR %s] ENTERED %s risk=%.3f%% %s",
                             sym, sig["direction"], _sr_risk,
                             sig.get("reason", ""))
            except Exception as e:
                log.warning("[SR %s] eval/entry error: %s", sym, e)

    # ═══════════════════════════════════════════════════════════════
    #  SMABO (SMA Crossover Breakout) — independent book, default OFF
    # ═══════════════════════════════════════════════════════════════
    def _process_imr(self, equity):
        """INDICES MEAN-REVERSION (8th book) — long-only D1 RSI2+IBS dip-buy on
        cash indices. ~5-min reconcile: per-symbol entry/exit from the D1 cache.
        Signal-only until IMR_TRADE_LIVE. Own magic +7000/+7001; 6xATR broker SL
        is a DISASTER stop only (executor never trails/ELCs these magics)."""
        import pandas as _pd
        import pickle as _pickle
        from pathlib import Path as _Path
        try:
            from config import (IMR_ENABLED, IMR_TRADE_LIVE, IMR_WHITELIST, IMR_PARAMS,
                                 IMR_FIXED_LOTS, IMR_MAX_CONCURRENT, IMR_SUB_OFFSETS,
                                 IMR_MAGIC_OFFSET as _IMR_OFF)
        except Exception:
            return
        if not IMR_ENABLED:
            return
        if not getattr(self, "_imr_init", False):
            try:
                from agent.indices_mr import evaluate as _ie, should_exit as _ix
                self._imr_eval, self._imr_should_exit = _ie, _ix
                self._imr_last_run = 0.0
                self._imr_log_dedupe = {}
                self._imr_init = True
                log.info("[IMR] indices-MR initialized (TRADE_LIVE=%s, whitelist=%s)",
                         IMR_TRADE_LIVE, sorted(IMR_WHITELIST))
            except Exception as e:
                log.error("[IMR] init failed: %s", e)
                self._imr_init = False
                return
        now_ts = time.time()
        if now_ts - getattr(self, "_imr_last_run", 0.0) < 60:
            return
        self._imr_last_run = now_ts
        CACHE = _Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
        d1 = {}
        for sym in IMR_WHITELIST:
            try:
                p = CACHE / ("raw_d1_" + sym.replace(".", "_") + ".pkl")
                if p.exists():
                    # FAIL-CLOSED on a STALE D1 cache (2026-07-10 audit): if the
                    # fetch-d1 job died, the file's mtime stops advancing — placing
                    # a live fixed-lot order on frozen daily data is dangerous.
                    # Skip the symbol if its cache wasn't refreshed in >12h.
                    if (now_ts - p.stat().st_mtime) > 12 * 3600:
                        log.warning("[IMR %s] SKIP: D1 cache stale (%.1fh old) — fetch-d1 job down?",
                                    sym, (now_ts - p.stat().st_mtime) / 3600)
                        continue
                    df = _pickle.load(open(p, "rb"))
                    df["time"] = _pd.to_datetime(df["time"])
                    if len(df) >= 210:
                        d1[sym] = df
            except Exception:
                pass
        if not d1:
            return
        # open IMR positions (magic base+7000/7001) from the disk sync file
        from config import AUX_SYMBOLS as _AUX
        _positions, _fresh = self._read_positions_json_meta()
        if not _fresh and IMR_TRADE_LIVE:
            # Fail closed on stale sync data (only matters once live — signal-only
            # mode still logs from whatever it has).
            log.warning("[IMR] SKIP: sync file stale/missing (fail-closed)")
            return
        open_imr = {}
        _base = {s: int(c.magic) for s, c in {**SYMBOLS, **_AUX}.items()}
        for x in _positions:
            if (int(x["magic"]) - _base.get(x["symbol"], -99999)) in IMR_SUB_OFFSETS:
                open_imr[x["symbol"]] = int(x["open_time"])
        n_open = len(open_imr)
        for sym in IMR_WHITELIST:
            if sym not in d1:   # aux symbols resolved via symbol_cfg() in executor
                continue
            try:
                if sym in open_imr:
                    held = max((datetime.now(timezone.utc)
                                - datetime.fromtimestamp(open_imr[sym], timezone.utc)).days, 0)
                    reason = self._imr_should_exit(d1[sym], held, IMR_PARAMS)
                    if reason:
                        log.info("[IMR %s] EXIT signal (%s, held %dd) %s", sym, reason, held,
                                 "[LIVE]" if IMR_TRADE_LIVE else "[SIGNAL-ONLY]")
                        if IMR_TRADE_LIVE:
                            self.executor.close_imr_position(sym, comment="IMR_exit")
                            break   # one in-process MT5 order per cycle (bridge hangs on the 2nd)
                    continue
                sig = self._imr_eval(d1[sym], IMR_PARAMS)
                if not sig:
                    continue
                bar_t = sig.get("bar_time")
                if self._imr_log_dedupe.get(sym) != bar_t:
                    self._imr_log_dedupe[sym] = bar_t
                    log.info("[IMR %s] ENTRY signal: %s  %s", sym, sig["reason"],
                             "[LIVE]" if IMR_TRADE_LIVE else "[SIGNAL-ONLY]")
                if not IMR_TRADE_LIVE or n_open >= IMR_MAX_CONCURRENT:
                    continue
                lot = float(IMR_FIXED_LOTS.get(sym, 0.10))
                ok = self.executor.open_imr_trade(
                    sym, lot, float(sig["sl_atr_mult"]), float(sig["atr14"]), _IMR_OFF)
                if ok:
                    n_open += 1
                    log.info("[IMR %s] ENTERED LONG %.2f lots", sym, lot)
                    break   # one in-process MT5 order per cycle (bridge hangs on the 2nd)
            except Exception as e:
                log.warning("[IMR %s] eval/entry error: %s", sym, e)

    def _process_trend(self, equity):
        """TREND-FOLLOWER (7th book) — diversified daily trend portfolio. Once/day:
        per instrument compute the D1 3-EMA signal and flip/close/open to match.
        Own magic +6000. Vol-targeted via a wide 3xATR stop (equal $-risk sizing)."""
        import pandas as _pd
        try:
            from config import (TREND_ENABLED, TREND_TRADE_LIVE, TREND_RISK_PCT,
                                 TREND_ATR_STOP, TREND_ATR_PERIOD, TREND_EMA_PAIRS,
                                 TREND_MIN_ABS_SIGNAL, TREND_REBALANCE_HOUR, TREND_BASKET,
                                 TREND_MAX_RISK_PCT, TREND_MAGIC_OFFSET as _TR_OFF,
                                 TREND_TP_ATR, TREND_TRAIL_ENABLED, TREND_TRAIL_ATR,
                                 TREND_TRAIL_LOOKBACK, TREND_LOCK_FRAC,
                                 TREND_LOCK_ACTIVATE_ATR, TREND_REVERSAL_EXIT_ENABLED,
                                 TREND_GIVEBACK_FRAC)
            from config import trend_exit_params as _trend_exit_params
            from config import trend_conviction as _trend_conv_cfg
            from config import trend_ema_pairs as _trend_ema_pairs
        except Exception:
            return
        if not TREND_ENABLED:
            return
        if not getattr(self, "_trend_init", False):
            try:
                from agent.trend_follower import (evaluate as _tr_eval,
                                                   chandelier_stop as _tr_chand, _atr as _tr_atr,
                                                   conviction as _tr_conv)
                self._trend_eval = _tr_eval
                self._trend_chandelier = _tr_chand
                self._trend_atr = _tr_atr
                self._trend_conviction = _tr_conv
                self._trend_last_run = 0.0
                self._trend_init = True
                log.info("[TREND] trend-follower initialized (TRADE_LIVE=%s, basket=%s)",
                         TREND_TRADE_LIVE, TREND_BASKET)
            except Exception as e:
                log.error("[TREND] init failed: %s", e)
                self._trend_init = False
                return

        # Reconcile every 5 min (idempotent — only acts on missing positions or
        # a daily signal flip; self-completes the basket if a startup fetch fails).
        now_ts = time.time()
        # 60s cadence: reconcile only acts on a signal flip (1 order/cycle now),
        # so a fresh startup fills the 4-symbol basket in ~4 min instead of 20;
        # idle cycles are disk-only reads (cheap, no MT5 contention).
        if now_ts - getattr(self, "_trend_last_run", 0.0) < 60:
            return
        self._trend_last_run = now_ts
        # ── Data WITHOUT hammering the flaky live bridge ──
        # (1) D1 from the DISK CACHE (D1 bars change once/day; refreshed by the
        #     scheduled fetch_d1 job). No in-process multi-symbol live fetch =
        #     no contention with the tick-bridge/dashboard/backstop clients.
        import pickle as _pickle
        from pathlib import Path as _Path
        CACHE = _Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
        _now_ts = time.time()
        d1_data = {}
        for sym in TREND_BASKET:
            try:
                p = CACHE / ("raw_d1_" + sym.replace(".", "_") + ".pkl")
                if p.exists():
                    # FAIL-CLOSED on a STALE D1 cache (2026-07-10 audit): skip a
                    # symbol whose cache wasn't refreshed in >12h (fetch-d1 down)
                    # rather than trade/trail on frozen daily bars.
                    if (_now_ts - p.stat().st_mtime) > 12 * 3600:
                        log.warning("[TREND %s] SKIP: D1 cache stale (%.1fh old) — fetch-d1 job down?",
                                    sym, (_now_ts - p.stat().st_mtime) / 3600)
                        continue
                    df = _pickle.load(open(p, "rb"))
                    df["time"] = _pd.to_datetime(df["time"])
                    if len(df) >= 260:
                        d1_data[sym] = df
            except Exception:
                pass
        if not d1_data:
            log.warning("[TREND] reconcile ABORT: no D1 cache read in-process")
            return
        log.info("[TREND] reconcile: d1=%d symbols loaded", len(d1_data))
        # positions from the disk sync file (no in-process MT5 read); specs static
        from config import TREND_SUB_OFFSETS, AUX_SYMBOLS
        _positions, _fresh = self._read_positions_json_meta()
        if not _fresh:
            # Fail CLOSED: a stale/frozen sync file would make open positions look
            # flat → duplicate opens. Skip the whole reconcile; broker SL/TP still
            # protect open legs. The sync watchdog restarts the daemon.
            log.warning("[TREND] reconcile SKIP: sync file stale/missing (fail-closed)")
            return
        _base = {s: int(c.magic) for s, c in {**SYMBOLS, **AUX_SYMBOLS}.items()}
        pos_dir = {}
        for x in _positions:
            off = int(x["magic"]) - _base.get(x["symbol"], -99999)
            if off in TREND_SUB_OFFSETS:
                pos_dir[x["symbol"]] = 1 if int(x["type"]) == 0 else -1
        # static (vol_min, tick_value, tick_size) — verified 2026-07-08, never change
        specs = {
            "XAUUSD":   (0.01, 1.0,      0.01),
            "BTCUSD":   (0.01, 0.01,     0.01),
            "ETHUSD":   (0.01, 0.01,     0.01),
            "JPN225ft": (1.0,  6.17e-05, 0.01),
            "NAS100.r": (0.1,  0.01,     0.01),
        }
        _base_params = {"MIN_ABS_SIGNAL": TREND_MIN_ABS_SIGNAL, "ATR_PERIOD": TREND_ATR_PERIOD}
        acted = 0
        did_write = False   # any in-process MT5 order/close this cycle (bridge cap = 1)
        mt5 = getattr(self.executor, "mt5", None)
        for sym in TREND_BASKET:
            try:
                # aux symbols (ETH/JPN/NAS) are intentionally NOT in SYMBOLS; the
                # executor resolves them via symbol_cfg(). Gate on cache presence.
                if sym not in d1_data:
                    continue
                # per-symbol EMA ensemble (2026-07-11 signal tune: ETH wants wider)
                params = dict(_base_params, EMA_PAIRS=_trend_ema_pairs(sym))
                sig = self._trend_eval(d1_data[sym], params)
                if sig is None:
                    continue
                target = int(sig["direction"])
                cur = pos_dir.get(sym, 0)
                # Re-entry block after a reversal exit: don't re-open the SAME
                # direction we just bailed on until the daily signal changes.
                # CLEAR it whenever the signal is no longer the blocked direction —
                # BEFORE the target==cur short-circuit, so a signal decaying to 0
                # (target==cur==0) still clears the block instead of locking the
                # symbol out of that direction forever (2026-07-10 review fix).
                _rev_blk = getattr(self, "_trend_rev_block", {})
                if _rev_blk.get(sym) is not None and _rev_blk.get(sym) != target:
                    _rev_blk.pop(sym, None)
                if target == cur:
                    continue
                if cur == 0 and _rev_blk.get(sym) == target:
                    continue
                # SELECTIVITY / CONVICTION GATE (2026-07-10, "PF 6.91 discipline"):
                # only OPEN a NEW position when the daily trend is strong enough
                # (per-symbol ADX/slope). Closes & flips (cur!=0) are NOT gated —
                # we always exit; the deferred reopen next cycle (cur==0) is gated.
                if cur == 0 and target != 0:
                    _adxm, _slpm = _trend_conv_cfg(sym)
                    if _adxm > 0 or _slpm > 0:
                        _adx, _slope = self._trend_conviction(d1_data[sym], TREND_ATR_PERIOD)
                        if _adx < _adxm or _slope < _slpm:
                            log.info("[TREND %s] entry SKIP: low conviction adx=%.0f(min %.0f) slope=%.2f(min %.2f)",
                                     sym, _adx, _adxm, _slope, _slpm)
                            continue
                log.info("[TREND %s] signal=%.2f target=%d cur=%d",
                         sym, sig["signal"], target, cur)
                if not TREND_TRADE_LIVE:
                    continue
                if cur != 0:
                    # FLIP/EXIT: close now and DEFER any opposite open to the next
                    # cycle. Issuing close+open together is 2-4 in-process order
                    # calls = the exact bridge hang. Next cycle sees cur==0 (sync
                    # lag < cadence) and opens once.
                    self.executor.close_trend_position(sym, comment="TREND_flip")
                    did_write = True
                    break
                if target != 0:
                    # entry ref = last D1 close (open_trade_explicit fills at the
                    # live tick internally; this only sets SL/TP distance + sizing)
                    entry = float(sig["close"])
                    atr = float(sig["atr"])
                    sl_dist = TREND_ATR_STOP * atr
                    # RISK CAP: on a small account min-lot would over-risk (gold
                    # 14%). Tighten the stop so even at min-lot a trade risks
                    # <= TREND_MAX_RISK_PCT. Keeps gold in the basket, safely.
                    # 2026-07-10 AUDIT FIX: open_trade_explicit opens 2 LEGS each
                    # forced to vmin, so the real min-lot risk is 2*vmin — use that
                    # in the cap or the effective cap silently doubles.
                    sp = specs.get(sym)
                    if sp:
                        vmin, tval, tsize = sp
                        legs_vmin = 2.0 * vmin      # 2-leg split, both at min-lot
                        cap = equity * TREND_MAX_RISK_PCT / 100.0
                        if tsize > 0 and tval > 0 and vmin > 0:
                            risk_min = (sl_dist / tsize) * tval * legs_vmin
                            if risk_min > cap:
                                sl_dist = cap * tsize / (tval * legs_vmin)
                    # Realistic ATR-distance TP (2026-07-08): a visible target at
                    # TREND_TP_ATR x ATR from entry (independent of the risk-capped
                    # stop). The Chandelier trail then protects profit en route.
                    tp_dist = TREND_TP_ATR * atr
                    if target == 1:
                        sl, tp = entry - sl_dist, entry + tp_dist
                    else:
                        sl, tp = entry + sl_dist, entry - tp_dist
                    ok = self.executor.open_trade_explicit(
                        sym, "LONG" if target == 1 else "SHORT", entry, sl, tp, tp,
                        risk_pct=TREND_RISK_PCT, magic_offsets=(_TR_OFF, _TR_OFF + 1),
                        strategy_name="TREND")
                    if ok:
                        acted += 1
                        did_write = True
                        log.info("[TREND %s] ENTERED %s risk=%.2f%%",
                                 sym, "LONG" if target == 1 else "SHORT", TREND_RISK_PCT)
                        break   # one successful order per cycle (bridge cap)
                    else:
                        log.warning("[TREND %s] ENTRY FAILED", sym)
                        # a broker reject did NOT write — keep trying the next
                        # basket symbol so one bad symbol can't block the rest.
            except Exception as e:
                log.warning("[TREND %s] rebalance err: %s", sym, e)
                break
        if acted:
            log.info("[TREND] rebalance: %d position opened/flipped", acted)

        # trend legs per symbol FROM DISK (ticket/tp/price_cur/sl) — all TREND
        # management is read-free; only the modify/close WRITE hits the bridge.
        trend_legs = {}
        for x in _positions:
            if (int(x["magic"]) - _base.get(x["symbol"], -99999)) in TREND_SUB_OFFSETS:
                trend_legs.setdefault(x["symbol"], []).append(x)

        def _pos_profit_pts(sym, cur):
            """(profit_pts, atr, sl_dist, legs) for an open trend symbol, from disk
            + D1 cache. sl_dist = the position's ACTUAL (risk-capped) stop distance."""
            legs = trend_legs.get(sym)
            if not legs or sym not in d1_data:
                return None, None, None, None
            entry = float(legs[0].get("price_open") or 0)
            px = float(legs[0].get("price_cur") or 0)
            sl = float(legs[0].get("sl") or 0)
            if entry <= 0 or px <= 0:
                return None, None, None, None
            df = d1_data[sym]
            atr = float(self._trend_atr(df["high"], df["low"], df["close"],
                                        TREND_ATR_PERIOD).iloc[-1])
            prof = (px - entry) if cur == 1 else (entry - px)
            sl_dist = abs(entry - sl) if sl > 0 else (TREND_ATR_STOP * atr)
            return prof, atr, sl_dist, legs

        # ── REVERSAL EXIT (peak-giveback) — the ACTIVE profit protector ──
        # Track each position's PEAK open profit; if profit rolls over and retraces
        # >= GIVEBACK_FRAC from that peak (once peak cleared the ATR activation),
        # close at market. Tighter/faster than the SL and catches reversals. One
        # write/cycle (respects did_write); the SL trail below is the backstop.
        if TREND_REVERSAL_EXIT_ENABLED and TREND_TRADE_LIVE and not did_write:
            if not hasattr(self, "_trend_peak"):
                self._trend_peak = {}
            if not hasattr(self, "_trend_rev_block"):
                self._trend_rev_block = {}
            live_keys = set()
            for sym, cur in pos_dir.items():
                if cur == 0:
                    continue
                prof, atr, sl_dist, legs = _pos_profit_pts(sym, cur)
                if prof is None:
                    continue
                key = tuple(sorted(int(l.get("ticket") or 0) for l in legs))
                live_keys.add(key)
                peak = max(self._trend_peak.get(key, prof), prof)
                self._trend_peak[key] = peak
                _, _, _gb, _act = _trend_exit_params(sym)   # per-symbol (H1-tuned)
                # ACTIVATE relative to the ACTUAL SL distance (2026-07-10 fix): the
                # backtest used STOP=3xATR, so ACT*ATR == (ACT/3)*sl_dist. Live SLs
                # are risk-capped (~0.2xATR), so the old ACT*ATR meant "arm at ~2.5R"
                # — unreachable before the SL. Scaling to sl_dist arms it at the
                # intended fraction of risk (e.g. ~0.17R) so small real profits are
                # protected instead of riding to the stop.
                _act_thresh = (_act / TREND_ATR_STOP) * sl_dist if TREND_ATR_STOP > 0 else _act * atr
                if peak >= _act_thresh and prof <= peak * (1.0 - _gb):
                    log.info("[TREND %s] REVERSAL EXIT: profit %.0f pts retraced from peak %.0f "
                             "(>= %.0f%% giveback) — closing", sym, prof, peak, _gb * 100)
                    if self.executor.close_trend_position(sym, comment="TREND_reversal"):
                        self._trend_peak.pop(key, None)
                        self._trend_rev_block[sym] = cur   # block same-dir re-entry until signal changes
                    did_write = True   # attempted a write either way (bridge cap)
                    break
            # forget peaks for positions that are no longer open
            for k in list(self._trend_peak.keys()):
                if k not in live_keys:
                    self._trend_peak.pop(k, None)

        # ── Chandelier + profit-lock TRAILING pass (broker-side SL backstop) ──
        # Only when NO order/close was written this cycle. One MT5 write/cycle.
        if TREND_TRAIL_ENABLED and TREND_TRADE_LIVE and not did_write:
            # static (digits, point, stops_level) — verified 2026-07-08 (min_gap only)
            _TRAIL_SPECS = {"XAUUSD": (2, 0.01, 20), "BTCUSD": (2, 0.01, 0),
                            "ETHUSD": (2, 0.01, 0), "JPN225ft": (2, 0.01, 50),
                            "NAS100.r": (2, 0.01, 50)}
            for sym, cur in pos_dir.items():
                if cur == 0 or sym not in d1_data:
                    continue
                legs = trend_legs.get(sym)
                if not legs:
                    continue
                try:
                    df = d1_data[sym]
                    _tr, _lk, _gb2, _act2 = _trend_exit_params(sym)   # per-symbol (H1-tuned)
                    tparams = {"ATR_PERIOD": TREND_ATR_PERIOD,
                               "TRAIL_LOOKBACK": TREND_TRAIL_LOOKBACK,
                               "TRAIL_ATR": _tr}
                    stop = self._trend_chandelier(df, cur, tparams)
                    if stop is None:
                        continue
                    tag = "Chandelier"
                    # PROFIT-LOCK ratchet: once open profit >= ACTIVATE x ATR, lock
                    # LOCK_FRAC of it and take the TIGHTER of {chandelier, lock}.
                    # Fixes the pullback case where the 22d-high chandelier locks
                    # almost nothing on a position already well in profit.
                    entry = float(legs[0].get("price_open") or 0)
                    px = float(legs[0].get("price_cur") or 0)
                    _sl0 = float(legs[0].get("sl") or 0)
                    atr = float(self._trend_atr(df["high"], df["low"], df["close"],
                                                TREND_ATR_PERIOD).iloc[-1])
                    if entry > 0 and px > 0 and atr > 0:
                        prof = (px - entry) if cur == 1 else (entry - px)
                        # activate profit-lock relative to ACTUAL SL distance (same
                        # 2026-07-10 fix as the reversal exit): risk-capped live SLs
                        # made ACT*ATR unreachable before the stop.
                        _sldist = abs(entry - _sl0) if _sl0 > 0 else (TREND_ATR_STOP * atr)
                        _lock_thresh = (_act2 / TREND_ATR_STOP) * _sldist if TREND_ATR_STOP > 0 else _act2 * atr
                        if prof >= _lock_thresh:
                            lock = (entry + _lk * prof) if cur == 1 \
                                else (entry - _lk * prof)
                            tighter = max(stop, lock) if cur == 1 else min(stop, lock)
                            if tighter != stop:
                                stop, tag = tighter, "ProfitLock%d%%" % int(_lk * 100)
                    dg, pt, sl_lvl = _TRAIL_SPECS.get(sym, (2, 0.01, 20))
                    min_gap = (sl_lvl + 2) * pt
                    if self.executor.trail_trend_sl(sym, stop, legs, min_gap, dg):
                        log.info("[TREND %s] trail SL -> %.2f (%s)", sym, stop, tag)
                        break   # one SLTP-modify per cycle (bridge safety)
                except Exception as e:
                    log.debug("[TREND %s] trail err: %s", sym, e)

    def _read_positions_json(self):
        """Open positions from the isolated sync daemon's disk file (reliable),
        instead of an in-process MT5 read (which fails under bridge contention)."""
        return self._read_positions_json_meta()[0]

    def _read_positions_json_meta(self, max_age=90.0):
        """Returns (positions, fresh). `fresh` is False if the sync file is
        missing, unparseable, or older than max_age seconds — callers that OPEN
        trades must fail CLOSED on stale data (a frozen file omitting a just-filled
        position would otherwise look flat and trigger a duplicate open)."""
        import json as _json
        from pathlib import Path as _Path
        try:
            p = _Path(__file__).resolve().parent.parent / "data" / "live_positions.json"
            if not p.exists():
                return [], False
            payload = _json.loads(p.read_text())
            ts = float(payload.get("ts", 0.0))
            fresh = (time.time() - ts) <= max_age if ts else False
            return payload.get("positions", []), fresh
        except Exception:
            return [], False

    def _ensure_fresh_candles(self):
        """Resilient candle backstop. Polls fresh OHLC from the executor's
        auto-reconnecting MT5 client into SharedState, throttled per (sym, tf),
        so a frozen/flapping tick bridge can NEVER starve the strategies. Only
        the live-traded symbols + the timeframes they need."""
        import pandas as _pd
        try:
            from config import SMABO_WHITELIST, SCALPER_WHITELIST
        except Exception:
            return
        mt5 = getattr(self.executor, "mt5", None)
        if mt5 is None:
            return
        now = time.time()
        if not hasattr(self, "_fresh_last"):
            self._fresh_last = {}
        jobs = []   # (sym, tf, mt5_tf, min_interval_s, bars)
        for sym in SCALPER_WHITELIST:
            jobs.append((sym, 1, 1, 20, 320))          # M1 for the scalper
        for sym in SMABO_WHITELIST:
            jobs.append((sym, 15, 15, 60, 320))         # M15 for SMABO
            jobs.append((sym, 60, 16385, 120, 240))     # H1 for SMABO's H4
        for sym, tf, mtf, interval, bars in jobs:
            if sym not in SYMBOLS:
                continue
            key = (sym, tf)
            if now - self._fresh_last.get(key, 0) < interval:
                continue
            self._fresh_last[key] = now
            try:
                r = mt5.copy_rates_from_pos(sym, mtf, 0, bars)
                if r is None or len(r) < 5:
                    continue
                df = _pd.DataFrame(r)
                if "time" not in df.columns:
                    continue
                df["time"] = _pd.to_datetime(df["time"], unit="s")
                self.state.update_candles(sym, tf, df)
            except Exception as e:
                log.debug("[fresh %s tf%s] backstop fetch err: %s", sym, tf, e)

    def _process_scalper(self, equity):
        """M1 SCALPER (6th book) — XAU-only mean-reversion micro-fade. Own magic
        +5000/+5001. Broker TP(=mean)/SL(=1xATR) + M1 time-stop enforced here."""
        try:
            from config import (SCALPER_ENABLED, SCALPER_TRADE_LIVE, SCALPER_RISK_PCT,
                                 SCALPER_POST_CLOSE_COOLDOWN_SECS, SCALPER_KILL_AFTER_LOSSES,
                                 SCALPER_TIME_STOP_BARS, SCALPER_WHITELIST, SCALPER_PARAMS,
                                 SCALPER_MAGIC_OFFSET as _SC_OFF)
        except Exception:
            return
        if not SCALPER_ENABLED:
            return
        if not getattr(self, "_scalper_init", False):
            try:
                from agent.m1_scalper import evaluate as _sc_eval
                self._scalper_eval = _sc_eval
                self._scalper_was_open = {}
                self._scalper_cooldown = {}
                self._scalper_dedupe = {}
                self._scalper_consec_losses = {}
                self._scalper_kill_until_date = {}
                self._scalper_last_kill_date = None
                self._scalper_last_eval_log = {}
                self._scalper_init = True
                log.info("[SCALP] M1 scalper initialized (TRADE_LIVE=%s, whitelist=%s)",
                         SCALPER_TRADE_LIVE, sorted(SCALPER_WHITELIST))
            except Exception as e:
                log.error("[SCALP] init failed: %s", e)
                self._scalper_init = False
                return

        now = time.time()
        today_utc = datetime.now(timezone.utc).date()
        if self._scalper_last_kill_date is None or self._scalper_last_kill_date < today_utc:
            self._scalper_consec_losses = {}
            self._scalper_kill_until_date = {}
            self._scalper_last_kill_date = today_utc

        # 1) time-stop: close any scalp open longer than N M1 bars (~minutes)
        for sym in SCALPER_WHITELIST:
            try:
                if self.executor.has_scalper_position(sym):
                    et = float(getattr(self.executor, "_scalper_entry_time", {}).get(sym, 0.0))
                    if et > 0 and (now - et) > SCALPER_TIME_STOP_BARS * 60:
                        self.executor.close_scalper_position(sym, comment="SCALP_time_stop")
                        log.info("[SCALP %s] TIME-STOP close after %d min",
                                 sym, SCALPER_TIME_STOP_BARS)
            except Exception as _ts:
                log.debug("[SCALP %s] time-stop err: %s", sym, _ts)

        # 2) detect signals + enter
        for sym in SCALPER_WHITELIST:
            try:
                if sym not in SYMBOLS:
                    continue
                if float(self._scalper_cooldown.get(sym, 0)) > now:
                    continue
                m1 = self.state.get_candles(sym, 1)
                sig = self._scalper_eval(m1, SCALPER_PARAMS)
                if not sig:
                    continue
                bar_t = sig.get("bar_time")
                if self._scalper_last_eval_log.get(sym) != bar_t:
                    self._scalper_last_eval_log[sym] = bar_t
                    log.info("[SCALP %s] SIGNAL %s entry=%.5f sl=%.5f tp=%.5f %s  %s",
                             sym, sig["direction"], sig["entry"], sig["sl"], sig["tp1"],
                             sig.get("reason", ""),
                             "[LIVE]" if SCALPER_TRADE_LIVE else "[SIGNAL-ONLY]")
                if self._scalper_dedupe.get(sym) == bar_t:
                    continue
                if not SCALPER_TRADE_LIVE:
                    continue
                _kd = self._scalper_kill_until_date.get(sym)
                if _kd is not None and today_utc <= _kd:
                    continue
                if self.executor.has_scalper_position(sym):
                    continue
                if (self.executor.has_position(sym) or self.executor.has_fvg_position(sym)
                        or self.executor.has_sr_position(sym)
                        or self.executor.has_smabo_position(sym)):
                    continue
                ok = self.executor.open_trade_explicit(
                    sym, sig["direction"], sig["entry"], sig["sl"], sig["tp1"], sig["tp2"],
                    risk_pct=SCALPER_RISK_PCT, magic_offsets=(_SC_OFF, _SC_OFF + 1),
                    strategy_name="SCALPER")
                if ok:
                    self._scalper_was_open[sym] = True
                    self._scalper_dedupe[sym] = bar_t
                    log.info("[SCALP %s] ENTERED %s risk=%.3f%%",
                             sym, sig["direction"], SCALPER_RISK_PCT)
            except Exception as e:
                log.warning("[SCALP %s] eval/entry error: %s", sym, e)

        # 3) detect closes → cooldown + per-symbol consec-loss kill (realized PnL)
        for sym in list(self._scalper_was_open.keys()):
            try:
                had = self._scalper_was_open.get(sym, False)
                is_open = self.executor.has_scalper_position(sym)
                if had and not is_open:
                    self._scalper_cooldown[sym] = now + SCALPER_POST_CLOSE_COOLDOWN_SECS
                    won, _pnl = False, 0.0
                    try:
                        from datetime import timedelta as _td
                        _now = datetime.now(timezone.utc)
                        _deals = self.executor.mt5.history_deals_get(
                            _now - _td(minutes=15), _now + _td(minutes=2)) or []
                        _cfg = SYMBOLS.get(sym)
                        _mag = ({int(_cfg.magic) + _SC_OFF, int(_cfg.magic) + _SC_OFF + 1}
                                if _cfg else set())
                        _pnl = sum(float(d.profit) + float(getattr(d, "swap", 0.0))
                                   + float(getattr(d, "commission", 0.0))
                                   for d in _deals if getattr(d, "symbol", "") == sym
                                   and int(getattr(d, "magic", -1)) in _mag)
                        won = _pnl > 0.0
                    except Exception:
                        pass
                    cl = self._scalper_consec_losses
                    if not won:
                        cl[sym] = cl.get(sym, 0) + 1
                        if cl[sym] >= SCALPER_KILL_AFTER_LOSSES:
                            self._scalper_kill_until_date[sym] = today_utc
                            log.warning("[SCALP %s] DAILY KILL — %d consec losses",
                                        sym, cl[sym])
                    else:
                        cl[sym] = 0
                    log.info("[SCALP %s] CLOSED pnl=%.2f", sym, _pnl)
                self._scalper_was_open[sym] = is_open
            except Exception as e:
                log.debug("[SCALP %s] close-check err: %s", sym, e)

    def _process_sma_breakout(self, equity):
        """Run the SMA-crossover-breakout signal detector across the SMABO
        whitelist. Independent of momentum / FVG / SR books — own magic range
        (+3000/+3001) and own risk budget (SMABO_RISK_PCT, default 0.25%).

        Operates in TWO modes (config flag SMABO_TRADE_LIVE):
          False: SIGNAL-ONLY — logs every detected setup but does NOT trade.
                 Default — code loads, no orders fire, until empirical
                 validation lands.
          True : LIVE — opens trades via executor.open_trade_explicit using
                 the SMABO_MAGIC_OFFSET book (independent of momentum/FVG/SR).

        Both modes also de-dupe signals per (symbol, bar_time) so we never
        process the same closed M15 bar twice.

        Also enforces a per-day consecutive-loss kill switch
        (SMABO_KILL_AFTER_LOSSES) — when N consecutive losses fire within
        the current UTC day, halts entries until next UTC midnight. Mirrors
        the [[project_dragon_sr_live_flip_20260608]] kill-if-WR<40%/5 rule.
        """
        try:
            from config import (SMABO_ENABLED, SMABO_TRADE_LIVE, SMABO_RISK_PCT,
                                SMABO_MAX_CONCURRENT,
                                SMABO_POST_CLOSE_COOLDOWN_SECS,
                                SMABO_KILL_AFTER_LOSSES, SMABO_WHITELIST,
                                SMABO_MAGIC_OFFSET as _SMABO_OFF)
        except Exception:
            return
        if not SMABO_ENABLED:
            return

        # Lazy-init detector + bookkeeping on first call.
        if not hasattr(self, "_smabo_strategy") or self._smabo_strategy is None:
            try:
                from agent.sma_breakout import SMABreakoutStrategy
                self._smabo_strategy = SMABreakoutStrategy(self.state)
                self._smabo_was_open = {}
                self._smabo_cooldown = {}
                self._smabo_signal_dedupe = {}    # per-symbol bar_time ENTERED (set on success)
                self._smabo_log_dedupe = {}       # per-symbol bar_time SIGNAL-logged
                self._smabo_consec_losses = {}    # per-SYMBOL daily consec-loss counter
                self._smabo_kill_until_date = {}  # per-SYMBOL kill date
                log.info("[SMABO] SMABreakoutStrategy initialized "
                         "(TRADE_LIVE=%s, whitelist=%s)",
                         SMABO_TRADE_LIVE, sorted(SMABO_WHITELIST))
            except Exception as e:
                log.error("[SMABO] init failed: %s", e)
                self._smabo_strategy = None
                return

        now = time.time()
        today_utc = datetime.now(timezone.utc).date()

        # NOTE: the daily kill is now checked PER-SYMBOL inside the detect loop
        # (entries-only) — NOT here. The old book-wide early-return silenced
        # evaluation + signal logging for BOTH symbols for the rest of the UTC
        # day (2026-07-06 audit: the primary under-trading cause). Removed.
        # Day rollover — reset the per-symbol counters at UTC midnight.
        last_check = getattr(self, "_smabo_last_kill_check_date", None)
        if last_check is None or last_check < today_utc:
            self._smabo_consec_losses = {}
            self._smabo_kill_until_date = {}
            self._smabo_last_kill_check_date = today_utc

        # 1) Detect signals across the SMABO whitelist.
        for sym in SMABO_WHITELIST:
            try:
                if sym not in SYMBOLS:
                    continue   # symbol not configured for this account
                if float(self._smabo_cooldown.get(sym, 0)) > now:
                    continue
                sig = self._smabo_strategy.evaluate(sym)
                if not sig:
                    continue
                bar_t = sig.get("bar_time")
                # Entry-consuming dedupe: skip only if we already ENTERED on this
                # bar (set after a successful open below). A signal whose entry
                # FAILED (e.g. transient bridge outage) is NOT deduped here, so it
                # retries on the next cycle until it fills or the bar rolls.
                # (2026-07-07: fixes signals lost to the flaky Wine/MT5 feed.)
                if self._smabo_signal_dedupe.get(sym) == bar_t:
                    continue
                # Log-dedupe: emit the SIGNAL / ENTRY-FAILED lines once per bar.
                fresh_bar = self._smabo_log_dedupe.get(sym) != bar_t
                if fresh_bar:
                    self._smabo_log_dedupe[sym] = bar_t

                # LOG the signal once per bar.
                if fresh_bar:
                    log.info("[SMABO %s] SIGNAL %s entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f "
                         "H4S=%.5f H4R=%.5f sma8=%.5f sma50=%.5f  %s",
                         sym, sig["direction"], sig["entry"], sig["sl"],
                         sig["tp1"], sig["tp2"],
                         sig.get("h4_support", 0), sig.get("h4_resistance", 0),
                         sig.get("sma_fast", 0), sig.get("sma_slow", 0),
                         "[SIGNAL-ONLY]" if not SMABO_TRADE_LIVE else "[LIVE]")

                if not SMABO_TRADE_LIVE:
                    continue   # observation mode — code loads, no trades fire

                # Per-symbol daily kill — blocks ENTRIES for THIS symbol only.
                # Signals still logged above; close-detection loop still runs.
                _kd = self._smabo_kill_until_date.get(sym)
                if _kd is not None and today_utc <= _kd:
                    log.info("[SMABO %s] SKIP: daily kill active (until %s)", sym, _kd)
                    continue

                # LIVE mode — concurrency cap check.
                n_open = sum(1 for s in self._smabo_was_open
                             if self._smabo_was_open.get(s))
                if n_open >= SMABO_MAX_CONCURRENT:
                    log.info("[SMABO %s] SKIP: max concurrent (%d) reached",
                             sym, n_open)
                    continue

                # Yield to momentum / FVG / SR to avoid stacked exposure.
                if (self.executor.has_position(sym)
                        or self.executor.has_fvg_position(sym)
                        or self.executor.has_sr_position(sym)):
                    log.info("[SMABO %s] SKIP: existing position "
                             "(momentum / FVG / SR)", sym)
                    continue
                # 2026-06-22: per-(SMABO, sym) 10-consec-loss halt gate
                if (self._master_brain
                        and self._master_brain.is_strategy_symbol_halted("smabo", sym)):
                    log.info("[SMABO %s] SKIP: strategy-symbol halt active", sym)
                    continue

                ok = self.executor.open_trade_explicit(
                    sym, sig["direction"], sig["entry"], sig["sl"],
                    sig["tp1"], sig["tp2"], risk_pct=SMABO_RISK_PCT,
                    magic_offsets=(_SMABO_OFF, _SMABO_OFF + 1),
                    strategy_name="SMABO")
                if ok:
                    self._smabo_was_open[sym] = True
                    self._smabo_signal_dedupe[sym] = bar_t  # consume bar ONLY on fill
                    log.info("[SMABO %s] ENTERED %s risk=%.3f%% %s",
                             sym, sig["direction"], SMABO_RISK_PCT,
                             sig.get("reason", ""))
                elif fresh_bar:
                    # Not deduped → will retry next cycle until it fills / bar rolls.
                    log.warning("[SMABO %s] ENTRY FAILED (will retry this bar): "
                                "open_trade_explicit returned False (%s)",
                                sym, sig.get("reason", ""))
            except Exception as e:
                log.warning("[SMABO %s] eval/entry error: %s", sym, e)

        # 1b) Runner breakeven — move the runner leg SL to BE after the TP1 leg
        #     fills (backtest parity 'SL->BE after TP1'). Safe: only tightens.
        for sym in SMABO_WHITELIST:
            try:
                if self.executor.has_smabo_position(sym):
                    self.executor.smabo_move_runner_to_be(sym)
            except Exception as _be:
                log.debug("[SMABO %s] BE-mgmt err: %s", sym, _be)

        # 2) Detect closes (broker-side TP/SL or manual) → arm cooldown
        #    + update consec-loss counter for the kill-switch.
        for sym in list(self._smabo_was_open.keys()):
            try:
                had_prev = self._smabo_was_open.get(sym, False)
                is_open = self.executor.has_smabo_position(sym)
                if had_prev and not is_open:
                    self._smabo_cooldown[sym] = now + SMABO_POST_CLOSE_COOLDOWN_SECS
                    # Realized win/loss from MT5 deal history. The old peak_r
                    # proxy read a DEAD key (plain sym belongs to momentum;
                    # SMABO uses sym+'_smabo', never written) → every close,
                    # TP wins included, counted as a loss and armed the kill.
                    # (2026-07-06 audit — root cause of the silent under-trading.)
                    won = False
                    _pnl = 0.0
                    try:
                        from datetime import timedelta as _td
                        _since = float(getattr(self.executor, "_smabo_entry_time", {})
                                       .get(sym, 0.0))
                        _now = datetime.now(timezone.utc)
                        _frm = (datetime.fromtimestamp(_since - 120, timezone.utc)
                                if _since > 0 else _now - _td(days=2))
                        _deals = self.executor.mt5.history_deals_get(
                            _frm, _now + _td(minutes=5)) or []
                        _cfg = SYMBOLS.get(sym)
                        _magics = ({int(_cfg.magic) + _SMABO_OFF,
                                    int(_cfg.magic) + _SMABO_OFF + 1}
                                   if _cfg else set())
                        _pnl = sum(float(d.profit) + float(getattr(d, "swap", 0.0))
                                   + float(getattr(d, "commission", 0.0))
                                   for d in _deals
                                   if getattr(d, "symbol", "") == sym
                                   and int(getattr(d, "magic", -1)) in _magics)
                        won = _pnl > 0.0
                    except Exception as _we:
                        log.debug("[SMABO %s] won-attribution err: %s", sym, _we)
                    _cl = self._smabo_consec_losses
                    if not won:
                        _cl[sym] = _cl.get(sym, 0) + 1
                        log.info("[SMABO %s] CLOSED (loss, pnl=%.2f) "
                                 "consec_losses=%d/%d",
                                 sym, _pnl, _cl[sym], SMABO_KILL_AFTER_LOSSES)
                        if _cl[sym] >= SMABO_KILL_AFTER_LOSSES:
                            self._smabo_kill_until_date[sym] = today_utc
                            log.warning("[SMABO %s] DAILY KILL ARMED — %d consec "
                                        "losses. Re-enables next UTC midnight.",
                                        sym, _cl[sym])
                    else:
                        _cl[sym] = 0
                        log.info("[SMABO %s] CLOSED (win, pnl=%.2f)", sym, _pnl)
                    # 2026-06-22: per-(SMABO, sym) 10-consec-loss halt
                    try:
                        if self._master_brain:
                            self._master_brain.record_strategy_symbol_close(
                                "smabo", sym, won=won)
                    except Exception as _e:
                        log.debug("[SMABO %s] strat-sym close record err: %s", sym, _e)
                self._smabo_was_open[sym] = is_open
            except Exception as e:
                log.debug("[SMABO %s] post-close check error: %s", sym, e)

    # ═══════════════════════════════════════════════════════════════
    #  FIB50 (Fib-50 Pullback Continuation) — independent book, default OFF
    # ═══════════════════════════════════════════════════════════════
    def _process_fib50(self, equity):
        """Run the Fib-50 pullback-continuation detector across the FIB50
        whitelist. Independent of momentum / FVG / SR / SMABO books — own
        magic range (+4000/+4001) and own risk budget (FIB50_RISK_PCT,
        default 0.20%).

        Operates in TWO modes (config flag FIB50_TRADE_LIVE):
          False: SIGNAL-ONLY — logs every detected setup but does NOT trade.
                 Default — code loads, no orders fire, until empirical
                 validation lands.
          True : LIVE — opens trades via executor.open_trade_explicit using
                 the FIB50_MAGIC_OFFSET book.

        Also enforces a per-day consec-loss kill switch
        (FIB50_KILL_AFTER_LOSSES). Mirrors _process_sma_breakout exactly.
        """
        try:
            from config import (FIB50_ENABLED, FIB50_TRADE_LIVE, FIB50_RISK_PCT,
                                FIB50_MAX_CONCURRENT,
                                FIB50_POST_CLOSE_COOLDOWN_SECS,
                                FIB50_KILL_AFTER_LOSSES, FIB50_WHITELIST,
                                FIB50_MAGIC_OFFSET as _FIB50_OFF)
        except Exception:
            return
        if not FIB50_ENABLED:
            return

        # Lazy-init detector + bookkeeping on first call.
        if not hasattr(self, "_fib50_strategy") or self._fib50_strategy is None:
            try:
                from agent.fib50_strategy import Fib50Strategy
                self._fib50_strategy = Fib50Strategy(self.state)
                self._fib50_was_open = {}
                self._fib50_cooldown = {}
                self._fib50_signal_dedupe = {}
                self._fib50_consec_losses = 0
                self._fib50_kill_until_date = None
                log.info("[FIB50] Fib50Strategy initialized "
                         "(TRADE_LIVE=%s, whitelist=%s)",
                         FIB50_TRADE_LIVE, sorted(FIB50_WHITELIST))
            except Exception as e:
                log.error("[FIB50] init failed: %s", e)
                self._fib50_strategy = None
                return

        now = time.time()
        today_utc = datetime.now(timezone.utc).date()

        if (self._fib50_kill_until_date is not None
                and today_utc <= self._fib50_kill_until_date):
            return
        last_check = getattr(self, "_fib50_last_kill_check_date", None)
        if last_check is None or last_check < today_utc:
            self._fib50_consec_losses = 0
            self._fib50_last_kill_check_date = today_utc

        # 1) Detect signals across the FIB50 whitelist.
        for sym in FIB50_WHITELIST:
            try:
                if sym not in SYMBOLS:
                    continue
                if float(self._fib50_cooldown.get(sym, 0)) > now:
                    continue
                sig = self._fib50_strategy.evaluate(sym)
                if not sig:
                    continue
                bar_t = sig.get("bar_time")
                if self._fib50_signal_dedupe.get(sym) == bar_t:
                    continue
                self._fib50_signal_dedupe[sym] = bar_t

                log.info("[FIB50 %s] SIGNAL %s entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f "
                         "A=%.5f B=%.5f fib50=%.5f  %s",
                         sym, sig["direction"], sig["entry"], sig["sl"],
                         sig["tp1"], sig["tp2"],
                         sig.get("swing_A", 0), sig.get("swing_B", 0),
                         sig.get("fib_50", 0),
                         "[SIGNAL-ONLY]" if not FIB50_TRADE_LIVE else "[LIVE]")

                if not FIB50_TRADE_LIVE:
                    continue

                n_open = sum(1 for s in self._fib50_was_open
                             if self._fib50_was_open.get(s))
                if n_open >= FIB50_MAX_CONCURRENT:
                    log.info("[FIB50 %s] SKIP: max concurrent (%d) reached",
                             sym, n_open)
                    continue

                # Yield to all other strategies to avoid stacked exposure.
                if (self.executor.has_position(sym)
                        or self.executor.has_fvg_position(sym)
                        or self.executor.has_sr_position(sym)
                        or self.executor.has_smabo_position(sym)):
                    log.info("[FIB50 %s] SKIP: existing position "
                             "(momentum / FVG / SR / SMABO)", sym)
                    continue
                # 2026-06-22: per-(FIB50, sym) 10-consec-loss halt gate
                if (self._master_brain
                        and self._master_brain.is_strategy_symbol_halted("fib50", sym)):
                    continue

                ok = self.executor.open_trade_explicit(
                    sym, sig["direction"], sig["entry"], sig["sl"],
                    sig["tp1"], sig["tp2"], risk_pct=FIB50_RISK_PCT,
                    magic_offsets=(_FIB50_OFF, _FIB50_OFF + 1),
                    strategy_name="FIB50")
                if ok:
                    self._fib50_was_open[sym] = True
                    log.info("[FIB50 %s] ENTERED %s risk=%.3f%% %s",
                             sym, sig["direction"], FIB50_RISK_PCT,
                             sig.get("reason", ""))
            except Exception as e:
                log.warning("[FIB50 %s] eval/entry error: %s", sym, e)

        # 2) Detect closes → arm cooldown + update kill-switch counter.
        for sym in list(self._fib50_was_open.keys()):
            try:
                had_prev = self._fib50_was_open.get(sym, False)
                is_open = self.executor.has_fib50_position(sym)
                if had_prev and not is_open:
                    self._fib50_cooldown[sym] = now + FIB50_POST_CLOSE_COOLDOWN_SECS
                    peak_r = 0.0
                    try:
                        peak_r = float(self.executor._peak_profit_r.get(sym, 0.0))
                    except Exception:
                        pass
                    won = peak_r >= 0.5
                    if not won:
                        self._fib50_consec_losses += 1
                        log.info("[FIB50 %s] CLOSED (loss, peak_r=%.2f) "
                                 "consec_losses=%d/%d",
                                 sym, peak_r, self._fib50_consec_losses,
                                 FIB50_KILL_AFTER_LOSSES)
                        if self._fib50_consec_losses >= FIB50_KILL_AFTER_LOSSES:
                            self._fib50_kill_until_date = today_utc
                            log.warning("[FIB50] DAILY KILL ARMED — "
                                        "%d consec losses today. "
                                        "Re-enables at next UTC midnight.",
                                        self._fib50_consec_losses)
                    else:
                        self._fib50_consec_losses = 0
                    # 2026-06-22: per-(FIB50, sym) 10-consec-loss halt
                    try:
                        if self._master_brain:
                            self._master_brain.record_strategy_symbol_close(
                                "fib50", sym, won=won)
                    except Exception as _e:
                        log.debug("[FIB50 %s] strat-sym close record err: %s", sym, _e)
                self._fib50_was_open[sym] = is_open
            except Exception as e:
                log.debug("[FIB50 %s] post-close check error: %s", sym, e)

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

        # ══════════════════════════════════════════════════════════════
        # 2026-06-05: NEWS-EVENT BLACKOUT — skip new entries near tier-1
        # econ events. Industry-standard prop-firm rule; effect on price
        # variance is real even with HFT-decayed direction signal.
        # External module owned by news-blackout agent; ImportError = fail-open.
        # ══════════════════════════════════════════════════════════════
        if NEWS_BLACKOUT_ENABLED:
            try:
                from agent.news_blackout import is_in_blackout
                if is_in_blackout(symbol):
                    return _ret(0, 0, 0, 0, "FLAT", "NEWS_BLACKOUT")
            except ImportError:
                pass  # module not yet deployed — fail-open
            except Exception as _ne:
                log.debug("[%s] News-blackout check skipped: %s", symbol, _ne)

        # ══════════════════════════════════════════════════════════════
        # 2026-06-05: DAILY-LOSS KILL GUARD — block all new entries once
        # the 3% kill has fired for the current UTC day. Flag is cleared
        # by _run_cycle daily-reset block at midnight UTC.
        # ══════════════════════════════════════════════════════════════
        if DAILY_LOSS_KILL_ENABLED and getattr(self, '_day_kill_fired_today', False):
            _today = datetime.now(timezone.utc).date()
            _until = getattr(self, '_day_kill_until', None)
            if _until is None or _today < _until:
                return _ret(0, 0, 0, 0, "FLAT", "DAILY_KILL_ACTIVE")
            # Belt-and-braces: flag survived past the reset window — clear it
            # here too so we don't hang indefinitely if the day-rollover block
            # somehow didn't run (e.g. process started after midnight with
            # stale state in the future).
            self._day_kill_fired_today = False
            self._day_kill_until = None

        # ══════════════════════════════════════════
        #  PRE-CHECK: Pullback fill (deferred signal)
        # ══════════════════════════════════════════
        if symbol in self._pending_pullback and not self.executor.has_position(symbol):
            pb = self._pending_pullback[symbol]
            # 2026-05-26 audit fix: compute REAL M1-bar elapsed from wall clock,
            # not per-cycle increment. Clamp to [0, 240] to defend against
            # NTP step / sleep+resume glitches (legacy `+= 1` per 0.5s cycle
            # made PULLBACK_MAX_WAIT_BARS=1 expire in first cycle).
            cur_minute = int(time.time() // 60)
            elapsed_min = cur_minute - pb.get("signal_minute", cur_minute)
            pb["bars_waited"] = max(0, min(elapsed_min, 240))
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
                # 2026-05-29 UNIT FIX: bars_waited is elapsed MINUTES, but
                # PULLBACK_MAX_WAIT_BARS is in H1 bars (the backtest's unit).
                # 1 H1 bar = 60 min. Comparing minutes directly to "bars" expired
                # the window in ~1 min, so every (re-)entry fell straight through to
                # direct entry and never actually waited for the retrace. ×60 makes
                # the live wait match the backtested pullback behavior.
                _expiry_min = PULLBACK_MAX_WAIT_BARS * 60
                if pb["bars_waited"] >= _expiry_min:
                    # 2026-05-16: fallback to direct entry instead of skipping.
                    # Old behavior (skip) caused 136/136 expiry rate that disabled the
                    # feature. Mirror backtest's pullback-or-signal-close semantics so
                    # live↔backtest converge. Per feedback_no_skip_trades: never miss
                    # a signal — at worst take it at a slightly later price.
                    log.info("[%s] PULLBACK EXPIRED after %d min (%d H1 bar) — fallback to direct entry",
                             symbol, pb["bars_waited"], PULLBACK_MAX_WAIT_BARS)
                    d, rs, rp, sa = pb["direction"], pb["score"], pb["risk_pct"], pb["atr"]
                    comp_l, comp_s = pb["comp_long"], pb["comp_short"]
                    # 2026-06-03: Re-validate the stored score against the
                    # CURRENT regime's MIN_SCORE before falling back. Without
                    # this re-check, the bypass at this site let USDJPY enter
                    # at raw_score=4.5 in low_vol (gate is 6.5) on 2026-06-01
                    # — pullback armed under one regime, fallback fired hours
                    # later when MIN_SCORE for the current regime had risen.
                    # HARD floor 5.0 in executor catches the worst of it; this
                    # closes the structural hole at the brain layer.
                    _pb_regime = pb.get("regime", "")
                    try:
                        _cur_min = self._get_adaptive_min_score(_pb_regime, symbol)
                        if float(rs) < float(_cur_min):
                            log.warning("[%s] PULLBACK FALLBACK BLOCKED: stored score=%.2f < current MIN_SCORE %.2f (regime=%s)",
                                        symbol, float(rs), float(_cur_min), _pb_regime)
                            self._pending_pullback.pop(symbol)
                            return _ret(0, 0, pb.get("signal_quality", 0), 0, d,
                                        "PULLBACK_BELOW_MIN_SCORE")
                    except Exception as e:
                        log.debug("[%s] pullback re-validate failed: %s", symbol, e)
                    # 2026-06-03 CTO audit (A5): ATR-freshness gate. Reject
                    # fallback if H1 ATR has expanded > 1.5× since signal arm.
                    # Catches the "armed in low-vol → 90min later vol exploded
                    # → fallback chases a now-different setup" pattern. Pullback
                    # FALLBACK trades show 44% EarlyLossCut_T1- rate vs 20% for
                    # filled pullbacks = stale-signal chasing.
                    try:
                        _arm_atr = float(pb.get("atr", 0) or 0)
                        _cur_atr = float(self._get_atr(symbol)) if hasattr(self, '_get_atr') else 0.0
                        if _arm_atr > 0 and _cur_atr > 0:
                            _ratio = _cur_atr / _arm_atr
                            if _ratio > 1.5:
                                log.warning("[%s] PULLBACK FALLBACK BLOCKED: ATR expanded %.2f× since arm (regime drift)",
                                            symbol, _ratio)
                                self._pending_pullback.pop(symbol)
                                return _ret(0, 0, pb.get("signal_quality", 0), 0, d,
                                            "PULLBACK_ATR_DRIFT")
                    except Exception as e:
                        log.debug("[%s] pullback ATR-freshness check failed: %s", symbol, e)
                    self._pending_pullback.pop(symbol)
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
            # Asymmetric cooldown: short same-direction-only after WIN, long
            # both-directions after LOSS. Executor exposes both via
            # _external_close_direction + _external_close_was_win.
            # 2026-05-26 audit fix: pop dir + win atomically under executor's lock
            # to eliminate the race where a new external close arrives between
            # the two pops and we get mismatched (dir, win) pair.
            ext_dirs = getattr(self.executor, '_external_close_direction', {})
            ext_wins = getattr(self.executor, '_external_close_was_win', {})
            _exec_lock = getattr(self.executor, '_lock', None)
            if _exec_lock is not None:
                with _exec_lock:
                    closed_dir = (ext_dirs.pop(symbol, "FLAT") if ext_dirs else "FLAT") or "FLAT"
                    was_win = bool(ext_wins.pop(symbol, False) if ext_wins else False)
            else:
                closed_dir = (ext_dirs.pop(symbol, "FLAT") if ext_dirs else "FLAT") or "FLAT"
                was_win = bool(ext_wins.pop(symbol, False) if ext_wins else False)
            # 2026-05-29 cooldown redesign. peak_r snapshot from executor lets us
            # detect big wins; master_brain streak drives loss escalation.
            _peak_r = 0.0
            try:
                _peak_r = float(getattr(self.executor, "_last_close_peak_r", {}).get(symbol, 0.0) or 0.0)
            except Exception:
                pass
            if was_win and closed_dir in ("LONG", "SHORT"):
                # WIN. Big win (caught a large move now likely exhausted) →
                # longer same-dir cooldown so we don't give it back. Normal win →
                # short same-dir window; opposite direction stays free.
                if _peak_r >= BIG_WIN_R_TRIGGER:
                    self._arm_cooldown(symbol, POST_BIG_WIN_SECS,
                                        f"POST_BIG_WIN_{closed_dir}({_peak_r:.1f}R)",
                                        blocked_direction=closed_dir)
                else:
                    self._arm_cooldown(symbol, COOLDOWN_WIN_SECS,
                                        f"WIN_{closed_dir}", blocked_direction=closed_dir)
                # Win resets the per-(symbol,direction) attempt-strike backoff.
                self._attempt_strikes.pop(f"{symbol}#{closed_dir}", None)
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
                # 2026-05-29: escalate the loss cooldown by consecutive-loss streak
                # (from master_brain, now persisted). 2 consec → 2×, 3 → 4×.
                try:
                    _streak = 0
                    mb = getattr(self, "_master_brain", None)
                    if mb is not None:
                        _streak = int(getattr(mb, "_symbol_losses", {}).get(symbol, 0))
                    _cd_secs = int(_cd_secs * STREAK_COOLDOWN_MULT.get(_streak, 1.0))
                except Exception:
                    pass
                self._arm_cooldown(symbol, _cd_secs,
                                    "LOSS" if closed_dir != "FLAT" else "BROKER_CLOSE",
                                    blocked_direction="BOTH")
                # 2026-05-29: per-(symbol,direction) exponential backoff. Each
                # consecutive same-direction loss roughly doubles the lockout for
                # THAT direction (30→60→120→240min, cap 4h), leaving the opposite
                # direction + other symbols fully tradeable. Structurally kills
                # the "re-fire the same losing setup" cascade (e.g. USDCAD 5×LONG).
                if closed_dir in ("LONG", "SHORT"):
                    _k = f"{symbol}#{closed_dir}"
                    _strikes = self._attempt_strikes.get(_k, 0) + 1
                    self._attempt_strikes[_k] = _strikes
                    _backoff = min(ATTEMPT_BACKOFF_BASE_SECS * (2 ** (_strikes - 1)),
                                   ATTEMPT_BACKOFF_CAP_SECS)
                    self._arm_cooldown(symbol, _backoff,
                                        f"ATTEMPT_BACKOFF_{closed_dir}(x{_strikes})",
                                        blocked_direction=closed_dir)
                    try:
                        self.state.update_agent("attempt_strikes", dict(self._attempt_strikes))
                    except Exception:
                        pass

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
            self._pending_pullback.pop(symbol, None)
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
        # 2026-06-04 CTO audit B9: regime classifier now returns "unknown"
        # for NaN BBW/ADX or exception paths (instead of polluting low_vol).
        # Caller must skip entries when classifier fails — we have no basis
        # for entry decisions without a regime.
        if regime == "unknown":
            self._log_decision(symbol, 0, 0, "FLAT", "REGIME_UNKNOWN", None, None,
                               "SKIP (regime classifier returned unknown — NaN/exception)")
            return self._build_flat_return(symbol, "REGIME_UNKNOWN") \
                if hasattr(self, '_build_flat_return') else \
                {"long_score": 0, "short_score": 0, "signal_quality": 0,
                 "min_quality": 0, "atr": 0, "regime": "unknown",
                 "direction": "FLAT", "gate": "REGIME_UNKNOWN"}

        # Pass per-symbol+regime learned component weights so the RL system
        # actually influences scoring (was a no-op for the life of the project).
        rl_weights = None
        if self._rl_learner is not None:
            try:
                rl_weights = self._rl_learner.get_weights(symbol, regime=regime)
            except Exception:
                rl_weights = None
        # 2026-05-26 audit fix #72: merge auto_tuned static component weights
        # under RL-learned weights. Previously COMPONENT_WEIGHTS_AUTO was dead
        # code — never read by either brain.py or momentum_scorer.py. Every
        # per-symbol weight recommendation (AUDJPY +$219, USOUSD +$614, etc.)
        # was unactionable until this wiring landed.
        try:
            from signals.momentum_scorer import get_component_weights
            effective_weights = get_component_weights(symbol, rl_weights)
        except Exception:
            effective_weights = rl_weights
        long_score, short_score, comp_long, comp_short = _score_with_components(
            ind, bi, weights=effective_weights)
        long_score = float(long_score)
        short_score = float(short_score)
        atr_val = float(ind["at"][bi])

        # 1d. Normalize to 0-100 scale
        raw_score = max(long_score, short_score)
        signal_quality = min(100.0, raw_score / SIGNAL_QUALITY_DIVISOR * 100)

        # ══════════════════════════════════════════════════════════════
        # 2026-06-21: ANCHORED VWAP REJECTION — momentum score BOOSTER (+1).
        # When price tests today's session VWAP (or rolling 24-bar anchor)
        # and rejects (wick pierces, body closes back on the leading-score
        # side) within the last `_AVWAP_LOOKBACK` bars, add `_AVWAP_BOOST_AMT`
        # to the leading raw score and recompute signal_quality. Decoupled,
        # fail-OPEN (any exception or missing data → no booster, no block).
        # Default OFF behind ANCHORED_VWAP_BOOSTER_ENABLED config flag.
        # ══════════════════════════════════════════════════════════════
        if (ANCHORED_VWAP_BOOSTER_ENABLED
                and _evaluate_avwap_booster is not None
                and h1_df is not None and len(h1_df) >= 26):
            try:
                _cand_dir = "LONG" if long_score >= short_score else "SHORT"
                _h = h1_df["high"].values.astype(np.float64)
                _l = h1_df["low"].values.astype(np.float64)
                _c = h1_df["close"].values.astype(np.float64)
                _vcol = ("tick_volume" if "tick_volume" in h1_df.columns
                         else ("volume" if "volume" in h1_df.columns else None))
                if _vcol is not None:
                    _v = h1_df[_vcol].values.astype(np.float64)
                    _av_res = _evaluate_avwap_booster(
                        _h, _l, _c, _v, _cand_dir,
                        anchor_bars=int(_AVWAP_ANCHOR_BARS),
                        lookback=int(_AVWAP_LOOKBACK),
                        boost_amount=float(_AVWAP_BOOST_AMT),
                    )
                    if _av_res.get("verdict") == "BOOST":
                        _bump = float(_av_res.get("boost", 0.0) or 0.0)
                        if _bump > 0.0:
                            if _cand_dir == "LONG":
                                long_score = float(long_score + _bump)
                            else:
                                short_score = float(short_score + _bump)
                            raw_score = max(long_score, short_score)
                            signal_quality = min(
                                100.0, raw_score / SIGNAL_QUALITY_DIVISOR * 100)
                            log.info(
                                "[%s] AVWAP rejection booster +%.2f "
                                "(dir=%s, freshness=%s bars, vwap=%.5f)",
                                symbol, _bump, _cand_dir,
                                _av_res.get("freshness"),
                                float(_av_res.get("vwap_now") or 0.0),
                            )
            except Exception as _avwap_err:
                # Fail OPEN — log and continue without the booster.
                log.debug("[%s] AVWAP booster skipped: %s", symbol, _avwap_err)

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

        # ══════════════════════════════════════════════════════════════
        # 2026-06-05: VWAP ENTRY GATE — index intraday momentum.
        # Research: Zarattini "Beat the Market" (SSRN 4824172). The full
        # edge of intraday momentum on US indices comes from being on the
        # correct side of session VWAP — Sharpe 1.33 over 17yrs on SPY.
        # LONG below VWAP / SHORT above VWAP is mean-revert against trend
        # and historically blows up the most.
        #
        # Implementation: session VWAP computed inline from M15 bars of
        # the current UTC day (the OrderFlowIntel module's _compute_vwap
        # is a rolling 20-bar VWAP — wrong granularity for "session" cut).
        # ══════════════════════════════════════════════════════════════
        if (VWAP_GATE_ENABLED and direction in ("LONG", "SHORT")
                and symbol in VWAP_GATE_SYMBOLS):
            try:
                _m15 = self.state.get_candles(symbol, 15)
                if _m15 is not None and len(_m15) >= 4:
                    # Filter to current UTC trading day. M15 'time' is a
                    # pd.Timestamp (UTC) per tick_streamer convention.
                    _today_utc = datetime.now(timezone.utc).date()
                    _times = _m15["time"]
                    if hasattr(_times.iloc[-1], "date"):
                        _day_mask = _times.apply(lambda t: t.date() == _today_utc)
                    else:
                        # Epoch-seconds fallback
                        _day_start_ts = datetime(_today_utc.year,
                                                 _today_utc.month,
                                                 _today_utc.day,
                                                 tzinfo=timezone.utc).timestamp()
                        _day_mask = _times.apply(lambda t: float(t) >= _day_start_ts)
                    _sess = _m15[_day_mask]
                    if len(_sess) >= 2:
                        _h = _sess["high"].values.astype(np.float64)
                        _l = _sess["low"].values.astype(np.float64)
                        _c = _sess["close"].values.astype(np.float64)
                        _vcol = "tick_volume" if "tick_volume" in _sess.columns else (
                            "volume" if "volume" in _sess.columns else None)
                        if _vcol is not None:
                            _v = _sess[_vcol].values.astype(np.float64)
                            _typical = (_h + _l + _c) / 3.0
                            _vol_sum = float(np.sum(_v))
                            if _vol_sum > 0:
                                _vwap = float(np.sum(_typical * _v) / _vol_sum)
                                _cur_price = float(h1_df["close"].iloc[-1])
                                if direction == "LONG" and _cur_price < _vwap:
                                    self._log_decision(
                                        symbol, long_score, short_score, direction,
                                        "VWAP_REJECT", None, None,
                                        "LONG blocked: price %.5f < VWAP %.5f" % (_cur_price, _vwap))
                                    return _ret(long_score, short_score, signal_quality,
                                                min_quality, "FLAT", "VWAP_REJECT",
                                                atr=atr_val, regime=regime,
                                                vwap=_vwap, cur_price=_cur_price)
                                if direction == "SHORT" and _cur_price > _vwap:
                                    self._log_decision(
                                        symbol, long_score, short_score, direction,
                                        "VWAP_REJECT", None, None,
                                        "SHORT blocked: price %.5f > VWAP %.5f" % (_cur_price, _vwap))
                                    return _ret(long_score, short_score, signal_quality,
                                                min_quality, "FLAT", "VWAP_REJECT",
                                                atr=atr_val, regime=regime,
                                                vwap=_vwap, cur_price=_cur_price)
            except Exception as _ve:
                log.warning("[%s] VWAP gate skipped: %s", symbol, _ve)

        # 1g. CONFIRMATION GATE (tightened 2026-06-04 CTO QUALITY OVERHAUL)
        # Was: trending regime requires at least 1 of {supertrend, breakout,
        # trend_persist}. BTC #753 -3R loss had ALL THREE at 0. Audit A2
        # found candle_pattern + trend_persist are strongest predictors
        # (53% WR, 40% absent-loss).
        # NEW (2026-06-04): require N-of-5 from the expanded set:
        #   {supertrend, breakout, trend_persist, candle_pattern, ema_stack}
        # - trending regime: require 3 of 5 (was 1 of 3) — quality stack
        # - volatile regime: require 2 of 5 — slight protection on spikes
        # - ranging/low_vol: untouched (mean-reversion may legitimately have
        #   silent trend components — gate would over-block)
        if regime in ("trending", "volatile"):
            _comp_dir = comp_long if direction == "LONG" else comp_short
            _confirm_keys = ("supertrend", "breakout", "trend_persist",
                             "candle_pattern", "ema_stack")
            try:
                _confirms = sum(
                    1 for _k in _confirm_keys
                    if float(_comp_dir.get(_k, 0) or 0) > 0
                )
            except Exception:
                _confirms = 5  # fail-open on missing component data
            _need = 3 if regime == "trending" else 2
            if _confirms < _need:
                self._log_decision(symbol, long_score, short_score, direction,
                                   "CONFIRM_MISSING", None, None,
                                   "%s confirms %d/5 < required %d" % (regime, _confirms, _need))
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

        # 2026-06-03: yield to active FVG trade on this symbol. Previously
        # has_position() only saw momentum's magic range (cfg.magic + [0,1,2])
        # — momentum was BLIND to FVG positions (magic + [1000,1001]), so both
        # strategies could open on the same symbol simultaneously. FVG already
        # yields to momentum at brain.py:1197; this closes the loop.
        if self.executor.has_fvg_position(symbol):
            self._log_decision(symbol, long_score, short_score, direction,
                               "FVG_POSITION_OPEN", None, None,
                               "SKIP (FVG position active — yielding)")
            return {**base_ret, "direction": direction, "gate": "FVG_POSITION_OPEN"}

        # 2026-06-21: yield to active SMABO trade on this symbol (4th-strategy
        # mutual-exclusion, mirror of the FVG yield above). SMABO runs on its
        # own magic range (+3000/+3001) so momentum dispatchers are blind to it
        # unless they explicitly check here.
        if self.executor.has_smabo_position(symbol):
            self._log_decision(symbol, long_score, short_score, direction,
                               "SMABO_POSITION_OPEN", None, None,
                               "SKIP (SMABO position active — yielding)")
            return {**base_ret, "direction": direction, "gate": "SMABO_POSITION_OPEN"}

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
        # 2026-06-03 CTO audit (A8): switched from CYCLE-based lookback (last
        # 3 cycles) to TIME-WINDOWED lookback (last 5 min). The cycle approach
        # was blind for crypto: BTC ticks every ~3s → only 9s of memory →
        # after ~10s of score≥7, the gate disarmed. CHFJPY (15min tick) was
        # fine. Time-window adapts naturally to any tick rate.
        # Block entries where raw_score jumped sharply within last 5 minutes.
        # Heuristic: if raw_score >= 7.0 NOW but ANY score in last 5 min
        # was < 6.0, this is a late-confirm trap — block it.
        LATE_HIGH = 7.0
        LATE_LOW = 6.0
        LATE_WINDOW_SECS = 300   # 5 min
        _now_ts = time.time()
        # _score_hist is now {symbol: [(ts, raw_score), ...]} time-pruned.
        # First load migrates legacy list-of-floats to list-of-tuples.
        _raw_hist = self._score_hist.get(symbol, [])
        if _raw_hist and not isinstance(_raw_hist[0], tuple):
            # Legacy format — discard, fresh start with time-stamped data
            _raw_hist = []
        # Append new point + prune to window
        _hist = [(t, s) for (t, s) in _raw_hist if _now_ts - t <= LATE_WINDOW_SECS]
        _hist.append((_now_ts, raw_score))
        # Cap at 200 entries as memory safety (5min × 60tps = 300 hard ceiling)
        if len(_hist) > 200:
            _hist = _hist[-200:]
        self._score_hist[symbol] = _hist
        # Gate check: any prior score (excluding the just-appended one) < LATE_LOW?
        _prior = [s for (t, s) in _hist[:-1]]
        if (raw_score >= LATE_HIGH and _prior and
                any(s < LATE_LOW for s in _prior)):
            self._log_decision(symbol, long_score, short_score, direction,
                               "LATE_MOMENTUM", None, None,
                               "SKIP (raw_score %.1f after recent <%.1f in last %ds — late confirm)"
                               % (raw_score, LATE_LOW, LATE_WINDOW_SECS))
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
        elif _regime_bias == "FLAT":
            # 2026-05-29: full both-side block for this (symbol, regime). Used
            # where BOTH directions bleed (e.g. BTCUSD low_vol). Hard skip.
            # (m15_dir/meta_prob not yet computed at this point in the pipeline.)
            self._log_decision(symbol, long_score, short_score, direction,
                               "REGIME_FLAT_BLOCK", "FLAT", None,
                               "SKIP (both directions blocked in %s regime)" % regime)
            return {**base_ret, "direction": "FLAT", "gate": "REGIME_FLAT_BLOCK"}
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
        # Per-symbol params from auto_tuned.RANGE_FILTER_PARAMS_AUTO (only
        # symbols where 5-fold WF proved the filter adds > $50 / 180d).
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
                    # Not in whitelist — skip filter (don't apply globally)
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

        # Gate 3e: CHASE guard (2026-06-03 CTO audit A1 + 2026-06-15 expansion)
        # 8d journal slice: LONG entries with entry_price in top 15% of trailing
        # 4H range (pos_4h >= 0.85) had WR 11% / -$31 PnL.
        # 2026-06-15 user observation: "we are entering on swing high don't we
        # have liquidity levels that is swept then we have to enter" — momentum
        # was chasing tops on every non-whitelisted sym (XAU/BTC/DJ30/etc.).
        # Two extensions:
        #   (a) Universal LONG-CHASE — gate ALL syms, not just the original
        #       4-sym whitelist. Threshold stays 0.85 (top 15% of 4h range).
        #   (b) Symmetric SHORT-CHASE-BOTTOM — gate SHORTs at pos_4h <= 0.15.
        #       Prior comment "SHORTs at extremes had +$6 PnL / 56% WR" was
        #       2026-06-03 data on disabled syms (USDCAD/USDJPY); the residual
        #       SHORT bleeders fire at 4h-range-bottom = chasing the move down.
        # Full ICT-style "sweep then enter" is a separate build (FVG strategy
        # already does it for its 7 whitelist syms; momentum doesn't yet).
        try:
            CHASE_LOOKBACK_BARS = 4         # H1 bars => trailing 4h
            CHASE_TOP_THRESHOLD = 0.85      # block LONG at pos_4h >= 0.85
            CHASE_BOT_THRESHOLD = 0.15      # block SHORT at pos_4h <= 0.15
            _hi_arr = ind.get("h")
            _lo_arr = ind.get("l")
            _cl_arr = ind.get("c")
            if (_hi_arr is not None and _lo_arr is not None
                    and _cl_arr is not None and bi >= 0
                    and bi < len(_hi_arr)):
                _start = max(0, bi - CHASE_LOOKBACK_BARS + 1)
                _hi4 = float(max(_hi_arr[_start:bi + 1]))
                _lo4 = float(min(_lo_arr[_start:bi + 1]))
                if _hi4 > _lo4:
                    _pos = (float(_cl_arr[bi]) - _lo4) / (_hi4 - _lo4)
                    if direction == "LONG" and _pos >= CHASE_TOP_THRESHOLD:
                        self._log_decision(symbol, long_score, short_score,
                                           direction, "LONG_CHASE_TOP", None, None,
                                           "SKIP (LONG at pos_4h=%.2f, top of recent range)" % _pos)
                        return {**base_ret, "direction": direction, "gate": "LONG_CHASE_TOP"}
                    if direction == "SHORT" and _pos <= CHASE_BOT_THRESHOLD:
                        self._log_decision(symbol, long_score, short_score,
                                           direction, "SHORT_CHASE_BOTTOM", None, None,
                                           "SKIP (SHORT at pos_4h=%.2f, bottom of recent range)" % _pos)
                        return {**base_ret, "direction": direction, "gate": "SHORT_CHASE_BOTTOM"}
        except Exception as e:
            log.debug("[%s] CHASE guard check failed: %s", symbol, e)

        # Gate 3f: ICT liquidity-sweep gate (2026-06-16)
        # Sniper-grade structural filter. CHASE guard (3e) blocks entries at
        # the top/bottom of the trailing 4h range; this gate goes further and
        # demands evidence that price has ALREADY swept stops (taken out a
        # recent swing) and reclaimed the level — the classic ICT stop-hunt-
        # then-reverse signature. No sweep → REJECT.
        #
        # LONG  : within last N H1 bars, ∃ bar where low < prior 5-bar swing-
        #         low AND close back above that swing-low (= longs got
        #         stopped, price reclaimed).
        # SHORT : symmetric — sweep ABOVE recent swing-high + reclaim below.
        #
        # Skips: direction != LONG/SHORT, flag disabled, or insufficient H1
        # history (graceful pass-through — the CHASE guard already protects
        # against the worst late entries).
        try:
            if (ICT_SWEEP_REQUIRED_FOR_MOMENTUM
                    and direction in ("LONG", "SHORT")
                    and h1_df is not None
                    and len(h1_df) >= ICT_SWEEP_FRACTAL_N + 2):
                _ict_lb = int(ICT_SWEEP_LOOKBACK_BARS)
                _ict_n = int(ICT_SWEEP_FRACTAL_N)
                _highs = h1_df["high"].values
                _lows = h1_df["low"].values
                _closes = h1_df["close"].values
                _swept = detect_liquidity_sweep(
                    _highs, _lows, _closes, direction,
                    lookback=_ict_lb, n=_ict_n,
                )
                if not _swept:
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "ICT_NO_SWEEP", None, None,
                                       "SKIP (%s no liquidity sweep+reclaim in last %d H1 bars)"
                                       % (direction, _ict_lb))
                    return {**base_ret, "direction": direction, "gate": "ICT_NO_SWEEP"}
        except Exception as e:
            # Fail OPEN — don't crash brain on data hiccups. CHASE guard above
            # is the backstop against the worst chase-the-extreme entries.
            log.debug("[%s] ICT sweep gate check failed: %s", symbol, e)

        # Gate 3g: ICT Discount/Premium zone gate (2026-06-19)
        # Companion to Gate 3f. Computes the H1 dealing range over the
        # last DISCOUNT_PREMIUM_LOOKBACK_BARS H1 bars and rejects any
        # entry on the wrong side of equilibrium (LONG in premium /
        # SHORT in discount). With strict_mode=True only DEEP zones
        # approve. Default OFF — ships dark for shadow A/B comparison.
        # Fail-OPEN: data hiccups don't block trades; Gate 3e/3f already
        # filter the worst chase entries.
        if DISCOUNT_PREMIUM_GATE_ENABLED:
            try:
                from agent.expert import evaluate_zone_gate
                h1_df = self.state.get_candles(symbol, 60)
                highs = h1_df['high'].values; lows = h1_df['low'].values; closes = h1_df['close'].values
                verdict = evaluate_zone_gate(highs, lows, closes, direction, strict_mode=DISCOUNT_PREMIUM_STRICT_MODE)
                if not verdict['approved']:
                    self._log_decision(symbol, long_score, short_score, direction, "DISCOUNT_PREMIUM",
                                       None, None, verdict['reason'])
                    return {**base_ret, 'direction': direction, 'gate': 'DISCOUNT_PREMIUM'}
            except Exception as e:
                log.debug("[%s] zone gate error: %s", symbol, e)  # fail-open

        # ════════════════════════════════════════════════════════════════
        # EXPERT_MODE — orchestrator slot (Gate 3.5)
        # ════════════════════════════════════════════════════════════════
        # Runs AFTER Gate 3f (structural ICT sweep) and BEFORE Gate 4
        # (position management / m15 confirm / META / MasterBrain) so
        # that cheap rejects (news/regime/D1-bias/SCSL/OB/Wyckoff/TV) can
        # short-circuit before the expensive ML and master-brain calls.
        # Conviction + dynamic SL/TP run last and surface optional fields
        # (size_mult / sl / tp1 / tp2 / runner) that the executor entry
        # path can pick up at PHASE 4. Fail-OPEN: orchestrator errors
        # never block trades — they degrade to the legacy pipeline.
        expert_verdict = None
        expert_size_mult = 1.0
        if self._expert_gate is not None and direction in ("LONG", "SHORT"):
            try:
                _loc = locals()
                tick = self.state.get_tick(symbol)
                entry_px = float(tick.bid) if tick and hasattr(tick, "bid") else 0.0
                # M15 candles for ASAT / dynamic-SLTP structural lookups.
                try:
                    m15_df = self.state.get_candles(symbol, 15)
                except Exception:
                    m15_df = None
                # m15_dir is not yet computed at this stage (it's set inside
                # Gate 5 below); resolve it cheaply if a sub-component needs it.
                try:
                    _m15_dir = self._get_m15_direction(symbol)
                except Exception:
                    _m15_dir = None
                ctx = {
                    "symbol":          symbol,
                    "direction":       direction,
                    "h1_df":           h1_df,
                    "m15_df":          m15_df,
                    "d1_df":           None,
                    "ind":             ind,
                    "bi":              _loc.get("bi", 0) or 0,
                    "atr":             float(_loc.get("atr_val", 0.0) or 0.0),
                    "entry_px":        entry_px,
                    "spread":          0.0,
                    "raw_score":       float(raw_score),
                    "signal_quality":  float(signal_quality),
                    "mtf_aligned":     int(_loc.get("mtf_aligned", 0) or 0),
                    "m15_dir":         _m15_dir,
                    "regime":          regime,
                    "min_quality":     float(_loc.get("min_quality", 0.0) or 0.0),
                    "comp_long":       _loc.get("comp_long"),
                    "comp_short":      _loc.get("comp_short"),
                    "signal_source":   "momentum",
                    "signal_class":    "MOMENTUM",
                    "sl_mult_base":    ATR_SL_MULTIPLIER,
                }
                expert_verdict = self._expert_gate.evaluate(ctx)
                if expert_verdict and expert_verdict.get("verdict") == "REJECT":
                    comp = expert_verdict.get("component", "expert")
                    reason = expert_verdict.get("reason", "EXPERT_REJECT")
                    gate_name = "EXPERT_" + str(comp).upper()
                    self._log_decision(symbol, long_score, short_score,
                                       direction, gate_name, None, None,
                                       "SKIP (%s)" % reason)
                    return {**base_ret, "direction": direction,
                            "gate": gate_name, "expert_reason": reason}
                if expert_verdict:
                    expert_size_mult = float(expert_verdict.get("size_mult", 1.0) or 1.0)
            except Exception as e:
                log.debug("[%s] ExpertGate evaluate failed: %s", symbol, e)
                expert_verdict = None
                expert_size_mult = 1.0

        # Gate 4: Position management (hold / reversal)
        current_dir = self.executor.get_position_direction(symbol)
        has_pos = current_dir != "FLAT"

        if has_pos and current_dir == direction:
            return {**base_ret, "direction": direction, "gate": "HOLD_SWING"}

        if has_pos and current_dir != direction:
            # DISABLED 2026-05-01: live 7d journal showed reversal exits had
            # 0% WR across 8 trades, -$29.70 PnL. Reversals always book a
            # loss to flip direction; trail/SL/TP handle exits cleanly. If
            # market truly reverses, we'll miss it — but losing -$30/wk to
            # churn is worse than missing the occasional flip.
            return {**base_ret, "direction": direction, "gate": "REVERSAL_DISABLED"}

        # Gate 5: MTF confirmation (M15 agrees OR high conviction override)
        # 2026-05-13 tightened: was passing any FLAT M15 on trending symbols
        # (Crypto/Index) regardless of quality — covered ~40% of bleeding
        # entries on DJ30/US2000. Now FLAT M15 also requires quality >= 70.
        # 2026-06-03 CTO audit (A4): MTF bypass collapses above signal_quality
        # ≥ 83 (raw ≥ 10.0). That bucket had WR 22% / PF 0.06 over 23 trades
        # vs PF 1.33-1.38 in the 7.5-10.0 raw band — high RL-boosted scores
        # are a REVERSE signal. Plus per-symbol toxicity: ETHUSD/USDCAD/EURUSD
        # bypass trades = catastrophic. Two new restrictions:
        #   (1) BYPASS_CEILING — cap bypass at signal_quality < 83 (= raw < 10)
        #   (2) BYPASS_WHITELIST — only listed symbols get M15 bypass at all
        # The 5 whitelist symbols are the ONLY ones with PF≥1 on bypass.
        MTF_BYPASS_CEILING = 83.0
        MTF_BYPASS_WHITELIST = {"USOUSD", "XAUUSD", "CHFJPY", "UK100.r", "BTCUSD"}
        m15_dir = self._get_m15_direction(symbol)
        m15_agrees = (m15_dir == direction)
        m15_flat = (m15_dir == "FLAT")
        in_bypass_wl = symbol in MTF_BYPASS_WHITELIST
        below_ceiling = signal_quality < MTF_BYPASS_CEILING
        m15_pass = (m15_agrees or
                    (in_bypass_wl and below_ceiling and
                     signal_quality >= MTF_OVERRIDE_QUALITY) or
                    (in_bypass_wl and below_ceiling and m15_flat and
                     signal_quality >= 65))
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
            # 2026-06-22: per-(strategy, symbol) N-consec-loss halt check.
            # Cheap upfront test — if this sym is halted for momentum, skip
            # all the heavier gates below.
            try:
                if self._master_brain.is_strategy_symbol_halted("momentum", symbol):
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "STRAT_SYM_HALT", m15_dir, meta_prob,
                                       "SKIP (momentum:%s halted by 10-consec-loss rule)" % symbol)
                    return {**base_ret, "direction": direction, "gate": "STRAT_SYM_HALT",
                            "m15_dir": m15_dir, "meta_prob": meta_prob,
                            "master_reason": "momentum:%s halted" % symbol}
            except Exception as _e:
                log.debug("strat-sym halt check err: %s", _e)
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
                # 2026-06-04 CTO audit B6 fix: was `risk_pct = entry_eval.risk_pct`
                # which OVERWROTE the SYMBOL_RISK_CAP seed at line 2085.
                # Per-symbol cap was unenforced after MasterBrain. Use min()
                # so MasterBrain can only LOWER risk vs the seed, never raise it
                # above the symbol-specific cap.
                _master_risk = float(entry_eval.get("risk_pct", risk_pct))
                risk_pct = min(risk_pct, _master_risk)
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

        # GVZ-VRP proxy (BIS WP 619, Prokopczuk et al. JBF 2017). XAUUSD only.
        # ATR-14 percentile vs 90-bar window: quiet=trend regime, panic=mean-rev.
        # Symmetric LONG+SHORT. Downsize joins min-chain unconditionally; upsize
        # gated by self._xau_vrp_boost (default False) per no-multiplier-stack rule.
        vrp_mult = self._xau_vol_regime_mult(symbol, ind, bi)

        # DE-STACK: take the MIN of the upside multipliers (most conservative).
        # Real-money safety — never compound boosts. Adaptive sizing still works
        # because any single multiplier dropping below 1.0 still reduces risk.
        adaptive_mult = min(conv_mult, rl_mult, sess_dow_mult, vrp_mult)
        risk_pct *= adaptive_mult
        log.info("[%s] ADAPTIVE x%.2f = min(conv=%.2f rl=%.2f sess*dow=%.2f vrp=%.2f) → risk=%.3f%%",
                 symbol, adaptive_mult, conv_mult, rl_mult, sess_dow_mult, vrp_mult, risk_pct)

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
                # 2026-06-04 CTO audit B6: when edge_score=0 (insufficient
                # data, default state), formula gave 0.60 = permanent 40%
                # drag on every trade for unfamiliar symbols. Fixed: treat
                # edge_score==0 as "no data, no discount" (mult=1.0); only
                # discount when edge_score < 0 (proven negative edge).
                edge_score = float(self._rl_learner.get_edge_score(symbol))
                if edge_score == 0.0:
                    edge_mult = 1.0
                else:
                    edge_mult = max(0.60, min(1.0, 0.60 + 0.40 * edge_score))
                protect_mult = min(dd_mult, streak_mult, edge_mult)
                if protect_mult < 1.0:
                    risk_pct *= protect_mult
                    log.info("[%s] PROTECT x%.2f (dd=%.2f streak=%.2f edge=%.2f(%.2f) eq=$%.0f) → risk=%.3f%%",
                             symbol, protect_mult, dd_mult, streak_mult, edge_mult,
                             edge_score, cur_equity, risk_pct)
            except Exception as e:
                log.debug("[%s] PROTECT mult error: %s", symbol, e)

        # 3a-PORTFOLIO: removed 2026-06-04 (CTO audit B6 — portfolio_risk
        # multiplier was already applied INSIDE MasterBrain.evaluate_entry()
        # → counting it again here was double-discount, contributing to the
        # 0.08% actual vs 0.5% intended risk drift. MasterBrain owns portfolio.

        # 3b. DD reduction — removed 2026-06-04 (CTO audit B6 — overlapped
        # with dd_mult inside PROTECT block above; same drawdown counted twice).
        # PROTECT.dd_mult is now the single source of truth for DD-based
        # de-risking.

        # 3c. Clamp — global MAX_RISK ceiling + per-symbol SYMBOL_RISK_CAP
        # enforcement. 2026-06-04 CTO audit B6: was clamping only against
        # the global MAX so per-symbol caps got bypassed when MasterBrain
        # boosted up. Now enforce both.
        _sym_cap = float(SYMBOL_RISK_CAP.get(symbol, MAX_RISK_PER_TRADE_PCT))
        risk_pct = max(0.1, min(risk_pct, MAX_RISK_PER_TRADE_PCT, _sym_cap))

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
            try:
                from config import MIN_EDGE_FRICTION_PCT_PER_SYMBOL
            except Exception:
                MIN_EDGE_FRICTION_PCT_PER_SYMBOL = {}
            is_high_conv = raw_score >= MIN_EDGE_HIGH_CONV_SCORE
            # 2026-05-26 audit fix: per-symbol friction cap. Proven-edge indices
            # need wider caps than the 25% global (DJ30 friction 57% repeats in
            # logs but symbol has 360d edge). 1.5x relaxation for high-conv
            # signals on per-sym caps too.
            _per_sym_cap = MIN_EDGE_FRICTION_PCT_PER_SYMBOL.get(symbol)
            if _per_sym_cap is not None:
                friction_cap = float(_per_sym_cap) * (1.5 if is_high_conv else 1.0)
            else:
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
                # Per-symbol retrace override (2026-05-29 3yr tune); falls back
                # to the global default. Live honors per-symbol retrace only;
                # the wait window stays global (see config note).
                from config import PULLBACK_CONFIG_PER_SYMBOL as _PB_CFG
                _pb_retrace = _PB_CFG.get(symbol, {}).get("retrace", PULLBACK_ATR_RETRACE)
                retrace = atr_val * _pb_retrace
                target = signal_price - retrace if direction == "LONG" else signal_price + retrace
                self._pending_pullback[symbol] = {
                    "direction": direction, "score": raw_score, "atr": smart_atr,
                    "risk_pct": risk_pct, "signal_price": signal_price,
                    "entry_target": target, "bars_waited": 0,
                    # 2026-05-26 audit fix: track wall-clock UTC minute at arm
                    # time so we can compute REAL M1-bar elapsed (was incrementing
                    # per 0.5s brain cycle → expired in 1 cycle at default
                    # PULLBACK_MAX_WAIT_BARS=1).
                    "signal_minute": int(time.time() // 60),
                    "regime": regime, "m15_dir": m15_dir, "meta_prob": meta_prob,
                    "comp_long": comp_long, "comp_short": comp_short,
                    "signal_quality": signal_quality,
                }
                log.info("[%s] PULLBACK: %s quality=%.0f%% signal=%.5f target=%.5f",
                         symbol, direction, signal_quality, signal_price, target)
                return {**base_ret, "direction": direction, "gate": "PULLBACK_WAIT",
                        "m15_dir": m15_dir, "meta_prob": meta_prob,
                        "pullback_target": target}

        # ── 2026-06-21 D1_BIAS_UNIFIED soft-filter (downsize, NOT reject) ──
        # Unified D1 trend bias across momentum + SR + FVG. When direction
        # opposes a non-NEUTRAL D1 bias, downsize risk_pct by the configured
        # factor (default 0.5x). NEVER blocks — honors no-skip-trades rule.
        # Fail-open: any data hiccup → 'NEUTRAL' → no effect.
        try:
            from config import D1_BIAS_UNIFIED_ENABLED, D1_BIAS_UNIFIED_DOWNSIZE
            if D1_BIAS_UNIFIED_ENABLED:
                from agent.expert.d1_bias_unified import get_d1_bias
                _d1b = get_d1_bias(symbol, h1_df)
                if _d1b in ("LONG", "SHORT") and _d1b != direction:
                    _dn = float(D1_BIAS_UNIFIED_DOWNSIZE)
                    if 0.0 < _dn < 1.0:
                        risk_pct = max(0.1, risk_pct * _dn)
                        log.info("[%s] D1_BIAS_UNIFIED counter-bias (bias=%s dir=%s) "
                                 "→ x%.2f risk=%.3f%%",
                                 symbol, _d1b, direction, _dn, risk_pct)
        except Exception as e:
            log.debug("[%s] D1_BIAS_UNIFIED skipped: %s", symbol, e)

        # ── 2026-06-16 EXPERT_MODE — apply size_mult tilt + explicit SL/TP ──
        # When the orchestrator returned a multiplier (e.g. RANGE_DAY downsize
        # × conviction A+ uplift), tilt risk_pct in place. Capped by
        # SYMBOL_RISK_CAP downstream — never exceeds the per-symbol ceiling.
        # When the orchestrator returned absolute SL/TP1/TP2 prices (ASAT
        # or DynamicExitPlanner), route through open_trade_explicit so the
        # executor honours those structural levels instead of computing
        # ATR×mult + fixed-R targets.
        _expert_sl  = expert_verdict.get("sl")  if expert_verdict else None
        _expert_tp1 = expert_verdict.get("tp1") if expert_verdict else None
        _expert_tp2 = expert_verdict.get("tp2") if expert_verdict else None
        if expert_verdict and expert_size_mult and expert_size_mult != 1.0:
            try:
                _cap = SYMBOL_RISK_CAP.get(symbol, MAX_RISK_PER_TRADE_PCT)
                risk_pct = float(min(_cap, risk_pct * expert_size_mult))
            except Exception:
                pass

        # Direct entry (fallback if pullback disabled or already pending)
        # 2026-05-12: pass raw_score so executor can scale TP per conviction.
        # 2026-05-17: pass regime so executor can use per-(sym, regime) SL.
        if (_expert_sl is not None and _expert_tp1 is not None
                and _expert_tp2 is not None):
            try:
                _tick2 = self.state.get_tick(symbol)
                _entry_px = (float(_tick2.ask) if direction == "LONG"
                             else float(_tick2.bid)) if _tick2 else 0.0
                success = self.executor.open_trade_explicit(
                    symbol, direction,
                    _entry_px, float(_expert_sl),
                    float(_expert_tp1), float(_expert_tp2),
                    risk_pct=risk_pct,
                    magic_offsets=[3000, 3001],
                    strategy_name="EXPERT",
                )
            except Exception as e:
                log.warning("[%s] EXPERT explicit-entry failed (%s) — "
                            "falling back to legacy open_trade", symbol, e)
                success = self.executor.open_trade(
                    symbol, direction, smart_atr,
                    risk_pct=risk_pct, score=raw_score, regime=regime)
        else:
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
            self._entry_metadata[symbol] = {
                "score": float(raw_score), "regime": str(regime),
                "direction": str(direction), "entry_price": float(entry_price),
                "risk_pct": float(risk_pct),
                "m15_dir": str(m15_dir) if m15_dir else "FLAT",
                "meta_prob": float(meta_prob) if meta_prob is not None else 0.0,
                "score_components": entry_components, "ts": time.time(),
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

            # 2026-06-22: per-(strategy, symbol) N-consec-loss tracking for
            # the momentum strategy. Halts (momentum, sym) for the rest of the
            # UTC day after N losses (config.PER_STRATEGY_SYMBOL_KILL_LOSSES).
            try:
                self._master_brain.record_strategy_symbol_close(
                    "momentum", symbol, won=(pnl > 0))
            except Exception as _e:
                log.debug("strat-sym close record err: %s", _e)

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
    # 2026-06-05: DAILY-LOSS HELPER (3% kill switch — prop-firm standard)
    # ═══════════════════════════════════════════════════════════════

    def _compute_daily_loss_pct(self) -> float:
        """
        Return today's loss as a positive percentage of start-of-day equity.
        Returns 0.0 if today is breakeven/profit OR if computation fails
        (fail-safe — never falsely trip the kill switch on a DB hiccup).

        Source-of-truth order:
          1. daily_stats table for today's UTC date (journal-truthful PnL +
             persisted start-of-day equity)
          2. Fallback to in-memory self._daily_loss (already maintained by
             _run_cycle) if DB row not yet written
        """
        import sqlite3
        try:
            today_iso = datetime.now(timezone.utc).date().isoformat()
            conn = sqlite3.connect(str(self._entry_metadata_db()), timeout=2.0)
            try:
                row = conn.execute(
                    "SELECT pnl, equity FROM daily_stats WHERE date = ?",
                    (today_iso,)
                ).fetchone()
            finally:
                conn.close()
            if row is not None:
                pnl = float(row[0] or 0.0)
                row_equity = float(row[1] or 0.0)
                # Prefer the brain's tracked start-of-day equity if present
                # (daily_stats.equity may be CURRENT equity, not start).
                base = float(self._daily_start_equity) if self._daily_start_equity > 0 else row_equity
                if pnl < 0 and base > 0:
                    return abs(pnl) / base * 100.0
                return 0.0
        except Exception as e:
            log.debug("daily_stats lookup failed (%s) — using in-memory daily_loss", e)
        # Fallback: in-memory tracker already computed by _run_cycle
        try:
            return float(self._daily_loss) if self._daily_loss > 0 else 0.0
        except Exception:
            return 0.0

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

        # 2026-06-08 workflow research: ML META gate ANTI-EDGE on XAU + JPN225.
        # Stream D backtest 60d: JPN225 ML-ON PF 4.88 → ML-OFF PF 14.19 (+191%),
        # XAU ML-ON PF 2.10 → ML-OFF PF 2.47 (+18%). Trade count more than
        # doubled on JPN225 (36→77), WR rose (78→88%), DD dropped (3.8→2.6%).
        # The model has good AUC (0.704/0.671) but its threshold mapping
        # specifically vetoes the highest-quality signals on these two syms.
        try:
            from config import ML_BYPASS_SYMBOLS
            if symbol in ML_BYPASS_SYMBOLS:
                return True
        except ImportError:
            pass

        prob = float(meta_prob)

        # Dynamic threshold based on signal quality.
        # 2026-06-08: high-tier boundary 60→58. Live observation: XAU SHORT
        # quality 59.2% repeatedly blocked at meta=0.40 vs 0.43 threshold —
        # mathematically 'normal' but conviction-wise indistinguishable from
        # the 60% tier. Boundary shift only affects the 58-60 band.
        if score >= 75:
            threshold = 0.25  # very high conviction — only block on hard rejection
        elif score >= 58:
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

    def _xau_vol_regime_mult(self, symbol, ind, bi):
        """GVZ-VRP proxy via ATR-14 percentile (BIS WP 619, Prokopczuk et al.
        J. Banking & Finance 2017). Quiet (low ATR pct) = trend regime, peer-
        reviewed positive forward returns on precious metals. Panicky (high
        pct) = mean-reversion regime.

        Symmetric LONG+SHORT per literature. Downsize joins the min() chain
        unconditionally (safe). Upsize gated by self._xau_vrp_boost (default
        False) to honor production-grade "no multiplier stacks" doctrine
        until journaled atr_pct90 validates the boost side.
        """
        if symbol != "XAUUSD":
            return 1.0
        try:
            atr_series = ind.get("at")
            if atr_series is None or bi < 90:
                return 1.0
            window = atr_series[bi - 89:bi + 1]
            window = window[~np.isnan(window)]
            if len(window) < 60:
                return 1.0
            cur = float(atr_series[bi])
            if not np.isfinite(cur) or cur <= 0:
                return 1.0
            pct = float((window < cur).sum()) / len(window)
            if pct <= 0.20:
                # quiet — VRP-positive trend regime → upsize (gated until validated)
                return 1.5 if getattr(self, "_xau_vrp_boost", False) else 1.0
            if pct >= 0.80:
                return 0.5  # panicky — mean-rev → downsize (always honored)
            return 1.0
        except Exception:
            return 1.0

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
        # 2026-06-04 CTO audit B9: low_vol was 43% of trades and 157% of net
        # losses. Root causes: (1) NaN→low_vol contaminated the bucket;
        # (2) except→low_vol contaminated the bucket again; (3) the
        # bbw<3.0 + ADX 22-25 knife-edge mis-classified most chop as
        # low_vol (forex/indices live in that ADX window most of the day).
        # Fixes: NaN/exception → "unknown" (caller skips); raise ADX
        # floor 22→25; combined rule "BBW>=0.8 AND ADX>=28 = trending"
        # removes the ADX cliff; mid-BBW (1.5-3.0) → ranging instead of
        # low_vol. Estimated journal swing -$64 → +$65 (+$130).
        try:
            bbw_val = float(ind["bbw"][bi])
            adx_val_raw = ind["adx"][bi]
            if np.isnan(bbw_val) or np.isnan(adx_val_raw):
                return "unknown"
            adx_val = float(adx_val_raw)

            # ADX-first ranging detection — raised floor 22 → 25
            if adx_val < 25:
                return "ranging"

            # Combined-rule trending: requires both volatility expansion
            # AND directional strength. Removes the ADX 22-vs-26 cliff.
            if bbw_val >= 0.8 and adx_val >= 28:
                if bbw_val >= 5.0:
                    return "volatile"
                return "trending"

            # Tight + weak ADX = genuine low_vol (smaller bucket now)
            if bbw_val < 1.5:
                return "low_vol"

            # Mid-BBW with ADX 25-28 = chop (was mis-classed trending/low_vol)
            if bbw_val < 3.0:
                return "ranging"

            if bbw_val < 5.0:
                return "trending" if adx_val > 30 else "volatile"

            return "volatile"
        except Exception:
            return "unknown"

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
        """Structured decision log line.

        2026-06-18 Tier 1 #2: every legacy `gate` string is canonicalized
        via agent.decision_reasons.canonicalize_gate() so the dashboard
        rejection Pareto can roll free-form variants up into stable codes.
        The original `gate` is preserved as `gate_legacy` in the dashboard
        payload for backward-compat. Fail-open: any canonicalizer exception
        falls back to the legacy string.
        """
        meta_str = ("%.2f" % meta_prob) if meta_prob is not None else "N/A"
        m15_str = str(m15_dir) if m15_dir else "N/A"
        # Canonicalize early so log line + dashboard payload stay in sync.
        try:
            from agent.decision_reasons import canonicalize_gate as _canon
            canonical = _canon(str(gate))
        except Exception:
            canonical = str(gate)
        log.info(
            "DECISION: %s | L=%.1f S=%.1f | DIR=%s | M15=%s | META=%s | GATE=%s | CODE=%s | %s",
            symbol, float(long_score), float(short_score),
            direction, m15_str, meta_str, gate, canonical, action_str
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
                "gate": str(canonical),          # canonical code primary
                "gate_legacy": str(gate),         # preserve legacy for transition
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
