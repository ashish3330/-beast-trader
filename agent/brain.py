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
                 level_memory=None, fvg_detector=None):
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
        self._daily_start_equity = float(0.0)
        self._daily_loss = float(0.0)
        self._last_day = None
        self._trade_log = []

        # ── HARD KILL SWITCH STATE ──
        self._weekly_start_equity = float(0.0)
        self._weekly_start_day = None          # Monday date for weekly reset
        self._kill_switch_active = False        # True = no new trades allowed
        self._kill_switch_reason = ""           # "daily" or "weekly"
        self._kill_switch_until = None          # datetime when kill switch resets

        # ── Entry metadata cache: symbol → {score, regime, direction, entry_price, ts} ──
        # Used by learning engine deal sync to attach brain metadata to SL/TP exits
        self._entry_metadata: Dict[str, dict] = {}

        # ── Candle-close tracking: only re-score when new candle appears ──
        self._last_candle_time: Dict[str, float] = {}
        self._last_scores: Dict[str, dict] = {}  # cached scores for dashboard between candles

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

        # ── SL-hit cooldown: block re-entry for 45 min after stop loss ──
        self._sl_cooldown: Dict[str, float] = {}  # symbol -> unix timestamp when cooldown expires

        # ── Indicator cache (recompute every cycle per symbol) ──
        self._ind_cache = {}       # symbol -> (indicators_dict, timestamp)
        self._ind_cache_ttl = 0.25  # 250ms cache — near-instant scoring

        # ── Pullback entry: deferred signals waiting for retrace ──
        self._pending_pullback: Dict[str, dict] = {}  # symbol -> {direction, score, atr, risk_pct, signal_price, bars_waited, ...}

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
        self._daily_start_equity = float(equity)
        self._last_day = datetime.now(timezone.utc).date()

        # Weekly equity tracking — reset every Monday 00:00 UTC
        self._weekly_start_equity = float(equity)
        today = datetime.now(timezone.utc).date()
        # Find the Monday of current week
        self._weekly_start_day = today - __import__('datetime').timedelta(days=today.weekday())
        self._kill_switch_active = False
        self._kill_switch_reason = ""
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
        """Main loop: every ~1s, evaluate all symbols."""
        while self.running:
            loop_start = time.time()
            self._cycle += 1

            try:
                self._run_cycle()
            except Exception as e:
                log.error("Dragon brain cycle %d error: %s", self._cycle, e, exc_info=True)

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, CYCLE_INTERVAL_S - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _run_cycle(self):
        """Single cycle: kill switch → guardian → daily reset → DD check → process symbols → exits → manage → dashboard."""
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
        if not self._kill_switch_active and daily_loss_pct >= DAILY_HARD_STOP_PCT:
            log.critical(
                "DAILY KILL SWITCH TRIGGERED: loss %.2f%% >= %.1f%% "
                "(equity $%.2f, day start $%.2f) — CLOSING ALL, NO NEW TRADES UNTIL TOMORROW",
                daily_loss_pct, DAILY_HARD_STOP_PCT, equity, self._daily_start_equity)
            self._kill_switch_active = True
            self._kill_switch_reason = "daily"
            self._kill_switch_until = None  # resets at next day boundary
            self.executor.close_all("DailyKillSwitch")

        # Check weekly hard stop
        if not self._kill_switch_active and weekly_loss_pct >= WEEKLY_HARD_STOP_PCT:
            log.critical(
                "WEEKLY KILL SWITCH TRIGGERED: loss %.2f%% >= %.1f%% "
                "(equity $%.2f, week start $%.2f) — CLOSING ALL, NO NEW TRADES UNTIL NEXT MONDAY",
                weekly_loss_pct, WEEKLY_HARD_STOP_PCT, equity, self._weekly_start_equity)
            self._kill_switch_active = True
            self._kill_switch_reason = "weekly"
            self._kill_switch_until = None  # resets at next Monday boundary
            self.executor.close_all("WeeklyKillSwitch")

        # If kill switch is active: manage existing positions ONLY, skip all new trade logic
        if self._kill_switch_active:
            # Log every 60 cycles (~30 seconds) so we know it's alive
            if self._cycle % 60 == 0:
                log.warning("KILL SWITCH ACTIVE (%s): daily_loss=%.2f%% weekly_loss=%.2f%% — "
                            "managing trailing SL only, no new trades",
                            self._kill_switch_reason, daily_loss_pct, weekly_loss_pct)

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

        # ═══ PROCESS EACH SYMBOL ═══
        scores_for_dashboard = {}
        for symbol in SYMBOLS:
            try:
                result = self._process_symbol(symbol, equity, dd_pct, daily_loss_pct)
                if result:
                    self._last_scores[symbol] = result  # cache for dashboard
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

        # ═══ INTELLIGENT EXITS (Dragon ExitIntelligence) ═══
        # ExitIntelligence.evaluate_exits() takes no args — it uses self.executor/state
        # from __init__, and closes positions directly via executor.close_position()
        if self._exit_intelligence:
            try:
                self._exit_intelligence.evaluate_exits()
            except Exception as e:
                log.warning("ExitIntelligence error: %s", e)

        # ═══ PUSH RL TRAIL ADJUSTMENTS TO EXECUTOR ═══
        if self._rl_learner:
            for sym in SYMBOLS:
                try:
                    adj = self._rl_learner.get_trail_adjustments(sym)
                    self.executor.set_rl_trail_adjustments(sym, adj)
                except Exception:
                    pass

        # ═══ MANAGE TRAILING SL + MTF EXIT + M15 REVERSAL EXIT ═══
        for symbol in SYMBOLS:
            try:
                if self.executor.has_position(symbol):
                    self.executor.manage_trailing_sl(symbol)

                    # MTF exit urgency check (skip JPN225ft — triggers too early, PF 1.87→1.13)
                    cfg = SYMBOLS.get(symbol)
                    skip_mtf_exit = cfg and cfg.symbol in ("JPN225ft",)
                    if self._mtf and not skip_mtf_exit:
                        try:
                            mtf = self._mtf.analyze(symbol)
                            urgency = mtf.get("exit_urgency", 0)
                            if urgency >= 0.7:
                                pnl = self._get_position_pnl(symbol)
                                if pnl > 0:  # only exit on urgency if in profit
                                    log.info("[%s] MTF EXIT: urgency=%.2f pnl=%.2f — closing",
                                             symbol, urgency, pnl)
                                    mtf_exit_dir = self.executor.get_position_direction(symbol)
                                    closed = self.executor.close_position(symbol, "DragonMTFExit")
                                    if closed:
                                        self._record_trade_result(symbol, reason="mtf_exit")
                                        self._log_trade(symbol, mtf_exit_dir,
                                                        0, "MTF_EXIT", pnl=pnl)
                                    continue
                        except Exception:
                            pass

                    self._check_m15_reversal_exit(symbol)
            except Exception as e:
                log.warning("[%s] Trailing/exit error: %s", symbol, e)

        # ═══ UPDATE DASHBOARD STATE ═══
        self.state.update_agent("cycle", int(self._cycle))
        # Don't overwrite balance — tick streamer sets it from MT5 account_info
        self.state.update_agent("profit", float(equity - self._daily_start_equity))
        self.state.update_agent("dd_pct", float(dd_pct))
        self.state.update_agent("peak_equity", float(max(equity, self.state.get_agent_state().get("peak_equity", equity))))
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
                    success = self.executor.open_trade(symbol, d, sa, risk_pct=rp)
                    if success:
                        self._log_trade(symbol, d, rs, "ENTRY_PULLBACK")
                        ep = self.executor._entry_prices.get(symbol, 0)
                        self._entry_metadata[symbol] = {
                            "score": float(rs), "regime": pb.get("regime", ""),
                            "direction": d, "entry_price": float(ep),
                            "risk_pct": float(rp), "m15_dir": pb.get("m15_dir", "FLAT"),
                            "meta_prob": pb.get("meta_prob", 0.0),
                            "score_components": comp_l if d == "LONG" else comp_s,
                            "ts": time.time(),
                        }
                        self.state.update_agent("entry_metadata", dict(self._entry_metadata))
                    return _ret(0, 0, pb.get("signal_quality", 0), 0, d,
                                "PULLBACK_ENTERED" if success else "PULLBACK_FAILED")
                if pb["bars_waited"] >= PULLBACK_MAX_WAIT_BARS:
                    log.info("[%s] PULLBACK EXPIRED after %d bars", symbol, pb["bars_waited"])
                    self._pending_pullback.pop(symbol)
                    return _ret(0, 0, 0, 0, "FLAT", "PULLBACK_EXPIRED")
            return _ret(0, 0, pb.get("signal_quality", 0), 0,
                        pb["direction"], "PULLBACK_WAIT",
                        pullback_target=pb["entry_target"], bars_waited=pb["bars_waited"])

        # ══════════════════════════════════════════
        #  PRE-CHECK: SL cooldown (45 min)
        # ══════════════════════════════════════════
        sl_expiry = self._sl_cooldown.get(symbol, 0)
        if time.time() < sl_expiry and not self.executor.has_position(symbol):
            self._pending_pullback.pop(symbol, None)
            mins_left = (sl_expiry - time.time()) / 60
            return _ret(0, 0, 0, 0, "FLAT", "SL_COOLDOWN", cooldown_mins=round(mins_left, 1))

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

        long_score, short_score, comp_long, comp_short = _score_with_components(ind, bi)
        long_score = float(long_score)
        short_score = float(short_score)
        atr_val = float(ind["at"][bi])

        # 1d. Normalize to 0-100 scale
        raw_score = max(long_score, short_score)
        signal_quality = min(100.0, raw_score / SIGNAL_QUALITY_DIVISOR * 100)

        # 1e. Regime + threshold
        regime = self._get_regime_from_bbw(ind, bi)
        # Per-symbol quality override, fallback to default
        from config import SIGNAL_QUALITY_SYMBOL
        sym_q = SIGNAL_QUALITY_SYMBOL.get(symbol)
        if sym_q:
            min_quality = float(sym_q.get(regime, SIGNAL_QUALITY_THRESHOLDS.get(regime, 50)))
        else:
            min_quality = float(SIGNAL_QUALITY_THRESHOLDS.get(regime, 50))

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

        # ══════════════════════════════════════════════
        #  PHASE 2: GATES (7 binary checks)
        # ══════════════════════════════════════════════

        base_ret = dict(long_score=long_score, short_score=short_score,
                        signal_quality=signal_quality, min_quality=min_quality,
                        atr=atr_val, regime=regime)

        # Gate 1: Session hours (non-crypto)
        if cfg.category != "Crypto":
            sess_start, sess_end = SYMBOL_SESSION_OVERRIDE.get(
                symbol, (SESSION_START_UTC, SESSION_END_UTC))
            if hour_utc < sess_start or hour_utc >= sess_end:
                return {**base_ret, "direction": direction, "gate": "SESSION"}

        # Gate 2: Direction bias
        allowed_dir = DIRECTION_BIAS.get(symbol)
        if allowed_dir and direction != allowed_dir:
            self._log_decision(symbol, long_score, short_score,
                               direction, "DIR_BIAS", None, None,
                               "SKIP (%s only %s)" % (symbol, allowed_dir))
            return {**base_ret, "direction": direction, "gate": "DIR_BIAS"}

        # Gate 3: Toxic hours
        exempt = TOXIC_HOUR_EXEMPT.get(symbol, set())
        if hour_utc in TOXIC_HOURS_UTC and hour_utc not in exempt:
            self._log_decision(symbol, long_score, short_score,
                               direction, "TOXIC_HOUR", None, None,
                               "SKIP (H%02d toxic)" % hour_utc)
            return {**base_ret, "direction": direction, "gate": "TOXIC_HOUR"}

        # Gate 4: Position management (hold / reversal)
        current_dir = self.executor.get_position_direction(symbol)
        has_pos = current_dir != "FLAT"

        if has_pos and current_dir == direction:
            return {**base_ret, "direction": direction, "gate": "HOLD_SWING"}

        if has_pos and current_dir != direction:
            reversal_min = min_quality + 12  # need +12% quality for reversal
            if signal_quality < reversal_min:
                return {**base_ret, "direction": direction, "gate": "REVERSAL_WEAK"}
            m15_dir = self._get_m15_direction(symbol)
            if m15_dir != direction:
                return {**base_ret, "direction": direction, "gate": "REVERSAL_M15",
                        "m15_dir": m15_dir}
            # Execute reversal
            meta_prob = self._meta_label_check(symbol, direction, ind, bi)
            risk_pct = SYMBOL_RISK_CAP.get(symbol, MAX_RISK_PER_TRADE_PCT)
            if self._master_brain:
                try:
                    rev_eval = self._master_brain.evaluate_entry(
                        symbol=symbol, direction=direction, score=raw_score,
                        regime=regime, meta_prob=meta_prob, m15_dir=m15_dir)
                    risk_pct = float(rev_eval.get("risk_pct", risk_pct))
                except Exception:
                    pass
            exit_pnl = self._get_position_pnl(symbol)
            self._record_trade_result(symbol, reason="reversal")
            self._log_trade(symbol, current_dir, raw_score, "REVERSAL", pnl=exit_pnl)
            self.executor.reverse_position(symbol, direction, atr_val, risk_pct=risk_pct)
            log.info("[%s] REVERSAL %s→%s quality=%.0f%% risk=%.2f%%",
                     symbol, current_dir, direction, signal_quality, risk_pct)
            return {**base_ret, "direction": direction, "gate": "REVERSAL",
                    "m15_dir": m15_dir, "meta_prob": meta_prob}

        # Gate 5: MTF confirmation (M15 agrees OR high conviction override)
        m15_dir = self._get_m15_direction(symbol)
        m15_agrees = (m15_dir == direction)
        m15_flat = (m15_dir == "FLAT")
        is_trend = cfg.category in ("Crypto", "Index")
        m15_pass = (m15_agrees or
                    signal_quality >= MTF_OVERRIDE_QUALITY or
                    (m15_flat and (signal_quality >= 65 or is_trend)))
        if not m15_pass:
            self._log_decision(symbol, long_score, short_score,
                               direction, "M15_DISAGREE", m15_dir, None,
                               "SKIP (M15=%s != %s)" % (m15_dir, direction))
            return {**base_ret, "direction": direction, "gate": "M15_DISAGREE",
                    "m15_dir": m15_dir}

        # Gate 6: Meta-label ML filter
        meta_prob = self._meta_label_check(symbol, direction, ind, bi)
        meta_pass = self._meta_passes(symbol, meta_prob, score=raw_score)
        if not meta_pass:
            self._log_decision(symbol, long_score, short_score,
                               direction, "META_REJECT", m15_dir, meta_prob,
                               "SKIP (meta=%.2f)" % (meta_prob or 0))
            return {**base_ret, "direction": direction, "gate": "META_REJECT",
                    "m15_dir": m15_dir, "meta_prob": meta_prob}

        # Gate 7: MasterBrain
        risk_pct = SYMBOL_RISK_CAP.get(symbol, MAX_RISK_PER_TRADE_PCT)
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

        # 3a. Conviction sizing (0-100 scale)
        if signal_quality >= 80:
            conv_mult = CONVICTION_SIZING_V2.get("80+", 1.5)
        elif signal_quality >= 65:
            conv_mult = CONVICTION_SIZING_V2.get("65-80", 1.2)
        elif signal_quality >= 55:
            conv_mult = CONVICTION_SIZING_V2.get("55-65", 1.0)
        else:
            conv_mult = CONVICTION_SIZING_V2.get("<55", 0.6)
        risk_pct *= conv_mult

        # 3b. Session/DOW applied to RISK (not score)
        dow = datetime.now(timezone.utc).weekday()
        sess_mult = self._get_session_multiplier(symbol, hour_utc)
        dow_mult = {0: 0.92, 1: 1.05, 2: 1.05, 3: 1.03, 4: 0.90}.get(dow, 1.0)
        risk_pct *= sess_mult * dow_mult

        # 3c. DD reduction
        if dd_pct >= DD_REDUCE_THRESHOLD:
            risk_pct *= 0.5
            log.info("[%s] DD %.1f%% — risk halved to %.3f%%", symbol, dd_pct, risk_pct)

        # 3d. Clamp
        risk_pct = max(0.1, min(risk_pct, MAX_RISK_PER_TRADE_PCT * 1.5))

        if conv_mult != 1.0:
            log.info("[%s] CONVICTION: quality=%.0f%% → x%.1f → risk=%.3f%%",
                     symbol, signal_quality, conv_mult, risk_pct)

        # ══════════════════════════════════════════════
        #  PHASE 4: EXECUTE
        # ══════════════════════════════════════════════

        smart_atr = float(atr_val)

        # Pullback entry: defer and wait for retrace
        if PULLBACK_ENTRY_ENABLED and symbol not in self._pending_pullback:
            tick = self.state.get_tick(symbol)
            signal_price = float(tick.bid) if tick and hasattr(tick, 'bid') else 0
            if signal_price > 0:
                retrace = atr_val * PULLBACK_ATR_RETRACE
                target = signal_price - retrace if direction == "LONG" else signal_price + retrace
                self._pending_pullback[symbol] = {
                    "direction": direction, "score": raw_score, "atr": smart_atr,
                    "risk_pct": risk_pct, "signal_price": signal_price,
                    "entry_target": target, "bars_waited": 0,
                    "regime": regime, "m15_dir": m15_dir, "meta_prob": meta_prob,
                    "comp_long": comp_long, "comp_short": comp_short,
                    "signal_quality": signal_quality,
                }
                log.info("[%s] PULLBACK: %s quality=%.0f%% signal=%.5f target=%.5f",
                         symbol, direction, signal_quality, signal_price, target)
                return {**base_ret, "direction": direction, "gate": "PULLBACK_WAIT",
                        "m15_dir": m15_dir, "meta_prob": meta_prob,
                        "pullback_target": target}

        # Direct entry (fallback if pullback disabled or already pending)
        success = self.executor.open_trade(symbol, direction, smart_atr, risk_pct=risk_pct)

        if success:
            self._log_trade(symbol, direction, raw_score, "ENTRY")
            entry_price = self.executor._entry_prices.get(symbol, 0)
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

            # R-multiple = actual PnL / intended dollar risk per trade
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

            # SL-hit cooldown: block re-entry for 45 min (3 M15 candles)
            # Prevents churn where same signal re-enters and hits SL repeatedly
            is_sl_exit = "sl" in reason.lower() or pnl < 0
            if is_sl_exit:
                import time as _time
                cooldown_secs = 2700  # 45 minutes
                self._sl_cooldown[symbol] = _time.time() + cooldown_secs
                # Publish to SharedState so scalp brain also respects cooldown
                self.state.update_agent("sl_cooldowns", dict(self._sl_cooldown))
                log.info("[%s] SL COOLDOWN: blocked for 45min after loss (pnl=%.2f, reason=%s)",
                         symbol, pnl, reason)

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
                    self._rl_learner.record_outcome(
                        symbol=symbol, direction=last_direction, pnl=pnl,
                        r_multiple=r_mult, score=last_score, regime=regime_now,
                        exit_reason=reason, score_components=cached_components,
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
        Smart ML filter. Scales threshold by score strength:
        - Score >= 8.0 (very high conviction): only block if ML < 0.30
        - Score 7.0-8.0 (high): block if ML < 0.40
        - Score 6.0-7.0 (normal): block if ML < 0.50 (standard)
        - Score < 6.0: block if ML < 0.50

        Also tracks ML block rate — if ML blocks > 80% of signals for a symbol
        over 50+ evaluations, auto-bypasses ML (model is broken for this symbol).
        """
        if meta_prob is None:
            return True  # No model = pure scoring, always pass

        prob = float(meta_prob)

        # Dynamic threshold based on score strength
        if score >= 8.0:
            threshold = 0.30  # very high score = only block on clear ML rejection
        elif score >= 7.0:
            threshold = 0.40
        else:
            threshold = META_PROB_THRESHOLD  # 0.50

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
        """
        try:
            bbw_val = float(ind["bbw"][bi])
            if np.isnan(bbw_val):
                return "low_vol"
            adx_val = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 25.0

            # BBW thresholds (percentage-based):
            # < 1.5% = very tight = ranging
            # 1.5-3.0% = normal = check ADX for trending
            # 3.0-5.0% = widening = volatile
            # > 5.0% = very wide = volatile
            if bbw_val < 1.5:
                if adx_val < 20:
                    return "ranging"
                else:
                    return "low_vol"
            elif bbw_val < 3.0:
                if adx_val > 25:
                    return "trending"
                else:
                    return "low_vol"
            elif bbw_val < 5.0:
                if adx_val > 30:
                    return "trending"
                else:
                    return "volatile"
            else:
                return "volatile"
        except Exception:
            return "low_vol"

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

        return base

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
