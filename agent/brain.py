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
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, MIN_SCORE,
    REGIME_PARAMS,
)
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

log = logging.getLogger("dragon.brain")

# ═══ CONSTANTS ═══
CYCLE_INTERVAL_S = 0.5           # 500ms decision cycle — real-time scoring
META_PROB_THRESHOLD = 0.50       # meta-label skip threshold (Dragon: was 0.40)
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
                 level_memory=None):
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

        # ── Indicator cache (recompute every cycle per symbol) ──
        self._ind_cache = {}       # symbol -> (indicators_dict, timestamp)
        self._ind_cache_ttl = 1.0  # 1s cache — real-time scoring

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
        """Single cycle: guardian → daily reset → DD check → process symbols → exits → manage → dashboard."""
        # ── EQUITY GUARDIAN: real-time P&L monitoring (runs FIRST every cycle) ──
        if self._guardian:
            try:
                self._guardian.monitor()
            except Exception as e:
                log.warning("Guardian error: %s", e)

        # ── Daily reset at midnight UTC ──
        today = datetime.now(timezone.utc).date()
        if today != self._last_day:
            self._last_day = today
            equity = float(self.state.get_agent_state().get("equity", STARTING_BALANCE))
            self._daily_start_equity = float(equity)
            self._daily_loss = float(0.0)
            log.info("New trading day — daily start equity: $%.2f", equity)

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
                    scores_for_dashboard[symbol] = result
            except Exception as e:
                log.error("[%s] Process error: %s", symbol, e)

        # ═══ MARKET OBSERVATION (learning engine watches patterns) ═══
        if self._learning_engine:
            try:
                self._learning_engine.observe_market()
            except Exception:
                pass

        # ═══ ENRICH SCORES (lightweight — no heavy computation) ═══
        for symbol, r in scores_for_dashboard.items():
            r.setdefault("regime", "unknown")
            r.setdefault("gates", {"tf": "pass", "ofi": "pass", "vol": "pass", "reg": "pass"})
            r.setdefault("vol_score", 0)
            r.setdefault("m15_dir", "flat")

        # ═══ INTELLIGENT EXITS (Dragon ExitIntelligence) ═══
        # ExitIntelligence.evaluate_exits() takes no args — it uses self.executor/state
        # from __init__, and closes positions directly via executor.close_position()
        if self._exit_intelligence:
            try:
                self._exit_intelligence.evaluate_exits()
            except Exception as e:
                log.warning("ExitIntelligence error: %s", e)

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
                                    closed = self.executor.close_position(symbol, "DragonMTFExit")
                                    if closed:
                                        self._record_trade_result(symbol)
                                        self._log_trade(symbol, self.executor.get_position_direction(symbol),
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
        Process one symbol through the full decision pipeline.
        Returns score info dict for dashboard, or None.
        """
        cfg = SYMBOLS[symbol]

        # ─── 1. SESSION FILTER (non-crypto: per-symbol or default 06-22 UTC) ───
        if cfg.category != "Crypto":
            hour_utc = int(datetime.now(timezone.utc).hour)
            sess_start, sess_end = SYMBOL_SESSION_OVERRIDE.get(
                symbol, (SESSION_START_UTC, SESSION_END_UTC))
            if hour_utc < sess_start or hour_utc >= sess_end:
                return {"long_score": 0.0, "short_score": 0.0,
                        "direction": "FLAT", "gate": "SESSION"}

        # ─── 2. GET H1 CANDLES ───
        h1_df = self.state.get_candles(symbol, 60)
        if h1_df is None or len(h1_df) < H1_MIN_BARS:
            return {"long_score": 0.0, "short_score": 0.0,
                    "direction": "FLAT", "gate": "NO_H1_DATA"}

        # ─── 3. COMPUTE SCORES using momentum_scorer._score() ───
        ind = self._get_indicators(symbol, h1_df)
        if ind is None:
            return None

        n = int(ind["n"])
        bi = n - 2  # completed bar (not the forming bar)
        if bi < 21 or np.isnan(ind["at"][bi]) or float(ind["at"][bi]) == 0.0:
            return {"long_score": 0.0, "short_score": 0.0,
                    "direction": "FLAT", "gate": "INSUFFICIENT_IND"}

        long_score, short_score = _score(ind, bi)
        long_score = float(long_score)
        short_score = float(short_score)
        atr_val = float(ind["at"][bi])

        # ─── 3b. SESSION + DAY-OF-WEEK ALPHA MULTIPLIERS ───
        hour_utc = int(datetime.now(timezone.utc).hour)
        dow = datetime.now(timezone.utc).weekday()
        sess_mult = self._get_session_multiplier(symbol, hour_utc)
        dow_mult = {0: 0.92, 1: 1.05, 2: 1.05, 3: 1.03, 4: 0.90}.get(dow, 1.0)
        alpha_mult = sess_mult * dow_mult
        long_score *= alpha_mult
        short_score *= alpha_mult

        # ─── 3b2. RL SCORE WEIGHT ADJUSTMENT ───
        if self._rl_learner:
            try:
                weights = self._rl_learner.get_score_weights(symbol)
                # Apply weights to component scores (multiply each component)
                # For now, scale the total score by average weight adjustment
                avg_weight = sum(weights.values()) / len(weights) if weights else 1.0
                long_score *= avg_weight
                short_score *= avg_weight
            except Exception as e:
                log.debug("[%s] RL weight adjustment failed: %s", symbol, e)

        # ─── 3c. REGIME DETECTION + ADAPTIVE MIN_SCORE ───
        regime = self._get_regime_from_bbw(ind, bi)
        adaptive_min = self._get_adaptive_min_score(regime, symbol=symbol)

        # ─── 4. DETERMINE DIRECTION (using regime-adaptive threshold) ───
        if long_score >= adaptive_min and long_score >= short_score:
            direction = "LONG"
            raw_score = long_score
        elif short_score >= adaptive_min and short_score > long_score:
            direction = "SHORT"
            raw_score = short_score
        else:
            direction = "FLAT"
            raw_score = float(max(long_score, short_score))
            # No signal — still return scores for dashboard
            self._log_decision(symbol, long_score, short_score,
                               "FLAT", "BELOW_MIN", None, None,
                               "NO ENTRY (score below %.1f, regime=%s)" % (adaptive_min, regime))
            return {"long_score": long_score, "short_score": short_score,
                    "direction": "FLAT", "gate": "BELOW_MIN_SCORE",
                    "atr": atr_val, "regime": regime,
                    "adaptive_min_score": adaptive_min}

        # ─── NEWS FILTER (avoid high-impact events) ───
        if self._calendar:
            skip, reason = self._calendar.should_skip_entry(symbol)
            if skip:
                self._log_decision(symbol, long_score, short_score, direction, "NEWS_SKIP", None, None, "SKIP (%s)" % reason)
                return {"long_score": long_score, "short_score": short_score,
                        "direction": direction, "gate": "NEWS_SKIP",
                        "atr": atr_val, "regime": regime,
                        "news_reason": reason}

        # ─── 5. GATE CHECKS ───

        # Gate A: M15 alignment — tiered by score strength
        # High conviction (score >= 8.0): FLAT M15 allowed (catches pullback entries)
        # Normal conviction: M15 must AGREE with H1 (no FLAT, no opposing)
        # Reversals: always require M15 agreement (checked separately below)
        m15_dir = self._get_m15_direction(symbol)
        m15_opposing = (
            (direction == "LONG" and m15_dir == "SHORT") or
            (direction == "SHORT" and m15_dir == "LONG")
        )
        if raw_score >= 8.0:
            m15_aligned = not m15_opposing  # FLAT allowed for high-conviction
        else:
            m15_aligned = (m15_dir == direction)  # must agree for normal entries

        # Gate B: Position management
        current_dir = self.executor.get_position_direction(symbol)
        has_pos = current_dir != "FLAT"

        if has_pos and current_dir == direction:
            # Same direction — hold (swing mode), no new entry
            self._log_decision(symbol, long_score, short_score,
                               direction, "HOLD", m15_dir, None,
                               "HOLD %s (swing mode)" % direction)
            return {"long_score": long_score, "short_score": short_score,
                    "direction": direction, "gate": "HOLD_SWING",
                    "atr": atr_val, "regime": regime, "m15_dir": m15_dir}

        if has_pos and current_dir != direction:
            # Reversal requires HIGHER conviction (2x spread cost, whipsaw risk)
            reversal_min = adaptive_min + 1.5
            if raw_score < reversal_min:
                self._log_decision(symbol, long_score, short_score,
                                   direction, "REVERSAL_WEAK", m15_dir, None,
                                   "HOLD (reversal %.1f < %.1f)" % (raw_score, reversal_min))
                return {"long_score": long_score, "short_score": short_score,
                        "direction": direction, "gate": "REVERSAL_WEAK",
                        "atr": atr_val, "regime": regime, "m15_dir": m15_dir}

            # M15 must confirm for reversals (stricter than entry)
            if m15_dir != direction:
                return {"long_score": long_score, "short_score": short_score,
                        "direction": direction, "gate": "REVERSAL_M15",
                        "m15_dir": m15_dir, "atr": atr_val, "regime": regime}

            meta_prob = self._meta_label_check(symbol, direction, ind, bi)
            exit_pnl = self._get_position_pnl(symbol)

            # Run MasterBrain for reversal risk scaling
            rev_risk = MAX_RISK_PER_TRADE_PCT  # safe default
            if self._master_brain:
                try:
                    rev_eval = self._master_brain.evaluate_entry(
                        symbol=symbol, direction=direction, score=raw_score,
                        regime=regime, meta_prob=meta_prob, m15_dir=m15_dir)
                    rev_risk = float(rev_eval.get("risk_pct", MAX_RISK_PER_TRADE_PCT))
                except Exception:
                    pass

            self._log_decision(symbol, long_score, short_score,
                               direction, "REVERSAL", m15_dir, meta_prob,
                               "REVERSAL %s->%s score=%.1f pnl=%.2f risk=%.2f%%" % (
                                   current_dir, direction, raw_score, exit_pnl, rev_risk))
            self._record_trade_result(symbol)
            self._log_trade(symbol, current_dir, raw_score, "REVERSAL", pnl=exit_pnl)
            self.executor.reverse_position(symbol, direction, atr_val, risk_pct=rev_risk)
            return {"long_score": long_score, "short_score": short_score,
                    "direction": direction, "gate": "REVERSAL",
                    "meta_prob": meta_prob, "atr": atr_val, "regime": regime,
                    "m15_dir": m15_dir}

        # No existing position — evaluate new entry
        if not m15_aligned:
            self._log_decision(symbol, long_score, short_score,
                               direction, "M15_DISAGREE", m15_dir, None,
                               "SKIP (M15=%s != H1=%s)" % (m15_dir, direction))
            return {"long_score": long_score, "short_score": short_score,
                    "direction": direction, "gate": "M15_DISAGREE",
                    "m15_dir": m15_dir, "atr": atr_val, "regime": regime}

        # ─── 5b. TICK MOMENTUM CHECK (avoid entering against micro-trend) ───
        ticks = self.state.get_tick_history(symbol, 10)
        if ticks and len(ticks) >= 5:
            tick_momentum = float(ticks[-1].bid) - float(ticks[-5].bid)
            if direction == "LONG" and tick_momentum < 0:
                was_delayed = self._tick_delayed.get(symbol, False)
                self._tick_delayed[symbol] = True
                if not was_delayed:
                    # First delay — skip this cycle, try next
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "TICK_DELAY", m15_dir, None,
                                       "DELAY (ticks falling %.5f, want LONG)" % tick_momentum)
                    return {"long_score": long_score, "short_score": short_score,
                            "direction": direction, "gate": "TICK_DELAY",
                            "tick_momentum": float(tick_momentum), "atr": atr_val,
                            "regime": regime, "m15_dir": m15_dir}
                # Already delayed once — proceed (don't block indefinitely)
                log.info("[%s] Tick delay expired, proceeding with LONG entry", symbol)
            elif direction == "SHORT" and tick_momentum > 0:
                was_delayed = self._tick_delayed.get(symbol, False)
                self._tick_delayed[symbol] = True
                if not was_delayed:
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "TICK_DELAY", m15_dir, None,
                                       "DELAY (ticks rising %.5f, want SHORT)" % tick_momentum)
                    return {"long_score": long_score, "short_score": short_score,
                            "direction": direction, "gate": "TICK_DELAY",
                            "tick_momentum": float(tick_momentum), "atr": atr_val,
                            "regime": regime, "m15_dir": m15_dir}
                log.info("[%s] Tick delay expired, proceeding with SHORT entry", symbol)
            else:
                # Ticks aligned with direction — clear delay flag
                self._tick_delayed[symbol] = False
        else:
            # Not enough ticks — clear delay flag, don't block
            self._tick_delayed[symbol] = False

        # ─── 5c. FRESH MOMENTUM GATE (per-symbol, if enabled) ───
        sym_mode = SMART_ENTRY_MODE.get(symbol, {})
        if sym_mode.get("fresh_momentum", False) and bi >= 3 and "mh" in ind:
            rsi_val = ind["rs"][bi] if not np.isnan(ind["rs"][bi]) else 50

            # RSI exhaustion check (tighter: 75/25 instead of 78/22)
            if (direction == "LONG" and rsi_val > 75) or (direction == "SHORT" and rsi_val < 25):
                self._log_decision(symbol, long_score, short_score,
                                   direction, "RSI_EXHAUST", m15_dir, None,
                                   "SKIP (RSI %.1f exhausted for %s)" % (rsi_val, direction))
                return {"long_score": long_score, "short_score": short_score,
                        "direction": direction, "gate": "RSI_EXHAUSTION",
                        "atr": atr_val, "regime": regime, "m15_dir": m15_dir}

            mh = ind["mh"]
            if not np.isnan(mh[bi]) and not np.isnan(mh[bi-1]) and not np.isnan(mh[bi-2]):
                if direction == "LONG":
                    fresh = mh[bi] > 0 or (mh[bi] > mh[bi-1] > mh[bi-2])
                else:
                    fresh = mh[bi] < 0 or (mh[bi] < mh[bi-1] < mh[bi-2])
                if not fresh:
                    # Allow 1-bar stale if RSI is in sweet zone (ideal entry zone)
                    rsi_sweet = (direction == "LONG" and 45 <= rsi_val <= 65) or \
                                (direction == "SHORT" and 35 <= rsi_val <= 55)
                    # Also allow if MACD just crossed (1-bar stale = just turned)
                    just_crossed = (direction == "LONG" and mh[bi] > mh[bi-1] and mh[bi-1] <= mh[bi-2]) or \
                                   (direction == "SHORT" and mh[bi] < mh[bi-1] and mh[bi-1] >= mh[bi-2])
                    if not (rsi_sweet or just_crossed):
                        self._log_decision(symbol, long_score, short_score,
                                           direction, "STALE_MOM", m15_dir, None,
                                           "SKIP (MACD not accelerating in %s direction)" % direction)
                        return {"long_score": long_score, "short_score": short_score,
                                "direction": direction, "gate": "STALE_MOMENTUM",
                                "atr": atr_val, "regime": regime, "m15_dir": m15_dir}

        # ─── 6. META-LABEL FILTER (optional) ───
        meta_prob = self._meta_label_check(symbol, direction, ind, bi)
        meta_pass = self._meta_passes(symbol, meta_prob, score=raw_score)

        if not meta_pass:
            self._log_decision(symbol, long_score, short_score,
                               direction, "META_REJECT", m15_dir, meta_prob,
                               "SKIP (meta=%.2f < %.2f)" % (
                                   meta_prob if meta_prob is not None else 0.0,
                                   META_PROB_THRESHOLD))
            return {"long_score": long_score, "short_score": short_score,
                    "direction": direction, "gate": "META_REJECT",
                    "meta_prob": meta_prob, "atr": atr_val, "regime": regime,
                    "m15_dir": m15_dir}

        # ─── 6b. MASTER BRAIN GATE (Dragon: evaluate_entry) ───
        risk_pct = MAX_RISK_PER_TRADE_PCT  # default if no MasterBrain
        master_approved = True
        master_info = {}

        if self._master_brain:
            try:
                entry_eval = self._master_brain.evaluate_entry(
                    symbol=symbol,
                    direction=direction,
                    score=raw_score,
                    regime=regime,
                    meta_prob=meta_prob,
                    m15_dir=m15_dir,
                )
                master_approved = bool(entry_eval.get("approved", True))
                risk_pct = float(entry_eval.get("risk_pct", MAX_RISK_PER_TRADE_PCT))
                master_info = entry_eval

                if not master_approved:
                    reject_reason = entry_eval.get("reason", "master_brain_reject")
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "MASTER_REJECT", m15_dir, meta_prob,
                                       "SKIP (MasterBrain: %s)" % reject_reason)
                    return {"long_score": long_score, "short_score": short_score,
                            "direction": direction, "gate": "MASTER_REJECT",
                            "meta_prob": meta_prob, "atr": atr_val, "regime": regime,
                            "master_reason": reject_reason, "m15_dir": m15_dir}
            except Exception as e:
                log.warning("[%s] MasterBrain evaluate_entry failed: %s — using default risk", symbol, e)
                # Graceful degradation: proceed with default risk
                risk_pct = MAX_RISK_PER_TRADE_PCT

        # ─── 6c. SMART ENTRY INTELLIGENCE (pullback + USD + volume) ───
        if self._smart_entry:
            try:
                sym_cat = SYMBOLS[symbol].category if symbol in SYMBOLS else "Forex"
                smart_eval = self._smart_entry.evaluate(symbol, direction, atr_val, sym_cat)
                if not smart_eval["approved"]:
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "SMART_REJECT", m15_dir, meta_prob,
                                       "SKIP (%s)" % smart_eval["reason"])
                    return {"long_score": long_score, "short_score": short_score,
                            "direction": direction, "gate": smart_eval["reason"],
                            "meta_prob": meta_prob, "atr": atr_val, "regime": regime,
                            "m15_dir": m15_dir}
                # Apply smart entry risk multiplier
                smart_mult = smart_eval["risk_mult"]
                risk_pct *= smart_mult
                if smart_mult != 1.0:
                    details = smart_eval.get("details", {})
                    pb_state = details.get("pullback", {}).get("state", "?")
                    usd_align = details.get("usd", {}).get("align", 0)
                    vol_ratio = details.get("volume", {}).get("ratio", 1.0)
                    log.info("[%s] SmartEntry: mult=%.2f (pullback=%s, usd=%.2f, vol=%.1fx) → risk=%.3f%%",
                             symbol, smart_mult, pb_state, usd_align, vol_ratio, risk_pct)
            except Exception as e:
                log.debug("[%s] SmartEntry failed: %s — proceeding", symbol, e)

        # ─── 6d. OBSERVER FEEDBACK (learning engine market intelligence) ───
        if self._learning_engine:
            try:
                # Check if observer says skip this symbol entirely
                skip, skip_reason = self._learning_engine.should_skip_symbol(symbol)
                if skip:
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "OBSERVER_SKIP", m15_dir, meta_prob,
                                       "SKIP (%s)" % skip_reason)
                    return {"long_score": long_score, "short_score": short_score,
                            "direction": direction, "gate": "OBSERVER_SKIP",
                            "meta_prob": meta_prob, "atr": atr_val, "regime": regime,
                            "m15_dir": m15_dir}

                # Apply observer entry bias to risk
                mq = self._learning_engine.get_market_quality(symbol)
                obs_bias = mq.get("entry_bias", 1.0)
                if obs_bias != 1.0:
                    risk_pct *= obs_bias
                    log.info("[%s] Observer: bias=%.2f (mom=%s, regime_stable=%s, vol=%s) → risk=%.3f%%",
                             symbol, obs_bias, mq["score_momentum"],
                             mq["regime_stable"], mq["volatility_regime"], risk_pct)
            except Exception:
                pass

        # ─── 6e. TRADE INTELLIGENCE (pattern + velocity + cross-symbol) ───
        if self._trade_intel:
            try:
                intel = self._trade_intel.get_entry_intelligence(symbol, direction, raw_score, regime)
                boost = intel.get("combined_boost", 1.0)
                if boost != 1.0:
                    risk_pct *= boost
                    log.info("[%s] TradeIntel: boost=%.2f (pattern=%.0f%% vel=%s cross=%.1f) → risk=%.3f%%",
                             symbol, boost,
                             intel["pattern_edge"]["wr"]*100,
                             intel["score_velocity"]["state"],
                             intel["cross_momentum"]["alignment"],
                             risk_pct)
            except Exception as e:
                log.debug("[%s] TradeIntel failed: %s — proceeding", symbol, e)

        # ─── 6f. PATTERN + ORDER FLOW + LEVEL INTELLIGENCE ───
        intel_boost = 1.0
        if self._pattern_learner:
            try:
                pat = self._pattern_learner.get_pattern_signal(symbol)
                if pat["pattern_bias"] != 0 and pat["confidence"] > 0.5:
                    # Positive bias for our direction = boost, negative = reduce
                    dir_mult = 1 if direction == "LONG" else -1
                    bias = pat["pattern_bias"] * dir_mult
                    intel_boost *= (1.0 + bias * 0.15)  # max +/-15% from patterns
            except Exception:
                pass

        if self._order_flow:
            try:
                flow = self._order_flow.get_flow_signal(symbol)
                if flow["exhaustion"]:
                    # Exhaustion = potential reversal, reduce risk
                    intel_boost *= 0.8
                elif abs(flow["bias"]) > 0.5:
                    dir_mult = 1 if direction == "LONG" else -1
                    if flow["bias"] * dir_mult > 0:
                        intel_boost *= 1.1  # flow supports our direction
                    else:
                        intel_boost *= 0.85  # flow opposes
            except Exception:
                pass

        if self._level_memory:
            try:
                tick_for_level = self.state.get_tick(symbol)
                cur_price_here = float(tick_for_level.bid) if tick_for_level and hasattr(tick_for_level, 'bid') else 0
                if cur_price_here > 0:
                    lvl = self._level_memory.get_level_intelligence(symbol, cur_price_here)
                    if lvl["at_learned_level"]:
                        # At a level with known bounce rate
                        if direction == "LONG" and lvl["support_strength"] > 0.6:
                            intel_boost *= 1.1  # entering at strong support = good
                        elif direction == "SHORT" and lvl["resistance_strength"] > 0.6:
                            intel_boost *= 1.1  # entering at strong resistance = good
                        elif direction == "LONG" and lvl["resistance_strength"] > 0.7:
                            intel_boost *= 0.8  # LONG into strong resistance = bad
            except Exception:
                pass

        if intel_boost != 1.0:
            risk_pct *= max(0.5, min(1.5, intel_boost))
            log.info("[%s] RL Intel boost: %.2f → risk=%.3f%%", symbol, intel_boost, risk_pct)

        # ─── 7. RISK CHECKS (warn only, never block — per user preference) ───
        risk_warnings = []
        if daily_loss_pct >= DAILY_LOSS_LIMIT_PCT:
            risk_warnings.append("DailyLoss=%.1f%%" % daily_loss_pct)
        if dd_pct >= DD_PAUSE_THRESHOLD:
            risk_warnings.append("DD=%.1f%%" % dd_pct)

        total_exposure = self.executor._get_total_exposure()
        if total_exposure + risk_pct > MAX_TOTAL_EXPOSURE_PCT:
            risk_warnings.append("Exposure=%.1f%%" % total_exposure)

        risk_ok = len(risk_warnings) == 0
        if risk_warnings:
            log.warning("[%s] RISK WARNINGS: %s — proceeding anyway",
                        symbol, ", ".join(risk_warnings))

        # ─── 8. MTF INTELLIGENCE — smart SL/TP + entry quality gate ───
        mtf_data = None
        smart_atr = float(atr_val)
        if self._mtf:
            try:
                mtf_data = self._mtf.analyze(symbol)
                entry_quality = mtf_data.get("entry_quality", 50)
                confluence = mtf_data.get("confluence", 2)
                optimal_sl = mtf_data.get("optimal_sl", 0)

                # Gate: reject entries with low MTF quality (< 30)
                if entry_quality < 20:
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "MTF_LOW", m15_dir, meta_prob,
                                       "SKIP (MTF quality=%d < 30, confluence=%d)" % (entry_quality, confluence))
                    return {"long_score": long_score, "short_score": short_score,
                            "direction": direction, "gate": "MTF_LOW_QUALITY",
                            "entry_quality": entry_quality, "confluence": confluence,
                            "atr": atr_val, "regime": regime, "m15_dir": m15_dir}

                # Use smart SL if available BUT cap at per-symbol ATR SL override
                # Smart SL can be TIGHTER (better structure) but never WIDER than grid-tuned max
                if optimal_sl > 0:
                    from config import SYMBOL_ATR_SL_OVERRIDE
                    sym_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, ATR_SL_MULTIPLIER)
                    max_sl = atr_val * sym_sl_mult  # grid-tuned maximum
                    capped_sl = min(optimal_sl, max_sl)  # never wider than tuned
                    smart_atr = capped_sl / ATR_SL_MULTIPLIER
                    log.info("[%s] MTF: quality=%d confluence=%d smartSL=%.2f (cap=%.2f ATR=%.2f)",
                             symbol, entry_quality, confluence, capped_sl, max_sl, atr_val)
            except Exception as e:
                log.debug("[%s] MTF analyze failed: %s — using ATR SL", symbol, e)

        # DD reduction
        if dd_pct >= DD_REDUCE_THRESHOLD:
            risk_pct *= 0.5
            log.info("[%s] DD %.1f%% >= %.1f%% — risk halved to %.3f%%",
                     symbol, dd_pct, DD_REDUCE_THRESHOLD, risk_pct)

        meta_str = "%.2f (PASS)" % meta_prob if meta_prob is not None else "N/A (pure scoring)"
        risk_str = "OK" if risk_ok else "WARN(%s)" % ",".join(risk_warnings)

        self._log_decision(symbol, long_score, short_score,
                           direction, "ENTER", m15_dir, meta_prob,
                           "ENTER (MTF q=%d c=%d)" % (
                               mtf_data.get("entry_quality", 0) if mtf_data else 0,
                               mtf_data.get("confluence", 0) if mtf_data else 0))

        # Get smart TP from MTF (liquidity + fibonacci based)
        mtf_tp = None
        if mtf_data:
            mtf_tp_val = mtf_data.get("optimal_tp", 0)
            if mtf_tp_val > 0:
                mtf_tp = mtf_tp_val

            # LIQUIDITY ZONE ENTRY FILTER: don't enter into nearby resistance/support
            liq = mtf_data.get("liquidity", {})
            if liq.get("at_liquidity", False):
                magnet_above = liq.get("magnet_above", 0)
                magnet_below = liq.get("magnet_below", 0)
                # LONG into resistance (strong magnet above but we're AT it) = bad
                if direction == "LONG" and magnet_below > magnet_above * 1.5:
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "LIQ_RESIST", m15_dir, meta_prob,
                                       "SKIP (at liquidity zone, support > resistance)")
                    return {"long_score": long_score, "short_score": short_score,
                            "direction": direction, "gate": "LIQUIDITY_RESISTANCE",
                            "atr": atr_val, "regime": regime, "m15_dir": m15_dir}
                # SHORT into support (strong magnet below but we're AT it) = bad
                if direction == "SHORT" and magnet_above > magnet_below * 1.5:
                    self._log_decision(symbol, long_score, short_score,
                                       direction, "LIQ_SUPPORT", m15_dir, meta_prob,
                                       "SKIP (at liquidity zone, resistance > support)")
                    return {"long_score": long_score, "short_score": short_score,
                            "direction": direction, "gate": "LIQUIDITY_SUPPORT",
                            "atr": atr_val, "regime": regime, "m15_dir": m15_dir}

        success = self.executor.open_trade(symbol, direction, smart_atr,
                                           risk_pct=risk_pct, smart_tp=mtf_tp)

        if success:
            self._log_trade(symbol, direction, raw_score, "ENTRY")
            # Get the lot size that was actually used from executor
            positions = self.executor.get_positions_info()
            lot_str = "?"
            for p in positions:
                if p["symbol"] == symbol:
                    lot_str = "%.2f" % float(p["volume"])
                    break

            log.info("DRAGON SCORED: %s %s %.1f | M15: %s (aligned) | META: %s | RISK: %s (%.2f%%) | REGIME: %s (min=%.1f) | MASTER: %s | -> ENTER %s lots",
                     symbol, direction, raw_score, m15_dir, meta_str, risk_str, risk_pct,
                     regime, adaptive_min,
                     "approved" if self._master_brain else "N/A",
                     lot_str)

        return {"long_score": long_score, "short_score": short_score,
                "direction": direction, "gate": "ENTERED" if success else "EXEC_FAIL",
                "meta_prob": meta_prob, "atr": atr_val,
                "risk_warnings": risk_warnings, "regime": regime,
                "adaptive_min_score": adaptive_min,
                "risk_pct": risk_pct,
                "master_info": master_info, "m15_dir": m15_dir}

    # ═══════════════════════════════════════════════════════════════
    #  MASTER BRAIN TRADE RESULT RECORDING
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

            # Record to learning engine for adaptive risk
            if self._learning_engine:
                entry_price = self.executor._entry_prices.get(symbol, 0)
                sl_dist = self.executor._entry_sl_dist.get(symbol, 0)
                # R-multiple = actual PnL / intended dollar risk per trade
                # NOT pnl / (sl_dist * 100) which was wrong by 100x
                equity = float(self.state.get_agent_state().get("equity", 1000))
                dollar_risk = equity * (MAX_RISK_PER_TRADE_PCT / 100.0)
                r_mult = pnl / dollar_risk if dollar_risk > 0 else 0
                self._learning_engine.record_trade(
                    symbol=symbol, direction=direction, pnl=pnl,
                    entry_price=entry_price, r_multiple=r_mult,
                    exit_reason=reason,
                )

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
                    self._rl_learner.record_outcome(
                        symbol=symbol, direction=last_direction, pnl=pnl,
                        r_multiple=r_mult, score=last_score, regime=regime_now,
                        exit_reason=reason,
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
        If M15 direction has flipped against our position, close it.
        This provides faster exit than waiting for H1 score to flip.
        """
        current_dir = self.executor.get_position_direction(symbol)
        if current_dir == "FLAT":
            return

        # Skip M15 exit for volatile symbols — M15 flips too often, kills winners
        # XAUUSD: 19% WR, 8/10 exits were M15 reversal
        cfg = SYMBOLS.get(symbol)
        if cfg and cfg.category == "Gold":
            return  # gold too choppy for M15 exit — let trail/SL handle

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
        Default: trending 6.0, ranging 8.0, volatile 7.0, low_vol 7.0.
        """
        # Check per-symbol override first
        if symbol and symbol in DRAGON_SYMBOL_MIN_SCORE:
            sym_scores = DRAGON_SYMBOL_MIN_SCORE[symbol]
            if regime in sym_scores:
                return float(sym_scores[regime])

        regime_min_scores = {
            "trending": float(6.0),
            "ranging":  float(8.0),
            "volatile": float(7.0),
            "low_vol":  float(7.0),
        }
        return float(regime_min_scores.get(regime, DRAGON_MIN_SCORE_BASELINE))

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
