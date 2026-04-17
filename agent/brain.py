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
)
from data.tick_streamer import SharedState
from execution.executor import Executor

# ── Momentum scorer internals (proven scoring system) ──
from signals.momentum_scorer import (
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, MIN_SCORE,
    REGIME_PARAMS,
)
from data.feature_engine import FeatureEngine

log = logging.getLogger("dragon.brain")

# ═══ CONSTANTS ═══
CYCLE_INTERVAL_S = 1.0           # 1-second decision cycle
META_PROB_THRESHOLD = 0.50       # meta-label skip threshold (Dragon: was 0.40)
META_AUC_MIN = 0.55              # minimum AUC to trust meta-label
H1_MIN_BARS = 100                # minimum H1 bars for scoring
M15_MIN_BARS = 50                # minimum M15 bars for direction check


class AgentBrain:
    """Hybrid trading agent: rule-based scoring + optional ML meta-label + MasterBrain gating."""

    def __init__(self, state: SharedState, mt5, executor: Executor,
                 meta_model=None, master_brain=None, exit_intelligence=None):
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
        self._ind_cache_ttl = 2.0  # seconds before recompute

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
        """Single cycle: daily reset, DD check, process symbols, intelligent exits, manage positions, update dashboard."""
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

        # ═══ MANAGE TRAILING SL + M15 REVERSAL EXIT ═══
        for symbol in SYMBOLS:
            try:
                if self.executor.has_position(symbol):
                    self.executor.manage_trailing_sl(symbol)
                    self._check_m15_reversal_exit(symbol)
            except Exception as e:
                log.warning("[%s] Trailing/M15 exit error: %s", symbol, e)

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

        # ─── 1. SESSION FILTER (non-crypto: 06-22 UTC only) ───
        if cfg.category != "Crypto":
            hour_utc = int(datetime.now(timezone.utc).hour)
            if hour_utc < SESSION_START_UTC or hour_utc >= SESSION_END_UTC:
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

        # ─── 3b. REGIME DETECTION + ADAPTIVE MIN_SCORE ───
        regime = self._get_regime_from_bbw(ind, bi)
        adaptive_min = self._get_adaptive_min_score(regime)

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

        # ─── 5. GATE CHECKS ───

        # Gate A: M15 must NOT oppose H1 direction (FLAT = allowed, opposite = blocked)
        # Changed from "must agree" to "must not oppose" — FLAT blocks missed the best
        # entries when M15 was consolidating before the next trend leg
        m15_dir = self._get_m15_direction(symbol)
        m15_opposing = (
            (direction == "LONG" and m15_dir == "SHORT") or
            (direction == "SHORT" and m15_dir == "LONG")
        )
        m15_aligned = not m15_opposing  # FLAT is now allowed

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
                    "atr": atr_val, "regime": regime}

        if has_pos and current_dir != direction:
            # Opposite direction — signal reversal: close and flip
            meta_prob = self._meta_label_check(symbol, direction, ind, bi)
            meta_pass = self._meta_passes(symbol, meta_prob)

            self._log_decision(symbol, long_score, short_score,
                               direction, "REVERSAL", m15_dir, meta_prob,
                               "REVERSAL %s -> %s" % (current_dir, direction))

            # Record the closing trade result with MasterBrain before reversal
            self._record_trade_result(symbol, reason="reversal_exit")

            # For reversals, don't require M15 alignment or meta-label
            # (the score itself is strong enough for a flip)
            self.executor.reverse_position(symbol, direction, atr_val)
            self._log_trade(symbol, direction, raw_score, "REVERSAL")
            return {"long_score": long_score, "short_score": short_score,
                    "direction": direction, "gate": "REVERSAL",
                    "meta_prob": meta_prob, "atr": atr_val, "regime": regime}

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
                            "regime": regime}
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
                            "regime": regime}
                log.info("[%s] Tick delay expired, proceeding with SHORT entry", symbol)
            else:
                # Ticks aligned with direction — clear delay flag
                self._tick_delayed[symbol] = False
        else:
            # Not enough ticks — clear delay flag, don't block
            self._tick_delayed[symbol] = False

        # ─── 6. META-LABEL FILTER (optional) ───
        meta_prob = self._meta_label_check(symbol, direction, ind, bi)
        meta_pass = self._meta_passes(symbol, meta_prob)

        if not meta_pass:
            self._log_decision(symbol, long_score, short_score,
                               direction, "META_REJECT", m15_dir, meta_prob,
                               "SKIP (meta=%.2f < %.2f)" % (
                                   meta_prob if meta_prob is not None else 0.0,
                                   META_PROB_THRESHOLD))
            return {"long_score": long_score, "short_score": short_score,
                    "direction": direction, "gate": "META_REJECT",
                    "meta_prob": meta_prob, "atr": atr_val, "regime": regime}

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
                            "master_reason": reject_reason}
            except Exception as e:
                log.warning("[%s] MasterBrain evaluate_entry failed: %s — using default risk", symbol, e)
                # Graceful degradation: proceed with default risk
                risk_pct = MAX_RISK_PER_TRADE_PCT

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

        # ─── 8. EXECUTE: MasterBrain risk-scaled lot sizing, ATR SL ───
        # DD reduction: halve size at DD threshold
        size_mult = 0.5 if dd_pct >= DD_REDUCE_THRESHOLD else 1.0
        adjusted_atr = float(atr_val / size_mult) if size_mult < 1.0 else float(atr_val)

        meta_str = "%.2f (PASS)" % meta_prob if meta_prob is not None else "N/A (pure scoring)"
        risk_str = "OK" if risk_ok else "WARN(%s)" % ",".join(risk_warnings)

        self._log_decision(symbol, long_score, short_score,
                           direction, "ENTER", m15_dir, meta_prob,
                           "ENTER")

        success = self.executor.open_trade(symbol, direction, adjusted_atr,
                                           risk_pct=risk_pct)

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
                "master_info": master_info}

    # ═══════════════════════════════════════════════════════════════
    #  MASTER BRAIN TRADE RESULT RECORDING
    # ═══════════════════════════════════════════════════════════════

    def _record_trade_result(self, symbol, reason="unknown"):
        """Record a trade result with MasterBrain when a position closes."""
        if not self._master_brain:
            return

        try:
            # Get PnL info from executor before the position is removed
            positions = self.executor.get_positions_info()
            pnl = float(0.0)
            direction = "FLAT"
            for p in positions:
                if p["symbol"] == symbol:
                    pnl = float(p.get("pnl", 0.0))
                    direction = str(p.get("direction", "FLAT")).upper()
                    break

            self._master_brain.record_trade_result(
                symbol=symbol,
                direction=direction,
                pnl=pnl,
            )
            log.info("[%s] MasterBrain recorded trade result: pnl=%.2f, reason=%s",
                     symbol, pnl, reason)
        except Exception as e:
            log.warning("[%s] MasterBrain record_trade_result failed: %s", symbol, e)

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

        m15_dir = self._get_m15_direction(symbol)
        if m15_dir == "FLAT":
            return  # ambiguous — hold

        if (current_dir == "LONG" and m15_dir == "SHORT") or \
           (current_dir == "SHORT" and m15_dir == "LONG"):
            log.info("[%s] M15 REVERSAL EXIT: position=%s, M15=%s",
                     symbol, current_dir, m15_dir)
            # Record result with MasterBrain before closing
            self._record_trade_result(symbol, reason="m15_reversal_exit")
            self.executor.close_position(symbol, "M15ReversalExit")
            self._log_trade(symbol, current_dir, 0.0, "M15_EXIT")

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

    def _meta_passes(self, symbol, meta_prob):
        """
        Determine if meta-label filter passes.
        Returns True if:
          - ML is disabled (pure scoring mode — always pass)
          - No model for this symbol (graceful degradation)
          - Model says probability >= threshold
        Returns False only if model actively rejects (prob < threshold).
        """
        if meta_prob is None:
            return True  # No model = pure scoring, always pass
        return float(meta_prob) >= META_PROB_THRESHOLD

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

    def _get_adaptive_min_score(self, regime):
        """
        Dragon-level adaptive MIN_SCORE by market regime.
        - trending: 6.0 (selective even in trends)
        - ranging:  8.0 (very selective, most signals are noise)
        - volatile: 7.0 (high bar for volatile markets)
        - low_vol:  7.0 (standard Dragon level)
        """
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

    def _log_trade(self, symbol, direction, score, action):
        """Log trade for dashboard display and update win streak."""
        entry = {
            "timestamp": str(datetime.now(timezone.utc).strftime("%H:%M:%S")),
            "symbol": str(symbol),
            "direction": str(direction).lower(),
            "score": float(round(score, 1)),
            "action": str(action),
            "pnl": float(0.0),
            "regime": str(self._get_regime(symbol)),
        }
        self._trade_log.append(entry)

        # Update win streak from closed positions for meta-label feature
        if action in ("M15_EXIT", "REVERSAL") or action.startswith("INTEL_EXIT"):
            # Check last PnL from executor positions
            positions = self.executor.get_positions_info()
            last_pnl = float(0.0)
            for p in positions:
                if p["symbol"] == symbol:
                    last_pnl = float(p.get("pnl", 0.0))
                    break
            if last_pnl > 0:
                self._recent_win_streak = int(max(self._recent_win_streak + 1, 1))
            elif last_pnl < 0:
                self._recent_win_streak = int(min(self._recent_win_streak - 1, -1))

        log.info("[%s] Trade logged: %s %s score=%.1f",
                 symbol, action, direction, score)
