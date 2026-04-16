"""
Beast Trader — Agent Brain.
Main decision loop running every 500ms.
Reads ticks, generates features, gets ML prediction, manages risk, executes.
"""
import time
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SYMBOLS, TICK_INTERVAL_MS, CONFIDENCE_THRESHOLD,
    MAX_RISK_PER_TRADE_PCT, DAILY_LOSS_LIMIT_PCT,
    DD_REDUCE_THRESHOLD, DD_PAUSE_THRESHOLD, DD_EMERGENCY_CLOSE,
    SESSION_START_UTC, SESSION_END_UTC, STARTING_BALANCE,
)
from data.tick_streamer import SharedState
from data.feature_engine import FeatureEngine
from models.signal_model import SignalModel
from execution.executor import Executor

log = logging.getLogger("beast.brain")


class AgentBrain:
    """Main trading agent decision loop."""

    def __init__(self, state: SharedState, mt5, feature_engine: FeatureEngine,
                 model: SignalModel, executor: Executor):
        self.state = state
        self.mt5 = mt5
        self.feature_engine = feature_engine
        self.model = model
        self.executor = executor
        self.running = False
        self._thread = None
        self._cycle = 0
        self._daily_start_equity = 0.0
        self._daily_loss = 0.0
        self._last_day = None
        self._trade_log = []

    def start(self):
        """Start the agent brain in a background thread."""
        self.running = True
        self.state.update_agent("running", True)

        # Initialize daily tracking
        equity = float(self.state.get_agent_state().get("equity", STARTING_BALANCE))
        self._daily_start_equity = equity
        self._last_day = datetime.now(timezone.utc).date()

        self._thread = threading.Thread(target=self._decision_loop, daemon=True, name="AgentBrain")
        self._thread.start()
        log.info("Agent brain started (cycle=%dms, confidence_threshold=%.2f)",
                 TICK_INTERVAL_MS, CONFIDENCE_THRESHOLD)

    def stop(self):
        self.running = False
        self.state.update_agent("running", False)
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Agent brain stopped after %d cycles", self._cycle)

    def _decision_loop(self):
        """Main loop: every 500ms, evaluate all symbols."""
        interval = TICK_INTERVAL_MS / 1000.0

        while self.running:
            loop_start = time.time()
            self._cycle += 1

            try:
                # Reset daily tracking at midnight UTC
                today = datetime.now(timezone.utc).date()
                if today != self._last_day:
                    self._last_day = today
                    equity = float(self.state.get_agent_state().get("equity", STARTING_BALANCE))
                    self._daily_start_equity = equity
                    self._daily_loss = 0.0
                    log.info("New trading day — daily start equity: $%.2f", equity)

                # Get current account state
                equity = float(self.state.get_agent_state().get("equity", STARTING_BALANCE))
                balance = float(self.state.get_agent_state().get("balance", STARTING_BALANCE))
                dd_pct = float(self.state.get_agent_state().get("dd_pct", 0))

                # Daily loss calculation
                daily_pnl = equity - self._daily_start_equity
                daily_loss_pct = abs(daily_pnl) / self._daily_start_equity * 100 if (
                    daily_pnl < 0 and self._daily_start_equity > 0
                ) else 0.0
                self._daily_loss = daily_loss_pct

                # ═══ EMERGENCY DD CHECK ═══
                if dd_pct >= DD_EMERGENCY_CLOSE:
                    log.critical("EMERGENCY DD %.1f%% >= %.1f%% — CLOSING ALL",
                                 dd_pct, DD_EMERGENCY_CLOSE)
                    self.executor.close_all("EmergencyDD")
                    time.sleep(10)
                    continue

                # ═══ PROCESS EACH SYMBOL ═══
                for symbol in SYMBOLS:
                    try:
                        self._process_symbol(symbol, equity, dd_pct, daily_loss_pct)
                    except Exception as e:
                        log.error("[%s] Process error: %s", symbol, e)

                # ═══ MANAGE TRAILING SL ═══
                for symbol in SYMBOLS:
                    try:
                        if self.executor.has_position(symbol):
                            self.executor.manage_trailing_sl(symbol)
                    except Exception as e:
                        log.warning("[%s] Trailing SL error: %s", symbol, e)

                # ═══ UPDATE DASHBOARD STATE ═══
                self.state.update_agent("cycle", self._cycle)
                self.state.update_agent("daily_loss", daily_loss_pct)
                self.state.update_agent("positions", self.executor.get_positions_info())
                self.state.update_agent("trade_log", self._trade_log[-50:])

                # Model confidence for dashboard
                confidences = {}
                for sym in SYMBOLS:
                    features = self.feature_engine.generate(sym)
                    if features is not None and self.model.has_model(sym):
                        pred = self.model.predict(sym, features)
                        confidences[sym] = pred
                    else:
                        confidences[sym] = {
                            "direction": "FLAT", "prob_up": 0.5,
                            "prob_down": 0.5, "confidence": 0.0,
                        }
                self.state.update_agent("model_confidence", confidences)

                # Feature importance for dashboard
                importances = {}
                for sym in SYMBOLS:
                    imp = self.model.get_importance(sym)
                    if imp:
                        importances[sym] = imp
                self.state.update_agent("feature_importance", importances)

            except Exception as e:
                log.error("Brain cycle error: %s", e)

            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_symbol(self, symbol, equity, dd_pct, daily_loss_pct):
        """Process a single symbol: generate features, predict, decide, execute."""
        cfg = SYMBOLS[symbol]

        # ═══ SESSION FILTER (non-crypto: 06-22 UTC) ═══
        if cfg.category != "Crypto":
            hour_utc = datetime.now(timezone.utc).hour
            if hour_utc < SESSION_START_UTC or hour_utc >= SESSION_END_UTC:
                return

        # ═══ GENERATE FEATURES ═══
        features = self.feature_engine.generate(symbol)
        if features is None:
            return

        # ═══ GET MODEL PREDICTION ═══
        if not self.model.has_model(symbol):
            return

        prediction = self.model.predict(symbol, features)
        direction = prediction["direction"]
        confidence = prediction["confidence"]

        # ═══ CURRENT POSITION ═══
        current_dir = self.executor.get_position_direction(symbol)
        has_pos = current_dir != "FLAT"

        # ═══ M15 REVERSAL DETECTION ═══
        # If we have a position and model predicts opposite direction with high confidence, close/flip
        if has_pos and direction != "FLAT" and direction != current_dir:
            if confidence >= CONFIDENCE_THRESHOLD * 0.8:  # Slightly lower threshold for reversals
                atr = self._get_atr(symbol)
                if atr > 0:
                    log.info("[%s] REVERSAL: %s -> %s (confidence=%.2f)",
                             symbol, current_dir, direction, confidence)
                    self.executor.reverse_position(symbol, direction, atr)
                    self._log_trade(symbol, direction, confidence, "REVERSAL")
                    return

        # ═══ RISK CHECKS (warn only, don't block) ═══
        risk_warnings = []
        if daily_loss_pct >= DAILY_LOSS_LIMIT_PCT:
            risk_warnings.append(f"Daily loss {daily_loss_pct:.1f}% >= {DAILY_LOSS_LIMIT_PCT}%")
        if dd_pct >= DD_PAUSE_THRESHOLD:
            risk_warnings.append(f"DD {dd_pct:.1f}% >= {DD_PAUSE_THRESHOLD}%")

        if risk_warnings:
            log.warning("[%s] RISK WARNINGS: %s — proceeding anyway", symbol, "; ".join(risk_warnings))

        # ═══ ENTRY DECISION ═══
        if not has_pos and direction != "FLAT" and confidence >= CONFIDENCE_THRESHOLD:
            atr = self._get_atr(symbol)
            if atr <= 0:
                return

            # DD reduction: halve size at DD threshold
            size_mult = 0.5 if dd_pct >= DD_REDUCE_THRESHOLD else 1.0

            log.info("[%s] SIGNAL: %s (confidence=%.2f, ATR=%.5f, size_mult=%.1f)",
                     symbol, direction, confidence, atr, size_mult)

            # Execute — adjusted ATR for size reduction (larger ATR = smaller lot)
            adjusted_atr = atr / size_mult if size_mult < 1.0 else atr
            success = self.executor.open_trade(symbol, direction, adjusted_atr)

            if success:
                self._log_trade(symbol, direction, confidence, "ENTRY")

    def _get_atr(self, symbol):
        """Get ATR from shared state indicators."""
        ind = self.state.get_indicators(symbol)
        if ind and "atr" in ind:
            return float(ind["atr"])
        return 0.0

    def _log_trade(self, symbol, direction, confidence, action):
        """Log trade for dashboard."""
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "symbol": symbol,
            "direction": direction.lower(),
            "confidence": round(confidence, 3),
            "action": action,
            "pnl": 0.0,
            "regime": self._get_regime(symbol),
        }
        self._trade_log.append(entry)
        log.info("[%s] Trade logged: %s %s conf=%.3f", symbol, action, direction, confidence)

    def _get_regime(self, symbol):
        """Determine market regime from indicators."""
        ind = self.state.get_indicators(symbol)
        if not ind:
            return "unknown"

        adx = ind.get("adx", 25)
        st_dir = ind.get("supertrend_dir", 0)
        rsi = ind.get("rsi", 50)

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
