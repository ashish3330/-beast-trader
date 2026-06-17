"""
Dragon Trader — Master Brain (The General).

Coordinates all trading decisions. No trade enters without MasterBrain approval.
Evaluates: regime, cross-TF confluence, ML confidence, recent performance,
symbol correlation, equity curve health.
"""
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from config import (
    SYMBOLS,
    CORRELATION_PAIRS,
    MAX_RISK_PER_TRADE_PCT,
    MAX_POSITIONS,
    DRAGON_MIN_SCORE_BASELINE,
    DRAGON_SCALP_MIN_SCORE,
    DRAGON_CONFIDENCE_FLOOR,
    DRAGON_MAX_CONSECUTIVE_LOSSES,
    DRAGON_BLACKLIST_HOURS,
    DRAGON_EQUITY_SLOPE_WINDOW,
    DRAGON_STANDBY_HOURS,
    DRAGON_RISK_SCALE_MIN,
    DRAGON_RISK_SCALE_MAX,
    DRAGON_LOSS_DAY_RISK_MULT,
    SESSION_START_UTC,
    SESSION_END_UTC,
    SCALP_SESSION_START,
    SCALP_SESSION_END,
    SYMBOL_SESSION_OVERRIDE,
)

# 2026-06-05: Correlation-cluster cap config. Imported defensively in case the
# other agent's config-side change has not landed yet — fall back to disabled.
try:
    from config import (
        CORRELATION_CAP_ENABLED,
        CORRELATION_CAP_PER_CLUSTER,
        CORRELATION_CLUSTERS,
    )
except ImportError:
    CORRELATION_CAP_ENABLED = False
    CORRELATION_CAP_PER_CLUSTER = 2
    CORRELATION_CLUSTERS = {}

log = logging.getLogger("dragon.master")

_MAX_HISTORY = 100
_MAX_DAILY_TRADES = 6


class MasterBrain:
    """Dragon Trader — Master Brain (The General).

    Coordinates all trading decisions. No trade enters without MasterBrain approval.
    Evaluates: regime, cross-TF confluence, ML confidence, recent performance,
    symbol correlation, equity curve health.
    """

    def __init__(self, state, mt5, executor, meta_model=None):
        self.state = state
        self.mt5 = mt5
        self.executor = executor
        self.mtf_intelligence = None  # set by run.py
        self.learning_engine = None  # set by run.py after init
        self.meta_model = meta_model
        self.portfolio_risk = None   # set by run.py — PortfolioRiskModel

        self._lock = threading.RLock()

        # Track recent trades per symbol: list of {"symbol", "direction", "pnl", "time"}
        self._trade_history: List[dict] = []  # last 100 trades
        # 2026-05-29 audit fix: restore streak/blacklist across restart. Was
        # in-memory only → every watchdog respawn zeroed the loss-streak and
        # blacklist, so the streak protection silently evaporated (this is why
        # USDCAD could lose 5x in a row — restarts kept resetting the counter).
        _ag = state.get_agent_state() if state else {}
        self._symbol_losses: Dict[str, int] = dict(_ag.get("mb_symbol_losses", {}))
        self._blacklisted: Dict[str, float] = dict(_ag.get("mb_blacklisted", {}))
        if self._symbol_losses or self._blacklisted:
            log.info("MasterBrain restored streak state: %d loss-counters, %d blacklisted",
                     len(self._symbol_losses), len(self._blacklisted))
        self._last_favorable_time: Dict[str, float] = {}  # symbol -> unix ts
        self._daily_trades: int = 0
        self._daily_pnl: float = 0.0
        self._last_day: Optional[int] = None  # day-of-year
        self._losing_day_yesterday: bool = False
        self._session_losses: int = 0  # consecutive losses in current session
        self._session_paused: bool = False  # circuit breaker tripped
        # _win_cooldown removed 2026-05-29 — win cooldown is brain-owned now.
        self._corr_cooldown: Dict[str, float] = {}  # symbol -> cooldown_expiry for correlated sequential entries

        # ── 2026-06-18 Tier 1 #6: per-strategy daily R tracking ──
        # Independent R-sum per strategy. Reset by reset_daily(). Auto-trips
        # the strategy_kill_switch row (item #1) when threshold breached.
        # Threshold map: PER_STRATEGY_DAILY_R_KILL in config.py.
        self._daily_strategy_r: Dict[str, float] = {
            "momentum": 0.0, "fvg": 0.0, "sr": 0.0,
        }
        # Per-strategy recent-loss tracker for N-consec-losses auto-trip.
        # value: list of unix ts of recent -1R+ closes.
        self._strategy_recent_losses: Dict[str, list] = {
            "momentum": [], "fvg": [], "sr": [],
        }

    # ──────────────────────────────────────────────
    #  TIER 1 #6 — EQUITY-CURVE 3-TIER RISK SCALER
    #
    #  Replaces the binary equity_slope ±0.7×/+1.3× with a 4-tier scaler.
    #  Composed multiplicatively into the existing de-stack chain
    #  (drift × learning × portfolio): the worst protect wins, then a
    #  single boost is applied, then capped at MAX_RISK_PER_TRADE_PCT.
    # ──────────────────────────────────────────────

    def compute_risk_tier(self, intraday_dd_pct: float = 0.0) -> tuple:
        """Return (tier_name, risk_multiplier) based on 7d R-sum and
        intraday DD %. Slots into protect_mults via evaluate_entry.

        Default OFF behind EQUITY_TIER_SCALER_ENABLED — when False this
        returns ("NEUTRAL", 1.0) so behaviour is unchanged.

        Fail-open: any exception returns ("NEUTRAL", 1.0).
        """
        try:
            from config import EQUITY_TIER_SCALER_ENABLED
            if not EQUITY_TIER_SCALER_ENABLED:
                return ("NEUTRAL", 1.0)
            from config import (
                EQUITY_TIER_GROWTH_R, EQUITY_TIER_LOCKDOWN_R,
                EQUITY_TIER_GROWTH_MULT, EQUITY_TIER_DEFENSE_MULT,
                EQUITY_TIER_LOCKDOWN_MULT,
                EQUITY_TIER_DEFENSE_DD_PCT, EQUITY_TIER_LOCKDOWN_DD_PCT,
            )
        except Exception:
            return ("NEUTRAL", 1.0)
        try:
            cutoff = time.time() - 7 * 86400
            with self._lock:
                # Use r_multiple where available, else pnl/equity proxy.
                r_7d = 0.0
                for t in self._trade_history:
                    if float(t.get("time", 0)) < cutoff:
                        continue
                    if "r_multiple" in t and t["r_multiple"] is not None:
                        r_7d += float(t["r_multiple"])
                    else:
                        # Crude proxy: $1 ≈ 0.01R on $2325 equity. Better than
                        # ignoring history entirely for the 7d window.
                        r_7d += float(t.get("pnl", 0.0)) / 50.0
            dd = float(max(0.0, intraday_dd_pct))
            if r_7d < EQUITY_TIER_LOCKDOWN_R or dd > EQUITY_TIER_LOCKDOWN_DD_PCT:
                return ("LOCKDOWN", float(EQUITY_TIER_LOCKDOWN_MULT))
            if r_7d < 0.0 or dd > EQUITY_TIER_DEFENSE_DD_PCT:
                return ("DEFENSE", float(EQUITY_TIER_DEFENSE_MULT))
            if r_7d > EQUITY_TIER_GROWTH_R and dd < 1.0:
                return ("GROWTH", float(EQUITY_TIER_GROWTH_MULT))
            return ("NEUTRAL", 1.0)
        except Exception as e:
            log.debug("compute_risk_tier failed (fail-open): %s", e)
            return ("NEUTRAL", 1.0)

    # ──────────────────────────────────────────────
    #  TIER 1 #6 — PER-STRATEGY DAILY R-CAP
    # ──────────────────────────────────────────────

    def record_strategy_r(self, strategy: str, r_multiple: float, brain=None) -> None:
        """Add closed-trade R to the per-strategy daily R-sum. If the
        threshold is breached, trip the strategy_kill_switch row via the
        AgentBrain helper (best-effort; passes silently if brain is None).

        Honours PER_STRATEGY_DAILY_R_CAP_ENABLED — when False the R is
        still tracked (cheap, no side effects) but no kill switch fires.
        """
        try:
            s = str(strategy or "").lower().strip()
            if s not in self._daily_strategy_r:
                return
            with self._lock:
                self._daily_strategy_r[s] = self._daily_strategy_r.get(s, 0.0) + float(r_multiple or 0.0)
                # Track recent -1R+ losses for N-consec auto-trip (item #1)
                if float(r_multiple) <= -0.95:
                    self._strategy_recent_losses[s].append(time.time())
                    # Trim to keep last hour only
                    cutoff = time.time() - 6 * 3600
                    self._strategy_recent_losses[s] = [
                        t for t in self._strategy_recent_losses[s] if t > cutoff
                    ]
                cur_r = self._daily_strategy_r[s]
                recent_losses = list(self._strategy_recent_losses[s])
            # Check breach
            try:
                from config import (
                    PER_STRATEGY_DAILY_R_CAP_ENABLED,
                    PER_STRATEGY_DAILY_R_KILL,
                    STRATEGY_KILL_CONSEC_LOSSES,
                    STRATEGY_KILL_CONSEC_WINDOW_HRS,
                    STRATEGY_KILL_AUTORESET_HRS,
                )
            except Exception:
                return
            if not PER_STRATEGY_DAILY_R_CAP_ENABLED:
                return
            threshold = float(PER_STRATEGY_DAILY_R_KILL.get(s, -3.0))
            tripped = False
            reason = ""
            if cur_r <= threshold:
                tripped = True
                reason = f"daily_r_sum={cur_r:.2f} <= {threshold:.2f}"
            # N-consec losses in window
            try:
                window_secs = float(STRATEGY_KILL_CONSEC_WINDOW_HRS) * 3600.0
                cutoff = time.time() - window_secs
                consec_in_window = sum(1 for t in recent_losses if t >= cutoff)
                if consec_in_window >= int(STRATEGY_KILL_CONSEC_LOSSES):
                    tripped = True
                    if reason:
                        reason += "; "
                    reason += (f"consec_losses={consec_in_window}"
                               f" >= {STRATEGY_KILL_CONSEC_LOSSES}"
                               f" in {STRATEGY_KILL_CONSEC_WINDOW_HRS}h")
            except Exception:
                pass
            if tripped and brain is not None and hasattr(brain, "_trip_strategy_kill"):
                try:
                    brain._trip_strategy_kill(s, reason, auto_reset_hrs=STRATEGY_KILL_AUTORESET_HRS)
                except Exception as e:
                    log.debug("record_strategy_r trip failed: %s", e)
        except Exception as e:
            log.debug("record_strategy_r failed (fail-open): %s", e)

    def get_daily_strategy_r(self) -> Dict[str, float]:
        """Snapshot of per-strategy daily R-sums (for dashboard / status)."""
        with self._lock:
            return dict(self._daily_strategy_r)

    # ──────────────────────────────────────────────
    #  ENTRY EVALUATION — the core intelligence gate
    # ──────────────────────────────────────────────

    def evaluate_entry(
        self,
        symbol: str,
        direction: str,
        score: float,
        regime: str,
        meta_prob: Optional[float],
        m15_dir: str,
        m5_dir: Optional[str] = None,
        m1_dir: Optional[str] = None,
        is_scalp: bool = False,
    ) -> dict:
        """Master evaluation of a potential trade.

        Returns
        -------
        dict
            {"approved": bool, "risk_pct": float, "reason": str, "confidence": float}

        Checks in order (any fail = reject):
        1. Is symbol blacklisted? (3 consecutive losses -> 24h ban)
        2. Score >= DRAGON_MIN_SCORE_BASELINE (7.0 swing, 6.5 scalp)?
        3. ML meta-label >= DRAGON_CONFIDENCE_FLOOR (0.65)?
        4. Cross-timeframe confluence (H1 + M15 must agree, M5 preferred)
        5. Symbol correlation (don't trade correlated pairs simultaneously)
        6. Equity curve health (if slope negative over last 20 trades, reduce risk)
        7. Recent performance (last 3 for this symbol all losses? -> reject)
        8. Daily trade count reasonable? (max ~6 trades per day across all symbols)
        9. Standby mode check (no favorable conditions for 4 hours -> skip)
        """
        result = {"approved": False, "risk_pct": 0.0, "reason": "", "confidence": 0.0}

        min_score = DRAGON_SCALP_MIN_SCORE if is_scalp else DRAGON_MIN_SCORE_BASELINE
        trade_type = "scalp" if is_scalp else "swing"

        # --- 0. Circuit breaker: 3 consecutive losses = pause 2 hours ---
        if self._session_paused:
            # Auto-reset after 2 hours
            if hasattr(self, '_pause_time') and (time.time() - self._pause_time) > 7200:
                self._session_paused = False
                self._session_losses = 0
                log.info("Circuit breaker RESET after 2h cooldown")
            else:
                result["reason"] = "circuit breaker — 3 consecutive losses (resets in %.0fh)" % (
                    (7200 - (time.time() - getattr(self, '_pause_time', time.time()))) / 3600)
                log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
                return result

        # --- 0b. Win cooldown REMOVED (2026-05-29 cooldown redesign) ---
        # This 30min BOTH-direction block shadowed brain._arm_cooldown's correct
        # 15min SAME-direction design → after any win both dirs were blocked,
        # killing legitimate opposite-side mean-reversion AND same-side
        # continuation compounding. Cooldowns are now single-source-of-truth in
        # the brain (R-scaled loss, POST_BIG_WIN, per-(sym,dir) attempt backoff).
        # MasterBrain stays portfolio-only per feedback_masterbrain_stripped.

        # --- 1. Blacklist check ---
        if self.is_symbol_blacklisted(symbol):
            result["reason"] = f"{symbol} blacklisted after {DRAGON_MAX_CONSECUTIVE_LOSSES} consecutive losses"
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result

        # --- 1b. Correlation-cluster cap (2026-06-05) ---
        # Prevent 8-correlated-indices-stacking. Industry rule: no more than
        # N positions per correlated cluster (US_INDICES, JPY_PAIRS, etc.).
        # Live evidence: EmergencyDD fired 13× in 14d, avg -5.29R = correlated
        # cluster events. FTMO/Topstep enforce equivalent VaR caps.
        cap_check = self._gate_correlation_cap(symbol)
        if cap_check is not None:
            result["reason"] = cap_check
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result

        # --- V5: Score, meta-label, and MTF checks REMOVED from MasterBrain ---
        # Brain's V5 pipeline already handles: score threshold, meta-label, MTF alignment
        # MasterBrain only keeps PORTFOLIO-LEVEL gates below (correlation, exposure, equity)
        mtf_confluence = 2  # default pass
        mtf_entry_quality = 50
        tf_agreement = "strong"

        # --- 5. Correlation cooldown: HARD BLOCK if cooldown active (2026-05-14) ---
        # Same fix as win-cooldown above. If a correlated symbol JUST CLOSED,
        # the cooldown is set to prevent immediate back-to-back correlated
        # entries (which compound risk). Warn-only allowed the bot to fire
        # right back into correlated exposure.
        now_ts = time.time()
        corr_cd_expiry = self._corr_cooldown.get(symbol, 0)
        if now_ts < corr_cd_expiry:
            mins_left = (corr_cd_expiry - now_ts) / 60
            result["reason"] = f"correlated cooldown {mins_left:.0f}min remaining"
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result
        # Open-correlated check stays warn-only (correlation ≠ identity)
        if self.get_correlated_exposure(symbol):
            log.warning("%s correlated symbol already open — entering anyway (no-skip rule)",
                        symbol)

        # --- 5b. Net directional exposure: warn at 3+ same-direction (no-skip rule) ---
        # User rule (2026-05-04): never skip trades on portfolio cap. Warn but enter.
        # Real risk control is upstream (per-symbol drift_detector multipliers + score gate).
        if self._check_net_directional(direction):
            log.warning("3+ positions already %s for %s — entering anyway (no-skip rule)",
                        direction, symbol)

        # --- 5c. Portfolio-level risk gate (VaR, concentration, correlation) ---
        portfolio_risk_mult = 1.0
        if self.portfolio_risk:
            try:
                pr = self.portfolio_risk.evaluate_portfolio_risk(symbol, direction)
                if not pr["approved"]:
                    result["reason"] = f"portfolio risk: {pr['reason']}"
                    log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
                    return result
                portfolio_risk_mult = pr.get("risk_multiplier", 1.0)
            except Exception as e:
                log.warning("Portfolio risk check failed (proceeding): %s", e)

        # --- 6. Equity curve health ---
        equity_slope = self.get_equity_slope()

        # --- 7. Recent performance --- V5: REMOVED (blacklist at 4 losses already handles this)
        # Was blocking good symbols after 3 losses. Brain's SL cooldown + blacklist are sufficient.

        # --- 8. Daily trade count (warn only, never block) ---
        self._maybe_reset_daily()
        if self._daily_trades >= _MAX_DAILY_TRADES:
            log.warning("WARN %s %s %s: daily trades %d >= %d — proceeding anyway",
                        trade_type, symbol, direction, self._daily_trades, _MAX_DAILY_TRADES)

        # --- 9. Standby check --- V5: REMOVED (was blocking after quiet Asian session)
        now = time.time()
        self._last_favorable_time[symbol] = now

        # --- All checks passed — calculate dynamic risk ---
        score_quality = max(0.0, min(1.0, (score - min_score) / 5.0))
        ml_quality = meta_prob if meta_prob is not None else 0.5
        # Use MTF intelligence for TF quality if available
        if self.mtf_intelligence and mtf_confluence > 0:
            tf_quality = min(1.0, mtf_entry_quality / 100.0)  # 0-1 from MTF quality
        else:
            tf_quality = 0.5 if self.mtf_intelligence else {"full": 1.0, "strong": 0.7, "none": 0.3}.get(tf_agreement, 0.3)
        if equity_slope > 0.01:
            equity_quality = 1.0
        elif equity_slope > -0.01:
            equity_quality = 0.5
        else:
            equity_quality = 0.3

        overall_confidence = (
            0.4 * score_quality
            + 0.3 * ml_quality
            + 0.2 * tf_quality
            + 0.1 * equity_quality
        )
        overall_confidence = max(0.0, min(1.0, overall_confidence))

        risk_pct = DRAGON_RISK_SCALE_MIN + (
            DRAGON_RISK_SCALE_MAX - DRAGON_RISK_SCALE_MIN
        ) * overall_confidence

        # ── DE-STACKED RISK ADJUSTMENTS (2026-05-14 audit fix) ──
        # Previous behaviour multiplied every adjustment independently:
        #   1.3 (equity_slope_up) × 1.3 (learning_boost) × 1.5 (momentum_boost)
        #   = 2.5× before the MAX_RISK cap, biasing upside compounding.
        # Now: collect all <1.0 adjustments and take their min (most cautious
        # protective scaler), separately collect >1.0 boosts and take only the
        # SINGLE strongest boost. Multiply: risk × min_protect × max_boost.
        protect_mults = []
        boost_mults = []

        if self._losing_day_yesterday:
            protect_mults.append(DRAGON_LOSS_DAY_RISK_MULT)
            log.info("De-stack: losing-day-yesterday mult=%.2f", DRAGON_LOSS_DAY_RISK_MULT)

        if equity_slope < -0.01:
            protect_mults.append(0.7)
            log.info("De-stack: negative equity slope %.4f → mult=0.70", equity_slope)
        elif equity_slope > 0.02:
            boost_mults.append(1.3)
            log.info("De-stack: positive equity slope %.4f → boost=1.30", equity_slope)

        # Adaptive risk from learning engine (per-symbol performance)
        if self.learning_engine:
            learn_mult = self.learning_engine.get_risk_multiplier(symbol)
            if learn_mult < 1.0:
                protect_mults.append(learn_mult)
                log.info("De-stack: learning %s mult=%.2f", symbol, learn_mult)
            elif learn_mult > 1.0:
                boost_mults.append(learn_mult)
                log.info("De-stack: learning %s boost=%.2f", symbol, learn_mult)

        # Drift-detector
        try:
            from agent import drift_detector
            drift_mult, drift_state = drift_detector.get_risk_multiplier(symbol)
            if drift_mult < 1.0:
                protect_mults.append(drift_mult)
                log.warning("De-stack: drift %s [%s] mult=%.2f", symbol, drift_state, drift_mult)
        except Exception as e:
            log.debug("drift_detector lookup failed for %s: %s", symbol, e)

        # Portfolio-level (concentration, correlation, VaR proximity) — always protective
        if portfolio_risk_mult < 1.0:
            protect_mults.append(portfolio_risk_mult)
            log.info("De-stack: portfolio mult=%.2f", portfolio_risk_mult)

        # ── 2026-06-18 Tier 1 #6: equity-curve 3-tier risk scaler ──
        # Compose into the de-stack chain so the worst protect wins. The
        # tier method honours EQUITY_TIER_SCALER_ENABLED — when False it
        # returns ("NEUTRAL", 1.0) and contributes nothing.
        try:
            intraday_dd = 0.0
            try:
                intraday_dd = float(getattr(self, "_intraday_dd_pct", 0.0) or 0.0)
            except Exception:
                intraday_dd = 0.0
            tier_name, tier_mult = self.compute_risk_tier(intraday_dd)
            if tier_mult < 1.0:
                protect_mults.append(float(tier_mult))
                log.info("De-stack: equity-tier [%s] mult=%.2f (7d R + DD)",
                         tier_name, tier_mult)
            elif tier_mult > 1.0:
                boost_mults.append(float(tier_mult))
                log.info("De-stack: equity-tier [%s] boost=%.2f", tier_name, tier_mult)
            # Stash for status/dashboard read.
            self._last_equity_tier = (tier_name, float(tier_mult))
        except Exception as _et_e:
            log.debug("equity-tier scaler skipped (fail-open): %s", _et_e)

        # Momentum size boost (gated)
        try:
            from config import MOMENTUM_SIZE_BOOST_ENABLED
            if MOMENTUM_SIZE_BOOST_ENABLED:
                from signals.momentum_signal import compute_momentum, size_multiplier
                ind = self.state.get_indicators(symbol) if self.state else {}
                df = self.state.get_candles(symbol, 60) if self.state else None
                mom = compute_momentum(ind or {}, df)
                mult = size_multiplier(mom, direction)
                if mult < 1.0:
                    protect_mults.append(mult)
                elif mult > 1.0:
                    boost_mults.append(mult)
                if mult != 1.0:
                    log.info("De-stack: momentum %s [%s/%s score=%.2f] %s=%.2f",
                             symbol, mom["regime"], mom["direction"], mom["score"],
                             "boost" if mult > 1.0 else "protect", mult)
        except Exception as e:
            log.debug("momentum size boost failed for %s: %s", symbol, e)

        # Apply: most-cautious protective + single strongest boost.
        applied_protect = min(protect_mults) if protect_mults else 1.0
        applied_boost = max(boost_mults) if boost_mults else 1.0
        risk_pct *= applied_protect * applied_boost
        log.info("De-stack final %s: protect=%.2f boost=%.2f → %.3f%%",
                 symbol, applied_protect, applied_boost, risk_pct)

        # Cap at max
        risk_pct = max(0.1, min(risk_pct, MAX_RISK_PER_TRADE_PCT))  # floor 0.1%, cap at max

        result["approved"] = True
        result["risk_pct"] = round(risk_pct, 4)
        result["reason"] = "approved"
        result["confidence"] = round(overall_confidence, 4)

        # Track daily trades
        with self._lock:
            self._daily_trades += 1

        log.info(
            "APPROVE %s %s %s | score=%.1f meta=%.3f tf=%s conf=%.3f risk=%.3f%%",
            trade_type, symbol, direction,
            score,
            meta_prob if meta_prob is not None else -1.0,
            tf_agreement,
            overall_confidence,
            risk_pct,
        )
        return result

    # ──────────────────────────────────────────────
    #  TRADE RESULT RECORDING
    # ──────────────────────────────────────────────

    def record_trade_result(self, symbol: str, direction: str, pnl: float,
                            r_multiple: float = None, strategy: str = None,
                            brain=None):
        """Record trade outcome for tracking.

        2026-06-18: optional kwargs r_multiple + strategy allow per-strategy
        R-cap (item #6) to roll up. Old call sites (positional) still work —
        new tracking is fail-open. `brain` lets us trip the strategy_kill_switch
        row on threshold breach.
        """
        now = time.time()
        entry = {"symbol": symbol, "direction": direction, "pnl": pnl, "time": now}
        if r_multiple is not None:
            entry["r_multiple"] = float(r_multiple)
        if strategy is not None:
            entry["strategy"] = str(strategy)

        with self._lock:
            self._trade_history.append(entry)
            if len(self._trade_history) > _MAX_HISTORY:
                self._trade_history = self._trade_history[-_MAX_HISTORY:]

            self._daily_pnl += pnl

        # Per-strategy R-cap forwarding (fail-open)
        if strategy is not None and r_multiple is not None:
            try:
                self.record_strategy_r(strategy, r_multiple, brain=brain)
            except Exception as _ps_e:
                log.debug("record_strategy_r failed: %s", _ps_e)

        with self._lock:

            if pnl < 0:
                self._symbol_losses[symbol] = self._symbol_losses.get(symbol, 0) + 1
                consec = self._symbol_losses[symbol]
                if consec >= DRAGON_MAX_CONSECUTIVE_LOSSES:
                    expiry = now + DRAGON_BLACKLIST_HOURS * 3600
                    self._blacklisted[symbol] = expiry
                    log.warning(
                        "BLACKLIST %s for %dh after %d consecutive losses",
                        symbol, DRAGON_BLACKLIST_HOURS, consec,
                    )
                    # Feed blacklist event back to observer for session learning
                    if self.learning_engine:
                        try:
                            hour = datetime.now(timezone.utc).hour
                            self.learning_engine.record_bad_hour(symbol, hour, "blacklist")
                        except Exception:
                            pass
                # Session circuit breaker: 3 consecutive losses
                self._session_losses += 1
                if self._session_losses >= 3:
                    self._session_paused = True
                    self._pause_time = time.time()
                    log.warning("CIRCUIT BREAKER: 3 consecutive losses — pausing 2 hours")
                    # Feed circuit breaker to observer
                    if self.learning_engine:
                        try:
                            hour = datetime.now(timezone.utc).hour
                            self.learning_engine.record_bad_hour(symbol, hour, "circuit_breaker")
                        except Exception:
                            pass
            else:
                self._symbol_losses[symbol] = 0
                self._session_losses = 0  # win resets circuit breaker
                # 2026-05-29: win cooldown moved to brain._arm_cooldown (15min
                # same-dir, or POST_BIG_WIN 60min same-dir on big wins). No
                # MasterBrain win timer — it was a both-dir shadow that blocked
                # legitimate re-entries.

            # 2026-05-29 audit fix: persist streak/blacklist so a restart doesn't
            # wipe the loss-streak protection.
            try:
                self.state.update_agent("mb_symbol_losses", dict(self._symbol_losses))
                self.state.update_agent("mb_blacklisted", dict(self._blacklisted))
            except Exception:
                pass

        # Set correlated cooldown so correlated symbols wait 30min
        self.set_correlated_cooldown(symbol)

        log.info(
            "RECORD %s %s pnl=%.2f | consec_losses=%d",
            symbol, direction, pnl, self._symbol_losses.get(symbol, 0),
        )

    # ──────────────────────────────────────────────
    #  BLACKLIST
    # ──────────────────────────────────────────────

    def is_symbol_blacklisted(self, symbol: str) -> bool:
        """Check if symbol is temporarily banned."""
        with self._lock:
            expiry = self._blacklisted.get(symbol)
            if expiry is None:
                return False
            if time.time() >= expiry:
                del self._blacklisted[symbol]
                self._symbol_losses[symbol] = 0
                log.info("UNBLACKLIST %s — ban expired", symbol)
                return False
            return True

    # ──────────────────────────────────────────────
    #  EQUITY SLOPE
    # ──────────────────────────────────────────────

    def get_equity_slope(self) -> float:
        """Calculate equity curve slope over last N trades. Positive = healthy.

        Uses simple linear regression (slope of cumulative PnL over trade index).
        Returns 0.0 if insufficient history.
        """
        with self._lock:
            recent = self._trade_history[-DRAGON_EQUITY_SLOPE_WINDOW:]

        n = len(recent)
        if n < 3:
            return 0.0

        # Cumulative PnL
        cum_pnl = []
        running = 0.0
        for t in recent:
            running += t["pnl"]
            cum_pnl.append(running)

        # Simple linear regression: slope of y=cum_pnl vs x=index
        x_mean = (n - 1) / 2.0
        y_mean = sum(cum_pnl) / n
        num = 0.0
        den = 0.0
        for i, y in enumerate(cum_pnl):
            num += (i - x_mean) * (y - y_mean)
            den += (i - x_mean) ** 2

        if den == 0:
            return 0.0
        return num / den

    # ──────────────────────────────────────────────
    #  CORRELATION CHECK
    # ──────────────────────────────────────────────

    def get_correlated_exposure(self, symbol: str) -> bool:
        """Check if we already have a position in a correlated symbol,
        OR if a correlated symbol closed recently (30-min cooldown)."""
        now = time.time()

        # Check sequential cooldown first (prevents back-to-back correlated entries)
        cooldown_expiry = self._corr_cooldown.get(symbol, 0)
        if now < cooldown_expiry:
            mins_left = (cooldown_expiry - now) / 60
            log.info("Correlated cooldown: %s blocked — %.0fmin remaining", symbol, mins_left)
            return True

        # Get currently open symbols from state
        open_symbols = set()
        try:
            positions = self.state.get_agent_state().get("positions", [])
            for pos in positions.values() if isinstance(positions, dict) else positions:
                sym = pos.get("symbol", "") if isinstance(pos, dict) else getattr(pos, "symbol", "")
                if sym:
                    open_symbols.add(sym)
        except Exception:
            return False

        if not open_symbols:
            return False

        for (sym_a, sym_b), corr in CORRELATION_PAIRS.items():
            if symbol == sym_a and sym_b in open_symbols:
                log.info(
                    "Correlated: %s blocked — %s open (corr=%.2f)",
                    symbol, sym_b, corr,
                )
                return True
            if symbol == sym_b and sym_a in open_symbols:
                log.info(
                    "Correlated: %s blocked — %s open (corr=%.2f)",
                    symbol, sym_a, corr,
                )
                return True

        return False

    def set_correlated_cooldown(self, closed_symbol: str):
        """When a position closes, set 30-min cooldown on correlated symbols."""
        now = time.time()
        cooldown_secs = 1800  # 30 minutes
        for (sym_a, sym_b), corr in CORRELATION_PAIRS.items():
            if closed_symbol == sym_a:
                self._corr_cooldown[sym_b] = now + cooldown_secs
                log.info("Correlated cooldown SET: %s blocked for 30min after %s closed", sym_b, closed_symbol)
            elif closed_symbol == sym_b:
                self._corr_cooldown[sym_a] = now + cooldown_secs
                log.info("Correlated cooldown SET: %s blocked for 30min after %s closed", sym_a, closed_symbol)

    # ──────────────────────────────────────────────
    #  NET DIRECTIONAL EXPOSURE
    # ──────────────────────────────────────────────

    def _check_net_directional(self, direction: str) -> bool:
        """Reject if 3+ positions already in same direction (portfolio imbalance)."""
        try:
            positions = self.state.get_agent_state().get("positions", [])
            pos_list = positions.values() if isinstance(positions, dict) else positions
            target = "BUY" if direction == "LONG" else "SELL"
            same_dir = sum(1 for p in pos_list
                          if (p.get("type", "") if isinstance(p, dict) else "") == target)
            return same_dir >= 3
        except Exception:
            return False

    # ──────────────────────────────────────────────
    #  CORRELATION-CLUSTER CAP (2026-06-05)
    # ──────────────────────────────────────────────

    def _gate_correlation_cap(self, symbol: str) -> Optional[str]:
        """Reject if the candidate symbol's correlated cluster already has
        CORRELATION_CAP_PER_CLUSTER open positions.

        Returns
        -------
        Optional[str]
            None  → pass (allow trade).
            str   → reject reason ("CORRELATION_CAP[...]").

        Research: industry rule = no hedge above 80% correl; FTMO/Topstep
        enforce VaR caps. Live evidence: EmergencyDD fires 13× in 14d, avg
        -5.29R = correlated cluster events (8 indices ≠ 8 uncorrelated bets).
        """
        if not CORRELATION_CAP_ENABLED:
            return None
        if not CORRELATION_CLUSTERS:
            return None

        try:
            # Find which cluster (if any) this symbol belongs to.
            # CORRELATION_CLUSTERS = {"US_INDICES": {...}, "JPY_PAIRS": {...}, ...}
            # Members may be a set/list/dict — use the `in` operator which works
            # for all three.
            for cluster_name, cluster_members in CORRELATION_CLUSTERS.items():
                if symbol not in cluster_members:
                    continue

                # Enumerate currently open positions via state (same pattern
                # used by get_correlated_exposure / _check_net_directional).
                open_symbols: List[str] = []
                try:
                    positions = self.state.get_agent_state().get("positions", [])
                    pos_iter = positions.values() if isinstance(positions, dict) else positions
                    for pos in pos_iter:
                        sym = (pos.get("symbol", "")
                               if isinstance(pos, dict)
                               else getattr(pos, "symbol", ""))
                        if sym:
                            open_symbols.append(sym)
                except Exception as e:
                    log.warning("Correlation-cap: position fetch failed: %s", e)
                    return None  # fail-open: don't block on lookup error

                open_in_cluster = sum(1 for s in open_symbols if s in cluster_members)
                if open_in_cluster >= CORRELATION_CAP_PER_CLUSTER:
                    log.warning(
                        "[%s] CORRELATION-CAP: cluster=%s already has %d open (max %d)",
                        symbol, cluster_name, open_in_cluster, CORRELATION_CAP_PER_CLUSTER,
                    )
                    return (
                        f"CORRELATION_CAP[{cluster_name}]: "
                        f"{open_in_cluster}/{CORRELATION_CAP_PER_CLUSTER} open"
                    )
                # Symbol belongs to at most one cluster — stop scanning.
                break
        except Exception as e:
            log.warning("Correlation-cap check failed for %s: %s", symbol, e)
            return None  # fail-open

        return None

    # ──────────────────────────────────────────────
    #  SCALP / SWING PERMISSION
    # ──────────────────────────────────────────────

    def should_allow_scalp(self, symbol: str) -> bool:
        """Master brain decides if scalping is allowed right now.

        Rules:
        - Only if swing brain has no active losing position for this symbol.
        - Only during scalp session hours.
        - Symbol must not be blacklisted.
        """
        if self.is_symbol_blacklisted(symbol):
            return False

        now_utc = datetime.now(timezone.utc)
        if not (SCALP_SESSION_START <= now_utc.hour < SCALP_SESSION_END):
            return False

        # Check if swing has an active losing position for this symbol
        try:
            positions = self.state.get_agent_state().get("positions", [])
            for pos in positions.values() if isinstance(positions, dict) else positions:
                sym = pos.get("symbol", "") if isinstance(pos, dict) else getattr(pos, "symbol", "")
                pnl = pos.get("pnl", 0) if isinstance(pos, dict) else getattr(pos, "pnl", 0)
                if sym == symbol and pnl < 0:
                    log.info("Scalp blocked for %s — swing position in loss", symbol)
                    return False
        except Exception:
            pass

        return True

    def should_allow_swing(self, symbol: str) -> bool:
        """Master brain decides if swing trading is allowed.

        Rules:
        - Symbol must not be blacklisted.
        - Must be within session hours (crypto exempt).
        - Max positions not exceeded.
        """
        if self.is_symbol_blacklisted(symbol):
            return False

        now_utc = datetime.now(timezone.utc)
        sym_cfg = SYMBOLS.get(symbol)
        if sym_cfg and sym_cfg.category != "Crypto":
            sess_start, sess_end = SYMBOL_SESSION_OVERRIDE.get(
                symbol, (SESSION_START_UTC, SESSION_END_UTC))
            if not (sess_start <= now_utc.hour < sess_end):
                return False

        # User rule (2026-05-04): never skip trades on position count.
        # Warn when above MAX_POSITIONS but allow the entry through.
        try:
            positions = self.state.get_agent_state().get("positions", [])
            count = len(positions) if isinstance(positions, dict) else len(list(positions))
            if count >= MAX_POSITIONS:
                log.warning("Position count %d >= MAX_POSITIONS %d — entering anyway (no-skip rule)",
                            count, MAX_POSITIONS)
        except Exception:
            pass

        return True

    # ──────────────────────────────────────────────
    #  DASHBOARD STATUS
    # ──────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return status for dashboard: confidence, mode, blacklisted symbols, etc."""
        with self._lock:
            equity_slope = self.get_equity_slope()
            now = time.time()
            blacklisted = {
                sym: round((exp - now) / 3600, 1)
                for sym, exp in self._blacklisted.items()
                if exp > now
            }
            recent_pnl = sum(t["pnl"] for t in self._trade_history[-10:])
            total_trades = len(self._trade_history)
            wins = sum(1 for t in self._trade_history if t["pnl"] > 0)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # Portfolio risk summary
        portfolio_summary = {}
        if self.portfolio_risk:
            try:
                portfolio_summary = self.portfolio_risk.get_status()
            except Exception:
                pass

        return {
            "equity_slope": round(equity_slope, 4),
            "equity_health": "healthy" if equity_slope > 0.01 else "flat" if equity_slope > -0.01 else "declining",
            "blacklisted_symbols": blacklisted,
            "symbol_losses": dict(self._symbol_losses),
            "daily_trades": self._daily_trades,
            "daily_pnl": round(self._daily_pnl, 2),
            "losing_day_yesterday": self._losing_day_yesterday,
            "recent_10_pnl": round(recent_pnl, 2),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "session_paused": self._session_paused,
            "session_losses": self._session_losses,
            "portfolio_risk": portfolio_summary,
        }

    # ──────────────────────────────────────────────
    #  DAILY RESET
    # ──────────────────────────────────────────────

    def reset_daily(self):
        """Reset daily counters. Called by run loop at midnight UTC."""
        with self._lock:
            self._losing_day_yesterday = self._daily_pnl < 0
            log.info(
                "DAILY RESET | pnl=%.2f losing=%s trades=%d strat_r=%s",
                self._daily_pnl, self._losing_day_yesterday, self._daily_trades,
                self._daily_strategy_r,
            )
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._session_losses = 0
            self._session_paused = False
            # 2026-06-18 Tier 1 #6: per-strategy daily R counters reset
            for s in self._daily_strategy_r:
                self._daily_strategy_r[s] = 0.0
            # Recent-loss tracking keeps a 6h rolling window — don't wipe.
            now_utc = datetime.now(timezone.utc)
            self._last_day = now_utc.timetuple().tm_yday

    # ──────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────

    def _maybe_reset_daily(self):
        """Auto-reset daily counters if the day changed."""
        now_utc = datetime.now(timezone.utc)
        today = now_utc.timetuple().tm_yday
        if self._last_day is None:
            self._last_day = today
        elif self._last_day != today:
            self.reset_daily()

    @staticmethod
    def _calc_tf_agreement(
        h1_dir: str, m15_dir: str, m5_dir: Optional[str], m1_dir: Optional[str]
    ) -> str:
        """Determine timeframe agreement level.

        Returns
        -------
        str
            "full" — all provided TFs agree.
            "strong" — H1 + M15 agree (M5/M1 may differ or be absent).
            "none" — H1 and M15 disagree.
        """
        if not m15_dir or m15_dir == "FLAT":
            return "strong"  # M15 neutral/absent — H1 leads, don't block

        if h1_dir != m15_dir:
            return "none"  # M15 actively opposes H1

        # H1 and M15 agree — check deeper TFs
        extras = [d for d in (m5_dir, m1_dir) if d is not None]
        if extras and all(d == h1_dir for d in extras):
            return "full"

        return "strong"
