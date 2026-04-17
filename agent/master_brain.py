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
)

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
        self.meta_model = meta_model

        self._lock = threading.RLock()

        # Track recent trades per symbol: list of {"symbol", "direction", "pnl", "time"}
        self._trade_history: List[dict] = []  # last 100 trades
        self._symbol_losses: Dict[str, int] = {}  # symbol -> consecutive loss count
        self._blacklisted: Dict[str, float] = {}  # symbol -> blacklist_expiry (unix ts)
        self._last_favorable_time: Dict[str, float] = {}  # symbol -> unix ts
        self._daily_trades: int = 0
        self._daily_pnl: float = 0.0
        self._last_day: Optional[int] = None  # day-of-year
        self._losing_day_yesterday: bool = False

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

        # --- 1. Blacklist check ---
        if self.is_symbol_blacklisted(symbol):
            result["reason"] = f"{symbol} blacklisted after {DRAGON_MAX_CONSECUTIVE_LOSSES} consecutive losses"
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result

        # --- 2. Score check ---
        if score < min_score:
            result["reason"] = f"score {score:.1f} < min {min_score:.1f}"
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result

        # --- 3. ML meta-label check ---
        if meta_prob is not None and meta_prob < DRAGON_CONFIDENCE_FLOOR:
            result["reason"] = f"meta_prob {meta_prob:.3f} < floor {DRAGON_CONFIDENCE_FLOOR}"
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result

        # --- 4. Cross-timeframe confluence ---
        # H1 direction is implied by `direction` (caller's regime/signal from H1)
        # M15 must agree
        h1_dir = direction  # the signal direction comes from H1 analysis
        tf_agreement = self._calc_tf_agreement(h1_dir, m15_dir, m5_dir, m1_dir)
        if tf_agreement == "none":
            result["reason"] = f"no TF confluence: H1={h1_dir} M15={m15_dir} M5={m5_dir}"
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result

        # --- 5. Correlation check ---
        if self.get_correlated_exposure(symbol):
            result["reason"] = f"correlated symbol already has open position"
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result

        # --- 6. Equity curve health ---
        equity_slope = self.get_equity_slope()

        # --- 7. Recent performance for this symbol ---
        with self._lock:
            recent_sym = [t for t in self._trade_history if t["symbol"] == symbol][-3:]
        if len(recent_sym) >= 3 and all(t["pnl"] < 0 for t in recent_sym):
            result["reason"] = f"last 3 trades for {symbol} all losses"
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result

        # --- 8. Daily trade count (warn only, never block) ---
        self._maybe_reset_daily()
        if self._daily_trades >= _MAX_DAILY_TRADES:
            log.warning("WARN %s %s %s: daily trades %d >= %d — proceeding anyway",
                        trade_type, symbol, direction, self._daily_trades, _MAX_DAILY_TRADES)

        # --- 9. Standby check ---
        now = time.time()
        last_fav = self._last_favorable_time.get(symbol, now)
        hours_since_favorable = (now - last_fav) / 3600.0
        if hours_since_favorable > DRAGON_STANDBY_HOURS and last_fav != now:
            result["reason"] = f"{symbol} in standby — {hours_since_favorable:.1f}h since last favorable"
            log.info("REJECT %s %s %s: %s", trade_type, symbol, direction, result["reason"])
            return result

        # Mark conditions as favorable since we passed all checks
        self._last_favorable_time[symbol] = now

        # --- All checks passed — calculate dynamic risk ---
        score_quality = max(0.0, min(1.0, (score - min_score) / 5.0))
        ml_quality = meta_prob if meta_prob is not None else 0.5
        tf_quality = {"full": 1.0, "strong": 0.7, "none": 0.3}.get(tf_agreement, 0.3)
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

        # Adjustments
        if self._losing_day_yesterday:
            risk_pct *= DRAGON_LOSS_DAY_RISK_MULT
            log.info("Risk halved (yesterday was losing day): %.3f%%", risk_pct)

        if equity_slope < -0.01:
            risk_pct *= 0.7
            log.info("Risk reduced 30%% (negative equity slope %.4f): %.3f%%", equity_slope, risk_pct)

        # Cap at max
        risk_pct = min(risk_pct, MAX_RISK_PER_TRADE_PCT)

        result["approved"] = True
        result["risk_pct"] = round(risk_pct, 4)
        result["reason"] = "approved"
        result["confidence"] = round(overall_confidence, 4)

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

    def record_trade_result(self, symbol: str, direction: str, pnl: float):
        """Record trade outcome for tracking."""
        now = time.time()
        entry = {"symbol": symbol, "direction": direction, "pnl": pnl, "time": now}

        with self._lock:
            self._trade_history.append(entry)
            if len(self._trade_history) > _MAX_HISTORY:
                self._trade_history = self._trade_history[-_MAX_HISTORY:]

            self._daily_pnl += pnl

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
            else:
                self._symbol_losses[symbol] = 0

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
        """Check if we already have a position in a correlated symbol."""
        # Get currently open symbols from state
        open_symbols = set()
        try:
            positions = self.state.get("positions", {})
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
            positions = self.state.get("positions", {})
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
            if not (SESSION_START_UTC <= now_utc.hour < SESSION_END_UTC):
                return False

        # Check max positions
        try:
            positions = self.state.get("positions", {})
            count = len(positions) if isinstance(positions, dict) else len(list(positions))
            if count >= MAX_POSITIONS:
                log.info("Swing blocked for %s — max positions %d reached", symbol, MAX_POSITIONS)
                return False
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
        }

    # ──────────────────────────────────────────────
    #  DAILY RESET
    # ──────────────────────────────────────────────

    def reset_daily(self):
        """Reset daily counters. Called by run loop at midnight UTC."""
        with self._lock:
            self._losing_day_yesterday = self._daily_pnl < 0
            log.info(
                "DAILY RESET | pnl=%.2f losing=%s trades=%d",
                self._daily_pnl, self._losing_day_yesterday, self._daily_trades,
            )
            self._daily_pnl = 0.0
            self._daily_trades = 0
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
        if not m15_dir or h1_dir != m15_dir:
            return "none"

        # H1 and M15 agree — check deeper TFs
        extras = [d for d in (m5_dir, m1_dir) if d is not None]
        if extras and all(d == h1_dir for d in extras):
            return "full"

        return "strong"
