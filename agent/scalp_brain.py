"""
Dragon Trader — Scalp Brain (M5 Scalper).

Parallel scalp decision loop (500ms cycle):
  1. Score M5 candles using scalp params (EMA 8/21/50, ST 1.5/7, MACD 5/13/4)
  2. M1 micro-direction for entry timing (EMA3 vs EMA8)
  3. H1 bias filter for direction alignment
  4. Session: 13-17 UTC only (London/NY overlap)
  5. Max 2 scalps per symbol per session
  6. Risk: MasterBrain-scaled (fallback 0.2%), SL: 1.5x ATR(M5), TP: 2R hard target
  7. Scalp trailing: BE@0.5R, lock@1R, trail@1.5R, trail@2R
  8. Uses separate magic numbers (base + 100 offset)
  9. MasterBrain gate: evaluate_entry() with is_scalp=True for approval + risk scaling
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
    SYMBOLS,
    SCALP_ENABLED, SCALP_RISK_PCT, SCALP_ATR_MULT,
    SCALP_MAGIC_OFFSET, SCALP_SESSION_START, SCALP_SESSION_END,
    SCALP_MAX_PER_SESSION, SCALP_TRAIL_STEPS,
    DD_EMERGENCY_CLOSE, STARTING_BALANCE,
    DRAGON_SCALP_MIN_SCORE,
)
from data.tick_streamer import SharedState
from execution.executor import Executor
from signals.scalp_scorer import (
    scalp_compute_indicators, scalp_score, _m1_micro_direction,
    SCALP_IND_DEFAULTS, SCALP_IND_OVERRIDES, MIN_SCALP_SCORE,
)
from signals.momentum_scorer import _ema, _supertrend

log = logging.getLogger("dragon.scalp_brain")

# ═══ CONSTANTS ═══
SCALP_CYCLE_S = 0.5              # 500ms decision cycle
M5_MIN_BARS = 100                # minimum M5 bars for scoring
H1_MIN_BARS = 50                 # minimum H1 bars for bias


class ScalpBrain:
    """M5 scalper: fast-cycle scoring + M1 entry timing + H1 direction filter + MasterBrain gating."""

    def __init__(self, state: SharedState, mt5, executor: Executor,
                 master_brain=None):
        """
        Args:
            state: SharedState from tick_streamer (thread-safe).
            mt5: MT5 connection (rpyc bridge).
            executor: Executor for order management.
            master_brain: Optional MasterBrain for entry gating + risk scaling.
                          If None, falls back to fixed SCALP_RISK_PCT.
        """
        self.state = state
        self.mt5 = mt5
        self.executor = executor
        self._master_brain = master_brain
        self.running = False
        self._thread = None
        self._cycle = int(0)

        # Session scalp counters: {symbol: count} — reset each session
        self._session_counts = {}
        self._last_session_day = None

        # Indicator cache for M5
        self._ind_cache = {}       # symbol -> (indicators_dict, timestamp)
        self._ind_cache_ttl = 2.0  # seconds before recompute

        # Trade log for dashboard
        self._trade_log = []

    # ═══════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════════════

    def start(self):
        """Start the scalp brain in a background thread."""
        if not SCALP_ENABLED:
            log.info("Scalp brain disabled (SCALP_ENABLED=False)")
            return

        self.running = True
        self.state.update_agent("scalp_running", True)
        self._thread = threading.Thread(
            target=self._decision_loop, daemon=True, name="ScalpBrain")
        self._thread.start()
        log.info("Dragon scalp brain started (cycle=%.1fs, session=%d-%d UTC, min_score=%.1f, master_brain=%s)",
                 SCALP_CYCLE_S, SCALP_SESSION_START, SCALP_SESSION_END,
                 DRAGON_SCALP_MIN_SCORE,
                 "ENABLED" if self._master_brain else "DISABLED")

    def stop(self):
        """Stop the scalp brain."""
        self.running = False
        self.state.update_agent("scalp_running", False)
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Dragon scalp brain stopped after %d cycles", self._cycle)

    # ═══════════════════════════════════════════════════════════════
    #  MAIN DECISION LOOP (500ms)
    # ═══════════════════════════════════════════════════════════════

    def _decision_loop(self):
        """Main loop: every 500ms, evaluate all symbols for scalp entries."""
        while self.running:
            loop_start = time.time()
            self._cycle += 1

            try:
                self._run_cycle()
            except Exception as e:
                log.error("Dragon scalp cycle %d error: %s", self._cycle, e, exc_info=True)

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, SCALP_CYCLE_S - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _run_cycle(self):
        """Single scalp cycle: session check, DD check, process all symbols, manage trailing."""
        now_utc = datetime.now(timezone.utc)
        hour_utc = int(now_utc.hour)

        # ── Session filter: 13-17 UTC only ──
        if hour_utc < SCALP_SESSION_START or hour_utc >= SCALP_SESSION_END:
            return

        # ── Reset session counters on new day ──
        today = now_utc.date()
        if today != self._last_session_day:
            self._last_session_day = today
            self._session_counts = {}
            log.info("Dragon scalp session counters reset for new day")

        # ── DD emergency check ──
        agent = self.state.get_agent_state()
        dd_pct = float(agent.get("dd_pct", 0.0))
        if dd_pct >= DD_EMERGENCY_CLOSE:
            return  # swing brain handles emergency close

        # ═══ PROCESS EACH SYMBOL ═══
        scalp_scores = {}
        for symbol in SYMBOLS:
            try:
                result = self._process_symbol(symbol)
                if result:
                    scalp_scores[symbol] = result
            except Exception as e:
                log.error("[%s] Dragon scalp process error: %s", symbol, e, exc_info=True)

        # ═══ MANAGE SCALP TRAILING SL + TRACK CLOSES ═══
        if not hasattr(self, '_last_scalp_close'):
            self._last_scalp_close = {}
        if not hasattr(self, '_scalp_was_open'):
            self._scalp_was_open = set()
        for symbol in SYMBOLS:
            try:
                has_scalp = self.executor.has_scalp_position(symbol)
                if has_scalp:
                    self.executor.manage_trailing_sl(symbol)
                    self._scalp_was_open.add(symbol)
                elif symbol in self._scalp_was_open:
                    # Scalp just closed — record time for re-entry cooldown
                    self._last_scalp_close[symbol] = time.time()
                    self._scalp_was_open.discard(symbol)
                    log.info("[%s] Scalp closed — 30min re-entry cooldown set", symbol)
            except Exception as e:
                log.warning("[%s] Dragon scalp trail error: %s", symbol, e)

        # ═══ UPDATE DASHBOARD ═══
        self.state.update_agent("scalp_cycle", int(self._cycle))
        self.state.update_agent("scalp_scores", scalp_scores)
        self.state.update_agent("scalp_log", list(self._trade_log[-50:]))

    # ═══════════════════════════════════════════════════════════════
    #  SYMBOL PROCESSING
    # ═══════════════════════════════════════════════════════════════

    def _process_symbol(self, symbol):
        """Process one symbol through the scalp pipeline."""
        cfg = SYMBOLS[symbol]

        # ── Session count limit ──
        count = self._session_counts.get(symbol, 0)
        if count >= SCALP_MAX_PER_SESSION:
            return {"direction": "FLAT", "gate": "SESSION_LIMIT",
                    "count": int(count)}

        # ── SL cooldown: respect swing brain's 45-min cooldown after SL hit ──
        sl_cooldowns = self.state.get_agent_state().get("sl_cooldowns", {})
        sl_expiry = sl_cooldowns.get(symbol, 0)
        if time.time() < sl_expiry:
            mins_left = (sl_expiry - time.time()) / 60
            return {"direction": "FLAT", "gate": "SL_COOLDOWN",
                    "cooldown_mins": round(mins_left, 1)}

        # ── Scalp re-entry cooldown: 30 min after last scalp close on same symbol ──
        last_scalp_close = getattr(self, '_last_scalp_close', {}).get(symbol, 0)
        if time.time() - last_scalp_close < 1800:  # 30 min
            return {"direction": "FLAT", "gate": "SCALP_REENTRY_COOLDOWN"}

        # ── Already have any position (swing or scalp)? ──
        if self.executor.has_position(symbol):
            return {"direction": "HOLD", "gate": "HAS_SWING_POS"}
        if self.executor.has_scalp_position(symbol):
            return {"direction": "HOLD", "gate": "HAS_SCALP"}

        # ── Get M5 candles ──
        m5_df = self.state.get_candles(symbol, 5)
        if m5_df is None or len(m5_df) < M5_MIN_BARS:
            return {"direction": "FLAT", "gate": "NO_M5_DATA"}

        # ── Compute M5 scores ──
        ind = self._get_m5_indicators(symbol, m5_df)
        if ind is None:
            return None

        n = int(ind["n"])
        bi = n - 2  # completed bar
        if bi < 21 or np.isnan(ind["at"][bi]) or float(ind["at"][bi]) == 0.0:
            return {"direction": "FLAT", "gate": "INSUFFICIENT_IND"}

        long_score, short_score = scalp_score(ind, bi)
        long_score = float(long_score)
        short_score = float(short_score)
        atr_val = float(ind["at"][bi])

        # ── Determine direction from score (Dragon threshold) ──
        if long_score >= DRAGON_SCALP_MIN_SCORE and long_score >= short_score:
            direction = "LONG"
            raw_score = long_score
        elif short_score >= DRAGON_SCALP_MIN_SCORE and short_score > long_score:
            direction = "SHORT"
            raw_score = short_score
        else:
            return {"long_score": long_score, "short_score": short_score,
                    "direction": "FLAT", "gate": "BELOW_MIN_SCORE",
                    "atr": atr_val}

        # ── H1 bias filter: must agree with scalp direction ──
        h1_dir = self._get_h1_bias(symbol)
        if h1_dir != direction:
            log.debug("[%s] DRAGON SCALP: H1=%s != SCALP=%s — skip", symbol, h1_dir, direction)
            return {"long_score": long_score, "short_score": short_score,
                    "direction": direction, "gate": "H1_DISAGREE",
                    "h1_dir": h1_dir, "atr": atr_val}

        # ── M1 micro-direction: must confirm entry timing ──
        m1_dir = _m1_micro_direction(self.state, symbol)
        if m1_dir != direction:
            log.debug("[%s] DRAGON SCALP: M1=%s != SCALP=%s — wait for timing",
                      symbol, m1_dir, direction)
            return {"long_score": long_score, "short_score": short_score,
                    "direction": direction, "gate": "M1_TIMING",
                    "m1_dir": m1_dir, "h1_dir": h1_dir, "atr": atr_val}

        # ── Meta probability from swing brain if available ──
        meta_prob = None
        agent = self.state.get_agent_state()
        swing_scores = agent.get("scores", {})
        sym_scores = swing_scores.get(symbol, {})
        meta_prob = sym_scores.get("meta_prob")
        meta_str = "%.2f" % float(meta_prob) if meta_prob is not None else "N/A"

        # ── MASTER BRAIN GATE (Dragon: evaluate_entry with is_scalp=True) ──
        risk_pct = SCALP_RISK_PCT  # default if no MasterBrain
        master_approved = True
        master_info = {}

        if self._master_brain:
            try:
                entry_eval = self._master_brain.evaluate_entry(
                    symbol=symbol,
                    direction=direction,
                    score=raw_score,
                    regime="scalp",
                    meta_prob=meta_prob,
                    m15_dir=m1_dir or "FLAT",
                    is_scalp=True,
                )
                master_approved = bool(entry_eval.get("approved", True))
                risk_pct = float(entry_eval.get("risk_pct", SCALP_RISK_PCT))
                master_info = entry_eval

                if not master_approved:
                    reject_reason = entry_eval.get("reason", "master_brain_reject")
                    log.info("[%s] DRAGON SCALP: MasterBrain rejected (%s)", symbol, reject_reason)
                    return {"long_score": long_score, "short_score": short_score,
                            "direction": direction, "gate": "MASTER_REJECT",
                            "meta_prob": meta_prob, "atr": atr_val,
                            "master_reason": reject_reason}
            except Exception as e:
                log.warning("[%s] MasterBrain evaluate_entry (scalp) failed: %s — using default risk",
                            symbol, e)
                risk_pct = SCALP_RISK_PCT

        # ── EXECUTE SCALP TRADE ──
        success = self.executor.open_scalp_trade(symbol, direction, atr_val,
                                                  risk_pct=risk_pct)

        if success:
            self._session_counts[symbol] = count + 1
            # Get lot size from positions
            positions = self.executor.get_positions_info()
            lot_str = "?"
            scalp_magic = int(cfg.magic) + SCALP_MAGIC_OFFSET
            for p in positions:
                if p["symbol"] == symbol and int(p["magic"]) == scalp_magic:
                    lot_str = "%.2f" % float(p["volume"])
                    break

            log.info("DRAGON SCALP: %s %s score=%.1f | M1=%s | H1=%s | META=%s | RISK=%.2f%% | MASTER=%s | -> ENTER %s lots",
                     symbol, direction, raw_score, m1_dir, h1_dir, meta_str,
                     risk_pct,
                     "approved" if self._master_brain else "N/A",
                     lot_str)

            self._trade_log.append({
                "timestamp": str(datetime.now(timezone.utc).strftime("%H:%M:%S")),
                "symbol": str(symbol),
                "direction": str(direction).lower(),
                "score": float(round(raw_score, 1)),
                "action": "SCALP_ENTRY",
                "m1": str(m1_dir),
                "h1": str(h1_dir),
                "session_count": int(count + 1),
                "risk_pct": float(risk_pct),
            })

        return {"long_score": long_score, "short_score": short_score,
                "direction": direction,
                "gate": "ENTERED" if success else "EXEC_FAIL",
                "m1_dir": m1_dir, "h1_dir": h1_dir,
                "meta_prob": meta_prob, "atr": atr_val,
                "risk_pct": risk_pct,
                "master_info": master_info}

    # ═══════════════════════════════════════════════════════════════
    #  SCALP TRADE RESULT RECORDING
    # ═══════════════════════════════════════════════════════════════

    def record_scalp_close(self, symbol, reason="scalp_exit"):
        """Record a scalp trade result with MasterBrain when a scalp position closes."""
        if not self._master_brain:
            return

        try:
            positions = self.executor.get_positions_info()
            pnl = float(0.0)
            direction = "FLAT"
            cfg = SYMBOLS.get(symbol)
            scalp_magic = int(cfg.magic) + SCALP_MAGIC_OFFSET if cfg else 0
            for p in positions:
                if p["symbol"] == symbol and int(p.get("magic", 0)) == scalp_magic:
                    pnl = float(p.get("pnl", 0.0))
                    direction = str(p.get("direction", "FLAT")).upper()
                    break

            self._master_brain.record_trade_result(
                symbol=symbol,
                direction=direction,
                pnl=pnl,
            )
            log.info("[%s] MasterBrain recorded scalp result: pnl=%.2f, reason=%s",
                     symbol, pnl, reason)
        except Exception as e:
            log.warning("[%s] MasterBrain record_trade_result (scalp) failed: %s", symbol, e)

    # ═══════════════════════════════════════════════════════════════
    #  INDICATORS
    # ═══════════════════════════════════════════════════════════════

    def _get_m5_indicators(self, symbol, m5_df):
        """Compute or return cached M5 indicators."""
        now = time.time()
        cached = self._ind_cache.get(symbol)
        if cached:
            ind, ts = cached
            if now - ts < self._ind_cache_ttl:
                return ind

        icfg = dict(SCALP_IND_DEFAULTS)
        icfg.update(SCALP_IND_OVERRIDES.get(symbol, {}))

        try:
            ind = scalp_compute_indicators(m5_df, symbol)
            self._ind_cache[symbol] = (ind, now)
            return ind
        except Exception as e:
            log.warning("[%s] M5 indicator computation failed: %s", symbol, e)
            return None

    def _get_h1_bias(self, symbol):
        """
        Determine H1 directional bias using EMA(15) vs EMA(40) + SuperTrend.
        Returns "LONG", "SHORT", or "FLAT".
        """
        h1_df = self.state.get_candles(symbol, 60)
        if h1_df is None or len(h1_df) < H1_MIN_BARS:
            return "FLAT"

        try:
            close = h1_df["close"].values.astype(np.float64)
            high = h1_df["high"].values.astype(np.float64)
            low = h1_df["low"].values.astype(np.float64)
            n = len(close)

            ema_s = _ema(close, 15)
            ema_l = _ema(close, 40)
            _, st_dir = _supertrend(high.copy(), low.copy(), close, 2.5, 10)

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
            log.warning("[%s] H1 bias check failed: %s", symbol, e)
            return "FLAT"
