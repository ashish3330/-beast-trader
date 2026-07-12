"""
Dragon Trader — Order Executor.
Risk-based lot sizing, ATR-based SL, trailing SL, signal reversal handling.
Institutional-grade execution: slippage tracking, partial fill handling,
requote retry, execution quality metrics, smart spread checks.
All values cast to float() for rpyc bridge compatibility.
"""
import time
import logging
import threading
from pathlib import Path
from collections import deque

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SYMBOLS, symbol_cfg, strategy_of_magic, MAX_RISK_PER_TRADE_PCT, MAX_TOTAL_EXPOSURE_PCT,
    ATR_SL_MULTIPLIER, TRAIL_STEPS, SUB2_TRAIL_STEPS,
    SCALP_RISK_PCT, SCALP_ATR_MULT, SCALP_MAGIC_OFFSET, SCALP_TRAIL_STEPS,
    SYMBOL_ATR_SL_OVERRIDE, SYMBOL_TRAIL_OVERRIDE,
    REGIME_TRAIL_DEFAULTS, SYMBOL_REGIME_TRAIL_OVERRIDE,
    SMART_ENTRY_MODE,
    EXECUTOR_MIN_REENTRY_SECS,
)

# 2026-06-05: new exit-logic constants. Imported lazily/defensively so the
# module still loads if config rollout lags. Defaults match config.py spec.
try:
    from config import EARLY_EXIT_REQUIRE_BAR_CLOSE
except Exception:
    EARLY_EXIT_REQUIRE_BAR_CLOSE = True
try:
    from config import TIME_STOP_ENABLED, TIME_STOP_BARS, TIME_STOP_MIN_PEAK_R
except Exception:
    TIME_STOP_ENABLED = True
    TIME_STOP_BARS = 12
    TIME_STOP_MIN_PEAK_R = 0.3
try:
    from config import BOS_INVALIDATION_ENABLED
except Exception:
    BOS_INVALIDATION_ENABLED = True
try:
    from config import VWAP_GATE_SYMBOLS
except Exception:
    VWAP_GATE_SYMBOLS = set()

# M15 bar duration in seconds — used by bar-close guard and time-stop
_M15_BAR_SEC = 15 * 60
# Lookback (M15 bars) for entry swing-pivot detection in BOS invalidation
_BOS_SWING_LOOKBACK = 20
# Window for swing-pivot detection (strict less-than across +/- this many bars)
_BOS_SWING_WINDOW = 5

# ═══ EXECUTION QUALITY CONSTANTS ═══
REQUOTE_RETCODE = 10004
REQUOTE_MAX_RETRIES = 3
REQUOTE_DELAY_SEC = 0.1
# 2026-07-12: arm a market-closed lockout after this many CONSECUTIVE all-None
# opens for a symbol (weekend indices return None, not retcode 10018). Streak
# guard so a transient bridge blip still retries every cycle.
NONE_OPEN_LOCKOUT_STREAK = 3
NONE_OPEN_LOCKOUT_SEC = 900.0   # 15 min; re-arms if still closed on next probe
SLIPPAGE_HISTORY_SIZE = 20
SPREAD_SPIKE_MULTIPLIER = 2.0   # reject if spread > 2x signal-time spread
SPREAD_SPIKE_DELAY_SEC = 5.0

# ═══ 3-SUB POSITION ARCHITECTURE ═══
# "Scaled exit IS the edge" — validated by user's backtests
# Sub0: 50% lot @ TP1 (2R) — take quick profit
# Sub1: 30% lot @ TP2 (3R) — take more profit
# Sub2: 20% lot @ wide TP  — let trailing SL ride the trend
SUB_SPLITS = [0.50, 0.30, 0.20]
# 2026-05-02: Sub2 R-cap reduced 50.0 → 5.0. With new 1.5-2.0x ATR SL the
# old 50R TP was 50 × 15pts = 750pts away → never hit. Now 5R is reachable
# (5 × 15pts = 75pts on indices) so the runner sub actually captures profit
# instead of relying on trail-only exit.
SUB_TP_R = [2.0, 3.0, 5.0]
SUB_MAGIC_OFFSETS = [0, 1, 2]  # sub0=base, sub1=base+1, sub2=base+2

# 2026-05-12: adaptive R:R per signal quality (hard tune). High-conviction
# scores let runners extend; low-conviction lock fast at smaller R.
# Default SUB_TP_R [2,3,5] sits in the middle. Edges:
#   score ≥ 9 (mega):      [2.5, 4.0, 8.0]  — let runners run, wider final TP
#   score 7-9 (strong):    [2.0, 3.0, 5.0]  — current default
#   score 6-7 (marginal):  [1.5, 2.5, 4.0]  — capture small wins faster
def adaptive_sub_tp_r(score: float, symbol: str = None) -> list:
    # 2026-05-13: per-symbol override from auto_tuned.py SUB_TP_R_OVERRIDE_AUTO
    # If a symbol has a tuned TP ladder, use it (overrides score-adaptive).
    if symbol:
        try:
            import auto_tuned as _at  # type: ignore
            override = getattr(_at, "SUB_TP_R_OVERRIDE_AUTO", {})
            if symbol in override:
                return override[symbol]
        except Exception:
            pass
    if score is None:
        return SUB_TP_R
    if score >= 9.0:
        return [2.5, 4.0, 8.0]
    if score < 7.0:
        return [1.5, 2.5, 4.0]
    return SUB_TP_R

# Trend-following symbols: single position (big runners need full lot riding the trend)
# Backtest proved: BTCUSD PF 3.17 single vs 0.99 with 3-sub (kills the edge)
SINGLE_POSITION_SYMBOLS = {"BTCUSD"}

# 2026-05-12: per-symbol broker-side deviation tolerance (in points). When
# the broker can't fill within `deviation` points of requested price, it
# REQUOTEs and our retry logic gives it 3 tries. Old global 50 was 5.0
# price units on indices (huge!) — broker happily slipped JPN225ft 500pts
# (5.0 price units) past requested. Tighter per-symbol caps force a
# requote+retry chain instead of a bad fill.
_DEVIATION_PER_SYMBOL = {
    "JPN225ft":  5,      # was 50 — slipped 500pts at 06:30 Asian sess
    "SPI200.r":  5,
    "GAS-Cr":    5,
    "NG-Cr":     10,
    "COPPER-Cr": 10,
    "DJ30.r":    10,
    "SP500.r":   5,
    "NAS100.r":  10,
    "GER40.r":   10,
    "FRA40.r":   10,
    "UK100.r":   10,
    "US2000.r":  5,
    "HK50.r":    10,
    "SWI20.r":   10,
    "UKOUSD":    10,
}
_DEVIATION_DEFAULT = 15   # was 50 — covers forex majors (1.5 pips slippage cap)

def _get_deviation(symbol: str) -> int:
    return _DEVIATION_PER_SYMBOL.get(symbol, _DEVIATION_DEFAULT)


# Spread filter: max spread as multiple of ATR
MAX_SPREAD_ATR_RATIO = 0.3  # reject if spread > 30% of ATR (default)

# 2026-05-12: per-symbol tighter spread caps for wide-spread CFDs that
# slipped catastrophically (JPN225ft -500pts, DJ30.r -100pts). Live slippage
# on these symbols exceeded backtest assumptions. Halving the spread tolerance
# kills trades during the worst microstructure moments without changing
# tested symbols. Empty in current data; populated per-evidence.
MAX_SPREAD_ATR_RATIO_BY_SYMBOL = {
    "JPN225ft":  0.15,   # was 0.3 — 500pt slippage 2026-05-12 06:30 Asian sess
    "SPI200.r":  0.20,
    "GAS-Cr":    0.20,
    "NG-Cr":     0.20,
    "COPPER-Cr": 0.20,
    "DJ30.r":    0.20,   # 100pt slippage observed
}

# 2026-05-12: latency-based pre-trade skip. If the symbol's rolling-avg
# order_send latency exceeds this, the broker book is wobbly and fills
# slip. Skip the entry rather than take a likely-bad fill. Average over
# last N orders (SLIPPAGE_HISTORY_SIZE = 20).
MAX_AVG_LATENCY_MS = 1500.0
MIN_LATENCY_HISTORY = 3      # need at least this many trades to gate

# 2026-05-12: rolling slippage abandonment. If a symbol's recent avg
# slippage exceeds this fraction of typical SL distance, stop trading it
# until conditions improve. Live evidence: JPN225ft 500pt vs ~380pt SL.
MAX_AVG_SLIPPAGE_VS_SL_RATIO = 0.5
MIN_SLIPPAGE_HISTORY = 3

log = logging.getLogger("dragon.executor")


class Executor:
    """Handles order execution, lot sizing, trailing SL, and position management.
    Includes institutional-grade execution quality tracking."""

    def __init__(self, mt5, state):
        self.mt5 = mt5
        self.state = state
        self._lock = threading.RLock()  # protects all internal state
        self._closing = {}        # symbol -> True if close in progress
        # Restore entry tracking from SharedState (survives restarts)
        agent = state.get_agent_state() if state else {}
        self._entry_prices = dict(agent.get("entry_prices", {}))
        self._entry_sl_dist = dict(agent.get("entry_sl_dist", {}))
        # 2026-05-26 audit fix: persist directions/dollar_risk/peak_r so cooldown
        # logic doesn't mis-route post-restart (5/01 USDCAD SL drift was peak_r
        # resetting to 0 → trail moved adversely on resume).
        self._directions = dict(agent.get("directions", {}))
        self._entry_dollar_risk = dict(agent.get("entry_dollar_risk", {}))
        self._peak_profit_r = dict(agent.get("peak_profit_r", {}))
        if self._entry_prices:
            log.info("Restored entry tracking for %d symbols from state (dirs=%d risks=%d peaks=%d)",
                     len(self._entry_prices), len(self._directions),
                     len(self._entry_dollar_risk), len(self._peak_profit_r))

        # ── Execution quality tracking ──
        self._slippage_history = {}   # symbol -> deque(maxlen=20) of slippage in points
        self._exec_latencies = {}     # symbol -> deque(maxlen=20) of latency in ms
        self._fill_counts = {}        # symbol -> {"full": int, "partial": int, "total": int}
        self._total_orders = {}       # symbol -> int (total order_send attempts)

        # ── RL trail adjustments (set by brain/run.py) ──
        self._rl_trail_adj = {}       # symbol -> {lock_threshold_mult, be_threshold_mult, trail_tightness_mult}
        self._current_regime = {}     # symbol -> current regime ("trending"/"ranging"/"volatile"/"low_vol")

        # ── 2026-06-05 BOS invalidation tracking ──
        # ticket -> {"swing": price, "direction": "LONG"/"SHORT", "anchor_time": ts}
        # Populated lazily on first _apply_trail cycle after a position opens.
        # Cleared in close_position / has_position external-close path.
        self._entry_swings = {}

        # ── 2026-06-04 stale CLOSE intent queue ──
        # When close_position fails after all order_send retries, persist the
        # intent so it gets re-attempted next cycle. Fixes the 2026-06-04 02:31
        # CHFJPY PeakGiveback that took 3 hours to actually close after the
        # rpyc bridge stalled. In-memory dict + DB-backed for restart survival.
        self._close_intents = {}   # symbol -> {"comment": str, "queued_at": float, "attempts": int}
        self._init_close_intents_table()
        self._load_close_intents()

    def _init_close_intents_table(self):
        """Create the close_intents table in trade_journal.db if missing."""
        try:
            import sqlite3
            from pathlib import Path as _Path
            _db = _Path(__file__).resolve().parent.parent / "data" / "trade_journal.db"
            conn = sqlite3.connect(str(_db), timeout=5.0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS close_intents (
                    symbol TEXT PRIMARY KEY,
                    comment TEXT NOT NULL,
                    queued_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at REAL DEFAULT 0
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("close_intents table init failed: %s", e)

    def _load_close_intents(self):
        """Restore queued intents on startup so a crash mid-close doesn't lose state."""
        try:
            import sqlite3
            from pathlib import Path as _Path
            _db = _Path(__file__).resolve().parent.parent / "data" / "trade_journal.db"
            conn = sqlite3.connect(str(_db), timeout=5.0)
            rows = conn.execute("SELECT symbol, comment, queued_at, attempts FROM close_intents").fetchall()
            conn.close()
            for sym, comment, queued_at, attempts in rows:
                self._close_intents[sym] = {
                    "comment": comment, "queued_at": float(queued_at),
                    "attempts": int(attempts),
                }
            if rows:
                log.warning("Loaded %d pending close intents from DB: %s",
                            len(rows), [r[0] for r in rows])
        except Exception as e:
            log.debug("close_intents load failed: %s", e)

    def _queue_close_intent(self, symbol, comment):
        """Persist a CLOSE intent that failed at order_send level."""
        intent = self._close_intents.get(symbol, {
            "comment": comment, "queued_at": time.time(), "attempts": 0,
        })
        intent["attempts"] += 1
        intent["last_attempt_at"] = time.time()
        self._close_intents[symbol] = intent
        try:
            import sqlite3
            from pathlib import Path as _Path
            _db = _Path(__file__).resolve().parent.parent / "data" / "trade_journal.db"
            conn = sqlite3.connect(str(_db), timeout=5.0)
            conn.execute(
                "INSERT OR REPLACE INTO close_intents "
                "(symbol, comment, queued_at, attempts, last_attempt_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol, comment, intent["queued_at"], intent["attempts"],
                 intent.get("last_attempt_at", time.time()))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("close_intent persist failed: %s", e)
        log.warning("[%s] CLOSE_INTENT QUEUED: %s (attempts=%d) — will retry next cycle",
                    symbol, comment, intent["attempts"])

    def _clear_close_intent(self, symbol):
        """Called after position is confirmed flat — removes the queued intent."""
        if symbol not in self._close_intents:
            return
        self._close_intents.pop(symbol, None)
        try:
            import sqlite3
            from pathlib import Path as _Path
            _db = _Path(__file__).resolve().parent.parent / "data" / "trade_journal.db"
            conn = sqlite3.connect(str(_db), timeout=5.0)
            conn.execute("DELETE FROM close_intents WHERE symbol=?", (symbol,))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("close_intent clear failed: %s", e)

    def drain_close_intents(self):
        """Re-attempt all queued close intents. Called once per manage_trailing cycle.
        Caps at 10 attempts per intent (~10 min at 60s retry) to avoid wedging on
        a structurally-impossible close.
        """
        if not self._close_intents:
            return
        # Snapshot to allow safe mutation during iteration
        for symbol in list(self._close_intents.keys()):
            intent = self._close_intents.get(symbol)
            if intent is None:
                continue
            # Verify symbol still has an open position; if not, clear
            try:
                if not self.has_position(symbol):
                    log.info("[%s] CLOSE_INTENT cleared: position already flat", symbol)
                    self._clear_close_intent(symbol)
                    continue
            except Exception:
                pass
            # Cap retries
            if intent["attempts"] >= 10:
                log.error("[%s] CLOSE_INTENT GIVE-UP after %d attempts — manual intervention required",
                          symbol, intent["attempts"])
                self._clear_close_intent(symbol)
                continue
            # Re-attempt the close
            log.info("[%s] CLOSE_INTENT RETRY %d/10 (queued %.0fs ago)",
                     symbol, intent["attempts"] + 1,
                     time.time() - intent["queued_at"])
            try:
                self.close_position(symbol, intent["comment"])
            except Exception as e:
                log.warning("[%s] CLOSE_INTENT retry exception: %s", symbol, e)

    # ═══════════════════════════════════════════════════════════════════════
    # INSTITUTIONAL EXECUTION ENGINE
    # ═══════════════════════════════════════════════════════════════════════

    def _send_order(self, request, symbol, context=""):
        """
        Central order_send wrapper with:
        - Requote retry (retcode 10004, up to 3 attempts with fresh price)
        - Latency measurement
        - Slippage tracking (requested vs actual fill price)
        - Partial fill detection and logging
        Returns (result, actual_volume) or (None, 0.0).
        """
        requested_price = float(request.get("price", 0))
        requested_volume = float(request.get("volume", 0))
        is_close = "position" in request  # close orders have a position ticket
        RETRY_RETCODES = {10004, 10006, 10018}  # REQUOTE, CONNECTION_LOST, LOCKED

        for attempt in range(1, REQUOTE_MAX_RETRIES + 1):
            t0 = time.monotonic()
            try:
                result = self.mt5.order_send(request)
            except Exception as e:
                log.error("[%s] %s order_send EXCEPTION (attempt %d/%d): %s",
                          symbol, context, attempt, REQUOTE_MAX_RETRIES, e)
                if attempt < REQUOTE_MAX_RETRIES:
                    time.sleep(REQUOTE_DELAY_SEC * attempt)  # exponential backoff
                    continue
                return None, 0.0
            latency_ms = (time.monotonic() - t0) * 1000.0

            # Track latency
            if symbol not in self._exec_latencies:
                self._exec_latencies[symbol] = deque(maxlen=SLIPPAGE_HISTORY_SIZE)
            self._exec_latencies[symbol].append(latency_ms)

            if result is None:
                log.error("[%s] %s order_send returned None (attempt %d/%d)",
                          symbol, context, attempt, REQUOTE_MAX_RETRIES)
                if attempt < REQUOTE_MAX_RETRIES:
                    time.sleep(REQUOTE_DELAY_SEC)
                    continue
                # 2026-07-12: all retries exhausted with a None result. On an OPEN
                # this is almost always a CLOSED market (weekend indices return None,
                # not retcode 10018) — but it can also be a transient bridge blip, so
                # only arm the cooldown after N CONSECUTIVE all-None opens for this
                # symbol (a blip clears in 1-2 cycles; a closed market persists).
                # Stops the every-60s JPN225ft/NAS100.r retry spam on weekends.
                if not is_close:
                    if not hasattr(self, "_open_none_streak"):
                        self._open_none_streak = {}
                    self._open_none_streak[symbol] = self._open_none_streak.get(symbol, 0) + 1
                    if self._open_none_streak[symbol] >= NONE_OPEN_LOCKOUT_STREAK:
                        if not hasattr(self, "_market_closed_until"):
                            self._market_closed_until = {}
                        self._market_closed_until[symbol] = time.time() + NONE_OPEN_LOCKOUT_SEC
                        log.warning("[%s] %d consecutive None opens — entries locked %dm "
                                    "(market closed / bridge saturated)", symbol,
                                    self._open_none_streak[symbol], int(NONE_OPEN_LOCKOUT_SEC / 60))
                return None, 0.0

            retcode = int(result.retcode)

            # ── TRANSIENT ERROR RETRY (requote, connection lost, locked) ──
            if retcode in RETRY_RETCODES and attempt < REQUOTE_MAX_RETRIES:
                log.warning("[%s] %s REQUOTE (attempt %d/%d) — retrying in %dms",
                            symbol, context, attempt, REQUOTE_MAX_RETRIES,
                            int(REQUOTE_DELAY_SEC * 1000))
                time.sleep(REQUOTE_DELAY_SEC)
                # Refresh price for retry
                tick = self.mt5.symbol_info_tick(symbol)
                if tick is not None:
                    order_type = int(request.get("type", 0))
                    if is_close:
                        # Close: reverse of position type
                        fresh_price = float(tick.bid) if order_type == 1 else float(tick.ask)
                    else:
                        fresh_price = float(tick.ask) if order_type == 0 else float(tick.bid)
                    request["price"] = float(fresh_price)
                    requested_price = fresh_price
                continue

            # ── SUCCESS ──
            if retcode in (10009, 10008):
                # a fill proves the market is open → clear any None-open streak
                if getattr(self, "_open_none_streak", None):
                    self._open_none_streak[symbol] = 0
                # Track fill counts
                if symbol not in self._fill_counts:
                    self._fill_counts[symbol] = {"full": 0, "partial": 0, "total": 0}
                self._fill_counts[symbol]["total"] += 1

                # ── SLIPPAGE TRACKING ──
                actual_price = float(result.price) if hasattr(result, 'price') and result.price else requested_price
                si = self.mt5.symbol_info(symbol)
                point = float(si.point) if si and si.point else 0.00001
                slippage_points = (actual_price - requested_price) / point

                # For sells, positive slippage means we got worse price
                order_type = int(request.get("type", 0))
                if order_type == 1:  # SELL
                    slippage_points = -slippage_points  # normalize: positive = worse

                if symbol not in self._slippage_history:
                    self._slippage_history[symbol] = deque(maxlen=SLIPPAGE_HISTORY_SIZE)
                self._slippage_history[symbol].append(slippage_points)

                # Warn if slippage exceeds 1 ATR
                atr = self._get_atr(symbol)
                if atr > 0:
                    slippage_abs = abs(actual_price - requested_price)
                    if slippage_abs > atr:
                        log.warning("[%s] %s UNUSUAL SLIPPAGE: %.5f (%.1f points) > 1 ATR (%.5f) — "
                                    "requested=%.5f actual=%.5f",
                                    symbol, context, slippage_abs, slippage_points, atr,
                                    requested_price, actual_price)
                    elif abs(slippage_points) > 0.5:
                        log.info("[%s] %s slippage: %.1f points (req=%.5f fill=%.5f lat=%.0fms)",
                                 symbol, context, slippage_points, requested_price, actual_price, latency_ms)

                # ── PARTIAL FILL HANDLING ──
                actual_volume = float(result.volume) if hasattr(result, 'volume') and result.volume else requested_volume
                if abs(actual_volume - requested_volume) > 0.001:
                    self._fill_counts[symbol]["partial"] += 1
                    log.warning("[%s] %s PARTIAL FILL: requested=%.2f actual=%.2f (%.1f%%)",
                                symbol, context, requested_volume, actual_volume,
                                actual_volume / requested_volume * 100 if requested_volume > 0 else 0)
                else:
                    self._fill_counts[symbol]["full"] += 1
                    actual_volume = requested_volume  # treat as full

                return result, actual_volume

            # ── FINAL FAILURE ──
            if attempt == REQUOTE_MAX_RETRIES or retcode not in RETRY_RETCODES:
                log.error("[%s] %s order failed [%d]: %s (attempt %d/%d, lat=%.0fms)",
                          symbol, context, retcode,
                          result.comment if hasattr(result, 'comment') else "?",
                          attempt, REQUOTE_MAX_RETRIES, latency_ms)
                # 2026-05-14: retcode 10018 = MARKET_CLOSED. Arm a 1hr cooldown
                # to stop the brain from retrying every cycle until market reopens.
                if retcode == 10018 and not is_close:
                    if not hasattr(self, "_market_closed_until"):
                        self._market_closed_until = {}
                    self._market_closed_until[symbol] = time.time() + 3600.0
                    log.warning("[%s] MARKET CLOSED — entries locked for 1h", symbol)
                return result, 0.0

        return None, 0.0

    # ═══════════════════════════════════════════════════════════════
    #  TIER 1 #3 — SPREAD-BLOWOUT PRE-ORDER SKIP
    # ═══════════════════════════════════════════════════════════════

    def _check_spread_blowout(self, symbol):
        """Return True if live spread > SPREAD_BLOWOUT_MULT × baseline.

        Source of baseline: signals.fvg_strategy.SPREAD (the most up-to-date
        per-sym table in the repo, mirrored from the BT). Falls back to None
        → no-op when symbol unknown.

        Exempt from [[feedback_no_skip_trades]]: spread blowout is broker
        friction, not a quality scorer. Default OFF via SPREAD_BLOWOUT_HARD_SKIP
        so the first 24h logs would-have-blocked frequency without acting.

        Returns True = "blowout detected, caller must skip".
        Fail-open: any exception → False (allow trade through).
        """
        try:
            from config import SPREAD_BLOWOUT_HARD_SKIP, SPREAD_BLOWOUT_MULT
            if not SPREAD_BLOWOUT_HARD_SKIP:
                return False
        except Exception:
            return False
        try:
            info = self.mt5.symbol_info(symbol)
            if info is None:
                return False
            point = float(info.point) if info.point else 0.00001
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                return False
            live_spread_price = float(tick.ask) - float(tick.bid)
            if live_spread_price <= 0:
                return False
            # Per-sym baseline from the FVG SPREAD table (shared across modules)
            try:
                from agent.fvg_strategy import SPREAD as _SPREAD_BASELINE
            except Exception:
                _SPREAD_BASELINE = {}
            baseline = float(_SPREAD_BASELINE.get(symbol, 0.0))
            if baseline <= 0:
                return False
            threshold = baseline * float(SPREAD_BLOWOUT_MULT)
            if live_spread_price > threshold:
                log.warning(
                    "[%s] R_SPREAD_BLOWOUT: live=%.5f > threshold=%.5f "
                    "(baseline=%.5f × %.2f). Skipping entry.",
                    symbol, live_spread_price, threshold, baseline,
                    float(SPREAD_BLOWOUT_MULT))
                # Push to dashboard if available (lazy)
                try:
                    from dashboard import v2_api as _v2  # type: ignore
                    _v2.push_decision({
                        "ts": time.time(), "symbol": str(symbol),
                        "long_score": 0.0, "short_score": 0.0,
                        "direction": "FLAT", "gate": "R_SPREAD_BLOWOUT",
                        "gate_legacy": "SPREAD_BLOWOUT",
                        "reason": (f"live={live_spread_price:.5f} > "
                                   f"threshold={threshold:.5f}"),
                        "m15_dir": "N/A", "regime": "", "meta_prob": None,
                    })
                except Exception:
                    pass
                return True
            return False
        except Exception as e:
            log.debug("[%s] spread blowout check failed (fail-open): %s", symbol, e)
            return False

    # ═══════════════════════════════════════════════════════════════
    #  TIER 1 EXT — PER-STRATEGY CONCURRENT POSITION CAP
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _strategy_for_magic(magic_int):
        """Map magic offset to strategy bucket. Mirrors SETUP.txt §2:
            base       → momentum
            base+1000  → fvg     (FVG_MAGIC_OFFSET)
            base+2000  → sr      (SR_MAGIC_OFFSET)
            base+3000  → smabo   (SMABO_MAGIC_OFFSET, 2026-06-21)
            base+4000  → fib50   (FIB50_MAGIC_OFFSET, 2026-06-21)
        Falls back to "momentum" for unknown offsets (cheapest mismatch).
        """
        try:
            mod = int(magic_int) % 10000
            if 4000 <= mod < 5000:
                return "fib50"
            if 3000 <= mod < 4000:
                return "smabo"
            if 2000 <= mod < 3000:
                return "sr"
            if 1000 <= mod < 2000:
                return "fvg"
            return "momentum"
        except Exception:
            return "momentum"

    def _count_strategy_positions(self, strategy):
        """Count currently open positions belonging to a strategy bucket.
        Returns 0 on any exception (fail-open)."""
        try:
            positions = self.mt5.positions_get()
            if positions is None:
                return 0
            return sum(1 for p in positions
                       if self._strategy_for_magic(getattr(p, "magic", 0)) == strategy)
        except Exception:
            return 0

    def _check_per_strategy_cap(self, symbol, strategy):
        """Return True if the per-strategy cap is breached and caller should skip.

        Default OFF via PER_STRATEGY_CAP_HARD_REJECT — shadow logs "would-have-blocked"
        frequency for 48h before enforcement. Honours [[feedback_no_skip_trades]]
        by being an opt-in concurrency cap (not a quality scorer).
        """
        try:
            from config import (
                PER_STRATEGY_CAP_HARD_REJECT,
                MAX_CONCURRENT_PER_STRATEGY,
            )
        except Exception:
            return False
        try:
            cap = int(MAX_CONCURRENT_PER_STRATEGY.get(strategy, 999))
        except Exception:
            cap = 999
        n = self._count_strategy_positions(strategy)
        if n >= cap:
            if PER_STRATEGY_CAP_HARD_REJECT:
                log.warning(
                    "[%s] R_STRATEGY_CAP %s=%d/%d — HARD REJECT (PER_STRATEGY_CAP_HARD_REJECT=True)",
                    symbol, strategy, n, cap)
                return True
            else:
                log.info(
                    "[%s] R_STRATEGY_CAP_SHADOW %s=%d/%d (would block; flag off)",
                    symbol, strategy, n, cap)
                return False
        return False

    # ═══════════════════════════════════════════════════════════════
    #  TIER 1 #4 — POSITION R-MULTIPLE LIVE TELEMETRY
    # ═══════════════════════════════════════════════════════════════

    def _push_position_r_telemetry(self, symbol, profit_r, peak_r, status="open"):
        """Lazy dashboard push of (symbol, profit_r, peak_r) — rate-limited.

        Never blocks the exit-management loop. Default ON via
        POSITION_R_TELEMETRY_ENABLED but the dashboard ignores the event
        if it doesn't have a tile rendered (no breakage).
        """
        try:
            from config import (
                POSITION_R_TELEMETRY_ENABLED,
                POSITION_R_TELEMETRY_MIN_INTERVAL_SEC,
            )
            if not POSITION_R_TELEMETRY_ENABLED:
                return
        except Exception:
            return
        try:
            now = time.time()
            last = getattr(self, "_pos_r_last_push", None)
            if last is None:
                last = {}
                self._pos_r_last_push = last
            if now - float(last.get(symbol, 0.0)) < float(POSITION_R_TELEMETRY_MIN_INTERVAL_SEC):
                return
            last[symbol] = now
            try:
                from dashboard import v2_api as _v2  # type: ignore
                payload = {
                    "ts": now,
                    "symbol": str(symbol),
                    "profit_r": float(profit_r),
                    "peak_r": float(peak_r),
                    "status": str(status),
                }
                # Use push_position_event if no dedicated R push exists; the
                # dashboard buffer treats it as a position update.
                if hasattr(_v2, "push_position_r"):
                    _v2.push_position_r(payload)
                elif hasattr(_v2, "push_position_event"):
                    _v2.push_position_event("r_update", payload)
            except Exception:
                pass
        except Exception:
            pass

    def _check_spread_spike(self, symbol, signal_spread=None):
        """
        Smart spread check at execution time.
        If current spread > 2x the spread at signal generation, delay 5s and recheck.
        Returns (ok_to_trade: bool, current_tick).
        """
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            return False, None

        if signal_spread is None or signal_spread <= 0:
            return True, tick  # no signal spread to compare, proceed

        current_spread = float(tick.ask) - float(tick.bid)
        if current_spread <= 0:
            return True, tick

        ratio = current_spread / signal_spread
        if ratio <= SPREAD_SPIKE_MULTIPLIER:
            return True, tick

        log.warning("[%s] SPREAD SPIKE: current=%.5f vs signal=%.5f (%.1fx) — "
                    "delaying %.0fs and retrying",
                    symbol, current_spread, signal_spread, ratio, SPREAD_SPIKE_DELAY_SEC)
        time.sleep(SPREAD_SPIKE_DELAY_SEC)

        # Recheck after delay
        tick2 = self.mt5.symbol_info_tick(symbol)
        if tick2 is None:
            return False, None

        current_spread2 = float(tick2.ask) - float(tick2.bid)
        ratio2 = current_spread2 / signal_spread if signal_spread > 0 else 0
        if ratio2 > SPREAD_SPIKE_MULTIPLIER:
            log.warning("[%s] SPREAD STILL SPIKED after delay: %.5f (%.1fx) — proceeding with caution",
                        symbol, current_spread2, ratio2)
        else:
            log.info("[%s] Spread normalized: %.5f (%.1fx) — proceeding",
                     symbol, current_spread2, ratio2)
        return True, tick2  # proceed regardless (never skip trades, warn only)

    def get_execution_stats(self):
        """
        Expose execution quality metrics per symbol for dashboard.
        Returns dict: {symbol: {avg_slippage_pts, fill_rate_pct, avg_latency_ms, total_orders}}.
        """
        stats = {}
        all_symbols = set(list(self._slippage_history.keys()) +
                          list(self._exec_latencies.keys()) +
                          list(self._fill_counts.keys()))
        for sym in all_symbols:
            slip_hist = self._slippage_history.get(sym, deque())
            lat_hist = self._exec_latencies.get(sym, deque())
            fills = self._fill_counts.get(sym, {"full": 0, "partial": 0, "total": 0})

            avg_slip = float(np.mean(list(slip_hist))) if len(slip_hist) > 0 else 0.0
            avg_lat = float(np.mean(list(lat_hist))) if len(lat_hist) > 0 else 0.0
            fill_rate = (fills["full"] / fills["total"] * 100.0) if fills["total"] > 0 else 100.0

            stats[sym] = {
                "avg_slippage_pts": round(avg_slip, 2),
                "max_slippage_pts": round(float(max(slip_hist, key=abs)) if slip_hist else 0.0, 2),
                "fill_rate_pct": round(fill_rate, 1),
                "partial_fills": fills["partial"],
                "avg_latency_ms": round(avg_lat, 1),
                "total_orders": fills["total"],
                "last_20_slippages": list(slip_hist),
            }
        return stats

    def set_rl_trail_adjustments(self, symbol, adj):
        """Set RL-learned trail parameter adjustments for a symbol.
        adj: dict with keys lock_threshold_mult, be_threshold_mult, trail_tightness_mult."""
        self._rl_trail_adj[symbol] = adj

    def set_current_regime(self, symbol, regime):
        """Brain calls this per cycle so executor can pick regime-conditional
        trail profile from SYMBOL_REGIME_TRAIL_OVERRIDE / REGIME_TRAIL_DEFAULTS."""
        if regime:
            self._current_regime[symbol] = regime

    def _resolve_trail_steps(self, symbol):
        """Resolution order (most-specific first):
          1. SYMBOL_REGIME_TRAIL_OVERRIDE[symbol][current_regime]  (per-cell tune)
          2. SYMBOL_TRAIL_OVERRIDE[symbol]                          (agent-tuned per-symbol)
          3. REGIME_TRAIL_DEFAULTS[current_regime]                  (regime default)
          4. TRAIL_STEPS (global default)
        """
        regime = self._current_regime.get(symbol)
        if regime:
            cell = SYMBOL_REGIME_TRAIL_OVERRIDE.get(symbol, {}).get(regime)
            if cell:
                return cell
        sym_trail = SYMBOL_TRAIL_OVERRIDE.get(symbol)
        if sym_trail:
            return sym_trail
        if regime:
            reg_default = REGIME_TRAIL_DEFAULTS.get(regime)
            if reg_default:
                return reg_default
        return TRAIL_STEPS

    # ═══════════════════════════════════════════════════════════════════════

    def open_trade(self, symbol, direction, atr, risk_pct=None, signal_spread=None,
                   smart_tp=None, score=None, regime=None):
        """
        Open 3 sub-positions with scaled TPs (the proven edge).
        Sub0: 50% @ TP1 (2R or smart_tp) — quick profit lock
        Sub1: 30% @ TP2 (3R or 1.5x smart_tp) — more profit
        Sub2: 20% @ wide TP  — trailing SL rides the trend
        All share same SL = ATR_SL_MULTIPLIER * ATR.
        smart_tp: MTF-computed optimal TP distance (from liquidity + fibonacci).
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            log.error("[%s] Unknown symbol", symbol)
            return False

        if self.has_position(symbol):
            log.info("[%s] Already has position, skipping", symbol)
            return False

        # daily per-symbol profit lock (target hit → no new entries till EOD)
        if self._daily_loss_blocked(symbol):
            log.warning("[%s] open reject: daily loss limit reached — locked till EOD", symbol)
            return False

        # ── 2026-06-18 Tier 1 #3: spread-blowout pre-order skip ──
        # Default OFF via SPREAD_BLOWOUT_HARD_SKIP. When enabled, blocks
        # entry when live spread > N× SPREAD baseline. Exempt from
        # [[feedback_no_skip_trades]] per upgrade-v2 §3.5 (broker friction).
        if self._check_spread_blowout(symbol):
            return False

        # ── 2026-06-18 Tier 1 ext: per-strategy concurrent position cap ──
        # Default OFF via PER_STRATEGY_CAP_HARD_REJECT. Hard-reject on
        # MAX_CONCURRENT_PER_STRATEGY breach. Caller's strategy bucket
        # is inferred from cfg.magic offset.
        _strat = self._strategy_for_magic(int(cfg.magic))
        if self._check_per_strategy_cap(symbol, _strat):
            return False

        # ── 2026-06-18 Tier 1 #1: per-strategy kill switch gate ──
        # Default OFF via STRATEGY_KILL_SWITCH_ENABLED. When tripped by
        # MasterBrain.record_strategy_r (item #6), blocks all entries for
        # this strategy until auto-reset or manual clear.
        try:
            brain = getattr(self, "_brain_ref", None)
            if brain is not None and hasattr(brain, "_is_strategy_killed"):
                if brain._is_strategy_killed(_strat):
                    log.warning("[%s] R_STRATEGY_KILLED %s — entry blocked", symbol, _strat)
                    return False
        except Exception:
            pass  # fail-open

        # 2026-05-14: market-closed lockout. If a previous order returned
        # retcode 10018 (MARKET_CLOSED), skip entries for 1h to stop retry storm.
        mc = getattr(self, "_market_closed_until", {}) or {}
        until = float(mc.get(symbol, 0))
        if until > 0 and time.time() < until:
            mins_left = (until - time.time()) / 60.0
            log.debug("[%s] Market closed — %.0fmin lockout remaining", symbol, mins_left)
            return False

        # ── Hard re-entry floor (independent of brain cooldown) ──
        # Last-resort guard: even if brain logic is buggy, never re-open within
        # EXECUTOR_MIN_REENTRY_SECS of an MT5-detected close on this symbol.
        ext = getattr(self, '_external_close_time', {}) or {}
        last_close = float(ext.get(symbol, 0))
        if last_close > 0:
            gap = time.time() - last_close
            if gap < EXECUTOR_MIN_REENTRY_SECS:
                log.warning("[%s] BLOCK re-open: only %.1fs since last close (floor=%ds)",
                            symbol, gap, EXECUTOR_MIN_REENTRY_SECS)
                return False

        # ── 2026-06-02: HARD MIN_SCORE floor (defensive against bypass paths) ──
        # USDJPY 2026-06-01 18:57 fired 3-sub LONG at raw_score=4.5 in low_vol
        # via pullback-fallback path (brain.py:1305) which doesn't re-check the
        # signal_quality gate at fill time. -$9.69 / 3-leg loss. This guard
        # catches any caller (pullback fallback, FVG, scalp, future paths) that
        # tries to enter at a sub-floor raw_score. Per memory feedback_no_skip_trades:
        # this is a score-quality guard, NOT a risk/spread/daily-loss block, so
        # it's in-scope to reject.
        if score is not None and float(score) < 5.0:
            log.warning("[%s] BLOCK entry: raw_score=%.2f below HARD floor 5.0 (caller bypass) dir=%s",
                        symbol, float(score), direction)
            return False

        si = self.mt5.symbol_info(symbol)
        if si is None:
            log.error("[%s] symbol_info returned None", symbol)
            return False

        # ── SMART SPREAD CHECK (vs signal-time spread) ──
        spread_ok, tick = self._check_spread_spike(symbol, signal_spread)
        if tick is None:
            log.error("[%s] symbol_info_tick returned None", symbol)
            return False

        price = float(tick.ask) if direction == "LONG" else float(tick.bid)
        point = float(si.point) if si.point else 0.00001
        digits = int(si.digits)

        # ── 2026-06-02: Same-price re-entry guard ──
        # DJ30 2026-06-01: LONG won +1.05R at 14:06 (entry 51334.55), then re-entered
        # LONG at EXACTLY 51334.55 at 14:49 (43 min later) → cut for -0.31R. The
        # post-win cooldown (1800s) had just expired and there was no price-epsilon
        # check, so the bot mechanically retook the same signal it just exited.
        # Block re-entry within 60 min of close if new price is within 0.05% of
        # last close's entry price. Catches the "stale signal" re-entry pattern.
        try:
            lcep = getattr(self, '_last_close_entry_price', {}) or {}
            last_entry = float(lcep.get(symbol, 0))
            ext2 = getattr(self, '_external_close_time', {}) or {}
            last_close_t = float(ext2.get(symbol, 0))
            if last_entry > 0 and last_close_t > 0:
                close_age = time.time() - last_close_t
                if close_age < 3600:  # 60 min window
                    price_diff_pct = abs(price - last_entry) / max(last_entry, 1e-9) * 100.0
                    if price_diff_pct < 0.05:
                        log.warning("[%s] BLOCK same-price re-entry: new=%.5f vs last_close_entry=%.5f "
                                    "(%.3f%% diff < 0.05%%, close_age=%.0fmin)",
                                    symbol, price, last_entry, price_diff_pct, close_age / 60.0)
                        return False
        except Exception as e:
            log.debug("[%s] same-price re-entry check failed: %s", symbol, e)

        # ── SPREAD FILTER (vs ATR) ──
        spread = float(tick.ask) - float(tick.bid)
        # Per-symbol tighter cap for wide-spread CFDs (live evidence of large
        # slippage). Defaults to MAX_SPREAD_ATR_RATIO.
        spread_cap = MAX_SPREAD_ATR_RATIO_BY_SYMBOL.get(symbol, MAX_SPREAD_ATR_RATIO)
        if atr > 0 and spread / float(atr) > spread_cap:
            log.warning("[%s] SKIP: spread %.5f > %.0f%% of ATR %.5f",
                        symbol, spread, spread_cap * 100, atr)
            return False

        # ── RECENT-LATENCY FILTER ──
        # If recent order_send latency has been high, the broker book is
        # wobbly. Live evidence: JPN225ft fill at 2510ms latency → 500pt slip.
        # Skip rather than take a likely-bad fill.
        lat_hist = self._exec_latencies.get(symbol)
        if lat_hist and len(lat_hist) >= MIN_LATENCY_HISTORY:
            avg_lat = sum(lat_hist) / len(lat_hist)
            if avg_lat > MAX_AVG_LATENCY_MS:
                log.warning("[%s] SKIP: avg recent latency %.0fms > %.0fms cap (%d trades)",
                            symbol, avg_lat, MAX_AVG_LATENCY_MS, len(lat_hist))
                return False

        # ── RECENT-SLIPPAGE FILTER ──
        # If this symbol's recent fills have slipped > 50% of typical SL
        # distance, the strategy edge is being eaten by microstructure. Skip.
        slip_hist = self._slippage_history.get(symbol)
        if slip_hist and len(slip_hist) >= MIN_SLIPPAGE_HISTORY:
            avg_slip_abs = sum(abs(s) for s in slip_hist) / len(slip_hist)
            # Use this symbol's current ATR×default SL mult as the comparator.
            base_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, ATR_SL_MULTIPLIER)
            typical_sl = float(atr) * base_mult / point if point > 0 else 1.0
            if typical_sl > 0 and avg_slip_abs / typical_sl > MAX_AVG_SLIPPAGE_VS_SL_RATIO:
                log.warning("[%s] SKIP: avg slippage %.1f pts = %.0f%% of typical SL %.1f pts (%d trades)",
                            symbol, avg_slip_abs,
                            avg_slip_abs / typical_sl * 100, typical_sl,
                            len(slip_hist))
                return False

        # ── SL DISTANCE ──
        # 2026-05-17: per-(symbol, regime) SL takes precedence when caller
        # supplies the regime. Falls back to per-symbol → global.
        try:
            from config import SYMBOL_ATR_SL_OVERRIDE_REGIME as _SL_REGIME
        except Exception:
            _SL_REGIME = {}
        _regime_sl_mult = _SL_REGIME.get(symbol, {}).get(regime) if regime else None
        if _regime_sl_mult is not None:
            base_sl_mult = float(_regime_sl_mult)
        else:
            base_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, ATR_SL_MULTIPLIER)
        sl_mult = base_sl_mult
        if hasattr(self, '_vol_model') and self._vol_model:
            try:
                vol_pred = self._vol_model.predict_from_state(symbol, self.state)
                if vol_pred and vol_pred > 0:
                    sl_mult = base_sl_mult * max(0.8, min(1.5, vol_pred))
            except Exception as e:
                log.debug("[%s] Vol model fallback: %s", symbol, e)

        # ── MOMENTUM-ADAPTIVE SL (deep tune v3 2026-05-11, gated separately) ──
        # SPLIT from trail/lock: enabling wider SL was inflating losses on
        # ranging days (live evidence: 72% WR but PF 0.6 because losses
        # -1.3R while wins stayed +0.3R). Now off by default until proven
        # separately profitable on a stable-regime backtest.
        try:
            from config import MOMENTUM_SL_ADAPTIVE_ENABLED
            if MOMENTUM_SL_ADAPTIVE_ENABLED and self.state is not None:
                from signals.momentum_signal import compute_momentum, sl_multiplier
                ind = self.state.get_indicators(symbol) or {}
                df = self.state.get_candles(symbol, 60)
                mom = compute_momentum(ind, df)
                sl_mult *= sl_multiplier(mom)
        except Exception as e:
            log.debug("[%s] momentum sl mult failed: %s", symbol, e)

        sl_dist = max(float(atr) * sl_mult, float(si.trade_stops_level) * point * 2)

        # ── RISK & LOT SIZING ──
        effective_risk = risk_pct if risk_pct is not None else MAX_RISK_PER_TRADE_PCT
        equity = float(self.state.get_agent_state().get("equity", 1000))
        risk_amount = equity * (effective_risk / 100.0)

        tick_value = float(si.trade_tick_value) if si.trade_tick_value else 1.0
        tick_size = float(si.trade_tick_size) if si.trade_tick_size else point

        sl_ticks = sl_dist / tick_size if tick_size > 0 else sl_dist / point
        if tick_value > 0 and sl_ticks > 0:
            total_volume = risk_amount / (sl_ticks * tick_value)
        else:
            total_volume = float(cfg.volume_min)

        vol_min = float(si.volume_min) if si.volume_min else 0.01
        vol_max = float(si.volume_max) if si.volume_max else 10.0
        vol_step = float(si.volume_step) if si.volume_step else 0.01

        total_volume = max(vol_min, min(vol_max, total_volume))

        # ── SMALL ACCOUNT PROTECTION (2026-05-02 reworked) ──
        # Old behaviour silently SHRANK the SL distance so vol_min × tighter_sl
        # stayed within MAX_RISK_OVER × intended. Result: actual $ at risk if SL
        # hit was up to 3× the intended amount, with NO clear warning surfaced
        # past the executor logs. Risk audit (a04b32...) flagged this as the #1
        # silent inflation channel before going real-money.
        # New behaviour: REJECT the entry when vol_min × intended_sl exceeds
        # 1.5× intended risk. Better to skip a marginal trade than silently
        # take 3× the position size we promised the user.
        # 2026-05-11: raised 1.5 → 3.0 so $1K account can trade broker-min-lot
        # indices (GER40.r min ≈ $20.95 = 2.1% on $1K intent of 1%). Hard kill
        # switches (4% daily / 10% weekly) bound the downside.
        MAX_RISK_OVER = 3.0
        # 2026-05-13: per-symbol whitelist — proven positive-EV symbols are
        # ALLOWED to take vol_min×SL even when it exceeds the cap. Account
        # is small ($1.3K) but these symbols have demonstrated edge in both
        # backtest and live history; blocking them throws away real PnL.
        # Bleeders / negative-EV symbols stay capped (cap stays as designed).
        # Per user memory: "warn only on risk size, never skip trades" —
        # the warn-only override applies only where there's earned edge.
        from config import VOL_MIN_WARN_ONLY_SYMBOLS
        VOL_MIN_ABSOLUTE_CAP_PCT = 3.0  # 2026-05-14: GAS-Cr blew 11.2% via whitelist
        if total_volume <= vol_min and tick_value > 0 and tick_size > 0:
            forced_risk = sl_ticks * tick_value * vol_min
            if forced_risk > risk_amount * MAX_RISK_OVER:
                forced_pct = forced_risk / equity * 100 if equity > 0 else 0
                # Absolute cap: regardless of whitelist, never let forced risk
                # exceed 3% of equity. Audit found GAS-Cr taking 11.2% on a
                # "proven-EV" whitelist entry → -$36 in 4 min after sharp loss.
                if forced_pct > VOL_MIN_ABSOLUTE_CAP_PCT:
                    log.warning(
                        "[%s] ENTRY REJECTED: forced risk $%.2f (%.2f%%) > %.1f%% "
                        "ABSOLUTE cap — vol_min×SL too large for current equity.",
                        symbol, forced_risk, forced_pct, VOL_MIN_ABSOLUTE_CAP_PCT,
                    )
                    return False
                if symbol in VOL_MIN_WARN_ONLY_SYMBOLS:
                    log.warning(
                        "[%s] VOL_MIN OVERRIDE: forced risk $%.2f (%.2f%%) "
                        "vs intended $%.2f — proven-EV whitelist, proceeding.",
                        symbol, forced_risk, forced_pct, risk_amount,
                    )
                else:
                    log.warning(
                        "[%s] ENTRY REJECTED: vol_min×SL would risk $%.2f (%.2f%%) "
                        "vs intended $%.2f (%.2f%%) — exceeds %.1fx cap. "
                        "Account too small for ATR-based SL on this symbol.",
                        symbol, forced_risk, forced_pct, risk_amount,
                        effective_risk, MAX_RISK_OVER,
                    )
                    return False

        # Exposure check (HARD BLOCK — was warn-only, caused account blowouts)
        current_exposure = self._get_total_exposure()
        new_risk_pct = (risk_amount / equity * 100) if equity > 0 else 100
        if current_exposure + new_risk_pct > MAX_TOTAL_EXPOSURE_PCT:
            log.warning("[%s] BLOCKED: Exposure %.1f%%+%.1f%% > %.1f%% limit",
                        symbol, current_exposure, new_risk_pct, MAX_TOTAL_EXPOSURE_PCT)
            return False

        # ── SAFETY: force single if 3 subs would each clamp to vol_min (3x intended risk) ──
        order_type = 0 if direction == "LONG" else 1
        opened = 0
        use_single = symbol in SINGLE_POSITION_SYMBOLS
        if not use_single and total_volume < vol_min * 3:
            # Can't split into 3 meaningful subs — each would clamp to vol_min = 3x risk
            use_single = True
            log.warning("[%s] Lot %.4f < 3x min %.2f — forcing single to avoid 3x risk",
                        symbol, total_volume, vol_min)

        if use_single:
            # SINGLE POSITION — for trend-following symbols (BTCUSD etc.)
            # Full lot, wide TP, trailing SL does the work
            volume = total_volume
            if vol_step > 0:
                volume = float(round(int(volume / vol_step) * vol_step, 2))
            volume = max(vol_min, min(vol_max, volume))

            # 2026-05-02: TP changed from sl_dist*50 → sl_dist*5.0. Old value
            # was set when SL was 0.3x ATR (50R = 15 ATR ≈ realistic runner).
            # New 1.5-2.0x ATR SL → 50R = 75-100 ATR ≈ never reachable.
            # 5R is realistic for a single-position runner; trail handles
            # exits past that.
            if direction == "LONG":
                sl = float(round(price - sl_dist, digits))
                tp = float(round(price + sl_dist * 5.0, digits))
            else:
                sl = float(round(price + sl_dist, digits))
                tp = float(round(price - sl_dist * 5.0, digits))

            request = {
                "action": int(1), "symbol": str(symbol), "volume": float(volume),
                "type": int(order_type), "price": float(price),
                "sl": float(sl), "tp": float(tp), "deviation": int(_get_deviation(symbol)),
                "magic": int(cfg.magic), "comment": str("Dragon_Single"),
                "type_filling": int(1), "type_time": int(0),
            }
            result, actual_vol = self._send_order(request, symbol, context="SINGLE")
            if result and int(result.retcode) in (10009, 10008):
                opened = 1
                actual_price = float(result.price) if hasattr(result, 'price') and result.price else price
                log.info("[%s] SINGLE OPENED %s %.2f lots @ %.5f SL=%.5f (trend-follower)",
                         symbol, direction, actual_vol, actual_price, sl)

            if opened == 0:
                return False
            with self._lock:
                self._entry_prices[symbol] = float(result.price) if hasattr(result, 'price') and result.price else float(price)
                self._entry_sl_dist[symbol] = float(sl_dist)
                self._directions[symbol] = direction
                # 2026-05-14: track ACTUAL dollar risk for correct R-multiple at close.
                # If VOL_MIN forced a larger lot than intended, the real risk is
                # sl_dist × tick_value/tick_size × actual_vol — NOT risk_amount.
                # Otherwise R-multiples report 12R losses on what were really 0.7R.
                try:
                    actual_dollar_risk = float(sl_dist) * (
                        float(tick_value) / max(float(tick_size), 1e-9)
                    ) * float(actual_vol)
                except Exception:
                    actual_dollar_risk = float(risk_amount)
                if actual_dollar_risk <= 0:
                    actual_dollar_risk = float(risk_amount)
                self._entry_dollar_risk[symbol] = actual_dollar_risk
            self.state.update_agent("entry_prices", dict(self._entry_prices))
            self.state.update_agent("entry_sl_dist", dict(self._entry_sl_dist))
            self.state.update_agent("directions", dict(self._directions))
            self.state.update_agent("entry_dollar_risk", dict(self._entry_dollar_risk))
            log.info("[%s] OPENED single %s %.2f lots (risk=$%.2f %.3f%%)",
                     symbol, direction, actual_vol, risk_amount, effective_risk)
            # Dashboard WS hook
            try:
                from dashboard import v2_api as _v2  # type: ignore
                _v2.push_position_event("opened", {
                    "symbol": symbol, "side": direction,
                    "size": float(actual_vol),
                    "price": float(self._entry_prices.get(symbol, price)),
                    "ts": time.time(),
                    "magic": int(cfg.magic),
                    "mode": "single",
                })
            except Exception:
                pass
            return True

        # ── OPEN 3 SUB-POSITIONS (for non-trend-following symbols) ──
        # 2026-05-12: adaptive R:R per signal quality. Hi-conv = wider TPs.
        # 2026-06-19: when config.ADAPTIVE_TP_ENABLED, resolve TP1/TP2 from
        # per-(sym × regime × score-tier) research dict in auto_tuned.py.
        # The 3-leg ladder is built as [tp1, tp2, tp3] where tp3 = min(tp2*1.6, 4R)
        # so the existing trail/exit machinery still keeps a runner. Fail-open:
        # on any exception, drop back to the legacy adaptive_sub_tp_r() path.
        sub_tp_r = None
        adaptive_tp_active = False
        try:
            from config import ADAPTIVE_TP_ENABLED as _ADTP_ON, ADAPTIVE_TP_FAIL_OPEN as _ADTP_FAILOPEN  # type: ignore
        except Exception:
            _ADTP_ON, _ADTP_FAILOPEN = False, True
        if _ADTP_ON:
            try:
                from agent.expert.adaptive_tp import get_adaptive_tp as _get_adaptive_tp  # type: ignore
                tp1_r, tp2_r = _get_adaptive_tp(symbol, regime, score)
                tp3_r = min(max(tp2_r * 1.6, tp2_r + 0.5), 4.0)
                sub_tp_r = [float(tp1_r), float(tp2_r), float(tp3_r)]
                adaptive_tp_active = True
                log.info("[%s] AdaptiveTP active: regime=%s score=%.1f → TP_R=[%.2f, %.2f, %.2f]",
                         symbol, str(regime), float(score) if score is not None else 0.0,
                         sub_tp_r[0], sub_tp_r[1], sub_tp_r[2])
            except Exception as _e:
                if not _ADTP_FAILOPEN:
                    log.error("[%s] AdaptiveTP failed (no fail-open): %s", symbol, _e)
                    return False
                log.warning("[%s] AdaptiveTP failed, fail-open → legacy ladder: %s", symbol, _e)
                sub_tp_r = None
        if sub_tp_r is None:
            sub_tp_r = adaptive_sub_tp_r(score, symbol=symbol)
        if sub_tp_r != SUB_TP_R and not adaptive_tp_active:
            log.info("[%s] Adaptive TP scaling: score=%.1f → TP_R=%s",
                     symbol, float(score) if score is not None else 0.0, sub_tp_r)
        total_filled_volume = 0.0
        fill_prices = []  # (volume, price) tuples for weighted avg entry
        for i, (split, tp_r, magic_off) in enumerate(zip(SUB_SPLITS, sub_tp_r, SUB_MAGIC_OFFSETS)):
            sub_vol = total_volume * split
            if vol_step > 0:
                sub_vol = float(round(int(sub_vol / vol_step) * vol_step, 2))
            sub_vol = max(vol_min, min(vol_max, sub_vol))

            # Smart TP: use MTF liquidity/fibonacci TP for Sub0/Sub1 if available
            if smart_tp and smart_tp > sl_dist * 1.5 and i < 2:
                # Sub0: smart_tp (targets liquidity zone)
                # Sub1: 1.5x smart_tp (beyond first zone)
                tp_dist = smart_tp if i == 0 else smart_tp * 1.5
                log.info("[%s] Sub%d using smart TP=%.5f (MTF liquidity/fib)", symbol, i, tp_dist)
            else:
                tp_dist = sl_dist * tp_r

            if direction == "LONG":
                sl = float(round(price - sl_dist, digits))
                tp = float(round(price + tp_dist, digits))
            else:
                sl = float(round(price + sl_dist, digits))
                tp = float(round(price - tp_dist, digits))

            sub_magic = int(cfg.magic) + magic_off
            sub_comment = f"Dragon_S{i}"

            request = {
                "action": int(1),
                "symbol": str(symbol),
                "volume": float(sub_vol),
                "type": int(order_type),
                "price": float(price),
                "sl": float(sl),
                "tp": float(tp),
                "deviation": int(_get_deviation(symbol)),
                "magic": int(sub_magic),
                "comment": str(sub_comment),
                "type_filling": int(1),
                "type_time": int(0),
            }

            result, actual_vol = self._send_order(request, symbol, context=f"SUB{i}")
            if result is None:
                continue

            retcode = int(result.retcode)
            if retcode in (10009, 10008):
                opened += 1
                actual_price = float(result.price) if hasattr(result, 'price') and result.price else price
                total_filled_volume += actual_vol
                fill_prices.append((actual_vol, actual_price))
                log.info("[%s] SUB%d OPENED %s %.2f lots @ %.5f SL=%.5f TP=%.5f (%.0fR)",
                         symbol, i, direction, actual_vol, actual_price, sl, tp, tp_r)

        if opened == 0:
            return False

        # Track entry using volume-weighted average fill price (not signal price)
        if fill_prices:
            total_vol = sum(v for v, _ in fill_prices)
            avg_fill = sum(v * p for v, p in fill_prices) / total_vol if total_vol > 0 else price
        else:
            avg_fill = price
        with self._lock:
            self._entry_prices[symbol] = float(avg_fill)
            self._entry_sl_dist[symbol] = float(sl_dist)
            self._directions[symbol] = direction
        self.state.update_agent("entry_prices", dict(self._entry_prices))
        self.state.update_agent("entry_sl_dist", dict(self._entry_sl_dist))
        self.state.update_agent("directions", dict(self._directions))

        actual_risk_usd = sl_dist / tick_size * tick_value * total_filled_volume if tick_size > 0 else 0
        log.info("[%s] OPENED %d/%d subs %s filled=%.2f/%.2f lots SL=%.2fpts REAL_RISK=$%.2f (%.1f%% equity) ATR=%.5f",
                 symbol, opened, len(SUB_SPLITS), direction, total_filled_volume, total_volume,
                 sl_dist, actual_risk_usd, actual_risk_usd / equity * 100 if equity > 0 else 0, atr)
        # Dashboard WS hook
        try:
            from dashboard import v2_api as _v2  # type: ignore
            _v2.push_position_event("opened", {
                "symbol": symbol, "side": direction,
                "size": float(total_filled_volume),
                "price": float(avg_fill),
                "ts": time.time(),
                "magic": int(cfg.magic),
                "mode": "swing_subs",
                "subs_filled": int(opened),
            })
        except Exception:
            pass
        return True

    # ════════════════════════════════════════════════════════════════════
    #  FVG STRATEGY — explicit-SL/TP entries on an isolated magic range
    #  (2026-05-29). Reuses the sizing/spread/exposure guards from open_trade
    #  but takes a caller-supplied entry/SL/TP1/TP2 (ICT sweep+FVG signal).
    #  2 legs: 50% @ TP1, 50% @ TP2; both share the FVG SL. Magic = base+1000/
    #  +1001 so momentum's trail/close never touch these tickets.
    # ════════════════════════════════════════════════════════════════════
    def _fvg_magics(self, symbol):
        from config import FVG_SUB_OFFSETS
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return []
        return [int(cfg.magic) + off for off in FVG_SUB_OFFSETS]

    def has_fvg_position(self, symbol) -> bool:
        try:
            positions = self.mt5.positions_get(symbol=symbol) or []
            fmag = set(self._fvg_magics(symbol))
            return any(int(p.magic) in fmag for p in positions)
        except Exception:
            return False

    def _static_si_tick(self, symbol):
        """READ-FREE (si, tick) for a trend/IMR symbol from STATIC specs +
        the sync daemon's disk quote — bypasses the contended in-process bridge
        read that fails for over-read symbols (BTC). Returns (None, None) if the
        symbol has no static spec or no fresh disk quote."""
        import types as _t
        import json as _json
        from pathlib import Path as _P
        try:
            from config import TREND_SYMBOL_SPECS
            spec = TREND_SYMBOL_SPECS.get(symbol)
            if not spec:
                return None, None
            p = _P(__file__).resolve().parent.parent / "data" / "live_positions.json"
            d = _json.loads(p.read_text())
            q = (d.get("quotes") or {}).get(symbol)
            if not q or float(q.get("bid", 0)) <= 0:
                return None, None
            si = _t.SimpleNamespace(
                point=spec["point"], digits=spec["digits"],
                trade_tick_value=spec["tick_value"], trade_tick_size=spec["tick_size"],
                volume_min=spec["vmin"], volume_max=spec["vmax"],
                volume_step=spec["vstep"], trade_stops_level=spec["stops"])
            tick = _t.SimpleNamespace(bid=float(q["bid"]), ask=float(q["ask"]))
            return si, tick
        except Exception:
            return None, None

    def _daily_loss_blocked(self, symbol):
        """True if `symbol` hit its daily max-loss limit (locked till UTC midnight by
        the brain's DAILY LOSS GATE). Checked in every open path."""
        lk = getattr(self, "_daily_loss_locked", None)
        return bool(lk and float(lk.get(symbol, 0)) > time.time())

    def _disk_positions(self, symbol, mags, max_age=90.0):
        """READ-FREE open legs for `symbol` from the sync daemon's disk file,
        filtered to `mags`. Fail-closed (None) on missing/stale/unparseable — a
        close must never act on frozen position data. Used when in-process
        positions_get returns None under bridge contention."""
        import json as _json
        import time as _time
        from pathlib import Path as _P
        try:
            p = _P(__file__).resolve().parent.parent / "data" / "live_positions.json"
            d = _json.loads(p.read_text())
            ts = float(d.get("ts", 0.0))
            if not ts or (_time.time() - ts) > max_age:
                return None
            return [x for x in d.get("positions", [])
                    if x.get("symbol") == symbol and int(x.get("magic", -1)) in mags]
        except Exception:
            return None

    def _close_magic_legs(self, symbol, mags, comment, label):
        """Close every open leg of `symbol` whose magic is in `mags`. Tries
        in-process reads (select+retry); on bridge-contention failure falls back
        to a READ-FREE close built from the disk sync position + disk quote — the
        action=1 close is a WRITE, and writes succeed when the contended reads
        return None (this is the class-fix for the recurring BTC 'CLOSE FAILED /
        UNPROTECTED' bug). Returns True if any leg closed."""
        positions = tk = None
        for _att in range(5):
            try:
                self.mt5.symbol_select(symbol, True)
                positions = self.mt5.positions_get(symbol=symbol)
                tk = self.mt5.symbol_info_tick(symbol)
            except Exception:
                positions = tk = None
            if positions is not None and tk is not None:
                break
            time.sleep(0.2 * (_att + 1))
        legs = []                       # normalized (ticket, ptype, volume, magic)
        bid = ask = None
        if positions is not None and tk is not None:
            for p in positions:
                if int(p.magic) in mags:
                    legs.append((int(p.ticket), int(p.type), float(p.volume), int(p.magic)))
            bid, ask = float(tk.bid), float(tk.ask)
        else:
            dpos = self._disk_positions(symbol, mags)
            _, dtick = self._static_si_tick(symbol)
            if dpos is None or dtick is None:
                log.warning("[%s %s] CLOSE FAILED (no in-proc read, no disk fallback): "
                            "positions=%s tick=%s — leg protected only by broker SL",
                            label, symbol, positions is not None, tk is not None)
                return False
            for x in dpos:
                legs.append((int(x["ticket"]), int(x["type"]), float(x["volume"]), int(x["magic"])))
            bid, ask = float(dtick.bid), float(dtick.ask)
            if legs:
                log.info("[%s %s] read-free close fallback: %d leg(s) via disk position + disk quote",
                         label, symbol, len(legs))
        any_closed = False
        for ticket, ptype, volume, magic in legs:
            ctype = 1 if ptype == 0 else 0
            cpx = bid if ptype == 0 else ask
            req = {"action": int(1), "symbol": str(symbol), "volume": float(volume),
                   "type": int(ctype), "position": int(ticket), "price": float(cpx),
                   "deviation": int(_get_deviation(symbol)), "magic": int(magic),
                   "comment": str(comment), "type_filling": int(1), "type_time": int(0)}
            r, _ = self._send_order(req, symbol, context=f"{label}_CLOSE_{comment}")
            if r and int(r.retcode) in (10009, 10008):
                any_closed = True
        if any_closed:
            log.info("[%s %s] CLOSED (%s)", label, symbol, comment)
        return any_closed

    def open_trade_explicit(self, symbol, direction, entry, sl, tp1, tp2,
                            risk_pct=None, magic_offsets=None, strategy_name="FVG"):
        """Open a 2-leg position with caller-supplied SL/TP at strategy-specific magics.

        2026-06-05: parametrized to support sweep-reclaim (SR) strategy.
        - magic_offsets: list/tuple of two sub-offsets to apply to cfg.magic
          (defaults to FVG_SUB_OFFSETS for backward compatibility).
        - strategy_name: "FVG" or "SR" — gates the existing-position check on
          the appropriate has_X_position() so each strategy is mutually-aware.
        """
        # Backward-compat: if magic_offsets is None, fall back to FVG offsets
        if magic_offsets is None:
            from config import FVG_SUB_OFFSETS
            magic_offsets = FVG_SUB_OFFSETS
        # TREND book may open AUX_SYMBOLS (ETH/JPN/NAS) that are deliberately not
        # in the always-on SYMBOLS scan loops; all other callers stay SYMBOLS-only.
        cfg = symbol_cfg(symbol) if strategy_name in ("TREND", "GOLD_SMC") else SYMBOLS.get(symbol)
        if cfg is None:
            return False
        # Strategy-specific existing-position guard
        if strategy_name == "SR":
            if self.has_sr_position(symbol):
                return False
        elif strategy_name == "SMABO":
            if self.has_smabo_position(symbol):
                return False
        elif strategy_name == "FIB50":
            if self.has_fib50_position(symbol):
                return False
        elif strategy_name == "TREND":
            # 2026-07-08: guard on the TREND magics (base+6000/6001), NOT FVG.
            # Prevents a duplicate trend open if the sync file lags a just-filled
            # position. (Previously fell through to has_fvg_position = wrong range,
            # and for AUX symbols that returns False = zero dup protection.)
            if self.has_trend_position(symbol):
                return False
        elif strategy_name == "GOLD_SMC":
            # GOLD_SMC (9th book, base+8000/8001): guard on its own magics so a
            # lagging sync file can't re-fire a duplicate 2-leg XAU entry.
            if self.has_gold_smc_position(symbol):
                return False
        else:  # default FVG
            if self.has_fvg_position(symbol):
                return False
        # market-closed lockout + re-entry floor (shared with momentum)
        mc = getattr(self, "_market_closed_until", {}) or {}
        if float(mc.get(symbol, 0)) > time.time():
            log.warning("[%s %s] open reject: market-closed lockout until %.0f",
                        strategy_name, symbol, float(mc.get(symbol, 0)))
            return False
        # daily per-symbol loss lock (max loss hit → no new entries till EOD)
        if self._daily_loss_blocked(symbol):
            log.warning("[%s %s] open reject: daily loss limit reached — locked till EOD",
                        strategy_name, symbol)
            return False
        # 2026-07-10 AUDIT FIX: ensure the symbol is in Market Watch AND retry —
        # AUX symbols (ETH/JPN/NAS) get dropped from selection after a bridge
        # restart, and the in-process symbol_info/tick read fails intermittently
        # under bridge contention (isolated reads always work). Without this,
        # EVERY entry was rejected (si=False tick=False live outage). 3 tries.
        si = tick = None
        for _att in range(5):
            try:
                self.mt5.symbol_select(symbol, True)
                si = self.mt5.symbol_info(symbol)
                tick = self.mt5.symbol_info_tick(symbol)
            except Exception:
                si = tick = None
            if si is not None and tick is not None:
                break
            time.sleep(0.2 * (_att + 1))       # escalating backoff to catch a free bridge window
        if si is None or tick is None:
            # READ-FREE FALLBACK (2026-07-12): the always-on SYMBOLS loops saturate
            # the bridge so BTC's in-process symbol read fails ~100%. Build si/tick
            # from STATIC specs + the sync daemon's disk quote so the order still
            # places. Isolated reads (sync daemon) are reliable; this bypasses the
            # trader's contended client entirely for the read half.
            si2, tick2 = self._static_si_tick(symbol)
            if si2 is not None and tick2 is not None:
                si, tick = si2, tick2
                log.info("[%s %s] read-free fallback: static specs + disk quote bid=%.2f",
                         strategy_name, symbol, float(tick.bid))
            else:
                log.warning("[%s %s] open reject after 5 tries (no disk fallback): si=%s tick=%s",
                            strategy_name, symbol, si is not None, tick is not None)
                return False
        point = float(si.point) if si.point else 0.00001
        digits = int(si.digits)
        # 2026-07-08: TREND passes SL/TP as absolute prices off the STALE D1 close,
        # but the order fills at the live tick. Re-anchor the distances to the live
        # fill reference so a risk-capped tight stop can never land on the wrong
        # side of the market (invalid-stops reject / instant stop-out).
        if strategy_name == "TREND":
            ref = float(tick.ask if direction == "LONG" else tick.bid)
            _sl_d = abs(float(entry) - float(sl))
            _tp_d = abs(float(entry) - float(tp1))
            entry = ref   # anchor sizing + fill fallback to the live reference
            if direction == "LONG":
                sl = ref - _sl_d
                tp1 = tp2 = (ref + _tp_d) if _tp_d > 0 else 0.0
            else:
                sl = ref + _sl_d
                tp1 = tp2 = (ref - _tp_d) if _tp_d > 0 else 0.0
        sl_dist = abs(float(entry) - float(sl))
        if sl_dist <= 0:
            log.warning("[%s %s] open reject: sl_dist<=0 (entry=%.2f sl=%.2f)",
                        strategy_name, symbol, float(entry), float(sl))
            return False

        # sizing — same math as open_trade
        effective_risk = risk_pct if risk_pct is not None else MAX_RISK_PER_TRADE_PCT
        equity = float(self.state.get_agent_state().get("equity", 1000))
        risk_amount = equity * (effective_risk / 100.0)
        tick_value = float(si.trade_tick_value) if si.trade_tick_value else 1.0
        tick_size = float(si.trade_tick_size) if si.trade_tick_size else point
        sl_ticks = sl_dist / tick_size if tick_size > 0 else sl_dist / point
        total_volume = (risk_amount / (sl_ticks * tick_value)
                        if tick_value > 0 and sl_ticks > 0 else float(cfg.volume_min))
        vol_min = float(si.volume_min) if si.volume_min else 0.01
        vol_max = float(si.volume_max) if si.volume_max else 10.0
        vol_step = float(si.volume_step) if si.volume_step else 0.01
        total_volume = max(vol_min, min(vol_max, total_volume))

        # exposure cap (shared, hard block)
        new_risk_pct = (risk_amount / equity * 100) if equity > 0 else 100
        if self._get_total_exposure() + new_risk_pct > MAX_TOTAL_EXPOSURE_PCT:
            log.warning("[%s %s] BLOCKED exposure", strategy_name, symbol)
            return False

        order_type = 0 if direction == "LONG" else 1
        if strategy_name == "TREND":
            log.info("[TREND %s] reached send: total_vol=%.3f vmin=%.3f sl_dist=%.2f",
                     symbol, total_volume, vol_min, sl_dist)
        # 2-leg split: 50/50. Each leg at least vol_min.
        leg_vol = max(vol_min, total_volume / 2.0)
        if vol_step > 0:
            leg_vol = float(round(int(leg_vol / vol_step) * vol_step, 2))
        leg_vol = max(vol_min, min(vol_max, leg_vol))
        sl_r = float(round(float(sl), digits))
        legs = [(magic_offsets[0], float(round(float(tp1), digits))),
                (magic_offsets[1], float(round(float(tp2), digits)))]
        opened = 0
        fill_px = float(entry)
        for off, tp_px in legs:
            request = {
                "action": int(1), "symbol": str(symbol), "volume": float(leg_vol),
                "type": int(order_type), "price": float(tick.ask if direction == "LONG" else tick.bid),
                "sl": sl_r, "tp": tp_px, "deviation": int(_get_deviation(symbol)),
                "magic": int(cfg.magic) + int(off), "comment": str(strategy_name),
                "type_filling": int(1), "type_time": int(0),
            }
            result, actual_vol = self._send_order(request, symbol, context=f"{strategy_name}_{off}")
            if result and int(result.retcode) in (10009, 10008):
                opened += 1
                if hasattr(result, "price") and result.price:
                    fill_px = float(result.price)
        if opened == 0:
            return False
        # record strategy entry tracking (separate from momentum dicts).
        # Each strategy gets its own time-tracking dict + persisted state key
        # so time-stop / BE-move logic doesn't cross strategies.
        with self._lock:
            if strategy_name == "SR":
                if not hasattr(self, "_sr_entry_time"):
                    self._sr_entry_time = {}
                self._sr_entry_time[symbol] = time.time()
                state_key = "sr_entry_time"
                state_dict = dict(self._sr_entry_time)
            elif strategy_name == "SMABO":
                if not hasattr(self, "_smabo_entry_time"):
                    self._smabo_entry_time = {}
                self._smabo_entry_time[symbol] = time.time()
                state_key = "smabo_entry_time"
                state_dict = dict(self._smabo_entry_time)
            elif strategy_name == "FIB50":
                if not hasattr(self, "_fib50_entry_time"):
                    self._fib50_entry_time = {}
                self._fib50_entry_time[symbol] = time.time()
                state_key = "fib50_entry_time"
                state_dict = dict(self._fib50_entry_time)
            elif strategy_name == "SCALPER":
                if not hasattr(self, "_scalper_entry_time"):
                    self._scalper_entry_time = {}
                self._scalper_entry_time[symbol] = time.time()
                state_key = "scalper_entry_time"
                state_dict = dict(self._scalper_entry_time)
            elif strategy_name in ("TREND", "GOLD_SMC"):
                # 2026-07-10 review: TREND/GOLD_SMC must NOT write FVG bookkeeping —
                # their exits are self-managed read-free and read none of these keys,
                # and writing _fvg_entry_time would silently reset a live FVG time-stop
                # if FVG is ever re-enabled on the same symbol.
                state_key = None
                state_dict = None
            else:
                if not hasattr(self, "_fvg_entry_time"):
                    self._fvg_entry_time = {}
                self._fvg_entry_time[symbol] = time.time()
                state_key = "fvg_entry_time"
                state_dict = dict(self._fvg_entry_time)
        try:
            if state_key is not None:
                self.state.update_agent(state_key, state_dict)
        except Exception:
            pass

        # 2026-06-19: also persist strategy-keyed entry price + SL distance so
        # _apply_trail's lookup at executor.py:2157-2158 finds the SR/FVG signal
        # SL distance (e.g. GER40.r SR 133.55pts) instead of falling back to
        # pos.price_open + ATR. Without this, profit_r calc drifts from the
        # SR/FVG-tuned SL and exit tiers fire on wrong R values.
        try:
            with self._lock:
                if strategy_name == "SR":
                    tk = symbol + "_sr"
                elif strategy_name == "SMABO":
                    tk = symbol + "_smabo"
                elif strategy_name == "FIB50":
                    tk = symbol + "_fib50"
                elif strategy_name in ("TREND", "GOLD_SMC"):
                    tk = None      # read-free self-managed exits — don't pollute FVG keys
                else:
                    tk = symbol + "_fvg"
                if tk is not None:
                    self._entry_prices[tk] = float(fill_px)
                    self._entry_sl_dist[tk] = float(abs(float(entry) - float(sl)))
            if tk is not None:
                self.state.update_agent("entry_prices", dict(self._entry_prices))
                self.state.update_agent("entry_sl_dist", dict(self._entry_sl_dist))
        except Exception:
            pass
        log.info("[%s %s] OPENED %s %d/2 legs @ %.5f SL=%.5f TP1=%.5f TP2=%.5f risk=%.2f%%",
                 strategy_name, symbol, direction, opened, fill_px, sl_r, legs[0][1], legs[1][1], effective_risk)
        return True

    def close_fvg_position(self, symbol, comment="FVG_close"):
        """Close all FVG-magic legs for symbol (used by the 4H time stop)."""
        try:
            positions = self.mt5.positions_get(symbol=symbol) or []
        except Exception:
            return False
        fmag = set(self._fvg_magics(symbol))
        any_closed = False
        for p in positions:
            if int(p.magic) not in fmag:
                continue
            ctype = 1 if int(p.type) == 0 else 0
            tk = self.mt5.symbol_info_tick(symbol)
            cpx = float(tk.bid) if int(p.type) == 0 else float(tk.ask)
            req = {"action": int(1), "symbol": str(symbol), "volume": float(p.volume),
                   "type": int(ctype), "position": int(p.ticket), "price": cpx,
                   "deviation": int(_get_deviation(symbol)), "magic": int(p.magic),
                   "comment": str(comment), "type_filling": int(1), "type_time": int(0)}
            r, _ = self._send_order(req, symbol, context=f"FVG_CLOSE_{comment}")
            if r and int(r.retcode) in (10009, 10008):
                any_closed = True
        if any_closed:
            with self._lock:
                if hasattr(self, "_fvg_entry_time"):
                    self._fvg_entry_time.pop(symbol, None)
                # 2026-06-19: also clear strategy-keyed entry/SL state and
                # peak-R tracking so a fresh FVG entry on the same symbol gets
                # clean profit_r math (mirrors close_position cleanup).
                fvg_tk = symbol + "_fvg"
                self._entry_prices.pop(fvg_tk, None)
                self._entry_sl_dist.pop(fvg_tk, None)
                if hasattr(self, "_peak_profit_r"):
                    self._peak_profit_r.pop(fvg_tk, None)
            try:
                self.state.update_agent("fvg_entry_time", dict(getattr(self, "_fvg_entry_time", {})))
                self.state.update_agent("entry_prices", dict(self._entry_prices))
                self.state.update_agent("entry_sl_dist", dict(self._entry_sl_dist))
            except Exception:
                pass
            log.info("[FVG %s] CLOSED (%s)", symbol, comment)
        return any_closed

    def _close_sr_position(self, symbol, comment="SR_close"):
        """Close all SR (sweep-reclaim) magic legs for symbol.

        2026-06-19: added so EarlyLossCut / PeakGiveback / TimeStop / BOS /
        HardDollarCap / VWAP exits route to the SR magic range when triggered
        on an SR ticket. Without this, close_position() (which only matches
        swing magics) silently no-ops on SR-only positions and the trade rides
        all the way to its SL — root cause of the GER40.r -5.31R bypass on
        2026-06-18.
        """
        try:
            from config import SR_SUB_OFFSETS
        except Exception:
            return False
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        sr_magics = {int(cfg.magic) + off for off in SR_SUB_OFFSETS}
        try:
            positions = self.mt5.positions_get(symbol=symbol) or []
        except Exception:
            return False
        any_closed = False
        for p in positions:
            if int(p.magic) not in sr_magics:
                continue
            ctype = 1 if int(p.type) == 0 else 0
            tk = self.mt5.symbol_info_tick(symbol)
            if tk is None:
                continue
            cpx = float(tk.bid) if int(p.type) == 0 else float(tk.ask)
            req = {"action": int(1), "symbol": str(symbol), "volume": float(p.volume),
                   "type": int(ctype), "position": int(p.ticket), "price": cpx,
                   "deviation": int(_get_deviation(symbol)), "magic": int(p.magic),
                   "comment": str(comment), "type_filling": int(1), "type_time": int(0)}
            r, _ = self._send_order(req, symbol, context=f"SR_CLOSE_{comment}")
            if r and int(r.retcode) in (10009, 10008):
                any_closed = True
        if any_closed:
            with self._lock:
                if hasattr(self, "_sr_entry_time"):
                    self._sr_entry_time.pop(symbol, None)
                # Clear strategy-keyed trail state so the next SR entry starts clean.
                sr_tk = symbol + "_sr"
                self._entry_prices.pop(sr_tk, None)
                self._entry_sl_dist.pop(sr_tk, None)
                if hasattr(self, "_peak_profit_r"):
                    self._peak_profit_r.pop(sr_tk, None)
            try:
                self.state.update_agent("sr_entry_time", dict(getattr(self, "_sr_entry_time", {})))
                self.state.update_agent("entry_prices", dict(self._entry_prices))
                self.state.update_agent("entry_sl_dist", dict(self._entry_sl_dist))
            except Exception:
                pass
            log.info("[SR %s] CLOSED (%s)", symbol, comment)
        return any_closed

    def manage_fvg_position(self, symbol):
        """Per-cycle FVG management: 4H time stop (close if TP1 not hit in window)
        and BE-move on the runner leg once the TP1 leg has closed."""
        from config import FVG_TIME_STOP_SECS, FVG_SUB_OFFSETS
        try:
            positions = self.mt5.positions_get(symbol=symbol) or []
        except Exception:
            return
        fmag = set(self._fvg_magics(symbol))
        fvg_pos = [p for p in positions if int(p.magic) in fmag]
        if not fvg_pos:
            return
        cfg = SYMBOLS.get(symbol)
        tp1_magic = int(cfg.magic) + FVG_SUB_OFFSETS[0]
        tp1_open = any(int(p.magic) == tp1_magic for p in fvg_pos)
        # Time stop: TP1 not yet closed and window elapsed → flatten.
        et = getattr(self, "_fvg_entry_time", {}).get(symbol, 0)
        if et and tp1_open and (time.time() - et) >= FVG_TIME_STOP_SECS:
            self.close_fvg_position(symbol, comment="FVG_TIME_STOP")
            return
        # BE-move: TP1 leg gone (hit) → move runner SL to entry.
        # 2026-06-04 CTO audit B15: fix EURUSD FVG_BE invalid-stops loop —
        # 60,595 errors logged because BE was being set on the wrong side of
        # current price. Now:
        #   (1) Check current price vs BE — for LONG, BE must be BELOW current
        #       bid (else broker rejects retcode 10016 "Invalid stops");
        #       for SHORT, BE must be ABOVE current ask.
        #   (2) Per-ticket attempt counter — give up after 3 failed tries so
        #       we don't spam the broker every cycle.
        if not tp1_open:
            if not hasattr(self, '_fvg_be_attempts'):
                self._fvg_be_attempts = {}
            try:
                tick = self.mt5.symbol_info_tick(symbol)
            except Exception:
                tick = None
            for p in fvg_pos:
                ticket = int(p.ticket)
                if self._fvg_be_attempts.get(ticket, 0) >= 3:
                    continue  # already tried 3 times — give up
                be = float(p.price_open)
                cur_sl = float(p.sl) if p.sl else 0.0
                improve = (be > cur_sl) if int(p.type) == 0 else (cur_sl == 0 or be < cur_sl)
                if not improve:
                    continue
                # Validate BE side relative to current price
                if tick is not None:
                    if int(p.type) == 0:  # LONG — SL must be < current bid
                        if be >= float(tick.bid):
                            # Price retraced below entry; BE is invalid. Bail.
                            self._fvg_be_attempts[ticket] = 3   # mark done so we stop trying
                            log.debug("[FVG %s] BE skip — entry %.5f >= bid %.5f (price retraced)",
                                      symbol, be, float(tick.bid))
                            continue
                    else:  # SHORT — SL must be > current ask
                        if be <= float(tick.ask):
                            self._fvg_be_attempts[ticket] = 3
                            log.debug("[FVG %s] BE skip — entry %.5f <= ask %.5f (price retraced)",
                                      symbol, be, float(tick.ask))
                            continue
                req = {"action": int(6), "symbol": str(symbol), "position": ticket,
                       "sl": float(round(be, int(self.mt5.symbol_info(symbol).digits))),
                       "tp": float(p.tp), "magic": int(p.magic)}
                try:
                    result, _ = self._send_order(req, symbol, context="FVG_BE")
                    if result and int(result.retcode) in (10009, 10025):
                        self._fvg_be_attempts[ticket] = 99  # success — stop retrying
                    else:
                        self._fvg_be_attempts[ticket] = self._fvg_be_attempts.get(ticket, 0) + 1
                except Exception:
                    self._fvg_be_attempts[ticket] = self._fvg_be_attempts.get(ticket, 0) + 1

    def close_position(self, symbol, comment="DragonClose"):
        """Close all sub-positions for a symbol (magic, magic+1, magic+2).
        Returns True if at least one position was closed.

        2026-05-14: also arms a same-cycle re-entry lockout via _just_closed
        dict so brain._process_symbol can see "this symbol JUST closed, skip
        new entries for the rest of THIS cycle". Prevents the DJ30 race where
        a peak-giveback close and a fresh entry signal fired in the same cycle.
        """
        # Prevent concurrent closes on same symbol
        with self._lock:
            if self._closing.get(symbol, False):
                log.debug("[%s] Already closing, skip duplicate", symbol)
                return False
            self._closing[symbol] = True
        # Mark just-closed so same-cycle entries see it
        if not hasattr(self, "_just_closed"):
            self._just_closed = {}
        self._just_closed[symbol] = time.time()

        try:
            return self._close_position_impl(symbol, comment)
        finally:
            with self._lock:
                self._closing.pop(symbol, None)

    def _close_position_impl(self, symbol, comment):
        cfg = SYMBOLS.get(symbol)
        is_orphan = cfg is None

        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            log.warning("[%s] close_position: positions_get returned None", symbol)
            return False

        # 2026-05-13 orphan support: close ANY position for symbols not in SYMBOLS
        # (no magic filter — disabled symbols still need to be closeable)
        if is_orphan:
            valid_magics = None  # accept all
        else:
            valid_magics = {int(cfg.magic) + off for off in SUB_MAGIC_OFFSETS}
        any_closed = False
        total_pnl = 0.0
        total_peak_r = 0.0
        for p in positions:
            if valid_magics is not None and int(p.magic) not in valid_magics:
                continue
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                log.warning("[%s] close_position: no tick (market closed?)", symbol)
                continue

            # Capture unrealized PnL BEFORE close — for accurate alert/journal
            try:
                total_pnl += float(getattr(p, "profit", 0))
            except Exception:
                pass

            close_type = 1 if int(p.type) == 0 else 0  # Reverse direction
            close_price = float(tick.bid) if int(p.type) == 0 else float(tick.ask)

            request = {
                "action": int(1),
                "symbol": str(symbol),
                "volume": float(p.volume),
                "type": int(close_type),
                "price": float(close_price),
                "position": int(p.ticket),
                "deviation": int(_get_deviation(symbol)),
                # 2026-05-13: use position's own magic (handles orphans where cfg is None)
                "magic": int(cfg.magic) if cfg is not None else int(p.magic),
                "comment": str(comment),
                "type_filling": int(1),
                "type_time": int(0),
            }

            result, _ = self._send_order(request, symbol, context=f"CLOSE_{comment}")
            if result and int(result.retcode) in (10009, 10008):
                actual_price = float(result.price) if hasattr(result, 'price') and result.price else close_price
                log.info("[%s] CLOSED %s @ %.5f (%s)", symbol,
                         "LONG" if int(p.type) == 0 else "SHORT", actual_price, comment)
                any_closed = True
            else:
                # 2026-06-04: order_send failed/exhausted for this sub-position.
                # Queue a CLOSE intent so the next brain cycle retries via
                # drain_close_intents() instead of waiting for the next
                # PeakGiveback/SL trigger (which could be hours).
                self._queue_close_intent(symbol, comment)

        # 2026-06-04: post-failure verification. If `any_closed=False` but the
        # broker actually has no positions left for this symbol, the close
        # silently executed on the broker side despite our retry failure
        # (rpyc bridge "result expired" path). Treat as success.
        if not any_closed:
            try:
                _verify = self.mt5.positions_get(symbol=symbol) or []
                _valid_now = (
                    {int(cfg.magic) + off for off in SUB_MAGIC_OFFSETS}
                    if cfg is not None else None
                )
                _still_open = any(
                    (_valid_now is None or int(p.magic) in _valid_now)
                    for p in _verify
                )
                if not _still_open:
                    log.warning("[%s] CLOSE silently executed on broker side "
                                "despite order_send failure — treating as closed",
                                symbol)
                    any_closed = True
                    self._clear_close_intent(symbol)
            except Exception as e:
                log.debug("[%s] post-failure verify: %s", symbol, e)

        # Only clear tracking if at least one close succeeded
        if any_closed:
            # Clear any pending intent — close succeeded.
            self._clear_close_intent(symbol)
            with self._lock:
                closed_dir = self._directions.get(symbol, "?")
                # 2026-05-14: capture peak_r BEFORE clearing tracking
                peak_r_captured = (self._peak_profit_r.get(symbol, 0.0)
                                   if hasattr(self, '_peak_profit_r') else 0.0)
                # Snapshot actual dollar_risk for correct R-multiple recording.
                try:
                    _actual_risk = float(self._entry_dollar_risk.get(symbol, 0) or 0)
                    if _actual_risk > 0:
                        if not hasattr(self, "_last_close_dollar_risk"):
                            self._last_close_dollar_risk = {}
                        self._last_close_dollar_risk[symbol] = _actual_risk
                except Exception:
                    pass
                # 2026-05-26 audit fix: snapshot entry_price before pop
                _ep_snap = float(self._entry_prices.get(symbol, 0) or 0)
                if _ep_snap > 0:
                    if not hasattr(self, "_last_close_entry_price"):
                        self._last_close_entry_price = {}
                    self._last_close_entry_price[symbol] = _ep_snap
                self._entry_prices.pop(symbol, None)
                self._entry_sl_dist.pop(symbol, None)
                self._directions.pop(symbol, None)
                self._entry_dollar_risk.pop(symbol, None)
                self._peak_profit_r.pop(symbol, None)
                # 2026-06-05: drop BOS swing entries for any ticket we just closed.
                try:
                    for _p in positions:
                        if valid_magics is not None and int(_p.magic) not in valid_magics:
                            continue
                        self._entry_swings.pop(int(_p.ticket), None)
                except Exception:
                    pass
                # Persist post-pop state
                try:
                    self.state.update_agent("entry_prices", dict(self._entry_prices))
                    self.state.update_agent("entry_sl_dist", dict(self._entry_sl_dist))
                    self.state.update_agent("directions", dict(self._directions))
                    self.state.update_agent("entry_dollar_risk", dict(self._entry_dollar_risk))
                    self.state.update_agent("peak_profit_r", dict(self._peak_profit_r))
                except Exception:
                    pass
            # 2026-05-14 BUG FIX: previously hardcoded pnl=0.0 r_multiple=0.0
            # → journal/alerts lost the actual PnL on PeakGiveback/HardCap/etc.
            # Now pass captured unrealized total_pnl + peak_r.
            alerter = getattr(self, "_alerter", None)
            if alerter is not None:
                try:
                    alerter.position_close(symbol, closed_dir,
                                            float(total_pnl), float(peak_r_captured),
                                            comment)
                except Exception:
                    pass
            # Dashboard WS hook (lazy import to avoid circular deps).
            try:
                from dashboard import v2_api as _v2  # type: ignore
                _v2.push_position_event("closed", {
                    "symbol": symbol,
                    "side": closed_dir,
                    "size": 0.0,
                    "price": 0.0,
                    "ts": time.time(),
                    "magic": 0,
                    "reason": comment,
                })
            except Exception:
                pass
        return any_closed

    def close_all(self, comment="DragonEmergency"):
        """Close ALL open positions on the account, by ticket — not by SYMBOLS list.
        Critical: kill switch must close any open position, including manual or
        positions on symbols recently dropped from the live universe."""
        try:
            all_positions = self.mt5.positions_get() or []
        except Exception as e:
            log.error("close_all: positions_get failed: %s", e)
            all_positions = []
        seen_symbols = set()
        for p in all_positions:
            try:
                sym = str(p.symbol)
                seen_symbols.add(sym)
            except Exception:
                continue
        for sym in seen_symbols:
            try:
                self.close_position(sym, comment)
            except Exception as e:
                log.warning("close_all: close_position(%s) failed: %s", sym, e)
        if not seen_symbols:
            log.info("close_all: no open positions to close")
        else:
            log.warning("close_all: closed positions on %d symbol(s): %s",
                        len(seen_symbols), sorted(seen_symbols))

    def reverse_position(self, symbol, new_direction, atr, risk_pct=None, signal_spread=None):
        """Close current position and open in opposite direction.

        2026-05-14: abort on close failure. Previous behaviour cleared internal
        tracking and opened the opposite direction anyway — if the broker still
        had the original position, that produced a double-position (one each
        direction). Latent landmine because reversals are currently disabled
        in brain.py:1519 but the method remains callable.
        """
        if self.has_position(symbol):
            old_dir = self._directions.get(symbol, "?")
            log.info("[%s] REVERSING %s -> %s", symbol, old_dir, new_direction)
            closed = self.close_position(symbol, "DragonReversal")
            if not closed:
                log.warning("[%s] Reversal ABORTED — close failed; refusing to open opposite to avoid double-position",
                            symbol)
                return False
            time.sleep(0.2)
        return self.open_trade(symbol, new_direction, atr, risk_pct=risk_pct, signal_spread=signal_spread)

    def open_scalp_trade(self, symbol, direction, atr, risk_pct=None, signal_spread=None):
        """
        Open a scalp trade with scalp-specific risk and SL/TP.
        SL = 1.5x ATR(M5), TP = 2R hard target, risk = SCALP_RISK_PCT equity.
        Uses magic = base magic + SCALP_MAGIC_OFFSET.
        risk_pct overrides SCALP_RISK_PCT if provided (from MasterBrain).
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            log.error("[%s] Unknown symbol for scalp", symbol)
            return False

        scalp_magic = int(cfg.magic) + SCALP_MAGIC_OFFSET

        # Check if already has scalp position
        if self.has_scalp_position(symbol):
            log.info("[%s] Already has scalp position, skipping", symbol)
            return False

        # ── Hard re-entry floor (mirrors open_trade) ──
        ext = getattr(self, '_external_close_time', {}) or {}
        last_close = float(ext.get(symbol, 0))
        if last_close > 0:
            gap = time.time() - last_close
            if gap < EXECUTOR_MIN_REENTRY_SECS:
                log.warning("[%s] BLOCK scalp re-open: only %.1fs since last close (floor=%ds)",
                            symbol, gap, EXECUTOR_MIN_REENTRY_SECS)
                return False

        si = self.mt5.symbol_info(symbol)
        if si is None:
            log.error("[%s] symbol_info returned None (scalp)", symbol)
            return False

        # ── SMART SPREAD CHECK (vs signal-time spread) ──
        spread_ok, tick = self._check_spread_spike(symbol, signal_spread)
        if tick is None:
            log.error("[%s] symbol_info_tick returned None (scalp)", symbol)
            return False

        # Prices
        price = float(tick.ask) if direction == "LONG" else float(tick.bid)
        point = float(si.point) if si.point else 0.00001
        digits = int(si.digits)

        # SL distance = 1.5x ATR(M5), respect minimum stop distance
        sl_dist = max(float(atr) * SCALP_ATR_MULT, float(si.trade_stops_level) * point * 2)

        # TP = 2R hard target
        tp_dist = sl_dist * 2.0

        if direction == "LONG":
            sl = float(round(price - sl_dist, digits))
            tp = float(round(price + tp_dist, digits))
        else:
            sl = float(round(price + sl_dist, digits))
            tp = float(round(price - tp_dist, digits))

        # Risk-based lot sizing
        effective_risk = risk_pct if risk_pct is not None else SCALP_RISK_PCT
        equity = float(self.state.get_agent_state().get("equity", 1000))
        risk_amount = equity * (effective_risk / 100.0)

        tick_value = float(si.trade_tick_value) if si.trade_tick_value else 1.0
        tick_size = float(si.trade_tick_size) if si.trade_tick_size else point

        sl_ticks = sl_dist / tick_size if tick_size > 0 else sl_dist / point
        if tick_value > 0 and sl_ticks > 0:
            volume = risk_amount / (sl_ticks * tick_value)
        else:
            volume = float(cfg.volume_min)

        # Clamp to broker limits
        vol_min = float(si.volume_min) if si.volume_min else 0.01
        vol_max = float(si.volume_max) if si.volume_max else 10.0
        vol_step = float(si.volume_step) if si.volume_step else 0.01

        volume = max(vol_min, volume)
        volume = min(vol_max, volume)
        if vol_step > 0:
            volume = float(round(int(volume / vol_step) * vol_step, 2))

        # Small account protection: cap SL so vol_min risk stays within 3x intended
        MAX_SCALP_RISK_OVER = 3.0
        if volume <= vol_min and tick_value > 0 and tick_size > 0:
            max_allowed = risk_amount * MAX_SCALP_RISK_OVER
            max_sl_ticks = max_allowed / (tick_value * vol_min) if vol_min > 0 else sl_ticks
            max_sl = max_sl_ticks * tick_size
            if sl_dist > max_sl and max_sl > 0:
                sl_dist = max(max_sl, float(si.trade_stops_level) * point * 2)
                sl_ticks = sl_dist / tick_size
                log.info("[%s] Scalp SL capped: risk $%.2f (%.1f%%)", symbol,
                         sl_ticks * tick_value * vol_min, sl_ticks * tick_value * vol_min / equity * 100)

        # Build order request — all values cast to float()
        order_type = 0 if direction == "LONG" else 1
        request = {
            "action": int(1),             # TRADE_ACTION_DEAL
            "symbol": str(symbol),
            "volume": float(volume),
            "type": int(order_type),
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(_get_deviation(symbol)),
            "magic": int(scalp_magic),
            "comment": str("DragonScalp"),
            "type_filling": int(1),       # IOC
            "type_time": int(0),
        }

        result, actual_vol = self._send_order(request, symbol, context="SCALP")
        if result is None:
            return False

        retcode = int(result.retcode)
        if retcode not in (10009, 10008):
            return False

        # Track entry with scalp key
        scalp_key = symbol + "_scalp"
        actual_price = float(result.price) if hasattr(result, 'price') and result.price else float(price)
        with self._lock:
            self._entry_prices[scalp_key] = actual_price
            self._entry_sl_dist[scalp_key] = float(sl_dist)
            self._directions[scalp_key] = direction
            # 2026-05-14: parity with single-trade path — store actual dollar risk
            # under scalp_key so close-time R-multiple recording uses the correct
            # denominator (was previously falling back to intended-risk bug).
            if not hasattr(self, "_entry_dollar_risk"):
                self._entry_dollar_risk = {}
            self._entry_dollar_risk[scalp_key] = float(risk_amount)

        log.info("[%s] SCALP OPENED %s %.2f lots @ %.5f SL=%.5f TP=%.5f (risk=$%.2f %.1f%%, ATR=%.5f)",
                 symbol, direction, actual_vol, actual_price, sl, tp, risk_amount, effective_risk, atr)
        # Dashboard WS hook
        try:
            from dashboard import v2_api as _v2  # type: ignore
            _v2.push_position_event("opened", {
                "symbol": symbol, "side": direction,
                "size": float(actual_vol),
                "price": float(actual_price),
                "ts": time.time(),
                "magic": int(scalp_magic),
                "mode": "scalp",
            })
        except Exception:
            pass
        return True

    def has_scalp_position(self, symbol) -> bool:
        """Check if we have an open scalp position for this symbol."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        scalp_magic = int(cfg.magic) + SCALP_MAGIC_OFFSET
        scalp_magic_old = int(cfg.magic) + 100  # backward compat
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) in (scalp_magic, scalp_magic_old) for p in positions)

    def has_fvg_position(self, symbol) -> bool:
        """Check if we have an open FVG-strategy position for this symbol.

        2026-06-03: previously has_position() only saw momentum's magic range
        (cfg.magic + [0,1,2]) so momentum was BLIND to FVG positions on the
        same symbol — both strategies could hold it simultaneously. Used by
        brain.py Gate 0a to make momentum yield to active FVG trades.
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        try:
            from config import FVG_SUB_OFFSETS
            fvg_magics = {int(cfg.magic) + off for off in FVG_SUB_OFFSETS}
        except Exception:
            return False
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) in fvg_magics for p in positions)

    def has_sr_position(self, symbol) -> bool:
        """Check if we have an open Sweep-Reclaim (SR) position for this symbol.

        2026-06-05: SR runs on its own magic range (cfg.magic + 2000/2001) so
        momentum/FVG dispatchers are blind to it unless they explicitly check
        here. Used by open_trade_explicit guard and brain.py SR yield gates.
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        try:
            from config import SR_SUB_OFFSETS
            sr_magics = {int(cfg.magic) + off for off in SR_SUB_OFFSETS}
        except Exception:
            return False
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) in sr_magics for p in positions)

    def has_smabo_position(self, symbol) -> bool:
        """Check if we have an open SMA-Breakout (SMABO) position for this symbol.

        2026-06-21: SMABO runs on its own magic range (cfg.magic + 3000/3001),
        so momentum/FVG/SR dispatchers are blind to it unless they explicitly
        check here. Used by open_trade_explicit guard and brain.py SMABO yield
        gates. Mirrors has_sr_position / has_fvg_position exactly.
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        try:
            from config import SMABO_SUB_OFFSETS
            smabo_magics = {int(cfg.magic) + off for off in SMABO_SUB_OFFSETS}
        except Exception:
            return False
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) in smabo_magics for p in positions)

    def smabo_move_runner_to_be(self, symbol) -> bool:
        """Backtest parity ('SL->BE after TP1'): once the SMABO TP1 leg (+3000)
        has closed, move the runner leg (+3001) SL to breakeven. ONLY ever
        tightens (moves SL toward entry), never loosens — cannot increase risk.
        No ELC/TimeStop/PeakGiveback (those aren't in the validated backtest).
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        try:
            from config import SMABO_SUB_OFFSETS
            tp1_magic = int(cfg.magic) + SMABO_SUB_OFFSETS[0]
            run_magic = int(cfg.magic) + SMABO_SUB_OFFSETS[1]
        except Exception:
            return False
        try:
            positions = self.mt5.positions_get(symbol=symbol) or []
        except Exception:
            return False
        tp1_open = any(int(p.magic) == tp1_magic for p in positions)
        runner = next((p for p in positions if int(p.magic) == run_magic), None)
        if tp1_open or runner is None:
            return False   # TP1 leg still open (not filled yet) or no runner
        si = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if si is None or tick is None:
            return False
        digits = int(si.digits)
        point = float(si.point) or 1e-5
        min_gap = (float(si.trade_stops_level or 0) + 2) * point
        entry = float(runner.price_open)
        cur_sl = float(runner.sl)
        is_long = int(runner.type) == 0
        cur_px = float(tick.bid) if is_long else float(tick.ask)
        be = round(entry, digits)
        # Improvement + validity guards (respect broker min-stop from price).
        if is_long:
            if be <= cur_sl or be >= cur_px - min_gap:
                return False
        else:
            if be >= cur_sl or be <= cur_px + min_gap:
                return False
        req = {"action": int(6), "symbol": str(symbol), "position": int(runner.ticket),
               "sl": float(be), "tp": float(runner.tp), "magic": int(runner.magic)}
        try:
            r, _ = self._send_order(req, symbol, context="SMABO_BE")
        except Exception as e:
            log.debug("[SMABO %s] BE modify err: %s", symbol, e)
            return False
        if r and int(r.retcode) in (10009, 10025):
            log.info("[SMABO %s] runner SL -> BE %.5f after TP1 filled", symbol, be)
            return True
        return False

    def trail_trend_sl(self, symbol, new_sl, legs, min_gap, digits) -> int:
        """Chandelier trail for TREND legs — READ-FREE. In-process positions_get/
        symbol_info/tick return empty/None under bridge contention (proven), so
        the caller passes `legs` (dicts from the disk sync file, each with ticket/
        type/sl/tp/price_cur) + static min_gap/digits. The ONLY bridge touch is the
        action=6 modify (a write — those succeed when reads don't). ONLY tightens
        (up for long, down for short), never past min-stop. Returns #legs moved."""
        tgt = round(float(new_sl), int(digits))
        n = 0
        for p in legs:
            try:
                tk = int(p.get("ticket") or 0)
                if not tk:
                    continue
                is_long = int(p.get("type", 0)) == 0
                cur_sl = float(p.get("sl") or 0.0)
                px = float(p.get("price_cur") or 0.0)
                tp = float(p.get("tp") or 0.0)
            except Exception:
                continue
            if px <= 0:                       # no live price on disk yet → can't validate
                continue
            if is_long:
                if (cur_sl and tgt <= cur_sl) or tgt >= px - min_gap:
                    continue
            else:
                if (cur_sl and tgt >= cur_sl) or tgt <= px + min_gap:
                    continue
            req = {"action": int(6), "symbol": str(symbol), "position": tk,
                   "sl": float(tgt), "tp": float(tp), "magic": int(p.get("magic", 0))}
            try:
                r, _ = self._send_order(req, symbol, context="TREND_TRAIL")
            except Exception as e:
                log.warning("[TREND %s] trail modify err: %s", symbol, e)
                continue
            if r and int(r.retcode) in (10009, 10025):
                n += 1
            elif r:
                log.warning("[TREND %s] trail modify retcode=%s %s", symbol,
                            int(r.retcode), getattr(r, "comment", ""))
        return n

    def has_scalper_position(self, symbol) -> bool:
        """Open M1-scalper (SCALP) position? Own magic range (base+5000/+5001)."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        try:
            from config import SCALPER_SUB_OFFSETS
            mags = {int(cfg.magic) + off for off in SCALPER_SUB_OFFSETS}
        except Exception:
            return False
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) in mags for p in positions)

    def close_scalper_position(self, symbol, comment="SCALP_close"):
        """Close all SCALP-magic legs for symbol (used by the M1 time-stop)."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        try:
            from config import SCALPER_SUB_OFFSETS
            mags = {int(cfg.magic) + off for off in SCALPER_SUB_OFFSETS}
        except Exception:
            return False
        try:
            positions = self.mt5.positions_get(symbol=symbol) or []
        except Exception:
            return False
        any_closed = False
        for p in positions:
            if int(p.magic) not in mags:
                continue
            ctype = 1 if int(p.type) == 0 else 0
            tk = self.mt5.symbol_info_tick(symbol)
            if tk is None:
                continue
            cpx = float(tk.bid) if int(p.type) == 0 else float(tk.ask)
            req = {"action": int(1), "symbol": str(symbol), "volume": float(p.volume),
                   "type": int(ctype), "position": int(p.ticket), "price": cpx,
                   "deviation": int(_get_deviation(symbol)), "magic": int(p.magic),
                   "comment": str(comment), "type_filling": int(1), "type_time": int(0)}
            r, _ = self._send_order(req, symbol, context=f"SCALP_CLOSE_{comment}")
            if r and int(r.retcode) in (10009, 10008):
                any_closed = True
        if any_closed:
            with self._lock:
                if hasattr(self, "_scalper_entry_time"):
                    self._scalper_entry_time.pop(symbol, None)
            log.info("[SCALP %s] CLOSED (%s)", symbol, comment)
        return any_closed

    def trend_position_dir(self, symbol) -> int:
        """TREND book direction: +1 long / -1 short / 0 flat (magic base+6000/6001)."""
        cfg = symbol_cfg(symbol)
        if cfg is None:
            return 0
        try:
            from config import TREND_SUB_OFFSETS
            mags = {int(cfg.magic) + off for off in TREND_SUB_OFFSETS}
        except Exception:
            return 0
        positions = self.mt5.positions_get(symbol=symbol) or []
        for p in positions:
            if int(p.magic) in mags:
                return 1 if int(p.type) == 0 else -1
        return 0

    def has_trend_position(self, symbol) -> bool:
        return self.trend_position_dir(symbol) != 0

    def close_trend_position(self, symbol, comment="TREND_close"):
        """Close all TREND-magic legs for symbol (signal flip / reversal exit).
        2026-07-12: routes through _close_magic_legs, which adds a READ-FREE disk
        fallback so the close still fires when in-process reads fail under bridge
        contention (was the recurring BTC 'CLOSE FAILED / UNPROTECTED' bug — the
        reversal couldn't execute and BTC rode to its broker SL)."""
        cfg = symbol_cfg(symbol)
        if cfg is None:
            return False
        try:
            from config import TREND_SUB_OFFSETS
            mags = {int(cfg.magic) + off for off in TREND_SUB_OFFSETS}
        except Exception:
            return False
        return self._close_magic_legs(symbol, mags, comment, "TREND")

    def has_gold_smc_position(self, symbol) -> bool:
        """Open GOLD_SMC position? Own magic range (base+8000/+8001). Fails OPEN
        (returns False) on an in-proc None read — combined with the sync-daemon
        fail-closed guard in the brain, that closes the duplicate-open hole."""
        cfg = symbol_cfg(symbol)
        if cfg is None:
            return False
        try:
            from config import GOLD_SMC_SUB_OFFSETS
            mags = {int(cfg.magic) + off for off in GOLD_SMC_SUB_OFFSETS}
        except Exception:
            return False
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) in mags for p in positions)

    def close_gold_smc_position(self, symbol, comment="GSMC_close"):
        """Close GOLD_SMC legs for symbol (EMA9-trail runner exit). Routes through
        _close_magic_legs → READ-FREE disk fallback under bridge contention. Closes
        only the GOLD_SMC-magic legs; broker holds SL/TP2 on the rest."""
        cfg = symbol_cfg(symbol)
        if cfg is None:
            return False
        try:
            from config import GOLD_SMC_SUB_OFFSETS
            mags = {int(cfg.magic) + off for off in GOLD_SMC_SUB_OFFSETS}
        except Exception:
            return False
        return self._close_magic_legs(symbol, mags, comment, "GOLD_SMC")

    def has_imr_position(self, symbol) -> bool:
        """Open indices-MR position? Own magic range (base+7000/+7001)."""
        cfg = symbol_cfg(symbol)
        if cfg is None:
            return False
        try:
            from config import IMR_SUB_OFFSETS
            mags = {int(cfg.magic) + off for off in IMR_SUB_OFFSETS}
        except Exception:
            return False
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) in mags for p in positions)

    def open_imr_trade(self, symbol, lot, sl_atr_mult, atr, magic_off):
        """Fixed-lot single-leg LONG for indices-MR. SL = fill - sl_atr_mult*ATR
        (DISASTER stop, anchored on the actual fill). No TP — detector-driven exit.
        NOTE: IMR magics are excluded from all trail/ELC/BE dispatch by design."""
        cfg = symbol_cfg(symbol)
        if cfg is None:
            return False
        # DUP-OPEN GUARD (2026-07-10 review): a stale/empty sync file could make the
        # brain think IMR is flat and re-fire a fixed-lot LONG. Mirror the TREND
        # guard. (has_imr_position fails OPEN on in-proc None, but combined with the
        # sync-daemon fail-closed fix this closes the duplicate-open hole.)
        if self.has_imr_position(symbol):
            return False
        if self._daily_loss_blocked(symbol):
            log.warning("[IMR %s] open reject: daily loss limit reached — locked till EOD", symbol)
            return False
        si = tick = None                      # select + retry under bridge contention
        for _att in range(5):
            try:
                self.mt5.symbol_select(symbol, True)
                si = self.mt5.symbol_info(symbol)
                tick = self.mt5.symbol_info_tick(symbol)
            except Exception:
                si = tick = None
            if si is not None and tick is not None:
                break
            time.sleep(0.2 * (_att + 1))
        if si is None or tick is None:
            si2, tick2 = self._static_si_tick(symbol)   # read-free fallback (disk quote)
            if si2 is not None and tick2 is not None:
                si, tick = si2, tick2
                log.info("[IMR %s] read-free fallback: static specs + disk quote bid=%.2f", symbol, float(tick.bid))
            else:
                log.warning("[IMR %s] open reject after 5 tries (no disk fallback): si=%s tick=%s", symbol, si is not None, tick is not None)
                return False
        digits = int(si.digits)
        point = float(si.point) or 0.01
        px = float(tick.ask)
        sl = round(px - sl_atr_mult * atr, digits)
        min_gap = (float(si.trade_stops_level or 0) + 2) * point
        if px - sl < min_gap:
            sl = round(px - min_gap, digits)
        req = {"action": int(1), "symbol": str(symbol), "volume": float(lot),
               "type": int(0), "price": float(px), "sl": float(sl), "tp": 0.0,
               "deviation": int(_get_deviation(symbol)),
               "magic": int(cfg.magic) + int(magic_off), "comment": "IMR",
               "type_filling": int(1), "type_time": int(0)}
        try:
            r, _ = self._send_order(req, symbol, context="IMR_open")
        except Exception as e:
            log.warning("[IMR %s] open err: %s", symbol, e)
            return False
        return bool(r and int(r.retcode) in (10009, 10008))

    def close_imr_position(self, symbol, comment="IMR_close"):
        """Close all indices-MR legs for symbol (detector exit signal). Routes
        through _close_magic_legs → READ-FREE disk fallback under bridge contention
        (2026-07-12: was select+retry only, which bailed when reads failed)."""
        cfg = symbol_cfg(symbol)
        if cfg is None:
            return False
        try:
            from config import IMR_SUB_OFFSETS
            mags = {int(cfg.magic) + off for off in IMR_SUB_OFFSETS}
        except Exception:
            return False
        return self._close_magic_legs(symbol, mags, comment, "IMR")

    def has_fib50_position(self, symbol) -> bool:
        """Check if we have an open Fib-50 Pullback Continuation position.

        2026-06-21: FIB50 runs on its own magic range (cfg.magic + 4000/4001),
        so momentum/FVG/SR/SMABO dispatchers are blind to it unless they
        explicitly check here. Used by open_trade_explicit guard and brain.py
        FIB50 yield gates. Mirrors has_sr_position / has_smabo_position.
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        try:
            from config import FIB50_SUB_OFFSETS
            fib50_magics = {int(cfg.magic) + off for off in FIB50_SUB_OFFSETS}
        except Exception:
            return False
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) in fib50_magics for p in positions)

    def get_open_symbols(self) -> list:
        """Return list of symbols that currently have open positions (swing or scalp)."""
        open_syms = []
        for symbol, cfg in SYMBOLS.items():
            positions = self.mt5.positions_get(symbol=symbol)
            if positions is None:
                continue
            swing_magic = int(cfg.magic)
            scalp_magic = int(cfg.magic) + SCALP_MAGIC_OFFSET
            for p in positions:
                pm = int(p.magic)
                if pm == swing_magic or pm == scalp_magic:
                    open_syms.append(symbol)
                    break
        return open_syms

    def manage_trailing_sl(self, symbol):
        """
        Apply stepped trailing SL based on profit in R multiples.
        Handles 3 swing sub-positions + scalp.
        When Sub0 (TP1) auto-closes, moves Sub1+Sub2 SL to BE+offset.
        """
        cfg = SYMBOLS.get(symbol)
        # 2026-05-13 orphan support: cfg=None for disabled symbols, but the
        # broker may still hold positions on them. Trail them using ANY magic
        # rather than the cfg-derived magic set.
        is_orphan = cfg is None

        # Force position sync first — clears stale entry data if position closed externally
        # 2026-06-19: widened to cover FVG (+1000/+1001) and SR (+2000/+2001) magic ranges.
        # Previously this early-return fired whenever ONLY a non-swing/non-scalp ticket
        # was open, making the FVG/SR dispatch at lines 1978-1981 dead code. Direct
        # consequence: GER40.r SR trades (#507, #513) bled the full ~5R+ SL with zero
        # EarlyLossCut/PeakGiveback/TimeStop/BOS log lines. Mirrors the 2028f2a scalp fix.
        if (not self.has_position(symbol)
                and not self.has_scalp_position(symbol)
                and not self.has_fvg_position(symbol)
                and not self.has_sr_position(symbol)
                and not self.has_smabo_position(symbol)
                and not self.has_fib50_position(symbol)):
            return

        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return

        if is_orphan:
            # Derive magic from the open position itself
            base_magic = int(positions[0].magic)
            swing_magics = {base_magic + off for off in SUB_MAGIC_OFFSETS}
            scalp_magic = base_magic + SCALP_MAGIC_OFFSET
            scalp_magic_old = base_magic + 100
        else:
            base_magic = int(cfg.magic)
            swing_magics = {base_magic + off for off in SUB_MAGIC_OFFSETS}
            scalp_magic = base_magic + SCALP_MAGIC_OFFSET
            scalp_magic_old = base_magic + 100  # backward compat for positions opened with old offset

        # Detect which subs are still open
        open_subs = set()
        for pos in positions:
            pm = int(pos.magic)
            if pm in swing_magics:
                open_subs.add(pm - base_magic)

        # If Sub0 (TP1) already closed by broker, move remaining to BE+offset
        entry = self._entry_prices.get(symbol)
        sl_dist = self._entry_sl_dist.get(symbol, 0)
        if entry and sl_dist > 0 and 0 not in open_subs and len(open_subs) > 0:
            self._move_remaining_to_be(symbol, positions, entry, sl_dist, swing_magics)

        # Apply trail — regime-aware resolution (per-(sym,regime) > regime default
        # > per-symbol legacy > global). Sub2 runner has its own profile.
        # 2026-06-05: FVG positions now also routed through _apply_trail so they
        # inherit EarlyLossCut/PeakGiveback/TimeStop/BOS/VWAP_Cross + actual trail.
        # Uses FVG_TRAIL_STEPS (no lock below TP1=1.5R to avoid clipping TPs).
        try:
            from config import FVG_MAGIC_OFFSET, FVG_SUB_OFFSETS, FVG_TRAIL_STEPS
            fvg_magics = {base_magic + off for off in FVG_SUB_OFFSETS}
        except Exception:
            fvg_magics = set()
            FVG_TRAIL_STEPS = None

        # 2026-06-05: SR (sweep-reclaim) positions routed through _apply_trail
        # the same way FVG is — own magic range, own trail profile. Yields all
        # the EarlyLossCut/PeakGiveback/TimeStop/BOS/VWAP_Cross exits for free.
        try:
            from config import SR_SUB_OFFSETS, SR_TRAIL_STEPS
            sr_magics = {base_magic + off for off in SR_SUB_OFFSETS}
        except Exception:
            sr_magics = set()
            SR_TRAIL_STEPS = None

        for pos in positions:
            pos_magic = int(pos.magic)
            if pos_magic in swing_magics:
                sub_idx = pos_magic - base_magic
                if sub_idx == 2:
                    trail = SUB2_TRAIL_STEPS
                else:
                    trail = self._resolve_trail_steps(symbol)
                self._apply_trail(symbol, pos, trail, symbol)
            elif pos_magic == scalp_magic or pos_magic == scalp_magic_old:
                self._apply_trail(symbol, pos, SCALP_TRAIL_STEPS, symbol + "_scalp")
            elif pos_magic in fvg_magics and FVG_TRAIL_STEPS is not None:
                self._apply_trail(symbol, pos, FVG_TRAIL_STEPS, symbol + "_fvg")
            elif pos_magic in sr_magics and SR_TRAIL_STEPS is not None:
                self._apply_trail(symbol, pos, SR_TRAIL_STEPS, symbol + "_sr")

    def _move_remaining_to_be(self, symbol, positions, entry, sl_dist, swing_magics):
        """When TP1 sub closes, lock remaining subs at BE + 20% of SL."""
        si = self.mt5.symbol_info(symbol)
        if si is None:
            return
        point = float(si.point) if si.point else 0.00001
        digits = int(si.digits)
        be_offset = sl_dist * 0.2  # Lock 0.2R profit on remaining

        for pos in positions:
            if int(pos.magic) not in swing_magics:
                continue
            direction = "LONG" if int(pos.type) == 0 else "SHORT"
            current_sl = float(pos.sl)

            if direction == "LONG":
                be_sl = float(round(entry + be_offset, digits))
                if current_sl >= be_sl:
                    continue  # already past BE
            else:
                be_sl = float(round(entry - be_offset, digits))
                if current_sl > 0 and current_sl <= be_sl:
                    continue

            # Check min stop distance
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            min_dist = float(si.trade_stops_level) * point
            if direction == "LONG":
                be_sl = min(be_sl, float(tick.bid) - min_dist)
                if be_sl <= current_sl:
                    continue
            else:
                be_sl = max(be_sl, float(tick.ask) + min_dist)
                if current_sl > 0 and be_sl >= current_sl:
                    continue

            request = {
                "action": int(6),
                "symbol": str(symbol),
                "position": int(pos.ticket),
                "sl": float(round(be_sl, digits)),
                "tp": float(pos.tp),
            }
            result = self.mt5.order_send(request)
            if result and int(result.retcode) in (10009, 10008):
                _bm = symbol_cfg(symbol)
                log.info("[%s] TP1 HIT — moved Sub%d SL to BE+0.2R: %.5f",
                         symbol, int(pos.magic) - (int(_bm.magic) if _bm else 0), be_sl)

    # ════════════════════════════════════════════════════════════════════
    # 2026-06-05 EXIT-LOGIC HELPERS (bar-close guard, time-stop, BOS, VWAP)
    # ════════════════════════════════════════════════════════════════════
    @staticmethod
    def _floor_to_m15(ts: float) -> int:
        """Floor a unix timestamp to the start of its M15 bucket."""
        try:
            return int(ts) - (int(ts) % _M15_BAR_SEC)
        except Exception:
            return 0

    def _compute_bars_in_trade(self, pos) -> int:
        """M15 bars elapsed since position open (pos.time is unix seconds)."""
        try:
            open_ts = float(getattr(pos, "time", 0) or 0)
            if open_ts <= 0:
                return 0
            elapsed = time.time() - open_ts
            if elapsed <= 0:
                return 0
            return int(elapsed // _M15_BAR_SEC)
        except Exception:
            return 0

    def _compute_entry_swing(self, symbol, direction, anchor_time):
        """Find the last opposite swing pivot at-or-before anchor_time on M15.
        LONG  → last swing LOW within the prior _BOS_SWING_LOOKBACK bars.
        SHORT → last swing HIGH within the prior _BOS_SWING_LOOKBACK bars.
        A swing pivot needs to be strictly lower/higher than _BOS_SWING_WINDOW
        bars before AND after. Returns float price or None.
        """
        try:
            df = self.state.get_candles(symbol, 15)
            if df is None or len(df) < (_BOS_SWING_WINDOW * 2 + 2):
                return None
            h = df["high"].values.astype(float)
            l = df["low"].values.astype(float)
            # Anchor index: last bar whose time <= anchor_time. Fallback to last bar.
            anchor_idx = len(df) - 1
            try:
                t_col = df["time"]
                # Normalise to unix seconds (handles datetime or numeric)
                if hasattr(t_col.iloc[-1], "timestamp"):
                    times = np.array([float(x.timestamp()) for x in t_col], dtype=float)
                else:
                    times = t_col.values.astype(float)
                _matches = np.where(times <= float(anchor_time))[0]
                if len(_matches) > 0:
                    anchor_idx = int(_matches[-1])
            except Exception:
                pass
            # Scan bars in window [anchor_idx - _BOS_SWING_LOOKBACK, anchor_idx]
            # The pivot bar itself needs _BOS_SWING_WINDOW bars on each side, so
            # candidate range is [start + W, anchor_idx - W].
            w = _BOS_SWING_WINDOW
            start = max(w, anchor_idx - _BOS_SWING_LOOKBACK)
            end = anchor_idx - w
            if end < start:
                return None
            swing = None
            for i in range(end, start - 1, -1):  # walk backward — most recent first
                if direction == "LONG":
                    pivot = l[i]
                    left = l[i - w:i]
                    right = l[i + 1:i + 1 + w]
                    if len(left) == w and len(right) == w \
                            and pivot < left.min() and pivot < right.min():
                        swing = float(pivot)
                        break
                else:
                    pivot = h[i]
                    left = h[i - w:i]
                    right = h[i + 1:i + 1 + w]
                    if len(left) == w and len(right) == w \
                            and pivot > left.max() and pivot > right.max():
                        swing = float(pivot)
                        break
            return swing
        except Exception as e:
            log.debug("[%s] _compute_entry_swing failed: %s", symbol, e)
            return None

    def _get_session_vwap(self, symbol):
        """Volume-weighted average price over the most recent M15 bars (session
        proxy). Returns float or None on insufficient data.
        Mirrors OrderFlowIntel._compute_vwap (agent/order_flow.py:298-313).
        """
        try:
            df = self.state.get_candles(symbol, 15)
            if df is None or len(df) < 5:
                return None
            # Session-proxy lookback: last 26 M15 bars ~= 6.5h (one US RTH session).
            n = min(26, len(df))
            h = df["high"].values.astype(float)[-n:]
            l = df["low"].values.astype(float)[-n:]
            c = df["close"].values.astype(float)[-n:]
            try:
                v = df["tick_volume"].values.astype(float)[-n:]
            except Exception:
                try:
                    v = df["volume"].values.astype(float)[-n:]
                except Exception:
                    v = np.ones(n, dtype=float)
            typical = (h + l + c) / 3.0
            total_v = float(np.sum(v))
            if total_v <= 0:
                return float(np.mean(typical))
            return float(np.sum(typical * v) / total_v)
        except Exception as e:
            log.debug("[%s] _get_session_vwap failed: %s", symbol, e)
            return None

    def _apply_trail(self, symbol, pos, trail_steps, tracking_key):
        """Apply trailing SL logic for a single position using the given trail profile."""
        direction = "LONG" if int(pos.type) == 0 else "SHORT"

        si = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if si is None or tick is None:
            return

        point = float(si.point) if si.point else 0.00001
        digits = int(si.digits)
        current_sl = float(pos.sl)
        entry = self._entry_prices.get(tracking_key, float(pos.price_open))
        sl_dist = self._entry_sl_dist.get(tracking_key, 0)

        # 2026-06-19: strategy-aware close router. close_position() only flattens
        # swing magics (SUB_MAGIC_OFFSETS) — calling it on an SR/FVG-only ticket
        # would be a no-op and let the full SL hit. Route through the per-strategy
        # close method so EarlyLossCut / PeakGiveback / TimeStop / BOS / HardCap
        # / VWAP exits actually flatten the right ticket.
        def _close_for_tracking_key(_comment):
            if isinstance(tracking_key, str) and tracking_key.endswith("_fvg"):
                return self.close_fvg_position(symbol, comment=_comment)
            if isinstance(tracking_key, str) and tracking_key.endswith("_sr"):
                return self._close_sr_position(symbol, comment=_comment)
            return self.close_position(symbol, comment=_comment)

        if sl_dist <= 0:
            # FIX 2026-04-29: previous fallback used `abs(entry - current_sl)` which is
            # the *trailed* SL distance — after trail moves SL to BE, this collapses to ~0
            # making profit_r explode (logged peaks like 133R, 189R). Use ATR-based
            # estimate instead so profit_r stays sane.
            atr_est = self._get_atr(symbol)
            if atr_est > 0:
                # Use ATR × typical SL multiplier as a stand-in
                sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, ATR_SL_MULTIPLIER)
                sl_dist = float(atr_est) * float(sl_mult)
            else:
                sl_dist = abs(entry - current_sl)
            if sl_dist <= 0:
                return

        cur_price = float(tick.bid) if direction == "LONG" else float(tick.ask)
        profit_dist = (cur_price - entry) if direction == "LONG" else (entry - cur_price)
        profit_r = profit_dist / sl_dist if sl_dist > 0 else 0

        # FIX 2026-04-29: clamp peak_r to sane max so ATR=0 / sl_dist=0 glitches can't
        # poison the trail state. 10R is well above any legitimate runner peak; anything
        # above means the math broke.
        if profit_r > 10.0 or profit_r < -10.0:
            profit_r = max(-10.0, min(10.0, profit_r))

        # ── PROFIT RATCHET: track peak R and enforce profit floor ──
        # Once trade hits 1R+, SL can NEVER go below 0.3R profit
        # Once trade hits 2R+, SL can NEVER go below 0.7R profit
        peak_key = f"_peak_r_{tracking_key}"
        prev_peak = getattr(self, '_peak_profit_r', {}).get(tracking_key, 0)
        # Don't let an existing inflated peak (>10R from earlier glitch) survive
        if prev_peak > 10.0:
            prev_peak = 10.0
        cur_peak = max(prev_peak, profit_r)
        self._peak_profit_r[tracking_key] = cur_peak

        # ── 2026-06-18 Tier 1 #4: position R-multiple live telemetry ──
        # Rate-limited dashboard push of (symbol, profit_r, peak_r).
        # Fire-and-forget; never blocks the exit-management loop.
        try:
            self._push_position_r_telemetry(symbol, profit_r, cur_peak, status="open")
        except Exception:
            pass

        # ── PEAK-GIVEBACK CIRCUIT BREAKER (CONSERVATIVE 2026-05-12) ──
        # If trade was at +0.7R or more, then current profit drops below
        # 50% of peak → close immediately at market. Don't wait for trail SL
        # to hit at potentially worse fill. Pure profit preservation.
        try:
            from config import PEAK_GIVEBACK_ENABLED, PEAK_GIVEBACK_TRIGGER_R, PEAK_GIVEBACK_FRAC
            # 2026-05-19: per-symbol override for high-PF symbols. Letting
            # small peaks ride beats prematurely closing at +0.45R on a
            # 0.92R peak (live SWI20 $20→$9 issue).
            try:
                from config import PEAK_GIVEBACK_PER_SYMBOL
                _trig, _frac = PEAK_GIVEBACK_PER_SYMBOL.get(
                    symbol, (PEAK_GIVEBACK_TRIGGER_R, PEAK_GIVEBACK_FRAC))
            except Exception:
                _trig, _frac = PEAK_GIVEBACK_TRIGGER_R, PEAK_GIVEBACK_FRAC
            if (PEAK_GIVEBACK_ENABLED and cur_peak >= _trig
                    and profit_r < cur_peak * _frac):
                log.warning(
                    "[%s] PEAK-GIVEBACK EXIT: peak=%.2fR current=%.2fR "
                    "(retraced %.0f%% from peak, trigger=%.1fR frac=%.2f) — closing at market",
                    symbol, cur_peak, profit_r,
                    (1 - profit_r / max(cur_peak, 0.01)) * 100, _trig, _frac)
                _close_for_tracking_key("PeakGiveback")
                return  # exit early, position closed
        except Exception as e:
            log.debug("[%s] peak-giveback check failed: %s", symbol, e)

        # ── HARD DOLLAR CAP (2026-05-14) — catastrophic-outlier guard ──
        # Closes:
        #   (a) ANY single position whose unrealized loss exceeds HARD_DOLLAR_CAP_PCT
        #       of equity. Catches gap-through losses (XAGUSD -17R, COPPER -36R).
        #   (b) PORTFOLIO aggregate unrealized loss > 2.5× per-position threshold.
        #       Audit 2026-05-14: 10 positions × -1.9% would silently DD -19% with
        #       only per-position cap. Close ALL when aggregate breach occurs.
        try:
            from config import HARD_DOLLAR_CAP_ENABLED, HARD_DOLLAR_CAP_PCT
            if HARD_DOLLAR_CAP_ENABLED:
                acct_state = self.state.get_agent_state() if self.state else {}
                equity = float(acct_state.get("equity", 0)) or float(acct_state.get("balance", 0))
                if equity <= 0:
                    try:
                        acc = self.client.account_info()
                        equity = float(acc.equity) if acc else 1000.0
                    except Exception:
                        equity = 1000.0
                max_loss = equity * HARD_DOLLAR_CAP_PCT
                unrealized = float(getattr(pos, "profit", 0))
                if unrealized < -max_loss:
                    log.warning(
                        "[%s] HARD-CAP EXIT: unrealized $%.2f < -$%.2f (%.1f%% of $%.0f equity)",
                        symbol, unrealized, max_loss, HARD_DOLLAR_CAP_PCT * 100, equity)
                    _close_for_tracking_key("HardDollarCap")
                    return
                # Portfolio aggregate check — 1.5× single-position threshold across ALL.
                # 2026-06-02: was 2.5×; tightened to 1.5× to catch correlated
                # risk-off clusters earlier (18:57 UTC 4-symbol cluster on
                # 2026-06-01 hit ~-$28 / -0.6% before any pos breached own cap).
                try:
                    all_pos = self.client.positions_get() or []
                    agg_unrealized = sum(float(getattr(p, "profit", 0)) for p in all_pos)
                    agg_threshold = 1.5 * max_loss
                    if agg_unrealized < -agg_threshold:
                        log.warning(
                            "PORTFOLIO HARD-CAP: aggregate unrealized $%.2f < -$%.2f "
                            "(%.1f%% of $%.0f equity, %d positions) — closing ALL",
                            agg_unrealized, agg_threshold, HARD_DOLLAR_CAP_PCT * 150, equity, len(all_pos))
                        for p in all_pos:
                            try:
                                self.close_position(p.symbol, comment="PortfolioHardCap")
                            except Exception:
                                pass
                        return
                except Exception:
                    pass
        except Exception as e:
            log.debug("[%s] hard-cap check failed: %s", symbol, e)

        # ── TIME-STOP — 2026-06-05 ──
        # Close stalled trades that never made meaningful progress. Research:
        # ATAS / KJTradingSystems (Davey 567k-backtest) — time stops cut avg
        # loss without harming avg win when conditioned on no-progress at
        # peak < X·R after N M15 bars. Fires BEFORE EarlyLossCut because a
        # stalled-at-zero trade isn't waiting on a -0.8R signal.
        try:
            if TIME_STOP_ENABLED:
                bars_in_trade = self._compute_bars_in_trade(pos)
                if bars_in_trade >= TIME_STOP_BARS and cur_peak < TIME_STOP_MIN_PEAK_R:
                    log.warning(
                        "[%s] TIME-STOP: bars=%d peak=%.2fR < %.2fR — closing",
                        symbol, bars_in_trade, cur_peak, TIME_STOP_MIN_PEAK_R)
                    _close_for_tracking_key("TimeStop_NoProgress")
                    return
        except Exception as e:
            log.debug("[%s] time-stop check failed: %s", symbol, e)

        # ── BOS INVALIDATION — 2026-06-05 ──
        # Exit when the M15 close breaches the protected swing pivot recorded
        # at entry time. Lazily computed on the first cycle after open.
        try:
            if BOS_INVALIDATION_ENABLED:
                ticket = int(getattr(pos, "ticket", 0) or 0)
                entry_swing = self._entry_swings.get(ticket)
                if entry_swing is None and ticket > 0:
                    # Lazy init: compute swing using the position's open time
                    # as anchor (works for both fresh opens and restart recovery).
                    swing_px = self._compute_entry_swing(
                        symbol, direction,
                        float(getattr(pos, "time", time.time()) or time.time()))
                    if swing_px is not None:
                        entry_swing = {
                            "swing": float(swing_px),
                            "direction": direction,
                            "anchor_time": float(getattr(pos, "time", time.time()) or time.time()),
                        }
                        self._entry_swings[ticket] = entry_swing
                        log.info(
                            "[%s] BOS swing recorded for ticket %d: %s @ %.5f",
                            symbol, ticket, direction, swing_px)
                if entry_swing is not None:
                    # Use the latest CLOSED M15 bar's close to avoid intra-bar wick noise
                    df = self.state.get_candles(symbol, 15)
                    m15_close = None
                    if df is not None and len(df) >= 2:
                        try:
                            m15_close = float(df["close"].iloc[-2])
                        except Exception:
                            m15_close = None
                    if m15_close is not None:
                        breach = (direction == "LONG" and m15_close < entry_swing["swing"]) \
                              or (direction == "SHORT" and m15_close > entry_swing["swing"])
                        if breach:
                            log.warning(
                                "[%s] BOS_INVALIDATION: %s m15_close=%.5f breached swing=%.5f — closing",
                                symbol, direction, m15_close, entry_swing["swing"])
                            self._entry_swings.pop(ticket, None)
                            _close_for_tracking_key("BOS_Invalidation")
                            return
        except Exception as e:
            log.debug("[%s] BOS-invalidation check failed: %s", symbol, e)

        # ── EARLY-LOSS-CUT (CONSERVATIVE 2026-05-12) ──
        # If trade is at -0.5R or worse AND profit_r hasn't been positive
        # in last N=10 cycles → close at market. Avoids the slippage tax on
        # full SL hit. Better to exit at -0.5R than -1.7R after slippage.
        try:
            from config import EARLY_EXIT_ENABLED, EARLY_EXIT_TRIGGER_R, EARLY_EXIT_CYCLES
            # 2026-05-14: TIERED early-loss-cut — react faster as loss deepens.
            # Previously 60-cycle wait at any threshold meant gap losses ran 2-17×
            # their intended -0.5R cap (XAUUSD -3.87R, XAGUSD -17.17R, DJ30 -1.86R).
            # New tiers:
            #   profit_r <= -0.5R: wait 60 cycles (30s) — slow bleed
            #   profit_r <= -1.0R: wait 10 cycles (5s)  — clearly losing
            #   profit_r <= -1.5R: close IMMEDIATELY    — catastrophic / gap
            # 2026-06-02: BUG FIX — outer `cur_peak < 0.3` gate created a dead
            # zone where any trade that ever peaked >=0.3R got NO downside
            # protection (EarlyLossCut disabled, PeakGiveback armed only at
            # >=0.7R). XAU 18:57 ran -3.06R / -$15.33 through this hole.
            # Fix: per-tier peak gate. Catastrophic (-1.5R) ALWAYS fires.
            if EARLY_EXIT_ENABLED and profit_r <= EARLY_EXIT_TRIGGER_R:
                if not hasattr(self, '_loss_streak'):
                    self._loss_streak = {}
                # 2026-05-26 audit fix: race bug — all 3 swing subs share
                # tracking_key=symbol → counter incremented 3× per cycle →
                # EARLY_EXIT_CYCLES=20 effectively fired at ~7 cycles instead
                # of 20. Key streak by (symbol, magic) so each sub increments
                # its OWN counter once per cycle.
                streak_key = f"{tracking_key}#{int(pos.magic)}"
                self._loss_streak[streak_key] = self._loss_streak.get(streak_key, 0) + 1
                streak = self._loss_streak[streak_key]
                # Determine tier — each tier has its own peak gate.
                # T3 ALWAYS fires (gap / catastrophic); T2 fires unless trade
                # peaked above PEAK_GIVEBACK trigger (0.7R, handled elsewhere);
                # T1 keeps original intent of letting briefly-profitable trades
                # ride out a -0.5R retrace.
                # 2026-06-03 CTO audit (A10): T1-SLOW fired 80×/30d for -$232 —
                # biggest exit-loss category. Recovery analysis showed BTCUSD
                # 67% / SPI200.r 33% recovery rate (would have come back) vs
                # 0% for USDCAD/ETHUSD/EURUSD/UK100/SP500. Add:
                #   (1) per-symbol base wait override for high-WR symbols
                #   (2) peak-conditional extension: if trade briefly went
                #       positive (peak >= 0.1R), the trade is "breathing" —
                #       double the wait to give it room.
                T1_WAIT_PER_SYMBOL = {
                    "BTCUSD":   30,   # 67% recovery — too aggressive at 20
                    "SPI200.r": 30,   # 33% recovery
                    "XAUUSD":   30,   # 25% recovery + biggest dollar loss
                }
                _base_t1 = T1_WAIT_PER_SYMBOL.get(symbol, EARLY_EXIT_CYCLES)
                # 2026-06-17 user req: ELC must catch ALL trades, never lose
                # whole 1R. Removed cur_peak gates from T1 and T2 — every trade
                # at -0.8R/-1.0R/-1.5R now triggers regardless of prior peak.
                # PEAK_GIVEBACK handles peak>=0.7R retraces via separate path.
                if profit_r <= -1.3:
                    # 2026-06-19: tightened -1.5 → -1.3 per 14d journal diagnostic.
                    # 12 of 14d losses bled past -2R = $188 of total losses ($488).
                    # Tightening catches them earlier; FVG/SR bypass fix (this
                    # session) also covers the strategy-keyed bypass class.
                    wait_required = 0   # immediate close — catastrophic
                    tier = "T3-IMMEDIATE"
                elif profit_r <= -1.0:
                    # T2-FAST: hard SL stop net. No peak gate.
                    wait_required = 10  # 5s wait
                    tier = "T2-FAST"
                elif profit_r <= -0.8:
                    # T1-SLOW: slow-bleed catcher. No peak gate.
                    # Peak-conditional double-wait kept (breathing room for
                    # trades that briefly went positive — still get cut,
                    # just with more recovery time).
                    wait_required = _base_t1
                    if cur_peak >= 0.1:
                        wait_required = wait_required * 2
                    tier = "T1-SLOW"
                else:
                    # No tier matched (profit_r above all thresholds) — clear streak.
                    self._loss_streak.pop(streak_key, None)
                    wait_required = None
                    tier = None
                if tier is not None and streak >= wait_required:
                    # 2026-06-05: bar-close guard for T1 ONLY.
                    # PaperToProfit 87-stop study + Davey 567k-backtest: never
                    # time-cut inside the entry candle — most wicks reverse.
                    # T2/T3 are catastrophic and EXEMPT from this guard.
                    if (tier == "T1-SLOW" and EARLY_EXIT_REQUIRE_BAR_CLOSE):
                        try:
                            _open_ts = float(getattr(pos, "time", 0) or 0)
                            if _open_ts > 0:
                                _now_bucket = self._floor_to_m15(time.time())
                                _entry_bucket = self._floor_to_m15(_open_ts)
                                if _now_bucket == _entry_bucket:
                                    log.debug(
                                        "[%s] T1-SLOW skipped: still inside entry M15 bar "
                                        "(open=%d now=%d) — wait for bar close",
                                        symbol, _entry_bucket, _now_bucket)
                                    # Don't reset the streak — when the bar closes
                                    # next cycle the condition will fire normally.
                                    return
                        except Exception as _e:
                            log.debug("[%s] T1 bar-close guard failed: %s", symbol, _e)
                    log.warning(
                        "[%s] EARLY-LOSS-CUT %s: profit_r %.2fR for %d cycles "
                        "(peak %.2fR) — closing to cap loss",
                        symbol, tier, profit_r, streak, cur_peak)
                    _close_for_tracking_key(f"EarlyLossCut_{tier}")
                    # Reset all sub-streaks for this symbol on close
                    self._loss_streak = {k: v for k, v in self._loss_streak.items()
                                         if not k.startswith(f"{tracking_key}#")}
                    return
            else:
                if hasattr(self, '_loss_streak'):
                    streak_key = f"{tracking_key}#{int(pos.magic)}"
                    self._loss_streak.pop(streak_key, None)
        except Exception as e:
            log.debug("[%s] early-loss-cut check failed: %s", symbol, e)

        atr = self._get_atr(symbol)
        if atr <= 0:
            atr = sl_dist

        # Adaptive trail: scale by current ATR vs 50-bar average (if enabled for this symbol)
        trail_scale = 1.0
        if SMART_ENTRY_MODE.get(symbol, {}).get("adaptive_trail", False):
            atr_avg = self._get_atr_avg(symbol)
            if atr_avg > 0 and atr > 0:
                ratio = atr / atr_avg
                trail_scale = max(0.6, min(1.5, ratio))

        new_sl = None
        action = ""

        # RL trail adjustments for this symbol
        rl_adj = self._rl_trail_adj.get(tracking_key, self._rl_trail_adj.get(symbol, {}))
        trail_tightness_mult = rl_adj.get("trail_tightness_mult", 1.0)
        lock_threshold_mult = rl_adj.get("lock_threshold_mult", 1.0)
        be_threshold_mult = rl_adj.get("be_threshold_mult", 1.0)

        # ── MOMENTUM-ADAPTIVE TRAIL (feature 2 v2 + 2026-05-14 dynamic enhancements) ──
        # 2026-05-11 deep tune: HIGH momentum = WIDER trail (1.5x) + DELAYED
        # lock thresholds (1.5x — BE at 0.75R instead of 0.5R). LOW momentum
        # = tighter both. Stacks multiplicatively with RL adj. Walk-forward
        # 5-fold confirmed +24.3% vs baseline (11/19 ROBUST, 1 OVERFIT).
        #
        # 2026-05-14 dynamic enhancements (real-time market reads):
        #  E1: RSI exhaustion tightener — LONG+RSI≥72 rising (or SHORT+RSI≤28 falling) → 0.65×
        #  E2: Score-velocity override — momentum score dropped ≥0.15 from prior peak → cap 0.7×
        #  E3: Volume-confirmation gate — widening allowed only if volume ≥1.2× 20-bar avg
        #  Safety clamp: lock_threshold_mult and be_threshold_mult capped at 2.0
        momentum_lock_mult = 1.0
        try:
            from config import MOMENTUM_TRAIL_ADAPTIVE_ENABLED
            if MOMENTUM_TRAIL_ADAPTIVE_ENABLED and self.state is not None:
                from signals.momentum_signal import (
                    compute_momentum, trail_multiplier, lock_threshold_mult as _mom_lock,
                )
                ind = self.state.get_indicators(symbol) or {}
                df = self.state.get_candles(symbol, 60)
                mom = compute_momentum(ind, df)
                base_mult = trail_multiplier(mom)

                # E3: Volume-confirmation gate — if we're about to WIDEN the trail
                # (mult > 1.0), require volume confirmation. Otherwise cap at 1.15×.
                if base_mult > 1.0 and df is not None and "tick_volume" in getattr(df, "columns", []):
                    try:
                        vol_now = float(df["tick_volume"].iloc[-1])
                        vol_avg = float(df["tick_volume"].tail(20).mean())
                        if vol_avg > 0 and vol_now < 1.2 * vol_avg:
                            base_mult = min(base_mult, 1.15)
                    except Exception:
                        pass

                trail_tightness_mult *= base_mult
                momentum_lock_mult = _mom_lock(mom)
                lock_threshold_mult *= momentum_lock_mult
                be_threshold_mult *= momentum_lock_mult

                # E1: RSI exhaustion — overbought LONG or oversold SHORT AND
                # accelerating → tighten trail. Track per-symbol prev RSI since
                # state.get_indicators only exposes the current scalar.
                rsi = float(ind.get("rsi") or 0)
                if not hasattr(self, "_rsi_prev"):
                    self._rsi_prev = {}
                rsi_prev = float(self._rsi_prev.get(tracking_key, rsi) or rsi)
                if rsi > 0:
                    if direction == "LONG" and rsi >= 72 and rsi >= rsi_prev:
                        trail_tightness_mult *= 0.65
                    elif direction == "SHORT" and rsi <= 28 and rsi <= rsi_prev:
                        trail_tightness_mult *= 0.65
                    self._rsi_prev[tracking_key] = rsi

                # E2: Score-velocity — momentum score dropped from prior tick → tighten.
                # 2026-05-14 guard: require score_now > 0 so a momentary state hiccup
                # (empty ind dict → DEAD regime → score=0) doesn't false-fire.
                if not hasattr(self, "_mom_score_prev"):
                    self._mom_score_prev = {}
                score_prev = float(self._mom_score_prev.get(tracking_key, 0) or 0)
                score_now = float(mom.get("score", 0) or 0)
                if score_now > 0 and score_prev >= 0.65 and score_now < (score_prev - 0.15):
                    trail_tightness_mult = min(trail_tightness_mult, 0.7)
                    lock_threshold_mult = min(lock_threshold_mult, 0.85)
                    be_threshold_mult = min(be_threshold_mult, 0.85)
                # Only update prev when we have a real read (skip score=0 hiccups)
                if score_now > 0:
                    self._mom_score_prev[tracking_key] = score_now

                # Safety clamp — stacked RL × momentum × velocity could otherwise push
                # BE/lock thresholds 3-4× and let winners give back too much.
                lock_threshold_mult = min(lock_threshold_mult, 2.0)
                be_threshold_mult = min(be_threshold_mult, 2.0)
        except Exception as e:
            log.debug("momentum trail mult failed for %s: %s", symbol, e)

        # ── VWAP-CROSS RUNNER EXIT — 2026-06-05 ──
        # For index symbols once we're in runner territory (>+1R), exit if the
        # current price crosses session VWAP against the position. Below 1R the
        # normal trail loop handles SL motion. Applies ONLY to VWAP_GATE_SYMBOLS.
        try:
            if symbol in VWAP_GATE_SYMBOLS and profit_r > 1.0:
                _vwap = self._get_session_vwap(symbol)
                if _vwap is not None and _vwap > 0:
                    _crossed = (direction == "LONG" and cur_price < _vwap) \
                            or (direction == "SHORT" and cur_price > _vwap)
                    if _crossed:
                        log.warning(
                            "[%s] VWAP_CROSS_EXIT: %s price=%.5f crossed vwap=%.5f "
                            "(profit=%.2fR runner zone) — closing",
                            symbol, direction, cur_price, _vwap, profit_r)
                        _close_for_tracking_key("VWAP_Cross_Exit")
                        return
        except Exception as e:
            log.debug("[%s] VWAP-cross check failed: %s", symbol, e)

        for r_threshold, step_type, param in trail_steps:
            # Apply RL threshold multipliers per step type
            effective_threshold = r_threshold
            if step_type == "lock":
                effective_threshold = r_threshold * lock_threshold_mult
            elif step_type == "be":
                effective_threshold = r_threshold * be_threshold_mult

            if profit_r >= effective_threshold:
                if step_type == "trail":
                    trail_dist = param * atr * trail_scale * trail_tightness_mult
                    new_sl = (cur_price - trail_dist) if direction == "LONG" else (cur_price + trail_dist)
                    if profit_r >= 1.5:
                        floor = entry + 0.5 * sl_dist if direction == "LONG" else entry - 0.5 * sl_dist
                        if direction == "LONG":
                            new_sl = max(new_sl, floor)
                        else:
                            new_sl = min(new_sl, floor)
                    action = f"TRAIL_{param}ATR@{profit_r:.1f}R"
                elif step_type == "lock":
                    new_sl = entry + param * sl_dist if direction == "LONG" else entry - param * sl_dist
                    action = f"LOCK_{param}R@{profit_r:.1f}R"
                elif step_type == "be":
                    new_sl = entry + 2 * point if direction == "LONG" else entry - 2 * point
                    action = f"BE@{profit_r:.1f}R"
                elif step_type == "reduce_sl":
                    # Reduce max loss: move SL to entry - param * sl_dist (e.g. 0.7 = 70% of original SL)
                    new_sl = entry - param * sl_dist if direction == "LONG" else entry + param * sl_dist
                    action = f"REDUCE_SL_{param}@{profit_r:.1f}R"
                break

        if new_sl is None:
            return

        # ── PROFIT RATCHET: enforce minimum profit floor based on peak ──
        # Peak >= 2R → floor at 0.7R; Peak >= 1R → floor at 0.3R
        # V5 tuned: looser ratchet lets winners run further (0.2/0.5 vs 0.3/0.7)
        if cur_peak >= 2.0:
            ratchet_floor = entry + 0.5 * sl_dist if direction == "LONG" else entry - 0.5 * sl_dist
        elif cur_peak >= 1.0:
            ratchet_floor = entry + 0.2 * sl_dist if direction == "LONG" else entry - 0.2 * sl_dist
        else:
            ratchet_floor = None

        if ratchet_floor is not None:
            if direction == "LONG":
                new_sl = max(new_sl, ratchet_floor)
            else:
                new_sl = min(new_sl, ratchet_floor)

        min_dist = float(si.trade_stops_level) * point
        if direction == "LONG":
            new_sl = min(new_sl, float(tick.bid) - min_dist)
            should_move = new_sl > current_sl
        else:
            new_sl = max(new_sl, float(tick.ask) + min_dist)
            should_move = new_sl < current_sl or current_sl == 0

        if not should_move:
            return

        new_sl_rounded = float(round(new_sl, digits))
        if new_sl_rounded == float(round(current_sl, digits)):
            return

        request = {
            "action": int(6),              # TRADE_ACTION_SLTP
            "symbol": str(symbol),
            "position": int(pos.ticket),
            "sl": float(new_sl_rounded),
            "tp": float(pos.tp),
        }

        result = self.mt5.order_send(request)
        if result and int(result.retcode) in (10009, 10008):
            log.info("[%s] SL MOVED %s: %.5f -> %.5f", symbol, action, current_sl, new_sl_rounded)
        elif result and int(result.retcode) not in (10025,):
            log.warning("[%s] SL modify failed [%d]: %s", symbol, int(result.retcode), result.comment)

    def has_position(self, symbol) -> bool:
        """Check if we have any open sub-position for this symbol.
        Syncs internal tracking with MT5 reality to prevent drift.

        2026-05-13: orphan support — if cfg is None (symbol removed from
        SYMBOLS dict) but a position still exists on broker, return True
        so manage_trailing_sl gets called (it has its own orphan handling).
        """
        cfg = SYMBOLS.get(symbol)
        try:
            positions = self.mt5.positions_get(symbol=symbol)
            if positions is None:
                with self._lock:
                    return symbol in self._directions and self._directions[symbol] != "FLAT"
            # Orphan check: cfg gone but broker has positions → manage them
            if cfg is None:
                return bool(positions)
            valid_magics = {int(cfg.magic) + off for off in SUB_MAGIC_OFFSETS}
            mt5_has = any(int(p.magic) in valid_magics for p in positions)

            # Sync internal tracking with MT5 reality
            if not mt5_has:
                with self._lock:
                    if symbol in self._directions:
                        # Capture direction + PnL sign BEFORE clearing so the
                        # brain's cooldown logic can route win→short-and-same-dir-only,
                        # loss→long-and-both-dirs. (2026-05-11 asymmetric cooldown.)
                        closed_direction = self._directions.get(symbol, "FLAT")
                        entry_price = self._entry_prices.get(symbol, 0.0)
                        last_peak = (self._peak_profit_r.get(symbol, 0.0)
                                     if hasattr(self, '_peak_profit_r') else 0.0)
                        # win heuristic: peak_r > 0.5 OR last broker tick favored us.
                        # We can't poll current tick safely here (may be in lock-only
                        # cycle). Use peak_r as a proxy — if the trade ever reached
                        # +0.5R or more before close, treat as WIN. SL hits typically
                        # have peak_r close to 0.
                        was_win = float(last_peak) >= 0.5

                        log.info("[%s] Position closed externally — clearing internal tracking (dir=%s peak_r=%.2f win=%s)",
                                 symbol, closed_direction, float(last_peak), was_win)
                        # 2026-05-14: arm 30s same-cycle re-entry lockout for broker-side
                        # closes (TP1 hit, manual close, SL hit) — close_position() path
                        # already arms this, but the external-detection path didn't.
                        if not hasattr(self, "_just_closed"):
                            self._just_closed = {}
                        self._just_closed[symbol] = __import__('time').time()
                        # 2026-05-14: snapshot actual dollar_risk for correct R-multiple
                        # at record time. Brain previously used intended risk (post-multipliers
                        # ~$1) leading to fake -12R recordings on real -0.7R losses.
                        try:
                            _actual_risk = float(getattr(self, "_entry_dollar_risk", {}).get(symbol, 0) or 0)
                            if _actual_risk > 0:
                                if not hasattr(self, "_last_close_dollar_risk"):
                                    self._last_close_dollar_risk = {}
                                self._last_close_dollar_risk[symbol] = _actual_risk
                        except Exception:
                            pass
                        # 2026-05-26 audit fix: snapshot entry_price BEFORE pop so
                        # downstream LevelMemory/journal sync gets real price not 0.
                        # Was the cause of 65/85 trades having entry_price=0 in journal.
                        _ep_snap = float(self._entry_prices.get(symbol, 0) or 0)
                        if _ep_snap > 0:
                            if not hasattr(self, "_last_close_entry_price"):
                                self._last_close_entry_price = {}
                            self._last_close_entry_price[symbol] = _ep_snap
                        self._entry_prices.pop(symbol, None)
                        self._entry_sl_dist.pop(symbol, None)
                        self._entry_dollar_risk.pop(symbol, None)
                        self._directions.pop(symbol, None)
                        # 2026-06-05: clear BOS swing entries for any ticket
                        # belonging to this symbol (we can't reach the ticket
                        # list since positions returned None — sweep the dict).
                        try:
                            cfg_ext = SYMBOLS.get(symbol)
                            _valid = ({int(cfg_ext.magic) + off for off in SUB_MAGIC_OFFSETS}
                                      if cfg_ext is not None else None)
                            # Best-effort: drop entries whose direction matches
                            # what we just cleared. Safer than orphan-leak; bounded
                            # by dict size (max ~tens of tickets).
                            for _tk in list(self._entry_swings.keys()):
                                _es = self._entry_swings.get(_tk)
                                if _es is None:
                                    continue
                                if _es.get("direction") == closed_direction:
                                    # Could be from another symbol — narrow by
                                    # confirming no live position holds this ticket.
                                    try:
                                        _alive = self.mt5.positions_get(ticket=int(_tk))
                                    except Exception:
                                        _alive = None
                                    if not _alive:
                                        self._entry_swings.pop(_tk, None)
                        except Exception:
                            pass
                        # Persist post-pop state so restart doesn't see stale entries
                        try:
                            self.state.update_agent("entry_prices", dict(self._entry_prices))
                            self.state.update_agent("entry_sl_dist", dict(self._entry_sl_dist))
                            self.state.update_agent("directions", dict(self._directions))
                            self.state.update_agent("entry_dollar_risk", dict(self._entry_dollar_risk))
                            self.state.update_agent("peak_profit_r", dict(self._peak_profit_r))
                        except Exception:
                            pass
                        # Track external close time for brain SL cooldown
                        if not hasattr(self, '_external_close_time'):
                            self._external_close_time = {}
                        self._external_close_time[symbol] = __import__('time').time()
                        # Direction + win/loss signal for asymmetric cooldown.
                        if not hasattr(self, '_external_close_direction'):
                            self._external_close_direction = {}
                            self._external_close_was_win = {}
                        self._external_close_direction[symbol] = closed_direction
                        self._external_close_was_win[symbol] = was_win
                        # Snapshot peak R before clearing — deal sync (5s cadence)
                        # reads this AFTER close to feed RL exit-rule learning.
                        # Earlier bug: pop happened immediately, so peak_r was always
                        # 0 by the time deal_sync looked it up.
                        last_peak = self._peak_profit_r.pop(symbol, None)
                        if last_peak is not None and last_peak > 0:
                            if not hasattr(self, '_last_close_peak_r'):
                                self._last_close_peak_r = {}
                            self._last_close_peak_r[symbol] = float(last_peak)

            return mt5_has
        except Exception:
            with self._lock:
                return symbol in self._directions and self._directions.get(symbol) != "FLAT"

    def get_position_direction(self, symbol) -> str:
        """Get current position direction (any sub-position)."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return "FLAT"
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return "FLAT"
        valid_magics = {int(cfg.magic) + off for off in SUB_MAGIC_OFFSETS}
        for p in positions:
            if int(p.magic) in valid_magics:
                return "LONG" if int(p.type) == 0 else "SHORT"
        return "FLAT"

    def get_positions_info(self):
        """Info on ALL our positions (every strategy book) for the dashboard.
        2026-07-08: single positions_get() over the whole book (was one per SYMBOLS
        symbol) — fewer bridge calls AND it now includes the trend/IMR AUX symbols
        (ETH/NAS/JPN/SP500/US2000) that a SYMBOLS-only scan silently dropped."""
        result = []
        try:
            positions = self.mt5.positions_get() or []
        except Exception:
            return result
        for p in positions:
            if int(p.magic) < 8000:      # not one of ours
                continue
            cfg = symbol_cfg(p.symbol)
            if cfg is None:
                continue
            mode = strategy_of_magic(int(p.magic), p.symbol)   # swing/scalp/trend/imr/…
            result.append({
                "symbol": p.symbol,
                "type": "BUY" if int(p.type) == 0 else "SELL",
                "volume": float(p.volume),
                "pnl": float(p.profit),
                "price_open": float(p.price_open),
                "sl": float(p.sl),
                "tp": float(p.tp),
                "magic": int(p.magic),
                "ticket": int(p.ticket),
                "duration": self._format_duration(float(p.time)),
                "mode": mode,
                "sub": int(p.magic) - int(cfg.magic),
            })
        return result

    def _get_total_exposure(self) -> float:
        """Total risk exposure as % of equity across ALL open positions, including
        manual entries and positions on dropped symbols. Closes the gap where
        manual XAUUSD position would not count toward the 12% live exposure cap."""
        equity = float(self.state.get_agent_state().get("equity", 1000))
        if equity <= 0:
            return 100.0
        try:
            all_positions = self.mt5.positions_get() or []
        except Exception as e:
            log.warning("_get_total_exposure: positions_get failed: %s", e)
            return 0.0
        total_risk_usd = 0.0
        si_cache = {}
        for p in all_positions:
            try:
                sl = float(p.sl)
                if sl <= 0:
                    # No SL set — treat as full-volume nominal risk (defensive)
                    continue
                sym = str(p.symbol)
                if sym not in si_cache:
                    si_cache[sym] = self.mt5.symbol_info(sym)
                si = si_cache[sym]
                if not si or not si.trade_tick_value or not si.trade_tick_size:
                    continue
                risk_pts = abs(float(p.price_open) - sl)
                risk_usd = risk_pts / float(si.trade_tick_size) * float(si.trade_tick_value) * float(p.volume)
                total_risk_usd += risk_usd
            except Exception:
                continue
        return (total_risk_usd / equity * 100) if equity > 0 else 0.0

    def _get_atr(self, symbol, period=14):
        """Get current ATR from H1 candles via state."""
        ind = self.state.get_indicators(symbol)
        if ind and "atr" in ind:
            return float(ind["atr"])

        # Fallback: compute from M15 candles
        df = self.state.get_candles(symbol, 15)
        if df is not None and len(df) > period + 1:
            h = df["high"].values.astype(float)
            l = df["low"].values.astype(float)
            c = df["close"].values.astype(float)
            tr = np.maximum(h[1:] - l[1:],
                            np.maximum(np.abs(h[1:] - c[:-1]),
                                       np.abs(l[1:] - c[:-1])))
            return float(np.mean(tr[-period:]))
        return 0.0

    def _get_atr_avg(self, symbol, lookback=50):
        """Get 50-bar average ATR for adaptive trailing."""
        df = self.state.get_candles(symbol, 60)
        if df is None or len(df) < lookback + 15:
            return 0.0
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        n = len(c)
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]),
                                   np.abs(l[1:] - c[:-1])))
        if len(tr) < lookback:
            return float(np.mean(tr))
        # 14-period ATR at each bar, then average last 50
        atr_vals = []
        for i in range(14, len(tr)):
            atr_vals.append(float(np.mean(tr[max(0,i-14):i])))
        if len(atr_vals) < lookback:
            return float(np.mean(atr_vals)) if atr_vals else 0.0
        return float(np.mean(atr_vals[-lookback:]))

    @staticmethod
    def _format_duration(open_time):
        """Format position duration."""
        elapsed = time.time() - open_time
        if elapsed < 0:
            elapsed = 0
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
