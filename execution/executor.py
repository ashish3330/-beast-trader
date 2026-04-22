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
    SYMBOLS, MAX_RISK_PER_TRADE_PCT, MAX_TOTAL_EXPOSURE_PCT,
    ATR_SL_MULTIPLIER, TRAIL_STEPS, SUB2_TRAIL_STEPS,
    SCALP_RISK_PCT, SCALP_ATR_MULT, SCALP_MAGIC_OFFSET, SCALP_TRAIL_STEPS,
    SYMBOL_ATR_SL_OVERRIDE, SYMBOL_TRAIL_OVERRIDE,
    SMART_ENTRY_MODE,
)

# ═══ EXECUTION QUALITY CONSTANTS ═══
REQUOTE_RETCODE = 10004
REQUOTE_MAX_RETRIES = 3
REQUOTE_DELAY_SEC = 0.1
SLIPPAGE_HISTORY_SIZE = 20
SPREAD_SPIKE_MULTIPLIER = 2.0   # reject if spread > 2x signal-time spread
SPREAD_SPIKE_DELAY_SEC = 5.0

# ═══ 3-SUB POSITION ARCHITECTURE ═══
# "Scaled exit IS the edge" — validated by user's backtests
# Sub0: 50% lot @ TP1 (2R) — take quick profit
# Sub1: 30% lot @ TP2 (3R) — take more profit
# Sub2: 20% lot @ wide TP  — let trailing SL ride the trend
SUB_SPLITS = [0.50, 0.30, 0.20]
SUB_TP_R = [2.0, 3.0, 50.0]  # TP in R-multiples (sub2 = wide, trailing exits)
SUB_MAGIC_OFFSETS = [0, 1, 2]  # sub0=base, sub1=base+1, sub2=base+2

# Trend-following symbols: single position (big runners need full lot riding the trend)
# Backtest proved: BTCUSD PF 3.17 single vs 0.99 with 3-sub (kills the edge)
SINGLE_POSITION_SYMBOLS = {"BTCUSD"}

# Spread filter: max spread as multiple of ATR
MAX_SPREAD_ATR_RATIO = 0.3  # reject if spread > 30% of ATR

log = logging.getLogger("dragon.executor")


class Executor:
    """Handles order execution, lot sizing, trailing SL, and position management.
    Includes institutional-grade execution quality tracking."""

    def __init__(self, mt5, state):
        self.mt5 = mt5
        self.state = state
        self._lock = threading.RLock()  # protects all internal state
        self._closing = {}        # symbol -> True if close in progress
        self._entry_prices = {}   # symbol -> entry_price
        self._entry_sl_dist = {}  # symbol -> sl_distance at entry
        self._directions = {}     # symbol -> "LONG" or "SHORT"

        # ── Execution quality tracking ──
        self._slippage_history = {}   # symbol -> deque(maxlen=20) of slippage in points
        self._exec_latencies = {}     # symbol -> deque(maxlen=20) of latency in ms
        self._fill_counts = {}        # symbol -> {"full": int, "partial": int, "total": int}
        self._total_orders = {}       # symbol -> int (total order_send attempts)

        # ── RL trail adjustments (set by brain/run.py) ──
        self._rl_trail_adj = {}       # symbol -> {lock_threshold_mult, be_threshold_mult, trail_tightness_mult}

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
                return result, 0.0

        return None, 0.0

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

    # ═══════════════════════════════════════════════════════════════════════

    def open_trade(self, symbol, direction, atr, risk_pct=None, signal_spread=None,
                   smart_tp=None):
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

        # ── SPREAD FILTER (vs ATR) ──
        spread = float(tick.ask) - float(tick.bid)
        if atr > 0 and spread / float(atr) > MAX_SPREAD_ATR_RATIO:
            log.warning("[%s] SKIP: spread %.5f > %.0f%% of ATR %.5f",
                        symbol, spread, MAX_SPREAD_ATR_RATIO * 100, atr)
            return False

        # ── SL DISTANCE ──
        base_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, ATR_SL_MULTIPLIER)
        sl_mult = base_sl_mult
        if hasattr(self, '_vol_model') and self._vol_model:
            try:
                vol_pred = self._vol_model.predict_from_state(symbol, self.state)
                if vol_pred and vol_pred > 0:
                    sl_mult = base_sl_mult * max(0.8, min(1.5, vol_pred))
            except Exception as e:
                log.debug("[%s] Vol model fallback: %s", symbol, e)
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

        # ── SMALL ACCOUNT PROTECTION: cap SL so vol_min risk stays within budget ──
        # If calculated lot < vol_min, we're forced to use vol_min = higher risk than intended
        # Solution: shrink SL distance so that vol_min * sl_ticks * tick_value <= max_allowed_risk
        MAX_RISK_OVER = 3.0  # max 3x intended risk (e.g., 1% intended → max 3% actual)
        if total_volume <= vol_min and tick_value > 0 and tick_size > 0:
            max_allowed_risk = risk_amount * MAX_RISK_OVER
            max_sl_ticks = max_allowed_risk / (tick_value * vol_min) if vol_min > 0 else sl_ticks
            max_sl_dist = max_sl_ticks * tick_size
            if sl_dist > max_sl_dist and max_sl_dist > 0:
                old_sl = sl_dist
                sl_dist = max(max_sl_dist, float(si.trade_stops_level) * point * 2)
                sl_ticks = sl_dist / tick_size
                actual_risk = sl_ticks * tick_value * vol_min
                actual_pct = actual_risk / equity * 100 if equity > 0 else 0
                log.info("[%s] SL CAPPED for small account: %.2f → %.2f (risk $%.2f = %.1f%% vs intended %.1f%%)",
                         symbol, old_sl, sl_dist, actual_risk, actual_pct, effective_risk)

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

            if direction == "LONG":
                sl = float(round(price - sl_dist, digits))
                tp = float(round(price + sl_dist * 50, digits))
            else:
                sl = float(round(price + sl_dist, digits))
                tp = float(round(price - sl_dist * 50, digits))

            request = {
                "action": int(1), "symbol": str(symbol), "volume": float(volume),
                "type": int(order_type), "price": float(price),
                "sl": float(sl), "tp": float(tp), "deviation": int(50),
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
            log.info("[%s] OPENED single %s %.2f lots (risk=$%.2f %.3f%%)",
                     symbol, direction, actual_vol, risk_amount, effective_risk)
            return True

        # ── OPEN 3 SUB-POSITIONS (for non-trend-following symbols) ──
        total_filled_volume = 0.0
        fill_prices = []  # (volume, price) tuples for weighted avg entry
        for i, (split, tp_r, magic_off) in enumerate(zip(SUB_SPLITS, SUB_TP_R, SUB_MAGIC_OFFSETS)):
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
                "deviation": int(50),
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

        actual_risk_usd = sl_dist / tick_size * tick_value * total_filled_volume if tick_size > 0 else 0
        log.info("[%s] OPENED %d/%d subs %s filled=%.2f/%.2f lots SL=%.2fpts REAL_RISK=$%.2f (%.1f%% equity) ATR=%.5f",
                 symbol, opened, len(SUB_SPLITS), direction, total_filled_volume, total_volume,
                 sl_dist, actual_risk_usd, actual_risk_usd / equity * 100 if equity > 0 else 0, atr)
        return True

    def close_position(self, symbol, comment="DragonClose"):
        """Close all sub-positions for a symbol (magic, magic+1, magic+2).
        Returns True if at least one position was closed."""
        # Prevent concurrent closes on same symbol
        with self._lock:
            if self._closing.get(symbol, False):
                log.debug("[%s] Already closing, skip duplicate", symbol)
                return False
            self._closing[symbol] = True

        try:
            return self._close_position_impl(symbol, comment)
        finally:
            with self._lock:
                self._closing.pop(symbol, None)

    def _close_position_impl(self, symbol, comment):
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False

        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            log.warning("[%s] close_position: positions_get returned None", symbol)
            return False

        valid_magics = {int(cfg.magic) + off for off in SUB_MAGIC_OFFSETS}
        any_closed = False
        for p in positions:
            if int(p.magic) not in valid_magics:
                continue
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                log.warning("[%s] close_position: no tick (market closed?)", symbol)
                continue

            close_type = 1 if int(p.type) == 0 else 0  # Reverse direction
            close_price = float(tick.bid) if int(p.type) == 0 else float(tick.ask)

            request = {
                "action": int(1),
                "symbol": str(symbol),
                "volume": float(p.volume),
                "type": int(close_type),
                "price": float(close_price),
                "position": int(p.ticket),
                "deviation": int(50),
                "magic": int(cfg.magic),
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

        # Only clear tracking if at least one close succeeded
        if any_closed:
            with self._lock:
                self._entry_prices.pop(symbol, None)
                self._entry_sl_dist.pop(symbol, None)
                self._directions.pop(symbol, None)
                # Clear peak profit tracking
                if hasattr(self, '_peak_profit_r'):
                    self._peak_profit_r.pop(symbol, None)
        return any_closed

    def close_all(self, comment="DragonEmergency"):
        """Close all positions."""
        for symbol in list(SYMBOLS.keys()):
            if self.has_position(symbol):
                self.close_position(symbol, comment)

    def reverse_position(self, symbol, new_direction, atr, risk_pct=None, signal_spread=None):
        """Close current position and open in opposite direction."""
        if self.has_position(symbol):
            old_dir = self._directions.get(symbol, "?")
            log.info("[%s] REVERSING %s -> %s", symbol, old_dir, new_direction)
            closed = self.close_position(symbol, "DragonReversal")
            if not closed:
                # Force clear internal tracking so we can still open new direction
                log.warning("[%s] Reversal close failed — force-clearing tracking", symbol)
                with self._lock:
                    self._entry_prices.pop(symbol, None)
                    self._entry_sl_dist.pop(symbol, None)
                    self._directions.pop(symbol, None)
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
            "deviation": int(50),
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

        log.info("[%s] SCALP OPENED %s %.2f lots @ %.5f SL=%.5f TP=%.5f (risk=$%.2f %.1f%%, ATR=%.5f)",
                 symbol, direction, actual_vol, actual_price, sl, tp, risk_amount, effective_risk, atr)
        return True

    def has_scalp_position(self, symbol) -> bool:
        """Check if we have an open scalp position for this symbol."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        scalp_magic = int(cfg.magic) + SCALP_MAGIC_OFFSET
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) == scalp_magic for p in positions)

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
        if cfg is None:
            return

        # Force position sync first — clears stale entry data if position closed externally
        if not self.has_position(symbol) and not self.has_scalp_position(symbol):
            return

        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return

        base_magic = int(cfg.magic)
        swing_magics = {base_magic + off for off in SUB_MAGIC_OFFSETS}
        scalp_magic = base_magic + SCALP_MAGIC_OFFSET

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

        # Apply trail — per-symbol override > Sub2 runner > default
        sym_trail = SYMBOL_TRAIL_OVERRIDE.get(symbol)
        for pos in positions:
            pos_magic = int(pos.magic)
            if pos_magic in swing_magics:
                sub_idx = pos_magic - base_magic
                if sub_idx == 2:
                    trail = SUB2_TRAIL_STEPS
                elif sym_trail:
                    trail = sym_trail  # per-symbol (e.g. XAUUSD 0.3R lock)
                else:
                    trail = TRAIL_STEPS
                self._apply_trail(symbol, pos, trail, symbol)
            elif pos_magic == scalp_magic:
                self._apply_trail(symbol, pos, SCALP_TRAIL_STEPS, symbol + "_scalp")

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
                log.info("[%s] TP1 HIT — moved Sub%d SL to BE+0.2R: %.5f",
                         symbol, int(pos.magic) - int(SYMBOLS[symbol].magic), be_sl)

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

        if sl_dist <= 0:
            sl_dist = abs(entry - current_sl)
            if sl_dist <= 0:
                return

        cur_price = float(tick.bid) if direction == "LONG" else float(tick.ask)
        profit_dist = (cur_price - entry) if direction == "LONG" else (entry - cur_price)
        profit_r = profit_dist / sl_dist if sl_dist > 0 else 0

        # ── PROFIT RATCHET: track peak R and enforce profit floor ──
        # Once trade hits 1R+, SL can NEVER go below 0.3R profit
        # Once trade hits 2R+, SL can NEVER go below 0.7R profit
        peak_key = f"_peak_r_{tracking_key}"
        prev_peak = getattr(self, '_peak_profit_r', {}).get(tracking_key, 0)
        cur_peak = max(prev_peak, profit_r)
        if not hasattr(self, '_peak_profit_r'):
            self._peak_profit_r = {}
        self._peak_profit_r[tracking_key] = cur_peak

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
        Syncs internal tracking with MT5 reality to prevent drift."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        try:
            positions = self.mt5.positions_get(symbol=symbol)
            if positions is None:
                with self._lock:
                    return symbol in self._directions and self._directions[symbol] != "FLAT"
            valid_magics = {int(cfg.magic) + off for off in SUB_MAGIC_OFFSETS}
            mt5_has = any(int(p.magic) in valid_magics for p in positions)

            # Sync internal tracking with MT5 reality
            if not mt5_has:
                with self._lock:
                    if symbol in self._directions:
                        log.info("[%s] Position closed externally — clearing internal tracking", symbol)
                        self._entry_prices.pop(symbol, None)
                        self._entry_sl_dist.pop(symbol, None)
                        self._directions.pop(symbol, None)
                        # Track external close time for brain SL cooldown
                        if not hasattr(self, '_external_close_time'):
                            self._external_close_time = {}
                        self._external_close_time[symbol] = __import__('time').time()
                        if hasattr(self, '_peak_profit_r'):
                            self._peak_profit_r.pop(symbol, None)

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
        """Get info on all our positions (swing subs + scalp) for dashboard."""
        result = []
        for symbol, cfg in SYMBOLS.items():
            positions = self.mt5.positions_get(symbol=symbol)
            if positions is None:
                continue
            base_magic = int(cfg.magic)
            swing_magics = {base_magic + off for off in SUB_MAGIC_OFFSETS}
            scalp_magic = base_magic + SCALP_MAGIC_OFFSET
            for p in positions:
                pm = int(p.magic)
                if pm not in swing_magics and pm != scalp_magic:
                    continue
                if pm == scalp_magic:
                    mode = "scalp"
                    sub = -1
                else:
                    mode = "swing"
                    sub = pm - base_magic  # 0, 1, or 2
                result.append({
                    "symbol": symbol,
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
                    "sub": sub,
                })
        return result

    def _get_total_exposure(self) -> float:
        """Calculate total risk exposure as % of equity across all subs."""
        equity = float(self.state.get_agent_state().get("equity", 1000))
        if equity <= 0:
            return 100.0

        total_risk = 0.0
        for symbol, cfg in SYMBOLS.items():
            if not self.has_position(symbol):
                continue
            positions = self.mt5.positions_get(symbol=symbol)
            if not positions:
                continue
            valid_magics = {int(cfg.magic) + off for off in SUB_MAGIC_OFFSETS}
            valid_magics.add(int(cfg.magic) + SCALP_MAGIC_OFFSET)
            si = self.mt5.symbol_info(symbol)
            for p in positions:
                if int(p.magic) not in valid_magics:
                    continue
                risk = abs(float(p.price_open) - float(p.sl)) * float(p.volume)
                if si and si.trade_tick_value and si.trade_tick_size:
                    risk_usd = risk / float(si.trade_tick_size) * float(si.trade_tick_value)
                    total_risk += risk_usd

        return (total_risk / equity * 100) if equity > 0 else 0.0

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
