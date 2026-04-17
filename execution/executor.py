"""
Dragon Trader — Order Executor.
Risk-based lot sizing, ATR-based SL, trailing SL, signal reversal handling.
All values cast to float() for rpyc bridge compatibility.
"""
import time
import logging
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SYMBOLS, MAX_RISK_PER_TRADE_PCT, MAX_TOTAL_EXPOSURE_PCT,
    ATR_SL_MULTIPLIER, TRAIL_STEPS, SUB2_TRAIL_STEPS,
    SCALP_RISK_PCT, SCALP_ATR_MULT, SCALP_MAGIC_OFFSET, SCALP_TRAIL_STEPS,
    SYMBOL_ATR_SL_OVERRIDE, SYMBOL_TRAIL_OVERRIDE,
)

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
    """Handles order execution, lot sizing, trailing SL, and position management."""

    def __init__(self, mt5, state):
        self.mt5 = mt5
        self.state = state
        self._entry_prices = {}   # symbol -> entry_price
        self._entry_sl_dist = {}  # symbol -> sl_distance at entry
        self._directions = {}     # symbol -> "LONG" or "SHORT"

    def open_trade(self, symbol, direction, atr, risk_pct=None):
        """
        Open 3 sub-positions with scaled TPs (the proven edge).
        Sub0: 50% @ TP1 (2R) — quick profit lock
        Sub1: 30% @ TP2 (3R) — more profit
        Sub2: 20% @ wide TP  — trailing SL rides the trend
        All share same SL = ATR_SL_MULTIPLIER * ATR.
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

        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            log.error("[%s] symbol_info_tick returned None", symbol)
            return False

        price = float(tick.ask) if direction == "LONG" else float(tick.bid)
        point = float(si.point) if si.point else 0.00001
        digits = int(si.digits)

        # ── SPREAD FILTER ──
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

        # Exposure check (warn only)
        current_exposure = self._get_total_exposure()
        new_risk_pct = (risk_amount / equity * 100) if equity > 0 else 100
        if current_exposure + new_risk_pct > MAX_TOTAL_EXPOSURE_PCT:
            log.warning("[%s] Exposure %.1f%%+%.1f%% > %.1f%% — proceeding",
                        symbol, current_exposure, new_risk_pct, MAX_TOTAL_EXPOSURE_PCT)

        # ── DETERMINE MODE: single (trend-followers) or 3-sub ──
        order_type = 0 if direction == "LONG" else 1
        opened = 0
        use_single = symbol in SINGLE_POSITION_SYMBOLS

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
            result = self.mt5.order_send(request)
            if result and int(result.retcode) in (10009, 10008):
                opened = 1
                log.info("[%s] SINGLE OPENED %s %.2f lots @ %.5f SL=%.5f (trend-follower)",
                         symbol, direction, volume, price, sl)
            elif result:
                log.error("[%s] Single order failed [%d]: %s", symbol, int(result.retcode), result.comment)

            if opened == 0:
                return False
            self._entry_prices[symbol] = float(price)
            self._entry_sl_dist[symbol] = float(sl_dist)
            self._directions[symbol] = direction
            log.info("[%s] OPENED single %s %.2f lots (risk=$%.2f %.3f%%)",
                     symbol, direction, volume, risk_amount, effective_risk)
            return True

        # ── OPEN 3 SUB-POSITIONS (for non-trend-following symbols) ──
        for i, (split, tp_r, magic_off) in enumerate(zip(SUB_SPLITS, SUB_TP_R, SUB_MAGIC_OFFSETS)):
            sub_vol = total_volume * split
            if vol_step > 0:
                sub_vol = float(round(int(sub_vol / vol_step) * vol_step, 2))
            sub_vol = max(vol_min, min(vol_max, sub_vol))

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

            result = self.mt5.order_send(request)
            if result is None:
                log.error("[%s] Sub%d order_send returned None", symbol, i)
                continue

            retcode = int(result.retcode)
            if retcode in (10009, 10008):
                opened += 1
                log.info("[%s] SUB%d OPENED %s %.2f lots @ %.5f SL=%.5f TP=%.5f (%.0fR)",
                         symbol, i, direction, sub_vol, price, sl, tp, tp_r)
            else:
                log.error("[%s] Sub%d failed [%d]: %s", symbol, i, retcode, result.comment)

        if opened == 0:
            return False

        # Track entry
        self._entry_prices[symbol] = float(price)
        self._entry_sl_dist[symbol] = float(sl_dist)
        self._directions[symbol] = direction

        log.info("[%s] OPENED %d/%d subs %s total=%.2f lots (risk=$%.2f %.3f%% ATR=%.5f)",
                 symbol, opened, len(SUB_SPLITS), direction, total_volume,
                 risk_amount, effective_risk, atr)
        return True

    def close_position(self, symbol, comment="DragonClose"):
        """Close all sub-positions for a symbol (magic, magic+1, magic+2)."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return

        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return

        valid_magics = {int(cfg.magic) + off for off in SUB_MAGIC_OFFSETS}
        for p in positions:
            if int(p.magic) not in valid_magics:
                continue
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
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

            result = self.mt5.order_send(request)
            if result and int(result.retcode) in (10009, 10008):
                log.info("[%s] CLOSED %s @ %.5f (%s)", symbol,
                         "LONG" if int(p.type) == 0 else "SHORT", close_price, comment)
            elif result:
                log.error("[%s] Close failed [%d]: %s", symbol, int(result.retcode), result.comment)

        # Clear tracking
        self._entry_prices.pop(symbol, None)
        self._entry_sl_dist.pop(symbol, None)
        self._directions.pop(symbol, None)

    def close_all(self, comment="DragonEmergency"):
        """Close all positions."""
        for symbol in list(SYMBOLS.keys()):
            if self.has_position(symbol):
                self.close_position(symbol, comment)

    def reverse_position(self, symbol, new_direction, atr):
        """Close current position and open in opposite direction."""
        if self.has_position(symbol):
            old_dir = self._directions.get(symbol, "?")
            log.info("[%s] REVERSING %s -> %s", symbol, old_dir, new_direction)
            self.close_position(symbol, "DragonReversal")
            time.sleep(0.2)  # Brief pause after close
        return self.open_trade(symbol, new_direction, atr)

    def open_scalp_trade(self, symbol, direction, atr, risk_pct=None):
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

        tick = self.mt5.symbol_info_tick(symbol)
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

        result = self.mt5.order_send(request)
        if result is None:
            log.error("[%s] scalp order_send returned None", symbol)
            return False

        retcode = int(result.retcode)
        if retcode not in (10009, 10008):
            log.error("[%s] Scalp order failed [%d]: %s", symbol, retcode, result.comment)
            return False

        # Track entry with scalp key
        scalp_key = symbol + "_scalp"
        self._entry_prices[scalp_key] = float(price)
        self._entry_sl_dist[scalp_key] = float(sl_dist)
        self._directions[scalp_key] = direction

        log.info("[%s] SCALP OPENED %s %.2f lots @ %.5f SL=%.5f TP=%.5f (risk=$%.2f %.1f%%, ATR=%.5f)",
                 symbol, direction, volume, price, sl, tp, risk_amount, effective_risk, atr)
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

        atr = self._get_atr(symbol)
        if atr <= 0:
            atr = sl_dist

        new_sl = None
        action = ""

        for r_threshold, step_type, param in trail_steps:
            if profit_r >= r_threshold:
                if step_type == "trail":
                    trail_dist = param * atr
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
                break

        if new_sl is None:
            return

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
        """Check if we have any open sub-position for this symbol."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        valid_magics = {int(cfg.magic) + off for off in SUB_MAGIC_OFFSETS}
        return any(int(p.magic) in valid_magics for p in positions)

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
