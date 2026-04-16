"""
Beast Trader — Order Executor.
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
    ATR_SL_MULTIPLIER, TRAIL_STEPS,
)

log = logging.getLogger("beast.executor")


class Executor:
    """Handles order execution, lot sizing, trailing SL, and position management."""

    def __init__(self, mt5, state):
        self.mt5 = mt5
        self.state = state
        self._entry_prices = {}   # symbol -> entry_price
        self._entry_sl_dist = {}  # symbol -> sl_distance at entry
        self._directions = {}     # symbol -> "LONG" or "SHORT"

    def open_trade(self, symbol, direction, atr):
        """
        Open a trade with risk-based lot sizing.
        SL = 3x ATR minimum.
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            log.error("[%s] Unknown symbol", symbol)
            return False

        # Check if already has position
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

        # Prices
        price = float(tick.ask) if direction == "LONG" else float(tick.bid)
        point = float(si.point) if si.point else 0.00001
        digits = int(si.digits)

        # SL distance = max(3x ATR, minimum stop distance)
        sl_dist = max(float(atr) * ATR_SL_MULTIPLIER, float(si.trade_stops_level) * point * 2)

        if direction == "LONG":
            sl = float(round(price - sl_dist, digits))
            tp = float(round(price + sl_dist * 50, digits))  # Wide TP — trailing does exits
        else:
            sl = float(round(price + sl_dist, digits))
            tp = float(round(price - sl_dist * 50, digits))

        # Risk-based lot sizing: risk_amount / (sl_points * tick_value)
        equity = float(self.state.get_agent_state().get("equity", 1000))
        risk_amount = equity * (MAX_RISK_PER_TRADE_PCT / 100.0)

        # sl_points in broker points
        sl_points = sl_dist / point
        tick_value = float(si.trade_tick_value) if si.trade_tick_value else 1.0
        tick_size = float(si.trade_tick_size) if si.trade_tick_size else point

        # Lot size = risk_amount / (sl in ticks * tick_value)
        sl_ticks = sl_dist / tick_size if tick_size > 0 else sl_points
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

        # Check total exposure won't exceed limit
        current_exposure = self._get_total_exposure()
        new_risk_pct = (risk_amount / equity * 100) if equity > 0 else 100
        if current_exposure + new_risk_pct > MAX_TOTAL_EXPOSURE_PCT:
            log.warning("[%s] Total exposure %.1f%% + %.1f%% would exceed %.1f%% limit",
                        symbol, current_exposure, new_risk_pct, MAX_TOTAL_EXPOSURE_PCT)
            # Still execute — user wants no blocks, just warn
            log.warning("[%s] WARNING: Proceeding despite exposure limit", symbol)

        # Build order request — all values cast to float()
        order_type = 0 if direction == "LONG" else 1  # BUY=0, SELL=1
        request = {
            "action": int(1),             # TRADE_ACTION_DEAL
            "symbol": str(symbol),
            "volume": float(volume),
            "type": int(order_type),
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(50),
            "magic": int(cfg.magic),
            "comment": str("Beast"),
            "type_filling": int(1),       # IOC
            "type_time": int(0),
        }

        result = self.mt5.order_send(request)
        if result is None:
            log.error("[%s] order_send returned None", symbol)
            return False

        retcode = int(result.retcode)
        if retcode not in (10009, 10008):
            log.error("[%s] Order failed [%d]: %s", symbol, retcode, result.comment)
            return False

        # Track entry
        self._entry_prices[symbol] = float(price)
        self._entry_sl_dist[symbol] = float(sl_dist)
        self._directions[symbol] = direction

        log.info("[%s] OPENED %s %.2f lots @ %.5f SL=%.5f (risk=$%.2f, ATR=%.5f)",
                 symbol, direction, volume, price, sl, risk_amount, atr)
        return True

    def close_position(self, symbol, comment="BeastClose"):
        """Close position for a symbol."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return

        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return

        for p in positions:
            if int(p.magic) != cfg.magic:
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

    def close_all(self, comment="BeastEmergency"):
        """Close all positions."""
        for symbol in list(SYMBOLS.keys()):
            if self.has_position(symbol):
                self.close_position(symbol, comment)

    def reverse_position(self, symbol, new_direction, atr):
        """Close current position and open in opposite direction."""
        if self.has_position(symbol):
            old_dir = self._directions.get(symbol, "?")
            log.info("[%s] REVERSING %s -> %s", symbol, old_dir, new_direction)
            self.close_position(symbol, "BeastReversal")
            time.sleep(0.2)  # Brief pause after close
        return self.open_trade(symbol, new_direction, atr)

    def manage_trailing_sl(self, symbol):
        """
        Apply stepped trailing SL based on profit in R multiples.
        Uses the moderate profile from ApexQuant.
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return

        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return

        my_positions = [p for p in positions if int(p.magic) == cfg.magic]
        if not my_positions:
            return

        pos = my_positions[0]
        direction = "LONG" if int(pos.type) == 0 else "SHORT"

        si = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if si is None or tick is None:
            return

        point = float(si.point) if si.point else 0.00001
        digits = int(si.digits)
        current_sl = float(pos.sl)
        entry = self._entry_prices.get(symbol, float(pos.price_open))
        sl_dist = self._entry_sl_dist.get(symbol, 0)

        if sl_dist <= 0:
            # Reconstruct from position
            sl_dist = abs(entry - current_sl)
            if sl_dist <= 0:
                return

        cur_price = float(tick.bid) if direction == "LONG" else float(tick.ask)
        profit_dist = (cur_price - entry) if direction == "LONG" else (entry - cur_price)
        profit_r = profit_dist / sl_dist if sl_dist > 0 else 0

        # Get current ATR for trailing
        atr = self._get_atr(symbol)
        if atr <= 0:
            atr = sl_dist

        new_sl = None
        action = ""

        # Walk through trail steps (highest first)
        for r_threshold, step_type, param in TRAIL_STEPS:
            if profit_r >= r_threshold:
                if step_type == "trail":
                    trail_dist = param * atr
                    new_sl = (cur_price - trail_dist) if direction == "LONG" else (cur_price + trail_dist)
                    # Floor: never trail below lock level
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

        # Enforce minimum stop distance
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

        # Modify SL via action:6
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
        """Check if we have an open position for this symbol."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return False
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(int(p.magic) == cfg.magic for p in positions)

    def get_position_direction(self, symbol) -> str:
        """Get current position direction."""
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return "FLAT"
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return "FLAT"
        for p in positions:
            if int(p.magic) == cfg.magic:
                return "LONG" if int(p.type) == 0 else "SHORT"
        return "FLAT"

    def get_positions_info(self):
        """Get info on all our positions for dashboard."""
        result = []
        for symbol, cfg in SYMBOLS.items():
            positions = self.mt5.positions_get(symbol=symbol)
            if positions is None:
                continue
            for p in positions:
                if int(p.magic) != cfg.magic:
                    continue
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
                })
        return result

    def _get_total_exposure(self) -> float:
        """Calculate total risk exposure as % of equity."""
        equity = float(self.state.get_agent_state().get("equity", 1000))
        if equity <= 0:
            return 100.0

        total_risk = 0.0
        for symbol in SYMBOLS:
            if symbol in self._entry_sl_dist and self.has_position(symbol):
                # Approximate risk from SL distance
                cfg = SYMBOLS[symbol]
                positions = self.mt5.positions_get(symbol=symbol)
                if positions:
                    for p in positions:
                        if int(p.magic) == cfg.magic:
                            risk = abs(float(p.price_open) - float(p.sl)) * float(p.volume)
                            si = self.mt5.symbol_info(symbol)
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
