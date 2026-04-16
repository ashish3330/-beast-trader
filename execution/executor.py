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
    ATR_SL_MULTIPLIER, TRAIL_STEPS,
    SCALP_RISK_PCT, SCALP_ATR_MULT, SCALP_MAGIC_OFFSET, SCALP_TRAIL_STEPS,
)

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
        Open a trade with risk-based lot sizing.
        SL = ATR_SL_MULTIPLIER * ATR minimum.
        risk_pct overrides MAX_RISK_PER_TRADE_PCT if provided (from MasterBrain).
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

        # Dynamic SL: use vol model if available, else static multiplier
        sl_mult = ATR_SL_MULTIPLIER
        if hasattr(self, '_vol_model') and self._vol_model:
            try:
                vol_pred = self._vol_model.predict_from_state(symbol, self.state)
                if vol_pred and vol_pred > 0:
                    # vol_pred > 1 = expecting expansion -> widen SL
                    # vol_pred < 1 = expecting contraction -> tighten SL
                    sl_mult = ATR_SL_MULTIPLIER * max(0.8, min(1.5, vol_pred))
                    log.debug("[%s] Vol model: pred=%.2f -> SL mult=%.2f", symbol, vol_pred, sl_mult)
            except:
                pass
        sl_dist = max(float(atr) * sl_mult, float(si.trade_stops_level) * point * 2)

        if direction == "LONG":
            sl = float(round(price - sl_dist, digits))
            tp = float(round(price + sl_dist * 50, digits))  # Wide TP — trailing does exits
        else:
            sl = float(round(price + sl_dist, digits))
            tp = float(round(price - sl_dist * 50, digits))

        # Risk-based lot sizing: risk_amount / (sl_points * tick_value)
        effective_risk = risk_pct if risk_pct is not None else MAX_RISK_PER_TRADE_PCT
        equity = float(self.state.get_agent_state().get("equity", 1000))
        risk_amount = equity * (effective_risk / 100.0)

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
            "comment": str("Dragon"),
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

        log.info("[%s] OPENED %s %.2f lots @ %.5f SL=%.5f (risk=$%.2f %.1f%%, ATR=%.5f)",
                 symbol, direction, volume, price, sl, risk_amount, effective_risk, atr)
        return True

    def close_position(self, symbol, comment="DragonClose"):
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
        Detects scalp vs swing positions via magic number offset and
        uses the appropriate trail profile.
        """
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return

        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return

        swing_magic = int(cfg.magic)
        scalp_magic = int(cfg.magic) + SCALP_MAGIC_OFFSET

        # Process both swing and scalp positions
        for pos in positions:
            pos_magic = int(pos.magic)
            if pos_magic == swing_magic:
                self._apply_trail(symbol, pos, TRAIL_STEPS, symbol)
            elif pos_magic == scalp_magic:
                self._apply_trail(symbol, pos, SCALP_TRAIL_STEPS, symbol + "_scalp")

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
        """Get info on all our positions (swing + scalp) for dashboard."""
        result = []
        for symbol, cfg in SYMBOLS.items():
            positions = self.mt5.positions_get(symbol=symbol)
            if positions is None:
                continue
            swing_magic = int(cfg.magic)
            scalp_magic = int(cfg.magic) + SCALP_MAGIC_OFFSET
            for p in positions:
                pm = int(p.magic)
                if pm != swing_magic and pm != scalp_magic:
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
                    "mode": "scalp" if pm == scalp_magic else "swing",
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
