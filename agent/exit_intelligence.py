"""
Dragon Trader — Exit Intelligence.
Runs every tick cycle on open positions to make smart exit decisions.
Never let a winner turn into a loser. Protect profits ruthlessly.
"""
import time
import logging
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("dragon.exits")


class ExitIntelligence:
    """Intelligent exit manager for open positions."""

    def __init__(self, state, executor):
        self.state = state
        self.executor = executor
        # Track per-position metrics
        self._peak_profit_r = {}   # symbol -> highest R achieved
        self._bars_in_trade = {}   # symbol -> bar count since entry
        self._last_check_time = {} # symbol -> last check timestamp

    def evaluate_exits(self):
        """Run exit evaluation on all open positions. Called every brain cycle."""
        for symbol in list(self.executor._directions.keys()):
            # Skip scalp keys
            if symbol.endswith("_scalp"):
                continue
            try:
                self._evaluate_position(symbol)
            except Exception as e:
                log.warning("[%s] Exit eval error: %s", symbol, e)

    def _evaluate_position(self, symbol):
        """Evaluate a single position for exit signals."""
        if not self.executor.has_position(symbol):
            self._cleanup(symbol)
            return

        direction = self.executor._directions.get(symbol, "FLAT")
        if direction == "FLAT":
            return

        entry = self.executor._entry_prices.get(symbol, 0)
        sl_dist = self.executor._entry_sl_dist.get(symbol, 0)
        if entry <= 0 or sl_dist <= 0:
            return

        # Get current price
        tick = self.state.get_tick(symbol)
        if tick is None:
            return
        cur_price = float(tick.bid) if direction == "LONG" else float(tick.ask)

        # Calculate profit in R
        profit_dist = (cur_price - entry) if direction == "LONG" else (entry - cur_price)
        profit_r = profit_dist / sl_dist if sl_dist > 0 else 0

        # Track peak profit
        if symbol not in self._peak_profit_r:
            self._peak_profit_r[symbol] = profit_r
        else:
            self._peak_profit_r[symbol] = max(self._peak_profit_r[symbol], profit_r)
        peak_r = self._peak_profit_r[symbol]

        # Track bars in trade
        self._bars_in_trade.setdefault(symbol, 0)
        now = time.time()
        last = self._last_check_time.get(symbol, now)
        if now - last > 3600:  # count H1 bars approximately
            self._bars_in_trade[symbol] += 1
            self._last_check_time[symbol] = now

        # EXIT CHECKS (any one triggers close):

        # 1. MOMENTUM DECAY: price gave back > 40% of peak profit and was > 1.5R
        if peak_r >= 1.5 and profit_r < peak_r * 0.6:
            log.info("[%s] EXIT: Momentum decay (peak=%.1fR, now=%.1fR, gave back %.0f%%)",
                     symbol, peak_r, profit_r, (1 - profit_r / peak_r) * 100)
            self.executor.close_position(symbol, "DragonMomentumDecay")
            self._cleanup(symbol)
            return

        # 2. OPPOSING SIGNAL: M15 strongly opposing our direction
        m15_strength = self._get_opposing_strength(symbol, direction)
        if profit_r > 0.3 and m15_strength > 0.7:
            log.info("[%s] EXIT: Opposing M15 signal (strength=%.2f, profit=%.1fR)",
                     symbol, m15_strength, profit_r)
            self.executor.close_position(symbol, "DragonOpposingSignal")
            self._cleanup(symbol)
            return

        # 3. STALE TRADE: been in trade > 20 H1 bars with < 0.5R profit
        if self._bars_in_trade.get(symbol, 0) > 20 and profit_r < 0.5:
            log.info("[%s] EXIT: Stale trade (%d bars, only %.1fR profit)",
                     symbol, self._bars_in_trade[symbol], profit_r)
            self.executor.close_position(symbol, "DragonStaleTrade")
            self._cleanup(symbol)
            return

        # 4. VOLATILITY SPIKE: if vol model predicts expansion and we're in profit, tighten
        # (This is handled by the trailing SL logic in executor, but we add an extra check)

        # 5. PROTECT BREAKEVEN: if was > 1R and now falling back toward entry
        if peak_r >= 1.0 and profit_r <= 0.1:
            log.info("[%s] EXIT: Protecting breakeven (peak=%.1fR, now=%.1fR)",
                     symbol, peak_r, profit_r)
            self.executor.close_position(symbol, "DragonProtectBE")
            self._cleanup(symbol)
            return

    def _get_opposing_strength(self, symbol, direction):
        """Check M15 for opposing signal strength. Returns 0-1."""
        from signals.momentum_scorer import _ema, _supertrend, IND_DEFAULTS, IND_OVERRIDES

        m15_df = self.state.get_candles(symbol, 15)
        if m15_df is None or len(m15_df) < 50:
            return 0.0

        try:
            close = m15_df["close"].values.astype(np.float64)
            high = m15_df["high"].values.astype(np.float64)
            low = m15_df["low"].values.astype(np.float64)
            n = len(close)

            icfg = dict(IND_DEFAULTS)
            icfg.update(IND_OVERRIDES.get(symbol, {}))

            ema_s = _ema(close, 15)
            ema_l = _ema(close, 40)
            _, st_dir = _supertrend(high.copy(), low.copy(), close,
                                     float(icfg["ST_F"]), int(icfg["ST_ATR"]))

            bi = n - 2
            if bi < 1:
                return 0.0

            # Count opposing signals
            opposing = 0.0

            if direction == "LONG":
                if float(ema_s[bi]) < float(ema_l[bi]):
                    opposing += 0.4
                if int(st_dir[bi]) == -1:
                    opposing += 0.4
                # Check if EMAs are accelerating against us
                if bi > 2 and float(ema_s[bi] - ema_l[bi]) < float(ema_s[bi - 2] - ema_l[bi - 2]):
                    opposing += 0.2
            else:
                if float(ema_s[bi]) > float(ema_l[bi]):
                    opposing += 0.4
                if int(st_dir[bi]) == 1:
                    opposing += 0.4
                if bi > 2 and float(ema_s[bi] - ema_l[bi]) > float(ema_s[bi - 2] - ema_l[bi - 2]):
                    opposing += 0.2

            return min(1.0, opposing)
        except Exception:
            return 0.0

    def _cleanup(self, symbol):
        """Clean up tracking data for a closed position."""
        self._peak_profit_r.pop(symbol, None)
        self._bars_in_trade.pop(symbol, None)
        self._last_check_time.pop(symbol, None)

    def get_status(self, symbol) -> dict:
        """Get exit intelligence status for dashboard."""
        return {
            "peak_profit_r": round(self._peak_profit_r.get(symbol, 0), 2),
            "bars_in_trade": self._bars_in_trade.get(symbol, 0),
        }
