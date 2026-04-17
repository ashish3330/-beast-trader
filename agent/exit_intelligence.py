"""
Dragon Trader — Exit Intelligence.
Runs every tick cycle on open positions to make smart exit decisions.
Never let a winner turn into a loser. Protect profits ruthlessly.
"""
import time
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS

log = logging.getLogger("dragon.exits")


class ExitIntelligence:
    """Intelligent exit manager for open positions."""

    def __init__(self, state, executor):
        self.state = state
        self.executor = executor
        self._peak_profit_r = {}
        self._bars_in_trade = {}
        self._last_check_time = {}

    def evaluate_exits(self):
        """Run exit evaluation on all open positions. Called every brain cycle."""
        # Weekend protection: close/tighten before Friday gap
        self._weekend_protection()

        for symbol in list(self.executor._directions.keys()):
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

        tick = self.state.get_tick(symbol)
        if tick is None:
            return
        cur_price = float(tick.bid) if direction == "LONG" else float(tick.ask)

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
        if now - last > 3600:
            self._bars_in_trade[symbol] += 1
            self._last_check_time[symbol] = now
        bars = self._bars_in_trade.get(symbol, 0)

        # ═══ EXIT CHECKS ═══

        # 1. MOMENTUM DECAY — only if Sub0 still open (no TP1 hit yet)
        # Once TP1 hits, Sub2 runner is trailing SL's responsibility
        if peak_r >= 1.5 and profit_r < peak_r * 0.6:
            cfg = SYMBOLS.get(symbol)
            sub0_still_open = True
            if cfg:
                positions = self.executor.mt5.positions_get(symbol=symbol)
                if positions:
                    sub0_still_open = any(int(p.magic) == int(cfg.magic) for p in positions)

            if sub0_still_open:
                log.info("[%s] EXIT: Momentum decay (peak=%.1fR, now=%.1fR, gave back %.0f%%)",
                         symbol, peak_r, profit_r, (1 - profit_r / peak_r) * 100)
                self.executor.close_position(symbol, "DragonMomentumDecay")
                self._cleanup(symbol)
                return
            # else: TP1 hit, Sub2 running — let trailing SL handle it

        # 2. OPPOSING M15 — scaled by profit (don't kill 4R+ runners)
        m15_strength = self._get_opposing_strength(symbol, direction)
        if profit_r < 2.0:
            reversal_threshold = 0.7   # low profit: close on moderate reversal
        elif profit_r < 4.0:
            reversal_threshold = 0.9   # medium profit: need strong reversal
        else:
            reversal_threshold = 999   # high profit: let trailing SL handle

        if profit_r > 0.3 and m15_strength > reversal_threshold:
            log.info("[%s] EXIT: Opposing M15 (strength=%.2f, profit=%.1fR)",
                     symbol, m15_strength, profit_r)
            self.executor.close_position(symbol, "DragonOpposingSignal")
            self._cleanup(symbol)
            return

        # 3. TIME DECAY — profit-aware (don't kill winners)
        if bars > 20 and profit_r < 0.5:
            log.info("[%s] EXIT: Stale trade (%d bars, %.1fR)", symbol, bars, profit_r)
            self.executor.close_position(symbol, "DragonStaleTrade")
            self._cleanup(symbol)
            return

        if bars >= 40 and profit_r < 3.0:
            log.info("[%s] EXIT: Time decay (%d bars, %.1fR < 3R)", symbol, bars, profit_r)
            self.executor.close_position(symbol, "DragonTimeDecay")
            self._cleanup(symbol)
            return

        if bars >= 60:
            # Hard ceiling — even big winners (regime will have shifted)
            log.info("[%s] EXIT: Hard time limit (%d bars, %.1fR)", symbol, bars, profit_r)
            self.executor.close_position(symbol, "DragonTimeLimit")
            self._cleanup(symbol)
            return

        # 4. PROTECT BREAKEVEN: was > 1R, now near entry
        if peak_r >= 1.0 and profit_r <= 0.1:
            log.info("[%s] EXIT: Protecting BE (peak=%.1fR, now=%.1fR)", symbol, peak_r, profit_r)
            self.executor.close_position(symbol, "DragonProtectBE")
            self._cleanup(symbol)
            return

    def _weekend_protection(self):
        """Close non-crypto positions before weekend gap risk (Friday 20:00 UTC+)."""
        now = datetime.now(timezone.utc)
        if now.weekday() != 4 or now.hour < 20:
            return

        for symbol in list(self.executor._directions.keys()):
            if symbol.endswith("_scalp"):
                continue
            cfg = SYMBOLS.get(symbol)
            if cfg and cfg.category == "Crypto":
                continue
            if not self.executor.has_position(symbol):
                continue

            entry = self.executor._entry_prices.get(symbol, 0)
            sl_dist = self.executor._entry_sl_dist.get(symbol, 0)
            if entry <= 0 or sl_dist <= 0:
                continue

            tick = self.state.get_tick(symbol)
            if tick is None:
                continue
            direction = self.executor._directions.get(symbol, "FLAT")
            cur_price = float(tick.bid) if direction == "LONG" else float(tick.ask)
            profit_dist = (cur_price - entry) if direction == "LONG" else (entry - cur_price)
            profit_r = profit_dist / sl_dist if sl_dist > 0 else 0

            if profit_r < 1.5:
                log.info("[%s] WEEKEND CLOSE: %.1fR < 1.5R, not worth gap risk", symbol, profit_r)
                self.executor.close_position(symbol, "DragonWeekendClose")
                self._cleanup(symbol)
            else:
                log.info("[%s] WEEKEND HOLD: %.1fR >= 1.5R, keeping with trail", symbol, profit_r)

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

            opposing = 0.0
            if direction == "LONG":
                if float(ema_s[bi]) < float(ema_l[bi]): opposing += 0.4
                if int(st_dir[bi]) == -1: opposing += 0.4
                if bi > 2 and float(ema_s[bi] - ema_l[bi]) < float(ema_s[bi - 2] - ema_l[bi - 2]):
                    opposing += 0.2
            else:
                if float(ema_s[bi]) > float(ema_l[bi]): opposing += 0.4
                if int(st_dir[bi]) == 1: opposing += 0.4
                if bi > 2 and float(ema_s[bi] - ema_l[bi]) > float(ema_s[bi - 2] - ema_l[bi - 2]):
                    opposing += 0.2

            return min(1.0, opposing)
        except Exception:
            return 0.0

    def _cleanup(self, symbol):
        self._peak_profit_r.pop(symbol, None)
        self._bars_in_trade.pop(symbol, None)
        self._last_check_time.pop(symbol, None)

    def get_status(self, symbol) -> dict:
        return {
            "peak_profit_r": round(self._peak_profit_r.get(symbol, 0), 2),
            "bars_in_trade": self._bars_in_trade.get(symbol, 0),
        }
