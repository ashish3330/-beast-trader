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
        self._entry_time = {}  # V5: track when position first seen for min hold

    def evaluate_exits(self):
        """V5: DISABLED — trail system + ratchet handle all exits.
        Exit intelligence was closing trades at breakeven/tiny profit
        (RSI divergence, opposing signal, momentum decay) before
        trailing SL could lock real profit. Only weekend protection remains."""
        self._weekend_protection()
        # All other exits disabled — trail handles it
        return

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

        # V5: minimum hold time — don't exit trades less than 15 minutes old
        # Prevents RSI divergence / momentum decay from killing fresh entries
        import time as _time
        entry_time = self._entry_time.get(symbol, 0)
        if entry_time == 0:
            self._entry_time[symbol] = _time.time()
            return  # just started tracking, skip this cycle
        if _time.time() - entry_time < 900:  # 15 minutes
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

        # 1. MOMENTUM DECAY — protect profits that are fading
        # DON'T cut trades at 3R+ — trailing SL handles runners
        # Cut if peak was 1.0-3R and we gave back 40%+ (tightened from 1.5R/50%)
        if peak_r >= 1.0 and peak_r < 3.0 and profit_r < peak_r * 0.6:
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

        # 2a. RSI DIVERGENCE EXIT — early reversal detection
        # Price making new high but RSI dropping = bearish divergence (close longs)
        # Price making new low but RSI rising = bullish divergence (close shorts)
        if profit_r > 0.1 and profit_r < 3.0:
            divergence = self._check_rsi_divergence(symbol, direction)
            if divergence:
                log.info("[%s] EXIT: RSI divergence detected (profit=%.1fR) — early reversal",
                         symbol, profit_r)
                self.executor.close_position(symbol, "DragonRSIDivergence")
                self._cleanup(symbol)
                return

        # 2b. OPPOSING M15 — scaled by profit (don't kill 3R+ runners)
        m15_strength = self._get_opposing_strength(symbol, direction)
        if profit_r < 1.5:
            reversal_threshold = 0.6   # tightened: exit faster when small profit
        elif profit_r < 3.0:
            reversal_threshold = 0.8   # moderate threshold for medium profits
        else:
            reversal_threshold = 999   # never exit runners via M15

        if profit_r > 0.2 and m15_strength > reversal_threshold:
            log.info("[%s] EXIT: Opposing M15 (strength=%.2f, profit=%.1fR)",
                     symbol, m15_strength, profit_r)
            self.executor.close_position(symbol, "DragonOpposingSignal")
            self._cleanup(symbol)
            return

        # 3. TIME DECAY — tightened: cut dead weight faster
        if bars > 20 and profit_r < 0.3:
            # Stale: 20+ hours, barely moved — cut (was 30h)
            log.info("[%s] EXIT: Stale trade (%d bars, %.1fR)", symbol, bars, profit_r)
            self.executor.close_position(symbol, "DragonStaleTrade")
            self._cleanup(symbol)
            return

        if bars >= 40 and profit_r < 1.0:
            # 40+ hours and not even 1R — give up (was 60h)
            log.info("[%s] EXIT: Time decay (%d bars, %.1fR < 1R)", symbol, bars, profit_r)
            self.executor.close_position(symbol, "DragonTimeDecay")
            self._cleanup(symbol)
            return

        # NO hard time limit for winning trades — let trailing SL decide
        # A trade at 5R after 80 bars should NOT be force-closed

        # 4. PROTECT BREAKEVEN: was > 0.8R, now near entry (tightened from 1.0R)
        if peak_r >= 0.8 and profit_r <= 0.15:
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

    def _check_rsi_divergence(self, symbol, direction):
        """Detect RSI divergence — early reversal signal.
        LONG: price new high + RSI dropping = bearish divergence → close
        SHORT: price new low + RSI rising = bullish divergence → close
        """
        try:
            h1_df = self.state.get_candles(symbol, 60)
            if h1_df is None or len(h1_df) < 15:
                return False

            close = h1_df["close"].values.astype(np.float64)
            high = h1_df["high"].values.astype(np.float64)
            low = h1_df["low"].values.astype(np.float64)
            n = len(close)
            bi = n - 2  # completed bar

            if bi < 12:
                return False

            # Compute RSI(14) manually
            deltas = np.diff(close[:bi+1])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            if len(gains) < 14:
                return False
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:]) or 0.001
            rsi_now = 100 - 100 / (1 + avg_gain / avg_loss)

            avg_gain_prev = np.mean(gains[-19:-5]) if len(gains) >= 19 else avg_gain
            avg_loss_prev = np.mean(losses[-19:-5]) if len(losses) >= 19 else (avg_loss or 0.001)
            avg_loss_prev = avg_loss_prev or 0.001  # guard both branches
            rsi_prev = 100 - 100 / (1 + avg_gain_prev / avg_loss_prev)

            if direction == "LONG":
                # Bearish divergence: price at/near 10-bar high but RSI dropping
                price_near_high = close[bi] >= np.max(high[bi-10:bi]) * 0.998
                rsi_dropping = rsi_now < rsi_prev - 3  # RSI dropped 3+ points
                return price_near_high and rsi_dropping
            else:
                # Bullish divergence: price at/near 10-bar low but RSI rising
                price_near_low = close[bi] <= np.min(low[bi-10:bi]) * 1.002
                rsi_rising = rsi_now > rsi_prev + 3
                return price_near_low and rsi_rising

        except Exception:
            return False

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
        self._entry_time.pop(symbol, None)

    def get_status(self, symbol) -> dict:
        return {
            "peak_profit_r": round(self._peak_profit_r.get(symbol, 0), 2),
            "bars_in_trade": self._bars_in_trade.get(symbol, 0),
        }
