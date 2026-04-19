"""
Dragon Trader — Equity Guardian.
Real-time equity and position P&L monitor. Acts like a human trader watching the screen.
Runs every brain cycle (1s) and takes immediate action when needed.

Actions:
1. Cut sharp losers — position drops 1.5R in < 5 bars, close immediately
2. Protect winners — tighten SL when position hits profit milestones
3. Daily drawdown control — close all if day loss exceeds limit
4. Portfolio heat reduction — tighten all SLs when too many positions open
5. Rapid reversal detection — price reverses sharply after entry, close fast
"""
import time
import logging
from datetime import datetime, timezone
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS, DD_EMERGENCY_CLOSE, DAILY_LOSS_LIMIT_PCT, STARTING_BALANCE

log = logging.getLogger("dragon.guardian")


class EquityGuardian:
    """Watches equity and positions in real-time, takes protective action."""

    def __init__(self, state, executor):
        self.state = state
        self.executor = executor
        self._entry_time = {}       # symbol -> entry timestamp
        self._peak_pnl = {}         # symbol -> highest P&L seen
        self._baseline_pnl = {}     # symbol -> P&L snapshot at first sight
        self._last_equity = None
        self._day_start_equity = None
        self._last_day = None
        self._rapid_loss_count = 0  # count of rapid cuts today

    def monitor(self):
        """Called every brain cycle (1s). Check equity + all positions."""
        try:
            # Skip weekends — market closed, can't close positions
            now = datetime.now(timezone.utc)
            if now.weekday() >= 5:  # Saturday/Sunday
                return

            # Skip if market likely closed (Friday after 22:00 UTC)
            if now.weekday() == 4 and now.hour >= 22:
                return

            agent = self.state.get_agent_state()
            equity = float(agent.get("equity", STARTING_BALANCE))
            balance = float(agent.get("balance", STARTING_BALANCE))

            # Daily reset
            today = now.date()
            if self._last_day != today:
                self._last_day = today
                self._day_start_equity = equity
                self._rapid_loss_count = 0

            if self._day_start_equity is None:
                self._day_start_equity = equity

            # ═══ 1. DAILY DRAWDOWN EMERGENCY ═══
            day_loss_pct = (self._day_start_equity - equity) / self._day_start_equity * 100 if self._day_start_equity > 0 else 0
            if day_loss_pct >= DAILY_LOSS_LIMIT_PCT * 1.5:
                # 1.5x daily limit = emergency close all
                log.critical("GUARDIAN: Day loss %.1f%% >= %.1f%% — CLOSING ALL",
                             day_loss_pct, DAILY_LOSS_LIMIT_PCT * 1.5)
                self.executor.close_all("GuardianDayLoss")
                return

            # ═══ 2. MONITOR EACH POSITION ═══
            positions = self.executor.get_positions_info()
            for pos in positions:
                sym = pos.get("symbol", "")
                pnl = float(pos.get("pnl", 0))
                mode = pos.get("mode", "swing")
                ticket = pos.get("ticket", 0)

                if not sym:
                    continue

                # Guard NaN pnl from MT5
                if not np.isfinite(pnl):
                    continue

                # Track entry time — record CURRENT pnl as baseline for new positions
                if sym not in self._entry_time:
                    self._entry_time[sym] = time.time()
                    self._baseline_pnl[sym] = pnl  # snapshot P&L at first sight

                # Track peak P&L
                if sym not in self._peak_pnl:
                    self._peak_pnl[sym] = pnl
                else:
                    self._peak_pnl[sym] = max(self._peak_pnl[sym], pnl)

                time_in_trade = time.time() - self._entry_time.get(sym, time.time())
                peak = self._peak_pnl.get(sym, 0)

                # ─── SHARP LOSS CUT (tightened: 1.5% in 5min) ───
                # Position loses more than 1.5% of equity FROM WHEN WE FIRST SAW IT
                if sym not in self._baseline_pnl:
                    continue  # no baseline yet, skip sharp loss check
                baseline = self._baseline_pnl[sym]
                pnl_change = pnl - baseline  # negative = got worse since we started watching
                loss_pct = abs(pnl_change) / equity * 100 if pnl_change < 0 and equity > 0 else 0
                if loss_pct > 1.5 and time_in_trade < 300:
                    log.warning("GUARDIAN: %s sharp loss $%.2f (%.1f%% equity in %.0fs) — CUTTING",
                                sym, pnl, loss_pct, time_in_trade)
                    self.executor.close_position(sym, "GuardianSharpLoss")
                    self._cleanup(sym)
                    self._rapid_loss_count += 1
                    continue

                # ─── PROFIT GIVEBACK PROTECTION (tightened: protect 50% of peak) ───
                # Was +$X, now giving back more than 50% of peak
                if peak > 0 and pnl < peak * 0.5 and pnl > 0:
                    # Still in profit but gave back 50%+ of peak
                    # Only act if peak was meaningful (> 0.4% of equity)
                    if peak / equity * 100 > 0.4:
                        log.info("GUARDIAN: %s gave back %.0f%% of peak ($%.2f → $%.2f) — CLOSING to protect",
                                 sym, (1 - pnl / peak) * 100, peak, pnl)
                        self.executor.close_position(sym, "GuardianGiveback")
                        self._cleanup(sym)
                        continue

                # ─── STALE LOSER (tightened: 1.5h, 0.75% equity) ───
                # Position has gotten WORSE by >0.75% equity over 1.5 hours
                if pnl_change < 0 and time_in_trade > 5400 and loss_pct > 0.75:
                    log.info("GUARDIAN: %s losing $%.2f for %.1f hours — CUTTING stale loser",
                             sym, pnl, time_in_trade / 3600)
                    self.executor.close_position(sym, "GuardianStaleLoser")
                    self._cleanup(sym)
                    continue

            # ═══ 3. PORTFOLIO HEAT CHECK ═══
            open_count = len([p for p in positions if p.get("mode") == "swing"])
            total_risk = sum(abs(float(p.get("pnl", 0))) for p in positions if float(p.get("pnl", 0)) < 0)
            heat_pct = total_risk / equity * 100 if equity > 0 else 0

            if heat_pct > 4.0 and open_count >= 3:
                # Too much heat — close the worst loser
                worst = min(positions, key=lambda p: float(p.get("pnl", 0)))
                if float(worst.get("pnl", 0)) < 0:
                    log.warning("GUARDIAN: Portfolio heat %.1f%% with %d positions — closing worst (%s $%.2f)",
                                heat_pct, open_count, worst["symbol"], worst["pnl"])
                    self.executor.close_position(worst["symbol"], "GuardianHeatReduce")
                    self._cleanup(worst["symbol"])

            # ═══ 4. RAPID LOSS CIRCUIT BREAKER ═══
            if self._rapid_loss_count >= 3:
                log.critical("GUARDIAN: %d rapid cuts today — market is hostile, closing all",
                             self._rapid_loss_count)
                self.executor.close_all("GuardianRapidLoss")
                self._rapid_loss_count = 0  # reset after close

            # Cleanup closed positions
            open_syms = {p["symbol"] for p in positions}
            for sym in list(self._entry_time.keys()):
                if sym not in open_syms:
                    self._cleanup(sym)

            self._last_equity = equity

        except Exception as e:
            log.warning("Guardian error: %s", e)

    def _cleanup(self, symbol):
        self._entry_time.pop(symbol, None)
        self._peak_pnl.pop(symbol, None)
        self._baseline_pnl.pop(symbol, None)
