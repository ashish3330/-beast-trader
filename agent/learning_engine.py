"""
Dragon Trader — Learning Engine.
Watches every trade, learns what works, adapts parameters automatically.
Runs as a background thread alongside the brain.

Capabilities:
1. Persistent trade journal (SQLite) — survives restarts
2. Rolling performance tracking (PF, WR, Sharpe per symbol, per regime)
3. Adaptive risk scaling — increase risk on winning symbols, decrease on losing
4. Auto regime detection — if recent PF drops, reduce aggression
5. Weekly ML model retraining trigger
6. Performance alerts to dashboard
"""
import time
import logging
import threading
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, SYMBOLS

log = logging.getLogger("dragon.learner")

JOURNAL_DB = DB_PATH.parent / "trade_journal.db"
JOURNAL_DB.parent.mkdir(parents=True, exist_ok=True)


class LearningEngine:
    """Watches trades, tracks performance, adapts parameters."""

    def __init__(self, state, master_brain, executor):
        self.state = state
        self.master_brain = master_brain
        self.executor = executor
        self._running = False
        self._thread = None

        # Adaptive risk multipliers per symbol (1.0 = normal)
        self.symbol_risk_mult: Dict[str, float] = {sym: 1.0 for sym in SYMBOLS}

        # Performance windows
        self._recent_trades: Dict[str, list] = {sym: [] for sym in SYMBOLS}  # last 20 per sym

        # Initialize SQLite journal
        self._init_db()
        self._load_recent_trades()

    def _init_db(self):
        """Create trade journal table if not exists."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    risk_pct REAL,
                    score REAL,
                    regime TEXT,
                    gate TEXT,
                    duration_bars INTEGER,
                    r_multiple REAL,
                    session_hour INTEGER,
                    day_of_week INTEGER,
                    exit_reason TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    equity REAL,
                    trades INTEGER,
                    wins INTEGER,
                    pnl REAL,
                    pf REAL,
                    max_dd REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    action TEXT,
                    detail TEXT
                )
            """)
            conn.commit()
            conn.close()
            log.info("Trade journal initialized: %s", JOURNAL_DB)
        except Exception as e:
            log.error("Failed to init trade journal: %s", e)

    def _load_recent_trades(self):
        """Load last 20 trades per symbol from journal on startup."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB))
            for sym in SYMBOLS:
                rows = conn.execute(
                    "SELECT pnl, r_multiple, regime, session_hour FROM trades "
                    "WHERE symbol=? ORDER BY id DESC LIMIT 20", (sym,)
                ).fetchall()
                self._recent_trades[sym] = [
                    {"pnl": r[0], "r_mult": r[1], "regime": r[2], "hour": r[3]}
                    for r in reversed(rows)
                ]
            conn.close()

            # Compute initial risk multipliers from loaded history
            self._update_risk_multipliers()
            total = sum(len(v) for v in self._recent_trades.values())
            log.info("Loaded %d recent trades from journal", total)
        except Exception as e:
            log.warning("Could not load trade history: %s", e)

    def record_trade(self, symbol: str, direction: str, pnl: float,
                     entry_price: float = 0, exit_price: float = 0,
                     risk_pct: float = 0, score: float = 0,
                     regime: str = "", gate: str = "",
                     duration_bars: int = 0, r_multiple: float = 0,
                     exit_reason: str = ""):
        """Record a completed trade to journal and update learning state."""
        now = datetime.now(timezone.utc)

        # Save to SQLite
        try:
            conn = sqlite3.connect(str(JOURNAL_DB))
            conn.execute("""
                INSERT INTO trades (timestamp, symbol, direction, entry_price, exit_price,
                    pnl, risk_pct, score, regime, gate, duration_bars, r_multiple,
                    session_hour, day_of_week, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now.isoformat(), symbol, direction, entry_price, exit_price,
                pnl, risk_pct, score, regime, gate, duration_bars, r_multiple,
                now.hour, now.weekday(), exit_reason,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.error("Failed to save trade: %s", e)

        # Update recent trades
        self._recent_trades.setdefault(symbol, []).append({
            "pnl": pnl, "r_mult": r_multiple, "regime": regime, "hour": now.hour
        })
        # Keep last 20
        if len(self._recent_trades[symbol]) > 20:
            self._recent_trades[symbol] = self._recent_trades[symbol][-20:]

        # Recalculate risk multipliers
        self._update_risk_multipliers()

        log.info("JOURNAL: %s %s pnl=%.2f R=%.2f risk_mult=%.2f",
                 symbol, direction, pnl, r_multiple,
                 self.symbol_risk_mult.get(symbol, 1.0))

    def _update_risk_multipliers(self):
        """Adaptive risk: increase for winning symbols, decrease for losing.

        Uses rolling 10-trade PF per symbol:
        - PF > 2.0: risk x 1.3 (pressing winners)
        - PF 1.5-2.0: risk x 1.15
        - PF 1.0-1.5: risk x 1.0 (normal)
        - PF 0.7-1.0: risk x 0.7 (pulling back)
        - PF < 0.7: risk x 0.5 (survival mode)
        """
        for sym in SYMBOLS:
            trades = self._recent_trades.get(sym, [])
            if len(trades) < 5:
                self.symbol_risk_mult[sym] = 1.0
                continue

            recent = trades[-10:]
            wins = [t["pnl"] for t in recent if t["pnl"] > 0]
            losses = [abs(t["pnl"]) for t in recent if t["pnl"] < 0]
            gross_p = sum(wins) if wins else 0
            gross_l = sum(losses) if losses else 0.01
            pf = gross_p / gross_l

            if pf > 2.0:
                self.symbol_risk_mult[sym] = 1.3
            elif pf > 1.5:
                self.symbol_risk_mult[sym] = 1.15
            elif pf > 1.0:
                self.symbol_risk_mult[sym] = 1.0
            elif pf > 0.7:
                self.symbol_risk_mult[sym] = 0.7
            else:
                self.symbol_risk_mult[sym] = 0.5

    def get_risk_multiplier(self, symbol: str) -> float:
        """Get adaptive risk multiplier for a symbol (called by MasterBrain)."""
        return self.symbol_risk_mult.get(symbol, 1.0)

    def get_symbol_stats(self, symbol: str) -> dict:
        """Get rolling stats for dashboard."""
        trades = self._recent_trades.get(symbol, [])
        if not trades:
            return {"trades": 0, "wr": 0, "pf": 0, "avg_r": 0, "risk_mult": 1.0}

        wins = sum(1 for t in trades if t["pnl"] > 0)
        gross_p = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_l = sum(abs(t["pnl"]) for t in trades if t["pnl"] < 0) or 0.01
        avg_r = sum(t.get("r_mult", 0) for t in trades) / len(trades)

        return {
            "trades": len(trades),
            "wr": round(wins / len(trades) * 100, 1),
            "pf": round(gross_p / gross_l, 2),
            "avg_r": round(avg_r, 2),
            "risk_mult": self.symbol_risk_mult.get(symbol, 1.0),
        }

    def get_all_stats(self) -> dict:
        """Get learning engine status for dashboard."""
        stats = {}
        for sym in SYMBOLS:
            stats[sym] = self.get_symbol_stats(sym)
        return stats

    def save_daily_stats(self):
        """Save end-of-day summary to journal."""
        try:
            now = datetime.now(timezone.utc)
            today = now.strftime("%Y-%m-%d")
            agent = self.state.get_agent_state()
            equity = agent.get("equity", 0)

            # Count today's trades from journal
            conn = sqlite3.connect(str(JOURNAL_DB))
            row = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), SUM(pnl) "
                "FROM trades WHERE date(timestamp)=?", (today,)
            ).fetchone()
            trades = row[0] or 0
            wins = row[1] or 0
            pnl = row[2] or 0

            conn.execute("""
                INSERT OR REPLACE INTO daily_stats (date, equity, trades, wins, pnl, pf, max_dd)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (today, equity, trades, wins, pnl, 0, 0))
            conn.commit()
            conn.close()

            log.info("DAILY STATS: %s equity=$%.2f trades=%d wins=%d pnl=$%.2f",
                     today, equity, trades, wins, pnl)
        except Exception as e:
            log.error("Failed to save daily stats: %s", e)

    def set_meta_model(self, meta_model):
        """Set reference to SignalModel for retraining."""
        self._meta_model = meta_model

    def set_mt5(self, mt5):
        """Set MT5 connection for retraining."""
        self._mt5 = mt5

    def _retrain_models(self):
        """Retrain ML meta-label models with latest data. Runs daily at 04:00 UTC."""
        if not hasattr(self, '_meta_model') or self._meta_model is None:
            return
        if not hasattr(self, '_mt5') or self._mt5 is None:
            return

        log.info("AUTO-RETRAIN: Starting ML model retraining...")
        retrained = 0
        for sym in SYMBOLS:
            try:
                metrics = self._meta_model.train(sym, self._mt5, None)
                if metrics and metrics.get("status") == "ok":
                    auc = metrics.get("test_auc", 0)
                    log.info("RETRAINED %s: AUC=%.3f FilteredPF=%.2f",
                             sym, auc, metrics.get("filtered_pf", 0))
                    retrained += 1
                else:
                    reason = metrics.get("reason", "unknown") if metrics else "no result"
                    log.warning("RETRAIN %s failed: %s", sym, reason)
            except Exception as e:
                log.warning("RETRAIN %s error: %s", sym, e)

        if retrained > 0:
            log.info("AUTO-RETRAIN: %d/%d models updated. Reloading...", retrained, len(SYMBOLS))
            try:
                self._meta_model.load_all()
                log.info("AUTO-RETRAIN: Models reloaded into live brain")

                # Log to journal
                conn = sqlite3.connect(str(JOURNAL_DB))
                conn.execute("INSERT INTO learning_log (timestamp, action, detail) VALUES (?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(), "retrain",
                     f"Retrained {retrained}/{len(SYMBOLS)} models"))
                conn.commit()
                conn.close()
            except Exception as e:
                log.error("AUTO-RETRAIN reload failed: %s", e)

    def _save_live_candles_to_cache(self):
        """Save all timeframe candles from SharedState to cache for retraining."""
        try:
            import pickle
            from pathlib import Path
            cache_dir = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)

            sym_names = {
                "XAUUSD": "xauusd", "XAGUSD": "XAGUSD", "BTCUSD": "BTCUSD",
                "NAS100.r": "NAS100_r", "JPN225ft": "JPN225ft", "USDJPY": "USDJPY",
            }
            tf_map = {1: "m1", 5: "m5", 15: "m15", 60: "h1"}
            updated = 0

            for sym, name in sym_names.items():
                for tf_min, tf_label in tf_map.items():
                    df = self.state.get_candles(sym, tf_min)
                    if df is not None and len(df) > 100:
                        fname = f"raw_{tf_label}_{name}.pkl"
                        pickle.dump(df, open(cache_dir / fname, "wb"))
                        updated += 1

            if updated > 0:
                log.info("CACHE UPDATE: Saved %d candle files (6 symbols x 4 TFs)", updated)
        except Exception as e:
            log.warning("Cache update error: %s", e)

    def start(self):
        """Start background learning thread."""
        self._running = True
        self._thread = threading.Thread(target=self._learning_loop, daemon=True,
                                        name="LearningEngine")
        self._thread.start()
        log.info("Learning Engine started (adaptive risk, trade journal, auto-retrain)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _sync_mt5_deals(self):
        """Poll MT5 for closed trades and record any we haven't seen yet."""
        if not hasattr(self, '_mt5') or self._mt5 is None:
            return
        if not hasattr(self, '_last_deal_ticket'):
            self._last_deal_ticket = 0

        try:
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            deals = self._mt5.history_deals_get(now - timedelta(hours=24), now)
            if not deals:
                return

            new_count = 0
            for d in deals:
                if int(d.magic) < 8000 or float(d.profit) == 0:
                    continue
                ticket = int(d.ticket)
                if ticket <= self._last_deal_ticket:
                    continue

                # Record new deal (dedup by ticket number, not fragile date+pnl)
                direction = "LONG" if int(d.type) == 0 else "SHORT"
                self.record_trade(
                    symbol=str(d.symbol), direction=direction,
                    pnl=float(d.profit), exit_reason=str(d.comment) or "SL/TP",
                )
                new_count += 1

                self._last_deal_ticket = max(self._last_deal_ticket, int(d.ticket))

            if new_count > 0:
                log.info("SYNC: Recorded %d new trades from MT5 deal history", new_count)
        except Exception as e:
            log.debug("MT5 deal sync error: %s", e)

    def _learning_loop(self):
        """Background loop: deal sync, daily stats, cache update, auto-retrain."""
        last_daily = None
        last_retrain = None
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                today = now.date()

                # Sync closed trades from MT5 every cycle
                self._sync_mt5_deals()

                # Daily stats at midnight UTC
                if last_daily != today:
                    if last_daily is not None:
                        self.save_daily_stats()
                    last_daily = today

                # Auto-retrain daily at 04:00 UTC (low activity, fresh data)
                if now.hour == 4 and last_retrain != today:
                    self._save_live_candles_to_cache()
                    self._retrain_models()
                    last_retrain = today

                # Push learning stats to state for dashboard
                self.state.update_agent("learning_stats", self.get_all_stats())

            except Exception as e:
                log.warning("Learning loop error: %s", e)

            time.sleep(30)  # check every 30 seconds
