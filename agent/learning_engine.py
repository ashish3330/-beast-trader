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

import numpy as np

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

        # ── Market observation state ──
        self._market_obs: Dict[str, dict] = {}   # symbol -> {regime, volatility, trend, score_history}
        self._score_history: Dict[str, list] = {sym: [] for sym in SYMBOLS}  # last 100 scores
        self._regime_history: Dict[str, list] = {sym: [] for sym in SYMBOLS}  # regime transitions
        self._missed_trades: Dict[str, list] = {sym: [] for sym in SYMBOLS}  # signals we skipped
        self._last_observe_time = 0

        # Initialize SQLite journal
        self._init_db()
        self._load_recent_trades()

    def _init_db(self):
        """Create trade journal table if not exists."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
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
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
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
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
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
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
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

        log.info("AUTO-RETRAIN: Starting ML model retraining (deploy only if better)...")
        retrained = 0
        for sym in SYMBOLS:
            try:
                # Save current AUC before retraining
                old_metrics = self._meta_model._train_metrics.get(sym, {})
                old_auc = float(old_metrics.get("test_auc", 0))

                metrics = self._meta_model.train(sym, self._mt5, None)
                if metrics and metrics.get("status") == "ok":
                    new_auc = float(metrics.get("test_auc", 0))
                    if new_auc > old_auc:
                        # Better — save and deploy
                        self._meta_model.save(sym)
                        retrained += 1
                        log.info("RETRAIN %s: UPGRADED AUC %.3f → %.3f (deployed)",
                                 sym, old_auc, new_auc)
                    else:
                        # Worse or same — reload old model, discard new
                        self._meta_model.load(sym)
                        log.info("RETRAIN %s: REJECTED AUC %.3f → %.3f (keeping old)",
                                 sym, old_auc, new_auc)
                else:
                    reason = metrics.get("reason", "unknown") if metrics else "no result"
                    log.warning("RETRAIN %s failed: %s — keeping old model", sym, reason)
                    self._meta_model.load(sym)  # restore old
            except Exception as e:
                log.warning("RETRAIN %s error: %s — keeping old model", sym, e)
                try: self._meta_model.load(sym)
                except: pass

        if retrained > 0:
            log.info("AUTO-RETRAIN: %d/%d models upgraded. Reloading...", retrained, len(SYMBOLS))
            try:
                self._meta_model.load_all()
                log.info("AUTO-RETRAIN: Upgraded models reloaded into live brain")

                # Log to journal
                conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
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

    # ═══════════════════════════════════════════════════════════════
    #  MARKET OBSERVER — watches every tick cycle, learns patterns
    # ═══════════════════════════════════════════════════════════════

    def observe_market(self):
        """Called every brain cycle (1s). Watches market patterns and builds intelligence.
        This runs in the brain thread, so keep it fast (<10ms)."""
        try:
            now = time.time()
            # Throttle to every 5s (don't need per-second observation)
            if now - self._last_observe_time < 5:
                return
            self._last_observe_time = now

            agent = self.state.get_agent_state()
            scores = agent.get("scores", {})

            for sym in SYMBOLS:
                sym_scores = scores.get(sym, {})
                ls = float(sym_scores.get("long_score", 0))
                ss = float(sym_scores.get("short_score", 0))
                regime = sym_scores.get("regime", "unknown")
                gate = sym_scores.get("gate", "")

                # Track score history (rolling 100)
                best = max(ls, ss)
                self._score_history[sym].append({
                    "t": now, "ls": ls, "ss": ss, "best": best,
                    "regime": regime, "gate": gate,
                })
                if len(self._score_history[sym]) > 100:
                    self._score_history[sym] = self._score_history[sym][-100:]

                # Track regime transitions
                rh = self._regime_history[sym]
                if not rh or rh[-1]["regime"] != regime:
                    rh.append({"t": now, "regime": regime})
                    if len(rh) > 50:
                        self._regime_history[sym] = rh[-50:]

                # Track missed signals (score was close to threshold but didn't enter)
                from config import DRAGON_SYMBOL_MIN_SCORE
                min_scores = DRAGON_SYMBOL_MIN_SCORE.get(sym, {})
                threshold = min_scores.get(regime, 7.0)
                if best >= threshold * 0.85 and best < threshold and "BELOW_MIN" in str(gate):
                    self._missed_trades[sym].append({
                        "t": now, "best": best, "threshold": threshold, "regime": regime,
                    })
                    if len(self._missed_trades[sym]) > 20:
                        self._missed_trades[sym] = self._missed_trades[sym][-20:]

                # Build market observation summary
                sh = self._score_history[sym]
                if len(sh) >= 10:
                    recent_scores = [s["best"] for s in sh[-10:]]
                    score_trend = "rising" if recent_scores[-1] > np.mean(recent_scores[:5]) else "falling"
                    avg_score = np.mean(recent_scores)
                    max_score = max(recent_scores)
                    vol_h1 = self._get_h1_volatility(sym)

                    self._market_obs[sym] = {
                        "score_trend": score_trend,
                        "avg_score": round(avg_score, 1),
                        "max_recent": round(max_score, 1),
                        "regime": regime,
                        "volatility": round(vol_h1, 4) if vol_h1 else 0,
                        "missed_count": len(self._missed_trades.get(sym, [])),
                        "regime_changes_1h": self._count_regime_changes(sym, 3600),
                    }

            # Push to dashboard
            self.state.update_agent("market_observation", self._market_obs)

        except Exception as e:
            log.debug("Market observe error: %s", e)

    def _get_h1_volatility(self, symbol):
        """Get current H1 volatility (ATR / price)."""
        try:
            h1 = self.state.get_candles(symbol, 60)
            if h1 is None or len(h1) < 20:
                return None
            c = h1["close"].values.astype(float)
            h = h1["high"].values.astype(float)
            l = h1["low"].values.astype(float)
            tr = np.maximum(h[-14:] - l[-14:],
                            np.maximum(np.abs(h[-14:] - c[-15:-1]),
                                       np.abs(l[-14:] - c[-15:-1])))
            atr = float(np.mean(tr))
            price = float(c[-1])
            return atr / price if price > 0 else 0
        except Exception:
            return None

    def _count_regime_changes(self, symbol, seconds):
        """Count regime changes in the last N seconds."""
        rh = self._regime_history.get(symbol, [])
        cutoff = time.time() - seconds
        return sum(1 for r in rh if r["t"] >= cutoff)

    def get_market_observation(self, symbol=None):
        """Get market observation for dashboard."""
        if symbol:
            return self._market_obs.get(symbol, {})
        return dict(self._market_obs)

    def get_missed_signals(self, symbol=None):
        """Get missed trade signals."""
        if symbol:
            return list(self._missed_trades.get(symbol, []))
        return {sym: list(trades) for sym, trades in self._missed_trades.items() if trades}

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
                stats = self.get_all_stats()
                stats["market_obs"] = self._market_obs
                self.state.update_agent("learning_stats", stats)

            except Exception as e:
                log.warning("Learning loop error: %s", e)

            time.sleep(30)  # check every 30 seconds
