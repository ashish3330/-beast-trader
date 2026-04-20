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
        self._load_hour_performance()
        self._last_persist_time = 0

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
                , source TEXT DEFAULT 'live')
            """)
            # Add source column to existing DBs that don't have it
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN source TEXT DEFAULT 'live'")
            except Exception:
                pass  # column already exists
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
            # Observer persistence: hour performance survives restarts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hour_performance (
                    symbol TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    updated TEXT,
                    PRIMARY KEY (symbol, hour)
                )
            """)
            # Observer persistence: regime patterns
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    regime TEXT,
                    score REAL,
                    direction TEXT
                )
            """)
            # Observer persistence: missed signals
            conn.execute("""
                CREATE TABLE IF NOT EXISTS missed_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    score REAL,
                    threshold REAL,
                    regime TEXT,
                    direction TEXT
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

    def _load_hour_performance(self):
        """Load accumulated hour performance from DB on startup."""
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            if not hasattr(self, '_hour_perf_db'):
                self._hour_perf_db = {}  # (symbol, hour) -> {wins, losses, total_pnl}
            rows = conn.execute("SELECT symbol, hour, wins, losses, total_pnl FROM hour_performance").fetchall()
            for sym, h, w, l, pnl in rows:
                self._hour_perf_db[(sym, h)] = {"wins": w, "losses": l, "total_pnl": pnl}
            conn.close()
            if rows:
                log.info("Loaded %d hour performance records from DB", len(rows))
        except Exception as e:
            self._hour_perf_db = {}
            log.debug("Could not load hour performance: %s", e)

    def _persist_observer(self):
        """Save observer intelligence to DB (survives restarts). Called every 60s."""
        try:
            now = time.time()
            if now - self._last_persist_time < 60:
                return
            self._last_persist_time = now

            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            ts = datetime.now(timezone.utc).isoformat()

            # 1. Save hour performance — ACCUMULATE from trade journal, not recent trades
            # Query the FULL trade journal (source of truth) to rebuild hour stats
            rows = conn.execute(
                "SELECT symbol, session_hour, "
                "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
                "SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses, "
                "SUM(pnl) as total_pnl "
                "FROM trades WHERE session_hour IS NOT NULL "
                "GROUP BY symbol, session_hour"
            ).fetchall()
            for sym, h, w, l, pnl in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO hour_performance (symbol, hour, wins, losses, total_pnl, updated) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (sym, int(h), int(w), int(l), round(float(pnl), 2), ts)
                )
                self._hour_perf_db[(sym, int(h))] = {"wins": int(w), "losses": int(l), "total_pnl": float(pnl)}

            # 2. Save recent regime transitions (last 10 per symbol)
            for sym in SYMBOLS:
                rh = self._regime_history.get(sym, [])
                for r in rh[-5:]:  # last 5 transitions
                    conn.execute(
                        "INSERT INTO regime_log (timestamp, symbol, regime, score, direction) "
                        "SELECT ?, ?, ?, 0, '' WHERE NOT EXISTS "
                        "(SELECT 1 FROM regime_log WHERE timestamp=? AND symbol=?)",
                        (datetime.fromtimestamp(r["t"], tz=timezone.utc).isoformat(),
                         sym, r["regime"],
                         datetime.fromtimestamp(r["t"], tz=timezone.utc).isoformat(), sym)
                    )

            # 3. Save missed signals (for future analysis)
            for sym in SYMBOLS:
                missed = self._missed_trades.get(sym, [])
                for m in missed[-5:]:
                    conn.execute(
                        "INSERT INTO missed_signals (timestamp, symbol, score, threshold, regime, direction) "
                        "SELECT ?, ?, ?, ?, ?, ? WHERE NOT EXISTS "
                        "(SELECT 1 FROM missed_signals WHERE timestamp=? AND symbol=?)",
                        (datetime.fromtimestamp(m["t"], tz=timezone.utc).isoformat(),
                         sym, m["best"], m["threshold"], m["regime"], m.get("dir", ""),
                         datetime.fromtimestamp(m["t"], tz=timezone.utc).isoformat(), sym)
                    )

            # 4. Trim old data (keep last 30 days)
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            conn.execute("DELETE FROM regime_log WHERE timestamp < ?", (cutoff,))
            conn.execute("DELETE FROM missed_signals WHERE timestamp < ?", (cutoff,))

            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("Observer persist error: %s", e)

    def record_trade(self, symbol: str, direction: str, pnl: float,
                     entry_price: float = 0, exit_price: float = 0,
                     risk_pct: float = 0, score: float = 0,
                     regime: str = "", gate: str = "",
                     duration_bars: int = 0, r_multiple: float = 0,
                     exit_reason: str = "", source: str = "live"):
        """Record a completed trade to journal and update learning state."""
        now = datetime.now(timezone.utc)

        # Save to SQLite
        try:
            conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
            conn.execute("""
                INSERT INTO trades (timestamp, symbol, direction, entry_price, exit_price,
                    pnl, risk_pct, score, regime, gate, duration_bars, r_multiple,
                    session_hour, day_of_week, exit_reason, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now.isoformat(), symbol, direction, entry_price, exit_price,
                pnl, risk_pct, score, regime, gate, duration_bars, r_multiple,
                now.hour, now.weekday(), exit_reason, source,
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

    # ═══════════════════════════════════════════════════════════════
    #  OBSERVER → BRAIN FEEDBACK (real-time intelligence for decisions)
    # ═══════════════════════════════════════════════════════════════

    def get_market_quality(self, symbol: str) -> dict:
        """Get real-time market quality assessment for a symbol.
        Called by brain before entry — combines ALL observer intelligence into one decision.

        Returns entry_bias (0.5-1.4) that scales risk based on market conditions.
        """
        obs = self._market_obs.get(symbol, {})
        if not obs:
            return {"entry_bias": 1.0, "score_momentum": 0, "regime_stable": True,
                    "near_miss_rate": 0, "volatility_regime": "stable", "reasons": []}

        reasons = []
        bias = 1.0

        # ─── 1. SCORE MOMENTUM (slope of recent scores) ───
        slope = obs.get("score_slope", 0)
        accel = obs.get("acceleration", 0)
        if slope > 0.1:
            bias *= 1.1
            reasons.append(f"score_rising({slope:+.2f})")
        elif slope < -0.1:
            bias *= 0.9
            reasons.append(f"score_falling({slope:+.2f})")
        if accel > 1.0:
            bias *= 1.05
            reasons.append("momentum_building")

        # ─── 2. DIRECTION CONSISTENCY ───
        dir_con = obs.get("dir_consistency", 0.5)
        if dir_con > 0.8:
            bias *= 1.1  # strong directional conviction
            reasons.append(f"dir_consistent({dir_con:.0%})")
        elif dir_con < 0.4:
            bias *= 0.9  # confused, no clear direction
            reasons.append("dir_confused")

        # ─── 3. REGIME STABILITY ───
        regime_changes = obs.get("regime_changes_1h", 0)
        regime_stable = obs.get("regime_stable", True)
        if not regime_stable or regime_changes >= 4:
            bias *= 0.8
            reasons.append(f"choppy({regime_changes}changes/hr)")
        elif regime_changes == 0:
            bias *= 1.05
            reasons.append("regime_locked")

        # ─── 4. VOLATILITY ───
        vol_expanding = obs.get("vol_expanding", False)
        if vol_expanding:
            bias *= 1.08  # expanding vol = moves happening
            reasons.append("vol_expanding")

        # ─── 5. PRICE INTELLIGENCE ───
        pi = obs.get("price_intel", {})
        trend_str = pi.get("trend_strength", 0)
        if abs(trend_str) > 1.0:
            bias *= 1.1  # strong H1 trend
            reasons.append(f"strong_trend({trend_str:+.1f})")
        if pi.get("at_key_level", False):
            bias *= 0.9  # at round number = reversal risk
            reasons.append("at_key_level")
        body_ratio = pi.get("h1_body_ratio", 0)
        if body_ratio > 0.7:
            bias *= 1.05  # strong conviction candles
            reasons.append("big_bodies")
        m5_mom = pi.get("m5_momentum", 0)
        if abs(m5_mom) > 1.5:
            bias *= 1.05  # M5 has momentum
            reasons.append(f"m5_momentum({m5_mom:+.1f})")

        # ─── 6. NEAR MISS RATE ───
        missed = obs.get("missed_count", 0)
        sh = self._score_history.get(symbol, [])
        near_miss_rate = missed / max(len(sh), 1)
        if near_miss_rate > 0.3:
            bias *= 1.05
            reasons.append("signal_building")

        # ─── 7. HOUR-OF-DAY PERFORMANCE ───
        current_wr = obs.get("current_hour_wr", 50)
        best_hours = obs.get("best_hours", [])
        worst_hours = obs.get("worst_hours", [])
        current_h = datetime.now(timezone.utc).hour
        if current_h in worst_hours:
            bias *= 0.85
            reasons.append(f"bad_hour({current_h})")
        elif current_h in best_hours:
            bias *= 1.1
            reasons.append(f"good_hour({current_h})")

        bias = max(0.5, min(1.4, bias))

        return {
            "entry_bias": round(bias, 2),
            "score_momentum": round(slope, 3),
            "regime_stable": regime_stable,
            "near_miss_rate": round(near_miss_rate, 2),
            "volatility_regime": "expanding" if vol_expanding else "stable",
            "reasons": reasons,
        }

    def should_skip_symbol(self, symbol: str) -> tuple:
        """Check if observer data suggests skipping this symbol entirely.
        Returns (skip: bool, reason: str).
        TOUGH checks — protects capital in bad conditions."""
        obs = self._market_obs.get(symbol, {})
        if not obs:
            return False, ""

        # 1. Skip if regime is extremely choppy (4+ changes per hour — tightened from 5)
        if obs.get("regime_changes_1h", 0) >= 4:
            return True, f"choppy_regime ({obs['regime_changes_1h']} changes/hr)"

        # 2. Skip if market is dead (no momentum building — tightened thresholds)
        if obs.get("max_recent", 10) < 3.0 and obs.get("avg_score", 10) < 2.5:
            return True, f"dead_market (max={obs.get('max_recent')}, avg={obs.get('avg_score')})"

        # 3. Skip if scores are consistently falling (momentum dying)
        slope = obs.get("score_slope", 0)
        accel = obs.get("acceleration", 0)
        if slope < -0.12 and accel < -0.3:
            return True, f"momentum_dying (slope={slope:.2f}, accel={accel:.2f})"

        # 4. Skip if at key price level with weak trend (reversal trap)
        pi = obs.get("price_intel", {})
        if pi.get("at_key_level") and abs(pi.get("trend_strength", 0)) < 0.5:
            return True, f"key_level_no_trend (trend={pi.get('trend_strength')})"

        # 5. Skip during worst performing hours for this symbol (tightened WR < 35%)
        worst = obs.get("worst_hours", [])
        current_h = datetime.now(timezone.utc).hour
        if current_h in worst and obs.get("current_hour_wr", 50) < 35:
            return True, f"worst_hour ({current_h}:00 WR={obs.get('current_hour_wr')}%)"

        # 6. Skip if direction is confused AND volatility contracting (whipsaw trap)
        dir_con = obs.get("dir_consistency", 0.5)
        if dir_con < 0.35 and not obs.get("vol_expanding", False):
            return True, f"confused_low_vol (dir_consistency={dir_con:.0%})"

        return False, ""

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
        """Retrain ML meta-label models with latest data. Runs daily at 04:00 UTC.
        Incorporates observer intelligence into training via sample weighting:
        - Trades during best hours get higher weight (model learns to favor them)
        - Trades during worst hours get lower weight (model learns to avoid them)
        - Symbols with good recent PF get priority retraining
        """
        if not hasattr(self, '_meta_model') or self._meta_model is None:
            return
        if not hasattr(self, '_mt5') or self._mt5 is None:
            return

        # Build observer context for training (hour weights from learned performance)
        observer_context = {}
        for sym in SYMBOLS:
            hour_weights = {}
            for (s, h), stats in self._hour_perf_db.items():
                if s == sym:
                    total = stats["wins"] + stats["losses"]
                    if total >= 3:
                        wr = stats["wins"] / total
                        # Good hours get weight 1.2, bad hours get 0.7
                        hour_weights[h] = 1.2 if wr >= 0.6 else (0.7 if wr <= 0.3 else 1.0)
            observer_context[sym] = {"hour_weights": hour_weights}

        # Store context on the model so train() can use it
        if hasattr(self._meta_model, '_observer_context'):
            self._meta_model._observer_context = observer_context
        else:
            self._meta_model._observer_context = observer_context

        log.info("AUTO-RETRAIN: Starting with observer context (%d symbols with hour data)...",
                 sum(1 for v in observer_context.values() if v["hour_weights"]))
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
            # Load last synced ticket from DB to avoid duplicates on restart
            self._last_deal_ticket = 0
            try:
                conn = sqlite3.connect(str(JOURNAL_DB), timeout=5.0)
                conn.execute("CREATE TABLE IF NOT EXISTS sync_state (key TEXT PRIMARY KEY, value INTEGER)")
                row = conn.execute("SELECT value FROM sync_state WHERE key='last_deal_ticket'").fetchone()
                conn.close()
                if row and row[0]:
                    self._last_deal_ticket = int(row[0])
                    log.info("Deal sync: resumed from ticket %d", self._last_deal_ticket)
            except Exception:
                pass

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
                # MT5 deal type: 0=BUY, 1=SELL — but for EXIT deals, this is the CLOSING action
                # A closing SELL means the original position was LONG, and vice versa
                # entry field: 0=DEAL_ENTRY_IN, 1=DEAL_ENTRY_OUT
                deal_type = int(d.type)
                is_exit = hasattr(d, 'entry') and int(d.entry) == 1
                if is_exit:
                    # Invert: closing SELL → was LONG, closing BUY → was SHORT
                    direction = "LONG" if deal_type == 1 else "SHORT"
                else:
                    direction = "LONG" if deal_type == 0 else "SHORT"
                exit_reason = str(d.comment) or "SL/TP"
                exit_price = float(d.price) if hasattr(d, 'price') else 0
                self.record_trade(
                    symbol=str(d.symbol), direction=direction,
                    pnl=float(d.profit),
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    source="mt5_deal",
                )
                new_count += 1

                # Feed ALL learning modules from deal sync (not just journal)
                sym_str = str(d.symbol)
                deal_pnl = float(d.profit)
                equity = float(self.state.get_agent_state().get("equity", 1000))
                dollar_risk = equity * 0.012  # 1.2% risk estimate
                deal_r_mult = deal_pnl / dollar_risk if dollar_risk > 0 else 0

                # Trade intelligence: SL re-entry tracking + pattern recording
                if hasattr(self, '_trade_intel') and self._trade_intel and is_exit:
                    comment_lower = exit_reason.lower()
                    if "sl" in comment_lower or "stop" in comment_lower:
                        self._trade_intel.record_stoploss(sym_str, direction, exit_price)
                    try:
                        self._trade_intel.record_pattern(
                            symbol=sym_str, direction=direction, score=0,
                            regime="unknown", m15_dir="FLAT",
                            pnl=deal_pnl, r_multiple=deal_r_mult)
                    except Exception:
                        pass

                # RL learner: scoring weight + exit rule learning
                if hasattr(self, '_rl_learner') and self._rl_learner:
                    try:
                        self._rl_learner.record_outcome(
                            symbol=sym_str, direction=direction, pnl=deal_pnl,
                            r_multiple=deal_r_mult, score=0, regime="unknown",
                            exit_reason=exit_reason)
                    except Exception:
                        pass

                # Level memory: learn which prices cause bounces/breaks
                if hasattr(self, '_level_memory') and self._level_memory:
                    try:
                        evt = "sl_hit" if "sl" in exit_reason.lower() else "tp_hit" if "tp" in exit_reason.lower() else "close"
                        self._level_memory.record_level_event(sym_str, exit_price, evt)
                    except Exception:
                        pass

                self._last_deal_ticket = max(self._last_deal_ticket, int(d.ticket))

            if new_count > 0:
                log.info("SYNC: Recorded %d new trades from MT5 deal history", new_count)
                # Persist last deal ticket so we don't re-sync after restart
                try:
                    conn = sqlite3.connect(str(JOURNAL_DB), timeout=5.0)
                    conn.execute("CREATE TABLE IF NOT EXISTS sync_state (key TEXT PRIMARY KEY, value INTEGER)")
                    conn.execute("INSERT OR REPLACE INTO sync_state (key, value) VALUES ('last_deal_ticket', ?)",
                                 (self._last_deal_ticket,))
                    conn.commit()
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            log.debug("MT5 deal sync error: %s", e)

    # ═══════════════════════════════════════════════════════════════
    #  MARKET OBSERVER — watches every tick cycle, learns patterns
    # ═══════════════════════════════════════════════════════════════

    def observe_market(self):
        """Called every brain cycle (1s). Deep market intelligence engine.
        Watches patterns, learns what works, builds real-time context.
        This runs in the brain thread, so keep it fast (<10ms)."""
        try:
            now = time.time()
            if now - self._last_observe_time < 1:
                return  # observe every 1s (was 5s)
            self._last_observe_time = now

            agent = self.state.get_agent_state()
            scores = agent.get("scores", {})
            from config import DRAGON_SYMBOL_MIN_SCORE

            for sym in SYMBOLS:
                sym_scores = scores.get(sym, {})
                ls = float(sym_scores.get("long_score", 0))
                ss = float(sym_scores.get("short_score", 0))
                regime = sym_scores.get("regime", "unknown")
                gate = sym_scores.get("gate", "")
                best = max(ls, ss)
                direction = "LONG" if ls > ss else "SHORT" if ss > ls else "FLAT"

                # ═══ 1. SCORE HISTORY (rolling 200) ═══
                self._score_history[sym].append({
                    "t": now, "ls": ls, "ss": ss, "best": best,
                    "dir": direction, "regime": regime, "gate": gate,
                })
                if len(self._score_history[sym]) > 200:
                    self._score_history[sym] = self._score_history[sym][-200:]

                # ═══ 2. REGIME TRACKING ═══
                rh = self._regime_history[sym]
                if not rh or rh[-1]["regime"] != regime:
                    rh.append({"t": now, "regime": regime})
                    if len(rh) > 50:
                        self._regime_history[sym] = rh[-50:]

                # ═══ 3. MISSED SIGNALS ═══
                min_scores = DRAGON_SYMBOL_MIN_SCORE.get(sym, {})
                threshold = min_scores.get(regime, 7.0)
                if best >= threshold * 0.85 and best < threshold and "BELOW_MIN" in str(gate):
                    self._missed_trades[sym].append({
                        "t": now, "best": best, "threshold": threshold,
                        "regime": regime, "dir": direction,
                    })
                    if len(self._missed_trades[sym]) > 30:
                        self._missed_trades[sym] = self._missed_trades[sym][-30:]

                # ═══ 4. DEEP PATTERN ANALYSIS (needs 20+ data points) ═══
                sh = self._score_history[sym]
                if len(sh) < 15:
                    continue

                recent = sh[-15:]
                recent_scores = [s["best"] for s in recent]
                recent_dirs = [s["dir"] for s in recent]
                recent_regimes = [s["regime"] for s in recent]

                # Score momentum: linear regression slope
                x = np.arange(len(recent_scores))
                slope = float(np.polyfit(x, recent_scores, 1)[0]) if len(recent_scores) >= 3 else 0
                score_trend = "rising" if slope > 0.05 else "falling" if slope < -0.05 else "flat"

                # Direction consistency: how often same direction in last 15 observations
                long_pct = sum(1 for d in recent_dirs if d == "LONG") / len(recent_dirs)
                short_pct = sum(1 for d in recent_dirs if d == "SHORT") / len(recent_dirs)
                dir_consistency = max(long_pct, short_pct)
                dominant_dir = "LONG" if long_pct > short_pct else "SHORT" if short_pct > long_pct else "FLAT"

                # Regime stability
                unique_regimes = len(set(recent_regimes))
                regime_changes_1h = self._count_regime_changes(sym, 3600)

                # Score acceleration: is the rate of change increasing?
                if len(sh) >= 30:
                    older = [s["best"] for s in sh[-30:-15]]
                    recent_avg = np.mean(recent_scores)
                    older_avg = np.mean(older)
                    acceleration = recent_avg - older_avg  # positive = building
                else:
                    acceleration = 0

                # ═══ 5. PRICE ACTION INTELLIGENCE ═══
                h1 = self.state.get_candles(sym, 60)
                m5 = self.state.get_candles(sym, 5)
                price_intel = self._analyze_price_patterns(sym, h1, m5)

                # ═══ 6. VOLATILITY ANALYSIS ═══
                vol_h1 = self._get_h1_volatility(sym)
                vol_expanding = False
                if hasattr(self, '_prev_vol') and sym in self._prev_vol and vol_h1:
                    vol_expanding = vol_h1 > self._prev_vol[sym] * 1.1
                if not hasattr(self, '_prev_vol'):
                    self._prev_vol = {}
                if vol_h1:
                    self._prev_vol[sym] = vol_h1

                # ═══ 7. HOUR-OF-DAY PERFORMANCE (which hours work?) ═══
                hour_perf = self._get_hour_performance(sym)

                # ═══ BUILD OBSERVATION ═══
                self._market_obs[sym] = {
                    "score_trend": score_trend,
                    "score_slope": round(slope, 3),
                    "avg_score": round(np.mean(recent_scores), 1),
                    "max_recent": round(max(recent_scores), 1),
                    "acceleration": round(acceleration, 2),
                    "dir_consistency": round(dir_consistency, 2),
                    "dominant_dir": dominant_dir,
                    "regime": regime,
                    "regime_stable": unique_regimes <= 2,
                    "regime_changes_1h": regime_changes_1h,
                    "volatility": round(vol_h1, 5) if vol_h1 else 0,
                    "vol_expanding": vol_expanding,
                    "missed_count": len(self._missed_trades.get(sym, [])),
                    "price_intel": price_intel,
                    "best_hours": hour_perf.get("best_hours", []),
                    "worst_hours": hour_perf.get("worst_hours", []),
                    "current_hour_wr": hour_perf.get("current_hour_wr", 50),
                }

            # Push to dashboard
            self.state.update_agent("market_observation", self._market_obs)

            # Persist observer intelligence to DB every 60s
            self._persist_observer()

        except Exception as e:
            log.debug("Market observe error: %s", e)

    def _analyze_price_patterns(self, symbol, h1, m5):
        """Detect key price patterns from candle data."""
        result = {"trend_strength": 0, "at_key_level": False,
                  "m5_momentum": 0, "h1_body_ratio": 0}
        try:
            if h1 is not None and len(h1) >= 20:
                c = h1["close"].values.astype(float)
                h = h1["high"].values.astype(float)
                l = h1["low"].values.astype(float)
                o = h1["open"].values.astype(float)

                # H1 trend strength: EMA slope normalized
                if len(c) >= 20:
                    ema = np.convolve(c, np.ones(8)/8, mode='valid')
                    if len(ema) >= 5:
                        slope = (ema[-1] - ema[-5]) / (np.std(c[-20:]) + 1e-10)
                        result["trend_strength"] = round(float(np.clip(slope, -2, 2)), 2)

                # At key level: near round number or recent high/low
                price = float(c[-1])
                if price > 0:
                    # Round number proximity (within 0.2% of round level)
                    magnitude = 10 ** max(0, int(np.log10(price)) - 1)
                    nearest_round = round(price / magnitude) * magnitude
                    dist_pct = abs(price - nearest_round) / price * 100
                    result["at_key_level"] = dist_pct < 0.2

                # H1 body ratio (big bodies = conviction, small = indecision)
                body = abs(c[-1] - o[-1])
                wick = h[-1] - l[-1]
                result["h1_body_ratio"] = round(float(body / wick if wick > 0 else 0), 2)

            if m5 is not None and len(m5) >= 12:
                c5 = m5["close"].values.astype(float)
                # M5 momentum: last 6 bars (30 min) direction
                if len(c5) >= 7:
                    change = (c5[-1] - c5[-7]) / (np.std(c5[-12:]) + 1e-10)
                    result["m5_momentum"] = round(float(np.clip(change, -3, 3)), 2)

        except Exception:
            pass
        return result

    def _get_hour_performance(self, symbol):
        """Which hours of day have this symbol performed best/worst?
        Uses PERSISTED DB data — survives restarts, accumulates over weeks."""
        result = {"best_hours": [], "worst_hours": [], "current_hour_wr": 50}
        try:
            # Combine DB data + in-memory recent trades for fullest picture
            hour_pnl = {}

            # 1. Load from persisted DB (accumulated across ALL restarts)
            for (sym, h), stats in self._hour_perf_db.items():
                if sym == symbol:
                    hour_pnl[h] = dict(stats)  # copy

            # 2. Overlay with in-memory recent trades (freshest data)
            trades = self._recent_trades.get(symbol, [])
            for t in trades:
                h = t.get("hour", 12)
                if h not in hour_pnl:
                    hour_pnl[h] = {"wins": 0, "losses": 0, "total_pnl": 0}
                if t["pnl"] > 0:
                    hour_pnl[h]["wins"] += 1
                else:
                    hour_pnl[h]["losses"] += 1
                hour_pnl[h]["total_pnl"] += t["pnl"]

            # Find best/worst hours (need >= 3 trades in that hour)
            for h, stats in hour_pnl.items():
                total = stats["wins"] + stats["losses"]
                if total >= 3:
                    wr = stats["wins"] / total * 100
                    if wr >= 60:
                        result["best_hours"].append(h)
                    elif wr <= 30:
                        result["worst_hours"].append(h)

            # Current hour WR
            current_h = datetime.now(timezone.utc).hour
            if current_h in hour_pnl:
                ch = hour_pnl[current_h]
                total = ch["wins"] + ch["losses"]
                if total >= 2:
                    result["current_hour_wr"] = round(ch["wins"] / total * 100)

        except Exception:
            pass
        return result

    def get_learned_session_mult(self, symbol: str, hour_utc: int) -> float:
        """Return a learned session multiplier based on accumulated hour performance.
        Needs >= 10 trades in that hour to override hardcoded defaults.
        Returns 1.0 if insufficient data (caller falls through to hardcoded).

        Mapping: WR >= 55% → 1.15, >= 60% → 1.20, <= 35% → 0.85, <= 30% → 0.80
        PnL-weighted: if WR is neutral but PnL is strongly positive/negative, adjust.
        """
        key = (symbol, hour_utc)
        stats = self._hour_perf_db.get(key)
        if not stats:
            return 1.0

        wins = stats["wins"]
        losses = stats["losses"]
        total = wins + losses
        if total < 10:
            return 1.0  # not enough data to learn from

        wr = wins / total
        pnl = stats["total_pnl"]
        avg_pnl = pnl / total

        # WR-based multiplier
        if wr >= 0.65:
            mult = 1.20
        elif wr >= 0.55:
            mult = 1.15
        elif wr >= 0.45:
            mult = 1.0
        elif wr >= 0.35:
            mult = 0.85
        else:
            mult = 0.80

        # PnL adjustment — if avg PnL is strongly positive despite low WR (big winners),
        # don't penalize as hard. If avg PnL is negative despite high WR (small winners, big losers),
        # reduce the boost.
        if avg_pnl > 0 and mult < 1.0:
            mult = min(1.0, mult + 0.05)  # partially rehabilitate profitable bad-WR hours
        elif avg_pnl < 0 and mult > 1.0:
            mult = max(1.0, mult - 0.05)  # reduce boost for unprofitable good-WR hours

        return round(mult, 2)

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

            time.sleep(5)  # check every 5 seconds (was 30s — too slow for deal sync)
