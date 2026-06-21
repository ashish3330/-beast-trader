"""
Dragon Trader — J.A.R.V.I.S. Dashboard.
Industry-grade real-time trading terminal with WebSocket push.
Port 8888 | Flask-SocketIO | lightweight-charts | Single-file HUD.
"""
import sys
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS, STARTING_BALANCE, DASHBOARD_PORT, TIMEFRAMES

log = logging.getLogger("dragon.dashboard")


def _sanitize(obj):
    """Recursively convert numpy types to Python native for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

app = Flask(__name__)
app.config["SECRET_KEY"] = "dragon-jarvis-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
IST = ZoneInfo("Asia/Kolkata")

# ── v2 endpoints (additive, do not touch v1) ──
try:
    from dashboard import v2_api as _v2_api
    _v2_api.register_v2(app, socketio)
except Exception as _e:
    log.warning("v2 dashboard endpoints disabled: %s", _e)
    _v2_api = None

# ── pro dashboard (Vue 3 SPA at /pro) — DISABLED 2026-05-03 ──
# User reverted to legacy UI. Files preserved (dashboard/pro_dashboard.py,
# dashboard/v2_api.py) for resume. Re-enable by uncommenting these 3 lines.
#
# try:
#     from dashboard.pro_dashboard import pro_dashboard_bp as _pro_bp
#     app.register_blueprint(_pro_bp)
# except Exception as _e:
#     log.warning("pro dashboard disabled: %s", _e)

# Shared state reference — set by run.py
_state = None
_executor = None

# ── Dashboard's OWN MT5 connection (port 18814, separate from agent) ──
_dash_mt5 = None
_dash_mt5_fails = 0
_dash_last_data = {"equity": 0, "balance": 0, "profit": 0}
_dash_last_positions = []
_dash_last_pos_map = {}

def _get_dash_mt5():
    """Get dashboard's dedicated MT5 connection. Auto-reconnects with backoff."""
    global _dash_mt5, _dash_mt5_fails
    if _dash_mt5 is not None:
        try:
            _dash_mt5.account_info()
            _dash_mt5_fails = 0
            return _dash_mt5
        except Exception as e:
            log.debug("Dashboard MT5 health check failed: %s", e)
            try: _dash_mt5.shutdown()
            except Exception: pass
            _dash_mt5 = None

    if _dash_mt5_fails > 5:
        _dash_mt5_fails -= 1  # slow retry backoff
        return None

    try:
        from mt5linux import MetaTrader5
        m = MetaTrader5(host='localhost', port=18814)
        m.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")
        from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
        m.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        _dash_mt5 = m
        # Only log first connect + every 50th reconnect (still visible if storm) — was flooding logs
        prev_was_none = (_dash_mt5_fails > 0)
        _dash_mt5_fails = 0
        if prev_was_none:
            log.info("Dashboard MT5 reconnected on port 18814")
        else:
            log.debug("Dashboard MT5 connected on port 18814")
        return m
    except Exception as e:
        _dash_mt5_fails += 1
        if _dash_mt5_fails <= 3:
            log.warning("Dashboard MT5 connect failed (%d/5): %s", _dash_mt5_fails, e)
        return None


def init_dashboard(state, executor=None):
    global _state, _executor
    _state = state
    _executor = executor
    # Wire v2 module to the same shared state/executor.
    try:
        from dashboard import v2_api
        v2_api.init_v2_state(state=state, executor=executor)
    except Exception as e:
        log.debug("v2_api init_v2_state skipped: %s", e)


# ═══════════════════════════════════════════════════════════════
# SESSION DETECTION
# ═══════════════════════════════════════════════════════════════
def _get_session():
    """Return current trading session based on UTC hour."""
    utc_h = datetime.now(timezone.utc).hour
    if 0 <= utc_h < 7:
        return ("ASIA", "#aa55ff")       # purple
    elif 7 <= utc_h < 12:
        return ("LONDON", "#0088ff")     # blue
    elif 12 <= utc_h < 16:
        return ("LON+NY", "#00ff88")     # green
    elif 16 <= utc_h < 21:
        return ("NEW YORK", "#ffaa00")   # gold
    else:
        return ("CLOSED", "#ff3355")     # red


# ═══════════════════════════════════════════════════════════════
# BACKGROUND PUSH THREADS
# ═══════════════════════════════════════════════════════════════
def _push_ticks():
    """Push tick data every 500ms."""
    while True:
        time.sleep(0.5)
        if _state is None:
            continue
        try:
            ticks = {}
            for sym, cfg in SYMBOLS.items():
                t = _state.get_tick(sym)
                if t:
                    hist = _state.get_tick_history(sym, 50)
                    sparkline = [h.bid for h in hist] if hist else []
                    ticks[sym] = {
                        "bid": t.bid, "ask": t.ask,
                        "spread": round((t.ask - t.bid) * (10 ** cfg.digits), 1),
                        "digits": cfg.digits,
                        "sparkline": sparkline,
                    }
            # Fetch account + positions from DEDICATED dashboard bridge (18814)
            mt5 = _get_dash_mt5()
            account_data = {"equity": 0, "balance": 0, "profit": 0}
            pos_map = {}
            positions_list = []

            if mt5:
                try:
                    info = mt5.account_info()
                    if info:
                        account_data = {
                            "equity": float(info.equity),
                            "balance": float(info.balance),
                            "profit": float(info.profit),
                        }
                    raw_pos = mt5.positions_get()
                    if not raw_pos:
                        # Retry with fresh connection
                        global _dash_mt5
                        _dash_mt5 = None
                        mt5 = _get_dash_mt5()
                        if mt5:
                            raw_pos = mt5.positions_get()
                    if raw_pos:
                        # Aggregate PnL per symbol for scanner (all subs combined)
                        sym_pnl = {}
                        sym_side = {}
                        for p in raw_pos:
                            if int(p.magic) < 8000: continue
                            sym = str(p.symbol)
                            pnl = float(p.profit)
                            side = "BUY" if int(p.type) == 0 else "SELL"
                            # Scalp magic = base + 100 exactly. Check against each symbol's scalp magic.
                            _pm = int(p.magic)
                            _scalp_magics = {cfg.magic + 100 for cfg in SYMBOLS.values()}
                            mode = "scalp" if _pm in _scalp_magics else "swing"
                            sym_pnl[sym] = sym_pnl.get(sym, 0.0) + pnl
                            sym_side[sym] = side  # all subs share same side
                            positions_list.append({
                                "symbol": sym, "type": side, "pnl": round(pnl, 2),
                                "volume": float(p.volume), "price_open": float(p.price_open),
                                "sl": float(p.sl), "tp": float(p.tp),
                                "magic": int(p.magic), "mode": mode,
                                "ticket": int(p.ticket),
                            })
                        for sym in sym_pnl:
                            pos_map[sym] = {"side": sym_side[sym], "pnl": round(sym_pnl[sym], 2)}
                except Exception as e:
                    log.debug("MT5 position fetch error: %s", e)

            # Cache good data, serve cached on failure
            if account_data["equity"] > 0:
                _dash_last_data.update(account_data)
                globals()['_dash_last_positions'] = positions_list
                globals()['_dash_last_pos_map'] = pos_map
            ticks["_account"] = _dash_last_data if account_data["equity"] == 0 else account_data
            ticks["_pos_map"] = _dash_last_pos_map if not pos_map else pos_map
            ticks["_positions"] = _dash_last_positions if not positions_list else positions_list

            # Include scores + MTF intelligence in tick push (real-time, not 5s delay)
            agent = _state.get_agent_state() if _state else {}
            if agent.get("scores"):
                ticks["_scores"] = agent["scores"]
            if agent.get("mtf_intelligence"):
                ticks["_mtf"] = agent["mtf_intelligence"]
            if agent.get("master_brain_status"):
                ticks["_master"] = agent["master_brain_status"]
            if agent.get("equity_history"):
                ticks["_eq_hist"] = agent["equity_history"][-300:]

            socketio.emit("tick_update", _sanitize(ticks))
        except Exception as e:
            log.debug("tick push error: %s", e)


def _push_chart():
    """Push chart candle data every 1s."""
    while True:
        time.sleep(1.0)
        if _state is None:
            continue
        try:
            charts = {}
            for sym in SYMBOLS:
                for tf in TIMEFRAMES:
                    df = _state.get_candles(sym, tf)
                    if df is not None and len(df) > 0:
                        # Send last 200 candles
                        tail = df.tail(200)
                        candles = []
                        volumes = []
                        for _, row in tail.iterrows():
                            ts = int(row["time"].timestamp()) if hasattr(row["time"], "timestamp") else int(row["time"])
                            candles.append({
                                "time": ts,
                                "open": round(float(row["open"]), 5),
                                "high": round(float(row["high"]), 5),
                                "low": round(float(row["low"]), 5),
                                "close": round(float(row["close"]), 5),
                            })
                            vol = int(row.get("tick_volume", 0)) if "tick_volume" in row.index else 0
                            volumes.append({"time": ts, "value": vol})
                        charts[f"{sym}_{tf}"] = {"candles": candles, "volumes": volumes}

                # EMA overlays from indicators
                ind = _state.get_indicators(sym)
                if ind:
                    charts[f"{sym}_indicators"] = {
                        "ema20": ind.get("ema20"),
                        "ema50": ind.get("ema50"),
                        "ema200": ind.get("ema200"),
                    }
            socketio.emit("chart_update", charts)
        except Exception as e:
            log.debug("chart push error: %s", e)


def _journal_trade_log(limit=50):
    """Recent CLOSED trades from the durable journal `trades` table (kept synced
    by the learner from MT5 deals), newest first, plus today's closed PnL (UTC
    day). Filters to >= fresh-start baseline if scripts/_comparison_baseline_*.json
    exists so the dashboard only shows post-reset trades."""
    import sqlite3
    import json as _bj
    from config import DB_PATH
    from pathlib import Path as _BP
    rows, today_pnl = [], 0.0
    # Look for the most recent baseline file. Currently a fixed-name path; if
    # the user does a new fresh-start later they should overwrite the same
    # file or this helper will keep using the old baseline.
    base_iso = None
    _bp = _BP(__file__).resolve().parent.parent / "scripts" / "_comparison_baseline_20260621.json"
    if _bp.exists():
        try:
            _b = _bj.loads(_bp.read_text())
            base_iso = _b.get("reset_at_iso")
        except Exception:
            base_iso = None
    try:
        with sqlite3.connect(str(DB_PATH.parent / "trade_journal.db"), timeout=3.0) as c:
            if base_iso:
                cur = c.execute(
                    "SELECT timestamp, symbol, direction, pnl, exit_reason FROM trades "
                    "WHERE timestamp >= ? ORDER BY id DESC LIMIT ?",
                    (base_iso, limit))
            else:
                cur = c.execute(
                    "SELECT timestamp, symbol, direction, pnl, exit_reason FROM trades "
                    "ORDER BY id DESC LIMIT ?", (limit,))
            for ts, sym, d, pnl, reason in cur.fetchall():
                try:
                    tstr = datetime.fromisoformat(ts).astimezone(IST).strftime("%m-%d %H:%M")
                except Exception:
                    tstr = str(ts)[:16]
                rows.append({
                    "timestamp": tstr, "symbol": str(sym),
                    "direction": str(d or "").lower(),
                    "pnl": round(float(pnl or 0), 2),
                    "action": str(reason or "CLOSE"),
                })
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_pnl = float(c.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE timestamp LIKE ?",
                (today_utc + "%",)).fetchone()[0] or 0)
    except Exception as e:
        log.debug("journal trade log read failed: %s", e)
    return rows, round(today_pnl, 2)


def _push_stats():
    """Push full stats every 1s — real-time dashboard."""
    while True:
        time.sleep(1.0)
        if _state is None:
            continue
        try:
            agent = _state.get_agent_state()
            session_name, session_color = _get_session()

            # Build scores breakdown
            scores = agent.get("scores", {})
            ml_conf = agent.get("model_confidence", {})
            positions = agent.get("positions", [])
            trade_log = agent.get("trade_log", [])
            equity_history = agent.get("equity_history", [])
            # feature_importance removed from dashboard (unused)
            mode = agent.get("mode", "HYBRID")

            # Position map for scanner — aggregate PnL across all subs per symbol
            pos_map = {}
            for p in positions:
                sym = p.get("symbol", "")
                pnl = p.get("pnl", 0)
                side = p.get("type", "FLAT")
                if sym in pos_map:
                    pos_map[sym]["pnl"] = round(pos_map[sym]["pnl"] + pnl, 2)
                else:
                    pos_map[sym] = {"side": side, "pnl": pnl}

            master_brain = agent.get("master_brain_status", {})
            mtf_intel = agent.get("mtf_intelligence", {})

            # Fetch closed trade history from MT5 — starting from fresh-start
            # baseline if available, else last 7 days. 2026-06-21: baseline
            # filter lets the dashboard show ONLY post-reset trades for the
            # 3-week strategy comparison window.
            mt5_trade_log = []
            today_closed_pnl = 0.0
            mt5 = _get_dash_mt5()
            if mt5:
                try:
                    from datetime import timedelta
                    import json as _bj
                    from pathlib import Path as _BP
                    now_utc = datetime.now(timezone.utc)
                    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                    # Try fresh-start baseline; fall back to 7d window.
                    fresh_from = None
                    _bp = (_BP(__file__).resolve().parent.parent
                           / "scripts" / "_comparison_baseline_20260621.json")
                    if _bp.exists():
                        try:
                            _b = _bj.loads(_bp.read_text())
                            fresh_from = datetime.fromisoformat(
                                _b["reset_at_iso"].replace("Z", "+00:00"))
                        except Exception:
                            fresh_from = None
                    from_dt = fresh_from if fresh_from is not None else (now_utc - timedelta(days=7))
                    deals = mt5.history_deals_get(from_dt, now_utc)
                    if deals:
                        for d in deals:
                            if int(d.magic) < 8000 or float(d.profit) == 0:
                                continue
                            deal_time = datetime.fromtimestamp(float(d.time), tz=timezone.utc)
                            mt5_trade_log.append({
                                "timestamp": deal_time.astimezone(IST).strftime("%m-%d %H:%M"),
                                "symbol": str(d.symbol),
                                "direction": "long" if int(d.type) == 0 else "short",
                                "pnl": round(float(d.profit), 2),
                                "volume": float(d.volume),
                                "action": str(d.comment) or "CLOSE",
                                "magic": int(d.magic),
                            })
                            # Sum today's closed PnL
                            if deal_time >= today_start:
                                today_closed_pnl += float(d.profit)
                except Exception as e:
                    log.debug("MT5 deal history error: %s", e)

            # Trade log: prefer live MT5 deals (has volume/magic). When the
            # dashboard MT5 connection is down, fall back to the DURABLE journal
            # (real closed trades) — NOT the stale in-memory entry-event log,
            # which was making the trade list show stale/incorrect rows.
            if mt5_trade_log:
                combined_log = list(reversed(mt5_trade_log[-50:]))
            else:
                combined_log, today_closed_pnl = _journal_trade_log(limit=50)

            # Real daily PnL = today's closed trades + open position PnL
            open_pnl = float(agent.get("equity", 0)) - float(agent.get("balance", 0))
            real_daily_pnl = round(today_closed_pnl + open_pnl, 2)

            data = {
                "equity": agent.get("equity", 0),
                "balance": agent.get("balance", 0),
                "profit": round(open_pnl, 2),
                "dd_pct": agent.get("dd_pct", 0),
                "daily_loss": agent.get("daily_loss", 0),
                "daily_pnl": real_daily_pnl,
                "today_closed_pnl": round(today_closed_pnl, 2),
                "cycle": agent.get("cycle", 0),
                "running": agent.get("running", False),
                "mode": mode,
                "positions": positions,
                "pos_map": pos_map,
                "session": session_name,
                "session_color": session_color,
                "scores": scores,
                "ml_confidence": ml_conf,
                "trade_log": combined_log,
                "equity_history": equity_history[-300:] if equity_history else [],
                "risk_pct": agent.get("risk_pct", 0),
                "num_positions": len(positions),
                "time": datetime.now(IST).strftime("%H:%M:%S IST"),
                "master_brain": master_brain,
                "mtf_intelligence": mtf_intel,
                "learning_stats": agent.get("learning_stats", {}),
            }
            # Sanitize numpy types (bool_, int64 etc) that break JSON
            socketio.emit("stats_update", _sanitize(data))
        except Exception as e:
            log.error("stats push error: %s", e, exc_info=True)


# ═══════════════════════════════════════════════════════════════
# SOCKETIO EVENT HANDLERS
# ═══════════════════════════════════════════════════════════════
@socketio.on("select_symbol")
def handle_select_symbol(data):
    log.info("Dashboard: select symbol %s", data)


@socketio.on("select_timeframe")
def handle_select_timeframe(data):
    log.info("Dashboard: select timeframe %s", data)


@socketio.on("close_all")
def handle_close_all(data=None):
    if _executor:
        _executor.close_all("DashboardCloseAll")
        emit("action_result", {"action": "close_all", "status": "ok"})
    else:
        emit("action_result", {"action": "close_all", "status": "error", "msg": "No executor"})


@socketio.on("close_symbol")
def handle_close_symbol(data):
    sym = data.get("symbol", "") if data else ""
    if _executor and sym:
        _executor.close_position(sym, "DashboardClose")
        emit("action_result", {"action": "close_symbol", "status": "ok", "symbol": sym})
    else:
        emit("action_result", {"action": "close_symbol", "status": "error"})


@socketio.on("close_losing")
def handle_close_losing(data=None):
    if _executor:
        agent = _state.get_agent_state() if _state else {}
        positions = agent.get("positions", [])
        closed = 0
        for p in positions:
            if p.get("pnl", 0) < 0:
                _executor.close_position(p.get("symbol", ""), "DashboardCloseLosing")
                closed += 1
        emit("action_result", {"action": "close_losing", "status": "ok", "closed": closed})
    else:
        emit("action_result", {"action": "close_losing", "status": "error"})


# ═══════════════════════════════════════════════════════════════
# REST FALLBACK
# ═══════════════════════════════════════════════════════════════
def _get_mt5_positions():
    """Fetch positions from dashboard's dedicated MT5 bridge."""
    mt5 = _get_dash_mt5()
    if not mt5: return _dash_last_positions
    try:
        raw = mt5.positions_get()
        if not raw: return []
        return [{"symbol": str(p.symbol), "type": "BUY" if int(p.type)==0 else "SELL",
                 "pnl": round(float(p.profit),2), "volume": float(p.volume),
                 "price_open": float(p.price_open), "sl": float(p.sl), "tp": float(p.tp),
                 "magic": int(p.magic), "mode": "scalp" if int(p.magic)>=8200 else "swing"}
                for p in raw if int(p.magic) >= 8000]
    except Exception:
        return []

@app.route("/api/data")
def api_data():
    if _state is None:
        return jsonify({"error": "State not initialized"})
    agent = _state.get_agent_state()
    ticks = {}
    for sym, cfg in SYMBOLS.items():
        t = _state.get_tick(sym)
        if t:
            ticks[sym] = {"bid": t.bid, "ask": t.ask,
                          "spread": (t.ask - t.bid) * (10 ** cfg.digits), "digits": cfg.digits}
    return jsonify({
        "equity": agent.get("equity", 0), "balance": agent.get("balance", 0),
        "profit": agent.get("profit", 0), "dd_pct": agent.get("dd_pct", 0),
        "positions": _get_mt5_positions(), "ticks": ticks,
        "scores": agent.get("scores", {}),
        "model_confidence": agent.get("model_confidence", {}),
        "cycle": agent.get("cycle", 0),
        "mode": agent.get("mode", ""),
        "time": datetime.now(IST).strftime("%H:%M:%S IST"),
    })


@app.route("/api/risk_locks")
def api_risk_locks():
    """Active re-entry cooldowns (DB-durable `cooldowns` table) + MasterBrain
    blacklist (from shared agent state). Drives the RISK LOCKS panel."""
    import sqlite3, time as _t
    from config import DB_PATH
    now = _t.time()
    cooldowns, blacklist = [], []
    try:
        with sqlite3.connect(str(DB_PATH.parent / "trade_journal.db"), timeout=3.0) as c:
            for sym, expiry, blocked, reason in c.execute(
                    "SELECT symbol, expiry, blocked, reason FROM cooldowns "
                    "WHERE expiry > ? ORDER BY expiry DESC", (now,)).fetchall():
                cooldowns.append({
                    "symbol": sym, "mins_left": round((float(expiry) - now) / 60.0, 1),
                    "blocked": blocked or "BOTH", "reason": reason or "",
                })
    except Exception as e:
        log.debug("risk_locks cooldown read failed: %s", e)
    if _state is not None:
        ag = _state.get_agent_state()
        bl = ag.get("mb_blacklisted", {}) or {}
        losses = ag.get("mb_symbol_losses", {}) or {}
        for sym, expiry in bl.items():
            try:
                mins = round((float(expiry) - now) / 60.0, 1)
            except Exception:
                mins = 0.0
            if mins > 0:
                blacklist.append({"symbol": sym, "mins_left": mins,
                                  "losses": int(losses.get(sym, 0))})
        blacklist.sort(key=lambda b: -b["mins_left"])
    return jsonify({"cooldowns": cooldowns, "blacklist": blacklist})


@app.route("/api/connection_health")
def api_connection_health():
    """MT5 connection health surface. Reads connection_events from journal,
    plus the live degraded streak from agent state. Frontend renders a badge:
    GREEN if no recent reconnects, AMBER if any, RED if degraded right now.
    """
    import sqlite3
    from config import DB_PATH
    out = {
        "degraded_streak": 0,
        "reconnects_24h": 0,
        "reconnects_7d": 0,
        "last_event": None,
        "events_recent": [],
        "status": "GREEN",
    }
    try:
        agent = _state.get_agent_state() if _state else {}
        out["degraded_streak"] = int(agent.get("mt5_degraded_streak", 0))
        out["last_event"] = agent.get("mt5_last_reconnect")
    except Exception:
        pass
    try:
        with sqlite3.connect(str(DB_PATH), timeout=3.0) as c:
            now = time.time()
            cutoff_24h = now - 86400
            cutoff_7d = now - 7 * 86400
            r24 = c.execute(
                "SELECT COUNT(*) FROM connection_events WHERE ts > ?",
                (cutoff_24h,),
            ).fetchone()
            r7 = c.execute(
                "SELECT COUNT(*) FROM connection_events WHERE ts > ?",
                (cutoff_7d,),
            ).fetchone()
            out["reconnects_24h"] = int(r24[0]) if r24 else 0
            out["reconnects_7d"] = int(r7[0]) if r7 else 0
            recent = c.execute(
                "SELECT ts, cause, downtime_ms, attempts FROM connection_events "
                "ORDER BY ts DESC LIMIT 20"
            ).fetchall()
            out["events_recent"] = [
                {"ts": r[0], "cause": r[1], "downtime_ms": r[2], "attempts": r[3]}
                for r in recent
            ]
    except sqlite3.OperationalError:
        # Table doesn't exist yet — first run before any reconnect.
        pass
    except Exception as e:
        out["error"] = str(e)
    if out["degraded_streak"] > 0:
        out["status"] = "RED"
    elif out["reconnects_24h"] > 5:
        out["status"] = "AMBER"
    return jsonify(out)


@app.route("/api/strategy_symbol_halts")
def api_strategy_symbol_halts():
    """Per-(strategy, symbol) consec-loss counters + active halts.

    Reads from the MasterBrain in-memory state via the global _state (also
    persisted to agent_state DB for restart survival). Returns a compact view
    for the dashboard so the user can see WHO is close to being halted and
    WHICH (strategy, symbol) combos are currently blocked.
    """
    out = {"threshold": 10, "enabled": True, "counters": [], "halted": []}
    try:
        from config import (PER_STRATEGY_SYMBOL_KILL_ENABLED,
                            PER_STRATEGY_SYMBOL_KILL_LOSSES)
        out["enabled"] = bool(PER_STRATEGY_SYMBOL_KILL_ENABLED)
        out["threshold"] = int(PER_STRATEGY_SYMBOL_KILL_LOSSES)
    except Exception:
        pass
    if _state is None:
        return jsonify(out)
    agent = _state.get_agent_state() if _state else {}
    losses = agent.get("mb_strategy_symbol_losses", {}) or {}
    halted = agent.get("mb_strategy_symbol_halted", {}) or {}
    # losses: {"strategy|symbol": count}
    for key, cnt in losses.items():
        try:
            strat, sym = key.split("|", 1)
            out["counters"].append({"strategy": strat, "symbol": sym,
                                    "consec_losses": int(cnt),
                                    "to_halt": max(0, out["threshold"] - int(cnt))})
        except Exception:
            continue
    out["counters"].sort(key=lambda x: x["consec_losses"], reverse=True)
    for key, halt_date in halted.items():
        try:
            strat, sym = key.split("|", 1)
            out["halted"].append({"strategy": strat, "symbol": sym,
                                  "halted_on": halt_date,
                                  "auto_clear": "next UTC midnight"})
        except Exception:
            continue
    return jsonify(out)


@app.route("/api/strategy_breakdown")
def api_strategy_breakdown():
    """Per-strategy PnL breakdown since the fresh-start baseline.

    Reads scripts/_comparison_baseline_20260621.json for the baseline
    timestamp, pulls MT5 deals from that point onwards, groups by magic-range
    → strategy bucket (momentum, fvg, sr, smabo, fib50), aggregates per-
    strategy + per-symbol stats. Drives the STRATEGY BATTLE dashboard panel.
    """
    import json as _json
    import time as _t
    from collections import defaultdict
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path
    out = {"baseline_iso": None, "elapsed_days": 0,
           "strategies": [], "per_symbol": {},
           "current_equity": 0, "baseline_equity": 0, "account_pnl": 0}
    baseline_p = (Path(__file__).resolve().parent.parent
                  / "scripts" / "_comparison_baseline_20260621.json")
    if not baseline_p.exists():
        out["error"] = "no baseline file — run fresh-start reset first"
        return jsonify(out)
    try:
        baseline = _json.loads(baseline_p.read_text())
        from_iso = baseline["reset_at_iso"]
        from_unix = _dt.fromisoformat(from_iso.replace("Z", "+00:00")).timestamp()
        baseline_eq = float(baseline.get("baseline_equity", 0))
    except Exception as e:
        out["error"] = f"baseline parse: {e}"
        return jsonify(out)
    out["baseline_iso"] = from_iso
    out["baseline_equity"] = baseline_eq
    out["elapsed_days"] = round((_t.time() - from_unix) / 86400.0, 2)

    base_magics = set()
    for sym, cfg in SYMBOLS.items():
        try:
            base_magics.add(int(cfg.magic))
        except Exception:
            pass

    def magic_to_strat(m):
        try:
            mi = int(m)
        except Exception:
            return "other"
        for base in base_magics:
            off = mi - base
            if off < 0 or off >= 5000:
                continue
            if off >= 4000:
                return "fib50"
            if off >= 3000:
                return "smabo"
            if off >= 2000:
                return "sr"
            if off >= 1000:
                return "fvg"
            return "momentum"
        return "other"

    mt5 = _get_dash_mt5()
    if mt5 is None:
        out["error"] = "MT5 bridge unavailable"
        return jsonify(out)
    try:
        info = mt5.account_info()
        out["current_equity"] = round(float(info.equity), 2)
        out["account_pnl"] = round(float(info.equity) - baseline_eq, 2)
    except Exception as e:
        out["error"] = f"account_info: {e}"
        return jsonify(out)

    try:
        from_dt = _dt.fromtimestamp(from_unix, _tz.utc)
        now_dt = _dt.now(_tz.utc)
        deals = mt5.history_deals_get(from_dt, now_dt) or []
    except Exception as e:
        out["error"] = f"history_deals_get: {e}"
        return jsonify(out)

    by_strat = defaultdict(list)
    by_sym_strat = defaultdict(lambda: defaultdict(list))
    for d in deals:
        try:
            if int(d.type) not in (0, 1):
                continue
            if int(d.entry) != 1:   # only close-side deals (entry=1 = exit)
                continue
        except Exception:
            continue
        strat = magic_to_strat(d.magic)
        p = float(d.profit)
        by_strat[strat].append(p)
        by_sym_strat[strat][str(d.symbol)].append(p)

    for strat in ("momentum", "fvg", "sr", "smabo", "fib50", "other"):
        pnls = by_strat.get(strat, [])
        if not pnls:
            continue
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = round(sum(pnls), 2)
        wr = round(100.0 * len(wins) / len(pnls), 1)
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) < 0 else 999.0
        out["strategies"].append({
            "name": strat,
            "trades": len(pnls), "wins": len(wins), "losses": len(losses),
            "wr_pct": wr, "pnl_usd": total,
            "avg_pnl": round(total / len(pnls), 2),
            "best": round(max(pnls), 2),
            "worst": round(min(pnls), 2),
            "pf": round(pf, 2) if pf < 999 else 999,
        })
        sym_rows = []
        for sym, sp in by_sym_strat[strat].items():
            sym_rows.append({
                "symbol": sym, "trades": len(sp),
                "pnl_usd": round(sum(sp), 2),
                "wr_pct": round(100.0 * sum(1 for x in sp if x > 0) / len(sp), 1),
            })
        sym_rows.sort(key=lambda x: x["pnl_usd"], reverse=True)
        out["per_symbol"][strat] = sym_rows

    out["strategies"].sort(key=lambda s: s["pnl_usd"], reverse=True)
    return jsonify(out)


@app.route("/api/close_all", methods=["POST"])
def close_all():
    if _executor:
        _executor.close_all("DashboardCloseAll")
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "msg": "No executor"})


@app.route("/api/close_symbol", methods=["POST"])
def close_symbol():
    data = request.get_json()
    sym = data.get("symbol", "")
    if _executor and sym:
        _executor.close_position(sym, "DashboardClose")
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"})


# ═══════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════

SYMBOL_LIST_JSON = json.dumps(list(SYMBOLS.keys()))
STARTING_BAL = STARTING_BALANCE

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>D.R.A.G.O.N — J.A.R.V.I.S. Trading Terminal</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script src="https://cdn.jsdelivr.net/npm/socket.io-client@4.7.4/dist/socket.io.min.js"></script>
<style>
/* ══════════════════════════════════════════════════════════════
   J.A.R.V.I.S. THEME — DRAGON TRADING TERMINAL
   ══════════════════════════════════════════════════════════════ */
:root {
  --bg: #020810;
  --bg1: rgba(0,15,30,0.8);
  --bg2: rgba(0,30,60,0.4);
  --bg3: rgba(0,50,90,0.3);
  --bdr: rgba(0,240,255,0.08);
  --bdr2: rgba(0,240,255,0.15);
  --bdr3: rgba(0,240,255,0.25);
  --t1: #ffffff;
  --t2: #b0e0ff;
  --t3: #70a0c0;
  --cyan: #00f0ff;
  --cyan-dim: rgba(0,240,255,0.15);
  --cyan-bg: rgba(0,240,255,0.06);
  --green: #00ff88;
  --green-bg: rgba(0,255,136,0.06);
  --red: #ff3355;
  --red-bg: rgba(255,51,85,0.06);
  --amber: #ffaa00;
  --amber-bg: rgba(255,170,0,0.06);
  --blue: #0088ff;
  --blue-bg: rgba(0,136,255,0.06);
  --purple: #aa55ff;
  --purple-bg: rgba(170,85,255,0.06);
  --gold: #ffcc00;
  --gold-bg: rgba(255,204,0,0.06);
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background: var(--bg); color: var(--t1);
  font-family: 'Rajdhani', sans-serif; font-size: 13px;
  overflow-x: hidden; min-height: 100vh;
}

/* ── GLOBAL HUD BACKGROUND ── */
body::before {
  content:''; position:fixed; inset:0; z-index:-2;
  background:
    radial-gradient(ellipse at 20% 50%, rgba(0,100,200,0.05) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 50%, rgba(0,240,255,0.03) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 0%, rgba(0,150,255,0.06) 0%, transparent 40%);
  pointer-events: none;
}
body::after {
  content:''; position:fixed; inset:0; z-index:-1;
  background-image:
    linear-gradient(rgba(0,240,255,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,240,255,0.02) 1px, transparent 1px);
  background-size: 60px 60px;
  animation: gridMove 20s linear infinite;
  pointer-events: none;
}
@keyframes gridMove { 0%{transform:translate(0,0)} 100%{transform:translate(60px,60px)} }

/* ── SCANLINE ── */
.scanline {
  position:fixed; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  opacity:0.08; z-index:9999; pointer-events:none;
  animation: scanDown 4s linear infinite;
}
@keyframes scanDown { 0%{top:-2px} 100%{top:100vh} }

/* ══════════════════════════════════════════════════════════════
   HEADER BAR — 60px
   ══════════════════════════════════════════════════════════════ */
.hdr {
  height: 60px; display: flex; align-items: center; justify-content: space-between;
  padding: 0 20px; position: sticky; top: 0; z-index: 100;
  background: linear-gradient(180deg, rgba(0,20,40,0.95), rgba(0,10,25,0.9));
  border-bottom: 1px solid var(--bdr3);
  box-shadow: 0 2px 30px rgba(0,150,255,0.08), inset 0 -1px 0 rgba(0,240,255,0.1);
}
.hdr::after {
  content:''; position:absolute; bottom:0; left:0; right:0; height:1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}

/* Arc Reactor */
.arc-reactor {
  width: 36px; height: 36px; position: relative;
  display: flex; align-items: center; justify-content: center;
  margin-right: 12px;
}
.arc-reactor .ring-outer {
  position: absolute; inset: 0; border-radius: 50%;
  border: 2px solid rgba(0,240,255,0.25);
  animation: arcSpin 6s linear infinite;
}
.arc-reactor .ring-inner {
  position: absolute; inset: 5px; border-radius: 50%;
  border: 1.5px dashed rgba(0,240,255,0.2);
  animation: arcSpin 4s linear infinite reverse;
}
.arc-reactor .core {
  width: 12px; height: 12px; border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan), 0 0 20px rgba(0,240,255,0.4), 0 0 40px rgba(0,240,255,0.15);
  animation: arcPulse 2s ease-in-out infinite;
}
@keyframes arcSpin { from{transform:rotate(0)} to{transform:rotate(360deg)} }
@keyframes arcPulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.7;transform:scale(0.85)} }

.hdr-left { display:flex; align-items:center; gap:6px; }
.logo-text {
  font-family:'Orbitron'; font-size:17px; font-weight:800; letter-spacing:3px;
  color: var(--cyan);
  text-shadow: 0 0 20px rgba(0,240,255,0.5), 0 0 40px rgba(0,240,255,0.2);
}
.logo-sub {
  font-family:'Share Tech Mono'; font-size:9px; color:var(--t3);
  letter-spacing:4px; margin-top:-2px;
}

/* Mode + Session badges */
.hdr-badges { display:flex; gap:8px; align-items:center; }
.pill {
  padding: 4px 14px; font-family:'JetBrains Mono'; font-size:10px; font-weight:600;
  letter-spacing: 1px; border: 1px solid; position:relative;
  clip-path: polygon(8px 0,100% 0,calc(100% - 8px) 100%,0 100%);
}
.pill-cyan { color:var(--cyan); background:var(--cyan-bg); border-color:rgba(0,240,255,0.3);
  box-shadow:0 0 12px rgba(0,240,255,0.15); }
.pill-session { font-weight:700; }

/* Header stats */
.hdr-stats { display:flex; gap:20px; align-items:center; }
.hdr-stat { text-align:center; position:relative; }
.hdr-stat::before {
  content:''; position:absolute; left:-10px; top:4px; bottom:4px; width:1px;
  background: linear-gradient(180deg,transparent,var(--bdr2),transparent);
}
.hdr-stat:first-child::before { display:none; }
.hdr-stat .lbl {
  font-family:'Orbitron'; font-size:7px; color:var(--t3);
  text-transform:uppercase; letter-spacing:1.5px; font-weight:500;
}
.hdr-stat .val {
  font-family:'JetBrains Mono'; font-size:15px; font-weight:700;
  text-shadow:0 0 12px rgba(0,240,255,0.15);
}
.hdr-stat .val.g { color:var(--green); text-shadow:0 0 12px rgba(0,255,136,0.3); }
.hdr-stat .val.r { color:var(--red); text-shadow:0 0 12px rgba(255,51,85,0.3); }
.hdr-stat .val.cy { color:var(--cyan); text-shadow:0 0 12px rgba(0,240,255,0.3); }
.sparkline-wrap { display:inline-block; vertical-align:middle; margin-left:4px; }
.sparkline-canvas { vertical-align:middle; }

/* Action buttons */
.hdr-actions { display:flex; gap:6px; align-items:center; }
.act-btn {
  padding:5px 14px; font-family:'Orbitron'; font-size:8px; font-weight:700;
  letter-spacing:1px; cursor:pointer; border:1px solid; transition:all 0.2s;
  clip-path: polygon(5px 0,100% 0,calc(100% - 5px) 100%,0 100%);
}
.act-red { color:var(--red); background:var(--red-bg); border-color:rgba(255,51,85,0.3); }
.act-red:hover { background:rgba(255,51,85,0.2); box-shadow:0 0 12px rgba(255,51,85,0.3); }
.act-amber { color:var(--amber); background:var(--amber-bg); border-color:rgba(255,170,0,0.3); }
.act-amber:hover { background:rgba(255,170,0,0.2); box-shadow:0 0 12px rgba(255,170,0,0.3); }
.act-select {
  background:#0a1a30; color:#ffffff; border:1px solid var(--cyan); padding:6px 12px;
  font-family:'Orbitron',sans-serif; font-size:9px; font-weight:600; cursor:pointer;
  letter-spacing:1px; appearance:none; -webkit-appearance:none;
  clip-path:polygon(5px 0,100% 0,calc(100% - 5px) 100%,0 100%);
}
.act-select:focus { outline:none; box-shadow:0 0 10px rgba(0,240,255,0.3); }
.act-select option { background:#0a1a30; color:#ffffff; font-family:'JetBrains Mono'; }

/* Session + Clock */
.hdr-right { display:flex; align-items:center; gap:14px; }
.clock {
  font-family:'Orbitron'; font-size:13px; color:var(--cyan); letter-spacing:2px;
  text-shadow:0 0 10px rgba(0,240,255,0.3);
}
.status-dot {
  width:8px; height:8px; border-radius:50%; background:var(--green);
  box-shadow:0 0 8px var(--green), 0 0 16px rgba(0,255,136,0.3);
  animation: dotPulse 2s infinite;
}
@keyframes dotPulse { 0%,100%{box-shadow:0 0 8px var(--green)} 50%{box-shadow:0 0 3px var(--green)} }

/* ══════════════════════════════════════════════════════════════
   MAIN GRID LAYOUT
   ══════════════════════════════════════════════════════════════ */
.main-grid {
  display: grid;
  grid-template-columns: 60fr 40fr;
  grid-template-rows: auto auto;
  gap: 1px;
  height: calc(100vh - 60px);
  padding: 0;
  background: rgba(0,240,255,0.03);
}
.main-grid > * { min-height: 0; overflow:hidden; }

/* ── CARD (HUD PANEL) ── */
.card {
  background: linear-gradient(135deg, rgba(0,15,30,0.85), rgba(0,10,25,0.92));
  border: 1px solid var(--bdr);
  position: relative;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent); opacity:0.3;
  z-index:2;
}
.card-h {
  padding: 8px 14px; display:flex; justify-content:space-between; align-items:center;
  border-bottom:1px solid var(--bdr);
  background:linear-gradient(90deg,rgba(0,40,80,0.2),transparent);
  flex-shrink:0;
}
.card-t {
  font-family:'Orbitron'; font-size:9px; font-weight:600; text-transform:uppercase;
  letter-spacing:2px; color:var(--cyan); text-shadow:0 0 10px rgba(0,240,255,0.2);
}
.card-badge {
  font-family:'JetBrains Mono'; font-size:9px; padding:2px 8px; color:var(--t3);
  border:1px solid var(--bdr); background:var(--bg2);
  clip-path: polygon(4px 0,100% 0,calc(100% - 4px) 100%,0 100%);
}
.card-b { flex:1; overflow-y:auto; overflow-x:hidden; position:relative; }
.card-b::-webkit-scrollbar { width:3px; }
.card-b::-webkit-scrollbar-thumb { background:rgba(0,240,255,0.2); border-radius:3px; }
.card-b::-webkit-scrollbar-track { background:transparent; }

/* ══════════════════════════════════════════════════════════════
   TICK CHART PANEL (top-left, 60%)
   ══════════════════════════════════════════════════════════════ */
.chart-panel { position:relative; }
.chart-controls {
  display:flex; align-items:center; gap:6px; padding:6px 14px;
  background:rgba(0,20,40,0.5); border-bottom:1px solid var(--bdr);
  flex-shrink:0;
}
.tf-btn {
  padding:3px 10px; font-family:'Orbitron'; font-size:8px; font-weight:600;
  letter-spacing:1px; cursor:pointer; border:1px solid var(--bdr);
  background:transparent; color:var(--t3); transition:all 0.2s;
}
.tf-btn.active {
  color:var(--cyan); border-color:rgba(0,240,255,0.3); background:var(--cyan-bg);
  box-shadow:0 0 8px rgba(0,240,255,0.15);
}
.tf-btn:hover:not(.active) { color:var(--t2); border-color:var(--bdr2); }
.sym-tab {
  padding:3px 12px; font-family:'JetBrains Mono'; font-size:9px; font-weight:600;
  cursor:pointer; border:1px solid var(--bdr); background:transparent; color:var(--t3);
  transition:all 0.2s; letter-spacing:0.5px;
  clip-path: polygon(4px 0,100% 0,calc(100% - 4px) 100%,0 100%);
}
.sym-tab.active {
  color:var(--cyan); border-color:rgba(0,240,255,0.3); background:var(--cyan-bg);
  box-shadow:0 0 8px rgba(0,240,255,0.15);
}
.sym-tab:hover:not(.active) { color:var(--t2); }
.tf-sep { width:1px; height:18px; background:var(--bdr2); margin:0 4px; }
#chart-container { flex:1; width:100%; position:relative; }

/* ══════════════════════════════════════════════════════════════
   MARKET SCANNER (top-right, 40%)
   ══════════════════════════════════════════════════════════════ */
.scanner { position:relative; }
.scanner .intel-scan {
  position:absolute; top:0; left:0; right:0; bottom:0; pointer-events:none; overflow:hidden; z-index:2;
}
.scanner .intel-scan::after {
  content:''; position:absolute; left:0; right:0; height:40px;
  background:linear-gradient(180deg,transparent,rgba(0,240,255,0.04),rgba(0,240,255,0.06),rgba(0,240,255,0.04),transparent);
  animation: scanline2 4s ease-in-out infinite;
}
@keyframes scanline2 { 0%{top:-10%} 50%{top:100%} 100%{top:-10%} }

.sym-card.selected { background:rgba(0,240,255,0.06) !important; border-left:2px solid var(--cyan) !important; }
.sym-card {
  padding:12px 14px; border-bottom:1px solid var(--bdr); position:relative;
  transition: background 0.3s;
}
.sym-card:hover { background:rgba(0,240,255,0.03); }
.sym-card:last-child { border-bottom:none; }

.sym-row1 { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.sym-name {
  font-family:'Orbitron'; font-size:12px; font-weight:700; letter-spacing:1.5px; color:var(--t1);
}
.sym-cat {
  font-family:'JetBrains Mono'; font-size:8px; font-weight:600; padding:1px 6px;
  letter-spacing:0.5px; margin-left:6px;
  clip-path: polygon(3px 0,100% 0,calc(100% - 3px) 100%,0 100%);
}
.cat-Gold { color:var(--gold); background:var(--gold-bg); border:1px solid rgba(255,204,0,0.2); }
.cat-Crypto { color:var(--purple); background:var(--purple-bg); border:1px solid rgba(170,85,255,0.2); }
.cat-Index { color:var(--blue); background:var(--blue-bg); border:1px solid rgba(0,136,255,0.2); }
.cat-Forex { color:var(--cyan); background:var(--cyan-bg); border:1px solid rgba(0,240,255,0.2); }

.sym-price {
  font-family:'JetBrains Mono'; font-size:16px; font-weight:700;
  text-shadow:0 0 12px rgba(0,240,255,0.15);
}
.sym-arrow { font-size:14px; margin-left:4px; }
.sym-arrow.up { color:var(--green); }
.sym-arrow.dn { color:var(--red); }

.sym-row2 { display:flex; gap:12px; align-items:center; margin-bottom:6px; font-size:11px; }
.sym-detail { font-family:'JetBrains Mono'; font-size:10px; color:var(--t3); }
.sym-detail span { color:var(--t2); }

/* Score bars */
.sym-scores { display:flex; gap:12px; margin-bottom:6px; }
.score-block { flex:1; }
.score-label {
  font-family:'Orbitron'; font-size:7px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1px; margin-bottom:2px;
}
.score-bar {
  height:4px; background:rgba(0,240,255,0.08); border-radius:0; overflow:hidden;
}
.score-fill {
  height:100%; transition:width 0.5s; border-radius:0;
  box-shadow:0 0 6px currentColor;
}

/* ML Confidence */
.ml-bar-wrap { margin-bottom:6px; }
.ml-label {
  font-family:'Orbitron'; font-size:7px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1px; margin-bottom:2px; display:flex; justify-content:space-between;
}
.ml-bar {
  height:3px; background:rgba(0,240,255,0.06); overflow:hidden;
}
.ml-fill {
  height:100%; transition:width 0.5s;
  box-shadow:0 0 8px currentColor;
}

/* Gate dots */
.gate-row { display:flex; gap:10px; align-items:center; margin-bottom:6px; }
.gate {
  display:flex; align-items:center; gap:3px;
  font-family:'Orbitron'; font-size:7px; color:var(--t3); letter-spacing:0.5px;
}
.gate-dot {
  width:7px; height:7px; border-radius:50%;
}
.gate-pass { background:var(--green); box-shadow:0 0 6px rgba(0,255,136,0.4); }
.gate-block { background:var(--red); box-shadow:0 0 6px rgba(255,51,85,0.4); }
.gate-na { background:rgba(0,240,255,0.15); }

/* Position status */
.sym-pos {
  font-family:'JetBrains Mono'; font-size:10px; font-weight:600;
  padding:2px 8px;
  clip-path: polygon(3px 0,100% 0,calc(100% - 3px) 100%,0 100%);
}
.pos-flat { color:var(--t3); background:rgba(0,240,255,0.04); border:1px solid var(--bdr); }
.pos-long { color:var(--green); background:var(--green-bg); border:1px solid rgba(0,255,136,0.2); }
.pos-short { color:var(--red); background:var(--red-bg); border:1px solid rgba(255,51,85,0.2); }

.sym-row-bottom { display:flex; justify-content:space-between; align-items:center; }
.mini-sparkline { height:20px; width:80px; }

/* ══════════════════════════════════════════════════════════════
   INTELLIGENCE PANEL (bottom-left, 50%)
   ══════════════════════════════════════════════════════════════ */
.intel-content { padding:10px 14px; }
.regime-badges { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
.regime-badge {
  font-family:'JetBrains Mono'; font-size:9px; font-weight:600; padding:3px 10px;
  clip-path: polygon(4px 0,100% 0,calc(100% - 4px) 100%,0 100%);
}
.regime-momentum { color:var(--blue); background:var(--blue-bg); border:1px solid rgba(0,136,255,0.2); }
.regime-mean_reversion { color:var(--purple); background:var(--purple-bg); border:1px solid rgba(170,85,255,0.2); }
.regime-breakout { color:var(--amber); background:var(--amber-bg); border:1px solid rgba(255,170,0,0.2); }
.regime-ranging { color:var(--cyan); background:var(--cyan-bg); border:1px solid rgba(0,240,255,0.2); }
.regime-volatile { color:var(--red); background:var(--red-bg); border:1px solid rgba(255,51,85,0.2); }
.regime-unknown { color:var(--t3); background:rgba(0,240,255,0.03); border:1px solid var(--bdr); }

/* Score breakdown bars */
.score-breakdown { margin-bottom:12px; }
.sb-row { display:flex; align-items:center; margin-bottom:4px; gap:6px; }
.sb-label {
  font-family:'JetBrains Mono'; font-size:9px; color:var(--t3);
  width:90px; text-align:right; flex-shrink:0;
}
.sb-bar {
  flex:1; height:6px; background:rgba(0,240,255,0.06); overflow:hidden;
}
.sb-fill {
  height:100%; transition:width 0.5s;
  box-shadow:0 0 4px currentColor;
}
.sb-val {
  font-family:'JetBrains Mono'; font-size:9px; color:var(--t2);
  width:35px; flex-shrink:0;
}
/* Risk locks (cooldown / blacklist) */
#risk-locks { margin-bottom:10px; max-height:150px; overflow-y:auto; }
.rl-row { display:flex; align-items:center; gap:6px; padding:3px 2px;
  border-bottom:1px solid rgba(0,240,255,0.05); font-family:'JetBrains Mono'; font-size:10px; }
.rl-sym { flex:1; color:var(--t2); letter-spacing:0.5px; }
.rl-tag { font-size:8px; padding:1px 5px; border-radius:3px; letter-spacing:0.5px; }
.rl-bl   { background:rgba(255,51,85,0.18); color:#ff5577; }
.rl-both { background:rgba(255,160,0,0.16); color:#ffaa33; }
.rl-dir  { background:rgba(0,240,255,0.12); color:#00d0ff; }
.rl-min { width:42px; text-align:right; color:var(--t3); }

/* Trade log */
.trade-table { width:100%; border-collapse:collapse; }
.trade-table th {
  padding:5px 10px; text-align:left; font-family:'Orbitron'; font-size:7px;
  font-weight:600; color:var(--t3); text-transform:uppercase; letter-spacing:1.5px;
  background:linear-gradient(90deg,rgba(0,30,60,0.4),transparent);
  border-bottom:1px solid var(--bdr2); position:sticky; top:0; z-index:1;
}
.trade-table td {
  padding:4px 10px; border-bottom:1px solid rgba(0,240,255,0.04); font-size:11px;
}
.trade-table tr:hover td { background:rgba(0,240,255,0.03); }
.tag {
  display:inline-block; padding:1px 7px; font-family:'JetBrains Mono';
  font-size:8px; font-weight:600; letter-spacing:0.5px;
  clip-path: polygon(3px 0,100% 0,calc(100% - 3px) 100%,0 100%);
}
.tag-long { color:var(--green); background:var(--green-bg); border:1px solid rgba(0,255,136,0.2); }
.tag-short { color:var(--red); background:var(--red-bg); border:1px solid rgba(255,51,85,0.2); }
.tag-flat { color:var(--t3); background:rgba(0,240,255,0.03); border:1px solid var(--bdr); }

.mono { font-family:'JetBrains Mono'; }
.g { color:var(--green); text-shadow:0 0 8px rgba(0,255,136,0.2); }
.r { color:var(--red); text-shadow:0 0 8px rgba(255,51,85,0.2); }
.cy { color:var(--cyan); }
.dim { color:var(--t3); }
.bright { font-weight:700; color:var(--t1); }

/* ══════════════════════════════════════════════════════════════
   PERFORMANCE PANEL (bottom-right, 50%)
   ══════════════════════════════════════════════════════════════ */
.perf-content { padding:10px 14px; }
#equity-chart-container { height:140px; width:100%; margin-bottom:10px; }

.perf-stats {
  display:grid; grid-template-columns:repeat(5,1fr); gap:6px; margin-bottom:10px;
}
.perf-stat {
  background:rgba(0,15,30,0.6); border:1px solid var(--bdr); padding:8px;
  position:relative; overflow:hidden;
  clip-path: polygon(0 0,calc(100% - 6px) 0,100% 6px,100% 100%,6px 100%,0 calc(100% - 6px));
}
.perf-stat::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
}
.perf-stat.ps-g::before { background:linear-gradient(90deg,var(--green),transparent); }
.perf-stat.ps-r::before { background:linear-gradient(90deg,var(--red),transparent); }
.perf-stat.ps-c::before { background:linear-gradient(90deg,var(--cyan),transparent); }
.perf-stat.ps-a::before { background:linear-gradient(90deg,var(--amber),transparent); }
.perf-stat.ps-b::before { background:linear-gradient(90deg,var(--blue),transparent); }
.perf-stat .ps-label {
  font-family:'Orbitron'; font-size:7px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1px; font-weight:500;
}
.perf-stat .ps-val {
  font-family:'JetBrains Mono'; font-size:14px; font-weight:700; margin-top:2px;
  text-shadow:0 0 10px rgba(0,240,255,0.1);
}

/* R-Multiple histogram */
.r-hist-wrap { margin-top:8px; }
.r-hist-title {
  font-family:'Orbitron'; font-size:8px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1.5px; margin-bottom:6px;
}
.r-hist { display:flex; align-items:flex-end; gap:2px; height:60px; }
.r-bar {
  flex:1; min-width:4px; position:relative; border-radius:1px 1px 0 0;
  transition: height 0.5s;
}
.r-bar-g { background:var(--green); box-shadow:0 -2px 8px rgba(0,255,136,0.2); }
.r-bar-r { background:var(--red); box-shadow:0 -2px 8px rgba(255,51,85,0.2); }
.r-labels { display:flex; gap:2px; margin-top:2px; }
.r-label { flex:1; text-align:center; font-family:'JetBrains Mono'; font-size:7px; color:var(--t3); }

/* ── HOLOGRAPHIC SEPARATOR ── */
.holo-sep {
  height:1px; margin:8px 0;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  opacity:0.15;
}

/* ── EMPTY STATE ── */
.empty {
  padding:30px; text-align:center; color:var(--t3);
  font-family:'Orbitron'; font-size:10px; letter-spacing:2px;
}

/* ── HUD CORNER DECORATIONS ── */
.hud-corner { position:absolute; width:12px; height:12px; z-index:3; }
.hud-corner-tl { top:0;left:0; border-top:2px solid var(--cyan); border-left:2px solid var(--cyan); }
.hud-corner-tr { top:0;right:0; border-top:2px solid var(--cyan); border-right:2px solid var(--cyan); }
.hud-corner-bl { bottom:0;left:0; border-bottom:2px solid var(--cyan); border-left:2px solid var(--cyan); }
.hud-corner-br { bottom:0;right:0; border-bottom:2px solid var(--cyan); border-right:2px solid var(--cyan); }

/* ── RESPONSIVE ── */
@media(max-width:1200px) {
  .main-grid { grid-template-columns:1fr; grid-template-rows:auto; }
  .hdr { flex-wrap:wrap; height:auto; padding:8px 12px; gap:8px; }
  .hdr-stats { flex-wrap:wrap; gap:10px; }
}
</style>
</head>
<body>
<div class="scanline"></div>

<!-- ══════════════ HEADER BAR ══════════════ -->
<header class="hdr">
  <div class="hdr-left">
    <div class="arc-reactor">
      <div class="ring-outer"></div>
      <div class="ring-inner"></div>
      <div class="core"></div>
    </div>
    <div>
      <div class="logo-text">D.R.A.G.O.N</div>
      <div class="logo-sub">INTELLIGENCE TRADING TERMINAL</div>
    </div>
  </div>

  <div class="hdr-badges">
    <div class="pill pill-cyan" id="h-mode">HYBRID</div>
    <div class="pill pill-session" id="h-session" style="color:#00ff88;background:rgba(0,255,136,0.06);border-color:rgba(0,255,136,0.3);">LONDON</div>
  </div>

  <div class="hdr-stats">
    <div class="hdr-stat">
      <div class="lbl">Balance</div>
      <div class="val cy" id="h-bal">---</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Equity</div>
      <div class="val cy" id="h-eq">---</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Float P&L</div>
      <div class="val" id="h-pnl">---</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Daily P&L</div>
      <div class="val" id="h-dpnl">---</div>
      <span class="sparkline-wrap"><canvas class="sparkline-canvas" id="dpnl-spark" width="40" height="14"></canvas></span>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Positions</div>
      <div class="val cy" id="h-pos">0</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Risk%</div>
      <div class="val" id="h-risk">0%</div>
    </div>
  </div>

  <div class="hdr-actions">
    <button class="act-btn act-red" id="btn-close-all" onclick="doCloseAll()">CLOSE ALL</button>
    <button class="act-btn act-amber" id="btn-close-losing" onclick="doCloseLosing()">CLOSE LOSING</button>
    <select class="act-select" id="close-sym-select">
      <option value="">Symbol...</option>
    </select>
    <button class="act-btn act-red" onclick="doCloseSym()">CLOSE</button>
  </div>

  <div class="hdr-right">
    <div class="status-dot" id="status-dot" title="Connection status"></div>
    <div class="clock" id="h-clock">00:00:00 IST</div>
    <div style="font-family:'JetBrains Mono';font-size:8px;color:var(--t3)" id="h-last-update">--</div>
  </div>
</header>

<!-- ══════════════ MAIN GRID ══════════════ -->
<div class="main-grid">

  <!-- ═══ EQUITY CURVE (top-left) ═══ -->
  <div class="card chart-panel">
    <div class="hud-corner hud-corner-tl"></div>
    <div class="hud-corner hud-corner-tr"></div>
    <div class="card-h">
      <span class="card-t">EQUITY CURVE</span>
      <span class="card-badge" id="eq-now">--</span>
    </div>
    <div id="chart-container"></div>
  </div>

  <!-- ═══ MARKET SCANNER (top-right) ═══ -->
  <div class="card scanner">
    <div class="hud-corner hud-corner-tl"></div>
    <div class="hud-corner hud-corner-tr"></div>
    <div class="intel-scan"></div>
    <div class="card-h">
      <span class="card-t">MARKET SCANNER</span>
      <span class="card-badge" id="scan-count">4 SYMBOLS</span>
    </div>
    <div class="card-b" id="scanner-body">
      <div class="empty">Awaiting tick data...</div>
    </div>
  </div>

  <!-- ═══ INTELLIGENCE (bottom-left) ═══ -->
  <div class="card">
    <div class="hud-corner hud-corner-bl"></div>
    <div class="card-h">
      <span class="card-t">DRAGON INTELLIGENCE</span>
      <span class="card-badge" id="intel-sym">---</span>
    </div>
    <div class="card-b" id="intel-body">
      <div class="intel-content">
        <div id="master-brain-status" style="margin-bottom:8px">
          <div class="empty">MasterBrain loading...</div>
        </div>
        <div class="holo-sep"></div>
        <div class="score-breakdown" id="score-breakdown">
          <div class="empty">Select a symbol for score breakdown</div>
        </div>
        <div class="holo-sep"></div>
        <div id="trade-log-wrap">
          <div class="empty">No recent trades</div>
        </div>
        <div class="holo-sep"></div>
        <!-- STRATEGY BATTLE (since fresh-start baseline) -->
        <div id="strategy-battle-wrap" style="margin-top:8px">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">
            <div style="font-family:Orbitron;font-size:8px;color:var(--t3);letter-spacing:1.5px">STRATEGY BATTLE — SINCE FRESH START</div>
            <div id="sb-baseline" style="font-size:9px;color:var(--t3);font-family:'JetBrains Mono'">loading…</div>
          </div>
          <div id="strategy-battle-body"><div class="empty">Loading strategy stats…</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ PERFORMANCE (bottom-right) ═══ -->
  <div class="card">
    <div class="hud-corner hud-corner-br"></div>
    <div class="card-h">
      <span class="card-t">PERFORMANCE</span>
      <span class="card-badge" id="perf-trades">0 TRADES</span>
    </div>
    <div class="card-b">
      <div class="perf-content">
        <div id="risk-locks">
          <div style="font-family:Orbitron;font-size:8px;color:var(--t3);letter-spacing:1.5px;margin-bottom:6px">RISK LOCKS — COOLDOWN / BLACKLIST</div>
          <div id="risk-locks-body"><div class="empty">No active locks</div></div>
        </div>
        <div class="holo-sep"></div>
        <div class="perf-stats" id="perf-stats">
          <div class="perf-stat ps-g"><div class="ps-label">Win Rate</div><div class="ps-val" id="ps-wr">--</div></div>
          <div class="perf-stat ps-c"><div class="ps-label">Profit Factor</div><div class="ps-val" id="ps-pf">--</div></div>
          <div class="perf-stat ps-b"><div class="ps-label">Sharpe</div><div class="ps-val" id="ps-sharpe">--</div></div>
          <div class="perf-stat ps-a"><div class="ps-label">Avg R</div><div class="ps-val" id="ps-avgr">--</div></div>
          <div class="perf-stat ps-r"><div class="ps-label">Max DD</div><div class="ps-val" id="ps-dd">--</div></div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
// ══════════════════════════════════════════════════════════════
// D.R.A.G.O.N. — CLIENT-SIDE ENGINE
// ══════════════════════════════════════════════════════════════

const SYMBOLS = """ + SYMBOL_LIST_JSON + r""";
const SYMBOL_META = """ + json.dumps({sym: {"category": cfg.category, "digits": cfg.digits} for sym, cfg in SYMBOLS.items()}) + r""";
const STARTING_BAL = """ + str(STARTING_BAL) + r""";

const $ = id => document.getElementById(id);
const f = (v, d=2) => v != null ? Number(v).toFixed(d) : '---';
const s = v => v >= 0 ? '+' + f(v) : f(v);

// ── STATE ──
let selectedSymbol = SYMBOLS[0];
let selectedTF = 5;
let lastTicks = {};
let prevPrices = {};
let dailyPnlHistory = [];
let chartData = {};
let equityHistory = [];

// ── SOCKET.IO (auto-reconnect, connection status) ──
const socket = io({
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: Infinity,
});

socket.on('connect', () => {
  console.log('Socket connected');
  const dot = document.getElementById('status-dot');
  if (dot) { dot.style.background = 'var(--green)'; dot.title = 'Connected'; }
});
socket.on('disconnect', () => {
  console.log('Socket disconnected — reconnecting...');
  const dot = document.getElementById('status-dot');
  if (dot) { dot.style.background = 'var(--red)'; dot.title = 'Disconnected'; }
});
socket.on('reconnect', (attempt) => {
  console.log('Reconnected after', attempt, 'attempts');
});

// ── LIGHTWEIGHT CHARTS ──
let mainChart = null;
let candleSeries = null;
let volumeSeries = null;
let ema20Series = null;
let ema50Series = null;
let ema200Series = null;
let equityChart = null;
let equitySeries = null;

// ── SYMBOL + TF TABS ──
function buildControls() {
  // sym-tabs / tf-btns removed with the per-symbol chart (2026-05-29); guard
  // in case the elements are absent.
  const symWrap = $('sym-tabs');
  if (symWrap) SYMBOLS.forEach(sym => {
    const btn = document.createElement('button');
    btn.className = 'sym-tab' + (sym === selectedSymbol ? ' active' : '');
    btn.textContent = sym;
    btn.onclick = () => selectSymbol(sym);
    symWrap.appendChild(btn);
  });

  const tfWrap = $('tf-btns');
  if (tfWrap) {
    const tfs = [{v:1,l:'M1'},{v:5,l:'M5'},{v:15,l:'M15'},{v:60,l:'H1'}];
    tfs.forEach(t => {
      const btn = document.createElement('button');
      btn.className = 'tf-btn' + (t.v === selectedTF ? ' active' : '');
      btn.textContent = t.l;
      btn.onclick = () => selectTF(t.v);
      tfWrap.appendChild(btn);
    });
  }

  // Populate close dropdown
  const sel = $('close-sym-select');
  SYMBOLS.forEach(sym => {
    const opt = document.createElement('option');
    opt.value = sym; opt.textContent = sym;
    sel.appendChild(opt);
  });
}

function selectSymbol(sym) {
  selectedSymbol = sym;
  socket.emit('select_symbol', {symbol: sym});
  // Update chart tabs
  document.querySelectorAll('.sym-tab').forEach(b => {
    b.classList.toggle('active', b.textContent === sym);
  });
  // Update scanner card highlight
  document.querySelectorAll('.sym-card').forEach(c => {
    const name = c.querySelector('.sym-name');
    if (name) c.classList.toggle('selected', name.textContent.includes(sym));
  });
  $('intel-sym').textContent = sym;
  refreshChart();
  // Force immediate intelligence panel update if we have cached data
  if (window._lastStatsData) updateIntelligence(window._lastStatsData);
}

function selectTF(tf) {
  selectedTF = tf;
  socket.emit('select_timeframe', {timeframe: tf});
  document.querySelectorAll('.tf-btn').forEach(b => {
    const map = {1:'M1',5:'M5',15:'M15',60:'H1'};
    b.classList.toggle('active', b.textContent === map[tf]);
  });
  refreshChart();
}

// ══════════════════════════════════════════════════════════════
// LIGHTWEIGHT CHARTS INIT
// ══════════════════════════════════════════════════════════════
function initCharts() {
  // 2026-05-29: per-symbol candle chart removed. The top-left panel now hosts
  // the account EQUITY CURVE (the only chart we keep). equityChart/equitySeries
  // are reused by the existing stats feed (equitySeries.setData).
  const container = $('chart-container');
  equityChart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight,
    layout: {
      background: { type: 'solid', color: '#020810' },
      textColor: 'rgba(0,200,255,0.5)',
      fontSize: 10,
      fontFamily: 'JetBrains Mono',
    },
    grid: {
      vertLines: { color: 'rgba(0,240,255,0.04)' },
      horzLines: { color: 'rgba(0,240,255,0.04)' },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: 'rgba(0,240,255,0.3)', width: 1, style: 2 },
      horzLine: { color: 'rgba(0,240,255,0.3)', width: 1, style: 2 },
    },
    rightPriceScale: { borderColor: 'rgba(0,240,255,0.1)', scaleMargins: { top: 0.12, bottom: 0.12 } },
    timeScale: { borderColor: 'rgba(0,240,255,0.1)', timeVisible: true, secondsVisible: false },
    watermark: {
      visible: true, text: 'EQUITY', fontSize: 44,
      color: 'rgba(0,240,255,0.04)', horzAlign: 'center', vertAlign: 'center',
    },
  });

  equitySeries = equityChart.addAreaSeries({
    lineColor: '#00f0ff',
    topColor: 'rgba(0,240,255,0.18)',
    bottomColor: 'rgba(0,240,255,0.02)',
    lineWidth: 2,
    priceLineVisible: false,
  });

  const resizeObserver = new ResizeObserver(() => {
    if (equityChart) equityChart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  resizeObserver.observe(container);
}

// ══════════════════════════════════════════════════════════════
// CHART DATA REFRESH
// ══════════════════════════════════════════════════════════════
function refreshChart() {
  return;  // 2026-05-29: per-symbol candle chart removed — no-op.
  const key = selectedSymbol + '_' + selectedTF;
  const data = chartData[key];
  if (!data) return;

  if (data.candles && data.candles.length > 0) {
    candleSeries.setData(data.candles);
  }
  if (data.volumes && data.volumes.length > 0) {
    // Color volumes
    const colored = data.volumes.map((v, i) => {
      const c = data.candles[i];
      const isUp = c && c.close >= c.open;
      return { ...v, color: isUp ? 'rgba(0,255,136,0.2)' : 'rgba(255,51,85,0.2)' };
    });
    volumeSeries.setData(colored);
  }

  // EMA overlays — compute from candle data
  if (data.candles && data.candles.length > 20) {
    const closes = data.candles.map(c => c.close);
    const times = data.candles.map(c => c.time);
    ema20Series.setData(calcEMA(closes, times, 20));
    ema50Series.setData(calcEMA(closes, times, 50));
    ema200Series.setData(calcEMA(closes, times, Math.min(200, closes.length)));
  }

  mainChart.timeScale().fitContent();
}

function calcEMA(closes, times, period) {
  if (closes.length < period) return [];
  const alpha = 2 / (period + 1);
  const result = [];
  let ema = closes.slice(0, period).reduce((a,b) => a+b, 0) / period;
  for (let i = period - 1; i < closes.length; i++) {
    if (i === period - 1) {
      ema = closes.slice(0, period).reduce((a,b) => a+b, 0) / period;
    } else {
      ema = alpha * closes[i] + (1 - alpha) * ema;
    }
    result.push({ time: times[i], value: parseFloat(ema.toFixed(5)) });
  }
  return result;
}

// ══════════════════════════════════════════════════════════════
// SPARKLINE DRAWING
// ══════════════════════════════════════════════════════════════
function drawSparkline(canvas, data, color) {
  if (!data || data.length < 2) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ══════════════════════════════════════════════════════════════
// WEBSOCKET HANDLERS
// ══════════════════════════════════════════════════════════════

socket.on('tick_update', function(ticks) {
  try {
  // Store previous prices for direction
  for (const sym of SYMBOLS) {
    if (lastTicks[sym]) prevPrices[sym] = lastTicks[sym].bid;
  }
  // Update equity in real-time from tick data
  if (ticks._account) {
    const a = ticks._account;
    if ($('h-bal')) $('h-bal').textContent = '$' + f(a.balance);
    if ($('h-eq')) $('h-eq').textContent = '$' + f(a.equity);
    const pnl = a.profit || 0;
    if ($('h-pnl')) {
      $('h-pnl').textContent = (pnl>=0?'+':'') + '$' + f(pnl);
      $('h-pnl').style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
    }
    delete ticks._account;
  }
  // Update positions directly from MT5 on every tick
  if (ticks._pos_map) {
    window._lastPosMap = ticks._pos_map;
    delete ticks._pos_map;
  }
  if (ticks._positions) {
    window._lastPositions = ticks._positions;
    // Update positions count in header — count unique symbols, not sub-positions
    if ($('h-pos')) {
      const uniqueSyms = new Set(ticks._positions.map(p => p.symbol));
      $('h-pos').textContent = uniqueSyms.size;
    }
    delete ticks._positions;
  }
  lastTicks = ticks;
  updateScanner();
  } catch(e) { console.error('tick_update error:', e); }
});

socket.on('chart_update', function(data) {
  // 2026-05-29: per-symbol candle chart removed — ignore candle pushes.
});

socket.on('stats_update', function(data) {
  try {
  window._lastStatsData = data;  // cache for selectSymbol
  updateHeader(data);
  updateIntelligence(data);
  updatePerformance(data);
  updateScanner();  // refresh scanner with latest scores/gates
  } catch(e) { console.error('stats_update error:', e); }
});

socket.on('action_result', function(data) {
  console.log('Action result:', data);
});

// ══════════════════════════════════════════════════════════════
// HEADER UPDATE
// ══════════════════════════════════════════════════════════════
function updateHeader(d) {
  $('h-bal').textContent = '$' + f(d.balance);
  $('h-eq').textContent = '$' + f(d.equity);
  // Last update timestamp
  const lu = $('h-last-update');
  if (lu) lu.textContent = 'upd ' + new Date().toLocaleTimeString('en-IN', {timeZone:'Asia/Kolkata',hour12:false});

  const pnl = d.profit || 0;
  const pnlEl = $('h-pnl');
  pnlEl.textContent = '$' + s(pnl);
  pnlEl.className = 'val ' + (pnl >= 0 ? 'g' : 'r');

  const dpnl = d.daily_pnl || 0;
  const dpnlEl = $('h-dpnl');
  dpnlEl.textContent = '$' + s(dpnl);
  dpnlEl.className = 'val ' + (dpnl >= 0 ? 'g' : 'r');

  // Daily P&L sparkline
  dailyPnlHistory.push(dpnl);
  if (dailyPnlHistory.length > 40) dailyPnlHistory = dailyPnlHistory.slice(-40);
  drawSparkline($('dpnl-spark'), dailyPnlHistory, dpnl >= 0 ? '#00ff88' : '#ff3355');

  // num_positions counts sub-positions; show unique symbols instead
  const posArr = d.positions || [];
  const uniqSyms = new Set(posArr.map(p => p.symbol));
  $('h-pos').textContent = uniqSyms.size || 0;
  const risk = d.risk_pct || d.dd_pct || 0;
  const riskEl = $('h-risk');
  riskEl.textContent = f(risk,1) + '%';
  riskEl.className = 'val ' + (risk > 5 ? 'r' : risk > 2 ? 'cy' : 'g');

  // Mode
  $('h-mode').textContent = (d.mode || 'HYBRID').toUpperCase();

  // Session
  if (d.session) {
    const sEl = $('h-session');
    sEl.textContent = d.session;
    sEl.style.color = d.session_color;
    sEl.style.borderColor = d.session_color;
    sEl.style.background = d.session_color + '10';
  }

  // Status dot
  $('status-dot').style.background = d.running ? 'var(--green)' : 'var(--red)';
  $('status-dot').style.boxShadow = d.running
    ? '0 0 8px var(--green), 0 0 16px rgba(0,255,136,0.3)'
    : '0 0 8px var(--red), 0 0 16px rgba(255,51,85,0.3)';
}

// ══════════════════════════════════════════════════════════════
// MARKET SCANNER UPDATE
// ══════════════════════════════════════════════════════════════
function updateScanner() {
  let html = '';
  SYMBOLS.forEach(sym => {
    const tick = lastTicks[sym];
    const meta = SYMBOL_META[sym];
    const prev = prevPrices[sym];
    const cat = meta ? meta.category : 'Forex';
    const digits = meta ? meta.digits : 2;

    // Always render full card — even without ticks, show scores/gates from stats

    const bid = tick ? tick.bid : null;
    const ask = tick ? tick.ask : null;
    const spread = tick ? tick.spread : null;
    const isUp = prev && bid ? bid > prev : true;
    const arrow = isUp ? '&#9650;' : '&#9660;';
    const arrowCls = isUp ? 'up' : 'dn';

    // V5 Scores (0-100 signal quality from brain)
    const sc = window._lastScores && window._lastScores[sym] ? window._lastScores[sym] : {};
    const signalQuality = sc.signal_quality || 0;  // 0-100 from brain
    const h1Score = signalQuality;  // already 0-100
    const rawLong = sc.long_score || 0;
    const rawShort = sc.short_score || 0;
    const minQuality = sc.min_quality || 55;
    const regime = sc.regime || 'unknown';

    // ML: meta_prob is LIVE signal confidence (0-1), AUC is static model quality
    const metaProb = sc.meta_prob != null ? sc.meta_prob : null;
    const mlAUC = window._lastML && window._lastML[sym] ? (window._lastML[sym].auc || 0) : 0;
    const mlEnabled = window._lastML && window._lastML[sym] ? window._lastML[sym].enabled : false;
    // Bar only shows live meta_prob — AUC is just a label, not a bar value
    const mlConf = metaProb != null ? metaProb : 0;

    // TF confluence
    const h1Dir = sc.direction || 'FLAT';
    const m15Dir = sc.m15_dir || 'flat';

    // Actual gate status from brain
    const gate = sc.gate || '';
    const masterReason = sc.master_reason || '';
    const riskPct = sc.risk_pct || 0;

    // V5 Gate pipeline: ordered chain, each gate is binary pass/fail
    // Order: SCORE → SESSION → SL_CD → DIR → TOXIC → MTF → META → MASTER → ENTERED
    const gateOrder = ['BELOW_MIN_SCORE','SESSION','SL_COOLDOWN','DIR_BIAS',
                       'TOXIC_HOUR','M15_DISAGREE','META_REJECT','MASTER_REJECT'];
    const passGates = ['ENTERED','HOLD_SWING','REVERSAL','PULLBACK_WAIT','PULLBACK_ENTERED','EXEC_FAILED'];
    function gateStatus(gateField, gateReject) {
      if (!gateField) return 'na';
      if (passGates.includes(gateField)) return 'pass';
      if (gateField === gateReject) return 'block';
      const myIdx = gateOrder.indexOf(gateReject);
      const curIdx = gateOrder.indexOf(gateField);
      if (curIdx >= 0 && myIdx >= 0) {
        return curIdx < myIdx ? 'na' : 'pass';
      }
      return 'na';
    }
    const gateItems = [
      {name:'SCORE', val:gateStatus(gate,'BELOW_MIN_SCORE')},
      {name:'DIR', val:gateStatus(gate,'DIR_BIAS')},
      {name:'TOXIC', val:gateStatus(gate,'TOXIC_HOUR')},
      {name:'MTF', val:gateStatus(gate,'M15_DISAGREE')},
      {name:'META', val:gateStatus(gate,'META_REJECT')},
      {name:'MASTER', val:gateStatus(gate,'MASTER_REJECT')},
    ];

    // Position
    const pos = window._lastPosMap && window._lastPosMap[sym] ? window._lastPosMap[sym] : null;
    let posTag;
    if (!pos) {
      posTag = '<span class="sym-pos pos-flat">FLAT</span>';
    } else {
      const side = pos.side === 'BUY' ? 'LONG' : 'SHORT';
      const pnl = pos.pnl || 0;
      const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
      const pnlStr = (pnl >= 0 ? '+$' : '-$') + f(Math.abs(pnl));
      posTag = `<span class="sym-pos" style="color:${pnlColor}">${side} ${pnlStr}</span>`;
    }

    // Sparkline
    const sparkData = tick ? (tick.sparkline || []) : [];

    // Score colors
    const h1Color = h1Score > 60 ? 'var(--green)' : h1Score > 30 ? 'var(--amber)' : 'var(--red)';
    const mlColor = mlConf > 0.6 ? 'var(--green)' : mlConf > 0.3 ? 'var(--amber)' : 'var(--red)';

    html += `<div class="sym-card${sym===selectedSymbol?' selected':''}" onclick="selectSymbol('${sym}')">
      <div class="sym-row1">
        <span>
          <span class="sym-name">${sym}</span>
          <span class="sym-cat cat-${cat}">${cat}</span>
        </span>
        <span class="sym-price">${bid ? f(bid, digits) : '---'} <span class="sym-arrow ${arrowCls}">${bid ? arrow : ''}</span></span>
      </div>
      <div class="sym-row2">
        <span class="sym-detail">Bid <span>${bid ? f(bid,digits) : '---'}</span></span>
        <span class="sym-detail">Ask <span>${ask ? f(ask,digits) : '---'}</span></span>
        <span class="sym-detail">Spread <span>${spread != null ? f(spread,1) : '---'}</span></span>
      </div>
      <div class="sym-scores">
        <div class="score-block">
          <div class="score-label">Signal: ${f(signalQuality,0)}% (L:${f(rawLong,1)} S:${f(rawShort,1)}) min:${f(minQuality,0)}%</div>
          <div class="score-bar"><div class="score-fill" style="width:${Math.min(100,h1Score)}%;background:${h1Color}"></div></div>
        </div>
        <div class="score-block">
          <div class="score-label" style="display:flex;justify-content:space-between">
            <span>H1: <span style="color:${h1Dir==='LONG'?'var(--green)':h1Dir==='SHORT'?'var(--red)':'var(--t3)'}">${h1Dir}</span></span>
            <span>M15: <span style="color:${m15Dir.toUpperCase()==='LONG'?'var(--green)':m15Dir.toUpperCase()==='SHORT'?'var(--red)':'var(--t3)'}">${m15Dir.toUpperCase()}</span></span>
            <span style="color:${h1Dir!=='FLAT'&&m15Dir.toUpperCase()===h1Dir?'var(--green)':'var(--t3)'}; font-size:9px">${h1Dir!=='FLAT'&&m15Dir.toUpperCase()===h1Dir?'ALIGNED':'—'}</span>
          </div>
        </div>
      </div>
      <div class="sym-row2" style="margin-bottom:4px">
        <span class="sym-detail">Regime <span style="color:var(--amber)">${regime.toUpperCase()}</span></span>
        <span class="sym-detail">Risk <span style="color:${riskPct>0?'var(--green)':'var(--t3)'}">${riskPct>0?f(riskPct,3)+'%':'—'}</span></span>
        <span class="sym-detail">Gate <span style="color:${gate==='ENTERED'||gate==='PULLBACK_WAIT'?'var(--green)':gate.includes('REJECT')||gate.includes('DISAGREE')?'var(--red)':'var(--amber)'}">${gate ? gate.replace('BELOW_MIN_SCORE','LOW_SCORE').replace('BELOW_MIN','LOW_SCORE') : '—'}</span></span>
        ${metaProb!=null?`<span class="sym-detail">ML <span style="color:${metaProb>0.6?'var(--green)':metaProb>0.4?'var(--amber)':'var(--red)'}">${f(metaProb*100,0)}%</span></span>`:''}
      </div>
      <div class="gate-row">
        ${gateItems.map(g => `<div class="gate"><div class="gate-dot ${g.val === 'pass' ? 'gate-pass' : g.val === 'block' ? 'gate-block' : 'gate-na'}"></div>${g.name}</div>`).join('')}
      </div>
      <div class="sym-row-bottom">
        ${posTag}
        <canvas class="mini-sparkline" id="spark-${sym.replace('.','_')}" width="80" height="20"></canvas>
      </div>
    </div>`;
  });

  $('scanner-body').innerHTML = html;

  // Draw mini sparklines
  SYMBOLS.forEach(sym => {
    const tick = lastTicks[sym];
    if (tick && tick.sparkline) {
      const canvas = document.getElementById('spark-' + sym.replace('.','_'));
      if (canvas) drawSparkline(canvas, tick.sparkline, '#00f0ff');
    }
  });
}

// ══════════════════════════════════════════════════════════════
// INTELLIGENCE UPDATE
// ══════════════════════════════════════════════════════════════
function updateIntelligence(d) {
  // Store for scanner use
  window._lastScores = d.scores || {};
  window._lastML = d.ml_confidence || {};
  window._lastPosMap = d.pos_map || {};

  const confidence = d.ml_confidence || {};

  // MasterBrain status
  const mb = d.master_brain || {};
  const mbWrap = $('master-brain-status');
  if (mbWrap) {
    const eqHealth = mb.equity_health || 'unknown';
    const eqColor = eqHealth === 'healthy' ? 'var(--green)' : eqHealth === 'flat' ? 'var(--amber)' : 'var(--red)';
    const blacklisted = mb.blacklisted_symbols || {};
    const blSyms = Object.entries(blacklisted).map(([s,h]) => s + ' (' + f(h,1) + 'h)').join(', ') || 'None';
    const blColor = Object.keys(blacklisted).length > 0 ? 'var(--red)' : 'var(--green)';

    mbWrap.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px">
        <div class="perf-stat ps-c"><div class="ps-label">Eq Health</div><div class="ps-val" style="color:${eqColor};font-size:11px">${eqHealth.toUpperCase()}</div></div>
        <div class="perf-stat ps-b"><div class="ps-label">Eq Slope</div><div class="ps-val" style="font-size:11px;color:${(mb.equity_slope||0)>=0?'var(--green)':'var(--red)'}">${f(mb.equity_slope||0,4)}</div></div>
        <div class="perf-stat ps-a"><div class="ps-label">Daily Trades</div><div class="ps-val" style="font-size:11px">${mb.daily_trades||0}</div></div>
        <div class="perf-stat ps-g"><div class="ps-label">Win Rate</div><div class="ps-val" style="font-size:11px;color:${(mb.win_rate||0)>=50?'var(--green)':'var(--red)'}">${f(mb.win_rate||0,1)}%</div></div>
      </div>
      <div style="display:flex;gap:12px;font-size:10px;font-family:'JetBrains Mono'">
        <span style="color:var(--t3)">Daily P&L: <span style="color:${(mb.daily_pnl||0)>=0?'var(--green)':'var(--red)'}">$${f(mb.daily_pnl||0,2)}</span></span>
        <span style="color:var(--t3)">Last 10: <span style="color:${(mb.recent_10_pnl||0)>=0?'var(--green)':'var(--red)'}">$${f(mb.recent_10_pnl||0,2)}</span></span>
        <span style="color:var(--t3)">Total: <span style="color:var(--cyan)">${mb.total_trades||0}</span></span>
        ${mb.losing_day_yesterday ? '<span style="color:var(--red)">PREV DAY LOSS (risk halved)</span>' : ''}
        ${mb.session_paused ? '<span style="color:var(--red);font-weight:700">CIRCUIT BREAKER ACTIVE (' + (mb.session_losses||0) + ' losses)</span>' : '<span style="color:var(--t3)">Session losses: ' + (mb.session_losses||0) + '/2</span>'}
      </div>
      <div style="margin-top:4px;font-size:10px;font-family:'JetBrains Mono';color:var(--t3)">
        Blacklisted: <span style="color:${blColor}">${blSyms}</span>
      </div>
    `;
  }

  // Score breakdown for selected symbol
  const sbWrap = $('score-breakdown');
  const symScores = d.scores && d.scores[selectedSymbol] ? d.scores[selectedSymbol] : {};
  const symML = confidence[selectedSymbol] || {};

  const symGate = symScores.gate || '';
  const isSessionClosed = symGate === 'SESSION';
  let sbhtml = '<div style="font-family:Orbitron;font-size:8px;color:var(--t3);letter-spacing:1.5px;margin-bottom:6px">' + selectedSymbol + ' SIGNAL BREAKDOWN'
    + (isSessionClosed ? ' <span style="color:var(--red);margin-left:8px">SESSION CLOSED</span>' : '')
    + '</div>';

  // V5 Core metrics — all on 0-100 scale
  const longQ = Math.min(100, (symScores.long_score||0) / 12.0 * 100);
  const shortQ = Math.min(100, (symScores.short_score||0) / 12.0 * 100);
  const metrics = [
    {label:'Signal Quality', val:symScores.signal_quality||0, max:100, color:'var(--cyan)'},
    {label:'Min Threshold', val:symScores.min_quality||50, max:100, color:'var(--amber)'},
    {label:'Long', val:longQ, max:100, color:'var(--green)'},
    {label:'Short', val:shortQ, max:100, color:'var(--red)'},
  ];
  metrics.forEach(m => {
    const pct = Math.min(100, (m.val / m.max) * 100);
    sbhtml += `<div class="sb-row">
      <div class="sb-label">${m.label}</div>
      <div class="sb-bar"><div class="sb-fill" style="width:${pct}%;background:${m.color};color:${m.color}"></div></div>
      <div class="sb-val">${f(m.val,0)}%</div>
    </div>`;
  });

  // ML meta-prob if available
  const metaP = symScores.meta_prob;
  if (metaP != null) {
    const pct = Math.min(100, metaP * 100);
    const color = metaP > 0.6 ? 'var(--green)' : metaP > 0.4 ? 'var(--amber)' : 'var(--red)';
    sbhtml += `<div class="sb-row">
      <div class="sb-label">ML Meta Prob</div>
      <div class="sb-bar"><div class="sb-fill" style="width:${pct}%;background:${color};color:${color}"></div></div>
      <div class="sb-val">${f(metaP,3)}</div>
    </div>`;
  }

  // Risk pct from MasterBrain
  if (symScores.risk_pct) {
    const pct = Math.min(100, symScores.risk_pct / 0.5 * 100);
    sbhtml += `<div class="sb-row">
      <div class="sb-label">Risk %</div>
      <div class="sb-bar"><div class="sb-fill" style="width:${pct}%;background:var(--blue);color:var(--blue)"></div></div>
      <div class="sb-val">${f(symScores.risk_pct,3)}%</div>
    </div>`;
  }

  // Gate status + direction + regime
  sbhtml += `<div style="margin-top:8px;display:flex;gap:12px;font-size:10px;font-family:'JetBrains Mono'">
    <span style="color:var(--t3)">Dir: <span style="color:${symScores.direction==='LONG'?'var(--green)':symScores.direction==='SHORT'?'var(--red)':'var(--t3)'}">${symScores.direction||'FLAT'}</span></span>
    <span style="color:var(--t3)">M15: <span style="color:var(--cyan)">${(symScores.m15_dir||'flat').toUpperCase()}</span></span>
    <span style="color:var(--t3)">Regime: <span style="color:var(--amber)">${(symScores.regime||'unknown').toUpperCase()}</span></span>
    <span style="color:var(--t3)">Gate: <span style="color:${(symScores.gate||'')=='ENTERED'?'var(--green)':'var(--amber)'}">${symScores.gate||'—'}</span></span>
  </div>`;

  // Master reject reason
  if (symScores.master_reason) {
    sbhtml += `<div style="margin-top:4px;font-size:10px;font-family:'JetBrains Mono';color:var(--red)">Master Reject: ${symScores.master_reason}</div>`;
  }

  sbWrap.innerHTML = sbhtml;

  // Trade log
  const tlWrap = $('trade-log-wrap');
  const trades = d.trade_log || [];
  $('perf-trades').textContent = trades.length + ' TRADES';
  if (trades.length > 0) {
    let thtml = '<table class="trade-table"><tr><th>Time</th><th>Symbol</th><th>Dir</th><th>P&L</th></tr>';
    trades.slice().reverse().forEach(t => {
      const dir = (t.direction || t.type || '').toUpperCase();
      const tagCls = dir === 'LONG' || dir === 'BUY' ? 'tag-long' : dir === 'SHORT' || dir === 'SELL' ? 'tag-short' : 'tag-flat';
      const pnl = t.pnl || t.profit || 0;
      thtml += `<tr>
        <td class="mono dim">${t.timestamp || t.time || ''}</td>
        <td class="bright">${t.symbol || ''}</td>
        <td><span class="tag ${tagCls}">${dir}</span></td>
        <td class="mono ${pnl >= 0 ? 'g' : 'r'}">$${s(pnl)}</td>
      </tr>`;
    });
    tlWrap.innerHTML = thtml + '</table>';
  } else {
    tlWrap.innerHTML = '<div class="empty">No recent trades</div>';
  }
}

// ══════════════════════════════════════════════════════════════
// PERFORMANCE UPDATE
// ══════════════════════════════════════════════════════════════
function updatePerformance(d) {
  // Equity curve
  const eqHist = d.equity_history || [];
  if (eqHist.length > 0) {
    const now = Math.floor(Date.now() / 1000);
    const eqData = eqHist.map((v, i) => ({
      time: now - (eqHist.length - i) * 60,
      value: typeof v === 'number' ? v : (v.equity || v.value || 0),
    }));
    equitySeries.setData(eqData);
    equityChart.timeScale().fitContent();
  } else {
    // Use balance as single point
    const eq = d.equity || STARTING_BAL;
    equityHistory.push(eq);
    if (equityHistory.length > 300) equityHistory = equityHistory.slice(-300);
    const now = Math.floor(Date.now() / 1000);
    const eqData = equityHistory.map((v, i) => ({
      time: now - (equityHistory.length - i) * 5,
      value: v,
    }));
    equitySeries.setData(eqData);
    equityChart.timeScale().fitContent();
  }

  // Rolling stats from trade log
  const trades = d.trade_log || [];
  if (trades.length > 0) {
    const pnls = trades.map(t => t.pnl || t.profit || 0);
    const wins = pnls.filter(p => p > 0);
    const losses = pnls.filter(p => p < 0);

    const wr = (wins.length / pnls.length * 100);
    const grossProfit = wins.reduce((a,b) => a+b, 0);
    const grossLoss = Math.abs(losses.reduce((a,b) => a+b, 0));
    const pf = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 99.99 : 0;
    const avgR = pnls.reduce((a,b) => a+b, 0) / pnls.length;

    // Sharpe approximation
    const mean = pnls.reduce((a,b) => a+b, 0) / pnls.length;
    const variance = pnls.reduce((a,b) => a + (b - mean) ** 2, 0) / pnls.length;
    const stddev = Math.sqrt(variance) || 1;
    const sharpe = mean / stddev * Math.sqrt(252);

    // Max drawdown from equity
    let peak = STARTING_BAL, maxDD = 0;
    let runningEq = STARTING_BAL;
    pnls.forEach(p => {
      runningEq += p;
      if (runningEq > peak) peak = runningEq;
      const dd = (peak - runningEq) / peak * 100;
      if (dd > maxDD) maxDD = dd;
    });

    $('ps-wr').textContent = f(wr, 1) + '%';
    $('ps-wr').className = 'ps-val ' + (wr >= 50 ? 'g' : 'r');
    $('ps-pf').textContent = f(pf, 2);
    $('ps-pf').className = 'ps-val ' + (pf >= 1 ? 'g' : 'r');
    $('ps-sharpe').textContent = f(sharpe, 2);
    $('ps-sharpe').className = 'ps-val ' + (sharpe >= 0 ? 'g' : 'r');
    $('ps-avgr').textContent = '$' + f(avgR);
    $('ps-avgr').className = 'ps-val ' + (avgR >= 0 ? 'g' : 'r');
    $('ps-dd').textContent = f(maxDD, 1) + '%';
    $('ps-dd').className = 'ps-val ' + (maxDD > 5 ? 'r' : 'cy');

  }
}
}

// ══════════════════════════════════════════════════════════════
// ACTION HANDLERS
// ══════════════════════════════════════════════════════════════
function doCloseAll() {
  if (confirm('CLOSE ALL POSITIONS?')) {
    socket.emit('close_all');
  }
}
function doCloseLosing() {
  if (confirm('Close all LOSING positions?')) {
    socket.emit('close_losing');
  }
}
function doCloseSym() {
  const sym = $('close-sym-select').value;
  if (sym && confirm('Close ' + sym + '?')) {
    socket.emit('close_symbol', {symbol: sym});
  }
}

// ══════════════════════════════════════════════════════════════
// CLOCK
// ══════════════════════════════════════════════════════════════
function updateClock() {
  $('h-clock').textContent = new Date().toLocaleTimeString('en-IN', {
    timeZone: 'Asia/Kolkata', hour12: false
  }) + ' IST';
}

// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════
// ── RISK LOCKS (cooldowns + blacklist) ──
function renderRiskLocks(d) {
  const body = $('risk-locks-body');
  if (!body) return;
  const cds = (d && d.cooldowns) || [];
  const bls = (d && d.blacklist) || [];
  if (!cds.length && !bls.length) {
    body.innerHTML = '<div class="empty">No active locks</div>';
    return;
  }
  let html = '';
  bls.forEach(b => {
    html += `<div class="rl-row"><span class="rl-sym">${b.symbol}</span>`
          + `<span class="rl-tag rl-bl">BLACKLIST ${b.losses}L</span>`
          + `<span class="rl-min">${b.mins_left}m</span></div>`;
  });
  cds.forEach(c => {
    const dirClass = c.blocked === 'BOTH' ? 'rl-both' : 'rl-dir';
    html += `<div class="rl-row"><span class="rl-sym">${c.symbol}</span>`
          + `<span class="rl-tag ${dirClass}">${c.blocked}</span>`
          + `<span class="rl-min">${c.mins_left}m</span></div>`;
  });
  body.innerHTML = html;
}
function pollRiskLocks() {
  fetch('/api/risk_locks').then(r => r.json()).then(renderRiskLocks).catch(() => {});
}

// ═══ STRATEGY BATTLE (per-strategy PnL since fresh-start baseline) ═══
function renderStrategyBattle(d) {
  const body = document.getElementById('strategy-battle-body');
  const baseEl = document.getElementById('sb-baseline');
  if (!body || !baseEl) return;
  if (d.error) {
    body.innerHTML = `<div class="empty">${d.error}</div>`;
    baseEl.textContent = '';
    return;
  }
  const elapsed = (d.elapsed_days || 0).toFixed(1);
  const acctPnl = (d.account_pnl || 0);
  const acctColor = acctPnl >= 0 ? 'var(--green)' : 'var(--red)';
  baseEl.innerHTML = `${elapsed}d &nbsp;|&nbsp; eq $${(d.current_equity||0).toFixed(2)} &nbsp;|&nbsp; <span style="color:${acctColor}">${acctPnl>=0?'+':''}${acctPnl.toFixed(2)}</span>`;
  const strats = d.strategies || [];
  if (strats.length === 0) {
    body.innerHTML = '<div class="empty">No closed trades since fresh start</div>';
    return;
  }
  let html = '<table class="trade-table" style="width:100%;font-size:10px"><tr>'
    + '<th style="text-align:left">Strategy</th><th>Trd</th><th>WR</th>'
    + '<th style="text-align:right">PnL $</th><th>PF</th>'
    + '<th style="text-align:right">Best</th><th style="text-align:right">Worst</th></tr>';
  strats.forEach((s, i) => {
    const pnlColor = s.pnl_usd >= 0 ? 'g' : 'r';
    const rank = i === 0 && s.pnl_usd > 0 ? '🥇 ' : '';
    const pf = s.pf >= 999 ? '∞' : s.pf.toFixed(2);
    html += `<tr style="cursor:pointer" onclick="toggleStratSyms('${s.name}')">`
      + `<td class="bright">${rank}<span style="text-transform:uppercase">${s.name}</span></td>`
      + `<td class="mono dim">${s.trades}</td>`
      + `<td class="mono">${s.wr_pct.toFixed(1)}%</td>`
      + `<td class="mono ${pnlColor}" style="text-align:right">${s.pnl_usd>=0?'+':''}$${s.pnl_usd.toFixed(2)}</td>`
      + `<td class="mono">${pf}</td>`
      + `<td class="mono g" style="text-align:right">+$${s.best.toFixed(2)}</td>`
      + `<td class="mono r" style="text-align:right">$${s.worst.toFixed(2)}</td></tr>`;
    // hidden per-sym row
    const syms = (d.per_symbol || {})[s.name] || [];
    if (syms.length > 0) {
      html += `<tr id="sb-syms-${s.name}" style="display:none"><td colspan="7" style="padding:4px 0 8px 16px">`;
      syms.forEach(sr => {
        const psColor = sr.pnl_usd >= 0 ? 'g' : 'r';
        html += `<div style="font-size:10px;font-family:'JetBrains Mono';color:var(--t3)">`
          + `<span class="bright">${sr.symbol}</span> `
          + `<span class="dim">trd ${sr.trades}</span> `
          + `<span class="dim">WR ${sr.wr_pct.toFixed(0)}%</span> `
          + `<span class="${psColor}">${sr.pnl_usd>=0?'+':''}$${sr.pnl_usd.toFixed(2)}</span>`
          + `</div>`;
      });
      html += '</td></tr>';
    }
  });
  html += '</table>';
  body.innerHTML = html;
}
function toggleStratSyms(strat) {
  const el = document.getElementById('sb-syms-' + strat);
  if (el) el.style.display = (el.style.display === 'none') ? 'table-row' : 'none';
}
function pollStrategyBattle() {
  fetch('/api/strategy_breakdown').then(r => r.json()).then(renderStrategyBattle).catch(() => {});
}

document.addEventListener('DOMContentLoaded', () => {
  buildControls();
  initCharts();
  updateClock();
  setInterval(updateClock, 1000);
  $('intel-sym').textContent = selectedSymbol;
  updateScanner();  // render cards immediately even without ticks
  pollRiskLocks();
  setInterval(pollRiskLocks, 5000);  // refresh cooldown/blacklist list every 5s
  pollStrategyBattle();
  setInterval(pollStrategyBattle, 10000);  // refresh strategy breakdown every 10s
});
</script>
</body>
</html>
"""


# ── Vue dashboard override (reactive, no-blink) ──
# Use make_response to bypass Jinja2 (Vue's {{ }} conflicts with Jinja)
try:
    from dashboard.vue_app import VUE_HTML
    from flask import make_response
    @app.route("/")
    def index():
        resp = make_response(VUE_HTML)
        resp.headers['Content-Type'] = 'text/html; charset=utf-8'
        # 2026-05-29: force fresh HTML every load. Without this the browser can
        # heuristically cache the page and keep serving an OLD build even across
        # refreshes (looked like new panels "not showing up").
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
except ImportError:
    @app.route("/")
    def index():
        return render_template_string(HTML)


def run_dashboard():
    """Start dashboard with WebSocket support in background threads."""
    log.info("Dashboard starting on port %d (WebSocket enabled)", DASHBOARD_PORT)

    # Start push threads
    threading.Thread(target=_push_ticks, daemon=True, name="DashPushTicks").start()
    # 2026-05-29: per-symbol candle chart removed — no candle push needed.
    # threading.Thread(target=_push_chart, daemon=True, name="DashPushChart").start()
    threading.Thread(target=_push_stats, daemon=True, name="DashPushStats").start()

    # v2 push threads (ticks:bulk, portfolio:update)
    try:
        if _v2_api is not None:
            _v2_api.start_push_threads()
    except Exception as e:
        log.warning("v2 push threads disabled: %s", e)

    socketio.run(app, host="0.0.0.0", port=DASHBOARD_PORT, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
