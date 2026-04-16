"""
Beast Trader — J.A.R.V.I.S. Dashboard.
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

log = logging.getLogger("beast.dashboard")

app = Flask(__name__)
app.config["SECRET_KEY"] = "beast-jarvis-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
IST = ZoneInfo("Asia/Kolkata")

# Shared state reference — set by run.py
_state = None
_executor = None


def init_dashboard(state, executor=None):
    global _state, _executor
    _state = state
    _executor = executor


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
            socketio.emit("tick_update", ticks)
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


def _push_stats():
    """Push full stats every 5s."""
    while True:
        time.sleep(5.0)
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
            feature_imp = agent.get("feature_importance", {})
            mode = agent.get("mode", "HYBRID")

            # Position map for scanner
            pos_map = {}
            for p in positions:
                sym = p.get("symbol", "")
                pnl = p.get("pnl", 0)
                side = p.get("type", "FLAT")
                pos_map[sym] = {"side": side, "pnl": pnl}

            data = {
                "equity": agent.get("equity", 0),
                "balance": agent.get("balance", 0),
                "profit": agent.get("profit", 0),
                "dd_pct": agent.get("dd_pct", 0),
                "daily_loss": agent.get("daily_loss", 0),
                "daily_pnl": agent.get("daily_pnl", 0),
                "cycle": agent.get("cycle", 0),
                "running": agent.get("running", False),
                "mode": mode,
                "positions": positions,
                "pos_map": pos_map,
                "session": session_name,
                "session_color": session_color,
                "scores": scores,
                "ml_confidence": ml_conf,
                "trade_log": trade_log[-10:] if trade_log else [],
                "equity_history": equity_history[-300:] if equity_history else [],
                "feature_importance": feature_imp,
                "risk_pct": agent.get("risk_pct", 0),
                "num_positions": len(positions),
                "time": datetime.now(IST).strftime("%H:%M:%S IST"),
            }
            socketio.emit("stats_update", data)
        except Exception as e:
            log.debug("stats push error: %s", e)


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
        "positions": agent.get("positions", []), "ticks": ticks,
        "time": datetime.now(IST).strftime("%H:%M:%S IST"),
    })


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
<title>B.E.A.S.T — J.A.R.V.I.S. Trading Terminal</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script src="https://cdn.socket.io/4.7.4/socket.io.min.js"></script>
<style>
/* ══════════════════════════════════════════════════════════════
   J.A.R.V.I.S. THEME — BEAST TRADING TERMINAL
   ══════════════════════════════════════════════════════════════ */
:root {
  --bg: #020810;
  --bg1: rgba(0,15,30,0.8);
  --bg2: rgba(0,30,60,0.4);
  --bg3: rgba(0,50,90,0.3);
  --bdr: rgba(0,240,255,0.08);
  --bdr2: rgba(0,240,255,0.15);
  --bdr3: rgba(0,240,255,0.25);
  --t1: #e0f4ff;
  --t2: rgba(0,200,255,0.7);
  --t3: rgba(0,200,255,0.35);
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
  background:var(--bg); color:var(--t2); border:1px solid var(--bdr2); padding:4px 8px;
  font-family:'JetBrains Mono'; font-size:10px; cursor:pointer;
}
.act-select option { background:#0a1a30; color:var(--t1); }

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
      <div class="logo-text">B.E.A.S.T</div>
      <div class="logo-sub">ML TRADING TERMINAL</div>
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
    <div class="status-dot" id="status-dot"></div>
    <div class="clock" id="h-clock">00:00:00 IST</div>
  </div>
</header>

<!-- ══════════════ MAIN GRID ══════════════ -->
<div class="main-grid">

  <!-- ═══ TICK CHART (top-left) ═══ -->
  <div class="card chart-panel">
    <div class="hud-corner hud-corner-tl"></div>
    <div class="hud-corner hud-corner-tr"></div>
    <div class="chart-controls">
      <div id="sym-tabs"></div>
      <div class="tf-sep"></div>
      <div id="tf-btns"></div>
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
      <span class="card-t">INTELLIGENCE</span>
      <span class="card-badge" id="intel-sym">---</span>
    </div>
    <div class="card-b" id="intel-body">
      <div class="intel-content">
        <div class="regime-badges" id="regime-badges"></div>
        <div class="holo-sep"></div>
        <div class="score-breakdown" id="score-breakdown">
          <div class="empty">Select a symbol for score breakdown</div>
        </div>
        <div class="holo-sep"></div>
        <div id="trade-log-wrap">
          <div class="empty">No recent trades</div>
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
        <div id="equity-chart-container"></div>
        <div class="perf-stats" id="perf-stats">
          <div class="perf-stat ps-g"><div class="ps-label">Win Rate</div><div class="ps-val" id="ps-wr">--</div></div>
          <div class="perf-stat ps-c"><div class="ps-label">Profit Factor</div><div class="ps-val" id="ps-pf">--</div></div>
          <div class="perf-stat ps-b"><div class="ps-label">Sharpe</div><div class="ps-val" id="ps-sharpe">--</div></div>
          <div class="perf-stat ps-a"><div class="ps-label">Avg R</div><div class="ps-val" id="ps-avgr">--</div></div>
          <div class="perf-stat ps-r"><div class="ps-label">Max DD</div><div class="ps-val" id="ps-dd">--</div></div>
        </div>
        <div class="r-hist-wrap">
          <div class="r-hist-title">Trade Distribution (R-Multiple)</div>
          <div class="r-hist" id="r-hist"></div>
          <div class="r-labels" id="r-labels"></div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
// ══════════════════════════════════════════════════════════════
// B.E.A.S.T. — CLIENT-SIDE ENGINE
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

// ── SOCKET.IO ──
const socket = io();

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
  const symWrap = $('sym-tabs');
  SYMBOLS.forEach(sym => {
    const btn = document.createElement('button');
    btn.className = 'sym-tab' + (sym === selectedSymbol ? ' active' : '');
    btn.textContent = sym;
    btn.onclick = () => selectSymbol(sym);
    symWrap.appendChild(btn);
  });

  const tfWrap = $('tf-btns');
  const tfs = [{v:1,l:'M1'},{v:5,l:'M5'},{v:15,l:'M15'},{v:60,l:'H1'}];
  tfs.forEach(t => {
    const btn = document.createElement('button');
    btn.className = 'tf-btn' + (t.v === selectedTF ? ' active' : '');
    btn.textContent = t.l;
    btn.onclick = () => selectTF(t.v);
    tfWrap.appendChild(btn);
  });

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
  document.querySelectorAll('.sym-tab').forEach(b => {
    b.classList.toggle('active', b.textContent === sym);
  });
  $('intel-sym').textContent = sym;
  refreshChart();
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
  const container = $('chart-container');

  mainChart = LightweightCharts.createChart(container, {
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
    rightPriceScale: {
      borderColor: 'rgba(0,240,255,0.1)',
      scaleMargins: { top: 0.1, bottom: 0.25 },
    },
    timeScale: {
      borderColor: 'rgba(0,240,255,0.1)',
      timeVisible: true,
      secondsVisible: false,
    },
    watermark: {
      visible: true,
      text: 'B.E.A.S.T',
      fontSize: 48,
      color: 'rgba(0,240,255,0.04)',
      horzAlign: 'center',
      vertAlign: 'center',
    },
  });

  candleSeries = mainChart.addCandlestickSeries({
    upColor: '#00ff88',
    downColor: '#ff3355',
    borderDownColor: '#ff3355',
    borderUpColor: '#00ff88',
    wickDownColor: '#ff3355',
    wickUpColor: '#00ff88',
  });

  volumeSeries = mainChart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
    color: 'rgba(0,240,255,0.15)',
  });
  mainChart.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.8, bottom: 0 },
  });

  ema20Series = mainChart.addLineSeries({
    color: '#00f0ff', lineWidth: 1, title: 'EMA20',
    priceLineVisible: false, lastValueVisible: false,
  });
  ema50Series = mainChart.addLineSeries({
    color: '#0088ff', lineWidth: 1, title: 'EMA50',
    priceLineVisible: false, lastValueVisible: false,
  });
  ema200Series = mainChart.addLineSeries({
    color: '#aa55ff', lineWidth: 1, title: 'EMA200',
    priceLineVisible: false, lastValueVisible: false,
  });

  // Equity curve chart
  const eqContainer = $('equity-chart-container');
  equityChart = LightweightCharts.createChart(eqContainer, {
    width: eqContainer.clientWidth,
    height: 140,
    layout: {
      background: { type: 'solid', color: 'transparent' },
      textColor: 'rgba(0,200,255,0.4)',
      fontSize: 9,
      fontFamily: 'JetBrains Mono',
    },
    grid: {
      vertLines: { color: 'rgba(0,240,255,0.03)' },
      horzLines: { color: 'rgba(0,240,255,0.03)' },
    },
    rightPriceScale: { borderColor: 'rgba(0,240,255,0.08)' },
    timeScale: { borderColor: 'rgba(0,240,255,0.08)', visible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });

  equitySeries = equityChart.addAreaSeries({
    lineColor: '#00f0ff',
    topColor: 'rgba(0,240,255,0.15)',
    bottomColor: 'rgba(0,240,255,0.02)',
    lineWidth: 2,
    priceLineVisible: false,
  });

  // Resize handler
  const resizeObserver = new ResizeObserver(() => {
    mainChart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    equityChart.applyOptions({ width: eqContainer.clientWidth });
  });
  resizeObserver.observe(container);
  resizeObserver.observe(eqContainer);
}

// ══════════════════════════════════════════════════════════════
// CHART DATA REFRESH
// ══════════════════════════════════════════════════════════════
function refreshChart() {
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
  // Store previous prices for direction
  for (const sym of SYMBOLS) {
    if (lastTicks[sym]) prevPrices[sym] = lastTicks[sym].bid;
  }
  lastTicks = ticks;
  updateScanner();
});

socket.on('chart_update', function(data) {
  // Store all chart data
  for (const [key, val] of Object.entries(data)) {
    if (!key.endsWith('_indicators')) {
      chartData[key] = val;
    }
  }
  refreshChart();
});

socket.on('stats_update', function(data) {
  updateHeader(data);
  updateIntelligence(data);
  updatePerformance(data);
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

  $('h-pos').textContent = d.num_positions || 0;
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

    if (!tick) {
      html += `<div class="sym-card">
        <div class="sym-row1"><span class="sym-name">${sym} <span class="sym-cat cat-${cat}">${cat}</span></span>
        <span class="sym-price dim">---</span></div>
        <div class="empty" style="padding:8px">Waiting...</div></div>`;
      return;
    }

    const isUp = prev ? tick.bid > prev : true;
    const arrow = isUp ? '&#9650;' : '&#9660;';
    const arrowCls = isUp ? 'up' : 'dn';

    // Scores (from last stats_update)
    const h1Score = window._lastScores && window._lastScores[sym] ? (window._lastScores[sym].h1_score || 0) : 0;
    const m5Score = window._lastScores && window._lastScores[sym] ? (window._lastScores[sym].m5_score || 0) : 0;
    const mlConf = window._lastML && window._lastML[sym] ? (window._lastML[sym].confidence || 0) : 0;

    // Gates
    const gates = window._lastScores && window._lastScores[sym] && window._lastScores[sym].gates
      ? window._lastScores[sym].gates : {};
    const gateItems = [
      {name:'TF', val:gates.tf},
      {name:'OFI', val:gates.ofi},
      {name:'VOL', val:gates.vol},
      {name:'REG', val:gates.reg},
    ];

    // Position
    const pos = window._lastPosMap && window._lastPosMap[sym] ? window._lastPosMap[sym] : null;
    let posTag;
    if (!pos) {
      posTag = '<span class="sym-pos pos-flat">FLAT</span>';
    } else {
      const side = pos.side === 'BUY' ? 'LONG' : 'SHORT';
      const cls = pos.side === 'BUY' ? 'pos-long' : 'pos-short';
      const rVal = pos.pnl ? (pos.pnl >= 0 ? '+' : '') + f(pos.pnl) : '';
      posTag = `<span class="sym-pos ${cls}">${side} ${rVal}</span>`;
    }

    // Sparkline
    const sparkData = tick.sparkline || [];

    // Score colors
    const h1Color = h1Score > 60 ? 'var(--green)' : h1Score > 30 ? 'var(--amber)' : 'var(--red)';
    const m5Color = m5Score > 60 ? 'var(--green)' : m5Score > 30 ? 'var(--amber)' : 'var(--red)';
    const mlColor = mlConf > 0.6 ? 'var(--green)' : mlConf > 0.3 ? 'var(--amber)' : 'var(--red)';

    html += `<div class="sym-card" onclick="selectSymbol('${sym}')">
      <div class="sym-row1">
        <span>
          <span class="sym-name">${sym}</span>
          <span class="sym-cat cat-${cat}">${cat}</span>
        </span>
        <span class="sym-price">${f(tick.bid, digits)} <span class="sym-arrow ${arrowCls}">${arrow}</span></span>
      </div>
      <div class="sym-row2">
        <span class="sym-detail">Bid <span>${f(tick.bid,digits)}</span></span>
        <span class="sym-detail">Ask <span>${f(tick.ask,digits)}</span></span>
        <span class="sym-detail">Spread <span>${f(tick.spread,1)}</span></span>
      </div>
      <div class="sym-scores">
        <div class="score-block">
          <div class="score-label">H1 Score ${f(h1Score,0)}%</div>
          <div class="score-bar"><div class="score-fill" style="width:${Math.min(100,h1Score)}%;background:${h1Color};color:${h1Color}"></div></div>
        </div>
        <div class="score-block">
          <div class="score-label">M5 Score ${f(m5Score,0)}%</div>
          <div class="score-bar"><div class="score-fill" style="width:${Math.min(100,m5Score)}%;background:${m5Color};color:${m5Color}"></div></div>
        </div>
      </div>
      <div class="ml-bar-wrap">
        <div class="ml-label"><span>ML Confidence</span><span>${f(mlConf*100,0)}%</span></div>
        <div class="ml-bar"><div class="ml-fill" style="width:${Math.min(100,mlConf*100)}%;background:${mlColor};color:${mlColor}"></div></div>
      </div>
      <div class="gate-row">
        ${gateItems.map(g => `<div class="gate"><div class="gate-dot ${g.val === true ? 'gate-pass' : g.val === false ? 'gate-block' : 'gate-na'}"></div>${g.name}</div>`).join('')}
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

  // Regime badges
  const regBadges = $('regime-badges');
  let rhtml = '';
  const confidence = d.ml_confidence || {};
  SYMBOLS.forEach(sym => {
    const sc = d.scores && d.scores[sym] ? d.scores[sym] : {};
    const regime = sc.regime || (confidence[sym] ? confidence[sym].regime : null) || 'unknown';
    const cls = 'regime-' + regime.toLowerCase().replace(/\s+/g,'_');
    rhtml += `<div class="regime-badge ${cls}">${sym}: ${regime.toUpperCase()}</div>`;
  });
  regBadges.innerHTML = rhtml;

  // Score breakdown for selected symbol
  const sbWrap = $('score-breakdown');
  const symScores = d.scores && d.scores[selectedSymbol] ? d.scores[selectedSymbol] : {};
  const components = symScores.components || symScores.breakdown || {};
  const compEntries = Object.entries(components);

  if (compEntries.length > 0) {
    let sbhtml = '';
    compEntries.forEach(([name, val]) => {
      const pct = Math.min(100, Math.max(0, Math.abs(val) * 100));
      const color = val > 0.5 ? 'var(--green)' : val > 0 ? 'var(--cyan)' : 'var(--red)';
      sbhtml += `<div class="sb-row">
        <div class="sb-label">${name}</div>
        <div class="sb-bar"><div class="sb-fill" style="width:${pct}%;background:${color};color:${color}"></div></div>
        <div class="sb-val">${f(val,2)}</div>
      </div>`;
    });
    sbWrap.innerHTML = sbhtml;
  } else {
    // Show ML confidence breakdown if available
    const mlConf = confidence[selectedSymbol];
    if (mlConf) {
      let sbhtml = '';
      const show = {prob_up:'P(Up)',prob_down:'P(Down)',confidence:'Confidence'};
      Object.entries(show).forEach(([key,label]) => {
        const val = mlConf[key] || 0;
        const pct = Math.min(100, val * 100);
        const color = val > 0.6 ? 'var(--green)' : val > 0.3 ? 'var(--amber)' : 'var(--red)';
        sbhtml += `<div class="sb-row">
          <div class="sb-label">${label}</div>
          <div class="sb-bar"><div class="sb-fill" style="width:${pct}%;background:${color};color:${color}"></div></div>
          <div class="sb-val">${f(val,3)}</div>
        </div>`;
      });
      sbWrap.innerHTML = sbhtml;
    }
  }

  // Feature importance
  const fimp = d.feature_importance || {};
  const fimpSym = fimp[selectedSymbol] || (Object.keys(fimp).length ? fimp[Object.keys(fimp)[0]] : null);
  if (fimpSym && Object.keys(fimpSym).length > 0) {
    const sorted = Object.entries(fimpSym).sort((a,b) => b[1] - a[1]).slice(0, 8);
    const maxVal = sorted[0][1] || 1;
    let fihtml = '<div class="holo-sep"></div>';
    sorted.forEach(([name, val]) => {
      const pct = (val / maxVal) * 100;
      fihtml += `<div class="sb-row">
        <div class="sb-label">${name}</div>
        <div class="sb-bar"><div class="sb-fill" style="width:${pct}%;background:var(--cyan);color:var(--cyan)"></div></div>
        <div class="sb-val">${f(val,3)}</div>
      </div>`;
    });
    sbWrap.innerHTML += fihtml;
  }

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

    // R-multiple histogram
    buildRHistogram(pnls);
  }
}

function buildRHistogram(pnls) {
  // Bucket into R-multiples: <-2R, -2R, -1R, -0.5R, 0, +0.5R, +1R, +2R, >+2R
  const buckets = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3];
  const labels = ['<-2R','-2R','-1R','-0.5R','BE','+0.5R','+1R','+2R','>2R'];
  const counts = new Array(buckets.length).fill(0);

  // Normalize by average loss as 1R
  const losses = pnls.filter(p => p < 0);
  const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((a,b) => a+b, 0) / losses.length) : 1;

  pnls.forEach(p => {
    const r = p / (avgLoss || 1);
    let idx = buckets.length - 1;
    for (let i = 0; i < buckets.length; i++) {
      if (r <= buckets[i]) { idx = i; break; }
    }
    counts[idx]++;
  });

  const maxCount = Math.max(...counts, 1);
  const histEl = $('r-hist');
  const labelsEl = $('r-labels');

  histEl.innerHTML = counts.map((c, i) => {
    const h = (c / maxCount) * 100;
    const cls = i < 4 ? 'r-bar-r' : 'r-bar-g';
    return `<div class="r-bar ${cls}" style="height:${Math.max(2,h)}%" title="${labels[i]}: ${c}"></div>`;
  }).join('');

  labelsEl.innerHTML = labels.map(l => `<div class="r-label">${l}</div>`).join('');
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
document.addEventListener('DOMContentLoaded', () => {
  buildControls();
  initCharts();
  updateClock();
  setInterval(updateClock, 1000);
  $('intel-sym').textContent = selectedSymbol;
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


def run_dashboard():
    """Start dashboard with WebSocket support in background threads."""
    log.info("Dashboard starting on port %d (WebSocket enabled)", DASHBOARD_PORT)

    # Start push threads
    threading.Thread(target=_push_ticks, daemon=True, name="DashPushTicks").start()
    threading.Thread(target=_push_chart, daemon=True, name="DashPushChart").start()
    threading.Thread(target=_push_stats, daemon=True, name="DashPushStats").start()

    socketio.run(app, host="0.0.0.0", port=DASHBOARD_PORT, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
