# dashboard/vue_app.py
# Dragon Trader — Vue 3 J.A.R.V.I.S. Dashboard
# Single-file Vue app served via Flask render_template_string
# Replaces innerHTML-based dashboard with Vue reactive bindings (no blinking)

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS, STARTING_BALANCE

SYMBOL_LIST_JSON = json.dumps(list(SYMBOLS.keys()))
SYMBOL_META_JSON = json.dumps({sym: {"category": cfg.category, "digits": cfg.digits} for sym, cfg in SYMBOLS.items()})
STARTING_BAL = STARTING_BALANCE

VUE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>D.R.A.G.O.N — J.A.R.V.I.S. Trading Terminal</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script src="https://cdn.socket.io/4.7.4/socket.io.min.js"></script>
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
<style>
/* ══════════════════════════════════════════════════════════════
   J.A.R.V.I.S. THEME — DRAGON TRADING TERMINAL (Vue Edition)
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
  font-variant-numeric: tabular-nums;
  min-width:70px; display:inline-block; text-align:center;
}
.hdr-stat .val.g { color:var(--green); text-shadow:0 0 12px rgba(0,255,136,0.3); }
.hdr-stat .val.r { color:var(--red); text-shadow:0 0 12px rgba(255,51,85,0.3); }
.hdr-stat .val.cy { color:var(--cyan); text-shadow:0 0 12px rgba(0,240,255,0.3); }

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
  min-width:145px; text-align:right;
  font-variant-numeric: tabular-nums;
}
.status-dot {
  width:8px; height:8px; border-radius:50%;
  transition: background 0.3s, box-shadow 0.3s;
}
.status-dot.connected { background:var(--green); box-shadow:0 0 8px var(--green), 0 0 16px rgba(0,255,136,0.3); animation: dotPulse 2s infinite; }
.status-dot.disconnected { background:var(--red); box-shadow:0 0 8px var(--red), 0 0 16px rgba(255,51,85,0.3); }
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
   TICK CHART PANEL
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
   MARKET SCANNER
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
  transition: background 0.3s; cursor:pointer;
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
   INTELLIGENCE PANEL
   ══════════════════════════════════════════════════════════════ */
.intel-content { padding:10px 14px; }
.regime-badges { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
.regime-badge {
  font-family:'JetBrains Mono'; font-size:9px; font-weight:600; padding:3px 10px;
  cursor:pointer;
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
   PERFORMANCE PANEL
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

/* ══════════════════════════════════════════════════════════════
   ADVANCED ANALYTICS
   ══════════════════════════════════════════════════════════════ */
/* Drawdown chart */
#dd-chart-container { height:100px; width:100%; margin-bottom:10px; }

/* Performance Attribution Table */
.attrib-table { width:100%; border-collapse:collapse; margin-bottom:10px; }
.attrib-table th {
  padding:4px 8px; text-align:left; font-family:'Orbitron'; font-size:7px;
  font-weight:600; color:var(--t3); text-transform:uppercase; letter-spacing:1px;
  background:linear-gradient(90deg,rgba(0,30,60,0.4),transparent);
  border-bottom:1px solid var(--bdr2);
}
.attrib-table td {
  padding:3px 8px; border-bottom:1px solid rgba(0,240,255,0.04); font-size:10px;
  font-family:'JetBrains Mono';
}
.attrib-table tr:hover td { background:rgba(0,240,255,0.03); }
.attrib-row-profit td { background:rgba(0,255,136,0.03); }
.attrib-row-loss td { background:rgba(255,51,85,0.03); }

/* Trade Distribution Charts */
.dist-charts { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px; }
.dist-chart-wrap { background:rgba(0,15,30,0.4); border:1px solid var(--bdr); padding:8px; }
.dist-title {
  font-family:'Orbitron'; font-size:8px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1.5px; margin-bottom:6px;
}
.dist-bars { display:flex; align-items:flex-end; gap:2px; height:50px; }
.dist-bar {
  flex:1; min-width:8px; position:relative; border-radius:1px 1px 0 0;
  transition: height 0.5s; cursor:default;
}
.dist-bar-g { background:var(--green); box-shadow:0 -2px 6px rgba(0,255,136,0.15); }
.dist-bar-r { background:var(--red); box-shadow:0 -2px 6px rgba(255,51,85,0.15); }
.dist-bar-mix { background:var(--cyan); box-shadow:0 -2px 6px rgba(0,240,255,0.15); }
.dist-labels { display:flex; gap:2px; margin-top:2px; }
.dist-label {
  flex:1; text-align:center; font-family:'JetBrains Mono'; font-size:7px; color:var(--t3);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}

/* Learning Engine Status */
.learn-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:6px; margin-bottom:10px; }
.learn-card {
  background:rgba(0,15,30,0.6); border:1px solid var(--bdr); padding:8px;
  position:relative; overflow:hidden;
  clip-path: polygon(0 0,calc(100% - 4px) 0,100% 4px,100% 100%,4px 100%,0 calc(100% - 4px));
}
.learn-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
}
.learn-card.lc-press::before { background:linear-gradient(90deg,var(--green),transparent); }
.learn-card.lc-reduce::before { background:linear-gradient(90deg,var(--red),transparent); }
.learn-card.lc-normal::before { background:linear-gradient(90deg,var(--cyan),transparent); }
.learn-sym {
  font-family:'Orbitron'; font-size:9px; font-weight:700; letter-spacing:1px; color:var(--t1);
  margin-bottom:4px;
}
.learn-row {
  display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:9px;
  color:var(--t3); margin-bottom:2px;
}
.learn-row .lv { font-weight:600; }

/* Analytics section title */
.analytics-title {
  font-family:'Orbitron'; font-size:8px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1.5px; margin:10px 0 6px 0;
}

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

/* ══════════════════════════════════════════════════════════════
   MTF INTELLIGENCE COMPONENTS
   ══════════════════════════════════════════════════════════════ */
/* Confluence dots row */
.mtf-confluence { display:flex; align-items:center; gap:6px; margin-bottom:6px; }
.mtf-dots { display:flex; gap:4px; align-items:center; }
.mtf-dot {
  width:8px; height:8px; border-radius:50%; position:relative;
  transition: background 0.3s, box-shadow 0.3s;
}
.mtf-dot-long { background:var(--green); box-shadow:0 0 6px rgba(0,255,136,0.5); }
.mtf-dot-short { background:var(--red); box-shadow:0 0 6px rgba(255,51,85,0.5); }
.mtf-dot-flat { background:rgba(100,120,140,0.4); box-shadow:none; }
.mtf-dot-label {
  font-family:'JetBrains Mono'; font-size:7px; color:var(--t3);
  position:absolute; top:-10px; left:50%; transform:translateX(-50%);
  white-space:nowrap; pointer-events:none; opacity:0;
  transition: opacity 0.2s;
}
.mtf-dot:hover .mtf-dot-label { opacity:1; }
.mtf-conf-count {
  font-family:'JetBrains Mono'; font-size:10px; font-weight:700;
  margin-left:4px; letter-spacing:0.5px;
}
.mtf-conf-0 { color:var(--t3); }
.mtf-conf-1 { color:var(--red); }
.mtf-conf-2 { color:var(--amber); }
.mtf-conf-3 { color:var(--green); }
.mtf-conf-4 { color:var(--green); text-shadow:0 0 8px rgba(0,255,136,0.4); }

/* Entry quality meter (inline bar) */
.eq-meter { display:flex; align-items:center; gap:6px; margin-bottom:6px; }
.eq-meter-label {
  font-family:'Orbitron'; font-size:7px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1px; width:28px; flex-shrink:0;
}
.eq-meter-bar {
  flex:1; height:6px; background:rgba(0,240,255,0.06); overflow:hidden; position:relative;
}
.eq-meter-fill {
  height:100%; transition:width 0.5s, background 0.3s;
  box-shadow:0 0 6px currentColor;
}
.eq-meter-val {
  font-family:'JetBrains Mono'; font-size:10px; font-weight:700; width:30px;
  text-align:right; flex-shrink:0;
}

/* Exit urgency indicator */
.exit-urgency {
  display:inline-flex; align-items:center; gap:4px; margin-left:8px;
  font-family:'JetBrains Mono'; font-size:9px; font-weight:600;
}
.exit-urgency-dot {
  width:6px; height:6px; border-radius:50%;
  transition: background 0.3s, box-shadow 0.3s;
}
.exit-urgency-low .exit-urgency-dot { background:var(--green); box-shadow:0 0 4px rgba(0,255,136,0.3); }
.exit-urgency-med .exit-urgency-dot { background:var(--amber); box-shadow:0 0 6px rgba(255,170,0,0.4); }
.exit-urgency-high .exit-urgency-dot { background:var(--red); box-shadow:0 0 8px rgba(255,51,85,0.5); animation:exitPulse 1s ease-in-out infinite; }
.exit-urgency-alarm .exit-urgency-dot { background:var(--red); box-shadow:0 0 12px rgba(255,51,85,0.8), 0 0 24px rgba(255,51,85,0.4); animation:exitPulse 0.5s ease-in-out infinite; }
@keyframes exitPulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.4)} }
.exit-urgency-low { color:var(--green); }
.exit-urgency-med { color:var(--amber); }
.exit-urgency-high { color:var(--red); }
.exit-urgency-alarm { color:var(--red); text-shadow:0 0 8px rgba(255,51,85,0.6); }

/* MTF Intelligence Panel (bottom-left enhancement) */
.mtf-panel { margin-top:8px; }
.mtf-panel-title {
  font-family:'Orbitron'; font-size:8px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1.5px; margin-bottom:8px;
}
.mtf-tf-grid {
  display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-bottom:10px;
}
.mtf-tf-cell {
  background:rgba(0,15,30,0.6); border:1px solid var(--bdr); padding:8px 6px;
  text-align:center; position:relative; overflow:hidden;
  clip-path: polygon(0 0,calc(100% - 4px) 0,100% 4px,100% 100%,4px 100%,0 calc(100% - 4px));
}
.mtf-tf-cell::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
}
.mtf-tf-cell.tf-long::before { background:linear-gradient(90deg,var(--green),transparent); }
.mtf-tf-cell.tf-short::before { background:linear-gradient(90deg,var(--red),transparent); }
.mtf-tf-cell.tf-flat::before { background:linear-gradient(90deg,rgba(100,120,140,0.3),transparent); }
.mtf-tf-label {
  font-family:'Orbitron'; font-size:8px; color:var(--t3); letter-spacing:1px;
  margin-bottom:4px;
}
.mtf-tf-dir {
  font-family:'JetBrains Mono'; font-size:12px; font-weight:700;
}
.mtf-tf-arrow { font-size:14px; display:block; margin-top:2px; }

.mtf-big-score {
  display:flex; align-items:center; justify-content:center; gap:16px;
  margin-bottom:10px; padding:8px;
  background:rgba(0,15,30,0.4); border:1px solid var(--bdr);
}
.mtf-confluence-big {
  font-family:'Orbitron'; font-size:36px; font-weight:900;
  text-shadow:0 0 20px rgba(0,240,255,0.3);
  line-height:1;
}
.mtf-confluence-label {
  font-family:'Orbitron'; font-size:7px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1.5px;
}
.mtf-eq-gauge {
  text-align:center;
}
.mtf-eq-gauge-val {
  font-family:'JetBrains Mono'; font-size:28px; font-weight:800;
  line-height:1;
}
.mtf-eq-gauge-label {
  font-family:'Orbitron'; font-size:7px; color:var(--t3); text-transform:uppercase;
  letter-spacing:1.5px;
}

.mtf-indicators {
  display:flex; gap:10px; flex-wrap:wrap; margin-bottom:6px;
  font-family:'JetBrains Mono'; font-size:10px;
}
.mtf-indicator {
  display:flex; align-items:center; gap:4px; color:var(--t3);
}
.mtf-indicator-val { font-weight:600; }

/* ── HUD CORNER DECORATIONS ── */
.hud-corner { position:absolute; width:12px; height:12px; z-index:3; }
.hud-corner-tl { top:0;left:0; border-top:2px solid var(--cyan); border-left:2px solid var(--cyan); }
.hud-corner-tr { top:0;right:0; border-top:2px solid var(--cyan); border-right:2px solid var(--cyan); }
.hud-corner-bl { bottom:0;left:0; border-bottom:2px solid var(--cyan); border-left:2px solid var(--cyan); }
.hud-corner-br { bottom:0;right:0; border-bottom:2px solid var(--cyan); border-right:2px solid var(--cyan); }

/* ── MasterBrain grid ── */
.mb-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-bottom:8px; }
.mb-info { display:flex; gap:12px; font-size:10px; font-family:'JetBrains Mono'; flex-wrap:wrap; }

/* ── RESPONSIVE ── */
@media(max-width:1200px) {
  .main-grid { grid-template-columns:1fr; grid-template-rows:auto; }
  .hdr { flex-wrap:wrap; height:auto; padding:8px 12px; gap:8px; }
  .hdr-stats { flex-wrap:wrap; gap:10px; }
}

/* ── MODAL ── */
.modal-overlay {
  position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:10000;
  display:flex; align-items:center; justify-content:center;
}
.modal-box {
  background:linear-gradient(135deg, rgba(0,15,30,0.98), rgba(0,10,25,0.99));
  border:1px solid var(--bdr3); padding:30px 40px; text-align:center;
  font-family:'Orbitron'; max-width:400px;
}
.modal-box h3 { color:var(--cyan); font-size:14px; letter-spacing:2px; margin-bottom:16px; }
.modal-box p { color:var(--t2); font-size:12px; margin-bottom:20px; font-family:'Rajdhani'; }
.modal-btns { display:flex; gap:12px; justify-content:center; }
.modal-btns button { padding:8px 24px; font-family:'Orbitron'; font-size:9px; font-weight:700; cursor:pointer; letter-spacing:1px; border:1px solid; }
.modal-confirm { color:var(--red); background:var(--red-bg); border-color:rgba(255,51,85,0.4); }
.modal-confirm:hover { background:rgba(255,51,85,0.25); }
.modal-cancel { color:var(--t3); background:var(--bg2); border-color:var(--bdr2); }
.modal-cancel:hover { background:rgba(0,240,255,0.1); }
</style>
</head>
<body>

<div id="app">
<div class="scanline"></div>

<!-- ══════════════ MODAL ══════════════ -->
<div class="modal-overlay" v-if="modal.show" @click.self="modal.show=false">
  <div class="modal-box">
    <h3>{{ modal.title }}</h3>
    <p>{{ modal.msg }}</p>
    <div class="modal-btns">
      <button class="modal-confirm" @click="modalConfirm">CONFIRM</button>
      <button class="modal-cancel" @click="modal.show=false">CANCEL</button>
    </div>
  </div>
</div>

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
    <div class="pill pill-cyan">{{ mode }}</div>
    <div class="pill pill-session" :style="{color: sessionColor, borderColor: sessionColor, background: sessionColor + '10'}">{{ session }}</div>
  </div>

  <div class="hdr-stats">
    <div class="hdr-stat">
      <div class="lbl">Balance</div>
      <div class="val cy">{{ fmtDollar(balance) }}</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Equity</div>
      <div class="val cy">{{ fmtDollar(equity) }}</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Float P&amp;L</div>
      <div class="val" :class="floatPnl >= 0 ? 'g' : 'r'">{{ fmtPnl(floatPnl) }}</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Daily P&amp;L</div>
      <div class="val" :class="dailyPnl >= 0 ? 'g' : 'r'">{{ fmtPnl(dailyPnl) }}</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Positions</div>
      <div class="val cy">{{ numPositions }}</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Risk%</div>
      <div class="val" :class="riskPct > 5 ? 'r' : riskPct > 2 ? 'cy' : 'g'">{{ fmtNum(riskPct, 1) }}%</div>
    </div>
  </div>

  <div class="hdr-actions">
    <button class="act-btn act-red" @click="showModal('CLOSE ALL', 'Close ALL open positions?', 'close_all')">CLOSE ALL</button>
    <button class="act-btn act-amber" @click="showModal('CLOSE LOSING', 'Close all LOSING positions?', 'close_losing')">CLOSE LOSING</button>
    <select class="act-select" v-model="closeSymSelect">
      <option value="">Symbol...</option>
      <option v-for="s in symbols" :key="s" :value="s">{{ s }}</option>
    </select>
    <button class="act-btn act-red" @click="doCloseSym">CLOSE</button>
  </div>

  <div class="hdr-right">
    <div class="status-dot" :class="connected ? 'connected' : 'disconnected'" :title="connected ? 'Connected' : 'Disconnected'"></div>
    <div class="clock">{{ clock }}</div>
    <div style="font-family:'JetBrains Mono';font-size:8px;color:var(--t3);min-width:80px;text-align:right;font-variant-numeric:tabular-nums">{{ lastUpdate }}</div>
  </div>
</header>

<!-- ══════════════ MAIN GRID ══════════════ -->
<div class="main-grid">

  <!-- ═══ TICK CHART (top-left) ═══ -->
  <div class="card chart-panel">
    <div class="hud-corner hud-corner-tl"></div>
    <div class="hud-corner hud-corner-tr"></div>
    <div class="chart-controls">
      <div>
        <button v-for="s in symbols" :key="s"
          class="sym-tab" :class="{active: selectedSymbol === s}"
          @click="selectSymbol(s)">{{ s }}</button>
      </div>
      <div class="tf-sep"></div>
      <div>
        <button v-for="t in timeframes" :key="t.v"
          class="tf-btn" :class="{active: selectedTF === t.v}"
          @click="selectTF(t.v)">{{ t.l }}</button>
      </div>
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
      <span class="card-badge">{{ symbols.length }} SYMBOLS</span>
    </div>
    <div class="card-b">
      <!-- SYMBOL CARDS — Vue reactive, no innerHTML -->
      <div v-for="sym in symbols" :key="sym"
        class="sym-card" :class="{selected: selectedSymbol === sym}"
        @click="selectSymbol(sym)">

        <!-- Row 1: Name + Price -->
        <div class="sym-row1">
          <span>
            <span class="sym-name">{{ sym }}</span>
            <span class="sym-cat" :class="'cat-' + getCat(sym)">{{ getCat(sym) }}</span>
          </span>
          <span class="sym-price">
            {{ getTickVal(sym, 'bid') !== null ? fmtPrice(getTickVal(sym, 'bid'), getDigits(sym)) : '---' }}
            <span v-if="getTickVal(sym, 'bid') !== null" class="sym-arrow" :class="isPriceUp(sym) ? 'up' : 'dn'">
              {{ isPriceUp(sym) ? '\u25B2' : '\u25BC' }}
            </span>
          </span>
        </div>

        <!-- Row 2: Bid/Ask/Spread -->
        <div class="sym-row2">
          <span class="sym-detail">Bid <span>{{ getTickVal(sym, 'bid') !== null ? fmtPrice(getTickVal(sym, 'bid'), getDigits(sym)) : '---' }}</span></span>
          <span class="sym-detail">Ask <span>{{ getTickVal(sym, 'ask') !== null ? fmtPrice(getTickVal(sym, 'ask'), getDigits(sym)) : '---' }}</span></span>
          <span class="sym-detail">Spread <span>{{ getTickVal(sym, 'spread') !== null ? fmtNum(getTickVal(sym, 'spread'), 1) : '---' }}</span></span>
        </div>

        <!-- Scores -->
        <div class="sym-scores">
          <div class="score-block">
            <div class="score-label">H1 L:{{ fmtNum(getScore(sym, 'long_score'), 1) }} S:{{ fmtNum(getScore(sym, 'short_score'), 1) }} min:{{ fmtNum(getScore(sym, 'adaptive_min_score') || 7.0, 1) }}</div>
            <div class="score-bar"><div class="score-fill" :style="{width: h1ScorePct(sym) + '%', background: h1ScoreColor(sym), color: h1ScoreColor(sym)}"></div></div>
          </div>
          <div class="score-block">
            <div class="score-label">TF: H1={{ getScore(sym, 'direction') || 'FLAT' }} M15={{ (getScore(sym, 'm15_dir') || 'flat').toUpperCase() }}</div>
            <div class="score-bar"><div class="score-fill" :style="{width: tfConfPct(sym) + '%', background: tfConfColor(sym), color: tfConfColor(sym)}"></div></div>
          </div>
        </div>

        <!-- Regime/ATR/Risk/Gate -->
        <div class="sym-row2" style="margin-bottom:4px">
          <span class="sym-detail">Regime <span style="color:var(--amber)">{{ (getScore(sym, 'regime') || 'unknown').toUpperCase() }}</span></span>
          <span class="sym-detail">ATR <span>{{ fmtNum(getScore(sym, 'atr') || 0, 2) }}</span></span>
          <span class="sym-detail">Risk <span :style="{color: getScore(sym, 'risk_pct') > 0 ? 'var(--green)' : 'var(--t3)'}">{{ getScore(sym, 'risk_pct') > 0 ? fmtNum(getScore(sym, 'risk_pct'), 3) + '%' : '\u2014' }}</span></span>
          <span class="sym-detail">Gate <span :style="{color: gateColor(getScore(sym, 'gate'))}">{{ getScore(sym, 'gate') || '\u2014' }}</span></span>
        </div>

        <!-- ML Bar -->
        <div class="ml-bar-wrap">
          <div class="ml-label">
            <span>ML {{ getML(sym, 'enabled') ? 'ON' : 'OFF' }}{{ getML(sym, 'auc') > 0 ? ' AUC:' + fmtNum(getML(sym, 'auc'), 2) : '' }}</span>
            <span>{{ getScore(sym, 'meta_prob') != null ? fmtNum(getScore(sym, 'meta_prob') * 100, 0) + '%' : 'no signal' }}</span>
          </div>
          <div class="ml-bar"><div class="ml-fill" :style="{width: mlBarPct(sym) + '%', background: mlBarColor(sym), color: mlBarColor(sym)}"></div></div>
        </div>

        <!-- Gate dots -->
        <div class="gate-row">
          <div v-for="g in getGateDots(sym)" :key="g.name" class="gate">
            <div class="gate-dot" :class="'gate-' + g.val"></div>{{ g.name }}
          </div>
        </div>

        <!-- MTF Confluence Bar -->
        <div v-if="getMtf(sym)" class="mtf-confluence">
          <span style="font-family:'Orbitron';font-size:7px;color:var(--t3);letter-spacing:1px;width:28px">MTF</span>
          <div class="mtf-dots">
            <div class="mtf-dot" :class="mtfDotClass(getMtf(sym, 'h1_dir'))" title="H1">
              <span class="mtf-dot-label">H1</span>
            </div>
            <div class="mtf-dot" :class="mtfDotClass(getMtf(sym, 'm15_dir'))" title="M15">
              <span class="mtf-dot-label">M15</span>
            </div>
            <div class="mtf-dot" :class="mtfDotClass(getMtf(sym, 'm5_dir'))" title="M5">
              <span class="mtf-dot-label">M5</span>
            </div>
            <div class="mtf-dot" :class="mtfDotClass(getMtf(sym, 'm1_dir'))" title="M1">
              <span class="mtf-dot-label">M1</span>
            </div>
          </div>
          <span class="mtf-conf-count" :class="'mtf-conf-' + (getMtf(sym, 'confluence') || 0)">{{ getMtf(sym, 'confluence') || 0 }}/4</span>
        </div>

        <!-- Entry Quality Meter -->
        <div v-if="getMtf(sym)" class="eq-meter">
          <span class="eq-meter-label">EQ</span>
          <div class="eq-meter-bar">
            <div class="eq-meter-fill" :style="{width: (getMtf(sym, 'entry_quality') || 0) + '%', background: eqColor(getMtf(sym, 'entry_quality') || 0), color: eqColor(getMtf(sym, 'entry_quality') || 0)}"></div>
          </div>
          <span class="eq-meter-val" :style="{color: eqColor(getMtf(sym, 'entry_quality') || 0)}">{{ getMtf(sym, 'entry_quality') || 0 }}</span>
        </div>

        <!-- Position + Sparkline -->
        <div class="sym-row-bottom">
          <span v-if="!posMap[sym]" class="sym-pos pos-flat">FLAT</span>
          <span v-else style="display:flex;align-items:center">
            <span class="sym-pos" :style="{color: posMap[sym].pnl >= 0 ? 'var(--green)' : 'var(--red)'}">
              {{ posMap[sym].side === 'BUY' ? 'LONG' : 'SHORT' }}
              {{ posMap[sym].pnl >= 0 ? '+$' : '-$' }}{{ fmtNum(Math.abs(posMap[sym].pnl), 2) }}
            </span>
            <span v-if="getMtf(sym, 'exit_urgency') > 0" class="exit-urgency" :class="exitUrgencyClass(getMtf(sym, 'exit_urgency'))">
              <span class="exit-urgency-dot"></span>{{ fmtNum(getMtf(sym, 'exit_urgency') * 100, 0) }}%
            </span>
          </span>
          <canvas :ref="el => { if(el) sparkRefs[sym] = el }" class="mini-sparkline" width="80" height="20"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ INTELLIGENCE (bottom-left) ═══ -->
  <div class="card">
    <div class="hud-corner hud-corner-bl"></div>
    <div class="card-h">
      <span class="card-t">DRAGON INTELLIGENCE</span>
      <span class="card-badge">{{ selectedSymbol }}</span>
    </div>
    <div class="card-b">
      <div class="intel-content">
        <!-- Regime badges -->
        <div class="regime-badges">
          <div v-for="sym in symbols" :key="sym"
            class="regime-badge" :class="'regime-' + getRegimeClass(sym)"
            @click="selectSymbol(sym)">
            {{ sym }}: {{ (getScore(sym, 'regime') || 'unknown').toUpperCase() }}
          </div>
        </div>

        <div class="holo-sep"></div>

        <!-- MasterBrain Status -->
        <div v-if="masterBrain">
          <div class="mb-grid">
            <div class="perf-stat ps-c">
              <div class="ps-label">Eq Health</div>
              <div class="ps-val" :style="{color: mbHealthColor, fontSize:'11px'}">{{ (masterBrain.equity_health || 'unknown').toUpperCase() }}</div>
            </div>
            <div class="perf-stat ps-b">
              <div class="ps-label">Eq Slope</div>
              <div class="ps-val" :style="{fontSize:'11px', color: (masterBrain.equity_slope||0) >= 0 ? 'var(--green)' : 'var(--red)'}">{{ fmtNum(masterBrain.equity_slope || 0, 4) }}</div>
            </div>
            <div class="perf-stat ps-a">
              <div class="ps-label">Daily Trades</div>
              <div class="ps-val" style="font-size:11px">{{ masterBrain.daily_trades || 0 }}</div>
            </div>
            <div class="perf-stat ps-g">
              <div class="ps-label">Win Rate</div>
              <div class="ps-val" :style="{fontSize:'11px', color: (masterBrain.win_rate||0) >= 50 ? 'var(--green)' : 'var(--red)'}">{{ fmtNum(masterBrain.win_rate || 0, 1) }}%</div>
            </div>
          </div>
          <div class="mb-info">
            <span style="color:var(--t3)">Daily P&amp;L: <span :style="{color: (masterBrain.daily_pnl||0) >= 0 ? 'var(--green)' : 'var(--red)'}">${{ fmtNum(masterBrain.daily_pnl || 0, 2) }}</span></span>
            <span style="color:var(--t3)">Last 10: <span :style="{color: (masterBrain.recent_10_pnl||0) >= 0 ? 'var(--green)' : 'var(--red)'}">${{ fmtNum(masterBrain.recent_10_pnl || 0, 2) }}</span></span>
            <span style="color:var(--t3)">Total: <span style="color:var(--cyan)">{{ masterBrain.total_trades || 0 }}</span></span>
            <span v-if="masterBrain.losing_day_yesterday" style="color:var(--red)">PREV DAY LOSS (risk halved)</span>
            <span v-if="masterBrain.session_paused" style="color:var(--red);font-weight:700">CIRCUIT BREAKER ACTIVE ({{ masterBrain.session_losses || 0 }} losses)</span>
            <span v-else style="color:var(--t3)">Session losses: {{ masterBrain.session_losses || 0 }}/2</span>
          </div>
          <div style="margin-top:4px;font-size:10px;font-family:'JetBrains Mono';color:var(--t3)">
            Blacklisted: <span :style="{color: mbBlacklistColor}">{{ mbBlacklistStr }}</span>
          </div>
        </div>
        <div v-else class="empty">MasterBrain loading...</div>

        <div class="holo-sep"></div>

        <!-- Score Breakdown for selected symbol -->
        <div class="score-breakdown">
          <div style="font-family:Orbitron;font-size:8px;color:var(--t3);letter-spacing:1.5px;margin-bottom:6px">
            {{ selectedSymbol }} SIGNAL BREAKDOWN
            <span v-if="selectedGate === 'SESSION'" style="color:var(--red);margin-left:8px">SESSION CLOSED</span>
          </div>

          <!-- Core metrics -->
          <div class="sb-row" v-for="m in breakdownMetrics" :key="m.label">
            <div class="sb-label">{{ m.label }}</div>
            <div class="sb-bar"><div class="sb-fill" :style="{width: m.pct + '%', background: m.color, color: m.color}"></div></div>
            <div class="sb-val">{{ fmtNum(m.val, 2) }}</div>
          </div>

          <!-- ML Meta Prob -->
          <div class="sb-row" v-if="selectedScores.meta_prob != null">
            <div class="sb-label">ML Meta Prob</div>
            <div class="sb-bar"><div class="sb-fill" :style="{width: Math.min(100, selectedScores.meta_prob * 100) + '%', background: selectedScores.meta_prob > 0.6 ? 'var(--green)' : selectedScores.meta_prob > 0.4 ? 'var(--amber)' : 'var(--red)', color: selectedScores.meta_prob > 0.6 ? 'var(--green)' : selectedScores.meta_prob > 0.4 ? 'var(--amber)' : 'var(--red)'}"></div></div>
            <div class="sb-val">{{ fmtNum(selectedScores.meta_prob, 3) }}</div>
          </div>

          <!-- ML AUC -->
          <div class="sb-row" v-if="selectedMLAuc > 0">
            <div class="sb-label">ML AUC</div>
            <div class="sb-bar"><div class="sb-fill" :style="{width: Math.min(100, Math.max(0, (selectedMLAuc - 0.5) / 0.5) * 100) + '%', background: selectedMLAuc >= 0.70 ? 'var(--green)' : selectedMLAuc >= 0.60 ? 'var(--amber)' : 'var(--red)', color: selectedMLAuc >= 0.70 ? 'var(--green)' : selectedMLAuc >= 0.60 ? 'var(--amber)' : 'var(--red)'}"></div></div>
            <div class="sb-val">{{ fmtNum(selectedMLAuc, 3) }}</div>
          </div>

          <!-- Risk pct -->
          <div class="sb-row" v-if="selectedScores.risk_pct">
            <div class="sb-label">Risk %</div>
            <div class="sb-bar"><div class="sb-fill" :style="{width: Math.min(100, selectedScores.risk_pct / 0.5 * 100) + '%', background: 'var(--blue)', color: 'var(--blue)'}"></div></div>
            <div class="sb-val">{{ fmtNum(selectedScores.risk_pct, 3) }}%</div>
          </div>

          <!-- Direction / M15 / Regime / Gate summary -->
          <div style="margin-top:8px;display:flex;gap:12px;font-size:10px;font-family:'JetBrains Mono';flex-wrap:wrap">
            <span style="color:var(--t3)">Dir: <span :style="{color: selectedScores.direction === 'LONG' ? 'var(--green)' : selectedScores.direction === 'SHORT' ? 'var(--red)' : 'var(--t3)'}">{{ selectedScores.direction || 'FLAT' }}</span></span>
            <span style="color:var(--t3)">M15: <span style="color:var(--cyan)">{{ (selectedScores.m15_dir || 'flat').toUpperCase() }}</span></span>
            <span style="color:var(--t3)">Regime: <span style="color:var(--amber)">{{ (selectedScores.regime || 'unknown').toUpperCase() }}</span></span>
            <span style="color:var(--t3)">Gate: <span :style="{color: selectedGate === 'ENTERED' ? 'var(--green)' : 'var(--amber)'}">{{ selectedGate || '\u2014' }}</span></span>
          </div>

          <!-- Master reject reason -->
          <div v-if="selectedScores.master_reason" style="margin-top:4px;font-size:10px;font-family:'JetBrains Mono';color:var(--red)">
            Master Reject: {{ selectedScores.master_reason }}
          </div>

          <!-- MTF Intelligence Panel -->
          <div v-if="selectedMtf" class="mtf-panel">
            <div class="holo-sep"></div>
            <div class="mtf-panel-title">MTF INTELLIGENCE — {{ selectedSymbol }}</div>

            <!-- 4-TF Direction Grid -->
            <div class="mtf-tf-grid">
              <div v-for="tf in [{k:'h1_dir',l:'H1'},{k:'m15_dir',l:'M15'},{k:'m5_dir',l:'M5'},{k:'m1_dir',l:'M1'}]" :key="tf.k"
                class="mtf-tf-cell" :class="'tf-' + (selectedMtf[tf.k] || 'FLAT').toLowerCase()">
                <div class="mtf-tf-label">{{ tf.l }}</div>
                <div class="mtf-tf-dir" :style="{color: mtfDirColor(selectedMtf[tf.k])}">{{ (selectedMtf[tf.k] || 'FLAT') }}</div>
                <span class="mtf-tf-arrow" :style="{color: mtfDirColor(selectedMtf[tf.k])}">{{ mtfArrow(selectedMtf[tf.k]) }}</span>
              </div>
            </div>

            <!-- Big Confluence + Entry Quality -->
            <div class="mtf-big-score">
              <div style="text-align:center">
                <div class="mtf-confluence-big" :class="'mtf-conf-' + (selectedMtf.confluence || 0)">{{ selectedMtf.confluence || 0 }}</div>
                <div class="mtf-confluence-label">Confluence</div>
              </div>
              <div style="width:1px;height:50px;background:var(--bdr2)"></div>
              <div class="mtf-eq-gauge">
                <div class="mtf-eq-gauge-val" :style="{color: eqColor(selectedMtf.entry_quality || 0)}">{{ selectedMtf.entry_quality || 0 }}</div>
                <div class="mtf-eq-gauge-label">Entry Quality</div>
              </div>
              <div style="width:1px;height:50px;background:var(--bdr2)"></div>
              <div style="text-align:center">
                <div class="mtf-eq-gauge-val" :style="{color: exitUrgencyColor(selectedMtf.exit_urgency || 0), fontSize:'24px'}">{{ fmtNum((selectedMtf.exit_urgency || 0) * 100, 0) }}%</div>
                <div class="mtf-eq-gauge-label">Exit Urgency</div>
              </div>
            </div>

            <!-- Regime + Volume + Swing indicators -->
            <div class="mtf-indicators">
              <div class="mtf-indicator">
                Regime: <span class="mtf-indicator-val" style="color:var(--amber)">{{ (getScore(selectedSymbol, 'regime') || 'unknown').toUpperCase() }}</span>
              </div>
              <div v-if="selectedMtf.volume_trend" class="mtf-indicator">
                Vol: <span class="mtf-indicator-val" :style="{color: selectedMtf.volume_trend === 'bullish' ? 'var(--green)' : selectedMtf.volume_trend === 'bearish' ? 'var(--red)' : 'var(--t3)'}">{{ (selectedMtf.volume_trend || 'flat').toUpperCase() }}</span>
              </div>
              <div v-if="selectedMtf.swing_structure" class="mtf-indicator">
                Swing: <span class="mtf-indicator-val" :style="{color: selectedMtf.swing_structure === 'uptrend' ? 'var(--green)' : selectedMtf.swing_structure === 'downtrend' ? 'var(--red)' : 'var(--cyan)'}">{{ (selectedMtf.swing_structure || 'sideways').toUpperCase() }}</span>
              </div>
              <div v-if="selectedMtf.momentum" class="mtf-indicator">
                Mom: <span class="mtf-indicator-val" :style="{color: selectedMtf.momentum > 0 ? 'var(--green)' : selectedMtf.momentum < 0 ? 'var(--red)' : 'var(--t3)'}">{{ fmtNum(selectedMtf.momentum, 2) }}</span>
              </div>
              <div v-if="selectedMtf.order_flow" class="mtf-indicator">
                Flow: <span class="mtf-indicator-val" :style="{color: selectedMtf.order_flow > 0 ? 'var(--green)' : selectedMtf.order_flow < 0 ? 'var(--red)' : 'var(--t3)'}">{{ fmtNum(selectedMtf.order_flow, 2) }}</span>
              </div>
            </div>
          </div>

          <!-- Feature importance -->
          <template v-if="topFeatures.length > 0">
            <div class="holo-sep"></div>
            <div class="sb-row" v-for="feat in topFeatures" :key="feat.name">
              <div class="sb-label">{{ feat.name }}</div>
              <div class="sb-bar"><div class="sb-fill" :style="{width: feat.pct + '%', background:'var(--cyan)', color:'var(--cyan)'}"></div></div>
              <div class="sb-val">{{ fmtNum(feat.val, 3) }}</div>
            </div>
          </template>
        </div>

        <div class="holo-sep"></div>

        <!-- Trade Log (paginated, latest first) -->
        <div>
          <div v-if="tradeLog.length > 0" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span class="dim" style="font-family:'JetBrains Mono';font-size:9px">
              {{ tradePageStart + 1 }}-{{ Math.min(tradePageStart + tradePageSize, tradeLog.length) }} of {{ tradeLog.length }}
            </span>
            <div style="display:flex;gap:4px">
              <button class="act-btn act-amber" style="padding:2px 8px;font-size:7px" @click="tradePage = Math.max(0, tradePage - 1)" :disabled="tradePage === 0">&lt; PREV</button>
              <button class="act-btn act-amber" style="padding:2px 8px;font-size:7px" @click="tradePage++" :disabled="tradePageStart + tradePageSize >= tradeLog.length">NEXT &gt;</button>
            </div>
          </div>
          <table v-if="tradeLog.length > 0" class="trade-table">
            <tr><th>Time</th><th>Symbol</th><th>Dir</th><th>Vol</th><th>P&amp;L</th><th>Exit</th></tr>
            <tr v-for="(t, i) in tradeLogPaged" :key="i">
              <td class="mono dim">{{ t.timestamp || t.time || '' }}</td>
              <td class="bright">{{ t.symbol || '' }}</td>
              <td><span class="tag" :class="tradeTagClass(t)">{{ tradeDir(t) }}</span></td>
              <td class="mono dim">{{ t.volume ? t.volume.toFixed(2) : '' }}</td>
              <td class="mono" :class="(t.pnl || 0) >= 0 ? 'g' : 'r'" style="font-weight:700">${{ fmtPnl(t.pnl || t.profit || 0) }}</td>
              <td class="mono dim" style="font-size:9px;max-width:80px;overflow:hidden;text-overflow:ellipsis">{{ t.action || '' }}</td>
            </tr>
          </table>
          <div v-else class="empty">No recent trades</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ PERFORMANCE (bottom-right) ═══ -->
  <div class="card">
    <div class="hud-corner hud-corner-br"></div>
    <div class="card-h">
      <span class="card-t">PERFORMANCE</span>
      <span class="card-badge">{{ tradeLog.length }} TRADES</span>
    </div>
    <div class="card-b">
      <div class="perf-content">
        <div id="equity-chart-container"></div>
        <div class="perf-stats">
          <div class="perf-stat ps-g">
            <div class="ps-label">Win Rate</div>
            <div class="ps-val" :class="perfStats.wr >= 50 ? 'g' : 'r'">{{ perfStats.wrStr }}</div>
          </div>
          <div class="perf-stat ps-c">
            <div class="ps-label">Profit Factor</div>
            <div class="ps-val" :class="perfStats.pf >= 1 ? 'g' : 'r'">{{ perfStats.pfStr }}</div>
          </div>
          <div class="perf-stat ps-b">
            <div class="ps-label">Sharpe</div>
            <div class="ps-val" :class="perfStats.sharpe >= 0 ? 'g' : 'r'">{{ perfStats.sharpeStr }}</div>
          </div>
          <div class="perf-stat ps-a">
            <div class="ps-label">Avg R</div>
            <div class="ps-val" :class="perfStats.avgR >= 0 ? 'g' : 'r'">{{ perfStats.avgRStr }}</div>
          </div>
          <div class="perf-stat ps-r">
            <div class="ps-label">Max DD</div>
            <div class="ps-val" :class="perfStats.maxDD > 5 ? 'r' : 'cy'">{{ perfStats.maxDDStr }}</div>
          </div>
        </div>
        <!-- R-Multiple Histogram -->
        <div class="r-hist-wrap">
          <div class="r-hist-title">Trade Distribution (R-Multiple)</div>
          <div class="r-hist">
            <div v-for="(bar, i) in rHistBars" :key="i"
              class="r-bar" :class="i < 4 ? 'r-bar-r' : 'r-bar-g'"
              :style="{height: bar.h + '%'}" :title="bar.label + ': ' + bar.count"></div>
          </div>
          <div class="r-labels">
            <div v-for="(bar, i) in rHistBars" :key="i" class="r-label">{{ bar.label }}</div>
          </div>
        </div>

        <div class="holo-sep"></div>

        <!-- ═══ DRAWDOWN OVERLAY ═══ -->
        <div class="analytics-title">DRAWDOWN</div>
        <div id="dd-chart-container"></div>

        <div class="holo-sep"></div>

        <!-- ═══ PERFORMANCE ATTRIBUTION TABLE ═══ -->
        <div class="analytics-title">PERFORMANCE ATTRIBUTION</div>
        <table v-if="perfAttribution.length > 0" class="attrib-table">
          <tr>
            <th>Symbol</th><th>PnL</th><th>W</th><th>L</th><th>WR%</th><th>PF</th><th>Avg R</th>
          </tr>
          <tr v-for="a in perfAttribution" :key="a.symbol"
            :class="a.pnl >= 0 ? 'attrib-row-profit' : 'attrib-row-loss'">
            <td style="font-weight:700;color:var(--t1)">{{ a.symbol }}</td>
            <td :class="a.pnl >= 0 ? 'g' : 'r'" style="font-weight:700">{{ a.pnl >= 0 ? '+' : '' }}{{ fmtNum(a.pnl, 2) }}</td>
            <td class="g">{{ a.wins }}</td>
            <td class="r">{{ a.losses }}</td>
            <td :class="a.wr >= 50 ? 'g' : 'r'">{{ fmtNum(a.wr, 1) }}%</td>
            <td :class="a.pf >= 1 ? 'g' : 'r'">{{ fmtNum(a.pf, 2) }}</td>
            <td :class="a.avgR >= 0 ? 'g' : 'r'">{{ fmtNum(a.avgR, 2) }}</td>
          </tr>
        </table>
        <div v-else class="empty">No attribution data</div>

        <div class="holo-sep"></div>

        <!-- ═══ TRADE DISTRIBUTION BY HOUR / DAY ═══ -->
        <div class="analytics-title">TRADE DISTRIBUTION</div>
        <div class="dist-charts">
          <!-- By Hour of Day -->
          <div class="dist-chart-wrap">
            <div class="dist-title">PnL by Hour (UTC)</div>
            <div class="dist-bars">
              <div v-for="(bar, i) in hourlyDistBars" :key="i"
                class="dist-bar" :class="bar.pnl >= 0 ? 'dist-bar-g' : 'dist-bar-r'"
                :style="{height: bar.h + '%'}"
                :title="bar.label + ': $' + fmtNum(bar.pnl, 2) + ' (' + bar.count + ' trades)'"></div>
            </div>
            <div class="dist-labels">
              <div v-for="(bar, i) in hourlyDistBars" :key="i" class="dist-label">{{ bar.label }}</div>
            </div>
          </div>
          <!-- By Day of Week -->
          <div class="dist-chart-wrap">
            <div class="dist-title">PnL by Day of Week</div>
            <div class="dist-bars">
              <div v-for="(bar, i) in dailyDistBars" :key="i"
                class="dist-bar" :class="bar.pnl >= 0 ? 'dist-bar-g' : 'dist-bar-r'"
                :style="{height: bar.h + '%'}"
                :title="bar.label + ': $' + fmtNum(bar.pnl, 2) + ' (' + bar.count + ' trades)'"></div>
            </div>
            <div class="dist-labels">
              <div v-for="(bar, i) in dailyDistBars" :key="i" class="dist-label">{{ bar.label }}</div>
            </div>
          </div>
        </div>

        <div class="holo-sep"></div>

        <!-- ═══ LEARNING ENGINE STATUS ═══ -->
        <div class="analytics-title">LEARNING ENGINE — ADAPTIVE RISK</div>
        <div v-if="Object.keys(learningStats).length > 0" class="learn-grid">
          <div v-for="(ls, sym) in learningStats" :key="sym"
            class="learn-card" :class="ls.risk_mult > 1.0 ? 'lc-press' : ls.risk_mult < 1.0 ? 'lc-reduce' : 'lc-normal'">
            <div class="learn-sym">{{ sym }}</div>
            <div class="learn-row">
              <span>Risk Mult</span>
              <span class="lv" :style="{color: ls.risk_mult > 1.0 ? 'var(--green)' : ls.risk_mult < 1.0 ? 'var(--red)' : 'var(--cyan)'}">{{ fmtNum(ls.risk_mult, 2) }}x</span>
            </div>
            <div class="learn-row">
              <span>Rolling PF</span>
              <span class="lv" :style="{color: ls.pf >= 1 ? 'var(--green)' : 'var(--red)'}">{{ fmtNum(ls.pf, 2) }}</span>
            </div>
            <div class="learn-row">
              <span>WR%</span>
              <span class="lv" :style="{color: ls.wr >= 50 ? 'var(--green)' : 'var(--red)'}">{{ fmtNum(ls.wr, 1) }}%</span>
            </div>
            <div class="learn-row">
              <span>Trades</span>
              <span class="lv" style="color:var(--t2)">{{ ls.trades || 0 }}</span>
            </div>
          </div>
        </div>
        <div v-else class="empty">Learning engine warming up...</div>
      </div>
    </div>
  </div>

</div><!-- main-grid -->
</div><!-- #app -->

<script>
// ══════════════════════════════════════════════════════════════
// D.R.A.G.O.N. — Vue 3 Reactive Dashboard Engine
// ══════════════════════════════════════════════════════════════

const SYMBOLS = """ + SYMBOL_LIST_JSON + r""";
const SYMBOL_META = """ + SYMBOL_META_JSON + r""";
const STARTING_BAL = """ + str(STARTING_BAL) + r""";

const { createApp, ref, reactive, computed, watch, onMounted, nextTick, toRaw } = Vue;

const app = createApp({
  setup() {
    // ── CORE STATE ──
    const symbols = ref(SYMBOLS);
    const selectedSymbol = ref(SYMBOLS[0]);
    const selectedTF = ref(5);
    const connected = ref(false);
    const clock = ref('00:00:00 IST');
    const lastUpdate = ref('--');
    const closeSymSelect = ref('');

    // ── HEADER STATE ──
    const mode = ref('HYBRID');
    const session = ref('---');
    const sessionColor = ref('#00f0ff');
    const balance = ref(0);
    const equity = ref(0);
    const floatPnl = ref(0);
    const dailyPnl = ref(0);
    const todayClosedPnl = ref(0);  // from MT5 history, updated every 5s
    const numPositions = ref(0);
    const riskPct = ref(0);
    const running = ref(false);
    const tradePage = ref(0);
    const tradePageSize = 10;

    // ── DATA STORES ──
    const ticks = reactive({});
    const prevPrices = reactive({});
    const scores = reactive({});
    const mlConfidence = reactive({});
    const posMap = reactive({});
    const positions = ref([]);
    const tradeLog = ref([]);
    const equityHistory = ref([]);
    const featureImportance = reactive({});
    const mtfIntelligence = reactive({});
    const masterBrain = ref(null);
    const chartDataStore = reactive({});
    const learningStats = reactive({});

    // ── MODAL ──
    const modal = reactive({ show: false, title: '', msg: '', action: '' });

    // ── SPARKLINE REFS ──
    const sparkRefs = reactive({});

    // ── TIMEFRAMES ──
    const timeframes = ref([
      { v: 1, l: 'M1' }, { v: 5, l: 'M5' }, { v: 15, l: 'M15' }, { v: 60, l: 'H1' }
    ]);

    // ═══════════════════════════════════════════
    // FORMATTING HELPERS
    // ═══════════════════════════════════════════
    function fmtNum(v, d) {
      if (v == null || isNaN(v)) return '---';
      return Number(v).toFixed(d || 2);
    }
    function fmtDollar(v) {
      if (v == null) return '---';
      return '$' + Number(v).toFixed(2);
    }
    function fmtPnl(v) {
      if (v == null) return '---';
      const n = Number(v);
      return (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(2);
    }
    function fmtPrice(v, d) {
      if (v == null) return '---';
      return Number(v).toFixed(d || 2);
    }

    // ═══════════════════════════════════════════
    // SYMBOL HELPERS
    // ═══════════════════════════════════════════
    function getCat(sym) { return SYMBOL_META[sym] ? SYMBOL_META[sym].category : 'Forex'; }
    function getDigits(sym) { return SYMBOL_META[sym] ? SYMBOL_META[sym].digits : 2; }

    function getTickVal(sym, field) {
      return ticks[sym] ? ticks[sym][field] : null;
    }

    function isPriceUp(sym) {
      const bid = getTickVal(sym, 'bid');
      const prev = prevPrices[sym];
      if (bid == null || prev == null) return true;
      return bid >= prev;
    }

    function getScore(sym, field) {
      return scores[sym] ? scores[sym][field] : null;
    }
    function getML(sym, field) {
      return mlConfidence[sym] ? mlConfidence[sym][field] : null;
    }

    // ═══════════════════════════════════════════
    // SCANNER COMPUTED HELPERS
    // ═══════════════════════════════════════════
    function h1ScorePct(sym) {
      const sc = scores[sym] || {};
      const raw = Math.max(sc.long_score || 0, sc.short_score || 0);
      return Math.min(100, raw / 14.0 * 100);
    }
    function h1ScoreColor(sym) {
      const pct = h1ScorePct(sym);
      return pct > 60 ? 'var(--green)' : pct > 30 ? 'var(--amber)' : 'var(--red)';
    }

    function tfConfPct(sym) {
      const sc = scores[sym] || {};
      const h1Dir = sc.direction || 'FLAT';
      const m15Dir = (sc.m15_dir || 'flat').toUpperCase();
      if (h1Dir !== 'FLAT' && m15Dir === h1Dir) return 100;
      if (h1Dir !== 'FLAT') return 50;
      return 0;
    }
    function tfConfColor(sym) {
      const pct = tfConfPct(sym);
      return pct === 100 ? 'var(--green)' : pct === 50 ? 'var(--amber)' : 'var(--red)';
    }

    function mlBarPct(sym) {
      const mp = getScore(sym, 'meta_prob');
      return mp != null ? Math.min(100, mp * 100) : 0;
    }
    function mlBarColor(sym) {
      const mp = getScore(sym, 'meta_prob');
      const v = mp != null ? mp : 0;
      return v > 0.6 ? 'var(--green)' : v > 0.3 ? 'var(--amber)' : 'var(--red)';
    }

    function gateColor(gate) {
      if (!gate) return 'var(--t3)';
      if (gate === 'ENTERED') return 'var(--green)';
      if (gate.includes('REJECT') || gate.includes('DISAGREE')) return 'var(--red)';
      return 'var(--amber)';
    }

    function getGateDots(sym) {
      const sc = scores[sym] || {};
      const gate = sc.gate || '';
      const gateOrder = ['SESSION','NO_H1_DATA','INSUFFICIENT_IND','BELOW_MIN_SCORE','M15_DISAGREE','TICK_DELAY','META_REJECT','MASTER_REJECT'];
      const gateMap = { SCORE: 'BELOW_MIN_SCORE', M15: 'M15_DISAGREE', META: 'META_REJECT', MASTER: 'MASTER_REJECT' };
      function status(gateKey) {
        if (!gate) return 'na';
        if (gate === 'ENTERED' || gate === 'HOLD_SWING' || gate === 'REVERSAL') return 'pass';
        const myReject = gateMap[gateKey];
        if (gate === myReject) return 'block';
        const myIdx = gateOrder.indexOf(myReject);
        const curIdx = gateOrder.indexOf(gate);
        if (curIdx >= 0 && myIdx >= 0 && curIdx < myIdx) return 'na';
        if (curIdx >= 0 && myIdx >= 0 && curIdx > myIdx) return 'pass';
        return 'na';
      }
      return [
        { name: 'SCORE', val: status('SCORE') },
        { name: 'M15', val: status('M15') },
        { name: 'META', val: status('META') },
        { name: 'MASTER', val: status('MASTER') },
      ];
    }

    function getRegimeClass(sym) {
      const sc = scores[sym] || {};
      const ml = mlConfidence[sym] || {};
      const regime = sc.regime || ml.regime || 'unknown';
      return regime.toLowerCase().replace(/\s+/g, '_');
    }

    // ═══════════════════════════════════════════
    // MTF INTELLIGENCE HELPERS
    // ═══════════════════════════════════════════
    function getMtf(sym, field) {
      const data = mtfIntelligence[sym];
      if (!data) return field ? null : null;
      if (!field) return data;
      return data[field] != null ? data[field] : null;
    }

    function mtfDotClass(dir) {
      if (!dir) return 'mtf-dot-flat';
      const d = dir.toUpperCase();
      if (d === 'LONG') return 'mtf-dot-long';
      if (d === 'SHORT') return 'mtf-dot-short';
      return 'mtf-dot-flat';
    }

    function mtfDirColor(dir) {
      if (!dir) return 'var(--t3)';
      const d = dir.toUpperCase();
      if (d === 'LONG') return 'var(--green)';
      if (d === 'SHORT') return 'var(--red)';
      return 'var(--t3)';
    }

    function mtfArrow(dir) {
      if (!dir) return '\u2500';
      const d = dir.toUpperCase();
      if (d === 'LONG') return '\u25B2';
      if (d === 'SHORT') return '\u25BC';
      return '\u2500';
    }

    function eqColor(val) {
      if (val > 60) return 'var(--green)';
      if (val > 30) return 'var(--amber)';
      return 'var(--red)';
    }

    function exitUrgencyClass(val) {
      if (val == null || val <= 0.2) return 'exit-urgency-low';
      if (val <= 0.5) return 'exit-urgency-med';
      if (val <= 0.7) return 'exit-urgency-high';
      return 'exit-urgency-alarm';
    }

    function exitUrgencyColor(val) {
      if (val <= 0.2) return 'var(--green)';
      if (val <= 0.5) return 'var(--amber)';
      return 'var(--red)';
    }

    const selectedMtf = computed(() => mtfIntelligence[selectedSymbol.value] || null);

    // ═══════════════════════════════════════════
    // INTELLIGENCE COMPUTED
    // ═══════════════════════════════════════════
    const selectedScores = computed(() => scores[selectedSymbol.value] || {});
    const selectedGate = computed(() => selectedScores.value.gate || '');
    const selectedMLAuc = computed(() => {
      const ml = mlConfidence[selectedSymbol.value];
      return ml ? (ml.auc || 0) : 0;
    });

    const breakdownMetrics = computed(() => {
      const sc = selectedScores.value;
      return [
        { label: 'Long Score', val: sc.long_score || 0, pct: Math.min(100, (sc.long_score || 0) / 14 * 100), color: 'var(--green)' },
        { label: 'Short Score', val: sc.short_score || 0, pct: Math.min(100, (sc.short_score || 0) / 14 * 100), color: 'var(--red)' },
        { label: 'Adaptive Min', val: sc.adaptive_min_score || 7, pct: Math.min(100, (sc.adaptive_min_score || 7) / 14 * 100), color: 'var(--amber)' },
        { label: 'ATR', val: sc.atr || 0, pct: 100, color: 'var(--cyan)' },
      ];
    });

    const topFeatures = computed(() => {
      const fi = featureImportance[selectedSymbol.value];
      if (!fi || Object.keys(fi).length === 0) return [];
      const sorted = Object.entries(fi).sort((a, b) => b[1] - a[1]).slice(0, 8);
      const maxVal = sorted[0][1] || 1;
      return sorted.map(([name, val]) => ({ name, val, pct: (val / maxVal) * 100 }));
    });

    const mbHealthColor = computed(() => {
      if (!masterBrain.value) return 'var(--t3)';
      const h = masterBrain.value.equity_health || 'unknown';
      return h === 'healthy' ? 'var(--green)' : h === 'flat' ? 'var(--amber)' : 'var(--red)';
    });
    const mbBlacklistStr = computed(() => {
      if (!masterBrain.value) return 'None';
      const bl = masterBrain.value.blacklisted_symbols || {};
      const entries = Object.entries(bl);
      return entries.length > 0 ? entries.map(([s, h]) => s + ' (' + Number(h).toFixed(1) + 'h)').join(', ') : 'None';
    });
    const mbBlacklistColor = computed(() => {
      if (!masterBrain.value) return 'var(--green)';
      return Object.keys(masterBrain.value.blacklisted_symbols || {}).length > 0 ? 'var(--red)' : 'var(--green)';
    });

    // ═══════════════════════════════════════════
    // TRADE LOG COMPUTED (latest first, paginated)
    // ═══════════════════════════════════════════
    // Backend sends latest first — no reverse needed
    const tradeLogReversed = computed(() => tradeLog.value);
    const tradePageStart = computed(() => tradePage.value * tradePageSize);
    const tradeLogPaged = computed(() => {
      const start = tradePageStart.value;
      return tradeLog.value.slice(start, start + tradePageSize);
    });

    function tradeDir(t) {
      return (t.direction || t.type || '').toUpperCase();
    }
    function tradeTagClass(t) {
      const dir = tradeDir(t);
      if (dir === 'LONG' || dir === 'BUY') return 'tag-long';
      if (dir === 'SHORT' || dir === 'SELL') return 'tag-short';
      return 'tag-flat';
    }

    // ═══════════════════════════════════════════
    // PERFORMANCE COMPUTED
    // ═══════════════════════════════════════════
    const perfStats = computed(() => {
      const trades = tradeLog.value;
      if (!trades || trades.length === 0) {
        return { wr: 0, pf: 0, sharpe: 0, avgR: 0, maxDD: 0,
          wrStr: '--', pfStr: '--', sharpeStr: '--', avgRStr: '--', maxDDStr: '--' };
      }
      const pnls = trades.map(t => t.pnl || t.profit || 0);
      const wins = pnls.filter(p => p > 0);
      const losses = pnls.filter(p => p < 0);
      const wr = wins.length / pnls.length * 100;
      const grossProfit = wins.reduce((a, b) => a + b, 0);
      const grossLoss = Math.abs(losses.reduce((a, b) => a + b, 0));
      const pf = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 99.99 : 0;
      const avgR = pnls.reduce((a, b) => a + b, 0) / pnls.length;
      const mean = avgR;
      const variance = pnls.reduce((a, b) => a + (b - mean) ** 2, 0) / pnls.length;
      const stddev = Math.sqrt(variance) || 1;
      const sharpe = mean / stddev * Math.sqrt(252);
      let peak = STARTING_BAL, maxDD = 0, runEq = STARTING_BAL;
      pnls.forEach(p => {
        runEq += p;
        if (runEq > peak) peak = runEq;
        const dd = (peak - runEq) / peak * 100;
        if (dd > maxDD) maxDD = dd;
      });
      return {
        wr, pf, sharpe, avgR, maxDD,
        wrStr: wr.toFixed(1) + '%',
        pfStr: pf.toFixed(2),
        sharpeStr: sharpe.toFixed(2),
        avgRStr: '$' + avgR.toFixed(2),
        maxDDStr: maxDD.toFixed(1) + '%',
      };
    });

    const rHistBars = computed(() => {
      const trades = tradeLog.value;
      const buckets = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3];
      const labels = ['<-2R', '-2R', '-1R', '-0.5R', 'BE', '+0.5R', '+1R', '+2R', '>2R'];
      const counts = new Array(buckets.length).fill(0);
      if (!trades || trades.length === 0) {
        return labels.map((l, i) => ({ label: l, count: 0, h: 2 }));
      }
      const pnls = trades.map(t => t.pnl || t.profit || 0);
      const losses = pnls.filter(p => p < 0);
      const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((a, b) => a + b, 0) / losses.length) : 1;
      pnls.forEach(p => {
        const r = p / (avgLoss || 1);
        let idx = buckets.length - 1;
        for (let i = 0; i < buckets.length; i++) {
          if (r <= buckets[i]) { idx = i; break; }
        }
        counts[idx]++;
      });
      const maxCount = Math.max(...counts, 1);
      return labels.map((l, i) => ({
        label: l,
        count: counts[i],
        h: Math.max(2, (counts[i] / maxCount) * 100),
      }));
    });

    // ═══════════════════════════════════════════
    // PERFORMANCE ATTRIBUTION (per-symbol stats from trade_log)
    // ═══════════════════════════════════════════
    const perfAttribution = computed(() => {
      const trades = tradeLog.value;
      if (!trades || trades.length === 0) return [];
      const bySymbol = {};
      trades.forEach(t => {
        const sym = t.symbol || 'UNKNOWN';
        const pnl = t.pnl || t.profit || 0;
        if (!bySymbol[sym]) bySymbol[sym] = { symbol: sym, pnls: [] };
        bySymbol[sym].pnls.push(pnl);
      });
      return Object.values(bySymbol).map(s => {
        const wins = s.pnls.filter(p => p > 0);
        const losses = s.pnls.filter(p => p < 0);
        const totalPnl = s.pnls.reduce((a, b) => a + b, 0);
        const grossP = wins.reduce((a, b) => a + b, 0);
        const grossL = Math.abs(losses.reduce((a, b) => a + b, 0));
        const avgLoss = losses.length > 0 ? grossL / losses.length : 1;
        return {
          symbol: s.symbol,
          pnl: Math.round(totalPnl * 100) / 100,
          wins: wins.length,
          losses: losses.length,
          wr: s.pnls.length > 0 ? wins.length / s.pnls.length * 100 : 0,
          pf: grossL > 0 ? grossP / grossL : (grossP > 0 ? 99.99 : 0),
          avgR: s.pnls.length > 0 ? (totalPnl / s.pnls.length) / (avgLoss || 1) : 0,
        };
      }).sort((a, b) => b.pnl - a.pnl);
    });

    // ═══════════════════════════════════════════
    // TRADE DISTRIBUTION (by hour and day of week)
    // ═══════════════════════════════════════════
    function parseTradeHour(t) {
      // Parse hour from timestamp like "04-15 14:30" or "2026-04-15 14:30:00"
      const ts = t.timestamp || t.time || '';
      const match = ts.match(/(\d{1,2}):(\d{2})/);
      return match ? parseInt(match[1]) : null;
    }

    function parseTradeDow(t) {
      // Parse day of week from timestamp — try MM-DD HH:MM format
      const ts = t.timestamp || t.time || '';
      // Try "MM-DD HH:MM" format (most common from MT5 deals)
      const match = ts.match(/^(\d{2})-(\d{2})\s/);
      if (match) {
        const year = new Date().getFullYear();
        const d = new Date(year, parseInt(match[1]) - 1, parseInt(match[2]));
        return d.getDay(); // 0=Sun, 1=Mon ... 6=Sat
      }
      // Try full date
      const d = new Date(ts);
      return isNaN(d.getTime()) ? null : d.getDay();
    }

    const hourlyDistBars = computed(() => {
      const trades = tradeLog.value;
      // Group into 6 buckets: 0-3, 4-7, 8-11, 12-15, 16-19, 20-23
      const bucketLabels = ['0-3', '4-7', '8-11', '12-15', '16-19', '20-23'];
      const bucketPnl = new Array(6).fill(0);
      const bucketCount = new Array(6).fill(0);
      if (trades && trades.length > 0) {
        trades.forEach(t => {
          const h = parseTradeHour(t);
          if (h !== null) {
            const idx = Math.min(5, Math.floor(h / 4));
            bucketPnl[idx] += (t.pnl || t.profit || 0);
            bucketCount[idx]++;
          }
        });
      }
      const maxAbs = Math.max(...bucketPnl.map(Math.abs), 0.01);
      return bucketLabels.map((l, i) => ({
        label: l,
        pnl: Math.round(bucketPnl[i] * 100) / 100,
        count: bucketCount[i],
        h: Math.max(3, (Math.abs(bucketPnl[i]) / maxAbs) * 100),
      }));
    });

    const dailyDistBars = computed(() => {
      const trades = tradeLog.value;
      const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      const dayPnl = new Array(7).fill(0);
      const dayCount = new Array(7).fill(0);
      if (trades && trades.length > 0) {
        trades.forEach(t => {
          const dow = parseTradeDow(t);
          if (dow !== null) {
            dayPnl[dow] += (t.pnl || t.profit || 0);
            dayCount[dow]++;
          }
        });
      }
      // Only show Mon-Fri (indices 1-5)
      const subset = [1, 2, 3, 4, 5];
      const maxAbs = Math.max(...subset.map(i => Math.abs(dayPnl[i])), 0.01);
      return subset.map(i => ({
        label: dayNames[i],
        pnl: Math.round(dayPnl[i] * 100) / 100,
        count: dayCount[i],
        h: Math.max(3, (Math.abs(dayPnl[i]) / maxAbs) * 100),
      }));
    });

    // ═══════════════════════════════════════════
    // ACTIONS
    // ═══════════════════════════════════════════
    function selectSymbol(sym) {
      selectedSymbol.value = sym;
      socket.emit('select_symbol', { symbol: sym });
      refreshChart();
    }

    function selectTF(tf) {
      selectedTF.value = tf;
      socket.emit('select_timeframe', { timeframe: tf });
      refreshChart();
    }

    function showModal(title, msg, action) {
      modal.title = title;
      modal.msg = msg;
      modal.action = action;
      modal.show = true;
    }
    function modalConfirm() {
      if (modal.action === 'close_all') socket.emit('close_all');
      else if (modal.action === 'close_losing') socket.emit('close_losing');
      else if (modal.action === 'close_symbol') socket.emit('close_symbol', { symbol: modal.data });
      modal.show = false;
    }
    function doCloseSym() {
      if (closeSymSelect.value) {
        showModal('CLOSE SYMBOL', 'Close ' + closeSymSelect.value + '?', 'close_symbol');
        modal.data = closeSymSelect.value;
      }
    }

    // ═══════════════════════════════════════════
    // SPARKLINE DRAWING
    // ═══════════════════════════════════════════
    function drawSparkline(canvas, data, color) {
      if (!canvas || !data || data.length < 2) return;
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

    function drawAllSparklines() {
      SYMBOLS.forEach(sym => {
        const tick = ticks[sym];
        if (tick && tick.sparkline && sparkRefs[sym]) {
          drawSparkline(sparkRefs[sym], tick.sparkline, '#00f0ff');
        }
      });
    }

    // ═══════════════════════════════════════════
    // LIGHTWEIGHT CHARTS
    // ═══════════════════════════════════════════
    let mainChart = null;
    let candleSeries = null;
    let volumeSeries = null;
    let ema20Series = null;
    let ema50Series = null;
    let ema200Series = null;
    let equityChart = null;
    let equitySeries = null;
    let localEquityHist = [];
    let ddChart = null;
    let ddSeries = null;

    function calcEMA(closes, times, period) {
      if (closes.length < period) return [];
      const alpha = 2 / (period + 1);
      const result = [];
      let ema = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
      for (let i = period - 1; i < closes.length; i++) {
        if (i === period - 1) {
          ema = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
        } else {
          ema = alpha * closes[i] + (1 - alpha) * ema;
        }
        result.push({ time: times[i], value: parseFloat(ema.toFixed(5)) });
      }
      return result;
    }

    function initCharts() {
      const container = document.getElementById('chart-container');
      if (!container) return;

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
          text: 'D.R.A.G.O.N',
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
      const eqContainer = document.getElementById('equity-chart-container');
      if (!eqContainer) return;

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

      // ── Drawdown chart ──
      const ddContainer = document.getElementById('dd-chart-container');
      if (ddContainer) {
        ddChart = LightweightCharts.createChart(ddContainer, {
          width: ddContainer.clientWidth,
          height: 100,
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

        ddSeries = ddChart.addHistogramSeries({
          color: 'rgba(255,51,85,0.5)',
          priceFormat: { type: 'custom', formatter: (v) => v.toFixed(1) + '%' },
          priceLineVisible: false,
          lastValueVisible: true,
        });
      }

      // Resize handler
      const resizeObserver = new ResizeObserver(() => {
        if (mainChart) mainChart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
        if (equityChart) equityChart.applyOptions({ width: eqContainer.clientWidth });
        if (ddChart && ddContainer) ddChart.applyOptions({ width: ddContainer.clientWidth });
      });
      resizeObserver.observe(container);
      resizeObserver.observe(eqContainer);
      if (ddContainer) resizeObserver.observe(ddContainer);
    }

    function refreshChart() {
      if (!candleSeries) return;
      const key = selectedSymbol.value + '_' + selectedTF.value;
      const data = chartDataStore[key];
      if (!data) return;

      if (data.candles && data.candles.length > 0) {
        candleSeries.setData(data.candles);
      }
      if (data.volumes && data.volumes.length > 0) {
        const colored = data.volumes.map((v, i) => {
          const c = data.candles[i];
          const isUp = c && c.close >= c.open;
          return { ...v, color: isUp ? 'rgba(0,255,136,0.2)' : 'rgba(255,51,85,0.2)' };
        });
        volumeSeries.setData(colored);
      }

      // EMA overlays
      if (data.candles && data.candles.length > 20) {
        const closes = data.candles.map(c => c.close);
        const times = data.candles.map(c => c.time);
        ema20Series.setData(calcEMA(closes, times, 20));
        ema50Series.setData(calcEMA(closes, times, 50));
        ema200Series.setData(calcEMA(closes, times, Math.min(200, closes.length)));
      }

      mainChart.timeScale().fitContent();
    }

    function updateEquityChart(eqHist) {
      if (!equitySeries) return;
      let eqData;
      if (eqHist && eqHist.length > 0) {
        const now = Math.floor(Date.now() / 1000);
        eqData = eqHist.map((v, i) => ({
          time: now - (eqHist.length - i) * 60,
          value: typeof v === 'number' ? v : (v.equity || v.value || 0),
        }));
        equitySeries.setData(eqData);
        equityChart.timeScale().fitContent();
      } else {
        const eq = equity.value || STARTING_BAL;
        localEquityHist.push(eq);
        if (localEquityHist.length > 300) localEquityHist = localEquityHist.slice(-300);
        const now = Math.floor(Date.now() / 1000);
        eqData = localEquityHist.map((v, i) => ({
          time: now - (localEquityHist.length - i) * 5,
          value: v,
        }));
        equitySeries.setData(eqData);
        equityChart.timeScale().fitContent();
      }

      // ── Update Drawdown Chart ──
      if (ddSeries && eqData && eqData.length > 0) {
        let peak = eqData[0].value;
        const ddData = eqData.map(pt => {
          if (pt.value > peak) peak = pt.value;
          const ddPct = peak > 0 ? -((peak - pt.value) / peak * 100) : 0;
          return { time: pt.time, value: ddPct, color: ddPct < -5 ? 'rgba(255,51,85,0.7)' : ddPct < -2 ? 'rgba(255,170,0,0.5)' : 'rgba(255,51,85,0.3)' };
        });
        ddSeries.setData(ddData);
        ddChart.timeScale().fitContent();
      }
    }

    // ═══════════════════════════════════════════
    // SOCKET.IO
    // ═══════════════════════════════════════════
    const socket = io({
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: Infinity,
    });

    socket.on('connect', () => {
      connected.value = true;
      console.log('Socket connected');
    });
    socket.on('disconnect', () => {
      connected.value = false;
      console.log('Socket disconnected');
    });
    socket.on('reconnect', (attempt) => {
      console.log('Reconnected after', attempt, 'attempts');
    });

    socket.on('tick_update', (data) => {
      try {
        // Save prev prices for arrow direction
        for (const sym of SYMBOLS) {
          if (ticks[sym]) prevPrices[sym] = ticks[sym].bid;
        }

        // Account data from MT5
        if (data._account) {
          const a = data._account;
          balance.value = a.balance || 0;
          equity.value = a.equity || 0;
          floatPnl.value = a.profit || 0;
          // Real-time daily PnL = today's closed PnL + current floating PnL
          dailyPnl.value = Math.round((todayClosedPnl.value + (a.profit || 0)) * 100) / 100;
          delete data._account;
        }

        // Position map from MT5
        if (data._pos_map) {
          // Clear and rebuild reactively
          Object.keys(posMap).forEach(k => delete posMap[k]);
          Object.assign(posMap, data._pos_map);
          delete data._pos_map;
        }

        // Positions list
        if (data._positions) {
          positions.value = data._positions;
          const uniqSyms = new Set(data._positions.map(p => p.symbol));
          numPositions.value = uniqSyms.size;
          delete data._positions;
        }

        // Update ticks reactively
        for (const [sym, val] of Object.entries(data)) {
          ticks[sym] = val;
        }

        lastUpdate.value = 'upd ' + new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });

        // Redraw sparklines after Vue updates the DOM
        nextTick(drawAllSparklines);
      } catch (e) { console.error('tick_update error:', e); }
    });

    socket.on('chart_update', (data) => {
      try {
        for (const [key, val] of Object.entries(data)) {
          if (!key.endsWith('_indicators')) {
            chartDataStore[key] = val;
          }
        }
        refreshChart();
      } catch (e) { console.error('chart_update error:', e); }
    });

    socket.on('stats_update', (data) => {
      try {
        // Header
        balance.value = data.balance || balance.value;
        equity.value = data.equity || equity.value;
        floatPnl.value = data.profit || 0;
        // Update today's closed PnL from backend (refreshes every 5s)
        if (data.today_closed_pnl !== undefined) todayClosedPnl.value = data.today_closed_pnl;
        dailyPnl.value = data.daily_pnl || 0;
        riskPct.value = data.risk_pct || data.dd_pct || 0;
        mode.value = (data.mode || 'HYBRID').toUpperCase();
        running.value = data.running || false;

        if (data.session) {
          session.value = data.session;
          sessionColor.value = data.session_color || '#00f0ff';
        }

        // Positions count from stats (use unique symbols)
        if (data.positions) {
          const uniqSyms = new Set(data.positions.map(p => p.symbol));
          numPositions.value = uniqSyms.size;
        }

        // Scores
        if (data.scores) {
          Object.keys(data.scores).forEach(k => { scores[k] = data.scores[k]; });
        }

        // ML confidence
        if (data.ml_confidence) {
          Object.keys(data.ml_confidence).forEach(k => { mlConfidence[k] = data.ml_confidence[k]; });
        }

        // Position map from stats
        if (data.pos_map) {
          Object.keys(posMap).forEach(k => delete posMap[k]);
          Object.assign(posMap, data.pos_map);
        }

        // MTF Intelligence
        if (data.mtf_intelligence) {
          Object.keys(data.mtf_intelligence).forEach(k => { mtfIntelligence[k] = data.mtf_intelligence[k]; });
        }

        // MasterBrain
        masterBrain.value = data.master_brain || null;

        // Trade log (reset page if count changed)
        const newLog = data.trade_log || [];
        if (newLog.length !== tradeLog.value.length) tradePage.value = 0;
        tradeLog.value = newLog;

        // Feature importance
        if (data.feature_importance) {
          Object.keys(data.feature_importance).forEach(k => { featureImportance[k] = data.feature_importance[k]; });
        }

        // Learning engine stats
        if (data.learning_stats) {
          Object.keys(learningStats).forEach(k => delete learningStats[k]);
          Object.assign(learningStats, data.learning_stats);
        }

        // Equity history
        equityHistory.value = data.equity_history || [];
        updateEquityChart(equityHistory.value);

        // Sparklines update
        nextTick(drawAllSparklines);
      } catch (e) { console.error('stats_update error:', e); }
    });

    socket.on('action_result', (data) => {
      console.log('Action result:', data);
    });

    // ═══════════════════════════════════════════
    // CLOCK
    // ═══════════════════════════════════════════
    function updateClock() {
      clock.value = new Date().toLocaleTimeString('en-IN', {
        timeZone: 'Asia/Kolkata', hour12: false
      }) + ' IST';
    }

    // ═══════════════════════════════════════════
    // LIFECYCLE
    // ═══════════════════════════════════════════
    onMounted(() => {
      initCharts();
      updateClock();
      setInterval(updateClock, 1000);
    });

    return {
      // State
      symbols, selectedSymbol, selectedTF, connected, clock, lastUpdate,
      mode, session, sessionColor, balance, equity, floatPnl, dailyPnl,
      numPositions, riskPct, running, closeSymSelect,
      ticks, prevPrices, scores, mlConfidence, posMap, positions,
      tradeLog, equityHistory, featureImportance, mtfIntelligence, masterBrain,
      modal, sparkRefs, timeframes,

      // Formatting
      fmtNum, fmtDollar, fmtPnl, fmtPrice,

      // Helpers
      getCat, getDigits, getTickVal, isPriceUp,
      getScore, getML,
      h1ScorePct, h1ScoreColor, tfConfPct, tfConfColor,
      mlBarPct, mlBarColor, gateColor, getGateDots,
      getRegimeClass,
      getMtf, mtfDotClass, mtfDirColor, mtfArrow,
      eqColor, exitUrgencyClass, exitUrgencyColor,

      // Computed
      selectedMtf, selectedScores, selectedGate, selectedMLAuc,
      breakdownMetrics, topFeatures,
      mbHealthColor, mbBlacklistStr, mbBlacklistColor,
      tradeLogReversed, tradeLogPaged, tradePage, tradePageSize, tradePageStart,
      perfStats, rHistBars,

      // Actions
      selectSymbol, selectTF,
      showModal, modalConfirm, doCloseSym,
      tradeDir, tradeTagClass,
    };
  }
});

app.mount('#app');
</script>
</body>
</html>
"""
