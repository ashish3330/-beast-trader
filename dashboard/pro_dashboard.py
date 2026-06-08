"""
Dragon Trader — PRO Dashboard (Vue 3 SPA, single-file).
Hedge-fund-grade terminal: Trading / Analytics / Risk / System / Journal.

Mounted at /pro by app.py (parent will register the blueprint).
All data flows in via:
    HTTP    GET   /api/v2/*    (read-only snapshots)
    Socket  ticks:bulk, portfolio:update, signal:scored, alert:new,
            position:opened, position:closed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Blueprint, render_template_string

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS, STARTING_BALANCE  # noqa: E402

pro_dashboard_bp = Blueprint("pro_dashboard", __name__, url_prefix="/pro")

# ── Bootstrap data injected at render time ──────────────────────────────────
SYMBOL_LIST = list(SYMBOLS.keys())
SYMBOL_META = {
    sym: {
        "category": cfg.category,
        "digits": cfg.digits,
        "magic": getattr(cfg, "magic", 0),
    }
    for sym, cfg in SYMBOLS.items()
}
SYMBOL_LIST_JSON = json.dumps(SYMBOL_LIST)
SYMBOL_META_JSON = json.dumps(SYMBOL_META)
STARTING_BAL = STARTING_BALANCE


# ════════════════════════════════════════════════════════════════════════════
#  THE PAGE
# ════════════════════════════════════════════════════════════════════════════
PRO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dragon Trader — PRO</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<!-- Vue 3 + Pinia + Vue Router (CDN, no build) -->
<script src="https://unpkg.com/vue@3.4.27/dist/vue.global.prod.js"></script>
<script src="https://unpkg.com/vue-router@4.3.2/dist/vue-router.global.prod.js"></script>
<!-- Pinia 2.x's IIFE depends on vue-demi being on the global. Without it, Pinia
     fails silently and `Pinia` is left undefined. Load vue-demi BEFORE pinia. -->
<script src="https://unpkg.com/vue-demi@0.14.10/lib/index.iife.js"></script>
<script src="https://unpkg.com/pinia@2.1.7/dist/pinia.iife.prod.js"></script>

<!-- Realtime + charting -->
<script src="https://cdn.jsdelivr.net/npm/socket.io-client@4.7.4/dist/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script src="https://cdn.jsdelivr.net/npm/apexcharts@3.49.1/dist/apexcharts.min.js"></script>

<style>
/* ════════════════════════════════════════════════════════════════════════
   PRO TERMINAL DESIGN SYSTEM
   ════════════════════════════════════════════════════════════════════════ */
:root {
  --bg-0: #07090d;
  --bg-1: #0c1018;
  --bg-2: #131824;
  --bg-3: #1b2231;
  --bg-card: #0f141d;
  --bdr-1: #1f2937;
  --bdr-2: #2c3a52;
  --bdr-3: #3d5172;
  --text-1: #e8edf6;
  --text-2: #a4b0c2;
  --text-3: #5e6b80;

  --accent: #00d8ff;
  --accent-2: #7a8eff;
  --green: #00d68f;
  --green-soft: rgba(0,214,143,0.10);
  --red: #ff4466;
  --red-soft: rgba(255,68,102,0.10);
  --amber: #f0a030;
  --amber-soft: rgba(240,160,48,0.10);
  --violet: #b08cff;

  --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --font-ui:   'Inter', system-ui, -apple-system, sans-serif;

  --shadow-1: 0 1px 0 rgba(255,255,255,0.02), 0 4px 18px rgba(0,0,0,0.35);
  --shadow-glow: 0 0 24px rgba(0,216,255,0.18);

  --r-sm: 4px;
  --r-md: 6px;
  --r-lg: 10px;
}

*,
*::before,
*::after { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  height: 100vh;
  background:
    radial-gradient(circle at 18% -10%, rgba(0,216,255,0.08) 0%, transparent 50%),
    radial-gradient(circle at 90% 110%, rgba(122,142,255,0.06) 0%, transparent 50%),
    var(--bg-0);
  color: var(--text-1);
  font-family: var(--font-ui);
  font-size: 13px;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app { height: 100vh; display: flex; flex-direction: column; }

::selection { background: rgba(0,216,255,0.35); color: var(--text-1); }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bdr-2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--bdr-3); }

button, input, select { font-family: inherit; font-size: inherit; color: inherit; }
button { cursor: pointer; }

.mono { font-family: var(--font-mono); font-feature-settings: 'tnum' 1, 'zero' 1; }

/* ── App chrome ────────────────────────────────────────────────────────── */
.topbar {
  display: flex;
  align-items: center;
  height: 48px;
  padding: 0 14px;
  background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%);
  border-bottom: 1px solid var(--bdr-1);
  gap: 16px;
  flex-shrink: 0;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 800;
  letter-spacing: 0.06em;
}
.brand-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 10px var(--accent), 0 0 18px rgba(0,216,255,0.35);
  animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}
.brand-text { font-size: 13px; }
.brand-text .pro {
  color: var(--accent);
  font-weight: 800;
  margin-left: 4px;
  letter-spacing: 0.16em;
}

.topnav { display: flex; gap: 2px; }
.topnav .tab {
  padding: 7px 14px;
  background: transparent;
  color: var(--text-2);
  border: 1px solid transparent;
  border-radius: var(--r-md);
  font-weight: 500;
  font-size: 12px;
  letter-spacing: 0.02em;
  transition: background .12s ease, color .12s ease, border-color .12s ease;
  display: flex; align-items: center; gap: 8px;
}
.topnav .tab:hover { background: var(--bg-2); color: var(--text-1); }
.topnav .tab.active {
  background: rgba(0,216,255,0.08);
  color: var(--accent);
  border-color: rgba(0,216,255,0.30);
}
.topnav .tab .kbd {
  font-family: var(--font-mono); font-size: 10px;
  padding: 1px 5px; border-radius: 3px;
  background: var(--bg-3); color: var(--text-3);
}

.topbar-spacer { flex: 1; }

.account-stats { display: flex; gap: 14px; align-items: center; }
.acct-stat {
  display: flex; flex-direction: column; align-items: flex-end; gap: 1px;
  padding: 0 10px; border-left: 1px solid var(--bdr-1);
}
.acct-stat:first-child { border-left: none; }
.acct-stat .label {
  font-size: 9.5px; color: var(--text-3);
  text-transform: uppercase; letter-spacing: 0.12em;
  font-weight: 600;
}
.acct-stat .value {
  font-family: var(--font-mono); font-weight: 600; font-size: 13.5px;
  color: var(--text-1);
  transition: color .2s ease;
}
.acct-stat .value.pos { color: var(--green); }
.acct-stat .value.neg { color: var(--red); }

.connection-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 9px; border-radius: 999px;
  font-size: 10.5px; font-weight: 600;
  background: var(--bg-2); color: var(--text-2);
  border: 1px solid var(--bdr-1);
}
.connection-pill .dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--text-3);
}
.connection-pill.ok { color: var(--green); border-color: rgba(0,214,143,0.30); }
.connection-pill.ok .dot { background: var(--green); box-shadow: 0 0 6px var(--green); }
.connection-pill.bad { color: var(--red); border-color: rgba(255,68,102,0.30); }
.connection-pill.bad .dot { background: var(--red); }

/* ── Layout grid for views ─────────────────────────────────────────────── */
.view {
  flex: 1; min-height: 0; min-width: 0;
  display: grid; gap: 1px;
  background: var(--bdr-1);
  overflow: hidden;
}

.panel {
  background: var(--bg-1);
  display: flex; flex-direction: column; min-height: 0; min-width: 0;
  position: relative;
}
.panel-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 12px;
  background: linear-gradient(180deg, var(--bg-2) 0%, var(--bg-1) 100%);
  border-bottom: 1px solid var(--bdr-1);
  flex-shrink: 0;
}
.panel-title {
  font-size: 10.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.10em;
  color: var(--text-2);
  display: flex; align-items: center; gap: 8px;
}
.panel-title .accent-bar {
  width: 3px; height: 12px; border-radius: 2px;
  background: var(--accent);
}
.panel-tools { display: flex; gap: 6px; }
.panel-body { flex: 1; min-height: 0; min-width: 0; overflow: auto; }

/* Card */
.card {
  background: var(--bg-card);
  border: 1px solid var(--bdr-1);
  border-radius: var(--r-md);
  padding: 12px 14px;
}
.card.hover-glow:hover {
  border-color: var(--bdr-2);
  box-shadow: var(--shadow-1);
}

/* Buttons */
.btn {
  background: var(--bg-2);
  color: var(--text-2);
  border: 1px solid var(--bdr-1);
  border-radius: var(--r-md);
  padding: 5px 10px;
  font-size: 11px; font-weight: 500;
  letter-spacing: 0.02em;
  transition: all .12s ease;
}
.btn:hover { background: var(--bg-3); color: var(--text-1); border-color: var(--bdr-2); }
.btn.danger { color: var(--red); border-color: rgba(255,68,102,0.25); }
.btn.danger:hover { background: var(--red-soft); border-color: var(--red); }
.btn.accent { color: var(--accent); border-color: rgba(0,216,255,0.25); }
.btn.accent:hover { background: rgba(0,216,255,0.08); border-color: var(--accent); }
.btn.sm { padding: 3px 7px; font-size: 10.5px; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* Inputs */
.input, select.input {
  background: var(--bg-2);
  border: 1px solid var(--bdr-1);
  border-radius: var(--r-sm);
  padding: 5px 8px; font-size: 11.5px; color: var(--text-1);
  transition: border-color .12s ease;
}
.input:focus { outline: none; border-color: var(--accent); }

/* ── Trading view layout ───────────────────────────────────────────────── */
.trading-view {
  grid-template-columns: 260px 1fr 320px;
  grid-template-rows: 1fr auto;
  grid-template-areas:
    "watchlist chart sidepanel"
    "watchlist positions positions";
}
.area-watchlist  { grid-area: watchlist; }
.area-chart      { grid-area: chart; }
.area-sidepanel  { grid-area: sidepanel; }
.area-positions  { grid-area: positions; }

@media (max-width: 1280px) {
  .trading-view {
    grid-template-columns: 220px 1fr 280px;
  }
}

/* ── Watchlist ─────────────────────────────────────────────────────────── */
.watchlist-toolbar {
  display: flex; gap: 4px; padding: 6px 8px;
  border-bottom: 1px solid var(--bdr-1);
  background: var(--bg-1);
}
.watchlist-toolbar .input { flex: 1; min-width: 0; }
.watchlist-tabs {
  display: flex; gap: 0; padding: 4px 8px 0;
  border-bottom: 1px solid var(--bdr-1);
  background: var(--bg-1); overflow-x: auto;
}
.watchlist-tab {
  padding: 5px 9px; font-size: 10.5px; font-weight: 600;
  background: transparent; color: var(--text-3);
  border: 1px solid transparent;
  border-bottom: 2px solid transparent;
  border-radius: 0;
  text-transform: uppercase; letter-spacing: 0.05em;
  white-space: nowrap;
}
.watchlist-tab:hover { color: var(--text-1); }
.watchlist-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

.symbol-row {
  display: grid;
  grid-template-columns: 1fr auto 60px;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  cursor: pointer;
  border-bottom: 1px solid var(--bdr-1);
  transition: background .12s ease;
  position: relative;
}
.symbol-row:hover { background: var(--bg-2); }
.symbol-row.active {
  background: rgba(0,216,255,0.05);
}
.symbol-row.active::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
  background: var(--accent);
}
.symbol-row .sym {
  font-family: var(--font-mono); font-weight: 600; font-size: 11.5px;
  color: var(--text-1);
}
.symbol-row .sym .cat {
  display: inline-block; font-size: 9px; color: var(--text-3);
  font-family: var(--font-ui); margin-left: 4px;
  letter-spacing: 0.05em; font-weight: 500;
}
.symbol-row .price {
  font-family: var(--font-mono); font-size: 11px; font-weight: 500;
  color: var(--text-2); text-align: right;
}
.symbol-row .change {
  font-family: var(--font-mono); font-size: 10.5px;
  text-align: right; padding: 2px 5px; border-radius: 3px;
}
.symbol-row .change.pos { color: var(--green); background: var(--green-soft); }
.symbol-row .change.neg { color: var(--red); background: var(--red-soft); }
.symbol-row .change.flat { color: var(--text-3); }
.symbol-row .spark { grid-column: 1 / -1; height: 18px; margin-top: 2px; }

@keyframes flashGreen {
  0%   { background-color: rgba(0,214,143,0.30); }
  100% { background-color: transparent; }
}
@keyframes flashRed {
  0%   { background-color: rgba(255,68,102,0.30); }
  100% { background-color: transparent; }
}
.symbol-row.flash-up   .price { animation: flashGreen .6s ease-out; }
.symbol-row.flash-down .price { animation: flashRed   .6s ease-out; }

/* ── Chart pane ────────────────────────────────────────────────────────── */
.chart-toolbar {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 10px;
  background: var(--bg-1);
  border-bottom: 1px solid var(--bdr-1);
  flex-shrink: 0;
}
.chart-toolbar .group {
  display: flex; gap: 0;
  background: var(--bg-2);
  border: 1px solid var(--bdr-1);
  border-radius: var(--r-md);
  overflow: hidden;
}
.chart-toolbar .group .btn {
  border: none; border-radius: 0;
  border-right: 1px solid var(--bdr-1);
  background: transparent;
  font-size: 10.5px; padding: 4px 9px;
}
.chart-toolbar .group .btn:last-child { border-right: none; }
.chart-toolbar .group .btn.active { background: rgba(0,216,255,0.10); color: var(--accent); }
.chart-symbol {
  font-family: var(--font-mono); font-weight: 700; font-size: 14px;
  color: var(--text-1);
  margin-right: 8px;
}
.chart-quote {
  font-family: var(--font-mono); font-size: 12.5px;
  color: var(--text-2); display: flex; gap: 10px;
}
.chart-quote .bid { color: var(--green); }
.chart-quote .ask { color: var(--red); }
.chart-quote .spread { color: var(--text-3); font-size: 10.5px; }

#tv-chart { flex: 1; min-height: 0; position: relative; }

/* ── Side panel ───────────────────────────────────────────────────────── */
.side-section {
  border-bottom: 1px solid var(--bdr-1);
  padding: 10px 12px;
}
.side-section h4 {
  margin: 0 0 8px 0;
  font-size: 9.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.10em;
  color: var(--text-3);
}
.kv-grid {
  display: grid; grid-template-columns: 1fr auto;
  row-gap: 4px; column-gap: 12px;
  font-size: 11.5px;
}
.kv-grid .k { color: var(--text-3); }
.kv-grid .v { font-family: var(--font-mono); color: var(--text-1); text-align: right; }

.dir-bias-pill {
  display: inline-flex; padding: 2px 7px; border-radius: 3px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.05em;
}
.dir-bias-pill.long  { background: var(--green-soft); color: var(--green); }
.dir-bias-pill.short { background: var(--red-soft);   color: var(--red); }
.dir-bias-pill.both  { background: var(--bg-3);       color: var(--text-2); }

.decision-row {
  display: grid;
  grid-template-columns: 60px auto 1fr;
  gap: 8px; padding: 4px 0;
  font-size: 10.5px;
  border-top: 1px solid var(--bdr-1);
}
.decision-row:first-child { border-top: none; }
.decision-row .ts { color: var(--text-3); font-family: var(--font-mono); }
.decision-row .score { font-family: var(--font-mono); font-weight: 600; }
.decision-row .gate { color: var(--text-2); font-size: 10px; }

/* ── Positions table ──────────────────────────────────────────────────── */
.positions {
  background: var(--bg-1);
  border-top: 1px solid var(--bdr-1);
  display: flex; flex-direction: column;
}
.positions.collapsed .positions-body { display: none; }
.positions-header {
  display: flex; align-items: center; gap: 12px;
  padding: 6px 12px;
  background: var(--bg-1);
  border-bottom: 1px solid var(--bdr-1);
  flex-shrink: 0; cursor: pointer;
}
.positions-header h3 {
  margin: 0; font-size: 10.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.10em;
  color: var(--text-2);
}
.positions-header .count-pill {
  background: var(--bg-3); color: var(--text-1);
  padding: 1px 7px; border-radius: 999px; font-size: 10px;
  font-family: var(--font-mono); font-weight: 600;
}
.positions-body {
  flex: 1; min-height: 0; overflow: auto;
  max-height: 220px;
}
.tbl {
  width: 100%; border-collapse: collapse;
  font-size: 11.5px;
}
.tbl thead th {
  position: sticky; top: 0; z-index: 1;
  background: var(--bg-2);
  text-align: left;
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--text-3);
  padding: 7px 10px;
  border-bottom: 1px solid var(--bdr-1);
  cursor: pointer; user-select: none;
}
.tbl thead th .sort-ind { color: var(--accent); margin-left: 4px; font-size: 9px; }
.tbl tbody td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--bdr-1);
  font-family: var(--font-mono);
  color: var(--text-1);
}
.tbl tbody tr:hover { background: var(--bg-2); }
.tbl .side-pill {
  display: inline-block; padding: 1px 6px; border-radius: 3px;
  font-size: 10px; font-weight: 600;
}
.tbl .side-pill.long  { background: var(--green-soft); color: var(--green); }
.tbl .side-pill.short { background: var(--red-soft);   color: var(--red); }
.tbl .pos { color: var(--green); }
.tbl .neg { color: var(--red); }
.tbl .num-bar {
  position: relative;
  display: inline-block;
  padding: 1px 4px;
  border-radius: 3px;
}

/* ── Analytics view ───────────────────────────────────────────────────── */
.analytics-view {
  grid-template-columns: 1fr 280px;
  grid-template-rows: auto 1fr auto;
  grid-template-areas:
    "toolbar toolbar"
    "grid sidebar"
    "alpha sidebar";
  padding: 0;
}
.area-anal-toolbar { grid-area: toolbar; }
.area-anal-grid    { grid-area: grid; }
.area-anal-alpha   { grid-area: alpha; }
.area-anal-sidebar { grid-area: sidebar; }

.anal-toolbar {
  display: flex; gap: 8px; align-items: center;
  padding: 8px 14px;
  background: var(--bg-1);
  border-bottom: 1px solid var(--bdr-1);
}
.window-pills { display: flex; gap: 0; background: var(--bg-2); border: 1px solid var(--bdr-1); border-radius: var(--r-md); overflow: hidden; }
.window-pills .btn {
  border: none; border-radius: 0; border-right: 1px solid var(--bdr-1);
  font-size: 10.5px; padding: 4px 11px;
}
.window-pills .btn:last-child { border-right: none; }
.window-pills .btn.active { background: rgba(0,216,255,0.10); color: var(--accent); }

.grid-2x2 {
  display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
  gap: 1px; background: var(--bdr-1);
  height: 100%; min-height: 0;
}
.grid-2x2 .panel { background: var(--bg-1); }

.heatmap-cell {
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-mono); font-size: 9.5px;
  color: var(--text-2);
  border-radius: 2px;
  cursor: pointer;
  transition: transform .12s ease;
}
.heatmap-cell:hover { transform: scale(1.06); z-index: 2; }
.hm-grid {
  display: grid;
  gap: 2px;
  padding: 8px 14px;
}

/* ── Risk view ────────────────────────────────────────────────────────── */
.risk-view {
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto 1fr;
  grid-template-areas:
    "gauges gauges"
    "exposure direction"
    "corr cluster";
  padding: 0;
}
.area-risk-gauges    { grid-area: gauges; }
.area-risk-exposure  { grid-area: exposure; }
.area-risk-direction { grid-area: direction; }
.area-risk-corr      { grid-area: corr; }
.area-risk-cluster   { grid-area: cluster; }

.gauges-row {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: var(--bdr-1);
}
.gauge {
  background: var(--bg-1);
  padding: 14px 18px;
  display: flex; flex-direction: column; gap: 4px;
}
.gauge .label {
  font-size: 10px; font-weight: 600;
  color: var(--text-3); text-transform: uppercase; letter-spacing: 0.10em;
}
.gauge .value {
  font-family: var(--font-mono); font-size: 22px; font-weight: 700;
  color: var(--text-1);
}
.gauge .value.warn { color: var(--amber); }
.gauge .value.danger { color: var(--red); }
.gauge .bar {
  width: 100%; height: 4px; background: var(--bg-3); border-radius: 2px; overflow: hidden;
  margin-top: 4px;
}
.gauge .bar > div { height: 100%; background: var(--accent); transition: width .3s ease; }

/* Correlation matrix cells */
.corr-grid { padding: 12px; overflow: auto; }
.corr-cell {
  width: 28px; height: 28px;
  display: inline-flex; align-items: center; justify-content: center;
  font-family: var(--font-mono); font-size: 8.5px;
  border-radius: 2px;
  cursor: pointer;
}

/* ── System view ──────────────────────────────────────────────────────── */
.system-view {
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto 1fr;
  grid-template-areas:
    "mt5 ml"
    "rl errors"
    "alerts process";
  padding: 0;
}
.area-sys-mt5     { grid-area: mt5; }
.area-sys-ml      { grid-area: ml; }
.area-sys-rl      { grid-area: rl; }
.area-sys-errors  { grid-area: errors; }
.area-sys-alerts  { grid-area: alerts; }
.area-sys-process { grid-area: process; }

.status-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 1px 7px; border-radius: 3px;
  font-size: 10px; font-weight: 600;
  font-family: var(--font-mono);
}
.status-pill.ok      { background: var(--green-soft); color: var(--green); }
.status-pill.warn    { background: var(--amber-soft); color: var(--amber); }
.status-pill.err     { background: var(--red-soft);   color: var(--red); }
.status-pill.idle    { background: var(--bg-3);       color: var(--text-3); }
.status-pill .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: currentColor;
}
.log-line {
  font-family: var(--font-mono); font-size: 10.5px;
  padding: 3px 12px;
  border-bottom: 1px solid var(--bdr-1);
  display: grid; grid-template-columns: 100px 80px 1fr; gap: 8px;
}
.log-line .ts { color: var(--text-3); }
.log-line .lvl.ERROR { color: var(--red); font-weight: 600; }
.log-line .lvl.WARN  { color: var(--amber); font-weight: 600; }
.log-line .lvl.INFO  { color: var(--accent); font-weight: 600; }
.log-line .msg { color: var(--text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ── Journal view ─────────────────────────────────────────────────────── */
.journal-view {
  grid-template-columns: 1fr 280px;
  grid-template-rows: auto 1fr;
  grid-template-areas:
    "filter filter"
    "table stats";
  padding: 0;
}
.area-journal-filter { grid-area: filter; }
.area-journal-table  { grid-area: table; }
.area-journal-stats  { grid-area: stats; }

.filter-bar {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  padding: 10px 14px;
  background: var(--bg-1);
  border-bottom: 1px solid var(--bdr-1);
}
.filter-bar label {
  display: flex; flex-direction: column; gap: 2px;
}
.filter-bar label span {
  font-size: 9px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--text-3);
}

.expand-row {
  background: var(--bg-2);
}
.expand-row td { padding: 10px 16px !important; }
.expand-row .meta-grid {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.expand-row .meta-cell {
  background: var(--bg-1); border: 1px solid var(--bdr-1);
  border-radius: var(--r-sm); padding: 6px 8px;
}
.expand-row .meta-cell .k { font-size: 9px; color: var(--text-3); text-transform: uppercase; }
.expand-row .meta-cell .v { font-family: var(--font-mono); font-size: 11px; color: var(--text-1); }

/* ── Skeletons ────────────────────────────────────────────────────────── */
.skeleton {
  background: linear-gradient(90deg, var(--bg-2) 0%, var(--bg-3) 50%, var(--bg-2) 100%);
  background-size: 200% 100%;
  animation: shimmer 1.4s ease-in-out infinite;
  border-radius: var(--r-sm);
  height: 16px; min-width: 60px;
}
@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* ── Modal (JARVIS confirm) ───────────────────────────────────────────── */
.modal-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.65);
  z-index: 100;
  display: flex; align-items: center; justify-content: center;
  backdrop-filter: blur(4px);
}
.modal {
  width: 420px;
  background: var(--bg-1);
  border: 1px solid var(--bdr-2);
  border-radius: var(--r-lg);
  padding: 22px 24px;
  box-shadow: var(--shadow-1), var(--shadow-glow);
}
.modal h3 {
  margin: 0 0 8px 0;
  font-size: 14px; font-weight: 700;
  color: var(--text-1);
}
.modal p {
  font-size: 12px; color: var(--text-2);
  margin: 0 0 14px 0; line-height: 1.5;
}
.modal-actions {
  display: flex; justify-content: flex-end; gap: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--bdr-1);
}

.tooltip-box {
  position: fixed; z-index: 200;
  background: var(--bg-3);
  border: 1px solid var(--bdr-2);
  border-radius: var(--r-sm);
  padding: 6px 9px;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-1);
  pointer-events: none;
  box-shadow: var(--shadow-1);
}

/* ── Misc utilities ────────────────────────────────────────────────────── */
.flex { display: flex; }
.flex-col { display: flex; flex-direction: column; }
.gap-1 { gap: 4px; } .gap-2 { gap: 8px; } .gap-3 { gap: 12px; } .gap-4 { gap: 16px; }
.flex-1 { flex: 1; min-width: 0; min-height: 0; }
.text-2 { color: var(--text-2); }
.text-3 { color: var(--text-3); }
.text-pos { color: var(--green); }
.text-neg { color: var(--red); }
.text-warn { color: var(--amber); }
.text-accent { color: var(--accent); }
.tabular { font-variant-numeric: tabular-nums; font-feature-settings: 'tnum' 1; }
.fade-enter-active, .fade-leave-active { transition: opacity .18s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

</style>
</head>
<body>
<div id="app"></div>
<div id="boot-error" style="display:none;position:fixed;top:0;left:0;right:0;background:#ff4466;color:#fff;font-family:monospace;font-size:13px;padding:14px;white-space:pre-wrap;z-index:99999;border-bottom:3px solid #fff;"></div>
<div id="boot-status" style="position:fixed;bottom:8px;right:8px;background:#0f141d;color:#5e6b80;font-family:monospace;font-size:11px;padding:6px 10px;border:1px solid #1f2937;border-radius:4px;z-index:99998;">booting...</div>

<script>
/* ════════════════════════════════════════════════════════════════════════
   ERROR OVERLAY — surfaces runtime errors so the page never silently blanks
   ════════════════════════════════════════════════════════════════════════ */
function _bootStatus(s) { try { document.getElementById('boot-status').textContent = s; } catch(_){} }
function _bootError(msg) {
  try {
    const el = document.getElementById('boot-error');
    el.textContent = (el.textContent ? el.textContent + '\n\n' : '') + msg;
    el.style.display = 'block';
    _bootStatus('ERROR — see top banner');
  } catch (e) { console.error(msg); }
}
window.addEventListener('error', (e) => {
  _bootError('[runtime] ' + (e.error ? (e.error.stack || e.error.message) : (e.message + ' @ ' + e.filename + ':' + e.lineno)));
});
window.addEventListener('unhandledrejection', (e) => {
  _bootError('[promise] ' + (e.reason && e.reason.stack ? e.reason.stack : String(e.reason)));
});

/* Verify CDN globals before destructure (so error message is meaningful) */
_bootStatus('checking CDNs...');
const _missing = [];
if (typeof Vue === 'undefined') _missing.push('Vue');
if (typeof VueRouter === 'undefined') _missing.push('VueRouter');
if (typeof Pinia === 'undefined') _missing.push('Pinia');
if (typeof io === 'undefined') _missing.push('socket.io (io)');
if (typeof LightweightCharts === 'undefined') _missing.push('LightweightCharts');
if (typeof ApexCharts === 'undefined') _missing.push('ApexCharts');
if (_missing.length) {
  _bootError('CDN libs missing: ' + _missing.join(', ') + '\nCheck network / firewall / DNS for unpkg.com, cdn.jsdelivr.net, cdn.socket.io.');
  throw new Error('cdn-missing: ' + _missing.join(','));
}
_bootStatus('CDNs OK, mounting Vue...');

/* ════════════════════════════════════════════════════════════════════════
   BOOTSTRAP DATA (server-injected)
   ════════════════════════════════════════════════════════════════════════ */
const SYMBOL_LIST = __SYMBOL_LIST__;
const SYMBOL_META = __SYMBOL_META__;
const STARTING_BAL = __STARTING_BAL__;

const { createApp, defineComponent, ref, reactive, computed, watch, onMounted, onUnmounted, nextTick, h, shallowRef, toRefs } = Vue;
const { createPinia, defineStore } = Pinia;
const { createRouter, createWebHashHistory, RouterLink, RouterView } = VueRouter;

/* ════════════════════════════════════════════════════════════════════════
   FORMATTERS
   ════════════════════════════════════════════════════════════════════════ */
const fmt = {
  money: (v, dp=2) => {
    if (v === null || v === undefined || isNaN(v)) return '—';
    const sign = v < 0 ? '-' : '';
    const abs = Math.abs(v);
    return sign + '$' + abs.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  },
  pct: (v, dp=2) => {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return (v >= 0 ? '+' : '') + v.toFixed(dp) + '%';
  },
  num: (v, dp=2) => {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  },
  digits: (sym) => SYMBOL_META[sym]?.digits ?? 5,
  price: (v, sym) => {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return v.toFixed(SYMBOL_META[sym]?.digits ?? 5);
  },
  R: (v, dp=2) => {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return (v >= 0 ? '+' : '') + v.toFixed(dp) + 'R';
  },
  ago: (ts) => {
    if (!ts) return '—';
    const ms = Date.now() - new Date(ts).getTime();
    if (ms < 0) return 'now';
    const s = Math.floor(ms / 1000);
    if (s < 60) return s + 's';
    const m = Math.floor(s / 60);
    if (m < 60) return m + 'm';
    const h = Math.floor(m / 60);
    if (h < 24) return h + 'h';
    return Math.floor(h / 24) + 'd';
  },
  duration: (sec) => {
    if (!sec || sec < 0) return '—';
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    if (h > 0) return h + 'h ' + m + 'm';
    return m + 'm';
  },
  ts: (t) => {
    if (!t) return '';
    const d = new Date(t);
    return d.toLocaleTimeString('en-GB', { hour12: false });
  },
  date: (t) => {
    if (!t) return '';
    const d = new Date(t);
    return d.toISOString().slice(0, 10);
  },
};

/* ════════════════════════════════════════════════════════════════════════
   STORES (Pinia)
   ════════════════════════════════════════════════════════════════════════ */

const useAppStore = defineStore('app', {
  state: () => ({
    connected: false,
    starting_balance: STARTING_BAL,
    selected_symbol: SYMBOL_LIST[0],
    selected_tf: 'H1',
    timeframes: ['M1','M5','M15','H1','H4','D1'],
    flash: {},  // { symbol: 'up'|'down'|null }
    last_tick_ts: {},
    confirm: null,  // {message, onConfirm}
    helpOpen: false,
  }),
  actions: {
    setSymbol(sym) {
      this.selected_symbol = sym;
      try { socket.emit('select_symbol', { symbol: sym }); } catch(e){}
    },
    setTF(tf) {
      this.selected_tf = tf;
      try { socket.emit('select_timeframe', { timeframe: tf }); } catch(e){}
    },
    flashSymbol(sym, dir) {
      this.flash[sym] = dir;
      setTimeout(() => { if (this.flash[sym] === dir) this.flash[sym] = null; }, 600);
    },
    askConfirm(msg, onConfirm) {
      this.confirm = { message: msg, onConfirm };
    },
    closeConfirm() { this.confirm = null; },
  },
});

const usePortfolioStore = defineStore('portfolio', {
  state: () => ({
    balance: 0,
    equity: 0,
    open_pnl: 0,
    daily_pnl: 0,
    drawdown_pct: 0,
    peak_equity: STARTING_BAL,
    positions: [],
    loading: true,
  }),
  getters: {
    pnlPct: (s) => s.starting_balance > 0 ? ((s.equity - STARTING_BAL) / STARTING_BAL) * 100 : 0,
    positionCount: (s) => s.positions.length,
  },
  actions: {
    async fetch() {
      try {
        const r = await fetch('/api/v2/portfolio');
        if (!r.ok) throw new Error(r.status);
        const d = await r.json();
        this.applyUpdate(d);
        this.loading = false;
      } catch (e) {
        console.warn('portfolio fetch fail', e);
      }
    },
    applyUpdate(d) {
      if (d.balance !== undefined) this.balance = d.balance;
      if (d.equity !== undefined) this.equity = d.equity;
      if (d.open_pnl !== undefined) this.open_pnl = d.open_pnl;
      if (d.daily_pnl !== undefined) this.daily_pnl = d.daily_pnl;
      if (d.drawdown_pct !== undefined) this.drawdown_pct = d.drawdown_pct;
      if (d.peak_equity !== undefined) this.peak_equity = d.peak_equity;
      if (d.positions !== undefined) this.positions = d.positions;
    },
  },
});

const useSymbolsStore = defineStore('symbols', {
  state: () => ({
    list: SYMBOL_LIST.map(s => ({
      symbol: s,
      category: SYMBOL_META[s]?.category || 'Misc',
      digits: SYMBOL_META[s]?.digits || 5,
      bid: 0, ask: 0, last: 0, prev: 0, change_pct: 0,
      ml_enabled: false, sparkline: [],
    })),
    config: {},
    rl_state: {},
    loading: true,
  }),
  getters: {
    bySymbol: (s) => Object.fromEntries(s.list.map(r => [r.symbol, r])),
  },
  actions: {
    async fetch() {
      try {
        const [r1, r2] = await Promise.all([
          fetch('/api/v2/symbols').then(r => r.ok ? r.json() : {}),
          fetch('/api/v2/rl_state').then(r => r.ok ? r.json() : {}),
        ]);
        if (r1 && r1.symbols) {
          for (const sym of r1.symbols) {
            const ix = this.list.findIndex(x => x.symbol === sym.symbol);
            if (ix >= 0) Object.assign(this.list[ix], sym);
            this.config[sym.symbol] = sym;
          }
        }
        this.rl_state = r2 || {};
        this.loading = false;
      } catch (e) {
        console.warn('symbols fetch fail', e);
      }
    },
    applyTicks(ticks) {
      const app = useAppStore();
      for (const t of ticks) {
        const r = this.list.find(x => x.symbol === t.symbol);
        if (!r) continue;
        const prev = r.bid;
        if (t.bid !== undefined) r.bid = t.bid;
        if (t.ask !== undefined) r.ask = t.ask;
        if (t.last !== undefined) r.last = t.last;
        if (t.prev !== undefined) r.prev = t.prev;
        if (t.change_pct !== undefined) r.change_pct = t.change_pct;
        if (Array.isArray(t.spark)) r.sparkline = t.spark.slice(-20);
        if (prev > 0 && t.bid && Math.abs(t.bid - prev) > 1e-9) {
          app.flashSymbol(t.symbol, t.bid > prev ? 'up' : 'down');
        }
        app.last_tick_ts[t.symbol] = Date.now();
      }
    },
  },
});

const useDecisionsStore = defineStore('decisions', {
  state: () => ({
    recent: [], // { ts, symbol, side, score, gate, reason }
  }),
  actions: {
    push(d) {
      this.recent.unshift({ ts: Date.now(), ...d });
      if (this.recent.length > 200) this.recent.length = 200;
    },
    forSymbol(sym, n = 10) {
      return this.recent.filter(r => r.symbol === sym).slice(0, n);
    },
    async fetchInitial() {
      try {
        const r = await fetch('/api/v2/decisions/recent?limit=200');
        if (!r.ok) return;
        const d = await r.json();
        this.recent = (d.decisions || []).map(x => ({...x, ts: x.ts || Date.now()}));
      } catch (e) { /* ignore */ }
    },
  },
});

const useAlertsStore = defineStore('alerts', {
  state: () => ({ feed: [] }),
  actions: {
    push(a) {
      this.feed.unshift({ ts: Date.now(), ...a });
      if (this.feed.length > 100) this.feed.length = 100;
    },
  },
});

/* ════════════════════════════════════════════════════════════════════════
   SOCKET.IO
   ════════════════════════════════════════════════════════════════════════ */
let socket;
function setupSocket(stores) {
  socket = io({ transports: ['websocket','polling'] });

  socket.on('connect',    () => stores.app.connected = true);
  socket.on('disconnect', () => stores.app.connected = false);

  // Throttle bulk tick re-renders to ≥100ms cadence
  let tickQueue = [];
  let tickTimer = null;
  const flushTicks = () => {
    if (tickQueue.length) {
      stores.symbols.applyTicks(tickQueue);
      tickQueue = [];
    }
    tickTimer = null;
  };
  socket.on('ticks:bulk', (data) => {
    const arr = Array.isArray(data) ? data : (data?.ticks || []);
    tickQueue.push(...arr);
    if (!tickTimer) tickTimer = setTimeout(flushTicks, 100);
  });
  // backward-compat with existing tick_update
  socket.on('tick_update', (data) => {
    const arr = Array.isArray(data) ? data : Object.entries(data).map(([s, v]) => ({ symbol: s, ...v }));
    tickQueue.push(...arr);
    if (!tickTimer) tickTimer = setTimeout(flushTicks, 100);
  });

  socket.on('portfolio:update', (d) => stores.portfolio.applyUpdate(d));
  socket.on('stats_update',     (d) => stores.portfolio.applyUpdate(d));

  socket.on('signal:scored', (d) => stores.decisions.push(d));
  socket.on('alert:new',     (a) => stores.alerts.push(a));
  socket.on('position:opened', (p) => {
    stores.alerts.push({ kind: 'open', symbol: p.symbol, message: 'Opened ' + p.symbol + ' ' + p.side });
  });
  socket.on('position:closed', (p) => {
    stores.alerts.push({ kind: 'close', symbol: p.symbol, message: 'Closed ' + p.symbol + ' P&L ' + fmt.money(p.profit) });
  });
}

/* ════════════════════════════════════════════════════════════════════════
   MINI SPARKLINE (canvas — fast, no DOM)
   ════════════════════════════════════════════════════════════════════════ */
const Sparkline = defineComponent({
  props: { data: Array, color: { type: String, default: '#7a8eff' }, height: { type: Number, default: 18 } },
  setup(props) {
    const canvas = ref(null);
    const draw = () => {
      const c = canvas.value;
      if (!c) return;
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth || 60;
      const h = props.height;
      c.width = w * dpr; c.height = h * dpr;
      const ctx = c.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);
      const d = props.data || [];
      if (d.length < 2) return;
      const min = Math.min(...d), max = Math.max(...d);
      const range = max - min || 1;
      const step = w / (d.length - 1);
      const trend = d[d.length-1] >= d[0];
      ctx.strokeStyle = trend ? '#00d68f' : '#ff4466';
      ctx.lineWidth = 1;
      ctx.beginPath();
      d.forEach((v, i) => {
        const x = i * step;
        const y = h - ((v - min) / range) * (h - 2) - 1;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };
    onMounted(draw);
    watch(() => props.data, draw, { deep: true });
    return () => h('canvas', { ref: canvas, style: { width: '100%', height: props.height + 'px', display: 'block' } });
  },
});

/* ════════════════════════════════════════════════════════════════════════
   TRADING VIEW
   ════════════════════════════════════════════════════════════════════════ */
const TradingView = defineComponent({
  components: { Sparkline },
  setup() {
    const app = useAppStore();
    const portfolio = usePortfolioStore();
    const symbols = useSymbolsStore();
    const decisions = useDecisionsStore();

    const watchlistSearch = ref('');
    const watchlistCat = ref('All');
    const watchlistSort = ref('alpha'); // alpha | change | volatility
    const showIndicators = reactive({ ema: true, bb: false, atr: false });
    const positionsCollapsed = ref(false);
    const posSort = ref({ key: 'profit', dir: -1 });

    const cats = computed(() => {
      const c = new Set(['All']);
      symbols.list.forEach(s => c.add(s.category));
      c.add('ML');
      return [...c];
    });

    const filteredSymbols = computed(() => {
      let arr = symbols.list.slice();
      if (watchlistCat.value === 'ML') {
        arr = arr.filter(s => s.ml_enabled);
      } else if (watchlistCat.value !== 'All') {
        arr = arr.filter(s => s.category === watchlistCat.value);
      }
      if (watchlistSearch.value.trim()) {
        const q = watchlistSearch.value.toLowerCase();
        arr = arr.filter(s => s.symbol.toLowerCase().includes(q));
      }
      const k = watchlistSort.value;
      arr.sort((a,b) => {
        if (k === 'alpha') return a.symbol.localeCompare(b.symbol);
        if (k === 'change') return (b.change_pct || 0) - (a.change_pct || 0);
        if (k === 'volatility') {
          const va = a.sparkline.length ? Math.max(...a.sparkline) - Math.min(...a.sparkline) : 0;
          const vb = b.sparkline.length ? Math.max(...b.sparkline) - Math.min(...b.sparkline) : 0;
          return vb - va;
        }
        return 0;
      });
      return arr;
    });

    const selectedConfig = computed(() => symbols.config[app.selected_symbol] || {});
    const selectedRl = computed(() => symbols.rl_state[app.selected_symbol] || {});
    const selectedTicker = computed(() => symbols.bySymbol[app.selected_symbol] || {});
    const recentDecisions = computed(() => decisions.forSymbol(app.selected_symbol, 10));

    /* ── Lightweight chart ─────────────────────────────────────────────── */
    const chartRef = ref(null);
    let chart = null, candleSeries = null, lineSeriesEMA15 = null, lineSeriesEMA40 = null, lineSeriesEMA80 = null, bbUpper = null, bbLower = null;
    let resizeObserver = null;

    const buildChart = () => {
      if (!chartRef.value) return;
      if (chart) { try { chart.remove(); } catch(e){} chart = null; }
      chart = LightweightCharts.createChart(chartRef.value, {
        layout: { background: { color: '#0c1018' }, textColor: '#a4b0c2', fontFamily: 'JetBrains Mono' },
        grid: { vertLines: { color: '#131824' }, horzLines: { color: '#131824' } },
        rightPriceScale: { borderColor: '#1f2937' },
        timeScale: { borderColor: '#1f2937', timeVisible: true, secondsVisible: false },
        crosshair: { mode: 1 },
        autoSize: false,
      });
      candleSeries = chart.addCandlestickSeries({
        upColor: '#00d68f', downColor: '#ff4466',
        borderUpColor: '#00d68f', borderDownColor: '#ff4466',
        wickUpColor: '#00d68f', wickDownColor: '#ff4466',
      });
      lineSeriesEMA15 = chart.addLineSeries({ color: '#7a8eff', lineWidth: 1, priceLineVisible: false });
      lineSeriesEMA40 = chart.addLineSeries({ color: '#f0a030', lineWidth: 1, priceLineVisible: false });
      lineSeriesEMA80 = chart.addLineSeries({ color: '#b08cff', lineWidth: 1, priceLineVisible: false });
      bbUpper = chart.addLineSeries({ color: 'rgba(0,216,255,0.4)', lineWidth: 1, priceLineVisible: false });
      bbLower = chart.addLineSeries({ color: 'rgba(0,216,255,0.4)', lineWidth: 1, priceLineVisible: false });
      const resize = () => {
        if (chart && chartRef.value) {
          chart.resize(chartRef.value.clientWidth, chartRef.value.clientHeight);
        }
      };
      resizeObserver = new ResizeObserver(resize);
      resizeObserver.observe(chartRef.value);
      resize();
    };

    const computeEMA = (data, period) => {
      const k = 2 / (period + 1);
      const out = []; let prev = null;
      for (const c of data) {
        prev = prev === null ? c.close : c.close * k + prev * (1 - k);
        out.push({ time: c.time, value: prev });
      }
      return out;
    };
    const computeBB = (data, period=20, mult=2) => {
      const upper = [], lower = [];
      for (let i = 0; i < data.length; i++) {
        if (i < period - 1) continue;
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) sum += data[j].close;
        const mean = sum / period;
        let v = 0;
        for (let j = i - period + 1; j <= i; j++) v += (data[j].close - mean) ** 2;
        const sd = Math.sqrt(v / period);
        upper.push({ time: data[i].time, value: mean + mult * sd });
        lower.push({ time: data[i].time, value: mean - mult * sd });
      }
      return { upper, lower };
    };

    const loadCandles = async () => {
      if (!chart) return;
      const sym = app.selected_symbol;
      const tf = app.selected_tf;
      try {
        const r = await fetch(`/api/v2/candles?symbol=${encodeURIComponent(sym)}&tf=${encodeURIComponent(tf)}&limit=400`);
        let candles = [];
        if (r.ok) {
          const d = await r.json();
          candles = d.candles || [];
        }
        if (candles.length === 0) {
          // Backend not yet wired — fall back to /api/data shape if available
          try {
            const r2 = await fetch('/api/data');
            if (r2.ok) {
              const dd = await r2.json();
              if (dd.charts && dd.charts[sym] && dd.charts[sym][tf]) {
                candles = dd.charts[sym][tf];
              }
            }
          } catch(e){}
        }
        candles = candles.map(c => ({
          time: typeof c.time === 'string' ? Math.floor(new Date(c.time).getTime()/1000) : c.time,
          open: +c.open, high: +c.high, low: +c.low, close: +c.close,
        }));
        candleSeries.setData(candles);
        if (showIndicators.ema) {
          lineSeriesEMA15.setData(computeEMA(candles, 15));
          lineSeriesEMA40.setData(computeEMA(candles, 40));
          lineSeriesEMA80.setData(computeEMA(candles, 80));
        } else {
          lineSeriesEMA15.setData([]); lineSeriesEMA40.setData([]); lineSeriesEMA80.setData([]);
        }
        if (showIndicators.bb) {
          const bb = computeBB(candles);
          bbUpper.setData(bb.upper); bbLower.setData(bb.lower);
        } else {
          bbUpper.setData([]); bbLower.setData([]);
        }
        // Markers from journal
        try {
          const j = await fetch(`/api/v2/journal?symbol=${encodeURIComponent(sym)}&since_days=30&limit=100`);
          if (j.ok) {
            const jd = await j.json();
            const markers = (jd.trades || []).map(t => ({
              time: Math.floor(new Date(t.entry_ts).getTime()/1000),
              position: t.side === 'LONG' ? 'belowBar' : 'aboveBar',
              color:    t.side === 'LONG' ? '#00d68f' : '#ff4466',
              shape:    t.side === 'LONG' ? 'arrowUp' : 'arrowDown',
              text: (t.profit > 0 ? '+' : '') + (t.profit_r || 0).toFixed(1) + 'R',
            }));
            candleSeries.setMarkers(markers);
          }
        } catch(e){ /* journal optional */ }
      } catch (e) {
        console.warn('candles fail', e);
      }
    };

    onMounted(async () => {
      await nextTick();
      buildChart();
      loadCandles();
    });

    onUnmounted(() => {
      if (resizeObserver) resizeObserver.disconnect();
      if (chart) try { chart.remove(); } catch(e){}
    });

    watch(() => [app.selected_symbol, app.selected_tf], loadCandles);
    watch(() => [showIndicators.ema, showIndicators.bb], loadCandles);

    /* ── Position table ────────────────────────────────────────────────── */
    const sortedPositions = computed(() => {
      const arr = portfolio.positions.slice();
      const k = posSort.value.key, d = posSort.value.dir;
      arr.sort((a,b) => {
        const va = a[k], vb = b[k];
        if (typeof va === 'number') return (va - vb) * d;
        return String(va || '').localeCompare(String(vb || '')) * d;
      });
      return arr;
    });
    const togglePosSort = (k) => {
      if (posSort.value.key === k) posSort.value.dir *= -1;
      else posSort.value = { key: k, dir: -1 };
    };

    const closePos = (p) => {
      app.askConfirm(
        `Close ${p.symbol} ${p.side}? Current P&L: ${fmt.money(p.profit)}`,
        () => {
          fetch('/api/close_symbol', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: p.symbol })
          }).catch(e=>console.warn(e));
          app.closeConfirm();
        }
      );
    };
    const closeAll = () => {
      app.askConfirm('Close ALL positions?', () => {
        fetch('/api/close_all', { method: 'POST' }).catch(e=>console.warn(e));
        app.closeConfirm();
      });
    };
    const closeLosing = () => {
      app.askConfirm('Close all losing positions?', () => {
        socket.emit('close_losing');
        app.closeConfirm();
      });
    };

    return {
      app, portfolio, symbols, decisions,
      watchlistSearch, watchlistCat, watchlistSort, cats,
      filteredSymbols, selectedConfig, selectedRl, selectedTicker, recentDecisions,
      showIndicators, chartRef, positionsCollapsed,
      sortedPositions, togglePosSort, posSort,
      closePos, closeAll, closeLosing,
      fmt, SYMBOL_META,
    };
  },
  template: `
  <div class="view trading-view">
    <!-- ── WATCHLIST ── -->
    <div class="panel area-watchlist">
      <div class="panel-header">
        <div class="panel-title"><span class="accent-bar"></span>Watchlist</div>
        <span class="text-3 mono" style="font-size:10px;">{{ filteredSymbols.length }}/{{ symbols.list.length }}</span>
      </div>
      <div class="watchlist-toolbar">
        <input class="input" v-model="watchlistSearch" placeholder="Search">
        <select class="input" v-model="watchlistSort" style="max-width: 90px;">
          <option value="alpha">A-Z</option>
          <option value="change">Chg %</option>
          <option value="volatility">Vol</option>
        </select>
      </div>
      <div class="watchlist-tabs">
        <button v-for="c in cats" :key="c" class="watchlist-tab" :class="{active: watchlistCat === c}" @click="watchlistCat = c">{{ c }}</button>
      </div>
      <div class="panel-body">
        <div v-for="s in filteredSymbols" :key="s.symbol"
             class="symbol-row"
             :class="{ active: app.selected_symbol === s.symbol, 'flash-up': app.flash[s.symbol]==='up', 'flash-down': app.flash[s.symbol]==='down' }"
             @click="app.setSymbol(s.symbol)">
          <div class="sym">{{ s.symbol }}<span class="cat">{{ s.category }}</span></div>
          <div class="price">{{ fmt.price(s.bid, s.symbol) }}</div>
          <div class="change" :class="{ pos: s.change_pct > 0, neg: s.change_pct < 0, flat: !s.change_pct }">
            {{ s.change_pct ? fmt.pct(s.change_pct) : '0.00%' }}
          </div>
          <div class="spark"><Sparkline :data="s.sparkline"></Sparkline></div>
        </div>
      </div>
    </div>

    <!-- ── CHART ── -->
    <div class="panel area-chart">
      <div class="chart-toolbar">
        <span class="chart-symbol">{{ app.selected_symbol }}</span>
        <span class="chart-quote">
          <span class="bid">B {{ fmt.price(selectedTicker.bid, app.selected_symbol) }}</span>
          <span class="ask">A {{ fmt.price(selectedTicker.ask, app.selected_symbol) }}</span>
          <span class="spread">spread {{ ((selectedTicker.ask - selectedTicker.bid) || 0).toFixed(SYMBOL_META[app.selected_symbol]?.digits || 5) }}</span>
        </span>
        <span style="flex:1;"></span>
        <div class="group">
          <button v-for="t in app.timeframes" :key="t" class="btn" :class="{active: app.selected_tf === t}" @click="app.setTF(t)">{{ t }}</button>
        </div>
        <div class="group">
          <button class="btn" :class="{active: showIndicators.ema}" @click="showIndicators.ema = !showIndicators.ema">EMA</button>
          <button class="btn" :class="{active: showIndicators.bb}"  @click="showIndicators.bb  = !showIndicators.bb">BB</button>
          <button class="btn" :class="{active: showIndicators.atr}" @click="showIndicators.atr = !showIndicators.atr">ATR</button>
        </div>
      </div>
      <div id="tv-chart" ref="chartRef"></div>
    </div>

    <!-- ── SIDE PANEL ── -->
    <div class="panel area-sidepanel">
      <div class="panel-header">
        <div class="panel-title"><span class="accent-bar"></span>{{ app.selected_symbol }}</div>
      </div>
      <div class="panel-body">
        <div class="side-section">
          <h4>Live Quote</h4>
          <div class="kv-grid">
            <div class="k">Bid</div><div class="v text-pos">{{ fmt.price(selectedTicker.bid, app.selected_symbol) }}</div>
            <div class="k">Ask</div><div class="v text-neg">{{ fmt.price(selectedTicker.ask, app.selected_symbol) }}</div>
            <div class="k">Spread</div><div class="v">{{ ((selectedTicker.ask - selectedTicker.bid) || 0).toFixed(SYMBOL_META[app.selected_symbol]?.digits || 5) }}</div>
            <div class="k">Change</div>
            <div class="v" :class="{ 'text-pos': selectedTicker.change_pct > 0, 'text-neg': selectedTicker.change_pct < 0 }">
              {{ fmt.pct(selectedTicker.change_pct || 0) }}
            </div>
          </div>
        </div>

        <div class="side-section">
          <h4>Tuned Config</h4>
          <div class="kv-grid">
            <div class="k">SL × ATR</div>      <div class="v">{{ selectedConfig.sl_mult ?? '—' }}</div>
            <div class="k">MQ Trend</div>      <div class="v">{{ selectedConfig.mq_trending ?? '—' }}</div>
            <div class="k">MQ Range</div>      <div class="v">{{ selectedConfig.mq_ranging ?? '—' }}</div>
            <div class="k">MQ Volatile</div>   <div class="v">{{ selectedConfig.mq_volatile ?? '—' }}</div>
            <div class="k">MQ LowVol</div>     <div class="v">{{ selectedConfig.mq_low_vol ?? '—' }}</div>
            <div class="k">Risk Cap %</div>    <div class="v">{{ selectedConfig.risk_cap_pct ?? '—' }}</div>
            <div class="k">Direction</div>
            <div class="v">
              <span class="dir-bias-pill" :class="(selectedConfig.direction_bias || 'both').toLowerCase()">
                {{ selectedConfig.direction_bias || 'BOTH' }}
              </span>
            </div>
            <div class="k">Toxic hrs</div>     <div class="v" style="font-size:10px;">{{ (selectedConfig.toxic_hours || []).join(', ') || 'none' }}</div>
            <div class="k">Trail</div>         <div class="v">{{ selectedConfig.trail_profile || 'standard' }}</div>
          </div>
          <div v-if="(selectedConfig.trail_steps || []).length" style="margin-top:6px; font-size:10px; color: var(--text-3); font-family: var(--font-mono);">
            <div v-for="(step, i) in (selectedConfig.trail_steps || []).slice(0,3)" :key="i">
              R≥{{ step.r }} → SL@{{ step.sl }}R · {{ step.note || '' }}
            </div>
          </div>
        </div>

        <div class="side-section">
          <h4>RL Trail State</h4>
          <div class="kv-grid">
            <div class="k">lock×</div>  <div class="v text-accent">{{ (selectedRl.lock_mult ?? 1).toFixed(2) }}</div>
            <div class="k">be×</div>    <div class="v text-accent">{{ (selectedRl.be_mult ?? 1).toFixed(2) }}</div>
            <div class="k">tight×</div> <div class="v text-accent">{{ (selectedRl.tight_mult ?? 1).toFixed(2) }}</div>
            <div class="k">trades</div> <div class="v">{{ selectedRl.n_trades ?? 0 }}</div>
          </div>
        </div>

        <div class="side-section">
          <h4>ML Model</h4>
          <div class="kv-grid">
            <div class="k">Status</div>
            <div class="v">
              <span class="status-pill" :class="selectedConfig.ml_enabled ? 'ok' : 'idle'">
                <span class="dot"></span>{{ selectedConfig.ml_enabled ? 'ON' : 'OFF' }}
              </span>
            </div>
            <div class="k">AUC</div>     <div class="v">{{ selectedConfig.ml_auc ? selectedConfig.ml_auc.toFixed(3) : '—' }}</div>
            <div class="k">N trees</div> <div class="v">{{ selectedConfig.ml_n_trees ?? '—' }}</div>
          </div>
        </div>

        <div class="side-section">
          <h4>Recent Decisions</h4>
          <div v-if="!recentDecisions.length" class="text-3" style="font-size: 10.5px;">No decisions yet</div>
          <div v-for="(d,i) in recentDecisions" :key="i" class="decision-row">
            <div class="ts">{{ fmt.ts(d.ts) }}</div>
            <div class="score" :class="{ 'text-pos': d.side === 'LONG', 'text-neg': d.side === 'SHORT' }">
              {{ (d.side || '?').slice(0,1) }}{{ (d.score ?? 0).toFixed(1) }}
            </div>
            <div class="gate">{{ d.gate || d.reason || '—' }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- ── POSITIONS ── -->
    <div class="panel area-positions positions" :class="{collapsed: positionsCollapsed}">
      <div class="positions-header" @click="positionsCollapsed = !positionsCollapsed">
        <h3>Open Positions</h3>
        <span class="count-pill">{{ portfolio.positions.length }}</span>
        <span class="text-3 mono" style="font-size:11px;">
          Total P&L
          <span :class="{ 'text-pos': portfolio.open_pnl > 0, 'text-neg': portfolio.open_pnl < 0 }">
            {{ fmt.money(portfolio.open_pnl) }}
          </span>
        </span>
        <span style="flex:1;"></span>
        <button class="btn sm" @click.stop="closeLosing">Close Losing</button>
        <button class="btn sm danger" @click.stop="closeAll">Close All</button>
        <span class="text-3" style="font-size:10px;">{{ positionsCollapsed ? '▼' : '▲' }}</span>
      </div>
      <div class="positions-body">
        <table class="tbl">
          <thead>
            <tr>
              <th @click="togglePosSort('symbol')">Symbol<span class="sort-ind" v-if="posSort.key==='symbol'">{{ posSort.dir>0?'▲':'▼' }}</span></th>
              <th @click="togglePosSort('side')">Side</th>
              <th @click="togglePosSort('volume')">Size</th>
              <th @click="togglePosSort('entry')">Entry</th>
              <th @click="togglePosSort('current')">Current</th>
              <th @click="togglePosSort('profit')">P&L<span class="sort-ind" v-if="posSort.key==='profit'">{{ posSort.dir>0?'▲':'▼' }}</span></th>
              <th @click="togglePosSort('profit_r')">R</th>
              <th @click="togglePosSort('duration')">Dur</th>
              <th>SL</th>
              <th>TP</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="!sortedPositions.length"><td colspan="11" class="text-3" style="text-align:center;padding:18px;">No open positions</td></tr>
            <tr v-for="p in sortedPositions" :key="p.ticket || p.symbol">
              <td>{{ p.symbol }}</td>
              <td><span class="side-pill" :class="p.side?.toLowerCase()">{{ p.side }}</span></td>
              <td>{{ fmt.num(p.volume, 2) }}</td>
              <td>{{ fmt.price(p.entry, p.symbol) }}</td>
              <td>{{ fmt.price(p.current, p.symbol) }}</td>
              <td :class="{ pos: p.profit > 0, neg: p.profit < 0 }">{{ fmt.money(p.profit) }}</td>
              <td :class="{ pos: p.profit_r > 0, neg: p.profit_r < 0 }">{{ fmt.R(p.profit_r) }}</td>
              <td>{{ fmt.duration(p.duration) }}</td>
              <td>{{ fmt.price(p.sl, p.symbol) }}</td>
              <td>{{ fmt.price(p.tp, p.symbol) }}</td>
              <td><button class="btn sm danger" @click="closePos(p)">×</button></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
  `,
});

/* ════════════════════════════════════════════════════════════════════════
   ANALYTICS VIEW
   ════════════════════════════════════════════════════════════════════════ */
const AnalyticsView = defineComponent({
  setup() {
    const window_ = ref('30');
    const stats = ref({});
    const attrib = ref([]);
    const winners = ref([]);
    const losers  = ref([]);
    const loading = ref(true);

    const equityRef = ref(null), ddRef = ref(null), bySymRef = ref(null), heatRef = ref(null), alphaRef = ref(null);
    let charts = {};

    const fetchAll = async () => {
      loading.value = true;
      try {
        const [s, a] = await Promise.all([
          fetch(`/api/v2/journal/stats?days=${window_.value}`).then(r => r.ok ? r.json() : {}),
          fetch(`/api/v2/alpha_attribution?days=${window_.value}`).then(r => r.ok ? r.json() : {}),
        ]);
        stats.value = s || {};
        attrib.value = a?.components || [];
        const ps = stats.value.per_symbol || {};
        const sorted = Object.entries(ps).map(([k,v]) => ({symbol:k, ...v})).sort((a,b)=>b.pnl-a.pnl);
        winners.value = sorted.filter(x=>x.pnl>0).slice(0,5);
        losers.value  = sorted.filter(x=>x.pnl<0).slice(-5).reverse();
        renderCharts();
      } catch (e) { console.warn(e); }
      loading.value = false;
    };

    const renderCharts = () => {
      Object.values(charts).forEach(c => { try { c.destroy(); } catch(e){} });
      charts = {};

      const cmnOpts = {
        chart: { background: 'transparent', foreColor: '#a4b0c2', toolbar: { show: false }, zoom: { enabled: false }, animations: { enabled: false } },
        grid: { borderColor: '#1f2937', strokeDashArray: 2 },
        tooltip: { theme: 'dark' },
        dataLabels: { enabled: false },
      };

      // Equity
      if (equityRef.value && stats.value.equity_curve) {
        charts.eq = new ApexCharts(equityRef.value, {
          ...cmnOpts,
          chart: { ...cmnOpts.chart, type: 'area', height: '100%' },
          series: [{ name: 'Equity', data: stats.value.equity_curve.map(p => [new Date(p.ts).getTime(), p.equity]) }],
          stroke: { curve: 'smooth', width: 2, colors: ['#00d8ff'] },
          fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.3, opacityTo: 0, stops: [0, 100] }, colors: ['#00d8ff'] },
          xaxis: { type: 'datetime', labels: { style: { colors: '#5e6b80' } } },
          yaxis: { labels: { formatter: v => '$' + v.toLocaleString(), style: { colors: '#5e6b80' } } },
        });
        charts.eq.render();
      }

      // Drawdown
      if (ddRef.value && stats.value.dd_curve) {
        charts.dd = new ApexCharts(ddRef.value, {
          ...cmnOpts,
          chart: { ...cmnOpts.chart, type: 'area', height: '100%' },
          series: [{ name: 'DD %', data: stats.value.dd_curve.map(p => [new Date(p.ts).getTime(), -Math.abs(p.dd_pct)]) }],
          stroke: { curve: 'stepline', width: 2, colors: ['#ff4466'] },
          fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.5, opacityTo: 0 }, colors: ['#ff4466'] },
          xaxis: { type: 'datetime', labels: { style: { colors: '#5e6b80' } } },
          yaxis: { labels: { formatter: v => v.toFixed(1) + '%', style: { colors: '#5e6b80' } } },
        });
        charts.dd.render();
      }

      // Per-symbol PnL
      if (bySymRef.value && stats.value.per_symbol) {
        const arr = Object.entries(stats.value.per_symbol).map(([k,v])=>({x:k, y:v.pnl||0})).sort((a,b)=>b.y-a.y);
        charts.bs = new ApexCharts(bySymRef.value, {
          ...cmnOpts,
          chart: { ...cmnOpts.chart, type: 'bar', height: '100%' },
          series: [{ name: 'PnL', data: arr }],
          plotOptions: { bar: { distributed: true, columnWidth: '70%', borderRadius: 2 } },
          colors: arr.map(x => x.y >= 0 ? '#00d68f' : '#ff4466'),
          xaxis: { labels: { style: { colors: '#5e6b80', fontSize: '9px' }, rotate: -55 } },
          yaxis: { labels: { formatter: v => '$' + v.toFixed(0), style: { colors: '#5e6b80' } } },
          legend: { show: false },
        });
        charts.bs.render();
      }

      // Hour×DOW heatmap (manual SVG-ish via DOM)
      if (heatRef.value && stats.value.by_hour_dow) {
        heatRef.value.innerHTML = '';
        const grid = document.createElement('div');
        grid.className = 'hm-grid';
        grid.style.gridTemplateColumns = 'auto repeat(24, 1fr)';
        // Header
        grid.appendChild(Object.assign(document.createElement('div'), {textContent: ''}));
        for (let h = 0; h < 24; h++) {
          const c = document.createElement('div');
          c.style.fontSize = '8.5px'; c.style.color = '#5e6b80';
          c.style.textAlign = 'center';
          c.textContent = h;
          grid.appendChild(c);
        }
        const dows = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
        const data = stats.value.by_hour_dow || {};
        // Find max for scaling
        let maxAbs = 0;
        for (let d=0; d<7; d++) for (let h=0; h<24; h++) {
          const v = (data[d] && data[d][h]) ? data[d][h].pnl : 0;
          if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
        }
        for (let d = 0; d < 7; d++) {
          const lbl = document.createElement('div');
          lbl.style.fontSize = '9.5px'; lbl.style.color = '#a4b0c2';
          lbl.style.alignSelf = 'center'; lbl.style.padding = '0 6px 0 0';
          lbl.textContent = dows[d];
          grid.appendChild(lbl);
          for (let h = 0; h < 24; h++) {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            cell.style.height = '22px';
            const v = (data[d] && data[d][h]) ? data[d][h].pnl : 0;
            const n = (data[d] && data[d][h]) ? data[d][h].n : 0;
            const intensity = maxAbs > 0 ? Math.min(1, Math.abs(v)/maxAbs) : 0;
            const col = v >= 0 ? `rgba(0,214,143,${0.10 + intensity*0.7})` : `rgba(255,68,102,${0.10 + intensity*0.7})`;
            cell.style.background = col;
            cell.title = `${dows[d]} ${h}:00 — PnL $${v.toFixed(2)} · ${n} trades`;
            grid.appendChild(cell);
          }
        }
        heatRef.value.appendChild(grid);
      }

      // Alpha attribution
      if (alphaRef.value && attrib.value.length) {
        const sorted = attrib.value.slice().sort((a,b)=>b.lift-a.lift);
        charts.alpha = new ApexCharts(alphaRef.value, {
          ...cmnOpts,
          chart: { ...cmnOpts.chart, type: 'bar', height: 280 },
          series: [
            { name: 'avgR ON',  data: sorted.map(c => ({ x: c.name + (c.lift > 0.2 ? ' ★' : ''), y: c.avgR_on })) },
            { name: 'avgR OFF', data: sorted.map(c => ({ x: c.name + (c.lift > 0.2 ? ' ★' : ''), y: c.avgR_off })) },
          ],
          colors: ['#00d68f', '#5e6b80'],
          plotOptions: { bar: { horizontal: true, columnWidth: '70%', borderRadius: 2 } },
          xaxis: { labels: { formatter: v => (typeof v === 'number') ? v.toFixed(2) + 'R' : v, style: { colors: '#5e6b80' } } },
          legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#a4b0c2' } },
          tooltip: { theme: 'dark', y: { formatter: (v, { dataPointIndex }) => {
            const c = sorted[dataPointIndex];
            return v.toFixed(2) + 'R · ' + (c?.n_on || 0) + ' on / ' + (c?.n_off || 0) + ' off · lift ' + (c?.lift || 0).toFixed(2);
          } } },
        });
        charts.alpha.render();
      }
    };

    onMounted(fetchAll);
    onUnmounted(() => { Object.values(charts).forEach(c => { try { c.destroy(); } catch(e){} }); });
    watch(window_, fetchAll);

    return { window_, stats, attrib, winners, losers, loading, equityRef, ddRef, bySymRef, heatRef, alphaRef, fmt };
  },
  template: `
  <div class="view analytics-view">
    <div class="area-anal-toolbar anal-toolbar">
      <div class="panel-title"><span class="accent-bar"></span>Analytics</div>
      <span style="flex:1;"></span>
      <div class="window-pills">
        <button class="btn" :class="{active: window_==='7'}"  @click="window_='7'">7D</button>
        <button class="btn" :class="{active: window_==='30'}" @click="window_='30'">30D</button>
        <button class="btn" :class="{active: window_==='90'}" @click="window_='90'">90D</button>
        <button class="btn" :class="{active: window_==='3650'}" @click="window_='3650'">All</button>
      </div>
    </div>
    <div class="area-anal-grid">
      <div class="grid-2x2">
        <div class="panel">
          <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>Equity Curve</div></div>
          <div class="panel-body" ref="equityRef"></div>
        </div>
        <div class="panel">
          <div class="panel-header"><div class="panel-title"><span class="accent-bar" style="background: var(--red);"></span>Drawdown</div></div>
          <div class="panel-body" ref="ddRef"></div>
        </div>
        <div class="panel">
          <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>Per-Symbol PnL</div></div>
          <div class="panel-body" ref="bySymRef"></div>
        </div>
        <div class="panel">
          <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>Hour × DOW Heatmap</div></div>
          <div class="panel-body" ref="heatRef"></div>
        </div>
      </div>
    </div>
    <div class="panel area-anal-alpha">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar" style="background: var(--violet);"></span>Alpha Attribution — avgR on/off (★ lift &gt; 0.2)</div></div>
      <div class="panel-body" ref="alphaRef"></div>
    </div>
    <div class="panel area-anal-sidebar">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar" style="background: var(--green);"></span>Top Winners / Losers</div></div>
      <div class="panel-body">
        <div class="side-section">
          <h4>Top 5 Winners</h4>
          <div v-if="!winners.length" class="text-3" style="font-size:10.5px;">—</div>
          <div v-for="w in winners" :key="w.symbol" class="kv-grid" style="margin-bottom:6px;">
            <div class="k mono">{{ w.symbol }}</div>
            <div class="v text-pos">{{ fmt.money(w.pnl) }} <span class="text-3" style="font-size:10px;">({{ w.n || 0 }})</span></div>
          </div>
        </div>
        <div class="side-section">
          <h4>Top 5 Losers</h4>
          <div v-if="!losers.length" class="text-3" style="font-size:10.5px;">—</div>
          <div v-for="l in losers" :key="l.symbol" class="kv-grid" style="margin-bottom:6px;">
            <div class="k mono">{{ l.symbol }}</div>
            <div class="v text-neg">{{ fmt.money(l.pnl) }} <span class="text-3" style="font-size:10px;">({{ l.n || 0 }})</span></div>
          </div>
        </div>
        <div class="side-section">
          <h4>Snapshot</h4>
          <div class="kv-grid">
            <div class="k">Trades</div>     <div class="v">{{ stats.n_trades || 0 }}</div>
            <div class="k">Win Rate</div>   <div class="v">{{ stats.win_rate ? (stats.win_rate*100).toFixed(1)+'%' : '—' }}</div>
            <div class="k">Profit Factor</div><div class="v">{{ stats.profit_factor ? stats.profit_factor.toFixed(2) : '—' }}</div>
            <div class="k">Avg R</div>      <div class="v">{{ stats.avg_r ? fmt.R(stats.avg_r) : '—' }}</div>
            <div class="k">Max DD</div>     <div class="v text-neg">{{ stats.max_dd_pct ? stats.max_dd_pct.toFixed(1)+'%' : '—' }}</div>
            <div class="k">Net PnL</div>
            <div class="v" :class="{ 'text-pos': stats.net_pnl > 0, 'text-neg': stats.net_pnl < 0 }">{{ fmt.money(stats.net_pnl || 0) }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
  `,
});

/* ════════════════════════════════════════════════════════════════════════
   RISK VIEW
   ════════════════════════════════════════════════════════════════════════ */
const RiskView = defineComponent({
  setup() {
    const portfolio = usePortfolioStore();
    const risk = ref({ var_pct: 0, exposure_pct: 0, concentration_flags: [], by_currency: [], dir_split: { LONG: 0, SHORT: 0, BOTH: 0 }, hrp: [] });
    const corr = ref({ symbols: [], matrix: [] });
    const exposureRef = ref(null), dirRef = ref(null);
    let charts = {};
    const hovered = ref(null);

    const fetchAll = async () => {
      try {
        const [r1, r2] = await Promise.all([
          fetch('/api/v2/portfolio_risk').then(r => r.ok ? r.json() : null),
          fetch('/api/v2/correlation').then(r => r.ok ? r.json() : null),
        ]);
        if (r1) risk.value = { ...risk.value, ...r1 };
        if (r2) corr.value = r2;
        renderCharts();
      } catch (e) { console.warn(e); }
    };

    const renderCharts = () => {
      Object.values(charts).forEach(c => { try { c.destroy(); } catch(e){} });
      charts = {};

      // Currency exposure bar
      if (exposureRef.value && risk.value.by_currency) {
        const arr = risk.value.by_currency.slice().sort((a,b) => Math.abs(b.net) - Math.abs(a.net));
        charts.exp = new ApexCharts(exposureRef.value, {
          chart: { background: 'transparent', foreColor: '#a4b0c2', type: 'bar', height: 240, toolbar: { show: false }, animations: { enabled: false } },
          series: [{ name: 'Net', data: arr.map(c => ({ x: c.ccy, y: c.net })) }],
          plotOptions: { bar: { distributed: true, columnWidth: '60%', borderRadius: 2 } },
          colors: arr.map(c => c.net >= 0 ? '#00d68f' : '#ff4466'),
          dataLabels: { enabled: false },
          grid: { borderColor: '#1f2937', strokeDashArray: 2 },
          xaxis: { labels: { style: { colors: '#5e6b80' } } },
          yaxis: { labels: { formatter: v => v.toFixed(2), style: { colors: '#5e6b80' } } },
          legend: { show: false },
          tooltip: { theme: 'dark' },
        });
        charts.exp.render();
      }

      // Direction pie
      if (dirRef.value) {
        const ds = risk.value.dir_split || { LONG: 0, SHORT: 0, BOTH: 0 };
        charts.dir = new ApexCharts(dirRef.value, {
          chart: { background: 'transparent', foreColor: '#a4b0c2', type: 'donut', height: 240, animations: { enabled: false } },
          series: [ds.LONG || 0, ds.SHORT || 0, ds.BOTH || 0],
          labels: ['LONG', 'SHORT', 'BOTH'],
          colors: ['#00d68f', '#ff4466', '#7a8eff'],
          legend: { position: 'bottom', labels: { colors: '#a4b0c2' } },
          dataLabels: { style: { colors: ['#07090d'] } },
          tooltip: { theme: 'dark' },
          plotOptions: { pie: { donut: { size: '60%' } } },
        });
        charts.dir.render();
      }
    };

    const cellColor = (v) => {
      if (v === null || v === undefined) return 'rgba(94,107,128,0.15)';
      const a = Math.min(0.85, 0.10 + Math.abs(v) * 0.75);
      return v >= 0
        ? `rgba(0,214,143,${a})`
        : `rgba(255,68,102,${a})`;
    };

    onMounted(fetchAll);
    onUnmounted(() => { Object.values(charts).forEach(c => { try { c.destroy(); } catch(e){} }); });

    return { risk, corr, exposureRef, dirRef, cellColor, hovered, fmt };
  },
  template: `
  <div class="view risk-view">
    <div class="area-risk-gauges gauges-row">
      <div class="gauge">
        <div class="label">VaR (95%)</div>
        <div class="value" :class="{ warn: risk.var_pct > 2, danger: risk.var_pct > 4 }">{{ fmt.pct(risk.var_pct || 0) }}</div>
        <div class="bar"><div :style="{ width: Math.min(100, (risk.var_pct||0) * 25) + '%' }"></div></div>
      </div>
      <div class="gauge">
        <div class="label">Total Exposure</div>
        <div class="value" :class="{ warn: risk.exposure_pct > 3, danger: risk.exposure_pct > 4 }">{{ fmt.pct(risk.exposure_pct || 0) }}</div>
        <div class="bar"><div :style="{ width: Math.min(100, (risk.exposure_pct||0) * 25) + '%' }"></div></div>
      </div>
      <div class="gauge">
        <div class="label">Concentration Flags</div>
        <div class="value" :class="{ warn: (risk.concentration_flags||[]).length, danger: (risk.concentration_flags||[]).length > 2 }">{{ (risk.concentration_flags || []).length }}</div>
        <div class="text-3" style="font-size:10px;">{{ (risk.concentration_flags || []).slice(0,2).join(', ') || 'OK' }}</div>
      </div>
      <div class="gauge">
        <div class="label">Open Positions</div>
        <div class="value">{{ portfolio.positions.length }}<span class="text-3" style="font-size:13px;"> / {{ risk.max_positions || 4 }}</span></div>
        <div class="bar"><div :style="{ width: Math.min(100, portfolio.positions.length / (risk.max_positions||4) * 100) + '%' }"></div></div>
      </div>
    </div>

    <div class="panel area-risk-exposure">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>Currency Exposure (Net)</div></div>
      <div class="panel-body" ref="exposureRef"></div>
    </div>

    <div class="panel area-risk-direction">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>Direction Bias Mix</div></div>
      <div class="panel-body" ref="dirRef"></div>
    </div>

    <div class="panel area-risk-corr">
      <div class="panel-header">
        <div class="panel-title"><span class="accent-bar"></span>Correlation Matrix</div>
        <div v-if="hovered" class="text-2 mono" style="font-size:10.5px;">{{ hovered.a }} ↔ {{ hovered.b }}: {{ hovered.v.toFixed(2) }}</div>
      </div>
      <div class="panel-body corr-grid">
        <table v-if="corr.symbols && corr.symbols.length" style="border-collapse: separate; border-spacing: 1px;">
          <thead>
            <tr>
              <th></th>
              <th v-for="s in corr.symbols" :key="s" class="mono" style="font-size:8.5px; color: var(--text-3); padding: 2px; transform: rotate(-45deg); height: 50px;">{{ s }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(s, i) in corr.symbols" :key="s">
              <td class="mono" style="font-size:9.5px; color: var(--text-2); padding-right:6px; text-align:right;">{{ s }}</td>
              <td v-for="(v, j) in corr.matrix[i] || []" :key="j" class="corr-cell"
                  :style="{ background: cellColor(v) }"
                  @mouseenter="hovered = { a: corr.symbols[i], b: corr.symbols[j], v }"
                  @mouseleave="hovered = null"
                  :title="corr.symbols[i] + ' ↔ ' + corr.symbols[j] + ': ' + v.toFixed(2)">
                {{ v.toFixed(1).replace('-0.0','0') }}
              </td>
            </tr>
          </tbody>
        </table>
        <div v-else class="text-3" style="padding:18px;">No correlation data</div>
      </div>
    </div>

    <div class="panel area-risk-cluster">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>HRP Clusters / Risk Multipliers</div></div>
      <div class="panel-body">
        <table class="tbl">
          <thead>
            <tr>
              <th>Symbol</th><th>Cluster</th><th>HRP Wt</th><th>Vol Factor</th><th>VaR Factor</th><th>Combined</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="!(risk.hrp || []).length"><td colspan="6" class="text-3" style="text-align:center;padding:14px;">No HRP data</td></tr>
            <tr v-for="r in (risk.hrp || [])" :key="r.symbol">
              <td>{{ r.symbol }}</td>
              <td><span class="status-pill idle">C{{ r.cluster ?? '?' }}</span></td>
              <td>{{ (r.hrp_weight ?? 0).toFixed(3) }}</td>
              <td>{{ (r.vol_factor ?? 1).toFixed(2) }}</td>
              <td>{{ (r.var_factor ?? 1).toFixed(2) }}</td>
              <td :class="{ 'text-pos': (r.combined_factor ?? 1) > 1, 'text-neg': (r.combined_factor ?? 1) < 1 }">{{ (r.combined_factor ?? 1).toFixed(2) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
  `,
});

/* ════════════════════════════════════════════════════════════════════════
   SYSTEM HEALTH VIEW
   ════════════════════════════════════════════════════════════════════════ */
const SystemView = defineComponent({
  setup() {
    const alerts = useAlertsStore();
    const app = useAppStore();
    const health = ref({ mt5: {}, ml: [], rl: [], errors: [], process: {}, latency: {} });
    const loading = ref(true);
    const latencyRef = ref(null);
    let chart;

    const fetchAll = async () => {
      try {
        const r = await fetch('/api/v2/system_health');
        if (r.ok) {
          health.value = { ...health.value, ...(await r.json()) };
        }
        renderLatency();
      } catch (e) { console.warn(e); }
      loading.value = false;
    };

    const renderLatency = () => {
      if (chart) try { chart.destroy(); } catch(e){}
      if (!latencyRef.value) return;
      const lat = health.value.latency || {};
      const buckets = lat.histogram || [];
      chart = new ApexCharts(latencyRef.value, {
        chart: { background: 'transparent', foreColor: '#a4b0c2', type: 'bar', height: 200, toolbar: { show: false }, animations: { enabled: false } },
        series: [{ name: 'count', data: buckets.map(b => ({ x: b.bucket + 'ms', y: b.count })) }],
        plotOptions: { bar: { columnWidth: '80%', borderRadius: 1 } },
        colors: ['#7a8eff'],
        dataLabels: { enabled: false },
        grid: { borderColor: '#1f2937' },
        xaxis: { labels: { style: { colors: '#5e6b80', fontSize: '10px' } } },
        yaxis: { labels: { style: { colors: '#5e6b80' } } },
        legend: { show: false },
        tooltip: { theme: 'dark' },
        annotations: lat.p50 ? {
          xaxis: [
            { x: lat.p50 + 'ms', borderColor: '#00d68f', label: { text: 'p50 ' + lat.p50 + 'ms', style: { background: '#00d68f', color: '#000' } } },
            { x: lat.p99 + 'ms', borderColor: '#ff4466', label: { text: 'p99 ' + lat.p99 + 'ms', style: { background: '#ff4466', color: '#000' } } },
          ],
        } : {},
      });
      chart.render();
    };

    let pollTimer;
    onMounted(() => {
      fetchAll();
      pollTimer = setInterval(fetchAll, 5000);
    });
    onUnmounted(() => {
      clearInterval(pollTimer);
      if (chart) try { chart.destroy(); } catch(e){}
    });

    const tickAge = (sym) => {
      const ts = app.last_tick_ts[sym];
      if (!ts) return Infinity;
      return (Date.now() - ts) / 1000;
    };

    return { health, alerts, loading, app, latencyRef, tickAge, fmt, SYMBOL_LIST };
  },
  template: `
  <div class="view system-view">
    <div class="panel area-sys-mt5">
      <div class="panel-header">
        <div class="panel-title"><span class="accent-bar"></span>MT5 Connection</div>
        <span class="status-pill" :class="health.mt5?.connected ? 'ok' : 'err'">
          <span class="dot"></span>{{ health.mt5?.connected ? 'CONNECTED' : 'DISCONNECTED' }}
        </span>
      </div>
      <div class="panel-body">
        <div class="kv-grid" style="padding: 10px 14px;">
          <div class="k">Account</div>     <div class="v">{{ health.mt5?.login || '—' }}</div>
          <div class="k">Server</div>      <div class="v" style="font-size:10.5px;">{{ health.mt5?.server || '—' }}</div>
          <div class="k">Last Sync</div>   <div class="v">{{ health.mt5?.last_sync ? fmt.ago(health.mt5.last_sync) : '—' }}</div>
          <div class="k">Symbols Active</div><div class="v">{{ Object.keys(app.last_tick_ts).length }}/{{ SYMBOL_LIST.length }}</div>
        </div>
        <table class="tbl">
          <thead>
            <tr><th>Symbol</th><th>Last Tick</th><th>Age</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr v-for="sym in SYMBOL_LIST" :key="sym">
              <td>{{ sym }}</td>
              <td>{{ app.last_tick_ts[sym] ? fmt.ts(app.last_tick_ts[sym]) : '—' }}</td>
              <td>{{ app.last_tick_ts[sym] ? fmt.ago(app.last_tick_ts[sym]) : '—' }}</td>
              <td>
                <span class="status-pill" :class="tickAge(sym) > 60 ? 'err' : tickAge(sym) > 10 ? 'warn' : 'ok'">
                  <span class="dot"></span>{{ tickAge(sym) > 60 ? 'STALE' : tickAge(sym) > 10 ? 'SLOW' : 'LIVE' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="panel area-sys-ml">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>ML Models</div></div>
      <div class="panel-body">
        <table class="tbl">
          <thead><tr><th>Symbol</th><th>Loaded</th><th>AUC</th><th>Trees</th><th>Trained</th><th>Status</th></tr></thead>
          <tbody>
            <tr v-if="!(health.ml || []).length"><td colspan="6" class="text-3" style="text-align:center;padding:14px;">No ML data</td></tr>
            <tr v-for="m in (health.ml || [])" :key="m.symbol">
              <td>{{ m.symbol }}</td>
              <td>{{ m.loaded ? '✓' : '—' }}</td>
              <td>{{ m.auc ? m.auc.toFixed(3) : '—' }}</td>
              <td>{{ m.n_trees || '—' }}</td>
              <td style="font-size:10.5px;">{{ m.trained_at ? fmt.ago(m.trained_at) : '—' }}</td>
              <td>
                <span class="status-pill" :class="m.status === 'PROD' ? 'ok' : m.status === 'NEW' ? 'warn' : m.status === 'RETRAIN_NEEDED' ? 'err' : 'idle'">
                  <span class="dot"></span>{{ m.status || 'IDLE' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="panel area-sys-rl">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>RL Trail State</div></div>
      <div class="panel-body">
        <table class="tbl">
          <thead><tr><th>Symbol</th><th>lock×</th><th>be×</th><th>tight×</th><th>n</th><th>Recent</th></tr></thead>
          <tbody>
            <tr v-if="!(health.rl || []).length"><td colspan="6" class="text-3" style="text-align:center;padding:14px;">No RL data</td></tr>
            <tr v-for="r in (health.rl || [])" :key="r.symbol">
              <td>{{ r.symbol }}</td>
              <td>{{ (r.lock_mult ?? 1).toFixed(2) }}</td>
              <td>{{ (r.be_mult ?? 1).toFixed(2) }}</td>
              <td>{{ (r.tight_mult ?? 1).toFixed(2) }}</td>
              <td>{{ r.n_trades || 0 }}</td>
              <td style="font-size:10px;">
                <span v-for="(o, i) in (r.recent_outcomes || [])" :key="i"
                      :style="{ display:'inline-block', width:'5px', height:'10px', background: o > 0 ? '#00d68f' : '#ff4466', margin:'0 1px' }"></span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="panel area-sys-errors">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar" style="background: var(--red);"></span>Recent Errors</div></div>
      <div class="panel-body">
        <div v-if="!(health.errors || []).length" class="text-3" style="padding:14px;">No errors</div>
        <div v-for="(e, i) in (health.errors || [])" :key="i" class="log-line">
          <span class="ts">{{ fmt.ts(e.ts) }}</span>
          <span class="lvl" :class="e.level || 'ERROR'">{{ e.level || 'ERROR' }}</span>
          <span class="msg" :title="e.message">{{ e.message }}</span>
        </div>
      </div>
    </div>

    <div class="panel area-sys-alerts">
      <div class="panel-header">
        <div class="panel-title"><span class="accent-bar"></span>Live Alerts</div>
        <span class="text-3 mono" style="font-size:10px;">{{ alerts.feed.length }}</span>
      </div>
      <div class="panel-body">
        <div v-if="!alerts.feed.length" class="text-3" style="padding:14px;">Awaiting events…</div>
        <div v-for="(a, i) in alerts.feed" :key="i" class="log-line">
          <span class="ts">{{ fmt.ts(a.ts) }}</span>
          <span class="lvl" :class="a.level || (a.kind === 'open' ? 'INFO' : a.kind === 'close' ? 'INFO' : 'WARN')">
            {{ (a.kind || a.level || 'EVT').toUpperCase() }}
          </span>
          <span class="msg" :title="a.message">{{ a.message || (a.symbol + ' ' + (a.kind || '')) }}</span>
        </div>
      </div>
    </div>

    <div class="panel area-sys-process">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>Process / Loop Latency</div></div>
      <div class="panel-body">
        <div class="kv-grid" style="padding: 10px 14px;">
          <div class="k">PID</div>     <div class="v">{{ health.process?.pid || '—' }}</div>
          <div class="k">Uptime</div>  <div class="v">{{ health.process?.uptime ? fmt.duration(health.process.uptime) : '—' }}</div>
          <div class="k">Memory</div>  <div class="v">{{ health.process?.memory_mb ? health.process.memory_mb.toFixed(0) + ' MB' : '—' }}</div>
          <div class="k">CPU</div>     <div class="v">{{ health.process?.cpu_pct ? health.process.cpu_pct.toFixed(1) + '%' : '—' }}</div>
          <div class="k">Loop p50</div><div class="v">{{ health.latency?.p50 ?? '—' }}<span class="text-3" style="font-size:10px;"> ms</span></div>
          <div class="k">Loop p99</div><div class="v">{{ health.latency?.p99 ?? '—' }}<span class="text-3" style="font-size:10px;"> ms</span></div>
        </div>
        <div ref="latencyRef" style="margin-top: 8px;"></div>
      </div>
    </div>
  </div>
  `,
});

/* ════════════════════════════════════════════════════════════════════════
   JOURNAL VIEW
   ════════════════════════════════════════════════════════════════════════ */
const JournalView = defineComponent({
  setup() {
    const filter = reactive({
      symbol: '',
      side: '',
      outcome: '',
      gate: '',
      since_days: '90',
      page: 1,
      page_size: 50,
      sort: 'entry_ts',
      sort_dir: -1,
    });
    const trades = ref([]);
    const total = ref(0);
    const stats = ref({});
    const loading = ref(false);
    const expanded = ref(new Set());

    const fetchData = async () => {
      loading.value = true;
      try {
        const params = new URLSearchParams();
        params.append('limit', filter.page_size);
        params.append('offset', (filter.page - 1) * filter.page_size);
        params.append('since_days', filter.since_days);
        if (filter.symbol) params.append('symbol', filter.symbol);
        if (filter.side) params.append('side', filter.side);
        if (filter.outcome) params.append('outcome', filter.outcome);
        if (filter.gate) params.append('gate', filter.gate);
        const r = await fetch('/api/v2/journal?' + params.toString());
        if (r.ok) {
          const d = await r.json();
          trades.value = d.trades || [];
          total.value = d.total || trades.value.length;
        }
        const s = await fetch('/api/v2/journal/stats?days=' + filter.since_days
          + (filter.symbol ? '&symbol=' + filter.symbol : ''));
        if (s.ok) stats.value = await s.json();
      } catch (e) { console.warn(e); }
      loading.value = false;
    };

    const toggleSort = (k) => {
      if (filter.sort === k) filter.sort_dir *= -1;
      else { filter.sort = k; filter.sort_dir = -1; }
    };

    const sortedTrades = computed(() => {
      const arr = trades.value.slice();
      const k = filter.sort, d = filter.sort_dir;
      arr.sort((a, b) => {
        const va = a[k], vb = b[k];
        if (typeof va === 'number') return (va - vb) * d;
        return String(va || '').localeCompare(String(vb || '')) * d;
      });
      return arr;
    });

    const totalPages = computed(() => Math.max(1, Math.ceil(total.value / filter.page_size)));

    const exportCsv = () => {
      const cols = ['entry_ts','symbol','side','score','regime','entry','exit','profit','profit_r','duration','gate','exit_reason'];
      const rows = trades.value.map(t => cols.map(c => {
        const v = t[c];
        if (v === null || v === undefined) return '';
        return ('' + v).replace(/"/g, '""');
      }).map(v => '"' + v + '"').join(','));
      const csv = cols.join(',') + '\n' + rows.join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'journal_' + new Date().toISOString().slice(0,10) + '.csv';
      a.click();
    };

    onMounted(fetchData);
    watch(() => [filter.symbol, filter.side, filter.outcome, filter.gate, filter.since_days, filter.page], fetchData);

    return { filter, trades, total, stats, loading, expanded, sortedTrades, totalPages, fetchData, exportCsv, toggleSort, fmt, SYMBOL_LIST };
  },
  template: `
  <div class="view journal-view">
    <div class="area-journal-filter filter-bar">
      <label>
        <span>Symbol</span>
        <select v-model="filter.symbol" class="input">
          <option value="">All</option>
          <option v-for="s in SYMBOL_LIST" :key="s" :value="s">{{ s }}</option>
        </select>
      </label>
      <label>
        <span>Side</span>
        <select v-model="filter.side" class="input">
          <option value="">All</option>
          <option value="LONG">LONG</option>
          <option value="SHORT">SHORT</option>
        </select>
      </label>
      <label>
        <span>Outcome</span>
        <select v-model="filter.outcome" class="input">
          <option value="">All</option>
          <option value="WIN">WIN</option>
          <option value="LOSS">LOSS</option>
        </select>
      </label>
      <label>
        <span>Gate</span>
        <input v-model="filter.gate" class="input" placeholder="e.g. ml_filter">
      </label>
      <label>
        <span>Since (days)</span>
        <select v-model="filter.since_days" class="input">
          <option value="7">7</option><option value="30">30</option><option value="90">90</option><option value="365">365</option><option value="3650">All</option>
        </select>
      </label>
      <span style="flex:1;"></span>
      <button class="btn accent" @click="fetchData">Refresh</button>
      <button class="btn" @click="exportCsv">Export CSV</button>
    </div>

    <div class="panel area-journal-table">
      <div class="panel-header">
        <div class="panel-title"><span class="accent-bar"></span>Trades <span class="text-3 mono" style="font-size:10px;margin-left:8px;">{{ total }} total · page {{ filter.page }}/{{ totalPages }}</span></div>
        <div class="panel-tools">
          <button class="btn sm" :disabled="filter.page <= 1" @click="filter.page--">‹ Prev</button>
          <button class="btn sm" :disabled="filter.page >= totalPages" @click="filter.page++">Next ›</button>
        </div>
      </div>
      <div class="panel-body">
        <table class="tbl">
          <thead>
            <tr>
              <th @click="toggleSort('entry_ts')">Date</th>
              <th @click="toggleSort('symbol')">Symbol</th>
              <th @click="toggleSort('side')">Side</th>
              <th @click="toggleSort('score')">Score</th>
              <th>Regime</th>
              <th @click="toggleSort('entry')">Entry</th>
              <th @click="toggleSort('exit')">Exit</th>
              <th @click="toggleSort('profit')">P&L</th>
              <th @click="toggleSort('profit_r')">R</th>
              <th @click="toggleSort('duration')">Dur</th>
              <th>Gate</th>
              <th>Exit Reason</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="loading"><td colspan="12" class="text-3" style="text-align:center;padding:18px;">Loading…</td></tr>
            <tr v-else-if="!trades.length"><td colspan="12" class="text-3" style="text-align:center;padding:18px;">No trades match filters</td></tr>
            <template v-for="t in sortedTrades" :key="t.id || (t.entry_ts + t.symbol)">
              <tr @click="expanded.has(t.id) ? expanded.delete(t.id) : expanded.add(t.id); expanded = new Set(expanded);" style="cursor:pointer;">
                <td style="font-size:10.5px;">{{ fmt.date(t.entry_ts) }} {{ fmt.ts(t.entry_ts) }}</td>
                <td>{{ t.symbol }}</td>
                <td><span class="side-pill" :class="t.side?.toLowerCase()">{{ t.side }}</span></td>
                <td>{{ (t.score ?? 0).toFixed(1) }}</td>
                <td style="font-size:10.5px;">{{ t.regime || '—' }}</td>
                <td>{{ fmt.price(t.entry, t.symbol) }}</td>
                <td>{{ fmt.price(t.exit, t.symbol) }}</td>
                <td :class="{ pos: t.profit > 0, neg: t.profit < 0 }">{{ fmt.money(t.profit) }}</td>
                <td :class="{ pos: t.profit_r > 0, neg: t.profit_r < 0 }">{{ fmt.R(t.profit_r) }}</td>
                <td>{{ fmt.duration(t.duration) }}</td>
                <td style="font-size:10.5px;">{{ t.gate || '—' }}</td>
                <td style="font-size:10.5px;">{{ t.exit_reason || '—' }}</td>
              </tr>
              <tr v-if="expanded.has(t.id)" class="expand-row">
                <td colspan="12">
                  <div class="meta-grid">
                    <div class="meta-cell" v-for="(v, k) in (t.entry_metadata || {})" :key="k">
                      <div class="k">{{ k }}</div><div class="v">{{ typeof v === 'number' ? v.toFixed(3) : v }}</div>
                    </div>
                  </div>
                  <div v-if="(t.score_components || []).length" style="margin-top:8px;">
                    <div class="text-3" style="font-size:9.5px;text-transform:uppercase;letter-spacing:0.08em;">Score components</div>
                    <div class="meta-grid" style="margin-top:6px;">
                      <div class="meta-cell" v-for="c in (t.score_components || [])" :key="c.name">
                        <div class="k">{{ c.name }}</div><div class="v">{{ (c.contrib ?? 0).toFixed(2) }}</div>
                      </div>
                    </div>
                  </div>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </div>

    <div class="panel area-journal-stats">
      <div class="panel-header"><div class="panel-title"><span class="accent-bar"></span>Filtered Stats</div></div>
      <div class="panel-body">
        <div class="side-section">
          <h4>Aggregates</h4>
          <div class="kv-grid">
            <div class="k">Trades</div>      <div class="v">{{ stats.n_trades || trades.length }}</div>
            <div class="k">Win Rate</div>    <div class="v">{{ stats.win_rate ? (stats.win_rate*100).toFixed(1)+'%' : '—' }}</div>
            <div class="k">Avg R</div>       <div class="v">{{ stats.avg_r ? fmt.R(stats.avg_r) : '—' }}</div>
            <div class="k">Profit Factor</div><div class="v">{{ stats.profit_factor ? stats.profit_factor.toFixed(2) : '—' }}</div>
            <div class="k">Net PnL</div>     <div class="v" :class="{ 'text-pos': stats.net_pnl > 0, 'text-neg': stats.net_pnl < 0 }">{{ fmt.money(stats.net_pnl || 0) }}</div>
            <div class="k">Max DD</div>      <div class="v text-neg">{{ stats.max_dd_pct ? stats.max_dd_pct.toFixed(1)+'%' : '—' }}</div>
            <div class="k">Best</div>        <div class="v text-pos">{{ stats.best_trade ? fmt.money(stats.best_trade) : '—' }}</div>
            <div class="k">Worst</div>       <div class="v text-neg">{{ stats.worst_trade ? fmt.money(stats.worst_trade) : '—' }}</div>
            <div class="k">Avg Hold</div>    <div class="v">{{ stats.avg_duration ? fmt.duration(stats.avg_duration) : '—' }}</div>
          </div>
        </div>
        <div class="side-section">
          <h4>Tip</h4>
          <div class="text-3" style="font-size:10.5px; line-height:1.5;">
            Click any row to inspect entry metadata and score components.
            Use <span class="mono">j</span> / <span class="mono">k</span> to navigate, <span class="mono">?</span> for keyboard help.
          </div>
        </div>
      </div>
    </div>
  </div>
  `,
});

/* ════════════════════════════════════════════════════════════════════════
   APP SHELL — top bar + nav + view router + modal
   ════════════════════════════════════════════════════════════════════════ */
const AppShell = defineComponent({
  setup() {
    const app = useAppStore();
    const portfolio = usePortfolioStore();
    const symbols = useSymbolsStore();
    const decisions = useDecisionsStore();
    const router = VueRouter.useRouter();

    const tabs = [
      { name: 'trading',   label: 'Trading',   key: '1' },
      { name: 'analytics', label: 'Analytics', key: '2' },
      { name: 'risk',      label: 'Risk',      key: '3' },
      { name: 'system',    label: 'System',    key: '4' },
      { name: 'journal',   label: 'Journal',   key: '5' },
    ];

    const currentTab = computed(() => router.currentRoute.value.name || 'trading');

    const onKeydown = (e) => {
      if (e.target?.tagName === 'INPUT' || e.target?.tagName === 'SELECT' || e.target?.tagName === 'TEXTAREA') return;
      if (e.key === '1') router.push('/trading');
      if (e.key === '2') router.push('/analytics');
      if (e.key === '3') router.push('/risk');
      if (e.key === '4') router.push('/system');
      if (e.key === '5') router.push('/journal');
      if (e.key === '?') app.helpOpen = !app.helpOpen;
      if (e.key === 'Escape') { app.confirm = null; app.helpOpen = false; }
      if (e.key === 'j' || e.key === 'k') {
        const ix = SYMBOL_LIST.indexOf(app.selected_symbol);
        const next = e.key === 'j' ? (ix+1) % SYMBOL_LIST.length : (ix-1+SYMBOL_LIST.length) % SYMBOL_LIST.length;
        app.setSymbol(SYMBOL_LIST[next]);
      }
      if (e.key === 'c') {
        // Close selected symbol if it has an open position
        const p = portfolio.positions.find(p => p.symbol === app.selected_symbol);
        if (p) {
          app.askConfirm('Close ' + p.symbol + ' ' + p.side + '? P&L: ' + fmt.money(p.profit), () => {
            fetch('/api/close_symbol', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ symbol: p.symbol })
            });
            app.closeConfirm();
          });
        }
      }
    };

    onMounted(() => {
      document.addEventListener('keydown', onKeydown);
      portfolio.fetch();
      symbols.fetch();
      decisions.fetchInitial();
      // Refresh portfolio every 5s as a fallback
      setInterval(() => { if (!app.connected) portfolio.fetch(); }, 5000);
    });
    onUnmounted(() => document.removeEventListener('keydown', onKeydown));

    return { app, portfolio, tabs, currentTab, fmt };
  },
  template: `
  <div class="topbar">
    <div class="brand">
      <div class="brand-dot"></div>
      <div class="brand-text">DRAGON<span class="pro">PRO</span></div>
    </div>
    <nav class="topnav">
      <router-link v-for="t in tabs" :key="t.name" :to="'/' + t.name" custom v-slot="{ navigate, isActive }">
        <button class="tab" :class="{ active: isActive || currentTab === t.name }" @click="navigate"><span>{{ t.label }}</span><span class="kbd">{{ t.key }}</span></button>
      </router-link>
    </nav>
    <span class="topbar-spacer"></span>
    <div class="account-stats">
      <div class="acct-stat">
        <div class="label">Balance</div>
        <div class="value mono">{{ fmt.money(portfolio.balance) }}</div>
      </div>
      <div class="acct-stat">
        <div class="label">Equity</div>
        <div class="value mono">{{ fmt.money(portfolio.equity) }}</div>
      </div>
      <div class="acct-stat">
        <div class="label">Open P&L</div>
        <div class="value mono" :class="{ pos: portfolio.open_pnl > 0, neg: portfolio.open_pnl < 0 }">{{ fmt.money(portfolio.open_pnl) }}</div>
      </div>
      <div class="acct-stat">
        <div class="label">DD</div>
        <div class="value mono" :class="{ neg: portfolio.drawdown_pct > 5, warn: portfolio.drawdown_pct > 3 }">{{ fmt.pct(-Math.abs(portfolio.drawdown_pct)) }}</div>
      </div>
      <div class="acct-stat">
        <div class="label">Positions</div>
        <div class="value mono">{{ portfolio.positions.length }}</div>
      </div>
    </div>
    <span class="connection-pill" :class="{ ok: app.connected, bad: !app.connected }">
      <span class="dot"></span>{{ app.connected ? 'LIVE' : 'OFFLINE' }}
    </span>
    <button class="btn sm" @click="app.helpOpen = true" title="Keyboard help">?</button>
  </div>

  <router-view v-slot="{ Component }">
    <transition name="fade" mode="out-in">
      <component :is="Component" />
    </transition>
  </router-view>

  <!-- JARVIS confirm modal -->
  <transition name="fade">
    <div v-if="app.confirm" class="modal-backdrop" @click.self="app.closeConfirm()">
      <div class="modal">
        <h3>Confirm Action</h3>
        <p>{{ app.confirm.message }}</p>
        <div class="modal-actions">
          <button class="btn" @click="app.closeConfirm()">Cancel</button>
          <button class="btn danger" @click="app.confirm.onConfirm()">Confirm</button>
        </div>
      </div>
    </div>
  </transition>

  <!-- Keyboard help -->
  <transition name="fade">
    <div v-if="app.helpOpen" class="modal-backdrop" @click.self="app.helpOpen = false">
      <div class="modal">
        <h3>Keyboard Shortcuts</h3>
        <table class="tbl" style="font-size:11px;">
          <tbody>
            <tr><td><span class="mono">1-5</span></td><td>Switch tabs</td></tr>
            <tr><td><span class="mono">j / k</span></td><td>Next / prev symbol</td></tr>
            <tr><td><span class="mono">c</span></td><td>Close selected symbol's position</td></tr>
            <tr><td><span class="mono">?</span></td><td>Toggle this help</td></tr>
            <tr><td><span class="mono">Esc</span></td><td>Close modal</td></tr>
          </tbody>
        </table>
        <div class="modal-actions" style="margin-top:14px;">
          <button class="btn accent" @click="app.helpOpen = false">Got it</button>
        </div>
      </div>
    </div>
  </transition>
  `,
});

/* ════════════════════════════════════════════════════════════════════════
   ROUTER + APP MOUNT
   ════════════════════════════════════════════════════════════════════════ */
const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/trading' },
    { path: '/trading',   name: 'trading',   component: TradingView },
    { path: '/analytics', name: 'analytics', component: AnalyticsView },
    { path: '/risk',      name: 'risk',      component: RiskView },
    { path: '/system',    name: 'system',    component: SystemView },
    { path: '/journal',   name: 'journal',   component: JournalView },
  ],
});

try {
  _bootStatus('creating Pinia + app...');
  const pinia = createPinia();
  const app = createApp(AppShell);
  app.config.errorHandler = (err, _instance, info) => {
    _bootError('[vue-error] ' + info + '\n' + (err && err.stack ? err.stack : err));
  };
  app.use(pinia);
  app.use(router);

  _bootStatus('setting up stores...');
  const stores = {
    app: useAppStore(pinia),
    portfolio: usePortfolioStore(pinia),
    symbols: useSymbolsStore(pinia),
    decisions: useDecisionsStore(pinia),
    alerts: useAlertsStore(pinia),
  };
  setupSocket(stores);

  _bootStatus('mounting Vue...');
  app.mount('#app');
  _bootStatus('mounted ✓');
  setTimeout(() => { try { document.getElementById('boot-status').style.display = 'none'; } catch(_){} }, 2000);
} catch (e) {
  _bootError('[mount] ' + (e.stack || e.message || String(e)));
}
</script>
</body>
</html>
"""


# Inject server-side data BEFORE rendering (the template uses Vue {{ }} too,
# so we replace placeholders manually rather than using Jinja).
def _render_page() -> str:
    page = PRO_HTML
    page = page.replace("__SYMBOL_LIST__", SYMBOL_LIST_JSON)
    page = page.replace("__SYMBOL_META__", SYMBOL_META_JSON)
    page = page.replace("__STARTING_BAL__", str(STARTING_BAL))
    return page


@pro_dashboard_bp.route("/", methods=["GET"])
def pro_index():
    """Render the pro dashboard SPA.

    The HTML payload contains Vue 3 mustache interpolation ({{ x }}) which
    collides with Jinja2's syntax. Return a raw Response with the page body
    rather than running it through Jinja's render_template_string.
    """
    from flask import Response
    return Response(_render_page(), mimetype="text/html; charset=utf-8")


@pro_dashboard_bp.route("/health", methods=["GET"])
def pro_health() -> dict:
    """Lightweight self-check used by parent app for blueprint verification."""
    return {"ok": True, "blueprint": "pro_dashboard", "symbols": len(SYMBOL_LIST)}
