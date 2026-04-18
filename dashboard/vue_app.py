# dashboard/vue_app.py
# Dragon Trader — Professional Trading Terminal Dashboard
# Single-file Vue 3 app served via Flask render_template_string
# All data via WebSocket (Socket.IO) reactive binding — no blinking

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
<title>Dragon Trader Terminal</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script src="https://cdn.socket.io/4.7.4/socket.io.min.js"></script>
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
<style>
/* ============================================================
   DRAGON TRADER — PROFESSIONAL TERMINAL THEME
   ============================================================ */
:root {
  --bg:       #0a0e17;
  --bg1:      #0f1520;
  --bg2:      #141b28;
  --bg3:      #1a2235;
  --bg-card:  #0d1219;
  --bdr:      #1e2a3a;
  --bdr2:     #2a3a50;
  --bdr3:     #354560;
  --t1:       #e1e8f0;
  --t2:       #a0b0c4;
  --t3:       #607080;
  --cyan:     #00c8e0;
  --green:    #00d68f;
  --green-bg: rgba(0,214,143,0.08);
  --red:      #ff4466;
  --red-bg:   rgba(255,68,102,0.08);
  --amber:    #f0a030;
  --amber-bg: rgba(240,160,48,0.08);
  --blue:     #3388ff;
  --blue-bg:  rgba(51,136,255,0.08);
  --purple:   #9966ff;
  --purple-bg:rgba(153,102,255,0.08);
  --gold:     #ffc800;
}
*{margin:0;padding:0;box-sizing:border-box}
body{
  background:var(--bg);color:var(--t1);
  font-family:'Inter',sans-serif;font-size:12px;
  overflow-x:hidden;min-height:100vh;
}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:var(--bdr2);border-radius:2px}
::-webkit-scrollbar-track{background:transparent}

/* ============================================================
   HEADER — 48px sticky
   ============================================================ */
.hdr{
  height:48px;display:flex;align-items:center;justify-content:space-between;
  padding:0 16px;position:sticky;top:0;z-index:100;
  background:var(--bg1);border-bottom:1px solid var(--bdr);
}
.hdr-left{display:flex;align-items:center;gap:12px}
.logo{
  font-family:'JetBrains Mono';font-size:14px;font-weight:700;
  color:var(--cyan);letter-spacing:2px;
}
.logo-sub{font-size:9px;color:var(--t3);letter-spacing:1px;margin-top:-2px}
.hdr-stats{display:flex;gap:16px;align-items:center}
.hdr-stat{text-align:center}
.hdr-stat .lbl{
  font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:1px;
  font-weight:500;
}
.hdr-stat .val{
  font-family:'JetBrains Mono';font-size:13px;font-weight:600;
  font-variant-numeric:tabular-nums;min-width:65px;display:inline-block;text-align:center;
}
.hdr-badges{display:flex;gap:6px;align-items:center}
.pill{
  padding:3px 10px;font-family:'JetBrains Mono';font-size:9px;font-weight:600;
  letter-spacing:1px;border-radius:3px;border:1px solid;
}
.pill-mode{color:var(--cyan);background:rgba(0,200,224,0.08);border-color:rgba(0,200,224,0.25)}
.pill-session{font-weight:700;border-radius:3px}
.hdr-actions{display:flex;gap:6px;align-items:center}
.act-btn{
  padding:4px 12px;font-family:'JetBrains Mono';font-size:9px;font-weight:600;
  letter-spacing:0.5px;cursor:pointer;border:1px solid;border-radius:3px;
  transition:all 0.15s;background:transparent;
}
.act-red{color:var(--red);border-color:rgba(255,68,102,0.3)}
.act-red:hover{background:var(--red-bg);box-shadow:0 0 8px rgba(255,68,102,0.2)}
.act-amber{color:var(--amber);border-color:rgba(240,160,48,0.3)}
.act-amber:hover{background:var(--amber-bg);box-shadow:0 0 8px rgba(240,160,48,0.2)}
.act-select{
  background:var(--bg2);color:var(--t1);border:1px solid var(--bdr2);padding:4px 10px;
  font-family:'JetBrains Mono';font-size:9px;cursor:pointer;border-radius:3px;
  appearance:none;-webkit-appearance:none;
}
.act-select option{background:var(--bg1);color:var(--t1)}
.hdr-right{display:flex;align-items:center;gap:12px}
.clock{
  font-family:'JetBrains Mono';font-size:12px;color:var(--cyan);
  font-variant-numeric:tabular-nums;min-width:100px;text-align:right;
}
.status-dot{
  width:7px;height:7px;border-radius:50%;transition:background 0.3s,box-shadow 0.3s;
}
.status-dot.on{background:var(--green);box-shadow:0 0 6px var(--green)}
.status-dot.off{background:var(--red);box-shadow:0 0 6px var(--red)}
.g{color:var(--green)}.r{color:var(--red)}.cy{color:var(--cyan)}.am{color:var(--amber)}
.dim{color:var(--t3)}.mono{font-family:'JetBrains Mono'}
.bright{font-weight:600;color:var(--t1)}

/* ============================================================
   MAIN LAYOUT — CSS Grid
   ============================================================ */
.main-grid{
  display:grid;
  grid-template-columns:1fr 320px;
  grid-template-rows:1fr 1fr;
  gap:1px;
  height:calc(100vh - 48px);
  background:var(--bdr);
}
.main-grid>*{overflow:hidden;min-height:0;background:var(--bg-card)}

/* ============================================================
   PANEL CARD
   ============================================================ */
.panel{display:flex;flex-direction:column;position:relative}
.panel-h{
  padding:6px 12px;display:flex;justify-content:space-between;align-items:center;
  border-bottom:1px solid var(--bdr);background:var(--bg1);flex-shrink:0;
}
.panel-t{
  font-family:'JetBrains Mono';font-size:10px;font-weight:600;text-transform:uppercase;
  letter-spacing:1.5px;color:var(--t2);
}
.panel-badge{
  font-family:'JetBrains Mono';font-size:9px;padding:2px 6px;color:var(--t3);
  border:1px solid var(--bdr);border-radius:2px;background:var(--bg2);
}
.panel-b{flex:1;overflow-y:auto;overflow-x:hidden;position:relative}

/* ============================================================
   CHART PANEL (top-left)
   ============================================================ */
.chart-panel{position:relative}
.chart-controls{
  display:flex;align-items:center;gap:4px;padding:4px 12px;
  background:var(--bg1);border-bottom:1px solid var(--bdr);flex-shrink:0;
}
.tf-btn{
  padding:3px 8px;font-family:'JetBrains Mono';font-size:9px;font-weight:500;
  cursor:pointer;border:1px solid transparent;border-radius:2px;
  background:transparent;color:var(--t3);transition:all 0.15s;
}
.tf-btn.active{color:var(--cyan);border-color:var(--bdr2);background:rgba(0,200,224,0.06)}
.tf-btn:hover:not(.active){color:var(--t2)}
.sym-tab{
  padding:3px 10px;font-family:'JetBrains Mono';font-size:9px;font-weight:600;
  cursor:pointer;border:1px solid transparent;border-radius:2px;
  background:transparent;color:var(--t3);transition:all 0.15s;letter-spacing:0.5px;
}
.sym-tab.active{color:var(--cyan);border-color:var(--bdr2);background:rgba(0,200,224,0.06)}
.sym-tab:hover:not(.active){color:var(--t2)}
.tf-sep{width:1px;height:16px;background:var(--bdr2);margin:0 4px}
#chart-container{flex:1;width:100%;position:relative}

/* ============================================================
   MARKET SCANNER (top-right)
   ============================================================ */
.sym-card{
  padding:10px 12px;border-bottom:1px solid var(--bdr);position:relative;
  transition:background 0.15s;cursor:pointer;
}
.sym-card:hover{background:rgba(0,200,224,0.02)}
.sym-card.selected{background:rgba(0,200,224,0.05);border-left:2px solid var(--cyan)}
.sym-card:last-child{border-bottom:none}
.sym-row1{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.sym-name{font-family:'JetBrains Mono';font-size:12px;font-weight:700;letter-spacing:1px}
.sym-cat{
  font-family:'JetBrains Mono';font-size:8px;font-weight:600;padding:1px 5px;
  border-radius:2px;margin-left:5px;
}
.cat-Gold{color:var(--gold);background:rgba(255,200,0,0.08);border:1px solid rgba(255,200,0,0.2)}
.cat-Crypto{color:var(--purple);background:var(--purple-bg);border:1px solid rgba(153,102,255,0.2)}
.cat-Index{color:var(--blue);background:var(--blue-bg);border:1px solid rgba(51,136,255,0.2)}
.cat-Forex{color:var(--cyan);background:rgba(0,200,224,0.08);border:1px solid rgba(0,200,224,0.2)}
.sym-price{
  font-family:'JetBrains Mono';font-size:14px;font-weight:700;
  font-variant-numeric:tabular-nums;
}
.sym-arrow{font-size:10px;margin-left:3px}
.sym-arrow.up{color:var(--green)}.sym-arrow.dn{color:var(--red)}
.sym-row2{display:flex;gap:8px;align-items:center;margin-bottom:4px;font-size:10px}
.sym-detail{font-family:'JetBrains Mono';font-size:9px;color:var(--t3)}
.sym-detail span{color:var(--t2)}
/* Score bar */
.score-bar-wrap{display:flex;align-items:center;gap:4px;margin-bottom:3px}
.score-label-sm{
  font-family:'JetBrains Mono';font-size:8px;color:var(--t3);width:52px;
  text-align:right;flex-shrink:0;text-transform:uppercase;letter-spacing:0.5px;
}
.score-bar{height:3px;flex:1;background:rgba(255,255,255,0.04);border-radius:1px;overflow:hidden}
.score-fill{height:100%;transition:width 0.4s;border-radius:1px}
.score-val-sm{
  font-family:'JetBrains Mono';font-size:8px;color:var(--t2);width:24px;flex-shrink:0;
  text-align:right;
}
/* MTF dots */
.mtf-row{display:flex;align-items:center;gap:5px;margin-bottom:3px}
.mtf-dot{
  width:6px;height:6px;border-radius:50%;transition:background 0.3s;
}
.mtf-dot-long{background:var(--green);box-shadow:0 0 4px rgba(0,214,143,0.4)}
.mtf-dot-short{background:var(--red);box-shadow:0 0 4px rgba(255,68,102,0.4)}
.mtf-dot-flat{background:var(--t3);opacity:0.4}
.mtf-dot-lbl{font-family:'JetBrains Mono';font-size:7px;color:var(--t3)}
.eq-bar-sm{
  height:3px;border-radius:1px;overflow:hidden;flex:1;
  background:rgba(255,255,255,0.04);
}
.eq-fill{height:100%;transition:width 0.4s;border-radius:1px}
/* Position badge */
.pos-badge{
  font-family:'JetBrains Mono';font-size:9px;font-weight:600;padding:2px 6px;
  border-radius:2px;display:inline-flex;align-items:center;gap:3px;
}
.pos-flat{color:var(--t3);background:rgba(255,255,255,0.03);border:1px solid var(--bdr)}
.pos-long{color:var(--green);background:var(--green-bg);border:1px solid rgba(0,214,143,0.2)}
.pos-short{color:var(--red);background:var(--red-bg);border:1px solid rgba(255,68,102,0.2)}
.exit-badge{
  font-family:'JetBrains Mono';font-size:8px;font-weight:600;margin-left:4px;
  padding:1px 4px;border-radius:2px;
}
.exit-low{color:var(--green);background:var(--green-bg)}
.exit-med{color:var(--amber);background:var(--amber-bg)}
.exit-high{color:var(--red);background:var(--red-bg)}

/* ============================================================
   INTELLIGENCE PANEL (bottom-left)
   ============================================================ */
.intel-content{padding:10px 12px}
.sep{height:1px;margin:8px 0;background:var(--bdr)}
.empty{padding:20px;text-align:center;color:var(--t3);font-size:10px;letter-spacing:1px}
/* MasterBrain grid */
.mb-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px}
.mb-cell{
  background:var(--bg2);border:1px solid var(--bdr);padding:6px;border-radius:2px;
}
.mb-lbl{font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:0.5px}
.mb-val{font-family:'JetBrains Mono';font-size:11px;font-weight:600;margin-top:1px}
.mb-info{display:flex;gap:10px;font-size:10px;font-family:'JetBrains Mono';flex-wrap:wrap;color:var(--t3)}
/* MTF TF grid */
.mtf-tf-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px}
.mtf-tf-cell{
  background:var(--bg2);border:1px solid var(--bdr);padding:6px;text-align:center;
  border-radius:2px;position:relative;overflow:hidden;
}
.mtf-tf-cell::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.mtf-tf-cell.tf-long::before{background:var(--green)}
.mtf-tf-cell.tf-short::before{background:var(--red)}
.mtf-tf-cell.tf-flat::before{background:var(--t3);opacity:0.3}
.mtf-tf-lbl{font-family:'JetBrains Mono';font-size:8px;color:var(--t3);letter-spacing:1px}
.mtf-tf-dir{font-family:'JetBrains Mono';font-size:11px;font-weight:700}
.mtf-tf-arrow{font-size:12px;display:block;margin-top:1px}
/* Big scores */
.big-scores{
  display:flex;align-items:center;justify-content:center;gap:20px;
  margin-bottom:8px;padding:8px;background:var(--bg2);border:1px solid var(--bdr);
  border-radius:3px;
}
.big-score-item{text-align:center}
.big-num{font-family:'JetBrains Mono';font-size:28px;font-weight:800;line-height:1}
.big-lbl{font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:1px;margin-top:2px}
.big-sep{width:1px;height:40px;background:var(--bdr2)}
/* Intel indicators */
.intel-row{
  display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px;
  font-family:'JetBrains Mono';font-size:9px;color:var(--t3);
}
.intel-val{font-weight:600}
/* Score breakdown */
.sb-row{display:flex;align-items:center;margin-bottom:3px;gap:5px}
.sb-label{
  font-family:'JetBrains Mono';font-size:9px;color:var(--t3);
  width:80px;text-align:right;flex-shrink:0;
}
.sb-bar{flex:1;height:4px;background:rgba(255,255,255,0.04);border-radius:1px;overflow:hidden}
.sb-fill{height:100%;transition:width 0.4s;border-radius:1px}
.sb-val{font-family:'JetBrains Mono';font-size:9px;color:var(--t2);width:32px;flex-shrink:0}
/* Intel subsections */
.intel-section{margin-bottom:6px}
.intel-section-title{
  font-family:'JetBrains Mono';font-size:8px;color:var(--t3);text-transform:uppercase;
  letter-spacing:1px;margin-bottom:4px;
}
.intel-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.intel-cell{
  background:var(--bg2);border:1px solid var(--bdr);padding:4px 6px;border-radius:2px;
  font-family:'JetBrains Mono';font-size:9px;
}
.intel-cell-lbl{color:var(--t3);font-size:8px}
.intel-cell-val{font-weight:600;margin-top:1px}
/* Fib levels */
.fib-levels{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:4px}
.fib-chip{
  font-family:'JetBrains Mono';font-size:8px;padding:1px 5px;
  border:1px solid var(--bdr);border-radius:2px;color:var(--t2);background:var(--bg2);
}
/* Candle patterns */
.pattern-chip{
  font-family:'JetBrains Mono';font-size:8px;padding:1px 5px;
  border-radius:2px;display:inline-block;margin:1px 2px;
}
.pat-bullish{color:var(--green);background:var(--green-bg);border:1px solid rgba(0,214,143,0.2)}
.pat-bearish{color:var(--red);background:var(--red-bg);border:1px solid rgba(255,68,102,0.2)}
.pat-neutral{color:var(--t3);background:rgba(255,255,255,0.03);border:1px solid var(--bdr)}

/* ============================================================
   PERFORMANCE PANEL (bottom-right)
   ============================================================ */
.perf-content{padding:10px 12px}
#equity-chart-container{height:120px;width:100%;margin-bottom:8px}
.perf-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:8px}
.perf-stat{
  background:var(--bg2);border:1px solid var(--bdr);padding:6px;border-radius:2px;
  position:relative;overflow:hidden;
}
.perf-stat::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.ps-g::before{background:var(--green)}.ps-r::before{background:var(--red)}
.ps-c::before{background:var(--cyan)}.ps-a::before{background:var(--amber)}
.ps-b::before{background:var(--blue)}.ps-p::before{background:var(--purple)}
.ps-lbl{font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:0.5px}
.ps-val{font-family:'JetBrains Mono';font-size:12px;font-weight:600;margin-top:1px}
/* Per-symbol table */
.sym-table{width:100%;border-collapse:collapse;font-size:10px}
.sym-table th{
  padding:3px 6px;text-align:left;font-family:'JetBrains Mono';font-size:8px;
  font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:0.5px;
  border-bottom:1px solid var(--bdr);background:var(--bg1);position:sticky;top:0;z-index:1;
}
.sym-table td{
  padding:3px 6px;border-bottom:1px solid rgba(255,255,255,0.02);font-family:'JetBrains Mono';
}
.sym-table tr:hover td{background:rgba(0,200,224,0.02)}
/* Learning badges */
.risk-badge{
  font-family:'JetBrains Mono';font-size:8px;font-weight:600;padding:1px 4px;
  border-radius:2px;
}
.risk-up{color:var(--green);background:var(--green-bg);border:1px solid rgba(0,214,143,0.2)}
.risk-down{color:var(--red);background:var(--red-bg);border:1px solid rgba(255,68,102,0.2)}
.risk-neutral{color:var(--t3);background:rgba(255,255,255,0.03);border:1px solid var(--bdr)}

/* ============================================================
   TRADE LOG (collapsible bottom)
   ============================================================ */
.trade-log-wrap{
  position:fixed;bottom:0;left:0;right:0;z-index:50;
  background:var(--bg1);border-top:1px solid var(--bdr);
  transition:max-height 0.3s;overflow:hidden;
}
.trade-log-wrap.collapsed{max-height:32px}
.trade-log-wrap.expanded{max-height:280px}
.trade-log-header{
  padding:6px 16px;display:flex;justify-content:space-between;align-items:center;
  cursor:pointer;border-bottom:1px solid var(--bdr);
}
.trade-log-header:hover{background:rgba(0,200,224,0.02)}
.trade-log-body{overflow-y:auto;max-height:248px}
.tl-table{width:100%;border-collapse:collapse}
.tl-table th{
  padding:4px 10px;text-align:left;font-family:'JetBrains Mono';font-size:8px;
  font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:0.5px;
  border-bottom:1px solid var(--bdr);background:var(--bg1);position:sticky;top:0;z-index:1;
}
.tl-table td{
  padding:3px 10px;border-bottom:1px solid rgba(255,255,255,0.02);
  font-family:'JetBrains Mono';font-size:10px;
}
.tl-table tr:hover td{background:rgba(0,200,224,0.02)}
.tag{
  display:inline-block;padding:1px 5px;font-family:'JetBrains Mono';
  font-size:8px;font-weight:600;border-radius:2px;
}
.tag-long{color:var(--green);background:var(--green-bg);border:1px solid rgba(0,214,143,0.2)}
.tag-short{color:var(--red);background:var(--red-bg);border:1px solid rgba(255,68,102,0.2)}
.tag-flat{color:var(--t3);background:rgba(255,255,255,0.03);border:1px solid var(--bdr)}
.tl-pager{display:flex;gap:4px;align-items:center}
.tl-pg-btn{
  padding:2px 8px;font-family:'JetBrains Mono';font-size:8px;font-weight:600;
  cursor:pointer;border:1px solid var(--bdr);border-radius:2px;
  background:transparent;color:var(--t3);transition:all 0.15s;
}
.tl-pg-btn:hover{color:var(--t2);border-color:var(--bdr2)}
.tl-pg-btn:disabled{opacity:0.3;cursor:default}

/* ============================================================
   MODAL
   ============================================================ */
.modal-overlay{
  position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:10000;
  display:flex;align-items:center;justify-content:center;
}
.modal-box{
  background:var(--bg1);border:1px solid var(--bdr2);padding:24px 32px;
  text-align:center;border-radius:4px;max-width:380px;
}
.modal-box h3{font-family:'JetBrains Mono';color:var(--cyan);font-size:13px;letter-spacing:1px;margin-bottom:12px}
.modal-box p{color:var(--t2);font-size:12px;margin-bottom:16px}
.modal-btns{display:flex;gap:10px;justify-content:center}
.modal-btns button{
  padding:6px 20px;font-family:'JetBrains Mono';font-size:10px;font-weight:600;
  cursor:pointer;border:1px solid;border-radius:3px;
}
.modal-confirm{color:var(--red);background:var(--red-bg);border-color:rgba(255,68,102,0.3)}
.modal-confirm:hover{background:rgba(255,68,102,0.15)}
.modal-cancel{color:var(--t3);background:var(--bg2);border-color:var(--bdr2)}
.modal-cancel:hover{background:var(--bg3)}

/* ============================================================
   RESPONSIVE
   ============================================================ */
@media(max-width:1200px){
  .main-grid{grid-template-columns:1fr;grid-template-rows:auto}
  .hdr{flex-wrap:wrap;height:auto;padding:6px 10px;gap:6px}
  .hdr-stats{flex-wrap:wrap;gap:8px}
}
</style>
</head>
<body>
<div id="app">

<!-- ============ MODAL ============ -->
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

<!-- ============ HEADER ============ -->
<header class="hdr">
  <div class="hdr-left">
    <div>
      <div class="logo">DRAGON</div>
      <div class="logo-sub">TRADING TERMINAL</div>
    </div>
  </div>

  <div class="hdr-badges">
    <div class="pill pill-mode">{{ mode }}</div>
    <div class="pill pill-session" :style="{color:sessionColor,borderColor:sessionColor,background:sessionColor+'12'}">{{ session }}</div>
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
      <div class="val" :class="floatPnl>=0?'g':'r'">{{ fmtPnl(floatPnl) }}</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Daily P&amp;L</div>
      <div class="val" :class="dailyPnl>=0?'g':'r'">{{ fmtPnl(dailyPnl) }}</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Positions</div>
      <div class="val cy">{{ numPositions }}</div>
    </div>
    <div class="hdr-stat">
      <div class="lbl">Risk%</div>
      <div class="val" :class="riskPct>5?'r':riskPct>2?'am':'g'">{{ fmtNum(riskPct,1) }}%</div>
    </div>
  </div>

  <div class="hdr-actions">
    <button class="act-btn act-red" @click="showModal('CLOSE ALL','Close ALL open positions?','close_all')">CLOSE ALL</button>
    <button class="act-btn act-amber" @click="showModal('CLOSE LOSING','Close all LOSING positions?','close_losing')">CLOSE LOSING</button>
    <select class="act-select" v-model="closeSymSelect">
      <option value="">Symbol...</option>
      <option v-for="s in symbols" :key="s" :value="s">{{ s }}</option>
    </select>
    <button class="act-btn act-red" @click="doCloseSym">CLOSE</button>
  </div>

  <div class="hdr-right">
    <div class="status-dot" :class="connected?'on':'off'" :title="connected?'Connected':'Disconnected'"></div>
    <div class="clock">{{ clock }}</div>
  </div>
</header>

<!-- ============ MAIN GRID ============ -->
<div class="main-grid" :style="{height: tradeLogOpen ? 'calc(100vh - 48px - 280px)' : 'calc(100vh - 48px - 32px)'}">

  <!-- === CHART (top-left) === -->
  <div class="panel chart-panel">
    <div class="chart-controls">
      <div>
        <button v-for="s in symbols" :key="s"
          class="sym-tab" :class="{active:selectedSymbol===s}"
          @click="selectSymbol(s)">{{ s }}</button>
      </div>
      <div class="tf-sep"></div>
      <div>
        <button v-for="t in timeframes" :key="t.v"
          class="tf-btn" :class="{active:selectedTF===t.v}"
          @click="selectTF(t.v)">{{ t.l }}</button>
      </div>
    </div>
    <div id="chart-container"></div>
  </div>

  <!-- === MARKET SCANNER (top-right) === -->
  <div class="panel">
    <div class="panel-h">
      <span class="panel-t">Market Scanner</span>
      <span class="panel-badge">{{ symbols.length }} symbols</span>
    </div>
    <div class="panel-b">
      <div v-for="sym in symbols" :key="sym"
        class="sym-card" :class="{selected:selectedSymbol===sym}"
        @click="selectSymbol(sym)">

        <!-- Row 1: Name + Price -->
        <div class="sym-row1">
          <span>
            <span class="sym-name">{{ sym }}</span>
            <span class="sym-cat" :class="'cat-'+getCat(sym)">{{ getCat(sym) }}</span>
          </span>
          <span class="sym-price">
            {{ getTickVal(sym,'bid')!=null ? fmtPrice(getTickVal(sym,'bid'),getDigits(sym)) : '---' }}
            <span v-if="getTickVal(sym,'bid')!=null" class="sym-arrow" :class="isPriceUp(sym)?'up':'dn'">
              {{ isPriceUp(sym) ? '\u25B2' : '\u25BC' }}
            </span>
          </span>
        </div>

        <!-- Spread -->
        <div class="sym-row2">
          <span class="sym-detail">Spd <span>{{ getTickVal(sym,'spread')!=null ? fmtNum(getTickVal(sym,'spread'),1) : '---' }}</span></span>
          <span class="sym-detail">Dir <span :style="{color:dirColor(getScore(sym,'direction'))}">{{ (getScore(sym,'direction')||'FLAT') }}</span></span>
        </div>

        <!-- H1 Score bar -->
        <div class="score-bar-wrap">
          <span class="score-label-sm">H1 Score</span>
          <div class="score-bar"><div class="score-fill" :style="{width:h1ScorePct(sym)+'%',background:h1ScoreColor(sym)}"></div></div>
          <span class="score-val-sm">{{ fmtNum(h1ScoreVal(sym),1) }}</span>
        </div>

        <!-- MTF confluence dots -->
        <div class="mtf-row" v-if="getMtf(sym)">
          <span class="mtf-dot-lbl">MTF</span>
          <div class="mtf-dot" :class="mtfDotClass(getMtf(sym,'h1_dir'))" title="H1"></div>
          <div class="mtf-dot" :class="mtfDotClass(getMtf(sym,'m15_dir'))" title="M15"></div>
          <div class="mtf-dot" :class="mtfDotClass(getMtf(sym,'m5_dir'))" title="M5"></div>
          <div class="mtf-dot" :class="mtfDotClass(getMtf(sym,'m1_dir'))" title="M1"></div>
          <span class="mono" style="font-size:9px;font-weight:600;margin-left:2px" :class="confClass(getMtf(sym,'confluence')||0)">{{ getMtf(sym,'confluence')||0 }}/4</span>
          <!-- Entry quality thin bar -->
          <div class="eq-bar-sm" style="width:40px">
            <div class="eq-fill" :style="{width:(getMtf(sym,'entry_quality')||0)+'%',background:eqColor(getMtf(sym,'entry_quality')||0)}"></div>
          </div>
          <span class="mono" style="font-size:8px;font-weight:600" :style="{color:eqColor(getMtf(sym,'entry_quality')||0)}">{{ getMtf(sym,'entry_quality')||0 }}</span>
        </div>

        <!-- Position or FLAT -->
        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px">
          <span v-if="!posMap[sym]" class="pos-badge pos-flat">FLAT</span>
          <span v-else style="display:flex;align-items:center">
            <span class="pos-badge" :class="posMap[sym].side==='BUY'?'pos-long':'pos-short'">
              {{ posMap[sym].side==='BUY'?'LONG':'SHORT' }}
            </span>
            <span style="margin-left:4px;font-weight:700;font-size:11px"
              :style="{color: posMap[sym].pnl>=0 ? 'var(--green)' : 'var(--red)'}">
              {{ posMap[sym].pnl>=0?'+':'' }}${{ fmtNum(posMap[sym].pnl,2) }}
            </span>
            <span v-if="getMtf(sym,'exit_urgency')>0" class="exit-badge"
              :class="exitBadgeClass(getMtf(sym,'exit_urgency'))">
              EXIT {{ fmtNum(getMtf(sym,'exit_urgency')*100,0) }}%
            </span>
          </span>
          <canvas :ref="el=>{if(el) sparkRefs[sym]=el}" width="60" height="16" style="opacity:0.6"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- === INTELLIGENCE (bottom-left) === -->
  <div class="panel">
    <div class="panel-h">
      <span class="panel-t">Intelligence</span>
      <span class="panel-badge">{{ selectedSymbol }}</span>
    </div>
    <div class="panel-b">
      <div class="intel-content">

        <!-- MasterBrain Status -->
        <div v-if="masterBrain" class="intel-section">
          <div class="intel-section-title">MasterBrain</div>
          <div class="mb-grid">
            <div class="mb-cell">
              <div class="mb-lbl">Eq Health</div>
              <div class="mb-val" :style="{color:mbHealthColor}">{{ (masterBrain.equity_health||'unknown').toUpperCase() }}</div>
            </div>
            <div class="mb-cell">
              <div class="mb-lbl">Daily Trades</div>
              <div class="mb-val">{{ masterBrain.daily_trades||0 }}</div>
            </div>
            <div class="mb-cell">
              <div class="mb-lbl">Win Rate</div>
              <div class="mb-val" :style="{color:(masterBrain.win_rate||0)>=50?'var(--green)':'var(--red)'}">{{ fmtNum(masterBrain.win_rate||0,1) }}%</div>
            </div>
            <div class="mb-cell">
              <div class="mb-lbl">Circuit</div>
              <div class="mb-val" :style="{color:masterBrain.session_paused?'var(--red)':'var(--green)'}">{{ masterBrain.session_paused ? 'PAUSED' : 'OK' }}</div>
            </div>
          </div>
          <div class="mb-info">
            <span>Daily P&L: <span :style="{color:(masterBrain.daily_pnl||0)>=0?'var(--green)':'var(--red)'}">${{ fmtNum(masterBrain.daily_pnl||0,2) }}</span></span>
            <span>Last 10: <span :style="{color:(masterBrain.recent_10_pnl||0)>=0?'var(--green)':'var(--red)'}">${{ fmtNum(masterBrain.recent_10_pnl||0,2) }}</span></span>
            <span>Total: <span class="cy">{{ masterBrain.total_trades||0 }}</span></span>
            <span>Session Losses: <span :style="{color:(masterBrain.session_losses||0)>=2?'var(--red)':'var(--t2)'}">{{ masterBrain.session_losses||0 }}/2</span></span>
          </div>
          <div v-if="mbBlacklistStr!=='None'" style="margin-top:3px;font-size:9px;font-family:'JetBrains Mono';color:var(--red)">
            Blacklisted: {{ mbBlacklistStr }}
          </div>
        </div>

        <div class="sep"></div>

        <!-- Selected Symbol MTF Intelligence -->
        <div v-if="selectedMtf" class="intel-section">
          <div class="intel-section-title">MTF Intelligence: {{ selectedSymbol }}</div>

          <!-- 4-TF Direction Grid -->
          <div class="mtf-tf-grid">
            <div v-for="tf in [{k:'h1_dir',l:'H1'},{k:'m15_dir',l:'M15'},{k:'m5_dir',l:'M5'},{k:'m1_dir',l:'M1'}]" :key="tf.k"
              class="mtf-tf-cell" :class="'tf-'+(selectedMtf[tf.k]||'FLAT').toLowerCase()">
              <div class="mtf-tf-lbl">{{ tf.l }}</div>
              <div class="mtf-tf-dir" :style="{color:mtfDirColor(selectedMtf[tf.k])}">{{ (selectedMtf[tf.k]||'FLAT') }}</div>
              <span class="mtf-tf-arrow" :style="{color:mtfDirColor(selectedMtf[tf.k])}">{{ mtfArrow(selectedMtf[tf.k]) }}</span>
            </div>
          </div>

          <!-- Big Numbers: Confluence | Entry Quality | Exit Urgency -->
          <div class="big-scores">
            <div class="big-score-item">
              <div class="big-num" :class="confClass(selectedMtf.confluence||0)">{{ selectedMtf.confluence||0 }}</div>
              <div class="big-lbl">Confluence</div>
            </div>
            <div class="big-sep"></div>
            <div class="big-score-item">
              <div class="big-num" :style="{color:eqColor(selectedMtf.entry_quality||0)}">{{ selectedMtf.entry_quality||0 }}</div>
              <div class="big-lbl">Entry Quality</div>
            </div>
            <div class="big-sep"></div>
            <div class="big-score-item">
              <div class="big-num" :style="{color:exitUrgencyColor(selectedMtf.exit_urgency||0),fontSize:'22px'}">{{ fmtNum((selectedMtf.exit_urgency||0)*100,0) }}%</div>
              <div class="big-lbl">Exit Urgency</div>
            </div>
          </div>

          <!-- Regime / Volume / Swing / Momentum / Order Flow -->
          <div class="intel-row">
            <span>Regime: <span class="intel-val am">{{ (getScore(selectedSymbol,'regime')||'unknown').toUpperCase() }}</span></span>
            <span v-if="selectedMtf.volume_trend">Vol: <span class="intel-val" :style="{color:selectedMtf.volume_trend==='bullish'?'var(--green)':selectedMtf.volume_trend==='bearish'?'var(--red)':'var(--t3)'}">{{ (selectedMtf.volume_trend||'flat').toUpperCase() }}</span></span>
            <span v-if="selectedMtf.swing_structure">Swing: <span class="intel-val" :style="{color:selectedMtf.swing_structure==='uptrend'?'var(--green)':selectedMtf.swing_structure==='downtrend'?'var(--red)':'var(--t3)'}">{{ (selectedMtf.swing_structure||'sideways').toUpperCase() }}</span></span>
            <span v-if="selectedMtf.momentum!=null">Mom: <span class="intel-val" :style="{color:selectedMtf.momentum>0?'var(--green)':selectedMtf.momentum<0?'var(--red)':'var(--t3)'}">{{ fmtNum(selectedMtf.momentum,2) }}</span></span>
            <span v-if="selectedMtf.order_flow!=null">Flow: <span class="intel-val" :style="{color:selectedMtf.order_flow>0?'var(--green)':selectedMtf.order_flow<0?'var(--red)':'var(--t3)'}">{{ fmtNum(selectedMtf.order_flow,2) }}</span></span>
          </div>

          <!-- Fibonacci Levels -->
          <div v-if="selectedMtf.fibonacci && selectedMtf.fibonacci.levels" class="intel-section" style="margin-top:4px">
            <div class="intel-section-title">Fibonacci Levels</div>
            <div class="fib-levels">
              <span v-for="(price,level) in selectedMtf.fibonacci.levels" :key="level" class="fib-chip">
                {{ level }}: {{ fmtNum(price,2) }}
              </span>
              <span v-if="selectedMtf.fibonacci.nearest_fib_dist!=null" class="fib-chip" style="color:var(--amber)">
                Dist: {{ fmtNum(selectedMtf.fibonacci.nearest_fib_dist,4) }}
              </span>
            </div>
          </div>

          <!-- Candle Patterns -->
          <div v-if="hasPatterns" class="intel-section">
            <div class="intel-section-title">Candle Patterns</div>
            <div>
              <template v-if="selectedMtf.candle_patterns_h1 && selectedMtf.candle_patterns_h1.patterns">
                <span v-for="p in selectedMtf.candle_patterns_h1.patterns" :key="'h1_'+p.name"
                  class="pattern-chip" :class="p.bias==='bullish'?'pat-bullish':p.bias==='bearish'?'pat-bearish':'pat-neutral'">
                  H1: {{ p.name }}
                </span>
              </template>
              <template v-if="selectedMtf.candle_patterns_m15 && selectedMtf.candle_patterns_m15.patterns">
                <span v-for="p in selectedMtf.candle_patterns_m15.patterns" :key="'m15_'+p.name"
                  class="pattern-chip" :class="p.bias==='bullish'?'pat-bullish':p.bias==='bearish'?'pat-bearish':'pat-neutral'">
                  M15: {{ p.name }}
                </span>
              </template>
            </div>
          </div>

          <!-- Divergence -->
          <div v-if="selectedMtf.mtf_divergence" class="intel-row" style="margin-top:4px">
            <span>Divergence: <span class="intel-val" :style="{color:selectedMtf.mtf_divergence.combined>0?'var(--red)':selectedMtf.mtf_divergence.combined<0?'var(--green)':'var(--t3)'}">
              {{ selectedMtf.mtf_divergence.combined>0?'BEARISH':selectedMtf.mtf_divergence.combined<0?'BULLISH':'NONE' }}
              <template v-if="selectedMtf.mtf_divergence.strength"> ({{ fmtNum(selectedMtf.mtf_divergence.strength,2) }})</template>
            </span></span>
          </div>

          <!-- Volatility Cycle + Correlation + Mean Reversion + Best TF + M1 Noise -->
          <div class="intel-grid" style="margin-top:4px">
            <div v-if="selectedMtf.volatility_cycle" class="intel-cell">
              <div class="intel-cell-lbl">Vol Cycle</div>
              <div class="intel-cell-val" :style="{color:selectedMtf.volatility_cycle.phase==='expanding'?'var(--amber)':selectedMtf.volatility_cycle.phase==='contracting'?'var(--cyan)':'var(--t3)'}">
                {{ (selectedMtf.volatility_cycle.phase||'stable').toUpperCase() }}
              </div>
            </div>
            <div v-if="selectedMtf.correlation_regime" class="intel-cell">
              <div class="intel-cell-lbl">Corr Regime</div>
              <div class="intel-cell-val cy">{{ typeof selectedMtf.correlation_regime==='object'?(selectedMtf.correlation_regime.regime||'normal').toUpperCase():(selectedMtf.correlation_regime||'').toUpperCase() }}</div>
            </div>
            <div v-if="selectedMtf.mean_reversion" class="intel-cell">
              <div class="intel-cell-lbl">Mean Rev</div>
              <div class="intel-cell-val" :style="{color:selectedMtf.mean_reversion.active?'var(--amber)':'var(--t3)'}">
                {{ selectedMtf.mean_reversion.active?'ACTIVE':'OFF' }}
                <template v-if="selectedMtf.mean_reversion.zscore!=null"> z={{ fmtNum(selectedMtf.mean_reversion.zscore,1) }}</template>
              </div>
            </div>
            <div v-if="selectedMtf.best_timeframe" class="intel-cell">
              <div class="intel-cell-lbl">Best TF</div>
              <div class="intel-cell-val cy">{{ typeof selectedMtf.best_timeframe==='object'?(selectedMtf.best_timeframe.tf||'H1'):selectedMtf.best_timeframe }}</div>
            </div>
            <div v-if="selectedMtf.m1_noise" class="intel-cell">
              <div class="intel-cell-lbl">M1 Noise</div>
              <div class="intel-cell-val" :style="{color:(selectedMtf.m1_noise.noise_level||0)>0.5?'var(--red)':'var(--green)'}">
                {{ fmtNum((selectedMtf.m1_noise.noise_level||0)*100,0) }}%
              </div>
            </div>
            <div v-if="selectedMtf.liquidity" class="intel-cell">
              <div class="intel-cell-lbl">Liquidity</div>
              <div class="intel-cell-val" :style="{color:selectedMtf.liquidity.at_liquidity?'var(--amber)':'var(--green)'}">
                {{ selectedMtf.liquidity.at_liquidity?'AT ZONE':'CLEAR' }}
              </div>
            </div>
          </div>
        </div>

        <div class="sep"></div>

        <!-- Signal Breakdown for selected symbol -->
        <div class="intel-section">
          <div class="intel-section-title">{{ selectedSymbol }} Signal Breakdown
            <span v-if="selectedGate==='SESSION'" style="color:var(--red);margin-left:6px">SESSION CLOSED</span>
          </div>
          <div class="sb-row" v-for="m in breakdownMetrics" :key="m.label">
            <div class="sb-label">{{ m.label }}</div>
            <div class="sb-bar"><div class="sb-fill" :style="{width:m.pct+'%',background:m.color}"></div></div>
            <div class="sb-val">{{ fmtNum(m.val,2) }}</div>
          </div>
          <div class="sb-row" v-if="selectedScores.meta_prob!=null">
            <div class="sb-label">ML Meta</div>
            <div class="sb-bar"><div class="sb-fill" :style="{width:Math.min(100,selectedScores.meta_prob*100)+'%',background:selectedScores.meta_prob>0.6?'var(--green)':selectedScores.meta_prob>0.4?'var(--amber)':'var(--red)'}"></div></div>
            <div class="sb-val">{{ fmtNum(selectedScores.meta_prob,3) }}</div>
          </div>
          <div class="sb-row" v-if="selectedMLAuc>0">
            <div class="sb-label">ML AUC</div>
            <div class="sb-bar"><div class="sb-fill" :style="{width:Math.min(100,Math.max(0,(selectedMLAuc-0.5)/0.5)*100)+'%',background:selectedMLAuc>=0.70?'var(--green)':selectedMLAuc>=0.60?'var(--amber)':'var(--red)'}"></div></div>
            <div class="sb-val">{{ fmtNum(selectedMLAuc,3) }}</div>
          </div>
          <!-- Direction / Gate summary -->
          <div style="margin-top:6px;display:flex;gap:8px;font-size:9px;font-family:'JetBrains Mono';flex-wrap:wrap">
            <span class="dim">Dir: <span :style="{color:dirColor(selectedScores.direction)}">{{ selectedScores.direction||'FLAT' }}</span></span>
            <span class="dim">M15: <span class="cy">{{ (selectedScores.m15_dir||'flat').toUpperCase() }}</span></span>
            <span class="dim">Gate: <span :style="{color:selectedGate==='ENTERED'?'var(--green)':'var(--amber)'}">{{ selectedGate||'\u2014' }}</span></span>
            <span v-if="selectedScores.master_reason" class="r" style="font-size:9px">Reject: {{ selectedScores.master_reason }}</span>
          </div>
        </div>

      </div>
    </div>
  </div>

  <!-- === PERFORMANCE (bottom-right) === -->
  <div class="panel">
    <div class="panel-h">
      <span class="panel-t">Performance</span>
      <span class="panel-badge">{{ tradeLog.length }} trades</span>
    </div>
    <div class="panel-b">
      <div class="perf-content">
        <!-- Equity Curve -->
        <div id="equity-chart-container"></div>

        <!-- Stats Grid -->
        <div class="perf-stats">
          <div class="perf-stat ps-g">
            <div class="ps-lbl">Win Rate</div>
            <div class="ps-val" :class="perfStats.wr>=50?'g':'r'">{{ perfStats.wrStr }}</div>
          </div>
          <div class="perf-stat ps-c">
            <div class="ps-lbl">Profit Factor</div>
            <div class="ps-val" :class="perfStats.pf>=1?'g':'r'">{{ perfStats.pfStr }}</div>
          </div>
          <div class="perf-stat ps-b">
            <div class="ps-lbl">Sharpe</div>
            <div class="ps-val" :class="perfStats.sharpe>=0?'g':'r'">{{ perfStats.sharpeStr }}</div>
          </div>
          <div class="perf-stat ps-a">
            <div class="ps-lbl">Avg R</div>
            <div class="ps-val" :class="perfStats.avgR>=0?'g':'r'">{{ perfStats.avgRStr }}</div>
          </div>
          <div class="perf-stat ps-r">
            <div class="ps-lbl">Max DD</div>
            <div class="ps-val" :class="perfStats.maxDD>5?'r':'cy'">{{ perfStats.maxDDStr }}</div>
          </div>
          <div class="perf-stat ps-p">
            <div class="ps-lbl">Total</div>
            <div class="ps-val cy">{{ tradeLog.length }}</div>
          </div>
        </div>

        <!-- Per-Symbol P&L Table -->
        <div class="intel-section">
          <div class="intel-section-title">Per-Symbol Breakdown</div>
          <table class="sym-table">
            <tr><th>Symbol</th><th>Trades</th><th>W</th><th>L</th><th>P&amp;L</th><th>WR%</th></tr>
            <tr v-for="s in perSymbolStats" :key="s.symbol">
              <td class="bright">{{ s.symbol }}</td>
              <td class="dim">{{ s.trades }}</td>
              <td class="g">{{ s.wins }}</td>
              <td class="r">{{ s.losses }}</td>
              <td class="mono" :class="s.pnl>=0?'g':'r'" style="font-weight:600">${{ fmtNum(s.pnl,2) }}</td>
              <td class="mono" :class="s.wr>=50?'g':'r'">{{ fmtNum(s.wr,0) }}%</td>
            </tr>
          </table>
        </div>

        <!-- Learning Engine Badges -->
        <div v-if="hasLearningStats" class="intel-section" style="margin-top:6px">
          <div class="intel-section-title">Learning Engine (Risk Multipliers)</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap">
            <div v-for="sym in symbols" :key="'lr_'+sym" style="display:flex;align-items:center;gap:3px">
              <span class="mono dim" style="font-size:9px">{{ sym }}</span>
              <span class="risk-badge" :class="learnRiskClass(sym)">{{ learnRiskVal(sym) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div><!-- main-grid -->

<!-- ============ TRADE LOG (collapsible bottom) ============ -->
<div class="trade-log-wrap" :class="tradeLogOpen?'expanded':'collapsed'">
  <div class="trade-log-header" @click="tradeLogOpen=!tradeLogOpen">
    <span class="panel-t" style="font-size:9px">Trade Log ({{ tradeLog.length }})</span>
    <div style="display:flex;align-items:center;gap:8px">
      <div v-if="tradeLog.length>0" class="tl-pager" @click.stop>
        <span class="dim mono" style="font-size:8px">{{ tradePageStart+1 }}-{{ Math.min(tradePageStart+tradePageSize,tradeLog.length) }}</span>
        <button class="tl-pg-btn" @click="tradePage=Math.max(0,tradePage-1)" :disabled="tradePage===0">&lt;</button>
        <button class="tl-pg-btn" @click="tradePage++" :disabled="tradePageStart+tradePageSize>=tradeLog.length">&gt;</button>
      </div>
      <span class="mono dim" style="font-size:10px">{{ tradeLogOpen ? '\u25BC' : '\u25B2' }}</span>
    </div>
  </div>
  <div class="trade-log-body" v-if="tradeLogOpen">
    <table v-if="tradeLog.length>0" class="tl-table">
      <tr><th>Time</th><th>Symbol</th><th>Dir</th><th>Volume</th><th>P&amp;L</th><th>Exit Reason</th></tr>
      <tr v-for="(t,i) in tradeLogPaged" :key="i">
        <td class="dim">{{ t.timestamp||t.time||'' }}</td>
        <td class="bright">{{ t.symbol||'' }}</td>
        <td><span class="tag" :class="tradeTagClass(t)">{{ tradeDir(t) }}</span></td>
        <td class="dim">{{ t.volume ? t.volume.toFixed(2) : '' }}</td>
        <td :class="(t.pnl||0)>=0?'g':'r'" style="font-weight:600">${{ fmtPnl(t.pnl||t.profit||0) }}</td>
        <td class="dim" style="font-size:9px;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ t.action||'' }}</td>
      </tr>
    </table>
    <div v-else class="empty">No recent trades</div>
  </div>
</div>

</div><!-- #app -->

<script>
// ============================================================
// DRAGON TRADER — Vue 3 Reactive Dashboard Engine
// ============================================================

const SYMBOLS = """ + SYMBOL_LIST_JSON + r""";
const SYMBOL_META = """ + SYMBOL_META_JSON + r""";
const STARTING_BAL = """ + str(STARTING_BAL) + r""";

const {createApp,ref,reactive,computed,watch,onMounted,nextTick,toRaw} = Vue;

const app = createApp({
  setup() {
    // ── CORE STATE ──
    const symbols = ref(SYMBOLS);
    const selectedSymbol = ref(SYMBOLS[0]);
    const selectedTF = ref(5);
    const connected = ref(false);
    const clock = ref('00:00:00 IST');
    const closeSymSelect = ref('');

    // ── HEADER ──
    const mode = ref('HYBRID');
    const session = ref('---');
    const sessionColor = ref('#00c8e0');
    const balance = ref(0);
    const equity = ref(0);
    const floatPnl = ref(0);
    const dailyPnl = ref(0);
    const todayClosedPnl = ref(0);
    const numPositions = ref(0);
    const riskPct = ref(0);

    // ── TRADE LOG ──
    const tradePage = ref(0);
    const tradePageSize = 20;
    const tradeLogOpen = ref(false);

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
    const learningStats = reactive({});
    const chartDataStore = reactive({});

    // ── MODAL ──
    const modal = reactive({show:false,title:'',msg:'',action:'',data:''});
    const sparkRefs = reactive({});
    const timeframes = ref([{v:1,l:'M1'},{v:5,l:'M5'},{v:15,l:'M15'},{v:60,l:'H1'}]);

    // ═══════════════════════════════════════
    // FORMATTING
    // ═════════��═════════════════════════════
    function fmtNum(v,d){if(v==null||isNaN(v))return'---';return Number(v).toFixed(d||2)}
    function fmtDollar(v){if(v==null)return'---';return'$'+Number(v).toFixed(2)}
    function fmtPnl(v){if(v==null)return'---';const n=Number(v);return(n>=0?'+$':'-$')+Math.abs(n).toFixed(2)}
    function fmtPrice(v,d){if(v==null)return'---';return Number(v).toFixed(d||2)}

    // ═══════════════════════════════════════
    // SYMBOL HELPERS
    // ══���═══════════════════���════════════════
    function getCat(sym){return SYMBOL_META[sym]?SYMBOL_META[sym].category:'Forex'}
    function getDigits(sym){return SYMBOL_META[sym]?SYMBOL_META[sym].digits:2}
    function getTickVal(sym,f){return ticks[sym]?ticks[sym][f]:null}
    function isPriceUp(sym){
      const bid=getTickVal(sym,'bid'), prev=prevPrices[sym];
      if(bid==null||prev==null)return true;
      return bid>=prev;
    }
    function getScore(sym,f){return scores[sym]?scores[sym][f]:null}
    function getML(sym,f){return mlConfidence[sym]?mlConfidence[sym][f]:null}

    function dirColor(dir){
      if(!dir)return'var(--t3)';
      if(dir==='LONG')return'var(--green)';
      if(dir==='SHORT')return'var(--red)';
      return'var(--t3)';
    }

    // ═══════════════════════════════════════
    // SCANNER COMPUTED
    // ═══════════════════════════════════════
    function h1ScoreVal(sym){
      const sc=scores[sym]||{};
      return Math.max(sc.long_score||0, sc.short_score||0);
    }
    function h1ScorePct(sym){return Math.min(100,h1ScoreVal(sym)/14.0*100)}
    function h1ScoreColor(sym){
      const p=h1ScorePct(sym);
      return p>60?'var(--green)':p>30?'var(--amber)':'var(--red)';
    }

    // ═══════════════════════════════════════
    // MTF HELPERS
    // ═══════════════════════════════════════
    function getMtf(sym,field){
      const d=mtfIntelligence[sym];
      if(!d)return field?null:null;
      if(!field)return d;
      return d[field]!=null?d[field]:null;
    }
    function mtfDotClass(dir){
      if(!dir)return'mtf-dot-flat';
      const d=dir.toUpperCase();
      if(d==='LONG')return'mtf-dot-long';
      if(d==='SHORT')return'mtf-dot-short';
      return'mtf-dot-flat';
    }
    function mtfDirColor(dir){
      if(!dir)return'var(--t3)';
      const d=dir.toUpperCase();
      if(d==='LONG')return'var(--green)';if(d==='SHORT')return'var(--red)';return'var(--t3)';
    }
    function mtfArrow(dir){
      if(!dir)return'\u2500';
      const d=dir.toUpperCase();
      if(d==='LONG')return'\u25B2';if(d==='SHORT')return'\u25BC';return'\u2500';
    }
    function confClass(v){return v>=4?'g':v>=3?'g':v>=2?'am':'r'}
    function eqColor(v){return v>60?'var(--green)':v>30?'var(--amber)':'var(--red)'}
    function exitUrgencyColor(v){
      if(v<=0.2)return'var(--green)';if(v<=0.5)return'var(--amber)';return'var(--red)';
    }
    function exitBadgeClass(v){
      if(v<=0.2)return'exit-low';if(v<=0.5)return'exit-med';return'exit-high';
    }

    const selectedMtf = computed(()=>mtfIntelligence[selectedSymbol.value]||null);
    const hasPatterns = computed(()=>{
      const m=selectedMtf.value;
      if(!m)return false;
      const h1=m.candle_patterns_h1, m15=m.candle_patterns_m15;
      return(h1&&h1.patterns&&h1.patterns.length>0)||(m15&&m15.patterns&&m15.patterns.length>0);
    });

    // ═══════════════════════════════════════
    // INTELLIGENCE COMPUTED
    // ══════════════���════════════════════════
    const selectedScores = computed(()=>scores[selectedSymbol.value]||{});
    const selectedGate = computed(()=>selectedScores.value.gate||'');
    const selectedMLAuc = computed(()=>{
      const ml=mlConfidence[selectedSymbol.value];
      return ml?(ml.auc||0):0;
    });
    const breakdownMetrics = computed(()=>{
      const sc=selectedScores.value;
      return[
        {label:'Long Score',val:sc.long_score||0,pct:Math.min(100,(sc.long_score||0)/14*100),color:'var(--green)'},
        {label:'Short Score',val:sc.short_score||0,pct:Math.min(100,(sc.short_score||0)/14*100),color:'var(--red)'},
        {label:'Adaptive Min',val:sc.adaptive_min_score||7,pct:Math.min(100,(sc.adaptive_min_score||7)/14*100),color:'var(--amber)'},
        {label:'ATR',val:sc.atr||0,pct:100,color:'var(--cyan)'},
      ];
    });

    const mbHealthColor = computed(()=>{
      if(!masterBrain.value)return'var(--t3)';
      const h=masterBrain.value.equity_health||'unknown';
      return h==='healthy'?'var(--green)':h==='flat'?'var(--amber)':'var(--red)';
    });
    const mbBlacklistStr = computed(()=>{
      if(!masterBrain.value)return'None';
      const bl=masterBrain.value.blacklisted_symbols||{};
      const entries=Object.entries(bl);
      return entries.length>0?entries.map(([s,h])=>s+' ('+Number(h).toFixed(1)+'h)').join(', '):'None';
    });

    // ════════���══════════════════════════════
    // TRADE LOG COMPUTED
    // ═════════���═════════════════════════��═══
    const tradePageStart = computed(()=>tradePage.value*tradePageSize);
    const tradeLogPaged = computed(()=>{
      const start=tradePageStart.value;
      return tradeLog.value.slice(start,start+tradePageSize);
    });
    function tradeDir(t){return(t.direction||t.type||'').toUpperCase()}
    function tradeTagClass(t){
      const d=tradeDir(t);
      if(d==='LONG'||d==='BUY')return'tag-long';
      if(d==='SHORT'||d==='SELL')return'tag-short';
      return'tag-flat';
    }

    // ═══════════════════════════════════════
    // PERFORMANCE COMPUTED
    // ═���════════��════════════════════════���═══
    const perfStats = computed(()=>{
      const trades=tradeLog.value;
      if(!trades||trades.length===0){
        return{wr:0,pf:0,sharpe:0,avgR:0,maxDD:0,
          wrStr:'--',pfStr:'--',sharpeStr:'--',avgRStr:'--',maxDDStr:'--'};
      }
      const pnls=trades.map(t=>t.pnl||t.profit||0);
      const wins=pnls.filter(p=>p>0);
      const losses=pnls.filter(p=>p<0);
      const wr=wins.length/pnls.length*100;
      const grossP=wins.reduce((a,b)=>a+b,0);
      const grossL=Math.abs(losses.reduce((a,b)=>a+b,0));
      const pf=grossL>0?grossP/grossL:grossP>0?99.99:0;
      const avgR=pnls.reduce((a,b)=>a+b,0)/pnls.length;
      const mean=avgR;
      const variance=pnls.reduce((a,b)=>a+(b-mean)**2,0)/pnls.length;
      const stddev=Math.sqrt(variance)||1;
      const sharpe=mean/stddev*Math.sqrt(252);
      let peak=STARTING_BAL,maxDD=0,runEq=STARTING_BAL;
      pnls.forEach(p=>{
        runEq+=p;if(runEq>peak)peak=runEq;
        const dd=(peak-runEq)/peak*100;if(dd>maxDD)maxDD=dd;
      });
      return{
        wr,pf,sharpe,avgR,maxDD,
        wrStr:wr.toFixed(1)+'%',pfStr:pf.toFixed(2),
        sharpeStr:sharpe.toFixed(2),avgRStr:'$'+avgR.toFixed(2),
        maxDDStr:maxDD.toFixed(1)+'%',
      };
    });

    // Per-symbol breakdown from trade log
    const perSymbolStats = computed(()=>{
      const trades=tradeLog.value;
      if(!trades||trades.length===0)return[];
      const map={};
      trades.forEach(t=>{
        const sym=t.symbol||'';
        if(!sym)return;
        if(!map[sym])map[sym]={symbol:sym,trades:0,wins:0,losses:0,pnl:0};
        map[sym].trades++;
        const p=t.pnl||t.profit||0;
        map[sym].pnl+=p;
        if(p>0)map[sym].wins++;else if(p<0)map[sym].losses++;
      });
      return Object.values(map).map(s=>({
        ...s,pnl:Math.round(s.pnl*100)/100,
        wr:s.trades>0?s.wins/s.trades*100:0,
      })).sort((a,b)=>b.pnl-a.pnl);
    });

    // Learning engine
    const hasLearningStats = computed(()=>Object.keys(learningStats).length>0);
    function learnRiskVal(sym){
      const s=learningStats[sym];
      if(!s)return'1.0x';
      return(s.risk_mult||1.0).toFixed(1)+'x';
    }
    function learnRiskClass(sym){
      const s=learningStats[sym];
      if(!s)return'risk-neutral';
      const m=s.risk_mult||1.0;
      if(m>1.05)return'risk-up';
      if(m<0.95)return'risk-down';
      return'risk-neutral';
    }

    // ════��══════════════════════════════════
    // ACTIONS
    // ═══════════════════════════════════════
    function selectSymbol(sym){
      selectedSymbol.value=sym;
      socket.emit('select_symbol',{symbol:sym});
      refreshChart();
    }
    function selectTF(tf){
      selectedTF.value=tf;
      socket.emit('select_timeframe',{timeframe:tf});
      refreshChart();
    }
    function showModal(title,msg,action){
      modal.title=title;modal.msg=msg;modal.action=action;modal.show=true;
    }
    function modalConfirm(){
      if(modal.action==='close_all')socket.emit('close_all');
      else if(modal.action==='close_losing')socket.emit('close_losing');
      else if(modal.action==='close_symbol')socket.emit('close_symbol',{symbol:modal.data});
      modal.show=false;
    }
    function doCloseSym(){
      if(closeSymSelect.value){
        showModal('CLOSE SYMBOL','Close '+closeSymSelect.value+'?','close_symbol');
        modal.data=closeSymSelect.value;
      }
    }

    // ════��══════════════════════════════════
    // SPARKLINES
    // ══════════════════��════════════════════
    function drawSparkline(canvas,data,color){
      if(!canvas||!data||data.length<2)return;
      const ctx=canvas.getContext('2d');
      const w=canvas.width,h=canvas.height;
      ctx.clearRect(0,0,w,h);
      const min=Math.min(...data),max=Math.max(...data);
      const range=max-min||1;
      ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=1;
      data.forEach((v,i)=>{
        const x=(i/(data.length-1))*w;
        const y=h-((v-min)/range)*h;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
      });
      ctx.stroke();
    }
    function drawAllSparklines(){
      SYMBOLS.forEach(sym=>{
        const tick=ticks[sym];
        if(tick&&tick.sparkline&&sparkRefs[sym]){
          drawSparkline(sparkRefs[sym],tick.sparkline,'#00c8e0');
        }
      });
    }

    // ═══��═══════════════════════��═══════════
    // LIGHTWEIGHT CHARTS
    // ═════��═════════════════════════════════
    let mainChart=null, candleSeries=null, volumeSeries=null;
    let ema20Series=null, ema50Series=null, ema200Series=null;
    let equityChart=null, equitySeries=null, ddSeries=null;
    let localEquityHist=[];

    function calcEMA(closes,times,period){
      if(closes.length<period)return[];
      const alpha=2/(period+1);const result=[];
      let ema=closes.slice(0,period).reduce((a,b)=>a+b,0)/period;
      for(let i=period-1;i<closes.length;i++){
        if(i===period-1)ema=closes.slice(0,period).reduce((a,b)=>a+b,0)/period;
        else ema=alpha*closes[i]+(1-alpha)*ema;
        result.push({time:times[i],value:parseFloat(ema.toFixed(5))});
      }
      return result;
    }

    function initCharts(){
      const container=document.getElementById('chart-container');
      if(!container)return;

      mainChart=LightweightCharts.createChart(container,{
        width:container.clientWidth, height:container.clientHeight,
        layout:{
          background:{type:'solid',color:'#0a0e17'},
          textColor:'rgba(160,176,196,0.6)',fontSize:10,
          fontFamily:'JetBrains Mono',
        },
        grid:{
          vertLines:{color:'rgba(30,42,58,0.5)'},
          horzLines:{color:'rgba(30,42,58,0.5)'},
        },
        crosshair:{
          mode:LightweightCharts.CrosshairMode.Normal,
          vertLine:{color:'rgba(0,200,224,0.3)',width:1,style:2},
          horzLine:{color:'rgba(0,200,224,0.3)',width:1,style:2},
        },
        rightPriceScale:{borderColor:'rgba(30,42,58,0.8)',scaleMargins:{top:0.1,bottom:0.25}},
        timeScale:{borderColor:'rgba(30,42,58,0.8)',timeVisible:true,secondsVisible:false},
        watermark:{visible:true,text:'DRAGON',fontSize:48,color:'rgba(0,200,224,0.03)',horzAlign:'center',vertAlign:'center'},
      });

      candleSeries=mainChart.addCandlestickSeries({
        upColor:'#00d68f',downColor:'#ff4466',
        borderDownColor:'#ff4466',borderUpColor:'#00d68f',
        wickDownColor:'#ff4466',wickUpColor:'#00d68f',
      });
      volumeSeries=mainChart.addHistogramSeries({
        priceFormat:{type:'volume'},priceScaleId:'volume',color:'rgba(0,200,224,0.1)',
      });
      mainChart.priceScale('volume').applyOptions({scaleMargins:{top:0.8,bottom:0}});

      ema20Series=mainChart.addLineSeries({color:'#00c8e0',lineWidth:1,title:'EMA20',priceLineVisible:false,lastValueVisible:false});
      ema50Series=mainChart.addLineSeries({color:'#3388ff',lineWidth:1,title:'EMA50',priceLineVisible:false,lastValueVisible:false});
      ema200Series=mainChart.addLineSeries({color:'#9966ff',lineWidth:1,title:'EMA200',priceLineVisible:false,lastValueVisible:false});

      // Equity curve chart
      const eqContainer=document.getElementById('equity-chart-container');
      if(!eqContainer)return;
      equityChart=LightweightCharts.createChart(eqContainer,{
        width:eqContainer.clientWidth,height:120,
        layout:{background:{type:'solid',color:'transparent'},textColor:'rgba(160,176,196,0.4)',fontSize:9,fontFamily:'JetBrains Mono'},
        grid:{vertLines:{color:'rgba(30,42,58,0.3)'},horzLines:{color:'rgba(30,42,58,0.3)'}},
        rightPriceScale:{borderColor:'rgba(30,42,58,0.5)'},
        timeScale:{borderColor:'rgba(30,42,58,0.5)',visible:false},
        crosshair:{mode:LightweightCharts.CrosshairMode.Normal},
      });
      equitySeries=equityChart.addAreaSeries({
        lineColor:'#00c8e0',topColor:'rgba(0,200,224,0.12)',bottomColor:'rgba(0,200,224,0.01)',
        lineWidth:2,priceLineVisible:false,
      });
      ddSeries=equityChart.addAreaSeries({
        lineColor:'rgba(255,68,102,0.5)',topColor:'rgba(255,68,102,0.0)',bottomColor:'rgba(255,68,102,0.08)',
        lineWidth:1,priceLineVisible:false,lastValueVisible:false,
        priceScaleId:'dd',
      });
      equityChart.priceScale('dd').applyOptions({scaleMargins:{top:0,bottom:0.7}});

      const resizeObserver=new ResizeObserver(()=>{
        if(mainChart)mainChart.applyOptions({width:container.clientWidth,height:container.clientHeight});
        if(equityChart)equityChart.applyOptions({width:eqContainer.clientWidth});
      });
      resizeObserver.observe(container);
      resizeObserver.observe(eqContainer);
    }

    function refreshChart(){
      if(!candleSeries)return;
      const key=selectedSymbol.value+'_'+selectedTF.value;
      const data=chartDataStore[key];
      if(!data)return;
      if(data.candles&&data.candles.length>0){
        candleSeries.setData(data.candles);
      }
      if(data.volumes&&data.volumes.length>0){
        const colored=data.volumes.map((v,i)=>{
          const c=data.candles[i];
          const isUp=c&&c.close>=c.open;
          return{...v,color:isUp?'rgba(0,214,143,0.15)':'rgba(255,68,102,0.15)'};
        });
        volumeSeries.setData(colored);
      }
      if(data.candles&&data.candles.length>20){
        const closes=data.candles.map(c=>c.close);
        const times=data.candles.map(c=>c.time);
        ema20Series.setData(calcEMA(closes,times,20));
        ema50Series.setData(calcEMA(closes,times,50));
        ema200Series.setData(calcEMA(closes,times,Math.min(200,closes.length)));
      }
      mainChart.timeScale().fitContent();
    }

    function updateEquityChart(eqHist){
      if(!equitySeries)return;
      if(eqHist&&eqHist.length>0){
        const now=Math.floor(Date.now()/1000);
        const eqData=eqHist.map((v,i)=>({
          time:now-(eqHist.length-i)*60,
          value:typeof v==='number'?v:(v.equity||v.value||0),
        }));
        equitySeries.setData(eqData);
        // Drawdown overlay
        let peak=eqData[0].value;
        const ddData=eqData.map(d=>{
          if(d.value>peak)peak=d.value;
          const dd=peak>0?(peak-d.value)/peak*100:0;
          return{time:d.time,value:dd};
        });
        ddSeries.setData(ddData);
        equityChart.timeScale().fitContent();
      } else {
        const eq=equity.value||STARTING_BAL;
        localEquityHist.push(eq);
        if(localEquityHist.length>300)localEquityHist=localEquityHist.slice(-300);
        const now=Math.floor(Date.now()/1000);
        const eqData=localEquityHist.map((v,i)=>({
          time:now-(localEquityHist.length-i)*5,value:v,
        }));
        equitySeries.setData(eqData);
        equityChart.timeScale().fitContent();
      }
    }

    // ══════════════���═════════���══════════════
    // SOCKET.IO
    // ═══════���════════════════���══════════════
    const socket=io({reconnection:true,reconnectionDelay:1000,reconnectionDelayMax:5000,reconnectionAttempts:Infinity});

    socket.on('connect',()=>{connected.value=true;console.log('Connected')});
    socket.on('disconnect',()=>{connected.value=false;console.log('Disconnected')});
    socket.on('reconnect',(attempt)=>{console.log('Reconnected after',attempt,'attempts')});

    socket.on('tick_update',(data)=>{
      try{
        for(const sym of SYMBOLS){if(ticks[sym])prevPrices[sym]=ticks[sym].bid}
        if(data._account){
          const a=data._account;
          balance.value=a.balance||0;
          equity.value=a.equity||0;
          floatPnl.value=a.profit||0;
          dailyPnl.value=Math.round((todayClosedPnl.value+(a.profit||0))*100)/100;
          delete data._account;
        }
        if(data._pos_map){
          Object.keys(posMap).forEach(k=>delete posMap[k]);
          Object.assign(posMap,data._pos_map);
          delete data._pos_map;
        }
        if(data._positions){
          positions.value=data._positions;
          const uniq=new Set(data._positions.map(p=>p.symbol));
          numPositions.value=uniq.size;
          delete data._positions;
        }
        for(const[sym,val]of Object.entries(data)){ticks[sym]=val}
        nextTick(drawAllSparklines);
      }catch(e){console.error('tick_update error:',e)}
    });

    socket.on('chart_update',(data)=>{
      try{
        for(const[key,val]of Object.entries(data)){
          if(!key.endsWith('_indicators'))chartDataStore[key]=val;
        }
        refreshChart();
      }catch(e){console.error('chart_update error:',e)}
    });

    socket.on('stats_update',(data)=>{
      try{
        balance.value=data.balance||balance.value;
        equity.value=data.equity||equity.value;
        floatPnl.value=data.profit||0;
        if(data.today_closed_pnl!==undefined)todayClosedPnl.value=data.today_closed_pnl;
        dailyPnl.value=data.daily_pnl||0;
        riskPct.value=data.risk_pct||data.dd_pct||0;
        mode.value=(data.mode||'HYBRID').toUpperCase();
        if(data.session){session.value=data.session;sessionColor.value=data.session_color||'#00c8e0'}
        if(data.positions){
          const uniq=new Set(data.positions.map(p=>p.symbol));
          numPositions.value=uniq.size;
        }
        if(data.scores)Object.keys(data.scores).forEach(k=>{scores[k]=data.scores[k]});
        if(data.ml_confidence)Object.keys(data.ml_confidence).forEach(k=>{mlConfidence[k]=data.ml_confidence[k]});
        if(data.pos_map){
          Object.keys(posMap).forEach(k=>delete posMap[k]);
          Object.assign(posMap,data.pos_map);
        }
        if(data.mtf_intelligence)Object.keys(data.mtf_intelligence).forEach(k=>{mtfIntelligence[k]=data.mtf_intelligence[k]});
        masterBrain.value=data.master_brain||null;
        if(data.learning_stats)Object.keys(data.learning_stats).forEach(k=>{learningStats[k]=data.learning_stats[k]});
        const newLog=data.trade_log||[];
        if(newLog.length!==tradeLog.value.length)tradePage.value=0;
        tradeLog.value=newLog;
        if(data.feature_importance)Object.keys(data.feature_importance).forEach(k=>{featureImportance[k]=data.feature_importance[k]});
        equityHistory.value=data.equity_history||[];
        updateEquityChart(equityHistory.value);
        nextTick(drawAllSparklines);
      }catch(e){console.error('stats_update error:',e)}
    });

    socket.on('action_result',(data)=>{console.log('Action result:',data)});

    // ═══════════════════════════════════════
    // CLOCK
    // ═══════════════════���═══════════════════
    function updateClock(){
      clock.value=new Date().toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour12:false})+' IST';
    }

    // ═══════════════════════════════════════
    // LIFECYCLE
    // ════════════════��══════════════════════
    onMounted(()=>{
      initCharts();
      updateClock();
      setInterval(updateClock,1000);
    });

    return{
      symbols,selectedSymbol,selectedTF,connected,clock,closeSymSelect,
      mode,session,sessionColor,balance,equity,floatPnl,dailyPnl,
      numPositions,riskPct,tradePage,tradePageSize,tradeLogOpen,
      ticks,prevPrices,scores,mlConfidence,posMap,positions,
      tradeLog,equityHistory,featureImportance,mtfIntelligence,masterBrain,learningStats,
      modal,sparkRefs,timeframes,
      fmtNum,fmtDollar,fmtPnl,fmtPrice,
      getCat,getDigits,getTickVal,isPriceUp,getScore,getML,dirColor,
      h1ScoreVal,h1ScorePct,h1ScoreColor,
      getMtf,mtfDotClass,mtfDirColor,mtfArrow,confClass,eqColor,
      exitUrgencyColor,exitBadgeClass,
      selectedMtf,hasPatterns,
      selectedScores,selectedGate,selectedMLAuc,breakdownMetrics,
      mbHealthColor,mbBlacklistStr,
      tradePageStart,tradeLogPaged,tradeDir,tradeTagClass,
      perfStats,perSymbolStats,
      hasLearningStats,learnRiskVal,learnRiskClass,
      selectSymbol,selectTF,showModal,modalConfirm,doCloseSym,
    };
  }
});
app.mount('#app');
</script>
</body>
</html>
"""
