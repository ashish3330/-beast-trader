"""
Beast Trader — JARVIS Dashboard.
Flask app on port 8888 with real-time data from shared state.
"""
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from flask import Flask, render_template_string, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS, STARTING_BALANCE, DASHBOARD_PORT

log = logging.getLogger("beast.dashboard")

app = Flask(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Shared state reference — set by run.py
_state = None
_executor = None


def init_dashboard(state, executor=None):
    global _state, _executor
    _state = state
    _executor = executor


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>B.E.A.S.T — ML Trading Agent</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/remixicon@3.5.0/fonts/remixicon.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #020810; --bg2: #051020; --bg3: #0a1a30;
  --cyan: #00f0ff; --cyan-dim: rgba(0,240,255,0.15); --cyan-glow: rgba(0,240,255,0.3);
  --blue: #0080ff; --blue-dim: rgba(0,128,255,0.1);
  --green: #00ff88; --green-dim: rgba(0,255,136,0.1);
  --red: #ff2060; --red-dim: rgba(255,32,96,0.1);
  --gold: #ffcc00; --gold-dim: rgba(255,204,0,0.1);
  --purple: #a855f7; --purple-dim: rgba(168,85,247,0.1);
  --text: #c0d8f0; --text2: #6088b0; --text3: #304060;
  --border: rgba(0,240,255,0.08); --border2: rgba(0,240,255,0.15);
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;overflow-x:hidden;min-height:100vh}

body::before{content:'';position:fixed;inset:0;z-index:-2;
  background:radial-gradient(circle at 50% 50%,rgba(0,240,255,0.02) 0%,transparent 70%)}
body::after{content:'';position:fixed;inset:0;z-index:-1;
  background-image:linear-gradient(rgba(0,240,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,240,255,0.03) 1px,transparent 1px);
  background-size:60px 60px;animation:gridMove 20s linear infinite}
@keyframes gridMove{0%{transform:translate(0,0)}100%{transform:translate(60px,60px)}}

.scanline{position:fixed;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  opacity:0.1;z-index:999;pointer-events:none;animation:scan 4s linear infinite}
@keyframes scan{0%{top:0}100%{top:100vh}}

.hud-header{
  background:linear-gradient(180deg,rgba(0,240,255,0.05),transparent);
  border-bottom:1px solid var(--border2);padding:12px 24px;
  display:flex;justify-content:space-between;align-items:center;position:relative}
.hud-header::after{content:'';position:absolute;bottom:-1px;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:0.3}

.logo{display:flex;align-items:center;gap:12px}
.logo-hex{width:40px;height:40px;position:relative;display:flex;align-items:center;justify-content:center}
.logo-hex::before{content:'';position:absolute;inset:0;background:var(--cyan-dim);border:1px solid var(--cyan);
  border-radius:8px;transform:rotate(0deg);animation:hexSpin 10s linear infinite}
@keyframes hexSpin{0%{transform:rotate(0deg);border-color:var(--cyan)}50%{border-color:var(--blue)}100%{transform:rotate(360deg);border-color:var(--cyan)}}
.logo-hex i{color:var(--cyan);font-size:18px;z-index:1}
.logo-title{font-family:'Orbitron';font-size:16px;font-weight:700;letter-spacing:3px}
.logo-title span{color:var(--cyan);text-shadow:0 0 20px var(--cyan-glow)}
.logo-sub{font-family:'Share Tech Mono';font-size:10px;color:var(--text2);letter-spacing:4px}

.hud-stats{display:flex;gap:24px}
.hud-stat{text-align:center;position:relative}
.hud-stat::before{content:'';position:absolute;left:-12px;top:4px;bottom:4px;width:1px;
  background:linear-gradient(180deg,transparent,var(--border2),transparent)}
.hud-stat:first-child::before{display:none}
.hud-stat .label{font-family:'Share Tech Mono';font-size:9px;color:var(--text3);letter-spacing:2px;text-transform:uppercase}
.hud-stat .val{font-family:'Orbitron';font-size:16px;font-weight:600}
.hud-stat .val.pos{color:var(--green);text-shadow:0 0 10px rgba(0,255,136,0.3)}
.hud-stat .val.neg{color:var(--red);text-shadow:0 0 10px rgba(255,32,96,0.3)}
.hud-stat .val.cyan{color:var(--cyan);text-shadow:0 0 10px var(--cyan-glow)}

.hud-right{display:flex;align-items:center;gap:16px}
.status-ring{width:10px;height:10px;border-radius:50%;background:var(--green);
  box-shadow:0 0 10px var(--green),0 0 20px rgba(0,255,136,0.3);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 10px var(--green),0 0 20px rgba(0,255,136,0.3)}
  50%{box-shadow:0 0 5px var(--green),0 0 10px rgba(0,255,136,0.1)}}
.clock{font-family:'Orbitron';font-size:14px;color:var(--cyan);letter-spacing:2px}

.hud-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;padding:12px;background:transparent}

.panel{background:var(--bg2);border:1px solid var(--border);border-radius:4px;overflow:hidden;
  position:relative;backdrop-filter:blur(10px)}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:0.2}
.panel-head{padding:10px 14px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;
  background:linear-gradient(180deg,rgba(0,240,255,0.03),transparent)}
.panel-title{font-family:'Orbitron';font-size:10px;font-weight:600;letter-spacing:2px;color:var(--cyan)}
.panel-title i{margin-right:6px;font-size:12px}
.badge{font-family:'Share Tech Mono';font-size:10px;padding:2px 8px;border-radius:2px;
  border:1px solid var(--border2);color:var(--text2)}
.panel-body{padding:0;max-height:320px;overflow-y:auto}
.panel-body::-webkit-scrollbar{width:3px}
.panel-body::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}

.span2{grid-column:span 2} .span3{grid-column:span 3}

.metrics{grid-column:1/-1;display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
.metric{background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:12px;
  position:relative;overflow:hidden}
.metric::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;opacity:0.5}
.metric.c-cyan::before{background:var(--cyan)} .metric.c-green::before{background:var(--green)}
.metric.c-red::before{background:var(--red)} .metric.c-gold::before{background:var(--gold)}
.metric.c-blue::before{background:var(--blue)} .metric.c-purple::before{background:var(--purple)}
.metric .m-icon{font-size:16px;margin-bottom:6px;opacity:0.5}
.metric .m-label{font-family:'Share Tech Mono';font-size:9px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase}
.metric .m-val{font-family:'Orbitron';font-size:18px;font-weight:700;margin-top:2px}
.metric .m-sub{font-family:'Share Tech Mono';font-size:9px;color:var(--text3);margin-top:2px}

table{width:100%;border-collapse:collapse}
th{padding:6px 12px;text-align:left;font-family:'Share Tech Mono';font-size:9px;color:var(--text3);
  letter-spacing:1.5px;text-transform:uppercase;background:rgba(0,240,255,0.02);position:sticky;top:0;z-index:1}
td{padding:6px 12px;border-bottom:1px solid var(--border);font-size:12px}
tr:hover td{background:rgba(0,240,255,0.02)}

.tag{display:inline-block;padding:1px 8px;border-radius:2px;font-family:'Share Tech Mono';
  font-size:9px;letter-spacing:1px;font-weight:600}
.tag-long{background:var(--green-dim);color:var(--green);border:1px solid rgba(0,255,136,0.2)}
.tag-short{background:var(--red-dim);color:var(--red);border:1px solid rgba(255,32,96,0.2)}
.tag-flat{background:rgba(48,64,96,0.2);color:var(--text3);border:1px solid var(--border)}
.tag-gold{background:var(--gold-dim);color:var(--gold)} .tag-idx{background:var(--blue-dim);color:var(--blue)}
.tag-crypto{background:var(--purple-dim);color:var(--purple)}

.mono{font-family:'Share Tech Mono'} .g{color:var(--green)} .r{color:var(--red)} .cy{color:var(--cyan)}
.dim{color:var(--text3)} .bright{font-weight:600;color:var(--text)}
.empty{padding:24px;text-align:center;color:var(--text3);font-size:11px}

.chart-wrap{padding:10px 14px;height:180px}

.conf-bar{height:4px;background:var(--bg);border-radius:2px;overflow:hidden;width:80px;display:inline-block;vertical-align:middle;margin-left:6px}
.conf-fill{height:100%;border-radius:2px;transition:width 0.3s}

.btn{font-family:'Share Tech Mono';font-size:10px;padding:4px 12px;border:1px solid var(--red);
  background:var(--red-dim);color:var(--red);border-radius:2px;cursor:pointer;letter-spacing:1px}
.btn:hover{background:rgba(255,32,96,0.3)}
.btn-row{padding:8px 14px;display:flex;gap:8px;flex-wrap:wrap}

.footer{grid-column:1/-1;text-align:center;padding:8px;font-family:'Share Tech Mono';
  font-size:9px;color:var(--text3);letter-spacing:3px;border-top:1px solid var(--border)}

@media(max-width:1100px){.hud-grid{grid-template-columns:1fr 1fr}.span3{grid-column:span 2}.metrics{grid-template-columns:repeat(3,1fr)}}
</style>
</head>
<body>
<div class="scanline"></div>

<div class="hud-header">
  <div class="logo">
    <div class="logo-hex"><i class="ri-brain-fill"></i></div>
    <div>
      <div class="logo-title"><span>B.E.A.S.T</span></div>
      <div class="logo-sub">ML TRADING AGENT v1.0</div>
    </div>
  </div>
  <div class="hud-stats">
    <div class="hud-stat"><div class="label">Equity</div><div class="val cyan" id="h-eq">---</div></div>
    <div class="hud-stat"><div class="label">Float P&L</div><div class="val" id="h-pnl">---</div></div>
    <div class="hud-stat"><div class="label">DD</div><div class="val" id="h-dd">---</div></div>
    <div class="hud-stat"><div class="label">Cycle</div><div class="val cyan" id="h-cycle">0</div></div>
  </div>
  <div class="hud-right">
    <div class="status-ring" id="status-ring"></div>
    <div class="clock" id="h-clock">00:00:00</div>
  </div>
</div>

<div class="hud-grid">
  <!-- Metrics row -->
  <div class="metrics">
    <div class="metric c-cyan"><div class="m-icon" style="color:var(--cyan)"><i class="ri-funds-fill"></i></div><div class="m-label">Balance</div><div class="m-val" id="m-bal">---</div><div class="m-sub" id="m-bal-chg"></div></div>
    <div class="metric c-green"><div class="m-icon" style="color:var(--green)"><i class="ri-line-chart-fill"></i></div><div class="m-label">Equity</div><div class="m-val" id="m-eq">---</div></div>
    <div class="metric c-gold"><div class="m-icon" style="color:var(--gold)"><i class="ri-flashlight-fill"></i></div><div class="m-label">Positions</div><div class="m-val" id="m-pos">0</div></div>
    <div class="metric c-purple"><div class="m-icon" style="color:var(--purple)"><i class="ri-brain-fill"></i></div><div class="m-label">Agent Cycles</div><div class="m-val" id="m-cycles">0</div></div>
    <div class="metric c-blue"><div class="m-icon" style="color:var(--blue)"><i class="ri-bar-chart-fill"></i></div><div class="m-label">Daily Loss</div><div class="m-val" id="m-dloss">0%</div></div>
    <div class="metric c-red"><div class="m-icon" style="color:var(--red)"><i class="ri-shield-fill"></i></div><div class="m-label">Max DD</div><div class="m-val" id="m-dd">0%</div></div>
  </div>

  <!-- Model confidence panel -->
  <div class="panel span2">
    <div class="panel-head">
      <span class="panel-title"><i class="ri-radar-fill"></i>MODEL INTELLIGENCE</span>
      <span class="badge" id="intel-n">0 symbols</span>
    </div>
    <div class="panel-body" id="intel-body"><div class="empty">Waiting for model...</div></div>
  </div>

  <!-- Tick chart -->
  <div class="panel">
    <div class="panel-head">
      <span class="panel-title"><i class="ri-line-chart-fill"></i>EQUITY CURVE</span>
    </div>
    <div class="panel-body"><div class="chart-wrap"><canvas id="eqChart"></canvas></div></div>
  </div>

  <!-- Feature importance -->
  <div class="panel">
    <div class="panel-head">
      <span class="panel-title"><i class="ri-bar-chart-grouped-fill"></i>FEATURE IMPORTANCE</span>
    </div>
    <div class="panel-body"><div class="chart-wrap"><canvas id="impChart"></canvas></div></div>
  </div>

  <!-- Positions -->
  <div class="panel">
    <div class="panel-head">
      <span class="panel-title"><i class="ri-flashlight-fill"></i>LIVE POSITIONS</span>
      <span class="badge" id="pos-n">0</span>
    </div>
    <div class="panel-body" id="pos-body"><div class="empty">No positions</div></div>
    <div class="btn-row">
      <button class="btn" onclick="closeAll()">CLOSE ALL</button>
      <select id="close-sym" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px;font-family:'Share Tech Mono';font-size:10px">
        <option value="">Select symbol</option>
        <option value="XAUUSD">XAUUSD</option>
        <option value="BTCUSD">BTCUSD</option>
        <option value="NAS100.r">NAS100.r</option>
        <option value="GER40.r">GER40.r</option>
      </select>
      <button class="btn" onclick="closeSym()">CLOSE SYMBOL</button>
    </div>
  </div>

  <!-- Trade log -->
  <div class="panel span3">
    <div class="panel-head">
      <span class="panel-title"><i class="ri-file-list-3-fill"></i>TRADE LOG</span>
      <span class="badge" id="trade-n">0</span>
    </div>
    <div class="panel-body" id="trade-body"><div class="empty">No trades yet</div></div>
  </div>

  <div class="footer">B.E.A.S.T ML TRADING AGENT &middot; TICK-LEVEL &middot; LIGHTGBM &middot; 1s REFRESH</div>
</div>

<script>
let eqChart=null, impChart=null;
let eqData=[];
const $=id=>document.getElementById(id);
const f=(v,d=2)=>v!=null?v.toFixed(d):'---';
const s=v=>v>=0?'+'+f(v):f(v);
const pc=v=>v>=0?'g':'r';

function initCharts(){
  eqChart=new Chart($('eqChart'),{
    type:'line',data:{labels:[],datasets:[{data:[],borderColor:'var(--cyan)',borderWidth:1.5,pointRadius:0,fill:true,
      backgroundColor:'rgba(0,240,255,0.05)',tension:0.3}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},
      plugins:{legend:{display:false}},
      scales:{x:{display:false},y:{ticks:{color:'#304060',font:{size:8},callback:v=>'$'+v},grid:{color:'rgba(0,240,255,0.05)'},border:{display:false}}}}
  });
  impChart=new Chart($('impChart'),{
    type:'bar',data:{labels:[],datasets:[{data:[],backgroundColor:'rgba(0,240,255,0.4)',borderRadius:2,barThickness:8}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,animation:{duration:0},
      plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#304060',font:{size:7}},grid:{color:'rgba(0,240,255,0.05)'},border:{display:false}},
        y:{ticks:{color:'#6088b0',font:{size:7,family:'Share Tech Mono'}},grid:{display:false},border:{display:false}}}}
  });
}

function confBar(conf){
  const pct=Math.min(100,conf*100);
  const color=conf>0.6?'var(--green)':conf>0.3?'var(--gold)':'var(--red)';
  return `<div class="conf-bar"><div class="conf-fill" style="width:${pct}%;background:${color}"></div></div>`;
}

function catTag(cat){
  const map={Gold:'tag-gold',Index:'tag-idx',Crypto:'tag-crypto'};
  return `<span class="tag ${map[cat]||'tag-flat'}">${cat}</span>`;
}

function dirTag(dir){
  if(dir==='LONG') return '<span class="tag tag-long">LONG</span>';
  if(dir==='SHORT') return '<span class="tag tag-short">SHORT</span>';
  return '<span class="tag tag-flat">FLAT</span>';
}

function update(){
  fetch('/api/data').then(r=>r.json()).then(d=>{
    // Header
    $('h-eq').textContent='$'+f(d.equity);
    $('h-pnl').className='val '+(d.profit>=0?'pos':'neg');
    $('h-pnl').textContent='$'+s(d.profit);
    $('h-dd').textContent=f(d.dd_pct,1)+'%';
    $('h-dd').className='val '+(d.dd_pct>5?'neg':'pos');
    $('h-cycle').textContent=d.cycle;
    $('h-clock').textContent=d.time;

    // Status ring
    $('status-ring').style.background=d.running?'var(--green)':'var(--red)';

    // Metrics
    const chg=d.balance-"""+str(STARTING_BALANCE)+""";
    $('m-bal').innerHTML='<span class="cy">$'+f(d.balance)+'</span>';
    $('m-bal-chg').innerHTML='<span class="'+(chg>=0?'g':'r')+'">'+s(chg)+'</span>';
    $('m-eq').textContent='$'+f(d.equity);
    $('m-pos').textContent=d.positions.length;
    $('m-cycles').textContent=d.cycle;
    $('m-dloss').innerHTML='<span class="'+(d.daily_loss>1?'r':'g')+'">'+f(d.daily_loss,1)+'%</span>';
    $('m-dd').innerHTML='<span class="'+(d.dd_pct>5?'r':'g')+'">'+f(d.dd_pct,1)+'%</span>';

    // Model Intelligence
    $('intel-n').textContent=Object.keys(d.confidence).length+' symbols';
    let ih='<table><tr><th>Symbol</th><th>Cat</th><th>Direction</th><th>P(Up)</th><th>P(Down)</th><th>Confidence</th><th>Bid</th><th>Ask</th><th>Spread</th></tr>';
    for(const[sym,c] of Object.entries(d.confidence)){
      const t=d.ticks[sym]||{};
      const cat=d.categories[sym]||'';
      ih+=`<tr><td class="bright">${sym}</td><td>${catTag(cat)}</td>
        <td>${dirTag(c.direction)}</td>
        <td class="mono ${c.prob_up>0.5?'g':'dim'}">${f(c.prob_up,3)}</td>
        <td class="mono ${c.prob_down>0.5?'r':'dim'}">${f(c.prob_down,3)}</td>
        <td class="mono">${f(c.confidence,2)} ${confBar(c.confidence)}</td>
        <td class="mono">${t.bid?f(t.bid,t.digits||2):'---'}</td>
        <td class="mono">${t.ask?f(t.ask,t.digits||2):'---'}</td>
        <td class="mono dim">${t.spread?f(t.spread,1):'---'}</td></tr>`;
    }
    $('intel-body').innerHTML=ih+'</table>';

    // Positions
    $('pos-n').textContent=d.positions.length;
    if(d.positions.length){
      let ph='<table><tr><th>Sym</th><th>Side</th><th>Vol</th><th>P&L</th><th>SL</th><th>Dur</th></tr>';
      d.positions.forEach(p=>{
        ph+=`<tr><td class="bright">${p.symbol}</td>
          <td><span class="tag ${p.type==='BUY'?'tag-long':'tag-short'}">${p.type}</span></td>
          <td class="mono">${f(p.volume,2)}</td>
          <td class="mono ${pc(p.pnl)}">$${s(p.pnl)}</td>
          <td class="mono dim">${f(p.sl,2)}</td>
          <td class="dim">${p.duration||''}</td></tr>`;
      });
      $('pos-body').innerHTML=ph+'</table>';
    }else{$('pos-body').innerHTML='<div class="empty"><i class="ri-radar-line" style="font-size:20px;display:block;margin-bottom:4px;color:var(--cyan)"></i>Awaiting signals</div>'}

    // Trade log
    $('trade-n').textContent=d.trades.length;
    if(d.trades.length){
      let th='<table><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Dir</th><th>Confidence</th><th>Regime</th></tr>';
      d.trades.slice(-20).reverse().forEach(t=>{
        th+=`<tr><td class="mono dim">${t.timestamp||''}</td><td class="bright">${t.symbol}</td>
          <td class="mono cy">${t.action||''}</td>
          <td>${dirTag((t.direction||'').toUpperCase())}</td>
          <td class="mono">${f(t.confidence,2)} ${confBar(t.confidence||0)}</td>
          <td class="dim">${t.regime||''}</td></tr>`;
      });
      $('trade-body').innerHTML=th+'</table>';
    }else{$('trade-body').innerHTML='<div class="empty">No trades yet</div>'}

    // Equity chart
    eqData.push(d.equity);
    if(eqData.length>300) eqData=eqData.slice(-300);
    eqChart.data.labels=eqData.map((_,i)=>i);
    eqChart.data.datasets[0].data=eqData;
    eqChart.update('none');

    // Feature importance chart (first symbol that has importance)
    const impKeys=Object.keys(d.importance);
    if(impKeys.length){
      const imp=d.importance[impKeys[0]];
      const sorted=Object.entries(imp).sort((a,b)=>b[1]-a[1]).slice(0,10);
      impChart.data.labels=sorted.map(s=>s[0]);
      impChart.data.datasets[0].data=sorted.map(s=>s[1]);
      impChart.update('none');
    }
  }).catch(e=>console.error(e));
}

function closeAll(){
  if(confirm('Close ALL positions?')) fetch('/api/close_all',{method:'POST'});
}
function closeSym(){
  const sym=$('close-sym').value;
  if(sym&&confirm('Close '+sym+'?')) fetch('/api/close_symbol',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym})});
}

initCharts();update();setInterval(update,1000);
setInterval(()=>{$('h-clock').textContent=new Date().toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour12:false})+' IST';},1000);
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/data')
def api_data():
    if _state is None:
        return jsonify({"error": "State not initialized"})

    agent = _state.get_agent_state()

    # Tick data for each symbol
    ticks = {}
    for sym, cfg in SYMBOLS.items():
        t = _state.get_tick(sym)
        if t:
            ticks[sym] = {
                "bid": t.bid, "ask": t.ask,
                "spread": (t.ask - t.bid) * (10 ** cfg.digits),
                "digits": cfg.digits,
            }

    categories = {sym: cfg.category for sym, cfg in SYMBOLS.items()}

    return jsonify({
        "equity": agent.get("equity", 0),
        "balance": agent.get("balance", 0),
        "profit": agent.get("profit", 0),
        "dd_pct": agent.get("dd_pct", 0),
        "daily_loss": agent.get("daily_loss", 0),
        "cycle": agent.get("cycle", 0),
        "running": agent.get("running", False),
        "positions": agent.get("positions", []),
        "confidence": agent.get("model_confidence", {}),
        "importance": agent.get("feature_importance", {}),
        "trades": agent.get("trade_log", []),
        "ticks": ticks,
        "categories": categories,
        "time": datetime.now(IST).strftime('%H:%M:%S IST'),
    })


@app.route('/api/close_all', methods=['POST'])
def close_all():
    if _executor:
        _executor.close_all("DashboardCloseAll")
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "msg": "No executor"})


@app.route('/api/close_symbol', methods=['POST'])
def close_symbol():
    data = request.get_json()
    sym = data.get("symbol", "")
    if _executor and sym:
        _executor.close_position(sym, "DashboardClose")
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "msg": "No executor or symbol"})


def run_dashboard():
    """Start dashboard in a thread."""
    log.info("Dashboard starting on port %d", DASHBOARD_PORT)
    app.run(host='0.0.0.0', port=DASHBOARD_PORT, debug=False, use_reloader=False)
