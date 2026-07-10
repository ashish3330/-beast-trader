"""
dashboard/v2_api.py — Rich dashboard endpoints (additive only).

Adds:
  - HTTP endpoints prefixed `/api/v2/` (read-only, defensive)
  - WebSocket events with colon-separated names: ticks:bulk, portfolio:update,
    signal:scored, alert:new, position:opened, position:closed.
  - Background push threads (started by register_v2()).

The module is wired into dashboard/app.py by register_v2(app, socketio).
NEVER blocks trading: every external call is wrapped in try/except, and all
endpoints return defaults on error (never 500).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import os
import pickle
import re
import collections
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from flask import jsonify, request

log = logging.getLogger("dragon.dashboard.v2")

ROOT = Path(__file__).resolve().parent.parent
RL_DB = ROOT / "data" / "rl_learner.db"
JOURNAL_DB = ROOT / "data" / "trade_journal.db"
LOG_PATH = ROOT / "logs" / "dragon.log"
BACKTEST_RESULTS_DIR = ROOT / "backtest" / "results"
MODEL_DIR = ROOT / "models" / "saved"

# Module-level refs (filled in register_v2)
_state = None
_executor = None
_socketio = None
_app = None

# Caches
_corr_cache = {"ts": 0.0, "data": None}
_ml_auc_cache = {"ts": 0.0, "data": {}}
_decision_buffer: "deque[dict]" = deque(maxlen=500)
_decision_lock = threading.Lock()
_alert_buffer: "deque[dict]" = deque(maxlen=200)
_alert_lock = threading.Lock()
_loop_latency_ring: "deque[float]" = deque(maxlen=600)  # last 600 decision-loop ms
_loop_latency_lock = threading.Lock()
_started_at = time.time()


# ────────────────────────────────────────────────────────────────────
#  Public entry points used from agent code (lazy, defensive)
# ────────────────────────────────────────────────────────────────────


def push_decision(payload: dict) -> None:
    """Called from agent/brain.py:_log_decision. Buffer + emit signal:scored."""
    try:
        with _decision_lock:
            _decision_buffer.append(payload)
        if _socketio is not None:
            try:
                _socketio.emit("signal:scored", payload)
            except Exception as e:
                log.debug("signal:scored emit failed: %s", e)
    except Exception as e:
        log.debug("push_decision error: %s", e)


def push_alert(payload: dict) -> None:
    """Called from agent/alerting.py SocketIOBackend (optional). Buffer + emit."""
    try:
        with _alert_lock:
            _alert_buffer.append(payload)
        if _socketio is not None:
            try:
                _socketio.emit("alert:new", payload)
            except Exception as e:
                log.debug("alert:new emit failed: %s", e)
    except Exception:
        pass


def push_position_event(kind: str, payload: dict) -> None:
    """Called from execution/executor.py on open/close. kind = 'opened'|'closed'|'r_update'."""
    if _socketio is None:
        return
    try:
        if kind == "opened":
            evt = "position:opened"
        elif kind == "closed":
            evt = "position:closed"
        else:
            # 2026-06-18 Tier 1 #4: also accept r_update telemetry
            evt = "position:r_update"
        _socketio.emit(evt, payload)
    except Exception as e:
        log.debug("position event emit failed: %s", e)


def push_position_r(payload: dict) -> None:
    """2026-06-18 Tier 1 #4: per-position R-multiple live telemetry.

    Called by execution/executor.py once per cycle (rate-limited) inside the
    exit-management loop. Payload: {ts, symbol, profit_r, peak_r, status}.
    Fire-and-forget — never blocks the executor.
    """
    if _socketio is None:
        return
    try:
        _socketio.emit("position:r_update", payload)
    except Exception as e:
        log.debug("position:r_update emit failed: %s", e)


def record_loop_latency_ms(ms: float) -> None:
    try:
        with _loop_latency_lock:
            _loop_latency_ring.append(float(ms))
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────


def _safe(fn):
    """Endpoint decorator: never 500. Return JSON `{}` on error with status 200."""
    from functools import wraps
    @wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            log.exception("v2 endpoint %s failed: %s", fn.__name__, e)
            return jsonify({"error": str(e), "data": None})
    return wrapper


def _agent():
    if _state is None:
        return {}
    try:
        return _state.get_agent_state() or {}
    except Exception:
        return {}


def _query_rl_db(sql: str, args: tuple = ()) -> List[tuple]:
    if not RL_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(RL_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, args).fetchall()
        conn.close()
        return rows
    except Exception as e:
        log.debug("rl_db query failed: %s", e)
        return []


def _query_journal_db(sql: str, args: tuple = ()) -> List[sqlite3.Row]:
    if not JOURNAL_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(JOURNAL_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, args).fetchall()
        conn.close()
        return rows
    except Exception as e:
        log.debug("journal_db query failed: %s", e)
        return []


def _ml_auc_for(symbol: str) -> Optional[float]:
    """Read test_auc from saved meta-label pkl (cached 5min)."""
    now = time.time()
    if now - _ml_auc_cache["ts"] < 300 and symbol in _ml_auc_cache["data"]:
        return _ml_auc_cache["data"].get(symbol)
    try:
        path = MODEL_DIR / f"{symbol.replace('.', '_')}_meta_lgb_ensemble.pkl"
        if not path.exists():
            path = MODEL_DIR / f"{symbol.replace('.', '_')}_meta_lgb.pkl"
        if not path.exists():
            _ml_auc_cache["data"][symbol] = None
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
        auc = metrics.get("test_auc") or metrics.get("auc")
        if auc is not None:
            auc = float(auc)
        _ml_auc_cache["data"][symbol] = auc
        _ml_auc_cache["ts"] = now
        return auc
    except Exception as e:
        log.debug("ml_auc read for %s failed: %s", symbol, e)
        return None


def _trail_summary(steps: list) -> str:
    """Format trail steps for compact display."""
    if not steps:
        return ""
    try:
        parts = []
        for step in steps[:4]:
            r, mode, lock = step[0], step[1], step[2]
            parts.append(f"{r}R/{mode[:3]}{lock}")
        return " | ".join(parts)
    except Exception:
        return str(steps)[:80]


# ────────────────────────────────────────────────────────────────────
#  HTTP endpoints
# ────────────────────────────────────────────────────────────────────


def _register_routes(app):
    @app.route("/api/v2/portfolio")
    @_safe
    def v2_portfolio():
        from config import SYMBOLS
        agent = _agent()
        balance = float(agent.get("balance", 0.0))
        equity = float(agent.get("equity", 0.0))
        dd_pct = float(agent.get("dd_pct", 0.0))
        positions_raw = []
        if _executor is not None:
            try:
                positions_raw = _executor.get_positions_info() or []
            except Exception as e:
                log.debug("get_positions_info failed: %s", e)
                positions_raw = agent.get("positions", []) or []
        else:
            positions_raw = agent.get("positions", []) or []

        positions: List[dict] = []
        exposure_class: Dict[str, float] = defaultdict(float)
        exposure_ccy: Dict[str, float] = defaultdict(float)
        open_pnl = 0.0
        now = time.time()

        for p in positions_raw:
            try:
                sym = str(p.get("symbol", ""))
                cfg = SYMBOLS.get(sym)
                side = "BUY" if str(p.get("type", "BUY")).upper() in ("BUY", "LONG") else "SELL"
                volume = float(p.get("volume", 0.0))
                entry = float(p.get("price_open", 0.0))
                pnl = float(p.get("pnl", 0.0))
                sl = float(p.get("sl", 0.0))
                tp = float(p.get("tp", 0.0))
                magic = int(p.get("magic", 0))

                # current price (use tick mid)
                current_price = entry
                spread = 0.0
                if _state is not None:
                    try:
                        t = _state.get_tick(sym)
                        if t:
                            current_price = float(t.bid) if side == "SELL" else float(t.ask)
                    except Exception:
                        pass

                # pnl_r: distance vs SL distance
                pnl_r = 0.0
                if sl and entry and abs(entry - sl) > 0:
                    sign = 1.0 if side == "BUY" else -1.0
                    pnl_r = sign * (current_price - entry) / abs(entry - sl)

                open_time = float(p.get("time", 0)) or 0.0
                duration_secs = max(0, int(now - open_time)) if open_time else 0

                positions.append({
                    "symbol": sym,
                    "side": side,
                    "volume": volume,
                    "entry_price": entry,
                    "current_price": current_price,
                    "pnl_usd": round(pnl, 2),
                    "pnl_r": round(pnl_r, 2),
                    "sl": sl,
                    "tp": tp,
                    "magic": magic,
                    "open_time": open_time,
                    "duration_secs": duration_secs,
                })
                open_pnl += pnl

                if cfg is not None:
                    exposure_class[cfg.category] += volume

                # Per-currency net direction (rough: split first/last 3 chars)
                if len(sym) >= 6 and sym[:3].isalpha() and sym[3:6].isalpha():
                    base, quote = sym[:3], sym[3:6]
                    sign = 1.0 if side == "BUY" else -1.0
                    exposure_ccy[base] += sign * volume
                    exposure_ccy[quote] -= sign * volume
            except Exception as e:
                log.debug("position parse error: %s", e)
                continue

        return jsonify({
            "balance": round(balance, 2),
            "equity": round(equity, 2),
            "drawdown_pct": round(dd_pct, 2),
            "open_pnl": round(open_pnl, 2),
            "positions": positions,
            "exposure_by_class": {k: round(v, 2) for k, v in exposure_class.items()},
            "exposure_by_currency": {k: round(v, 2) for k, v in exposure_ccy.items()},
        })

    @app.route("/api/v2/symbols")
    @_safe
    def v2_symbols():
        from config import (
            SYMBOLS, SYMBOL_ATR_SL_OVERRIDE, ATR_SL_MULTIPLIER,
            SIGNAL_QUALITY_SYMBOL, SIGNAL_QUALITY_THRESHOLDS,
            DIRECTION_BIAS, SYMBOL_RISK_CAP, MAX_RISK_PER_TRADE_PCT,
            TOXIC_HOURS_PER_SYMBOL, SYMBOL_TRAIL_OVERRIDE,
            DRAGON_ML_ENABLED, TRAIL_STEPS,
        )
        out = []
        for sym, cfg in SYMBOLS.items():
            sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(sym, ATR_SL_MULTIPLIER)
            quality_overrides = SIGNAL_QUALITY_SYMBOL.get(sym, {}) or {}
            min_quality_trending = quality_overrides.get(
                "trending", SIGNAL_QUALITY_THRESHOLDS.get("trending", 45)
            )
            trail = SYMBOL_TRAIL_OVERRIDE.get(sym, TRAIL_STEPS)
            out.append({
                "symbol": sym,
                "category": cfg.category,
                "digits": cfg.digits,
                "sl_atr_mult": float(sl_mult),
                "min_quality_trending": int(min_quality_trending),
                "direction_bias": DIRECTION_BIAS.get(sym),
                "risk_cap": float(SYMBOL_RISK_CAP.get(sym, MAX_RISK_PER_TRADE_PCT)),
                "toxic_hours": sorted(list(TOXIC_HOURS_PER_SYMBOL.get(sym, set()))),
                "trail_profile_summary": _trail_summary(trail),
                "ml_enabled": bool(DRAGON_ML_ENABLED.get(sym, False)),
                "ml_auc": _ml_auc_for(sym),
            })
        return jsonify(out)

    @app.route("/api/v2/rl_state")
    @_safe
    def v2_rl_state():
        # trail_adjustments rows
        rows = _query_rl_db(
            "SELECT symbol, lock_threshold_mult, be_threshold_mult, "
            "trail_tightness_mult, updated FROM trail_adjustments"
        )
        # counts per symbol
        count_rows = _query_rl_db(
            "SELECT symbol, COUNT(*) as n FROM trade_outcomes GROUP BY symbol"
        )
        counts = {r["symbol"]: int(r["n"]) for r in count_rows}
        # per-regime weight cell counts
        try:
            reg_rows = _query_rl_db(
                "SELECT symbol, COUNT(*) as n FROM regime_weights GROUP BY symbol"
            )
            reg_cells = {r["symbol"]: int(r["n"]) for r in reg_rows}
        except Exception:
            reg_cells = {}
        out = []
        for r in rows:
            sym = r["symbol"]
            out.append({
                "symbol": sym,
                "lock_threshold_mult": float(r["lock_threshold_mult"]) if r["lock_threshold_mult"] is not None else 1.0,
                "be_threshold_mult": float(r["be_threshold_mult"]) if r["be_threshold_mult"] is not None else 1.0,
                "trail_tightness_mult": float(r["trail_tightness_mult"]) if r["trail_tightness_mult"] is not None else 1.0,
                "n_trades_in_db": counts.get(sym, 0),
                "regime_cells": reg_cells.get(sym, 0),
                "last_updated": r["updated"],
            })
        return jsonify(out)

    @app.route("/api/v2/rl_health")
    @_safe
    def v2_rl_health():
        """RL learner health snapshot from DB — surfaces what the RL is doing
        and proves it's actually learning (or stuck)."""
        try:
            peak_row = _query_rl_db(
                "SELECT peak_equity, peak_ts, updated FROM equity_peak WHERE id=1"
            )
            peak = float(peak_row[0]["peak_equity"]) if peak_row else 0.0
            n_trades = _query_rl_db("SELECT COUNT(*) as n FROM trade_outcomes")[0]["n"]
            n_weight_adj = _query_rl_db(
                "SELECT COUNT(*) as n FROM score_weights WHERE ABS(weight - 1.0) > 0.001"
            )[0]["n"]
            n_regime_cells = _query_rl_db(
                "SELECT COUNT(*) as n FROM regime_weights"
            )[0]["n"]
            n_trail_adj = _query_rl_db(
                "SELECT COUNT(*) as n FROM trail_adjustments WHERE "
                "ABS(lock_threshold_mult - 1.0) > 0.001 OR "
                "ABS(be_threshold_mult - 1.0) > 0.001 OR "
                "ABS(trail_tightness_mult - 1.0) > 0.001"
            )[0]["n"]
            # learning velocity: audit rows in last 24h
            recent_audit = _query_rl_db(
                "SELECT COUNT(*) as n FROM rl_audit_log WHERE "
                "timestamp > datetime('now', '-24 hours')"
            )[0]["n"]
            return jsonify({
                "available": True,
                "peak_equity": round(peak, 2),
                "tracked_trades": int(n_trades or 0),
                "weight_cells_adjusted": int(n_weight_adj or 0),
                "regime_cells_adjusted": int(n_regime_cells or 0),
                "trail_symbols_adjusted": int(n_trail_adj or 0),
                "audit_events_24h": int(recent_audit or 0),
            })
        except Exception as e:
            return jsonify({"available": False, "error": str(e)})

    @app.route("/api/v2/rl_regime_weights")
    @_safe
    def v2_rl_regime_weights():
        """Full per-regime weight cells — for dashboard inspection."""
        try:
            rows = _query_rl_db(
                "SELECT symbol, regime, component, weight, win_count, loss_count, updated "
                "FROM regime_weights ORDER BY symbol, regime, component"
            )
            return jsonify([
                {"symbol": r["symbol"], "regime": r["regime"],
                 "component": r["component"], "weight": float(r["weight"]),
                 "wins": int(r["win_count"] or 0),
                 "losses": int(r["loss_count"] or 0),
                 "updated": r["updated"]}
                for r in rows
            ])
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route("/api/v2/correlation")
    @_safe
    def v2_correlation():
        now = time.time()
        # 60s cache
        if _corr_cache["data"] is not None and now - _corr_cache["ts"] < 60:
            data = _corr_cache["data"]
            return jsonify({
                **data,
                "last_update_secs_ago": int(now - _corr_cache["ts"]),
            })

        matrix_dict: Optional[dict] = None
        # Try direct portfolio_risk module on agent state
        try:
            agent = _agent()
            pr = agent.get("portfolio_risk") or {}
            matrix_dict = pr.get("correlation_matrix") or None
        except Exception:
            matrix_dict = None

        if not matrix_dict:
            # Try to call the live portfolio_risk module if attached to executor
            try:
                pr_obj = getattr(_executor, "_portfolio_risk", None) if _executor else None
                if pr_obj and hasattr(pr_obj, "get_correlation_matrix_dict"):
                    matrix_dict = pr_obj.get_correlation_matrix_dict() or None
            except Exception:
                matrix_dict = None

        if not matrix_dict:
            payload = {"symbols": [], "matrix": [], "last_update_secs_ago": -1}
            _corr_cache["data"] = payload
            _corr_cache["ts"] = now
            return jsonify(payload)

        symbols = sorted(matrix_dict.keys())
        matrix = []
        for s1 in symbols:
            row = []
            for s2 in symbols:
                row.append(float(matrix_dict.get(s1, {}).get(s2, 0.0)))
            matrix.append(row)

        payload = {
            "symbols": symbols,
            "matrix": matrix,
            "last_update_secs_ago": 0,
        }
        _corr_cache["data"] = payload
        _corr_cache["ts"] = now
        return jsonify(payload)

    @app.route("/api/v2/portfolio_risk")
    @_safe
    def v2_portfolio_risk():
        agent = _agent()
        pr = agent.get("portfolio_risk") or {}
        return jsonify({
            "var_pct": float(pr.get("var_pct", 0.0) or 0.0),
            "var_breached": bool(pr.get("var_breached", False)),
            "heat_map": pr.get("heat_map") or {},
            "flags": pr.get("flags") or [],
        })

    @app.route("/api/v2/journal")
    @_safe
    def v2_journal():
        try:
            limit = int(request.args.get("limit", 200))
            limit = max(1, min(limit, 5000))
        except Exception:
            limit = 200
        symbol = request.args.get("symbol")
        since_days = request.args.get("since_days")
        clauses = ["1=1"]
        args: list = []
        if symbol:
            clauses.append("symbol = ?")
            args.append(symbol)
        if since_days:
            try:
                days = float(since_days)
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
                clauses.append("timestamp >= ?")
                args.append(cutoff)
            except Exception:
                pass
        sql = (
            "SELECT id, timestamp, symbol, direction, entry_price, exit_price, pnl, "
            "risk_pct, score, regime, gate, duration_bars, r_multiple, "
            "session_hour, day_of_week, exit_reason, source "
            f"FROM trades WHERE {' AND '.join(clauses)} "
            "ORDER BY id DESC LIMIT ?"
        )
        args.append(limit)
        rows = _query_journal_db(sql, tuple(args))
        out = [dict(r) for r in rows]
        return jsonify({"count": len(out), "rows": out})

    @app.route("/api/v2/journal/stats")
    @_safe
    def v2_journal_stats():
        try:
            days = float(request.args.get("days", 30))
        except Exception:
            days = 30
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        rows = _query_journal_db(
            "SELECT timestamp, symbol, pnl, session_hour, day_of_week "
            "FROM trades WHERE timestamp >= ? ORDER BY id ASC",
            (cutoff,),
        )

        total = 0
        winners = 0
        losers = 0
        pnl_total = 0.0
        pnl_per_sym: Dict[str, float] = defaultdict(float)
        wins_per_sym: Dict[str, int] = defaultdict(int)
        loss_per_sym: Dict[str, int] = defaultdict(int)
        gain_sum_sym: Dict[str, float] = defaultdict(float)
        loss_sum_sym: Dict[str, float] = defaultdict(float)
        by_hour: Dict[str, float] = defaultdict(float)
        by_dow: Dict[str, float] = defaultdict(float)

        DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # equity curve: start at 0 cumulative, track running peak and drawdown
        equity_pts = []
        cum = 0.0
        peak = 0.0
        for r in rows:
            try:
                pnl = float(r["pnl"] or 0.0)
                sym = r["symbol"]
                hour = r["session_hour"]
                dow = r["day_of_week"]
                ts = r["timestamp"]
                total += 1
                pnl_total += pnl
                pnl_per_sym[sym] += pnl
                if pnl > 0:
                    winners += 1
                    wins_per_sym[sym] += 1
                    gain_sum_sym[sym] += pnl
                elif pnl < 0:
                    losers += 1
                    loss_per_sym[sym] += 1
                    loss_sum_sym[sym] += abs(pnl)
                if hour is not None:
                    by_hour[str(int(hour))] += pnl
                if dow is not None:
                    try:
                        by_dow[DOW_NAMES[int(dow)]] += pnl
                    except Exception:
                        pass

                cum += pnl
                if cum > peak:
                    peak = cum
                dd_pct = 0.0
                if peak > 0:
                    dd_pct = ((peak - cum) / peak) * 100.0
                equity_pts.append({"ts": ts, "dd_pct": round(dd_pct, 2)})
            except Exception:
                continue

        pf_per_sym: Dict[str, float] = {}
        wr_per_sym: Dict[str, float] = {}
        for sym in pnl_per_sym:
            n = wins_per_sym[sym] + loss_per_sym[sym]
            wr_per_sym[sym] = round(100.0 * wins_per_sym[sym] / n, 1) if n else 0.0
            ls = loss_sum_sym.get(sym, 0.0)
            pf_per_sym[sym] = round(gain_sum_sym.get(sym, 0.0) / ls, 2) if ls > 0 else (
                999.0 if gain_sum_sym.get(sym, 0.0) > 0 else 0.0
            )

        # downsample drawdown curve for response payload
        if len(equity_pts) > 500:
            step = max(1, len(equity_pts) // 500)
            equity_pts = equity_pts[::step]

        return jsonify({
            "total_trades": total,
            "winners": winners,
            "losers": losers,
            "pnl_total": round(pnl_total, 2),
            "pnl_per_symbol": {k: round(v, 2) for k, v in pnl_per_sym.items()},
            "pf_per_symbol": pf_per_sym,
            "wr_per_symbol": wr_per_sym,
            "by_hour": {k: round(v, 2) for k, v in by_hour.items()},
            "by_dow": {k: round(v, 2) for k, v in by_dow.items()},
            "drawdown_curve": equity_pts,
        })

    @app.route("/api/v2/alpha_attribution")
    @_safe
    def v2_alpha_attribution():
        try:
            days = float(request.args.get("days", 30))
        except Exception:
            days = 30
        since_ts = time.time() - days * 86400
        rows = _query_rl_db(
            "SELECT symbol, won, pnl, r_multiple, score, components_json "
            "FROM trade_outcomes WHERE ts >= ? ORDER BY ts DESC",
            (since_ts,),
        )

        portfolio: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {"on": [], "off": [], "on_r": [], "off_r": []}
        )
        by_sym: Dict[Tuple[str, str], Dict[str, list]] = defaultdict(
            lambda: {"on": [], "off": [], "on_r": [], "off_r": []}
        )
        for r in rows:
            comps_json = r["components_json"]
            if not comps_json:
                continue
            try:
                comps = json.loads(comps_json) or {}
            except Exception:
                continue
            if not isinstance(comps, dict):
                continue
            sym = r["symbol"]
            pnl = float(r["pnl"] or 0.0)
            r_mult = float(r["r_multiple"] or 0.0)
            for cname, cval in comps.items():
                try:
                    cv = float(cval)
                except Exception:
                    continue
                key = "on" if cv > 0 else "off"
                portfolio[cname][key].append(pnl)
                portfolio[cname][key + "_r"].append(r_mult)
                by_sym[(sym, cname)][key].append(pnl)
                by_sym[(sym, cname)][key + "_r"].append(r_mult)

        per_component: Dict[str, dict] = {}
        for cname, d in portfolio.items():
            n_on = len(d["on"])
            n_off = len(d["off"])
            avg_r_on = (sum(d["on_r"]) / n_on) if n_on else 0.0
            avg_r_off = (sum(d["off_r"]) / n_off) if n_off else 0.0
            per_component[cname] = {
                "avg_r_on": round(avg_r_on, 3),
                "avg_r_off": round(avg_r_off, 3),
                "lift": round(avg_r_on - avg_r_off, 3),
                "n": n_on + n_off,
            }

        sym_components: Dict[str, list] = defaultdict(list)
        for (sym, cname), d in by_sym.items():
            n_on = len(d["on"])
            if n_on < 3:
                continue
            avg_r_on = (sum(d["on_r"]) / n_on) if n_on else 0.0
            n_off = len(d["off"])
            avg_r_off = (sum(d["off_r"]) / n_off) if n_off else 0.0
            sym_components[sym].append({
                "component": cname,
                "lift": round(avg_r_on - avg_r_off, 3),
                "avg_r_on": round(avg_r_on, 3),
                "n": n_on,
            })
        # sort each list by lift desc
        for sym in sym_components:
            sym_components[sym].sort(key=lambda x: -x["lift"])

        return jsonify({
            "per_component": per_component,
            "by_symbol": dict(sym_components),
        })

    @app.route("/api/v2/system_health")
    @_safe
    def v2_system_health():
        from config import SYMBOLS
        agent = _agent()
        # MT5 connected proxy — agent loop publishes 'running' + 'mt5_connected' if available
        mt5_connected = bool(agent.get("mt5_connected", agent.get("running", False)))

        with _loop_latency_lock:
            samples = sorted(_loop_latency_ring)
        n = len(samples)
        p50 = samples[int(n * 0.5)] if n else 0.0
        p99 = samples[int(n * 0.99)] if n else 0.0

        # error count: tail dragon.log for ERROR lines in last hour
        errors_last_hour = 0
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            if LOG_PATH.exists():
                # tail efficiently: read last ~500KB only
                size = LOG_PATH.stat().st_size
                read_n = min(size, 500_000)
                with open(LOG_PATH, "rb") as f:
                    f.seek(size - read_n)
                    chunk = f.read(read_n).decode("utf-8", errors="ignore")
                for line in chunk.splitlines():
                    if " ERROR " in line and line[:19] >= cutoff_str:
                        errors_last_hour += 1
        except Exception:
            pass

        ml_loaded = 0
        ml_total = len(SYMBOLS)
        for sym in SYMBOLS:
            try:
                p1 = MODEL_DIR / f"{sym.replace('.', '_')}_meta_lgb_ensemble.pkl"
                p2 = MODEL_DIR / f"{sym.replace('.', '_')}_meta_lgb.pkl"
                if p1.exists() or p2.exists():
                    ml_loaded += 1
            except Exception:
                pass

        return jsonify({
            "mt5_connected": mt5_connected,
            "decision_loop_p50_ms": round(p50, 2),
            "decision_loop_p99_ms": round(p99, 2),
            "errors_last_hour": errors_last_hour,
            "uptime_secs": int(time.time() - _started_at),
            "warmup_complete": bool(agent.get("running", False) and agent.get("cycle", 0) > 5),
            "ml_models_loaded": ml_loaded,
            "ml_models_total": ml_total,
        })

    @app.route("/api/v2/backtest/latest")
    @_safe
    def v2_backtest_latest():
        if not BACKTEST_RESULTS_DIR.exists():
            return jsonify({})
        candidates = sorted(
            BACKTEST_RESULTS_DIR.glob("validate_*.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return jsonify({})
        path = candidates[0]
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return jsonify({})

        # Parse "  SYMBOL | NN trades | WR XX.X% | PF X.XX | PnL $XXX.XX | DD X.X% ..."
        pat = re.compile(
            r"^\s*([A-Za-z0-9._\-]+)\s*\|\s*(\d+)\s*trades\s*\|\s*WR\s*([\d.]+)%\s*\|\s*"
            r"PF\s*([\d.]+)\s*\|\s*PnL\s*\$\s*(-?[\d.]+)\s*\|\s*DD\s*([\d.]+)%"
        )
        per_symbol = []
        total_trades = 0
        total_pnl = 0.0
        winners = 0
        losers = 0
        for line in text.splitlines():
            m = pat.match(line)
            if not m:
                continue
            sym, trades, wr, pf, pnl, dd = m.groups()
            trades_i = int(trades)
            wr_f = float(wr)
            pnl_f = float(pnl)
            per_symbol.append({
                "symbol": sym,
                "pf": float(pf),
                "pnl": pnl_f,
                "wr": wr_f,
                "dd": float(dd),
                "trades": trades_i,
            })
            total_trades += trades_i
            total_pnl += pnl_f
            wins_est = int(round(trades_i * wr_f / 100.0))
            winners += wins_est
            losers += max(trades_i - wins_est, 0)

        # Pull explicit TOTAL line if present
        m = re.search(r"TOTAL:\s*(\d+)\s*trades\s*\|\s*PnL\s*\$([\-\d.]+)", text)
        if m:
            total_trades = int(m.group(1))
            try:
                total_pnl = float(m.group(2))
            except Exception:
                pass

        # Try to extract days from "Period: NN days"
        days = None
        m2 = re.search(r"Period:\s*(\d+)\s*days", text)
        if m2:
            days = int(m2.group(1))

        return jsonify({
            "timestamp": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "file": path.name,
            "days": days,
            "total_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "winners": winners,
            "losers": losers,
            "per_symbol": per_symbol,
        })

    @app.route("/api/v2/decisions/recent")
    @_safe
    def v2_decisions_recent():
        try:
            limit = int(request.args.get("limit", 100))
            limit = max(1, min(limit, 500))
        except Exception:
            limit = 100
        with _decision_lock:
            buf = list(_decision_buffer)
        # newest first
        buf.reverse()
        return jsonify(buf[:limit])

    log.info("v2 routes registered (/api/v2/*)")


# ────────────────────────────────────────────────────────────────────
#  Background push threads
# ────────────────────────────────────────────────────────────────────


def _push_ticks_bulk():
    """Every 500ms emit ticks:bulk for all symbols (incl. live trend/IMR basket)."""
    from config import SYMBOLS
    try:
        from config import AUX_SYMBOLS as _AUX
    except Exception:
        _AUX = {}
    _DISP = {**SYMBOLS, **_AUX}
    while True:
        time.sleep(0.5)
        if _state is None or _socketio is None:
            continue
        try:
            ticks = []
            for sym, cfg in _DISP.items():
                try:
                    t = _state.get_tick(sym)
                    if not t:
                        continue
                    ticks.append({
                        "symbol": sym,
                        "bid": float(t.bid),
                        "ask": float(t.ask),
                        "spread": round((float(t.ask) - float(t.bid)) * (10 ** cfg.digits), 2),
                    })
                except Exception:
                    continue
            _socketio.emit("ticks:bulk", {"ts": time.time(), "ticks": ticks})
        except Exception as e:
            log.debug("ticks:bulk push error: %s", e)


def _push_portfolio_update():
    """Every 1s emit portfolio:update summary."""
    while True:
        time.sleep(1.0)
        if _state is None or _socketio is None:
            continue
        try:
            agent = _agent()
            balance = float(agent.get("balance", 0.0))
            equity = float(agent.get("equity", 0.0))
            dd_pct = float(agent.get("dd_pct", 0.0))
            open_pnl = equity - balance
            n_pos = 0
            try:
                if _executor is not None:
                    pos = _executor.get_positions_info() or []
                    n_pos = len(pos)
                else:
                    n_pos = len(agent.get("positions", []) or [])
            except Exception:
                n_pos = len(agent.get("positions", []) or [])
            _socketio.emit("portfolio:update", {
                "balance": round(balance, 2),
                "equity": round(equity, 2),
                "drawdown_pct": round(dd_pct, 2),
                "open_pnl": round(open_pnl, 2),
                "n_positions": int(n_pos),
            })
        except Exception as e:
            log.debug("portfolio:update push error: %s", e)


# ────────────────────────────────────────────────────────────────────
#  Registration entry point
# ────────────────────────────────────────────────────────────────────


def register_v2(app, socketio, state=None, executor=None) -> None:
    """Wire v2 endpoints + start push threads. Idempotent."""
    global _app, _socketio, _state, _executor
    _app = app
    _socketio = socketio
    if state is not None:
        _state = state
    if executor is not None:
        _executor = executor

    # Hook alerter SocketIO backend if available
    try:
        from agent.alerting import get_default_alerter, SocketIOBackend  # type: ignore
        a = get_default_alerter()
        already = any(getattr(b, "name", "") == "socketio" for b in getattr(a, "_backends", []))
        if not already:
            a._backends.append(SocketIOBackend(socketio, push_alert))
    except Exception as e:
        log.debug("SocketIOBackend wire failed (alerts will not stream): %s", e)

    # Register routes (only once — Flask raises if duplicate rule)
    if not getattr(app, "_v2_registered", False):
        _register_routes(app)
        app._v2_registered = True


def init_v2_state(state=None, executor=None) -> None:
    """Late-binding helper: dashboard/app.py calls init_dashboard() before
    routes know about state/executor. v2_api also needs the refs."""
    global _state, _executor
    if state is not None:
        _state = state
    if executor is not None:
        _executor = executor


def start_push_threads() -> None:
    """Start v2 background WS push threads. Call once after socketio is ready."""
    threading.Thread(target=_push_ticks_bulk, daemon=True, name="V2PushTicksBulk").start()
    threading.Thread(target=_push_portfolio_update, daemon=True, name="V2PushPortfolio").start()
    log.info("v2 push threads started (ticks:bulk, portfolio:update)")
