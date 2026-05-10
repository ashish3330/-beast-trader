#!/usr/bin/env python3
"""Daily learning summary — single aggregator across all 4 learning loops.

Pulls last-24h state from each loop and writes a unified daily report:
  1. RL learner: per-symbol weight changes since prior snapshot
  2. Drift detector: symbol-level state shifts (OK ↔ LIGHT ↔ HEAVY)
  3. ML meta-labels: model AUC vs install AUC (decay detection)
  4. Live trades: PnL, top contributors, top leakers, regime split
  5. Live-vs-backtest divergence: aggregate ratio, flagged symbols

Output:
  logs/learning_summary.log     — append-only daily history
  logs/learning_summary_latest.json   — latest snapshot

Schedule: daily at 05:00 UTC (after live-drift at 04:30).

Why: "smarter day by day" without a daily report is faith, not measurement.
"""
from __future__ import annotations
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
RL_DB = ROOT / "data" / "rl_learner.db"
JOURNAL_DB = ROOT / "data" / "trade_journal.db"
DRIFT_JSON = ROOT / "logs" / "live_drift.json"
LATEST_JSON = ROOT / "logs" / "learning_summary_latest.json"
PRIOR_JSON = ROOT / "logs" / "learning_summary_prior.json"
LOG_FILE = ROOT / "logs" / "learning_summary.log"


def _setup_log():
    log = logging.getLogger("dragon.learning")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.FileHandler(LOG_FILE, mode="a")
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(h)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(sh)
    return log


def _live_universe():
    sys.path.insert(0, str(ROOT))
    import config
    return set(config.SYMBOLS.keys())


def _rl_snapshot(LIVE):
    """Per-symbol RL snapshot: number of components touched, total |delta_weight|, win/loss counts."""
    out = {}
    if not RL_DB.exists():
        return out
    try:
        con = sqlite3.connect(f"file:{RL_DB}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT symbol, component, weight, win_count, loss_count FROM score_weights"
        ).fetchall()
        con.close()
    except Exception:
        return out
    for sym, comp, w, wins, losses in rows:
        if sym not in LIVE:
            continue
        d = out.setdefault(sym, {"components": {}, "total_wins": 0, "total_losses": 0})
        d["components"][comp] = float(w)
        d["total_wins"] += int(wins or 0)
        d["total_losses"] += int(losses or 0)
    return out


def _drift_snapshot(LIVE):
    """Per-symbol drift state from journal."""
    out = {}
    if not JOURNAL_DB.exists():
        return out
    try:
        con = sqlite3.connect(f"file:{JOURNAL_DB}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT symbol, state, wr_recent, pf_recent, n_recent FROM symbol_drift_state"
        ).fetchall()
        con.close()
    except Exception:
        return out
    for sym, state, wr, pf, n in rows:
        if sym not in LIVE:
            continue
        out[sym] = {"state": state, "wr": wr, "pf": pf, "n": n}
    return out


def _trades_24h(LIVE):
    """Last 24h trades from journal — per-symbol PnL + counts."""
    out = {"per_symbol": {}, "total_pnl": 0.0, "total_trades": 0, "wins": 0, "losses": 0}
    if not JOURNAL_DB.exists():
        return out
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    try:
        con = sqlite3.connect(f"file:{JOURNAL_DB}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT symbol, COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), "
            "COALESCE(SUM(pnl),0), COALESCE(AVG(r_multiple),0) "
            "FROM trades WHERE timestamp >= ? GROUP BY symbol",
            (cutoff,)
        ).fetchall()
        con.close()
    except Exception:
        return out
    for sym, n, wins, pnl, avg_r in rows:
        if sym not in LIVE:
            continue
        out["per_symbol"][sym] = {
            "trades": int(n), "wins": int(wins or 0),
            "pnl": float(pnl), "avg_r": float(avg_r),
        }
        out["total_pnl"] += float(pnl)
        out["total_trades"] += int(n)
        out["wins"] += int(wins or 0)
        out["losses"] += int(n) - int(wins or 0)
    return out


def _ml_auc_snapshot(LIVE):
    """ML model AUC per symbol from saved model metadata."""
    out = {}
    model_dir = ROOT / "models" / "saved"
    if not model_dir.exists():
        return out
    # Each model saves AUC in its filename or metadata; brain logs AUC at startup.
    # Simplest: parse the most recent dragon.log boot block for "Meta-label model loaded (AUC=...)"
    log_path = ROOT / "logs" / "dragon.log"
    if not log_path.exists():
        return out
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            tail = f.read().decode("utf-8", errors="replace").splitlines()
    except Exception:
        return out
    # Find latest "Meta-label filter ENABLED" boundary and read AUC lines just above it
    boundaries = [i for i, l in enumerate(tail) if "Meta-label filter ENABLED" in l]
    if not boundaries:
        return out
    last_boundary = boundaries[-1]
    block = tail[max(0, last_boundary - 100):last_boundary]
    for line in block:
        # "[XAUUSD] Meta-label model loaded (AUC=0.554 >= 0.55)"
        # "[DJ30.r] Meta-label AUC=0.537 < 0.55 — disabled for this symbol"
        if "Meta-label" not in line or "AUC=" not in line:
            continue
        try:
            sym_part = line.split("[", 1)[1].split("]", 1)[0]
            if sym_part not in LIVE:
                continue
            auc_part = line.split("AUC=", 1)[1]
            auc_str = auc_part.split()[0].rstrip(",")
            auc = float(auc_str)
            disabled = "disabled" in line
            out[sym_part] = {"auc": auc, "enabled": not disabled}
        except (ValueError, IndexError):
            continue
    return out


def _live_drift_snapshot():
    """Read latest live-drift JSON if available."""
    if not DRIFT_JSON.exists():
        return None
    try:
        return json.loads(DRIFT_JSON.read_text())
    except Exception:
        return None


def _format_top(d, key, n=5, reverse=True):
    """Top N (sym, value) by |key|."""
    pairs = [(s, v.get(key, 0)) for s, v in d.items()]
    pairs.sort(key=lambda x: x[1], reverse=reverse)
    return pairs[:n]


def main() -> int:
    log = _setup_log()
    log.info("════════════════════════════════════════════════════════════")
    log.info("DAILY LEARNING SUMMARY — %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    log.info("════════════════════════════════════════════════════════════")

    LIVE = _live_universe()
    log.info("Live universe: %d symbols", len(LIVE))

    # 1. RL state
    rl = _rl_snapshot(LIVE)
    rl_total_wins = sum(d["total_wins"] for d in rl.values())
    rl_total_losses = sum(d["total_losses"] for d in rl.values())
    rl_components = sum(len(d["components"]) for d in rl.values())
    log.info("[RL] %d/%d symbols have weights | %d component cells | total wins=%d losses=%d",
             len(rl), len(LIVE), rl_components, rl_total_wins, rl_total_losses)

    # Compare to prior to detect actual learning
    prior = {}
    if PRIOR_JSON.exists():
        try:
            prior = json.loads(PRIOR_JSON.read_text())
        except Exception:
            prior = {}
    prior_rl = (prior.get("rl_components") or {})
    weight_changes = []
    for sym, d in rl.items():
        psym = prior_rl.get(sym, {}).get("components", {})
        for comp, w in d["components"].items():
            pw = psym.get(comp)
            if pw is not None and abs(w - pw) > 0.01:
                weight_changes.append((sym, comp, pw, w, w - pw))
    if weight_changes:
        weight_changes.sort(key=lambda x: abs(x[4]), reverse=True)
        log.info("[RL] %d weight changes since last summary; top 3:", len(weight_changes))
        for sym, comp, pw, w, d in weight_changes[:3]:
            log.info("[RL]   %s.%s %.2f → %.2f (Δ%+.2f)", sym, comp, pw, w, d)
    elif prior:
        log.info("[RL] no weight changes since prior summary (no recent trade outcomes)")

    # 2. Drift state
    drift = _drift_snapshot(LIVE)
    state_counts = {}
    for d in drift.values():
        state_counts[d["state"]] = state_counts.get(d["state"], 0) + 1
    log.info("[DRIFT] %d/%d symbols have state | counts: %s",
             len(drift), len(LIVE), dict(state_counts))
    heavy = [(s, d) for s, d in drift.items() if d["state"] == "HEAVY"]
    if heavy:
        log.warning("[DRIFT] HEAVY drift on: %s",
                    ", ".join(f"{s}(wr={d['wr']:.0%},pf={d['pf']:.2f})" for s, d in heavy[:5]))

    # 3. ML AUC
    ml = _ml_auc_snapshot(LIVE)
    if ml:
        enabled = [s for s, d in ml.items() if d["enabled"]]
        disabled = [s for s, d in ml.items() if not d["enabled"]]
        log.info("[ML] %d symbols with models | enabled=%d disabled=%d",
                 len(ml), len(enabled), len(disabled))
        if disabled:
            log.info("[ML] disabled (AUC<0.55): %s", ", ".join(disabled))
        # AUC decay detection vs prior
        prior_ml = prior.get("ml_models") or {}
        decayed = []
        for sym, d in ml.items():
            p = prior_ml.get(sym)
            if p and "auc" in p and d["auc"] < p["auc"] - 0.03:
                decayed.append((sym, p["auc"], d["auc"]))
        if decayed:
            log.warning("[ML] AUC decay detected: %s",
                        ", ".join(f"{s}({a1:.3f}→{a2:.3f})" for s, a1, a2 in decayed))

    # 4. Trades 24h
    trades = _trades_24h(LIVE)
    if trades["total_trades"] > 0:
        wr = trades["wins"] / trades["total_trades"] * 100
        log.info("[TRADES 24h] %d trades | wins=%d losses=%d WR=%.1f%% | PnL=$%+.2f",
                 trades["total_trades"], trades["wins"], trades["losses"], wr, trades["total_pnl"])
        # Top contributors / leakers
        per = trades["per_symbol"]
        if per:
            top_pnl = sorted(per.items(), key=lambda kv: kv[1]["pnl"], reverse=True)
            best = [(s, d) for s, d in top_pnl[:3] if d["pnl"] > 0]
            worst = [(s, d) for s, d in top_pnl[-3:] if d["pnl"] < 0]
            if best:
                log.info("[TRADES] top earners: %s",
                         ", ".join(f"{s}(${d['pnl']:+.0f},n={d['trades']})" for s, d in best))
            if worst:
                log.info("[TRADES] top leakers: %s",
                         ", ".join(f"{s}(${d['pnl']:+.0f},n={d['trades']})" for s, d in worst))
    else:
        log.info("[TRADES 24h] no trades in last 24h (markets closed or kill switch active)")

    # 4b. Connection health (ResilientMT5Client telemetry)
    try:
        import sqlite3
        from config import DB_PATH
        with sqlite3.connect(str(DB_PATH), timeout=3.0) as c:
            cutoff = (datetime.now(timezone.utc).timestamp() - 86400)
            rows = c.execute(
                "SELECT cause, downtime_ms, attempts FROM connection_events "
                "WHERE ts > ? ORDER BY ts DESC",
                (cutoff,),
            ).fetchall()
        n = len(rows)
        if n == 0:
            log.info("[MT5 CONN] 0 reconnects in 24h — clean")
        else:
            total_downtime_s = sum(r[1] for r in rows) / 1000.0
            avg_attempts = sum(r[2] for r in rows) / n
            causes = {}
            for r in rows:
                causes[r[0]] = causes.get(r[0], 0) + 1
            top_cause = max(causes.items(), key=lambda kv: kv[1])
            warn = " ⚠ HIGH" if n > 5 else ""
            log.info("[MT5 CONN]%s %d reconnects in 24h | total_downtime=%.1fs avg_attempts=%.1f top_cause=%s(%d)",
                     warn, n, total_downtime_s, avg_attempts, top_cause[0], top_cause[1])
    except sqlite3.OperationalError:
        log.info("[MT5 CONN] table not yet created (no reconnects since ResilientMT5Client deploy)")
    except Exception as e:
        log.warning("[MT5 CONN] snapshot failed: %s", e)

    # 5. Live-vs-backtest
    live_drift = _live_drift_snapshot()
    if live_drift:
        agg = live_drift.get("agg_ratio")
        sym_warns = live_drift.get("symbol_warnings", [])
        agg_warns = live_drift.get("aggregate_warnings", [])
        if agg is not None:
            log.info("[DRIFT MONITOR] live/backtest ratio = %.0f%% | aggregate flags=%d symbol flags=%d",
                     agg * 100, len(agg_warns), len(sym_warns))
        else:
            log.info("[DRIFT MONITOR] insufficient data for ratio calc")

    # Save current as prior for next run
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "live_universe_count": len(LIVE),
        "rl_components": rl,
        "drift_state": drift,
        "ml_models": ml,
        "trades_24h": trades,
    }
    LATEST_JSON.write_text(json.dumps(snapshot, indent=2))
    PRIOR_JSON.write_text(json.dumps(snapshot, indent=2))

    log.info("════════════════════════════════════════════════════════════")
    return 0


if __name__ == "__main__":
    sys.exit(main())
