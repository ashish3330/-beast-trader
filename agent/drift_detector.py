"""
Per-symbol live drift detector + adaptive risk reduction.

Watches recent N-trade WR/PF per symbol from the live trade journal.
When live performance falls below configured baselines, returns a
risk multiplier < 1.0 so master_brain reduces exposure WITHOUT
skipping signals (user rule: never skip, only adjust).

Severely drifted symbols also get queued for retrain via a JSON queue
file that a separate consumer job can process — the brain itself does
not spawn long-running training subprocesses.

Author intent: this is the "auto-detect drift, halve risk, re-tune"
feedback loop the existing tuner pipeline never had a live trigger for.

Wiring:
  - scripts/drift_monitor.py runs update_all() every 5 min via launchd
  - master_brain.calculate_swing_risk() reads get_risk_multiplier()
  - retrain consumer (TBD) drains data/retrain_queue.json
"""
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, Tuple

log = logging.getLogger("dragon.drift")

ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DB = ROOT / "data" / "trade_journal.db"
RETRAIN_QUEUE = ROOT / "data" / "retrain_queue.json"

# Window + thresholds
N_RECENT_TRADES = 10            # trailing trades evaluated per symbol
MIN_TRADES_FOR_DECISION = 5     # below this, stay OK regardless
WR_LIGHT = 0.50                 # WR < 50% AND PF < 1.0 → LIGHT
WR_HEAVY = 0.30                 # WR < 30% OR PF < 0.5 → HEAVY
PF_LIGHT = 1.0
PF_HEAVY = 0.5

LIGHT_RISK_MULT = 0.5
HEAVY_RISK_MULT = 0.25

MIN_RETRAIN_GAP_HOURS = 24      # cooldown between retrain queues per symbol
LOOKBACK_DAYS = 30              # only consider trades within this window


def _init_table() -> None:
    try:
        conn = sqlite3.connect(str(JOURNAL_DB), timeout=5.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS symbol_drift_state ("
            "symbol TEXT PRIMARY KEY, "
            "state TEXT NOT NULL, "
            "wr_recent REAL, "
            "pf_recent REAL, "
            "n_recent INTEGER, "
            "last_updated_ts REAL, "
            "last_retrain_ts REAL DEFAULT 0)"
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("symbol_drift_state init failed: %s", e)


def _classify(wr: float, pf: float, n: int) -> str:
    if n < MIN_TRADES_FOR_DECISION:
        return "OK"
    if wr < WR_HEAVY or pf < PF_HEAVY:
        return "HEAVY"
    if wr < WR_LIGHT and pf < PF_LIGHT:
        return "LIGHT"
    return "OK"


def update_all() -> Dict[str, dict]:
    """Recompute drift state for every symbol with recent trades."""
    _init_table()
    out: Dict[str, dict] = {}
    try:
        conn = sqlite3.connect(str(JOURNAL_DB), timeout=10.0)
        symbols = [r[0] for r in conn.execute(
            f"SELECT DISTINCT symbol FROM trades "
            f"WHERE timestamp >= datetime('now','-{LOOKBACK_DAYS} days')"
        ).fetchall()]

        for sym in symbols:
            rows = conn.execute(
                "SELECT pnl FROM trades "
                "WHERE symbol=? AND timestamp >= datetime('now','-' || ? || ' days') "
                "ORDER BY id DESC LIMIT ?",
                (sym, LOOKBACK_DAYS, N_RECENT_TRADES)
            ).fetchall()
            n = len(rows)
            if n == 0:
                continue

            pnls = [float(r[0]) for r in rows]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / n
            gross_win = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            if gross_loss > 0:
                pf = gross_win / gross_loss
            elif gross_win > 0:
                pf = 999.0
            else:
                pf = 1.0

            state = _classify(wr, pf, n)
            now = time.time()

            prior_row = conn.execute(
                "SELECT state, last_retrain_ts FROM symbol_drift_state WHERE symbol=?",
                (sym,)
            ).fetchone()
            prior_state = prior_row[0] if prior_row else None
            last_retrain_ts = float(prior_row[1]) if prior_row else 0.0

            # Queue retrain on first entry into HEAVY state
            if state == "HEAVY" and prior_state != "HEAVY":
                hours_since = (now - last_retrain_ts) / 3600
                if hours_since >= MIN_RETRAIN_GAP_HOURS:
                    _queue_retrain(sym, wr, pf, n)
                    last_retrain_ts = now
                    log.warning(
                        "DRIFT HEAVY %s: WR=%.0f%% PF=%.2f n=%d — queued retrain",
                        sym, wr * 100, pf, n,
                    )
                else:
                    log.info(
                        "DRIFT HEAVY %s but retrain cooldown %.1fh remaining",
                        sym, MIN_RETRAIN_GAP_HOURS - hours_since,
                    )
            elif prior_state and prior_state != state:
                log.info(
                    "DRIFT TRANSITION %s: %s -> %s (WR=%.0f%% PF=%.2f n=%d)",
                    sym, prior_state, state, wr * 100, pf, n,
                )

            conn.execute(
                "INSERT OR REPLACE INTO symbol_drift_state "
                "(symbol, state, wr_recent, pf_recent, n_recent, last_updated_ts, last_retrain_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sym, state, wr, pf, n, now, last_retrain_ts),
            )
            out[sym] = {"state": state, "wr": round(wr, 3), "pf": round(pf, 2), "n": n}

        conn.commit()
        conn.close()
    except Exception as e:
        log.error("drift update_all failed: %s", e)
    return out


def get_risk_multiplier(symbol: str) -> Tuple[float, str]:
    """Cheap per-cycle read. Returns (multiplier, state)."""
    try:
        conn = sqlite3.connect(str(JOURNAL_DB), timeout=2.0)
        row = conn.execute(
            "SELECT state FROM symbol_drift_state WHERE symbol=?", (symbol,)
        ).fetchone()
        conn.close()
        if not row:
            return 1.0, "OK"
        state = row[0]
        mult = {"OK": 1.0, "LIGHT": LIGHT_RISK_MULT, "HEAVY": HEAVY_RISK_MULT}.get(state, 1.0)
        return mult, state
    except Exception:
        return 1.0, "OK"


def _queue_retrain(symbol: str, wr: float, pf: float, n: int) -> None:
    try:
        if RETRAIN_QUEUE.exists():
            with open(RETRAIN_QUEUE) as f:
                queue = json.load(f)
        else:
            queue = {"items": []}
        queue.setdefault("items", []).append({
            "symbol": symbol,
            "queued_ts": time.time(),
            "queued_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trigger_wr": round(wr, 3),
            "trigger_pf": round(pf, 2),
            "trigger_n": n,
        })
        with open(RETRAIN_QUEUE, "w") as f:
            json.dump(queue, f, indent=2)
    except Exception as e:
        log.warning("queue_retrain failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    result = update_all()
    print(json.dumps(result, indent=2, default=str))
