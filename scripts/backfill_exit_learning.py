#!/usr/bin/env python3 -B
"""
One-shot backfill: populate exit_learning from rl_learner.db.trade_outcomes.

The exit_learning table was created but never written to (indentation bug fixed
2026-05-04). Run once after deploying the rl_learner.py fix so the table is
not waiting on the next 50 trades to gain useful aggregates.

Idempotent — safely re-runnable.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RL_DB = ROOT / "data" / "rl_learner.db"


def main() -> None:
    conn = sqlite3.connect(str(RL_DB), timeout=10.0)
    rows = conn.execute(
        "SELECT symbol, exit_reason, r_multiple FROM trade_outcomes "
        "ORDER BY ts ASC"
    ).fetchall()

    # bucket by (symbol, simplified_exit_reason), keeping last 50 per bucket
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sym, reason, r_multiple in rows:
        if not sym or r_multiple is None:
            continue
        reason = str(reason or "")
        if "sl" in reason.lower():
            key = "SL"
        else:
            key = reason.split("[")[0].strip() or "UNKNOWN"
        buckets[(sym, key)].append(float(r_multiple))
        if len(buckets[(sym, key)]) > 50:
            buckets[(sym, key)] = buckets[(sym, key)][-50:]

    now_iso = datetime.now(timezone.utc).isoformat()
    written = 0
    for (sym, key), samples in buckets.items():
        n = len(samples)
        avg_r = float(sum(samples) / n)
        best_r = float(max(samples))
        worst_r = float(min(samples))
        conn.execute(
            "INSERT OR REPLACE INTO exit_learning "
            "(symbol, exit_reason, count, avg_r, best_r, worst_r, updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sym, key, n, avg_r, best_r, worst_r, now_iso),
        )
        written += 1
    conn.commit()
    conn.close()
    print(f"backfilled exit_learning: {written} (symbol, exit_reason) rows from "
          f"{len(rows)} historical outcomes")


if __name__ == "__main__":
    main()
