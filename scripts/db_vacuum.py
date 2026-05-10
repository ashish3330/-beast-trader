#!/usr/bin/env python3 -B
"""
Weekly SQLite VACUUM for Dragon DBs.

beast.db is hit by daily DELETEs (per the audit, finding #9: 1GB and growing
without VACUUM). trade_journal.db accumulates closed deals + connection_events.
SQLite reclaims space only on VACUUM.

Schedule: launchd com.dragon.db-vacuum, Sunday 21:30 UTC (after weekly
walk-forward at Sun 06:30, before market reopen Sun 22:00 UTC).

Logs before/after sizes so a leak is visible.
"""
import logging
import logging.handlers
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "db_vacuum.log"

DBS = (
    ROOT / "data" / "trade_journal.db",
    ROOT / "data" / "beast.db",
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("db-vacuum")
_h = logging.handlers.RotatingFileHandler(
    str(LOG_PATH), maxBytes=2 * 1024 * 1024, backupCount=4
)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_h)


def vacuum_db(path: Path) -> int:
    if not path.exists():
        log.warning("DB not found, skipping: %s", path)
        return 0
    size_before = path.stat().st_size
    t0 = time.time()
    try:
        # No timeout argument because VACUUM is exclusive — let it block.
        # isolation_level=None enables autocommit so the VACUUM commits cleanly.
        with sqlite3.connect(str(path), isolation_level=None, timeout=300) as c:
            # WAL truncate first to flush any pending journal pages.
            try:
                c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as e:
                log.debug("wal_checkpoint failed (ok if non-WAL): %s", e)
            c.execute("VACUUM")
        size_after = path.stat().st_size
        elapsed = time.time() - t0
        delta = size_before - size_after
        log.info("VACUUM %s | before=%d after=%d freed=%d (%.1f%%) in %.1fs",
                 path.name, size_before, size_after, delta,
                 100.0 * delta / max(1, size_before), elapsed)
        return delta
    except sqlite3.OperationalError as e:
        log.error("VACUUM %s FAILED (likely locked by trader): %s", path.name, e)
        return 0
    except Exception as e:
        log.error("VACUUM %s exception: %s", path.name, e)
        return 0


def main():
    log.info("Weekly VACUUM starting on %d DBs", len(DBS))
    total_freed = 0
    for db in DBS:
        total_freed += vacuum_db(db)
    log.info("Weekly VACUUM done | total_freed_bytes=%d", total_freed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
