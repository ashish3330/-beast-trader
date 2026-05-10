#!/usr/bin/env python3 -B
"""
Chaos test: prove the durability stack heals from a bridge kill.

Procedure:
  1. Snapshot current trader state (PID, last brain cycle, last log mtime).
  2. Verify dragon brain is currently healthy.
  3. Kill the rpyc bridge process for MT5_PORT.
  4. Watch dragon.log for ~120s, recording:
       - Did `MT5 DEGRADED cycle ...` lines appear? (proves facade detected drop)
       - Did the trader process die? (it should NOT — we want graceful degrade)
       - Did `MT5 RECOVERED` appear? (proves facade reconnected)
       - Was a new connection_events row written? (proves telemetry flowed)
  5. Print PASS/FAIL summary and recovery time.

This is destructive — it kills your live bridge. Do not run during active
trading hours unless you mean it. Use --dry-run to preview.
"""
import argparse
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import MT5_PORT, DB_PATH

DRAGON_LOG = ROOT / "logs" / "dragon.log"
WATCH_SECONDS = 120
SLO_RECOVERY_S = 90  # PASS if MT5 RECOVERED within this window


def find_bridge_pid(port: int) -> int | None:
    """Find the wineserver PID listening on the given port."""
    try:
        r = subprocess.run(
            ["lsof", "-iTCP:%d" % port, "-sTCP:LISTEN", "-Pn", "-Fp"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if line.startswith("p"):
                return int(line[1:])
    except Exception as e:
        print(f"lsof error: {e}")
    return None


def trader_pid() -> int | None:
    try:
        r = subprocess.run(
            ["pgrep", "-f", "beast-trader/run.py"],
            capture_output=True, text=True, timeout=5,
        )
        s = r.stdout.strip()
        return int(s.splitlines()[0]) if s else None
    except Exception:
        return None


def tail_seek(path: Path, start_offset: int) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(start_offset)
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def count_events_since(ts: float) -> int:
    try:
        with sqlite3.connect(str(DB_PATH), timeout=3.0) as c:
            r = c.execute(
                "SELECT COUNT(*) FROM connection_events WHERE ts > ?",
                (ts,),
            ).fetchone()
            return int(r[0]) if r else 0
    except sqlite3.OperationalError:
        return 0
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't actually kill the bridge — just probe state.")
    ap.add_argument("--port", type=int, default=MT5_PORT,
                    help=f"Bridge port to kill (default: MT5_PORT={MT5_PORT})")
    args = ap.parse_args()

    print(f"== chaos: kill bridge on port {args.port} ==")

    tpid = trader_pid()
    if not tpid:
        print("FAIL: dragon trader not running. Start it first.")
        return 2
    print(f"trader PID: {tpid}")

    bpid = find_bridge_pid(args.port)
    if not bpid:
        print(f"FAIL: no bridge listening on port {args.port}")
        return 2
    print(f"bridge PID on {args.port}: {bpid}")

    if not DRAGON_LOG.exists():
        print(f"FAIL: dragon.log not found at {DRAGON_LOG}")
        return 2
    start_offset = DRAGON_LOG.stat().st_size
    start_ts = time.time()
    pre_event_count = count_events_since(start_ts - 60)
    print(f"baseline: log offset={start_offset} pre_events_60s={pre_event_count}")

    if args.dry_run:
        print("DRY RUN — skipping kill. Probes look good.")
        return 0

    print(f"kill -9 {bpid}")
    try:
        os.kill(bpid, 9)
    except ProcessLookupError:
        print("bridge already gone — proceeding to watch")
    except PermissionError:
        print(f"FAIL: cannot kill PID {bpid} (permission denied)")
        return 2

    # Verify port is actually closed.
    time.sleep(2)
    port_open = False
    try:
        with socket.create_connection(("localhost", args.port), timeout=2):
            port_open = True
    except Exception:
        pass
    print(f"post-kill port {args.port} open={port_open}")

    # Watch the log.
    degraded_seen_at = None
    recovered_seen_at = None
    trader_died = False
    deadline = time.time() + WATCH_SECONDS

    while time.time() < deadline:
        time.sleep(2)
        # Trader still alive?
        cur_tpid = trader_pid()
        if cur_tpid != tpid:
            trader_died = True
            print(f"WARN: trader PID changed {tpid} → {cur_tpid} (process restarted)")
            tpid = cur_tpid

        new_text = tail_seek(DRAGON_LOG, start_offset)
        for line in new_text.splitlines():
            if "MT5 DEGRADED" in line and degraded_seen_at is None:
                degraded_seen_at = time.time()
                print(f"[{degraded_seen_at - start_ts:6.1f}s] DEGRADED detected: {line[:160]}")
            elif "MT5 RECOVERED" in line and recovered_seen_at is None:
                recovered_seen_at = time.time()
                print(f"[{recovered_seen_at - start_ts:6.1f}s] RECOVERED: {line[:160]}")
        if recovered_seen_at:
            break

    post_events = count_events_since(start_ts)
    print()
    print("== results ==")
    print(f"  trader_restart_count   = {1 if trader_died else 0} (target: 0)")
    print(f"  degraded_log_emitted   = {bool(degraded_seen_at)} (target: True)")
    print(f"  recovered_log_emitted  = {bool(recovered_seen_at)} (target: True)")
    print(f"  connection_events_new  = {post_events} (target: >=1)")
    if recovered_seen_at:
        recovery_s = recovered_seen_at - start_ts
        print(f"  recovery_time_s        = {recovery_s:.1f} (SLO: <{SLO_RECOVERY_S}s)")
    else:
        recovery_s = None

    pass_ = (
        not trader_died
        and degraded_seen_at is not None
        and recovered_seen_at is not None
        and post_events >= 1
        and recovery_s is not None
        and recovery_s < SLO_RECOVERY_S
    )
    print()
    print("PASS ✓" if pass_ else "FAIL ✗")
    return 0 if pass_ else 1


if __name__ == "__main__":
    sys.exit(main())
