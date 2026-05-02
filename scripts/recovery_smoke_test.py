#!/usr/bin/env python3
"""
Dragon Trader — Recovery Smoke Test.

⚠️  DANGER: --actually-test will SIGTERM the live agent.
    Default mode is --dry-run. Only override on a quiet demo session.

Verifies that the live agent can be restarted (by launchd) without losing
SQLite state. Procedure:

  1. Snapshot SHA-256 of every DB in data/*.db.
  2. Confirm launchd watchdog plist is loaded and the agent process is up.
  3. (real test only) Send SIGTERM to the agent process.
  4. Wait up to N seconds for it to come back via launchd KeepAlive.
  5. Re-snapshot DB checksums; confirm tables still queryable.
  6. Print PASS/FAIL.

Why undertested historically: peak_R clamp + revert-trail bugs both came from
state-recovery edge cases. This script exercises the restart path.

Usage:
    python3 -B scripts/recovery_smoke_test.py             # dry-run (default)
    python3 -B scripts/recovery_smoke_test.py --actually-test
"""
from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RUN_PY = REPO_ROOT / "run.py"
WATCHDOG_LABEL = "com.dragon.watchdog"
RESTART_TIMEOUT_S = 90  # generous; launchd ThrottleInterval=30 + boot time
POLL_INTERVAL_S = 2


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _snapshot_dbs() -> Dict[str, Tuple[int, str]]:
    """Map db_name -> (size, sha256). Skips 0-byte files."""
    out: Dict[str, Tuple[int, str]] = {}
    for db in sorted(DATA_DIR.glob("*.db")):
        size = db.stat().st_size
        if size == 0:
            continue
        out[db.name] = (size, _sha256(db))
    return out


def _query_tables(db_path: Path) -> List[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def _find_agent_pids() -> List[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "beast-trader/run.py"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return []
    return [int(p) for p in out.splitlines() if p.strip().isdigit()]


def _launchd_loaded(label: str) -> bool:
    try:
        out = subprocess.check_output(
            ["launchctl", "list"], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return False
    return any(label in line for line in out.splitlines())


def _wait_for_pid(prev_pids: List[int], timeout_s: int) -> List[int]:
    """Wait until a NEW pid (not in prev_pids) appears."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pids = _find_agent_pids()
        new = [p for p in pids if p not in prev_pids]
        if new:
            return new
        time.sleep(POLL_INTERVAL_S)
    return []


# ═══════════════════════════════════════════════════════════════
#  Test phases
# ═══════════════════════════════════════════════════════════════

def phase_preflight(verbose: bool = True) -> Tuple[bool, Dict]:
    """Validate the framework without touching the live agent."""
    info: Dict = {}
    info["data_dir_exists"] = DATA_DIR.is_dir()
    info["run_py_exists"] = RUN_PY.is_file()
    info["agent_pids"] = _find_agent_pids()
    info["watchdog_loaded"] = _launchd_loaded(WATCHDOG_LABEL)

    snap = _snapshot_dbs()
    info["dbs"] = {k: {"size": v[0], "sha256": v[1][:12] + "..."} for k, v in snap.items()}

    # Verify each DB is queryable
    table_check: Dict[str, List[str]] = {}
    for db in sorted(DATA_DIR.glob("*.db")):
        if db.stat().st_size == 0:
            continue
        try:
            table_check[db.name] = _query_tables(db)
        except Exception as e:
            table_check[db.name] = [f"ERROR: {e}"]
    info["tables"] = table_check

    ok = (
        info["data_dir_exists"]
        and info["run_py_exists"]
        and len(snap) > 0
        and all(not t or not t[0].startswith("ERROR") for t in table_check.values())
    )
    if verbose:
        print("── Preflight ──")
        print(f"  data dir:        {DATA_DIR} ({'OK' if info['data_dir_exists'] else 'MISSING'})")
        print(f"  run.py:          {RUN_PY} ({'OK' if info['run_py_exists'] else 'MISSING'})")
        print(f"  agent PIDs:      {info['agent_pids'] or 'none'}")
        print(f"  watchdog loaded: {info['watchdog_loaded']}")
        print(f"  DBs:")
        for name, meta in info["dbs"].items():
            tbls = table_check.get(name, [])
            print(f"    - {name:<24} {meta['size']:>10} B  sha={meta['sha256']}  tables={len(tbls)}")
    return ok, info


def phase_real_restart(prev_snap: Dict[str, Tuple[int, str]],
                       timeout_s: int = RESTART_TIMEOUT_S) -> Tuple[bool, Dict]:
    """Live SIGTERM + wait for relaunch + verify state intact."""
    report: Dict = {}
    pids = _find_agent_pids()
    report["pids_before"] = pids
    if not pids:
        report["error"] = "No agent process found. Is the agent running?"
        return False, report
    if not _launchd_loaded(WATCHDOG_LABEL):
        report["error"] = (f"Watchdog plist '{WATCHDOG_LABEL}' is not loaded — "
                           "without launchd KeepAlive the agent will not restart.")
        return False, report

    for pid in pids:
        print(f"  Sending SIGTERM to pid {pid}...")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    print(f"  Waiting up to {timeout_s}s for relaunch...")
    new_pids = _wait_for_pid(pids, timeout_s)
    report["pids_after"] = new_pids
    if not new_pids:
        report["error"] = f"Agent did not relaunch within {timeout_s}s."
        return False, report

    # Give the new process a moment to open its DB connections.
    time.sleep(5)

    new_snap = _snapshot_dbs()
    report["dbs_before"] = {k: {"size": v[0], "sha256": v[1]} for k, v in prev_snap.items()}
    report["dbs_after"] = {k: {"size": v[0], "sha256": v[1]} for k, v in new_snap.items()}

    # State must NOT be lost. Two acceptable cases:
    #   - identical sha (no writes during the restart window)
    #   - same-or-larger size (the new process wrote, but nothing was lost)
    failures: List[str] = []
    for name, (size_before, sha_before) in prev_snap.items():
        if name not in new_snap:
            failures.append(f"{name}: missing after restart")
            continue
        size_after, sha_after = new_snap[name]
        if size_after < size_before:
            failures.append(f"{name}: shrunk {size_before}->{size_after} bytes")
        # Tables still queryable
        try:
            tbls = _query_tables(DATA_DIR / name)
            if not tbls:
                failures.append(f"{name}: no tables after restart")
        except Exception as e:
            failures.append(f"{name}: query failed — {e}")

    report["failures"] = failures
    return (len(failures) == 0), report


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recovery smoke test for Dragon Trader. Default is --dry-run.",
        epilog="DANGER: --actually-test sends SIGTERM to the live agent.",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="Verify framework only; do not signal the agent (default).")
    g.add_argument("--actually-test", action="store_true",
                   help="Real test: SIGTERM agent, wait for launchd relaunch, verify state.")
    ap.add_argument("--timeout", type=int, default=RESTART_TIMEOUT_S,
                    help=f"Seconds to wait for relaunch (default {RESTART_TIMEOUT_S}).")
    args = ap.parse_args()

    print(f"=== Dragon Recovery Smoke Test ({datetime.now().isoformat(timespec='seconds')}) ===")

    ok, info = phase_preflight()
    if not ok:
        print("\nPREFLIGHT FAILED. Aborting.")
        return 2

    if args.actually_test:
        print("\n!!! REAL TEST: SIGTERM will be sent to the live agent. !!!")
        print("!!! Demo accounts only — abort with Ctrl-C in the next 5 seconds.")
        try:
            for i in range(5, 0, -1):
                print(f"  ... {i}", flush=True)
                time.sleep(1)
        except KeyboardInterrupt:
            print("Aborted by user.")
            return 130
        prev = _snapshot_dbs()
        ok2, report = phase_real_restart(prev, timeout_s=args.timeout)
        print("\n── Restart report ──")
        print(f"  pids before: {report.get('pids_before')}")
        print(f"  pids after : {report.get('pids_after')}")
        if report.get("error"):
            print(f"  ERROR: {report['error']}")
        for f in report.get("failures", []):
            print(f"  FAIL: {f}")
        print(f"\nResult: {'PASS' if ok2 else 'FAIL'}")
        return 0 if ok2 else 1

    print("\nDry-run complete. Framework looks healthy.")
    print("To exercise the real restart path, re-run with --actually-test")
    print("(WARNING: that will SIGTERM the live agent).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
