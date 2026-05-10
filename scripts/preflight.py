"""
Pre-flight startup checks. Called from run.py before brain initialization.

Verifies the machine is in a state where the trader can plausibly run, and
warns loudly if anything is off. Does NOT block startup — the durability
stack handles transient issues — but a printed-loud WARNING shows up at
boot time so you know what's degraded.

Usage:
    from scripts.preflight import run_preflight
    run_preflight(log)
"""
import logging
import shutil
import socket
import sqlite3
import subprocess
from pathlib import Path

log = logging.getLogger("dragon.preflight")

MIN_DISK_FREE_GB = 5.0
BROKER_HOSTS = ("vantagemarkets.com", "vantage.com")
DEFAULT_MT5_PORT = 18813
DNS_TIMEOUT_S = 3
PORT_TIMEOUT_S = 3


def _disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024 ** 3)
    except Exception:
        return -1.0


def _can_resolve(host: str) -> bool:
    try:
        socket.setdefaulttimeout(DNS_TIMEOUT_S)
        socket.gethostbyname(host)
        return True
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(None)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=PORT_TIMEOUT_S):
            return True
    except Exception:
        return False


def _db_writable(path: Path) -> bool:
    try:
        with sqlite3.connect(str(path), timeout=3.0) as c:
            c.execute("CREATE TABLE IF NOT EXISTS preflight_canary (ts REAL)")
            c.execute("DELETE FROM preflight_canary")
        return True
    except Exception as e:
        log.warning("preflight: DB write probe failed on %s: %s", path, e)
        return False


def _mt5_keeper_loaded() -> bool:
    try:
        r = subprocess.run(
            ["launchctl", "list", "com.dragon.mt5-keeper"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def run_preflight(parent_log=None, mt5_port: int = DEFAULT_MT5_PORT,
                  db_path: Path | None = None) -> dict:
    """Run all checks, log per-check results, return a dict of results.

    Never raises. Caller decides what to do with degraded results.
    """
    results: dict = {}
    out = parent_log or log

    out.info("══ PRE-FLIGHT CHECKS ══")

    # 1. Disk free
    root = Path(__file__).resolve().parent.parent
    free_gb = _disk_free_gb(root)
    results["disk_free_gb"] = free_gb
    if free_gb < 0:
        out.warning("PREFLIGHT: disk free check FAILED (could not query)")
    elif free_gb < MIN_DISK_FREE_GB:
        out.warning("PREFLIGHT: LOW DISK — %.1fGB free (< %.1fGB threshold)",
                    free_gb, MIN_DISK_FREE_GB)
    else:
        out.info("PREFLIGHT: disk free %.1fGB ✓", free_gb)

    # 2. Broker DNS
    resolvable = [h for h in BROKER_HOSTS if _can_resolve(h)]
    results["broker_dns_resolved"] = resolvable
    if not resolvable:
        out.warning("PREFLIGHT: NO broker hostname resolvable — DNS or network down")
    else:
        out.info("PREFLIGHT: broker DNS ok (%s) ✓", resolvable[0])

    # 3. MT5 RPC port
    port_up = _port_open("localhost", mt5_port)
    results["mt5_port_listening"] = port_up
    if not port_up:
        out.warning("PREFLIGHT: MT5 port %d NOT listening "
                    "(mt5-keeper should heal this within ~2min)", mt5_port)
    else:
        out.info("PREFLIGHT: MT5 port %d open ✓", mt5_port)

    # 4. Journal DB writable
    if db_path is not None:
        ok = _db_writable(db_path)
        results["journal_db_writable"] = ok
        if not ok:
            out.warning("PREFLIGHT: journal DB write probe FAILED on %s", db_path)
        else:
            out.info("PREFLIGHT: journal DB writable ✓")

    # 5. mt5-keeper loaded (advisory)
    keeper = _mt5_keeper_loaded()
    results["mt5_keeper_loaded"] = keeper
    if not keeper:
        out.warning("PREFLIGHT: com.dragon.mt5-keeper NOT loaded — "
                    "MT5 won't auto-relaunch on crash. Run: "
                    "launchctl bootstrap gui/$(id -u) "
                    "~/Library/LaunchAgents/com.dragon.mt5-keeper.plist")
    else:
        out.info("PREFLIGHT: mt5-keeper plist loaded ✓")

    out.info("══ PRE-FLIGHT DONE ══")
    return results
