#!/usr/bin/env python3 -B
"""
MT5 connection watchdog — auto-recovery for the Wine MT5 / bridge / dragon stack.

Why this exists: 2026-05-01 outage. After device-off recovery the Wine MT5
terminal lost broker auth at ~03:35. Bridge processes stayed up holding
stale handles. Dragon trader restart-looped on (-6 Terminal: Authorization
failed) for 45+ minutes without ever healing. KeepAlive=true on the trader
plist guards process death — not connection-level failures.

This watchdog observes (passively, no MT5 IPC contention) and escalates:

  Tier 1 (after N=3 consecutive failures, ~3 min) — bounce the bridge
    launchd jobs + the trader, so they re-init against MT5 from scratch.

  Tier 2 (after N=6 consecutive failures, ~6 min) — force-kill MT5 +
    relaunch /Applications/MetaTrader 5.app. ONLY heals if the user has
    "Save account information" enabled in the MT5 GUI Login dialog;
    without that, the kill/relaunch yields a fresh terminal that still
    needs a human GUI login.

  Tier 3 (after N=12 consecutive failures, ~12 min) — log CRITICAL,
    stop attempting auto-heal, keep monitoring so we log RECOVERED when
    the human intervenes.

Cooldowns on each tier prevent re-bouncing a service we just bounced.

Probes are passive: read tail of logs/dragon.log + check bridge port +
verify dragon process is running. No mt5.initialize() call here — that
would compete with the live trader's IPC session.

Runs as com.dragon.watchdog launchd job, KeepAlive=true.
"""
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import MT5_PORT  # noqa: E402

LOG_PATH = ROOT / "logs" / "watchdog.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
DRAGON_LOG = ROOT / "logs" / "dragon.log"

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("watchdog")
_stream_h = logging.StreamHandler()
_stream_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_stream_h)

PROBE_INTERVAL_S = 60
TIER1_THRESHOLD = 3
TIER2_THRESHOLD = 6
TIER3_THRESHOLD = 12
TIER1_COOLDOWN_S = 300
TIER2_COOLDOWN_S = 600
UID = os.getuid()

FAIL_PATTERNS = ("MT5 initialize failed", "Failed to connect to MT5",
                 "MT5 connect error", "MT5 login failed")
OK_PATTERNS = ("MT5 connected:", "Tick streamer started")


def _bridge_port_open() -> bool:
    try:
        with socket.create_connection(("localhost", MT5_PORT), timeout=3):
            return True
    except Exception:
        return False


def _trader_running() -> bool:
    try:
        r = subprocess.run(
            ["pgrep", "-f", "beast-trader/run.py"],
            capture_output=True, text=True, timeout=5
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def _tail_dragon_log(max_bytes: int = 16384) -> str:
    if not DRAGON_LOG.exists():
        return ""
    try:
        sz = DRAGON_LOG.stat().st_size
        with open(DRAGON_LOG, "rb") as f:
            f.seek(max(0, sz - max_bytes))
            return f.read().decode("utf-8", errors="ignore")
    except Exception as e:
        log.warning("tail_dragon_log failed: %s", e)
        return ""


def probe_health() -> tuple[bool, str]:
    """Composite health check. Returns (healthy, reason)."""
    if not _bridge_port_open():
        return False, "bridge_port_unreachable"
    if not _trader_running():
        return False, "trader_process_not_running"

    tail = _tail_dragon_log()
    if not tail:
        return False, "no_dragon_log"

    # Walk the tail backward to find the most recent OK or FAIL signal.
    last_ok_line = None
    last_fail_line = None
    for line in tail.splitlines():
        if any(p in line for p in OK_PATTERNS):
            last_ok_line = line
        elif any(p in line for p in FAIL_PATTERNS):
            last_fail_line = line

    # Timestamp prefix is HH:MM:SS — lexically comparable within a day.
    def _ts(line: str | None) -> str:
        return (line or "")[:8]

    if last_fail_line and last_ok_line:
        if _ts(last_fail_line) > _ts(last_ok_line):
            return False, f"failing_after_ok | {last_fail_line[:120]}"
        return True, f"connected | {last_ok_line[:120]}"
    if last_fail_line:
        return False, f"failing | {last_fail_line[:120]}"
    if last_ok_line:
        return True, f"connected | {last_ok_line[:120]}"
    return False, "no_recent_signal_in_log"


def _run(cmd: list[str], timeout: int = 30) -> bool:
    log.info("EXEC %s", " ".join(cmd))
    try:
        r = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("EXEC %s rc=%d stderr=%s",
                        cmd[0], r.returncode, (r.stderr or "").strip()[:200])
        return r.returncode == 0
    except Exception as e:
        log.warning("EXEC %s exception: %s", cmd[0], e)
        return False


def heal_tier1():
    """Bounce bridge launchd jobs + trader. Cheap, fast, low blast radius."""
    log.warning("TIER1 RECOVERY: bouncing bridge launchd jobs + trader")
    _run(["launchctl", "kickstart", "-k", f"gui/{UID}/com.dragon.bridge-tick"])
    _run(["launchctl", "kickstart", "-k", f"gui/{UID}/com.dragon.bridge-dashboard"])
    _run(["launchctl", "kickstart", "-k", f"gui/{UID}/com.dragon.trader"])


def heal_tier2():
    """Force-kill MT5 + relaunch. Heals only if MT5 has saved auto-login."""
    log.error("TIER2 RECOVERY: force-killing MT5.app and relaunching "
              "(only auto-heals if MT5 'Save account information' is enabled)")
    _run(["pkill", "-9", "-f", "MetaTrader 5.app"])
    _run(["pkill", "-9", "-f", "wine64-preloader"])
    _run(["pkill", "-9", "-f", "wineserver"])
    time.sleep(3)
    _run(["open", "-a", "/Applications/MetaTrader 5.app"])
    log.warning("TIER2: MT5 relaunched.")


def main():
    log.info("Watchdog starting | probe=%ds tier1=%d tier2=%d tier3=%d",
             PROBE_INTERVAL_S, TIER1_THRESHOLD, TIER2_THRESHOLD, TIER3_THRESHOLD)
    fail_count = 0
    last_tier1_at = 0.0
    last_tier2_at = 0.0
    last_state: bool | None = None
    tier3_logged = False
    healthy_streak = 0
    HEARTBEAT_EVERY = 10  # log "still healthy" every 10 probes (~10 min)

    while True:
        ok, reason = probe_health()
        now = time.time()

        if ok:
            if last_state is False:
                log.info("RECOVERED after %d fails | %s", fail_count, reason)
                healthy_streak = 0
            healthy_streak += 1
            if healthy_streak % HEARTBEAT_EVERY == 0:
                log.info("HEARTBEAT healthy_streak=%d | %s", healthy_streak, reason)
            fail_count = 0
            tier3_logged = False
            last_state = True
        else:
            fail_count += 1
            log.warning("UNHEALTHY fail#%d | %s", fail_count, reason)
            last_state = False

            if fail_count >= TIER3_THRESHOLD:
                if not tier3_logged:
                    log.critical("TIER3: %d consecutive fails. STOPPING self-heal. "
                                 "Likely needs human MT5 GUI login.", fail_count)
                    tier3_logged = True
            elif fail_count >= TIER2_THRESHOLD and (now - last_tier2_at) > TIER2_COOLDOWN_S:
                heal_tier2()
                last_tier2_at = now
            elif fail_count >= TIER1_THRESHOLD and (now - last_tier1_at) > TIER1_COOLDOWN_S:
                heal_tier1()
                last_tier1_at = now

        time.sleep(PROBE_INTERVAL_S)


if __name__ == "__main__":
    if "--once" in sys.argv:
        ok, reason = probe_health()
        print(f"{'HEALTHY' if ok else 'UNHEALTHY'}: {reason}")
        sys.exit(0 if ok else 1)
    try:
        main()
    except KeyboardInterrupt:
        log.info("Watchdog interrupted")
