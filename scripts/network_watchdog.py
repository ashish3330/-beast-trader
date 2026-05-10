#!/usr/bin/env python3 -B
"""
Network watchdog — detects upstream connectivity loss separate from MT5 issues.

Why this exists: when the Mac's WiFi flaps or the ISP drops, the rpyc bridge
inside Wine MT5 may still be locally listening (tier1 healing won't help) but
the broker connection is dead. The trader will keep cycling, executing logic
on stale ticks, and the main watchdog can't tell the difference between
"local stack broken" and "internet down".

This script probes:
  1. DNS resolution of broker hostname (vantagemarkets.com)
  2. Generic internet reach (1.1.1.1 TCP:53)
  3. Local MT5 RPC port (sanity)

On persistent loss:
  - WARN: log NETWORK_DOWN, but DO NOT bounce trader (no point — bridge will
    just fail to reach broker on restart anyway).
  - On RECOVERED: kickstart trader to force a fresh broker session.

Runs as com.dragon.network-watchdog launchd job, KeepAlive=true.
"""
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "network_watchdog.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

import logging.handlers
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_rot = logging.handlers.RotatingFileHandler(
    str(LOG_PATH), maxBytes=10 * 1024 * 1024, backupCount=5
)
_rot.setFormatter(_fmt)
_stream = logging.StreamHandler()
_stream.setFormatter(_fmt)
log = logging.getLogger("net-watchdog")
log.setLevel(logging.INFO)
log.addHandler(_rot)
log.addHandler(_stream)
log.propagate = False

# ── Config ──────────────────────────────────────────────────────────────
PROBE_INTERVAL_S = 30
DOWN_THRESHOLD = 3        # consecutive fails before declaring NETWORK_DOWN
UP_THRESHOLD = 2          # consecutive successes before declaring NETWORK_UP
PROBE_TIMEOUT_S = 5
HEARTBEAT_EVERY = 20      # log "still up" every 20 probes (~10 min)
UID = os.getuid()

# Broker hostname inferred from MT5_SERVER. VantageInternational-Demo →
# vantagemarkets.com. Resolve generally, fall back if hostname not set.
BROKER_HOSTS = ("vantagemarkets.com", "vantage.com")
INTERNET_SANITY_HOST = "1.1.1.1"
INTERNET_SANITY_PORT = 53


def can_resolve(host: str) -> bool:
    try:
        socket.setdefaulttimeout(PROBE_TIMEOUT_S)
        socket.gethostbyname(host)
        return True
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(None)


def can_reach(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_S):
            return True
    except Exception:
        return False


def probe() -> tuple[bool, str]:
    """Returns (network_up, reason)."""
    # Layer 1: internet alive at all
    if not can_reach(INTERNET_SANITY_HOST, INTERNET_SANITY_PORT):
        return False, f"internet_unreachable ({INTERNET_SANITY_HOST}:{INTERNET_SANITY_PORT})"
    # Layer 2: DNS works (catches captive portals + DNS server flaps)
    if not any(can_resolve(h) for h in BROKER_HOSTS):
        return False, f"dns_failed ({','.join(BROKER_HOSTS)})"
    return True, "ok"


def kickstart_trader():
    log.warning("Forcing trader restart for fresh broker session")
    try:
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{UID}/com.dragon.trader"],
            timeout=10, capture_output=True,
        )
    except Exception as e:
        log.warning("kickstart failed: %s", e)


def main():
    log.info("Network watchdog starting | probe=%ds down_thr=%d up_thr=%d",
             PROBE_INTERVAL_S, DOWN_THRESHOLD, UP_THRESHOLD)
    fail_streak = 0
    succ_streak = 0
    state = "UP"        # UP | DOWN
    healthy_streak = 0

    while True:
        ok, reason = probe()
        if ok:
            fail_streak = 0
            succ_streak += 1
            healthy_streak += 1

            if state == "DOWN" and succ_streak >= UP_THRESHOLD:
                log.warning("NETWORK RECOVERED after %d successful probes — kickstarting trader",
                            succ_streak)
                state = "UP"
                kickstart_trader()
            elif state == "UP" and healthy_streak % HEARTBEAT_EVERY == 0:
                log.info("HEARTBEAT streak=%d | %s", healthy_streak, reason)
        else:
            succ_streak = 0
            fail_streak += 1
            log.warning("NETWORK probe FAIL #%d | %s", fail_streak, reason)
            if state == "UP" and fail_streak >= DOWN_THRESHOLD:
                log.error("NETWORK DOWN declared after %d consecutive fails — trader healing suspended",
                          fail_streak)
                state = "DOWN"

        time.sleep(PROBE_INTERVAL_S)


if __name__ == "__main__":
    if "--once" in sys.argv:
        ok, reason = probe()
        print(f"{'UP' if ok else 'DOWN'}: {reason}")
        sys.exit(0 if ok else 1)
    try:
        main()
    except KeyboardInterrupt:
        log.info("Network watchdog interrupted")
