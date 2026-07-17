#!/usr/bin/env python3 -B
"""
MT5 Keeper — keeps MetaTrader 5.app alive at the process level.

Why this exists: `com.dragon.mt5` plist used `open -a MetaTrader 5.app` with
KeepAlive=false. If MT5 quit (crash, accidental cmd-Q, OS update), nothing
restarted it — the trader and watchdog could only react to symptoms, not
restore the app itself. Watchdog tier2 force-kills + relaunches but only
after 6 minutes of consecutive failures.

This keeper closes the gap: presence-check every KEEPER_INTERVAL_S seconds,
relaunch if missing. Cheaper and more reliable than launchd KeepAlive on
the `open` command (which exits immediately and would loop hot).

Layered with the rest of the durability stack:
  1. mt5-keeper (this)         — ensures MT5.app is running
  2. ResilientMT5Client        — handles transient rpyc drops (<8s)
  3. com.dragon.watchdog       — detects log/port/rpyc-silent-freeze
  4. com.dragon.trader         — KeepAlive=true on the brain process

Runs as com.dragon.mt5-keeper, KeepAlive=true.
"""
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "mt5_keeper.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

import logging.handlers
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_rot = logging.handlers.RotatingFileHandler(
    str(LOG_PATH), maxBytes=10 * 1024 * 1024, backupCount=5
)
_rot.setFormatter(_fmt)
_stream = logging.StreamHandler()
_stream.setFormatter(_fmt)
log = logging.getLogger("mt5-keeper")
log.setLevel(logging.INFO)
log.addHandler(_rot)
log.addHandler(_stream)
log.propagate = False

# ── Config ──────────────────────────────────────────────────────────────
KEEPER_INTERVAL_S = 15        # presence check cadence (was 30 — chaos test 2026-05-11 showed 90s SLO needs faster polling)
BOOT_GRACE_S = 90             # wait this long after a relaunch before re-probing
MIN_RELAUNCH_INTERVAL_S = 120 # never relaunch faster than this (anti-flap)
RPYC_PROBE_PORTS = (18813, 18814)  # bridge ports we expect to come up after boot
RPYC_PROBE_TIMEOUT_S = 3
HEARTBEAT_EVERY = 20          # log "still healthy" every N intervals (~10 min)

# Trade-path wedge flag (shared contract, 2026-07-17). Failure mode from
# 2026-07-12: the MT5 trade server drops but the data path stays alive —
# process present, TCP ports accept, reads succeed — so BOTH of this keeper's
# probes report "healthy" while every order_send returns None for hours. The
# executor writes this flag when order_send returns None while the market is
# open. A fresh flag here forces the tier-2 remedy (full MT5 terminal
# relaunch, the only intervention that provably heals the wedge), after which
# the flag is deleted.
WEDGE_FLAG = ROOT / "data" / "trade_wedge.flag"
WEDGE_FLAG_FRESH_S = 900      # ignore (and clean up) flags older than 15 min


def mt5_running() -> bool:
    """True if the MetaTrader 5 macOS app process is alive."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", "MetaTrader 5.app/Contents/MacOS/MetaTrader 5"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(r.stdout.strip())
    except Exception as e:
        log.warning("pgrep failed: %s", e)
        # Conservative: assume running if we can't tell, to avoid spurious relaunches.
        return True


def bridge_port_listening(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=RPYC_PROBE_TIMEOUT_S):
            return True
    except Exception:
        return False


def check_trade_wedge(now: float, last_relaunch: float) -> float:
    """Wedge-flag remedy. Returns the (possibly updated) last_relaunch ts.

    If data/trade_wedge.flag exists and is fresh (mtime within
    WEDGE_FLAG_FRESH_S), relaunch the MT5 terminal via the existing tier-2
    mechanism (relaunch_mt5), then delete the flag. The flag is deleted ONLY
    after a successful relaunch — if the anti-flap window blocks us or the
    relaunch fails, the flag survives so the next cycle (or the watchdog)
    retries. A stale flag is deleted without action so a leftover file from a
    pre-reboot event can never trigger a surprise relaunch.
    """
    try:
        if not WEDGE_FLAG.exists():
            return last_relaunch
        age = now - WEDGE_FLAG.stat().st_mtime
    except Exception as e:
        log.warning("wedge flag stat failed: %s", e)
        return last_relaunch

    if age > WEDGE_FLAG_FRESH_S:
        log.warning("Stale trade_wedge.flag (age=%.0fs) — removing without action", age)
        try:
            WEDGE_FLAG.unlink()
        except OSError:
            pass
        return last_relaunch

    since_last = now - last_relaunch
    if since_last < MIN_RELAUNCH_INTERVAL_S:
        log.warning("TRADE WEDGE flag fresh (age=%.0fs) but anti-flap window holds "
                    "(%.0fs since last relaunch < %ds) — will retry",
                    age, since_last, MIN_RELAUNCH_INTERVAL_S)
        return last_relaunch

    log.error("TRADE WEDGE flag detected (age=%.0fs) — process/TCP probes are "
              "blind to this failure; relaunching MT5 terminal", age)
    if relaunch_mt5():
        try:
            WEDGE_FLAG.unlink()
            log.info("trade_wedge.flag cleared after relaunch")
        except OSError as e:
            log.warning("could not clear trade_wedge.flag: %s", e)
        return now
    log.error("wedge-triggered relaunch FAILED — keeping flag for retry")
    return last_relaunch


def relaunch_mt5() -> bool:
    """Force a clean relaunch of MetaTrader 5.app.

    Order:
      1. pkill the macOS app + wine helpers (idempotent if already dead)
      2. brief settle
      3. open -a /Applications/MetaTrader 5.app
    """
    log.warning("RELAUNCH: tearing down stale MT5/wine processes")
    for pat in ("MetaTrader 5.app/Contents/MacOS/MetaTrader 5",
                "wine64-preloader",
                "wineserver"):
        try:
            subprocess.run(["pkill", "-9", "-f", pat], timeout=5,
                           capture_output=True)
        except Exception as e:
            log.warning("pkill %s failed: %s", pat, e)
    time.sleep(3)
    try:
        r = subprocess.run(
            ["open", "-a", "/Applications/MetaTrader 5.app"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            log.error("open MT5 failed rc=%d stderr=%s",
                      r.returncode, (r.stderr or "").strip()[:200])
            return False
        log.warning("RELAUNCH: open -a MetaTrader 5.app issued. Waiting %ds for boot.",
                    BOOT_GRACE_S)
        return True
    except Exception as e:
        log.error("RELAUNCH exception: %s", e)
        return False


BRIDGE_PORT_TO_PLIST = {
    18813: "com.dragon.bridge-tick",
    18814: "com.dragon.bridge-dashboard",
}
BRIDGE_DOWN_GRACE_S = 20       # tolerate this long before kicking the plist
BRIDGE_KICK_COOLDOWN_S = 180   # don't bounce same plist faster than this


def kickstart_plist(label: str) -> bool:
    """Force-restart a launchd job. Used when an in-Wine bridge is dead but
    the outer wine64 wrapper is still alive (so launchctl wouldn't notice
    the failure on its own)."""
    log.warning("KICKING plist %s", label)
    try:
        uid = os.getuid()
        r = subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/{label}"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception as e:
        log.error("kickstart %s failed: %s", label, e)
        return False


def main():
    log.info("MT5 Keeper starting | interval=%ds boot_grace=%ds min_relaunch=%ds",
             KEEPER_INTERVAL_S, BOOT_GRACE_S, MIN_RELAUNCH_INTERVAL_S)
    last_relaunch = 0.0
    healthy_streak = 0
    # Per-port: when did we first observe it down? When did we last kick?
    bridge_down_since: dict[int, float] = {}
    bridge_last_kick: dict[int, float] = {}

    while True:
        now = time.time()
        in_boot_grace = (now - last_relaunch) < BOOT_GRACE_S

        if in_boot_grace:
            time.sleep(KEEPER_INTERVAL_S)
            continue

        # ── TRADE-WEDGE FLAG (2026-07-17) ──
        # A wedged trade server passes both probes below (process alive, TCP
        # accepts) — the executor's flag file is the only signal. Fresh flag
        # → tier-2 MT5 relaunch → delete flag.
        new_last = check_trade_wedge(now, last_relaunch)
        if new_last != last_relaunch:
            last_relaunch = new_last
            time.sleep(KEEPER_INTERVAL_S)
            continue  # boot grace applies from next iteration

        # ── BRIDGE-PORT PROBE ──
        # Wine wraps Python.exe — if the bridge_server_*.py inside Wine dies,
        # the outer wine64 wrapper keeps running, so launchctl never notices
        # the failure (no exit signal). We poll the rpyc ports and force a
        # plist kickstart when a port stays dead past BRIDGE_DOWN_GRACE_S.
        # Validated by chaos test 2026-05-11.
        for port, plist_label in BRIDGE_PORT_TO_PLIST.items():
            if bridge_port_listening(port):
                bridge_down_since.pop(port, None)
                continue
            first_down = bridge_down_since.setdefault(port, now)
            down_for = now - first_down
            since_last_kick = now - bridge_last_kick.get(port, 0)
            if down_for >= BRIDGE_DOWN_GRACE_S and since_last_kick >= BRIDGE_KICK_COOLDOWN_S:
                log.error("BRIDGE port %d (%s) DOWN for %.0fs — kicking plist",
                          port, plist_label, down_for)
                if kickstart_plist(plist_label):
                    bridge_last_kick[port] = now
                    bridge_down_since.pop(port, None)

        if mt5_running():
            healthy_streak += 1
            if healthy_streak % HEARTBEAT_EVERY == 0:
                ports_up = [p for p in RPYC_PROBE_PORTS if bridge_port_listening(p)]
                log.info("HEARTBEAT streak=%d | ports_up=%s",
                         healthy_streak, ports_up)
        else:
            healthy_streak = 0
            since_last = now - last_relaunch
            if since_last < MIN_RELAUNCH_INTERVAL_S:
                log.warning("MT5 missing but anti-flap window holds "
                            "(%.0fs since last relaunch < %ds) — skipping",
                            since_last, MIN_RELAUNCH_INTERVAL_S)
            else:
                log.error("MT5 NOT RUNNING — initiating relaunch")
                if relaunch_mt5():
                    last_relaunch = now

        time.sleep(KEEPER_INTERVAL_S)


if __name__ == "__main__":
    if "--once" in sys.argv:
        running = mt5_running()
        ports = {p: bridge_port_listening(p) for p in RPYC_PROBE_PORTS}
        print(f"mt5_running={running} ports={ports}")
        sys.exit(0 if running else 1)
    try:
        main()
    except KeyboardInterrupt:
        log.info("Keeper interrupted")
