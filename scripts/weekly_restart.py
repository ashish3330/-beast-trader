#!/usr/bin/env python3 -B
"""
Weekly trader+MT5 restart — clears Wine memory bloat and stale rpyc handles.

Wine processes accumulate memory over weeks. Bridge processes inside MT5 hold
rpyc state that can drift. A clean restart between trading weeks keeps both
fresh. Schedule: Sunday 21:00 UTC (1h before forex market reopen Sunday 22:00).

Sequence:
  1. bootout trader     — stop the brain cleanly
  2. pkill MT5 + Wine   — full teardown (mt5-keeper will relaunch)
  3. wait 90s           — let mt5-keeper relaunch + bridge re-attach
  4. bootstrap trader   — brain boots fresh against clean MT5

Logs every step. If anything fails, logs ERROR but does NOT block — the
durability stack will heal on its own once a human notices.
"""
import logging
import logging.handlers
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "weekly_restart.log"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("weekly-restart")
_h = logging.handlers.RotatingFileHandler(
    str(LOG_PATH), maxBytes=2 * 1024 * 1024, backupCount=4
)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_h)

UID = os.getuid()
RELAUNCH_GRACE_S = 90


def run(cmd: list[str], timeout: int = 30) -> bool:
    log.info("EXEC %s", " ".join(cmd))
    try:
        r = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("rc=%d stderr=%s", r.returncode,
                        (r.stderr or "").strip()[:200])
        return r.returncode == 0
    except Exception as e:
        log.error("exec failed: %s", e)
        return False


def main():
    log.info("════════ WEEKLY RESTART ════════")
    # 1. Stop trader cleanly so launchd doesn't fight us.
    log.info("Step 1/4: bootout trader")
    run(["launchctl", "bootout", f"gui/{UID}/com.dragon.trader"], timeout=20)
    time.sleep(2)

    # 2. Tear down MT5 + Wine. mt5-keeper will pick this up.
    log.info("Step 2/4: tearing down MT5 + Wine")
    for pat in ("MetaTrader 5.app/Contents/MacOS/MetaTrader 5",
                "wine64-preloader",
                "wineserver"):
        run(["pkill", "-9", "-f", pat])
    time.sleep(3)

    # 3. Wait for keeper to relaunch MT5 and bridge servers to listen.
    log.info("Step 3/4: waiting %ds for mt5-keeper to relaunch MT5", RELAUNCH_GRACE_S)
    time.sleep(RELAUNCH_GRACE_S)

    # 4. Bring trader back up.
    log.info("Step 4/4: bootstrap trader")
    plist = Path.home() / "Library" / "LaunchAgents" / "com.dragon.trader.plist"
    if not plist.exists():
        log.error("trader plist missing at %s — cannot bootstrap", plist)
        return 1
    run(["launchctl", "bootstrap", f"gui/{UID}", str(plist)], timeout=30)

    log.info("════════ WEEKLY RESTART DONE ════════")
    return 0


if __name__ == "__main__":
    sys.exit(main())
