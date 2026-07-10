#!/usr/bin/env python3 -B
"""FEED WATCHDOG — detects a stale candle feed and auto-restarts the tick bridge.

The Wine/MT5 tick bridge occasionally freezes (EOFError flap), so the strategy
detector stops seeing new bars → zero trades, silently. This checks every run:
  broker's latest M15 bar  vs  the bar the trader last evaluated (from its log).
If the trader is more than STALE_MIN minutes behind, restart com.dragon.bridge-tick.

Both times are MT5 server-time strings, so the comparison is timezone-clean.
Run every 5 min via launchd (com.dragon.feed-watchdog.plist).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER  # noqa: E402

LOG = Path("/Users/ashish/Documents/beast-trader/logs/dragon_stderr.log")
SYNC_JSON = Path("/Users/ashish/Documents/beast-trader/data/live_positions.json")
SYNC_STALE_SEC = 120    # positions sync file older than this → restart sync daemon
STALE_MIN = 35          # trader >35 min behind broker → feed considered frozen
CHECK_SYM = "XAUUSD"


def _kick(job):
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
    subprocess.run(["launchctl", "kickstart", "-k", f"gui/{uid}/{job}"], timeout=30)


def check_sync_daemon():
    """The trend/IMR books read open positions from live_positions.json (written
    by com.dragon.sync-positions every ~20s). If it freezes, those books fail
    CLOSED (skip cycles) — but they must resume, so restart the daemon if stale."""
    try:
        age = time.time() - float(json.loads(SYNC_JSON.read_text()).get("ts", 0.0))
    except Exception as e:
        _log(f"[syncwd] cannot read {SYNC_JSON.name} ({e}) — restarting sync daemon")
        _kick("com.dragon.sync-positions")
        return
    if age > SYNC_STALE_SEC:
        _log(f"[syncwd] STALE sync file: age={age:.0f}s > {SYNC_STALE_SEC} → restarting sync daemon")
        _kick("com.dragon.sync-positions")
    else:
        _log(f"[syncwd] OK: sync file age={age:.0f}s")


def _log(m):
    print(m, flush=True)


def broker_latest_bar():
    from mt5linux import MetaTrader5
    m = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    try:
        m.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")
        m.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        r = m.copy_rates_from_pos(CHECK_SYM, 15, 0, 1)
        return pd.to_datetime(r[0]["time"], unit="s")
    finally:
        try:
            m.shutdown()
        except Exception:
            pass


def trader_last_eval_bar():
    # tail the log, find the last '[SMABO XAUUSD] eval bar <ts>' line
    try:
        out = subprocess.run(["grep", "-E", f"SMABO {CHECK_SYM}.*eval bar", str(LOG)],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    m = None
    for line in out.strip().splitlines()[-1:]:
        mm = re.search(r"eval bar (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if mm:
            m = mm.group(1)
    return pd.to_datetime(m) if m else None


def restart_bridge():
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
    subprocess.run(["launchctl", "kickstart", "-k",
                    f"gui/{uid}/com.dragon.bridge-tick"], timeout=30)


def main():
    check_sync_daemon()   # positions sync file freshness (trend/IMR depend on it)
    try:
        broker = broker_latest_bar()
    except Exception as e:
        _log(f"[feedwd] broker fetch failed (MT5 down?): {e}")
        return
    trader = trader_last_eval_bar()
    if trader is None:
        _log("[feedwd] no trader heartbeat found — skipping (bot may be starting)")
        return
    lag_min = (broker - trader).total_seconds() / 60.0
    # 2026-07-10 AUDIT FIX: the heartbeat source ('[SMABO XAUUSD] eval bar') is
    # DEAD (SMABO OFF since 2026-07-08), so `trader` is frozen at 2026-07-07 and
    # this fired 837 false STALE-FEED bridge kills (SIGTERM every 5min → EOFError
    # storms, can land mid-order_send). mt5-keeper ALREADY restarts a DOWN bridge
    # with a proper 180s cooldown, so this branch is now LOG-ONLY (no restart)
    # until a real heartbeat (brain-written file) replaces the dead SMABO grep.
    if lag_min > STALE_MIN:
        _log(f"[feedwd] STALE-FEED lag={lag_min:.0f}min (broker={broker} trader={trader}) "
             f"— restart DISABLED (stale heartbeat source; mt5-keeper owns bridge health)")
    else:
        _log(f"[feedwd] OK: broker={broker} trader={trader} lag={lag_min:.0f}min")


if __name__ == "__main__":
    main()
