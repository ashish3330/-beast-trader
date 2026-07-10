#!/usr/bin/env python3 -B
"""POSITION SYNC — isolated single-reader of open positions -> disk JSON.

The brain's IN-PROCESS MT5 reads (multi-symbol copy_rates / positions_get) fail
under contention with the tick-bridge/dashboard/executor clients — but an
ISOLATED short-lived process reads reliably (proven all session). So this one
job reads all open positions every run and writes them to data/live_positions.json;
the trend + indices-MR books read that file instead of hitting MT5 in-process.

Runs every ~20s via launchd (com.dragon.sync-positions). Cheap: one positions_get.
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "live_positions.json"
HARD_TIMEOUT_SEC = 15   # a hung rpyc/Wine positions_get must not freeze the file


def _hard_exit(signum, frame):
    # Bridge read hung — die so launchd (StartInterval=20) respawns a clean one.
    print("[sync] HARD TIMEOUT — killing hung run so the file can refresh", flush=True)
    os._exit(1)


def main():
    signal.signal(signal.SIGALRM, _hard_exit)
    signal.alarm(HARD_TIMEOUT_SEC)
    from mt5linux import MetaTrader5
    m = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    m.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    m.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    try:
        pos = m.positions_get() or []
        # current price per unique symbol (isolated process → reads are reliable);
        # the trend trail needs ticket+tp+live price to modify WITHOUT any
        # in-process read (which fails under bridge contention).
        px = {}
        for s in {str(p.symbol) for p in pos}:
            try:
                t = m.symbol_info_tick(s)
                px[s] = {"bid": float(t.bid), "ask": float(t.ask)}
            except Exception:
                px[s] = {"bid": 0.0, "ask": 0.0}
        rows = [{"symbol": str(p.symbol), "magic": int(p.magic),
                 "type": int(p.type), "open_time": int(p.time),
                 "volume": float(p.volume), "sl": float(p.sl),
                 "tp": float(p.tp), "ticket": int(p.ticket),
                 "price_open": float(p.price_open),
                 "price_cur": px.get(str(p.symbol), {}).get(
                     "bid" if int(p.type) == 0 else "ask", 0.0)} for p in pos]
        payload = {"ts": time.time(), "n": len(rows), "positions": rows}
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(OUT)                       # atomic
        print(f"[sync] wrote {len(rows)} positions")
    finally:
        try:
            m.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
