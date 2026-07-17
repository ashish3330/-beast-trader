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
    # FAIL-CLOSED (2026-07-10 review): initialize()/login() return False (not raise)
    # on error, and positions_get() returns None on an MT5 error state. Writing a
    # fresh {n:0} file in any of these cases falsely claims FLAT — the age-based
    # freshness guard passes, trend/IMR see no positions, and fire DUPLICATE live
    # opens. So on ANY failure we DON'T write: the file goes stale and consumers
    # fail closed as designed.
    if not m.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        print("[sync] initialize failed — NOT writing (fail-closed)"); return
    if not m.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print("[sync] login failed — NOT writing (fail-closed)"); return
    try:
        pos = m.positions_get()
        if pos is None:
            print("[sync] positions_get returned None — NOT writing (fail-closed)"); return
        # LIVE QUOTES for ALL trend/IMR symbols (not just open ones) — the ORDER
        # path reads these from disk so a flat symbol (e.g. BTC) can be ENTERED
        # WITHOUT any in-process symbol_info_tick, which fails under bridge
        # contention for symbols hammered by the always-on loops (2026-07-12 fix).
        try:
            # 2026-07-17: feed EVERY active strategy's symbols (not just TREND/IMR)
            # so SR/FVG/Momentum/ASAT/SMABO/Scalper can actually trade their full
            # tuned whitelists live (the order path reads these quotes from disk).
            from config import (TREND_BASKET, IMR_WHITELIST, SR_WHITELIST,
                                FVG_WHITELIST, MOMENTUM_SYMBOL_WHITELIST,
                                ASAT_SYMBOL_WHITELIST, SMABO_WHITELIST, SCALPER_WHITELIST)
            quote_syms = (set(TREND_BASKET) | set(IMR_WHITELIST) | set(SR_WHITELIST)
                          | set(FVG_WHITELIST) | set(MOMENTUM_SYMBOL_WHITELIST)
                          | set(ASAT_SYMBOL_WHITELIST) | set(SMABO_WHITELIST)
                          | set(SCALPER_WHITELIST))
        except Exception:
            quote_syms = set()
        quote_syms |= {str(p.symbol) for p in pos}
        px = {}
        for s in quote_syms:
            try:
                m.symbol_select(s, True)
                t = m.symbol_info_tick(s)
                if t and float(t.bid) > 0:
                    px[s] = {"bid": float(t.bid), "ask": float(t.ask)}
            except Exception:
                pass
        rows = [{"symbol": str(p.symbol), "magic": int(p.magic),
                 "type": int(p.type), "open_time": int(p.time),
                 "volume": float(p.volume), "sl": float(p.sl),
                 "tp": float(p.tp), "ticket": int(p.ticket),
                 "price_open": float(p.price_open),
                 "profit": float(getattr(p, "profit", 0.0) or 0.0),  # live $ P/L (read-free)
                 "price_cur": px.get(str(p.symbol), {}).get(
                     "bid" if int(p.type) == 0 else "ask", 0.0)} for p in pos]
        payload = {"ts": time.time(), "n": len(rows), "positions": rows, "quotes": px}
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
