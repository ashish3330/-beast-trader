#!/usr/bin/env python3 -B
"""Refresh the H1 cache for the GOLD_SMC strategy (2026-07-13).

GOLD_SMC (agent/gold_smc.py) reads raw_h1_<sym>.pkl and fail-closes if it's >3h
stale. Nothing was refreshing it — the learning engine only clobbered it to 500
bars intermittently, so after the weekend the XAU H1 cache sat 21h stale on the
Friday close and GOLD_SMC never fired. This isolated job (own MT5 connection,
never competes with the live trader) keeps H1 fresh. Mirrors scripts/fetch_d1.py.
Scheduled ~every 15 min (H1 bars form hourly; tight cadence keeps the 3h gate
happy and picks up each new bar promptly).
"""
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,  # noqa: E402
                    GOLD_SMC_SYMBOL)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
H1 = 16385
COUNT = 1500
SYMS = sorted({GOLD_SMC_SYMBOL})   # H1 consumers (GOLD_SMC); extend if more added


def _connect():
    from mt5linux import MetaTrader5
    m = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    if not m.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        raise RuntimeError(f"init failed: {m.last_error()}")
    if not m.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"login failed: {m.last_error()}")
    return m


def main():
    m = _connect()
    ok, fail = 0, []
    for sym in SYMS:
        for attempt in range(3):
            try:
                m.symbol_select(sym, True)
                time.sleep(0.3)
                r = m.copy_rates_from_pos(sym, H1, 0, COUNT)
                if r is None or len(r) == 0:
                    time.sleep(1.5 * (attempt + 1))     # empty → back off + retry
                    continue
                df = pd.DataFrame(r)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                fn = "raw_h1_" + sym.replace(".", "_") + ".pkl"
                pickle.dump(df, open(CACHE / fn, "wb"))
                print(f"  {sym:12s} {len(df):>5} H1 bars -> {str(df['time'].iloc[-1])[:16]}")
                ok += 1
                break
            except (EOFError, OSError, ConnectionError) as e:
                print(f"  {sym:12s} bridge err {attempt+1}: {e} — reconnecting")
                try:
                    m.shutdown()
                except Exception:
                    pass
                time.sleep(2)
                m = _connect()
        else:
            fail.append(sym)
    try:
        m.shutdown()
    except Exception:
        pass
    print(f"H1 cache refresh: {ok} ok, {len(fail)} failed: {fail}")


if __name__ == "__main__":
    main()
