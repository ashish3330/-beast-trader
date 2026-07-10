#!/usr/bin/env python3 -B
"""Refresh the DAILY (D1) cache for the trend-follower basket.

Runs as an ISOLATED process (its own MT5 connection) so it never competes with
the live trader's clients. The trend engine reads these pkls instead of fetching
D1 live — decoupling it from the flaky Wine/MT5 bridge. Scheduled every few hours
(D1 bars only change once/day). Mirrors scripts/fetch_m15.py.
"""
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,  # noqa: E402
                    TREND_BASKET, IMR_WHITELIST)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
D1 = 16408
COUNT = 3000
SYMS = sorted(set(TREND_BASKET) | set(IMR_WHITELIST))   # trend + indices-MR


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
                if not m.symbol_select(sym, True):
                    pass
                r = m.copy_rates_from_pos(sym, D1, 0, COUNT)
                if r is None or len(r) == 0:
                    fail.append(sym)
                    break
                df = pd.DataFrame(r)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                fn = "raw_d1_" + sym.replace(".", "_") + ".pkl"
                pickle.dump(df, open(CACHE / fn, "wb"))
                print(f"  {sym:12s} {len(df):>5} D1 bars -> {str(df['time'].iloc[-1])[:10]}")
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
    print(f"D1 cache refresh: {ok} ok, {len(fail)} failed: {fail}")


if __name__ == "__main__":
    main()
