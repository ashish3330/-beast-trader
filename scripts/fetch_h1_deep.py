#!/usr/bin/env python3 -B
"""One-off: deep-fetch H1 history for the trend basket symbols that only have
500 bars cached (XAUUSD/BTCUSD/JPN225ft/NAS100.r), for the per-symbol intraday
exit tune. Isolated MT5 connection (reliable reads). Saves raw_h1_<SYM>.pkl."""
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER  # noqa: E402

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
H1 = 16385
COUNT = 50000
SYMS = ["XAUUSD", "BTCUSD", "JPN225ft", "NAS100.r"]


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
    for sym in SYMS:
        best = None
        for attempt in range(3):
            try:
                m.symbol_select(sym, True)
                r = m.copy_rates_from_pos(sym, H1, 0, COUNT)
                if r is not None and len(r):
                    best = r
                    break
            except (EOFError, OSError, ConnectionError) as e:
                print(f"  {sym}: bridge err {attempt+1}: {e} — reconnect")
                try:
                    m.shutdown()
                except Exception:
                    pass
                time.sleep(2)
                m = _connect()
        if best is None or len(best) == 0:
            print(f"  {sym}: NO DATA")
            continue
        df = pd.DataFrame(best)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        fn = "raw_h1_" + sym.replace(".", "_") + ".pkl"
        pickle.dump(df, open(CACHE / fn, "wb"))
        print(f"  {sym:10s} {len(df):>6} H1 bars  {str(df['time'].iloc[0])[:10]} -> {str(df['time'].iloc[-1])[:10]}  -> {fn}")
    try:
        m.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
