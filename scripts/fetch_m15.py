#!/usr/bin/env python3 -B
"""Fetch M15 history for the live universe (+ ICT reference assets) so the
ICT liquidity-sweep + FVG backtest can run on real 15M data, not just USDCAD.

Writes raw_m15_<safe>.pkl into the shared cache. Reconnects on bridge drops.
TF M15 = 15. 50000 bars ≈ 17 months.
"""
import pickle, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, SYMBOLS

CACHE_DIR = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
MT5_TF_M15 = 15
CANDLE_COUNT = 50000

# Explicit always-fetch list — these symbols MUST be cached even if absent from
# config.SYMBOLS (e.g. FVG-only symbols, ICT reference assets). Re-added
# 2026-06-14 so the FVG/SR backtests never silently skip required symbols.
_FETCH_ALWAYS = ["EURUSD", "NAS100.r", "XAUUSD",
                 "SPI200.r", "JPN225ft", "ETHUSD", "USOUSD"]

# Live universe + ICT reference-card assets + explicit always-fetch list.
TARGETS = sorted(set(list(SYMBOLS.keys()) + _FETCH_ALWAYS))


def _filename(symbol):
    safe = symbol.replace(".", "_").replace("-", "_")
    return f"raw_m15_{safe}.pkl"


def _connect():
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        raise RuntimeError(f"init failed: {mt5.last_error()}")
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"login failed: {mt5.last_error()}")
    return mt5


def main():
    mt5 = _connect()
    try:
        info = mt5.account_info()
        print(f"Account {info.login} ${info.balance:.2f} on {info.server}\n")
    except Exception:
        pass

    ok, fail = 0, []
    for i, sym in enumerate(TARGETS, 1):
        for attempt in range(3):
            try:
                if not mt5.symbol_select(sym, True):
                    print(f"  {sym:14s} symbol_select FAILED"); fail.append(sym); break
                rates = mt5.copy_rates_from_pos(sym, MT5_TF_M15, 0, CANDLE_COUNT)
                if rates is None or len(rates) == 0:
                    print(f"  {sym:14s} NO RATES"); fail.append(sym); break
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                pickle.dump(df, open(CACHE_DIR / _filename(sym), "wb"))
                days = (df["time"].max() - df["time"].min()).days
                print(f"  [{i:>2}/{len(TARGETS)}] {sym:14s} {len(df):>6} M15 bars / {days:>4}d")
                ok += 1; break
            except (EOFError, OSError, ConnectionError) as e:
                print(f"  {sym:14s} bridge error {attempt+1}: {e} — reconnecting")
                try: mt5.shutdown()
                except Exception: pass
                time.sleep(2)
                mt5 = _connect()
        else:
            fail.append(sym)

    try: mt5.shutdown()
    except Exception: pass
    print(f"\nDONE: {ok} ok, {len(fail)} failed: {fail}")


if __name__ == "__main__":
    main()
