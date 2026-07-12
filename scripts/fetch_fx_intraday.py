#!/usr/bin/env python3 -B
"""Fetch M1 / M5 / M15 FX intraday for the SMC scalping strategy (2026-07-12).
Isolated connection. Saves raw_m1_/raw_m5_/raw_m15_<SYM>.pkl."""
import pickle, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
TFS = [("m5", 5, 80000), ("m15", 15, 80000)]
FX = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP",
      "EURJPY","GBPJPY","AUDJPY","EURCHF","EURAUD","EURCAD","EURNZD","GBPCHF",
      "GBPAUD","CADJPY","CHFJPY","NZDJPY"]


def _connect():
    from mt5linux import MetaTrader5
    m = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    m.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    m.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    return m


def main():
    syms = sys.argv[1:] or FX
    m = _connect()
    for s in syms:
        for tf, code, n in TFS:
            got = False
            for attempt in range(6):               # retry on EMPTY too (bridge contention)
                try:
                    m.symbol_select(s, True)
                    time.sleep(0.3)                  # let the terminal populate + throttle
                    r = m.copy_rates_from_pos(s, code, 0, n)
                    if r is not None and len(r) > 0:
                        df = pd.DataFrame(r); df["time"] = pd.to_datetime(df["time"], unit="s")
                        pickle.dump(df, open(CACHE / f"raw_{tf}_{s}.pkl", "wb"))
                        print(f"  {s:8s} {tf}: {len(df):>6} bars {str(df['time'].iloc[0])[:16]} -> {str(df['time'].iloc[-1])[:16]}", flush=True)
                        got = True; break
                    time.sleep(1.5 * (attempt + 1))  # empty → back off + retry
                except (EOFError, OSError, ConnectionError) as e:
                    print(f"  {s:8s} {tf}: err {attempt+1} {e} — reconnect", flush=True); time.sleep(2)
                    try: m.shutdown()
                    except Exception: pass
                    m = _connect()
            if not got:
                print(f"  {s:8s} {tf}: FAILED after retries", flush=True)
            time.sleep(0.5)                          # throttle between requests
    try: m.shutdown()
    except Exception: pass


if __name__ == "__main__":
    main()
