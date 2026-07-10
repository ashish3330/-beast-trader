#!/usr/bin/env python3 -B
"""Fetch D1 + H1 for the FX research universe (2026-07-10)."""
import pickle, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
D1, H1 = 16408, 16385
FX = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD",
      "EURGBP","EURJPY","GBPJPY","AUDJPY","EURCHF","EURAUD","EURCAD","EURNZD",
      "GBPCHF","GBPAUD","CADJPY","CHFJPY","NZDJPY"]
from mt5linux import MetaTrader5
m = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
m.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")
m.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
for s in FX:
    for tf, code, n in [("d1", D1, 3000), ("h1", H1, 50000)]:
        try:
            m.symbol_select(s, True)
            r = m.copy_rates_from_pos(s, code, 0, n)
            if r is None or len(r) == 0:
                print(f"{s} {tf}: none"); continue
            df = pd.DataFrame(r); df["time"] = pd.to_datetime(df["time"], unit="s")
            pickle.dump(df, open(CACHE / f"raw_{tf}_{s}.pkl", "wb"))
            print(f"{s} {tf}: {len(df)} bars {str(df['time'].iloc[0])[:10]}->{str(df['time'].iloc[-1])[:10]}")
        except Exception as e:
            print(f"{s} {tf}: err {e}")
m.shutdown()
