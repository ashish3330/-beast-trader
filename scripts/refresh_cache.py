"""
Refresh H1 cache from MT5 bridge — pulls 50K candles per symbol.
Saves to /Users/ashish/Documents/xauusd-trading-bot/cache/raw_h1_*.pkl
"""
import pickle
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

CACHE_DIR = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = {
    "XAUUSD":   "raw_h1_xauusd.pkl",
    "XAGUSD":   "raw_h1_XAGUSD.pkl",
    "BTCUSD":   "raw_h1_BTCUSD.pkl",
    "NAS100.r": "raw_h1_NAS100_r.pkl",
    "JPN225ft": "raw_h1_JPN225ft.pkl",
    "USDJPY":   "raw_h1_USDJPY.pkl",
    "USDCHF":   "raw_h1_USDCHF.pkl",
    "USDCAD":   "raw_h1_USDCAD.pkl",
    "EURJPY":   "raw_h1_EURJPY.pkl",
    "EURAUD":   "raw_h1_EURAUD.pkl",
    "EURUSD":   "raw_h1_EURUSD.pkl",
    "GBPUSD":   "raw_h1_GBPUSD.pkl",
    "GBPJPY":   "raw_h1_GBPJPY.pkl",
    "AUDJPY":   "raw_h1_AUDJPY.pkl",
    "AUDUSD":   "raw_h1_AUDUSD.pkl",
    "NZDUSD":   "raw_h1_NZDUSD.pkl",
    "EURGBP":   "raw_h1_EURGBP.pkl",
    "EURCHF":   "raw_h1_EURCHF.pkl",
    "SP500.r":  "raw_h1_SP500_r.pkl",
    "GER40.r":  "raw_h1_GER40_r.pkl",
    "UK100.r":  "raw_h1_UK100_r.pkl",
    "ETHUSD":   "raw_h1_ETHUSD.pkl",
}

MT5_TF_H1 = 16385
CANDLE_COUNT = 50000


def refresh_all():
    from mt5linux import MetaTrader5

    print(f"Connecting to MT5 bridge on {MT5_HOST}:{MT5_PORT}...")
    mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 login failed: {mt5.last_error()}")
        return
    info = mt5.account_info()
    print(f"Connected: {info.name} | Balance: ${info.balance:.2f}\n")

    total_new = 0
    for symbol, filename in SYMBOLS.items():
        path = CACHE_DIR / filename
        print(f"  {symbol:12s} → {filename}...", end=" ", flush=True)
        try:
            rates = mt5.copy_rates_from_pos(symbol, MT5_TF_H1, 0, CANDLE_COUNT)
            if rates is None or len(rates) == 0:
                print("NO DATA")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

            # Check old cache
            if path.exists():
                old_df = pickle.load(open(path, "rb"))
                old_count = len(old_df)
            else:
                old_count = 0

            pickle.dump(df, open(path, "wb"))
            days = (df["time"].max() - df["time"].min()).days
            print(f"{len(df)} candles ({days}d) [was {old_count}]")
            total_new += len(df)
        except Exception as e:
            print(f"ERROR: {e}")

    mt5.shutdown()
    print(f"\nDone. {total_new} total candles cached.")


if __name__ == "__main__":
    refresh_all()
