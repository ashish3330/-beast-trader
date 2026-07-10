"""
Targeted H1 cache re-fetch for symbols whose cache is shorter than the
deep-tune lookback. Writes to the EXACT filename the backtest reads
(symbol_meta.json -> filename) and refreshes that symbol's meta bars/days.

Only touches the symbols passed on the argv (default: the 8 known-short
live symbols). Reconnects on bridge drops. Backs up each pkl before overwrite.
"""
import pickle, json, sys, time, shutil
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

CACHE_DIR = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
META_PATH = CACHE_DIR / "symbol_meta.json"
MT5_TF_H1 = 16385
CANDLE_COUNT = 50000

DEFAULT_SHORT = ["DJ30.r", "JPN225ft", "SWI20.r", "XAUUSD",
                 "BTCUSD", "NAS100.r", "USDJPY", "USOUSD"]


def _connect():
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        raise RuntimeError(f"init failed: {mt5.last_error()}")
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"login failed: {mt5.last_error()}")
    return mt5


def _filename(symbol, meta):
    if symbol in meta and "filename" in meta[symbol]:
        return meta[symbol]["filename"]
    return f"raw_h1_{symbol.replace('.', '_').replace('-', '_')}.pkl"


def main():
    syms = sys.argv[1:] or DEFAULT_SHORT
    meta = json.load(open(META_PATH)) if META_PATH.exists() else {}
    mt5 = _connect()
    info = mt5.account_info()
    print(f"Account {info.login} ${info.balance:.2f} on {info.server}")
    print(f"Re-fetching {len(syms)} symbols, {CANDLE_COUNT} H1 bars each\n")

    ok, fail = [], []
    for i, sym in enumerate(syms, 1):
        fn = _filename(sym, meta)
        for attempt in range(4):
            try:
                if not mt5.symbol_select(sym, True):
                    print(f"  {sym:12s} symbol_select FAILED"); fail.append(sym); break
                rates = mt5.copy_rates_from_pos(sym, MT5_TF_H1, 0, CANDLE_COUNT)
                if rates is None or len(rates) == 0:
                    print(f"  {sym:12s} NO RATES"); fail.append(sym); break
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                dst = CACHE_DIR / fn
                if dst.exists():
                    shutil.copy(dst, str(dst) + ".bak.pretune")
                pickle.dump(df, open(dst, "wb"))
                days = (df["time"].max() - df["time"].min()).days
                meta.setdefault(sym, {})
                meta[sym]["filename"] = fn
                meta[sym]["bars"] = int(len(df))
                meta[sym]["days"] = int(days)
                json.dump(meta, open(META_PATH, "w"), indent=2)
                flag = "" if days >= 1095 else "  <-- STILL <3yr (broker cap)"
                print(f"  [{i}/{len(syms)}] {sym:12s} -> {fn:22s} {len(df):>6} bars / "
                      f"{df['time'].min().date()}->{df['time'].max().date()} ({days}d ~{days/365:.2f}y){flag}")
                ok.append((sym, days))
                break
            except (EOFError, OSError, ConnectionError) as e:
                print(f"  {sym:12s} bridge error attempt {attempt+1}: {e} — reconnecting")
                try: mt5.shutdown()
                except Exception: pass
                time.sleep(3)
                mt5 = _connect()
        else:
            fail.append(sym)
        time.sleep(1)  # breathe between symbols to spare the live bridge

    try: mt5.shutdown()
    except Exception: pass
    print(f"\nDone. ok={len(ok)} fail={len(fail)}")
    still_short = [s for s, d in ok if d < 1095]
    if still_short:
        print(f"STILL <3yr (broker-capped): {still_short}")
    if fail:
        print(f"FAILED: {fail}")


if __name__ == "__main__":
    main()
