"""
Extended H1 cache refresh — pulls macro Vantage universe (forex/metals/indices/crypto/oil/commodities).
Writes raw_h1_*.pkl + symbol_meta.json with point/spread/contract from MT5.

- Categorizes via symbol_info.path (the broker's own grouping)
- Reconnects on EOFError / connection drops
- Saves meta after every symbol so progress isn't lost
"""
import pickle, json, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

CACHE_DIR = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
META_PATH = CACHE_DIR / "symbol_meta.json"

MACRO_SYMBOLS = [
    # Forex Majors (7)
    "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF", "AUDUSD", "NZDUSD",
    # Forex Crosses
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "EURGBP", "EURAUD", "EURCHF", "EURCAD", "EURNZD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDNZD",
    "NZDCAD", "NZDCHF", "CADCHF",
    # Metals (4)
    "XAUUSD", "XAGUSD", "XPDUSD.r", "XPTUSD.r",
    # Major Indices
    "NAS100.r", "SP500.r", "GER40.r", "JPN225ft", "UK100.r",
    "FRA40.r", "US2000.r", "DJ30.r", "EU50.r", "SPI200.r",
    "HK50.r", "CHINA50.r", "NETH25.r", "SWI20.r",
    # Liquid Crypto
    "BTCUSD", "ETHUSD", "LTCUSD", "SOLUSD", "XRPUSD", "BCHUSD", "ADAUSD", "DOGUSD",
    # Oil
    "USOUSD", "UKOUSD", "CL-OIL",
    # Commodities
    "COPPER-Cr", "NG-Cr", "GAS-Cr",
]

MT5_TF_H1 = 16385
CANDLE_COUNT = 50000


def _filename(symbol):
    safe = symbol.replace(".", "_").replace("-", "_")
    return f"raw_h1_{safe}.pkl"


def _connect():
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        raise RuntimeError(f"init failed: {mt5.last_error()}")
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"login failed: {mt5.last_error()}")
    return mt5


def _category_from_path(path):
    if not path: return "Other"
    head = path.split("\\")[0]
    m = {
        "Forex": "Forex",
        "Forex Major": "Forex",
        "Gold": "Metal",
        "Silver": "Metal",
        "Oil": "Oil",
        "CFDs.r": "Index",
        "Nikkei": "Index",
        "Crypto Currency": "Crypto",
        "Commodities.r": "Commodity",
    }
    return m.get(head, head)


def refresh():
    mt5 = _connect()
    info = mt5.account_info()
    print(f"Account {info.login} ${info.balance:.2f} on {info.server}\n")

    meta = json.load(open(META_PATH)) if META_PATH.exists() else {}

    ok, fail = 0, []
    for i, sym in enumerate(MACRO_SYMBOLS, 1):
        # Reconnect periodically to keep the rpyc connection healthy
        for attempt in range(3):
            try:
                if not mt5.symbol_select(sym, True):
                    print(f"  {sym:14s} symbol_select FAILED")
                    fail.append(sym); break
                si = mt5.symbol_info(sym)
                if si is None:
                    print(f"  {sym:14s} symbol_info NONE")
                    fail.append(sym); break

                rates = mt5.copy_rates_from_pos(sym, MT5_TF_H1, 0, CANDLE_COUNT)
                if rates is None or len(rates) == 0:
                    print(f"  {sym:14s} NO RATES (path={si.path})")
                    fail.append(sym); break

                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                path = CACHE_DIR / _filename(sym)
                pickle.dump(df, open(path, "wb"))
                days = (df["time"].max() - df["time"].min()).days
                spread_price = float(si.spread) * float(si.point)
                cat = _category_from_path(si.path)

                meta[sym] = {
                    "filename": _filename(sym),
                    "path": si.path,
                    "category": cat,
                    "point": float(si.point),
                    "spread_price": round(spread_price, 8),
                    "spread_points": int(si.spread),
                    "digits": int(si.digits),
                    "contract_size": float(si.trade_contract_size),
                    "min_lot": float(si.volume_min),
                    "lot_step": float(si.volume_step),
                    "stops_level": int(si.trade_stops_level),
                    "bars": int(len(df)),
                    "days": int(days),
                }
                json.dump(meta, open(META_PATH, "w"), indent=2)  # save after each symbol
                print(f"  [{i:>2}/{len(MACRO_SYMBOLS)}] {sym:14s} [{cat:9s}] {len(df):>6} bars / {days:>5}d | pt={si.point} spread={spread_price:.6f}")
                ok += 1
                break
            except (EOFError, OSError, ConnectionError) as e:
                print(f"  {sym:14s} bridge error attempt {attempt+1}: {e} — reconnecting")
                try: mt5.shutdown()
                except: pass
                time.sleep(2)
                mt5 = _connect()
        else:
            fail.append(sym)

    try: mt5.shutdown()
    except: pass
    print(f"\nDone. ok={ok} fail={len(fail)}")
    if fail:
        print(f"FAILED: {fail}")
    print(f"Meta written: {META_PATH}")


if __name__ == "__main__":
    refresh()
