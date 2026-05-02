"""
Recover the 24 'NO RATES' symbols from refresh_extended.py.

Strategies tried (in order) per symbol:
  a) symbol_select(True) + sleep 1s + copy_rates_from_pos
  b) copy_rates_range(2024-01-01 .. 2026-05-01)
  c) copy_rates_from(now, 5000)
  d) market_book_add then retry copy_rates_from_pos
  e) symbol_info_tick — if a tick is live, mark TICK_ONLY

Status: OK_<bars> / TICK_ONLY / DEAD / ERROR_<msg>
Saves successful rates to raw_h1_<safe>.pkl AND atomically merges into
cache/symbol_meta.json. Writes backtest/results/recovery_report.json.
"""
import pickle, json, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

CACHE_DIR = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
META_PATH = CACHE_DIR / "symbol_meta.json"
REPORT_PATH = Path("/Users/ashish/Documents/beast-trader/backtest/results/recovery_report.json")

NO_RATES_SYMBOLS = [
    "GBPCAD", "GBPNZD", "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "CADCHF",
    "FRA40.r", "DJ30.r", "EU50.r", "HK50.r", "CHINA50.r", "NETH25.r", "SWI20.r",
    "LTCUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGUSD",
    "USOUSD", "CL-OIL", "NG-Cr", "GAS-Cr",
]

MT5_TF_H1 = 16385
CANDLE_COUNT = 5000
UTC = timezone.utc


def _filename(symbol):
    safe = symbol.replace(".", "_").replace("-", "_")
    return f"raw_h1_{safe}.pkl"


def _connect(max_attempts=12):
    """Connect to MT5 bridge — retry with backoff (bridge can take 30-60s to recover)."""
    from mt5linux import MetaTrader5
    last_err = None
    for attempt in range(max_attempts):
        try:
            mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
            if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
                raise RuntimeError(f"init failed: {mt5.last_error()}")
            if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                raise RuntimeError(f"login failed: {mt5.last_error()}")
            return mt5
        except (ConnectionRefusedError, EOFError, OSError, RuntimeError) as e:
            last_err = e
            wait = min(5 + attempt * 5, 30)
            print(f"  [connect] attempt {attempt+1}/{max_attempts} failed: {e} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"could not reconnect to MT5 bridge after {max_attempts} attempts: {last_err}")


def _category_from_path(path):
    if not path: return "Other"
    head = path.split("\\")[0]
    m = {
        "Forex": "Forex", "Forex Major": "Forex",
        "Gold": "Metal", "Silver": "Metal", "Oil": "Oil",
        "CFDs.r": "Index", "Nikkei": "Index",
        "Crypto Currency": "Crypto", "Commodities.r": "Commodity",
    }
    return m.get(head, head)


def _snapshot_symbol_info(mt5_obj, sym):
    """Pull all needed attrs in one rpyc round-trip into local primitives."""
    si = mt5_obj.symbol_info(sym)
    if si is None:
        return None
    return {
        "path": str(si.path) if si.path else "",
        "point": float(si.point),
        "spread_points": int(si.spread),
        "digits": int(si.digits),
        "contract_size": float(si.trade_contract_size),
        "min_lot": float(si.volume_min),
        "lot_step": float(si.volume_step),
        "stops_level": int(si.trade_stops_level),
    }


def _save_rates(sym, rates, si_snap):
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    out = CACHE_DIR / _filename(sym)
    pickle.dump(df, open(out, "wb"))
    days = (df["time"].max() - df["time"].min()).days
    if si_snap is None:
        si_snap = {"path": "", "point": 0.0, "spread_points": 0, "digits": 0,
                   "contract_size": 0.0, "min_lot": 0.0, "lot_step": 0.0, "stops_level": 0}
    spread_price = si_snap["spread_points"] * si_snap["point"]
    entry = {
        "filename": _filename(sym),
        "path": si_snap["path"],
        "category": _category_from_path(si_snap["path"]),
        "point": si_snap["point"],
        "spread_price": round(spread_price, 8),
        "spread_points": si_snap["spread_points"],
        "digits": si_snap["digits"],
        "contract_size": si_snap["contract_size"],
        "min_lot": si_snap["min_lot"],
        "lot_step": si_snap["lot_step"],
        "stops_level": si_snap["stops_level"],
        "bars": int(len(df)),
        "days": int(days),
        "recovered_via": None,  # filled by caller
    }
    return entry, len(df)


def _try_strategy(mt5_obj, sym, strat):
    """Run one strategy. Return (rates_or_None, error_msg_or_None)."""
    try:
        if strat == "a":
            mt5_obj.symbol_select(sym, True)
            time.sleep(1)
            r = mt5_obj.copy_rates_from_pos(sym, MT5_TF_H1, 0, CANDLE_COUNT)
        elif strat == "b":
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2026, 5, 1, tzinfo=UTC)
            r = mt5_obj.copy_rates_range(sym, MT5_TF_H1, start, end)
        elif strat == "c":
            r = mt5_obj.copy_rates_from(sym, MT5_TF_H1, datetime.now(tz=UTC), CANDLE_COUNT)
        elif strat == "d":
            try:
                mt5_obj.market_book_add(sym)
            except Exception:
                pass
            time.sleep(0.5)
            r = mt5_obj.copy_rates_from_pos(sym, MT5_TF_H1, 0, CANDLE_COUNT)
            try:
                mt5_obj.market_book_release(sym)
            except Exception:
                pass
        else:
            return None, f"unknown_strat_{strat}"
        if r is None or len(r) == 0:
            return None, "empty"
        return r, None
    except (EOFError, OSError, ConnectionError) as e:
        raise
    except Exception as e:
        return None, str(e)[:80]


def _check_tick(mt5_obj, sym):
    try:
        t = mt5_obj.symbol_info_tick(sym)
        if t is None:
            return False
        bid = getattr(t, "bid", 0) or 0
        ask = getattr(t, "ask", 0) or 0
        return bool(bid > 0 or ask > 0)
    except Exception:
        return False


def recover():
    mt5 = _connect()
    info = mt5.account_info()
    print(f"Account {info.login} ${info.balance:.2f} on {info.server}\n")
    print(f"Recovering {len(NO_RATES_SYMBOLS)} 'NO RATES' symbols...\n")

    # Load existing meta — preserve all entries
    meta = json.load(open(META_PATH)) if META_PATH.exists() else {}
    print(f"Existing meta: {len(meta)} entries\n")

    results = {}
    recovered = []
    tick_only = []
    dead = []

    def reconnect():
        nonlocal mt5
        try: mt5.shutdown()
        except: pass
        time.sleep(2)
        mt5 = _connect()

    for i, sym in enumerate(NO_RATES_SYMBOLS, 1):
        # Resume: skip if already in meta with bars > 0
        if sym in meta and meta[sym].get("bars", 0) > 0 and (CACHE_DIR / _filename(sym)).exists():
            nbars = meta[sym]["bars"]
            via = meta[sym].get("recovered_via", "?")
            results[sym] = f"OK_{nbars}"
            recovered.append((sym, nbars, via))
            print(f"  [{i:>2}/{len(NO_RATES_SYMBOLS)}] {sym:14s} ALREADY RECOVERED ({nbars} bars, via {via}) — skipping")
            continue

        status = None
        rates = None
        winning_strat = None
        last_err = None

        # Try strategies a-d, with bridge-error reconnect
        for strat in ("a", "b", "c", "d"):
            tries = 0
            while tries < 3:
                tries += 1
                try:
                    r, err = _try_strategy(mt5, sym, strat)
                    if r is not None:
                        rates = r
                        winning_strat = strat
                    else:
                        last_err = err
                    break
                except (EOFError, OSError, ConnectionError) as e:
                    print(f"  {sym:14s} bridge error on strat {strat}: {e} — reconnecting")
                    reconnect()
                    continue
            if rates is not None:
                break

        if rates is not None:
            # Snapshot + save with reconnect-on-EOF retry
            saved = False
            for save_attempt in range(3):
                try:
                    si_snap = _snapshot_symbol_info(mt5, sym)
                    entry, nbars = _save_rates(sym, rates, si_snap)
                    entry["recovered_via"] = winning_strat
                    meta[sym] = entry
                    # atomic merge: write to tmp, then rename
                    tmp = META_PATH.with_suffix(".json.tmp")
                    json.dump(meta, open(tmp, "w"), indent=2)
                    tmp.replace(META_PATH)
                    status = f"OK_{nbars}"
                    recovered.append((sym, nbars, winning_strat))
                    print(f"  [{i:>2}/{len(NO_RATES_SYMBOLS)}] {sym:14s} RECOVERED via strat({winning_strat}) — {nbars} bars")
                    saved = True
                    break
                except (EOFError, OSError, ConnectionError) as e:
                    print(f"  {sym:14s} bridge error during save (attempt {save_attempt+1}): {e} — reconnecting")
                    reconnect()
                    continue
                except Exception as e:
                    traceback.print_exc()
                    status = f"ERROR_save_{str(e)[:40]}"
                    dead.append(sym)
                    print(f"  [{i:>2}/{len(NO_RATES_SYMBOLS)}] {sym:14s} ERROR saving: {e}")
                    break
            if not saved and status is None:
                # save_attempts exhausted via EOFError loop — write pickle without symbol_info
                try:
                    entry, nbars = _save_rates(sym, rates, None)
                    entry["recovered_via"] = winning_strat
                    meta[sym] = entry
                    tmp = META_PATH.with_suffix(".json.tmp")
                    json.dump(meta, open(tmp, "w"), indent=2)
                    tmp.replace(META_PATH)
                    status = f"OK_{nbars}_noinfo"
                    recovered.append((sym, nbars, winning_strat))
                    print(f"  [{i:>2}/{len(NO_RATES_SYMBOLS)}] {sym:14s} RECOVERED (no symbol_info) — {nbars} bars")
                except Exception as e:
                    status = f"ERROR_final_{str(e)[:40]}"
                    dead.append(sym)
        else:
            # Strategy e: tick check
            try:
                has_tick = _check_tick(mt5, sym)
            except (EOFError, OSError, ConnectionError) as e:
                reconnect()
                try:
                    has_tick = _check_tick(mt5, sym)
                except Exception:
                    has_tick = False
            if has_tick:
                status = "TICK_ONLY"
                tick_only.append(sym)
                print(f"  [{i:>2}/{len(NO_RATES_SYMBOLS)}] {sym:14s} TICK_ONLY (live but no H1 history)")
            else:
                status = f"DEAD ({last_err or 'no_data'})"
                dead.append(sym)
                print(f"  [{i:>2}/{len(NO_RATES_SYMBOLS)}] {sym:14s} DEAD — last_err: {last_err}")

        results[sym] = status

    try: mt5.shutdown()
    except: pass

    report = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "account": int(info.login),
        "server": info.server,
        "total_attempted": len(NO_RATES_SYMBOLS),
        "recovered_count": len(recovered),
        "tick_only_count": len(tick_only),
        "dead_count": len(dead),
        "recovered": [{"symbol": s, "bars": n, "via_strategy": v} for s, n, v in recovered],
        "tick_only": tick_only,
        "dead": dead,
        "per_symbol": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(REPORT_PATH, "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"RECOVERY SUMMARY")
    print(f"{'='*60}")
    print(f"  Recovered: {len(recovered)}/{len(NO_RATES_SYMBOLS)}")
    print(f"  Tick-only: {len(tick_only)}")
    print(f"  Dead:      {len(dead)}")
    if recovered:
        print(f"\n  Recovered symbols:")
        for s, n, v in recovered:
            print(f"    {s:14s} {n:>5} bars (strat {v})")
    if tick_only:
        print(f"\n  Tick-only (live, no H1 history): {tick_only}")
    if dead:
        print(f"\n  Dead: {dead}")
    print(f"\nReport: {REPORT_PATH}")
    print(f"Meta:   {META_PATH}  ({len(meta)} entries)")


if __name__ == "__main__":
    recover()
