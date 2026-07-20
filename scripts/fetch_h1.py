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
                    GOLD_SMC_SYMBOL, TREND_BASKET)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
H1 = 16385
COUNT = 60000       # SEED depth for the first-ever pull of a symbol (no cache yet).
INCREMENTAL = 1000  # 2026-07-20: steady-state pulls only the recent TAIL, then MERGEs into
                    # the deep cache. The old full 60k × 13-symbol re-pull EVERY run was a
                    # huge Wine-bridge load spike that helped trigger the MT5 trade-path
                    # wedges (order_send drops). Incremental = ~98% less bridge traffic;
                    # the 60k depth is retained via merge (never shrinks — see loop guard).
DEEP_KEEP = 60000   # cap merged history length (matches the old max depth).
# 2026-07-15 root-cause fix: the scheduled H1 refresh only covered GOLD_SMC (XAU),
# so the whole TREND basket's H1 (exit-tuner + intraday) went truncated/stale. Cover
# every H1 consumer — GOLD_SMC symbol + the full TREND basket.
SYMS = sorted({GOLD_SMC_SYMBOL, *TREND_BASKET,
               # 2026-07-17: all active-strategy H1 consumers (SR ADX gate, Momentum,
               # ASAT, FVG) so every book's symbols have fresh H1.
               "GER40.r", "DJ30.r", "SPI200.r", "USDCAD", "USOUSD", "EURUSD",
               "SP500.r", "US2000.r"})


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
                # consumers read XAU as lowercase (raw_h1_xauusd.pkl); others as-is
                _key = "xauusd" if sym == "XAUUSD" else sym.replace(".", "_")
                path = CACHE / ("raw_h1_" + _key + ".pkl")
                # incremental tail in steady state; full SEED only if no deep cache yet
                pull = INCREMENTAL if path.exists() else COUNT
                r = m.copy_rates_from_pos(sym, H1, 0, pull)
                if r is None or len(r) == 0:
                    time.sleep(1.5 * (attempt + 1))     # empty → back off + retry
                    continue
                df = pd.DataFrame(r)
                df["time"] = pd.to_datetime(df["time"], unit="s")   # tz-naive
                out = df
                # MERGE the recent pull into the deep cache, NEVER shrinking it — a merge
                # failure or a short pull must never truncate the 60k history (the old
                # recurring cache-truncation bug). On any error, keep the existing cache.
                try:
                    if path.exists():
                        prev = pickle.load(open(path, "rb"))
                        if prev is not None and len(prev):
                            pv = prev.copy()
                            if not pd.api.types.is_datetime64_any_dtype(pv["time"]):
                                pv["time"] = pd.to_datetime(pv["time"], errors="coerce")
                            if getattr(pv["time"].dt, "tz", None) is not None:
                                pv["time"] = pv["time"].dt.tz_localize(None)
                            merged = (pd.concat([pv, df], ignore_index=True)
                                      .drop_duplicates(subset="time", keep="last")
                                      .sort_values("time").tail(DEEP_KEEP).reset_index(drop=True))
                            out = merged if len(merged) >= len(prev) else prev   # never shrink
                except Exception as me:
                    print(f"  {sym:12s} merge skip ({me}) — keeping deep cache")
                    out = None if path.exists() else df   # never overwrite deep cache on error
                if out is not None:
                    pickle.dump(out, open(path, "wb"))
                    print(f"  {sym:12s} {len(out):>5} H1 bars -> {str(out['time'].iloc[-1])[:16]}")
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
