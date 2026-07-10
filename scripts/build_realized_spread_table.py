#!/usr/bin/env python3 -B
"""
Dragon Trader — Realized-spread table builder.

2026-06-18 Tier 1 #10 of DRAGON_BOT_UPGRADE_v2.md.

Samples mt5.symbol_info(sym).spread every 60s for a window (default 7 days)
and writes per-(symbol, hour_utc) p50/p95 percentiles to
`data/spread_realized_per_session.json`. The BT cost model loads this file
when `BT_VARIABLE_SPREAD_ENABLED=True` and falls back to static SPREAD when
the file is missing — so this script is OPT-IN data collection that NEVER
affects live behaviour.

USAGE
-----
    # Run for 7 days in the background:
    nohup python3 -B scripts/build_realized_spread_table.py --days 7 \
        --symbols XAUUSD,EURUSD,UK100.r,BTCUSD,DJ30.r,SP500.r,US2000.r,GER40.r \
        > logs/spread_realized.log 2>&1 &

The output JSON is loaded by `backtest/cost_model.py:_load_realized_spread_table`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Allow `from data... import ...` etc.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

log = logging.getLogger("spread_builder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--symbols", type=str, required=False,
                    default="XAUUSD,EURUSD,UK100.r,BTCUSD")
    ap.add_argument("--interval-sec", type=float, default=60.0)
    ap.add_argument("--out", type=str,
                    default="data/spread_realized_per_session.json")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    log.info("Sampling %d symbols for %.2f days, interval=%.1fs",
             len(symbols), args.days, args.interval_sec)

    # MT5 import is optional — script remains importable on dev boxes without MT5.
    try:
        import MetaTrader5 as mt5  # type: ignore
        if not mt5.initialize():
            log.error("MT5 initialize() failed")
            sys.exit(2)
    except Exception as e:
        log.error("MT5 not available (%s). Aborting.", e)
        sys.exit(3)

    # samples[symbol][hour_utc] = [spread_in_price, ...]
    samples: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    end_ts = time.time() + args.days * 86400.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_flush = time.time()
    while time.time() < end_ts:
        hour_utc = datetime.now(timezone.utc).hour
        for sym in symbols:
            try:
                info = mt5.symbol_info(sym)
                if info is None:
                    continue
                point = float(info.point) if info.point else 0.00001
                spread_price = float(info.spread) * point
                if spread_price > 0:
                    samples[sym][hour_utc].append(spread_price)
            except Exception as e:
                log.debug("sample %s failed: %s", sym, e)

        # Flush every 5 minutes
        if time.time() - last_flush > 300:
            _flush(samples, out_path)
            last_flush = time.time()
        time.sleep(args.interval_sec)

    _flush(samples, out_path)
    log.info("Done. Wrote %s", out_path)


def _flush(samples, out_path):
    out = {}
    for sym, by_hour in samples.items():
        out[sym] = {}
        for h, vals in by_hour.items():
            out[sym][str(int(h))] = {
                "p50": _percentile(vals, 50.0),
                "p95": _percentile(vals, 95.0),
                "n": len(vals),
            }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, out_path)
    log.info("Flushed %d symbols to %s", len(out), out_path)


if __name__ == "__main__":
    main()
