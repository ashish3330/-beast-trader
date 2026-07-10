#!/usr/bin/env python3 -B
"""Audit: reconcile SMABO backtest signal frequency vs live (Jun21-Jul6).

1. Reproduce 365d XAU run (verify the 421-trades claim on current config).
2. Run the SAME loop over the live window (bar_time >= 2026-06-21) and list
   every signal bar_time, with and without open_until suppression, with and
   without the ADX gate (which only went live 2026-07-05).
3. Locate the 8 live XAU signal bars in the cache by matching entry==close.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import sma_breakout_backtest as bt  # noqa: E402

SYM = "XAUUSD"
LIVE_ENTRIES = [4073.21, 4126.24, 4134.97, 4137.19, 4147.15, 4176.12,
                4187.53, 4183.50]

m15 = bt._load(SYM, "m15")
print(f"cache bars={len(m15)}  first={m15['time'].iloc[0]}  last={m15['time'].iloc[-1]}")

# ── 3. locate live signal bars by entry==close match ─────────────────────
print("\n── live XAU signals located in cache (entry == bar close):")
for e in LIVE_ENTRIES:
    hits = m15[np.isclose(m15["close"], e, atol=0.005)]
    recent = hits[hits["time"] >= pd.Timestamp("2026-06-20", tz="UTC")]
    ts = [str(t) for t in recent["time"].tail(3)]
    print(f"  entry={e:<9} -> {ts}")

# ── 2. live-window run ────────────────────────────────────────────────────
h1 = bt._resample_h1(m15)
spread = bt.SPREAD[SYM]
arr = (m15["high"].values, m15["low"].values, m15["close"].values)
t0 = pd.Timestamp("2026-06-21 11:00", tz="UTC")  # log rotated Jun21 16:41 IST
start_i = int(np.searchsorted(m15["time"].values, t0.to_datetime64()))
print(f"\nlive-window start_i={start_i} ({m15['time'].iloc[start_i]}) n={len(m15)}")


def run(adx_off, suppress):
    if adx_off:
        os.environ["SMABO_XAUUSD_ADX_MIN"] = "0"
    else:
        os.environ.pop("SMABO_XAUUSD_ADX_MIN", None)
    state = bt._FakeState(m15, h1)
    strat = bt.SMABreakoutStrategy(state)
    sigs, trades = [], []
    open_until = -1
    for i in range(start_i, len(m15) - 1):
        if suppress and i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(SYM, None)
        sig = strat.evaluate(SYM)
        if sig is None:
            continue
        sigs.append((m15["time"].iloc[i], sig["direction"], sig["entry"]))
        if suppress:
            tr = bt._simulate_trade(sig, arr, entry_i=i, spread=spread)
            if tr is None:
                continue
            trades.append(tr)
            open_until = i + tr["bars_held"]
    return sigs, trades


for label, adx_off, suppress in [
        ("ADX18 + open_until (current cfg, BT-style trades)", False, True),
        ("ADX18, RAW signals every bar (no suppression)", False, False),
        ("ADX OFF + open_until (pre-Jul5 era, BT-style)", True, True),
        ("ADX OFF, RAW signals (pre-Jul5 live comparable)", True, False)]:
    sigs, trades = run(adx_off, suppress)
    print(f"\n== {label}: signals={len(sigs)} trades={len(trades)}")
    for t, d, e in sigs[:60]:
        print(f"   {t}  {d:<5} entry={e:.2f}")

os.environ.pop("SMABO_XAUUSD_ADX_MIN", None)
