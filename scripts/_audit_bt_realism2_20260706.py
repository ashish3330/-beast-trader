#!/usr/bin/env python3 -B
"""Audit part 2:
A. Official 365d XAU run (verify 421-trade claim, current config).
B. Same but with NO-look-ahead H1 slicing (exclude current-hour H1 bar).
C. Overlay live daily-kill rule (3 consec peak_r<0.5 losses/day -> dead rest
   of day) + 900s post-close cooldown on the 365d trade stream.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import sma_breakout_backtest as bt  # noqa: E402

SYM = "XAUUSD"
DAYS = 365


def run(no_lookahead):
    m15 = bt._load(SYM, "m15")
    keep = int(DAYS * 24 * 4)
    if keep < len(m15):
        m15 = m15.iloc[-keep:].reset_index(drop=True)
    n = len(m15)
    h1 = bt._resample_h1(m15)
    spread = bt.SPREAD[SYM]
    state = bt._FakeState(m15, h1)

    if no_lookahead:
        orig = bt._FakeState.get_candles

        def patched(self, symbol, tf):
            if tf != 60:
                return orig(self, symbol, tf)
            c = self.cursor
            if c >= len(self._m15_ns):
                end = len(self.h1)
            else:
                end = int(np.searchsorted(self._h1_ns, int(self._m15_ns[c]),
                                          side="right"))
                # cursor bar CLOSES at cursor_time+15m; if that close lands on
                # the hour boundary the current H1 bar is complete -> keep it,
                # else drop it (it contains future intra-hour m15 data).
                cur_t = pd.Timestamp(self._m15_ns[c])
                if (cur_t + pd.Timedelta(minutes=15)).minute != 0:
                    end -= 1
            if end <= 0:
                return None
            lo = max(0, end - bt.H1_WINDOW)
            return self.h1.iloc[lo: end]
        state.get_candles = patched.__get__(state, bt._FakeState)

    strat = bt.SMABreakoutStrategy(state)
    arr = (m15["high"].values, m15["low"].values, m15["close"].values)
    trades = []
    open_until = -1
    for i in range(max(800, 60), n - 1):
        if i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(SYM, None)
        sig = strat.evaluate(SYM)
        if sig is None:
            continue
        tr = bt._simulate_trade(sig, arr, entry_i=i, spread=spread)
        if tr is None:
            continue
        tr["time"] = m15["time"].iloc[i]
        trades.append(tr)
        open_until = i + tr["bars_held"]
    return trades, m15


def summ(tag, trades):
    Rs = np.array([t["R"] for t in trades]) if trades else np.array([0.0])
    wins = Rs[Rs > 0]
    losses = Rs[Rs <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else 999
    print(f"{tag}: trades={len(trades)} totalR={Rs.sum():+.1f} PF={pf:.2f} "
          f"WR={(Rs > 0).mean()*100:.1f}%")
    return Rs


tr_a, m15 = run(False)
summ("A. official 365d (current cfg)     ", tr_a)
tr_b, _ = run(True)
summ("B. no-look-ahead H1 slice          ", tr_b)

# C. daily-kill + cooldown overlay on stream A
kept = []
consec = 0
dead_day = None
last_close_t = None
for t in tr_a:
    day = t["time"].date()
    if dead_day == day:
        continue
    if last_close_t is not None and (t["time"] - last_close_t).total_seconds() < 900:
        continue
    kept.append(t)
    close_i = t["entry_i"] + t["bars_held"]
    last_close_t = m15["time"].iloc[min(close_i, len(m15) - 1)]
    lost = t["peak_r"] < 0.5          # live proxy: peak_r>=0.5 == win
    close_day = last_close_t.date()
    if close_day != day and dead_day != close_day:
        pass
    if lost:
        consec += 1
        if consec >= 3:
            dead_day = close_day
            consec = 0
    else:
        consec = 0
summ("C. + live daily-kill(3)+cooldown   ", kept)
