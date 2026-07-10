#!/usr/bin/env python3 -B
"""GOLD_H4 backtest — D1-regime / H4-trigger long-only Donchian breakout, XAUUSD.

Same math as agent/gold_h4.py. REAL costs charged (mandatory):
  - spread  $0.30/oz  (entry filled at ask = bar close + spread; bars are bid)
  - slip    $0.10/oz  per side (2 sides)
  - swap    swap_long = -80.55 points/lot/day (MT5 symbol_info, mode=POINTS)
            -> -$0.8055/night at 0.01 lot, x1.4 triple-swap weekend factor
Fill model (conservative, no look-ahead):
  - signal on H4 bar CLOSE -> filled at that close (live bot enters at market
    right after candle close) + spread + slip.
  - stops checked INTRABAR against the stop level known BEFORE the bar
    (previous bar's ratcheted chandelier / initial SL) -> no same-bar trail peek.
  - D1 hard-exit (close < SMA200) acts at the OPEN of the first H4 bar of the
    next day (daily close only known after the day ends).
Sizing: FIXED 0.01 lot ($1 per $1.00 move). SKIP signal when 2xATR stop > $70
(3.0% of $2355). Max 1 position. Params frozen: Donchian20 / 2xATR / 3xATR.

DATA-QUALITY NOTE (2026-07-08): broker "H4" history 2007-2017 is only ~260
bars/year = DAILY-granularity backfill, NOT real H4 (true density ~1540/yr
starts 2018). Pre-2018 bars are DROPPED; the instructed 2010-2020 IS window is
impossible. Split used: IS 2018-2020 (3y true H4) / OOS 2021-2026 (5.5y).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.gold_h4 import DEFAULT_PARAMS, _atr  # noqa: E402

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
ACCOUNT = 2355.0
SPREAD = 0.30        # $/oz, charged once (buy at ask)
SLIP = 0.10          # $/oz per side
SWAP_NIGHT = 0.8055  # $/night at 0.01 lot (80.55 pts/lot/day, point=$0.01/oz)
SWAP_WEEKEND_FACTOR = 1.4   # 5 weekday rollovers charge 7 days (triple-swap day)
P = DEFAULT_PARAMS


def load():
    h4 = pickle.load(open(C / "raw_h4_XAUUSD.pkl", "rb"))
    d1 = pickle.load(open(C / "raw_d1_XAUUSD.pkl", "rb"))
    for df in (h4, d1):
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
    h4 = h4[h4["time"] >= "2018-01-01"].reset_index(drop=True)  # pre-2018 = D1 backfill
    return h4, d1


def run():
    h4, d1 = load()

    # ── D1 regime flags, usable only from the NEXT day (shift 1: during any
    #    H4 bar of day X we only know day X-1's daily close) ────────────────
    dc = d1.set_index(d1["time"].dt.normalize())["close"]
    sma_f = dc.rolling(P["SMA_FAST_D1"]).mean()
    sma_s = dc.rolling(P["SMA_SLOW_D1"]).mean()
    regime = ((dc > sma_s) & (sma_f > sma_s)).shift(1)          # entry gate
    hard_ex = (dc < sma_s).shift(1)                             # exit gate
    day = h4["time"].dt.normalize()
    reg = day.map(regime).fillna(False).values.astype(bool)
    hex_ = day.map(hard_ex).fillna(False).values.astype(bool)

    # ── H4 indicators (prior-N donchian excludes current bar) ──────────────
    close, high, low = h4["close"], h4["high"], h4["low"]
    donch = close.rolling(P["DONCHIAN_N"]).max().shift(1).values
    atr = _atr(high, low, close, P["ATR_PERIOD"]).values
    o, hi, lo, cl, tm = (h4["open"].values, high.values, low.values,
                         close.values, h4["time"].values)
    is_new_day = (day != day.shift(1)).values

    trades = []
    pos = None   # dict(entry, entry_time, sl, hi_close)
    skipped_risk = 0

    warmup = P["DONCHIAN_N"] + P["ATR_PERIOD"] + 2   # D1 regime warms up from 2007 pickle
    for i in range(warmup, len(h4)):
        if pos is not None:
            # 1) D1 hard exit at open of first bar of a new day
            if is_new_day[i] and hex_[i]:
                _book(trades, pos, float(o[i]) - SLIP, tm[i], "D1_SMA200")
                pos = None
            else:
                # 2) intrabar stop vs level known BEFORE this bar
                if lo[i] <= pos["sl"]:
                    _book(trades, pos, pos["sl"] - SLIP, tm[i],
                          "SL" if pos["sl"] <= pos["init_sl"] else "TRAIL")
                    pos = None
                else:
                    # 3) ratchet chandelier at bar close
                    pos["hi_close"] = max(pos["hi_close"], float(cl[i]))
                    pos["sl"] = max(pos["sl"],
                                    pos["hi_close"] - P["TRAIL_ATR_MULT"] * float(atr[i]))
        if pos is None and reg[i] and np.isfinite(donch[i]) and cl[i] > donch[i] \
                and np.isfinite(atr[i]) and atr[i] > 0:
            sl_dist = P["SL_ATR_MULT"] * float(atr[i])
            if sl_dist * P["USD_PER_UNIT_MIN_LOT"] > P["MAX_RISK_USD"]:
                skipped_risk += 1
                continue
            entry = float(cl[i]) + SPREAD + SLIP           # buy at ask + slip
            pos = {"entry": entry, "entry_time": tm[i],
                   "init_sl": float(cl[i]) - sl_dist,       # SL off bid close
                   "sl": float(cl[i]) - sl_dist, "hi_close": float(cl[i])}
    if pos is not None:
        _book(trades, pos, float(cl[-1]) - SLIP, tm[-1], "EOD_OPEN")

    df = pd.DataFrame(trades)
    print(f"H4 bars {len(h4)}  {str(tm[0])[:10]} -> {str(tm[-1])[:10]}  "
          f"| trades {len(df)}  | skipped by $70 risk-gate: {skipped_risk}")
    yr = pd.to_datetime(df["entry_time"]).dt.year
    for label, mask in [("FULL 2018-2026", yr >= 0),
                        ("IS 2018-2020", (yr >= 2018) & (yr <= 2020)),
                        ("OOS 2021-2026", yr >= 2021)]:
        _metrics(df[mask], label)
    print("\nPer-year:")
    for y, g in df.groupby(yr):
        _metrics(g, str(y))
    print("\nExit mix:", df["reason"].value_counts().to_dict())
    print("Hold days: median %.1f  mean %.1f  max %.0f"
          % (df["days"].median(), df["days"].mean(), df["days"].max()))
    print("Total swap charged: $%.2f  | total spread+slip: $%.2f"
          % (df["swap"].sum(), (SPREAD + 2 * SLIP) * len(df)))
    return df


def _book(trades, pos, exit_px, exit_time, reason):
    nights = np.busday_count(pd.Timestamp(pos["entry_time"]).date(),
                             pd.Timestamp(exit_time).date())
    swap = nights * SWAP_NIGHT * SWAP_WEEKEND_FACTOR
    pnl = (exit_px - pos["entry"]) * P["USD_PER_UNIT_MIN_LOT"] - swap
    trades.append({"entry_time": pos["entry_time"], "exit_time": exit_time,
                   "entry": pos["entry"], "exit": exit_px, "pnl": pnl,
                   "swap": swap, "days": float(nights), "reason": reason})


def _metrics(df, label):
    if len(df) < 5:
        print(f"  {label:<18} n={len(df)} (insufficient)")
        return
    pnl = df["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    wr = len(wins) / len(pnl) * 100
    # daily-equity Sharpe on the fixed $2355 base
    d = df.copy()
    d["d"] = pd.to_datetime(d["exit_time"]).dt.normalize()
    daily = d.groupby("d")["pnl"].sum()
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    dr = daily.reindex(idx).fillna(0.0) / ACCOUNT
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    eq = ACCOUNT + pnl.cumsum()
    dd = (eq - np.maximum.accumulate(np.r_[ACCOUNT, eq])[1:]).min()
    print(f"  {label:<18} n={len(pnl):>3}  WR={wr:5.1f}%  PF={pf:5.2f}  "
          f"Sharpe={sharpe:+5.2f}  PnL=${pnl.sum():+8.2f}  "
          f"avgW=${wins.mean() if len(wins) else 0:6.2f}  "
          f"avgL=${losses.mean() if len(losses) else 0:7.2f}  maxDD=${dd:7.2f}")


if __name__ == "__main__":
    run()
