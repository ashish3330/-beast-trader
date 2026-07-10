#!/usr/bin/env python3 -B
"""INDICES MEAN-REVERSION backtest (final, 2026-07-08 — live-verified costs).

Long-only RSI(2)+IBS dip-buying above SMA200 on SP500.r / US2000.r / JPN225ft.
Mirrors agent/indices_mr.py exactly (parity asserted below) so live == test.

Entry (last closed D1 bar): close>SMA200, RSI2<15, IBS<0.30, close<close[1]
  -> BUY next open, broker SL = fill - 6.0xATR(14) (disaster only, never moved).
Exit (first wins, next open): RSI2>65 or close>SMA5; 7-trading-day time-stop;
  6xATR SL intraday.

COSTS (live-verified via mt5.symbol_info on 2026-07-08):
  SP500.r : spread 0.6 idx-pts, swap_mode=2 -> -$1.4972/lot/night (USD)
  US2000.r: spread 0.48, swap_mode=3 -> -$0.5868/lot/night (USD)
  JPN225ft: spread 14.1 idx-pts (bar-median 14.05 — the old 8.0 was WRONG),
            swap_mode=0 = DISABLED -> $0 swap
  Slippage charged BOTH sides. Swap nights = calendar nights (weekend included,
  approximates the Friday triple-swap; swap_rollover3days=5).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.indices_mr import evaluate as imr_evaluate, _rsi, _atr  # noqa: E402

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
# sym: (min_lot, $/index-pt at min lot, swap_$/night at min lot, spread_pts, slip_pts/side)
SPEC = {
    "SP500.r":  (0.10, 0.10,   -0.14972, 0.60, 0.5),
    "US2000.r": (0.10, 0.10,   -0.05868, 0.48, 0.5),
    "JPN225ft": (1.00, 0.0062,  0.0,     14.1, 3.0),
}
EQUITY = 2360.0
OOS_CUT = pd.Timestamp("2023-01-01")


def load(sym):
    p = C / ("raw_d1_" + sym.replace(".", "_") + ".pkl")
    if not p.exists():
        return None
    df = pickle.load(open(p, "rb"))
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def backtest(sym, rsi_thr=15.0, ibs_thr=0.30, df=None):
    df = load(sym) if df is None else df
    if df is None:
        return None
    o, h, l, c, t = df["open"], df["high"], df["low"], df["close"], df["time"]
    sma200 = c.rolling(200).mean()
    sma5 = c.rolling(5).mean()
    rsi2 = _rsi(c, 2)
    ibs = (c - l) / (h - l).replace(0, np.nan)
    atr = _atr(h, l, c, 14)
    minlot, dpp, swap_night, spread, slip = SPEC[sym]
    rt_cost = (spread + 2 * slip) * dpp          # $ round-trip at min lot

    trades = []
    i, n = 200, len(df)
    while i < n - 1:
        if (c[i] > sma200[i] and rsi2[i] < rsi_thr and ibs[i] < ibs_thr
                and c[i] < c[i - 1] and np.isfinite(atr[i])):
            entry = o[i + 1]                     # fill next open
            sl = entry - 6.0 * atr[i]
            j, exit_px, exit_i, held = i + 1, None, None, 0
            while j < n - 1:
                held = j - i                      # closed bars since fill
                if l[j] <= sl:                    # intraday disaster stop
                    exit_px, exit_i = sl, j
                    break
                if rsi2[j] > 65 or c[j] > sma5[j] or held >= 7:
                    exit_px, exit_i = o[j + 1], j + 1   # exit next open
                    break
                j += 1
            if exit_px is None:                   # still open at data end
                exit_px, exit_i = c[n - 1], n - 1
            nights = max((t[exit_i] - t[i + 1]).days, 1)
            pnl = (exit_px - entry) * dpp - rt_cost + swap_night * nights
            trades.append({"t": t[i + 1], "pnl": pnl, "held": held,
                           "sl_hit": l[j] <= sl if j < n - 1 else False})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(trades)


def parity_check(sym, rsi_thr=15.0, ibs_thr=0.30, lookback=700):
    """Assert agent.indices_mr.evaluate fires on exactly the same closed bars
    as the vectorized mask above (windowed replay over the last `lookback`)."""
    df = load(sym)
    c, h, l = df["close"], df["high"], df["low"]
    sma200 = c.rolling(200).mean()
    rsi2 = _rsi(c, 2)
    ibs = (c - l) / (h - l).replace(0, np.nan)
    atr = _atr(h, l, c, 14)
    mask = ((c > sma200) & (rsi2 < rsi_thr) & (ibs < ibs_thr)
            & (c < c.shift(1)) & atr.notna())
    params = {"RSI_ENTRY": rsi_thr, "IBS_ENTRY": ibs_thr}
    n = len(df)
    mism = 0
    for i in range(max(210, n - lookback), n - 1):
        sig = imr_evaluate(df.iloc[: i + 2], params)   # bar i = last closed
        if bool(sig) != bool(mask[i]):
            mism += 1
    status = "OK" if mism == 0 else f"FAIL ({mism} mismatches)"
    print(f"  parity {sym}: detector vs vectorized over last {lookback} bars -> {status}")
    return mism == 0


def stats(tr, label):
    if tr is None or len(tr) < 5:
        print(f"  {label:<28} n<5")
        return None
    pnl = tr["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf")
    wr = (pnl > 0).mean()
    yrs = max((tr["t"].iloc[-1] - tr["t"].iloc[0]).days / 365.0, 0.25)
    ret = pnl / EQUITY
    sharpe = ret.mean() / ret.std() * np.sqrt(len(tr) / yrs) if ret.std() > 0 else 0.0
    print(f"  {label:<28} n={len(tr):<4} WR={wr*100:.0f}%  PF={pf:.2f}  tot=${pnl.sum():+.0f}  "
          f"avg=${pnl.mean():+.2f}  ~{len(tr)/yrs:.0f}/yr  Sharpe~{sharpe:.2f}  "
          f"maxloss=${pnl.min():+.1f}  SLhits={int(tr['sl_hit'].sum())}")
    return {"n": len(tr), "wr": wr, "pf": pf, "sharpe": sharpe, "tot": pnl.sum()}


def main():
    print("=== INDICES MR (RSI2<15, IBS<0.30, long-only) — costs: spread+2xslip+swap(calendar nights) ===")
    allt = []
    for sym in SPEC:
        tr = backtest(sym)
        if tr is None or not len(tr):
            print(f"  {sym}: no trades")
            continue
        allt.append(tr)
        stats(tr, f"{sym} FULL")
        stats(tr[tr["t"] < OOS_CUT], f"{sym} IS (<2023)")
        stats(tr[tr["t"] >= OOS_CUT], f"{sym} OOS (>=2023)")
        print()
    basket = pd.concat(allt).sort_values("t") if allt else None
    if basket is not None:
        stats(basket, "BASKET FULL")
        stats(basket[basket["t"] < OOS_CUT], "BASKET IS (<2023)")
        stats(basket[basket["t"] >= OOS_CUT], "BASKET OOS (>=2023)")

    print("\n=== live-detector parity (agent/indices_mr.py) ===")
    ok = all(parity_check(s) for s in SPEC)
    print(f"  parity: {'ALL OK' if ok else 'BROKEN — DO NOT DEPLOY'}")

    print("\n=== allowed sweep: RSI2 {10,15,20} x IBS {0.2,0.3,0.4} (basket, full sample) ===")
    for r in (10, 15, 20):
        for ib in (0.2, 0.3, 0.4):
            ts = [backtest(s, r, ib) for s in SPEC]
            ts = [t for t in ts if t is not None and len(t)]
            if not ts:
                continue
            b = pd.concat(ts)
            pnl = b["pnl"]
            pf = pnl[pnl > 0].sum() / abs(pnl[pnl <= 0].sum()) if (pnl <= 0).any() else float("inf")
            oos = b[b["t"] >= OOS_CUT]["pnl"]
            opf = oos[oos > 0].sum() / abs(oos[oos <= 0].sum()) if (oos <= 0).any() else float("inf")
            print(f"  RSI2<{r:<2} IBS<{ib}: n={len(b):<3} WR={(pnl>0).mean()*100:.0f}% "
                  f"PF={pf:.2f} OOS_PF={opf:.2f} tot=${pnl.sum():+.0f}")


if __name__ == "__main__":
    main()
