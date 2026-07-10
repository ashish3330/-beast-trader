#!/usr/bin/env python3 -B
"""PER-SYMBOL hard-tune of the TREND exit model (2026-07-09).

Backtests the LIVE exit stack on D1 bars and sweeps its params per symbol with
walk-forward (IS = first 70%, OOS = last 30%). The exit stack being tuned:
  - initial 3xATR catastrophic stop
  - Chandelier trail: HH(lookback) - TRAIL_ATR*ATR  (long; mirror for short)
  - profit-lock SL: entry + LOCK_FRAC * peak_profit  (only-tightens, arms >= ACT*ATR)
  - peak-giveback reversal EXIT: close when profit retraces GIVEBACK_FRAC from peak
  - realistic ATR-distance TP (TP_ATR)  [or off]
  - re-entry block: after a giveback/stop exit, no same-dir re-entry until the
    daily signal changes (matches live _trend_rev_block)

Intraday path is approximated from D1 OHLC (the only history with depth):
  peak profit within a bar := bar.high-entry (long)  — the intraday peak, exactly
  what the live per-60s tracker sees. Exits checked against bar low/high with a
  CONSERVATIVE adverse-first ordering (stop/giveback before TP) so the sim never
  flatters itself. Costs: per-round-trip spread from the data (spread-only, no
  slippage, per user's live model).

Signal = the SAME 3-EMA ensemble as agent/trend_follower.evaluate (no look-ahead:
signal from prior close, act at next open).

Usage:  python3 -B scripts/_trend_exit_tune.py SYMBOL   (writes JSON to stdout)
"""
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

EMA_PAIRS = [(16, 64), (32, 128), (64, 256)]
MIN_ABS_SIGNAL = 0.34
ATR_PERIOD = 20
ATR_STOP = 3.0          # initial catastrophic stop (fixed; entry signal is validated)

# ── sweep grid (exit params only; entry/signal fixed) ──
GRID = {
    "GIVEBACK": [0.25, 0.35, 0.50, 1.01],     # 1.01 = giveback exit OFF
    "LOCK":     [0.5, 0.6, 0.7, 0.8, 0.9],
    "ACT":      [0.3, 0.5],
    "TRAIL":    [1.0, 1.5, 2.0, 2.5, 3.0],
    "TP":       [6.0, 999.0],                 # 999 = no TP
}


def load(sym):
    p = C / ("raw_d1_" + sym.replace(".", "_") + ".pkl")
    if not p.exists():
        return None
    df = pickle.load(open(p, "rb"))
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def atr_series(h, l, c, n):
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def signal_series(close):
    sig = pd.Series(0.0, index=close.index)
    for f, s in EMA_PAIRS:
        sig = sig + np.sign(close.ewm(span=f).mean() - close.ewm(span=s).mean())
    sig /= len(EMA_PAIRS)
    return sig.apply(lambda v: 0 if abs(v) < MIN_ABS_SIGNAL else (1 if v > 0 else -1))


def simulate(df, p, cost_frac):
    """Walk D1 bars with the exit stack. Returns list of trade returns (fraction)."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    atr = atr_series(df["high"], df["low"], df["close"], ATR_PERIOD).values
    sig = signal_series(df["close"]).values
    hh = df["high"].rolling(22).max().values      # chandelier lookback fixed at 22
    ll = df["low"].rolling(22).min().values
    GB, LK, AC, TR, TP = p["GIVEBACK"], p["LOCK"], p["ACT"], p["TRAIL"], p["TP"]

    trades = []
    pos = 0          # 0 flat / +1 long / -1 short
    entry = sl = tp = peak = 0.0
    blocked = 0      # direction blocked for re-entry after a giveback/stop exit
    n = len(df)
    start = 260      # need EMA(256) + ATR warmup
    for t in range(start, n):
        a = atr[t - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        s = int(sig[t - 1])                 # yesterday's signal → act at today's open

        if pos == 0:
            if blocked and s != blocked:
                blocked = 0                 # signal changed → clear the block
            if s != 0 and s != blocked:
                pos = s
                entry = o[t]
                peak = 0.0
                sl = entry - ATR_STOP * a if s == 1 else entry + ATR_STOP * a
                tp = (entry + TP * a) if s == 1 else (entry - TP * a)
                if TP >= 999:
                    tp = None
            continue

        # ── manage open position on bar t ──
        # 1) flip on signal reversal → exit at this bar's OPEN, re-enter opposite
        if s != 0 and s != pos:
            ret = ((o[t] - entry) / entry) * pos - cost_frac
            trades.append(ret)
            pos_new = s
            entry = o[t]; peak = 0.0; pos = pos_new; blocked = 0
            sl = entry - ATR_STOP * a if pos == 1 else entry + ATR_STOP * a
            tp = (entry + TP * a) if pos == 1 else (entry - TP * a)
            if TP >= 999:
                tp = None
            continue

        # 2) CAUSAL: set SL + giveback from the peak/HH known as of the PRIOR bar,
        #    check this bar's exits, THEN update peak with this bar's extreme (so a
        #    single bar can't set the peak AND exit on its own pullback = look-ahead).
        if pos == 1:
            chand = hh[t - 1] - TR * a
            sl = max(sl, chand)
            if peak >= AC * a:
                sl = max(sl, entry + LK * peak)
            gb_level = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= AC * a) else -1e18
            exit_px = None
            if l[t] <= sl:                          # stop (chandelier/profit-lock)
                exit_px = sl
            elif gb_level > -1e17 and l[t] <= gb_level:   # peak-giveback reversal
                exit_px = gb_level
                blocked = pos
            elif tp is not None and h[t] >= tp:     # TP (adverse-first: after stop/gb)
                exit_px = tp
            if exit_px is not None:
                trades.append(((exit_px - entry) / entry) * pos - cost_frac)
                pos = 0
            else:
                peak = max(peak, h[t] - entry)      # update AFTER exit checks
        else:  # short
            chand = ll[t - 1] + TR * a
            sl = min(sl, chand)
            if peak >= AC * a:
                sl = min(sl, entry - LK * peak)
            gb_level = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= AC * a) else 1e18
            exit_px = None
            if h[t] >= sl:
                exit_px = sl
            elif gb_level < 1e17 and h[t] >= gb_level:
                exit_px = gb_level
                blocked = pos
            elif tp is not None and l[t] <= tp:
                exit_px = tp
            if exit_px is not None:
                trades.append(((exit_px - entry) / entry) * pos - cost_frac)
                pos = 0
            else:
                peak = max(peak, entry - l[t])
    return trades


def metrics(rets):
    if not rets:
        return {"n": 0, "ret": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0, "sharpe": 0.0}
    r = np.array(rets)
    wins = r[r > 0].sum(); losses = -r[r < 0].sum()
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = float((peak - eq).max())
    return {"n": len(r), "ret": float(r.sum()),
            "pf": float(wins / losses) if losses > 0 else (99.0 if wins > 0 else 0.0),
            "wr": float((r > 0).mean()),
            "dd": dd,
            "sharpe": float(r.mean() / r.std() * np.sqrt(252 / max(1, len(r) / (len(r))))) if r.std() > 0 else 0.0}


def main():
    sym = sys.argv[1]
    df = load(sym)
    if df is None or len(df) < 400:
        print(json.dumps({"symbol": sym, "error": "insufficient data"})); return
    # per-round-trip cost from the data spread (spread-only, no slippage)
    med_spread_pts = float(np.nanmedian(df["spread"].values)) if "spread" in df else 0.0
    px = float(df["close"].iloc[-1])
    cost_frac = 2.0 * (med_spread_pts * 0.01) / px if px > 0 else 0.0002
    cost_frac = max(cost_frac, 0.0001)

    split = int(len(df) * 0.70)
    df_is = df.iloc[:split].reset_index(drop=True)
    df_oos = df.iloc[split - 260:].reset_index(drop=True)   # keep warmup lead-in

    results = []
    keys = list(GRID.keys())
    for combo in product(*[GRID[k] for k in keys]):
        p = dict(zip(keys, combo))
        mis = metrics(simulate(df_is, p, cost_frac))
        moos = metrics(simulate(df_oos, p, cost_frac))
        results.append({"p": p, "is": mis, "oos": moos})

    # robustness: OOS return positive, IS positive, decent trade count, rank by
    # blended (OOS ret * 0.6 + IS ret * 0.4) then OOS PF as tiebreak.
    def score(r):
        return r["oos"]["ret"] * 0.6 + r["is"]["ret"] * 0.4
    robust = [r for r in results if r["oos"]["ret"] > 0 and r["is"]["ret"] > 0 and r["oos"]["n"] >= 5]
    robust.sort(key=lambda r: (score(r), r["oos"]["pf"]), reverse=True)

    # baseline = the CURRENT live defaults
    base_p = {"GIVEBACK": 0.35, "LOCK": 0.6, "ACT": 0.3, "TRAIL": 2.5, "TP": 6.0}
    base = {"p": base_p, "is": metrics(simulate(df_is, base_p, cost_frac)),
            "oos": metrics(simulate(df_oos, base_p, cost_frac))}
    # anchor = PURE FLIP (no trail/lock/giveback/TP; just signal + wide 3xATR stop).
    # If the exit model isn't well above this on OOS, the extra machinery is noise.
    flip_p = {"GIVEBACK": 1.01, "LOCK": 0.0, "ACT": 99.0, "TRAIL": 99.0, "TP": 999.0}
    flip = {"p": "pure_flip", "is": metrics(simulate(df_is, flip_p, cost_frac)),
            "oos": metrics(simulate(df_oos, flip_p, cost_frac))}

    out = {"symbol": sym, "bars": len(df), "cost_frac": round(cost_frac, 5),
           "split_date": str(df["time"].iloc[split])[:10],
           "pure_flip": flip, "baseline": base, "top": robust[:8],
           "n_robust": len(robust), "n_total": len(results)}
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
