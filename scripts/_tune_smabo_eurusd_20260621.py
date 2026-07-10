#!/usr/bin/env python3 -B
"""
SMABO per-symbol coord-descent tune — EURUSD, 365d window.

Builds on backtest/sma_breakout_backtest.py harness but injects a `params`
dict into SMABreakoutStrategy(state, params=...) so we can sweep axes
without editing config.

Coord-descent over:
    FAST_SMA, SLOW_SMA, TRAIL_SMA, HTF_LOOKBACK_BARS, MIN_RR

Two passes, then a top-3 stability re-run.

Writes JSON results to scripts/_tune_smabo_eurusd_20260621.json.
"""
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.sma_breakout import SMABreakoutStrategy  # noqa: E402

# ─── Cache + simulator copied from backtest/sma_breakout_backtest.py ────
CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

SPREAD_EURUSD = 0.00015
SYMBOL = "EURUSD"
DAYS = 365
START_CAPITAL = 5000.0
RISK_PCT = 0.0025
MIN_BARS_FOR_BT = 2000
M15_WINDOW = 800
H1_WINDOW = 240
TIME_STOP_BARS = 96
TIME_STOP_PEAK_R = 0.5

MIN_TRADES_FLOOR = 80


def _load(symbol, base="m15"):
    for v in (f"raw_{base}_{symbol}.pkl",
              f"raw_{base}_{symbol.replace('.', '_')}.pkl",
              f"raw_{base}_{symbol.lower()}.pkl"):
        p = CACHE / v
        if p.exists():
            df = pickle.load(open(p, "rb")).copy()
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").reset_index(drop=True)
            try:
                df["time"] = df["time"].dt.as_unit("ns")
            except Exception:
                pass
            cols = ["time", "open", "high", "low", "close"]
            return df[cols].astype({c: float for c in cols if c != "time"})
    return None


def _resample_h1(m15):
    d = m15.set_index("time")
    out = pd.DataFrame({
        "open": d["open"].resample("1h").first(),
        "high": d["high"].resample("1h").max(),
        "low": d["low"].resample("1h").min(),
        "close": d["close"].resample("1h").last(),
    }).dropna().reset_index()
    return out


class _FakeState:
    def __init__(self, m15, h1):
        self.m15 = m15
        self.h1 = h1
        self.cursor = 0
        self._m15_ns = m15["time"].astype("int64").values
        self._h1_ns = h1["time"].astype("int64").values

    def set_cursor(self, i):
        self.cursor = i

    def get_candles(self, symbol, tf):
        c = self.cursor
        if tf == 15:
            lo = max(0, c + 1 - M15_WINDOW)
            base = self.m15.iloc[lo: c + 1]
            if len(base) == 0:
                return None
            last_close = base["close"].iloc[-1]
            last_time = base["time"].iloc[-1]
            row = {"time": last_time + pd.Timedelta(minutes=15),
                   "open": last_close, "high": last_close,
                   "low": last_close, "close": last_close}
            return pd.concat([base, pd.DataFrame([row])], ignore_index=True)
        if tf == 60:
            if c >= len(self._m15_ns):
                end = len(self.h1)
            else:
                end = int(np.searchsorted(self._h1_ns,
                                          int(self._m15_ns[c]),
                                          side="right"))
            if end == 0:
                return None
            lo = max(0, end - H1_WINDOW)
            return self.h1.iloc[lo: end]
        return None


def _simulate_trade(sig, m15_arr, entry_i, spread):
    H, L, C = m15_arr
    n = len(C)
    d = 1 if sig["direction"] == "LONG" else -1
    entry = sig["entry"] + d * spread
    sl = sig["sl"]
    tp1 = sig["tp1"]
    tp2 = sig["tp2"]
    risk = (entry - sl) if d == 1 else (sl - entry)
    if risk <= 0:
        return None

    tp1_hit = False
    runner_open = True
    r_tp1 = 0.0
    r_run = 0.0
    peak_r = 0.0
    bars_held = 0
    runner_sl = sl

    end = min(n, entry_i + 1 + 800)
    for j in range(entry_i + 1, end):
        bars_held += 1
        hi, lo = H[j], L[j]
        cur_r = (hi - entry) / risk if d == 1 else (entry - lo) / risk
        if cur_r > peak_r:
            peak_r = cur_r

        if d == 1:
            if not tp1_hit:
                if lo <= sl:
                    r_tp1 = -1.0
                    r_run = -1.0
                    runner_open = False
                    break
                if hi >= tp1:
                    r_tp1 = (tp1 - entry) / risk
                    tp1_hit = True
                    runner_sl = entry
            else:
                if lo <= runner_sl:
                    r_run = (runner_sl - entry) / risk
                    runner_open = False
                    break
                if hi >= tp2:
                    r_run = (tp2 - entry) / risk
                    runner_open = False
                    break
        else:
            if not tp1_hit:
                if hi >= sl:
                    r_tp1 = -1.0
                    r_run = -1.0
                    runner_open = False
                    break
                if lo <= tp1:
                    r_tp1 = (entry - tp1) / risk
                    tp1_hit = True
                    runner_sl = entry
            else:
                if hi >= runner_sl:
                    r_run = (entry - runner_sl) / risk
                    runner_open = False
                    break
                if lo <= tp2:
                    r_run = (entry - tp2) / risk
                    runner_open = False
                    break

        if bars_held >= TIME_STOP_BARS and peak_r < TIME_STOP_PEAK_R:
            close_px = C[j]
            r_now = (close_px - entry) / risk if d == 1 else (entry - close_px) / risk
            if not tp1_hit:
                r_tp1 = r_now
                r_run = r_now
            else:
                r_run = r_now
            runner_open = False
            break
    else:
        if runner_open:
            close_px = C[end - 1]
            r_now = (close_px - entry) / risk if d == 1 else (entry - close_px) / risk
            if not tp1_hit:
                r_tp1 = r_now
                r_run = r_now
            else:
                r_run = r_now

    return {"R": 0.5 * r_tp1 + 0.5 * r_run, "tp1_hit": tp1_hit, "dir": d,
            "entry_i": entry_i, "bars_held": bars_held, "peak_r": peak_r}


def _summarize(trades, n_bars):
    if not trades:
        return None
    Rs = np.array([t["R"] for t in trades])
    wins = Rs[Rs > 0]
    losses = Rs[Rs <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    equity = [START_CAPITAL]
    risk_d = START_CAPITAL * RISK_PCT
    for r in Rs:
        equity.append(equity[-1] + r * risk_d)
    eq = np.array(equity)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    return {
        "trades": int(len(trades)),
        "wr": float((Rs > 0).mean()),
        "pf": float(pf),
        "total_R": float(Rs.sum()),
        "avg_R": float(Rs.mean()),
        "max_dd_pct": float(abs(dd.min()) * 100.0),
    }


def _run_one(params, m15, h1):
    """Run a single combo and return summary stats."""
    state = _FakeState(m15, h1)
    strat = SMABreakoutStrategy(state, params=params)
    arr = (m15["high"].values, m15["low"].values, m15["close"].values)
    n = len(m15)
    trades = []
    open_until = -1
    start_i = max(800, 60)
    for i in range(start_i, n - 1):
        if i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(SYMBOL, None)
        sig = strat.evaluate(SYMBOL)
        if sig is None:
            continue
        tr = _simulate_trade(sig, arr, entry_i=i, spread=SPREAD_EURUSD)
        if tr is None:
            continue
        trades.append(tr)
        open_until = i + tr["bars_held"]
    return _summarize(trades, n)


def main():
    print(f"Loading {SYMBOL} cache...")
    m15 = _load(SYMBOL, "m15")
    if m15 is None:
        print("NO CACHE")
        sys.exit(1)
    n_bars_keep = int(DAYS * 24 * 4)
    if n_bars_keep < len(m15):
        m15 = m15.iloc[-n_bars_keep:].reset_index(drop=True)
    n = len(m15)
    if n < MIN_BARS_FOR_BT:
        print(f"INSUFFICIENT BARS {n}")
        sys.exit(1)
    h1 = _resample_h1(m15)
    print(f"  {n} M15 bars, {len(h1)} H1 bars")

    # ─── Axis values ───────────────────────────────────────────────
    AXES = {
        "FAST_SMA":          [5, 8, 13, 21],
        "SLOW_SMA":          [20, 50, 100, 200],
        "TRAIL_SMA":         [10, 14, 20, 34],
        "HTF_LOOKBACK_BARS": [20, 30, 50, 80, 120],
        "MIN_RR":            [1.5, 2.0, 2.5, 3.0],
    }
    DEFAULTS = {
        "FAST_SMA": 8,
        "SLOW_SMA": 50,
        "TRAIL_SMA": 20,
        "HTF_LOOKBACK_BARS": 50,
        "MIN_RR": 2.0,
    }

    all_results = []
    locked = dict(DEFAULTS)

    def score(s):
        """Score combos: PF, then total_R, only if trades >= floor."""
        if s is None or s["trades"] < MIN_TRADES_FLOOR:
            return -1e9
        # Sanity DD ceiling 30 — penalize but still rank
        if s["max_dd_pct"] >= 30.0:
            return s["pf"] - 100  # hard-rank below acceptable
        return s["pf"]

    print("\n=== Pass 1 ===")
    for axis, values in AXES.items():
        print(f"\n  Axis {axis}: testing {values} (locked: {locked})")
        axis_results = []
        for v in values:
            params = dict(locked)
            params[axis] = v
            t0 = time.time()
            s = _run_one(params, m15, h1)
            dt = time.time() - t0
            entry = {"pass": 1, "axis": axis, "params": dict(params),
                     "summary": s, "dt": dt}
            all_results.append(entry)
            axis_results.append(entry)
            if s is None:
                print(f"    {axis}={v}: NO TRADES  ({dt:.1f}s)")
            else:
                print(f"    {axis}={v}: trades={s['trades']:4d} "
                      f"PF={s['pf']:.2f} totalR={s['total_R']:+.1f} "
                      f"WR={s['wr']*100:.1f}% DD={s['max_dd_pct']:.1f}% "
                      f"({dt:.1f}s)")
        # pick best on this axis with floor + DD ceiling
        scored = [(score(e["summary"]), e) for e in axis_results]
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        if scored[0][0] > -1e8:
            locked[axis] = best["params"][axis]
            print(f"    --> LOCK {axis}={locked[axis]}")
        else:
            print(f"    --> NO QUALIFIER on {axis}; keeping {locked[axis]}")

    print(f"\n  Pass 1 winners: {locked}")

    print("\n=== Pass 2 ===")
    for axis, values in AXES.items():
        print(f"\n  Axis {axis}: testing {values} (locked: {locked})")
        axis_results = []
        for v in values:
            params = dict(locked)
            params[axis] = v
            t0 = time.time()
            s = _run_one(params, m15, h1)
            dt = time.time() - t0
            entry = {"pass": 2, "axis": axis, "params": dict(params),
                     "summary": s, "dt": dt}
            all_results.append(entry)
            axis_results.append(entry)
            if s is None:
                print(f"    {axis}={v}: NO TRADES  ({dt:.1f}s)")
            else:
                print(f"    {axis}={v}: trades={s['trades']:4d} "
                      f"PF={s['pf']:.2f} totalR={s['total_R']:+.1f} "
                      f"WR={s['wr']*100:.1f}% DD={s['max_dd_pct']:.1f}% "
                      f"({dt:.1f}s)")
        scored = [(score(e["summary"]), e) for e in axis_results]
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        if scored[0][0] > -1e8:
            locked[axis] = best["params"][axis]
            print(f"    --> LOCK {axis}={locked[axis]}")
        else:
            print(f"    --> NO QUALIFIER on {axis}; keeping {locked[axis]}")

    print(f"\n  Pass 2 winners (final coord-descent): {locked}")

    # ─── Top-3 stability re-run ───────────────────────────────────
    print("\n=== Top-3 stability check ===")
    qual = [e for e in all_results
            if e["summary"] is not None
            and e["summary"]["trades"] >= MIN_TRADES_FLOOR
            and e["summary"]["max_dd_pct"] < 30.0]
    qual.sort(key=lambda x: x["summary"]["pf"], reverse=True)
    seen = set()
    top3 = []
    for e in qual:
        key = tuple(sorted(e["params"].items()))
        if key in seen:
            continue
        seen.add(key)
        top3.append(e)
        if len(top3) == 3:
            break
    final_candidates = []
    for e in top3:
        s = _run_one(e["params"], m15, h1)
        final_candidates.append({"params": e["params"], "summary": s})
        print(f"  {e['params']} --> PF={s['pf']:.2f} totalR={s['total_R']:+.1f} "
              f"trades={s['trades']} DD={s['max_dd_pct']:.1f}%")

    # Pick best winner subject to acceptance criteria
    final_candidates.sort(key=lambda c: c["summary"]["pf"], reverse=True)
    winner = final_candidates[0] if final_candidates else None

    out = {
        "symbol": SYMBOL,
        "days": DAYS,
        "axes": AXES,
        "defaults": DEFAULTS,
        "min_trades_floor": MIN_TRADES_FLOOR,
        "pass1_pass2_locked": locked,
        "all_results": all_results,
        "top3_rerun": final_candidates,
        "winner": winner,
    }
    out_path = ROOT / "scripts" / "_tune_smabo_eurusd_20260621.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    if winner is not None:
        print(f"WINNER: {winner['params']}")
        print(f"   PF={winner['summary']['pf']:.2f} "
              f"totalR={winner['summary']['total_R']:+.1f} "
              f"trades={winner['summary']['trades']} "
              f"DD={winner['summary']['max_dd_pct']:.1f}%")


if __name__ == "__main__":
    main()
