#!/usr/bin/env python3 -B
"""Parametric SMABO coord-descent tuner — invoke per symbol.

Mirrors _tune_smabo_xauusd_20260621.py exactly but accepts --symbol --spread
so all 6 candidate syms reuse one harness.

CLI:
    python3 -B scripts/_tune_smabo_parametric_20260621.py --symbol NAS100.r
    python3 -B scripts/_tune_smabo_parametric_20260621.py --symbol CHFJPY
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

from agent.sma_breakout import SMABreakoutStrategy  # noqa: E402

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_CAPITAL = 5000.0
RISK_PCT = 0.0025
MIN_BARS_FOR_BT = 2000
M15_WINDOW = 800
H1_WINDOW = 240
TIME_STOP_BARS = 96
TIME_STOP_PEAK_R = 0.5
MIN_TRADES = 80

# Spread table per sym (mirrors backtest/sma_breakout_backtest.py SPREAD).
SPREAD_TABLE = {
    "EURUSD": 0.00015, "GBPUSD": 0.00020, "USDJPY": 0.015, "USDCAD": 0.00020,
    "USDCHF": 0.00020, "AUDJPY": 0.020, "EURJPY": 0.020, "GBPJPY": 0.025,
    "CHFJPY": 0.020,
    "XAUUSD": 0.30, "XAGUSD": 0.030, "BTCUSD": 30.0, "ETHUSD": 2.0,
    "NAS100.r": 1.50, "SP500.r": 0.50, "DJ30.r": 2.0, "US2000.r": 0.50,
    "GER40.r": 2.0, "UK100.r": 2.0, "JPN225ft": 10.0, "SPI200.r": 2.0,
    "SWI20.r": 3.0, "XPTUSD.r": 1.0, "USOUSD": 0.05,
}

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


def _load_m15(symbol):
    for v in (f"raw_m15_{symbol}.pkl",
              f"raw_m15_{symbol.replace('.', '_')}.pkl",
              f"raw_m15_{symbol.lower()}.pkl"):
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


def _simulate_trade(sig, H, L, C, entry_i, spread):
    n = len(C)
    d = 1 if sig["direction"] == "LONG" else -1
    entry = sig["entry"] + d * spread
    sl = sig["sl"]
    tp1 = sig["tp1"]
    tp2 = sig["tp2"]
    risk = (entry - sl) if d == 1 else (sl - entry)
    if risk <= 0:
        return None
    tp1_hit = False; runner_open = True; r_tp1 = 0.0; r_run = 0.0
    peak_r = 0.0; bars_held = 0; runner_sl = sl
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
                    r_tp1 = -1.0; r_run = -1.0; runner_open = False; break
                if hi >= tp1:
                    r_tp1 = (tp1 - entry) / risk; tp1_hit = True; runner_sl = entry
            else:
                if lo <= runner_sl:
                    r_run = (runner_sl - entry) / risk; runner_open = False; break
                if hi >= tp2:
                    r_run = (tp2 - entry) / risk; runner_open = False; break
        else:
            if not tp1_hit:
                if hi >= sl:
                    r_tp1 = -1.0; r_run = -1.0; runner_open = False; break
                if lo <= tp1:
                    r_tp1 = (entry - tp1) / risk; tp1_hit = True; runner_sl = entry
            else:
                if hi >= runner_sl:
                    r_run = (entry - runner_sl) / risk; runner_open = False; break
                if lo <= tp2:
                    r_run = (entry - tp2) / risk; runner_open = False; break
        if bars_held >= TIME_STOP_BARS and peak_r < TIME_STOP_PEAK_R:
            close_px = C[j]
            r_now = (close_px - entry) / risk if d == 1 else (entry - close_px) / risk
            if not tp1_hit:
                r_tp1 = r_now; r_run = r_now
            else:
                r_run = r_now
            runner_open = False; break
    else:
        if runner_open:
            close_px = C[end - 1]
            r_now = (close_px - entry) / risk if d == 1 else (entry - close_px) / risk
            if not tp1_hit:
                r_tp1 = r_now; r_run = r_now
            else:
                r_run = r_now
    return {"R": 0.5 * r_tp1 + 0.5 * r_run, "tp1_hit": tp1_hit, "dir": d,
            "entry_i": entry_i, "bars_held": bars_held, "peak_r": peak_r}


def _summarize(trades):
    if not trades:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "total_R": 0.0,
                "max_dd_pct": 0.0, "avg_R": 0.0}
    Rs = np.array([t["R"] for t in trades])
    wins = Rs[Rs > 0]; losses = Rs[Rs <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    equity = [START_CAPITAL]
    risk_d = START_CAPITAL * RISK_PCT
    for r in Rs:
        equity.append(equity[-1] + r * risk_d)
    eq = np.array(equity)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    return {
        "trades": int(len(trades)), "pf": float(pf),
        "wr": float((Rs > 0).mean()), "total_R": float(Rs.sum()),
        "avg_R": float(Rs.mean()),
        "max_dd_pct": float(abs(dd.min()) * 100.0),
    }


_DATA = {}


def _ensure_data(symbol, days):
    m15 = _load_m15(symbol)
    if m15 is None:
        raise RuntimeError(f"No M15 cache for {symbol}")
    n_keep = int(days * 24 * 4)
    if n_keep < len(m15):
        m15 = m15.iloc[-n_keep:].reset_index(drop=True)
    h1 = _resample_h1(m15)
    _DATA["m15"] = m15
    _DATA["h1"] = h1
    _DATA["H"] = m15["high"].values
    _DATA["L"] = m15["low"].values
    _DATA["C"] = m15["close"].values
    _DATA["n"] = len(m15)


def run_combo(symbol, spread, params):
    m15 = _DATA["m15"]; h1 = _DATA["h1"]
    H = _DATA["H"]; L = _DATA["L"]; C = _DATA["C"]; n = _DATA["n"]
    if n < MIN_BARS_FOR_BT:
        return None
    state = _FakeState(m15, h1)
    strat = SMABreakoutStrategy(state, params=params)
    trades = []
    open_until = -1
    start_i = max(800, 60)
    for i in range(start_i, n - 1):
        if i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(symbol, None)
        sig = strat.evaluate(symbol)
        if sig is None:
            continue
        tr = _simulate_trade(sig, H, L, C, entry_i=i, spread=spread)
        if tr is None:
            continue
        trades.append(tr)
        open_until = i + tr["bars_held"]
    return _summarize(trades)


def is_better(a, b):
    if a is None or a["trades"] < MIN_TRADES:
        return False
    if b is None or b["trades"] < MIN_TRADES:
        return True
    if a["pf"] > b["pf"]:
        return True
    if a["pf"] == b["pf"] and a["total_R"] > b["total_R"]:
        return True
    return False


def coord_descent(symbol, spread):
    locked = dict(DEFAULTS)
    all_results = []
    print(f"[BASELINE] params={locked}", flush=True)
    base = run_combo(symbol, spread, locked)
    print(f"  trades={base['trades']} PF={base['pf']:.2f} totR={base['total_R']:+.1f} "
          f"WR={base['wr']*100:.1f}% DD={base['max_dd_pct']:.1f}%", flush=True)
    all_results.append({"pass": 0, "params": dict(locked), "summary": base})

    for pass_i in range(1, 3):
        print(f"\n========== PASS {pass_i} ==========", flush=True)
        for axis, values in AXES.items():
            print(f"\n-- {axis} (locked: {locked}) --", flush=True)
            best_for_axis = None
            best_val = locked[axis]
            for v in values:
                trial = dict(locked)
                trial[axis] = v
                summ = run_combo(symbol, spread, trial)
                if summ is None:
                    continue
                tag = "FLOOR" if summ["trades"] < MIN_TRADES else "OK"
                print(f"   {axis}={v:>5} trd={summ['trades']:>4} PF={summ['pf']:.2f} "
                      f"R={summ['total_R']:+7.1f} WR={summ['wr']*100:5.1f}% "
                      f"DD={summ['max_dd_pct']:5.1f}% [{tag}]", flush=True)
                all_results.append({"pass": pass_i, "axis": axis,
                                    "params": dict(trial), "summary": summ})
                if is_better(summ, best_for_axis):
                    best_for_axis = summ
                    best_val = v
            if best_for_axis is not None:
                locked[axis] = best_val
                print(f"   [WIN] {axis}={best_val} PF={best_for_axis['pf']:.2f}", flush=True)
            else:
                print(f"   [NONE] {axis} stays {locked[axis]}", flush=True)
        summ = run_combo(symbol, spread, locked)
        print(f"\n[PASS {pass_i} LOCKED] {locked} -> PF={summ['pf']:.2f} "
              f"R={summ['total_R']:+.1f} trd={summ['trades']}", flush=True)
        all_results.append({"pass": pass_i, "axis": "LOCKED",
                            "params": dict(locked), "summary": summ})
    return locked, all_results, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sym = args.symbol
    spread = SPREAD_TABLE.get(sym, 0.0002)
    out_json = args.out or str(ROOT / "scripts" /
        f"_tune_smabo_{sym.replace('.','_').lower()}_20260621.json")

    t0 = time.time()
    _ensure_data(sym, args.days)
    print(f"Loaded {sym}: {_DATA['n']} bars  spread={spread}\n", flush=True)

    locked, all_results, baseline = coord_descent(sym, spread)

    # Top-3 stability check
    seen = set(); scored = []
    for r in all_results:
        s = r["summary"]
        if s["trades"] < MIN_TRADES:
            continue
        k = json.dumps(r["params"], sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        scored.append((s["pf"], s["total_R"], r["params"], s))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top3 = scored[:3]
    print("\n========== TOP-3 ==========", flush=True)
    rerun = []
    for pf, tr, p, s in top3:
        s2 = run_combo(sym, spread, p)
        print(f"  {p}", flush=True)
        print(f"   orig: PF={s['pf']:.2f} R={s['total_R']:+.1f} trd={s['trades']}", flush=True)
        print(f"   rerun:PF={s2['pf']:.2f} R={s2['total_R']:+.1f} trd={s2['trades']} "
              f"DD={s2['max_dd_pct']:.1f}%", flush=True)
        rerun.append({"params": p, "summary": s2})

    # Winner selection — best PF subject to DD<30%, trades>=MIN_TRADES
    winner = None
    for entry in rerun:
        s = entry["summary"]
        if s["trades"] < MIN_TRADES or s["max_dd_pct"] >= 30.0:
            continue
        if winner is None or s["pf"] > winner["summary"]["pf"]:
            winner = entry

    out = {
        "symbol": sym, "days": args.days, "spread": spread,
        "min_trades": MIN_TRADES,
        "baseline": {"params": DEFAULTS, "summary": baseline},
        "locked_after_passes": {"params": locked,
                                "summary": run_combo(sym, spread, locked)},
        "top3": rerun, "winner": winner,
        "all_combo_count": sum(1 for r in all_results if r.get("axis") != "LOCKED"),
        "all_results": all_results,
        "elapsed_sec": time.time() - t0,
    }
    Path(out_json).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote: {out_json}", flush=True)
    print(f"\n========== FINAL ({sym}) ==========", flush=True)
    print(f"Baseline: PF={baseline['pf']:.2f} R={baseline['total_R']:+.1f}", flush=True)
    if winner:
        s = winner["summary"]
        print(f"Winner:   params={winner['params']}", flush=True)
        print(f"          PF={s['pf']:.2f} R={s['total_R']:+.1f} trd={s['trades']} "
              f"DD={s['max_dd_pct']:.1f}% WR={s['wr']*100:.1f}%", flush=True)
    else:
        print("Winner:   NONE (no combo cleared trade floor + DD ceiling)", flush=True)


if __name__ == "__main__":
    main()
