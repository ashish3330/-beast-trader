#!/usr/bin/env python3 -B
"""SMABO BTCUSD coord-descent tuner (365d window).

Strategy: coord-descent over 5 axes (FAST_SMA, SLOW_SMA, TRAIL_SMA,
HTF_LOOKBACK, MIN_RR), 2 passes. Top-3 candidates then re-run as combined
stability check.

Mirrors backtest/sma_breakout_backtest.py for the simulation loop, but
injects params via SMABreakoutStrategy(state, params={...}) for each combo.

Outputs JSON to /tmp/smabo_btcusd_tune_20260621.json
"""
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
SYMBOL = "BTCUSD"
DAYS = 365
SPREAD_BTC = 30.0
START_CAPITAL = 5000.0
RISK_PCT = 0.0025
MIN_BARS_FOR_BT = 2000
M15_WINDOW = 800
H1_WINDOW = 240
TIME_STOP_BARS = 96
TIME_STOP_PEAK_R = 0.5
MIN_TRADES_FLOOR = 40

# Axes (per task spec).
AXES = {
    "FAST_SMA":          [5, 8, 13, 21],
    "SLOW_SMA":          [20, 50, 100, 200],
    "TRAIL_SMA":         [10, 14, 20, 34],
    "HTF_LOOKBACK_BARS": [20, 30, 50, 80, 120],
    "MIN_RR":            [1.5, 2.0, 2.5, 3.0],
}
DEFAULT_PARAMS = {
    "FAST_SMA": 8,
    "SLOW_SMA": 50,
    "TRAIL_SMA": 20,
    "HTF_LOOKBACK_BARS": 50,
    "MIN_RR": 2.0,
}


# ── data load ───────────────────────────────────────────────────────────
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


def _summarize(trades):
    if not trades:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "total_R": 0.0,
                "avg_R": 0.0, "max_dd_pct": 0.0}
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
        "pf": float(pf),
        "wr": float((Rs > 0).mean()),
        "avg_R": float(Rs.mean()),
        "total_R": float(Rs.sum()),
        "max_dd_pct": float(abs(dd.min()) * 100.0),
    }


# Pre-load data once
print(f"[loading] M15 cache for {SYMBOL} ({DAYS}d window)...", flush=True)
m15_full = _load_m15(SYMBOL)
if m15_full is None:
    raise SystemExit("BTCUSD cache missing")
n_bars_keep = int(DAYS * 24 * 4)
m15 = m15_full.iloc[-n_bars_keep:].reset_index(drop=True) if n_bars_keep < len(m15_full) else m15_full
h1 = _resample_h1(m15)
print(f"  M15 bars: {len(m15)}, H1 bars: {len(h1)}", flush=True)


def run_combo(params):
    """Run one combo through the BT loop. Returns summary dict."""
    state = _FakeState(m15, h1)
    strat = SMABreakoutStrategy(state, params=params)
    arr = (m15["high"].values, m15["low"].values, m15["close"].values)
    trades = []
    open_until = -1
    n = len(m15)
    start_i = max(800, 60)
    for i in range(start_i, n - 1):
        if i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(SYMBOL, None)
        sig = strat.evaluate(SYMBOL)
        if sig is None:
            continue
        tr = _simulate_trade(sig, arr, entry_i=i, spread=SPREAD_BTC)
        if tr is None:
            continue
        trades.append(tr)
        open_until = i + tr["bars_held"]
    s = _summarize(trades)
    return s


def score(s):
    """Score = PF subject to MIN_TRADES_FLOOR (else 0)."""
    if s["trades"] < MIN_TRADES_FLOOR:
        return -1.0
    return s["pf"]


def fmt(p, s):
    return (f"  params={p}  trades={s['trades']:>4}  PF={s['pf']:>5.2f}  "
            f"WR={s['wr']*100:>5.1f}%  totalR={s['total_R']:>+7.1f}  "
            f"DD={s['max_dd_pct']:>5.1f}%")


# ── BASELINE ────────────────────────────────────────────────────────────
print("\n[BASELINE]", flush=True)
t0 = time.time()
base_s = run_combo(DEFAULT_PARAMS.copy())
print(fmt(DEFAULT_PARAMS, base_s) + f"  ({time.time()-t0:.1f}s)", flush=True)

results = []  # all combos
results.append({"params": DEFAULT_PARAMS.copy(), "summary": base_s, "stage": "baseline"})


# ── COORD-DESCENT ───────────────────────────────────────────────────────
locked = DEFAULT_PARAMS.copy()
axis_order = list(AXES.keys())

for pass_idx in (1, 2):
    print(f"\n========== PASS {pass_idx} ==========", flush=True)
    for axis in axis_order:
        print(f"\n[PASS {pass_idx}] sweeping {axis} (locked={locked})", flush=True)
        best_score = -1.0
        best_val = locked[axis]
        best_summary = None
        for v in AXES[axis]:
            p = locked.copy()
            p[axis] = v
            # Skip degenerate FAST >= SLOW combos.
            if p["FAST_SMA"] >= p["SLOW_SMA"]:
                print(f"  SKIP {axis}={v}: FAST>=SLOW", flush=True)
                continue
            # Skip if already done identically in this run (cache).
            cached = next((r for r in results if r["params"] == p), None)
            if cached:
                s = cached["summary"]
            else:
                t0 = time.time()
                s = run_combo(p)
                dt = time.time() - t0
                results.append({"params": p.copy(), "summary": s,
                                "stage": f"pass{pass_idx}_{axis}", "secs": dt})
            sc = score(s)
            tag = "  "
            if sc > best_score:
                best_score = sc
                best_val = v
                best_summary = s
                tag = "* "
            note = "" if s["trades"] >= MIN_TRADES_FLOOR else "  [<floor]"
            print(f" {tag}{axis}={v:<5}  trades={s['trades']:>4} "
                  f"PF={s['pf']:>5.2f} totalR={s['total_R']:>+6.1f} "
                  f"DD={s['max_dd_pct']:>5.1f}%{note}", flush=True)
        locked[axis] = best_val
        print(f"  -> LOCKED {axis}={best_val}", flush=True)

print(f"\n[POST-COORD-DESCENT] locked = {locked}", flush=True)


# ── TOP-3 COMBINED STABILITY ────────────────────────────────────────────
# Re-rank all combos that met floor by PF, take top 3, re-run.
scored = [r for r in results
          if r["summary"]["trades"] >= MIN_TRADES_FLOOR
          and r["summary"]["max_dd_pct"] < 30.0]
scored.sort(key=lambda r: r["summary"]["pf"], reverse=True)
top3 = scored[:3]

print("\n========== TOP-3 STABILITY RE-RUN ==========", flush=True)
final = []
for r in top3:
    p = r["params"]
    t0 = time.time()
    s = run_combo(p)
    dt = time.time() - t0
    final.append({"params": p, "summary": s, "secs": dt})
    print(fmt(p, s) + f"  ({dt:.1f}s)", flush=True)


# ── PICK WINNER ─────────────────────────────────────────────────────────
# Must beat baseline AND acceptance: PF >= max(1.3, base_pf*1.2), totalR > base_totalR, DD < 30.
PF_FLOOR = max(1.3, base_s["pf"] * 1.20)
TOTR_FLOOR = base_s["total_R"]

acceptable = [f for f in final
              if f["summary"]["pf"] >= PF_FLOOR
              and f["summary"]["total_R"] > TOTR_FLOOR
              and f["summary"]["max_dd_pct"] < 30.0
              and f["summary"]["trades"] >= MIN_TRADES_FLOOR]
acceptable.sort(key=lambda f: f["summary"]["pf"], reverse=True)

winner = acceptable[0] if acceptable else None

print("\n========== ACCEPTANCE ==========", flush=True)
print(f"  PF_FLOOR  = {PF_FLOOR:.3f}", flush=True)
print(f"  TOTR_FLOOR= {TOTR_FLOOR:+.2f}", flush=True)
if winner:
    print("  WINNER:", flush=True)
    print(fmt(winner["params"], winner["summary"]), flush=True)
else:
    print("  NO COMBO MET ACCEPTANCE.", flush=True)

# Also report best-by-PF regardless of acceptance (for diagnostics).
all_scored = [r for r in results
              if r["summary"]["trades"] >= MIN_TRADES_FLOOR]
all_scored.sort(key=lambda r: r["summary"]["pf"], reverse=True)
print("\n[BEST 5 BY PF (n>=floor)]", flush=True)
for r in all_scored[:5]:
    print(fmt(r["params"], r["summary"]), flush=True)


# ── PERSIST ─────────────────────────────────────────────────────────────
out = {
    "symbol": SYMBOL,
    "days": DAYS,
    "min_trades_floor": MIN_TRADES_FLOOR,
    "baseline": {"params": DEFAULT_PARAMS, "summary": base_s},
    "axes": AXES,
    "combos_tried": len([r for r in results if r["stage"] != "baseline"]),
    "all_results": results,
    "top3_stability": final,
    "winner": winner,
    "acceptance": {"pf_floor": PF_FLOOR, "totR_floor": TOTR_FLOOR,
                   "dd_ceiling": 30.0, "min_trades": MIN_TRADES_FLOOR},
}
OUT = Path("/tmp/smabo_btcusd_tune_20260621.json")
OUT.write_text(json.dumps(out, indent=2, default=str))
print(f"\n[wrote] {OUT}", flush=True)
