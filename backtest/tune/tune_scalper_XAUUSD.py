#!/usr/bin/env python3 -B
"""Hard-tune the LIVE M1 SCALPER book (6th book) — XAUUSD only.

Self-contained. Mirrors agent/m1_scalper.evaluate() + the brain's execution
(SL=SL_ATR*ATR, TP=rolling mean, M1 time-stop, one concurrent, post-close
cooldown) so the backtest matches live. REAL spread charged both sides.

Method
------
  * Data: raw_m1_xauusd.pkl (M1). Window stated at runtime.
  * Objective: total R (primary) + PF + WR. Scalper = many small trades, so
    PF/WR/cost matter as much as R.
  * WF 60/40 chronological: TRAIN = first 60% of bars, TEST = last 40%.
    A best-on-train param value SHIPS only if it also stays >= baseline on the
    held-out TEST (neutral floor) with PF>=1.0 and enough test trades.
  * One-at-a-time per-param sweep -> per-param SHIP / SHIP_NONE, then the
    surviving winners are combined and WF-validated together.

Cost: XAU M1 spread ~= 0.30 price (current live symbol spread = 30 pts; the
100k-bar sample median = 23 pts). We charge 0.30 (conservative). SPREAD env
overrides for sensitivity.

Run:  python3 -B backtest/tune/tune_scalper_XAUUSD.py
Emits a human table + writes results_scalper_XAUUSD.json next to this file.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
SYMBOL = "XAUUSD"
SPREAD = float(os.getenv("SPREAD", "0.30"))     # price units, charged once (round-trip)
TRAIN_FRAC = 0.60                                # WF: first 60% train, last 40% test
MIN_TEST_TRADES = 20                             # ignore param values with too-thin test

# Live baseline = config.SCALPER_PARAMS (+ brain time-stop / cooldown).
BASELINE = {
    "PERIOD": 20, "BB_MULT": 2.0, "RSI_PERIOD": 2, "RSI_LOW": 5.0, "RSI_HIGH": 95.0,
    "SL_ATR": 1.0, "ADX_MAX": 18.0, "H_START": 7, "H_END": 20,
    "TIME_STOP": 10,        # config.SCALPER_TIME_STOP_BARS (M1 bars/minutes)
    "COOLDOWN": 1,          # config.SCALPER_POST_CLOSE_COOLDOWN_SECS=60s ~= 1 M1 bar
}


def _wilder(s, period):
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


# ── indicator cache keyed by the params that change indicator arrays ──
_ICACHE = {}


def _indicators(m1, period, bb_mult, rsi_p):
    key = (period, round(bb_mult, 3), rsi_p)
    if key in _ICACHE:
        return _ICACHE[key]
    close, high, low = m1["close"], m1["high"], m1["low"]
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    lower = (mid - bb_mult * sd).values
    upper = (mid + bb_mult * sd).values
    midv = mid.values
    prev_c = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()],
                   axis=1).max(axis=1)
    atr_s = _wilder(tr, 14)
    atr = atr_s.values
    atr_ma = pd.Series(atr).rolling(20).mean().values
    d = close.diff()
    rsi = (100 - 100 / (1 + _wilder(d.clip(lower=0), rsi_p)
           / _wilder(-d.clip(upper=0), rsi_p).replace(0, np.nan))).values
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * _wilder(pd.Series(plus_dm), 14) / atr_s
    mdi = 100 * _wilder(pd.Series(minus_dm), 14) / atr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = _wilder(dx.fillna(0), 14).values
    out = (lower, upper, midv, atr, atr_ma, rsi, adx)
    _ICACHE[key] = out
    return out


def _load():
    for v in (f"raw_m1_{SYMBOL}.pkl", f"raw_m1_{SYMBOL.lower()}.pkl"):
        p = CACHE / v
        if p.exists():
            df = pd.read_pickle(p).copy()
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
            return df[["time", "open", "high", "low", "close"]].astype(
                {"open": float, "high": float, "low": float, "close": float})
    raise FileNotFoundError("raw_m1_xauusd.pkl not found")


def _prep(m1):
    """Attach non-indicator arrays used by every sim."""
    m1 = m1.copy()
    m1["_H"] = m1["high"].values
    m1["_L"] = m1["low"].values
    m1["_C"] = m1["close"].values
    m1["_hour"] = pd.to_datetime(m1["time"]).dt.hour.values
    return m1


def sim(m1, p, lo, hi):
    """Vectorized indicators + event-loop fills over bar window [lo, hi).
    Returns metrics dict. Identical gate/exit logic to m1_scalper.evaluate."""
    period = int(p["PERIOD"]); bb_mult = float(p["BB_MULT"]); rsi_p = int(p["RSI_PERIOD"])
    rsi_lo = float(p["RSI_LOW"]); rsi_hi = float(p["RSI_HIGH"])
    sl_atr = float(p["SL_ATR"]); adx_max = float(p["ADX_MAX"])
    h_start = int(p["H_START"]); h_end = int(p["H_END"])
    time_stop = int(p["TIME_STOP"]); cooldown = int(p["COOLDOWN"])

    lower, upper, midv, atr, atr_ma, rsi, adx = _indicators(m1, period, bb_mult, rsi_p)
    H, L, C = m1["_H"].values, m1["_L"].values, m1["_C"].values
    hour = m1["_hour"].values

    n = len(m1)
    hi = min(hi, n)
    trades = []
    open_until = lo - 1
    start = max(lo, period + 30, 40)
    for i in range(start, hi - 1):
        if i <= open_until:
            continue
        if not (h_start <= hour[i] < h_end):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(midv[i]):
            continue
        if adx[i] >= adx_max:                        # ranges only
            continue
        if not (np.isfinite(atr_ma[i]) and a > atr_ma[i]):   # ATR-expansion gate
            continue
        c = C[i]
        direction = None
        if c < lower[i] and rsi[i] < rsi_lo:
            direction = "LONG"
        elif c > upper[i] and rsi[i] > rsi_hi:
            direction = "SHORT"
        if direction is None:
            continue
        entry = c
        if direction == "LONG":
            sl = entry - sl_atr * a; tp = midv[i]
            if tp <= entry:
                continue
        else:
            sl = entry + sl_atr * a; tp = midv[i]
            if tp >= entry:
                continue
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        exit_px, exit_j = None, None
        for j in range(i + 1, min(i + 1 + time_stop, hi)):
            if direction == "LONG":
                if L[j] <= sl:
                    exit_px, exit_j = sl, j; break
                if H[j] >= tp:
                    exit_px, exit_j = tp, j; break
            else:
                if H[j] >= sl:
                    exit_px, exit_j = sl, j; break
                if L[j] <= tp:
                    exit_px, exit_j = tp, j; break
        if exit_px is None:
            exit_j = min(i + time_stop, hi - 1)
            exit_px = C[exit_j]
        gross = (exit_px - entry) if direction == "LONG" else (entry - exit_px)
        gross -= SPREAD                              # real spread, round-trip
        trades.append(gross / risk)
        open_until = exit_j + cooldown               # one concurrent + post-close cooldown

    if not trades:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "avg_R": 0.0, "total_R": 0.0, "dd_R": 0.0}
    R = np.array(trades)
    wins, losses = R[R > 0], R[R <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    eq = np.concatenate([[0.0], np.cumsum(R)])
    dd = float(abs((eq - np.maximum.accumulate(eq)).min()))
    return {"trades": int(len(R)), "wr": round(float((R > 0).mean()), 4),
            "pf": round(float(pf), 4), "avg_R": round(float(R.mean()), 4),
            "total_R": round(float(R.sum()), 2), "dd_R": round(dd, 1)}


def evaluate(m1, p, split):
    """Return full / train / test metrics for a param set."""
    n = len(m1)
    return {"full": sim(m1, p, 0, n),
            "train": sim(m1, p, 0, split),
            "test": sim(m1, p, split, n)}


# ── param grids (sensible ranges around the live baseline) ──
GRIDS = {
    "PERIOD":     [14, 20, 26, 34],
    "BB_MULT":    [1.8, 2.0, 2.3, 2.6],
    "RSI_PERIOD": [2, 3, 4],
    "RSI":        [(3, 97), (5, 95), (10, 90), (15, 85)],   # (LOW, HIGH) symmetric
    "SL_ATR":     [0.8, 1.0, 1.3, 1.6, 2.0],
    "ADX_MAX":    [14, 16, 18, 22, 25],
    "SESSION":    [(0, 24), (7, 20), (7, 17), (8, 18), (12, 20), (13, 22)],
    "TIME_STOP":  [8, 10, 15, 20, 30],
    "COOLDOWN":   [0, 1, 3, 5],
}


def _apply(base, name, val):
    p = dict(base)
    if name == "RSI":
        p["RSI_LOW"], p["RSI_HIGH"] = float(val[0]), float(val[1])
    elif name == "SESSION":
        p["H_START"], p["H_END"] = int(val[0]), int(val[1])
    else:
        p[name] = val
    return p


def _val_of(p, name):
    if name == "RSI":
        return (p["RSI_LOW"], p["RSI_HIGH"])
    if name == "SESSION":
        return (p["H_START"], p["H_END"])
    return p[name]


def main():
    global SPREAD
    m1 = _prep(_load())
    n = len(m1)
    split = int(n * TRAIN_FRAC)
    t0, tsplit, t1 = (str(m1["time"].iloc[0])[:16], str(m1["time"].iloc[split])[:16],
                      str(m1["time"].iloc[-1])[:16])
    print(f"M1 bars={n}  window {t0} -> {t1}  spread={SPREAD}")
    print(f"WF split @ {split} ({int(TRAIN_FRAC*100)}/{100-int(TRAIN_FRAC*100)})  train->{tsplit}")

    base = evaluate(m1, BASELINE, split)
    b_full, b_tr, b_te = base["full"], base["train"], base["test"]
    print(f"\nBASELINE  full: R={b_full['total_R']:>7} PF={b_full['pf']:<5} "
          f"WR={int(b_full['wr']*100)}% n={b_full['trades']} DD={b_full['dd_R']}")
    print(f"          train:R={b_tr['total_R']:>7} PF={b_tr['pf']:<5} n={b_tr['trades']}"
          f"   test:R={b_te['total_R']:>7} PF={b_te['pf']:<5} n={b_te['trades']}")

    per_param = []
    winners = {}
    print(f"\n{'param':>10} {'value':>10} | {'trainR':>7} {'trPF':>5} | "
          f"{'testR':>7} {'tePF':>5} {'teN':>4} | verdict")
    print("-" * 78)
    for name, vals in GRIDS.items():
        base_tr_R = b_tr["total_R"]; base_te_R = b_te["total_R"]
        rows = []
        for v in vals:
            p = _apply(BASELINE, name, v)
            ev = evaluate(m1, p, split)
            rows.append((v, ev))
        # best on TRAIN
        best = max(rows, key=lambda r: r[1]["train"]["total_R"])
        bv, bev = best
        tr, te = bev["train"], bev["test"]
        cur = _val_of(BASELINE, name)
        is_cur = (bv == cur)
        # SHIP rule: beats baseline on train AND holds >= baseline on test (neutral)
        ship = (not is_cur and tr["total_R"] > base_tr_R + 1e-9
                and te["total_R"] >= base_te_R and te["pf"] >= 1.0
                and te["trades"] >= MIN_TEST_TRADES)
        verdict = "SHIP" if ship else ("=baseline" if is_cur else "SHIP_NONE")
        if ship:
            winners[name] = bv
        per_param.append({
            "param": name, "baseline_value": cur, "best_train_value": bv,
            "grid": [str(v) for v in vals],
            "train": tr, "test": te, "full": bev["full"],
            "verdict": verdict,
        })
        print(f"{name:>10} {str(bv):>10} | {tr['total_R']:>7} {tr['pf']:>5} | "
              f"{te['total_R']:>7} {te['pf']:>5} {te['trades']:>4} | {verdict}")

    # ── combined: stack all shipped winners, WF-validate together ──
    combo = dict(BASELINE)
    for name, v in winners.items():
        combo = _apply(combo, name, v)
    cev = evaluate(m1, combo, split)
    c_full, c_tr, c_te = cev["full"], cev["train"], cev["test"]
    combined_ship = bool(winners) and (c_te["total_R"] >= b_te["total_R"]
                                       and c_full["total_R"] > b_full["total_R"]
                                       and c_te["pf"] >= 1.0)
    print("\n" + "=" * 78)
    print(f"WINNERS: {winners if winners else 'NONE'}")
    print(f"COMBINED  full: R={c_full['total_R']:>7} PF={c_full['pf']:<5} "
          f"WR={int(c_full['wr']*100)}% n={c_full['trades']} DD={c_full['dd_R']}")
    print(f"          train:R={c_tr['total_R']:>7} PF={c_tr['pf']:<5} n={c_tr['trades']}"
          f"   test:R={c_te['total_R']:>7} PF={c_te['pf']:<5} n={c_te['trades']}")
    print(f"COMBINED SHIP: {combined_ship}")

    # human-readable proposed params
    proposed = {k: combo[k] for k in BASELINE}

    # ── spread sensitivity: re-price baseline vs combined at other costs ──
    _orig = SPREAD
    sens = []
    for sp in (0.23, 0.30, 0.40):
        SPREAD = sp
        _ICACHE.clear()                      # indicators are price-only; fills change
        bb = sim(m1, BASELINE, 0, n)
        cc = sim(m1, combo, 0, n)
        sens.append({"spread": sp,
                     "baseline": {"total_R": bb["total_R"], "pf": bb["pf"]},
                     "combined": {"total_R": cc["total_R"], "pf": cc["pf"]}})
    SPREAD = _orig
    _ICACHE.clear()
    print("\nSPREAD SENSITIVITY (full-window R / PF):")
    for s in sens:
        print(f"  spread={s['spread']}: base R={s['baseline']['total_R']:>7} "
              f"PF={s['baseline']['pf']:<6}  combo R={s['combined']['total_R']:>7} "
              f"PF={s['combined']['pf']}")

    out = {
        "book": "SCALPER", "symbol": SYMBOL,
        "data": {"bars": n, "start": t0, "end": t1, "spread_price": SPREAD,
                 "wf_split_idx": split, "wf_train_frac": TRAIN_FRAC},
        "baseline": {"params": BASELINE, "full": b_full, "train": b_tr, "test": b_te},
        "per_param": per_param,
        "winners": {k: (list(v) if isinstance(v, tuple) else v) for k, v in winners.items()},
        "combined": {"params": proposed, "full": c_full, "train": c_tr, "test": c_te,
                     "ship": combined_ship},
        "spread_sensitivity": sens,
        "notes": (
            "WF 60/40 chronological. Per-param SHIP = beats baseline train AND >= "
            "baseline test (neutral) with test PF>=1.0 and test n>="
            f"{MIN_TEST_TRADES}. Spread {SPREAD}/side charged round-trip (live XAU "
            "M1 spread 30pts current / 23pts sample-median; 0.30 conservative). "
            "TP=rolling mean kept (design). Sim mirrors agent/m1_scalper.evaluate + "
            "brain time-stop/one-concurrent/post-close-cooldown. DEFERRED to user: "
            "no config.py/brain.py edits."),
    }
    outpath = Path(__file__).resolve().parent / "results_scalper_XAUUSD.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
