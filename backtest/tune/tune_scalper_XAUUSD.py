#!/usr/bin/env python3 -B
"""Hard-tune the LIVE M1 SCALPER book (6th book) — XAUUSD only.

Self-contained. Mirrors agent/m1_scalper.evaluate() + the brain's execution
(SL=SL_ATR*ATR, TP=rolling mean, M1 time-stop, one concurrent, post-close
cooldown) so the backtest matches live. REAL spread charged both sides.

Method (v2 — hardened 2026-07-15)
---------------------------------
  * Data: raw_m1_xauusd_tune100k.pkl (100k M1, tune-dedicated file — the live
    sync daemon truncates raw_m1_xauusd.pkl to 500 bars; loader REFUSES <50k).
  * Objective: total R (primary) + PF + WR.
  * WF 60/40 chronological: TRAIN = first 60% of bars, TEST = last 40%.
  * 3-THIRDS cross-window guard: candidate must be >= baseline total_R in at
    least 2 of 3 chronological thirds (kills single-regime artifacts).
  * SPREAD-0.40 fragility guard: candidate must keep total_R > 0 and PF >= 1.0
    on the TEST window repriced at 0.40 spread (scalpers are cost-fragile).
  * One-at-a-time per-param sweep -> verdict, then surviving winners combined
    and re-validated through ALL the same guards.
  * Grid-edge winners flagged (extend grid before trusting).

Cost: XAU M1 spread ~= 0.30 price charged round-trip (live symbol spread 30pts
current / 23pts 100k-sample median; 0.30 conservative). SPREAD env overrides.

Run:  python3 -B backtest/tune/tune_scalper_XAUUSD.py
Emits a human table + writes results_scalper_XAUUSD.json next to this file and
the summary schema to allstrat/results_scalper.json.
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
STRESS_SPREAD = 0.40                             # cost-fragility guard level
TRAIN_FRAC = 0.60                                # WF: first 60% train, last 40% test
MIN_TEST_TRADES = 20                             # ignore param values with too-thin test
MIN_BARS = 50_000                                # refuse truncated daemon cache

# Live baseline = config.SCALPER_PARAMS (+ brain time-stop / cooldown).
# 2026-08-04: synced to ACTUAL live params (was stale: RSI 5/95, BB 2.0,
# TIME_STOP 10). Live ships RSI 10/90, BB_MULT 1.8, SL_ATR 0.8, TIME_STOP 30
# (SCALPER_TIME_STOP_BARS). Grid still spans the old values to re-confirm.
BASELINE = {
    "PERIOD": 20, "BB_MULT": 1.8, "RSI_PERIOD": 2, "RSI_LOW": 10.0, "RSI_HIGH": 90.0,
    "SL_ATR": 0.8, "ADX_MAX": 18.0, "H_START": 7, "H_END": 20,
    "TIME_STOP": 30,        # config.SCALPER_TIME_STOP_BARS (M1 bars/minutes)
    "COOLDOWN": 1,          # config.SCALPER_POST_CLOSE_COOLDOWN_SECS=60s ~= 1 M1 bar
}

# 2026-08-04 toxic-hour filter, mirrors config.SCALPER_PARAMS["HOUR_BLACKLIST"]
# so backtest == live. Bar-TZ hours skipped in sim() below (US session-open +
# news spikes that break M1 mean-reversion; thin late-US).
HOUR_BLACKLIST = frozenset({13, 15, 18, 19})


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
    for v in (f"raw_m1_{SYMBOL.lower()}_tune100k.pkl",
              f"raw_m1_{SYMBOL}.pkl", f"raw_m1_{SYMBOL.lower()}.pkl"):
        p = CACHE / v
        if p.exists():
            df = pd.read_pickle(p).copy()
            if len(df) < MIN_BARS:
                print(f"  SKIP {v}: only {len(df)} bars (<{MIN_BARS}) — daemon-truncated")
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
            return df[["time", "open", "high", "low", "close"]].astype(
                {"open": float, "high": float, "low": float, "close": float})
    raise FileNotFoundError(
        "no M1 cache with >=50k bars — run the tune fetch (raw_m1_xauusd_tune100k.pkl)")


def _prep(m1):
    """Attach non-indicator arrays used by every sim."""
    m1 = m1.copy()
    m1["_H"] = m1["high"].values
    m1["_L"] = m1["low"].values
    m1["_C"] = m1["close"].values
    m1["_hour"] = pd.to_datetime(m1["time"]).dt.hour.values
    return m1


def sim(m1, p, lo, hi, spread=None):
    """Vectorized indicators + event-loop fills over bar window [lo, hi).
    Returns metrics dict. Identical gate/exit logic to m1_scalper.evaluate."""
    spread = SPREAD if spread is None else spread
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
        if hour[i] in HOUR_BLACKLIST:                # toxic-hour filter (== live)
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
        gross -= spread                              # real spread, round-trip
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


def thirds(m1, p):
    """Metrics on the 3 chronological thirds (cross-window robustness)."""
    n = len(m1)
    cuts = [0, n // 3, 2 * n // 3, n]
    return [sim(m1, p, cuts[k], cuts[k + 1]) for k in range(3)]


def guards(m1, p, split, base_thirds, base_te_R):
    """Run the two robustness guards on a candidate param set.
    Returns (thirds_ok, thirds_wins, stress, stress_ok, cand_thirds)."""
    cand_thirds = thirds(m1, p)
    wins = sum(1 for c, b in zip(cand_thirds, base_thirds)
               if c["total_R"] >= b["total_R"])
    thirds_ok = wins >= 2
    stress = sim(m1, p, split, len(m1), spread=STRESS_SPREAD)   # TEST window @ 0.40
    stress_ok = stress["total_R"] > 0 and stress["pf"] >= 1.0
    return thirds_ok, wins, stress, stress_ok, cand_thirds


# ── param grids (sensible ranges around the live baseline) ──
# TIME_STOP/COOLDOWN grids EXTENDED past the prior run's edge-winners (30 / 5).
GRIDS = {
    "PERIOD":     [14, 20, 26, 34],
    "BB_MULT":    [1.8, 2.0, 2.3, 2.6],
    "RSI_PERIOD": [2, 3, 4],
    "RSI":        [(3, 97), (5, 95), (10, 90), (15, 85)],   # (LOW, HIGH) symmetric
    "SL_ATR":     [0.6, 0.7, 0.8, 1.0, 1.3, 1.6],
    "ADX_MAX":    [14, 16, 18, 22, 25],
    "SESSION":    [(0, 24), (7, 20), (7, 17), (8, 18), (12, 20), (13, 22)],
    "TIME_STOP":  [8, 10, 15, 20, 30, 45, 60],
    "COOLDOWN":   [0, 1, 3, 5, 8, 12],
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


def _jsonable(v):
    return list(v) if isinstance(v, tuple) else v


def main():
    m1 = _prep(_load())
    n = len(m1)
    split = int(n * TRAIN_FRAC)
    t0, tsplit, t1 = (str(m1["time"].iloc[0])[:16], str(m1["time"].iloc[split])[:16],
                      str(m1["time"].iloc[-1])[:16])
    print(f"M1 bars={n}  window {t0} -> {t1}  spread={SPREAD}")
    print(f"WF split @ {split} ({int(TRAIN_FRAC*100)}/{100-int(TRAIN_FRAC*100)})  train->{tsplit}")

    base = evaluate(m1, BASELINE, split)
    b_full, b_tr, b_te = base["full"], base["train"], base["test"]
    base_thirds = thirds(m1, BASELINE)
    base_stress = sim(m1, BASELINE, split, n, spread=STRESS_SPREAD)
    print(f"\nBASELINE  full: R={b_full['total_R']:>7} PF={b_full['pf']:<5} "
          f"WR={int(b_full['wr']*100)}% n={b_full['trades']} DD={b_full['dd_R']}")
    print(f"          train:R={b_tr['total_R']:>7} PF={b_tr['pf']:<5} n={b_tr['trades']}"
          f"   test:R={b_te['total_R']:>7} PF={b_te['pf']:<5} n={b_te['trades']}")
    print(f"          thirds R: {[t['total_R'] for t in base_thirds]}"
          f"  test@0.40: R={base_stress['total_R']} PF={base_stress['pf']}")

    # Guard chain per candidate value:
    #   WF        : train improves AND test >= baseline (PF>=1, n>=MIN_TEST_TRADES)
    #   THIRDS    : >= baseline R in >=2/3 chronological thirds
    #   SPREAD040 : test window repriced @0.40 stays R>0, PF>=1
    #   EFFECT    : full-window R >= 1.10x baseline AND full PF >= baseline PF
    #               (kills flat-surface noise "winners")
    # Selection: walk grid values best-train-first; first value passing ALL
    # guards ships (plateau fallback — a failing grid-edge best-on-train no
    # longer masks a robust interior value).
    base_tr_R = b_tr["total_R"]; base_te_R = b_te["total_R"]
    base_full_R = b_full["total_R"]; base_full_pf = b_full["pf"]

    def _judge(p):
        ev = evaluate(m1, p, split)
        tr, te, fu = ev["train"], ev["test"], ev["full"]
        if not (tr["total_R"] > base_tr_R + 1e-9
                and te["total_R"] >= base_te_R and te["pf"] >= 1.0
                and te["trades"] >= MIN_TEST_TRADES):
            return "REJECT_WF", ev, None, None
        th_ok, th_w, st, st_ok, _ = guards(m1, p, split, base_thirds, base_te_R)
        if not th_ok:
            return "REJECT_THIRDS", ev, th_w, st
        if not st_ok:
            return "REJECT_SPREAD040", ev, th_w, st
        if not (fu["total_R"] >= 1.10 * base_full_R and fu["pf"] >= base_full_pf):
            return "REJECT_EFFECT", ev, th_w, st
        return "SHIP", ev, th_w, st

    per_param = []
    winners = {}
    print(f"\n{'param':>10} {'value':>10} | {'trainR':>7} {'trPF':>5} | "
          f"{'testR':>7} {'tePF':>5} {'teN':>4} | {'3rds':>4} {'@.40':>6} | verdict")
    print("-" * 92)
    for name, vals in GRIDS.items():
        rows = []
        for v in vals:
            p = _apply(BASELINE, name, v)
            ev = evaluate(m1, p, split)
            rows.append((v, ev))
        rows.sort(key=lambda r: r[1]["train"]["total_R"], reverse=True)
        cur = _val_of(BASELINE, name)
        top_v = rows[0][0]                      # best-on-train (reported)
        pick, verdict, ev, th_w, st = None, None, None, None, None
        rejects = []
        for v, _ev in rows:
            if v == cur:                        # walked down to baseline: keep
                verdict, ev = "KEEP(baseline)", _ev
                pick = cur
                break
            vd, _ev2, _th, _st = _judge(_apply(BASELINE, name, v))
            if vd == "SHIP":
                pick, ev, th_w, st = v, _ev2, _th, _st
                edge = (v == vals[0] or v == vals[-1])
                plateau = (v != top_v)
                verdict = ("SHIP" + ("_PLATEAU" if plateau else "")
                           + ("(grid-edge!)" if edge else ""))
                winners[name] = v
                break
            rejects.append(f"{v}:{vd}")
        if verdict is None:                     # nothing shipped, baseline not in grid
            verdict, ev, pick = "SHIP_NONE", rows[0][1], rows[0][0]
        tr, te = ev["train"], ev["test"]
        per_param.append({
            "param": name, "baseline_value": _jsonable(cur),
            "best_train_value": _jsonable(top_v),
            "picked_value": _jsonable(pick),
            "grid": [str(v) for v in vals],
            "train": tr, "test": te, "full": ev["full"],
            "thirds_wins": th_w,
            "stress_test_040": st,
            "rejected_above_pick": rejects,
            "verdict": verdict,
        })
        print(f"{name:>10} {str(pick):>10} | {tr['total_R']:>7} {tr['pf']:>5} | "
              f"{te['total_R']:>7} {te['pf']:>5} {te['trades']:>4} | "
              f"{('-' if th_w is None else str(th_w)+'/3'):>4} "
              f"{('-' if st is None else str(st['total_R'])):>6} | {verdict}"
              + (f"  [rejected: {', '.join(rejects)}]" if rejects else ""))

    # ── combined: winners can interact (isolated-vs-combined gap), so validate
    # every non-empty SUBSET of shipped winners through the full guard chain and
    # keep the passing subset with the highest full-window R. ──
    from itertools import combinations

    def _combo_eval(sub):
        p = dict(BASELINE)
        for name in sub:
            p = _apply(p, name, winners[name])
        ev = evaluate(m1, p, split)
        fu, te = ev["full"], ev["test"]
        th_ok, th_w, st, st_ok, cth = guards(m1, p, split, base_thirds,
                                             b_te["total_R"])
        ok = (te["total_R"] >= b_te["total_R"] and te["pf"] >= 1.0
              and fu["total_R"] > b_full["total_R"] and fu["pf"] >= b_full["pf"]
              and th_ok and st_ok)
        return p, ev, th_w, st, cth, ok

    subset_report = []
    best_sub, best_pack = None, None
    names = list(winners)
    for r in range(len(names), 0, -1):
        for sub in combinations(names, r):
            p, ev, th_w, st, cth, ok = _combo_eval(sub)
            fu = ev["full"]
            subset_report.append({"subset": list(sub), "full_R": fu["total_R"],
                                  "full_pf": fu["pf"], "test_R": ev["test"]["total_R"],
                                  "thirds_wins": th_w, "test040_R": st["total_R"],
                                  "pass": ok})
            print(f"  subset {str(sub):<40} full R={fu['total_R']:>7} PF={fu['pf']:<6} "
                  f"test R={ev['test']['total_R']:>7} 3rds={th_w}/3 "
                  f"@.40={st['total_R']:>6} {'PASS' if ok else 'fail'}")
            if ok and (best_pack is None or fu["total_R"] > best_pack[1]["full"]["total_R"]):
                best_sub, best_pack = sub, (p, ev, th_w, st, cth)

    if best_pack is not None:
        combo, cev, c_th_w, c_st, c_thirds = best_pack
        combined_ship = True
        winners = {k: winners[k] for k in best_sub}
    else:
        combo = dict(BASELINE)
        for name, v in winners.items():
            combo = _apply(combo, name, v)
        cev = evaluate(m1, combo, split)
        _, c_th_w, c_st, _, c_thirds = guards(m1, combo, split, base_thirds,
                                              b_te["total_R"])
        combined_ship = False
    c_full, c_tr, c_te = cev["full"], cev["train"], cev["test"]
    print("\n" + "=" * 92)
    print(f"WINNERS: {winners if winners else 'NONE'}")
    print(f"COMBINED  full: R={c_full['total_R']:>7} PF={c_full['pf']:<5} "
          f"WR={int(c_full['wr']*100)}% n={c_full['trades']} DD={c_full['dd_R']}")
    print(f"          train:R={c_tr['total_R']:>7} PF={c_tr['pf']:<5} n={c_tr['trades']}"
          f"   test:R={c_te['total_R']:>7} PF={c_te['pf']:<5} n={c_te['trades']}")
    print(f"          thirds R: {[t['total_R'] for t in c_thirds]} ({c_th_w}/3)"
          f"  test@0.40: R={c_st['total_R']} PF={c_st['pf']}")
    print(f"COMBINED SHIP: {combined_ship}")

    # human-readable proposed params
    proposed = {k: combo[k] for k in BASELINE}

    # ── spread sensitivity: re-price baseline vs combined at other costs ──
    sens = []
    for sp in (0.23, 0.30, 0.40, 0.50):
        bb = sim(m1, BASELINE, 0, n, spread=sp)
        cc = sim(m1, combo, 0, n, spread=sp)
        sens.append({"spread": sp,
                     "baseline": {"total_R": bb["total_R"], "pf": bb["pf"]},
                     "combined": {"total_R": cc["total_R"], "pf": cc["pf"]}})
    print("\nSPREAD SENSITIVITY (full-window R / PF):")
    for s in sens:
        print(f"  spread={s['spread']}: base R={s['baseline']['total_R']:>7} "
              f"PF={s['baseline']['pf']:<6}  combo R={s['combined']['total_R']:>7} "
              f"PF={s['combined']['pf']}")

    notes = (
        "WF 60/40 chronological + 3-thirds cross-window (winner must be >= "
        "baseline R in >=2/3 thirds) + spread-0.40 fragility guard on the TEST "
        "window (scalpers are cost-fragile; anything that dies at 0.40 is "
        "rejected). Per-param SHIP additionally requires beats-baseline train, "
        f">=baseline test R, test PF>=1.0, test n>={MIN_TEST_TRADES}. Spread "
        f"{SPREAD} charged round-trip (live XAU M1 spread 30pts current / 23pts "
        "100k-sample median). BASELINE already includes the shipped SL_ATR 0.8. "
        "TP=rolling mean kept (design). Sim mirrors agent/m1_scalper.evaluate + "
        "brain time-stop/one-concurrent/post-close-cooldown. Data: tune-dedicated "
        "raw_m1_xauusd_tune100k.pkl (daemon truncates the shared file to 500 "
        "bars). DEFERRED to user: no config.py/brain.py edits.")

    out = {
        "book": "SCALPER", "symbol": SYMBOL,
        "data": {"bars": n, "start": t0, "end": t1, "spread_price": SPREAD,
                 "wf_split_idx": split, "wf_train_frac": TRAIN_FRAC},
        "baseline": {"params": BASELINE, "full": b_full, "train": b_tr, "test": b_te,
                     "thirds": base_thirds, "test_at_040": base_stress},
        "per_param": per_param,
        "winners": {k: _jsonable(v) for k, v in winners.items()},
        "combined": {"params": proposed, "full": c_full, "train": c_tr, "test": c_te,
                     "thirds": c_thirds, "thirds_wins": c_th_w, "test_at_040": c_st,
                     "ship": combined_ship, "subset_search": subset_report},
        "spread_sensitivity": sens,
        "notes": notes,
    }
    outpath = Path(__file__).resolve().parent / "results_scalper_XAUUSD.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")

    # ── allstrat summary (deliverable schema) ──
    summary = {
        "strategy": "scalper", "symbol": SYMBOL,
        "baseline": {"params": BASELINE, "full": b_full, "train": b_tr, "test": b_te,
                     "thirds_R": [t["total_R"] for t in base_thirds],
                     "test_at_040": {"total_R": base_stress["total_R"],
                                     "pf": base_stress["pf"]}},
        "per_param": [{"name": r["param"], "current": r["baseline_value"],
                       "best": r["picked_value"], "verdict": r["verdict"]}
                      for r in per_param],
        "combined": {"params": proposed, "ship": combined_ship,
                     "shipped_winners": {k: _jsonable(v) for k, v in winners.items()},
                     "full": c_full, "test": c_te,
                     "thirds_R": [t["total_R"] for t in c_thirds],
                     "test_at_040": {"total_R": c_st["total_R"], "pf": c_st["pf"]},
                     "subset_search": subset_report},
        "spread_sensitivity": sens,
        "notes": notes,
    }
    allstrat = Path(__file__).resolve().parent / "allstrat"
    allstrat.mkdir(exist_ok=True)
    spath = allstrat / "results_scalper.json"
    spath.write_text(json.dumps(summary, indent=2))
    print(f"wrote {spath}")


if __name__ == "__main__":
    main()
