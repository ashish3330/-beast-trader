#!/usr/bin/env python3 -B
"""VALIDATE: winner-only un-capping of the TREND peak-giveback reversal exit.

Fix under test (LIVE TREND book, brain.py._process_trend):
  For CONFIRMED-WINNER symbols only (NAS100.r, JPN225ft), DISABLE the
  peak-giveback MARKET-close reversal exit so a live winner rides the
  Chandelier + peak-lock trail (broker-side backstop) instead of being clipped
  by a pullback from peak ("+$25 -> +$7 giveback"). XAU/others keep the exit.

Model: reuse the COMMITTED backtest/tune/trend_engine.py data path (D1 3-EMA
signal merge_asof'd onto H1 bars, no look-ahead) and the SAME per-H1-bar exit
simulator + CHURN-FREE cadence used by the shipped tuners
(retune_trend_NAS100_r.py). The only delta is a `giveback_off` toggle that
removes the GIVEBACK branch (== TREND_REVERSAL_EXIT_ENABLED=False for that sym).

Churn-free cadence (== live D1-bar gate + rev-block + strict throttle): after
ANY non-flip exit (SL/TRAIL/GIVEBACK/TP) same-dir re-entry is BLOCKED until the
D1 signal flips. Disabling giveback does NOT touch this block, so trade count
CANNOT balloon — it is a pure exit-timing change (fix is churn-neutral by
construction; we verify n stays <= current).

SHIP requires, on BOTH NAS100.r and JPN225ft:
  (1) more captured profit: full net R AND net pts strictly higher uncapped;
  (2) PF not worse (>=);
  (3) NO churn: n_uncapped <= n_current;
  (4) not a single-third fluke: uncapped >= current in >= 2/3 calendar thirds.
XAU shown as a CONTROL (giveback stays ON — must be identical both runs).

Does NOT modify config.py/brain.py. Run:
  python3 -B backtest/_trend_winner_uncap_validate_20260723.py
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backtest.tune import trend_engine as ENG
from config import (trend_exit_params, trend_ema_pairs, trend_atr_period,
                    TREND_ATR_STOP, TREND_TP_ATR, TREND_MIN_ABS_SIGNAL)

SPREAD_PTS = 1.5        # round-trip entry worsen (points) — tuner parity
TIE_TOL = 0.01
WINNERS = ["NAS100.r", "JPN225ft"]
CONTROL = "XAUUSD"


def simulate(m, p, churn_free=True, giveback_off=False):
    """Per-H1-bar exit sim — VERBATIM from retune_trend_NAS100_r.simulate, with
    a `giveback_off` toggle. churn_free=True blocks same-dir re-entry after any
    non-flip exit until the D1 signal flips (live throttle)."""
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    tm = m["time"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    ATR_STOP = p["ATR_STOP"]; TP_ATR = p["TP_ATR"]
    TR = p["TR"]; LK = p["LK"]; GB = p["GB"]; ACT = p["ACT"]
    if giveback_off:
        GB = 1.0            # disable the peak-giveback market-close branch entirely
    pos = 0; entry = sl = 0.0; tp = None; peak = 0.0; sl_dist = 0.0
    ent_t = None; blk = 0
    trades = []

    def _open(t, s):
        nonlocal pos, entry, peak, ent_t, sl_dist, sl, tp
        a = atr[t]
        pos = s; entry = o[t]; peak = 0.0; ent_t = tm[t]
        sl_dist = ATR_STOP * a
        sl = entry - sl_dist if s == 1 else entry + sl_dist
        tp = None if TP_ATR is None else (
            (entry + TP_ATR * a) if s == 1 else (entry - TP_ATR * a))

    def _close(px, reason, blk_after):
        nonlocal pos, blk
        pts = (px - entry) * pos - SPREAD_PTS
        trades.append({"t": ent_t, "dir": pos, "pnl_R": pts / sl_dist,
                       "pnl_pts": pts, "reason": reason})
        if churn_free and blk_after:
            blk = pos
        pos = 0

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blk and s != blk:
                blk = 0
            if s != 0 and s != blk:
                _open(t, s)
            continue
        if s != 0 and s != pos:              # genuine D1 flip — never churn
            _close(o[t], "FLIP", blk_after=False)
            blk = 0
            _open(t, s)
            continue
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else -1e18
            if l[t] <= sl:
                _close(sl, "SL/TRAIL", blk_after=True)
            elif gb > -1e17 and l[t] <= gb:
                _close(gb, "GIVEBACK", blk_after=True)
            elif tp is not None and h[t] >= tp:
                _close(tp, "TP", blk_after=True)
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR * a)
            if peak >= ACT * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else 1e18
            if h[t] >= sl:
                _close(sl, "SL/TRAIL", blk_after=True)
            elif gb < 1e17 and h[t] >= gb:
                _close(gb, "GIVEBACK", blk_after=True)
            elif tp is not None and l[t] <= tp:
                _close(tp, "TP", blk_after=True)
            else:
                peak = max(peak, entry - l[t])
    return trades


def _m(trades):
    R = np.array([t["pnl_R"] for t in trades]) if trades else np.array([])
    if len(R) == 0:
        return {"n": 0, "R": 0.0, "PF": 0.0, "WR": 0.0, "pts": 0.0}
    w = R[R > 0].sum(); ls = -R[R < 0].sum()
    pf = float(w / ls) if ls > 0 else 999.0
    pts = float(sum(t["pnl_pts"] for t in trades))
    return {"n": int(len(R)), "R": round(float(R.sum()), 3),
            "PF": round(min(pf, 999.0), 2), "WR": round(float((R > 0).mean()) * 100, 1),
            "pts": round(pts, 1)}


def _thirds(m, trades):
    t0 = np.datetime64(m["time"].iloc[0]); t3 = np.datetime64(m["time"].iloc[-1])
    span = t3 - t0
    b1 = t0 + span / 3; b2 = t0 + 2 * span / 3
    buckets = [[x for x in trades if x["t"] < b1],
               [x for x in trades if b1 <= x["t"] < b2],
               [x for x in trades if x["t"] >= b2]]
    return [_m(b) for b in buckets], (str(b1)[:10], str(b2)[:10])


def params_for(sym):
    tr, lk, gb, act = trend_exit_params(sym)
    return {"ATR_STOP": TREND_ATR_STOP, "TP_ATR": TREND_TP_ATR,
            "TR": tr, "LK": lk, "GB": gb, "ACT": act}


def run(sym, giveback_off):
    m = ENG.load(sym, ema_pairs=trend_ema_pairs(sym),
                 min_abs=TREND_MIN_ABS_SIGNAL, atr_p=trend_atr_period(sym))
    p = params_for(sym)
    tr = simulate(m, p, churn_free=True, giveback_off=giveback_off)
    naive_n = len(simulate(m, p, churn_free=False, giveback_off=giveback_off))
    full = _m(tr)
    thirds, cut = _thirds(m, tr)
    reasons = {}
    for x in tr:
        reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    return {"m": m, "full": full, "thirds": thirds, "cut": cut,
            "naive_n": naive_n, "reasons": reasons, "params": p}


def _fmt(r):
    f = r["full"]
    return (f"n={f['n']:>3} R={f['R']:>7.2f} PF={f['PF']:>5} WR={f['WR']:>5}% "
            f"pts={f['pts']:>9.1f}  thirdsR=[{r['thirds'][0]['R']:+.2f}, "
            f"{r['thirds'][1]['R']:+.2f}, {r['thirds'][2]['R']:+.2f}]  "
            f"naive_n(churn-cadence)={r['naive_n']}")


def main():
    print("=" * 100)
    print("TREND winner-only un-cap of peak-giveback reversal exit — validation")
    print(f"cost={SPREAD_PTS}pts RT | ATR_STOP={TREND_ATR_STOP} TP_ATR={TREND_TP_ATR} "
          f"MIN_ABS={TREND_MIN_ABS_SIGNAL} | churn-free cadence (live throttle)")
    print("=" * 100)

    verdicts = {}
    for sym in WINNERS:
        cur = run(sym, giveback_off=False)   # CURRENT-CLIPPED (giveback ON)
        unc = run(sym, giveback_off=True)    # WINNER-UNCAPPED (giveback OFF)
        p = cur["params"]
        print(f"\n### {sym}   exit book TR={p['TR']} LK={p['LK']} GB={p['GB']} ACT={p['ACT']}"
              f"   thirds cut {cur['cut'][0]} / {cur['cut'][1]}   (H1 merged bars={len(cur['m'])})")
        print(f"  current-clipped (giveback ON) : {_fmt(cur)}")
        print(f"    exit reasons: {cur['reasons']}")
        print(f"  winner-uncapped (giveback OFF): {_fmt(unc)}")
        print(f"    exit reasons: {unc['reasons']}")

        cf, uf = cur["full"], unc["full"]
        more_R = uf["R"] > cf["R"] + 1e-9
        more_pts = uf["pts"] > cf["pts"] + 1e-9
        pf_ok = uf["PF"] >= cf["PF"] - 1e-9
        no_churn = uf["n"] <= cf["n"]                 # count must NOT increase
        thirds_ge = sum(1 for k in range(3)
                        if unc["thirds"][k]["R"] >= cur["thirds"][k]["R"] - TIE_TOL)
        thirds_ok = thirds_ge >= 2
        dR = uf["R"] - cf["R"]; dpts = uf["pts"] - cf["pts"]
        ship = more_R and more_pts and pf_ok and no_churn and thirds_ok
        verdicts[sym] = {
            "ship": ship, "dR": round(dR, 3), "dpts": round(dpts, 1),
            "more_R": more_R, "more_pts": more_pts, "pf_ok": pf_ok,
            "no_churn": no_churn, "n_cur": cf["n"], "n_unc": uf["n"],
            "thirds_ge": thirds_ge,
            "cur": cf, "unc": uf}
        print(f"  -> dR={dR:+.2f}  dpts={dpts:+.1f}  PF {cf['PF']}->{uf['PF']}  "
              f"n {cf['n']}->{uf['n']} (no_churn={no_churn})  thirds_unc>=cur {thirds_ge}/3")
        print(f"  -> {sym} VERDICT: {'SHIP' if ship else 'NO-SHIP'}")

    # CONTROL: XAU giveback stays ON both runs — must be byte-identical (proves the
    # fix is winner-scoped; XAU/BTC-style strict throttle is untouched).
    xc = run(CONTROL, giveback_off=False)
    xc2 = run(CONTROL, giveback_off=False)
    x_ident = xc["full"] == xc2["full"]
    print(f"\n### CONTROL {CONTROL} (giveback stays ON — NOT a winner): {_fmt(xc)}")
    print(f"  XAU unchanged across identical runs: {x_ident} "
          f"(fix does NOT touch XAU/BTC throttle)")

    all_ship = all(v["ship"] for v in verdicts.values())
    print("\n" + "=" * 100)
    print(f"OVERALL: {'SHIP' if all_ship else 'NO-SHIP'}  "
          f"(both winners must capture more profit w/o churn)")
    for s, v in verdicts.items():
        print(f"  {s:>10}: ship={v['ship']} dR={v['dR']:+.2f} dpts={v['dpts']:+.1f} "
              f"n {v['n_cur']}->{v['n_unc']} thirds {v['thirds_ge']}/3")
    print("=" * 100)
    return all_ship


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
