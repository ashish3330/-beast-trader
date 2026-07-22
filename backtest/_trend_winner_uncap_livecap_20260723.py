#!/usr/bin/env python3 -B
"""EXPLORATORY: does disabling the peak-giveback help under the LIVE risk-capped
stop (where the giveback ACTIVATION scales to a much tighter sl_dist, as in
brain.py._process_trend)? The canonical fixed-3xATR engine says NO; this checks
whether the tighter live stop + earlier-arming giveback changes that.

CAVEAT (stated, not hidden): live sl_dist = min(3xATR, cap for TREND_MAX_RISK_PCT
at min-lot) depends on account equity/min-lot, which a price-only backtest cannot
pin. So we SWEEP the effective capped-stop multiple CAP_ATR as a proxy and, for
each, scale the giveback/lock activation to sl_dist exactly like brain.py:
  _act_thresh = min((ACT/3.0)*sl_dist, 0.5*sl_dist)   # 0.5 = PEAK_GIVEBACK_ACTIVATE_R
  _lock_thresh = (ACT/3.0)*sl_dist
This is the mechanism the task blames for clipping live winners. If disabling the
giveback does not beat keeping it across this sweep, the NO-SHIP is robust.
Does NOT modify config/brain. Run: python3 -B backtest/_trend_winner_uncap_livecap_20260723.py
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
                    TREND_TP_ATR, TREND_MIN_ABS_SIGNAL)

SPREAD_PTS = 1.5
WINNERS = ["NAS100.r", "JPN225ft"]
CAP_ATRS = [0.2, 0.35, 0.5, 1.0, 1.5, 2.0, 3.0]   # effective capped-stop multiples
REF_STOP = 3.0            # config TREND_ATR_STOP — activation-scaling reference
PEAK_GB_ACT_R = 0.5       # config PEAK_GIVEBACK_ACTIVATE_R ceiling


def simulate(m, tp_atr, tr, lk, gb, act, cap_atr, giveback_off):
    """Live-faithful exit sim: stop placed at cap_atr*ATR; giveback/lock arm at a
    fraction of that (tight) sl_dist per brain.py. churn-free cadence (block same-
    dir re-entry after any non-flip exit until D1 flip). tp measured off ORIGINAL
    ATR (TP is not risk-capped in live)."""
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    tm = m["time"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    if giveback_off:
        gb = 1.0
    pos = 0; entry = sl = 0.0; tp = None; peak = 0.0; sl_dist = 0.0
    act_gb = act_lk = 0.0; ent_t = None; blk = 0
    trades = []

    def _open(t, s):
        nonlocal pos, entry, peak, ent_t, sl_dist, sl, tp, act_gb, act_lk
        a = atr[t]
        pos = s; entry = o[t]; peak = 0.0; ent_t = tm[t]
        sl_dist = cap_atr * a
        # brain.py activation scaling to the (capped) sl_dist
        act_gb = min((act / REF_STOP) * sl_dist, PEAK_GB_ACT_R * sl_dist)
        act_lk = (act / REF_STOP) * sl_dist
        sl = entry - sl_dist if s == 1 else entry + sl_dist
        tp = None if tp_atr is None else (
            (entry + tp_atr * a) if s == 1 else (entry - tp_atr * a))

    def _close(px, reason, blk_after):
        nonlocal pos, blk
        pts = (px - entry) * pos - SPREAD_PTS
        trades.append({"t": ent_t, "dir": pos, "pnl_R": pts / sl_dist,
                       "pnl_pts": pts, "reason": reason})
        if blk_after:
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
        if s != 0 and s != pos:
            _close(o[t], "FLIP", blk_after=False); blk = 0; _open(t, s); continue
        if pos == 1:
            chand = hh[t] - tr * a
            sl = max(sl, chand)
            if peak >= act_lk:
                sl = max(sl, entry + lk * peak)
            gbp = entry + peak * (1.0 - gb) if (gb < 1.0 and peak >= act_gb) else -1e18
            if l[t] <= sl:
                _close(sl, "SL/TRAIL", True)
            elif gbp > -1e17 and l[t] <= gbp:
                _close(gbp, "GIVEBACK", True)
            elif tp is not None and h[t] >= tp:
                _close(tp, "TP", True)
            else:
                peak = max(peak, h[t] - entry)
        else:
            chand = ll[t] + tr * a
            sl = min(sl, chand)
            if peak >= act_lk:
                sl = min(sl, entry - lk * peak)
            gbp = entry - peak * (1.0 - gb) if (gb < 1.0 and peak >= act_gb) else 1e18
            if h[t] >= sl:
                _close(sl, "SL/TRAIL", True)
            elif gbp < 1e17 and h[t] >= gbp:
                _close(gbp, "GIVEBACK", True)
            elif tp is not None and l[t] <= tp:
                _close(tp, "TP", True)
            else:
                peak = max(peak, entry - l[t])
    return trades


def _agg(trades):
    R = np.array([t["pnl_R"] for t in trades]) if trades else np.array([])
    if len(R) == 0:
        return 0, 0.0, 0.0, 0.0
    w = R[R > 0].sum(); ls = -R[R < 0].sum()
    pf = float(w / ls) if ls > 0 else 999.0
    pts = float(sum(t["pnl_pts"] for t in trades))
    return len(R), round(float(R.sum()), 2), round(min(pf, 999.0), 2), round(pts, 1)


def main():
    print("=" * 104)
    print("LIVE-CAP sensitivity: peak-giveback ON vs OFF across effective capped-stop multiples")
    print(f"cost={SPREAD_PTS}pts | activation scaled to sl_dist per brain.py "
          f"(min((ACT/3)*sl, 0.5*sl)) | churn-free cadence")
    print("=" * 104)
    any_material = False
    for sym in WINNERS:
        m = ENG.load(sym, ema_pairs=trend_ema_pairs(sym),
                     min_abs=TREND_MIN_ABS_SIGNAL, atr_p=trend_atr_period(sym))
        tr, lk, gb, act = trend_exit_params(sym)
        print(f"\n### {sym}  TR={tr} LK={lk} GB={gb} ACT={act}  (H1 bars={len(m)})")
        print(f"{'CAP_ATR':>8} | {'ON: n/R/PF/pts':>34} | {'OFF: n/R/PF/pts':>34} | "
              f"{'dR':>7} {'dpts':>9} {'nGB':>4}")
        for cap in CAP_ATRS:
            on = simulate(m, TREND_TP_ATR, tr, lk, gb, act, cap, giveback_off=False)
            off = simulate(m, TREND_TP_ATR, tr, lk, gb, act, cap, giveback_off=True)
            n_on, R_on, pf_on, pts_on = _agg(on)
            n_off, R_off, pf_off, pts_off = _agg(off)
            n_gb = sum(1 for x in on if x["reason"] == "GIVEBACK")
            dR = R_off - R_on; dpts = pts_off - pts_on
            flag = ""
            if n_gb > 0 and dR > 0.05 and pts_off > pts_on and n_off <= n_on:
                flag = " <== giveback-off helps"; any_material = True
            print(f"{cap:>8} | {n_on:>3}/{R_on:>7.2f}/{pf_on:>5}/{pts_on:>10.1f} | "
                  f"{n_off:>3}/{R_off:>7.2f}/{pf_off:>5}/{pts_off:>10.1f} | "
                  f"{dR:>+7.2f} {dpts:>+9.1f} {n_gb:>4}{flag}")
    print("\n" + "=" * 104)
    print(f"Any capped-stop regime where giveback-OFF materially helps w/o churn: {any_material}")
    print("=" * 104)

    # ── THIRDS robustness at representative LIVE caps (small-account NAS ~0.25-0.5xATR) ──
    print("\nCALENDAR-THIRDS robustness (net R per third) at representative live caps")
    for sym in WINNERS:
        m = ENG.load(sym, ema_pairs=trend_ema_pairs(sym),
                     min_abs=TREND_MIN_ABS_SIGNAL, atr_p=trend_atr_period(sym))
        tr, lk, gb, act = trend_exit_params(sym)
        t0 = np.datetime64(m["time"].iloc[0]); t3 = np.datetime64(m["time"].iloc[-1])
        b1 = t0 + (t3 - t0) / 3; b2 = t0 + 2 * (t3 - t0) / 3
        def thirds_R(trades):
            buckets = [[x for x in trades if x["t"] < b1],
                       [x for x in trades if b1 <= x["t"] < b2],
                       [x for x in trades if x["t"] >= b2]]
            return [round(float(sum(x["pnl_R"] for x in b)), 2) for b in buckets]
        print(f"\n### {sym}  thirds cut {str(b1)[:10]} / {str(b2)[:10]}")
        for cap in (0.35, 0.5):
            on = simulate(m, TREND_TP_ATR, tr, lk, gb, act, cap, giveback_off=False)
            off = simulate(m, TREND_TP_ATR, tr, lk, gb, act, cap, giveback_off=True)
            ron, roff = thirds_R(on), thirds_R(off)
            ge = sum(1 for k in range(3) if roff[k] >= ron[k] - 0.01)
            print(f"  cap={cap}: ON thirdsR={ron}  OFF thirdsR={roff}  "
                  f"OFF>=ON in {ge}/3")


if __name__ == "__main__":
    main()
