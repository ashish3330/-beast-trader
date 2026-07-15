#!/usr/bin/env python3 -B
"""Re-entry cooldown tune, ALL 5 TREND symbols (2026-07-15, inline).

Tests TREND_REENTRY_BLOCK_HOURS (live=2h). After ANY non-flip exit, block
same-direction re-entry for `block_bars` H1 bars (1 bar = 1h). This is the
honest lever for the "re-enter again and again" chase-churn. Sweep + WF 60/40
+ 3-thirds. Live-faithful giveback/lock/trail from trend_engine. Does NOT
touch config.py / brain.py.
"""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import trend_engine as te

DFLT = [(16, 64), (32, 128), (64, 256)]
WIDE = [(16, 96), (32, 160), (64, 256)]
SYMS = {
    "XAUUSD":   dict(EMA=DFLT, TR=2.5, LK=0.5, GB=0.30, ACT=0.5, cost=7.9e-05),
    "BTCUSD":   dict(EMA=DFLT, TR=3.0, LK=0.5, GB=0.30, ACT=0.5, cost=5.3e-04),
    "ETHUSD":   dict(EMA=WIDE, TR=2.5, LK=0.5, GB=0.35, ACT=0.3, cost=2.75e-03),
    "JPN225ft": dict(EMA=DFLT, TR=3.0, LK=0.6, GB=0.35, ACT=0.4, cost=4.12e-04),
    "NAS100.r": dict(EMA=DFLT, TR=2.5, LK=0.6, GB=0.35, ACT=0.5, cost=3.0e-04),
}
MIN_ABS, ATR_P, ATR_STOP, TP = 0.34, 20, 3.0, 6.0
BARS = [0, 2, 4, 6, 8, 12, 24]     # 2 = current (TREND_REENTRY_BLOCK_HOURS 2.0)


def simulate_cd(m, TR, LK, GB, ACT, block_bars, cost):
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    pos = 0; entry = sl = peak = 0.0; tp = None; e_atr = 0.0
    blocked = 0; blocked_until = -1; trades = []

    def _close(px):
        nonlocal pos
        ret = ((px - entry) / entry) * pos - cost
        ru = (ATR_STOP * e_atr) / entry if entry > 0 and e_atr > 0 else 0.0
        trades.append(ret / ru if ru > 0 else 0.0); pos = 0

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blocked and (s != blocked or t >= blocked_until):
                blocked = 0
            if s != 0 and s != blocked:
                pos = s; entry = o[t]; peak = 0.0; e_atr = a
                sl = entry - ATR_STOP * a if s == 1 else entry + ATR_STOP * a
                tp = (entry + TP * a) if s == 1 else (entry - TP * a)
            continue
        if s != 0 and s != pos:            # flip: immediate reverse, no cooldown
            _close(o[t]); blocked = 0
            pos = s; entry = o[t]; peak = 0.0; e_atr = a
            sl = entry - ATR_STOP * a if pos == 1 else entry + ATR_STOP * a
            tp = (entry + TP * a) if pos == 1 else (entry - TP * a)
            continue
        armed = peak >= ACT * a
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if armed:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and armed) else -1e18
            ex = None
            if l[t] <= sl:
                ex = sl
            elif gb > -1e17 and l[t] <= gb:
                ex = gb
            elif h[t] >= tp:
                ex = tp
            if ex is not None:
                d = pos; _close(ex)
                if block_bars > 0:
                    blocked = d; blocked_until = t + block_bars
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR * a)
            if armed:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and armed) else 1e18
            ex = None
            if h[t] >= sl:
                ex = sl
            elif gb < 1e17 and h[t] >= gb:
                ex = gb
            elif l[t] <= tp:
                ex = tp
            if ex is not None:
                d = pos; _close(ex)
                if block_bars > 0:
                    blocked = d; blocked_until = t + block_bars
            else:
                peak = max(peak, entry - l[t])
    return trades


def stat(rs):
    n = len(rs); tot = sum(rs)
    w = [r for r in rs if r > 0]; ls = [r for r in rs if r < 0]
    pf = (sum(w) / -sum(ls)) if ls else float("inf")
    return n, tot, pf


def main():
    summ = {}
    for sym, c in SYMS.items():
        try:
            m = te.load(sym, ema_pairs=c["EMA"], min_abs=MIN_ABS, atr_p=ATR_P)
        except Exception as e:
            print(f"\n### {sym}: LOAD FAILED ({e})"); continue
        print(f"\n### {sym}  ({len(m)} bars)")
        print(f"{'hrs':>4} {'n':>6} {'totR':>9} {'PF':>7} {'trainR':>8} {'testR':>8}  thirds")
        rows = {}
        for b in BARS:
            tr = simulate_cd(m, c["TR"], c["LK"], c["GB"], c["ACT"], b, c["cost"])
            n, tot, pf = stat(tr)
            k = int(len(tr) * 0.6)
            trn = stat(tr[:k])[1]; tst = stat(tr[k:])[1]
            n3 = len(tr) // 3
            th = [round(stat(tr[i*n3:(i+1)*n3] if i < 2 else tr[2*n3:])[1], 1) for i in range(3)]
            rows[b] = (n, tot, pf, trn, tst, th)
            tag = " <cur(2h)" if b == 2 else ""
            print(f"{b:>4} {n:>6} {tot:>+9.2f} {pf:>7.2f} {trn:>+8.2f} {tst:>+8.2f}  {th}{tag}")
        base = rows[2]
        cands = []
        for b, r in rows.items():
            if b == 2:
                continue
            th_ok = sum(1 for i in range(3) if r[5][i] >= base[5][i] - 0.05) >= 2
            if r[1] >= base[1] - 1e-9 and r[4] >= base[4] - 1e-9 and th_ok:
                cands.append((r[1], b))
        pick = max(cands)[1] if cands else 2
        summ[sym] = (pick, rows[pick][1]-base[1], rows[pick][4]-base[4], rows[pick][0], base[0])
        v = "KEEP 2h" if pick == 2 else f"SHIP {pick}h"
        print(f"  => {v}   (n {base[0]}->{rows[pick][0]}, fulldR {rows[pick][1]-base[1]:+.2f}, testdR {rows[pick][4]-base[4]:+.2f})")
    print("\n======== SUMMARY (robust pick per symbol) ========")
    for sym, (pick, dR, dT, npick, nbase) in summ.items():
        print(f"  {sym:9} -> {pick}h  (trades {nbase}->{npick}, fulldR {dR:+.2f}, testdR {dT:+.2f})")


if __name__ == "__main__":
    main()
