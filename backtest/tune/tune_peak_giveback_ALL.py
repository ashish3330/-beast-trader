#!/usr/bin/env python3 -B
"""Peak-giveback ARM-R tune, ALL 5 TREND symbols (2026-07-15, inline).

Hypothesis: PEAK_GIVEBACK_ACTIVATE_R 0.5 -> 2.5 (arm giveback later so winners
run). Live-faithful sim (2h re-entry cooldown), giveback disabled until
peak_R >= arm_R. Strict: WF 60/40 + 3-thirds stability + winner-size check.
Does NOT touch config.py / brain.py.
"""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import trend_engine as te

DFLT = [(16, 64), (32, 128), (64, 256)]
WIDE = [(16, 96), (32, 160), (64, 256)]
# per-symbol baseline (config.py TREND_EXIT_PER_SYMBOL + defaults) + RT cost
SYMS = {
    "XAUUSD":   dict(EMA=DFLT, TR=2.5, LK=0.5, GB=0.30, ACT=0.5, cost=7.9e-05),
    "BTCUSD":   dict(EMA=DFLT, TR=3.0, LK=0.5, GB=0.30, ACT=0.5, cost=5.3e-04),
    "ETHUSD":   dict(EMA=WIDE, TR=2.5, LK=0.5, GB=0.35, ACT=0.3, cost=2.75e-03),
    "JPN225ft": dict(EMA=DFLT, TR=3.0, LK=0.6, GB=0.35, ACT=0.4, cost=4.12e-04),
    "NAS100.r": dict(EMA=DFLT, TR=2.5, LK=0.6, GB=0.35, ACT=0.5, cost=3.0e-04),
}
MIN_ABS, ATR_P, ATR_STOP, TP = 0.34, 20, 3.0, 6.0
ARMS = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 99.0]
REENTRY_BARS = 2


def simulate_arm(m, TR, LK, GB, ACT, arm_R, cost):
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    pos = 0; entry = sl = peak = 0.0; tp = None; e_atr = 0.0
    blocked = 0; blocked_until = -1; trades = []

    def _close(px, reason):
        nonlocal pos
        ret = ((px - entry) / entry) * pos - cost
        ru = (ATR_STOP * e_atr) / entry if entry > 0 and e_atr > 0 else 0.0
        trades.append(ret / ru if ru > 0 else 0.0); pos = 0

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t]); arm_px = arm_R * ATR_STOP * a
        if pos == 0:
            if blocked and (s != blocked or t >= blocked_until):
                blocked = 0
            if s != 0 and s != blocked:
                pos = s; entry = o[t]; peak = 0.0; e_atr = a
                sl = entry - ATR_STOP * a if s == 1 else entry + ATR_STOP * a
                tp = (entry + TP * a) if s == 1 else (entry - TP * a)
            continue
        if s != 0 and s != pos:
            _close(o[t], "flip"); blocked = 0
            pos = s; entry = o[t]; peak = 0.0; e_atr = a
            sl = entry - ATR_STOP * a if pos == 1 else entry + ATR_STOP * a
            tp = (entry + TP * a) if pos == 1 else (entry - TP * a)
            continue
        armed = (peak >= ACT * a) and (peak >= arm_px)
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and armed) else -1e18
            ex = None
            if l[t] <= sl:
                ex = sl
            elif gb > -1e17 and l[t] <= gb:
                ex = gb; blocked = pos; blocked_until = t + REENTRY_BARS
            elif h[t] >= tp:
                ex = tp
            if ex is not None:
                _close(ex, "x")
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR * a)
            if peak >= ACT * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and armed) else 1e18
            ex = None
            if h[t] >= sl:
                ex = sl
            elif gb < 1e17 and h[t] >= gb:
                ex = gb; blocked = pos; blocked_until = t + REENTRY_BARS
            elif l[t] <= tp:
                ex = tp
            if ex is not None:
                _close(ex, "x")
            else:
                peak = max(peak, entry - l[t])
    return trades


def stat(rs):
    n = len(rs); tot = sum(rs)
    w = [r for r in rs if r > 0]; ls = [r for r in rs if r < 0]
    pf = (sum(w) / -sum(ls)) if ls else float("inf")
    aw = sum(w) / len(w) if w else 0.0
    return n, tot, pf, aw


def main():
    summary = {}
    for sym, c in SYMS.items():
        try:
            m = te.load(sym, ema_pairs=c["EMA"], min_abs=MIN_ABS, atr_p=ATR_P)
        except Exception as e:
            print(f"\n### {sym}: LOAD FAILED ({e})"); continue
        print(f"\n### {sym}  ({len(m)} bars)   TR{c['TR']} LK{c['LK']} GB{c['GB']} ACT{c['ACT']}")
        print(f"{'armR':>5} {'n':>5} {'totR':>9} {'PF':>7} {'avgWin':>7} {'trainR':>8} {'testR':>8}  thirds")
        rows = {}
        for arm in ARMS:
            tr = simulate_arm(m, c["TR"], c["LK"], c["GB"], c["ACT"], arm, c["cost"])
            n, tot, pf, aw = stat(tr)
            k = int(len(tr) * 0.6)
            trn = stat(tr[:k])[1]; tst = stat(tr[k:])[1]
            n3 = len(tr) // 3
            th = [round(stat(tr[i*n3:(i+1)*n3] if i < 2 else tr[2*n3:])[1], 1) for i in range(3)]
            rows[arm] = (n, tot, pf, aw, trn, tst, th)
            tag = " <cur" if arm == 0.5 else (" <2.5" if arm == 2.5 else "")
            print(f"{arm:>5} {n:>5} {tot:>+9.2f} {pf:>7.2f} {aw:>+7.3f} {trn:>+8.2f} {tst:>+8.2f}  {th}{tag}")
        base = rows[0.5]
        # pick best arm that PASSES: full-R >= base AND test-R >= base AND thirds >=base in >=2/3
        cands = []
        for arm, r in rows.items():
            if arm == 0.5:
                continue
            th_ok = sum(1 for i in range(3) if r[6][i] >= base[6][i] - 0.05) >= 2
            if r[1] >= base[1] - 1e-9 and r[5] >= base[5] - 1e-9 and th_ok:
                cands.append((r[1], arm))
        pick = max(cands)[1] if cands else 0.5
        summary[sym] = (pick, rows[pick][1] - base[1], rows[pick][5] - base[5])
        verdict = "KEEP 0.5 (inert/no robust gain)" if pick == 0.5 else f"SHIP arm={pick}R"
        print(f"  => {verdict}   (2.5R vs 0.5R: fulldR {rows[2.5][1]-base[1]:+.2f}, testdR {rows[2.5][5]-base[5]:+.2f})")
    print("\n======== SUMMARY (robust pick per symbol) ========")
    for sym, (pick, dR, dT) in summary.items():
        print(f"  {sym:9} -> arm_R {pick}  (fulldR {dR:+.2f}, testdR {dT:+.2f})")


if __name__ == "__main__":
    main()
