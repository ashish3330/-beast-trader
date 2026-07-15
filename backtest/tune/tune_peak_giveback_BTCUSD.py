#!/usr/bin/env python3 -B
"""Peak-giveback ARM-R tune for BTCUSD (2026-07-15, inline).

Tests the user hypothesis: PEAK_GIVEBACK_ACTIVATE_R should be 2.5R, not 0.5R —
i.e. don't arm the peak-giveback reversal exit until the trade is +2.5R, so
winners run. Reuses trend_engine + the retune_trend_BTCUSD simulate_live (2h
re-entry cooldown = live-faithful), adding an R-based arm gate on the GIVEBACK
only (the profit-LOCK keeps its ACT arm). Strict: WF 60/40 + 3-thirds stability.
Does NOT touch config.py / brain.py.
"""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import trend_engine as te

SYMBOL = "BTCUSD"
BASE = dict(EMA_PAIRS=[(16, 64), (32, 128), (64, 256)], MIN_ABS=0.34, ATR_P=20,
            ATR_STOP=3.0, TP_ATR=6.0, TRAIL=3.0, LOCK=0.5, GIVEBACK=0.30, ACT=0.5)
COST = 0.00053                     # BTC round-trip spread (retune convention)
ARMS = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 99.0]   # 99 = giveback effectively off
REENTRY_BARS = 2                    # TREND_REENTRY_BLOCK_HOURS=2.0 -> 2 H1 bars


def simulate_arm(m, TR, LK, GB, ACT, ATR_STOP, TP, arm_R, cost):
    """simulate_live verbatim except: giveback arms only once peak_R >= arm_R
    (peak_R = peak / (ATR_STOP*atr)). Lock keeps its ACT*atr arm."""
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    pos = 0; entry = sl = peak = 0.0; tp = None; e_atr = 0.0
    blocked = 0; blocked_until = -1; trades = []

    def _close(ex_price, reason):
        nonlocal pos
        ret = ((ex_price - entry) / entry) * pos - cost
        r_unit = (ATR_STOP * e_atr) / entry if entry > 0 and e_atr > 0 else 0.0
        trades.append({"pnl_R": ret / r_unit if r_unit > 0 else 0.0, "reason": reason})
        pos = 0

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        arm_px = arm_R * ATR_STOP * a          # peak (price) needed to arm giveback
        if pos == 0:
            if blocked and (s != blocked or t >= blocked_until):
                blocked = 0
            if s != 0 and s != blocked:
                pos = s; entry = o[t]; peak = 0.0; e_atr = a
                sl = entry - ATR_STOP * a if s == 1 else entry + ATR_STOP * a
                tp = None if TP >= 999 else ((entry + TP * a) if s == 1 else (entry - TP * a))
            continue
        if s != 0 and s != pos:
            _close(o[t], "flip"); blocked = 0
            pos = s; entry = o[t]; peak = 0.0; e_atr = a
            sl = entry - ATR_STOP * a if pos == 1 else entry + ATR_STOP * a
            tp = None if TP >= 999 else ((entry + TP * a) if pos == 1 else (entry - TP * a))
            continue
        armed = (peak >= ACT * a) and (peak >= arm_px)
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and armed) else -1e18
            ex = ereason = None
            if l[t] <= sl:
                ex, ereason = sl, "stop/trail"
            elif gb > -1e17 and l[t] <= gb:
                ex, ereason = gb, "giveback"; blocked = pos; blocked_until = t + REENTRY_BARS
            elif tp is not None and h[t] >= tp:
                ex, ereason = tp, "tp"
            if ex is not None:
                _close(ex, ereason)
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR * a)
            if peak >= ACT * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and armed) else 1e18
            ex = ereason = None
            if h[t] >= sl:
                ex, ereason = sl, "stop/trail"
            elif gb < 1e17 and h[t] >= gb:
                ex, ereason = gb, "giveback"; blocked = pos; blocked_until = t + REENTRY_BARS
            elif tp is not None and l[t] <= tp:
                ex, ereason = tp, "tp"
            if ex is not None:
                _close(ex, ereason)
            else:
                peak = max(peak, entry - l[t])
    return trades


def stats(trades):
    rs = [t["pnl_R"] for t in trades]
    n = len(rs); tot = sum(rs)
    wins = [r for r in rs if r > 0]; loss = [r for r in rs if r < 0]
    gp = sum(wins); gl = -sum(loss)
    pf = gp / gl if gl > 0 else float("inf")
    aw = gp / len(wins) if wins else 0.0
    al = -gl / len(loss) if loss else 0.0
    return dict(n=n, tot=tot, pf=pf, avg_win=aw, avg_loss=al, nw=len(wins), nl=len(loss))


def main():
    m = te.load(SYMBOL, ema_pairs=BASE["EMA_PAIRS"], min_abs=BASE["MIN_ABS"], atr_p=BASE["ATR_P"])
    print(f"{SYMBOL}: {len(m)} merged bars  {m.index[0]} -> {m.index[-1]}")
    tp = 999.0 if BASE["TP_ATR"] is None else BASE["TP_ATR"]
    print(f"\n{'armR':>5} {'n':>4} {'totR':>8} {'PF':>7} {'avgWin':>7} {'avgLoss':>8} "
          f"{'trainR':>8} {'testR':>8} {'thirds R (n)':>28}")
    rows = {}
    for arm in ARMS:
        tr = simulate_arm(m, BASE["TRAIL"], BASE["LOCK"], BASE["GIVEBACK"], BASE["ACT"],
                          BASE["ATR_STOP"], tp, arm, COST)
        st = stats(tr)
        k = int(len(tr) * 0.60)
        trn = stats(tr[:k])["tot"] if k else 0.0
        tst = stats(tr[k:])["tot"] if len(tr) - k else 0.0
        th = []
        n3 = len(tr) // 3
        for i in range(3):
            seg = tr[i*n3:(i+1)*n3] if i < 2 else tr[2*n3:]
            s3 = stats(seg); th.append((round(s3["tot"], 2), s3["n"]))
        rows[arm] = dict(st=st, train=trn, test=tst, thirds=th)
        thstr = " ".join(f"{r:+.1f}({nn})" for r, nn in th)
        tag = " <== current" if arm == 0.5 else (" <== USER 2.5" if arm == 2.5 else "")
        print(f"{arm:>5} {st['n']:>4} {st['tot']:>+8.2f} {st['pf']:>7.2f} "
              f"{st['avg_win']:>+7.3f} {st['avg_loss']:>+8.3f} {trn:>+8.2f} {tst:>+8.2f} "
              f"{thstr:>28}{tag}")

    base = rows[0.5]; u25 = rows[2.5]
    print("\n--- VERDICT: 2.5R vs 0.5R (current) ---")
    print(f"  full R : {base['st']['tot']:+.2f} -> {u25['st']['tot']:+.2f}  "
          f"(delta {u25['st']['tot']-base['st']['tot']:+.2f})")
    print(f"  test R : {base['test']:+.2f} -> {u25['test']:+.2f}  "
          f"({'PASS' if u25['test'] >= base['test'] else 'FAIL'} held-out)")
    print(f"  avg win: {base['st']['avg_win']:+.3f} -> {u25['st']['avg_win']:+.3f}  "
          f"(winners {'run bigger' if u25['st']['avg_win']>base['st']['avg_win'] else 'NOT bigger'})")
    wins3 = sum(1 for i in range(3) if u25['thirds'][i][0] >= base['thirds'][i][0])
    print(f"  thirds : 2.5R >= 0.5R in {wins3}/3 windows")
    best = max(rows, key=lambda a: rows[a]['st']['tot'])
    print(f"  best-total-R arm = {best}R (R={rows[best]['st']['tot']:+.2f}); "
          f"best-test-R arm = {max(rows, key=lambda a: rows[a]['test'])}R")


if __name__ == "__main__":
    main()
