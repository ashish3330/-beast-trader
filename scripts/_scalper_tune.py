#!/usr/bin/env python3 -B
"""Grid-tune the M1 scalper (research features) for XAU. In-process (fast).
Ranks by PF with a frequency view; walk-forward checks the top picks."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._scalper_run import run  # noqa: E402

SYM = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
SESSIONS = [(0, 24), (7, 17), (7, 20), (8, 18), (12, 20)]
ADX = [15, 18, 22]
RSI = [(5, 95), (10, 90)]
BBM = [1.8, 2.0, 2.5]
SLA = [1.0, 1.5]
TS = [10, 15, 20]
KEYS = ("H_START", "H_END", "ADX_MAX", "RSI_LOW", "RSI_HIGH", "BB_MULT", "SL_ATR", "TIME_STOP", "TP_MEAN")


def setenv(cfg):
    hs, he, adx, rl, rh, bb, sl, ts = cfg
    for k in KEYS:
        os.environ.pop(f"SCALP_{k}", None)
    for k, v in zip(KEYS, (hs, he, adx, rl, rh, bb, sl, ts, 1)):
        os.environ[f"SCALP_{k}"] = str(v)


def folds4(cfg):
    setenv(cfg)
    fps, favg = [], []
    for f in range(4):
        fr = run(SYM, days=None, fold=f, folds=4)
        if fr.get("status") == "OK":
            fps.append(fr.get("pf", 0)); favg.append(fr.get("avg_R", 0))
        else:
            fps.append(0); favg.append(-1)
    return fps


rows = []
for hs, he in SESSIONS:
    for adx in ADX:
        for rl, rh in RSI:
            for bb in BBM:
                for sl in SLA:
                    for ts in TS:
                        cfg = (hs, he, adx, rl, rh, bb, sl, ts)
                        setenv(cfg)
                        r = run(SYM, days=None)
                        if r.get("status") != "OK" or r.get("trades", 0) < 150:
                            continue
                        rows.append({"cfg": cfg, **r})

# Screen by full PF first, then walk-forward the top 20 and rank by ROBUSTNESS
rows.sort(key=lambda r: r["pf"], reverse=True)
print(f"total configs passing trade floor: {len(rows)}")
cand = rows[:20]
for r in cand:
    fps = folds4(r["cfg"])
    r["folds"] = fps
    r["min_fold"] = min(fps)
    r["npos"] = sum(1 for p in fps if p > 1.0)

# Ship rule: all 4 folds > 1.0, then maximize min-fold PF (robust), tiebreak per_day.
robust = [r for r in cand if r["npos"] == 4 and r["max_dd_R"] <= 20]
robust.sort(key=lambda r: (r["min_fold"], r["per_day"]), reverse=True)

print(f"\n{'session':>8} {'adx':>3} {'rsi':>6} {'bb':>4} {'sl':>4} {'ts':>3} | "
      f"{'/day':>5} {'WR':>4} {'PF':>6} {'DD':>5} {'minFold':>7} {'folds':>4}")
print("-" * 78)
for r in (robust[:10] or cand[:10]):
    hs, he, adx, rl, rh, bb, sl, ts = r["cfg"]
    print(f"{hs:>2}-{he:<4} {adx:>3} {rl:>2}/{rh:<3} {bb:>4} {sl:>4} {ts:>3} | "
          f"{r['per_day']:>5} {int(r['wr']*100):>3}% {r['pf']:>6} {r['max_dd_R']:>5} "
          f"{r.get('min_fold',0):>7} {r.get('npos',0)}/4")

if robust:
    w = robust[0]
    hs, he, adx, rl, rh, bb, sl, ts = w["cfg"]
    print(f"\nSHIP (most robust): H_START={hs} H_END={he} ADX_MAX={adx} RSI_LOW={rl} "
          f"RSI_HIGH={rh} BB_MULT={bb} SL_ATR={sl} TIME_STOP={ts} TP_MEAN=1")
    print(f"  full PF={w['pf']} per_day={w['per_day']} WR={int(w['wr']*100)}% "
          f"DD={w['max_dd_R']} folds={[round(x,2) for x in w['folds']]}")
else:
    print("\nNO config had all 4 folds positive — scalper edge not robust enough.")
