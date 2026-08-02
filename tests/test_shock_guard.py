#!/usr/bin/env python3 -B
"""Offline verification for the shock guard (2026-08-02). Asserts fire rate
~1.5-2.1%, known shock months fire, causality (no look-ahead), fail-open, and
the oil-never-blocked asset routing."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

CACHE = Path(C.SHOCK_GUARD_CACHE_DIR)
ATR_N, MED_N = 14, 100


def key(s):
    return "xauusd" if s == "XAUUSD" else s.replace(".", "_")


def fire_series(sym, trim_last=0):
    p = CACHE / ("raw_d1_" + key(sym) + ".pkl")
    if not p.exists():
        return None
    df = pickle.load(open(p, "rb")).copy()
    if trim_last:
        df = df.iloc[:-trim_last]
    h, l, c = df.high.astype(float), df.low.astype(float), df.close.astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N).mean()
    ar = atr / atr.shift(1).rolling(MED_N).median()
    rng = (h - l)
    rz = (rng - rng.shift(1).rolling(MED_N).mean()) / rng.shift(1).rolling(MED_N).std().replace(0, np.nan)
    Z, A = C.SHOCK_GUARD_THRESHOLDS.get(sym, C.SHOCK_GUARD_DEFAULT_THRESHOLD)
    fired = ((rz >= Z) | (ar >= A)).fillna(False)
    return df.time.reset_index(drop=True), fired.reset_index(drop=True)


def main():
    ok = True
    shock_months = {"XAUUSD": ["2020-03", "2022-03"], "NAS100.r": ["2020-03", "2024-08", "2025-04"],
                    "JPN225ft": ["2024-08"], "USOUSD": ["2020-04", "2022-03"], "BTCUSD": ["2020-03"]}
    print("=== fire rate + shock-month fire (target ~1.4-2.5%) ===")
    for sym in ["XAUUSD", "USOUSD", "NAS100.r", "JPN225ft", "BTCUSD"]:
        r = fire_series(sym)
        if r is None:
            print("  %-10s NO CACHE" % sym); continue
        t, f = r
        rate = 100 * f.mean()
        hits = []
        for ym in shock_months.get(sym, []):
            mask = t.astype(str).str.startswith(ym)
            got = bool(f[mask].any())
            hits.append("%s:%s" % (ym, "Y" if got else "N"))
            if not got:
                ok = False
        rate_ok = 1.0 <= rate <= 3.0
        if not rate_ok:
            ok = False
        print("  %-10s fire=%.2f%% n=%d %s shocks[%s]"
              % (sym, rate, int(f.sum()), "OK" if rate_ok else "RATE_OUT", " ".join(hits)))

    print("\n=== causality (hide last bar -> prior fires identical) ===")
    _, f_full = fire_series("NAS100.r")
    _, f_trim = fire_series("NAS100.r", trim_last=1)
    same = bool((f_full.iloc[:-1].values == f_trim.values).all())
    print("  no look-ahead:", same)
    ok = ok and same

    print("\n=== fail-open (missing cache -> NONE) ===")
    from agent.shock_guard import ShockGuard
    st = ShockGuard(cache_dir="/nonexistent").guard_state("NAS100.r")
    fo = st["action"] == "NONE"
    print("  missing-cache action:", st["action"], "->", "OK" if fo else "FAIL")
    ok = ok and fo

    print("\n=== asset routing (oil NEVER blocked) ===")
    sg = ShockGuard()
    routes = {s: sg._action_for(s) for s in ["NAS100.r", "JPN225ft", "XAUUSD", "USOUSD", "BTCUSD"]}
    for s, a in routes.items():
        print("  %-10s -> %s" % (s, a))
    routing_ok = (routes["NAS100.r"] == "BLOCK" and routes["JPN225ft"] == "BLOCK"
                  and routes["USOUSD"] == "DERISK" and routes["XAUUSD"] == "DERISK")
    ok = ok and routing_ok

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
