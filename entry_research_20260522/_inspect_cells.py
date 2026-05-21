#!/usr/bin/env python3 -B
"""Inspect HIGH_CONFIDENCE + BLACKLIST_NEG_EV cells per symbol."""
import json
from pathlib import Path

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
p = Path(__file__).resolve().parent / "10_stat_edge.json"
d = json.load(open(p))

print("=" * 78)
print("HIGH-CONFIDENCE + BLACKLIST CELLS  (n>=15 trades)")
print("=" * 78)
for sym, info in d["symbols"].items():
    hc = []
    wl = []
    bl = []
    for k, m in info["cells"].items():
        if m["recommend"] == "HIGH_CONFIDENCE":
            hc.append((k, m))
        elif m["recommend"] == "WHITELIST_OK":
            wl.append((k, m))
        elif m["recommend"] == "BLACKLIST_NEG_EV":
            bl.append((k, m))

    print(f"\n{sym}  baseline PnL=${info['baseline']['pnl']:.0f}  "
          f"deltaBL=${info['delta_bl']:+.2f}")
    if hc:
        print("  HIGH_CONFIDENCE:")
        for k, m in sorted(hc):
            h, d2 = k.split("_")
            print(f"    {DOW[int(d2)]} {h}h  n={m['n']:3d} pf={m['pf']:5.2f} "
                  f"wr={m['wr']:5.1f}% pnl=${m['pnl']:8.2f}")
    if wl:
        print("  WHITELIST_OK (PF>=2 n>=15):")
        for k, m in sorted(wl):
            h, d2 = k.split("_")
            print(f"    {DOW[int(d2)]} {h}h  n={m['n']:3d} pf={m['pf']:5.2f} "
                  f"wr={m['wr']:5.1f}% pnl=${m['pnl']:8.2f}")
    if bl:
        print("  BLACKLIST_NEG_EV (PF<1 n>=15):")
        for k, m in sorted(bl):
            h, d2 = k.split("_")
            print(f"    {DOW[int(d2)]} {h}h  n={m['n']:3d} pf={m['pf']:5.2f} "
                  f"wr={m['wr']:5.1f}% pnl=${m['pnl']:8.2f}")

print("\n" + "=" * 78)
print("PORTFOLIO SUMMARY")
print("=" * 78)
p2 = d["portfolio"]
print(f"  base    ${p2['baseline_pnl']:>12,.2f}")
print(f"  WL      ${p2['whitelist_pnl']:>12,.2f}  delta={p2['delta_wl']:+,.2f}")
print(f"  BL      ${p2['blacklist_pnl']:>12,.2f}  delta={p2['delta_bl']:+,.2f}")
