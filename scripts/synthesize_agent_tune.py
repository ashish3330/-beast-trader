#!/usr/bin/env python3 -B
"""
Synthesize per-symbol tune findings into a config patch.

Reads:  backtest/results/agent_tune_20260513/*.json
Writes: backtest/results/agent_tune_20260513/synthesized.json
        backtest/results/agent_tune_20260513/config_patch.py

Decision rules:
  - Walk-forward PF >= 1.2 AND in-sample PF >= 1.5 → APPLY new params
  - Walk-forward shrinkage < 70% → APPLY
  - Else: KEEP current params (or DISABLE if very weak)
  - WF trades < 10 → KEEP current (insufficient data)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "backtest" / "results" / "agent_tune_20260513"

LIVE = {"XAUUSD","XAGUSD","BTCUSD","ETHUSD","DJ30.r","FRA40.r","GER40.r","HK50.r",
        "JPN225ft","SPI200.r","SWI20.r","US2000.r","COPPER-Cr","GAS-Cr","NG-Cr","UKOUSD",
        "AUDJPY","CADJPY","CHFJPY","GBPJPY","AUDUSD","EURAUD","EURUSD","USDCAD","USDCHF"}


def decide(d):
    """Returns (action, reason, params).
    action: APPLY / KEEP / DISABLE"""
    b = d.get("best")
    wf = d.get("walk_forward_60d")
    sym = d["symbol"]
    if not b:
        return "DISABLE", "no viable in-sample result", None
    if not wf or wf["trades"] < 10:
        return "KEEP", f"WF n={wf['trades'] if wf else 0} insufficient", None
    # Negative WF → disable if also weak IS
    if wf["pnl"] < 0:
        if b["pf"] < 1.3:
            return "DISABLE", f"WF -${abs(wf['pnl']):.0f} + weak IS PF {b['pf']:.2f}", None
        return "KEEP", f"WF -${abs(wf['pnl']):.0f} but IS strong — uncertain", None
    if wf["pf"] < 1.2:
        return "KEEP", f"WF PF {wf['pf']:.2f} < 1.2 — params unstable", None
    if b["pf"] < 1.5:
        return "KEEP", f"IS PF {b['pf']:.2f} < 1.5 — marginal edge", None
    return "APPLY", f"IS PF {b['pf']:.2f} WF PF {wf['pf']:.2f} +${wf['pnl']:.0f}", b


def main():
    decisions = {}
    for f in sorted(DIR.glob("*.json")):
        sym = f.stem
        if sym not in LIVE: continue
        d = json.loads(f.read_text())
        action, reason, params = decide(d)
        decisions[sym] = {"action": action, "reason": reason, "params": params,
                          "is_pnl": d.get("best", {}).get("pnl", 0) if d.get("best") else 0,
                          "wf_pnl": d.get("walk_forward_60d", {}).get("pnl") if d.get("walk_forward_60d") else None}

    summary = {"APPLY": [], "KEEP": [], "DISABLE": []}
    for sym, d in decisions.items():
        summary[d["action"]].append(sym)

    print("=== SYNTHESIZED CONFIG PATCH ===\n")
    for action in ["APPLY", "KEEP", "DISABLE"]:
        syms = summary[action]
        print(f"  {action:<8} ({len(syms):>2}): {', '.join(sorted(syms)) or 'none'}")

    print("\n=== APPLY DETAILS ===")
    apply_total_wf = 0
    for sym in sorted(summary["APPLY"]):
        d = decisions[sym]
        p = d["params"]
        wf = d["wf_pnl"] or 0
        apply_total_wf += wf
        print(f"  {sym:<11} SL={p['sl']}  trail={p['trail']:<8}  ratchet={p['ratchet']}  "
              f"qual={p['qual']}  →  WF +${wf:.0f}")
    print(f"\n  Net WF for APPLY group: ${apply_total_wf:+,.0f}")

    print("\n=== DISABLE DETAILS ===")
    for sym in sorted(summary["DISABLE"]):
        d = decisions[sym]
        print(f"  {sym:<11} {d['reason']}")

    # Save synthesized output
    out = {
        "decisions": decisions,
        "summary": {k: sorted(v) for k, v in summary.items()},
        "apply_total_wf": apply_total_wf,
    }
    out_file = DIR / "synthesized.json"
    out_file.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Saved: {out_file}")

    # Generate config patch
    patch_lines = ["# Auto-synthesized 2026-05-13 from agent_tune_20260513 — DO NOT HAND-EDIT",
                   "# Apply: SL overrides + trail profile per symbol\n"]
    sl_overrides = []
    trail_overrides = []
    for sym in sorted(summary["APPLY"]):
        p = decisions[sym]["params"]
        sl_overrides.append(f'    "{sym}": {p["sl"]},')
        trail_overrides.append(f'    "{sym}": "{p["trail"]}",  # WF +${decisions[sym]["wf_pnl"]:.0f}')

    patch_lines.append("SL_OVERRIDE_AUTO = {")
    patch_lines.extend(sl_overrides)
    patch_lines.append("}\n")
    patch_lines.append("TRAIL_PROFILE_AUTO = {")
    patch_lines.extend(trail_overrides)
    patch_lines.append("}\n")
    patch_lines.append("DISABLE_AUTO = " + repr(sorted(summary["DISABLE"])))

    patch_file = DIR / "config_patch.py"
    patch_file.write_text("\n".join(patch_lines))
    print(f"  Saved: {patch_file}")


if __name__ == "__main__":
    main()
