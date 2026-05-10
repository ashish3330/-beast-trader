#!/usr/bin/env python3 -B
"""Compare baseline vs candidate backtest JSONs.

For each candidate, vs baseline, report:
  - aggregate PnL delta
  - per-symbol regression count (symbols that went profitable→unprofitable)
  - per-symbol improvement count
  - aggregate trade count delta
  - PASS / FAIL verdict (PASS if PnL > baseline by >=5% AND regressions <= 2)

Usage: python3 -B scripts/compare_momentum_results.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "backtest" / "results" / "momentum_tune"

PNL_GAIN_THRESHOLD_PCT = 5.0   # candidate must beat baseline by 5%
MAX_REGRESSIONS = 2            # at most this many profitable→unprofitable flips


def load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def compare(baseline: dict, cand: dict, label: str) -> dict:
    bs = baseline.get("symbols", {})
    cs = cand.get("symbols", {})
    bt = baseline.get("total", {"pnl": 0, "trades": 0})
    ct = cand.get("total", {"pnl": 0, "trades": 0})

    pnl_delta = ct["pnl"] - bt["pnl"]
    pnl_pct = (pnl_delta / abs(bt["pnl"]) * 100) if bt["pnl"] else 0.0
    trade_delta = ct["trades"] - bt["trades"]

    regressions = []
    improvements = []
    new_winners = []
    new_losers = []
    for sym in sorted(set(bs) | set(cs)):
        b = bs.get(sym, {"pnl": 0, "trades": 0})
        c = cs.get(sym, {"pnl": 0, "trades": 0})
        b_p, c_p = b["pnl"], c["pnl"]
        if b_p > 0 and c_p < 0:
            regressions.append((sym, b_p, c_p))
        elif b_p < 0 and c_p > 0:
            improvements.append((sym, b_p, c_p))
        elif b_p == 0 and c_p > 50:
            new_winners.append((sym, c_p))
        elif b_p == 0 and c_p < -50:
            new_losers.append((sym, c_p))

    pass_pnl = pnl_pct >= PNL_GAIN_THRESHOLD_PCT
    pass_regr = len(regressions) <= MAX_REGRESSIONS
    verdict = "PASS" if pass_pnl and pass_regr else "FAIL"

    return {
        "label": label,
        "verdict": verdict,
        "pnl_baseline": bt["pnl"],
        "pnl_candidate": ct["pnl"],
        "pnl_delta": pnl_delta,
        "pnl_pct": pnl_pct,
        "trade_delta": trade_delta,
        "regressions": regressions,
        "improvements": improvements,
        "new_winners": new_winners,
        "new_losers": new_losers,
        "pass_pnl": pass_pnl,
        "pass_regr": pass_regr,
    }


def fmt_row(name: str, baseline: dict, cands: list[dict]) -> None:
    print(f"\n{'═' * 80}")
    print(f"  {name}")
    print(f"{'═' * 80}")
    bt = baseline.get("total", {"pnl": 0, "trades": 0})
    print(f"  Baseline:  ${bt['pnl']:>9,.0f} | {bt['trades']:>5} trades")
    print(f"  {'─' * 76}")
    for c in cands:
        sign = "+" if c["pnl_delta"] >= 0 else ""
        verdict_color = c["verdict"]
        print(f"  {c['label']:<20} ${c['pnl_candidate']:>9,.0f} "
              f"| Δ {sign}${c['pnl_delta']:>+8,.0f} ({sign}{c['pnl_pct']:>+6.1f}%) "
              f"| trades Δ {c['trade_delta']:>+5} "
              f"| regr {len(c['regressions'])} "
              f"→ {verdict_color}")
        if c["regressions"]:
            for sym, bp, cp in c["regressions"][:3]:
                print(f"      regression {sym}: ${bp:.0f} → ${cp:.0f}")


def main():
    baseline = load("baseline")
    if not baseline:
        print("FAIL: baseline.json not found. Run baseline backtest first.")
        return 1

    candidates = []
    for name in ("cand1_size_boost", "cand2_trail", "cand4_minscore", "combined"):
        cand = load(name)
        if not cand:
            print(f"  (skipping {name} — JSON not found)")
            continue
        candidates.append(compare(baseline, cand, name))

    if not candidates:
        print("FAIL: no candidate JSONs found.")
        return 1

    fmt_row("MOMENTUM-ADAPTIVE TUNING — RESULTS", baseline, candidates)

    # Recommend deployments
    print(f"\n{'═' * 80}")
    print("  DEPLOYMENT RECOMMENDATION")
    print(f"{'═' * 80}")
    winners = [c for c in candidates if c["verdict"] == "PASS"]
    if not winners:
        print("  No candidate passed both gates (>=5% PnL gain AND <=2 regressions).")
        print("  → Keep all flags OFF.")
    else:
        print("  Passed gates (eligible for walk-forward validation):")
        for w in winners:
            print(f"    ✓ {w['label']:<20} +{w['pnl_pct']:.1f}% PnL, "
                  f"{len(w['regressions'])} regressions")
        print()
        print("  Next step: walk-forward k-fold each winner. Deploy only those")
        print("  whose test PnL holds within 25% of train PnL.")

    # Save consolidated report
    report = {
        "baseline": {"pnl": baseline["total"]["pnl"], "trades": baseline["total"]["trades"]},
        "candidates": [{
            "label": c["label"],
            "pnl": c["pnl_candidate"],
            "pnl_delta": c["pnl_delta"],
            "pnl_pct": c["pnl_pct"],
            "trade_delta": c["trade_delta"],
            "regressions_count": len(c["regressions"]),
            "regressions": c["regressions"],
            "improvements_count": len(c["improvements"]),
            "verdict": c["verdict"],
        } for c in candidates],
        "winners": [c["label"] for c in winners] if winners else [],
    }
    out = RESULTS / "comparison_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
