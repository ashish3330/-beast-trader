#!/usr/bin/env python3 -B
"""Auto-decide per-symbol tuning based on baseline backtest + live evidence.

Reads:
  - backtest/results/per_sym_tune_2026_05_12/baseline.log  (current config)
  - data/trade_journal.db  (live trades for slippage signal)

Outputs:
  - decision per symbol: KEEP / TIGHTEN / DISABLE / BOOST
  - config patch to apply

Decision rules:
  PF < 0.8  + n>=30  → DISABLE (clear bleeder)
  PF 0.8-1.0          → TIGHTEN +1.0 MIN_SCORE
  PF 1.0-1.3          → TIGHTEN +0.3 MIN_SCORE
  PF >= 1.3           → KEEP
  PF >= 2.0 + n>=200  → could BOOST (loosen MIN_SCORE -0.5, more trades)

  Override: if live evidence shows avg slippage > 30% of avg SL distance
    AND backtest PF < 1.5 → TIGHTEN +1.0 (slippage tax + marginal edge)
"""
import re
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "backtest" / "results" / "per_sym_tune_2026_05_12" / "baseline.log"
DB = ROOT / "data" / "trade_journal.db"

# Parse backtest summary
LINE_RE = re.compile(
    r"^\s+(\S+)\s+\|\s+(\d+) trades \|\s+WR\s+([\d.]+)%\s+\|\s+PF\s+([\d.]+|inf)"
    r"\s+\|\s+PnL \$\s*(-?[\d.]+)\s+\|\s+DD\s+([\d.]+)%"
)

def parse():
    if not BASELINE.exists():
        return {}
    syms = {}
    for line in BASELINE.read_text(errors="ignore").splitlines():
        m = LINE_RE.match(line)
        if m:
            sym, n, wr, pf, pnl, dd = m.groups()
            syms[sym] = {
                "trades": int(n), "wr": float(wr),
                "pf": 99.0 if pf == "inf" else float(pf),
                "pnl": float(pnl), "dd": float(dd),
            }
    return syms


def live_slippage_signal():
    """For each symbol, compute (avg_loss_R, n_trades) over last 7 days.
    If avg_loss_R worse than -1.3 = significant slippage tax."""
    out = {}
    with sqlite3.connect(str(DB)) as c:
        rows = c.execute(
            "SELECT symbol, AVG(r_multiple) FROM trades "
            "WHERE timestamp > datetime('now','-7 days') AND pnl < 0 "
            "GROUP BY symbol HAVING COUNT(*) >= 3"
        ).fetchall()
    for sym, avg_loss_r in rows:
        out[sym] = avg_loss_r
    return out


def decide(syms, live_loss_r):
    decisions = {}
    for sym, s in syms.items():
        pf = s["pf"]; n = s["trades"]; pnl = s["pnl"]
        live_loss = live_loss_r.get(sym, -1.0)

        # Slippage-tax override
        if live_loss < -1.5 and pf < 1.5:
            decisions[sym] = ("TIGHTEN", "+1.0", f"Live loss R={live_loss:.2f} + BT PF {pf:.2f} weak")
            continue

        if n < 10:
            decisions[sym] = ("KEEP", "0", f"n={n} too small to act")
            continue

        if pf < 0.8 and n >= 30:
            decisions[sym] = ("DISABLE", "", f"PF {pf:.2f} on {n} trades — bleeder")
        elif pf < 1.0:
            decisions[sym] = ("TIGHTEN", "+1.0", f"PF {pf:.2f} marginal-negative")
        elif pf < 1.3:
            decisions[sym] = ("TIGHTEN", "+0.3", f"PF {pf:.2f} marginal")
        elif pf >= 2.0 and n >= 200:
            decisions[sym] = ("BOOST", "-0.3", f"PF {pf:.2f} strong, lower MIN_SCORE")
        else:
            decisions[sym] = ("KEEP", "0", f"PF {pf:.2f} adequate")
    return decisions


def main():
    syms = parse()
    if not syms:
        print("No baseline found, run backtest first")
        return 1
    live_loss = live_slippage_signal()

    decisions = decide(syms, live_loss)
    # Sort by PnL impact for visibility
    rows = sorted(decisions.items(),
                  key=lambda kv: syms[kv[0]]["pnl"])

    print(f"{'SYM':<12} {'PF':>5} {'PnL':>8} {'n':>5} {'liveLoss_R':>11} → {'ACTION':<10} {'arg':<6}  reason")
    print("-" * 100)
    actions_summary = defaultdict(list)
    for sym, (act, arg, reason) in rows:
        s = syms[sym]
        ll = f"{live_loss.get(sym, 0):.2f}" if sym in live_loss else "no-data"
        print(f"{sym:<12} {s['pf']:>5.2f} {s['pnl']:>+8.0f} {s['trades']:>5} {ll:>11} → {act:<10} {arg:<6}  {reason}")
        actions_summary[act].append(sym)

    print()
    print("=== SUMMARY ===")
    for act, syms_list in sorted(actions_summary.items()):
        print(f"  {act:<10} ({len(syms_list):>2}): {', '.join(syms_list)}")

    # Generate config patch
    print()
    print("=== CONFIG PATCH ===")
    disable_syms = actions_summary.get("DISABLE", [])
    tighten = {s: arg for s, (a, arg, _) in decisions.items() if a == "TIGHTEN"}
    if disable_syms:
        print(f"# Remove from SYMBOLS dict: {disable_syms}")
    if tighten:
        print(f"# DRAGON_SYMBOL_MIN_SCORE adjustments:")
        for s, arg in tighten.items():
            print(f"#   {s}: +{arg.lstrip('+')} on all regimes")


if __name__ == "__main__":
    sys.exit(main() or 0)
