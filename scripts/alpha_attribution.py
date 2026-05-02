"""
Alpha attribution — per-component PnL contribution over a window.

Reads data/rl_learner.db trade_outcomes.components_json and aggregates each
score component's average contribution to win/loss outcomes. Surfaces which
of the 11 momentum components actually drive PnL vs which are dead weight.

Output: ranked table per symbol AND portfolio-wide.

Usage:
    python3 -B scripts/alpha_attribution.py                  # all-time, all symbols
    python3 -B scripts/alpha_attribution.py --days 30        # last 30d
    python3 -B scripts/alpha_attribution.py --symbol XAUUSD  # one symbol
    python3 -B scripts/alpha_attribution.py --min-trades 10  # require N trades per component
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "rl_learner.db"


def load_outcomes(symbol: str | None, since_ts: float | None):
    if not DB.exists():
        print(f"NOT FOUND: {DB}")
        sys.exit(1)
    conn = sqlite3.connect(str(DB))
    sql = ("SELECT symbol, won, pnl, r_multiple, score, components_json "
           "FROM trade_outcomes WHERE 1=1")
    args: list = []
    if symbol:
        sql += " AND symbol = ?"
        args.append(symbol)
    if since_ts:
        sql += " AND ts >= ?"
        args.append(since_ts)
    sql += " ORDER BY ts DESC"
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return rows


def aggregate(rows, min_trades: int = 5):
    """Group by (symbol, component) and compute avg PnL when component was active."""
    # comp_buckets: (symbol, comp_name) -> {"on_trades": [...], "off_trades": [...]}
    bucket: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"on": [], "off": [], "on_r": [], "off_r": []})
    portfolio: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"on": [], "off": [], "on_r": [], "off_r": []})

    skipped_no_comps = 0
    for sym, won, pnl, r_mult, score, comps_json in rows:
        if not comps_json:
            skipped_no_comps += 1
            continue
        try:
            comps = json.loads(comps_json) or {}
        except Exception:
            skipped_no_comps += 1
            continue
        if not isinstance(comps, dict):
            skipped_no_comps += 1
            continue
        # Active components: nonzero score
        for cname, cval in comps.items():
            try:
                cv = float(cval)
            except Exception:
                continue
            key = "on" if cv > 0 else "off"
            bucket[(sym, cname)][key].append(float(pnl))
            bucket[(sym, cname)][key + "_r"].append(float(r_mult))
            portfolio[cname][key].append(float(pnl))
            portfolio[cname][key + "_r"].append(float(r_mult))
    return bucket, portfolio, skipped_no_comps


def render_table(by_sym, portfolio, min_trades: int):
    print()
    print("══════════════════════════════════════════════════════════════════════")
    print("  PORTFOLIO-WIDE ALPHA ATTRIBUTION  (per-component contribution)")
    print("══════════════════════════════════════════════════════════════════════")
    print(f"{'COMPONENT':<22} {'N_ON':>6} {'PnL_ON':>9} {'avgR_ON':>8} {'PnL_OFF':>9} {'avgR_OFF':>9} {'LIFT':>8}")
    print("-" * 75)
    rows_p = []
    for cname, d in portfolio.items():
        if len(d["on"]) + len(d["off"]) < min_trades:
            continue
        n_on = len(d["on"])
        pnl_on = sum(d["on"])
        avg_r_on = sum(d["on_r"]) / max(len(d["on_r"]), 1)
        pnl_off = sum(d["off"])
        avg_r_off = sum(d["off_r"]) / max(len(d["off_r"]), 1)
        lift = avg_r_on - avg_r_off
        rows_p.append((cname, n_on, pnl_on, avg_r_on, pnl_off, avg_r_off, lift))
    # Sort by lift desc — components that ADD edge first
    rows_p.sort(key=lambda x: -x[6])
    for cname, n_on, pnl_on, avg_r_on, pnl_off, avg_r_off, lift in rows_p:
        marker = "★" if lift > 0.2 else (" " if lift > 0 else "✗")
        print(f"{cname:<22} {n_on:>6} ${pnl_on:>7.0f} {avg_r_on:>+8.2f} ${pnl_off:>7.0f} {avg_r_off:>+9.2f} {lift:>+7.2f} {marker}")

    print()
    print("══════════════════════════════════════════════════════════════════════")
    print("  PER-SYMBOL SUMMARY  (top 3 components per symbol by lift)")
    print("══════════════════════════════════════════════════════════════════════")
    sym_components: dict[str, list] = defaultdict(list)
    for (sym, cname), d in by_sym.items():
        if len(d["on"]) < min_trades:
            continue
        avg_r_on = sum(d["on_r"]) / max(len(d["on_r"]), 1)
        avg_r_off = sum(d["off_r"]) / max(len(d["off_r"]), 1)
        lift = avg_r_on - avg_r_off
        sym_components[sym].append((cname, lift, len(d["on"]), avg_r_on))
    for sym in sorted(sym_components):
        comps = sorted(sym_components[sym], key=lambda x: -x[1])[:3]
        if not comps:
            continue
        line = f"  {sym:<10}  "
        for c, lift, n, avg_r in comps:
            line += f"{c}:{lift:+.2f} (n={n}) | "
        print(line)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--days", type=int, default=None,
                    help="lookback window; default = all-time")
    ap.add_argument("--min-trades", type=int, default=5)
    args = ap.parse_args()

    since_ts = None
    if args.days:
        since_ts = time.time() - args.days * 86400

    rows = load_outcomes(args.symbol, since_ts)
    print(f"Loaded {len(rows)} trade outcomes "
          f"({'all time' if not args.days else f'last {args.days}d'}, "
          f"{'all syms' if not args.symbol else args.symbol})")

    by_sym, portfolio, skipped = aggregate(rows, args.min_trades)
    if skipped:
        print(f"Skipped {skipped} outcomes with no components_json "
              f"(likely backtest-bootstrapped seed data — live trades will populate)")
    if not portfolio:
        print()
        print("No usable component data yet. Live trades populate components_json on")
        print("close via agent/brain.py:1294 → rl_learner.record_outcome(score_components=...).")
        print("Run again after enough live trades have closed (typically 20+ per symbol).")
        return

    render_table(by_sym, portfolio, args.min_trades)


if __name__ == "__main__":
    main()
