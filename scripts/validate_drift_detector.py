#!/usr/bin/env python3 -B
"""
Drift-detector historical validation.

Question this answers: would the drift detector + risk multipliers have
made us *more* or *less* profitable over the historical trade journal?

Method (event-driven replay, no future leakage):
  - Walk every closed trade in trade_journal.db.trades in chronological order.
  - At trade T for symbol S: compute drift state from last N trades of S that
    closed BEFORE T (no peek). Multiplier ∈ {1.0, 0.5, 0.25}.
  - The trade's actual PnL is scaled by the multiplier (risk halves → PnL halves).
  - Aggregate adjusted PnL, win rate, profit factor across the run.

Output: original vs. drift-adjusted PnL, PF, plus per-symbol breakdown.

This validates *the mechanism*. It does not simulate signals that were never
taken (drift never blocks a signal — it only reduces size — so the trade set
is identical, only the PnL magnitudes differ).
"""
import sqlite3
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.drift_detector import (  # noqa: E402
    HEAVY_RISK_MULT, LIGHT_RISK_MULT,
    MIN_TRADES_FOR_DECISION, N_RECENT_TRADES,
    PF_HEAVY, PF_LIGHT, WR_HEAVY, WR_LIGHT,
)

JOURNAL_DB = ROOT / "data" / "trade_journal.db"


def classify(window: list[float]) -> str:
    """Mirror drift_detector._classify on a fixed window of pnls."""
    n = len(window)
    if n < MIN_TRADES_FOR_DECISION:
        return "OK"
    wins = sum(1 for p in window if p > 0)
    wr = wins / n
    gross_w = sum(p for p in window if p > 0)
    gross_l = abs(sum(p for p in window if p < 0))
    if gross_l > 0:
        pf = gross_w / gross_l
    elif gross_w > 0:
        pf = 999.0
    else:
        pf = 1.0
    if wr < WR_HEAVY and pf < PF_HEAVY:
        return "HEAVY"
    if wr < WR_LIGHT and pf < PF_LIGHT:
        return "LIGHT"
    return "OK"


MULT = {"OK": 1.0, "LIGHT": LIGHT_RISK_MULT, "HEAVY": HEAVY_RISK_MULT}


def run() -> None:
    conn = sqlite3.connect(str(JOURNAL_DB))
    rows = conn.execute(
        "SELECT timestamp, symbol, pnl FROM trades "
        "WHERE pnl IS NOT NULL "
        "ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()

    per_sym_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=N_RECENT_TRADES))
    state_counts = {"OK": 0, "LIGHT": 0, "HEAVY": 0}
    per_sym_orig: dict[str, float] = defaultdict(float)
    per_sym_adj: dict[str, float] = defaultdict(float)
    per_sym_n: dict[str, int] = defaultdict(int)
    total_orig = 0.0
    total_adj = 0.0
    wins_orig = 0
    gw_orig = 0.0
    gl_orig = 0.0
    wins_adj = 0
    gw_adj = 0.0
    gl_adj = 0.0

    for _ts, sym, pnl in rows:
        if not sym or pnl is None:
            continue
        pnl = float(pnl)

        # Classify BEFORE adding this trade — no future leakage
        window = list(per_sym_history[sym])
        state = classify(window)
        mult = MULT[state]
        state_counts[state] += 1

        adj_pnl = pnl * mult

        per_sym_orig[sym] += pnl
        per_sym_adj[sym] += adj_pnl
        per_sym_n[sym] += 1
        total_orig += pnl
        total_adj += adj_pnl

        if pnl > 0:
            wins_orig += 1
            gw_orig += pnl
        elif pnl < 0:
            gl_orig += abs(pnl)
        if adj_pnl > 0:
            wins_adj += 1
            gw_adj += adj_pnl
        elif adj_pnl < 0:
            gl_adj += abs(adj_pnl)

        # Append AFTER classification so this trade joins the window for next time
        per_sym_history[sym].append(pnl)

    n = len(rows)
    pf_orig = (gw_orig / gl_orig) if gl_orig else float("inf")
    pf_adj = (gw_adj / gl_adj) if gl_adj else float("inf")
    wr_orig = wins_orig / n if n else 0
    wr_adj = wins_adj / n if n else 0

    print(f"Replayed {n} trades from {JOURNAL_DB.name}")
    print(f"  state distribution: {state_counts}")
    print()
    print(f"  ORIGINAL : net=${total_orig:+.2f}  WR={wr_orig*100:.1f}%  PF={pf_orig:.2f}  wins={wins_orig}")
    print(f"  W/ DRIFT : net=${total_adj:+.2f}  WR={wr_adj*100:.1f}%  PF={pf_adj:.2f}  wins={wins_adj}")
    delta = total_adj - total_orig
    print(f"  DELTA    : ${delta:+.2f} ({(delta / abs(total_orig) * 100) if total_orig else 0:+.1f}%)")
    print()
    print("Per-symbol contribution to delta (top movers):")
    deltas = sorted(
        ((sym, per_sym_adj[sym] - per_sym_orig[sym], per_sym_orig[sym], per_sym_adj[sym], per_sym_n[sym])
         for sym in per_sym_orig),
        key=lambda r: r[1],
    )
    for sym, d, o, a, count in deltas[:5]:
        print(f"  {sym:10s} n={count:3d}  orig=${o:+8.2f}  adj=${a:+8.2f}  delta=${d:+8.2f}")
    print("  ...")
    for sym, d, o, a, count in deltas[-5:]:
        print(f"  {sym:10s} n={count:3d}  orig=${o:+8.2f}  adj=${a:+8.2f}  delta=${d:+8.2f}")


if __name__ == "__main__":
    run()
