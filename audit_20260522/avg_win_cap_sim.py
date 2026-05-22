"""Simulate AvgWinLossCap firing on last 30d losses for various (MULT, floor).

For each closed loss, compute the rolling avg_win (same logic as
_get_avg_win_dollars: last 30 wins for that symbol within 30d window) **as of
the trade's close time** to mirror what the executor would have known.

Then for each (MULT, floor) pair:
- cap = max(avg_win * MULT, floor)
- If |pnl| >= cap → AvgWinLossCap would have fired.
- "Savings" = |pnl| - cap (assumes cap closes at the cap level; conservative).

We classify each loss by category (TrailSL / EarlyLossCut / Guardian / etc.) so
we can see WHICH layer AvgWinLossCap would have pre-empted.

Read-only. Writes JSON for the markdown.
"""
import sqlite3
import json
from collections import defaultdict
from pathlib import Path

DB = "/Users/ashish/Documents/beast-trader/data/trade_journal.db"
OUT = "/Users/ashish/Documents/beast-trader/audit_20260522/avg_win_cap_sim.json"


def categorize(r):
    if not r:
        return "unknown"
    if r.startswith("[sl"):
        return "TrailSL"
    if r.startswith("EarlyLossCut"):
        return "EarlyLossCut"
    if r.startswith("PeakGiveback"):
        return "PeakGiveback"
    if r.startswith("Guardian"):
        return "Guardian"
    if r.startswith("Emergency"):
        return "EmergencyDD"
    if r.startswith("Dragon"):
        return "DragonReversal"
    if r.startswith("Daily"):
        return "DailyKillSwitch"
    if r.startswith("HardDollar"):
        return "HardDollarCap"
    if r.startswith("AvgWinLoss"):
        return "AvgWinLossCap"
    if r == "reversal":
        return "DragonReversal"
    return "other"


def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    # All losses in last 30d.
    losses = con.execute(
        "SELECT id, timestamp, symbol, pnl, r_multiple, exit_reason "
        "FROM trades WHERE timestamp >= date('now','-30 days') AND pnl < 0 "
        "ORDER BY timestamp ASC"
    ).fetchall()

    # For each loss, compute rolling avg_win as of its timestamp.
    # _get_avg_win_dollars requires >=5 wins in last 30d; lookback=30 wins.
    rows = []
    for L in losses:
        ts = L["timestamp"]
        sym = L["symbol"]
        # Wins: last 30 wins for this symbol within the 30d trailing window
        # ending at ts. Use date arithmetic via SQLite.
        wins = con.execute(
            "SELECT pnl FROM trades WHERE symbol=? AND pnl>0 "
            "AND timestamp < ? "
            "AND timestamp >= date(?, '-30 days') "
            "ORDER BY id DESC LIMIT 30",
            (sym, ts, ts),
        ).fetchall()
        win_vals = [float(w[0]) for w in wins]
        if len(win_vals) < 5:
            avg_win = 0.0  # cap disabled — fewer than 5 wins, returns 0
        else:
            avg_win = sum(win_vals) / len(win_vals)
        rows.append({
            "id": L["id"],
            "ts": ts,
            "symbol": sym,
            "pnl": float(L["pnl"]),
            "r": float(L["r_multiple"] or 0),
            "exit_reason": L["exit_reason"],
            "category": categorize(L["exit_reason"]),
            "avg_win_at_close": round(avg_win, 4),
            "n_wins_window": len(win_vals),
        })
    con.close()

    # Sweep (MULT, floor)
    sweep = []
    for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        for floor in [0.0, 0.5, 1.0, 1.5, 2.0]:
            n_fires = 0
            saved = 0.0
            total_loss = 0.0
            cat_fires = defaultdict(int)
            cat_saved = defaultdict(float)
            for r in rows:
                if r["avg_win_at_close"] <= 0:
                    # cap function returned 0 (insufficient wins) → no fire
                    continue
                cap = max(r["avg_win_at_close"] * mult, floor)
                if cap <= 0:
                    continue
                if abs(r["pnl"]) >= cap:
                    n_fires += 1
                    save = abs(r["pnl"]) - cap
                    saved += save
                    cat_fires[r["category"]] += 1
                    cat_saved[r["category"]] += save
            total_loss = sum(abs(r["pnl"]) for r in rows)
            sweep.append({
                "MULT": mult, "floor": floor,
                "fires": n_fires, "total_losses": len(rows),
                "saved": round(saved, 2),
                "total_loss": round(total_loss, 2),
                "save_pct": round(100 * saved / total_loss, 1) if total_loss else 0,
                "by_category": {c: {"fires": cat_fires[c], "saved": round(cat_saved[c], 2)} for c in cat_fires},
            })

    # Per-symbol diag at current (2.0, 2.0) and proposed alternatives
    sym_stats = defaultdict(lambda: {
        "n_losses": 0, "tot_loss": 0.0, "avg_loss": 0.0,
        "max_loss": 0.0, "n_with_avg_win": 0,
        "avg_avg_win": 0.0,
        "fires_at_2.0_2.0": 0, "saved_at_2.0_2.0": 0.0,
        "fires_at_1.0_1.0": 0, "saved_at_1.0_1.0": 0.0,
        "fires_at_1.0_0.5": 0, "saved_at_1.0_0.5": 0.0,
        "fires_at_1.5_1.0": 0, "saved_at_1.5_1.0": 0.0,
        "fires_at_0.75_0.5": 0, "saved_at_0.75_0.5": 0.0,
    })
    for r in rows:
        s = sym_stats[r["symbol"]]
        s["n_losses"] += 1
        s["tot_loss"] += abs(r["pnl"])
        if abs(r["pnl"]) > s["max_loss"]:
            s["max_loss"] = abs(r["pnl"])
        if r["avg_win_at_close"] > 0:
            s["n_with_avg_win"] += 1
            s["avg_avg_win"] += r["avg_win_at_close"]
            for mult, floor in [(2.0, 2.0), (1.0, 1.0), (1.0, 0.5), (1.5, 1.0), (0.75, 0.5)]:
                cap = max(r["avg_win_at_close"] * mult, floor)
                key = f"fires_at_{mult}_{floor}"
                save_key = f"saved_at_{mult}_{floor}"
                if abs(r["pnl"]) >= cap:
                    s[key] += 1
                    s[save_key] += abs(r["pnl"]) - cap
    for sym, s in sym_stats.items():
        if s["n_losses"]:
            s["avg_loss"] = round(s["tot_loss"] / s["n_losses"], 3)
            s["tot_loss"] = round(s["tot_loss"], 2)
            s["max_loss"] = round(s["max_loss"], 2)
        if s["n_with_avg_win"]:
            s["avg_avg_win"] = round(s["avg_avg_win"] / s["n_with_avg_win"], 3)
        for k, v in list(s.items()):
            if isinstance(v, float):
                s[k] = round(v, 2)

    out = {
        "n_losses_total": len(rows),
        "n_with_avg_win": sum(1 for r in rows if r["avg_win_at_close"] > 0),
        "sweep": sweep,
        "per_symbol": dict(sym_stats),
        "per_category_summary": _category_summary(rows),
        "rows": rows,  # detailed for transparency
    }
    Path(OUT).write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT}")
    print(f"n_losses = {len(rows)}")
    print(f"n_with_avg_win = {out['n_with_avg_win']}")
    # Quick top-line print
    print("\nSweep summary (fires / saved $ of total ${:,.2f}):".format(
        rows[0]["pnl"] if rows else 0))
    tot = sum(abs(r["pnl"]) for r in rows)
    print(f"Total loss = ${tot:.2f}")
    for s in sweep:
        print(f"  MULT={s['MULT']:.2f} floor=${s['floor']:.2f}  fires={s['fires']:3d}  saved=${s['saved']:6.2f}  ({s['save_pct']}%)")


def _category_summary(rows):
    cat = defaultdict(lambda: {"n": 0, "loss": 0.0})
    for r in rows:
        cat[r["category"]]["n"] += 1
        cat[r["category"]]["loss"] += abs(r["pnl"])
    return {c: {"n": v["n"], "loss": round(v["loss"], 2)} for c, v in cat.items()}


if __name__ == "__main__":
    main()
