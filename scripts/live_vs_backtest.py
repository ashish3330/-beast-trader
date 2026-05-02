#!/usr/bin/env python3
"""Compare last-30d live performance (data/trade_journal.db) vs 180d backtest pass1 projection.

Outputs:
  backtest/results/live_vs_backtest.json
  backtest/results/live_vs_backtest.md
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
JOURNAL = ROOT / "data" / "trade_journal.db"
BT_PASS1 = ROOT / "backtest" / "results" / "tune_180d_pass1.json"
OUT_JSON = ROOT / "backtest" / "results" / "live_vs_backtest.json"
OUT_MD = ROOT / "backtest" / "results" / "live_vs_backtest.md"

LIVE_WINDOW_DAYS = 30
BT_DAYS = 180
DIVERGENCE_THRESH = 0.5  # |ratio - 1| > 0.5 flags


def load_live() -> dict:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=LIVE_WINDOW_DAYS)).isoformat()
    con = sqlite3.connect(f"file:{JOURNAL}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute(
        """
        SELECT symbol,
               COUNT(*)                                       AS trades,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)       AS wins,
               COALESCE(SUM(pnl), 0.0)                        AS pnl,
               COALESCE(AVG(r_multiple), 0.0)                 AS avg_r,
               MIN(timestamp)                                 AS first_ts,
               MAX(timestamp)                                 AS last_ts
        FROM trades
        WHERE timestamp >= ?
        GROUP BY symbol
        """,
        (cutoff,),
    )
    rows = cur.fetchall()
    cur.execute(
        "SELECT COUNT(*), COALESCE(SUM(pnl),0.0) FROM trades WHERE timestamp >= ?",
        (cutoff,),
    )
    total_trades, total_pnl = cur.fetchone()
    con.close()

    per_sym = {}
    for sym, n, wins, pnl, avg_r, first_ts, last_ts in rows:
        per_sym[sym] = {
            "trades": int(n),
            "wins": int(wins or 0),
            "wr": round(100.0 * (wins or 0) / n, 2) if n else 0.0,
            "pnl": round(float(pnl), 2),
            "avg_r": round(float(avg_r), 3),
            "first_ts": first_ts,
            "last_ts": last_ts,
        }
    return {"per_sym": per_sym, "total_trades": int(total_trades), "total_pnl": round(float(total_pnl), 2)}


def load_backtest() -> dict:
    raw = json.loads(BT_PASS1.read_text())
    out = {}
    for sym, payload in raw.get("results", {}).items():
        best = (payload or {}).get("best") or {}
        result = best.get("result") or {}
        if not result:
            continue
        out[sym] = {
            "trades": int(result.get("trades", 0) or 0),
            "wr": float(result.get("wr", 0.0) or 0.0),
            "pf": float(result.get("pf", 0.0) or 0.0),
            "pnl": float(result.get("pnl", 0.0) or 0.0),
            "avg_r": float(result.get("avg_r", 0.0) or 0.0),
            "dd": float(result.get("dd", 0.0) or 0.0),
        }
    return out


def main() -> None:
    live = load_live()
    bt = load_backtest()

    rows = []
    syms = sorted(set(live["per_sym"].keys()) | set(bt.keys()))
    for sym in syms:
        l = live["per_sym"].get(sym)
        b = bt.get(sym)

        live_trades = l["trades"] if l else 0
        live_pnl = l["pnl"] if l else 0.0
        live_wr = l["wr"] if l else None
        live_avg_r = l["avg_r"] if l else None

        bt_trades = b["trades"] if b else 0
        bt_pnl = b["pnl"] if b else 0.0
        bt_pf = b["pf"] if b else None
        bt_wr = b["wr"] if b else None

        live_per_day = live_pnl / LIVE_WINDOW_DAYS
        bt_per_day = (bt_pnl / BT_DAYS) if b else 0.0

        if bt_per_day != 0.0:
            ratio = live_per_day / bt_per_day
            divergence = abs(ratio - 1.0)
        elif live_per_day == 0.0:
            ratio = 1.0
            divergence = 0.0
        else:
            ratio = None  # backtest projects $0/day but live moved
            divergence = float("inf")

        flag = (
            divergence == float("inf")
            or (ratio is not None and abs(ratio - 1.0) > DIVERGENCE_THRESH)
        )

        rows.append({
            "sym": sym,
            "live_30d_trades": live_trades,
            "live_30d_wr": live_wr,
            "live_30d_pnl": round(live_pnl, 2),
            "live_avg_r": live_avg_r,
            "live_per_day": round(live_per_day, 4),
            "backtest_180d_trades": bt_trades,
            "backtest_180d_pf": bt_pf,
            "backtest_180d_wr": bt_wr,
            "backtest_180d_pnl": round(bt_pnl, 2),
            "backtest_per_day": round(bt_per_day, 4),
            "ratio": round(ratio, 3) if isinstance(ratio, (int, float)) else None,
            "divergence": (None if divergence == float("inf") else round(divergence, 3)),
            "divergence_flag": bool(flag),
        })

    out_payload = {
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "live_window_days": LIVE_WINDOW_DAYS,
        "backtest_days": BT_DAYS,
        "divergence_threshold": DIVERGENCE_THRESH,
        "totals": {
            "live_trades_30d": live["total_trades"],
            "live_pnl_30d": live["total_pnl"],
        },
        "rows": rows,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_payload, indent=2))

    # Markdown sorted by absolute divergence desc (None/inf first)
    def sort_key(r):
        d = r["divergence"]
        if d is None:
            return (0, 0.0)  # inf-divergence symbols first
        return (1, -d)

    sorted_rows = sorted(rows, key=sort_key)
    top = sorted_rows[:10]

    md = []
    md.append("# Live (30d) vs Backtest (180d pass-1) — Per-Symbol Divergence")
    md.append("")
    md.append(f"- Captured: {out_payload['captured_at']}")
    md.append(f"- Live trades (30d): **{live['total_trades']}**")
    md.append(f"- Live PnL (30d): **${live['total_pnl']:.2f}**")
    md.append(f"- Backtest source: `backtest/results/tune_180d_pass1.json` ({BT_DAYS}d)")
    md.append(f"- Divergence flag threshold: |ratio - 1| > {DIVERGENCE_THRESH}")
    md.append("")
    md.append("## Top 10 Most-Divergent Symbols")
    md.append("")
    md.append("| Sym | Live trades | Live WR% | Live PnL | Live $/day | BT trades | BT PF | BT $/day | Ratio | Divergence | Flag |")
    md.append("|-----|------------:|---------:|---------:|-----------:|----------:|------:|---------:|------:|-----------:|:----:|")
    for r in top:
        md.append(
            f"| {r['sym']} | {r['live_30d_trades']} | "
            f"{('%.1f' % r['live_30d_wr']) if r['live_30d_wr'] is not None else '-'} | "
            f"{r['live_30d_pnl']:.2f} | {r['live_per_day']:.3f} | "
            f"{r['backtest_180d_trades']} | "
            f"{('%.2f' % r['backtest_180d_pf']) if r['backtest_180d_pf'] is not None else '-'} | "
            f"{r['backtest_per_day']:.3f} | "
            f"{('%.2f' % r['ratio']) if r['ratio'] is not None else 'inf'} | "
            f"{('%.2f' % r['divergence']) if r['divergence'] is not None else 'inf'} | "
            f"{'YES' if r['divergence_flag'] else ''} |"
        )

    md.append("")
    md.append("## All Symbols (sorted by abs divergence desc)")
    md.append("")
    md.append("| Sym | Live trades | Live WR% | Live PnL | Live $/day | BT trades | BT PF | BT WR% | BT $/day | Ratio | Divergence | Flag |")
    md.append("|-----|------------:|---------:|---------:|-----------:|----------:|------:|-------:|---------:|------:|-----------:|:----:|")
    for r in sorted_rows:
        md.append(
            f"| {r['sym']} | {r['live_30d_trades']} | "
            f"{('%.1f' % r['live_30d_wr']) if r['live_30d_wr'] is not None else '-'} | "
            f"{r['live_30d_pnl']:.2f} | {r['live_per_day']:.3f} | "
            f"{r['backtest_180d_trades']} | "
            f"{('%.2f' % r['backtest_180d_pf']) if r['backtest_180d_pf'] is not None else '-'} | "
            f"{('%.1f' % r['backtest_180d_wr']) if r['backtest_180d_wr'] is not None else '-'} | "
            f"{r['backtest_per_day']:.3f} | "
            f"{('%.2f' % r['ratio']) if r['ratio'] is not None else 'inf'} | "
            f"{('%.2f' % r['divergence']) if r['divergence'] is not None else 'inf'} | "
            f"{'YES' if r['divergence_flag'] else ''} |"
        )

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"live_trades_30d={live['total_trades']} live_pnl_30d={live['total_pnl']:.2f}")


if __name__ == "__main__":
    main()
