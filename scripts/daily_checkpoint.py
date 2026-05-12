#!/usr/bin/env python3 -B
"""Daily checkpoint — single-screen honest snapshot of the trader.

Usage: python3 -B scripts/daily_checkpoint.py
"""
import json
import sqlite3
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "trade_journal.db"


def fetch_dashboard():
    try:
        return json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8888/api/data", timeout=3).read())
    except Exception as e:
        return {"error": str(e)}


def fetch_connection_health():
    try:
        return json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8888/api/connection_health", timeout=3).read())
    except Exception:
        return {}


def trades_today():
    """Closed trades since midnight UTC."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with sqlite3.connect(str(DB)) as c:
        rows = c.execute(
            "SELECT symbol, direction, pnl, r_multiple, exit_reason "
            "FROM trades WHERE timestamp >= ? ORDER BY id DESC",
            (today,)
        ).fetchall()
    return rows


def trades_24h():
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    with sqlite3.connect(str(DB)) as c:
        return c.execute(
            "SELECT count(*), sum(pnl), "
            "sum(CASE WHEN pnl>0 THEN 1 ELSE 0 END), "
            "sum(CASE WHEN pnl>0 THEN pnl ELSE 0 END), "
            "sum(CASE WHEN pnl<0 THEN -pnl ELSE 0 END) "
            "FROM trades WHERE timestamp > ?",
            (cutoff,)
        ).fetchone()


def services():
    out = subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True
    ).stdout
    rows = []
    for line in out.splitlines():
        if "com.dragon" in line:
            parts = line.split()
            pid = parts[0]
            exit_code = parts[1]
            name = parts[2]
            rows.append((name, pid, exit_code))
    return rows


def fmt_money(v): return f"${v:+,.2f}" if v else "$0.00"


def main():
    print()
    print("═" * 60)
    print(f"  DRAGON DAILY CHECKPOINT — {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}")
    print("═" * 60)

    # Account
    d = fetch_dashboard()
    if "error" in d:
        print(f"\n  ✗ Dashboard unreachable: {d['error']}")
    else:
        eq = d.get("equity", 0)
        bal = d.get("balance", 0)
        floating = d.get("profit", 0)
        dd = d.get("dd_pct", 0)
        print(f"\n  ACCOUNT  equity ${eq:,.2f}   balance ${bal:,.2f}   "
              f"floating {fmt_money(floating)}   dd {dd:.2f}%")

    # 24h trades
    n, pnl, wins, gross_p, gross_l = trades_24h()
    n = n or 0
    pnl = pnl or 0
    wins = wins or 0
    gross_p = gross_p or 0
    gross_l = gross_l or 0.01
    pf = gross_p / gross_l
    wr = (wins / n * 100) if n else 0
    print(f"\n  24h PnL  {fmt_money(pnl)}   trades {n}   WR {wr:.0f}%   PF {pf:.2f}")
    print(f"  expected ~25/day from backtest projection")

    # Today
    rows = trades_today()
    if rows:
        total_pnl = sum(r[2] for r in rows)
        wins_today = sum(1 for r in rows if r[2] > 0)
        print(f"\n  TODAY    {len(rows)} trades   "
              f"{wins_today}W/{len(rows)-wins_today}L   "
              f"PnL {fmt_money(total_pnl)}")
        # Top 3 winners + losers
        rows_by_pnl = sorted(rows, key=lambda r: r[2])
        if len(rows) >= 2:
            print("    worst:", end=" ")
            for sym, dr, p, r, ex in rows_by_pnl[:3]:
                print(f"{sym}({fmt_money(p)} R={r:.1f})", end="  ")
            print()
            print("    best: ", end=" ")
            for sym, dr, p, r, ex in rows_by_pnl[-3:]:
                print(f"{sym}({fmt_money(p)} R={r:.1f})", end="  ")
            print()

    # Open positions
    positions = d.get("positions", [])
    if positions:
        print(f"\n  OPEN     {len(positions)} positions, floating "
              f"{fmt_money(sum(p['pnl'] for p in positions))}")
        for p in sorted(positions, key=lambda x: x["pnl"]):
            sym = p["symbol"]
            t = p["type"]
            vol = p["volume"]
            opp = p["price_open"]
            pnl = p["pnl"]
            print(f"    {sym:<10} {t:<5} {vol:>5} @ {opp:>10.4f}   "
                  f"floating {fmt_money(pnl)}")

    # Connection health
    ch = fetch_connection_health()
    if ch:
        rc24 = ch.get("reconnects_24h", 0)
        deg = ch.get("degraded_streak", 0)
        status = ch.get("status", "?")
        print(f"\n  CONN     {status}   reconnects/24h: {rc24}   "
              f"degraded streak: {deg}")

    # Services
    svcs = services()
    running = [s for s in svcs if s[1] != "-"]
    scheduled = [s for s in svcs if s[1] == "-"]
    bad_exit = [s for s in svcs if s[2] not in ("0", "-")]
    print(f"\n  SERVICES  running: {len(running)}   scheduled: {len(scheduled)}   "
          f"failed: {sum(1 for _, _, ec in svcs if ec not in ('0', '-'))}")
    if bad_exit:
        for name, pid, ec in bad_exit:
            print(f"    ⚠ {name:<30}  exit={ec}")

    print()


if __name__ == "__main__":
    main()
