#!/usr/bin/env python3 -B
"""
Dragon 7-day post-fix review — fires once on 2026-06-09 09:00 IST via launchd.
Writes report to logs/weekly_review_20260609.md.
"""
import sqlite3, subprocess, json, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "trade_journal.db"
LOG = ROOT / "logs" / "dragon.log"
REPORT = ROOT / "logs" / "weekly_review_20260609.md"
WATCH_START = "2026-06-02"
BASELINE_30D = -49.08  # 2026-06-02 audit baseline

ACTIVE_SYMBOLS = [
    "XAUUSD", "DJ30.r", "JPN225ft", "SPI200.r", "SWI20.r",
    "BTCUSD", "ETHUSD", "AUDJPY", "EURUSD", "NAS100.r",
    "UK100.r", "XPTUSD.r", "CHFJPY", "USOUSD",
]


def query_trades():
    conn = sqlite3.connect(str(DB), timeout=10.0)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT symbol,
               COUNT(*) n,
               ROUND(SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) wr,
               ROUND(SUM(pnl),2) pnl,
               ROUND(SUM(CASE WHEN pnl>0 THEN pnl ELSE 0 END) /
                     NULLIF(ABS(SUM(CASE WHEN pnl<0 THEN pnl ELSE 0 END)),0), 2) pf
        FROM trades WHERE timestamp >= ?
        GROUP BY symbol ORDER BY pnl
        """, (WATCH_START,)
    ).fetchall()
    agg = cur.execute(
        """
        SELECT COUNT(*) n,
               ROUND(SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) wr,
               ROUND(SUM(pnl),2) pnl,
               ROUND(SUM(CASE WHEN pnl>0 THEN pnl ELSE 0 END) /
                     NULLIF(ABS(SUM(CASE WHEN pnl<0 THEN pnl ELSE 0 END)),0), 2) pf
        FROM trades WHERE timestamp >= ?
        """, (WATCH_START,)
    ).fetchone()
    conn.close()
    return rows, agg


def grep_log(pattern):
    if not LOG.exists():
        return -1
    try:
        out = subprocess.run(
            ["grep", "-c", pattern, str(LOG)],
            capture_output=True, text=True, timeout=20
        )
        return int(out.stdout.strip() or "0")
    except Exception:
        return -1


def fetch_health():
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8888/api/connection_health", timeout=5
        ) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def main():
    rows, agg = query_trades()
    t3_immediate = grep_log("EARLY-LOSS-CUT T3-IMMEDIATE")
    hard_floor = grep_log("BLOCK entry: raw_score=")
    same_price = grep_log("BLOCK same-price re-entry")
    health = fetch_health()

    now = datetime.now(timezone.utc).isoformat()
    n, wr, pnl, pf = agg or (0, 0, 0, 0)
    pnl = pnl or 0.0
    delta = pnl - (BASELINE_30D / 30 * 7)  # weekly-normalized baseline

    verdict = "BLEEDING" if pnl < -5 else ("BREAKEVEN" if pnl < 5 else "PROFITABLE")

    lines = [
        f"# Dragon 7-day Post-Fix Review",
        f"Generated: {now}  |  Window: since {WATCH_START}",
        f"",
        f"## Aggregate",
        f"- Trades: **{n}**  |  WR: **{wr}%**  |  PnL: **${pnl}**  |  PF: **{pf}**",
        f"- Baseline (30d to 2026-06-02): -$49.08 → weekly-normalized -$11.45",
        f"- Delta vs baseline: **{delta:+.2f}**  →  **{verdict}**",
        f"",
        f"## Per-symbol PnL",
        f"| Symbol | N | WR | PnL | PF |",
        f"|---|---|---|---|---|",
    ]
    for sym, n_s, wr_s, pnl_s, pf_s in rows:
        active = " ✓" if sym in ACTIVE_SYMBOLS else " (disabled?)"
        lines.append(f"| {sym}{active} | {n_s} | {wr_s}% | ${pnl_s} | {pf_s} |")

    lines += [
        f"",
        f"## Bug-fix telemetry (grep counts in dragon.log)",
        f"- `EARLY-LOSS-CUT T3-IMMEDIATE` (XAU bypass fix firings): **{t3_immediate}**",
        f"- `BLOCK entry: raw_score=` (HARD MIN_SCORE floor catches): **{hard_floor}**",
        f"- `BLOCK same-price re-entry` (DJ30 guard fires): **{same_price}**",
        f"",
        f"## Connection health",
        f"```json",
        json.dumps(health, indent=2)[:1200],
        f"```",
        f"",
        f"## Recommendation rubric",
        f"- If aggregate PnL > +$20 AND no symbol < -$15: **KEEP universe**, watch another week.",
        f"- If PnL still negative but trimmed symbols would flip it positive: **further trim** the worst PF<0.3 symbols.",
        f"- If PnL strongly positive AND disabled symbols (USDJPY/SP500/US2000) show improved backtest in current regime: **restore one** to A/B test.",
        f"- If reconnects 7d > 100 OR status still AMBER: **fix connection** before any code/config changes.",
    ]

    REPORT.write_text("\n".join(lines))
    print(f"WROTE {REPORT}")
    print(f"Aggregate: n={n} wr={wr}% pnl=${pnl} pf={pf} → {verdict}")


if __name__ == "__main__":
    # Date guard: only run on the target date (launchd plist fires June 9 every
    # year; this restricts execution to 2026-06-09 specifically).
    today = datetime.now().strftime("%Y-%m-%d")
    if today != "2026-06-09":
        print(f"SKIP: today={today}, target=2026-06-09")
    else:
        main()
        # Self-cleanup: unload the launchd plist so it doesn't fire again.
        plist = Path.home() / "Library" / "LaunchAgents" / "com.dragon.weekly-review.plist"
        if plist.exists():
            subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
            plist.unlink(missing_ok=True)
            print(f"REMOVED {plist}")
