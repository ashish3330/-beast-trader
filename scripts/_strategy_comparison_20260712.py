#!/usr/bin/env python3 -B
"""3-week strategy PnL comparison — run at T+3wk after fresh-start reset.

Reads scripts/_comparison_baseline_20260621.json for the baseline timestamp,
queries MT5 history_deals_get from that timestamp onwards, groups by magic
range → strategy bucket, and prints a per-strategy leaderboard.

Magic ranges (per SETUP.txt + executor._strategy_for_magic):
  base + 0        → momentum
  base + 1000     → fvg
  base + 2000     → sr
  base + 3000     → smabo
  base + 4000     → fib50  (TRADE_LIVE=False — no live trades; signals only)

CLI:
    python3 -B scripts/_strategy_comparison_20260712.py
    python3 -B scripts/_strategy_comparison_20260712.py --from-iso 2026-06-21T11:00Z

Output columns per strategy:
    trades, wins, WR, total PnL ($), avg PnL/trade, biggest win/loss,
    win streak, lose streak, profit factor
"""
import argparse
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def magic_to_strategy(magic_int, base_magics):
    """Map a magic to strategy bucket, given the set of base magics per sym."""
    try:
        m = int(magic_int)
    except Exception:
        return "unknown"
    for base in base_magics:
        off = m - base
        if off < 0 or off >= 5000:
            continue
        if 4000 <= off < 5000:
            return "fib50"
        if 3000 <= off < 4000:
            return "smabo"
        if 2000 <= off < 3000:
            return "sr"
        if 1000 <= off < 2000:
            return "fvg"
        if 0 <= off < 1000:
            return "momentum"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json",
                        default=str(ROOT / "scripts" / "_comparison_baseline_20260621.json"))
    parser.add_argument("--from-iso", default=None,
                        help="Override baseline ISO timestamp.")
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline_json).read_text())
    from_iso = args.from_iso or baseline["reset_at_iso"]
    from_unix = datetime.datetime.fromisoformat(from_iso.replace("Z", "+00:00")).timestamp()
    print(f"Baseline: {from_iso}  (account {baseline['mt5_account']})")
    print(f"Baseline equity: ${baseline['baseline_equity']:.2f}")

    # Connect to MT5 + pull current equity + all deals since baseline
    from mt5linux import MetaTrader5
    from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, SYMBOLS
    mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    info = mt5.account_info()
    cur_eq = float(info.equity); cur_bal = float(info.balance)
    elapsed_d = (datetime.datetime.now(datetime.timezone.utc).timestamp() - from_unix) / 86400
    print(f"\nCurrent equity: ${cur_eq:.2f}  balance: ${cur_bal:.2f}")
    print(f"Elapsed since baseline: {elapsed_d:.1f} days")
    print(f"Account PnL:    ${cur_eq - baseline['baseline_equity']:+.2f}  "
          f"({(cur_eq/baseline['baseline_equity']-1)*100:+.2f}%)")

    # Pull deals from baseline to now
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    from_dt = datetime.datetime.fromtimestamp(from_unix, datetime.timezone.utc)
    deals = mt5.history_deals_get(from_dt, now_dt) or []
    print(f"\nDeals fetched: {len(deals)}")
    mt5.shutdown()

    # Strategy bucketing — need base magics from SYMBOLS dict
    base_magics = set()
    for sym, cfg in SYMBOLS.items():
        try:
            base_magics.add(int(cfg.magic))
        except Exception:
            pass

    # Group deals by strategy + symbol
    buckets = defaultdict(lambda: {"trades": [], "pnl": 0.0})
    by_sym = defaultdict(lambda: defaultdict(lambda: {"trades": 0, "pnl": 0.0}))
    for d in deals:
        # Only count entry/exit deals (type 0=buy, 1=sell). Skip credit/balance.
        if int(d.type) not in (0, 1):
            continue
        # Only count "close" deals (entry=0 is open, entry=1 is close).
        if int(d.entry) != 1:
            continue
        strat = magic_to_strategy(d.magic, base_magics)
        buckets[strat]["trades"].append(d)
        buckets[strat]["pnl"] += float(d.profit)
        by_sym[strat][d.symbol]["trades"] += 1
        by_sym[strat][d.symbol]["pnl"] += float(d.profit)

    print("\n" + "=" * 90)
    print("STRATEGY LEADERBOARD")
    print("=" * 90)
    print(f"{'Strategy':<12} {'Trd':>5} {'Wins':>5} {'WR':>6} "
          f"{'PnL $':>9} {'avg $':>7} {'best':>8} {'worst':>8} {'PF':>5}")
    print("-" * 90)
    summary = {}
    for strat in ("momentum", "fvg", "sr", "smabo", "fib50", "other"):
        if strat not in buckets:
            continue
        b = buckets[strat]
        pnls = [float(d.profit) for d in b["trades"]]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = (len(wins) / len(pnls) * 100) if pnls else 0
        total = sum(pnls)
        avg = total / len(pnls) if pnls else 0
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) < 0 else float("inf")
        best = max(pnls) if pnls else 0
        worst = min(pnls) if pnls else 0
        print(f"{strat:<12} {len(pnls):>5} {len(wins):>5} {wr:>5.1f}% "
              f"{total:>+8.2f} {avg:>+7.2f} {best:>+8.2f} {worst:>+8.2f} "
              f"{pf:>5.2f}")
        summary[strat] = {"trades": len(pnls), "wins": len(wins), "wr_pct": wr,
                          "pnl_usd": total, "avg_pnl": avg, "best": best,
                          "worst": worst, "pf": pf}

    # Per-sym breakdown for the live strategies
    print("\n" + "=" * 90)
    print("PER-SYMBOL BREAKDOWN")
    print("=" * 90)
    for strat in ("momentum", "fvg", "sr", "smabo"):
        if strat not in by_sym or not by_sym[strat]:
            continue
        print(f"\n[{strat.upper()}]")
        rows = sorted(by_sym[strat].items(),
                      key=lambda kv: kv[1]["pnl"], reverse=True)
        for sym, s in rows:
            print(f"  {sym:<12} trades={s['trades']:>3}  pnl=${s['pnl']:>+8.2f}")

    # Verdict
    print("\n" + "=" * 90)
    if summary:
        ranked = sorted(summary.items(), key=lambda kv: kv[1]["pnl_usd"], reverse=True)
        print(f"WINNER: {ranked[0][0].upper()}  PnL=${ranked[0][1]['pnl_usd']:+.2f}  "
              f"PF={ranked[0][1]['pf']:.2f}")
        print("\nFull ranking:")
        for i, (s, st) in enumerate(ranked, 1):
            print(f"  {i}. {s:<10} ${st['pnl_usd']:>+8.2f}  PF {st['pf']:.2f}  "
                  f"{st['trades']} trades  WR {st['wr_pct']:.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
