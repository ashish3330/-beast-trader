"""Read-only: pull MT5 deal history since a date, attribute each deal to a
strategy via its magic offset, and report per (strategy, symbol) P/L.
Does NOT call login() — relies on the already-initialized bridge session."""
import sys
from datetime import datetime, timezone
from collections import defaultdict
from mt5linux import MetaTrader5

HOST, PORT = "localhost", 18813

# symbol -> base magic (from config.py SYMBOLS)
BASE = {
    8100: "XAUUSD", 8370: "EURUSD", 8250: "UK100.r", 8130: "BTCUSD",
    8320: "DJ30.r", 8470: "US2000.r", 8240: "SP500.r", 8200: "GER40.r",
    # legacy / commented but may have old deals:
    8210: "NAS100.r", 8230: "JPN225ft", 8500: "SPI200.r", 8440: "SWI20.r",
    8140: "ETHUSD", 8260: "AUDJPY", 8150: "XPTUSD.r", 8280: "CHFJPY",
    8480: "USOUSD", 8380: "USDCAD", 8390: "USDJPY", 8460: "UKOUSD",
}
OFFSET_STRAT = {
    0: "momentum", 1: "momentum", 2: "momentum",
    500: "scalp", 501: "scalp",
    1000: "FVG", 1001: "FVG",
    2000: "SR", 2001: "SR",
    3000: "SMABO/ASAT/OB", 3001: "SMABO/ASAT/OB", 3002: "SMABO/ASAT/OB",
    4000: "FIB50", 4001: "FIB50",
}

def attribute(magic):
    if magic == 0:
        return "manual"
    for base in BASE:
        off = magic - base
        if off in OFFSET_STRAT:
            return OFFSET_STRAT[off]
    return f"magic{magic}"

since = datetime(2026, 6, 22, 0, 0, tzinfo=timezone.utc)
now = datetime(2026, 6, 25, 0, 0, tzinfo=timezone.utc)

mt5 = MetaTrader5(host=HOST, port=PORT, timeout=30)
deals = mt5.history_deals_get(since, now)
if deals is None:
    print("history_deals_get returned None; last_error:", mt5.last_error())
    sys.exit(1)

# aggregate by (strategy, symbol) using the deal's OWN symbol field.
agg = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
comments = defaultdict(lambda: defaultdict(int))  # (strat,sym) -> comment -> count
for d in deals:
    if d.entry == 1:  # realized close
        strat = attribute(d.magic)
        sym = d.symbol or "?"
        net = d.profit + d.swap + d.commission
        k = (strat, sym)
        agg[k]["n"] += 1
        agg[k]["pnl"] += net
        if net > 0:
            agg[k]["wins"] += 1
        comments[k][(d.comment or "").strip()] += 1

print(f"{'STRATEGY':<16}{'SYMBOL':<12}{'N':>4}{'WINS':>6}{'WR%':>7}{'NET_PnL':>12}")
print("-" * 57)
tot = 0.0
for (strat, sym), v in sorted(agg.items(), key=lambda x: (x[0][0], -x[1]["pnl"])):
    wr = 100 * v["wins"] / v["n"] if v["n"] else 0
    tot += v["pnl"]
    print(f"{strat:<16}{sym:<12}{v['n']:>4}{v['wins']:>6}{wr:>6.0f}%{v['pnl']:>12.2f}")
print("-" * 57)
print(f"{'TOTAL':<32}{'':>10}{tot:>14.2f}")

print("\n=== comments per (strategy,symbol) for disambiguation ===")
for k in sorted(comments):
    cs = ", ".join(f"'{c}'×{n}" for c, n in comments[k].items())
    print(f"{k[0]:<16}{k[1]:<12} {cs}")
