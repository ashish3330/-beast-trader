#!/usr/bin/env python3 -B
"""
RL daily summary — run anytime to see how the bot is getting smarter.

Outputs:
  - 24h trade counts + RL learning events
  - Per-symbol weight evolution (current vs default)
  - Trail multiplier adaptations
  - Recent REVERT/UN_REVERT activity
  - Exit-reason intelligence (which exits work for which symbols)
"""
import sqlite3, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ACTIVE = ['DJ30.r','JPN225ft','SPI200.r','SWI20.r','US2000.r',
          'XAUUSD','BTCUSD','AUDJPY','EURUSD']

c = sqlite3.connect('data/rl_learner.db')
cur = c.cursor()
now = time.time()

print("\n" + "="*70)
print(f"  RL DAILY SUMMARY  {time.strftime('%Y-%m-%d %H:%M')}")
print("="*70)

# 1. Data flow
print("\n[1] TRADE DATA INGESTION")
for label, secs in [("Last 24h", 86400), ("Last 7d", 604800), ("Last 30d", 2592000)]:
    n = cur.execute(f'SELECT COUNT(*) FROM trade_outcomes WHERE ts > {now - secs}').fetchone()[0]
    n_active = cur.execute(f"""SELECT COUNT(*) FROM trade_outcomes WHERE ts > {now - secs}
                                AND symbol IN ({','.join('?'*len(ACTIVE))})""", ACTIVE).fetchone()[0]
    print(f"  {label:<12}: {n:>4} total  /  {n_active:>4} active universe")

# 2. Learning events 24h
print("\n[2] LEARNING EVENTS (last 24h)")
for action in ['WEIGHT_UPDATE','REVERT','UN_REVERT','EXIT_UPDATE','REGIME_WEIGHT_UPDATE']:
    n = cur.execute(f"""SELECT COUNT(*) FROM rl_audit_log
                         WHERE action LIKE '{action}%' AND timestamp > datetime('now','-24 hours')""").fetchone()[0]
    icon = "✓" if action == "WEIGHT_UPDATE" and n > 0 else "↻" if action == "REVERT" else "→"
    print(f"  {icon} {action:<25}: {n}")

# 3. Per-symbol score weights drift from defaults
print("\n[3] PER-SYMBOL SCORE WEIGHTS (drift from 1.0 default)")
print(f"  {'symbol':<12} {'n_cells':>7} {'min_w':>6} {'max_w':>6} {'mean':>6} {'wins':>5} {'losses':>7}")
for sym in ACTIVE:
    row = cur.execute("""SELECT COUNT(*), MIN(weight), MAX(weight), AVG(weight),
                          SUM(win_count), SUM(loss_count)
                          FROM score_weights WHERE symbol=?""", (sym,)).fetchone()
    n, mn, mx, mean, w, l = row if row else (0,1,1,1,0,0)
    drift = "" if n == 0 else (" learning" if abs((mean or 1) - 1.0) < 0.05 else " ADAPTED")
    print(f"  {sym:<12} {n or 0:>7} {(mn or 1):>6.2f} {(mx or 1):>6.2f} {(mean or 1):>6.2f} "
          f"{w or 0:>5} {l or 0:>7}{drift}")

# 4. Trail adaptations
print("\n[4] TRAIL MULTIPLIER ADAPTATIONS (1.0 = default, <1.0 = tighter)")
print(f"  {'symbol':<12} {'lock_m':>7} {'be_m':>5} {'tight_m':>8} {'updated':>20}")
for sym in ACTIVE:
    row = cur.execute("""SELECT lock_threshold_mult, be_threshold_mult, trail_tightness_mult,
                          datetime(updated,'localtime') FROM trail_adjustments
                          WHERE symbol=?""", (sym,)).fetchone()
    if row:
        adapted = " ADAPTED" if row[0] != 1.0 or row[2] != 1.0 else " default"
        print(f"  {sym:<12} {row[0]:>7.2f} {row[1]:>5.2f} {row[2]:>8.2f} {row[3]:>20}{adapted}")
    else:
        print(f"  {sym:<12}  (no learned trail yet — using defaults)")

# 5. Recent REVERT/UN_REVERT activity
print("\n[5] RECENT REVERT / UN_REVERT EVENTS (last 7d)")
for r in cur.execute("""SELECT datetime(timestamp,'localtime'), symbol, action, detail
                         FROM rl_audit_log
                         WHERE action IN ('REVERT','UN_REVERT')
                         AND timestamp > datetime('now','-7 days')
                         ORDER BY id DESC LIMIT 12""").fetchall():
    print(f"  {r[0]} {r[1]:<12} {r[2]:<10} {(r[3] or '')[:50]}")

# 6. Exit-reason intelligence
print("\n[6] WHAT EXITS WORK PER SYMBOL (count ≥ 5)")
print(f"  {'symbol':<12} {'exit':<22} {'n':>3} {'avgR':>7} {'best':>6} {'worst':>7}")
for r in cur.execute(f"""SELECT symbol, exit_reason, count, avg_r, best_r, worst_r
                          FROM exit_learning
                          WHERE symbol IN ({','.join('?'*len(ACTIVE))})
                          AND count >= 5
                          ORDER BY avg_r DESC LIMIT 15""", ACTIVE).fetchall():
    edge = "++" if r[3] > 1.0 else "+" if r[3] > 0 else "-"
    print(f"  {r[0]:<12} {r[1]:<22} {r[2]:>3} {r[3]:>+7.2f} {r[4]:>+6.2f} {r[5]:>+7.2f} {edge}")

# 7. Health score
print("\n[7] OVERALL RL HEALTH")
n24 = cur.execute("SELECT COUNT(*) FROM trade_outcomes WHERE ts > ?", (now-86400,)).fetchone()[0]
wu = cur.execute("""SELECT COUNT(*) FROM rl_audit_log WHERE action='WEIGHT_UPDATE'
                     AND timestamp > datetime('now','-24 hours')""").fetchone()[0]
rv = cur.execute("""SELECT COUNT(*) FROM rl_audit_log WHERE action='REVERT'
                     AND timestamp > datetime('now','-24 hours')""").fetchone()[0]
adapted_syms = sum(1 for s in ACTIVE if (cur.execute(
    "SELECT trail_tightness_mult FROM trail_adjustments WHERE symbol=?", (s,)).fetchone() or (1,))[0] != 1.0)
print(f"  Trades/24h: {n24}        WEIGHT_UPDATES/24h: {wu}        REVERTs/24h: {rv}")
print(f"  Symbols with adapted trail: {adapted_syms}/{len(ACTIVE)}")
print(f"  Score: ", end="")
if n24 > 10 and wu > 0 and rv < wu * 2 and adapted_syms > 3:
    print("HEALTHY — bot is actively learning and improving")
elif n24 > 5:
    print("LEARNING — accumulating data, weights stabilizing")
else:
    print("WARMUP — need more trades to drive learning")

print()
c.close()
