"""READ-ONLY synthetic learn trace for RL audit 3/6.
Copies rl_learner.db to a tmp path so the real DB is untouched, then
injects synthetic outcomes through the real RLLearner code path."""
import sys, os, shutil, sqlite3, tempfile, json
from pathlib import Path

# Ensure repo root on path (this file lives under audit_pre_monday)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

src = ROOT / 'data' / 'rl_learner.db'
tmpdir = Path(tempfile.mkdtemp(prefix='rlauditro_'))
dst = tmpdir / 'rl_learner.db'
shutil.copy(src, dst)
print(f'Temp DB copy: {dst}')

# Rebind RL_DB inside the module so all RLLearner DB IO goes to tmp.
import agent.rl_learner as rlmod
rlmod.RL_DB = Path(dst)

# Sanity print
print(f'RL_DB rebound: {rlmod.RL_DB}')

class FakeState:
    pass

rl = rlmod.RLLearner(FakeState())
print(f'\n[init] _reverted_at present: {hasattr(rl, "_reverted_at")} type={type(rl._reverted_at).__name__}')
print(f'[init] _weights["DJ30.r"]["ema_stack"]={rl._weights["DJ30.r"]["ema_stack"]:.3f}')
print(f'[init] reverted[DJ30.r]={rl._reverted.get("DJ30.r")}')
print(f'[init] n_updates={rl._n_updates}, n_regime_updates={rl._n_regime_updates}')

# Reset the symbol's outcomes so synthetic test starts clean
rl._trade_outcomes['DJ30.r'] = []
rl._weights['DJ30.r'] = dict(rlmod.DEFAULT_WEIGHTS)
rl._last_update_ts.pop('DJ30.r', None)
rl._reverted['DJ30.r'] = False
rl._reverted_at.pop('DJ30.r', None)

print('\n=== Synthetic 1: WEIGHT_UPDATE learning ===')
# Inject 15 trades. ema_stack always on; supertrend always on; rsi sometimes on
# Design WR-by-component so ema_stack/supertrend should boost (WR>55%)
# and rsi/macd_hist should be neutral or reduce.
import time as _time

def _inject(rl, symbol, regime, want_won, components, peak_r=2.0):
    pnl = 20.0 if want_won else -10.0
    rm = 2.0 if want_won else -1.0
    rl.record_outcome(
        symbol=symbol, direction='LONG', pnl=pnl, r_multiple=rm,
        score=7.0, regime=regime, exit_reason='TP_T1',
        score_components=components, peak_r=peak_r if want_won else 0.5,
    )

COMPS = rlmod.SCORE_COMPONENTS
# 15 trades: ema_stack 11W/4L (73%), supertrend 11W/4L (73%), rsi 2W/8L (20%)
plan = []
for i in range(15):
    won = (i % 5 != 4)  # 12/15 wins ~ 80%
    comps = {c: 0.0 for c in COMPS}
    comps['ema_stack'] = 1.5
    comps['supertrend'] = 1.5
    # rsi flips pattern: when rsi is on, force LOSS most of the time
    rsi_on = (i % 3 == 0)
    if rsi_on:
        comps['rsi'] = 1.0
        won = (i in (0, 9))   # only 2/5 wins when rsi on
    plan.append((won, comps))

for i, (won, comps) in enumerate(plan):
    _inject(rl, 'DJ30.r', 'trending', won, comps)

print(f'[after] _n_updates={rl._n_updates}, _n_regime_updates={rl._n_regime_updates}')
print(f'[after] outcomes tracked: {len(rl._trade_outcomes["DJ30.r"])}')
print(f'[after] rolling_pf: {rl._rolling_pf.get("DJ30.r")}')
print()
print('[after] Final per-component weights:')
for c, w in rl._weights['DJ30.r'].items():
    moved = (abs(w - 1.0) > 1e-6)
    arrow = ' <-- MOVED' if moved else ''
    print(f'  {c:18s} = {w:.4f}{arrow}')

print('\n[after] Per-regime cells for DJ30.r/trending:')
reg_cells = rl._regime_weights.get('DJ30.r', {}).get('trending', {})
if reg_cells:
    for c, w in reg_cells.items():
        print(f'  {c:18s} = {w:.4f}')
else:
    print('  (none — regime cell needs same threshold)')

# Verify DB writeback
conn = sqlite3.connect(str(dst))
print('\n[db] score_weights for DJ30.r:')
for r in conn.execute("SELECT component, weight, win_count, loss_count FROM score_weights WHERE symbol='DJ30.r' ORDER BY component"):
    print(' ', r)
print('\n[db] regime_weights for DJ30.r:')
for r in conn.execute("SELECT regime, component, weight, win_count, loss_count FROM regime_weights WHERE symbol='DJ30.r'"):
    print(' ', r)
print('\n[db] regime_trail_adjustments for DJ30.r:')
for r in conn.execute("SELECT symbol, regime, lock_threshold_mult, be_threshold_mult, trail_tightness_mult FROM regime_trail_adjustments WHERE symbol='DJ30.r'"):
    print(' ', r)
print('\n[db] exit_learning for DJ30.r:')
for r in conn.execute("SELECT exit_reason, count, avg_r FROM exit_learning WHERE symbol='DJ30.r'"):
    print(' ', r)
print('\n[db] rl_audit_log for DJ30.r (last 8):')
for r in conn.execute("SELECT action, detail FROM rl_audit_log WHERE symbol='DJ30.r' ORDER BY id DESC LIMIT 8"):
    print(' ', r)
conn.close()

print('\n=== Synthetic 2: REVERT cycle ===')
# Now drive PF below 0.5 on USDJPY to trigger SEVERE revert
rl._trade_outcomes['USDJPY'] = []
rl._weights['USDJPY'] = dict(rlmod.DEFAULT_WEIGHTS)
rl._weights['USDJPY']['ema_stack'] = 1.2  # pre-adjusted weight
rl._reverted['USDJPY'] = False
rl._reverted_at.pop('USDJPY', None)
# 12 trades, 1 win, 11 losses; pnl: 1W=+10, 11L=-20 → PF = 10/220 = 0.045
for i in range(12):
    won = (i == 0)
    pnl = 10.0 if won else -20.0
    rm = 1.0 if won else -1.0
    rl.record_outcome(
        symbol='USDJPY', direction='LONG', pnl=pnl, r_multiple=rm,
        score=7.0, regime='trending', exit_reason='SL',
        score_components={c: 1.0 for c in rlmod.SCORE_COMPONENTS}, peak_r=0.5,
    )

print(f'[USDJPY] reverted={rl._reverted.get("USDJPY")} reverted_at={rl._reverted_at.get("USDJPY")}')
print(f'[USDJPY] rolling_pf={rl._rolling_pf.get("USDJPY"):.4f}')
print(f'[USDJPY] weights after revert (should all be 1.0):')
for c, w in rl._weights['USDJPY'].items():
    if abs(w - 1.0) > 1e-6:
        print(f'  *** {c} = {w} (NOT RESET!)')
all_reset = all(abs(w - 1.0) < 1e-6 for w in rl._weights['USDJPY'].values())
print(f'  all reset to 1.0: {all_reset}')

conn = sqlite3.connect(str(dst))
print('\n[db] revert_state:')
for r in conn.execute("SELECT * FROM revert_state"):
    print(' ', r)
print('\n[db] rl_audit_log REVERT entries:')
for r in conn.execute("SELECT symbol, action, detail FROM rl_audit_log WHERE action='REVERT'"):
    print(' ', r)
conn.close()

# Now test restart — instantiate a NEW RLLearner pointing at same DB and verify
# the REVERT state survives.
print('\n=== Synthetic 3: Restart persistence ===')
rl2 = rlmod.RLLearner(FakeState())
print(f'[rl2] reverted[USDJPY]={rl2._reverted.get("USDJPY")}')
print(f'[rl2] reverted_at[USDJPY]={rl2._reverted_at.get("USDJPY")}')
print(f'[rl2] weights[USDJPY][ema_stack] after restart: {rl2._weights.get("USDJPY", {}).get("ema_stack")}')

# Try UN_REVERT: simulate 12 winning trades AFTER the revert
import time as _t
_t.sleep(0.01)
# Bump reverted_at backwards so the > 3600s elapsed gate passes
rl2._reverted_at['USDJPY'] = _t.time() - 7200  # 2 hours ago
for i in range(12):
    rl2.record_outcome(
        symbol='USDJPY', direction='LONG', pnl=20.0, r_multiple=2.0,
        score=7.0, regime='trending', exit_reason='TP',
        score_components={c: 1.0 for c in rlmod.SCORE_COMPONENTS}, peak_r=2.5,
    )
print(f'[after-recovery] reverted[USDJPY]={rl2._reverted.get("USDJPY")}')
print(f'[after-recovery] rolling_pf[USDJPY]={rl2._rolling_pf.get("USDJPY"):.2f}')

# Audit health
print('\n=== Health summary ===')
print(rl2.health_summary(force_log=True))

# Cleanup
shutil.rmtree(str(tmpdir), ignore_errors=True)
print('\nDone. Temp DB removed.')
