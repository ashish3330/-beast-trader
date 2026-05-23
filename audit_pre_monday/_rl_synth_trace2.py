"""Second synthetic trace: drive RULE 1 (high giveback) to fire
regime_trail_adjustments writer. Verifies TASK J writer path works."""
import sys, shutil, sqlite3, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

src = ROOT / 'data' / 'rl_learner.db'
tmpdir = Path(tempfile.mkdtemp(prefix='rlauditro2_'))
dst = tmpdir / 'rl_learner.db'
shutil.copy(src, dst)

import agent.rl_learner as rlmod
rlmod.RL_DB = Path(dst)

class FakeState: pass
rl = rlmod.RLLearner(FakeState())

# Reset target symbol
sym = 'DJ30.r'
rl._trade_outcomes[sym] = []
rl._weights[sym] = dict(rlmod.DEFAULT_WEIGHTS)
rl._trail_adjustments[sym] = {
    'lock_threshold_mult': 1.0, 'be_threshold_mult': 1.0, 'trail_tightness_mult': 1.0
}

# Inject 15 trades in 'trending' regime: half wins, ALL with high peak (2R)
# but exited at 0.5R → giveback = 1.5R → 75% of peak → RULE 1 fires
print('=== Synthetic high-giveback (RULE 1) ===')
for i in range(15):
    won = (i % 2 == 0)  # 50% WR, neutral so no weight update fires
    peak = 2.0
    rm = 0.5 if won else -1.0   # high giveback on wins
    pnl = 5.0 if won else -10.0
    rl.record_outcome(
        symbol=sym, direction='LONG', pnl=pnl, r_multiple=rm,
        score=7.0, regime='trending', exit_reason='SL',
        score_components={c: 1.0 for c in rlmod.SCORE_COMPONENTS}, peak_r=peak,
    )

print(f'  global trail_tightness_mult: {rl._trail_adjustments[sym]["trail_tightness_mult"]}')
print(f'  global lock_threshold_mult: {rl._trail_adjustments[sym]["lock_threshold_mult"]}')
print(f'  regime_trail[DJ30.r][trending]: {rl._regime_trail.get(sym, {}).get("trending")}')

conn = sqlite3.connect(str(dst))
print('\n[db] regime_trail_adjustments:')
for r in conn.execute("SELECT * FROM regime_trail_adjustments"):
    print(' ', r)
print('\n[db] trail_adjustments:')
for r in conn.execute("SELECT * FROM trail_adjustments WHERE symbol=?", (sym,)):
    print(' ', r)
print('\n[db] rl_audit_log REGIME_EXIT_UPDATE entries:')
for r in conn.execute("SELECT symbol, action, detail FROM rl_audit_log WHERE action LIKE 'REGIME_EXIT_UPDATE%' ORDER BY id DESC LIMIT 5"):
    print(' ', r)
print('\n[db] rl_audit_log EXIT_UPDATE entries:')
for r in conn.execute("SELECT symbol, action, detail FROM rl_audit_log WHERE action='EXIT_UPDATE' ORDER BY id DESC LIMIT 3"):
    print(' ', r)
conn.close()

# Now verify the read path: get_trail_adjustments(sym, regime='trending') uses regime overlay
print('\n=== Read path: get_trail_adjustments(sym, regime=trending) ===')
no_reg = rl.get_trail_adjustments(sym)
with_reg = rl.get_trail_adjustments(sym, regime='trending')
print(f'  no regime arg:   {no_reg}')
print(f'  regime=trending: {with_reg}')

# Verify executor would NOT see regime overlay (brain doesn't pass regime= per H3)
print('\n=== Confirm brain.py:927 call path: WITHOUT regime — does it see regime trail? ===')
print(f'  result: {rl.get_trail_adjustments(sym)} — should equal global trail not regime trail')

shutil.rmtree(str(tmpdir), ignore_errors=True)
print('\nDone.')
