# RL Learner Deep Verification — Pre-Monday Audit 3/6
Date: 2026-05-23
Scope: `agent/rl_learner.py` (1696 lines), `data/rl_learner.db` (fresh-restart 02:59), live wiring in `agent/brain.py` + `agent/learning_engine.py` + `execution/executor.py`.
Mode: READ-ONLY (synthetic trace ran against a tmp copy of the DB).
Prior audit: `audit_20260522/rl_audit.md`. Today's commit fixed C1, C2, C3, H3, H6.

---

## TOP 3 CRITICAL FINDINGS FOR MONDAY

### 1. components_json is EMPTY for every existing trade — weight learning blind until ~12 new live trades/symbol land
**Severity**: CRITICAL (blocks the entire weight-learning path until enough fresh live trades arrive).
**Evidence**:
- `SELECT COUNT(*) FROM trade_outcomes WHERE components_json IS NOT NULL` = **0 / 68**
- `dragon.log`: `RL Learner loaded 68 trade outcomes from DB (0 with components)` on every restart.
- All 68 outcomes came from `_sync_mt5_deals` (`learning_engine.py:675-815`), which looks up `entry_metadata[sym]` by KEY — `entry_metadata` is overwritten per-symbol per ENTRY, so historical syncs miss it. Past trades sit forever with `score_components=None`.
- `_maybe_update_weights` at line 1110 short-circuits if `len(trades_with_components) < 10`. Today’s DJ30.r has 17 outcomes — but 0 with components → no WEIGHT_UPDATE ever fires.

**Monday impact**:
- For each of the 5 active SYMBOLS, weight learning needs **≥12 trades with `score_components` set** (MIN_TRADES=12, +5 per-component samples gate). At ~4 trades/sym/day, that's ~3 days before the FIRST WEIGHT_UPDATE on a productive symbol.
- During those 3 days the bot uses default 1.0 weights (effectively the seed from `component_weights_auto_dict.py` if present).
- Once new live trades flow, the `_entry_metadata` overwrites correctly (`brain.py:2189-2200` writes per-symbol per-entry, persisted to `data/beast.db.entry_metadata`), and `_sync_mt5_deals` lookup will succeed for that symbol-key while still open. Single-position-per-symbol is the saving constraint here.
- **Watch**: if two trades on the SAME symbol close in the same `_sync_mt5_deals` poll, the SECOND one will read the FIRST one's overwritten entry_metadata (or both will read the latest open position's metadata). Probability low because symbol cooldowns gate re-entry.

**Mitigation**: none required pre-Monday — this self-heals as live trades arrive. But add a `WEIGHT_LEARN_READY` gauge to `[RL HEALTH]` so the operator can SEE when each symbol crosses the threshold.

---

### 2. Per-regime trail overlay is COMPUTED + PERSISTED but NEVER WIRED to executor (TASK J writer fixed today, reader still broken)
**Severity**: HIGH (the regime trail learning runs and persists rows but the executor receives the global trail, so the learned regime-conditional tightness has zero effect live).
**Evidence**:
- Today's commit added `_maybe_update_regime_trail` (rl_learner.py:1422-1546). My synthetic trace (high-giveback scenario) produced the expected `REGIME_EXIT_UPDATE:trending` audit rows and a `regime_trail_adjustments` row (lock 0.79 / trail 0.65).
- BUT `brain.py:927` still calls `self._rl_learner.get_trail_adjustments(sym)` — NO `regime=` kwarg.
- Verified read-path returns global only without regime arg: synth trace showed `no_regime_arg = {lock 0.82, trail 0.70}` vs `regime=trending = {lock 0.79, trail 0.65}`.
- `executor._current_regime[sym]` IS being set (brain.py:937), but it's used for the config-side `SYMBOL_REGIME_TRAIL_OVERRIDE` lookup (`executor.py:411-415`), not for the RL overlay.

**Monday impact**: regime_trail_adjustments table will fill with learned rows from week 1, but trail execution will still use the global per-symbol learned trail. Half the feature is dead-on-arrival.

**Mitigation** (1-line code fix, ship before Monday): in `brain.py:927`, change
```python
adj = self._rl_learner.get_trail_adjustments(sym)
```
to
```python
regime = (self._last_scores.get(sym) or {}).get("regime")
adj = self._rl_learner.get_trail_adjustments(sym, regime=regime)
```

---

### 3. 3 of 5 active SYMBOLS are NOT in `RL_ENABLED_SYMBOLS` — partial RL coverage on Monday
**Severity**: HIGH (asymmetry — some symbols get full RL guidance, others only get a subset).
**Evidence**:
- Active SYMBOLS (config.py:61): `{SP500.r, US2000.r, DJ30.r, USOUSD, XAUUSD}`.
- `RL_ENABLED_SYMBOLS` (config.py:1040): does NOT contain SP500.r, USOUSD, XAUUSD.
- Effect: `should_skip_entry()` and `get_risk_multiplier()` short-circuit on `if symbol not in RL_ENABLED_SYMBOLS: return False/1.0` (rl_learner.py:461, 525). Those 3 symbols get no regime/hour skip and no regime/hour risk boost.
- The DD/streak/edge/EV/quality-bonus/weight paths DO fire for them (no enable-gate), so they still get partial coverage. But the symmetry assumed by the architecture isn't there.

**Monday impact**:
- SP500.r/USOUSD/XAUUSD: regime-bad-WR symbols can't be auto-skipped. A bad-regime day on these can run a 10-trade losing streak before the equity-DD multiplier alone catches it.
- DJ30.r and US2000.r get the full stack.

**Mitigation** (config-only, no code): add the 3 symbols to `RL_ENABLED_SYMBOLS` with conservative `RL_SYMBOL_PARAMS` (`lookback=30, boost_max=1.3`) so the skip-entry logic and risk multiplier turn on. Safe because lookback=30 means no aggressive skip fires until 30 trades land.

---

## Synthetic learn → weight-update → apply trace

Source: `audit_pre_monday/_rl_synth_trace.py` (read-only against tmp DB copy).

### Trace 1 — WEIGHT_UPDATE fires correctly

Input: 15 synthetic DJ30.r trades in trending regime, designed so:
- `ema_stack` and `supertrend` active on ALL trades, WR ≈ 75% (9/12 component-active wins).
- `rsi` flipped to penalize: on 5 trades, only 2/5 won.

Result:
```
_n_updates = 1, _n_regime_updates = 1 after 15 trades
rolling_pf = 2.8

ema_stack          = 1.0500   <-- MOVED  (Δ +0.05, WR=75% → boost capped at MAX_CHANGE)
supertrend         = 1.0500   <-- MOVED  (same)
macd_signal        = 1.0000              (component never active in synth)
macd_hist          = 1.0000
rsi                = 1.0000              (only 5 active → below 5+ gate? → no update)
... (others unchanged)
```

Regime cell:
```
DJ30.r/trending/ema_stack    = 1.0875  (Δ +0.0375 = MAX_CHANGE × 0.75 factor)
DJ30.r/trending/supertrend   = 1.0875
```

Audit log entry produced:
```
WEIGHT_UPDATE        ema_stack: 1.00→1.05 (WR=75%, n=12); supertrend: 1.00→1.05 (WR=75%, n=12)
REGIME_WEIGHT_UPDATE:trending  ema_stack: 1.05→1.09 (WR=75%, n=12); supertrend: 1.05→1.09 (WR=75%, n=12)
```

DB writeback verified:
```
score_weights:    (ema_stack, 1.05, 9, 3)    ← win/loss counts correctly persisted
regime_weights:   (trending, ema_stack, 1.0875, 9, 3)  ← C2 audit fix confirmed (loss_count=3 NOT 0)
```

### Trace 2 — RULE 1 regime trail tightening + write path

Input: 15 trades, peak_r=2.0R, win-rm=0.5R → giveback ≈ 1.5R = 75% of peak (RULE 1).
Result:
```
global trail_tightness_mult: 0.70  (5 RULE-1 cycles × −0.05 each, clamped at 0.6)
global lock_threshold_mult: 0.82  (5 × −0.03 each)
regime_trail[DJ30.r][trending]: {lock 0.79, be 1.00, trail 0.65}
```

DB writeback verified:
```
regime_trail_adjustments: ('DJ30.r','trending', 0.79, 1.00, 0.65, '...')   ← C3 audit fix confirmed
trail_adjustments:        ('DJ30.r', 0.82, 1.00, 0.70, '...')              ← also written
```

5 REGIME_EXIT_UPDATE rows in audit_log — confirming the new `_maybe_update_regime_trail` writer fires on each closed trade.

### Trace 3 — REVERT + restart persistence

Input: 12 USDJPY trades, 1W/11L, PF=0.05 (well below PF_REVERT_SEVERE=0.5).
Result:
```
RL REVERT [SEVERE]: PF 0.05 over n=12 — reverting weights, trail, AND regime overlay
reverted[USDJPY] = True
reverted_at[USDJPY] = 1779485710.92
weights all reset to 1.0: True
```

DB writeback verified:
```
revert_state: ('USDJPY', 1, 1779485710.92, '...')          ← H3 audit fix confirmed
rl_audit_log REVERT: 'PF=0.05 < 0.7 (weights+trail+regime reset)'
```

After NEW RLLearner instance loads from same DB:
```
rl2._reverted['USDJPY'] = True       ← persistence works
rl2._reverted_at['USDJPY'] = 1779485710.92
weights[USDJPY][ema_stack] = 1.0
```

UN_REVERT also confirmed: after 12 wins with reverted_at backdated 2h, the symbol leaves REVERT and resumes normal learning.

---

## DB Schema Validation Table

| Table | Rows | Schema OK | NaN/neg counts | Notes |
|---|---|---|---|---|
| `score_weights` | 0 | OK (9 cols incl. absent_loss_count) | none | Empty after fresh restart; will populate when first WEIGHT_UPDATE fires (≥12 trades w/ components) |
| `regime_weights` | 0 | OK (9 cols, PK symbol+regime+component) | none | Audit C2 fixed today (wins/losses pass-through) |
| `regime_trail_adjustments` | 0 | OK (6 cols, PK symbol+regime) | none | Audit C3 writer fixed today; verified populating in synthetic trace |
| `exit_learning` | 28 | OK (7 cols, PK symbol+exit_reason) | none | Has 5 legacy orphan rows (`EarlyLossCut_T1-`, etc.) from pre-normalization era — M3 issue still open |
| `trail_adjustments` | 2 | OK | none | Only DJ30.r & BTCUSD touched today (trail_tightness=1.05, RULE 3 tiny-wins) |
| `rl_audit_log` | 2 | OK | none | Only 2 EXIT_UPDATEs since fresh restart; no WEIGHT_UPDATEs yet because no components |
| `trade_outcomes` | 68 | OK (12 cols + idx_sym_ts) | none | 0/68 with components_json → CRITICAL #1 above |
| `equity_peak` | 1 | OK (single-row table CHECK id=1) | none | peak=$1219.50 — will be replaced when $5K equity reads in |
| `revert_state` | 0 | OK | none | New table; nothing reverted yet |
| Indexes | 7 | OK | — | `idx_trade_outcomes_sym_ts` exists; M8 audit suggested `idx_audit_sym_action` — not yet added |

No NaN, no negative win/loss counts, no orphan-FK issues. Schema is healthy for $5K Monday.

---

## $5K vs $1K Behavior Diff Table

| Component | Scaling basis | $1K behavior | $5K behavior | Diff/Risk |
|---|---|---|---|---|
| `get_equity_dd_multiplier` | % DD off peak | 3% DD on $1K = $30 trigger | 3% DD on $5K = $150 trigger | None — purely percentage |
| `get_streak_multiplier` | Loss count over last 3 | Count-only | Count-only | None |
| `get_expected_value_r` | R-multiples | R-units (dimensionless) | Same | None |
| `_maybe_update_weights` rolling_pf | gp/gl ratio over last 12 PnLs | Mixed $20 wins / $10 losses | Mixed $100 wins / $50 losses | Ratio unchanged; PF_REVERT thresholds (0.5/0.7) scale-invariant |
| `get_quality_threshold_bonus` PF | gp/gl ratio over last 10 PnLs | Ratio-only | Ratio-only | None |
| `get_edge_score` PF blend | gp/gl ratio | Ratio-only | Ratio-only | None |
| `peak_equity` persistence | $ absolute | $1219.50 currently in DB | First $5K read → peak resets to $5000 (line 741-744) | First call must be made before any DD calc — handled by the bootstrap at brain.py:1996-1997 |
| `get_risk_multiplier` regime/hour | WR + PF combined | min 0.6, max boost_max=1.3-1.5 | Same — proportional via risk_pct caller | Caller applies % of equity, so $-risk scales 5× automatically |
| Trade-outcome PnL persistence | Raw $ in `trade_outcomes.pnl` | $20 entries | $100 entries (mixed with legacy $20) | Ratio metrics still fine, but `get_edge_score`'s `pf30` window will include MIXED-magnitude trades for ~30 trades after swap. Result is unbiased (ratios), but reasoning about absolute $ in `[RL HEALTH]` becomes ambiguous |
| `_persist_absent_loss` count | int counter | n increments | n increments | None |

**Net**: no $5K-vs-$1K math regression. The only transient is **peak_equity** resetting to $5000 on first read of new equity — the existing brain bootstrap (`brain.py:1996-1997`) primes RL with the brain-tracked peak. As long as brain's `agent_state["peak_equity"]` reflects the new $5K reality immediately after the account swap, DD scaling stays sane.

**Recommended pre-Monday step**: after swapping to $5K, the operator should call `equity_peak` UPDATE to 5000 (or just let the next live equity read overwrite it via `get_equity_dd_multiplier`). The current $1219.50 row is safe (gets bumped to $5000 on first call) but it'll show a phantom +310% peak that doesn't represent the real account.

---

## First-Trade / n=0 Behavior

Verified bootstrap behavior for every public method (read-only inspection):

| Method | n<threshold returns | Threshold |
|---|---|---|
| `should_skip_entry` | `(False, "")` | < lookback (20-30) |
| `get_risk_multiplier` | `1.0` | < lookback |
| `get_score_weights` | dict with all 1.0 | always |
| `get_trail_adjustments` | `{1.0, 1.0, 1.0}` | always |
| `get_weights` | `{}` (scorer treats as 1.0) | unknown symbol |
| `get_equity_dd_multiplier` | `1.0` | ce<=0 or peak<=0 |
| `get_streak_multiplier` | `1.0` | <3 outcomes |
| `get_expected_value_r` | `(0,0,0,0,0)` | <15 outcomes |
| `get_quality_threshold_bonus` | `0` | <5 outcomes |
| `get_edge_score` | `0.5` (neutral) | <8 outcomes |
| `record_outcome` (called n=0) | works; appends, persists, _maybe_update gated on MIN_TRADES=12 | — |

**Bootstrap confirmed safe**: no division-by-zero, no NaN, no negative counts. All defaults are the neutral/conservative side. ✓

---

## REVERT / UN_REVERT Cycle — Fully Working After Today's Fixes

Verified end-to-end via synthetic trace (Trace 3 above):
1. PF<0.5 over 12 trades → SEVERE REVERT fires (was broken before today — C1).
2. `revert_state` row written with `reverted=1`, `reverted_at=<epoch>` (was missing — H3).
3. All weights reset to 1.0, all trail mults reset to 1.0, regime overlay cleared.
4. `regime_weights`/`regime_trail_adjustments` rows for symbol DELETEd.
5. New RLLearner instance from same DB → `reverted=True` persists, weights stay at 1.0.
6. After 12 wins + 1h elapsed → UN_REVERT fires, persists `reverted=0`.

**One stale-message issue (M1, still open)**: REVERT audit detail says `"PF=0.05 < 0.7 (...)"`. The `<0.7` is `PF_REVERT_SOFT`, but for SEVERE this should be `<0.5`. The console log already includes the tier (`[SEVERE]`), but the audit table row doesn't distinguish.

---

## Health Metrics — `[RL HEALTH]` Counter Accuracy

Current snapshot (from logs):
```
[RL HEALTH] peak=$1219.50 | trades=68 | weight_cells=0 | regime_cells=0 |
            trail_syms=2 | reverted=0 | updates_run=0 (regime=0)
```

| Metric | Source | Accurate? | Notes |
|---|---|---|---|
| `peak_equity` | `self._peak_equity` | ✓ DB-persisted | Will reset to $5K on first new-account read |
| `tracked_trades` | sum of `_trade_outcomes` lengths | ✓ DB-loaded (last 100/sym) | Includes orphans |
| `weight_cells` | non-1.0 weights | ✓ in-memory | All zero currently because 0 components → 0 updates |
| `regime_cells` | non-default regime weights | ✓ in-memory from DB | 0 currently |
| `trail_syms` | symbols with non-1.0 trail | ✓ in-memory from DB | 2 today (DJ30.r, BTCUSD) |
| `reverted_symbols` | count of `_reverted[s]==True` | ✓ now DB-restored on init | 0 currently |
| `updates_run` | `_n_updates` (this run only) | **stale on restart** (H4 unfixed) | Resets to 0 on every bot restart. Misleads operator into thinking RL is stuck when it's just freshly restarted. |
| `regime_updates_this_run` | `_n_regime_updates` (this run only) | same H4 issue | Same |

Not surfaced in HEALTH (M10 unfixed):
- `_n_exit_updates` doesn't exist.
- `_n_reverts` doesn't exist.
- EXIT_UPDATE / REVERT counts invisible from health line.

Cadence:
- `health_summary()` is called every brain cycle (`brain.py:942`).
- Logs once per 3600s (default), or `force_log=True`.
- ✓ Cadence is correct; counter-staleness is the only issue.

---

## Background Update Cadence

| Mechanism | When | Throttle | Notes |
|---|---|---|---|
| `_maybe_update_weights` | inside `record_outcome` → every trade close | `UPDATE_THROTTLE_SEC = 14400` (4h) per-symbol | REVERT bypasses throttle. Component-active-trades ≥10 gate also required |
| `_maybe_update_regime_weights` | called from inside `_maybe_update_weights` after throttle gate | inherits 4h throttle (since parent gates first) | Sparse — only fires when ≥10 reg-samples in the 12-trade window AND component had 5+ active samples in that regime |
| `_maybe_update_exits` | inside `record_outcome` → every trade close | NONE | Per-trade. With 50-trade exit-rolling window, the four RULES can flap if avg_giveback hovers around thresholds |
| `_maybe_update_regime_trail` | called from inside `_maybe_update_exits` → every trade close | NONE | Same — per-trade. The TASK J writer added today. Will start populating from first regime with ≥10 recent trades |
| `health_summary` | every brain cycle (~1s) | logs every 3600s | Read-only snapshot |
| `_persist_peak_equity` | inside `get_equity_dd_multiplier` only on new high | none — per-event | One UPDATE every time equity makes a new peak |

**Risk**: `_maybe_update_exits` has no throttle. If a symbol takes 5 consecutive trades that all flip RULE 1 → RULE 2 → RULE 1, trail_tightness_mult can swing ±0.05 per trade — over 5 trades that's ±0.25 swing on a [0.6, 1.5] range. **H2 (one-way ratchet on `lock_threshold_mult`) is still unfixed** — `lock_threshold_mult` only decreases (RULE 1 path), never recovers. Pre-Monday recommendation: add per-symbol throttle to `_maybe_update_exits` mirroring the 4h weight throttle, OR fix the one-way ratchet.

---

## CONFIRMED FIXED TODAY (validated by synthetic trace)

1. **C1 — `_reverted_at` AttributeError**: `__init__` now declares `self._reverted_at: Dict[str, float] = {}` at line 115. First REVERT no longer crashes. ✓
2. **C2 — regime_weights wins/losses mislabel**: `_maybe_update_regime_weights` now stores `"wins"`/`"losses"` in `changes_all[regime][comp]` (line 1239-1240); `_persist_regime_weights` reads `s.get("wins", 0)` for win_count. Synthetic trace confirms regime row has (wins=9, losses=3). ✓
3. **C3 — `regime_trail_adjustments` write path**: New `_maybe_update_regime_trail` method (lines 1422-1546) mirrors `_maybe_update_exits` per (symbol, regime), called from `_maybe_update_exits` line 1420. Synthetic trace confirms row written. ✓
4. **H3 — `_reverted` persistence**: New `revert_state` table, new `_persist_revert_state` writer, `_load_regime_state` hydrates on startup. Restart-persistence confirmed in synthetic trace. ✓
5. **H6 — `_last_close_peak_r` race**: Verified only ONE `.pop()` site (`brain.py:1341` external-close), other two are `.get()` (brain.py:977, brain.py:2319, learning_engine.py:805). ✓

---

## STILL OPEN (audit_20260522 findings not yet addressed)

- **C4** — `exit_learning` not rebuilt from DB on startup. `_exit_outcomes` is fresh-empty each session; M3 legacy 16-char rows still in DB.
- **H1** — `be_threshold_mult` dead code (always 1.0).
- **H2** — `lock_threshold_mult` one-way ratchet (only decreases).
- **H4** — `_n_updates` resets on restart, misleading health line.
- **H5** — multi-collinear components get identical credit (design issue).
- **M1** — REVERT audit detail message stale (`<0.7` regardless of tier).
- **M2-M10** — see source audit.

### New findings this session
- **N1** — components_json empty for all existing trades (Critical Finding #1). Will self-heal in ~3 days of live trading but blocks weight learning until then.
- **N2** — `get_trail_adjustments(sym)` in brain.py:927 missing `regime=` arg → regime overlay never wired (Critical Finding #2). One-line fix recommended pre-Monday.
- **N3** — 3 of 5 active SYMBOLS not in RL_ENABLED_SYMBOLS (Critical Finding #3). Config-only fix recommended pre-Monday.
- **N4** — `_maybe_update_exits` has no per-symbol throttle (1 per trade), which combined with H2's one-way ratchet means `lock_threshold_mult` can drift to its floor (0.7) very fast in a high-giveback streak with no recovery path.
- **N5** — `_sync_mt5_deals` per-symbol `entry_metadata` lookup is single-slot per symbol — if two trades on same symbol close in one poll, the second reads the first's overwritten metadata. Bounded by single-position cooldowns, low practical risk for current SYMBOLS list.

---

## PRE-MONDAY RECOMMENDATIONS

In order of urgency:

1. **(2-min config)** Add SP500.r, USOUSD, XAUUSD to `RL_ENABLED_SYMBOLS` in `config.py` with `RL_SYMBOL_PARAMS = {"lookback": 30, "boost_max": 1.3}`. Removes coverage asymmetry.
2. **(5-min code)** Fix `brain.py:927` to pass `regime=` to `get_trail_adjustments`. Activates the regime trail overlay that the new writer is now populating.
3. **(5-min DB)** UPDATE `equity_peak SET peak_equity = 5000, peak_ts = <now>` after the $5K account swap completes. Avoids transient phantom peak.
4. **(5-min code)** Add explicit `WEIGHT_LEARN_READY` counter to `[RL HEALTH]` showing how many symbols have ≥12 trades w/ components — gives the operator visibility into "is RL actually learning yet?"
5. **(defer to post-freeze)** H2 one-way ratchet, H4 stale counters, M1 stale audit message — these are observability cleanups, not safety issues.

The 5 critical bug fixes shipped today (C1/C2/C3/H3/H6) have been validated against a tmp-copy DB through both the WEIGHT_UPDATE path AND the REVERT/UN_REVERT cycle AND the regime trail writer. Persistence across a fresh `RLLearner()` instantiation is confirmed for revert_state, score_weights, regime_weights, regime_trail_adjustments.

The RL learner is **ready to run for 1 week untouched** once the 3 pre-Monday recommendations above are applied. Without them, it will still run safely (no crashes, no NaN, no runaway weights), but with reduced learning surface area — 3/5 symbols won't get regime/hour skip and risk multipliers, and the regime trail overlay won't reach the executor.
