# RL Learner Audit — 2026-05-22

Source under audit: `/Users/ashish/Documents/beast-trader/agent/rl_learner.py` (1489 lines)
DB under audit: `/Users/ashish/Documents/beast-trader/data/rl_learner.db` (905 KB)
Live log evidence: `/Users/ashish/Documents/beast-trader/logs/dragon.log`

## Top 3 Most-Impactful Findings

1. **REVERT path silently AttributeError-crashes on first fire each session — safety reset no longer works** (CRITICAL, `rl_learner.py:1024`). `_reverted_at` is never initialised in `__init__`. The first REVERT for any symbol per process raises `AttributeError` on line 1024 before reaching the lazy-init guard at line 1025. The in-memory `_weights`/`_trail_adjustments`/`_regime_*` resets (lines 1012-1022) have already happened, AND `_reverted[symbol]=True` was set (line 1023), but the audit log entry, `_persist_weights`, `_persist_trail`, and `DELETE FROM regime_weights` (lines 1027-1037) NEVER run. Net effect: REVERT happens transiently in memory, but on restart the DB resurrects the bad weights. **The audit log has had zero REVERT entries since 2026-05-14** despite 11 symbols currently sitting at PF<0.5 over 12 trades (GBPCHF 0.05, UK100.r 0.07, EURCHF 0.12, US2000.r 0.17, GBPJPY 0.22, …). The two-tier safety net is therefore inoperative.

2. **`regime_weights.win_count` is mislabelled as total samples; `loss_count` is always 0** (HIGH, `rl_learner.py:1188-1190` + `rl_learner.py:1222-1223`). The dict passed into `_persist_regime_weights` (`changes_all[regime][comp]`) only stores `{"old","new","wr","n"}` — there is no `"losses"` key. Line 1223 does `s.get("losses", 0)` → always 0. Line 1222 computes `int(s.get("n",0)) - int(s.get("losses",0))` → always equal to `n` (total). All 67 rows in `regime_weights` show `loss_count=0`; `win_count` is the trade total (e.g. ETHUSD/trending=12, identical for all 8 components). Downstream tuners reading this table see "zero losses everywhere" — corrupt evidence trail.

3. **`regime_trail_adjustments` table has zero rows because no code ever writes to it** (HIGH, `rl_learner.py` schema:262, no INSERT anywhere). The schema exists, the ALTER for `symbol` column exists, the read path (`_load_regime_state`:619-633) exists, the REVERT-time DELETE-from-regime exists (only for `regime_weights`, not `regime_trail_adjustments`). But `grep -n "INSERT.*regime_trail"` returns nothing — `_maybe_update_exits` (line 1287-1362) only writes the symbol-level `trail_adjustments` table. TASK J was implemented half-way: read+fallback paths shipped, writer not. Plus the brain only ever calls `get_trail_adjustments(sym)` without `regime=` (brain.py:844), so even if rows existed they wouldn't flow to executor.

---

## CRITICAL (data corruption / non-learning)

### C1. `_reverted_at` AttributeError disables auto-revert safety
**File**: `agent/rl_learner.py:1024`
**Bug**:
```python
self._reverted_at[symbol] = time.time() if hasattr(self, "_reverted_at") else None
if not hasattr(self, "_reverted_at"):
    self._reverted_at = {symbol: time.time()}
```
`self._reverted_at[symbol] = X` first requires `self._reverted_at` to exist (Python `__getattr__` then `__setitem__`). The `hasattr` is on the VALUE side, not on whether to do the assignment. `_reverted_at` is not declared in `__init__` (lines 85-146). First REVERT per process raises `AttributeError`, caller (`brain.py:2199-2200` try/except `log.debug`) swallows silently.
**Evidence**:
- DB shows 11 symbols at PF<0.5 over last 12 trades (would all be SEVERE-revert candidates), last REVERT in audit_log is 2026-05-14 (8 days ago).
- `dragon.log` shows `RL HEALTH ... reverted=0` for all recent restarts despite obviously failing symbols.
- Bug introduced in commit `a446568` (2026-05-12 RL self-stabilizing), aligned exactly with the REVERT stoppage.
**Impact**: The "auto-revert if rolling PF drops below 0.7" safety promised in the module docstring (line 9) is non-functional. Weights drift into bad territory and stay there. US2000.r has had 85 WEIGHT_UPDATEs (1.00→0.90) over 9 days while bleeding — no revert ever fires. Combined with throttle (4h between updates), the symbol can only deteriorate at ~5%/4h per update.
**Suggested fix**: Add `self._reverted_at: Dict[str, float] = {}` to `__init__` (around line 111). Remove the convoluted `hasattr` dance at lines 1024-1026; just `self._reverted_at[symbol] = time.time()`.

### C2. `regime_weights.loss_count` always 0; `win_count` is mislabelled as total
**File**: `agent/rl_learner.py:1188-1190` (writer) + `1222-1223` (persist)
**Bug**: `changes_all[regime][comp] = {"old", "new", "wr", "n"}` lacks `"wins"` and `"losses"`. Then:
```python
int(s.get("n", 0)) - int(s.get("losses", 0)) if s else None,  # win_count column
int(s.get("losses", 0)) if s else None,                       # loss_count column
```
Always passes `n` to `win_count` and `0` to `loss_count`.
**Evidence**: All 67 `regime_weights` rows have `loss_count=0`. `win_count` matches the total sample size, e.g. ETHUSD/trending=12 (we see WR=58% in audit_log meaning ~7 wins/5 losses — should be split that way).
**Impact**: Downstream `synthesize_auto_tuned.py` and per-regime tuners can't trust the win/loss split. Per the module's own claim (line 1369-1374 docstring for `_persist_weights`), the win_count/loss_count fields exist specifically so the table is auditable. The regime table fails that contract.
**Suggested fix**: In `_maybe_update_regime_weights` (line 1188), store `"wins": wins, "losses": losses` in the changes_all entry. Then in `_persist_regime_weights` (line 1222), use `int(s.get("wins", 0))` for the win_count column.

### C3. `regime_trail_adjustments` is write-orphaned (zero rows ever)
**File**: `agent/rl_learner.py` — schema at 262-271, read at 619-633, no INSERT anywhere
**Bug**: Dead feature. `_maybe_update_exits` (lines 1287-1362) only persists to symbol-level `trail_adjustments`. There's no `_maybe_update_regime_trail` function. The TASK J 2026-05-17 "global-per-regime fallback" cache (`_regime_trail_global`, line 126) is loaded from a table that nothing ever populates.
**Evidence**: `SELECT COUNT(*) FROM regime_trail_adjustments` → 0. `grep -rn "INSERT.*regime_trail"` returns no matches.
**Impact**: Per-regime trail learning advertised in the architecture (line 117-127 comments) never actually happens. Plus brain.py:844 calls `get_trail_adjustments(sym)` with no `regime=` arg, so the regime overlay couldn't reach executor even if rows existed.
**Suggested fix** (two parts):
  1. Add `_maybe_update_regime_trail(symbol, regime)` mirroring `_maybe_update_exits` but writing per `(symbol, regime)` cell. Or explicitly delete the table + read path until ready.
  2. Brain should pass current regime when wiring trail: `self._rl_learner.get_trail_adjustments(sym, regime=self._last_scores.get(sym, {}).get("regime"))`.

### C4. `exit_learning` table reflects only current session — never restored at startup
**File**: `agent/rl_learner.py:98` (init) — `_exit_outcomes` dict, never hydrated from DB
**Bug**: `_exit_outcomes` is in-memory list per `f"{symbol}_{exit_key}"`. On every restart it begins empty. The `INSERT OR REPLACE` at line 957-963 overwrites the persisted row each time `record_outcome` fires, but the data being overwritten is "outcomes seen this session only".
**Evidence**: `BTCUSD|SL` shows `count=1` despite the actual trade journal having dozens of BTCUSD SL exits. Restarts at 19:39, 20:01, 20:04, 20:37 today wipe `_exit_outcomes` four times in 1 hour.
**Impact**: External consumers reading `exit_learning` cannot rely on counts/averages — they reflect only since the most recent restart. Daily learning loop can't accumulate a real exit-reason picture.
**Suggested fix**: Add hydration step in `_load_state`:
```python
for sym, er in conn.execute("SELECT symbol, exit_reason FROM exit_learning"):
    key = f"{sym}_{er}"
    # would need stored raw samples — schema only has aggregates.
```
Realistically: backfill `_exit_outcomes` from `trade_outcomes` rows on startup (group by symbol+exit_key and rebuild last 50 R-multiples). The data is already there.

---

## HIGH (sub-optimal learning / misapplied)

### H1. `be_threshold_mult` is dead code in RL — never adjusted
**File**: `agent/rl_learner.py:1287-1362` (`_maybe_update_exits`)
**Bug**: Of three trail multipliers stored/loaded/persisted (`lock_threshold_mult`, `be_threshold_mult`, `trail_tightness_mult`), only `lock_*` and `trail_*` get touched inside the four exit-learning rules. `be_threshold_mult` defaults to 1.0 and stays there for the life of the symbol.
**Evidence**: `SELECT COUNT(*) FROM trail_adjustments WHERE abs(be_threshold_mult - 1.0) > 0.001` → 0 across 28 rows. Audit log shows `be_m=1.00` on every EXIT_UPDATE entry.
**Impact**: The executor (executor.py:1564, 1647) reads `be_threshold_mult` and combines it with momentum-adaptive scaling — but the RL-side contribution is always neutral. We're shipping a feature surface that does nothing.
**Suggested fix**: Either implement a rule (e.g. "if avg_giveback at BE level is high, tighten BE threshold"), or delete the column to stop pretending. The executor's momentum-adaptive scaling already provides dynamic BE behaviour.

### H2. `lock_threshold_mult` is monotone-decreasing — can never recover
**File**: `agent/rl_learner.py:1321`
**Bug**: Only RULE 1 (line 1316-1325) touches `lock_threshold_mult` and it ONLY decreases (`max(0.7, x - 0.03)`). RULES 2/3/4 only touch `trail_tightness_mult`. So once `lock_threshold_mult` is driven to 0.73 by a high-giveback period, it can never climb back even if the symbol enters a good-trail regime.
**Evidence**: 17/28 symbols sit at `lock_threshold_mult=0.76` (the typical 8-decrement landing zone from 1.0). DJ30.r at 0.73 hasn't moved in 12+ EXIT_UPDATEs.
**Impact**: Asymmetric ratchet. Once tight, always tight. The "self-correcting" promise is one-way.
**Suggested fix**: Add an increment path in RULE 2 or a new rule — symmetrical to RULE 1 — e.g. "if avg_giveback < 10% of peak for 10+ trades, raise lock_threshold_mult by 0.02 toward 1.0".

### H3. UN_REVERT can never fire in production because `_reverted` is in-memory only
**File**: `agent/rl_learner.py:111, 1046-1051`
**Bug**: `_reverted: Dict[str, bool] = {sym: False for sym in SYMBOLS}` is rebuilt fresh on every `__init__`. There's no DB column or table for this. So a REVERT that fired in session A doesn't carry over to session B; session B starts with `_reverted=False`, falls through to WEIGHT_UPDATE branch even if the symbol is still bleeding.
**Evidence**: `dragon.log` shows `reverted=0` every cycle. There was exactly ONE UN_REVERT in 9 days of audit_log — only when a REVERT and recovery both happened inside the same long-running process (BTCUSD 2026-05-18).
**Impact**: REVERT/UN_REVERT lifecycle is broken across restarts. The two-tier safety design (severe + soft) implicitly assumes session continuity that doesn't exist.
**Suggested fix**: Persist `_reverted` and `_reverted_at` to a small DB table (or to `equity_peak`-style single-row table), hydrate on startup. This also fixes C1 if `_reverted_at` is initialised at load.

### H4. `_n_updates` counter resets on restart — health metric is misleading
**File**: `agent/rl_learner.py:136, 1127, 1276`
**Bug**: `_n_updates` is process-local. `[RL HEALTH] updates_run=0` means "no updates since most recent process start" — but dragon.log shows 4 restarts in 1 hour. The metric reads like "RL is stalled" when it's really "RL restarted recently and no trade has closed yet".
**Evidence**: `dragon.log` 19:39→20:01→20:04→20:37 — four startup log lines, all followed by `updates_run=0`.
**Impact**: Operators can't tell if RL is dead vs just freshly restarted. Misleads incident triage.
**Suggested fix**: Either rename to `updates_this_run` AND surface a separate `updates_today` counter (queried from `rl_audit_log` rows with `timestamp > now-24h AND action='WEIGHT_UPDATE'`) — or persist `_n_updates` cross-restart.

### H5. Multi-collinear scoring components get identical credit
**File**: `agent/rl_learner.py:1066-1088`
**Bug**: When 8 components light up together (correlated indicators: ema_stack + supertrend + macd_signal + macd_hist + heikin_ashi + structure + breakout + momentum_vel all fire on the same trending bar), they record IDENTICAL wins/losses. The learner can't differentiate them — they all get the same delta and converge to the same weight.
**Evidence**: ETHUSD/trending — all 8 components at exactly `1.02916666666667`, win_count=12, loss_count=0. EURUSD/low_vol — 5 of 8 at exactly `1.22`.
**Impact**: RL credit assignment is undifferentiated. Reducing the weight on `macd_hist` to "test if it's the real edge" is impossible — `macd_signal` (correlated) gets reduced equally and you can't isolate the effect. Fundamental design limit, not a typo bug.
**Suggested fix** (non-trivial, out of scope for quick fix): Per-component conditional WR (e.g. WR(macd_hist=1 | ema_stack=0)) requires more samples but yields independent signal. Alternative: orthogonalize the component scores via PCA before feeding to RL.

### H6. Two `pop()` callers race on `_last_close_peak_r`
**File**: `agent/brain.py:1212` + `agent/learning_engine.py:798`
**Bug**: Both call sites `pop()` from `executor._last_close_peak_r`. Whichever runs first gets the peak_r; the other gets 0. `record_outcome` then receives `peak_r=0` half the time → `_maybe_update_exits` sees no giveback signal → trail learning never tightens despite real giveback.
**Evidence**: `grep -rn "_last_close_peak_r"` shows two `pop`s (brain.py:1212, learning_engine.py:798) and one `get` (brain.py:2192). The `get` at 2192 doesn't pop, so it's idempotent — but the two pops are mutually destructive.
**Impact**: Half the time `_maybe_update_exits` sees `peak_r=0` and the giveback rules never fire. Trail learning is undersized.
**Suggested fix**: Use `.get()` everywhere and pop only at end-of-cycle in ONE site (probably brain.py). Or, even simpler, archive `_last_close_peak_r` into a non-destructive `_last_close_peak_r_history` that the executor clears periodically.

---

## MEDIUM (cleanup, edge cases)

### M1. Audit log REVERT detail still says "< 0.8" — message is stale
**File**: `agent/rl_learner.py:1027`
```python
self._audit(symbol, "REVERT", f"PF={rolling_pf:.2f} < {PF_REVERT} (weights+trail+regime reset)")
```
Uses `PF_REVERT` (legacy alias = `PF_REVERT_SOFT` = 0.7) but doesn't distinguish severe vs soft. Audit log can't tell which tier fired. Fix: include `tier` (already computed at line 1008).

### M2. `_persist_absent_loss` can pre-insert weight=1.0 before a learned weight is persisted
**File**: `agent/rl_learner.py:1418-1426`
On a fresh symbol, the first significant-loss `record_outcome` writes the row with `weight=1.0` before any `_maybe_update_weights` has run. If RL had already adjusted the in-memory weight (between init and first persist), `_persist_absent_loss` could race-overwrite. Low probability in practice — caller is single-threaded — but the pattern is fragile.
Fix: insert with `weight = self._weights[sym].get(comp, 1.0)` instead of literal `1.0`.

### M3. `EarlyLossCut_T1-` / `EarlyLossCut_T2-` orphaned rows in exit_learning
**File**: `data/rl_learner.db` (exit_learning table), pre-fix data
After the 2026-05-16 normalization fix (line 906-915), new exits collapse to `EarlyLossCut`. The truncated 16-char rows (`EarlyLossCut_T1-`, `EarlyLossCut_T2-`, `GuardianHeatRedu`, `GuardianSharpLos`, `GuardianStaleLos`) are now stuck in the table.
Fix: one-shot DELETE on those legacy rows (run-once cleanup script). Or backfill them into the canonical key.

### M4. `_persist_regime_weights` runs DB IO outside the lock
**File**: `agent/rl_learner.py:1204-1228`
Caller releases `self._lock` at line 1187 (block end). Then iterates `changes_all` at 1192-1202 (no lock) and calls `_persist_regime_weights` (no lock). Concurrent REVERT could DELETE FROM regime_weights mid-persist. Practical risk low (record_outcome is the only writer in single-threaded mode), but the locking contract is inconsistent vs `_persist_weights` (also lock-less).
Fix: Either accept it and document, or wrap persist calls in a writer lock.

### M5. Regime gating excludes the dominant regime (`volatile` = 3861/4359 trades)
**File**: `agent/rl_learner.py:1167-1175`
Update rule fires only when WR > 0.55 or WR < 0.40. Volatile regime sits in 0.40-0.55 range broadly. Result: regime_weights covers only `trending` (17 cells) and `low_vol` (50 cells) — the volatile majority is never learned.
Not a bug; design choice. But also: regime is computed from the LAST 12 trades window — recent regime composition skews. USDCAD's overall volatile WR is 94% but its last 12 trades were all in `low_vol`, so the volatile edge is invisible to the learner.
Fix (optional): widen the per-regime window separately from the global-update window. Allow learning from any 30-trade regime cell regardless of where the global PF stands.

### M6. `_maybe_update_regime_weights` requires `len(reg_trades) >= 10` of the 12-trade recent slice
**File**: `agent/rl_learner.py:1153`
Within 12 recent trades, 10+ must share regime. So regime learning is only possible if the symbol traded in a single regime for ~85% of recent activity. Combined with M5, this is very restrictive — most regime cells never get a chance to update.

### M7. `_persist_weights` writes ALL components every call, even unchanged ones
**File**: `agent/rl_learner.py:1378-1399`
Loops over `self._weights[symbol].items()` (all 11 components) and writes each row regardless of whether it changed in this update. Causes 11 SQL writes per WEIGHT_UPDATE event, most of which are no-op upserts. Minor performance issue; main effect is the `updated` timestamp churn on unchanged rows.
Fix: write only the components that appear in `component_stats` (changed ones).

### M8. `rl_audit_log` has no index — query performance degrades over time
**File**: `agent/rl_learner.py:198-205` (schema)
1145 rows now, growing. Filters by `action` and `symbol` are common but slow without index. Not critical at current volume, but should be addressed before 100K rows.
Fix: `CREATE INDEX IF NOT EXISTS idx_audit_sym_action ON rl_audit_log(symbol, action)`.

### M9. 17 orphan symbols accumulating
**File**: `dragon.log` startup line
RL is preserving state for 17 symbols not in current `SYMBOLS` (AUDUSD, BCHUSD, CADJPY, COPPER-Cr, EURAUD, EURCHF, EURGBP, FRA40.r, GAS-Cr, GBPAUD, GBPCHF, GBPJPY, GBPUSD, GER40.r, HK50.r, UKOUSD, USDCHF). State is preserved by design (per the `_ensure_symbol` change comment at line 305-315) — but never reactivated, never cleaned up. Causes confusion and bloats `trade_outcomes` queries.
Fix (operational, not code): periodic `cleanup_orphans.py` script that requires explicit consent before deleting > N-day-old orphan state.

### M10. `_n_updates` only increments on WEIGHT_UPDATE — not on EXIT_UPDATE or REVERT
**File**: `agent/rl_learner.py:1127`
Increment is inside the `if changes:` branch. REGIME_WEIGHT_UPDATE has its own counter (`_n_regime_updates`, line 1202). But EXIT_UPDATE and REVERT have no counter at all — they don't show in `[RL HEALTH]`. Operators can't see exit-rule churn from health.
Fix: add `_n_exit_updates` and `_n_reverts` counters, surface in `health_summary`.

---

## CONFIRMED WORKING

1. **Score weight injection into live scoring** — `brain.py:1317` calls `get_weights(symbol, regime=regime)`, the result is passed to `_score_with_components` at `signals/momentum_scorer.py:655` where `sl = sum(comp_l[c] * float(weights.get(c, 1.0)) for c in comp_l)`. Verified end-to-end.

2. **`record_outcome` is wired on both close paths** — `brain.py:1216` (external close) and `brain.py:2193` (internal close) and `learning_engine.py:801` (journal-driven). All three call `record_outcome` with `score_components` and `peak_r`.

3. **`trade_outcomes` persistence + restart restore** — verified by `_load_state` lines 387-429 hydrates the last 100 outcomes per symbol from `trade_outcomes`. After today's 4 restarts, `RL Learner loaded 2897 trade outcomes from DB (486 with components)` confirms the pipeline.

4. **Weight clamps work** — `WEIGHT_MIN=0.3, WEIGHT_MAX=2.5, MAX_CHANGE=0.05`. DB inspection: `MIN(weight)=0.9`, `MAX(weight)=1.4` across 291 score_weights rows. No runaway. Per-update delta is bounded by the wr×0.2 formula plus the clamp.

5. **Update throttle prevents flapping** — `UPDATE_THROTTLE_SEC=14400` enforced at line 1056-1059. US2000.r weight updates spaced 75-90 min in some cases, never below 4h between updates after the throttle was added.

6. **`absent_loss_count` is being populated** — TASK I works. 27 (symbol, component) pairs have non-zero counts, top entries DJ30.r/candle_pattern=6, BTC trade #753 pattern visibility confirmed.

7. **`score_weights.win_count` / `loss_count` populated correctly** — Unlike the regime table (see C2), `score_weights` upsert at lines 1382-1399 passes the full `component_stats` dict including `wins` and `losses`. DB shows 87/291 rows with `loss_count > 0`. Working.

8. **`get_quality_threshold_bonus` and EV gate** — `brain.py:1348` calls `get_quality_threshold_bonus`, and `brain.py:1985` calls `get_expected_value_r`. Both are tied to live signal gating. Verified call paths.

9. **DD-aware risk multiplier (`get_equity_dd_multiplier`)** — peak equity persisted at line 705, restored at line 656-663. Live log shows `peak=$1298.49` consistent across all 4 restarts today.

10. **Orphan-symbol preservation** — `_load_state` at lines 305-321 properly handles symbols in DB but not in current `SYMBOLS`. 17 orphans correctly preserved with state intact across today's restarts.

11. **Streak multiplier (`get_streak_multiplier`)** — pure in-memory computation over `_trade_outcomes`; works correctly since outcomes restore from DB.

12. **Exit rule learning fires on giveback signal** — 830 EXIT_UPDATE entries in audit log, most recent 2026-05-22 04:36. RULE 1/2/3/4 logic in `_maybe_update_exits` works for `trail_tightness_mult` (just one-way for `lock_threshold_mult` — see H2).
