# Safety Layers Audit — 2026-05-22

Scope: 4 safety layers shipped today + interactions, edge cases, score-tier
dispatch and aggressive trail-profile rewrites. **READ-ONLY**, no edits made.

---

## TOP 3 MOST-IMPACTFUL FINDINGS

### #1 — POST_BIG_WIN_COOLDOWN is effectively DEAD CODE (high impact)
`/Users/ashish/Documents/beast-trader/agent/brain.py:888` reads
`self.executor._peak_profit_r.get(symbol, 0.0)` — but by then `close_position()`
has already popped it at `executor.py:926`. So `last_peak` is **always 0.0** when
the trail-close branch (which is the only place POST_BIG_WIN lives) inspects it.

Live evidence (`logs/dragon.log.1` 18:59:32-33):
```
[JPN225ft] PEAK-GIVEBACK EXIT: peak=2.52R current=0.79R ...
[JPN225ft] CLOSED LONG (PeakGiveback)
[ALERT] pnl=6.1 R=2.524...
[JPN225ft] COOLDOWN ARMED: LOSS_trail_close for 45m blocks=BOTH    ← wrong!
```
This was a clear winner (+2.52R / +$6.1) but cooldown was armed as a LOSS.

Compounding factor: POST_BIG_WIN is ONLY in the trail-close branch
(`brain.py:887-936`). It is **completely absent** from the external/broker-close
branch (`brain.py:1190-1249`), so any TP-hit big win also bypasses it.

Even when `last_pnl` is read (line 907), only `close_position()` writes
`_last_close_pnl` (executor.py:928-930). External-close path does not — so for
broker-side TP hits, both gates (`peak_r >= 3.0`, `last_pnl >= $15`) are 0/0
and POST_BIG_WIN never fires.

Search confirms ZERO `BIG WIN` log lines across `dragon.log`, `dragon.log.1`,
`dragon.log.2` — the feature has never fired in production.

**Fix outline**: in `executor.close_position()` snapshot peak into
`_last_close_peak_r[symbol]` BEFORE the pop (and mirror in external-close path).
Brain reads from `_last_close_peak_r` instead of `_peak_profit_r`. Also wire
POST_BIG_WIN into the broker-close branch.

---

### #2 — `_loss_streak[tracking_key]` is shared across 3 sub-positions → EARLY_EXIT fires 3× faster than designed
`executor.py:1238` calls `_apply_trail(symbol, pos, trail, symbol)` — note the
4th arg is `symbol`, **not** `symbol + "_sub{i}"`. The function is invoked
once per open sub (Sub0/Sub1/Sub2), all mutating the same
`_loss_streak[symbol]` counter:

`executor.py:1517` — `self._loss_streak[tracking_key] = self._loss_streak.get(tracking_key, 0) + 1`

So a single brain cycle increments the streak THREE times. With
`EARLY_EXIT_MARGINAL_CYCLES=8`, the marginal-tier exit fires after
~3 cycles, not 8. With `EARLY_EXIT_SWING_CYCLES=20`, it fires after ~7
cycles. The "10s wait for swing" comment at `config.py:249` is wrong;
actual wait is ~3s.

Same per-sub amplification affects `_rsi_prev`, `_mom_score_prev`, and
`_peak_profit_r` (each sub re-writes the same key in the per-cycle loop).
For peak_r it's harmless (all subs share entry, so `profit_r` is identical
per sub). For RSI/score history it's harmless. For `_loss_streak` it is
material.

Live evidence (`logs/dragon.log.1` 08:50:56-08:56:00, BTCUSD):
EARLY-LOSS-CUT fires every ~15s consistently while `profit_r` drifts
between -0.38R and -0.51R. With 3 subs the design wait of "8 cycles"
collapses to ~3 cycles of real elapsed time (which is the observed
~15-30s cadence between firings).

**Fix outline**: change call at executor.py:1238 to pass
`f"{symbol}_sub{pos_magic - base_magic}"` as tracking_key — or use a
separate counter dict keyed by ticket.

---

### #3 — `lock_threshold_mult` safety clamp (=2.0) interacts badly with high-R lock thresholds in the new aggressive profiles
The safety clamp at `executor.py:1636-1637` caps `lock_threshold_mult` and
`be_threshold_mult` at 2.0. Combined with the new `_RUNNER_NO_BE`
(top step `(8.0, "lock", 7.0)`) and `_TIGHT_LOCK` (top step `(6.0, "lock", 5.5)`)
from `auto_tuned.py:27-69`:

When RL pushes a `lock_threshold_mult=2.0` (its allowed maximum), the
effective top-tier thresholds become 16R / 12R — but `profit_r` is hard
clamped to ±10R at `executor.py:1361`. **Result: the high-R lock steps are
unreachable.** The iteration then falls through to lower steps:
e.g. `_RUNNER_NO_BE` at 5R profit with mult=2.0 would skip
(8.0→16R), (5.0→10R), (3.5→7R), (2.5→5R)... → first match
`(1.8, "lock", 1.4)` (effective 3.6R) → still skip → `(1.3, "lock", 0.95)`
(effective 2.6R) → match → lock 0.95R only.

So a trade at 5R peak with max RL adjustment locks only 0.95R instead of
the intended 4.2R. PEAK_GIVEBACK still saves the trade (closes at
0.7×5R=3.5R retrace), but the executor's trail behavior diverges sharply
from the configured profile.

This is less catastrophic than #1/#2 because PEAK_GIVEBACK is the de
facto profit floor for high-R trades. But the trail profile is no longer
the source of truth, the giveback breaker is.

**Fix outline**: stop multiplying lock R-thresholds by `lock_threshold_mult` in
high-R regime, or cap the multiplier at 1.2 instead of 2.0, or apply
the multiplier ONLY to BE/sub-1R steps.

---

## Layer flow diagram

`_apply_trail(symbol, pos, trail_steps, tracking_key)` runs PER sub-position
each cycle (executor.py:1230-1240 outer loop in `manage_trailing_sl`).
Inside `_apply_trail`, in source-code order:

1. Hydrate `sl_dist` from in-memory cache → DB entry_metadata →
   ATR/trailed fallback (`executor.py:1304-1352`).
2. Compute `profit_r = profit_dist / sl_dist`, clamp to ±10R (line 1361-1362).
3. Update `cur_peak = max(prev_peak, profit_r)`; persist (line 1364-1375).
4. **PEAK_GIVEBACK** check (line 1377-1402) — closes if
   `cur_peak >= trig` AND `profit_r < cur_peak * frac`. Per-symbol override
   from `PEAK_GIVEBACK_PER_SYMBOL`. Returns on close.
5. **HARD_DOLLAR_CAP** per-position check (line 1404-1429) — closes if
   `pos.profit < -equity * 0.020`. Returns on close.
6. **HARD_DOLLAR_CAP** aggregate check (line 1431-1447) — closes ALL positions
   if `sum(pos.profit) < -equity * 0.050`. Returns on close.
7. **AVG_WIN_LOSS_CAP** check (line 1451-1479) — only if `pos.profit < 0` AND
   `_get_avg_win_dollars(symbol)` returns > 0 (>= 5 wins in 30d). Closes if
   loss exceeds `max(avg_win × 2.0, $2.0)`. Returns on close.
8. **EARLY_EXIT** score-tier dispatch (line 1481-1543) — looks up entry score
   via `_get_entry_score(symbol)`. Marginal (<7.0) → -0.3R / 8 cycles. Swing
   (≥7.0 or score=0) → -1.0R / 20 cycles. Catastrophic gate (-1.5R) always
   immediate. Returns on close.
9. Compute RL/momentum adjustments to lock/be/trail multipliers (line 1545-1639),
   clamp lock/be mult at 2.0 (line 1636-1637).
10. Iterate `trail_steps` top-down; first matching `profit_r >= R_effective`
    sets `new_sl` and breaks (line 1641-1670).
11. Profit ratchet floor (line 1675-1689): peak ≥ 2R → floor ≥ 0.5R; peak ≥ 1R → floor ≥ 0.2R.
12. Min-stop clamp + monotonic SL guard (line 1691-1704) → `order_send(action=6)` SLTP modify.

POST_BIG_WIN_COOLDOWN lives OUTSIDE `_apply_trail`. It runs in
`brain.py:887-936`, after `manage_trailing_sl()` returns and only when the
trail-close branch detects `had_pos_before AND NOT has_position()` —
i.e. the trade was closed by one of layers 4-8 above (or trail-SL hit).

---

## CRITICAL bugs (P0)

- **brain.py:888 reads stale `_peak_profit_r`** — already popped by
  `executor.py:926`. POST_BIG_WIN_COOLDOWN gate-r path is dead.
  *Live evidence: zero `BIG WIN` log lines in 3 rotated logs (>72h).*

- **POST_BIG_WIN absent from broker-close branch** —
  `brain.py:1190-1249` arms only WIN/LOSS asymmetric cooldown for TP hits.
  Any broker-side TP fill bypasses POST_BIG_WIN entirely.

- **`_last_close_pnl` only set in close_position()** — `executor.py:928-930`.
  External-close path (`_record_external_close_immediate`, lines 1818-1914)
  stashes `_external_close_pnl[symbol]` but never `_last_close_pnl`. So
  `brain.py:907` reads 0 for TP-hit closes → dollar-threshold path also dead.

- **`_loss_streak` shared across sub-positions** — `executor.py:1238`
  passes `symbol` (not `symbol+sub_idx`) as `tracking_key`. EARLY_EXIT fires
  ~3× faster than configured.

---

## INTERACTION CONFLICTS

- **PEAK_GIVEBACK wins over trail SL** by design. With `_RUNNER_NO_BE`
  locking 7R at 10R peak, and PEAK_GIVEBACK firing at 70% retrace (= drop
  to 7R), they intersect at the same level. PEAK_GIVEBACK closes at
  market while trail waits for SL touch — PEAK_GIVEBACK always wins the
  race, making the high-R lock thresholds redundant for the protective
  function (they only matter if PEAK_GIVEBACK_ENABLED=False).

- **AVG_WIN_LOSS_CAP fires before EARLY_EXIT for most symbols**.
  Trade journal (rolling 30d):
    - BTCUSD avg_win=$0.82 → cap=max($1.64, $2)=$2.0
    - ETHUSD avg_win=$0.58 → cap=$2.0 (floor)
    - JPN225ft avg_win=$0.57 → cap=$2.0
  $2.0 loss on a small-lot position is often hit BEFORE the -0.3R
  marginal trigger AND BEFORE the 8-cycle wait. So
  AVG_WIN_LOSS_CAP effectively short-circuits the score-tiered
  EARLY_EXIT logic on small-dollar symbols.

- **HARD_DOLLAR_CAP fires later than AVG_WIN_LOSS_CAP for normal lots**.
  HARD = 2% × $1.3K = $26. AVG_WIN cap = $2-$12. AVG wins on a
  per-position basis but the code-order is HARD first then AVG. On
  gap-through losses where both thresholds breach in the same cycle,
  HARD fires first (line 1424 before line 1469). On gradual losses
  AVG fires first because its threshold is lower. **Net: works as
  intended** but the code-order suggests reversal would be cleaner
  (run AVG first so HARD is the catastrophic fallback only).

- **POST_BIG_WIN BOTH + WIN_dir directional stacking** — `_arm_cooldown`
  at brain.py:255-291 uses `max(existing, new_expiry)`; BOTH dominates
  directional. So order doesn't matter — same final state. **No conflict.**

- **PEAK_GIVEBACK + EARLY_EXIT cannot both fire** — they each `return`
  after `close_position()`. Whoever runs first in source-code order
  wins. PEAK_GIVEBACK runs first (line 1392) so it preempts. EARLY_EXIT
  only matters when `cur_peak < 0.5R` (PEAK_GIVEBACK_TRIGGER_R), as
  written in the EARLY_EXIT condition `cur_peak < 0.3` at line 1514.

---

## EDGE CASES NOT HANDLED

- **`_last_close_pnl[symbol]` missing on first close after restart** —
  `executor.py:928` guards with `hasattr`. `brain.py:907` uses
  `.get(symbol, 0.0)`. So missing → `last_pnl=0` → POST_BIG_WIN gate
  fails on pnl. (Combined with #1, the whole feature stays inert.)

- **`_get_avg_win_dollars(symbol)` returns 0** when < 5 wins in 30d.
  AVG_WIN_LOSS_CAP then NO-OPS. New / restarted / recently-added symbols
  get no avg-win protection — only HARD_DOLLAR_CAP and EARLY_EXIT.
  Probably intentional, but worth noting.

- **peak_r > 10R glitch** — clamped at `executor.py:1361-1362`. Even if
  ATR=0 / sl_dist=0 spike happens, peak ≤ 10R is enforced. PEAK_GIVEBACK
  with default frac=0.7 fires at 7R retrace from a 10R peak. Same with
  per-symbol overrides at frac=0.4 (firing at 4R from 10R). Worst case
  closes a real runner early but not catastrophic.

- **3-sub-position cap firing per-pos vs per-symbol** — HARD_DOLLAR_CAP
  and AVG_WIN_LOSS_CAP check `pos.profit` (single sub), not the symbol
  aggregate. With 3 subs each at -$3 (total -$9 on a symbol with
  avg_win=$1.90, cap=$3.80), neither fires per-sub. **Aggregation gap**
  for AVG_WIN — could under-protect symbols whose loss is spread
  across subs. HARD_DOLLAR_CAP has the portfolio aggregate check
  (line 1431-1445) so its symbol-level aggregate gap is partially
  covered.

- **Entry metadata absent after symbol re-enabled mid-trade** —
  `_get_entry_score` returns 0 → `is_marginal=False` (because
  `entry_score > 0` part of the AND is False) → falls into swing tier.
  Restart-orphaned positions silently get swing treatment. Conservative
  but inconsistent.

- **PEAK_GIVEBACK trigger=0.5R vs EARLY_EXIT `cur_peak < 0.3`** — there's
  a narrow window (peak 0.3-0.5R) where neither PEAK_GIVEBACK nor
  EARLY_EXIT will fire (PEAK_GIVEBACK requires cur_peak ≥ 0.5;
  EARLY_EXIT requires cur_peak < 0.3). So a trade that reached +0.4R
  then reversed to -0.5R is only protected by HARD_DOLLAR_CAP and
  AVG_WIN_LOSS_CAP. Probably OK because those will catch it.

---

## SCORE-TIER DISPATCH CORRECTNESS

- `_get_entry_score(symbol)` (`executor.py:1999-2011`) reads
  `state.get_agent_state()['entry_metadata'][symbol]['score']`. The
  metadata is persisted to SQLite (`brain.py:360-373` writes,
  `337-358` reads on startup). 72h TTL. So **survives restart** within
  the TTL window.

- However on bot restart there is a race: `_load_entry_metadata` runs
  during `__init__`, but `state.get_agent_state()` reads
  `agent_state` which is set via
  `self.state.update_agent("entry_metadata", ...)` at brain.py:356.
  Confirmed wired. **No race observed in code.**

- Marginal tier dispatch logic (line 1504):
  `is_marginal = entry_score < _STIER and entry_score > 0`. When score=0
  (metadata missing OR genuine 0), `is_marginal=False` → swing tier.
  Reasonable fallback.

- `_resolve_trail_steps` at line 395-423 also keys on score-tier:
  marginal score (0 < score < 7) → `_MARGINAL_TRAIL`. **Same gate as
  EARLY_EXIT** so both layers consistently classify a position.

- Live log confirms `T1-MARGINAL-SCALP` firings for BTCUSD (8 cycles
  threshold), DJ30.r (8 cycles threshold). Tier dispatch works.

---

## AGGRESSIVE TRAIL PROFILE CORRECTNESS

- Profiles `_TIGHT_LOCK`, `_AGGR_LOCK`, `_RUNNER_NO_BE` parsed
  successfully — `auto_tuned.py:27-69`. Verified at runtime with
  `python3 -c "import auto_tuned; print(auto_tuned._TIGHT_LOCK)"` —
  10/9/9 tuples respectively, all `(float, str, float)` form.

- `_MARGINAL_TRAIL` lives in executor.py:382-393 (class attribute,
  not auto_tuned.py). 10 tuples. Syntactically correct.

- Trail iteration breaks on FIRST match (line 1670). Profile order is
  descending R-threshold so top steps fire first when reached. Correct.

- High-R steps `(6.0, "lock", 5.5)`, `(8.0, "lock", 7.0)` are
  UNREACHABLE in practice because:
    (a) `profit_r` is clamped to ±10R (executor.py:1361-1362),
    (b) `lock_threshold_mult` cap at 2.0 multiplies thresholds,
  combined effect: a 7R-lock step at 8R threshold can only fire if RL
  pushes lock_mult down (<1.25). If RL stays at default 1.0, it fires
  at profit_r ≥ 8R, which is achievable but rare.

- `lock_threshold_mult=2.0` × `(0.7, "lock", 0.45)` → effective 1.4R
  threshold. So with maximum RL widening, a 5R-peak trade locks 0.45R
  via this step. PEAK_GIVEBACK at frac=0.7 fires at 3.5R retrace from
  5R peak — closes at market before the trail can give back further.
  PEAK_GIVEBACK is the de facto floor.

---

## CONFIRMED WORKING (with log evidence)

- **PEAK_GIVEBACK fires at configured threshold** —
  `logs/dragon.log` 20:07:08:
  `[SP500.r] PEAK-GIVEBACK EXIT: peak=1.23R current=0.37R (retraced 70%, trigger=1.0R frac=0.40)`
  Matches `PEAK_GIVEBACK_PER_SYMBOL['SP500.r']=(1.0, 0.4)`.

- **PEAK_GIVEBACK uses global default when no per-symbol override** —
  `logs/dragon.log` 19:56:19:
  `[BTCUSD] PEAK-GIVEBACK EXIT: peak=0.66R current=0.45R (retraced 32%, trigger=0.5R frac=0.70)`
  Matches `PEAK_GIVEBACK_TRIGGER_R=0.5, PEAK_GIVEBACK_FRAC=0.7`.

- **EARLY-LOSS-CUT score-tier dispatch labels correctly** —
  `T1-MARGINAL-SCALP` for sub-1R triggers, `T2-MARGINAL/SWING` for -1R,
  `T3-IMMEDIATE` for -1.5R. Confirmed by tier string in log.

- **PEAK_GIVEBACK closes via `close_position` and PnL/peak are alerted
  correctly** (post 2026-05-14 fix at executor.py:931-941). Alerter
  log shows real `pnl=6.1 R=2.524` rather than 0.0/0.0.

- **AVG_WIN_LOSS_CAP code path is wired** (config flag default True,
  log code at executor.py:1470-1476). **NOT YET FIRED in production logs**
  searched — likely because EARLY_EXIT marginal tier (-0.3R) or
  HARD_DOLLAR_CAP fires first on most losing trades. To validate, would
  need a setup where avg_win × 2 < intended position SL.

- **Trail-close branch reaches the brain cooldown arming logic** —
  JPN225ft 18:59:33 sequence (PEAK-GIVEBACK → CLOSED → COOLDOWN ARMED)
  proves the `had_pos_before AND NOT has_position()` check at
  brain.py:887 fires. (Just routes to wrong branch — see #1.)

- **Auto_tuned trail profiles loaded and used** — verified via
  `_resolve_trail_steps` priority chain; `SYMBOL_REGIME_TRAIL_OVERRIDE`
  cells from auto_tuned.py:72-85 take precedence when regime is set.

---

## REFERENCED FILES

- `/Users/ashish/Documents/beast-trader/execution/executor.py`
  - `_apply_trail`: lines 1292-1719
  - PEAK_GIVEBACK: 1377-1402
  - HARD_DOLLAR_CAP: 1404-1449
  - AVG_WIN_LOSS_CAP: 1451-1479
  - EARLY_EXIT: 1481-1543
  - `_get_entry_score`: 1999-2011
  - `_get_avg_win_dollars`: 2013-2048
  - `_resolve_trail_steps`: 395-423
  - `_MARGINAL_TRAIL`: 382-393
  - `close_position` pop site: 905-926
  - `_last_close_pnl` set: 928-930
  - tracking_key call sites: 1238, 1240
  - lock_threshold_mult clamp: 1636-1637
  - profit_r clamp: 1361-1362
- `/Users/ashish/Documents/beast-trader/agent/brain.py`
  - POST_BIG_WIN block: 887-936
  - `_arm_cooldown`: 255-291
  - external-close cooldown branch: 1190-1249
  - entry_metadata persist: 360-373
  - entry_metadata load: 337-358
- `/Users/ashish/Documents/beast-trader/config.py`
  - PEAK_GIVEBACK_*: 181-186
  - PEAK_GIVEBACK_PER_SYMBOL: 217-231
  - AVG_WIN_LOSS_CAP_*: 193-199
  - POST_BIG_WIN_*: 204-210
  - EARLY_EXIT_*: 236-249
  - HARD_DOLLAR_CAP_*: 262-263
- `/Users/ashish/Documents/beast-trader/auto_tuned.py`
  - `_TIGHT_LOCK`: 27-41
  - `_AGGR_LOCK`: 45-56
  - `_RUNNER_NO_BE`: 57-69
  - TRAIL_OVERRIDE_REGIME_AUTO: 72-85
  - TRAIL_OVERRIDE_AUTO: 240-267
