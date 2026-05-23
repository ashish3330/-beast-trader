# PRE-MONDAY AUDIT 4/6 — SAFETY LAYER FIRE-CONDITIONS

Date: 2026-05-23  
Scope: Verify the 7 safety layers wired today are correctly armed and ordered.  
Method: Code-walk + synthetic boundary tests.  
Read-only (no live trades, no edits).

---

## EXECUTIVE SUMMARY

| Layer                   | Code Path                              | Verdict |
|-------------------------|----------------------------------------|---------|
| 1. HARD_DOLLAR_CAP      | `executor.py:1417-1455`                | OK |
| 2. AVG_WIN_LOSS_CAP     | `executor.py:1463-1485`                | DISABLED by config (intentional) |
| 3. PEAK_GIVEBACK        | `executor.py:1387-1408`                | OK |
| 4. EarlyLossCut (tier)  | `executor.py:1492-1558`                | OK |
| 5. POST_BIG_WIN cooldown| `brain.py:982-1015`                    | **BUG** in direction selection (see L5 below) |
| 6. LOSS_STREAK cooldown | `brain.py:1029-1062`                   | OK |
| 7. MARGINAL_TRAIL       | `executor.py:395-423, _MARGINAL_TRAIL` | OK |

Plus a `JUST_CLOSED 180s` hard re-entry floor in `brain._process_symbol` (line 1401-1406) and an executor-level `EXECUTOR_MIN_REENTRY_SECS` floor (line 462). Both are belt-and-braces around the cooldown system.

**One real bug found** plus **two scaling recommendations** for the $5K Monday account.

---

## ORDER OF EXECUTION

### Per-cycle in `brain.run_cycle()`
```
1.  warmup gate (cycles 1-5 — no entries)
2.  push RL adjustments + regime → executor                  (brain.py:917-944)
3.  EXIT MANAGEMENT — runs BEFORE entries (race-safety)     (brain.py:946-1066)
        for each symbol with a position:
          executor.manage_trailing_sl(symbol)
              → for each open sub:
                  executor._apply_trail(symbol, pos, trail_steps, key)
                      ORDER inside _apply_trail:
                        a) compute profit_r + cur_peak                  (lines 1360-1381)
                        b) PEAK_GIVEBACK check                          (1383-1408)
                        c) HARD_DOLLAR_CAP — single + portfolio         (1410-1455)
                        d) AVG_WIN_LOSS_CAP                             (1457-1485)
                        e) EarlyLossCut (tier-dispatched)               (1487-1558)
                        f) trail-step move (lock/be/trail/reduce_sl)    (1560-1733)
          _check_m15_reversal_exit(symbol)
        if position closed this cycle:
          read _last_close_peak_r, _last_close_pnl, _directions    [BUG: _directions popped]
          decide was_win (peak_r >= 0.5)
            → was_win:
                is_big_win? (peak_r >= 10R OR pnl >= $15)
                  YES → arm POST_BIG_WIN cooldown (30min)
                  NO  → arm small-win cooldown (15min, same-direction-only)
            → was_loss:
                track in _loss_history
                if recent_losses >= 2 in 4h window:
                   extend cd_secs to LOSS_STREAK_COOLDOWN_SECS (1h)
                arm cooldown BOTH-directions
4.  ExitIntelligence.evaluate_exits()                                  (brain.py:1068-1072)
5.  ENTRY PROCESSING                                                   (brain.py:1074+)
        for each symbol:
          _process_symbol → reads JUST_CLOSED (180s) + cooldown gate
```

**Race-condition status:** The 2026-05-14 reordering (exits before entries + `_just_closed`
180s + `_arm_cooldown` same-cycle) defends the DJ30 close-and-reopen race that broke last week.
Confirmed across `brain.py:917-1086`.

**No layer is short-circuited inside `_apply_trail`**: each safety section uses its own
try/except scoped only around its own code, and every fire path calls
`self.close_position(...); return`, so trail-step logic at line 1560+ never runs once a
safety layer fires. Good.

---

## PER-LAYER FIRE-CONDITION TABLE

| # | Layer | Trigger | $-equivalent @ $5K | Action on fire | Per-symbol override |
|---|-------|---------|--------------------|----------------|---------------------|
| 1 | HARD_DOLLAR_CAP | `unrealized < -equity × 0.04` (single) `or aggregate < -equity × 0.10` (portfolio) | -$200 / -$500 | `close_position("HardDollarCap")` or `close_all("PortfolioHardCap")` | none — global only |
| 2 | AVG_WIN_LOSS_CAP | `pos.profit < -max(MULT × avg_win, MIN_DOLLAR)` AND 5+ wins in last 30d | dynamic | `close_position("AvgWinLossCap")` | none |
| 3 | PEAK_GIVEBACK | `cur_peak >= trigger_R AND profit_r < cur_peak × frac` | dollar-agnostic | `close_position("PeakGiveback")` | `PEAK_GIVEBACK_PER_SYMBOL` — 9 syms override to (1.0R, 0.4) |
| 4 | EarlyLossCut | `profit_r <= tier_trig AND cur_peak < 0.3 AND streak >= tier_cycles` | varies | `close_position("EarlyLossCut_T*")` | `EARLY_EXIT_DISABLED_SYMBOLS` (empty), score-tier dispatch via `_get_entry_score` |
| 5 | POST_BIG_WIN | (peak_r ≥ 10R OR pnl ≥ $15) on close | $15 = **0.3% @ $5K** | `_arm_cooldown(BIG_WIN_..., 1800s)` | none |
| 6 | LOSS_STREAK | ≥ 2 losses on symbol in 4h window | n/a | `_arm_cooldown(LOSS_STREAK_..., 3600s, BOTH)` | none (global `LOSS_STREAK_COUNT`) |
| 7 | MARGINAL_TRAIL | `0 < entry_score < 7.0` (resolves trail profile) | n/a | swaps `trail_steps` to aggressive scalp lock | none — score-tier only |

Plus implicit:
- **JUST_CLOSED 180s** (`brain.py:1401`) — blocks re-entry for 180 s after any close.
- **EXECUTOR_MIN_REENTRY_SECS** (`executor.py:462`) — second floor at entry layer.

---

## BOUNDARY-CASE TEST RESULTS

### L1 — HARD_DOLLAR_CAP @ $5K equity
```
pct = 0.04 → per-pos cap = -$200.00, portfolio cap (2.5×) = -$500.00
  unrealized = -$199.99 → fires=False  (correct: strict <)
  unrealized = -$200.00 → fires=False  (boundary equal → NOT fire)
  unrealized = -$200.01 → fires=True   (correct)
  unrealized = -$300.00 → fires=True
  agg       = -$499.99 → fires=False
  agg       = -$500.01 → fires=True   → triggers close_all
```
Verdict: **correct**. Caveat: comparison is strict `<`, so an exact -$200.00 won't fire. In practice MT5 unrealized is a float — equal is functionally never seen.

### L3 — PEAK_GIVEBACK
```
Global (0.5R, 0.7):
  peak 0.50R, cur 0.35R → fires=False (boundary)
  peak 0.50R, cur 0.34R → fires=True
  peak 1.00R, cur 0.70R → fires=False (boundary)
  peak 1.00R, cur 0.69R → fires=True
  peak 2.00R, cur 1.40R → fires=False (boundary)
  peak 2.00R, cur 1.00R → fires=True

SWI20.r override (1.0R, 0.4):
  peak 0.90R, cur 0.0R  → fires=False (under trigger)
  peak 1.00R, cur 0.39R → fires=True
  peak 1.50R, cur 0.59R → fires=True
  peak 2.00R, cur 0.81R → fires=False (boundary 0.80R floor)
```
Verdict: **correct**, per-symbol override correctly substitutes both `trigger_R` AND `frac`.

### L4 — EarlyLossCut tier dispatch
```
score=6.0 (marginal):  trig=-0.3R, cyc=8
  -0.3R streak=0  → SKIP (need 8)
  -0.3R streak=8  → FIRE T1-MARGINAL-SCALP
  -0.5R streak=5  → SKIP (still need 8, accumulating)
  -1.0R streak=8  → FIRE T2-MARGINAL (wait=min(10,8)=8)
  -1.5R streak=0  → FIRE T3-IMMEDIATE (wait=0)

score=8.0 (swing):  trig=-1.0R, cyc=20
  -0.3R any streak    → SKIP (outer gate, profit_r > -1.0)
  -1.0R streak=9      → SKIP (need 10)
  -1.0R streak=10     → FIRE T2-SWING
  -1.5R streak=0      → FIRE T3-IMMEDIATE

score=0 (no metadata): treated as swing (safe fallback).
score=6.0 with cur_peak ≥ 0.3R → outer gate skips (won't cut after profit shown).
```
Verdict: **correct**, including the protective `cur_peak < 0.3` outer gate.

### L5 — POST_BIG_WIN
```
R≥10 OR $≥15:
  peak=15R,  pnl=$15    → BIG_WIN=True
  peak=5R,   pnl=$14.99 → BIG_WIN=False (small-win path)
  peak=5R,   pnl=$15    → BIG_WIN=True
  peak=3R,   pnl=$5     → BIG_WIN=False (small-win cooldown)
  peak=0.6R, pnl=$8     → was_win=True, BIG_WIN=False (small-win path)
```
Verdict: **logic correct**.  **HOWEVER — direction selection bug:**

### **BUG (Layer 5): direction lost between close and cooldown arm**

`executor._close_position_impl` pops `self._directions[symbol]` at **line 921** (inside the lock-protected `if any_closed:` block).

`brain.run_cycle()` later (lines 1004-1005) tries to recover the closed direction with:
```python
closed_dir = (self.executor._directions.get(symbol, "BOTH")
              if hasattr(self.executor, "_directions") else "BOTH")
```
Since the entry was popped by `close_position`, this always returns `"BOTH"`. The downstream:
```python
blk = "BOTH" if POST_BIG_WIN_BLOCK_BOTH else closed_dir
```
With `POST_BIG_WIN_BLOCK_BOTH=False` (intentional, per user 2026-05-22), the user's intent is "block only the direction we just won in — opposite direction still allowed (mean-reversion)". The bug makes it block BOTH every time.

**Same bug applies to the small-win branch** at line 1018-1020 (`COOLDOWN_WIN_SECS`).

**Trail evidence:** `executor.py:906` captures `closed_dir = self._directions.get(symbol, "?")` *before* popping at line 921, but it's a local variable and isn't exposed to brain. Other safety paths (broker-side detection at executor.py:1762) save into `self._external_close_direction` — but the trail-driven close path doesn't.

**Recommended fix (one-line):** in `executor._close_position_impl` just before `self._directions.pop(symbol, None)` at line 921, mirror the existing pattern used for `_last_close_peak_r` / `_last_close_pnl`:
```python
if not hasattr(self, "_last_close_dir"):
    self._last_close_dir = {}
self._last_close_dir[symbol] = self._directions.get(symbol, "BOTH")
```
Then in `brain.py:1004` replace the read with:
```python
closed_dir = (getattr(self.executor, "_last_close_dir", {}).get(symbol, "BOTH"))
```
Apply in **both** the BIG_WIN branch (line 1004-1015) and the small-win branch (line 1018-1020).

### L6 — LOSS_STREAK
```
count=2, window=14400s (4h), cooldown=3600s (1h):
  history = [now-100, now-50, now]       → recent=3 → FIRE
  history = [now]                          → recent=1 → SKIP
  history = [now-14401 (outside), now]     → recent=1 → SKIP
  history = [now-14401, now-100, now]      → recent=2 → FIRE
  history = []                             → SKIP
```
Verdict: **correct**, sliding-window prune at brain.py:1047-1048 handles old entries.

### L7 — MARGINAL_TRAIL dispatch (`_resolve_trail_steps`)
```
entry_score = 0     → SWING/REGIME/GLOBAL (safe fallback — missing metadata)
entry_score = 5.99  → MARGINAL
entry_score = 6.99  → MARGINAL
entry_score = 7.0   → SWING/REGIME/GLOBAL (boundary excluded, intentional)
entry_score = 7.01  → SWING/REGIME/GLOBAL
entry_score = 10.0  → SWING/REGIME/GLOBAL
```
Verdict: **correct**, dispatch uses strict `0 < score < 7.0` so missing metadata never accidentally locks scalps.

---

## PER-SYMBOL OVERRIDE VALIDATION

### `PEAK_GIVEBACK_PER_SYMBOL` (config.py:222-236)
- Defined for: SWI20.r, DJ30.r, NAS100.r, SP500.r, JPN225ft, XPTUSD.r, AUDJPY, XAGUSD, XAUUSD — all `(1.0R, 0.4)`.
- **Active universe overlap** (post-concentration to top 5): SP500.r, DJ30.r → both have overrides. US2000.r, USOUSD, XAUUSD: US2000.r and USOUSD have **no override** → global (0.5R, 0.7) applies. XAUUSD has the override.
- Lookup at executor.py:1393-1395 uses `.get(symbol, (PEAK_GIVEBACK_TRIGGER_R, PEAK_GIVEBACK_FRAC))` — correct fallback.
- **Recommendation:** add **US2000.r** and **USOUSD** to `PEAK_GIVEBACK_PER_SYMBOL` if the user wants the looser (1.0R, 0.4) profile across the live top-5. Currently they're stuck on the tight 0.5R/0.7 default, which will close earlier than the index siblings.

### `EARLY_EXIT_DISABLED_SYMBOLS` (config.py:247)
- Currently `set()` (empty). All symbols get tiered EarlyLossCut.
- Code at executor.py:1518 reads `symbol not in _EX_DISABLED` — correct.
- No bug; this is a deliberate kill-switch for symbols where EarlyLossCut over-fires.

### `VWAP_BUFFER_PER_SYMBOL` (config.py:992-1000)
- Per-active-symbol settings:
    - DJ30.r → 1.0 (filter ON)
    - SP500.r → 0.0 (filter OFF)
    - US2000.r → 0.0 (OFF)
    - USOUSD → 0.0 (OFF)
    - XAUUSD → **not in the dict** → falls back to default 0.5 at brain.py:1819
- Lookup at brain.py:1819 reads `.get(symbol, 0.5)` — fallback default is 0.5, not 0.0. So XAUUSD silently uses 0.5 ATR. The other 4 syms get explicit values. Verify with user intent — XAUUSD was added today (commit f0deba0) and may need an explicit entry.
- **Recommendation:** add explicit `"XAUUSD": 0.5` to make intent grep-able, OR confirm 0.5 default is correct.

---

## $5K-ACCOUNT ADJUSTMENT RECOMMENDATIONS

| Threshold | Current | Equity-implied | Verdict @ $5K | Action |
|-----------|---------|----------------|---------------|--------|
| `HARD_DOLLAR_CAP_PCT` | 4% | -$200 single / -$500 portfolio | OK — > 2 × MAX_RISK_PER_TRADE_PCT (2%) = $100 risk per trade | keep |
| `POST_BIG_WIN_DOLLAR_THRESHOLD` | $15 | **0.3% of equity** | **Too easy to trigger.** At $5K, 2% risk = $100 per trade, so 1R win on a trade with 1:7 RR is $14, and even 1R wins on a normal trade easily push past $15. Will fire on nearly EVERY positive close. | **Scale: $15 → $50** (1% of equity) or `max($15, 0.01 × equity)` |
| `POST_BIG_WIN_R_THRESHOLD` | 10R | n/a | OK — R is equity-invariant | keep |
| `AVG_WIN_LOSS_CAP_MIN_DOLLAR` | $1 | n/a | disabled, but at $5K with $50+ avg wins the $1 floor would be irrelevant anyway | keep |
| `LOSS_STREAK_COUNT` | 2 | n/a | OK | keep |
| `LOSS_STREAK_WINDOW_SECS` | 4h | n/a | OK | keep |
| `PEAK_GIVEBACK_TRIGGER_R` | 0.5R | n/a | OK — R-relative | keep |
| `EARLY_EXIT_MARGINAL_TRIGGER_R` / `_SWING_TRIGGER_R` | -0.3 / -1.0 | n/a | OK | keep |
| `DAILY_LOSS_LIMIT_PCT` / `DD_*` | 5% / 8-12% | $250 / $400-$600 | OK | keep |
| `DAILY_HARD_STOP_PCT` / `WEEKLY_HARD_STOP_PCT` | 40% / 50% | $2000 / $2500 | effectively disabled by user 2026-05-13 | keep (per existing policy) |

### Critical action item

**`POST_BIG_WIN_DOLLAR_THRESHOLD = 15.0`** was set on a $1.2K account where $15 = 1.25% of equity = a meaningful trade. At $5K Monday, $15 is 0.3% — every single non-trivial winning close will arm a 30-min cooldown on that symbol's direction. Combined with the new doubled risk (2%, average win-per-trade scales up), this dollar-threshold becomes ~always-true and the cooldown becomes a permanent state for any working symbol.

Suggested replacement (config-only, no code change):
```python
POST_BIG_WIN_DOLLAR_THRESHOLD = 50.0   # 2026-05-23: scaled for $5K acct (1% of equity)
```
OR even better — make it equity-relative:
```python
POST_BIG_WIN_DOLLAR_PCT = 0.01   # 1% of equity
```
and compute the threshold at read time. Defer this until equity is confirmed to be at $5K Monday morning; if Monday opens at $1.2K, keep $15.

---

## LOGGING COMPLETENESS

| Layer | Log call | Severity | Symbol-tagged | Contains key state |
|-------|----------|----------|---------------|---------------------|
| L1 single | executor.py:1431 | warning | yes | $loss, $cap, pct, equity |
| L1 portfolio | executor.py:1443 | warning | (`PORTFOLIO`) | $agg, $threshold, n_positions |
| L2 | executor.py:1477 | warning | yes | unrealized, cap, avg_win, mult, floor |
| L3 | executor.py:1400 | warning | yes | peak_R, current_R, retrace%, trig, frac |
| L4 | executor.py:1542 | warning | yes | tier (T1/T2/T3), profit_R, streak, peak_R |
| L5 | brain.py:1008 | info | yes | peak_R, $pnl, cooldown_h, direction |
| L6 | brain.py:1051 | warning | yes | n_recent, window_min, cooldown_h |
| L7 | (resolve only) | none | — | dispatch is silent — `_apply_trail` logs the trail-step move when one fires |

**Gaps:**
1. **L5 (POST_BIG_WIN) uses `log.info`** — not `warning`. If the user is grepping `grep WARN` to see safety fires, BIG_WIN events are invisible. Recommend `log.warning`.
2. **L7 (MARGINAL_TRAIL dispatch) has no log line at all.** When marginal trail is selected, there's no breadcrumb. Adding a single `log.debug("[%s] trail=MARGINAL (score=%.2f)", symbol, entry_score)` in `_resolve_trail_steps` would help.
3. The `_arm_cooldown` log at brain.py:294 is suppressed when `was_active=True` (cooldown already on). That's fine because the warning that *just* called arm (BIG_WIN / LOSS_STREAK) has already logged the reason. No action needed.

### Exit-reason normalization for RL
rl_learner.py:945-954 collapses `EarlyLossCut_T1-…/T2-…/T3-…` into `EarlyLossCut`. **Good.**
However `HardDollarCap`, `PortfolioHardCap`, `PeakGiveback`, `AvgWinLossCap` each become their own RL bucket. That may or may not be intentional. If user wants finer-grained learning, leave as-is; if user wants `Safety` as a single bucket, expand normalization. Recommend leaving as-is — each safety layer has different cost characteristics worth tracking separately.

---

## FINDINGS — ACTION ITEMS

| Priority | Item | Effort |
|----------|------|--------|
| **P0** | Fix `closed_dir` read after pop bug (brain.py:1004-1005 reads popped dict) — add `_last_close_dir` capture in `executor._close_position_impl`. Otherwise every safety-driven WIN close blocks BOTH directions, defeating `POST_BIG_WIN_BLOCK_BOTH=False`. | 5-line patch |
| **P1** | Scale `POST_BIG_WIN_DOLLAR_THRESHOLD` from $15 → ~1% of equity for Monday's $5K account. | 1-line config |
| **P2** | Add US2000.r and USOUSD to `PEAK_GIVEBACK_PER_SYMBOL` (or document why they should use tighter global default). | 2-line config |
| **P2** | Add explicit `"XAUUSD": 0.5` to `VWAP_BUFFER_PER_SYMBOL` for grep-ability. | 1-line config |
| **P3** | Promote L5 BIG_WIN log from `info` → `warning`. | 1-line patch |
| **P3** | Add `log.debug` breadcrumb when MARGINAL_TRAIL is selected. | 2-line patch |

---

## CONFIDENCE

- All 7 layers fire under their declared trigger conditions (verified via synthetic dispatch).
- All 7 layers correctly *don't* fire at boundary thresholds.
- Order-of-execution inside `_apply_trail` is correct (PEAK → HARD_CAP → AVG_WIN → EARLY_LOSS → trail) and **no layer short-circuits another** — each uses scoped try/except with `return` after `close_position`.
- Brain-level cooldown ordering (WIN-path-vs-LOSS-path) is structurally sound.
- The one direction-loss bug at brain.py:1004 is the only **functional defect** found; everything else is either equity-scaling tuning or cosmetic logging.

## FILES INSPECTED

- `/Users/ashish/Documents/beast-trader/config.py` (safety knobs, lines 179-330)
- `/Users/ashish/Documents/beast-trader/agent/brain.py` (close-handler + cooldown arm, lines 261-320, 900-1086, 1395-1410)
- `/Users/ashish/Documents/beast-trader/execution/executor.py` (close + _apply_trail + helpers, lines 372-423, 820-960, 1189-1733, 2014-2063)
- `/Users/ashish/Documents/beast-trader/agent/rl_learner.py` (exit_reason normalization, lines 940-1000)
