# Entry Pipeline Audit — 2026-05-22

Scope: `agent/brain.py` `_process_symbol` (lines 1083-2081) and the upstream cycle
that feeds it (`_run_cycle` lines 647-1077). All line refs are READ-ONLY analysis;
no code was modified.

## Gate sequence (definitive ordering)

The brain runs gates in the following order. Items prefixed `[CYCLE]` are checked
once per cycle in `_run_cycle` before `_process_symbol` is even called.

| # | File:line | Gate | Checks | On skip |
|---|-----------|------|--------|---------|
| C1 | brain.py:732-743 | DAILY_KILL_SWITCH | `daily_loss_pct >= DAILY_HARD_STOP_PCT` and equity>0 | Sets kill-switch, calls `close_all`, returns from cycle |
| C2 | brain.py:747-759 | WEEKLY_KILL_SWITCH | `weekly_loss_pct >= WEEKLY_HARD_STOP_PCT` and equity>0 | Same, weekly |
| C3 | brain.py:762-810 | KILL_SWITCH_ACTIVE_HOLD | `_kill_switch_active` | Returns from cycle entirely — only trail mgmt runs |
| C4 | brain.py:813-818 | EMERGENCY_DD | `dd_pct >= DD_EMERGENCY_CLOSE` | `close_all`, sleep 10s, return |
| C5 | brain.py:820-832 | WARMUP | `_cycle <= 5` | Returns from cycle, dashboard-only scoring |
| C6 | brain.py:863-938 | TRAIL/EXIT MGMT (before entries) | Per-symbol position management; arms post-close cooldowns | n/a — runs before entries |
| C7 | brain.py:940-945 | EXIT_INTELLIGENCE | Calls `evaluate_exits()` | n/a |
| C8 | brain.py:952-959 | Per-symbol loop → `_process_symbol` | — | — |
| 0 | brain.py:1114-1191 | PULLBACK FILL/EXPIRY | If `_pending_pullback[symbol]` exists, check fill or expiry | returns PULLBACK_WAIT / PULLBACK_FALLBACK |
| 1 | brain.py:1200-1249 | EXTERNAL_CLOSE_DETECT | Broker-side close marker → arms cooldown | n/a — side-effect only |
| 2 | brain.py:1258-1263 | EARLY_COOLDOWN (BOTH-side) | `_cooldown_blocked == "BOTH"` and active and `not has_position` | returns SL_COOLDOWN |
| 3 | brain.py:1271-1276 | JUST_CLOSED 180s | `executor._just_closed[symbol]` within 180s | returns JUST_CLOSED |
| 4 | brain.py:1283-1285 | H1_DATA | `len(h1_df) < H1_MIN_BARS(100)` | returns NO_H1_DATA |
| 5 | brain.py:1288-1297 | CANDLE_CLOSE_CACHE | `EVAL_ON_CANDLE_CLOSE` and same candle as last cycle | returns cached `_last_scores` |
| 6 | brain.py:1300-1306 | INSUFFICIENT_IND | `bi < 21` or ATR NaN/0 | returns INSUFFICIENT_IND |
| 7 | brain.py:1310 | REGIME_DETECT | `_get_regime_from_bbw` | n/a |
| 8 | brain.py:1320-1335 | SCORE + min_quality | `signal_quality < min_quality` per regime/symbol | (handled in #9) |
| 9 | brain.py:1344-1352 | RL_QUALITY_BONUS | RL streak bonus increases `min_quality` for non-proven syms | n/a — modifies threshold |
| 10 | brain.py:1355-1367 | BELOW_MIN_SCORE | direction selection vs `min_quality` | returns BELOW_MIN_SCORE |
| 11 | brain.py:1378-1392 | CONFIRM_MISSING | trending regime + (supertrend, breakout, trend_persist) all 0 | returns CONFIRM_MISSING |
| 12 | brain.py:1410-1414 | POSITION_OPEN | `executor.has_position(symbol)` | returns POSITION_OPEN |
| 13 | brain.py:1423-1429 | BAR_REENTRY | same H1 bucket as last entry | returns BAR_REENTRY |
| 14 | brain.py:1441-1451 | LATE_MOMENTUM | score≥7.0 now but any of last 3 < 6.0 | returns LATE_MOMENTUM |
| 15 | brain.py:1454-1458 | SESSION | non-Crypto and `hour_utc` outside session | returns SESSION |
| 16 | brain.py:1473-1516 | DIR_BIAS (adaptive) | direction != allowed, with RL-WR or A+ override | returns DIR_BIAS |
| 17 | brain.py:1522-1527 | DIRECTIONAL_COOLDOWN | post-win same-dir cooldown active for direction | returns COOLDOWN |
| 18 | brain.py:1532-1544 | RL_SKIP | RL has ≥8 trades in (regime, hour, dir) with WR<15% | returns RL_SKIP |
| 19 | brain.py:1547-1556 | TOXIC_HOUR | global or per-symbol toxic-hour set | returns TOXIC_HOUR / TOXIC_HOUR_SYM |
| 20 | brain.py:1561-1574 | CALENDAR | high-impact news; hard-block opt-in via CALENDAR_HARD_BLOCK_SYMBOLS | returns CALENDAR (only if in opt-in set) |
| 21 | brain.py:1579-1593 | FVG_BIAS | warn-only — never returns | n/a |
| 22 | brain.py:1600-1620 | TREND_FILTER | H1 EMA200; hard-block opt-in via TREND_FILTER_HARD_BLOCK_SYMBOLS | returns TREND_FILTER (only if in opt-in set) |
| 23 | brain.py:1622-1652 | MTF_CASCADE | W1/D1/H4 verdict==REJECT | returns MTF_CASCADE |
| 24 | brain.py:1660-1681 | RANGE_EXTREME | ranging regime + at recent range extreme | returns RANGE_EXTREME |
| 25 | brain.py:1683-1712 | VWAP_WRONG_SIDE | close > ±buf*ATR from VWAP on wrong side | returns VWAP_WRONG_SIDE |
| 26 | brain.py:1714-1735 | FIB_ZONE | per-symbol fib-as-filter; outside [zone_lo, zone_hi] | returns FIB_ZONE |
| 27 | brain.py:1738-1750 | POSITION_HOLD/REVERSAL | redundant `executor.get_position_direction` check | returns HOLD_SWING or REVERSAL_DISABLED |
| 28 | brain.py:1756-1768 | M15_DISAGREE | M15 != direction unless quality bypass | returns M15_DISAGREE |
| 29 | brain.py:1776-1783 | META_REJECT | ML meta-label prob below tier threshold | returns META_REJECT |
| 30 | brain.py:1797-1814 | MASTERBRAIN | `master_brain.evaluate_entry` returns approved=False | returns MASTER_REJECT |
| 31 | brain.py:1824-1912 | RISK SIZING | adaptive_mult, protect_mult, portfolio_mult, DD halve | n/a (modifies risk_pct) |
| 32 | brain.py:1934-1972 | MIN_EDGE | friction% > cap unless A+ | returns MIN_EDGE_REJECT |
| 33 | brain.py:1984-1998 | EV_REJECT | `ev_after_cost < threshold` unless A+/65%+ | returns EV_REJECT |
| 34 | brain.py:2014-2039 | PULLBACK ARM | `regime in PULLBACK_REGIMES` or re-entry → defer | returns PULLBACK_WAIT |
| 35 | brain.py:2044-2076 | EXECUTE | `executor.open_trade` → V5 ENTRY log | returns ENTERED / EXEC_FAILED |

## CRITICAL (broken / silently skipping trades)

1. **brain.py:2070 — `NameError: name 'atr' is not defined` on every direct entry.**
   `"atr_at_entry": float(atr) if atr else 0.0` references `atr`, which is never
   assigned in `_process_symbol`'s scope (only `atr_val` at 1324 and `smart_atr`
   at 2006 exist). The block sits inside `if success:` without a try/except.
   The exception bubbles up to `_run_cycle`'s outer `except Exception as e:`
   at brain.py:958 → `log.error("[%s] Process error: %s", ...)`. Net effect:
   `_last_entry_bar`, alerter callback, entry_metadata persistence, AND the
   V5 ENTRY log line all fail silently for every direct (non-pullback) entry
   that succeeds. Pullback-fill path (line 1138-1145) does not have this bug
   because it omits `atr_at_entry`. Use `atr_val` (the actual variable).

2. **Pullback-armed direction can no longer reach EXECUTE.**
   Line 2015: `if use_pullback and symbol not in self._pending_pullback:` — once
   pullback is armed, the function returns PULLBACK_WAIT. Direct execute at
   line 2044 is only reached when `use_pullback=False` (pullback disabled OR
   regime not in {trending, volatile} AND not re-entry). Combined with the
   2026-05-22 change `PULLBACK_ATR_RETRACE 0.2 → 0.8` + `MAX_WAIT_BARS 1 → 5`,
   a much larger fraction of signals will park in pullback. The fallback at
   line 1156-1188 fires after `_pb_wait_eff` *bars*, which on H1 is 5 hours
   max — long enough for the original signal to be stale. This is intentional
   per the research note but watch for cases where signal expires *before*
   the bar count due to brain-restart loss of `_pending_pullback`.

3. **`_pending_pullback` is in-memory only — restarts wipe pending entries.**
   No SQLite persistence parallel to `entry_metadata` / `kill_switch`. Restart
   between signal and fill = lost signal. Real-money risk now that retrace was
   widened to 0.8 ATR (less likely to fill in one bar).

4. **Gate 27 (POSITION_HOLD/REVERSAL at 1738-1750) is a redundant re-check.**
   Gate 12 already returned at 1410-1414 on `has_position`. By the time we reach
   1738, `has_position()` must be False (or we'd already have returned). The
   `current_dir == direction` and `current_dir != direction` branches at
   1741-1750 are dead — `current_dir` is always "FLAT" here. Two extra rpyc
   `positions_get` round-trips per cycle for dead branches.

5. **`atr_at_entry` field unreachable from pullback fallback too.**
   Pullback fill (line 1138-1145) and pullback-expiry fallback (1177-1184) both
   build their metadata without `atr_at_entry` or `sl_dist_at_entry`. The
   comment at 2059 ("trail survives bot restart even if ATR has since shrunk")
   only covers the direct-entry path. Pullback entries do not get the restart
   protection.

6. **Calendar `should_skip_entry` runs even when CALENDAR_HARD_BLOCK_SYMBOLS is
   empty.** brain.py:1561-1574. The full calendar query (which may load a JSON
   file or hit network) runs every cycle for every symbol, then the result is
   typically logged as a warning and discarded. With `CALENDAR_HARD_BLOCK_SYMBOLS
   = {}` (current config 930-933), the only effect is the warning log line.
   Move the hard-block-set membership check FIRST.

## HIGH (suboptimal ordering / redundant)

7. **Expensive gates run before cheap dedup.** Cycle warmup (C5), kill-switch
   (C3), and TRAIL/EXIT (C6) run before bar-dedup. Inside `_process_symbol`,
   the order H1_DATA (cheap dict access) → REGIME → SCORE → CONFIRM is fine,
   but **POSITION_OPEN (Gate 12, expensive rpyc) runs AFTER scoring and after
   `_score_with_components`**. Score is computed even when a position is
   already open. Move `executor.has_position(symbol)` ahead of indicator
   compute when possible — though the dashboard wants live scores even for
   held positions.

8. **`has_position(symbol)` is called 3+ times per `_process_symbol` cycle.**
   Lines 1260, 1410, 1523, plus `get_position_direction` at 1738. Each hits
   the rpyc bridge. Cache the result once at function entry into a local var.

9. **Gate ordering: CALENDAR (20) before MTF_CASCADE (23) is fine, but VWAP
   (25) and FIB (26) are O(N) indicator pulls placed AFTER MTF (which is the
   most expensive — 200 H1 bars of rolling aggregation).** If MTF_CASCADE
   would reject, the cheaper VWAP/FIB gates above it could have rejected
   first. Reverse: check VWAP / FIB before MTF.

10. **DIR_BIAS (16) reads `_rl_learner._trade_outcomes[symbol]` directly —
    private attribute access from outside the module.** Should be via accessor;
    minor coupling concern.

11. **LATE_MOMENTUM gate (14) writes `_score_hist[symbol]` BEFORE checking
    the prior history.** Lines 1441-1442: `prev_scores = list(...); self._score_hist[symbol] = (prev_scores + [raw_score])[-5:]`. Because
    `prev_scores` is taken before the append, the current score is correctly
    excluded from the lookback — but the append fires even when we then reject
    in the same call. That's fine for "rolling 5", but means the gate's
    own rejections are visible in subsequent calls (no drift detected, just
    confirmation of intent).

12. **`raw_score` recomputed redundantly.** Set to `max(long, short)` at 1327,
    then reset to whichever direction at 1357/1360. The 1327 value is used
    only for `signal_quality` at 1328, so the reset is fine but `raw_score`
    has two semantically different meanings within the same function.

13. **Score component dict access in CONFIRM_MISSING (line 1383) silently
    fail-opens on exception.** A KeyError on `comp_dir` would set
    `_confirms=1`, passing the gate. This is OK for missing-component data
    but masks legitimate bugs.

14. **`RL_SKIP` (gate 18) runs after DIR_BIAS — RL can override dir-bias but
    not vice versa.** If a symbol is direction-biased LONG but RL has 8+
    trades with <15% WR in (this regime, this hour, LONG), the bias check
    passes (allowed), then RL rejects. Two separate stat sources disagree
    silently. Acceptable but log explicitly when this happens.

15. **MTF_CASCADE reduced mode only triggers when bars<200.** Line 1639. With
    `H1_MIN_BARS=100` enforced at 1284, the cascade runs in reduced (D1, H4
    only) mode for 100 ≤ bars < 200. W1 silently disabled but the log is at
    `debug` level (line 1640) — not visible at INFO.

16. **MIN_EDGE/EV `try/except` swallows ALL errors at debug-level.** Line 2000.
    If `sl_mult_base`, `spread`, or RL EV fetch fails, the trade proceeds
    without these gates. Should at minimum log at WARNING when the gate is
    bypassed due to error.

17. **`base_ret` is built at line 1398-1400 with `min_quality` from regime,
    but the MIN_EDGE/EV gates DO NOT include `min_quality` in their return
    dicts.** Inconsistent dashboard payload depending on where we exit.

18. **`POSITION_OPEN` gate (12) added 2026-05-17 says "Gate 0a" in comment but
    is structurally Gate 0c (after pullback PRE-CHECK and JUST_CLOSED).**
    Naming drift; misleading.

19. **`BAR_REENTRY` gate (13) uses `int(time.time() // 3600)` which is UTC-aligned,
    but the H1 candle bucket from `state.get_candles` may not be — depends on
    data source. If MT5 H1 candles open at non-UTC hours (broker time), the
    bucket can mis-match the actual candle. Verify alignment with
    `tick_streamer.get_candles` to be safe.

20. **`_master_brain.evaluate_entry` (gate 30) increments `_daily_trades`
    inside `evaluate_entry`** (master_brain.py:316-317) — but only when
    `result["approved"]=True`. If the brain later fails MIN_EDGE or EV
    (gates 32-33), the `_daily_trades` counter increments even though no
    trade fires. Inflates the dashboard counter and skews the daily-trade
    soft cap.

## NO-SKIP RULE VIOLATIONS

The user's `feedback_no_skip_trades.md` rule says: skip is only forbidden for
**risk / daily-loss / spread** categories. Other categories may skip.

21. **brain.py:1547-1556 — TOXIC_HOUR returns a SKIP gate.** Toxic hours are
    a hard-block category, not in the no-skip rule. ✓ OK.

22. **MIN_EDGE (line 1965-1972) skips on `friction_pct > friction_cap`.**
    Friction is `spread * 2.5 / sl_dist`. Spread is in the no-skip category.
    **This is a skip on spread cost. VIOLATION** of the user's stated rule.
    Should warn-only. The A+ tier (quality≥75%) bypass exists, but normal
    setups skip purely on spread metrics. Reconcile with user intent.

23. **EV_REJECT (line 1990-1998) skips on `ev_after_cost < threshold`.** Cost
    component is friction (spread). Combined risk-adjusted statistical edge.
    Borderline — friction is one input but EV also includes WR / RR. Not a
    pure "spread skip" but the cost subtraction does include spread. Consider
    A+/quality-only bypass without the friction subtraction.

24. **EARLY_COOLDOWN (line 1260-1263) at the top of `_process_symbol`.**
    Cooldown after a recent loss/win is a re-entry timing constraint, not a
    risk/loss/spread category. ✓ OK per rule.

25. **MasterBrain win_cooldown (master_brain.py:134-139)** was upgraded from
    warn-only to HARD BLOCK on 2026-05-14 per the explicit user rationale
    that the no-skip rule was for risk/loss/spread — post-close win cooldown
    is a legitimate skip. ✓ Documented exception.

26. **MasterBrain `_check_net_directional` (master_brain.py:174-176)** — warn
    only when 3+ same-direction positions exist. ✓ OK per rule.

27. **MasterBrain `get_correlated_exposure` at line 167-169** — warn only
    when correlated symbol open. ✓ OK.

28. **MasterBrain `_corr_cooldown` (master_brain.py:160-165)** — HARD BLOCK
    on correlation cooldown. Same as win_cooldown — re-entry timing, not the
    spread/risk/loss category. ✓ Documented exception.

29. **`HARD_DOLLAR_CAP` (`config.py:262`) closes a position when unrealized
    loss exceeds 2% equity.** This is daily-loss / risk territory. Per rule,
    should warn-only for entries — but this fires on EXISTING positions
    (executor side), not entries. ✓ OK (exit, not entry).

30. **MasterBrain daily-trade count gate (master_brain.py:199-201)** — warn
    only when daily_trades ≥ _MAX_DAILY_TRADES. ✓ OK per rule.

31. **Brain has no spread check at entry-time.** Good — would otherwise
    be a no-skip violation. Spread shows up only inside MIN_EDGE friction
    formula (covered in finding #22).

## DEAD CODE

32. **brain.py:56 — `from signals.industry_gates import compute_gate_indicators,
    evaluate_entry_gates`** — neither symbol used anywhere in brain.py.
    Confirmed via grep: only line 56 references them.

33. **brain.py:44 — `SMART_ENTRY_MODE` imported from config but never used.**
    Confirmed: only line 44 in brain.py.

34. **brain.py:184 — `self._smart_entry = smart_entry`** stored but never
    referenced anywhere in `_process_symbol` or helpers. The whole
    smart-entry module appears unused by brain.

35. **brain.py:190-192 — `_pattern_learner`, `_order_flow`, `_level_memory`**
    all stored as instance attrs but never read in the entry pipeline. Only
    `_fvg` and `_trade_intel` are actually used.

36. **brain.py:2286-2335 — `_check_m15_reversal_exit`** starts with
    `return  # DISABLED — 0% WR, -$946/week across all symbols` and then has
    ~40 lines of unreachable code below it.

37. **brain.py:1741-1750 — Position HOLD/REVERSAL branches** unreachable
    because Gate 12 returned at 1410-1414 on `has_position`. See finding #4.

38. **brain.py:2070 `atr_at_entry`** — see finding #1; the field is never
    successfully written for any entry.

## Score-tier integration (question 3)

**Brain side does NOT need a score-tier branch — executor side is sufficient.**

Rationale:
- The brain already implements a quality-based tiered policy via the **A+ /
  quality≥65%/<65% tiers** (brain.py:1929-1933) which control MIN_EDGE and
  EV bypass.
- Brain.py:1824-1831 (`CONVICTION_SIZING_V2`) further scales risk_pct by
  signal quality. Score 6-7 maps to "<55" or "55-65" tier with 0.6x or 1.0x
  risk — already a marginal-tier sizing branch.
- `min_quality` from `SIGNAL_QUALITY_THRESHOLDS` (40-45% per regime) is
  already a per-regime floor; tightening it further for marginal trades
  would conflate the threshold (which decides IF we enter) with the trail
  policy (which decides HOW we manage). Conceptual separation is cleaner.
- The executor's `SCORE_TIER_THRESHOLD=7.0` aligns with `CONVICTION_SIZING_V2`'s
  "65-80" bucket (≈7.8-9.6 raw) — both kick in around the same point.

**Recommendation:** Keep brain entry-side as quality-tiered; keep score-tier
in executor for exit/trail/early-cut. The two systems address different
parts of the trade lifecycle and should not be cross-coupled.

## Cooldown ordering (question 5)

Cooldowns are evaluated in this order:
1. **External-close detection** (brain.py:1200-1249) — arms cooldown from
   broker close events. Side-effect only.
2. **Early BOTH-direction cooldown** (brain.py:1258-1263) — skips fast for
   `_cooldown_blocked == "BOTH"`. Cheap, runs before scoring. ✓ Good
   placement.
3. **JUST_CLOSED 180s** (brain.py:1271-1276) — skips immediately. ✓ Good.
4. **Directional cooldown** (brain.py:1522-1527) — checked AFTER scoring,
   regime, direction selection. Because directional cooldowns are
   direction-specific (post-win same-dir block), the check must happen
   after the direction is resolved at line 1356. ✓ Required ordering.
5. **MasterBrain win_cooldown** (master_brain.py:134-139) — checked inside
   `evaluate_entry`, AFTER all brain gates. Redundant if brain's directional
   cooldown already fired. **Possible double-checking** of same `_win_cooldown`
   logic: brain owns `_sl_cooldown` with `blocked_direction`; master_brain
   owns `_win_cooldown` separately. Two parallel data structures for related
   semantics.
6. **MasterBrain correlated cooldown** (master_brain.py:160-165) — same.
7. **News calendar** (brain.py:1561) — runs as part of Gate 20, AFTER scoring.
   With `CALENDAR_HARD_BLOCK_SYMBOLS` empty (config 930), this is currently
   a warning-only gate. **Performance cost** of the calendar lookup per
   cycle is paid even when no symbol is in the hard-block set.
8. **POST_BIG_WIN 5h cooldown** (brain.py:892-921) — armed inside trail/exit
   loop (`_run_cycle` C6), evaluated via the standard `_arm_cooldown` →
   `_cooldown_active` path. ✓ Single source of truth.

**Conclusion:** ordering is mostly correct (cheap BOTH-cooldown + JUST_CLOSED
fire first). The expensive checks (scoring, indicators, MTF) only run if the
cheap cooldowns pass. **One inefficiency:** brain.py:1258 only blocks when
`blocked == "BOTH"` — symbols under a directional cooldown (after wins)
fall through to score the bar, compute indicators, and only later get
rejected at Gate 17 (1523). For symbols with frequent post-win cooldowns
this is wasted work. Acceptable because the directional cooldown is the
common-case POST_WIN path.

**Two cooldown sources of truth** (brain.`_sl_cooldown` vs. MasterBrain.`_win_cooldown`+`_corr_cooldown`)
is the main architectural smell. Consolidate post-close cooldowns into the
brain's `_arm_cooldown`; let MasterBrain only check it.

## MasterBrain integration (question 6)

**Called once per entry** (brain.py:1798-1812). Inside `evaluate_entry`
(master_brain.py:79-328):

- **Score / meta-prob / MTF gates were explicitly REMOVED from MasterBrain
  in V5** (master_brain.py:147-149 comment). The current MasterBrain only
  does:
  - Circuit breaker (3 consecutive losses → 2h pause)
  - Win cooldown (30 min, hard block)
  - Symbol blacklist
  - Correlation cooldown (hard block)
  - Net directional exposure (warn only)
  - Portfolio risk gate (VaR / concentration)
  - Equity slope (modifies risk sizing)
  - Drift detector (modifies risk sizing)
  - Daily trade count (warn only)
- **No double-gating** on score/meta/MTF — the brain owns those.
- **One double-gating concern**: `is_symbol_blacklisted` (master_brain.py:141-145)
  fires on 4 consecutive losses, while the brain's `_sl_cooldown` tracks
  post-loss cooldown per trade. The brain doesn't have a multi-loss-streak
  blacklist of its own, so this is unique to MasterBrain. ✓ No overlap.

**Risk:** the MasterBrain blacklist permanently blocks a symbol for 24h
(`DRAGON_BLACKLIST_HOURS`) with no warn-only escape. This is implicitly a
risk-category skip, but it triggers only after 4 consecutive losses on a
single symbol. Document this as an explicit no-skip exception, or weaken
to warn-after-3 / block-after-5.

## Pullback flow (question 7)

After today's `PULLBACK_ATR_RETRACE 0.2 → 0.8` and `MAX_WAIT_BARS 1 → 5`:

- **Arm path** (brain.py:2015-2039): runs when `use_pullback=True` AND
  pullback not already pending. Stores signal_price, computes target at
  `signal - 0.8*ATR` (LONG) / `signal + 0.8*ATR` (SHORT). Returns
  PULLBACK_WAIT.
- **Fill path** (brain.py:1119-1149): on each subsequent cycle, checks
  current bid against `entry_target`. If hit, executes `open_trade`. ✓
  Fires correctly.
- **Fallback path** (brain.py:1156-1188): after `_pb_wait_eff` bars (5 H1
  bars = 5 hours) without fill, executes direct `open_trade` at current
  price. ✓ Fires correctly.
- **Edge case 1**: pullback armed in a "trending" regime, then regime flips
  to "ranging" during the wait. The fill check (1119-1149) doesn't re-check
  regime — entry will still fire at target. Probably intended (signal
  remains valid) but worth confirming.
- **Edge case 2**: signal direction at arm-time may not match current
  signal direction 4 bars later. The fill check trusts `pb["direction"]`.
  If market reversed, the fill executes against the new trend. Mitigation:
  the wait window is bounded by `_pb_wait_eff` so worst case is one stale
  entry per cycle.
- **Edge case 3**: `_pending_pullback` is in-memory only — see finding #3.
- **Edge case 4**: pullback ARM at 2015 has `if use_pullback and symbol
  not in self._pending_pullback` — but the same function ALSO has the
  fill check at 1114 with `if symbol in self._pending_pullback`. The
  function returns early on the fill branch, so a single cycle cannot both
  fill and re-arm — but two distinct entries on consecutive cycles can
  arm again immediately after a previous fill if `BAR_REENTRY` doesn't
  block (which it should, gate 13). ✓ Bar dedup protects.

## VWAP filter interaction (question 8)

**Order:** VWAP (line 1683-1712) → FIB (1714-1735) → POSITION_HOLD (1738) →
M15 (1756).

- **VWAP and FIB are independent.** VWAP checks `close vs vwap ± atr*buf`;
  FIB checks retracement of last Williams Fractal swing. No shared state.
- **VWAP and RANGE_EXTREME (1660-1681) overlap** when regime=="ranging":
  in a tight range, the VWAP and the range midpoint are usually very
  close. VWAP_WRONG_SIDE may fire for symbols already near range high/low.
  Both rejections at the same time is fine (cumulative correctness) but
  measure block-rate to ensure they're not redundant in practice.
- **VWAP per-symbol override** at config.py:978-983 with DJ30.r=0.0
  (disabled). Confirmed: `_vw_buf_mult > 0` guard at brain.py:1693 short-
  circuits cleanly when 0. ✓ Disable path works.
- **NaN guard `vw != vw`** at line 1696 is correct Python idiom for NaN
  detection without numpy import in that block.
- **VWAP indicator field assumed present in `ind["vwap"]`** — line 1691
  uses `ind.get("vwap")`. If `_compute_indicators` doesn't add vwap (older
  config), the gate silently no-ops. Should log once when vwap absent.

## RECOMMENDATIONS

1. **FIX `atr` undefined bug at brain.py:2070** — change to `atr_val`. This
   is silently breaking the V5 ENTRY log, alerter, entry_metadata, and
   `_last_entry_bar` for every direct entry. This finding alone justifies
   the audit.

2. **Persist `_pending_pullback` to SQLite** alongside `entry_metadata` and
   `kill_switch`. With `MAX_WAIT_BARS=5` (5 hours), restarts during the wait
   silently drop pending signals.

3. **Move CALENDAR_HARD_BLOCK_SYMBOLS membership check BEFORE the
   `should_skip_entry` call** (brain.py:1561-1574). With the set empty,
   we're paying for a calendar query every cycle for no behavior change.

4. **Remove redundant `POSITION_HOLD/REVERSAL` branch (brain.py:1738-1750)**
   and the second `has_position()` call. Gate 12 (1410) already returned
   if a position exists.

5. **Reconcile MIN_EDGE/EV with no-skip rule** (finding #22-23). Either
   make them warn-only (matching user rule literal interpretation) or
   document an explicit exception. Currently MIN_EDGE skips on a metric
   composed of spread — borderline violation.

6. **Remove dead imports and unused fields** (findings #32-35): `industry_gates`
   imports, `SMART_ENTRY_MODE`, `_smart_entry`, `_pattern_learner`,
   `_order_flow`, `_level_memory`. Trims ~15 lines + clarifies that
   smart_entry module is not actively wired into the entry pipeline.

7. **Cache `has_position(symbol)` once per `_process_symbol`** (finding #8).
   Eliminates 2-3 rpyc round-trips per symbol per cycle. With ~30 symbols
   at 2 Hz, that's 100+ saved rpyc calls/sec — meaningful for the latency
   budget on the rpyc bridge under load.

8. **Audit MasterBrain `_daily_trades++` placement** (finding #20). Move
   the increment to after the brain confirms execution success
   (`open_trade` returned True), not inside `evaluate_entry`.

9. **Reorder Gate 23 (MTF_CASCADE) below Gate 25-26 (VWAP, FIB)** — VWAP
   and FIB are cheap dict access + a small slice. MTF_CASCADE is the
   heaviest gate. Reject early with cheap gates.

10. **Make `_check_m15_reversal_exit` actually empty** — delete the
    unreachable code at brain.py:2294-2335 to reduce file size and
    confusion. Method is documented disabled since 2026-04-21.
