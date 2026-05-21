# Dragon Trading Bot Code Audit — 2026-05-21

READ-ONLY audit of core trading engine. 26 findings.

## CRITICAL (Fix Immediately)

### 1. agent/learning_engine.py:616 — Bare `except: pass` swallows ML retrain failure
- After RL retrain, reload uses `except: pass` silently.
- Impact: broken meta-label model stays live → -3R to -5R per bad entry.
- Fix: log error and disable ML mode gracefully.

### 2. agent/brain.py:829 — Exception silences M15 check during cooldown gate
- `get_candles()` failure caught with `except Exception: pass` and `continue`.
- Impact: skips scoring for symbol; next entry fires with score=0 → automatic loss.
- Fix: log and return early with "M15_UNAVAILABLE" gate.

### 3. execution/executor.py:628 & 708 — Lot rounding can zero out volume
- `int(volume / vol_step)` rounds DOWN to 0 when volume < vol_step.
- Impact: zero-volume order rejected silently; entry never opens, no notification.
- Fix: clamp result to `vol_min` AFTER rounding.

### 4. agent/brain.py:1410-1416 — LATE_MOMENTUM gate too aggressive
- Blocks entries when score jumped from <6.0 to ≥7.0 (legitimate confirmation).
- Impact: 124 false blocks/day; +1.2R missed wins per false rejection.
- Fix: require jump >3R and recent low, not just "any prev < 6.0".

### 5. agent/brain.py:236 — Cooldown direction state lost on restart
- `_cooldown_blocked` dict defaults to "BOTH" on startup, losing directional saves.
- Impact: post-win cooldowns revert to both-direction → missed free-direction entries.
- Fix: persist and restore from SharedState.

## HIGH

### 6. execution/executor.py:882-883 — peak_r captured conditionally, may be zero
- Only captures peak_r if `_peak_profit_r` dict exists (initialized lazily).
- Impact: RL learns zero peak on early TP exits → skews exit rules.
- Fix: initialize `_peak_profit_r` in `__init__()` not lazily.

### 7. agent/brain.py:1749 — Hard-coded relative path breaks journaling
- External close uses `"data/trade_journal.db"` relative path.
- Impact: creates orphan DB when brain runs from different CWD; main DB never updated.
- Fix: use absolute path from `DB_PATH` config.

### 8. agent/rl_learner.py:174 — Bare except blocks swallow ALTER TABLE errors
- Only catches `sqlite3.OperationalError` but not other failures.
- Impact: schema migration silently fails; new columns return NULL; learning corrupts.
- Fix: log all exceptions, only swallow "already exists".

### 9. agent/brain.py:1178-1180 — Entry metadata dict can be corrupted without warning
- Reads score_components but doesn't log if missing/corrupted.
- Impact: RL trains on incomplete data → biased weights across symbol cohort.
- Fix: log when critical metadata missing.

### 10. execution/executor.py:1402 — Early-loss-cut tier logic fires on oscillation
- Loss streak increments every cycle even if position oscillating (not worsening).
- Impact: noisy positions close faster than smooth drawdowns; inconsistent exits.
- Fix: only increment when genuinely getting worse.

## MEDIUM

### 11. agent/brain.py:175-176 — Entry-rate guard dict not pre-initialized
- `_score_hist` built incrementally; first entry has empty history.
- Impact: LATE_MOMENTUM gate can't check prior scores on first entry → weak check.
- Fix: initialize dicts for all SYMBOLS at startup.

### 12. agent/learning_engine.py:271 — Trade dedup uses brittle PnL tolerance
- `abs(pnl - ?) < 0.02` tolerance may catch different trades with similar PnLs.
- Impact: occasional valid trades dropped; RL trains on incomplete sample.
- Fix: use stricter dedup: (symbol, direction, pnl, entry_price).

### 13. execution/executor.py:1356-1359 — Hard-cap equity fallback can be zero
- If account_info() fails or returns 0, equity becomes 0 and hard-cap disables.
- Impact: critical safety circuit disabled on connection glitch; cascade loss unchecked.
- Fix: close ALL positions if equity can't be determined.

### 14. agent/brain.py:500 & 504 — Date parsing exceptions swallow corruption
- Corrupted ISO date strings silently convert to None.
- Impact: kill-switch loses pre-restart loss history; re-baselines incorrectly.
- Fix: log corruption and use prior good value if available.

### 15. execution/executor.py:220-228 — Requote refresh doesn't check if price worsens
- Refreshed price on retry not checked against original.
- Impact: market order retries can slip; 90pt fill worse than original request.
- Fix: only refresh if new price same or better.

### 16. agent/brain.py:341 — Entry metadata cutoff 72h too short for slow trades
- Runners can last 60-90+ hours; metadata deleted before trade closes.
- Impact: long-duration winners lose entry context → RL doesn't learn winning setup.
- Fix: increase cutoff to 7 days.

## LIVE ↔ BACKTEST DIVERGENCE

### 23. Entry-rate deduplication — Live has (brain.py:1102), backtest doesn't
- Impact: live fires 2.5× fewer entries (204 vs 83 BTC trades/180d); PF mismatch.
- Fix: port dedup logic to backtest.

### 24. LATE_MOMENTUM gate — Live has, backtest doesn't
- Impact: 5-8% PnL difference on volatile symbols.
- Fix: add gate to backtest with same thresholds.

### 25. PEAK_GIVEBACK circuit — Live (executor.py:1315-1338), backtest doesn't
- Impact: live exits early on retracement; backtest shows higher PnL.
- Fix: implement in backtest exit logic.

### 26. EVAL_ON_CANDLE_CLOSE=False — Live rescores mid-candle, backtest scores on close only
- Impact: live makes 2-3× more entry attempts; trade count predictions off.
- Fix: align evaluation timing in both.

## TOP 5 TO FIX FIRST

1. **Lot size zero-out bug** (executor.py:628) — silently prevents entries
2. **RL retrain error handling** (learning_engine.py:616) — broken ML stays live
3. **Hard-coded path** (executor.py:1749) — orphan DB loses external closes
4. **LATE_MOMENTUM overblock** (brain.py:1410) — 124 false rejections/day
5. **Cooldown direction loss** (brain.py:236) — missed free-direction entries
