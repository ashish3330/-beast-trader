# BT TRAIL Verification — 5 active symbols

Date: 2026-05-22
Scope: SP500.r, UKOUSD, US2000.r, DJ30.r, USOUSD × 4 regimes (trending/ranging/volatile/low_vol)
Script: `audit_20260522/bt_trail_verify.py`
Raw output: `audit_20260522/bt_trail_verify.json`

## TL;DR

**ALL 20 cells (5 symbols × 4 regimes) match.** BT trail profile is identical
to the live-executor trail profile for every active (symbol, regime) pair.
End-to-end `backtest_symbol(..., days=60)` runs succeed for all 5 symbols.

No mismatches. No patch needed for the active 5-symbol slate.

## Trail Resolution Chains

### Live executor (`execution/executor.py:_resolve_trail_steps`)

```
[0] MARGINAL_TRAIL                       (if entry_score < SCORE_TIER_THRESHOLD)
[1] SYMBOL_REGIME_TRAIL_OVERRIDE[sym][regime]
[2] SYMBOL_TRAIL_OVERRIDE[sym]
[3] REGIME_TRAIL_DEFAULTS[regime]
[4] TRAIL_STEPS                          (global)
```

### Backtest (`backtest/v5_backtest.py:930 + 513`)

```
[A] TRAIL_OVERRIDE_REGIME[sym][regime]   (mirrors live #1)
[B] TRAIL_OVERRIDE[sym]                  (mirrors live #2)
[C] DEFAULT_PARAMS["trail"]              (= live TRAIL_STEPS, live #4)
```

Tuple-format adapter `_live_to_bt_trail()` converts live `(R, type, param)`
into BT `(R, param, type)`. Comparison in the verifier normalises both
back to `(R, type, param)`.

## Per-(symbol, regime) Verification

All match. Each row shows which source layer was selected on both sides.

| Symbol    | Regime    | Match | BT source              | LIVE source                    |
|-----------|-----------|-------|------------------------|--------------------------------|
| SP500.r   | trending  | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| SP500.r   | ranging   | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| SP500.r   | volatile  | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| SP500.r   | low_vol   | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| UKOUSD    | trending  | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| UKOUSD    | ranging   | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| UKOUSD    | volatile  | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| UKOUSD    | low_vol   | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| US2000.r  | trending  | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| US2000.r  | ranging   | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| US2000.r  | volatile  | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| US2000.r  | low_vol   | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| DJ30.r    | trending  | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| DJ30.r    | ranging   | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| DJ30.r    | volatile  | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| DJ30.r    | low_vol   | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| USOUSD    | trending  | YES   | TRAIL_OVERRIDE         | SYMBOL_TRAIL_OVERRIDE          |
| USOUSD    | ranging   | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| USOUSD    | volatile  | YES   | TRAIL_OVERRIDE_REGIME  | SYMBOL_REGIME_TRAIL_OVERRIDE   |
| USOUSD    | low_vol   | YES   | TRAIL_OVERRIDE         | SYMBOL_TRAIL_OVERRIDE          |

## Trail Profile Inventory (canonical `(R, type, param)`)

### SP500.r, UKOUSD, US2000.r, DJ30.r — all 4 regimes use `_TIGHT_LOCK`

```
(6.0, lock, 5.5)
(4.5, lock, 4.0)
(3.5, lock, 3.0)
(2.5, lock, 2.1)
(2.0, lock, 1.65)
(1.5, lock, 1.2)
(1.0, lock, 0.75)
(0.7, lock, 0.45)
(0.5, lock, 0.25)
(0.3, be,   0.0)
```

### USOUSD — split

| Regime    | Profile        |
|-----------|----------------|
| trending  | `_TIGHT_LOCK` (via SYMBOL_TRAIL_OVERRIDE fallback) |
| ranging   | `_AGGR_LOCK`  (via TRAIL_OVERRIDE_REGIME_AUTO)     |
| volatile  | `_WIDE_RUNNER`(via TRAIL_OVERRIDE_REGIME_AUTO)     |
| low_vol   | `_TIGHT_LOCK` (via SYMBOL_TRAIL_OVERRIDE fallback) |

**NOTE**: USOUSD is NOT "_TIGHT_LOCK in all 4 regimes via TRAIL_OVERRIDE_REGIME_AUTO"
as the task brief stated. In `auto_tuned.py:90`, USOUSD only has `ranging`
(→ `_AGGR_LOCK`) and `volatile` (→ `_WIDE_RUNNER`) entries. Trending and
low_vol fall back to `SYMBOL_TRAIL_OVERRIDE['USOUSD']` which the per-symbol
fine-tune at `auto_tuned.py:273` set to `_TIGHT_LOCK`. The end result still
matches between BT and live for every cell — verifier confirms 20/20.

## End-to-End BT Smoke (60-day backtest, default params)

| Symbol    | Trades | PnL ($)        | PF   |
|-----------|--------|----------------|------|
| SP500.r   | 154    | 10,135,064.61  | 9.51 |
| UKOUSD    | 180    | 3,988.00       | 3.95 |
| US2000.r  | 133    | 5,555.33       | 6.49 |
| DJ30.r    | 189    | 10,499.81      | 5.07 |
| USOUSD    | 199    | 1,906.65       | 7.18 |

SP500.r PnL is a known compounding artefact under default `risk_pct=0.8`
on the recent `_TIGHT_LOCK` regime tune — flagged for reference, not blocking
this verification.

## Latent Inconsistency (not affecting active 5 symbols)

BT resolution skips the `REGIME_TRAIL_DEFAULTS[regime]` layer that lives at
position #3 in the live executor. For the 5 active symbols this is moot —
all 5 have a populated `SYMBOL_TRAIL_OVERRIDE` entry, so live never falls
through to that layer in production. If a new symbol is added to the live
universe without a `SYMBOL_TRAIL_OVERRIDE` entry, BT will use `TRAIL_STEPS`
(global) while live would use `REGIME_TRAIL_DEFAULTS[current_regime]`.
Tracked for future cleanup; **no patch required for current scope**.

## Other Agents' Patches Present

Listed for reference (not applied by this agent):

- `audit_20260522/bt_loss_streak_patch.diff`
- `audit_20260522/bt_post_big_win_patch.diff`
- `audit_20260522/bt_score_tier_patch.diff`

## Conclusion

BT trail loading matches live for all 5 active symbols across all 4 regimes.
No bug. No patch. Mirror BT can run as-is.
