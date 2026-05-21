# MTF Confluence Research — 2026-05-22

**Scope:** 8 symbols (DJ30.r, SWI20.r, XAUUSD, AUDJPY, EURUSD, US2000.r, UKOUSD, JPN225ft) · 180-day in-sample · 5-fold walk-forward · 5 variants tested (cap met).
**Driver:** `entry_research_20260522/_run_mtf_confluence.py` (READ-ONLY, monkey-patches `signals.mtf_trend`).
**Live infra touched:** none — backtest params + module attribute swaps only.

## TL;DR — DO NOT SHIP. Keep current MTF cascade.

0 / 32 (symbol × variant) combinations met the ship gate (Δ ≥ $30 AND WF avg PF > 1.5 AND ≥3/5 folds positive).

| Variant | Decision | Why |
|---|---|---|
| BASELINE (`opposed >= 2 → REJECT`) | KEEP | Already PF 3-6 / 5/5 WF positive across most symbols. |
| REQUIRE_3_OF_3 | REJECT | Collapses trade flow to **0** on every symbol → catastrophic regression. |
| ADD_M15 (4-of-4) | REJECT | Same collapse as 3-of-3 (additive gate, can only be stricter). |
| REJECT_AGAINST_W1 | REJECT (no upside) | Identical to baseline on 7/8 symbols. DJ30.r loses $340 (−4%). No symbol gains ≥ $30. |
| WEEKLY_BIAS_ONLY | REJECT | All symbols → 0 trades except DJ30.r (47 trades, PF 6.76, $604 — but Δ = −$7,517 vs baseline $8,122). |

## Baseline reference (180 d, in-sample, audit_fix_gates ON, costs ON)

| Symbol | Trades | PF | WR | PnL | DD |
|---|---:|---:|---:|---:|---:|
| DJ30.r   | 380 | 3.98 | 55.8 % | $+8,122 | 7.3 % |
| SWI20.r  | 360 | 5.40 | 92.2 % | $  +736 | 2.6 % |
| XAUUSD   |  77 | 2.79 | 62.3 % | $  +450 | 3.0 % |
| AUDJPY   |  40 | 2.64 | 67.5 % | $  +212 | 3.8 % |
| EURUSD   | 154 | 2.64 | 68.2 % | $  +997 | 8.5 % |
| US2000.r | 378 | 4.04 | 57.9 % | $+21,979| 6.6 % |
| UKOUSD   | 292 | 3.40 | 59.2 % | $+5,716 | 7.0 % |
| JPN225ft |  58 | 6.20 | 75.9 % | $  +933 | 4.8 % |

## Walk-forward (BASELINE) — sanity reference

5×36-day non-overlapping folds anchored at most-recent candle. (XAUUSD & JPN225ft caches hold only ~29 d of H1 data, so 4 of their 5 folds have insufficient history.)

| Symbol | PF per fold | Positive folds |
|---|---|---:|
| DJ30.r   | 3.93 / 2.46 / 1.86 / 4.08 / 5.83 | 5/5 |
| SWI20.r  | 5.76 / 5.14 / 4.81 / 2.98 / 5.92 | 5/5 |
| US2000.r | 4.20 / 1.89 / 4.02 / 6.37 / 2.92 | 5/5 |
| UKOUSD   | 1.79 / 4.43 / 1.41 / 0.48 / 1.22 | 4/5 |
| AUDJPY   | 4.22 / 2.84 / 2.02 / 1.78 / 0.60 | 4/5 |
| EURUSD   | 0.29 / 2.67 / 3.72 / 2.92 / 0.45 | 3/5 |
| XAUUSD   | 2.79 / — / — / — / — | 1/5 (data) |
| JPN225ft | 6.20 / — / — / — / — | 1/5 (data) |

## Variant Δ vs baseline (in-sample 180 d)

| Symbol | REQUIRE_3_OF_3 | ADD_M15 | REJECT_AGAINST_W1 | WEEKLY_BIAS_ONLY |
|---|---:|---:|---:|---:|
| DJ30.r   | −$8,122 | −$8,122 |   −$341 | −$7,517 |
| SWI20.r  |   −$736 |   −$736 |    +$0 |   −$736 |
| XAUUSD   |   −$450 |   −$450 |    +$0 |   −$450 |
| AUDJPY   |   −$212 |   −$212 |    +$0 |   −$212 |
| EURUSD   |   −$997 |   −$997 |    +$0 |   −$997 |
| US2000.r | −$21,979| −$21,979|    +$0 | −$21,979|
| UKOUSD   | −$5,716 | −$5,716 |    +$0 | −$5,716 |
| JPN225ft |   −$933 |   −$933 |    +$0 |   −$933 |

## Why the strict variants collapse — important data note

`signals/mtf_trend.py::precompute_mtf_trends` aggregates higher TFs by stride-sampling the H1 stream (`h1_closes[::bars_per]`). For W1, `bars_per = 168`, so W1 EMA(20/50/200) needs ≈ 200 × 168 = **33,600 H1 bars** (~3.8 years) to be fully populated. Symbols in this study:

- **XAUUSD, JPN225ft:** 500 H1 bars (29 d) → W1 has 3 stride samples → EMA fails → **W1 is always FLAT** in the precomputed array. The baseline cascade therefore effectively runs on D1+H4 only for these two.
- **DJ30.r:** 5,000 H1 bars (309 d) → ~30 W1 samples → W1 partially populated.
- **SWI20.r:** 3,551 H1 bars (448 d) → ~21 W1 samples → W1 mostly FLAT.
- **AUDJPY, EURUSD, US2000.r, UKOUSD:** 33–50 k H1 bars → W1 properly populated.

For symbols whose W1 is mostly FLAT, the strict-alignment variants (REQUIRE_3_OF_3, ADD_M15, WEEKLY_BIAS_ONLY) treat FLAT as non-aligned and reject every signal. This is a **cache-extent artifact** for XAUUSD/JPN225ft, but the same behaviour would occur in live trading whenever a long enough W1 history is unavailable (e.g. a newly added symbol). Even when we ignore the limited symbols, the strict gates still wipe out trade flow on the data-rich symbols too, because true W1↔D1↔H4 SNIPER alignment is rare on H1-stride EMAs.

## REJECT_AGAINST_W1 — why it's a no-op

For 7 / 8 symbols the variant produces the *exact same* result as baseline. Reason: the baseline already rejects when `opposed ≥ 2`. The case "W1 opposes, D1 + H4 agree with entry" requires both D1 and H4 to be non-flat and aligned with entry, with W1 explicitly opposing. On the studied window this combination happens only on DJ30.r and only ~12 times (380 → 368 trades, $8,122 → $7,782 = −4.2 %). Conclusion: the additional W1 veto **costs $341 with zero meaningful safety upgrade** on this universe.

## Recommendation

**Keep `MTF_CASCADE_ENABLED = True` with the existing `opposed >= 2 → REJECT` logic.** No change to `signals/mtf_trend.py` is justified by this study.

If a future tightening attempt is wanted, prerequisites:

1. **Extend H1 caches to ≥ 3.8 years** for every backtested symbol so W1 EMA stack is populated end-to-end (current XAUUSD/JPN225ft caches make any W1-dependent test meaningless).
2. **Use true H4/D1 candle aggregation** rather than stride-sampling H1 closes (OHLC resample). Stride-sampling gives EMA values that are *not* equivalent to true higher-TF EMAs once we move past trend-direction parity into 3-EMA-stack alignment.
3. **Re-run with the parameter-loosened M15 proxy** (EMA-15 vs EMA-40 only, instead of 20/50/200) so the 4th gate isn't strictly equivalent to the H1 entry-TF score itself.

## Files written

- `entry_research_20260522/04_mtf_confluence.json` — full raw results + decisions
- `entry_research_20260522/04_mtf_confluence.md`   — this report
- `entry_research_20260522/_run_mtf_confluence.py` — re-runnable driver (READ-ONLY)
