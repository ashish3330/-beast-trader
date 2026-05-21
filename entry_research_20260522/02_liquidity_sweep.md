# SMC Liquidity Sweep / Stop Hunt — Research Findings

**Date:** 2026-05-22
**Time budget:** ~3h (used ~45m)
**Symbols:** DJ30.r, SWI20.r, XAUUSD, AUDJPY, EURUSD, US2000.r, UKOUSD, JPN225ft
**Window:** 180d nominal (data-limited: XAUUSD/JPN225ft only ~21d, SWI20.r ~55d, rest ~120-126d)
**Approach:** Read-only research. Implemented sweep detector + parallel backtest in
`run_liquidity_sweep.py`, reusing v5 backtest primitives (no source edits).

## TL;DR — NO WINNER

No variant met the ship criteria (ΔPnL ≥ $30 AND avg WF PF > 1.5 AND ≥3/5 positive folds)
on a **majority** of symbols. The "winner" by ranking (iter5) ships only **2/8** symbols
(US2000.r + UKOUSD), and regresses DJ30.r by **-$770**. Recommendation: **do not deploy**.

## Concept & Implementation

LONG sweep: bar's low pierced N-bar swing low (wick ≤ X·ATR), close recovered ≥ Y·ATR
above the prior low, close > EMA20. SHORT symmetric. Optional BOS gate (strict).

Two modes:
- **filter**: must have normal-pass signal AND sweep alignment in same direction
- **additional**: take sweep entry when normal signal sub-threshold (sweep "discovers"
  trades the score gate rejected)

## Baseline (no sweep) — sanity vs `backtest_symbol`

| Sym       | Trades | PF   | WR    | PnL       | DD   | WF avgPF | pos folds |
|-----------|-------:|-----:|------:|----------:|-----:|---------:|----------:|
| DJ30.r    | 538    | 4.23 | 53.0% | $17,326.82| 4.7% | 3.72     | 5/5       |
| SWI20.r   | 355    | 3.12 | 88.2% | $431.61   | 1.9% | 3.47     | 5/5       |
| XAUUSD    | 77     | 3.24 | 62.3% | $454.54   | 2.6% | 3.24     | 4/5       |
| AUDJPY    | 405    | 5.34 | 74.1% | $3,388.53 | 4.4% | 4.92     | 5/5       |
| EURUSD    | 318    | 3.04 | 60.4% | $1,496.60 | 6.3% | 3.01     | 5/5       |
| US2000.r  | 450    | 4.08 | 63.6% | $23,185.32| 6.2% | 4.03     | 5/5       |
| UKOUSD    | 375    | 3.51 | 64.0% | $8,032.09 | 7.0% | 2.59     | 5/5       |
| JPN225ft  | 69     | 6.97 | 76.8% | $1,387.49 | 5.4% | 6.42     | 2/5       |

Baseline already PF 3-7. Headroom for additive signals is thin.

## Iterations (5)

| # | Label                                       | Mode       | Cfg                       | Ships | ΣΔPnL    |
|---|---------------------------------------------|------------|---------------------------|------:|---------:|
| 1 | iter1_additional_N20_X0.4_Y0.5_loose        | additional | N=20 X=0.4 Y=0.5 BOS=off  | 2/8   | +$961.49 |
| 2 | iter2_filter_N20_X0.4_Y0.5_loose            | filter     | N=20 X=0.4 Y=0.5 BOS=off  | 0/8   | -$55,619.95 |
| 3 | iter3_additional_N20_X0.4_Y0.5_strict_bos   | additional | N=20 X=0.4 Y=0.5 BOS=on   | 0/8   | $0.00    |
| 4 | iter4_additional_N30_X0.6_Y0.5_loose        | additional | N=30 X=0.6 Y=0.5 BOS=off  | 0/8   | -$23.13  |
| 5 | iter5_additional_N10_X0.2_Y0.3_loose        | additional | N=10 X=0.2 Y=0.3 BOS=off  | 2/8   | +$1,215.86 |

### Why iter2 (filter) collapsed
Sweep + normal-aligned events are very rare: 0-14 trades per symbol on 120-180d.
The current entry-score gate already screens most aspects of "trend continuation",
so the intersection (score-pass AND sweep) ≈ {}. Filter mode loses 95-100% of
baseline PnL.

### Why iter3 (strict BOS) added zero
Requiring a BOS (prior structure break) before a sweep removed every candidate
event in the additional flow that wasn't already a normal-pass. BOS + sweep + reversal
is structurally rare on H1 within 120-180d. **Strict-BOS sweep is not a usable
extra signal on this universe.**

### iter5 — the best of a weak field
- US2000.r: +$632.49 (+8 trades), WF 5/5 positive, avg PF 4.03 — ships
- UKOUSD: +$1,421.14 (+14 trades), WF 5/5 positive, avg PF 2.75 — ships
- DJ30.r: **-$770.43** — regressed
- EURUSD: -$80.34 — regressed
- XAUUSD: -$25.32 — small regression

The net "+$1.2K" total is dominated by UKOUSD; DJ30's $-770 cuts it down.
Only US2000.r and UKOUSD show consistent gain. **No universal pattern**.

## Raw sweep event density (pre-gate)

For variant 1 (N=20, X=0.4, Y=0.5, BOS=off):
- DJ30.r 2910 bars → 19 events (11L/8S)
- AUDJPY 3025 bars → 9 events
- EURUSD 3025 bars → 11 events
- US2000.r 2909 bars → 18 events
- UKOUSD 2718 bars → 8 events
- SWI20.r 1316 bars → 0 events
- XAUUSD 500 bars → 3 events
- JPN225ft 500 bars → 0 events

True N-bar-sweep-with-close-recovery is a **~5-10/year-per-symbol** signal on H1.
That is not a steady stream — it's a tail. Per-fold trade counts are 0-2 — too few
for walk-forward statistical significance.

## Recommendation — DO NOT SHIP

1. **No universal winner across the 8-symbol universe.**
2. **Per-symbol "wins" (US2000.r, UKOUSD, DJ30.r) are noisy** — different variants
   pick different symbols, indicating overfit on small N.
3. **Filter mode is destructive** — sweep events don't intersect with the existing
   score gate meaningfully.
4. **Strict-BOS sweep is structurally too rare on H1** (0 added trades anywhere).

If sweep is to be revisited:
- Consider as a **per-symbol** override (e.g., enable on UKOUSD only, since both
  best variants ship it). But 2 data points (iter1 vs iter5) is insufficient
  evidence — wait for more sweep events or test on M15 where signal density may
  be higher.
- Try **counter-trend** sweep: current code requires `close > EMA20` for LONG
  sweep — i.e. sweep in trend direction. Pure "stop hunt reversal" SMC concept
  is the opposite: sweep ABOVE swing high → SHORT against current uptrend.
  This was not the spec-described version; future research could test it
  separately.

## Live-bot-bleed connection

User reported live bot -$37 over 30d. Sweep entries do not address this:
- Baseline backtest projects 30d ~$3K-$5K on this universe (extrapolating from
  180d). Live-vs-BT gap is from execution, not entry pattern. Sweep adds
  $0-$1.4K/180d on best symbol — well within backtest noise.
- The bleed is more likely from already-known issues: spread/slip on indices,
  position-management timing on UKOUSD, or the BTC #753-style late-exit pattern.

## Artifacts

- `/Users/ashish/Documents/beast-trader/entry_research_20260522/run_liquidity_sweep.py` — runnable detector + parallel backtest
- `/Users/ashish/Documents/beast-trader/entry_research_20260522/02_liquidity_sweep.json` — full numeric results
- `/Users/ashish/Documents/beast-trader/entry_research_20260522/02_liquidity_sweep.md` — this file
