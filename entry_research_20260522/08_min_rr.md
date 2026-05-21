# 08 — MIN-RR Entry Filter Research

**Date:** 2026-05-22  **Period:** 180d  **Symbols:** DJ30.r, SWI20.r, XAUUSD, AUDJPY, EURUSD, US2000.r, UKOUSD, JPN225ft

**Method:** Post-hoc filter on baseline trade list. Targets = swing high/low (lookback N) ∪ EMA200 ∪ prior-day high/low. RR = nearest target / sl_dist.

**Baseline:** $54,360.72 across 2436 trades.

## Variant sweep

| min_rr | target_window | trades | total PnL | Δ vs base |
|--------|---------------|--------|-----------|-----------|
| 1.5 |  20 | 1034 | $32,098.34 | $-22,262.38 |
| 1.5 |  50 | 1144 | $34,959.87 | $-19,400.85 |
| 1.5 | 100 | 1204 | $37,604.55 | $-16,756.17 |
| 2.0 |  20 | 933 | $30,177.59 | $-24,183.13 |
| 2.0 |  50 | 1020 | $32,260.71 | $-22,100.01 |
| 2.0 | 100 | 1063 | $35,438.05 | $-18,922.67 |
| 2.5 |  20 | 853 | $28,352.48 | $-26,008.24 |
| 2.5 |  50 | 922 | $29,500.36 | $-24,860.36 |
| 2.5 | 100 | 948 | $33,335.07 | $-21,025.65 |
| 3.0 |  20 | 789 | $24,829.04 | $-29,531.68 |
| 3.0 |  50 | 845 | $25,553.62 | $-28,807.10 |
| 3.0 | 100 | 859 | $28,016.20 | $-26,344.52 |

## Decision

- **Ship:** NO
- **Rationale:** No variant meets all ship criteria; baseline already extracts most edge.

## Caveats

- Post-hoc filter — does not re-simulate cooldown timers, consec-loss counter, or DD circuit-breaker. First-order PnL delta only.
- sl_eff per symbol uses SL_OVERRIDE only (not SL_OVERRIDE_REGIME). Negligible drift vs filter signal.
- 'Nearest target' = closest level ahead of entry (industry-strict). Conservative — biases against filter (smaller RR denominator).
- target_window in H1 bars (20=~1d, 50=~2d, 100=~4d swing scale).