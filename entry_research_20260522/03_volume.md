# VOLUME-CONFIRMED Entry Filter — Research Report
Date: 2026-05-22
Symbols: ETHUSD, GBPUSD, EURUSD, GBPJPY, EURJPY, USDCAD, GER40.r, SP500.r
  (8 syms with ≥180d H1 cache and valid tick_volume; XAUUSD/XAGUSD/BTCUSD/
   USDJPY/NAS100.r/JPN225ft caches only have 20-29 days and were excluded.)
Days: 180   WF folds: 5
Iterations used: 2 / 5
Runtime: 8.5s (iter-1) + 6.1s (iter-2) = 14.6s total

## Method

Volume signals are precomputed by the existing `_compute_indicators` in
`signals/momentum_scorer.py`:
  - `ind["vol"]`     — H1 tick_volume per bar
  - `ind["vol_sma"]` — 20-bar SMA of tick_volume
  - `ind["vwap"]`    — 20-bar rolling volume-weighted average price

The backtest is run UNMODIFIED. Each variant is implemented by monkey-
patching `_score_with_components` in a subprocess (READ-ONLY w.r.t. repo
source). When a volume condition fails:
  - filter-mode → return scores (0, 0) so the trade is rejected by the
    quality gate.
  - boost-mode  → multiply scores by `boost_score_mult` only when the
    condition is TRUE (lets marginal entries pass).

POC-HVN computes a crude rolling 24-bin volume-weighted price histogram
over the last 50 H1 bars and locates the densest bucket centre.

## Ship Criteria (per variant)
- Portfolio ΔPnL vs baseline ≥ +$30
- WF avg PF > 1.5
- ≥ 3 / 5 folds positive

## Baseline (no filter)

Portfolio: n=2827  PnL=$18,047.82  PF=3.24  WR=71.5%  DD=$379.51

| Symbol  | n   | PnL    | PF   | WR    |
|---------|-----|--------|------|-------|
| ETHUSD  | 861 | 3859.75 | 3.21 | 75.8% |
| GBPUSD  | 207 | 1337.73 | 2.24 | 58.5% |
| EURUSD  | 318 | 2080.47 | 2.50 | 65.4% |
| GBPJPY  | 427 | 1168.59 | 2.20 | 82.2% |
| EURJPY  | 241 | 1268.90 | 3.21 | 74.7% |
| USDCAD  | 222 | 2785.82 | 4.53 | 71.2% |
| GER40.r | 359 | 3282.52 | 3.61 | 72.4% |
| SP500.r | 192 | 2264.04 | 10.19 | 46.9% |

## Iteration 1 — Broad sweep (11 variants)

| Variant | Mode | n | PnL | Δ vs base | PF | WR | WF PF | +Folds | SHIP |
|---|---|---|---|---|---|---|---|---|---|
| tick_ratio_filter_1.0       | filter | 1974 | $9,797  | -$8,251  | 2.91 | 70.0% | 3.00 | 5/5 | NO |
| tick_ratio_filter_1.2       | filter | 1597 | $8,576  | -$9,472  | 3.12 | 68.1% | 3.15 | 5/5 | NO |
| tick_ratio_filter_1.5       | filter | 1069 | $4,751  | -$13,296 | 2.69 | 64.4% | 2.72 | 5/5 | NO |
| tick_ratio_filter_2.0       | filter |  486 | $2,225  | -$15,823 | 2.84 | 60.7% | 2.77 | 5/5 | NO |
| **vwap_side_filter**        | filter | 2714 | $22,634 | **+$4,586** | 3.75 | 72.5% | 3.92 | 5/5 | **YES** |
| poc_hvn_filter_0.5atr       | filter |  632 | $886    | -$17,162 | 1.83 | 57.1% | 1.80 | 4/5 | NO |
| poc_hvn_filter_1.0atr       | filter |  992 | $1,858  | -$16,190 | 2.12 | 61.1% | 2.11 | 5/5 | NO |
| poc_hvn_filter_1.5atr       | filter | 1332 | $3,863  | -$14,185 | 2.67 | 63.7% | 2.60 | 5/5 | NO |
| **tick_ratio_boost_1.5x_1.15** | boost | 2643 | $18,607 | **+$559** | 3.47 | 72.5% | 3.69 | 5/5 | **YES** (marginal) |
| **vwap_side_boost_1.15**    | boost  | 2817 | $24,525 | **+$6,478** | 3.46 | 73.2% | 3.85 | 5/5 | **YES** |

### Iter-1 takeaways
- **Tick-volume ratio as a filter HURTS** (all four thresholds 1.0/1.2/1.5/2.0
  regress by $8-16K). Tick-volume bursts on H1 forex/index data correlate
  with NOISE, not edge.
- **POC-HVN filter HURTS** at every threshold (mean reversion bias rather
  than breakout). Entries inside a volume node die against the dominant
  trend.
- **VWAP-side filter WORKS** (+$4,586). Filtering LONGs to price-above-VWAP
  and SHORTs to price-below-VWAP keeps the trend-aligned subset.
- **VWAP-boost @ 1.15× WORKS** but flips GBPUSD negative (-$1,421). Risky.

## Iteration 2 — Refinement of VWAP variant (9 variants)

| Variant | Δ vs base | PF | WF PF | Regr>$200 | Sign-flips | SHIP |
|---|---|---|---|---|---|---|
| vwap_filter_band_0.1atr     | +$4,460 | 3.72 | 3.92 | 1 | 0 | YES |
| vwap_filter_band_0.25atr    | +$4,588 | 3.69 | 3.87 | 1 | 0 | YES |
| **vwap_filter_band_0.5atr** | **+$5,821** | 3.75 | 3.92 | **0** | **0** | **YES** |
| vwap_boost_1.05             | +$13    | 3.33 | 3.60 | 2 | 1 | NO |
| vwap_boost_1.10             | +$5,084 | 3.45 | 3.77 | 1 | 1 | YES |
| vwap_boost_1.20             | +$8,595 | 3.59 | 3.86 | 2 | 1 | YES (but GBPUSD flips) |
| vwap_AND_tick1.2_filter     | -$9,728 | 3.13 | 3.16 | 7 | 0 | NO |
| vwap_AND_tick1.5_filter     | -$13,159 | 2.81 | 2.81 | 8 | 0 | NO |
| vwap_OR_tick1.5_filter      | +$4,951 | 3.73 | 3.90 | 1 | 0 | YES |

### Iter-2 takeaways
- Adding a 0.5×ATR buffer around the VWAP cleanly resolves the EURJPY
  regression while INCREASING portfolio gain ($4,586 → $5,821).
- Combining VWAP with tick-ratio (AND) destroys edge — tick-ratio's noise
  drags the composite into negative territory.
- VWAP_OR_tick is only slightly worse than pure VWAP — confirms tick-ratio
  has no incremental information once VWAP is in the gate.
- Boost variants are NOT robust: higher multipliers gain more PnL but
  flip GBPUSD to negative — a clear overfit / leverage-the-noise pattern.

## Per-Symbol Detail — RECOMMENDED variant `vwap_filter_band_0.5atr`

| Symbol  | Baseline | Variant | Δ Δ$    |
|---------|----------|---------|---------|
| ETHUSD  | 3859.75  | 3867.89 |   +8.14 |
| GBPUSD  | 1337.73  | 1233.84 | -103.89 |
| EURUSD  | 2080.47  | 2358.90 | +278.43 |
| GBPJPY  | 1168.59  | 1125.89 |  -42.70 |
| EURJPY  | 1268.90  | 1252.54 |  -16.36 |
| USDCAD  | 2785.82  | 5510.25 | +2724.43 |
| GER40.r | 3282.52  | 3998.84 | +716.32 |
| SP500.r | 2264.04  | 4520.62 | +2256.58 |

- All 8 symbols stay positive.
- Max regression: GBPUSD -$104 (within noise).
- Big wins: USDCAD (+$2,724), SP500.r (+$2,257), GER40.r (+$716).
- Portfolio DD: $379 → $299 (-21%)
- 5/5 WF folds positive, WF PF 3.92 (vs 3.24 PF baseline).

WF fold PnL: [$2,521, $3,965, $6,213, $8,000, $3,169]
WF fold PF : [4.06, 4.47, 4.47, 4.33, 2.25] — last fold weakest but still PF>2.

## VERDICT — SHIP `vwap_filter_band_0.5atr`

**Proposed gate (drop into `backtest_symbol` and live `brain.py` as opt-in):**

```python
# VWAP-SIDE entry filter with 0.5×ATR buffer (2026-05-22)
if p.get("vwap_side_filter_enabled"):
    vw = ind["vwap"][bi]
    if not np.isnan(vw):
        atr_buf = float(ind["at"][bi]) * 0.5
        if direction == 1 and float(c[bi]) <= (vw - atr_buf):
            continue   # LONG below VWAP - 0.5*ATR → skip
        if direction == -1 and float(c[bi]) >= (vw + atr_buf):
            continue   # SHORT above VWAP + 0.5*ATR → skip
```

Effect: filter rejects ~5-10% of marginal counter-VWAP entries.
Result over 180d/8sym: +$5,821 portfolio (+32% over baseline), with
no symbol regressing by more than $104, all 5 WF folds positive.

### Negative results (DO NOT ship)
- Tick-volume ratio thresholds (1.0/1.2/1.5/2.0) — regressed by $8-16K.
- POC/HVN distance filters — regressed by $14-17K (mean-reversion drag).
- Boost variants — gain PnL but flip GBPUSD to negative (single-symbol
  risk; rejected on robustness criterion).
- VWAP AND tick_ratio composite — strictness kills edge (-$10K to -$13K).

### Caveat
Test universe is 8 long-history symbols. The current LIVE set includes
XAUUSD/XAGUSD/BTCUSD/USDJPY/NAS100.r/JPN225ft, whose H1 caches only have
20-29 days and were excluded. Before live deploy, refresh those caches
and run `vwap_filter_band_0.5atr` on them to confirm no symbol-specific
breakage. Per-symbol delta should be ≥ -$50 on the full 14-symbol set.
