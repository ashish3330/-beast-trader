# ETHUSD per-symbol tune — 2026-05-22T00:43:18

- Source repo: `/Users/ashish/Documents/beast-trader`
- Backtest: `backtest_symbol('ETHUSD', days=180, params=P)`  (READ-ONLY)
- Tune days: **180**  •  WF folds: **[60, 90, 120, 150, 180]**  •  RNG seed: 20260522
- Workers: 8  •  Elapsed: **52.9s**

## Dimensions explored
1. **SL ATR mult** ∈ [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5]
2. **Trail profile** ∈ ['_TIGHT_LOCK', '_WIDE_RUNNER', '_AGGR_LOCK', '_RUNNER_NO_BE', '_WIDE_RUNNER_BE07', '_COMMODITY_AGGRESSIVE', '_ETH_LIVE']
3. **Pullback ATR retrace** ∈ [0.4, 0.6, 0.8, 1.0, 1.2]
4. **Pullback wait bars** ∈ [3, 5, 7]
5. **VWAP buffer** ∈ ['0.0_disabled', '0.5_default']  *(see VWAP note in JSON)*
6. **min_quality** (uniform per regime) ∈ [28, 30, 33, 35, 38]
7. **direction_bias per regime** (trending/volatile) ∈ ['LONG', 'SHORT', 'BOTH']

## Baseline (live config)
- n=896  PF=7.48  PnL=$+8761.47  WR=84.4%  DD=1.5%

## Phase A — coarse random sweep (80 trials)
Top 10 (by PnL):

|  # |   PnL    |  PF  |  WR  | n  |  DD  | SL  | Trail | pb_atr/wait | VWAP | mq | dir(t/v) |
|----|----------|------|------|----|------|-----|-------|--------------|------|----|----------|
| 1 | $+323886.00 | 3.24 | 58.8% | 818 | 4.3% | 0.3 | _RUNNER_NO_BE | 0.6/7 | 0.0_disabled | 28 | SHORT/BOTH |
| 2 | $+298663.67 | 3.14 | 72.7% | 824 | 3.8% | 0.5 | _COMMODITY_AGGRESSIVE | 0.4/5 | 0.0_disabled | 38 | LONG/SHORT |
| 3 | $+158298.44 | 3.64 | 79.5% | 1183 | 2.0% | 0.7 | _COMMODITY_AGGRESSIVE | 0.6/5 | 0.5_default | 30 | LONG/BOTH |
| 4 | $+144660.35 | 3.46 | 54.4% | 759 | 6.8% | 0.3 | _AGGR_LOCK | 0.8/7 | 0.0_disabled | 28 | SHORT/BOTH |
| 5 | $+105572.35 | 3.58 | 71.2% | 1021 | 3.5% | 0.5 | _RUNNER_NO_BE | 0.6/5 | 0.5_default | 33 | SHORT/BOTH |
| 6 | $+94497.04 | 2.45 | 72.0% | 1098 | 3.3% | 0.5 | _COMMODITY_AGGRESSIVE | 0.6/3 | 0.5_default | 30 | BOTH/BOTH |
| 7 | $+88158.85 | 3.10 | 71.0% | 748 | 5.6% | 0.5 | _COMMODITY_AGGRESSIVE | 0.8/7 | 0.0_disabled | 38 | BOTH/SHORT |
| 8 | $+82065.81 | 3.29 | 69.7% | 1020 | 3.8% | 0.5 | _RUNNER_NO_BE | 0.4/5 | 0.0_disabled | 28 | LONG/BOTH |
| 9 | $+77645.61 | 2.53 | 49.5% | 772 | 8.6% | 0.3 | _AGGR_LOCK | 1.0/3 | 0.5_default | 28 | BOTH/BOTH |
| 10 | $+73467.38 | 3.50 | 55.6% | 639 | 5.3% | 0.3 | _WIDE_RUNNER_BE07 | 0.4/5 | 0.0_disabled | 38 | BOTH/SHORT |

## Phase B — neighborhood compose (24 trials)
Top 5 (by PnL):

|  # |   PnL    |  PF  |  WR  | n  |  DD  | SL  | Trail | pb_atr/wait | VWAP | mq | dir(t/v) |
|----|----------|------|------|----|------|-----|-------|--------------|------|----|----------|
| 1 | $+974807.10 | 2.36 | 60.2% | 689 | 7.8% | 0.3 | _COMMODITY_AGGRESSIVE | 0.4/5 | 0.0_disabled | 38 | BOTH/SHORT |
| 2 | $+370126.55 | 3.27 | 58.5% | 849 | 4.0% | 0.3 | _RUNNER_NO_BE | 0.6/5 | 0.0_disabled | 28 | SHORT/BOTH |
| 3 | $+323886.00 | 3.24 | 58.8% | 818 | 4.3% | 0.3 | _RUNNER_NO_BE | 0.6/7 | 0.0_disabled | 33 | LONG/BOTH |
| 4 | $+266091.55 | 3.29 | 58.2% | 846 | 3.6% | 0.3 | _RUNNER_NO_BE | 0.4/7 | 0.0_disabled | 28 | SHORT/BOTH |
| 5 | $+207551.35 | 3.09 | 56.1% | 779 | 4.6% | 0.3 | _AGGR_LOCK | 0.6/7 | 0.0_disabled | 28 | SHORT/BOTH |

## Phase C — walk-forward (top-5)

|  # |   PnL    |   Δ    |  PF  | n  | WF avg_pf | WF pos | delta_ok | wf_ok | WINNER |
|----|----------|--------|------|----|-----------|--------|----------|-------|--------|
| 1 | $+974807.10 | $+966045.63 | 2.36 | 689 | 1.77 | 3/5 | True | True | YES |
| 2 | $+370126.55 | $+361365.08 | 3.27 | 849 | 3.09 | 5/5 | True | True | YES |
| 3 | $+323886.00 | $+315124.53 | 3.24 | 818 | 3.07 | 5/5 | True | True | YES |
| 4 | $+323886.00 | $+315124.53 | 3.24 | 818 | 3.07 | 5/5 | True | True | YES |
| 5 | $+298663.67 | $+289902.20 | 3.14 | 824 | 2.32 | 5/5 | True | True | YES |

## Winner
- **SL ATR mult**: 0.3
- **Trail profile**: _COMMODITY_AGGRESSIVE  →  steps [(2.0, 'lock', 1.5), (1.0, 'lock', 0.7), (0.7, 'lock', 0.5), (0.5, 'lock', 0.35), (0.35, 'lock', 0.25), (0.25, 'lock', 0.17), (0.18, 'lock', 0.12), (0.12, 'lock', 0.07), (0.08, 'lock', 0.03), (0.05, 'be', 0.0)]
- **Pullback**: ATR=0.4  wait_bars=5
- **VWAP**: 0.0_disabled
- **min_quality** (uniform): 38
- **direction_bias_regime**: trending=BOTH, volatile=SHORT, ranging/low_vol=BOTH
- **PnL**: $+974807.10   Δvs baseline: $+966045.63
- **PF**: 2.36  WR: 60.2%  n: 689  DD: 7.8%
- **WF**: avg_pf=1.77  pos=3/5
- Folds: [{'days': 60, 'pnl': -2.59, 'pf': 0.99, 'n': 113}, {'days': 90, 'pnl': -57.85, 'pf': 0.78, 'n': 114}, {'days': 120, 'pnl': 187488.92, 'pf': 2.35, 'n': 461}, {'days': 150, 'pnl': 142664.94, 'pf': 2.35, 'n': 551}, {'days': 180, 'pnl': 974807.1, 'pf': 2.36, 'n': 689}]
