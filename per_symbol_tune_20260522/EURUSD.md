# EURUSD per-symbol tune (per_symbol_tune_20260522)

- Started: 2026-05-21T19:13:09.376589Z
- Elapsed: 24s
- Phase A iters: 80
- Phase B iters: 20
- Phase C iters: 15

## Baseline (live config, 180d)
- trades=338  pf=3.74  pnl=$+4255  wr=72.5%  dd=4.4%  avg_r=0.71

## Top 10 Phase A+B by PnL (180d)

| Δ | pnl | pf | n | wr | dd | SL | trail | pb_atr | pb_wait | vwap | mQ | toxic |
|---|-----|----|---|----|----|----|-------|--------|---------|------|----|-------|
| $+65060 | $+69315 | 2.88 | 359 | 49.6% | 9.0% | 0.3 | WIDE_RUNNER | 0.6 | 3 | 1.0 | 40 | [] |
| $+55626 | $+59881 | 2.92 | 326 | 48.8% | 8.1% | 0.3 | WIDE_RUNNER | 0.6 | 3 | 0.7 | 43 | [] |
| $+55222 | $+59476 | 2.92 | 327 | 48.6% | 8.1% | 0.3 | WIDE_RUNNER | 0.6 | 3 | 1.0 | 43 | [] |
| $+53446 | $+57701 | 2.62 | 451 | 50.6% | 7.7% | 0.3 | RUNNER_NO_BE | 0.6 | 3 | 0.5 | 33 | [5, 20] |
| $+48828 | $+53083 | 3.86 | 301 | 52.5% | 8.5% | 0.3 | RANGE_TIGHT | 0.6 | 3 | 1.0 | 43 | [] |
| $+33190 | $+37444 | 3.20 | 445 | 54.4% | 6.9% | 0.4 | TREND_LOOSE | 0.6 | 5 | 0.5 | 33 | [6, 20] |
| $+31685 | $+35940 | 2.56 | 447 | 51.9% | 7.5% | 0.4 | TREND_LOOSE | 0.5 | 5 | 0.5 | 30 | [6, 20] |
| $+30972 | $+35227 | 3.00 | 459 | 59.0% | 6.3% | 0.4 | RUNNER_NO_BE | 0.8 | 6 | 0.7 | 35 | [] |
| $+30952 | $+35207 | 2.77 | 425 | 52.9% | 7.2% | 0.4 | TREND_LOOSE | 0.5 | 5 | 0.3 | 33 | [6, 20] |
| $+29868 | $+34123 | 2.62 | 433 | 52.9% | 7.8% | 0.4 | TREND_LOOSE | 0.5 | 5 | 0.5 | 33 | [6, 20] |

## Phase C — Walk-forward (top-3)

| Δ | 180d-pf | WF-pf | WF-pos | PASS | SL | trail | pb | vwap | mQ | toxic |
|---|---------|-------|--------|------|----|-------|----|------|----|-------|
| $+65060 | 2.88 | 2.13 | 4/5 | YES | 0.3 | WIDE_RUNNER | 0.6/3 | 1.0 | 40 | [] |
| $+55626 | 2.92 | 2.72 | 5/5 | YES | 0.3 | WIDE_RUNNER | 0.6/3 | 0.7 | 43 | [] |
| $+55222 | 2.92 | 2.72 | 5/5 | YES | 0.3 | WIDE_RUNNER | 0.6/3 | 1.0 | 43 | [] |

## SHIP WINNER

- **Δ PnL**: $+65060
- **WF**: avg_pf=2.13, pos=4/5
- **180d**: pf=2.88, pnl=$+69315, n=359, wr=49.6%, dd=9.0%

```python
# Winning params for EURUSD
SL_atr_mult     = 0.3
Trail           = 'WIDE_RUNNER'
Pullback ATR    = 0.6
Pullback wait   = 3
VWAP buffer     = 1.0  (ATR×1.0)
min_quality     = 40
Toxic hours     = []
```
