# Task 05 — Order Block (ICT/SMC) re-entry research

_Generated 2026-05-21T18:42:51.117148+00:00_  

Universe: XAUUSD, XAGUSD, NAS100.r, SP500.r, GER40.r, USDCAD, EURUSD, GBPJPY  
Window: 180 days, 5-fold walk-forward  
Iterations used: 5/5  

## Concept

Order Block = last opposing candle before an impulsive structure-breaking move. When price returns into the candle's [open, close] zone, enter in the impulse direction. Tested as FILTER (must overlap fresh OB) and ADDITIONAL (OB grants entries up to 15 quality-points below threshold).

## Baseline (no OB)

Full 180d: {"trades": 1845, "pnl": 10636.77, "pf": 3.71, "wr": 65.7, "worst_symbol_dd": 6.94, "ob_signal_extra": 0}  
WF 5-fold: {"trades": 1314, "pnl": 4560.02, "pf_avg": 3.55, "wr_avg": 66.8, "positive_folds": 5, "worst_symbol_dd": 6.89}

## Variant Summary (walk-forward aggregate)

| Variant | trades | PF avg | WR avg | PnL | +folds | worst DD | SHIP |
|---|---:|---:|---:|---:|---:|---:|:-:|
| FILTER_2.0x_lb20_fresh | 4 | 399.6 | 40.0 | $-6.81 | 2/5 | 0.5% | no |
| FILTER_1.5x_lb20_fresh | 8 | 399.6 | 40.0 | $-13.05 | 2/5 | 0.5% | no |
| FILTER_2.5x_lb30_fresh | 1 | 0.0 | 0.0 | $-5.01 | 0/5 | 0.5% | no |
| FILTER_2.0x_lb15_f3 | 13 | 10.64 | 22.0 | $-30.49 | 1/5 | 1.88% | no |
| ADDITIONAL_2.0x_lb20_fresh | 1314 | 3.55 | 66.8 | $4560.02 | 5/5 | 6.89% | no |

## Per-fold detail (selected variants)

### FILTER_2.0x_lb20_fresh
| fold | trades | PF | WR | PnL | worst DD |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | $0.0 | 0% |
| 2 | 1 | 0.0 | 0.0 | $-5.01 | 0.5% |
| 3 | 1 | 999 | 100.0 | $0.9 | 0% |
| 4 | 1 | 0.0 | 0.0 | $-4.93 | 0.49% |
| 5 | 1 | 999 | 100.0 | $2.23 | 0% |

### FILTER_1.5x_lb20_fresh
| fold | trades | PF | WR | PnL | worst DD |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.0 | 0.0 | $-4.99 | 0.5% |
| 2 | 2 | 0.0 | 0.0 | $-7.93 | 0.5% |
| 3 | 2 | 999 | 100.0 | $1.77 | 0% |
| 4 | 1 | 0.0 | 0.0 | $-4.93 | 0.49% |
| 5 | 2 | 999 | 100.0 | $3.03 | 0% |

### FILTER_2.5x_lb30_fresh
| fold | trades | PF | WR | PnL | worst DD |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | $0.0 | 0% |
| 2 | 1 | 0.0 | 0.0 | $-5.01 | 0.5% |
| 3 | 0 | 0 | 0 | $0.0 | 0% |
| 4 | 0 | 0 | 0 | $0.0 | 0% |
| 5 | 0 | 0 | 0 | $0.0 | 0% |

### FILTER_2.0x_lb15_f3
| fold | trades | PF | WR | PnL | worst DD |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.0 | 0.0 | $-18.86 | 1.88% |
| 2 | 1 | 0.0 | 0.0 | $-5.01 | 0.5% |
| 3 | 2 | 0.11 | 50.0 | $-7.28 | 0.82% |
| 4 | 1 | 0.0 | 0.0 | $-4.93 | 0.49% |
| 5 | 5 | 53.11 | 60.0 | $5.59 | 0.01% |

### ADDITIONAL_2.0x_lb20_fresh
| fold | trades | PF | WR | PnL | worst DD |
|---:|---:|---:|---:|---:|---:|
| 1 | 244 | 2.59 | 63.1 | $528.02 | 3.99% |
| 2 | 286 | 4.53 | 72.7 | $1074.55 | 2.78% |
| 3 | 269 | 2.73 | 67.3 | $670.62 | 4.75% |
| 4 | 226 | 2.44 | 65.5 | $461.18 | 5.63% |
| 5 | 289 | 5.48 | 65.4 | $1825.65 | 6.89% |

## Ship eligibility breakdown

- **FILTER_2.0x_lb20_fresh**: REJECT — pf>=1.30=PASS, pos_folds>=4=FAIL, beat_baseline_10pct=FAIL, worst_dd<=12pct=PASS, trades>=60=FAIL
- **FILTER_1.5x_lb20_fresh**: REJECT — pf>=1.30=PASS, pos_folds>=4=FAIL, beat_baseline_10pct=FAIL, worst_dd<=12pct=PASS, trades>=60=FAIL
- **FILTER_2.5x_lb30_fresh**: REJECT — pf>=1.30=FAIL, pos_folds>=4=FAIL, beat_baseline_10pct=FAIL, worst_dd<=12pct=PASS, trades>=60=FAIL
- **FILTER_2.0x_lb15_f3**: REJECT — pf>=1.30=PASS, pos_folds>=4=FAIL, beat_baseline_10pct=FAIL, worst_dd<=12pct=PASS, trades>=60=FAIL
- **ADDITIONAL_2.0x_lb20_fresh**: REJECT — pf>=1.30=PASS, pos_folds>=4=PASS, beat_baseline_10pct=FAIL, worst_dd<=12pct=PASS, trades>=60=PASS

## Honest verdict

- **No variant passes ship-eligibility.** Baseline WF PnL $4560.02; best OB variant: **ADDITIONAL_2.0x_lb20_fresh** at $4560.02.
- OB re-entry as implemented does not provide a net edge on the live 8-symbol set over a 180d window with the current scoring stack already filtering low-conviction entries.
- ICT/SMC concepts assume manual structure marking with discretion; mechanical auto-detection on H1 may be picking up too many low-quality zones, OR existing MTF/fib/audit-fix gates already capture most of the OB edge.