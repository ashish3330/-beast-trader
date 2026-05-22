# SP500.r hard-tune — 2026-05-23 02:26

## Baseline (live config, risk_pct=2.0, 180d)
- trades=238  PF=6.48  PnL=$+6,659  DD=5.70%  WR=71.0%

## Phase A — single-dim top-3 per dim
### dir_bias
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| dir_bias[volatile]=None | 422 | 4.90 | $+12,362 | $+5,703 | 7.10% |
| dir_bias[volatile]=BOTH | 422 | 4.90 | $+12,362 | $+5,703 | 7.10% |
| dir_bias[trending]=None | 273 | 6.79 | $+8,187 | $+1,528 | 4.70% |

### loss_streak
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| ls_cd=3600 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |
| ls_cd=7200 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |
| ls_cd=10800 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |

### min_q
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| min_q=22 | 244 | 7.16 | $+7,952 | $+1,293 | 5.70% |
| min_q=25 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |
| min_q=28 | 224 | 6.14 | $+6,249 | $-409 | 5.60% |

### pb_atr
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| pb_atr=0.6 | 255 | 6.09 | $+7,619 | $+960 | 5.60% |
| pb_atr=0.8 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |
| pb_atr=1.0 | 251 | 5.41 | $+5,025 | $-1,634 | 5.60% |

### pb_wait
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| pb_wait=3 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |
| pb_wait=5 | 94 | 1.96 | $+483 | $-6,175 | 8.40% |
| pb_wait=7 | 81 | 1.83 | $+391 | $-6,268 | 8.40% |

### post_big_win
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| pbw_cd=10800 | 223 | 7.37 | $+6,845 | $+186 | 5.60% |
| pbw_cd=1800 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |
| pbw_cd=3600 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |

### sl
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| sl=0.7 | 264 | 4.77 | $+15,159 | $+8,500 | 5.20% |
| sl=1.0 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |
| sl=1.5 | 244 | 4.45 | $+2,569 | $-4,090 | 7.90% |

### trail
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| trail=_TIGHT_LOCK | 237 | 5.70 | $+5,962 | $-697 | 5.70% |
| trail=_AGGR_LOCK | 237 | 5.70 | $+5,962 | $-697 | 5.70% |
| trail=_RUNNER_NO_BE | 237 | 5.70 | $+5,962 | $-697 | 5.70% |

### vwap
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| vwap=0.0 | 238 | 6.48 | $+6,659 | $+0 | 5.70% |
| vwap=1.0 | 224 | 4.47 | $+4,212 | $-2,447 | 6.10% |
| vwap=1.5 | 230 | 4.38 | $+4,147 | $-2,512 | 5.60% |

## Phase B — top-10 combos
| label | trades | PF | PnL | Δ | DD |
|---|---:|---:|---:|---:|---:|
| sl=0.7 + min_q=22 + pb_atr=0.6 + pbw_cd=10800 + dir_bias[trending]=None | 228 | 5.86 | $+9,531 | $+2,872 | 5.10% |
| sl=0.7 + min_q=22 + pb_atr=0.6 + pbw_cd=10800 + dir_bias[volatile]=None | 35 | 1.04 | $+6 | $-6,653 | 8.50% |
| sl=0.7 + min_q=22 + pb_atr=0.6 + pbw_cd=10800 + dir_bias[volatile]=BOTH | 35 | 1.04 | $+6 | $-6,653 | 8.50% |

## Phase C — 5-fold WF on top-3
### sl=0.7 + min_q=22 + pb_atr=0.6 + pbw_cd=10800 + dir_bias[trending]=None
- Full 180d: trades=228 PF=5.86 PnL=$+9,531 Δ=$+2,872
- WF pos folds: 5/5  avg_pf=3.14  total_wf_pnl=$+3,226
| fold | trades | PF | PnL | WR |
|---|---:|---:|---:|---:|
| 1 | 68 | 2.14 | $+377 | 66.2% |
| 2 | 70 | 3.48 | $+616 | 71.4% |
| 3 | 67 | 2.49 | $+491 | 70.1% |
| 4 | 57 | 2.89 | $+501 | 70.2% |
| 5 | 82 | 4.71 | $+1,240 | 81.7% |

### sl=0.7 + min_q=22 + pb_atr=0.6 + pbw_cd=10800 + dir_bias[volatile]=None
- Full 180d: trades=35 PF=1.04 PnL=$+6 Δ=$-6,653
- WF pos folds: 5/5  avg_pf=2.5  total_wf_pnl=$+3,065
| fold | trades | PF | PnL | WR |
|---|---:|---:|---:|---:|
| 1 | 39 | 1.38 | $+70 | 64.1% |
| 2 | 93 | 3.13 | $+961 | 71.0% |
| 3 | 60 | 1.92 | $+297 | 66.7% |
| 4 | 82 | 3.43 | $+1,257 | 72.0% |
| 5 | 57 | 2.63 | $+480 | 70.2% |

### sl=0.7 + min_q=22 + pb_atr=0.6 + pbw_cd=10800 + dir_bias[volatile]=BOTH
- Full 180d: trades=35 PF=1.04 PnL=$+6 Δ=$-6,653
- WF pos folds: 5/5  avg_pf=2.5  total_wf_pnl=$+3,065
| fold | trades | PF | PnL | WR |
|---|---:|---:|---:|---:|
| 1 | 39 | 1.38 | $+70 | 64.1% |
| 2 | 93 | 3.13 | $+961 | 71.0% |
| 3 | 60 | 1.92 | $+297 | 66.7% |
| 4 | 82 | 3.43 | $+1,257 | 72.0% |
| 5 | 57 | 2.63 | $+480 | 70.2% |

### sl=0.7
- Full 180d: trades=264 PF=4.77 PnL=$+15,159 Δ=$+8,500
- WF pos folds: 5/5  avg_pf=3.6  total_wf_pnl=$+3,526
| fold | trades | PF | PnL | WR |
|---|---:|---:|---:|---:|
| 1 | 64 | 1.96 | $+352 | 64.1% |
| 2 | 61 | 4.16 | $+671 | 73.8% |
| 3 | 55 | 2.81 | $+528 | 69.1% |
| 4 | 51 | 4.1 | $+554 | 70.6% |
| 5 | 67 | 4.98 | $+1,422 | 80.6% |

### dir_bias[volatile]=None
- Full 180d: trades=422 PF=4.90 PnL=$+12,362 Δ=$+5,703
- WF pos folds: 5/5  avg_pf=2.46  total_wf_pnl=$+2,836
| fold | trades | PF | PnL | WR |
|---|---:|---:|---:|---:|
| 1 | 98 | 1.63 | $+241 | 67.3% |
| 2 | 87 | 2.4 | $+554 | 65.5% |
| 3 | 96 | 2.62 | $+645 | 66.7% |
| 4 | 94 | 2.07 | $+401 | 63.8% |
| 5 | 102 | 3.6 | $+995 | 76.5% |

### dir_bias[volatile]=BOTH
- Full 180d: trades=422 PF=4.90 PnL=$+12,362 Δ=$+5,703
- WF pos folds: 5/5  avg_pf=2.46  total_wf_pnl=$+2,836
| fold | trades | PF | PnL | WR |
|---|---:|---:|---:|---:|
| 1 | 98 | 1.63 | $+241 | 67.3% |
| 2 | 87 | 2.4 | $+554 | 65.5% |
| 3 | 96 | 2.62 | $+645 | 66.7% |
| 4 | 94 | 2.07 | $+401 | 63.8% |
| 5 | 102 | 3.6 | $+995 | 76.5% |

### dir_bias[trending]=None
- Full 180d: trades=273 PF=6.79 PnL=$+8,187 Δ=$+1,528
- WF pos folds: 4/5  avg_pf=2.61  total_wf_pnl=$+2,066
| fold | trades | PF | PnL | WR |
|---|---:|---:|---:|---:|
| 1 | 60 | 1.89 | $+203 | 58.3% |
| 2 | 65 | 2.53 | $+405 | 63.1% |
| 3 | 76 | 3.59 | $+568 | 72.4% |
| 4 | 37 | 0.85 | $-28 | 51.4% |
| 5 | 91 | 4.18 | $+919 | 76.9% |

### dir_bias[trending]=LONG
- Full 180d: trades=273 PF=6.79 PnL=$+8,187 Δ=$+1,528
- WF pos folds: 4/5  avg_pf=2.61  total_wf_pnl=$+2,066
| fold | trades | PF | PnL | WR |
|---|---:|---:|---:|---:|
| 1 | 60 | 1.89 | $+203 | 58.3% |
| 2 | 65 | 2.53 | $+405 | 63.1% |
| 3 | 76 | 3.59 | $+568 | 72.4% |
| 4 | 37 | 0.85 | $-28 | 51.4% |
| 5 | 91 | 4.18 | $+919 | 76.9% |

## Ship decision
- **SHIP**
- gates: Δ≥$30.0, WF pos ≥3/5, avg PF ≥1.5
- winner: `sl=0.7`
- Δ=$+8,500  WF 5/5  avg_pf=3.6

### Winner overlay (apply to auto_tuned.py / config.py)
```json
{
  "sl": 0.7,
  "trail_name": null,
  "min_q": null,
  "pb_atr": null,
  "pb_wait": null,
  "vwap_buf": null,
  "post_big_win_secs": null,
  "loss_streak_secs": null,
  "dir_bias": null,
  "range_filter": "live",
  "fold_id": null
}
```

_elapsed: 33.8s  total BTs: 108_

---

## Post-tune verification — fine SL sweep + composability

**Important context:** between this script's design and execution, a parallel
agent applied a prior round of SP500.r hard-tune to `auto_tuned.py` /
`config.py` (commits not in git yet, file mtime 02:25). Those changes:

| key | prior live | applied by parallel agent | source |
|---|---|---|---|
| SL multiplier | 0.2 | **1.0** | auto_tuned.SL_OVERRIDE_AUTO |
| dir_bias_regime | `{volatile: LONG}` | `{volatile: LONG, ranging: SHORT}` | auto_tuned.DIRECTION_BIAS_REGIME_AUTO |
| pullback ATR | 0.6 | 0.8 | config.PULLBACK_ATR_RETRACE_PER_SYMBOL |
| pullback wait | 4 | 3 | config.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL |
| vwap buffer | 1.0 | 0.0 | config.VWAP_BUFFER_PER_SYMBOL |

The sweep above was re-run AGAINST this post-application state. So our
baseline (238 / PF 6.48 / +$6,659) is the live state AFTER that prior
agent's SHIP, and my Δ measures incremental improvement on top.

### Fine-grained SL sweep around the winner (180d)

| SL | trades | PF | PnL | DD | WR |
|---|---:|---:|---:|---:|---:|
| 0.40 | 41 | 1.38 | $+138 | 8.70% | 43.9% |
| 0.50 | 42 | 1.31 | $+105 | 8.50% | 47.6% |
| 0.60 | 43 | 1.06 | $+20 | 8.90% | 46.5% |
| **0.65** | 218 | 7.32 | $+12,422 | 5.90% | 69.3% |
| **0.70** | **264** | **4.77** | **$+15,159** | **5.20%** | **71.2%** |
| 0.75 | 228 | 7.92 | $+11,061 | 4.10% | 73.2% |
| 0.80 | 286 | 6.09 | $+13,093 | 3.50% | 74.1% |
| 0.90 | 282 | 4.92 | $+9,053 | 3.80% | 73.8% |
| 1.00 (baseline) | 238 | 6.48 | $+6,659 | 5.70% | 71.0% |

Note: between sl=0.60 and sl=0.65 there is a sharp discontinuity (43→218
trades) — at 0.60 ATR, friction/SL exceeds the 25% MIN_EDGE_REJECT
threshold for SP500.r's spread (2 points / 14-pt SL = 14% ✓ pass; but
2/8.5 = 23% close to threshold). Below 0.65 the entry gate fires
aggressively. Tune zone is sl ∈ [0.65, 0.90] — broad plateau.

### Fine-grained SL WF (5-fold disjoint 36d)

| SL | WF pos | avg PF | total PnL |
|---|---:|---:|---:|
| 0.60 | 5/5 | 2.96 | $+3,348 |
| **0.70** | **5/5** | **3.60** | **$+3,526** |
| 0.80 | 5/5 | 3.09 | $+2,764 |

sl=0.7 dominates on both avg_pf and total WF PnL.

### Composability — sl=0.7 + dir_bias[volatile]=None DOES NOT compose

| config | n | PF | PnL | DD |
|---|---:|---:|---:|---:|
| baseline (sl=1.0, current live) | 238 | 6.48 | $+6,659 | 5.70% |
| **sl=0.7 alone** | **264** | **4.77** | **$+15,159** | **5.20%** |
| dir_bias[volatile]=None alone | 384 | 4.13 | $+8,607 | 7.10% |
| sl=0.7 + dir_bias[volatile]=None | 33 | 0.98 | $-5 | 9.90% |
| sl=0.7 + dir_bias[volatile]=BOTH | 33 | 0.98 | $-5 | 9.90% |

Removing the volatile-LONG bias OR tightening SL each helps individually,
but combined the trade count collapses to 33 (from 264) and PnL goes to
zero. The two changes interact through the cooldown cascade — removing
the volatile bias allows volatile-SHORT entries, those tend to lose at
tight SL=0.7, arming cooldowns that block other entries.

**Ship rule: keep dir_bias_regime as-is (`{volatile: LONG, ranging: SHORT}`),
ONLY change SL from 1.0 → 0.7.**

### Final ship recommendation

Apply ONE change:

```python
# auto_tuned.py — SL_OVERRIDE_AUTO
'SP500.r': 0.7,  # was 1.0 — hard tune 2026-05-23 Δ+$8,500 WF 5/5 avg_pf=3.60
```

Verification:
- 180d full-window: +$15,159 vs $+6,659 baseline (Δ +$8,500, **+128%**)
- WF 5/5 folds positive — every 36d window profitable
- Avg PF 3.60 across folds (≥1.5 gate ✓)
- DD 5.20% (lower than 5.70% baseline)
- WR 71.2% (essentially unchanged from 71.0%)
- Broad plateau sl ∈ [0.65, 0.90] all positive — robust pick, not overfit

DO NOT bundle with dir_bias[volatile] changes — they nullify each other.