# UKOUSD Per-Symbol Hard-Tune — 2026-05-23
**Symbol:** UKOUSD | **Days:** 180 | **Equity:** $1219.0 | **Risk:** 2.0% | **Workers:** 6 | **Elapsed:** 111.0s

## Anchor (live config)
- SL=0.2  trail=_TIGHT_LOCK  PB=(0.8,5)  VWAP=0.5  mQ=25  PBW=10800s  LS=18000s  DIR=LONG

## Baseline (anchor under risk=2.0%, equity=$1219)
- trades: **103**  PF: **4.96**  WR: **51.5%**  PnL: **$+44,513**  DD: **8.60%**

## Phase A — Top SL × Trail (anchor knobs fixed)
| Rank | SL | Trail | Trades | PF | WR | PnL | Δ |
|---:|---:|:--|---:|---:|---:|---:|---:|
| 1 | 0.2 | _TIGHT_LOCK | 103 | 4.96 | 51.5% | $+44,513 | $+0 |
| 2 | 0.2 | _TREND_LOOSE | 105 | 4.59 | 50.5% | $+44,048 | $-465 |

## Phase B/C — Tuned Finalists

### Finalist #1
- SL: **0.2**
- Trail: **_TREND_LOOSE**
- Pullback: ATR=0.4, wait=3 bars
- VWAP buffer: 0.7 (0.7)
- min_quality (all regimes): **22**
- POST_BIG_WIN cooldown: **1800s**
- LOSS_STREAK cooldown: **3600s**
- Direction bias: **LONG**
- **In-sample 180d**: trades=139 PF=14.74 WR=58.3% PnL=$+754,943 DD=7.4% Δ=$+710,430
- **WF**: 5/5 positive, avg_pf=9.28, total=$+24,328

  | Fold | Trades | PF | WR | PnL |
  |---:|---:|---:|---:|---:|
  | 1 | 29 | 2.59 | 41.4% | $+575 |
  | 2 | 28 | 4.06 | 46.4% | $+1,180 |
  | 3 | 27 | 13.95 | 59.3% | $+3,839 |
  | 4 | 39 | 7.54 | 59.0% | $+13,731 |
  | 5 | 29 | 18.26 | 65.5% | $+5,003 |
- **Ship**: YES (Δ≥$30.0: True, WF pos≥3/5: True, avg PF≥1.5: True)

### Finalist #2
- SL: **0.2**
- Trail: **_TIGHT_LOCK**
- Pullback: ATR=0.4, wait=3 bars
- VWAP buffer: 0.7 (0.7)
- min_quality (all regimes): **22**
- POST_BIG_WIN cooldown: **1800s**
- LOSS_STREAK cooldown: **3600s**
- Direction bias: **LONG**
- **In-sample 180d**: trades=137 PF=13.69 WR=58.4% PnL=$+547,409 DD=7.4% Δ=$+502,896
- **WF**: 5/5 positive, avg_pf=8.95, total=$+20,713

  | Fold | Trades | PF | WR | PnL |
  |---:|---:|---:|---:|---:|
  | 1 | 28 | 2.54 | 39.3% | $+547 |
  | 2 | 28 | 3.91 | 46.4% | $+1,116 |
  | 3 | 27 | 14.06 | 59.3% | $+3,975 |
  | 4 | 37 | 7.38 | 59.5% | $+10,688 |
  | 5 | 29 | 16.87 | 65.5% | $+4,387 |
- **Ship**: YES (Δ≥$30.0: True, WF pos≥3/5: True, avg PF≥1.5: True)

## Verdict
- **SHIP**: Finalist #1
- Cfg: SL=0.2, trail=_TREND_LOOSE, PB=(0.4,3), VWAP=0.7, mQ=22, PBW=1800s, LS=3600s, DIR=LONG
- Δ=$+710,430  WF 5/5 pos  avg_pf=9.28

_Tune ran 111s (Phase-A 63 BTs, Phase-B 182 BTs)._
