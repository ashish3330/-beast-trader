# DJ30.r per-symbol HARD-TUNE (2026-05-23)

- Days: 180
- Risk %: 2.0
- Start equity: $1219.0
- BT = mirror-aware (loads live `config.py` + `auto_tuned.py` overlays via reload)
- Workers: 6 (multiprocessing.Pool, maxtasksperchild=1 → per-BT process recycle, no state leak)
- READ-ONLY on source — all overlays injected in worker process memory only
- Elapsed: 105s (well under 2h cap)
- **Baseline (live config + risk_pct=2.0, equity=$1,219)**: trades=94 pf=2.9 wr=50.0% pnl=$6536.58 dd=8.7%

## Caveats / honest-read

1. **EMERGENCY DD interaction**: the BT triggers an 8% emergency-DD bail
   (v5_backtest.py:1226). At risk_pct=2.0 the baseline trips it after only 94
   trades — entire trailing 180d-window is truncated. Most "winning" configs
   simply *avoid* this trip by filtering harder (higher mq, wider vwap, tighter
   sl) so the run continues to completion (200+ trades). Some of the "+$222K"
   PnL is just continuing to trade where baseline halted.
2. **Compounding artefact**: risk_pct=2.0 with start_equity=$1,219 → after 200
   trades at PF ≥ 2.35 equity grows exponentially; per-trade dollar risk grows
   with it. Δ "$+114,580" is mathematically real but assumes broker lets us
   compound from $1.2K → $120K+ in 180d (no lot caps, no liquidity, etc.).
   Real-money expectation is **much** lower; treat as relative ranking signal.
3. **Trail-dim is largely inactive at low mq**: live executor (BT-mirrored)
   routes raw_score < 7.0 entries to a hard-coded `MARGINAL_TRAIL` regardless
   of any TRAIL_OVERRIDE patch. mq=25 → raw_score ≈ 3.0 → MARGINAL_TRAIL fires
   for ~all entries → flat trail dim in Phase A. Trail meaningfully diverges
   only when mq is high enough that raw_score ≥ 7 — which happens at mq=35+
   (raw_score ≈ 4.2). Phase B shows _TIGHT_LOCK and _AGGR_LOCK still
   identical even at mq=35 → MARGINAL_TRAIL still dominating most fills.
4. **WF fold DD = 8.7% on fold 1**: borderline emergency-DD trip even in the
   winner config. Real money should expect drawdowns close to the bail
   threshold.
5. **LS_CD cooldown was unobservable**: every value in `GRID_LS_CD` gave
   identical PnL → no LOSS_STREAK event fired in the baseline window, so the
   dim provides no signal in this dataset.

## Phase A — top-3 per dim
### sl
- `1.0` → pnl=$7381.68 pf=2.61 wr=67.9% n=361 dd=6.8% score=7382
- `0.2` → pnl=$7854.08 pf=3.06 wr=47.3% n=91 dd=8.9% score=7147
- `1.5` → pnl=$3527.73 pf=2.21 wr=68.8% n=401 dd=5.5% score=3528
### trail
- `_TIGHT_LOCK` → pnl=$5304.88 pf=2.85 wr=50.0% n=92 dd=8.7% score=4934
- `_AGGR_LOCK` → pnl=$5304.88 pf=2.85 wr=50.0% n=92 dd=8.7% score=4934
- `_RUNNER_NO_BE` → pnl=$5304.88 pf=2.85 wr=50.0% n=92 dd=8.7% score=4934
### mq
- `38` → pnl=$222480.77 pf=5.54 wr=49.5% n=206 dd=7.3% score=222481
- `35` → pnl=$101082.57 pf=3.08 wr=51.4% n=220 dd=8.3% score=98050
- `25` → pnl=$6536.58 pf=2.9 wr=50.0% n=94 dd=8.7% score=6079
### pb_atr
- `0.6` → pnl=$9823.75 pf=3.33 wr=56.0% n=91 dd=8.2% score=9627
- `0.8` → pnl=$6536.58 pf=2.9 wr=50.0% n=94 dd=8.7% score=6079
- `0.4` → pnl=$2525.55 pf=3.66 wr=57.1% n=35 dd=8.4% score=2425
### pb_wait
- `3` → pnl=$1658.81 pf=3.52 wr=39.3% n=28 dd=8.2% score=1626
- `7` → pnl=$1611.72 pf=3.93 wr=39.1% n=23 dd=9.3% score=1402
- `5` → pnl=$-108.90 pf=0.23 wr=12.5% n=8 dd=8.9% score=-1109
### vwap_buf
- `1.0` → pnl=$29527.62 pf=2.9 wr=50.8% n=132 dd=9.5% score=25098
- `0.7` → pnl=$22854.34 pf=2.94 wr=51.2% n=127 dd=9.5% score=19426
- `0.5` → pnl=$9727.26 pf=3.51 wr=53.3% n=75 dd=9.4% score=8365
### pbw_cd
- `5400` → pnl=$12303.60 pf=3.14 wr=53.2% n=94 dd=8.7% score=11442
- `7200` → pnl=$12303.60 pf=3.14 wr=53.2% n=94 dd=8.7% score=11442
- `10800` → pnl=$6536.58 pf=2.9 wr=50.0% n=94 dd=8.7% score=6079
### ls_cd
- `3600` → pnl=$6536.58 pf=2.9 wr=50.0% n=94 dd=8.7% score=6079
- `7200` → pnl=$6536.58 pf=2.9 wr=50.0% n=94 dd=8.7% score=6079
- `10800` → pnl=$6536.58 pf=2.9 wr=50.0% n=94 dd=8.7% score=6079
### dir_bias
- `db2` → pnl=$7343.04 pf=2.36 wr=48.7% n=119 dd=9.1% score=6535
- `db0` → pnl=$6536.58 pf=2.9 wr=50.0% n=94 dd=8.7% score=6079
- `db1` → pnl=$6536.58 pf=2.9 wr=50.0% n=94 dd=8.7% score=6079

## Phase A → Top-2 per dim (fed to Phase B)
```
{
  "sl": [
    1.0,
    0.2
  ],
  "trail": [
    "_TIGHT_LOCK",
    "_AGGR_LOCK"
  ],
  "mq": [
    38,
    35
  ],
  "pb_atr": [
    0.6,
    0.8
  ],
  "pb_wait": [
    3,
    7
  ],
  "vwap_buf": [
    1.0,
    0.7
  ],
  "pbw_cd": [
    5400,
    7200
  ],
  "ls_cd": [
    3600,
    7200
  ],
  "dir_bias": [
    "db2",
    "db0"
  ]
}
```

## Phase B — top-10 cartesian
1. sl=0.2 tr=_TIGHT_LOCK mq=35 pb=0.8/7 vw=1.0 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$121116.40 pf=2.35 n=203 dd=9.0% score=109005
2. sl=0.2 tr=_AGGR_LOCK mq=35 pb=0.8/7 vw=1.0 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$121116.40 pf=2.35 n=203 dd=9.0% score=109005
3. sl=0.2 tr=_TIGHT_LOCK mq=35 pb=0.8/3 vw=1.0 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$64217.96 pf=3.55 n=154 dd=8.9% score=58438
4. sl=0.2 tr=_AGGR_LOCK mq=35 pb=0.8/3 vw=1.0 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$64217.96 pf=3.55 n=154 dd=8.9% score=58438
5. sl=0.2 tr=_TIGHT_LOCK mq=35 pb=0.8/3 vw=0.7 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$37025.85 pf=2.99 n=149 dd=8.1% score=36656
6. sl=0.2 tr=_AGGR_LOCK mq=35 pb=0.8/3 vw=0.7 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$37025.85 pf=2.99 n=149 dd=8.1% score=36656
7. sl=0.2 tr=_TIGHT_LOCK mq=35 pb=0.6/3 vw=1.0 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$17509.78 pf=4.63 n=81 dd=8.6% score=16459
8. sl=0.2 tr=_AGGR_LOCK mq=35 pb=0.6/3 vw=1.0 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$17509.78 pf=4.63 n=81 dd=8.6% score=16459
9. sl=0.2 tr=_TIGHT_LOCK mq=35 pb=0.6/3 vw=0.7 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$13043.39 pf=4.35 n=79 dd=8.4% score=12522
10. sl=0.2 tr=_AGGR_LOCK mq=35 pb=0.6/3 vw=0.7 pbwcd=5400 lscd=3600 db={'ranging': 'BOTH'} → pnl=$13043.39 pf=4.35 n=79 dd=8.4% score=12522

## Phase C — 5-fold WF on top-5
- **SHIP** sl=0.2 tr=_TIGHT_LOCK mq=35 pb=0.8/7 vw=1.0 → in-sample=$121116.40, Δ=$+114579.82, WF 5/5 avg_pf=3.37
- **SHIP** sl=0.2 tr=_AGGR_LOCK mq=35 pb=0.8/7 vw=1.0 → in-sample=$121116.40, Δ=$+114579.82, WF 5/5 avg_pf=3.37
- **SHIP** sl=0.2 tr=_TIGHT_LOCK mq=35 pb=0.8/3 vw=1.0 → in-sample=$64217.96, Δ=$+57681.38, WF 5/5 avg_pf=3.05
- **SHIP** sl=0.2 tr=_AGGR_LOCK mq=35 pb=0.8/3 vw=1.0 → in-sample=$64217.96, Δ=$+57681.38, WF 5/5 avg_pf=3.05
- **SHIP** sl=0.2 tr=_TIGHT_LOCK mq=35 pb=0.8/3 vw=0.7 → in-sample=$37025.85, Δ=$+30489.27, WF 5/5 avg_pf=3.12

## Phase C — winner per-fold breakdown

Disjoint 36-day folds (each starts at default $1,000 equity → no cross-fold compounding):

| Fold | Trades | PF   | PnL    | DD   |
|------|--------|------|--------|------|
| 1    | 19     | 5.83 | $2263  | 8.7% |
| 2    | 15     | 2.29 | $252   | 8.3% |
| 3    | 36     | 2.04 | $611   | 8.7% |
| 4    | 53     | 2.79 | $1948  | 7.4% |
| 5    | 49     | 3.90 | $3423  | 6.4% |
| TOT  | 172    | 3.37 (avg PF) | **$8,496** | — |

All 5/5 positive. Avg PF 3.37 ≥ 1.5 threshold.

## Ship decision

Rule: Δ ≥ $30 AND ≥3/5 WF folds positive AND avg PF ≥ 1.5.

**SHIP — verdict: cautious GREEN**.

Winner:
- `SL_OVERRIDE['DJ30.r'] = 0.2`  (no change vs current live)
- `TRAIL_OVERRIDE['DJ30.r'] = _TIGHT_LOCK`  (no change vs current live)
- `SIGNAL_QUALITY_SYMBOL['DJ30.r'] = {all 4 regimes: 35}`  (current live=25 → tighten to 35)
- `PULLBACK_ATR_RETRACE_PER_SYMBOL['DJ30.r'] = 0.8`  (no change)
- `PULLBACK_MAX_WAIT_BARS_PER_SYMBOL['DJ30.r'] = 7`  (current=4 → 7)
- `VWAP_BUFFER_PER_SYMBOL['DJ30.r'] = 1.0`  (current=0.0 disabled → 1.0 enabled, ATR-buffered)
- `POST_BIG_WIN_COOLDOWN_SECS = 5400`  (current=10800 → 1.5h)  *global, affects other symbols*
- `LOSS_STREAK_COOLDOWN_SECS = 3600` *not observed-effective in this window*
- `DIRECTION_BIAS_REGIME_AUTO['DJ30.r'] = {'ranging': 'BOTH'}`  (current=`{'ranging':'LONG'}` → remove LONG bias)

Real meaningful changes (vs current live):
- **mq 25 → 35** (tighten signal-quality threshold)
- **vwap_buf 0.0 → 1.0** (re-enable VWAP-side filter with widest 1.0×ATR buffer)
- **pb_wait 4 → 7** (longer pullback window — 53% retrace rate per #01 research holds)
- **dir_bias ranging LONG → BOTH** (allow ranging SHORTs, previously blocked)
- **POST_BIG_WIN cooldown 3h → 1.5h** (warning: this is global, applies to all symbols)

Risk acknowledgements:
- WF fold DD reaches 8.7% (≈ EMERGENCY DD bail) — real money sees ≥8% DD in
  bad runs.
- In-sample $121K PnL is compounding fantasy at $1.2K start equity — do not
  expect 99× return in 180d live.
- POST_BIG_WIN_COOLDOWN_SECS is a global constant — changing 10800→5400
  affects all symbols, not just DJ30. Recommend a per-symbol override
  mechanism, or split the deploy.

Holdable single-dim drop-ins (lower risk, easier to validate):
1. **VWAP buf 0.0→1.0 only**: Phase A showed pnl=$29,528 vs baseline $6,537,
   PF 2.90, DD 9.5%. Strong improvement on one knob.
2. **pb_atr 0.8→0.6 only**: Phase A showed pnl=$9,824 vs $6,537, PF 3.33,
   DD 8.2% — smaller but stable.
3. **POST_BIG_WIN_COOLDOWN_SECS 10800→5400 only**: Phase A pnl=$12,304,
   PF 3.14, DD 8.7% (caveat: global var, not symbol-specific).

If shipping just the cleanest single change: **enable VWAP filter with
buffer=1.0** — that's the largest single-knob improvement that survives WF.