# 10 — Statistical Edge Map across (symbol, hour, day_of_week)

Date: 2026-05-22
Status: **DO NOT SHIP** — no variant passes the walk-forward ship rule.

---

## Concept

For every (symbol × hour × day_of_week) cell, compute historical WR / PF /
avg-R / sample-size from a 360-day backtest. Classify cells:

| Recommendation     | Rule                              |
| ------------------ | --------------------------------- |
| HIGH_CONFIDENCE    | n>=20 AND PF>=2.0 AND WR>=55%     |
| WHITELIST_OK       | n>=15 AND PF>=2.0                 |
| BLACKLIST_NEG_EV   | n>=15 AND PF<1.0                  |
| LOW_SAMPLE         | n<15                              |
| NEUTRAL            | everything else                   |

Two variants:

- **WHITELIST_HIGH_CONFIDENCE** — allow only WHITELIST_OK / HIGH_CONFIDENCE cells.
- **BLACKLIST_NEGATIVE_EV** — block only BLACKLIST_NEG_EV cells.

Ship rule: Δ ≥ $30 AND walk-forward (5-fold) avg PF > 1.5 AND ≥ 3/4 positive folds.

---

## Headline portfolio numbers (in-sample, 360d, 8 symbols)

| Variant            | Trades kept | Portfolio PnL  | Δ vs baseline |
| ------------------ | ----------: | -------------: | ------------: |
| baseline           | 3,645       | **$246,262**   | —             |
| WL (n>=15, PF>=2)  | 333 (9%)    | $ 75,056       | **-$171,206** |
| BL_n15_pf1.0       | 3,490       | $250,133       | +$3,871       |
| BL_n10_pf1.0       | 3,316       | $258,663       | **+$12,401**  |
| BL_n10_pf0.8       | 3,406       | $258,523       | +$12,262      |
| BL_n12_pf0.9       | 3,448       | $254,312       | +$8,050       |
| BL_n20_pf1.2       | 3,623       | $246,265       | +$3 (noise)   |

Portfolio walk-forward (per-fold deltas, 4 folds shown — fold 0 is the
initial train slice):

| Variant            | avg Δ      | avg PF | pos folds | Ship? |
| ------------------ | ---------: | -----: | --------: | :---: |
| WL_n15_pf2.0       | **-$59,219** | 12.87 | 0 / 4     | NO    |
| BL_n15_pf1.0       | +$294       | 3.43   | 1 / 4     | NO    |
| BL_n10_pf1.0       | -$904       | 3.46   | 1 / 4     | NO    |
| BL_n10_pf0.8       | -$774       | 3.45   | 0 / 4     | NO    |
| BL_n12_pf0.9       | +$95        | 3.44   | 1 / 4     | NO    |
| BL_n20_pf1.2       | +$0         | 3.40   | 0 / 4     | NO    |

**No variant ships.**

---

## Why walk-forward fails (root-cause)

In folds 1-3 the train slice (1/5..3/5 of 3,645 trades) is too thin for
any cell to reach n_min trades — per-symbol per-fold averages ~18 trades,
and a cell needs n>=15 *from a single symbol*. The result is that almost
every fold reports Δ = $0 because no cells are flagged BL. Only fold 4
(train = 2,916 trades) accumulates enough cell statistics, and that
single fold dominates the in-sample delta.

This isn't a tuning failure of the BL thresholds — it's a **data-density
ceiling**. The (sym, hour, dow) granularity is too fine for 360 days of
H1 H1 data on most symbols.

Whitelist is even worse: it discards 91% of trades and curve-fits hard
(fold 4 alone reports $-155K delta because the in-sample "high quality"
cells regress to baseline on the held-out slice).

---

## Cell map highlights (in-sample, 360d full set)

### DJ30.r — base $49,595

WHITELIST_OK cells:

| Cell       | n  | PF    | WR    | PnL    |
| ---------- | -- | ----: | ----: | -----: |
| Tue 07h    | 17 | 2.64  | 52.9% | $ 372  |
| Wed 08h    | 16 | 22.53 | 43.8% | $ 367  |
| Wed 15h    | 16 | 2.02  | 43.8% | $ 841  |
| Thu 20h    | 15 | 2.87  | 60.0% | $ 339  |
| Fri 15h    | 18 | 9.14  | 44.4% | **$8,238** |
| Fri 16h    | 16 | 10.39 | 43.8% | $3,806 |

BLACKLIST_NEG_EV cells (the 07h danger zone is consistent across DOW):

| Cell       | n  | PF   | WR    | PnL    |
| ---------- | -- | ---: | ----: | -----: |
| Mon 07h    | 17 | 0.51 | 35.3% | $-195  |
| Wed 07h    | 16 | 0.34 | 37.5% | $-240  |
| Thu 07h    | 16 | 0.25 | 12.5% | $-209  |

### US2000.r — base $180,163 (largest single-symbol contributor)

HIGH_CONFIDENCE cell:

| Cell       | n  | PF    | WR    | PnL     |
| ---------- | -- | ----: | ----: | ------: |
| Wed 08h    | 23 | 87.24 | 87.0% | $4,893  |

Strong WHITELIST_OK afternoons (Mon-Tue NY session):

| Cell       | n  | PF    | WR    | PnL      |
| ---------- | -- | ----: | ----: | -------: |
| Mon 14h    | 17 | 11.11 | 76.5% | $6,167   |
| Mon 15h    | 16 | 16.39 | 43.8% | $9,136   |
| Mon 16h    | 16 | 34.31 | 68.8% | **$15,871** |
| Mon 17h    | 15 |  9.75 | 66.7% | $8,884   |
| Mon 18h    | 16 |  5.63 | 75.0% | $4,364   |
| Tue 15h    | 16 |  7.85 | 37.5% | $4,119   |
| Wed 07h    | 19 |  4.62 | 84.2% | $2,054   |
| Mon 11h    | 15 |  3.00 | 73.3% | $   542  |
| Thu 19h    | 19 |  4.26 | 63.2% | $4,288   |

BLACKLIST_NEG_EV (clear losers — Wed afternoon + Thu/Fri 07h):

| Cell       | n  | PF   | WR    | PnL      |
| ---------- | -- | ---: | ----: | -------: |
| Wed 17h    | 17 | 0.29 | 41.2% | **$-1,477** |
| Wed 10h    | 18 | 0.66 | 38.9% | $-956    |
| Thu 07h    | 15 | 0.41 | 60.0% | $-456    |
| Fri 07h    | 18 | 0.54 | 77.8% | $-195    |

### AUDJPY — base $11,190

WHITELIST_OK:

| Cell       | n  | PF    | WR    | PnL    |
| ---------- | -- | ----: | ----: | -----: |
| Mon 07h    | 18 | 26.29 | 83.3% | $ 410  |
| Mon 18h    | 15 | 10.11 | 60.0% | $ 156  |
| Wed 08h    | 15 | 12.02 | 80.0% | $ 172  |

BLACKLIST_NEG_EV:

| Cell       | n  | PF   | WR    | PnL    |
| ---------- | -- | ---: | ----: | -----: |
| Wed 07h    | 22 | 0.98 | 59.1% | $   -3 |
| Fri 07h    | 16 | 0.10 | 37.5% | $-139  |

### SWI20.r — base $1,510

Only one cell crosses n>=15 (Mon 18h, n=15, PF=999 — every trade a win).
Data is too sparse for granular blacklisting.

### XAUUSD, EURUSD, UKOUSD, JPN225ft

No cells passed n>=15 thresholds — sample density too low. **These four
symbols cannot use (sym, hour, dow) edge maps at 360d.**

---

## Cross-symbol pattern: the 07h problem

DJ30, AUDJPY and US2000 all flag **hour 7 (UTC)** as a recurring BL
cell on at least one weekday:

- DJ30: Mon 07h, Wed 07h, Thu 07h
- AUDJPY: Wed 07h, Fri 07h
- US2000: Thu 07h, Fri 07h

This is the London open (08:00 BST) / NY pre-open quiet hour — high
spread, low directionality. Already partially handled by
`SESSION` filters in v5_backtest.py, but the BL data suggests at
least an additional filter could help for these three symbols.

This is the only finding I'd extract as a real edge from the map — it
generalises across symbols (3 of 8) AND has an economic rationale
(spread-during-news window). The other cells are likely curve-fit on
the in-sample sample.

---

## Why this doesn't ship — alignment with prior feedback

- **`feedback_selectivity_edge`** (2026-04-10): relaxing selectivity
  hurts PF. This research confirms: a WHITELIST that cuts 91% of trades
  destroys $171K of PnL; the "selectivity is the edge" insight applies to
  signal-level filters, not cell-level filters with too few samples.
- **`feedback_dont_overfit_backtest_when_live_bleeding`** and
  **`feedback_swi20_top_filter_rejected`** (2026-05-21): per-cell filters
  driven by 360d backtest anecdotes that fail walk-forward must be
  rejected. Same principle here.
- **`feedback_test_before_change`**: ran baseline first, ran 5 variants,
  WF every one — confirms NO variant ships under the strict rule.

---

## Output files

- `10_stat_edge.json` — full payload: per-symbol cell maps, variants,
  portfolio WF detail, recommendation.
- `10_stat_edge.md` — this document.
- `10_stat_edge_variants.json` — per-symbol variant sweep (intermediate).
- `10_stat_edge_portfolio_wf.json` — portfolio-level WF folds (intermediate).
- `_run_stat_edge.py` — base run (variant WL + BL).
- `_run_stat_edge_v2.py` — per-symbol variant sweep.
- `_run_stat_edge_v3.py` — portfolio WF sweep.
- `_inspect_cells.py` — pretty-print HC/WL/BL cells.
- `_finalize_json.py` — merge step.

---

## Recommended next step (NOT in scope for this 3h study)

The only finding with WF-survivable economic signal is the **07h
(London pre-open) blacklist for DJ30 / AUDJPY / US2000**. To test it
properly:

1. Hard-code the three (sym, 07h) blocks into `auto_tuned.SESSION` or a
   new `TIME_BLACKLIST` dict.
2. Backtest 360d with that single static block — no fitted thresholds.
3. Walk-forward 5-fold; if positive in ≥3 folds, ship.

Anything more granular (full (hour, dow) per-symbol map) will not
survive walk-forward at this data density.
