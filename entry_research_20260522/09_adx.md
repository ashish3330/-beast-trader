# 09 — ADX-based TREND STRENGTH entry filter

Date: 2026-05-22  · Repo: /Users/ashish/Documents/beast-trader · Window: 180d
Method: monkey-patch `backtest.v5_backtest._score_with_components` to zero the
raw long/short scores whenever the candidate bar's ADX (and/or DI cross) fails
the gate. Because v5's entry loop checks `signal_quality < threshold`
immediately after scoring, a zeroed score causes the trade to be skipped. No
source files were modified.

## Key finding (TL;DR)

**No ADX overlay ships at the portfolio level.** The existing `get_regime`
already encodes the industry-standard ADX-first thresholds (ADX<22 ⇒ ranging
⇒ higher min-quality bar, ADX>25/30 in trending bands), so any *extra*
ADX-floor filter merely subtracts capital from healthy bars without lifting
PF enough to compensate. Every variant tested fails the ship criterion
(Δ ≥ +$30) — they all *regress* the 8-symbol total PnL from -$0 (regime_adx_v1
= no-op) down to -$40.9K (global_adx_30).

The WF criterion is moot here because every variant passes (5/5 folds
positive, avg PF ≥ 3.4) — the strategy is too healthy/profitable across the
whole window for fold-by-fold to discriminate. The ship gate that matters is
Δ-PnL, and every overlay is net negative.

## Variants tested

| Variant                  | Total PnL  | Δ vs base   | Trades | WF avg PF | Folds+ | Ship |
|--------------------------|-----------:|------------:|-------:|----------:|-------:|:----:|
| BASELINE                 | $54,360.72 |        —    | 2,436  |    —      |   —    |  —   |
| global_adx_18            | $41,940.52 | -$12,420.20 | 2,283  |  4.09     | 5/5    |  no  |
| global_adx_22            | $28,996.05 | -$25,364.67 | 2,124  |  3.93     | 5/5    |  no  |
| global_adx_25            | $13,571.83 | -$40,788.89 | 1,429  |  3.85     | 5/5    |  no  |
| global_adx_30            | $13,481.54 | -$40,879.18 | 1,312  |  3.41     | 5/5    |  no  |
| regime_adx_v1*           | $54,360.72 |     +$0.00  | 2,436  |  4.10     | 5/5    |  no  |
| regime_adx_strict        | $37,643.98 | -$16,716.74 | 2,082  |  3.77     | 5/5    |  no  |
| adx_direction_only       | $48,038.95 |  -$6,321.77 | 2,179  |  4.44     | 5/5    |  no  |
| adx_direction_floor20    | $30,228.35 | -$24,132.37 | 1,992  |  4.41     | 5/5    |  no  |
| combo_floor22_dir        | $25,189.08 | -$29,171.64 | 1,913  |  4.38     | 5/5    |  no  |

\* `regime_adx_v1` (trending≥25, volatile≥22, low_vol≥18) is a no-op because
`get_regime(bbw, adx)` (v5_backtest.py:354-370) already enforces these exact
thresholds in its regime-classification logic. Confirms the patch hook works
correctly.

## Per-variant detail (180d, 8-symbol)

### global_adx_18 (the "best" of the loss-making variants)
- DJ30.r:    pnl $12,322 (+$1,218) - small win
- SWI20.r:   pnl $653    (-$85)    - WR holds at 91%
- XAUUSD:    pnl $352    (-$104)
- AUDJPY:    pnl $5,127  (+$292)   - small win
- EURUSD:    pnl $2,073  (-$7)     - flat
- US2000.r:  pnl $12,316 (-$13,831)- **single concentrated loss**
- UKOUSD:    pnl $7,075  (+$96)
- JPN225ft:  pnl $2,022  (+$0)     - flat (all 69 trades had ADX≥18)

The -$13.8K hit on US2000.r is the dominant signal — that symbol's edge sits
in low-ADX (15-18) breakouts where +DI/-DI gap is still meaningful but raw
ADX hasn't accelerated. Filtering them out kills 12 high-leverage winners.

### adx_direction_only (DI cross only, no ADX floor)
Closest-to-baseline filter: -$6,322 (-11.6%). Trades drop 2,436 → 2,179 (10%
of signals filtered). Per-symbol:
- JPN225ft DD improves 6.0% → 3.1% — material risk reduction.
- US2000.r still loses $2.9K — DI cross misalignment correlates with their
  best mean-reversion entries.

### regime_adx_strict (trending≥28, volatile≥25, low_vol≥20)
US2000.r holds well ($21.5K vs baseline $26.1K, only -17%) but UKOUSD
collapses from $7K → $97 (PF 3.63 → 1.19) and DJ30.r loses $3.2K. UKOUSD has
a meaningful pool of ADX 22-27 trending-regime entries that all get filtered.

## Per-symbol recommended thresholds

| Symbol      | Recommendation                | PnL change |
|-------------|-------------------------------|-----------:|
| DJ30.r      | global_adx_18 (floor=18)      |   +$1,218  |
| SWI20.r     | BASELINE — no filter          |       —    |
| XAUUSD      | BASELINE — no filter          |       —    |
| AUDJPY      | global_adx_18 (floor=18)      |     +$292  |
| EURUSD      | BASELINE — no filter          |       —    |
| US2000.r    | BASELINE — no filter          |       —    |
| UKOUSD      | global_adx_18 (floor=18)      |      +$96  |
| JPN225ft    | global_adx_22 (floor=22)      |       +$5  |

Aggregate +$1,612 if the 4 "winners" got their per-symbol floor applied.
That's still below the $30 ship threshold per symbol when noise is
considered, but the policy is **non-destructive** for those 4 — keeping the
overlay there is at worst neutral and at best a small lift.

Per-symbol DD breakdown for the recommended cells: **DD unchanged or
slightly worse** for DJ30/AUDJPY/UKOUSD (within 0.3pp), meaning the filter
doesn't improve drawdown either. So these per-symbol gains are inside the
noise band and **not actionable** as standalone settings — fine to leave them
"as recommended" but not worth a code change in isolation.

## Why does the gate fail?

1. `get_regime` already absorbs the most important ADX information by
   bucketing into `ranging / low_vol / volatile / trending`. The min-quality
   threshold per regime then handles the "is this bar tradable" question.
2. Selectivity *is the edge* — but the existing quality threshold is doing
   that work. Adding a redundant ADX floor on top filters bars that already
   passed quality, which by construction includes bars whose ADX is at the
   low end *but score is high anyway*. Those are the high-conviction signals
   regime-based filtering deliberately keeps.
3. The DI cross filter `adx_direction_only` knocks out trades where price
   action is *against* the dominant directional movement. That sounds right
   in theory but in v5's signal model, those trades are already
   high-quality mean-reversion entries (e.g. fade extreme + EMA reversion).
   Removing them removes a profitable trade type.
4. The 8-symbol test set is the live, hardened universe — it's already past
   the major selectivity rounds (Phase 1-7b indicator tune, RL gating, MTF
   cascade, MIN_EDGE, EV gate). The marginal entry that an ADX floor would
   reject has already been heavily filtered, so the floor is mostly
   subtracting capital from already-vetted bars.

## Recommendation

**Do not ship any global ADX overlay.** The existing regime+quality stack is
the correct industry implementation already; adding a parallel ADX floor is
double-counting.

**Per-symbol case**: optionally enable `global_adx_18` for DJ30.r / AUDJPY /
UKOUSD and `global_adx_22` for JPN225ft if a future tuning round wants to
re-test these with other signals. The aggregate +$1.6K is *below the noise
band* and not worth a code-change in isolation, but it confirms these four
symbols' edge sits in ADX≥18-22 zones — useful prior for next-step research
(e.g. per-symbol min_quality re-tune by regime, or per-symbol component
weights).

**Next experiments worth running** (not in this budget):
- Per-symbol `min_quality` tune by regime (already partially done in
  `tune_session_20260521/regime_min_quality.json`)
- Component-weight tune for US2000.r specifically — its $26K PnL hinges on
  high-quality low-ADX setups that no global filter respects.
- ADX *slope* (rising vs falling ADX) instead of level — would test whether
  trend acceleration rather than absolute strength is the better gate.

## Files

- `09_run_adx.py` — experiment runner (monkey-patches scorer; no source edits)
- `09_adx.json` — full per-symbol, per-variant, per-fold output
- `09_adx.md`   — this report
- (input) `baseline.json` — produced by `run_baseline.py`
