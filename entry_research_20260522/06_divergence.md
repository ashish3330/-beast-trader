# 06 — Momentum Divergence Entries

_Generated: 2026-05-22 00:13:35  •  Symbols: 8  •  180d baseline + 5-fold WF (540d, embargo 1d)_

## Concept
- **Regular divergence** — reversal. Price HH + RSI LH → SHORT setup. Price LL + RSI HL → LONG setup.
- **Hidden divergence** — continuation. In uptrend, Price HL + RSI LL → LONG (trend resuming).
- Detection: Williams 5-bar fractals over the last 60 bars (lr=2, min gap 5 bars). RSI is primary momentum series.

## 180-day backtest

| Variant | Trades | PnL $ | Δ vs baseline |
|---------|-------:|------:|--------------:|
| baseline | 1309 | +9736.39 | — |
| REGULAR_DIV_FILTER | 1184 | +9292.79 | -443.60 |
| HIDDEN_DIV_BOOST | 304 | +509.94 | -9226.45 |
| DIV_REVERSAL_ENTRY | 1200 | +11994.58 | +2258.19 |
| REGULAR_DIV_FILTER_H4 | 1081 | +6336.89 | -3399.50 |
| HIDDEN_DIV_BOOST_H4 | 367 | +3457.07 | -6279.32 |

## Walk-Forward 5-fold

| Variant | Folds | PF μ | PF σ | Σ PnL $ | Folds positive |
|---------|------:|-----:|-----:|--------:|---------------:|
| DIV_REVERSAL_ENTRY | 5 | 2.83 | 0.99 | +12809.63 | 5/5 |
| REGULAR_DIV_FILTER | 5 | 2.89 | 1.65 | +9237.97 | 5/5 |
| REGULAR_DIV_FILTER_H4 | 5 | 2.46 | 0.90 | +7258.13 | 5/5 |
| HIDDEN_DIV_BOOST_H4 | 5 | 2.63 | 1.45 | +4184.42 | 5/5 |

## Ship decision

Ship if Δ ≥ $30 **AND** WF PF μ > 1.50 **AND** ≥ 3/5 folds positive.

| Variant | Δ 180d $ | WF PF μ | Folds+ | SHIP |
|---------|---------:|--------:|-------:|:----:|
| DIV_REVERSAL_ENTRY | +2258.19 | 2.83 | 5/5 | **YES** |
| REGULAR_DIV_FILTER | -443.60 | 2.89 | 5/5 | no |
| REGULAR_DIV_FILTER_H4 | -3399.50 | 2.46 | 5/5 | no |
| HIDDEN_DIV_BOOST_H4 | -6279.32 | 2.63 | 5/5 | no |

**Overall: SHIP at least one variant.**
