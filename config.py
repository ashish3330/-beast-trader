"""
Dragon Trader — Tick-Level ML Trading Agent.
Account: 25106421, $2,500, VantageMarkets-Demo.
MT5 bridge: localhost:18813 (rpyc via Wine).
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ═══ MT5 CREDENTIALS ═══
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "25106421"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "R4q9Tyq$")
MT5_SERVER = os.getenv("MT5_SERVER", "VantageMarkets-Demo")
MT5_HOST = "localhost"
MT5_PORT = 18813  # Separate bridge from ApexQuant (18812)


@dataclass
class SymbolConfig:
    symbol: str
    magic: int
    category: str = "Forex"
    digits: int = 5
    tick_value: float = 1.0      # per lot per point — updated at runtime
    volume_min: float = 0.01
    volume_max: float = 10.0
    volume_step: float = 0.01


# ═══ 28 SYMBOLS — k-fold-validated + user-confirmed 2026-05-09 ═══
# K-fold time-series CV (5 folds, 7d embargo, 540d) is the canonical overfit gate.
# 1-fold holdout this morning was a weaker test that mis-flagged 9 symbols as
# OVERFIT — k-fold revealed they're actually ROBUST. NAS100.r kept after user
# confirmed live profitability (k-fold test PF 22.99 / PnL +$1.2k is selectivity
# not overfit when test PnL is positive).
#
# Still dropped: EURGBP (k-fold PF 0.85), EURCHF (PF 0.87 + neg PnL),
#   USDJPY (test PF 3.62 but n=24 too under-sampled to deploy).
# Backups: config.py.bak.20260509-31sym, .bak.20260509-19sym, .bak.20260509-27sym
# 2026-05-14 PHASE 8 GRADE — top-5 360d performers ONLY for $1.2K account.
# Decision rule per user: tier-up to dormant pool when equity >= $5000.
#
# ACTIVE (top 5 by 360d PnL post-Phase 8):
#   US2000.r  +$36,580 PF 3.20  ←  primary driver, A+ both horizons
#   SWI20.r   +$1,184  PF 4.03  ←  consistent A+
#   JPN225ft  +$854    PF 4.96  ←  highest PF, low data but stable
#   DJ30.r    +$703    PF 2.11  ←  strong 180d $2.7K
#   SPI200.r  +$622    PF 2.52
#
# DORMANT (positive 360d but not top 5 — re-enable at equity ≥ $5000):
#   XAUUSD +$622 / EURUSD +$395 / GER40.r +$353 / UKOUSD +$291 /
#   BTCUSD +$197 / AUDJPY +$48 / XAGUSD +$20
#   Tunes are preserved in auto_tuned.py — toggle the SYMBOLS entry back ON.
#
# REMOVED (negative 360d — drop entirely until evidence reverses):
#   EURAUD -$18 / COPPER-Cr -$40 / HK50.r -$77 / NG-Cr -$80 / GAS-Cr -$86
SYMBOLS: Dict[str, SymbolConfig] = {
    # 2026-05-15: expanded from top-5 to 9 A-grade symbols per user direction.
    # All 9 are positive on BOTH 180d AND 360d backtests. Phase 9 hard tune
    # in progress will refine SL×IND×ST_F×min_quality per symbol.
    #
    # ── ACTIVE TIER (5-sym lock per 2026-06-17 user req for 30-day no-touch) ──
    # XAUUSD + EURUSD + UK100.r + BTCUSD + DJ30.r ONLY.
    # ── 2026-06-24 user req: lock universe to BTC + XAU only (SMABO profitable
    #    subset since 2026-06-22; all other syms + strategies turned off). ──
    "XAUUSD":     SymbolConfig("XAUUSD",     8100, "Gold",      2),
    # "EURUSD":     SymbolConfig("EURUSD",     8370, "Forex",     5),   # OFF 2026-06-24
    # "UK100.r":    SymbolConfig("UK100.r",    8250, "Index",     2),   # OFF 2026-06-24
    "BTCUSD":     SymbolConfig("BTCUSD",     8130, "Crypto",    2),
    # "DJ30.r":     SymbolConfig("DJ30.r",     8320, "Index",     2),   # OFF 2026-06-24

    # ── DISABLED 2026-06-17 (30-day no-touch lock to top-5) ──
    # "JPN225ft":   SymbolConfig("JPN225ft",   8230, "Index",     2),
    # "SPI200.r":   SymbolConfig("SPI200.r",   8500, "Index",     2),
    # "SWI20.r":    SymbolConfig("SWI20.r",    8440, "Index",     2),
    # "ETHUSD":     SymbolConfig("ETHUSD",     8140, "Crypto",    2),
    # "AUDJPY":     SymbolConfig("AUDJPY",     8260, "Forex",     3),
    # "NAS100.r":   SymbolConfig("NAS100.r",   8210, "Index",     2),
    # "XPTUSD.r":   SymbolConfig("XPTUSD.r",   8150, "Gold",      2),
    # "CHFJPY":     SymbolConfig("CHFJPY",     8280, "Forex",     3),
    # "USOUSD":     SymbolConfig("USOUSD",     8480, "Commodity", 3),
    # 2026-06-17 RE-ENABLED via universe-scan workflow wf_cf52aee9-6b5:
    # 180d BT on current sniper config: PF 4.05 / 156 tr / +$2467 / DD <=15%.
    # Prior 2026-06-02 CTO DISABLE was on the pre-sniper config — sniper
    # filters fixed the chop signals that bled then.
    # "US2000.r":   SymbolConfig("US2000.r",   8470, "Index",     2),   # OFF 2026-06-24 (SMABO -22.85 since Mon)
    # 2026-06-17 RE-ENABLED via universe-scan: 180d PF 8.35 / 60 tr / +$1366.
    # Highest PF in the entire universe scan. Pairs with DJ30.r US-indices coverage.
    # "SP500.r":    SymbolConfig("SP500.r",    8240, "Index",     2),   # OFF 2026-06-24
    # 2026-05-29 DISABLED — 0% WR over 12 trades.
    # "USDCAD":     SymbolConfig("USDCAD",     8380, "Forex",     5),
    # 2026-06-02 CTO DISABLE — 30d WR 0.0% / PF 0.0.
    # "USDJPY":     SymbolConfig("USDJPY",     8390, "Forex",     3),

    # ── DORMANT TIER (uncomment when equity ≥ $5000) ──
    # "XAGUSD":     SymbolConfig("XAGUSD",     8140, "Gold",      3),  # PF 1.33 marginal (next wave)
    # 2026-06-17 RE-ENABLED via universe-scan: 180d PF 4.39 / 335 tr / +$4534.
    # BEST overall profile in the scan — 72.8% WR, 5.2% DD. Single biggest
    # expansion add. Was disabled with "weak 360d" pre-sniper-config note.
    # "GER40.r":    SymbolConfig("GER40.r",    8200, "Index",     2),   # OFF 2026-06-24
    # "UKOUSD":     SymbolConfig("UKOUSD",     8460, "Commodity", 3),  # weak 360d

    # ── GATED AT EQUITY ≥ $8000 (user policy 2026-05-16) ──
    # User explicitly said: keep copper and gas disabled until account grows to $8K.
    # Walk-forward also flagged these so the lockout is defensible:
    #   COPPER-Cr — pass1 $1082 → walk-forward test PF 0.05 / $-82 (OVERFIT)
    #   GAS-Cr    — walk-forward WEAK (test PF 1.30, marginal)
    #   NG-Cr     — walk-forward WEAK (test PF 1.43, marginal)
    # Do NOT uncomment without re-running pass2 + walk-forward at the equity threshold.
    # "COPPER-Cr":  SymbolConfig("COPPER-Cr",  8410, "Commodity", 3),
    # "GAS-Cr":     SymbolConfig("GAS-Cr",     8420, "Commodity", 3),
    # "NG-Cr":      SymbolConfig("NG-Cr",      8430, "Commodity", 3),

    # ── REMOVED (negative 360d, do not re-enable without re-tune) ──
    # "HK50.r"  -$77/360d  PF 0.15
    # "EURAUD"  -$18/360d  PF 0.86
}

# ═══ AUX_SYMBOLS — order-placement registry for the self-contained TREND (+6000)
# and IMR (+7000) books, 2026-07-08. ═══
# These are DELIBERATELY kept OUT of SYMBOLS: every always-on management loop
# (kill-switch trailing @brain:1235, mtf dashboard, momentum scan) iterates
# `for sym in SYMBOLS` and fires LIVE mt5.positions_get(symbol=...) per symbol —
# adding them there would pile ~25 live bridge calls/cycle onto the flaky Wine
# bridge and starve the scalper / sync / order paths (root cause of "only BTC
# ever trades"). Instead the executor's order methods (open_trade_explicit /
# open_imr_trade) fall back to this registry ONLY to resolve base magic + volume
# specs when SYMBOLS.get() is None. Trend/IMR read D1 from disk cache and
# positions from the sync file, so these symbols never touch the bridge except
# at the single (throttled, one-per-cycle) order call. Emergency close-all still
# covers them (it uses positions_get() over ALL positions, not a SYMBOLS scan).
AUX_SYMBOLS: Dict[str, SymbolConfig] = {
    "ETHUSD":     SymbolConfig("ETHUSD",     8140, "Crypto",    2),   # TREND
    "NAS100.r":   SymbolConfig("NAS100.r",   8210, "Index",     2),   # TREND
    "JPN225ft":   SymbolConfig("JPN225ft",   8230, "Index",     2),   # TREND + IMR
    "SP500.r":    SymbolConfig("SP500.r",    8240, "Index",     2),   # IMR
    "US2000.r":   SymbolConfig("US2000.r",   8470, "Index",     2),   # IMR
    "GER40.r":    SymbolConfig("GER40.r",    8200, "Index",     2),   # SR + Momentum (2026-07-17)
    "SPI200.r":   SymbolConfig("SPI200.r",   8500, "Index",     2),   # SR + FVG
    "DJ30.r":     SymbolConfig("DJ30.r",     8320, "Index",     2),   # Momentum + ASAT + FIB50
    "EURUSD":     SymbolConfig("EURUSD",     8370, "Forex",     5),   # Momentum + FVG
    "USDCAD":     SymbolConfig("USDCAD",     8380, "Forex",     5),   # FVG
    "USOUSD":     SymbolConfig("USOUSD",     8480, "Commodity", 3),   # FVG
}


def symbol_cfg(symbol):
    """Base magic / volume specs for any strategy symbol — SYMBOLS first, then the
    trend/IMR AUX registry. Use for order placement + magic attribution so the
    trend/IMR books work without polluting the always-on SYMBOLS scan loops."""
    return SYMBOLS.get(symbol) or AUX_SYMBOLS.get(symbol)


# magic sub-offset → strategy label (single source of truth for classification /
# dashboard labelling). Keep in sync with the *_SUB_OFFSETS / *_MAGIC_OFFSET defs.
_STRATEGY_BY_OFFSET = {
    0: "swing", 1: "swing", 2: "swing",          # momentum SUB_MAGIC_OFFSETS
    500: "scalp", 501: "scalp",                  # SCALP_MAGIC_OFFSET (momentum micro)
    1000: "fvg", 1001: "fvg",
    2000: "sr", 2001: "sr",
    3000: "smabo", 3001: "smabo",
    4000: "fib50", 4001: "fib50",
    5000: "scalper", 5001: "scalper",            # M1 SCALPER
    6000: "trend", 6001: "trend",
    7000: "imr", 7001: "imr",
    8000: "gold_smc", 8001: "gold_smc",          # GOLD_SMC (XAUUSD H1 hybrid SMC)
    9000: "asat", 9001: "asat",                  # ASAT (moved off 3000/3001 SMABO collision 2026-07-17)
}


def strategy_of_magic(magic, symbol):
    """Map a live position's magic to its strategy label (trend/imr/scalper/…),
    resolving the base magic via symbol_cfg so AUX (trend/IMR) symbols classify
    correctly instead of defaulting to 'swing'."""
    cfg = symbol_cfg(symbol)
    if cfg is None:
        return "swing"
    return _STRATEGY_BY_OFFSET.get(int(magic) - int(cfg.magic), "swing")

# Per-symbol ML meta-label toggle (Round 6 backtest with retrained models)
# ML ON: symbols where ML filter improves PF (verified per-symbol comparison)
# ML OFF: symbols where ML over-filters good signals (reduces trade count + PF)
DRAGON_ML_ENABLED = {
    "XAUUSD":   True,    # AUC 0.801
    "XAGUSD":   True,    # AUC 0.802
    "BTCUSD":   False,   # weak model, trend needs all signals
    "ETHUSD":   False,   # NEW: no model yet
    "NAS100.r": True,    # AUC 0.799
    "JPN225ft": False,   # AUC 0.665 too weak
    "USDJPY":   False,   # no model retrained for V5 yet
    "USDCAD":   False,   # ML over-filters
    "SP500.r":  False,   # NEW: no model yet
    "GER40.r":  False,   # NEW: no model yet
    "EURJPY":   False,   # NEW: no model yet
    "EURUSD":   False,   # NEW: no model yet
    "GBPUSD":   False,   # NEW: no model yet
    "GBPJPY":   False,   # NEW: no model yet
}

# ═══ MOMENTUM-ADAPTIVE FEATURES (gated — only enable after walk-forward proves) ═══
# Tested 2026-05-10. ALL FOUR DEFAULT OFF until backtest + walk-forward say
# otherwise. Validation pipeline: backtest/results/momentum_tune/.
#
# Feature 1: Position-size momentum multiplier. When momentum.score > 0.7 and
# direction-aligned with the entry signal, scale risk_pct up to 1.3x. Capped
# by MAX_RISK_PER_TRADE_PCT — never breaches account-level safety.
def _envbool(key: str, default: bool) -> bool:
    """Env var override for momentum flags so backtests can A/B without
    editing config. Truthy: 1/true/yes/on (case-insensitive)."""
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# DEPLOYED 2026-05-10: walk-forward 19/19 ROBUST, +21.8% test PnL vs baseline
# (in-sample 180d +24.0%, k-fold 5×540d +21.8% — minimal shrinkage = real edge).
# Backtest results: backtest/results/momentum_tune/.
MOMENTUM_SIZE_BOOST_ENABLED = _envbool("MOMENTUM_SIZE_BOOST_ENABLED", True)

# Feature 2: Adaptive trail. score >= 0.7 → trail mult 0.7x (tighter — lock
# explosive bursts). score <= 0.3 → trail mult 1.2x (wider — let slow moves
# breathe). Multiplier applied to the existing TRAIL_STEPS distance.
# DEPLOYED 2026-05-11 (deep tune v2): widens trail+SL+lock-thresholds on
# high momentum so trends extend; tightens on weak signals. Walk-forward
# 5-fold +24.3% test PnL vs baseline.
MOMENTUM_TRAIL_ADAPTIVE_ENABLED = _envbool("MOMENTUM_TRAIL_ADAPTIVE_ENABLED", True)

# 2026-05-11 (deep tune v3): split the SL-widening from trail+lock so we
# can run trail-extends-winners WITHOUT enlarging losses on ranging days.
# Live evidence: 32 trades at 72% WR but PF 0.6 — wins +0.3R avg, losses
# -1.3R avg. The 1.3x wider SL was inflating losses. Default this flag
# OFF until proven separately profitable on a quieter-regime backtest.
MOMENTUM_SL_ADAPTIVE_ENABLED = _envbool("MOMENTUM_SL_ADAPTIVE_ENABLED", False)

# 2026-05-12: MTF cascade entry gate. Reject entries where 2+ higher TFs
# (W1, D1, H4) oppose the entry direction. Sniper-grade trend filter.
# Aggregated from H1 candles (no separate data feed needed) so it works
# identically in live and backtest. Default ON pending backtest validation.
MTF_CASCADE_ENABLED = _envbool("MTF_CASCADE_ENABLED", True)

# ═══ CONSERVATIVE PROFIT PROTECTION (2026-05-12) ═══
# Live evidence: SL hits cost 1.5-1.7R due to spread+slippage, while wins
# capped at +0.6R. Backtest can't model this. Add two live-only guards:
#
# 1. PEAK-GIVEBACK CIRCUIT BREAKER — if profit retraces past 50% of peak,
#    close at market. Forces 50% retention of best profit ever achieved.
PEAK_GIVEBACK_ENABLED = _envbool("PEAK_GIVEBACK_ENABLED", True)
PEAK_GIVEBACK_TRIGGER_R = 0.7   # only kick in after trade reached +0.7R
PEAK_GIVEBACK_FRAC = 0.5        # close if current < peak * 0.5

# 2026-05-19: per-symbol peak-giveback override for high-PF symbols that
# benefit from letting small peaks ride. Default (0.7R, 0.5) was too tight
# for SWI20 — closed at +0.45R on a 0.92R peak ($20 → $9). For these
# symbols, only fire on 1.5R+ peaks AND require 60% retrace (frac=0.4).
# Schema: {symbol: (trigger_r, frac)}. Cell-miss → global defaults.
PEAK_GIVEBACK_PER_SYMBOL: Dict[str, tuple] = {
    # 2026-05-20: dropped trigger 1.5R→1.0R after DJ30 gave back from 1.75R
    # peak. New trail (with tighter 1R+ params + 0.7R lock) handles sub-1R;
    # peak-giveback covers 1R-1.5R sweet spot where trail still gives back
    # 0.5R. Frac stays 0.4 (close on 60% retrace from peak).
    # 2026-06-04 CTO audit B2: per-symbol tightening based on follow-through
    # analysis — 63% of PG exits left favorable continuation. Raise trigger so
    # we don't cut at +0.4R when +2.5% (5R) was on the table (JPN225 case).
    "SWI20.r":  (1.0, 0.4),
    "DJ30.r":   (1.0, 0.4),   # best PG override, keep
    "NAS100.r": (1.0, 0.4),
    "SP500.r":  (1.0, 0.4),
    "JPN225ft": (1.3, 0.4),   # exits at +0.41R left +2.5% follow-through
    "XPTUSD.r": (1.0, 0.4),
    "SPI200.r": (1.2, 0.4),   # 6 exits all <0.24R = trigger too low
    "US2000.r": (1.0, 0.4),   # peaks <=0.06R typically, raise trigger
    "ETHUSD":   (1.2, 0.4),   # cutting at 0.06R peaks, missed 3% runners
    "XAUUSD":   (1.0, 0.4),   # saving from -14R SLs but trigger needs lifting
    "AUDJPY":   (1.0, 0.4),
}

# 2. EARLY-LOSS-CUT — if trade goes to -0.5R and stays there for N cycles
#    without reaching positive territory, close at market. Saves the
#    spread/slippage cost of full SL hit (which would be -1.5-1.7R live).
EARLY_EXIT_ENABLED = _envbool("EARLY_EXIT_ENABLED", True)
EARLY_EXIT_TRIGGER_R = -0.8     # threshold: -0.5R or worse
EARLY_EXIT_CYCLES = 60          # 2026-05-16: 60→20 cycles (~30s→10s). Trade #753 BTCUSD
                                # bled from -0.5R to -3.0R in the 30s T1-SLOW window because
                                # the slow-bleed tier waited too long while a momentum spike
                                # walked price against it. 10s is enough to filter noise on
                                # forex but not let a high-vol asset run -3R against a -1R SL.
EARLY_EXIT_REQUIRE_BAR_CLOSE = True  # 2026-06-05: T1 (slow tier) only fires after entry M15 bar closes; T2/T3 unchanged. Research: PaperToProfit 87-stop study + Davey — 10s polling fires inside signal-candle wick noise.

# 3. HARD DOLLAR LOSS CAP (2026-05-14) — catastrophic-outlier guard.
#    Live evidence (2026-05-13/14):
#      COPPER-Cr -$44 / -36.3R single trade (GuardianHeatReduce)
#      XAGUSD    -$33 / -17.2R single trade (EarlyLossCut too slow)
#      GER40.r   -$9.5 / -7.7R single trade
#    These are GAP-THROUGH losses — price jumps past SL faster than the
#    bot can react. EARLY_EXIT_CYCLES (60 cycles = 30s) is too slow for
#    instant gaps. Hard $-cap closes ANY position whose unrealized loss
#    exceeds this fraction of equity, regardless of R.
#    Default 2% of equity = ~$25 on $1.25K account. One bad trade can't
#    wipe out 30 winners anymore.
HARD_DOLLAR_CAP_ENABLED = _envbool("HARD_DOLLAR_CAP_ENABLED", True)
HARD_DOLLAR_CAP_PCT = 0.020     # 2% of equity = max single-trade loss

# Feature 3: Pyramid into winners. When existing position is +1.5R unrealized
# AND momentum still aligned, open a half-size add at next pullback to EMA20.
# Only one pyramid per parent position. Tracked in entry_metadata.
MOMENTUM_PYRAMID_ENABLED = _envbool("MOMENTUM_PYRAMID_ENABLED", False)
MOMENTUM_PYRAMID_TRIGGER_R = 1.5
MOMENTUM_PYRAMID_SIZE_FRAC = 0.5    # of base position

# Feature 4: Regime-adaptive MIN_SCORE delta. TRENDING_HARD → -0.5 (catch more
# of the move). RANGING/DEAD → +1.0 (be picky). Bounded so MIN_SCORE never
# drops below 6.0 floor regardless of regime — selectivity edge stays intact.
MOMENTUM_MIN_SCORE_ADAPTIVE_ENABLED = _envbool("MOMENTUM_MIN_SCORE_ADAPTIVE_ENABLED", False)
MOMENTUM_MIN_SCORE_FLOOR = 6.0

# Feature 5: Momentum-adaptive TP per (sym × regime × score-tier). 2026-06-19.
# Source dict: auto_tuned.ADAPTIVE_TP_PER_SYM_REGIME (research-derived 2026-06-19).
# Resolver: agent.expert.adaptive_tp.get_adaptive_tp(symbol, regime, score).
# Replaces the legacy 3-leg SUB_TP_R ladder with a 2-leg (TP1, TP2) per-cell tune
# when enabled. Default OFF for shadow rollout — A/B requires journal labelling
# at trade close so the lift can be measured per cell.
# ADAPTIVE_TP_FAIL_OPEN: on any exception inside the resolver, executor falls
# back to adaptive_sub_tp_r() (legacy path). Default True — never block a trade
# on a TP-tuning bug.
ADAPTIVE_TP_ENABLED = _envbool("ADAPTIVE_TP_ENABLED", True)  # 2026-06-19: ON per user req — tune results landed, flip now
ADAPTIVE_TP_FAIL_OPEN = _envbool("ADAPTIVE_TP_FAIL_OPEN", True)

# ═══ DRAGON RISK MANAGEMENT (aggressive but survivable — demo phase) ═══
# 90-day PF 1.72 (recent market harder) — stay aggressive but not suicidal
# Compound growth sim: 0.8% risk = $1K → $7.3K/year (630%) with ~30% peak DD
MAX_RISK_PER_TRADE_PCT = 0.5        # 2026-05-29: HALVED 1.0→0.5 — live WR 35% over 85 trades (-$56/3d). Damage control per feedback_dont_overfit_backtest_when_live_bleeding. Restore to 1.0 only after WR recovers >50% over 30+ trades.
MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "80.0"))  # 2026-07-13: raised 25→50 (user) so the bot trades ALONGSIDE the manual XAU shorts (~30% exposure) instead of being blocked. Still a hard kill-switch safety net; env-tunable.

# ═══ FVG STRATEGY (2026-05-29 — ICT sweep+FVG, runs alongside momentum) ═══
# Separate magic range (base+1000/+1001 → 9100-9501) so it never collides with
# momentum (8100-8500) or scalp (8600-9000). FVG yields to momentum (skips a
# symbol momentum already holds) but manages its own isolated positions.
# Tuned config: +496R/2yr backtest (10 syms, PF 1.06-1.53). UNPROVEN LIVE →
# conservative 0.25% risk until it earns trust. Backtest has no slippage/
# concurrency — expect live to be materially worse.
FVG_ENABLED = False                 # 2026-07-22 audit: DEPRECATED — completely DARK (0 live signals across all 5 whitelist symbols, ever; per-symbol time-stops ignored live/executor hardcodes global; stuck positions throw 'manage error' every cycle). No realized edge. Revisit only if a symbol produces a firing positive-expectancy setup.
FVG_RISK_PCT = 0.25                 # half of momentum — unproven strategy
FVG_MAGIC_OFFSET = 1000             # FVG legs at base+1000, base+1001
FVG_SUB_OFFSETS = [1000, 1001]
FVG_MAX_CONCURRENT = 7              # 2026-05-29: 3→7 = one per whitelist symbol
                                    # (full coverage). Each leg 0.25%×size_mult,
                                    # so 7 concurrent ≈ 1.75% base FVG exposure.
FVG_TIME_STOP_SECS = 6 * 3600       # close if TP1 not hit within 6h (tuned)

# ═══ 2026-06-05 — MOMENTUM STRATEGY MASTER TOGGLE ═══
# The 11-indicator momentum-score-then-confirm system was proved on 2026-06-05
# to systematically enter at swing extremes ("buy swing high / sell swing low"
# pathology — see feedback_value_entry_research_20260605.md). Disabled here
# while sweep-reclaim is being validated. FVG continues to run independently.
MOMENTUM_ENABLED = True     # 2026-07-15 user: ALL books ON
                            # The 11-indicator score still has the late-entry
                            # pathology, but it's empirically positive on a subset
                            # of symbols (US indices + gold + Japan) where the
                            # "trend confirmation" maps to real index drift +
                            # gold safe-haven flow. Disabled on FX/silver/small-cap
                            # where the late-entry trap dominates.

# Per-symbol whitelist — momentum runs ONLY on these symbols.
# Empirical basis: 14d live (2026-05-22 to 2026-06-05) journal split:
#   XAUUSD net  +$43 (35 momentum trades)  ← gold has persistent drift
#   DJ30.r net  +$75 (similar count)       ← US index after-hours has trend
#   SPI200.r net +$16                     ← Asian indices drift
#   JPN225ft  flat to slight positive
# All other symbols (USDCAD -$41, XAGUSD -$31, US2000 -$23, SP500 -$28,
# EURUSD -$24, UK100 -$21, USDJPY -$20, BTC -$14, ETH -$12) were net negative
# AND the bleeding majors were the bulk of trade count. Permanently restricted.
# Add a symbol here ONLY after 30+ live trades show net positive on it.
MOMENTUM_SYMBOL_WHITELIST = {"XAUUSD"}  # 2026-07-18 architect: XAU-only — live GER40/DJ30/JPN momentum PF 0.24/0.00/0.20; only XAU is live+BT positive (6/6 live, BT 2.07). Re-add indices when equity >= $5K.
# 2026-06-17 expansion: added GER40/SP500/US2000 (universe-scan winners) +
# DJ30/BTC/UK100/EURUSD (live universe — were excluded from momentum, only
# traded via SR). JPN225ft kept for legacy (no-op since not in SYMBOLS).

# ═══ ICT-STYLE LIQUIDITY SWEEP GATE (2026-06-16) ═══
# Sniper-grade entry gate: require liquidity grab (stop hunt + reclaim) on
# H1 within last 12-24 bars before allowing a momentum entry.
#   LONG  : within lookback, some bar must low < prior 5-bar swing-low AND
#           close back above that swing-low (stop-hunt below then reclaim).
#   SHORT : symmetric — high > prior 5-bar swing-high AND close < it.
# Background: feedback_value_entry_research_20260605 + user observation
# 2026-06-15 ("are we entering on swing high — shouldn't we wait for the
# liquidity sweep first?"). Gate 3e CHASE guard blocks the worst late
# entries by 4h-range position; this gate is the structural ICT version.
# Default ON for sniper-grade entries; env-toggle to disable for testing.
ICT_SWEEP_REQUIRED_FOR_MOMENTUM = _envbool("ICT_SWEEP_REQUIRED_FOR_MOMENTUM", True)
ICT_SWEEP_LOOKBACK_BARS = 24    # how many recent H1 bars to scan for a sweep
ICT_SWEEP_FRACTAL_N = 5         # swing-pivot lookback (N bars either side)

# ────────────────────────────────────────────────────────────────────────
# Gate 3g — Tick-volume Order-Flow Imbalance (participation filter)
# ────────────────────────────────────────────────────────────────────────
# Sits between Gate 3f (ICT sweep) and Gate 4 (position mgmt). Requires
# the latest H1 bar's tick_volume to exceed (BREAKOUT) or at least meet
# (MEAN_REVERT) a multiple of its 20-bar MA. Filters "no-participation"
# false moves where a signal-grade move prints on dry tape (Wyckoff
# no-demand / no-supply). Ships dark (False) for A/B comparison vs the
# live no-gate baseline before becoming default-on.
TV_VOLUME_GATE_ENABLED        = _envbool("TV_VOLUME_GATE_ENABLED", True)
TV_VOLUME_LOOKBACK_BARS       = 20      # 20-bar MA, excludes current bar
TV_VOLUME_BREAKOUT_MIN_RATIO  = 1.30    # 30% above 20-bar MA = real participation
TV_VOLUME_REVERT_MIN_RATIO    = 0.70    # >=70% of MA = enough liquidity to fade
TV_VOLUME_HARD_FLOOR_RATIO    = 0.30    # absolute REJECT (TV_DEAD_TAPE) for any setup
TV_VOLUME_PER_SYMBOL          = {}      # {sym: {'BREAKOUT': float, 'MEAN_REVERT': float}}
TV_VOLUME_WARN_ONLY_SYMBOLS   = set()   # REJECT -> log.warning only (still passes)
TV_VOLUME_TIMEFRAME           = 60      # candle timeframe in minutes (H1=60)

# 2026-06-08 workflow Proposal 2: dropped SPI200.r — Stream D raw-edge BT
# confirms PF 0.43 / -$70 / DD 8.1% on 60d. ML gate is a band-aid; no
# real edge exists. Live journal +$5.68 came from FVG/SR, not momentum.
# DJ30.r NOT re-added despite live +$75 — 3yr BT PF 0.51 (per project
# memory) outweighs 14d live anecdote; FVG can still catch DJ30 setups.

# 2026-06-08 workflow Proposal 1: ML meta-label gate BYPASS for symbols
# where the ML model has been empirically shown to remove edge. Stream D
# 60d backtest: JPN225 ML-ON PF 4.88 → ML-OFF PF 14.19 (+191% PF, +114%
# trades, +DD reduction 3.8→2.6%); XAU ML-ON PF 2.10 → ML-OFF PF 2.47.
# AUC is good (0.704/0.671) but threshold mapping vetoes the highest-
# quality signals on these two. Revert: remove entry.
ML_BYPASS_SYMBOLS = {"XAUUSD", "JPN225ft"}
# 2026-06-05 PM v2: DROPPED DJ30.r after 3-year H1 backtest validation.
#   - 3yr PF 0.51, 2H PF 0.07 (severely decaying — not lucky-bad, actively broken NOW)
#   - The +$75 / 14d live was a lucky window in a dead 3-year edge
# SPI200.r kept: 3yr PF 1.23, 2H PF 1.33 (stable, slightly improving — real edge)
# XAUUSD + JPN225ft kept on FAITH: their H1 caches are only 29 days (need refresh
# before we can validate). Their live 14d positive may also be luck — flag for next
# session: refresh raw_h1_xauusd.pkl / raw_h1_JPN225ft.pkl to 1yr+ then re-validate.

# ═══ 2026-06-05 — LIQUIDITY-SWEEP-RECLAIM STRATEGY (replaces momentum) ═══
# Research basis: Adam Grimes / Linda Raschke / Toby Crabel / Zarattini —
# all explicitly avoid indicator-stack entries. Single-bar event detector that
# enters at the structural inflection (the reclaim close), with a structural
# stop 0.1 ATR beyond the sweep wick.
SR_ENABLED = True            # 2026-07-15 user: ALL books ON
SR_TRADE_LIVE = True         # 2026-07-15 user: ALL books ON
                             # (10 signals across 5 syms). Retro hit rate 40%
                             # over N=10 is noisy — accepted as a demo trial
                             # at 0.25% risk. Re-evaluate after 30+ live trades:
                             # kill if WR < 40% over 5+ consecutive losses
                             # (per feedback_dont_overfit_backtest_when_live_bleeding).
SR_RISK_PCT = 0.25           # half of momentum's 0.5% — conservative until proven
SR_MAGIC_OFFSET = 2000       # own magic range (base+2000/+2001) — never collides
SR_SUB_OFFSETS = [2000, 2001]
SR_MAX_CONCURRENT = 4        # cap concurrent SR trades across whole portfolio
SR_POST_CLOSE_COOLDOWN_SECS = 900   # 15min between SR trades on same symbol
# SR-specific trail steps — entry is at +0R (just took the reclaim).
# No lock below TP1=1.0R; above TP1 the broker closes 50% and we trail the
# runner with progressive locks. Mirror's FVG_TRAIL_STEPS shape.
SR_TRAIL_STEPS = [
    (5.0, "trail", 0.4),
    (3.0, "trail", 0.5),
    (2.0, "lock",  1.5),       # at TP2, lock 1.5R on the runner
    (1.0, "lock",  0.4),       # at TP1, lock 0.4R on the runner half
    # No lock below 1R — let the structural stop do its job; EarlyLossCut
    # at -0.8R + bar-close guard already covers the loss side.
]

# ════════════════════════════════════════════════════════════════════════
# SMA CROSSOVER BREAKOUT (SMABO) — 4th strategy, 2026-06-21
# ════════════════════════════════════════════════════════════════════════
# HTF-aware trend-continuation breakout. 4H S/R (last 50 H4 bars), M15
# SMA(8/50) crossover + SMA(20) trail. Independent magic range (+3000/+3001).
# Default OFF until backtest validation lands. Mirrors SR/FVG enable/risk
# convention so the wiring agent can flip TRADE_LIVE without code changes.
# Detector at agent/sma_breakout.py — pure-function (read-only state).
# Backtest at backtest/sma_breakout_backtest.py.
# ════════════════════════════════════════════════════════════════════════
# PER-(STRATEGY × SYMBOL) N-CONSEC-LOSS HALT — 2026-06-22
# ════════════════════════════════════════════════════════════════════════
# Independent halt per (strategy, symbol). After PER_STRATEGY_SYMBOL_KILL_LOSSES
# consecutive losses on a (strat, sym) combo, that combo is blocked from new
# entries until next UTC midnight (or manual clear). Other syms/strategies
# unaffected. Persisted to agent_state DB.
# Implementation: agent/master_brain.py — is_strategy_symbol_halted(),
# record_strategy_symbol_close(). Strategies call these around their entry +
# close paths.
PER_STRATEGY_SYMBOL_KILL_ENABLED = _envbool("PER_STRATEGY_SYMBOL_KILL_ENABLED", True)
PER_STRATEGY_SYMBOL_KILL_LOSSES = int(os.getenv("PER_STRATEGY_SYMBOL_KILL_LOSSES", "10"))


SMABO_ENABLED = _envbool("SMABO_ENABLED", False)            # 2026-07-18 architect: OFF — live PF 0.58 / 120 trades / -$295 (XAU 0.66, BTC 0.38). Worst bot bleed; live n=120 overrides BT.
# 2026-06-21 PM: FLIPPED LIVE after 2-batch hard tune produced 5 WF-validated
# per-sym overrides (XAU, BTC, SP500.r, US2000.r, NAS100.r). DJ30 + GER40 +
# UK100 still anti-edge at defaults → blacklisted to prevent untuned bleed.
# EUR + CHFJPY also blacklisted (no edge found in tunes).
# Net: SMABO trades 4 live syms (XAU/BTC/SP500/US2000), all with tuned params.
SMABO_TRADE_LIVE = _envbool("SMABO_TRADE_LIVE", True)
SMABO_RISK_PCT = float(os.getenv("SMABO_RISK_PCT", "0.25")) # conservative until proven
SMABO_MAGIC_OFFSET = 3000                                   # base+3000/+3001 — own range
SMABO_SUB_OFFSETS = [3000, 3001]
SMABO_MAX_CONCURRENT = int(os.getenv("SMABO_MAX_CONCURRENT", "4"))
SMABO_POST_CLOSE_COOLDOWN_SECS = int(os.getenv("SMABO_POST_CLOSE_COOLDOWN_SECS", "900"))
SMABO_KILL_AFTER_LOSSES = int(os.getenv("SMABO_KILL_AFTER_LOSSES", "3"))   # daily kill at N consec losses
# Whitelist expanded 2026-06-21 PM with WF-validated additions (NAS100, US2000).
# NAS100.r added to support live deployment (not in SYMBOLS dict yet — needs
# config.SYMBOLS update before it can fire).
# 2026-06-24 user req: SMABO restricted to its two profitable syms since Mon
# (XAU +12.89, BTC +10.53). SP500/US2000 dropped — US2000 bled -22.85, SP500 ~flat.
# Prior whitelist preserved for reference:
#   {"XAUUSD","EURUSD","BTCUSD","DJ30.r","SP500.r","US2000.r","NAS100.r"}
SMABO_WHITELIST = {"XAUUSD", "BTCUSD"}
# EURUSD blacklisted 2026-06-21 — hard tune found NO edge (best PF 0.81, -106R/365d,
# DD 29%). Structural mismatch: H4 S/R trend-breakout with 1:2 RR doesn't fit
# low-vol FX. 42-combo sweep + 5-axis coord-descent confirmed anti-edge.
# CHFJPY blacklisted 2026-06-21 PM — sweep returned NO winner (no combo cleared
# trade floor + DD ceiling on 60-combo coord-descent).
# DJ30 + GER40 + UK100 blacklisted 2026-06-21 PM at live-flip time — defaults
# are anti-edge (PF 0.85-0.90 baseline) and tunes either failed WF (DJ30 had
# recent-fold decay -48R/-22R; GER40 combined PF only 1.04) or weren't run
# (UK100). Re-enable per-sym after individual tune lands WF-validated overrides.
# NOTE: BTCUSD SMA-breakout edge decayed in the recent chop regime. Instead of
# blacklisting, BTC now runs a dedicated MEAN-REVERSION strategy (BTCMR, own
# magic range) suited to its ranging behaviour — see agent/btc_mean_reversion.py.
SMABO_SYMBOL_BLACKLIST: set = {"EURUSD", "CHFJPY", "DJ30.r", "GER40.r", "UK100.r"}
# Per-symbol param overrides — tuned 2026-06-21 via coord-descent + 5-fold WF.
# Detector reads these from sma_breakout.py:255 (sym_ov = SMABO_PARAM_OVERRIDES.get(symbol, {})).
# Format: {"SYMBOL": {"FAST_SMA": int, "SLOW_SMA": int, "TRAIL_SMA": int,
#                      "HTF_LOOKBACK_BARS": int, "MIN_RR": float}}
SMABO_PARAM_OVERRIDES: dict = {
    # XAU 365d: PF 1.02→1.23 (+654% R), DD 9.0→6.3%, 545 trades.
    # WF 5-fold: 4/5 folds positive R, combined PF 1.15, +61R, DD 6.2%.
    # ADX_MIN 18 (2026-07-05 Fable fleet, validated): skip SMA breakouts in chop
    # (ADX<18). Fixes decayed recent folds — all 4 WF folds positive, PF 1.23->1.42,
    # DD 6.4%->4.0%, 421 trades. Single-knob, plateau-robust (ADX 16-20 all lift).
    # MAX-PROFIT tune 2026-07-06 (25-combo MIN_RR x ADX_MIN grid, 2yr + 4-fold WF):
    # MIN_RR=2.0 + ADX_MIN=16 → PF 1.50, +193R, DD 4.1%, ALL 4 folds positive
    # (min-fold PF 1.30 = most robust in the grid), WR 39%. Nearly 2x the profit
    # of the original (+101R) at lower DD. User chose profit over win-rate.
    "XAUUSD": {"FAST_SMA": 8, "SLOW_SMA": 20, "TRAIL_SMA": 34,
               "HTF_LOOKBACK_BARS": 50, "MIN_RR": 2.0, "ADX_MIN": 16},
    # BTC 365d: PF 0.85→1.22 (anti-edge → edge), DD 19.4→5.2%, 489 trades.
    # WF 5-fold: 4/5 folds positive R, combined PF 1.25, +82R, DD 4.4%.
    # BTC switched to MEAN-REVERSION 2026-07-05 (breakout edge dead in chop).
    # Fable fleet winner: CONFIRM entry + BB2.5 + RSI20/80 + ADX<20 range gate.
    # Fades stretched moves back to the mean; recent WF fold strongly + (+0.37R
    # vs breakout's −0.15R). Routes through SMABO live pipeline via STRATEGY_MODE.
    "BTCUSD": {"STRATEGY_MODE": "mean_reversion",
               "CONFIRM": 1, "BB_MULT": 2.5, "RSI_LOW": 20, "RSI_HIGH": 80,
               "ADX_MAX": 20, "SL_ATR": 2.0, "TIME_STOP_BARS": 16},
    # SP500.r 365d: PF 0.90→1.14 (anti-edge → edge), 355 trades, DD 5.6%.
    # WF 5-fold: 4/5 folds positive (only fold 3 -3.8R), combined PF 1.12,
    # +31R, DD 5.3%. STRONGEST WF result of the second batch.
    "SP500.r": {"FAST_SMA": 21, "SLOW_SMA": 100, "TRAIL_SMA": 34,
                "HTF_LOOKBACK_BARS": 50, "MIN_RR": 3.0},
    # US2000.r 365d: PF 0.85→1.14 (anti-edge → edge), 642 trades, DD 10.1%.
    # WF 5-fold: 3/5 folds positive, wins much bigger than losses (+33/+29/+44
    # vs -12/-3), combined PF 1.20, +91R, DD 8.7%.
    "US2000.r": {"FAST_SMA": 13, "SLOW_SMA": 20, "TRAIL_SMA": 14,
                 "HTF_LOOKBACK_BARS": 30, "MIN_RR": 3.0},
    # NAS100.r 365d: PF 1.07→1.13 (modest lift, baseline already positive),
    # 819 trades, DD 9.7%. WF 5-fold: 3/5 folds positive, combined PF 1.14,
    # +79R, DD 10.3%. NOT in live SYMBOLS — override ready for future addition.
    "NAS100.r": {"FAST_SMA": 8, "SLOW_SMA": 50, "TRAIL_SMA": 14,
                 "HTF_LOOKBACK_BARS": 50, "MIN_RR": 2.5},
    # DJ30.r and GER40.r SKIPPED — DJ30 had recent-fold decay (folds 4+5 both
    # negative -48 / -22R), GER40 combined PF only 1.04 with 3/5 positive.
    # Both remain at defaults (effectively anti-edge baseline).
}
# Strategy params (defaults match agent/sma_breakout.py — overridable via env
# or per-symbol overrides above).
SMABO_FAST_SMA = int(os.getenv("SMABO_FAST_SMA", "8"))
SMABO_SLOW_SMA = int(os.getenv("SMABO_SLOW_SMA", "50"))
SMABO_TRAIL_SMA = int(os.getenv("SMABO_TRAIL_SMA", "20"))
SMABO_HTF_LOOKBACK_BARS = int(os.getenv("SMABO_HTF_LOOKBACK_BARS", "50"))  # 4H bars for S/R
SMABO_MIN_RR = float(os.getenv("SMABO_MIN_RR", "2.0"))
# SMABO trail steps — entry at +0R, TP1 at +2R minimum (per spec). No lock
# below TP1 (would clip the primary edge). Post-TP1 trail with SMA20 fallback
# handled by executor's _apply_trail (or kept simple: post-TP1 → BE, then
# trail with progressive locks above 2R).
SMABO_TRAIL_STEPS = [
    (5.0, "trail", 0.4),
    (3.0, "trail", 0.5),
    (2.0, "lock",  0.5),       # at TP1, lock 0.5R on the runner half
    # No lock below 2R — never clip the 1:2 minimum target.
]

# ══════════════════════════════════════════════════════════════════════════
# INDICES MEAN-REVERSION (IMR) — 8th strategy, 2026-07-08. Long-only D1 RSI(2)+IBS
# dip-buys above SMA200 on cash indices. Detector agent/indices_mr.py, backtest
# scripts/_indices_mr_backtest.py (live-verified costs incl swap): basket OOS>=2023
# PF 2.22 / WR 69% / Sharpe 1.64 (full PF 1.52). THE account-appropriate strategy —
# min-lot sizing = ~0.3% risk/trade. 6xATR SL = DISASTER only, NEVER trailed. Exits
# detector-driven at next D1 open. Signal-only burn-in first (flip live after a week).
IMR_ENABLED = _envbool("IMR_ENABLED", True)
IMR_TRADE_LIVE = _envbool("IMR_TRADE_LIVE", True)    # LIVE 2026-07-10 (validated OOS PF 2.22, signal-only burn-in done)
IMR_MAGIC_OFFSET = 7000
IMR_SUB_OFFSETS = [7000, 7001]
IMR_WHITELIST = {"US2000.r"}  # 2026-07-22 audit: SP500.r DROPPED — min-lot risks 2.2% > 1.5% cap = STRUCTURALLY UNFILLABLE, rejecting ~1,893x/day (each a Wine bridge round-trip feeding the trade-path wedge), 0 fills ever. JPN225ft dropped 07-18 (WF-fail). US2000-only remains (marginal but fillable).
IMR_FIXED_LOTS = {"SP500.r": 0.10, "US2000.r": 0.10, "JPN225ft": 1.0}
IMR_PARAMS = {"RSI_ENTRY": 15.0, "IBS_ENTRY": 0.30, "RSI_EXIT": 65.0,
              "SMA_TREND": 200, "SMA_EXIT": 5, "ATR_PERIOD": 14,
              "SL_ATR": 6.0, "TIME_STOP_DAYS": 7}
IMR_MAX_CONCURRENT = 3       # Per-trade risk is now capped at 1.5% in the executor (in-executor cap, 2026-07-17).
                             # NOTE: all 3 IMR_WHITELIST syms are correlated equity indices, so 3 concurrent
                             # trades can stack up to ~4.5% aggregate index risk at the per-trade cap.
IMR_DECISION_HOUR_UTC = 1

# ── GOLD SMC (Hybrid SMC + Momentum/Breakout) — 8th strategy, 2026-07-12 ──
# The one validated new edge from the SMC research: XAUUSD H1 (D1 bias), BASELINE
# params (the tune overfit to the recent gold bull; baseline PF 1.15, positive
# across BOTH halves of 11.7yr). SIGNAL-ONLY first (prove live like IMR did).
# Own magic +8000/+8001. Detector: agent/gold_smc.py.
GOLD_SMC_ENABLED = _envbool("GOLD_SMC_ENABLED", True)
GOLD_SMC_TRADE_LIVE = _envbool("GOLD_SMC_TRADE_LIVE", True)    # LIVE (demo) 2026-07-12: full 2-leg path wired, broker-side SL/TP2 always protect
GOLD_SMC_SYMBOL = "XAUUSD"
GOLD_SMC_MAGIC_OFFSET = 8000
GOLD_SMC_SUB_OFFSETS = [8000, 8001]
GOLD_SMC_RISK_PCT = float(os.getenv("GOLD_SMC_RISK_PCT", "0.30"))
GOLD_SMC_PARAMS = {"SWING": 5, "SWEEP_LB": 10, "SEQ": 8, "SL_ATR": 0.2,
                   "TP1_R": 1.5, "TP2_R": 2.0, "BIAS_EMA": 50,
                   # 4-of-5 confluence + regime gate (2026-07-18): bias+sweep+EMA-cross
                   # stay mandatory; require >=MIN_CONFL of [BOS,FVG,VWAP,MACD/RSI,candle];
                   # regime gate (ADX>=ADX_MIN AND ATR not in bottom ATR_PCT_MIN of its
                   # trailing ATR_PCT_WIN) keeps churn <=1.4x vs the old all-5-AND.
                   # Must stay IN SYNC with scripts/_forex_smc_backtest.py.
                   "MIN_CONFL": 4, "ADX_N": 14, "ADX_MIN": 16.0,
                   "ATR_PCT_WIN": 500, "ATR_PCT_MIN": 0.30}

# ── EMERGENCY EXIT GATE (2026-07-12) — portfolio-wide statistical exits from the
# journal's per-symbol win-rate / avg-win-pts / avg-loss-pts. Detector:
# agent/emergency_exit.py. Runs over ALL open positions (any strategy), closes
# read-free via _close_magic_legs. LOG-ONLY until EMERGENCY_EXIT_LIVE (a hard
# avg-win cap deletes 38-56% of winning points on this bot's history — the
# 'trail' mode is the tail-safe default; 'hard' is the literal exit-at-avg-win). ──
EMERGENCY_EXIT_ENABLED = _envbool("EMERGENCY_EXIT_ENABLED", True)
EMERGENCY_EXIT_LIVE = _envbool("EMERGENCY_EXIT_LIVE", False)      # log-only "would-exit" first
EMERGENCY_EXIT_LOOKBACK = int(os.getenv("EMERGENCY_EXIT_LOOKBACK", "80"))       # recent trades/sym
EMERGENCY_EXIT_MIN_SAMPLES = int(os.getenv("EMERGENCY_EXIT_MIN_SAMPLES", "15"))
EMERGENCY_EXIT_CFG = {
    "win_mode": os.getenv("EMERGENCY_EXIT_WIN_MODE", "trail"),   # "trail" (tail-safe) | "hard"
    "win_mult": float(os.getenv("EMERGENCY_EXIT_WIN_MULT", "1.0")),
    "giveback": float(os.getenv("EMERGENCY_EXIT_GIVEBACK", "0.35")),
    "loss_cut": _envbool("EMERGENCY_EXIT_LOSS_CUT", True),
    "loss_mult": float(os.getenv("EMERGENCY_EXIT_LOSS_MULT", "1.0")),
}

# ── EMERGENCY LOSS CAP (2026-07-15) — fixed per-symbol open-$ loss cap that fires
# from the FIRST trade. Unlike EMERGENCY_EXIT (which needs journal samples +
# min_samples, and whose avg-loss stat is entry-price-dependent), this is a
# hand-tuned constant per symbol: when a (symbol, strategy) group's LIVE open loss
# reaches -cap, close it. No journal, no min_samples — works on trade #1.
#   • MANUAL legs (hand-placed XAU shorts; magics in EMERGENCY_MANUAL_MAGICS) are
#     NEVER auto-closed.
#   • XAUUSD is intentionally omitted from the cap dict — gold is run by GOLD_SMC's
#     own BE/EMA9 exit + the manual shorts, so this gate leaves it alone.
# LOG-ONLY until EMERGENCY_LOSS_CAP_LIVE. Dollar caps are STARTING values — tune. ──
EMERGENCY_LOSS_CAP_ENABLED = _envbool("EMERGENCY_LOSS_CAP_ENABLED", True)
EMERGENCY_LOSS_CAP_LIVE    = _envbool("EMERGENCY_LOSS_CAP_LIVE", False)  # 2026-07-15 user: OFF — no per-symbol position capping
EMERGENCY_MANUAL_MAGICS    = {0, 2024}     # hand-placed trades — never auto-closed by any emergency gate
EMERGENCY_LOSS_CAP_USD = {                 # per-symbol open-$ loss cap (positive $). XAUUSD omitted on purpose.
    # HARD-TUNED 2026-07-15 via per-trade MAE R-sweep + 60/40 walk-forward, one
    # agent per symbol, against the live TREND book (D1 3-EMA, +6000). Only caps
    # that improved total-R on BOTH WF folds ship. Result: TREND's own chandelier
    # trail + peak-giveback already bound adverse excursion <1R on BTC/ETH/NAS100,
    # so a fixed cap there only clips fat-tailed winners (net-negative) — those are
    # DELIBERATELY LEFT OUT (uncapped). JPN225ft is the sole shipper: a clean gap
    # between winner MAE (≤0.69R) and 2 rare tail blowups (~0.95R) means $10 (0.8R)
    # cuts exactly the blowups, clips 0 winners, +on both folds (+$1/3.5yr — tail
    # guard, not alpha). $12+ is inert (above max MAE). See scratchpad tune scripts.
    #   REJECTED by backtest (uncapped): ETHUSD (−4$/6.5yr), NAS100.r.
    # BTCUSD: backtest says uncap (clips winners), but USER OVERRIDE 2026-07-15 —
    # live keeps bleeding ~-$20/trade; per feedback_dont_overfit_backtest_when_live_bleeding
    # we act on live pain. $15 (~1.2R) caps the bleed; being re-tuned by the
    # full-mechanism 10-agent sweep. Revisit once that lands.
    "BTCUSD":   15.0,
    "JPN225ft": 10.0,
    # NAS100.r: USER OVERRIDE 2026-07-15 — live gave back a +$6.82 win then
    # re-entered and ran to -$22 UNCAPPED (it was left out of the dict). Cap it.
    "NAS100.r": 12.0,
}

# ── DAILY LOSS GATE (2026-07-12) — per-symbol daily $ MAX-LOSS circuit breaker.
# Once a symbol's TODAY P/L (realized closed-trades $ from the journal + open $
# from the sync file) drops to -limit, CLOSE that symbol's open positions (cut the
# bleeding) and BLOCK new entries on it for the rest of the UTC day. LOSS ONLY —
# profit is never capped, winners keep running. Only symbols listed are gated.
# LOG-ONLY until DAILY_LOSS_GATE_LIVE. Dollar limits are STARTING values — tune. ──
DAILY_LOSS_GATE_ENABLED = _envbool("DAILY_LOSS_GATE_ENABLED", True)
DAILY_LOSS_GATE_LIVE = _envbool("DAILY_LOSS_GATE_LIVE", False)   # 2026-07-15 user: OFF — no per-symbol position capping
DAILY_LOSS_LIMIT_USD = {                                         # max $ loss/symbol/day (positive)
    "XAUUSD": 40.0, "BTCUSD": 60.0, "ETHUSD": 30.0,
    "JPN225ft": 40.0, "NAS100.r": 15.0, "SP500.r": 30.0, "US2000.r": 30.0,  # NAS 40→15: once -$15/day, close + BLOCK re-entry for the day
}

# ════════════════════════════════════════════════════════════════════════
# M1 SCALPER (SCALP) — 6th strategy, 2026-07-07. XAU-only M1 mean-reversion fade.
# ════════════════════════════════════════════════════════════════════════
# Research-backed (session gate + ADX<18 regime + ATR-expansion + target-mean)
# and hard-tuned: PF 1.43, 4.4 trades/day, ALL 4 walk-forward folds + (1.31-1.62),
# DD 6.9R, spread-charged. Own magic +5000/+5001. Detector: agent/m1_scalper.py.
# Backtest: scripts/_scalper_run.py. CAVEAT: 52d M1 data, no slippage modelled —
# demo-validate live before trusting. Conservative risk until proven.
SCALPER_ENABLED = _envbool("SCALPER_ENABLED", True)
SCALPER_TRADE_LIVE = _envbool("SCALPER_TRADE_LIVE", True)
SCALPER_RISK_PCT = float(os.getenv("SCALPER_RISK_PCT", "0.15"))
SCALPER_MAGIC_OFFSET = 5000
SCALPER_SUB_OFFSETS = [5000, 5001]
SCALPER_MAX_CONCURRENT = int(os.getenv("SCALPER_MAX_CONCURRENT", "1"))
SCALPER_POST_CLOSE_COOLDOWN_SECS = int(os.getenv("SCALPER_POST_CLOSE_COOLDOWN_SECS", "60"))
SCALPER_KILL_AFTER_LOSSES = int(os.getenv("SCALPER_KILL_AFTER_LOSSES", "6"))
SCALPER_TIME_STOP_BARS = int(os.getenv("SCALPER_TIME_STOP_BARS", "30"))  # 2026-07-15 tune: 10→30 (WF+cross-window, PF↑). M1 bars ≈ minutes
SCALPER_WHITELIST = {"XAUUSD"}  # 2026-07-23: BTCUSD REVERTED — after adding it, wedges climbed to a sustained 10-11/hr (was 0-4) causing hourly bot restarts, while BTC-scalp delivered zero value (unvalidated XAU-tuned params, hadn't even traded — session starts 07:00). Pre-committed rollback: the M1 BTC scan added bridge load; XAU-only keeps the scalper lean. (If wedges DON'T drop after this, it was active-hours not BTC — but BTC-scalp had no value either way.)
SCALPER_PARAMS = {
    "PERIOD": 20, "BB_MULT": 1.8, "RSI_PERIOD": 2, "RSI_LOW": 10.0, "RSI_HIGH": 90.0,  # RSI 10/90 (07-15); BB_MULT 2.0→1.8 (07-18 R1: PF +3.8%, net +36%, DD -22%, spread-robust interior peak)
    "SL_ATR": 0.8, "ADX_MAX": 18.0, "H_START": 7, "H_END": 20,  # SL_ATR 1.0→0.8 (2026-07-15 tune on refetched 100k M1: PF 1.07→1.28, DD↓, +both folds)
}

# ════════════════════════════════════════════════════════════════════════
# TREND-FOLLOWER (TREND) — 7th strategy, 2026-07-07. THE ROBUST CORE.
# ════════════════════════════════════════════════════════════════════════
# Diversified volatility-targeted time-series trend-following on DAILY bars
# (Moskowitz-Ooi-Pedersen 2012 / AQR century-of-evidence). Own magic +6000.
# Backtest scripts/_trend_backtest.py: Sharpe 0.58 over 23yr, 0.53/0.66 across
# both OOS halves (robust, not curve-fit), 0.84 last-5yr, ~10%/yr, DD ~16-30%,
# WR 52%. Signal = 3-speed EMA-crossover ensemble; wide 3xATR catastrophic stop;
# exit/flip on signal change (daily rebalance). Detector agent/trend_follower.py.
# This is the SLOW, low-turnover, positive-skew antidote to the intraday losers.
TREND_ENABLED = _envbool("TREND_ENABLED", True)
TREND_TRADE_LIVE = _envbool("TREND_TRADE_LIVE", True)
TREND_RISK_PCT = float(os.getenv("TREND_RISK_PCT", "0.30"))   # per-instrument, to 3xATR stop
TREND_MAGIC_OFFSET = 6000
TREND_SUB_OFFSETS = [6000, 6001]
TREND_ATR_STOP = float(os.getenv("TREND_ATR_STOP", "3.0"))    # catastrophic tail guard
TREND_ATR_PERIOD = 20
# 2026-07-15: crypto ATR_PERIOD 14 was applied then REVERTED same day. It shipped
# off the stale 50k-bar ETH H1 (cross-confirmed on BTC D1), but re-tuning on the
# freshly-refetched 20k H1 flipped the optimum to 28 — the param is window-unstable
# (14↔28), i.e. overfit, not edge. Reverted to the global 20 for all symbols per
# feedback_validate_backtests. Resolver kept as infra for a future robust per-sym tune.
TREND_ATR_PERIOD_PER_SYMBOL = {}   # empty → trend_atr_period() falls back to global 20
def trend_atr_period(symbol):
    """Per-symbol ATR period for the TREND book; falls back to the global."""
    return int(TREND_ATR_PERIOD_PER_SYMBOL.get(symbol, TREND_ATR_PERIOD))
TREND_EMA_PAIRS = [(16, 64), (32, 128), (64, 256)]            # 3-speed ensemble
# HARDEST TUNE 2026-07-11 (signal-param sweep, 4-fold rolling WF + churn guard,
# scripts/_trend_signal_tune.py): the current 3-EMA signal is the validated
# optimum for XAU/BTC/JPN/NAS (nothing beats it) and regime/ADX-gating the signal
# does NOT survive (consistent with the prior regime-overfit). The ONE robust
# improvement: ETHUSD prefers a WIDER slow-leg ensemble (+8.7% return, 3/4 OOS
# folds, sane turnover). Caveat: weaker in the most-recent fold — monitor.
TREND_EMA_PAIRS_PER_SYMBOL = {
    "ETHUSD": [(16, 96), (32, 160), (64, 256)],
}


def trend_ema_pairs(symbol):
    return TREND_EMA_PAIRS_PER_SYMBOL.get(symbol, TREND_EMA_PAIRS)


# STATIC symbol specs (probed live 2026-07-12) so the ORDER path can size + place
# a trade WITHOUT any in-process symbol_info read — those fail under bridge
# contention for symbols hammered by the always-on loops (BTC entries were failing
# 100%). Price comes from the sync daemon's disk "quotes". These specs don't change.
TREND_SYMBOL_SPECS = {
    "BTCUSD":   {"digits": 2, "point": 0.01, "tick_value": 0.01, "tick_size": 0.01, "vmin": 0.01, "vmax": 100.0, "vstep": 0.01, "stops": 0},
    "ETHUSD":   {"digits": 2, "point": 0.01, "tick_value": 0.01, "tick_size": 0.01, "vmin": 0.01, "vmax": 100.0, "vstep": 0.01, "stops": 0},
    "JPN225ft": {"digits": 2, "point": 0.01, "tick_value": 6.182418438444751e-05, "tick_size": 0.01, "vmin": 1.0, "vmax": 20000.0, "vstep": 1.0, "stops": 50},
    "NAS100.r": {"digits": 2, "point": 0.01, "tick_value": 0.01, "tick_size": 0.01, "vmin": 0.1, "vmax": 500.0, "vstep": 0.1, "stops": 50},
    "SP500.r":  {"digits": 2, "point": 0.01, "tick_value": 0.01, "tick_size": 0.01, "vmin": 0.1, "vmax": 500.0, "vstep": 0.1, "stops": 50},
    "US2000.r": {"digits": 2, "point": 0.01, "tick_value": 0.01, "tick_size": 0.01, "vmin": 0.1, "vmax": 500.0, "vstep": 0.1, "stops": 50},
    "XAUUSD":   {"digits": 2, "point": 0.01, "tick_value": 1.0, "tick_size": 0.01, "vmin": 0.01, "vmax": 100.0, "vstep": 0.01, "stops": 20},
}
TREND_MIN_ABS_SIGNAL = float(os.getenv("TREND_MIN_ABS_SIGNAL", "0.34"))  # need >=2/3 agree
TREND_REBALANCE_HOUR = int(os.getenv("TREND_REBALANCE_HOUR", "1"))  # act on first cycle after this UTC hour
# GOOD-5 basket (Sharpe 0.65, both OOS halves robust). Includes GOLD.
# On a small account, gold/NAS100 min-lots would over-risk, so TREND_MAX_RISK_PCT
# caps EVERY trade by tightening its stop to fit — gold stays in, safely.
TREND_BASKET = ["XAUUSD", "JPN225ft", "NAS100.r"]  # 2026-07-22 audit: ETHUSD DROPPED (live PF 0.11 / net -$2.6, same 'short a rising crypto' pattern as BTC, worse economics). BTCUSD DROPPED 07-21 — live 30d net -$56.36 / PF 0.29 (loses 3x what it wins), all shorts into a rising BTC; big asymmetric losses -11/-11, -20/-20 per 2-leg trade vs +2/+3 wins. Matches the R2 tune finding (BTC = TREND's decayed weakest link). NAS100 PF 1.41 / JPN 1.08 kept. BTC still trades in SR (validated PF 1.46).
# 2026-07-08 (user): LOW RISK for all symbols — cap every trade at 1.0% even at
# min-lot (was 2.5%, which pinned index min-lots at the ceiling). The stop is
# tightened to fit this cap; the Chandelier trail (below) then protects profit.
TREND_MAX_RISK_PCT = float(os.getenv("TREND_MAX_RISK_PCT", "1.0"))  # hard cap/trade even at min-lot

# ── Realistic exit model, 2026-07-08 (user): the book previously rode with NO TP
#    (tp=0) until the EMA flipped and did NO trailing — so BTC showed no usable
#    target and gave back open profit. Now: a real ATR-distance TP + a daily
#    Chandelier trailing stop that only ever tightens. ──
TREND_TP_ATR = float(os.getenv("TREND_TP_ATR", "6.0"))        # realistic target = 6xATR from entry (2x the 3xATR stop)
TREND_TRAIL_ENABLED = _envbool("TREND_TRAIL_ENABLED", True)
TREND_TRAIL_ATR = float(os.getenv("TREND_TRAIL_ATR", "2.5"))  # Chandelier default (per-symbol overrides below); H1 tune: 2.0 churns, 2.5-3.0 is the realistic region
TREND_TRAIL_LOOKBACK = int(os.getenv("TREND_TRAIL_LOOKBACK", "22"))  # highest-high / lowest-low window (D1 bars)
# 2026-07-09 (user): the 2.5xATR chandelier is anchored to the 22d high, so on a
# pullback it locks almost nothing (NAS +270pts profit but SL locked only +61).
# Add a PROFIT-LOCK ratchet: once open profit >= ACTIVATE_ATR x ATR, lock
# LOCK_FRAC of it; the effective stop is the TIGHTER of {chandelier, profit-lock}.
# HARD-TUNE 2026-07-09 (per-symbol D1 sweep + ETH H1 intraday validation):
# the D1 sweep's optimum ran monotonically to the tightest edge (TR1.0/LK0.9) — an
# intra-bar fill ARTIFACT. ETH H1 confirmed a CHURN CLIFF at LOCK>=0.7 (trade count
# 25x, PF->100). So LOCK is held at 0.6 (just below the cliff); DO NOT raise it.
# Only-tightens either way; the peak-giveback below is the active reversal exit.
TREND_LOCK_FRAC = float(os.getenv("TREND_LOCK_FRAC", "0.6"))          # SL backstop: lock 60% of peak (0.7 = churn cliff)
TREND_LOCK_ACTIVATE_ATR = float(os.getenv("TREND_LOCK_ACTIVATE_ATR", "0.3"))  # ...once profit >= 0.3xATR
# 2026-07-09 (user): a tight SL still gives back on a fast reversal between cycles.
# Add a PEAK-GIVEBACK reversal exit — close at market when open profit retraces
# GIVEBACK_FRAC from its peak (a reversal shows up as profit rolling over). This
# is the ACTIVE exit (tighter, market-close, checked every 60s from the disk
# price); the profit-lock SL above is the broker-side backstop if we disconnect.
TREND_REVERSAL_EXIT_ENABLED = _envbool("TREND_REVERSAL_EXIT_ENABLED", True)
# 0.30 (was 0.35): the peak-giveback is the ARTIFACT-FREE way to keep more profit
# on reversal (a market close, not an SL that churns) — now keeps 70% of peak.
TREND_GIVEBACK_FRAC = float(os.getenv("TREND_GIVEBACK_FRAC", "0.30"))  # close if profit falls to 70% of peak
# 2026-07-14 (user "why no trades"): the reversal-exit re-entry block used to hold
# "until the daily signal flips" — but reversal exits are PROFIT-TAKING, so a whole
# basket would bank one giveback each and go dormant for days. Make it a TIME cooldown:
# after a reversal exit, block same-dir re-entry for this many hours, then re-participate.
TREND_REENTRY_BLOCK_HOURS = float(os.getenv("TREND_REENTRY_BLOCK_HOURS", "6.0"))  # 2026-07-15 user: 2->6h + now applies after ANY exit (trail/SL/reversal), not just reversal — stop the bank-a-win-then-re-enter-into-a-loss churn
# 2026-07-13 (user): peak-giveback must ARM once a trade reaches 0.5R for ANY
# symbol. Applied as a ceiling on the per-symbol activation (min of tuned vs this)
# so protection is live no later than 0.5R everywhere. 1R = entry->SL distance.
PEAK_GIVEBACK_ACTIVATE_R = float(os.getenv("PEAK_GIVEBACK_ACTIVATE_R", "0.5"))

# ── WINNER-ONLY un-cap of the peak-giveback reversal exit (2026-07-23) ──
# The peak-giveback + D1-bar throttle were added to stop the BTC/NAS re-entry
# CHURN. But for the two CONFIRMED live winners (NAS100.r +$34/30d PF1.41,
# JPN225ft positive) the peak-giveback MARKET-close also CLIPS a position that is
# riding a winning trend — it exits on a pullback instead of letting the winner
# ride the Chandelier + peak-lock broker trail. For these symbols ONLY, disable
# the giveback market-close and let the trail (broker-side SL, always tightens)
# protect. This does NOT touch the entry throttle / D1-bar gate / reversal-block,
# so it CANNOT reintroduce entry churn (trade count is unchanged), and the
# broker-side chandelier + peak-lock SL still protects (no naked/wrong-side SL).
# XAU/BTC and every other symbol KEEP the strict giveback + throttle.
# Validated on the REAL NAS100.r/JPN225ft D1+H1 caches:
#   backtest/_trend_winner_uncap_validate_20260723.py  (canonical fixed-3xATR engine)
#   backtest/_trend_winner_uncap_livecap_20260723.py   (live risk-capped-stop sweep)
# Finding: churn-free (n identical) + SL-safe; under the live tight-capped stop
# it captures MORE NAS profit (3/3 thirds), and is inert for JPN (its giveback
# never fires). Gated behind this flag so it is trivially reversible.
TREND_WINNER_SYMBOLS = {"NAS100.r", "JPN225ft"}
TREND_WINNER_DISABLE_GIVEBACK = _envbool("TREND_WINNER_DISABLE_GIVEBACK", True)

# 24/7 (crypto) symbols — never subject to the None-open market-closed lockout: a
# None there is bridge contention, not a closed market, so keep retrying (2026-07-13).
ALWAYS_OPEN_SYMBOLS = {"BTCUSD", "ETHUSD"}

# ── PER-SYMBOL exit params, INTRADAY(H1)-validated 2026-07-09 ──
# Each symbol's D1 sweep optimum ran to the tightest edge (churn artifact); the
# H1 (hourly) backtest exposed a churn CLIFF where the profit-lock is tapped
# intraday + re-enters. These are the winners from the REALISTIC-turnover region
# (<=40 trades/yr), per symbol. deep H1: XAU/NAS 12yr, BTC/ETH 6yr, JPN 3.5yr.
# (TRAIL, LOCK, GIVEBACK, ACT). Falls back to the globals above for any other sym.
# HARD re-tune 2026-07-10 (10-agent workflow: 1728-config sweep x 4 ROLLING
# walk-forward folds + block-bootstrap + adversarial refutation, on deep H1).
# VERDICT: NO robust tunable exit edge — every symbol's tighter "winner" was a
# single-fold / n=1 / churn-cliff artifact that FAILED all-folds-positive or
# bootstrap-5th-pctile>=0. XAU & BTC had n_robust=0 (nothing survives). So we
# KEEP the conservative-wide values, with only the 3 safe nudges the adversarial
# agents endorsed (all LOOSEN, none tighten the churn-prone TRAIL/LOCK levers):
#   XAU GIVEBACK 0.25→0.30, BTC GIVEBACK 0.25→0.30, JPN ACT 0.5→0.3.
# Do NOT tighten TRAIL/LOCK chasing the backtest spike — that is the curve-fit
# trap this account has been burned by. Revisit only with new data/features.
TREND_EXIT_PER_SYMBOL = {
    "XAUUSD":   {"TRAIL": 2.5, "LOCK": 0.5, "GIVEBACK": 0.30, "ACT": 0.5},
    "BTCUSD":   {"TRAIL": 3.0, "LOCK": 0.5, "GIVEBACK": 0.30, "ACT": 0.5},  # n_robust=0; widest, no tunable edge
    "ETHUSD":   {"TRAIL": 2.5, "LOCK": 0.5, "GIVEBACK": 0.35, "ACT": 0.3},  # current beats all tighter (kept)
    "JPN225ft": {"TRAIL": 3.0, "LOCK": 0.6, "GIVEBACK": 0.35, "ACT": 0.4},  # ACT 0.3→0.4 (2026-07-15 tune: +on both WF folds, PF 2.49→3.12)
    "NAS100.r": {"TRAIL": 2.5, "LOCK": 0.6, "GIVEBACK": 0.35, "ACT": 0.5},  # current beats all tighter (kept)
}

# ── SELECTIVITY / CONVICTION GATE, 2026-07-10 (the "PF 6.91 discipline") ──
# On top of the all-3-EMA-agree signal, only ENTER when the daily trend is
# genuinely strong: ADX(14) >= ADX_MIN AND |slow-EMA slope over 10 bars| >=
# SLOPE_MIN * ATR. Fewer, higher-conviction trades = higher PF. Tuned per-symbol
# on deep H1 walk-forward (scripts/_trend_selectivity.py) — kept ONLY where it
# improved BOTH IS and OOS PF at a sane trade count:
#   NAS100.r slope>=0.5: OOS PF 7.8->16.0 (88->54 trades, 94% WR)
#   BTCUSD  ADX>=30:     OOS PF 1.52->18.0 (+2.51 ret, 90 trades, 90% WR)
# XAU/ETH/JPN: no gate — baseline already PF 56/22/2.3; a gate just cuts winners.
# 2026-07-10 ROLLING-WF RESEARCH VERDICT (scripts/_trend_entry_research.py +
# _trend_entry_validate.py, 4 anchored folds): the earlier single-split conviction
# gates DO NOT survive rigorous validation and were REMOVED —
#   BTC ADX>=30: "beat 3/4" but TRIPLED trade count (32->96) = churn/cost artifact
#   NAS slope>=0.5: beat only 1/4 folds — NAS baseline is 4/4-positive on its own
#   DIST filter: churn artifact (ETH trades 175->718). No entry filter robustly
#   improves the book. Trade the validated signal + exits; add NO unvalidated gate.
TREND_CONVICTION_PER_SYMBOL = {}


def trend_conviction(symbol):
    """(adx_min, slope_min) daily conviction gate for a trend symbol; (0,0)=no gate."""
    d = TREND_CONVICTION_PER_SYMBOL.get(symbol)
    return (d["ADX_MIN"], d["SLOPE_MIN"]) if d else (0.0, 0.0)


def trend_exit_params(symbol):
    """Intraday-tuned (TRAIL, LOCK, GIVEBACK, ACT) for a trend symbol; falls back
    to the global TREND_* defaults for any symbol not in the per-symbol table."""
    d = TREND_EXIT_PER_SYMBOL.get(symbol)
    if d:
        return d["TRAIL"], d["LOCK"], d["GIVEBACK"], d["ACT"]
    return TREND_TRAIL_ATR, TREND_LOCK_FRAC, TREND_GIVEBACK_FRAC, TREND_LOCK_ACTIVATE_ATR

# ════════════════════════════════════════════════════════════════════════
# FIB-50 PULLBACK CONTINUATION (FIB50) — 5th strategy, 2026-06-21
# ════════════════════════════════════════════════════════════════════════
# Continuation entry at 50% Fibonacci retracement after a strong impulse.
# Independent magic range (+4000/+4001) — no collision with momentum (base),
# FVG (+1000), SR (+2000), SMABO (+3000). M15 detector with fractal swing
# pivots (N=5). Default OFF until backtest validation lands.
# Detector at agent/fib50_strategy.py — pure-function (read-only state).
# Backtest at backtest/fib50_backtest.py.
FIB50_ENABLED = _envbool("FIB50_ENABLED", True)             # 2026-07-15 user: ALL books ON
FIB50_TRADE_LIVE = _envbool("FIB50_TRADE_LIVE", False)      # 2026-07-16 tune: DISABLED — live bleeder, NEGATIVE every symbol/window (XAU -42R, EUR -184R, BTC -103R, JPN -157R /520d), no WF-valid tune. Make-it-best call.
FIB50_RISK_PCT = float(os.getenv("FIB50_RISK_PCT", "0.20")) # conservative until proven
FIB50_MAGIC_OFFSET = 4000                                   # base+4000/+4001 — own range
FIB50_SUB_OFFSETS = [4000, 4001]
FIB50_MAX_CONCURRENT = int(os.getenv("FIB50_MAX_CONCURRENT", "4"))
FIB50_POST_CLOSE_COOLDOWN_SECS = int(os.getenv("FIB50_POST_CLOSE_COOLDOWN_SECS", "900"))
FIB50_KILL_AFTER_LOSSES = int(os.getenv("FIB50_KILL_AFTER_LOSSES", "3"))   # daily kill at N consec losses
FIB50_WHITELIST = {"XAUUSD", "EURUSD", "BTCUSD", "DJ30.r", "SP500.r", "JPN225ft"}
FIB50_SYMBOL_BLACKLIST: set = set()                                        # surgical kill switch
FIB50_PARAM_OVERRIDES: dict = {}                                           # per-symbol param overrides
# Strategy params (defaults match agent/fib50_strategy.py).
FIB50_SWING_PIVOT_N = int(os.getenv("FIB50_SWING_PIVOT_N", "5"))
FIB50_MIN_IMPULSE_ATR = float(os.getenv("FIB50_MIN_IMPULSE_ATR", "2.0"))
FIB50_MIN_RR = float(os.getenv("FIB50_MIN_RR", "1.5"))
FIB50_MAX_SL_R = float(os.getenv("FIB50_MAX_SL_R", "8.0"))   # wide cap — fib-50 SLs are naturally 5-8x ATR
FIB50_ATR_BUFFER = float(os.getenv("FIB50_ATR_BUFFER", "0.20"))
FIB50_SETUP_TTL_BARS = int(os.getenv("FIB50_SETUP_TTL_BARS", "20"))
# FIB50 trail steps — TP1 at 0.5*retrace, TP2 at swing extreme. Post-TP1 → BE.
# No lock below 1.5R (would clip TP1 minimum).
FIB50_TRAIL_STEPS = [
    (5.0, "trail", 0.4),
    (3.0, "trail", 0.5),
    (1.5, "lock",  0.3),       # at TP1, lock 0.3R on the runner half
    # No lock below 1.5R — never clip the TP1 target.
]


# ═══ ICT DISCOUNT / PREMIUM ZONE GATE (2026-06-19) ═══
# Computes H1 dealing range (highest-high → lowest-low over LOOKBACK bars),
# splits at 50% equilibrium. Rejects direction-vs-zone mismatches:
#   • LONG  in premium  (>= 50%) → REJECT (chasing top)
#   • SHORT in discount (<= 50%) → REJECT (selling bottom)
# STRICT_MODE=True: only deep zones approve (LONG bottom 30%, SHORT top 30%).
# Pure-function module at agent/expert/discount_premium_zone.py.
# Brain wraps in try/except — FAILS-OPEN on data hiccups.
# Default OFF — ships dark for 14d live shadow A/B before considering enable.
DISCOUNT_PREMIUM_GATE_ENABLED = _envbool("DISCOUNT_PREMIUM_GATE_ENABLED", False)
DISCOUNT_PREMIUM_LOOKBACK_BARS = int(os.getenv("DISCOUNT_PREMIUM_LOOKBACK_BARS", "60"))
DISCOUNT_PREMIUM_STRICT_MODE = _envbool("DISCOUNT_PREMIUM_STRICT_MODE", False)


# 2026-06-05: FVG trail steps — DIFFERENT from momentum TRAIL_STEPS.
# FVG entries have explicit broker-side TP1=1.5R / TP2=3.0R. Locking profit
# below 1.5R would clip TP1 hits, destroying the strategy's primary edge.
# So: no lock below 1.5R. Above 1.5R (post-TP1 runner), trail conservatively.
# All other exits (EarlyLossCut, PeakGiveback, TimeStop_NoProgress, BOS_Invalidation,
# VWAP_Cross_Exit) apply uniformly via _apply_trail — those protect against losses
# without interfering with TP targets.
FVG_TRAIL_STEPS = [
    (5.0, "trail", 0.4),       # let runner ride above 5R
    (3.0, "trail", 0.5),       # at TP2, trail 0.5 ATR
    (2.0, "lock",  1.0),       # between TP1 and TP2, lock +1R
    (1.5, "lock",  0.5),       # AT TP1 (broker closes 50%), lock 0.5R on runner half
    # No lock below 1.5R — let broker TP1 trigger on its own.
]
# 2026-05-29: trimmed to the 7 symbols positive over the RECENT 180d (not just
# the 2yr tune). NAS100.r (-8.8R), SP500.r (flat), USDJPY (+1.2R marginal) are
# BENCHED until they recover — deploy what works now, not 2 years ago.
FVG_WHITELIST = {
    # 2026-06-04 CTO audit B15: removed US2000.r — disabled in SYMBOLS dict but
    # FVG was still trading it (FVG_WHITELIST is independent of SYMBOLS).
    # Closes the leak. Re-add when US2000.r returns to SYMBOLS.
    # 2026-06-19 audit: XAUUSD removed — was bleeding -$53/7d via FVG path
    # while SR was already blacklisted. Orphaned + losing.
    "ETHUSD", "EURUSD", "SPI200.r", "USOUSD",  # JPN225ft dropped 2026-07-15: FVG held-out negative under every config
    # 2026-06-05 PM: ADDED USDCAD after universe backtest.
    # M15 PF 1.38 on 201 trades (+17.73R) — TIER-1 add, like-for-like timeframe.
    # Note: USDCAD was a MOMENTUM loser (-$41 / WR 6% / 14d) but FVG signal type
    # has different edge structure here. The signal-mismatch is real not noise.
    "USDCAD",
}
# Tuned global params (validated +496R basket). TP 2.0/4.0 is load-bearing.
FVG_PARAMS = {
    "SWING_LOOKBACK": 4, "TIME_STOP_HOURS": 6.0,
    "SETUP_EXPIRY_BARS_15M": 36, "TP1_R": 2.0, "TP2_R": 4.0,
    "SWEEP_TO_FVG_BARS": 12,
}
# Per-symbol overrides (only one materially helps: XAUUSD wants tighter swings).
FVG_PARAM_OVERRIDES = {
    "XAUUSD":   {"SWING_LOOKBACK": 2},   # +13.9R → +96.2R, PF 1.06 → 1.36
    # 2026-07-15 all-strategy hard-tune (WF 60/40 + 3-window cross-val, fresh m15).
    # Live-hookable subset of the FVG agent's per-sym ships; DEPTH/DISP axes are not
    # read by fvg_strategy.py so they're skipped.
    "EURUSD":   {"SWING_LOOKBACK": 4,  "TIME_STOP_HOURS": 8.0,  "SETUP_EXPIRY_BARS_15M": 18, "SWEEP_TO_FVG_BARS": 16, "SWING_MEMORY": 30},  # test -5.5→+14.4R
    "USOUSD":   {"SWING_LOOKBACK": 4,  "TIME_STOP_HOURS": 3.0,  "SETUP_EXPIRY_BARS_15M": 28},  # TIME_STOP 5→3, test 19.4→22.3R
    "ETHUSD":   {"SWING_LOOKBACK": 13, "TIME_STOP_HOURS": 6.0,  "SETUP_EXPIRY_BARS_15M": 6, "SWING_MEMORY": 30},  # 2026-07-18 R2: SWING 10→13 (broad plateau SL12-16, 3-thirds 2.92/1.14/3.53, test PF 2.81, 0.79x churn)
    "SPI200.r": {"SWING_LOOKBACK": 10, "TIME_STOP_HOURS": 4.0,  "SETUP_EXPIRY_BARS_15M": 24, "TP2_R": 5.0},  # TP2 4→5, test 25.4→25.9R
    "USDCAD":   {"TIME_STOP_HOURS": 18.0},  # 2026-07-18 tune-loop R1: 12→18 (interior peak, 3-thirds up 1.47/1.81/2.10, OOS PF 2.49, zero churn 188→188)
    # JPN225ft DROPPED from FVG_WHITELIST below (held-out still negative under every config)
}

# 2026-06-14: per-symbol time-stop overrides (hours). Falls back to
# FVG_PARAMS["TIME_STOP_HOURS"] if symbol absent. Lets us extend the
# time-stop on symbols (e.g. SPI200.r / JPN225ft) where the FVG retrace
# fill is structurally slower without affecting the rest of the universe.
FVG_TIME_STOP_HOURS_PER_SYMBOL = {
    # populated by the FVG 5-agent tune (2026-06-09 PM)
}

# 2026-06-14: per-symbol FVG trail-steps override. Falls back to FVG_TRAIL_STEPS
# defined above when symbol absent. Same shape as FVG_TRAIL_STEPS — list of
# (r_threshold, "lock"|"trail", value) tuples sorted descending by threshold.
FVG_TRAIL_PER_SYMBOL = {
    # populated by per-symbol trail tuners
}

# 2026-06-14: per-symbol FVG EarlyLossCut override. Each entry is
# {"enabled": bool, "r_threshold": float, "bar_close_guard": bool}. Falls back
# to the strategy-default ELC config when symbol absent.
FVG_ELC_PER_SYMBOL = {
    # populated by FVG audit / strategy-aware ELC tune
}

# 2026-06-14: SR per-symbol param overrides — same shape as FVG_PARAM_OVERRIDES.
# Keys read by sweep_reclaim detector: ATR_EXPANSION_MIN, BODY_RATIO_MIN,
# WICK_RATIO_MIN, HTF_REQUIRED, DAILY_LOSS_KILL_R.
# 2026-07-17: SR gets its OWN symbol whitelist, decoupled from the global SYMBOLS
# list (same self-contained pattern as TREND_BASKET / FVG_WHITELIST). SR trades
# ONLY these — its own config + symbols — resolved via AUX_SYMBOLS. Found by the
# SR-expand hard-tune (23 syms tested; WF 60/40 + 3-window + real-detector cross-
# check). Baseline SR bled on all 23; these 3 pass every gate with the tuned
# overrides below + global TP 2.0/2.0 & ADX<=25 (sweep_reclaim.py).
SR_WHITELIST = {"BTCUSD", "GER40.r", "SPI200.r"}

SR_PARAM_OVERRIDES = {
    # 2026-07-17 SR-expand tune winners (auth = real per-bar detector, full cache):
    "BTCUSD":   {"ATR_EXPANSION_MIN": 1.0, "BODY_RATIO_MIN": 0.15, "WICK_RATIO_MIN": 0.0, "HTF_REQUIRED": True, "DIRECTION_FILTER": "BOTH", "DAILY_LOSS_KILL_R": 2.5},  # auth PF 1.46 +37R/520d
    "GER40.r":  {"ATR_EXPANSION_MIN": 0.6, "BODY_RATIO_MIN": 0.30, "WICK_RATIO_MIN": 0.0, "HTF_REQUIRED": True, "DIRECTION_FILTER": "LONG", "DAILY_LOSS_KILL_R": 2.5},  # auth PF 1.32 +40R (LONG-only, DAX-uptrend regime-dep)
    "SPI200.r": {"ATR_EXPANSION_MIN": 1.4, "BODY_RATIO_MIN": 0.15, "WICK_RATIO_MIN": 0.0, "HTF_REQUIRED": True, "DIRECTION_FILTER": "BOTH", "DAILY_LOSS_KILL_R": 2.5},  # auth PF 1.93 +25R (strongest)
    # ── legacy per-sym overrides (only bind if a symbol is added back to SR_WHITELIST) ──
    # 2026-06-14: 10-agent workflow (wf_833e6497-e92) per-sym overrides.
    # Keys: ATR_EXPANSION_MIN, BODY_RATIO_MIN, WICK_RATIO_MIN, HTF_REQUIRED, DAILY_LOSS_KILL_R.
    "EURUSD":   {"ATR_EXPANSION_MIN": 1.0, "BODY_RATIO_MIN": 0.3,  "WICK_RATIO_MIN": 0.4,  "HTF_REQUIRED": True,  "DAILY_LOSS_KILL_R": 2.5},
    "USOUSD":   {"ATR_EXPANSION_MIN": 1.3, "BODY_RATIO_MIN": 0.65, "WICK_RATIO_MIN": 0.55, "HTF_REQUIRED": True,  "DAILY_LOSS_KILL_R": 2.0},
    "ETHUSD":   {"ATR_EXPANSION_MIN": 1.4, "BODY_RATIO_MIN": 0.6,  "WICK_RATIO_MIN": 0.35, "HTF_REQUIRED": True,  "DAILY_LOSS_KILL_R": 2.0},
    "JPN225ft": {"ATR_EXPANSION_MIN": 0.8, "BODY_RATIO_MIN": 0.3,  "WICK_RATIO_MIN": 0.6,  "HTF_REQUIRED": False, "DAILY_LOSS_KILL_R": 2.0},
    "DJ30.r":   {"ATR_EXPANSION_MIN": 1.4, "BODY_RATIO_MIN": 0.65, "WICK_RATIO_MIN": 0.3,  "HTF_REQUIRED": True,  "DAILY_LOSS_KILL_R": 2.5},
    "UK100.r":  {"ATR_EXPANSION_MIN": 1.3, "BODY_RATIO_MIN": 0.55, "WICK_RATIO_MIN": 0.35, "HTF_REQUIRED": True,  "DAILY_LOSS_KILL_R": 2.0},
    "NAS100.r": {"ATR_EXPANSION_MIN": 1.4, "BODY_RATIO_MIN": 0.65, "WICK_RATIO_MIN": 0.55, "HTF_REQUIRED": True,  "DAILY_LOSS_KILL_R": 2.0},
}

# 2026-06-14: SR blacklist — detector returns None early for any symbol in this
# set. Used to surgically disable SR on losers without breaking the universe.
SR_SYMBOL_BLACKLIST = {
    # 2026-07-17: BTCUSD REMOVED from blacklist — the -206R bleed was at defaults
    # (TP 1.3/wick 0.25); it now trades the tuned SR config (SR_WHITELIST + override
    # + TP 2.0/ADX25): auth PF 1.46 +37R. Kept blacklist below for the true losers.
    # 2026-06-14: defensive blacklist from 10-agent workflow (wf_833e6497-e92).
    # XAUUSD live -25.2R / 17% WR / n=12 — disable SR until structure recovers.
    "XAUUSD",
    # 2026-06-19 audit additions: stale-state firing on commented-out syms
    "XPTUSD.r",
    "AUDJPY",
    "CHFJPY",
    # 2026-07-18 R2: GER40.r REMOVED from blacklist. The 06-19 "structural-negative"
    # entry was at OLD defaults (wick 0.25/TP 1.3); the 07-17 expand tune re-validated
    # it (SR_WHITELIST + override, TP 2.0/ADX25: PF 1.26 +35R, test PF 1.168, 3 thirds
    # positive) but left this line, so the detector silently rejected it = validated
    # config was DEAD CODE, never fired live. Contradiction resolved; GER40 now trades.
    # 2026-06-19 confluence BT (wf_e13cc011-06a) surfaced SR anti-edge on
    # indices universe-wide:
    #   SP500.r SR  -$910 / 466 tr   ← worst PnL contribution
    #   US2000.r SR -$1068 / 477 tr  ← worst sym overall
    #   EURUSD SR   -$46 / 9 tr      ← thin sample, all losing
    #   UK100.r SR  -$18 / 2 tr      ← thin sample, anti-edge
    # SR is structurally broken on trending indices (mean-revert pattern
    # vs continuation regime). Restrict SR to syms with proven live edge.
    "SP500.r",
    "US2000.r",
    "EURUSD",
    "UK100.r",
}

# ═══ 2026-06-21 — EQH/EQL LIQUIDITY-POOL FILTER (sweep_reclaim) ═══════════
# Optional filter that ranks sweep_reclaim signals by whether the swept
# level coincides with a cluster of equal highs (EQH) or equal lows (EQL).
# Clusters of ≥2 equal pivots within 0.10×ATR14 concentrate resting stops —
# institutions deliberately sweep these "liquidity pools" before reversal.
# Reference: ICT / SMC liquidity-engineering, Daily Price Action.
#
#   EQH_EQL_FILTER_ENABLED = False  → filter inert (current behavior)
#   EQH_EQL_FILTER_ENABLED = True   → cluster sweeps get full size;
#                                     non-cluster sweeps downsized to 0.6×
#   EQH_EQL_STRICT          = True  → non-cluster sweeps are rejected
#                                     (no signal emitted)
# Both flags default OFF — opt-in after A/B journal validation.
EQH_EQL_FILTER_ENABLED = False
EQH_EQL_STRICT         = False

# ═══ 2026-06-16 — WYCKOFF SPRING / UPTHRUST DETECTOR ═══════════════════════
# Higher-conviction superset of sweep_reclaim. Adds (1) validated H1 trading-
# range context, (2) multi-touch S/R level, (3) optional low-volume TEST bar
# as the entry trigger. Refs: Wyckoff "Method", Pruden "Three Skills", Tom
# Williams VSA, Anna Coulling VPA. Default OFF — A/B test phase.
WYCKOFF_ENABLED = False       # master toggle for Wyckoff Spring/Upthrust detector
WYCKOFF_TRADE_LIVE = False    # False = signal-only/log mode; True = open trades
                              # via executor (mirrors SR_TRADE_LIVE pattern).
WYCKOFF_RISK_PCT = 0.30       # risk per trade (% equity). Slightly higher than
                              # SR (0.25) — higher-conviction setup. Cap to 0.5.
WYCKOFF_MAGIC_OFFSET = 3000   # FVG=+1000, SR=+2000, Wyckoff=+3000. Non-colliding.
WYCKOFF_SUB_OFFSETS = [3000, 3001]    # sub-offsets for TP1 and TP2 legs
WYCKOFF_MAX_CONCURRENT = 3    # max concurrent Wyckoff trades portfolio-wide
                              # (lower than SR — setup is rare).
WYCKOFF_POST_CLOSE_COOLDOWN_SECS = 1800  # 30min per-symbol cooldown after close
                                         # — prevents re-firing on same TR.

# ── Trading-range validation parameters ───────────────────────────────────
WYCKOFF_RANGE_LOOKBACK_BARS  = 48     # H1 bars (~2 days) used to detect the TR
WYCKOFF_RANGE_MIN_DURATION   = 20     # minimum H1 bars of consolidation required
WYCKOFF_RANGE_MAX_HEIGHT_ATR = 4.0    # TR top-to-bottom <= N*ATR14(H1)
WYCKOFF_RANGE_BODY_ATR_MEDIAN = 0.6   # median H1 body / ATR14 must be <= N

# ── Multi-touch level parameters ──────────────────────────────────────────
WYCKOFF_LEVEL_BAND_ATR = 0.20         # band width around S/R level (ATR mult)
WYCKOFF_MIN_TOUCHES = 2               # minimum prior touches before event

# ── Spring / Upthrust event thresholds ────────────────────────────────────
WYCKOFF_SPRING_WICK_ATR_MIN   = 0.30  # spring wick below support >= N*ATR(M15)
WYCKOFF_UPTHRUST_WICK_ATR_MIN = 0.30  # symmetric upthrust above resistance

# ── Test bar parameters (post-event retest) ───────────────────────────────
WYCKOFF_TEST_REQUIRED        = True   # require a low-volume test bar before entry
WYCKOFF_TEST_LOOKAHEAD_BARS  = 4      # M15 bars to look forward for test bar (1h)
WYCKOFF_TEST_VOL_RATIO_MAX   = 0.70   # test volume <= N * spring volume
WYCKOFF_TEST_HOLD_BUFFER_ATR = 0.10   # test extreme must hold within N*ATR(M15)

# ── HTF trend filter ──────────────────────────────────────────────────────
WYCKOFF_HTF_TREND_FILTER = "BLOCK_AGAINST_DAILY"  # OFF | BLOCK_AGAINST_DAILY | STRICT
                                                  # BLOCK_AGAINST_DAILY: no Spring
                                                  # in D1 downtrend, no Upthrust
                                                  # in D1 uptrend.
WYCKOFF_DAILY_EMA_PERIOD = 50         # D1 EMA used for HTF trend filter
WYCKOFF_ADX_REGIME_MAX   = 30         # H1 ADX14 must be <= N (TR-only regimes)

# ── Cooldown / SL / TP / time-stop ────────────────────────────────────────
WYCKOFF_COOLDOWN_BARS_AFTER = 24      # M15 bars to suppress re-fire (6h)
WYCKOFF_SL_BUFFER_ATR = 0.15          # stop = extreme +/- N*ATR(M15)
WYCKOFF_TP1_R = 1.5                   # first partial @ +1.5R (higher-conviction)
WYCKOFF_TP2_R = 3.0                   # runner @ +3R
WYCKOFF_TIME_STOP_BARS = 16           # close if peak_R < 0.3 after 16 bars (4h)
WYCKOFF_TIME_STOP_PEAK_R = 0.3        # peak-R floor for time-stop

# ── Universe gates (start OFF; populate after A/B validation) ─────────────
WYCKOFF_SYMBOL_WHITELIST = None       # None = all symbols allowed; set() to disable
WYCKOFF_SYMBOL_BLACKLIST = set()      # symbols where Wyckoff is empirically harmful
WYCKOFF_PARAM_OVERRIDES = {}          # per-symbol overrides, e.g.
                                      # {'XAUUSD': {'SPRING_WICK_ATR_MIN': 0.40,
                                      #             'TP1_R': 2.0}}

# ── Executor trail steps (mirrors SR_TRAIL_STEPS shape; routed by magic 3000/3001).
# At +1.5R the broker takes TP1 (close 50%); above that, progressive locks on the
# runner. Same fix pattern as feedback_fvg_trail_integration_20260605 to avoid
# the silent-bypass bug.
WYCKOFF_TRAIL_STEPS = [
    (5.0, "trail", 0.4),
    (3.0, "trail", 0.5),
    (2.0, "lock",  1.0),   # between TP1 and TP2, lock +1R
    (1.5, "lock",  0.5),   # at TP1, lock 0.5R on runner half
    # No lock below 1.5R — let broker TP1 trigger on its own.
]

# ═══ ASAT — Asymmetric Structure-Aware Profit Targets (2026-06-16) ═══
# Replaces the flat per-symbol SUB_TP_R ladder for momentum entries with a
# structure-aware TP1 (fixed 1.5R partial) + TP2 anchored to the next
# significant D1 swing extreme. SL is widened (when needed) to sit just
# beyond the protective M15 structure swing so the stop is INVALIDATION-based.
#   Pure-function module at agent/expert/Asymmetric Structure-Aware Profit Targets (ASAT).py
#   Exported as `compute_asat_levels` from agent.expert.
#   Literature anchors: ICT BSL/SSL liquidity, Wyckoff Phase D/E, SMC
#   "draw on liquidity", Bourgade & Hassani 2009.08821.
ASAT_ENABLED = True                     # Master kill-switch. Default OFF for A/B test against current SUB_TP_R ladder.
ASAT_SYMBOL_WHITELIST = {"XAUUSD"}  # 2026-07-22 audit: DJ30.r DROPPED — architecturally UNREACHABLE (ASAT runs only off the momentum path, which is XAU-only, so DJ30 can never fire; the OOS 339R was false confidence). XAU kept (small-n).
ASAT_TP1_R = 2.0                         # 2026-07-15 tune: 1.5→2.0 (strongest single lever, improved all 5 syms). Close 50% at TP1.
ASAT_TP2_FALLBACK_R = 3.0                # TP2 R-multiple used when no valid D1 swing is found. Fixed by user spec.
ASAT_TP2_MIN_R = 2.0                     # Reject D1 swings closer than this in R-units (redundant with TP1).
ASAT_TP2_MAX_R = 5.0                     # Cap D1 swings beyond this — avoid moonshots that never fill (0/163 outcomes reached >=3R in 60d).
ASAT_FRACTAL_N = 4                       # 2026-07-18 tune-loop R1: 3→4 (9-bar pivot). DJ30.r PF +17%/OOS +18%, 0.96x churn, 3-thirds up; XAU neutral (2.73→2.71). Global scalar.
ASAT_SWING_LOOKBACK_M15 = 60             # Max M15 bars back to scan for protective swing (~15h).
ASAT_D1_FRACTAL_N = 3                    # Symmetric D1 fractal half-window (3 each side = 7-bar pivot, SMC convention).
ASAT_D1_SWING_MEMORY = 20                # Recent D1 swings kept alive as TP2 candidates (same as fvg_strategy.SWING_MEMORY).
ASAT_D1_SWING_MAX_AGE_DAYS = 30          # Reject D1 swings older than this — older liquidity statistically less reliable as a draw.
ASAT_D1_MIN_BARS = 20                    # 2026-07-15 tune: 30->20 so ASAT activates within the 500-bar live window (was inert: needed 720>500). Min closed D1 bars.
ASAT_SL_STRUCT_BUFFER_ATR = 0.25         # ATR-units of buffer below swing-low / above swing-high for structural SL.
ASAT_SL_MAX_ATR = 4.5                    # 2026-07-15 tune: 3.5->4.5. Hard cap on sl_dist in ATR units.
ASAT_SL_MIN_ATR = 0.5                    # Minimum sl_dist in ATR units; floors degenerate stops where structure is too close to entry.
ASAT_HARD_REJECT_ON_OVERSIZED_SL = False # If True, oversized structural SL blocks trade entirely; if False, fall back to existing path.
ASAT_FAIL_OPEN = True                    # On data shortfall / exception, fall back to existing SUB_TP_R path (warn, don't skip on infra).
ASAT_REQUIRE_UNMITIGATED = True          # When True, D1 swings already taken out by subsequent D1 close are excluded — ICT "unmitigated liquidity".
ASAT_LOG_EVERY_DECISION = True           # INFO-log every ASAT computation so the tuning agent has dense data to grade lift.
ASAT_MAGIC_OFFSETS = (9000, 9001)        # 2026-07-17 FIX: was (3000, 3001) which COLLIDED with SMABO_SUB_OFFSETS on shared
                                         # whitelist syms (XAUUSD → identical magics 11100/11101: SMABO BE-mover yanked ASAT
                                         # runner stops, close-detection cross-fired, journal mis-attributed). 9000/9001 is a
                                         # free range: momentum +0/+1/+2, scalp +500/+501, FVG +1000, SR +2000, SMABO +3000,
                                         # FIB50 +4000, SCALPER +5000, TREND +6000, IMR +7000, GOLD_SMC +8000. Must stay in
                                         # sync with brain.py EXPERT open path ([9000, 9001]).

# Per-symbol ASAT trail steps — mirrors FVG_TRAIL_STEPS shape. No lock <1.5R
# (would clip TP1), post-TP1 lock 0.5R, trail at TP2. Wired by executor._apply_trail
# when a position's magic falls in ASAT_MAGIC_OFFSETS range.
ASAT_TRAIL_STEPS = [
    (5.0, "trail", 0.4),
    (3.0, "trail", 0.5),
    (2.0, "lock",  1.0),
    (1.5, "lock",  0.5),
]

# ═══ RANGE-DAY CLASSIFIER (D1 ADX session-stamped regime gate, 2026-06-16) ═
# Wilder DMI/ADX(14) on the D1 resampled frame, stamped once per (symbol,
# UTC trading day). Classifies each day into TREND_DAY (ADX>=25) /
# NEUTRAL_DAY (15<=ADX<25) / RANGE_DAY (ADX<15) and uses +DI/-DI to derive
# directional bias. Pure-function module at
#   agent/expert/RangeDayClassifier (D1 ADX session-stamped regime gate).py
# Exported as `rdc_*` from agent.expert. Wired into brain as Gate 0d
# (before Gate 1 / Session hours) — see project spec.
#
# Policy:
#   TREND_DAY  : block pure mean-revert UNLESS aligned with D1 bias.
#   NEUTRAL_DAY: pass-through.
#   RANGE_DAY  : block pure momentum/breakout UNLESS aligned with D1 bias;
#                when aligned, downsize by RDC_RANGE_DAY_SIZE_MULT (0.5×).
#
# Literature anchors: Wilder (1978) "New Concepts" ADX thresholds; Wyckoff
# Phase A/B/C/D/E (Phase B = range, Phase D = markup); ICT Daily Bias
# doctrine; Connors & Alvarez "High Probability ETF Trading" (mean-revert
# edge concentrates on low-ADX days).
RANGE_DAY_CLASSIFIER_ENABLED = True         # Master kill switch. False = component is a no-op pass-through so it can be A/B tested against current behavior.
RDC_ADX_PERIOD_D1 = 14                       # Wilder DMI/ADX lookback on the D1 resampled frame. Standard Wilder value; reduce to 10 for faster regime detection.
RDC_ADX_TREND_THRESHOLD = 25.0               # D1 ADX >= this = TREND_DAY (only trend/breakout setups allowed). Wilder's classic trend threshold.
RDC_ADX_RANGE_THRESHOLD = 15.0               # D1 ADX < this = RANGE_DAY (only mean-revert setups allowed). Per Connors/Wyckoff Phase B.
RDC_D1_MIN_BARS = 30                         # Minimum closed D1 bars required after resample. Below this, returns None and falls through.
RDC_ALLOW_ALL_ON_UNKNOWN = True              # If data is insufficient or NaN, do not block; warn-only. Respects [[feedback_no_skip_trades]] for ambiguous classifications.
RDC_REQUIRE_DI_ALIGNMENT_ON_RANGE = True     # On RANGE_DAY, momentum/breakout signals must align with D1 dominant DI side; else skip.
RDC_RANGE_DAY_SIZE_MULT = 0.5                # Risk multiplier applied to momentum/breakout signals that pass the DI alignment check on a range day.
RDC_HARD_BLOCK_SYMBOLS = set()               # Symbols where regime violations hard-skip. Empty default = warn-only across the board; opt-in per symbol.
RDC_BYPASS_SYMBOLS = {"XAUUSD", "BTCUSD"}    # Symbols that ignore the classifier entirely (e.g. XAU has different D1 ADX dynamics; preserves surgical-4 momentum edge per [[project_dragon_session_20260605_final]]).
RDC_SIGNAL_CLASS_MAP = {                     # Maps internal signal source -> RDC class. Allows per-strategy policy tuning.
    "momentum":         "MOMENTUM",
    "nr7":              "BREAKOUT",
    "sweep_reclaim":    "MEAN_REVERT",
    "fvg":              "FVG_REVERSAL",
    "pullback":         "PULLBACK",
    "wyckoff_spring":   "MEAN_REVERT",
    "wyckoff_upthrust": "MEAN_REVERT",
}
RDC_LOG_STAMPS_TO_DB = False                 # Persist daily stamps to trade_journal.db.daily_regime so backtest can be replayed deterministically and journal trades can be sliced by D1 regime.

# ═══ ORDER BLOCK DETECTION + RETEST ENTRY (ICT/SMC, 2026-06-16) ═══════════
# ICT-canonical Order Block detector with H1 anchor + M15 retest. Pure-function
# module at agent/expert/Order Block Detection + Retest Entry (ICT/SMC).py,
# exported as `detect_order_block` from agent.expert. Default OFF — observation-
# mode A/B rollout mirrors the SR_TRADE_LIVE pattern (see project_dragon_session_20260605_final).
# Anchor literature: ICT Order Block mentorship, Wyckoff descent / SMC, Bourgade
# & Hassani arXiv 2009.08821 (structural inflection > N-of-K confirmation lag),
# Adam Grimes / Linda Raschke "first touch of fresh value".
OB_ENABLED = False                       # Master toggle. Default OFF — A/B test in observation mode first.
OB_TRADE_LIVE = False                    # When ENABLED + TRADE_LIVE=False, signals are journaled only (no orders). Flip True after ≥30 dry signals show edge.
OB_AS_GATE = True                        # True = confluence gate after Gate 3f (ICT sweep). False = independent book like FVG/SR (own magic, own risk).
OB_SCAN_LOOKBACK = 60                    # Max H1 bars back to search for the most recent valid OB anchor (~2.5 trading days).
OB_IMPULSE_BARS = 3                      # H1 bars after anchor that form the displacement leg (classic ICT 'three drives' window).
OB_IMPULSE_ATR_MULT = 1.5                # Displacement leg size in ATR(14) units. Below = consolidation noise. Matches Crabel's expansion-bar threshold.
OB_SWING_WIN = 10                        # Prior-swing window to confirm Break-of-Structure (BOS). ~1 trading day on H1.
OB_MAX_AGE_BARS = 48                     # OB stales after this many H1 bars even if untouched (~2 trading days).
OB_MITIGATION_ATR = 0.15                 # Tolerance for OB body re-entry by intermediate bars; exceeding = mitigated → skip.
OB_ENTRY_BUFFER_ATR = 0.10               # Wick may overshoot OB boundary by this × ATR(M15) and still count as a retest (covers spread/wick noise).
OB_INVAL_ATR = 0.20                      # M15 close beyond OB body by this × ATR invalidates the block for current cycle.
OB_SL_BUFFER_ATR = 0.15                  # Stop placed this × ATR(M15) beyond OB body extreme (looser than sweep_reclaim's 0.10 — OB body wider than sweep wick).
OB_TP1_R = 1.5                           # First TP in R multiples (close 50%). Matches FVG TP1_R.
OB_TP2_R = 3.0                           # Runner TP. Matches FVG TP2_R.
OB_MAX_RISK_PCT_OF_PRICE = 0.012         # Reject if stop-distance/entry > 1.2% — OB too wide / structural stop unreasonable.
OB_RISK_PCT = 0.25                       # Account risk per OB trade when independent book. Mirrors SR_RISK_PCT — conservative until proven.
OB_MAGIC_OFFSET = 3000                   # Own magic-range base (parent+3000/+3001). FVG=+1000, SR=+2000, ASAT also uses (3000,3001) → revisit collision if OB activates as independent book alongside ASAT.
OB_SUB_OFFSETS = [3000, 3001]            # Sub-magics for TP1/TP2 legs (mirrors FVG_SUB_OFFSETS pattern).
OB_MAX_CONCURRENT = 3                    # Portfolio cap on concurrent OB trades (FVG=7, SR=4 — OB starts tightest until edge proven).
OB_POST_CLOSE_COOLDOWN_SECS = 1800       # 30min cooldown per symbol after an OB trade closes (2× SR's 15min — H1-anchored retest cadence is slower).
OB_WHITELIST = set()                     # Optional symbol whitelist (empty = trade all SYMBOLS). Populate with FVG-style 'proven over 30d' list once data exists.
OB_SYMBOL_BLACKLIST = {"XAUUSD"}         # Defensive blacklist (mirrors SR_SYMBOL_BLACKLIST). XAUUSD seeded due to current freeze / 15.2R drawdown.
OB_PARAM_OVERRIDES = {}                  # Per-symbol overrides for OB_IMPULSE_ATR_MULT, OB_MAX_AGE_BARS, OB_SL_BUFFER_ATR, OB_TP1_R, OB_TP2_R. Same shape as FVG_PARAM_OVERRIDES.
OB_HTF_BIAS_REQUIRED = True              # Require D1 EMA agreement (or H1 EMA200 fallback). Disable only for ranging instruments where HTF bias is noise.
OB_TRAIL_STEPS = [                       # Same shape as FVG_TRAIL_STEPS. No lock below 1.5R to let TP1 trigger cleanly.
    (5.0, "trail", 0.4),
    (3.0, "trail", 0.5),
    (2.0, "lock",  1.0),
    (1.5, "lock",  0.5),
]
OB_TRAIL_PER_SYMBOL = {}                 # Per-symbol trail override (populated by future tuner).
OB_ELC_PER_SYMBOL = {}                   # Per-symbol EarlyLossCut override; same {enabled, r_threshold, bar_close_guard} shape as FVG_ELC_PER_SYMBOL.
OB_TIME_STOP_HOURS = 8.0                 # Close entire position if TP1 not hit within this many hours from fill (H1 anchor → ~8 H1 bars).

# 2026-05-13: vol_min × SL cap override whitelist.
# On a $1.3K account, broker minimum lot × ATR-based SL forces some symbols
# above the MAX_RISK_OVER=3.0 cap (e.g. XAGUSD min lot risks $51 vs $1.32
# intended = 38x). The cap exists to prevent silent risk inflation but it
# hard-blocks proven positive-EV symbols entirely. Symbols in this set get
# a WARN-ONLY override instead of REJECT — they trade at vol_min even when
# forced risk exceeds the cap. List should ONLY contain symbols with:
#   - live EV >= +0.20R (RL-tracked), AND
#   - PF > 1.5 over 50+ recent trades
# Symbols in the universe but NOT in this set still get the hard cap
# (e.g. EURUSD/CHFJPY etc — they shouldn't risk-inflate themselves).
VOL_MIN_WARN_ONLY_SYMBOLS = {
    "XAGUSD",       # live EV +0.36R, BT PF 8.82
    "XAUUSD",       # large lot value; allow despite borderline EV
    "COPPER-Cr",    # live EV +0.60R, BT PF 1.11
    "UKOUSD",       # live EV +0.80R, BT PF 16.85
    "GAS-Cr",       # BT PF 6.93 (cold-start live)
    "NG-Cr",        # BT PF 2.80 historical
    "US2000.r",     # BT PF 1.55, live n=high
    "SPI200.r",     # BT PF ∞ (positive only)
    "JPN225ft",     # BT PF 2.76
    # 2026-05-13 added: crypto blocked by vol_min×SL (BTC $6 vs $1.26
    # intended = 4.8x cap; ETH similar). Single-position trend followers
    # with $1.3K account simply cannot fit smaller. BT shows positive edge.
    "BTCUSD",       # BT PF 2.28 (180d), single-position trend follower
    "ETHUSD",       # BT PF 6.58 (180d), single-position trend follower
    # Indices that may need similar override
    "DJ30.r",       # BT PF 1.84, live earning
    "GER40.r",      # BT PF 2.07
    "HK50.r",       # live EV +0.27R, +0.31R 30d
    "FRA40.r",      # live earning despite BT loss (live +0.40R)
    "SWI20.r",      # BT PF 3.04
}
DAILY_LOSS_LIMIT_PCT = 5.0         # warning at 5% (was 3% — now scaled with 1% per-trade risk)
MAX_POSITIONS = 999                # effectively uncapped — master_brain.py:527 was already warn-only per no-skip rule
DD_REDUCE_THRESHOLD = 8.0          # halve risk at 8% DD (was 6% — scaled with new risk)
DD_PAUSE_THRESHOLD = 12.0          # warn at 12% DD (was 10%)
DD_EMERGENCY_CLOSE = 12.0          # 2026-05-11: raised 8→12 with new risk envelope. Below kill switch but above normal noise.

# ═══ BOT-ONLY DRAWDOWN GATE (2026-07-18) ═══
# The emergency-DD / kill-switch gates historically read RAW MT5 account
# equity (info.equity), which INCLUDES the user's MANUAL trades (magic 0/2024).
# A manual trade that drains equity ratchets nothing and dragged the *bot* into
# a phantom drawdown that froze all trading (the 10,882-loop EmergencyDD
# incident on a flat account). This synthetic curve tracks ONLY the bot's own
# books (baseline + bot-realized + bot-unrealized) so the gate reacts to the
# bot's drawdown, never the manual drain — while a genuine bot blowup still
# stops it (realized loss stays baked into the curve even when flat).
#
# SHIP DARK: when this flag is False the tracker still COMPUTES + shadow-logs
# `bot_dd_pct` every cycle, but every consumer keeps using RAW dd_pct/equity —
# i.e. behaviour is byte-for-byte identical to today. Flip to True only after
# shadow logs confirm bot_dd stays flat on manual drains AND moves on real bot
# losses. Fail-safe: any read error / stale positions / accumulator gap →
# `healthy=False` → consumers fall back to RAW dd_pct (the over-stops side).
BOT_EQUITY_GATE_ENABLED     = True    # 2026-07-18 ARMED: shadow logs confirmed healthy=True + bot_dd tracks distinct-from-raw over a full session (mkt closed = zero-trade-risk flip). 12% emergency + 3% daily now on bot curve; 40%/50% hard kills stay RAW-armed.
BOT_EQUITY_OVERLAP_SECS     = 120.0   # INERT since 2026-07-18 journal-SUM graft (no watermark/re-scan seam)
BOT_EQUITY_ACCUM_THROTTLE_S = 5.0     # min seconds between local journal-SUM realized reads (off-tick)
BOT_EQUITY_PERSIST_THROTTLE_S = 60.0  # min seconds between durable-KV state writes
BOT_EQUITY_RECENT_TICKETS_MAX = 2000  # INERT since 2026-07-18 journal-SUM graft (no dedup ring; SUM is authoritative)
BOT_EQUITY_POS_MAX_AGE_S    = 3.0     # reused positions snapshot older than this → unhealthy → RAW fallback

# ═══ HARD KILL SWITCHES (cannot be bypassed) ═══
# 2026-05-11: bumped daily 2→4, weekly 5→10 to accommodate broker-min-lot
# trades where actual risk per trade can hit ~2.1% (e.g. GER40.r on $1K).
# Worst-case: 2 SL hits same day → halt; 4-5 SL hits same week → halt.
DAILY_HARD_STOP_PCT = 40.0         # 2026-05-13 user override: was 4.0 (tripped today,
                                   # force-closed 8 positions = -$101 single event).
                                   # 40% effectively disables — bot can't realistically
                                   # lose 40% in one day with 1% risk × 25 syms cap.
WEEKLY_HARD_STOP_PCT = 50.0        # 2026-05-13 user override: was 10.0. Matched to daily.

# ═══ RE-ENTRY COOLDOWNS (one source of truth) ═══
# 2026-05-11: asymmetric directional cooldown. Wins get a SHORT same-direction-
# only window (don't chase extended moves; opposite-direction allowed for
# mean-reversion). Losses get a LONG both-directions block (avoid revenge).
# Break-even/unknown closes default to symmetric loss cooldown.
COOLDOWN_WIN_SECS          = 1800  # 2026-05-29: 15→30min — don't re-chase same dir right after a win. Same-direction only.
# 2026-06-03 CTO audit (A6): 45-60min post-loss-cooldown window had avg
# -$0.95 / -0.73R PnL (worst entry window across the 180d journal). The 45min
# default lapses RIGHT INTO that dead zone. 75min skips past it.
COOLDOWN_LOSS_SECS         = 4500  # 75min — SL hit / closed at loss. Both directions.
COOLDOWN_BROKER_CLOSE_SECS = 4500  # 75min — default if win/loss can't be determined
COOLDOWN_SL_HIT_SECS       = 4500  # 75min — loss-tagged exit (legacy alias for COOLDOWN_LOSS_SECS)
COOLDOWN_SCALP_CLOSE_SECS  = 1800  # 30min — scalp closed
EXECUTOR_MIN_REENTRY_SECS  = 60    # belt-and-braces hard floor: executor refuses re-open within Ns of any close
# 2026-05-29 cooldown redesign (single source of truth = brain._arm_cooldown):
COOLDOWN_LOSS_SMALL_MULT   = 0.67  # |r|<0.5R  → ~30min (barely moved, fast retry ok)
COOLDOWN_LOSS_BIG_MULT     = 2.0   # |r|>=1.0R → ~90min (market rejected thesis hard)
STREAK_COOLDOWN_MULT       = {2: 2.0, 3: 4.0}  # consec-loss escalation multiplier on loss cooldown
POST_BIG_WIN_SECS          = 3600  # 60min same-dir — don't chase an exhausted move just caught
BIG_WIN_R_TRIGGER          = 3.0   # r_multiple / peak_r threshold for POST_BIG_WIN
ATTEMPT_BACKOFF_BASE_SECS  = 1800  # per-(symbol,direction) anti-spam base (30min)
ATTEMPT_BACKOFF_CAP_SECS   = 14400 # 4h cap — kills the "re-fire same losing setup" cascade

# ═══ PER-SYMBOL RISK CAP (override MAX_RISK for specific symbols) ═══
SYMBOL_RISK_CAP: Dict[str, float] = {
    "BTCUSD": 0.4,                 # HALVED: high variance asset
    "USDCAD": 4.0,                 # 5x Forex
    "EURJPY": 4.0,                 # 5x Forex
    "EURUSD": 4.0,                 # 5x Forex
    "USDJPY": 4.0,                 # 5x Forex
    "GBPUSD": 4.0,                 # 5x Forex
    "GBPJPY": 4.0,                 # 5x Forex
}

# 2026-05-17: per-(symbol, regime) risk cap. Overlays SYMBOL_RISK_CAP
# when (sym, regime) cell set. Schema {sym: {regime: float}}.
SYMBOL_RISK_CAP_REGIME: Dict[str, Dict[str, float]] = {}

# ═══ TICK STREAMING ═══
TICK_INTERVAL_MS = 500             # poll ticks every 500ms
CANDLE_WINDOW = 500                # keep last 500 candles per TF
TIMEFRAMES = [1, 5, 15, 60]       # M1, M5, M15, H1

# ═══ ML MODEL ═══
MODEL_DIR = Path(__file__).parent / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CONFIDENCE_THRESHOLD = 0.60        # min probability to trade
PREDICTION_HORIZON_TF = 15         # M15 forward return for labels

# ═══ TRADING MODE ═══
TRADING_MODE = "hybrid"            # "swing", "scalp", or "hybrid" (both)

# ═══ TRAILING SL — DEFAULT AGGRESSIVE profile ═══
# Lock amounts at each R-threshold are env-overridable for sweeping. Values
# below are post-tune defaults; sweep tooling: scripts/sweep_trail_lock.py.
def _envfloat(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default

_TRAIL_LOCK_AT_15R = _envfloat("DRAGON_TRAIL_LOCK_AT_15R", 1.0)  # 2026-06-19: 0.7→1.0 — give runners breathing room
_TRAIL_LOCK_AT_10R = _envfloat("DRAGON_TRAIL_LOCK_AT_10R", 0.6)  # 2026-06-19: 0.4→0.6 — let 1R wins extend
_TRAIL_LOCK_AT_07R = _envfloat("DRAGON_TRAIL_LOCK_AT_07R", 0.4)  # 2026-06-19: 0.2→0.4 — stop clipping winners in 0.5-1R band

# 2026-05-12 (CONSERVATIVE REVERSAL):
# User feedback after live bleed: trades reaching profit then reversing past
# entry, hitting SL with slippage = -1.5R losses while wins capped at +0.6R.
# Backtest's "let winners run" thesis doesn't survive real broker friction
# on small account. NEW strategy: AGGRESSIVE PROFIT LOCK + EARLY LOSS CUT.
#
# Locks profit at EVERY R-level above 0.3R. Combined with EARLY_EXIT_ENABLED
# (closes losing trades before full SL slippage) and PEAK_GIVEBACK_ENABLED
# (closes if profit retraces 50% from peak).
#
# Tradeoff: backtest will show LOWER PnL (because backtest doesn't model
# slippage realistically). Live should show LESS bleeding because actual
# broker friction is bounded by aggressive book-profit/cut-loss.
TRAIL_STEPS = [
    (8.0, "trail", 0.3),
    (4.0, "trail", 0.5),
    (2.0, "trail", 0.6),
    (1.5, "lock",  _TRAIL_LOCK_AT_15R),
    (1.0, "lock",  _TRAIL_LOCK_AT_10R),
    (0.7, "lock",  _TRAIL_LOCK_AT_07R),
    # 2026-06-05: removed (0.5, "lock", 0.15) and (0.3, "be", 0.0).
    # Research: Zarattini Beat-the-Market + Concretum + Davey 567k-backtest agree —
    # moving SL to BE before structure shift clips winners in entry-bar noise band.
    # Bleed evidence: 51 live trades 14d in -0.3R to -1R bucket = -$205.
]

# ═══ 2026-05-13 — PER-REGIME TRAIL PROFILES ═══
# Same symbol behaves differently in trending vs ranging vs volatile vs low_vol.
# Per-regime trail profiles let trends run while locking range chop fast.
# Brain pushes current regime to executor; executor picks the matching profile.
# Falls back to SYMBOL_TRAIL_OVERRIDE then TRAIL_STEPS if regime not set.
#
# Format: (R_threshold, step_type, param)
#   step_type "trail" → SL trails param×SL_dist behind price once peak >= R
#   step_type "lock"  → SL locks at entry + param×SL_dist (param > 0 = profit lock)
#   step_type "be"    → SL moves to entry (BE) once peak >= R
REGIME_TRAIL_DEFAULTS: Dict[str, list] = {
    # TRENDING — let winners run, but lock aggressively the first 1R
    "trending": [
        (15.0, "trail", 0.3),   # mega run: trail 0.3R behind
        (8.0,  "trail", 0.4),
        (4.0,  "trail", 0.5),
        (2.0,  "lock",  1.0),   # at 2R peak, lock 1R (50% capture)
        (1.0,  "lock",  0.5),   # at 1R peak, lock 0.5R
        (0.3,  "be",    0.0),   # BE at 0.3R (very aggressive)
    ],
    # RANGING — capture quick, no runner expected
    "ranging": [
        (4.0,  "trail", 0.5),
        (2.0,  "lock",  1.2),   # at 2R, lock 1.2R (60% capture — choppy may reverse)
        (1.0,  "lock",  0.6),
        (0.3,  "be",    0.0),
    ],
    # VOLATILE — wider trails (chop) but lock fast once locked
    "volatile": [
        (6.0,  "trail", 0.6),
        (3.0,  "trail", 0.8),
        (1.5,  "lock",  0.7),
        (0.7,  "lock",  0.3),
        (0.3,  "be",    0.0),
    ],
    # LOW_VOL — slow moves, tight trail
    "low_vol": [
        (3.0,  "trail", 0.4),
        (1.5,  "lock",  0.8),
        (1.0,  "lock",  0.5),
        (0.5,  "lock",  0.2),
        (0.3,  "be",    0.0),
    ],
}

# Per-(symbol, regime) override (filled by tuner). Format:
#   {"DJ30.r": {"trending": [(...), ...], "ranging": [...]}, ...}
# Falls back to REGIME_TRAIL_DEFAULTS[regime] if symbol's regime cell empty.
#
# 2026-05-13 regime-trail tune: ULTRA_TIGHT (BE@0.2R, lock 1.5R at 2R peak)
# dominated the sweep for 4 symbols. Apply to all regimes for those — they
# benefit from aggressive locking regardless of market condition.
_ULTRA_TIGHT = [
    (2.0, "lock", 1.5),   # at 2R peak, lock 1.5R (75% capture)
    (1.0, "lock", 0.7),   # at 1R peak, lock 0.7R (70% capture)
    (0.5, "lock", 0.2),   # at 0.5R, lock 0.2R (40% capture)
    (0.2, "be",   0.0),   # at 0.2R peak, move to BE
]

# 2026-05-13: COMMODITIES/METALS aggressive trail — root cause analysis:
# These symbols have wide ATR-based SL (often 3.0 × ATR). Dollar profit
# accumulates fast (large lots × big tick_value) but R-multiple stays
# LOW because SL is so wide. Example: GAS-Cr at +$22 was only 0.108R
# → no trail fired (BE was at 0.5R = +$100 needed).
# Aggressive trail captures the dollar profit at low R-fractions.
# 2026-05-13 v2: even tighter. GAS-Cr at +$38 was only 0.185R but trail
# was locked at 0.05R = only $5 captured. Adding finer-grained steps so
# every increment of profit gets locked progressively.
_COMMODITY_AGGRESSIVE = [
    (2.0, "lock", 1.5),    # 2R peak → lock 1.5R (75% capture)
    (1.0, "lock", 0.7),    # 1R peak → lock 0.7R (70% capture)
    (0.7, "lock", 0.5),    # 0.7R peak → lock 0.5R (NEW)
    (0.5, "lock", 0.35),   # 0.5R peak → lock 0.35R (70% capture)
    (0.35, "lock", 0.25),  # NEW
    (0.25, "lock", 0.17),  # NEW — captures progressive profit
    (0.18, "lock", 0.12),  # NEW
    (0.12, "lock", 0.07),  # NEW — locks $4-5 of $25-30 profit on commodities
    (0.08, "lock", 0.03),  # NEW — locks even modest moves
    (0.05, "be",   0.0),   # BE at 0.05R (super aggressive — capture floor)
]

SYMBOL_REGIME_TRAIL_OVERRIDE: Dict[str, Dict[str, list]] = {
    # +$7,534 / 180d PF 8.39 with ULTRA_TIGHT
    "UKOUSD":    {r: _ULTRA_TIGHT for r in ("trending", "ranging", "volatile", "low_vol")},
    # +$1,180 / 180d PF 3.96
    "EURAUD":    {r: _ULTRA_TIGHT for r in ("trending", "ranging", "volatile", "low_vol")},
    # +$548 / 180d PF 2.60
    "US2000.r":  {r: _ULTRA_TIGHT for r in ("trending", "ranging", "volatile", "low_vol")},
    # +$343 / 180d PF 2.92
    "SWI20.r":   {r: _ULTRA_TIGHT for r in ("trending", "ranging", "volatile", "low_vol")},

    # 2026-05-13 commodities/metals — wide SL = need low-R trail
    # All 4 regimes use the aggressive profile because the SL math
    # is the same regardless of regime.
    "GAS-Cr":    {r: _COMMODITY_AGGRESSIVE for r in ("trending", "ranging", "volatile", "low_vol")},
    "NG-Cr":     {r: _COMMODITY_AGGRESSIVE for r in ("trending", "ranging", "volatile", "low_vol")},
    "COPPER-Cr": {r: _COMMODITY_AGGRESSIVE for r in ("trending", "ranging", "volatile", "low_vol")},
    # 2026-05-26 audit fix: _COMMODITY_AGGRESSIVE has BE@0.05R = too tight for
    # XAUUSD's avg_peak 1.67R / avg_giveback 1.38R (79% peak-give). Live evidence
    # is wins clipped at +0.35R while losses run full -1R. _RUNNER_NO_BE keeps
    # tight trail above 1R but removes the BE rug-pull that kills sub-1R wins.
    "XAUUSD":    {r: [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5), (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)] for r in ("trending", "ranging", "volatile", "low_vol")},
    "XAGUSD":    {r: _COMMODITY_AGGRESSIVE for r in ("trending", "ranging", "volatile", "low_vol")},
    "BTCUSD":    {r: _COMMODITY_AGGRESSIVE for r in ("trending", "ranging", "volatile", "low_vol")},
    "ETHUSD":    {r: _COMMODITY_AGGRESSIVE for r in ("trending", "ranging", "volatile", "low_vol")},
}

# ═══ TRAILING SL — AGGRESSIVE DENSE LOCKS (every 0.1-0.2R, BE early) ═══
# 2026-05-12 (CONSERVATIVE): per-symbol overrides also restored to AGGRESSIVE
# profit lock. Live evidence shows wins capped at 0.6R, losses at -1.7R due
# to broker friction. AGGRESSIVE early locks + EARLY_EXIT_ENABLED + PEAK_GIVEBACK
# is the live-tested approach. Backtest can't model this — accept it.
SYMBOL_TRAIL_OVERRIDE: Dict[str, list] = {
    "AUDUSD": [
        (8.0, "trail", 0.3),
        (4.0, "trail", 0.5),
        (2.0, "trail", 0.8),
        (1.5, "lock", 0.7),
        (1.0, "lock", 0.4),
        (0.7, "lock", 0.2),
        (0.5, "be",   0.0),
    ],
    # 2026-05-12: ALL per-symbol trails stripped of sub-1R BE/lock steps.
    # Same evidence as default: early protection was killing wins. AUDUSD
    # exception kept above (walk-forward proved AUDUSD needs BE).
    # 2026-05-14: XAU wins were getting BE-stopped at 0.3R → tiny profits ($0.02-$1.30).
    # Net last 25 trades = -$49.81 despite 60% WR. Letting winners breathe by moving
    # BE to 1.5R. PEAK_GIVEBACK (50% retrace from peak ≥ 0.7R) still protects gains.
    "XAUUSD": [
        (8.0, "trail", 0.3),
        (4.0, "trail", 0.5),
        (2.5, "trail", 0.7),
        (2.0, "lock",  0.7),
        (1.5, "be",    0.0),
    ],
    "XAGUSD": [
        (3.0, "trail", 0.4), (2.5, "lock", 1.5), (2.0, "lock", 1.2),
        (1.5, "lock", 1.0), (1.0, "lock", 0.7),
        (0.7, "lock", 0.4), (0.4, "be", 0.0),
    ],
    # 2026-06-02: crypto BE removed. (0.4, "be", 0.0) caused trail-too-tight
    # whipsaw on BTC: bot caught BTC sell-off 72690→71017 (-2.3%) but trail
    # locked BE at +0.4R, normal vol wick stopped each leg flat. Captured
    # $0.57 across 3 SHORTs on a $1700 favorable move. Same pattern ate ETH
    # twice (-$7.35). Earliest protection is now (0.7, "lock", 0.15) — gives
    # the trade room to develop before any breakeven shift.
    "BTCUSD": [
        (6.0, "trail", 0.3), (4.0, "trail", 0.5), (2.5, "trail", 0.8),
        (1.5, "lock", 0.5), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15),
    ],
    "ETHUSD": [
        (6.0, "trail", 0.3), (4.0, "trail", 0.5), (2.5, "trail", 0.8),
        (1.5, "lock", 0.5), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15),
    ],
    "NAS100.r": [
        (6.0, "trail", 0.3), (4.0, "trail", 0.5), (2.5, "trail", 0.8),
        (1.5, "lock", 0.5), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15), (0.4, "be", 0.0),
    ],
    "JPN225ft": [
        (4.0, "trail", 0.3), (2.5, "trail", 0.5), (1.5, "trail", 0.8),
        (1.2, "lock", 0.8), (1.0, "lock", 0.6),
        (0.8, "lock", 0.4), (0.5, "lock", 0.2), (0.3, "be", 0.0),
    ],
    "SP500.r": [
        (4.0, "trail", 0.3), (2.5, "trail", 0.5), (1.5, "trail", 0.8),
        (1.0, "lock", 0.5), (0.7, "lock", 0.3),
        (0.4, "be", 0.0),
    ],
    "GER40.r": [
        (6.0, "trail", 0.3), (4.0, "trail", 0.5), (2.5, "trail", 0.8),
        (1.5, "lock", 0.5), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15), (0.4, "be", 0.0),
    ],
    "USDCAD": [
        (6.0, "trail", 0.3), (4.0, "trail", 0.5), (2.5, "trail", 0.8),
        (1.5, "lock", 0.5), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15), (0.4, "be", 0.0),
    ],
    "EURJPY": [
        (3.0, "trail", 0.4), (2.0, "trail", 0.6), (1.5, "lock", 0.5),
        (1.0, "lock", 0.4), (0.7, "lock", 0.2), (0.3, "be", 0.0),
    ],
    "EURUSD": [
        (5.0, "trail", 0.3), (3.0, "trail", 0.5), (2.0, "trail", 0.8),
        (1.5, "lock", 0.7), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15), (0.4, "be", 0.0),
    ],
    "USDJPY": [
        (4.0, "trail", 0.3), (2.5, "trail", 0.5), (2.0, "trail", 0.7),
        (1.5, "lock", 0.8), (1.2, "lock", 0.7), (1.0, "lock", 0.5),
        (0.7, "lock", 0.3), (0.5, "be", 0.0),
    ],
    "GBPUSD": [
        (5.0, "trail", 0.3), (3.0, "trail", 0.5), (2.0, "trail", 0.8),
        (1.5, "lock", 0.7), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15), (0.4, "be", 0.0),
    ],
    "GBPJPY": [
        (6.0, "trail", 0.3), (4.0, "trail", 0.5), (2.5, "trail", 0.8),
        (1.5, "lock", 0.5), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15), (0.4, "be", 0.0),
    ],
}

# ═══ TRAILING SL — Sub2 RUNNER (wider for big moves, but still locks profit) ═══
# Sub2 runner is the EXTENSION sub — meant to ride biggest moves. Keeping
# the high-R steps; sub-1R steps would not fire on runners that hit 2R+
# anyway, but kept for safety against deep retraces below 1R after 2R hit.
SUB2_TRAIL_STEPS = [
    (10.0, "trail", 0.3),
    (8.0, "trail", 0.4),
    (6.0, "trail", 0.5),
    (4.0, "trail", 0.7),
    (2.0, "trail", 0.8),
    (1.5, "lock",  0.7),
    (1.0, "lock",  0.4),
]

# ═══ SCALP CONFIG ═══
# DISABLED 2026-05-02 — 7d audit: scalp -$51 on 91 trades (45% WR, -$1.22/trade,
# losing 3x faster than swing). h=14-16 UTC bled -$66 net (NY-open chop).
# Scalp also ignored SYMBOL_ATR_SL_OVERRIDE (used hardcoded 1.5x globally
# instead of per-symbol values). Will revisit only after backtest validates a
# scalp config that's PF >= 1.5 in current regime.
SCALP_ENABLED = False
SCALP_RISK_PCT = 0.2              # 0.2% equity per scalp trade (was 0.5)
SCALP_ATR_MULT = 1.5              # SL = 1.5x ATR(M5)
SCALP_MAGIC_OFFSET = 500          # scalp magic = base magic + 500 (was 100, collided with EURJPY 8210)
SCALP_SESSION_START = 13           # scalp session 13:00 UTC
SCALP_SESSION_END = 17             # scalp session 17:00 UTC
SCALP_MAX_PER_SESSION = 2          # max 2 scalps per symbol per session

# ═══ SCALP TRAILING SL (tight profile) ═══
SCALP_TRAIL_STEPS = [
    # Aggressive scalp trail — lock fast, BE early
    (2.0, "trail", 0.5),
    (1.5, "lock",  0.8),
    (1.0, "lock",  0.6),
    (0.8, "lock",  0.5),
    (0.6, "lock",  0.3),
    (0.4, "lock",  0.2),
    (0.3, "lock",  0.1),
    (0.15, "be",   0.0),            # BE at 0.15R (was 0.7R — way too late for scalps)
]

# ═══ SESSION FILTER ═══
SESSION_START_UTC = 6              # non-crypto default: 06:00 UTC
SESSION_END_UTC = 22               # non-crypto default: 22:00 UTC

# Per-symbol session overrides (start_utc, end_utc)
# JPN225ft: Asian session starts at 00:00 UTC — default 06:00 misses best hours
# XAUUSD/XAGUSD: London+NY overlap 06-22 is optimal, keep default
# 2026-05-02 — narrower windows for indices to stop [10021] No prices errors:
#   GER40.r (DAX cash):    07-21 UTC (Frankfurt 09:00-21:00 local)
#   NAS100.r/SP500.r:      13-21 UTC (NY cash 09:30-16:00 EST + 1h pre/post)
SYMBOL_SESSION_OVERRIDE: Dict[str, Tuple[int, int]] = {
    "JPN225ft": (0, 22),           # include Asian session (00-22 UTC)
    "GER40.r":  (7, 21),           # Frankfurt cash hours — fixes [10021] outside
    "NAS100.r": (13, 21),          # NY cash session (incl. pre-market 1h)
    "SP500.r":  (13, 21),          # NY cash session (incl. pre-market 1h)
}

# ═══ MIN_EDGE friction thresholds (2026-05-17) ═══
# Pre-trade structural cost gate in brain.py: rejects a signal when
#   (spread * 2.5) / SL_distance > threshold.
# Default threshold of 25% blocks A-grade signals on tight-spread mean-reverters
# (observed: USDJPY scoring 8.2 LONG was being blocked every cycle — 6.2K hits
# in current log). High-conviction signals (raw_score >= MIN_EDGE_HIGH_CONV_SCORE)
# use a relaxed 37.5% threshold (1.5x); high score statistically overrides
# slightly worse cost ratios. A+ quality (>=75%) still bypasses entirely upstream.
MIN_EDGE_FRICTION_PCT = 0.25            # default friction-vs-SL cap
MIN_EDGE_FRICTION_PCT_HIGH_CONV = 0.375 # 1.5x relaxation for A-grade signals
MIN_EDGE_HIGH_CONV_SCORE = 7.0          # raw_score threshold for high-conviction tier
# 2026-05-26 audit fix: per-symbol friction cap. Proven-edge indices have
# wider spread/ATR ratios that the global 25% cap rejects (DJ30 logs show
# friction 57% repeated, US2000 79%). These caps unblock the proven edge
# without lifting the global default. brain.py & v5_backtest.py both read.
MIN_EDGE_FRICTION_PCT_PER_SYMBOL: Dict[str, float] = {
    # 2026-05-29 REVERTED the index relaxations — live evidence: SP500 0% WR/5,
    # US2000 0% WR/5 after relaxation. The friction cap was correctly blocking
    # these. Audit agents warned: relaxing friction lets through trades BT
    # blocks for good reason. Keep only USOUSD (commodity spread is structural
    # and it was a live WINNER +$2.55/67% WR).
    "USOUSD":    0.40,   # commodity broker spread — live winner under relaxation
    # 2026-06-04 CTO audit B12: CHFJPY (88% WR, +$2.42 net) blocked 24× and
    # DJ30.r (40% WR but +0.27R avg, +$28 net 3d) blocked 7× under default 25%
    # cap. Both are proven live winners — friction is the limiter, not the EV.
    "CHFJPY":    0.40,
    "DJ30.r":    0.40,
}

# ═══ ATR SL ═══
ATR_SL_MULTIPLIER = 1.5           # SL = 1.5x ATR default (was 3.0 — KEY FIX for PF)

# Per-symbol ATR SL multiplier overrides (grid search + baseline backtest)
# Normalized 2026-05-01 from over-tight 0.2-0.4x range. The previous values
# meant SLs were 0.3x ATR for indices = 2.27 points on SP500.r (vs typical
# 7+ point swings) → every trade got SL'd by normal noise within minutes.
# User saw SP500.r LONG opened+closed in 5 min ("bleeding so small SL").
# USDCAD was the opposite extreme (5.5x ATR — wildly loose), letting losses
# run far past where they should have stopped. New values target 1.5-2.0x
# ATR per industry-standard practice.
SYMBOL_ATR_SL_OVERRIDE: Dict[str, float] = {
    "XAUUSD":   1.5,   # was 0.3 — gold needs room
    "XAGUSD":   1.5,   # was 1.2 — slight bump for noise tolerance
    "BTCUSD":   1.5,   # was 1.0
    "ETHUSD":   1.5,
    "NAS100.r": 2.0,   # was 0.3 — indices swing 7+ pts/H1
    "JPN225ft": 2.0,   # was 0.3
    "SP500.r":  2.0,   # was 0.3 — 2.0x ATR ≈ 15pts (was 2.27 = stopped on noise)
    "GER40.r":  2.0,   # was 0.3
    "USDCAD":   1.5,   # was 5.5 — wildly loose, contributed to drifted half
    "EURJPY":   1.5,   # was 0.2
    "EURUSD":   1.5,   # was 0.2
    "USDJPY":   1.5,   # was 0.4
    "GBPUSD":   1.5,   # was 0.2
    "GBPJPY":   2.0,   # unchanged — was working
}

# 2026-05-17: per-(symbol, regime) SL override. Overlays on top of
# SYMBOL_ATR_SL_OVERRIDE when (symbol, regime) cell is set.
# Schema: {symbol: {regime: float}} where regime in {trending, ranging,
# volatile, low_vol}. Cell-miss → falls back to per-symbol → global.
# Populated by auto_tuned.SL_OVERRIDE_REGIME_AUTO via the merge block below.
SYMBOL_ATR_SL_OVERRIDE_REGIME: Dict[str, Dict[str, float]] = {}

# ═══ SMART ENTRY — Per-Symbol Intelligence Mode ═══
# Backtested: cherry-pick best strategy per symbol (avg PF 2.28 vs 2.19 base)
SMART_ENTRY_MODE: Dict[str, Dict[str, bool]] = {
    # adaptive_trail: scale trail by current ATR vs 50-bar avg ATR
    # fresh_momentum: require MACD acceleration + RSI not exhausted
    "XAUUSD":   {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 2.18)
    "XAGUSD":   {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 2.44)
    "BTCUSD":   {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 5.30)
    "NAS100.r": {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 1.75)
    "USDJPY":   {"adaptive_trail": False, "fresh_momentum": False},  # BASE best (PF 1.79)
    "JPN225ft": {"adaptive_trail": True,  "fresh_momentum": True},   # BOTH: PF 2.73→2.98
    "USDCAD":   {"adaptive_trail": False, "fresh_momentum": True},   # FR.MOM: PF 1.47→1.52
}

# ═══ DASHBOARD ═══
DASHBOARD_PORT = 8888
STARTING_BALANCE = 2500.0

# ═══ SQLITE ═══
DB_PATH = Path(__file__).parent / "data" / "beast.db"

# ═══ DRAGON-SPECIFIC CONSTANTS ═══
DRAGON_MIN_SCORE_BASELINE = 7.0    # minimum score for H1 swing entry

# ═══ M15 PRIMARY SCORING (lower thresholds — M15 has 4x more signals) ═══
DRAGON_M15_MIN_SCORE_BASELINE = 6.0
DRAGON_M15_SYMBOL_MIN_SCORE: Dict[str, Dict[str, float]] = {
    # NOTE 2026-06-03: this dict (DRAGON_M15_SYMBOL_MIN_SCORE) is currently
    # DEAD CONFIG — no Python code reads it. Live gate uses SIGNAL_QUALITY_SYMBOL
    # (0-100 scale, set at config.py:1166+). Keeping the dict for future M15
    # raw-score gate wiring but the CTO score tweaks live in SIGNAL_QUALITY_SYMBOL.
    "XAUUSD":   {"trending": 5.5, "ranging": 6.5, "volatile": 6.0, "low_vol": 6.0},
    "XAGUSD":   {"trending": 5.5, "ranging": 6.0, "volatile": 6.5, "low_vol": 6.5},
    "BTCUSD":   {"trending": 6.0, "ranging": 7.0, "volatile": 6.5, "low_vol": 6.5},  # raised +1.0: anti-churn
    "NAS100.r": {"trending": 6.0, "ranging": 6.0, "volatile": 6.0, "low_vol": 6.0},
    "JPN225ft": {"trending": 6.0, "ranging": 6.0, "volatile": 6.5, "low_vol": 6.5},
    "USDJPY":   {"trending": 6.0, "ranging": 6.5, "volatile": 6.5, "low_vol": 6.5},
    "USDCAD":   {"trending": 6.0, "ranging": 6.0, "volatile": 6.0, "low_vol": 6.0},
}

# ═══ MEAN-REVERSION STRATEGY (fires when momentum is flat) ═══
# DISABLED 2026-05-02 — config flag was True but agent/brain.py never instantiated
# or fired MR strategy (dead code per audit). MR_SL_ATR_MULT=1.0 was also too
# tight under current regime (would have replicated the SP500.r 0.3-ATR bleed).
# Kept config block for future re-enable, but explicitly flagged off.
MR_ENABLED = False
MR_MIN_SCORE = 3.0              # Lower bar: BB touch(1) + RSI(1) + EMA dist(1) = 3
MR_RISK_DISCOUNT = 0.7          # 70% of momentum risk (less conviction)
MR_SL_ATR_MULT = 1.5            # was 1.0 — bumped to match new momentum SL floor
MR_TRAIL_STEPS = [
    (1.5, "trail", 0.5),        # Trail at 1.5R (quick exit for reversion)
    (1.0, "lock", 0.3),         # Lock 0.3R at 1R
    (0.5, "be", 0.0),           # BE at 0.5R
]
# Only fire MR in ranging/low_vol regimes (not trending/volatile)
MR_ALLOWED_REGIMES = {"ranging", "low_vol"}

# ═══ INDUSTRY ENTRY GATES (per-symbol, from V2 backtest validation) ═══
INDUSTRY_GATES_ENABLED: Dict[str, bool] = {
    "XAUUSD":   False,  # mega: PF=1.66 (gates OFF better in walk-forward)
    "XAGUSD":   True,   # mega: PF=2.33 WF=4.76
    "BTCUSD":   False,  # mega: PF=4.05 (gates hurt crypto)
    "NAS100.r": False,  # gates kill NAS (PF 1.01→1.75 OFF)
    "JPN225ft": True,   # mega: PF=2.42 WF=3.37
    "USDJPY":   True,   # mega: PF=2.39 WF=2.22
    "USDCAD":   False,  # gates hurt CAD (PF 1.23→1.29 OFF)
}

# ═══ PRIMARY TIMEFRAME (M15 for entries, H1 for bias) ═══
PRIMARY_TF = 15                 # M15 = primary signal timeframe
BIAS_TF = 60                    # H1 = directional bias only
EVAL_ON_CANDLE_CLOSE = True     # 2026-05-29: flipped True. Per-tick scoring made
                                # the bot fire at the intrabar SCORE PEAK = local
                                # price extreme (lagging-indicator scoring peaks
                                # when price peaks). Live forensics: low_vol score
                                # 9-10 = 0% WR / -$27; the EarlyLossCut_T1- cluster
                                # (-$71.74) is this exact enter-at-extreme pattern.
                                # Score ONCE per closed M15 bar. Memory-compliant:
                                # changes sampling cadence, does NOT block signals.
                                # (Prior per-tick rationale below — kept for context.)
                                # - RL adaptive bonus updates
                                # - portfolio risk shifts
                                # Cost: ~10× more CPU on score path. Acceptable
                                # given the bot was missing real-time changes
                                # that mattered for entry decisions.

# Per-symbol regime MIN_SCORE overrides
# 2026-05-12 autonomous tune: TIGHTEN +1.0 for marginal symbols (PF 0.8-1.0
# in current config), BOOST -0.3 for strong (PF >= 2.0 with 200+ trades).
DRAGON_SYMBOL_MIN_SCORE: Dict[str, Dict[str, float]] = {
    # KEPT (already in original spec)
    "XAUUSD":   {"trending": 6.0, "ranging": 7.0, "volatile": 6.5, "low_vol": 6.5},
    "XAGUSD":   {"trending": 7.0, "ranging": 7.5, "volatile": 7.2, "low_vol": 7.2},
    "JPN225ft": {"trending": 7.0, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},
    "USDCAD":   {"trending": 7.0, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},
    # BOOSTED (lower MIN_SCORE -0.3, more trades on proven winners)
    "BTCUSD":   {"trending": 6.7, "ranging": 7.7, "volatile": 7.2, "low_vol": 7.2},  # was 7.0 (PF 2.28)
    "ETHUSD":   {"trending": 5.7, "ranging": 6.7, "volatile": 6.2, "low_vol": 6.2},  # NEW (PF 6.58 — let it trade)
    "EURAUD":   {"trending": 5.7, "ranging": 6.7, "volatile": 6.2, "low_vol": 6.2},  # NEW (PF 2.69)
    # TIGHTENED +1.0 (marginal performers — only highest-conviction)
    # HK50.r/FRA40.r removed from TIGHTEN 2026-05-12: live 30d +0.31R/+0.40R
    # (BT loss was on backtest data; live evidence supersedes — keep normal MIN_SCORE)
    "COPPER-Cr":{"trending": 7.0, "ranging": 8.0, "volatile": 7.5, "low_vol": 7.5},
    "SPI200.r": {"trending": 6.3, "ranging": 7.3, "volatile": 6.8, "low_vol": 6.8},  # +0.3
    "CADJPY":   {"trending": 6.3, "ranging": 7.3, "volatile": 6.8, "low_vol": 6.8},  # +0.3
    "GBPJPY":   {"trending": 6.3, "ranging": 7.3, "volatile": 6.8, "low_vol": 6.8},  # +0.3
    "CHFJPY":   {"trending": 7.0, "ranging": 8.0, "volatile": 7.5, "low_vol": 7.5},
}
DRAGON_SCALP_MIN_SCORE = 6.5       # minimum score for scalp entry
DRAGON_CONFIDENCE_FLOOR = 0.56     # ML meta-label floor (tuned: XAUUSD WR 37.3%→38.3% at 0.56)
DRAGON_MAX_CONSECUTIVE_LOSSES = 4  # blacklist symbol after 4 consecutive losses (was 3)
DRAGON_BLACKLIST_HOURS = 12        # hours to ban symbol after consecutive losses (was 24)
DRAGON_EQUITY_SLOPE_WINDOW = 20    # trades to measure equity slope
DRAGON_STANDBY_HOURS = 4           # hours of no favorable conditions before standby
DRAGON_RISK_SCALE_MIN = 0.5        # min risk % (floor for low confidence — tuned from 0.4)
DRAGON_RISK_SCALE_MAX = 1.2        # max risk % (tuned from 1.0 — PF 2.73 at 1.2%, DD still 19.4%)
DRAGON_LOSS_DAY_RISK_MULT = 0.5    # halve risk after losing day

# ═══════════════════════════════════════════════════════════════════
#  V5 SIGNAL QUALITY SCORING (0-100 scale, raw H1, NO blending)
# ═══════════════════════════════════════════════════════════════════
# Raw H1 score / divisor * 100 → normalized 0-100
# 12.0 = practical ceiling (11 components, ~12 in strong trends)
SIGNAL_QUALITY_DIVISOR = 12.0

# Per-regime minimum signal_quality (0-100) for entry
# V5 tuned: 45% beats 50% by $212/90d (+25%). Trail handles exit quality.
SIGNAL_QUALITY_THRESHOLDS: Dict[str, int] = {
    # 2026-06-04 CTO QUALITY OVERHAUL — user directive "quality trades only,
    # best setups". Memory [[feedback_selectivity_edge]]: relaxing 22→28 trades
    # dropped PF 6.91→2.04. Selectivity IS the edge.
    # Bumped across the board:
    #   trending 40 → 55 (raw 6.6) — require real conviction in trends
    #   ranging  42 → 60 (raw 7.2) — chop should be hardest to enter
    #   volatile 45 → 55 (raw 6.6) — slight relax (vol spikes generate high scores)
    #   low_vol  45 → 65 (raw 7.8) — kills the bleeder regime
    # Expected effect: ~50-70% fewer entries, much higher WR per trade.
    "trending": 55,    # 6.6 raw
    "ranging":  60,    # 7.2 raw — chop demands extra conviction
    "volatile": 55,    # 6.6 raw
    "low_vol":  65,    # 7.8 raw — the bleeder bucket, structural tighten
}

# Per-symbol quality override (where optimal differs from default)
SIGNAL_QUALITY_SYMBOL: Dict[str, Dict[str, int]] = {
    # Cleared 2026-04-29 — over-tuned to backtest regime, live bled at low thresholds.
    # All symbols use SIGNAL_QUALITY_THRESHOLDS default (45%).
}

# MTF high-conviction override: skip M15 gate if signal_quality >= this
# 2026-06-04 CTO QUALITY OVERHAUL: 75 → 80. Audit B4 showed bypass aggregate
# PF 0.62, with score 9.0-10.0 PF 1.38 and score ≥10 PF 0.06 (REVERSE signal).
# Raising threshold means only "true monster" signals bypass — most signals
# go through normal M15 confirmation.
MTF_OVERRIDE_QUALITY = 80  # 9.6 raw — truly exceptional only

# Conviction sizing on 0-100 scale (replaces old CONVICTION_SIZING)
CONVICTION_SIZING_V2: Dict[str, float] = {
    "80+":   1.5,    # 150% risk — PF 16.21 at this level
    "65-80": 1.2,    # 120% risk
    "55-65": 1.0,    # standard
    "<55":   0.6,    # reduced (only reachable via near-miss or reversal)
}

# ═══ DIRECTION BIAS (backtest-proven edge) ═══
# Restrict symbols to directions with PF > 1.5 (skip marginal/losing direction)
# None = both directions allowed
DIRECTION_BIAS: Dict[str, str] = {
    # 2026-05-06: removed stale XAUUSD/XAGUSD/USDCAD entries — markets shifted
    # to LONG-trend on metals; old SHORT bias was rejecting every signal.
    # Direction bias for these now flows from direction_bias_auto_dict.py
    # (180d backtest, periodically re-run) rather than ad-hoc 7d snapshots.
}

# 2026-05-17: per-(symbol, regime) direction bias. Schema {sym: {regime:
# 'LONG'|'SHORT'|'BOTH'}}. Overrides per-symbol DIRECTION_BIAS when the
# (sym, regime) cell is set. 'BOTH' explicitly allows both sides; cell-miss
# falls back to per-symbol DIRECTION_BIAS. Populated by auto_tuned.
DIRECTION_BIAS_REGIME: Dict[str, Dict[str, str]] = {}

# ═══ CONVICTION-BASED POSITION SIZING ═══
# Score 9+  → PF 16.21 (monster edge) → max size
# Score 8-9 → PF 2.31 (solid) → full size
# Score 7-8 → PF 5.76 (good) → standard size
# Score 6-7 → PF 1.30 (marginal) → reduced size
CONVICTION_SIZING: Dict[str, float] = {
    "9+":  1.5,    # 150% of base risk — high conviction, proven PF 16.21
    "8-9": 1.2,    # 120% of base risk
    "7-8": 1.0,    # 100% standard
    "6-7": 0.6,    # 60% reduced — marginal edge
}

# ═══ CONVICTION TIERING A+/B+/B (Phase 1 — 2026-06-14) ═══
# Discrete 3-tier classifier that replaces the continuous CONVICTION_SIZING_V2
# ladder. Lives in agent/expert/conviction_tiering.py; wired in brain.py
# Phase 3 between Gate 7 MasterBrain and the adaptive_mult min-chain.
#
# Behaviour:
#   • A+   raw≥9.0 + SQ≥75 + n_strong≥6 + HTF≥2 + M15 ok + sweep/absorption
#         + regime∈{trending,volatile}  →  size_mult 2.0×
#   • B+   raw∈[7,9) + SQ≥60 + n_strong≥4 + (HTF ok OR M15 ok) → size_mult 1.0×
#   • B    raw∈[6,7) marginal → SKIP (size_mult 0.0)
#   • FAIL raw<6.0 → SKIP
# Joins existing min-chain so 2.0× still gets capped by RL/DD/VRP/SYMBOL_RISK_CAP.
#
# Default OFF — must be A/B tested 7d demo before flipping vs legacy ladder.
CONVICTION_TIERING_ENABLED: bool = False
CONV_APLUS_RAW_MIN: float = 9.0           # PF 16.21 bucket
CONV_APLUS_SQ_MIN: int = 75               # below MTF_OVERRIDE_QUALITY (80)
CONV_APLUS_STRONG_MIN: int = 6            # ≥0.5 components out of 11
CONV_APLUS_ALLOWED_REGIMES = {"trending", "volatile"}  # None = allow all
CONV_APLUS_SIZE_MULT: float = 2.0
CONV_BPLUS_RAW_MIN: float = 7.0           # = MIN_EDGE_HIGH_CONV_SCORE
CONV_BPLUS_SQ_MIN: int = 60
CONV_BPLUS_STRONG_MIN: int = 4
CONV_BPLUS_SIZE_MULT: float = 1.0
CONV_B_RAW_MIN: float = 6.0               # = DRAGON_M15_MIN_SCORE_BASELINE
CONV_HTF_MIN_ALIGNED: int = 2             # of W1/D1/H4 from Gate 3d
CONV_OB_LOOKBACK_BARS: int = 24           # = ICT_SWEEP_LOOKBACK_BARS
CONV_OB_FRACTAL_N: int = 5                # = ICT_SWEEP_FRACTAL_N
CONV_REQUIRE_STRUCTURAL_APLUS: bool = True
CONV_BPLUS_REQUIRE_HTF_OR_M15: bool = True
CONV_TIER_LOG_SCORECARD: bool = True      # turn OFF in prod once tuned
CONV_TIER_PER_SYMBOL_OVERRIDES: Dict[str, Dict[str, Any]] = {}

# ═══ TOXIC HOURS — block entries during consistently losing hours ═══
# H01-04: low liquidity noise (but exempt crypto + JPN)
# H07-H08 REMOVED from toxic — was blocking all forex/gold right after session open
TOXIC_HOURS_UTC: set = {1, 2, 3, 4}
# Per-symbol overrides: some symbols trade well during "toxic" hours
# 2026-05-11: live trace caught BCHUSD signal=6.0 (would have approved) being
# rejected by UTC h04 toxic. Crypto symbols have 24/7 liquidity independent of
# forex/equities low-liquidity hours, so they should be exempt by default.
TOXIC_HOUR_EXEMPT: Dict[str, set] = {
    "BTCUSD":   {1, 2, 3, 4},  # crypto 24/7
    "BCHUSD":   {1, 2, 3, 4},  # crypto 24/7
    "ETHUSD":   {1, 2, 3, 4},  # crypto 24/7
    "JPN225ft": {1, 2, 3, 4},  # Asian index prime hours
}

# Per-symbol EXTRA toxic hours (added on top of TOXIC_HOURS_UTC).
# Added 2026-05-01: USDCAD bleeds at NY open (h=15-16 UTC), -$28 over 6 trades.
TOXIC_HOURS_PER_SYMBOL: Dict[str, set] = {
    "USDCAD": {15, 16},  # NY open USD volatility — 6 trades / -$28 / 7d
    # 2026-06-14: 10-agent workflow (wf_833e6497-e92) per-sym toxic hours.
    "EURUSD":   {3, 5, 8, 11, 13, 14, 15},
    "USOUSD":   {9, 10, 14, 18, 19, 22, 23},
    "ETHUSD":   {0, 5, 11, 17, 19},
    "SPI200.r": {3, 4, 5, 6, 7, 8, 9, 12, 13, 17, 18, 22, 23},
    "JPN225ft": {1, 7, 10, 22, 23},
    "DJ30.r":   {4, 5, 6, 7, 10, 16, 22, 23},
    "UK100.r":  {10, 14, 15, 16, 17, 18, 20},
    "NAS100.r": {3, 4, 14, 15, 16, 17},
}

# ═══ NEWS CALENDAR — high-impact event hard-block opt-in ═══
# CalendarFilter (agent/calendar_filter.py) checks ±30min around high-impact
# Forex Factory events. Default behavior (per no-skip rule): warn-only.
# Symbols in this set get a HARD SKIP during the event window. Reserve for
# pairs that historically blow up on major releases.
CALENDAR_HARD_BLOCK_SYMBOLS: set = {
    # 2026-05-26 audit additions:
    "USOUSD",   # EIA Crude Inventories Wed 14:30 UTC — catastrophic gap risk
    "USDJPY",   # BOJ Rate Decision + Q&A drag — wide gap-through risk
    # Default behavior for non-listed symbols: WARN-only at event boundaries
    # (per no-skip rule). This set upgrades to HARD-SKIP for the ±30min window.
}

# ═══ LONG-TERM TREND FILTER (H1 EMA(200) — proxy for D1 trend) ═══
# Audit 2026-05-06: ~40% of historical entries were counter to the long-term
# trend. Brain now logs every counter-trend entry as a warning. Symbols here
# upgrade the warning to a hard skip — reserved for trending markets where
# counter-trend reversion fades hard.
TREND_FILTER_HARD_BLOCK_SYMBOLS: set = {
    # Trending forex majors (per-symbol live evidence):
    "EURUSD",   # was 28% WR last 7d when LONG-only — D1 trend was DOWN
    "GBPUSD",   # was 38% WR last 7d
    "USDJPY",   # 22% WR — strong USD up trend, counter-shorts bled
    # Indices typically trend hard; keep these strict:
    "JPN225ft", # 30% WR pre-tune — clear counter-trend
    # Crypto: long horizons matter
    "BTCUSD",   # 25% WR
    "ETHUSD",   # 50% WR but giveback heavy — trend matters
    # 2026-05-14: XAU 3-of-3 last losses were SHORT counter-trend; -$49.81 net over 25 trades.
    "XAUUSD",
    # 2026-06-17 expansion: indices trend hard — enforce HTF alignment.
    "GER40.r",
    "SP500.r",
    "US2000.r",
    "DJ30.r",
    "UK100.r",
}

# ═══ D1 SWING-STRUCTURE BIAS (HH/HL + BOS/CHoCH) — Gate 3c replacement ═══
# Replaces the H1-EMA(200) "D1-trend proxy" at Gate 3c with a proper
# SMC/ICT swing-structure read on confirmed D1 bars resampled from H1.
# Algorithm + module: agent/expert/"D1 Swing-Structure Bias (HH/HL + ...).py"
#
# Roll-out is gated so we can A/B vs the current EMA200 path:
#   ENABLED=False                        -> EMA200 only (status quo)
#   ENABLED=True, REPLACES_EMA200=False  -> shadow-log only, no trade impact
#   ENABLED=True, REPLACES_EMA200=True   -> D1 structure becomes Gate 3c
# Env-toggle via _envbool so backtests can A/B without editing config.
D1_STRUCTURE_BIAS_ENABLED = _envbool("D1_STRUCTURE_BIAS_ENABLED", True)
D1_STRUCTURE_REPLACES_EMA200 = _envbool("D1_STRUCTURE_REPLACES_EMA200", False)

D1_STRUCTURE_MIN_BARS = 60          # ~2 months of confirmed D1 — gate passes below
D1_STRUCTURE_FRACTAL_N = 2          # 5-bar Williams pivot (2 + center + 2)
D1_STRUCTURE_SWING_MIN_COUNT = 4    # need 2 prior + 2 latest swings each side
D1_STRUCTURE_BOS_LOOKBACK_BARS = 30 # scan window for most-recent BOS/CHoCH
D1_STRUCTURE_FRESHNESS_BARS = 10    # event older than this loses 'fresh' weight
D1_STRUCTURE_MIN_SWING_SPACING = 2  # dedupe adjacent fractals < N bars apart

# Symbols where verdict=REJECT actually blocks the trade (others = WARN-log).
# Mirrors TREND_FILTER_HARD_BLOCK_SYMBOLS pattern. Start empty; populate from
# the journal after the shadow-log burn-in.
D1_STRUCTURE_HARD_BLOCK_SYMBOLS: set = set()

# If True, strength=0.5 (partial HH or HL only) only logs WARN, never REJECT.
# False = full strict bias mode (partial counter-bias can still REJECT).
D1_STRUCTURE_WEAK_BIAS_WARNS = True

# Optional risk multiplier when verdict=SNIPER (D1 bias + fresh same-dir BOS).
# 0.0 disables; 0.25 = +25% size on full-stack confluence. Plumbed but inert
# by default — flip on only after the shadow-log proves the SNIPER tier has
# materially better expectancy than baseline.
D1_STRUCTURE_SNIPER_UPLIFT = 0.0

# ═══ PULLBACK ENTRY — wait for retrace before entering ═══
# Instead of entering at signal bar close, require price to pull back
# towards the signal direction before entering (better fill, higher WR)
PULLBACK_ENTRY_ENABLED = True    # 2026-05-16: RE-ENABLED to match backtest behavior.
                                 # Backtest v5_backtest.py:733-746 has ALWAYS simulated
                                 # pullback (retrace 0.2 ATR, 1-bar lookahead, fallback
                                 # to direct on miss) — its $21,273/180d already counts
                                 # pullback fills. Live was diverging by skipping. Now
                                 # mirrors backtest exactly: wait 1 bar, fallback to
                                 # direct entry on expiry (no skipped trades).
PULLBACK_ATR_RETRACE = 0.2
PULLBACK_MAX_WAIT_BARS = 1       # 2026-05-16: 3→1 to match backtest's 1-bar lookahead.
PULLBACK_REGIMES = {"trending", "volatile"}  # only wait for pullback in these regimes

# ═══ PER-SYMBOL PULLBACK OVERRIDES (2026-05-29 deep 3yr-H1 tune) ═══
# Tuned per symbol on 3yr H1 with realistic expiry-bar fallback fills, then
# dual-period validated (winner must beat default PF in BOTH the recent year
# AND the older window, with PF>1 real edge in each). Only 3 symbols produced
# a robust improvement; all favor a SHALLOWER retrace than the 0.2 default —
# i.e. for these, waiting for a deep pullback costs more than it gains.
#   retrace = ATR multiple for the limit (0.0 = enter at signal price, no wait)
#   wait    = max H1 bars to wait (BACKTEST unit). NOTE: live currently honors
#             only `retrace` per-symbol; the live wait window stays at the global
#             PULLBACK_MAX_WAIT_BARS because live measures the wait in minutes,
#             not H1 bars (a separate unit reconciliation). All tuned winners
#             have wait=1 = the global default, so nothing is lost by this.
PULLBACK_CONFIG_PER_SYMBOL: Dict[str, dict] = {
    "SPI200.r": {"retrace": 0.0, "wait": 1},
    "US2000.r": {"retrace": 0.1, "wait": 1},
    "USOUSD":   {"retrace": 0.0, "wait": 1},
    # 2026-06-17 EURUSD hard-tune winner: enter at signal close, no wait
    # (standalone PF=10.12, WF strict 5-fold 4/5 pf>=1.5 + 4/5 pnl>0)
    "EURUSD":   {"retrace": 0.0, "wait": 1},
}

# ═══ CORRELATION PAIRS ═══
# Won't open simultaneous positions in both symbols if correlation >= threshold
# Correlation pairs — all 19 live universe (calibrated estimates from typical H1 corr).
# Used to block opening N+1th highly-correlated position. Keep dropped-symbol entries
# harmless (they self-skip if symbol not in active SYMBOLS).
CORRELATION_PAIRS: Dict[Tuple[str, str], float] = {
    # Index correlations (US risk-on cluster)
    ("DJ30.r", "US2000.r"): 0.75,
    # Index correlations (European cluster)
    ("GER40.r", "FRA40.r"): 0.85,
    ("GER40.r", "UK100.r"): 0.70,
    ("FRA40.r", "UK100.r"): 0.70,
    ("GER40.r", "SWI20.r"): 0.65,
    # Asia-Pacific
    ("HK50.r", "US2000.r"): 0.55,
    # USD-quote forex (move together)
    ("USDCAD", "USDCHF"): 0.55,
    # JPY-cross cluster (risk-on/off proxy)
    ("AUDJPY", "CADJPY"): 0.80,
    ("AUDJPY", "CHFJPY"): 0.65,
    ("CADJPY", "CHFJPY"): 0.65,
    # GBP cross cluster
    ("GBPAUD", "GBPCHF"): 0.55,
    # 2026-05-26 audit additions:
    ("BTCUSD", "ETHUSD"): 0.75,    # Crypto cluster (was cold-start blind)
    ("USDJPY", "CHFJPY"): 0.60,    # JPY-stacked exposure (USD+CHF vs JPY same side)
    ("USOUSD", "UKOUSD"): 0.85,    # WTI/Brent — same-direction = doubled oil risk
}

# ═══ RL LEARNING — per-symbol toggle + tuned params ═══
# Updated 2026-05-09 for 19-symbol live universe. Conservative: enable RL only on
# symbols with ≥100 walk-forward-test trades and ROBUST verdict. Others bypass RL
# (returns 1.0 multiplier from get_risk_multiplier).
RL_ENABLED_SYMBOLS = {
    # Indices with strong tuned PnL + robust walk-forward
    "DJ30.r", "US2000.r", "GER40.r", "HK50.r", "SWI20.r", "UK100.r",
    # 2026-05-26 audit additions — confirmed >100 WF trades + ROBUST verdict
    "SP500.r", "NAS100.r",
    # Forex pairs that have enough trade history + robust
    "USDCAD", "USDCHF", "EURUSD", "AUDJPY", "CADJPY", "CHFJPY", "GBPAUD",
    "USDJPY",   # 2026-05-26 audit: was missing despite live universe entry
    # Commodities with healthy sample
    "UKOUSD", "COPPER-Cr",
    # Crypto
    "ETHUSD",
    # 2026-06-03 CTO audit: previously-deferred symbols now have sufficient
    # data. regime_weights shows XAU trending 43W/18L (61 trades, 70.5% WR)
    # and BTC trending 25W/44L (69 trades, 36.2% WR) — both well past the
    # 100-trade burn-in concern. The 2026-04-29 H1 cache bug that motivated
    # the deferral is long fixed. RL was learning their weights but blind to
    # their entry/risk decisions (half-deployed state). Enabling closes the loop.
    "XAUUSD", "BTCUSD", "JPN225ft", "SPI200.r",
    # NOT yet: USOUSD (3 trades), XPTUSD.r (no data) — still below burn-in
    # NOT: GBPCHF (only 186 trades in 180d — under-sampled), FRA40.r (54 trades), NG-Cr (178 — borderline keep)
}
# Per-symbol RL params — generic defaults for new universe; refine after demo data lands.
RL_SYMBOL_PARAMS: Dict[str, Dict] = {
    # Index defaults: 30d lookback, mild boost ceiling
    "DJ30.r":     {"lookback": 30, "boost_max": 1.3},
    "US2000.r":   {"lookback": 30, "boost_max": 1.3},
    "GER40.r":    {"lookback": 30, "boost_max": 1.3},
    "HK50.r":     {"lookback": 30, "boost_max": 1.3},
    "SWI20.r":    {"lookback": 30, "boost_max": 1.3},
    "UK100.r":    {"lookback": 30, "boost_max": 1.3},
    # Forex defaults: 20d lookback, slightly higher ceiling
    "USDCAD":     {"lookback": 30, "boost_max": 1.5},
    "USDCHF":     {"lookback": 20, "boost_max": 1.4},
    "EURUSD":     {"lookback": 20, "boost_max": 1.4},
    "AUDJPY":     {"lookback": 20, "boost_max": 1.4},
    "CADJPY":     {"lookback": 20, "boost_max": 1.4},
    "CHFJPY":     {"lookback": 20, "boost_max": 1.4},
    "GBPAUD":     {"lookback": 20, "boost_max": 1.4},
    "USDJPY":     {"lookback": 20, "boost_max": 1.4},  # 2026-05-26 audit
    # 2026-05-26 audit additions
    "SP500.r":    {"lookback": 30, "boost_max": 1.3},
    "NAS100.r":   {"lookback": 30, "boost_max": 1.3},
    # Commodities
    "UKOUSD":     {"lookback": 30, "boost_max": 1.4},
    "COPPER-Cr":  {"lookback": 30, "boost_max": 1.3},
    # Crypto
    "ETHUSD":     {"lookback": 20, "boost_max": 1.3},
    # 2026-06-03 CTO audit additions (matching RL_ENABLED_SYMBOLS expansion):
    "XAUUSD":     {"lookback": 30, "boost_max": 1.4},  # Gold: 30d lookback, moderate ceiling
    "BTCUSD":     {"lookback": 20, "boost_max": 1.3},  # Crypto: shorter lookback (faster regime shift), tight ceiling
    "JPN225ft":   {"lookback": 30, "boost_max": 1.3},  # Index defaults
    "SPI200.r":   {"lookback": 30, "boost_max": 1.3},  # Index defaults
}

# ═══ AUTO-TUNED OVERRIDES (loop optimizer output) ═══
# Layered ON TOP of hand-tuned values above. Generated by scripts/synthesize_auto_tuned.py.
# To revert: delete /auto_tuned.py
try:
    import auto_tuned as _at  # type: ignore
    SYMBOL_ATR_SL_OVERRIDE.update(getattr(_at, "SL_OVERRIDE_AUTO", {}))
    # 2026-05-17: deep-merge per-(symbol, regime) SL overrides — never blanket
    # overwrite per-symbol fallback dict.
    for _s, _rd in getattr(_at, "SL_OVERRIDE_REGIME_AUTO", {}).items():
        SYMBOL_ATR_SL_OVERRIDE_REGIME.setdefault(_s, {}).update(_rd)
    SIGNAL_QUALITY_SYMBOL.update(getattr(_at, "SIGNAL_QUALITY_SYMBOL_AUTO", {}))
    DIRECTION_BIAS.update(getattr(_at, "DIRECTION_BIAS_AUTO", {}))
    # 2026-05-17: deep-merge per-(symbol, regime) direction bias.
    for _s, _rd in getattr(_at, "DIRECTION_BIAS_REGIME_AUTO", {}).items():
        DIRECTION_BIAS_REGIME.setdefault(_s, {}).update(_rd)
    SYMBOL_RISK_CAP.update(getattr(_at, "RISK_CAP_AUTO", {}))
    # 2026-05-17: deep-merge per-(symbol, regime) risk caps.
    for _s, _rd in getattr(_at, "RISK_CAP_REGIME_AUTO", {}).items():
        SYMBOL_RISK_CAP_REGIME.setdefault(_s, {}).update(_rd)
    for _s, _hours in getattr(_at, "TOXIC_HOURS_PER_SYMBOL_AUTO", {}).items():
        TOXIC_HOURS_PER_SYMBOL.setdefault(_s, set()).update(set(_hours))
    SYMBOL_TRAIL_OVERRIDE.update(getattr(_at, "TRAIL_OVERRIDE_AUTO", {}))
    # 2026-05-17: deep-merge per-(symbol, regime) trail overrides.
    for _s, _rd in getattr(_at, "TRAIL_OVERRIDE_REGIME_AUTO", {}).items():
        SYMBOL_REGIME_TRAIL_OVERRIDE.setdefault(_s, {}).update(_rd)

    # 2026-05-11: live-evidence overrides on top of auto_tuned. These
    # take precedence because auto_tuned was regenerated against a 180d
    # window during a different market regime; current live signals show
    # the auto-tuned bias is stale.
    #   ETHUSD: 4512 SHORT signals vs 2280 LONG in 24h → opening to BOTH
    #     (LONG bias was blocking 1408 signals/hour). Backtest is 13.77 vs
    #     11.70 PF — close enough to let live data decide.
    #   XAUUSD: gold rallying. 126 LONG signals vs 35 SHORT in 24h. SHORT
    #     bias was blocking peak-9.0 LONG signals every cycle. Opening to
    #     BOTH so the user's "no gold trades" stops.
    #   GBPUSD: bias was LONG but backtest LONG = PF 0.16 / -$83 over 180d
    #     vs SHORT PF 1.84 / +$437. Live signals 3:1 SHORT. Flip to SHORT.
    # DIRECTION_BIAS.pop("ETHUSD", None)   # 2026-05-29: lifted to test 3yr ETHUSD:LONG bias (was → BOTH)
    # 2026-06-03 CTO audit (A9): XAU was popped to BOTH 2026-05-11 when "no
    # gold trades" complaint was active. 12d live evidence shows SHORT 28t /
    # 60.7% WR / +$39.62 vs LONG 6t / 50% WR / -$6.50. Restore SHORT bias.
    DIRECTION_BIAS["XAUUSD"] = "SHORT"
    DIRECTION_BIAS["GBPUSD"] = "SHORT"   # flip from LONG (BT SHORT PF 1.84 vs LONG 0.16)

    # 2026-05-11 parallel-agent audit follow-up — 3 more bias overrides:
    #   BCHUSD: live 8x more high-conv LONG (64 L≥6 vs 8 S≥6). BT SHORT
    #     edge (PF 5.03) is based on 144 trades in different regime — let
    #     current market decide.
    #   UKOUSD: live shows 73 high-conv SHORT vs 0 LONG (oil clearly
    #     selling). BT LONG PF 9.78 was the old regime.
    #   USDCAD: BT SHORT PF 14.44 > LONG PF 12.82 (auto_tuned chose wrong).
    DIRECTION_BIAS.pop("BCHUSD", None)   # → BOTH (live LONG surge)
    DIRECTION_BIAS.pop("UKOUSD", None)   # → BOTH (live SHORT surge)
    DIRECTION_BIAS.pop("USDCAD", None)   # → BOTH (BT SHORT > LONG)

    # 2026-05-11 parallel-agent audit (#2) — 12 symbols' quality thresholds
    # were 2pp too tight: peak live quality consistently missed by exactly
    # 2pp AND backtest had decent (≥0.3 tr/day) profitable performance
    # (PF≥1.5). Drop each by 2pp. Ranging regime keeps a +5pp offset.
    SIGNAL_QUALITY_SYMBOL.update({
        "BCHUSD":  {"trending": 48, "ranging": 48, "volatile": 48, "low_vol": 48},
        "SP500.r": {"trending": 38, "ranging": 43, "volatile": 38, "low_vol": 38},
        "GER40.r": {"trending": 35, "ranging": 40, "volatile": 35, "low_vol": 35},
        "US2000.r":{"trending": 35, "ranging": 40, "volatile": 35, "low_vol": 35},
        "NG-Cr":   {"trending": 35, "ranging": 40, "volatile": 35, "low_vol": 35},
        "UKOUSD":  {"trending": 48, "ranging": 48, "volatile": 48, "low_vol": 48},
        "EURUSD":  {"trending": 38, "ranging": 43, "volatile": 38, "low_vol": 38},
        "USDCAD":  {"trending": 33, "ranging": 38, "volatile": 33, "low_vol": 33},
        "USDCHF":  {"trending": 35, "ranging": 40, "volatile": 35, "low_vol": 35},
        "EURAUD":  {"trending": 35, "ranging": 40, "volatile": 35, "low_vol": 35},
        "GBPAUD":  {"trending": 35, "ranging": 40, "volatile": 35, "low_vol": 35},
        "GBPCHF":  {"trending": 35, "ranging": 40, "volatile": 35, "low_vol": 35},
    })
    # 2026-06-03 CTO tightenings — applied AFTER the 2026-05-11 block so these
    # take precedence (.update() last-write-wins). Thresholds are signal_quality
    # on the 0-100 scale (= raw_score / 12 * 100).
    #   BTCUSD trending: 40 (baseline) → 58 (= raw 7.0). RL recorded 25W/44L
    #     (36% WR) on BTC trending + 3× HARD-floor bypass fires today.
    #   XAUUSD trending: 40 (baseline) → 50 (= raw 6.0). Today's XAU losers all
    #     scored raw 5.0-5.5 = quality 42-46%. RL shows XAU edge IS real
    #     (43W/18L trending = 70.5% WR) so sub-floor entries dilute it.
    #   EURUSD low_vol: 38 → 92 (= raw 11.04). 30d data: 17 trades all in
    #     low_vol, all losing across scores 5.0-10.5. Score not predictive in
    #     low_vol EURUSD — effective disable until regime shifts.
    SIGNAL_QUALITY_SYMBOL.update({
        "BTCUSD":  {"trending": 58, "ranging": 60, "volatile": 50, "low_vol": 50},
        "XAUUSD":  {"trending": 50, "ranging": 55, "volatile": 50, "low_vol": 50},
        "EURUSD":  {"trending": 38, "ranging": 43, "volatile": 38, "low_vol": 92},
    })
    # 2026-06-04 CTO QUALITY OVERHAUL — bleeder per-symbol tightening on top
    # of the new (55/60/55/65) baseline. Symbols still losing money over 3d
    # post-trim are dampened to "premium only" thresholds.
    # Also bumping XAU/BTC/EURUSD so they're at-or-above the new baseline (55
    # trending) — yesterday's settings of 50/58/38 would be LOOSER than the
    # new baseline. Restoring "at least as strict as baseline".
    SIGNAL_QUALITY_SYMBOL.update({
        # ETHUSD — losing in volatile/trending. Audit A2 ETH weights universally
        # floored at 0.95 = symbol-level edge problem. Premium only.
        "ETHUSD":  {"trending": 65, "ranging": 70, "volatile": 65, "low_vol": 70},
        # UK100.r — 28.6% WR over 7 trades since trim. Lift bar.
        "UK100.r": {"trending": 60, "ranging": 65, "volatile": 60, "low_vol": 95},
        # NAS100.r — 0% WR over 2 trades. Sample thin but bleeding-window data.
        "NAS100.r":{"trending": 60, "ranging": 65, "volatile": 65, "low_vol": 95},
        # USOUSD — 70% WR but PF 0.73 (small wins, big losses). Slight raise.
        "USOUSD":  {"trending": 55, "ranging": 60, "volatile": 55, "low_vol": 70},
        # Restore strict baseline-alignment for the prior CTO tightenings:
        "BTCUSD":  {"trending": 60, "ranging": 65, "volatile": 55, "low_vol": 60},
        "XAUUSD":  {"trending": 55, "ranging": 60, "volatile": 55, "low_vol": 60},
        "EURUSD":  {"trending": 55, "ranging": 60, "volatile": 55, "low_vol": 95},
    })
    # 2026-06-17 EURUSD hard-tune winner override (must come AFTER all earlier
    # SIGNAL_QUALITY_SYMBOL updates to win .update() last-write priority).
    # Winner cfg: 50/50/50/50 (= raw 6.0 uniform) — standalone PF=10.12,
    # WF strict 5-fold passed (pf_ok 4/5 + pnl_ok 4/5).
    SIGNAL_QUALITY_SYMBOL["EURUSD"] = {
        "trending": 50, "ranging": 50, "volatile": 50, "low_vol": 50,
    }
except ImportError:
    pass

# ═══ 2026-06-05 — RESEARCH-DRIVEN ENTRY/EXIT ADDITIONS ═══
# Time-stop (no-progress cut) — KJTradingSystems / ATAS / Davey
TIME_STOP_ENABLED = True
TIME_STOP_BARS = 12              # 12 × M15 = 3 hours
TIME_STOP_MIN_PEAK_R = 0.3       # close if peak never reached +0.3R

# Break-of-structure invalidation exit — Strike, ICT
BOS_INVALIDATION_ENABLED = True

# ═══ Anchored VWAP Rejection Booster — institutional respect of fair value ═══
# Source: Brian Shannon, "Maximum Trading Gains with Anchored VWAP" (2022).
# When price tests session/anchored VWAP and rejects (wick pierces, body closes
# back on the trend side) within `ANCHORED_VWAP_LOOKBACK_BARS`, add the boost
# to raw_score BEFORE the quality-threshold check. Default OFF — A/B validate
# before flipping live. Decoupled detector lives in
# agent/expert/anchored_vwap_rejection.py.
ANCHORED_VWAP_BOOSTER_ENABLED = False
ANCHOR_BARS_DEFAULT           = 24    # 24 H1 bars ≈ one trading day
ANCHORED_VWAP_LOOKBACK_BARS   = 5     # rejection freshness window (bars)
ANCHORED_VWAP_BOOST_AMOUNT    = 1.0   # +1 raw_score on confirmed rejection

# VWAP entry gate (index intraday momentum) — Zarattini SSRN 4824172
VWAP_GATE_ENABLED = True
VWAP_GATE_SYMBOLS = {
    # 2026-06-17 user req: VWAP exit must work on all live syms.
    "XAUUSD", "EURUSD", "BTCUSD",      # added — was indices-only before
    "DJ30.r", "UK100.r",                # existing live indices
    # Other indices (kept for if-re-enabled symbols, harmless when not in SYMBOLS):
    "SP500.r", "NAS100.r", "US2000.r",
    "JPN225ft", "SPI200.r", "GER40.r", "SWI20.r", "HK50.r",
}

# News blackout window (skip new entries near tier-1 econ events)
NEWS_BLACKOUT_ENABLED = True
NEWS_BLACKOUT_MIN_BEFORE = 5     # minutes before event
NEWS_BLACKOUT_MIN_AFTER = 15     # minutes after event

# ═══ 2026-06-16 — Extended News Blackout v2 (tier-1 4h window + flatten) ═══
# Phase-2 module: agent/expert/Extended News Blackout Windows v2 ...
# Tiered windows + pre-event POSITION FLATTEN + live FF calendar merge.
# Default OFF so live runs identical to v1 ±5/+15min until A/B-validated.
# Phase-3 will wire in brain.py: entry-side gate (line ~1528) + flatten hook
# in _run_cycle (line ~1117 before manage_trailing_sl).

NEWS_BLACKOUT_V2_ENABLED = True
# Master ON/OFF for the v2 extended blackout module. When False, brain falls
# back to existing v1 is_in_blackout(). A/B vs v1 ±5/+15min implementation.

NEWS_BLACKOUT_TIER1_MIN_BEFORE = 120
# Minutes BEFORE a tier-1 event to start blocking new entries
# (FOMC, NFP, CPI, ECB, BOE, BOJ, Powell, Core PCE).

NEWS_BLACKOUT_TIER1_MIN_AFTER = 120
# Minutes AFTER a tier-1 event to keep blocking. 120+120 = 4h total per spec.

NEWS_BLACKOUT_TIER2_MIN_BEFORE = 15
# Minutes before tier-2 events (PPI, ISM, claims). Kept tight — these don't
# cause 4h variance.

NEWS_BLACKOUT_TIER2_MIN_AFTER = 30
# Minutes after tier-2 events.

NEWS_FLATTEN_ENABLED = True
# Master ON/OFF for the pre-event position flatten behavior. Independent of
# blackout entry-side gate so we can ship gate-only first, then enable flatten
# after observing one event cycle.

NEWS_FLATTEN_LEAD_MIN = 30
# Minutes before a tier-1 event to start force-closing existing positions.

NEWS_FLATTEN_FORCE_ALL_AT_R = -0.5
# If a position is worse than this R-value at flatten time, force close
# regardless of other rules.

NEWS_FLATTEN_KEEP_WINNER_R = 0.5
# Winners at or above this R-value at T-30m are allowed to ride through the
# event (existing trail logic manages them).

NEWS_BLACKOUT_OPT_OUT = set()
# Symbols that ignore news blackout entirely (e.g. {'BTCUSD','ETHUSD'} if a
# crypto-only strategy says FX events don't matter — currently empty because
# Fed events DO move BTC).

NEWS_BLACKOUT_TIER1_FORCE_SYMBOLS = {"XAUUSD", "USDJPY", "USOUSD"}
# Symbols where ALL relevant events get upgraded to tier-1 (gold is news-
# sensitive even on PPI; USOUSD blows up on EIA crude).

NEWS_CALENDAR_FILTER_MERGE_ENABLED = True
# Whether to merge live Forex Factory feed (calendar_filter.py) with the
# hardcoded deterministic calendar. False = hardcoded-only (safer on internet
# outage); True = redundant dual-source coverage.

NEWS_BLACKOUT_TIER1_EVENTS = {"FOMC", "NFP", "US_CPI", "ECB", "BOE", "BOJ",
                              "POWELL_SPEECH", "CORE_PCE"}
# Set of event kinds that get the 4-hour window.

NEWS_BLACKOUT_TIER2_EVENTS = {"US_PPI", "US_RETAIL", "ISM_PMI", "UNEMP_CLAIMS",
                              "FOMC_MINUTES", "GDP_ADV"}
# Set of event kinds that get the ±15/30m tier-2 window.

NEWS_BLACKOUT_LOG_LEVEL = "WARNING"
# Log level for blackout/flatten events. WARNING during break-in week, INFO
# after.

# Correlation cluster cap — industry rule: ≤2 positions per correlated cluster
CORRELATION_CAP_ENABLED = True
CORRELATION_CAP_PER_CLUSTER = 2
CORRELATION_CLUSTERS = {
    "US_INDICES": {"DJ30.r", "SP500.r", "NAS100.r", "US2000.r"},
    "EU_INDICES": {"GER40.r", "UK100.r", "SWI20.r"},
    "ASIA_INDICES": {"JPN225ft", "SPI200.r", "HK50.r"},
    "JPY_PAIRS":  {"USDJPY", "AUDJPY", "CHFJPY", "GBPJPY", "CADJPY", "EURJPY"},
    "GOLD_SILVER": {"XAUUSD", "XAGUSD", "XPTUSD.r"},
    "CRYPTO":     {"BTCUSD", "ETHUSD"},
}

# ═══ 2026-07-18 — CORRELATED-GROUP + PER-SYMBOL HEAT CAPS (drawdown control) ═══
# The count-based CORRELATION_CAP above limits the *number* of legs per fine
# cluster, but nothing bounds the aggregate OPEN RISK (distance-to-SL × size) of
# a whole correlated bloc — nor of a SINGLE over-risk symbol traded by several
# books at once. On the live ~$2.4K account, min-lot (0.01) at a 2×ATR stop is
# already ~1.33% on XAUUSD, and FOUR books trade XAU (momentum, gold_smc, ASAT,
# scalper) — so they can silently stack ~5% of correlated gold risk one way.
# These caps sum the open-risk (% equity) of every OPEN position that shares the
# candidate's correlation GROUP and, separately, its SYMBOL, and REJECT the open
# if either the group cap, the per-symbol cap, or the portfolio cap would breach.
# ENTRIES ONLY — never touches management/exits. Fail-CLOSED: an unmapped symbol
# is its own singleton group at GROUP_HEAT_DEFAULT_CAP, and a positions-read
# failure / non-positive equity blocks the entry.
#
# NOTE (deliberately coarser than CORRELATION_CLUSTERS): all equity indices move
# together on a risk-off day regardless of region, so US/EU/ASIA are merged into
# a single INDEX heat group here (the architect's top RISK recommendation).
CORRELATION_GROUPS: Dict[str, str] = {
    # Equity indices — one risk-off bloc (US + EU + Asia-Pacific)
    "DJ30.r": "INDEX", "SP500.r": "INDEX", "NAS100.r": "INDEX",
    "US2000.r": "INDEX", "GER40.r": "INDEX", "UK100.r": "INDEX",
    "SWI20.r": "INDEX", "FRA40.r": "INDEX", "JPN225ft": "INDEX",
    "SPI200.r": "INDEX", "HK50.r": "INDEX",
    # Crypto
    "BTCUSD": "CRYPTO", "ETHUSD": "CRYPTO", "BCHUSD": "CRYPTO",
    # Precious metals (gold complex)
    "XAUUSD": "GOLD", "XAGUSD": "GOLD", "XPTUSD.r": "GOLD",
}

# Aggregate open-risk cap per correlated GROUP, as % of equity. Sum of
# distance-to-SL × size over all same-group open positions must stay under this.
# 2026-07-20 RECALIBRATED: original caps (INDEX 1.0/CRYPTO 0.5) were set BELOW a
# single position's real open-risk — a TREND entry uses a wide chandelier stop
# (~1.0-1.5% risk), so the caps blocked the FIRST trade, not just stacking. Raised
# so ONE-TWO positions per group fit while heavy stacking (the 4-book XAU stack)
# is still blocked by the per-symbol caps + portfolio ceiling.
GROUP_HEAT_CAPS: Dict[str, float] = {
    "INDEX":  3.0,
    "CRYPTO": 2.0,
    # GOLD group cap 2.0% == the XAUUSD per-symbol cap below: XAU is the only
    # heavily-traded name in the gold complex (4 books), so its per-symbol cap
    # IS effectively the binding gold-bloc cap; keeping them equal avoids a
    # group cap that would silently override (and loosen) the per-symbol XAU cap.
    "GOLD":   2.0,
}
# Per-SYMBOL aggregate open-risk cap, as % of equity — for the over-risk names
# that several books trade concurrently. Binds BEFORE the group cap. XAUUSD is
# the headline case: momentum + gold_smc + ASAT + scalper can each open a
# min-lot XAU leg (~1.33% at 2×ATR), so without this they stack ~5% one way.
PER_SYMBOL_HEAT_CAPS: Dict[str, float] = {
    "XAUUSD": 2.0,
    "XAGUSD": 1.5,
    "NAS100.r": 1.5,
    "DJ30.r": 1.5,
    "GER40.r": 1.5,
    "BTCUSD": 1.5,
    "ETHUSD": 1.5,
}
# Unmapped symbol → its own group; use this cap (fail-closed but never blocks a
# lone uncorrelated trade whose per-trade risk stays under it).
GROUP_HEAT_DEFAULT_CAP = 2.0   # 2026-07-20: 1.0→2.0 so a lone forex/unmapped trade (small risk) is never blocked
# Portfolio-wide aggregate open-risk cap across ALL SL-defined open positions.
# 2026-07-20: 2.0→6.0 — the old 2.0% was consumed by a SINGLE wide-SL index position
# (a 1.0-lot JPN225 orphan risked 3.46% alone), which froze ALL trading portfolio-wide.
PORTFOLIO_HEAT_CAP_PCT = 6.0
# Master toggle (env-overridable for A/B backtests).
GROUP_HEAT_CAPS_ENABLED = _envbool("GROUP_HEAT_CAPS_ENABLED", True)


def correlation_group(symbol: str) -> str:
    """Coarse correlation group for heat caps. Unmapped → the symbol itself as a
    singleton group (fail-closed: it can never silently join another group)."""
    return CORRELATION_GROUPS.get(symbol, symbol)

# Daily-loss kill — RE-ENABLE at 3% (was DAILY_HARD_STOP_PCT=40% effectively off).
# FTMO/FundedNext/Topstep converge on 3-5%; live 14d EmergencyDD fired 13× avg -5.29R.
DAILY_LOSS_KILL_ENABLED = True
DAILY_LOSS_KILL_PCT = 3.0

# ═══ 2026-06-16 — SESSION-CONDITIONAL SETUP LOGIC (SCSL) ═══
# Phase-2 module: agent/expert/session_setup.py.
# Classifies each candidate as Asia-range-scalp / London-breakout /
# NY-continuation / NY-late-fade and rejects mismatched setups.
# Default DISABLED so live runs identical to pre-change until A/B-validated.
# Phase-3 will wire as Gate 3g (immediately after Gate 3f ICT_NO_SWEEP).

SESSION_SETUP_ENABLED = True
# Symbols whose natural session is Asian or are 24/7 crypto — skip SCSL
# entirely to avoid blanket-blocking them outside London/NY.
SESSION_SETUP_BYPASS_SYMBOLS = {"BTCUSD", "ETHUSD", "BCHUSD", "JPN225ft"}

# UTC hour boundaries per session. Late-NY-fade overlays the last 2h of NY.
SCSL_SESSION_BANDS = {
    "ASIA":         (0, 7),
    "LONDON":       (7, 13),
    "NY":           (13, 21),
    "LATE_NY_FADE": (19, 21),
    "ASIA_PREP":    (21, 24),
}

# Setup-classifier knobs
SCSL_RANGE_POS_LB = 24            # H1 bars for range-position computation.
SCSL_BB_BREAKOUT_PCT = 0.85       # bb_width pctile (over 100 bars) >= ⇒ expansion.
SCSL_BB_RANGE_PCT = 0.35          # bb_width pctile <= ⇒ squeeze / mean-revert.
SCSL_EMA20_TREND_DIST_ATR = 0.5   # |close - EMA20| in ATR multiples for TREND_CONT.
SCSL_FADE_POS_HI = 0.80           # range-pos >= ⇒ SHORT tagged FADE (top of range).
SCSL_FADE_POS_LO = 0.20           # range-pos <= ⇒ LONG  tagged FADE (bottom of range).

# Warn-only mode — compute classification + verdict, log it, but do NOT reject.
# First 5-7 day burn-in to collect setup distribution before hard-skip flip.
SCSL_LOG_ONLY = True

# Skip the gate (fail-open) if fewer H1 bars are available, so the gate
# doesn't blanket-block at startup.
SCSL_MIN_H1_BARS = 120

# 5x5 allow-map. Each cell is (allowed_bool, min_quality_bump_added_to_regime_threshold).
# Bump ADDS to the caller's base_min_quality. Hot-tunable per-cell.
#   Asia    = thin liquidity, false breakouts dominate → revert only
#   London  = expansion session → breakouts + trend follow work
#   NY      = trend-continuation prime
#   Late-NY = fade window
SCSL_ALLOW_MAP = {
    "ASIA": {
        "RANGE_REVERT":  (True,  0),
        "BREAKOUT_CONT": (False, 99),
        "TREND_CONT":    (False, 5),
        "FADE":          (False, 99),
        "MIXED":         (False, 99),
    },
    "ASIA_PREP": {
        "RANGE_REVERT":  (True,  5),
        "BREAKOUT_CONT": (False, 99),
        "TREND_CONT":    (False, 10),
        "FADE":          (False, 99),
        "MIXED":         (False, 99),
    },
    "LONDON": {
        "RANGE_REVERT":  (False, 99),
        "BREAKOUT_CONT": (True,  0),
        "TREND_CONT":    (True,  0),
        "FADE":          (False, 99),
        "MIXED":         (False, 99),
    },
    "NY": {
        "RANGE_REVERT":  (False, 99),
        "BREAKOUT_CONT": (True,  0),
        "TREND_CONT":    (True,  0),
        "FADE":          (False, 99),
        "MIXED":         (False, 99),
    },
    "LATE_NY_FADE": {
        "RANGE_REVERT":  (False, 99),
        "BREAKOUT_CONT": (False, 10),
        "TREND_CONT":    (True,  5),
        "FADE":          (True,  0),
        "MIXED":         (False, 99),
    },
}

# Per-symbol allow-map overrides — schema {sym: {session: {setup: (allowed, bump)}}}.
# Empty default; populate after journal sweep finds a symbol that deviates
# from the global map.
SCSL_PER_SYMBOL_OVERRIDE = {}

# ═══ 2026-06-16 — SETUP INVALIDATOR (per-setup-type structural watcher) ═══
# Phase-2 module: agent/expert/setup_invalidator.py.
# At entry, brain tags each fill with setup_type + structural invalidation
# level. Every cycle, this watcher checks H1 CLOSED bars: if the structural
# condition fails → CLOSE at market. Runs ADJACENT to manage_trailing_sl
# but BEFORE ExitIntelligence / EarlyLossCut so structural fails take
# precedence over softer peak/trail logic.
#
# Default DISABLED + shadow-mode LIVE_CLOSE=False so we can A/B precision
# of the invalidation signal before arming real closes.
# Phase-3 will wire into brain.py _run_cycle at the line right after the
# manage_trailing_sl loop (~brain.py:1149) and before exit_intelligence.

# Master kill-switch. Default False until A/B precision burn-in completes.
SETUP_INVALIDATOR_ENABLED = True

# Log-only (shadow) mode — emit INVALIDATION_DETECTED journal rows but DO
# NOT close. Used to measure precision before arming live closes.
SETUP_INVALIDATOR_LIVE_CLOSE = False

# Strategy magics this watches. Manual trades (no magic) + scalp.r excluded.
SETUP_INVALIDATOR_WATCHED_MAGICS = {"momentum", "fvg", "sr"}

# Setup-type names to skip. Lets us blacklist a one-off broken classifier
# without disabling the whole component.
SETUP_INVALIDATOR_EXCLUDE_SETUPS = set()

# ATR multiplier added/subtracted from the structural level before declaring
# "broken". 0.10*ATR ≈ 1 tick on indices, prevents noise-wick triggers.
INVAL_STRUCT_BUFFER_ATR = 0.10

# Bars to wait for breakout follow-through (Williams "Long-Term Secrets"
# uses 3-bar confirmation).
BREAKOUT_FT_BARS = 3

# Follow-through must clear breakout level by this ATR multiple. 0.25 ATR
# ≈ a real expansion candle (not a noise-wick).
BREAKOUT_FT_ATR_MULT = 0.25

# Min favorable R reached before time-invalidation skips. Below this we
# call the setup dead.
DEFAULT_MIN_PROGRESS_R = 0.30

# Per-setup-type H1 time-stop in bars. Order Blocks decay fastest
# (institutional fills happen quickly); momentum allowed longest because
# trend lag is real.
SETUP_TIME_INVAL_BARS = {
    "WYCKOFF_SPRING":        6,
    "WYCKOFF_UPTHRUST":      6,
    "ORDER_BLOCK_LONG":      4,
    "ORDER_BLOCK_SHORT":     4,
    "FVG_RETEST":            6,
    "BREAKOUT":              3,
    "SWEEP_RECLAIM":         6,
    "MOMENTUM_CONTINUATION": 8,
}

# Per-symbol override of {time_inval_bars, min_progress_r}. Same shape as
# FVG_PARAM_OVERRIDES — populated after journal sweep finds a symbol that
# deviates from the global map.
SETUP_INVALIDATOR_PER_SYMBOL_OVERRIDES = {}

# Min seconds between invalidation evaluations for one symbol (de-bounce —
# H1 bar closes only ~once per 3600s anyway, so this just avoids fetching
# the candle frame on every tick).
SETUP_INVALIDATOR_CHECK_INTERVAL_SEC = 60

# Write setup_invalidated rows into trade journal for learning_engine
# feedback.
SETUP_INVALIDATOR_TAG_JOURNAL = True


# ═══════════════════════════════════════════════════════════════════════
# 2026-06-16 — DYNAMIC SL/TP — ATR + structure + regime (DynamicExitPlanner)
# Phase-2 module: agent/expert/Dynamic SL/TP — ...py
# Replaces flat ATR×mult SL + fixed-R TP with structure-anchored SL +
# 3-tier TP cascade (D1 swing → H4 supply/demand zone → 3R fallback) with
# regime-conditional TP2 multiplier. Default DISABLED — A/B with momentum
# book on demo first, then roll out to top-5 universe.
# Phase-3 will wire as a planning step BETWEEN entry-gate exit and
# executor.open_trade in agent/brain.py::_process_symbol (~line 2782).
# ═══════════════════════════════════════════════════════════════════════

# Master kill switch. When False, executor uses legacy fixed-R SL/TP.
DYNAMIC_EXIT_ENABLED = True

# Symbol whitelist; empty set = all symbols. Lets us roll out to top-5
# universe first (SP500/US2000/DJ30/USOUSD/XAUUSD).
DYNAMIC_EXIT_SYMBOLS = set()

# Which strategy paths the planner is wired into. FVG/SR keep their
# existing SL/TP unless explicitly added.
DYNAMIC_EXIT_STRATEGIES = {"momentum"}

# Structural lookbacks (closed bars only — index n-2 to mirror
# fvg_strategy convention).
DE_M15_SWING_LB = 20   # M15 bars for protective structural SL
DE_H4_SWING_LB  = 40   # H4 bars (synthesized from H1) for supply/demand
DE_D1_SWING_LB  = 30   # D1 bars (synthesized from H1) for TP2 anchor

# ATR clamps + structural buffer.
DE_ATR_FLOOR_MULT     = 1.0   # SL never tighter than 1.0 × ATR
DE_ATR_CAP_MULT       = 3.0   # SL never wider than 3.0 × ATR
DE_STRUCT_BUFFER_ATR  = 0.25  # ATR cushion past the swing extreme
DE_SPREAD_BUFFER_MULT = 1.5   # spread multiplier added to SL buffer

# TP geometry.
DE_TP1_R           = 1.5   # TP1 partial at 1.5R (spec)
DE_TP2_R_FLOOR     = 1.8   # TP2 never closer than 1.8R (must > TP1)
DE_TP2_R_CAP       = 5.0   # TP2 never further than 5R
DE_TP2_FALLBACK_R  = 3.0   # default TP2 when no D1/H4 within reach
DE_D1_SWING_MAX_R  = 5.0   # only target D1 swing if within this many R
DE_H4_ZONE_MAX_R   = 5.0   # only target H4 zone if within this many R
DE_RUNNER_R        = 5.0   # Sub2/runner wide TP target

# H4 supply/demand zone detection.
DE_ZONE_CLUSTER_ATR = 0.5  # bars within 0.5 ATR cluster into a zone
DE_ZONE_MIN_TOUCHES = 2    # ≥2 touches qualifies as zone

# Regime tilt on TP2 ONLY (SL stays structure-anchored). SL regime
# handling already lives in SYMBOL_ATR_SL_OVERRIDE_REGIME.
DE_REGIME_TP_MULT = {
    "trending": 1.20,
    "volatile": 1.15,
    "ranging":  0.80,
    "low_vol":  0.90,
    "unknown":  1.00,
}

# Magic offsets for Sub0/Sub1/Sub2 under DynamicExit (reserved range —
# no collision with 0..9 momentum, 1000 FVG, 2000 SR).
DE_MAGIC_OFFSETS = [3000, 3001, 3002]

# If planner throws, executor falls back to legacy fixed-R rather than
# skipping the trade. Honours [[feedback_no_skip_trades]].
DE_FALLBACK_ON_ERROR = True

# Emit one log line per planned entry with all distances + source tags
# for journal cross-ref.
DE_LOG_PLANS = True

# ════════════════════════════════════════════════════════════════════════
# 2026-06-16 — EXPERT_MODE master flag (ExpertGate orchestrator).
# ════════════════════════════════════════════════════════════════════════
# Wires all 11 expert components (news_v2, range_day, d1_struct, SCSL,
# order_block, wyckoff, tick_volume, conviction, ASAT/dynamic_sltp,
# setup_invalidator) into brain.py via a single sequenced gate that
# slots between Gate 3f (ICT sweep) and Gate 4 (position management).
#
# When True, the orchestrator runs each sub-component in its documented
# order. Individual sub-component enable flags (e.g. NEWS_BLACKOUT_V2_ENABLED,
# RANGE_DAY_CLASSIFIER_ENABLED, OB_ENABLED, WYCKOFF_ENABLED, …) STILL
# govern whether the sub-component activates inside the orchestrator —
# so flipping EXPERT_MODE_ENABLED True alone is safe: each piece stays
# off until its own flag is also flipped.
#
# Default True per 2026-06-16 user directive ("integrate all 11 expert
# components"). Set False to disable the orchestrator entry-side hook
# wholesale (sub-components also stay off via their own flags).
EXPERT_MODE_ENABLED = _envbool("EXPERT_MODE_ENABLED", True)

# Optional: cap how many REJECT reasons get logged per cycle to keep the
# journal readable. Each ExpertGate REJECT writes a single decision row
# tagged with the failing component name (e.g. "EXPERT_RANGE_DAY_SKIP").
EXPERT_MODE_LOG_REJECTS = _envbool("EXPERT_MODE_LOG_REJECTS", True)


# ════════════════════════════════════════════════════════════════════════
# 2026-06-18 — DRAGON_BOT_UPGRADE_v2 Tier 1 flags.
# ════════════════════════════════════════════════════════════════════════
# All ten Tier-1 items ship behind env flags that DEFAULT OFF (shadow mode).
# Live enablement is a single env var per item. Rollback: unset the env var.
#
# Implementation guarantees:
#   • Every new feature defaults OFF (shadow / data-collection mode)
#   • Every new feature is wrapped in try/except → fail-OPEN (no skip)
#   • Quality scorers (FVG, ConvictionTier) only DOWNSIZE — never REJECT
#     (preserves [[feedback_no_skip_trades]])
#   • The three exceptions where SKIP IS the correct policy are explicitly
#     called out: per-strategy kill switch (#1), per-strategy daily R cap
#     (#6), spread blowout (#9). All three are off-by-default.
# ════════════════════════════════════════════════════════════════════════


def _envint(key: str, default: int) -> int:
    """Env var override for integer flags. Matches _envbool / _envfloat style."""
    import os as _os
    v = _os.getenv(key)
    if v is None:
        return int(default)
    try:
        return int(v)
    except ValueError:
        return int(default)


# ── #1 PER-STRATEGY KILL SWITCH ─────────────────────────────────────────────
# Independent kill switches for each of momentum/fvg/sr. Auto-trips on
# (a) N consecutive losses in M hours, or (b) per-strategy daily R-sum
# breach (item #6). DB-persisted in `strategy_kill_switch` table created
# by AgentBrain._init_strategy_kill_switch_table().
#
# Default OFF for the 48h shadow-mode window so the table populates +
# auto-trip events are logged without taking action. Flip to True after
# Pareto histogram + 90d journal confirm thresholds are sane.
STRATEGY_KILL_SWITCH_ENABLED = _envbool("STRATEGY_KILL_SWITCH_ENABLED", True)  # 2026-06-19: ON per audit (3 consec losses arm 24h)
STRATEGY_KILL_CONSEC_LOSSES = _envint("STRATEGY_KILL_CONSEC_LOSSES", 3)
STRATEGY_KILL_CONSEC_WINDOW_HRS = _envfloat("STRATEGY_KILL_CONSEC_WINDOW_HRS", 4.0)
STRATEGY_KILL_AUTORESET_HRS = _envfloat("STRATEGY_KILL_AUTORESET_HRS", 6.0)

# ── #3 SPREAD-BLOWOUT PRE-ORDER SKIP ───────────────────────────────────────
# Live Vantage spreads on tier-1 news minutes spike 3-8×. A trade fired at
# 8× spread bleeds 1.5R before the market even moves. Exempt from no-skip
# policy because spread blowout = broker friction, not a quality scorer.
#
# Default OFF for 24h calibration. Flip after confirming healthy spreads
# never trip the threshold.
SPREAD_BLOWOUT_HARD_SKIP = _envbool("SPREAD_BLOWOUT_HARD_SKIP", False)
SPREAD_BLOWOUT_MULT = _envfloat("SPREAD_BLOWOUT_MULT", 2.5)

# ── #4 POSITION R-MULTIPLE LIVE TELEMETRY ──────────────────────────────────
# Pure read-side dashboard broadcast — no control-flow change. The exit
# manager already computes profit_r per cycle; we tap it and push to
# dashboard.v2_api.push_position_r() so the live tile shows "XAU +0.6R
# about to give back".
POSITION_R_TELEMETRY_ENABLED = _envbool("POSITION_R_TELEMETRY_ENABLED", True)
POSITION_R_TELEMETRY_MIN_INTERVAL_SEC = _envfloat(
    "POSITION_R_TELEMETRY_MIN_INTERVAL_SEC", 5.0)

# ── #5 NEWS PRE-FLATTEN CONFIRMATION BANNER ────────────────────────────────
# Today NEWS_FLATTEN closes positions 30min pre-event silently. This
# extension fires a Telegram/Slack/dashboard banner 45min before so the
# user has a heads-up. Skeleton ships with the alerter wiring; full
# Telegram/Slack delivery degrades gracefully if backends absent.
NEWS_PRE_FLATTEN_ALERT_ENABLED = _envbool("NEWS_PRE_FLATTEN_ALERT_ENABLED", True)
NEWS_PRE_FLATTEN_LEAD_MINUTES = _envfloat("NEWS_PRE_FLATTEN_LEAD_MINUTES", 45.0)

# ── #6 PER-STRATEGY DAILY R-CAP ────────────────────────────────────────────
# Each strategy shares global DAILY_LOSS_KILL_PCT=3% today. SR can fire
# 5 fast losses (-1.25R total = -0.3%) without tripping global. New per-
# strategy caps allow a single strategy to be paused for the day without
# nuking the others. Depends on #1 kill switch table.
PER_STRATEGY_DAILY_R_CAP_ENABLED = _envbool(
    "PER_STRATEGY_DAILY_R_CAP_ENABLED", True)  # 2026-06-19: ON per audit (caps -3R momentum, -2.5R FVG, -2R SR per day)
PER_STRATEGY_DAILY_R_KILL = {
    "momentum": _envfloat("STRATEGY_DAILY_R_KILL_MOMENTUM", -3.0),
    "fvg":      _envfloat("STRATEGY_DAILY_R_KILL_FVG", -2.5),
    "sr":       _envfloat("STRATEGY_DAILY_R_KILL_SR", -2.0),
}

# ── #6 PER-STRATEGY CONCURRENT POSITION CAP (extension, Diff 4) ────────────
# MAX_POSITIONS=999 is warn-only today. 3 strategies × 8 syms = up to 24
# concurrent positions theoretically. In a coordinated USD-strength move
# you could legitimately hold 7 correlated positions, none tripping
# cluster cap. New hard caps per strategy magic offset.
PER_STRATEGY_CAP_HARD_REJECT = _envbool("PER_STRATEGY_CAP_HARD_REJECT", False)
MAX_CONCURRENT_PER_STRATEGY = {
    "momentum": _envint("MAX_CONCURRENT_MOMENTUM", 5),
    "fvg":      _envint("MAX_CONCURRENT_FVG", 3),
    "sr":       _envint("MAX_CONCURRENT_SR", 3),
}

# ── #6 EQUITY-CURVE 3-TIER RISK SCALER ─────────────────────────────────────
# Replaces get_equity_slope() binary ±0.7×/+1.3× with 4-tier scaler:
#   GROWTH   (7d R-sum > +3.0R AND today_DD < 1%)   → 1.20× risk
#   NEUTRAL  (default)                              → 1.00× risk
#   DEFENSE  (7d R-sum < 0 OR today_DD > 1.5%)      → 0.60× risk
#   LOCKDOWN (7d R-sum < -5R OR today_DD > 2.5%)    → 0.30× risk
# Threshold values env-tunable. Composed multiplicatively with existing
# de-stack chain (drift, learning, portfolio) so the worst-case scaler
# wins (min protect × max boost ≤ MAX_RISK_PER_TRADE_PCT cap).
EQUITY_TIER_SCALER_ENABLED = _envbool("EQUITY_TIER_SCALER_ENABLED", True)  # 2026-06-19: ON per audit (DEFENSE@1.5%DD, LOCKDOWN@2.5%DD)
EQUITY_TIER_GROWTH_R = _envfloat("EQUITY_TIER_GROWTH_R", 3.0)
EQUITY_TIER_LOCKDOWN_R = _envfloat("EQUITY_TIER_LOCKDOWN_R", -5.0)
EQUITY_TIER_GROWTH_MULT = _envfloat("EQUITY_TIER_GROWTH_MULT", 1.20)
EQUITY_TIER_DEFENSE_MULT = _envfloat("EQUITY_TIER_DEFENSE_MULT", 0.60)
EQUITY_TIER_LOCKDOWN_MULT = _envfloat("EQUITY_TIER_LOCKDOWN_MULT", 0.30)
EQUITY_TIER_DEFENSE_DD_PCT = _envfloat("EQUITY_TIER_DEFENSE_DD_PCT", 1.5)
EQUITY_TIER_LOCKDOWN_DD_PCT = _envfloat("EQUITY_TIER_LOCKDOWN_DD_PCT", 2.5)

# ── #7 CONVICTION-TIER SHADOW MODE ─────────────────────────────────────────
# CONVICTION_TIERING_ENABLED is False today. Shadow mode lets us compute
# tier/size_mult/scorecard and journal it WITHOUT changing the live size.
# After 14d shadow data the dashboard can show "would-have" PnL split by
# tier — then user flips Phase B with per-symbol whitelist.
#
# When CONVICTION_TIER_SHADOW_ONLY=True, the brain stores the tier in
# decision telemetry but forces size_mult=1.0 (no behaviour change).
CONVICTION_TIER_SHADOW_ENABLED = _envbool("CONVICTION_TIER_SHADOW_ENABLED", False)
CONVICTION_TIER_SHADOW_ONLY = _envbool("CONVICTION_TIER_SHADOW_ONLY", True)

# ── #8 FVG QUALITY PRE-FILTER ──────────────────────────────────────────────
# Today _make_signal accepts any sweep+FVG passing the degenerate-stop
# guard. New 3-factor scorer (sweep_depth, fvg_displacement, reclaim
# strength) returns ∈ [0,1]. Applied as size_mult ×= 0.5 + 0.5*quality,
# bounded [0.5, 1.0]. Quality <0.2 still passes at 0.5× — honours
# [[feedback_no_skip_trades]].
FVG_QUALITY_FILTER_ENABLED = _envbool("FVG_QUALITY_FILTER_ENABLED", False)
FVG_QUALITY_MIN_DEPTH_ATR = _envfloat("FVG_QUALITY_MIN_DEPTH_ATR", 0.3)
FVG_QUALITY_MAX_DEPTH_ATR = _envfloat("FVG_QUALITY_MAX_DEPTH_ATR", 1.5)
FVG_QUALITY_MIN_DISP_ATR = _envfloat("FVG_QUALITY_MIN_DISP_ATR", 0.5)
FVG_QUALITY_MAX_DISP_ATR = _envfloat("FVG_QUALITY_MAX_DISP_ATR", 1.2)

# ── #9 VIX TERM-STRUCTURE REGIME GATE ──────────────────────────────────────
# Cboe term-structure macro filter. VIX9D/VIX/VIX3M relationship signals
# the realised-vol regime:
#   CONTANGO        (VIX3M > VIX > VIX9D)  normal       1.00× risk
#   BACKWARDATION   (VIX9D > VIX > VIX3M)  stress       0.50× risk
#   SPIKE           (VIX >= 30)            panic        HALT new entries
#   UNKNOWN         (data unavailable)     fail-open    1.00× risk
# Data sources tried in order (per VIX_DATA_SOURCE, falls back to other):
#   yfinance tickers '^VIX9D' '^VIX' '^VIX3M'
#   MT5 symbols     'VIX9D'  'VIX'  'VIX3M' (if exposed by broker)
# Cached VIX_CACHE_SECONDS (default 1h) to avoid hammering the feed.
# Wired into master_brain.evaluate_entry de-stack chain as protect_mults.
# Default OFF — shadow-observe via the self-test before enabling.
VIX_REGIME_GATE_ENABLED = _envbool("VIX_REGIME_GATE_ENABLED", False)
VIX_DATA_SOURCE = os.getenv("VIX_DATA_SOURCE", "yfinance")  # 'yfinance' | 'mt5'
VIX_SPIKE_THRESHOLD = _envfloat("VIX_SPIKE_THRESHOLD", 30.0)
VIX_CACHE_SECONDS = _envint("VIX_CACHE_SECONDS", 3600)
VIX_BACKWARDATION_RISK_MULT = _envfloat("VIX_BACKWARDATION_RISK_MULT", 0.50)

# ── #11 D1-BIAS UNIFIED (soft-filter: downsize, NOT reject) ────────────────
# Single source of truth for D1 trend bias used by momentum + SR + FVG.
# Module: agent/expert/d1_bias_unified.py — D1-resampled EMA(200) with a
# 0.3% neutral band. When direction opposes the (non-NEUTRAL) bias, the
# entry's risk is downsized by D1_BIAS_UNIFIED_DOWNSIZE (default 0.5x).
# Honors the "never skip trades" rule — the trade still fires.
# Default OFF: shadow-validate via journal before enabling in live.
D1_BIAS_UNIFIED_ENABLED = _envbool("D1_BIAS_UNIFIED_ENABLED", False)
D1_BIAS_UNIFIED_DOWNSIZE = _envfloat("D1_BIAS_UNIFIED_DOWNSIZE", 0.5)

# ── #10 VARIABLE-SPREAD BT MODEL ───────────────────────────────────────────
# Today BT uses static SPREAD[symbol]. Live spreads vary 1.5-2× by
# session and spike 3-8× on news. New flag enables per-(symbol, hour_utc)
# lookup in CostModel. Source: scripts/build_realized_spread_table.py
# samples mt5.symbol_info(sym).spread every 60s for 7d. Falls back to
# static SPREAD if table is missing (off-path, BT-only — no live impact).
BT_VARIABLE_SPREAD_ENABLED = _envbool("BT_VARIABLE_SPREAD_ENABLED", False)
BT_VARIABLE_SPREAD_MODE = "p50"  # "p50" | "p95" | "off"
BT_VARIABLE_SPREAD_TABLE_PATH = "data/spread_realized_per_session.json"


# ════════════════════════════════════════════════════════════════════════
# 2026-06-21 — Day-type routing (Dalton / Steidlmayer framework).
# ════════════════════════════════════════════════════════════════════════
# agent/expert/day_type_classifier.py classifies the current session as
# TREND_UP / TREND_DOWN / NORMAL / DOUBLE_DIST / UNKNOWN from the H1 OHLC
# series. ExpertGate.{_run_day_type_routing} translates the verdict into:
#   • TREND_UP    → momentum LONG +1 raw_score, momentum SHORT -1
#   • TREND_DOWN  → symmetric
#   • NORMAL      → SR signals 1.2× size, momentum 0.8×
#   • DOUBLE_DIST → ALL signals 0.7× (chop tax)
#   • UNKNOWN     → no-op (neutral)
#
# Default OFF — flip after 14d shadow journal confirms classifier matches
# operator intuition on the live universe.
DAY_TYPE_ROUTING_ENABLED = _envbool("DAY_TYPE_ROUTING_ENABLED", False)
DAY_TYPE_IB_BARS = _envint("DAY_TYPE_IB_BARS", 2)
DAY_TYPE_SCORE_BOOST = _envfloat("DAY_TYPE_SCORE_BOOST", 1.0)
DAY_TYPE_SCORE_PENALTY = _envfloat("DAY_TYPE_SCORE_PENALTY", 1.0)
DAY_TYPE_NORMAL_SR_MULT = _envfloat("DAY_TYPE_NORMAL_SR_MULT", 1.2)
DAY_TYPE_NORMAL_MOMENTUM_MULT = _envfloat("DAY_TYPE_NORMAL_MOMENTUM_MULT", 0.8)
DAY_TYPE_DOUBLE_DIST_MULT = _envfloat("DAY_TYPE_DOUBLE_DIST_MULT", 0.7)

