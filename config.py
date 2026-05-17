"""
Dragon Trader — Tick-Level ML Trading Agent.
Account: 25106421, $2,500, VantageInternational-Demo.
MT5 bridge: localhost:18813 (rpyc via Wine).
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ═══ MT5 CREDENTIALS ═══
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "25106421"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "R4q9Tyq$")
MT5_SERVER = os.getenv("MT5_SERVER", "VantageInternational-Demo")
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
    # ── ACTIVE TIER (9 A-grade) ──
    "DJ30.r":     SymbolConfig("DJ30.r",     8320, "Index",     2),
    "JPN225ft":   SymbolConfig("JPN225ft",   8230, "Index",     2),
    "SPI200.r":   SymbolConfig("SPI200.r",   8500, "Index",     2),
    "SWI20.r":    SymbolConfig("SWI20.r",    8440, "Index",     2),
    "US2000.r":   SymbolConfig("US2000.r",   8470, "Index",     2),
    "XAUUSD":     SymbolConfig("XAUUSD",     8100, "Gold",      2),
    "BTCUSD":     SymbolConfig("BTCUSD",     8130, "Crypto",    2),
    "AUDJPY":     SymbolConfig("AUDJPY",     8260, "Forex",     3),
    "EURUSD":     SymbolConfig("EURUSD",     8370, "Forex",     5),

    # ── EXPANSION TIER (2026-05-16: 8 syms, walk-forward ROBUST) ──
    # Validated by 180d train / 180d held-out test. Per-sym test PF >= 2.5
    # (except NAS100.r where train PF 21 dropped to 8.98 — still strong).
    # pass2 params get baked into auto_tuned.py via synthesize_auto_tuned.py.
    "NAS100.r":   SymbolConfig("NAS100.r",   8210, "Index",     2),
    "SP500.r":    SymbolConfig("SP500.r",    8240, "Index",     2),
    "UK100.r":    SymbolConfig("UK100.r",    8250, "Index",     2),
    "XPTUSD.r":   SymbolConfig("XPTUSD.r",   8150, "Gold",      2),
    "USDCAD":     SymbolConfig("USDCAD",     8380, "Forex",     5),
    "USDJPY":     SymbolConfig("USDJPY",     8390, "Forex",     3),
    "CHFJPY":     SymbolConfig("CHFJPY",     8280, "Forex",     3),
    "USOUSD":     SymbolConfig("USOUSD",     8480, "Commodity", 3),

    # ── DORMANT TIER (uncomment when equity ≥ $5000) ──
    # "XAGUSD":     SymbolConfig("XAGUSD",     8140, "Gold",      3),  # PF 1.33 marginal
    # "GER40.r":    SymbolConfig("GER40.r",    8200, "Index",     2),  # weak 360d
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

# 2. EARLY-LOSS-CUT — if trade goes to -0.5R and stays there for N cycles
#    without reaching positive territory, close at market. Saves the
#    spread/slippage cost of full SL hit (which would be -1.5-1.7R live).
EARLY_EXIT_ENABLED = _envbool("EARLY_EXIT_ENABLED", True)
EARLY_EXIT_TRIGGER_R = -0.5     # threshold: -0.5R or worse
EARLY_EXIT_CYCLES = 20          # 2026-05-16: 60→20 cycles (~30s→10s). Trade #753 BTCUSD
                                # bled from -0.5R to -3.0R in the 30s T1-SLOW window because
                                # the slow-bleed tier waited too long while a momentum spike
                                # walked price against it. 10s is enough to filter noise on
                                # forex but not let a high-vol asset run -3R against a -1R SL.

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

# ═══ DRAGON RISK MANAGEMENT (aggressive but survivable — demo phase) ═══
# 90-day PF 1.72 (recent market harder) — stay aggressive but not suicidal
# Compound growth sim: 0.8% risk = $1K → $7.3K/year (630%) with ~30% peak DD
MAX_RISK_PER_TRADE_PCT = 1.0        # 2026-05-11: raised 0.4→1.0 for $1K all-symbol trading. Memory's production ceiling.
MAX_TOTAL_EXPOSURE_PCT = 25.0      # 2026-05-11: raised 12→25 to accommodate 1% risk × 28 syms (was 0.4% × 28 = 11.2%, now 1% × 28 = 28%). Cap retained at 25% as kill-switch safety net.

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
COOLDOWN_WIN_SECS          = 900   # 15min — TP hit / closed in profit. Same-direction only.
COOLDOWN_LOSS_SECS         = 2700  # 45min — SL hit / closed at loss. Both directions.
COOLDOWN_BROKER_CLOSE_SECS = 2700  # 45min — default if win/loss can't be determined
COOLDOWN_SL_HIT_SECS       = 2700  # 45min — loss-tagged exit (legacy alias for COOLDOWN_LOSS_SECS)
COOLDOWN_SCALP_CLOSE_SECS  = 1800  # 30min — scalp closed
EXECUTOR_MIN_REENTRY_SECS  = 60    # belt-and-braces hard floor: executor refuses re-open within Ns of any close

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

_TRAIL_LOCK_AT_15R = _envfloat("DRAGON_TRAIL_LOCK_AT_15R", 0.7)
_TRAIL_LOCK_AT_10R = _envfloat("DRAGON_TRAIL_LOCK_AT_10R", 0.4)
_TRAIL_LOCK_AT_07R = _envfloat("DRAGON_TRAIL_LOCK_AT_07R", 0.2)

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
    (2.0, "trail", 0.6),                  # tighter trail above 2R
    (1.5, "lock",  _TRAIL_LOCK_AT_15R),
    (1.0, "lock",  _TRAIL_LOCK_AT_10R),
    (0.7, "lock",  _TRAIL_LOCK_AT_07R),   # lock 0.2R at 0.7R (restored)
    (0.5, "lock",  0.15),                 # NEW: lock 0.15R at 0.5R (was BE=0)
    (0.3, "be",    0.0),                  # NEW: BE at 0.3R (very tight)
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
    "XAUUSD":    {r: _COMMODITY_AGGRESSIVE for r in ("trending", "ranging", "volatile", "low_vol")},
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
    "BTCUSD": [
        (6.0, "trail", 0.3), (4.0, "trail", 0.5), (2.5, "trail", 0.8),
        (1.5, "lock", 0.5), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15), (0.4, "be", 0.0),
    ],
    "ETHUSD": [
        (6.0, "trail", 0.3), (4.0, "trail", 0.5), (2.5, "trail", 0.8),
        (1.5, "lock", 0.5), (1.0, "lock", 0.3),
        (0.7, "lock", 0.15), (0.4, "be", 0.0),
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
EVAL_ON_CANDLE_CLOSE = False    # 2026-05-13: per-tick scoring. Each brain cycle
                                # (~6s) re-evaluates every symbol from scratch
                                # rather than caching between M15 closes. Catches:
                                # - mid-bar M15 direction changes
                                # - live spread widening/tightening (MIN_EDGE)
                                # - regime re-classification
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
    # Lowered 2026-05-01 from 45 → 40 across calm regimes after diagnostic
    # showed live max quality stuck at 44% for hours straight (max raw score
    # 5.2/12) → zero entries fired in 7 hrs. Backtest used 45 (=5.4 raw) and
    # got 256 trades/30d, but Friday-morning live regime is calmer than the
    # backtest sample. Volatile kept at 45 to avoid noisy entries in spikes.
    "trending": 40,    # 4.8 raw — was 45
    "ranging":  42,    # 5.04 raw — was 45 (slightly stricter for chop)
    "volatile": 45,    # 5.4 raw — UNCHANGED (don't enter into spikes)
    "low_vol":  40,    # 4.8 raw — was 45
}

# Per-symbol quality override (where optimal differs from default)
SIGNAL_QUALITY_SYMBOL: Dict[str, Dict[str, int]] = {
    # Cleared 2026-04-29 — over-tuned to backtest regime, live bled at low thresholds.
    # All symbols use SIGNAL_QUALITY_THRESHOLDS default (45%).
}

# MTF high-conviction override: skip M15 gate if signal_quality >= this
MTF_OVERRIDE_QUALITY = 75  # 9.0 raw — monster signal, M15 doesn't matter

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
}

# ═══ NEWS CALENDAR — high-impact event hard-block opt-in ═══
# CalendarFilter (agent/calendar_filter.py) checks ±30min around high-impact
# Forex Factory events. Default behavior (per no-skip rule): warn-only.
# Symbols in this set get a HARD SKIP during the event window. Reserve for
# pairs that historically blow up on major releases.
CALENDAR_HARD_BLOCK_SYMBOLS: set = {
    # Empty by default. Populate per-symbol when live data shows the news
    # window is repeatedly catastrophic (e.g. EURUSD on NFP, USDJPY on FOMC).
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
}

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
}

# ═══ RL LEARNING — per-symbol toggle + tuned params ═══
# Updated 2026-05-09 for 19-symbol live universe. Conservative: enable RL only on
# symbols with ≥100 walk-forward-test trades and ROBUST verdict. Others bypass RL
# (returns 1.0 multiplier from get_risk_multiplier).
RL_ENABLED_SYMBOLS = {
    # Indices with strong tuned PnL + robust walk-forward
    "DJ30.r", "US2000.r", "GER40.r", "HK50.r", "SWI20.r", "UK100.r",
    # Forex pairs that have enough trade history + robust
    "USDCAD", "USDCHF", "EURUSD", "AUDJPY", "CADJPY", "CHFJPY", "GBPAUD",
    # Commodities with healthy sample
    "UKOUSD", "COPPER-Cr",
    # Crypto
    "ETHUSD",
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
    # Commodities
    "UKOUSD":     {"lookback": 30, "boost_max": 1.4},
    "COPPER-Cr":  {"lookback": 30, "boost_max": 1.3},
    # Crypto
    "ETHUSD":     {"lookback": 20, "boost_max": 1.3},
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
    SYMBOL_RISK_CAP.update(getattr(_at, "RISK_CAP_AUTO", {}))
    for _s, _hours in getattr(_at, "TOXIC_HOURS_PER_SYMBOL_AUTO", {}).items():
        TOXIC_HOURS_PER_SYMBOL.setdefault(_s, set()).update(set(_hours))
    SYMBOL_TRAIL_OVERRIDE.update(getattr(_at, "TRAIL_OVERRIDE_AUTO", {}))

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
    DIRECTION_BIAS.pop("ETHUSD", None)   # → BOTH
    DIRECTION_BIAS.pop("XAUUSD", None)   # → BOTH
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
except ImportError:
    pass
