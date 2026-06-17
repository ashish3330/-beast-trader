"""
Dragon Trader — Structured DECISION-log reason codes.

Tier 1 #2 of DRAGON_BOT_UPGRADE_v2.md. Replaces free-form `GATE=` strings with
a stable enum so the dashboard can compute a per-gate rejection Pareto and the
per-quarter compounded explainability ROI surface (+0.05 PF baseline target).

Codes are grouped by gate category so the histogram tile can collapse easily:

  R_GATE_*        — early structural gates (regime, position, bar re-entry)
  R_QUALITY_*     — score / quality / confirmation gates
  R_FILTER_*      — TF / trend / direction / cooldown / dir-bias gates
  R_CHASE_*       — chase / extreme / fib gates
  R_ICT_*         — ICT sweep gate
  R_VWAP_*        — VWAP cross gate
  R_EXPERT_*      — ExpertGate orchestrator REJECTs (per-sub-component)
  R_NEWS_*        — calendar / news blackout gates
  R_M15_*         — M15 cascade gates
  R_META_*        — ML meta-label / master-brain gates
  R_EV_*          — EV / friction gates
  R_BAR_*         — H1 / M15 bar gates
  R_SPREAD_*      — spread blowout / spike (Tier 1 #3)
  R_STRATEGY_*    — per-strategy kill switch / cap (Tier 1 #1, #6)
  R_PER_*         — per-strategy resource caps

Every code is a constant str so it shows up unchanged in logs / journal / dash.
Aliases for legacy gate strings let us refactor the 28 call sites without
breaking the dashboard rejection histogram during the transition window.

Honours [[feedback_no_skip_trades]] — these are CODES not POLICIES. The brain
still decides whether a code blocks vs warns vs downsizes. Quality scorers
that downsize (FVG, ConvictionTier) emit telemetry codes (R_QUALITY_*) but
DO NOT use them as reject reasons.
"""
from __future__ import annotations

# ── Early structural gates ──────────────────────────────────────────────────
R_REGIME_UNKNOWN       = "R_REGIME_UNKNOWN"        # regime classifier returned unknown
R_REGIME_FLAT_BLOCK    = "R_REGIME_FLAT_BLOCK"     # both directions blocked in this regime
R_POSITION_OPEN        = "R_POSITION_OPEN"         # already open in this symbol/dir
R_FVG_POSITION_OPEN    = "R_FVG_POSITION_OPEN"     # FVG position active — yielding
R_BAR_REENTRY          = "R_BAR_REENTRY"           # entry already taken on this H1 bar
R_LATE_MOMENTUM        = "R_LATE_MOMENTUM"         # raw_score after recent <floor in window — late confirm

# ── Quality / confirmation gates ────────────────────────────────────────────
R_BELOW_MIN_SCORE      = "R_BELOW_MIN_SCORE"       # quality < min_quality
R_CONFIRM_MISSING      = "R_CONFIRM_MISSING"       # N-of-5 confirmation under threshold
R_MIN_EDGE_REJECT      = "R_MIN_EDGE_REJECT"       # friction > tier edge

# ── Filter / direction gates ────────────────────────────────────────────────
R_DIR_BIAS             = "R_DIR_BIAS"              # SHORT-only / LONG-only symbol blocking other dir
R_COOLDOWN             = "R_COOLDOWN"              # per-direction cooldown active
R_RL_SKIP              = "R_RL_SKIP"               # RL-driven skip
R_TOXIC_HOUR           = "R_TOXIC_HOUR"            # per-sym blocked UTC hour
R_TREND_FILTER         = "R_TREND_FILTER"          # counter-trend vs H1 EMA200
R_MTF_CASCADE          = "R_MTF_CASCADE"           # 2+ higher TFs oppose

# ── Range / chase / fib gates ───────────────────────────────────────────────
R_RANGE_EXTREME        = "R_RANGE_EXTREME"         # at range boundary in ranging regime
R_FIB_ZONE             = "R_FIB_ZONE"              # outside the fib retracement zone
R_LONG_CHASE_TOP       = "R_LONG_CHASE_TOP"        # LONG at top of pos_4h
R_SHORT_CHASE_BOTTOM   = "R_SHORT_CHASE_BOTTOM"    # SHORT at bottom of pos_4h

# ── ICT gates ───────────────────────────────────────────────────────────────
R_ICT_NO_SWEEP         = "R_ICT_NO_SWEEP"          # no liquidity sweep+reclaim in window

# ── VWAP gates ──────────────────────────────────────────────────────────────
R_VWAP_REJECT          = "R_VWAP_REJECT"           # LONG below / SHORT above session VWAP

# ── Expert orchestrator gates (per sub-component) ───────────────────────────
R_EXPERT_GENERIC       = "R_EXPERT_GENERIC"        # ExpertGate sub-component reject (no specific tag)
R_EXPERT_D1_BIAS       = "R_EXPERT_D1_BIAS"        # D1 structure bias opposing
R_EXPERT_NEWS_V2       = "R_EXPERT_NEWS_V2"        # 4h tier-1 event window
R_EXPERT_RANGE_DAY     = "R_EXPERT_RANGE_DAY"      # range day classifier
R_EXPERT_SESSION       = "R_EXPERT_SESSION"        # session setup gate
R_EXPERT_TV_VOLUME     = "R_EXPERT_TV_VOLUME"      # tick volume / order flow imbalance
R_EXPERT_OB            = "R_EXPERT_OB"             # order block
R_EXPERT_WYCKOFF       = "R_EXPERT_WYCKOFF"        # wyckoff spring
R_EXPERT_CONVICTION    = "R_EXPERT_CONVICTION"     # conviction tier B reject (post-shadow)
R_EXPERT_INVALIDATOR   = "R_EXPERT_INVALIDATOR"    # setup invalidator pre-set

# ── Calendar / news ─────────────────────────────────────────────────────────
R_CALENDAR             = "R_CALENDAR"              # static calendar ±30min blackout

# ── M15 / meta / master ─────────────────────────────────────────────────────
R_M15_DISAGREE         = "R_M15_DISAGREE"          # M15 dir != H1 dir
R_META_REJECT          = "R_META_REJECT"           # ML meta-label below floor
R_MASTER_REJECT        = "R_MASTER_REJECT"         # MasterBrain rejected

# ── EV ──────────────────────────────────────────────────────────────────────
R_EV_REJECT            = "R_EV_REJECT"             # expected-value after costs < floor

# ── Tier-1 NEW gates ────────────────────────────────────────────────────────
R_SPREAD_BLOWOUT       = "R_SPREAD_BLOWOUT"        # live spread > N× baseline (#3)
R_STRATEGY_KILLED      = "R_STRATEGY_KILLED"       # per-strategy kill switch tripped (#1)
R_STRATEGY_DAILY_R_CAP = "R_STRATEGY_DAILY_R_CAP"  # per-strategy daily R-cap (#6)
R_STRATEGY_CAP         = "R_STRATEGY_CAP"          # per-strategy concurrent position cap (Tier 1 ext)

# ── Conviction-tier shadow telemetry codes (NOT reject reasons) ─────────────
# These are emitted alongside an APPROVE/SHIP decision when shadow mode is
# active so the dashboard can correlate tier vs realized R post-trade.
R_CONV_SHADOW_A_PLUS   = "R_CONV_SHADOW_A_PLUS"    # shadow tier observation: A+
R_CONV_SHADOW_B_PLUS   = "R_CONV_SHADOW_B_PLUS"    # shadow tier observation: B+
R_CONV_SHADOW_B        = "R_CONV_SHADOW_B"         # shadow tier observation: B (would-block)

# ── FVG quality scorer telemetry (NOT reject — downsize per no-skip) ────────
R_FVG_QUALITY_LOW      = "R_FVG_QUALITY_LOW"       # quality < 0.2 — still passes at 0.5×
R_FVG_QUALITY_MID      = "R_FVG_QUALITY_MID"       # 0.2 ≤ quality < 0.6
R_FVG_QUALITY_HIGH     = "R_FVG_QUALITY_HIGH"      # 0.6 ≤ quality

# ── Equity-curve risk tier telemetry ────────────────────────────────────────
R_EQUITY_TIER_LOCKDOWN = "R_EQUITY_TIER_LOCKDOWN"  # 7d R < -5R OR DD > 2.5%
R_EQUITY_TIER_DEFENSE  = "R_EQUITY_TIER_DEFENSE"   # 7d R < 0 OR DD > 1.5%
R_EQUITY_TIER_NEUTRAL  = "R_EQUITY_TIER_NEUTRAL"   # default
R_EQUITY_TIER_GROWTH   = "R_EQUITY_TIER_GROWTH"    # 7d R > +3R AND DD < 1%

# ── ConvictionTier (Phase B/C — when shadow is over) ────────────────────────
R_CONV_A_PLUS          = "R_CONV_A_PLUS"
R_CONV_B_PLUS          = "R_CONV_B_PLUS"
R_CONV_B               = "R_CONV_B"

# ── Backwards-compat aliases ────────────────────────────────────────────────
# Map free-form GATE= strings emitted today by brain.py:_log_decision() to the
# new canonical codes. The brain helper `canonicalize_gate()` returns the
# canonical code if the legacy is in this dict, otherwise echoes the input
# (so unknown legacy strings continue to flow uninterrupted — fail-open).
_LEGACY_TO_CANONICAL = {
    "REGIME_UNKNOWN":       R_REGIME_UNKNOWN,
    "REGIME_FLAT_BLOCK":    R_REGIME_FLAT_BLOCK,
    "BELOW_MIN":            R_BELOW_MIN_SCORE,
    "BELOW_MIN_SCORE":      R_BELOW_MIN_SCORE,
    "POSITION_OPEN":        R_POSITION_OPEN,
    "FVG_POSITION_OPEN":    R_FVG_POSITION_OPEN,
    "BAR_REENTRY":          R_BAR_REENTRY,
    "LATE_MOMENTUM":        R_LATE_MOMENTUM,
    "DIR_BIAS":             R_DIR_BIAS,
    "COOLDOWN":             R_COOLDOWN,
    "RL_SKIP":              R_RL_SKIP,
    "CALENDAR":             R_CALENDAR,
    "TREND_FILTER":         R_TREND_FILTER,
    "MTF_CASCADE":          R_MTF_CASCADE,
    "RANGE_EXTREME":        R_RANGE_EXTREME,
    "FIB_ZONE":             R_FIB_ZONE,
    "LONG_CHASE_TOP":       R_LONG_CHASE_TOP,
    "SHORT_CHASE_BOTTOM":   R_SHORT_CHASE_BOTTOM,
    "ICT_NO_SWEEP":         R_ICT_NO_SWEEP,
    "VWAP_REJECT":          R_VWAP_REJECT,
    "M15_DISAGREE":         R_M15_DISAGREE,
    "META_REJECT":          R_META_REJECT,
    "MASTER_REJECT":        R_MASTER_REJECT,
    "MIN_EDGE_REJECT":      R_MIN_EDGE_REJECT,
    "EV_REJECT":            R_EV_REJECT,
    "CONFIRM_MISSING":      R_CONFIRM_MISSING,
}


def canonicalize_gate(gate: str) -> str:
    """Return the canonical R_* code for a legacy gate string, or echo input.

    Fail-open: unknown gate strings continue to flow through the dashboard
    rejection histogram unchanged. This lets us refactor the 28 brain.py
    call sites in any order without breaking the histogram during transit.
    """
    if not isinstance(gate, str):
        return str(gate)
    # Strip dynamic suffixes (e.g. "TOXIC_HOUR_17") so they cluster together
    base = gate.split("[")[0].split(":")[0].strip()
    # Toxic hour: brain.py emits "TOXIC_HOUR_<H>" → strip the hour
    if base.startswith("TOXIC_HOUR"):
        return R_TOXIC_HOUR
    # Expert REJECT prefixes
    if base.startswith("EXPERT_"):
        rest = base[len("EXPERT_"):]
        if rest.startswith("D1") or rest.startswith("D1_"):
            return R_EXPERT_D1_BIAS
        if rest.startswith("NEWS"):
            return R_EXPERT_NEWS_V2
        if rest.startswith("RANGE"):
            return R_EXPERT_RANGE_DAY
        if rest.startswith("SESSION"):
            return R_EXPERT_SESSION
        if rest.startswith("TV") or rest.startswith("TICK"):
            return R_EXPERT_TV_VOLUME
        if rest.startswith("OB") or rest.startswith("ORDER_BLOCK"):
            return R_EXPERT_OB
        if rest.startswith("WYCKOFF"):
            return R_EXPERT_WYCKOFF
        if rest.startswith("CONV") or rest.startswith("TIER_"):
            return R_EXPERT_CONVICTION
        if rest.startswith("INVALID") or rest.startswith("SETUP_INVALID"):
            return R_EXPERT_INVALIDATOR
        return R_EXPERT_GENERIC
    return _LEGACY_TO_CANONICAL.get(base, gate)


# Export every R_* code by introspection so unit tests / dashboards can
# discover the full vocabulary.
ALL_CODES = sorted([k for k in globals().keys() if k.startswith("R_")])


__all__ = ["ALL_CODES", "canonicalize_gate"] + ALL_CODES
