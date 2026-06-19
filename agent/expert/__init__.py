"""
agent.expert
════════════
Stateless, REPL-testable expert modules. Each module is a pure-function
helper that takes data (candle arrays, comp dicts, …) and returns a
decision dict — no MT5, no journal, no logger side effects.

Public exports
──────────────
  • classify_conviction(...)        from .conviction_tiering
  • classify_session(...)           from .session_setup
  • classify_setup(...)             from .session_setup
  • evaluate_session_setup(...)     from .session_setup (SCSL gate)
  • SETUP_TYPES                     from .session_setup
  • compute_asat_levels(...)        from "Asymmetric Structure-Aware Profit Targets (ASAT).py"
  • d1_structure_evaluate(...)      from "D1 Swing-Structure Bias (HH/HL + BOS/CHoCH) — ...py"
  • D1StructureBias                 (per-symbol memoizing wrapper)
  • get_d1_frame / find_swings / structure_bias / last_struct_event /
    d1_bias_verdict                 (D1 structure helpers, same module)
  • WyckoffSpringUpthrustStrategy   re-exported from agent.wyckoff_spring
  • detect_wyckoff_spring_upthrust  one-shot wrapper around the strategy
  • detect_order_block(...)         from "Order Block Detection + Retest Entry (ICT/SMC).py"
  • htf_bias(...)                   from same Order Block module
  • wilder_atr(...)                 from same Order Block module
  • rdc_classify_regime(...)        from "RangeDayClassifier (D1 ADX session-stamped regime gate).py"
  • rdc_stamp_regime(...)           same module — cache-aware stamp wrapper
  • rdc_evaluate(...)               same module — policy decision
  • rdc_classify_signal(...)        same module — source -> RDC class
  • rdc_clear_cache(...)            same module — day-rollover invalidation
  • classify_tv_setup(...)          from .tick_volume_gate (BREAKOUT vs MEAN_REVERT)
  • tick_volume_imbalance(...)      from .tick_volume_gate (ratio/MA/cur helper)
  • evaluate_tv_gate(...)           from .tick_volume_gate (Gate 3g entry point)
  • build_invalidation_spec(...)    from .setup_invalidator (entry-time metadata)
  • evaluate_invalidation_for_position(...) per-position decision
  • evaluate_setup_invalidations(...)portfolio-loop orchestrator
  • derive_momentum_setup_type(...) gate-trail → setup_type tag
  • bars_since_entry_h1(...)        helper used by brain integration
"""

from .conviction_tiering import classify_conviction
from .session_setup import (
    classify_session,
    classify_setup,
    evaluate as evaluate_session_setup,
    SETUP_TYPES,
)
from .tick_volume_gate import (
    classify_setup as classify_tv_setup,
    tick_volume_imbalance,
    evaluate_tv_gate,
)
from .discount_premium_zone import (
    compute_zone,
    evaluate_zone_gate,
)
from .setup_invalidator import (
    build_invalidation_spec,
    evaluate_invalidation_for_position,
    evaluate_invalidations as evaluate_setup_invalidations,
    derive_momentum_setup_type,
    bars_since_entry_h1,
    SETUP_TIME_INVAL_BARS_DEFAULT,
    INVAL_STRUCT_BUFFER_ATR_DEFAULT,
    BREAKOUT_FT_BARS_DEFAULT,
    BREAKOUT_FT_ATR_MULT_DEFAULT,
)

# ── ASAT — file name has spaces/parens per spec, load via importlib. ──
from importlib import util as _util
from pathlib import Path as _Path
import sys as _sys

_HERE = _Path(__file__).resolve().parent
_ASAT_PATH = _HERE / "Asymmetric Structure-Aware Profit Targets (ASAT).py"

compute_asat_levels = None  # type: ignore[assignment]
if _ASAT_PATH.exists():
    _spec = _util.spec_from_file_location("agent.expert._asat", str(_ASAT_PATH))
    if _spec is not None and _spec.loader is not None:
        _asat = _util.module_from_spec(_spec)
        try:
            _spec.loader.exec_module(_asat)  # type: ignore[arg-type]
            compute_asat_levels = getattr(_asat, "compute_asat_levels", None)
        except Exception:  # pragma: no cover — defensive
            compute_asat_levels = None

# ── Wyckoff Spring/Upthrust — canonical module lives at agent/wyckoff_spring.py
#    (mirrors agent/sweep_reclaim.py + agent/fvg_strategy.py layout). Re-export
#    here so brain.py / consumers can also do `from agent.expert import ...`.
WyckoffSpringUpthrustStrategy = None  # type: ignore[assignment]
detect_wyckoff_spring_upthrust = None  # type: ignore[assignment]
try:
    from agent.wyckoff_spring import (
        WyckoffSpringUpthrustStrategy as _WyckoffSpringUpthrustStrategy,
    )
    WyckoffSpringUpthrustStrategy = _WyckoffSpringUpthrustStrategy

    def detect_wyckoff_spring_upthrust(state, symbol, params=None):
        """Convenience wrapper: instantiate strategy + return evaluate() result.

        Use the class directly for repeated calls (it maintains per-symbol
        cooldown + bar-time dedupe). This wrapper is for one-shot tests.
        """
        return _WyckoffSpringUpthrustStrategy(state, params=params).evaluate(symbol)
except Exception:  # pragma: no cover — defensive
    pass


# ── Order Block detector ─────────────────────────────────────────────────
# Filename per spec is literally
#   "Order Block Detection + Retest Entry (ICT/SMC).py"
# The "/" makes "(ICT" a subdirectory and "SMC).py" the file inside it
# (this is the exact path the spec authored). Load via importlib.
_OB_PATH = _HERE / "Order Block Detection + Retest Entry (ICT" / "SMC).py"

detect_order_block = None  # type: ignore[assignment]
htf_bias           = None  # type: ignore[assignment]
wilder_atr         = None  # type: ignore[assignment]
if _OB_PATH.exists():
    _ob_spec = _util.spec_from_file_location("agent.expert._order_block", str(_OB_PATH))
    if _ob_spec is not None and _ob_spec.loader is not None:
        _ob_mod = _util.module_from_spec(_ob_spec)
        _sys.modules["agent.expert._order_block"] = _ob_mod
        try:
            _ob_spec.loader.exec_module(_ob_mod)  # type: ignore[arg-type]
            detect_order_block = getattr(_ob_mod, "detect_order_block", None)
            htf_bias           = getattr(_ob_mod, "htf_bias",           None)
            wilder_atr         = getattr(_ob_mod, "wilder_atr",         None)
        except Exception:  # pragma: no cover — defensive
            detect_order_block = None
            htf_bias           = None
            wilder_atr         = None


# ── D1 Swing-Structure Bias — file name has spaces / parens / em-dash ──
_D1_PATH = _HERE / (
    "D1 Swing-Structure Bias (HH/HL + BOS/CHoCH) — "
    "replaces H1-EMA200 trend proxy at Gate 3c.py"
)

get_d1_frame = None  # type: ignore[assignment]
find_swings = None  # type: ignore[assignment]
structure_bias = None  # type: ignore[assignment]
last_struct_event = None  # type: ignore[assignment]
d1_bias_verdict = None  # type: ignore[assignment]
d1_structure_evaluate = None  # type: ignore[assignment]
D1StructureBias = None  # type: ignore[assignment]

if _D1_PATH.exists():
    _spec_d1 = _util.spec_from_file_location("agent.expert._d1_structure", str(_D1_PATH))
    if _spec_d1 is not None and _spec_d1.loader is not None:
        _d1_mod = _util.module_from_spec(_spec_d1)
        _sys.modules["agent.expert._d1_structure"] = _d1_mod
        try:
            _spec_d1.loader.exec_module(_d1_mod)  # type: ignore[arg-type]
            get_d1_frame = getattr(_d1_mod, "get_d1_frame", None)
            find_swings = getattr(_d1_mod, "find_swings", None)
            structure_bias = getattr(_d1_mod, "structure_bias", None)
            last_struct_event = getattr(_d1_mod, "last_struct_event", None)
            d1_bias_verdict = getattr(_d1_mod, "d1_bias_verdict", None)
            d1_structure_evaluate = getattr(_d1_mod, "evaluate", None)
            D1StructureBias = getattr(_d1_mod, "D1StructureBias", None)
        except Exception:  # pragma: no cover — defensive
            pass


# ── Dynamic SL/TP — DynamicExitPlanner ──────────────────────────────────
# Filename per spec is literally
#   "Dynamic SL/TP — ATR + structure + regime (DynamicExitPlanner).py"
# The "/" creates "Dynamic SL" as a subdirectory and the rest as the file
# inside it (same loading pattern as the Order Block module).
_DE_PATH = _HERE / "Dynamic SL" / (
    "TP — ATR + structure + regime (DynamicExitPlanner).py"
)

compute_exit_plan       = None  # type: ignore[assignment]
DynamicExitPlanner      = None  # type: ignore[assignment]
ExitPlan                = None  # type: ignore[assignment]
de_find_swing_extreme   = None  # type: ignore[assignment]
de_find_next_swing      = None  # type: ignore[assignment]
de_find_h4_zone         = None  # type: ignore[assignment]
de_aggregate_ohlc       = None  # type: ignore[assignment]

if _DE_PATH.exists():
    _spec_de = _util.spec_from_file_location(
        "agent.expert._dynamic_exit", str(_DE_PATH))
    if _spec_de is not None and _spec_de.loader is not None:
        _de_mod = _util.module_from_spec(_spec_de)
        _sys.modules["agent.expert._dynamic_exit"] = _de_mod
        try:
            _spec_de.loader.exec_module(_de_mod)  # type: ignore[arg-type]
            compute_exit_plan     = getattr(_de_mod, "compute_plan",            None)
            DynamicExitPlanner    = getattr(_de_mod, "DynamicExitPlanner",      None)
            ExitPlan              = getattr(_de_mod, "ExitPlan",                None)
            de_find_swing_extreme = getattr(_de_mod, "find_swing_extreme",      None)
            de_find_next_swing    = getattr(_de_mod, "find_next_swing_extreme", None)
            de_find_h4_zone       = getattr(_de_mod, "find_h4_zone",            None)
            de_aggregate_ohlc     = getattr(_de_mod, "aggregate_ohlc",          None)
        except Exception:  # pragma: no cover — defensive
            pass


# ── RangeDayClassifier — file name has spaces / parens per spec ─────────
_RDC_PATH = _HERE / (
    "RangeDayClassifier (D1 ADX session-stamped regime gate).py"
)

rdc_classify_regime  = None  # type: ignore[assignment]
rdc_stamp_regime     = None  # type: ignore[assignment]
rdc_evaluate         = None  # type: ignore[assignment]
rdc_classify_signal  = None  # type: ignore[assignment]
rdc_clear_cache      = None  # type: ignore[assignment]

if _RDC_PATH.exists():
    _spec_rdc = _util.spec_from_file_location(
        "agent.expert._range_day_classifier", str(_RDC_PATH))
    if _spec_rdc is not None and _spec_rdc.loader is not None:
        _rdc_mod = _util.module_from_spec(_spec_rdc)
        _sys.modules["agent.expert._range_day_classifier"] = _rdc_mod
        try:
            _spec_rdc.loader.exec_module(_rdc_mod)  # type: ignore[arg-type]
            rdc_classify_regime  = getattr(_rdc_mod, "classify_regime",  None)
            rdc_stamp_regime     = getattr(_rdc_mod, "stamp_regime",     None)
            rdc_evaluate         = getattr(_rdc_mod, "evaluate",         None)
            rdc_classify_signal  = getattr(_rdc_mod, "classify_signal",  None)
            rdc_clear_cache      = getattr(_rdc_mod, "clear_cache",      None)
        except Exception:  # pragma: no cover — defensive
            pass


# ── Extended News Blackout v2 — 4-hour tier-1 event guard + flatten ──
# Filename per spec has spaces + parens. Load via importlib.
_NEWS_V2_PATH = _HERE / (
    "Extended News Blackout Windows v2 (4-hour tier-1 event guard + pre-event flatten).py"
)

get_blackout_state         = None  # type: ignore[assignment]
is_in_blackout_v2          = None  # type: ignore[assignment]
enforce_pre_event_flatten  = None  # type: ignore[assignment]

if _NEWS_V2_PATH.exists():
    _spec_news = _util.spec_from_file_location(
        "agent.expert._news_blackout_v2", str(_NEWS_V2_PATH))
    if _spec_news is not None and _spec_news.loader is not None:
        _news_mod = _util.module_from_spec(_spec_news)
        _sys.modules["agent.expert._news_blackout_v2"] = _news_mod
        try:
            _spec_news.loader.exec_module(_news_mod)  # type: ignore[arg-type]
            get_blackout_state        = getattr(_news_mod, "get_blackout_state",        None)
            is_in_blackout_v2         = getattr(_news_mod, "is_in_blackout",            None)
            enforce_pre_event_flatten = getattr(_news_mod, "enforce_pre_event_flatten", None)
        except Exception:  # pragma: no cover — defensive
            pass


__all__ = [
    "classify_conviction",
    "classify_session",
    "classify_setup",
    "evaluate_session_setup",
    "SETUP_TYPES",
    "compute_asat_levels",
    "WyckoffSpringUpthrustStrategy",
    "detect_wyckoff_spring_upthrust",
    "detect_order_block",
    "htf_bias",
    "wilder_atr",
    "get_d1_frame",
    "find_swings",
    "structure_bias",
    "last_struct_event",
    "d1_bias_verdict",
    "d1_structure_evaluate",
    "D1StructureBias",
    "rdc_classify_regime",
    "rdc_stamp_regime",
    "rdc_evaluate",
    "rdc_classify_signal",
    "rdc_clear_cache",
    "classify_tv_setup",
    "tick_volume_imbalance",
    "evaluate_tv_gate",
    "compute_zone",
    "evaluate_zone_gate",
    "build_invalidation_spec",
    "evaluate_invalidation_for_position",
    "evaluate_setup_invalidations",
    "derive_momentum_setup_type",
    "bars_since_entry_h1",
    "SETUP_TIME_INVAL_BARS_DEFAULT",
    "INVAL_STRUCT_BUFFER_ATR_DEFAULT",
    "BREAKOUT_FT_BARS_DEFAULT",
    "BREAKOUT_FT_ATR_MULT_DEFAULT",
    "compute_exit_plan",
    "DynamicExitPlanner",
    "ExitPlan",
    "de_find_swing_extreme",
    "de_find_next_swing",
    "de_find_h4_zone",
    "de_aggregate_ohlc",
    "get_blackout_state",
    "is_in_blackout_v2",
    "enforce_pre_event_flatten",
]
