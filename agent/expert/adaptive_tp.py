"""Momentum-adaptive TP resolver (per-symbol × regime × score-tier).

2026-06-19: Builds on the research finding that one TP ladder per symbol
under-fits the per-(regime, momentum-quality) winner peak_R distribution.
Buckets keyed by (regime, strong|weak):

    trending_strong  trending_weak
    ranging_strong   ranging_weak
    volatile_strong  volatile_weak
    low_vol_default

Quality split:
    score >= score_min (default 8) → strong
    score <  score_min               → weak

Source dict lives in auto_tuned.ADAPTIVE_TP_PER_SYM_REGIME — populated by
the research workflow per [[project_dragon_session_20260619]] (this commit).

Default fallbacks (TP1_R=1.5, TP2_R=3.0) match the symmetric mid-band of the
legacy SUB_TP_R [2,3,5] ladder collapsed to a 2-leg system.

Gated by config.ADAPTIVE_TP_ENABLED (default False — shadow rollout).
Callers should fall back to legacy adaptive_sub_tp_r() when disabled.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import logging

log = logging.getLogger("dragon.adaptive_tp")

# Default per [[user research findings]]: TP1 floor 0.8R, TP2 ceiling 4R.
# Mid-band fallback used when no per-cell tune exists.
DEFAULT_TP1_R: float = 1.5
DEFAULT_TP2_R: float = 3.0

# Floor + ceiling enforced regardless of dict value.
TP1_R_FLOOR: float = 0.8
TP2_R_FLOOR_ABS: float = 1.5     # absolute TP2 floor
TP2_R_FLOOR_MULT: float = 1.5    # TP2 >= TP2_R_FLOOR_MULT * TP1
TP2_R_CEILING: float = 4.0       # beyond this is structure-exit territory

# Default score threshold used when a cell does not specify one.
DEFAULT_STRONG_SCORE_MIN: float = 8.0

# Allowed regime tokens (defensive — brain passes lowercase strings).
_KNOWN_REGIMES = ("trending", "ranging", "volatile", "low_vol")


def _load_dict() -> Dict[str, Any]:
    """Import auto_tuned.ADAPTIVE_TP_PER_SYM_REGIME lazily so tests / repl
    don't pay the cost on import and missing-file conditions never crash
    the trading path."""
    try:
        import auto_tuned as _at  # type: ignore
        d = getattr(_at, "ADAPTIVE_TP_PER_SYM_REGIME", None)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}


def _clamp_tp(tp1_r: float, tp2_r: float) -> Tuple[float, float]:
    """Enforce TP1 >= floor, TP2 >= max(1.5×TP1, 1.5R), TP2 <= 4R."""
    try:
        t1 = float(tp1_r)
    except Exception:
        t1 = DEFAULT_TP1_R
    try:
        t2 = float(tp2_r)
    except Exception:
        t2 = DEFAULT_TP2_R
    if t1 < TP1_R_FLOOR:
        t1 = TP1_R_FLOOR
    floor_t2 = max(TP2_R_FLOOR_ABS, TP2_R_FLOOR_MULT * t1)
    if t2 < floor_t2:
        t2 = floor_t2
    if t2 > TP2_R_CEILING:
        t2 = TP2_R_CEILING
    return (t1, t2)


def _normalize_regime(regime: Optional[str]) -> Optional[str]:
    if regime is None:
        return None
    try:
        r = str(regime).strip().lower()
    except Exception:
        return None
    if r in _KNOWN_REGIMES:
        return r
    # Common aliases seen in journal/BT cells
    if r in ("trend",):
        return "trending"
    if r in ("range",):
        return "ranging"
    if r in ("vol", "high_vol"):
        return "volatile"
    if r in ("lowvol", "quiet"):
        return "low_vol"
    return None


def get_adaptive_tp(
    symbol: str,
    regime: Optional[str],
    score: Optional[float],
) -> Tuple[float, float]:
    """Resolve (tp1_r, tp2_r) for a momentum entry.

    Args:
        symbol: e.g. "XAUUSD".
        regime: one of {"trending","ranging","volatile","low_vol"} (or None).
        score: raw momentum score (typically 0-12 scale).

    Returns:
        (tp1_r, tp2_r) — always clamped to floors/ceiling.
        Falls back to (DEFAULT_TP1_R, DEFAULT_TP2_R) when no cell matches.
    """
    cfg = _load_dict()
    sym_cells = cfg.get(symbol) if isinstance(cfg, dict) else None
    if not isinstance(sym_cells, dict) or not sym_cells:
        return (DEFAULT_TP1_R, DEFAULT_TP2_R)

    reg = _normalize_regime(regime)

    # low_vol short-circuits to its own bucket (no strong/weak split).
    if reg == "low_vol":
        cell = sym_cells.get("low_vol_default")
        if isinstance(cell, dict):
            return _clamp_tp(cell.get("tp1_r", DEFAULT_TP1_R),
                             cell.get("tp2_r", DEFAULT_TP2_R))
        return (DEFAULT_TP1_R, DEFAULT_TP2_R)

    if reg is None:
        # No regime info — try a generic low_vol_default if present, else default.
        cell = sym_cells.get("low_vol_default")
        if isinstance(cell, dict):
            return _clamp_tp(cell.get("tp1_r", DEFAULT_TP1_R),
                             cell.get("tp2_r", DEFAULT_TP2_R))
        return (DEFAULT_TP1_R, DEFAULT_TP2_R)

    # Look up score_min from the strong cell to determine the split.
    strong_key = f"{reg}_strong"
    weak_key = f"{reg}_weak"
    strong_cell = sym_cells.get(strong_key) if isinstance(sym_cells.get(strong_key), dict) else None
    weak_cell = sym_cells.get(weak_key) if isinstance(sym_cells.get(weak_key), dict) else None

    threshold = DEFAULT_STRONG_SCORE_MIN
    if strong_cell and "score_min" in strong_cell:
        try:
            threshold = float(strong_cell["score_min"])
        except Exception:
            threshold = DEFAULT_STRONG_SCORE_MIN

    try:
        s = float(score) if score is not None else 0.0
    except Exception:
        s = 0.0

    pick = strong_cell if (s >= threshold and strong_cell) else weak_cell
    # Cascade: strong missing? try weak. weak missing? try low_vol_default. else default.
    if pick is None:
        pick = strong_cell or weak_cell or sym_cells.get("low_vol_default")
    if not isinstance(pick, dict):
        return (DEFAULT_TP1_R, DEFAULT_TP2_R)

    return _clamp_tp(pick.get("tp1_r", DEFAULT_TP1_R),
                     pick.get("tp2_r", DEFAULT_TP2_R))


__all__ = [
    "get_adaptive_tp",
    "DEFAULT_TP1_R",
    "DEFAULT_TP2_R",
    "TP1_R_FLOOR",
    "TP2_R_CEILING",
]
