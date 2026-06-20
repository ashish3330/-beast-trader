"""
agent.expert.vix_regime_gate
════════════════════════════
VIX term-structure regime detector + risk multiplier.

Theory (Cboe term-structure macro filter)
─────────────────────────────────────────
The relationship between VIX9D (9-day), VIX (30-day), and VIX3M (3-month)
implied vols characterises the realised-vol regime:

  • CONTANGO     VIX3M > VIX > VIX9D    normal risk-on            mult = 1.00
  • BACKWARDATION VIX9D > VIX > VIX3M   short-term stress         mult = 0.50
  • SPIKE        VIX > 30 (any state)   panic — halt new entries  mult = 0.00
  • UNKNOWN      data unavailable       fail-open                 mult = 1.00

Data hierarchy
──────────────
1. MT5 symbol 'VIX' / 'VIX9D' / 'VIX3M' (when VIX_DATA_SOURCE='mt5' and
   MetaTrader5 + symbols are available).
2. yfinance tickers '^VIX' / '^VIX9D' / '^VIX3M' (when yfinance installed).
3. UNKNOWN regime — risk_mult = 1.0 (fail-open, does not block trading).

Cache
─────
Term-structure quotes are cached for VIX_CACHE_SECONDS (default 3600s = 1h)
to avoid hammering the data source on every signal evaluation.

Wired into agent.master_brain.evaluate_entry via the de-stack chain:
  protect_mults.append(vix['risk_mult'])
  if vix['regime'] == 'SPIKE': result['approved'] = False; return

Default OFF behind config flag VIX_REGIME_GATE_ENABLED.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ── Config (env-tunable, sane defaults) ───────────────────────────────────
_DEFAULT_CACHE_SECS = 3600
_DEFAULT_SPIKE_VIX = 30.0
_DEFAULT_DATA_SOURCE = "yfinance"
_DEFAULT_BACKWARDATION_MULT = 0.5
_DEFAULT_CONTANGO_MULT = 1.0
_DEFAULT_SPIKE_MULT = 0.0  # not directly applied — SPIKE halts via brain check

# ── Pull from config when possible — gracefully fall back if absent ───────
try:  # pragma: no cover - import side path
    from config import (
        VIX_REGIME_GATE_ENABLED,
        VIX_DATA_SOURCE,
        VIX_SPIKE_THRESHOLD,
        VIX_CACHE_SECONDS,
        VIX_BACKWARDATION_RISK_MULT,
    )
except Exception:  # config flags optional → degrade gracefully
    VIX_REGIME_GATE_ENABLED = False
    VIX_DATA_SOURCE = _DEFAULT_DATA_SOURCE
    VIX_SPIKE_THRESHOLD = _DEFAULT_SPIKE_VIX
    VIX_CACHE_SECONDS = _DEFAULT_CACHE_SECS
    VIX_BACKWARDATION_RISK_MULT = _DEFAULT_BACKWARDATION_MULT


# ── Internal cache (module-level) ─────────────────────────────────────────
_CACHE: Dict[str, object] = {"ts": 0.0, "value": None}


def _now() -> float:
    return time.time()


def _try_mt5_quote(symbol: str) -> Optional[float]:
    """Best-effort spot read of a VIX-like MT5 symbol's last tick."""
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception:
        return None
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        if not info.visible:
            try:
                mt5.symbol_select(symbol, True)
            except Exception:
                pass
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        # bid+ask midpoint, fallback to last
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        last = float(getattr(tick, "last", 0.0) or 0.0)
        return last if last > 0 else None
    except Exception as e:
        log.debug("MT5 VIX-like read failed for %s: %s", symbol, e)
        return None


def _try_mt5_term_structure() -> Optional[Tuple[float, float, float]]:
    """Return (vix9d, vix, vix3m) from MT5 if all three present."""
    v9 = _try_mt5_quote("VIX9D")
    vx = _try_mt5_quote("VIX")
    v3 = _try_mt5_quote("VIX3M")
    if v9 and vx and v3:
        return float(v9), float(vx), float(v3)
    return None


def _try_yfinance_term_structure() -> Optional[Tuple[float, float, float]]:
    """Return (vix9d, vix, vix3m) from yfinance free Cboe feed."""
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    try:
        tickers = yf.Tickers("^VIX9D ^VIX ^VIX3M")
        out: Dict[str, float] = {}
        for sym in ("^VIX9D", "^VIX", "^VIX3M"):
            try:
                t = tickers.tickers.get(sym)
                if t is None:
                    return None
                # fast_info is cheapest; fall back to history
                px: Optional[float] = None
                try:
                    fi = getattr(t, "fast_info", None)
                    if fi is not None:
                        # fast_info supports dict-like access via attributes
                        px = float(
                            getattr(fi, "last_price", None)
                            or getattr(fi, "lastPrice", None)
                            or 0.0
                        ) or None
                except Exception:
                    px = None
                if not px:
                    hist = t.history(period="2d", interval="1d")
                    if hist is not None and not hist.empty:
                        px = float(hist["Close"].iloc[-1])
                if not px or px <= 0:
                    return None
                out[sym] = px
            except Exception as e:
                log.debug("yfinance read failed %s: %s", sym, e)
                return None
        return out["^VIX9D"], out["^VIX"], out["^VIX3M"]
    except Exception as e:
        log.debug("yfinance term-structure fetch failed: %s", e)
        return None


def _classify(
    v9d: float, vix: float, v3m: float, spike_thresh: float
) -> Tuple[str, float]:
    """Classify regime and emit risk multiplier."""
    # Spike supersedes term-structure: panic regime halts new entries.
    if vix is not None and vix >= float(spike_thresh):
        return "SPIKE", float(_DEFAULT_SPIKE_MULT)
    # Backwardation: short-end vol > long-end vol = stress
    if v9d > vix > v3m:
        return "BACKWARDATION", float(VIX_BACKWARDATION_RISK_MULT)
    # Contango: long-end > short-end = normal carry
    if v3m > vix > v9d:
        return "CONTANGO", float(_DEFAULT_CONTANGO_MULT)
    # Mixed / flat term-structure: treat as neutral but log it
    return "MIXED", float(_DEFAULT_CONTANGO_MULT)


def _fetch_term_structure() -> Optional[Tuple[float, float, float]]:
    """Source selection per VIX_DATA_SOURCE with auto-fallback."""
    src = str(VIX_DATA_SOURCE or _DEFAULT_DATA_SOURCE).lower()
    primary = _try_mt5_term_structure if src == "mt5" else _try_yfinance_term_structure
    secondary = _try_yfinance_term_structure if src == "mt5" else _try_mt5_term_structure
    try:
        ts = primary()
        if ts is not None:
            return ts
    except Exception as e:
        log.debug("VIX primary source %s failed: %s", src, e)
    try:
        ts = secondary()
        if ts is not None:
            return ts
    except Exception as e:
        log.debug("VIX secondary source failed: %s", e)
    return None


def get_vix_regime(force_refresh: bool = False) -> Dict[str, object]:
    """
    Public API. Returns {regime, risk_mult, vix9d, vix, vix3m, source, age_s}.

    Fail-open: any error path returns UNKNOWN with risk_mult = 1.0.
    """
    now = _now()
    cached = _CACHE.get("value")
    cached_ts = float(_CACHE.get("ts") or 0.0)
    cache_age = now - cached_ts
    if (
        not force_refresh
        and cached is not None
        and cache_age < float(VIX_CACHE_SECONDS or _DEFAULT_CACHE_SECS)
    ):
        out = dict(cached)  # type: ignore[arg-type]
        out["age_s"] = round(cache_age, 1)
        out["cached"] = True
        return out

    try:
        ts = _fetch_term_structure()
        if ts is None:
            result: Dict[str, object] = {
                "regime": "UNKNOWN",
                "risk_mult": 1.0,
                "vix9d": None,
                "vix": None,
                "vix3m": None,
                "source": "none",
                "age_s": 0.0,
                "cached": False,
                "reason": "no_data",
            }
            # Don't poison cache with UNKNOWN — let next call retry sooner.
            return result

        v9d, vix_spot, v3m = ts
        regime, mult = _classify(v9d, vix_spot, v3m, VIX_SPIKE_THRESHOLD)
        src_used = "mt5" if str(VIX_DATA_SOURCE).lower() == "mt5" else "yfinance"
        result = {
            "regime": regime,
            "risk_mult": float(mult),
            "vix9d": round(v9d, 3),
            "vix": round(vix_spot, 3),
            "vix3m": round(v3m, 3),
            "source": src_used,
            "age_s": 0.0,
            "cached": False,
        }
        _CACHE["value"] = result
        _CACHE["ts"] = now
        return result
    except Exception as e:
        log.debug("get_vix_regime fail-open: %s", e)
        return {
            "regime": "UNKNOWN",
            "risk_mult": 1.0,
            "vix9d": None,
            "vix": None,
            "vix3m": None,
            "source": "error",
            "age_s": 0.0,
            "cached": False,
            "reason": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────

def _self_test() -> int:
    print("== vix_regime_gate self-test ==")
    print("Config:")
    print(f"  VIX_REGIME_GATE_ENABLED   = {VIX_REGIME_GATE_ENABLED}")
    print(f"  VIX_DATA_SOURCE           = {VIX_DATA_SOURCE}")
    print(f"  VIX_SPIKE_THRESHOLD       = {VIX_SPIKE_THRESHOLD}")
    print(f"  VIX_CACHE_SECONDS         = {VIX_CACHE_SECONDS}")
    print(f"  VIX_BACKWARDATION_MULT    = {VIX_BACKWARDATION_RISK_MULT}")

    # 1. Pure classification sanity checks (no I/O).
    cases = [
        # (v9d, vix, v3m, expected_regime)
        (12.0, 14.0, 16.0, "CONTANGO"),
        (22.0, 18.0, 15.0, "BACKWARDATION"),
        (45.0, 35.0, 25.0, "SPIKE"),
        (15.0, 15.0, 15.0, "MIXED"),
        (10.0, 12.0, 11.0, "MIXED"),  # not strict contango
    ]
    fails = 0
    for v9, vx, v3, expected in cases:
        regime, mult = _classify(v9, vx, v3, _DEFAULT_SPIKE_VIX)
        ok = regime == expected
        if not ok:
            fails += 1
        print(
            f"  classify v9={v9} vix={vx} v3m={v3} "
            f"→ {regime} mult={mult}  expected={expected}  "
            f"{'OK' if ok else 'FAIL'}"
        )

    # 2. Public API contract — must never raise.
    out = get_vix_regime()
    assert isinstance(out, dict), "get_vix_regime must return dict"
    assert "regime" in out and "risk_mult" in out, "missing required keys"
    assert isinstance(out["risk_mult"], float), "risk_mult must be float"
    assert out["regime"] in {
        "CONTANGO",
        "BACKWARDATION",
        "SPIKE",
        "MIXED",
        "UNKNOWN",
    }, f"unexpected regime {out['regime']}"
    print(f"  get_vix_regime() → {out}")

    # 3. Cache hit on second call (only if fetch succeeded).
    out2 = get_vix_regime()
    if out.get("regime") != "UNKNOWN":
        assert out2.get("cached") is True, "second call should be cached"
        print("  cache hit OK")
    else:
        print("  (no data available — UNKNOWN fail-open verified)")

    print(f"\nResult: {'PASS' if fails == 0 else f'FAIL ({fails} bad classifications)'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print("Run with --self-test to validate.")
