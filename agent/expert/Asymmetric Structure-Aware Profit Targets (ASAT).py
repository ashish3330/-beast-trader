"""ASAT — Asymmetric Structure-Aware Profit Targets
─────────────────────────────────────────────────
Pure-function module that replaces the flat per-symbol SUB_TP_R ladder for
momentum entries with a structure-aware TP1 (fixed 1.5R partial) + TP2
anchored to the next significant D1 swing extreme. SL is widened (when
needed) to sit just beyond the protective structure swing so the stop is
INVALIDATION-based, not noise-based.

Literature anchors:
  * ICT (Inner Circle Trader) — sweep prior swing → target opposite-side
    liquidity (BSL = D1 swing highs, SSL = swing lows).
  * Wyckoff Phase D/E — markup leg projects from spring (LONG) toward prior
    distribution high; markdown from upthrust toward prior support.
  * SMC (Smart Money Concepts) — "draw on liquidity" = nearest unmitigated
    higher-TF swing.
  * Bourgade & Hassani 2009.08821 — fixed-R exits leave EV on the table
    when a structural draw is present.

Module is pure functions: takes pandas OHLC frames + scalars, returns a
dict of price levels (or None). No class state, no side effects (except
optional logging).
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
#  Config import (lazy / soft — file is meant to be importable in tests
#  even if config.py path is not set up).
# ──────────────────────────────────────────────────────────────────────
try:  # pragma: no cover — config in production env
    import config as _cfg  # type: ignore
except Exception:  # pragma: no cover
    _cfg = None  # type: ignore

log = logging.getLogger("asat")


def _cfg_get(name: str, default):
    """Read a config flag with a safe default — keeps tests independent
    of config.py wiring."""
    if _cfg is None:
        return default
    return getattr(_cfg, name, default)


# ──────────────────────────────────────────────────────────────────────
#  Helpers — fractal detection & OHLC resampling (mirror fvg_strategy)
# ──────────────────────────────────────────────────────────────────────
def _resample_d1(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Resample a time-indexed H1 (or finer) OHLC frame to D1.

    Drops the currently-forming D1 bar (mirrors fvg_strategy._daily_bias
    convention — see [[feedback_entry_pipeline_bugs_20260525]]).
    Returns None on bad input.
    """
    if df is None or len(df) == 0:
        return None
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df = df.copy()
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df = df.set_index("time").sort_index()
            else:
                return None
        cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in cols):
            return None
        o = df["open"].resample("1D").first()
        h = df["high"].resample("1D").max()
        l = df["low"].resample("1D").min()
        c = df["close"].resample("1D").last()
        out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
        # Drop the forming bar (last row may still be live).
        if len(out) >= 1:
            out = out.iloc[:-1]
        return out
    except Exception as e:
        log.debug("ASAT _resample_d1 failed: %s", e)
        return None


def _normalize_index(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Make sure df has a clean DatetimeIndex and OHLC cols. Returns
    None if it can't be coerced."""
    if df is None or len(df) == 0:
        return None
    try:
        cols = ["open", "high", "low", "close"]
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df = df.copy()
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df = df.set_index("time").sort_index()
            else:
                return None
        if not all(c in df.columns for c in cols):
            return None
        return df[cols].astype(float)
    except Exception as e:
        log.debug("ASAT _normalize_index failed: %s", e)
        return None


def _find_fractals(df: pd.DataFrame, n: int):
    """Confirmed symmetric fractal highs/lows over a `n`-bar half-window.
    Returns (highs, lows) as lists of (time_ns, level), CLOSED bars only.

    A bar at index i is a swing-high if H[i] is strictly the unique max
    over H[i-n:i+n+1] (excludes ties to avoid plateau false-positives).
    """
    highs, lows = [], []
    if df is None or len(df) < (2 * n + 1):
        return highs, lows
    H = df["high"].values
    L = df["low"].values
    T = df.index
    nbars = len(df)
    for i in range(n, nbars - n):
        seg_h = H[i - n:i + n + 1]
        seg_l = L[i - n:i + n + 1]
        if H[i] == seg_h.max() and (seg_h == H[i]).sum() == 1:
            try:
                highs.append((T[i].value, float(H[i])))
            except Exception:
                highs.append((int(i), float(H[i])))
        if L[i] == seg_l.min() and (seg_l == L[i]).sum() == 1:
            try:
                lows.append((T[i].value, float(L[i])))
            except Exception:
                lows.append((int(i), float(L[i])))
    return highs, lows


def _find_last_protective_swing(m15_df: pd.DataFrame, direction: str,
                                n: int, lookback: int,
                                entry_px: Optional[float] = None) -> Optional[float]:
    """Find the most recent CONFIRMED symmetric fractal on the M15 frame
    within the last `lookback` bars (closed bars only — iloc[-2] convention).

    LONG  → swing-LOW strictly BELOW entry_px (protective stop side).
    SHORT → swing-HIGH strictly ABOVE entry_px.

    If `entry_px` is None, the side-of-entry constraint is skipped (legacy
    behaviour, kept for test compatibility).

    Returns the swing price or None.
    """
    if m15_df is None or len(m15_df) < (2 * n + 1):
        return None
    # Restrict scan window to the lookback (plus the symmetric fractal
    # padding on the right edge — last `n` bars cannot be confirmed yet).
    win = m15_df.iloc[-(lookback + n + 2):-1]  # drop the forming bar
    if len(win) < (2 * n + 1):
        return None
    highs, lows = _find_fractals(win, n)
    if direction == "LONG":
        if not lows:
            return None
        # Iterate most-recent first; require below entry for "protective".
        for (_, lvl) in reversed(lows):
            if entry_px is None or lvl < float(entry_px):
                return lvl
        return None
    else:
        if not highs:
            return None
        for (_, lvl) in reversed(highs):
            if entry_px is None or lvl > float(entry_px):
                return lvl
        return None


def _is_unmitigated(d1: pd.DataFrame, level: float, kind: str, after_ts) -> bool:
    """Mitigation test on the D1 frame.

    LONG  candidate (kind="high"): no D1 bar AFTER `after_ts` may have
                                    a CLOSE strictly above `level`.
    SHORT candidate (kind="low"):  no D1 bar AFTER `after_ts` may have
                                    a CLOSE strictly below `level`.
    """
    if d1 is None or len(d1) == 0:
        return False
    try:
        # `after_ts` may be a ns int (from T[i].value) or a Timestamp.
        if isinstance(after_ts, (int, np.integer)):
            after = pd.Timestamp(after_ts, unit="ns", tz="UTC")
        else:
            after = pd.Timestamp(after_ts)
        post = d1[d1.index > after]
        if len(post) == 0:
            return True
        if kind == "high":
            return bool((post["close"] <= level).all())
        else:
            return bool((post["close"] >= level).all())
    except Exception as e:
        log.debug("ASAT _is_unmitigated failed: %s", e)
        # Fail-safe: treat as mitigated so we don't ship a bad target.
        return False


def _harvest_d1_swings(d1: pd.DataFrame, n: int, memory: int,
                       max_age_days: int):
    """Pull the last `memory` confirmed D1 fractals; drop anything older
    than `max_age_days` from the last D1 close.

    Returns dict with "highs" / "lows" → list of (ts_ns, level).
    """
    out = {"highs": [], "lows": []}
    if d1 is None or len(d1) < (2 * n + 1):
        return out
    highs, lows = _find_fractals(d1, n)
    # age filter
    if max_age_days and max_age_days > 0 and len(d1) > 0:
        try:
            last_ts = d1.index[-1]
            cutoff_ns = (last_ts - pd.Timedelta(days=max_age_days)).value
            highs = [(t, lvl) for (t, lvl) in highs if t >= cutoff_ns]
            lows = [(t, lvl) for (t, lvl) in lows if t >= cutoff_ns]
        except Exception:
            pass
    out["highs"] = highs[-memory:]
    out["lows"] = lows[-memory:]
    return out


def _round_to_digits(price: float, digits: Optional[int]) -> float:
    """Round to symbol digits if provided; otherwise return as-is."""
    if digits is None or digits < 0:
        return float(price)
    try:
        q = 10 ** int(digits)
        return math.floor(price * q + 0.5) / q
    except Exception:
        return float(price)


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────
def compute_asat_levels(symbol: str,
                        direction: str,
                        entry_px: float,
                        atr_h1: float,
                        h1_df: Optional[pd.DataFrame],
                        m15_df: Optional[pd.DataFrame],
                        sl_mult_base: float,
                        symbol_digits: Optional[int] = None,
                        **kwargs) -> Optional[dict]:
    """Compute Asymmetric Structure-Aware Profit Targets for a single
    candidate momentum entry.

    Returns dict on success:
        {
          "sl": float, "sl_dist": float,
          "tp1": float, "tp2": float,
          "tp2_source": "D1_SWING" | "FALLBACK_3R" | "CAPPED_5R" | "MIN_2R_FLOOR",
          "swing_ref": float, "d1_target": float|None,
          "rr_realised": float, "splits": [0.50, 0.50],
        }

    Returns None when ASAT is disabled OR not enough data OR the structural
    SL is oversized — caller MUST fall back to the existing SUB_TP_R path
    (see ASAT_FAIL_OPEN / ASAT_HARD_REJECT_ON_OVERSIZED_SL).
    """
    enabled = _cfg_get("ASAT_ENABLED", False)
    if not enabled:
        return None

    whitelist = _cfg_get("ASAT_SYMBOL_WHITELIST", set()) or set()
    if whitelist and symbol not in whitelist:
        return None

    fail_open = _cfg_get("ASAT_FAIL_OPEN", True)

    try:
        # ── Param fetch ────────────────────────────────────────────
        TP1_R = float(_cfg_get("ASAT_TP1_R", 1.5))
        TP2_FB_R = float(_cfg_get("ASAT_TP2_FALLBACK_R", 3.0))
        TP2_MIN_R = float(_cfg_get("ASAT_TP2_MIN_R", 2.0))
        TP2_MAX_R = float(_cfg_get("ASAT_TP2_MAX_R", 5.0))
        FRACTAL_N = int(_cfg_get("ASAT_FRACTAL_N", 3))
        LOOKBACK_M15 = int(_cfg_get("ASAT_SWING_LOOKBACK_M15", 60))
        D1_FRACTAL_N = int(_cfg_get("ASAT_D1_FRACTAL_N", 3))
        D1_MEM = int(_cfg_get("ASAT_D1_SWING_MEMORY", 20))
        D1_MAX_AGE = int(_cfg_get("ASAT_D1_SWING_MAX_AGE_DAYS", 30))
        D1_MIN_BARS = int(_cfg_get("ASAT_D1_MIN_BARS", 30))
        SL_BUF = float(_cfg_get("ASAT_SL_STRUCT_BUFFER_ATR", 0.25))
        SL_MAX = float(_cfg_get("ASAT_SL_MAX_ATR", 3.5))
        SL_MIN = float(_cfg_get("ASAT_SL_MIN_ATR", 0.5))
        REQUIRE_UNMIT = bool(_cfg_get("ASAT_REQUIRE_UNMITIGATED", True))
        LOG_DECISIONS = bool(_cfg_get("ASAT_LOG_EVERY_DECISION", True))

        # ── Input sanity ───────────────────────────────────────────
        if direction not in ("LONG", "SHORT"):
            return None
        if not (isinstance(entry_px, (int, float)) and math.isfinite(entry_px) and entry_px > 0):
            return None
        if not (isinstance(atr_h1, (int, float)) and math.isfinite(atr_h1) and atr_h1 > 0):
            return None
        if not (isinstance(sl_mult_base, (int, float)) and math.isfinite(sl_mult_base) and sl_mult_base > 0):
            sl_mult_base = 2.0  # safety default

        sign = +1 if direction == "LONG" else -1

        # ── Normalize frames ───────────────────────────────────────
        h1_norm = _normalize_index(h1_df)
        m15_norm = _normalize_index(m15_df)
        if h1_norm is None or len(h1_norm) < D1_MIN_BARS * 24:
            return None
        if m15_norm is None or len(m15_norm) < (LOOKBACK_M15 + 2 * FRACTAL_N + 1):
            return None

        # ── 1. Resample H1 → D1 ────────────────────────────────────
        d1 = _resample_d1(h1_norm)
        if d1 is None or len(d1) < D1_MIN_BARS:
            return None

        # ── 2. Protective M15 swing ────────────────────────────────
        swing_ref = _find_last_protective_swing(
            m15_norm, direction, FRACTAL_N, LOOKBACK_M15,
            entry_px=float(entry_px),
        )
        if swing_ref is None:
            return None

        # ── 3. Structural SL ───────────────────────────────────────
        if direction == "LONG":
            struct_sl = float(swing_ref) - SL_BUF * float(atr_h1)
            atr_sl = float(entry_px) - sl_mult_base * float(atr_h1)
            sl = min(struct_sl, atr_sl)  # deeper of the two for LONG
        else:
            struct_sl = float(swing_ref) + SL_BUF * float(atr_h1)
            atr_sl = float(entry_px) + sl_mult_base * float(atr_h1)
            sl = max(struct_sl, atr_sl)  # deeper of the two for SHORT

        sl_dist = abs(float(entry_px) - sl)

        # cap
        if sl_dist > SL_MAX * float(atr_h1):
            if LOG_DECISIONS:
                log.info(
                    "ASAT[%s] reject: SL too wide sl_dist=%.5f atr=%.5f cap=%.2fATR",
                    symbol, sl_dist, atr_h1, SL_MAX,
                )
            return None  # caller decides fallback vs skip via ASAT_HARD_REJECT_ON_OVERSIZED_SL

        # floor
        if sl_dist < SL_MIN * float(atr_h1):
            sl_dist = SL_MIN * float(atr_h1)
            sl = float(entry_px) - sign * sl_dist

        # ── 4. TP1 fixed ───────────────────────────────────────────
        tp1 = float(entry_px) + sign * TP1_R * sl_dist

        # ── 5. D1 swing candidate set ──────────────────────────────
        swings = _harvest_d1_swings(d1, D1_FRACTAL_N, D1_MEM, D1_MAX_AGE)
        cands = []  # list of (t_ns, level) — pre-distance-filter
        if direction == "LONG":
            for (t, lvl) in swings["highs"]:
                if lvl <= float(entry_px):
                    continue
                if REQUIRE_UNMIT and not _is_unmitigated(d1, lvl, "high", t):
                    continue
                cands.append((t, lvl))
        else:
            for (t, lvl) in swings["lows"]:
                if lvl >= float(entry_px):
                    continue
                if REQUIRE_UNMIT and not _is_unmitigated(d1, lvl, "low", t):
                    continue
                cands.append((t, lvl))

        valid = [
            lvl for (_, lvl) in cands
            if TP2_MIN_R * sl_dist <= abs(lvl - float(entry_px)) <= TP2_MAX_R * sl_dist
        ]
        too_close = any(abs(lvl - float(entry_px)) < TP2_MIN_R * sl_dist for (_, lvl) in cands)
        too_far = any(abs(lvl - float(entry_px)) > TP2_MAX_R * sl_dist for (_, lvl) in cands)

        d1_target: Optional[float] = None
        if valid:
            d1_target = min(valid, key=lambda c: abs(c - float(entry_px)))
            tp2 = float(d1_target)
            src = "D1_SWING"
        elif too_close and not too_far:
            tp2 = float(entry_px) + sign * TP2_MIN_R * sl_dist
            src = "MIN_2R_FLOOR"
        elif too_far and not too_close:
            tp2 = float(entry_px) + sign * TP2_MAX_R * sl_dist
            src = "CAPPED_5R"
        else:
            tp2 = float(entry_px) + sign * TP2_FB_R * sl_dist
            src = "FALLBACK_3R"

        # ── 6. Sanity ──────────────────────────────────────────────
        tp1_dist = abs(tp1 - float(entry_px))
        tp2_dist = abs(tp2 - float(entry_px))
        if tp2_dist < 1.5 * tp1_dist:
            tp2 = float(entry_px) + sign * TP2_FB_R * sl_dist
            src = "FALLBACK_3R"
            d1_target = None
            tp2_dist = abs(tp2 - float(entry_px))

        # Direction-monotonicity: tp1 strictly between entry and tp2.
        if direction == "LONG":
            if not (float(entry_px) < tp1 < tp2):
                if LOG_DECISIONS:
                    log.info("ASAT[%s] degenerate LONG levels e=%.5f tp1=%.5f tp2=%.5f", symbol, entry_px, tp1, tp2)
                return None
        else:
            if not (float(entry_px) > tp1 > tp2):
                if LOG_DECISIONS:
                    log.info("ASAT[%s] degenerate SHORT levels e=%.5f tp1=%.5f tp2=%.5f", symbol, entry_px, tp1, tp2)
                return None

        # ── 7. Round to digits ─────────────────────────────────────
        sl_r = _round_to_digits(sl, symbol_digits)
        tp1_r = _round_to_digits(tp1, symbol_digits)
        tp2_r = _round_to_digits(tp2, symbol_digits)
        swing_ref_r = _round_to_digits(float(swing_ref), symbol_digits)
        d1_target_r = _round_to_digits(d1_target, symbol_digits) if d1_target is not None else None

        out = {
            "sl": sl_r,
            "sl_dist": float(sl_dist),
            "tp1": tp1_r,
            "tp2": tp2_r,
            "tp2_source": src,
            "swing_ref": swing_ref_r,
            "d1_target": d1_target_r,
            "rr_realised": float(tp2_dist / sl_dist) if sl_dist > 0 else 0.0,
            "splits": [0.50, 0.50],
        }

        if LOG_DECISIONS:
            log.info(
                "ASAT[%s] dir=%s e=%.5f sl=%.5f tp1=%.5f tp2=%.5f src=%s "
                "swing=%.5f d1_tgt=%s rr=%.2f",
                symbol, direction, entry_px, sl_r, tp1_r, tp2_r, src,
                swing_ref_r, d1_target_r, out["rr_realised"],
            )

        return out

    except Exception as e:
        log.warning("ASAT[%s] internal error: %s — fail_open=%s", symbol, e, fail_open)
        if fail_open:
            return None
        # If user wants hard-fail, propagate as None too — caller logs.
        return None


# ──────────────────────────────────────────────────────────────────────
#  Self-test (synthetic data)
# ──────────────────────────────────────────────────────────────────────
def _synth_h1(n_days: int = 60, start_px: float = 100.0, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic H1 OHLC frame with embedded swings."""
    rng = np.random.default_rng(seed)
    nbars = n_days * 24
    idx = pd.date_range("2026-01-01", periods=nbars, freq="1h", tz="UTC")
    # Drifty noise with occasional spikes to seed swings.
    drift = np.linspace(0, 8.0, nbars)
    noise = rng.normal(0, 0.4, nbars).cumsum() * 0.5
    base = start_px + drift + noise
    # Add periodic swing peaks/troughs every ~36h.
    for i in range(36, nbars, 36):
        if (i // 36) % 2 == 0:
            base[i] += 3.0
        else:
            base[i] -= 3.0
    close = base
    open_ = np.concatenate([[start_px], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, nbars))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, nbars))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )
    df.index.name = "time"
    return df


def _synth_m15(h1: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Build a synthetic M15 frame by interpolating H1, with embedded
    swing lows/highs both BELOW and ABOVE the final close so LONG and
    SHORT protective-swing tests both have valid references."""
    rng = np.random.default_rng(seed)
    nbars = len(h1) * 4
    idx = pd.date_range(h1.index[0], periods=nbars, freq="15min", tz="UTC")
    h1_close = h1["close"].values
    closes = np.repeat(h1_close, 4) + rng.normal(0, 0.15, nbars).cumsum() * 0.02
    open_ = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(open_, closes) + np.abs(rng.normal(0, 0.08, nbars))
    low = np.minimum(open_, closes) - np.abs(rng.normal(0, 0.08, nbars))
    final_px = closes[-1]
    # Embed a protective swing-LOW (for LONG): close to but BELOW final_px,
    # so structural SL stays within ASAT_SL_MAX_ATR cap (=3.5*ATR).
    # Also clear out competing closer swing-lows that would sit ABOVE final_px.
    sl_idx = nbars - 28
    # First, lift all lows in the protective window so the deepest dip is
    # unambiguously our embedded swing.
    for j in range(nbars - 60, nbars - 1):
        if low[j] < final_px - 0.05 and j not in range(sl_idx - 3, sl_idx + 4):
            low[j] = final_px - 0.05
    for j in range(sl_idx - 3, sl_idx + 4):
        low[j] = min(low[j], final_px - 0.35)
    low[sl_idx] = final_px - 0.55
    high[sl_idx] = min(high[sl_idx], final_px - 0.10)
    # Embed a protective swing-HIGH (for SHORT): close to but ABOVE final_px.
    sh_idx = nbars - 14
    for j in range(nbars - 60, nbars - 1):
        if high[j] > final_px + 0.05 and j not in range(sh_idx - 3, sh_idx + 4):
            high[j] = final_px + 0.05
    for j in range(sh_idx - 3, sh_idx + 4):
        high[j] = max(high[j], final_px + 0.35)
    high[sh_idx] = final_px + 0.55
    low[sh_idx] = max(low[sh_idx], final_px + 0.10)
    # Make sure open/close don't violate high/low at the manipulated bars.
    for j in (sl_idx, sh_idx):
        for arr in (open_, closes):
            arr[j] = max(low[j], min(high[j], arr[j]))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": closes},
        index=idx,
    )
    df.index.name = "time"
    return df


def _run_self_test() -> int:
    print("ASAT self-test starting...")
    # Force-enable for the test even if config.py default is False.
    if _cfg is not None:
        setattr(_cfg, "ASAT_ENABLED", True)
        setattr(_cfg, "ASAT_SYMBOL_WHITELIST", set())
        setattr(_cfg, "ASAT_LOG_EVERY_DECISION", False)
    else:
        # Monkey-patch _cfg_get behaviour by stuffing a tiny shim.
        class _Shim:
            ASAT_ENABLED = True
            ASAT_SYMBOL_WHITELIST = set()
            ASAT_LOG_EVERY_DECISION = False
        globals()["_cfg"] = _Shim()

    h1 = _synth_h1(n_days=60)
    m15 = _synth_m15(h1)
    entry_px = float(h1["close"].iloc[-1])
    atr_h1 = float((h1["high"] - h1["low"]).rolling(14).mean().iloc[-2])
    print(f"  entry_px={entry_px:.4f} atr_h1={atr_h1:.4f}")

    # ── Test 1: LONG happy-path ──────────────────────────────────
    out = compute_asat_levels(
        symbol="TEST", direction="LONG",
        entry_px=entry_px, atr_h1=atr_h1,
        h1_df=h1, m15_df=m15,
        sl_mult_base=2.0, symbol_digits=4,
    )
    assert out is not None, "T1 LONG: expected non-None output"
    assert out["sl"] < entry_px, f"T1 LONG: sl {out['sl']} must be < entry {entry_px}"
    assert out["tp1"] > entry_px, "T1 LONG: tp1 must be > entry"
    assert out["tp2"] > out["tp1"], "T1 LONG: tp2 must be > tp1"
    assert out["splits"] == [0.50, 0.50]
    assert out["tp2_source"] in {"D1_SWING", "FALLBACK_3R", "CAPPED_5R", "MIN_2R_FLOOR"}
    assert out["rr_realised"] > 0
    print(f"  T1 LONG  ok: sl={out['sl']} tp1={out['tp1']} tp2={out['tp2']} src={out['tp2_source']} rr={out['rr_realised']:.2f}")

    # ── Test 2: SHORT happy-path ────────────────────────────────
    out2 = compute_asat_levels(
        symbol="TEST", direction="SHORT",
        entry_px=entry_px, atr_h1=atr_h1,
        h1_df=h1, m15_df=m15,
        sl_mult_base=2.0, symbol_digits=4,
    )
    assert out2 is not None, "T2 SHORT: expected non-None output"
    assert out2["sl"] > entry_px, "T2 SHORT: sl must be > entry"
    assert out2["tp1"] < entry_px, "T2 SHORT: tp1 must be < entry"
    assert out2["tp2"] < out2["tp1"], "T2 SHORT: tp2 must be < tp1"
    print(f"  T2 SHORT ok: sl={out2['sl']} tp1={out2['tp1']} tp2={out2['tp2']} src={out2['tp2_source']} rr={out2['rr_realised']:.2f}")

    # ── Test 3: ASAT_ENABLED=False → None ───────────────────────
    setattr(_cfg, "ASAT_ENABLED", False)
    out3 = compute_asat_levels(
        symbol="TEST", direction="LONG",
        entry_px=entry_px, atr_h1=atr_h1,
        h1_df=h1, m15_df=m15, sl_mult_base=2.0,
    )
    assert out3 is None, "T3: disabled module must return None"
    print("  T3 disabled ok")
    setattr(_cfg, "ASAT_ENABLED", True)

    # ── Test 4: symbol whitelist mismatch → None ────────────────
    setattr(_cfg, "ASAT_SYMBOL_WHITELIST", {"OTHER"})
    out4 = compute_asat_levels(
        symbol="TEST", direction="LONG",
        entry_px=entry_px, atr_h1=atr_h1,
        h1_df=h1, m15_df=m15, sl_mult_base=2.0,
    )
    assert out4 is None, "T4: whitelist mismatch must return None"
    print("  T4 whitelist ok")
    setattr(_cfg, "ASAT_SYMBOL_WHITELIST", set())

    # ── Test 5: insufficient data → None ────────────────────────
    out5 = compute_asat_levels(
        symbol="TEST", direction="LONG",
        entry_px=entry_px, atr_h1=atr_h1,
        h1_df=h1.iloc[:50], m15_df=m15.iloc[:50],
        sl_mult_base=2.0,
    )
    assert out5 is None, "T5: insufficient data must return None"
    print("  T5 insufficient data ok")

    # ── Test 6: oversized SL → None ─────────────────────────────
    # Use absurdly small atr to force struct_sl distance to blow the cap.
    out6 = compute_asat_levels(
        symbol="TEST", direction="LONG",
        entry_px=entry_px, atr_h1=0.0001,  # cap = 3.5 * 0.0001 = 0.00035
        h1_df=h1, m15_df=m15, sl_mult_base=2.0,
    )
    assert out6 is None, "T6: oversized SL must return None"
    print("  T6 oversized SL ok")

    # ── Test 7: invalid direction → None ────────────────────────
    out7 = compute_asat_levels(
        symbol="TEST", direction="BOGUS",
        entry_px=entry_px, atr_h1=atr_h1,
        h1_df=h1, m15_df=m15, sl_mult_base=2.0,
    )
    assert out7 is None, "T7: bad direction must return None"
    print("  T7 invalid direction ok")

    # ── Test 8: round-to-digits sanity ──────────────────────────
    out8 = compute_asat_levels(
        symbol="TEST", direction="LONG",
        entry_px=entry_px, atr_h1=atr_h1,
        h1_df=h1, m15_df=m15, sl_mult_base=2.0,
        symbol_digits=2,
    )
    assert out8 is not None
    # rounded to 2 dp
    for k in ("sl", "tp1", "tp2"):
        v = out8[k]
        assert round(v, 2) == v, f"T8: {k}={v} not rounded to 2dp"
    print("  T8 rounding ok")

    print("ASAT self-test PASSED")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sys.exit(_run_self_test())
