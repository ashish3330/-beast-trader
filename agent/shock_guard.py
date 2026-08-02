"""Geopolitical / volatility SHOCK GUARD — price-based, causal (2026-08-02).

The economic-calendar filter (agent/calendar_filter.py) covers SCHEDULED news
(NFP/FOMC/CPI). It cannot see UNSCHEDULED geopolitical shocks — an Iran/Israel
escalation, a surprise oil move, a flash risk-off. External news APIs are
unreliable from this headless Wine box (GDELT rate-limited/timeouts), so the
PRIMARY defense is PRICE-BASED: detect the market's violent reaction (ATR
explosion / abnormal single-bar range) and act on it.

Validated on the real D1 caches (workflow wf_593b4142, 2026-08-02): fires on
~1.5-2.1% of bars, catches every marquee shock in-sample (COVID Mar-2020,
Russia-Ukraine Feb-2022, gold/oil Mar-2022, negative-WTI Apr-2020, yen-carry
Aug-2024, tariff crash Apr-2025, 2026 mideast gold/oil spikes) with no
top-percentile misses.

CRITICAL — the value is ASYMMETRIC by asset class (a blanket all-symbol block is
net-NEGATIVE):
  * INDICES (NAS100.r, JPN225ft, .r indices): post-shock 3d forward return is
    NEGATIVE → BLOCK new entries during cooldown (drawdown reducer;
    NAS100 P&L +31%->+62%, DD 35.1->33.5).
  * GOLD / OIL / BTC: they TREND through the shock (oil fwd +1.03% vs -0.03%
    normal) → blocking is net-NEGATIVE (+30%->-4%). DE-RISK only (halve size),
    NEVER block direction.

Fail-OPEN by construction: any error → {"action": "NONE"} → callers trade normally.
Everything is causal (rolling windows shifted so the forming bar never sees itself)
so there is zero look-ahead.
"""
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import (SHOCK_GUARD_CACHE_DIR, SHOCK_GUARD_COOLDOWN_BARS,
                        SHOCK_GUARD_THRESHOLDS, SHOCK_GUARD_DEFAULT_THRESHOLD,
                        SHOCK_GUARD_BLOCK_SYMBOLS, SHOCK_GUARD_DERISK_SYMBOLS)
except Exception:  # pragma: no cover — safe fallbacks so import never breaks the bot
    SHOCK_GUARD_CACHE_DIR = "/Users/ashish/Documents/xauusd-trading-bot/cache"
    SHOCK_GUARD_COOLDOWN_BARS = 3
    SHOCK_GUARD_THRESHOLDS = {}
    SHOCK_GUARD_DEFAULT_THRESHOLD = (4.5, 3.2)
    SHOCK_GUARD_BLOCK_SYMBOLS = set()
    SHOCK_GUARD_DERISK_SYMBOLS = set()

log = logging.getLogger("dragon.shock")

_ATR_N = 14
_MED_N = 100          # trailing window for atr median + range mean/std
_RECOMPUTE_TTL = 1800  # s — D1 bars form daily; recompute the detector at most every 30 min


def _cache_key(symbol: str) -> str:
    # mirror scripts/fetch_h1.py: XAUUSD -> xauusd, others -> replace('.', '_')
    return "xauusd" if symbol == "XAUUSD" else symbol.replace(".", "_")


class ShockGuard:
    """Per-symbol price-shock detector with a K-bar cooldown. Public API:
    guard_state(symbol) -> {"fired","in_cooldown","asset_class","action","reason"}."""

    def __init__(self, cache_dir=None):
        self._cache_dir = Path(cache_dir or SHOCK_GUARD_CACHE_DIR)
        # per-symbol memo of the last computed detector result + when
        self._memo = {}          # symbol -> (ts_computed, fired_bool, last_bar_time, rng_z, atr_ratio)
        # durable-ish in-memory cooldown: symbol -> bar_time (pandas Timestamp) of last fire
        self._last_fire_bar = {}
        log.info("ShockGuard initialized (cooldown=%d D1 bars, cache=%s)",
                 SHOCK_GUARD_COOLDOWN_BARS, self._cache_dir)

    # ── asset routing: BLOCK (indices) vs DERISK (gold/oil/btc) vs NONE ──
    def _action_for(self, symbol: str) -> str:
        if symbol in SHOCK_GUARD_BLOCK_SYMBOLS:
            return "BLOCK"
        if symbol in SHOCK_GUARD_DERISK_SYMBOLS:
            return "DERISK"
        return "NONE"   # UNLISTED SYMBOLS NEVER BLOCK (the oil-asymmetry safeguard)

    def _asset_class(self, symbol: str) -> str:
        if symbol in SHOCK_GUARD_BLOCK_SYMBOLS:
            return "index"
        if symbol in SHOCK_GUARD_DERISK_SYMBOLS:
            return "commodity_or_crypto"
        return "other"

    def _compute_fired(self, symbol: str):
        """Return (fired, last_bar_time, rng_z, atr_ratio) computed CAUSALLY on the
        symbol's D1 cache, or None on any failure (→ fail-open)."""
        path = self._cache_dir / ("raw_d1_" + _cache_key(symbol) + ".pkl")
        if not path.exists():
            return None
        df = pickle.load(open(path, "rb"))
        if df is None or len(df) < _MED_N + _ATR_N + 5:
            return None
        df = df.copy()
        h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        pc = c.shift(1)
        tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = tr.rolling(_ATR_N).mean()
        # CAUSAL: trailing median/mean/std EXCLUDE the current bar (shift(1))
        atr_med = atr.shift(1).rolling(_MED_N).median()
        atr_ratio = atr / atr_med
        rng = (h - l)
        rng_mean = rng.shift(1).rolling(_MED_N).mean()
        rng_std = rng.shift(1).rolling(_MED_N).std()
        rng_z = (rng - rng_mean) / rng_std.replace(0, np.nan)
        Z, A = SHOCK_GUARD_THRESHOLDS.get(symbol, SHOCK_GUARD_DEFAULT_THRESHOLD)
        i = len(df) - 1
        _rz = float(rng_z.iloc[i]) if pd.notna(rng_z.iloc[i]) else 0.0
        _ar = float(atr_ratio.iloc[i]) if pd.notna(atr_ratio.iloc[i]) else 0.0
        fired = (_rz >= Z) or (_ar >= A)
        return fired, df["time"].iloc[i], _rz, _ar

    def guard_state(self, symbol: str) -> dict:
        """Fail-OPEN. Returns the current guard state for `symbol`."""
        none_state = {"fired": False, "in_cooldown": False,
                      "asset_class": self._asset_class(symbol),
                      "action": "NONE", "reason": ""}
        try:
            now = time.time()
            memo = self._memo.get(symbol)
            if memo is None or (now - memo[0]) >= _RECOMPUTE_TTL:
                res = self._compute_fired(symbol)
                if res is None:
                    return none_state          # missing/short cache → fail-open
                fired, bar_time, rz, ar = res
                self._memo[symbol] = (now, fired, bar_time, rz, ar)
                if fired:
                    self._last_fire_bar[symbol] = bar_time
            else:
                _, fired, bar_time, rz, ar = memo

            # in cooldown if the current D1 bar is within K bars of the last fire.
            in_cd = False
            lastfire = self._last_fire_bar.get(symbol)
            if lastfire is not None and bar_time is not None:
                try:
                    bars_since = int((pd.Timestamp(bar_time).normalize()
                                      - pd.Timestamp(lastfire).normalize()).days)
                    in_cd = 0 <= bars_since < SHOCK_GUARD_COOLDOWN_BARS
                except Exception:
                    in_cd = bool(fired)
            action = self._action_for(symbol)
            reason = ("rng_z=%.1f atr_ratio=%.1f (last fire %s)"
                      % (rz, ar, str(lastfire)[:10])) if (fired or in_cd) else ""
            return {"fired": bool(fired), "in_cooldown": bool(in_cd),
                    "asset_class": self._asset_class(symbol),
                    "action": action if in_cd else "NONE", "reason": reason}
        except Exception as e:
            log.debug("shock guard_state(%s) failed (fail-open): %s", symbol, e)
            return none_state

    def derisk_mult(self, symbol: str, default_mult: float) -> float:
        """Size multiplier for DERISK symbols in cooldown (else 1.0). Fail-open → 1.0."""
        try:
            st = self.guard_state(symbol)
            if st["action"] == "DERISK" and st["in_cooldown"]:
                return float(default_mult)
        except Exception:
            pass
        return 1.0
