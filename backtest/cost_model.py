"""
Beast Trader — Transaction Cost Model.

Spread is always applied (live system pays it). Slippage / commission / swap
are OPT-IN via flags on `CostModel(...)` so existing backtests stay
spread-only-by-default (per user feedback 2026-04-12: "live system has no
slippage").

Per-symbol tables below approximate institutional-tier broker costs
(conservative). Override at construction time if needed.
"""
from __future__ import annotations

from typing import Optional


# ────────────────────────────────────────────────────────────────────────
# Per-symbol slippage scaling (fraction-of-spread).
# Quoted as (base_frac, max_frac). Realised slippage is clipped between
# `base_frac * spread` and `max_frac * spread` after the size-relative
# component (see CostModel.entry_cost / exit_cost).
# Majors: 10–30% of spread. Thin/exotic: 50–100% of spread.
# ────────────────────────────────────────────────────────────────────────
SLIPPAGE_BPS: dict[str, tuple[float, float]] = {
    # Majors / liquid metals
    "EURUSD":   (0.10, 0.30),
    "GBPUSD":   (0.10, 0.30),
    "USDJPY":   (0.10, 0.30),
    "USDCAD":   (0.10, 0.30),
    "XAUUSD":   (0.15, 0.40),
    # Crosses / less liquid
    "EURJPY":   (0.20, 0.50),
    "GBPJPY":   (0.20, 0.60),
    "AUDJPY":   (0.20, 0.60),
    "XAGUSD":   (0.30, 0.80),
    # Indices CFD
    "NAS100.r": (0.20, 0.50),
    "SP500.r":  (0.20, 0.50),
    "GER40.r":  (0.20, 0.60),
    "JPN225ft": (0.30, 0.80),
    # Crypto CFD (thin)
    "BTCUSD":   (0.50, 1.00),
    "ETHUSD":   (0.50, 1.00),
}
_DEFAULT_SLIP = (0.30, 0.80)


# ────────────────────────────────────────────────────────────────────────
# Round-turn commission (USD per lot, charged once on close).
# Forex / metals: ~$7/lot RT (institutional).
# Indices / crypto-CFD: ~$0.50 per CFD contract — translated to USD per lot
# by treating "lot" as one CFD contract for indices & crypto.
# ────────────────────────────────────────────────────────────────────────
COMMISSION: dict[str, float] = {
    # Forex
    "EURUSD":   7.0,
    "GBPUSD":   7.0,
    "USDJPY":   7.0,
    "USDCAD":   7.0,
    "EURJPY":   7.0,
    "GBPJPY":   7.0,
    "AUDJPY":   7.0,
    # Metals
    "XAUUSD":   7.0,
    "XAGUSD":   7.0,
    # Indices
    "NAS100.r": 0.50,
    "SP500.r":  0.50,
    "GER40.r":  0.50,
    "JPN225ft": 0.50,
    # Crypto
    "BTCUSD":   0.50,
    "ETHUSD":   0.50,
}
_DEFAULT_COMMISSION = 7.0


# ────────────────────────────────────────────────────────────────────────
# Overnight financing (swap) — USD per lot per day, by side.
# Conservative institutional-tier numbers; sign convention:
#   negative = trader PAYS broker (typical), positive = trader receives.
# Triple swap on Wednesday is applied for forex (industry convention),
# multiplied at apply-time, not encoded here.
# ────────────────────────────────────────────────────────────────────────
SWAP: dict[str, dict[str, float]] = {
    # Forex (USD/lot/day; long pays for short-rate currencies, etc.)
    "EURUSD":   {"long": -2.50, "short": -0.50},
    "GBPUSD":   {"long": -2.20, "short": -0.40},
    "USDJPY":   {"long":  0.10, "short": -3.00},
    "USDCAD":   {"long":  0.20, "short": -2.80},
    "EURJPY":   {"long": -1.80, "short": -1.20},
    "GBPJPY":   {"long": -0.80, "short": -2.50},
    "AUDJPY":   {"long": -0.40, "short": -2.20},
    # Metals
    "XAUUSD":   {"long": -3.50, "short": -0.80},
    "XAGUSD":   {"long": -0.40, "short": -0.20},
    # Indices CFD
    "NAS100.r": {"long": -2.00, "short": -0.50},
    "SP500.r":  {"long": -1.50, "short": -0.40},
    "GER40.r":  {"long": -1.00, "short": -0.30},
    "JPN225ft": {"long":  0.05, "short": -1.20},
    # Crypto CFD (typically expensive both sides)
    "BTCUSD":   {"long": -8.00, "short": -8.00},
    "ETHUSD":   {"long": -3.00, "short": -3.00},
}
_DEFAULT_SWAP = {"long": -1.50, "short": -1.50}


def get_slippage_range(symbol: str) -> tuple[float, float]:
    return SLIPPAGE_BPS.get(symbol, _DEFAULT_SLIP)


def get_commission(symbol: str) -> float:
    return COMMISSION.get(symbol, _DEFAULT_COMMISSION)


def get_swap_per_day(symbol: str, direction: int) -> float:
    """Return USD-per-lot-per-day swap for a given side. direction: +1 long, -1 short."""
    side = "long" if direction == 1 else "short"
    return SWAP.get(symbol, _DEFAULT_SWAP)[side]


class CostModel:
    """Spread + (optional) slippage + (optional) commission + (optional) swap.

    Spread: always applied via entry_cost/exit_cost (this matches existing
    backtest behaviour; do not break it).
    Slippage: opt-in (`with_slippage=True`). Modelled as fraction-of-spread
    plus an ATR-relative size term, clipped to a per-symbol envelope.
    Commission: opt-in (`with_commission=True`). USD per lot, charged once
    on close (round-turn).
    Swap: opt-in (`with_swap=True`). Apply per-day via `swap_charge`,
    triple-on-Wednesday convention for forex.
    """

    def __init__(
        self,
        spread: float,
        point: float = 0.01,
        symbol: Optional[str] = None,
        slippage_pts: float = 0.0,
        with_slippage: bool = False,
        with_commission: bool = False,
        with_swap: bool = False,
    ):
        self.spread = float(spread)
        self.point = float(point)
        self.symbol = symbol
        # Legacy fixed slippage (price units). Preserved for backwards-compat
        # callers that pass slippage_pts directly. New flag-driven path uses
        # _dynamic_slip() below.
        self.slippage = float(slippage_pts) * self.point
        self.with_slippage = bool(with_slippage)
        self.with_commission = bool(with_commission)
        self.with_swap = bool(with_swap)
        # Track accumulated extras so the simulator can report a breakdown.
        self.cum_slippage_usd = 0.0
        self.cum_commission_usd = 0.0
        self.cum_swap_usd = 0.0

    # ── slippage ─────────────────────────────────────────────────────
    def _dynamic_slip(self, signed_size: float = 0.0, atr: float = 0.0) -> float:
        """Slippage in price units for ONE side.
        Formula (per-side):
            slip = clip(0.1*spread + 0.05*|signed_size|/atr_pts, base*spread, max*spread)
        ATR is measured in price units; |signed_size|/atr is dimensionless.
        Falls back to base*spread when atr<=0.
        """
        if not self.with_slippage:
            return self.slippage  # legacy path (0 by default)
        base, mx = get_slippage_range(self.symbol or "")
        size_term = 0.0
        if atr and atr > 0:
            size_term = 0.05 * (abs(signed_size) / atr) * self.spread
        raw = 0.10 * self.spread + size_term
        lo = base * self.spread
        hi = mx * self.spread
        if raw < lo:
            return lo
        if raw > hi:
            return hi
        return raw

    # ── per-side price adjustments ───────────────────────────────────
    def entry_cost(self, direction: int, signed_size: float = 0.0, atr: float = 0.0) -> float:
        """Signed price adjustment to entry (positive = worse for longs)."""
        slip = self._dynamic_slip(signed_size, atr)
        return (self.spread / 2.0 + slip) * direction

    def exit_cost(self, direction: int, signed_size: float = 0.0, atr: float = 0.0) -> float:
        """Signed price adjustment to exit (negative = worse for longs)."""
        slip = self._dynamic_slip(signed_size, atr)
        return -(self.spread / 2.0 + slip) * direction

    # ── commission (USD/lot, round-turn, charged once at close) ──────
    def commission_charge(self, lots: float = 1.0) -> float:
        if not self.with_commission:
            return 0.0
        per_lot = get_commission(self.symbol or "")
        usd = per_lot * float(lots)
        self.cum_commission_usd += usd
        return usd

    # ── swap (USD/lot, applied per day held) ─────────────────────────
    def swap_charge(
        self,
        direction: int,
        lots: float,
        days_held: float,
        triple_wednesday_count: int = 0,
    ) -> float:
        """Total swap cost USD for the position.
        days_held: float number of overnight crossings of 17:00 NY rollover.
        triple_wednesday_count: number of those rollovers that fell on a
            Wednesday (forex industry convention applies 3x swap then).
        """
        if not self.with_swap or lots <= 0 or days_held <= 0:
            return 0.0
        per_day = get_swap_per_day(self.symbol or "", direction)
        regular_days = max(0.0, float(days_held) - float(triple_wednesday_count))
        usd = per_day * float(lots) * regular_days
        usd += per_day * float(lots) * float(triple_wednesday_count) * 3.0
        self.cum_swap_usd += usd
        return usd

    # ── helpers ──────────────────────────────────────────────────────
    def round_trip_cost(self) -> float:
        """Spread + 2*slippage in price units (excludes commission/swap)."""
        slip = self._dynamic_slip()
        return self.spread + 2.0 * slip

    def round_trip_cost_per_lot(self, tick_value: float) -> float:
        cost_points = self.round_trip_cost() / self.point
        return cost_points * tick_value

    def __repr__(self) -> str:
        return (
            f"CostModel(symbol={self.symbol}, spread={self.spread}, "
            f"slip={self.with_slippage}, comm={self.with_commission}, "
            f"swap={self.with_swap})"
        )


def count_overnight_rollovers(entry_ts, exit_ts) -> tuple[float, int]:
    """Count overnight 17:00 NY rollovers between entry and exit.

    Returns (n_rollovers, n_wednesday_rollovers).
    Wednesday rollovers trigger 3x forex swap (industry convention).
    Inputs are pandas-datetime-like objects.

    Implementation: iterate day boundaries at 17:00 America/New_York
    between entry and exit; count those that fall on Wednesday.
    """
    import pandas as pd

    if entry_ts is None or exit_ts is None:
        return 0.0, 0
    try:
        et = pd.Timestamp(entry_ts)
        xt = pd.Timestamp(exit_ts)
    except Exception:
        return 0.0, 0
    if et.tzinfo is None:
        et = et.tz_localize("UTC")
    if xt.tzinfo is None:
        xt = xt.tz_localize("UTC")
    if xt <= et:
        return 0.0, 0

    # Convert to NY time to find 17:00 rollovers
    et_ny = et.tz_convert("America/New_York")
    xt_ny = xt.tz_convert("America/New_York")

    # First rollover candidate: today at 17:00 NY (if entry is before 17:00)
    cur = et_ny.normalize() + pd.Timedelta(hours=17)
    if cur <= et_ny:
        cur = cur + pd.Timedelta(days=1)

    n = 0
    n_wed = 0
    while cur < xt_ny:
        # weekday(): Mon=0..Sun=6 ; Wed=2
        if cur.weekday() == 2:
            n_wed += 1
        n += 1
        cur = cur + pd.Timedelta(days=1)
    return float(n), int(n_wed)
