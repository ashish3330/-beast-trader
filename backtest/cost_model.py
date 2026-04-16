"""
Beast Trader — Transaction Cost Model.
Spread + slippage per side, applied at entry and exit.
"""


class CostModel:
    """Simple cost model: fixed spread + fixed slippage per side."""

    def __init__(self, spread: float, slippage_pts: float = 1.0, point: float = 0.01):
        """
        Args:
            spread: spread in price units (e.g. 0.30 for XAUUSD)
            slippage_pts: slippage in points per side (1 point default)
            point: point size (e.g. 0.01 for XAUUSD)
        """
        self.spread = spread
        self.slippage = slippage_pts * point
        self.point = point

    def entry_cost(self, direction: int) -> float:
        """Cost applied to entry price.
        LONG: pay half spread + slippage (price moves up)
        SHORT: pay half spread + slippage (price moves down)
        Returns signed adjustment to entry price (positive = worse for longs).
        """
        return (self.spread / 2.0 + self.slippage) * direction

    def exit_cost(self, direction: int) -> float:
        """Cost applied to exit price.
        LONG exit (sell): price moves down by half spread + slippage
        SHORT exit (buy): price moves up by half spread + slippage
        Returns signed adjustment to exit price (negative = worse for longs).
        """
        return -(self.spread / 2.0 + self.slippage) * direction

    def round_trip_cost(self) -> float:
        """Total cost for a round trip in price units."""
        return self.spread + 2.0 * self.slippage

    def round_trip_cost_per_lot(self, tick_value: float) -> float:
        """Total round-trip cost in USD per lot."""
        cost_points = self.round_trip_cost() / self.point
        return cost_points * tick_value

    def __repr__(self):
        return (f"CostModel(spread={self.spread}, slippage={self.slippage:.4f}, "
                f"round_trip={self.round_trip_cost():.4f})")
