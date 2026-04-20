"""
Vectorized FVG (Fair Value Gap) Detector for backtesting.
Standalone — requires only numpy. No state objects or live data.

Usage:
    from fvg_vectorized import detect_fvg
    result = detect_fvg(open_arr, high_arr, low_arr, close_arr)
"""
import numpy as np
from typing import List, Dict, Tuple


# --- Configuration ---
MIN_FVG_SIZE_ATR = 0.3   # FVG must be >= 30% of ATR to count
MAX_FVG_AGE_BARS = 50    # Expire FVGs older than this
ATR_PERIOD = 14


def detect_fvg(
    open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_size_atr: float = MIN_FVG_SIZE_ATR,
    max_age: int = MAX_FVG_AGE_BARS,
) -> Dict:
    """
    Detect Fair Value Gaps across entire OHLC series (vectorized).

    Parameters
    ----------
    open, high, low, close : np.ndarray
        Price arrays of equal length.
    min_size_atr : float
        Minimum FVG size as fraction of ATR.
    max_age : int
        FVGs older than this are expired and ignored.

    Returns
    -------
    dict with keys:
        fvg_bullish : np.ndarray[bool]  — True at bar where bullish FVG forms
        fvg_bearish : np.ndarray[bool]  — True at bar where bearish FVG forms
        fvg_zones   : list[dict]        — Active (unfilled) FVG zones with metadata
        inside_bull_fvg : np.ndarray[bool] — True where price is inside an open bullish FVG
        inside_bear_fvg : np.ndarray[bool] — True where price is inside an open bearish FVG
        dist_nearest_bull : np.ndarray[float] — Distance to nearest bullish FVG below (0 if inside/none)
        dist_nearest_bear : np.ndarray[float] — Distance to nearest bearish FVG above (0 if inside/none)
    """
    n = len(high)
    assert n == len(low) == len(close) == len(open), "Arrays must be same length"

    high = high.astype(np.float64)
    low = low.astype(np.float64)
    close = close.astype(np.float64)

    # --- ATR (rolling 14-period) ---
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    # Cumsum trick for rolling mean ATR
    atr = np.full(n, np.nan)
    cs = np.cumsum(tr)
    atr[ATR_PERIOD - 1:] = (cs[ATR_PERIOD - 1:] - np.concatenate([[0], cs[:-ATR_PERIOD]])) / ATR_PERIOD
    # Backfill early bars with expanding mean
    for i in range(ATR_PERIOD - 1):
        atr[i] = cs[i] / (i + 1)

    # --- Vectorized FVG detection ---
    # Bullish: low[i] > high[i-2]  (bar i creates the gap looking back 2 bars)
    # Bearish: high[i] < low[i-2]
    fvg_bullish = np.zeros(n, dtype=bool)
    fvg_bearish = np.zeros(n, dtype=bool)

    if n >= 3:
        bull_gap = low[2:] - high[:-2]  # positive = bullish FVG
        bear_gap = low[:-2] - high[2:]  # positive = bearish FVG

        min_gaps = atr[2:] * min_size_atr

        fvg_bullish[2:] = bull_gap >= min_gaps
        fvg_bearish[2:] = bear_gap >= min_gaps

    # --- Track FVG zones and fill status ---
    # We iterate once through bars to track fills (unavoidable for state tracking)
    fvg_zones: List[Dict] = []
    inside_bull_fvg = np.zeros(n, dtype=bool)
    inside_bear_fvg = np.zeros(n, dtype=bool)
    dist_nearest_bull = np.zeros(n, dtype=np.float64)
    dist_nearest_bear = np.zeros(n, dtype=np.float64)

    # Collect all FVG creation events
    bull_indices = np.where(fvg_bullish)[0]
    bear_indices = np.where(fvg_bearish)[0]

    # Build sorted event list: (bar_idx, direction, top, bottom)
    events = []
    for i in bull_indices:
        top = float(low[i])         # upper boundary
        bottom = float(high[i - 2])  # lower boundary
        events.append((int(i), "bullish", top, bottom))
    for i in bear_indices:
        top = float(low[i - 2])     # upper boundary
        bottom = float(high[i])      # lower boundary
        events.append((int(i), "bearish", top, bottom))

    events.sort(key=lambda x: x[0])

    # Scan forward: maintain active FVG list, check fills, compute per-bar metrics
    active_fvgs: List[Dict] = []  # {start_idx, direction, top, bottom, filled, fill_idx}
    event_ptr = 0

    for bar in range(n):
        # Add new FVGs that start at this bar
        while event_ptr < len(events) and events[event_ptr][0] == bar:
            _, direction, top, bottom = events[event_ptr]
            active_fvgs.append({
                "start_idx": bar,
                "end_idx": -1,
                "direction": direction,
                "top": top,
                "bottom": bottom,
                "filled": False,
                "fill_idx": -1,
            })
            event_ptr += 1

        # Check fills and compute bar metrics
        price_h = high[bar]
        price_l = low[bar]
        price_mid = (price_h + price_l) / 2.0

        nearest_bull_dist = 0.0
        nearest_bear_dist = 0.0
        found_bull = False
        found_bear = False

        for fvg in active_fvgs:
            if fvg["filled"]:
                continue

            # Age check — expire old FVGs
            age = bar - fvg["start_idx"]
            if age > max_age:
                fvg["filled"] = True
                fvg["end_idx"] = bar
                fvg["fill_idx"] = bar
                continue

            # Fill check (only bars AFTER creation)
            if bar > fvg["start_idx"]:
                if fvg["direction"] == "bullish" and price_l <= fvg["bottom"]:
                    fvg["filled"] = True
                    fvg["end_idx"] = bar
                    fvg["fill_idx"] = bar
                    continue
                elif fvg["direction"] == "bearish" and price_h >= fvg["top"]:
                    fvg["filled"] = True
                    fvg["end_idx"] = bar
                    fvg["fill_idx"] = bar
                    continue

            # FVG still active — check if current bar is inside it
            if fvg["direction"] == "bullish":
                if price_l <= fvg["top"] and price_h >= fvg["bottom"]:
                    inside_bull_fvg[bar] = True
                # Distance: how far is price above the FVG top (0 if inside or below)
                dist = price_l - fvg["top"]
                if dist > 0:
                    # Price is above this bullish FVG
                    if not found_bull or dist < nearest_bull_dist:
                        nearest_bull_dist = dist
                        found_bull = True
            else:  # bearish
                if price_h >= fvg["bottom"] and price_l <= fvg["top"]:
                    inside_bear_fvg[bar] = True
                # Distance: how far is price below the FVG bottom (0 if inside or above)
                dist = fvg["bottom"] - price_h
                if dist > 0:
                    # Price is below this bearish FVG
                    if not found_bear or dist < nearest_bear_dist:
                        nearest_bear_dist = dist
                        found_bear = True

        dist_nearest_bull[bar] = nearest_bull_dist
        dist_nearest_bear[bar] = nearest_bear_dist

    # Final zone list: only unfilled ones at end of series
    final_zones = [
        {
            "start_idx": f["start_idx"],
            "end_idx": n - 1,
            "direction": f["direction"],
            "top": f["top"],
            "bottom": f["bottom"],
            "filled": f["filled"],
        }
        for f in active_fvgs
        if not f["filled"]
    ]

    return {
        "fvg_bullish": fvg_bullish,
        "fvg_bearish": fvg_bearish,
        "fvg_zones": final_zones,
        "inside_bull_fvg": inside_bull_fvg,
        "inside_bear_fvg": inside_bear_fvg,
        "dist_nearest_bull": dist_nearest_bull,
        "dist_nearest_bear": dist_nearest_bear,
    }


# --- Quick self-test ---
if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    # Generate synthetic trending price with gaps
    returns = np.random.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.002, n))

    result = detect_fvg(open_, high, low, close)

    bull_count = result["fvg_bullish"].sum()
    bear_count = result["fvg_bearish"].sum()
    zones_open = len(result["fvg_zones"])
    inside_bull = result["inside_bull_fvg"].sum()
    inside_bear = result["inside_bear_fvg"].sum()

    print(f"Bars: {n}")
    print(f"Bullish FVGs detected: {bull_count}")
    print(f"Bearish FVGs detected: {bear_count}")
    print(f"Active (unfilled) zones at end: {zones_open}")
    print(f"Bars inside bullish FVG: {inside_bull}")
    print(f"Bars inside bearish FVG: {inside_bear}")
    print(f"Max dist to nearest bull FVG: {result['dist_nearest_bull'].max():.4f}")
    print(f"Max dist to nearest bear FVG: {result['dist_nearest_bear'].max():.4f}")

    if zones_open > 0:
        print(f"\nSample active zones:")
        for z in result["fvg_zones"][:5]:
            print(f"  {z['direction']:8s} | bar {z['start_idx']:3d} | "
                  f"[{z['bottom']:.4f} - {z['top']:.4f}]")
