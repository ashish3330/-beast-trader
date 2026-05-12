"""
Multi-timeframe trend aggregation from H1 candles.

Industry-grade trend filter: instead of running separate D1/H4/W1 data
feeds (more cache, more state, more failure modes), aggregate from the
H1 candle stream we already have:

  H4  = 4 H1 candles aggregated (OHLC)
  D1  = 24 H1 candles aggregated
  W1  = 168 H1 candles aggregated

Trend direction per TF: EMA20 vs EMA50 vs EMA200 on the aggregated
closes. Same logic for live and backtest because both have H1 candles.

This is what AHL / Winton call "the trend cascade" — entry requires
alignment across MULTIPLE timeframes, not just one.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Bars-per-TF when aggregating from H1
BARS_PER_TF = {
    "M30": 0.5,   # half H1 (only used if we aggregate M30; current uses M15 directly)
    "H1":  1,
    "H4":  4,
    "H6":  6,
    "H12": 12,
    "D1":  24,
    "W1":  168,   # 24*7 — markets trade ~5d/week but use 168 to cover gap
}


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    if len(arr) < period:
        return np.array([np.nan] * len(arr))
    alpha = 2.0 / (period + 1.0)
    ema = np.zeros_like(arr, dtype=float)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    return ema


def _aggregate_closes(h1_closes: np.ndarray, bars_per: int) -> np.ndarray:
    """Take every Nth H1 close as a higher-TF close proxy.

    Faster than full OHLC aggregation and equivalent for trend-direction
    purposes (EMA on stride-sampled closes ≈ EMA on resampled closes).
    """
    if bars_per <= 1:
        return h1_closes
    # Stride-sample
    return h1_closes[::bars_per]


def trend_direction(h1_closes: np.ndarray, tf: str) -> str:
    """Return 'LONG', 'SHORT', or 'FLAT' for the given timeframe.

    Logic: EMA20 vs EMA50 vs EMA200 on aggregated closes:
      EMA20 > EMA50 > EMA200 → LONG
      EMA20 < EMA50 < EMA200 → SHORT
      else                   → FLAT
    """
    if h1_closes is None or len(h1_closes) < 10:
        return "FLAT"
    bars_per = BARS_PER_TF.get(tf, 1)
    tf_closes = _aggregate_closes(h1_closes, bars_per)
    # Need enough bars for EMA200 on the aggregated TF
    min_needed = 50 if tf in ("D1", "W1") else 200
    if len(tf_closes) < min_needed:
        # Not enough history for full trend; use shorter EMAs
        if len(tf_closes) < 10:
            return "FLAT"
        ema_s = _ema(tf_closes, min(10, len(tf_closes) // 2))
        ema_l = _ema(tf_closes, min(20, len(tf_closes) - 1))
        if np.isnan(ema_s[-1]) or np.isnan(ema_l[-1]):
            return "FLAT"
        if ema_s[-1] > ema_l[-1]:
            return "LONG"
        if ema_s[-1] < ema_l[-1]:
            return "SHORT"
        return "FLAT"

    ema20 = _ema(tf_closes, 20)
    ema50 = _ema(tf_closes, 50)
    ema200_period = min(200, len(tf_closes) - 1)
    ema200 = _ema(tf_closes, ema200_period)

    s, l, t = ema20[-1], ema50[-1], ema200[-1]
    if np.isnan(s) or np.isnan(l) or np.isnan(t):
        return "FLAT"
    if s > l > t:
        return "LONG"
    if s < l < t:
        return "SHORT"
    return "FLAT"


def precompute_mtf_trends(h1_closes: np.ndarray, tfs=("W1", "D1", "H4")) -> dict:
    """Vectorized precompute of trend direction per H1 bar per TF.

    Returns {tf_name: array of {-1, 0, 1}} same length as h1_closes:
      +1 = LONG, -1 = SHORT, 0 = FLAT.

    Use this ONCE before the trade-decision loop, then look up per bar.
    Avoids re-computing EMA on 500-bar windows for every signal.
    """
    out = {}
    n = len(h1_closes)
    if n < 30:
        return {tf: np.zeros(n, dtype=np.int8) for tf in tfs}

    for tf in tfs:
        bars_per = BARS_PER_TF.get(tf, 1)
        # Sample-stride the H1 closes to higher TF
        # For each H1 bar i, the higher-TF "current close" is the most
        # recent stride-sample at or before i.
        tf_closes = h1_closes[::bars_per] if bars_per > 1 else h1_closes
        if len(tf_closes) < 10:
            out[tf] = np.zeros(n, dtype=np.int8)
            continue

        # Compute EMAs on aggregated TF
        p20 = 20
        p50 = 50
        p200 = min(200, len(tf_closes) - 1)
        a20 = 2.0 / (p20 + 1)
        a50 = 2.0 / (p50 + 1)
        a200 = 2.0 / (p200 + 1)
        ema20 = np.zeros_like(tf_closes, dtype=float)
        ema50 = np.zeros_like(tf_closes, dtype=float)
        ema200 = np.zeros_like(tf_closes, dtype=float)
        ema20[0] = ema50[0] = ema200[0] = tf_closes[0]
        for i in range(1, len(tf_closes)):
            ema20[i] = a20 * tf_closes[i] + (1 - a20) * ema20[i - 1]
            ema50[i] = a50 * tf_closes[i] + (1 - a50) * ema50[i - 1]
            ema200[i] = a200 * tf_closes[i] + (1 - a200) * ema200[i - 1]

        # Direction at each TF bar
        # LONG: ema20 > ema50 > ema200
        # SHORT: ema20 < ema50 < ema200
        long_mask = (ema20 > ema50) & (ema50 > ema200)
        short_mask = (ema20 < ema50) & (ema50 < ema200)
        tf_dir = np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(np.int8)

        # Map back to H1-bar resolution by repeating
        # Each TF bar represents bars_per H1 bars
        if bars_per > 1:
            h1_dir = np.repeat(tf_dir, bars_per)[:n]
            # Pad if needed
            if len(h1_dir) < n:
                h1_dir = np.concatenate([h1_dir, np.zeros(n - len(h1_dir), dtype=np.int8)])
        else:
            h1_dir = tf_dir[:n] if len(tf_dir) >= n else np.concatenate(
                [tf_dir, np.zeros(n - len(tf_dir), dtype=np.int8)])

        out[tf] = h1_dir

    return out


def mtf_verdict_at_bar(precomputed: dict, bar_idx: int,
                        entry_direction: int) -> str:
    """Look up MTF verdict at a specific H1 bar.

    entry_direction: +1 (LONG) or -1 (SHORT).
    Returns 'SNIPER' | 'STRONG' | 'OK' | 'REJECT'.
    """
    aligned = 0
    opposed = 0
    flat = 0
    total = len(precomputed)
    for tf, dirs in precomputed.items():
        if bar_idx >= len(dirs):
            flat += 1
            continue
        d = dirs[bar_idx]
        if d == 0:
            flat += 1
        elif d == entry_direction:
            aligned += 1
        else:
            opposed += 1
    if opposed >= 2:
        return "REJECT"
    if opposed == 1:
        return "OK"
    if aligned == total:
        return "SNIPER"
    return "STRONG"


def mtf_cascade(h1_df: pd.DataFrame, entry_direction: str, tfs=("D1", "H4")) -> dict:
    """Compute MTF cascade alignment for an entry signal.

    Returns:
        {
          "tfs": {"D1": "LONG", "H4": "LONG", ...},
          "aligned": int (count of TFs matching entry_direction OR flat),
          "opposed": int (count of TFs opposite to entry direction),
          "total": int,
          "verdict": "SNIPER" | "STRONG" | "OK" | "REJECT"
        }

    verdict:
      - SNIPER: 0 opposed, all aligned (full strength signal)
      - STRONG: 0 opposed, some flat (most aligned)
      - OK:     1 opposed (mixed but proceedable)
      - REJECT: 2+ opposed (don't enter)
    """
    if h1_df is None or len(h1_df) < 30:
        return {"tfs": {}, "aligned": 0, "opposed": 0, "total": 0,
                "verdict": "OK"}
    closes = h1_df["close"].values if "close" in h1_df else h1_df.values

    tf_dirs = {tf: trend_direction(closes, tf) for tf in tfs}
    aligned = sum(1 for d in tf_dirs.values() if d == entry_direction)
    opposed = sum(1 for d in tf_dirs.values() if d != "FLAT" and d != entry_direction)
    flat = sum(1 for d in tf_dirs.values() if d == "FLAT")
    total = len(tfs)

    if opposed >= 2:
        verdict = "REJECT"
    elif opposed == 1:
        verdict = "OK"
    elif aligned == total:
        verdict = "SNIPER"
    else:
        verdict = "STRONG"

    return {
        "tfs": tf_dirs,
        "aligned": aligned,
        "opposed": opposed,
        "flat": flat,
        "total": total,
        "verdict": verdict,
    }
