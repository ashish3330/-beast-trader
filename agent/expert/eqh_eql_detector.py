#!/usr/bin/env python3 -B
"""
Equal-Highs / Equal-Lows (EQH/EQL) Liquidity-Pool Detector
══════════════════════════════════════════════════════════

Locates clusters of approximately-equal recent swing highs (EQH) or lows
(EQL). Such clusters concentrate resting stop-orders and pending breakouts
— in ICT / SMC parlance they are "liquidity pools" institutions deliberately
sweep before reversing.

This module is a pure-function helper (no MT5, no journal, no logger
side effects). Designed to plug into ``agent/sweep_reclaim.py`` as an
optional filter on the swept_level: a sweep that takes out an EQH/EQL
cluster carries higher edge than a sweep of an isolated pivot.

Public API
──────────
    find_eqh_eql(highs, lows, lookback=50,
                 tolerance_atr=0.10, min_touches=2,
                 atr=None)
        Returns ``list[dict]`` of clusters:
            {
              "price_level": float,      # centroid of touch prices
              "type": "EQH" | "EQL",
              "touches": int,            # number of qualifying pivots
              "freshness_bars": int,     # bars since most recent touch
              "members": [int, ...],     # indices in source array (debug)
              "spread_atr": float,       # max-min within cluster / ATR
            }

    is_sweeping_eqh_eql(swept_price, eqh_eql_list,
                        tolerance_atr=0.20, atr=None)
        Returns ``True`` if swept_price is within ``tolerance_atr * ATR``
        of any cluster centroid; ``False`` otherwise (incl. empty list).

Design notes
────────────
  * Stateless: every call recomputes from the supplied OHLC arrays. Caller
    owns caching.
  * Fail-open: bad / short inputs → return ``[]`` from find, ``False`` from
    the sweep check. Never raises out to the caller.
  * ATR-relative tolerances: a "tolerance_atr" of 0.10 means "two highs are
    equal if within 0.10 × ATR14 of each other". When ATR is not provided,
    we compute Wilder ATR14 inline from the highs/lows/closes (closes
    approximated by the midpoint when not supplied).
  * Pivot detection: strict 3-bar fractal (centre strictly extreme vs ±1).
    Cheaper than a full SWING_WIN=3 fractal and adequate for liquidity-
    cluster identification (we care about local extremes, not deep pivots).
  * No external dependencies beyond numpy.

CLI
───
    python3 -B agent/expert/eqh_eql_detector.py --self-test
"""

from __future__ import annotations

import logging
import sys

import numpy as np

log = logging.getLogger("dragon.expert.eqh_eql")

# ────────────────────────────────────────────────────────────────────────
#  Defaults — small, conservative. Caller overrides per integration.
# ────────────────────────────────────────────────────────────────────────
DEFAULT_LOOKBACK         = 50
DEFAULT_TOLERANCE_ATR    = 0.10
DEFAULT_MIN_TOUCHES      = 2
DEFAULT_SWEEP_TOLERANCE  = 0.20
ATR_PERIOD               = 14


# ────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ────────────────────────────────────────────────────────────────────────
def _to_np(arr) -> np.ndarray | None:
    """Coerce a list / pandas Series / numpy array → 1-D float64 ndarray.

    Returns ``None`` if coercion fails or the array is empty.
    """
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=float).reshape(-1)
        if a.size == 0:
            return None
        return a
    except Exception as e:  # pragma: no cover — defensive
        log.debug("eqh_eql._to_np failed: %s", e)
        return None


def _wilder_atr(H: np.ndarray, L: np.ndarray, C: np.ndarray | None,
                period: int = ATR_PERIOD) -> float:
    """Wilder ATR over numpy arrays — returns latest value (or 0).

    When ``C`` is None, we approximate with the bar midpoint (H+L)/2. This
    is good enough for tolerance scaling, where the absolute value only
    needs to be in the right ballpark.
    """
    n = len(H)
    if n < period + 1:
        return 0.0
    if C is None or len(C) != n:
        C = (H + L) / 2.0
    try:
        tr = np.empty(n)
        tr[0] = H[0] - L[0]
        for i in range(1, n):
            tr[i] = max(
                H[i] - L[i],
                abs(H[i] - C[i - 1]),
                abs(L[i] - C[i - 1]),
            )
        atr = np.empty(n)
        atr[period] = tr[1:period + 1].mean()
        k = 1.0 / period
        for i in range(period + 1, n):
            atr[i] = atr[i - 1] * (1 - k) + tr[i] * k
        return float(atr[-1])
    except Exception as e:  # pragma: no cover — defensive
        log.debug("eqh_eql._wilder_atr failed: %s", e)
        return 0.0


def _fractal_pivots(values: np.ndarray, kind: str) -> list[int]:
    """Return indices of strict 3-bar fractal pivots in ``values``.

    kind="high" → values[i] > values[i-1] and values[i] > values[i+1]
    kind="low"  → values[i] < values[i-1] and values[i] < values[i+1]

    The endpoints are excluded (need ±1 neighbours). Equality is *not* a
    pivot — keeps things strict so a long flat region doesn't generate
    spurious pivots.
    """
    out: list[int] = []
    n = len(values)
    if n < 3:
        return out
    if kind == "high":
        for i in range(1, n - 1):
            v = values[i]
            if v > values[i - 1] and v > values[i + 1]:
                out.append(i)
    elif kind == "low":
        for i in range(1, n - 1):
            v = values[i]
            if v < values[i - 1] and v < values[i + 1]:
                out.append(i)
    return out


def _cluster_pivots(
    pivot_indices: list[int],
    pivot_values: np.ndarray,
    tolerance: float,
    min_touches: int,
    array_len: int,
    kind: str,
) -> list[dict]:
    """Greedy single-pass clustering: sort pivots by price, group those
    within ``tolerance`` (absolute price units) of each other.

    Returns ``min_touches+`` clusters as dicts.
    """
    if not pivot_indices or tolerance <= 0:
        return []

    # Pair (price, original_idx) sorted by price.
    pairs = sorted(
        [(float(pivot_values[i]), int(i)) for i in pivot_indices],
        key=lambda t: t[0],
    )

    clusters: list[list[tuple[float, int]]] = []
    cur: list[tuple[float, int]] = [pairs[0]]
    for price, idx in pairs[1:]:
        # If within tolerance of the *running centroid*, join cluster.
        cur_centroid = sum(p for p, _ in cur) / len(cur)
        if abs(price - cur_centroid) <= tolerance:
            cur.append((price, idx))
        else:
            clusters.append(cur)
            cur = [(price, idx)]
    clusters.append(cur)

    out: list[dict] = []
    for c in clusters:
        if len(c) < min_touches:
            continue
        prices = [p for p, _ in c]
        idxs   = [i for _, i in c]
        centroid = sum(prices) / len(prices)
        spread   = max(prices) - min(prices)
        freshness = array_len - 1 - max(idxs)
        out.append({
            "price_level":    float(centroid),
            "type":           "EQH" if kind == "high" else "EQL",
            "touches":        len(c),
            "freshness_bars": int(freshness),
            "members":        sorted(idxs),
            "spread":         float(spread),
        })
    return out


# ────────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────────
def find_eqh_eql(
    highs,
    lows,
    lookback: int = DEFAULT_LOOKBACK,
    tolerance_atr: float = DEFAULT_TOLERANCE_ATR,
    min_touches: int = DEFAULT_MIN_TOUCHES,
    atr: float | None = None,
    closes=None,
) -> list[dict]:
    """Find Equal-High / Equal-Low clusters in the last ``lookback`` bars.

    Parameters
    ──────────
    highs, lows  : array-like — full OHLC high / low series.
    lookback     : int        — search window (counted back from the most
                                recent bar). The pivots themselves must
                                fall inside this window.
    tolerance_atr: float      — two pivots are "equal" if within
                                tolerance_atr × ATR14.
    min_touches  : int        — minimum touches for a cluster to qualify.
    atr          : float?     — caller-supplied ATR14. If None we compute
                                it from ``highs/lows[/closes]``.
    closes       : optional   — passed through to ATR for accuracy.

    Returns
    ───────
    list[dict] — sorted by ``freshness_bars`` ascending (most recent first).
    Empty list on any failure / no qualifying cluster.
    """
    try:
        H = _to_np(highs)
        L = _to_np(lows)
        if H is None or L is None or len(H) != len(L):
            return []
        n = len(H)
        if n < 5 or lookback < 3 or min_touches < 1:
            return []

        # Limit search window to last `lookback` bars.
        lookback = int(min(lookback, n))
        H_win = H[-lookback:]
        L_win = L[-lookback:]
        C_full = _to_np(closes) if closes is not None else None
        if C_full is not None and len(C_full) >= lookback:
            C_win = C_full[-lookback:]
        else:
            C_win = None

        # ATR resolution: caller can pass it pre-computed (cheap) or we
        # compute it from the window. Fall back to a tiny epsilon so the
        # tolerance product is never literally zero (which would collapse
        # every pivot to its own cluster).
        atr_val = float(atr) if atr is not None and atr > 0 else _wilder_atr(
            H_win, L_win, C_win, period=ATR_PERIOD
        )
        if atr_val <= 0:
            # Last-resort proxy: average bar range / 10.
            atr_val = float(np.mean(H_win - L_win)) if len(H_win) else 0.0
        if atr_val <= 0:
            return []
        tol_price = float(tolerance_atr) * atr_val

        # Pivots — strict 3-bar fractals inside the window.
        hi_piv = _fractal_pivots(H_win, "high")
        lo_piv = _fractal_pivots(L_win, "low")

        eqh = _cluster_pivots(hi_piv, H_win, tol_price, int(min_touches),
                              len(H_win), "high")
        eql = _cluster_pivots(lo_piv, L_win, tol_price, int(min_touches),
                              len(L_win), "low")

        out = eqh + eql
        out.sort(key=lambda d: d["freshness_bars"])
        return out
    except Exception as e:  # pragma: no cover — defensive
        log.debug("find_eqh_eql failed: %s", e)
        return []


def is_sweeping_eqh_eql(
    swept_price: float,
    eqh_eql_list: list[dict],
    tolerance_atr: float = DEFAULT_SWEEP_TOLERANCE,
    atr: float | None = None,
    cluster_type: str | None = None,
) -> bool:
    """Return True if ``swept_price`` is within ``tolerance_atr × ATR`` of
    any qualifying cluster centroid.

    Parameters
    ──────────
    swept_price   : float  — the wick extreme that broke the swing pivot.
    eqh_eql_list  : list   — output of ``find_eqh_eql``.
    tolerance_atr : float  — proximity threshold (×ATR).
    atr           : float? — ATR14 in price units. If None and the cluster
                             dicts don't carry one, we fall back to using
                             ``tolerance_atr`` itself as a fractional
                             distance threshold (clip ±tolerance_atr * price
                             / 100 — last-resort, very loose).
    cluster_type  : str?   — restrict to "EQH" or "EQL" only. None = both.

    Returns
    ───────
    bool — True on first match, False otherwise. Empty / malformed input
    returns False (fail-open from the *signal* side: filter does nothing).
    """
    try:
        if not eqh_eql_list or swept_price is None:
            return False
        price = float(swept_price)
        if not np.isfinite(price):
            return False

        # Tolerance resolution: prefer caller's ATR, else fall back to a
        # crude % of price proxy.
        if atr is not None and atr > 0:
            tol = float(tolerance_atr) * float(atr)
        else:
            tol = float(tolerance_atr) * abs(price) / 100.0

        if tol <= 0:
            return False

        for cluster in eqh_eql_list:
            if not isinstance(cluster, dict):
                continue
            if cluster_type and cluster.get("type") != cluster_type:
                continue
            level = cluster.get("price_level")
            if level is None:
                continue
            if abs(price - float(level)) <= tol:
                return True
        return False
    except Exception as e:  # pragma: no cover — defensive
        log.debug("is_sweeping_eqh_eql failed: %s", e)
        return False


# ────────────────────────────────────────────────────────────────────────
#  Self-test  — `python3 -B agent/expert/eqh_eql_detector.py --self-test`
# ────────────────────────────────────────────────────────────────────────
def _run_self_test() -> int:
    """Synthesize a 120-bar series with a deliberate EQH and EQL cluster,
    confirm both detection and sweep-membership pass.
    """
    rng = np.random.default_rng(7)
    n = 120
    closes = 100.0 + np.cumsum(rng.normal(0, 0.20, n))
    highs  = closes + np.abs(rng.normal(0, 0.15, n))
    lows   = closes - np.abs(rng.normal(0, 0.15, n))

    # Plant 3 equal highs near 102.50 at indices 40, 70, 95.
    for j in (40, 70, 95):
        highs[j] = 102.50
        # Make sure the neighbours are strictly lower (fractal pivot).
        for off in (-1, 1):
            highs[j + off] = min(highs[j + off], 102.30)

    # Plant 3 equal lows near 97.80 at indices 50, 75, 100.
    for j in (50, 75, 100):
        lows[j] = 97.80
        for off in (-1, 1):
            lows[j + off] = max(lows[j + off], 98.00)

    clusters = find_eqh_eql(highs, lows, lookback=120,
                            tolerance_atr=0.50, min_touches=2)
    print(f"[self-test] clusters found: {len(clusters)}")
    for c in clusters:
        print(f"    {c['type']} @ {c['price_level']:.4f}  touches={c['touches']}"
              f"  freshness={c['freshness_bars']}  spread={c['spread']:.4f}")

    eqh_hits = [c for c in clusters if c["type"] == "EQH"]
    eql_hits = [c for c in clusters if c["type"] == "EQL"]
    assert eqh_hits, "EQH cluster not detected"
    assert eql_hits, "EQL cluster not detected"
    assert any(abs(c["price_level"] - 102.50) < 0.10 for c in eqh_hits), \
        "EQH centroid wrong"
    assert any(abs(c["price_level"] - 97.80) < 0.10 for c in eql_hits), \
        "EQL centroid wrong"

    atr_proxy = float(np.mean(highs - lows))
    # Sweep prices: a wick that prints just above the EQH and just below the EQL.
    assert is_sweeping_eqh_eql(102.55, clusters,
                               tolerance_atr=0.50, atr=atr_proxy), \
        "EQH sweep not recognized"
    assert is_sweeping_eqh_eql(97.75, clusters,
                               tolerance_atr=0.50, atr=atr_proxy), \
        "EQL sweep not recognized"

    # Negative case: a wick far from any cluster must NOT register.
    assert not is_sweeping_eqh_eql(95.00, clusters,
                                   tolerance_atr=0.50, atr=atr_proxy), \
        "False positive on far-away price"

    # Fail-open: empty list / nonsense input must return False, never raise.
    assert is_sweeping_eqh_eql(100.0, [], tolerance_atr=0.50) is False
    assert is_sweeping_eqh_eql(float("nan"), clusters,
                               tolerance_atr=0.50, atr=atr_proxy) is False
    assert find_eqh_eql(None, None) == []
    assert find_eqh_eql([], []) == []

    print("[self-test] OK — EQH/EQL detector + sweep filter pass")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(_run_self_test())
    print("agent/expert/eqh_eql_detector.py — pass --self-test to run checks")
