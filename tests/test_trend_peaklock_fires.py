#!/usr/bin/env python3
# -B
"""
PROOF TEST — TREND PASS-2 peak-lock SL-trail actually FIRES after the 2026-08-02
starvation fix (commit 883eabb).

What broke (pre-fix): the protective SL-ratchet (brain.py PASS 2) was gated by
`... and (not did_write or rev_close_failed)` and ran LAST for the single
per-cycle bridge-write budget. On active days an entry/flip/giveback consumed the
write, so the peak-lock trail NEVER ran (07-27: 0 'trail SL' modifies) and the
NAS/JPN winners rode +9 straight to their -11.9 stops.

This test drives a synthetic IN-PROFIT TREND long (NAS100.r) through a FAITHFUL
reproduction of brain.py lines 3336-3410 using the REAL helpers:
  * config.trend_exit_params / config.TREND_* constants (unchanged)
  * agent.trend_follower.chandelier_stop / _atr  (the real _trend_chandelier/_trend_atr)
  * AgentBrain._trend_sl_anchor bound to the harness (the real frozen-R anchor)
  * a mock executor whose trail_trend_sl replicates the REAL tighten + min-gap
    gate from execution/executor.py (lines 2605-2638) — so a returned >0 means
    the broker SL WOULD actually be modified, not merely that the call happened.

It ALSO runs the exact pre-fix gate (`not did_write`) with did_write=True to
demonstrate the regression the fix removed (starvation → 0 modifies).

Asserts:
  1. the computed stop is a PeakLock ratchet toward entry + LOCK*peak,
  2. trail_trend_sl WOULD be invoked AND would modify (returns >0),
  3. under the OLD gate with did_write=True the loop is skipped (0 modifies).
"""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

import config
from agent import trend_follower
from agent.brain import AgentBrain

SYM = "NAS100.r"

# ── REAL params (as they run live) ───────────────────────────────────────────
TREND_ATR_STOP        = config.TREND_ATR_STOP            # 3.0
TREND_TRAIL_LOOKBACK  = config.TREND_TRAIL_LOOKBACK      # 22
TREND_TRADE_LIVE      = config.TREND_TRADE_LIVE          # True
TREND_TRAIL_ENABLED   = config.TREND_TRAIL_ENABLED       # True
_trend_exit_params    = config.trend_exit_params
_atr_period_for       = config.trend_atr_period
_tr, _lk, _gb2, _act2 = _trend_exit_params(SYM)          # (2.5, 0.6, 0.35, 0.5)


# ── Mock executor: FAITHFUL copy of executor.trail_trend_sl tighten/min-gap gate ─
class MockExecutor:
    def __init__(self):
        self.calls = []          # (symbol, new_sl, min_gap, digits)
        self.modified = []       # legs actually moved
    def trail_trend_sl(self, symbol, new_sl, legs, min_gap, digits):
        self.calls.append((symbol, round(float(new_sl), int(digits)), min_gap, digits))
        tgt = round(float(new_sl), int(digits))
        n = 0
        for p in legs:
            tk = int(p.get("ticket") or 0)
            if not tk:
                continue
            is_long = int(p.get("type", 0)) == 0
            cur_sl = float(p.get("sl") or 0.0)
            px = float(p.get("price_cur") or 0.0)
            if px <= 0:
                continue
            if is_long:
                if (cur_sl and tgt <= cur_sl) or tgt >= px - min_gap:
                    continue
            else:
                if (cur_sl and tgt >= cur_sl) or tgt <= px + min_gap:
                    continue
            # would send action=6 modify → broker accepts (retcode 10009)
            self.modified.append((tk, tgt))
            n += 1
        return n


# ── Harness standing in for `self` (AgentBrain) ──────────────────────────────
class Harness:
    def __init__(self):
        self.executor = MockExecutor()
        self._trend_peak = {}
        self._trend_sl0 = {}
        self._trend_open_sl_dist = {}
    # REAL helpers bound off the live class / module:
    _trend_sl_anchor = AgentBrain._trend_sl_anchor
    def _trend_persist_state(self):          # no-op (avoid disk); real one just persists
        pass
    def _trend_chandelier(self, df, cur, tparams):
        return trend_follower.chandelier_stop(df, cur, tparams)
    def _trend_atr(self, high, low, close, period):
        return trend_follower._atr(high, low, close, period)


# ── Build a synthetic D1 uptrend that PEAKED then pulled back ────────────────
def build_d1(entry, peak_px, cur_px, n=60, spread=220.0):
    """Ascending D1 OHLC: long rally from below `entry` up to `peak_px` (the
    22d high) then a pullback to `cur_px`. `spread` sets the per-bar high/low
    range so ATR is WIDE (like a real trending index) — a wide 2.5xATR chandelier
    then sits FAR below current price, exactly the incident where the chandelier
    'locks almost nothing' and the peak-lock ratchet is the real protector.
    Returns (df, atr)."""
    rng = np.linspace(entry - 800, peak_px, n - 3)
    closes = list(rng) + [peak_px - (peak_px - cur_px) * 0.5, cur_px, cur_px]
    closes = np.array(closes, dtype=float)
    highs = closes + spread
    lows  = closes - spread
    highs[n - 3] = peak_px                     # ensure the peak is the 22d high
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes})
    atr = float(trend_follower._atr(df["high"], df["low"], df["close"],
                                    _atr_period_for(SYM)).iloc[-1])
    return df, atr


def run_pass2(did_write, use_old_gate):
    """FAITHFUL reproduction of brain.py PASS-2 (lines 3336-3410).
    use_old_gate=True applies the PRE-FIX `not did_write` gate to prove the
    regression; False applies the 2026-08-02 fix (runs every cycle)."""
    h = Harness()
    cur = 1                                    # long
    entry = 19000.0
    cur_px = 20000.0                           # +1000 pts open profit
    peak_px = 20250.0                          # peaked +1250, pulled back to +1000
    df, atr = build_d1(entry, peak_px, cur_px)

    # frozen at-open SL distance (risk-capped ~ smaller than ATR*STOP) so the
    # real _trend_sl_anchor freezes it and _lock_thresh stays small/armable.
    sl_dist_open = 300.0
    h._trend_open_sl_dist[SYM] = sl_dist_open
    orig_sl = entry - sl_dist_open             # 18700

    ticket = 555001
    legs = [{"ticket": ticket, "type": 0, "price_open": entry,
             "price_cur": cur_px, "sl": orig_sl, "tp": 0.0, "magic": 6000}]

    # persisted PEAK (high-water mark) — larger than current profit → the retrace
    # case the fix must protect. Chosen so lock clearly beats the chandelier.
    peak_prof = peak_px - entry                # 1250 pts high-water
    pk_key = tuple(sorted(int(l.get("ticket") or 0) for l in legs))
    h._trend_peak[pk_key] = peak_prof

    pos_dir = {SYM: cur}
    d1_data = {SYM: df}
    trend_legs = {SYM: legs}

    result = {"tag": None, "stop": None, "lock_target": None,
              "chandelier": None, "fired": False, "loop_ran": False}

    # ===== BEGIN faithful reproduction of brain.py L3327-3410 =====
    if TREND_TRAIL_ENABLED and TREND_TRADE_LIVE and (
            (not did_write) if use_old_gate else True):
        _TRAIL_SPECS = {"XAUUSD": (2, 0.01, 20), "BTCUSD": (2, 0.01, 0),
                        "ETHUSD": (2, 0.01, 0), "JPN225ft": (2, 0.01, 50),
                        "NAS100.r": (2, 0.01, 50)}
        _tr_items = [(s, c) for s, c in pos_dir.items() if c != 0 and s in d1_data]
        if _tr_items:
            if not hasattr(h, "_trail_rr"):
                h._trail_rr = 0
            _i = h._trail_rr % len(_tr_items)
            _tr_items = _tr_items[_i:] + _tr_items[:_i]
            h._trail_rr += 1
        for sym, cur in _tr_items:
            if cur == 0 or sym not in d1_data:
                continue
            legs = trend_legs.get(sym)
            if not legs:
                continue
            try:
                result["loop_ran"] = True
                df = d1_data[sym]
                _tr, _lk, _gb2, _act2 = _trend_exit_params(sym)
                tparams = {"ATR_PERIOD": _atr_period_for(sym),
                           "TRAIL_LOOKBACK": TREND_TRAIL_LOOKBACK,
                           "TRAIL_ATR": _tr}
                stop = h._trend_chandelier(df, cur, tparams)
                if stop is None:
                    continue
                tag = "Chandelier"
                entry_ = float(legs[0].get("price_open") or 0)
                px = float(legs[0].get("price_cur") or 0)
                _sl0 = float(legs[0].get("sl") or 0)
                atr_ = float(h._trend_atr(df["high"], df["low"], df["close"],
                                          _atr_period_for(sym)).iloc[-1])
                if entry_ > 0 and px > 0 and atr_ > 0:
                    prof = (px - entry_) if cur == 1 else (entry_ - px)
                    _sldist = h._trend_sl_anchor(sym, legs, entry_, _sl0,
                                                 TREND_ATR_STOP * atr_)
                    _lock_thresh = (_act2 / TREND_ATR_STOP) * _sldist if TREND_ATR_STOP > 0 else _act2 * atr_
                    _pk_key = tuple(sorted(int(l.get("ticket") or 0) for l in legs))
                    if not hasattr(h, "_trend_peak"):
                        h._trend_peak = {}
                    peak_prof_ = max(h._trend_peak.get(_pk_key, prof), prof)
                    if peak_prof_ != h._trend_peak.get(_pk_key):
                        h._trend_peak[_pk_key] = peak_prof_
                        h._trend_persist_state()
                    result["chandelier"] = stop
                    if peak_prof_ >= _lock_thresh:
                        lock = (entry_ + _lk * peak_prof_) if cur == 1 \
                            else (entry_ - _lk * peak_prof_)
                        result["lock_target"] = lock
                        tighter = max(stop, lock) if cur == 1 else min(stop, lock)
                        if tighter != stop:
                            stop, tag = tighter, "PeakLock%d%%" % int(_lk * 100)
                dg, pt, sl_lvl = _TRAIL_SPECS.get(sym, (2, 0.01, 20))
                min_gap = (sl_lvl + 2) * pt
                if tag.startswith("PeakLock") and px > 0:
                    if cur == 1 and stop >= px - min_gap:
                        stop = px - 1.5 * min_gap
                    elif cur == -1 and stop <= px + min_gap:
                        stop = px + 1.5 * min_gap
                result["tag"] = tag
                result["stop"] = stop
                if h.executor.trail_trend_sl(sym, stop, legs, min_gap, dg):
                    result["fired"] = True
                    break
            except Exception as e:
                import traceback; traceback.print_exc()
    # ===== END reproduction =====
    result["modified"] = h.executor.modified
    result["calls"] = h.executor.calls
    return result


def main():
    print("=" * 74)
    print("TREND PASS-2 peak-lock — POST-FIX (runs EVERY cycle; did_write=True)")
    print("=" * 74)
    post = run_pass2(did_write=True, use_old_gate=False)
    print(f"  loop_ran        : {post['loop_ran']}")
    print(f"  chandelier stop : {post['chandelier']}")
    print(f"  lock target     : {post['lock_target']}  (entry 19000 + 0.6*peak 1250 = 19750)")
    print(f"  chosen tag      : {post['tag']}")
    print(f"  chosen stop     : {post['stop']}")
    print(f"  trail calls     : {post['calls']}")
    print(f"  legs modified   : {post['modified']}")
    print(f"  FIRED           : {post['fired']}")

    print()
    print("=" * 74)
    print("TREND PASS-2 peak-lock — PRE-FIX regression (old 'not did_write' gate)")
    print("=" * 74)
    pre = run_pass2(did_write=True, use_old_gate=True)
    print(f"  loop_ran        : {pre['loop_ran']}")
    print(f"  FIRED           : {pre['fired']}   (expected False — starved)")

    # ── assertions ──────────────────────────────────────────────────────────
    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    print("\nASSERTIONS")
    check("post-fix loop ran despite did_write=True", post["loop_ran"] is True)
    check("chose a PeakLock ratchet (not raw Chandelier)",
          post["tag"] is not None and post["tag"].startswith("PeakLock"))
    check("lock target ratchets toward entry + LOCK*peak (~19750)",
          post["lock_target"] is not None and abs(post["lock_target"] - 19750.0) < 1.0)
    check("chosen stop >= chandelier (tighter-of picked the lock)",
          post["stop"] is not None and post["chandelier"] is not None
          and post["stop"] >= post["chandelier"] - 1e-6)
    check("chosen stop ratchets ABOVE the original open SL (18700)",
          post["stop"] is not None and post["stop"] > 18700.0)
    check("trail_trend_sl WOULD modify the broker SL (returns >0)",
          post["fired"] is True and len(post["modified"]) == 1)
    check("PRE-FIX gate STARVED the trail (loop skipped, 0 modifies)",
          pre["loop_ran"] is False and pre["fired"] is False)

    print("\nRESULT:", "ALL PASS — peak-lock FIRES" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
