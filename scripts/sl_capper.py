#!/usr/bin/env python3 -B
"""SL CAPPER — every run, cap the stop-loss of EVERY open position (incl.
MANUAL trades) to a max distance from entry. Only ever TIGHTENS, never loosens.

Rule (user req 2026-07-05): any position whose SL sits > CAP from entry (or has
NO SL) gets its SL moved to exactly CAP from entry, on the loss side.

Unit: CAP is a raw PRICE distance (price units), per-symbol overridable below.
Run every 5 min via launchd (see launchd/com.dragon.slcapper.plist).

Safety:
  * DRY_RUN=1 (default) → logs intended changes, sends nothing.
  * Never widens an existing stop.
  * Respects broker trade_stops_level (clamps to tightest legal level).
  * If price already worse than CAP against entry, sets the tightest LEGAL stop
    below/above current price to cap further loss (does not force-close).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER  # noqa: E402

DRY_RUN = os.getenv("SLCAP_DRY_RUN", "1") == "1"

# Max SL distance from entry, in PRICE units, per symbol. GOLD ONLY by user req
# (2026-07-05). Symbols not listed here are LEFT ALONE (BTC, forex, etc.).
CAP_PRICE_DIST = {
    "XAUUSD": float(os.getenv("SLCAP_XAUUSD", "50")),
}
# 0 = do NOT cap unlisted symbols (leave their SL untouched, incl. manual).
CAP_DEFAULT = float(os.getenv("SLCAP_DEFAULT", "0"))
STOP_BUFFER_PTS = 5  # extra broker-points beyond trade_stops_level for safety

# MANUAL-ONLY guard (2026-07-10): the cap must hit MANUAL XAU trades only — NOT
# the bot's own strategy legs. The TREND book now trades XAU with a wide 3xATR
# stop (~$100+) and a per-symbol tuned exit model; clamping it to 50 would wreck
# it. The SCALPER (+5000) also owns XAU. Every bot position has magic >= 8000
# (base magics start at 8100); manual GUI trades have magic 0. So skip magic
# >= BOT_MAGIC_FLOOR. Set SLCAP_INCLUDE_BOT=1 to cap bot legs too (not advised).
BOT_MAGIC_FLOOR = int(os.getenv("SLCAP_BOT_MAGIC_FLOOR", "8000"))
INCLUDE_BOT = os.getenv("SLCAP_INCLUDE_BOT", "0") == "1"


def _connect():
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        raise RuntimeError(f"init failed: {mt5.last_error()}")
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"login failed: {mt5.last_error()}")
    return mt5


def _log(msg):
    print(msg, flush=True)


def process(mt5):
    positions = mt5.positions_get() or []
    _log(f"[slcap] {'DRY-RUN' if DRY_RUN else 'LIVE'} — {len(positions)} open position(s)")
    modified = 0
    for p in positions:
        sym = p.symbol
        # MANUAL-ONLY: never touch the bot's own strategy legs (trend/scalper/etc.)
        if not INCLUDE_BOT and int(p.magic) >= BOT_MAGIC_FLOOR:
            continue
        si = mt5.symbol_info(sym)
        tick = mt5.symbol_info_tick(sym)
        if si is None or tick is None:
            _log(f"[slcap] {sym} #{p.ticket} no symbol_info/tick — skip")
            continue
        point = si.point
        digits = si.digits
        is_buy = int(p.type) == 0
        entry = float(p.price_open)
        sl = float(p.sl)
        cap = CAP_PRICE_DIST.get(sym, CAP_DEFAULT)
        if cap <= 0:      # unlisted symbol (e.g. BTC) — leave its SL alone
            continue
        cur = float(tick.bid) if is_buy else float(tick.ask)
        cur_dist = abs(entry - sl) if sl > 0 else float("inf")

        # Only act if the current stop is beyond the cap (or missing).
        if cur_dist <= cap + point:
            continue

        # Desired stop = CAP from entry, on the loss side.
        desired = entry - cap if is_buy else entry + cap
        # Broker minimum stop distance from CURRENT price.
        min_gap = (si.trade_stops_level + STOP_BUFFER_PTS) * point
        if is_buy:
            legal_max = cur - min_gap          # SL must be <= this for a BUY
            new_sl = min(desired, legal_max)    # can't sit above legal_max
            # never loosen: keep tighter of (existing sl, new_sl) but new must be > sl (tighter=closer to price=higher)
            if sl > 0:
                new_sl = max(new_sl, sl)
            valid = new_sl < cur
        else:
            legal_min = cur + min_gap          # SL must be >= this for a SELL
            new_sl = max(desired, legal_min)
            if sl > 0:
                new_sl = min(new_sl, sl)
            valid = new_sl > cur

        new_sl = round(new_sl, digits)
        if not valid or (sl > 0 and abs(new_sl - sl) < point):
            _log(f"[slcap] {sym} #{p.ticket} {'BUY' if is_buy else 'SELL'} "
                 f"entry={entry:.{digits}f} sl={sl:.{digits}f} cur_dist={cur_dist:.2f} "
                 f"> cap={cap} but no valid tighten (price moved) — skip")
            continue

        _log(f"[slcap] {sym} #{p.ticket} {'BUY' if is_buy else 'SELL'} magic={p.magic} "
             f"entry={entry:.{digits}f} old_sl={sl:.{digits}f} (dist {cur_dist:.2f}) "
             f"-> new_sl={new_sl:.{digits}f} (dist {abs(entry-new_sl):.2f})")
        if DRY_RUN:
            continue
        req = {"action": int(6), "symbol": str(sym), "position": int(p.ticket),
               "sl": float(new_sl), "tp": float(p.tp), "magic": int(p.magic)}
        res = mt5.order_send(req)
        rc = int(res.retcode) if res is not None else -1
        if rc in (10009, 10025):
            modified += 1
            _log(f"[slcap]   OK retcode={rc}")
        else:
            _log(f"[slcap]   FAIL retcode={rc} comment={getattr(res,'comment','?')}")
    _log(f"[slcap] done — {modified} modified")


def main():
    mt5 = _connect()
    try:
        process(mt5)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
