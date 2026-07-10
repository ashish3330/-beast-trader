"""
One-off LIVE FVG execution test (DEMO). Drives the real
executor.open_trade_explicit() path to prove an FVG-magic order fills and can
be closed end-to-end. Opens a TINY 2-leg FVG position on EURUSD, confirms it,
then closes it immediately. Safe to run alongside the live bot.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, SYMBOLS
from execution.executor import Executor

SYMBOL = "EURUSD"


class StubState:
    def get_agent_state(self): return {"equity": 5000.0}
    def update_agent(self, *a, **k): pass
    def get_indicators(self, *a, **k): return {}
    def get_candles(self, *a, **k): return None


def connect():
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        raise RuntimeError(f"init failed: {mt5.last_error()}")
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"login failed: {mt5.last_error()}")
    return mt5


def fvg_positions(mt5, symbol):
    base = int(SYMBOLS[symbol].magic)
    fmag = {base + 1000, base + 1001}
    return [p for p in (mt5.positions_get(symbol=symbol) or []) if int(p.magic) in fmag]


def main():
    mt5 = connect()
    acct = mt5.account_info()
    print(f"Account {acct.login} ${acct.balance:.2f} on {acct.server}")
    ex = Executor(mt5, StubState())

    if ex.has_fvg_position(SYMBOL):
        print(f"[ABORT] {SYMBOL} already has an FVG position — pick another symbol."); return
    if ex.has_position(SYMBOL):
        print(f"[WARN] momentum holds {SYMBOL}; open_trade_explicit will still place FVG legs (separate magic).")

    tick = mt5.symbol_info_tick(SYMBOL)
    si = mt5.symbol_info(SYMBOL)
    entry = float(tick.ask)
    pip = 10 * float(si.point)              # ~1 pip = 10 points on a 5-digit quote
    sl  = round(entry - 100 * float(si.point), si.digits)   # ~10 pip stop
    tp1 = round(entry + 150 * float(si.point), si.digits)
    tp2 = round(entry + 300 * float(si.point), si.digits)
    base = int(SYMBOLS[SYMBOL].magic)
    print(f"\n[OPEN] {SYMBOL} LONG entry~{entry} SL={sl} TP1={tp1} TP2={tp2} "
          f"risk=0.05% FVG-magics={base+1000}/{base+1001}")

    ok = ex.open_trade_explicit(SYMBOL, "LONG", entry, sl, tp1, tp2, risk_pct=0.05)
    print(f"[OPEN] open_trade_explicit returned: {ok}")
    time.sleep(2)

    pos = fvg_positions(mt5, SYMBOL)
    print(f"\n[CONFIRM] FVG positions now open: {len(pos)}")
    for p in pos:
        print(f"   ticket={p.ticket} magic={p.magic} vol={p.volume} type={'BUY' if p.type==0 else 'SELL'} "
              f"open={p.price_open} sl={p.sl} tp={p.tp} pnl={p.profit:.2f}")

    if not pos:
        print("[RESULT] No FVG position opened — live execution path FAILED. Check log/retcode above.")
        mt5.shutdown(); return

    print("\n[CLOSE] closing FVG legs...")
    ex.close_fvg_position(SYMBOL, comment="FVG_live_test_close")
    time.sleep(2)
    remaining = fvg_positions(mt5, SYMBOL)
    print(f"[CLOSE] FVG positions remaining: {len(remaining)}")
    print(f"\n[RESULT] LIVE FVG EXECUTION PATH: {'OK (opened + closed)' if not remaining else 'OPENED but CLOSE INCOMPLETE — check manually'}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
