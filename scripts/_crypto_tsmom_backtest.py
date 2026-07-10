#!/usr/bin/env python3 -B
"""CRYPTO TSMOM-LF backtest (2026-07-08) — BTCUSD + ETHUSD, long/flat daily trend.

Rules (identical math to agent/crypto_tsmom.py — parity-checked below):
  ENTRY  (closed D1 bar, fill next open): close>SMA200 AND close>close[90]
         AND close = new 20-day Donchian(close) high. BTC extra: min-lot SL
         risk (3xATR x $/pt x 0.01) <= 3.0% of equity, else SKIP.
  SL     broker-side, signal close - 3.0xATR(20). No TP.
  EXIT   3.5xATR Chandelier from highest close since entry (up-only), or
         D1 close<SMA200 -> out next open.

REAL COSTS charged (live MT5 specs pulled 2026-07-08, VantageMarkets-Demo):
  BTCUSD spread $17.00, ETHUSD $2.46; slippage 0.05% per side;
  SWAP mode 5 (interest on price): BTC -25%/yr, ETH -38%/yr, charged per
  CALENDAR day held (weekends included == triple-day handled) on avg price.
Sizing: fixed min lot 0.01 both symbols ($0.01 per $1 price move).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.crypto_tsmom import evaluate as detector_evaluate  # noqa: E402

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
EQUITY = 2355.0
LOTS = 0.01
DPP = 1.0          # $ per $1 price move per 1.0 lot (tick_value/tick_size)
SLIP_FRAC = 0.0005  # 0.05% per side
# sym: (spread_$, swap_long %/yr, btc_gate)
SPEC = {
    "BTCUSD": (17.00, -25.0, True),
    "ETHUSD": (2.46, -38.0, False),
}
SHIP = dict(SMA_PERIOD=200, ROC_PERIOD=90, DON_PERIOD=20, ATR_PERIOD=20,
            SL_ATR_MULT=3.0, TRAIL_ATR_MULT=3.5, BTC_MAX_RISK_PCT=3.0)


def _atr(h, l, c, n):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def load(sym):
    df = pickle.load(open(C / f"raw_d1_{sym}.pkl", "rb"))
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def backtest(df, sym, p):
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    t = df["time"].values
    n = len(df)
    sma_p, roc_p, don_p = p["SMA_PERIOD"], p["ROC_PERIOD"], p["DON_PERIOD"]
    slm, trm = p["SL_ATR_MULT"], p["TRAIL_ATR_MULT"]
    sma = pd.Series(c).rolling(sma_p).mean().values
    don = pd.Series(c).rolling(don_p).max().values
    atr = _atr(df["high"], df["low"], df["close"], p["ATR_PERIOD"]).values
    spread, swap_pct, gate = SPEC[sym]
    gate_cap = EQUITY * p["BTC_MAX_RISK_PCT"] / 100.0
    dpp_min = DPP * LOTS

    trades = []
    i = max(sma_p, roc_p, don_p, p["ATR_PERIOD"])
    while i < n - 1:
        ok = (c[i] > sma[i] and c[i] > c[i - roc_p] and c[i] >= don[i]
              and np.isfinite(atr[i]) and atr[i] > 0)
        if ok and gate and slm * atr[i] * dpp_min > gate_cap:
            ok = False  # BTC vol gate: implied min-lot risk too big -> SKIP
        if not ok:
            i += 1
            continue
        entry = o[i + 1] + spread + SLIP_FRAC * o[i + 1]   # buy at ask + slip
        sl = c[i] - slm * atr[i]
        if o[i + 1] <= sl:                                  # gapped through SL
            i += 1
            continue
        hh = c[i]
        j = i + 1
        exit_px, reason = None, "eod"
        while j < n:
            if l[j] <= sl:                                  # broker stop hit
                exit_px = sl - SLIP_FRAC * sl
                reason = "trail" if sl > entry else "sl"
                break
            hh = max(hh, c[j])
            sl = max(sl, hh - trm * atr[j])                 # up-only chandelier
            if c[j] < sma[j]:                               # regime exit
                if j + 1 < n:
                    exit_px = o[j + 1] - SLIP_FRAC * o[j + 1]
                    j += 1
                else:
                    exit_px = c[j]
                reason = "sma200"
                break
            j += 1
        if exit_px is None:
            j = n - 1
            exit_px = c[j]
        days = max(1, int((t[j] - t[i + 1]) / np.timedelta64(1, "D")))
        swap = 0.5 * (entry + exit_px) * LOTS * (swap_pct / 100) / 360 * days
        gross = (exit_px - entry) * dpp_min
        trades.append({"t_in": t[i + 1], "t": t[j], "days": days,
                       "gross": gross, "swap": swap, "pnl": gross + swap,
                       "reason": reason, "sym": sym})
        i = j + 1
    return pd.DataFrame(trades)


def stats(tr, label):
    if tr is None or len(tr) < 3:
        print(f"  {label:<30} n={0 if tr is None else len(tr)} (too few)")
        return None
    pnl = tr["pnl"]
    w, lo = pnl[pnl > 0], pnl[pnl <= 0]
    pf = w.sum() / abs(lo.sum()) if lo.sum() < 0 else 999
    wr = (pnl > 0).mean()
    yrs = max((tr["t"].iloc[-1] - tr["t"].iloc[0]).days / 365.25, 0.25)
    ret = pnl / EQUITY
    sh = ret.mean() / ret.std() * np.sqrt(len(tr) / yrs) if ret.std() > 0 else 0
    print(f"  {label:<30} n={len(tr):<3} WR={wr*100:3.0f}%  PF={pf:5.2f}  "
          f"tot=${pnl.sum():+8.2f}  avg=${pnl.mean():+6.2f}  "
          f"swap=${tr['swap'].sum():+7.2f}  {len(tr)/yrs:4.1f}/yr  Sharpe~{sh:.2f}")
    return dict(n=len(tr), wr=wr, pf=pf, tot=pnl.sum(), sharpe=sh)


def main():
    print("=== CRYPTO TSMOM-LF (SMA200/ROC90/Don20, SL 3.0xATR, trail 3.5xATR) ===")
    print(f"    costs: spread + 0.05%/side slip + swap(-25%..-38%/yr on price); "
          f"lots={LOTS} fixed; BTC gate risk<=3% of ${EQUITY:.0f}\n")
    dfs = {s: load(s) for s in SPEC}
    cut = pd.Timestamp("2023-01-01")
    allt = []
    for sym in SPEC:
        tr = backtest(dfs[sym], sym, SHIP)
        allt.append(tr)
        stats(tr, f"{sym} FULL")
        stats(tr[tr["t"] < cut], f"{sym} IS (<2023)")
        stats(tr[tr["t"] >= cut], f"{sym} OOS (>=2023)")
        if len(tr):
            rc = tr["reason"].value_counts().to_dict()
            print(f"    exits: {rc}   avg hold {tr['days'].mean():.0f}d\n")
    basket = pd.concat(allt).sort_values("t").reset_index(drop=True)
    stats(basket, "BASKET FULL")
    stats(basket[basket["t"] < cut], "BASKET IS (<2023)")
    ob = stats(basket[basket["t"] >= cut], "BASKET OOS (>=2023)")

    print("\n=== yearly walk-forward (basket, by exit year) ===")
    for y, g in basket.groupby(basket["t"].dt.year):
        pnl = g["pnl"]
        pf = (pnl[pnl > 0].sum() / abs(pnl[pnl <= 0].sum())
              if (pnl <= 0).any() and pnl[pnl <= 0].sum() < 0 else 999)
        print(f"  {y}: n={len(g):<3} tot=${pnl.sum():+8.2f}  PF={pf:5.2f}  "
              f"WR={(pnl>0).mean()*100:3.0f}%")

    print("\n=== plateau sweep: SMA x ROC x DON x TRAIL (basket total PnL / PF) ===")
    smas, rocs, dons, trs = [150, 175, 200, 225, 250], [60, 90, 120], \
        [15, 20, 25, 30], [2.5, 3.0, 3.5, 4.0, 4.5]
    res = []
    for s_ in smas:
        for r_ in rocs:
            for d_ in dons:
                for m_ in trs:
                    p = dict(SHIP, SMA_PERIOD=s_, ROC_PERIOD=r_,
                             DON_PERIOD=d_, TRAIL_ATR_MULT=m_)
                    b = pd.concat([backtest(dfs[s], s, p) for s in SPEC])
                    pnl = b["pnl"]
                    pf = (pnl[pnl > 0].sum() / abs(pnl[pnl <= 0].sum())
                          if (pnl <= 0).any() and pnl[pnl <= 0].sum() < 0 else 999)
                    oos = pnl[b["t"] >= cut]
                    res.append((s_, r_, d_, m_, pnl.sum(), pf, len(b), oos.sum()))
    R = pd.DataFrame(res, columns=["sma", "roc", "don", "trail",
                                   "tot", "pf", "n", "oos_tot"])
    pos = (R["tot"] > 0).mean()
    pos_oos = (R["oos_tot"] > 0).mean()
    print(f"  {len(R)} cells | full-sample profitable: {pos*100:.0f}% | "
          f"OOS profitable: {pos_oos*100:.0f}%")
    print(f"  tot  min/med/max: ${R['tot'].min():+.0f} / ${R['tot'].median():+.0f}"
          f" / ${R['tot'].max():+.0f} | PF med {R['pf'].median():.2f}")
    ship = R[(R.sma == 200) & (R.roc == 90) & (R.don == 20) & (R.trail == 3.5)]
    print(f"  SHIPPED cell: tot=${ship['tot'].iloc[0]:+.2f} PF={ship['pf'].iloc[0]:.2f} "
          f"n={ship['n'].iloc[0]} oos=${ship['oos_tot'].iloc[0]:+.2f}")
    nb = R[(R.sma.isin([175, 200, 225])) & (R.roc == 90)
           & (R.don.isin([15, 20, 25])) & (R.trail.isin([3.0, 3.5, 4.0]))]
    print(f"  neighbours (27 cells): profitable {(nb['tot']>0).mean()*100:.0f}%, "
          f"PF min/med {nb['pf'].min():.2f}/{nb['pf'].median():.2f}, "
          f"OOS>0 {(nb['oos_tot']>0).mean()*100:.0f}%")

    # detector parity: replay last 700 bars bar-by-bar through agent module
    print("\n=== detector parity (agent/crypto_tsmom.py vs vectorized, ETHUSD) ===")
    df = dfs["ETHUSD"]
    c = df["close"]
    sma = c.rolling(200).mean()
    don = c.rolling(20).max()
    mism = 0
    for k in range(len(df) - 700, len(df)):
        sig = detector_evaluate(df.iloc[: k + 1], SHIP)
        vec = bool(c[k] > sma[k] and c[k] > c[k - 90] and c[k] >= don[k])
        if sig is None or sig["entry"] != vec:
            mism += 1
    print(f"  700 bars replayed, mismatches: {mism}")
    if ob:
        print(f"\nGATE (OOS basket): Sharpe {ob['sharpe']:.2f} (>0.5?), "
              f"WR {ob['wr']*100:.0f}% (>55?), PF {ob['pf']:.2f} (1.0-2.5?)")


if __name__ == "__main__":
    main()
