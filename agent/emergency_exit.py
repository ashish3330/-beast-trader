#!/usr/bin/env python3 -B
"""EMERGENCY EXIT GATE (2026-07-12) — statistical, per-symbol profit/loss exits
driven by the live journal's win-rate + avg-win-points + avg-loss-points.

Two problems this addresses:
  1. Losing trades ride to the full SL — the TREND peak-giveback protector only
     fires on trades that FIRST built a peak, so a trade that's red from entry is
     never cut early. The avg-loss cut here fires regardless of any peak.
  2. Winners give profit back. The avg-win logic banks the typical winner.

WIN-side modes (per-symbol avg_win_pts from the journal):
  - "hard"  : exit the moment profit >= avg_win_pts. LITERAL request, but on this
              bot's history it deletes 38-56% of total winning points (caps the
              right tail where trend profit lives) — see the deploy notes.
  - "trail" : ARM a giveback-trail once profit >= avg_win_pts, then exit only if
              profit retraces >= GIVEBACK from its peak. Lets tail winners run but
              stops them turning into givebacks. DEFAULT (tail-safe).

LOSS-side (avg_loss_pts from the journal):
  - exit when loss magnitude >= avg_loss_pts * LOSS_MULT (a statistical stop that
    fires before the wider broker SL — cuts the average loser early).

Pure functions; the brain owns cadence, position reads, peak tracking, and the
one-write-per-cycle bridge cap.
"""
import sqlite3


def compute_symbol_stats(db_path, lookback=80, min_samples=15):
    """Per-symbol {n, wr, avg_win_pts, avg_loss_pts} from the most recent
    `lookback` closed trades. Symbols with < min_samples or no winners are
    omitted (the gate skips anything without a valid stat)."""
    out = {}
    try:
        c = sqlite3.connect(db_path)
        rows = c.execute(
            "SELECT symbol, entry_price, exit_price, direction FROM trades "
            "WHERE exit_price IS NOT NULL AND entry_price IS NOT NULL "
            "ORDER BY id DESC").fetchall()
        c.close()
    except Exception:
        return out
    per = {}
    for sym, e, x, dr in rows:
        if not e or not x:
            continue
        if len(per.get(sym, ())) >= lookback:
            continue
        dirn = 1 if dr in (1, "1", "BUY", "LONG", "buy", "long") else -1
        per.setdefault(sym, []).append((float(x) - float(e)) * dirn)
    for sym, pts in per.items():
        if len(pts) < min_samples:
            continue
        wins = [p for p in pts if p > 0]
        loss = [-p for p in pts if p < 0]
        if not wins:                       # no winners → no valid avg-win → skip
            continue
        out[sym] = {
            "n": len(pts),
            "wr": len(wins) / len(pts),
            "avg_win_pts": sum(wins) / len(wins),
            "avg_loss_pts": (sum(loss) / len(loss)) if loss else 0.0,
        }
    return out


def decide(profit_pts, peak_pts, stat, cfg):
    """Return (should_exit, reason) for one open position/group.
      profit_pts : current open profit in PRICE POINTS (signed, + = winning)
      peak_pts   : running peak of profit_pts since entry (for trail mode)
      stat       : {avg_win_pts, avg_loss_pts, wr, n} for the symbol
      cfg        : {win_mode, win_mult, giveback, loss_cut, loss_mult}
    """
    if not stat:
        return False, None
    aw = float(stat.get("avg_win_pts", 0.0))
    al = float(stat.get("avg_loss_pts", 0.0))
    # ── WIN side ──
    if aw > 0:
        arm = aw * float(cfg.get("win_mult", 1.0))
        if cfg.get("win_mode", "trail") == "hard":
            if profit_pts >= arm:
                return True, f"avg_win_hard(+{profit_pts:.1f}>={arm:.1f}pts)"
        else:                              # trail: arm at avg-win, exit on giveback
            if peak_pts >= arm:
                gb = float(cfg.get("giveback", 0.35))
                if profit_pts <= peak_pts * (1.0 - gb):
                    return True, (f"avg_win_trail(peak{peak_pts:.1f} -> {profit_pts:.1f}, "
                                  f">={int(gb*100)}% giveback)")
    # ── LOSS side (cut the average loser early, before the broker SL) ──
    if cfg.get("loss_cut", True) and al > 0:
        cut = -al * float(cfg.get("loss_mult", 1.0))
        if profit_pts <= cut:
            return True, f"avg_loss_cut({profit_pts:.1f}<={cut:.1f}pts)"
    return False, None
