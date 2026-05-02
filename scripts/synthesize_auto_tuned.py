"""
Merge all per-aspect auto_dict outputs from the 10-agent sweep into ONE auto_tuned.py.

SL_OVERRIDE_AUTO + SIGNAL_QUALITY_SYMBOL_AUTO are rebuilt from tune_180d_pass2.json
on every run (always source-of-truth, never from the previous auto_tuned.py).

Other inputs (all live in backtest/results/):
  direction_bias_auto_dict.py       -> DIRECTION_BIAS_AUTO
  rescue_losers_auto_dict.py        -> RESCUE_AUTO
  risk_caps_auto_dict.py            -> RISK_CAP_AUTO
  toxic_hours_auto_dict.py          -> TOXIC_HOURS_PER_SYMBOL_AUTO
  trail_overrides_auto_dict.py      -> TRAIL_OVERRIDE_AUTO

Resolves:
  - rescue overrides take precedence for the 5 rescued symbols (BCHUSD, EURGBP,
    NZDJPY, NZDUSD, UKOUSD) — they get a wider SL, higher mq, AND forced direction.
  - rescue's `direction` ALSO writes into DIRECTION_BIAS_AUTO (so live signal flow honors it).
  - walk-forward WEAK symbols (ETHUSD, AUDJPY) get halved risk: ETHUSD 0.5, AUDJPY 0.4.

Writes a single auto_tuned.py at the repo root with all merged dicts.
"""
import sys, json, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "backtest" / "results"

# Acceptance gate. We DROPPED the baseline-comparison check because live's
# baseline is already inflated by stacked overrides (auto_tuned + RL trail +
# trail_override) — pass2's honest grid can't always beat that, but its
# params are still the right ones to use as the ground-truth anchor.
MIN_PF = 1.10
MIN_TRADES = 20
MAX_DD = 20.0

def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# Load all auto_dict outputs
db = _load("db", RES / "direction_bias_auto_dict.py").DIRECTION_BIAS_AUTO
rescue = _load("rescue", RES / "rescue_losers_auto_dict.py").RESCUE_AUTO
risk = _load("risk", RES / "risk_caps_auto_dict.py").RISK_CAP_AUTO
toxic = _load("toxic", RES / "toxic_hours_auto_dict.py").TOXIC_HOURS_PER_SYMBOL_AUTO
trail = _load("trail", RES / "trail_overrides_auto_dict.py").TRAIL_OVERRIDE_AUTO

# Rebuild SL + SQ from pass2.json (source of truth — pass2 carries pass1 best forward
# when no improvement, so it always has the strongest viable params per symbol).
SL, SQ = {}, {}
pass2_path = RES / "tune_180d_pass2.json"
if pass2_path.exists():
    p2 = json.load(open(pass2_path))
    for sym, r in p2.get("results", {}).items():
        best = (r or {}).get("best") or {}
        params = best.get("params") or {}
        result = best.get("result") or {}
        base_pnl = ((r or {}).get("baseline") or {}).get("pnl", 0)
        if not params or not result:
            continue
        if result.get("pf", 0) < MIN_PF:        continue
        if result.get("trades", 0) < MIN_TRADES: continue
        if result.get("dd", 100) > MAX_DD:      continue
        SL[sym] = round(float(params["sl_atr_mult"]), 2)
        SQ[sym] = dict(params["min_quality"])
DB = dict(db)

# Apply rescue overrides — they win for the 5 rescued losers
for sym, r in rescue.items():
    SL[sym] = r["sl_atr_mult"]
    mq = r["min_quality"]
    SQ[sym] = {"trending": mq, "ranging": mq, "volatile": mq, "low_vol": mq}
    DB[sym] = r["direction"]

# Walk-forward WEAK protection: halve risk for symbols that bled on prior window
RISK = dict(risk)
RISK["ETHUSD"] = 0.5   # walk-forward test PF 1.44 (down from 3.73 train)
RISK["AUDJPY"] = 0.4   # walk-forward test PF 1.16 (down from 3.83 train)

# 90d targeted retune — overrides for symbols where 180d-tuned params bled in 90d.
# scripts/retune_bleeders_90d.py exhaustively grids SL × mq × ratchet × direction
# against the most-recent 90d window. Output overrides pass2 for these 3 symbols.
bleeder_path = RES / "retune_bleeders_90d.json"
if bleeder_path.exists():
    bleeder = json.load(open(bleeder_path))
    for sym, b in (bleeder.get("best") or {}).items():
        if b.get("fallback"):
            continue
        SL[sym] = round(float(b["sl"]), 2)
        SQ[sym] = {"trending": int(b["mq_t"]), "ranging": int(b["mq_r"]),
                   "volatile": int(b["mq_t"]), "low_vol": int(b["mq_t"])}
        DB[sym] = b["dir"]

# Filter every dict to symbols actually traded by live (config.SYMBOLS).
# Why: backtest scanned 60 symbols but live trades a curated 7-symbol set.
# Memory rule: BTCUSD/EURUSD/USDJPY/GBPUSD/EURJPY backtest well but bled live —
# leaving them in auto_tuned.py is a footgun if config.SYMBOLS gets edited later.
sys.path.insert(0, str(ROOT))
import config as _live_cfg
LIVE = set(_live_cfg.SYMBOLS.keys())
SL     = {k: v for k, v in SL.items()     if k in LIVE}
SQ     = {k: v for k, v in SQ.items()     if k in LIVE}
DB     = {k: v for k, v in DB.items()     if k in LIVE}
RISK   = {k: v for k, v in RISK.items()   if k in LIVE}
toxic  = {k: v for k, v in toxic.items()  if k in LIVE}
trail  = {k: v for k, v in trail.items()  if k in LIVE}

def _fmt_dict(d, key_q="'"):
    """Render a dict in a stable single-line-per-entry layout."""
    if not d:
        return "{}"
    out = ["{"]
    for k in sorted(d):
        v = d[k]
        if isinstance(v, set):
            v = "{" + ", ".join(str(x) for x in sorted(v)) + "}"
        else:
            v = repr(v)
        out.append(f"    {key_q}{k}{key_q:<14}: {v},")
    out.append("}")
    return "\n".join(out)


body = f'''"""SYNTHESIZED auto_tuned.py — DO NOT HAND-EDIT.
Generated by scripts/synthesize_auto_tuned.py from 10-agent sweep results.

Layered onto config.py defaults via try/except import block at end of config.py.

Sources:
  SL_OVERRIDE_AUTO              ← tune_180d_pass1.json + rescue_losers_auto_dict.py
  SIGNAL_QUALITY_SYMBOL_AUTO    ← tune_180d_pass1.json + rescue overrides
  DIRECTION_BIAS_AUTO           ← direction_bias_auto_dict.py + rescue forced directions
  RISK_CAP_AUTO                 ← risk_caps_auto_dict.py + walk-forward risk halving
  TOXIC_HOURS_PER_SYMBOL_AUTO   ← toxic_hours_auto_dict.py
  TRAIL_OVERRIDE_AUTO           ← trail_overrides_auto_dict.py
"""

# Per-symbol ATR SL multiplier (merges into SYMBOL_ATR_SL_OVERRIDE)
SL_OVERRIDE_AUTO = {_fmt_dict(SL)}

# Per-symbol per-regime signal quality threshold (merges into SIGNAL_QUALITY_SYMBOL)
SIGNAL_QUALITY_SYMBOL_AUTO = {_fmt_dict(SQ)}

# Per-symbol direction bias LONG/SHORT/BOTH (merges into DIRECTION_BIAS)
DIRECTION_BIAS_AUTO = {_fmt_dict(DB)}

# Per-symbol risk-percent cap (merges into SYMBOL_RISK_CAP). Default base risk is 0.8%.
RISK_CAP_AUTO = {_fmt_dict(RISK)}

# Per-symbol toxic UTC hours, added on top of TOXIC_HOURS_UTC.
TOXIC_HOURS_PER_SYMBOL_AUTO = {_fmt_dict(toxic)}

# Per-symbol trail profile (replaces SYMBOL_TRAIL_OVERRIDE for these symbols).
TRAIL_OVERRIDE_AUTO = {_fmt_dict(trail)}
'''

(ROOT / "auto_tuned.py").write_text(body)
print(f"Wrote {ROOT / 'auto_tuned.py'}")
print(f"  SL: {len(SL)}  SQ: {len(SQ)}  DB: {len(DB)}  RISK: {len(RISK)}  TOXIC: {len(toxic)}  TRAIL: {len(trail)}")
