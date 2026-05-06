#!/usr/bin/env python3 -B
"""
Filter the tuner output to keep only symbols whose 180d validation
shows a sane improvement. Drops:
  - Symbols where tuned_trades < 50% of base_trades (over-filtering)
  - Symbols where tuned_pf > 20 (lucky outliers / overfit)
  - Symbols where tuned_pnl < base_pnl (no improvement)

Writes a filtered version to component_weights_auto_dict.py — the same path
the runtime reads. Original tuner output is preserved at
component_weights_auto_dict_unfiltered.py for audit.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "backtest" / "results"
AUTO = RES / "component_weights_auto_dict.py"
AUTO_UNFILT = RES / "component_weights_auto_dict_unfiltered.py"
VAL = RES / "component_weights_validate.json"

MIN_TRADE_RATIO = 0.5    # tuned_trades must be ≥ 50% of base
MAX_PF_TUNED = 20.0      # PF > 20 = suspect overfit
MIN_PNL_LIFT = 5.0       # require >$5 improvement


def main() -> None:
    if not VAL.exists():
        print(f"missing {VAL} — run validate_component_weights.py first")
        sys.exit(1)

    # Backup original tuner output once
    if not AUTO_UNFILT.exists() and AUTO.exists():
        shutil.copy(AUTO, AUTO_UNFILT)
        print(f"backed up {AUTO.name} -> {AUTO_UNFILT.name}")

    # Load both
    spec = importlib.util.spec_from_file_location("cw", AUTO_UNFILT if AUTO_UNFILT.exists() else AUTO)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    auto = getattr(mod, "COMPONENT_WEIGHTS_AUTO", {})

    val = json.load(open(VAL))
    per_sym = val["results"]

    kept: dict = {}
    rejected: list = []
    for sym, weights in auto.items():
        b = per_sym.get(sym, {}).get("BASE")
        t = per_sym.get(sym, {}).get("TUNED")
        if not b or not t:
            rejected.append((sym, "no validation data"))
            continue
        b_pnl = float(b.get("pnl", 0)); t_pnl = float(t.get("pnl", 0))
        b_n = int(b.get("trades", 0)); t_n = int(t.get("trades", 0))
        t_pf = float(t.get("pf", 0))

        # Reject: lost too many signals
        if b_n > 0 and t_n / b_n < MIN_TRADE_RATIO:
            rejected.append((sym, f"trades collapsed {b_n}->{t_n}"))
            continue
        # Reject: PF spike — likely lucky outliers
        if t_pf > MAX_PF_TUNED:
            rejected.append((sym, f"PF {t_pf:.1f} > {MAX_PF_TUNED} (overfit suspect)"))
            continue
        # Reject: no PnL lift
        if t_pnl - b_pnl < MIN_PNL_LIFT:
            rejected.append((sym, f"no lift ${b_pnl:.0f}->${t_pnl:.0f}"))
            continue
        kept[sym] = weights

    # Write filtered auto_dict
    with open(AUTO, "w") as f:
        f.write('"""AUTO-GENERATED — filtered by scripts/filter_component_weights.py.\n')
        f.write(f'Acceptance: trades>=50% base, PF<{MAX_PF_TUNED}, PnL lift>${MIN_PNL_LIFT}.\n"""\n\n')
        f.write("COMPONENT_WEIGHTS_AUTO = {\n")
        for sym in sorted(kept):
            f.write(f"    {sym!r}: {{\n")
            for c, v in sorted(kept[sym].items()):
                f.write(f"        {c!r}: {v},\n")
            f.write("    },\n")
        f.write("}\n")

    print(f"\nKept {len(kept)}/{len(auto)} symbols")
    print(f"Wrote {AUTO}")
    print(f"\nRejected:")
    for sym, reason in rejected:
        print(f"  {sym:10s}  {reason}")
    print(f"\nKept symbols: {sorted(kept.keys())}")


if __name__ == "__main__":
    main()
