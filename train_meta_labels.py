"""
Dragon Trader — Train Meta-Label ML Models (Offline from Cache).
Loads H1 candles from /Users/ashish/Documents/xauusd-trading-bot/cache/
and trains LightGBM meta-label models for all 6 Dragon symbols.
"""
import pickle
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

CACHE_DIR = "/Users/ashish/Documents/xauusd-trading-bot/cache"
SYMBOL_CACHE = {
    "XAUUSD":   f"{CACHE_DIR}/raw_h1_xauusd.pkl",
    "XAGUSD":   f"{CACHE_DIR}/raw_h1_XAGUSD.pkl",
    "BTCUSD":   f"{CACHE_DIR}/raw_h1_BTCUSD.pkl",
    "NAS100.r": f"{CACHE_DIR}/raw_h1_NAS100_r.pkl",
    "JPN225ft": f"{CACHE_DIR}/raw_h1_JPN225ft.pkl",
    "USDJPY":   f"{CACHE_DIR}/raw_h1_USDJPY.pkl",
    "USDCHF":   f"{CACHE_DIR}/raw_h1_USDCHF.pkl",
    "USDCAD":   f"{CACHE_DIR}/raw_h1_USDCAD.pkl",
    "EURJPY":   f"{CACHE_DIR}/raw_h1_EURJPY.pkl",
}


class MockMT5:
    """Mock MT5 connection that loads cached H1 pickle files."""

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        path = SYMBOL_CACHE.get(symbol)
        if not path:
            print(f"No cache for {symbol}")
            return None
        with open(path, "rb") as f:
            df = pickle.load(f)

        # Convert to structured numpy array like MT5 returns
        result = np.zeros(
            len(df),
            dtype=[
                ("time", "i8"), ("open", "f8"), ("high", "f8"),
                ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
                ("spread", "i4"), ("real_volume", "i8"),
            ],
        )
        if hasattr(df["time"].iloc[0], "timestamp"):
            result["time"] = df["time"].apply(lambda x: int(x.timestamp())).values
        else:
            result["time"] = df["time"].values.astype("int64") // 10**9
        result["open"] = df["open"].values
        result["high"] = df["high"].values
        result["low"] = df["low"].values
        result["close"] = df["close"].values
        result["tick_volume"] = df["tick_volume"].values
        result["spread"] = df["spread"].values
        result["real_volume"] = df["real_volume"].values
        return result


def main():
    from models.signal_model import SignalModel

    mock_mt5 = MockMT5()
    model = SignalModel()

    print("=" * 70)
    print("DRAGON TRADER — META-LABEL ML TRAINING")
    print("=" * 70)

    all_metrics = {}
    for symbol in list(SYMBOL_CACHE.keys()):
        print(f"\n{'=' * 60}")
        print(f"Training {symbol}...")
        print(f"{'=' * 60}")
        try:
            metrics = model.train(symbol, mock_mt5, None)
            all_metrics[symbol] = metrics
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_metrics[symbol] = {"status": "error", "reason": str(e)}

    print(f"\n{'=' * 70}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 70}")
    for sym, m in all_metrics.items():
        if m and m.get("status") == "ok":
            print(
                f"  {sym:12s}  AUC={m['test_auc']:.3f}  Acc={m['test_accuracy']:.3f}  "
                f"Prec@conf={m['precision_at_conf']:.3f}  "
                f"FilteredPF={m['filtered_pf']:.2f}  "
                f"WinRate={m['base_win_rate']*100:.1f}%  "
                f"Signals={m['n_signals']}  Trees={m['n_trees']}"
            )
        else:
            reason = m.get("reason", "unknown") if m else "no result"
            print(f"  {sym:12s}  FAILED: {reason}")

    print(f"\nModels saved to: /Users/ashish/Documents/beast-trader/models/saved/")
    print("Done.")


if __name__ == "__main__":
    main()
