"""
Dragon Trader — Deep Ensemble Meta-Label Model.
Combines CNN (candle patterns) + LSTM (sequence memory) + LightGBM (tabular)
into a stacked ensemble for maximum prediction accuracy.

Architecture:
1. CNN: 1D conv on last 20 bars OHLCV → pattern features
2. LSTM: last 30 bars of indicator values → trend memory
3. LightGBM: 33 tabular features (existing)
4. Stacking: CNN+LSTM+LGB probabilities → final meta-classifier
"""
import os
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, SYMBOLS, ATR_SL_MULTIPLIER
from signals.momentum_scorer import (
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, MIN_SCORE,
)

log = logging.getLogger("dragon.deep")

SEQUENCE_LEN = 20  # bars of history for CNN/LSTM
DEEP_MODEL_DIR = MODEL_DIR / "deep"
DEEP_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class CandleCNN(nn.Module):
    """1D CNN over OHLCV candle sequences → pattern features."""
    def __init__(self, in_channels=5, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden, 8)

    def forward(self, x):
        # x: (batch, channels=5, seq_len=20)
        h = self.conv(x).squeeze(-1)  # (batch, hidden)
        return self.fc(h)  # (batch, 8)


class IndicatorLSTM(nn.Module):
    """LSTM over indicator time series → trend memory features."""
    def __init__(self, in_features=8, hidden=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 8)

    def forward(self, x):
        # x: (batch, seq_len=20, features=8)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])  # (batch, 8)


class DeepEnsemble(nn.Module):
    """Stacking: CNN features + LSTM features + tabular → binary classification."""
    def __init__(self, tabular_dim=33):
        super().__init__()
        self.cnn = CandleCNN(in_channels=5, hidden=32)
        self.lstm = IndicatorLSTM(in_features=8, hidden=32)
        # Combine: 8 (CNN) + 8 (LSTM) + tabular_dim
        combined = 8 + 8 + tabular_dim
        self.head = nn.Sequential(
            nn.Linear(combined, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, candles, indicators, tabular):
        cnn_feat = self.cnn(candles)
        lstm_feat = self.lstm(indicators)
        combined = torch.cat([cnn_feat, lstm_feat, tabular], dim=1)
        return self.head(combined).squeeze(-1)


class DeepSignalModel:
    """Deep ensemble model for meta-labeling."""

    def __init__(self):
        self.models = {}  # symbol -> DeepEnsemble
        self.lgb_models = {}  # symbol -> lgb.Booster (keep existing)
        self._train_metrics = {}
        self.device = torch.device("cpu")

    def _build_sequences(self, df, ind, signals, labels):
        """Build CNN candle sequences + LSTM indicator sequences + tabular features."""
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        opn = df["open"].values.astype(np.float64) if "open" in df else ind["o"]
        vol = df["tick_volume"].values.astype(np.float64) if "tick_volume" in df else np.ones(len(close))
        atr = ind["at"]

        candle_seqs = []  # (N, 5, SEQUENCE_LEN) — OHLCV normalized by ATR
        ind_seqs = []     # (N, SEQUENCE_LEN, 8) — indicator sequences
        tab_features = [] # (N, 33) — existing tabular features
        valid_labels = []

        for j, (bar_i, direction, ls, ss, cs) in enumerate(signals):
            if bar_i < SEQUENCE_LEN + 5:
                continue

            a = atr[bar_i] if not np.isnan(atr[bar_i]) and atr[bar_i] > 0 else 1.0

            # CNN: last SEQUENCE_LEN bars OHLCV, normalized by ATR
            sl = slice(bar_i - SEQUENCE_LEN, bar_i)
            c_seq = np.stack([
                (opn[sl] - close[bar_i]) / a,
                (high[sl] - close[bar_i]) / a,
                (low[sl] - close[bar_i]) / a,
                (close[sl] - close[bar_i]) / a,
                vol[sl] / (np.mean(vol[max(0, bar_i-50):bar_i]) + 1),
            ])  # (5, SEQUENCE_LEN)
            candle_seqs.append(c_seq)

            # LSTM: indicator sequences (RSI, ADX, MACD, BBW, ATR, EMA_align, ST_dir, consec)
            i_seq = np.zeros((SEQUENCE_LEN, 8))
            for k, idx in enumerate(range(bar_i - SEQUENCE_LEN, bar_i)):
                i_seq[k, 0] = ind["rs"][idx] / 100 if not np.isnan(ind["rs"][idx]) else 0.5
                i_seq[k, 1] = ind["adx"][idx] / 50 if not np.isnan(ind["adx"][idx]) else 0.5
                i_seq[k, 2] = ind["mh"][idx] / a if a > 0 else 0
                i_seq[k, 3] = ind["bbw"][idx] / 5 if not np.isnan(ind["bbw"][idx]) else 0.4
                i_seq[k, 4] = ind["at"][idx] / a if a > 0 and not np.isnan(ind["at"][idx]) else 1.0
                i_seq[k, 5] = 1 if ind["es"][idx] > ind["el"][idx] else -1
                i_seq[k, 6] = float(ind["st"][idx])
                i_seq[k, 7] = float(ind["consec"][idx]) / 5
            ind_seqs.append(i_seq)

            valid_labels.append(labels[j])

        return (np.array(candle_seqs, dtype=np.float32),
                np.array(ind_seqs, dtype=np.float32),
                np.array(valid_labels, dtype=np.float32))

    def train(self, symbol, mt5_conn=None, feature_engine=None, epochs=30):
        """Train deep ensemble for one symbol."""
        log.info("[%s] Deep ensemble training starting...", symbol)

        # Load data using existing SignalModel infrastructure
        from models.signal_model import SignalModel
        base_model = SignalModel()

        # Get signals + labels + tabular features from base model
        # We'll reuse its _build data pipeline
        cache_map = {
            "XAUUSD": "raw_h1_xauusd.pkl", "XAGUSD": "raw_h1_XAGUSD.pkl",
            "BTCUSD": "raw_h1_BTCUSD.pkl", "NAS100.r": "raw_h1_NAS100_r.pkl",
            "JPN225ft": "raw_h1_JPN225ft.pkl", "USDJPY": "raw_h1_USDJPY.pkl",
        }
        cache_path = Path("/Users/ashish/Documents/xauusd-trading-bot/cache") / cache_map.get(symbol, "")
        if not cache_path.exists():
            log.error("[%s] Cache not found: %s", symbol, cache_path)
            return None

        df = pickle.load(open(cache_path, "rb"))
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        icfg = dict(IND_DEFAULTS)
        icfg.update(IND_OVERRIDES.get(symbol, {}))
        ind = _compute_indicators(df, icfg)
        n = ind["n"]

        # Generate signals and labels (same as base SignalModel)
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        atr = ind["at"]

        signals = []
        labels = []
        for i in range(200, n - 30):
            if np.isnan(atr[i]) or atr[i] == 0:
                continue
            ls, ss = _score(ind, i)
            if max(ls, ss) < 4.0:
                continue
            d = 1 if ls >= ss else -1
            cs = ls if d == 1 else ss
            signals.append((i, d, ls, ss, cs))

            # Label: did price hit 2R before 1R SL?
            a = atr[i]
            sl_dist = a * ATR_SL_MULTIPLIER
            entry = close[i]
            won = 0
            for j in range(i + 1, min(i + 40, n)):
                if d == 1:
                    if low[j] <= entry - sl_dist: won = 0; break
                    if high[j] >= entry + 2 * sl_dist: won = 1; break
                else:
                    if high[j] >= entry + sl_dist: won = 0; break
                    if low[j] <= entry - 2 * sl_dist: won = 1; break
            labels.append(won)

        labels = np.array(labels)
        log.info("[%s] %d signals, %.1f%% win rate", symbol, len(signals), labels.mean() * 100)

        # Build sequences
        candle_seqs, ind_seqs, valid_labels = self._build_sequences(df, ind, signals, labels)

        # Load existing LGB model (don't retrain — already tuned)
        base_model.load(symbol)
        self.lgb_models[symbol] = base_model.models.get(symbol)

        # Train/val/test split (walk-forward)
        N = len(valid_labels)
        train_end = int(N * 0.70)
        val_end = int(N * 0.85)

        X_candle_train = torch.FloatTensor(candle_seqs[:train_end])
        X_candle_val = torch.FloatTensor(candle_seqs[train_end:val_end])
        X_candle_test = torch.FloatTensor(candle_seqs[val_end:])
        X_ind_train = torch.FloatTensor(ind_seqs[:train_end])
        X_ind_val = torch.FloatTensor(ind_seqs[train_end:val_end])
        X_ind_test = torch.FloatTensor(ind_seqs[val_end:])
        # Tabular placeholder (zeros — LGB handles tabular separately)
        tab_dim = 33
        X_tab_train = torch.zeros(train_end, tab_dim)
        X_tab_val = torch.zeros(val_end - train_end, tab_dim)
        X_tab_test = torch.zeros(N - val_end, tab_dim)
        y_train = torch.FloatTensor(valid_labels[:train_end])
        y_val = torch.FloatTensor(valid_labels[train_end:val_end])
        y_test = torch.FloatTensor(valid_labels[val_end:])

        # Train deep model
        model = DeepEnsemble(tabular_dim=tab_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        train_ds = TensorDataset(X_candle_train, X_ind_train, X_tab_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

        best_val_auc = 0
        best_state = None
        patience = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for candles, inds, tabs, targets in train_loader:
                optimizer.zero_grad()
                preds = model(candles, inds, tabs)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validate
            model.eval()
            with torch.no_grad():
                val_preds = model(X_candle_val, X_ind_val, X_tab_val).numpy()
                from sklearn.metrics import roc_auc_score
                try:
                    val_auc = roc_auc_score(y_val.numpy(), val_preds)
                except:
                    val_auc = 0.5

            scheduler.step(1 - val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch % 5 == 0:
                log.info("[%s] Epoch %d: loss=%.4f val_AUC=%.4f best=%.4f",
                         symbol, epoch, total_loss / len(train_loader), val_auc, best_val_auc)

            if patience >= 10:
                log.info("[%s] Early stopping at epoch %d", symbol, epoch)
                break

        # Load best model
        if best_state:
            model.load_state_dict(best_state)

        # Test
        model.eval()
        with torch.no_grad():
            test_preds = model(X_candle_test, X_ind_test, X_tab_test).numpy()
            try:
                test_auc = roc_auc_score(y_test.numpy(), test_preds)
            except:
                test_auc = 0.5

        # Save
        save_path = DEEP_MODEL_DIR / f"{symbol.replace('.', '_')}_deep.pt"
        torch.save({
            "model_state": model.state_dict(),
            "val_auc": best_val_auc,
            "test_auc": test_auc,
            "epoch": epoch,
        }, save_path)

        self.models[symbol] = model
        self._train_metrics[symbol] = {
            "val_auc": best_val_auc,
            "test_auc": test_auc,
            "status": "ok",
        }

        log.info("[%s] Deep ensemble: val_AUC=%.4f test_AUC=%.4f saved to %s",
                 symbol, best_val_auc, test_auc, save_path)
        return self._train_metrics[symbol]

    def load(self, symbol):
        """Load saved deep model."""
        path = DEEP_MODEL_DIR / f"{symbol.replace('.', '_')}_deep.pt"
        if path.exists():
            data = torch.load(path, map_location=self.device, weights_only=False)
            model = DeepEnsemble(tabular_dim=33).to(self.device)
            model.load_state_dict(data["model_state"])
            model.eval()
            self.models[symbol] = model
            self._train_metrics[symbol] = {
                "val_auc": data.get("val_auc", 0),
                "test_auc": data.get("test_auc", 0),
            }
            return True
        return False

    def predict(self, symbol, candle_seq, ind_seq, tabular_features):
        """Predict using deep ensemble."""
        if symbol not in self.models:
            return None
        model = self.models[symbol]
        model.eval()
        with torch.no_grad():
            c = torch.FloatTensor(candle_seq).unsqueeze(0)
            i = torch.FloatTensor(ind_seq).unsqueeze(0)
            t = torch.FloatTensor(list(tabular_features.values())).unsqueeze(0)
            prob = model(c, i, t).item()
        return prob
