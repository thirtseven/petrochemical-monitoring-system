import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional

class DataPreprocessor:
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")
        return self.scaler.transform(X)

    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        labels = []
        for i in range(len(X) - seq_length + 1):
            sequences.append(X[i:i + seq_length])
            labels.append(y[i + seq_length - 1])
        return np.array(sequences), np.array(labels)

    def create_sliding_windows(self, X: np.ndarray, window_size: int = 50, step: int = 1) -> np.ndarray:
        windows = []
        for i in range(0, len(X) - window_size + 1, step):
            windows.append(X[i:i + window_size])
        return np.array(windows)

    def add_noise(self, X: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        noise = np.random.randn(*X.shape) * noise_level
        return X + noise

    def handle_missing_values(self, X: np.ndarray, strategy: str = "mean") -> np.ndarray:
        if strategy == "mean":
            col_means = np.nanmean(X, axis=0)
            indices = np.where(np.isnan(X))
            X[indices] = np.take(col_means, indices[1])
        elif strategy == "forward":
            df = pd.DataFrame(X)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            X = df.values
        return X

    def split_data(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def compute_statistics(self, X: np.ndarray) -> dict:
        return {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "min": np.min(X, axis=0),
            "max": np.max(X, axis=0),
            "median": np.median(X, axis=0)
        }
