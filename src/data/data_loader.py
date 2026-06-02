import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os

class PetrochemicalDataLoader:
    def __init__(self, data_path: str = "data/raw/"):
        self.data_path = data_path
        self.fault_names = {
            0: "Normal",
            1: "A/C feed ratio, B composition constant",
            2: "B composition, A/C ratio constant",
            3: "D feed temperature",
            4: "Reactor cooling water inlet temperature",
            5: "Condenser cooling water inlet temperature",
            6: "A feed loss",
            7: "C header pressure loss",
            8: "A, B, C feed composition",
            9: "D feed temperature",
            10: "C feed temperature",
            11: "Reactor cooling water inlet temperature",
            12: "Condenser cooling water inlet temperature",
            13: "Reaction kinetics",
            14: "Reactor cooling water valve",
            15: "Condenser cooling water valve",
            16: "Unknown",
            17: "Unknown",
            18: "Unknown",
            19: "Unknown",
            20: "Unknown"
        }

    def load_tennessee_eastman(self, fault_id: int = 0, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        if fault_id == 0:
            filename = f"TEP_FaultFree_Training.csv" if split == "train" else f"TEP_FaultFree_Testing.csv"
        else:
            filename = f"TEP_Faulty_Training.csv" if split == "train" else f"TEP_Faulty_Testing.csv"

        filepath = os.path.join(self.data_path, filename)

        if not os.path.exists(filepath):
            print(f"Dataset not found at {filepath}. Generating synthetic data...")
            return self._generate_synthetic_data(fault_id, split)

        df = pd.read_csv(filepath)

        if fault_id > 0:
            df = df[df["faultNumber"] == fault_id]

        feature_cols = [col for col in df.columns if col.startswith("xmeas") or col.startswith("xmv")]
        X = df[feature_cols].values
        y = df["faultNumber"].values if "faultNumber" in df.columns else np.zeros(len(df))

        return X, y

    def _generate_synthetic_data(self, fault_id: int, split: str, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42 if split == "train" else 24)
        n_features = 52
        X = np.random.randn(n_samples, n_features)

        if fault_id > 0:
            fault_start = n_samples // 2
            X[fault_start:, fault_id % n_features] += np.sin(np.linspace(0, 10, n_samples - fault_start)) * 2
            X[fault_start:, (fault_id + 1) % n_features] += np.random.randn(n_samples - fault_start) * 0.5
            y = np.concatenate([np.zeros(fault_start), np.ones(n_samples - fault_start)])
        else:
            y = np.zeros(n_samples)

        return X, y

    def load_all_faults(self, split: str = "train") -> dict:
        data = {}
        data[0] = self.load_tennessee_eastman(fault_id=0, split=split)
        for fault_id in range(1, 21):
            try:
                data[fault_id] = self.load_tennessee_eastman(fault_id=fault_id, split=split)
            except Exception as e:
                print(f"Skipping fault {fault_id}: {e}")
        return data

    def get_feature_names(self) -> list:
        measured = [f"xmeas_{i}" for i in range(1, 42)]
        manipulated = [f"xmv_{i}" for i in range(1, 12)]
        return measured + manipulated
