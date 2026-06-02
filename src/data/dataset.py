import torch
from torch.utils.data import Dataset
import numpy as np

class PetrochemicalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int = 50):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length + 1

    def __getitem__(self, idx):
        sequence = self.X[idx:idx + self.seq_length]
        label = self.y[idx + self.seq_length - 1]
        return sequence, label

class AnomalyDetectionDataset(Dataset):
    def __init__(self, normal_data: np.ndarray, anomaly_data: np.ndarray = None):
        self.normal_data = torch.FloatTensor(normal_data)
        self.anomaly_data = torch.FloatTensor(anomaly_data) if anomaly_data is not None else None
        self.has_anomaly = anomaly_data is not None

    def __len__(self):
        if self.has_anomaly:
            return len(self.normal_data) + len(self.anomaly_data)
        return len(self.normal_data)

    def __getitem__(self, idx):
        if idx < len(self.normal_data):
            return self.normal_data[idx], 0
        else:
            anomaly_idx = idx - len(self.normal_data)
            return self.anomaly_data[anomaly_idx], 1
