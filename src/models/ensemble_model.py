import torch
import torch.nn as nn
import numpy as np
from typing import List

class EnsembleDetector:
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        predictions = []
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                predictions.append(pred.cpu().numpy() * weight)
        return np.sum(predictions, axis=0)

    def get_anomaly_scores(self, x: torch.Tensor) -> dict:
        scores = {}
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'get_anomaly_score'):
                    score = model.get_anomaly_score(x)
                else:
                    score = model(x)
                scores[f"model_{i}"] = score.cpu().numpy()
        return scores

    def detect_anomalies(self, x: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        predictions = self.predict(x)
        return (predictions > threshold).astype(int)
