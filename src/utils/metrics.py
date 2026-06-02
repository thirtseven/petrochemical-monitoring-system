import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            accuracy_score, roc_auc_score, confusion_matrix)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_scores is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['auc_roc'] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return metrics

def calculate_anomaly_threshold(scores: np.ndarray, percentile: float = 95) -> float:
    return np.percentile(scores, percentile)

def calculate_detection_delay(true_anomalies: np.ndarray, detected_anomalies: np.ndarray) -> float:
    delays = []
    in_anomaly = False
    start_idx = 0

    for i in range(len(true_anomalies)):
        if true_anomalies[i] == 1 and not in_anomaly:
            in_anomaly = True
            start_idx = i
        elif true_anomalies[i] == 0 and in_anomaly:
            in_anomaly = False
            detected_in_range = detected_anomalies[start_idx:i]
            if np.any(detected_in_range):
                first_detected = np.argmax(detected_in_range)
                delays.append(first_detected)

    return np.mean(delays) if delays else float('inf')

def calculate_alarm_rate(y_pred: np.ndarray, window_size: int = 100) -> np.ndarray:
    alarm_rate = np.zeros(len(y_pred))
    for i in range(window_size, len(y_pred)):
        alarm_rate[i] = np.mean(y_pred[i-window_size:i])
    return alarm_rate
