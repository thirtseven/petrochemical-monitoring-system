import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional

class Visualizer:
    def __init__(self, style: str = "seaborn-v0_8-whitegrid", figsize: tuple = (12, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

    def plot_training_history(self, train_losses: list, val_losses: list, save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(train_losses, label='Training Loss', color=self.colors[0], linewidth=2)
        ax.plot(val_losses, label='Validation Loss', color=self.colors[1], linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_anomaly_scores(self, scores: np.ndarray, labels: np.ndarray, threshold: float, save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        normal_idx = labels == 0
        anomaly_idx = labels == 1

        ax.scatter(np.where(normal_idx)[0], scores[normal_idx], c=self.colors[0], label='Normal', alpha=0.6, s=20)
        ax.scatter(np.where(anomaly_idx)[0], scores[anomaly_idx], c=self.colors[1], label='Anomaly', alpha=0.6, s=20)
        ax.axhline(y=threshold, color=self.colors[2], linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Anomaly Score', fontsize=12)
        ax.set_title('Anomaly Detection Results', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, save_path: Optional[str] = None):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(fpr, tpr, color=self.colors[2], linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color=self.colors[4], linestyle='--', linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, feature_names: list, importance_scores: np.ndarray, top_n: int = 20, save_path: Optional[str] = None):
        indices = np.argsort(importance_scores)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(top_n), importance_scores[indices], color=self.colors[2])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_time_series(self, data: np.ndarray, anomaly_mask: np.ndarray, feature_idx: int = 0, save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        time_steps = np.arange(len(data))

        ax.plot(time_steps, data[:, feature_idx], color=self.colors[0], linewidth=1, alpha=0.7, label='Normal')

        anomaly_indices = np.where(anomaly_mask)[0]
        if len(anomaly_indices) > 0:
            ax.scatter(anomaly_indices, data[anomaly_indices, feature_idx], 
                      c=self.colors[1], s=30, label='Anomaly', zorder=5)

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel(f'Feature {feature_idx}', fontsize=12)
        ax.set_title('Time Series with Anomalies', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_comparison(self, model_names: list, metrics: dict, save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))

        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx] if len(metrics) > 1 else axes
            bars = ax.bar(model_names, values, color=self.colors[:len(model_names)])
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'Model Comparison - {metric_name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
