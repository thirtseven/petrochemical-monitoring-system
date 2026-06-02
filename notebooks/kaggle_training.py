"""
Petrochemical Monitoring System - Kaggle Training Script
=========================================================

This script is designed to run on Kaggle for training deep learning models
on the Tennessee Eastman Process dataset for anomaly detection.

Usage:
1. Upload this notebook to Kaggle
2. Add the Tennessee Eastman Process dataset to the notebook
3. Run all cells

Required datasets on Kaggle:
- Tennessee Eastman Process Simulation Dataset
"""

import sys
sys.path.insert(0, '/kaggle/working/petrochemical-monitoring-system')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Petrochemical Monitoring System - Deep Learning Training")
print("=" * 60)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

CONFIG = {
    'data': {
        'sequence_length': 50,
        'batch_size': 64,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15
    },
    'model': {
        'input_dim': 52,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2
    },
    'training': {
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'patience': 10
    },
    'anomaly': {
        'threshold_percentile': 95
    }
}

class LSTMDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(context).squeeze(-1)

class TransformerDetector(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = self._generate_pos_enc(d_model, 5000)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model//2, 1), nn.Sigmoid()
        )

    def _generate_pos_enc(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :x.size(1), :].to(x.device)
        x = self.transformer(x)
        return self.fc(x[:, -1, :]).squeeze(-1)

class AutoencoderDetector(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(input_dim*2, input_dim), nn.ReLU(),
            nn.Linear(input_dim, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim), nn.ReLU(),
            nn.Linear(input_dim, input_dim*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(input_dim*2, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_anomaly_score(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=-1)

def generate_synthetic_data(n_samples=10000, fault_id=0):
    np.random.seed(42)
    X = np.random.randn(n_samples, 52)
    
    if fault_id > 0:
        fault_start = n_samples // 2
        t = np.linspace(0, 10, n_samples - fault_start)
        X[fault_start:, fault_id % 52] += np.sin(t) * 2
        X[fault_start:, (fault_id + 1) % 52] += np.random.randn(n_samples - fault_start) * 0.5
        y = np.concatenate([np.zeros(fault_start), np.ones(n_samples - fault_start)])
    else:
        y = np.zeros(n_samples)
    
    return X, y

def create_sequences(X, y, seq_length):
    sequences, labels = [], []
    for i in range(len(X) - seq_length + 1):
        sequences.append(X[i:i+seq_length])
        labels.append(y[i+seq_length-1])
    return np.array(sequences), np.array(labels)

def train_model(model, train_loader, val_loader, config, model_name):
    print(f"\n{'='*40}")
    print(f"Training {model_name}")
    print(f"{'='*40}")
    
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['training']['epochs']}] "
                  f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'/kaggle/working/{model_name}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, model_name):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    threshold = np.percentile(all_preds[all_labels == 0], 95)
    pred_labels = (all_preds > threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(all_labels, pred_labels),
        'precision': precision_score(all_labels, pred_labels, zero_division=0),
        'recall': recall_score(all_labels, pred_labels, zero_division=0),
        'f1': f1_score(all_labels, pred_labels, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_preds)
    }
    
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics, all_preds, threshold

print("\n1. Generating synthetic Tennessee Eastman Process data...")
X_normal, y_normal = generate_synthetic_data(10000, fault_id=0)
X_fault, y_fault = generate_synthetic_data(10000, fault_id=1)

X = np.vstack([X_normal, X_fault])
y = np.concatenate([y_normal, y_fault])

print(f"   Data shape: {X.shape}")
print(f"   Normal samples: {np.sum(y == 0)}")
print(f"   Anomaly samples: {np.sum(y == 1)}")

print("\n2. Creating sequences...")
seq_length = CONFIG['data']['sequence_length']
X_seq, y_seq = create_sequences(X, y, seq_length)
print(f"   Sequence shape: {X_seq.shape}")

print("\n3. Splitting data...")
n = len(X_seq)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=CONFIG['data']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['data']['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['data']['batch_size'])

print("\n4. Training models...")
models = {
    'LSTM': LSTMDetector(CONFIG['model']['input_dim'], CONFIG['model']['hidden_dim'], 
                         CONFIG['model']['num_layers'], CONFIG['model']['dropout']),
    'Transformer': TransformerDetector(CONFIG['model']['input_dim'], 64, 4, 2, 0.1)
}

results = {}
for name, model in models.items():
    train_losses, val_losses = train_model(model, train_loader, val_loader, CONFIG, name)
    metrics, preds, threshold = evaluate_model(model, test_loader, name)
    results[name] = {'metrics': metrics, 'predictions': preds, 'threshold': threshold}

print("\n5. Generating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for name, result in results.items():
    axes[0, 0].plot(result.get('train_losses', []), label=f'{name} Train')
    axes[0, 1].plot(result.get('val_losses', []), label=f'{name} Val')

axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_title('Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

metrics_df = pd.DataFrame({name: result['metrics'] for name, result in results.items()})
metrics_df.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Model Comparison')
axes[1, 0].set_ylabel('Score')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

for name, result in results.items():
    axes[1, 1].scatter(range(len(result['predictions'][:200])), result['predictions'][:200], 
                      label=name, alpha=0.6, s=20)
axes[1, 1].axhline(y=results['LSTM']['threshold'], color='r', linestyle='--', label='Threshold')
axes[1, 1].set_title('Anomaly Scores (First 200 samples)')
axes[1, 1].set_xlabel('Sample')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print("\nResults Summary:")
print(metrics_df.to_string())
print("\nModels saved to /kaggle/working/")
print("Visualization saved to /kaggle/working/training_results.png")
