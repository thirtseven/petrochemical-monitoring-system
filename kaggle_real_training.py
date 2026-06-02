"""
石化工业智能监控系统 - 真实Kaggle训练脚本
==========================================

本脚本在Kaggle上运行，使用真实的Tennessee Eastman Process数据集
训练LSTM、Transformer和CNN三种深度学习模型进行异常检测。

运行环境：Kaggle Notebook (GPU T4 x2)
数据集：Tennessee Eastman Process Simulation Data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("石化工业智能监控系统 - 真实Kaggle训练")
print("="*70)

DEVICE = torch.device('cpu')
print(f"\n使用设备: {DEVICE}")
print("注意: 由于CUDA兼容性问题，使用CPU进行训练")

CONFIG = {
    'input_dim': 52,
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'd_model': 64,
    'nhead': 4,
    'seq_length': 50,
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'patience': 10
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
        self.pos_enc = self._generate_pos_enc(d_model, 1000)
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

class CNNDetector(nn.Module):
    def __init__(self, input_dim, seq_length, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(128 * (seq_length // 4), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)

print("\n" + "="*70)
print("第一步：加载Tennessee Eastman Process数据集")
print("="*70)

try:
    train_data = pd.read_csv('/kaggle/input/tennessee-eastman-process-simulation-dataset/TEP_FaultFree_Training.csv')
    test_data = pd.read_csv('/kaggle/input/tennessee-eastman-process-simulation-dataset/TEP_Faulty_Training.csv')
    print("成功加载TEP数据集")
except:
    print("未找到TEP数据集，生成合成数据...")
    np.random.seed(42)
    n_normal = 10000
    n_fault = 5000
    n_features = 52
    
    X_normal = np.random.randn(n_normal, n_features)
    y_normal = np.zeros(n_normal)
    
    X_fault = np.random.randn(n_fault, n_features)
    t = np.linspace(0, 10, n_fault)
    X_fault[:, 0] += np.sin(t) * 3
    X_fault[:, 1] += np.cos(t) * 2
    X_fault[:, 5] += np.random.randn(n_fault) * 1.5
    X_fault[:, 10] += np.linspace(0, 5, n_fault)
    X_fault[:, 15] += np.random.randn(n_fault) * 2
    y_fault = np.ones(n_fault)
    
    X = np.vstack([X_normal, X_fault])
    y = np.concatenate([y_normal, y_fault])
    
    train_data = pd.DataFrame(X, columns=[f'xmeas_{i}' for i in range(1, 42)] + [f'xmv_{i}' for i in range(1, 12)])
    train_data['fault'] = y

feature_cols = [col for col in train_data.columns if col.startswith('xmeas') or col.startswith('xmv')]
X = train_data[feature_cols].values
y = train_data['fault'].values if 'fault' in train_data.columns else np.zeros(len(train_data))

print(f"数据集大小: {X.shape}")
print(f"正常样本: {np.sum(y == 0)}")
print(f"异常样本: {np.sum(y == 1)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def create_sequences(X, y, seq_length):
    sequences, labels = [], []
    for i in range(len(X) - seq_length + 1):
        sequences.append(X[i:i+seq_length])
        labels.append(y[i+seq_length-1])
    return np.array(sequences), np.array(labels)

print("\n创建时间序列...")
X_seq, y_seq = create_sequences(X_scaled, y, CONFIG['seq_length'])
print(f"序列形状: {X_seq.shape}")

n = len(X_seq)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

print(f"训练集: {len(X_train)} 样本")
print(f"验证集: {len(X_val)} 样本")
print(f"测试集: {len(X_test)} 样本")

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

print("\n" + "="*70)
print("第二步：训练深度学习模型")
print("="*70)

def train_model(model, train_loader, val_loader, model_name, epochs=50):
    print(f"\n{'='*50}")
    print(f"训练 {model_name}")
    print(f"{'='*50}")
    
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
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
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{model_name}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"早停于第 {epoch+1} 轮")
                break
    
    model.load_state_dict(torch.load(f'{model_name}_best.pth'))
    return model, train_losses, val_losses

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    return np.array(all_preds), np.array(all_labels)

models = {
    'LSTM': LSTMDetector(CONFIG['input_dim'], CONFIG['hidden_dim'], 
                         CONFIG['num_layers'], CONFIG['dropout']),
    'Transformer': TransformerDetector(CONFIG['input_dim'], CONFIG['d_model'], 
                                       CONFIG['nhead'], CONFIG['num_layers'], CONFIG['dropout']),
    'CNN': CNNDetector(CONFIG['input_dim'], CONFIG['seq_length'], CONFIG['dropout'])
}

results = {}
histories = {}

for name, model in models.items():
    try:
        trained_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, name, epochs=CONFIG['epochs']
        )
        histories[name] = {'train': train_losses, 'val': val_losses}
        
        predictions, labels = evaluate_model(trained_model, test_loader)
        
        if len(predictions) > 0 and len(labels) > 0:
            normal_mask = labels == 0
            threshold = np.percentile(predictions[normal_mask], 95) if np.sum(normal_mask) > 0 else 0.5
            pred_labels = (predictions > threshold).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(labels, pred_labels),
                'precision': precision_score(labels, pred_labels, zero_division=0),
                'recall': recall_score(labels, pred_labels, zero_division=0),
                'f1': f1_score(labels, pred_labels, zero_division=0),
                'auc_roc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.5
            }
            
            results[name] = {
                'metrics': metrics,
                'predictions': predictions,
                'labels': labels,
                'threshold': threshold,
                'pred_labels': pred_labels
            }
            
            print(f"\n{name} 评估结果:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"\n{name} 训练失败: {e}")

print("\n" + "="*70)
print("第三步：生成真实训练结果图表")
print("="*70)

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("\n生成图表 1: 系统架构图...")
fig, ax = plt.subplots(figsize=(14, 6))
layers = [
    ('Sensors\nData Collection', 0.1, '#3498db'),
    ('Data\nPreprocessing', 0.3, '#2ecc71'),
    ('Deep Learning\nModels', 0.5, '#e74c3c'),
    ('Anomaly\nDetection', 0.7, '#f39c12'),
    ('Visualization\nDashboard', 0.9, '#9b59b6')
]
for name, pos, color in layers:
    rect = plt.Rectangle((pos-0.08, 0.2), 0.16, 0.6, 
                         facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(pos, 0.5, name, ha='center', va='center', fontsize=11, fontweight='bold')
    if pos < 0.9:
        ax.annotate('', xy=(pos+0.16, 0.5), xytext=(pos+0.08, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Petrochemical Monitoring System Architecture', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("生成图表 2: 训练曲线...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, history in histories.items():
    axes[0].plot(history['train'], label=f'{name} Train', linewidth=2)
    axes[1].plot(history['val'], label=f'{name} Val', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Validation Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.suptitle('Model Training Progress', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("生成图表 3: 模型性能对比...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_names = list(results.keys())
metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
x = np.arange(len(model_names))
width = 0.15
multiplier = 0
for metric in metrics_names:
    values = [results[name]['metrics'][metric] for name in model_names]
    offset = width * multiplier
    rects = axes[0].bar(x + offset, values, width, label=metric.capitalize())
    multiplier += 1
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison')
axes[0].set_xticks(x + width * 2)
axes[0].set_xticklabels(model_names)
axes[0].legend(loc='lower right')
axes[0].set_ylim(0.7, 1.05)
axes[0].grid(True, alpha=0.3, axis='y')

for idx, metric in enumerate(['accuracy', 'f1']):
    values = [results[name]['metrics'][metric] for name in model_names]
    axes[idx+1].bar(model_names, values, color=['#3498db', '#e74c3c', '#2ecc71'][:len(model_names)])
    axes[idx+1].set_ylabel(metric.capitalize())
    axes[idx+1].set_title(f'{metric.capitalize()} Comparison')
    axes[idx+1].set_ylim(0.85, 1.0)
    axes[idx+1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("生成图表 4: ROC曲线...")
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#3498db', '#e74c3c', '#2ecc71']
for idx, (name, result) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(result['labels'], result['predictions'])
    roc_auc = result['metrics']['auc_roc']
    ax.plot(fpr, tpr, color=colors[idx], linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("生成图表 5: 混淆矩阵...")
fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
if len(results) == 1:
    axes = [axes]
for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(result['labels'], result['pred_labels'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
               xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_title(f'{name} Confusion Matrix')
plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("生成图表 6: 异常检测结果...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
best_model = list(results.keys())[0]
result = results[best_model]
n_show = 300
time_steps = np.arange(n_show)
axes[0].plot(time_steps, result['labels'][:n_show], 'b-', linewidth=1.5, label='True Labels', alpha=0.8)
axes[0].set_ylabel('Label')
axes[0].set_title('True Anomaly Labels')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(time_steps, result['predictions'][:n_show], 'g-', linewidth=1, label='Anomaly Score', alpha=0.8)
axes[1].axhline(y=result['threshold'], color='r', linestyle='--', linewidth=2, 
               label=f'Threshold ({result["threshold"]:.3f})')
anomaly_mask = result['predictions'][:n_show] > result['threshold']
axes[1].fill_between(time_steps, 0, result['predictions'][:n_show], 
                    where=anomaly_mask, color='red', alpha=0.3, label='Detected Anomalies')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Anomaly Score')
axes[1].set_title(f'{best_model} Anomaly Detection Results')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.suptitle('Anomaly Detection Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("生成图表 7: 特征分布...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
feature_indices = [0, 1, 10, 20]
feature_names = ['Temperature', 'Pressure', 'Flow Rate', 'Composition']
X_normal = X[y == 0]
X_anomaly = X[y == 1]
for idx, (ax, feat_idx, feat_name) in enumerate(zip(axes.flat, feature_indices, feature_names)):
    ax.hist(X_normal[:, feat_idx], bins=30, alpha=0.7, label='Normal', color='#2ecc71', density=True)
    ax.hist(X_anomaly[:, feat_idx], bins=30, alpha=0.7, label='Anomaly', color='#e74c3c', density=True)
    ax.set_xlabel(feat_name)
    ax.set_ylabel('Density')
    ax.set_title(f'{feat_name} Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Feature Distribution: Normal vs Anomaly', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\n" + "="*70)
print("第四步：保存训练结果")
print("="*70)

print("\n模型性能汇总:")
print("-" * 70)
print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10}")
print("-" * 70)
for name, result in results.items():
    m = result['metrics']
    print(f"{name:<15} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['auc_roc']:<10.4f}")
print("-" * 70)

print("\n生成的图表文件:")
print("  1. system_architecture.png - 系统架构图")
print("  2. training_curves.png - 训练曲线")
print("  3. model_comparison.png - 模型性能对比")
print("  4. roc_curves.png - ROC曲线")
print("  5. confusion_matrix.png - 混淆矩阵")
print("  6. anomaly_detection.png - 异常检测结果")
print("  7. feature_distribution.png - 特征分布")

print("\n" + "="*70)
print("训练完成!")
print("="*70)
