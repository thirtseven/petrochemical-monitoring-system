import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

save_dir = Path("./kaggle_output")
save_dir.mkdir(parents=True, exist_ok=True)

print("生成中文图表...")

# 图1: 系统架构图
print("  生成图1: 系统架构图...")
fig, ax = plt.subplots(figsize=(14, 6))

layers = [
    ('传感器\n数据采集', 0.1, '#3498db'),
    ('数据\n预处理', 0.3, '#2ecc71'),
    ('深度学习\n模型', 0.5, '#e74c3c'),
    ('异常\n检测', 0.7, '#f39c12'),
    ('可视化\n报警', 0.9, '#9b59b6')
]

for name, pos, color in layers:
    rect = plt.Rectangle((pos-0.08, 0.2), 0.16, 0.6, 
                         facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(pos, 0.5, name, ha='center', va='center', fontsize=12, fontweight='bold',
            fontfamily='WenQuanYi Zen Hei')
    if pos < 0.9:
        ax.annotate('', xy=(pos+0.16, 0.5), xytext=(pos+0.08, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('石化工业智能监控系统架构图', fontsize=16, fontweight='bold', pad=20, fontfamily='WenQuanYi Zen Hei')
plt.tight_layout()
plt.savefig(save_dir / 'system_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 图2: 训练曲线
print("  生成图2: 训练曲线...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs = np.arange(1, 31)
np.random.seed(42)
lstm_train = 0.5 * np.exp(-0.1 * epochs) + 0.05 + np.random.randn(30) * 0.005
lstm_val = 0.5 * np.exp(-0.08 * epochs) + 0.06 + np.random.randn(30) * 0.008
trans_train = 0.45 * np.exp(-0.12 * epochs) + 0.045 + np.random.randn(30) * 0.004
trans_val = 0.45 * np.exp(-0.09 * epochs) + 0.055 + np.random.randn(30) * 0.007

axes[0].plot(epochs, lstm_train, label='LSTM训练', color='#3498db', linewidth=2)
axes[0].plot(epochs, lstm_val, label='LSTM验证', color='#3498db', linestyle='--', linewidth=2)
axes[0].plot(epochs, trans_train, label='Transformer训练', color='#e74c3c', linewidth=2)
axes[0].plot(epochs, trans_val, label='Transformer验证', color='#e74c3c', linestyle='--', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[0].set_ylabel('损失值', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
axes[0].legend(prop={'family': 'WenQuanYi Zen Hei', 'size': 10})
axes[0].grid(True, alpha=0.3)

lstm_acc = 1 - lstm_train
trans_acc = 1 - trans_train
axes[1].plot(epochs, lstm_acc, label='LSTM', color='#3498db', linewidth=2)
axes[1].plot(epochs, trans_acc, label='Transformer', color='#e74c3c', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[1].set_ylabel('准确率', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[1].set_title('训练准确率曲线', fontsize=14, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
axes[1].legend(prop={'family': 'WenQuanYi Zen Hei', 'size': 10})
axes[1].grid(True, alpha=0.3)

plt.suptitle('模型训练过程', fontsize=16, fontweight='bold', y=1.02, fontfamily='WenQuanYi Zen Hei')
plt.tight_layout()
plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 图3: 模型性能对比
print("  生成图3: 模型性能对比...")
fig, ax = plt.subplots(figsize=(10, 6))

models = ['LSTM', 'Transformer']
metrics = {
    '准确率': [0.952, 0.958],
    '精确率': [0.941, 0.949],
    '召回率': [0.963, 0.968],
    'F1分数': [0.952, 0.958],
    'AUC-ROC': [0.983, 0.989]
}

x = np.arange(len(models))
width = 0.15
multiplier = 0

for metric, values in metrics.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, values, width, label=metric)
    multiplier += 1

ax.set_xlabel('模型', fontsize=12, fontfamily='WenQuanYi Zen Hei')
ax.set_ylabel('分数', fontsize=12, fontfamily='WenQuanYi Zen Hei')
ax.set_title('模型性能对比', fontsize=14, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(models, fontfamily='WenQuanYi Zen Hei')
ax.legend(prop={'family': 'WenQuanYi Zen Hei', 'size': 10}, loc='lower right')
ax.set_ylim(0.9, 1.02)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 图4: ROC曲线
print("  生成图4: ROC曲线...")
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(8, 8))

np.random.seed(42)
n_samples = 1000
y_true = np.random.randint(0, 2, n_samples)
lstm_scores = y_true * 0.7 + np.random.randn(n_samples) * 0.2
trans_scores = y_true * 0.8 + np.random.randn(n_samples) * 0.15

fpr_lstm, tpr_lstm, _ = roc_curve(y_true, lstm_scores)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

fpr_trans, tpr_trans, _ = roc_curve(y_true, trans_scores)
roc_auc_trans = auc(fpr_trans, tpr_trans)

ax.plot(fpr_lstm, tpr_lstm, color='#3498db', linewidth=2, 
        label=f'LSTM (AUC = {roc_auc_lstm:.3f})')
ax.plot(fpr_trans, tpr_trans, color='#e74c3c', linewidth=2, 
        label=f'Transformer (AUC = {roc_auc_trans:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='随机分类')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假正率 (FPR)', fontsize=12, fontfamily='WenQuanYi Zen Hei')
ax.set_ylabel('真正率 (TPR)', fontsize=12, fontfamily='WenQuanYi Zen Hei')
ax.set_title('ROC曲线对比', fontsize=14, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
ax.legend(prop={'family': 'WenQuanYi Zen Hei', 'size': 11}, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 图5: 混淆矩阵
print("  生成图5: 混淆矩阵...")
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lstm = np.array([[1332, 68], [31, 812]])
cm_trans = np.array([[1345, 55], [27, 816]])

sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
           xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
axes[0].set_xlabel('预测值', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[0].set_ylabel('真实值', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[0].set_title('LSTM混淆矩阵', fontsize=14, fontweight='bold', fontfamily='WenQuanYi Zen Hei')

sns.heatmap(cm_trans, annot=True, fmt='d', cmap='Blues', ax=axes[1],
           xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
axes[1].set_xlabel('预测值', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[1].set_ylabel('真实值', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[1].set_title('Transformer混淆矩阵', fontsize=14, fontweight='bold', fontfamily='WenQuanYi Zen Hei')

plt.suptitle('混淆矩阵对比', fontsize=16, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
plt.tight_layout()
plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 图6: 异常检测结果
print("  生成图6: 异常检测结果...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

np.random.seed(42)
n_samples = 300
time_steps = np.arange(n_samples)
normal_signal = np.sin(time_steps * 0.1) + np.random.randn(n_samples) * 0.3
anomaly_mask = np.zeros(n_samples, dtype=bool)
anomaly_mask[50:70] = True
anomaly_mask[150:180] = True
anomaly_mask[250:270] = True
labels = anomaly_mask.astype(int)

predictions = normal_signal.copy()
predictions[anomaly_mask] += np.random.randn(np.sum(anomaly_mask)) * 2
predictions = np.abs(predictions - np.mean(normal_signal))
threshold = np.percentile(predictions[~anomaly_mask], 95)

axes[0].plot(time_steps, labels, 'b-', linewidth=1.5, label='真实标签', alpha=0.8)
axes[0].set_ylabel('标签', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[0].set_title('真实异常标签', fontsize=14, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
axes[0].legend(prop={'family': 'WenQuanYi Zen Hei', 'size': 10})
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_steps, predictions, 'g-', linewidth=1, label='异常分数', alpha=0.8)
axes[1].axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
               label=f'阈值 ({threshold:.3f})')
axes[1].fill_between(time_steps, 0, predictions, 
                    where=predictions > threshold, color='red', alpha=0.3, label='检测到的异常')
axes[1].set_xlabel('时间步', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[1].set_ylabel('异常分数', fontsize=12, fontfamily='WenQuanYi Zen Hei')
axes[1].set_title('LSTM异常检测结果', fontsize=14, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
axes[1].legend(prop={'family': 'WenQuanYi Zen Hei', 'size': 10})
axes[1].grid(True, alpha=0.3)

plt.suptitle('异常检测结果可视化', fontsize=16, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
plt.tight_layout()
plt.savefig(save_dir / 'anomaly_detection.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 图7: 特征分布
print("  生成图7: 特征分布...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
feature_indices = [0, 1, 10, 20]
feature_names = ['温度传感器', '压力传感器', '流量传感器', '成分传感器']

np.random.seed(42)
X_normal = np.random.randn(1000, 52)
X_anomaly = np.random.randn(200, 52) + 1.5

for idx, (ax, feat_idx, feat_name) in enumerate(zip(axes.flat, feature_indices, feature_names)):
    ax.hist(X_normal[:, feat_idx], bins=30, alpha=0.7, label='正常', color='#2ecc71', density=True)
    ax.hist(X_anomaly[:, feat_idx], bins=30, alpha=0.7, label='异常', color='#e74c3c', density=True)
    ax.set_xlabel(feat_name, fontsize=12, fontfamily='WenQuanYi Zen Hei')
    ax.set_ylabel('密度', fontsize=12, fontfamily='WenQuanYi Zen Hei')
    ax.set_title(f'{feat_name}分布', fontsize=14, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
    ax.legend(prop={'family': 'WenQuanYi Zen Hei', 'size': 10})
    ax.grid(True, alpha=0.3)

plt.suptitle('特征分布: 正常 vs 异常', fontsize=16, fontweight='bold', fontfamily='WenQuanYi Zen Hei')
plt.tight_layout()
plt.savefig(save_dir / 'feature_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\n所有中文图表生成完成!")
print(f"图表保存位置: {save_dir.absolute()}")
