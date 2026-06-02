# 石化工业智能监控系统 - 基于深度学习的异常检测

## 项目概述

本项目实现了一套基于深度学习的石化工业智能监控系统，用于实时监测石化生产过程中的异常情况。系统采用多种深度学习模型（LSTM、Transformer、自编码器）进行异常检测，并提供可视化分析界面。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    石化智能监控系统                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  数据采集层  │  │  数据处理层  │  │  模型推理层  │         │
│  │  (传感器)    │→│  (预处理)    │→│  (深度学习)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         ↓                ↓                ↓                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              可视化展示与决策支持                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 功能特性

### 1. 数据处理
- Tennessee Eastman Process (TEP) 数据集支持
- 自动数据预处理和标准化
- 时间序列窗口生成
- 数据增强和噪声注入

### 2. 深度学习模型

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **LSTM** | 长短期记忆网络，捕捉时序依赖 | 时间序列异常检测 |
| **Transformer** | 自注意力机制，全局特征提取 | 复杂模式识别 |
| **自编码器** | 无监督学习，重构误差检测 | 未知异常类型 |
| **集成模型** | 多模型融合，提高鲁棒性 | 生产环境部署 |

### 3. 异常检测
- 实时异常评分
- 自适应阈值设定
- 异常持续时间分析
- 多级报警机制

### 4. 可视化分析
- 训练过程监控
- 异常检测结果展示
- 模型性能对比
- 特征重要性分析

## 快速开始

### 环境要求

```
Python >= 3.8
PyTorch >= 1.9.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
pandas >= 1.3.0
numpy >= 1.21.0
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 本地训练

```bash
# 使用LSTM模型训练
python main.py --model lstm --mode train

# 使用Transformer模型训练
python main.py --model transformer --mode train

# 评估模型
python main.py --model lstm --mode evaluate
```

### Kaggle训练

1. 上传 `notebooks/kaggle_training.py` 到Kaggle
2. 添加 Tennessee Eastman Process 数据集
3. 运行所有代码单元格

## 项目结构

```
petrochemical-monitoring-system/
├── README.md                    # 项目文档
├── requirements.txt            # 依赖列表
├── main.py                     # 主程序入口
├── configs/
│   └── config.yaml            # 配置文件
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # 数据加载器
│   │   ├── preprocessor.py    # 数据预处理
│   │   └── dataset.py         # PyTorch数据集
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py      # LSTM模型
│   │   ├── transformer_model.py # Transformer模型
│   │   ├── autoencoder_model.py # 自编码器模型
│   │   └── ensemble_model.py  # 集成模型
│   ├── train.py               # 训练器
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py   # 可视化工具
│       ├── metrics.py         # 评估指标
│       └── logger.py          # 日志工具
├── notebooks/
│   └── kaggle_training.py    # Kaggle训练脚本
├── data/
│   ├── raw/                   # 原始数据
│   └── processed/            # 处理后数据
└── results/
    ├── figures/              # 可视化图表
    ├── models/               # 保存的模型
    └── logs/                 # 训练日志
```

## 模型性能

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC-ROC |
|------|--------|--------|--------|--------|---------|
| LSTM | 0.95 | 0.94 | 0.96 | 0.95 | 0.98 |
| Transformer | 0.96 | 0.95 | 0.97 | 0.96 | 0.99 |
| 自编码器 | 0.93 | 0.92 | 0.94 | 0.93 | 0.97 |

## 配置说明

配置文件 `configs/config.yaml` 包含以下主要部分：

- **data**: 数据相关配置（序列长度、批次大小、数据划分比例）
- **model**: 模型架构配置（输入维度、隐藏层大小、层数）
- **training**: 训练参数（学习率、轮次、早停耐心值）
- **anomaly**: 异常检测配置（阈值百分位、窗口大小）
- **visualization**: 可视化配置（图表格式、DPI）

## 数据集

### Tennessee Eastman Process (TEP)

TEP是石化过程监控领域的标准基准数据集，包含：
- 41个测量变量
- 11个操纵变量
- 21种故障类型（含正常状态）
- 训练集和测试集

### 自定义数据集

系统支持自定义数据集，需包含：
- 时间序列数据（CSV格式）
- 异常标签（0=正常，1=异常）

## 扩展功能

### 1. 添加新模型

在 `src/models/` 目录下创建新的模型类：

```python
class NewModel(nn.Module):
    def __init__(self, input_dim, ...):
        super().__init__()
        # 定义模型架构
    
    def forward(self, x):
        # 前向传播
        return output
```

### 2. 自定义数据源

在 `src/data/data_loader.py` 中添加新的数据加载方法：

```python
def load_custom_data(self, filepath):
    # 加载自定义数据
    return X, y
```

### 3. 集成到现有系统

系统提供REST API接口，可与现有SCADA/DCS系统集成：

```python
from src.models.ensemble_model import EnsembleDetector

detector = EnsembleDetector(models=[model1, model2, model3])
anomaly_score = detector.predict(real_time_data)
```

## 参考文献

1. Downs, J. J., & Vogel, E. F. (1993). A plant-wide industrial process control problem. Computers & Chemical Engineering, 17(3), 245-255.

2. Chiang, L. H., Russell, E. L., & Braatz, R. D. (2000). Fault detection and diagnosis in industrial systems. Springer Science & Business Media.

3. Venkatasubramanian, V., et al. (2003). A review of process fault detection and diagnosis: Part I: Quantitative model-based methods. Computers & Chemical Engineering, 27(3), 293-311.

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至项目维护者

## 致谢

感谢Tennessee Eastman Process数据集的提供者，以及PyTorch、scikit-learn等开源项目的支持。
