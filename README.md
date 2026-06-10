# 波士顿房价预测 — 机器学习回归项目

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-green)](https://xgboost.readthedocs.io)

## 项目概述

使用机器学习技术对经典**波士顿房价数据集**进行回归预测。根据 13 个房屋及周边环境特征，构建多种回归模型预测房价中位数（MEDV）。

### 核心结果

| 模型 | RMSE | R² |
|------|------|-----|
| 线性回归 | 4.93 | 0.669 |
| 随机森林 | 2.79 | 0.894 |
| **GBDT（最优）** | **2.49** | **0.915** |
| XGBoost | 2.56 | 0.911 |

- **最优模型**: Gradient Boosting (GBDT)，R² = **0.915**
- **最关键特征**: LSTAT（低收入比例，43%）+ RM（房间数，30%）

## 项目结构

```
├── .gitignore
├── README.md                    # 项目说明
├── 报告.md                      # 完整实验报告（含双语分析）
├── 实验报告_波士顿房价预测.docx  # Word 版报告
├── code/
│   ├── main.py                  # 一键运行代码（6 个 Scikit-learn 模型 + PaddlePaddle 风格）
│   └── requirements.txt         # 依赖清单
└── output/                      # 生成的可视化结果
    ├── 01_房价分布直方图.png      # EDA：目标变量分布
    ├── 02_相关性热力图.png         # 特征相关性矩阵
    ├── 03_重要特征散点图.png       # Top-4 特征 vs 房价
    ├── 04_模型性能对比.png         # 多模型 RMSE / R² 对比
    ├── 05_预测值vs实际值.png       # 最优模型预测效果
    ├── 06_残差分析图.png           # 残差诊断
    ├── 07_特征重要性.png           # XGBoost 特征重要性
    ├── 08_submission.csv          # 预测结果文件
    ├── 09_PaddlePaddle_损失曲线.png  # PaddlePaddle 训练曲线
    └── 10_PaddlePaddle_预测值vs实际值.png
```

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
cd code
pip install -r requirements.txt
```

### 运行代码

```bash
cd code
python main.py
```

代码将自动：
1. 从 UCI 仓库下载波士顿房价数据集
2. 执行完整的 EDA（生成 3 张分析图）
3. 训练 6 种回归模型（含 5 折交叉验证）
4. 输出最优模型及特征重要性
5. 生成全部可视化结果至 `output/` 目录


