# -*- coding: utf-8 -*-
"""
============================================================================
实验项目一：波士顿房价预测
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
from copy import deepcopy

# ---------------------------------------------------------------------------
# Scikit-learn 模型与工具
# ---------------------------------------------------------------------------
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 全局设置
# ---------------------------------------------------------------------------
# 中文字体支持
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 1. 数据加载
# ============================================================================
print("=" * 70)
print("实验项目一：波士顿房价预测 -- 完整复现")
print("=" * 70)


def load_boston_data():
    """
    从 UCI 仓库加载波士顿房价数据集。
    该数据集共 506 个样本，13 个特征 + 1 个目标变量（MEDV）。
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df["MEDV"] = target
    return df, feature_names


df, feature_names = load_boston_data()
print(f"\n[1] 数据集加载完成：{df.shape[0]} 个样本，{len(feature_names)} 个特征")
print(df.head(8).to_string())
print("\n基本统计量：")
print(df.describe().round(2))

# 检查缺失值
missing = df.isnull().sum()
print(f"\n缺失值统计：{(' 无缺失值' if not missing.any() else str(missing[missing > 0]))}")

# ============================================================================
# 2. 探索性数据分析（EDA）
# ============================================================================
print("\n" + "=" * 70)
print("[2] 探索性数据分析（EDA）")
print("=" * 70)

# 2.1 目标变量分布直方图
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["MEDV"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(df["MEDV"].mean(), color="red", linestyle="--", linewidth=1.5,
           label=f'均值 = {df["MEDV"].mean():.2f}')
ax.axvline(df["MEDV"].median(), color="green", linestyle="-.", linewidth=1.5,
           label=f'中位数 = {df["MEDV"].median():.2f}')
ax.set_xlabel("房价中位数 MEDV（千美元）")
ax.set_ylabel("频数")
ax.set_title("波士顿房价分布直方图")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_房价分布直方图.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] 01_房价分布直方图.png")

# 2.2 相关性热力图
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title("特征与目标变量相关性热力图", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_相关性热力图.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] 02_相关性热力图.png")

# 输出与 MEDV 相关性排序
corr_with_target = corr_matrix["MEDV"].drop("MEDV").sort_values()
print("\n  各特征与 MEDV 的相关性（从负到正）：")
for feat, val in corr_with_target.items():
    hashes = "#" * int(abs(val) * 20) + "-" * (20 - int(abs(val) * 20))
    print(f"    {feat:>8s}: {val:>6.3f}  [{hashes}]")

# 2.3 Top-4 最重要特征散点图
top4_feats = corr_matrix["MEDV"].drop("MEDV").abs().sort_values(ascending=False).head(4).index.tolist()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, feat in enumerate(top4_feats):
    axes[i].scatter(df[feat], df["MEDV"], alpha=0.5, s=15, color="steelblue", edgecolors="k", linewidth=0.3)
    z = np.polyfit(df[feat], df["MEDV"], 1)
    p = np.poly1d(z)
    x_sorted = np.sort(df[feat])
    axes[i].plot(x_sorted, p(x_sorted), "r--", linewidth=1.5, alpha=0.8)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel("MEDV（千美元）")
    axes[i].set_title(f"{feat} vs 房价  (r={corr_matrix.loc[feat, 'MEDV']:.3f})")
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_重要特征散点图.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] 03_重要特征散点图.png")
print(f"  最重要的 4 个特征：{', '.join(top4_feats)}")

# ============================================================================
# 3. 数据预处理与特征工程
# ============================================================================
print("\n" + "=" * 70)
print("[3] 数据预处理与特征工程")
print("=" * 70)

X = df[feature_names].values
y = df["MEDV"].values

# 划分训练集 / 测试集（80% / 20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\n  训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

# ---------------------------------------------------------------------------
# 方式 A：StandardScaler 标准化
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  [OK] StandardScaler 标准化完成")

# ---------------------------------------------------------------------------
# 方式 B：Min-Max 归一化
# ---------------------------------------------------------------------------
X_train_minmax = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8)
X_test_minmax = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8)
print("  [OK] Min-Max 归一化完成（参考 PaddlePaddle 方式）")

# ============================================================================
# 4. 模型训练与评估
# ============================================================================
print("\n" + "=" * 70)
print("[4] 模型训练与评估")
print("=" * 70)

# 定义模型
models = {
    "线性回归 (Linear Regression)": LinearRegression(),
    "岭回归 (Ridge)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso 回归": Lasso(alpha=0.01, random_state=RANDOM_STATE),
    "随机森林 (Random Forest)": RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE
    ),
    "梯度提升树 (GBDT)": GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=RANDOM_STATE, verbosity=0
    ),
}

# ----------------------------- 5折交叉验证 -----------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

print("\n  ----- 5折交叉验证结果 (RMSE) -----")
print(f"  {'模型':<30s} {'平均 RMSE':<12s} {'标准差':<10s} {'平均 R2':<10s}")
print("  " + "-" * 62)

for name, model in models.items():
    neg_mse_scores = cross_val_score(model, X_train_scaled, y_train,
                                     cv=kfold, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse_scores)
    r2_scores = cross_val_score(model, X_train_scaled, y_train,
                                cv=kfold, scoring="r2")
    cv_results[name] = {
        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(),
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
    }
    print(f"  {name:<30s} {rmse_scores.mean():>8.4f} +- {rmse_scores.std():.4f}  "
          f"{r2_scores.mean():>7.4f} +- {r2_scores.std():.4f}")

# ----------------------------- 在测试集上评估 -----------------------------
print("\n  ----- 测试集评估结果 -----")
print(f"  {'模型':<30s} {'RMSE':<10s} {'R2':<10s} {'MAE':<10s}")
print("  " + "-" * 60)

test_results = []
best_model_name = None
best_model_r2 = -np.inf
best_model_obj = None

for name, model in models.items():
    model_clone = deepcopy(model)
    model_clone.fit(X_train_scaled, y_train)
    y_pred = model_clone.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    test_results.append({
        "model": name,
        "RMSE": rmse,
        "R2": r2,
        "MAE": mae,
    })
    print(f"  {name:<30s} {rmse:<8.4f}  {r2:<8.4f}  {mae:<8.4f}")

    if r2 > best_model_r2:
        best_model_r2 = r2
        best_model_name = name
        best_model_obj = model_clone

print(f"\n  >>> 最优模型: {best_model_name} (测试集 R2 = {best_model_r2:.4f})")

test_df = pd.DataFrame(test_results)

# ============================================================================
# 5. 模型性能对比可视化
# ============================================================================
print("\n" + "=" * 70)
print("[5] 结果可视化")
print("=" * 70)

# 5.1 模型性能对比柱状图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.Set2(np.linspace(0, 1, len(test_df)))
names_short = [n.split(" (")[0] for n in test_df["model"]]

ax1 = axes[0]
bars1 = ax1.bar(names_short, test_df["RMSE"], color=colors, edgecolor="black", alpha=0.85)
for bar, val in zip(bars1, test_df["RMSE"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)
ax1.set_ylabel("RMSE（千美元）")
ax1.set_title("各模型 RMSE 对比（越低越好）")
ax1.tick_params(axis="x", rotation=30)
ax1.grid(True, alpha=0.3, axis="y")

ax2 = axes[1]
bars2 = ax2.bar(names_short, test_df["R2"], color=colors, edgecolor="black", alpha=0.85)
for bar, val in zip(bars2, test_df["R2"]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax2.set_ylabel("R2 决定系数")
ax2.set_title("各模型 R2 对比（越高越好）")
ax2.tick_params(axis="x", rotation=30)
ax2.grid(True, alpha=0.3, axis="y")

plt.suptitle("模型性能对比", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_模型性能对比.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] 04_模型性能对比.png")

# 5.2 最优模型：预测值 vs 实际值
y_best_pred = best_model_obj.predict(X_test_scaled)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_best_pred, alpha=0.6, color="steelblue",
           edgecolors="k", linewidth=0.5, s=40)
min_val = min(y_test.min(), y_best_pred.min())
max_val = max(y_test.max(), y_best_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="完美预测线")
ax.set_xlabel("实际房价（千美元）")
ax.set_ylabel("预测房价（千美元）")
ax.set_title(f"最优模型 [{best_model_name}] 预测值 vs 实际值\n"
             f"RMSE = {np.sqrt(mean_squared_error(y_test, y_best_pred)):.4f}, "
             f"R2 = {r2_score(y_test, y_best_pred):.4f}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_预测值vs实际值.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] 05_预测值vs实际值.png")

# 5.3 残差分析
residuals = y_test - y_best_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_best_pred, residuals, alpha=0.6, color="steelblue",
                edgecolors="k", linewidth=0.5, s=30)
axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
axes[0].set_xlabel("预测房价（千美元）")
axes[0].set_ylabel("残差（千美元）")
axes[0].set_title("残差分布图（理想状态：随机散布于 y=0 两侧）")
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=20, edgecolor="black", alpha=0.7,
             color="steelblue", density=True)
from scipy.stats import norm
mu, sigma = residuals.mean(), residuals.std()
x_range = np.linspace(residuals.min(), residuals.max(), 100)
axes[1].plot(x_range, norm.pdf(x_range, mu, sigma), "r-", linewidth=2,
             label=f"正态拟合 (mu={mu:.2f}, sigma={sigma:.2f})")
axes[1].set_xlabel("残差（千美元）")
axes[1].set_ylabel("密度")
axes[1].set_title("残差分布直方图")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"残差分析 -- {best_model_name}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_残差分析图.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] 06_残差分析图.png")

# 5.4 特征重要性分析
print("\n  ----- 特征重要性分析 -----")

xgb_model = xgb.XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    random_state=RANDOM_STATE, verbosity=0
)
xgb_model.fit(X_train_scaled, y_train)

importance = xgb_model.feature_importances_
sorted_idx = np.argsort(importance)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(sorted_idx)), importance[sorted_idx],
               color=plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_idx))),
               edgecolor="black")
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel("特征重要性（F-score）")
ax.set_title("XGBoost 特征重要性排序", fontsize=14)
ax.grid(True, alpha=0.3, axis="x")

for bar, val in zip(bars, importance[sorted_idx]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", ha="left", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_特征重要性.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] 07_特征重要性.png")

print(f"\n  XGBoost 特征重要性排名：")
for rank, idx in enumerate(reversed(sorted_idx), 1):
    print(f"    {rank}. {feature_names[idx]:>8s}: {importance[idx]:.4f}")

# ============================================================================
# 6. 生成预测结果文件（submission.csv）
# ============================================================================
print("\n" + "=" * 70)
print("[6] 生成预测结果文件")
print("=" * 70)

y_full_pred = best_model_obj.predict(scaler.transform(X))

submission_df = pd.DataFrame({
    "ID": np.arange(1, len(y_full_pred) + 1),
    "MEDV": np.round(y_full_pred, 2),
})
submission_path = os.path.join(OUTPUT_DIR, "08_submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"  [OK] 08_submission.csv 已保存（共 {len(submission_df)} 条预测记录）")
print(submission_df.head(10).to_string(index=False))

# ============================================================================
# 7. PaddlePaddle 风格线性回归（NumPy 实现）
# ============================================================================
print("\n" + "=" * 70)
print("[7] PaddlePaddle 风格线性回归（NumPy 实现）")
print("=" * 70)


class LinearRegressionNumpy:
    """使用 NumPy 实现的线性回归（模拟 PaddlePaddle Linear 层 + SGD）"""

    def __init__(self, n_features, learning_rate=0.01):
        limit = np.sqrt(6.0 / (n_features + 1))
        self.w = np.random.uniform(-limit, limit, size=(n_features, 1))
        self.b = 0.0
        self.lr = learning_rate

    def forward(self, X):
        return X @ self.w + self.b

    def compute_loss(self, y_pred, y_true):
        diff = y_pred - y_true.reshape(-1, 1)
        return np.mean(diff ** 2)

    def backward(self, X, y_pred, y_true):
        m = X.shape[0]
        diff = y_pred - y_true.reshape(-1, 1)
        dw = (2.0 / m) * (X.T @ diff)
        db = (2.0 / m) * np.sum(diff)
        return dw, db

    def sgd_step(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def train(self, X, y, epochs=500, batch_size=20, verbose=True):
        n = X.shape[0]
        loss_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch)
                dw, db = self.backward(X_batch, y_pred, y_batch)
                self.sgd_step(dw, db)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch + 1:3d}/{epochs}, Loss = {avg_loss:.4f}")

        return loss_history

    def predict(self, X):
        return self.forward(X).flatten()


print("  训练 PaddlePaddle 风格线性回归模型...")

pp_model = LinearRegressionNumpy(n_features=13, learning_rate=0.01)
loss_history = pp_model.train(
    X_train_minmax, y_train,
    epochs=500, batch_size=20, verbose=True
)

y_pp_pred = pp_model.predict(X_test_minmax)
pp_rmse = np.sqrt(mean_squared_error(y_test, y_pp_pred))
pp_r2 = r2_score(y_test, y_pp_pred)
pp_mae = mean_absolute_error(y_test, y_pp_pred)

print(f"\n  PaddlePaddle 风格线性回归测试集结果：")
print(f"    RMSE = {pp_rmse:.4f}   R2 = {pp_r2:.4f}   MAE = {pp_mae:.4f}")

# 损失曲线图
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(loss_history, color="red", linewidth=1.5, alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("PaddlePaddle 风格线性回归 -- 训练损失下降曲线")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_PaddlePaddle_损失曲线.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] 09_PaddlePaddle_损失曲线.png")

# PaddlePaddle 模型预测 vs 实际值
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pp_pred, alpha=0.6, color="coral",
           edgecolors="k", linewidth=0.5, s=40)
min_val = min(y_test.min(), y_pp_pred.min())
max_val = max(y_test.max(), y_pp_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="完美预测线")
ax.set_xlabel("实际房价（千美元）")
ax.set_ylabel("预测房价（千美元）")
ax.set_title(f"PaddlePaddle 风格线性回归\nRMSE = {pp_rmse:.4f}, R2 = {pp_r2:.4f}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_PaddlePaddle_预测值vs实际值.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] 10_PaddlePaddle_预测值vs实际值.png")

# ============================================================================
# 8. 综合对比总结
# ============================================================================
print("\n" + "=" * 70)
print("[8] 综合对比总结")
print("=" * 70)

all_results = test_df.copy()
all_results.loc[len(all_results)] = {
    "model": "PaddlePaddle 风格线性回归",
    "RMSE": pp_rmse,
    "R2": pp_r2,
    "MAE": pp_mae,
}

print("\n  所有模型测试集性能对比：")
print(f"  {'模型':<35s} {'RMSE':<10s} {'R2':<10s} {'MAE':<10s}")
print("  " + "-" * 65)
for _, row in all_results.iterrows():
    print(f"  {row['model']:<35s} {row['RMSE']:<8.4f}  {row['R2']:<8.4f}  {row['MAE']:<8.4f}")

print("\n" + "=" * 70)
print("复现完成！所有输出文件已保存至:", OUTPUT_DIR)
print("=" * 70)
