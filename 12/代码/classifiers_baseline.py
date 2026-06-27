import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# ===================== 配置 =====================
DATA_PATH = "../data/feature_matrix.npz"
FIG_DIR = "../figures/"
CLASS_NAMES = ['静坐', '站立', '行走', '上楼', '下楼', '跑步']

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 加载数据 =====================
dataset = np.load(DATA_PATH, allow_pickle=True)
X = dataset['X']
y = dataset['y']
subject_ids = dataset['subject_ids']

print(f"加载数据集成功：样本数={X.shape[0]}, 特征维度={X.shape[1]}")
print(f"被试数量：{len(np.unique(subject_ids))}人，类别数量：{len(np.unique(y))}类")
os.makedirs(FIG_DIR, exist_ok=True)

# ===================== LOSO 留一被试验证 =====================
def loso_cv(model, X, y, subject_ids):
    """严格留一被试交叉验证：每次留1个被试做测试，其余做训练"""
    subjects = np.unique(subject_ids)
    y_true_all = []
    y_pred_all = []
    
    for test_sub in subjects:
        train_mask = subject_ids != test_sub
        test_mask = subject_ids == test_sub
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 仅用训练集拟合标准化（严格防泄漏）
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    return np.array(y_true_all), np.array(y_pred_all)

# ===================== 基线模型定义 =====================
models = {
    "高斯朴素贝叶斯": GaussianNB(),
    "线性判别分析 LDA": LinearDiscriminantAnalysis(),
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42)
}

# ===================== 模型训练与评估 =====================
print("\n" + "="*60)
print("基线模型 LOSO 验证结果")
print("="*60)
print(f"{'模型':<18} {'总体准确率':<12}")
print("-"*40)

results = {}
for name, model in models.items():
    y_true, y_pred = loso_cv(model, X, y, subject_ids)
    acc = accuracy_score(y_true, y_pred)
    results[name] = {"y_true": y_true, "y_pred": y_pred, "acc": acc}
    print(f"{name:<18} {acc:<12.2%}")

# 打印最佳模型的详细分类报告
best_name = max(results, key=lambda k: results[k]["acc"])
print("\n" + "="*60)
print(f"最佳模型：{best_name} 详细分类报告")
print("="*60)
print(classification_report(
    results[best_name]["y_true"], 
    results[best_name]["y_pred"], 
    target_names=CLASS_NAMES,
    digits=4
))

# 绘制精度对比柱状图
plt.figure(figsize=(10, 6))
names = list(results.keys())
accs = [results[n]["acc"] for n in names]
bars = plt.bar(names, accs, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], width=0.6)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2%}', ha='center', va='bottom', fontsize=11)
plt.ylim(0, 1.0)
plt.ylabel('LOSO 准确率')
plt.title('基线分类模型精度对比')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'accuracy_comparison.png'), dpi=300)
plt.close()

# ===================== 二维决策边界可视化（彻底修复内存溢出） =====================
print("\n" + "="*60)
print("生成二维决策边界对比图")
print("="*60)

# PCA降到2维用于可视化
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

# ========== 核心修复：固定网格点数 + 裁剪离群范围 ==========
GRID_POINTS = 200  # 单轴网格点数，固定大小，内存绝对可控
# 用分位数裁剪范围，去掉极端离群点，避免坐标无限拉大
x_min, x_max = np.percentile(X_2d[:, 0], [1, 99])
y_min, y_max = np.percentile(X_2d[:, 1], [1, 99])
# 固定点数生成网格，替代步长控制
xx = np.linspace(x_min, x_max, GRID_POINTS)
yy = np.linspace(y_min, y_max, GRID_POINTS)
xx, yy = np.meshgrid(xx, yy)

# 子图：3个模型的决策边界
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9']

for i, (name, model) in enumerate(models.items()):
    ax = axes[i]
    # 在二维特征上训练模型
    model.fit(X_2d, y)
    # 预测网格
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    # 画决策边界
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.tab10)
    # 画样本散点
    for j, label in enumerate(np.unique(y)):
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=colors[j], label=CLASS_NAMES[j], alpha=0.7, s=30)
    ax.set_title(f'{name} 决策边界')
    ax.set_xlabel('主成分1')
    ax.set_ylabel('主成分2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.3)

axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'decision_boundary.png'), dpi=300)
plt.close()

# ===================== 总结 =====================
print("\n" + "="*60)
print("D6 基线分类器任务完成")
print("="*60)
print("1. 完成3种基线模型的LOSO交叉验证，输出精度对比")
print("2. 生成最佳模型的详细分类报告")
print("3. 生成PCA二维空间下的决策边界对比图")
print(f"全部图表已保存至：{FIG_DIR}")