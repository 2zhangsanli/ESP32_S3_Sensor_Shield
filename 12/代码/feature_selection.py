import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ===================== 配置 =====================
DATA_PATH = "../data/feature_matrix.npz"
TOP_K = 15  # 选取Top-K特征
FIG_DIR = "../figures/"

# 类别名称
CLASS_NAMES = ['静坐', '站立', '行走', '上楼', '下楼', '跑步']

# ===================== 加载数据 =====================
dataset = np.load(DATA_PATH, allow_pickle=True)
X = dataset['X']
y = dataset['y']
feature_names = dataset['feature_names']

print(f"加载数据集成功：样本数={X.shape[0]}, 特征维度={X.shape[1]}, 类别数={len(np.unique(y))}")
os.makedirs(FIG_DIR, exist_ok=True)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 一、特征选择（3种方法，B级达标） =====================
print("\n" + "="*60)
print("【1】特征选择方法对比")
print("="*60)

# 方法1：随机森林特征重要性（Embedded 嵌入法）
print("\n方法1：随机森林特征重要性（嵌入法）")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)
importances_rf = rf.feature_importances_
# 排序取Top-K
idx_rf = np.argsort(importances_rf)[::-1][:TOP_K]
top_rf_names = feature_names[idx_rf]
top_rf_scores = importances_rf[idx_rf]
for i in range(TOP_K):
    print(f"  Top{i+1:2d}: {top_rf_names[i]:25s} 重要性: {top_rf_scores[i]:.4f}")

# 方法2：互信息（Filter 过滤法）
print("\n方法2：互信息（过滤法）")
mi_scores = mutual_info_classif(X, y, random_state=42)
idx_mi = np.argsort(mi_scores)[::-1][:TOP_K]
top_mi_names = feature_names[idx_mi]
top_mi_scores = mi_scores[idx_mi]
for i in range(TOP_K):
    print(f"  Top{i+1:2d}: {top_mi_names[i]:25s} 互信息: {top_mi_scores[i]:.4f}")

# 方法3：ANOVA F值（Filter 过滤法）
print("\n方法3：ANOVA F检验（过滤法）")
f_scores, _ = f_classif(X, y)
idx_f = np.argsort(f_scores)[::-1][:TOP_K]
top_f_names = feature_names[idx_f]
top_f_scores = f_scores[idx_f]
for i in range(TOP_K):
    print(f"  Top{i+1:2d}: {top_f_names[i]:25s} F值: {top_f_scores[i]:.2f}")

# 绘制特征重要性图（以随机森林为例）
plt.figure(figsize=(12, 6))
plt.barh(range(TOP_K), top_rf_scores[::-1], color='#45b7d1')
plt.yticks(range(TOP_K), top_rf_names[::-1], fontsize=10)
plt.xlabel('特征重要性得分')
plt.title(f'Top {TOP_K} 特征重要性排名（随机森林）')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), dpi=300)
plt.close()
print(f"\n特征重要性图已保存：{FIG_DIR}feature_importance.png")

# ===================== 二、PCA 降维分析 =====================
print("\n" + "="*60)
print("【2】PCA 主成分分析")
print("="*60)

pca = PCA()
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 找95%方差对应的主成分数
n_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"保留95%方差需要的主成分数：{n_95}")
print(f"前10个主成分方差解释率：{cumulative_variance[9]:.2%}")

# 绘制累计方差曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 
         marker='o', markersize=3, color='#ff6b6b', linewidth=2)
plt.axhline(y=0.95, color='gray', linestyle='--', label='95% 方差线')
plt.axvline(x=n_95, color='gray', linestyle='--')
plt.xlabel('主成分数量')
plt.ylabel('累计方差解释率')
plt.title('PCA 累计方差解释率曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'pca_variance.png'), dpi=300)
plt.close()
print(f"PCA方差图已保存：{FIG_DIR}pca_variance.png")

# ===================== 三、t-SNE 二维可视化 =====================
print("\n" + "="*60)
print("【3】t-SNE 二维可视化")
print("="*60)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X)

# 绘制散点图
plt.figure(figsize=(10, 8))
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9']
for i, label in enumerate(np.unique(y)):
    mask = y == label
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                c=colors[i], label=CLASS_NAMES[int(label)], alpha=0.7, s=40)

plt.xlabel('t-SNE 维度1')
plt.ylabel('t-SNE 维度2')
plt.title('高维特征 t-SNE 二维可视化')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'tsne_visualization.png'), dpi=300)
plt.close()
print(f"t-SNE可视化图已保存：{FIG_DIR}tsne_visualization.png")

# ===================== 结果总结 =====================
print("\n" + "="*60)
print("D5 特征选择与降维任务完成")
print("="*60)
print("1. 完成3种特征选择方法，输出Top-K特征排名")
print("2. 完成PCA降维分析，生成累计方差曲线")
print("3. 完成t-SNE二维可视化，直观展示特征可分性")
print("全部图表已保存至 figures 目录")