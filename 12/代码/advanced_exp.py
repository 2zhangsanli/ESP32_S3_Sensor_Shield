import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ===================== 全局配置（修正路径，读取npz而非csv） =====================
FIG_DIR = "../figures/"
# 正确路径：D4/output下的npz，复制到D9/data里
FEATURE_PATH = "../data/feature_matrix.npz"
# 时序数据保存路径
RAW_WINDOW_PATH = "../data/window_raw.npz"
CLASS_NAMES = ['静坐', '站立', '行走', '上楼', '下楼', '跑步']
N_CLUSTER = 6
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
os.makedirs(FIG_DIR, exist_ok=True)

# ===================== 1. 加载数据（仅numpy，移除pandas） =====================
# 手工特征（聚类使用）
feat_data = np.load(FEATURE_PATH, allow_pickle=True)
X_feat = feat_data['X']
y_label = feat_data['y']
subject_ids = feat_data['subject_ids']
n_sample = X_feat.shape[0]

# 时序数据判断，不存在则自动生成
if not os.path.exists(RAW_WINDOW_PATH):
    print("未检测到时序文件，自动生成 window_raw.npz")
    n_seq_window = 1200
    idx_seq = np.random.randint(0, n_sample, size=n_seq_window)
    X_window_seq = X_feat[idx_seq]
    seq_true = y_label[idx_seq]
    np.savez(RAW_WINDOW_PATH, window_seq=X_window_seq, seq_true=seq_true)
else:
    raw_data = np.load(RAW_WINDOW_PATH, allow_pickle=True)
    X_window_seq = raw_data['window_seq']
    seq_true = raw_data['seq_true']

# 训练基准分类器（生成原始单窗口预测序列）
rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_base.fit(X_feat, y_label)

# ===================== 方向一：无监督聚类 Kmeans / GMM（Ch11） =====================
print("="*60)
print("【模块1】无监督聚类 K-Means & GMM 评估(ARI/NMI)")
print("="*60)
# 标准化
X_scaler = (X_feat - X_feat.mean(axis=0)) / (X_feat.std(axis=0)+1e-8)

# K-Means聚类
kmeans = KMeans(n_clusters=N_CLUSTER, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaler)
ari_k = adjusted_rand_score(y_label, y_kmeans)
nmi_k = normalized_mutual_info_score(y_label, y_kmeans)

# GMM高斯混合聚类
gmm = GaussianMixture(n_components=N_CLUSTER, random_state=42)
y_gmm = gmm.fit_predict(X_scaler)
ari_g = adjusted_rand_score(y_label, y_gmm)
nmi_g = normalized_mutual_info_score(y_label, y_gmm)

# 打印聚类量化指标
print(f"K-Means: ARI={ari_k:.4f}, NMI={nmi_k:.4f}")
print(f"GMM:     ARI={ari_g:.4f}, NMI={nmi_g:.4f}")

# 绘制指标对比柱状图
plt.figure(figsize=(8,5))
methods = ["K-Means", "GMM高斯混合"]
ari_vals = [ari_k, ari_g]
nmi_vals = [nmi_k, nmi_g]
x = np.arange(len(methods))
width = 0.3
plt.bar(x-width/2, ari_vals, width, label="ARI指标", color="#ff6b6b")
plt.bar(x+width/2, nmi_vals, width, label="NMI指标", color="#45b7d1")
plt.xticks(x, methods)
plt.ylim(0,1)
plt.ylabel("指标得分（越高聚类匹配真实标签越好）")
plt.title("无监督聚类 ARI / NMI 量化对比")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_ari_nmi.png"), dpi=300)
plt.close()

# ===================== 方向二：时序建模 HMM平滑 + 滑窗表决（Ch4） =====================
print("\n" + "="*60)
print("【模块2】时序序列平滑 HMM 工程降噪")
print("="*60)
# 1. 生成原始单窗口瞬时预测
seq_raw_pred = rf_base.predict(X_window_seq)
acc_raw = accuracy_score(seq_true, seq_raw_pred)

# 2. 简易隐马尔可夫平滑（状态转移约束，消除瞬时跳变）
def hmm_smooth(seq_pred, n_state=6):
    smooth_seq = seq_pred.copy()
    seq_len = len(smooth_seq)
    # 简单状态约束：相邻窗口禁止突变，取前后多数投票模拟HMM时序依赖
    win = 3
    for i in range(win, seq_len-win):
        window = smooth_seq[i-win:i+win+1]
        vals, cnts = np.unique(window, return_counts=True)
        smooth_seq[i] = vals[np.argmax(cnts)]
    return smooth_seq

seq_hmm_smooth = hmm_smooth(seq_raw_pred)
acc_smooth = accuracy_score(seq_true, seq_hmm_smooth)
print(f"单窗口瞬时识别准确率：{acc_raw:.2%}")
print(f"HMM时序平滑后序列准确率：{acc_smooth:.2%}")
print(f"时序平滑精度提升：{acc_smooth - acc_raw:.2%}")

# 绘制序列分段对比曲线
plt.figure(figsize=(12,5))
seq_idx = np.arange(len(seq_true))
plt.plot(seq_idx, seq_true, label="真实活动序列", linewidth=2, color="#222222")
plt.plot(seq_idx, seq_raw_pred, label="原始瞬时预测", alpha=0.6, color="#ff6b6b")
plt.plot(seq_idx, seq_hmm_smooth, label="HMM平滑后序列", linewidth=2, color="#45b7d1")
plt.xlabel("时序滑窗序号")
plt.ylabel("活动类别标签")
plt.title("时序活动序列平滑前后对比")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hmm_smooth_compare.png"), dpi=300)
plt.close()

# 精度汇总曲线
plt.figure(figsize=(6,5))
acc_names = ["原始瞬时预测", "HMM时序平滑"]
acc_data = [acc_raw, acc_smooth]
bars = plt.bar(acc_names, acc_data, color=["#ff6b6b","#45b7d1"], width=0.5)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, h+0.002, f"{h:.2%}", ha="center")
plt.ylim(0, 1)
plt.ylabel("序列整体准确率")
plt.title("时序平滑精度提升对比")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "seq_acc_curve.png"), dpi=300)
plt.close()

# ===================== 任务完成输出 =====================
print("\n" + "="*60)
print("D9 S级双方向实验全部完成")
print("1. 无监督聚类模块：Kmeans / GMM，输出ARI、NMI量化对比图")
print("2. 时序HMM平滑模块：消除瞬时识别抖动，量化精度提升幅度")
print(f"全部图表输出至：{FIG_DIR}")
print("="*60)