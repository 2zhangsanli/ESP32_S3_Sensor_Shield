import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import (confusion_matrix, classification_report,
                               roc_curve, auc, precision_recall_curve)
from scipy import stats

# ===================== 全局配置 =====================
DATA_PATH = "../data/feature_matrix.npz"
FIG_DIR = "../figures/"
CLASS_NAMES = ['静坐', '站立', '行走', '上楼', '下楼', '跑步']
N_CLASS = len(CLASS_NAMES)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)  # 固定随机种子，S级可复现要求
os.makedirs(FIG_DIR, exist_ok=True)

# ===================== 加载数据 =====================
dataset = np.load(DATA_PATH, allow_pickle=True)
X = dataset['X']
y = dataset['y']
subject_ids = dataset['subject_ids']
print(f"数据集：样本{X.shape[0]}，特征{X.shape[1]}，被试{len(np.unique(subject_ids))}，类别{N_CLASS}")

# ===================== 标准化工具（仅训练集拟合，防泄漏） =====================
def scale_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std

# ===================== 1. K-Fold 随机交叉验证（B级必做） =====================
def kfold_evaluate(model, X, y, n_split=5):
    kf = KFold(n_splits=n_split, shuffle=True, random_state=42)
    acc_list = []
    y_true_all, y_pred_all, y_proba_all = [], [], []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = scale_train_test(X[train_idx], X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        acc_list.append(np.mean(y_pred == y_te))
        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)
    return np.array(acc_list), np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all)

# ===================== 2. LOSO 留一被试交叉验证（A级必做） =====================
def loso_evaluate(model, X, y, groups):
    logo = LeaveOneGroupOut()
    acc_list = []
    y_true_all, y_pred_all, y_proba_all = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_tr, X_te = scale_train_test(X[train_idx], X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        acc_list.append(np.mean(y_pred == y_te))
        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)
    return np.array(acc_list), np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all)

# ===================== 3. 绘制混淆矩阵 =====================
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASS))
    ax.set_yticks(range(N_CLASS))
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticklabels(CLASS_NAMES)
    # 填充数字
    for i in range(N_CLASS):
        for j in range(N_CLASS):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="black")
    plt.colorbar(im)
    plt.title("混淆矩阵（LOSO最优模型）")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return cm

# ===================== 4. 多分类ROC & PR曲线 =====================
def plot_roc_pr(y_true, y_proba, save_roc, save_pr):
    # One-vs-Rest ROC
    fig, ax = plt.subplots(figsize=(8,6))
    auc_scores = []
    for cls in range(N_CLASS):
        y_bin = (y_true == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, cls])
        auc_val = auc(fpr, tpr)
        auc_scores.append(auc_val)
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[cls]} AUC={auc_val:.3f}")
    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("多分类 ROC 曲线(One vs Rest)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_roc, dpi=300)
    plt.close()

    # PR曲线
    fig, ax = plt.subplots(figsize=(8,6))
    for cls in range(N_CLASS):
        y_bin = (y_true == cls).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, cls])
        ax.plot(rec, prec, label=CLASS_NAMES[cls])
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title("多分类 P-R 曲线")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_pr, dpi=300)
    plt.close()
    return np.mean(auc_scores)

# ===================== 5. 置信区间计算（A级） =====================
def bootstrap_ci(acc_array, n_boot=10000, alpha=0.05):
    boot_means = []
    n = len(acc_array)
    for _ in range(n_boot):
        sample = np.random.choice(acc_array, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    low = int(n_boot * alpha / 2)
    high = int(n_boot * (1 - alpha / 2))
    return boot_means[low], boot_means[high]

# ===================== 6. 模型独立样本t检验【修复：适配不等长数组】 =====================
def paired_ttest(acc1, acc2):
    # 改用独立样本t检验，两组长度不一致也能计算
    t_stat, p_val = stats.ttest_ind(acc1, acc2, equal_var=False)
    return t_stat, p_val

# ===================== 主执行流程 =====================
if __name__ == "__main__":
    # 选用D7最优模型：随机森林
    best_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    print("="*65)
    print("【1】5折随机K-Fold交叉验证（B级）")
    print("="*65)
    kf_accs, kf_ytrue, kf_ypred, kf_yproba = kfold_evaluate(best_model, X, y, n_split=5)
    print(f"K-Fold 平均准确率：{np.mean(kf_accs):.2%}，各折精度：{np.round(kf_accs,3)}")

    print("\n" + "="*65)
    print("【2】LOSO 留一被试交叉验证（A级）")
    print("="*65)
    loso_accs, loso_ytrue, loso_ypred, loso_yproba = loso_evaluate(best_model, X, y, groups=subject_ids)
    loso_mean_acc = np.mean(loso_accs)
    print(f"LOSO 平均准确率：{loso_mean_acc:.2%}，各被试精度：{np.round(loso_accs,3)}")
    print(f"乐观偏差（Kfold - LOSO）：{np.mean(kf_accs)-loso_mean_acc:.2%}")

    # B级产出：混淆矩阵、分类报告、ROC、PR
    print("\n【3】生成混淆矩阵 & 分类报告")
    cm = plot_confusion_matrix(loso_ytrue, loso_ypred, os.path.join(FIG_DIR, "confusion_matrix.png"))
    print(classification_report(loso_ytrue, loso_ypred, target_names=CLASS_NAMES, digits=4))

    print("\n【4】绘制ROC、PR曲线")
    mean_auc = plot_roc_pr(loso_ytrue, loso_yproba,
                           os.path.join(FIG_DIR, "roc_curve.png"),
                           os.path.join(FIG_DIR, "pr_curve.png"))
    print(f"各类别平均AUC：{mean_auc:.3f}")

    # A级：自举置信区间
    print("\n" + "="*65)
    print("【5】LOSO准确率95%置信区间（自举法）")
    print("="*65)
    ci_low, ci_high = bootstrap_ci(loso_accs)
    print(f"平均准确率 {loso_mean_acc:.2%} 的95%置信区间：[{ci_low:.2%}, {ci_high:.2%}]")

    # S级：模型显著性检验（修复后可正常运行，支持长度不同数组）
    print("\n【6】模型独立样本t检验（S级，对比Kfold与LOSO精度差异显著性）")
    t, p = paired_ttest(kf_accs, loso_accs)
    print(f"t统计量：{t:.4f}，p值：{p:.4f}")
    if p < 0.05:
        print("结论：两种评估方式的准确率存在统计显著差异（随机划分高估泛化能力）")
    else:
        print("结论：无显著差异")

    # 绘制Kfold vs LOSO精度对比柱状图
    plt.figure(figsize=(8,5))
    eval_types = ["5折K-Fold随机划分", "LOSO留一被试划分"]
    avg_accs = [np.mean(kf_accs), loso_mean_acc]
    bars = plt.bar(eval_types, avg_accs, color=["#ff6b6b", "#45b7d1"], width=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, h+0.002, f"{h:.2%}", ha="center")
    plt.ylim(0, 1)
    plt.ylabel("平均准确率")
    plt.title("随机K-Fold vs LOSO留一被试 泛化精度对比")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "loso_vs_kfold_acc.png"), dpi=300)
    plt.close()

    print("\n" + "="*65)
    print("D8 模型评估全部完成，产出文件：")
    print(f"1. 混淆矩阵：{FIG_DIR}/confusion_matrix.png")
    print(f"2. ROC曲线：{FIG_DIR}/roc_curve.png")
    print(f"3. PR曲线：{FIG_DIR}/pr_curve.png")
    print(f"4. 评估协议对比图：{FIG_DIR}/loso_vs_kfold_acc.png")
    print("="*65)