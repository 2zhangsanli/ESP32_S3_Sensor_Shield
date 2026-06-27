import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# ===================== 全局配置 =====================
DATA_PATH = "../data/feature_matrix.npz"
FIG_DIR = "../figures/"
CLASS_NAMES = ['静坐', '站立', '行走', '上楼', '下楼', '跑步']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs(FIG_DIR, exist_ok=True)

# ===================== 加载数据 =====================
dataset = np.load(DATA_PATH, allow_pickle=True)
X = dataset['X']
y = dataset['y']
subject_ids = dataset['subject_ids']
print(f"数据集加载完成：样本{X.shape[0]}，特征维度{X.shape[1]}，被试{len(np.unique(subject_ids))}")

# ===================== 严格LOSO留一被试交叉验证（防数据泄露） =====================
def loso_evaluate(base_model, param_grid, X, y, subject_ids):
    all_acc = []
    subject_list = np.unique(subject_ids)
    print(f"\n========== {base_model.__class__.__name__} 网格搜索调参 ==========")
    grid = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, n_jobs=-1)

    for test_sub in subject_list:
        # 划分训练/测试集
        train_mask = subject_ids != test_sub
        test_mask = subject_ids == test_sub
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # 仅用训练集标准化，杜绝泄露
        train_mean = X_train.mean(axis=0)
        train_std = X_train.std(axis=0)
        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        # 网格搜索最优超参
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"被试{test_sub} 最优参数：{grid.best_params_}")
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        all_acc.append(acc)
        print(f"被试{test_sub} 测试准确率：{acc:.2%}")

    mean_acc = np.mean(all_acc)
    print(f"{base_model.__class__.__name__} 平均LOSO准确率：{mean_acc:.2%}")
    return mean_acc, grid.best_params_

# ===================== B级要求：3种非线性/集成模型 =====================
model_configs = [
    {
        "name": "RBF核SVM",
        "model": SVC(kernel="rbf"),
        "params": {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1]}
    },
    {
        "name": "K近邻KNN",
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
    },
    {
        "name": "随机森林RF",
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100], "max_depth": [6, 10, None]}
    }
]

# ===================== 批量训练评估 =====================
result_dict = {}
print("="*70)
print("D7 非线性/集成模型 网格搜索+LOSO验证")
print("="*70)

for cfg in model_configs:
    avg_acc, best_params = loso_evaluate(cfg["model"], cfg["params"], X, y, subject_ids)
    result_dict[cfg["name"]] = {"avg_acc": avg_acc, "best_params": best_params}

# 打印汇总对比表
print("\n" + "="*70)
print("多模型综合对比汇总表")
print(f"{'模型名称':<12} {'平均LOSO准确率':<16} 最优超参数")
print("-"*70)
for name, res in result_dict.items():
    print(f"{name:<12} {res['avg_acc']:<16.2%} {res['best_params']}")

# ===================== 绘制精度对比柱状图 =====================
plt.figure(figsize=(10,6))
names = list(result_dict.keys())
accs = [result_dict[n]["avg_acc"] for n in names]
bars = plt.bar(names, accs, color=["#ff6b6b", "#4ecdc4", "#45b7d1"], width=0.6)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, h+0.005, f"{h:.2%}", ha="center")
plt.ylim(0, 1)
plt.ylabel("LOSO平均准确率")
plt.title("非线性/集成模型精度对比")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_acc_compare.png"), dpi=300)
plt.close()

# ===================== 任务完成提示 =====================
print("\n" + "="*70)
print("D7 B级任务全部完成")
print("1. 实现3类模型：RBF-SVM、KNN、随机森林")
print("2. 内置GridSearchCV网格搜索超参数调优")
print("3. 严格LOSO留一被试验证，无数据泄露")
print("4. 输出模型对比表格与精度柱状图")
print(f"图表保存路径：{FIG_DIR}/model_acc_compare.png")