import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 解决中文方框乱码
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_PATH = "../feature"
SAVE_IMG_PATH = "../screenshots"
ACT_NAME = ["sit", "stand", "walk", "jog", "upstairs", "downstairs"]

# 加载特征
train_npz = np.load(os.path.join(FEATURE_PATH, "train_feat.npz"))
X_train, y_train = train_npz["feat"], train_npz["label"]
test_npz = np.load(os.path.join(FEATURE_PATH, "test_feat.npz"))
X_test, y_test = test_npz["feat"], test_npz["label"]

# 训练随机森林
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"测试集总体识别准确率：{acc:.4f}")
print("\n六类活动分类报告：")
print(classification_report(y_test, y_pred, target_names=ACT_NAME))

# 混淆矩阵绘图保存
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6), dpi=150)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(range(6), ACT_NAME, rotation=30, fontsize=8)
plt.yticks(range(6), ACT_NAME, fontsize=8)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("人体活动识别混淆矩阵")

os.makedirs(SAVE_IMG_PATH, exist_ok=True)
plt.savefig(os.path.join(SAVE_IMG_PATH, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()
print("\n混淆矩阵图片已保存至 12.D3/screenshots")