"""ESP32-S3 HAR 训练脚本 - 生成 rf_params.json"""
import os, numpy as np, pandas as pd, json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = "./dataset"
WINDOW_SIZE = 128
STEP = 64
SEED = 42

LABEL_MAP = {"sit": 0, "stand": 1, "walk": 2, "upstairs": 3, "downstairs": 4, "run": 5}
NAMES = list(LABEL_MAP.keys())

print("Loading data...")
X, y = [], []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".csv"): continue
    parts = fname.split("_")
    if len(parts) < 3 or parts[2] not in LABEL_MAP: continue
    label = LABEL_MAP[parts[2]]
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    data = df[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values
    for i in range(0, len(data) - WINDOW_SIZE + 1, STEP):
        X.append(data[i:i+WINDOW_SIZE])
        y.append(label)
X, y = np.array(X), np.array(y)
print(f"  {len(X)} windows from {len(os.listdir(DATA_DIR))} files")

print("Extracting features...")
feats = []
for w in X:
    f = []
    for col in range(6):
        cd = w[:, col]
        f += [np.mean(cd), np.std(cd), np.max(cd), np.min(cd), np.ptp(cd)]
    feats.append(f)
Xf = np.array(feats)

Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, random_state=SEED, stratify=y)
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

print("Training Random Forest (15 trees, max_depth=10)...")
rf = RandomForestClassifier(n_estimators=15, max_depth=10, random_state=SEED)
rf.fit(Xtr, ytr)
yp = rf.predict(Xte)
print(f"  Accuracy: {accuracy_score(yte, yp):.4f}")
print(classification_report(yte, yp, target_names=NAMES))

# Save PC models
joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Export ESP32 JSON
params = {
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "n_estimators": rf.n_estimators,
    "n_classes": rf.n_classes_,
    "classes": rf.classes_.tolist(),
    "trees": []
}
for tree in rf.estimators_:
    t = tree.tree_
    params["trees"].append({
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "value": t.value.tolist()
    })

with open("rf_params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False)
print(f"\nDone! rf_params.json: {os.path.getsize('rf_params.json')} bytes")
