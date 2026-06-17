import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ===================== 配置参数（2.56s窗口，50%重叠，假设采样率100Hz） =====================
FS = 100               # 采样率 100Hz
WINDOW_SEC = 2.56      # 窗口时长2.56s
WINDOW_SIZE = int(WINDOW_SEC * FS)  # 单窗口点数 256
STEP = int(WINDOW_SIZE * 0.5)      # 重叠50%，步长128
LOWPASS_CUT = 10       # 低通滤波截止10Hz（抗混叠）
HIGHPASS_CUT = 0.3     # 高通分离运动/重力分量
RAW_DATA_PATH = r"C:\Users\30767\Desktop\HAR_人体活动识别_小组12\12.D2\data\calibrated"
SAVE_DATASET_PATH = "../dataset"
FIG_SAVE_PATH = "../screenshots"
ACT_LABEL = {
    "sit": 0,
    "stand": 1,
    "walk": 2,
    "jog": 3,
    "upstairs": 4,
    "downstairs": 5
}

# ===================== 1. 滤波工具：巴特沃斯高低通 =====================
def butter_lowpass(signal, cut, fs, order=4):
    nyq = 0.5 * fs
    normal_cut = cut / nyq
    b, a = butter(order, normal_cut, btype="low", analog=False)
    return filtfilt(b, a, signal)

def butter_highpass(signal, cut, fs, order=4):
    nyq = 0.5 * fs
    normal_cut = cut / nyq
    b, a = butter(order, normal_cut, btype="high", analog=False)
    return filtfilt(b, a, signal)

# ===================== 2. 异常值剔除（3σ原则） =====================
def remove_outlier(sig):
    mu = np.mean(sig)
    std = np.std(sig)
    sig[np.abs(sig - mu) > 3 * std] = mu
    return sig

# ===================== 3. 单文件完整预处理流水线 =====================
def signal_preprocess(raw_df):
    raw = raw_df[["ax","ay","az","gx","gy","gz","mx","my","mz"]].values
    clean_all = []
    for ch in range(9):
        s = raw[:, ch]
        s = remove_outlier(s)                # 异常值剔除
        s_low = butter_lowpass(s, LOWPASS_CUT, FS)
        s_high = butter_highpass(s, HIGHPASS_CUT, FS)
        clean_all.append(np.column_stack([s_low, s_high]))
    return np.concatenate(clean_all, axis=1)

# ===================== 4. 滑动窗口 + 纯净窗口标签策略 =====================
def sliding_window_labeled(data, win_len, step, label, subj_id):
    win_list, lab_list, subj_list = [], [], []
    for i in range(0, len(data) - win_len + 1, step):
        win = data[i:i+win_len]
        win_list.append(win)
        lab_list.append(label)
        subj_list.append(subj_id)
    return np.array(win_list), np.array(lab_list), np.array(subj_list)

# ===================== 5. 批量读取所有CSV =====================
all_win, all_lab, all_subj = [], [], []
for fname in os.listdir(RAW_DATA_PATH):
    if not fname.endswith(".csv"):
        continue
    subj_str, act_str = fname.replace(".csv", "").split("_", 1)
    subj_id = int(subj_str.replace("sub", ""))
    lab = ACT_LABEL[act_str]
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, fname))
    sig_clean = signal_preprocess(df)
    win_data, win_lab, win_sub = sliding_window_labeled(sig_clean, WINDOW_SIZE, STEP, lab, subj_id)
    all_win.extend(win_data)
    all_lab.extend(win_lab)
    all_subj.extend(win_sub)

X_all = np.array(all_win)
y_all = np.array(all_lab)
subj_all = np.array(all_subj)

# ===================== 6. 分层划分训练/测试，Pipeline防止数据泄漏 =====================
X_train, X_test, y_train, y_test, subj_train, subj_test = train_test_split(
    X_all, y_all, subj_all, test_size=0.3, random_state=42, stratify=y_all
)
# Pipeline：仅在训练集拟合标准化，测试集仅transform
pipe = Pipeline([("scaler", StandardScaler())])
# 重塑：(样本,窗口,通道) → (样本,窗口*通道) 适配sklearn
n_sample_train, n_win, n_feat = X_train.shape
X_train_flat = X_train.reshape(-1, n_win * n_feat)
X_train_scaled = pipe.fit_transform(X_train_flat)
X_test_flat = X_test.reshape(-1, n_win * n_feat)
X_test_scaled = pipe.transform(X_test_flat)

# ===================== 7. 保存窗口数据集（带被试ID、标签） =====================
os.makedirs(SAVE_DATASET_PATH, exist_ok=True)
np.savez(os.path.join(SAVE_DATASET_PATH, "windowed_dataset.npz"),
         X_train=X_train_scaled, y_train=y_train, subj_train=subj_train,
         X_test=X_test_scaled, y_test=y_test, subj_test=subj_test)
print("窗口化数据集已保存至 dataset/windowed_dataset.npz")

# ===================== 8. 绘制预处理前后波形对比图（当日交付物） =====================
os.makedirs(FIG_SAVE_PATH, exist_ok=True)
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 取第一段加速度x轴做对比
raw_ax = pd.read_csv(os.path.join(RAW_DATA_PATH, "sub1_sit.csv"))["ax"].values[:500]
clean_ax = signal_preprocess(pd.read_csv(os.path.join(RAW_DATA_PATH, "sub1_sit.csv")))[:500, 0]
plt.figure(figsize=(10,4), dpi=150)
plt.subplot(1,2,1)
plt.plot(raw_ax, label="原始未预处理信号")
plt.title("滤波前原始加速度ax")
plt.legend()
plt.subplot(1,2,2)
plt.plot(clean_ax, label="高低通滤波+去噪后信号", color="orange")
plt.title("预处理后加速度ax")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_SAVE_PATH, "signal_compare.png"), dpi=300)
plt.show()
print("预处理前后波形对比图已保存至 screenshots/signal_compare.png")