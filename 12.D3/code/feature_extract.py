import numpy as np
import os

DATASET_PATH = "../dataset"
FEATURE_SAVE_PATH = "../feature"

# 单窗口提取时域+频域特征
def extract_single_window(window):
    feat_list = []
    # 共9轴信号
    for ch in range(window.shape[1]):
        sig = window[:, ch]
        # 时域
        mean_sig = np.mean(sig)
        var_sig = np.var(sig)
        std_sig = np.std(sig)
        max_sig = np.max(sig)
        min_sig = np.min(sig)
        rms_sig = np.sqrt(np.mean(sig ** 2))
        zero_cross = np.sum(np.diff(np.sign(sig)) != 0)
        # 频域FFT
        fft_vals = np.abs(np.fft.fft(sig))
        fft_sum = np.sum(fft_vals)
        fft_energy = np.sum(fft_vals ** 2)
        feat_list.extend([mean_sig, var_sig, std_sig, max_sig, min_sig, rms_sig, zero_cross, fft_sum, fft_energy])
    return np.array(feat_list)

def batch_extract(X):
    feat_all = []
    for sample in X:
        # 恢复 (win_len, channel)
        win_shape = (256, 18)
        win = sample.reshape(win_shape)
        feat = extract_single_window(win)
        feat_all.append(feat)
    return np.array(feat_all)

# 加载预处理后的窗口数据集
data_npz = np.load(os.path.join(DATASET_PATH, "windowed_dataset.npz"))
X_train = data_npz["X_train"]
y_train = data_npz["y_train"]
X_test = data_npz["X_test"]
y_test = data_npz["y_test"]

# 提取特征
X_train_feat = batch_extract(X_train)
X_test_feat = batch_extract(X_test)

# 保存特征
os.makedirs(FEATURE_SAVE_PATH, exist_ok=True)
np.savez(os.path.join(FEATURE_SAVE_PATH, "train_feat.npz"), feat=X_train_feat, label=y_train)
np.savez(os.path.join(FEATURE_SAVE_PATH, "test_feat.npz"), feat=X_test_feat, label=y_test)
print("特征提取完成，文件保存至 12.D3/feature")