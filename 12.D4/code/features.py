import numpy as np
import pandas as pd
import os
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

# ===================== 消融实验开关（S层可配置流水线） =====================
USE_TIME_FEATURE = True    # 时域特征
USE_FREQ_FEATURE = True    # 频域特征
USE_FUSION_FEATURE = True  # 跨传感器融合特征

# 读取D3数据集（绝对路径，不会找不到文件）
DATASET_PATH = r"C:\Users\30767\Desktop\HAR_人体活动识别_小组12\12.D3\dataset\windowed_dataset.npz"
# 输出路径：向上一级，保存到12.D4外层feature文件夹（规范目录）
SAVE_CSV_PATH = "../feature/feature_matrix.csv"
SAVE_NPZ_PATH = "../feature/all_features.npz"

FS = 100        # 采样率和D3统一
WIN_LEN = 256   # 单窗口点数
CHANNEL_NUM = 18# 通道总数
ACT_LABEL = {0:"sit",1:"stand",2:"walk",3:"jog",4:"upstairs",5:"downstairs"}

# ===================== 单通道时域特征（单通道12个，满足B层要求） =====================
def extract_time_series(sig):
    N = len(sig)
    mean_sig = np.mean(sig)
    var_sig = np.var(sig)
    std_sig = np.std(sig)
    max_sig = np.max(sig)
    min_sig = np.min(sig)
    peak2peak = max_sig - min_sig
    median_sig = np.median(sig)
    rms_sig = np.sqrt(np.mean(np.square(sig)))
    skewness = skew(sig)
    kurtosis_val = kurtosis(sig)
    zero_cross = np.sum(np.diff(np.sign(sig)) != 0) / N
    sma = np.sum(np.abs(sig)) / N
    return [
        mean_sig, var_sig, std_sig, max_sig, min_sig, peak2peak,
        median_sig, rms_sig, skewness, kurtosis_val, zero_cross, sma
    ]

# ===================== 单通道频域特征 =====================
def extract_freq_series(sig, fs):
    N = len(sig)
    fft_vals = np.abs(rfft(sig))
    freqs = rfftfreq(N, d=1/fs)[1:]
    fft_mag = fft_vals[1:]
    peak_freq = freqs[np.argmax(fft_mag)]
    psd = fft_mag ** 2
    total_psd = np.sum(psd) + 1e-10
    spectral_centroid = np.sum(freqs * fft_mag) / total_psd
    psd_norm = psd / total_psd
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    band_mask = (freqs >= 0.5) & (freqs <= 3)
    band_energy_ratio = np.sum(psd[band_mask]) / total_psd
    peak_count = np.sum(fft_mag > np.mean(fft_mag))
    bandwidth = freqs[-1] - freqs[0]
    return [peak_freq, spectral_centroid, spectral_entropy, band_energy_ratio, peak_count, bandwidth]

# ===================== A层 跨传感器融合特征 =====================
def extract_fusion_feature(window):
    ax, ay, az = window[:,0], window[:,1], window[:,2]
    gx, gy, gz = window[:,3], window[:,4], window[:,5]
    mx, my, mz = window[:,6], window[:,7], window[:,8]
    acc_mag_mean = np.mean(np.sqrt(ax**2 + ay**2 + az**2))
    gyr_mag_std = np.std(np.sqrt(gx**2 + gy**2 + gz**2))
    corr_ax_gx = np.corrcoef(ax, gx)[0,1]
    mag_mag_var = np.var(np.sqrt(mx**2 + my**2 + mz**2))
    return [acc_mag_mean, gyr_mag_std, corr_ax_gx, mag_mag_var]

# ===================== 单窗口特征提取（修复一维扁平化数组报错） =====================
def get_window_feature(flat_window):
    # D3输出是一维数组，还原为 (256, 18) 时序窗口
    window = flat_window.reshape(WIN_LEN, CHANNEL_NUM)
    total_feat = []
    for ch in range(window.shape[1]):
        sig = window[:, ch]
        if USE_TIME_FEATURE:
            total_feat.extend(extract_time_series(sig))
        if USE_FREQ_FEATURE:
            total_feat.extend(extract_freq_series(sig, FS))
    if USE_FUSION_FEATURE:
        total_feat.extend(extract_fusion_feature(window))
    return np.array(total_feat)

if __name__ == "__main__":
    # 加载D3预处理后的窗口数据集
    data_npz = np.load(DATASET_PATH)
    X_train = data_npz["X_train"]
    y_train = data_npz["y_train"]
    X_test = data_npz["X_test"]
    y_test = data_npz["y_test"]

    print("正在提取训练集特征...")
    train_feat = np.array([get_window_feature(w) for w in X_train])
    print("正在提取测试集特征...")
    test_feat = np.array([get_window_feature(w) for w in X_test])

    # 合并训练+测试全部样本
    all_X = np.concatenate([train_feat, test_feat])
    all_y = np.concatenate([y_train, y_test])
    text_label = [ACT_LABEL[int(i)] for i in all_y]

    # 生成表头列名【已修复：频域使用freq_names，不再混用fusion_names】
    col_names = []
    time_names = ["均值","方差","标准差","最大值","最小值","峰峰值","中位数","RMS","偏度","峰度","过零率","SMA"]
    freq_names = ["主频","频谱质心","谱熵","0.5-3Hz能量比","谱峰数","带宽"]
    fusion_names = ["加速度合模均值","陀螺合模标准差","ax-gx相关系数","磁场模方差"]
    ch_cnt = CHANNEL_NUM
    for ch in range(ch_cnt):
        if USE_TIME_FEATURE:
            col_names += [f"通道{ch}_{n}" for n in time_names]
        if USE_FREQ_FEATURE:
            col_names += [f"通道{ch}_{n}" for n in freq_names] # 修复此处变量名
    if USE_FUSION_FEATURE:
        col_names += [f"融合_{n}" for n in fusion_names]
    col_names += ["标签数值","活动类别"]

    # 创建12.D4外层feature文件夹，文件输出到规范位置
    os.makedirs("../feature", exist_ok=True)
    df = pd.DataFrame(all_X, columns=col_names[:-2])
    df["标签数值"] = all_y
    df["活动类别"] = text_label
    df.to_csv(SAVE_CSV_PATH, index=False, encoding="utf-8-sig")
    np.savez(SAVE_NPZ_PATH,
             train_feat=train_feat, train_label=y_train,
             test_feat=test_feat, test_label=y_test)
    print(f"特征矩阵保存成功：{SAVE_CSV_PATH}")
    print(f"单个窗口特征总维度：{train_feat.shape[1]}")