import numpy as np
import os
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
import pandas as pd

# ===================== 可配置特征流水线（S级消融开关） =====================
FEATURE_CONFIG = {
    "time_domain": True,       # 时域特征组（B级必开，单轴12维）
    "frequency_domain": True,  # 频域特征组（B级必开，单轴7维）
    "cross_axis": False,       # 跨轴相关系数特征（A级）
    "magnitude": False         # 向量模长统计特征（A级）
}

FS = 50  # 采样率
AXIS_NAMES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
N_AXES = len(AXIS_NAMES)

# 路径
DATA_PATH = "../data/windowed_dataset.npz"
OUTPUT_NPZ = "../output/feature_matrix.npz"
OUTPUT_CSV = "../output/feature_matrix.csv"

# ===================== 单轴时域特征提取 =====================
def extract_time_domain(signal):
    """单轴时域特征：12维"""
    feat = []
    # 1. 均值
    feat.append(np.mean(signal))
    # 2. 标准差
    feat.append(np.std(signal))
    # 3. 均方根 RMS
    feat.append(np.sqrt(np.mean(signal**2)))
    # 4. 峰峰值
    feat.append(np.max(signal) - np.min(signal))
    # 5. 最大值
    feat.append(np.max(signal))
    # 6. 最小值
    feat.append(np.min(signal))
    # 7. 中位数
    feat.append(np.median(signal))
    # 8. 偏度
    feat.append(skew(signal))
    # 9. 峰度
    feat.append(kurtosis(signal))
    # 10. 过零率（去直流后）
    signal_centered = signal - np.mean(signal)
    zero_crossings = np.sum(np.diff(np.sign(signal_centered)) != 0)
    feat.append(zero_crossings / len(signal))
    # 11. 信号幅值面积 SMA
    feat.append(np.sum(np.abs(signal)) / FS)
    # 12. 四分位距 IQR
    q75, q25 = np.percentile(signal, [75, 25])
    feat.append(q75 - q25)
    return np.array(feat)

TIME_FEATURE_NAMES = [
    'mean', 'std', 'rms', 'peak2peak', 'max', 'min',
    'median', 'skew', 'kurtosis', 'zero_cross_rate', 'sma', 'iqr'
]

# ===================== 单轴频域特征提取 =====================
def extract_frequency_domain(signal, fs=FS):
    """单轴频域特征：7维"""
    feat = []
    n = len(signal)
    # FFT 计算单边功率谱
    freqs = np.fft.rfftfreq(n, 1/fs)
    fft_vals = np.fft.rfft(signal)
    psd = np.abs(fft_vals) ** 2 / (n * fs)
    psd_norm = psd / np.sum(psd)  # 归一化功率谱

    # 1. 主频
    dominant_idx = np.argmax(psd)
    feat.append(freqs[dominant_idx])
    # 2. 频谱质心
    feat.append(np.sum(freqs * psd_norm))
    # 3. 谱熵
    psd_norm_pos = psd_norm[psd_norm > 0]
    feat.append(-np.sum(psd_norm_pos * np.log2(psd_norm_pos)))
    # 4. 谱峰个数
    peaks, _ = find_peaks(psd, height=np.max(psd)*0.1)
    feat.append(len(peaks))
    # 5. 低频能量比（0-5Hz）
    low_band_mask = freqs <= 5
    feat.append(np.sum(psd[low_band_mask]) / np.sum(psd))
    # 6. 频谱标准差
    feat.append(np.std(psd))
    # 7. 谱峭度
    feat.append(kurtosis(psd))
    return np.array(feat)

FREQ_FEATURE_NAMES = [
    'dominant_freq', 'spectral_centroid', 'spectral_entropy',
    'peak_count', 'low_energy_ratio', 'spectral_std', 'spectral_kurtosis'
]

# ===================== 跨轴融合特征（A级） =====================
def extract_cross_axis(signal_3axis):
    """三轴两两相关系数：3维"""
    corr_matrix = np.corrcoef(signal_3axis.T)
    # 取上三角相关系数
    feat = [
        corr_matrix[0, 1],  # x-y相关
        corr_matrix[0, 2],  # x-z相关
        corr_matrix[1, 2]   # y-z相关
    ]
    return np.array(feat)

def extract_magnitude(signal_3axis):
    """向量模长的时域统计：12维（与时域特征对应）"""
    mag = np.sqrt(np.sum(signal_3axis ** 2, axis=1))
    return extract_time_domain(mag)

# ===================== 主特征提取流水线 =====================
def extract_all_features(window):
    """单个窗口提取全部特征，返回特征向量与名称列表"""
    feats = []
    names = []

    # 1. 单轴时域特征
    if FEATURE_CONFIG["time_domain"]:
        for i, axis_name in enumerate(AXIS_NAMES):
            axis_feat = extract_time_domain(window[:, i])
            feats.extend(axis_feat)
            names.extend([f"{axis_name}_td_{n}" for n in TIME_FEATURE_NAMES])

    # 2. 单轴频域特征
    if FEATURE_CONFIG["frequency_domain"]:
        for i, axis_name in enumerate(AXIS_NAMES):
            axis_feat = extract_frequency_domain(window[:, i])
            feats.extend(axis_feat)
            names.extend([f"{axis_name}_fd_{n}" for n in FREQ_FEATURE_NAMES])

    # 3. 跨轴相关特征（A级）
    if FEATURE_CONFIG["cross_axis"]:
        acc_corr = extract_cross_axis(window[:, 0:3])
        gyro_corr = extract_cross_axis(window[:, 3:6])
        feats.extend(acc_corr)
        feats.extend(gyro_corr)
        names.extend(['acc_corr_xy', 'acc_corr_xz', 'acc_corr_yz'])
        names.extend(['gyro_corr_xy', 'gyro_corr_xz', 'gyro_corr_yz'])

    # 4. 向量模长特征（A级）
    if FEATURE_CONFIG["magnitude"]:
        acc_mag_feat = extract_magnitude(window[:, 0:3])
        gyro_mag_feat = extract_magnitude(window[:, 3:6])
        feats.extend(acc_mag_feat)
        feats.extend(gyro_mag_feat)
        names.extend([f"acc_mag_td_{n}" for n in TIME_FEATURE_NAMES])
        names.extend([f"gyro_mag_td_{n}" for n in TIME_FEATURE_NAMES])

    return np.array(feats), names

# ===================== 主程序 =====================
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)

    # 加载窗口化数据集
    dataset = np.load(DATA_PATH, allow_pickle=True)
    X_windows = dataset['X']  # (324, 128, 6)
    y = dataset['y']
    subject_ids = dataset['subject_ids']
    print(f"加载数据集成功：共 {len(X_windows)} 个窗口样本")

    # 逐个窗口提取特征
    all_features = []
    feature_names = None
    for i, window in enumerate(X_windows):
        feat_vec, names = extract_all_features(window)
        all_features.append(feat_vec)
        if feature_names is None:
            feature_names = names

    X_feat = np.array(all_features)
    print(f"特征提取完成，特征矩阵维度：{X_feat.shape}")
    print(f"单样本特征维度：{len(feature_names)} 维")

    # 统计各特征组维度
    dim_td = N_AXES * len(TIME_FEATURE_NAMES) if FEATURE_CONFIG["time_domain"] else 0
    dim_fd = N_AXES * len(FREQ_FEATURE_NAMES) if FEATURE_CONFIG["frequency_domain"] else 0
    dim_cross = 6 if FEATURE_CONFIG["cross_axis"] else 0
    dim_mag = 2 * len(TIME_FEATURE_NAMES) if FEATURE_CONFIG["magnitude"] else 0
    print(f"  时域特征：{dim_td} 维")
    print(f"  频域特征：{dim_fd} 维")
    print(f"  跨轴相关：{dim_cross} 维")
    print(f"  模长特征：{dim_mag} 维")

    # 保存为 npz
    np.savez(
        OUTPUT_NPZ,
        X=X_feat,
        y=y,
        subject_ids=subject_ids,
        feature_names=np.array(feature_names)
    )

    # 保存为 csv（带列名，方便查看）
    df = pd.DataFrame(X_feat, columns=feature_names)
    df['label'] = y
    df['subject_id'] = subject_ids
    df.to_csv(OUTPUT_CSV, index=False)

    print("="*50)
    print(f"特征矩阵已保存至：{OUTPUT_NPZ}")
    print(f"CSV格式已保存至：{OUTPUT_CSV}")
    print("D4 特征提取任务完成")