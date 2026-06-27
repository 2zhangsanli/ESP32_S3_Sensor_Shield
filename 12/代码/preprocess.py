import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

# ===================== 配置参数 =====================
FS = 50                  # 采样率 Hz
WINDOW_SEC = 2.56        # 窗口长度 s
OVERLAP_RATIO = 0.5      # 重叠率
CUTOFF_HZ = 20           # 低通截止频率 Hz
ORDER = 4                # 巴特沃斯阶数

# 标签映射
LABEL_MAP = {
    'sit': 0,
    'stand': 1,
    'walk': 2,
    'upstairs': 3,
    'downstairs': 4,
    'run': 5
}

# 路径
DATA_DIR = "../data/calibrated/"
OUTPUT_NPZ = "../output/windowed_dataset.npz"
FIG_PATH = "../figures/waveform_compare.png"

# ===================== 工具函数 =====================
def parse_filename(filename):
    """解析文件名：data_p2_walk_001.csv → subject=2, label=walk"""
    name = os.path.basename(filename).replace('.csv', '')
    parts = name.split('_')
    subject = int(parts[1].replace('p', ''))
    label_str = parts[2]
    label = LABEL_MAP[label_str]
    return subject, label

def remove_outliers(data, n_std=3):
    """3σ原则截断异常值"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    upper = mean + n_std * std
    lower = mean - n_std * std
    return np.clip(data, lower, upper)

def remove_gravity(acc_data, fs=FS, win_sec=1.0):
    """滑动平均提取重力分量，相减得到运动加速度"""
    win_len = int(fs * win_sec)
    gravity = np.zeros_like(acc_data)
    for i in range(acc_data.shape[1]):
        # 卷积实现滑动平均，边缘镜像填充
        kernel = np.ones(win_len) / win_len
        gravity[:, i] = np.convolve(acc_data[:, i], kernel, mode='same')
    return acc_data - gravity

def butter_lowpass_filter(data, cutoff=CUTOFF_HZ, fs=FS, order=ORDER):
    """巴特沃斯低通滤波，零相位filtfilt"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

def sliding_window(data, window_size, step_size):
    """滑动窗口分割：(总样本数, 特征数) → (窗口数, 窗口长, 特征数)"""
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // step_size + 1
    windows = []
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows.append(data[start:end, :])
    return np.array(windows)

# ===================== 主预处理流水线 =====================
def preprocess_pipeline(file_path):
    """单个文件完整预处理"""
    # 1. 读取数据
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    acc = data[:, 0:3]   # 三轴加速度
    gyro = data[:, 3:6]  # 三轴陀螺仪

    # 2. 异常值截断
    acc = remove_outliers(acc)
    gyro = remove_outliers(gyro)

    # 3. 加速度去重力
    acc_motion = remove_gravity(acc)

    # 4. 低通滤波去噪
    acc_filtered = butter_lowpass_filter(acc_motion)
    gyro_filtered = butter_lowpass_filter(gyro)

    # 5. 拼接六轴特征
    signal = np.column_stack([acc_filtered, gyro_filtered])
    return signal, acc  # 返回预处理结果和原始加速度，用于画图

# ===================== 主程序 =====================
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)

    window_size = int(WINDOW_SEC * FS)
    step_size = int(window_size * (1 - OVERLAP_RATIO))

    all_X = []       # 所有窗口样本
    all_y = []       # 标签
    all_subject = [] # 被试ID
    demo_raw = None
    demo_processed = None

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"共找到 {len(csv_files)} 个数据文件")

    for i, file_path in enumerate(csv_files):
        subject, label = parse_filename(file_path)
        signal_processed, acc_raw = preprocess_pipeline(file_path)

        # 滑动窗口
        windows = sliding_window(signal_processed, window_size, step_size)
        n_win = len(windows)

        all_X.append(windows)
        all_y.extend([label] * n_win)
        all_subject.extend([subject] * n_win)

        # 选第一个行走文件存为演示画图
        if i == 12 and demo_raw is None:  # walk文件
            demo_raw = acc_raw
            demo_processed = signal_processed[:, 0:3]

    # 拼接完整数据集
    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y)
    subject_ids = np.array(all_subject)

    # 保存为npz（保留多维结构，比csv高效）
    np.savez(
        OUTPUT_NPZ,
        X=X,
        y=y,
        subject_ids=subject_ids,
        feature_names=['ax','ay','az','gx','gy','gz'],
        window_size=window_size,
        step_size=step_size,
        fs=FS
    )

    # ===================== 生成对比图 =====================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = np.arange(len(demo_raw)) / FS

    for i, axis_name in enumerate(['X轴', 'Y轴', 'Z轴']):
        axes[i].plot(t, demo_raw[:, i], label='原始加速度', alpha=0.6, color='#ff6b6b')
        axes[i].plot(t, demo_processed[:, i], label='去重力+滤波后', color='#4ecdc4')
        axes[i].set_ylabel(f'{axis_name}加速度 (m/s²)')
        axes[i].legend(loc='upper right')
        axes[i].grid(alpha=0.3)

    axes[0].set_title('预处理前后加速度波形对比（行走样本）')
    axes[-1].set_xlabel('时间 (s)')
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()

    # ===================== 输出统计信息 =====================
    print("="*50)
    print("窗口化数据集生成完成")
    print(f"窗口长度：{WINDOW_SEC}s ({window_size} 点)")
    print(f"重叠率：{OVERLAP_RATIO*100}%")
    print(f"总样本数：{len(X)} 个窗口")
    print(f"特征维度：{X.shape[1]} × {X.shape[2]}（窗口长 × 特征数）")
    print(f"被试数量：{len(np.unique(subject_ids))} 人")
    print(f"类别数量：{len(np.unique(y))} 类")
    print(f"数据集已保存：{OUTPUT_NPZ}")
    print(f"对比图已保存：{FIG_PATH}")

    # ===================== 标准化防泄漏示例 =====================
    print("\n" + "="*50)
    print("标准化防泄漏验证（LOSO模式）")
    # 模拟留一被试：被试1为测试集，其余为训练集
    train_mask = subject_ids != 1
    test_mask = subject_ids == 1

    X_train = X[train_mask]
    X_test = X[test_mask]

    # 仅用训练集拟合标准化器
    scaler = StandardScaler()
    # 展平成二维拟合，再还原三维
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)

    # 分别变换训练集和测试集
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    print(f"训练集形状：{X_train_scaled.shape}")
    print(f"测试集形状：{X_test_scaled.shape}")
    print("标准化参数仅来自训练集，严格杜绝数据泄漏")