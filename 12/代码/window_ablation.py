import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# ===================== 配置 =====================
FS = 50  # 采样率
DATA_DIR = "../data/calibrated/"
FIG_PATH = "../figures/window_ablation.png"

# 待对比的参数组合（窗口长度s, 重叠率）
configs = [
    (1.28, 0.0),   # 短窗口，无重叠
    (1.28, 0.5),   # 短窗口，50%重叠
    (2.56, 0.0),   # 基准窗口，无重叠
    (2.56, 0.5),   # 基准配置（默认参数）
    (2.56, 0.75),  # 基准窗口，高重叠
    (5.12, 0.0),   # 长窗口，无重叠
    (5.12, 0.5),   # 长窗口，50%重叠
]

# ===================== 工具函数 =====================
def count_windows(file_path, window_size, step_size):
    """计算单个文件可生成的窗口数"""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    n_samples = len(data)
    if n_samples < window_size:
        return 0
    return (n_samples - window_size) // step_size + 1

# ===================== 主程序 =====================
if __name__ == "__main__":
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"数据集文件总数：{len(csv_files)} 个")
    print(f"单文件时长：10s，采样率：{FS}Hz，单文件样本数：{10*FS}")
    print("="*70)
    print(f"{'窗口长度(s)':<12} {'重叠率':<8} {'窗口长(点)':<12} {'步长(点)':<10} {'总窗口数':<10}")
    print("-"*70)

    results = []
    for win_sec, overlap in configs:
        win_len = int(win_sec * FS)
        step = int(win_len * (1 - overlap))
        total = 0
        for f in csv_files:
            total += count_windows(f, win_len, step)
        results.append((win_sec, overlap, win_len, step, total))
        print(f"{win_sec:<12} {overlap:<8.0%} {win_len:<12} {step:<10} {total:<10}")

    print("="*70)

    # ===================== 生成对比图 =====================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    x_labels = [f"{w}s\n{o:.0%}重叠" for w, o, _, _, _ in results]
    totals = [r[4] for r in results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x_labels, totals, color='#45b7d1', edgecolor='#2d8fba', width=0.6)
    
    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.title('窗口参数对样本数量的影响（消融实验）', fontsize=14)
    plt.ylabel('总窗口样本数', fontsize=12)
    plt.xlabel('窗口长度 + 重叠率配置', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()
    
    print(f"\n消融实验对比图已保存至：{FIG_PATH}")

    # ===================== 分析结论打印 =====================
    print("\n" + "="*70)
    print("消融实验分析结论：")
    print("1. 窗口长度固定时，重叠率越高，样本数量越多，数据利用率越高，但样本冗余度也随之增大。")
    print("2. 重叠率固定时，窗口长度越短，样本数量越多，时间分辨率越高，但单窗口包含的动作周期越少，特征稳定性下降。")
    print("3. 2.56s窗口+50%重叠为均衡配置：单窗口可覆盖1~2个完整步态周期，样本量充足且冗余度适中。")
    print("4. 5.12s长窗口特征更稳定，但时间分辨率低，难以捕捉快速动作切换；1.28s短窗口适合瞬态动作识别。")