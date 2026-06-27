import numpy as np
import json
import matplotlib.pyplot as plt
import os

# 路径配置（适配你的目录结构）
data_dir = "../data/calib_raw/"
param_path = "../params/calib_params.json"
fig_path = "../figures/acc_modulus_compare.png"
g_true = 9.81  # 标准重力加速度 m/s²

# 文件名映射
file_map = {
    'px': 'acc_x_up.csv',
    'nx': 'acc_x_down.csv',
    'py': 'acc_y_up.csv',
    'ny': 'acc_y_down.csv',
    'pz': 'acc_z_up.csv',
    'nz': 'acc_z_down.csv'
}

# 1. 读取6个姿态数据并取均值
positions = list(file_map.keys())
raw_means = []
for pos in positions:
    file_path = os.path.join(data_dir, file_map[pos])
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    acc_raw = data[:, 0:3]  # 前三列为ax, ay, az
    raw_means.append(np.mean(acc_raw, axis=0))
raw_means = np.array(raw_means)

# 2. 最小二乘球拟合法求解零偏与刻度因子
A = np.column_stack([raw_means, np.ones(6)])
rhs = -np.sum(raw_means ** 2, axis=1)
params_ls, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)

bias = -params_ls[:3] / 2
radius = np.sqrt(np.sum(bias ** 2) - params_ls[3])
scale = g_true / radius  # 统一刻度因子
M = np.diag([scale, scale, scale])  # 标定矩阵（对角简化版）

# 3. 计算标定前后合加速度模长
raw_modulus = np.linalg.norm(raw_means, axis=1)
cal_means = (raw_means - bias) * scale
cal_modulus = np.linalg.norm(cal_means, axis=1)

# 4. 生成对比图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 5))
x = np.arange(6)
width = 0.35
plt.bar(x - width/2, raw_modulus, width, label='标定前', color='#ff6b6b')
plt.bar(x + width/2, cal_modulus, width, label='标定后', color='#4ecdc4')
plt.axhline(y=g_true, color='r', linestyle='--', label='标准重力加速度')
plt.xticks(x, ['+X', '-X', '+Y', '-Y', '+Z', '-Z'])
plt.ylabel('合加速度模长 (m/s²)')
plt.title('加速度计六位置标定前后合加速度对比')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=300)
plt.close()

# 5. 保存标定参数
calib_params = {
    "acc_bias": bias.tolist(),
    "acc_scale_matrix": M.tolist(),
    "acc_calibration_note": "六位置球拟合法，最小二乘求解",
    "raw_modulus_range": [float(np.min(raw_modulus)), float(np.max(raw_modulus))],
    "cal_modulus_range": [float(np.min(cal_modulus)), float(np.max(cal_modulus))]
}

os.makedirs(os.path.dirname(param_path), exist_ok=True)
if os.path.exists(param_path):
    with open(param_path, 'r') as f:
        all_params = json.load(f)
else:
    all_params = {}
all_params.update(calib_params)
with open(param_path, 'w') as f:
    json.dump(all_params, f, indent=4, ensure_ascii=False)

# 控制台输出结果
print("===== 加速度计标定完成 =====")
print(f"三轴零偏（原始单位）：{bias.round(4)}")
print(f"标定前模长范围：{np.min(raw_modulus):.4f} ~ {np.max(raw_modulus):.4f} m/s²")
print(f"标定后模长范围：{np.min(cal_modulus):.4f} ~ {np.max(cal_modulus):.4f} m/s²")
print(f"对比图已保存：{fig_path}")
print(f"参数已保存：{param_path}")