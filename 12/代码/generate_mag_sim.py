import numpy as np
import os

# 输出路径
output_path = "../data/calib_raw/mag_rotate.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 1. 生成均匀分布的球面采样点（模拟全方位旋转）
num_samples = 2000
theta = np.random.uniform(0, 2*np.pi, num_samples)  # 方位角
phi = np.arccos(np.random.uniform(-1, 1, num_samples))  # 极角（均匀球面）

# 标准球坐标（单位地磁场，幅值约500 LSB，符合HMC5883L量程）
field_strength = 500.0
mx_true = field_strength * np.sin(phi) * np.cos(theta)
my_true = field_strength * np.sin(phi) * np.sin(theta)
mz_true = field_strength * np.cos(phi)
mag_true = np.column_stack([mx_true, my_true, mz_true])

# 2. 模拟软铁干扰（三轴缩放不同，变成椭球）
soft_iron_scale = np.diag([0.92, 1.08, 1.05])  # 软铁缩放矩阵
mag_soft = mag_true @ soft_iron_scale.T

# 3. 模拟硬铁干扰（整体偏移，球心平移）
hard_iron_bias = np.array([85.0, -60.0, 45.0])  # 硬铁偏移量
mag_raw = mag_soft + hard_iron_bias

# 4. 添加随机测量噪声
noise_level = 3.0
mag_raw += np.random.normal(0, noise_level, mag_raw.shape)

# 5. 保存为CSV
np.savetxt(
    output_path,
    mag_raw,
    delimiter=',',
    header='mx,my,mz',
    comments=''
)

print(f"模拟磁力计数据已生成：{output_path}")
print(f"数据点数：{num_samples}")
print(f"模拟硬铁偏移：{hard_iron_bias}")
print(f"模拟软铁缩放：{np.diag(soft_iron_scale)}")
print("可直接用于椭球拟合标定实验")