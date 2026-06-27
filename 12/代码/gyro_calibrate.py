import numpy as np
import json
import matplotlib.pyplot as plt
import os

# 路径配置
data_path = "../data/calib_raw/gyro_static.csv"
param_path = "../params/calib_params.json"
fig_path = "../figures/gyro_static_bias.png"
fs = 50  # 采样率

# 1. 读取数据
data = np.loadtxt(data_path, delimiter=',', skiprows=1)
gyro_raw = data[:, 0:3]  # 前三列为gx, gy, gz
time = np.arange(len(gyro_raw)) / fs

# 2. 计算零偏
gyro_bias = np.mean(gyro_raw, axis=0)
gyro_cal = gyro_raw - gyro_bias

# 3. 绘制零偏对比图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 6))
for i, axis in enumerate(['X', 'Y', 'Z']):
    plt.subplot(3, 1, i+1)
    plt.plot(time, gyro_raw[:, i], label='原始数据', alpha=0.6, color='#ff6b6b')
    plt.axhline(y=gyro_bias[i], color='r', linestyle='--', label=f'零偏={gyro_bias[i]:.4f}')
    plt.ylabel(f'{axis}轴角速度')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
plt.suptitle('陀螺仪静态零偏标定结果')
plt.tight_layout()
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=300)
plt.close()

# 4. 保存参数
gyro_params = {
    "gyro_bias": gyro_bias.tolist(),
    "gyro_calibration_note": "静态零偏标定，静置数据均值校正"
}

if os.path.exists(param_path):
    with open(param_path, 'r') as f:
        all_params = json.load(f)
else:
    all_params = {}
all_params.update(gyro_params)
with open(param_path, 'w') as f:
    json.dump(all_params, f, indent=4, ensure_ascii=False)

print("===== 陀螺仪标定完成 =====")
print(f"三轴零偏：{gyro_bias.round(4)}")
print(f"零偏图已保存：{fig_path}")
print(f"参数已保存：{param_path}")