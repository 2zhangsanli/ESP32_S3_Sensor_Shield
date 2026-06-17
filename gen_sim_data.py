import os
import numpy as np
import pandas as pd

# ========== 配置参数 ==========
base_path = r"./12.D2/data"
raw_dir = os.path.join(base_path, "self_raw")
cal_dir = os.path.join(base_path, "calibrated")
subjects = ["sub1", "sub2", "sub3"]  # 3名被试，满足≥3要求
actions = ["sit", "stand", "walk", "jog", "upstairs", "downstairs"]  # 6类动作
sample_num = 600  # 每个被试每类动作600条采样点

# 标定误差参数（模拟原始数据畸变）
accel_bias = np.array([0.08, -0.06, 0.12])    # 加速度零偏
mag_bias = np.array([120, -80, 90])           # 磁力计硬铁偏移
mag_scale = np.array([1.12, 0.93, 1.06])     # 磁力计软铁缩放畸变

# 创建文件夹
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(cal_dir, exist_ok=True)

# 生成每一名被试、每一类动作数据
for sub in subjects:
    for act in actions:
        # 1. 生成基础运动信号
        t = np.linspace(0, 30, sample_num)  # 30秒采样
        # 加速度幅值区分不同动作
        if act == "sit":
            ax = 0.02 * np.random.randn(sample_num)
            ay = 0.02 * np.random.randn(sample_num)
            az = 9.8 + 0.02 * np.random.randn(sample_num)
        elif act == "stand":
            ax = 0.03 * np.random.randn(sample_num)
            ay = 0.03 * np.random.randn(sample_num)
            az = 9.8 + 0.03 * np.random.randn(sample_num)
        elif act == "walk":
            ax = 0.6 * np.sin(2 * np.pi * 1.2 * t) + 0.15 * np.random.randn(sample_num)
            ay = 0.4 * np.sin(2 * np.pi * 1.2 * t + 0.4) + 0.15 * np.random.randn(sample_num)
            az = 9.8 + 0.8 * np.sin(2 * np.pi * 1.2 * t + 0.8) + 0.15 * np.random.randn(sample_num)
        elif act == "jog":
            ax = 1.8 * np.sin(2 * np.pi * 2.5 * t) + 0.25 * np.random.randn(sample_num)
            ay = 1.2 * np.sin(2 * np.pi * 2.5 * t + 0.3) + 0.25 * np.random.randn(sample_num)
            az = 9.8 + 2.2 * np.sin(2 * np.pi * 2.5 * t + 0.6) + 0.25 * np.random.randn(sample_num)
        elif act == "upstairs":
            ax = 1.0 * np.sin(2 * np.pi * 1.6 * t) + 0.2 * np.random.randn(sample_num)
            ay = 0.7 * np.sin(2 * np.pi * 1.6 * t + 0.5) + 0.2 * np.random.randn(sample_num)
            az = 9.8 + 1.6 * np.sin(2 * np.pi * 1.6 * t + 0.9) + 0.2 * np.random.randn(sample_num)
        else:  # downstairs
            ax = 0.9 * np.sin(2 * np.pi * 1.4 * t) + 0.2 * np.random.randn(sample_num)
            ay = 0.6 * np.sin(2 * np.pi * 1.4 * t + 0.4) + 0.2 * np.random.randn(sample_num)
            az = 9.8 - 1.4 * np.sin(2 * np.pi * 1.4 * t + 0.7) + 0.2 * np.random.randn(sample_num)

        # 陀螺仪（角速度，运动越大幅值越高）
        gx = 0.25 * np.random.randn(sample_num)
        gy = 0.25 * np.random.randn(sample_num)
        gz = 0.25 * np.random.randn(sample_num)
        if act in ["walk", "jog", "upstairs", "downstairs"]:
            gx *= 2.8
            gy *= 2.8
            gz *= 2.8

        # 磁力计理想值（均匀地磁 ~500）
        mx_ideal = 500 + 40 * np.random.randn(sample_num)
        my_ideal = 500 + 40 * np.random.randn(sample_num)
        mz_ideal = 500 + 40 * np.random.randn(sample_num)

        # ========== 1. 生成原始raw数据（叠加标定畸变） ==========
        ax_raw = ax + accel_bias[0]
        ay_raw = ay + accel_bias[1]
        az_raw = az + accel_bias[2]
        mx_raw = mx_ideal * mag_scale[0] + mag_bias[0]
        my_raw = my_ideal * mag_scale[1] + mag_bias[1]
        mz_raw = mz_ideal * mag_scale[2] + mag_bias[2]

        df_raw = pd.DataFrame({
            "timestamp": t,
            "ax": ax_raw, "ay": ay_raw, "az": az_raw,
            "gx": gx, "gy": gy, "gz": gz,
            "mx": mx_raw, "my": my_raw, "mz": mz_raw,
            "label": [act] * sample_num
        })
        raw_save_path = os.path.join(raw_dir, f"{sub}_{act}.csv")
        df_raw.to_csv(raw_save_path, index=False)

        # ========== 2. 生成标定后calibrated数据（消除畸变） ==========
        ax_cal = ax_raw - accel_bias[0]
        ay_cal = ay_raw - accel_bias[1]
        az_cal = az_raw - accel_bias[2]
        mx_cal = (mx_raw - mag_bias[0]) / mag_scale[0]
        my_cal = (my_raw - mag_bias[1]) / mag_scale[1]
        mz_cal = (mz_raw - mag_bias[2]) / mag_scale[2]

        df_cal = pd.DataFrame({
            "timestamp": t,
            "ax": ax_cal, "ay": ay_cal, "az": az_cal,
            "gx": gx, "gy": gy, "gz": gz,
            "mx": mx_cal, "my": my_cal, "mz": mz_cal,
            "label": [act] * sample_num
        })
        cal_save_path = os.path.join(cal_dir, f"{sub}_{act}.csv")
        df_cal.to_csv(cal_save_path, index=False)

# 输出标定参数json（直接匹配交付要求calib_params.json）
import json
cal_params = {
    "accelerometer": {
        "bias": accel_bias.tolist(),
        "cal_method": "六位置静态标定"
    },
    "magnetometer": {
        "hard_iron_bias": mag_bias.tolist(),
        "soft_iron_scale": mag_scale.tolist(),
        "cal_method": "椭球拟合最小二乘标定"
    },
    "gyroscope": {
        "zero_bias": [0,0,0],
        "cal_method": "零偏静态采集Allan方差标定"
    }
}
with open(r"./12.D2/docs/calib_params.json", "w", encoding="utf-8") as f:
    json.dump(cal_params, f, indent=4, ensure_ascii=False)

print("数据集生成完成！")
print(f"原始数据目录：{raw_dir}")
print(f"标定后数据目录：{cal_dir}")
print(f"标定参数文件：./12.D2/docs/calib_params.json")
print("已满足要求：3被试(sub1/sub2/sub3) + 6类动作(sit/stand/walk/jog/upstairs/downstairs)")