import numpy as np
import json
import os
import glob

raw_dir = "../data/raw/"
cal_dir = "../data/calibrated/"
param_path = "../params/calib_params.json"

# 读取标定参数
with open(param_path, 'r') as f:
    params = json.load(f)
acc_bias = np.array(params["acc_bias"])
acc_M = np.array(params["acc_scale_matrix"])
gyro_bias = np.array(params["gyro_bias"])

os.makedirs(cal_dir, exist_ok=True)

# 批量处理所有CSV
csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
for file_path in csv_files:
    filename = os.path.basename(file_path)
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    
    # 前三列加速度，后三列陀螺仪（根据你的实际列数调整）
    acc_raw = data[:, 0:3]
    gyro_raw = data[:, 3:6]
    
    # 标定校正
    acc_cal = (acc_raw - acc_bias) @ acc_M.T
    gyro_cal = gyro_raw - gyro_bias
    
    # 保存
    cal_data = np.column_stack([acc_cal, gyro_cal])
    np.savetxt(
        os.path.join(cal_dir, filename),
        cal_data,
        delimiter=',',
        header='ax_cal,ay_cal,az_cal,gx_cal,gy_cal,gz_cal',
        comments=''
    )
    print(f"已标定：{filename}")

print(f"\n全部处理完成，共 {len(csv_files)} 个文件，输出目录：{cal_dir}")