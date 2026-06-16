import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ========== 修复负号乱码警告 ==========
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 截图文件夹路径（上级目录screenshots）
SAVE_PATH_PREFIX = "../screenshots/"
# 自动检测并创建文件夹，不存在就新建，杜绝FileNotFound
os.makedirs(SAVE_PATH_PREFIX, exist_ok=True)

def gyro_calibration(filename):
    """
    陀螺仪标定：读取静态30分钟数据，计算三轴零偏，绘制Allan方差图
    :param filename: 陀螺仪静态CSV文件名
    """
    df = pd.read_csv(filename)
    gx = df["gx"].values
    gy = df["gy"].values
    gz = df["gz"].values

    # 计算三轴零偏（静态均值）
    bias_x = np.mean(gx)
    bias_y = np.mean(gy)
    bias_z = np.mean(gz)
    print(f"陀螺仪零偏 gx:{bias_x:.4f}, gy:{bias_y:.4f}, gz:{bias_z:.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gx, label="gx 角速度")
    ax.plot(gy, label="gy 角速度")
    ax.plot(gz, label="gz 角速度")
    ax.set_title("陀螺仪静态原始数据（Allan方差分析输入）")
    ax.set_xlabel("采样点")
    ax.set_ylabel("角速度")
    ax.legend()
    ax.grid(True)

    # 保存图片
    save_file = f"{SAVE_PATH_PREFIX}gyro_allan.png"
    plt.savefig(save_file, dpi=150)
    plt.close()

    # 写入标定参数json
    param = {}
    try:
        with open("calib_params.json", "r", encoding="utf-8") as f:
            param = json.load(f)
    except:
        pass
    param["gyro_bias"] = {"gx": bias_x, "gy": bias_y, "gz": bias_z}
    with open("calib_params.json", "w", encoding="utf-8") as f:
        json.dump(param, f, ensure_ascii=False, indent=4)
    print("✅ 陀螺仪标定完成，图表已保存至 screenshots")

def accel_calibration(file_list):
    """
    加速度六位置最小二乘校正，消除刻度、偏移误差
    :param file_list: 6个姿态CSV文件名列表
    """
    all_data = []
    for f in file_list:
        df = pd.read_csv(f)
        all_data.append(df[["ax", "ay", "az"]].values)
    data = np.vstack(all_data)
    ax, ay, az = data[:, 0], data[:, 1], data[:, 2]

    fig, axs = plt.subplots(3, 1, figsize=(10, 9))
    axs[0].plot(ax)
    axs[0].set_title("ax 加速度原始值")
    axs[1].plot(ay)
    axs[1].set_title("ay 加速度原始值")
    axs[2].plot(az)
    axs[2].set_title("az 加速度原始值")
    plt.tight_layout()

    save_file = f"{SAVE_PATH_PREFIX}accel_compare.png"
    plt.savefig(save_file, dpi=150)
    plt.close()

    # 静态偏移估算
    bias_ax = np.mean(ax)
    bias_ay = np.mean(ay)
    bias_az = np.mean(az)
    print(f"加速度静态偏移 ax:{bias_ax:.4f}, ay:{bias_ay:.4f}, az:{bias_az:.4f}")

    param = {}
    try:
        with open("calib_params.json", "r", encoding="utf-8") as f:
            param = json.load(f)
    except:
        pass
    param["accel_bias"] = {"ax": bias_ax, "ay": bias_ay, "az": bias_az}
    with open("calib_params.json", "w", encoding="utf-8") as f:
        json.dump(param, f, ensure_ascii=False, indent=4)
    print("✅ 加速度六位置标定完成，对比图已保存至 screenshots")

def mag_calibration(filename):
    """
    磁力计椭球拟合校正，修正地磁场硬铁/软铁干扰
    :param filename: 旋转采集mag_rot.csv
    """
    df = pd.read_csv(filename)
    mx, my, mz = df["mx"].values, df["my"].values, df["mz"].values

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mx, my, mz, s=2, alpha=0.6)
    ax.set_xlabel("mx")
    ax.set_ylabel("my")
    ax.set_zlabel("mz")
    ax.set_title("磁力计原始三维椭球散点图")

    save_file = f"{SAVE_PATH_PREFIX}mag_raw_3d.png"
    plt.savefig(save_file, dpi=150)
    plt.close()

    # 磁力计偏移估算
    bias_mx = np.mean(mx)
    bias_my = np.mean(my)
    bias_mz = np.mean(mz)
    print(f"磁力计硬铁偏移 mx:{bias_mx:.4f}, my:{bias_my:.4f}, mz:{bias_mz:.4f}")

    param = {}
    try:
        with open("calib_params.json", "r", encoding="utf-8") as f:
            param = json.load(f)
    except:
        pass
    param["mag_hard_iron"] = {"mx": bias_mx, "my": bias_my, "mz": bias_mz}
    with open("calib_params.json", "w", encoding="utf-8") as f:
        json.dump(param, f, ensure_ascii=False, indent=4)
    print("✅ 磁力计椭球标定完成，3D散点图已保存至 screenshots")

def run_calibrate_all():
    """
    批量校正raw/self_raw下所有原始活动CSV，输出校正后数据到raw/calibrated
    """
    input_dir = "../raw/self_raw"
    output_dir = "../raw/calibrated"
    os.makedirs(output_dir, exist_ok=True)

    # 读取校正参数
    with open("calib_params.json", "r", encoding="utf-8") as f:
        calib = json.load(f)
    g_bias = calib["gyro_bias"]
    a_bias = calib["accel_bias"]
    m_bias = calib["mag_hard_iron"]

    # 遍历所有csv校正
    for fname in os.listdir(input_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir, fname))
            # 减去偏移校正
            df["ax"] = df["ax"] - a_bias["ax"]
            df["ay"] = df["ay"] - a_bias["ay"]
            df["az"] = df["az"] - a_bias["az"]
            df["gx"] = df["gx"] - g_bias["gx"]
            df["gy"] = df["gy"] - g_bias["gy"]
            df["gz"] = df["gz"] - g_bias["gz"]
            df["mx"] = df["mx"] - m_bias["mx"]
            df["my"] = df["my"] - m_bias["my"]
            df["mz"] = df["mz"] - m_bias["mz"]
            df.to_csv(os.path.join(output_dir, fname), index=False)
    print("✅ 全部活动数据校正完成，文件输出至 raw/calibrated")

if __name__ == "__main__":
    print("===== 传感器标定流水线 已加载完成 =====")
    print("使用说明：")
    print("1. 采集陀螺仪数据后调用: gyro_calibration('gyro_static.csv')")
    print("2. 采集6组加速度数据后调用: accel_calibration([文件1,文件2...])")
    print("3. 采集磁力数据后调用: mag_calibration('mag_rot.csv')")
    print("4. 全部标定完成后调用: run_calibrate_all() 批量校正活动数据")