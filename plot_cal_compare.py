import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取sub1静坐原始/标定磁力数据
raw_df = pd.read_csv("./12.D2/data/self_raw/sub1_sit.csv")
cal_df = pd.read_csv("./12.D2/data/calibrated/sub1_sit.csv")

# 1. 磁力椭球对比图
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(raw_df["mx"], raw_df["my"], s=3, c="#ff4444", label="标定前(椭球畸变)")
plt.xlabel("Mx")
plt.ylabel("My")
plt.title("磁力计标定前分布")
plt.legend()

plt.subplot(1,2,2)
plt.scatter(cal_df["mx"], cal_df["my"], s=3, c="#2288dd", label="标定后(标准球面)")
plt.xlabel("Mx")
plt.ylabel("My")
plt.title("磁力计标定后分布")
plt.legend()
plt.tight_layout()
plt.savefig("./12.D2/screenshots/mag_ellipse_compare.png", dpi=150)

# 2. 加速度合模长对比
raw_mod = np.sqrt(raw_df["ax"]**2 + raw_df["ay"]**2 + raw_df["az"]**2)
cal_mod = np.sqrt(cal_df["ax"]**2 + cal_df["ay"]**2 + cal_df["az"]**2)
plt.figure(figsize=(10,4))
plt.plot(raw_mod, c="red", label="标定前合加速度模长", alpha=0.7)
plt.plot(cal_mod, c="blue", label="标定后合加速度模长", alpha=0.8)
plt.axhline(9.8, c="black", linestyle="--", label="标准重力9.8m/s²")
plt.xlabel("采样点")
plt.ylabel("加速度模长 m/s²")
plt.title("静坐状态加速度模长标定前后对比")
plt.legend()
plt.tight_layout()
plt.savefig("./12.D2/screenshots/accel_mod_compare.png", dpi=150)

print("对比图已生成至screenshots文件夹，可直接插入标定报告.md")