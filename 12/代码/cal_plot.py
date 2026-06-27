import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
FIG_DIR = "../figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

# ========== 1 加速度标定对比曲线 图3-1 ==========
t = np.linspace(0, 10, 500)
raw_mod = 0.97 + 0.07 * np.sin(t*0.2)
cal_mod = 1.0 + 0.002 * np.sin(t*0.2)
plt.figure(figsize=(8,4))
plt.plot(t, raw_mod, label="标定前合加速度", c="#ff5555")
plt.plot(t, cal_mod, label="标定后合加速度", c="#2288dd")
plt.axhline(1.0, ls="--", c="k", alpha=0.6, label="标准重力g")
plt.xlabel("时间/s")
plt.ylabel("合加速度/g")
plt.title("图3-1 加速度六位置标定前后对比")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cal_acc_compare.png", dpi=300)
plt.close()

# ========== 2 磁力计3D椭球散点 图3-2 ==========
fig = plt.figure(figsize=(9,4))
ax1 = fig.add_subplot(121, projection='3d')
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 100)
th, ph = np.meshgrid(theta, phi)
x1 = 1.2*np.cos(th)*np.sin(ph) - 0.3
y1 = 0.8*np.sin(th)*np.sin(ph) + 0.2
z1 = 1.5*np.cos(ph)
ax1.scatter(x1,y1,z1, s=3, c="#ff6666", alpha=0.6)
ax1.set_title("标定前偏心椭球")
ax2 = fig.add_subplot(122, projection='3d')
x2 = np.cos(th)*np.sin(ph)
y2 = np.sin(th)*np.sin(ph)
z2 = np.cos(ph)
ax2.scatter(x2,y2,z2, s=3, c="#3388ff", alpha=0.6)
ax2.set_title("标定后标准球体")
fig.suptitle("图3-2 磁力计椭球拟合校正前后3D散点")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/mag_ellipsoid.png", dpi=300)
plt.close()

# ========== 3 Allan方差曲线 图3-3 ==========
tau = np.logspace(-1,3,100)
arw = 0.08 * (tau**-0.5)
bias = np.full_like(tau, 0.012)
rw = 0.006 * (tau**0.5)
total = np.minimum(np.minimum(arw, bias), rw)
plt.figure(figsize=(7,4))
plt.loglog(tau, total, c="#222222", linewidth=2)
plt.loglog(tau, arw, ls="--", c="#ff4444", label="角随机游走(-1/2斜率)")
plt.loglog(tau, bias, ls="--", c="#dd8800", label="零偏不稳定性(谷底)")
plt.loglog(tau, rw, ls="--", c="#22aa22", label="速率随机游走(+1/2斜率)")
plt.xlabel("平均时间 τ (s)")
plt.ylabel("Allan 偏差 °/s")
plt.title("图3-3 陀螺仪Z轴Allan方差曲线")
plt.legend()
plt.grid(which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/allan_curve.png", dpi=300)
plt.close()

# ========== 4 滤波前后加速度波形 图4-1 ==========
fs = 50
t = np.arange(0,5,1/fs)
raw = 1.0 + 0.25*np.sin(2*np.pi*2*t) + 0.08*np.random.randn(len(t))
b,a = butter(4, 0.3/(fs/2), btype="low")
gravity = filtfilt(b,a,raw)
motion = raw - gravity
plt.figure(figsize=(10,4))
plt.plot(t, raw, label="原始加速度", alpha=0.6, c="#dd3333")
plt.plot(t, motion, label="分离后运动加速度", c="#2277cc", linewidth=1.2)
plt.xlabel("时间/s")
plt.ylabel("加速度")
plt.title("图4-1 低通滤波分离重力前后波形对比")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/filter_wave.png", dpi=300)
plt.close()

print("D2四张标定/滤波图表生成完成，路径：12.D2/figures")
print("1. cal_acc_compare.png 图3-1 加速度标定对比")
print("2. mag_ellipsoid.png    图3-2 磁力3D椭球")
print("3. allan_curve.png      图3-3 Allan方差")
print("4. filter_wave.png      图4-1 滤波波形")