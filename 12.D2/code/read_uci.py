import pandas as pd
import os

# 修正相对路径：.. 代表向上一层，从code文件夹跳到12.D2目录，匹配raw文件夹位置
root_path = "../raw/UCI HAR Dataset"

# 1. 读取活动标签对照表
label_path = os.path.join(root_path, "activity_labels.txt")
activity_df = pd.read_csv(label_path, sep=" ", names=["编号", "活动名称"])
print("===== 人体活动标签对应表 =====")
print(activity_df)

# 2. 读取训练集被试ID（LOSO评估核心）
sub_train_path = os.path.join(root_path, "train/subject_train.txt")
subject_df = pd.read_csv(sub_train_path, names=["被试编号"])
unique_subject = sorted(subject_df["被试编号"].unique())
print(f"\n===== 训练集被试信息 =====")
print(f"参与实验总人数：{len(unique_subject)} 人")
print(f"被试编号列表：{unique_subject}")

# 3. 读取原始加速度时序数据（六轴核心数据）
acc_x_path = os.path.join(root_path, "train/Inertial Signals/total_acc_x_train.txt")
acc_x_df = pd.read_csv(acc_x_path, sep="\s+", header=None)
print(f"\n===== 加速度X轴数据信息 =====")
print(f"数据维度(样本数,单窗口采样点)：{acc_x_df.shape}")
print("前5行原始数据：")
print(acc_x_df.head())