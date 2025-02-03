import numpy as np
from filterpy.kalman import KalmanFilter
import csv
# 初始化卡尔曼滤波器
kf = KalmanFilter(dim_x=4, dim_z=2)  # 状态: [x, vx, y, vy], 观测: [x, y]
kf.F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])  # 状态转移矩阵
kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # 观测矩阵
kf.Q = np.eye(4) * 1  # 过程噪声协方差矩阵（需要调整）
kf.R = np.eye(2) * 100  # 观测噪声协方差矩阵（需要调整）
kf.P = np.eye(4) * 1  # 初始协方差矩阵（需要调整）

# 设置筛选条件（这里假设对所有帧使用相同的条件）
x_range = (500, 560)
y_range = (-300, -280)

x2_range=(575,615)
y2=(-400, -300)

# 存储预测和更新后的中心位置
predicted_centers = []
updated_centers = []

# 循环处理每一帧数据
for i in range(1, 501):  # 从1.npz到6.npz
    data = np.load(f'./event5/{i}.npz')
    x = data['x']
    y = data['y']
    if i>300:
        x_range=x2_range
        y_range=y2
    if i>437:
        x_range=x2_range
        y_range=(-300, -260)
    # 筛选点
    mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
    selected_x = x[mask]
    selected_y = y[mask]

    # 观测到的中心位置
    observed_center_x = np.mean(selected_x)
    observed_center_y = np.mean(selected_y)
    observed_center = np.array([[observed_center_x], [observed_center_y]])

    # 如果是第一帧，则初始化滤波器状态
    if i == 1:
        initial_center_x =511.2
        initial_center_y = -270
        kf.x = np.array([[initial_center_x], [0], [initial_center_y], [0]])  # 初始状态（位置和速度）

    # 预测当前帧的状态
    kf.predict()
    predicted_center_x = kf.x[0][0]
    predicted_center_y = kf.x[2][0]
    predicted_centers.append((predicted_center_x, predicted_center_y))

    # 更新滤波器状态以匹配当前帧的观测值
    kf.update(observed_center)
    updated_center_x = kf.x[0][0]
    updated_center_y = -1*kf.x[2][0]
    updated_centers.append((updated_center_x, updated_center_y))

    #输出当前帧的预测和更新结果（可选）
    print(
         f'Frame {i}: Predicted ({predicted_center_x:.4f}, {predicted_center_y:.4f}), Updated ({updated_center_x:.4f}, {updated_center_y:.4f})')
    # with open('results3.csv', 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([i, predicted_center_x, predicted_center_y, updated_center_x, updated_center_y])
#
# # # # 输出所有帧的预测和更新结果（可选，以列表形式）
# # # print("Predicted Centers:", predicted_centers)
# # # print("Updated Centers:", updated_centers)
import matplotlib.pyplot as plt
updated_centers_x = [center[0] for center in updated_centers]
updated_centers_y = [center[1] for center in updated_centers]
# # #print(updated_centers_x)
# # # 文件名列表（从10到100）
file_names = [i for i in range(8, 2008, 4)]

# 绘制 x 坐标随文件名变化的图
plt.figure(figsize=(10, 5))
ax = plt.gca()
# # plt.plot([0, 20], [583, 583], color='g', linestyle='--', linewidth=2,label='img')
# # plt.plot([20, 40], [583, 583], color='g', linestyle='--', linewidth=2)
# # plt.plot([40, 60], [584, 584], color='g', linestyle='--', linewidth=2)
# # plt.plot([60, 80], [586, 586], color='g', linestyle='--', linewidth=2)
# # plt.plot([80, 100], [587.5, 587.5], color='g', linestyle='--', linewidth=2)
# # points_of_interest = [(1,  582.03), (7.84, 582.12),(14.43, 582.35),(20.9,583.29),(27.86,583.297),(34.33,583.70),(41,584.12),(47.75,584.79),(54.35,585.2),(61,585.99),(67.7,586.63),(74.49,586.8),(81,587.9),(87.9,588),(94.6,588.712)]
# #
# # # 为了在x_values中找到最接近的点来绘制线，我们可以创建一个y值的数组，其中大部分值为NaN，
# # # 只在对应的x位置设置有效的y值。但这里为了简化，我们直接绘制点并用plot连接它们。
# #
# # # 提取点的x和y坐标
# # x_points = [point[0] for point in points_of_interest]
# # y_points = [point[1] for point in points_of_interest]
# #
# #
# # # ax.axhline(y=583, color='r', linestyle='--',label='img')  # 红色虚线
# # # 绘制并连接点
# # ax.plot(x_points, y_points, marker='.', label='eye tracking machine')
#
#
#
# plt.plot(file_names, updated_centers_x, linestyle='-', color='b',label='our approach', markersize=0.5, linewidth=0.5)
# # # 读取 CSV 文件
# import pandas as pd
# df = pd.read_csv('D:/pycharm/神经网络/idea/analyse/事件/C.csv')  # 将 'your_file.csv' 替换为你的 CSV 文件路径
#
# # 获取 'true_xx' 列的数据
# true_xx_values = df['true_x'].values
#
# # 生成横坐标
# x_values = [(1 + 6.6 * i) for i in range(301)]
#
# # 绘制图形
# plt.plot(x_values, true_xx_values,linestyle='-', color='r', label='100FPS machine', markersize=0.5, linewidth=0.5)
#
#
# df = pd.read_csv('D:/pycharm/神经网络/idea/analyse/事件/results2.csv')  # 将 'your_file.csv' 替换为你的 CSV 文件路径
#
# # 获取 'true_xx' 列的数据
# true_xx_values = df['x'].values
#
# # 生成横坐标
# x_values = [20* i for i in range(100)]
#
# # 绘制图形
# plt.plot(x_values, true_xx_values, linestyle='-', color='y', label='50FPS video', markersize=0.5, linewidth=1)
#
#
# # 添加标题和标签
# plt.xlabel('Time(ms)')
# plt.ylabel('Horizontal Pixel Position')
# plt.grid(True)
# # ax.set_xlim(700, 750)
# # ax.set_ylim(600,608)
# ax.set_xlim(0, 2000)
# plt.legend()
# plt.show()



# #
# # 绘制 y 坐标随文件名变化的图

# # # plt.plot([0, 20], [312, 312], color='g', linestyle='--', linewidth=2,label='img')
# # # plt.plot([20, 40], [312, 312], color='g', linestyle='--', linewidth=2)
# # # plt.plot([40, 60], [312, 312], color='g', linestyle='--', linewidth=2)
# # # plt.plot([60, 80], [311, 311], color='g', linestyle='--', linewidth=2)
# # # plt.plot([80, 100], [311, 311], color='g', linestyle='--', linewidth=2)
# # ax.set_xlim(0, 2000)
# # plt.plot(file_names, updated_centers_x, marker='o', linestyle='-', color='r',label='img+event')
# # points_of_interest = [(1,  312.13), (7.84, 312.28),(14.43, 312.239),(20.9,312.239),(27.86,312.1816),(34.33,312.088),(41,312.1744),(47.75,312.008),(54.35,312.124),(61,311.8432),(67.7,311.879),(74.49,311.785),(81,311.4616),(87.9,311.512),(94.6,311.519)]
# #
# # # 为了在x_values中找到最接近的点来绘制线，我们可以创建一个y值的数组，其中大部分值为NaN，
# # # 只在对应的x位置设置有效的y值。但这里为了简化，我们直接绘制点并用plot连接它们。
# #
# # # 提取点的x和y坐标
# # x_points = [point[0] for point in points_of_interest]
# # y_points = [point[1] for point in points_of_interest]
# #
# #
# # # ax.axhline(y=583, color='r', linestyle='--',label='img')  # 红色虚线
# # # 绘制并连接点
# # ax.plot(x_points, y_points, marker='.', label='eye tracking machine')
# plt.title('X')
# plt.xlabel('t/ms')
# plt.ylabel('Y values')
# plt.grid(True)
# plt.legend()
# plt.show()
#

#

#
# plt.figure(figsize=(10, 5))
# ax = plt.gca()
# plt.plot(file_names, updated_centers_y, linestyle='-', color='b',label='our approach', markersize=2, linewidth=1)
# # 读取 CSV 文件
# import pandas as pd
# df = pd.read_csv('D:/pycharm/神经网络/idea/analyse/事件/C.csv')  # 将 'your_file.csv' 替换为你的 CSV 文件路径
#
# # 获取 'true_xx' 列的数据
# true_xx_values = df['B'].values
#
# # 生成横坐标
# x_values = [(1 + 6.6 * i) for i in range(301)]
#
# # 绘制图形
# plt.plot(x_values, true_xx_values,linestyle='-', color='r', label='100FPS machine', markersize=0.5, linewidth=0.5)
#
#
# df = pd.read_csv('D:/pycharm/神经网络/idea/analyse/事件/results2.csv')  # 将 'your_file.csv' 替换为你的 CSV 文件路径
#
# # 获取 'true_xx' 列的数据
# true_xx_values = df['yy'].values
#
# # 生成横坐标
# x_values = [20* i for i in range(100)]
#
# # 绘制图形
# plt.plot(x_values, true_xx_values, linestyle='-', color='y', label='50FPS video', markersize=2, linewidth=1)
#
#
# # 添加标题和标签
# plt.xlabel('Time(ms)')
# plt.ylabel('Vertical Pixel Position')
# plt.grid(True)
# ax.set_xlim(0,2000)
# # ax.set_xlim(1400, 1500)
# # ax.set_ylim(352,358)
# plt.legend()
# plt.show()