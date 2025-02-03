import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义欧拉角（弧度）
leyex, leyey, leyez = -0.02125, -0.02756, 0.62705  # 第一个方向的欧拉角
reyex, reyey, reyez = 0.03817, -0.03178, 0.64100  # 第二个方向的欧拉角

# 定义一个从原点出发的单位向量（例如，Z轴方向的向量）
vector = np.array([0, 0, 1])

# 欧拉角转旋转矩阵的函数
def euler_to_rotmat(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

# 计算旋转后的向量
rotmat1 = euler_to_rotmat(leyex, leyey, leyez)
rotated_vector1 = rotmat1 @ vector

rotmat2 = euler_to_rotmat(reyex, reyey, reyez)
rotated_vector2 = rotmat2 @ vector

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制原向量
ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], length=1, normalize=True, label='Original Vector', color='r')

# 绘制第一个旋转后的向量
ax.quiver(0, 0, 0, rotated_vector1[0], rotated_vector1[1], rotated_vector1[2], length=1, normalize=True, label='Rotated Vector 1', color='b')

# 绘制第二个旋转后的向量
ax.quiver(0, 0, 0, rotated_vector2[0], rotated_vector2[1], rotated_vector2[2], length=1, normalize=True, label='Rotated Vector 2', color='g')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 添加图例
ax.legend()

# 显示图形
plt.show()