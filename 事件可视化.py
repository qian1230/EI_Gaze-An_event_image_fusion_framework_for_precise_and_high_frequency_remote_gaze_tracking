import numpy as np
import matplotlib.pyplot as plt

# 加载npz文件
data = np.load('./event2/470.npz')
x = data['x']
y = data['y']
p = data['p']
print(x[100],y[100])
# 定义区域的边界
x_min = 0
y_min =0
x_max = x_min + 1280 # 区域的宽度为72
y_max = y_min-720 # 区域的高度为24


# (38, 4) 10.30786418914795
# 筛选出在指定区域内的点
mask = (x >=x_min) & (x < x_max) & (y >= y_max) & (y < y_min)
selected_x = x[mask]
selected_y = y[mask]
selected_p = p[mask]
#print(selected_x.size)
# 创建绘图
plt.figure()

# 绘制p=1的点为红色
plt.scatter(selected_x[selected_p == 1], selected_y[selected_p == 1], color='red', s=0.05)

# 绘制p=0的点为蓝色
plt.scatter(selected_x[selected_p == 0], selected_y[selected_p == 0], color='blue', s=0.05)


# 设置坐标轴的范围为区域的大小
plt.xlim(x_min, x_max)
plt.ylim(y_max, y_min)

# 设置坐标轴的比例相同，以确保图像不会被拉伸
plt.gca().set_aspect('equal', adjustable='box')

plt.axis('off')
# 显示图像
plt.show()