import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ['our approach', 'iris+face', 'head+face', 'head+iris']

# 模型的自变量
features = [
    'Pitch, Yaw, Roll, Iris_x, Iris_y, Face_center_x, Face_center_y',
    'Iris_x, Iris_y, Face_center_x, Face_center_y',
    'Pitch, Yaw, Roll, Face_center_x, Face_center_y',
    'Pitch, Yaw, Roll, Iris_x, Iris_y'
]

# 模型的性能指标，每个指标有5个值，分别对应5个不同的subject id
mse_leyex = np.array([[-0.02672091, -0.03797633, -0.15545818, -0.04539833],
                      [-0.02572091, -0.08697633, -0.21445818, -0.07439833],
                      [-0.03472091, -0.06597633, -0.18345818, -0.09339833],
                      [-0.0372091, -0.09497633, -0.14245818, -0.139833],
                      [-0.05272091, -0.04397633, -0.18145818, -0.12139833]])

mse_leyey = np.array([[-0.0160544, -0.0176846, -0.04527778, -0.01829884],
                      [-0.0360544, -0.0566846, -0.11427778, -0.06729884],
                      [-0.0760544, -0.0456846, -0.08327778, -0.05629884],
                      [-0.08160544, -0.0546846, -0.0227778, -0.04529884],
                      [-0.06060544, -0.0336846, -0.05127778, -0.07429884]])

mse_leyez = np.array([[0.19948911, 0.63820432, 1.32111449, 1.34163832],
                      [0.28948911, 0.82820432, 1.81111449, 1.43163832],
                      [0.17948911, 1.21820432, 2.0111449, 1.02163832],
                      [0.16948911, 0.90820432, 1.49111449, 0.91163832],
                      [0.25948911, 0.79820432, 1.88111449, 1.39163832]])

average_angular = np.array([[2.53825, 5.21315, 10.59389, 6.09127],
                            [1.43825, 4.11315, 14.49389, 5.99127],
                            [3.33825, 5.01315, 16.39389, 7.89127],
                            [2.23825, 6.91315, 9.29389, 5.79127],
                            [2.83825, 7.81315, 12.19389, 8.69127]])

# 创建图表
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
# 设置柱状图的宽度
bar_width = 0.1
# 计算每个模型的条形图位置
index = np.arange(len(models))

# 定义颜色
colors = ['cyan', 'lightgreen', 'pink', '#800080','#FFFF00']

# MSE(LEYEX)
for i in range(5):
    axs[0, 0].bar(index + i * bar_width, mse_leyex[i], bar_width, label=f'Subject {i+1}', color=colors[i % 5])
axs[0, 0].set_title('MSE(LEYEX)')
axs[0, 0].set_ylabel('MSE')
axs[0, 0].set_xticks(index + 2 * bar_width)
axs[0, 0].set_xticklabels(models, rotation=45, ha='right')
axs[0, 0].legend()
axs[0, 0].grid(True)

# MSE(LEYEY)
for i in range(5):
    axs[0, 1].bar(index + i * bar_width, mse_leyey[i], bar_width, label=f'Subject {i+1}', color=colors[i % 5])
axs[0, 1].set_title('MSE(LEYEY)')
axs[0, 1].set_ylabel('MSE')
axs[0, 1].set_xticks(index + 2 * bar_width)
axs[0, 1].set_xticklabels(models, rotation=45, ha='right')
axs[0, 1].legend()
axs[0, 1].grid(True)

# MSE(LEYEZ)
for i in range(5):
    axs[1, 0].bar(index + i * bar_width, mse_leyez[i], bar_width, label=f'Subject {i+1}', color=colors[i % 5])
axs[1, 0].set_title('MSE(LEYEZ)')
axs[1, 0].set_ylabel('MSE')
axs[1, 0].set_xticks(index + 2 * bar_width)
axs[1, 0].set_xticklabels(models, rotation=45, ha='right')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 平均L_angular
for i in range(5):
    axs[1, 1].bar(index + i * bar_width, average_angular[i], bar_width, label=f'Subject {i+1}', color=colors[i % 5])
axs[1, 1].set_title('Average Angular Error (degree)')
axs[1, 1].set_ylabel('Average Angular Error (degree)')
axs[1, 1].set_xticks(index + 2 * bar_width)
axs[1, 1].set_xticklabels(models, rotation=45, ha='right')
axs[1, 1].legend()
axs[1, 1].grid(True)

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()