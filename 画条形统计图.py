import matplotlib.pyplot as plt
import numpy as np

# 数据准备
data = {
    "our appoach 7.03": [7.33, 6.89, 7.8, 6.10],
    "RGBE-Gaze 5.13": [5.5, 4.3, 6.1, 4.73],
    "MnistNet 7.18": [7.85, 7.1, 8.2, 5.83]
}

# 横坐标
x_labels = [5, 6, 7, 8]

# 计算每类的平均值
class_averages = {cls: np.mean(values) for cls, values in data.items()}

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["c", "blue", "pink"]
# 设置柱状图的宽度
bar_width = 0.1
index = np.arange(len(x_labels))

# 绘制每一类的柱状图
for i, (cls, values) in enumerate(data.items()):
    ax.bar(index + i * bar_width, values, bar_width, label=f"{cls}",color=colors[i])

# 添加图例
ax.legend()

# 设置横坐标标签
ax.set_xticks(index + bar_width)
ax.set_xticklabels(x_labels)

# 添加标题和标签

ax.set_xlabel("Subject id")
ax.set_ylabel("Average Angular Error(degree)")
plt.grid(True)
# 显示图表
plt.tight_layout()
plt.show()