import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 训练阶段 ---

# 读取训练数据
train_file_path = 'extracted_data.csv'  # 替换为你的训练数据文件路径
train_data = pd.read_csv(train_file_path)

# 提取训练特征和目标值
train_features = train_data[['pitch', 'yaw', 'roll', 'Face_Center_X', 'Face_Center_Y', 'iris_x', 'iris_y']]
train_targets = train_data[['LEYEX', 'LEYEY', 'LEYEZ']]

# 数据归一化处理
scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)  # 归一化特征

# 转换为PyTorch张量
train_features_tensor = torch.tensor(train_features_scaled, dtype=torch.float32)
train_targets_tensor = torch.tensor(train_targets.values, dtype=torch.float32)

# 定义模型
class EulerAnglePredictor(nn.Module):
    def __init__(self):
        super(EulerAnglePredictor, self).__init__()
        self.fc1 = nn.Linear(7, 4)  # 输入7个特征
        self.dropout = nn.Dropout(0.5)  # Dropout 概率为 0.5
        self.fc2 = nn.Linear(4, 3)  # 输出3个欧拉角

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = EulerAnglePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    train_outputs = model(train_features_tensor)
    train_loss = criterion(train_outputs, train_targets_tensor)
    train_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}')

print("训练完成！")

# 保存模型和归一化参数
torch.save(model.state_dict(), 'model.pth')
np.save('scaler_mean.npy', scaler.min_)
np.save('scaler_scale.npy', scaler.scale_)


# --- 测试阶段 ---

# 读取测试数据
test_file_path = 'C1.csv'  # 替换为你的测试数据文件路径
test_data = pd.read_csv(test_file_path)

# 提取测试特征和目标值
test_features = test_data[['pitch', 'yaw', 'roll', 'Face_Center_X', 'Face_Center_Y', 'iris_x', 'iris_y']]
test_targets = test_data[['LEYEX', 'LEYEY', 'LEYEZ']]

# 加载归一化参数
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

# 对测试数据进行归一化
test_features_scaled = scaler.transform(test_features)

# 转换为PyTorch张量
test_features_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
test_targets_tensor = torch.tensor(test_targets.values, dtype=torch.float32)

# 加载模型
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 定义角度误差计算函数
def angular_error(g, g_hat):
    dot_product = torch.sum(g * g_hat, dim=1)
    norm_g = torch.norm(g, dim=1)
    norm_g_hat = torch.norm(g_hat, dim=1)
    cos_theta = dot_product / (norm_g * norm_g_hat)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angular_error_rad = torch.acos(cos_theta)
    angular_error_deg = angular_error_rad * (180.0 / np.pi)
    return torch.mean(angular_error_deg)

# 评估模型
with torch.no_grad():
    test_outputs = model(test_features_tensor)
    test_loss = nn.MSELoss()(test_outputs, test_targets_tensor)
    avg_angular_error = angular_error(test_targets_tensor, test_outputs)

print(f"测试集损失 (MSE): {test_loss.item():.4f}")
print(f"测试集平均角度误差 (度): {avg_angular_error.item():.4f}")