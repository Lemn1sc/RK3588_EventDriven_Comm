import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 第一步：准备数据 (这里生成虚拟的传感器数据)
# ==========================================
# 假设我们有 10000 条连续的数据（比如某种波形或温度）
data = np.sin(np.linspace(0, 100, 10000)) + np.random.normal(0, 0.1, 10000)

seq_length = 50  # 用过去 50 个时间步
features = 1  # 每个时间步只有 1 个特征（比如单纯的温度值）

X, Y = [], []
# 制造滑动窗口数据
for i in range(len(data) - seq_length):
    X.append(data[i: i + seq_length])  # 前50个数据作为输入
    Y.append(data[i + seq_length])  # 第51个数据作为预测目标

# 转换为 PyTorch 张量
# GRU 要求的输入形状是 (Batch, Seq_Len, Features)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # 形状变为 (9950, 50, 1)
Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # 形状变为 (9950, 1)

# 使用 DataLoader 批量加载数据
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# ==========================================
# 第二步：定义 GRU 模型
# ==========================================
class MyGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyGRUModel, self).__init__()
        # batch_first=True 意味着输入的维度是 (Batch, Seq, Feature)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 1. 数据进入 GRU
        out, _ = self.gru(x)
        # 2. 我们只需要序列最后一个时间步的输出去进行预测
        last_time_step_out = out[:, -1, :]
        # 3. 通过全连接层得到最终预测值
        prediction = self.fc(last_time_step_out)
        return prediction


# 实例化模型
# 输入特征数:1, 隐藏层大小:32, GRU层数:1, 预测输出数:1
model = MyGRUModel(input_size=1, hidden_size=32, num_layers=1, output_size=1)

# 如果有 GPU 就用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==========================================
# 第三步：设置损失函数(Loss)和优化器(Optimizer)
# ==========================================
criterion = nn.MSELoss()  # 回归任务通常用均方误差(MSE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam优化器最常用

# ==========================================
# 第四步：开始训练循环
# ==========================================
epochs = 20  # 整个数据集训练 20 遍

print(f"开始在 {device} 上训练...")
for epoch in range(epochs):
    total_loss = 0
    model.train()  # 设置为训练模式

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 1. 前向传播：让模型猜结果
        predictions = model(batch_x)

        # 2. 计算误差：对比猜的结果和真实结果
        loss = criterion(predictions, batch_y)

        # 3. 反向传播：更新模型参数（这三句是死记硬背的套路）
        optimizer.zero_grad()  # 清空旧梯度
        loss.backward()  # 计算新梯度
        optimizer.step()  # 更新参数

        total_loss += loss.item()

    # 每轮打印一次平均 Loss（Loss 越小说明越聪明）
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

# ==========================================
# 第五步：保存模型 (为了后续部署到 RK3588)
# ==========================================
torch.save(model.state_dict(), "my_trained_gru.pth")
print("训练完成！模型已保存为 my_trained_gru.pth")
