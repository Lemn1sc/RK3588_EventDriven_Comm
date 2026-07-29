import torch
import torch.nn as nn

# 1. 必须重新定义一遍你的模型结构（要和训练时一模一样）
class MyGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        last_time_step_out = out[:, -1, :]
        prediction = self.fc(last_time_step_out)
        return prediction

# 2. 实例化模型
model = MyGRUModel(input_size=1, hidden_size=32, num_layers=1, output_size=1)

# 3. 加载你训练好的权重 (.pth文件)
model.load_state_dict(torch.load("my_trained_gru.pth"))
model.eval() # 切换到预测模式

# 4. 创建一个虚拟输入 (非常重要！)
# 维度必须是: (Batch_Size, Seq_Length, Features)
# 注意：RKNN 对动态形状支持不好，这里我们固定 Batch 为 1, 长度为 50, 特征为 1
dummy_input = torch.randn(1, 50, 1)

# 5. 导出为 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "my_gru.onnx",     # 导出的文件名
    export_params=True,
    opset_version=12,  # 建议使用 12，对 NPU 比较友好
    input_names=['input'],
    output_names=['output']
)

print("恭喜！你已经得到了 my_gru.onnx 文件。")