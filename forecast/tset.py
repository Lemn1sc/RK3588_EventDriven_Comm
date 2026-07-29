import onnxruntime as ort
import numpy as np
import time
import math

# 1. 加载模型
session = ort.InferenceSession('my_gru.onnx')

# 2. 构造有规律的输入数据 (模拟正弦波的一段)
# 我们模拟从时间点 t=10.0 开始，步长为 0.01 的 50 个点
# (这个步长要和训练时的 np.linspace(0, 100, 10000) 对应，即 100/10000 = 0.01)
t_start = 10.0
step = 0.01
t_steps = np.arange(t_start, t_start + 50 * step, step)

# 生成这 50 个点作为输入
# 形状变换为 (1, 50, 1)
input_wave = np.sin(t_steps).astype(np.float32).reshape(1, 50, 1)

# 计算数学上的“标准答案” (即第 51 个点)
t_next = t_start + 50 * step
ground_truth = math.sin(t_next)

# 3. 运行推理
start_time = time.time()

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_wave})

end_time = time.time()

# 4. 打印结果对比
predicted_value = outputs[0][0][0]

print("--- 预测验证 ---")
print(f"输入序列起始时间: t={t_start}")
print(f"模型预测的第51个点 (t={t_next:.2f}): {predicted_value:.6f}")
print(f"数学计算的标准答案 (sin({t_next:.2f})): {ground_truth:.6f}")

# 计算误差
error = abs(predicted_value - ground_truth)
print(f"预测绝对误差: {error:.6f}")

print(f"推理耗时: {(end_time - start_time)*1000:.2f} ms")

if error < 0.1:
    print("结论：模型预测非常准确，它已经掌握了正弦波规律！")
else:
    print("结论：误差较大，可能需要增加训练轮数(Epochs)或检查数据归一化。")