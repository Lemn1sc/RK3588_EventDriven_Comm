# 项目进度记录

> 最后更新: 2026-07-29

## 仓库分支

| 分支 | 内容 | 负责 |
|------|------|------|
| `main` | 默认分支（空，占位） | — |
| `event` | 事件端：8维事件向量、模型探查、环境搭建 | Lemn1sc |
| `gru` | GRU 控制端：事件链、规则引擎、GRU 训练与导出 | liang772635 |

---

## 已完成

### T0.1 环境搭建 ✅

- [x] `requirements.txt` / `environment.yml` — 依赖清单
- [x] `scripts/setup_env.bat` — Windows 一键建 conda 环境
- [x] `scripts/check_env.py` — 环境自检（numpy/cv2/torch/ultralytics + CUDA）
- [x] Python 3.10 + CUDA 版 PyTorch 跑通

### T0.2 模型输出探查 ✅

- [x] `scripts/inspect_models.py` — 一键跑 yolov8n（检测）和 yolov8n-pose（姿态）
- [x] `docs/07_模型输出结构_T0.2.md` — 两个模型的输出字段、shape、COCO 关键点顺序文档
- [x] 模型已下载到 `models/pt/`（yolov8n.pt, yolov8n-pose.pt）
- [x] 确认检测输出 `boxes.xyxyn/conf/cls`、姿态输出 `keypoints.xy/xyn/conf`

### T0.3 事件向量契约 ✅

- [x] `src/common/event_vector.py` — 8 维 float32 事件向量（EventType / EventVector / build_event / event_from_detections / event_from_result）
  - 向量布局: `[event_type, confidence, bbox_x1~y2, has_event, severity]`
  - severity = 置信度 × 高宽比异常度（站立≈0，躺卧→高）
  - 每帧只表征 severity 最高的主目标，非 person 类/低置信过滤
- [x] `scripts/check_event_vector.py` — 纯 numpy 契约验证（不依赖 ultralytics/torch）
- [x] `scripts/check_event_vector_real.py` — 真实模型输出验证

### GRU 控制端 ✅（gru 分支）

- [x] `event_chain_demo.py` — 事件链演示
- [x] `event_rule_engine.py` — 事件规则引擎
- [x] `forecast/` — GRU 模型训练与导出
  - `weather forecast.py` — GRU 训练脚本
  - `my_trained_gru.pth` — 训练好的 GRU 权重
  - `export_onnx.py` — 转 ONNX
  - `my_gru.onnx` — 导出的 ONNX 模型

---

## 待完成

### T0.4 localhost 全链路模拟

- [ ] `src/edge_node/` — 边缘端节点（读视频 → 检测 → 决策 → 发帧）
- [ ] `src/remote_node/` — 远端节点（收帧 → pose 姿态估计 → 跌倒判定）
- [ ] `src/transport/` — UDP 收发 + 优先级队列 + 轻量重传
- [ ] 双进程 pipeline 跑通

### T0.5 仿真器

- [ ] `src/simulator/` — 事件 + 信道仿真，生成 GRU 训练数据

### T0.6 GRU 控制端训练

- [ ] `src/controller/` — GRUOffloadController 定义 + 仿真数据监督训练

### T0.7 接入真实数据集

- [ ] `data/datasets/` — UR Fall Detection / Le2i Fall Detection

### T0.8 对比实验框架

- [ ] `experiments/` — 4 基线 + 指标统计 + 出图

### T0.9（可选）RKNN 转换验证

- [ ] WSL2 + rknn-toolkit2 + simulator 验证

---

## 关键设计决定

1. **事件端不做通信决策** — 只输出 8 维向量，GRU 控制端消费时序做卸载决策
2. **单帧只表征一个主目标** — 多人取 severity 最高者，非置信度最高者
3. **severity = 置信度 × 高宽比异常度** — 低置信自动打折，抑制鬼影框误报
4. **纯高宽比的局限** — 无法区分真跌倒与画面截断，需 GRU 利用时序区分
5. **远端 Pose 不适合 RL** — 分类问题用监督学习；GRU 控制端适合 RL 升级
