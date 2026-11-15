# 项目进度记录

> 最后更新: 2026-11-15

## 仓库分支

| 分支 | 内容 | 负责 |
|------|------|------|
| `main` | 默认分支（空，占位） | — |
| `event` | 事件端：8维事件向量、模型探查、环境搭建 | Lemn1sc |
| **`event-refactor`** | **当前主线**（从 `event` 分出，35 个提交）：数据修正、跨视角验证、特征消融、阈值标定、RKNN 转换 | Lemn1sc |
| `gru` | GRU 控制端：事件链、规则引擎、GRU 训练与导出 | liang772635 |

> 2026-08-27 之后的工作全部在 `event-refactor`。**下面「已完成」的 T0.x 只记录到 8 月底**；
> 9~11 月的进展见本文末尾「2026-09~11 进展」。

## 2026-09~11 进展（摘要）

> 细节见 `docs/08` §10~§14 与 `docs/04` T0.9。

| 时间 | 工作 | 关键结论 |
|---|---|---|
| 09-11 | 修度量模块（ROC 并列 / 秩公式 AUC）、拆特征集、LOVO 验证 | 原 AUC 实现会随排序跳变，det 单帧真实 AUC 0.537 而非 0.715 |
| 09-12 | 换视角验证（URFD cam1） | 检出率 87.2% → 69.0%；仅 15/30 段两视角时间同步 |
| 09-18 | **发现并修正数据格式问题** | 旧 URFD 是 [深度\|RGB] **拼接图**，`aspect` 被放大一倍 → 重建干净数据全量重跑 |
| 09-18 | MCF 多机位接入 | **检测器在躺地时失效**：跌倒前 69.3% → 躺地 **9.4%**，8 机位一致 |
| 09-18 | 重标定 judge 阈值 + 消融 | 姿态**没有可测量增量**；单帧 `aspect` 规则被标定后与 GRU 打平 |
| 09-18 | MCF 交叉检验 | **GRU 的价值是跨视角鲁棒性**（单帧掉 0.21、GRU 掉 0.09） |
| 09-20~23 | 在线 demo + 决策延迟 | 中位 **0.80 s**（24 帧），28/30 段可检出 |
| 10-23 | **RKNN 转换** | ONNX 12.1MB → **5.4MB**，simulator 验证框 IoU **0.973~0.992** |
| 11-15 | 建 `vm_share/`、.rknn 入库 | 板端部署产物已就位 |

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

### T0.7 接入真实数据集（首测）✅

- [x] 数据就位：`data/datasets/URFD/`（30 跌倒 + 40 日常，GitCode 镜像下载，官方站/GitHub 国内不可达）
- [x] `scripts/detect_fall_video.py` — 视频级联检测（检测→事件向量→疑似才调姿态；`--always-pose` 全帧）
- [x] `scripts/bench_pose_fall.py` — 70 段全帧姿态批量基准
- [x] `docs/08_真实数据集首测_T0.7.md` — 首测记录（两个关键发现 + 3 个 bug 复盘）

**实测结论**（详见 docs/08）：
- 单帧高宽比触发器在 URFD 低位视角下漏检 ~90%（仅 3/30 达 severity≥0.5，8 段全程检不出人）
- 姿态判定单帧：67% 检出 / 20% 误报，fall 与 adl 占比分布重叠、无干净阈值
- → 触发策略留给 GRU（T0.6）；T0.4 固定规则需改时序触发

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

### T0.7 接入真实数据集（数据与首测已完成，见"已完成"）

- [x] `data/datasets/URFD/` — UR Fall Detection 一号摄像头视频已就位并跑通首测
- [ ] （可选）Le2i Fall Detection（含 Office 场景，作财务室代理；数据源见 docs/05、docs/08）

### T0.8 对比实验框架

- [ ] `experiments/` — 4 基线 + 指标统计 + 出图

### T0.9（可选）RKNN 转换验证 ✅ 已完成（2026-10-23）

- [x] VMware Ubuntu 20.04 + rknn-toolkit2 1.5.2（板端 librknnrt 1.5.2）
- [x] ONNX 12.1MB → `.rknn` **5.4MB**，int8 量化（200 张分层校准图）
- [x] simulator 验证：解码后框 **IoU 0.973~0.992**，躺地样本反而最稳
- [x] 产物入库：`models/rknn/yolov8n.rknn`
- [ ] 板端实测延迟与数值（需板子）

---

## 关键设计决定

1. **事件端不做通信决策** — 只输出 8 维向量，GRU 控制端消费时序做卸载决策
2. **单帧只表征一个主目标** — 多人取 severity 最高者，非置信度最高者
3. **severity = 置信度 × 高宽比异常度** — 低置信自动打折，抑制鬼影框误报
4. **纯高宽比的局限** — 无法区分真跌倒与画面截断，需 GRU 利用时序区分
5. **远端 Pose 不适合 RL** — 分类问题用监督学习；GRU 控制端适合 RL 升级
6. ~~**低位视角下纯高宽比连"真跌倒"都不触发** — URFD cam0 实测仅 3/30 段达 severity≥0.5~~
   ⚠️ **已作废**：那是基于旧**拼接图**数据的结论。干净数据上 **25/30 段**能触发、
   单帧基线 AUC **0.89**。原因见 docs/08 §12。
7. ~~**单帧姿态几何无法分开跌倒与日常躺卧/蹲坐** — 实测 67% 检出 / 20% 误报~~
   ⚠️ **已收窄**：干净数据 + 阈值标定后，单帧 `aspect` 规则 AUC **0.9708**，
   与 GRU（0.9692）**打平**；而**姿态没有可测量增量**（只用框 0.9604 > 加姿态 0.9458）。
   → 现在的主张不是「必须靠时序」，而是「**GRU 的价值在跨视角稳定性**」。
   见 docs/08 §14。
