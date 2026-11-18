# 项目进度记录

> 最后更新: 2026-11-16

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
| **11-16** | **事件层（T0.10）：事件形成 + 轨迹 + 事件级评测 + 载荷；`TRACK_LOST` 运动评分** | MCF 强告警召回 8.8% → `TRACK_LOST` 67.9%；conf≥0.30 时 54.2%/94.9% |
| **11-16** | **板端部署首测：RKNN load/init/infer + 摄像头共享内存 + 端到端闭环** | 65.6 ms/帧（15.2 FPS）；100 帧 0 bad；检测画面置信度偏低，方向/颜色待确认 |
| **11-17** | **板子重装 Debian 12 + 模型按 2.3.2 重转** | kernel 6.1.141 / `librknnrt` 2.3.2 / RKNPU 0.9.8；`yolov8n_232.rknn` **4.1 MB**，simulator IoU **0.973~0.992** |

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

### T0.10 事件层 ✅（2026-11-16）

- [x] `src/edge_node/event_tracker.py` — 逐帧证据 → 带生命周期的事件（纯 Python）
  - candidate → active → confirmed → expired；断档/TTL 关闭；只升不降；
  - `kind_first_frame` 记录“首次成为告警”的帧，避免把“人出现”误当提前量；
  - `TRACK_LOST` 作为独立事件：检测器失效本身变成信号。
- [x] `src/edge_node/payload.py` — metadata / ROI / FULL / CLIP 四档载荷
- [x] `src/common/calibration.py` — 可靠性曲线、Brier/ECE、保序回归
- [x] `scripts/make_event_traces.py` — URFD/MCF → 事件轨迹（逐帧证据 + 真值 + 预测 + 事件流）
- [x] `scripts/eval_event_layer.py` — 事件级召回/误报/提前量/分阶段/分机位/可靠性
- [x] 自检：`check_event_tracker.py` / `check_payload.py` / `check_calibration.py`

**第一版结果**（现有 0.25 帧表，详见 docs/09 §8.3）：

| 数据集 | 告警口径 | 召回 | 精确率 | 误报/min | 提前量中位 |
|---|---|---:|---:|---:|---:|
| URFD | 全部（默认） | 1.000 | 0.455 | 5.43 | +0.72 s |
| URFD | conf≥0.30 | 1.000 | 0.545 | 3.77 | +0.72 s |
| URFD | conf≥0.55 | 1.000 | 0.600 | 3.02 | +0.70 s |
| URFD | 强告警 | 0.900 | 0.750 | 1.36 | +0.90 s |
| MCF | 全部（默认） | 0.679 | 0.921 | 1.37 | +0.56 s |
| MCF | conf≥0.30 | 0.542 | 0.949 | 0.68 | +0.56 s |
| MCF | conf≥0.55 | 0.383 | 0.979 | 0.20 | +0.72 s |
| MCF | 强告警 | 0.088 | 1.000 | 0.00 | +0.84 s |

**关键**：MCF 单帧几何只覆盖 8.8% 的事件；`TRACK_LOST` 把它拉到 67.9%——这正是
“把检测器失效变成事件信号”的收益。`TRACK_LOST` 现在按“消失前变扁/下沉/是否贴边”
打分（`d_aspect`/`dy`/`border`），分数写进 `confidence`/`reliability`，`≥0.55` 时建议
紧急度 HIGH；默认保留全部事件，控制端按 `--min-confidence` 选工作点。URFD 在
conf≥0.55 时召回仍 100%、误报 5.43→3.02/min；MCF 在 conf≥0.30 时精确率 0.921→0.949，
召回 0.679→0.542。详见 docs/09 §4.1 §8.3。

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

## 待完成（**任务细节见 `docs/03_开发路线图.md`，此处只记状态**）

| 任务 | 状态 | 说明 |
|---|---|---|
| T0.4 localhost 全链路 | ⬜ | `src/edge_node/`、`remote_node/`、`transport/` 均为空 |
| T0.5 仿真器 | ⬜ | `src/simulator/` 为空 —— **五个核心指标全靠它，目前一个未测** |
| T0.6 GRU 控制端 | 🔶 | 训练与 LOVO 已在 `scripts/` 完成；未搬进 `src/controller/` |
| T0.7 真实数据集 | ✅ | URFD + MCF 已接入，官方逐帧标注已拿到 |
| T0.8 对比实验框架 | ⬜ | `experiments/baselines/` 为空 |
| T0.9 RKNN 转换 | 🔶 | 2.3.2 模型已转（IoU 0.973~0.992，4.1 MB）；Debian 12 板端待复测 |
| T0.10 事件层 | 🔶 | 事件流/轨迹/评测/载荷第一版完成；前景证据待补 |

> 8 月之后完成/推进的：**T0.7 ✅**、**T0.9 🔶**（板端已跑通，余优化项）、
> **T0.10 🔶**（第一版完成，前景证据待补）；其余见上面「2026-09~11 进展」。

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
