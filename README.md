# 基于 RK3588 的事件驱动无线通信

> SRTP 科研训练项目 · 三端架构：事件端 → GRU 控制端 → WiFi 无线传输

## 一句话概括

本地 RK3588 运行轻量模型（YOLOv8n）做初筛，**GRU 控制端**根据事件时序与信道状态判断"本地能否搞定 / 是否值得传"，仅在**有意义的时刻**通过 WiFi 把任务事件驱动地卸载到远端（PC / 另一块板），从而相比周期性传输大幅降低带宽与延迟。

## 当前选定方案

**组合3：检测 + 姿态估计的级联卸载（跌倒检测 / 养老监护场景）**

- 事件端：`yolov8n` 检测"有没有人"（RK3588 NPU）
- 控制端：GRU 编码检测框时序变化，判断是否疑似异常（如跌倒）
- 卸载：仅疑似异常时把帧发给远端 `yolov8n-pose` 做姿态确认
- 优势：事件极度稀疏（99% 时间不通信），GRU 时序价值最明显，对比效果最悬殊

> 次选方案（组合1：轻量+重量级检测）用于最快跑通 pipeline，详见 `docs/02_技术方案.md`。

## 目录结构

> 图例：✅ 已实现　⬜ 待实现（骨架目录，只有 `.gitkeep`）

```
RK3588_EventDriven_Comm/
├── README.md / PROGRESS.md
├── docs/                       # 全部规划与实验记录（提问先看这里）
│   ├── 00_讨论记录摘要.md
│   ├── 01_项目规划.md          # 目标、范围、里程碑
│   ├── 02_技术方案.md          # 三端架构 + 组合1/2/3
│   ├── 03_开发路线图.md
│   ├── 04_无板子阶段任务.md    # ★ 任务清单 + T0.9 RKNN 转换结果
│   ├── 05_资源汇总.md
│   ├── 06_问题记录.md          # ★ 已知问题与踩坑
│   ├── 07_模型输出结构_T0.2.md
│   ├── 08_真实数据集首测_T0.7.md  # ★ 完整实验记录（§3~§14）
│   ├── 09_事件层接口.md        # ★ 事件流/可靠性/载荷 + 事件级结果（T0.10）
│   └── 汇报_event端工作进展.md
├── src/
│   ├── common/                 # ✅ 事件向量契约、缓存版本、评测指标、可靠性校准
│   ├── edge_node/              # 🔶 事件形成/载荷（T0.10）；检测+卸载决策待做
│   ├── remote_node/            # ⬜ 远端：姿态估计服务
│   ├── controller/             # ⬜ GRU 控制端（当前逻辑在 scripts/train_gru_fall.py）
│   ├── transport/              # ⬜ WiFi / UDP 传输
│   └── simulator/              # ⬜ 事件流 + 信道仿真器 —— **核心指标依赖它，目前为空**
├── experiments/
│   ├── baselines/              # ⬜ 周期/降采样/固定阈值/GRU 四条基线
│   └── results/                # ✅ 结果 JSON，文档里的数字都能从这里复算
├── scripts/                    # ✅ 31 个 py 脚本，分六类：
│   │                           #   数据准备  build_urfd_clean / prepare_urfd_views / make_rknn_calib
│   │                           #   特征训练  gru_features / train_gru_fall
│   │                           #   评测分析  analyze_lovo / calibrate_judge / best_single_frame_rule / mcf_*
│   │                           #   事件层    make_event_traces / eval_event_layer
│   │                           #   演示转换  demo_fall_detect / convert_to_rknn / detect_fall_video
│   │                           #   板端部署  board_check_rknn / board_decode_yolo / board_camera_edge
│   └── check_*.py              #   自检（9 个：契约/指标/缓存/特征/事件/校准/载荷）
├── models/{pt,onnx,rknn}/      # 模型；`rknn/yolov8n_232.rknn` 已入库（4.2MB，2.3.2），
│                               # 旧 `yolov8n.rknn`（5.4MB，1.5.2）保留对照；其余 gitignore
├── vm_share/{to_vm,from_vm}/   # Windows ↔ Ubuntu 虚拟机传文件（内容 gitignore）
├── data/{datasets,recordings}/ # 数据集与缓存，约 16 GB，全部 gitignore
├── notebooks/                  # ⬜ 空
└── configs/                    # ⬜ 空（超参仍硬编码在脚本顶部）
```

## 快速开始（无板子阶段）

见 `docs/04_无板子阶段任务.md`。核心路径：
1. PC 上跑通 `yolov8n` 检测 + `yolov8n-pose` 姿态
2. 用 localhost 双进程模拟"边缘→远端"全链路
3. 下载跌倒数据集，搭对比实验框架
4. 写事件仿真器 + 训练 GRU 控制端

## 快速开始（板端）

把 `scripts/board_*.py` 传到板端 `/home/model/` 后：

```bash
# 模型自检：load / init / infer / 延迟
python3 board_check_rknn.py --model yolov8n_232.rknn --image bus.jpg

# 终端 A：队友的采集端（有限流，100 帧）
cd /home/camera && ./cameraApp 640 480 100

# 终端 B：共享内存 NV12 → RKNN → DFL/NMS → 画框
cd /home/model && python3 board_camera_edge.py --model yolov8n_232.rknn --stride 2
```

> 当前检测画面的方向/置信度仍在排查；先用 `--rotate` / `--swap-uv` /
> `--save-raw` / `--dump-nv12` 诊断，见 docs/06「2026-11-16 · 板端摄像头 + RKNN」。

## 环境

| 用途 | 环境 | 说明 |
|---|---|---|
| **算法（Windows）** | conda `rk` 环境，Python 3.10.20<br>torch 2.3.1+cu121 · ultralytics 8.2.103 · opencv 4.10 | `conda` 不在 Git Bash 的 PATH 上，用 `scripts/rk.sh` 调 |
| **模型转换（Linux）** | **VMware Ubuntu 20.04**，Python 3.8<br>rknn-toolkit2 **2.3.2**（对应板端 librknnrt 2.3.2） | 工具链只有 Linux 版；输出 `models/rknn/yolov8n_232.rknn` |
| **目标板** | ATK-DLRK3588（NPU 6 TOPS），Buildroot 2021.11 / kernel 5.10.160 | 已接入；`librknnrt` 1.5.2 + `rknn_toolkit_lite2` 1.5.2（cp310 aarch64） |

> 两台机器之间用 [`vm_share/`](vm_share/README.md) 传文件（VMware 共享文件夹双向同步）。

### 常用命令

```bash
scripts/rk.sh scripts/demo_fall_detect.py --batch 5 --no-video   # 快速看效果
scripts/rk.sh scripts/gru_features.py list                       # 看有哪些特征集
scripts/rk.sh scripts/train_gru_fall.py --protocol lovo          # LOVO 验证

# 事件层（T0.10）：帧表 → 事件轨迹 → 事件级评测
scripts/rk.sh scripts/make_event_traces.py --dataset all
scripts/rk.sh scripts/eval_event_layer.py --dataset all
scripts/rk.sh scripts/eval_event_layer.py --dataset all --min-confidence 0.30  # 控制端工作点
```

> 详细的环境排障（PowerShell 的 `&`、`.bat` 编码、numpy 版本冲突）见
> [docs/04 T0.1「运行环境」](docs/04_无板子阶段任务.md)。
