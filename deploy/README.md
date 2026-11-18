# 板端部署包 · PC/VM 侧准备清单

板子重装为 **Debian 12 / kernel 6.1.141 / librknnrt 2.3.2 / RKNPU 0.9.8**。
模型需要用 **rknn-toolkit2 2.3.2** 重转；板端需要 **rknn_toolkit_lite2 2.3.2**。
下面这些都可以在板子不在手边时先做完。

## 1. 下载文件

### VM（模型转换，Ubuntu 20.04 / Python 3.8）

在 `airockchip/rknn-toolkit2` 的 **v2.3.2 tag** 下：

```
rknn-toolkit2/packages/x86_64/
├── rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
└── requirements_cp38-2.3.2.txt
```

仓库：<https://github.com/airockchip/rknn-toolkit2/tree/v2.3.2>

国内镜像（GitCode）：<https://gitcode.com/gh_mirrors/rk/rknn-toolkit2>

> 如果 VM 不是 Python 3.8，就下对应 cpXX 的 wheel（`requirements_cpXX-2.3.2.txt` 同理）。
> 不要用 `master` 分支——它现在已经是 2.5.x，找 v2.3.2 tag。

### 板端（Debian 12 / Python 3.11）

```
rknn-toolkit-lite2/packages/
└── rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

可选：板子不能 apt 时，再下一个 cp311 aarch64 的 numpy wheel（`numpy<2`）。

## 2. VM 里重转（2026-11-17 实测路径）

VM 是 Ubuntu 20.04 / Python 3.8。VM 缺 `python3.8-venv` 时不用折腾 venv，
直接装到用户目录即可（会替换旧的 1.5.2，旧模型已经转好了，不需要它）：

```bash
# 1) 转换工具（wheel 用 pip download 从清华源拿）
python3 -m pip install --user --no-deps \
    ~/rknn_wheels/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 2) 2.3.2 的硬依赖
python3 -m pip install --user "torch==2.4.0+cpu" \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu

python3 -m pip install --user \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "onnx>=1.16.1,<1.17" "protobuf>=4.21.6,<=4.25.4" "onnxruntime==1.16.3"
```

实测踩坑：

- **`No module named 'torch'`**：2.3.2 的 `rknn_utils` 在 import 阶段硬依赖 torch
  （1.5.2 没这个问题），装 CPU 版 `torch==2.4.0+cpu` 即可；
- **`torch 2.4.0+cpu` 报 `<=2.4.0` 不兼容**：PEP 440 把 `+cpu` 当成比 `2.4.0` 大，
  实际是同一版本，忽略；
- **`build` 报 opset 19**：模型是 opset 19，onnxruntime 1.14 只支持到 opset 18，
  升到 **1.16.3**（Python 3.8 的最高可用版本）；
- **onnx/protobuf**：`onnx>=1.16.1`、`protobuf 4.21.6~4.25.4`；
- **`quantized_dtype`**：2.3.2 用 **`w8a8`**，不是 `asymmetric_quantized-8`；
- `setuptools` 太新 / `onnxoptimizer` 编译失败：先
  `apt install build-essential cmake`，必要时 `pip install "setuptools<82"`。

转换（输出用新文件名，别覆盖旧的 1.5.2 模型）：

```bash
cd /path/to/RK3588_EventDriven_Comm
python3 scripts/convert_to_rknn.py --out models/rknn/yolov8n_232.rknn
```

脚本会做 simulator vs onnxruntime 的逐图对比。判读标准：平均 IoU > 0.9、
最大 Δconf < 0.1。结果和旧的 1.5.2 接近就可以。

## 3. 组部署包

建一个目录（例如 `deploy/pkg/`），放：

```
pkg/
├── README_board.md
├── check_board_env.sh
├── board_check_rknn.py
├── board_decode_yolo.py
├── board_camera_edge.py
├── yolov8n_232.rknn
├── bus.jpg
└── wheels/
    └── rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

来源：

- `scripts/board_*.py`：本仓库；
- `yolov8n_232.rknn`：上一步转换产物；
- `bus.jpg`：`data/recordings/detect/bus.jpg`；
- wheel：第 1 步下载；
- `README_board.md` / `check_board_env.sh`：本目录。

打包：

```bash
tar -czf rk3588_deploy_$(date +%Y%m%d).tar.gz -C deploy/pkg .
```

把压缩包 + 一句话说明发给队友：**先跑 `check_board_env.sh`，把 `board_env.txt` 发回**。

## 4. 队友在板端要做的

见 `deploy/README_board.md`。核心顺序：

1. `sh check_board_env.sh > board_env.txt`
2. 装 `rknn_toolkit_lite2` 2.3.2
3. `board_check_rknn.py` 验证 `yolov8n_232.rknn`
4. 检查 `/dev/video*` 和 `cameraApp` 的 `ldd`
5. 跑 `board_camera_edge.py`，发回一张 `raw_000000.jpg`

## 5. 注意

- 旧 `/home/camera/cameraApp` 是 Buildroot 预编译的，Debian 12 上可能缺库；
  先 `ldd` 看，缺库就按 camera 仓库重编。
- 板子重装后 `/dev/video22` 可能变号，`v4l2-ctl --list-devices` 为准。
- 旧的 `models/rknn/yolov8n.rknn`（1.5.2）先保留，便于对照；确认新模型没问题后
  再决定是否替换入库。
