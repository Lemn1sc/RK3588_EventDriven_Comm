# RK3588 板端部署包（Debian 12 / librknnrt 2.3.2）

这个包用于把事件端的检测模型和脚本部署到重装后的 RK3588 板子上。
旧系统是 Buildroot + librknnrt 1.5.2，新系统是 **Debian 12 / kernel 6.1.141 /
librknnrt 2.3.2 / RKNPU 0.9.8**，所以旧 `yolov8n.rknn` 不能直接沿用，
请使用包里的 **`yolov8n_232.rknn`**。

## 包里有什么

```
README_board.md                 本文件
check_board_env.sh              板端环境自检（只读）
board_check_rknn.py             模型 load/init/infer/延迟自检
board_decode_yolo.py            DFL 解码 + NMS + 画框
board_camera_edge.py            摄像头共享内存 → RKNN → 画框
yolov8n_232.rknn                按 rknn-toolkit2 2.3.2 重转的模型
bus.jpg                         模型测试图
wheels/                         板端 rknnlite wheel（及可选 numpy wheel）
```

## 0. 先跑环境自检

把整个目录拷到板子（例如 `/home/model/deploy/`），然后：

```bash
cd /home/model/deploy
sh check_board_env.sh > board_env.txt 2>&1
cat board_env.txt
```

把 `board_env.txt` 发回。重点确认：

- `python3 --version`（Debian 12 应为 3.11.x）
- `librknnrt` 是否为 **2.3.2**
- `rknnlite` 是否能 import
- `/dev/video*` 和 `v4l2-ctl --list-devices`
- `cameraApp` 的 `ldd` 是否缺库

## 1. 安装 rknnlite 2.3.2

包内 wheel：

```
wheels/
├── rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
├── numpy-1.26.4-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
├── psutil-7.2.2-...aarch64.whl
└── ruamel_yaml-0.19.1-py3-none-any.whl
```

Debian 12 的 pip 默认会拦系统安装（PEP 668），二选一：

```bash
# 推荐：venv
python3 -m venv /home/model/rknn_env
. /home/model/rknn_env/bin/activate
python3 -m pip install --no-index --find-links wheels \
    rknn-toolkit-lite2 numpy psutil ruamel.yaml

# 或者：直接装到系统 Python
python3 -m pip install --break-system-packages --no-index --find-links wheels \
    rknn-toolkit-lite2 numpy psutil ruamel.yaml
```

如果板子已经有 `python3-numpy`（Debian 12 自带 1.24.2），也可以只装 lite2：

```bash
python3 -m pip install --no-index --find-links wheels rknn-toolkit-lite2
```

验证（`get_sdk_version()` 要放在 `init_runtime()` 之后）：

```bash
python3 - <<'PY'
from rknnlite.api import RKNNLite
r = RKNNLite()
print("load:", r.load_rknn("yolov8n_232.rknn"))
print("init:", r.init_runtime())
try:
    print("sdk:", r.get_sdk_version())
except Exception as e:
    print("sdk:", e)
r.release()
PY
```

## 2. 模型自检

```bash
python3 board_check_rknn.py --model yolov8n_232.rknn --image bus.jpg
python3 board_check_rknn.py --model yolov8n_232.rknn --repeat 20 --save-npz board_outs.npz
python3 board_decode_yolo.py --npz board_outs.npz --image bus.jpg --out bus_detected.jpg
```

期望：

- `load / init` 都是 `ret=0`；
- 9 个 **float32** 输出（3 尺度 × 64/80/1）；
- `bus.jpg` 上 3 person + 1 bus；
- 单帧 60~70 ms（约 15 FPS）。

如果 `load_rknn` 返回非 0，把完整错误发回，不要继续往下跑。

## 3. 摄像头兼容性（先检查，不急着跑）

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
ldd /home/camera/cameraApp | grep "not found"
ldd /home/camera/camera_consumer | grep "not found"
```

- 如果 `/dev/video22` 不在了，把实际设备号发回；
- 如果 `ldd` 有 `not found`，说明 Buildroot 的预编译程序在 Debian 12 上缺库，
  需要按 camera 仓库重编（C11/CMake）。

重编时注意两处硬编码：

- `src/main.c:21`：`#define MAIN_VIDEO_DEV "/dev/video22"` → 改成实际设备号；
- `CMakeLists.txt`：`SDK_DIR="/home/karis/rk3588_linux_sdk"` → Debian 12 上
  要指向实际的 `librga` 头文件/库，或改成系统路径。

另外，`cameraApp` 的 RGA 旋转角固定 270°；如果新装后画面方向不对，
先记下来，后面可以用 `board_camera_edge.py --rotate` 补偿，或改 C 代码。

Debian 12 上可以直接本机编译（不用交叉工具链）：

```bash
cmake -S . -B build
cmake --build build -j
```

前提是 `gcc cmake make` 和 `librga` 的头文件/库已经就位。

## 4. 摄像头 + 检测闭环

先确认 `cameraApp` 能单独跑通，再：

**终端 A：**

```bash
cd /home/camera
./cameraApp 640 480 100
```

**终端 B：**

```bash
cd /home/model/deploy
python3 board_camera_edge.py --model yolov8n_232.rknn --stride 2 --out-dir camera_out
```

重点看：

- `[stream] total=100`
- `[infer] ... boxes=... conf=...`
- `[END] total=100 received=100 bad=0`
- `camera_out/frame_*.jpg` 的方向和颜色

如果检测置信度仍然只有 0.2~0.4、框贴左右边缘，先不要改代码，把
`camera_out/frame_000000.jpg` 和加 `--save-raw` 后的 `raw_000000.jpg`
发回。方向问题可以用 `--rotate 0/90/180/270` 快速试。

## 5. 需要发回的东西

1. `board_env.txt`
2. `bus_detected.jpg`（模型自检画框图）
3. `camera_out/frame_000000.jpg` + `raw_000000.jpg`（如果摄像头跑起来了）
4. 任何一步的完整报错
