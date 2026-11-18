#!/usr/bin/env python3
"""板端 RKNN 自检：模型能否 load / init / inference，输出形状与延迟。

这个脚本只做**运行性检查**，不做完整精度对比。它回答四个问题：

1. 板端 `rknnlite` 和 `librknnrt` 是否可用；
2. `.rknn` 能否 load、init_runtime；
3. 推理是否返回 9 个输出，形状是否符合 YOLOv8 的 3 尺度结构；
4. 单帧推理大概多少毫秒。

输入约定（与转换时一致）：
- NHWC `(1,640,640,3)`
- RGB
- 0~255
- **不要再除 255**（RKNN config 里已经 mean=0/std=255）

用法（在 RK3588 板上）：
    python3 scripts/board_check_rknn.py --model models/rknn/yolov8n.rknn
    python3 scripts/board_check_rknn.py --model yolov8n.rknn --image bus.jpg
    python3 scripts/board_check_rknn.py --model yolov8n.rknn --repeat 20 --save-npz out.npz

如果板上没有 OpenCV，可以不传 --image，脚本会用 114 灰的哑输入测试运行链路。
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "rknn" / "yolov8n.rknn"


def letterbox_rgb(image_path: Path, size: int) -> np.ndarray:
    """BGR 图片 → letterbox → RGB 0~255 NHWC (1,size,size,3)。"""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"读不到图片：{image_path}")

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    rgb = canvas[:, :, ::-1].astype(np.float32)  # BGR → RGB, 0~255
    return rgb[None]                              # NHWC


def dummy_input(size: int) -> np.ndarray:
    return np.full((1, size, size, 3), 114, dtype=np.float32)


def get_core_mask(rknn_lite, name: str):
    mapping = {
        "auto": "NPU_CORE_AUTO",
        "0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "0_1": "NPU_CORE_0_1",
        "0_1_2": "NPU_CORE_0_1_2",
    }
    attr = mapping[name]
    if not hasattr(rknn_lite, attr):
        print(f"[!] 当前 rknnlite 没有 {attr}，改用默认 core mask")
        return None
    return getattr(rknn_lite, attr)


def shape_summary(outputs) -> tuple[bool, bool]:
    """检查是否像 YOLOv8 的 9 输出（3 尺度 × 64/80/1 通道）。

    返回 (数量看起来对, 通道看起来对)。
    """
    shapes = [tuple(np.asarray(o).shape) for o in outputs]
    n_ok = len(outputs) == 9
    channels = set()
    hws = set()
    for s in shapes:
        if len(s) == 4:
            _, c, h, w = s
        elif len(s) == 3:
            c, h, w = s
        else:
            continue
        channels.add(int(c))
        hws.add((int(h), int(w)))
    ch_ok = {64, 80, 1}.issubset(channels) and len(hws) == 3
    return n_ok, ch_ok


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--image", type=Path, default=None,
                   help="测试图片；不传则用 114 灰的哑输入")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--core-mask", choices=["auto", "0", "1", "2", "0_1", "0_1_2"],
                   default="0_1_2")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--save-npz", type=Path, default=None,
                   help="把输入和 9 个输出存成 npz，供 PC 侧对比")
    args = p.parse_args()

    if not args.model.exists():
        print(f"✗ 模型不存在：{args.model}")
        print("  用法：python3 scripts/board_check_rknn.py --model /path/to/yolov8n.rknn")
        return 2

    try:
        from rknnlite.api import RKNNLite
    except Exception as exc:
        print(f"✗ 无法 import rknnlite：{type(exc).__name__}: {exc}")
        print("  板端需要安装 rknn_toolkit_lite2（aarch64 wheel），版本要与 librknnrt 匹配。")
        print("  也可以改用 C API（librknnrt）做同样的检查。")
        return 3

    if args.image is not None:
        if not args.image.exists():
            print(f"✗ 图片不存在：{args.image}")
            return 2
        inp = letterbox_rgb(args.image, args.size)
        print(f"[输入] {args.image} → {inp.shape} {inp.dtype} "
              f"range=[{inp.min():.1f},{inp.max():.1f}] RGB NHWC")
    else:
        inp = dummy_input(args.size)
        print(f"[输入] 哑输入（{args.size}×{args.size} 灰）→ {inp.shape} {inp.dtype}")

    rknn = RKNNLite()
    try:
        # 1) load
        print(f"[load] {args.model} ({args.model.stat().st_size / 2**20:.1f} MB)")
        ret = rknn.load_rknn(str(args.model))
        print(f"[load] ret={ret}")
        if ret != 0:
            print("✗ load_rknn 失败：检查模型是否完整、是否与板端 runtime 版本匹配。")
            return 4

        # 3) init
        core_mask = get_core_mask(RKNNLite, args.core_mask)
        if core_mask is None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(core_mask=core_mask)
        print(f"[init] core_mask={args.core_mask} ret={ret}")
        if ret != 0:
            print("✗ init_runtime 失败：检查 /proc/rknpu/version、librknnrt、驱动权限。")
            return 5
        try:
            print(f"[SDK] {rknn.get_sdk_version()}")
        except Exception:
            pass

        # 4) 预热 + 计时
        print(f"[warmup] {args.warmup} 次")
        for _ in range(max(0, args.warmup)):
            rknn.inference(inputs=[inp])

        times = []
        outputs = None
        print(f"[infer] {args.repeat} 次")
        for _ in range(max(1, args.repeat)):
            t0 = time.perf_counter()
            outputs = rknn.inference(inputs=[inp])
            times.append((time.perf_counter() - t0) * 1000.0)

        if not outputs:
            print("✗ inference 返回空输出")
            return 6

        print(f"\n[输出] {len(outputs)} 个")
        for i, o in enumerate(outputs):
            a = np.asarray(o)
            finite = np.isfinite(a).all() if a.dtype.kind == "f" else True
            print(f"  [{i}] shape={a.shape} dtype={a.dtype} "
                  f"min={a.min():.4g} max={a.max():.4g} finite={finite}")
            if a.dtype == np.int8:
                print("       ↑ int8 输出：后续 DFL 解码前必须按 output attrs 做反量化")

        n_ok, ch_ok = shape_summary(outputs)
        print(f"\n[形状] 9 输出={'✓' if n_ok else '✗'}  "
              f"通道集合(64/80/1)={'✓' if ch_ok else '✗'}")
        if not (n_ok and ch_ok):
            print("  ⚠ 输出不像 YOLOv8 的 3 尺度结构；检查是否加载了正确的模型/导出格式。")

        ms = np.array(times)
        print(f"[延迟] mean={ms.mean():.1f} ms  median={np.median(ms):.1f} ms  "
              f"min={ms.min():.1f} ms  max={ms.max():.1f} ms  "
              f"≈{1000.0 / ms.mean():.1f} FPS（仅 NPU 推理）")

        if args.save_npz:
            np.savez_compressed(args.save_npz, input=inp,
                                **{f"out{i}": np.asarray(o) for i, o in enumerate(outputs)})
            print(f"[save] {args.save_npz}")

        print("\n结论：模型可以 load / init / inference。")
        print("下一步：用 scripts/convert_to_rknn.py 的 decode() 解框，"
              "并与 PC 端同一张图对比 IoU。")
        return 0
    finally:
        try:
            rknn.release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
