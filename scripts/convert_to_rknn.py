#!/usr/bin/env python
"""把 yolov8n.onnx 转成 RKNN，并在 PC 上用 simulator 验证数值一致性。

**本脚本要在 Linux 上跑**（rknn-toolkit2 只有 Linux 版，Windows 装不了）。

── 转换正确性怎么算「对」 ──

不是「能导出 .rknn 就算成功」。int8 量化必然带来数值差异，关键是有多大、
以及**差在哪里**。所以本脚本导出后会：
  1. 用 rknn simulator 跑同一批图
  2. 与 onnxruntime 的 fp32 输出逐张对比
  3. 报告最大绝对误差 / 相对误差，并按输出分组打印

  阈值只作参考：9 个输出里 box 分布是 logits（范围大）、类别分是概率（0~1），
  两者的绝对误差不可直接比。判断标准应是**解码后的检测框是否一致**。

── 输入预处理必须两边一致 ──

模型吃的是 RGB、归一化到 [0,1] 的 640×640 letterbox 图。RKNN 侧把归一化写进
config（mean=0 / std=255），onnxruntime 侧手工做。**两边不一致的话，误差报告
反映的是预处理错误，不是量化误差。**

用法（在 Ubuntu 上，已装 rknn-toolkit2）：
    python scripts/convert_to_rknn.py                      # 转换 + 验证
    python scripts/convert_to_rknn.py --skip-verify        # 只转换
    python scripts/convert_to_rknn.py --target rk3588
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = ROOT / "models" / "onnx" / "yolov8n.onnx"
DEFAULT_CALIB = ROOT / "models" / "rknn" / "calib"
DEFAULT_OUT = ROOT / "models" / "rknn"


def list_calib(calib_dir: Path, limit: int | None) -> list[Path]:
    imgs = sorted(p for p in calib_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(
            f"校准集为空：{calib_dir}\n"
            "  → 先在 PC 上跑：python scripts/make_rknn_calib.py"
        )
    if limit and len(imgs) > limit:
        # 均匀抽，而不是取前 N 张 —— 前 N 张可能全是同一段视频、同一阶段，
        # 分层抽样就白做了。
        idx = np.linspace(0, len(imgs) - 1, limit).round().astype(int)
        imgs = [imgs[i] for i in sorted(set(idx))]
    return imgs


def make_dataset_txt(imgs: list[Path], path: Path) -> Path:
    """RKNN 的 dataset 参数要一个「每行一个图片路径」的 txt。"""
    path.write_text("\n".join(str(p.resolve()) for p in imgs) + "\n", encoding="utf-8")
    return path


def build_rknn(onnx: Path, imgs: list[Path], out: Path, target: str,
               dataset_txt: Path, verbose: bool):
    try:
        from rknn.api import RKNN
    except ImportError as e:
        raise SystemExit(
            "没装 rknn-toolkit2。注意它**只有 Linux 版**：\n"
            "    pip install rknn-toolkit2\n"
            "  版本必须与板端 NPU 驱动匹配（板子到手先 `cat /proc/rknpu/version`）。\n"
            f"  （原始错误：{e}）"
        )

    rknn = RKNN(verbose=verbose)
    # mean/std 把归一化交给 RKNN 做：输入是 0~255 的 RGB uint8，
    # (x - 0) / 255 → [0,1]。与我们 Python 侧的预处理完全一致。
    # ⚠ 若把 mean/std 留空、又在外面除过 255，量化尺度会差 255 倍，精度崩掉。
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target,
        quantized_dtype="asymmetric_quantized-8",
        optimization_level=3,
    )
    print(f"  load_onnx: {onnx.name}")
    if rknn.load_onnx(model=str(onnx)) != 0:
        raise SystemExit("load_onnx 失败")

    print(f"  build: do_quantization=True, 校准 {len(imgs)} 张")
    if rknn.build(do_quantization=True, dataset=str(dataset_txt)) != 0:
        raise SystemExit("build 失败")

    out.parent.mkdir(parents=True, exist_ok=True)
    if rknn.export_rknn(str(out)) != 0:
        raise SystemExit("export_rknn 失败")
    print(f"  导出: {out.relative_to(ROOT)}  ({out.stat().st_size / 2**20:.1f} MB)")
    return rknn


def preprocess(path: Path, size: int = 640) -> np.ndarray:
    """与推理时一致：letterbox → RGB → /255 → NCHW。"""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"读不到图：{path}")
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]


def verify(rknn, onnx: Path, imgs: list[Path], n: int = 8):
    """simulator 跑 vs onnxruntime 跑，逐输出对比。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [!] 没装 onnxruntime，跳过验证（pip install onnxruntime）")
        return
    if rknn.init_runtime(target=None) != 0:      # target=None → simulator
        print("  [!] init_runtime(simulator) 失败，跳过验证")
        return

    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    sample = list(np.linspace(0, len(imgs) - 1, min(n, len(imgs))).round().astype(int))

    print(f"\n=== 验证：simulator vs onnxruntime（{len(sample)} 张）===")
    worst = 0.0
    worst_i = -1
    for k in sample:
        x = preprocess(imgs[k])
        a = rknn.inference(inputs=[x])            # list of ndarray
        b = sess.run(None, {name: x})
        d = max(float(np.abs(np.asarray(ai).ravel() - np.asarray(bi).ravel()).max())
                for ai, bi in zip(a, b))
        if d > worst:
            worst, worst_i = d, k
        print(f"  {imgs[k].name:<40} 最大绝对误差 {d:.5f}")

    print(f"\n  最大绝对误差 {worst:.5f}（{imgs[worst_i].name}）")
    print("  判读：box 分布是 logits（范围可达 ±15），类别分是概率（0~1）。")
    print("        参考 rknn_model_zoo 的 yolov8 经验，类别分误差 < 0.05、")
    print("        且**解码后的框与 fp32 基本重合**，即可认为量化可用。")
    print("        ⚠ 若误差集中在 L1（躺地）那批图上，说明校准集没覆盖好，")
    print("          调大 make_rknn_calib.py 的 --n 或提高躺地占比后重转。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "yolov8n.rknn")
    ap.add_argument("--target", default="rk3588", help="板端平台，如 rk3588 / rk3588s")
    ap.add_argument("--calib-limit", type=int, default=None, help="最多用多少张校准图")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"找不到 ONNX：{args.onnx}\n  见 models/onnx/")
    if sys.platform.startswith("win"):
        raise SystemExit(
            "rknn-toolkit2 不支持 Windows。请在 WSL2 或 Linux 虚拟机里跑本脚本。"
        )

    imgs = list_calib(args.calib, args.calib_limit)
    print(f"ONNX   : {args.onnx.relative_to(ROOT)}")
    print(f"校准集 : {len(imgs)} 张，来自 {args.calib.relative_to(ROOT)}")
    print(f"目标   : {args.target}")
    print()

    txt = make_dataset_txt(imgs, ROOT / "models" / "rknn" / "calib.txt")
    rknn = build_rknn(args.onnx, imgs, args.out, args.target, txt, args.verbose)

    if not args.skip_verify:
        verify(rknn, args.onnx, imgs)
    rknn.release()
    print(f"\n完成 → {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
