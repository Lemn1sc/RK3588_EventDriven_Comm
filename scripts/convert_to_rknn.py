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


def show_path(p: Path) -> str:
    """尽量显示相对仓库的路径；--out 传相对路径时不至于崩溃。"""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


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
            "  版本必须与板端 **librknnrt** 匹配：\n"
            "    板端 `strings /usr/lib/librknnrt.so | grep -i version`\n"
            "    RKNPU 驱动看 `/sys/kernel/debug/rknpu/version`"
            "（`/proc/rknpu/version` 可能不存在）。\n"
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
        # 2.x 用 w8a8；1.5.x 的写法是 asymmetric_quantized-8。
        quantized_dtype="w8a8",
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
    print(f"  导出: {show_path(out)}  ({out.stat().st_size / 2**20:.1f} MB)")
    return rknn


def preprocess(path: Path, size: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """letterbox → RGB。

    返回 **(RKNN 输入, onnxruntime 输入)** 两个张量 —— 它们**必须不同**：

      - RKNN 侧：喂 **0~255 的 RGB**。config 里已经设了 mean=0/std=255，
        归一化由 RKNN 自己做。若这里先除过 255，就会被除两次，输入变成
        0~0.004 的灰图 —— 验证结果会一片漆黑，但**不会报错**。
      - onnxruntime 侧：喂 **0~1 的 RGB**（ONNX 模型本身不含归一化）。

    两边必须是同一个 [0,1] 语义，误差才反映**量化**而不是预处理差异。
    """
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

    rgb255 = canvas[:, :, ::-1].astype(np.float32)          # RGB, 0~255
    rknn_in = rgb255[None]                                  # NHWC (1,640,640,3)
    ort_in = (rgb255 / 255.0).transpose(2, 0, 1)[None]      # NCHW, 0~1
    return rknn_in, ort_in


STRIDES = (8, 16, 32)


def decode(outs, conf_thres=0.25):
    """9 输出（3 尺度 × box分布/类别分/概率和）→ xyxy 框、置信度、类别。

    box 输出是 (1,64,H,W) 的 DFL logits：reshape 成 (4,16,H,W) 后对 16 个 bin
    做 softmax、再与 0..15 加权求和，得到「到 4 条边的距离」（网格单位）。
    ⚠ 若某版本的输出是 NHWC 排布（少见），这里会解不出框 —— 下面的形状检查
      会先把这类问题暴露出来，而不是给一个假的误差。
    """
    outs = [np.asarray(o) for o in outs]
    # 按形状把 9 个输出归到 3 个尺度：box=(·,64,H,W), score=(·,80,H,W), sum=(·,1,H,W)
    by_hw: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for o in outs:
        a = o[0] if o.ndim == 4 else o
        if a.ndim != 3:
            continue
        c, h, w = a.shape
        slot = by_hw.setdefault((h, w), {})
        if c == 64:
            slot["box"] = a
        elif c == 80:
            slot["score"] = a
        elif c == 1:
            slot["sum"] = a

    boxes, confs, clss = [], [], []
    for (h, w), slot in sorted(by_hw.items(), key=lambda kv: -kv[0][0]):
        if "box" not in slot or "score" not in slot:
            continue
        stride = 640 // h
        box = slot["box"].reshape(4, 16, h, w)
        b = np.exp(box - box.max(1, keepdims=True))
        b /= b.sum(1, keepdims=True)
        d = (b * np.arange(16).reshape(1, 16, 1, 1)).sum(1)      # (4,h,w)
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        score = slot["score"]
        c = score.max(0)
        m = c > conf_thres
        if not m.any():
            continue
        boxes.append(np.stack([(gx + .5 - d[0])[m] * stride, (gy + .5 - d[1])[m] * stride,
                               (gx + .5 + d[2])[m] * stride, (gy + .5 + d[3])[m] * stride], 1))
        confs.append(c[m])
        clss.append(score.argmax(0)[m])
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    B = np.concatenate(boxes); C = np.concatenate(confs); K = np.concatenate(clss)
    keep = []
    import cv2
    nms_boxes = np.stack([B[:, 0], B[:, 1], B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]], 1)
    for i in cv2.dnn.NMSBoxes(nms_boxes.tolist(), C.tolist(), conf_thres, 0.7):
        keep.append(int(i))
    return B[keep], C[keep], K[keep]


def match_boxes(b_ref, c_ref, b_hyp, c_hyp, iou_thr=0.5):
    """按 IoU 贪心匹配两批框，返回 (匹配数, 平均 IoU, 最大置信度差)。"""
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    used, ious, dconf = set(), [], 0.0
    for i in range(len(b_ref)):
        best, bj = 0.0, -1
        for j in range(len(b_hyp)):
            if j in used:
                continue
            v = iou(b_ref[i], b_hyp[j])
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= iou_thr:
            used.add(bj)
            ious.append(best)
            dconf = max(dconf, abs(float(c_ref[i]) - float(c_hyp[bj])))
    return len(ious), (float(np.mean(ious)) if ious else 0.0), dconf


def verify(rknn, onnx: Path, imgs: list[Path], n: int = 8):
    """simulator vs onnxruntime —— 比**解码后的框**，而不是原始张量。"""
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

    print(f"\n=== 先看两边输出的 dtype / 形状（确认没有重排或类型不一致）===")
    rknn_in, ort_in = preprocess(imgs[sample[0]])
    a0 = rknn.inference(inputs=[rknn_in])
    b0 = sess.run(None, {name: ort_in})
    print(f"  RKNN : {len(a0)} 个输出")
    for o in a0:
        print(f"    {str(np.asarray(o).shape):<20} {np.asarray(o).dtype}")
    print(f"  ONNX : {len(b0)} 个输出")
    for o in b0:
        print(f"    {str(np.asarray(o).shape):<20} {np.asarray(o).dtype}")
    print()

    print(f"=== 验证：解码后的检测框（{len(sample)} 张，IoU≥0.5 算匹配）===")
    print(f"{'图':<40}{'fp32框':>7}{'rknn框':>7}{'匹配':>6}{'平均IoU':>9}{'最大Δconf':>10}")
    worst = (1.0, -1)
    for k in sample:
        rknn_in, ort_in = preprocess(imgs[k])
        a = rknn.inference(inputs=[rknn_in])
        b = sess.run(None, {name: ort_in})
        br, cr, _ = decode(a)
        bo, co, _ = decode(b)
        nm, miou, dcf = match_boxes(bo, co, br, cr)
        print(f"  {imgs[k].name:<38}{len(bo):>7}{len(br):>7}{nm:>6}{miou:>9.3f}{dcf:>10.3f}")
        if miou < worst[0] or (miou == worst[0] and nm < len(bo)):
            worst = (miou, k)

    print()
    print("  判读标准：")
    print("    · 匹配数应 ≥ fp32 框数（允许 rknn 多报几个低分框）")
    print("    · 平均 IoU > 0.9 → 框基本重合，量化可用")
    print("    · 最大 Δconf < 0.1 → 置信度没被量化吃掉")
    print("    · 若 L1（躺地）那几张明显更差，回去加大 make_rknn_calib.py 的 --n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "yolov8n.rknn")
    ap.add_argument("--target", default="rk3588", help="板端平台，如 rk3588 / rk3588s")
    ap.add_argument("--calib-limit", type=int, default=None, help="最多用多少张校准图")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="跳过 build，直接加载已存在的 .rknn。"
                         "⚠ 只能连真机跑（simulator 不支持 load_rknn 出来的模型）")
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

    if args.reuse:
        # RKNN 限制：load_rknn() 出来的模型**不能在 simulator 上推理**，
        # 必须 init_runtime(target=<真机>) 才行。所以 --reuse 只对真机有意义。
        # 官方原话："RKNN model that loaded by 'load_rknn' not support inference
        # on the simulator, please set 'target' first!"
        raise SystemExit(
            "--reuse 只能配合真机使用（simulator 不支持 load_rknn 出来的模型）。\n"
            "  现在要做 PC 上的数值验证，请**去掉 --reuse**，走 load_onnx + build：\n"
            "      python3 scripts/convert_to_rknn.py\n"
            "  （重建约 40 秒，但只有这样才拿得到 simulator 的推理结果。）"
        )
    else:
        txt = make_dataset_txt(imgs, ROOT / "models" / "rknn" / "calib.txt")
        rknn = build_rknn(args.onnx, imgs, args.out, args.target, txt, args.verbose)

    if not args.skip_verify:
        verify(rknn, args.onnx, imgs)
    rknn.release()
    print(f"\n完成 → {show_path(args.out)}")


if __name__ == "__main__":
    main()
