#!/usr/bin/env python3
"""板端 YOLOv8 RKNN 输出解码：DFL → NMS → 画框。

输入是 `board_check_rknn.py --save-npz` 产出的 npz（含 `input` 和 `out0..out8`），
以及原始图片。输出一张画框图，终端打印框坐标。

这个脚本不需要 rknnlite，只依赖 numpy + cv2，方便在板上直接看结果。

用法：
    python3 board_decode_yolo.py --npz board_outs.npz --image bus.jpg
    python3 board_decode_yolo.py --npz board_outs.npz --image bus.jpg \
        --conf 0.25 --iou 0.7 --out bus_detected.jpg
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")


def decode(outs, conf_thres=0.25, iou_thres=0.7):
    """9 输出（3 尺度 × box分布/类别分/概率和）→ xyxy 框、置信度、类别。"""
    import cv2

    outs = [np.asarray(o) for o in outs]
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
        d = (b * np.arange(16).reshape(1, 16, 1, 1)).sum(1)
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        score = slot["score"]
        c = score.max(0)
        m = c > conf_thres
        if not m.any():
            continue
        boxes.append(np.stack([
            (gx + .5 - d[0])[m] * stride, (gy + .5 - d[1])[m] * stride,
            (gx + .5 + d[2])[m] * stride, (gy + .5 + d[3])[m] * stride], 1))
        confs.append(c[m])
        clss.append(score.argmax(0)[m])

    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    B = np.concatenate(boxes)
    C = np.concatenate(confs)
    K = np.concatenate(clss)
    nms_boxes = np.stack([B[:, 0], B[:, 1], B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]], 1)
    keep = [int(i) for i in cv2.dnn.NMSBoxes(nms_boxes.tolist(), C.tolist(), conf_thres, iou_thres)]
    return B[keep], C[keep], K[keep]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--image", type=Path, default=None,
                   help="原始图片；不传则画在 640×640 letterbox 输入上")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--out", type=Path, default=Path("board_detected.jpg"))
    args = p.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    out_keys = sorted((k for k in npz.files if k.startswith("out")),
                      key=lambda s: int(s[3:]))
    outs = [npz[k] for k in out_keys]
    inp = npz["input"] if "input" in npz.files else None

    B, C, K = decode(outs, args.conf, args.iou)
    print(f"输出 {len(outs)} 个 · 框 {len(B)} 个（conf>{args.conf}, iou>{args.iou}）")

    import cv2
    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise SystemExit(f"读不到图片：{args.image}")
        h, w = img.shape[:2]
        size = int(inp.shape[1]) if inp is not None else 640
        r = min(size / h, size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        left, top = (size - nw) // 2, (size - nh) // 2

        # 640 letterbox 坐标 → 原图坐标
        if len(B):
            B = B.copy()
            B[:, [0, 2]] = (B[:, [0, 2]] - left) / r
            B[:, [1, 3]] = (B[:, [1, 3]] - top) / r
            B[:, [0, 2]] = np.clip(B[:, [0, 2]], 0, w - 1)
            B[:, [1, 3]] = np.clip(B[:, [1, 3]], 0, h - 1)
    else:
        img = cv2.cvtColor(inp[0].astype(np.uint8), cv2.COLOR_RGB2BGR) if inp is not None else None
        if img is None:
            raise SystemExit("npz 里没有 input，且没传 --image")

    for i, (b, c, k) in enumerate(zip(B, C, K)):
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"cls{int(k)} {c:.3f}"
        cv2.putText(img, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if i < 30:
            print(f"  [{i}] cls={int(k)} conf={c:.3f} xyxy=({x1},{y1},{x2},{y2})")

    cv2.imwrite(str(args.out), img)
    print(f"画框图 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
