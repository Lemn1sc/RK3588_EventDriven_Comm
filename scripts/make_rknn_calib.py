#!/usr/bin/env python
"""为 RKNN int8 量化生成**校准集**。

── 为什么不能随便抽图 ──

RKNN 的 int8 量化靠校准集统计各层的激活范围。校准集若不能代表部署时的输入分布，
量化误差就会集中在**没被覆盖到的那部分输入**上。

而本项目的关键场景恰恰是少数：跌倒/躺地帧在整段监控里占比很低。如果只从"随便
抽几帧"校准，很可能抽到的全是站立场景 —— 量化后恰恰在**躺地**这一刻掉精度，
而那正是唯一需要报警的时刻。所以这里**按阶段分层抽样**，保证覆盖到。

分层依据是 URFD 官方逐帧标注：
    未躺地(label -1) / 正在跌倒(label 0) / 躺地(label 1)

产出：默认 `models/rknn/calib/`，200 张 640×640 的 JPEG（RKNN 读取时会再缩放，
存 640 是为了和模型输入一致，减少转换时的额外变换）。

用法：
    python scripts/make_rknn_calib.py                # 默认 200 张
    python scripts/make_rknn_calib.py --n 300
    python scripts/make_rknn_calib.py --data data/datasets/URFD_clean
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / "data" / "datasets" / "URFD" / "_official"
DEFAULT_DATA = ROOT / "data" / "datasets" / "URFD_clean"
DEFAULT_OUT = ROOT / "models" / "rknn" / "calib"

# 分层配比：跌倒相关的三个阶段要**刻意多给**，因为它们在真实监控里占比极低，
# 但恰恰是唯一需要报警的场景；日常段代表大多数时间的输入，也不能缺。
MIX = {          # (目录, label) -> 张数占比
    ("fall", -1): 0.20,     # 跌倒段里的"未躺地"帧（走进去、还没倒）
    ("fall", 0): 0.15,      # 正在跌倒（官方标注的 30 帧）
    ("fall", 1): 0.35,      # 躺地 —— 权重最高，最容易量化掉精度
    ("adl", -1): 0.25,      # 日常活动（大多数时间的输入分布）
    ("adl", 1): 0.05,       # 日常里的正常躺卧（ADL 也有 label=1）
}


def load_labels() -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = defaultdict(dict)
    for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        p = OFFICIAL / f
        if not p.exists():
            raise SystemExit(f"找不到官方标注：{p}\n  见 docs/08 §12.4")
        for r in csv.reader(open(p, encoding="utf-8")):
            if r and r[0]:
                out[r[0]][int(r[1])] = int(r[2])
    return dict(out)


def letterbox(frame, size=640):
    """等比缩放到 size×size，灰边填充（与推理时一致，避免缩放方式引入偏差）。"""
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=200, help="总张数")
    ap.add_argument("--size", type=int, default=640, help="输出边长")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"数据集不存在：{args.data}")
    labels = load_labels()

    # 先按 (目录, label) 收集候选帧号
    pool: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for split in ("fall", "adl"):
        for v in sorted((args.data / split).glob("*.mp4")):
            seq = v.stem.replace("-cam0", "")
            lab = labels.get(seq, {})
            cap = cv2.VideoCapture(str(v))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for i in range(n):
                L = lab.get(i + 1)               # 标注帧号从 1 开始
                if L is not None:
                    pool[(split, L)].append((str(v), i))

    if not pool:
        raise SystemExit("没有可用的标注帧 —— 检查数据和标注是否就位")

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    # 清掉旧文件，避免换参数后新旧混在一起（校准集混入旧图会让结果不可复现）
    for old in args.out.glob("*.jpg"):
        old.unlink()

    print(f"分层抽样（目标 {args.n} 张）→ {args.out}")
    total = 0
    for (split, L), frac in sorted(MIX.items(), key=lambda x: str(x[0])):
        want = int(round(args.n * frac))
        cands = pool.get((split, L), [])
        if not cands:
            print(f"  [!] {split} label={L:<3} 无候选帧，跳过")
            continue
        take = min(want, len(cands))
        idx = rng.choice(len(cands), take, replace=False)
        by_video: dict[str, list[int]] = defaultdict(list)
        for j in idx:
            v, i = cands[j]
            by_video[v].append(i)

        got = 0
        for v, frames in by_video.items():
            cap = cv2.VideoCapture(v)
            for i in sorted(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, fr = cap.read()
                if not ok:
                    continue
                name = f"{Path(v).stem}_f{i:04d}_L{L}.jpg"
                cv2.imwrite(str(args.out / name), letterbox(fr, args.size),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                got += 1
            cap.release()
        total += got
        print(f"  {split:<5} label={L:<3} 候选 {len(cands):>5}  取 {got:>4} 张"
              f"  （配比 {frac:.0%}）")

    print(f"\n完成：{total} 张 → {args.out}")
    print("  说明：文件名带 L{label}，便于核对分层是否如预期。")


if __name__ == "__main__":
    main()
