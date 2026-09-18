#!/usr/bin/env python
"""从 Kaggle 完整包重建**干净的 URFD cam0 RGB** 数据集。

── 为什么需要 ──

`data/datasets/URFD/{fall,adl}/*.mp4`（GitCode 镜像）**不是纯 RGB**，而是
「左半深度图 + 右半 RGB」的左右拼接（70 段全部如此，见 docs/08 §12）。
后果是人在右半幅里、横向只占一半宽，**`aspect = h/w` 被系统性放大一倍** ——
而项目的单帧触发器用的正是 `h/w`，于是「躺地」永远够不到阈值。

本脚本从 `urfd_full.zip` 的 `*-cam0-rgb-*.png`（原始 Kinect RGB，640×480）重转，
产出**不带拼接、不带纵向压扁**的纯 RGB 视频。

产出（默认 `data/datasets/URFD_clean/`）：
    fall/fall-01-cam0.mp4 ... fall-30-cam0.mp4
    adl/adl-01-cam0.mp4   ... adl-40-cam0.mp4

用法：
    python scripts/build_urfd_clean.py --list     # 只核对帧数，不转换
    python scripts/build_urfd_clean.py            # 转换
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = ROOT / "data" / "downloads" / "urfd_full.zip"
DEFAULT_OUT = ROOT / "data" / "datasets" / "URFD_clean"
OFFICIAL = ROOT / "data" / "datasets" / "URFD" / "_official"

FPS = 30.0   # URFD 原始 30fps（与现有 cam0 mp4 一致）
ENTRY = re.compile(r"^((?:fall|adl)-\d+)-cam0-rgb-(\d+)\.png$")


def scan(z: zipfile.ZipFile) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """{(类型, 序列): [(帧号, zip 路径)]}，按帧号排序。"""
    g: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for name in z.namelist():
        m = ENTRY.match(name.split("/")[-1])
        if not m:
            continue
        seq, idx = m.group(1), int(m.group(2))
        g[(seq.split("-")[0], seq)].setdefault(idx, name)
    return {k: sorted(v.items()) for k, v in g.items()}


def official_counts() -> dict[str, int]:
    """官方逐帧标注里每个序列的帧数 —— 用来核对转出来的视频没多没少。

    标注 CSV 没有表头，列是 `序列, 帧号, label, ...`。
    """
    out: dict[str, int] = {}
    for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        p = OFFICIAL / f
        if not p.exists():
            continue
        for r in csv.reader(open(p, encoding="utf-8")):
            if not r or not r[0]:
                continue
            out[r[0]] = max(out.get(r[0], 0), int(r[1]))
    return out


def convert(z: zipfile.ZipFile, frames: list[tuple[int, str]], out: Path) -> int:
    import cv2
    import numpy as np

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n = 0
    try:
        for _, path in frames:
            arr = cv2.imdecode(np.frombuffer(z.read(path), np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                print(f"      ! 无法解码 {path}，跳过", file=sys.stderr)
                continue
            if writer is None:
                h, w = arr.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         FPS, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"无法创建视频 {out}（OpenCV 没有可用的 mp4 编码器）")
            writer.write(arr)
            n += 1
    finally:
        if writer is not None:
            writer.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="只核对帧数，不转换")
    args = ap.parse_args()

    if not args.zip.exists():
        raise SystemExit(
            f"找不到压缩包：{args.zip}\n"
            "  curl -L -o data/downloads/urfd_full.zip \\\n"
            "    \"https://www.kaggle.com/api/v1/datasets/download/mireillejabbour/fall-dataset\""
        )

    with zipfile.ZipFile(args.zip) as z:
        groups = scan(z)
        off = official_counts()

        fall = sorted(k for k in groups if k[0] == "fall")
        adl = sorted(k for k in groups if k[0] == "adl")
        print(f"cam0 RGB：fall {len(fall)} 段 / adl {len(adl)} 段")

        mism = []
        for k in sorted(groups):
            seq = k[1]
            n = len(groups[k])
            if seq in off and off[seq] != n:
                mism.append((seq, n, off[seq]))
        if off:
            print(f"与官方逐帧标注的帧数核对：{'全部一致' if not mism else f'{len(mism)} 段不符'}")
            for s, a, b in mism[:8]:
                print(f"   {s}: 包内 {a} 帧 vs 官方标注 {b} 帧")
        else:
            print("（未找到官方标注，跳过帧数核对）")

        if args.list:
            return

        rel = args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out
        print(f"\n转换 → {rel}")
        total = 0
        for i, k in enumerate(sorted(groups), 1):
            grp, seq = k
            dst = args.out / grp / f"{seq}-cam0.mp4"
            n = convert(z, groups[k], dst)
            total += n
            print(f"  [{i}/{len(groups)}] {grp}/{seq}-cam0.mp4  {n:4d} 帧"
                  f"  ({dst.stat().st_size / 2**20:.1f} MiB)", flush=True)

    print(f"\n完成：{total:,} 帧 / {len(groups)} 段 → {rel}")


if __name__ == "__main__":
    main()
