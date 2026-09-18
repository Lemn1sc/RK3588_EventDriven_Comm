#!/usr/bin/env python
"""把 Kaggle 版 URFD（PNG 图像序列）转成本项目管线能吃的 mp4。

为什么要单独做这一步：
- 现有 `data/datasets/URFD/*.mp4` 来自 GitCode 镜像，是 **640×240**（原始 640×480
  被裁掉了下半部分），且只有 cam0。
- Kaggle 版是 **640×480** 的原始 PNG 序列，且 fall-01..07 同时有 cam0 和 cam1。
- `cy_drop` 是「相对基线的比值」，取景一变分母就变。要回答「换视角结论还成立吗」，
  两个视角必须来自**同一批原始帧、同一套预处理** —— 所以 cam0 也重新从这里转，
  不复用旧 mp4。

产出（默认 `data/datasets/URFD_view/`）：
    cam0/fall/fall-01.mp4 ... fall-30.mp4
    cam1/fall/fall-01.mp4 ... fall-30.mp4

可以传**多个包**合并：完整包 `urfd_full.zip` 里 cam1 的 RGB 恰好缺 fall-01/02
（它有这两段的 cam1 深度图，只是没有 RGB）。

> fall-01/02 的 cam1 RGB 由一个补充包提供
> （`tanmaydacha/urfd-dataset`，2 GB），**该包已在本仓库清掉** —— 它 5032 帧里
> 只有这 270 帧是完整包没有的，其余 4762 帧与完整包 CRC 逐帧相同。这 270 帧
> 已经落成 `data/datasets/URFD_view/cam1/fall/fall-0{1,2}.mp4`。
> 若要从零重转，先把它下回来再用 `--zip` 一起传：
>     curl -L -o data/downloads/urfd_kaggle.zip \
>       "https://www.kaggle.com/api/v1/datasets/download/tanmaydacha/urfd-dataset"

用法：
    python scripts/prepare_urfd_views.py --list      # 只列出内容不转换
    python scripts/prepare_urfd_views.py             # 用默认的完整包转换
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FPS = 30.0          # URFD 原始帧率，与现有 cam0 mp4 一致（70 段全是 30.00）
DEFAULT_ZIPS = [ROOT / "data" / "downloads" / "urfd_full.zip"]
DEFAULT_OUT = ROOT / "data" / "datasets" / "URFD_view"

# fall-01-cam1-rgb-007.png → ('fall-01', 'cam1', 7)
ENTRY = re.compile(r"^(fall-\d+)-(cam[01])-rgb-(\d+)\.png$")


def scan(z: zipfile.ZipFile) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """{(序列, 视角): [(帧号, zip 内路径)]}，按帧号排序。

    Kaggle 包里有 `fall-02-cam1-rgb (1)` 这种带 ` (1)` 后缀的**重复目录**
    （浏览器重复下载留下的）。同一 (序列, 视角) 出现两次时只保留帧数更多的那个，
    否则会把副本和正本混在一起、帧号重复。
    """
    groups: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for name in z.namelist():
        base = name.split("/")[-1]
        m = ENTRY.match(base)
        if not m:
            continue
        seq, view, idx = m.group(1), m.group(2), int(m.group(3))
        key = (seq, view)
        if idx in groups[key] and groups[key][idx] != name:
            # 同一 (序列,视角,帧号) 有多个副本 → 按路径字典序留第一个，保证结果确定
            if name < groups[key][idx]:
                groups[key][idx] = name
        else:
            groups[key][idx] = name
    return {k: sorted((i, p) for i, p in v.items()) for k, v in groups.items()}


def scan_all(zips: list[Path]) -> dict[tuple[str, str], list[tuple[int, str, Path]]]:
    """合并多个包的扫描结果 → {(序列,视角): [(帧号, zip 内路径, 来自哪个包)]}。

    合并规则：**先给的包优先**，后给的包只补前者没有的 (序列,视角,帧号)。这样
    「完整包 + 补齐包」的重叠部分由完整包说了算 —— 两包重叠帧的 CRC 已比对过一致，
    但把规则写明确比依赖巧合可靠。

    每条帧自带来源包：实测里没有哪个序列是跨包拼的（fall-01/02 整体来自补齐包），
    但把来源记在每条上，就不必依赖这个假设成立。
    """
    merged: dict[tuple[str, str], dict[int, tuple[str, Path]]] = defaultdict(dict)
    for zp in zips:
        if not zp.exists():
            raise SystemExit(f"找不到压缩包：{zp}")
        with zipfile.ZipFile(zp) as z:
            for key, frames in scan(z).items():
                slot = merged[key]
                for idx, path in frames:
                    slot.setdefault(idx, (path, zp))
    return {k: sorted((i, p, zp) for i, (p, zp) in v.items()) for k, v in merged.items()}


def convert(zips: dict[Path, zipfile.ZipFile],
            frames: list[tuple[int, str, Path]], out: Path) -> int:
    """写 mp4。返回实际写出的帧数。"""
    import cv2
    import numpy as np

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n = 0
    try:
        for _, path, zp in frames:
            arr = cv2.imdecode(np.frombuffer(zips[zp].read(path), np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                print(f"      ! 无法解码 {path}，跳过", file=sys.stderr)
                continue
            if writer is None:
                h, w = arr.shape[:2]
                writer = cv2.VideoWriter(
                    str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
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
    ap.add_argument("--zip", type=Path, nargs="+", default=DEFAULT_ZIPS,
                    help="可以给多个包；先给的优先，后面的只补前面缺的帧")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="只列出内容，不转换")
    args = ap.parse_args()

    groups = scan_all(args.zip)

    views = sorted({v for _, v in groups})
    seqs = sorted({s for s, _ in groups})
    print(f"压缩包：{', '.join(z.name for z in args.zip)}")
    print(f"视角：{views}    序列：{len(seqs)} 段 ({seqs[0]} .. {seqs[-1]})")
    print()
    print(f"{'序列':<10}{'cam0 帧':>9}{'cam1 帧':>9}   帧对齐")
    for s in seqs:
        n0 = len(groups.get((s, "cam0"), []))
        n1 = len(groups.get((s, "cam1"), []))
        mark = "一致" if n0 == n1 else f"差 {abs(n0 - n1)}"
        print(f"{s:<10}{n0 or '-':>9}{n1 or '-':>9}   {mark}")

    only0 = [s for s in seqs if (s, "cam0") in groups and (s, "cam1") not in groups]
    only1 = [s for s in seqs if (s, "cam1") in groups and (s, "cam0") not in groups]
    pairs = [s for s in seqs if (s, "cam0") in groups and (s, "cam1") in groups]
    if only0:
        print(f"\n[!] 只有 cam0、没有 cam1 的序列（无法配对）：{only0}")
    if only1:
        print(f"\n[!] 只有 cam1、没有 cam0 的序列（无法配对）：{only1}")

    if args.list:
        return

    out_disp = args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out
    print(f"\n转换 → {out_disp}")
    zips = {zp: zipfile.ZipFile(zp) for zp in args.zip}
    try:
        total = 0
        for (seq, view), frames in sorted(groups.items()):
            out = args.out / view / "fall" / f"{seq}.mp4"
            n = convert(zips, frames, out)
            total += n
            print(f"  {view}/{seq}.mp4  {n} 帧  ({out.stat().st_size / 2**20:.1f} MiB)", flush=True)
    finally:
        for z in zips.values():
            z.close()

    print(f"\n完成：{total} 帧 / {len(groups)} 个视频")
    print(f"可用于配对比较（两视角都有）：{len(pairs)} 段")
    if only0:
        print(f"[!] 只有 cam0 的序列（无法配对）：{only0}")


if __name__ == "__main__":
    main()
