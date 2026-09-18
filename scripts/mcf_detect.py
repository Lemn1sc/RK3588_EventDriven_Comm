#!/usr/bin/env python
"""在 MCF（Multiple Cameras Fall Dataset）上跑检测，产「逐帧是否检出人」的帧表。

── 为什么要做这个 ──

docs/08 §10 用 URFD 的 cam1 做过跨视角对比，但那个实验有三处硬伤：
① cam1 没有 ADL → 零负样本，AUC 算不出；② 只有 7 段有 cam1（后补齐到 30 段）；
③ 两视角时间对齐不可靠（6/30 段偏 ≥16 帧）。

MCF 把这三条一起解决了：8 个机位同拍一个房间，**标注是逐相机、逐帧给的**
（每个相机有自己的帧号），所以不需要估计任何对齐。而且同一段跌倒在每个机位都
发生一次 —— 可以直接回答「**同一次跌倒，哪个机位看得到人、哪个看不到**」。

这比「换个数据集还准不准」更根本：先问检测器在不在，再问时序模型好不好。

── 为什么只跑标注帧 ──

chute01~22 全量 188,513 帧，但被标注的只有 **15,624 帧（8%）**，且每个机位的
标注帧数完全相同（1,953 帧，其中 930 帧跌倒）—— 因为同一批动作在每个机位都被
标了。未标注的 92% 对「跌倒 vs 非跌倒」的判定没有信息，跑它们是纯浪费。
所以按标注区间 seek 过去读，工作量降到 1/12。

── 数据集的坑（2026-09-18 实测）──

1. **帧率标注是错的**：AVI 头部写 `dwMicroSecPerFrame=8333`（=120fps），但内容
   不符 —— chute01 里「站立→倾倒→落地→躺平」跨 55 帧，120fps 下只有 0.46 秒
   （人做不到），25fps 下 2.2 秒才合理。**实际约 25fps**，本脚本不依赖它
   （所有窗口都是按帧数定义的）。
2. **`_labels.csv` 是第三方重标，有错**：chute23 里有个 `cam=55`（应为 cam3），
   且被标了跌倒而官方说 chute23/24 是纯干扰；chute24 **完全没有标注**。
   所以本脚本默认只用 **chute01~22**。
3. 8 台相机都是**高处俯视**（广角鱼眼），没有 URFD cam0 那种平视视角 ——
   两者互补，不能互相替代。

用法：
    python scripts/mcf_detect.py --list          # 只看标注规模，不跑模型
    python scripts/mcf_detect.py                 # 全部跑（断点续跑）
    python scripts/mcf_detect.py --chutes 1 2    # 只跑指定场景
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DATA_DIR = ROOT / "data" / "datasets" / "MCF"
LABELS = DATA_DIR / "_labels.csv"
OUT_DIR = ROOT / "data" / "recordings" / "mcf"
FRAME_TABLE = OUT_DIR / "mcf_frames.v1.npz"

DEFAULT_CHUTES = list(range(1, 23))   # chute23/24 的标注不可用，见模块 docstring
CAMS = list(range(1, 9))

# 与 gru_features 的帧表同构，便于复用后续的特征配方
FRAME_COLS = [
    "index", "has_event", "event_type", "confidence", "severity",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "aspect_ratio",
    "pose_invoked", "torso_angle", "hip_ankle_ratio",
]
FRAME_DIM = len(FRAME_COLS)
C = {name: i for i, name in enumerate(FRAME_COLS)}

SPEC = {
    "format": "mcf-frames.v1",
    "columns": FRAME_COLS,
    "detector": "yolov8n.pt",
    "label_semantics": {"1": "fall", "0": "non-fall"},
    "units": "norm01",               # bbox 为 0~1 归一化坐标（与 URFD 帧表一致）
    "window_hint": "MCF 标注区间均为 31 帧，WINDOW=30 时每段约 2 个窗口",
    "source": "MCF chute01-22 x cam1-8, labeled spans only",
}


def read_labels(path: Path) -> tuple[dict[tuple[int, int], list[tuple[int, int, int]]], int]:
    """[(场景, 相机)] → [(起始帧, 结束帧, 标签)]，标签 1=跌倒 / 0=非跌倒。

    端点是**闭区间**：chute01/cam1 是 `1052-1082` / `1083-1113` / `1114-1144`，
    严丝合缝不重叠，说明约定如此。

    但重标质量参差：chute02/cam1 写成 `400-430` / `430-460`，**共用第 430 帧**。
    这类边界重叠按「后一段优先」处理 —— 把前一段的 end 截到 `下一段 start - 1`。
    一处只影响 1 帧，且**返回调整计数**让调用方报出来，不静默吞掉。
    """
    out: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ch = int(float(r["chute"]))
                cam = int(float(r["cam"]))
                s = int(float(r["start"]))
                e = int(float(r["end"]))
                lab = int(float(r["label"]))
            except (KeyError, ValueError):
                continue
            if cam not in CAMS:          # 挡下 cam=55 那个录入错误
                continue
            out[(ch, cam)].append((s, e, lab))

    adjusted = 0
    for key, spans in out.items():
        spans.sort()
        for i, (s, e, lab) in enumerate(spans[:-1]):
            nxt = spans[i + 1][0]
            if nxt <= e:                 # 边界重叠：前一段让位
                adjusted += e - nxt + 1
                spans[i] = (s, nxt - 1, lab)
        out[key] = [t for t in spans if t[1] >= t[0]]   # 截到空段的直接丢掉
    return dict(out), adjusted


def row_to_frame(row: dict) -> np.ndarray:
    f = np.full(FRAME_DIM, np.nan, dtype=np.float64)
    f[C["index"]] = row["frame"]
    f[C["has_event"]] = row["has_event"]
    f[C["event_type"]] = row["event_type"]
    f[C["confidence"]] = row["confidence"]
    f[C["severity"]] = row["severity"]
    f[C["bbox_x1"]] = row["bbox_x1"]
    f[C["bbox_y1"]] = row["bbox_y1"]
    f[C["bbox_x2"]] = row["bbox_x2"]
    f[C["bbox_y2"]] = row["bbox_y2"]
    f[C["aspect_ratio"]] = row["aspect_ratio"]
    f[C["pose_invoked"]] = row["pose_invoked"]
    f[C["torso_angle"]] = row["torso_angle_raw"]
    f[C["hip_ankle_ratio"]] = row["hip_ankle_ratio_raw"]
    return f


def run_one(cap, spans, det, pose, always_pose) -> tuple[np.ndarray, np.ndarray]:
    """按标注区间 seek 读取并逐帧检测。

    返回 (帧表, 标签数组)，两者按**读取顺序**对齐 —— 这个顺序就是区间顺序，
    不是原始帧号顺序。逐帧的真实帧号存在 `index` 列里，分析时以它为准。
    """
    import cv2
    import detect_fall_video as dfv

    frames, labels, n_expect = [], [], 0
    for s, e, lab in spans:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for k in range(e - s + 1):
            ok, img = cap.read()
            if not ok:
                break
            _, row = dfv.process_frame(img, det, pose, always_pose)
            row["frame"] = s + k          # 存**原始帧号**，不是数组下标
            frames.append(row_to_frame(row))
            labels.append(lab)
        n_expect += e - s + 1
    if len(frames) != n_expect:
        print(f"      ! 期望 {n_expect} 帧，实读 {len(frames)} 帧（seek 到文件尾了？）",
              file=sys.stderr)
    return (np.stack(frames) if frames else np.zeros((0, FRAME_DIM), dtype=np.float64),
            np.asarray(labels, dtype=np.int8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chutes", type=int, nargs="+", default=DEFAULT_CHUTES)
    ap.add_argument("--cams", type=int, nargs="+", default=CAMS)
    ap.add_argument("--always-pose", action="store_true",
                    help="每帧都跑姿态（慢很多，默认只在疑似时调）")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--list", action="store_true", help="只统计标注规模，不跑模型")
    args = ap.parse_args()

    if not LABELS.exists():
        raise SystemExit(
            f"找不到标注：{LABELS}\n"
            "  数据来源：https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset\n"
            "  解压后应为 data/datasets/MCF/chute01..24/cam1..8.avi + _labels.csv"
        )
    labs, adjusted = read_labels(LABELS)
    if adjusted:
        print(f"[!] 标注里有 {adjusted} 帧同时落在两个相邻区间内（原本被重复计数），"
              f"已按「后一段优先」归给后一段")

    jobs = [(ch, cam) for ch in args.chutes for cam in args.cams
            if (ch, cam) in labs and (DATA_DIR / f"chute{ch:02d}" / f"cam{cam}.avi").exists()]
    n_frames = sum(sum(e - s + 1 for s, e, _ in labs[k]) for k in jobs)
    n_fall = sum(sum(e - s + 1 for s, e, l in labs[k] if l == 1) for k in jobs)
    print(f"待跑：{len(jobs)} 个 (场景,机位)  标注帧 {n_frames:,}（跌倒 {n_fall:,}）")

    if args.list:
        per_cam = defaultdict(lambda: [0, 0])
        for ch, cam in jobs:
            for s, e, l in labs[(ch, cam)]:
                per_cam[cam][0] += e - s + 1
                per_cam[cam][1] += (e - s + 1) if l == 1 else 0
        print(f"\n{'机位':>5}{'标注帧':>10}{'跌倒帧':>10}")
        for cam in sorted(per_cam):
            a, b = per_cam[cam]
            print(f"{cam:>5}{a:>10,}{b:>10,}")
        return

    from ultralytics import YOLO
    import detect_fall_video as dfv

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[str, np.ndarray] = {}
    have: set[str] = set()
    if args.resume and FRAME_TABLE.exists():
        npz = np.load(FRAME_TABLE, allow_pickle=True)
        # 帧表和标签**都要留下**：早先版本只留帧表，续跑后已跑过的段就丢了标签，
        # 最终落盘的 npz 里那些段只剩帧、没有 label —— 静默少一半数据。
        done = {k: npz[k] for k in npz.files if not k.startswith("__")}
        have = {k for k in done if not k.endswith("_label")}
        print(f"续跑：已有 {len(have)} 段，跳过")

    det = YOLO(str(dfv.PT_DIR / "yolov8n.pt"))
    pose = YOLO(str(dfv.PT_DIR / "yolov8n-pose.pt"))
    todo = [(ch, cam) for ch, cam in jobs if f"chute{ch:02d}/cam{cam}" not in have]
    print(f"实跑 {len(todo)} 段 → {FRAME_TABLE.relative_to(ROOT)}\n")

    t0 = time.time()
    for i, (ch, cam) in enumerate(todo, 1):
        import cv2
        key = f"chute{ch:02d}/cam{cam}"
        cap = cv2.VideoCapture(str(DATA_DIR / f"chute{ch:02d}" / f"cam{cam}.avi"))
        tab, lab = run_one(cap, labs[(ch, cam)], det, pose, args.always_pose)
        cap.release()
        done[key] = tab
        done[f"{key}_label"] = lab

        el = time.time() - t0
        eta = el / i * (len(todo) - i)
        seen = int(tab[:, C["has_event"]].sum())
        print(f"  [{i}/{len(todo)}] {key:14s} {len(tab):5d} 帧  检出 {seen:5d}"
              f" ({seen / max(len(tab), 1):4.0%})  用时 {el:.0f}s  剩余 {eta / 60:.1f}min",
              flush=True)
        # 每段即落盘：跑一半崩了不用从头来
        save(FRAME_TABLE, done, n_jobs=len(done) // 2)

    print(f"\n完成：{len(done) // 2} 段 → {FRAME_TABLE.relative_to(ROOT)}")


def save(path: Path, arrays: dict, **meta) -> None:
    # 落盘前先验：每个帧表都必须有配对的 _label，且长度一致。
    # 少了标签的段在分析里会被整段忽略或算错，而文件本身看起来完全正常 ——
    # 这正是「跑出一份看起来对、其实少一半」的那类静默失效，宁可在写盘前炸掉。
    frames = {k for k in arrays if not k.endswith("_label")}
    for k in sorted(frames):
        lk = f"{k}_label"
        if lk not in arrays:
            raise SystemExit(f"[!] {k} 没有配对的 {lk} —— 帧表与标签不完整，拒绝落盘")
        if len(arrays[lk]) != len(arrays[k]):
            raise SystemExit(
                f"[!] {k} 有 {len(arrays[k])} 帧，但 {lk} 有 {len(arrays[lk])} 个标签"
            )
    stray = {k for k in arrays if k.endswith("_label") and k[:-6] not in frames}
    if stray:
        raise SystemExit(f"[!] 有标签却没有对应帧表：{sorted(stray)}")

    from common.cache import save_npz
    save_npz(path, SPEC, arrays, **meta)


if __name__ == "__main__":
    main()
