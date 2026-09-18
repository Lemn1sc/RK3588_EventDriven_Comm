"""第 3 步 · 原始帧表 + 特征集注册表。

── 为什么要拆 ──

原来「跑模型」和「算特征」是焊在一起的：`extract_features()` 一边跑 yolov8n
一边把结果直接变成 4 维特征，中间量（框的位置、面积）全丢了。后果是
**想换一个特征就得重跑一遍模型**（pose 那一路要 10~20 分钟），而项目文档自己写的
是"GRU 编码检测框时序（位置、大小、高宽比）"、docs/08 §7.1 还建议"相对近 30 帧
基线的突降比"—— 两条都没实现，因为实现成本太高。

现在拆成两段：

    跑模型  →  原始帧表 frames_<mode>.v1.npz   （贵；只在换模型/改判定时需要）
    帧表    →  特征缓存 feats_<featset>.npz     （纯数组运算，秒级，可反复跑）

`bbox` 进表是关键：位置和面积从此再也不会丢，任意新特征都能从这一张表重算，
**永不回头找模型**。

── 两张帧表 ──

    frames_det.v1   always_pose=False  只跑检测（贵在检测，快）
    frames_pose.v1  always_pose=True   检测 + 全帧姿态（10~20 分钟，唯一慢步骤）

两张表都由 `detect_fall_video.process_frame` 产出，与旧路径同源，
所以 `det.v1` / `pose.v1` 特征可以从表里精确重算出来 —— 这也是重构正确性的验收方式。

用法：
    python scripts/gru_features.py list
    python scripts/gru_features.py frames --mode det  [--only GLOB] [--resume] [--recompute]
    python scripts/gru_features.py frames --mode pose [--resume] [--dump-csv DIR]
    python scripts/gru_features.py feats --featsets det.v1 det.v2a det.v2 pose.v1 pose.v2
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Windows 控制台默认 GBK，打不出 ✓（沿用仓库其它脚本的做法）
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.cache import load_npz, save_npz  # noqa: E402
import detect_fall_video as dfv  # noqa: E402  复用 process_frame，保证与旧路径同源

DATA_DIR = ROOT / "data" / "datasets" / "URFD"
CACHE_DIR = ROOT / "data" / "recordings" / "detect_fall"

# 每个数据集一个缓存目录。切换数据集**必须**连缓存一起换：
# 干净 RGB 与旧拼接图产出的帧表**维度、列名、规格指纹全都一样**，只是数值不同 ——
# 放在同一目录下会静默命中旧缓存，跑出一份看起来完全正常、其实还是老数据的结果。
# 这正是 common/cache.py 开头说的第 3 类失效（读到别人的缓存），用目录隔离来防。
DATASET_CACHE = {
    "URFD": CACHE_DIR,                                  # 旧：GitCode 镜像（实为 [深度|RGB] 拼接图）
    "URFD_clean": ROOT / "data" / "recordings" / "detect_fall_clean",   # 新：Kaggle 原始 RGB
}


def set_dataset(data_dir: Path) -> None:
    """切换数据集根目录与其缓存目录（必须在任何用到 DATA_DIR 的函数之前调用）。"""
    global DATA_DIR, CACHE_DIR, FRAME_TABLE, FEAT_CACHE
    DATA_DIR = Path(data_dir)
    if not DATA_DIR.is_dir():
        raise SystemExit(f"数据集目录不存在：{DATA_DIR}")
    CACHE_DIR = DATASET_CACHE.get(DATA_DIR.name,
                                  ROOT / "data" / "recordings" / f"detect_fall_{DATA_DIR.name}")
    FRAME_TABLE = {m: CACHE_DIR / f"frames_{m}.v1.npz" for m in ("det", "pose")}
    FEAT_CACHE = {n: CACHE_DIR / f"feats_{n}.npz" for n in FEATSETS}
    print(f"[数据集] {rel(DATA_DIR)}   缓存 → {rel(CACHE_DIR)}")


# ══════════════════════════ 原始帧表 ══════════════════════════

# 每帧的原始量。`nan` 表示"这一帧没有这个量"（无人帧的框、没跑姿态的帧的几何量）。
FRAME_COLS = [
    "index", "has_event", "event_type", "confidence", "severity",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "aspect_ratio",
    "pose_invoked", "torso_angle", "hip_ankle_ratio",
]
FRAME_DIM = len(FRAME_COLS)
C = {name: i for i, name in enumerate(FRAME_COLS)}  # 按名取列，避免魔法数字

FRAME_TABLE = {"det": CACHE_DIR / "frames_det.v1.npz", "pose": CACHE_DIR / "frames_pose.v1.npz"}

# 原始量用 float64 存：文件才 MB 级，但换来的是「从表里重算出的 v1 特征与旧缓存
# 逐位一致」可以被严格验证 —— 用 float32 存会引入 ~1e-8 的表示误差，让"重构没改
# 行为"这件事永远只能"容差内一致"。
FRAME_DTYPE = np.float64

FRAME_SPEC = {
    "det": {"frameset": "det.v1", "columns": FRAME_COLS, "always_pose": False,
            "detector": "yolov8n.pt", "dtype": "float64",
            "storage": "raw geometry (rounding applied in v1 feature recipes)"},
    "pose": {"frameset": "pose.v1", "columns": FRAME_COLS, "always_pose": True,
             "detector": "yolov8n.pt", "posemodel": "yolov8n-pose.pt", "dtype": "float64",
             "storage": "raw geometry (rounding applied in v1 feature recipes)"},
}


def rel(p) -> Path | str:
    """尽量显示成仓库相对路径；不在仓库内时原样返回。

    报错信息本身绝不能因为路径转换再抛一次异常 —— 那样真正的病因就被埋了。
    """
    try:
        return Path(p).resolve().relative_to(ROOT)
    except (ValueError, OSError):
        return p


def require_inputs() -> None:
    """提取前的输入检查：数据/模型缺失时给出能照着做的提示。"""
    problems = []
    if not DATA_DIR.is_dir() or not any(DATA_DIR.glob("*/*.mp4")):
        problems.append(
            f"数据集缺失或为空：{rel(DATA_DIR)}\n"
            "      下载：https://gitcode.com/Universal-Tool/fa314 （URFD_video.zip，国内可直连）\n"
            "      解压出 URFD/{fall,adl}/*.mp4（30 段跌倒 + 40 段日常）放到该目录，详见 docs/08 §1"
        )
    for name in ("yolov8n.pt", "yolov8n-pose.pt"):
        if not (dfv.PT_DIR / name).exists():
            problems.append(f"模型缺失：{rel(dfv.PT_DIR / name)}"
                            "（ultralytics 首次加载时会自动下载到该目录）")
    if problems:
        raise SystemExit("输入检查未通过：\n  - " + "\n  - ".join(problems))


def video_list() -> list[tuple[str, float]]:
    """[(相对键, 标签)]，键形如 'fall/fall-01-cam0.mp4'。显式排序保证顺序确定。"""
    out = []
    for split in ("fall", "adl"):
        for v in sorted((DATA_DIR / split).glob("*.mp4")):
            out.append((f"{split}/{v.name}", 1.0 if split == "fall" else 0.0))
    return out


def row_to_frame(row: dict) -> np.ndarray:
    """process_frame 的一行日志 → 定长帧向量。

    ⚠ 几何量存**原始值**，不预先四舍五入。`process_frame` 把 torso_angle 写成
    1 位小数、hip_ankle_ratio 写成 3 位小数的字符串是给 CSV 看的；而它调用
    `judge()` 时用的是**原始** float。若表里存四舍五入后的值，阈值边界上就会翻转
    （实测 45.04° 存成 45.0，>45° 的判定从命中变不命中，导致 4/70 段单帧基线对不上）。
    需要旧精度的场合（复刻 v1 特征）在特征配方里再做一次同样的四舍五入即可。
    """
    f = np.full(FRAME_DIM, np.nan, dtype=FRAME_DTYPE)
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
    f[C["torso_angle"]] = row["torso_angle_raw"]                  # 原始精度（nan 表示缺失）
    f[C["hip_ankle_ratio"]] = row["hip_ankle_ratio_raw"]
    return f


def build_frame_table(mode: str, only: str | None, resume: bool, dump_csv: Path | None) -> dict:
    """跑模型产帧表。每段视频写完即落盘，中断后 --resume 可续。"""
    from ultralytics import YOLO

    require_inputs()
    path = FRAME_TABLE[mode]
    always_pose = bool(FRAME_SPEC[mode]["always_pose"])

    done: dict = {}
    if resume and path.exists():
        npz, _ = load_npz(path, FRAME_SPEC[mode], what=f"[{mode}] 帧表")
        done = {k for k in npz.files if not k.endswith("_label") and not k.startswith("__")}
        done = {k: npz[k] for k in done}
        print(f"续跑：已有 {len(done)} 段，跳过")

    det = YOLO(str(dfv.PT_DIR / "yolov8n.pt"))
    pose = YOLO(str(dfv.PT_DIR / "yolov8n-pose.pt"))
    videos = [(k, lab) for k, lab in video_list()
              if (only is None or Path(k).match(only)) and k not in done]

    print(f"产出帧表 [{mode}]：{len(videos)} 段 × 每帧"
          f"{'检测+姿态' if always_pose else '仅检测'} → {rel(path)}")
    t0 = time.time()
    for i, (key, label) in enumerate(videos, 1):
        import cv2

        cap = cv2.VideoCapture(str(DATA_DIR / key))
        rows = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, row = dfv.process_frame(frame, det, pose, always_pose)
            row["frame"] = len(rows)
            rows.append(row_to_frame(row))
        cap.release()
        done[key] = np.stack(rows) if rows else np.zeros((0, FRAME_DIM), dtype=FRAME_DTYPE)
        if dump_csv:
            dump_csv.mkdir(parents=True, exist_ok=True)
            out = dump_csv / (key.replace("/", "_").rsplit(".", 1)[0] + "_frames.csv")
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(FRAME_COLS)
                w.writerows(done[key])

        el = time.time() - t0
        eta = el / i * (len(videos) - i)
        print(f"  [{i}/{len(videos)}] {key:26s} {len(rows):4d} 帧  "
              f"用时 {el:.0f}s  预计剩余 {eta / 60:.1f}min", flush=True)
        # 每段即落盘：跑到第 60 段崩了也不用从头再来
        save_npz(path, FRAME_SPEC[mode],
                 {**done,
                  **{f"{k}_label": np.array([lab])
                     for k, lab in video_list() if k in done}},
                 n_videos=len(done), always_pose=always_pose)

    print(f"帧表完成：{len(done)} 段 → {rel(path)}")
    return done


def load_frame_table(mode: str) -> dict:
    npz, _ = load_npz(FRAME_TABLE[mode], FRAME_SPEC[mode], what=f"[{mode}] 帧表")
    keys = sorted(k for k in npz.files if not k.endswith("_label") and not k.startswith("__"))
    return {k: npz[k] for k in keys}


# ══════════════════════════ 特征集 ══════════════════════════

# ── 基线参数（det.v2 系列用）──
BASELINE_B = 30    # 因果基线的长度（只取当前帧之前、且有目标的最近 N 帧）
MIN_BASE = 5       # 有效历史不足该值时，相对量输出 0（中性）；只影响每段开头约 0.2s
MAX_GAP = 30       # 距上一个有目标帧超过该值视为新轨迹，基线清空重算

# 新特征共用的归一化常数（全部进指纹）
CY_BASE_MIN = 0.1        # cy 基线下限，防止除零放大
SIZE_BASE_MIN = 1e-3
CY_DROP_CLIP = [-1.0, 1.0]
SIZE_LOG2_DIV = 2.0
SIZE_LOG_CLIP = [-1.0, 1.0]

V1_CONF_CLIP = [0.0, 1.0]
V1_SEV_CLIP = [0.0, 1.0]
V1_ASPECT_CLIP = [0.3, 4.0]
V1_ASPECT_DIV = 4.0
V1_TORSO_DIV = 90.0
V1_HIP_CLIP = [0.0, 1.0]
V1_HIP_MISSING = 1.0


def _v1_common(t: np.ndarray, f: np.ndarray) -> None:
    """det.v1 的前 4 维：与旧 feats_from_event 逐位等价。"""
    has = t[:, C["has_event"]] > 0.5
    f[:, 0] = has
    f[:, 1] = np.clip(t[:, C["confidence"]], *V1_CONF_CLIP)
    f[:, 2] = np.clip(t[:, C["severity"]], *V1_SEV_CLIP)
    asp = np.clip(t[:, C["aspect_ratio"]], *V1_ASPECT_CLIP) / V1_ASPECT_DIV
    f[:, 3] = np.where(np.isnan(asp), 0.0, asp)


def _cy_size(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """框中心 y 与"线性尺度" sqrt(面积)。

    用 sqrt(面积) 而不是面积：面积是二次量，人起身/倒下时变化被平方放大，
    sqrt 后接近线性尺度，更接近物理直觉也更好学。
    """
    x1, y1, x2, y2 = (t[:, C[k]] for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
    cy = (y1 + y2) / 2.0
    size = np.sqrt(np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0))
    has = t[:, C["has_event"]] > 0.5
    cy = np.where(has, cy, np.nan)
    size = np.where(has, np.clip(size, 0.0, 1.0), np.nan)
    return cy, size


def causal_baseline(cy: np.ndarray, size: np.ndarray, has: np.ndarray):
    """因果基线的中位数序列。

    三个性质决定了特征是否有物理意义：

    1. **因果**（只取当前帧之前）：跌倒瞬间 `cy_drop` 从 0 跳到正的大值；人已经躺地
       不动之后，基线慢慢漂到躺姿、`cy_drop` 回落归零 —— "瞬态 vs 平稳"这个项目
       最核心的区分被**直接写进了特征**，而不是指望 GRU 从原始 cy 里自己悟出来。
    2. **中位数**：跌倒那几帧本身就是离群点，用均值会被自己污染。
    3. **断裂保护**：人有目标帧间断超过 MAX_GAP 视为新轨迹、基线清空。不加这一条
       会出真 bug —— 人消失很久后重现，旧基线会让 cy_drop 产生一个纯属虚构的尖峰。
    """
    T = len(cy)
    cy_base = np.full(T, np.nan)
    size_base = np.full(T, np.nan)
    hist_cy: list[float] = []
    hist_size: list[float] = []
    last_valid = -(10 ** 9)
    for i in range(T):
        if not has[i]:
            continue                      # 无人帧不更新基线，也不产出基线
        if i - last_valid > MAX_GAP:      # 断裂：视为新轨迹
            hist_cy.clear()
            hist_size.clear()
        if len(hist_cy) >= MIN_BASE:
            cy_base[i] = float(np.median(hist_cy))
            size_base[i] = float(np.median(hist_size))
        hist_cy.append(float(cy[i]))
        hist_size.append(float(size[i]))
        hist_cy = hist_cy[-BASELINE_B:]
        hist_size = hist_size[-BASELINE_B:]
        last_valid = i
    return cy_base, size_base


def _v2_extra(t: np.ndarray, f: np.ndarray, off: int) -> None:
    """det.v2 的后 4 维：位置、尺度、以及相对因果基线的两个突变比。

    旧特征把框在哪、多大全丢了，而 URFD cam0 是低位视角 —— docs/08 §3 实测跌倒时
    高宽比几乎不变（30 段里只有 3 段 severity≥0.5），也就是说 v1 里唯一带时序价值的
    aspect 维在 cam0 上近乎噪声，**真正可靠的线索（框中心下沉）恰恰被丢了**。
    """
    has = t[:, C["has_event"]] > 0.5
    cy, size = _cy_size(t)
    cy_base, size_base = causal_baseline(cy, size, has)

    cy0 = np.where(has, cy, 0.0)
    size0 = np.where(has, size, 0.0)
    f[:, off + 0] = np.nan_to_num(cy0)
    f[:, off + 1] = np.nan_to_num(size0)

    # 相对基线的中心下沉比：除以基线 → 与取景、人在画面中的绝对位置无关，
    # 换摄像头不用重调阈值。
    drop = (cy_base - cy) / np.maximum(cy_base, CY_BASE_MIN)
    f[:, off + 2] = np.where(np.isnan(cy_base), 0.0, np.clip(drop, *CY_DROP_CLIP))
    # 尺度突变用 log2 而不是线性比值：缩小一半 → -0.5、放大一倍 → +0.5 是对称的，
    # 线性比值下 0.5 与 2.0 严重不对称，会给训练引入偏置。
    ratio = np.log2(np.maximum(size, SIZE_BASE_MIN) / np.maximum(size_base, SIZE_BASE_MIN))
    f[:, off + 3] = np.where(np.isnan(size_base), 0.0,
                             np.clip(ratio / SIZE_LOG2_DIV, *SIZE_LOG_CLIP))


def build_det_v1(t):
    f = np.zeros((len(t), 4), dtype=np.float32)
    _v1_common(t, f)
    return f


def build_det_v2a(t):
    """det.v1 + 位置 + 尺度（只补"框在哪、多大"）。"""
    f = np.zeros((len(t), 6), dtype=np.float32)
    _v1_common(t, f)
    has = t[:, C["has_event"]] > 0.5
    cy, size = _cy_size(t)
    f[:, 4] = np.nan_to_num(np.where(has, cy, 0.0))
    f[:, 5] = np.nan_to_num(np.where(has, size, 0.0))
    return f


def build_det_v2(t):
    """det.v1 + 位置 + 尺度 + 相对因果基线的中心下沉比 / 尺度突变比。"""
    f = np.zeros((len(t), 8), dtype=np.float32)
    _v1_common(t, f)
    _v2_extra(t, f, off=4)
    return f


def _round_like_old(x: np.ndarray, nd: int) -> np.ndarray:
    """按旧代码写日志时的精度（torso 1 位、hip_ankle 3 位）再取一次。

    旧 `feats_from_pose` 是从**日志行**取几何量的，而日志行里存的是四舍五入结果
    （原始值只用于 judge 判定）。所以要逐位复刻 v1 特征，这一步四舍五入不能省。
    """
    return np.array([float(f"{v:.{nd}f}") if np.isfinite(v) else np.nan for v in x],
                    dtype=np.float64)


def _pose_dims(t: np.ndarray, f: np.ndarray, off: int) -> None:
    """姿态 2 维（躯干倾角归一 + 髋踝高差比）。

    关键点缺失时按"直立"处理（与旧代码一致：hip_ankle 缺失 → 1.0）。
    """
    torso = _round_like_old(t[:, C["torso_angle"]], 1) / V1_TORSO_DIV
    f[:, off + 0] = np.nan_to_num(np.clip(torso, 0.0, 1.0))
    hip = _round_like_old(t[:, C["hip_ankle_ratio"]], 3)
    f[:, off + 1] = np.where(np.isnan(hip), V1_HIP_MISSING,
                             np.clip(np.nan_to_num(hip), *V1_HIP_CLIP))


def build_pose_v1(t):
    """与旧 feats_from_pose 逐位等价。"""
    f = np.zeros((len(t), 6), dtype=np.float32)
    _v1_common(t, f)
    _pose_dims(t, f, off=4)
    return f


def build_pose_v2(t):
    """det.v2 的 8 维 + 2 维姿态量。

    刻意做成"det.v2 + 姿态附加"：这样 det.v2 与 pose.v2 的对比里 8 维完全相同、
    只差 2 个姿态量，才是公平的消融；否则改了 det 没改 pose，"det ≈ pose"那个
    结论就成了特征工程不对称的产物。
    """
    f = np.zeros((len(t), 10), dtype=np.float32)
    _v1_common(t, f)
    _v2_extra(t, f, off=4)
    _pose_dims(t, f, off=8)
    return f


@dataclass(frozen=True)
class FeatSet:
    name: str
    frames: str                 # 依赖哪张原始帧表
    names: tuple[str, ...]
    spec: dict                  # 进指纹的一切常数
    build: Callable[[np.ndarray], np.ndarray]

    @property
    def mode(self) -> str:
        return self.name.split(".")[0]

    @property
    def dim(self) -> int:
        return len(self.names)


_COMMON_SPEC = {"baseline_B": BASELINE_B, "min_base": MIN_BASE, "max_gap": MAX_GAP,
                "cy_base_min": CY_BASE_MIN, "size_base_min": SIZE_BASE_MIN,
                "cy_drop_clip": CY_DROP_CLIP, "size_log2_div": SIZE_LOG2_DIV,
                "size_log_clip": SIZE_LOG_CLIP, "no_event_value": 0.0}

FEATSETS: dict[str, FeatSet] = {
    "det.v1": FeatSet(
        "det.v1", "det",
        ("has_event", "confidence", "severity", "aspect_norm"),
        {"featset": "det.v1", "dim": 4, "frames": "det.v1",
         "conf_clip": V1_CONF_CLIP, "sev_clip": V1_SEV_CLIP,
         "aspect_clip": V1_ASPECT_CLIP, "aspect_div": V1_ASPECT_DIV},
        build_det_v1),
    "det.v2a": FeatSet(
        "det.v2a", "det",
        ("has_event", "confidence", "severity", "aspect_norm", "cy", "size"),
        {"featset": "det.v2a", "dim": 6, "frames": "det.v1", "extends": "det.v1",
         "conf_clip": V1_CONF_CLIP, "sev_clip": V1_SEV_CLIP,
         "aspect_clip": V1_ASPECT_CLIP, "aspect_div": V1_ASPECT_DIV,
         "size_form": "sqrt(w*h) clipped [0,1]", **_COMMON_SPEC},
        build_det_v2a),
    "det.v2": FeatSet(
        "det.v2", "det",
        ("has_event", "confidence", "severity", "aspect_norm",
         "cy", "size", "cy_drop", "size_log_ratio"),
        {"featset": "det.v2", "dim": 8, "frames": "det.v1", "extends": "det.v2a",
         "conf_clip": V1_CONF_CLIP, "sev_clip": V1_SEV_CLIP,
         "aspect_clip": V1_ASPECT_CLIP, "aspect_div": V1_ASPECT_DIV,
         "size_form": "sqrt(w*h) clipped [0,1]", **_COMMON_SPEC},
        build_det_v2),
    "pose.v1": FeatSet(
        "pose.v1", "pose",
        ("has_event", "confidence", "severity", "aspect_norm",
         "torso_tilt_norm", "hip_ankle_ratio"),
        {"featset": "pose.v1", "dim": 6, "frames": "pose.v1",
         "conf_clip": V1_CONF_CLIP, "sev_clip": V1_SEV_CLIP,
         "aspect_clip": V1_ASPECT_CLIP, "aspect_div": V1_ASPECT_DIV,
         "torso_div": V1_TORSO_DIV, "hip_clip": V1_HIP_CLIP,
         "hip_missing": V1_HIP_MISSING},
        build_pose_v1),
    "pose.v2": FeatSet(
        "pose.v2", "pose",
        ("has_event", "confidence", "severity", "aspect_norm",
         "cy", "size", "cy_drop", "size_log_ratio",
         "torso_tilt_norm", "hip_ankle_ratio"),
        {"featset": "pose.v2", "dim": 10, "frames": "pose.v1", "extends": "det.v2",
         "conf_clip": V1_CONF_CLIP, "sev_clip": V1_SEV_CLIP,
         "aspect_clip": V1_ASPECT_CLIP, "aspect_div": V1_ASPECT_DIV,
         "torso_div": V1_TORSO_DIV, "hip_clip": V1_HIP_CLIP,
         "hip_missing": V1_HIP_MISSING,
         "size_form": "sqrt(w*h) clipped [0,1]", **_COMMON_SPEC},
        build_pose_v2),
}

FEAT_CACHE = {n: CACHE_DIR / f"feats_{n}.npz" for n in FEATSETS}


# ══════════════════════ 单帧基线（从帧表重算） ══════════════════════
#
# 单帧基线是"不接时序、只看当前帧"的对照方法。它原来有两套口径：
#   det 模式用 severity≥0.5，pose 模式用 judge 的融合判定（含框线索），
# 两者被并排成"单帧检出率"比较，但根本不是同一条规则。
# 现在两条都能从帧表重算，于是**任何特征集都能同时给出两条基线**，口径不必再混。
SEVERITY_SUSPECT = 0.5   # 与 common.event_vector.SEVERITY_SUSPECT 一致
SF_RULES = ("severity>=0.5", "judge(框+姿态融合)")


def judge_from_table(t: np.ndarray) -> np.ndarray:
    """逐帧复刻 detect_fall_video.judge 的融合判定：框线索 OR 躯干倾角 OR 髋踝高差。

    阈值直接从 detect_fall_video 引，避免两处各写一份。
    """
    from common.event_vector import ASPECT_LYING, EventType

    aspect = t[:, C["aspect_ratio"]]
    torso = t[:, C["torso_angle"]]
    hip = t[:, C["hip_ankle_ratio"]]
    hit = (t[:, C["event_type"]] == int(EventType.SUSPECTED)) & (aspect < ASPECT_LYING)
    # nan 比较恒为 False，正合"这一帧没有该几何量 → 该线索不参与判定"
    hit |= np.nan_to_num(torso, nan=-1.0) > dfv.TORSO_TILT_MAX_DEG
    hit |= np.nan_to_num(hip, nan=1.0) < dfv.HIP_ANKLE_MIN_RATIO
    return hit


def single_frame_baselines(t: np.ndarray) -> dict[str, np.ndarray]:
    """帧表 → {规则名: 每帧 0/1}。"""
    return {
        SF_RULES[0]: (t[:, C["severity"]] >= SEVERITY_SUSPECT).astype(np.float32),
        SF_RULES[1]: judge_from_table(t).astype(np.float32),
    }


def build_feats(featset: str, table: dict) -> dict:
    """帧表 → 特征。纯数组运算，不碰模型。"""
    fs = FEATSETS[featset]
    out = {}
    for key, t in table.items():
        f = fs.build(t)
        assert f.shape == (len(t), fs.dim), f"{featset}/{key}: {f.shape} != ({len(t)}, {fs.dim})"
        out[key] = np.nan_to_num(f.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_all_feats(featsets: list[str]) -> None:
    tables: dict[str, dict] = {}
    for name in featsets:
        fs = FEATSETS[name]
        if fs.frames not in tables:
            tables[fs.frames] = load_frame_table(fs.frames)
        feats = build_feats(name, tables[fs.frames])
        labels = {f"{k}_label": np.array([1.0 if k.startswith("fall/") else 0.0])
                  for k in feats}
        save_npz(FEAT_CACHE[name], fs.spec, {**feats, **labels},
                 n_videos=len(feats), frameset=fs.frames)
        print(f"  {name:9s} {fs.dim:2d} 维 × {len(feats)} 段  ← 帧表 {fs.frames}"
              f"  → {rel(FEAT_CACHE[name])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --data 挂在每个子命令上（子解析器各自独立）。切换数据集会同时切缓存目录，
    # 理由见 set_dataset 上方那段注释。
    data_help = "数据集根目录（默认 data/datasets/URFD）；换数据集会同时换缓存目录"

    p = sub.add_parser("frames", help="跑模型产原始帧表（贵；pose 10~20 分钟）")
    p.add_argument("--mode", choices=["det", "pose"], required=True)
    p.add_argument("--only", default=None, help="只跑匹配该 glob 的视频，如 'fall/fall-01*'")
    p.add_argument("--resume", action="store_true", help="已有段落跳过，断点续跑")
    p.add_argument("--dump-csv", type=Path, default=None, help="额外导出逐帧 CSV 便于核对")
    p.add_argument("--data", type=Path, default=DATA_DIR, help=data_help)

    p = sub.add_parser("feats", help="帧表 → 特征缓存（秒级，不碰模型）")
    p.add_argument("--featsets", nargs="+", default=list(FEATSETS), choices=list(FEATSETS))
    p.add_argument("--data", type=Path, default=DATA_DIR, help=data_help)

    sub.add_parser("list", help="列出已注册的特征集")

    args = ap.parse_args()
    if args.cmd != "list" and getattr(args, "data", None) is not None:
        set_dataset(args.data)

    if args.cmd == "list":
        print(f"{'特征集':10s} {'维度':>4s}  {'帧表':8s} 特征")
        for fs in FEATSETS.values():
            print(f"{fs.name:10s} {fs.dim:4d}  {fs.frames:8s} {', '.join(fs.names)}")
        return
    if args.cmd == "frames":
        build_frame_table(args.mode, args.only, args.resume, args.dump_csv)
        return
    if args.cmd == "feats":
        print("帧表 → 特征缓存：")
        build_all_feats(args.featsets)


if __name__ == "__main__":
    main()
