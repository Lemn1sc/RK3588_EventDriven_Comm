#!/usr/bin/env python
"""把现有检测帧表 → 事件轨迹（逐帧证据 + 真值事件 + 预测事件）。

── 为什么需要这一步 ──

控制端要跑“拥塞时优先传紧急事件”的仿真，需要的是**事件到达过程**：

    t=2.7s  SUSPECTED_FALL  建议紧急度 High  置信度 0.31
    t=2.9s  TRACK_LOST      建议紧急度 Medium 置信度 0.20
    t=3.1s  FALL_CONFIRMED  建议紧急度 Critical

而不是一堆视频或逐帧 CSV。本脚本把 `frames_det.v1.npz` 的逐帧检测结果喂给
`edge_node.EventTracker`，产出四样东西：

    per_frame.npz     逐帧证据 + 真值阶段/标签（供评估、回放）
    gt_events.jsonl   真值跌倒事件（URFD 官方逐帧标注 / MCF 区间标注）
    pred_events.jsonl 事件形成器输出的事件（带生命周期）
    stream.jsonl      事件流 START/UPDATE/CLOSE（可直接给控制端回放）

**这里不跑模型**：输入是已有的帧表，所以秒级完成。要换检测阈值/证据源，先重跑
`gru_features.py frames` / `mcf_detect.py`，再回来跑本脚本。

── 真值口径 ──

- URFD：官方 CSV 的 label（-1=未躺地, 0=正在跌倒, 1=躺地）。跌倒事件 = 第一个
  label=0 的帧 → 最后一个 label>=0 的帧。ADL 没有真值跌倒事件。
- MCF：`_labels.csv` 的区间（1=跌倒）。每个 label=1 的区间是一个真值事件；
  非跌倒区间按位置关系分成 pre / post（用于分阶段检出率）。

── 一个重要的基线选择 ──

本脚本默认**只使用现有帧表里置信度 >= 0.25 的框**（因为帧表就是这么产出的）。
它不会凭空造出低置信候选；`TRACK_LOST` 是从“上一帧还在、这一帧没了”派生的，
用于演示事件层如何把检测器失效变成信号。低阈值/前景证据需要先重跑检测，
见 docs/09 的“下一步”。

用法：
    python scripts/make_event_traces.py --dataset all
    python scripts/make_event_traces.py --dataset mcf --no-track-lost
    python scripts/make_event_traces.py --dataset urfd --out-dir data/recordings/event_traces
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from edge_node.event_tracker import (  # noqa: E402
    EventKind, EventTracker, EventTrackerConfig, Evidence, Urgency,
)

URFD_FRAME_TABLE = ROOT / "data/recordings/detect_fall_clean/frames_det.v1.npz"
URFD_OFFICIAL = ROOT / "data/datasets/URFD/_official"
MCF_FRAME_TABLE = ROOT / "data/recordings/mcf/mcf_frames.v1.npz"
DEFAULT_OUT = ROOT / "data/recordings/event_traces"

# 输入帧表的列（与 scripts/gru_features.py、scripts/mcf_detect.py 一致）
FRAME_COLS = [
    "index", "has_event", "event_type", "confidence", "severity",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "aspect_ratio",
    "pose_invoked", "torso_angle", "hip_ankle_ratio",
]
FC = {n: i for i, n in enumerate(FRAME_COLS)}

# 输出逐帧表的列
TRACE_COLS = [
    "index", "timestamp", "has_event", "pred_kind_id", "confidence", "severity",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "aspect_ratio",
    "gt_phase", "gt_label",
]
TC = {n: i for i, n in enumerate(TRACE_COLS)}

PHASE_OTHER, PHASE_PRE, PHASE_DURING, PHASE_POST = 0, 1, 2, 3


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(p)


def load_npz(path: Path, what: str):
    if not path.exists():
        raise SystemExit(
            f"找不到{what}：{rel(path)}\n"
            "  先跑对应的帧表脚本（见 docs/04 / docs/09），本脚本不跑模型。")
    return np.load(path, allow_pickle=True)


# ══════════════════════════ URFD 真值 ══════════════════════════

def load_urfd_official() -> dict[str, dict[int, int]]:
    """{序列名: {1-based 帧号: label}}。label: -1 / 0 / 1。"""
    out: dict[str, dict[int, int]] = defaultdict(dict)
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        path = URFD_OFFICIAL / name
        if not path.exists():
            print(f"[!] 缺少官方标注 {rel(path)}，URFD 真值阶段将全部记为 other",
                  file=sys.stderr)
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                seq = row[0].strip()
                try:
                    frame = int(float(row[1]))
                    label = int(float(row[2]))
                except ValueError:
                    continue
                out[seq][frame] = label
    return dict(out)


def urfd_seq(key: str) -> str:
    """'fall/fall-01-cam0.mp4' → 'fall-01'。"""
    return Path(key).stem.split("-cam")[0]


def urfd_truth(frames: np.ndarray, key: str, official: dict[str, dict[int, int]],
               is_fall: bool) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """返回 (gt_label, gt_phase, gt_event)。帧号 0-based，对应帧表 index。"""
    n = len(frames)
    seq = urfd_seq(key)
    lab_by_frame = official.get(seq, {})

    labs = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        frame_1based = int(frames[i, FC["index"]]) + 1
        labs[i] = lab_by_frame.get(frame_1based, -1)

    gt_label = np.zeros(n, dtype=np.int8)
    gt_phase = np.full(n, PHASE_OTHER, dtype=np.int8)
    event = None

    if is_fall:
        during = np.flatnonzero(labs == 0)
        lying = np.flatnonzero(labs == 1)
        onset = int(during[0]) if during.size else (int(lying[0]) if lying.size else None)
        end = None
        if during.size or lying.size:
            end = int(max(during[-1] if during.size else -1,
                          lying[-1] if lying.size else -1))
        if onset is not None and end is not None:
            event = {"start_frame": onset, "end_frame": end,
                     "during_start": int(during[0]) if during.size else None,
                     "during_end": int(during[-1]) if during.size else None,
                     "lying_start": int(lying[0]) if lying.size else None,
                     "lying_end": int(lying[-1]) if lying.size else None}
        for i in range(n):
            if labs[i] == 0:
                gt_phase[i] = PHASE_DURING
                gt_label[i] = 1
            elif labs[i] == 1:
                gt_phase[i] = PHASE_POST
                gt_label[i] = 1
            elif onset is not None and i < onset:
                gt_phase[i] = PHASE_PRE

    return gt_label, gt_phase, event


# ══════════════════════════ MCF 真值 ══════════════════════════

def mcf_key_parts(key: str) -> tuple[int, int]:
    """'chute01/cam1' → (1, 1)。"""
    chute = int(key[5:7])
    cam = int(key.split("/cam")[1])
    return chute, cam


def mcf_truth(spans: list[tuple[int, int, int]]) -> tuple[dict, dict, list[dict]]:
    """返回 (label_by_frame, phase_by_frame, gt_events)。"""
    label_by_frame: dict[int, int] = {}
    phase_by_frame: dict[int, int] = {}
    spans = sorted(spans)
    for si, (s, e, lab) in enumerate(spans):
        prev_lab = spans[si - 1][2] if si > 0 else None
        next_lab = spans[si + 1][2] if si + 1 < len(spans) else None
        for f in range(s, e + 1):
            label_by_frame[f] = lab
            if lab == 1:
                phase_by_frame[f] = PHASE_DURING
            elif prev_lab == 1:
                phase_by_frame[f] = PHASE_POST
            elif next_lab == 1:
                phase_by_frame[f] = PHASE_PRE
            else:
                phase_by_frame[f] = PHASE_OTHER

    gt_events = [
        {"start_frame": s, "end_frame": e, "label": lab}
        for s, e, lab in spans if lab == 1
    ]
    return label_by_frame, phase_by_frame, gt_events


# ══════════════════════════ 证据生成 ══════════════════════════

def detector_kind(row: np.ndarray, args) -> tuple[EventKind, dict]:
    """把一行检测结果映射成证据类型。返回 (kind, meta)。"""
    aspect = float(row[FC["aspect_ratio"]])
    severity = float(row[FC["severity"]])
    torso = float(row[FC["torso_angle"]])
    hip = float(row[FC["hip_ankle_ratio"]])

    reasons = []
    if np.isfinite(aspect) and aspect < args.aspect_threshold:
        reasons.append(f"aspect {aspect:.2f}<{args.aspect_threshold}")
    if np.isfinite(severity) and severity >= args.severity_threshold:
        reasons.append(f"severity {severity:.2f}>={args.severity_threshold}")
    if np.isfinite(torso) and torso > args.torso_threshold:
        reasons.append(f"torso {torso:.1f}>{args.torso_threshold}")
    if np.isfinite(hip) and hip < args.hip_ankle_threshold:
        reasons.append(f"hip_ankle {hip:.2f}<{args.hip_ankle_threshold}")

    meta = {"aspect": (aspect if np.isfinite(aspect) else None),
            "reasons": reasons}
    return (EventKind.SUSPECTED_FALL if reasons else EventKind.PERSON), meta


def make_evidence(row: np.ndarray, args) -> Evidence | None:
    if float(row[FC["has_event"]]) <= 0.5:
        return None
    kind, meta = detector_kind(row, args)
    bbox = (float(row[FC["bbox_x1"]]), float(row[FC["bbox_y1"]]),
            float(row[FC["bbox_x2"]]), float(row[FC["bbox_y2"]]))
    return Evidence(kind=kind, confidence=float(row[FC["confidence"]]),
                    bbox=bbox, source="detector", meta=meta)


@dataclass
class _TrackPoint:
    """轨迹上的一次检出。保存中心点/宽高比，供消失时算“消失前发生了什么”。"""
    frame: int
    cx: float
    cy: float
    aspect: float       # h/w，与项目其它地方一致
    bbox: tuple[float, float, float, float]


class TrackLostEmitter:
    """从“上一帧还有、这一帧没了”派生 TRACK_LOST 证据。

    ── 为什么要升级 ──

    旧逻辑只要“短断档 + 之前连续命中”就触发，结果 URFD 的 ADL 视频结尾
    （人正常走出画面）产生大量误报：36 个误报里绝大多数是这种。而真正的跌倒
    在消失前通常有两个特征之一：

    - 中心点**向下**移动（dy > 0，图像 y 轴向下）；
    - 宽高比**变扁**（h/w 下降，`d_aspect > 0`）。

    所以现在把“消失”变成一条**打分证据**：

        score = base
              + w_aspect * clip(d_aspect / aspect_scale, 0, 1)
              + w_dy     * clip(dy / dy_scale, 0, 1)
              + w_border * (1 if 不贴边 else 0)

    - `mode=all`：永远发（复现旧基线），score 只作 confidence；
    - `mode=motion`：score 低于 `min_score` 就不发 —— 过滤“正常走出画面”；
    - `mode=motion_no_border`：再加“贴边不触发”，更保守；
    - `score >= suspected_score` 时直接把 kind 升级成 `SUSPECTED_FALL`（HIGH），
      这样“消失前明显下沉/变扁”的事件会进入强告警，而不是混在 MEDIUM 的
      `TRACK_LOST` 里。

    这些阈值都可以用 CLI 调；默认值来自 URFD/MCF 轨迹的特征分析（见 docs/09）。
    """

    def __init__(self, args):
        self.args = args
        self.points: deque[_TrackPoint] = deque(
            maxlen=max(64, int(args.track_lost_history)))
        self.run = 0

    def _clear(self) -> None:
        self.points.clear()
        self.run = 0

    def _border(self, bbox: tuple[float, float, float, float]) -> bool:
        m = float(self.args.track_lost_border_margin)
        x1, y1, x2, y2 = bbox
        return x1 <= m or y1 <= m or x2 >= 1.0 - m or y2 >= 1.0 - m

    def update(self, frame_idx: int, row: np.ndarray, has: bool) -> Evidence | None:
        if has:
            x1, y1, x2, y2 = (float(row[FC["bbox_x1"]]), float(row[FC["bbox_y1"]]),
                              float(row[FC["bbox_x2"]]), float(row[FC["bbox_y2"]]))
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                self.points.append(_TrackPoint(
                    frame=int(frame_idx), cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0,
                    aspect=h / w, bbox=(x1, y1, x2, y2)))
                self.run += 1
            return None

        if not self.points:
            return None
        last = self.points[-1]
        gap = frame_idx - last.frame
        if gap > self.args.track_lost_gap:
            # 断档太长：不再认为是“刚才那个人消失了”
            self._clear()
            return None
        if self.run < self.args.track_lost_min_run:
            return None

        window = int(self.args.track_lost_window)
        hist = [p for p in self.points if p.frame >= last.frame - window]
        ref = hist[0] if len(hist) >= 2 else last
        dy = last.cy - ref.cy                 # 正 = 向下（图像坐标）
        dx = abs(last.cx - ref.cx)
        d_aspect = ref.aspect - last.aspect   # 正 = 变扁
        border = self._border(last.bbox)

        def clip01(v: float) -> float:
            return float(min(1.0, max(0.0, v)))

        asp_term = clip01(d_aspect / max(self.args.track_lost_aspect_scale, 1e-9))
        dy_term = clip01(dy / max(self.args.track_lost_dy_scale, 1e-9))
        score = (self.args.track_lost_base_score
                 + self.args.track_lost_aspect_weight * asp_term
                 + self.args.track_lost_dy_weight * dy_term
                 + self.args.track_lost_border_weight * (0.0 if border else 1.0))
        score = clip01(score)

        if self.args.track_lost_mode == "motion" and score < self.args.track_lost_min_score:
            self._clear()
            return None
        if (self.args.track_lost_mode == "motion_no_border"
                and (score < self.args.track_lost_min_score or border)):
            self._clear()
            return None

        upgrade = (self.args.track_lost_as_suspected
                   or score >= self.args.track_lost_suspected_score)
        kind = EventKind.SUSPECTED_FALL if upgrade else EventKind.TRACK_LOST
        high_urgency = score >= self.args.track_lost_high_urgency_score
        ev = Evidence(
            kind=kind, confidence=score, bbox=last.bbox, source="track_lost",
            reliability=score,
            urgency=Urgency.HIGH if high_urgency else Urgency.MEDIUM,
            meta={
                "gap": int(gap), "run": int(self.run), "window": window,
                "dy": round(dy, 6), "dx": round(dx, 6),
                "d_aspect": round(d_aspect, 6),
                "vertical_ratio": round(dy / max(dx, 1e-6), 6),
                "border": bool(border), "score": round(score, 6),
                "high_urgency": bool(high_urgency),
                "mode": self.args.track_lost_mode,
            })
        # 只触发一次；下次重新出现再建立新轨迹
        self._clear()
        return ev


def run_tracker(frames: np.ndarray, args, fps: float,
                gt_label: np.ndarray, gt_phase: np.ndarray,
                with_track_lost: bool) -> tuple[np.ndarray, list[dict], list[dict]]:
    """逐帧喂 tracker，返回 (逐帧表, 事件流, 预测事件)。"""
    cfg = EventTrackerConfig(
        max_gap_frames=args.max_gap, ttl_frames=args.ttl,
        min_hits=args.min_hits, match_iou=args.match_iou,
        match_center_dist=args.match_center_dist, update_every=0)
    tracker = EventTracker(cfg)
    lost = TrackLostEmitter(args)
    n = len(frames)
    table = np.full((n, len(TRACE_COLS)), np.nan, dtype=np.float64)
    stream: list[dict] = []
    last_ts = 0.0

    for i in range(n):
        frame_idx = int(frames[i, FC["index"]])
        ts = frame_idx / fps
        last_ts = ts
        has = float(frames[i, FC["has_event"]]) > 0.5
        evidences: list[Evidence] = []
        det = make_evidence(frames[i], args)
        if det is not None:
            evidences.append(det)

        if with_track_lost:
            tl = lost.update(frame_idx, frames[i], has)
            if tl is not None:
                evidences.append(tl)

        updates = tracker.update(frame_idx, ts, evidences)
        stream.extend(u.to_dict() for u in updates)

        max_kind = max((int(e.kind) for e in evidences), default=0)
        table[i, TC["index"]] = frame_idx
        table[i, TC["timestamp"]] = ts
        table[i, TC["has_event"]] = 1.0 if has else 0.0
        table[i, TC["pred_kind_id"]] = float(max_kind)
        table[i, TC["confidence"]] = float(frames[i, FC["confidence"]])
        table[i, TC["severity"]] = float(frames[i, FC["severity"]])
        for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "aspect_ratio"):
            table[i, TC[name]] = float(frames[i, FC[name]])
        table[i, TC["gt_phase"]] = float(gt_phase[i])
        table[i, TC["gt_label"]] = float(gt_label[i])

    last_frame = int(frames[-1, FC["index"]]) if n else 0
    stream.extend(u.to_dict() for u in tracker.close_all(last_frame, last_ts))
    preds = tracker.snapshots(include_expired=True)
    return table, stream, preds


# ══════════════════════════ 写盘 ══════════════════════════

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_dataset(dataset: str, out_dir: Path, tables: dict[str, np.ndarray],
                 gt_events: list[dict], pred_events: list[dict],
                 stream: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "trace_cols": TRACE_COLS, "frame_cols": FRAME_COLS}
    arrays = {k: v for k, v in tables.items()}
    arrays["__meta__"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(out_dir / "per_frame.npz", **arrays)
    write_jsonl(out_dir / "gt_events.jsonl", gt_events)
    write_jsonl(out_dir / "pred_events.jsonl", pred_events)
    write_jsonl(out_dir / "stream.jsonl", stream)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n[{dataset}] 写出 → {rel(out_dir)}")
    print(f"  逐帧表 {len(tables)} 段 · 真值事件 {len(gt_events)} · "
          f"预测事件 {len(pred_events)} · 事件流 {len(stream)} 条")


# ══════════════════════════ 数据集 ══════════════════════════

def trace_urfd(args) -> None:
    npz = load_npz(URFD_FRAME_TABLE, "URFD 干净帧表")
    official = load_urfd_official()
    keys = sorted(k for k in npz.files if not k.startswith("__") and not k.endswith("_label"))
    if not keys:
        raise SystemExit("URFD 帧表为空")

    tables: dict[str, np.ndarray] = {}
    gt_events: list[dict] = []
    pred_events: list[dict] = []
    stream: list[dict] = []
    gt_id = 1
    t0 = time.time()

    for i, key in enumerate(keys, 1):
        frames = npz[key]
        is_fall = key.startswith("fall/")
        gt_label, gt_phase, gt = urfd_truth(frames, key, official, is_fall)
        table, s, preds = run_tracker(frames, args, args.fps_urfd, gt_label, gt_phase,
                                      with_track_lost=not args.no_track_lost)
        tables[key] = table

        seq = urfd_seq(key)
        if gt is not None:
            gt_events.append({
                "dataset": "urfd", "event_id": gt_id, "kind": "FALL",
                "video": key, "camera": "cam0", "seq": seq,
                "start_frame": gt["start_frame"], "end_frame": gt["end_frame"],
                "start_ts": round(gt["start_frame"] / args.fps_urfd, 6),
                "end_ts": round(gt["end_frame"] / args.fps_urfd, 6),
                "duration_s": round((gt["end_frame"] - gt["start_frame"] + 1) / args.fps_urfd, 6),
                "meta": {"during_start": gt["during_start"], "during_end": gt["during_end"],
                         "lying_start": gt["lying_start"], "lying_end": gt["lying_end"]},
            })
            gt_id += 1
        for p in preds:
            pred_events.append({"dataset": "urfd", "video": key, "camera": "cam0",
                                "seq": seq, **p})
        for u in s:
            stream.append({"dataset": "urfd", "video": key, **u})

        if i % 10 == 0 or i == len(keys):
            print(f"  [{i}/{len(keys)}] {key:30s} {time.time()-t0:.0f}s", flush=True)

    meta = {
        "dataset": "urfd", "fps": args.fps_urfd,
        "source_frame_table": rel(URFD_FRAME_TABLE),
        "frame_index": "0-based（帧表 index）",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracker_config": vars(EventTrackerConfig(
            max_gap_frames=args.max_gap, ttl_frames=args.ttl, min_hits=args.min_hits,
            match_iou=args.match_iou, match_center_dist=args.match_center_dist)),
        "evidence_rule": evidence_rule(args),
        "counts": {"videos": len(keys), "gt_events": len(gt_events),
                   "pred_events": len(pred_events), "stream": len(stream)},
        "notes": "URFD 真值来自官方逐帧标注；ADL 没有跌倒真值事件。",
    }
    save_dataset("urfd", args.out_dir / "urfd", tables, gt_events, pred_events, stream, meta)


def trace_mcf(args) -> None:
    import mcf_detect

    npz = load_npz(MCF_FRAME_TABLE, "MCF 帧表")
    spans_by_key, adjusted = mcf_detect.read_labels(mcf_detect.LABELS)
    keys = sorted(k for k in npz.files if not k.startswith("__") and not k.endswith("_label"))
    if not keys:
        raise SystemExit("MCF 帧表为空")

    tables: dict[str, np.ndarray] = {}
    gt_events: list[dict] = []
    pred_events: list[dict] = []
    stream: list[dict] = []
    gt_id = 1
    t0 = time.time()

    for i, key in enumerate(keys, 1):
        frames = npz[key]
        chute, cam = mcf_key_parts(key)
        label_by_frame, phase_by_frame, gt_spans = mcf_truth(spans_by_key.get((chute, cam), []))
        n = len(frames)
        gt_label = np.zeros(n, dtype=np.int8)
        gt_phase = np.full(n, PHASE_OTHER, dtype=np.int8)
        for row in range(n):
            f = int(frames[row, FC["index"]])
            gt_label[row] = label_by_frame.get(f, 0)
            gt_phase[row] = phase_by_frame.get(f, PHASE_OTHER)

        table, s, preds = run_tracker(frames, args, args.fps_mcf, gt_label, gt_phase,
                                      with_track_lost=not args.no_track_lost)
        tables[key] = table

        for g in gt_spans:
            gt_events.append({
                "dataset": "mcf", "event_id": gt_id, "kind": "FALL",
                "video": key, "camera": f"cam{cam}", "chute": chute,
                "start_frame": g["start_frame"], "end_frame": g["end_frame"],
                "start_ts": round(g["start_frame"] / args.fps_mcf, 6),
                "end_ts": round(g["end_frame"] / args.fps_mcf, 6),
                "duration_s": round((g["end_frame"] - g["start_frame"] + 1) / args.fps_mcf, 6),
                "meta": {},
            })
            gt_id += 1
        for p in preds:
            pred_events.append({"dataset": "mcf", "video": key, "camera": f"cam{cam}",
                                "chute": chute, **p})
        for u in s:
            stream.append({"dataset": "mcf", "video": key, **u})

        if i % 20 == 0 or i == len(keys):
            print(f"  [{i}/{len(keys)}] {key:18s} {time.time()-t0:.0f}s", flush=True)

    meta = {
        "dataset": "mcf", "fps": args.fps_mcf,
        "source_frame_table": rel(MCF_FRAME_TABLE),
        "frame_index": "原始帧号（帧表 index；标注区间之间不连续）",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracker_config": vars(EventTrackerConfig(
            max_gap_frames=args.max_gap, ttl_frames=args.ttl, min_hits=args.min_hits,
            match_iou=args.match_iou, match_center_dist=args.match_center_dist)),
        "evidence_rule": evidence_rule(args),
        "counts": {"videos": len(keys), "gt_events": len(gt_events),
                   "pred_events": len(pred_events), "stream": len(stream)},
        "notes": f"MCF 标注区间边界重叠调整 {adjusted} 帧；帧率按约 25fps（AVI 头写 120fps 不可信）。",
    }
    save_dataset("mcf", args.out_dir / "mcf", tables, gt_events, pred_events, stream, meta)


def evidence_rule(args) -> dict:
    return {
        "detector_conf_threshold_in_frame_table": 0.25,
        "suspected_if_any": {
            "aspect_lt": args.aspect_threshold,
            "severity_ge": args.severity_threshold,
            "torso_angle_gt": args.torso_threshold,
            "hip_ankle_ratio_lt": args.hip_ankle_threshold,
        },
        "track_lost": {
            "enabled": not args.no_track_lost,
            "mode": args.track_lost_mode,
            "max_gap_frames": args.track_lost_gap,
            "min_run_frames": args.track_lost_min_run,
            "window_frames": args.track_lost_window,
            "min_score": args.track_lost_min_score,
            "suspected_score": args.track_lost_suspected_score,
            "high_urgency_score": args.track_lost_high_urgency_score,
            "base_score": args.track_lost_base_score,
            "aspect_weight": args.track_lost_aspect_weight,
            "dy_weight": args.track_lost_dy_weight,
            "border_weight": args.track_lost_border_weight,
            "aspect_scale": args.track_lost_aspect_scale,
            "dy_scale": args.track_lost_dy_scale,
            "border_margin": args.track_lost_border_margin,
            "as_suspected_fall": args.track_lost_as_suspected,
        },
    }


def build_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["urfd", "mcf", "all"], default="all")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fps-urfd", type=float, default=30.0)
    p.add_argument("--fps-mcf", type=float, default=25.0)

    p.add_argument("--aspect-threshold", type=float, default=0.7,
                   help="aspect=h/w 低于该值 → SUSPECTED_FALL")
    p.add_argument("--severity-threshold", type=float, default=0.5)
    p.add_argument("--torso-threshold", type=float, default=45.0,
                   help="躯干倾角大于该值 → SUSPECTED_FALL（有姿态几何时）")
    p.add_argument("--hip-ankle-threshold", type=float, default=0.3)

    p.add_argument("--no-track-lost", action="store_true",
                   help="关闭 TRACK_LOST 证据（默认开启）")
    p.add_argument("--track-lost-mode", choices=["all", "motion", "motion_no_border"],
                   default="all",
                   help="all=全部发出、只打 confidence（推荐）；"
                        "motion=消失前有下沉/变扁才发；motion_no_border=再加贴边不触发")
    p.add_argument("--track-lost-gap", type=int, default=2,
                   help="消失后多少帧内补一条 TRACK_LOST")
    p.add_argument("--track-lost-min-run", type=int, default=5,
                   help="消失前至少连续检出多少帧才认为是一条轨迹")
    p.add_argument("--track-lost-window", type=int, default=30,
                   help="用消失前多少帧内的检出计算 dy / d_aspect")
    p.add_argument("--track-lost-history", type=int, default=64,
                   help="轨迹历史缓存的最大长度（帧）")
    p.add_argument("--track-lost-min-score", type=float, default=0.35,
                   help="motion 模式下低于该分数不触发")
    p.add_argument("--track-lost-suspected-score", type=float, default=1.1,
                   help="分数达到该值时升级成 SUSPECTED_FALL；默认 1.1=不升级，"
                        "保留 TRACK_LOST 语义，由 confidence/urgency 表达可靠性")
    p.add_argument("--track-lost-high-urgency-score", type=float, default=0.55,
                   help="TRACK_LOST 分数达到该值时建议紧急度 HIGH（否则 MEDIUM）")
    p.add_argument("--track-lost-base-score", type=float, default=0.15)
    p.add_argument("--track-lost-aspect-weight", type=float, default=0.45)
    p.add_argument("--track-lost-dy-weight", type=float, default=0.30)
    p.add_argument("--track-lost-border-weight", type=float, default=0.10,
                   help="不贴边时加的分；贴边不会减分，只是拿不到这一项")
    p.add_argument("--track-lost-aspect-scale", type=float, default=1.0,
                   help="d_aspect 达到该值记满分")
    p.add_argument("--track-lost-dy-scale", type=float, default=0.15,
                   help="dy 达到该值记满分（归一化坐标）")
    p.add_argument("--track-lost-border-margin", type=float, default=0.02)
    p.add_argument("--track-lost-as-suspected", action="store_true",
                   help="把所有 TRACK_LOST 强制升级成 SUSPECTED_FALL")

    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--max-gap", type=int, default=30)
    p.add_argument("--ttl", type=int, default=300)
    p.add_argument("--match-iou", type=float, default=0.10)
    p.add_argument("--match-center-dist", type=float, default=0.20)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = build_args(argv)
    print(f"输出目录：{rel(args.out_dir)}")
    print(f"证据规则：aspect<{args.aspect_threshold} / severity>={args.severity_threshold} "
          f"/ torso>{args.torso_threshold} / hip_ankle<{args.hip_ankle_threshold}；"
          f"TRACK_LOST={'off' if args.no_track_lost else args.track_lost_mode}"
          f"(min_score={args.track_lost_min_score}, suspected>="
          f"{args.track_lost_suspected_score})")
    if args.dataset in ("urfd", "all"):
        trace_urfd(args)
    if args.dataset in ("mcf", "all"):
        trace_mcf(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
