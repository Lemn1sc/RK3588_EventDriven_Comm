#!/usr/bin/env python
"""评测事件层：事件级召回/误报/提前量 + 分阶段检测率 + 可靠性。

输入是 `scripts/make_event_traces.py` 产出的轨迹目录：

    <trace-root>/<dataset>/
        per_frame.npz
        gt_events.jsonl
        pred_events.jsonl
        meta.json

── 为什么要事件级而不是帧级 ──

帧级 AUC 回答“模型排序好不好”，但通信系统关心的是：

    “这次跌倒有没有被上报？提前了多少？误报了几次？每次误报占用多少带宽？”

所以本脚本的主口径是**事件**：

- 一个真值跌倒事件，只要有一个“告警类”预测事件落在
  `[onset - lead_window, end + tol]` 内，就算召回；
- 未被任何真值事件匹配的告警事件算误报；
- 同一真值事件被多个预测事件覆盖，算重复；
- 首个告警事件的 `start - onset` 是**提前量**（负数=提前，正数=延迟）。

`--alarm-kinds` 默认把 `PERSON_ABNORMAL / TRACK_LOST / SUSPECTED_FALL /
FALL_CONFIRMED` 视为告警；`PERSON` 只是“看到人”，不算跌倒告警。
如果只想看强告警，改成 `SUSPECTED_FALL,FALL_CONFIRMED`。

── 可靠性 ──

用逐帧 `confidence` 对逐帧 `gt_label` 做可靠性曲线（Brier / ECE）。这是一个
**第一版代理指标**：YOLO confidence 不是“跌倒概率”，事件层还没有做真正的
urgency 校准。它至少能说明“高置信帧是否更可能是跌倒”，以及不同数据集的校准差异。
真正的 urgency 校准应该按事件类型/机位分别做，见 docs/09。

用法：
    python scripts/eval_event_layer.py --dataset all
    python scripts/eval_event_layer.py --dataset mcf --alarm-kinds SUSPECTED_FALL,FALL_CONFIRMED
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.calibration import reliability_curve  # noqa: E402
from edge_node.event_tracker import EventKind  # noqa: E402

TRACE_COLS = [
    "index", "timestamp", "has_event", "pred_kind_id", "confidence", "severity",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "aspect_ratio",
    "gt_phase", "gt_label",
]
TC = {n: i for i, n in enumerate(TRACE_COLS)}

PHASE_NAMES = {0: "other", 1: "pre", 2: "during", 3: "post"}
ALARM_KINDS_DEFAULT = "PERSON_ABNORMAL,TRACK_LOST,SUSPECTED_FALL,FALL_CONFIRMED"


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(p)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(trace_root: Path, dataset: str):
    d = trace_root / dataset
    if not (d / "meta.json").exists():
        raise SystemExit(
            f"找不到事件轨迹：{rel(d)}\n"
            "  先跑：python scripts/make_event_traces.py --dataset " + dataset)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    gt = read_jsonl(d / "gt_events.jsonl")
    pred = read_jsonl(d / "pred_events.jsonl")
    npz = np.load(d / "per_frame.npz", allow_pickle=True)
    tables = {k: npz[k] for k in npz.files if not k.startswith("__")}
    return meta, gt, pred, tables


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def alarm_start_frame(p: dict, alarm_kinds: set[str]) -> int:
    """事件第一次成为**告警类型**的帧。

    不能用 `p["start_frame"]`：一个人可能在事件开始 2 秒后才从 PERSON 升级成
    SUSPECTED_FALL。用事件起点算提前量，会把“人出现在画面里”误当成提前预警。
    """
    kf = p.get("kind_first_frame") or {}
    vals = [int(v) for k, v in kf.items() if k in alarm_kinds]
    return min(vals) if vals else int(p["start_frame"])


def alarm_confidence(p: dict, alarm_kinds: set[str]) -> float:
    """事件第一次成为告警类型时的 confidence。

    事件最终 confidence 是“最后一次证据”的；如果 TRACK_LOST 之后又来了一个
    检测框，最终 confidence 会被覆盖。要按告警当时的分数过滤，必须看
    `kind_confidence`（按类型保存的首次分数）。
    """
    kf = p.get("kind_first_frame") or {}
    kc = p.get("kind_confidence") or {}
    cand = [(int(kf[k]), float(kc.get(k, p.get("confidence", 0.0))))
            for k in alarm_kinds if k in kf]
    if not cand:
        return float(p.get("confidence", 0.0))
    cand.sort()
    return cand[0][1]


def event_metrics(gt: list[dict], pred: list[dict], fps: float,
                  alarm_kinds: set[str], min_hits: int,
                  lead_window: int, tol: int,
                  total_frames: int,
                  min_confidence: float = 0.0) -> tuple[dict, list[dict], list[dict]]:
    """返回 (汇总指标, 未命中真值, 误报事件)。"""
    preds = [p for p in pred
             if p.get("kind") in alarm_kinds
             and int(p.get("hits", 0)) >= min_hits
             and alarm_confidence(p, alarm_kinds) >= min_confidence]
    # 按视频分组，避免跨视频匹配（尤其 MCF 每个机位是独立视频）
    by_video_pred: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for p in preds:
        by_video_pred[p["video"]].append((p, alarm_start_frame(p, alarm_kinds)))
    for v in by_video_pred:
        by_video_pred[v].sort(key=lambda t: (t[1], t[0]["event_id"]))

    used: set[tuple[str, int]] = set()
    matched: list[tuple[dict, dict, int]] = []
    missed: list[dict] = []

    for g in sorted(gt, key=lambda x: (x["video"], x["start_frame"])):
        lo = g["start_frame"] - lead_window
        hi = g["end_frame"] + tol
        cands = []
        for p, astart in by_video_pred.get(g["video"], []):
            key = (p["video"], p["event_id"])
            if key in used:
                continue
            if overlap(astart, p["end_frame"], lo, hi) <= 0:
                continue
            ov = overlap(astart, p["end_frame"],
                         g["start_frame"], g["end_frame"])
            # 有重叠优先；没有重叠时按“离 onset 越近越好”排序（仍在窗口内）。
            score = ov * 10000.0 - abs(astart - g["start_frame"])
            cands.append((score, p, astart))
        if not cands:
            missed.append(g)
            continue
        cands.sort(key=lambda t: (-t[0], t[1]["start_frame"]))
        _, p, astart = cands[0]
        used.add((p["video"], p["event_id"]))
        matched.append((g, p, astart))

    false_alarms = [p for p in preds if (p["video"], p["event_id"]) not in used]

    # 重复：同一真值事件区间内出现的告警事件数 > 1
    duplicates = 0
    for g in gt:
        n = sum(1 for p, astart in by_video_pred.get(g["video"], [])
                if overlap(astart, p["end_frame"],
                           g["start_frame"], g["end_frame"]) > 0)
        duplicates += max(0, n - 1)

    n_gt, n_pred, n_match = len(gt), len(preds), len(matched)
    total_min = total_frames / fps / 60.0 if fps > 0 else 0.0
    leads = [(astart - g["start_frame"]) / fps for g, p, astart in matched]
    durations = [(p["end_frame"] - astart + 1) / fps for _, p, astart in matched]

    def pct(xs, q):
        return round(float(np.percentile(xs, q)), 4) if xs else None

    return {
        "n_gt": n_gt,
        "n_alarm_events": n_pred,
        "n_matched": n_match,
        "recall": round(n_match / n_gt, 4) if n_gt else None,
        "precision": round(n_match / n_pred, 4) if n_pred else None,
        "false_alarms": len(false_alarms),
        "false_alarms_per_gt": round(len(false_alarms) / n_gt, 4) if n_gt else None,
        "false_alarms_per_min": (round(len(false_alarms) / total_min, 4)
                                 if total_min > 0 else None),
        "duplicates": duplicates,
        "duplicates_per_gt": round(duplicates / n_gt, 4) if n_gt else None,
        "lead_time_s_median": pct(leads, 50),
        "lead_time_s_p10": pct(leads, 10),
        "lead_time_s_p90": pct(leads, 90),
        "early_lead_s_median": (round(float(np.median([-x for x in leads if x < 0])), 4)
                                if any(x < 0 for x in leads) else None),
        "late_latency_s_median": (round(float(np.median([x for x in leads if x >= 0])), 4)
                                  if any(x >= 0 for x in leads) else None),
        "event_duration_s_median": pct(durations, 50),
        "total_duration_min": round(total_min, 4),
    }, missed, false_alarms


def phase_metrics(tables: dict[str, np.ndarray], alarm_ids: set[int]) -> dict:
    """按 gt_phase 聚合“有框率 / 告警事件率”。"""
    counts = defaultdict(int)
    has = defaultdict(int)
    alarm = defaultdict(int)
    for arr in tables.values():
        if len(arr) == 0:
            continue
        phases = arr[:, TC["gt_phase"]].astype(int)
        has_ev = arr[:, TC["has_event"]] > 0.5
        pred_kind = arr[:, TC["pred_kind_id"]].astype(int)
        is_alarm = np.isin(pred_kind, list(alarm_ids))
        for ph in PHASE_NAMES:
            m = phases == ph
            c = int(m.sum())
            counts[ph] += c
            has[ph] += int((m & has_ev).sum())
            alarm[ph] += int((m & is_alarm).sum())
    out = {}
    for ph, name in PHASE_NAMES.items():
        c = counts[ph]
        out[name] = {
            "frames": c,
            "has_event_rate": round(has[ph] / c, 4) if c else None,
            "alarm_rate": round(alarm[ph] / c, 4) if c else None,
        }
    return out


def per_camera_metrics(gt: list[dict], pred: list[dict], alarm_kinds: set[str],
                       min_hits: int, lead_window: int, tol: int,
                       fps: float, total_frames: int,
                       min_confidence: float = 0.0) -> dict:
    out = {}
    cams = sorted({g.get("camera", "?") for g in gt})
    for cam in cams:
        g = [x for x in gt if x.get("camera") == cam]
        p = [x for x in pred if x.get("camera") == cam]
        m, _, _ = event_metrics(g, p, fps, alarm_kinds, min_hits,
                                lead_window, tol, total_frames, min_confidence)
        # 只保留事件级核心字段，避免 JSON 里堆太多重复结构
        out[cam] = {k: m[k] for k in ("n_gt", "n_alarm_events", "n_matched",
                                      "recall", "precision", "false_alarms",
                                      "duplicates_per_gt", "lead_time_s_median")}
    return out


def calibration(tables: dict[str, np.ndarray]) -> dict:
    """逐帧 confidence 对 gt_label 的可靠性曲线。"""
    ys, ps, ps_all = [], [], []
    for arr in tables.values():
        if len(arr) == 0:
            continue
        y = arr[:, TC["gt_label"]]
        p = arr[:, TC["confidence"]]
        has = arr[:, TC["has_event"]] > 0.5
        ys.append(y[has])
        ps.append(p[has])
        ps_all.append(p)   # 无人帧 confidence=0，也放进来
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    p_all = np.concatenate(ps_all) if ps_all else np.array([])
    out = {}
    if y.size:
        out["with_event"] = reliability_curve(y, p, n_bins=10)
    else:
        out["with_event"] = None
    if p_all.size:
        # 全帧的 y 需要按同样的行数拼；上面只拼了 has 帧，这里单独拼
        ys_all = [arr[:, TC["gt_label"]] for arr in tables.values() if len(arr)]
        y_all = np.concatenate(ys_all)
        out["all_frames"] = reliability_curve(y_all, p_all, n_bins=10)
    else:
        out["all_frames"] = None
    return out


def evaluate(dataset: str, trace_root: Path, out_dir: Path,
             alarm_kinds: set[str], min_hits: int, lead_window: int, tol: int,
             tag: str = "", min_confidence: float = 0.0) -> dict:
    meta, gt, pred, tables = load_dataset(trace_root, dataset)
    fps = float(meta.get("fps", 30.0))
    total_frames = int(sum(len(a) for a in tables.values()))
    alarm_ids = {int(EventKind[k]) for k in alarm_kinds}

    metrics, missed, false_alarms = event_metrics(
        gt, pred, fps, alarm_kinds, min_hits, lead_window, tol, total_frames,
        min_confidence=min_confidence)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "eval_event_layer.py",
        "dataset": dataset,
        "tag": tag,
        "trace_root": rel(trace_root / dataset),
        "fps": fps,
        "total_frames": total_frames,
        "alarm_kinds": sorted(alarm_kinds),
        "min_hits": min_hits,
        "min_confidence": min_confidence,
        "lead_window_frames": lead_window,
        "tolerance_frames": tol,
        "counts_from_meta": meta.get("counts"),
        "alarm_kind_counts": dict(Counter(
            p.get("kind") for p in pred
            if p.get("kind") in alarm_kinds
            and int(p.get("hits", 0)) >= min_hits
            and alarm_confidence(p, alarm_kinds) >= min_confidence)),
        "event_metrics": metrics,
        "per_phase": phase_metrics(tables, alarm_ids),
        "per_camera": per_camera_metrics(gt, pred, alarm_kinds, min_hits,
                                         lead_window, tol, fps, total_frames,
                                         min_confidence=min_confidence),
        "calibration": calibration(tables),
        "missed_gt": [
            {"event_id": g.get("event_id"), "video": g["video"],
             "start_frame": g["start_frame"], "end_frame": g["end_frame"]}
            for g in missed
        ],
        "false_alarms": [
            {"event_id": p.get("event_id"), "video": p["video"],
             "kind": p.get("kind"), "start_frame": p["start_frame"],
             "alarm_start_frame": alarm_start_frame(p, alarm_kinds),
             "alarm_confidence": round(alarm_confidence(p, alarm_kinds), 6),
             "end_frame": p["end_frame"], "hits": p.get("hits")}
            for p in false_alarms[:200]   # 落盘只留前 200 条，完整数据在 pred_events.jsonl
        ],
        "notes": (
            "事件级口径；false_alarms_per_min 的分母是轨迹里出现的帧数"
            "（MCF 只含标注区间，不是完整视频时长）。calibration 是逐帧 confidence "
            "对 gt_label 的代理校准，不是 urgency 概率。"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"event_layer_{dataset}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result["_out_path"] = rel(out_path)
    return result


def print_summary(dataset: str, r: dict) -> None:
    m = r["event_metrics"]
    print(f"\n══════ {dataset.upper()} ══════")
    print(f"  真值事件 {m['n_gt']} · 告警事件 {m['n_alarm_events']} · 命中 {m['n_matched']}")
    print(f"  事件召回 {m['recall']} · 精确率 {m['precision']} · 误报 {m['false_alarms']} "
          f"({m['false_alarms_per_min']}/min)")
    print(f"  重复 {m['duplicates_per_gt']}/真值 · 提前量中位 {m['lead_time_s_median']}s "
          f"(p10 {m['lead_time_s_p10']}, p90 {m['lead_time_s_p90']})")
    print("  分阶段：")
    for ph in ("pre", "during", "post", "other"):
        x = r["per_phase"][ph]
        if x["frames"]:
            print(f"    {ph:7s} 帧 {x['frames']:6d}  有框率 {x['has_event_rate']}  "
                  f"告警率 {x['alarm_rate']}")
    cal = r["calibration"].get("with_event")
    if cal:
        print(f"  可靠性(有框帧)：Brier {cal['brier']} · ECE {cal['ece']} · n={cal['n']}")
    print(f"  结果 → {r['_out_path']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace-root", type=Path,
                   default=ROOT / "data/recordings/event_traces")
    p.add_argument("--out-dir", type=Path, default=ROOT / "experiments/results")
    p.add_argument("--dataset", choices=["urfd", "mcf", "all"], default="all")
    p.add_argument("--alarm-kinds", default=ALARM_KINDS_DEFAULT,
                   help=f"逗号分隔；默认 {ALARM_KINDS_DEFAULT}")
    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="事件 confidence 低于该值不算告警（模拟控制端丢弃低可靠事件）")
    p.add_argument("--lead-window-frames", type=int, default=60,
                   help="允许提前多少帧出现告警")
    p.add_argument("--tolerance-frames", type=int, default=30,
                   help="真值结束后多少帧内仍算命中")
    p.add_argument("--tag", default="",
                   help="结果文件名后缀，便于保存不同 alarm-kinds 口径")
    args = p.parse_args(argv)

    alarm_kinds = {s.strip() for s in args.alarm_kinds.split(",") if s.strip()}
    known = {"PERSON", "PERSON_ABNORMAL", "TRACK_LOST", "SUSPECTED_FALL",
             "FALL_CONFIRMED", "BACKGROUND"}
    bad = alarm_kinds - known
    if bad:
        raise SystemExit(f"未知事件类型：{sorted(bad)}；可选 {sorted(known)}")

    datasets = ["urfd", "mcf"] if args.dataset == "all" else [args.dataset]
    results = []
    for ds in datasets:
        if not (args.trace_root / ds / "meta.json").exists():
            print(f"[!] 跳过 {ds}：找不到轨迹，先跑 make_event_traces.py", file=sys.stderr)
            continue
        r = evaluate(ds, args.trace_root, args.out_dir, alarm_kinds,
                     args.min_hits, args.lead_window_frames, args.tolerance_frames,
                     tag=args.tag, min_confidence=args.min_confidence)
        print_summary(ds, r)
        results.append({k: v for k, v in r.items() if not k.startswith("_")})

    if len(results) > 1:
        suffix = f"_{args.tag}" if args.tag else ""
        summary_path = args.out_dir / f"event_layer_summary{suffix}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n汇总 → {rel(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
