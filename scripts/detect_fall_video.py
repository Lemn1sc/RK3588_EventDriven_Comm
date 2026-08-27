"""T0.4 前奏 · 真实视频跌倒检测（单机 · 检测 + 姿态级联）。

在拿到跌倒数据集（UR Fall Detection / Le2i）或自拍视频后，用它跑通
"真实检测跌倒"的完整流程 —— 组合3 级联卸载在单机上的最小版：

    每帧: yolov8n 检测 → 8 维事件向量(event_vector) → severity 达阈值
          (SUSPECTED) 才调用 yolov8n-pose 做姿态几何判定 → 画框标注

关键点：姿态模型只在"疑似帧"才跑 —— 这就是事件驱动卸载的本地模拟。
`pose 调用次数 / 总帧数` 就是卸载率，T0.8 对比实验要统计的核心指标。

跌倒几何量按 docs/07 建议（远端 pose 判定）：
    肩中点↔髋中点连线与垂直方向夹角、髋↔踝高度差(相对框高)、整体框高宽比。
关键点按 conf 过滤后再算，防遮挡 (0,0) 点污染（docs/07 实测要点）。

用法:
    python scripts/detect_fall_video.py 0                              # 摄像头
    python scripts/detect_fall_video.py data/datasets/URFD/video_01.avi
    python scripts/detect_fall_video.py <视频> --always-pose           # 每帧都跑姿态(调试)
    python scripts/detect_fall_video.py <图片>                         # 单张图

产出（默认 data/recordings/detect_fall/<源名>_annotated.* / _events.csv）：
    标注视频(或图) + 逐帧 CSV（事件向量字段 + 姿态几何量 + 跌倒判定）。
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np

# Windows 控制台默认 GBK，打不出 ✓（沿用 check_event_vector_real.py 的做法）
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PT_DIR = ROOT / "models" / "pt"
OUT_ROOT = ROOT / "data" / "recordings" / "detect_fall"

sys.path.insert(0, str(ROOT / "src"))
from common.event_vector import EventType, event_from_result  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CONF_THRES = 0.25       # 检测置信度阈值（与 src/common/event_vector.py 默认一致）
KP_CONF = 0.3           # 姿态关键点可见度阈值，低于此值弃用（docs/07 强调防 (0,0) 污染）
POSE_TRIGGER_SEV = 0.5  # severity ≥ 该值 → SUSPECTED → 调姿态（与契约 SEVERITY_SUSPECT 一致）

# COCO 17 关键点索引（yolov8-pose 固定顺序，见 docs/07）
I_SHO_L, I_SHO_R = 5, 6     # 左右肩
I_HIP_L, I_HIP_R = 11, 12   # 左右髋
I_ANK_L, I_ANK_R = 15, 16   # 左右踝

# 简单骨架连线（画图用）
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),         # 肩→肘→腕
    (5, 11), (6, 12), (11, 12),                      # 肩→髋
    (11, 13), (13, 15), (12, 14), (14, 16),          # 髋→膝→踝
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def fall_geometry(kpts_xy, kpts_conf, bbox_wh):
    """由单人 17 关键点算跌倒几何量（docs/07 建议的远端判定量）。

    关键点按 KP_CONF 过滤，遮挡/出画的 (0,0) 点不会混进均值；
    肩/髋缺失（关键量不足）返回 None，调用方退化为纯 bbox 判定。
    返回 {"torso_angle": 肩↔髋连线与垂直方向夹角(度), "hip_ankle_ratio": 髋踝高差/框高}。
    """
    def mid(i, j):
        if kpts_conf[i] < KP_CONF or kpts_conf[j] < KP_CONF:
            return None
        return (kpts_xy[i] + kpts_xy[j]) / 2.0

    sho = mid(I_SHO_L, I_SHO_R)
    hip = mid(I_HIP_L, I_HIP_R)
    ank = mid(I_ANK_L, I_ANK_R)
    if sho is None or hip is None:
        return None
    torso = hip - sho  # 指向下方
    angle = float(np.degrees(np.arctan2(abs(float(torso[0])), max(abs(float(torso[1])), 1e-6))))
    hip_ankle = None
    if ank is not None:
        # 髋↔踝高度差（相对框高）：像素 y 向下，站立时髋在踝上方 → 差值较大(~0.5)，
        # 躺平时两者同高 → ≈0。必须取绝对值，否则站立帧恒为 0 会误报（实测踩坑）。
        hip_ankle = float(abs(hip[1] - ank[1])) / max(bbox_wh[1], 1e-6)
    return {"torso_angle": angle, "hip_ankle_ratio": hip_ankle}


def judge(event, geom, aspect_ratio):
    """综合框高宽比 + 姿态几何量给最终跌倒判定。返回 (是否跌倒, 命中线索串)。"""
    reasons = []
    if event.event_type == EventType.SUSPECTED and aspect_ratio < 0.7:
        reasons.append(f"bbox_hw={aspect_ratio:.2f}")
    if geom is not None:
        if geom["torso_angle"] > 45:
            reasons.append(f"tilt={geom['torso_angle']:.0f}deg")
        if geom["hip_ankle_ratio"] is not None and geom["hip_ankle_ratio"] < 0.3:
            reasons.append(f"hipankle={geom['hip_ankle_ratio']:.2f}")
    return bool(reasons), "+".join(reasons)


def pick_person(kxy, kconf, center):
    """从多人姿态结果里挑主目标：可见关键点质心最接近检测主目标中心的那位。"""
    best, best_d = 0, float("inf")
    for i in range(len(kxy)):
        vis = kconf[i] >= KP_CONF
        if int(vis.sum()) < 4:
            continue
        cx = float(kxy[i, vis, 0].mean())
        cy = float(kxy[i, vis, 1].mean())
        d = (cx - center[0]) ** 2 + (cy - center[1]) ** 2
        if d < best_d:
            best_d, best = d, i
    return best


def draw_pose(img, kxy, kconf):
    """在图上画关键点和骨架（只画可见度达标的点/线）。"""
    for i, j in SKELETON:
        if kconf[i] >= KP_CONF and kconf[j] >= KP_CONF:
            cv2.line(img, tuple(kxy[i].astype(int)), tuple(kxy[j].astype(int)), (0, 255, 0), 2)
    for i in range(17):
        if kconf[i] >= KP_CONF:
            x, y = kxy[i].astype(int)
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)


def process_frame(frame, det_model, pose_model, always_pose):
    """跑一帧的级联：检测 → 事件向量 → (疑似)姿态 → 判定 → 标注。返回 (标注图, 日志行 dict)。"""
    h, w = frame.shape[:2]
    row = {"frame": 0, "has_event": 0, "event_type": 0, "confidence": 0.0, "severity": 0.0,
           "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 0, "bbox_y2": 0,
           "pose_invoked": 0, "aspect_ratio": float("nan"),
           "torso_angle": "", "hip_ankle_ratio": "", "fall": 0, "reason": ""}

    r = det_model(frame, verbose=False)[0]
    ev = event_from_result(r, conf_thres=CONF_THRES)
    row.update(has_event=int(ev.has_event), event_type=int(ev.event_type),
               confidence=ev.confidence, severity=ev.severity)
    if ev.has_event:
        x1, y1, x2, y2 = ev.bbox
        row.update(bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2)
        pix = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        cv2.rectangle(frame, (pix[0], pix[1]), (pix[2], pix[3]), (255, 0, 0), 2)
        aspect_ratio = (y2 - y1) / max(x2 - x1, 1e-6)
        row["aspect_ratio"] = aspect_ratio

    # 事件驱动卸载：仅疑似帧（或 --always-pose 调试）才跑姿态
    if ev.event_type == EventType.SUSPECTED or always_pose:
        pr = pose_model(frame, verbose=False)[0]
        k = pr.keypoints
        row["pose_invoked"] = 1
        if k is not None and len(k):
            kxy = k.xy.detach().cpu().numpy()
            kconf = k.conf.detach().cpu().numpy() if k.conf is not None else np.ones_like(kxy[..., 0])
            # 兜底：个别帧 pose 输出异常（实测 1 人 0 关键点 shape=(1,0,2)、conf 缺失），
            # 此时无法算几何量，跳过判定只记 pose_invoked=1（不算 bug，是模型异常输出）。
            if kxy.ndim == 3 and kxy.shape[1] >= 4 and kconf.shape == kxy.shape[:2]:
                cx = (row["bbox_x1"] + row["bbox_x2"]) / 2.0 * w
                cy = (row["bbox_y1"] + row["bbox_y2"]) / 2.0 * h
                i = pick_person(kxy, kconf, (cx, cy))
                geom = fall_geometry(kxy[i], kconf[i], (pix[2] - pix[0], pix[3] - pix[1]) if ev.has_event else (1.0, 1.0))
                if geom is not None:
                    row["torso_angle"] = f"{geom['torso_angle']:.1f}"
                    if geom["hip_ankle_ratio"] is not None:
                        row["hip_ankle_ratio"] = f"{geom['hip_ankle_ratio']:.3f}"
                fall, reason = judge(ev, geom, row["aspect_ratio"])
                row.update(fall=int(fall), reason=reason)
                draw_pose(frame, kxy[i], kconf[i])

    label = f"{EventType(row['event_type']).name} sev={row['severity']:.2f}"
    if row["fall"]:
        label += f"  [FALL {row['reason']}]"
        cv2.rectangle(frame, (0, 0), (240, 28), (0, 0, 255), -1)
    cv2.putText(frame, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame, row


def write_csv_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="视频/图片路径，或摄像头索引(0)")
    ap.add_argument("--always-pose", action="store_true", help="每帧都跑姿态（调试用，默认仅疑似帧）")
    ap.add_argument("--output", type=Path, default=None, help="输出目录，默认 data/recordings/detect_fall/")
    args = ap.parse_args()

    out_dir = args.output or OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    src = args.source
    is_cam = str(src).isdigit()
    src_path = Path(src)
    is_img = src_path.is_file() and src_path.suffix.lower() in IMAGE_EXTS

    det_model = YOLO(str(PT_DIR / "yolov8n.pt"))
    pose_model = YOLO(str(PT_DIR / "yolov8n-pose.pt"))

    if is_img:
        frame = cv2.imread(str(src_path))
        if frame is None:
            sys.exit(f"读不到图片: {src}")
        annot, row = process_frame(frame, det_model, pose_model, args.always_pose)
        row["frame"] = 0
        out_img = out_dir / f"{src_path.stem}_annotated.png"
        out_csv = out_dir / f"{src_path.stem}_events.csv"
        cv2.imwrite(str(out_img), annot)
        write_csv_rows(out_csv, [row])
        print(f"图片处理完成 → {out_img} / {out_csv}")
        print(f"  事件类型={EventType(row['event_type']).name} severity={row['severity']:.3f} "
              f"pose调用={row['pose_invoked']} 跌倒={bool(row['fall'])} {row['reason']}")
        return

    cap = cv2.VideoCapture(int(src) if is_cam else src)
    if not cap.isOpened():
        sys.exit(f"无法打开数据源: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = "webcam" if is_cam else src_path.stem
    out_video = out_dir / f"{stem}_annotated.mp4"
    out_csv = out_dir / f"{stem}_events.csv"

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rows, fid = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annot, row = process_frame(frame, det_model, pose_model, args.always_pose)
        row["frame"] = fid
        writer.write(annot)
        rows.append(row)
        fid += 1
    cap.release()
    writer.release()
    write_csv_rows(out_csv, rows)

    n_pose = sum(r["pose_invoked"] for r in rows)
    n_fall = sum(r["fall"] for r in rows)
    n_sus = sum(r["event_type"] == int(EventType.SUSPECTED) for r in rows)
    print(f"视频处理完成 → {out_video} / {out_csv}")
    print(f"  总帧数={fid}  疑似帧={n_sus}  pose调用={n_pose}（卸载率 {n_pose/max(fid,1)*100:.1f}%）  判定跌倒帧={n_fall}")


if __name__ == "__main__":
    main()
