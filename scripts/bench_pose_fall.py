"""T0.8 前奏 · 全帧姿态跌倒判定批量基准（验证"远端能否判准"）。

对 data/datasets/URFD/{fall,adl} 下所有视频逐帧跑 检测+姿态（等价 --always-pose），
统计每段视频的跌倒帧占比，判断仅靠姿态几何量能否区分跌倒/日常。
触发策略（单帧高宽比/GRU）不在本脚本范围内 —— 这里只回答"姿态判定本身准不准"。

判定规则（docs/07 建议 + detect_fall_video.py 的 judge）：
    框高宽比<0.7 或 躯干倾角>45° 或 髋踝高差/框高<0.3 → 该帧判跌倒。
    一段视频跌倒帧占比 ≥ FALL_RATIO_THRES 记为"检出跌倒"。

用法:
    python scripts/bench_pose_fall.py [--data data/datasets/URFD] [--out data/recordings/detect_fall/bench_pose.csv]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
PT_DIR = ROOT / "models" / "pt"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ultralytics import YOLO  # noqa: E402
import detect_fall_video as dfv  # noqa: E402  复用 process_frame（always_pose=True 全帧跑姿态）

FALL_RATIO_THRES = 0.2  # 跌倒帧占比 ≥ 该值记为该视频"检出跌倒"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "datasets" / "URFD")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "recordings" / "detect_fall" / "bench_pose.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    det = YOLO(str(PT_DIR / "yolov8n.pt"))
    pose = YOLO(str(PT_DIR / "yolov8n-pose.pt"))

    rows = []
    for split in ("fall", "adl"):
        for video in sorted((args.data / split).glob("*.mp4")):
            cap = cv2.VideoCapture(str(video))
            n_fall = n_total = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _, row = dfv.process_frame(frame, det, pose, always_pose=True)
                n_total += 1
                n_fall += int(row["fall"])
            cap.release()
            ratio = n_fall / max(n_total, 1)
            detected = ratio >= FALL_RATIO_THRES
            rows.append([split, video.name, n_total, n_fall, f"{ratio:.3f}", int(detected)])
            mark = "✓检出" if detected else "—"
            print(f"[{split}] {video.name:20s} 帧={n_total:4d} 跌倒帧={n_fall:4d} 占比={ratio:.2f} {mark}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "video", "frames", "fall_frames", "fall_ratio", "detected"])
        w.writerows(rows)

    fs = [r for r in rows if r[0] == "fall"]
    ad = [r for r in rows if r[0] == "adl"]
    tp = sum(r[5] for r in fs)
    fp = sum(r[5] for r in ad)
    print("\n=== 汇总 ===")
    print(f"跌倒视频检出: {tp}/{len(fs)}  (检出率 {tp / len(fs) * 100:.0f}%)" if fs else "无 fall 数据")
    print(f"日常视频误报: {fp}/{len(ad)}  (误报率 {fp / len(ad) * 100:.0f}%)" if ad else "无 adl 数据")
    print(f"明细已存: {args.out}")


if __name__ == "__main__":
    main()
