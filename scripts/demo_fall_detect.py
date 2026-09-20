#!/usr/bin/env python
"""跌倒检测在线 demo —— 逐帧跑完整链路，输出带实时信息的视频。

链路：YOLOv8n 检测 → 事件向量 → det.v2 的 8 维特征 → 30 帧滑窗 → GRU → 决策

产出两样东西：
1. **标注视频**（`*_demo.mp4`）—— 右panel 实时显示当前状态、GRU 概率曲线、
   特征值；画面上画出检测框。看了就知道系统在干什么。
2. **控制台摘要** —— 决策延迟（从官方标注的「开始跌倒」到「首次报警」隔多少帧）。
   这是产品真正关心的指标，不是 AUC。

用法：
    # 单段（带官方标注，能算出延迟）
    python scripts/demo_fall_detect.py --video data/datasets/URFD_clean/fall/fall-03-cam0.mp4 \
        --label-seq fall-03

    # 批量跑前 5 段，只出摘要
    python scripts/demo_fall_detect.py --batch 5

    # 用自己的视频（无标注，只看效果）
    python scripts/demo_fall_detect.py --video path/to/x.mp4

注意：特征是用「累积帧表 + 重建」算的，与训练时**逐位一致** —— 走的是
`FEATSETS['det.v2'].build()` 这条同源代码路径，不手工拼特征。

代价是每帧对整张表重算一遍（O(T²)）。演示视频只有几百帧，可以忽略；
但**板端要改成流式增量**：`cy_drop` / `size_log_ratio` 的因果基线本质是
「最近 30 个有目标帧的中位数」，只需维护一个长度 30 的队列，是 O(1) 内存。
（改完必须拿本脚本对同一段视频跑一遍、逐位比对特征，确认两边一致 ——
手工拼特征最容易在这里出错，先前测决策延迟时就吃过一次亏。）
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import detect_fall_video as dfv          # noqa: E402  复用 process_frame，与训练同源
import gru_features as gf                # noqa: E402
import train_gru_fall as tg              # noqa: E402

OFFICIAL = ROOT / "data" / "datasets" / "URFD" / "_official"
OUT_DIR = ROOT / "data" / "recordings" / "demo"

# 面板配色 (BGR)
C_BG = (32, 32, 36)
C_TXT = (235, 235, 235)
C_DIM = (140, 140, 145)
C_OK = (120, 200, 120)
C_WARN = (60, 190, 245)
C_ALERT = (70, 70, 235)
C_GRID = (70, 70, 76)


def _font(size: int):
    """尽量用中文字体；找不到就退回 cv2 内置（英文）。"""
    from PIL import ImageFont
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return None


class Canvas:
    """把 cv2 图像包一层，支持中文与抗锯齿绘制（PIL 画，cv2 显示）。"""

    def __init__(self, w: int, h: int):
        from PIL import Image, ImageDraw
        self._Image, self._Draw = Image, ImageDraw
        self.fonts: dict[int, object] = {}
        self.img = Image.new("RGB", (w, h), (C_BG[2], C_BG[1], C_BG[0]))
        self.d = ImageDraw.Draw(self.img)

    def font(self, size: int):
        if size not in self.fonts:
            f = _font(size)
            from PIL import ImageFont
            self.fonts[size] = f if f is not None else ImageFont.load_default()
        return self.fonts[size]

    def text(self, xy, s, size=14, color=C_TXT, anchor="la"):
        rgb = (color[2], color[1], color[0])
        self.d.text(xy, s, font=self.font(size), fill=rgb, anchor=anchor)

    def rect(self, box, fill=None, outline=None, width=1):
        f = (fill[2], fill[1], fill[0]) if fill else None
        o = (outline[2], outline[1], outline[0]) if outline else None
        self.d.rectangle(box, fill=f, outline=o, width=width)

    def paste(self, bgr, xy):
        self.img.paste(self._Image.fromarray(bgr[:, :, ::-1]), xy)

    def array(self):
        return np.array(self.img)[:, :, ::-1].copy()


def official_labels(seq: str) -> dict[int, int]:
    """{帧号: label}，label: -1 未躺地 / 0 正在跌倒 / 1 躺地。"""
    out: dict[int, int] = {}
    for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        p = OFFICIAL / f
        if not p.exists():
            continue
        for r in csv.reader(open(p, encoding="utf-8")):
            if r and r[0] == seq:
                out[int(r[1])] = int(r[2])
    return out


def run(video: Path, model, out_path: Path | None, thr: float,
        labels: dict[int, int] | None, show: bool = True) -> dict:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    PANEL = 300 if show else 0
    canvas_w, canvas_h = W + PANEL, max(H, 470)
    writer = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (canvas_w, canvas_h))

    tbl: list[np.ndarray] = []          # 累积帧表（保证特征与训练逐位一致）
    probs: list[float] = []             # 逐帧 GRU 概率
    first_alert: int | None = None
    rows_meta: list[dict] = []
    t = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # process_frame 会**原地**在帧上画框和骨架（它是给标注视频用的）。
        # 传副本，把原始帧留给 compose 自己画 —— 否则两套标注会叠在一起。
        _, row = dfv.process_frame(frame.copy(), det_model, pose_model, False)
        row["frame"] = t
        tbl.append(gf.row_to_frame(row))

        # ── 特征：累积帧表 → det.v2（与训练同一条代码路径）──
        feats = gf.FEATSETS["det.v2"].build(np.stack(tbl))
        feats = np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        p = 0.0
        if len(feats) >= tg.WINDOW:
            win = feats[-tg.WINDOW:][None, ...]
            with torch.no_grad():
                p = float(torch.sigmoid(model(torch.tensor(win, device=next(model.parameters()).device)))[0])
        probs.append(p)
        if p >= thr and first_alert is None:
            first_alert = t

        rows_meta.append({"frame": t, "prob": p, **{n: float(v) for n, v in zip(gf.FEATSETS["det.v2"].names, feats[-1])}})

        if writer is not None:
            writer.write(compose(frame, canvas_w, canvas_h, W, PANEL, t, fps, row,
                                 feats[-1], p, probs, thr, labels, first_alert))
        t += 1

    cap.release()
    if writer is not None:
        writer.release()

    return {"video": video.name, "frames": t, "fps": fps,
            "first_alert": first_alert, "probs": probs, "rows": rows_meta,
            "out": str(out_path) if out_path else None}


def compose(frame, cw, ch, W, PANEL, t, fps, row, feat, prob, probs, thr,
            labels, first_alert):
    """左：视频 + 检测框；右：状态面板。"""
    canvas = np.full((ch, cw, 3), C_BG, np.uint8)
    canvas[:frame.shape[0], :W] = frame

    # 检测框 + 标注
    if row["has_event"]:
        x1, y1, x2, y2 = (int(row[k] * (W if k.startswith("bbox_x") else frame.shape[0]))
                          for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
        col = C_ALERT if prob >= thr else C_OK
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        cv2.putText(canvas, f"person {row['confidence']:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    if PANEL == 0:
        return canvas

    P = W
    c = Canvas(cw, ch)
    c.paste(canvas, (0, 0))

    # ── 状态徽章 ──
    alert = prob >= thr
    soon = (not alert) and prob >= thr * 0.6
    col = C_ALERT if alert else (C_WARN if soon else C_OK)
    txt = "报警 · FALL" if alert else ("疑似 · SUSPECT" if soon else "正常 · NORMAL")
    c.rect((P + 14, 12, cw - 14, 52), fill=col)
    c.text((P + 26, 32), txt, 19, (20, 20, 20), anchor="lm")

    # ── GRU 概率 ──
    c.text((P + 14, 68), "GRU 跌倒概率", 13, C_DIM)
    c.text((P + 14, 88), f"{prob:.3f}", 28, col)
    c.text((P + 108, 98), f"阈值 {thr:.2f}", 12, C_DIM)
    bx, by, bw, bh = P + 14, 122, cw - P - 28, 11
    c.rect((bx, by, bx + bw, by + bh), fill=C_GRID)
    c.rect((bx, by, bx + int(bw * min(prob, 1.0)), by + bh), fill=col)
    c.rect((bx + int(bw * thr), by - 3, bx + int(bw * thr) + 2, by + bh + 3), fill=C_WARN)

    # ── 概率曲线（最近 150 帧）──
    c.text((P + 14, 148), "概率曲线（最近 150 帧）", 12, C_DIM)
    gx, gy, gw, gh = P + 14, 162, cw - P - 28, 52
    c.rect((gx, gy, gx + gw, gy + gh), fill=(24, 24, 28))
    c.rect((gx, gy + gh - int(gh * thr), gx + gw, gy + gh - int(gh * thr) + 1), fill=C_GRID)
    if labels:                       # 真值底色：橙=正在跌倒，红=躺地
        for i in range(150):
            idx = t - 149 + i
            if idx in labels and labels[idx] >= 0:
                xx = gx + int(gw * i / 150)
                c.rect((xx, gy + gh - 4, xx + 1, gy + gh), fill=C_WARN if labels[idx] == 0 else C_ALERT)
    seg = probs[-150:]
    if len(seg) > 1:
        pts = [(gx + int(gw * i / 150), gy + gh - int(gh * min(p, 1.0)))
               for i, p in enumerate(seg)]
        for a, b in zip(pts, pts[1:]):
            c.d.line([a, b], fill=(col[2], col[1], col[0]), width=2)

    # ── 当前特征 ──
    c.text((P + 14, 228), "当前帧特征（det.v2）", 12, C_DIM)
    for i, (n, v) in enumerate(zip(gf.FEATSETS["det.v2"].names, feat)):
        yy = 248 + i * 18
        c.text((P + 14, yy), n, 12, C_DIM)
        c.text((cw - 16, yy), f"{v:+.3f}", 12, C_TXT, anchor="ra")

    # ── 底部：帧/时间 + 报警状态（409 起，特征表到 248+7*18=374，不重叠）──
    c.rect((P + 14, 392, cw - 14, 394), fill=C_GRID)
    c.text((P + 14, 404), f"帧 {t}    {t / fps:.2f} s", 12, C_DIM)
    if first_alert is not None:
        c.text((P + 14, 424), f"首次报警：第 {first_alert} 帧"
                              f"  ({first_alert / fps:.2f}s)", 13, C_ALERT)
    else:
        c.text((P + 14, 424), "尚未报警", 13, C_DIM)
    if labels:
        on = [f for f, l in sorted(labels.items()) if l == 0]
        if on:
            c.text((P + 14, 444), f"真值跌倒起点：第 {min(on) - 1} 帧", 12, C_WARN)

    return c.array()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--label-seq", default=None, help="官方标注的序列名，如 fall-03（用于画真值、算延迟）")
    ap.add_argument("--thr", type=float, default=0.9, help="报警阈值（默认 0.90）")
    ap.add_argument("--out", type=Path, default=None, help="输出视频路径")
    ap.add_argument("--outdir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-video", action="store_true", help="只跑不写视频（快）")
    ap.add_argument("--batch", type=int, default=0, help="批量跑前 N 段跌倒视频，只出摘要")
    args = ap.parse_args()

    global det_model, pose_model
    from ultralytics import YOLO
    det_model = YOLO(str(dfv.PT_DIR / "yolov8n.pt"))
    pose_model = YOLO(str(dfv.PT_DIR / "yolov8n-pose.pt"))
    model = tg.load_checkpoint(dfv.PT_DIR / "gru_fall_det.v2.pt", "det.v2").eval()
    model.to("cpu")

    jobs = []
    if args.batch:
        for i in range(1, args.batch + 1):
            seq = f"fall-{i:02d}"
            v = ROOT / "data" / "datasets" / "URFD_clean" / "fall" / f"{seq}-cam0.mp4"
            if v.exists():
                jobs.append((v, seq, None))
    else:
        if args.video is None:
            raise SystemExit("需要 --video 或 --batch N")
        jobs.append((args.video, args.label_seq, args.out))

    results = []
    for v, seq, out in jobs:
        labels = official_labels(seq) if seq else None
        op = None
        if not args.no_video:
            op = out or (args.outdir / f"{v.stem}_demo.mp4")
        print(f"处理 {v.name} ...", flush=True)
        results.append(run(v, model, op, args.thr, labels, show=not args.no_video))

    print("\n" + "=" * 66)
    print("摘要")
    print("=" * 66)
    for r in results:
        labels = official_labels(r["video"].split("-cam0")[0]) if "-cam0" in r["video"] else {}
        onset = None
        if labels:
            on = [f for f, l in sorted(labels.items()) if l == 0]
            onset = (min(on) - 1) if on else None      # 标注从 1 开始 → 转 0 基
        fa = r["first_alert"]
        lat = (fa - onset) if (fa is not None and onset is not None) else None
        line = f"  {r['video']:<22} {r['frames']:4d} 帧"
        if onset is not None:
            line += f"  跌倒起点 {onset:4d}"
        line += f"  首次报警 {'—' if fa is None else fa}"
        if lat is not None:
            line += f"  延迟 {lat:3d} 帧 ({lat / r['fps']:.2f}s)"
        print(line)
        if r["out"]:
            print(f"      → {Path(r['out']).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
