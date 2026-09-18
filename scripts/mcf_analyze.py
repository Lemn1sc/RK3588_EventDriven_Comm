#!/usr/bin/env python
"""分析 MCF 的检测结果：同一次跌倒，8 个机位各自能不能看到人。

输入 `scripts/mcf_detect.py` 产出的帧表，输出三张表：

1. **按机位**：跌倒帧检出率 vs 非跌倒帧检出率 —— 回答「哪个机位整体上更差」。
2. **按场景**：每个场景的跌倒，8 个机位里有几个检得到 —— 回答「同一次跌倒的
   机位间差异有多大」，这是配对比较，不受场景难度影响。
3. **事件级**：把每个连续跌倒区间当作一个「事件」，看它有没有被任何机位漏掉。
   帧级检出率会被"检出一半"这种中间状态模糊掉，事件级更能反映"这次跌倒报没报"。

用法：
    python scripts/mcf_analyze.py
    python scripts/mcf_analyze.py --out experiments/results/mcf_view_detect.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAME_TABLE = ROOT / "data" / "recordings" / "mcf" / "mcf_frames.v1.npz"

FRAME_COLS = [
    "index", "has_event", "event_type", "confidence", "severity",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "aspect_ratio",
    "pose_invoked", "torso_angle", "hip_ankle_ratio",
]
C = {n: i for i, n in enumerate(FRAME_COLS)}

# 一个跌倒区间算「被看见」的门槛：区间内至少这么多比例的帧检出人才算。
# 定在 0.5 是保守的 —— 只要过半帧能看到，这条轨迹就有机会被时序模型接住。
SEEN_FRAC = 0.5


def load() -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    if not FRAME_TABLE.exists():
        raise SystemExit(
            f"找不到帧表：{FRAME_TABLE}\n  先跑：python scripts/mcf_detect.py"
        )
    npz = np.load(FRAME_TABLE, allow_pickle=True)
    out = {}
    for k in npz.files:
        if k.endswith("_label") or k.startswith("__"):
            continue
        ch, _, cam = k.replace("chute", "").partition("/cam")
        out[(int(ch), int(cam))] = (npz[k], npz[f"{k}_label"])
    return out


def spans_from(labels: np.ndarray, frames: np.ndarray, has: np.ndarray):
    """把逐帧标签还原成连续区间：[(起始帧号, 结束帧号, 标签, 检出帧数, 总帧数)]。

    帧表是按标注区间顺序拼的，原始帧号在 index 列里 —— 区间边界以帧号是否连续判断。
    """
    spans = []
    i = 0
    n = len(labels)
    while i < n:
        j = i
        while (j + 1 < n and labels[j + 1] == labels[i]
               and frames[j + 1] == frames[j] + 1):
            j += 1
        seg = has[i:j + 1]
        spans.append((int(frames[i]), int(frames[j]), int(labels[i]),
                      int(seg.sum()), len(seg)))
        i = j + 1
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments" / "results" / "mcf_view_detect.json")
    args = ap.parse_args()

    data = load()
    chutes = sorted({ch for ch, _ in data})
    cams = sorted({cam for _, cam in data})

    # ── 表 1：按机位 ──
    per_cam = defaultdict(lambda: {"fall_seen": 0, "fall_tot": 0,
                                   "non_seen": 0, "non_tot": 0})
    # ── 表 2：按场景 ──  (chute) -> {cam: fall 检出率}
    per_chute = defaultdict(dict)
    # ── 表 3：事件级 ──
    events = []

    for (ch, cam), (tab, lab) in sorted(data.items()):
        has = tab[:, C["has_event"]] > 0.5
        frames = tab[:, C["index"]].astype(int)
        f_mask = lab == 1
        n_mask = lab == 0
        per_cam[cam]["fall_seen"] += int(has[f_mask].sum())
        per_cam[cam]["fall_tot"] += int(f_mask.sum())
        per_cam[cam]["non_seen"] += int(has[n_mask].sum())
        per_cam[cam]["non_tot"] += int(n_mask.sum())
        fr = has[f_mask].mean() if f_mask.any() else float("nan")
        per_chute[ch][cam] = float(fr)
        for s, e, l, seen, tot in spans_from(lab, frames, has):
            if l == 1:
                events.append({"chute": ch, "cam": cam, "start": s, "end": e,
                               "frames": tot, "seen": seen,
                               "rate": seen / tot, "seen_ok": seen / tot >= SEEN_FRAC})

    print("=" * 74)
    print("表 1 · 按机位（标注帧上的检出率）")
    print("=" * 74)
    print(f"{'机位':>5}{'跌倒帧检出':>12}{'非跌倒帧检出':>14}{'差':>9}   跌倒帧数")
    for cam in cams:
        d = per_cam[cam]
        fr = d["fall_seen"] / max(d["fall_tot"], 1)
        nr = d["non_seen"] / max(d["non_tot"], 1)
        print(f"{cam:>5}{fr:>11.1%}{nr:>13.1%}{fr - nr:>+9.1%}   {d['fall_tot']:>6,}")

    tot_fs = sum(d["fall_seen"] for d in per_cam.values())
    tot_ft = sum(d["fall_tot"] for d in per_cam.values())
    tot_ns = sum(d["non_seen"] for d in per_cam.values())
    tot_nt = sum(d["non_tot"] for d in per_cam.values())
    print(f"{'合计':>5}{tot_fs / tot_ft:>11.1%}{tot_ns / tot_nt:>13.1%}"
          f"{tot_fs / tot_ft - tot_ns / tot_nt:>+9.1%}   {tot_ft:>6,}")

    print()
    print("=" * 74)
    print("表 2 · 按场景：一次跌倒有几个机位看得到（配对比较，不受场景难度影响）")
    print("=" * 74)
    print(f"{'场景':>6}{'8 机位均值':>12}{'最好':>9}{'最差':>9}{'机位间极差':>12}   逐机位")
    counts = defaultdict(int)
    for ch in chutes:
        v = per_chute[ch]
        vals = [v[c] for c in cams if not np.isnan(v.get(c, float('nan')))]
        if not vals:
            continue
        spread = max(vals) - min(vals)
        nb = sum(1 for x in vals if x >= SEEN_FRAC)
        counts[nb] += 1
        bar = "".join("#" if v[c] >= SEEN_FRAC else ("+" if v[c] >= 0.2 else ".")
                      for c in cams)
        print(f"{'chute%02d' % ch:>6}{np.mean(vals):>11.1%}{max(vals):>9.0%}"
              f"{min(vals):>9.0%}{spread:>11.0%}   {bar}")

    print()
    print("  逐机位记号：# = 检出率 ≥ 50%（算「看得到」），+ = 20%~50%，. = < 20%")
    print()
    print(f"{'看得见的机位数':>12}{'场景数':>8}")
    for k in sorted(counts):
        print(f"{k:>12}{counts[k]:>8}")
    if counts:
        ok_all = counts.get(len(cams), 0)
        ok_none = counts.get(0, 0)
        print(f"\n  8 个机位都看得到：{ok_all}/{len(chutes)} 个场景")
        print(f"  8 个机位都看不到：{ok_none}/{len(chutes)} 个场景")

    print()
    print("=" * 74)
    print(f"表 3 · 事件级：每个跌倒区间是否「≥{SEEN_FRAC:.0%} 的帧检出人」")
    print("=" * 74)
    by_ev = defaultdict(lambda: {"ok": 0, "tot": 0})
    for e in events:
        by_ev[e["chute"]]["tot"] += 1
        by_ev[e["chute"]]["ok"] += int(e["seen_ok"])
    tot_ok = sum(v["ok"] for v in by_ev.values())
    tot_ev = sum(v["tot"] for v in by_ev.values())
    print(f"  跌倒区间总数: {tot_ev}")
    print(f"  其中被该机位看见: {tot_ok} ({tot_ok / max(tot_ev, 1):.1%})")
    print(f"  完全没检出的区间（0 帧检出）: "
          f"{sum(1 for e in events if e['seen'] == 0)}")
    worst = sorted(events, key=lambda e: e["rate"])[:8]
    print("\n  最差的 8 个事件:")
    print(f"    {'场景':>8}{'机位':>6}{'帧区间':>16}{'检出/总':>11}{'率':>8}")
    for e in worst:
        print(f"    {'chute%02d' % e['chute']:>8}{e['cam']:>6}"
              f"{'[%d,%d]' % (e['start'], e['end']):>16}"
              f"{'%d/%d' % (e['seen'], e['frames']):>11}{e['rate']:>8.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "seen_frac_threshold": SEEN_FRAC,
        "per_cam": {str(c): per_cam[c] for c in cams},
        "per_chute": {f"chute{ch:02d}": per_chute[ch] for ch in chutes},
        "events": events,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
