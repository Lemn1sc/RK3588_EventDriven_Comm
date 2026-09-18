#!/usr/bin/env python
"""在干净数据上重标定远端 `judge` 规则的三条阈值。

── 为什么要重标定 ──

`judge()` 的三条线索各自带一个阈值，现行值是 `ASPECT_LYING=0.7` /
`TORSO_TILT_MAX_DEG=45` / `HIP_ANKLE_MIN_RATIO=0.3`，**是在旧拼接图数据上定的**。
而 docs/08 §12 查明那批数据的 `aspect` 被系统性放大一倍 —— 也就是说第①条
（框高宽比 < 0.7）在旧数据上命中率只有 **0.1%**，**形同虚设**；那套阈值实际上是
在「只有②③起作用」的前提下凑出来的。换到几何正确的数据上，①重新活了，
三条的平衡必然变了。

不重标定就无法判断**姿态到底值不值**（也就是「值不值得为它付卸载代价」）——
因为 judge 规则同时含框线索和姿态线索，现行阈值下的表现说明不了姿态的贡献。

── 口径 ──

- **真值**：URFD 官方逐帧标注（label: -1 未躺地 / 0 正在跌倒 / 1 躺地）。
- **优化目标**：**视频级 AUC**（与项目其余实验同口径）——
  每段视频的分数取「该帧规则命中的帧占比」，再对 30 跌倒 vs 40 日常算 AUC。
  刻意不用帧级 AUC：帧级会被「同一段视频内相邻帧高度相关」严重高估。
- **标定数据**：干净 `pose` 帧表（每帧都有姿态 —— 这正是远端确认服务在做的事）。

用法：
    python scripts/calibrate_judge.py              # 网格搜索 + 报告
    python scripts/calibrate_judge.py --list-only  # 只看当前阈值与逐条线索的区分度
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.metrics import roc_auc  # noqa: E402
import detect_fall_video as dfv     # noqa: E402  现行阈值从这里引，避免两处各写一份

CLEAN = ROOT / "data" / "recordings" / "detect_fall_clean"
POSES = CLEAN / "frames_pose.v1.npz"
OFFICIAL = ROOT / "data" / "datasets" / "URFD" / "_official"

COLS = ["index", "has_event", "event_type", "confidence", "severity",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "aspect_ratio",
        "pose_invoked", "torso_angle", "hip_ankle_ratio"]
C = {n: i for i, n in enumerate(COLS)}


def official_labels() -> dict[str, dict[int, int]]:
    """{序列: {帧号: label}}。label: -1 未躺地 / 0 正在跌倒 / 1 躺地。"""
    out: dict[str, dict[int, int]] = defaultdict(dict)
    for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        p = OFFICIAL / f
        if not p.exists():
            raise SystemExit(f"找不到官方标注：{p}\n  见 docs/08 §12.4")
        for r in csv.reader(open(p, encoding="utf-8")):
            if r and r[0]:
                out[r[0]][int(r[1])] = int(r[2])
    return dict(out)


def load_videos():
    """→ [(视频键, 是否跌倒, 逐帧 aspect / torso / hip, 逐帧官方 label)]。"""
    z = np.load(POSES, allow_pickle=True)
    lab = official_labels()
    out = []
    for k in sorted(z.files):
        if k.endswith("_label") or k.startswith("__"):
            continue
        t = z[k]
        seq = Path(k).stem.replace("-cam0", "")          # fall/fall-01-cam0.mp4 → fall-01
        idx = t[:, C["index"]].astype(int)
        out.append({
            "key": k,
            "y": 1.0 if k.startswith("fall/") else 0.0,
            "aspect": t[:, C["aspect_ratio"]],
            "torso": t[:, C["torso_angle"]],
            "hip": t[:, C["hip_ankle_ratio"]],
            "severity": t[:, C["severity"]],
            "label": np.array([lab.get(seq, {}).get(int(i) + 1, np.nan) for i in idx]),
        })
    return out


def hits(v: dict, t1: float, t2: float, t3: float, gate: bool) -> np.ndarray:
    """三线索 OR。gate=True 时第①条仍带 SUSPECTED 门（保持现行结构）。

    NaN 的处置与 `gru_features.judge_from_table` 一致：
    缺该几何量 → 该线索不参与判定（torso 用 -1、hip 用 1.0，比较恒不成立）。
    """
    a, to, h, sev = v["aspect"], v["torso"], v["hip"], v["severity"]
    c1 = np.nan_to_num(a, nan=np.inf) < t1
    if gate:
        c1 &= sev >= 0.5
    c2 = np.nan_to_num(to, nan=-1.0) > t2
    c3 = np.nan_to_num(h, nan=1.0) < t3
    return c1 | c2 | c3


def video_auc(vids, t1, t2, t3, gate) -> float:
    y = np.array([v["y"] for v in vids])
    s = np.array([float(np.mean(hits(v, t1, t2, t3, gate))) for v in vids])
    return roc_auc(y, s, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-only", action="store_true", help="只看现状与逐线索区分度")
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / "experiments" / "results" / "judge_calibration.json")
    args = ap.parse_args()

    vids = load_videos()
    print(f"载入 {len(vids)} 段（跌倒 {int(sum(v['y'] for v in vids))} / "
          f"日常 {int(len(vids) - sum(v['y'] for v in vids))}）")

    from common.event_vector import ASPECT_LYING
    cur = (ASPECT_LYING, dfv.TORSO_TILT_MAX_DEG, dfv.HIP_ANKLE_MIN_RATIO)
    print(f"\n现行阈值：aspect<{cur[0]}  torso>{cur[1]}  hip<{cur[2]}")

    # ── 逐条线索的区分度（帧级 AUC，用官方 label 的真值）──
    #   只作「哪条线索有信号」的参考；标定本身用的是视频级 AUC。
    print("\n=== 逐条线索的帧级区分度（正类 = 正在跌倒 or 躺地；负类 = 未躺地）===")
    print(f"{'线索':<26}{'帧级 AUC':>10}{'命中(正)':>10}{'命中(负)':>10}")
    per_crit = {}
    for name, arr, thr, mode in (
        ("① 框高宽比 < 0.7", "aspect", ASPECT_LYING, "lt"),
        ("② 躯干倾角 > 45°", "torso", dfv.TORSO_TILT_MAX_DEG, "gt"),
        ("③ 髋踝高差 < 0.3", "hip", dfv.HIP_ANKLE_MIN_RATIO, "lt"),
    ):
        pos, neg, sc = [], [], []
        for v in vids:
            m = np.isfinite(v["label"])
            if not m.any():
                continue
            yy = (v["label"][m] >= 0).astype(float)     # -1 → 0；0/1 → 1
            x = v[arr][m]
            x = np.nan_to_num(x, nan=(-1.0 if mode == "gt" else np.inf))
            sc.append(x); pos.append(yy)
            # 命中率
            hitm = (x > thr) if mode == "gt" else (x < thr)
            per_crit.setdefault(name, []).append((hitm, yy))
        if not sc:
            continue
        X = np.concatenate(sc); Y = np.concatenate(pos)
        # 分数方向必须与规则一致：规则是 x < thr 命中的，就用 -x 当分数，
        # 否则算出来的 AUC 会小于 0.5，看着像「这条线索是反向的」。
        if mode == "lt":
            X = -X
        print(f"{name:<26}{roc_auc(Y, X, check=False):>10.4f}", end="")
        hf = np.concatenate([h for h, _ in per_crit[name]])
        yf = np.concatenate([y for _, y in per_crit[name]])
        print(f"{np.mean(hf[yf == 1]):>10.1%}{np.mean(hf[yf == 0]):>10.1%}")

    if args.list_only:
        return

    # ── 网格搜索：最大化视频级 AUC ──
    print("\n=== 网格搜索（目标：视频级 AUC）===")
    grid1 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
    grid2 = [20, 30, 40, 45, 50, 60, 70, 80]
    grid3 = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    best = {}
    for gate in (True, False):
        b = (None, -1.0)
        for t1, t2, t3 in product(grid1, grid2, grid3):
            a = video_auc(vids, t1, t2, t3, gate)
            if a > b[1]:
                b = ((t1, t2, t3), a)
        best["gate" if gate else "nogate"] = b
        tag = "①带 SUSPECTED 门（现行结构）" if gate else "①不带门（直接用 aspect）"
        print(f"  {tag}: {b[0]}  →  视频级 AUC {b[1]:.4f}")

    cur_auc_g = video_auc(vids, *cur, True)
    cur_auc_n = video_auc(vids, *cur, False)
    print(f"\n现行阈值 {cur}: 带门 AUC {cur_auc_g:.4f}   不带门 AUC {cur_auc_n:.4f}")
    for k in ("gate", "nogate"):
        print(f"  重标定后（{k}）: {best[k][1]:+.4f}  "
              f"（{best[k][1] - (cur_auc_g if k == 'gate' else cur_auc_n):+.4f}）")

    # ── 留一嵌套标定：在 69 段上调阈值，在留出的那 1 段上取分数 ──
    #   上面的网格搜索是在**同一批数据**上挑最优阈值再原地评估，
    #   那个 0.9487 是乐观有偏的。只有嵌套才给出可对外报告的数字。
    print("\n=== 留一嵌套标定（在 69 段上调阈值 → 在留出段上取分数）===")
    nested = {}
    for gate in (True, False):
        y = np.array([v["y"] for v in vids])
        scores = np.zeros(len(vids))
        for i in range(len(vids)):
            tr = [v for j, v in enumerate(vids) if j != i]
            best_t, best_a = None, -1.0
            for t1, t2, t3 in product(grid1, grid2, grid3):
                a = video_auc(tr, t1, t2, t3, gate)
                if a > best_a:
                    best_a, best_t = a, (t1, t2, t3)
            scores[i] = float(np.mean(hits(vids[i], *best_t, gate)))
        nested["gate" if gate else "nogate"] = roc_auc(y, scores, check=False)
        tag = "带 SUSPECTED 门" if gate else "不带门"
        print(f"  {tag}: 嵌套标定后的视频级 AUC = {nested['gate' if gate else 'nogate']:.4f}")

    # ── 消融：0.9458 里有多少是姿态挣的？──
    #   对每个线索子集**各自重新嵌套标定**，这样比较的才是「这条线索本身值多少」，
    #   而不是「在别的线索已经校准好的前提下再加它」。
    print("\n=== 线索消融（每个子集各自嵌套标定，留出集上取值）===")
    subsets = {
        "① 只用框": (0,), "② 只用躯干": (1,), "③ 只用髋踝": (2,),
        "①+②": (0, 1), "①+③": (0, 2), "②+③": (1, 2), "①+②+③": (0, 1, 2),
    }
    grids = (grid1, grid2, grid3)
    abl = {}
    y = np.array([v["y"] for v in vids])
    print(f"{'线索组合':<14}{'嵌套 AUC':>10}   相对只用框")
    base = None
    for label, idxs in subsets.items():
        scores = np.zeros(len(vids))
        for i in range(len(vids)):
            tr = [v for j, v in enumerate(vids) if j != i]
            best_t, best_a = None, -1.0
            for combo in product(*[grids[k] for k in idxs]):
                thr = [None, None, None]
                for k, val in zip(idxs, combo):
                    thr[k] = val
                a = video_auc(tr, thr[0] or 0.9, thr[1] or 80, thr[2] or 0.15, gate=False)
                if a > best_a:
                    best_a, best_t = a, combo
            thr = [None, None, None]
            for k, val in zip(idxs, best_t):
                thr[k] = val
            m = np.zeros(len(vids[i]["aspect"]), dtype=bool)
            if 0 in idxs:
                m |= np.nan_to_num(vids[i]["aspect"], nan=np.inf) < thr[0]
            if 1 in idxs:
                m |= np.nan_to_num(vids[i]["torso"], nan=-1.0) > thr[1]
            if 2 in idxs:
                m |= np.nan_to_num(vids[i]["hip"], nan=1.0) < thr[2]
            scores[i] = float(np.mean(m))
        a = roc_auc(y, scores, check=False)
        abl[label] = a
        if label == "① 只用框":
            base = a
        print(f"{label:<14}{a:>10.4f}   {'' if label == '① 只用框' else f'{a - base:+.4f}'}")

    # ── 口径对齐：severity 规则的阈值也要标定，否则和「标定过的 aspect 规则」不可比 ──
    #   一直引用的 0.8908 用的是固定 `SEVERITY_SUSPECT=0.5`，那是**没标定过**的；
    #   拿它去和标定过的 ①（0.9604）比，等于让单帧规则吃两套待遇。
    print("\n=== 口径对齐：把 severity 规则的阈值也标定（同样嵌套）===")
    gs = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    sev_scores = np.zeros(len(vids))
    for i in range(len(vids)):
        tr = [v for j, v in enumerate(vids) if j != i]
        best_t, best_a = None, -1.0
        for t in gs:
            yt = np.array([v["y"] for v in tr])
            st = np.array([float(np.mean(np.nan_to_num(v["severity"], nan=0.0) >= t))
                           for v in tr])
            a = roc_auc(yt, st, check=False)
            if a > best_a:
                best_a, best_t = a, t
        sev_scores[i] = float(np.mean(np.nan_to_num(vids[i]["severity"], nan=0.0) >= best_t))
    sev_nested = roc_auc(y, sev_scores, check=False)
    y_all = np.array([v["y"] for v in vids])
    sev_fixed = roc_auc(y_all, np.array([float(np.mean(np.nan_to_num(v["severity"], nan=0.0) >= 0.5))
                                         for v in vids]), check=False)
    print(f"  severity 固定阈值 0.5（现引用值）: {sev_fixed:.4f}")
    print(f"  severity 嵌套标定阈值:            {sev_nested:.4f}")
    print(f"  ① 只用框 嵌套标定:                {abl.get('① 只用框', float('nan')):.4f}")
    print("\n  → 三者都是「单帧、不需要姿态」的规则，应当互相比较；")
    print("    要与 GRU 比，用的是其中最高的那个。")
    abl["severity_fixed0.5"] = sev_fixed
    abl["severity_nested"] = sev_nested

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "current": {"aspect": cur[0], "torso": cur[1], "hip": cur[2],
                    "auc_gated": cur_auc_g, "auc_ungated": cur_auc_n},
        "best_in_sample": {k: {"thresholds": list(v[0]), "auc": v[1]} for k, v in best.items()},
        "nested_lovo": nested,
        "ablation_nested": abl,
        "note": "真值 = URFD 官方逐帧标注；目标 = 视频级 AUC；数据 = 干净 pose 帧表。"
                "best_in_sample 是原地挑阈值再评估，乐观有偏；应引用 nested_lovo。",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {args.out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
