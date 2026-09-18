#!/usr/bin/env python
"""系统地比一遍「哪种单帧规则最好」——两个数据集、多个候选量、嵌套标定。

── 为什么要做 ──

此前只单项测过 `aspect`（框高宽比）。但 cam0 上早就实测过「跌倒时高宽比几乎不变」
（§3），而**框中心高度 `cy`** 才是真正可靠的线索 —— 只是它一直以「相对因果基线的
突降比」形式出现在 GRU 特征里，从没被当作**单帧规则**评估过。

本脚本把候选量都摆在同一口径下比：

| 候选 | 直觉 |
|---|---|
| `aspect = h/w` | 人躺下时框变扁 |
| `cy`（框中心高度） | 人躺下时框在画面里更靠下 |
| `size = sqrt(w·h)` | 躺姿的投影面积变化 |
| `confidence` | 检测置信度本身（躺地时偏低，见 §11.6） |
| `severity` | 现行规则：置信度 × 高宽比异常度 |

**口径**：与项目其余实验一致 —— 单位取「视频」（URFD）或「标注区间」（MCF），
分数取该单位内规则命中的帧占比；阈值用**留一嵌套标定**（在 N−1 折上调、
只在留出折上取分数），避免原地挑阈值的乐观偏差。

MCF 的折按**场景**划分（同场景 8 机位拍的是同一段跌倒，按机位分会泄漏）。

用法：
    python scripts/best_single_frame_rule.py
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

CLEAN = ROOT / "data" / "recordings" / "detect_fall_clean" / "frames_det.v1.npz"
MCF = ROOT / "data" / "recordings" / "mcf" / "mcf_frames.v1.npz"
OFFICIAL = ROOT / "data" / "datasets" / "URFD" / "_official"

COLS = ["index", "has_event", "event_type", "confidence", "severity",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "aspect_ratio",
        "pose_invoked", "torso_angle", "hip_ankle_ratio"]
C = {n: i for i, n in enumerate(COLS)}

# 候选量：(名字, 由帧表算出来的数组, 方向)   方向 'lt' = 越小越可疑
def quantities(t: np.ndarray) -> dict[str, np.ndarray]:
    has = t[:, C["has_event"]] > 0.5
    x1, y1, x2, y2 = (t[:, C[k]] for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
    w = np.maximum(x2 - x1, 0.0)
    h = np.maximum(y2 - y1, 0.0)
    nan = np.nan
    return {
        "aspect": np.where(has, np.nan_to_num(t[:, C["aspect_ratio"]], nan=nan), nan),
        "cy": np.where(has, (np.nan_to_num(y1, nan=nan) + np.nan_to_num(y2, nan=nan)) / 2, nan),
        "size": np.where(has, np.sqrt(w * h), nan),
        "conf": np.where(has, t[:, C["confidence"]], nan),
        "severity": np.where(has, t[:, C["severity"]], nan),
    }


# 每个量的阈值网格。
# ⚠ 网格**必须够密**，且两端要覆盖到「规则退化为恒真/恒假」的哨兵档（-1.0 / 99.0）。
#   实测过一次教训：网格里漏掉 [1.6, 2.0, 3.0] 三档，正好跳过 MCF 的最优点，
#   AUC 从 0.7594 掉到 0.7216 —— 而且不会报任何错，只会给一个偏低的数。
#   下面的 run() 还会检查最优阈值是否落在端点，落在端点就提示网格可能没覆盖到最优点。
def _lin(a, b, step):
    n = int(round((b - a) / step))
    return [round(a + i * step, 4) for i in range(n + 1)]


GRID = {
    "aspect":   (_lin(0.4, 3.0, 0.1) + [99.0], "lt"),
    "cy":       (_lin(0.10, 0.95, 0.025) + [-1.0], "gt"),
    "size":     (_lin(0.02, 0.80, 0.02) + [-1.0], "gt"),
    "conf":     (_lin(0.10, 0.95, 0.025) + [-1.0], "gt"),
    "severity": (_lin(0.02, 0.95, 0.02) + [-1.0], "gt"),
}


def load_urfd():
    """→ 单位列表，每个单位带 (折的键, 标签, {量名: 逐帧数组})。"""
    z = np.load(CLEAN, allow_pickle=True)
    lab = defaultdict(dict)
    for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        for r in csv.reader(open(OFFICIAL / f, encoding="utf-8")):
            if r and r[0]:
                lab[r[0]][int(r[1])] = int(r[2])
    units = []
    for k in sorted(z.files):
        if k.endswith("_label") or k.startswith("__"):
            continue
        units.append({"fold": k, "y": 1.0 if k.startswith("fall/") else 0.0,
                      "q": quantities(z[k])})
    return units, "视频"


def load_mcf():
    z = np.load(MCF, allow_pickle=True)
    units = []
    for k in sorted(z.files):
        if k.endswith("_label") or k.startswith("__"):
            continue
        ch = int(k.replace("chute", "").split("/")[0])
        t, lab = z[k], z[f"{k}_label"]
        fr = t[:, C["index"]].astype(int)
        q = quantities(t)
        i, n = 0, len(lab)
        while i < n:                       # 按连续标注区间切
            j = i
            while j + 1 < n and lab[j + 1] == lab[i] and fr[j + 1] == fr[j] + 1:
                j += 1
            if j + 1 - i >= 30:            # 与 GRU 侧同一批区间（≥ 一个窗口）
                units.append({"fold": ch, "y": float(lab[i]),
                              "q": {kk: vv[i:j + 1] for kk, vv in q.items()}})
            i = j + 1
    return units, "标注区间"


def score(units, name, thr, mode) -> np.ndarray:
    out = np.empty(len(units))
    for i, u in enumerate(units):
        v = u["q"][name]
        hit = (v < thr) if mode == "lt" else (v > thr)
        out[i] = float(np.nanmean(hit)) if len(v) else 0.0
    return out


def nested(units, name, grid, mode) -> float:
    """留一嵌套标定：在其余折上调阈值 → 只在留出折上取分数。"""
    y = np.array([u["y"] for u in units])
    folds = sorted({u["fold"] for u in units})
    sc = np.zeros(len(units))
    for held in folds:
        tr = [i for i, u in enumerate(units) if u["fold"] != held]
        te = [i for i, u in enumerate(units) if u["fold"] == held]
        if not tr or not te:
            continue
        sub = [units[i] for i in tr]
        yt = np.array([u["y"] for u in sub])
        bt, ba = None, -1.0
        for t in grid:
            a = roc_auc(yt, score(sub, name, t, mode), check=False)
            if a > ba:
                ba, bt = a, t
        s = score([units[i] for i in te], name, bt, mode)
        for j, i in enumerate(te):
            sc[i] = s[j]
    return roc_auc(y, sc, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / "experiments" / "results" / "best_sf_rule.json")
    args = ap.parse_args()

    out = {}
    for tag, loader in (("URFD cam0", load_urfd), ("MCF", load_mcf)):
        units, unit_name = loader()
        y = np.array([u["y"] for u in units])
        print(f"\n{'=' * 70}\n{tag}  单位={unit_name}  n={len(units)}"
              f"（正 {int(y.sum())} / 负 {len(units) - int(y.sum())}）"
              f"  折数={len({u['fold'] for u in units})}\n{'=' * 70}")
        print(f"{'候选量':<10}{'方向':<6}{'嵌套 AUC':>10}   最优阈值(众数)")
        res = {}
        for name, (grid, mode) in GRID.items():
            a = nested(units, name, grid, mode)
            # 取众数阈值只为展示
            picks = []
            folds = sorted({u["fold"] for u in units})
            for held in folds:
                tr = [u for u in units if u["fold"] != held]
                if not tr:
                    continue
                yt = np.array([u["y"] for u in tr])
                bt, ba = None, -1.0
                for t in grid:
                    v = roc_auc(yt, score(tr, name, t, mode), check=False)
                    if v > ba:
                        ba, bt = v, t
                picks.append(bt)
            from collections import Counter
            mode_thr = Counter(picks).most_common(1)[0][0]
            # 最优阈值落在网格端点 → 说明真正的最优可能在网格外，这一格偏乐观或偏悲观
            interior = grid[1:-1]
            flag = "  [!] 落在网格端点，最优可能在网格外" if mode_thr not in interior else ""
            res[name] = {"auc": a, "mode_threshold": mode_thr, "dir": mode, "at_boundary": bool(flag)}
            print(f"{name:<10}{mode:<6}{a:>10.4f}   {mode_thr}{flag}")
        out[tag] = {"unit": unit_name, "n": len(units), "rules": res}
        best = max(res, key=lambda k: res[k]["auc"])
        print(f"\n  → 最好的是 **{best}**（{res[best]['auc']:.4f}，阈值 {res[best]['mode_threshold']}）")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {args.out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
