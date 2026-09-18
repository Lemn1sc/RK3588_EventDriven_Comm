#!/usr/bin/env python
"""在 MCF 上标定单帧规则，和 GRU 做同口径对比 —— 检验「GRU 的价值是鲁棒性」。

── 要回答什么 ──

docs/08 §14.4：在 URFD cam0 上，一条标定过阈值的单帧 `aspect` 规则 AUC 0.9708，
与 GRU（0.9617~0.9767）**打平**。但同一类的 severity 规则在 MCF 上只有 0.5278，
而 GRU 仍有 0.8770。于是有个假设：

    **单帧规则峰值高但对视角敏感；GRU 峰值不更高但跨视角稳得多。**

本脚本在 MCF 上做两件事来检验它。

── 检验 A：同口径嵌套标定 ──

和 URFD 完全一样的做法（留一调阈值 → 只在留出段上取分数），只是折的单位从
「视频」换成**场景（chute）** —— 必须按场景，因为同一场景的 8 个机位拍的是
同一段跌倒，按机位分会泄漏。算出的 span 级 AUC 直接和 `mcf_gru.py` 的
GRU 数字比。

── 检验 B：逐机位最优阈值 ──

对每个机位**单独**标定最优阈值。如果 8 个机位的最优阈值差得很远，说明这条
规则是**机位相关**的 —— 换机位就得重标，这正是「不够鲁棒」的直接证据。
这一项不需要留出集，看的是阈值本身的离散程度。

用法：
    python scripts/mcf_calibrate_sf.py
    python scripts/mcf_calibrate_sf.py --out-json experiments/results/mcf_sf_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.metrics import roc_auc  # noqa: E402
import mcf_gru as mg                # noqa: E402  复用区间解析，保证与 GRU 同单位

FRAME_TABLE = mg.FRAME_TABLE
C = mg.C

# 两条单帧规则：都用边缘端已有的量，都不需要姿态
# 阈值网格要覆盖到「规则退化为恒真」为止，否则最优值会撞在边界上，
# 逐机位比较就变成比较「谁的边界更松」，看不出真实差异。
RULES = {
    "aspect":   ("aspect",   "lt", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
                                    1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 99.0]),
    "severity": ("severity", "gt", [0.60, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10,
                                    0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]),
}


def build_units():
    """→ [(场景, 机位, 区间序号, 标签, 该区间的帧值字典)]，单位与 mcf_gru 一致。"""
    npz = np.load(FRAME_TABLE, allow_pickle=True)
    spans = mg.load_spans()
    units = []
    for (ch, cam), sp in sorted(spans.items()):
        tab = npz[f"chute{ch:02d}/cam{cam}"]
        for si, (a, b, lab) in enumerate(sp):
            if b - a < mg.tg.WINDOW:      # 与 GRU 用同一批区间（短于窗口的跳过）
                continue
            units.append({
                "chute": ch, "cam": cam, "span": si, "y": float(lab),
                "aspect": np.nan_to_num(tab[a:b, C["aspect_ratio"]], nan=np.inf),
                "severity": np.nan_to_num(tab[a:b, C["severity"]], nan=0.0),
            })
    return units


def span_scores(units, field: str, mode: str, thr: float) -> np.ndarray:
    """区间分数 = 该区间内规则命中的帧占比（与 URFD 侧口径一致）。"""
    out = np.empty(len(units))
    for i, u in enumerate(units):
        v = u[field]
        out[i] = float(np.mean(v < thr if mode == "lt" else v >= thr))
    return out


def nested_auc(units, field: str, mode: str, grid) -> tuple[float, list]:
    """按场景留一：在其余场景上挑阈值 → 只在留出场景的区间上取分数。

    返回 (可对外报告的 AUC, 每折选中的阈值)。
    """
    y = np.array([u["y"] for u in units])
    chutes = sorted({u["chute"] for u in units})
    scores = np.zeros(len(units))
    picked = []
    for held in chutes:
        tr = [i for i, u in enumerate(units) if u["chute"] != held]
        te = [i for i, u in enumerate(units) if u["chute"] == held]
        if not tr or not te:
            continue
        sub = [units[i] for i in tr]
        yt = np.array([u["y"] for u in sub])
        best_t, best_a = None, -1.0
        for t in grid:
            a = roc_auc(yt, span_scores(sub, field, mode, t), check=False)
            if a > best_a:
                best_a, best_t = a, t
        picked.append(best_t)
        s = span_scores([units[i] for i in te], field, mode, best_t)
        for j, i in enumerate(te):
            scores[i] = s[j]
    return roc_auc(y, scores, check=False), picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / "experiments" / "results" / "mcf_sf_calibration.json")
    args = ap.parse_args()

    units = build_units()
    y = np.array([u["y"] for u in units])
    print(f"区间 {len(units)} 个（跌倒 {int(y.sum())} / 非跌倒 {len(units) - int(y.sum())}）"
          f"，场景 {len({u['chute'] for u in units})} 个，机位 8 个")
    print()

    gru = json.loads((ROOT / "experiments" / "results" / "mcf_gru.json").read_text(encoding="utf-8"))
    out = {"n_units": len(units), "n_fall": int(y.sum()), "rules": {}}

    print("=" * 76)
    print("检验 A · 同口径嵌套标定（按场景留一）")
    print("=" * 76)
    for name, (field, mode, grid) in RULES.items():
        # 固定阈值的参照（用现行值）
        fixed_thr = {"aspect": 0.7, "severity": 0.5}[name]
        a_fixed = roc_auc(y, span_scores(units, field, mode, fixed_thr), check=False)
        a_nest, picked = nested_auc(units, field, mode, grid)
        out["rules"][name] = {"auc_fixed": a_fixed, "fixed_threshold": fixed_thr,
                              "auc_nested": a_nest, "picked_thresholds": picked}
        print(f"\n[{name} 规则]  固定阈值 {fixed_thr} → AUC {a_fixed:.4f}")
        print(f"              嵌套标定      → AUC {a_nest:.4f}  （{a_nest - a_fixed:+.4f}）")

    print(f"\n[GRU 参照]  mcf_gru.py 的 det.v2 GRU AUC = {gru['det.v2']['auc_gs']:.4f}")
    best_rule = max(out["rules"], key=lambda k: out["rules"][k]["auc_nested"])
    b = out["rules"][best_rule]["auc_nested"]
    g = gru["det.v2"]["auc_gs"]
    print(f"\n  最好的单帧规则：{best_rule}（{b:.4f}）  vs  GRU（{g:.4f}）  →  差 {g - b:+.4f}")

    print()
    print("=" * 76)
    print("检验 B · 逐机位各自的最优阈值（阈值本身稳不稳）")
    print("=" * 76)
    for name, (field, mode, grid) in RULES.items():
        per = {}
        for cam in range(1, 9):
            sub = [u for u in units if u["cam"] == cam]
            ys = np.array([u["y"] for u in sub])
            if len(set(ys)) < 2:
                continue
            best_t, best_a = None, -1.0
            for t in grid:
                a = roc_auc(ys, span_scores(sub, field, mode, t), check=False)
                if a > best_a:
                    best_a, best_t = a, t
            per[cam] = {"thr": best_t, "auc": best_a}
        thrs = [v["thr"] for v in per.values()]
        aucs = [v["auc"] for v in per.values()]
        out["rules"][name]["per_camera"] = per
        # 判定基准用**有效网格**（去掉 99/0 这类「规则退化为恒真」的哨兵档），
        # 否则会拿一个无意义的极差当分母，什么都说成「一致」。
        finite = [t for t in grid if 0.0 < t < 10.0]
        span = max(finite) - min(finite)
        spread = max(thrs) - min(thrs)
        ratio = max(thrs) / max(min(thrs), 1e-9)
        out["rules"][name]["threshold_spread"] = spread
        out["rules"][name]["threshold_ratio"] = ratio
        print(f"\n[{name}]  8 个机位各自的最优阈值：{sorted(thrs)}")
        print(f"          阈值跨度 = {spread:.2f}（有效网格宽 {span:.2f}，占 {spread / span:.0%}）"
              f"   最大/最小 = {ratio:.1f}×")
        print(f"          各机位 AUC：{min(aucs):.3f} ~ {max(aucs):.3f}")
        print("          → " + ("阈值明显随**机位**变化，换机位要重标"
                               if spread > 0.3 * span
                               else "阈值在机位间比较一致"))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {args.out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
