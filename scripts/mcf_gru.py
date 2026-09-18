#!/usr/bin/env python
"""在 MCF 上做「GRU vs 单帧基线」的对照 —— 用另一个数据集检验 docs/08 §13 的结论。

── 为什么要做 ──

§13 在干净 URFD 上得到的结论是：**单帧基线是强基线（AUC 0.89），GRU 相对它仍有
显著但有限的增益（+0.078，p≈0.001）**。这个结论是在 URFD cam0 一个视角、
一套受试上得到的。换一个**完全不同**的数据集能不能复现，是它可信度的关键。

MCF 的差异足够大：8 个机位、高地俯视、广角鱼眼、另一批受试与房间。而且它自带
逐帧标注，不需要像 URFD 那样靠视频级弱监督。

── 与 URFD 管线的三点不同（都是数据性质决定的）──

1. **标签是逐帧的**，不是视频级的。所以单位是「标注区间」而不是「视频」：
   每个连续标注区间取一个窗口，标签取该区间的标签。
2. **LOVO 按场景（chute）划分，不能按 (场景,机位)**。同一个场景的 8 个机位拍的
   是**同一段跌倒**，按机位划分会把同一事件放进训练和测试两边 —— 分数会虚高。
   这是本脚本最容易写错的地方。
3. **自助法也按场景重采样**（而不是按区间），理由同上：区间在场景内是相关的。

用法：
    python scripts/mcf_gru.py --featsets det.v1 det.v2a det.v2
    python scripts/mcf_gru.py --out-json experiments/results/mcf_gru.json
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

import gru_features as gf          # noqa: E402  特征配方 / 单帧基线 / 帧表列定义
import train_gru_fall as tg        # noqa: E402  GRU 模型与训练循环
from common.metrics import roc_auc  # noqa: E402

FRAME_TABLE = ROOT / "data" / "recordings" / "mcf" / "mcf_frames.v1.npz"
C = gf.C


def load_spans() -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    """从帧表还原连续标注区间：{(场景,机位): [(起始下标, 结束下标, 标签)]}。

    帧表是按区间顺序拼的，原始帧号存在 `index` 列里 —— 用「帧号是否连续」判断
    区间边界，而不是位置下标。标签列单存在 `<key>_label`。
    """
    npz = np.load(FRAME_TABLE, allow_pickle=True)
    out = {}
    for k in npz.files:
        if k.endswith("_label") or k.startswith("__"):
            continue
        tab, lab = npz[k], npz[f"{k}_label"]
        fr = tab[:, C["index"]].astype(int)
        ch, cam = k.replace("chute", "").split("/cam")
        spans, i, n = [], 0, len(lab)
        while i < n:
            j = i
            while j + 1 < n and lab[j + 1] == lab[i] and fr[j + 1] == fr[j] + 1:
                j += 1
            spans.append((i, j + 1, int(lab[i])))
            i = j + 1
        out[(int(ch), int(cam))] = spans
    return out


def build_units(featset: str):
    """→ [(场景, 机位, 区间序号, 标签, 窗口张量, 单帧基线分数)]。

    单帧基线分数沿用项目口径：窗口内「severity ≥ 阈值」的帧占比，并额外记录
    max severity（两种聚合都留着，报告时以前者为主，与 URFD 的 tpr0_sf 同源）。
    """
    fs = gf.FEATSETS[featset]
    npz = np.load(FRAME_TABLE, allow_pickle=True)
    spans = load_spans()
    units = []
    for (ch, cam), sp in sorted(spans.items()):
        key = f"chute{ch:02d}/cam{cam}"
        tab = npz[key]
        feats = np.nan_to_num(fs.build(tab).astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
        base = gf.single_frame_baselines(tab)["severity>=0.5"]
        sev = np.nan_to_num(tab[:, C["severity"]].astype(float), nan=0.0)
        for si, (a, b, lab) in enumerate(sp):
            seg = feats[a:b]
            if len(seg) < tg.WINDOW:
                continue                      # 短于窗口的区间切不出窗口，跳过
            win = seg[:tg.WINDOW][None, ...]  # 31 帧的区间只放得下一个窗口
            units.append((ch, cam, si, lab, win,
                          float(base[a:a + tg.WINDOW].mean()),
                          float(sev[a:a + tg.WINDOW].max())))
    return units


def lovo(units, featset: str):
    """按**场景**留一：每折留出一个 chute 的全部机位与区间。

    返回每个区间一行 [场景, 标签, GRU 分数, 基线分数, max severity 分数]。
    """
    chutes = sorted({u[0] for u in units})
    rows = []
    for held in chutes:
        tr = [u for u in units if u[0] != held]
        te = [u for u in units if u[0] == held]
        if not tr or not te:
            continue
        tx = np.concatenate([u[4] for u in tr])
        ty = np.array([u[3] for u in tr], dtype=np.float32)
        model = tg.train_model(tx, ty, gf.FEATSETS[featset].dim, verbose=False)
        model.eval()
        import torch
        with torch.no_grad():
            for u in te:
                p = float(torch.sigmoid(model(torch.tensor(u[4], device=tg.DEVICE))).cpu().numpy()[0])
                rows.append([u[0], u[3], p, u[5], u[6]])
    return np.array(rows, dtype=object) if rows else None


def bootstrap_by_scenario(arr, n_boot=4000, seed=0):
    """按场景重采样（不是按区间）—— 场景内区间相关，按区间重采样会低估不确定性。"""
    chutes = sorted({r[0] for r in arr})
    y = np.array([float(r[1]) for r in arr])
    gs = np.array([float(r[2]) for r in arr])
    sf = np.array([float(r[3]) for r in arr])
    rng = np.random.default_rng(seed)
    idx_of = {c: np.where(np.array([r[0] for r in arr]) == c)[0] for c in chutes}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(chutes, len(chutes), replace=True)
        idx = np.concatenate([idx_of[c] for c in pick])
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc(y[idx], gs[idx], check=False) - roc_auc(y[idx], sf[idx], check=False))
    if not diffs:
        return None
    d = np.array(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--featsets", nargs="+", default=["det.v1", "det.v2a", "det.v2"],
                    choices=[n for n in gf.FEATSETS if n.startswith("det")])
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / "experiments" / "results" / "mcf_gru.json")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    if not FRAME_TABLE.exists():
        raise SystemExit(f"找不到 MCF 帧表：{FRAME_TABLE}\n  先跑：python scripts/mcf_detect.py")

    print("=" * 76)
    print("MCF：GRU vs 单帧基线（LOVO 按场景留一，8 机位同场景不跨折）")
    print("=" * 76)
    out = {}
    for featset in args.featsets:
        units = build_units(featset)
        n_fall = sum(1 for u in units if u[3] == 1)
        print(f"\n[{featset}] 区间 {len(units)} 个（跌倒 {n_fall} / 非跌倒 {len(units) - n_fall}）"
              f"，场景 {len({u[0] for u in units})} 个")
        arr = lovo(units, featset)
        if arr is None:
            print("  没有可用区间，跳过")
            continue
        y = np.array([float(r[1]) for r in arr])
        gs = np.array([float(r[2]) for r in arr])
        sf = np.array([float(r[3]) for r in arr])
        a_gs, a_sf = roc_auc(y, gs, check=False), roc_auc(y, sf, check=False)
        bs = bootstrap_by_scenario(arr, n_boot=args.n_boot)
        print(f"  单帧基线 AUC = {a_sf:.4f}")
        print(f"  GRU      AUC = {a_gs:.4f}   （增益 {a_gs - a_sf:+.4f}）")
        if bs:
            lo, hi, p = bs
            print(f"  配对自助法（按场景重采样 {args.n_boot} 次）: "
                  f"95%CI [{lo:+.4f}, {hi:+.4f}]  p≈{p:.4f}  "
                  f"→ {'显著' if lo > 0 else '不显著（区间跨 0）'}")
        out[featset] = {"auc_gs": a_gs, "auc_sf": a_sf,
                        "n_units": len(arr), "n_fall": int(y.sum()),
                        "ci": bs[:2] if bs else None, "p": bs[2] if bs else None}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {args.out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
