"""LOVO 结果的配对自助法分析（不依赖 torch/ultralytics，纯 numpy）。

用法：
    python scripts/train_gru_fall.py --protocol lovo --out-json experiments/results/metrics_lovo.json
    python scripts/analyze_lovo.py

── 为什么需要这一步 ──

`train_gru_fall.py --protocol lovo` 把 AUC 的最小刻度从 0.0125 细化到 0.00083，
但**刻度细不等于差异显著**。两个特征集差 0.0033，对应 1200 个「正负样本对」里只有
4 对翻转 —— 是真差异还是噪声？这要靠重采样来回答。

重采样的单位必须是**视频**而不是样本对：同一段视频贡献的 30 个正样本对并不独立，
按对重采样会把置信区间算得远窄于实际。

两个特征集用**同一组**重采样索引（配对），从而得到 AUC 差自身的分布 ——
配对能消掉「这次抽到了哪批视频」这个共同噪声，比分别比较两个区间灵敏得多。

⚠ 局限：这里的自助法只重采样**测试视频**，没有重采样「训练过程」。所以它反映的是
「换一批测试视频结果会怎么变」，不包含「换一次训练随机性结果会怎么变」。真实的
不确定性还要更大一些。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.metrics import roc_auc  # noqa: E402

#: 想比较的配对。方向是「后者 − 前者」。
PAIRS = [("det.v1", "det.v2a"), ("det.v2a", "det.v2"), ("det.v1", "det.v2"),
         ("pose.v1", "pose.v2"), ("det.v2", "pose.v2"), ("det.v1", "pose.v1")]


def load_scores(path: Path, how: str = "max"):
    """读 LOVO json → (特征集名列表, 视频键, 标签, {特征集: 分数向量})。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("protocol") != "lovo":
        raise SystemExit(f"这个 json 不是 LOVO 结果（protocol={d.get('protocol')}）：{path}\n"
                         f"  → 先跑 python scripts/train_gru_fall.py --protocol lovo")
    names = list(d["per_seed"])
    per = {n: d["per_seed"][n][0]["per_video_score"][how] for n in names}
    keys = sorted(per[names[0]])
    labels = np.array([1.0 if k.startswith("fall/") else 0.0 for k in keys])
    return names, keys, labels, {n: np.array([per[n][k] for k in keys]) for n in names}


def bootstrap(names, labels, scores, n_boot: int, seed: int = 0):
    """以视频为单位重采样，返回 {特征集: 每轮 AUC 数组}。

    退化重采样（抽到的全是同一类，AUC 无定义）直接沿用原始索引，相当于该轮不计入
    额外噪声 —— 在 70 段 / 30 正的规模下这种情况几乎不会发生。
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot = {name: np.empty(n_boot) for name in names}
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        if labels[idx].sum() in (0, n):
            idx = np.arange(n)
        y = labels[idx]
        for name in names:
            boot[name][b] = roc_auc(y, scores[name][idx])
    return boot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=ROOT / "experiments/results/metrics_lovo.json")
    ap.add_argument("--agg", default="max", help="视频级聚合方式（max/mean/top3mean）")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names, keys, labels, scores = load_scores(args.json, args.agg)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    boot = bootstrap(names, labels, scores, args.n_boot, args.seed)
    point = {n: roc_auc(labels, scores[n]) for n in names}

    print(f"LOVO 自助法（{args.n_boot} 次，以 {len(keys)} 段视频为单位重采样，"
          f"聚合方式 {args.agg}）")
    print(f"分母 {n_pos} 跌倒 / {n_neg} 日常 → AUC 最小刻度 1/{n_pos * n_neg} = "
          f"{1 / (n_pos * n_neg):.5f}\n")

    print(f"{'特征集':10s} {'AUC':>7s}   {'95% 置信区间':>20s}")
    for n in names:
        lo, hi = np.percentile(boot[n], [2.5, 97.5])
        print(f"{n:10s} {point[n]:7.4f}   [{lo:.4f}, {hi:.4f}]")

    print("\n配对 AUC 差（同一次重采样下相减，消掉「抽到哪批视频」这个共同噪声）")
    for a, b in PAIRS:
        if a not in boot or b not in boot:
            continue
        diff = boot[b] - boot[a]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p = float((diff <= 0).mean())
        p = min(p, 1 - p) * 2                       # 双侧 p
        verdict = "显著" if (lo > 0 or hi < 0) else "不显著（区间跨 0）"
        print(f"  {b:9s} − {a:9s} = {point[b] - point[a]:+.4f}   "
              f"95%CI [{lo:+.4f}, {hi:+.4f}]   p≈{p:.3f}  → {verdict}")

    print("\n单个工作点的脆弱性：零误报阈值由日常视频里分数最高的一段决定")
    for n in names:
        s = scores[n]
        neg_max = np.where(labels == 0, s, -np.inf)
        thr = float(neg_max.max())
        who = keys[int(np.argmax(neg_max))]
        hit = int(((s > thr) & (labels == 1)).sum())
        print(f"  {n:10s} 阈值被 {who.split('/')[1]:18s} 顶到 {thr:.4f} → 只能捞回 {hit}/{n_pos}")
    print("\n注意：不同特征集的阈值是被**不同的**日常视频顶上去的，所以横向比较这个"
          "工作点等于在比较「谁更倒霉地碰上一段难例」，不是一个稳健的评测口径。")


if __name__ == "__main__":
    main()
