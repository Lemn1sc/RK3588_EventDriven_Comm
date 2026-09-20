"""第 1 步 · 评测指标回归自检（不依赖 torch/ultralytics，纯 numpy）。

用法：python scripts/check_metrics.py   看到 "评测指标 OK ✓" 即通过。

重点验的是"并列分数"这件事 —— 它是真实数据里发生过的病态情况：
det 模式的基线分数 = "该视频中触发帧的占比"，实测 70 段里 **67 段并列在 0.0**，
只有 3 段非零。旧实现用不稳定的 np.argsort 排序，并列样本的先后任意，
于是 AUC 成了排序的副产品（实测打乱输入顺序，AUC 在 0.20~0.84 之间跳）。
本脚本把这些性质固化成断言。
"""
import io
import sys
from pathlib import Path

import numpy as np

# Windows 控制台默认 GBK：中文与 ✓ 会乱码。stdout / stderr **都要设** ——
# 报错信息走的是 stderr，只设 stdout 的话出错时提示反而是乱码的，最难读的时候读不了。
for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common.metrics import (  # noqa: E402
    agg_scores, auc_rank, roc_auc, roc_curve, ties_report, tpr_at_fpr,
)


def check_hand_computed():
    """小手算：无并列时 阶梯积分 == 秩公式 == 手算值。"""
    labels = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.4, 0.35, 0.1])
    # 手算：正样本(0.9,0.8,0.4) 与 负样本(0.35,0.1) 两两比较
    # 0.9>0.35,0.1 / 0.8>0.35,0.1 / 0.4>0.35,0.1 → 6 对全对 → AUC = 6/6 = 1.0
    assert abs(auc_rank(labels, scores) - 1.0) < 1e-12
    assert abs(roc_auc(labels, scores) - 1.0) < 1e-12

    # 反过来：正样本分数全低于负样本 → AUC = 0
    assert abs(roc_auc(labels, -scores) - 0.0) < 1e-12

    # 完全分不开（全同分）→ AUC = 0.5
    assert abs(roc_auc([1, 0, 1, 0], [0.5] * 4) - 0.5) < 1e-12
    print("  [1] 无并列时 阶梯积分 == 秩公式 == 手算值 ✓")


def make_real_pathological():
    """复刻真实病态分布：3 段高分跌倒 + 27 段 0 分跌倒 + 40 段 0 分日常。

    数据来自 data/recordings/detect_fall/gru_features_det.npz 的实测结构。
    """
    scores = np.array([0.534, 0.554, 0.681] + [0.0] * 27 + [0.0] * 40)
    labels = np.array([1] * 30 + [0] * 40)
    return labels, scores


def check_pathological_auc():
    """真实病态分布下 AUC 唯一且等于 0.55（手算见 check_metrics 文档）。"""
    labels, scores = make_real_pathological()

    # 秩公式手算：27 个并列正样本 + 40 个并列负样本 占秩 1..67，平均秩 34；
    # 3 个高分正样本占秩 68,69,70。
    #   (68+69+70 + 27*34 - 30*31/2) / (30*40)
    # = (207 + 918 - 465) / 1200 = 660/1200 = 0.55
    expected = (207 + 918 - 465) / 1200
    assert abs(expected - 0.55) < 1e-12

    got = roc_auc(labels, scores)
    assert abs(got - 0.55) < 1e-9, f"期望 0.55，实得 {got}"
    print(f"  [2] 真实病态分布（67 段并列）AUC = {got:.4f}（唯一确定值）✓")


def check_permutation_invariance():
    """★核心回归★ 打乱输入顺序，AUC 必须一模一样。

    这正是旧实现失败的地方：它用不稳定的 argsort，同一份数据换个顺序
    就能报出 0.20~0.84 之间的任意值。
    """
    labels, scores = make_real_pathological()
    base = roc_auc(labels, scores)

    rng = np.random.RandomState(0)
    seen = set()
    for _ in range(200):
        p = rng.permutation(len(scores))
        seen.add(round(roc_auc(labels[p], scores[p]), 12))
    assert seen == {round(base, 12)}, f"打乱顺序后 AUC 变了：{sorted(seen)}"

    # 只打乱"并列块内部"的相邻样本（旧实现最容易被这种输入影响）
    order = np.argsort(-scores, kind="mergesort")
    for _ in range(200):
        # 在并列的 0 分块（占 67 个）内部随机洗牌
        blk = order[3:].copy()
        rng.shuffle(blk)
        perm = np.concatenate([order[:3], blk])
        assert abs(roc_auc(labels[perm], scores[perm]) - base) < 1e-12
    print("  [3] 打乱顺序 / 打乱并列块内部，AUC 恒定不变（旧实现会变）✓")


def check_tpr_at_zero_fpr():
    """@零误报的诚实值：只有 3 段高分跌倒能在不误报任何日常的前提下被检出。"""
    labels, scores = make_real_pathological()
    fpr, tpr = roc_curve(labels, scores)
    got = tpr_at_fpr(fpr, tpr, 0.0)
    assert abs(got - 3 / 30) < 1e-12, f"期望 0.1，实得 {got}"
    print(f"  [4] @FPR=0 时检出率 = {got * 100:.0f}%（3/30，诚实值）✓")


def check_curve_shape():
    """阶梯必须单调不减、从 (0,0) 到 (1,1)、且同分样本只产生一个工作点。"""
    labels, scores = make_real_pathological()
    fpr, tpr = roc_curve(labels, scores)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert abs(fpr[-1] - 1.0) < 1e-12 and abs(tpr[-1] - 1.0) < 1e-12
    assert np.all(np.diff(fpr) >= -1e-12), "FPR 非单调"
    assert np.all(np.diff(tpr) >= -1e-12), "TPR 非单调"
    # 70 段只有 4 个不同取值 → 阶梯最多 4 个工作点 + 起点 = 5 个点
    assert len(fpr) <= 5, f"并列块没被合并：折线有 {len(fpr)} 个点"

    # 无并列时工作点数应等于样本数 + 1
    f2, t2 = roc_curve([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert len(f2) == 5, len(f2)
    print("  [5] 阶梯单调、首末点正确、并列块已合并 ✓")


def check_ties_report():
    labels, scores = make_real_pathological()
    rep = ties_report(labels, scores)
    assert rep["n"] == 70 and rep["n_distinct"] == 4, rep
    assert rep["max_group"] == 67, rep
    assert rep["n_tied_samples"] == 67, rep
    assert rep["n_tied_groups"] == 1, rep
    # 无并列时不应误报
    assert ties_report([1, 0], [0.9, 0.1])["max_group"] == 1
    print(f"  [6] 并列体检：{rep['n']} 段 / {rep['n_distinct']} 个取值 / "
          f"最大并列组 {rep['max_group']} 段 ✓")


def check_agg():
    """三种聚合手算。聚合方式会显著影响结论，所以必须钉死语义。"""
    p = [0.1, 0.9, 0.5, 0.3]
    assert abs(agg_scores(p, "max") - 0.9) < 1e-12
    assert abs(agg_scores(p, "mean") - 0.45) < 1e-12
    # top3mean = 排序后前三 [0.9,0.5,0.3] 的均值
    assert abs(agg_scores(p, "top3mean") - (0.9 + 0.5 + 0.3) / 3) < 1e-12
    # 空输入（视频短于窗口）→ 0 分而不是崩溃
    assert agg_scores([], "max") == 0.0
    # 窗口数少于 3 时 top3mean 退化为 mean（而不是报错）
    assert abs(agg_scores([0.8, 0.2], "top3mean") - 0.5) < 1e-12
    try:
        agg_scores(p, "median")
    except ValueError:
        pass
    else:
        raise AssertionError("未知聚合方式应报错")
    print("  [7] 聚合 max / mean / top3mean 语义正确、空输入不崩 ✓")


if __name__ == "__main__":
    print("评测指标自检（纯 numpy，不需要 torch）\n")
    check_hand_computed()
    check_pathological_auc()
    check_permutation_invariance()
    check_tpr_at_zero_fpr()
    check_curve_shape()
    check_ties_report()
    check_agg()
    print("\n评测指标 OK ✓")
