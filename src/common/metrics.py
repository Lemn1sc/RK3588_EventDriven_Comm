"""评测指标 · ROC/AUC 与视频级聚合（第 1 步修正）。

为什么单独成模块而不是留在 `scripts/train_gru_fall.py`：
- 不依赖 torch / ultralytics，任何环境可裸跑（与 `common/event_vector.py` 同一原则），
  这样 `scripts/check_metrics.py` 能当纯 numpy 回归测试用；
- `experiments/` 的基线框架将来要复用同一把"尺子"，尺子只能有一把。

── 为什么必须处理"并列分数" ──

基线分数是"该视频中触发帧的占比"（k/T 的有理数），70 段里大量取值完全相同。
实测（`data/recordings/detect_fall/gru_features_det.npz`）：

    70 段里只有 4 个不同取值，**67 段并列在 0.0**，只有 3 段非零。

旧实现用 `np.argsort(scores)`（默认 quicksort，**不保证稳定**）排序，并列样本
之间的先后顺序是任意的 → ROC 折线的走向任意 → AUC 不是唯一值。实测把同一份数据
的输入顺序打乱 20 次，旧实现给出的 AUC 在 0.20 ~ 0.84 之间跳动；而按并列取平均秩
的**正确值只有 0.55**（≈ 随机猜测）。

本模块的做法：把"分数相同"的样本归为一个**并列块**，每个块只产生**一个** ROC 工作点
（同分样本任何阈值都分不开，物理上只能算同一个工作点），并用秩公式独立算一遍 AUC
互相校验。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "roc_curve", "auc_from_curve", "auc_rank", "roc_auc",
    "ties_report", "format_ties", "tpr_at_fpr",
    "agg_scores", "AGG_FUNCS", "DEFAULT_AGGS",
]

# numpy 2.0 移除了 np.trapz（改名为 np.trapezoid）。requirements 现 pin 1.26.4
# 尚未受影响，但换环境就炸，所以统一走这个 shim。
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _as_pair(labels, scores):
    """校验并归一成等长 float64 一维数组。"""
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError(f"标签与分数长度不一致：{labels.shape} vs {scores.shape}")
    if labels.size == 0:
        raise ValueError("标签/分数为空，无法计算 ROC")
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("标签必须是 0/1")
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        # 单类样本时 ROC 无定义（这也是 LOVO 单折会被跳过的原因）
        raise ValueError(f"ROC 需要正负样本都存在，当前 正={int(n_pos)} 负={int(n_neg)}")
    return labels, scores


def roc_curve(labels, scores):
    """ROC 阶梯：降序稳定排序 → 并列块只追加一个工作点。

    返回 (fpr, tpr)。首点 (0,0)、末点 (1,1) 自然达成，无需补点。

    与旧实现的唯一区别：一次推进一整块同分样本，而不是一个个推进。对没有并列的
    数据（例如 GRU 输出的浮点概率），结果与旧实现完全相同。
    """
    labels, scores = _as_pair(labels, scores)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())

    order = np.argsort(-scores, kind="mergesort")  # 稳定排序：并列块内顺序确定
    s_sorted = scores[order]
    l_sorted = labels[order]

    fpr = [0.0]
    tpr = [0.0]
    tp = fp = 0
    i, n = 0, len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        block = l_sorted[i : j + 1]
        tp += int((block == 1).sum())
        fp += int((block == 0).sum())
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        i = j + 1
    return np.asarray(fpr), np.asarray(tpr)


def auc_from_curve(fpr, tpr):
    """梯形积分。并列块的斜线段恰好等价于"并列算半个"的标准约定。"""
    return float(_trapz(tpr, fpr))


def auc_rank(labels, scores):
    """秩公式 AUC（并列取平均秩），等价 Mann-Whitney U，不依赖阶梯实现。

        AUC = (Σ rank(正样本) − n₊(n₊+1)/2) / (n₊ · n₋)

    这是文献里的标准定义，也是"正确值"的来源。它与 auc_from_curve 在数学上恒等，
    两条独立实现互相校验（见 roc_auc）。
    """
    labels, scores = _as_pair(labels, scores)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(s_sorted), dtype=np.float64)
    i, n = 0, len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # 平均秩
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_auc(labels, scores, check: bool = True):
    """算 AUC，并断言"阶梯积分"与"秩公式"一致。

    两者数学上恒等，不相等就说明阶梯实现写错了 —— 这是修并列 bug 最可靠的护栏。
    """
    fpr, tpr = roc_curve(labels, scores)
    auc = auc_from_curve(fpr, tpr)
    if check:
        auc_ref = auc_rank(labels, scores)
        if abs(auc - auc_ref) > 1e-9:
            raise AssertionError(
                f"阶梯积分 AUC={auc!r} 与秩公式 AUC={auc_ref!r} 不一致，"
                "说明 roc_curve 的并列处理有误"
            )
    return auc


def ties_report(labels, scores) -> dict:
    """并列结构的体检报告。让"分数大量并列"这类问题以后不再静默出现。"""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    vals, cnts = np.unique(scores, return_counts=True)
    tied = cnts > 1
    return {
        "n": int(scores.size),
        "n_distinct": int(vals.size),
        "max_group": int(cnts.max()) if cnts.size else 0,
        "n_tied_groups": int(tied.sum()),
        "n_tied_samples": int(cnts[tied].sum()) if cnts.size else 0,
    }


def format_ties(rep: dict) -> str:
    """一行体检结论；有并列时显式提示 AUC 必须按秩公式读。"""
    s = (f"{rep['n']} 段 / {rep['n_distinct']} 个不同取值"
         f" / 最大并列组 {rep['max_group']} 段")
    if rep["max_group"] > 1:
        s += "  ⚠ 存在并列，AUC 必须按秩公式（本模块已强制）"
    return s


def tpr_at_fpr(fpr, tpr, target: float) -> float:
    """FPR ≤ target 时的最高检出率（TPR）。

    配合并列分组后的阶梯使用：得到的是"实际可达的最优工作点"，而不是任意路径上的值。
    """
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    ok = fpr <= target + 1e-9
    return float(tpr[ok].max()) if ok.any() else 0.0


# ── 视频级聚合 ──
# GRU 给的是"每个 30 帧窗口的跌倒概率"，要汇总成一个"这段视频是否跌倒"的分数。
# 取 max 是最松的口径，且会引入一个与标签无关的偏置：窗口数 ∝ 视频长度，
# 本数据集窗口数在 3~38 之间（相差 12 倍），长视频"碰运气"的机会多得多。
# 因此同时报 mean / top3mean，让结论对聚合方式的敏感性可见。
def agg_scores(probs, how: str) -> float:
    """窗口概率序列 → 视频级分数。how ∈ {max, mean, top3mean}。"""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return 0.0
    if how == "max":
        return float(p.max())
    if how == "mean":
        return float(p.mean())
    if how == "top3mean":
        return float(np.sort(p)[::-1][:3].mean())
    raise ValueError(f"未知聚合方式：{how}（可选 {sorted(AGG_FUNCS)}）")


AGG_FUNCS = {"max", "mean", "top3mean"}
DEFAULT_AGGS = ("max", "mean", "top3mean")
