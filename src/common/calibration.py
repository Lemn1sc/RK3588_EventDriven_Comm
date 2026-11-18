"""置信度校准 · 把“模型分数”变成“事件为真的概率”。

── 为什么事件层需要它 ──

控制端要算“这个事件值不值得占带宽”，需要的是：

    P(事件为真 | 证据) × 紧急度 / 传输代价

而不是模型原始分数。YOLO 的 confidence 是分类器分数，不是概率：同一个 0.4 在
俯视躺姿和正视角站立下含义完全不同（见 docs/08 §11.6）。事件层把证据送到控制端
之前，至少应该给一个**可解释的可靠性**。

本模块只提供两把最常用的尺子，不依赖 sklearn / scipy：

- `brier_score` / `reliability_curve` / `expected_calibration_error`：评估校准质量；
- `IsotonicCalibrator`：PAVA 保序回归，把分数映射成单调概率。

为什么是保序回归而不是 logistic：样本量小、分数与真实概率的关系未必是 S 形；
保序回归只假设“分数越高、真实概率不下降”，更稳，也不引入需要调的超参。

 ── 一个使用上的提醒 ──

校准必须**按场景/机位**做，或者把场景作为特征。把 URFD 平视的 0.4 和 MCF 俯视的
0.4 混在一起校准，得到的只是一条“两个分布的平均曲线”，对任何单一机位都不准。
详见 docs/09 的“可靠性校准”一节。
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "brier_score", "reliability_curve", "expected_calibration_error",
    "IsotonicCalibrator", "isotonic_fit",
]


def _as_pair(y_true, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.shape != s.shape:
        raise ValueError(f"标签与分数长度不一致：{y.shape} vs {s.shape}")
    if y.size == 0:
        raise ValueError("标签/分数为空，无法校准")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("标签必须是 0/1")
    if not np.all(np.isfinite(s)):
        raise ValueError("分数里有 NaN/Inf")
    s = np.clip(s, 0.0, 1.0)
    return y, s


def brier_score(y_true, p) -> float:
    """Brier score = mean((p - y)^2)，越小越好；0 为完美。"""
    y, s = _as_pair(y_true, p)
    return float(np.mean((s - y) ** 2))


def reliability_curve(y_true, p, n_bins: int = 10) -> dict[str, Any]:
    """可靠性曲线 + Brier + ECE。

    返回的每个 bin 含 `count / mean_pred / frac_pos`；空 bin 的 mean/frac 为 None，
    这样 JSON 落盘时不会出现 NaN（NaN 不是合法 JSON）。
    """
    if n_bins < 2:
        raise ValueError("n_bins 至少为 2")
    y, s = _as_pair(y_true, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # p=1.0 要落进最后一个 bin，不能越界；p=0.0 落进第一个。
    idx = np.clip(np.floor(s * n_bins).astype(np.int64), 0, n_bins - 1)

    counts, mean_pred, frac_pos = [], [], []
    ece = 0.0
    n = float(y.size)
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        counts.append(c)
        if c == 0:
            mean_pred.append(None)
            frac_pos.append(None)
        else:
            mp = float(s[m].mean())
            fp = float(y[m].mean())
            mean_pred.append(round(mp, 6))
            frac_pos.append(round(fp, 6))
            ece += (c / n) * abs(fp - mp)

    return {
        "n": int(y.size),
        "n_bins": int(n_bins),
        "bin_edges": [round(float(e), 6) for e in edges],
        "counts": counts,
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "brier": round(brier_score(y, s), 6),
        "ece": round(float(ece), 6),
    }


def expected_calibration_error(y_true, p, n_bins: int = 10) -> float:
    """分箱加权的 |真实频率 − 平均预测|。"""
    return float(reliability_curve(y_true, p, n_bins=n_bins)["ece"])


class IsotonicCalibrator:
    """保序回归（PAVA）校准器。

    `fit(x, y)` 学一条单调不减的映射；`transform(x)` 用线性插值，两端保持常数。
    分数越高，校准后的概率不会下降 —— 这是保序回归的唯一假设。
    """

    def __init__(self) -> None:
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, x, y) -> "IsotonicCalibrator":
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.shape != y.shape:
            raise ValueError(f"x/y 长度不一致：{x.shape} vs {y.shape}")
        if x.size == 0:
            raise ValueError("x/y 为空")
        if not np.all(np.isfinite(x)):
            raise ValueError("x 里有 NaN/Inf")
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("y 必须是 0/1")

        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]

        # PAVA：每个块记录 [权重, 权重×均值, x_min, x_max]；相邻块均值逆序就合并。
        blocks: list[list[float]] = []
        for xi, yi in zip(xs, ys):
            blocks.append([1.0, float(yi), float(xi), float(xi)])
            while len(blocks) >= 2:
                prev, cur = blocks[-2], blocks[-1]
                if prev[1] / prev[0] <= cur[1] / cur[0] + 1e-15:
                    break
                b = blocks.pop()
                a = blocks.pop()
                blocks.append([a[0] + b[0], a[1] + b[1], a[2], b[3]])

        knots_x = np.array([(b[2] + b[3]) / 2.0 for b in blocks], dtype=np.float64)
        knots_y = np.array([b[1] / b[0] for b in blocks], dtype=np.float64)
        knots_y = np.clip(knots_y, 0.0, 1.0)

        # 相同 x 的块合并（平均值），保证 np.interp 的 x 严格递增。
        ux, inv = np.unique(knots_x, return_inverse=True)
        if ux.size != knots_x.size:
            sums = np.zeros(ux.size)
            counts = np.zeros(ux.size)
            for i, b in enumerate(inv):
                sums[b] += knots_y[i]
                counts[b] += 1
            knots_x, knots_y = ux, sums / np.maximum(counts, 1)

        self.x_, self.y_ = knots_x, knots_y
        return self

    def transform(self, x) -> np.ndarray:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("IsotonicCalibrator 还没 fit")
        x = np.asarray(x, dtype=np.float64)
        out = np.interp(x, self.x_, self.y_,
                        left=float(self.y_[0]), right=float(self.y_[-1]))
        return np.clip(out, 0.0, 1.0)

    def fit_transform(self, x, y) -> np.ndarray:
        return self.fit(x, y).transform(x)


def isotonic_fit(x, y) -> IsotonicCalibrator:
    """便捷入口：返回已 fit 的 `IsotonicCalibrator`。"""
    return IsotonicCalibrator().fit(x, y)
