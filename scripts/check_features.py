"""第 3 步 · 时序特征定点自检（不依赖 torch/ultralytics，纯 numpy）。

用法：python scripts/check_features.py   看到 "时序特征 OK ✓" 即通过。

这里验的不是"指标变好了"，而是**特征有没有它声称的物理含义**。
用合成帧表把每条性质钉死：

  * `cy_drop` 只在"突变"时亮起 —— 跌倒是瞬态，躺地不动是稳态，两者必须分开
  * 人从画面消失很久后重现，不能凭空产生一个假的突降尖峰（断裂保护）
  * `size_log_ratio` 对称（缩小一半 = -0.5，放大一倍 = +0.5）
  * v2 的前 4 维与 v1 逐位相同，pose.v2 的前 8 维与 det.v2 逐位相同
    （否则 det vs pose 的对比就不是公平消融）
"""
import io
import sys
from pathlib import Path

import numpy as np

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import gru_features as g  # noqa: E402


def table(rows) -> np.ndarray:
    """rows: [(has_event, (x1,y1,x2,y2) 或 None, {附加列})] → (T, FRAME_DIM)"""
    t = np.full((len(rows), g.FRAME_DIM), np.nan)
    for i, item in enumerate(rows):
        has, bbox = item[0], item[1]
        extra = item[2] if len(item) > 2 else {}
        t[i, g.C["index"]] = i
        t[i, g.C["has_event"]] = 1.0 if has else 0.0
        t[i, g.C["confidence"]] = 0.9 if has else 0.0
        t[i, g.C["severity"]] = 0.0
        if has and bbox is not None:
            x1, y1, x2, y2 = bbox
            t[i, g.C["bbox_x1"]] = x1
            t[i, g.C["bbox_y1"]] = y1
            t[i, g.C["bbox_x2"]] = x2
            t[i, g.C["bbox_y2"]] = y2
            t[i, g.C["aspect_ratio"]] = (y2 - y1) / max(x2 - x1, 1e-6)
        for col, val in extra.items():
            t[i, g.C[col]] = val
    return t


def box_with_cy(cy: float, w: float = 0.1, h: float = 0.2):
    """构造一个高宽比 2.0（直立）的框，其中心 y 等于 cy。"""
    return (0.45, max(cy - h / 2, 0.0), 0.45 + w, max(cy - h / 2, 0.0) + h)


def idx(name: str) -> int:
    return g.FEATSETS["det.v2"].names.index(name)


def col(f: np.ndarray, featset: str, name: str, t: int) -> float:
    return float(f[t, g.FEATSETS[featset].names.index(name)])


def check_baseline_warmup():
    """开头历史不足 MIN_BASE 帧时，相对量应为 0（中性），不能乱给值。"""
    t = table([(True, box_with_cy(0.4)) for _ in range(20)])
    f = g.build_det_v2(t)
    warm = f[: g.MIN_BASE - 1, idx("cy_drop")]
    assert np.allclose(warm, 0.0), warm
    # 第 MIN_BASE 帧起基线生效，但 cy 没变过 → 突降比仍是 0
    assert abs(col(f, "det.v2", "cy_drop", g.MIN_BASE)) < 1e-9
    print(f"  [1] 开头 {g.MIN_BASE - 1} 帧相对量为 0（历史不足即为中性）✓")


def check_cy_drop_is_transient():
    """★核心性质★ 突变时亮起，稳定后回落归零 —— 这正是"跌倒 vs 躺地不动"的分界。

    前 30 帧 cy=0.4 恒定，第 31 帧骤降到 0.15，之后一直保持 0.15。
    """
    rows = [(True, box_with_cy(0.4))] * 30 + [(True, box_with_cy(0.15))] * 41
    f = g.build_det_v2(table(rows))

    at_drop = col(f, "det.v2", "cy_drop", 30)
    assert abs(at_drop - (0.4 - 0.15) / 0.4) < 1e-6, f"突变帧应给出 0.625，实得 {at_drop}"

    # 躺稳之后（再过 40 帧），基线漂到新位置 → 突降比回落归零
    settled = col(f, "det.v2", "cy_drop", 70)
    assert abs(settled) < 1e-6, f"稳态应回落归零，实得 {settled}"

    # 稳定站立段全程为 0
    assert np.allclose(f[:30, idx("cy_drop")], 0.0)
    print(f"  [2] cy_drop 突变帧={at_drop:.3f}，躺稳 40 帧后回落到 {settled:.3f} ✓")


def check_gap_protection():
    """人消失很久后重现 → 视为新轨迹，不得凭空产生假的突降尖峰。

    边界按**索引间隔**钉死（不是"中间空了几个帧"）：
    间隔恰好 = MAX_GAP 时基线保留，> MAX_GAP 才重置。
    """
    def build(n_empty: int):
        rows = [(True, box_with_cy(0.4))] * 30 + [(False, None)] * n_empty
        rows += [(True, box_with_cy(0.15))]
        return table(rows), 30 + n_empty      # 重现帧的索引

    # 中间空 29 帧 → 索引间隔 30 == MAX_GAP → 基线保留 → 读出突降
    t, i = build(g.MAX_GAP - 1)
    assert i - 29 == g.MAX_GAP
    v = col(g.build_det_v2(t), "det.v2", "cy_drop", i)
    assert abs(v - 0.625) < 1e-6, f"间隔 {g.MAX_GAP} 不该重置基线，实得 {v}"

    # 中间空 30 帧 → 索引间隔 31 > MAX_GAP → 基线清空 → 中性 0
    t, i = build(g.MAX_GAP)
    assert i - 29 == g.MAX_GAP + 1
    f = g.build_det_v2(t)
    v = col(f, "det.v2", "cy_drop", i)
    assert v == 0.0, f"间隔 {g.MAX_GAP + 1} 应重置基线，实得 {v}"
    # 重现后的头几帧都应是 0（历史还没攒够）
    assert np.allclose(f[i:i + g.MIN_BASE, idx("cy_drop")], 0.0)
    print(f"  [3] 断裂保护：间隔 {g.MAX_GAP} 保留基线、间隔 {g.MAX_GAP + 1} 重置 ✓")


def check_size_features():
    """size = sqrt(面积)（线性尺度），size_log_ratio 对称。"""
    w, h = 0.2, 0.3
    t = table([(True, (0.4, 0.3, 0.4 + w, 0.3 + h))] * 30)
    f = g.build_det_v2a(t)
    assert abs(col(f, "det.v2a", "size", 0) - np.sqrt(w * h)) < 1e-6

    # 尺度突变：先稳定在大框，然后面积变为 1/4（线性尺度减半）
    big = (0.4, 0.3, 0.4 + w, 0.3 + h)
    half = (0.4, 0.3, 0.4 + w / 2, 0.3 + h / 2)   # w,h 各减半 → 面积 1/4 → sqrt 减半
    rows = [(True, big)] * 30 + [(True, half)] * 5
    f = g.build_det_v2(table(rows))
    v = col(f, "det.v2", "size_log_ratio", 30)
    assert abs(v - (-0.5)) < 1e-6, f"面积 1/4 → log2 比值应为 -0.5，实得 {v}"

    # 对称性：面积 ×4（线性 ±2 倍）→ +0.5
    rows = [(True, half)] * 30 + [(True, big)] * 5
    f = g.build_det_v2(table(rows))
    v = col(f, "det.v2", "size_log_ratio", 30)
    assert abs(v - 0.5) < 1e-6, f"面积 ×4 → 应为 +0.5，实得 {v}"
    print("  [4] size=sqrt(面积)；size_log_ratio 对称（-0.5 / +0.5）✓")


def check_no_event_frames():
    """无人帧：全部维度为 0，且不影响基线。"""
    rows = [(True, box_with_cy(0.4))] * 30 + [(False, None)] * 10 + [(True, box_with_cy(0.4))]
    f = g.build_det_v2(table(rows))
    assert np.allclose(f[30:40], 0.0), "无人帧应当全 0"
    # 无人帧不更新基线：重现后 cy 没变过 → 突降比仍为 0（若无人帧污染了基线，
    # 这里会因为基线被 0 拉低而给出大值）
    assert abs(col(f, "det.v2", "cy_drop", 40)) < 1e-6
    print("  [5] 无人帧全 0 且不污染基线 ✓")


def check_value_ranges():
    """所有维度都必须落在合理值域内（相对量在 [-1,1]）。"""
    rng = np.random.RandomState(0)
    rows = []
    for _ in range(200):
        has = rng.rand() > 0.2
        if has:
            cy = rng.rand() * 0.8 + 0.1
            b = box_with_cy(cy, rng.rand() * 0.4 + 0.02, rng.rand() * 0.4 + 0.02)
        else:
            b = None
        rows.append((has, b))
    for name in ("det.v1", "det.v2a", "det.v2"):
        f = g.build_feats(name, {"x": table(rows)})["x"]
        assert np.all(np.isfinite(f)), f"{name} 出现非有限值"
        assert f.min() >= -1.0 - 1e-6 and f.max() <= 1.0 + 1e-6, (name, f.min(), f.max())
    print("  [6] 值域检查：全部维度有限且落在 [-1, 1] ✓")


def check_extension_consistency():
    """v2 系列必须严格"包含"它的前身，否则消融实验不是公平对照。"""
    rng = np.random.RandomState(1)
    rows = []
    for _ in range(120):
        has = rng.rand() > 0.25
        rows.append((has, box_with_cy(rng.rand() * 0.7 + 0.15) if has else None,
                     {"torso_angle": rng.rand() * 90, "hip_ankle_ratio": rng.rand()}))
    t = table(rows)
    v1 = g.build_det_v1(t)
    v2a = g.build_det_v2a(t)
    v2 = g.build_det_v2(t)
    p1 = g.build_pose_v1(t)
    p2 = g.build_pose_v2(t)

    assert np.array_equal(v2[:, :4], v1), "det.v2 前 4 维必须 == det.v1"
    assert np.array_equal(v2a[:, :4], v1), "det.v2a 前 4 维必须 == det.v1"
    assert np.array_equal(v2[:, :6], v2a), "det.v2 前 6 维必须 == det.v2a"
    assert np.array_equal(p2[:, :8], v2), "pose.v2 前 8 维必须 == det.v2"
    assert np.array_equal(p2[:, 8:], p1[:, 4:]), "pose.v2 的 2 个姿态量必须 == pose.v1 的对应维"
    print("  [7] 扩展关系：v2a⊃v1、v2⊃v2a、pose.v2⊃det.v2（消融是公平的）✓")


def check_registry_sane():
    """注册表自身的自洽性：维度、名字数、特征集/帧表命名对应。"""
    for name, fs in g.FEATSETS.items():
        assert fs.dim == len(fs.names), name
        assert fs.spec["dim"] == fs.dim, f"{name}: spec 里的维度与注册表不一致"
        assert fs.spec["featset"] == name, name
        # 注册表的 frames 是「需要哪张帧表」(det/pose)，spec 里记的是帧表的身份名
        assert fs.spec["frames"] == g.FRAME_SPEC[fs.frames]["frameset"], name
        assert fs.mode in ("det", "pose") and fs.name.split(".")[0] == fs.mode, name
        assert g.FEAT_CACHE[name].name == f"feats_{name}.npz", name
    # 两个 det 与 pose 的对照必须共享同一套基础维度
    assert g.FEATSETS["pose.v2"].dim == g.FEATSETS["det.v2"].dim + 2
    print(f"  [8] 注册表自洽：{len(g.FEATSETS)} 个特征集，命名/维度/指纹一致 ✓")


if __name__ == "__main__":
    print("时序特征定点自检（纯 numpy，不需要 torch）\n")
    check_baseline_warmup()
    check_cy_drop_is_transient()
    check_gap_protection()
    check_size_features()
    check_no_event_frames()
    check_value_ranges()
    check_extension_consistency()
    check_registry_sane()
    print("\n时序特征 OK ✓")
