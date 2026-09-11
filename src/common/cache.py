"""缓存/产物的版本机制（第 2 步）。

── 为什么需要 ──

原来 `load_or_extract()` 的判据只有一条：**文件存在就读**。这有两种静默失效：

1. **维度变了**：det 特征从 4 维改成 8 维后，若新代码漏改某处，旧的 `(T,4)` 会被
   照样读进来，喂给 `in_dim=4` 的模型 —— 跑出来的"结果"完全正常，只是毫无意义。
2. **维度没变但语义变了**：例如把 `clip(aspect, 0.3, 4.0) / 4.0` 改成
   `clip(aspect, 0.5, 3.0) / 3.0`，仍然是 1 维。任何"看维度"的检查都拦不住。
   而这恰恰是最容易被悄悄改坏的一类。
3. **读到别人的缓存**：两个不同特征集的缓存若重名，直接读错文件。

静默读到旧缓存的后果是「跑出一份看起来完全正常、但毫无意义的实验结果」——
科研代码里最难查的一类 bug。

── 三层防护 ──

1. **文件名带身份**：`feats_det.v1.npz` 与 `feats_det.v2.npz` 并存、互不覆盖。
   这是"结果可向后对照"的物理保证：旧特征集的缓存和权重都还在，随时能重跑出旧数字。
2. **内容指纹**：npz 内固定写 `__meta__` 与 `__spec_sha1__`。指纹 = 把"一切会影响
   数值的东西"写成 canonical JSON 后取 sha1 前 8 位。**关键在于 hash 的输入必须包含
   常数**（如 clip 的上下界），否则上面第 2 类失效拦不住。
3. **加载即严格校验**：缓存缺失 / 无版本信息 / 指纹不符，三种情况一律**报错停下**，
   并说明怎么修。刻意不做"兼容读取" —— 旧缓存维度恰好也是 4，若允许无标记缓存通过，
   就等于给"新旧混用"开后门。

本模块不依赖 torch / ultralytics（与 `event_vector.py`、`metrics.py` 同一原则），
这样 `scripts/check_cache.py` 能在任意环境裸跑。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["CACHE_FORMAT", "SPEC_KEY", "META_KEY", "spec_sha1", "save_npz",
           "load_npz", "read_meta", "spec_diff"]

CACHE_FORMAT = 1
SPEC_KEY = "__spec_sha1__"
META_KEY = "__meta__"


def spec_sha1(spec: dict) -> str:
    """特征/缓存规格 → 8 位十六进制指纹。

    canonical JSON（键排序 + 紧凑分隔符）保证字典书写顺序不影响结果；
    指纹的输入必须把**所有会影响数值的常数**都包含进去。
    """
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def spec_diff(cached: dict, expect: dict) -> dict:
    """逐字段差异，供报错时直接指出「哪里变了」。"""
    diff = {}
    for k in sorted(set(cached) | set(expect)):
        a, b = cached.get(k, "<缺失>"), expect.get(k, "<缺失>")
        if a != b:
            diff[k] = {"缓存": a, "当前代码": b}
    return diff


def save_npz(path, spec: dict, arrays: dict, **extra_meta) -> dict:
    """保存带版本信息的 npz。`arrays` 是 {key: ndarray}（沿用原有键名约定）。"""
    path = Path(path)
    meta = {"format": CACHE_FORMAT, "spec": spec, "spec_sha1": spec_sha1(spec)}
    meta.update(extra_meta)
    payload = dict(arrays)
    payload[SPEC_KEY] = np.array([meta["spec_sha1"]])
    payload[META_KEY] = np.array([json.dumps(meta, ensure_ascii=False, sort_keys=True)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return meta


def read_meta(path) -> dict | None:
    """读版本信息；没有则返回 None（调用方负责决定是否判失败）。"""
    npz = np.load(path, allow_pickle=True)
    if META_KEY not in npz.files:
        return None
    return json.loads(str(npz[META_KEY][0]))


def load_npz(path, expect_spec: dict, item_keys=None, expect_dim: int | None = None,
             what: str = "缓存"):
    """严格加载带版本信息的 npz。

    三种失败都抛 SystemExit 并给出可执行的修复办法（不做兼容读取）：
    - 文件不存在          → 提示加 --recompute
    - 文件存在但没有版本信息 → 提示改名留档后重新生成（这是引入版本机制前的旧缓存）
    - 指纹与当前代码不符    → 打印逐字段差异，提示重新生成

    参数
        item_keys:  需要逐个检查形状的键（如 70 段视频名）；None 则跳过形状检查
        expect_dim: item_keys 中每个数组应有的第二维长度（特征维度）
    返回 (npz, meta)
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{what}不存在：{path}\n  → 加 --recompute 重新生成，或检查路径")

    npz = np.load(path, allow_pickle=True)
    if META_KEY not in npz.files:
        raise SystemExit(
            f"{what}没有版本信息：{path}\n"
            f"  → 这是引入版本机制（第 2 步）之前的旧缓存。请不要手工补标记，那样等于\n"
            f"    给「新旧混用」开后门；正确做法是把它改名留档（如 *.v1.npz）后\n"
            f"    用 --recompute 重新生成，两者可逐值对照。"
        )

    meta = json.loads(str(npz[META_KEY][0]))
    want = spec_sha1(expect_spec)
    if meta.get("spec_sha1") != want:
        diff = spec_diff(meta.get("spec", {}), expect_spec)
        detail = "\n".join(f"      {k}: {v}" for k, v in diff.items()) or "      （规格相同）"
        raise SystemExit(
            f"{what}与当前代码不匹配：{path}\n"
            f"  缓存指纹 {meta.get('spec_sha1')} / 当前代码 {want}\n"
            f"  差异字段：\n{detail}\n"
            f"  → 确认这是有意的改动后，用 --recompute 重新生成；\n"
            f"    旧缓存请改名留档（保留向后对照的能力）。"
        )

    if item_keys is not None and expect_dim is not None:
        for k in item_keys:
            if k not in npz.files:
                raise SystemExit(f"{what}缺少条目 {k}：{path}")
            arr = npz[k]
            if arr.ndim != 2 or arr.shape[1] != expect_dim:
                raise SystemExit(
                    f"{what}里 {k} 的形状是 {arr.shape}，期望 (帧数, {expect_dim})：{path}\n"
                    f"  → 第二道保险：即使指纹碰巧一致，形状不对也立刻停下。"
                )
    return npz, meta
