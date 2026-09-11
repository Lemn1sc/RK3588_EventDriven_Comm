"""第 2 步 · 缓存版本机制自检（不依赖 torch/ultralytics，纯 numpy）。

用法：python scripts/check_cache.py   看到 "缓存版本机制 OK ✓" 即通过。

验的是**三条失败路径都必须判失败**（而不是"兼容读取"）：
  1. 文件不存在
  2. 文件存在但没有版本信息（引入本机制之前的旧缓存）
  3. 指纹与当前代码不符
外加两条正向：指纹稳定、形状不符也要拦。
"""
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common.cache import META_KEY, load_npz, read_meta, save_npz, spec_sha1  # noqa: E402

SPEC = {"featset": "det.v1", "names": ["has_event", "aspect_norm"], "dim": 2,
        "aspect_clip": [0.3, 4.0]}  # 常数也必须进指纹


def expect_fail(fn, keyword: str, label: str):
    """断言某次加载必须失败，且报错里含指定关键词（保证提示是可操作的）。"""
    try:
        fn()
    except SystemExit as e:
        msg = str(e)
        assert keyword in msg, f"{label}: 报错信息里应含「{keyword}」，实际是：\n{msg}"
        return msg
    raise AssertionError(f"{label}: 本应判失败，却通过了 —— 这正是静默复用旧缓存的入口")


def check_fingerprint_stability():
    """指纹只取决于内容：键顺序无关，改常数必变。"""
    a = dict(SPEC)
    b = {"dim": 2, "aspect_clip": [0.3, 4.0], "names": ["has_event", "aspect_norm"],
         "featset": "det.v1"}  # 同样的内容、不同的书写顺序
    assert spec_sha1(a) == spec_sha1(b), "键顺序不该影响指纹"
    assert len(spec_sha1(a)) == 8

    # ★关键★ 只改一个常数（维度不变），指纹必须变
    c = dict(SPEC, aspect_clip=[0.5, 3.0])
    assert spec_sha1(c) != spec_sha1(a), "改了 clip 常数却没换指纹 → 语义漂移拦不住"

    # 只改特征名顺序（数值会变）也必须换指纹
    d = dict(SPEC, names=["aspect_norm", "has_event"])
    assert spec_sha1(d) != spec_sha1(a)
    print("  [1] 指纹只取决于内容：键顺序无关、改常数/改顺序必变 ✓")


def check_roundtrip(tmp: Path):
    """正常路径：存了能读回来，且数组逐值一致。"""
    path = tmp / "feats_det.v1.npz"
    feats = {"fall/fall-01.mp4": np.random.RandomState(0).rand(10, 2).astype(np.float32)}
    save_npz(path, SPEC, feats, note="自检")
    npz, meta = load_npz(path, SPEC, item_keys=list(feats), expect_dim=2)
    assert meta["spec_sha1"] == spec_sha1(SPEC)
    assert np.array_equal(npz["fall/fall-01.mp4"], feats["fall/fall-01.mp4"])
    assert read_meta(path)["note"] == "自检"
    print("  [2] 正常路径：存入→读回，数组逐值一致 ✓")


def check_missing(tmp: Path):
    msg = expect_fail(lambda: load_npz(tmp / "不存在.npz", SPEC),
                      "不存在", "文件不存在")
    assert "--recompute" in msg, "提示里应告诉用户怎么修"
    print("  [3] 缓存不存在 → 判失败，并提示 --recompute ✓")


def check_no_meta(tmp: Path):
    """引入本机制之前的旧缓存：必须判失败，不能放行。"""
    path = tmp / "old_cache.npz"
    np.savez(path, **{"fall/fall-01.mp4": np.zeros((10, 2), dtype=np.float32)})
    assert read_meta(path) is None, "这个文件本来就不该有元数据"
    msg = expect_fail(lambda: load_npz(path, SPEC), "没有版本信息", "无元数据的旧缓存")
    assert "改名留档" in msg, "提示里应告诉用户怎么处置旧缓存"
    print("  [4] 旧缓存（无版本信息）→ 判失败，不做兼容读取 ✓")


def check_wrong_spec(tmp: Path):
    """指纹不符：必须判失败，并指出哪个字段变了。"""
    path = tmp / "feats_det.v1.npz"
    msg = expect_fail(lambda: load_npz(path, dict(SPEC, aspect_clip=[0.5, 3.0])),
                      "不匹配", "指纹不符")
    assert "aspect_clip" in msg, f"报错应指出差异字段，实际：\n{msg}"
    print("  [5] 指纹不符（只改了常数）→ 判失败，且指出差异字段 ✓")


def check_shape_guard(tmp: Path):
    """第二道保险：即使指纹一致，形状不对也要拦（防手工拼装的文件）。"""
    path = tmp / "shape.npz"
    # 故意用"正确的指纹"存"错误维度"的数据
    save_npz(path, SPEC, {"v": np.zeros((10, 5), dtype=np.float32)})
    expect_fail(lambda: load_npz(path, SPEC, item_keys=["v"], expect_dim=2),
                "形状", "维度不符")
    print("  [6] 指纹一致但形状不符 → 第二道保险仍然拦下 ✓")


def check_versions_coexist(tmp: Path):
    """两个版本并存、互不覆盖 —— 这是"结果可向后对照"的物理保证。"""
    p1, p2 = tmp / "feats_det.v1.npz", tmp / "feats_det.v2.npz"
    spec2 = dict(SPEC, dim=8, names=SPEC["names"] + ["cy", "size"], aspect_clip=[0.3, 4.0])
    save_npz(p1, SPEC, {"v": np.zeros((10, 2), dtype=np.float32)})
    save_npz(p2, spec2, {"v": np.zeros((10, 8), dtype=np.float32)})
    assert p1.exists() and p2.exists()
    assert read_meta(p1)["spec_sha1"] != read_meta(p2)["spec_sha1"]
    # v2 的代码读 v1 的缓存必须失败（而不是悄悄用）
    expect_fail(lambda: load_npz(p1, spec2, item_keys=["v"], expect_dim=8),
                "不匹配", "v2 代码读 v1 缓存")
    print("  [7] 新旧版本并存互不覆盖；拿新代码读旧缓存会当场报错 ✓")


if __name__ == "__main__":
    print("缓存版本机制自检（纯 numpy）\n")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_fingerprint_stability()
        check_roundtrip(tmp)
        check_missing(tmp)
        check_no_meta(tmp)
        check_wrong_spec(tmp)
        check_shape_guard(tmp)
        check_versions_coexist(tmp)
    print("\n缓存版本机制 OK ✓")
