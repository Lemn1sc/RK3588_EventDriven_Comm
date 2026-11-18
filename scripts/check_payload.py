#!/usr/bin/env python
"""事件载荷自检（需要 cv2，但不需要模型 / 数据）。

覆盖：
- METADATA 是合法 JSON，且 NaN 被清成 null；
- ROI 是 JPEG、尺寸不超过 max_side、坏框回退整帧不抛异常；
- FULL 是 JPEG、解码后尺寸与原图一致；
- 三种载荷的字节数关系（metadata 最小）；
- 灰度帧能编码；
- CLIP 在当前 OpenCV 有 mp4v 时能出 MP4，没有则明确跳过。

用法：
    python scripts/check_payload.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from edge_node.payload import (  # noqa: E402
    PayloadMode, encode_clip, encode_full, encode_metadata, encode_roi, payload_sizes,
)

FAILED: list[str] = []
SKIPPED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"✓ {name}")
    else:
        FAILED.append(name)
        print(f"✗ {name}" + (f"\n    {detail}" if detail else ""))


def expect_raises(name: str, fn) -> None:
    try:
        fn()
    except (ValueError, RuntimeError):
        check(name, True)
    except Exception as exc:
        check(name, False, f"抛了非预期异常：{type(exc).__name__}: {exc}")
    else:
        check(name, False, "应当抛异常但没有")


def make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = (xx / max(w - 1, 1) * 255).astype(np.uint8)
    img[..., 1] = (yy / max(h - 1, 1) * 255).astype(np.uint8)
    img[..., 2] = 128
    noise = np.random.default_rng(0).integers(0, 40, (h, w, 3), dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def is_jpeg(b: bytes) -> bool:
    return len(b) > 4 and b[:2] == b"\xff\xd8"


def test_metadata() -> None:
    ev = {"event_id": 7, "kind": "SUSPECTED_FALL", "bbox": [0.2, 0.3, 0.5, 0.8],
          "nan_field": float("nan")}
    b = encode_metadata(ev)
    d = json.loads(b.decode("utf-8"))
    check("METADATA 是合法 JSON", d["event_id"] == 7)
    check("METADATA 清掉 NaN", d["nan_field"] is None)
    check("METADATA 很小", len(b) < 512, f"{len(b)} B")


def test_roi() -> None:
    import cv2

    frame = make_frame()
    bbox = (0.30, 0.30, 0.60, 0.80)
    b = encode_roi(frame, bbox, max_side=224, quality=70, padding=0.1)
    check("ROI 是 JPEG", is_jpeg(b), b[:8].hex())
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    check("ROI 能解码", img is not None and img.size > 0)
    check("ROI 最长边不超过 max_side", max(img.shape[:2]) <= 224, str(img.shape))
    check("ROI 不是整帧", img.shape[0] < frame.shape[0] or img.shape[1] < frame.shape[1],
          f"{img.shape} vs {frame.shape}")

    # 坏框：零面积 / 反向 / 越界，都应该回退或裁剪，而不是抛异常
    for name, bad in [("零面积", (0.5, 0.5, 0.5, 0.5)),
                      ("None", None),
                      ("越界", (-0.5, -0.5, 1.5, 1.5))]:
        try:
            bb = encode_roi(frame, bad)
            check(f"ROI 坏框（{name}）不抛异常且非空", len(bb) > 0, f"{len(bb)} B")
        except Exception as exc:
            check(f"ROI 坏框（{name}）不抛异常且非空", False,
                  f"{type(exc).__name__}: {exc}")


def test_full_and_gray() -> None:
    import cv2

    frame = make_frame()
    b = encode_full(frame, quality=80)
    check("FULL 是 JPEG", is_jpeg(b))
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    check("FULL 尺寸与原图一致", img.shape[:2] == frame.shape[:2],
          f"{img.shape[:2]} vs {frame.shape[:2]}")

    gray = frame[..., 1]
    gb = encode_full(gray)
    gi = cv2.imdecode(np.frombuffer(gb, np.uint8), cv2.IMREAD_COLOR)
    check("灰度帧编码为 3 通道", gi is not None and gi.ndim == 3 and gi.shape[2] == 3)

    expect_raises("空帧应报错", lambda: encode_full(np.zeros((0, 0, 3), np.uint8)))


def test_sizes() -> None:
    frame = make_frame()
    ev = {"event_id": 1, "kind": "SUSPECTED_FALL", "bbox": [0.3, 0.3, 0.6, 0.8]}
    sizes = payload_sizes(frame, ev)
    print(f"    实测大小：{sizes}")
    check("三种载荷都非空", all(v > 0 for v in sizes.values()), str(sizes))
    check("METADATA < ROI", sizes["metadata"] < sizes["roi"], str(sizes))
    check("METADATA < FULL", sizes["metadata"] < sizes["full"], str(sizes))
    check("ROI < FULL", sizes["roi"] < sizes["full"], str(sizes))


def test_clip() -> None:
    frames = [make_frame(64, 64) for _ in range(6)]
    try:
        b = encode_clip(frames, fps=10.0)
    except RuntimeError as exc:
        SKIPPED.append("CLIP（当前 OpenCV 无 mp4v）")
        print(f"· 跳过 CLIP：{exc}")
        return
    check("CLIP 非空", len(b) > 0, f"{len(b)} B")
    check("CLIP 是 MP4（ftyp）", len(b) > 12 and b[4:8] == b"ftyp", b[:12].hex())


def test_mode_ids() -> None:
    check("PayloadMode 数值稳定",
          [int(m) for m in (PayloadMode.METADATA, PayloadMode.ROI,
                            PayloadMode.FULL, PayloadMode.CLIP)] == [0, 1, 2, 3])


def main() -> int:
    for fn in (test_metadata, test_roi, test_full_and_gray, test_sizes,
               test_clip, test_mode_ids):
        print(f"\n── {fn.__name__} ──")
        try:
            fn()
        except Exception as exc:
            FAILED.append(fn.__name__)
            print(f"✗ {fn.__name__} 抛异常：{type(exc).__name__}: {exc}")

    print()
    if SKIPPED:
        print(f"跳过：{SKIPPED}")
    if FAILED:
        print(f"失败 {len(FAILED)} 项：{FAILED}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
