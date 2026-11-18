"""事件载荷 · 同一个事件按“信息量 / 字节数”提供多种版本。

── 为什么事件层要管载荷 ──

控制端在信道差 / 拥塞时，需要回答一个具体问题：

    “这个高紧急度事件，在 deadline 之前，最小能发多少信息？”

如果事件层只给一个 bbox 和置信度，控制端没法选；如果只给整帧，又太贵。
所以事件层提供四档：

    METADATA  几十字节  事件 id / 类型 / 置信度 / bbox / 建议紧急度
    ROI       几 KB     框裁剪 + 缩放后的 JPEG（远端确认用）
    FULL      几十 KB   整帧 JPEG（需要看上下文时）
    CLIP      几百 KB   1 秒片段（最后手段 / 演示用）

控制端根据当前 goodput、队列和剩余 deadline 选一个**能按时送达**的版本；
事件层只负责“能提供什么”和“各是多少字节”，不替控制端做决策。

── 依赖 ──

cv2 只在函数内部 import：事件形成器是纯 Python，不能让 `import edge_node` 连带
拉起 OpenCV；纯 Python 自检（check_event_tracker）也不该被 cv2 阻塞。
"""
from __future__ import annotations

import json
import os
import tempfile
from enum import IntEnum
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "PayloadMode", "encode_metadata", "encode_roi", "encode_full", "encode_clip",
    "payload_sizes", "mode_name",
]


class PayloadMode(IntEnum):
    METADATA = 0
    ROI = 1
    FULL = 2
    CLIP = 3


_MODE_NAMES = {m: m.name.lower() for m in PayloadMode}


def mode_name(mode: PayloadMode) -> str:
    return _MODE_NAMES[PayloadMode(int(mode))]


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    """统一成 uint8 HxWx3 BGR。灰度/RGBA 都转，避免 imencode 在不同输入上行为不一。"""
    import cv2

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame 为空，无法编码载荷")
    img = frame
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"不支持的帧形状：{img.shape}")
    return img


def _norm_bbox(bbox) -> tuple[float, float, float, float] | None:
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([x1, y1, x2, y2])):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2)))


def _crop_pixels(frame: np.ndarray, bbox, padding: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """归一化 bbox → 像素裁剪。非法/零面积框回退到整帧，不抛异常。"""
    h, w = frame.shape[:2]
    b = _norm_bbox(bbox)
    if b is None:
        return frame, (0, 0, w, h)
    x1, y1, x2, y2 = b
    if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
        return frame, (0, 0, w, h)

    pad = max(0.0, float(padding))
    x1, y1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
    x2, y2 = min(1.0, x2 + pad), min(1.0, y2 + pad)
    px1, py1 = int(np.floor(x1 * w)), int(np.floor(y1 * h))
    px2, py2 = int(np.ceil(x2 * w)), int(np.ceil(y2 * h))
    px1, py1 = max(0, min(w - 1, px1)), max(0, min(h - 1, py1))
    px2, py2 = max(px1 + 1, min(w, px2)), max(py1 + 1, min(h, py2))
    return frame[py1:py2, px1:px2], (px1, py1, px2, py2)


def _imencode(img: np.ndarray, quality: int) -> bytes:
    import cv2

    q = int(max(1, min(100, quality)))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def encode_metadata(event: Mapping[str, Any]) -> bytes:
    """事件元数据。紧凑 JSON，几十字节级；不含图像。"""
    payload = dict(event)
    # NaN/Inf 不是合法 JSON；事件快照本来就不该有，但外部传入时挡一下。
    for k, v in list(payload.items()):
        if isinstance(v, float) and not np.isfinite(v):
            payload[k] = None
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def encode_roi(frame: np.ndarray, bbox, max_side: int = 224,
               quality: int = 70, padding: float = 0.1,
               upscale: bool = True) -> bytes:
    """框裁剪 → JPEG。`upscale=True` 时小框也会放大到 `max_side`，方便远端姿态模型。

    如果 bbox 非法/零面积，回退为整帧缩放 —— 调用方不需要为坏框写分支。
    """
    import cv2

    img = _as_bgr(frame)
    crop, _ = _crop_pixels(img, bbox, padding)
    h, w = crop.shape[:2]
    scale = float(max_side) / float(max(h, w))
    if scale < 1.0 or (upscale and scale > 1.0):
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        crop = cv2.resize(crop, (nw, nh), interpolation=interp)
    return _imencode(crop, quality)


def encode_full(frame: np.ndarray, quality: int = 80) -> bytes:
    """整帧 JPEG。"""
    return _imencode(_as_bgr(frame), quality)


def encode_clip(frames: Iterable[np.ndarray], fps: float = 25.0,
                max_side: int | None = None) -> bytes:
    """短片段 MP4。用于最后手段 / 演示；需要当前 OpenCV 有 mp4v 编码器。

    实现走临时文件：OpenCV 的 VideoWriter 没有内存编码接口。文件在 finally 里删除。
    """
    import cv2

    frames = [_as_bgr(f) for f in frames]
    if not frames:
        return b""
    h, w = frames[0].shape[:2]
    if max_side is not None:
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale < 1.0:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            frames = [cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames]
            h, w = nh, nw

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError("当前 OpenCV 无法创建 mp4v 编码器；请用 ROI/FULL 替代")
        try:
            for f in frames:
                if f.shape[:2] != (h, w):
                    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(f)
        finally:
            writer.release()
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def payload_sizes(frame: np.ndarray, event: Mapping[str, Any],
                  bbox=None) -> dict[str, int]:
    """返回 METADATA / ROI / FULL 的实际字节数（CLIP 需要多帧，不在这里算）。"""
    ev = dict(event)
    if bbox is not None:
        ev["bbox"] = [float(v) for v in bbox]
    return {
        "metadata": len(encode_metadata(ev)),
        "roi": len(encode_roi(frame, ev.get("bbox"))),
        "full": len(encode_full(frame)),
    }
