"""T0.3 · 事件向量契约自检（不依赖 ultralytics/torch，纯 numpy）。

用法：python scripts/check_event_vector.py   看到 "事件向量契约 OK ✓" 即通过。
"""
import io
import sys
from pathlib import Path

import numpy as np

# Windows 控制台默认 GBK，打不出 ✓。用 isinstance 而非 hasattr：
# typeshed 把 sys.stdout 标注为 TextIO（无 reconfigure），isinstance 能让类型检查器收窄。
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common.event_vector import (  # noqa: E402
    EVENT_DIM, NONE_EVENT, EventType, EventVector,
    build_event, event_from_detections,
)


def check_roundtrip():
    ev = EventVector(EventType.SUSPECTED, 0.83, (0.1, 0.2, 0.5, 0.9), True, 0.66)
    arr = ev.to_array()
    assert arr.shape == (EVENT_DIM,) and arr.dtype == np.float32
    back = EventVector.from_array(arr)
    assert back.event_type == EventType.SUSPECTED
    assert abs(back.confidence - 0.83) < 1e-4
    assert back.has_event is True
    print("  [1] 序列化/反序列化往返一致 ✓")


def check_severity():
    # 直立瘦高框（h/w≈3）→ 异常度 0 → severity≈0 → PERSON
    stand = build_event((0.4, 0.1, 0.5, 0.7), 0.9)
    assert stand.event_type == EventType.PERSON and stand.severity < 0.1
    # 躺卧扁宽框（h/w≈0.4）→ 异常度 1 → severity≈conf → SUSPECTED
    lie = build_event((0.1, 0.5, 0.7, 0.7), 0.9)
    assert lie.event_type == EventType.SUSPECTED and lie.severity > 0.8
    # 低置信躺卧 → severity 被打折
    faint = build_event((0.1, 0.5, 0.7, 0.7), 0.2)
    assert faint.severity < 0.3
    print("  [2] severity=置信度×高宽比异常度，站立≈0 躺卧高 ✓")


def check_detections():
    assert event_from_detections(np.empty((0, 4)), [], []) is NONE_EVENT
    # 两人：一站一躺，应挑躺卧者（severity 更高）为主目标
    boxes = [[0.4, 0.1, 0.5, 0.7], [0.1, 0.5, 0.7, 0.7]]
    ev = event_from_detections(boxes, [0.9, 0.85], [0, 0])
    assert ev.event_type == EventType.SUSPECTED
    # 非 person 类被忽略
    assert event_from_detections([[0.1, 0.5, 0.7, 0.7]], [0.9], [5]) is NONE_EVENT
    # 低于阈值被过滤
    assert event_from_detections([[0.4, 0.1, 0.5, 0.7]], [0.1], [0]) is NONE_EVENT
    print("  [3] 多人挑主目标 + 类别/阈值过滤 ✓")


if __name__ == "__main__":
    check_roundtrip()
    check_severity()
    check_detections()
    print("事件向量契约 OK ✓")
